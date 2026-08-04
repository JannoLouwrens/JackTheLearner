"""One job contract, two executors.

Tier 2 onward needs a GPU, and this box has none. What it has is two EPHEMERAL
free tiers — Colab (T4, VM torn down after the script) and Kaggle (30 h/week,
P100 or 2xT4, 12 h session cap). Neither offers a persistent process, so the unit
of work is: ship a self-contained script, run it, retrieve artifacts, forget the
machine.

The contract exists so a job never has to know which backend it landed on. That
is what makes failover (T0.11) a routing decision rather than a rewrite, and it
is why the 30 free Kaggle hours can be spent deliberately — short jobs go to
Colab, long ones to Kaggle, and neither choice touches the job.

Credentials live on this box and only this box: Colab's ADC at ~/.config/gcloud,
Kaggle's token at ~/.kaggle/access_token. That is precisely why the ladder loop
runs here rather than in a cloud runner.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

COLAB = "/data/venvs/colab/bin/colab"
KAGGLE = "/data/venvs/kaggle/bin/kaggle"
BUDGET_FILE = Path(__file__).parent / "gpu_budget.json"

# Kaggle's free allowance. Colab's is unpublished and elastic, so it is not
# budgeted — it is simply tried first for short work.
KAGGLE_WEEKLY_HOURS = 30.0


@dataclass
class JobResult:
    backend: str
    ok: bool
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    artifacts: dict = field(default_factory=dict)   # name -> local path
    gpu_name: str = ""
    message: str = ""


class Budget:
    """Weekly GPU-hour accounting, so a run cannot quietly exhaust the quota.

    Deliberately a plain JSON file: the value is not the mechanism, it is that a
    number exists somewhere a person can read before wondering why Kaggle stopped
    granting GPUs mid-week.
    """

    def __init__(self, path: Path = BUDGET_FILE):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {"weeks": {}}

    @staticmethod
    def _week() -> str:
        return time.strftime("%G-W%V")

    def used_hours(self, backend: str) -> float:
        return float(self.data["weeks"].get(self._week(), {}).get(backend, 0.0))

    def remaining(self, backend: str) -> float:
        if backend != "kaggle":
            return float("inf")
        return max(0.0, KAGGLE_WEEKLY_HOURS - self.used_hours("kaggle"))

    def charge(self, backend: str, seconds: float) -> None:
        wk = self.data["weeks"].setdefault(self._week(), {})
        wk[backend] = round(wk.get(backend, 0.0) + seconds / 3600.0, 4)
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")

    def afford(self, backend: str, est_hours: float) -> bool:
        return self.remaining(backend) >= est_hours


def _run(cmd: list[str], timeout: int) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def run_on_colab(script: Path, gpu: str = "T4", timeout_s: int = 900,
                 fetch: Optional[list[str]] = None) -> JobResult:
    """Provision a fresh Colab VM, run the script, retrieve files, release the VM."""
    t0 = time.time()
    session = f"ladder-{int(t0)}"
    keep = bool(fetch)
    cmd = [COLAB, "--auth", "adc", "run", "--gpu", gpu,
           "--timeout", str(timeout_s - 60), "-s", session]
    if keep:
        cmd.append("--keep")
    cmd.append(str(script))

    try:
        rc, out, err = _run(cmd, timeout_s)
    except subprocess.TimeoutExpired:
        return JobResult("colab", False, duration_s=time.time() - t0,
                         message=f"timed out after {timeout_s}s")

    artifacts = {}
    if keep and rc == 0:
        tmp = Path(tempfile.mkdtemp(dir="/data"))
        for remote in fetch or []:
            local = tmp / Path(remote).name
            drc, _, derr = _run([COLAB, "--auth", "adc", "download",
                                 "-s", session, remote, str(local)], 300)
            if drc == 0 and local.exists():
                artifacts[remote] = str(local)
        _run([COLAB, "--auth", "adc", "stop", "-s", session], 120)

    gpu_name = ""
    for line in out.splitlines():
        if "Tesla" in line or "NVIDIA" in line:
            gpu_name = line.strip()[:80]
            break

    return JobResult("colab", rc == 0, out, err, time.time() - t0,
                     artifacts, gpu_name,
                     "" if rc == 0 else f"exit {rc}")


def run_on_kaggle(script: Path, timeout_s: int = 1800,
                  fetch: Optional[list[str]] = None) -> JobResult:
    """Push a kernel, poll to completion, retrieve output.

    Kaggle has no ephemeral-run primitive — a kernel is pushed, queued, and its
    output collected afterwards, so this polls rather than blocking on a process.
    """
    t0 = time.time()
    slug = f"jack-ladder-{int(t0)}"
    work = Path(tempfile.mkdtemp(dir="/data"))
    (work / "kernel.py").write_text(script.read_text())

    # `kaggle config view` prints text, not JSON; read the username from it.
    cfg = subprocess.run([KAGGLE, "config", "view"], capture_output=True, text=True).stdout
    username = next((l.split(":", 1)[1].strip() for l in cfg.splitlines()
                     if l.strip().startswith("- username")), None)
    if not username:
        return JobResult("kaggle", False, message="could not determine Kaggle username")

    (work / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{username}/{slug}",
        "title": slug,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
    }, indent=2))

    rc, out, err = _run([KAGGLE, "kernels", "push", "-p", str(work)], 300)
    if rc != 0:
        return JobResult("kaggle", False, out, err, time.time() - t0,
                         message=f"push failed: {err.strip()[:200]}")

    deadline = time.time() + timeout_s
    status = "queued"
    while time.time() < deadline:
        time.sleep(20)
        _, s_out, _ = _run([KAGGLE, "kernels", "status", f"{username}/{slug}"], 120)
        low = s_out.lower()
        if "complete" in low:
            status = "complete"; break
        if "error" in low or "cancel" in low:
            status = "error"; break

    artifacts = {}
    if status == "complete":
        outdir = work / "out"
        outdir.mkdir(exist_ok=True)
        _run([KAGGLE, "kernels", "output", f"{username}/{slug}", "-p", str(outdir)], 300)
        for f in outdir.rglob("*"):
            if f.is_file():
                artifacts[f.name] = str(f)

    return JobResult("kaggle", status == "complete", "", "", time.time() - t0,
                     artifacts, "", "" if status == "complete" else f"status={status}")


def submit(script: Path, prefer: str = "colab", est_hours: float = 0.1,
           gpu: str = "T4", timeout_s: int = 900,
           fetch: Optional[list[str]] = None) -> JobResult:
    """Run a job on whichever backend can take it. The job does not know which.

    Order: try `prefer`, fall back to the other. Kaggle is checked against its
    weekly budget first — the 30 free hours are the scarce resource, so short
    jobs belong on Colab and Kaggle is spent on work that needs the session length.
    """
    budget = Budget()
    order = ["colab", "kaggle"] if prefer == "colab" else ["kaggle", "colab"]
    attempts = []

    for backend in order:
        if backend == "kaggle" and not budget.afford("kaggle", est_hours):
            attempts.append(f"kaggle: {budget.remaining('kaggle'):.1f}h left, need {est_hours}h")
            continue
        res = (run_on_colab(script, gpu, timeout_s, fetch) if backend == "colab"
               else run_on_kaggle(script, timeout_s, fetch))
        budget.charge(backend, res.duration_s)
        if res.ok:
            res.message = (res.message + f" | attempts: {attempts}") if attempts else res.message
            return res
        attempts.append(f"{backend}: {res.message}")

    return JobResult(order[-1], False, message="all backends failed: " + "; ".join(attempts))
