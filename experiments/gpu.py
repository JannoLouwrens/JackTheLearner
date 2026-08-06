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

# Colab VMs start in /content, and `colab download` will not resolve a relative
# remote path. Verified 2026-08-04.
COLAB_CWD = "/content"

# Kaggle assigns a Tesla P100 (sm_60) regardless of the accelerator requested —
# nvidiaTeslaT4 and gpuT4x2 both return P100 — while its preinstalled
# torch 2.10+cu128 ships kernels for sm_70 and above only. Every CUDA op then
# raises "no kernel image is available for execution on the device".
# cu121 wheels are the last line carrying Pascal, so jobs prepend this.
# Verified 2026-08-04: torch 2.5.1+cu121, arch_list sm_50..sm_90, matmul OK.
KAGGLE_TORCH_FIX = """
import subprocess as _sp, sys as _sys
_sp.run([_sys.executable, "-m", "pip", "install", "-q", "torch==2.5.1",
         "--index-url", "https://download.pytorch.org/whl/cu121"], check=False)
for _m in [m for m in list(_sys.modules) if m.startswith("torch")]:
    del _sys.modules[_m]
"""


REPO_URL = "https://github.com/JannoLouwrens/JackTheLearner"


def repo_preamble(ref: str = "main") -> str:
    """Put the repo on an ephemeral machine that has never seen it.

    Every training spec from Tier 1 onward needs UnifiedBrain, and shipping it
    inline is not viable — it is 4700 lines and it drifts. The repo is public, so
    the cheapest correct answer is to clone it and pin a ref, which also makes a
    GPU result attributable to an exact commit rather than to "whatever was on the
    box that day".

    Only numpy is needed beyond torch, and both backends ship it, so there is no
    dependency install to go wrong.
    """
    return f"""
import subprocess as _sp, sys as _sys, os as _os
_sp.run(["git", "clone", "--depth", "50", "-q", "{REPO_URL}", "/tmp/jack"], check=True)
_sp.run(["git", "-C", "/tmp/jack", "checkout", "-q", "{ref}"], check=True)
_sys.path.insert(0, "/tmp/jack")
_os.environ["MUJOCO_GL"] = "disabled"
# Backend-neutral output directory. Colab VMs start in /content; Kaggle kernels
# must write to /kaggle/working or the file is never collected. A job that
# hardcodes either one breaks the moment failover moves it (T0.11), so jobs write
# to JACK_OUT and never name a backend.
JACK_OUT = "/kaggle/working" if _os.path.isdir("/kaggle/working") else "/content"
_os.environ["JACK_OUT"] = JACK_OUT
print("JACK_OUT", JACK_OUT, flush=True)
print("REPO", _sp.run(["git", "-C", "/tmp/jack", "rev-parse", "--short", "HEAD"],
                      capture_output=True, text=True).stdout.strip(), flush=True)
"""


def assert_ref_is_current(ref: str = "main") -> None:
    """Refuse to ship a job whose code is not the code being tested.

    The VM clones from GitHub, so anything uncommitted or unpushed simply is not
    there. On 2026-08-05 that cost two GPU runs and produced a WRONG DIAGNOSIS:
    a fix to UnifiedBrain existed only in the working tree, the clone ran the
    published file, and the job died on both backends with a terse "exit 1" that
    looked like an infrastructure fault rather than stale code. Worse was the
    near-miss before it -- had the missing method been an OPTIONAL path instead of
    an AttributeError, the run would have SUCCEEDED and silently measured the old
    model, and the ladder would have recorded that number as evidence.

    A GPU result is only attributable to a commit if the commit is what ran. So
    this is checked, not documented.
    """
    def git(*a) -> tuple[int, str]:
        p = subprocess.run(["git", "-C", str(Path(__file__).parent.parent), *a],
                           capture_output=True, text=True)
        return p.returncode, p.stdout.strip()

    rc, dirty = git("status", "--porcelain", "--untracked-files=no")
    if rc == 0 and dirty:
        # OUTPUTS, not inputs. These are written BY a run and never read by the
        # remote job, so a modification to one says nothing about whether the
        # GPU's code matches ours. Including them deadlocked the guard against
        # itself: Budget.charge() writes gpu_budget.json at the end of every job,
        # so the first GPU run dirtied the tree and blocked the second. A guard
        # that fails on its own side effects trains people to bypass it.
        outputs = {"experiments/gpu_budget.json", "experiments/ledger.json",
                   "CHECKLIST.md", "docs/LOOP_JOURNAL.md"}
        # Parse by splitting, not by column: git() strips stdout, which eats the
        # leading space of the FIRST porcelain line only (' M path' -> 'M path'),
        # so a column-3 slice yielded 'periments/gpu_budget.json' for whichever
        # file happened to be listed first — and the exclusion silently missed
        # it. That is why this guard kept firing on its own budget file after
        # being "fixed": the fix was validated against subprocess output that
        # had not been stripped, i.e. against code that was not the code running.
        def _path(ln: str) -> str:
            parts = ln.strip().split(None, 1)
            return parts[1] if len(parts) == 2 else ln.strip()
        offending = [ln for ln in dirty.splitlines()
                     if _path(ln) not in outputs]
        if offending:
            raise RuntimeError(
                "Uncommitted changes to tracked files -- the GPU would run "
                "DIFFERENT code than you are testing:\n"
                + "\n".join(offending) + "\nCommit and push first."
            )

    git("fetch", "-q", "origin")
    rc, _ = git("merge-base", "--is-ancestor", "HEAD", f"origin/{ref}")
    if rc != 0:
        _, head = git("log", "--oneline", "-1")
        raise RuntimeError(
            f"HEAD ({head}) is not on origin/{ref}. The VM clones from GitHub, so "
            "unpushed work is invisible to it. Push before submitting."
        )


def build_job(body: str, ref: str = "main", verify: bool = True) -> Path:
    """Wrap a script body into a runnable job file: clone the repo, then run it.

    verify=True refuses to build unless HEAD is committed and pushed, because a
    result computed from stale code is worse than no result -- it looks like
    evidence. Pass verify=False only for jobs whose body is fully self-contained.
    """
    if verify:
        assert_ref_is_current(ref)
    f = Path(tempfile.mkdtemp(dir="/data")) / "job.py"
    f.write_text(repo_preamble(ref) + "\n" + body)
    return f


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
    fetch_errors: list[str] = []
    if keep:
        tmp = Path(tempfile.mkdtemp(dir="/data"))
        for remote in fetch or []:
            # Absolute paths only. `colab download` resolves nothing relative:
            # "marker.json" returns "File or directory not found" while
            # "/content/marker.json" succeeds. The VM's CWD is /content, verified.
            remote_abs = remote if remote.startswith("/") else f"{COLAB_CWD}/{remote}"
            local = tmp / Path(remote).name
            key = Path(remote).name
            drc, dout, derr = _run([COLAB, "--auth", "adc", "download",
                                    "-s", session, remote_abs, str(local)], 300)
            if drc == 0 and local.exists():
                # Keyed by BASENAME on both backends. Colab used to key by full
                # remote path and Kaggle by filename, so res.artifacts[...] found
                # nothing after a failover -- the job contract held right up until
                # the moment it mattered.
                artifacts[key] = str(local)
            else:
                # A silent download failure is indistinguishable from a job that
                # never wrote its artifact, and the two need opposite fixes. On
                # 2026-08-05 T1.08 returned "no artifact; stdout tail: " with an
                # EMPTY tail and no other information, which was unactionable.
                fetch_errors.append(
                    f"{remote_abs}: rc={drc} exists={local.exists()} "
                    f"{(derr or dout or '').strip()[:200]}"
                )
        # Always release the VM, even if the run failed — a kept session that is
        # never stopped holds a GPU and burns quota silently.
        _run([COLAB, "--auth", "adc", "stop", "-s", session], 120)

    gpu_name = ""
    for line in out.splitlines():
        if "Tesla" in line or "NVIDIA" in line:
            gpu_name = line.strip()[:80]
            break

    msg = "" if rc == 0 else f"exit {rc}"
    if fetch_errors:
        msg = (msg + " | " if msg else "") + "fetch failed: " + "; ".join(fetch_errors)
    # A run that produced no stdout at all did not execute the job body: the
    # preamble prints REPO <sha> before anything else. Say so rather than
    # reporting an empty tail.
    if rc == 0 and not out.strip():
        msg = (msg + " | " if msg else "") + \
              "remote produced NO stdout — the preamble prints 'REPO <sha>', so " \
              "the script body almost certainly never ran (clone or import failed)"
    return JobResult("colab", rc == 0, out, err, time.time() - t0,
                     artifacts, gpu_name, msg)


def run_on_kaggle(script: Path, timeout_s: int = 1800,
                  fetch: Optional[list[str]] = None) -> JobResult:
    """Push a kernel, poll to completion, retrieve output.

    Kaggle has no ephemeral-run primitive — a kernel is pushed, queued, and its
    output collected afterwards, so this polls rather than blocking on a process.
    """
    t0 = time.time()
    slug = f"jack-ladder-{int(t0)}"
    work = Path(tempfile.mkdtemp(dir="/data"))
    # Pascal-compatible torch first, then the job itself.
    (work / "kernel.py").write_text(KAGGLE_TORCH_FIX + "\n" + script.read_text())

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

    rc, out, err = _run([KAGGLE, "kernels", "push", "-p", str(work),
                         "--accelerator", "nvidiaTeslaT4"], 300)
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
