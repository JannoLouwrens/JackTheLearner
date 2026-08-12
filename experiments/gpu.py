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
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .protocol import is_code_dirt

COLAB = "/data/venvs/colab/bin/colab"
KAGGLE = "/data/venvs/kaggle/bin/kaggle"
BUDGET_FILE = Path(__file__).parent / "gpu_budget.json"

# Append-only receipt for every remote dispatch. See `_record_submission`.
SUBMISSION_LOG = Path(__file__).parent / "gpu_submissions.jsonl"

# Kaggle's free allowance. Colab's is unpublished and elastic, so it is not
# budgeted — it is simply tried first for short work.
KAGGLE_WEEKLY_HOURS = 30.0

# Hours a provider billed for a run that returned nothing get their own bucket
# per week: `{backend}` is work, `{backend}_failed` is waste. Both count against
# the quota; only one of them bought anything.
FAILED_SUFFIX = "_failed"

# How many job ids the budget remembers for idempotency. Only a reattach needs
# to find its own id, and that happens within hours of the original push.
MAX_TRACKED_JOBS = 500

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
import subprocess as _sp, sys as _sys, os as _os
_sp.run([_sys.executable, "-m", "pip", "install", "-q", "torch==2.5.1",
         "--index-url", "https://download.pytorch.org/whl/cu121"], check=False)
# Pin torch for every later pip install in this job. On 2026-08-09 T2.02's own
# dependency install (stable-baselines3, whose torch range 2.5.1 satisfies)
# nevertheless dragged torch up to 2.13.0+cu130 — no sm_60 kernels — and the
# P100 died in .to(device) six minutes in. PEP 440 says ==2.5.1 is satisfied
# by the installed 2.5.1+cu121, so constrained installs leave it alone, and a
# dependency that genuinely cannot live with 2.5.1 now fails resolution loudly
# instead of silently un-fixing Pascal.
open("/tmp/jack_torch_pin.txt", "w").write("torch==2.5.1\\n")
_os.environ["PIP_CONSTRAINT"] = "/tmp/jack_torch_pin.txt"
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


def offending_dirt(porcelain_lines) -> list:
    """Which uncommitted lines mean the GPU would run DIFFERENT code than we test.

    OUTPUTS, not inputs, are excluded. These are written BY a run and never read
    by the remote job, so a modification to one says nothing about whether the
    GPU's code matches ours. Including them deadlocked the guard against itself:
    `Budget.charge()` writes `gpu_budget.json` at the end of every job, so the
    first GPU run dirtied the tree and blocked the second. A guard that fails on
    its own side effects trains people to bypass it.

    The exclusion is `protocol.is_code_dirt` — NOT a second list that happens to
    agree. On 2026-08-12 the two lists disagreed by exactly one entry each way:
    this file knew `gpu_budget.json` was an output and `protocol.py` did not,
    while `protocol.py`'s stamp called a `LOOP_JOURNAL.md` edit uncommitted code
    and blocked 47 specs on it. Both organs answer ONE question — does this
    uncommitted file mean the code moved — so there is now one predicate and
    zero permitted difference. T0.22 P15 pins them together.

    A function over a fixture list, not an inline filter over the real tree, for
    the reason `is_code_dirt` was extracted: a predicate exercisable only by
    dirtying the repo it audits is a predicate nothing will ever test.
    """
    return [ln for ln in porcelain_lines if is_code_dirt(ln)]


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
        offending = offending_dirt(dirty.splitlines())
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
    # Identity of the remote unit of work: the Kaggle kernel `user/slug`, the
    # Colab session. Two JobResults with the same id are the SAME compute, so
    # the budget may bill it once — that is what makes a reattach free.
    job_id: str = ""
    # Where the backend's own console log landed locally, when it is a file
    # rather than a pipe. Kaggle returns one; it is EVIDENCE, not an artifact,
    # and conflating the two cost T1.02 a completed run (see `result_json`).
    log_path: str = ""
    # The window the provider actually meters, when it is narrower than this
    # box's wall clock. `duration_s` includes pushing the kernel, waiting in
    # Kaggle's queue and downloading artifacts; none of that is GPU time, and
    # billing it is how a 30 h ceiling closed a week at 37.4554 h. None means
    # "no narrower window is known" and the full duration is charged.
    billable_s: Optional[float] = None

    @property
    def charge_seconds(self) -> float:
        return self.duration_s if self.billable_s is None else self.billable_s


class Budget:
    """Weekly GPU-hour accounting, so a run cannot quietly exhaust the quota.

    Deliberately a plain JSON file: the value is not the mechanism, it is that a
    number exists somewhere a person can read before wondering why Kaggle stopped
    granting GPUs mid-week.
    """

    def __init__(self, path: Path = BUDGET_FILE):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}
        # Lazily migrate older files rather than hand-editing the accounting
        # record: a budget written before 2026-08-09 has weeks and nothing else.
        self.data.setdefault("weeks", {})
        self.data.setdefault("charged_jobs", {})
        self.data.setdefault("overruns", [])

    @staticmethod
    def _week() -> str:
        # %U weeks start SUNDAY, matching Kaggle's actual quota reset. The
        # original %G-W%V (ISO, Monday-start) kept charging Sunday's runs to
        # the exhausted week, so the tracker refused jobs for the entire first
        # day of every fresh Kaggle quota. Usage recorded under the old keys
        # was migrated on 2026-08-08 — migrated, not copied: the two formats
        # share a namespace ("2026-W32" means Aug 3-9 in ISO but Aug 9-15 in
        # %U), so a leftover ISO entry silently blocks the %U week it collides
        # with. Old-format keys must be REMOVED once their hours are re-filed.
        return time.strftime("%Y-W%U")

    def productive_hours(self, backend: str) -> float:
        """Hours that bought a result."""
        return float(self.data["weeks"].get(self._week(), {}).get(backend, 0.0))

    def failed_hours(self, backend: str) -> float:
        """Hours the provider still billed for a run that returned nothing.

        Kept in a separate bucket because it is WASTE, and waste that cannot be
        seen cannot be reduced. It counts against the quota all the same — a
        crashed kernel occupied a real GPU.
        """
        key = backend + FAILED_SUFFIX
        return float(self.data["weeks"].get(self._week(), {}).get(key, 0.0))

    def used_hours(self, backend: str) -> float:
        return self.productive_hours(backend) + self.failed_hours(backend)

    def remaining(self, backend: str) -> float:
        if backend != "kaggle":
            return float("inf")
        return max(0.0, KAGGLE_WEEKLY_HOURS - self.used_hours("kaggle"))

    def overruns(self) -> list:
        return list(self.data.get("overruns", []))

    def charge(self, backend: str, seconds: float, *,
               ok: bool = True, job_id: str = "") -> bool:
        """Bill `seconds` of `backend`. Returns True if this call actually billed.

        `job_id` makes billing idempotent per unit of remote compute. Without it,
        `JACK_REUSE_KERNEL` — which reattaches to a kernel that is already
        running and deliberately skips `afford()` because reattaching is free —
        billed the whole kernel a second time on every recovery.

        `ok=False` files the hours as waste. Before 2026-08-09 this call sat
        above `if res.ok`, so a kernel that crashed, timed out or lost its
        artifact download was indistinguishable in the record from work.
        """
        if job_id and job_id in self.data["charged_jobs"]:
            return False
        key = backend if ok else backend + FAILED_SUFFIX
        wk = self.data["weeks"].setdefault(self._week(), {})
        wk[key] = round(wk.get(key, 0.0) + seconds / 3600.0, 4)
        if job_id:
            self.data["charged_jobs"][job_id] = {
                "week": self._week(), "backend": backend,
                "hours": round(seconds / 3600.0, 4), "ok": ok,
            }
            # Unbounded growth would eventually make the file the largest thing
            # in the repo. Insertion order is the age order.
            while len(self.data["charged_jobs"]) > MAX_TRACKED_JOBS:
                self.data["charged_jobs"].pop(next(iter(self.data["charged_jobs"])))
        # `afford()` gates on the DECLARED estimate and this bills the ACTUAL
        # elapsed time, so nothing prevents an overrun — but an overrun that
        # leaves no mark is how week 31 closed at 37.4554 of a 30.0 h ceiling
        # with T0.12 green throughout, and denied T1.02 its 0.7 h.
        used = self.used_hours(backend)
        if backend == "kaggle" and used > KAGGLE_WEEKLY_HOURS:
            self.data["overruns"].append({
                "week": self._week(), "backend": backend,
                "used_hours": round(used, 4), "ceiling": KAGGLE_WEEKLY_HOURS,
                "job_id": job_id, "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            print(f"!! GPU BUDGET OVERRUN: {backend} {used:.4f}h of "
                  f"{KAGGLE_WEEKLY_HOURS}h this week ({self._week()}) — "
                  f"the ceiling is not being enforced, only observed",
                  file=sys.stderr, flush=True)
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")
        return True

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
                         message=f"timed out after {timeout_s}s", job_id=session)
    # Colab keeps the VM alive across the artifact download (`--keep`, released
    # only by `stop` below), so wall clock here IS the held window: `billable_s`
    # stays None and `charge_seconds` falls back to `duration_s`.

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
                     artifacts, gpu_name, msg, job_id=session)


def _kaggle_log_streams(path: Path) -> tuple[str, str]:
    """Split a Kaggle kernel log into (stdout, stderr).

    Kaggle has no stdout pipe — the console arrives afterwards as a JSON array
    of `{stream_name, time, data}` records. Until 2026-08-12 nothing parsed it,
    so `JobResult.stdout` was ALWAYS empty on Kaggle and every spec's
    "fall back to the printed RESULT line" branch was dead code on the one
    backend that runs the long jobs. Unparseable is not fatal: the raw text is
    still more useful than nothing, so it is returned as stdout.
    """
    try:
        recs = json.loads(path.read_text())
        if not isinstance(recs, list):
            raise ValueError("not a record array")
    except Exception:
        return path.read_text(errors="replace"), ""
    out, err = [], ""
    errs = []
    for r in recs:
        if not isinstance(r, dict):
            continue
        (errs if r.get("stream_name") == "stderr" else out).append(str(r.get("data", "")))
    return "".join(out), "".join(errs)


def _kaggle_collect(outdir: Path, slug: str) -> tuple[dict, str, str, str]:
    """Sort what `kernels output` downloaded into (artifacts, log, stdout, stderr).

    `kernels output` ships the console log ALONGSIDE the artifacts, named after
    the kernel. It is not an artifact. On 2026-08-11 T1.02 took it as one,
    `json.loads`'d the log's own array-of-records straight into `dict.update`,
    and died with "dictionary update sequence element #0 has length 3" — AFTER
    0.66 paid GPU-hours had already produced the right answer, which was sitting
    in that very log on the `RESULT` line. Separated here, once, rather than in
    each spec: the log becomes stdout, where every spec already knows to look.
    """
    artifacts, log_path, out, err = {}, "", "", ""
    for f in sorted(outdir.rglob("*")):
        if not f.is_file():
            continue
        if f.name == f"{slug}.log":
            log_path = str(f)
            out, err = _kaggle_log_streams(f)
            continue
        artifacts[f.name] = str(f)
    return artifacts, log_path, out, err


def result_json(res: "JobResult", name: str) -> dict:
    """The one sanctioned way to read a GPU job's JSON result.

    Artifact named `name` first, then the `RESULT {...}` line the job printed.
    Both are checked; a job that delivered NEITHER raises, and the message says
    which of the two was tried, because "no result" and "the wrong file" need
    opposite fixes.

    THE SCAR (2026-08-11). T1.02 hand-rolled this as
    `artifacts.get("/content/out.json") or next(iter(artifacts.values()))` —
    two defects in one line. Artifacts are keyed by BASENAME on both backends,
    so the first lookup could never hit; and the `next(iter(...))` fallback
    accepts *any* file the backend happened to return. It got the console log,
    and a completed 0.66 GPU-hour run became a ValueError. A blind pick is not
    a fallback: it is a guess about which file is the answer.
    """
    key = Path(name).name
    if key != name:
        raise ValueError(
            f"result_json takes a BASENAME; {name!r} is a remote path. "
            f"Both backends key artifacts by basename — pass {key!r}.")
    path = res.artifacts.get(key)
    if path:
        return json.loads(Path(path).read_text())
    for line in (res.stdout or "").splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise RuntimeError(
        f"job produced no result: no artifact named {key!r} "
        f"(got {sorted(res.artifacts)}) and no 'RESULT ' line in "
        f"{len(res.stdout or '')} chars of stdout")


def run_on_kaggle(script: Path, timeout_s: int = 1800,
                  fetch: Optional[list[str]] = None) -> JobResult:
    """Push a kernel, poll to completion, retrieve output.

    Kaggle has no ephemeral-run primitive — a kernel is pushed, queued, and its
    output collected afterwards, so this polls rather than blocking on a process.
    """
    t0 = time.time()
    work = Path(tempfile.mkdtemp(dir="/data"))

    # `kaggle config view` prints text, not JSON; read the username from it.
    cfg = subprocess.run([KAGGLE, "config", "view"], capture_output=True, text=True).stdout
    username = next((l.split(":", 1)[1].strip() for l in cfg.splitlines()
                     if l.strip().startswith("- username")), None)
    if not username:
        return JobResult("kaggle", False, message="could not determine Kaggle username")

    # REATTACH, not resubmit. If the local runner dies while a kernel runs
    # (session restart SIGPIPEd T2.01 v3's waiter at ~80 min in), the kernel
    # keeps computing and its artifact survives on Kaggle — the only thing lost
    # is the process that was waiting. JACK_REUSE_KERNEL=<slug or user/slug>
    # skips the push and polls/fetches that kernel instead, so recovering a
    # finished run costs zero GPU quota. One-shot by design: the env var names
    # ONE kernel, so only the next kaggle submission reuses it.
    reuse = os.environ.get("JACK_REUSE_KERNEL", "").strip()
    if reuse:
        slug = reuse.split("/", 1)[1] if "/" in reuse else reuse
        # The slug embeds the original submission epoch. Charge the budget from
        # THERE: Kaggle bills the kernel's full wall time whether or not anyone
        # local was watching, and a budget file that only counts the reattach
        # window would drift optimistic exactly when runs are long.
        try:
            t0 = float(slug.rsplit("-", 1)[-1])
        except ValueError:
            pass
    else:
        slug = f"jack-ladder-{int(t0)}"
    # Pascal-compatible torch first, then the job itself.
    (work / "kernel.py").write_text(KAGGLE_TORCH_FIX + "\n" + script.read_text())

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

    job_id = f"{username}/{slug}"

    if not reuse:
        rc, out, err = _run([KAGGLE, "kernels", "push", "-p", str(work),
                             "--accelerator", "nvidiaTeslaT4"], 300)
        if rc != 0:
            # A push that never landed ran nothing. Bill zero, not the 300 s the
            # CLI may have spent failing.
            return JobResult("kaggle", False, out, err, time.time() - t0,
                             message=f"push failed: {err.strip()[:200]}",
                             job_id=job_id, billable_s=0.0)

    # The metered window opens when the kernel exists on Kaggle and closes when
    # it reaches a terminal status. Pushing and downloading the output happen on
    # THIS box and are not GPU time; charging them is part of why `used_hours`
    # stopped being an account of hours consumed. On a reattach the local `t0`
    # was already rewound to the slug's submission epoch above, because the
    # kernel ran whether or not anyone here was watching.
    t_meter_open = t0 if reuse else time.time()

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
    billable_s = time.time() - t_meter_open

    artifacts, log_path, k_out, k_err = {}, "", "", ""
    if status == "complete":
        outdir = work / "out"
        outdir.mkdir(exist_ok=True)
        _run([KAGGLE, "kernels", "output", f"{username}/{slug}", "-p", str(outdir)], 300)
        artifacts, log_path, k_out, k_err = _kaggle_collect(outdir, slug)

    return JobResult("kaggle", status == "complete", k_out, k_err, time.time() - t0,
                     artifacts, "", "" if status == "complete" else f"status={status}",
                     job_id=job_id, log_path=log_path, billable_s=billable_s)


def _head_sha() -> str:
    p = subprocess.run(["git", "-C", str(Path(__file__).parent.parent),
                        "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else ""


def _record_submission(entry: dict, log: Optional[Path] = None) -> dict:
    """Append one line to the submission receipt log, durably.

    THE SCAR (2026-08-11, 7th overseer audit). Commit `6b001e7` handed off a
    claim that a T1.02 GPU poll was in flight. Nothing had been submitted. That
    false claim passed every gate the project owns: `gpu_budget.json` was
    unchanged, which reads as "nothing spent"; the ledger was unchanged, which
    reads as "not run yet"; and the only thing contradicting it was prose that
    no gate reads. An iteration that never called `submit()` and one whose
    submission died in flight left byte-identical evidence.

    So a dispatch now leaves a trace that is written BEFORE the remote call and
    survives the process dying mid-flight — a claim of "I submitted X" becomes
    checkable rather than believable. The log is append-only and never read by
    the ladder's decisions; it is evidence, not state.

    fsync'd because the whole point is surviving a kill: a line sitting in the
    page cache of a process that gets SIGKILLed is exactly the absent receipt
    this exists to prevent.
    """
    path = Path(log) if log is not None else SUBMISSION_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return entry


def submissions(log: Optional[Path] = None) -> list[dict]:
    """Every recorded dispatch, oldest first. Unparseable lines are skipped."""
    path = Path(log) if log is not None else SUBMISSION_LOG
    if not path.exists():
        return []
    out = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def last_submission(log: Optional[Path] = None) -> Optional[dict]:
    recs = submissions(log)
    return recs[-1] if recs else None


def submit(script: Path, prefer: str = "colab", est_hours: float = 0.1,
           gpu: str = "T4", timeout_s: int = 900,
           fetch: Optional[list[str]] = None,
           budget: Optional["Budget"] = None,
           journal: Optional[Path] = None) -> JobResult:
    """Run a job on whichever backend can take it. The job does not know which.

    Order: try `prefer`, fall back to the other. Kaggle is checked against its
    weekly budget first — the 30 free hours are the scarce resource, so short
    jobs belong on Colab and Kaggle is spent on work that needs the session length.

    `budget` exists so a test can exercise this routing without writing to the
    real accounting file. A function that hard-codes the path to the record it
    mutates cannot be tested except by corrupting it.
    """
    budget = budget or Budget()
    order = ["colab", "kaggle"] if prefer == "colab" else ["kaggle", "colab"]
    attempts = []
    # Reattaching to an already-finished kernel consumes no quota, so it must
    # not be blocked by the affordability gate — that exact gate turned a
    # zero-cost recovery of T2.01 v4 into a Colab failover and an ERROR.
    reuse = bool(os.environ.get("JACK_REUSE_KERNEL", "").strip())
    if reuse:
        # A reattach names ONE Kaggle kernel. Walking the normal order would run
        # `prefer="colab"` first — paying a full fresh job to recover a finished
        # free one, and returning a DIFFERENT run's numbers if it succeeded.
        # Colab burned 0.99 h on exactly this shape of waste on 2026-08-11.
        order = ["kaggle"]

    head = _head_sha()
    for backend in order:
        if backend == "kaggle" and not reuse and not budget.afford("kaggle", est_hours):
            attempts.append(f"kaggle: {budget.remaining('kaggle'):.1f}h left, need {est_hours}h")
            continue
        # BEFORE the call, so a job killed in flight still leaves evidence it
        # existed. `attempt_id` is what links this to its outcome line.
        started = time.time()
        attempt_id = f"{int(started * 1000)}-{os.getpid()}-{backend}"
        _record_submission({"phase": "attempt", "attempt_id": attempt_id,
                            "ts": started, "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "backend": backend, "prefer": prefer,
                            "est_hours": est_hours, "timeout_s": timeout_s,
                            "script": str(script), "head": head,
                            "pid": os.getpid()}, journal)
        res = (run_on_colab(script, gpu, timeout_s, fetch) if backend == "colab"
               else run_on_kaggle(script, timeout_s, fetch))
        _record_submission({"phase": "result", "attempt_id": attempt_id,
                            "ts": time.time(), "backend": backend,
                            "job_id": res.job_id, "ok": bool(res.ok),
                            "duration_s": res.duration_s,
                            "charge_seconds": res.charge_seconds,
                            "message": (res.message or "")[:500]}, journal)
        # Charge the metered window, labelled by outcome, once per remote job.
        # All three of those qualifiers were missing until 2026-08-09.
        budget.charge(backend, res.charge_seconds, ok=res.ok, job_id=res.job_id)
        if res.ok:
            res.message = (res.message + f" | attempts: {attempts}") if attempts else res.message
            return res
        attempts.append(f"{backend}: {res.message}")

    return JobResult(order[-1], False, message="all backends failed: " + "; ".join(attempts))
