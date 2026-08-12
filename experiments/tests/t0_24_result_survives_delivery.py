"""T0.24 — once the kernel has computed the answer, nothing may drop it.

THE SCAR (2026-08-11 21:47 UTC). Kaggle kernel
`jannolouwrens/jack-ladder-1786482462` ran T1.02's three arms across three
seeds, finished `complete`, charged 0.6561 h, and printed every number this
project needed. The ledger recorded:

    ERROR: ValueError: dictionary update sequence element #0 has length 3;
           2 is required

Nothing was wrong with the science. The answer died in the last ten metres,
and three independent defects had to line up for it:

  1. `run_on_kaggle` returned `stdout=""`, always. Kaggle has no stdout pipe —
     the console arrives afterwards as a JSON array of
     `{stream_name, time, data}` records — and nobody parsed it. So every
     spec's "fall back to the printed RESULT line" branch was DEAD CODE on the
     one backend that runs the long jobs.
  2. That log file was handed back inside `artifacts`, indistinguishable from
     a result the job deliberately wrote.
  3. T1.02 asked for `artifacts["/content/out.json"]` — a remote path, while
     both backends key by BASENAME, so the lookup could never hit — and fell
     through to `next(iter(artifacts.values()))`, which took the log. Parsed,
     it is a list of 3-key dicts; fed to `dict.update` it raises the message
     above, which describes nothing that happened.

Any one of the three is survivable. Together they convert a correct, paid-for
measurement into a crash whose text points nowhere near the cause. The lesson
generalises past Kaggle: **a run's cost is committed the moment the provider
finishes, so every line after that point is an uninsured chance to throw the
answer away** — and unlike a failed run, this failure leaves the budget spent
and the ledger red.

Six properties, each able to fail alone:

  P0  `_kaggle_log_streams` on the REAL 2026-08-11 log fixture recovers the
      RESULT line and splits stderr out. This is the property that makes the
      recovery possible at all.
  P1  `_kaggle_collect` keeps `<slug>.log` OUT of artifacts, reports it as
      `log_path`, and routes it to stdout — while a genuine artifact beside it
      still comes through. Both halves matter: a collector that dropped
      everything would pass the first half.
  P2  `result_json` reads the named artifact when there is one.
  P3  `result_json` falls back to the `RESULT ` line, which is the branch that
      recovers this exact run.
  P4  `result_json` REFUSES a remote path (`/content/out.json`) instead of
      silently missing, and raises rather than guessing when a job delivered
      neither an artifact nor a RESULT line. A helper that returns something
      no matter what is the bug with better manners.
  P5  with `JACK_REUSE_KERNEL` set, `submit` never touches Colab. A reattach
      names one finished Kaggle kernel; walking the normal `prefer` order pays
      for a fresh job to recover a free one — and could return a different
      run's numbers.

THE CONTROL is the pre-fix delivery replayed verbatim: every downloaded file
becomes an artifact, then `next(iter(...))` picks one. Against the same real
log fixture it MUST still raise `dictionary update sequence element #0 has
length 3`. If the control ever passes, the fixture has stopped reproducing the
bug and this spec is guarding air.

No network, no submission, no GPU: `submit` is exercised against stub backends
and a temporary budget file.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .. import gpu
from ..gpu import Budget, JobResult, _kaggle_collect, _kaggle_log_streams, result_json
from ..protocol import Ledger, run_spec
from ..registry import BY_ID

SLUG = "jack-ladder-1786482462"

# The real log's shape, reproduced record for record: Kaggle interleaves stdout
# and stderr as timestamped records, and the payload line arrives near the end
# after the notebook-conversion noise.
PAYLOAD = {"device": "cuda", "seeds": {"0": {"structured": {"heldout": 0.02498}}}}
LOG_RECORDS = [
    {"stream_name": "stdout", "time": 11.27, "data": "REPO d0c8a6e\n"},
    {"stream_name": "stderr", "time": 12.01, "data": "SyntaxWarning: invalid escape\n"},
    {"stream_name": "stdout", "time": 2200.4, "data": "RESULT " + json.dumps(PAYLOAD) + "\n"},
    {"stream_name": "stdout", "time": 2201.0, "data": "[NbConvertApp] Writing 303629 bytes\n"},
]


def _fixture(tmp: Path) -> Path:
    """A downloaded kernel output directory: one log, one real artifact."""
    out = tmp / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{SLUG}.log").write_text(json.dumps(LOG_RECORDS))
    (out / "t201.json").write_text(json.dumps({"from": "artifact"}))
    return out


def _pre_fix_delivery(outdir: Path) -> dict:
    """THE CONTROL. The delivery path exactly as it stood on 2026-08-11.

    Every file is an artifact; the result is whichever one iteration yields
    first. Kept executable because a scar nobody can re-run is a story.
    """
    artifacts = {f.name: str(f) for f in sorted(outdir.rglob("*")) if f.is_file()}
    cache: dict = {}
    path = artifacts.get("/content/out.json") or next(iter(artifacts.values()), None)
    cache.update(json.loads(Path(path).read_text()))   # ValueError lives here
    return cache


def _experiment(seed: int) -> dict:
    failed: list[str] = []
    tmp = Path(tempfile.mkdtemp(dir="/data"))
    outdir = _fixture(tmp)

    # P0 — the log parses, and the answer is in it.
    s_out, s_err = _kaggle_log_streams(outdir / f"{SLUG}.log")
    result_line = [l for l in s_out.splitlines() if l.startswith("RESULT ")]
    if len(result_line) != 1 or json.loads(result_line[0][7:]) != PAYLOAD:
        failed.append("P0:RESULT line not recovered from the kaggle log")
    if "SyntaxWarning" in s_out or "SyntaxWarning" not in s_err:
        failed.append("P0:stderr not separated from stdout")

    # P1 — the log is evidence, not an artifact; the real artifact survives.
    arts, log_path, c_out, c_err = _kaggle_collect(outdir, SLUG)
    if f"{SLUG}.log" in arts:
        failed.append("P1:kernel log offered as an artifact")
    if log_path != str(outdir / f"{SLUG}.log"):
        failed.append("P1:log_path not reported")
    if arts.get("t201.json") != str(outdir / "t201.json"):
        failed.append("P1:genuine artifact lost by the collector")
    if c_out != s_out or c_err != s_err:
        failed.append("P1:collector did not route the log to stdout/stderr")

    # P2 — a named artifact is read.
    r_art = JobResult("kaggle", True, c_out, c_err, artifacts=arts)
    if result_json(r_art, "t201.json") != {"from": "artifact"}:
        failed.append("P2:named artifact not returned")

    # P3 — no artifact, RESULT line: the branch that recovers this very run.
    r_line = JobResult("kaggle", True, c_out, c_err, artifacts={})
    try:
        if result_json(r_line, "out.json") != PAYLOAD:
            failed.append("P3:RESULT line not used as the fallback")
    except Exception as e:
        failed.append(f"P3:fallback raised {type(e).__name__}: {e}")

    # P4 — refuse a remote path; refuse to guess when nothing was delivered.
    try:
        result_json(r_art, "/content/out.json")
        failed.append("P4:remote path accepted instead of refused")
    except ValueError:
        pass
    except Exception as e:
        failed.append(f"P4:remote path raised {type(e).__name__}, want ValueError")
    try:
        result_json(JobResult("kaggle", True, "no payload here", artifacts={}), "out.json")
        failed.append("P4:returned a result for a job that delivered none")
    except RuntimeError:
        pass
    except Exception as e:
        failed.append(f"P4:empty delivery raised {type(e).__name__}, want RuntimeError")

    # P5 — a reattach is kaggle-only. Stub both backends; neither runs anything.
    called: list[str] = []

    def _stub(backend, ok):
        def f(*a, **k):
            called.append(backend)
            return JobResult(backend, ok, "RESULT {}", job_id=f"u/{SLUG}", billable_s=0.0)
        return f

    real_colab, real_kaggle = gpu.run_on_colab, gpu.run_on_kaggle
    prev = os.environ.get("JACK_REUSE_KERNEL")
    try:
        gpu.run_on_colab = _stub("colab", True)
        gpu.run_on_kaggle = _stub("kaggle", True)
        os.environ["JACK_REUSE_KERNEL"] = f"jannolouwrens/{SLUG}"
        res = gpu.submit(tmp / "job.py", prefer="colab", est_hours=99.0,
                         budget=Budget(tmp / "budget.json"),
                         journal=tmp / "submissions.jsonl")
        if "colab" in called:
            failed.append("P5:reattach routed to colab — a free recovery would have paid")
        if called != ["kaggle"] or res.backend != "kaggle":
            failed.append(f"P5:reattach did not go straight to kaggle (called={called})")
    finally:
        gpu.run_on_colab, gpu.run_on_kaggle = real_colab, real_kaggle
        if prev is None:
            os.environ.pop("JACK_REUSE_KERNEL", None)
        else:
            os.environ["JACK_REUSE_KERNEL"] = prev

    return {"properties_failed": len(failed), "failures": failed,
            "properties_checked": 6}


def _control(seed: int) -> dict:
    """The 2026-08-11 delivery on the 2026-08-11 fixture. It must still break."""
    tmp = Path(tempfile.mkdtemp(dir="/data"))
    outdir = _fixture(tmp)
    # Only the log, as Kaggle actually returned it that night — the job wrote
    # /content/out.json, which Kaggle never collects.
    (outdir / "t201.json").unlink()
    try:
        _pre_fix_delivery(outdir)
    except ValueError as e:
        return {"pre_fix_raised": "ValueError", "pre_fix_message": str(e)}
    except Exception as e:
        return {"pre_fix_raised": type(e).__name__, "pre_fix_message": str(e)}
    return {"pre_fix_raised": "", "pre_fix_message": "the old path SUCCEEDED"}


def _check(m: dict, c: dict) -> bool:
    m["control_reproduces_scar"] = (
        c.get("pre_fix_raised") == "ValueError"
        and "dictionary update sequence element #0 has length 3" in c.get("pre_fix_message", ""))
    return m["properties_failed"] == 0 and m["control_reproduces_scar"]


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.24"], _experiment, _check, control_fn=_control, ledger=ledger)
