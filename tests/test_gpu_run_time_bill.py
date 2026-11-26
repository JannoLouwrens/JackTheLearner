"""S15 (3): a Kaggle kernel is billed its RUN time, not its queue time.

`…268291` was billed 10.02 h for a kernel whose own log spans 5.09 h.
1. poll: QUEUED x3 then RUNNING x2 then COMPLETE -> the meter opens at the
   first RUNNING (60 s of queue unbilled), not at push
2. poll: first answer RUNNING (a reattach) -> the meter stays at t_open
3. refund(keep_hours=) lowers a bill to the measured hours in its own week,
   once; a keep above the bill is refused
Control: the same stub with a start the poll cannot read bills all 120 s.

Run: python -m tests.test_gpu_run_time_bill  (CLI and clock stubbed; no network)
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import gpu  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


def poll(answers):
    clock, seq = Clock(), iter(answers)
    real_time, real_run = gpu.time, gpu._run
    gpu.time = clock
    gpu._run = lambda cmd, timeout: (0, f'has status "KernelWorkerStatus.{next(seq)}"', "")
    try:
        return gpu._kaggle_poll("u/k", 10_000, 1000.0)
    finally:
        gpu.time, gpu._run = real_time, real_run


def main():
    bad = 0
    st, bill, _ = poll(["QUEUED"] * 3 + ["RUNNING"] * 2 + ["COMPLETE"])
    # polls at 1020..1120; first RUNNING at 1080 -> opens 1060; complete 1120
    ok = st == "complete" and bill == 60.0
    print(("PASS" if ok else "FAIL"), "queued kernel billed from first RUNNING:", st, bill)
    bad += not ok
    st, bill, _ = poll(["RUNNING"] * 2 + ["COMPLETE"])
    ok = bill == 60.0
    print(("PASS" if ok else "FAIL"), "first answer RUNNING keeps t_open:", bill)
    bad += not ok
    # control: a start the poll cannot see is the pre-fix push-time meter
    st, bill, _ = poll(["QUEUED"] * 3 + ["STARTED"] * 2 + ["COMPLETE"])
    ok = bill == 120.0
    print(("PASS" if ok else "FAIL"), "control: unseen start bills the queue:", bill)
    bad += not ok

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "b.json"
        p.write_text(json.dumps({"weeks": {"W": {"kaggle": 33.1461}}, "charged_jobs": {
            "j": {"week": "W", "backend": "kaggle", "hours": 10.0189, "ok": True}}}))
        b = gpu.Budget(p)
        r1 = b.refund("j", "run time", keep_hours=5.0862)
        r2 = b.refund("j", "again", keep_hours=1.0)
        r3 = gpu.Budget(p).refund("j", "too much", keep_hours=99.0)
        data = json.loads(p.read_text())
        rec, wk = data["charged_jobs"]["j"], data["weeks"]["W"]["kaggle"]
        ok = (r1 and not r2 and not r3 and rec["hours"] == 5.0862
              and rec["refunded_hours"] == 4.9327 and wk == 28.2134)
        print(("PASS" if ok else "FAIL"), "refund to run time:", rec, wk)
        bad += not ok
    print("RESULT", json.dumps({"failed": bad}))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
