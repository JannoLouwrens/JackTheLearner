"""Next-up (v2), ruling 124: Kaggle's quota refusal marks that week FULL in the
budget file (`closed_by: kaggle`), so no later job is planned into it — even
when our own meter still reads hours left (the account is shared).
Run: python -m tests.test_gpu_week_full"""
import sys
import tempfile
from pathlib import Path

from experiments import gpu as G

QUOTA = "push failed: Kernel push error: Maximum weekly GPU quota of 30.00 hours reached."


def _submit(tmp: Path, message: str):
    """Drive the shipped `submit` with a stub Kaggle; return (budget, pushes)."""
    pushes = []

    def fake_kaggle(script, timeout_s, fetch, journal, **kw):
        pushes.append(1)
        return G.JobResult("kaggle", False, message=message, job_id=f"j{len(pushes)}",
                           billable_s=0.0)
    saved = (G.run_on_kaggle, G.STDOUT_DIR)
    G.run_on_kaggle, G.STDOUT_DIR = fake_kaggle, tmp / "out"
    try:
        b = G.Budget(tmp / "budget.json")
        G.submit(tmp / "job.py", prefer="kaggle", est_hours=1.0, budget=b,
                 journal=tmp / "receipts.jsonl")
        G.submit(tmp / "job.py", prefer="kaggle", est_hours=1.0, budget=b,
                 journal=tmp / "receipts.jsonl")
    finally:
        G.run_on_kaggle, G.STDOUT_DIR = saved
    return G.Budget(tmp / "budget.json"), pushes


def test_quota_refusal_closes_the_week():
    with tempfile.TemporaryDirectory() as d:
        b, pushes = _submit(Path(d), QUOTA)
        assert b.used_hours("kaggle") == 0.0            # our meter: 30 h left
        assert b.full("kaggle") and b.remaining("kaggle") == 0.0
        assert b.remaining_range("kaggle") == (0.0, 0.0)
        assert not b.afford("kaggle", 0.1)
        rec = b.data["full_weeks"][f"{G.week_key()}:kaggle"]
        assert rec["closed_by"] == "kaggle" and "quota" in rec["said"]
        assert pushes == [1], "the second submit must be refused before any push"


def test_other_push_failure_leaves_the_week_open():
    with tempfile.TemporaryDirectory() as d:
        b, pushes = _submit(Path(d), "push failed: 401 Unauthorized")
        assert not b.full("kaggle") and b.remaining("kaggle") == G.KAGGLE_WEEKLY_HOURS
        assert pushes == [1, 1]


def test_next_week_is_open_again():
    with tempfile.TemporaryDirectory() as d:
        b, _ = _submit(Path(d), QUOTA)
        saved = G.week_key
        G.week_key = lambda t=None: "2099-W01"
        try:
            assert not b.full("kaggle") and b.remaining("kaggle") == G.KAGGLE_WEEKLY_HOURS
        finally:
            G.week_key = saved


def main() -> int:
    bad = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("PASS", name)
            except Exception as e:
                bad += 1
                print("FAIL", name, repr(e))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
