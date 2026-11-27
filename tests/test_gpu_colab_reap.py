"""S17: a failed Colab submit releases the orphan VM it left and retries once; a live local session is never touched."""
import sys
from types import SimpleNamespace as NS

from experiments import gpu as G


class FakeClient:
    def __init__(self, endpoints):
        self.server = list(endpoints)
        self.unassigned = []

    def list_assignments(self):
        return [NS(endpoint=e) for e in self.server]

    def unassign(self, endpoint):
        self.unassigned.append(endpoint)
        self.server.remove(endpoint)


def test_timed_out_assign_that_did_assign_is_unassigned():
    c = FakeClient(["orphan-ep"])            # server assigned; nothing recorded locally
    assert G.colab_reap(c, set()) == ["orphan-ep"]
    assert c.unassigned == ["orphan-ep"] and c.server == []


def test_live_local_session_is_never_touched():
    c = FakeClient(["live-ep", "orphan-ep"])
    assert G.colab_reap(c, {"live-ep"}) == ["orphan-ep"]
    assert c.server == ["live-ep"] and "live-ep" not in c.unassigned
    c2 = FakeClient(["live-ep"])
    assert G.colab_reap(c2, {"live-ep"}) == [] and c2.unassigned == []


def test_reap_source_runs_standalone():
    import inspect
    ns = {}
    exec(inspect.getsource(G.colab_reap), ns)     # what the colab venv executes
    c = FakeClient(["x"])
    assert ns["colab_reap"](c, set()) == ["x"]


class _MP:
    def __init__(self):
        self.saved = []

    def setattr(self, obj, name, val):
        self.saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    def undo(self):
        for obj, name, val in reversed(self.saved):
            setattr(obj, name, val)


def _patch(monkeypatch, results, reaped):
    calls = []

    def once(script, gpu, timeout_s, fetch):
        calls.append(1)
        return results[len(calls) - 1]
    monkeypatch.setattr(G, "_run_on_colab_once", once)
    monkeypatch.setattr(G, "colab_reap_orphans", lambda: reaped)
    return calls


def test_failed_run_with_orphan_retries_once(monkeypatch):
    fail = G.JobResult("colab", False, message="exit 1: ReadTimeout")
    calls = _patch(monkeypatch, [fail, G.JobResult("colab", True)], ["orphan-ep"])
    r = G.run_on_colab("s.py")
    assert r.ok and len(calls) == 2 and "orphan-ep" in r.message


def test_retry_happens_at_most_once(monkeypatch):
    fail = G.JobResult("colab", False, message="exit 1")
    calls = _patch(monkeypatch, [fail, fail, fail], ["o1"])
    assert not G.run_on_colab("s.py").ok and len(calls) == 2


def test_plain_job_failure_and_busy_refusal_not_retried(monkeypatch):
    calls = _patch(monkeypatch, [G.JobResult("colab", False, message="exit 1")], [])
    assert not G.run_on_colab("s.py").ok and len(calls) == 1
    busy = G.JobResult("colab", False, message="session busy — refused before running")
    reaped = []
    monkeypatch.setattr(G, "_run_on_colab_once", lambda *a: busy)
    monkeypatch.setattr(G, "colab_reap_orphans", lambda: reaped.append(1) or ["x"])
    assert G.run_on_colab("s.py") is busy and reaped == []


def main() -> int:
    import inspect
    bad = 0
    for name, fn in list(globals().items()):
        if not name.startswith("test_"):
            continue
        mp = _MP()
        try:
            fn(*([mp] if inspect.signature(fn).parameters else []))
            print("PASS", name)
        except Exception as e:
            bad += 1
            print("FAIL", name, repr(e))
        finally:
            mp.undo()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
