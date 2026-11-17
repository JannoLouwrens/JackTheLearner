"""pilot_b's partial life log — a killed run still returns its finished lives.

ORDERS J11 prep (11-17 20:20), before the rule and before any pilot seed. Why:
J8's packed learner kernels wrote a cell's lives only when the run ENDED; a
child killed at CHILD_TIMEOUT_S (8.5 h) or a kernel hitting Kaggle's cap
returned `lives: []`. J11's 30 learners are priced at 4.6-7.4 h against that
8.5 h cut — 13% margin on the slowest host seen (pilot_b2 "PILOT COST").

Not a pilot run, no reading of any arm. Seed 381 (the card's smoke seed; never
301-345, never the pilot's 401+), J8's default world, PARTIAL_EVERY = 1.

  identity  run_floor(held8) with partial_path == without: every life row.
  prefix    the partial file after the run == the run's finished lives.
  kill      `pilot_b cell` (held8) SIGKILLed after its 2nd partial write:
            run_parallel's fallback `from_partial` returns >= 2 lives, marked
            cut, equal to the uninterrupted run's first lives row for row.
  control   the same kill with no partial file -> `not_found`, 0 lives (J8).
  learner   (jack-std only, skipped elsewhere) run_b 700 decisions with a
            partial path: lives and actor weights identical to without.

Run: taskset -c 3 nice -n 19 /data/venvs/jack-std/bin/python \
         -m experiments.tests.test_pilot_b_partial
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SEED, T_FLOOR = 381, 9000

_CHILD = r'''
import sys; sys.path.insert(0, %r)
from experiments import pilot_b as pb
pb.PARTIAL_EVERY = 1
pb.run_cell("held8", %d, 10**7, partial_path=%r)
'''


def _killed_cell(pf: Path, partial: bool) -> tuple:
    """Start a held8 cell; kill it once 2 lives are in the partial file (or,
    without a partial file, after the same wall time the partial run took)."""
    code = _CHILD % (str(REPO), SEED, str(pf) if partial else None)
    p = subprocess.Popen([sys.executable, "-c", code], cwd=REPO,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    t0 = time.time()
    return p, t0


def _wait_lives(pf: Path, n: int, p, limit_s: float = 900) -> float:
    t0 = time.time()
    while time.time() - t0 < limit_s:
        if p.poll() is not None:
            raise RuntimeError(f"child exited rc={p.returncode}: {p.stderr.read()[-500:]}")
        if pf.exists() and len(json.loads(pf.read_text())["lives"]) >= n:
            return time.time() - t0
        time.sleep(1)
    raise TimeoutError("no partial lives")


def main() -> int:
    from experiments import pilot_b as pb
    pb.PARTIAL_EVERY = 1
    ok, out = True, {}
    tmp = Path(tempfile.mkdtemp(dir="/data/tmp"))

    base = pb.run_floor("held8", SEED, T_FLOOR)
    pf = tmp / "floor.partial.json"
    withp = pb.run_floor("held8", SEED, T_FLOOR, partial_path=pf)
    part = json.loads(pf.read_text())
    finished = [l for l in withp["lives"] if l["cause"] != "unfinished"]
    out["identity"] = base["lives"] == withp["lives"]
    out["prefix"] = part["lives"] == finished and len(finished) >= 2
    out["floor_lives"] = len(finished)
    ok &= out["identity"] and out["prefix"]

    # kill: a real `cell` child, killed mid-run
    pk = tmp / "kill.partial.json"
    p, _ = _killed_cell(pk, True)
    took = _wait_lives(pk, 2, p)
    os.kill(p.pid, signal.SIGKILL)
    p.wait()
    rec = pb.from_partial("held8", SEED, pk)
    n = len(rec["lives"])
    out["kill"] = {"lives": n, "cut": rec.get("cut"), "decisions": rec.get("decisions_reached"),
                   "rc": p.returncode, "wall_s": round(took, 1)}
    out["kill_ok"] = (p.returncode == -signal.SIGKILL and rec.get("cut") is True and n >= 2
                      and rec["lives"] == finished[:n])
    ok &= out["kill_ok"]

    # control: same kill, no partial file -> nothing comes back (J8's behaviour)
    pc = tmp / "ctrl.partial.json"
    p, _ = _killed_cell(pc, False)
    time.sleep(took)
    alive = p.poll() is None
    os.kill(p.pid, signal.SIGKILL)
    p.wait()
    ctl = pb.from_partial("held8", SEED, pc)
    out["control"] = {"alive_at_kill": alive, "lives": len(ctl["lives"]),
                      "not_found": ctl.get("not_found")}
    out["control_fails"] = alive and ctl.get("not_found") is True and not ctl["lives"]
    ok &= out["control_fails"]

    try:
        import sbx  # noqa: F401
        have_sbx = True
    except ImportError:
        have_sbx = False
    if have_sbx:
        from experiments.tests.test_pilot_b_knobs import _run, WARM
        pb.WARMUP = WARM
        pb._single_thread()
        a = _run(pb)
        b = _run(pb, partial_path=tmp / "b.partial.json")
        out["learner"] = {"lives_equal": a["run"]["lives"] == b["run"]["lives"],
                          "actor_diff": float(np.abs(a["actor"] - b["actor"]).max()),
                          "lives": len(a["run"]["lives"])}
        ok &= out["learner"]["lives_equal"] and out["learner"]["actor_diff"] == 0.0
    else:
        out["learner"] = "skipped (no sbx in this venv)"

    out["PASS"] = bool(ok)
    print(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
