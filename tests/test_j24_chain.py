"""J24 (5): a J24 cell cut and carried on IS the unbroken cell.

  identity  for each J24 arm (K-bins16, -blind, -shuffled): the unbroken
            `pilot_b2.run_pilot_cell` == `j24_chain.run_cell_carried` cut after
            chunk 1 and after chunk 2, each segment a NEW process — every life
            row, every trace row, every actor weight, the decision count. The
            shuffled twin's own permutation RNG must cross the carry for this.
  planted   a carry that drops the observation normaliser must NOT be
            identical (else identity proves nothing).
  refusals  a non-J24 arm; a chunk off the train/trace boundary.

Short runs (T 576 = 3 chunks of 192, warm-up 150, trace window 64, smoke seed
951 — never 941–943): they say nothing about eating. Under the box's budget:

    taskset -c 3 nice -n 19 /data/venvs/jack-std/bin/python tests/test_j24_chain.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SEED, T, WARM, CHUNK, TRACE = 951, 576, 150, 192, 64
ARMS = ("K-bins16", "K-bins16-blind", "K-bins16-shuffled")


def _short():
    from experiments import pilot_b as pb
    pb.WARMUP, pb.TRACE_EVERY = WARM, TRACE


def _flat(model):
    import jax
    import numpy as np
    leaves = jax.tree_util.tree_leaves(model.policy.actor_state.params)
    return [float(x) for x in np.concatenate(
        [np.asarray(l, dtype=np.float64).ravel() for l in leaves])]


def _child(argv):
    """child <unbroken|carried> <arm> <seed> <carry|-> <max_chunks|-> <out> [plant]"""
    _short()
    mode, arm, seed, carry, mc, out = argv[:6]
    plant = tuple(argv[6:])
    keep = {}
    if mode == "unbroken":
        import sbx
        orig = sbx.SAC.learn

        def learn(self, *a, **k):                # catch the model for its weights
            keep["model"] = self
            return orig(self, *a, **k)
        sbx.SAC.learn = learn
        from experiments.pilot_b2 import PILOT_RULE, run_pilot_cell
        rec = run_pilot_cell(arm, int(seed), PILOT_RULE["food_floor_only"], T=T,
                             eye_frame=PILOT_RULE["eye_frame"],
                             motor_frame=PILOT_RULE["motor_frame"],
                             native_substep=PILOT_RULE["native_substep"])
    else:
        import sbx
        orig = sbx.SAC.learn

        def learn(self, *a, **k):
            keep["model"] = self
            return orig(self, *a, **k)
        sbx.SAC.learn = learn
        from experiments.j24_chain import run_cell_carried
        rec = run_cell_carried(arm, int(seed), T, carry_dir=carry, chunk=CHUNK,
                               max_chunks=None if mc == "-" else int(mc), plant=plant)
    rec["_actor"] = _flat(keep["model"])
    Path(out).write_text(json.dumps(rec))


def _seg(*args):
    p = subprocess.run([sys.executable, __file__, "child", *map(str, args)], cwd=REPO,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(Path(args[5]).read_text())


def _strip(rec):
    drop = {"wall_s", "cpu_s", "decisions_per_s", "decisions_per_cpu_s", "segments", "_actor"}
    r = {k: v for k, v in rec.items() if k not in drop}
    r["lives"] = [{k: v for k, v in L.items() if not k.endswith("_s")} for L in r["lives"]]
    return r


def _carried(work, arm, *plant):
    tag = arm + "".join("_" + p for p in plant)
    d = work / f"carry_{tag}"
    a = _seg("carried", arm, SEED, d, 1, work / f"c_{tag}_0.json", *plant)
    b = _seg("carried", arm, SEED, d, 1, work / f"c_{tag}_1.json", *plant)
    c = _seg("carried", arm, SEED, d, "-", work / f"c_{tag}_2.json", *plant)
    assert a["cut"] and b["cut"] and a["decisions"] == CHUNK and b["decisions"] == 2 * CHUNK
    return c


def test_identity_and_planted():
    work = Path(tempfile.mkdtemp(prefix="j24chain_", dir="/data/tmp" if Path("/data/tmp").is_dir() else None))
    for arm in ARMS:
        u = _seg("unbroken", arm, SEED, "-", "-", work / f"u_{arm}.json")
        c = _carried(work, arm)
        assert u["decisions"] == c["decisions"] == T, arm
        assert len(c["segments"]) == 3 and c["segments"][1]["resumed_from"], arm
        assert _strip(u) == _strip(c), f"{arm}: carried cell is not the unbroken one"
        assert u["_actor"] == c["_actor"], f"{arm}: actor weights differ"
        print(f"identity {arm}: {len(u['lives'])} lives, {len(u.get('trace', []))} trace rows",
              flush=True)
    p = _carried(work, "K-bins16-shuffled", "no_norm")
    u = json.loads((work / "u_K-bins16-shuffled.json").read_text())
    assert u["_actor"] != p["_actor"], "planted: dropping the normaliser changed nothing"
    print("planted no_norm differs", flush=True)


def test_refusals():
    _short()
    from experiments.forge.r1_run import Refused
    from experiments.j24_chain import run_cell_carried
    try:
        run_cell_carried("K-row128", SEED, T, carry_dir="/nonexistent/x", chunk=CHUNK)
        raise AssertionError("a non-J24 arm ran")
    except Refused:
        pass
    try:
        run_cell_carried("K-bins16", SEED, T, carry_dir="/nonexistent/x", chunk=100)
        raise AssertionError("an off-boundary chunk ran")
    except Refused:
        pass
    print("refusals ok", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        _child(sys.argv[2:])
    else:
        test_refusals()
        test_identity_and_planted()
        print("ALL PASS", flush=True)
