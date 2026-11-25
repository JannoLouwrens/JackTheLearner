"""Round 1, J18 step 4: a run that is cut and carried on IS the unbroken run.

The only code change allowed between rungs S and M is that a 300k run can
resume (`r1_run._run_carried`, saved through `brain_b.save_life`). That is
only safe if a carried run cannot be told from an unbroken one, so:

  identity   for every arm that runs at rung M (E-food, E-food-lr0, K1, K2,
             N3, C2): ONE learn call (the rung-S path, `carry_dir=None`) ==
             the same run cut after chunk 1 and after chunk 2, each segment a
             NEW process — every life row, every trace row, every actor
             weight, the same decision count. K1's reset falls inside
             segment 2, after a load.
  death      the same identity through a life that ENDS after a load (death,
             respawn, the world's RNG): the twin, 5,760 decisions.
  planted    a carry that drops ONE thing must NOT be identical (else the
             identity test proves nothing): the observation normaliser; the
             trace so far; the policy's exploration-noise keys (SB3's load
             splits them once more); the actor alias (sbx loads `actor` as a
             separate copy, so the noise is never drawn again).
  cut = lost a cut record is `cut` + `resumable` and the LOCKED reader scores
             it as a lost run (status cut, score 0) — an unfinished carry
             never reads as a result.
  refusals   a chunk that would split a collection round or a trace window;
             a carry that is not whole; another run's carry.

Short runs (T 576 = 3 chunks of 192, warm-up 150, trace window 64, seed 1381 —
no rung's seed): they say nothing about eating. Two batches, each under the
box's 15 CPU-minutes:

    taskset -c 3 nice -n 19 /data/venvs/jack-std/bin/python tests/test_forge_r1_carry.py identity
    taskset -c 3 nice -n 19 /data/venvs/jack-std/bin/python tests/test_forge_r1_carry.py -identity
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.forge import r1_run as rr  # noqa: E402

SEED, T, WARM, CHUNK, TRACE = 1381, 576, 150, 192, 64
K1_RESET = 250                                      # inside segment 2 (192..384)
M_ARMS = ["E-food", "E-food-lr0", "K1", "K2", "N3", "C2"]


def _short_spec():
    s = rr.load_spec()
    s["baseline_settings"]["learning_starts"] = WARM
    for c in s["candidates"]:
        if c["id"] == "K1":
            c["settings"]["param_resets"] = [K1_RESET]
        if c["id"] == "C2":
            c["settings"]["learning_starts"] = WARM
    return s


def _flat(model):
    import jax
    import numpy as np
    leaves = jax.tree_util.tree_leaves(model.policy.actor_state.params)
    return np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in leaves])


def _child(argv):
    """One segment in its own process:
    child <arm> <carry|-> <max_chunks|-> <out> <T> <chunk> [plant]"""
    from experiments import pilot_b as pb
    arm, carry, mc, out = argv[:4]
    T_, chunk = int(argv[4]), int(argv[5])
    plant = tuple(argv[6:])
    pb.TRACE_EVERY = TRACE
    keep = {}
    rec = rr.run_arm(arm, SEED, T_, None, spec=_short_spec(), keep=keep,
                     carry_dir=None if carry == "-" else carry, chunk=chunk,
                     max_chunks=None if mc == "-" else int(mc), plant=plant)
    rec["_actor"] = _flat(keep["model"]).tolist()
    for k in ("wall_s", "cpu_s", "decisions_per_s", "decisions_per_cpu_s"):
        rec.pop(k, None)
    Path(out).write_text(json.dumps(rec))


def _seg(arm, carry, mc, out, *plant, T_=T, chunk=CHUNK):
    p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "child", arm,
                        str(carry), str(mc), str(out), str(T_), str(chunk), *plant], cwd=REPO,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-1500:]
    return json.loads(Path(out).read_text())


_WORK = None
_CACHE = {}


def _work():
    global _WORK
    if _WORK is None:
        _WORK = Path(tempfile.mkdtemp(prefix="r1carry_",
                                      dir="/data/tmp" if Path("/data/tmp").is_dir() else None))
    return _WORK


def _unbroken(arm):
    if ("u", arm) not in _CACHE:
        _CACHE[("u", arm)] = _seg(arm, "-", "-", _work() / f"u_{arm}.json")
    return _CACHE[("u", arm)]


def _carried(arm, *plant):
    """Three processes: 0->192 (save), 192->384 (load, save), 384->576 (load)."""
    key = ("c", arm, plant)
    if key not in _CACHE:
        tag = arm + "".join("_" + p for p in plant)
        d = _work() / f"carry_{tag}"
        a = _seg(arm, d, 1, _work() / f"c_{tag}_0.json", *plant)
        b = _seg(arm, d, 1, _work() / f"c_{tag}_1.json", *plant)
        c = _seg(arm, d, "-", _work() / f"c_{tag}_2.json", *plant)
        shutil.rmtree(d, ignore_errors=True)            # ~40 MB a carry
        _CACHE[key] = (a, b, c)
    return _CACHE[key]


def _same(u, c):
    return {k: u[k] == c[k] for k in ("lives", "trace", "_actor", "decisions")}


def refused(f):
    try:
        f()
    except rr.Refused:
        return True
    return False


# ---- PURE ------------------------------------------------------------------
def test_chunk_rule_refuses_a_split_round_or_window():
    rr.chunk_rule(300_000, rr.CHUNK, 32, 5000)               # rung M as it runs
    spec = rr.load_spec()
    assert spec["rungs"]["M"]["decisions"] % rr.CHUNK == 0
    assert spec["baseline_settings"]["train_freq"] == 32
    assert refused(lambda: rr.chunk_rule(300_000, 5000, 32, 5000)), "splits a round"
    assert refused(lambda: rr.chunk_rule(300_000, 3200, 32, 5000)), "splits a trace window"
    assert refused(lambda: rr.chunk_rule(150_000, 20_000, 32, 5000)), "T off a boundary"
    assert refused(lambda: rr.chunk_rule(960, 0, 32, 64))


def test_production_chunk_is_the_trace_window_times_rounds():
    from experiments import pilot_b as pb
    assert rr.CHUNK % pb.TRACE_EVERY == 0 and rr.CHUNK % pb.TRAIN_FREQ == 0
    assert rr.SEG_WALL_S < pb.CHILD_TIMEOUT_S < rr.REMOTE_TIMEOUT_S


def test_a_carry_is_whole_only_with_its_last_file():
    d = Path(tempfile.mkdtemp(dir="/data/tmp" if Path("/data/tmp").is_dir() else None))
    assert not rr.carry_whole(d) and not rr.carry_whole(None)
    for n in rr.CARRY_NEEDS[:-1]:
        (d / n).write_text("x")
    assert not rr.carry_whole(d), "carry.json is written last"
    (d / "carry.json").write_text(json.dumps({"arm": "K1", "seed": 1}))
    assert rr.carry_whole(d)
    # another run's carry is refused BEFORE anything is loaded
    assert refused(lambda: rr._carry_load(d, None, None, {"arm": "K2", "seed": 1}))
    (d / "life.pkl").unlink()
    assert refused(lambda: rr._carry_load(d, None, None, {"arm": "K1", "seed": 1}))
    shutil.rmtree(d)


def test_a_reward_wrapper_is_not_carried():
    assert refused(lambda: rr.run_arm("K6", SEED, T, None, spec=_short_spec(),
                                      carry_dir="/nonexistent/x", chunk=CHUNK))


# ---- LEARNER (sbx) ---------------------------------------------------------
def _has_sbx():
    try:
        import sbx  # noqa: F401
        return True
    except ImportError:
        return False


def _identity(arm):
    u = _unbroken(arm)
    a, b, c = _carried(arm)
    assert a["cut"] and a["resumable"] and a["decisions"] == CHUNK
    assert b["cut"] and b["resumable"] and b["decisions"] == 2 * CHUNK
    assert "cut" not in c and c["decisions"] == u["decisions"] == T
    assert len(u["lives"]) >= 1 and len(u["trace"]) == T // TRACE
    assert all(_same(u, c).values()), (arm, _same(u, c))
    segs = c["forge"]["segments"]
    assert [(s["start"], s["end"]) for s in segs] == [(0, 192), (192, 384), (384, 576)]
    assert segs[0]["resumed_from"] is None and segs[1]["resumed_from"] and segs[2]["resumed_from"]
    assert {k: v for k, v in c["forge"].items() if k != "segments"} == u["forge"]
    return u, c


def test_learner_identity_baseline():
    import numpy as np
    u, _ = _identity("E-food")
    twin = _unbroken("E-food-lr0")
    assert not np.array_equal(u["_actor"], twin["_actor"]), "control: the baseline learned"


def test_learner_identity_twin():
    _identity("E-food-lr0")


def test_learner_identity_k1_reset_after_a_load():
    u, c = _identity("K1")
    assert c["forge"]["param_resets_done"] == 1
    assert _carried("K1")[0]["decisions"] < K1_RESET < _carried("K1")[1]["decisions"]


def test_learner_identity_k2_tqc():
    _, c = _identity("K2")
    assert c["forge"]["algo"] == "TQC"


def test_learner_identity_n3():
    _, c = _identity("N3")
    assert c["forge"]["learner_kwargs"]["n_steps"] == 3


def test_learner_identity_c2():
    _identity("C2")


def test_learner_identity_through_a_death_after_a_load():
    """The short runs above never end a life. Here the never-trains twin lives
    long enough to die and be respawned AFTER a load (the world's own RNG, the
    respawn, the needs — all from the carried objects), 3 chunks of 1,920."""
    T_, chunk = 5760, 1920
    u = _seg("E-food-lr0", "-", "-", _work() / "u_death.json", T_=T_, chunk=chunk)
    d = _work() / "carry_death"
    cuts = [_seg("E-food-lr0", d, mc, _work() / f"c_death_{i}.json", T_=T_, chunk=chunk)
            for i, mc in enumerate((1, 1, "-"))]
    shutil.rmtree(d, ignore_errors=True)
    c = cuts[2]
    ends, cum = [], 0
    for life in u["lives"]:
        cum += life["decisions"]
        if life["cause"] != "unfinished":
            ends.append(cum)
    assert any(e > chunk for e in ends), ("control: no life ended after the first load", ends)
    assert len(u["lives"]) >= 2 and all(_same(u, c).values()), (_same(u, c), ends)
    assert [len(x["lives"]) for x in cuts[:2]] == [sum(e <= chunk for e in ends),
                                                   sum(e <= 2 * chunk for e in ends)]


def test_learner_planted_broken_carries_differ():
    u = _unbroken("E-food")
    no_norm = _carried("E-food", "no_norm")[2]
    assert not _same(u, no_norm)["_actor"], "a dropped normaliser must change the learner"
    no_trace = _carried("E-food", "no_trace")[2]
    same = _same(u, no_trace)
    assert not same["trace"] and same["lives"] and same["_actor"], same
    no_key = _carried("E-food", "no_sde_key")[2]
    assert not _same(u, no_key)["_actor"], "load's extra noise split must show"
    no_rewire = _carried("E-food", "no_rewire")[2]
    assert not _same(u, no_rewire)["_actor"], "a frozen noise draw must show"


def test_learner_a_cut_run_is_a_lost_run_to_the_locked_reader():
    from experiments.forge import r1_read
    a, _, c = _carried("E-food")
    assert r1_read.score_run(a, T, 0.5)["status"] == "cut"
    assert r1_read.score_run(a, T, 0.5)["score"] == 0.0
    assert r1_read.score_run(dict(c, rc=0), T, 0.5)["status"] == "ok"


if __name__ == "__main__":
    if sys.argv[1:2] == ["child"]:
        _child(sys.argv[2:])
        sys.exit(0)
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    for pick in sys.argv[1:]:          # "identity" = only those; "-identity" = the rest
        tests = [(n, f) for n, f in tests if (pick.lstrip("-") in n) != pick.startswith("-")]
    sbx_ok = _has_sbx()
    bad = skipped = 0
    for n, f in tests:
        if n.startswith("test_learner_") and not sbx_ok:
            skipped += 1
            print("SKIP", n, "(no sbx in this venv — run under /data/venvs/jack-std)")
            continue
        try:
            f()
            print("PASS", n, flush=True)
        except Exception as e:   # noqa: BLE001
            bad += 1
            print("FAIL", n, repr(e)[:1500], flush=True)
    if _WORK is not None:
        shutil.rmtree(_WORK, ignore_errors=True)
    print(f"{len(tests) - bad - skipped}/{len(tests)} passed, {skipped} skipped")
    sys.exit(1 if bad else 0)
