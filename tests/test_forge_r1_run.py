"""Round 1's kernel (F3 / ORDERS J18): every arm of the frozen rule runs with
the rule's settings and nothing else; each wrong reading is planted and caught.

Two tiers. PURE (any venv): settings, the learner call, packing, the job, the
chain. LEARNER (needs sbx — /data/venvs/jack-std): short runs, seed 1381 (no
rung's seed, never E2c's 801-805), warm-up cut to 300, T = 700 — they say
nothing about eating, only that each knob reaches the learner:

  identity  the baseline through run_arm == pilot_b2.run_pilot_cell("E-food"),
            E2c's own code path: every life row, every trace row (SAC's
            losses), every actor weight.
  twin      lr 0: the actor is its initial weights (control: baseline moved).
  K1        a reset point inside the run is taken (control: baseline 0).
  K2        the learner is TQC dropping 2 x 2 quantiles.
  K5        3 updates per 32 decisions: exactly 3/32 of the baseline's.
  K6        the replay buffer holds reward x 0.01 (same warm-up decisions as
            the baseline, whose buffer holds the world's reward).
  N3 C1 C2  the setting is on the built learner.

    taskset -c 3 nice -n 19 /data/venvs/jack-std/bin/python tests/test_forge_r1_run.py
"""
import ast
import copy
import subprocess
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.forge import r1_run as rr  # noqa: E402

SEED, T, WARM = 1381, 700, 300
BASE_KW = dict(learning_rate=3e-4, buffer_size=150_000, batch_size=256, tau=0.005,
               gamma=0.995, learning_starts=5000, train_freq=32, gradient_steps=32,
               ent_coef=0.1, use_sde=True, seed=7, verbose=0,
               policy_kwargs=dict(layer_norm=True, net_arch=[256, 256]))


def spec():
    return rr.load_spec()


def call(arm, rung="S", T_=150_000, s=None):
    s = s or spec()
    return rr.learner_call(rr.arm_settings(s, arm), 7, T_, rung)


def refused(f):
    try:
        f()
    except rr.Refused:
        return True
    return False


# ---- PURE ------------------------------------------------------------------
def test_ten_arms_in_the_rules_order():
    assert rr.arms(spec()) == ["E-food", "E-food-lr0", "K1", "K2", "K3", "K5", "K6",
                               "N3", "C1", "C2"]
    for a in rr.arms(spec()):
        call(a)


def test_baseline_call_is_make_bs_call():
    assert call("E-food") == ("SAC", BASE_KW)
    # and that literal is what pilot_b.make_b really passes (sbx stubbed)
    from experiments import pilot_b as pb
    got = {}
    fake = types.ModuleType("sbx")
    fake.SAC = lambda policy, venv, **kw: got.update(policy=policy, **kw)
    real = sys.modules.get("sbx")
    sys.modules["sbx"] = fake
    try:
        pb.make_b("VENV", 7, 150_000)
    finally:
        if real is None:
            del sys.modules["sbx"]
        else:
            sys.modules["sbx"] = real
    assert got.pop("policy") == "MlpPolicy"
    assert got == BASE_KW, got


def _diff(arm, rung="S"):
    algo, kw = call(arm, rung)
    d = {k: v for k, v in kw.items() if BASE_KW.get(k, "absent") != v}
    return algo, d


def test_each_arm_changes_only_what_the_rule_says():
    assert _diff("E-food-lr0") == ("SAC", {"learning_rate": 0.0})
    assert _diff("K1") == ("SAC", {"param_resets": [50_000, 100_000]})
    assert _diff("K1", "M")[1] == {"param_resets": [50_000, 100_000, 150_000,
                                                    200_000, 250_000]}
    assert _diff("K2") == ("TQC", {"policy_kwargs": dict(
        layer_norm=True, net_arch=[256, 256], top_quantiles_to_drop_per_net=2)})
    assert _diff("K3") == ("SAC", {"ent_coef": "auto"})
    assert _diff("K5") == ("SAC", {"gradient_steps": 3})
    assert _diff("K6") == ("SAC", {"ent_coef": "auto"})
    assert rr.arm_settings(spec(), "K6")["learner_reward_scale"] == 0.01
    assert all("learner_reward_scale" not in rr.arm_settings(spec(), a)
               for a in rr.arms(spec()) if a != "K6")
    assert _diff("N3") == ("SAC", {"n_steps": 3})
    assert _diff("C1") == ("SAC", {"learning_rate": 0.000198331, "batch_size": 512,
                                   "tau": 0.00761166})
    assert _diff("C2") == ("SAC", {"learning_rate": 0.000378774, "batch_size": 128,
                                   "tau": 0.00588961, "learning_starts": 2500})


def test_buffer_is_the_whole_run():
    assert call("E-food", "M", 300_000)[1]["buffer_size"] == 300_000


def test_planted_wrong_terms_are_refused():
    s = spec()
    assert refused(lambda: rr.arm_settings(s, "K9")), "unknown arm"
    for plant in ({"weight_decay": 0.01}, {"norm_obs": False}, {"buffer": "100k"}):
        t = copy.deepcopy(s)
        t["candidates"][0]["settings"].update(plant)
        assert refused(lambda: rr.arm_settings(t, "K1")), plant
    assert refused(lambda: call("K1", "L")), "K1 at a rung the rule names no resets for"
    assert refused(lambda: call("K1", None)), "K1 on a smoke run of the rule's spec"
    t = copy.deepcopy(s)
    t["baseline_settings"]["top_quantiles_to_drop_per_net"] = 2
    assert refused(lambda: call("E-food", s=t)), "truncation without TQC"
    t = copy.deepcopy(s)
    t["baseline_settings"]["layer_norm"] = "all"
    assert refused(lambda: call("E-food", s=t))
    t = copy.deepcopy(s)
    t["baseline_settings"]["algo"] = "sbx.PPO"
    assert refused(lambda: call("E-food", s=t))


def test_packing_rung_s():
    s = spec()
    g = rr.groups(s, "S")
    flat = [c for v in g.values() for c in v]
    assert len(flat) == 30 and len(set(flat)) == 30 and len(g) == 8
    assert all(len(v) <= rr.PER_KERNEL for v in g.values())
    assert {sd for _, sd in flat} == {1401, 1402, 1403}
    assert all(sum(1 for a, sd in flat if a == arm) == 3 for arm in rr.arms(s))
    for v in g.values():                       # no arm twice on one host
        assert len({a for a, _ in v}) == len(v)
    assert refused(lambda: rr.groups(s, "S", ["K1"])), "S runs every arm"


def test_packing_above_s_needs_the_ladders_entrants():
    s = spec()
    assert refused(lambda: rr.groups(s, "M"))
    assert refused(lambda: rr.groups(s, "M", ["E-food", "K9"]))
    flat = [c for v in rr.groups(s, "M", ["E-food", "E-food-lr0", "K2"]).values() for c in v]
    assert len(flat) == 15 and {sd for _, sd in flat} == {1411, 1412, 1413, 1414, 1415}


def test_job_pins_one_commit_and_the_frozen_spec():
    pin = "a" * 40
    body = rr.job_body(rr.groups(spec(), "S")["g0"], "S", pin)
    ast.parse(body)
    assert repr(pin) in body and repr(rr.spec_sha256()) in body
    assert "('E-food', 1401)" in body and "mujoco==3.17.0" in body
    assert body.index("checkout") < body.index("from experiments.forge.r1_run")
    assert refused(lambda: rr.job_body([], "S", "main"))
    assert refused(lambda: rr.job_body([], "S", "a" * 39))


def test_pin_must_hold_the_rule_and_the_kernel():
    freeze = subprocess.run(["git", "-C", str(REPO), "rev-parse", "594752d0"],
                            capture_output=True, text=True).stdout.strip()
    assert len(freeze) == 40
    assert not rr.pin_has_rule(freeze), "the freeze commit has no kernel file"
    assert not rr.pin_has_rule("0" * 40)
    assert refused(lambda: rr.kernel("g0", "S", freeze))


def test_chain_waits_out_the_guard_and_survives_a_failure():
    calls, naps = [], []
    script = {"g0": [RuntimeError("Uncommitted changes to tracked files -- x"), None],
              "g1": [RuntimeError("kaggle said no")],
              "g2": [rr.Refused("already home")],
              "g3": [None]}
    real = rr.kernel

    def fake(name, rung, pin, climbers=None):
        calls.append(name)
        e = script[name].pop(0)
        if e:
            raise e
    rr.kernel = fake
    try:
        done = rr.chain(["g0", "g1", "g2", "g3"], "S", "a" * 40, sleep=naps.append)
    finally:
        rr.kernel = real
    assert calls == ["g0", "g0", "g1", "g2", "g3"] and naps == [rr.GUARD_WAIT_S]
    assert done["g0"] == "home" and done["g3"] == "home"
    assert done["g1"].startswith("failed") and done["g2"].startswith("refused")


def test_cells_from_artifacts_never_invents():
    import json
    import tempfile
    d = Path(tempfile.mkdtemp(dir="/data/tmp" if Path("/data/tmp").is_dir() else None))
    (d / "r1_K1_s1401.json").write_text(json.dumps({"arm": "K1", "seed": 1401, "lives": [1, 2]}))
    (d / "r1_K2_s1401.partial.json").write_text(json.dumps({"lives": [1], "decisions": 9000}))
    art = {f.name: str(f) for f in d.iterdir()}
    r = rr.cells_from_artifacts(art, [("K1", 1401), ("K2", 1401), ("K3", 1401)])["cells"]
    assert r["K1_s1401"]["lives"] == [1, 2] and "cut" not in r["K1_s1401"]
    assert r["K2_s1401"]["cut"] and r["K2_s1401"]["decisions_reached"] == 9000
    assert r["K3_s1401"]["not_found"] and r["K3_s1401"]["lives"] == []
    assert rr.cells_from_artifacts({}, [("K1", 1401)]) is None


# ---- J18 step 4: the kernel side of the carry (pure) -------------------------
def _tmp():
    import tempfile
    return Path(tempfile.mkdtemp(dir="/data/tmp" if Path("/data/tmp").is_dir() else None))


def _whole(d):
    d.mkdir(parents=True, exist_ok=True)
    for n in rr.CARRY_NEEDS:
        (d / n).write_text("x")
    return d


M_CLIMB = ["E-food", "E-food-lr0", "K2"]


def test_job_body_carries_above_s_only():
    pin = "b" * 40
    g = rr.groups(spec(), "M", M_CLIMB)["g0"]
    s_body = rr.job_body(rr.groups(spec(), "S")["g0"], "S", pin)
    m0, m1 = rr.job_body(g, "M", pin), rr.job_body(g[:2], "M", pin, segment=1)
    for b in (s_body, m0, m1):
        ast.parse(b)
        assert b.index("_T0 = _t.time()") < b.index("pip")   # deadline counts pip install
    assert "carry_root" not in s_body and "SEG_WALL_S" not in s_body
    assert "carry_root" in m0 and "resume_root=None" in m0 and "deadline=_T0 + SEG_WALL_S" in m0
    assert "resume_root='/kaggle/input'" in m1 and "\"segment\": 1" in m1
    assert repr(tuple(g[2])) not in m1                         # only the runs carried on
    assert refused(lambda: rr.job_body(g, "S", pin, segment=1)), "S never carries"
    assert refused(lambda: rr.job_body(g, "M", pin, segment=rr.MAX_SEGMENTS))


def test_find_carry_one_whole_or_refused():
    root = _tmp()
    assert refused(lambda: rr.find_carry(root, "K2", 1411)), "none: never a fresh start"
    part = root / "a" / rr.carry_name("K2", 1411)
    part.mkdir(parents=True)
    (part / "model.zip").write_text("x")                       # no carry.json: not whole
    assert refused(lambda: rr.find_carry(root, "K2", 1411))
    deep = _whole(root / "notebooks" / "u" / "slug" / "carry" / rr.carry_name("K2", 1411))
    assert rr.find_carry(root, "K2", 1411) == deep
    assert refused(lambda: rr.find_carry(root, "K2", 1412)), "another seed's carry is not his"
    _whole(root / "b" / rr.carry_name("K2", 1411))
    assert refused(lambda: rr.find_carry(root, "K2", 1411)), "two carries: which life?"


class _FakeProc:
    def __init__(self, argv, **kw):
        self.argv, self.returncode = argv, 0
        out = Path(argv[argv.index("--out") + 1])
        arm, seed = argv[argv.index("--arm") + 1], int(argv[argv.index("--seed") + 1])
        out.write_text(__import__("json").dumps({"arm": arm, "seed": seed, "lives": []}))
        _FakeProc.seen.append(argv)

    def communicate(self, timeout=None):
        return b"", b""


def test_run_parallel_passes_each_run_its_own_carry_and_the_deadline():
    import subprocess as sp
    real, _FakeProc.seen = sp.Popen, []
    out, carry, mount = _tmp(), _tmp(), _tmp()
    c1 = _whole(mount / "x" / "carry" / rr.carry_name("K2", 1411))
    sp.Popen = _FakeProc
    try:
        rr.run_parallel([("K2", 1411), ("E-food", 1411)], out / "a", "M",
                        carry_root=carry, deadline=123.5)
        a = _FakeProc.seen
        assert [x[x.index("--carry-dir") + 1] for x in a] == [
            str(carry / "carry_K2_s1411"), str(carry / "carry_E-food_s1411")]
        assert all(x[x.index("--deadline") + 1] == "123.5" for x in a)
        assert not any("--resume-from" in x for x in a), "segment 0 starts fresh"
        _FakeProc.seen = []
        rr.run_parallel([("K2", 1411)], out / "b", "M", carry_root=carry, resume_root=mount)
        b = _FakeProc.seen[0]
        assert b[b.index("--resume-from") + 1] == str(c1)
        _FakeProc.seen = []
        assert refused(lambda: rr.run_parallel([("K2", 1411), ("E-food", 1411)], out / "c",
                                               "M", carry_root=carry, resume_root=mount))
        assert _FakeProc.seen == [], "one missing carry refuses before ANY run starts"
        assert refused(lambda: rr.run_parallel([("K2", 1411)], out / "d", "M",
                                               resume_root=mount))
        _FakeProc.seen = []
        rr.run_parallel([("K2", 1401)], out / "e", "S")
        assert "--carry-dir" not in _FakeProc.seen[0], "rung S's child is rung S's"
    finally:
        sp.Popen = real


def _seg(i, cells, pin="c" * 40, dur=100.0, job=None):
    return {"round": "r1", "rung": "M", "pin": pin, "spec_sha256": "z", "segment": i,
            "versions": {"v": i}, "cells": cells,
            "kernel": {"job_id": job or f"u/k{i}", "duration_s": dur,
                       "charge_seconds": 0, "name": "g0", "pin": pin}}


def _cut(a, sd):
    return {"arm": a, "seed": sd, "cut": True, "resumable": True, "lives": [1]}


def _fin(a, sd):
    return {"arm": a, "seed": sd, "lives": [1, 2, 3], "decisions": 300000}


def test_assemble_gives_each_run_once_from_its_last_segment():
    s0 = _seg(0, {"K2_s1411": _cut("K2", 1411), "E-food_s1411": _fin("E-food", 1411)})
    s1 = _seg(1, {"K2_s1411": _fin("K2", 1411)}, dur=50.0)
    raw = rr.assemble([s0, s1])
    assert raw["cells"]["K2_s1411"] == _fin("K2", 1411)
    assert raw["cells"]["E-food_s1411"] == _fin("E-food", 1411)
    assert raw["kernel"]["duration_s"] == 150.0 and raw["kernel"]["job_id"] == ["u/k0", "u/k1"]
    assert raw["pin"] == "c" * 40 and raw["rung"] == "M" and raw["segments"] == 2
    assert rr.resumable(s0) == [("K2", 1411)] and rr.resumable(s1) == []
    assert refused(lambda: rr.assemble([s0, _seg(1, {"K2_s1411": _fin("K2", 1411)},
                                                   pin="d" * 40)])), "one pin"
    assert refused(lambda: rr.assemble([s0, _seg(1, {"E-food_s1411": _fin("E-food", 1411)})]))
    assert refused(lambda: rr.assemble([s0, _seg(2, {"K2_s1411": _fin("K2", 1411)})]))
    s1n = _seg(1, {"K2_s1411": _fin("K2", 1411)})
    s1n["kernel"]["duration_s"] = None
    assert rr.assemble([s0, s1n])["kernel"]["duration_s"] is None, "no cost is stated as none"


def _fake_gpu(script):
    """experiments.gpu stand-in: each submit pops the next payload."""
    import json
    calls = []

    class Sub:
        ok, message, charge_seconds = True, "", 0

    def submit(job, **kw):
        calls.append(kw)
        payload = script.pop(0)
        sub = Sub()
        sub.job_id, sub.duration_s = f"u/k{len(calls) - 1}", 1000.0 * len(calls)
        sub.artifacts = {"forge_r1_cells.json": json.dumps(payload)}
        return sub
    fake = types.SimpleNamespace(build_job=lambda body: body, submit=submit,
                                 result_json=lambda sub, n: json.loads(sub.artifacts[n]))
    return fake, calls


def _with_fakes(script, f):
    real_gpu = sys.modules.get("experiments.gpu")
    fake, calls = _fake_gpu(script)
    sys.modules["experiments.gpu"] = fake
    real = (rr.OUT, rr.SEGS, rr.pin_has_rule)
    d = _tmp()
    rr.OUT, rr.SEGS, rr.pin_has_rule = d, d / "segments", lambda pin: True
    try:
        return f(d, calls)
    finally:
        rr.OUT, rr.SEGS, rr.pin_has_rule = real
        if real_gpu is not None:
            sys.modules["experiments.gpu"] = real_gpu
        else:
            sys.modules.pop("experiments.gpu", None)


def test_kernel_carries_a_cut_run_into_a_mounted_next_segment():
    pin = "e" * 40
    g0 = rr.groups(spec(), "M", M_CLIMB)["g0"]
    a, sd = g0[0]
    first = {f"{x}_s{y}": _fin(x, y) for x, y in g0}
    first[f"{a}_s{sd}"] = _cut(a, sd)
    p0 = dict(_seg(0, first, pin=pin), spec_sha256=rr.spec_sha256())
    p1 = dict(_seg(1, {f"{a}_s{sd}": _fin(a, sd)}, pin=pin), spec_sha256=rr.spec_sha256())

    def go(d, calls):
        r0 = rr.kernel("g0", "M", pin, M_CLIMB)
        assert r0["next_segment"] == 1
        assert list(d.glob("M_g*.json")) == [], "nothing the reader globs until the group ends"
        assert (d / "segments" / "M_g0_seg0.json").exists()
        assert calls[0]["kernel_sources"] is None and calls[0]["fetch_only"] is True
        assert not set(rr.CARRY_NEEDS) & set(calls[0]["fetch"]), "carries stay on Kaggle"
        assert rr.cell_file(a, sd) in calls[0]["fetch"]
        assert refused(lambda: rr.kernel("g0", "M", pin, M_CLIMB, segment=2)), "needs seg 1"
        assert refused(lambda: rr.kernel("g0", "M", "f" * 40, M_CLIMB, segment=1)), "one pin"
        raw = rr.kernel("g0", "M", pin, M_CLIMB, segment=1)
        assert calls[1]["kernel_sources"] == ["u/k0"], "segment 1 mounts segment 0's kernel"
        assert [p.name for p in d.glob("M_g*.json")] == ["M_g0.json"]
        assert raw["cells"][f"{a}_s{sd}"] == _fin(a, sd) and len(raw["cells"]) == len(g0)
        assert raw["kernel"]["duration_s"] == 1000.0 + 2000.0
        assert refused(lambda: rr.kernel("g0", "M", pin, M_CLIMB)), "a group runs once"
    _with_fakes([p0, p1], go)


def test_kernel_stops_carrying_at_max_segments_and_the_run_stays_cut():
    pin = "e" * 40
    g0 = rr.groups(spec(), "M", M_CLIMB)["g0"]
    a, sd = g0[0]
    first = {f"{x}_s{y}": _fin(x, y) for x, y in g0}
    first[f"{a}_s{sd}"] = _cut(a, sd)
    segs = [dict(_seg(0, first, pin=pin), spec_sha256=rr.spec_sha256())] + [
        dict(_seg(i, {f"{a}_s{sd}": _cut(a, sd)}, pin=pin), spec_sha256=rr.spec_sha256())
        for i in range(1, rr.MAX_SEGMENTS)]

    def go(d, calls):
        for i in range(rr.MAX_SEGMENTS):
            r = rr.kernel("g0", "M", pin, M_CLIMB, segment=i)
        assert "next_segment" not in r and (d / "M_g0.json").exists()
        assert r["cells"][f"{a}_s{sd}"]["cut"], "a run out of segments is a lost run"
        assert len(calls) == rr.MAX_SEGMENTS
    _with_fakes(segs, go)


def test_chain_carries_segments_in_the_same_session():
    calls = []
    real = rr.kernel

    def fake(name, rung, pin, climbers=None, segment=0):
        calls.append((name, segment))
        return {"next_segment": 1, "cells": {"K2_s1411": _cut("K2", 1411)}} if (
            name == "g0" and segment == 0) else {"cells": {}}
    rr.kernel = fake
    try:
        done = rr.chain(["g0", "g1"], "M", "a" * 40, M_CLIMB, sleep=lambda s: None)
    finally:
        rr.kernel = real
    assert calls == [("g0", 0), ("g0", 1), ("g1", 0)] and done == {"g0": "home", "g1": "home"}


def test_a_rung_m_cell_refuses_to_run_without_a_carry():
    assert refused(lambda: rr.main(["cell", "--arm", "K2", "--seed", "1411", "--rung", "M",
                                    "--out", "/dev/null"]))


# ---- LEARNER (sbx) ---------------------------------------------------------
def _has_sbx():
    try:
        import sbx  # noqa: F401
        return True
    except ImportError:
        return False


def _flat(model):
    import jax
    import numpy as np
    leaves = jax.tree_util.tree_leaves(model.policy.actor_state.params)
    return np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in leaves])


def _short_spec():
    s = spec()
    s["baseline_settings"]["learning_starts"] = WARM
    for c in s["candidates"]:
        if c["id"] == "K1":
            c["settings"]["param_resets"] = [400]
    return s


_RUNS = {}


def _run(arm, T_=T):
    if (arm, T_) not in _RUNS:
        keep = {}
        keep["hook"] = lambda m: keep.update(init=_flat(m).copy())
        rec = rr.run_arm(arm, SEED, T_, None, spec=_short_spec(), keep=keep)
        keep["final"] = _flat(keep["model"])
        _RUNS[(arm, T_)] = (rec, keep)
    return _RUNS[(arm, T_)]


def test_learner_identity_baseline_is_e2cs_e_food():
    import numpy as np
    from experiments import pilot_b as pb
    from experiments import pilot_b2 as p2
    rec, keep = _run("E-food")
    got = {}
    real_mk, real_warm = pb.make_b, pb.WARMUP

    def mk(venv, *a, **k):
        got["m"] = real_mk(venv, *a, **k)
        return got["m"]
    pb.make_b, pb.WARMUP = mk, WARM
    try:
        ref = p2.run_pilot_cell("E-food", SEED, True, T, eye_frame="body",
                                motor_frame="body", native_substep=True)
    finally:
        pb.make_b, pb.WARMUP = real_mk, real_warm
    assert len(rec["lives"]) >= 1 and rec["lives"] == ref["lives"], "life rows"
    assert rec["trace"] == ref["trace"], "trace rows"
    assert rec["world"] == ref["world"] and rec["handed"] == ref["handed"] == "food"
    assert rec["world"]["food_floor_only"] is True and rec["world"]["native_substep"] is True
    assert rec["world"]["eye_frame"] == "body" and rec["world"]["motor_frame"] == "body"
    assert rec["world"]["lazy_contact_audio"] is True and rec["world"]["difficulty"] == 0.75
    assert np.array_equal(keep["final"], _flat(got["m"])), "actor weights"
    assert not np.array_equal(keep["final"], keep["init"]), "control: the baseline learned"
    assert rec["forge"]["param_resets_done"] == 0, "control for K1"


def test_learner_twin_never_trains():
    import numpy as np
    rec, keep = _run("E-food-lr0")
    assert rec["lr"] == 0.0 and np.array_equal(keep["final"], keep["init"])


def test_learner_k1_reset_is_taken():
    rec, keep = _run("K1")
    assert rec["forge"]["param_resets_done"] == 1
    # a run ends on a whole collection round: 22 x 32 = 704 decisions
    assert keep["model"].replay_buffer.pos == rec["decisions"] == 704, "replay kept through the reset"


def test_learner_k2_is_tqc():
    rec, keep = _run("K2")
    m = keep["model"]
    assert type(m).__name__ == "TQC" and rec["forge"]["algo"] == "TQC"
    assert m.policy.n_target_quantiles == 2 * 25 - 2 * 2
    assert rec["decisions"] == 704 and len(rec["lives"]) >= 1


def test_learner_k5_updates_3_per_32():
    base = _run("E-food")[1]["model"]._n_updates
    k5 = _run("K5")[1]["model"]._n_updates
    assert base > 0 and k5 * 32 == base * 3, (base, k5)


def test_learner_k6_buffer_holds_scaled_reward():
    import numpy as np
    b = _run("E-food")[1]["model"].replay_buffer.rewards[:WARM].ravel()
    k = _run("K6")[1]["model"].replay_buffer.rewards[:WARM].ravel()
    assert np.abs(b).max() > 0, "the warm-up paid some reward"
    assert np.allclose(k, 0.01 * b, rtol=1e-5, atol=1e-9)
    assert not np.allclose(k, b)
    assert _run("K6")[0]["forge"]["learner_reward_scale"] == 0.01
    assert _run("K6")[0]["forge"]["learner_kwargs"]["ent_coef"] == "auto"


def test_learner_n3_and_controls_are_on_the_learner():
    m = _run("N3", 64)[1]["model"]
    assert "NStep" in type(m.replay_buffer).__name__ and m.n_steps == 3
    c2 = _run("C2", 64)[1]["model"]
    assert (c2.batch_size, c2.learning_starts) == (128, 2500)
    assert abs(c2.tau - 0.00588961) < 1e-12
    c1 = _run("C1", 64)[1]["model"]
    assert c1.batch_size == 512 and abs(c1.lr_schedule(1.0) - 0.000198331) < 1e-12


if __name__ == "__main__":
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
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
            print("FAIL", n, repr(e), flush=True)
    print(f"{len(tests) - bad - skipped}/{len(tests)} passed, {skipped} skipped")
    sys.exit(1 if bad else 0)
