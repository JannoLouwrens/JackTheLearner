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
