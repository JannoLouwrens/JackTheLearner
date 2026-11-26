"""Round 1's KERNEL (F3 / ORDERS J18): how one arm of the frozen rule is run.

The rule is `rounds/r1/spec.json` (frozen 594752d0). This file adds no term to
it: every learner setting an arm runs with is READ from the spec
(`arm_settings`), a setting this file does not know how to apply is REFUSED,
and the world is E2c's E-food arm through the same `pilot_b2.pilot_world` call
E2c's kernels made (PILOT_RULE's frames, food_floor_only, native_substep).

One code path for all ten arms (baseline, never-trains twin, K1 K2 K3 K5 K6 N3
C1 C2), `run_arm`: `pilot_b.run_b`'s body with the learner built from the
spec. `experiments/tests/test_forge_r1_run.py` proves the baseline through
here is `pilot_b2.run_pilot_cell("E-food")` life for life and weight for
weight, and that each candidate's change reaches the learner.

What a kernel returns is RAW (`/data/jack-logs/forge/r1/raw/`). Nothing here
writes `results.jsonl`: J18 accepts no result before ORDERS S13 (b)'s
judge-hash tripwire holds, and scoring is the reader's job.

Every kernel of a rung runs ONE commit (`--pin`): the job checks it out after
the launcher's clone, and each record carries the sha it actually ran
(`forge.code_head`), so kernels pushed hours apart cannot run different code.

    python -m experiments.forge.r1_run groups --rung S
    python -m experiments.forge.r1_run chain --rung S --names g0 g2 g4 g6 --pin <sha>
    python -m experiments.forge.r1_run --round r1L groups --rung L   # another frozen round
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
ROUNDS = Path(__file__).resolve().parent / "rounds"
ROUND = "r1"                  # which frozen rule this process runs; `set_round`
SPEC_PATH = ROUNDS / ROUND / "spec.json"
OUT = Path("/data/jack-logs/forge") / ROUND / "raw"
WORLD_ARM = "E-food"          # pilot_b2.E2C_ARMS: arm K + handed 'food', cossin
PER_KERNEL = 4                # E2c's packing: 4 runs on Kaggle's 4 cores
REMOTE_TIMEOUT_S = int(9.5 * 3600)
EST_HOURS = {"S": 4.0, "M": 7.5, "L": 7.5}   # L: every segment runs to SEG_WALL_S
GUARD_RETRIES, GUARD_WAIT_S = 18, 600   # another lane's dirt: wait, never bypass

# ---- J18 step 4: a run that can be cut and carried on ------------------------
# The ONLY change between rungs S and M. A run given a `carry_dir` learns in
# whole CHUNKs; when the next chunk would not fit before `deadline` it saves
# itself through `brain_b.save_life` and returns a `cut` + `resumable` record
# (the locked reader scores `cut` as 0 — a carry nobody finishes is a lost run,
# by the rule). The next kernel loads the life and goes on; the finished record
# is the unbroken run's, life for life (tests/test_forge_r1_carry.py).
# Why chunks and not a stop callback: SB3 drops the transition on which a
# callback returns False (collect_rollouts returns BEFORE _store_transition),
# so a callback stop is not a clean cut. A chunk ends where the unbroken run
# also ends a collection round AND a trace window, so nothing is split.
CHUNK = 20_000                     # = lcm(train_freq 32, pilot_b.TRACE_EVERY 5,000)
SEG_WALL_S = int(7.5 * 3600)       # a kernel's cells save themselves by here
CHUNK_MARGIN = 1.3                 # the next chunk must fit 1.3x the slowest so far
MAX_SEGMENTS = 3                   # kernels one run may take; then it is a lost run
CARRY_NEEDS = ("model.zip", "replay.pkl", "life.pkl", "key.npy", "carry.pkl",
               "carry.json")       # carry.json is written LAST: its presence = whole


class Refused(RuntimeError):
    pass


def set_round(name: str) -> None:
    """Run another FROZEN round with this same kernel (J18, round 1-L): the
    rule, the raw folder, the cell files and the kernel's summary all follow
    the round's name. Only a round whose `rounds/<name>/spec.json` exists and
    calls itself `name` is taken; round r1's paths are unchanged."""
    import re
    global ROUND, SPEC_PATH, OUT, SEGS
    if not re.fullmatch(r"r[0-9]+[A-Za-z]?", str(name)):
        raise Refused(f"r1: {name!r} is not a round name (r<n> or r<n><letter>)")
    path = ROUNDS / name / "spec.json"
    if not path.is_file():
        raise Refused(f"r1: round {name} has no frozen rule at {path}")
    if json.loads(path.read_text()).get("round") != name:
        raise Refused(f"r1: {path} does not call itself round {name}")
    ROUND, SPEC_PATH = name, path
    OUT = Path("/data/jack-logs/forge") / name / "raw"
    SEGS = OUT / "segments"


def summary_file() -> str:
    return f"forge_{ROUND}_cells.json"


def load_spec(path=None) -> dict:
    """The frozen rule. A continuation round (one naming a `parent`) is checked
    by `forge.continuation.validate_round` on every load (ruling 123); round 1's
    rule is read as it always was (its validator is judge-locked)."""
    spec = json.loads(Path(path or SPEC_PATH).read_text())
    if "parent" in spec:
        from experiments.forge import Refused as SpecRefused
        from experiments.forge.continuation import validate_round
        try:
            validate_round(spec)
        except SpecRefused as e:      # one refusal type for the runner's callers
            raise Refused(f"r1: {e}") from e
    return spec


def spec_sha256(path=None) -> str:
    return hashlib.sha256(Path(path or SPEC_PATH).read_bytes()).hexdigest()


def arms(spec: dict) -> list:
    """Baseline, twin(s), then the candidates in the rule's order."""
    return [spec["baseline"], *spec["twins"], *[c["id"] for c in spec["candidates"]]]


# What each rule setting MEANS for the learner. `fixed`: the value is part of
# how every arm runs (run_arm hard-codes it) and the rule must say exactly it.
_FIXED = {"buffer": "the whole run", "norm_obs": True}
_NOTES = ("note", "reset_every")      # prose / the number param_resets derives from
_KNOWN = {"algo", "net_arch", "layer_norm", "lr", "batch_size", "tau", "gamma",
          "learning_starts", "train_freq", "gradient_steps", "ent_coef", "use_sde",
          "n_steps", "top_quantiles_to_drop_per_net", "param_resets",
          "learner_reward_scale"}


def arm_settings(spec: dict, arm: str) -> dict:
    """The rule's settings for one arm: the baseline's, overlaid by the arm's
    own. An unknown arm or an unknown setting is refused, never guessed."""
    st = dict(spec["baseline_settings"])
    if arm == spec["baseline"]:
        over = {}
    elif arm in spec["twins"]:
        over = spec["twin_settings"][arm]
    else:
        cand = {c["id"]: c for c in spec["candidates"]}
        if arm not in cand:
            raise Refused(f"r1: {arm!r} is not an arm of the frozen rule")
        over = cand[arm]["settings"]
    st.update(over)
    for k in _NOTES:
        st.pop(k, None)
    for k, v in _FIXED.items():
        if st.pop(k, v) != v:
            raise Refused(f"r1: arm {arm}: {k} must be {v!r} (run_arm has no other)")
    bad = sorted(set(st) - _KNOWN)
    if bad:
        raise Refused(f"r1: arm {arm}: no knob for setting(s) {bad} — "
                      "a term the kernel cannot meet is an INBOX line")
    return st


def learner_call(st: dict, seed: int, T: int, rung: str | None) -> tuple:
    """(algo name, keyword arguments) for sbx — pure, so a test can read it.
    For the baseline this is `pilot_b.make_b`'s call exactly; a keyword is
    added ONLY when the arm's setting leaves sbx's default."""
    if st["algo"].split()[0] not in ("sbx.SAC", "sbx.TQC"):
        raise Refused(f"r1: unknown algo {st['algo']!r}")
    algo = st["algo"].split()[0].split(".")[1]
    if st["layer_norm"] != "critic":          # sbx: layer_norm=True is critic-only
        raise Refused("r1: layer_norm must be 'critic'")
    pk = dict(layer_norm=True, net_arch=list(st["net_arch"]))
    if "top_quantiles_to_drop_per_net" in st:
        if algo != "TQC":
            raise Refused("r1: quantile truncation needs TQC")
        pk["top_quantiles_to_drop_per_net"] = int(st["top_quantiles_to_drop_per_net"])
    kw = dict(learning_rate=float(st["lr"]), buffer_size=max(int(T), 50_000),
              batch_size=int(st["batch_size"]), tau=float(st["tau"]),
              gamma=float(st["gamma"]), learning_starts=int(st["learning_starts"]),
              train_freq=int(st["train_freq"]), gradient_steps=int(st["gradient_steps"]),
              ent_coef=st["ent_coef"], use_sde=bool(st["use_sde"]), seed=int(seed),
              verbose=0, policy_kwargs=pk)
    if int(st["n_steps"]) != 1:
        kw["n_steps"] = int(st["n_steps"])
    if "param_resets" in st:
        pr = st["param_resets"]
        if isinstance(pr, dict):
            if rung not in pr:
                raise Refused(f"r1: the rule names no reset points for rung {rung!r}")
            pr = pr[rung]
        kw["param_resets"] = [int(x) for x in pr]
    return algo, kw


def _scaled(env, scale: float):
    """K6: the reward AS THE LEARNER SEES IT times `scale`. Sits OUTSIDE
    PilotLives, so the lives, the meals and the metric read the world's own
    reward; only the replay buffer gets the scaled one."""
    import gymnasium as gym

    class LearnerRewardScale(gym.RewardWrapper):
        def reward(self, reward):
            return float(reward) * scale
    return LearnerRewardScale(env)


# His world does not pickle as it stands: it holds module references
# (`w.mujoco`, `w.pg`), jack_env's move-cost closure and the native sub-step's
# cffi handles. `brain_b.save_life`
# pickles the env with plain `pickle.dumps`, so the carried path hands it a
# PilotLives whose `__reduce__` writes the SAME object graph with two extra
# reducers — a module by its name, a bound method by its function and its
# object, a nested function by its code (found again
# in its defining function's constants) plus the values it closed over. No
# world file is edited and nothing is rebuilt from a seed: the loaded life is
# the saved objects.
def _find_code(code, name: str, line: int):
    for c in code.co_consts:
        if hasattr(c, "co_code"):
            if c.co_name == name and c.co_firstlineno == line:
                return c
            got = _find_code(c, name, line)
            if got is not None:
                return got
    return None


def _nested_function(module: str, qualname: str, line: int, cells: tuple, defaults,
                     kwdefaults):
    import importlib
    import types
    mod = importlib.import_module(module)
    parts = qualname.split(".<locals>.")
    owner = mod
    for a in parts[0].split("."):
        owner = getattr(owner, a)
    code = _find_code(owner.__code__, parts[-1], line)
    if code is None or len(code.co_freevars) != len(cells):
        raise Refused(f"r1: the saved life's function {qualname} is not in this code")
    f = types.FunctionType(code, mod.__dict__, parts[-1], defaults,
                           tuple(types.CellType(v) for v in cells))
    f.__qualname__, f.__kwdefaults__ = qualname, kwdefaults
    return f


def _bound(func, obj):
    import types
    return types.MethodType(func, obj)


def _life_dumps(obj) -> bytes:
    import importlib
    import io
    import pickle
    import types

    class P(pickle.Pickler):
        def reducer_override(self, o):
            if isinstance(o, types.ModuleType):
                return importlib.import_module, (o.__name__,)
            if isinstance(o, types.MethodType):
                # plain pickle looks a bound method up by its function's NAME on
                # the object; the world's patched-in methods (`synth.step` is
                # `_lazy_step`, `w.decide` is `_native_decide`) are not there
                return _bound, (o.__func__, o.__self__)
            if isinstance(o, types.FunctionType) and "<locals>" in o.__qualname__:
                return _nested_function, (
                    o.__module__, o.__qualname__, o.__code__.co_firstlineno,
                    tuple(c.cell_contents for c in (o.__closure__ or ())),
                    o.__defaults__, o.__kwdefaults__)
            return NotImplemented
    buf = io.BytesIO()
    P(buf, protocol=pickle.HIGHEST_PROTOCOL).dump(obj)
    return buf.getvalue()


def _life_loads(blob: bytes, native: bool):
    import pickle
    w = carried_lives_cls().__new__(carried_lives_cls())
    w.__dict__.update(pickle.loads(blob))
    if native:                      # the C twins, rebuilt on the LOADED world
        from experiments.native_substep import apply_native_substep
        apply_native_substep(w.unwrapped.w)
    return w


_CARRIED_CLS = None


def carried_lives_cls():
    """pilot_b.PilotLives, picklable. Same methods, same attributes."""
    global _CARRIED_CLS
    if _CARRIED_CLS is None:
        from experiments import pilot_b as pb

        class CarriedLives(pb.PilotLives):
            def __reduce__(self):
                # The native sub-step's C twins (cffi handles) cannot cross a
                # save and need not: `_native_decide` loads them from the
                # Python objects at the top of every decision and writes them
                # back at the bottom, so between decisions they hold nothing.
                world = self.unwrapped.w
                native = world.__dict__.pop("_native", None)
                try:
                    blob = _life_dumps(dict(self.__dict__))
                finally:
                    if native is not None:
                        world._native = native
                return _life_loads, (blob, native is not None)
        CarriedLives.__qualname__ = "CarriedLives"
        _CARRIED_CLS = CarriedLives
    return _CARRIED_CLS


def carry_name(arm: str, seed: int) -> str:
    return f"carry_{arm}_s{seed}"


def carry_whole(d) -> bool:
    return d is not None and all((Path(d) / n).is_file() for n in CARRY_NEEDS)


def chunk_rule(T: int, chunk: int, train_freq: int, trace_every: int) -> None:
    """A carried run may only be cut where the unbroken run is whole."""
    if chunk <= 0 or chunk % train_freq or chunk % trace_every or T % chunk:
        raise Refused(f"r1: a carried run is cut on whole chunks only: chunk {chunk} must "
                      f"be a multiple of train_freq {train_freq} and of the trace window "
                      f"{trace_every}, and divide T {T}")


def _carry_save(d, model, venv, tr, meta: dict, plant=()) -> None:
    """save_life + what it does not hold (the policy's exploration-noise keys,
    the observation normaliser, the trace so far, the clocks). Written beside `d` and renamed, carry.json last."""
    import pickle
    import shutil
    from experiments.brain_b import save_life
    d = Path(d)
    tmp = d.with_name(d.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    save_life(model, str(tmp))
    import numpy as np
    from experiments.brain_b import _key_data
    pol = model.policy
    with open(tmp / "carry.pkl", "wb") as f:
        pickle.dump({"policy_keys": (None if "no_sde_key" in plant else
                                     {k: np.asarray(_key_data(getattr(pol, k)))
                                      for k in ("key", "noise_key")}),
                     "obs_rms": None if "no_norm" in plant else venv.obs_rms,
                     "ret_rms": venv.ret_rms, "returns": venv.returns,
                     "trace": None if "no_trace" in plant else list(tr)}, f)
    (tmp / "carry.json").write_text(json.dumps(meta) + "\n")
    shutil.rmtree(d, ignore_errors=True)
    tmp.rename(d)


def _carry_load(d, venv, algo_cls, want: dict, plant=()):
    """-> (model, trace so far, meta). Refuses a carry that is not whole or is
    another run's (arm, seed, T, rung, the rule, the settings)."""
    import pickle
    from experiments.brain_b import resume_life
    d = Path(d)
    if not carry_whole(d):
        raise Refused(f"r1: {d} is not a whole carry {CARRY_NEEDS}")
    meta = json.loads((d / "carry.json").read_text())
    bad = [k for k, v in want.items() if meta.get(k) != v]
    if bad:
        raise Refused(f"r1: the carry at {d} is another run's: {bad} differ")
    model = resume_life(str(d), venv, algo=algo_cls)
    with open(d / "carry.pkl", "rb") as f:
        c = pickle.load(f)
    # sbx pickles `policy` AND the alias `actor` as separate copies, so after a
    # load `model.actor.reset_noise` belongs to a policy nobody acts with: with
    # use_sde the exploration noise would never be drawn again (SB3 resets it
    # through `self.actor` at each collection round). A fresh model has
    # `actor is policy.actor`; put that back.
    if "no_rewire" not in plant:
        model.actor, model.qf = model.policy.actor, model.policy.qf
        if getattr(model.actor.reset_noise, "__self__", None) is not model.policy:
            raise Refused("r1: the loaded learner's noise reset is not its policy's")
    # SB3's load() ends with `policy.reset_noise()` when use_sde is on: one
    # extra split of the policy's key, so the loaded run would draw the NEXT
    # round's noise. The keys as they were at the save replace it.
    if c["policy_keys"] is not None:
        import jax.numpy as jnp
        for k, v in c["policy_keys"].items():
            setattr(model.policy, k, jnp.asarray(v))
    # resume_life's env.reset() fed one observation to the FRESH normaliser;
    # the carried statistics replace it whole.
    if c["obs_rms"] is not None:
        venv.obs_rms = c["obs_rms"]
    venv.ret_rms, venv.returns = c["ret_rms"], c["returns"]
    return model, list(c["trace"] or []), meta


def run_arm(arm: str, seed: int, T: int, rung: str | None = None, *,
            partial_path=None, spec: dict | None = None, keep: dict | None = None,
            carry_dir=None, resume_from=None, deadline: float | None = None,
            chunk: int = CHUNK, max_chunks: int | None = None, plant=()) -> dict:
    """One run of `arm` for T decisions on E-food's world. `rung` names which
    of the rule's rungs this is (None = a smoke/test run, stamped as such).
    `keep` (tests): receives the model and the env.

    `carry_dir` None (rung S, every test before J18 step 4): ONE learn call,
    the code rung S ran. Given: `_run_carried` — the same learner in whole
    chunks, loaded from `resume_from` (or `carry_dir`) if a whole carry is
    there, saved to `carry_dir` when `deadline` (epoch seconds) is near.
    `max_chunks` / `plant` exist for the tests only."""
    if carry_dir is not None:
        return _run_carried(arm, seed, T, rung, partial_path=partial_path, spec=spec,
                            keep=keep, carry_dir=carry_dir, resume_from=resume_from,
                            deadline=deadline, chunk=chunk, max_chunks=max_chunks,
                            plant=plant)
    from experiments import pilot_b as pb
    from experiments.pilot_b2 import E2C_ENCODING, PILOT_RULE, arm_cfg, pilot_world
    spec_given = spec is not None
    spec = load_spec() if spec is None else spec
    st = arm_settings(spec, arm)
    algo, kw = learner_call(st, seed, T, rung)
    scale = float(st.get("learner_reward_scale", 1.0))
    pb._single_thread()
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import sbx
    world = pilot_world(WORLD_ARM, PILOT_RULE["food_floor_only"], PILOT_RULE["eye_frame"],
                        PILOT_RULE["motor_frame"], PILOT_RULE["native_substep"])
    holder = {}

    def _mk():
        w = pb.PilotLives(world(seed), partial_path=partial_path)
        holder["w"] = w
        return w if scale == 1.0 else _scaled(w, scale)

    venv = VecNormalize(DummyVecEnv([_mk]), norm_obs=True, norm_reward=False,
                        clip_obs=10.0)
    venv.seed(seed)
    model = getattr(sbx, algo)("MlpPolicy", venv, **kw)
    if keep is not None:
        keep.update(model=model, env=holder["w"], venv=venv)
        keep.get("hook", lambda m: None)(model)
    tr = []
    t_wall, t_cpu = time.time(), time.process_time()
    model.learn(total_timesteps=int(T), progress_bar=False,
                callback=pb._trace_callback(tr, pb.TRACE_EVERY))
    wall, cpu = time.time() - t_wall, time.process_time() - t_cpu
    w = holder["w"]
    lives = list(w.lives)
    if w.n > 0:
        lives.append(w._life_row({"ate_total": dict(w.env.w.drives.ate_total)},
                                 "unfinished"))
    return _record(arm, seed, T, rung, spec, spec_given, st, algo, kw, scale, model, w,
                   lives, tr, wall, cpu)


def _record(arm, seed, T, rung, spec, spec_given, st, algo, kw, scale, model, w, lives,
            tr, wall, cpu) -> dict:
    """The raw record of a finished run — one writer for both paths."""
    from experiments import pilot_b as pb
    from experiments.pilot_b2 import E2C_ENCODING, PILOT_RULE, arm_cfg
    resets = getattr(model, "reset_idx", 0)
    return {"arm": arm, "seed": int(seed), "T": int(T), "lr": kw["learning_rate"],
            "decisions": int(model.num_timesteps), "wall_s": round(wall, 1),
            "cpu_s": round(cpu, 1),
            "decisions_per_s": round(model.num_timesteps / wall, 2),
            "decisions_per_cpu_s": round(model.num_timesteps / cpu, 2),
            "world": pb._world_stamp(w.env), "trace": tr, "trace_every": pb.TRACE_EVERY,
            "pilot_arm": WORLD_ARM, "handed": arm_cfg(WORLD_ARM)["handed"],
            "handed_encoding": E2C_ENCODING,
            "food_floor_only": bool(PILOT_RULE["food_floor_only"]),
            "eye_frame": PILOT_RULE["eye_frame"], "motor_frame": PILOT_RULE["motor_frame"],
            "forge": {"round": spec["round"], "arm": arm, "rung": rung,
                      "rung_decisions": (spec["rungs"][rung]["decisions"]
                                         if rung in spec["rungs"] else None),
                      "spec_sha256": None if spec_given else spec_sha256(),
                      "algo": algo, "settings": st,
                      "learner_kwargs": {k: v for k, v in kw.items()},
                      "learner_reward_scale": scale, "param_resets_done": int(resets),
                      "code_head": _code_head()},
            "lives": lives}


def _run_carried(arm, seed, T, rung, *, partial_path, spec, keep, carry_dir, resume_from,
                 deadline, chunk, max_chunks, plant) -> dict:
    """run_arm's body, learning in whole chunks so it can be cut and carried."""
    from experiments import pilot_b as pb
    from experiments.pilot_b2 import PILOT_RULE, pilot_world
    spec_given = spec is not None
    spec = load_spec() if spec is None else spec
    st = arm_settings(spec, arm)
    algo, kw = learner_call(st, seed, T, rung)
    scale = float(st.get("learner_reward_scale", 1.0))
    if scale != 1.0:
        raise Refused("r1: a learner-side reward wrapper is not carried (K6 did not climb)")
    chunk_rule(int(T), int(chunk), kw["train_freq"], pb.TRACE_EVERY)
    pb._single_thread()
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import sbx
    world = pilot_world(WORLD_ARM, PILOT_RULE["food_floor_only"], PILOT_RULE["eye_frame"],
                        PILOT_RULE["motor_frame"], PILOT_RULE["native_substep"])
    src = next((d for d in (resume_from, carry_dir) if carry_whole(d)), None)
    venv = VecNormalize(DummyVecEnv([lambda: carried_lives_cls()(
                            world(seed), partial_path=partial_path)]),
                        norm_obs=True, norm_reward=False, clip_obs=10.0)
    want = {"arm": arm, "seed": int(seed), "T": int(T), "rung": rung, "settings": st,
            "spec_sha256": None if spec_given else spec_sha256(), "chunk": int(chunk)}
    if src is None:
        venv.seed(seed)
        model = getattr(sbx, algo)("MlpPolicy", venv, **kw)
        tr, meta = [], dict(want, wall_s=0.0, cpu_s=0.0, segments=[])
    else:
        model, tr, meta = _carry_load(src, venv, getattr(sbx, algo), want, plant=plant)
        if int(model.num_timesteps) != meta["decisions"] or meta["decisions"] % chunk:
            raise Refused(f"r1: the carry at {src} does not sit on a chunk boundary")
    # sbx adds to policy_kwargs in place when it builds; a loaded learner was
    # not built here, so the record takes the learner's own (the same dict
    # object in an unbroken run).
    kw = dict(kw, policy_kwargs=dict(model.policy_kwargs))
    w = model.env.envs[0]                       # the carried life, or the fresh one
    w.partial_path = None if partial_path is None else Path(partial_path)
    if keep is not None:
        keep.update(model=model, env=w, venv=venv)
        keep.get("hook", lambda m: None)(model)
    start, slowest, n_chunks = int(model.num_timesteps), 0.0, 0
    t_wall, t_cpu = time.time(), time.process_time()
    while model.num_timesteps < T:
        if max_chunks is not None and n_chunks >= max_chunks:
            break
        if (deadline is not None and n_chunks > 0
                and time.time() + CHUNK_MARGIN * slowest > deadline):
            break
        if deadline is not None and n_chunks == 0 and time.time() >= deadline:
            break
        t0 = time.time()
        # SB3 reads total_timesteps as an increment when reset_num_timesteps=False
        model.learn(total_timesteps=int(chunk), progress_bar=False,
                    reset_num_timesteps=model.num_timesteps == 0,
                    callback=pb._trace_callback(tr, pb.TRACE_EVERY))
        slowest, n_chunks = max(slowest, time.time() - t0), n_chunks + 1
        if model.num_timesteps % chunk:
            raise RuntimeError(f"r1: a chunk ended off its boundary at {model.num_timesteps}")
    wall = meta["wall_s"] + time.time() - t_wall
    cpu = meta["cpu_s"] + time.process_time() - t_cpu
    segs = meta["segments"] + [{"start": start, "end": int(model.num_timesteps),
                                "code_head": _code_head(), "resumed_from": src and str(src)}]
    lives = list(w.lives)
    if model.num_timesteps < T:                 # cut: save the life, return what the
        _carry_save(carry_dir, model, venv, tr,  # reader will count as a lost run
                    dict(want, decisions=int(model.num_timesteps), wall_s=wall, cpu_s=cpu,
                         segments=segs), plant=plant)
        return {"arm": arm, "seed": int(seed), "T": int(T), "cut": True, "resumable": True,
                "decisions": int(model.num_timesteps),
                "decisions_reached": int(model.num_timesteps), "lives": lives,
                "carry": carry_name(arm, seed), "wall_s": round(wall, 1),
                "forge": {"round": spec["round"], "arm": arm, "rung": rung,
                          "segments": segs, "code_head": _code_head()}}
    if w.n > 0:
        lives.append(w._life_row({"ate_total": dict(w.env.w.drives.ate_total)},
                                 "unfinished"))
    rec = _record(arm, seed, T, rung, spec, spec_given, st, algo, kw, scale, model, w,
                  lives, tr, wall, cpu)
    rec["forge"]["segments"] = segs
    return rec


def _code_head() -> str:
    p = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else ""


# ---- packing ---------------------------------------------------------------
def tranches(spec: dict, rung: str) -> dict | None:
    """A rung that names its own entrants per tranche (J18 round 1-L, ruling
    122): `rungs[rung]["tranches"] = {name: [{"arms": [...], "seeds": [...]},
    ...]}`. -> {name: [(arm, seed), ...]} in the rule's order (each block
    SEED-major), or None for a rung without tranches. Refused: an arm not of
    the rule, a seed not of the rung, a cell in two tranches, a bad name."""
    import re
    tr = spec["rungs"][rung].get("tranches")
    if tr is None:
        return None
    if not isinstance(tr, dict) or not tr:
        raise Refused(f"r1: rung {rung}: tranches must name at least one tranche")
    rung_seeds, known, seen, out = set(spec["rungs"][rung]["seeds"]), set(arms(spec)), set(), {}
    for name, blocks in tr.items():
        if not re.fullmatch(r"[0-9A-Za-z]+", str(name)):
            raise Refused(f"r1: tranche name {name!r} (letters and digits only)")
        if not isinstance(blocks, list) or not blocks:
            raise Refused(f"r1: tranche {name}: no blocks")
        out[name] = []
        for b in blocks:
            bad_a = [a for a in b.get("arms", []) if a not in known]
            bad_s = [x for x in b.get("seeds", []) if x not in rung_seeds]
            if not b.get("arms") or not b.get("seeds") or bad_a or bad_s:
                raise Refused(f"r1: tranche {name}: block {b} (arms not of the rule "
                              f"{bad_a}, seeds not of rung {rung} {bad_s})")
            for sd in b["seeds"]:
                for a in b["arms"]:
                    if (a, sd) in seen:
                        raise Refused(f"r1: cell ({a}, {sd}) is in two tranches/blocks")
                    seen.add((a, sd))
                    out[name].append((a, sd))
    return out


def cells(spec: dict, rung: str, climbers: list | None = None) -> list:
    """(arm, seed) for a rung, SEED-major so no arm sits wholly on one slow
    host. S: all ten arms. Above S the entrants are the ladder's to name
    (`climbers` = the arm ids experiments.forge.climb let through, plus the
    baseline, twin and yardstick) — there is no default. A rung with
    `tranches` names its own entrants per arm and seed: every tranche's cells
    in the rule's order, and `climbers` is refused (two sources of entrants)."""
    tr = tranches(spec, rung)
    if tr is not None:
        if climbers is not None:
            raise Refused(f"r1: rung {rung} names its entrants in its tranches")
        return [c for v in tr.values() for c in v]
    if rung == "S":
        if climbers is not None:
            raise Refused("r1: rung S runs every arm of the rule")
        who = arms(spec)
    else:
        if not climbers:
            raise Refused(f"r1: rung {rung} needs the ladder's entrants")
        unknown = [a for a in climbers if a not in arms(spec)]
        if unknown:
            raise Refused(f"r1: not arms of the rule: {unknown}")
        who = list(climbers)
    return [(a, s) for s in spec["rungs"][rung]["seeds"] for a in who]


def groups(spec: dict, rung: str, climbers: list | None = None) -> dict:
    """Kernels of PER_KERNEL runs. A tranched rung packs each tranche on its
    own (no kernel spans two tranches: tranche 2 may never run) and names the
    groups g<tranche>-<j>, so tranche 2's raw files never collide with
    tranche 1's and still match the reader's `{rung}_g*.json`."""
    n = PER_KERNEL
    tr = tranches(spec, rung)
    if tr is not None:
        if climbers is not None:
            raise Refused(f"r1: rung {rung} names its entrants in its tranches")
        return {f"g{t}-{j}": c[n * j:n * (j + 1)]
                for t, c in tr.items() for j in range((len(c) + n - 1) // n)}
    c = cells(spec, rung, climbers)
    return {f"g{j}": c[n * j:n * (j + 1)] for j in range((len(c) + n - 1) // n)}


def cell_file(arm: str, seed: int) -> str:
    return f"{ROUND}_{arm}_s{seed}.json"


def find_carry(root, arm: str, seed: int) -> Path:
    """The ONE whole carry for (arm, seed) under `root` (a previous segment's
    kernel output, mounted by Kaggle at a depth never promised — J10 11-19).
    None found or two found -> Refused: a continuation never starts a run
    fresh (J10-2's require_carry, here for the forge)."""
    name = carry_name(arm, seed)
    hits = sorted(Path(d) for d, _dirs, _f in os.walk(root)
                  if Path(d).name == name and carry_whole(d))
    if len(hits) != 1:
        raise Refused(f"r1: {len(hits)} whole carries named {name} under {root}: "
                      f"{[str(h) for h in hits]}")
    return hits[0]


def run_parallel(cell_list: list, outdir, rung: str, carry_root=None, resume_root=None,
                 deadline: float | None = None) -> dict:
    """All cells at once as child processes (pilot_b2.run_pilot_parallel's
    shape): own file, partial file every 10 lives, a timed-out cell is `cut`
    with the lives it finished.

    `carry_root` (rung M on, J18 step 4): each cell carries in
    `carry_root/carry_<arm>_s<seed>` and saves itself before `deadline`.
    `resume_root`: a continuation — every cell's carry is found there FIRST,
    and one missing carry refuses the whole kernel before any cell runs."""
    from experiments.pilot_b import CHILD_TIMEOUT_S, from_partial, partial_file
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if resume_root is not None and carry_root is None:
        raise Refused("r1: a continuation needs somewhere to carry to")
    src = ({c: find_carry(resume_root, *c) for c in map(tuple, cell_list)}
           if resume_root is not None else {})
    procs = []
    for arm, seed in cell_list:
        f = outdir / cell_file(arm, seed)
        argv = [sys.executable, "-m", "experiments.forge.r1_run", "cell", "--round", ROUND,
                "--arm", arm,
                "--seed", str(seed), "--rung", rung, "--out", str(f),
                "--partial", str(partial_file(f))]
        if carry_root is not None:
            argv += ["--carry-dir", str(Path(carry_root) / carry_name(arm, seed))]
            if deadline is not None:
                argv += ["--deadline", repr(float(deadline))]
            if (arm, seed) in src:
                argv += ["--resume-from", str(src[(arm, seed)])]
        procs.append((arm, seed, f, subprocess.Popen(
            argv, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)))
    res, t0 = {}, time.time()
    for arm, seed, f, p in procs:
        try:
            _, err = p.communicate(timeout=max(1, CHILD_TIMEOUT_S - (time.time() - t0)))
        except subprocess.TimeoutExpired:
            p.kill()
            _, err = p.communicate()
        rec = (json.loads(f.read_text()) if f.exists()
               else from_partial(arm, seed, partial_file(f)))
        rec["rc"] = p.returncode
        if p.returncode != 0:
            rec["error"] = f"rc={p.returncode}: {err.decode()[-800:]}"
        res[f"{arm}_s{seed}"] = rec
        print(f"{arm} s{seed}: rc {p.returncode}, {len(rec['lives'])} lives", flush=True)
    return res


def cells_from_artifacts(artifacts: dict, cell_list: list) -> dict | None:
    """A kernel with no summary file: assemble from the per-cell files, final
    first, else the partial (marked `cut`), else `not_found` — never invented."""
    from experiments.pilot_b import from_partial, partial_file
    res, found = {}, 0
    for arm, seed in cell_list:
        final = cell_file(arm, seed)
        part = partial_file(Path(final)).name
        if final in artifacts:
            rec = json.loads(Path(artifacts[final]).read_text())
            found += 1
        elif part in artifacts:
            rec = from_partial(arm, seed, Path(artifacts[part]))
            found += 1
        else:
            rec = from_partial(arm, seed, Path("/nonexistent") / part)
        rec["from_cell_file"] = True
        res[f"{arm}_s{seed}"] = rec
    return {"cells": res, "assembled": True} if found else None


_JOB = r"""
import subprocess as _s, sys as _y, time as _t
_T0 = _t.time()
_PIN = __PIN__
if _s.run(["git", "-C", "/tmp/jack", "checkout", "-q", _PIN]).returncode != 0:
    _s.run(["git", "-C", "/tmp/jack", "fetch", "-q", "--depth", "1", "origin", _PIN], check=True)
    _s.run(["git", "-C", "/tmp/jack", "checkout", "-q", _PIN], check=True)
_got = _s.run(["git", "-C", "/tmp/jack", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
assert _got == _PIN, ("round pin not checked out", _got, _PIN)
print("FORGE_PIN", _got, flush=True)
_s.run([_y.executable, "-m", "pip", "install", "-q"] + __PINS__, check=True)
import json, os
os.chdir("/tmp/jack")
from experiments.forge.r1_run import run_parallel, set_round, spec_sha256, summary_file
set_round(__ROUND__)
assert spec_sha256() == __SPEC__, "the kernel's spec.json is not the frozen rule"
from experiments.brain_b import versions
__RUN__
json.dump({"cells": r, "versions": versions(), "round": __ROUND__, "rung": __RUNG__,
           "pin": _PIN, "spec_sha256": __SPEC__, "segment": __SEGMENT__},
          open(os.path.join(os.environ["JACK_OUT"], summary_file()), "w"))
print("DONE", flush=True)
"""


_RUN_PLAIN = 'r = run_parallel(__CELLS__, os.environ["JACK_OUT"], __RUNG__)'
# rung M on: carries live in the kernel's OUTPUT (the next segment mounts this
# kernel); the deadline counts from the job's first line, pip install included.
_RUN_CARRY = ('from experiments.forge.r1_run import SEG_WALL_S\n'
              'r = run_parallel(__CELLS__, os.environ["JACK_OUT"], __RUNG__,\n'
              '                 carry_root=os.path.join(os.environ["JACK_OUT"], "carry"),\n'
              '                 resume_root=__RESUME__, deadline=_T0 + SEG_WALL_S)')


def carries(rung: str) -> bool:
    """Rung S ran unbroken (one learn call); every rung above it carries."""
    return rung != "S"


def job_body(cell_list: list, rung: str, pin: str, segment: int = 0) -> str:
    from experiments.pilot_b import KERNEL_PINS
    if len(pin) != 40 or any(c not in "0123456789abcdef" for c in pin):
        raise Refused("r1: --pin must be a full 40-hex commit sha")
    if not 0 <= int(segment) < MAX_SEGMENTS or (segment and not carries(rung)):
        raise Refused(f"r1: segment {segment} of rung {rung} is not allowed "
                      f"(rung S never carries; at most {MAX_SEGMENTS} kernels a run)")
    run = (_RUN_CARRY.replace("__RESUME__", repr("/kaggle/input" if segment else None))
           if carries(rung) else _RUN_PLAIN)
    return (_JOB.replace("__RUN__", run).replace("__SEGMENT__", repr(int(segment)))
            .replace("__CELLS__", repr([tuple(c) for c in cell_list]))
            .replace("__PINS__", repr(list(KERNEL_PINS)))
            .replace("__RUNG__", repr(str(rung))).replace("__PIN__", repr(pin))
            .replace("__ROUND__", repr(ROUND))
            .replace("__SPEC__", repr(spec_sha256())))


def pin_has_rule(pin: str) -> bool:
    """The pinned commit must hold THIS frozen spec and this kernel file."""
    p = subprocess.run(["git", "-C", str(REPO), "show", f"{pin}:experiments/forge/rounds/{ROUND}/spec.json"],
                       capture_output=True)
    q = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e",
                        f"{pin}:experiments/forge/r1_run.py"], capture_output=True)
    return (p.returncode == 0 and q.returncode == 0
            and hashlib.sha256(p.stdout).hexdigest() == spec_sha256())


SEGS = OUT / "segments"   # set_round moves it too; NOT globbed by the reader (`{rung}_g*.json` in OUT only)


def seg_file(rung: str, name: str, segment: int) -> Path:
    return SEGS / f"{rung}_{name}_seg{segment}.json"


def resumable(payload: dict) -> list:
    """(arm, seed) of the runs a segment saved to carry on."""
    return [(r["arm"], r["seed"]) for r in payload["cells"].values()
            if r.get("resumable") and r.get("cut")]


def assemble(segs: list) -> dict:
    """Segment payloads (0, 1, ...) of one group -> the ONE raw file the locked
    reader reads: each run's record from the last segment that ran it, cost =
    the SUM of the segments' kernel durations (every run of the group is billed
    the whole chain — never less than it held). Refuses segments that do not
    share the round, rung, spec and pin, or that run a cell twice in one."""
    first = segs[0]
    for i, p in enumerate(segs):
        for k in ("round", "rung", "pin", "spec_sha256"):
            if p.get(k) != first.get(k):
                raise Refused(f"r1: segment {i} differs on {k}: {p.get(k)} vs {first.get(k)}")
        if p.get("segment", 0) != i:
            raise Refused(f"r1: segment {i} says it is segment {p.get('segment')}")
        if i and set(p["cells"]) != {f"{a}_s{sd}" for a, sd in resumable(segs[i - 1])}:
            raise Refused(f"r1: segment {i} ran other runs than segment {i - 1} carried")
    cells = {}
    for p in segs:
        cells.update(p["cells"])
    durs = [(p.get("kernel") or {}).get("duration_s") for p in segs]
    dur = sum(durs) if all(isinstance(d, (int, float)) for d in durs) else None
    k0 = first.get("kernel") or {}
    return {"cells": cells, "versions": first.get("versions"), "round": first["round"],
            "rung": first["rung"], "pin": first["pin"], "spec_sha256": first["spec_sha256"],
            "segments": len(segs),
            "kernel": {"job_id": [(p.get("kernel") or {}).get("job_id") for p in segs],
                       "duration_s": dur,
                       "segment_duration_s": durs,
                       "charge_seconds": [(p.get("kernel") or {}).get("charge_seconds")
                                          for p in segs],
                       "name": k0.get("name"), "pin": first["pin"]}}


def kernel(name: str, rung: str, pin: str, climbers: list | None = None,
           segment: int = 0) -> dict:
    """Submit one group as a Kaggle CPU kernel -> OUT/<rung>_<name>.json (raw).

    Rung S (never carries): one kernel, its payload IS the raw file.
    Rung M on (J18 step 4): segment n's payload goes to SEGS; segment n > 0
    mounts segment n-1's kernel and runs ONLY the runs it carried. The raw
    file is assembled when no run is left to carry or the last allowed segment
    is home — never before, so the reader sees each run once. Only the summary
    and the per-cell files are downloaded, never the carries (replay buffers)."""
    from experiments.gpu import build_job, result_json, submit
    from experiments.pilot_b import partial_file
    spec = load_spec()
    cell_list = groups(spec, rung, climbers)[name]
    if not pin_has_rule(pin):
        raise Refused(f"r1: commit {pin[:10]} does not hold the frozen rule + kernel")
    out = OUT / f"{rung}_{name}.json"
    if out.exists():
        raise Refused(f"r1: {out} is already home — a rung's kernel runs once")
    sources, prior = None, []
    if carries(rung):
        if seg_file(rung, name, segment).exists():
            raise Refused(f"r1: {seg_file(rung, name, segment)} is already home")
        prior = [json.loads(seg_file(rung, name, i).read_text()) for i in range(segment)
                 if seg_file(rung, name, i).exists()]
        if len(prior) != segment:
            raise Refused(f"r1: segment {segment} of {rung}/{name} needs segments "
                          f"0..{segment - 1} home")
        if segment:
            if prior[-1].get("pin") != pin:
                raise Refused("r1: a continuation runs the pin its run started on")
            cell_list = resumable(prior[-1])
            if not cell_list:
                raise Refused(f"r1: segment {segment - 1} of {rung}/{name} carried nothing")
            job = (prior[-1].get("kernel") or {}).get("job_id")
            if not job or "/" not in str(job):
                raise Refused(f"r1: segment {segment - 1} names no kernel to mount")
            sources = [job]
    elif segment:
        raise Refused("r1: rung S never carries")
    body = job_body(cell_list, rung, pin, segment)
    fetch = [summary_file()]
    if carries(rung):
        fetch += [n for a, sd in cell_list
                  for n in (cell_file(a, sd), partial_file(Path(cell_file(a, sd))).name)]
    sub = submit(build_job(body), prefer="kaggle-cpu", est_hours=EST_HOURS[rung],
                 timeout_s=REMOTE_TIMEOUT_S, fetch=fetch, kernel_sources=sources,
                 fetch_only=carries(rung))
    if sub.ok and summary_file() in sub.artifacts:
        r = result_json(sub, summary_file())
    else:
        r = cells_from_artifacts(sub.artifacts, cell_list)
        if r is None:
            raise RuntimeError(f"forge r1 kernel {rung}/{name}: {sub.message}")
        r.update(round=spec["round"], rung=rung, pin=pin, spec_sha256=spec_sha256(),
                 segment=int(segment))
        print(f"forge r1 kernel {rung}/{name}: no summary ({sub.message or 'missing'}); "
              "assembled from per-cell files", flush=True)
    # what the scoreboard's cost line needs, read off the launcher's receipt
    r["kernel"] = {"job_id": sub.job_id, "duration_s": sub.duration_s,
                   "charge_seconds": sub.charge_seconds, "name": name, "pin": pin}
    OUT.mkdir(parents=True, exist_ok=True)
    for k, rec in r["cells"].items():
        print(f"{k}: {len(rec['lives'])} lives" + (" (cut)" if rec.get("cut") else "")
              + (" (carried on)" if rec.get("resumable") else "")
              + (f" ERROR {rec['error']}" if "error" in rec else ""), flush=True)
    if not carries(rung):
        out.write_text(json.dumps(r, indent=1) + "\n")
        return r
    SEGS.mkdir(parents=True, exist_ok=True)
    seg_file(rung, name, segment).write_text(json.dumps(r, indent=1) + "\n")
    left = resumable(r)
    if left and segment + 1 < MAX_SEGMENTS:
        r["next_segment"] = segment + 1
        return r
    raw = assemble(prior + [r])
    out.write_text(json.dumps(raw, indent=1) + "\n")
    return raw


def chain(names: list, rung: str, pin: str, climbers: list | None = None,
          sleep=time.sleep) -> dict:
    """Kernels one after another = ONE Kaggle session held. A kernel that fails
    is logged and the chain goes on (its cells stay missing — a missing run is
    the reader's to count, never refilled silently). The launcher's guard
    (another lane's uncommitted code, an unpushed HEAD) is waited out, not
    bypassed."""
    done = {}
    for name in names:
        seg = 0
        for attempt in range(GUARD_RETRIES * MAX_SEGMENTS + MAX_SEGMENTS):
            try:
                r = kernel(name, rung, pin, climbers, segment=seg) if seg else \
                    kernel(name, rung, pin, climbers)
                if isinstance(r, dict) and r.get("next_segment"):
                    seg = r["next_segment"]         # same session: carry on at once
                    print(f"{name}: segment {seg - 1} home, carrying "
                          f"{len(resumable(r))} runs on", flush=True)
                    continue
                done[name] = "home"
                break
            except Refused as e:
                done[name] = f"refused: {e}"
                break
            except RuntimeError as e:
                guard = ("Uncommitted changes" in str(e) or "Untracked experiment" in str(e)
                         or "is not on origin" in str(e))
                if guard and attempt < GUARD_RETRIES * MAX_SEGMENTS:
                    print(f"{name}: launcher guard ({str(e).splitlines()[0]}); "
                          f"waiting {GUARD_WAIT_S}s [{attempt + 1}/{GUARD_RETRIES}]", flush=True)
                    sleep(GUARD_WAIT_S)
                    continue
                done[name] = f"failed: {str(e)[:300]}"
                break
        print(f"CHAIN {rung}/{name}: {done[name]}", flush=True)
    print("CHAIN DONE", json.dumps(done), flush=True)
    return done


def main(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["cell", "groups", "kernel", "chain"])
    p.add_argument("--arm")
    p.add_argument("--seed", type=int)
    p.add_argument("--rung", default="S")
    p.add_argument("--round", default="r1", help="the frozen rule: rounds/<round>/spec.json")
    p.add_argument("--decisions", type=int, default=None,
                   help="cell: a smoke length; stamped rung=None unless it is the rung's")
    p.add_argument("--out")
    p.add_argument("--partial", default=None)
    p.add_argument("--name")
    p.add_argument("--names", nargs="+")
    p.add_argument("--pin")
    p.add_argument("--climbers", nargs="+", default=None)
    p.add_argument("--carry-dir", default=None)
    p.add_argument("--resume-from", default=None)
    p.add_argument("--deadline", type=float, default=None)
    p.add_argument("--segment", type=int, default=0)
    a = p.parse_args(argv)
    if a.round != ROUND:
        set_round(a.round)
    spec = load_spec()
    if a.cmd == "cell":
        T = spec["rungs"][a.rung]["decisions"]
        rung = a.rung
        if a.decisions is not None and a.decisions != T:
            if any(a.seed in r["seeds"] for r in spec["rungs"].values()):
                raise Refused("r1: a smoke run never uses a rung's seed")
            T, rung = a.decisions, None
        elif a.seed not in spec["rungs"][a.rung]["seeds"]:
            raise Refused(f"r1: seed {a.seed} is not one of rung {a.rung}'s")
        elif tranches(spec, a.rung) is not None and (a.arm, a.seed) not in cells(spec, a.rung):
            raise Refused(f"r1: ({a.arm}, {a.seed}) is in none of rung {a.rung}'s tranches")
        if rung is not None and carries(rung) and a.carry_dir is None:
            raise Refused(f"r1: rung {rung} runs carry (J18 step 4): --carry-dir")
        rec = run_arm(a.arm, a.seed, T, rung, partial_path=a.partial,
                      carry_dir=a.carry_dir, resume_from=a.resume_from, deadline=a.deadline)
        Path(a.out).write_text(json.dumps(rec) + "\n")
    elif a.cmd == "groups":
        for k, v in groups(spec, a.rung, a.climbers).items():
            print(k, v)
    elif a.cmd == "kernel":
        kernel(a.name, a.rung, a.pin, a.climbers, segment=a.segment)
    elif a.cmd == "chain":
        chain(a.names, a.rung, a.pin, a.climbers)


if __name__ == "__main__":
    main(sys.argv[1:])
