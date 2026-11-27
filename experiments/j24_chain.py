"""J24 (5): one J24 learner cell that can be cut and carried on (ORDERS J24 (5)).

`pilot_b2.run_pilot_cell` runs a cell in ONE `model.learn` call; a 500k cell
does not fit one Kaggle session, so the order says: "carry through `save_life`
(`forge/r1_run.py`'s carry; one whole carry or refuse)". This module is that
carry for the J24 arms and nothing else — the forge's machinery reused, not
rewritten: `carried_lives_cls` (the picklable PilotLives), `_carry_save`,
`_carry_load` (whole-or-refuse, another run's carry refused, sbx's actor alias
and SDE keys put back, the normaliser carried), `chunk_rule`, `carry_whole`.

What differs from the forge's `_run_carried`: the learner is built by
`pilot_b.make_b` from the J24 arm's own config exactly as `run_b` builds it
(not from a round spec), the world is `pilot_world(arm, PILOT_RULE frames)` —
so the blind / shuffled-bins wrappers (and ShuffledBins' own RNG) are inside
the carried life — and the finished record is `run_b`'s record + the keys
`run_pilot_cell` adds, so `j17_late_meals` reads it unchanged.

Identity (tests/test_j24_chain.py): a cell cut after chunk 1 and 2, each
segment a new process, == the unbroken `run_pilot_cell` — every life row,
every trace row, every actor weight — for all three J24 arms.
"""
import json
import time
from pathlib import Path

from experiments.forge import r1_run as rr

CHUNK = rr.CHUNK                 # 20,000 = lcm(train_freq 32, trace window 5,000)
SEG_WALL_S = rr.SEG_WALL_S
CHUNK_MARGIN = rr.CHUNK_MARGIN
J24_ONLY = ("K-bins16", "K-bins16-blind", "K-bins16-shuffled")
# J24 follow-up DRAFT (prereg/eye_bins16_k1.md, NOT frozen): the same three
# cells with forge K1's resets. sbx keeps `param_resets` / `reset_idx` on the
# model, and SB3's save/load carries both (tests/test_j24_chain.py proves a
# carried K1 cell IS the unbroken one, a reset in every segment).
J24K1_ONLY = ("K1-bins16", "K1-bins16-blind", "K1-bins16-shuffled")
CARRIED_ARMS = J24_ONLY + J24K1_ONLY


def carry_name(arm: str, seed: int) -> str:
    return f"j24carry_{arm}_s{seed}"


def _learner_kw(c: dict) -> dict:
    """run_pilot_cell's kwargs for run_b, for a learner arm (same order, same keys)."""
    kw = {} if c["lr"] is None else {"lr": c["lr"]}
    if c.get("trace"):
        kw["trace"] = True
    if "ent_coef" in c:
        kw["ent_coef"] = c["ent_coef"]
    for k in ("policy", "policy_kwargs", "use_sde", "optimizer", "critic",
              "param_resets"):
        if k in c:
            kw[k] = c[k]
    return kw


def run_cell_carried(arm: str, seed: int, T: int, *, carry_dir, resume_from=None,
                     deadline: float | None = None, chunk: int = CHUNK,
                     max_chunks: int | None = None, partial_path=None, plant=()) -> dict:
    """One J24 cell in whole chunks. Loads a whole carry from `resume_from` (or
    `carry_dir`) if there is one; when the next chunk would not fit before
    `deadline` (epoch s), saves to `carry_dir` and returns a `cut` record.
    `max_chunks` / `plant` exist for the tests only."""
    if arm not in CARRIED_ARMS:
        raise rr.Refused(f"j24_chain: {arm} is not a J24 arm {CARRIED_ARMS}")
    from experiments import pilot_b as pb
    from experiments.pilot_b2 import PILOT_RULE, arm_cfg, pilot_world
    c = arm_cfg(arm)
    kw = _learner_kw(c)
    lr = kw.pop("lr", 3e-4)
    trace = kw.pop("trace", False)
    ent_coef = kw.pop("ent_coef", 0.1)
    param_resets = kw.pop("param_resets", None)
    if set(kw):                                  # J24 arms set none of these
        raise rr.Refused(f"j24_chain: learner options {sorted(kw)} are not carried")
    trace_every = pb.TRACE_EVERY if trace else pb.TRAIN_FREQ
    rr.chunk_rule(int(T), int(chunk), pb.TRAIN_FREQ, trace_every)
    pb._single_thread()
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from sbx import SAC
    world = pilot_world(arm, PILOT_RULE["food_floor_only"], PILOT_RULE["eye_frame"],
                        PILOT_RULE["motor_frame"], PILOT_RULE["native_substep"])
    src = next((d for d in (resume_from, carry_dir) if rr.carry_whole(d)), None)
    venv = VecNormalize(DummyVecEnv([lambda: rr.carried_lives_cls()(
                            world(seed), partial_path=partial_path)]),
                        norm_obs=True, norm_reward=False, clip_obs=10.0)
    want = {"j24_arm": arm, "seed": int(seed), "T": int(T), "cfg": json.loads(json.dumps(c)),
            "chunk": int(chunk), "warmup": int(pb.WARMUP), "trace_every": int(pb.TRACE_EVERY)}
    if src is None:
        venv.seed(seed)
        model = pb.make_b(venv, seed, T, lr, use_sde=True, gamma=c["gamma"],
                          ent_coef=ent_coef, param_resets=param_resets)
        tr, meta = [], dict(want, wall_s=0.0, cpu_s=0.0, segments=[])
    else:
        model, tr, meta = rr._carry_load(src, venv, SAC, want, plant=plant)
        if int(model.num_timesteps) != meta["decisions"] or meta["decisions"] % chunk:
            raise rr.Refused(f"j24_chain: the carry at {src} does not sit on a chunk boundary")
        if (list(getattr(model, "param_resets", None) or []) != list(param_resets or [])
                or int(getattr(model, "reset_idx", 0)) != meta.get("param_resets_done", 0)):
            raise rr.Refused(f"j24_chain: the carry at {src} lost its reset schedule")
    w = model.env.envs[0]
    w.partial_path = None if partial_path is None else Path(partial_path)
    if trace:
        w.trace_ref = tr
    start, slowest, n = int(model.num_timesteps), 0.0, 0
    t_wall, t_cpu = time.time(), time.process_time()
    while model.num_timesteps < T:
        if max_chunks is not None and n >= max_chunks:
            break
        if deadline is not None and n > 0 and time.time() + CHUNK_MARGIN * slowest > deadline:
            break
        if deadline is not None and n == 0 and time.time() >= deadline:
            break
        t0 = time.time()
        model.learn(total_timesteps=int(chunk), progress_bar=False,
                    reset_num_timesteps=model.num_timesteps == 0,
                    **({"callback": pb._trace_callback(tr, pb.TRACE_EVERY)} if trace else {}))
        slowest, n = max(slowest, time.time() - t0), n + 1
        if model.num_timesteps % chunk:
            raise RuntimeError(f"j24_chain: a chunk ended off its boundary at "
                               f"{model.num_timesteps}")
    wall = meta["wall_s"] + time.time() - t_wall
    cpu = meta["cpu_s"] + time.process_time() - t_cpu
    segs = meta["segments"] + [{"start": start, "end": int(model.num_timesteps),
                                "code_head": rr._code_head(),
                                "resumed_from": src and str(src)}]
    lives = list(w.lives)
    if model.num_timesteps < T:
        rr._carry_save(carry_dir, model, venv, tr,
                       dict(want, decisions=int(model.num_timesteps), wall_s=wall, cpu_s=cpu,
                            segments=segs,
                            **({} if param_resets is None else
                               {"param_resets_done": int(model.reset_idx)})),
                       plant=plant)
        return {"arm": "b", "pilot_arm": arm, "seed": int(seed), "T": int(T), "cut": True,
                "resumable": True, "decisions": int(model.num_timesteps),
                "lives": lives, "carry": carry_name(arm, seed), "wall_s": round(wall, 1),
                "segments": segs}
    if w.n > 0:
        lives.append(w._life_row({"ate_total": dict(w.env.w.drives.ate_total)},
                                 "unfinished"))
    # run_b's record, key for key (lr / gamma / ent_coef as run_b writes them) ...
    rec = {"arm": "b", "seed": seed, "T": int(T), "lr": lr,
           "decisions": int(model.num_timesteps), "wall_s": round(wall, 1),
           "cpu_s": round(cpu, 1),
           "decisions_per_s": round(model.num_timesteps / wall, 2),
           "decisions_per_cpu_s": round(model.num_timesteps / cpu, 2),
           "pink_draws": int(getattr(model.policy, "n_pink", 0)),
           **({} if c["gamma"] is None else {"gamma": float(model.gamma)}),
           "world": pb._world_stamp(w.env),
           **({"trace": tr, "trace_every": pb.TRACE_EVERY} if trace else {}),
           **({} if ent_coef == 0.1 else {"ent_coef_setting": str(ent_coef)}),
           **({} if param_resets is None else
              {"param_resets": [int(x) for x in param_resets],
               "param_resets_done": int(getattr(model, "reset_idx", 0))}),
           "lives": lives}
    # ... + what run_pilot_cell adds for a J24 arm
    rec["pilot_arm"] = arm
    rec["food_floor_only"] = bool(PILOT_RULE["food_floor_only"])
    rec["eye_frame"], rec["motor_frame"] = PILOT_RULE["eye_frame"], PILOT_RULE["motor_frame"]
    rec["eye"], rec["blind"] = c["eye"], bool(c.get("blind"))
    if c.get("shuffled"):
        rec["shuffled"] = True if c["shuffled"] is True else c["shuffled"]
    rec["paint"] = c["paint"]
    rec["segments"] = segs
    return rec
