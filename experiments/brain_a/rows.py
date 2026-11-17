"""Brain A's known-answer rows (J14): the pinned runner, UNMODIFIED, on §2's first three tasks.

Source: `sheeprl==0.8.5` (PyPI; tag v0.8.5 = 22b4f164), `algo=dreamer_v3_5` with the smallest preset
`algo=dreamer_v3_5_1M`, plus `algo.compile.enabled=True` (J5: 9.8x on a T4, the official code's
`jax.jit` analogue). Code only, never weights: every run starts from zero. Line diff against the
official agent (`danijar/dreamerv3@e01491f`): `experiments/brain_a/README.md` §"Line diff" (11-16).

## THE RULE — committed 2026-11-17 before any seed runs (ruling 33's form)

Rows and bands, exactly as PUBLISHED_PRIORS §2 states them (official v1 size-S, ratio 512, repeat 2,
4 envs, 5 seeds; "pass = our mean within ~2 SD at each point", made exact here as
|our mean - published mean| <= 2 x published SD):

| row | task (DMC proprio)  | point (env steps) | published mean +- SD | pass band       |
|-----|---------------------|-------------------|----------------------|-----------------|
| A1  | cartpole_balance    | 100K              | 794 +- 52            | 690 .. 898      |
| A1  | cartpole_balance    | 200K              | 966 +- 17            | 932 .. 1000     |
| A2  | cartpole_swingup    | 100K              | 478 +- 20            | 438 .. 518      |
| A2  | cartpole_swingup    | 200K              | 778 +- 19            | 740 .. 816      |
| A3  | ball_in_cup_catch   | 200K              | 949 +- 9             | 931 .. 967      |

- Seeds 0, 1, 2 per row. A row PASSES when its 3-seed mean is inside the band at EVERY point.
- The ONE extension, stated now: if a row misses at 3 seeds, seeds 3 and 4 run (fresh) and that row
  is judged once more at 5 seeds, same bands. Nothing after that: a miss at 5 is a named miss.
- Score at a point P = mean return of the training episodes that END in env steps (0.9 P, P]
  (the official score files are training-episode returns). A seed with no episode in that window
  is a VOID seed, not a zero; a row with a VOID seed is not judged and says so.
- Control, one per row, seed 0: the never-trains twin (`algo.learning_starts` past the run's last
  step: dreamer_v3_5 plays its untrained policy from step 1 — no random prefill — and never takes
  a gradient step).
  It MUST land below the band's lower edge at every point, else the row is VOID (the task is
  too easy to tell a learner from none at that point).
- Reported, not judged: §2's step-0 check (reward and value loss 5.541, dyn = rep = 1.0, actor std
  0.893, predicted reward and value 0) read from the first logged step of each seed.
- Settings that differ from the published protocol (ours beside theirs): agent v2 port (sheeprl
  `dreamer_v3_5`) vs the v1 agent of the 2023 score files; size `1m` (~0.64M params) vs size S —
  §2: "No score exists for size1m"; compile on (numerics, not recipe); everything else as published
  (repeat 2, 4 envs, train ratio 512 = sheeprl `replay_ratio` 0.5, batch 16 x 64).
  A miss can therefore mean a broken build OR that 1m is too small: the miss is written with both
  readings and not explained away.
- Not run on this box (ruling 32): Kaggle GPU from Sat 11-21 00:00 UTC through `gpu.py`
  (prefer="kaggle"), pushed first. Cost on record: ~7 T4-h at 3 seeds (J5, compiled 25.9 steps/s).

## Unit note to verify in the smoke (J14 item 4)

sheeprl's `policy_step` counts decisions x envs (NOT simulator frames); with action repeat 2,
env steps = 2 x policy_step. `POINTS` below are env steps; `total_steps` is policy steps.
The smoke must confirm this from the logged `Rewards/rew_avg` steps before any GPU run.
"""
from __future__ import annotations

ACTION_REPEAT = 2

ROWS = {
    "A1": {"domain": "cartpole", "task": "balance", "points": {100_000: (794, 52), 200_000: (966, 17)}},
    "A2": {"domain": "cartpole", "task": "swingup", "points": {100_000: (478, 20), 200_000: (778, 19)}},
    "A3": {"domain": "ball_in_cup", "task": "catch", "points": {200_000: (949, 9)}},
}
SEEDS = (0, 1, 2)
EXTENSION_SEEDS = (3, 4)
CONTROL_SEED = 0

BASE = ["exp=dreamer_v3_5_dmc_walker_walk", "algo=dreamer_v3_5_1M",
        "env.wrapper.from_pixels=False", "env.wrapper.from_vectors=True",
        "algo.cnn_keys.encoder=[]", "algo.mlp_keys.encoder=[state]",
        "env.num_envs=4", "env.capture_video=False", "env.action_repeat=2",
        "algo.replay_ratio=0.5",   # = train ratio 512 / (16 x 64); the exp file's value, pinned
                                   # because the algo default is 0.03125 (train ratio 32 = the
                                   # §2 control "ratio 32" — must never leak into a row)
        "algo.compile.enabled=True"]


def last_env_step(row: str) -> int:
    return max(ROWS[row]["points"])


def overrides(row: str, seed: int, control: bool = False) -> list:
    """Command-line overrides for one run. The runner itself is never edited."""
    r = ROWS[row]
    total_policy = last_env_step(row) // ACTION_REPEAT
    out = BASE + [f"env.wrapper.domain_name={r['domain']}", f"env.wrapper.task_name={r['task']}",
                  f"seed={seed}", f"algo.total_steps={total_policy}"]
    if control:
        out.append(f"algo.learning_starts={total_policy + 1}")
    return out


def band(mean: float, sd: float) -> tuple:
    return (mean - 2 * sd, min(1000.0, mean + 2 * sd))


def score_at(episodes, point_env: int):
    """episodes: [(end_env_step, return)]. Mean return of episodes ending in (0.9 P, P]; None = VOID."""
    xs = [ret for step, ret in episodes if 0.9 * point_env < step <= point_env]
    return sum(xs) / len(xs) if xs else None


def judge(row: str, per_seed: dict, control_episodes) -> dict:
    """per_seed: {seed: [(end_env_step, return)]}. Returns the row's verdict per the rule above."""
    pts = ROWS[row]["points"]
    out = {"row": row, "seeds": sorted(per_seed), "points": {}}
    verdict = "PASS"
    for p, (m, sd) in sorted(pts.items()):
        lo, hi = band(m, sd)
        scores = {s: score_at(eps, p) for s, eps in per_seed.items()}
        ctrl = score_at(control_episodes, p)
        rec = {"band": (lo, hi), "scores": scores, "control": ctrl}
        if any(v is None for v in scores.values()) or ctrl is None:
            rec["reading"] = "VOID (no episode in window)"
            verdict = "VOID"
        elif ctrl >= lo:
            rec["reading"] = "VOID (never-trains control inside or above band)"
            verdict = "VOID"
        else:
            mean = sum(scores.values()) / len(scores)
            rec["mean"] = mean
            rec["reading"] = "in band" if lo <= mean <= hi else "MISS"
            if rec["reading"] == "MISS" and verdict == "PASS":
                verdict = "MISS"
        out["points"][p] = rec
    out["verdict"] = verdict
    out["extension_due"] = verdict == "MISS" and len(per_seed) == len(SEEDS)
    return out
