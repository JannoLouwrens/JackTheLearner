"""J10 (2') — the one-kernel walk row: SBX SAC, the RL Zoo's Humanoid entry, Humanoid-v4.

PUBLISHED_PRIORS §16 (J10 question): the cheapest published walk. Humanoid-v4
at 1M steps, 10 seeds (SimBaV2's table): SAC 4,825 [3,784-5,866]. At 2M on v3
(SB3 Zoo, ONE model, 150 episodes): SAC 6,232 +- 280. On Gymnasium Humanoid a
return near 5,000 is STANDING (5 a step x 1,000 steps), so the row also demands
forward metres.

RULE (frozen in this commit, BEFORE the kernel body exists and before seed 0):

  env        Gymnasium `Humanoid-v4`, every default (venv /data/venvs/jack-std:
             gymnasium 1.4.0, mujoco 3.17.0, sbx 0.28.0, jax 0.11.2), one env.
  learner    sbx.SAC with the Zoo's Humanoid entry: n_timesteps 2e6,
             MlpPolicy, learning_starts 10,000; everything else the SB3 SAC
             defaults, SET EXPLICITLY because SBX's defaults are not SB3's:
             lr 3e-4, buffer 1e6, batch 256, tau 0.005, gamma 0.99,
             train_freq 1, gradient_steps 1, ent_coef "auto", net [256, 256].
             VERIFIED 11-18 10:15 (before seed 0): DLR-RM/rl-baselines3-zoo
             @ d802099a (sac.yml last touched b8a11d6e, 2026-10-16), file
             sha256 b05466c5..., `Humanoid-v4: <<: *mujoco-defaults,
             n_timesteps 2e6`; mujoco-defaults = MlpPolicy, learning_starts
             10000; no normalize, no wrapper. Matches the block above. SBX
             0.28 hidden defaults read from source and equal SB3's for the
             non-SDE path: log_std clamp [-20, 2], ent_coef init 1.0,
             target_entropy auto, n_critics 2, net [256, 256], policy_delay 1.
  differs from the published recipe (all listed): SBX (JAX) not SB3 (torch),
             same algorithm; the 2M figure is on v3, ours on v4 (v4 is the
             1M band's env); gymnasium 1.4 / mujoco 3.17 vs the papers'
             versions; Kaggle CPU, not the authors' hardware.
  seeds      0, 1, 2 (one chained run each); extension none.
  eval       at 1M and at 2M: 10 deterministic episodes, env seeds
             10_000..10_009, separate eval env. Per seed: mean return R and
             forward metres per 1,000 steps M = (final x - initial x)
             summed over the 10 episodes / steps taken x 1,000, x = the
             CENTRE OF MASS x (`info["x_position"]` on v4 is mass_center,
             not the torso; corrected 11-18 before seed 0 — the earlier
             "torso" contradicted the named source). Initial x = mass_center
             right after reset, same function.
  PASS       at 2M: R >= 5,950 (6,232 - 280) AND M >= 12.0, on >= 2 of 3
             seeds. Return alone is "STANDS", never "walks".
  1M check   median R over 3 seeds inside [3,784, 5,866]. Outside it the row
             is a NAMED MISS of the published curve (reported beside the 2M
             reading, whatever that is) and the cause is looked for before
             (3) uses the recipe.
  control    late start: same seed 0, learning_starts = 2,000,001 (never
             trains), 50k steps then the same eval; must stay R < 500. If it
             does not, the row is VOID (the eval or the reward is broken).
  resume     0.046 CPU-s a step on the box -> 2M ~ 26 CPU-h: does NOT fit
             one 12 h kernel. Each kernel trains until 10.5 h wall or the
             next 500k mark, then saves model + replay buffer
             (`save_replay_buffer`, ~3.1 GB at 1e6 x 376-dim obs) to its
             output; the next kernel attaches the previous one as a kernel
             source, loads both, and continues with reset_num_timesteps=False.
             The step counter, the ent_coef state and the RNG of the policy
             come from the checkpoint; the env is re-seeded with
             seed*1000 + segment (disclosed: env RNG is not carried across).
             A segment that dies without a checkpoint is re-run from the last
             checkpoint, never skipped. Evals happen only at exactly 1M and 2M.
  budget     Kaggle CPU only (box: nice 19, the 50k control smoke at most).

STATUS: rule + eval + late-start control (`evaluate`, `control_run`; box smoke
11-18). Training segments with resume are the next J10 slot.
"""
from __future__ import annotations

ENV_ID = "Humanoid-v4"
SEEDS = (0, 1, 2)
TOTAL_STEPS = 2_000_000
CHECK_STEPS = 1_000_000
SAC_KW = dict(learning_rate=3e-4, buffer_size=1_000_000, learning_starts=10_000,
              batch_size=256, tau=0.005, gamma=0.99, train_freq=1,
              gradient_steps=1, ent_coef="auto",
              policy_kwargs=dict(net_arch=[256, 256]))
EVAL_SEEDS = tuple(range(10_000, 10_010))
PASS_RETURN = 5_950.0
PASS_METRES_PER_1000 = 12.0
PASS_MIN_SEEDS = 2
CHECK_BAND_1M = (3_784.0, 5_866.0)
CONTROL_LEARNING_STARTS = TOTAL_STEPS + 1
CONTROL_STEPS = 50_000
CONTROL_MAX_RETURN = 500.0
SEGMENT_WALL_H = 10.5
SEGMENT_STEPS = 500_000


def verdict(r2m: dict, m2m: dict, r1m: dict, control_r: float) -> dict:
    """Pure reading of the frozen rule. r2m/m2m/r1m: seed -> value."""
    if control_r >= CONTROL_MAX_RETURN:
        return {"reading": "VOID", "why": f"late-start control R={control_r:.1f} >= {CONTROL_MAX_RETURN}"}
    walks = [s for s in SEEDS if r2m.get(s, -1e9) >= PASS_RETURN and m2m.get(s, -1e9) >= PASS_METRES_PER_1000]
    stands = [s for s in SEEDS if r2m.get(s, -1e9) >= PASS_RETURN]
    vals = sorted(r1m[s] for s in SEEDS if s in r1m)
    med = vals[len(vals) // 2] if len(vals) == 3 else None
    in_band = med is not None and CHECK_BAND_1M[0] <= med <= CHECK_BAND_1M[1]
    reading = "PASS" if len(walks) >= PASS_MIN_SEEDS else (
        "STANDS" if len(stands) >= PASS_MIN_SEEDS else "FAIL")
    return {"reading": reading, "walk_seeds": walks, "stand_seeds": stands,
            "median_1m": med, "check_1m": "IN BAND" if in_band else "NAMED MISS"}


def make_env():
    import gymnasium as gym
    return gym.make(ENV_ID)


def _com_x(env) -> float:
    from gymnasium.envs.mujoco.humanoid_v4 import mass_center
    u = env.unwrapped
    return float(mass_center(u.model, u.data)[0])


def evaluate(policy_fn, seeds=EVAL_SEEDS) -> dict:
    """The rule's eval: one episode per env seed on a SEPARATE env.
    policy_fn(obs) -> action (deterministic). Returns mean R, M and per-episode rows."""
    env = make_env()
    eps, metres, steps = [], 0.0, 0
    for sd in seeds:
        obs, _ = env.reset(seed=int(sd))
        x0, ret, n, done, info = _com_x(env), 0.0, 0, False, {}
        while not done:
            obs, r, term, trunc, info = env.step(policy_fn(obs))
            ret += float(r); n += 1; done = term or trunc
        dx = float(info["x_position"]) - x0
        eps.append({"seed": int(sd), "R": ret, "steps": n, "dx": dx, "fell": bool(term)})
        metres += dx; steps += n
    env.close()
    return {"R": sum(e["R"] for e in eps) / len(eps),
            "M": metres / max(steps, 1) * 1000.0, "episodes": eps}


def control_run(seed: int = 0, steps: int = CONTROL_STEPS) -> dict:
    """Late-start control: the Zoo entry with learning_starts past the end,
    `steps` env steps (never trains), then the rule's eval of its policy."""
    import time
    from sbx import SAC
    kw = dict(SAC_KW, learning_starts=CONTROL_LEARNING_STARTS)
    t0 = time.time()
    model = SAC("MlpPolicy", make_env(), seed=seed, verbose=0, **kw)
    model.learn(total_timesteps=steps)
    t_learn = time.time() - t0
    ev = evaluate(lambda o: model.predict(o, deterministic=True)[0])
    return {"seed": seed, "steps": steps, "learn_s": t_learn, **ev,
            "trained": int(model._n_updates) if hasattr(model, "_n_updates") else None}


if __name__ == "__main__":
    import json, sys
    if sys.argv[1:2] == ["control"]:
        n = int(sys.argv[2]) if len(sys.argv) > 2 else CONTROL_STEPS
        r = control_run(0, n)
        print(json.dumps({k: v for k, v in r.items() if k != "episodes"}))
        print("VOID" if r["R"] >= CONTROL_MAX_RETURN else "control holds (R < 500)")
