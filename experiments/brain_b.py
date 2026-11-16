"""brain_b.py — Brain B (the replay learner) from published code, reproducing
published numbers before it is trusted. ORDERS J3; PLAN stage 1, part "His
brain", step 2->3 (combine, reproduce). Prior: PUBLISHED_PRIORS.md §5.

NOT A REWRITE. Brain B *is* SBX's `SAC` (sbx-rl 0.28.0) with
`policy_kwargs=dict(layer_norm=True)`. This file only builds it, runs §5's
known-answer rows and writes their numbers. No ledger row (ORDERS J3: the
table goes to `docs/research/KNOWN_ANSWERS.md`).

Environment: `/data/venvs/jack-std` (Python 3.12.13), built 11-16 with §5's
pins exactly: stable-baselines3==2.9.2, sb3-contrib==2.9.2, sbx-rl==0.28.0,
rl-zoo3==2.9.2, gymnasium==1.4.0, mujoco==3.17.0, dm-control==1.0.50,
shimmy==2.0.1, pink-noise-rl==2.0.1 (+ torch 2.15.0+cpu, jax/jaxlib 0.11.2,
flax 0.12.12 as resolved). The project venv is not touched. On Kaggle the
pins row 1 imports are pip-installed over Kaggle's image (Python 3.13 there,
3.12 here — a stated difference); each cell records the versions it ran with.

Sources
- SAC: Haarnoja et al. 2018 (arXiv 1801.01290, 1812.05905); code SB3
  `stable_baselines3.SAC` (PyTorch) and SBX `sbx.SAC` (JAX), both DLR-RM.
- Settings: rl-zoo3 2.9.2 `hyperparams/sac.yml` entry `Pendulum-v1`:
  n_timesteps 20000, MlpPolicy, learning_rate 1e-3; all else library default.
- Reference: Open RL Benchmark run store, SB3 SAC Pendulum-v1, 13 seeds,
  mean return of all 100 training episodes -298.4 +- 10.3 (§5 row 1).

=============================================================================
THE RULE — row 1, committed before any run (11-16, this commit).
=============================================================================
Task       gymnasium 1.4.0 `Pendulum-v1`, default (200-step time limit =
           truncated), wrapped in SB3 `Monitor`.
Steps      20,000 environment steps = exactly 100 training episodes.
Metric     per seed: mean of the returns of ALL 100 training episodes (as the
           reference states it — training episodes, exploration included, not
           a deterministic evaluation).
Seeds      0, 1, 2, 3, 4 (5 seeds; `seed=` given to the algorithm, which seeds
           env, torch/jax and numpy).
Arms       lib   SB3 SAC (PyTorch), Zoo entry unmodified: lr 1e-3, every other
                 setting the library default (buffer 1e6, batch 256,
                 learning_starts 100, tau 0.005, gamma 0.99, train_freq 1,
                 gradient_steps 1, ent_coef "auto", net_arch [256, 256]).
           ours  SBX SAC, the same Zoo entry, plus layer_norm=True.
           ctrl  CONTROL THAT MUST FAIL: SB3 SAC with learning_starts = 20,001,
                 i.e. uniform random actions for all 100 episodes and no
                 gradient step — the same plumbing with the learning removed.
Accept     an arm REPRODUCES iff its 5-seed mean lies in [-314.7, -282.1]
           (§5: -298.4 +- 1.58 x 10.3). `lib` is run first; `ours` counts only
           if `lib` reproduces (otherwise the apparatus is broken, not the
           arm). The run is VOID if `ctrl`'s mean lies inside the band or
           above -282.1.
A miss means the build is broken (ORDERS J3): fix it, do not go on to row 3.

ROW 1 RESULT (11-16, kernel jack-cpu-1794791696): lib -319.5 +- 34.8
(per seed -312.5 -281.8 -305.6 -321.8 -376.0) MISSES; ours -285.9 +- 10.0
(in band, does not count); ctrl -1227.6 (holds). Verdict: lib MISSES.
Reference re-pulled 11-16 from the Open RL Benchmark store itself
(/data/jack-logs/brain_b/orb/, summary in KNOWN_ANSWERS.md): 14 finished
runs, -298.3 +- 9.9, range [-315.1, -283.5] — §5's number is right. Two of
our five lib seeds fall below the reference's worst. By 4-episode blocks the
loss is in BOTH phases: lib seed 4 learns ~1 block late (eps 13-16: -757 vs
reference -277 mean) and collapses once after learning (eps 21-24: -903);
lib eps 21-100 average -170.6 vs reference -156.8 (range -178.0..-131.4).
Reference stack (run configs): Python 3.8, SB3 1.x, gym 0.21, Jan 2023.
Ours: Python 3.13 on Kaggle, SB3 2.9.2, gymnasium 1.4.0, torch 2.11.0+cpu.

=============================================================================
THE RULE — row 1b, the re-registration, committed before any run (11-16).
=============================================================================
Why re-register: the row-1 rule took 5 seeds. Our lib's seed-level sd was
34.8, so the 5-seed mean's standard error (15.6) is the band's half-width
(16.3): a build that matched our lib's spread would miss about a third of
the time even if its centre were right. 5 seeds cannot tell "broken" from
"heavy tail". Row 1 STAYS RECORDED AS A MISS; row 1b is a stronger test on
fresh seeds, not a re-scoring.
Seeds      5..19 (15 fresh seeds; none of row 1's).
Task, steps, metric, arms (lib, ours), settings: exactly row 1's.
ctrl       seeds 5..9 (it never learns; 5 seeds suffice for the void check).
Accept     band UNCHANGED [-314.7, -282.1] on the 15-seed mean. lib first;
           ours counts only if lib reproduces. VOID if ctrl's mean >= -314.7.
Secondary  (reported, does not decide): two-sided Mann-Whitney U of lib's
           15 per-seed means against the reference's 14 (p reported); the
           count of lib seeds below the reference's worst (-315.1).
If lib misses 1b: the stack is the suspect, not the seed count. Next step
is the stack hunt — SB3 1.8 / gym 0.21 / Python 3.8 (the reference's) vs
ours, one difference at a time — before row 3.

ROW 1b RESULT (11-16, kernel jack-cpu-1794795073): lib -309.3 +- 13.7
REPRODUCES; ours -302.0 +- 22.2 REPRODUCES; ctrl -1227.7 (valid). Secondary:
Mann-Whitney vs the reference's 14 runs lib p = 0.011, ours p = 0.42; below
the reference's worst lib 5/15, ours 2/15. Next: row 3.

=============================================================================
THE RULE — row 3, committed before any run (11-16).
=============================================================================
Task       gymnasium 1.4.0 `MountainCarContinuous-v0`, default (999-step time
           limit = truncated; reward +100 at the flag minus 0.1*action^2),
           wrapped in SB3 `Monitor`. Sparse: no shaping.
Steps      50,000 environment steps.
Arms       r3lib   SB3 SAC (PyTorch) with the rl-zoo3 2.9.2 `sac.yml` entry
                   `MountainCarContinuous-v0` UNMODIFIED: lr 3e-4, buffer
                   50,000, batch 512, ent_coef 0.1 (fixed), train_freq 32,
                   gradient_steps 32, gamma 0.9999, tau 0.01, learning_starts
                   0, use_sde True, policy_kwargs log_std_init -3.67,
                   net_arch [64, 64], use_expln True; all else library default.
           r3ctrl  CONTROL THAT MUST FAIL: r3lib with learning_starts =
                   50,001 — SB3's warm-up (uniform random actions) for all
                   50k steps and no gradient step (learning removed, plumbing
                   kept); evaluated with its untrained deterministic policy.
Metric     per seed, after training: the mean return of 10 deterministic
           evaluation episodes (`deterministic=True`) on a fresh Monitor-
           wrapped env reset with seed 10_000 + seed (§5 row 3: "deterministic
           evaluation"; the Zoo's own evaluation also uses 10 episodes).
           Training returns are recorded beside it and do not decide.
Seeds      0, 1, 2, 3, 4.
Accept     r3lib REPRODUCES iff its metric is > 90 on >= 3 of the 5 seeds
           (§5: public store 9 of 10 seeds solved, 93.4-95.3). VOID if r3ctrl
           scores > 90 on any seed.
           A miss means the build is broken: the stack hunt, not row 3b.
Order      r3lib first. An SBX (`ours`) or pink-noise variant is a LATER rule,
           registered only after r3lib reproduces (§5: SBX's use_sde is a
           held noise draw, not SB3's gSDE — it is a different recipe and
           must pass this row on its own before it is used).

RESULT row 3 (11-16, Kaggle jack-cpu-1794803164): r3lib REPRODUCES — eval
94.8 94.5 95.5 95.0 93.5, 5 of 5 > 90 (first training episode > 90: ep 2-4);
r3ctrl -0.3 -2.8 -3.5 -0.9 -0.8, 0 of 5 (run valid). Stack: Python 3.13
(Kaggle image), SB3 2.9.2, torch 2.11.0+cpu, gymnasium 1.4.0.

=============================================================================
THE RULE — row 3b, Brain B on row 3, committed before any run (11-16).
=============================================================================
Why    §5: "any SBX or pink-noise variant must pass the same row before it is
       used". Brain B is SBX; r3lib passing says nothing about SBX.
Arms   r3ours  SBX 0.28.0 `SAC` + policy_kwargs layer_norm=True, with every
               r3lib setting SBX accepts, values unchanged: lr 3e-4, buffer
               50,000, batch 512, ent_coef 0.1 (fixed), train_freq 32,
               gradient_steps 32, gamma 0.9999, tau 0.01, learning_starts 0,
               use_sde True, net_arch [64, 64], log_std_init -3.67,
               use_expln True. SBX code unmodified.
       r3octrl CONTROL THAT MUST FAIL: r3ours with learning_starts = 50,001
               (uniform random warm-up throughout, no gradient step).
Task, steps, metric, seeds, threshold: exactly row 3's (above), unchanged.
Accept r3ours REPRODUCES iff its metric is > 90 on >= 3 of seeds 0-4. VOID if
       r3octrl scores > 90 on any seed.
Known before running (read in sbx 0.28.0 source, not guessed):
       - `SACPolicy.__init__` accepts `log_std_init` and `use_expln` and never
         uses them: the actor's log-std is a state-dependent Dense head
         clipped to [-20, 2] (initial std ~ 1, not SB3's exp(-3.67) = 0.025).
       - SBX `use_sde` is NOT gSDE: it holds the same noise KEY between
         `reset_noise()` calls, so the standard-normal draw is fixed while
         mean and std still move with the state. SB3's off-policy loop calls
         `reset_noise()` at the start of every rollout (each 32 steps here),
         so the draw is held 32 steps.
       So r3ours runs a DIFFERENT exploration recipe under the Zoo's numbers.
       That is the point of the row: B is used only if this recipe solves
       the sparse task by the same bar.
A miss  B's exploration is not the published one and is not trusted on
       sparse reward. Next is then a later rule (pink noise per life,
       sequence as long as a life, §5 line "never both"), not a re-run.

THE RULE — row 2, the PPO reference on Pendulum, committed before any run (11-16).
Source  rl-zoo3 2.9.2 `hyperparams/ppo.yml`, entry `Pendulum-v1`, read verbatim
        from the pinned package in /data/venvs/jack-std (11-16 05:45):
          n_envs: 4 / n_timesteps: 1e5 / policy: MlpPolicy / n_steps: 1024 /
          gae_lambda: 0.95 / gamma: 0.9 / n_epochs: 10 / ent_coef: 0.0 /
          learning_rate: 1e-3 / clip_range: 0.2 / use_sde: True / sde_sample_freq: 4
        No `normalize` key. 1e5 with 4 envs x 1024 = 102,400 steps (25 rollouts),
        matching §5's 102,400. Everything not listed: SB3 2.9.2 PPO defaults.
Arms    r2lib   SB3 PPO (PyTorch) with exactly that entry, make_vec_env(4 envs,
                DummyVecEnv, Monitor per env), device cpu, one torch thread.
        r2ctrl  CONTROL THAT MUST FAIL: r2lib with learning_rate = 0.0 (rollouts
                and the update loop run, the weights never move).
Seeds   0..4 (5 seeds per arm, §5 table).
Metric  §5 row 2 "last-100-episode training return": the mean of the model's
        ep_info_buffer at the end of learn() (SB3 keeps the last 100 finished
        episodes across the 4 envs; = the logged rollout/ep_rew_mean).
Accept  r2lib REPRODUCES iff the 5-seed mean of the metric lies in
        [-213.2, -162.0] (§5: -187.6 +- 17.0). VOID unless r2ctrl completes all
        5 seeds and every r2ctrl seed is BELOW -213.2. Known before running: §5
        also quotes -184.5 +- 80.7 for the same config from another seed set; if
        the seed sd is near 80, a 5-seed mean misses the band by chance often. A
        miss is recorded as a miss; any re-registration (more seeds) is a new
        rule with the reason, as rows 1 -> 1b. Row 4 (MountainCar PPO) is not in
        this rule: §5 gives 87.5 +- 4.2 but no accept band.

ROW 2 RESULT (11-16, kernel jack-cpu-1794807554): r2lib per seed -204.6
-178.5 -210.5 -217.2 -297.0, mean -221.5 (sd 44.6) — MISSES the band
[-213.2, -162.0]; median -210.5 is inside; seed 4 alone carries the miss.
r2ctrl -1257.7 -1214.1 -1239.3 -1236.2 -1239.4 (all < -213.2: valid).

THE RULE — row 2b, the re-registration, committed before any run (11-16).
Why: as rows 1 -> 1b. Seed sd 44.6 makes the 5-seed mean's standard error
(20.0) close to the band's half-width (25.6): 5 seeds cannot tell "broken"
from "heavy tail". Row 2 STAYS RECORDED AS A MISS; 2b is a stronger test on
fresh seeds, not a re-scoring.
Seeds   r2lib 5..19 (15 fresh); r2ctrl 5..9 (void check only).
Task, steps, metric, settings: exactly row 2's.
Accept  band UNCHANGED [-213.2, -162.0] on the 15-seed mean. VOID unless
        r2ctrl completes and every r2ctrl seed is below -213.2.
If r2lib misses 2b: the stack is the suspect (SB3 2.9.2 / gymnasium 1.4 /
Python 3.13 vs the reference's), the stack hunt comes next, not row 4.

ROW 2B RESULT (11-16, kernel jack-cpu-1794808095): r2lib 15-seed mean -199.5
(per seed -195.8 -192.7 -195.0 -260.9 -211.6 -228.9 -165.6 -193.2 -173.7
-189.8 -183.3 -177.8 -217.2 -192.7 -214.2) — INSIDE [-213.2, -162.0]:
r2lib REPRODUCES. r2ctrl -1241.5 -1229.3 -1224.5 -1266.3 -1235.5 (valid).
Row 2 (5 seeds) stays a miss on the record; 2b is the stronger test.

THE RULE — row 4, the PPO reference on MountainCarContinuous, committed before any run (11-16).
Source  rl-zoo3 2.9.2 `hyperparams/ppo.yml`, entry `MountainCarContinuous-v0`,
        read verbatim from the pinned package in /data/venvs/jack-std (11-16 06:15):
          normalize: true / n_envs: 1 / n_timesteps: 20000 / policy: MlpPolicy /
          batch_size: 256 / n_steps: 8 / gamma: 0.9999 / learning_rate: 7.77e-05 /
          ent_coef: 0.00429 / clip_range: 0.1 / n_epochs: 10 / gae_lambda: 0.9 /
          max_grad_norm: 5 / vf_coef: 0.19 / use_sde: True /
          policy_kwargs: dict(log_std_init=-3.29, ortho_init=False)
        `normalize: true` = VecNormalize with the Zoo's own default kwargs plus
        gamma=0.9999 (exp_manager.py line 438 copies the entry's gamma in).
        batch_size 256 > the 8-step buffer: SB3 then uses the whole 8 as one
        mini-batch; kept verbatim (it is what the public runs ran).
        Everything not listed: SB3 2.9.2 PPO defaults.
Arms    r4lib   SB3 PPO (PyTorch), that entry, make_vec_env(1 env, DummyVecEnv,
                Monitor) wrapped in VecNormalize(gamma=0.9999); device cpu, one
                torch thread.
        r4ctrl  CONTROL THAT MUST FAIL: r4lib with learning_rate = 0.0.
Seeds   r4lib 0..4; r4ctrl 0..4.
Metric  §5 row 4 "training return" = the long form's "Final train (last 100
        episodes)": mean of ep_info_buffer at the end of learn() (Monitor sits
        INSIDE VecNormalize, so these are raw, unnormalised returns). At
        20,000 steps that is every finished episode (~20+), not 100.
Band    The long form's own acceptance formula (control_arm_engineering.md,
        "Acceptance rule ... m +/- 3 s sqrt(1/k + 1/n)", which produced rows 1
        and 2's bands): m 87.5, s 4.2, n 10 public seeds, k 5 ->
        87.5 +/- 6.9 = [80.6, 94.4].
Accept  r4lib REPRODUCES iff the 5-seed mean lies in [80.6, 94.4]. VOID unless
        r4ctrl completes all 5 seeds and every r4ctrl seed is BELOW 80.6.
        A miss is recorded as a miss; a re-registration (more seeds) is a new
        rule with its reason, as 1 -> 1b and 2 -> 2b.

Ours beside theirs (every setting of `ours` that differs from the reference run)
| setting        | reference (SB3 SAC, Zoo) | ours (Brain B)          |
| library        | PyTorch SB3 2.x          | SBX 0.28.0 (JAX)        |
| LayerNorm      | none                     | layer_norm=True         |
| everything else| Zoo Pendulum-v1 entry    | identical               |
§5 "Take as-is" lists B's in-world settings (lr 3e-4, 5-10k warm-up,
observations normalised, 1-2 updates per decision). Those are NOT used on
row 1: the row reproduces the Zoo entry, and §5 says "SAC, Zoo entry (lr
1e-3) ... then ours". The in-world settings are J3's later step and stage 2's.

Threads: one per cell (torch.set_num_threads(1); XLA single-threaded) so the
box rule (<= 2-3 workers, nice 19) and Kaggle's 4 cores are both respected.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = Path("/data/jack-logs/brain_b")
STD_PY = "/data/venvs/jack-std/bin/python"

ROW1_SEEDS = (0, 1, 2, 3, 4)
ROW1_STEPS = 20_000
ROW1_BAND = (-314.7, -282.1)
ROW1_ARMS = ("lib", "ours", "ctrl")
ROW3_SEEDS = (0, 1, 2, 3, 4)
ROW3_STEPS = 50_000
ROW3_ARMS = ("r3lib", "r3ctrl")
ROW3B_ARMS = ("r3ours", "r3octrl")
ROW2_ARMS = ("r2lib", "r2ctrl")
ROW2_SEEDS = (0, 1, 2, 3, 4)
ROW2_STEPS = 102_400
ROW2_BAND = (-213.2, -162.0)
ROW2B_SEEDS = tuple(range(5, 20))
ROW2B_CTRL_SEEDS = tuple(range(5, 10))
ROW3_EVAL_EPISODES = 10
ROW3_SOLVED = 90.0
ROW1B_SEEDS = tuple(range(5, 20))
ROW4_ARMS = ("r4lib", "r4ctrl")
ROW4_SEEDS = (0, 1, 2, 3, 4)
ROW4_STEPS = 20_000
ROW4_BAND = (80.6, 94.4)
ROW1B_CTRL_SEEDS = tuple(range(5, 10))
PINS = ["stable-baselines3==2.9.2", "sb3-contrib==2.9.2", "sbx-rl==0.28.0",
        "rl-zoo3==2.9.2", "gymnasium==1.4.0", "mujoco==3.17.0",
        "dm-control==1.0.50", "shimmy==2.0.1", "pink-noise-rl==2.0.1"]
# Row 1 on Kaggle installs only what row 1 imports: Kaggle's image is Python
# 3.13 and dm-control's `labmaze` has no 3.13 wheel (kernel jack-cpu-1794791632
# died in pip on it, 11-16 01:14, before any cell ran). Same pinned versions.
ROW1_KERNEL_PINS = ["stable-baselines3==2.9.2", "sbx-rl==0.28.0", "gymnasium==1.4.0"]
CHILD_TIMEOUT_S = int(5.0 * 3600)
REMOTE_TIMEOUT_S = int(6.0 * 3600)


def _single_thread():
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false "
                          "intra_op_parallelism_threads=1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")


def make_row1(arm: str, seed: int, steps: int = ROW1_STEPS):
    """Build the arm's model on a Monitor-wrapped Pendulum-v1."""
    _single_thread()
    import gymnasium as gym
    from stable_baselines3.common.monitor import Monitor
    env = Monitor(gym.make("Pendulum-v1"))
    if arm == "lib":
        import torch
        torch.set_num_threads(1)
        from stable_baselines3 import SAC
        return SAC("MlpPolicy", env, learning_rate=1e-3, seed=seed, device="cpu")
    if arm == "ours":
        from sbx import SAC
        return SAC("MlpPolicy", env, learning_rate=1e-3, seed=seed,
                   policy_kwargs=dict(layer_norm=True))
    if arm == "ctrl":
        import torch
        torch.set_num_threads(1)
        from stable_baselines3 import SAC
        return SAC("MlpPolicy", env, learning_rate=1e-3, seed=seed, device="cpu",
                   learning_starts=steps + 1)
    raise ValueError(f"arm {arm!r} not in {ROW1_ARMS}")


def versions() -> dict:
    from importlib.metadata import version, PackageNotFoundError
    out = {"python": sys.version.split()[0]}
    for p in ["stable-baselines3", "sbx-rl", "gymnasium", "torch", "jax",
              "jaxlib", "flax", "numpy"]:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def run_row1_cell(arm: str, seed: int, steps: int = ROW1_STEPS) -> dict:
    t0 = time.time()
    model = make_row1(arm, seed, steps)
    model.learn(total_timesteps=steps)
    env = model.get_env().envs[0]
    rets = [float(r) for r in env.get_episode_rewards()]
    return {"row": 1, "arm": arm, "seed": seed, "steps": steps,
            "episodes": len(rets), "mean_return": sum(rets) / max(1, len(rets)),
            "returns": rets, "wall_s": round(time.time() - t0, 1),
            "versions": versions()}


def verdict_row1(cells: list, seeds=None, ctrl_seeds=None) -> dict:
    lo, hi = ROW1_BAND
    seeds = list(seeds or ROW1_SEEDS)
    ctrl_seeds = list(ctrl_seeds or seeds)
    by = {}
    for c in cells:
        by.setdefault(c["arm"], []).append(c)
    out = {}
    for arm, cs in by.items():
        got = sorted(c["seed"] for c in cs)
        m = sum(c["mean_return"] for c in cs) / len(cs)
        sd = (sum((c["mean_return"] - m) ** 2 for c in cs) / max(1, len(cs) - 1)) ** .5
        want = ctrl_seeds if arm == "ctrl" else seeds
        complete = (sorted(c["seed"] for c in cs) == sorted(want)
                    and all(c["episodes"] == ROW1_STEPS // 200 for c in cs))
        out[arm] = {"seeds": got, "mean": round(m, 1), "sd": round(sd, 1),
                    "per_seed": [round(c["mean_return"], 1) for c in sorted(cs, key=lambda c: c["seed"])],
                    "in_band": lo <= m <= hi, "complete": complete}
    ctrl = out.get("ctrl")
    void = ctrl is None or not ctrl["complete"] or ctrl["mean"] >= lo
    lib_ok = "lib" in out and out["lib"]["complete"] and out["lib"]["in_band"]
    ours_ok = lib_ok and "ours" in out and out["ours"]["complete"] and out["ours"]["in_band"]
    out["verdict"] = ("VOID" if void else
                      ("lib REPRODUCES; ours " + ("REPRODUCES" if ours_ok else "MISSES"))
                      if lib_ok else "lib MISSES (apparatus broken)")
    return out


def run_parallel(cells: list, outdir, workers: int = 4) -> list:
    """Run (arm, seed) cells as child processes, `workers` at a time."""
    import subprocess
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pending = list(cells)
    running, done = [], []
    while pending or running:
        while pending and len(running) < workers:
            arm, seed = pending.pop(0)
            f = outdir / f"row1_{arm}_s{seed}.json"
            p = subprocess.Popen([sys.executable, "-m", "experiments.brain_b", "cell",
                                  arm, str(seed), str(f)], cwd=REPO,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            running.append((arm, seed, f, p, time.time()))
        time.sleep(5)
        for item in list(running):
            arm, seed, f, p, t0 = item
            if p.poll() is None and time.time() - t0 < CHILD_TIMEOUT_S:
                continue
            if p.poll() is None:
                p.kill()
            err = p.stderr.read().decode()[-800:]
            running.remove(item)
            rec = (json.loads(f.read_text()) if f.exists()
                   else {"arm": arm, "seed": seed, "error": f"rc={p.returncode}: {err}"})
            done.append(rec)
            print(f"{arm} s{seed}: {rec.get('mean_return', rec.get('error'))}", flush=True)
    return done


_JOB = r'''
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q"] + __PINS__, check=True)
import json, os
os.chdir("/tmp/jack")
from experiments.brain_b import run_parallel, verdict_row1
r = run_parallel(__CELLS__, os.environ["JACK_OUT"])
ok = [c for c in r if "error" not in c]
json.dump({"cells": r, "verdict": __VERDICT__ if ok else None},
          open(os.path.join(os.environ["JACK_OUT"], "brain_b_row1.json"), "w"))
print("DONE", flush=True)
'''


def row1_cells() -> list:
    # lib first (the rule: the unmodified library is run first), then ours, ctrl
    return [(a, s) for a in ROW1_ARMS for s in ROW1_SEEDS]


def row1_kernel() -> dict:
    """Submit row 1 (3 arms x 5 seeds) as one Kaggle CPU kernel; write the
    cells and verdict to OUT/row1.json."""
    from experiments.gpu import build_job, result_json, submit
    body = (_JOB.replace("__CELLS__", repr(row1_cells())).replace("__PINS__", repr(ROW1_KERNEL_PINS))
            .replace("__VERDICT__", "verdict_row1(ok)"))
    sub = submit(build_job(body), prefer="kaggle-cpu", est_hours=3.0,
                 timeout_s=REMOTE_TIMEOUT_S, fetch=["brain_b_row1.json"])
    if not sub.ok:
        raise RuntimeError(f"brain_b row1 kernel: {sub.message}")
    r = result_json(sub, "brain_b_row1.json")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "row1.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r.get("verdict"), indent=1), flush=True)
    return r


def row1b_cells() -> list:
    # lib first, then ours, then ctrl (row 1b rule)
    return ([("lib", s) for s in ROW1B_SEEDS] + [("ours", s) for s in ROW1B_SEEDS]
            + [("ctrl", s) for s in ROW1B_CTRL_SEEDS])


def row1b_kernel() -> dict:
    """Submit row 1b (lib, ours x 15 fresh seeds; ctrl x 5) as one Kaggle CPU
    kernel; write cells and verdict to OUT/row1b.json."""
    from experiments.gpu import build_job, result_json, submit
    body = (_JOB.replace("__CELLS__", repr(row1b_cells())).replace("__PINS__", repr(ROW1_KERNEL_PINS))
            .replace("__VERDICT__", f"verdict_row1(ok, seeds={list(ROW1B_SEEDS)!r}, "
                                    f"ctrl_seeds={list(ROW1B_CTRL_SEEDS)!r})"))
    sub = submit(build_job(body), prefer="kaggle-cpu", est_hours=3.0,
                 timeout_s=REMOTE_TIMEOUT_S, fetch=["brain_b_row1.json"])
    if not sub.ok:
        raise RuntimeError(f"brain_b row1b kernel: {sub.message}")
    r = result_json(sub, "brain_b_row1.json")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "row1b.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r.get("verdict"), indent=1), flush=True)
    return r


def make_row3(arm: str, seed: int, steps: int = ROW3_STEPS):
    """Row 3: the Zoo's sparse-MountainCar SAC gSDE entry, unmodified."""
    _single_thread()
    import gymnasium as gym
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    if arm not in ROW3_ARMS + ROW3B_ARMS:
        raise ValueError(f"arm {arm!r} not in {ROW3_ARMS + ROW3B_ARMS}")
    env = Monitor(gym.make("MountainCarContinuous-v0"))
    if arm in ROW3B_ARMS:                       # row 3b: Brain B (SBX), rule above
        from sbx import SAC as SBXSAC
        return SBXSAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=50_000,
                      batch_size=512, ent_coef=0.1, train_freq=32, gradient_steps=32,
                      gamma=0.9999, tau=0.01,
                      learning_starts=(steps + 1 if arm == "r3octrl" else 0),
                      use_sde=True, seed=seed,
                      policy_kwargs=dict(layer_norm=True, log_std_init=-3.67,
                                         net_arch=[64, 64], use_expln=True))
    return SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=50_000,
               batch_size=512, ent_coef=0.1, train_freq=32, gradient_steps=32,
               gamma=0.9999, tau=0.01,
               learning_starts=(steps + 1 if arm == "r3ctrl" else 0),
               use_sde=True, seed=seed, device="cpu",
               policy_kwargs=dict(log_std_init=-3.67, net_arch=[64, 64], use_expln=True))


def run_row3_cell(arm: str, seed: int, steps: int = ROW3_STEPS,
                  n_eval: int = ROW3_EVAL_EPISODES) -> dict:
    t0 = time.time()
    model = make_row3(arm, seed, steps)
    model.learn(total_timesteps=steps)
    train = [float(r) for r in model.get_env().envs[0].get_episode_rewards()]
    import gymnasium as gym
    ev = gym.make("MountainCarContinuous-v0")
    obs, _ = ev.reset(seed=10_000 + seed)
    rets = []
    for _ in range(n_eval):
        done, tot = False, 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = ev.step(a)
            tot += float(r)
            done = term or trunc
        rets.append(tot)
        obs, _ = ev.reset()
    return {"row": 3, "arm": arm, "seed": seed, "steps": steps,
            "eval_returns": rets, "eval_mean": sum(rets) / len(rets),
            "episodes": len(train), "mean_return": sum(train) / max(1, len(train)),
            "train_returns": train, "wall_s": round(time.time() - t0, 1),
            "versions": versions()}


def verdict_row3(cells: list) -> dict:
    by = {}
    for c in cells:
        by.setdefault(c["arm"], []).append(c)
    out = {}
    for arm, cs in by.items():
        cs = sorted(cs, key=lambda c: c["seed"])
        out[arm] = {"seeds": [c["seed"] for c in cs],
                    "eval_per_seed": [round(c["eval_mean"], 1) for c in cs],
                    "solved": sum(c["eval_mean"] > ROW3_SOLVED for c in cs),
                    "complete": [c["seed"] for c in cs] == list(ROW3_SEEDS)}
    if "r3ours" in out or "r3octrl" in out:     # row 3b
        ctrl, arm = out.get("r3octrl"), out.get("r3ours")
        name, miss = "r3ours", "r3ours MISSES (B's exploration fails the sparse row)"
    else:
        ctrl, arm = out.get("r3ctrl"), out.get("r3lib")
        name, miss = "r3lib", "r3lib MISSES (build broken)"
    void = ctrl is None or not ctrl["complete"] or ctrl["solved"] > 0
    ok = arm is not None and arm["complete"] and arm["solved"] >= 3
    out["verdict"] = "VOID" if void else (f"{name} REPRODUCES" if ok else miss)
    return out


def row3_cells() -> list:
    return [(a, s) for a in ROW3_ARMS for s in ROW3_SEEDS]   # r3lib first


def row3b_cells() -> list:
    return [(a, s) for a in ROW3B_ARMS for s in ROW3_SEEDS]  # r3ours first


def row3_kernel(cells=None, name: str = "row3") -> dict:
    """Submit row 3 (r3lib, r3ctrl x 5 seeds) — or row 3b's cells — as one
    Kaggle CPU kernel; write OUT/<name>.json."""
    from experiments.gpu import build_job, result_json, submit
    body = (_JOB.replace("__CELLS__", repr(cells or row3_cells())).replace("__PINS__", repr(ROW1_KERNEL_PINS))
            .replace("from experiments.brain_b import run_parallel, verdict_row1",
                     "from experiments.brain_b import run_parallel, verdict_row3")
            .replace("__VERDICT__", "verdict_row3(ok)"))
    sub = submit(build_job(body), prefer="kaggle-cpu", est_hours=3.0,
                 timeout_s=REMOTE_TIMEOUT_S, fetch=["brain_b_row1.json"])
    if not sub.ok:
        raise RuntimeError(f"brain_b {name} kernel: {sub.message}")
    r = result_json(sub, "brain_b_row1.json")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r.get("verdict"), indent=1), flush=True)
    return r


def make_row2(arm: str, seed: int, steps: int = ROW2_STEPS):
    """Row 2: the Zoo's Pendulum-v1 PPO entry, unmodified (rule above)."""
    _single_thread()
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    if arm not in ROW2_ARMS:
        raise ValueError(f"arm {arm!r} not in {ROW2_ARMS}")
    env = make_vec_env("Pendulum-v1", n_envs=4, seed=seed)
    return PPO("MlpPolicy", env, n_steps=1024, gae_lambda=0.95, gamma=0.9,
               n_epochs=10, ent_coef=0.0,
               learning_rate=(0.0 if arm == "r2ctrl" else 1e-3),
               clip_range=0.2, use_sde=True, sde_sample_freq=4,
               seed=seed, device="cpu")


def run_row2_cell(arm: str, seed: int, steps: int = ROW2_STEPS) -> dict:
    t0 = time.time()
    model = make_row2(arm, seed, steps)
    model.learn(total_timesteps=steps)
    last = [float(e["r"]) for e in model.ep_info_buffer]
    return {"row": 2, "arm": arm, "seed": seed, "steps": steps,
            "timesteps": int(model.num_timesteps), "last_n": len(last),
            "mean_return": sum(last) / max(1, len(last)), "last_returns": last,
            "wall_s": round(time.time() - t0, 1), "versions": versions()}


def verdict_row2(cells: list, seeds=None, ctrl_seeds=None) -> dict:
    seeds = list(seeds or ROW2_SEEDS)
    ctrl_seeds = list(ctrl_seeds or seeds)
    by = {}
    for c in cells:
        by.setdefault(c["arm"], []).append(c)
    out = {}
    for arm, cs in by.items():
        cs = sorted(cs, key=lambda c: c["seed"])
        vals = [c["mean_return"] for c in cs]
        out[arm] = {"seeds": [c["seed"] for c in cs],
                    "per_seed": [round(v, 1) for v in vals],
                    "mean": round(sum(vals) / len(vals), 1),
                    "complete": [c["seed"] for c in cs] == (ctrl_seeds if arm == "r2ctrl" else seeds)}
    ctrl, lib = out.get("r2ctrl"), out.get("r2lib")
    void = ctrl is None or not ctrl["complete"] or max(ctrl["per_seed"]) >= ROW2_BAND[0]
    ok = (lib is not None and lib["complete"]
          and ROW2_BAND[0] <= lib["mean"] <= ROW2_BAND[1])
    out["verdict"] = "VOID" if void else ("r2lib REPRODUCES" if ok else "r2lib MISSES")
    return out


def make_row4(arm: str, seed: int, steps: int = ROW4_STEPS):
    """Row 4: the Zoo's MountainCarContinuous-v0 PPO entry, unmodified (rule above)."""
    _single_thread()
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize
    if arm not in ROW4_ARMS:
        raise ValueError(f"arm {arm!r} not in {ROW4_ARMS}")
    env = VecNormalize(make_vec_env("MountainCarContinuous-v0", n_envs=1, seed=seed),
                       gamma=0.9999)
    return PPO("MlpPolicy", env, batch_size=256, n_steps=8, gamma=0.9999,
               learning_rate=(0.0 if arm == "r4ctrl" else 7.77e-05),
               ent_coef=0.00429, clip_range=0.1, n_epochs=10, gae_lambda=0.9,
               max_grad_norm=5, vf_coef=0.19, use_sde=True,
               policy_kwargs=dict(log_std_init=-3.29, ortho_init=False),
               seed=seed, device="cpu")


def run_row4_cell(arm: str, seed: int, steps: int = ROW4_STEPS) -> dict:
    t0 = time.time()
    model = make_row4(arm, seed, steps)
    model.learn(total_timesteps=steps)
    last = [float(e["r"]) for e in model.ep_info_buffer]
    return {"row": 4, "arm": arm, "seed": seed, "steps": steps,
            "timesteps": int(model.num_timesteps), "last_n": len(last),
            "mean_return": sum(last) / max(1, len(last)), "last_returns": last,
            "wall_s": round(time.time() - t0, 1), "versions": versions()}


def verdict_row4(cells: list) -> dict:
    by = {}
    for c in cells:
        by.setdefault(c["arm"], []).append(c)
    out = {}
    for arm, cs in by.items():
        cs = sorted(cs, key=lambda c: c["seed"])
        vals = [c["mean_return"] for c in cs]
        out[arm] = {"seeds": [c["seed"] for c in cs],
                    "per_seed": [round(v, 1) for v in vals],
                    "mean": round(sum(vals) / len(vals), 1),
                    "complete": [c["seed"] for c in cs] == list(ROW4_SEEDS)}
    ctrl, lib = out.get("r4ctrl"), out.get("r4lib")
    void = ctrl is None or not ctrl["complete"] or max(ctrl["per_seed"]) >= ROW4_BAND[0]
    ok = (lib is not None and lib["complete"]
          and ROW4_BAND[0] <= lib["mean"] <= ROW4_BAND[1])
    out["verdict"] = "VOID" if void else ("r4lib REPRODUCES" if ok else "r4lib MISSES")
    return out


def row4_cells() -> list:
    return [(a, s) for a in ROW4_ARMS for s in ROW4_SEEDS]   # r4lib first


def row2_cells() -> list:
    return [(a, s) for a in ROW2_ARMS for s in ROW2_SEEDS]   # r2lib first


def row2b_cells() -> list:
    return ([("r2lib", s) for s in ROW2B_SEEDS]
            + [("r2ctrl", s) for s in ROW2B_CTRL_SEEDS])


def row2_kernel(cells=None, name: str = "row2", verdict: str = "verdict_row2(ok)",
                imp: str = "from experiments.brain_b import run_parallel, verdict_row2") -> dict:
    """Submit row 2 (r2lib, r2ctrl x 5 seeds) — or row 2b — as one Kaggle CPU kernel."""
    from experiments.gpu import build_job, result_json, submit
    body = (_JOB.replace("__CELLS__", repr(cells or row2_cells())).replace("__PINS__", repr(ROW1_KERNEL_PINS))
            .replace("from experiments.brain_b import run_parallel, verdict_row1", imp)
            .replace("__VERDICT__", verdict))
    sub = submit(build_job(body), prefer="kaggle-cpu", est_hours=3.0,
                 timeout_s=REMOTE_TIMEOUT_S, fetch=["brain_b_row1.json"])
    if not sub.ok:
        raise RuntimeError(f"brain_b {name} kernel: {sub.message}")
    r = result_json(sub, "brain_b_row1.json")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r.get("verdict"), indent=1), flush=True)
    return r


def main(argv):
    if len(argv) >= 5 and argv[1] == "cell":      # cell <arm> <seed> <out.json> [steps]
        if argv[2] in ROW4_ARMS:
            steps = int(argv[5]) if len(argv) > 5 else ROW4_STEPS
            rec = run_row4_cell(argv[2], int(argv[3]), steps)
        elif argv[2] in ROW2_ARMS:
            steps = int(argv[5]) if len(argv) > 5 else ROW2_STEPS
            rec = run_row2_cell(argv[2], int(argv[3]), steps)
        elif argv[2] in ROW3_ARMS + ROW3B_ARMS:
            steps = int(argv[5]) if len(argv) > 5 else ROW3_STEPS
            rec = run_row3_cell(argv[2], int(argv[3]), steps)
        else:
            steps = int(argv[5]) if len(argv) > 5 else ROW1_STEPS
            rec = run_row1_cell(argv[2], int(argv[3]), steps)
        Path(argv[4]).write_text(json.dumps(rec))
        return 0
    if len(argv) >= 2 and argv[1] == "row2b-kernel":
        row2_kernel(row2b_cells(), "row2b",
                    f"verdict_row2(ok, seeds={list(ROW2B_SEEDS)!r}, "
                    f"ctrl_seeds={list(ROW2B_CTRL_SEEDS)!r})")
        return 0
    if len(argv) >= 2 and argv[1] == "row4-kernel":
        row2_kernel(row4_cells(), "row4", "verdict_row4(ok)",
                    "from experiments.brain_b import run_parallel, verdict_row4")
        return 0
    if len(argv) >= 2 and argv[1] == "row2-kernel":
        row2_kernel()
        return 0
    if len(argv) >= 2 and argv[1] == "row3b-kernel":
        row3_kernel(row3b_cells(), "row3b")
        return 0
    if len(argv) >= 2 and argv[1] == "row3-kernel":
        row3_kernel()
        return 0
    if len(argv) >= 2 and argv[1] == "row1b-kernel":
        row1b_kernel()
        return 0
    if len(argv) >= 2 and argv[1] == "row1-kernel":
        row1_kernel()
        return 0
    if len(argv) >= 2 and argv[1] == "smoke":     # smoke [steps]: all arms, seed 0
        steps = int(argv[2]) if len(argv) > 2 else 1000
        for arm in ROW1_ARMS:
            rec = run_row1_cell(arm, 0, steps)
            print(arm, rec["episodes"], round(rec["mean_return"], 1), rec["wall_s"], "s",
                  flush=True)
        print(json.dumps(versions()))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
