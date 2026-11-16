"""pilot_b.py — Brain B lives in his world: the 20-run pilot at the entry rung.

ORDERS J8; PLAN stage 1 (ready-list item 4), part "Learning to eat" + "His
brain", step 4 (a PILOT of the last 10%: it fixes W and T for the decisive
test, it moves no seat). Prior: PUBLISHED_PRIORS.md §1 ("Take as-is", "Ours
beside theirs"), §5 (Arm B), §8; rulings 19, 22, 26, 28.

NOT A NEW LEARNER. Brain B is `experiments/brain_b.py`'s build: SBX 0.28.0
`SAC` with `policy_kwargs=dict(layer_norm=True)`, code unmodified, run in
`/data/venvs/jack-std` (§5's pins). It reproduced J3 rows 1b (Pendulum,
-302.0, 15 seeds) and 3b (sparse MountainCar, 5/5 > 90). This file only wires
it to `jack_env.JackEnv` and logs every life.

THE WORLD (the entry rung, ruling 22): `JackEnv(difficulty=0.75,
reward="dr100-eat", random_needs=NEEDS_S_TRP (1/6, trp_env's published value),
random_layout=True)`. Episode = one life. Death -> terminated=True. A life
still going at LIFE_CAP = 9,000 decisions (1,800 sim-s, J1's cap) is ended
by THIS wrapper -> truncated=True (SB3 bootstraps through it), and the runner
puts a new body in the SAME world the way world_card does (`W0.respawn()`,
then the random-needs draw jack_env makes at every respawn). So W0's lives
persist in one world, as in every run here. Terminal observation on a death
step is the REBORN body's (J9 fixes that; harmless to a model-free learner
that never bootstraps through a death).

OBSERVATIONS are normalised by SB3's `VecNormalize(norm_obs=True,
norm_reward=False, clip_obs=10)` — running mean/var updated while training,
FROZEN (`training=False`) for any evaluation. Rewards are not normalised.

OURS BESIDE THEIRS (§1 "Take as-is" / §5 "Arm B"; every setting that differs
from the published recipe is named):

| setting | published (§1, §5) | ours | note |
|---|---|---|---|
| learner | SBX SAC + LayerNorm, 2x256 | same (sbx 0.28.0) | unmodified code |
| learning rate | 3e-4 | 3e-4 | |
| batch | 256 | 256 | |
| replay | >= whole run, never < 50k | T (whole run) | |
| tau | 0.005 | 0.005 | |
| discount | 0.99 to start; 0.99-0.997 = 20-67 s | 0.995 (= 1 - 1/200: 40 s at 0.2 s) | J8: "a 40 s horizon" |
| warm-up | 5,000-10,000 decisions | 5,000 (uniform random) | |
| updates/decision | 1-2 | 1 (32 updates every 32 decisions) | cadence of row 3b |
| exploration | row 3b's, as passed: SBX use_sde | SBX use_sde, noise draw held per 32-decision rollout, ent_coef 0.1 FIXED | DIFFERS from §5's default ent_coef "auto"; it is the setting that passed row 3b (§5: "gSDE with a FIXED entropy coefficient 0.1") |
| obs normalisation | yes | VecNormalize running stats, frozen at eval | |
| resets | none | none | |
| death / cap | terminated / truncated | same | |
| reward | x100 drive difference + pay on eating | dr100-eat (eat_pay 100) | jack_env option |
| start state | energy randomised | random_needs 1/6 at every birth | |
| layout | new each life | random_layout (foods re-placed each life) | inside one persistent room |
| memory | none in plain B | none | |

Known SBX quirks (read in the source for row 3b, not guessed): SBX's actor
ignores `log_std_init`/`use_expln` (initial std ~1), and `use_sde` holds one
standard-normal draw between `reset_noise()` calls (SB3's off-policy loop
calls it at the start of each rollout, i.e. every train_freq = 32 decisions).

LOG PER LIFE (one JSON row): decisions, seconds lived, floor meals (eat
events on every floor food the dial made), apples, cause ("energy" /
"integrity" / "capped"), intake (sum of eaten meal sizes read from the
world's own `drives._food` table at this rung, + NU_APPLE per apple),
basal drain (BASAL_B x seconds), sustained (capped AND intake >= basal drain
— ruling 22's mark), summed reward. A run's last, unfinished life is logged
with cause "unfinished" and never counts as sustained.

SMOKE (J8 item 2, measured 11-16 08:10-08:19 UTC, box, nice 19, seed 0,
20,000 decisions WITH learning, `smoke --seed 0 --decisions 20000`, log
/data/jack-logs/pilot_b/smoke_s0_20000.json): 497.8 s wall, 770.9 CPU-s ->
**40.2 decisions/s wall, 25.9 decisions per CPU-second** (JAX ran ~1.55 cores
despite the single-thread flags; size by CPU-seconds). 8 lives: 6 energy
deaths (0-1 meals), 1 unfinished, and life 5 (decisions 5,487-14,486, the
first after warm-up) CAPPED and SUSTAINED: 23 floor meals, intake 7.13 vs
basal drain 3.0. One seed: an anecdote, not a reading.
Sizing: 300,000 decisions = 11,580 CPU-s = 3.2 CPU-h per run; 4 runs on a
4-core Kaggle kernel ~3.2 h if a Kaggle core = a box core, 6.4 h if it is
half as fast -> fits one 9-hour kernel. T = 300,000 stands for every arm.

=============================================================================
THE RULE (J8 item 3) — committed 11-16 before any kernel. A PILOT: it moves no
seat; it fixes W and T for the decisive test and reads one of three answers.
=============================================================================
World      as above (d 0.75, dr100-eat, random_needs 1/6, random_layout, cap
           9,000 decisions). Each run = one fresh world `JackEnv(seed=s)` run
           for T decisions, lives in a row in that world.
T          300,000 decisions per run, every arm (warm-up included: B's 5,000
           random decisions are counted, harness check #7).
Arms/seeds B        Brain B as above, 20 runs, seeds 301..320.
           floor    5 runs each, seeds 301..305, same world and T:
             noop   do-nothing (world_card row a: zero action).
             held8  uniform(-1,1)^8 held 8 decisions (world_card row d8,
                    `Held(seed, 8)`; J1's best random row).
             blr0   Brain B with learning rate 0 (same plumbing, warm-up,
                    noise and normalisation; the network never moves).
           ceiling  J1's registered row e (privileged serve, d = 1, 200/200
                    lives capped at 99.4% of perfect harvest) — NOT re-run;
                    it proves only that the mark is reachable in the physics
                    (ruling 22). The rung here is easier (meal 0.31, 3 foods).
Per life   the row in LOG PER LIFE above; read through `experiments.harness`:
           primary `outcome_sustained(lives, W, T)` (capped AND intake >=
           basal drain, W consecutive), secondary `outcome(lives, W, T, 5)`
           (theta = 5 meals, "first signs of eating", decides nothing). A
           run's "unfinished" last life is passed as not capped.
Floor gate floor arms (noop + held8 + blr0, 15 runs) must total <= 1 run
           crossing the sustained mark at W = 1 by T; else the pilot is VOID
           (the mark is too easy at this rung) and says so.
Readings   with W in {1, 2, 3} all reported; n_B(W) = B runs crossing the
           sustained mark by T at W.
           SUSTAINED            n_B(1) >= 5 of 20 (and floor gate holds);
           EATS BUT STARVES     not sustained, but >= 5 of 20 B runs cross
                                the theta = 5 mark at W = 3;
           DOES NOT LEARN AT THIS BUDGET   neither.
           (5 of 20 against the floor's <= 1 of 15: Boschloo one-sided
           p ~ 0.05 — a pilot's bar, not the decisive test's.)
Fixes      W for the decisive test = the largest W in {1, 2, 3} with
           n_B(W) >= 5; T for the decisive test = 300,000 unless the median
           crossing time of the crossing B runs at that W is <= 150,000, then
           150,000 (never lower than the record's "reliable by 100k-200k").
           If DOES NOT LEARN: no W is fixed; the next step is J8's
           exploration arms and the record's longer budgets, not a decisive
           test at this T.
Reported   per arm: lives per run, meals per 1,000 decisions over the last
also       50,000 decisions (§5's last-10% outcome), share of lives capped,
           causes of death; per B run the decision of the first sustained
           life. Harness checks #2 (blr0 vs B), #4 (seed 301 run twice on
           the box for 20,000 decisions -> identical life logs), #7, #8 paid
           where the logs allow, rows in HARNESS_CHECKS.md.
Where      Kaggle CPU, 4 runs per kernel: B in 5 kernels, floor in 4 (noop
           and held8 need no learner). Then J8 item 6's two reported-only
           exploration arms (white noise; pink noise at selection only),
           5 seeds each, same T — registered in their own commit.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DIFFICULTY = 0.75          # ruling 22: d0
REWARD = "dr100-eat"
LIFE_CAP = 9000            # decisions = 1,800 sim-s (J1's cap)
WARMUP = 5000
GAMMA = 0.995
TRAIN_FREQ = 32
OUT = Path("/data/jack-logs/pilot_b")


def _single_thread():
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false "
                          "intra_op_parallelism_threads=1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")


def make_world(seed: int):
    from experiments.jack_env import JackEnv, NEEDS_S_TRP
    return JackEnv(seed=seed, difficulty=DIFFICULTY, reward=REWARD,
                   random_needs=NEEDS_S_TRP, random_layout=True)


try:
    import gymnasium as _gym
    _Wrapper = _gym.Wrapper
except ImportError:  # pragma: no cover
    _Wrapper = object


class PilotLives(_Wrapper):
    """One episode = one life; cap -> truncated + new body in the same world.

    Logs every finished life into `self.lives` (see module docstring).
    """

    def __init__(self, env, cap: int = LIFE_CAP):
        super().__init__(env)
        self.cap = int(cap)
        self.lives = []
        self._capped = False
        self._started = False
        self._new_life_counters()

    def _new_life_counters(self):
        self.n = 0
        self.rew = 0.0
        self.t0 = None
        self.ate0 = None

    def _meal_sizes(self) -> dict:
        from experiments.drives import NU_APPLE
        sizes = {k: float(nu) for k, (_gid, nu) in self.env.w.drives._food.items()}
        sizes.setdefault("apple", NU_APPLE)
        return sizes

    def reset(self, *, seed=None, options=None):
        e = self.env
        if self._capped:
            # world_card's censored respawn: same world, new body; plus the
            # random-needs draw jack_env makes at every respawn.
            e.w.respawn()
            if e.random_needs is not None:
                e._draw_needs()
            e._pending_obs = None
            e._truncated = False
            self._capped = False
        o, info = e.reset(seed=seed if not self._started else None)
        self._started = True
        self._new_life_counters()
        self.t0 = e.w.sim_seconds
        self.ate0 = dict(info["ate_total"])
        return o, info

    def _life_row(self, info, cause: str) -> dict:
        from experiments.drives import BASAL_B
        e = self.env
        ate = {k: int(v) - int(self.ate0.get(k, 0)) for k, v in info["ate_total"].items()}
        sizes = self._meal_sizes()
        intake = float(sum(ate[k] * sizes.get(k, 0.0) for k in ate))
        secs = float(e.w.sim_seconds - self.t0)
        drain = BASAL_B * secs
        floor = int(sum(v for k, v in ate.items() if k != "apple"))
        return {"decisions": int(self.n), "seconds": round(secs, 1),
                "floor_meals": floor, "apples": int(ate.get("apple", 0)),
                "cause": cause, "intake": round(intake, 4),
                "basal_drain": round(drain, 4),
                "sustained": bool(cause == "capped" and intake >= drain),
                "reward": round(self.rew, 3)}

    def step(self, action):
        o, r, term, trunc, info = self.env.step(action)
        self.n += 1
        self.rew += float(r)
        if not term and self.n >= self.cap:
            trunc = True
        if term:
            self.lives.append(self._life_row(info, info["death_cause"] or "death"))
        elif trunc:
            self._capped = True
            self.lives.append(self._life_row(info, "capped"))
        return o, r, term, trunc, info

    def unfinished(self, info) -> dict:
        return self._life_row(info, "unfinished")


def make_b(venv, seed: int, T: int, lr: float = 3e-4):
    """Brain B with §1/§5's settings (table above). lr=0 is the floor arm."""
    from sbx import SAC
    return SAC("MlpPolicy", venv, learning_rate=lr, buffer_size=max(int(T), 50_000),
               batch_size=256, tau=0.005, gamma=GAMMA, learning_starts=WARMUP,
               train_freq=TRAIN_FREQ, gradient_steps=TRAIN_FREQ, ent_coef=0.1,
               use_sde=True, seed=seed, verbose=0,
               policy_kwargs=dict(layer_norm=True, net_arch=[256, 256]))


def run_b(seed: int, T: int, lr: float = 3e-4, arm: str = "b") -> dict:
    """One learning run of T decisions; returns the per-life log + timing."""
    _single_thread()
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    holder = {}

    def _mk():
        w = PilotLives(make_world(seed))
        holder["w"] = w
        return w

    venv = VecNormalize(DummyVecEnv([_mk]), norm_obs=True, norm_reward=False,
                        clip_obs=10.0)
    venv.seed(seed)
    model = make_b(venv, seed, T, lr)
    t_wall, t_cpu = time.time(), time.process_time()
    model.learn(total_timesteps=int(T), progress_bar=False)
    wall, cpu = time.time() - t_wall, time.process_time() - t_cpu
    w = holder["w"]
    lives = list(w.lives)
    if w.n > 0:
        lives.append(w._life_row({"ate_total": dict(w.env.w.drives.ate_total)},
                                 "unfinished"))
    return {"arm": arm, "seed": seed, "T": int(T), "lr": lr,
            "decisions": int(model.num_timesteps), "wall_s": round(wall, 1),
            "cpu_s": round(cpu, 1),
            "decisions_per_s": round(model.num_timesteps / wall, 2),
            "decisions_per_cpu_s": round(model.num_timesteps / cpu, 2),
            "lives": lives}


def run_floor(arm: str, seed: int, T: int) -> dict:
    """noop / held8 for T decisions in the same world wrapper (no learner)."""
    from experiments.world_card import DoNothing, Held
    pol = DoNothing(seed) if arm == "noop" else Held(seed, 8)
    w = PilotLives(make_world(seed))
    o, info = w.reset(seed=seed)
    pol.new_life()
    t_wall, t_cpu = time.time(), time.process_time()
    for _ in range(int(T)):
        o, r, term, trunc, info = w.step(pol.act(o))
        if term or trunc:
            o, info = w.reset()
            pol.new_life()
    lives = list(w.lives)
    if w.n > 0:
        lives.append(w._life_row(info, "unfinished"))
    wall, cpu = time.time() - t_wall, time.process_time() - t_cpu
    return {"arm": arm, "seed": seed, "T": int(T), "decisions": int(T),
            "wall_s": round(wall, 1), "cpu_s": round(cpu, 1), "lives": lives}


def harness_lives(run: dict, mark: str = "sustained") -> list:
    """A run's log as `experiments.harness` reads it (unfinished = not capped)."""
    if mark == "sustained":
        return [(l["decisions"], l["cause"] == "capped", l["intake"], l["basal_drain"])
                for l in run["lives"]]
    return [(l["decisions"], l["floor_meals"] + l["apples"]) for l in run["lives"]]


def main(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["smoke"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--decisions", type=int, default=20_000)
    a = p.parse_args(argv)
    if a.cmd == "smoke":
        OUT.mkdir(parents=True, exist_ok=True)
        res = run_b(a.seed, a.decisions, arm="b-smoke")
        try:
            from experiments.brain_b import versions
            res["versions"] = versions()
        except Exception as ex:  # noqa: BLE001
            res["versions"] = repr(ex)
        path = OUT / f"smoke_s{a.seed}_{a.decisions}.json"
        path.write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps({k: v for k, v in res.items() if k != "lives"}))
        print(f"lives {len(res['lives'])}; causes "
              f"{[l['cause'] for l in res['lives']]}; floor meals "
              f"{[l['floor_meals'] for l in res['lives']]}")
        print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
