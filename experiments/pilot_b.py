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
PARTIAL_EVERY = 10         # finished lives between partial-file writes (PilotLives)


def _single_thread(pin_cpu=None):
    """Thread hygiene. MEASURED 11-17 (harness/single_thread_probe.py): the
    XLA_FLAGS below do NOT bound jax 0.11 — its `tf_XLAEigen` intra-op pool
    is sized to the CPUs in the affinity mask (4 threads here) and did ~46%
    of a learning run's CPU at 1.6 cores; that is INBOX 11-16 19:40's "2.4
    cores". The one lever that bounds it is the affinity mask itself, set
    BEFORE jax's backend starts: pin_cpu=<cpu id> (or env JACK_PIN_CPU)
    -> 1 pool thread, 1.00 cores, 9% less CPU, 1.46x the wall, and the
    learned actor + every life bit-identical to unpinned (probe docstring).
    Default None = unchanged (no box run of B without it)."""
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false "
                          "intra_op_parallelism_threads=1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if pin_cpu is None and os.environ.get("JACK_PIN_CPU", "") != "":
        pin_cpu = int(os.environ["JACK_PIN_CPU"])
    if pin_cpu is not None:
        xb = sys.modules.get("jax._src.xla_bridge")
        if xb is not None and getattr(xb, "_backends", None):
            raise RuntimeError("pin_cpu must be set before jax's backend starts")
        os.sched_setaffinity(0, {int(pin_cpu)})


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

    def __init__(self, env, cap: int = LIFE_CAP, partial_path=None,
                 partial_every: int | None = None):
        super().__init__(env)
        self.cap = int(cap)
        self.lives = []
        # J11 prep (11-17 20:20): every `partial_every` finished lives the log
        # so far is written atomically to `partial_path` (None = never, the
        # default, J8's behaviour). A run killed at CHILD_TIMEOUT_S or by
        # Kaggle's cap then still returns its finished lives instead of none.
        self.partial_path = None if partial_path is None else Path(partial_path)
        self.partial_every = max(1, int(PARTIAL_EVERY if partial_every is None
                                        else partial_every))
        self.total_decisions = 0
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
        self.total_decisions += 1
        self.rew += float(r)
        if not term and self.n >= self.cap:
            trunc = True
        if term:
            self.lives.append(self._life_row(info, info["death_cause"] or "death"))
        elif trunc:
            self._capped = True
            self.lives.append(self._life_row(info, "capped"))
        if ((term or trunc) and self.partial_path is not None
                and len(self.lives) % self.partial_every == 0):
            self.write_partial()
        return o, r, term, trunc, info

    def write_partial(self) -> None:
        """Atomic: a reader never sees a half-written file."""
        tmp = self.partial_path.with_name(self.partial_path.name + ".tmp")
        tmp.write_text(json.dumps({"partial": True,
                                   "decisions": int(self.total_decisions),
                                   "lives": self.lives}) + "\n")
        os.replace(tmp, self.partial_path)

    def unfinished(self, info) -> dict:
        return self._life_row(info, "unfinished")


def make_b(venv, seed: int, T: int, lr: float = 3e-4, use_sde: bool = True,
           gamma: float | None = None, ent_coef=0.1):
    """Brain B with §1/§5's settings (table above). lr=0 is the floor arm;
    use_sde=False is the item-6 exploration arms' plain SAC sampling.
    gamma=None is GAMMA (J8's 0.995); J11's arm G passes another discount.
    ent_coef (11-20, J15 ruling 57): default 0.1 FIXED = unchanged; E2c's auto
    twins pass "auto" (SBX: learned alpha, target entropy -dim(action))."""
    from sbx import SAC
    return SAC("MlpPolicy", venv, learning_rate=lr, buffer_size=max(int(T), 50_000),
               batch_size=256, tau=0.005, gamma=GAMMA if gamma is None else float(gamma),
               learning_starts=WARMUP,
               train_freq=TRAIN_FREQ, gradient_steps=TRAIN_FREQ, ent_coef=ent_coef,
               use_sde=use_sde, seed=seed, verbose=0,
               policy_kwargs=dict(layer_norm=True, net_arch=[256, 256]))


# ---- J8 item 6: two REPORTED-ONLY exploration arms (ruling 28) -------------
# Registered 11-16 16:30 in their own commit, after the 20 + 15 were launched.
# Source: FIELD_WATCH N1, arXiv:2611.11868 (30 seeds, sparse tasks: gSDE
# learned slower than white noise, z -0.75; pink +0.35); pink noise for
# off-policy RL is Eberhard et al. 2023 (ICLR), code `pink-noise-rl` 2.0.1
# (already pinned in jack-std, brain_b.PINS). Same world, T, seeds 301..305,
# learner and every other setting as arm `b`; ONLY action selection differs:
#   bwhite  SBX's plain SAC sampling (use_sde=False): a fresh N(0,1) draw
#           every decision -> a = tanh(mu + sigma * eps_t), eps iid.
#   bpink   same squashed Gaussian, but eps_t is a unit-variance pink (beta 1)
#           signal per action dim, `pink.cnrl.ColoredNoiseProcess`, seq_len =
#           LIFE_CAP (the paper's "episode length"; regenerated every 9,000
#           decisions, not reset at a death — the paper's SB3 noise behaves the
#           same), rng seeded by the run seed. Applied ONLY in the policy's
#           `_predict` (the action the body takes); SBX's jitted actor/critic
#           losses draw their own Gaussian samples, so the update is untouched.
# Warm-up (5,000 uniform random decisions) is the same for all three. These
# arms choose nothing and move no seat: they report, beside `b` (row 3b's held
# draw per 32 decisions), whether row 3b's exploration is the right default
# for the decisive test. Read the same way as `b` (sustained mark, W 1..3).
EXPLORE_ARMS = ("bwhite", "bpink")
EXPLORE_SEEDS = tuple(range(301, 306))


def _pink_selection(model, seed: int) -> None:
    """Swap the action-selection draw of an SBX SAC for pink noise (bpink)."""
    import jax
    import jax.numpy as jnp
    from pink.cnrl import ColoredNoiseProcess
    from sbx.common.policies import BaseJaxPolicy
    pol = model.policy
    gen = ColoredNoiseProcess(beta=1.0, size=(pol.action_space.shape[0], LIFE_CAP),
                              rng=np.random.default_rng(seed))

    @jax.jit
    def _loc_std(actor_state, obs):
        base = actor_state.apply_fn(actor_state.params, obs).distribution
        return base.loc, base.stddev()

    def _predict(observation, deterministic=False):
        if deterministic:
            return BaseJaxPolicy.select_action(pol.actor_state, observation)
        loc, std = _loc_std(pol.actor_state, observation)
        eps = jnp.asarray(gen.sample(), dtype=loc.dtype)
        pol.n_pink = getattr(pol, "n_pink", 0) + 1
        return jnp.tanh(loc + std * eps)

    pol._predict = _predict


TRACE_EVERY = 5000                      # decisions per training-trace window


def trace_window(actions) -> dict:
    """One window of the training trace from the actions B actually took
    (env scale, rows = decisions): per-dim mean, per-dim std, and the share of
    action components at |a| >= 0.95 (pinned at the box edge). Pure numpy."""
    a = np.asarray(actions, dtype=np.float64).reshape(len(actions), -1)
    return {"mean": [round(float(x), 4) for x in a.mean(0)],
            "std": [round(float(x), 4) for x in a.std(0)],
            "sat": round(float((np.abs(a) >= 0.95).mean()), 4)}


def trace_row(t: int, actions, last: dict) -> dict:
    """One trace row: the action window (``trace_window``) plus the learner's
    last logged ``train/*`` values. A learner key that collides with a window
    key (SBX logs ``train/std`` — the policy's own std — which used to
    overwrite the window's per-dim action ``std`` after the first window) is
    stored as ``train_<key>``; the window's value is never overwritten."""
    row = {"t": int(t), **trace_window(actions)}
    for k, v in last.items():
        val = (round(float(v), 5) if isinstance(v, (int, float, np.floating, np.integer))
               else None)
        row["train_" + k if k in row else k] = val
    return row


def _trace_callback(out: list, every: int = TRACE_EVERY):
    """SB3 callback that only READS: the actions taken (``locals['actions']``)
    and the last values SBX's SAC.train logged (actor/critic loss, ent_coef).
    Never touches an RNG, the buffer or the params (test_pilot_b_trace proves
    the lives are identical with it on and off)."""
    from stable_baselines3.common.callbacks import BaseCallback

    class _Trace(BaseCallback):
        def _on_training_start(self):
            self._acts, self._last = [], {}
            rec = self.model.logger.record

            def record(key, value, *a, **k):
                if key.startswith("train/"):
                    self._last[key[6:]] = value
                return rec(key, value, *a, **k)
            self.model.logger.record = record

        def _on_step(self):
            self._acts.append(np.array(self.locals["actions"], dtype=np.float64).ravel())
            if self.num_timesteps % every == 0:
                out.append(trace_row(self.num_timesteps, self._acts, self._last))
                self._acts = []
            return True
    return _Trace()


def run_b(seed: int, T: int, lr: float = 3e-4, arm: str = "b",
          world=None, gamma: float | None = None, partial_path=None,
          trace: bool = False, ent_coef=0.1) -> dict:
    """One learning run of T decisions; returns the per-life log + timing.
    arm "bwhite"/"bpink" = the item-6 exploration arms (block above).

    Two knobs for J11's pilot arms, added 11-17 BEFORE its rule exists (no
    arm is defined here; `tests/test_pilot_b_knobs.py` proves default =
    unchanged and each knob's effect): `world` = a callable seed -> JackEnv
    replacing `make_world` (the cures are jack_env options: move_cost_dial,
    spawn_clear, reward dr100-eat-pos); `gamma` = the learner's discount.
    Either one, when given, is recorded in the returned dict.
    `partial_path`: see PilotLives (None = no partial file, unchanged).
    `trace` (11-19, J15 E2c; default False = unchanged, no key added): every
    TRACE_EVERY decisions the returned dict's "trace" list gets one row --
    the window's action mean/std/saturation and SAC's last actor_loss,
    critic_loss, ent_coef -- so a learner that ends below its lr-0 twin
    (J11 partial 12/40: every arm) can be read for WHY, not only that.
    `ent_coef` (11-20, ruling 57): 0.1 = unchanged, no key added; any other
    value is recorded as rec["ent_coef_setting"]."""
    _single_thread()
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    holder = {}

    def _mk():
        w = PilotLives((make_world if world is None else world)(seed),
                       partial_path=partial_path)
        holder["w"] = w
        return w

    venv = VecNormalize(DummyVecEnv([_mk]), norm_obs=True, norm_reward=False,
                        clip_obs=10.0)
    venv.seed(seed)
    model = make_b(venv, seed, T, lr, use_sde=(arm not in EXPLORE_ARMS), gamma=gamma,
                   ent_coef=ent_coef)
    if arm == "bpink":
        _pink_selection(model, seed)
    t_wall, t_cpu = time.time(), time.process_time()
    tr = []
    model.learn(total_timesteps=int(T), progress_bar=False,
                **({"callback": _trace_callback(tr, TRACE_EVERY)} if trace else {}))
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
            "pink_draws": int(getattr(model.policy, "n_pink", 0)),
            **({} if gamma is None else {"gamma": float(model.gamma)}),
            **({} if world is None else {"world": _world_stamp(w.env)}),
            **({"trace": tr, "trace_every": TRACE_EVERY} if trace else {}),
            **({} if ent_coef == 0.1 else {"ent_coef_setting": str(ent_coef)}),
            "lives": lives}


def _world_stamp(env) -> dict:
    """The options a non-default world was actually built with (read off the
    env, not off the caller's intent)."""
    return {k: (v if isinstance(v, (bool, int, float, str, type(None))) else repr(v))
            for k in ("difficulty", "reward", "random_layout", "lazy_contact_audio",
                      "spawn_clear", "move_cost_dial", "native_substep", "eye",
                      "food_floor_only", "eye_frame", "motor_frame")
            for v in [getattr(env, k, "absent")]}


def run_floor(arm: str, seed: int, T: int, world=None, partial_path=None) -> dict:
    """noop / held8 for T decisions in the same world wrapper (no learner).
    `world` as in `run_b` (None = `make_world`)."""
    from experiments.world_card import DoNothing, Held
    pol = DoNothing(seed) if arm == "noop" else Held(seed, 8)
    w = PilotLives((make_world if world is None else world)(seed),
                   partial_path=partial_path)
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
            "wall_s": round(wall, 1), "cpu_s": round(cpu, 1),
            **({} if world is None else {"world": _world_stamp(w.env)}),
            "lives": lives}


def harness_lives(run: dict, mark: str = "sustained") -> list:
    """A run's log as `experiments.harness` reads it (unfinished = not capped)."""
    if mark == "sustained":
        return [(l["decisions"], l["cause"] == "capped", l["intake"], l["basal_drain"])
                for l in run["lives"]]
    return [(l["decisions"], l["floor_meals"] + l["apples"]) for l in run["lives"]]


# ---- Kaggle CPU kernels (J8 item 4) ---------------------------------------
T_RUN = 300_000
B_SEEDS = tuple(range(301, 321))
FLOOR_SEEDS = tuple(range(301, 306))
FLOOR_ARMS = ("noop", "held8", "blr0")
# §5's pins exactly (brain_b.PINS) for what a cell imports; mujoco pinned for W0.
# No dm-control: Kaggle is Python 3.13 and labmaze has no wheel (brain_b scar).
KERNEL_PINS = ["stable-baselines3==2.9.2", "sbx-rl==0.28.0", "gymnasium==1.4.0",
               "mujoco==3.17.0", "pink-noise-rl==2.0.1"]
# pink-noise-rl (§5 pin, numpy-only) for bpink: e1's first kernel (11-16 23:22)
# died at import, `No module named 'pink'` — the box smoke had it, Kaggle did not.
# Box smoke: 25.9 decisions per CPU-s, JAX ~1.55 cores -> 300k = 3.2 CPU-h a
# run; 4 runs on Kaggle's 4 cores (~6.2 cores asked) ~ 3.2-5 h wall.
CHILD_TIMEOUT_S = int(8.5 * 3600)
REMOTE_TIMEOUT_S = int(9.5 * 3600)


def run_cell(arm: str, seed: int, T: int = T_RUN, partial_path=None) -> dict:
    pp = dict(partial_path=partial_path)
    if arm == "b":
        return run_b(seed, T, **pp)
    if arm == "blr0":
        return run_b(seed, T, lr=0.0, arm="blr0", **pp)
    if arm in EXPLORE_ARMS:
        return run_b(seed, T, arm=arm, **pp)
    if arm in ("noop", "held8"):
        return run_floor(arm, seed, T, **pp)
    raise ValueError(f"unknown arm {arm!r}")


def all_cells() -> list:
    """The registered 35 runs: B x20 then the three floor arms x5."""
    return ([("b", s) for s in B_SEEDS]
            + [(a, s) for a in FLOOR_ARMS for s in FLOOR_SEEDS])


def kernel_groups() -> dict:
    """B in 5 kernels (k0..k4), floor in 4 (f0..f3), <= 4 cells each. Floor
    cells are interleaved by seed so no kernel carries more than two blr0
    learners (noop/held8 cost almost nothing)."""
    b = [("b", s) for s in B_SEEDS]
    groups = {f"k{i}": b[4 * i:4 * i + 4] for i in range(5)}
    floor = [(a, s) for s in FLOOR_SEEDS for a in FLOOR_ARMS]
    for i in range(4):
        groups[f"f{i}"] = floor[4 * i:4 * i + 4]
    # item 6 (reported only): e0 white 301-304, e1 pink 301-304, e2 both 305
    groups["e0"] = [("bwhite", s) for s in EXPLORE_SEEDS[:4]]
    groups["e1"] = [("bpink", s) for s in EXPLORE_SEEDS[:4]]
    groups["e2"] = [("bwhite", EXPLORE_SEEDS[4]), ("bpink", EXPLORE_SEEDS[4])]
    return groups


def run_parallel(cells: list, outdir, T: int = T_RUN) -> dict:
    """All cells at once as child processes; a dead/timed-out cell keeps rc+err."""
    import subprocess
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    procs = []
    for arm, seed in cells:
        f = outdir / f"pilot_{arm}_s{seed}.json"
        procs.append((arm, seed, f, subprocess.Popen(
            [sys.executable, "-m", "experiments.pilot_b", "cell", "--arm", arm,
             "--seed", str(seed), "--decisions", str(T), "--out", str(f),
             "--partial", str(partial_file(f))],
            cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)))
    res = {}
    t0 = time.time()
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


def partial_file(final: Path) -> Path:
    return final.with_name(final.stem + ".partial.json")


def from_partial(arm: str, seed: int, pf: Path) -> dict:
    """A cell with no final file: its finished lives from the partial file,
    marked `cut` with the decisions it reached; none found -> `not_found`,
    never invented."""
    if pf.exists():
        part = json.loads(pf.read_text())
        return {"arm": arm, "seed": seed, "lives": part["lives"], "cut": True,
                "decisions_reached": part["decisions"], "from_partial": True}
    return {"arm": arm, "seed": seed, "lives": [], "not_found": True}


def cells_from_artifacts(artifacts: dict, cells: list) -> dict | None:
    """J11 prep (11-17 20:50): a kernel cut by Kaggle's cap or reattached after
    its watcher timed out has no `pilot_b_cells.json`, but every cell left
    `pilot_<arm>_s<seed>.json` (finished) or its `.partial.json` (written every
    PARTIAL_EVERY lives) in JACK_OUT, and Kaggle returns all of JACK_OUT. This
    assembles the cells from those, exactly as run_parallel would have: final
    file first, else from_partial (marked `cut`), else `not_found` — never
    invented. None when no cell file came back at all (tests/test_pilot_b_partial)."""
    res, found = {}, 0
    for arm, seed in cells:
        final = f"pilot_{arm}_s{seed}.json"
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


_JOB = r'''
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q"] + __PINS__, check=True)
import json, os
os.chdir("/tmp/jack")
from experiments.pilot_b import run_parallel
from experiments.brain_b import versions
r = run_parallel(__CELLS__, os.environ["JACK_OUT"], __T__)
json.dump({"cells": r, "versions": versions()},
          open(os.path.join(os.environ["JACK_OUT"], "pilot_b_cells.json"), "w"))
print("DONE", flush=True)
'''


def job_body(cells: list, T: int = T_RUN) -> str:
    return (_JOB.replace("__CELLS__", repr(list(cells)))
            .replace("__PINS__", repr(KERNEL_PINS)).replace("__T__", repr(int(T))))


def kernel(name: str) -> dict:
    """Submit one group (k0..k4, f0..f3) as one Kaggle CPU kernel; write the
    returned runs to OUT/<name>.json (failed cells kept with their error)."""
    from experiments.gpu import build_job, result_json, submit
    cells = kernel_groups()[name]
    sub = submit(build_job(job_body(cells)), prefer="kaggle-cpu", est_hours=5.0,
                 timeout_s=REMOTE_TIMEOUT_S, fetch=["pilot_b_cells.json"])
    if sub.ok and "pilot_b_cells.json" in sub.artifacts:
        r = result_json(sub, "pilot_b_cells.json")
    else:
        r = cells_from_artifacts(sub.artifacts, cells)
        if r is None:
            raise RuntimeError(f"pilot_b kernel {name}: {sub.message}")
        print(f"pilot_b kernel {name}: no summary ({sub.message or 'missing'}); "
              f"assembled from per-cell files", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(r, indent=1) + "\n")
    for k, rec in r["cells"].items():
        print(f"{k}: {len(rec['lives'])} lives"
              + (f" ERROR {rec['error']}" if "error" in rec else ""), flush=True)
    return r


def main(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["smoke", "cell", "kernel"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--decisions", type=int, default=20_000)
    p.add_argument("--arm", default="b")  # b, blr0, noop, held8, bwhite, bpink
    p.add_argument("--out", default=None)
    p.add_argument("--partial", default=None)
    p.add_argument("--name", default=None)
    a = p.parse_args(argv)
    if a.cmd == "cell":
        res = run_cell(a.arm, a.seed, a.decisions, partial_path=a.partial)
        Path(a.out).write_text(json.dumps(res) + "\n")
        return
    if a.cmd == "kernel":
        kernel(a.name)
        return
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
