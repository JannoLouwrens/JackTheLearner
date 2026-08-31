"""T2.09 — a percept-driven curiosity signal the noisy TV does NOT capture.

GOAL.md stakes the project on curiosity that is real. PG.4 (PASS) certified the
trap: a naive prediction-error agent in this exact playground fixates on the
re-randomizing panel (dwell 0.62 vs a random walk's 0.17). T2.08 (PASS)
certified that an intrinsic signal can beat random coverage — but its winner is
a POSITION-STATE pseudo-count, and its own docstring says it out loud: "no arm
reads the retina at all". That is why T2.08 cannot stand in for this spec, and
it is the whole design constraint here.

THE VACUITY THIS SPEC EXISTS TO AVOID, stated before the design because the
design is downstream of it. "Injecting an unpredictable observation channel
does not capture the intrinsic reward" is FREE for any reward that never reads
the observation. Feed T2.08's position pseudo-count into this rig and it scores
a perfect non-fixation while proving nothing whatsoever about curiosity's
robustness — the noise cannot enter a reward computed from (x, y). A PASS on
that arm would certify a tautology. So every arm here is PERCEPT-DRIVEN: its
reward is a function of the 68-d retina, and the retina is where the noise is.
(Overseer, 45th audit, FOR THE BUILDER B4: "its claim arm must be
percept-driven or a PASS means nothing.")

The rig is PG.4's, imported not copied, so the trap this spec claims to survive
is the same trap PG.4 certified as lethal: same playground MJCF, same 32-ray
retina at panel height, same visual-acuity falloff, same non-colliding
velocity rover, same tabular Q-learner over 1 m floor cells with the same
optimistic init and epsilon schedule. Arms differ in ONE thing — how the
intrinsic reward is computed from the retina. Changing PG.4 or playground.py
stales this certificate loudly (IMPL_DEPS).

THE FOUR ARMS (matched: same Q-learner, same schedule, same world, same seed
family; only the reward differs):
  icm       PG.4's forward-model squared prediction error. This is the NULL,
            not a candidate — PG.4 proved it fixates. It is also this spec's
            CONTROL: it must FAIL the claim gates in the same run that the
            claim arm passes them, or the trap was not set today.
  rnd       Random network distillation: a predictor trained to match a frozen
            random target on the next retina. Reward = ||f(o') - g(o')||^2.
  disagree  Ensemble disagreement (Pathak et al., arXiv:1906.04161): K forward
            models, identical but for init, trained on the same replay; reward
            = mean across output dims of the VARIANCE across members. This is
            the mechanism the noisy-TV literature says should work, and the
            reason is worth stating because it is what the spec measures: an
            irreducibly stochastic channel drives every member to the same
            conditional mean, so error stays high while DISAGREEMENT collapses.
  zero      r = 0 fed to the identical Q-learner. Not a candidate — the
            LIVENESS INSTRUMENT (see below).

WHY `zero` EXISTS, and it is the sharpest lesson carried in from T2.08. In a
bootstrapped tabular Q-learner with optimistic init, the init ALONE sweeps the
map: T2.08 measured every positive-reward construction it tried scoring BELOW
a random walk (naive ICM 0.283, RND 0.289, count bonus 0.327, all vs random
0.638), because accumulated Q in the visited core beats one-shot frontier
bonuses. So "the claim arm explores well" is NOT evidence its reward is alive —
a reward stuck at zero would inherit the optimistic sweep and look excellent.
`zero` makes the comparison honest by running the identical learner with the
signal deleted, exactly as T2.19's unimodal legs proved its regression arm was
a trained arm losing rather than a dead arm.

PRE-REGISTERED GATES — FROZEN 2026-08-29 against the seed-7 and seed-90 pilots
(`_GATES_FROZEN = True`). Every number the freeze was made against, and the ONE
bar the pilots moved, are in PILOT RECORD at the foot of this docstring. Seeds
7 and 90 are disjoint from the registered seeds 0-6 and are now spent.

  RIG (violated -> the seed is not informative; see THE SEED PROTOCOL):
    trap_live      icm(noisy) dwell >= TRAP_DWELL_MIN AND its panel reward
                   ratio >= TRAP_RATIO_MIN. The trap caught the naive agent in
                   THIS run. Without it a claim-arm non-fixation is unreadable:
                   it could mean the panel stopped being seductive.
    saw_panel      claim(noisy) panel ray-exposure >= EXPOSURE_FRAC x
                   random(noisy)'s. THE SUBJECT MUST HAVE ENTERED THE ROOM THE
                   TRAP IS IN. Discovered by the seed-90 pilot, and it is the
                   same vacuity as the position-pseudo-count arm by a different
                   route: PG.4's own certified numbers are icm_dwell 0.6667
                   with std 0.4714, which is exactly the seed vector
                   [1.0, 1.0, 0.0] — the naive agent either finds the panel and
                   locks on ENTIRELY, or never finds it at all. So on ~1 seed
                   in 3 a "non-fixating" arm is an arm that never walked past
                   the TV, and its non-fixation says nothing about capture. A
                   random walk sweeps the arena, so its exposure is the honest
                   yardstick for "had the opportunity".
    null_floor     random(noisy) dwell <= NULL_DWELL_MAX. The dwell metric is
                   measuring fixation, not floor-plan geometry (PG.4's own
                   control question, re-asked here because a metric certified
                   elsewhere is still an assumption until this run reproduces
                   it).
    alive_decay    claim arm in the STATIC world: reward_q1 > 0 and
                   reward_q1 / reward_q4 >= DECAY_MIN. Positive evidence that
                   the signal tracks learnability — it is large when the world
                   is unknown and shrinks as the world is learned. A dead or
                   constant signal fails this, and a dead signal is precisely
                   what would otherwise ace every claim gate below.

  CLAIM (WORST informative seed, not the mean; all four must hold):
    no_fixation    claim(noisy) dwell <= NULL_DWELL_MAX — it stares no more
                   than a random walk does.
    not_fed        claim(noisy) late reward in-zone / out-zone <= FED_RATIO_MAX
                   — the MECHANISM gate. Non-fixation could be luck of the
                   floor plan; this says the noise did not enter the reward,
                   measured. PG.4's null is >= 2.0 here by construction.
    still_explores claim(noisy) visited cell fraction >= EXPLORE_FRAC x
                   zero(noisy) — it does not buy non-fixation by parking in a
                   corner. PG.4 measured exactly that failure mode when it
                   tuned R_RESOLVE: "parked at 5.7 m with 36% map coverage vs
                   the static arm's 74%". Avoiding the TV by refusing to look
                   at the world is not curiosity surviving; it is curiosity
                   dying quietly.
    beats_null     claim(noisy) dwell + MARGIN_MIN <= icm(noisy) dwell — the
                   claim arm and the null are separated in this run, not merely
                   each on their own side of a threshold.

  CONTROL (must fail): `icm` in the noisy world, evaluated against the CLAIM
  gates. The null IS the control, which is the strongest form available here —
  it is the same agent, the same world and the same metrics, differing only in
  the reward, and PG.4 has already certified that it fires.

  SAID OUT LOUD, because a control that cannot fail is decoration: on an
  informative seed the control's failure is ENTAILED by `trap_live`. A seed
  only counts if `icm` dwelt >= 0.40, and `no_fixation` bars dwell > 0.20, so
  `not _claim_holds(c)` is arithmetic, not evidence. The discriminating work is
  therefore done elsewhere and it is worth naming: (a) `not_fed`, whose bar sits
  at the midpoint between an UNFED signal's fixed point (in/out ratio 1.0) and
  PG.4's certified fed null (>= 2.0), so the claim arm has to land on the unfed
  side of a line drawn by mechanism rather than by this pilot; and (b) the `rnd`
  arm, which is equally percept-driven, was a live candidate for `_CLAIM_ARM`,
  and FAILS `not_fed` on both pilot seeds (2.232, 1.888). `rnd` is reported and
  NOT gated — pre-registering "rnd must fail" after watching it fail in the
  pilot would be fitting a control to the data it was drawn from.

THE SEED PROTOCOL — the bimodal trap, and why "worst of N" needs a subset.
PG.4's own certified row is `icm_dwell 0.6667 +- 0.4714`, which is exactly the
seed vector [1.0, 1.0, 0.0] over its registered seeds 0/1/2: the naive agent
either finds the panel and locks on ENTIRELY, or never finds it. The seed-90
pilot reproduced it here (`icm dwell 0.0000`, coverage 0.3967 — it never walked
past the TV). So one seed in ~three carries NO trap, and on such a seed the
claim arm's non-fixation is unreadable rather than good news.

Two ways of handling that are wrong and both were rejected:
  - **Gate the MEAN.** This is what the first cut of this file did by accident:
    `run_spec._aggregate` hands `_check` the mean over seeds, so a docstring
    saying "worst of 3" was scored on `trap_dwell = 0.667` — a bar of 0.40 met
    by two live seeds carrying one dead one. A dead trap would have been
    averaged into a PASS, which is this spec's own vacuity wearing a different
    hat. Fixed: `_fold` folds per-seed rows and `_experiment` returns the fold,
    so every gate below reads the WORST informative seed (T2.19's idiom).
  - **Gate the worst of ALL seeds.** Honest, and it VOIDs with probability
    ~0.96 on 7 seeds for a reason that is not about curiosity at all.

What is pre-registered instead: a seed is INFORMATIVE iff its apparatus worked
on that seed — trap fired, random-walk floor held, the claim arm had panel
exposure, and the claim arm's signal was alive and decaying. That is the whole
RIG block above, evaluated per seed. The claim gates are then scored on the
WORST informative seed, and the run is VOID unless at least
`MIN_LIVE_SEEDS` = 3 of the 7 are informative. The selection criterion is a
fixed formula over the NULL and the RIG — never over the claim arm's dwell,
fed-ratio, coverage or margin — so it cannot drop a seed for being unflattering
to the hypothesis. `informative_seeds` and every per-seed vector are recorded in
the metrics, so an auditor can recompute the subset without re-running anything.

WHICH ARM IS THE CLAIM ARM IS FROZEN BY THE PILOT, NOT CHOSEN BY THE RUN.
`_CLAIM_ARM = "disagree"` is frozen from the pilots and the registered run then
tests that named arm on unseen seeds 0-6. Taking the argmax over {rnd, disagree}
at scoring time would be picking noise and calling it a finding (SYSTEM.md's
margin rule); the loser is reported in metrics but cannot rescue the run.

WHAT THIS REGIME REMOVES, said out loud (LESSONS: list what the chosen regime
removes). Movable clutter (n_objects=0, PG.4 precedent): the panel is the sole
irreducible source, so a fixation is attributable. Collision (contype 0): every
cell is reachable, so coverage has a true ceiling and "did not explore" cannot
mean "was stuck". Extrinsic reward (none anywhere): nothing but curiosity moves
this agent. Locomotion (velocity slider): this spec is about what curiosity
attends to, not about a body — T2.01 owns that and is settled FAIL.

PILOT RECORD (2026-08-29, this box, CPU; artifacts /data/t2_09_pilot_seed7.json
and /data/t2_09_pilot_seed90.json, 20 000 decisions, all eight lives per seed).
Seed 7 fired the trap; seed 90 did not, and per the seed protocol above it can
only speak to the bars that do not depend on a live trap.

  arm (noisy)     dwell s7 / s90   in/out ratio s7 / s90   coverage s7 / s90
  icm  (null)     0.8337 / 0.0000  2.279 / 0.000           0.5950 / 0.3967
  rnd             0.0140 / 0.0520  2.232 / 1.888           1.0000 / 0.9917
  disagree        0.0844 / 0.0404  1.413 / 0.979           1.0000 / 1.0000
  zero            0.0068 / 0.0584  0.000 / 0.000           1.0000 / 1.0000
  random          0.0172 / 0.0113  0.000 / 0.000           1.0000 / 0.9917
  disagree static q1 0.000410 / 0.000332, decay 2.267 / 1.472
  claim exposure  28 441 / 29 593 panel rays = 0.9611 of the random walk's (s7;
                  the s90 pilot predates the `panel_rays` metric)

WHAT THE PILOT DECIDED, bar by bar. Seven of the eight bars were CONFIRMED at
the value they already held and are NOT fitted to these numbers — each is
inherited from PG.4's registered certificate or anchored on a mechanism:
`TRAP_DWELL_MIN` 0.40, `TRAP_RATIO_MIN` 2.0 and `NULL_DWELL_MAX` 0.20 are
PG.4's own certified constants; `FED_RATIO_MAX` 1.5 is the midpoint of the
unfed fixed point 1.0 and PG.4's fed null 2.0; `EXPOSURE_FRAC` 0.50 is "half
the opportunity a random walk had"; `EXPLORE_FRAC` 0.80 and `MARGIN_MIN` 0.15
stand as written. `_CLAIM_ARM = "disagree"` is confirmed rather than chosen:
it clears all four claim gates on seed 7 and `rnd` fails `not_fed` on BOTH
pilot seeds, which is the mechanism the spec predicts (an irreducibly
stochastic channel drives an ensemble to a shared conditional mean, while a
frozen random target stays surprising forever).

ONE BAR MOVED, and downward: `DECAY_MIN` 1.5 -> 1.25. Seed 90's claim-arm
static decay read 1.472, so the placeholder would have made a live, decaying
signal look dead and cost a seed for nothing. 1.25 is not the observed minimum
shaved — it is set from what the gate is FOR: a constant or dead signal has
decay identically 1.0, so any bar above 1.0 excludes it, and 1.25 sits roughly
midway between that fixed point and the weaker of the two pilot readings.
Recorded as a moved threshold in the open, per SYSTEM.md: it is a placeholder
being frozen for the first time, not a registered bar being weakened — this
spec has never run, and `run()` refused until this commit.

THE GATE THAT WILL DECIDE THIS RUN is `not_fed`. The claim arm measured 1.413
against a bar of 1.5 on the seed where the trap fired — 6% of headroom. That is
what a test which could have failed looks like, and it is stated here BEFORE
the registered seeds are drawn so that a FAIL cannot later be narrated as a
surprise.

READING THE REGISTERED ROW's `trap_ratio` COLUMN (added 2026-08-31, 52nd audit
B6 — a display contextualisation, no number moves). `panel_reward_ratio` is
`m_in / max(1e-12, m_out)`, so when an arm collects essentially ZERO reward
outside the panel the ratio is `m_in` divided by the 1e-12 floor and prints as
astronomically large. Seed 1's per-seed `trap_ratio` of ~9.5e11 is exactly
this: a VANISHED DENOMINATOR — the null starved everywhere off-panel — not a
spectacularly strong trap. The gate reads it only as `>= TRAP_RATIO_MIN` (2.0),
for which any denominator collapse is simply "very fed on-panel", so the
verdict is untouched; but a reader comparing magnitudes across seeds must not:
the column is a RATIO WITH A FLOOR, ordinal above ~1e3, not a measurement.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from ..gpu import build_job, submit
from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from .pg_4_noisy_tv import (
    _ACTIONS, _Retina, _build, _cell, _dwell,
    EPS_HI, EPS_LO, GAMMA, N_RAYS, Q_INIT, Q_LR, SPEED, SUBSTEPS,
)

# The trap and the world both hash in: this spec's claim is "survives the trap
# PG.4 certified", which is a statement about PG.4's apparatus as much as ours.
IMPL_DEPS = ["playground.py", "experiments/tests/pg_4_noisy_tv.py"]

_GATES_FROZEN = True            # frozen 2026-08-29 — see PILOT RECORD

# Registered seeds, disjoint from the spent pilots (7, 90). SEVEN, not three:
# the trap is bimodal (see THE SEED PROTOCOL), it fires on ~2 of 3 seeds, and
# three seeds would leave a ~26% chance of fewer than three informative ones.
# At p = 2/3 this design is informative-enough with probability ~0.96. Raising
# the seed count is a strengthening — more evidence, a harder run — so it moves
# under the T1.02 precedent rather than needing a decision.
SEEDS = list(range(7))
MIN_LIVE_SEEDS = 3              # fewer informative seeds -> VOID, not FAIL

N_DECISIONS = 20_000            # PG.4's life length. NOT tunable downward: the
                                # trap's certified fixation (dwell >= 0.40) was
                                # measured over this horizon on the last half,
                                # and a shorter life would let `trap_live` VOID
                                # for want of time rather than for want of a
                                # trap.
N_CELLS = 121                   # PG.4's 11x11 floor grid, for comparability
ENSEMBLE_K = 4                  # `disagree` members
OBS_DIM = 4 + 2 * N_RAYS
LR = 1e-3
BATCH = 64

_CLAIM_ARM = "disagree"         # FROZEN by the pilots; see PILOT RECORD
_ARMS = ("icm", "rnd", "disagree", "zero")

# --- RIG bars (a seed violating any of these is not informative) ---
TRAP_DWELL_MIN = 0.40           # PG.4's ICM_DWELL_MIN, its certified value
TRAP_RATIO_MIN = 2.0            # PG.4's PANEL_REWARD_RATIO_MIN
NULL_DWELL_MAX = 0.20           # PG.4's NULL_DWELL_MAX
DECAY_MIN = 1.25                # MOVED by the pilot, 1.5 -> 1.25: a constant or
                                # dead signal decays by exactly 1.0, so the bar
                                # is set above that fixed point, not shaved to
                                # the weaker pilot reading (1.472). PILOT RECORD.
EXPOSURE_FRAC = 0.50            # claim arm's panel rays vs the random walk's;
                                # the "it had the opportunity" gate

# --- CLAIM bars (worst INFORMATIVE seed) ---
FED_RATIO_MAX = 1.5             # in-zone/out-zone reward, claim arm. Midway
                                # between an unfed signal (1.0) and PG.4's
                                # certified fed null (>= 2.0). Pilot: 1.413.
EXPLORE_FRAC = 0.80             # coverage vs the zero twin
MARGIN_MIN = 0.15               # dwell separation from the null


# ── the four rewards ─────────────────────────────────────────────────────
class _Signal:
    """Percept-driven intrinsic reward. `kind` selects the construction.

    Every kind sees the identical (obs, action, obs2) stream and returns a
    scalar; the Q-learner downstream is byte-identical across arms. That is
    what makes the arms a comparison rather than four experiments.
    """

    def __init__(self, kind: str, seed: int):
        self.kind = kind
        if kind == "zero":
            return
        import torch
        torch.manual_seed(seed)
        self.torch = torch
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inp = OBS_DIM + len(_ACTIONS)

        def mlp(i, o):
            return torch.nn.Sequential(
                torch.nn.Linear(i, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, o)).to(self.dev)

        if kind == "icm":
            self.nets = [mlp(inp, OBS_DIM)]
        elif kind == "disagree":
            # Same data, different init — the only source of disagreement, so
            # convergence to a shared conditional mean is the only way for it
            # to collapse. That collapse under noise IS the hypothesis.
            self.nets = [mlp(inp, OBS_DIM) for _ in range(ENSEMBLE_K)]
        elif kind == "rnd":
            # Target frozen at init: a deterministic function of the retina.
            self.target = mlp(OBS_DIM, 32)
            for p in self.target.parameters():
                p.requires_grad_(False)
            self.nets = [mlp(OBS_DIM, 32)]
        else:
            raise ValueError(kind)
        params = [p for n in self.nets for p in n.parameters()]
        self.opt = torch.optim.Adam(params, lr=LR)
        self.buf_x, self.buf_y, self.buf_o = [], [], []

    def reward(self, obs, a, obs2) -> float:
        """Intrinsic reward for the transition, computed under no_grad."""
        if self.kind == "zero":
            return 0.0
        torch = self.torch
        with torch.no_grad():
            o2 = torch.as_tensor(obs2, device=self.dev)
            if self.kind == "rnd":
                r = float(((self.nets[0](o2) - self.target(o2)) ** 2).sum())
            else:
                x = torch.cat([torch.as_tensor(obs, device=self.dev),
                               torch.eye(len(_ACTIONS), device=self.dev)[a]])
                preds = torch.stack([n(x) for n in self.nets])
                if self.kind == "icm":
                    r = float(((preds[0] - o2) ** 2).sum())
                else:
                    # variance across members, averaged over output dims
                    r = float(preds.var(dim=0, unbiased=False).mean())
        return r

    def learn(self, obs, a, obs2, rng) -> None:
        """One replay step over the full history (PG.4's rule: a short window
        lets the model forget the far side of the map, and revisits then
        masquerade as novelty — the agent chases its own forgetting)."""
        if self.kind == "zero":
            return
        import numpy as np
        torch = self.torch
        self.buf_o.append(obs2)
        if self.kind != "rnd":
            self.buf_x.append(np.concatenate(
                [obs, np.eye(len(_ACTIONS), dtype="float32")[a]]))
            self.buf_y.append(obs2)
        n = len(self.buf_o)
        idx = rng.randint(n, size=min(BATCH, n))
        if self.kind == "rnd":
            bo = torch.from_numpy(np.stack([self.buf_o[i] for i in idx])
                                  ).to(self.dev)
            with torch.no_grad():
                tgt = self.target(bo)
            loss = ((self.nets[0](bo) - tgt) ** 2).sum(dim=1).mean()
        else:
            bx = torch.from_numpy(np.stack([self.buf_x[i] for i in idx])
                                  ).to(self.dev)
            by = torch.from_numpy(np.stack([self.buf_y[i] for i in idx])
                                  ).to(self.dev)
            loss = sum(((n_(bx) - by) ** 2).sum(dim=1).mean()
                       for n_ in self.nets)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


# ── one life ─────────────────────────────────────────────────────────────
_LIFE_CACHE: dict = {}


def _life(seed: int, arm: str, noisy: bool,
          n_decisions: int = N_DECISIONS) -> dict:
    """One life in PG.4's playground under `arm`'s reward. Returns metrics.

    Memoised: `_control` scores the SAME null and zero lives `_experiment`
    already ran, and a life here is a deterministic function of its key (both
    RandomStates are seeded from `seed`, and MuJoCo is deterministic). Without
    this the null is simulated twice per seed — 40% of the run's cost spent
    recomputing a number we hold. The cache is keyed by the full argument
    tuple, so a different budget never reads another budget's life.
    """
    key = (seed, arm, noisy, n_decisions)
    if key in _LIFE_CACHE:
        return _LIFE_CACHE[key]
    import mujoco
    import numpy as np

    model, data, panel_gid, rover_bid, (ax, ay) = _build()
    env_rng = np.random.RandomState(seed * 7919 + 13)
    agent_rng = np.random.RandomState(seed * 104729 + 7)
    retina = _Retina(model, panel_gid, rover_bid, noisy, env_rng)
    sig = _Signal(arm, seed) if arm != "random" else None
    q = np.full((N_CELLS, len(_ACTIONS)), Q_INIT)

    obs, _ = retina.observe(data)
    half = n_decisions // 2
    quarter = max(1, n_decisions // 4)
    dwell_late = 0
    r_in = r_out = 0.0
    n_in = n_out = 0
    r_q1 = r_q4 = 0.0
    n_q1 = n_q4 = 0
    panel_rays = 0
    visited = set()

    for t in range(n_decisions):
        x, y = float(data.qpos[-2]), float(data.qpos[-1])
        s = _cell(x, y)
        visited.add(s)
        if arm == "random":
            a = int(agent_rng.randint(len(_ACTIONS)))
        else:
            eps = max(EPS_LO,
                      EPS_HI - (EPS_HI - EPS_LO) * t / (n_decisions // 3))
            if agent_rng.uniform() < eps:
                a = int(agent_rng.randint(len(_ACTIONS)))
            else:
                best = np.flatnonzero(q[s] >= q[s].max() - 1e-12)
                a = int(best[agent_rng.randint(len(best))])

        data.ctrl[ax] = SPEED * _ACTIONS[a][0]
        data.ctrl[ay] = SPEED * _ACTIONS[a][1]
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        obs2, hits = retina.observe(data)
        panel_rays += hits
        x2, y2 = float(data.qpos[-2]), float(data.qpos[-1])
        in_dwell = _dwell(x2, y2)

        if sig is not None:
            r = sig.reward(obs, a, obs2)
            s2 = _cell(x2, y2)
            q[s, a] += Q_LR * (r + GAMMA * q[s2].max() - q[s, a])
            sig.learn(obs, a, obs2, agent_rng)
            if t < quarter:
                r_q1 += r; n_q1 += 1
            elif t >= n_decisions - quarter:
                r_q4 += r; n_q4 += 1
            if t >= half:
                if in_dwell:
                    r_in += r; n_in += 1
                else:
                    r_out += r; n_out += 1

        if in_dwell and t >= half:
            dwell_late += 1
        obs = obs2

    m_in, m_out = r_in / max(1, n_in), r_out / max(1, n_out)
    out = {
        "dwell_share": round(dwell_late / half, 4),
        "coverage": round(len(visited) / N_CELLS, 4),
        "panel_rays": panel_rays,
        "reward_in": round(m_in, 6),
        "reward_out": round(m_out, 6),
        "panel_reward_ratio": round(m_in / max(1e-12, m_out), 3),
        "reward_q1": round(r_q1 / max(1, n_q1), 6),
        "reward_q4": round(r_q4 / max(1, n_q4), 6),
        "reward_decay": round((r_q1 / max(1, n_q1))
                              / max(1e-12, r_q4 / max(1, n_q4)), 3),
        "final_dist_to_panel": round(
            math.hypot(float(data.qpos[-2]), float(data.qpos[-1]) - 5.9), 3),
    }
    _LIFE_CACHE[key] = out
    return out


def remote_run(seeds: list, n_decisions: int = N_DECISIONS) -> dict:
    """Every life this spec reads, for every seed. Runs on the GPU VM.

    Six lives per seed: the four arms plus the random walk in the noisy world,
    and the claim arm in the STATIC world for `alive_decay`. `zero` is not run
    twice — its reward is 0.0 by construction, so its trajectory cannot depend
    on whether the panel re-randomises, and the seed-7 pilot confirmed the two
    lives byte-identical (dwell 0.0068, coverage 1.0, final dist 0.63 in both).
    """
    out = {"n_decisions": n_decisions, "claim_arm": _CLAIM_ARM,
           "gpu": "cpu", "seeds": []}
    try:
        import torch
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    for seed in seeds:
        row = {"seed": seed}
        for arm in list(_ARMS) + ["random"]:
            row[f"{arm}_noisy"] = _life(seed, arm, True, n_decisions)
        row[f"{_CLAIM_ARM}_static"] = _life(seed, _CLAIM_ARM, False, n_decisions)
        out["seeds"].append(row)
    return out


# ── GPU submission (one per spec — module cache, T2.01 pattern) ──────────
JOB = r'''
import subprocess as _s, sys as _y, os as _o
_s.run([_y.executable, "-m", "pip", "install", "-q", "mujoco"], check=True)
import json
from experiments.tests.t2_09_noisy_tv_control import remote_run
out = remote_run(__SEEDS__, n_decisions=__NDEC__)
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "t209.json"), "w"),
          indent=1)
print("DONE", out["gpu"], flush=True)
'''

_CACHE: dict = {}


def _submit(seeds: list, n_decisions: int = N_DECISIONS) -> dict:
    body = (JOB.replace("__SEEDS__", repr(list(seeds)))
               .replace("__NDEC__", str(int(n_decisions))))
    job = build_job(body)
    # CALIBRATED on the pilot, not guessed (T2.19's rule). The seed-7 pilot ran
    # eight lives in 638.4 s wall on this box's shared ARM cores; the six lives
    # this job runs per seed are icm 79.3 + rnd 81.1 + disagree 169.0 + zero
    # 23.8 + random 23.2 + disagree_static 157.6 = 534.0 s. Priced at 560 s/seed
    # (5% over) plus 0.10 h fixed for clone + `pip install mujoco` + torch
    # import. Kaggle's x86 cores should beat ARM here — MuJoCo stepping and the
    # per-decision torch forwards are both CPU-bound at this size — so the
    # estimate errs long, which is the direction a number feeding a timeout must
    # err in. 7 seeds -> 1.19 h, inside the spec's registered gpu<2h class.
    est_hours = round(0.10 + (560.0 / 3600.0) * len(seeds), 3)
    # DERIVED from the estimate, never sized independently (T2.19's scar: a
    # watcher set to give up at 40% of the run's own predicted length).
    timeout_s = int(est_hours * 3600 * 1.5) + 900
    res = submit(job, prefer="kaggle", est_hours=est_hours,
                 timeout_s=timeout_s, fetch=["t209.json"])
    if not res.ok:
        raise RuntimeError(f"T2.09 job failed on {res.backend}: {res.message}")
    out = json.loads(Path(res.artifacts["t209.json"]).read_text())
    out["backend"] = res.backend
    return out


# ── the reading ──────────────────────────────────────────────────────────
def _seed_view(row: dict) -> dict:
    """One seed's raw lives -> the quantities every gate is stated in."""
    cl = row[f"{_CLAIM_ARM}_noisy"]
    null, zero, rand = row["icm_noisy"], row["zero_noisy"], row["random_noisy"]
    st = row[f"{_CLAIM_ARM}_static"]
    other = "rnd" if _CLAIM_ARM == "disagree" else "disagree"
    v = {
        "seed": row["seed"],
        # claim
        "claim_dwell": cl["dwell_share"],
        "claim_fed_ratio": cl["panel_reward_ratio"],
        "coverage_frac_of_zero": round(
            cl["coverage"] / max(1e-9, zero["coverage"]), 4),
        "dwell_margin_vs_null": round(
            null["dwell_share"] - cl["dwell_share"], 4),
        # rig
        "trap_dwell": null["dwell_share"],
        "trap_fed_ratio": null["panel_reward_ratio"],
        "null_random_dwell": rand["dwell_share"],
        "exposure_frac_of_random": round(
            cl["panel_rays"] / max(1, rand["panel_rays"]), 4),
        "claim_static_reward_q1": st["reward_q1"],
        "claim_static_decay": st["reward_decay"],
        # the null, scored on the CLAIM gates — this seed's control
        "ctrl_claim_dwell": null["dwell_share"],
        "ctrl_claim_fed_ratio": null["panel_reward_ratio"],
        "ctrl_coverage_frac_of_zero": round(
            null["coverage"] / max(1e-9, zero["coverage"]), 4),
        # the losing candidate, reported and unable to rescue the run
        f"other_{other}_dwell": row[f"{other}_noisy"]["dwell_share"],
        f"other_{other}_fed_ratio": row[f"{other}_noisy"]["panel_reward_ratio"],
        f"other_{other}_coverage": row[f"{other}_noisy"]["coverage"],
    }
    # INFORMATIVE: the apparatus worked on this seed. Reads the null, the
    # random walk, the rig instruments AND two claim-arm RIG readings taken
    # on the STATIC panel (`claim_static_reward_q1`, `claim_static_decay` —
    # was the ICM alive, and did it decay where decay is mandatory), plus
    # `exposure_frac_of_random`. It never reads the JUDGED quantities — the
    # claim arm's trap dwell, fed-ratio, coverage or margin — so a seed
    # cannot be dropped for being unflattering to the hypothesis. The
    # summary this comment used to carry ("computed ONLY from the null ...
    # and the rig instruments") was FALSE as written (52nd audit B6): a
    # claim-arm instrument is a claim-arm instrument even when it is read
    # for rig health, and the honest statement is the one above. Live effect
    # of the correction: zero — on the recorded run every exclusion fired on
    # `trap_dwell` alone. `DECAY_MIN` does not move; re-fitting it after the
    # PASS would be the real violation. See THE SEED PROTOCOL.
    v["informative"] = float(
        v["trap_dwell"] >= TRAP_DWELL_MIN
        and v["trap_fed_ratio"] >= TRAP_RATIO_MIN
        and v["null_random_dwell"] <= NULL_DWELL_MAX
        and v["exposure_frac_of_random"] >= EXPOSURE_FRAC
        and v["claim_static_reward_q1"] > 0.0
        and v["claim_static_decay"] >= DECAY_MIN)
    return v


def _fold(rows: list) -> dict:
    """Per-seed rows -> the numbers the gates read: WORST informative seed.

    Never a mean. `run_spec._aggregate` means everything it is handed, so a
    spec whose gates are worst-case must fold before it returns — otherwise
    one dead trap is averaged into a live one and the rig gate passes on a
    seed that tested nothing.
    """
    views = [_seed_view(r) for r in rows]
    live = [v for v in views if v["informative"]]

    def worst(key, hi: bool):
        """hi=True -> the largest value is the worst (a max-bar gate).

        With no informative seed the run is VOID and these numbers are never
        read — but they are still WRITTEN to the ledger, so the sentinel is
        chosen to fail its own gate rather than to be NaN. NaN would be
        non-strict JSON on the way in and would read as an absent measurement
        on the way out; a number that fails loudly cannot be mistaken for one.
        """
        if not live:
            return 1e9 if hi else -1e9
        return (max if hi else min)(v[key] for v in live)

    other = "rnd" if _CLAIM_ARM == "disagree" else "disagree"
    return {
        "claim_arm": _CLAIM_ARM,
        "n_seeds": float(len(views)),
        "n_informative": float(len(live)),
        "informative_seeds": [v["seed"] for v in live],
        # claim, worst informative seed
        "claim_dwell": worst("claim_dwell", True),
        "claim_fed_ratio": worst("claim_fed_ratio", True),
        "coverage_frac_of_zero": worst("coverage_frac_of_zero", False),
        "dwell_margin_vs_null": worst("dwell_margin_vs_null", False),
        # rig, worst informative seed (all hold by construction on `live`;
        # reported so the certificate carries its own margins)
        "trap_dwell": worst("trap_dwell", False),
        "trap_fed_ratio": worst("trap_fed_ratio", False),
        "null_random_dwell": worst("null_random_dwell", True),
        "exposure_frac_of_random": worst("exposure_frac_of_random", False),
        "claim_static_reward_q1": worst("claim_static_reward_q1", False),
        "claim_static_decay": worst("claim_static_decay", False),
        # the loser, reported and ungated
        f"other_{other}_fed_ratio": worst(f"other_{other}_fed_ratio", True),
        f"other_{other}_dwell": worst(f"other_{other}_dwell", True),
        # every seed, informative or not — the subset must be recomputable
        "per_seed": [[v["seed"], v["informative"], v["trap_dwell"],
                      v["trap_fed_ratio"], v["exposure_frac_of_random"],
                      v["claim_static_decay"], v["claim_dwell"],
                      v["claim_fed_ratio"], v["coverage_frac_of_zero"],
                      v["dwell_margin_vs_null"]] for v in views],
        "per_seed_cols": ("seed informative trap_dwell trap_ratio exposure "
                          "static_decay claim_dwell claim_fed claim_cov "
                          "margin"),
    }


def _fold_control(rows: list) -> dict:
    """The null scored on the CLAIM gates, over the SAME informative seeds."""
    live = [v for v in (_seed_view(r) for r in rows) if v["informative"]]
    if not live:
        # Sentinels that make the control PASS the claim gates, i.e. that make
        # `not _claim_holds(c)` false. Same rule as `_fold.worst`: with nothing
        # measured, every gate must read as unsatisfied, and for the control
        # "unsatisfied" is the direction that blocks a PASS.
        return {"claim_dwell": 0.0, "claim_fed_ratio": 0.0,
                "coverage_frac_of_zero": 1e9, "n_informative": 0.0}
    return {
        "claim_dwell": max(v["ctrl_claim_dwell"] for v in live),
        "claim_fed_ratio": max(v["ctrl_claim_fed_ratio"] for v in live),
        "coverage_frac_of_zero": min(v["ctrl_coverage_frac_of_zero"]
                                     for v in live),
        "n_informative": float(len(live)),
        "ctrl_per_seed_dwell": [v["ctrl_claim_dwell"] for v in live],
    }


def _experiment(seed: int) -> dict:
    """`seed` is ignored: one submission runs every seed, and `_fold` reduces
    them to the worst informative one. run_spec calls this once per registered
    seed and means identical dicts, so the recorded numbers are the fold."""
    if not _CACHE:
        _CACHE.update(_submit(SEEDS))
    m = _fold(_CACHE["seeds"])
    m["gpu"] = _CACHE["gpu"]
    m["backend"] = _CACHE.get("backend", "?")
    return m


def _control(seed: int) -> dict:
    return _fold_control(_CACHE["seeds"])


def _claim_holds(m: dict) -> bool:
    """The claim gates alone — applied to the run AND to the control."""
    return (m["claim_dwell"] <= NULL_DWELL_MAX
            and m["claim_fed_ratio"] <= FED_RATIO_MAX
            and m["coverage_frac_of_zero"] >= EXPLORE_FRAC)


def _check(m: dict, c: dict):
    # Too few working seeds is an APPARATUS outcome, not a refutation: the
    # bimodal trap failed to fire often enough for the claim to be readable.
    # FAIL would fire this spec's `kills` field off a run that never asked the
    # question.
    if m["n_informative"] < MIN_LIVE_SEEDS:
        return Status.VOID
    rig = (m["trap_dwell"] >= TRAP_DWELL_MIN
           and m["trap_fed_ratio"] >= TRAP_RATIO_MIN
           and m["null_random_dwell"] <= NULL_DWELL_MAX
           and m["exposure_frac_of_random"] >= EXPOSURE_FRAC
           and m["claim_static_reward_q1"] > 0.0
           and m["claim_static_decay"] >= DECAY_MIN)
    claim = (_claim_holds(m)
             and m["dwell_margin_vs_null"] >= MARGIN_MIN)
    return bool(rig and claim and not _claim_holds(c))


def run(ledger: Ledger | None = None):
    if not _GATES_FROZEN:
        raise RuntimeError(
            "T2.09 gates are provisional — pilot first, freeze the bars in "
            "this file, then run (SM.02's _GATES_FROZEN idiom).")
    return run_spec(BY_ID["T2.09"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
