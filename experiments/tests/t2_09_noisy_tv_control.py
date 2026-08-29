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

PRE-REGISTERED GATES. Bars marked PILOT are placeholders until a seed-90 pilot
freezes them (_GATES_FROZEN); run() refuses until then, SM.02's idiom.

  RIG (violated -> VOID, "the apparatus did not test the claim"):
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

  CLAIM (worst of 3 seeds; all four must hold):
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

WHICH ARM IS THE CLAIM ARM IS FROZEN BY THE PILOT, NOT CHOSEN BY THE RUN.
`_CLAIM_ARM` is set from the seed-90 pilot and the registered run then tests
that named arm on unseen seeds 0/1/2. Taking the argmax over {rnd, disagree} at
scoring time would be picking noise and calling it a finding (SYSTEM.md's
margin rule); the loser is reported in metrics but cannot rescue the run.

WHAT THIS REGIME REMOVES, said out loud (LESSONS: list what the chosen regime
removes). Movable clutter (n_objects=0, PG.4 precedent): the panel is the sole
irreducible source, so a fixation is attributable. Collision (contype 0): every
cell is reachable, so coverage has a true ceiling and "did not explore" cannot
mean "was stuck". Extrinsic reward (none anywhere): nothing but curiosity moves
this agent. Locomotion (velocity slider): this spec is about what curiosity
attends to, not about a body — T2.01 owns that and is settled FAIL.
"""
from __future__ import annotations

import math

from ..protocol import Ledger, run_spec
from ..registry import BY_ID
from .pg_4_noisy_tv import (
    _ACTIONS, _Retina, _build, _cell, _dwell,
    EPS_HI, EPS_LO, GAMMA, N_RAYS, Q_INIT, Q_LR, SPEED, SUBSTEPS,
)

# The trap and the world both hash in: this spec's claim is "survives the trap
# PG.4 certified", which is a statement about PG.4's apparatus as much as ours.
IMPL_DEPS = ["playground.py", "experiments/tests/pg_4_noisy_tv.py"]

_GATES_FROZEN = False           # pilot (seed 90) freezes the bars below

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

_CLAIM_ARM = "disagree"         # PILOT freezes this; see docstring
_ARMS = ("icm", "rnd", "disagree", "zero")

# --- RIG bars (VOID if violated) ---
TRAP_DWELL_MIN = 0.40           # PILOT — PG.4's ICM_DWELL_MIN, its certified value
TRAP_RATIO_MIN = 2.0            # PILOT — PG.4's PANEL_REWARD_RATIO_MIN
NULL_DWELL_MAX = 0.20           # PILOT — PG.4's NULL_DWELL_MAX
DECAY_MIN = 1.5                 # PILOT — claim arm's static reward q1/q4
EXPOSURE_FRAC = 0.50            # PILOT — claim arm's panel rays vs random's;
                                # the "it had the opportunity" gate

# --- CLAIM bars (worst seed) ---
FED_RATIO_MAX = 1.5             # PILOT — in-zone/out-zone reward, claim arm
EXPLORE_FRAC = 0.80             # PILOT — coverage vs the zero twin
MARGIN_MIN = 0.15               # PILOT — dwell separation from the null


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


def _experiment(seed: int) -> dict:
    """Claim arm + null + liveness instruments, one seed."""
    noisy = {a: _life(seed, a, True) for a in _ARMS}
    noisy["random"] = _life(seed, "random", True)
    static_claim = _life(seed, _CLAIM_ARM, False)

    cl, null, zero = noisy[_CLAIM_ARM], noisy["icm"], noisy["zero"]
    out = {
        "claim_arm": _CLAIM_ARM,
        # claim
        "claim_dwell": cl["dwell_share"],
        "claim_fed_ratio": cl["panel_reward_ratio"],
        "claim_coverage": cl["coverage"],
        "zero_coverage": zero["coverage"],
        "coverage_frac_of_zero": round(
            cl["coverage"] / max(1e-9, zero["coverage"]), 4),
        "dwell_margin_vs_null": round(null["dwell_share"] - cl["dwell_share"], 4),
        # rig
        "trap_dwell": null["dwell_share"],
        "trap_fed_ratio": null["panel_reward_ratio"],
        "null_random_dwell": noisy["random"]["dwell_share"],
        "claim_panel_rays": cl["panel_rays"],
        "random_panel_rays": noisy["random"]["panel_rays"],
        "exposure_frac_of_random": round(
            cl["panel_rays"] / max(1, noisy["random"]["panel_rays"]), 4),
        "trap_panel_rays": null["panel_rays"],
        "claim_static_decay": static_claim["reward_decay"],
        "claim_static_reward_q1": static_claim["reward_q1"],
        "claim_static_coverage": static_claim["coverage"],
    }
    # the losing candidate, reported and unable to rescue the run
    other = "rnd" if _CLAIM_ARM == "disagree" else "disagree"
    out[f"other_{other}_dwell"] = noisy[other]["dwell_share"]
    out[f"other_{other}_fed_ratio"] = noisy[other]["panel_reward_ratio"]
    out[f"other_{other}_coverage"] = noisy[other]["coverage"]
    return out


def _control(seed: int) -> dict:
    """The null IS the control: naive percept-driven ICM in the noisy world,
    scored on the CLAIM gates. It must fail them."""
    icm = _life(seed, "icm", True)
    zero = _life(seed, "zero", True)
    return {
        "claim_dwell": icm["dwell_share"],
        "claim_fed_ratio": icm["panel_reward_ratio"],
        "claim_coverage": icm["coverage"],
        "zero_coverage": zero["coverage"],
        "coverage_frac_of_zero": round(
            icm["coverage"] / max(1e-9, zero["coverage"]), 4),
    }


def _claim_holds(m: dict) -> bool:
    """The claim gates alone — applied to the run AND to the control."""
    return (m["claim_dwell"] <= NULL_DWELL_MAX
            and m["claim_fed_ratio"] <= FED_RATIO_MAX
            and m["coverage_frac_of_zero"] >= EXPLORE_FRAC)


def _check(m: dict, c: dict) -> bool:
    rig = (m["trap_dwell"] >= TRAP_DWELL_MIN
           and m["trap_fed_ratio"] >= TRAP_RATIO_MIN
           and m["null_random_dwell"] <= NULL_DWELL_MAX
           and m["exposure_frac_of_random"] >= EXPOSURE_FRAC
           and m["claim_static_reward_q1"] > 0.0
           and m["claim_static_decay"] >= DECAY_MIN)
    claim = (_claim_holds(m)
             and m["dwell_margin_vs_null"] >= MARGIN_MIN)
    return rig and claim and not _claim_holds(c)


def run(ledger: Ledger | None = None):
    if not _GATES_FROZEN:
        raise RuntimeError(
            "T2.09 gates are provisional — pilot (seed 90) first, freeze the "
            "bars in this file, then run (SM.02's _GATES_FROZEN idiom).")
    return run_spec(BY_ID["T2.09"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
