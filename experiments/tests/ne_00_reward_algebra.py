"""NE.00 — The homeostatic reward algebra is what we think it is.

Exact value iteration on two tabular drive MDPs, no MuJoCo, no torch, no body.
This settles the reward FORM before anything trains under it: drive reduction
r = d(h) - d(h') is PBRS-equivalent to a cost of deviation on continuing tasks,
is not farmable, is the unique self-termination-safe member of its family once
death is reachable, and loses all three properties the moment anyone clips it.
Registered 2026-08-24 from docs/research/NEEDS_AND_DEATH.md §7 (queue top).

THE PILOT DID NOT SURVIVE, AND THIS IS DECLARED. The registry notes record a
2026-08-09 pilot (scratchpad/drive_algebra4.py) whose scratchpad is gone. Its
QUALITATIVE predictions are what this implementation gates on; its PINNED
COUNTS (suicide 11/11 col, 8/11 cc, 0/11 dr; rho_first at 0.70 x d_max; best
cycle -0.0045) were properties of that pilot's exact parameterisation and are
REPORTED here, gated only where the claim's direction lives:

  suicide_col == N_LIVING   (all 11 living states — the pathology is total)
  suicide_dr  == 0          (drive reduction is self-termination-safe)
  0 < suicide_cc < N_LIVING (cost of deviation strictly between — it diverges
                             from DR exactly as prediction (d) requires)
  0 < rho_first <= d_max    (a finite survival bonus reconciles cc with dr)

Gating 8/11 exactly would gate on an unrecoverable script's incidental
constants; gating the direction is what the hypothesis's (d) actually asserts.
The measured counts land in the ledger row either way.

## THE TWO MDPS

MDP-A (continuing, 35 states — the registry's "energy x integrity, foraging
feeds but injures"): energy e in 0..6, integrity i in 0..4, h = (e/6, i/4),
d(h) = ((1-h0)^4 + (1-h1)^4)^(1/2)  (Keramati-Gutkin, n=4, m=2 — the same
exponents LC.00 shipped). Each step energy depletes by 1 (floor 0; CONTINUING,
no death). REST: integrity +1 (cap). FORAGE: energy +3 (cap), and with
probability 0.5 integrity -1 (floor) — feeding injures, which is what makes
the optimal policy non-constant (rest when fed, forage when starving) and the
MDP capable of discriminating reward forms at all.

MDP-B (death chain, 11 living states + DEAD): i in 0..10, h = i/10,
d = (1-h)^2. EAT: i -> min(10, i+1). WAIT: i -> i-1; from i=0, WAIT is death.
d(DEAD) := d_max = 1.0 — homeostatic death occurs at MAXIMUM deviation, which
is precisely the PBRS-precondition violation (Phi(terminal) != 0, Grzes 2017)
prediction (d) rests on.

## THE FIVE PREDICTIONS AND THEIR GATES (pre-registered; _check written
## before the first run of this file)

(a) IDENTITY. On MDP-A the greedy-action SETS of r_DR = d-d' and r_CC = -d'
    are identical in every state at every gamma in {0.9, 0.95, 0.99}
    (dr_cc_identical == 1). Sets, not argmaxes: PBRS shifts Q by a state-only
    term and (1-gamma) scales it, so even ties must agree.
(b) TELESCOPING. Over 2,000 random CLOSED state paths in MDP-A, the
    undiscounted DR return telescopes to exactly d_0 - d_T = 0:
    telescope_max_abs <= 1e-9.
(c) NOT FARMABLE. 32 random closed cycles from MDP-A's setpoint, discounted
    at 0.95: every cycle scores strictly below staying at setpoint
    (cycle_best_discounted < 0).
(d) DEATH DIVERGENCE. MDP-B suicide counts as gated above, plus the rho
    sweep: the first rho on a 5%-of-d_max grid where cc+rho's greedy policy
    equals DR's, reported as rho_first_frac.
(e) CLIPPING BREAKS IT. On MDP-B's deplete-3-then-eat-3 closed cycle,
    max(0, d-d') (NetHackEat's shipped form) is strictly net positive
    (clip_cycle_return > 0) while exact DR on the same cycle is 0
    (|dr_cycle_return| <= 1e-9).

## VALIDITY GATES (Status.VOID, not FAIL — the registry's own control text:
## an MDP that cannot tell two rewards apart proves nothing)

  policy_nonconstant == 1   the DR-optimal policy uses both actions somewhere
  event_differs_min > 0     the +1-per-consumption-event reward (the declared
                            non-potential reference) produces a DIFFERENT
                            greedy policy at EVERY gamma (computed in
                            _control; this is the discrimination control)

## ERROR, NOT FAIL — the m/n inequality direction

The reward of a fixed one-unit intake must be strictly LARGER when more
deprived (checked on MDP-B at i=2 vs i=8, and monotone over the chain).
Two of three literature sources misprint the exponent inequality; implementing
from either builds a risk-seeking agent. A reversed direction here is an
implementation defect, so it RAISES (run_spec records ERROR), never FAILs.
With n=4, m=2 (n/m = 2 > 1, convex) the check passes; flip the exponents and
it fires — verified by hand before registration.

Seeds vary only the stochastic scans in (b) and (c); the VI computations are
exact and identical across seeds, so their _std aggregates are legitimately 0.
"""
from __future__ import annotations

import random

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

GAMMAS = (0.9, 0.95, 0.99)
GAMMA_D = 0.95           # the declared gamma for (c), (d) and the rho sweep
N_EXP, M_EXP = 4, 2      # Keramati-Gutkin exponents; n > m >= 1 asserted
VI_TOL = 1e-12
TIE_TOL = 1e-9
N_PATHS = 2000           # (b)
N_CYCLES = 32            # (c)
RHO_GRID = 20            # (d): rho in {0.05, 0.10, ..., 1.00} x d_max

# MDP-A geometry
E_MAX, I_MAX = 6, 4      # 7 x 5 = 35 states
FORAGE_GAIN = 3
P_INJURY = 0.5

# MDP-B geometry
N_LIVING = 11            # i = 0..10
EAT_GAIN = 1


# ── MDP-A: continuing two-need world ────────────────────────────────────────

def _d_a(e: int, i: int) -> float:
    h0, h1 = e / E_MAX, i / I_MAX
    return ((1.0 - h0) ** N_EXP + (1.0 - h1) ** N_EXP) ** (1.0 / M_EXP)


def _states_a():
    return [(e, i) for e in range(E_MAX + 1) for i in range(I_MAX + 1)]


def _trans_a(s, a):
    """[(prob, s', consumption_event)] — exact distribution, REST=0, FORAGE=1."""
    e, i = s
    e2 = max(0, e - 1)
    if a == 0:                                   # REST
        return [(1.0, (e2, min(I_MAX, i + 1)), 0.0)]
    e3 = min(E_MAX, e2 + FORAGE_GAIN)            # FORAGE
    event = 1.0 if e3 > e2 else 0.0
    return [(P_INJURY, (e3, max(0, i - 1)), event),
            (1.0 - P_INJURY, (e3, i), event)]


def _reward_a(form: str, s, s2, event: float) -> float:
    if form == "dr":
        return _d_a(*s) - _d_a(*s2)
    if form == "cc":
        return -_d_a(*s2)
    if form == "event":
        return event
    raise ValueError(form)


def _vi_a(form: str, gamma: float):
    """Exact VI on MDP-A. Returns per-state greedy-action SETS (ties kept)."""
    states = _states_a()
    v = {s: 0.0 for s in states}
    while True:
        delta = 0.0
        for s in states:
            best = max(
                sum(p * (_reward_a(form, s, s2, ev) + gamma * v[s2])
                    for p, s2, ev in _trans_a(s, a))
                for a in (0, 1))
            delta = max(delta, abs(best - v[s]))
            v[s] = best
        if delta < VI_TOL:
            break
    greedy = {}
    for s in states:
        q = [sum(p * (_reward_a(form, s, s2, ev) + gamma * v[s2])
                 for p, s2, ev in _trans_a(s, a)) for a in (0, 1)]
        top = max(q)
        greedy[s] = frozenset(a for a in (0, 1) if q[a] >= top - TIE_TOL)
    return greedy


# ── MDP-B: the death chain ───────────────────────────────────────────────────

def _d_b(i: int) -> float:
    return (1.0 - i / (N_LIVING - 1)) ** (N_EXP / M_EXP)


D_MAX_B = 1.0            # d at the death boundary (h = 0), by construction

# actions: 0 = WAIT (i-1; death from i=0), 1 = EAT (min(10, i+1))


def _reward_b(form: str, i: int, i2, rho: float = 0.0) -> float:
    d2 = D_MAX_B if i2 is None else _d_b(i2)
    if form == "col":
        return -0.1
    if form == "cc":
        return rho - d2
    if form == "dr":
        return _d_b(i) - d2
    raise ValueError(form)


def _vi_b(form: str, gamma: float, rho: float = 0.0):
    """Exact VI on MDP-B (DEAD is absorbing, value 0). Greedy sets per state."""
    v = [0.0] * N_LIVING
    while True:
        delta = 0.0
        for i in range(N_LIVING):
            i_wait = i - 1 if i > 0 else None
            i_eat = min(N_LIVING - 1, i + EAT_GAIN)
            q_wait = _reward_b(form, i, i_wait, rho) + (
                gamma * v[i_wait] if i_wait is not None else 0.0)
            q_eat = _reward_b(form, i, i_eat, rho) + gamma * v[i_eat]
            best = max(q_wait, q_eat)
            delta = max(delta, abs(best - v[i]))
            v[i] = best
        if delta < VI_TOL:
            break
    greedy = []
    for i in range(N_LIVING):
        i_wait = i - 1 if i > 0 else None
        i_eat = min(N_LIVING - 1, i + EAT_GAIN)
        q = (_reward_b(form, i, i_wait, rho) + (
                gamma * v[i_wait] if i_wait is not None else 0.0),
             _reward_b(form, i, i_eat, rho) + gamma * v[i_eat])
        top = max(q)
        greedy.append(frozenset(a for a in (0, 1) if q[a] >= top - TIE_TOL))
    return greedy


def _suicide_count(greedy) -> int:
    """States where WAIT — the path that ends in death — is STRICTLY optimal
    (WAIT in the greedy set, EAT not)."""
    return sum(1 for g in greedy if g == frozenset((0,)))


# ── the experiment ───────────────────────────────────────────────────────────

def _experiment(seed: int) -> dict:
    rng = random.Random(f"ne00-{seed}")
    states = _states_a()
    m: dict = {}

    # the m/n inequality direction — ERROR, not FAIL, when reversed
    intake = [_d_b(i) - _d_b(min(N_LIVING - 1, i + 1))
              for i in range(N_LIVING - 1)]
    if not (intake[2] > intake[8] and
            all(intake[k] >= intake[k + 1] for k in range(len(intake) - 1))):
        raise RuntimeError(
            "m/n inequality direction reversed: a fixed intake must reward "
            f"MORE when more deprived; intake-by-state {intake!r}. This is an "
            "implementation defect (check drive exponents n=%d m=%d), not a "
            "refuted hypothesis." % (N_EXP, M_EXP))

    # (a) identity + non-constancy across all gammas
    identical = 1.0
    nonconstant = 1.0
    for g in GAMMAS:
        pol_dr = _vi_a("dr", g)
        pol_cc = _vi_a("cc", g)
        if any(pol_dr[s] != pol_cc[s] for s in states):
            identical = 0.0
        acts = set()
        for s in states:
            acts |= pol_dr[s]
        if acts != {0, 1}:
            nonconstant = 0.0
    m["dr_cc_identical"] = identical
    m["policy_nonconstant"] = nonconstant

    # (b) undiscounted telescoping over random closed paths
    worst = 0.0
    for _ in range(N_PATHS):
        length = rng.randrange(2, 21)
        path = [rng.choice(states) for _ in range(length)]
        path.append(path[0])
        total = sum(_d_a(*path[t]) - _d_a(*path[t + 1])
                    for t in range(len(path) - 1))
        worst = max(worst, abs(total))
    m["telescope_max_abs"] = worst

    # (c) discounted closed cycles from the setpoint score below staying put
    setpoint = (E_MAX, I_MAX)
    others = [s for s in states if s != setpoint]
    best_cycle = float("-inf")
    for _ in range(N_CYCLES):
        mid = [rng.choice(others) for _ in range(rng.randrange(1, 10))]
        path = [setpoint] + mid + [setpoint]
        ret = sum((GAMMA_D ** t) * (_d_a(*path[t]) - _d_a(*path[t + 1]))
                  for t in range(len(path) - 1))
        best_cycle = max(best_cycle, ret)
    m["cycle_best_discounted"] = best_cycle

    # (d) death divergence
    m["suicide_col"] = float(_suicide_count(_vi_b("col", GAMMA_D)))
    m["suicide_cc"] = float(_suicide_count(_vi_b("cc", GAMMA_D)))
    m["suicide_dr"] = float(_suicide_count(_vi_b("dr", GAMMA_D)))
    pol_dr_b = _vi_b("dr", GAMMA_D)
    rho_first = 0.0
    for k in range(1, RHO_GRID + 1):
        rho = k / RHO_GRID * D_MAX_B
        if _vi_b("cc", GAMMA_D, rho=rho) == pol_dr_b:
            rho_first = rho
            break
    m["rho_first_frac"] = rho_first / D_MAX_B

    # (e) clipping makes the deplete-and-eat cycle net positive
    cyc = [10, 9, 8, 7, 8, 9, 10]
    dr_ret = sum(_d_b(cyc[t]) - _d_b(cyc[t + 1]) for t in range(len(cyc) - 1))
    clip_ret = sum(max(0.0, _d_b(cyc[t]) - _d_b(cyc[t + 1]))
                   for t in range(len(cyc) - 1))
    m["dr_cycle_return"] = dr_ret
    m["clip_cycle_return"] = clip_ret

    confirmed = sum((
        identical == 1.0,
        worst <= 1e-9,
        best_cycle < 0.0,
        (m["suicide_col"] == float(N_LIVING) and m["suicide_dr"] == 0.0
         and 0.0 < m["suicide_cc"] < float(N_LIVING)
         and 0.0 < m["rho_first_frac"] <= 1.0),
        clip_ret > 0.0 and abs(dr_ret) <= 1e-9,
    ))
    m["reward_algebra_predictions_confirmed"] = float(confirmed)
    return m


def _control(seed: int) -> dict:
    """The discrimination control: the +1-per-consumption-event reward (a
    non-potential form) must induce a DIFFERENT greedy policy than drive
    reduction at EVERY gamma. If it cannot, the MDP tells reward forms apart
    nowhere and every equality in _experiment is vacuous -> VOID."""
    states = _states_a()
    differ_fracs = []
    for g in GAMMAS:
        pol_dr = _vi_a("dr", g)
        pol_ev = _vi_a("event", g)
        differ_fracs.append(
            sum(1 for s in states if pol_dr[s] != pol_ev[s]) / len(states))
    return {"event_differs_min": min(differ_fracs),
            "event_differs_mean": sum(differ_fracs) / len(differ_fracs)}


def _check(m: dict, c: dict):
    # validity first: an instrument that cannot discriminate proves nothing
    if m.get("policy_nonconstant", 0.0) != 1.0:
        return Status.VOID
    if c.get("event_differs_min", 0.0) <= 0.0:
        return Status.VOID

    return (m.get("dr_cc_identical") == 1.0
            and m.get("telescope_max_abs", 1.0) <= 1e-9
            and m.get("cycle_best_discounted", 0.0) < 0.0
            and m.get("suicide_col") == float(N_LIVING)
            and m.get("suicide_dr") == 0.0
            and 0.0 < m.get("suicide_cc", 0.0) < float(N_LIVING)
            and 0.0 < m.get("rho_first_frac", 0.0) <= 1.0
            and m.get("clip_cycle_return", 0.0) > 0.0
            and abs(m.get("dr_cycle_return", 1.0)) <= 1e-9)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["NE.00"], _experiment, _check,
                    control_fn=_control, ledger=ledger or Ledger())
