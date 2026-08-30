"""w_1_heat_balance.py — W.1: Temperature obeys the heat balance we published.

The World seat is held BY VERDICT on a *speed* measurement (4-6x faster than
Craftax) and its FIDELITY has never been measured. Five instruments now say W0
is the bottleneck (LC.03's darkroom control, LC.03 v2's one-learner-in-five,
DP.05's FAIL, SH.01's ORACLE_CANNOT, SH.02's saturated null). This is the first
spec that asks the world whether it obeys its own published law.

WHAT IS UNDER TEST, AND WHAT IS NOT
-----------------------------------
The overlay under test is the one SHIPPED in `experiments/needs.py`:

    C_eff * dT/dt = M(T, P) - k_eff(wet, occ) * (T - T_env) - E_evap(wet, T)

That is a lumped-capacitance heat balance, so the comparison the spec asks for
is well posed. The registry's REGISTRATION NOTE binds: *gate the shipped code;
do not write a second thermal model to pass this spec.* The reference model
below is therefore a **yardstick, not a world** — it is never stepped inside
W0, nothing else imports it, and it exists only to reproduce the four numbers
the spec pre-registered. `experiments/thermal.py` (PS.02's linear overlay) is a
DIFFERENT model for a different spec and is deliberately not touched here.

THE REFERENCE, AND ITS SOURCES
------------------------------
Constants are sourced, not invented (the registry `notes` field is the citation
of record):

    1 met      = 58.2 W/m2            Du Bois A = 1.8481 m2 at 175 cm / 70 kg
    h_r        = 4.7 W/m2K            h_c = 3.0 natural, 8.6*v^0.53 forced
    T_skin     = 33.7 C neutral       T_core = 36.8 C neutral (Gagge two-node,
                                      as shipped in CBE pythermalcomfort)
    c_p        = 3470 J/kg/K          *** BURTON'S 1935 ASSUMPTION. NEVER
                                      MEASURED. *** Kept for reconcilability
                                      with ASHRAE/Gagge. The MEASURED value is
                                      2980 J/kg/K (Xu, Rioux & Castellani,
                                      Temperature 2022,
                                      doi:10.1080/23328940.2022.2088034), which
                                      shortens every time constant by 14%. Both
                                      are reported; the registered bars were
                                      derived against Burton, so Burton gates.

ALL THREE REGISTERED CONSTANTS RECONSTRUCT — BUT ONE IS MISLABELLED
-------------------------------------------------------------------
Recomputed from the sourced constants above, before the shipped model was
touched at all. hA = 7.7*1.8481 = 14.2304 W/K, M = 58.2*1.8481 = 107.559 W,
tau = 70*3470/hA = 17069.1 s:

    (a) 20 + M/hA                             -> 27.5584 C   registered 27.55  OK
    (b) 20 + 17*exp(-3600/tau)                -> 33.7674 C   registered 33.767 OK
    (c) (4.7+3.0) / (4.7+8.6*5^0.53)          -> 0.30947     registered 0.3095 OK

**Check (a) is not a thermoneutral AMBIENT, and the registry's wording says it
is.** `20 + M/hA` is the steady-state BODY temperature of a 1-met nude body in
the 20 C still air of check (b)'s scenario. Read as an ambient it is
unreconstructable: it would require a neutral body temperature of 35.108 C,
which is neither the neutral skin (33.7, giving 26.14) nor the neutral core
(36.8, giving 29.24) this spec cites, and no combination of the sourced
constants produces it. Read as a body temperature it is exact to four
significant figures and it makes checks (a) and (b) two readings — steady state
and 1-hour transient — of one scenario.

**The BAR IS NOT MOVED: 27.55 +/- 1.0 C, exactly as registered.** Only the
label is corrected, and the correction is verified by the registered CONTROL,
which under this reading passes (a) and (b) to 4 and 6 significant figures
respectively and fails (c) — precisely the behaviour the registry demands of
it. Under the "ambient" reading the control could not pass (a) at all, and a
control that cannot satisfy its own registered clause is the tell. Both
alternative readings ship as diagnostics (`a_ref_ambient_from_*`) so the next
reader can re-derive this in one line instead of trusting the paragraph.

WHAT THE MEASUREMENT FOUND (recorded so a reader need not re-derive it)
----------------------------------------------------------------------
1. THE FORM IS RIGHT. The shipped overlay is a real heat balance, not a
   thermometer: its equilibrium moves with k — occluding the sky 0 -> 0.9 cuts
   k_eff by OCC_CUT and moves the settled body from 34.00 to 38.92 C in the
   same 20 C air, a 4.92 C shift — so `falsified_by`'s "equilibrium independent
   of h" clause does not fire. Integrated net flux equals C_eff*dT with a
   convergence ratio of exactly 2.000 under dt halving.

2. IT IS 71x FASTER THAN PHYSIOLOGY, AND THAT IS W.7's TRANSFORMATION, NOT AN
   ERROR. Shipped open-loop tau = C_EFF/K_DRY = 240.0 s against the reference's
   17069.1 s -> 1/71.1. The world's declared day compression is 86400/SIM_DAY_S
   = 86400/1200 = 72.0. The two agree to 1.2%: the world is SELF-CONSISTENTLY
   time-compressed. Reported, never gated — the spec told us to run at 1x
   wall-clock and hand the compression to W.7, and this is the number W.7 will
   need.

2b. THE DECLARED TIME CONSTANT IS NOT THE ONE THE WORLD RELAXES WITH.
   `TAU_T = 240 s` is derived as C_EFF/K_DRY, which assumes a constant heat
   source. Shivering is proportional feedback with gain C_SH = 33.33 W/C
   against K_DRY = 14.29 W/C, so the closed-loop conductance is 47.62 W/C and
   the MEASURED relaxation is 72.0 s — 3.33x faster than the constant the world
   publishes, whenever the body is below 37 C, which at night is always.
   Reported, not gated: check (b) fails at 20.0 C on the measured constant and
   at 20.0000052 C on the declared one, against 33.767, so no verdict here
   turns on the distinction.

3. IT HAS NO WIND TERM AT ALL. `k_eff(skin_wetness, sky_occlusion)` takes no
   velocity, and `wind` does not occur anywhere in `experiments/*.py`. Raising
   wind 0 -> 5 m/s therefore changes the shipped time constant by exactly
   nothing: ratio 1.0 against the registered 0.3095. **The shipped world is
   structurally identical to this spec's own deliberately-broken control on
   check (c).** A windy night is not colder than a still night in W0, so no
   policy can ever learn to seek a wind-break for being a wind-break. (W0's
   shelter is not thereby decorative — `needs.py` shelters via `sky_occlusion`
   cutting `k_eff` by `OCC_CUT`, which does work. The missing affordance is
   wind specifically; W.3 is the spec that will price it.)

4. THE FINDING WITH THE LADDER ATTACHED: AT THE WORLD'S OWN NIGHT AMBIENT,
   COLD CANNOT KILL. Shivering in `needs.py` has gain C_SH = SHIVER_CAP/6 =
   33.33 W/C against K_DRY = 14.29 W/C, so the closed loop

       M_BASAL + C_SH*(37 - T) = K_DRY*(T - T_env)

   parks the body at T = (100 + 33.33*37 + 14.286*T_env)/47.62, i.e. **34.000 C
   in a 20 C ambient — flat, forever**, reached in ~50 closed-loop tau and drift
   over the following hour under the 0.01 C diagnostic bar. The reference in
   the same still air settles at 27.56 C, below the world's OWN declared
   incapacitation threshold `T_COLD_DEATH = 28.0`. Solving the shipped loop for
   T = 28 gives a lethal ambient of **exactly 0.0 C**, and the world's night is
   `T_DAY - DELTA_T_NIGHT` = 30 - 10 = **20 C**. So a night in the open is
   survivable indefinitely by a body that does nothing at all.

   This is a DESIGN CHOICE, not a defect: `needs.py` states it in the open
   ("a night in the open equilibrates ~3.0 C cold ... survivable once, costly",
   the §2.3 pedagogy) and NE.01 calibrated `DELTA_T_NIGHT` 12 -> 10 to sit
   mid-band. W.1 does not overturn it; W.1 prices it. GOAL.md's curriculum
   claim is *"cold nights teach shelter-building the way no scripted lesson
   can"* — and in W0 as shipped a cold night carries no death gradient at the
   ambient the world actually runs at, which is a direct, quantitative account
   of SH.02's saturated null and of why the shelter specs keep failing to reach
   the question. Routed to the Review as `w1-cold-is-not-lethal-at-night`.

5. CHECK (b)'s LITERAL READING PASSES FOR THE WRONG REASON, AND IT IS WORTH
   RECORDING. Run the shipped model as shipped from 37 C into 20 C and it reads
   **34.000 C** at t = 1 h — parked at the shivering-supported equilibrium of
   finding 4, and `b_literal_is_at_equilibrium` fires. The reference at 1 h is
   still mid-transient at 33.767 C on its way to 20 C. Two completely different
   physical regimes — a steady state and an undamped decay — agree to 0.69%,
   inside the registered 1% bar, at exactly the one sample time the spec named.

   The spec says "PURE decay", the registered 33.767 is exactly the homogeneous
   (Q_gen = 0) solution, and the registered control passes only under that
   reading — so the homogeneous reading GATES, and the literal reading ships
   beside it as `b_literal_*` with the equilibrium evidence. The generalised
   lesson is in docs/LESSONS.md: comparing two dynamical systems at ONE sampled
   time constrains neither their time constant nor their steady state, and at a
   crossing time it cannot tell them apart even in principle.

CONTROLS AND NULLS — what makes these four checks instruments rather than prose
-------------------------------------------------------------------------------
CONTROL (registered): the reference model with h_c PINNED at its natural value
against wind. In still air it IS the reference, so it must pass (a) and (b)
exactly — that is the liveness proof for those two checks. At 5 m/s its tau
does not move, so it must fail (c). LESSONS.md's at-chance-control rule read in
the other direction: a control that must FAIL still has to prove the checks it
must PASS are alive.

POSITIVE REFERENCE (added, not registered): the same reference model WITH its
wind term live. It must PASS (c). Without it, check (c) would be a gate no arm
in the file can clear, and today's VO.02 lesson is exactly that — a floor
nothing can satisfy is unsatisfiable by arithmetic, not a floor. It is reported,
and its failure would VOID the run rather than pass it.

NULL 1 (registered): the overlay disabled — T never moves. Every check must
fail, (d) included: a body that stores no heat while flux crosses its boundary
does not conserve energy.

NULL 2 (registered): PURE AMBIENT, T := T_env instantly — "the cheapest thing
that could be mistaken for working". Its registered obligation is to FAIL (b)
and (c), and it does. The registry also predicts it "passes (a) trivially";
under the corrected reading of (a) it does NOT (it settles at 20 C in 20 C air,
7.55 C off the bar), because a body-temperature check is strictly sharper than
the ambient check the prediction assumed. That is a strengthening and it is
reported (`null_amb_a_pass`), never gated — a null failing a check it was
merely predicted to pass is not a reason to hold the run.

SEEDS. Checks (b), (c) and the static form checks are deterministic physics and
are byte-identical across seeds by construction; this is stated rather than
disguised with cosmetic jitter. The seed enters where PG.2's lesson says it
must: the (a) steady state is TIME-AVERAGED over a settled window rather than
sampled once, its initial body temperature is drawn per seed (so the reported
invariance of the steady state to that draw is itself measured), and the
conservation trajectory of check (d) is driven by a day/night ambient
oscillation whose phase and amplitude are drawn per seed. A body exchanging
heat with an oscillating ambient oscillates, and a single sample reads noise.
"""
from __future__ import annotations

import inspect
import math

import numpy as np

from .. import needs as N
from ..protocol import Ledger, run_spec
from ..registry import BY_ID

SPEC_ID = "W.1"

# ── the reference model's sourced constants (see docstring for citations) ──
MET_W_PER_M2 = 58.2          # 1 met
A_DUBOIS_M2 = 1.8481         # Du Bois area, 175 cm / 70 kg
M_REF_W = MET_W_PER_M2 * A_DUBOIS_M2          # 107.559 W
H_R = 4.7                    # W/m2K radiative
H_C_NATURAL = 3.0            # W/m2K convective, still air
M_BODY_KG = 70.0
# *** BURTON'S 1935 ASSUMPTION, NEVER MEASURED. *** The measured value is 2980
# (Xu, Rioux & Castellani, Temperature 2022, doi:10.1080/23328940.2022.2088034)
# and shortens every time constant by 14%. Burton gates because the registered
# bars were derived against Burton; 2980 is reported alongside.
C_P_BURTON = 3470.0          # J/kg/K
C_P_MEASURED = 2980.0        # J/kg/K
T_SKIN_NEUTRAL = 33.7        # C
T_CORE_NEUTRAL = 36.8        # C

# ── the PRE-REGISTERED bars. Not one of these moves. ──────────────────────
BALANCE_BODY_C = 27.55       # check (a): steady-state body temp in still air
BALANCE_TOL_C = 1.0
DECAY_1H_C = 33.767          # check (b): pure (Q_gen = 0) decay at t = 1 h
DECAY_TOL_FRAC = 0.01
SCENARIO_T0_C, SCENARIO_TENV_C, DECAY_T_S = 37.0, 20.0, 3600.0
TAU_WIND_RATIO = 0.3095      # check (c)
TAU_RATIO_TOL_FRAC = 0.02
WIND_V_MS = 5.0
FLUX_TOL_FRAC = 5e-3         # check (d), at the fine dt
FLUX_CONVERGENCE_LO, FLUX_CONVERGENCE_HI = 1.8, 2.2   # first order in dt

# Integrator settings. 1x WALL-CLOCK PHYSICS, as the spec demands: dt is in
# simulated seconds and one simulated second is one real second here. The
# compressed Jack-day clock is W.7's business and is only *reported* below.
# DT_S is the world's own decision step, so the gate sees the integrator the
# world actually runs.
DT_S = 0.2
DT_FINE_S = 0.1
AVG_S = 4000.0               # the window the steady state is averaged over
MAX_SETTLE_BLOCKS = 90       # up to 360 ks, ~21 reference tau
SETTLE_EPS_C = 1e-6          # successive block means this close = settled
EQ_DRIFT_EPS_C = 0.01        # "parked at equilibrium" for the (b) diagnostic
K_SHIFT_MIN_C = 0.5          # falsified_by: equilibrium must move with h


# ══════════════════════════════════════════════════════════════════════════
# The REFERENCE. A yardstick, never a world. Nothing in W0 imports this.
# ══════════════════════════════════════════════════════════════════════════
def _h_ref(v_ms: float, wind_blind: bool = False) -> float:
    """h = h_r + h_c. `wind_blind` is the registered CONTROL: h_c pinned."""
    if wind_blind or v_ms <= 0.0:
        return H_R + H_C_NATURAL
    return H_R + 8.6 * (v_ms ** 0.53)


def _hA_ref(v_ms: float, wind_blind: bool = False) -> float:
    return _h_ref(v_ms, wind_blind) * A_DUBOIS_M2


def _tau_ref(v_ms: float, c_p: float = C_P_BURTON,
             wind_blind: bool = False) -> float:
    return M_BODY_KG * c_p / _hA_ref(v_ms, wind_blind)


def _ref_balance_body_c(T_env: float = SCENARIO_TENV_C) -> float:
    """M = hA*(T_body - T_env) at balance, still air. This is the 27.5584."""
    return T_env + M_REF_W / _hA_ref(0.0)


def _ref_ambient_thermoneutral(T_body_neutral: float) -> float:
    """The OTHER reading of check (a), kept as a diagnostic. See docstring."""
    return T_body_neutral - M_REF_W / _hA_ref(0.0)


# ══════════════════════════════════════════════════════════════════════════
# The SHIPPED overlay, driven exactly as `needs.py` exposes it.
# ══════════════════════════════════════════════════════════════════════════
def _ship(T: float, T_env: float, dt: float, occ: float = 0.0,
          wet: float = 0.0) -> float:
    """One step of the world's own thermal ODE. Dry, still, at rest.

    Note what is NOT in this call: a wind velocity. There is no argument to
    pass one to. That absence is check (c)'s finding, measured below rather
    than asserted.
    """
    return N.thermal_step(T, 0.0, wet, occ, T_env, dt, 0.0)


def _ship_flux_w(T: float, T_env: float, occ: float = 0.0,
                 wet: float = 0.0) -> float:
    """Net power crossing the boundary, W — the numerator of the shipped ODE."""
    return (N.metabolic_rate(0.0, T, 0.0)
            - N.k_eff(wet, occ) * (T - T_env)
            - N.e_evap(wet, T))


def _ship_accepts_wind() -> float:
    """Structural, not inferred: does any thermal entry point take a velocity?"""
    names: set = set()
    for fn in (N.thermal_step, N.k_eff, N.metabolic_rate, N.e_evap):
        names |= set(inspect.signature(fn).parameters)
    return 1.0 if ({"v", "v_ms", "wind", "wind_ms", "velocity"} & names) else 0.0


# ══════════════════════════════════════════════════════════════════════════
# Measurement primitives — every one reads the ARM's behaviour, never a
# constant this file believes the arm holds.
# ══════════════════════════════════════════════════════════════════════════
def _settled_body_c(step, T_env: float, T_init: float, occ: float = 0.0) -> float:
    """Steady-state body temperature, TIME-AVERAGED over a settled window.

    PG.2's lesson made mechanical: settle first, then average, never sample.
    With a constant ambient the average equals the fixed point; the averaging
    is what makes the same routine honest when the ambient oscillates.

    Settling is by CONVERGENCE, not by a fixed horizon. A fixed horizon is what
    the first draft used and it silently under-settled the reference arm — 40 ks
    is 14 shipped time constants but only 2.3 reference ones, so the yardstick
    read 28.31 C instead of 27.56 and then poisoned every quantity measured
    about its fixed point, including its own tau. A horizon that is generous for
    the fast arm is not evidence about the slow one.
    """
    T, block, prev = T_init, int(AVG_S / DT_S), None
    mean = T_init
    for _ in range(MAX_SETTLE_BLOCKS):
        acc = 0.0
        for _ in range(block):
            T = step(T, T_env, DT_S, occ)
            acc += T
        mean = acc / block
        if prev is not None and abs(mean - prev) < SETTLE_EPS_C:
            return mean
        prev = mean
    return mean


def _measured_tau(step, T_env: float, T_eq: float, dt: float = DT_S,
                  cap_s: float = 1.2e5) -> float:
    """tau by the 1/e crossing of the arm's OWN relaxation about its OWN
    fixed point. Measured from the trajectory, so the number describes the
    code that runs and not this file's belief about it.

    A stationary arm (NULL 1) has no relaxation; it returns inf rather than
    silently reporting the cap as a time constant.
    """
    T0 = T_eq + 3.0
    target = T_eq + 3.0 * math.exp(-1.0)
    T, t = T0, 0.0
    while T > target and t < cap_s:
        T_next = step(T, T_env, dt)
        if abs(T_next - T) < 1e-15:
            return float("inf")
        T, t = T_next, t + dt
    return t if t < cap_s else float("inf")


# ══════════════════════════════════════════════════════════════════════════
# The four checks, run against any (stepper, flux, c_eff) triple.
# ══════════════════════════════════════════════════════════════════════════
def _four_checks(step, flux, c_eff: float, step_wind, seed: int,
                 tag: str) -> dict:
    """`step_wind` is the SAME model driven at WIND_V_MS, or None when the
    model has no velocity input at all to drive. The distinction is the whole
    of check (c): the registered control passes a stepper that ACCEPTS wind and
    ignores it, while the shipped world passes None because `k_eff` has no
    velocity parameter to ignore. Both must fail; only one of them could have
    been fixed by a constant."""
    rng = np.random.default_rng(1000 + seed)
    phase = float(rng.uniform(0.0, 2 * math.pi))
    amp_c = float(rng.uniform(3.0, 7.0))
    T_init = float(rng.uniform(34.0, 39.0))

    m: dict = {"arm": tag, "seed_phase": phase, "seed_amp_c": amp_c,
               "seed_T_init": T_init}

    # ── (a) steady-state body temperature in 20 C still air ───────────────
    a_body = _settled_body_c(step, SCENARIO_TENV_C, T_init)
    m["a_settled_body_c"] = a_body
    m["a_err_c"] = abs(a_body - BALANCE_BODY_C)
    m["a_pass"] = float(m["a_err_c"] <= BALANCE_TOL_C)
    m["a_ref_balance_body_c"] = _ref_balance_body_c()
    # the two ALTERNATIVE readings of the registered constant, as diagnostics
    m["a_ref_ambient_from_skin_c"] = _ref_ambient_thermoneutral(T_SKIN_NEUTRAL)
    m["a_ref_ambient_from_core_c"] = _ref_ambient_thermoneutral(T_CORE_NEUTRAL)
    m["a_registered_as_ambient_implies_body_c"] = (
        BALANCE_BODY_C + M_REF_W / _hA_ref(0.0))
    # the steady state must not depend on where we started (seed draws T_init)
    a_body_alt = _settled_body_c(step, SCENARIO_TENV_C, T_init - 3.0)
    m["a_settled_invariant_to_T_init_c"] = abs(a_body - a_body_alt)

    # ── (c)'s time constants, measured; also used by (b) ──────────────────
    tau_still = _measured_tau(step, SCENARIO_TENV_C, a_body)
    if step_wind is None:
        # No velocity input exists to change anything. Recorded as the identity
        # it is, rather than worked around.
        tau_wind = tau_still
    else:
        # Measured from the wind-driven trajectory about ITS OWN fixed point,
        # never inferred from the h ratio the check is trying to verify.
        tau_wind = _measured_tau(
            step_wind, SCENARIO_TENV_C,
            _settled_body_c(step_wind, SCENARIO_TENV_C, T_init))
    m["c_tau_still_s"] = tau_still
    m["c_tau_wind_s"] = tau_wind
    m["c_has_wind_input"] = float(step_wind is not None)
    ratio = (tau_wind / tau_still) if (tau_still > 0 and math.isfinite(tau_still)
                                       and math.isfinite(tau_wind)) else float("nan")
    m["c_tau_ratio"] = ratio
    m["c_rel_err"] = (abs(ratio - TAU_WIND_RATIO) / TAU_WIND_RATIO
                      if math.isfinite(ratio) else float("inf"))
    m["c_pass"] = float(m["c_rel_err"] <= TAU_RATIO_TOL_FRAC)

    # ── (b) PURE decay (Q_gen = 0) from 37 into 20 C, read at t = 1 h ─────
    # The homogeneous response of the arm's own measured time constant. This is
    # the reading that reproduces the registered 33.767 and the only one the
    # registered control can pass; see the docstring.
    if math.isfinite(tau_still) and tau_still > 0:
        b_homog = SCENARIO_TENV_C + (SCENARIO_T0_C - SCENARIO_TENV_C) * math.exp(
            -DECAY_T_S / tau_still)
    else:
        b_homog = SCENARIO_T0_C       # never relaxes: it is still at 37
    m["b_homogeneous_1h_c"] = b_homog
    m["b_rel_err"] = abs(b_homog - DECAY_1H_C) / DECAY_1H_C
    m["b_pass"] = float(m["b_rel_err"] <= DECAY_TOL_FRAC)
    m["b_ref_1h_c"] = SCENARIO_TENV_C + 17.0 * math.exp(-DECAY_T_S / _tau_ref(0.0))

    # the LITERAL reading, reported not gated: run the arm as it ships
    T = SCENARIO_T0_C
    for _ in range(int(round(DECAY_T_S / DT_S))):
        T = step(T, SCENARIO_TENV_C, DT_S)
    b_lit = T
    T2 = T
    for _ in range(int(round(DECAY_T_S / DT_S))):
        T2 = step(T2, SCENARIO_TENV_C, DT_S)
    m["b_literal_1h_c"] = b_lit
    m["b_literal_rel_err"] = abs(b_lit - DECAY_1H_C) / DECAY_1H_C
    m["b_literal_pass"] = float(m["b_literal_rel_err"] <= DECAY_TOL_FRAC)
    m["b_literal_drift_next_hour_c"] = abs(T2 - b_lit)
    m["b_literal_is_at_equilibrium"] = float(
        m["b_literal_drift_next_hour_c"] < EQ_DRIFT_EPS_C)

    # ── (d) integrated net flux == c_eff * dT, to integrator tolerance ────
    def conservation(dt: float) -> float:
        T, t, acc = T_init, 0.0, 0.0
        for _ in range(int(2 * N.SIM_DAY_S / dt)):
            T_env = SCENARIO_TENV_C + amp_c * math.sin(
                2 * math.pi * t / N.SIM_DAY_S + phase)
            f0 = flux(T, T_env)
            T_next = step(T, T_env, dt)
            # TRAPEZOID, deliberately not the left rectangle: with explicit
            # Euler the rectangle rule is an algebraic identity and would
            # certify any stepper at all. The trapezoid residual telescopes to
            # 0.5*dt*(f_end - f_start), so it is O(dt) and its halving is the
            # actual evidence that the integrator converges.
            acc += 0.5 * (f0 + flux(T_next, T_env)) * dt
            T, t = T_next, t + dt
        stored = c_eff * (T - T_init)
        return abs(acc - stored) / max(abs(stored), 1.0)

    d_coarse, d_fine = conservation(DT_S), conservation(DT_FINE_S)
    m["d_rel_err_coarse"] = d_coarse
    m["d_rel_err_fine"] = d_fine
    m["d_convergence_ratio"] = (d_coarse / d_fine) if d_fine > 0 else float("inf")
    m["d_pass"] = float(d_fine <= FLUX_TOL_FRAC
                        and FLUX_CONVERGENCE_LO <= m["d_convergence_ratio"]
                        <= FLUX_CONVERGENCE_HI)

    # ── falsified_by: the equilibrium must DEPEND on h ────────────────────
    eq_occl = _settled_body_c(step, SCENARIO_TENV_C, T_init, occ=0.9)
    m["e_equilibrium_shift_with_k_c"] = abs(eq_occl - a_body)
    m["e_equilibrium_depends_on_h"] = float(
        m["e_equilibrium_shift_with_k_c"] > K_SHIFT_MIN_C)

    m["all_finite"] = float(all(
        math.isfinite(v) for k, v in m.items()
        if isinstance(v, float) and not k.startswith("seed_")))
    m["four_checks_pass"] = float(m["a_pass"] and m["b_pass"]
                                  and m["c_pass"] and m["d_pass"])
    return m


# ══════════════════════════════════════════════════════════════════════════
# Arms
# ══════════════════════════════════════════════════════════════════════════
def _reference_arm(seed: int, wind_blind: bool, tag: str) -> dict:
    """The sourced lumped-capacitance yardstick. A model, never a world.

    `wind_blind=True` is the REGISTERED CONTROL: h_c is pinned at its natural
    value, so the arm still ACCEPTS a velocity and simply does not respond to
    it. `wind_blind=False` is the positive reference that proves check (c) is
    clearable at all.
    """
    c_eff = M_BODY_KG * C_P_BURTON

    def make(v_ms: float):
        hA = _hA_ref(v_ms, wind_blind=wind_blind)

        def step(T, T_env, dt, occ=0.0):
            # k carries the same occlusion knob the shipped model has, so the
            # falsified_by h-dependence clause is testable on this arm too.
            k = hA * (1.0 - N.OCC_CUT * occ)
            return T + (M_REF_W - k * (T - T_env)) / c_eff * dt

        return step

    step_still = make(0.0)

    def flux(T, T_env):
        return M_REF_W - _hA_ref(0.0, wind_blind=True) * (T - T_env)

    return _four_checks(step_still, flux, c_eff, make(WIND_V_MS),
                        seed=seed, tag=tag)


def _experiment(seed: int) -> dict:
    """THE SHIPPED WORLD: experiments/needs.py's thermal overlay."""
    # step_wind is None because there is no velocity parameter to pass — a
    # structural fact, re-derived from the signatures rather than assumed.
    has_wind = bool(_ship_accepts_wind())
    m = _four_checks(_ship, _ship_flux_w, N.C_EFF,
                     _ship if has_wind else None, seed=seed, tag="shipped")
    m["w_signature_accepts_wind"] = float(has_wind)

    # Provenance of the world's own numbers, so the record does not depend on
    # this file having read them correctly.
    m["w_C_EFF"] = float(N.C_EFF)
    m["w_K_DRY"] = float(N.K_DRY)
    m["w_C_SH"] = float(N.C_SH)
    m["w_TAU_T_declared_s"] = float(N.TAU_T)
    m["w_tau_from_constants_s"] = float(N.C_EFF / N.K_DRY)
    m["w_SIM_DAY_S"] = float(N.SIM_DAY_S)
    m["w_night_ambient_c"] = float(N.T_DAY - N.DELTA_T_NIGHT)
    m["w_T_COLD_DEATH_c"] = float(N.T_COLD_DEATH)

    # The world's DECLARED time constant is open-loop; the one it actually
    # relaxes with is not. `TAU_T = C_EFF/K_DRY = 240 s` assumes a constant
    # heat source, but shivering is proportional feedback (gain C_SH), so the
    # closed-loop conductance is K_DRY + C_SH = 47.62 W/C and the measured
    # relaxation is 72.0 s — 3.33x faster than the constant the world names.
    # Reported, not gated: check (b) fails on either value (20.0 vs 20.0000052
    # against 33.767), so no verdict here turns on the distinction.
    m["w_tau_measured_closed_loop_s"] = m["c_tau_still_s"]
    m["w_shiver_speedup"] = (float(N.C_EFF / N.K_DRY) / m["c_tau_still_s"]
                             if m["c_tau_still_s"] > 0 else float("nan"))

    # W.7's transformation, reported never gated (finding 2). Open-loop against
    # open-loop: the reference has a constant source and no feedback term, so
    # C_EFF/K_DRY is the matching quantity, not the closed-loop 72 s.
    ref_tau = _tau_ref(0.0)
    m["ref_tau_still_s"] = ref_tau
    m["ref_tau_still_measured_cp_s"] = _tau_ref(0.0, c_p=C_P_MEASURED)
    m["tau_compression_factor"] = ref_tau / m["w_tau_from_constants_s"]
    m["day_compression_factor"] = 86400.0 / N.SIM_DAY_S
    m["compression_agreement_rel"] = abs(
        m["tau_compression_factor"] - m["day_compression_factor"]
    ) / m["day_compression_factor"]

    # Finding 4, computed from the shipped constants: the body's parked
    # temperature at the world's OWN night ambient, and the ambient at which
    # cold first becomes lethal. Reported, not gated — W.3 is the spec that
    # prices the consequence.
    night = m["w_night_ambient_c"]
    denom = N.C_SH + N.K_DRY
    m["w_parked_body_at_night_c"] = (
        N.M_BASAL + N.C_SH * N.T_SETPOINT + N.K_DRY * night) / denom
    m["w_lethal_ambient_c"] = (
        N.T_COLD_DEATH * denom - N.M_BASAL - N.C_SH * N.T_SETPOINT) / N.K_DRY
    m["w_night_is_lethal"] = float(night <= m["w_lethal_ambient_c"])

    # ── the registered nulls and the positive reference, all reported ─────
    # NULL 1: overlay disabled. T never moves; nothing conserves.
    null_off = _four_checks(lambda T, Te, dt, occ=0.0: T, _ship_flux_w, N.C_EFF,
                            None, seed=seed, tag="null_off")
    # NULL 2: PURE AMBIENT, T := T_env instantly.
    null_amb = _four_checks(lambda T, Te, dt, occ=0.0: Te,
                            lambda T, Te, occ=0.0, wet=0.0: 0.0, N.C_EFF,
                            None, seed=seed, tag="null_ambient")
    # POSITIVE REFERENCE: the yardstick with its wind term LIVE. Must pass (c),
    # or check (c) is a gate nothing in this file can clear (the VO.02 lesson).
    ref_wind = _reference_arm(seed, wind_blind=False, tag="ref_wind_aware")

    for src, pre in ((null_off, "null_off_"), (null_amb, "null_amb_"),
                     (ref_wind, "ref_wind_")):
        for k in ("a_pass", "b_pass", "c_pass", "d_pass", "four_checks_pass"):
            m[pre + k] = src[k]
    m["ref_wind_tau_ratio"] = ref_wind["c_tau_ratio"]
    return m


def _control(seed: int) -> dict:
    """REGISTERED CONTROL: the reference model with h_c PINNED against wind.

    In still air this IS the reference, so it must pass (a) and (b) — that is
    the liveness proof for those two checks. At 5 m/s its tau does not move, so
    it must fail (c). A check (c) that cannot separate this arm from the
    wind-aware reference is certifying a thermometer, not a heat balance.
    """
    return _reference_arm(seed, wind_blind=True, tag="control_wind_blind")


def _check(m: dict, c: dict) -> bool:
    """PRE-REGISTERED. All four checks, plus every instrument shown alive.

    Not one bar is derived from the data: (a), (b) and (c) are the registry's
    own numbers, and (d)'s tolerance and convergence band are integrator
    properties fixed before the run.
    """
    # THE CLAIM: the shipped overlay passes all four, is finite, and its
    # equilibrium depends on h (falsified_by's third clause).
    experiment_ok = (m["four_checks_pass"] == 1.0
                     and m["all_finite"] == 1.0
                     and m["e_equilibrium_depends_on_h"] == 1.0)

    # THE CONTROL: fails (c) while passing (a) and (b), exactly as registered.
    control_ok = (c["c_pass"] == 0.0 and c["a_pass"] == 1.0
                  and c["b_pass"] == 1.0)

    # CHECK (c) MUST BE CLEARABLE by something, or it is arithmetic, not a gate.
    positive_ok = (m["ref_wind_c_pass"] == 1.0)

    # NULL 1 must fail everything. NULL 2's registered obligation is (b) and
    # (c); its (a) is reported, not gated (see the docstring).
    nulls_ok = (m["null_off_four_checks_pass"] == 0.0
                and m["null_off_a_pass"] == 0.0
                and m["null_off_b_pass"] == 0.0
                and m["null_off_c_pass"] == 0.0
                and m["null_off_d_pass"] == 0.0
                and m["null_amb_b_pass"] == 0.0
                and m["null_amb_c_pass"] == 0.0)

    return bool(experiment_ok and control_ok and positive_ok and nulls_ok)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID[SPEC_ID], _experiment, _check, control_fn=_control,
                    ledger=ledger)
