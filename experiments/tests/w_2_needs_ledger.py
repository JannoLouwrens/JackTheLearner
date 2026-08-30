"""w_2_needs_ledger.py — W.2: Needs are a conserved ledger, and they can kill.

W.1 asked whether W0's temperature obeys the heat balance the world publishes
(it does not: 3 of 4 checks failed). This spec asks the adjacent question about
the other six meters: **does the needs bookkeeping conserve, and are the deaths
it advertises the deaths it delivers?**

The boundary against `NE.01` is the registry's own and it binds: NE.01 asks
whether death is REACHABLE and SPREAD (the policy landscape); W.2 asks whether
the METERS CONSERVE (the bookkeeping). NE.01 can pass over a leaking ledger —
that is exactly why this spec exists. The registry's `kills` field says why it
matters: *"a needs system that does not conserve is a system where Jack can
learn to exploit the bookkeeping instead of the world — the survival analogue
of the noisy TV."*

WHAT IS UNDER TEST
------------------
`experiments/needs.py`'s `NeedLayer`, as shipped, driven inside the real
playground (`playground.make_playground`, PG.8's world). Nothing here
re-implements the integrator: every arm below is the shipped object, stepped
through its own public contract (`begin_decision` / `substep` / `decide`), and
every intake is a REAL recorded world event (mouth contact with a food geom,
mouth near the pool surface) rather than a number written into the state.

THE BODY IS A STATUE, AND THAT IS THE POINT
-------------------------------------------
Jack is teleported (root qpos + `mj_forward`) and never actuated, so
`p_mech = 0` and the metabolic hub sits at `M_BASAL` whenever the body is at
its setpoint. `substep(model, data, dt)` is called once per decision with
`dt = 0.2` — the world's own decision step (§2.3). For the needs arithmetic
that is identical to `frame_skip` substeps summing to 0.2 s: `substep` only
accumulates `dt` and the power integral, and the power integral is zero either
way for a body at rest. This is a BOOKKEEPING audit; the locomotion the statue
does not do is NE.01's subject, not this spec's.

THE FOUR REGISTERED CHECKS
--------------------------
(a) *"Hunger, thirst and sleep pressure integrate to their closed-form
    solutions within 1%."* Measured on the DEPRIVED life (nothing met, in the
    open, by day) over the window in which all three meters are unclipped and
    the body is alive — which the world itself bounds at 450 s, when water
    empties. In that window `T` sits exactly on 37.0 (the world's `K_DRY` is
    defined as `M_BASAL/(T_SETPOINT - T_DAY)`, making 37 an exact fixed point
    by day), so `M == M_BASAL` and the closed forms are the plain ones:

        e(t) = 1 - B_E t          w(t) = 1 - B_W t
        p(t) = 1 - (1 - p0) exp(-t / TAU_WAKE)

    "Within 1%" is read against FULL METER SCALE — these are meters normalised
    to [0, 1] and a relative bar would be unbounded at the zero crossing the
    thirst meter makes inside the window. Stated, not assumed: the bar is
    0.01 in meter units.

(b) *"Energy in equals energy out to 1e-6 relative over a 10-day life."*
    Measured on the SATED life — 10 Jack-days (12,000 s) of the scripted
    homeostat, eating apples and drinking from the pool. **The reconstruction
    is built only from quantities the layer PUBLISHES** — `last_power_w`,
    `last_dt`, `kappa_act`, `ate_total`, `drank_total` and the state it
    returns — never from its private `_ate`/`_drank`/`_power_dt`. It applies
    the published law

        e -= (M(p_mech, T, kappa_act) / M_BASAL) * B_E * dt   + intake
        w -= (1 + C_SW * max(0, T - 37)) * B_W * dt           + intake

    at every decision, clips exactly as `decide()` does, and compares to the
    state the layer actually returned. That is NOT a tautology: if `decide()`
    double-counted a BMR on top of its met-unit metabolism — the specific trap
    the registry's `notes` field names, worth 25% and silent — this residual
    would carry it. The saturation mass (intake discarded at the clip) is
    reported separately, because unrecorded saturation is precisely the
    bookkeeping a policy could farm.

(c) *"Each need independently reaches a lethal threshold at the pre-registered
    deadline ... when and only when it is not met."* Human deadlines divided by
    W.7's DECLARED compression `k`, with both forms reported, per the registry's
    REGISTRATION NOTE. `k = 86400 / SIM_DAY_S = 72.0`.

    **A temperature is not a duration and is not divided by k.** The registered
    band (28-40 C) is a THRESHOLD; it is gated against the world's declared
    `T_COLD_DEATH` / `T_HOT_DEATH` directly, and the accompanying question —
    can the world actually REACH either bound? — is measured, not asserted.

(d) *"Sleep pressure discharges 4.3x faster than it accumulates (tau_wake
    18.2 h vs tau_sleep 4.2 h)."* Both time constants are MEASURED by
    log-linear relaxation of the shipped update about its own fixed point (1.0
    awake, 0.0 asleep), never read off the declared constants — W.1's lesson,
    where a declared open-loop `TAU_T = 240 s` hid a closed-loop 72 s.

WHAT THE MEASUREMENT FOUND
--------------------------
1. THE INTEGRATORS ARE EXACT, NOT MERELY WITHIN 1%. Max absolute deviation
   from the closed forms over the 450 s window is ~1e-13 for all three meters —
   floating-point, not integration error. The sleep update is written as the
   exact exponential map (`p + (1-p)(1-exp(-dt/tau))`), so it composes without
   discretisation error, and the two linear drains are linear. Check (a) passes
   with thirteen orders of magnitude of room.

2. THE LEDGER CONSERVES — EXACTLY. Over a 10-Jack-day sated life (60,000
   decisions, 17 eating events, 53 drinking events) the reconstruction from
   published diagnostics reproduces the returned state to a maximum deviation
   of **0.0** in meter units — bit for bit, not merely inside 1e-6. The
   met-unit double-counting trap the registry warns about (worth 25% and
   silent) is NOT present. Check (b) passes.

   The saturation is large and is reported rather than netted away:
   `b_clip_w = -21.18` meter-units of drinking discarded at the ceiling
   against `b_drain_w = 26.67` consumed — the homeostat throws away 44% of
   what it drinks, because `NU_DRINK = 0.9` refills a tank that is rarely more
   than half empty. `b_clip_e = 0.0`: no food is ever wasted. That asymmetry
   is exactly the shape a policy could farm if the ledger ever paid for intake
   rather than for meter level, and it belongs in the record.

3. THE DEADLINES ARE 6x AND 12x SHORT, AND THAT IS THE VERDICT. This is the
   check that fails, and it fails by margins no tolerance choice can rescue:

       need     human      / k=72     W0 measured    short by
       thirst   3 d        3600 s     569.8 s        6.32x
       hunger   3 wk       25200 s    1700.4 s       14.82x   (2100 s at
                                                     basal drain: 12.00x)

   The pre-registered tolerance is a FACTOR OF 2, derived from the spread in
   the spec's own sources (food "3-4 weeks"; the 1981 hunger-strike deaths at
   46-73 days; water "~3 days, faster in heat") — the tightest band those
   sources support. The verdict does not turn on it: a 5x band still fails
   both, and `c_widest_tol_that_would_pass` is reported so the next reader can
   see how far the bar would have to move.

   The deeper finding is that **W0 has no single time-compression factor**, so
   W.7's premise (*"only the need-accumulation clock is scaled"*) already has a
   counterexample waiting for it. Implied k per subsystem, all from shipped
   constants:

       day length      86400 / 1200          =  72.0
       thermal tau     17069 / 240           =  71.1   (W.1, finding 2)
       sleep tau_wake  65520 / 700           =  93.6
       sleep tau_sleep 15120 / 160           =  94.5
       thirst          259200 / 570          = 454.7
       hunger          1814400 / 2100        = 864.0

   Thermal and day agree to 1.2% and sleep is a further 30% off, but hunger and
   thirst are one and two orders of magnitude away. In Jack-days: a human dies
   of thirst after 3 days and starves after 21; W0's Jack dies of thirst in
   **0.475 of a day** and starves in **1.42 days**. The ratios BETWEEN needs
   are wrong too — human hunger:thirst is 7.00:1, W0's is 2.98:1 (3.68:1 at
   basal drain) — so no single choice of k can repair this: k is one number and
   two independent ratios are off. It is a re-scaling of the world's constants,
   which is a redesign, and it is routed to the Review rather than done here.

4. COLD CANNOT KILL A DRY BODY, BUT IT KILLS A WET ONE IN 54 SECONDS. W.1
   found the dry night ambient (20 C) parks the body at 34.0 C, forever, with
   the lethal ambient at exactly 0.0 C. Measured here from the needs side:
   the dry statue's minimum body temperature across a full night is 33.99 C,
   and it never dies of cold at any horizon. Soak the same statue — sit it in
   the pool, where `KAPPA_WET = 4` quintuples `k_eff` and `C_EV` sheds another
   50 W — and it dies of **hypothermia at t = 854 s**, 54 s after nightfall,
   at a minimum of 26.5 C. So the cold death mode IS reachable, and the only
   route to it is water. That is a sharpening of W.1's finding, not a
   contradiction: *dry cold is decorative in W0; wet cold is lethal.*

5. SHELTER COOKS HIM BY DAY. `sky_occlusion` is the world's only shelter
   mechanism and it cuts `k_eff` by `OCC_CUT = 0.7` with no day/night
   awareness. Fully roofed at the day ambient of 30 C the statue dies of
   **hyperthermia at t = 182.4 s** (T peaks 40.33 C), because shivering stops
   above 37 and the closed loop parks at `30 + M_BASAL/(K_DRY*0.3) = 53.3 C`.
   The same roof at night is worth about 4 C of warmth. **W0's shelter is a
   trap by day and a boon by night, and nothing in the world tells him which** —
   that is a fully-specified, learnable rule, and it is arguably the most
   GOAL-shaped thing this measurement found: it is exactly the kind of
   consequence a curious agent could discover by poking. Reported, not gated;
   `W.3` ("cold kills, and shelter is why it does not") is the spec that
   prices it, and it now inherits a measured second half — *heat kills, and
   shelter is why*.

THE "27.5 C" IN THE REGISTERED CONTROL IS AMBIGUOUS — AND, MEASURED, IT DOES
NOT MATTER. THE W.1 TIE-BREAKER DOES NOT APPLY HERE.
----------------------------------------------------------------------------
The registry's control reads *"a SATED agent — fed, watered, rested, at 27.5 C
— must survive an arbitrarily long life."* W.1 found the neighbouring 27.55 C
in ITS registry entry mislabelled (an ambient that was really a body
temperature), so the obvious move is to expect the same defect here and read
27.5 C as an ambient. **That expectation was written down, tested, and was
wrong, and this paragraph is the correction.**

Read as an AMBIENT the shipped closed loop parks the body at
`(M_BASAL + C_SH*T_SETPOINT + K_DRY*27.5) / (C_SH + K_DRY) = 36.25 C` —
comfortably alive. Read as a BODY temperature, 27.5 C is 0.5 C below the
world's own `T_COLD_DEATH = 28.0`, which looks immediately fatal — and it is
not. Measured (`ctl_body275_*`): the sated body starts at 27.5 C, shivering
supplies `C_SH*(37 - 27.5) = 316.7 W` against an ambient it is *colder* than,
and it climbs back through 28 C after a measured **5.0 s** of dwell — a
quarter of the world's `DEATH_T_S = 20 s` continuous-dwell grace. It survives
the full horizon and ends at 36.99 C.

**So both readings satisfy the clause the registry wrote, and the control
cannot decide between them.** `ctl_275_reading_decidable` records that as a 0,
so no later reader resolves the label by analogy to W.1. The rule W.1 added to
LESSONS.md — *a control that fails a check it was registered to PASS is a
fault in the CHECK* — is a one-way instrument: it fires only when a reading is
UNSATISFIABLE, and neither reading here is. Nothing in this spec's verdict
turns on the question (check (c) fails under both), and the sated control is a
live instrument under both. Recorded and left open.

The by-product is worth more than the label: **W0's lethal temperature bounds
are dwell-gated, not instantaneous**, so "outside 28-40 C" is not by itself a
death. `ctl_body275_s_below_cold` measures the dwell and `ctl_body275_grace_s`
names the window it is compared against.

CONTROLS, NULLS AND RIG GATES
-----------------------------
CONTROL 1 (registered): the SATED agent must survive an arbitrarily long life.
Run to 10 Jack-days in `_control` independently of check (b)'s life, and the
two are required to agree — a control measured in the same call as the claim
is not much of a control.

CONTROL 2 (registered): *"each need ablated in turn must remove exactly its own
death mode and no other."* Implemented by MEETING the need with the world's own
affordances rather than by monkey-patching a constant out, which is both
stronger and cheaper: meeting water must convert the death from dehydration to
starvation and must not remove it; meeting food must leave the dehydration
death untouched at its original time; meeting neither must reproduce it.

NULL (registered): FROZEN-NEEDS, whose meters never move. Implemented by
calling the layer's own `new_body()` after every decision, so the meters are
re-pinned to their setpoints while the world clock, the food respawn timers and
the death machinery all keep running. It must not die at any horizon — if it
does, W0's lethality is being driven by a clock rather than by a need.

RIG GATES -> VOID, NEVER FAIL. Every scripted intervention asserts its own
precondition before the arm it feeds is believed: the pool placement must put
the mouth near the surface and NOT submerged, the food placement must contact
the mouth mask AND leave the sky unoccluded, the land placement must leave the
sky unoccluded, the roof must occlude all nine rays, and the dry day statue
must hold exactly 37.0 C. **This is not decoration — it is a scar from this
file's own construction.** `PlaygroundParams.mutate` drops an object on seed 1,
which shifts the humanoid root's `qpos` address from 43 to 36; a hard-coded 43
silently teleported nothing, and seed 1 then reported that meeting water does
NOT remove the dehydration death — a clean, plausible, entirely false
refutation of control 2, on one seed out of three. See docs/LESSONS.md.

SEEDS. Three, per the registry. The seed enters through the world
(`PlaygroundParams(seed).mutate(...)`, NE.01's convention — object sizes,
positions and pool size all move) and through the microsleep RNG. It does NOT
enter the thermal or need arithmetic, which is deterministic; the deadlines are
therefore expected to be identical across seeds and their identity is REPORTED
(`c_thirst_s_spread`, `c_hunger_s_spread`) rather than disguised with cosmetic
jitter. What the seeds genuinely test is whether the scripted interventions
survive a mutated world — which, as the rig-gate paragraph records, is the
thing that actually broke.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from .. import needs as N
from ..protocol import Ledger, borrow_metrics, run_spec
from ..registry import BY_ID

SPEC_ID = "W.2"
IMPL_DEPS = ["playground.py", "experiments/needs.py", "experiments/drives.py"]

REPO = Path(__file__).resolve().parents[2]

# ══════════════════════════════════════════════════════════════════════════
# THE PRE-REGISTERED BARS. Not one of these moves.
# ══════════════════════════════════════════════════════════════════════════
INTEGRATOR_TOL = 0.01            # (a) "within 1%", in meter units (full scale 1)
CONSERVATION_TOL_REL = 1e-6      # (b) the registry's own figure
LIFE_DAYS = 10                   # (b) "over a 10-day life", in JACK-days
SLEEP_TAU_WAKE_H = 18.2          # (d) the registry's parenthetical, precise form
SLEEP_TAU_SLEEP_H = 4.2
SLEEP_RATIO = SLEEP_TAU_WAKE_H / SLEEP_TAU_SLEEP_H          # 4.333333...
SLEEP_RATIO_ROUNDED = 4.3        # the same number as printed to 2 s.f.
SLEEP_RATIO_TOL_FRAC = 0.01

# (c) human deadlines, from the registry `notes` field's own sourcing.
HUMAN_THIRST_S = 3.0 * 86400.0        # water ~3 days (faster in heat)
HUMAN_HUNGER_S = 21.0 * 86400.0       # food 3-4 weeks; 3 is the tighter end
HUMAN_T_COLD_C = 28.0                 # hypothermia: severe band opens
HUMAN_T_HOT_C = 40.0                  # hyperthermia: emergency
# The tolerance is a FACTOR, and it is the tightest band the spec's own
# sources support: "3-4 weeks" is 1.33x on its face and the 1981 hunger-strike
# deaths span 46-73 days (1.59x). 2.0 is generous to the world, deliberately.
DEADLINE_TOL_FACTOR = 2.0

# W.7's DECLARED compression, computed from the world rather than transcribed.
K_DECLARED = 86400.0 / N.SIM_DAY_S    # 72.0

# ── harness constants, fixed before the run ───────────────────────────────
DT_S = 0.2                       # the world's own decision step (§2.3)
LIFE_S = LIFE_DAYS * N.SIM_DAY_S # 12,000 s
DEPRIVED_HORIZON_S = 3000.0      # >> any single-need deadline in W0
REACH_HORIZON_S = 3000.0         # for the cold/hot reachability arms
WAKE_FIT_S = 3500.0              # long enough for ~5 tau_wake
SLEEP_FIT_S = 1200.0             # long enough for ~7 tau_sleep
TAU_FIT_LO, TAU_FIT_HI = 0.15, 0.90   # the |p - fixed point| band fitted
TAU_FIT_LO_SLEEP = 0.02
NEED_TOPUP = 0.5                 # the homeostat attends to a meter below this
THERMONEUTRAL_EPS_C = 1e-9       # the day fixed point must be EXACT
CLOSED_FORM_WINDOW_S = 1.0 / N.B_W    # 450 s: where thirst clips. Set by the
                                      # world, not chosen by this file.

_BORROW_CACHE: dict = {}


def _borrowed():
    """PS.01's j0/alpha/P_bar(1) — NeedLayer refuses defaults, and is right to."""
    if "b" not in _BORROW_CACHE:
        _BORROW_CACHE["b"] = borrow_metrics(
            "PS.01", ("j0_ms", "alpha", "mean_power_w_full_strength"))
    return _BORROW_CACHE["b"]


# ══════════════════════════════════════════════════════════════════════════
# The rig: the real playground, and every index into it resolved BY NAME.
# ══════════════════════════════════════════════════════════════════════════
class Rig:
    """One mutated world plus the scripted placements, with their preconditions.

    Every model index here is resolved by NAME on the world actually built.
    `PlaygroundParams.mutate` changes the object inventory per seed, so a
    numeric qpos address captured from seed 0 addresses a different joint on
    seed 1 — see the module docstring and docs/LESSONS.md.
    """

    FAR = (20.0, 20.0, -5.0)             # where an uneaten food geom is parked
    LAND = (0.0, -1.6, 1.4)              # dry, open, unoccluded
    ROOF_GAP_M = 0.18                    # occluder height above the head geom
    POOL_XY = (2.6, -2.4)                # NE.01's pool centre
    POOL_ROOT_Z = -0.10                  # puts the head geom just above surface

    def __init__(self, seed: int):
        sys.path.insert(0, str(REPO))
        import mujoco
        from playground import PlaygroundParams, make_playground

        self.mj = mujoco
        p = PlaygroundParams(seed=seed)
        if seed > 0:                                  # PS.01/NE.01 convention
            p = p.mutate(np.random.RandomState(seed))
        self.params = p
        self.model, self.data, _ = make_playground(
            p, with_water=True, with_humanoid=True)
        self.pool = (self.POOL_XY[0], self.POOL_XY[1], p.pool_size, 0.0)

        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if jid < 0:
            raise RuntimeError("no humanoid root joint in this world")
        self.root_adr = int(self.model.jnt_qposadr[jid])

        self.food_adr = {}
        self.food_r = {}
        for name in N.FOOD:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                continue
            bid = int(self.model.geom_bodyid[gid])
            js = [j for j in range(self.model.njnt)
                  if int(self.model.jnt_bodyid[j]) == bid]
            if not js:
                continue
            self.food_adr[name] = int(self.model.jnt_qposadr[js[0]])
            self.food_r[name] = float(self.model.geom_size[gid][0])
        self._placed = None

    # ── layer construction ────────────────────────────────────────────
    def layer(self, seed: int, state=None):
        b = _borrowed()
        lay = N.NeedLayer(self.model, j0=b.values["j0_ms"],
                          alpha=b.values["alpha"],
                          p_max=b.values["mean_power_w_full_strength"],
                          pool=self.pool, state=state, seed=seed)
        self.head = int(next(iter(lay.body.head_geoms)))
        self.head_r = float(self.model.geom_size[self.head][0])
        return lay

    # ── placement primitives ──────────────────────────────────────────
    def _root(self, xyz):
        d, a = self.data, self.root_adr
        d.qpos[a:a + 3] = xyz
        d.qpos[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)
        d.qvel[:] = 0.0

    def _food_at(self, name, xyz):
        a = self.food_adr[name]
        self.data.qpos[a:a + 3] = xyz
        self.data.qpos[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)

    def place(self, mode: str, force: bool = False) -> None:
        """Teleport into one of the four scripted configurations. Cached: a
        placement costs an `mj_forward`, and 60,000 identical ones would
        dominate the run."""
        if mode == self._placed and not force:
            return
        self._root(self.LAND if mode != "pool"
                   else (self.pool[0], self.pool[1], self.POOL_ROOT_Z))
        for name in self.food_adr:
            self._food_at(name, self.FAR)
        self.mj.mj_forward(self.model, self.data)
        if mode in ("eat", "roof"):
            hp = np.array(self.data.geom_xpos[self.head], dtype=float)
            name = "apple" if "apple" in self.food_adr else next(iter(self.food_adr))
            gap = self.head_r + self.food_r[name]
            dz = -0.9 * gap if mode == "eat" else self.ROOF_GAP_M
            self._food_at(name, (hp[0], hp[1], hp[2] + dz))
            self.mj.mj_forward(self.model, self.data)
        self._placed = mode

    # ── the rig gates: every intervention proves its own precondition ──
    def preconditions(self, lay) -> dict:
        out = {}
        self.place("pool", force=True)
        out["rig_pool_mouth_near_surface"] = float(lay._mouth_near_surface(self.data))
        out["rig_pool_head_not_submerged"] = float(not lay._head_under(self.data))
        self.place("land", force=True)
        out["rig_land_sky_open"] = float(lay._sky_occlusion(self.model, self.data) == 0.0)
        self.place("eat", force=True)
        mouth = N._contact_partners(self.data, lay._mouth_mask)
        food_gids = {int(self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_GEOM, n))
                     for n in self.food_adr}
        out["rig_food_touches_mouth"] = float(bool(mouth & food_gids))
        out["rig_food_sky_open"] = float(lay._sky_occlusion(self.model, self.data) == 0.0)
        self.place("roof", force=True)
        out["rig_roof_occludes_all"] = float(lay._sky_occlusion(self.model, self.data) == 1.0)
        self.place("land", force=True)
        return out


# ══════════════════════════════════════════════════════════════════════════
# The scripted lives. One driver, six policies, all on the shipped layer.
# ══════════════════════════════════════════════════════════════════════════
def _drive(rig: Rig, lay, mode: str, horizon_s: float, *,
           audit: bool = False, sleep_at_night: bool = False,
           trace_closed_form: bool = False, frozen: bool = False) -> dict:
    """Step the shipped layer to death or to the horizon.

    `mode` selects which needs the scripted homeostat MEETS:
        "none"  nothing            "water" drink on demand
        "food"  eat on demand      "both"  the sated homeostat
        "soak"  sit in the pool    "roof"  sit under full occlusion
    """
    t = 0.0
    rec: dict = {}
    # (b)'s reconstruction, from PUBLISHED diagnostics only.
    r_e, r_w = lay.state.e, lay.state.w
    drain_e = drain_w = intake_e = intake_w = clip_e = clip_w = 0.0
    max_dev_e = max_dev_w = 0.0
    # (a)'s closed-form trace.
    p0 = lay.state.p
    cf_e = cf_w = cf_p = 0.0
    cf_until = min(CLOSED_FORM_WINDOW_S, horizon_s)
    t_min_c, t_max_c = lay.state.T, lay.state.T
    s_below_cold = s_above_hot = 0.0
    ate_prev = dict(lay.ate_total)
    drank_prev = lay.drank_total

    while not lay.dead and t < horizon_s - 1e-9:
        s = lay.state
        if mode == "soak":
            place = "pool"
        elif mode == "roof":
            place = "roof"
        elif mode in ("water", "both") and s.w < NEED_TOPUP:
            place = "pool"
        elif mode in ("food", "both") and s.e < NEED_TOPUP:
            place = "eat"
        else:
            place = "land"
        rig.place(place)
        asleep = bool(sleep_at_night and place == "land" and lay.is_night())
        lay.set_asleep(asleep)

        lay.begin_decision()
        lay.substep(rig.model, rig.data, DT_S)
        lay.decide()
        t += DT_S
        new = lay.state

        if audit:
            # Everything on the right-hand side is PUBLISHED by the layer.
            dt, pw = lay.last_dt, lay.last_power_w
            m_rate = N.metabolic_rate(pw, s.T, lay.kappa_act)
            d_e = (m_rate / N.M_BASAL) * N.B_E * dt
            d_w = (1.0 + N.C_SW * max(0.0, s.T - N.T_SETPOINT)) * N.B_W * dt
            i_e = sum((lay.ate_total[k] - ate_prev[k]) * N.FOOD[k][0]
                      for k in lay.ate_total)
            i_w = (lay.drank_total - drank_prev) * N.NU_DRINK
            ate_prev = dict(lay.ate_total)
            drank_prev = lay.drank_total
            pre_e, pre_w = r_e - d_e + i_e, r_w - d_w + i_w
            post_e = float(np.clip(pre_e, 0.0, 1.0))
            post_w = float(np.clip(pre_w, 0.0, 1.0))
            drain_e += d_e; drain_w += d_w
            intake_e += i_e; intake_w += i_w
            clip_e += post_e - pre_e; clip_w += post_w - pre_w
            r_e, r_w = post_e, post_w
            max_dev_e = max(max_dev_e, abs(r_e - new.e))
            max_dev_w = max(max_dev_w, abs(r_w - new.w))

        if trace_closed_form and t <= cf_until + 1e-9:
            cf_e = max(cf_e, abs(new.e - (1.0 - N.B_E * t)))
            cf_w = max(cf_w, abs(new.w - (1.0 - N.B_W * t)))
            cf_p = max(cf_p, abs(new.p - (1.0 - (1.0 - p0)
                                          * math.exp(-t / N.TAU_WAKE))))
        t_min_c = min(t_min_c, new.T)
        t_max_c = max(t_max_c, new.T)
        # Dwell inside each lethal band. `_check_death` needs DEATH_T_S = 20 s
        # of CONTINUOUS dwell, so a body that dips past a threshold and climbs
        # straight back out is not killed by it — the distinction that decides
        # how the control's "27.5 C" reads.
        s_below_cold += DT_S if new.T <= N.T_COLD_DEATH else 0.0
        s_above_hot += DT_S if new.T >= N.T_HOT_DEATH else 0.0
        if frozen and not lay.dead:
            # The FROZEN-NEEDS null: re-pin every meter to its setpoint through
            # the layer's OWN api, so the world clock, the food respawn timers
            # and the death machinery all keep running while the needs do not.
            # Checked AFTER `dead`, never before — resetting a body that has
            # just died would make the null unfalsifiable by construction.
            lay.new_body()

    rec.update({
        "life_s": float(lay.death_record["t"]) if lay.dead else float(t),
        "dead": float(lay.dead),
        "cause": lay.death_record["cause"] if lay.dead else "",
        "t_min_c": float(t_min_c), "t_max_c": float(t_max_c),
        "s_below_cold": s_below_cold, "s_above_hot": s_above_hot,
        "e_end": float(lay.state.e), "w_end": float(lay.state.w),
        "p_end": float(lay.state.p), "T_end": float(lay.state.T),
        "wet_end": float(lay.state.skin_wetness),
        "eat_events": int(sum(lay.ate_total.values())),
        "drink_events": int(lay.drank_total),
        "cf_err_e": cf_e, "cf_err_w": cf_w, "cf_err_p": cf_p,
        "cf_window_s": float(cf_until),
        "resid_e": max_dev_e, "resid_w": max_dev_w,
        "drain_e": drain_e, "drain_w": drain_w,
        "intake_e": intake_e, "intake_w": intake_w,
        "clip_e": clip_e, "clip_w": clip_w,
        "horizon_s": float(horizon_s),
    })
    return rec


def _fit_tau(ts, xs, fixed_point: float, lo: float, hi: float) -> tuple:
    """Time constant by log-linear relaxation about `fixed_point`.

    W.1's lesson, applied to the sleep clock: never read a declared constant.
    The band [lo, hi] on |x - fixed_point| is fixed before the run and keeps
    the fit off both the initial transient and the floating-point floor.
    """
    t, y = [], []
    for a, b in zip(ts, xs):
        d = abs(b - fixed_point)
        if lo <= d <= hi and d > 0.0:
            t.append(a)
            y.append(math.log(d))
    if len(t) < 50:
        return float("nan"), len(t)
    slope = float(np.polyfit(np.array(t), np.array(y), 1)[0])
    return (-1.0 / slope if slope < 0 else float("nan")), len(t)


def _sleep_clocks(rig: Rig, seed: int) -> dict:
    """tau_wake then tau_sleep, both measured, both on a body kept alive by
    the sated homeostat so the thirst clock cannot truncate the fit."""
    lay = rig.layer(seed)
    ts, ps = [], []
    t = 0.0
    while not lay.dead and t < WAKE_FIT_S:
        s = lay.state
        rig.place("pool" if s.w < NEED_TOPUP
                  else ("eat" if s.e < NEED_TOPUP else "land"))
        lay.set_asleep(False)
        lay.begin_decision(); lay.substep(rig.model, rig.data, DT_S); lay.decide()
        t += DT_S; ts.append(t); ps.append(lay.state.p)
    tau_w, n_w = _fit_tau(ts, ps, 1.0, TAU_FIT_LO, TAU_FIT_HI)

    ts2, ps2 = [], []
    t2 = 0.0
    while not lay.dead and t2 < SLEEP_FIT_S:
        s = lay.state
        rig.place("pool" if s.w < NEED_TOPUP
                  else ("eat" if s.e < NEED_TOPUP else "land"))
        lay.set_asleep(True)
        lay.begin_decision(); lay.substep(rig.model, rig.data, DT_S); lay.decide()
        t2 += DT_S; ts2.append(t2); ps2.append(lay.state.p)
    tau_s, n_s = _fit_tau(ts2, ps2, 0.0, TAU_FIT_LO_SLEEP, TAU_FIT_HI)

    ratio = tau_w / tau_s if tau_s and np.isfinite(tau_s) else float("nan")
    return {
        "d_tau_wake_s": tau_w, "d_tau_sleep_s": tau_s,
        "d_n_wake": float(n_w), "d_n_sleep": float(n_s),
        "d_tau_wake_declared_s": float(N.TAU_WAKE),
        "d_tau_sleep_declared_s": float(N.TAU_SLEEP),
        "d_ratio": ratio,
        "d_ratio_rel_err": abs(ratio - SLEEP_RATIO) / SLEEP_RATIO,
        "d_ratio_rel_err_vs_rounded": abs(ratio - SLEEP_RATIO_ROUNDED) / SLEEP_RATIO_ROUNDED,
        "d_fit_died": float(lay.dead),
    }


# ══════════════════════════════════════════════════════════════════════════
# THE EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════
def _experiment(seed: int) -> dict:
    rig = Rig(seed)
    probe = rig.layer(seed)
    m: dict = dict(rig.preconditions(probe))

    # ── (a) closed forms + the thirst deadline, on the DEPRIVED life ──
    dep = _drive(rig, rig.layer(seed), "none", DEPRIVED_HORIZON_S,
                 trace_closed_form=True)
    m["a_err_e"], m["a_err_w"], m["a_err_p"] = (dep["cf_err_e"], dep["cf_err_w"],
                                                dep["cf_err_p"])
    m["a_window_s"] = dep["cf_window_s"]
    m["a_worst"] = max(dep["cf_err_e"], dep["cf_err_w"], dep["cf_err_p"])
    m["a_pass"] = float(m["a_worst"] <= INTEGRATOR_TOL)
    # The regime the closed forms assume, asserted rather than believed: by day
    # the dry statue's T is an EXACT fixed point of the shipped thermal step.
    m["rig_thermoneutral_day"] = float(
        abs(dep["t_max_c"] - N.T_SETPOINT) <= THERMONEUTRAL_EPS_C
        and abs(dep["t_min_c"] - N.T_SETPOINT) <= THERMONEUTRAL_EPS_C)

    # _round6 on the aggregate would erase a 1e-13 residual to 0.0, so the
    # order of magnitude ships beside it and the record keeps what was measured.
    m["a_worst_log10"] = float(np.log10(max(m["a_worst"], 1e-300)))

    m["c_thirst_s"] = dep["life_s"]
    m["c_thirst_cause"] = dep["cause"]
    m["c_thirst_is_dehydration"] = float(dep["cause"] == "dehydration")
    m["c_thirst_from_constants_s"] = float(1.0 / N.B_W + N.DEATH_W_S)

    # ── (c) the hunger deadline: water met, food not ──────────────────
    wat = _drive(rig, rig.layer(seed), "water", DEPRIVED_HORIZON_S)
    m["c_hunger_s"] = wat["life_s"]
    m["c_hunger_cause"] = wat["cause"]
    m["c_hunger_is_starvation"] = float(wat["cause"] == "starvation")
    m["c_hunger_from_constants_s"] = float(1.0 / N.B_E + N.DEATH_E_S)
    m["c_hunger_drinks"] = float(wat["drink_events"])

    # ── (b) conservation over a 10-Jack-day SATED life ────────────────
    sat = _drive(rig, rig.layer(seed), "both", LIFE_S, audit=True,
                 sleep_at_night=True)
    for k in ("resid_e", "resid_w", "drain_e", "drain_w", "intake_e",
              "intake_w", "clip_e", "clip_w", "eat_events", "drink_events",
              "life_s", "dead", "e_end", "w_end", "t_min_c", "t_max_c"):
        m["b_" + k] = sat[k]
    m["b_cause"] = sat["cause"]
    m["needs_ledger_error"] = float(max(sat["resid_e"], sat["resid_w"]))
    m["b_resid_log10"] = float(np.log10(max(m["needs_ledger_error"], 1e-300)))
    m["b_pass"] = float(m["needs_ledger_error"] <= CONSERVATION_TOL_REL)
    m["b_alive_at_10d"] = float(not sat["dead"])
    # The intake channels must have FIRED, or (b) audited an empty ledger.
    m["rig_sated_ate"] = float(sat["eat_events"] > 0)
    m["rig_sated_drank"] = float(sat["drink_events"] > 0)

    # ── (c) the temperature bounds, and whether the world can reach them ──
    m["c_cold_threshold_c"] = float(N.T_COLD_DEATH)
    m["c_hot_threshold_c"] = float(N.T_HOT_DEATH)
    m["c_cold_threshold_ok"] = float(abs(N.T_COLD_DEATH - HUMAN_T_COLD_C) < 1e-9)
    m["c_hot_threshold_ok"] = float(abs(N.T_HOT_DEATH - HUMAN_T_HOT_C) < 1e-9)

    soak = _drive(rig, rig.layer(seed), "soak", REACH_HORIZON_S)
    roof = _drive(rig, rig.layer(seed), "roof", REACH_HORIZON_S)
    m["c_cold_reachable"] = float(soak["cause"] == "hypothermia")
    m["c_cold_route_life_s"] = soak["life_s"]
    m["c_cold_route_t_min_c"] = soak["t_min_c"]
    m["c_hot_reachable"] = float(roof["cause"] == "hyperthermia")
    m["c_hot_route_life_s"] = roof["life_s"]
    m["c_hot_route_t_max_c"] = roof["t_max_c"]
    # Dry cold, at the world's own night ambient, must NOT kill (W.1's finding
    # measured from the needs side). Reported, and it is the reason the soaked
    # route is the only route.
    m["c_dry_night_t_min_c"] = wat["t_min_c"]
    m["c_dry_night_kills_cold"] = float(wat["cause"] == "hypothermia")

    # ── (c) the deadline arithmetic, both forms, per the REGISTRATION NOTE ──
    m["c_k_declared"] = K_DECLARED
    m["c_thirst_human_s"] = HUMAN_THIRST_S
    m["c_hunger_human_s"] = HUMAN_HUNGER_S
    m["c_thirst_pred_s"] = HUMAN_THIRST_S / K_DECLARED
    m["c_hunger_pred_s"] = HUMAN_HUNGER_S / K_DECLARED
    m["c_thirst_ratio"] = m["c_thirst_pred_s"] / max(m["c_thirst_s"], 1e-9)
    m["c_hunger_ratio"] = m["c_hunger_pred_s"] / max(m["c_hunger_s"], 1e-9)
    m["c_hunger_ratio_at_basal"] = (m["c_hunger_pred_s"]
                                    / m["c_hunger_from_constants_s"])
    m["c_thirst_deadline_ok"] = float(
        1.0 / DEADLINE_TOL_FACTOR <= m["c_thirst_ratio"] <= DEADLINE_TOL_FACTOR)
    m["c_hunger_deadline_ok"] = float(
        1.0 / DEADLINE_TOL_FACTOR <= m["c_hunger_ratio"] <= DEADLINE_TOL_FACTOR)
    # How far the bar would have to move to pass — so the reader can see that
    # the verdict does not turn on the tolerance this file chose.
    m["c_widest_tol_that_would_pass"] = float(
        max(m["c_thirst_ratio"], m["c_hunger_ratio"]))

    m["c_pass"] = float(m["c_thirst_deadline_ok"] == 1.0
                        and m["c_hunger_deadline_ok"] == 1.0
                        and m["c_cold_threshold_ok"] == 1.0
                        and m["c_hot_threshold_ok"] == 1.0
                        and m["c_cold_reachable"] == 1.0
                        and m["c_hot_reachable"] == 1.0)

    # ── the implied-k table: the finding behind (c), and W.7's input ─────
    m["k_from_day_length"] = K_DECLARED
    m["k_from_thermal_tau"] = 70.0 * 3470.0 / (7.7 * 1.8481) / (N.C_EFF / N.K_DRY)
    m["k_from_tau_wake"] = SLEEP_TAU_WAKE_H * 3600.0 / N.TAU_WAKE
    m["k_from_tau_sleep"] = SLEEP_TAU_SLEEP_H * 3600.0 / N.TAU_SLEEP
    m["k_from_thirst"] = HUMAN_THIRST_S / m["c_thirst_from_constants_s"]
    m["k_from_hunger"] = HUMAN_HUNGER_S / m["c_hunger_from_constants_s"]
    ks = [m["k_from_day_length"], m["k_from_thermal_tau"], m["k_from_tau_wake"],
          m["k_from_tau_sleep"], m["k_from_thirst"], m["k_from_hunger"]]
    m["k_spread_factor"] = max(ks) / min(ks)
    m["c_thirst_in_jack_days"] = m["c_thirst_s"] / N.SIM_DAY_S
    m["c_hunger_in_jack_days"] = m["c_hunger_s"] / N.SIM_DAY_S
    m["c_hunger_over_thirst_w0"] = m["c_hunger_s"] / max(m["c_thirst_s"], 1e-9)
    m["c_hunger_over_thirst_human"] = HUMAN_HUNGER_S / HUMAN_THIRST_S

    # ── (d) the sleep clocks, both MEASURED ───────────────────────────
    m.update(_sleep_clocks(rig, seed))
    m["d_pass"] = float(np.isfinite(m["d_ratio"])
                        and m["d_ratio_rel_err"] <= SLEEP_RATIO_TOL_FRAC)
    m["d_margin_frac_of_bar"] = float(
        1.0 - m["d_ratio_rel_err"] / SLEEP_RATIO_TOL_FRAC)

    # ── the registered NULL: FROZEN-NEEDS must never die ──────────────
    frz = _drive(rig, rig.layer(seed), "none", LIFE_S, frozen=True)
    m["null_frozen_dead"] = frz["dead"]
    m["null_frozen_life_s"] = frz["life_s"]
    m["null_frozen_cause"] = frz["cause"]

    m["all_finite"] = float(all(
        np.isfinite(v) for v in m.values() if isinstance(v, float)))
    m["four_checks_pass"] = float(m["a_pass"] == 1.0 and m["b_pass"] == 1.0
                                  and m["c_pass"] == 1.0 and m["d_pass"] == 1.0)

    rig_ok = all(m[k] == 1.0 for k in (
        "rig_pool_mouth_near_surface", "rig_pool_head_not_submerged",
        "rig_land_sky_open", "rig_food_touches_mouth", "rig_food_sky_open",
        "rig_roof_occludes_all", "rig_thermoneutral_day", "rig_sated_ate",
        "rig_sated_drank"))
    m["rig_ok"] = float(rig_ok)
    if not rig_ok:
        bad = [k for k in m if k.startswith("rig_") and m[k] == 0.0]
        m["void_reason"] = ("VOID — scripted intervention did not take effect: "
                            + ",".join(bad))
    return m


# ══════════════════════════════════════════════════════════════════════════
# THE REGISTERED CONTROLS
# ══════════════════════════════════════════════════════════════════════════
def _control(seed: int) -> dict:
    """CONTROL 1: the SATED agent must survive an arbitrarily long life.
    CONTROL 2: meeting one need removes exactly its own death mode, no other.

    Control 2 is implemented by MEETING the need through the world's own
    affordances rather than by patching a constant out of `needs.py`. That is
    the stronger form: it exercises the eating and drinking paths a policy
    would use, so a specificity failure here would also be a real failure for
    a real agent.
    """
    rig = Rig(seed)
    c: dict = {}
    sat = _drive(rig, rig.layer(seed), "both", LIFE_S, sleep_at_night=True)
    c["ctl_sated_alive_at_10d"] = float(not sat["dead"])
    c["ctl_sated_life_s"] = sat["life_s"]
    c["ctl_sated_cause"] = sat["cause"]
    c["ctl_sated_eat_events"] = float(sat["eat_events"])
    c["ctl_sated_drink_events"] = float(sat["drink_events"])
    c["ctl_sated_t_min_c"] = sat["t_min_c"]

    none = _drive(rig, rig.layer(seed), "none", DEPRIVED_HORIZON_S)
    water = _drive(rig, rig.layer(seed), "water", DEPRIVED_HORIZON_S)
    food = _drive(rig, rig.layer(seed), "food", DEPRIVED_HORIZON_S)
    c["ctl_meet_none_cause"] = none["cause"]
    c["ctl_meet_none_life_s"] = none["life_s"]
    c["ctl_meet_water_cause"] = water["cause"]
    c["ctl_meet_water_life_s"] = water["life_s"]
    c["ctl_meet_food_cause"] = food["cause"]
    c["ctl_meet_food_life_s"] = food["life_s"]
    # Specificity, stated as the three propositions the registry asks for.
    c["ctl_water_removes_thirst_death"] = float(water["cause"] != "dehydration")
    c["ctl_water_leaves_hunger_death"] = float(water["cause"] == "starvation")
    c["ctl_food_leaves_thirst_death"] = float(
        food["cause"] == "dehydration"
        and abs(food["life_s"] - none["life_s"]) < DT_S + 1e-9)
    c["ctl_specificity_ok"] = float(
        c["ctl_water_removes_thirst_death"] == 1.0
        and c["ctl_water_leaves_hunger_death"] == 1.0
        and c["ctl_food_leaves_thirst_death"] == 1.0)

    # ── the "27.5 C" reading, both ways, MEASURED (see the docstring) ──
    body = _drive(rig, rig.layer(seed, state=N.NeedState(T=27.5)), "both",
                  REACH_HORIZON_S)
    c["ctl_body275_cause"] = body["cause"]
    c["ctl_body275_life_s"] = body["life_s"]
    c["ctl_body275_survives"] = float(not body["dead"])
    c["ctl_body275_s_below_cold"] = body["s_below_cold"]
    c["ctl_body275_grace_s"] = float(N.DEATH_T_S)
    c["ctl_body275_t_end_c"] = body["T_end"]
    denom = N.C_SH + N.K_DRY
    c["ctl_ambient275_parked_body_c"] = float(
        (N.M_BASAL + N.C_SH * N.T_SETPOINT + N.K_DRY * 27.5) / denom)
    c["ctl_ambient275_survivable"] = float(
        N.T_COLD_DEATH < c["ctl_ambient275_parked_body_c"] < N.T_HOT_DEATH)
    # Both readings satisfy the registered clause, so the label is undecidable
    # from the control — reported so no future reader resolves it by analogy.
    c["ctl_275_reading_decidable"] = float(
        c["ctl_body275_survives"] != c["ctl_ambient275_survivable"])
    return c


def _check(m: dict, c: dict) -> bool:
    """PRE-REGISTERED. Four checks, both controls, the null, and every
    instrument shown alive. Not one bar is derived from the data.
    """
    # RIG HEALTH first: a scripted intervention that did not take effect makes
    # every arm downstream of it a fiction. VOID, never FAIL (`void_reason`
    # is already set in the metrics, which the recorder reads).
    if m.get("rig_ok", 0.0) != 1.0:
        return False

    # THE CLAIM: all four registered checks, finite.
    experiment_ok = (m["four_checks_pass"] == 1.0 and m["all_finite"] == 1.0)

    # CONTROL 1: the sated agent survives 10 Jack-days, and the independent
    # run in `_control` agrees with the one check (b) audited.
    control1_ok = (c["ctl_sated_alive_at_10d"] == 1.0
                   and m["b_alive_at_10d"] == 1.0)
    # CONTROL 2: specificity — meeting a need removes its own death and no other.
    control2_ok = (c["ctl_specificity_ok"] == 1.0)

    # THE NULL: frozen meters must never die. If this fires, W0's lethality is
    # a clock and every deadline above is measuring the wrong thing.
    null_ok = (m["null_frozen_dead"] == 0.0)

    return bool(experiment_ok and control1_ok and control2_ok and null_ok)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID[SPEC_ID], _experiment, _check, control_fn=_control,
                    ledger=ledger)
