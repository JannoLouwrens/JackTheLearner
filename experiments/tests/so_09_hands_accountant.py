"""SO.09 — A life the hands bought is not evidence, and the harness says so.

GOAL.md's sentence is *"Never puppeteering: what is left must still be found,
learned, and chosen by him."* SO.06 certified the channel; SO.07 will score
what it buys; THIS spec is the accountant that makes the sentence enforceable:
every provisioning event is logged, every run that would claim learning gets
`hand_share` and `hand_contact_frac` computed against ceilings declared HERE,
before any such run exists, and a run over its ceiling is REFUSED — not
reported with a caveat. de Haan et al. 2019 is why the ceiling sits on the
hand's CONTRIBUTION rather than on the outcome: a helper in the world is
correlated with success by construction, and only withdrawal plus a log can
separate them. And PURPOSE_AND_SCAFFOLDING.md §2.6(ii) is why this is an
empirical accountant rather than a shaping theorem: potential-based shaping is
provably VACUOUS on W0's zero-reward world, so no invariance guarantee is
available and safety has to be measured.

THE ACCOUNTANT (`audit()`), a pure function of a recorded trace + a hand log.
Four independent detectors, any of which refuses the run:

  1. hand_share    — the fraction of need-restoration (consumption) events
                     causally downstream of a logged placement: same object,
                     eaten within `WINDOW_S` of the drop. Over
                     `HAND_SHARE_MAX` -> REFUSED.
  2. hand_contact_frac — the fraction of placements landing within
                     `CONTACT_DIST_M` of the body at drop time. This is
                     SO.07's C-GIVE policy made a number; the ceiling is
                     `HAND_CONTACT_MAX = 0.0` because a contact placement IS
                     puppeteering, not a matter of degree.
  3. unlogged provisioning — a food geom that moves more than `TELEPORT_M`
                     between adjacent decisions is a provisioning event (no
                     process in W0 teleports food: eating starts a respawn
                     timer and leaves the geom in place, and passive spheres
                     do not cross 1.5 m in 0.2 s). Every such event must match
                     a log row (same object, same decision window, position
                     within `POS_TOL_M`); an unmatched one -> REFUSED. This is
                     what upgrades an unlogged placement from *unreported* to
                     *mechanically impossible to miss*.
  4. the extended energy identity (owned here per the registration note):
                     Sum_t max(0, delta e_t) == Sum_f nu_f n_f + Sum_hands
                     nu_g n_g, reconciled against `drives.DriveLayer.ate_total`
                     (already keyed per source — and in SO.06's relocation
                     venue the hand's gifts ARE food geoms, so both sums read
                     the same counter and the reconciliation is: recovered
                     intake `e_t - e_{t-1} + drain_t` must equal
                     `nu_f * delta ate_total_f` at every decision). A positive
                     delta-e with no logged consumption event (the struck
                     C-GIVE variant: "restores e directly") leaves a residual
                     -> REFUSED. CLIP CAVEAT, stated rather than hidden:
                     `DriveLayer.decide` clips e into [0, 1], and a clipped
                     decision destroys the equality's information, so those
                     decisions are excluded from the residual and their count
                     is REPORTED (`n_unreconciled`). A hack hiding entirely
                     inside clipped decisions is not caught by leg 4; legs 1-3
                     do not share the blind spot.

WHAT IS MEASURED (seed 0, one seed — the accountant is deterministic and
every gate is an exact invariant or a known answer; nothing here is a noisy
learning claim).

  CLAIM LEG — a measured CLEAN life: 300 decisions, an honest hand that drops
  the farthest food item at `DROP_RANGE_M` from the body whenever
  `e < E_FLOOR` (cooldown `COOLDOWN_S`), every drop logged via SO.06's `Hand`.
  The accountant must ACCEPT it, with exact bookkeeping: every drop detected
  as a teleport, every teleport matched to a log row, contact fraction exactly
  0.0, residual at float dust. Venue fact measured before registration (probe,
  seed 0): e crosses 0.90 near t=24 s and no food is eaten by the random
  policy in 60 s, so acceptance does not hinge on the policy's luck.

  KNOWN-ANSWER LEGS — two synthesised traces with hand_share computed by
  construction: 3 of 4 restorations attributed (0.75 > ceiling, MUST refuse)
  and 1 of 4 (0.25 < ceiling, MUST accept). These pin the attribution
  arithmetic policy-free, because the measured clean life's share is
  degenerate (no restorations at all).

THE CONTROLS, all of which must fire (the registry's control clause plus the
two the falsifier names):

  C-GIVE   — a MEASURED puppeteered life: same trigger, but every placement in
             body contact (`CGIVE_GAP_M` from the torso axis). The registry
             pre-authorised a SYNTHESISED log here because SO.07 has not run;
             executing the declared policy in the live world is strictly
             stronger than synthesising its log, and the registry's re-buy
             note stands: when SO.07's real C-GIVE arm exists, its log is
             replayed through this accountant as a re-buy. MUST be refused
             (contact_frac 1.0 > 0.0).
  UNLOGGED — the clean hand's drops with the log line withheld
             (`Hand.place(log=False)`, decisions 40 and 90). Both must be
             detected as provisioning events, matched to nothing, REFUSED.
  DIRECT-E — the struck C-GIVE variant, as telemetry: the accepted clean
             trace with `+DIRECT_E_DELTA` written into e from one decision on
             (exactly what a live `state.e` write produces in the recorded
             stream). The identity must leave a residual of DIRECT_E_DELTA
             and REFUSE. This is G-A's own positive control ("the identity
             must be violated and the detector must say so").

VOID LANES, instrument death rather than refutation: a life in which the
e-trigger never fired (no drop, nothing exercised), or a drop whose
displacement sits under `TELEPORT_M` (the detector's floor is the venue's
geometry; at seed 0 the farthest food is 3-7 m out, so this lane is theory).

VENUE: no file outside `experiments/tests/` is touched. The `Hand` is
IMPORTED from SO.06's module rather than re-derived (the LG.02/SO.08 rule:
one mechanism, one implementation), and that file is declared in IMPL_DEPS so
a change to the channel reads as drift on this certificate too.
"""

from __future__ import annotations

import math

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from .. import drives
from ..w0 import W0
from .so_06_hand_reaches_through_world import Hand

# The verdict composes over the world, the body, the drive layer and the
# channel: a change to any of them must read as drift, not as a certificate.
IMPL_DEPS = ["playground.py", "experiments/w0.py", "experiments/drives.py",
             "experiments/tests/so_06_hand_reaches_through_world.py"]

# --- the lives --------------------------------------------------------------
LIFE_DECISIONS = 300          # 60 simulated seconds at 0.2 s per decision
J0, ALPHA = 1.0, 0.02         # PS.01's calibrated impact channel, as SO.06
E_FLOOR = 0.90                # the hand drops when e falls below this
COOLDOWN_S = 10.0             # and not more often than this
DROP_RANGE_M = 3.0            # an honest drop lands here (SO.06's reach)
CGIVE_GAP_M = 0.19            # a puppeteer's drop: touching the torso
                              # (capsule r 0.07 + food r 0.131 - 0.011 overlap)
UNLOGGED_DROPS = (40, 90)     # the unlogged leg's scheduled sabotage decisions

# --- the DECLARED CEILINGS and the accountant's constants -------------------
# Declared here, before SO.07 or any provisioned learning run exists. Per-spec
# ceilings may be stricter than these; they may never be looser without the
# strengthen-only lane.
WINDOW_S = 30.0               # attribution: eaten within this after the drop
HAND_SHARE_MAX = 0.5          # over this fraction hand-fed -> not evidence
HAND_CONTACT_MAX = 0.0        # a contact placement is puppeteering, full stop
CONTACT_DIST_M = 0.75         # horizontal m from body centre = "in contact"
TELEPORT_M = 1.5              # no passive food crosses this in one decision
POS_TOL_M = 0.25              # a matched teleport must land where the log says
RESID_TOL = 1e-6              # the identity's float-dust allowance
DIRECT_E_DELTA = 0.05         # the direct-restore sabotage's size


# ── the accountant ──────────────────────────────────────────────────────────
def audit(trace: dict, log: list) -> dict:
    """The pure function every provisioned run must pass through.

    `trace` is per-decision telemetry: t, e, dt, power_w, ate (cumulative
    per-source counts), body_xy, food_pos. `log` is the hand's own record
    (SO.06 `Hand.log` rows). Returns the report, including `refused`.
    """
    t = trace["t"]
    e = trace["e"]
    n = len(t)
    nu = dict(drives.FOOD_GEOMS)
    foods = [f for f in trace["ate"]]

    # 1. consumption events and hand attribution
    events = []                                   # (k, name)
    for f in foods:
        c = trace["ate"][f]
        for k in range(n):
            prev = c[k - 1] if k else 0
            if c[k] > prev:
                events.append((k, f))
    attributed = 0
    for k, f in events:
        if any(r["object"] == f and r["t"] <= t[k] <= r["t"] + WINDOW_S
               for r in log):
            attributed += 1
    hand_share = attributed / len(events) if events else 0.0

    # 2. contact placements
    contact = 0
    for r in log:
        # the body sample taken at the drop instant: the drop happens between
        # decisions, at the previous decision's clock, so its t equals a
        # sample's t exactly; fall back to the first later sample.
        k = next((i for i in range(n) if t[i] >= r["t"]), n - 1)
        bx, by = trace["body_xy"][k]
        d = math.hypot(r["position"][0] - bx, r["position"][1] - by)
        if d < CONTACT_DIST_M:
            contact += 1
    contact_frac = contact / len(log) if log else 0.0

    # 3. teleports vs the log — the unlogged-provisioning detector
    teleports, matched = [], 0
    for f in foods:
        p = trace["food_pos"][f]
        for k in range(1, n):
            if float(np.linalg.norm(p[k] - p[k - 1])) > TELEPORT_M:
                teleports.append((k, f))
    for k, f in teleports:
        ok = any(r["object"] == f and t[k - 1] <= r["t"] < t[k]
                 and float(np.linalg.norm(
                     np.asarray(r["position"]) - trace["food_pos"][f][k]))
                 < POS_TOL_M
                 for r in log)
        matched += int(ok)
    n_unlogged = len(teleports) - matched
    log_unmatched = len(log) - sum(
        1 for r in log if any(
            r["object"] == f and t[k - 1] <= r["t"] < t[k]
            for k, f in teleports))

    # 4. the extended energy identity, per decision, clip-aware
    resid, unrec = 0.0, 0
    for k in range(n):
        e_prev = e[k - 1] if k else trace["e0"]
        if e[k] in (0.0, 1.0) or e_prev in (0.0, 1.0):
            unrec += 1
            continue
        drain = (drives.BASAL_B * trace["dt"][k]
                 + drives.KAPPA * trace["power_w"][k] * trace["dt"][k])
        expect = sum(nu[f] * (trace["ate"][f][k]
                              - (trace["ate"][f][k - 1] if k else 0))
                     for f in foods)
        resid += abs(e[k] - e_prev + drain - expect)

    over_share = hand_share > HAND_SHARE_MAX
    over_contact = contact_frac > HAND_CONTACT_MAX
    unlogged_bad = n_unlogged > 0
    resid_bad = resid > RESID_TOL
    return {
        "hand_share": hand_share, "n_restorations": float(len(events)),
        "n_attributed": float(attributed),
        "hand_contact_frac": contact_frac, "n_placements": float(len(log)),
        "n_teleports": float(len(teleports)), "n_matched": float(matched),
        "n_unlogged": float(n_unlogged),
        "n_log_unmatched": float(log_unmatched),
        "residual_abs": float(resid), "n_unreconciled": float(unrec),
        "over_share": float(over_share), "over_contact": float(over_contact),
        "refused": float(over_share or over_contact or unlogged_bad
                         or resid_bad),
    }


# ── the lives that feed it ──────────────────────────────────────────────────
def _farthest_food(w: W0, hands: dict) -> str:
    p = np.asarray(w.data.xpos[w.rover_bid][:2], dtype=float)
    return max(hands, key=lambda f: float(np.linalg.norm(
        np.asarray(w.data.geom_xpos[hands[f].gid][:2]) - p)))


def _drop_target(w: W0, contact: bool, radius: float) -> np.ndarray:
    """An honest target DROP_RANGE_M out on the first in-bounds bearing, or a
    puppeteer's target touching the torso axis."""
    p = np.asarray(w.data.xpos[w.rover_bid], dtype=float)
    if contact:
        return np.array([p[0] + CGIVE_GAP_M, p[1], radius])
    arena = float(w.params.arena_size) - 0.3
    for j in range(16):
        a = 2.0 * math.pi * j / 16 + math.pi / 16
        t = np.array([p[0] + DROP_RANGE_M * math.cos(a),
                      p[1] + DROP_RANGE_M * math.sin(a), radius])
        if abs(t[0]) < arena and abs(t[1]) < arena:
            return t
    return np.array([0.0, 0.0, radius])          # centre is always in bounds


def _life(seed: int, policy: str) -> tuple[dict, list, dict]:
    """One recorded life. `policy`: "clean" (honest, logged), "give"
    (contact placements, logged), "unlogged" (honest drops, log withheld)."""
    w = W0(seed=seed, j0=J0, alpha=ALPHA, lethal=False)
    rng = np.random.RandomState(seed * 7717 + 3)
    hands = {}
    for name in drives.FOOD_GEOMS:
        try:
            hands[name] = Hand("owner", w, name, rng)
        except (KeyError, ValueError):
            continue
    trace = {"t": [], "e": [], "dt": [], "power_w": [], "body_xy": [],
             "ate": {f: [] for f in hands},
             "food_pos": {f: [] for f in hands},
             "e0": float(w.drives.state.e)}
    drops = {"n": 0, "min_disp": float("inf")}
    last_drop_t = -1e9
    for k in range(LIFE_DECISIONS):
        want = False
        if k >= 1:
            if policy == "unlogged":
                want = k in UNLOGGED_DROPS
            else:
                want = (w.drives.state.e < E_FLOOR
                        and w.sim_seconds - last_drop_t >= COOLDOWN_S)
        if want:
            f = _farthest_food(w, hands)
            h = hands[f]
            before = np.asarray(w.data.geom_xpos[h.gid], dtype=float).copy()
            target = _drop_target(w, contact=(policy == "give"),
                                  radius=h.radius)
            h.place(target, log=(policy != "unlogged"))
            drops["n"] += 1
            drops["min_disp"] = min(
                drops["min_disp"], float(np.linalg.norm(target - before)))
            last_drop_t = w.sim_seconds
        w.decide(rng.uniform(-1.0, 1.0, w.action_dim))
        trace["t"].append(float(w.sim_seconds))
        trace["e"].append(float(w.drives.state.e))
        trace["dt"].append(float(w.drives.last_dt))
        trace["power_w"].append(float(w.drives.last_power_w))
        trace["body_xy"].append(
            tuple(float(v) for v in w.data.xpos[w.rover_bid][:2]))
        for f, h in hands.items():
            trace["ate"][f].append(int(w.drives.ate_total.get(f, 0)))
            trace["food_pos"][f].append(
                np.asarray(w.data.geom_xpos[h.gid], dtype=float).copy())
    log = sorted((r for h in hands.values() for r in h.log),
                 key=lambda r: r["t"])
    return trace, log, drops


# ── the known-answer synthesised traces ─────────────────────────────────────
def _synth(share_high: bool) -> tuple[dict, list]:
    """A 12-decision trace whose hand_share is exact by construction:
    3 of 4 restorations attributed (0.75) or 1 of 4 (0.25). Energy series
    built to satisfy the identity; one teleport co-timed with the one log row."""
    n, dt = 12, 1.0
    nu = drives.NU_FLOORFOOD
    t = [float(k + 1) for k in range(n)]
    eat_a = [3, 5, 7] if share_high else [3]          # obj0, attributed
    eat_b = [9] if share_high else [5, 7, 9]          # obj1, no placement
    ate = {"obj0": [], "obj1": []}
    e, e0 = [], 0.3
    prev = e0
    ca = cb = 0
    for k in range(n):
        ca += int(k in eat_a)
        cb += int(k in eat_b)
        ate["obj0"].append(ca)
        ate["obj1"].append(cb)
        prev = prev - drives.BASAL_B * dt + nu * (int(k in eat_a)
                                                 + int(k in eat_b))
        e.append(prev)
    pos_a = [np.array([6.0, 6.0, 0.131])] * 2 + [np.array([1.0, 0.0, 0.131])] * (n - 2)
    trace = {"t": t, "e": e, "dt": [dt] * n, "power_w": [0.0] * n,
             "body_xy": [(0.0, 0.0)] * n,
             "ate": ate,
             "food_pos": {"obj0": pos_a,
                          "obj1": [np.array([-6.0, 6.0, 0.131])] * n},
             "e0": e0}
    log = [{"t": 2.0, "agent": "owner", "object": "obj0",
            "position": [1.0, 0.0, 0.131], "needs": [0.3, 1.0, 0.0]}]
    return trace, log


# The clean life is measured once and read twice: the experiment audits it and
# the control edits its telemetry (T2.01's module-cache pattern — run_spec
# calls _experiment before _control in the same process).
_CLEAN: dict = {}


def _clean(seed: int):
    if seed not in _CLEAN:
        _CLEAN[seed] = _life(seed, "clean")
    return _CLEAN[seed]


def _experiment(seed: int) -> dict:
    trace, log, drops = _clean(seed)
    r = audit(trace, log)
    m = {"clean_n_drops": float(drops["n"]),
         "clean_min_disp_m": round(drops["min_disp"], 3),
         "clean_accepted": float(r["refused"] == 0.0),
         "clean_hand_share": r["hand_share"],
         "clean_contact_frac": r["hand_contact_frac"],
         "clean_n_teleports": r["n_teleports"],
         "clean_n_matched": r["n_matched"],
         "clean_n_unlogged": r["n_unlogged"],
         "clean_n_log_rows": r["n_placements"],
         "clean_residual": r["residual_abs"],
         "clean_n_unreconciled": r["n_unreconciled"],
         "clean_n_restorations": r["n_restorations"]}

    ta, la = _synth(share_high=True)
    tb, lb = _synth(share_high=False)
    ra, rb = audit(ta, la), audit(tb, lb)
    m.update({
        "synth_hi_share": ra["hand_share"],
        "synth_hi_share_exact": float(abs(ra["hand_share"] - 0.75) < 1e-12),
        "synth_hi_refused": ra["refused"],
        "synth_hi_residual": ra["residual_abs"],
        "synth_lo_share": rb["hand_share"],
        "synth_lo_share_exact": float(abs(rb["hand_share"] - 0.25) < 1e-12),
        "synth_lo_refused": rb["refused"],
    })
    m["hand_share_audited"] = float(
        m["clean_accepted"] == 1.0
        and m["clean_contact_frac"] == 0.0
        and m["clean_n_unlogged"] == 0.0
        and m["clean_n_teleports"] == m["clean_n_matched"] == m["clean_n_drops"]
        and m["clean_n_log_rows"] == m["clean_n_drops"]
        and m["clean_residual"] <= RESID_TOL
        and m["synth_hi_share_exact"] == 1.0 and m["synth_hi_refused"] == 1.0
        and m["synth_hi_residual"] <= RESID_TOL
        and m["synth_lo_share_exact"] == 1.0 and m["synth_lo_refused"] == 0.0)
    return m


def _control(seed: int) -> dict:
    # C-GIVE, measured: every placement in body contact. MUST be refused.
    tg, lg, dg = _life(seed, "give")
    rg = audit(tg, lg)
    c = {"c_give_n_drops": float(dg["n"]),
         "c_give_refused": rg["refused"],
         "c_give_contact_frac": rg["hand_contact_frac"],
         "c_give_over_contact": rg["over_contact"],
         "c_give_hand_share": rg["hand_share"],
         "c_give_n_restorations": rg["n_restorations"]}

    # UNLOGGED: the same honest drops with the log line withheld. MUST refuse.
    tu, lu, du = _life(seed, "unlogged")
    ru = audit(tu, lu)
    c.update({"c_unl_n_drops": float(du["n"]),
              "c_unl_min_disp_m": round(du["min_disp"], 3),
              "c_unl_refused": ru["refused"],
              "c_unl_n_teleports": ru["n_teleports"],
              "c_unl_n_matched": ru["n_matched"],
              "c_unl_n_unlogged": ru["n_unlogged"]})

    # DIRECT-E: the accepted clean trace with a direct restore written in.
    tc, lc, _ = _clean(seed)
    te = {k: (list(v) if isinstance(v, list) else v) for k, v in tc.items()}
    te["ate"] = tc["ate"]
    te["food_pos"] = tc["food_pos"]
    kk = LIFE_DECISIONS // 2
    te["e"] = list(tc["e"][:kk]) + [x + DIRECT_E_DELTA for x in tc["e"][kk:]]
    rd = audit(te, lc)
    c.update({"c_de_refused": rd["refused"],
              "c_de_residual": rd["residual_abs"]})
    return c


def _check(m: dict, c: dict):
    # VOID: the trigger never fired or a drop sat under the detector's floor —
    # the instrument was never exercised; nothing about hands is refuted.
    if (m.get("clean_n_drops", 0.0) < 1.0
            or c.get("c_give_n_drops", 0.0) < 1.0
            or c.get("c_unl_n_drops", 0.0) < 1.0):
        return Status.VOID
    if (m.get("clean_min_disp_m", 0.0) <= TELEPORT_M
            or c.get("c_unl_min_disp_m", 0.0) <= TELEPORT_M):
        return Status.VOID
    return (m.get("hand_share_audited", 0.0) == 1.0
            and c.get("c_give_refused", 0.0) == 1.0
            and c.get("c_give_contact_frac", 0.0) == 1.0
            and c.get("c_unl_refused", 0.0) == 1.0
            and c.get("c_unl_n_unlogged", 0.0) == c.get("c_unl_n_drops", -1.0)
            and c.get("c_unl_n_matched", 1.0) == 0.0
            and c.get("c_de_refused", 0.0) == 1.0
            and c.get("c_de_residual", 0.0) > RESID_TOL)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["SO.09"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
