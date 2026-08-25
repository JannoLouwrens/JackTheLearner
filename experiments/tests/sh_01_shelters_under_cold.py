"""SH.01 — Under cold, he shelters — and prefers the shelter that works.

HYPOTHESIS (registry, unchanged). With a thermal drive active, time spent
sheltered rises far above an otherwise identical agent whose thermal drive is
disabled, sheltering BEGINS before the lethal threshold rather than after it,
and when offered two shelters he prefers the one that actually retains heat.

THE WORLD. Two wind-breaks (`playground._shelter_fragments`) at (-1, 1) and
(1, 1) — geometrically and visually IDENTICAL, verified by `thermal._smoke`
against the live model. The fire is placed 50 m away (`FIRE_DIST`), which by
equation (2) makes its warmth ~exp(-(50/1.5)^2) ~ 0: THE SHELTER IS THE ONLY
REFUGE. That is pre-registered and deliberate — the registry's kills clause
worries about "an agent that survives cold by some other means", so the world
offers no other means; PS.02 already certified the fire's warmth separately.
Which hut WORKS is redrawn per life (playground textures geoms by id, so a
fixed assignment would leak through vision — playground.py's own caution), and
the huts differ in nothing an eye can see: the only discriminator in the whole
world is the felt warmth inside the working one.

## HOW THE COLD REACHES THE POLICY, SAID OUT LOUD

The certified cores read a FIXED modality dict (`cores.MODALITIES` + the
6-dim `placebo` slot); widening it voids LC.01/LC.02's admission and
throughput certificates — the T2.02 obs-dim scar, and `thermal.py`'s own
docstring says a new sense stays an overlay until a spec pays for the
widening. This spec does not pay it. Instead the 2-float thermal sense
(PS.02's certified channel) rides in the core's PLACEBO INPUT SLOT, zero-
padded to 6: `[core, skin, 0, 0, 0, 0]` replaces W0's placebo noise,
IDENTICALLY IN EVERY ARM (learner, twin, control — all of them feel; only the
reward term differs). Within this test the slot is just an input the certified
architecture already owns; no control here relies on placebo being noise, and
the substitution is declared here rather than discovered in a diff. If SH.01
passes, it is the evidence the contract-widening decision has been waiting
for — that goes to DECISIONS_NEEDED, not into this file.

Without the sense in the observation the preference clause is unlearnable IN
PRINCIPLE, not merely hard: the huts are visually identical, the assignment is
redrawn per life, and the policy is memoryless — no reactive policy could
tell the working hut from the cosmetic one except by feeling it.

## ARMS (the LOOP_JOURNAL pre-registration of 2026-08-19, implemented as written)

* LEARNER — `cores.build_arm("ppo-needs")` trained by `cores.lc_update`
  through `survival._TargetRing`/`_gae` (the certified update, not a fork;
  the two-kernels lesson). Reward is the homeo-dr shape on the thermal
  deviation and NOTHING else:

      r_t = d_th(t) - d_th(t+1),   d_th = |Tb - TB_HEALTHY| / CORE_SPAN

  with death charged as d_th(t) - 1.0 (the deviation of the least-deviant
  dead body, `survival.D_DEATH`'s lower-bound rule; |TB_LETHAL - TB_HEALTHY|
  = CORE_SPAN exactly).
* DRIVE-DISABLED TWIN — byte-identical loop: same world, same lethality, same
  observation (thermal sense INCLUDED), same architecture, same init seed,
  same exploration schedule, same update cadence. ONLY the reward term is
  zeroed (r = 0 everywhere, terminal included). Its binding loss still trains
  the encoders, so the twin is "him without the motive", not "him unplugged".
* RANDOM WALK — `w0.random_action` at matched decisions (the registry's
  second null; also the world-lethality tripwire: if random lives don't
  freeze, the world cannot test the claim and the run is VOID).
* ORACLE — the must-succeed REFERENCE (T3.07's pattern): identical to the
  learner in every byte except that the placebo slot additionally carries
  the unit direction to the working hut. If even privileged perception
  cannot learn to shelter on this rig, the training budget — not the
  shipped system — is what failed, and the run is VOID, not FAIL. The
  learner's FAIL is licensed ONLY by the oracle's success.

## GATES (pre-registered after a seed-90 pilot, disjoint from seeds 0/1/2)

1. SHELTER CONTRAST: per-seed Welch z on per-life sheltered fractions,
   learner vs twin, over lives starting in the final EVAL_FRAC of the
   decision budget: z >= Z_MIN in EVERY seed. The CONTRAST is gated, never
   the absolute — the twin may sit in huts for enclosure reasons, which is
   exactly what the twin and the cosmetic control exist to subtract.
2. PREFERENCE: pooled eval-window occupancy, working / (working + cosmetic)
   >= PREF_MIN in every seed, with at least MIN_HUT_DEC hut-decisions so the
   ratio has a denominator. The assignment redraw must itself be balanced
   (BALANCE_LO..HI) or the preference could be a site preference wearing a
   thermal costume.
3. ANTICIPATION: median over eval lives of time-to-lethal at FIRST shelter
   entry — computed against the OUTSIDE law (`time_to_lethal_s(tb, t_cold)`),
   i.e. the time he still had when he chose to shelter — >= LEAD_MIN_S in
   every seed. Foresight vs reflex, the registry's first-class metric.

CONTROL (registry: THE COSMETIC SHELTER, "disable the mechanism" direction).
The identical learner — drive ACTIVE, cold still killing — in a world where
BOTH huts are cosmetic (equation (4) never fires; geoms unchanged). Location
then buys nothing, the reward is policy-independent, and the shelter contrast
must COLLAPSE (z < Z_MIN vs the same seed's twin). If it survives its own
sabotage, the "sheltering" was never thermal and the test measures nothing.

VOID, not FAIL, when the run could not test the claim: refused borrow
(PS.01's j0/alpha), an optimiser that never stepped, a world whose random
lives never freeze, or huts no arm ever entered (unreachable geometry is a
rig fault, not a refuted motive).

## THE SPAWN CURRICULUM, AND WHY IT IS NOT PUPPETEERING

Pilot v1 (seed 90, 2026-08-19, N=3000, in docs/LOOP_JOURNAL.md) measured the
rig dead: hut occupancy was 0.0 in EVERY arm — learner, twin AND random — over
~9,000 decisions, while 31 random lives ended frozen. The cause is arithmetic, not
misfortune: outside a hut the thermal field is spatially FLAT (the fire is
50 m away by this spec's own design), so the shaped reward carries zero
spatial gradient, and the probability of blundering through a 3-walled
opening during a 22–45 s life is ~0 (measured 0 entries in ~2,900 random
decisions). The learner and its twin therefore receive byte-identical
training signal forever — the claim's CONTRAST had no headroom (the BA.02
lesson, caught by pilot instead of by three VOIDs this time).

The amendment is in the TEST, not the world (no IMPL_DEPS cascade): a
pre-registered fraction of lives (CURRICULUM_FRAC) spawns INSIDE a hut, drawn
per life from the same per-life RNG in EVERY arm — learner, twin, random,
control all get the identical spawn schedule, so no arm differs from another
in where it wakes up. This is GOAL.md's "their hands may leave things in his
world for him to find — never puppeteering": being born under a roof is not
being told to seek one. What keeps the claim honest is the scoring rule:
EVERY gate — occupancy contrast, preference, anticipation — is computed ONLY
over lives spawned OUTSIDE. A learner that merely stays where it was left
scores nothing; only sheltering it SOUGHT counts.

Contrast headroom, measured before registration (seed-90 probes, journal):
a hand-crafted P-controller that knows the hut location enters 4/12 lives
(median entry 13.5 s, median lead 10.9 s; failures = stuck on arena clutter),
so entry from an outside spawn is physically possible and the LEAD_MIN_S
gate is reachable. Detection verified: shelter_index()=0 when placed at the
hut centre.

## PILOT RECORD, AND WHY THE REGISTERED RUN HAS NOT BEEN LAUNCHED

PILOT v2 (seed 90, N=3000/arm, curriculum active, 2026-08-19): the
curriculum delivers the experience (inside-spawn lives shelter from birth)
but the learner shows ZERO transfer to seeking — eval z_shelter 0.0,
hut_dec 0, 29 lives. ORACLE PILOT (same seed, same budget): the reference
arm ALSO reads 0.0 — with the exact working-hut direction in its
observation, 1469 optimiser steps produce 1 hut-touching outside life of
21. By this spec's own reference gate that run would record VOID: the rig
at N=3000 cannot produce the behaviour under ANY perception, so nothing is
attributable to the shipped learner. The registered run is deliberately
NOT launched until an oracle pilot at a larger budget (N scaled toward the
full cpu<2h envelope, ~N=10000/arm) shows the reference learning; if it
cannot, the finding belongs to the learning-core bakeoff (LC.04), not to
this spec's ledger row.

ORACLE PILOT AT THE FULL ENVELOPE (seed 90, N=10000/arm, 2026-08-24 23:13 ->
23:29 UTC, experiments/sh01_oracle_pilot.py, artifact
/data/sh01_oracle_pilot.json, launched at commit bdac2af): **ORACLE_CANNOT.**
z_shelter 0.0 (frac_shelt 0.0 in BOTH arms' eval lives), pref_working 0.0,
hut_dec 0.0, over n_eval_lives 27 (oracle) / 24 (twin). The instrument was
alive — the oracle logged 3,100 shelter-decisions across all 83 lives
(curriculum inside-spawns, which no gate scores) and froze in fewer lives
than the twin (74/83 vs 89/92), so huts shelter and the cold kills; the
VOID carve-outs (dead optimiser, unreachable huts, non-lethal world) do not
apply. What failed is TRANSFER TO SEEKING: 4,969 optimiser steps with the
exact working-hut direction IN THE OBSERVATION produce zero sheltering from
an outside spawn at the full cpu<2h envelope. Per the decision rule above,
now fired: **SH.01 stays unlaunched — no ledger row, no envelope growth, no
re-roll.** The finding is D10 evidence (fourth instrument: the certified
ppo-needs core cannot learn W0 behaviours at reachable envelopes even under
privileged perception — agreeing with LC.03 v2, where ppo-needs was a
non-learner at a 4x envelope). Any path forward is the D10 redesign
(learning core and/or world), not this spec's compute.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from ..render import ensure_gl

ensure_gl()

from .. import drives, thermal                                    # noqa: E402
from ..cores import ACTION_DIM, LC_BATCH, build_arm, lc_update, \
    make_optimizer                                                # noqa: E402
from ..protocol import Ledger, Status, borrow_metrics, run_spec   # noqa: E402
from ..registry import BY_ID                                      # noqa: E402
from ..survival import EXPLORE_STD, SEG, _gae, _TargetRing, \
    _updates_due                                                  # noqa: E402
from ..w0 import SIM_S_PER_DECISION, W0, random_action            # noqa: E402

# The claim is about the world, the body and the certified learner together.
IMPL_DEPS = ["experiments/thermal.py", "experiments/w0.py", "playground.py",
             "experiments/cores.py", "experiments/survival.py",
             "experiments/drives.py"]

# ── the world (pre-registered) ──────────────────────────────────────────
HUT_GEOMS = (("hutA", -1.0, 1.0), ("hutB", 1.0, 1.0))
HUT_XY = ((-1.0, 1.0), (1.0, 1.0))
FIRE_DIST = 50.0             # the fire is out of reach: shelter or freeze
N_DECISIONS = 6000           # per arm-seed; 1200 simulated seconds
LIFE_CAP = 300               # decisions; 60 s — a life the cold cannot end
TRAIN_RATIO = 0.5            # updates per decision (survival smoke's regime)
EVAL_FRAC = 1.0 / 3.0        # occupancy scored on lives starting in final 1/3
CURRICULUM_FRAC = 0.3        # lives spawning INSIDE a hut (identical schedule
                             # in every arm; NO gate ever scores these lives)
ARM = "ppo-needs"

# ── pre-registered gates (margins set after the seed-90 pilot) ──────────
Z_MIN = 3.0                  # the registry's own bar; it does not move
PREF_MIN = 0.70              # working-hut share of all hut time
MIN_HUT_DEC = 50             # eval hut-decisions before a ratio is evidence
LEAD_MIN_S = 5.0             # median seconds of life left at first entry
BALANCE_LO, BALANCE_HI = 0.25, 0.75   # per-life working-hut redraw balance
RAND_FROZEN_MIN = 5          # random lives that must end frozen (lethality)

_CACHE: dict = {}


def _calibration() -> tuple:
    b = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    if not b.ok:
        return None, None, {**b.provenance, "borrow_refusal": b.refusal}
    return b.values["j0_ms"], b.values["alpha"], b.provenance


def _core_obs(tw: thermal.ThermalWorld,
              oracle_xy: Optional[tuple] = None) -> Dict[str, np.ndarray]:
    """The certified modality dict, thermal riding the placebo slot.

    See the module docstring: [core, skin, 0, 0, 0, 0] replaces W0's placebo
    noise in EVERY arm, so no arm differs from another in what it can feel.
    The REFERENCE arm alone (T3.07's must-succeed pattern) additionally gets
    the unit direction to the working hut in dims 2-3 — privileged perception
    that licenses reading the shipped learner's failure as FAIL, not VOID.
    """
    obs = tw.observe()
    th = obs.pop("thermal")
    pad = np.zeros(obs["placebo"].shape[0], dtype=np.float32)
    pad[0], pad[1] = th[0], th[1]
    if oracle_xy is not None:
        d = np.asarray(oracle_xy, dtype=np.float64) - np.asarray(tw._xy())
        n = float(np.linalg.norm(d))
        if n > 1e-6:
            pad[2], pad[3] = d[0] / n, d[1] / n
    obs["placebo"] = pad
    return obs


def _d_th(tb: float) -> float:
    """Thermal deviation in CORE_SPAN units — the homeo-dr distance."""
    return abs(tb - thermal.TB_HEALTHY) / thermal.CORE_SPAN


def _run_arm(seed: int, mode: str, n_decisions: int = N_DECISIONS) -> dict:
    """One arm-seed: `n_decisions` of lethal cold, lives ending frozen/capped.

    mode  "learner"  thermal drive on, trained
          "twin"     reward identically zero, everything else byte-identical
          "random"   random_action, no core
          "ctrl"     the learner, both huts cosmetic (the registry's control)
          "oracle"   the learner + privileged hut direction (the must-succeed
                     reference; its failure VOIDs the run, per T3.07)
    """
    j0, alpha, _ = _calibration()
    w = W0(seed=seed, j0=j0, alpha=alpha, lethal=False, shelters=HUT_GEOMS)
    spawns = w.legal_spawns()
    mj = w.mujoco

    core = opt = ring = None
    if mode in ("learner", "twin", "ctrl", "oracle"):
        torch.manual_seed(9_000 + seed)          # same init for all arms
        core = build_arm(ARM)
        core.eval()
        opt = make_optimizer(core)
    rng = np.random.RandomState(seed * 6553 + 11)
    gen = torch.Generator().manual_seed(seed * 7919 + 3)

    lives: List[dict] = []
    optimiser_steps = 0
    k = 0
    life = 0
    t0 = time.perf_counter()

    while k < n_decisions:
        life_seed = seed * 1000 + life
        lrng = np.random.RandomState(life_seed * 131 + 7)
        work_a = bool(lrng.randint(2))           # redrawn per life
        if mode == "ctrl":
            shel = (HUT_XY[0] + (False,), HUT_XY[1] + (False,))
        else:
            shel = (HUT_XY[0] + (work_a,), HUT_XY[1] + (not work_a,))
        widx = 0 if work_a else 1                # label kept even under ctrl

        # the curriculum draw precedes the spawn draw so every arm, sharing
        # life_seed, gets the identical schedule (inside/outside AND which hut)
        inside_spawn = bool(lrng.uniform() < CURRICULUM_FRAC)
        c_hut = int(lrng.randint(2))
        sp = spawns[lrng.randint(len(spawns))]
        if inside_spawn:
            sp = HUT_XY[c_hut]
        w._place(float(sp[0]), float(sp[1]))
        mj.mj_forward(w.model, w.data)
        w.drives.state = drives.DriveState()
        w._prev_drive = drives.DriveState()
        tw = thermal.ThermalWorld(w, seed=life_seed, fire_dist=FIRE_DIST,
                                  shelters=shel)
        t_cold = tw.state.t_cold

        orc = HUT_XY[widx] if mode == "oracle" else None

        if ring is None and core is not None:
            ring = _TargetRing(_core_obs(tw, orc))

        seg_obs: list = []
        seg_act: list = []
        seg_rew: list = []
        seg_val: list = []

        def _close(boot: float, terminal: bool):
            nonlocal seg_obs, seg_act, seg_rew, seg_val, optimiser_steps
            if not seg_rew:
                return
            adv, tgt = _gae(seg_rew, seg_val, boot, terminal)
            for i in range(len(seg_rew)):
                ring.push(seg_obs[i], seg_act[i], float(tgt[i]),
                          float(adv[i]))
            seg_obs, seg_act, seg_rew, seg_val = [], [], [], []

        n_dec = shelt = work = cosm = 0
        lead_s = float("nan")
        entered = False
        frozen = False
        start_k = k

        while k < n_decisions and n_dec < LIFE_CAP:
            obs = _core_obs(tw, orc)
            if mode == "random":
                a = random_action(rng)
                v = 0.0
            else:
                obs_t = {kk: torch.from_numpy(vv).unsqueeze(0)
                         for kk, vv in obs.items()}
                with torch.no_grad():
                    s = core.shared_state(obs_t, dropped=W0.DROPPED)
                    mean = core.act(obs_t, s).squeeze(0).numpy()
                    v = float(core.critic(s))
                frac = min(1.0, k / max(1, n_decisions - 1))
                std = EXPLORE_STD[0] + (EXPLORE_STD[1] - EXPLORE_STD[0]) * frac
                a = np.clip(mean + std * rng.randn(ACTION_DIM), -1.0, 1.0)

            d_before = _d_th(tw.state.tb)
            tw.decide(a, SIM_S_PER_DECISION)
            frozen = tw.frozen
            d_after = 1.0 if frozen else _d_th(tw.state.tb)
            r = 0.0 if mode == "twin" else (d_before - d_after)

            idx = tw.shelter_index()
            if idx >= 0:
                shelt += 1
                if idx == widx:
                    work += 1
                else:
                    cosm += 1
                if not entered:
                    entered = True
                    # the time he still HAD, by the outside law, when he
                    # first put a wall between himself and the night
                    lead_s = thermal.time_to_lethal_s(tw.state.tb, t_cold)

            if core is not None:
                seg_obs.append({kk: torch.from_numpy(vv.copy())
                                for kk, vv in obs.items()})
                seg_act.append(torch.from_numpy(a.astype(np.float32)))
                seg_rew.append(r)
                seg_val.append(v)
                if frozen or len(seg_rew) >= SEG:
                    _close(boot=v, terminal=frozen)
                if ring.n >= LC_BATCH:
                    for _ in range(_updates_due(k + 1, TRAIN_RATIO)):
                        batch, targets = ring.sample(gen)
                        lc_update(core, batch, targets, opt,
                                  dropped=W0.DROPPED)
                        optimiser_steps += 1

            n_dec += 1
            k += 1
            if frozen:
                break

        lives.append({"start_k": start_k, "n": n_dec, "shelt": shelt,
                      "work": work, "cosm": cosm, "lead_s": lead_s,
                      "frozen": frozen, "widx": widx,
                      "inside_spawn": inside_spawn,
                      "full": frozen or n_dec >= LIFE_CAP})
        life += 1

    return {"lives": lives, "optimiser_steps": optimiser_steps,
            "wall_s": time.perf_counter() - t0,
            "finite": float(bool(np.all(np.isfinite(w.data.qpos))
                                 and np.all(np.isfinite(w.data.qvel))))}


# ── scoring ─────────────────────────────────────────────────────────────
def _eval_lives(run: dict, n_decisions: int = N_DECISIONS) -> List[dict]:
    """Gate-eligible lives: late, complete, and spawned OUTSIDE.

    Inside-spawned (curriculum) lives never reach a gate — occupancy an agent
    was handed is not occupancy it sought (module docstring)."""
    lo = n_decisions * (1.0 - EVAL_FRAC)
    return [L for L in run["lives"]
            if L["start_k"] >= lo and L["full"] and not L["inside_spawn"]]


def _fracs(lives: List[dict]) -> np.ndarray:
    return np.array([L["shelt"] / L["n"] for L in lives if L["n"] > 0])


def _welch_z(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / max(se, 1e-9))


def _pooled(lives: List[dict], key: str) -> int:
    return int(sum(L[key] for L in lives))


def _score(learn: dict, twin: dict, n_decisions: int = N_DECISIONS) -> dict:
    eL, eT = _eval_lives(learn, n_decisions), _eval_lives(twin, n_decisions)
    fL, fT = _fracs(eL), _fracs(eT)
    work, cosm = _pooled(eL, "work"), _pooled(eL, "cosm")
    hut = work + cosm
    leads = [L["lead_s"] for L in eL if np.isfinite(L["lead_s"])]
    bal = np.mean([L["widx"] == 0 for L in learn["lives"]]) \
        if learn["lives"] else 0.5
    return {
        "z_shelter": _welch_z(fL, fT),
        "frac_shelt_learn": float(fL.mean()) if len(fL) else 0.0,
        "frac_shelt_twin": float(fT.mean()) if len(fT) else 0.0,
        "pref_working": work / hut if hut else 0.0,
        "hut_dec": float(hut),
        "lead_med_s": float(np.median(leads)) if leads else 0.0,
        "n_eval_lives_learn": float(len(eL)),
        "n_eval_lives_twin": float(len(eT)),
        "balance": float(bal),
    }


def _experiment(seed: int) -> dict:
    j0, _, prov = _calibration()
    if j0 is None:
        return {"borrow_ok": 0.0}
    learn = _run_arm(seed, "learner")
    twin = _run_arm(seed, "twin")
    rand = _run_arm(seed, "random")
    orc = _run_arm(seed, "oracle")            # the must-succeed reference
    _CACHE[seed] = twin                       # the control contrasts vs it
    s = _score(learn, twin)
    s_ref = _score(orc, twin)

    rand_frozen = sum(L["frozen"] for L in rand["lives"])
    eR = _eval_lives(rand)
    # the unreachable-geometry tripwire must not be satisfied by occupancy
    # the curriculum handed out: count OUTSIDE-spawned entries only
    hut_any = sum(_pooled([L for L in r["lives"] if not L["inside_spawn"]],
                          "shelt")
                  for r in (learn, twin, rand))
    m = {
        "borrow_ok": 1.0,
        **s,
        "frac_shelt_rand": float(_fracs(eR).mean()) if eR else 0.0,
        "rand_frozen": float(rand_frozen),
        "n_lives_learn": float(len(learn["lives"])),
        "n_lives_twin": float(len(twin["lives"])),
        "n_lives_rand": float(len(rand["lives"])),
        "mean_life_s_learn": float(np.mean(
            [L["n"] for L in learn["lives"]])) * SIM_S_PER_DECISION,
        "mean_life_s_twin": float(np.mean(
            [L["n"] for L in twin["lives"]])) * SIM_S_PER_DECISION,
        "optimiser_steps": float(learn["optimiser_steps"]),
        "twin_optimiser_steps": float(twin["optimiser_steps"]),
        "ref_z_shelter": s_ref["z_shelter"],
        "ref_frac_shelt": s_ref["frac_shelt_learn"],
        "ref_ok": float(s_ref["z_shelter"] >= Z_MIN),
        "hut_dec_any_arm": float(hut_any),
        "physics_finite": min(learn["finite"], twin["finite"],
                              rand["finite"]),
        "wall_s_learn": learn["wall_s"],
    }
    m["seed_gates_ok"] = float(
        m["z_shelter"] >= Z_MIN
        and m["pref_working"] >= PREF_MIN and m["hut_dec"] >= MIN_HUT_DEC
        and m["lead_med_s"] >= LEAD_MIN_S
        and BALANCE_LO <= m["balance"] <= BALANCE_HI
        and m["rand_frozen"] >= RAND_FROZEN_MIN
        and m["physics_finite"] == 1.0)
    m["void_rig"] = float(
        m["optimiser_steps"] == 0.0 or m["rand_frozen"] == 0.0
        or m["hut_dec_any_arm"] == 0.0)
    return m


def _control(seed: int) -> dict:
    """The cosmetic-shelter world: same learner, same drive, nothing to find.

    Must fail to reproduce the shelter contrast — if directed 'sheltering'
    appears where no shelter works, the contrast was never thermal.
    """
    j0, _, _ = _calibration()
    if j0 is None or seed not in _CACHE:
        return {"ctrl_caught": 0.0, "ctrl_z_shelter": 0.0}
    ctrl = _run_arm(seed, "ctrl")
    s = _score(ctrl, _CACHE[seed])
    return {"ctrl_z_shelter": s["z_shelter"],
            "ctrl_frac_shelt": s["frac_shelt_learn"],
            "ctrl_pref_labelled": s["pref_working"],
            "ctrl_caught": float(s["z_shelter"] < Z_MIN)}


def _check(m: dict, c: dict):
    if m.get("borrow_ok", 0.0) != 1.0:
        return Status.VOID           # an uncalibrated world refutes nothing
    if m.get("void_rig", 0.0) > 0.0:
        return Status.VOID           # the rig, not the claim, failed
    if m.get("ref_ok", 0.0) != 1.0:
        return Status.VOID           # the privileged reference could not
                                     # learn -> the rig cannot produce the
                                     # behaviour and the shipped learner's
                                     # weakness is not attributable (T3.07)
    return bool(m["seed_gates_ok"] == 1.0
                and c["ctrl_caught"] == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["SH.01"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":
    # seed-90 pilot, disjoint from the registered seeds (PG.6/SM.01/PS.02
    # precedent). Short budget; prints what the gates need with margin.
    import sys
    n = 3000 if "--pilot" in sys.argv else N_DECISIONS
    t0 = time.perf_counter()
    learn = _run_arm(90, "learner", n)
    print(f"learner: {learn['wall_s']:.0f}s wall, "
          f"{learn['optimiser_steps']} steps, "
          f"{len(learn['lives'])} lives")
    twin = _run_arm(90, "twin", n)
    rand = _run_arm(90, "random", n)
    s = _score(learn, twin, n)
    eR = _eval_lives(rand, n)
    print({k: round(v, 4) for k, v in s.items()})
    print("rand frac:", round(float(_fracs(eR).mean()), 4) if eR else 0.0,
          "rand frozen:", sum(L["frozen"] for L in rand["lives"]),
          "total wall:", round(time.perf_counter() - t0, 1))
