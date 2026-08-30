"""SH.02 — Born sheltered, he stays while it is cold — and only while it is cold.

HYPOTHESIS (registry, unchanged). Spawned INSIDE a hut under lethal cold, the
certified learner's sheltered fraction is far above its drive-disabled twin's
(staying warm is CHOSEN, not inherited: a random policy drifts out through the
opening and freezes); the contrast is need-contingent (with the cold disabled,
learner and twin are indistinguishable); and it is thermally DISCRIMINATING:
lives born in the working hut hold shelter far longer than lives born in the
cosmetic one, where staying buys nothing.

SUCCESSOR TO SH.01 (parked 2026-08-25, ORACLE_CANNOT). SH.01's oracle pilot
localised the failure exactly at SEEKING: outside a hut the thermal field is
spatially FLAT (the fire is 50 m away by design), so no gradient ever reaches
the policy, and 4,969 optimiser steps with the working hut's direction IN the
observation produced zero sheltering from an outside spawn. Its curriculum
inside-spawn lives, by contrast, DID shelter from birth. This spec inverts the
geometry rather than growing the envelope: EVERY life spawns inside a hut, so
the felt-warmth gradient at the boundary is local, dense and immediate — the
signal shape the certified cores are actually screened on.

SCOPE, STATED HONESTLY (registry notes, implemented as written). This is
occupancy-MAINTENANCE, one step below SH.01's seeking and two below building.
Seeking stays with the D10/world redesign. A PASS here re-arms that discussion
with the missing half-fact — that the drive can steer behaviour when the
gradient reaches it — and nothing more. Relocation cosmetic->working (2 m) is
REPORTED as a first-class metric and NEVER gated: a life that walks from the
dead hut to the live one has done something this spec did not ask for, and
reading it as evidence for a claim it was not pre-registered against is how a
metric becomes a story.

## THE ARM IS PRE-REGISTERED, AND IT IS NOT SH.01's

`wm-latent` — LC.03 v2's ONLY 3-sigma learner (t_null 4.65 / t_twin 4.00 /
needs_rise +0.022 / clt +92.2, every conjunct green, 400k decisions/arm-seed).
NOT `ppo-needs`, which SH.01 used and which LC.03 v2 measured as a NON-learner
at a 4x envelope (t_null 1.06, data_starved). A result earned by the one core
the screen cleared is the only kind the D10 discussion can use; a second
non-learner FAIL would be a fact about `ppo-needs`, which the ladder already
has.

## THE WORLD (reused from SH.01 verbatim — the two-kernels lesson)

Two wind-breaks (`playground._shelter_fragments`) at (-1, 1) and (1, 1),
geometrically and visually IDENTICAL. Which one WORKS is redrawn per life
(a fixed assignment would leak through vision — playground.py's caution). The
fire sits `FIRE_DIST` = 50 m away, so by equation (2) its warmth is ~0: the
shelter is the only refuge, and the registry's "survives by some other means"
worry has no other means available. Equation (4) gives a working hut
`SHELTER_LEAK` = 0.15, which postpones freezing ~6.7x rather than abolishing
it — an agent that stays under the roof forever still dies, so the world stays
consequential in both directions.

Thermal riding the PLACEBO SLOT, exactly as SH.01 declares: the certified
cores read a FIXED modality dict and widening it voids LC.01/LC.02's
admission and throughput certificates (the T2.02 obs-dim scar). The 2-float
PS.02 channel is zero-padded to 6 and replaces W0's placebo noise IDENTICALLY
IN EVERY ARM — learner, twin, warm, control and reference all feel; only the
reward term and the world differ. No control here relies on placebo being
noise, and the substitution is declared rather than discovered in a diff.

## ARMS

* LEARNER — `wm-latent` trained by `cores.lc_update` through
  `survival._TargetRing`/`_gae` (the certified update, not a fork). Reward is
  the homeo-dr shape on the thermal deviation and NOTHING else:

      r_t = d_th(t) - d_th(t+1),   d_th = |Tb - TB_HEALTHY| / CORE_SPAN

  with death charged as d_th(t) - 1.0 (`survival.D_DEATH`'s lower-bound rule;
  |TB_LETHAL - TB_HEALTHY| = CORE_SPAN exactly).
* DRIVE-DISABLED TWIN — byte-identical loop: same world, same lethality, same
  observation (thermal INCLUDED), same architecture, same init seed, same
  exploration schedule, same update cadence. ONLY the reward is zeroed
  (r = 0 everywhere, terminal included). Its binding loss still trains the
  encoders, so the twin is "him without the motive", not "him unplugged".
* RANDOM WALK — `w0.random_action` at matched decisions. The registry's second
  null AND the world-lethality tripwire: if random lives born under a roof do
  not drift out and freeze, the world cannot test the claim and the run is
  VOID, not FAIL.
* WARM — the learner in a thermally inert world (`ThermalWorld(inert=True)`):
  nothing cools, nothing dies. This is the need-contingency comparator; see
  the identity disclosure below for what it can and cannot show.
* REFERENCE (must-succeed, T3.07's pattern) — identical to the learner except
  the placebo slot additionally carries the unit direction to the working hut.
  If even privileged perception cannot hold a roof it was BORN under at this
  envelope, the rig — not the shipped system — is what failed, and the run is
  VOID. The learner's FAIL is licensed only by the reference's success.

## THE NEED-CONTINGENCY CLAUSE, AND THE IDENTITY IT WOULD HAVE BEEN

The registry's second conjunct reads "with the cold disabled, learner and twin
are indistinguishable". Implemented literally — inert world, learner vs twin —
that gate is UNFALSIFIABLE BY ARITHMETIC and would have gone green forever:
under `inert=True` the body temperature never moves, so d_th is constant, so
r = d_th(t) - d_th(t+1) = 0 at every step, terminal included. The learner's
reward is then byte-identical to the twin's zero and the two arms are THE SAME
OBJECT. `warm_reward_abs` records that identity as a live assertion rather
than a claim (it must read exactly 0.0, and it would stop reading 0.0 the day
someone adds a term to the reward), and no gate is placed on it. Recording a
tautology as a passing conjunct is the `ME.9` defect this project found on
2026-08-30, one file over; see also LESSONS.md on gates that are satisfiable —
or unsatisfiable — by arithmetic rather than by evidence.

What IS gated, because it can fail: the same LEARNER, in the COLD world versus
the INERT world, at MATCHED EXPOSURE.

    z_need = welch(frac_shelt_cold_learner, frac_shelt_warm_learner) >= Z_MIN

Sheltering that is thermal must not survive the removal of the cold. The
matched-exposure truncation is load-bearing and pre-registered: inert lives
never freeze and therefore always run to `LIFE_CAP`, while cold lives end when
the body does, and a longer life is more opportunity to wander out. Comparing
untruncated fractions would bias the gate TOWARD passing. Both sides are
therefore scored over each life's FIRST `MATCH_DEC` decisions only (the
D1_CONTROL_ARCHITECTURE rule: match the exposure, report both).

CONTROL (registry: BOTH-COSMETIC, the "disable the mechanism" direction, and
the conjunct that actually carries the enclosure-preference burden). The
identical learner — drive ACTIVE, cold still lethal, reward still live and
policy-dependent — in a world where BOTH huts are cosmetic. Equation (4) never
fires anywhere, geoms unchanged, so location buys nothing and every contrast
must COLLAPSE: the shelter contrast (z < Z_MIN vs the same seed's twin) AND
the working-vs-cosmetic discrimination (which is a label with nothing behind
it there). If "sheltering" survives a world where shelter does not work, it
was never thermal and the test measures nothing.

## GATES (PROVISIONAL until the seed-90 pilot freezes them — `_GATES_FROZEN`)

1. MAINTENANCE CONTRAST: per-seed Welch z on per-life sheltered fractions,
   learner vs twin, over lives starting in the final `EVAL_FRAC` of the
   decision budget: z >= `Z_MIN` in EVERY seed. The CONTRAST is gated, never
   the absolute — a twin born under a roof will sit there for reasons that
   have nothing to do with warmth, which is exactly what the twin exists to
   subtract.
2. NEED-CONTINGENCY: `z_need` >= `Z_MIN` in every seed, at matched exposure.
3. THERMAL DISCRIMINATION: within the learner's eval lives, Welch z of
   sheltered fraction for WORKING-born lives against COSMETIC-born lives
   >= `Z_DISC` in every seed, with at least `MIN_BORN_LIVES` on each side so
   the contrast has a denominator. He cannot be said to feel warmth if he
   holds a dead roof exactly as hard as a live one.
4. Redraw balance in `BALANCE_LO..HI` (a preference for a SITE would otherwise
   read as a preference for warmth) and `physics_finite`.

VOID, not FAIL, when the run could not test the claim: a refused borrow
(PS.01's j0/alpha), an optimiser that never stepped, a world whose random
lives never freeze, a reference that cannot learn — and one more that is
specific to a born-inside design and has no analogue in SH.01:

    HEADROOM. If the TWIN or the RANDOM walk already spends more than
    `HEADROOM_MAX` of its life under the roof it was placed under, then
    "staying" is what the rig does to anyone and there is no room above it for
    a choice to show. That is the BA.02 failure shape — a contrast gated where
    the null has already saturated — and it is a rig fault, not a refutation.

Neither of the two conjunct-carrying contrasts can be manufactured by the
curriculum, because there is no curriculum: every arm gets the identical
per-life schedule (which hut works, which hut he is born in, where the fire
is), drawn from the same per-life RNG, and the arms differ only in the reward
term, the world's inertness, and what rides in the placebo slot.
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

# ── the world (reused from SH.01, unchanged) ────────────────────────────
HUT_GEOMS = (("hutA", -1.0, 1.0), ("hutB", 1.0, 1.0))
HUT_XY = ((-1.0, 1.0), (1.0, 1.0))
FIRE_DIST = 50.0             # the fire is out of reach: hold the roof or freeze
N_DECISIONS = 6000           # per arm-seed; 1200 simulated seconds
LIFE_CAP = 300               # decisions; 60 s — a life the cold cannot end
TRAIN_RATIO = 0.5            # updates per decision (survival smoke's regime)
EVAL_FRAC = 1.0 / 3.0        # gates scored on lives starting in the final 1/3
MATCH_DEC = 100              # matched exposure for the need-contingency pair
ARM = "wm-latent"            # LC.03 v2's only 3-sigma learner

# ── gates. PROVISIONAL: `run()` refuses until a seed-90 pilot freezes them ──
_GATES_FROZEN = False
_PILOT_ARTIFACT = "/data/sh02_pilot_seed90.json"

Z_MIN = 3.0                  # the registry's own bar; it does not move
Z_DISC = 3.0                 # working-born vs cosmetic-born, same learner
MIN_BORN_LIVES = 5           # per side, before a discrimination z is evidence
BALANCE_LO, BALANCE_HI = 0.25, 0.75   # per-life working-hut redraw balance
RAND_FROZEN_MIN = 5          # random lives that must end frozen (lethality)
HEADROOM_MAX = 0.85          # a null this sheltered leaves no room for a choice

_CACHE: dict = {}


def _calibration() -> tuple:
    b = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    if not b.ok:
        return None, None, {**b.provenance, "borrow_refusal": b.refusal}
    return b.values["j0_ms"], b.values["alpha"], b.provenance


def _core_obs(tw: thermal.ThermalWorld,
              oracle_xy: Optional[tuple] = None) -> Dict[str, np.ndarray]:
    """The certified modality dict, thermal riding the placebo slot.

    `[core, skin, 0, 0, 0, 0]` replaces W0's placebo noise in EVERY arm, so no
    arm differs from another in what it can feel. The REFERENCE arm alone
    additionally gets the unit direction to the working hut in dims 2-3.
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
    """One arm-seed. EVERY life spawns inside a hut; the schedule is shared.

    mode  "learner"  thermal drive on, trained, lethal cold
          "twin"     reward identically zero, everything else byte-identical
          "random"   random_action, no core
          "warm"     the learner in a thermally INERT world (see the identity
                     disclosure in the module docstring)
          "ctrl"     the learner, BOTH huts cosmetic (the registry's control)
          "oracle"   the learner + privileged working-hut direction (the
                     must-succeed reference; its failure VOIDs the run)
    """
    j0, alpha, _ = _calibration()
    w = W0(seed=seed, j0=j0, alpha=alpha, lethal=False, shelters=HUT_GEOMS)
    mj = w.mujoco

    core = opt = ring = None
    if mode != "random":
        torch.manual_seed(9_000 + seed)          # same init for all arms
        core = build_arm(ARM)
        core.eval()
        opt = make_optimizer(core)
    rng = np.random.RandomState(seed * 6553 + 11)
    gen = torch.Generator().manual_seed(seed * 7919 + 3)

    lives: List[dict] = []
    optimiser_steps = 0
    reward_abs = 0.0
    k = 0
    life = 0
    t0 = time.perf_counter()

    while k < n_decisions:
        life_seed = seed * 1000 + life
        lrng = np.random.RandomState(life_seed * 131 + 7)
        # The per-life draws happen in this order in EVERY arm, from the same
        # RNG, so no arm differs from another in the world it wakes up in.
        work_a = bool(lrng.randint(2))           # which hut works, per life
        if mode == "ctrl":
            shel = (HUT_XY[0] + (False,), HUT_XY[1] + (False,))
        else:
            shel = (HUT_XY[0] + (work_a,), HUT_XY[1] + (not work_a,))
        widx = 0 if work_a else 1                # label kept even under ctrl
        c_hut = int(lrng.randint(2))             # which hut he is BORN in
        lrng.uniform()                           # schedule parity with SH.01's
        lrng.randint(1 << 30)                    #   curriculum/spawn draws

        w._place(float(HUT_XY[c_hut][0]), float(HUT_XY[c_hut][1]))
        mj.mj_forward(w.model, w.data)
        w.drives.state = drives.DriveState()
        w._prev_drive = drives.DriveState()
        tw = thermal.ThermalWorld(w, seed=life_seed, inert=(mode == "warm"),
                                  fire_dist=FIRE_DIST, shelters=shel)

        orc = HUT_XY[widx] if mode == "oracle" else None

        if ring is None and core is not None:
            ring = _TargetRing(_core_obs(tw, orc))

        seg_obs: list = []
        seg_act: list = []
        seg_rew: list = []
        seg_val: list = []

        def _close(boot: float, terminal: bool):
            nonlocal seg_obs, seg_act, seg_rew, seg_val
            if not seg_rew:
                return
            adv, tgt = _gae(seg_rew, seg_val, boot, terminal)
            for i in range(len(seg_rew)):
                ring.push(seg_obs[i], seg_act[i], float(tgt[i]),
                          float(adv[i]))
            seg_obs, seg_act, seg_rew, seg_val = [], [], [], []

        n_dec = shelt = work = cosm = 0
        shelt_first = 0                          # matched-exposure numerator
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
            reward_abs += abs(r)

            idx = tw.shelter_index()
            if idx >= 0:
                shelt += 1
                if n_dec < MATCH_DEC:
                    shelt_first += 1
                if idx == widx:
                    work += 1
                else:
                    cosm += 1

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
                      "shelt_first": shelt_first,
                      "n_first": min(n_dec, MATCH_DEC),
                      "work": work, "cosm": cosm, "frozen": frozen,
                      "widx": widx, "born": c_hut,
                      "born_working": bool(c_hut == widx),
                      "full": frozen or n_dec >= LIFE_CAP})
        life += 1

    return {"lives": lives, "optimiser_steps": optimiser_steps,
            "reward_abs": reward_abs,
            "wall_s": time.perf_counter() - t0,
            "finite": float(bool(np.all(np.isfinite(w.data.qpos))
                                 and np.all(np.isfinite(w.data.qvel))))}


# ── scoring ─────────────────────────────────────────────────────────────
def _eval_lives(run: dict, n_decisions: int = N_DECISIONS) -> List[dict]:
    """Gate-eligible lives: late and complete. Every life spawned inside, so
    unlike SH.01 there is no outside-spawn filter to apply."""
    lo = n_decisions * (1.0 - EVAL_FRAC)
    return [L for L in run["lives"] if L["start_k"] >= lo and L["full"]]


def _fracs(lives: List[dict]) -> np.ndarray:
    return np.array([L["shelt"] / L["n"] for L in lives if L["n"] > 0])


def _fracs_matched(lives: List[dict]) -> np.ndarray:
    """Sheltered fraction over each life's FIRST MATCH_DEC decisions.

    An inert life never freezes and always runs to LIFE_CAP; a cold life ends
    when the body does. Untruncated, the need-contingency comparison would be
    reading life length, not sheltering."""
    return np.array([L["shelt_first"] / L["n_first"]
                     for L in lives if L["n_first"] > 0])


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
    bw = [L for L in eL if L["born_working"]]
    bc = [L for L in eL if not L["born_working"]]
    # relocation: cosmetic-born lives that put themselves under the LIVE roof.
    # REPORTED, NEVER GATED (registry notes).
    reloc_dec = _pooled(bc, "work")
    reloc_lives = sum(1 for L in bc if L["work"] > 0)
    bal = np.mean([L["widx"] == 0 for L in learn["lives"]]) \
        if learn["lives"] else 0.5
    return {
        "z_shelter": _welch_z(fL, fT),
        "frac_shelt_learn": float(fL.mean()) if len(fL) else 0.0,
        "frac_shelt_twin": float(fT.mean()) if len(fT) else 0.0,
        "z_disc": _welch_z(_fracs(bw), _fracs(bc)),
        "frac_born_working": float(_fracs(bw).mean()) if bw else 0.0,
        "frac_born_cosmetic": float(_fracs(bc).mean()) if bc else 0.0,
        "n_born_working": float(len(bw)),
        "n_born_cosmetic": float(len(bc)),
        "reloc_frac": (reloc_dec / max(1, _pooled(bc, "n"))) if bc else 0.0,
        "reloc_life_share": (reloc_lives / len(bc)) if bc else 0.0,
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
    warm = _run_arm(seed, "warm")             # need-contingency comparator
    orc = _run_arm(seed, "oracle")            # the must-succeed reference
    _CACHE[seed] = twin                       # the control contrasts vs it
    s = _score(learn, twin)
    s_ref = _score(orc, twin)

    eW = _eval_lives(warm)
    eL = _eval_lives(learn)
    eR = _eval_lives(rand)
    eT = _eval_lives(twin)
    m = {
        "borrow_ok": 1.0,
        **s,
        # need-contingency, at matched exposure (module docstring)
        "z_need": _welch_z(_fracs_matched(eL), _fracs_matched(eW)),
        "frac_matched_cold": float(_fracs_matched(eL).mean()) if eL else 0.0,
        "frac_matched_warm": float(_fracs_matched(eW).mean()) if eW else 0.0,
        # the disclosed identity: inert => d_th constant => r == 0 exactly
        "warm_reward_abs": float(warm["reward_abs"]),
        "frac_shelt_rand": float(_fracs(eR).mean()) if len(eR) else 0.0,
        "rand_frozen": float(sum(L["frozen"] for L in rand["lives"])),
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
        "physics_finite": min(learn["finite"], twin["finite"],
                              rand["finite"], warm["finite"]),
        "wall_s_learn": learn["wall_s"],
    }
    # HEADROOM: a null that already holds the roof leaves no room above it.
    m["headroom_twin"] = float(_fracs(eT).mean()) if len(eT) else 0.0
    m["headroom_ok"] = float(m["headroom_twin"] <= HEADROOM_MAX
                             and m["frac_shelt_rand"] <= HEADROOM_MAX)
    m["seed_gates_ok"] = float(
        m["z_shelter"] >= Z_MIN
        and m["z_need"] >= Z_MIN
        and m["z_disc"] >= Z_DISC
        and m["n_born_working"] >= MIN_BORN_LIVES
        and m["n_born_cosmetic"] >= MIN_BORN_LIVES
        and BALANCE_LO <= m["balance"] <= BALANCE_HI
        and m["physics_finite"] == 1.0)
    m["void_rig"] = float(
        m["optimiser_steps"] == 0.0
        or m["rand_frozen"] < RAND_FROZEN_MIN
        or m["headroom_ok"] == 0.0)
    return m


def _control(seed: int) -> dict:
    """The BOTH-COSMETIC world: same learner, live reward, nothing to hold.

    Must fail to reproduce EITHER contrast — if directed 'sheltering' and a
    working-vs-cosmetic preference appear where no shelter works, they were
    never thermal.
    """
    j0, _, _ = _calibration()
    if j0 is None or seed not in _CACHE:
        return {"ctrl_caught": 0.0, "ctrl_z_shelter": 0.0, "ctrl_z_disc": 0.0}
    ctrl = _run_arm(seed, "ctrl")
    s = _score(ctrl, _CACHE[seed])
    return {"ctrl_z_shelter": s["z_shelter"],
            "ctrl_z_disc": s["z_disc"],
            "ctrl_frac_shelt": s["frac_shelt_learn"],
            "ctrl_caught": float(s["z_shelter"] < Z_MIN
                                 and s["z_disc"] < Z_DISC)}


def _check(m: dict, c: dict):
    if m.get("borrow_ok", 0.0) != 1.0:
        return Status.VOID           # an uncalibrated world refutes nothing
    if m.get("void_rig", 0.0) > 0.0:
        return Status.VOID           # the rig, not the claim, failed
    if m.get("ref_ok", 0.0) != 1.0:
        return Status.VOID           # privileged perception could not hold the
                                     # roof either -> nothing is attributable
                                     # to the shipped learner (T3.07)
    return bool(m["seed_gates_ok"] == 1.0
                and c["ctrl_caught"] == 1.0)


def run(ledger: Ledger | None = None):
    if not _GATES_FROZEN:
        raise RuntimeError(
            "SH.02 gates are PROVISIONAL. Run the seed-90 pilot "
            f"(`python -m experiments.tests.sh_02_born_sheltered --pilot`), "
            f"read {_PILOT_ARTIFACT}, freeze Z_MIN/Z_DISC/HEADROOM_MAX "
            "against it in a commit, then set _GATES_FROZEN = True. "
            "A gate fitted to the run it judges is not a gate.")
    return run_spec(BY_ID["SH.02"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":
    # seed-90 pilot, disjoint from the registered seeds (PG.6/SM.01/PS.02/
    # SH.01 precedent). Writes the artifact the gate-freezing commit must cite.
    import json
    import sys

    n = 3000 if "--pilot" in sys.argv else N_DECISIONS
    seed = 90
    out: Dict[str, object] = {"seed": seed, "n_decisions": n, "arm": ARM}
    runs = {}
    for mode in ("learner", "twin", "random", "warm", "oracle", "ctrl"):
        t0 = time.perf_counter()
        runs[mode] = _run_arm(seed, mode, n)
        out[f"wall_s_{mode}"] = round(time.perf_counter() - t0, 1)
        out[f"lives_{mode}"] = len(runs[mode]["lives"])
        out[f"steps_{mode}"] = runs[mode]["optimiser_steps"]
        print(f"{mode:8s} {out[f'wall_s_{mode}']:7.1f}s  "
              f"{out[f'lives_{mode}']:4d} lives  "
              f"{out[f'steps_{mode}']:5d} steps", flush=True)

    s = _score(runs["learner"], runs["twin"], n)
    eL, eW = _eval_lives(runs["learner"], n), _eval_lives(runs["warm"], n)
    eR, eT = _eval_lives(runs["random"], n), _eval_lives(runs["twin"], n)
    out.update({k: round(float(v), 4) for k, v in s.items()})
    out["z_need"] = round(_welch_z(_fracs_matched(eL), _fracs_matched(eW)), 4)
    out["frac_matched_cold"] = round(
        float(_fracs_matched(eL).mean()) if eL else 0.0, 4)
    out["frac_matched_warm"] = round(
        float(_fracs_matched(eW).mean()) if eW else 0.0, 4)
    out["warm_reward_abs"] = runs["warm"]["reward_abs"]
    out["headroom_twin"] = round(float(_fracs(eT).mean()) if len(eT) else 0.0,
                                 4)
    out["frac_shelt_rand"] = round(
        float(_fracs(eR).mean()) if len(eR) else 0.0, 4)
    out["rand_frozen"] = int(sum(L["frozen"] for L in runs["random"]["lives"]))
    s_ref = _score(runs["oracle"], runs["twin"], n)
    out["ref_z_shelter"] = round(s_ref["z_shelter"], 4)
    out["ref_frac_shelt"] = round(s_ref["frac_shelt_learn"], 4)
    s_ctrl = _score(runs["ctrl"], runs["twin"], n)
    out["ctrl_z_shelter"] = round(s_ctrl["z_shelter"], 4)
    out["ctrl_z_disc"] = round(s_ctrl["z_disc"], 4)
    out["ctrl_frac_shelt"] = round(s_ctrl["frac_shelt_learn"], 4)

    print(json.dumps(out, indent=1))
    with open(_PILOT_ARTIFACT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", _PILOT_ARTIFACT)
