"""PS.01 unit (a) — re-derive the ENERGY economy against the drain that is paid.

> **This docstring is the pre-registration. It is committed BEFORE the module is
> run** (`LOOP_JOURNAL.md` 2026-08-10 05:30 fixed the criterion; this file fixes
> the arithmetic that implements it). The α calibration inside PS.01 is the
> worked precedent: state the criterion, solve, verify on held-out seeds. Nothing
> here may be re-solved after seeing a PS.01 verdict — that is the shape the
> journal explicitly forbade ("NOT to be done by adjusting constants until PS.01
> turns green").

## What was refuted

`PURPOSE_AND_SCAFFOLDING.md` §2.3 justified the food layout with one line:
two floor foods supply `2 × 0.08 / 90 = 1.78e-3` /s against a basal drain of
`1.67e-3` /s, therefore *"subsistence on the floor is possible and activity on
the floor is not."* PS.01 (ledger FAIL, 2026-08-10T05:29) measured what that
predicts and got the inverse: a random policy pays **6.57e-3 /s** and rests for
**3.6e-5** of its life, so it starves at t ≈ 90 s while the do-nothing statue —
which cannot eat at all — still has energy at the 600 s horizon.

The arithmetic was correct and the comparison was not: it priced supply against
**basal**, a regime nothing in a life is ever in.

## What this module adds, and why the refutation was not yet complete

The 6.57e-3 /s above is itself measured **in the regime the repair is meant to
eliminate**. PS.01's rollout applies §2.2's weakness — `ctrl *= gear_scale =
0.4 + 0.6·min(e, i)` — and its energy is pinned at 0 for 84.8% of the life, so
`gear_scale = 0.4` for most of the run and 293 W is a *starving* body's power.
The supply must fund a body that is **not** starving, or the derivation is
self-referential: it sizes the food against the drain of an agent the food has
already failed to feed.

So `P̄` is measured here at full strength (`e = i = 1` held every decision) and
as a function of DUTY CYCLE `D` — the fraction of decisions on which the policy
acts at all — because `drain(D) = b + κ·P̄(D)` assumes `P̄(D) = D·P̄(1)` and that
is an assumption about a contact-rich body, not a fact. It is measured, not
argued (`LESSONS.md`: "a claim about how a mechanism behaves is a two-line
experiment").

## THE CRITERION, fixed before the search

Let `b = BASAL_B`, `κ = KAPPA`, `S_f = 2·ν_floor / T_floor` (floor supply rate),
`S_max = S_f + ν_apple / T_apple` (every food in the world, perfectly harvested).

  **C1 — the world must be able to feed a fully active body at all.**
      `S_max ≥ ε⁻¹ · drain(1)`,  `drain(1) = b + κ·P̄(1)`.
      This is skill-independent and it is the criterion §2.3 never wrote down.
      Under the shipped numbers it is FALSE: 5.94e-3 available against 6.57e-3
      demanded even at the *weakened* power, so no policy of any competence
      could sustain constant activity. A world that cannot be survived by a
      perfect forager is not a control problem, it is a countdown.
      `ε = FORAGE_EFF = 0.8`: the world must feed an agent that misses one
      respawn in five. A perfect forager is not a policy anything can learn.

  **C2 — floor food alone subsists a body acting SOME of the time.**
      `S_f = min( PAL · b , b + κ·P̄(D_ALT) )`.
      Two independent anchors, and the SMALLER is taken — the harsher world —
      because a free parameter with two defensible settings must not be resolved
      by which one scores better (`LESSONS.md`, "prefer the filter that rejects
      less"). The anchors:
        · `PAL = 1.7` — human physical-activity level. Total energy expenditure
          sustainable over months is ~1.6–1.7× basal metabolic rate; 2.0–2.4×
          is the endurance ceiling. GOAL.md: biology is the reference
          implementation, and this is the one number in the derivation that
          comes from outside this repo.
        · `D_ALT = 0.25` — the journal's own form: pre-register a duty cycle
          `D < 1` and derive the supply from the measured drain at it.
      The duty cycle `D*` that `S_f` actually funds is DERIVED and REPORTED:
      `D* = (S_f − b) / (κ·P̄(1))`. **`0 < D* < 1` is required**; `D* ≥ 1` would
      mean floor food funds constant activity and C3 would be violated.

  **C3 — floor food alone must NOT fund constant activity.**  `S_f < drain(1)`.
      This is §2.3's design intent — *"he does not have to climb to survive; he
      has to climb to be able to do anything"* — restated against the drain an
      acting body pays instead of against basal. It is the clause that keeps the
      ladder load-bearing.

  **C4 — the statue must still lose.**  Unchanged and not re-derived: the
      do-nothing policy cannot reach food, so its energy reaches 0 at `t = 1/b`
      regardless of any supply constant. C1–C3 are what make *some* reachable
      behaviour beat it; PS.01's own domination clause is a separate defect
      (its probe cannot forage) and is routed to `INTEGRATION_QUEUE.md`, not
      repaired here.

## WHICH KNOB, fixed before the search

C1–C3 pin two RATES (`S_f`, `S_max`); the split of each rate into `(ν, T)` is
free. The rule, stated now: **move the respawn period, never the per-item
value.** `ν_apple / ν_floor` is the incentive ratio between climbing and
foraging, which §2.3 calls load-bearing; a respawn change leaves every per-item
value — and therefore that ratio — untouched, and changes only the rate the
criterion actually constrains.

## Held out

Seeds **3, 4, 5** — PS.01 runs 0, 1, 2, so every world here is one the spec has
never been scored on, and each is one `PlaygroundParams.mutate` step (PG.8's
convention) rather than a re-seeded copy of the same world.

Run:  python -m experiments.calibrations.ps01_energy
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "experiments" / "ledger.json"

# ── pre-registered, in the docstring above ──────────────────────────────
SEEDS = (3, 4, 5)                 # held out from PS.01's 0, 1, 2
DUTIES = (0.0, 0.125, 0.25, 0.5, 1.0)
N_DECISIONS = 400                 # 80 simulated seconds per (seed, duty)
SIM_S_PER_DECISION = 0.2          # w0.py's accounting unit, as in PS.01
CTRL_SCALE = 0.4                  # PS.01's random policy, unchanged
PAL = 1.7                         # human physical-activity level (C2 anchor 1)
D_ALT = 0.25                      # the journal's duty-cycle form (C2 anchor 2)
FORAGE_EFF = 0.8                  # C1: the world must feed an imperfect forager


def _measured_j0_alpha() -> tuple[float, float]:
    """`j0`/`alpha` come from the LEDGER, not from a constant retyped here.

    They are measurements PS.01 recorded (2026-08-10T05:29) and a copy of a
    measurement is a transcription that can drift — `LESSONS.md`, "when you can
    reference, reference".
    """
    e = json.loads(LEDGER.read_text())["results"]["PS.01"]["metrics"]
    return float(e["j0_ms"]), float(e["alpha"])


def _power_at_duty(seed: int, duty: float, j0: float, alpha: float) -> dict:
    """Mean mechanical power of a FULL-STRENGTH body acting on `duty` of its
    decisions. `e` and `i` are pinned at 1.0 every decision so `gear_scale`
    stays 1.0 — this measures the drain the supply has to fund, not the drain a
    body already starved by the shortfall happens to produce.
    """
    import mujoco
    import numpy as np
    sys.path.insert(0, str(REPO))
    from experiments import drives
    from experiments.tests.ps_01_drive_calibration import _build

    model, data, water, pool = _build(seed, fall=False)
    layer = drives.DriveLayer(model, j0=j0, alpha=alpha, pool=pool)
    rng = np.random.RandomState(seed * 7717 + 3)
    dt = float(model.opt.timestep)
    frame_skip = max(1, int(round(SIM_S_PER_DECISION / dt)))

    power_dt = rest_dt = total_dt = 0.0
    n_act = 0
    for _ in range(N_DECISIONS):
        layer.state = drives.DriveState(e=1.0, i=1.0, w=layer.state.w)
        layer.begin_decision()
        act = rng.random_sample() < duty
        n_act += int(act)
        ctrl = (rng.uniform(-CTRL_SCALE, CTRL_SCALE, model.nu) if act
                else np.zeros(model.nu)) * layer.gear_scale()
        for _ in range(frame_skip):
            data.ctrl[:model.nu] = ctrl
            water.apply(model, data)
            mujoco.mj_step(model, data)
            layer.substep(model, data, dt)
        layer.decide()
        power_dt += layer.last_power_w * layer.last_dt
        rest_dt += layer.last_rest_dt
        total_dt += layer.last_dt
    return {"power_w": power_dt / total_dt, "rest_frac": rest_dt / total_dt,
            "acted_frac": n_act / N_DECISIONS}


def main() -> dict:
    import numpy as np
    sys.path.insert(0, str(REPO))
    from experiments import drives

    b, kappa = drives.BASAL_B, drives.KAPPA
    j0, alpha = _measured_j0_alpha()
    print(f"j0={j0:.4f} m/s  alpha={alpha:.5f}   (from the PS.01 ledger entry)")
    print(f"b={b:.6g} /s  kappa={kappa:.6g} /J   nu_floor={drives.NU_FLOORFOOD} "
          f"nu_apple={drives.NU_APPLE}")
    print(f"\nMEAN MECHANICAL POWER at FULL STRENGTH, seeds {SEEDS}, "
          f"{N_DECISIONS} decisions each\n")
    print(f"  {'duty':>6} {'acted':>7} {'P_bar W':>10} {'+-':>7} "
          f"{'rest_frac':>10} {'drain /s':>10} {'x basal':>8}")

    pbar: dict[float, float] = {}
    rows = []
    for duty in DUTIES:
        ps = [_power_at_duty(s, duty, j0, alpha) for s in SEEDS]
        p = float(np.mean([r["power_w"] for r in ps]))
        sd = float(np.std([r["power_w"] for r in ps]))
        rf = float(np.mean([r["rest_frac"] for r in ps]))
        af = float(np.mean([r["acted_frac"] for r in ps]))
        drain = b + kappa * p
        pbar[duty] = p
        rows.append({"duty": duty, "power_w": p, "power_sd": sd,
                     "rest_frac": rf, "acted_frac": af, "drain": drain})
        print(f"  {duty:6.3f} {af:7.3f} {p:10.2f} {sd:7.2f} {rf:10.4f} "
              f"{drain:10.3e} {drain / b:8.2f}")

    p1 = pbar[1.0]
    drain1 = b + kappa * p1
    # linearity of the drain in duty, which drain(D) = b + kappa*P_bar*D assumes
    lin = [(d, pbar[d] / (d * p1)) for d in DUTIES if d > 0]
    print("\n  linearity  P(D) / (D * P(1)):  "
          + "  ".join(f"D={d:.3f}: {r:.3f}" for d, r in lin))

    # ── C2: the floor supply rate, the smaller of two anchors ───────────
    s_pal = PAL * b
    s_alt = b + kappa * pbar[D_ALT]
    s_f = min(s_pal, s_alt)
    which = "PAL" if s_pal <= s_alt else f"D_ALT={D_ALT}"
    d_star = (s_f - b) / (kappa * p1)

    # ── C1: the apple must close the gap for an imperfect forager ───────
    s_max_req = drain1 / FORAGE_EFF
    apple_req = s_max_req - s_f

    t_floor = 2.0 * drives.NU_FLOORFOOD / s_f
    t_apple = drives.NU_APPLE / apple_req

    s_f_old = 2.0 * drives.NU_FLOORFOOD / drives.RESPAWN_FLOORFOOD_S
    s_max_old = s_f_old + drives.NU_APPLE / drives.RESPAWN_APPLE_S

    print(f"\n  drain(1)            {drain1:.4e} /s   ({drain1 / b:.2f}x basal, "
          f"P_bar = {p1:.1f} W at full strength)")
    print(f"  C2 anchors: PAL*b = {s_pal:.4e}   b+k*P(D_ALT) = {s_alt:.4e}"
          f"   -> take {which}")
    print(f"  C2 S_f              {s_f:.4e} /s   ({s_f / b:.2f}x basal)")
    print(f"  C2 D* funded        {d_star:.4f}          (required 0 < D* < 1)")
    print(f"  C3 S_f < drain(1)   {s_f < drain1}   ({drain1 / s_f:.2f}x short of "
          f"constant activity)")
    print(f"  C1 S_max required   {s_max_req:.4e} /s  (drain(1) / {FORAGE_EFF})")
    print(f"     apple must supply{apple_req:.4e} /s")
    print(f"\n  SHIPPED TODAY:  S_f {s_f_old:.4e}  S_max {s_max_old:.4e}  "
          f"vs drain(1) {drain1:.4e}")
    print(f"     C1 today: {'HOLDS' if s_max_old >= s_max_req else 'VIOLATED'} — "
          f"a perfect forager harvests {s_max_old / drain1:.2f}x of what constant "
          f"activity costs")
    print("\n  DERIVED CONSTANTS (respawn moves, per-item value does not):")
    print(f"     RESPAWN_FLOORFOOD_S  {drives.RESPAWN_FLOORFOOD_S:.1f} -> "
          f"{t_floor:.1f} s")
    print(f"     RESPAWN_APPLE_S      {drives.RESPAWN_APPLE_S:.1f} -> "
          f"{t_apple:.1f} s")

    ok = (0.0 < d_star < 1.0) and (s_f < drain1)
    print(f"\n  CRITERION SATISFIABLE: {ok}")
    return {"rows": rows, "drain1": drain1, "s_f": s_f, "d_star": d_star,
            "t_floor": t_floor, "t_apple": t_apple, "anchor": which,
            "s_f_old": s_f_old, "s_max_old": s_max_old, "ok": ok}


if __name__ == "__main__":
    main()
