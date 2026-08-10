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

## POST-HOC EXTENSION, added 2026-08-10 AFTER the first run, marked as such

The pre-registered set was `(b, ν_floor, respawn)` with `κ` explicitly
exculpated by §2.3's refutation note — *"the error is not in κ: 293 W producing
3.9× basal is what κ was chosen to do."* **That exculpation was itself derived
from the starving-body measurement**, and the full-strength run refutes it:

    P̄(1) = 1434.8 ± 22.2 W   ->   drain(1) = 15.38 x basal, not 3.9 x

`κ`'s defining sentence in §2.2 is not the number `1.67e-5`; it is a claim about
this body — *"vigorous activity (~200 W) roughly triples b"*. 200 W is a human
premise applied to a MuJoCo Humanoid-v5 whose actuators deliver **7.2×** that
under the same random policy the drain is priced against. So the same class of
defect §2.3 was caught for, in the same measurement: a constant defined by a
premise that is false about the body it constrains (`LESSONS.md` — "assert
contracts against the source of truth, not against another constant").

Both solutions are computed and printed, so the choice is legible:

  · **SOLUTION A — κ frozen, the pre-registered set only.** Satisfies C1–C3 and
    requires `RESPAWN_APPLE_S = 17.1 s`. Arithmetically valid; not a world. An
    apple that returns every 17 s is not a climb-gated resource, and §2.3's
    whole claim is that the ladder is the difference between subsistence and
    activity. Reported, not shipped.
  · **SOLUTION B — κ re-derived from the measured power, restoring §2.2's own
    sentence:** `drain(1) = 3·b` (§2.2: "roughly triples"), hence
    `κ = 2b / P̄(1)`. Nothing about the intent changes; only the wrong power
    estimate it was computed from. C1–C3 are then re-solved with the same
    criterion and the same knob rule. SHIPPED.

Identifying "the random policy at duty 1" with §2.2's "vigorous activity" is a
choice and it is the conservative one: random flailing is the most wasteful
policy this body can execute, so any learned policy costs less than the constant
the world is sized against.

**This does not turn PS.01 green and must not be checked against whether it
does.** `ok_random_survives` and `ok_statue_starves` fail for probe-policy
reasons no supply constant can reach (a random policy cannot forage; the statue
dies at `t = 1/b` = exactly the 600 s horizon), and both are routed to
`INTEGRATION_QUEUE.md` as a spec redesign.

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
VIGOROUS_MULT = 3.0               # §2.2 verbatim: vigorous activity "roughly
                                  # triples b". SOLUTION B re-derives kappa from
                                  # the MEASURED power to honour that sentence.


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
    # linearity of the drain in duty, which drain(D) = b + kappa*P_bar*D assumes
    lin = [(d, pbar[d] / (d * p1)) for d in DUTIES if d > 0]
    print("\n  linearity  P(D) / (D * P(1)):  "
          + "  ".join(f"D={d:.3f}: {r:.3f}" for d, r in lin))

    def solve(k: float) -> dict:
        """C1-C3 at a given kappa. Identical arithmetic for both solutions."""
        drain1 = b + k * p1
        # C2 — the smaller of the two anchors, i.e. the harsher world
        s_pal = PAL * b
        s_alt = b + k * pbar[D_ALT]
        s_f = min(s_pal, s_alt)
        d_star = (s_f - b) / (k * p1)
        s_max_req = drain1 / FORAGE_EFF          # C1
        apple_req = s_max_req - s_f
        return {
            "kappa": k, "drain1": drain1, "s_pal": s_pal, "s_alt": s_alt,
            "s_f": s_f, "anchor": "PAL" if s_pal <= s_alt else f"D_ALT={D_ALT}",
            "d_star": d_star, "s_max_req": s_max_req, "apple_req": apple_req,
            "t_floor": 2.0 * drives.NU_FLOORFOOD / s_f,
            "t_apple": drives.NU_APPLE / apple_req if apple_req > 0 else float("inf"),
            "ok": (0.0 < d_star < 1.0) and (s_f < drain1),
        }

    def report(name: str, s: dict) -> None:
        print(f"\n── {name} ── kappa = {s['kappa']:.4e} /J")
        print(f"  drain(1)            {s['drain1']:.4e} /s   "
              f"({s['drain1'] / b:.2f}x basal, P_bar = {p1:.1f} W full strength)")
        print(f"  C2 anchors: PAL*b = {s['s_pal']:.4e}   "
              f"b+k*P(D_ALT) = {s['s_alt']:.4e}   -> take {s['anchor']} (smaller)")
        print(f"  C2 S_f              {s['s_f']:.4e} /s   ({s['s_f'] / b:.2f}x basal)")
        print(f"  C2 D* funded        {s['d_star']:.4f}          (required 0 < D* < 1)")
        print(f"  C3 S_f < drain(1)   {s['s_f'] < s['drain1']}   "
              f"({s['drain1'] / s['s_f']:.2f}x short of constant activity)")
        print(f"  C1 S_max required   {s['s_max_req']:.4e} /s  "
              f"(drain(1) / {FORAGE_EFF}); apple must supply {s['apple_req']:.4e} /s")
        print(f"  -> RESPAWN_FLOORFOOD_S  {drives.RESPAWN_FLOORFOOD_S:.1f} -> "
              f"{s['t_floor']:.1f} s")
        print(f"  -> RESPAWN_APPLE_S      {drives.RESPAWN_APPLE_S:.1f} -> "
              f"{s['t_apple']:.1f} s")
        print(f"  CRITERION SATISFIABLE: {s['ok']}")

    s_f_old = 2.0 * drives.NU_FLOORFOOD / drives.RESPAWN_FLOORFOOD_S
    s_max_old = s_f_old + drives.NU_APPLE / drives.RESPAWN_APPLE_S
    drain1_shipped = b + kappa * p1
    print(f"\n  SHIPPED TODAY:  S_f {s_f_old:.4e}  S_max {s_max_old:.4e}  "
          f"vs drain(1) {drain1_shipped:.4e}")
    print(f"     C1 today: "
          f"{'HOLDS' if s_max_old * FORAGE_EFF >= drain1_shipped else 'VIOLATED'}"
          f" — a PERFECT forager harvests {s_max_old / drain1_shipped:.2f}x of what "
          f"constant activity costs")

    a = solve(kappa)
    report("SOLUTION A — kappa frozen (the pre-registered set only)", a)

    # SOLUTION B: restore §2.2's own sentence — "vigorous activity roughly
    # triples b" — against the MEASURED power instead of its 200 W premise.
    kappa_b = (VIGOROUS_MULT - 1.0) * b / p1
    bsol = solve(kappa_b)
    report(f"SOLUTION B — kappa re-derived so drain(1) = {VIGOROUS_MULT:.0f}x basal",
           bsol)
    print(f"\n  kappa {kappa:.4e} -> {kappa_b:.4e}  "
          f"({kappa / kappa_b:.2f}x smaller; its 200 W premise is "
          f"{p1 / 200.0:.2f}x wrong for this body)")
    print("\n  SHIPPING SOLUTION B. A is arithmetically valid and is not a world:"
          f" an apple returning every {a['t_apple']:.1f} s is not climb-gated.")
    return {"rows": rows, "p1": p1, "linearity": lin, "A": a, "B": bsol,
            "s_f_old": s_f_old, "s_max_old": s_max_old,
            "drain1_shipped": drain1_shipped}


if __name__ == "__main__":
    main()
