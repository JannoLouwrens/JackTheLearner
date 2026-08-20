"""LC.03 second-nonstationarity probe — is the constant-std residual a food
ratchet in the persistent world, or eating shot noise on a 14-life ruler?

CONTEXT (the twin check's fix-fails branch, 2026-08-20): with the exploration
schedule flattened to (0.3, 0.3) the frozen dreamer-xs twin STILL reads
life_gain +17.9 s against a pre-registered |gain| <= 10 s bar
(/data/lc03_twin_check.json). The gain is carried by lives 11-12 (189/196 s);
at e0=0.3 a basal-only life is exactly e0/BASAL_B = 180 s, so those lives ATE.
Food is the only energy source, food geoms are FREE BODIES the world never
re-places (drives.new_body resets timers' owner state, never positions), and
the apple's nu=0.50 is worth +300 s of basal life. Two hypotheses, different
repairs:

  RATCHET     eating opportunity trends with RUN position — e.g. the apple
              leaves its platform once and is ground-reachable forever after,
              or floor food migrates under the rover's traffic. A second
              nonstationarity in the RIG; repair at the rig level.
  SHOT NOISE  eating is flat across the run and the elevated lives are simply
              the lives that ate (one floor food ~ +48 s basal-equivalent on
              ONE life ~ +10 s on a 4-life third-mean). Then the rig is fine
              and the RULER cannot resolve a 10 s bar at 14 lives; repair the
              ruler (food-corrected spans and/or more lives), not the rig.

METHOD: replay the twin check's EXACT constant arm (seed 90, dreamer-xs,
12k decisions, e0=0.3, explore_std=(0.3, 0.3), train=False) with a
`reward_fn` probe that logs and returns r unchanged — no RNG is consumed and
no reward is altered, so the trajectory is bit-identical to the checked run.
Logged per decision: life index, energy, apple height, cumulative eat counts.

READINGS (stated before launch):
  - eats_by_life: eat events attributed to the life they happened in.
  - apple_z_min/max per run third: did the apple ever leave the platform?
  - corrected life_gain: span - (energy eaten that life) / BASAL_B, the
    basal-equivalent lifespan with food backed out. corrected gain within
    the +/-10 s bar while raw gain is outside it => food explains the whole
    residual; WHICH hypothesis it is then reads off eats_by_life's trend.
VERIFICATION: the replay's life_spans must equal the twin check's constant
arm exactly, or the probe measured a different run and every reading is void.
"""
import json
import os
import sys
import time

os.chdir("/home/opc/jackthelearner")
sys.path.insert(0, "/home/opc/jackthelearner")

from experiments.protocol import borrow_metrics          # noqa: E402
from experiments.survival import run_survival            # noqa: E402
from experiments import drives                           # noqa: E402

SEED = 90
N_DEC = 12_000
E0 = 0.3
ARM = "dreamer-xs"
STD = (0.3, 0.3)
OUT = "/data/lc03_food_probe.json"
CHECK = "/data/lc03_twin_check.json"

log_rows = []          # (k, life, e, apple_z, ate_apple, ate_obj0, ate_obj1)
_k = [0]
_apple_bid = [None]    # resolved once; -1 if a mutated world dropped the apple


def _probe(r, w, obs, core):
    if _apple_bid[0] is None:
        try:
            _apple_bid[0] = int(w.model.body("apple").id)
        except (KeyError, ValueError):
            _apple_bid[0] = -1
    az = (float(w.data.xpos[_apple_bid[0]][2])
          if _apple_bid[0] >= 0 else float("nan"))
    t = w.drives.ate_total
    log_rows.append((int(_k[0]), int(w.life),
                     round(float(w.drives.state.e), 4), round(az, 3),
                     int(t.get("apple", 0)), int(t.get("obj0", 0)),
                     int(t.get("obj1", 0))))
    _k[0] += 1
    return r


def main() -> None:
    b = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    assert b.ok, f"PS.01 borrow refused: {b.refusal}"
    j0, alpha = b.values["j0_ms"], b.values["alpha"]
    print(f"borrowed j0={j0} alpha={alpha}", flush=True)

    t0 = time.time()
    r = run_survival(SEED, j0=j0, alpha=alpha, e0=E0, n_decisions=N_DEC,
                     policy="core", arm=ARM, train=False, explore_std=STD,
                     reward_fn=_probe)
    print(f"replay done in {time.time() - t0:.0f}s: "
          f"life_gain {r['life_gain']:+.2f} n_lives {r['n_lives']:.0f}",
          flush=True)

    with open(CHECK) as f:
        ref = json.load(f)["constant"]["life_spans"]
    spans = r["life_spans"]
    identical = (len(spans) == len(ref)
                 and all(abs(a - b) < 1e-6 for a, b in zip(spans, ref)))

    # eats and energy-eaten attributed per life
    n_lives = len(spans)
    eats_by_life = [[0, 0, 0] for _ in range(n_lives + 1)]
    energy_by_life = [0.0] * (n_lives + 1)
    nu = {"apple": drives.NU_APPLE, "obj0": drives.NU_FLOORFOOD,
          "obj1": drives.NU_FLOORFOOD}
    prev = (0, 0, 0)
    for (k, life, e, az, na, n0, n1) in log_rows:
        if life <= n_lives:
            d = (na - prev[0], n0 - prev[1], n1 - prev[2])
            for j, name in enumerate(("apple", "obj0", "obj1")):
                eats_by_life[life][j] += d[j]
                energy_by_life[life] += d[j] * nu[name]
        prev = (na, n0, n1)

    corrected = [s - energy_by_life[i] / drives.BASAL_B * 1.0
                 for i, s in enumerate(spans)]
    third = n_lives // 3
    corr_gain = (sum(corrected[-third:]) / third
                 - sum(corrected[:third]) / third) if third else 0.0

    thirds = [log_rows[:len(log_rows) // 3],
              log_rows[len(log_rows) // 3: 2 * len(log_rows) // 3],
              log_rows[2 * len(log_rows) // 3:]]
    apple_z = [{"min": min(x[3] for x in t), "max": max(x[3] for x in t)}
               for t in thirds if t]

    out = {
        "replay_identical_to_check": bool(identical),
        "life_spans": spans,
        "raw_life_gain": r["life_gain"],
        "corrected_life_gain": corr_gain,
        "corrected_spans": [round(c, 1) for c in corrected],
        "eats_by_life": eats_by_life[:n_lives],
        "energy_eaten_by_life": [round(e, 3) for e in energy_by_life[:n_lives]],
        "apple_z_by_run_third": apple_z,
        "ate_total_final": {kk: int(v) for kk, v in
                            (("apple", log_rows[-1][4]),
                             ("obj0", log_rows[-1][5]),
                             ("obj1", log_rows[-1][6]))},
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("identical_replay:", identical, flush=True)
    print("raw gain %+.2f  corrected gain %+.2f" % (r["life_gain"], corr_gain),
          flush=True)
    print("eats_by_life:", eats_by_life[:n_lives], flush=True)
    print("apple_z thirds:", apple_z, flush=True)


if __name__ == "__main__":
    main()
