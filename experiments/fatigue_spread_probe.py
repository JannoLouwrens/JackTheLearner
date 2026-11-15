"""J6(d) step 2f diagnosis: where does NE.01's fatigue spread come from, and why is it ~.09 under dB?
Usage: python -m experiments.fatigue_spread_probe ARM SEED LIFE DEC_CAP  (nice 19)
Replays ONE NE.01 random life under the race's own arm map (drink_race_kcpu._apply) (byte-same rng seeds as ne_01._random_lives),
logging e, f, T, gear_scale and mean mechanical power per decision. Then runs the same arm's drop probe
(ne_01._drop_probe, unchanged). Prints each trajectory's p90-p10 of f (the NE.01 spread definition),
plus e and gear, and the p90-p10 of P/p_max (= f's steady state, needs.py:549).
No ledger, no threshold. Output /data/tmp/j6d/fatigue_<arm>_s<seed>_l<life>.json."""
import os, sys, json
REPO = "/home/opc/jackthelearner"
os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np
from experiments import drink_race_kcpu as KC   # the race's own arm map (shipped = heat_race)


def _p9010(x):
    x = np.asarray(x, float)
    return float(np.percentile(x, 90) - np.percentile(x, 10)) if len(x) >= 10 else 0.0


def main(arm, seed, life, dcap):
    _, mods = KC._apply(arm)
    ne = mods["NE.01"]
    cap = min(ne._windows()["life_cap"], dcap)
    p, model, data, water, pool = ne._build(seed)
    layer = ne._layer(model, pool, seed * 131 + life)
    null = ne._DisabledNeed() if life == 0 else None
    rng = np.random.RandomState(seed * 9973 + life * 17)
    rows = []
    k = 0
    while not layer.dead and k < cap:
        g = layer.gear_scale()
        ctrl = rng.uniform(-ne.CTRL_SCALE, ne.CTRL_SCALE, model.nu) * g
        ne._step_decision(model, data, water, layer, ctrl, null=null)
        s = layer.state
        rows.append([s.e, s.f, s.T, g, layer.last_power_w / layer.p_max, s.w])
        k += 1
    r = np.array(rows)
    rnd = {"k": k, "dead": int(layer.dead),
           "cause": (layer.death_record or {}).get("cause", "none"),
           "spread_e": _p9010(1 - r[:, 0]), "spread_f": _p9010(r[:, 1]),
           "spread_gear": _p9010(r[:, 3]), "spread_Pfrac": _p9010(r[:, 4]),
           "f_p10_p50_p90": [float(np.percentile(r[:, 1], q)) for q in (10, 50, 90)],
           "Pfrac_p10_p50_p90": [float(np.percentile(r[:, 4], q)) for q in (10, 50, 90)],
           "T_min_max": [float(r[:, 2].min()), float(r[:, 2].max())],
           "corr_f_e": float(np.corrcoef(r[:, 1], r[:, 0])[0, 1]),
           "corr_f_gear": float(np.corrcoef(r[:, 1], r[:, 3])[0, 1]),
           "every_500": [[i] + [round(float(v), 3) for v in r[i]] for i in range(0, k, 500)]}
    print("RANDOM", json.dumps({k_: v for k_, v in rnd.items() if k_ != "every_500"}), flush=True)
    drop = ne._drop_probe(seed)
    tr = drop["traj"]
    fi = ne.NEED_KEYS.index("f")
    ei = ne.NEED_KEYS.index("e")
    dr = {"n": int(len(tr)), "died": drop["died"], "cause": drop["cause"],
          "spread_f": _p9010(tr[:, fi]), "spread_e": _p9010(tr[:, ei]),
          "f_p10_p50_p90": [float(np.percentile(tr[:, fi], q)) for q in (10, 50, 90)]}
    print("DROP", json.dumps(dr), flush=True)
    os.makedirs("/data/tmp/j6d", exist_ok=True)
    out = {"arm": arm, "seed": seed, "life": life, "random": rnd, "drop": dr}
    open(f"/data/tmp/j6d/fatigue_{arm}_s{seed}_l{life}.json", "w").write(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
