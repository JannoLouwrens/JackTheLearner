"""RL² bandit row, task check (PUBLISHED_PRIORS §6 "state across episodes";
Duan et al. 2016, arXiv:1611.02779v2, Table 1, row n = 100, k = 5).

Published row (total reward over 100 pulls, mean of 1,000 instances, each arm
p_i ~ U[0, 1], Bernoulli rewards): Random 49.9, Gittins 78.3, TS 74.7,
OTS 77.9, UCB1 78.0, ε-Greedy 75.4, Greedy 74.8, RL² 78.7.

WHAT THIS IS. Before any learner is raced on this row, the TASK we build must
land the paper's hyperparameter-free baselines. Learner-free; nothing in Jack
is touched; it says the instrument is the paper's instrument, not that
anything learns. RL² itself is NOT run: its recipe (TRPO, 250,000-step
batches, ≤ 1,000 iterations, GRU 256) floors at ≥ 23 CPU-h a seed on this box
(`experiments/rl2_cost_probe.py`), > 60 CPU-min → Kaggle or it waits
(ruling 32).

Arms (no tunable hyperparameter, so no grid search to repeat):
- random  — uniform arm each pull.
- greedy  — paper: "the same initialization as UCB1": Beta(1,1) pseudo-counts
            (one success, one failure per arm); pull argmax of the posterior
            mean (s+1)/(n+2); ties broken uniformly at random.
- ts      — Thompson sampling, Beta(1,1) prior (the true U[0,1] distribution,
            as the paper says it provides), one posterior draw per arm.
Not here: Gittins (paper used the best of several approximations, unnamed per
setup), OTS / UCB1 / ε-greedy (each grid-searched; grid not published).

RULE (committed before the first instance runs; ruling 33):
- Instances: 1,000 (the paper's count), instance i uses
  numpy.random.default_rng(10_000 + i) for both the arm means and the pulls;
  every arm sees the same arm means on instance i.
- Per arm, ours = mean total reward, SE = sample SD / √1000. The paper's SE is
  taken as equal to ours (same n, same task). REPRODUCES iff
  |ours − published| ≤ 3 · SE · √2. MISSES otherwise. No extension clause.
- Control that must fail: `ts-blind` — Thompson sampling whose posterior is
  updated with a reward drawn from an independent coin (p = 0.5), not the
  pulled arm's reward. It must land outside TS's band (VOID otherwise: the
  band could not tell a learning bandit algorithm from a blind one).
- The row reports all three arms; the task check PASSES iff all three
  reproduce and the control fails.

Run: `$PY -m experiments.rl2_bandits` (seconds, box, nice 19). Writes
/data/jack-logs/rl2_bandits/baselines.json.
"""
import json, math, os, sys
import numpy as np

K, N, INST, SEED0 = 5, 100, 1000, 10_000
PUBLISHED = {"random": 49.9, "greedy": 74.8, "ts": 74.7}


def play(rule, rng, p):
    s = np.zeros(K); f = np.zeros(K); total = 0
    for _ in range(N):
        if rule == "random":
            a = int(rng.integers(K))
        elif rule == "greedy":
            m = (s + 1) / (s + f + 2)
            a = int(rng.choice(np.flatnonzero(m == m.max())))
        else:  # ts, ts-blind
            a = int(np.argmax(rng.beta(s + 1, f + 1)))
        r = int(rng.random() < p[a])
        total += r
        upd = r if rule != "ts-blind" else int(rng.random() < 0.5)
        s[a] += upd; f[a] += 1 - upd
    return total


def main():
    out = {}
    for rule in ("random", "greedy", "ts", "ts-blind"):
        tot = []
        for i in range(INST):
            rng = np.random.default_rng(SEED0 + i)
            p = rng.random(K)
            tot.append(play(rule, rng, p))
        tot = np.array(tot, float)
        out[rule] = {"mean": tot.mean(), "sd": tot.std(ddof=1), "se": tot.std(ddof=1) / math.sqrt(INST)}
    rows = {}
    for rule, pub in PUBLISHED.items():
        o = out[rule]; tol = 3 * o["se"] * math.sqrt(2)
        rows[rule] = dict(o, published=pub, tol=tol, verdict="REPRODUCES" if abs(o["mean"] - pub) <= tol else "MISSES")
    ts_tol = rows["ts"]["tol"]
    ctrl = out["ts-blind"]; ctrl_fails = abs(ctrl["mean"] - PUBLISHED["ts"]) > ts_tol
    rows["ts-blind"] = dict(ctrl, verdict="holds (outside TS band)" if ctrl_fails else "BREACHED -> VOID")
    passed = ctrl_fails and all(rows[r]["verdict"] == "REPRODUCES" for r in PUBLISHED)
    res = {"rows": rows, "task_check": ("PASS" if passed else ("VOID" if not ctrl_fails else "MISS"))}
    for r, v in rows.items():
        print(f"{r:9s} ours {v['mean']:.2f} (SD {v['sd']:.1f}, SE {v['se']:.2f})"
              + (f" pub {v['published']} tol ±{v['tol']:.2f}" if "published" in v else "") + f" -> {v['verdict']}")
    print("task check:", res["task_check"])
    d = "/data/jack-logs/rl2_bandits"; os.makedirs(d, exist_ok=True)
    json.dump(res, open(f"{d}/baselines.json", "w"), indent=1)


if __name__ == "__main__":
    main()
