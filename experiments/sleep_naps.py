"""sleep_naps.py — LAB 11-18. Seat: Consolidation (held BY ANALYSIS). PLAN part
"Memory and sleep", right-hand column: "whether concentrated sleep beats the
same learning spread evenly"; PRIORS §6 "also open: sleep tied to his sleep
NEED". A PILOT on a standard task: no ledger row, no seat moves.

THE BET. A creature that learns only while asleep acts all day on yesterday's
brain. A newborn's days are short and lengthen (infants nap many times a day,
adults sleep once). At the SAME number of sleeps and the SAME number of
learning steps, sleeps that come early and often and then thin out ("naps")
beat equal-length days — and the same naps in reverse order must lose.
Why it matters for him: asleep his body is idle, so the learning is free
(END-FIT "live AND learn in real time on a person's own computer"); the
question is what acting-without-learning costs, and whether WHEN he sleeps
buys it back.

PRIOR (PRIORS §6, §5). Learning only between stretches of acting is on record
at short stretches: TD3's and DDPG's original code trained once per episode
(1,000 steps of 1M); SPEQ (one preprint) 75,000 critic updates per 10,000
steps beat the same spread evenly "only with dropout". Both are stretches of
<= 1% of the run. Nothing published for a learner that gets only a handful of
sleeps inside its whole budget, nor for the ORDER of the sleeps. §6 prices the
real test in his world at 160 kernel-hours and predicts "little difference".
This pilot is the cheap first look: brain B on the row it reproduced.

=============================================================================
THE RULE — committed before any seed runs.
=============================================================================
Learner   brain B exactly as `brain_b.make_row1("ours")`: SBX SAC, Zoo
          Pendulum-v1 entry (lr 1e-3), layer_norm, everything else default.
          It reproduced this row 11-16 (row 1b: -302.0 +- 22.2 over 15 seeds,
          band [-314.7, -282.1]).
Task      Pendulum-v1, 20,000 steps = 100 training episodes.
Sleep     arms other than `spread` run with gradient_steps = 0 (no learning
          awake). At each sleep time the learner does as many gradient steps
          as decisions lived since the last sleep (one per decision, the
          library's own ratio), from the replay as it then stands. No sleep
          at step 20,000 (it could change nothing measured).
Arms      spread     the reproduced row itself: one update per step, native.
          ep200      sleep every 200 steps (each episode).  99 sleeps.
          days1000   every 1,000.                           19 sleeps.
          days2500   every 2,500.                            7 sleeps.
          days5000   every 5,000.                            3 sleeps.
          naps7      7 sleeps at 200 x 2.107^i: 200, 421, 888, 1871, 3942,
                     8306, 17500 — days2500's sleep count, its 17,500
                     updates and its last sleep, only the timing differs.
          reversed7  CONTROL THAT MUST LOSE: naps7's seven waking stretches
                     in reverse order (9194, 13558, 15629, 16612, 17079,
                     17300, 17500). Same count, same updates, same last sleep.
          sleepless  CONTROL THAT MUST FAIL: never sleeps, never learns.
Seeds     20..24 (fresh: rows 1 and 1b used 0..19), every arm, paired.
Metric    PRIMARY per seed: mean return of all 100 training episodes (the
          known-answer row's own metric — it charges him for every day lived
          on an old brain). SECONDARY, reported only: mean of the last 25
          episodes (what he ends up able to do).
Margin    40 return points. Brain B's seed sd on this row is 22.2, so the
          difference of two 5-seed means has SE ~14; 40 is ~2.9 SE.
VOID      spread's mean < -335 (the apparatus no longer reproduces the row:
          -302 minus 3.3 SE of a 5-seed mean), or sleepless's mean >= -900
          (never learning must fail; row 1's control read -1227), or any
          cell missing.
Record's  ep200 within 40 of spread (per-episode training is the published
check     TD3/DDPG practice). A miss is reported as disagreeing with the
          record; it voids nothing.
Reading   PLAN's question, per days-K arm: "COSTS" iff spread - daysK >= 40
          AND spread ahead on >= 4 of 5 seeds; otherwise "no measurable cost".
The bet   naps7 is ALIVE iff  naps7 - days2500 >= 40 AND ahead on >= 4/5
          seeds,  AND  naps7 - reversed7 >= 40 AND ahead on >= 4/5 seeds.
          If the first holds and the second does not: VOID (the control did
          not lose, so order is not what was measured). Otherwise DEAD.
Unequal by nature, said here: what he lives after his last sleep is never
learned from, so total updates are spread 19,900; ep200 19,800; days1000
19,000; days2500 = naps7 = reversed7 17,500; days5000 15,000.
What this cannot say: anything about his world (a PASS carries its world);
about brain A; about more than one update per decision in sleep.

    /data/venvs/jack-std/bin/python -m experiments.sleep_naps cell ARM SEED OUT.json [STEPS]
    $PY -m experiments.sleep_naps kernel          # one Kaggle CPU session, < 1 h
    $PY -m experiments.sleep_naps score FILE.json
    $PY -m experiments.sleep_naps selftest        # schedules + planted verdicts
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = Path("/data/jack-logs/sleep_naps")
STEPS = 20_000
SEEDS = (20, 21, 22, 23, 24)
MARGIN = 40.0
VOID_SPREAD_BELOW = -335.0
VOID_SLEEPLESS_ABOVE = -900.0
NAPS7 = (200, 421, 888, 1871, 3942, 8306, 17500)
ARMS = ("spread", "ep200", "days1000", "days2500", "days5000", "naps7",
        "reversed7", "sleepless")
KERNEL_PINS = ["stable-baselines3==2.9.2", "sbx-rl==0.28.0", "gymnasium==1.4.0"]


def schedule(arm: str, steps: int = STEPS) -> list:
    """Sleep times (decision counts) for a sleeping arm; [] for sleepless.
    `spread` has no schedule (native one update per step)."""
    if arm == "sleepless":
        return []
    if arm == "naps7":
        return list(NAPS7)
    if arm == "reversed7":
        gaps = [b - a for a, b in zip((0,) + NAPS7[:-1], NAPS7)][::-1]
        out, t = [], 0
        for g in gaps:
            t += g
            out.append(t)
        return out
    k = {"ep200": 200, "days1000": 1000, "days2500": 2500, "days5000": 5000}[arm]
    return list(range(k, steps, k))


def run_cell(arm: str, seed: int, steps: int = STEPS) -> dict:
    from experiments import brain_b
    from stable_baselines3.common.callbacks import BaseCallback
    t0 = time.time()
    model = brain_b.make_row1("ours", seed, steps)
    sleeps = []
    if arm == "spread":
        model.learn(total_timesteps=steps)
    else:
        model.gradient_steps = 0          # awake: act, store, never learn
        sched = schedule(arm, steps) if steps == STEPS else [
            t for t in schedule(arm, STEPS) if t < steps]
        due = {t: t - p for p, t in zip([0] + sched[:-1], sched)}

        class Sleep(BaseCallback):
            def _on_step(self) -> bool:
                n = due.get(self.num_timesteps)
                if n:
                    self.model.train(gradient_steps=n, batch_size=self.model.batch_size)
                    sleeps.append([int(self.num_timesteps), int(n)])
                return True

        model.learn(total_timesteps=steps, callback=Sleep())
    env = model.get_env().envs[0]
    rets = [float(r) for r in env.get_episode_rewards()]
    return {"arm": arm, "seed": seed, "steps": steps, "episodes": len(rets),
            "mean_return": sum(rets) / max(1, len(rets)),
            "last25": sum(rets[-25:]) / max(1, len(rets[-25:])),
            "returns": rets, "sleeps": sleeps, "n_updates": int(model._n_updates),
            "wall_s": round(time.time() - t0, 1), "versions": brain_b.versions()}


def cells() -> list:
    # sleepless first (seconds each), then the arms the rule's clauses need
    order = ("sleepless", "spread", "days2500", "naps7", "reversed7", "ep200",
             "days1000", "days5000")
    return [(a, s) for a in order for s in SEEDS]


def run_parallel(todo: list, outdir, workers: int = 4) -> list:
    import subprocess
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pending, running, done = list(todo), [], []
    while pending or running:
        while pending and len(running) < workers:
            arm, seed = pending.pop(0)
            f = outdir / f"sleep_{arm}_s{seed}.json"
            p = subprocess.Popen([sys.executable, "-m", "experiments.sleep_naps", "cell",
                                  arm, str(seed), str(f)], cwd=REPO,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            running.append((arm, seed, f, p))
        time.sleep(3)
        for item in list(running):
            arm, seed, f, p = item
            if p.poll() is None:
                continue
            err = p.stderr.read().decode()[-800:]
            running.remove(item)
            rec = (json.loads(f.read_text()) if f.exists()
                   else {"arm": arm, "seed": seed, "error": f"rc={p.returncode}: {err}"})
            done.append(rec)
            print(f"{arm} s{seed}: {rec.get('mean_return', rec.get('error'))}", flush=True)
    return done


def _pair(by, a, b, key="mean_return"):
    """mean(a) - mean(b) and on how many seeds a is ahead."""
    d = [by[a][s][key] - by[b][s][key] for s in SEEDS]
    return sum(d) / len(d), sum(1 for x in d if x > 0)


def verdict(recs: list) -> dict:
    by = {a: {} for a in ARMS}
    for c in recs:
        if "error" not in c and c.get("arm") in by and c.get("steps") == STEPS:
            by[c["arm"]][c["seed"]] = c
    missing = [(a, s) for a in ARMS for s in SEEDS if s not in by[a]]
    out = {"missing": missing, "arms": {}}
    if missing:
        out["verdict"] = "VOID (cells missing)"
        return out
    for a in ARMS:
        m = [by[a][s]["mean_return"] for s in SEEDS]
        l = [by[a][s]["last25"] for s in SEEDS]
        mu = sum(m) / len(m)
        sd = (sum((x - mu) ** 2 for x in m) / (len(m) - 1)) ** 0.5
        out["arms"][a] = {"mean": round(mu, 1), "sd": round(sd, 1),
                          "per_seed": [round(x, 1) for x in m],
                          "last25": round(sum(l) / len(l), 1),
                          "sleeps": len(by[a][SEEDS[0]]["sleeps"]),
                          "updates": by[a][SEEDS[0]]["n_updates"]}
    A = out["arms"]
    void = []
    if A["spread"]["mean"] < VOID_SPREAD_BELOW:
        void.append(f"spread {A['spread']['mean']} < {VOID_SPREAD_BELOW}")
    if A["sleepless"]["mean"] >= VOID_SLEEPLESS_ABOVE:
        void.append(f"sleepless {A['sleepless']['mean']} >= {VOID_SLEEPLESS_ABOVE}")
    out["void"] = void
    d, _ = _pair(by, "spread", "ep200")
    out["record_check_ep200"] = {"spread_minus_ep200": round(d, 1),
                                 "agrees": abs(d) < MARGIN}
    out["cost_of_days"] = {}
    for a in ("ep200", "days1000", "days2500", "days5000", "naps7", "reversed7"):
        d, n = _pair(by, "spread", a)
        out["cost_of_days"][a] = {"spread_minus": round(d, 1), "spread_ahead": n,
                                  "reading": "COSTS" if d >= MARGIN and n >= 4
                                  else "no measurable cost"}
    d1, n1 = _pair(by, "naps7", "days2500")
    d2, n2 = _pair(by, "naps7", "reversed7")
    c1, c2 = d1 >= MARGIN and n1 >= 4, d2 >= MARGIN and n2 >= 4
    out["bet"] = {"naps7_minus_days2500": round(d1, 1), "ahead": n1,
                  "naps7_minus_reversed7": round(d2, 1), "ahead_rev": n2}
    out["verdict"] = ("VOID: " + "; ".join(void) if void else
                      "ALIVE" if c1 and c2 else
                      "VOID (naps beat days but the reversed control did not lose)" if c1
                      else "DEAD")
    return out


_JOB = r'''
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q"] + __PINS__, check=True)
import json, os
os.chdir("/tmp/jack")
from experiments.sleep_naps import run_parallel, verdict, cells
r = run_parallel(cells(), os.environ["JACK_OUT"])
json.dump({"cells": r, "verdict": verdict(r)},
          open(os.path.join(os.environ["JACK_OUT"], "sleep_naps.json"), "w"))
print("DONE", flush=True)
'''


def kernel() -> dict:
    from experiments.gpu import build_job, result_json, submit
    body = _JOB.replace("__PINS__", repr(KERNEL_PINS))
    sub = submit(build_job(body), prefer="kaggle-cpu", est_hours=1.0,
                 timeout_s=int(2.5 * 3600), fetch=["sleep_naps.json"])
    if not sub.ok:
        raise RuntimeError(f"sleep_naps kernel: {sub.message}")
    r = result_json(sub, "sleep_naps.json")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pilot.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r.get("verdict"), indent=1), flush=True)
    return r


def selftest() -> int:
    """Schedules are what the rule says; the verdict fires each way on planted
    cells (a checker never seen firing is untested)."""
    assert schedule("reversed7") == [9194, 13558, 15629, 16612, 17079, 17300, 17500]
    assert schedule("days2500") == [2500 * i for i in range(1, 8)]
    for a in ("naps7", "reversed7", "days2500"):
        s = schedule(a)
        assert len(s) == 7 and s[-1] == 17500, (a, s)
    assert len(schedule("ep200")) == 99 and len(schedule("days1000")) == 19
    assert schedule("days5000") == [5000, 10000, 15000] and schedule("sleepless") == []

    def plant(means):
        return [{"arm": a, "seed": s, "steps": STEPS, "mean_return": means[a] + (s - 22),
                 "last25": -150.0, "sleeps": [[1, 1]] * len(schedule(a) if a != "spread" else []),
                 "n_updates": 0} for a in ARMS for s in SEEDS]
    base = dict(spread=-300, ep200=-305, days1000=-330, days2500=-420, days5000=-600,
                naps7=-340, reversed7=-800, sleepless=-1230)
    assert verdict(plant(base))["verdict"] == "ALIVE"
    assert verdict(plant({**base, "naps7": -400}))["verdict"] == "DEAD"
    assert verdict(plant({**base, "reversed7": -350}))["verdict"].startswith("VOID (naps")
    assert verdict(plant({**base, "sleepless": -400}))["verdict"].startswith("VOID: sleepless")
    assert verdict(plant({**base, "spread": -500}))["verdict"].startswith("VOID: spread")
    assert verdict(plant(base)[1:])["verdict"] == "VOID (cells missing)"
    v = verdict(plant(base))
    assert v["cost_of_days"]["days2500"]["reading"] == "COSTS"
    assert v["cost_of_days"]["ep200"]["reading"] == "no measurable cost"
    assert v["record_check_ep200"]["agrees"]
    print("sleep_naps selftest PASS")
    return 0


def main(argv):
    if len(argv) >= 5 and argv[1] == "cell":
        steps = int(argv[5]) if len(argv) > 5 else STEPS
        Path(argv[4]).write_text(json.dumps(run_cell(argv[2], int(argv[3]), steps)))
        return 0
    if len(argv) >= 2 and argv[1] == "kernel":
        kernel()
        return 0
    if len(argv) >= 3 and argv[1] == "score":
        r = json.loads(Path(argv[2]).read_text())
        print(json.dumps(verdict(r["cells"] if isinstance(r, dict) else r), indent=1))
        return 0
    if len(argv) >= 2 and argv[1] == "selftest":
        return selftest()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
