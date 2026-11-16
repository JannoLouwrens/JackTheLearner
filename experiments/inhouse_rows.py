"""inhouse_rows.py — the in-house arms on known-answer rows 1 and 3 (ORDERS J3 (4)).

The owner's fair chance (11-15 19:31): every in-house learner meets the same
published numbers Brain B met, through ONE thin adapter, before any of them is
raced again in his world (PLAN stage 1; ruling 24).

WHAT RUNS. The arm's own core (`cores.build_arm(name, with_placebo=False)`)
and the seated update (`cores.lc_update` — tanh-squashed mean, AWR weights,
POLICY_SEAT "tanh-awr"), fed by the seated target machinery reused from
`survival.py` (`_TargetRing`, `_gae`, `_updates_due`, GAMMA, SEG, EXPLORE_STD
decayed linearly over the run). Nothing here re-implements an update; only the
outer loop is new, because `survival.run_survival` builds W0 internally and
editing it would stale every LC/XL certificate. The loop below mirrors it line
for line where the semantics exist in a gymnasium task (see `run_cell`).
`ppo-needs` under today's update IS the seated tanh-AWR learner (plain
PPOCore + lc_update); the other cores add their binding terms.

THE ADAPTER (all of it):
  obs     the task's observation vector goes into the `proprio` slot (12
          wide; zero-padded: Pendulum 3, MountainCar 2). Every other modality
          is DROPPED, i.e. replaced by the core's learned `missing` embedding —
          the path W0 already uses for `language`.
  action  the core emits 8 dims in [-1, 1]; the task reads dim 0 times the
          task's bound (Pendulum 2.0, MountainCar 1.0). Dims 1-7 are explored
          and regressed like the rest; they never touch the task.
  reward  the task's own reward, raw (no scaling: the seated GAE normalises
          advantages per segment, value targets stay in task units).
  ends    `terminated` closes the segment terminal=True (boot 0); a time-limit
          `truncated` closes it terminal=False with boot = V(s_t) — the seated
          `_close_segment(boot=v)` convention, a known quirk kept, not fixed.

OURS BESIDE THEIRS (every setting that differs from the published recipe):
  learner    published SAC (rows 1, 3); ours = the arm's core + tanh-AWR.
  gamma      0.95 seated (published: row 1 SAC 0.99, row 3 0.9999). Kept: the
             arm is tested as seated. A miss on row 3's sparse 999-step task
             partly reads this horizon (0.95 ~ 20 steps); said in advance.
  replay     256-transition ring of GAE-labelled rows (LC.02's timed shape)
             vs published 1e6 / 5e4.
  batch      16 vs 256 / 512.  lr 3e-4 vs 1e-3 / 3e-4.  updates 0.25 per
             decision (LC.03's train_ratio) vs 1 / 1.
  explore    Gaussian std 0.5 -> 0.1 linearly over the run vs SAC entropy
             (row 3: gSDE). Deterministic eval = tanh(actor) mean.
  stack      project venv (torch 2.8.0+cpu, gymnasium 1.1.1) — the venv the
             cores are certified in; §5 pins gymnasium 1.4.0. Pendulum-v1 and
             MountainCarContinuous-v0 dynamics are unchanged between them.

=============================================================================
THE RULE — committed before any run (11-16).
=============================================================================
Arms     every in-house arm by its `cores.build_arm` name; this file's first
         registered runs are `ppo-needs` (the seated tanh-AWR learner) and
         `wm-latent` (the seated Learning-core holder). The others
         (`ppo-lp`, `dreamer-xs`, `wm-efe`; `wm-dyn`, `wm-latent+ring` once
         their cores are wired through this adapter) run under this same
         rule unchanged.
Control  per arm and row, the SAME core and loop with the optimiser never
         stepped (`train=False`, survival's frozen twin), same seeds.
Seeds    0-4 (arm and control).
Row 1    Pendulum-v1, budget 60,000 steps (3x published 20,000). Metric: the
         mean TRAINING return of each aligned 100-episode window (steps
         0-20k, 20k-40k, 40k-60k; §5 row 1 is the mean training return over
         20k steps = 100 episodes). A seed REACHES iff any window mean
         >= -314.7 (the band's lower edge; better is not a miss).
Row 3    MountainCarContinuous-v0 (sparse), budget 150,000 steps (3x 50,000).
         Metric: mean return of 10 deterministic episodes (episode e reset
         with seed 10_000 + seed + 1_000 e), at 50k, 100k and 150k. A seed
         REACHES iff any checkpoint > 90.
Verdict  per arm and row: PASS iff >= 3 of 5 arm seeds reach AND the control
         reaches on <= 1 of 5 (stated on the control's distribution — row
         4c's lesson: a per-seed "every control fails" clause is brittle on a
         task a lucky init can solve; the actor's last layer is 0.01-scaled,
         so a frozen mean ~0 should not, and a control reaching on 2+ seeds
         makes the row VOID for that arm). Else FAIL. A FAIL is set aside
         with its numbers, not deleted, and may return repaired (ruling 24).
Prior    the record (PUBLISHED_PRIORS §1, §5) predicts small-replay on-policy
         AWR with gamma 0.95 misses row 3 (sparse, long horizon) and likely
         row 1 at this budget. Said before the run; the test can fail.

Usage: $PY -m experiments.inhouse_rows cell ARM ROW SEED [ctrl] [--steps N]
       $PY -m experiments.inhouse_rows run OUTDIR ARM[,ARM] [--workers 2]
       $PY -m experiments.inhouse_rows verdict OUTDIR
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .cores import ACTION_DIM, LC_BATCH, MODALITIES, build_arm, lc_update, make_optimizer
from .survival import EXPLORE_STD, SEG, _gae, _TargetRing, _updates_due

TRAIN_RATIO = 0.25
SEEDS = (0, 1, 2, 3, 4)
ROWS = {
    1: {"env": "Pendulum-v1", "pub_steps": 20_000, "budget": 60_000,
        "scale": 2.0, "reach": -314.7},
    3: {"env": "MountainCarContinuous-v0", "pub_steps": 50_000,
        "budget": 150_000, "scale": 1.0, "reach": 90.0},
}
N_EVAL = 10
KEYS = list(MODALITIES)
DROPPED = tuple(k for k in KEYS if k != "proprio")


def _obs_dict(x: np.ndarray) -> dict:
    out = {k: np.zeros(d, dtype=np.float32) for k, d in MODALITIES.items()}
    out["proprio"][: len(x)] = np.asarray(x, dtype=np.float32)
    return out


def _to_t(o: dict) -> dict:
    return {k: torch.from_numpy(v).unsqueeze(0) for k, v in o.items()}


def _eval(core, env_id: str, scale: float, seed: int) -> float:
    import gymnasium as gym
    env = gym.make(env_id)
    rets = []
    for ep in range(N_EVAL):
        x, _ = env.reset(seed=10_000 + seed + 1_000 * ep)
        done, ret = False, 0.0
        while not done:
            with torch.no_grad():
                a = core.act_deterministic(_to_t(_obs_dict(x)), dropped=DROPPED)
            x, r, term, trunc, _ = env.step(np.array([float(a[0, 0]) * scale]))
            ret += float(r)
            done = term or trunc
        rets.append(ret)
    env.close()
    return float(np.mean(rets))


def run_cell(arm: str, row: int, seed: int, train: bool = True,
             steps: int | None = None) -> dict:
    """One arm-row-seed. Mirrors survival.run_survival's 'core' branch."""
    import gymnasium as gym
    torch.set_num_threads(1)
    R = ROWS[row]
    n = steps or R["budget"]
    t0 = time.process_time()
    rng = np.random.RandomState(seed * 6553 + 11)
    gen = torch.Generator().manual_seed(seed * 7919 + 3)
    torch.manual_seed(9_000 + seed)
    core = build_arm(arm, with_placebo=False)
    core.eval()
    opt = make_optimizer(core) if train else None
    env = gym.make(R["env"])
    x, _ = env.reset(seed=seed)
    obs = _obs_dict(x)
    ring = _TargetRing(obs)
    seg_obs, seg_act, seg_rew, seg_val = [], [], [], []
    ep_ret, ep_rets, ep_end_step = 0.0, [], []
    checkpoints, opt_steps = {}, 0
    ck_at = {R["pub_steps"] * m for m in (1, 2, 3)} if row == 3 else set()

    def close(boot: float, terminal: bool):
        nonlocal seg_obs, seg_act, seg_rew, seg_val
        if not seg_rew:
            return
        adv, tgt = _gae(seg_rew, seg_val, boot, terminal)
        for i in range(len(seg_rew)):
            ring.push(seg_obs[i], seg_act[i], float(tgt[i]), float(adv[i]))
        seg_obs, seg_act, seg_rew, seg_val = [], [], [], []

    for k in range(n):
        ot = _to_t(obs)
        with torch.no_grad():
            s = core.shared_state(ot, dropped=DROPPED)
            mean = core.act(ot, s).squeeze(0).numpy()
            v = float(core.critic(s))
        frac = min(1.0, k / max(1, n - 1))
        std = EXPLORE_STD[0] + (EXPLORE_STD[1] - EXPLORE_STD[0]) * frac
        a = np.clip(mean + std * rng.randn(ACTION_DIM), -1.0, 1.0)
        x2, r, term, trunc, _ = env.step(np.array([a[0] * R["scale"]]))
        ep_ret += float(r)
        seg_obs.append({kk: torch.from_numpy(vv.copy()) for kk, vv in obs.items()})
        seg_act.append(torch.from_numpy(a.astype(np.float32)))
        seg_rew.append(float(r))
        seg_val.append(v)
        if term or trunc or len(seg_rew) >= SEG:
            close(boot=v, terminal=bool(term))
        if train and ring.n >= LC_BATCH:
            for _ in range(_updates_due(k + 1, TRAIN_RATIO)):
                batch, targets = ring.sample(gen)
                lc_update(core, batch, targets, opt, dropped=DROPPED)
                opt_steps += 1
        if term or trunc:
            ep_rets.append(ep_ret)
            ep_end_step.append(k + 1)
            ep_ret = 0.0
            x2, _ = env.reset()
        obs = _obs_dict(x2)
        if (k + 1) in ck_at:
            checkpoints[str(k + 1)] = _eval(core, R["env"], R["scale"], seed)
    env.close()
    out = {"arm": arm, "row": row, "seed": seed, "ctrl": not train, "steps": n,
           "episodes": len(ep_rets), "opt_steps": opt_steps,
           "cpu_s": round(time.process_time() - t0, 1),
           "returns": [round(r, 2) for r in ep_rets]}
    if row == 1:
        w = R["pub_steps"]
        out["windows"] = [round(float(np.mean([r for r, e in zip(ep_rets, ep_end_step)
                                               if i * w < e <= (i + 1) * w])), 2)
                          for i in range(n // w)
                          if any(i * w < e <= (i + 1) * w for e in ep_end_step)]
        out["metric"] = max(out["windows"]) if out["windows"] else None
    else:
        out["checkpoints"] = checkpoints
        if not checkpoints:      # a short smoke: evaluate once at the end
            checkpoints[str(n)] = _eval(core, R["env"], R["scale"], seed)
        out["metric"] = max(checkpoints.values())
    out["reached"] = (out["metric"] is not None and
                      (out["metric"] >= R["reach"] if row == 1 else out["metric"] > R["reach"]))
    return out


def cells(arms: list) -> list:
    out = []
    for arm in arms:
        for row in ROWS:
            for s in SEEDS:
                out += [(arm, row, s, False), (arm, row, s, True)]
    return out


def _name(arm, row, seed, ctrl) -> str:
    return f"{arm}_r{row}_s{seed}{'_ctrl' if ctrl else ''}.json"


def run_all(outdir: str, arms: list, workers: int = 2) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cells(arms) if not (out / _name(*c)).exists()]
    py = sys.executable
    live = []
    while todo or live:
        while todo and len(live) < workers:
            arm, row, seed, ctrl = todo.pop(0)
            cmd = ["nice", "-n", "19", py, "-m", "experiments.inhouse_rows", "cell",
                   arm, str(row), str(seed)] + (["ctrl"] if ctrl else []) + \
                  ["--out", str(out / _name(arm, row, seed, ctrl))]
            print("START", arm, row, seed, "ctrl" if ctrl else "", flush=True)
            live.append(((arm, row, seed, ctrl), subprocess.Popen(
                cmd, env={**os.environ, "OMP_NUM_THREADS": "1"})))
        time.sleep(2)
        for c, p in list(live):
            if p.poll() is not None:
                print("END", *c, "rc", p.returncode, flush=True)
                live.remove((c, p))
    print(json.dumps(verdict(out), indent=1), flush=True)


def verdict(outdir) -> dict:
    by = {}
    for f in Path(outdir).glob("*_r*_s*.json"):
        r = json.loads(f.read_text())
        by.setdefault((r["arm"], r["row"]), {"arm": {}, "ctrl": {}})[
            "ctrl" if r["ctrl"] else "arm"][r["seed"]] = r
    res = {}
    for (arm, row), d in sorted(by.items()):
        a = [d["arm"].get(s) for s in SEEDS]
        c = [d["ctrl"].get(s) for s in SEEDS]
        key = f"{arm} row {row}"
        if None in a or None in c:
            res[key] = {"reading": "INCOMPLETE"}
            continue
        ka = sum(x["reached"] for x in a)
        kc = sum(x["reached"] for x in c)
        res[key] = {"arm_metric": [x["metric"] for x in a], "arm_reached": ka,
                    "ctrl_metric": [x["metric"] for x in c], "ctrl_reached": kc,
                    "reading": "VOID" if kc > 1 else ("PASS" if ka >= 3 else "FAIL")}
    return res


def main(argv: list) -> None:
    if argv[1] == "cell":
        arm, row, seed = argv[2], int(argv[3]), int(argv[4])
        steps = int(argv[argv.index("--steps") + 1]) if "--steps" in argv else None
        r = run_cell(arm, row, seed, train="ctrl" not in argv, steps=steps)
        if "--out" in argv:
            Path(argv[argv.index("--out") + 1]).write_text(json.dumps(r))
        print(json.dumps({k: v for k, v in r.items() if k != "returns"}))
    elif argv[1] == "run":
        w = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 2
        run_all(argv[2], argv[3].split(","), workers=w)
    elif argv[1] == "verdict":
        print(json.dumps(verdict(argv[2]), indent=1))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
