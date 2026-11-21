"""Brain A's known-answer rows (J14): the pinned runner, UNMODIFIED, on §2's first three tasks.

Source: `sheeprl==0.8.5` (PyPI; tag v0.8.5 = 22b4f164), `algo=dreamer_v3_5` with the smallest preset
`algo=dreamer_v3_5_1M`, plus `algo.compile.enabled=True` (J5: 9.8x on a T4, the official code's
`jax.jit` analogue). Code only, never weights: every run starts from zero. Line diff against the
official agent (`danijar/dreamerv3@e01491f`): `experiments/brain_a/README.md` §"Line diff" (11-16).

## THE RULE — committed 2026-11-17 before any seed runs (ruling 33's form)

Rows and bands, exactly as PUBLISHED_PRIORS §2 states them (official v1 size-S, ratio 512, repeat 2,
4 envs, 5 seeds; "pass = our mean within ~2 SD at each point", made exact here as
|our mean - published mean| <= 2 x published SD):

| row | task (DMC proprio)  | point (env steps) | published mean +- SD | pass band       |
|-----|---------------------|-------------------|----------------------|-----------------|
| A1  | cartpole_balance    | 100K              | 794 +- 52            | 690 .. 898      |
| A1  | cartpole_balance    | 200K              | 966 +- 17            | 932 .. 1000     |
| A2  | cartpole_swingup    | 100K              | 478 +- 20            | 438 .. 518      |
| A2  | cartpole_swingup    | 200K              | 778 +- 19            | 740 .. 816      |
| A3  | ball_in_cup_catch   | 200K              | 949 +- 9             | 931 .. 967      |

- Seeds 0, 1, 2 per row. A row PASSES when its 3-seed mean is inside the band at EVERY point.
- The ONE extension, stated now: if a row misses at 3 seeds, seeds 3 and 4 run (fresh) and that row
  is judged once more at 5 seeds, same bands. Nothing after that: a miss at 5 is a named miss.
- Score at a point P = mean return of the training episodes that END in env steps (0.9 P, P]
  (the official score files are training-episode returns). A seed with no episode in that window
  is a VOID seed, not a zero; a row with a VOID seed is not judged and says so.
- Control, one per row, seed 0: the never-trains twin (`algo.learning_starts` past the run's last
  step: dreamer_v3_5 plays its untrained policy from step 1 — no random prefill — and never takes
  a gradient step).
  CORRECTION 11-17 14:35, before any seed ran (box smoke, A2 control): `learning_starts = total+1`
  still trained ~2 gradient steps at the final iteration (the runner floors it to whole iterations);
  it is now `2 x total` — no gradient step at all, as the line above says. Stronger, never looser:
  the old form's extra steps came after every scored episode, so no reading could have moved.
  It MUST land below the band's lower edge at every point, else the row is VOID (the task is
  too easy to tell a learner from none at that point).
- Reported, not judged: §2's step-0 check (reward and value loss 5.541, dyn = rep = 1.0, actor std
  0.893, predicted reward and value 0) read from the first logged step of each seed.
- Settings that differ from the published protocol (ours beside theirs): agent v2 port (sheeprl
  `dreamer_v3_5`) vs the v1 agent of the 2023 score files; size `1m` (~0.64M params) vs size S —
  §2: "No score exists for size1m"; compile on (numerics, not recipe); everything else as published
  (repeat 2, 4 envs, train ratio 512 = sheeprl `replay_ratio` 0.5, batch 16 x 64).
  A miss can therefore mean a broken build OR that 1m is too small: the miss is written with both
  readings and not explained away.
- Not run on this box (ruling 32): Kaggle GPU from Sat 11-21 00:00 UTC through `gpu.py`
  (prefer="kaggle"), pushed first. Cost on record: ~7 T4-h at 3 seeds (J5, compiled 25.9 steps/s).

## Unit note to verify in the smoke (J14 item 4)

sheeprl's `policy_step` counts decisions x envs (NOT simulator frames); with action repeat 2,
env steps = 2 x policy_step. `POINTS` below are env steps; `total_steps` is policy steps.
The smoke must confirm this from the logged `Rewards/rew_avg` steps before any GPU run.

Box smoke 11-17 13:39 (A2 overrides, seed 0, compile off, CPU, total 1,200 policy steps, killed at
7 min — 0.19 gradient steps/s on this box): imports and steps; first logged losses at step 1,100:
reward_loss 5.541, Values/value 0.0, Values/return 0.0 = §2's step-0 check. NO `Rewards/rew_avg`
by policy step 1,100 — consistent with policy_step = decisions x 4 envs (a 500-decision cartpole
episode ends at 2,000), inconsistent with per-decision counting (would end at 500). Inference from
an absence: the positive check (an episode logged at 2,000) is the Kaggle smoke's first line.
"""
from __future__ import annotations

ACTION_REPEAT = 2

ROWS = {
    "A1": {"domain": "cartpole", "task": "balance", "points": {100_000: (794, 52), 200_000: (966, 17)}},
    "A2": {"domain": "cartpole", "task": "swingup", "points": {100_000: (478, 20), 200_000: (778, 19)}},
    "A3": {"domain": "ball_in_cup", "task": "catch", "points": {200_000: (949, 9)}},
}
SEEDS = (0, 1, 2)
EXTENSION_SEEDS = (3, 4)
CONTROL_SEED = 0

BASE = ["exp=dreamer_v3_5_dmc_walker_walk", "algo=dreamer_v3_5_1M",
        "env.wrapper.from_pixels=False", "env.wrapper.from_vectors=True",
        "algo.cnn_keys.encoder=[]", "algo.mlp_keys.encoder=[state]",
        "env.num_envs=4", "env.capture_video=False", "env.action_repeat=2",
        "algo.replay_ratio=0.5",   # = train ratio 512 / (16 x 64); the exp file's value, pinned
                                   # because the algo default is 0.03125 (train ratio 32 = the
                                   # §2 control "ratio 32" — must never leak into a row)
        "algo.compile.enabled=True"]


def last_env_step(row: str) -> int:
    return max(ROWS[row]["points"])


def overrides(row: str, seed: int, control: bool = False, base: list = None) -> list:
    """Command-line overrides for one run. The runner itself is never edited.
    `base` = another arm's settings (A2-v1S); None = the frozen rows' BASE, byte for byte."""
    r = ROWS[row]
    total_policy = last_env_step(row) // ACTION_REPEAT
    out = (BASE if base is None else list(base)) + [
        f"env.wrapper.domain_name={r['domain']}", f"env.wrapper.task_name={r['task']}",
        f"seed={seed}", f"algo.total_steps={total_policy}"]
    if control:
        out = [o for o in out if not o.startswith("algo.learning_starts=")]   # Hydra refuses a key twice
        # 2 x total, not total + 1: the runner floors learning_starts to whole iterations
        # (sheeprl utils.off_policy_schedule: learning_starts // num_envs), so total + 1 = 100,001 // 4
        # = 25,000 = total_iters trained ~2 gradient steps on the LAST iteration (box smoke 11-17 14:30:
        # Params/replay_ratio 0.001, losses logged at the final step). Correction 11-17, before any seed.
        out.append(f"algo.learning_starts={2 * total_policy}")
    return out


def band(mean: float, sd: float) -> tuple:
    return (mean - 2 * sd, min(1000.0, mean + 2 * sd))


def score_at(episodes, point_env: int):
    """episodes: [(end_env_step, return)]. Mean return of episodes ending in (0.9 P, P]; None = VOID."""
    xs = [ret for step, ret in episodes if 0.9 * point_env < step <= point_env]
    return sum(xs) / len(xs) if xs else None


def judge(row: str, per_seed: dict, control_episodes) -> dict:
    """per_seed: {seed: [(end_env_step, return)]}. Returns the row's verdict per the rule above."""
    pts = ROWS[row]["points"]
    out = {"row": row, "seeds": sorted(per_seed), "points": {}}
    verdict = "PASS"
    for p, (m, sd) in sorted(pts.items()):
        lo, hi = band(m, sd)
        scores = {s: score_at(eps, p) for s, eps in per_seed.items()}
        ctrl = score_at(control_episodes, p)
        rec = {"band": (lo, hi), "scores": scores, "control": ctrl}
        if any(v is None for v in scores.values()) or ctrl is None:
            rec["reading"] = "VOID (no episode in window)"
            verdict = "VOID"
        elif ctrl >= lo:
            rec["reading"] = "VOID (never-trains control inside or above band)"
            verdict = "VOID"
        else:
            mean = sum(scores.values()) / len(scores)
            rec["mean"] = mean
            rec["reading"] = "in band" if lo <= mean <= hi else "MISS"
            if rec["reading"] == "MISS" and verdict == "PASS":
                verdict = "MISS"
        out["points"][p] = rec
    out["verdict"] = verdict
    out["extension_due"] = verdict == "MISS" and len(per_seed) == len(SEEDS)
    return out


# ---------------------------------------------------------------------------------------------
# The kernel (J14 item 1). One Kaggle GPU kernel per row: the never-trains control FIRST (cheap, no
# gradient step, and it carries the unit check), then the seeds in order. Every run's record is
# written to `brain_a_rows_<row>.json` the moment it ends, so a session cut keeps what finished.
#
# Per-episode returns come from the runner's own stdout line, printed once per finished episode
# (sheeprl dreamer_v3_5.py: `Rank-0: policy_step={policy_step}, reward_env_{i}={ep_rew}`), NOT from
# the `Rewards/rew_avg` TensorBoard scalar — that one is a mean over a log interval (5,000 policy
# steps), and the rule scores individual episodes ending inside (0.9 P, P].
#
# UNIT CHECK (the smoke's inference made positive): a DMC episode is 1,000 frames and never
# terminates early, so with action repeat 2 and 4 envs it ends after 500 decisions; if policy_step
# counts decisions x envs, the FIRST episode line of a run reads policy_step=2000, for all 4 envs.
# Anything else stops the run before it spends GPU time, and the row records UNIT_FAIL.
# ---------------------------------------------------------------------------------------------
import json as _json
import os as _os
import re as _re
import signal as _signal
import subprocess as _sp
import sys as _sys
import time as _time

NUM_ENVS = 4
EPISODE_FRAMES = 1000
FIRST_EPISODE_POLICY_STEP = EPISODE_FRAMES // ACTION_REPEAT * NUM_ENVS   # 2,000
EP_RE = _re.compile(r"policy_step=(\d+), reward_env_(\d+)=([-+0-9.eE]+|nan|inf)")
ROW_DEADLINE_S = 9.0 * 3600        # in-kernel stop: the current run is cut (SIGINT) and kept
REMOTE_TIMEOUT_S = int(10.5 * 3600)  # box watcher; Kaggle's own session cap is 12 h
EST_HOURS = {"A1": 2.6, "A2": 2.6, "A3": 2.6}   # J5: 3 x (0.54 + 0.23) T4-h + control ~0.25


def _set(ovr: list, key: str, value) -> list:
    """Replace `key=` in an override list (Hydra refuses a key given twice). Smoke use only."""
    out = [o for o in ovr if not o.startswith(key + "=")]
    return out + [f"{key}={value}"]


def parse_episodes(text: str) -> list:
    """Runner stdout -> [(end_env_step, return, env_index)], env step = ACTION_REPEAT x policy_step."""
    out = []
    for m in EP_RE.finditer(text):
        out.append((ACTION_REPEAT * int(m.group(1)), float(m.group(3)), int(m.group(2))))
    return out


def unit_check(text: str):
    """None while no episode has ended; else (ok, reason) from the first episode lines."""
    eps = [(int(m.group(1)), int(m.group(2))) for m in EP_RE.finditer(text)]
    if not eps:
        return None
    first = eps[0][0]
    envs = sorted(i for s, i in eps if s == first)
    ok = first == FIRST_EPISODE_POLICY_STEP and envs == list(range(NUM_ENVS))
    return ok, f"first episode at policy_step={first} for envs {envs} (expect " \
               f"{FIRST_EPISODE_POLICY_STEP} for {list(range(NUM_ENVS))})"


def _first_scalars(logdir: str) -> dict:
    """§2's step-0 check, reported only: the first logged value of each Loss/State/Values/Grads tag."""
    try:
        import glob
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        out = {}
        for f in glob.glob(_os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True):
            ea = EventAccumulator(_os.path.dirname(f), size_guidance={"scalars": 0})
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                if tag.split("/")[0] in ("Loss", "State", "Values", "Grads", "Params"):
                    e = min(ea.Scalars(tag), key=lambda e: e.step)
                    out[tag] = (e.step, round(e.value, 4))
        return out
    except Exception as e:  # reported-only: never fails a run
        return {"error": repr(e)[:200]}


def run_one(row: str, seed: int, control: bool, out_dir: str, device: str = "cuda",
            deadline: float = None, extra: tuple = (), cmd_prefix: tuple = None,
            base: list = None, tag: str = None) -> dict:
    """One unmodified runner process. `extra` = (key, value) replacements, SMOKE ONLY.
    `cmd_prefix` replaces `python -m sheeprl` (tests drive a fake runner through it)."""
    name = f"{tag or row}_s{seed}" + ("_control" if control else "")
    logdir = _os.path.join(out_dir, "logs_" + name)
    ovr = overrides(row, seed, control, base)
    for k, v in extra:
        ovr = _set(ovr, k, v)
    cmd = list(cmd_prefix or [_sys.executable, "-m", "sheeprl"]) + ovr + [
        f"fabric.accelerator={device}", f"log_root={logdir}"]
    logf = _os.path.join(out_dir, name + ".log")
    print("RUN", name, " ".join(cmd), flush=True)
    t0 = _time.time()
    rec = {"name": name, "row": row, "seed": seed, "control": control, "cmd": cmd,
           "cut": False, "unit": None}
    with open(logf, "w") as log:
        p = _sp.Popen(cmd, cwd=out_dir, stdout=log, stderr=_sp.STDOUT)
        while p.poll() is None:
            _time.sleep(5)
            if rec["unit"] is None:
                u = unit_check(open(logf, errors="replace").read())
                if u is not None:
                    rec["unit"] = {"ok": u[0], "reason": u[1]}
                    print("UNIT", name, u[0], u[1], flush=True)
                    if not u[0]:
                        p.kill()
                        break
            if deadline is not None and _time.time() > deadline:
                rec["cut"] = True
                p.send_signal(_signal.SIGINT)
                try:
                    p.wait(120)
                except _sp.TimeoutExpired:
                    p.kill()
                break
        p.wait()
    text = open(logf, errors="replace").read()
    if rec["unit"] is None:
        u = unit_check(text)
        rec["unit"] = None if u is None else {"ok": u[0], "reason": u[1]}
    rec["rc"] = p.returncode
    rec["wall_s"] = round(_time.time() - t0, 1)
    rec["episodes"] = parse_episodes(text)
    steps = [int(s) for s in _re.findall(r"policy_step=(\d+)", text)]
    rec["last_episode_policy_step"] = max(steps) if steps else None
    rec["first_scalars"] = _first_scalars(logdir)
    if rec["rc"] not in (0, None) and not rec["cut"]:
        rec["log_tail"] = text[-3000:]
    return rec


def run_row(row: str, seeds, out_dir: str, device: str = "cuda", control: bool = True,
            deadline_s: float = ROW_DEADLINE_S, extra: tuple = (), cmd_prefix: tuple = None,
            base: list = None, tag: str = None) -> dict:
    """Control first, then each seed; the file is rewritten after every run.
    `base`/`tag`: another arm of the same row (A2-v1S) — its own settings and its own file name."""
    t_end = _time.time() + deadline_s
    res = {"row": row, "seeds": list(seeds), "control": control, "runs": [], "stopped": None}
    if tag:
        res["arm"] = tag
    path = _os.path.join(out_dir, f"brain_a_rows_{tag or row}.json")
    plan = ([(CONTROL_SEED, True)] if control else []) + [(s, False) for s in seeds]
    for seed, ctl in plan:
        if _time.time() > t_end:
            res["runs"].append({"name": f"{tag or row}_s{seed}" + ("_control" if ctl else ""), "row": row,
                                "seed": seed, "control": ctl, "not_started": "row deadline"})
            continue
        r = run_one(row, seed, ctl, out_dir, device, t_end, extra, cmd_prefix, base, tag)
        res["runs"].append(r)
        _json.dump(res, open(path, "w"))
        if not (r["unit"] or {}).get("ok", False):
            res["stopped"] = f"UNIT_FAIL or no episode in {r['name']}: {r['unit']}"
            break
    _json.dump(res, open(path, "w"))
    print("ROW_DONE", row, res["stopped"], flush=True)
    return res


def judge_files(row: str, paths) -> dict:
    """Box side: merge one or more kernel files for a row (base + extension) and apply the rule.
    Cut, failed or unstarted runs are left out of per_seed, so judge() sees them as missing ->
    reported, never scored as zeros."""
    per_seed, ctrl, notes = {}, None, []
    for pth in paths:
        for r in _json.load(open(pth))["runs"]:
            ok = "episodes" in r and not r.get("cut") and r.get("rc") == 0 and (r["unit"] or {}).get("ok")
            if not ok:
                notes.append(f"{r['name']}: not complete ({r.get('not_started') or r.get('rc')}, cut={r.get('cut')})")
                continue
            eps = [(s, ret) for s, ret, _ in r["episodes"]]
            if r["control"]:
                ctrl = eps
            else:
                per_seed[r["seed"]] = eps
    if ctrl is None or not per_seed:
        return {"row": row, "verdict": "NOT JUDGED", "notes": notes, "seeds": sorted(per_seed)}
    want = set(SEEDS) | (set(EXTENSION_SEEDS) if set(EXTENSION_SEEDS) & set(per_seed) else set())
    out = judge(row, per_seed, ctrl)
    missing = sorted(want - set(per_seed))
    if missing:
        out["verdict"] = "NOT JUDGED"
        notes.append(f"seeds missing: {missing}")
    out["notes"] = notes
    return out


_KERNEL_TAIL = r'''
from experiments.brain_a.rows import run_row
_r = run_row(__ROW__, __SEEDS__, WORK, "cuda", control=__CONTROL__)
print("RESULT " + json.dumps({"row": _r["row"], "stopped": _r["stopped"],
      "runs": [(x["name"], x.get("rc"), x.get("cut"), len(x.get("episodes", []))) for x in _r["runs"]]}), flush=True)
print("DONE", flush=True)
'''


def kernel_body(row: str, seeds=SEEDS, control: bool = True) -> str:
    """speed.py's tested install + version preamble (everything before `def one(`), then run_row."""
    from experiments.brain_a import speed
    head = speed._JOB.split("\ndef one(")[0]
    assert "INSTALL" in head and "WORK =" in head, "speed._JOB preamble moved"
    head = (head.replace("__DEV__", repr("cuda")).replace("__RUNS__", "[]")
            .replace("__BASE__", "[]").replace("__PIP__", repr(["sheeprl==0.8.5", "dm_control"])))
    tail = (_KERNEL_TAIL.replace("__ROW__", repr(row)).replace("__SEEDS__", repr(list(seeds)))
            .replace("__CONTROL__", repr(control)))
    return head + "\nprint('VERSIONS', json.dumps(_versions()), flush=True)\n" + tail


# ---------------------------------------------------------------------------------------------
# ARM A2-v1S (J14 (b)) — FROZEN by ruling 67, rule text `docs/research/prereg/a2_size_s.md`
# (freeze commit 8f13ab28). Row A2 again with the PUBLISHED agent and size: sheeprl's v1 port
# (`dreamer_v3`) at `dreamer_v3_S`. Bands, score window, seeds 0-2, the one extension (3-4) and the
# never-trains control are ROWS["A2"]'s, unchanged — `judge()` above is the judge, untouched.
# Pins: `fabric.precision=32-true` (the v1 exp file says bf16-mixed; A1-A3 ran 32-true, so only
# agent + size move against our rows) and `algo.learning_starts=1024` (the official prefill; the
# exp file says 1300). One seed per kernel; a timing cell before seed 0 — no score is read from it.
#
# Ours beside theirs: sheeprl's v1 port, not the official JAX code (danijar/dreamerv3 ec526b9);
# 32-true vs the official float16; compile on (numerics, not recipe). Control: in the v1 runner a
# run that never reaches learning_starts plays uniform-random actions (dreamer_v3.py:759), where
# the v2 runner's twin played its untrained policy — neither takes a gradient step.
# ---------------------------------------------------------------------------------------------
V1S_TAG = "A2v1S"
V1S_ROW = "A2"
BASE_V1S = ["exp=dreamer_v3_dmc_walker_walk", "algo=dreamer_v3_S",
            "env.wrapper.from_pixels=False", "env.wrapper.from_vectors=True",
            "algo.cnn_keys.encoder=[]", "algo.mlp_keys.encoder=[state]",
            # the v1 exp names its decoder keys itself ([rgb]); v3_5's follow the encoder's.
            # Timing cell 1 (jack-ladder-1795246441, 11-21 07:40) died in 40 s on it:
            # "CNN keys of the decoder must be contained in the encoder ones ... ['rgb']".
            "algo.cnn_keys.decoder=[]", "algo.mlp_keys.decoder=[state]",
            "env.num_envs=4", "env.capture_video=False", "env.action_repeat=2",
            "algo.replay_ratio=0.5", "algo.compile.enabled=True",
            "fabric.precision=32-true", "algo.learning_starts=1024"]
TIMING_POLICY_STEPS = 6000      # 1,024 prefill + ~2,488 gradient steps at ratio 0.5
TIMING_LOG_EVERY = 1000
V1S_EST_HOURS = 4.0             # a GUESS until the timing cell reads; the draft's 3.5-7 T4-h a seed


def time_scalars(logdir: str) -> dict:
    """Every logged value of the runner's own `Time/*` scalars: {tag: [(step, value), ...]}."""
    try:
        import glob
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        out = {}
        for f in glob.glob(_os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True):
            ea = EventAccumulator(_os.path.dirname(f), size_guidance={"scalars": 0})
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                if tag.startswith("Time/"):
                    out[tag] = sorted((e.step, round(e.value, 3)) for e in ea.Scalars(tag))
        return out
    except Exception as e:
        return {"error": repr(e)[:200]}


def project_seed_hours(time_series: dict, total_policy: int, learning_starts: int = 1024,
                       replay_ratio: float = 0.5):
    """T4-hours for one full seed from the runner's own rates. The FIRST logged window is dropped
    (it carries torch.compile); None when fewer than two windows of either rate exist — an
    unreadable meter is not a zero (LESSONS 11)."""
    tr = [v for _, v in time_series.get("Time/sps_train", [])][1:]
    en = [v for _, v in time_series.get("Time/sps_env_interaction", [])][1:]
    if not tr or not en or min(tr) <= 0 or min(en) <= 0:
        return None
    sps_tr, sps_en = sum(tr) / len(tr), sum(en) / len(en)
    grad = (total_policy - learning_starts) * replay_ratio
    return {"sps_train": round(sps_tr, 2), "sps_env": round(sps_en, 2), "windows": len(tr),
            "hours": round((grad / sps_tr + total_policy / sps_en) / 3600, 2)}


def timing_cell(out_dir: str, device: str = "cuda", cmd_prefix: tuple = None) -> dict:
    """Seed 0, TIMING_POLICY_STEPS policy steps, the arm's own settings; reads rates only."""
    extra = (("algo.total_steps", TIMING_POLICY_STEPS), ("metric.log_every", TIMING_LOG_EVERY),
             ("checkpoint.every", 10 ** 9), ("checkpoint.save_last", "False"),
             ("buffer.checkpoint", "False"))
    r = run_one(V1S_ROW, 0, False, out_dir, device, None, extra, cmd_prefix, BASE_V1S, V1S_TAG + "_timing")
    ts = time_scalars(_os.path.join(out_dir, "logs_" + r["name"]))
    total = last_env_step(V1S_ROW) // ACTION_REPEAT
    res = {"arm": V1S_TAG, "timing_only": True, "rc": r["rc"], "wall_s": r["wall_s"], "unit": r["unit"],
           "cmd": r["cmd"], "time_scalars": ts, "projection": project_seed_hours(ts, total),
           "n_episodes_seen_NOT_SCORED": len(r.get("episodes", [])), "log_tail": r.get("log_tail")}
    _json.dump(res, open(_os.path.join(out_dir, "brain_a_v1s_timing.json"), "w"))
    print("TIMING " + _json.dumps({k: res[k] for k in ("rc", "wall_s", "unit", "projection")}), flush=True)
    return res


_KERNEL_TAIL_V1S = r"""
from experiments.brain_a.rows import run_row, BASE_V1S, V1S_ROW, V1S_TAG
_r = run_row(V1S_ROW, __SEEDS__, WORK, "cuda", control=__CONTROL__, base=BASE_V1S, tag=V1S_TAG)
os.replace(os.path.join(WORK, "brain_a_rows_" + V1S_TAG + ".json"), os.path.join(WORK, __FETCH__))
print("RESULT " + json.dumps({"arm": V1S_TAG, "stopped": _r["stopped"],
      "runs": [(x["name"], x.get("rc"), x.get("cut"), len(x.get("episodes", []))) for x in _r["runs"]]}), flush=True)
print("DONE", flush=True)
"""
_KERNEL_TAIL_TIMING = r"""
from experiments.brain_a.rows import timing_cell
_t = timing_cell(WORK, "cuda")
print("DONE", flush=True)
"""


def _kernel_head() -> str:
    from experiments.brain_a import speed
    head = speed._JOB.split("\ndef one(")[0]
    assert "INSTALL" in head and "WORK =" in head, "speed._JOB preamble moved"
    head = (head.replace("__DEV__", repr("cuda")).replace("__RUNS__", "[]")
            .replace("__BASE__", "[]").replace("__PIP__", repr(["sheeprl==0.8.5", "dm_control"])))
    return head + "\nprint('VERSIONS', json.dumps(_versions()), flush=True)\n"


def v1s_fetch_name(seeds, control: bool) -> str:
    return "brain_a_v1s_" + "_".join(f"s{s}" for s in seeds) + ("_c" if control else "") + ".json"


def kernel_body_v1s(seeds, control: bool) -> str:
    """One kernel = the control (when asked) then the given seed(s). One seed per kernel by rule."""
    return _kernel_head() + (_KERNEL_TAIL_V1S.replace("__SEEDS__", repr(list(seeds)))
                             .replace("__CONTROL__", repr(control))
                             .replace("__FETCH__", repr(v1s_fetch_name(seeds, control))))


def kernel_body_timing() -> str:
    return _kernel_head() + _KERNEL_TAIL_TIMING


OUT = "/data/jack-logs/brain_a/rows"


def main(argv) -> int:
    """kernel ROW [--ext]   (Kaggle GPU; --ext = the rule's one extension, seeds 3-4, no control)
    score ROW            (judge every harvested file of the row)
    smoke                (box, CPU: A2 control, 2,100 policy steps, compile off — the unit check)
    v1s-timing           (arm A2-v1S: the timing cell, Kaggle GPU; rates only, no score)
    v1s-kernel SEED [--control] [--est=H]   (arm A2-v1S: one seed per kernel)
    v1s-score            (judge every harvested A2-v1S file by row A2's rule)"""
    _os.makedirs(OUT, exist_ok=True)
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "smoke":
        d = _os.path.join(OUT, "smoke")
        _os.makedirs(d, exist_ok=True)
        r = run_row("A2", (), d, "cpu", control=True, deadline_s=1200,
                    extra=(("algo.total_steps", 2100), ("algo.learning_starts", 4200),
                           ("algo.compile.enabled", "False")))
        print(_json.dumps({k: v for k, v in r["runs"][0].items() if k != "episodes"}, default=str)[:3000])
        print("episodes", r["runs"][0].get("episodes"))
        return 0 if (r["runs"][0]["unit"] or {}).get("ok") else 1
    if cmd == "score" and len(argv) > 2:
        import glob
        row = argv[2]
        paths = sorted(glob.glob(_os.path.join(OUT, f"rows_{row}*.json")))
        print(_json.dumps(judge_files(row, paths), indent=1, default=str))
        return 0
    if cmd == "kernel" and len(argv) > 2:
        from experiments.gpu import build_job, result_json, submit
        row, ext = argv[2], "--ext" in argv
        seeds = EXTENSION_SEEDS if ext else SEEDS
        sub = submit(build_job(kernel_body(row, seeds, control=not ext)), prefer="kaggle",
                     est_hours=EST_HOURS[row] * (2 / 3.5 if ext else 1), timeout_s=REMOTE_TIMEOUT_S,
                     fetch=[f"brain_a_rows_{row}.json"])
        print("SUBMIT", sub.backend, sub.ok, sub.job_id, sub.message, flush=True)
        if not sub.ok:
            return 1
        r = result_json(sub, f"brain_a_rows_{row}.json")
        dst = _os.path.join(OUT, f"rows_{row}{'_ext' if ext else ''}.json")
        open(dst, "w").write(_json.dumps(r))
        print("WROTE", dst, flush=True)
        return 0
    if cmd in ("v1s-timing", "v1s-kernel"):
        from experiments.gpu import build_job, result_json, submit
        if cmd == "v1s-timing":
            body, name, est, dst = kernel_body_timing(), "brain_a_v1s_timing.json", 0.5, "v1s_timing.json"
        else:
            seeds = tuple(int(a) for a in argv[2:] if a.isdigit())
            control = "--control" in argv
            if len(seeds) != 1 or seeds[0] not in SEEDS + EXTENSION_SEEDS:
                print("one seed per kernel, from", SEEDS + EXTENSION_SEEDS)
                return 2
            est = float(next((a.split("=")[1] for a in argv if a.startswith("--est=")), V1S_EST_HOURS))
            body, name = kernel_body_v1s(seeds, control), v1s_fetch_name(seeds, control)
            dst = name.replace("brain_a_", "")
        sub = submit(build_job(body), prefer="kaggle", est_hours=est, timeout_s=REMOTE_TIMEOUT_S, fetch=[name])
        print("SUBMIT", sub.backend, sub.ok, sub.job_id, sub.message, flush=True)
        if not sub.ok:
            return 1
        open(_os.path.join(OUT, dst), "w").write(_json.dumps(result_json(sub, name)))
        print("WROTE", _os.path.join(OUT, dst), flush=True)
        return 0
    if cmd == "v1s-score":
        import glob
        paths = sorted(glob.glob(_os.path.join(OUT, "v1s_s*.json")))
        print(_json.dumps(judge_files(V1S_ROW, paths), indent=1, default=str))
        return 0
    print(main.__doc__)
    return 2


if __name__ == "__main__":
    _sys.exit(main(_sys.argv))
