"""J14 (b): brain A in his world — a SMOKE, no reading of behaviour.

Question it answers, and only this: does the pinned runner (`sheeprl==0.8.5`, `algo=dreamer_v3_5`,
preset `dreamer_v3_5_1M`, the SAME overrides as A's known-answer rows in `rows.py` BASE except the
environment) run UNMODIFIED on `jack_env` for 20,000 decisions, take gradient steps without a
non-finite loss, and at what decisions/s?

DONE = all of, written 2026-11-18 before the first run:
  (1) STEPS     the runner exits rc 0 having reached algo.total_steps (its last logged step
                >= 0.9 x total), no cut.
  (2) LEARNS    gradient steps happened: `Loss/world_model_loss` logged at >= 2 distinct steps.
  (3) NO NaN    every logged Loss/* value is finite, and no "nan"/"inf" episode return.
  (4) SPEED     decisions/s = total decisions / wall seconds of the runner process, reported.
Anything about meals, lives or return is NOT read: 20k decisions is a smoke (philosophy 4 —
nothing here may be quoted as A eating or not).

His world (J11 ruling 39's pilot world, arm K's settings): `food_floor_only=True`,
`random_layout=True`, `spawn_clear` 1.55 m, d0 = 0.75, reward `dr100-eat`, random needs 1/6,
movement drain ramped in (`move_cost_dial`, arm K), `lazy_contact_audio` + `native_substep`
(speed, decisions proven identical), and the body-frame eye (`eye_frame="body"`, 16 rays).
`motor_frame="body"` too — the INBOX default proposed 11-18 07:40 for the pilot's learner arms
(no ruling yet); `--motor world` gives today's world-frame push.

Settings that differ from A's rows (ours beside theirs): environment `jack_env` via sheeprl's own
`env=gym` config (the id is `experiments.brain_a.in_world:JackWorld-v0`; gymnasium imports the
module, which registers it — the runner is not edited); `env.action_repeat=1` (one decision is
already 40 physics sub-steps; DMC's 2 is a property of DMC's control rate); a life ends at death
(terminated) — no time limit; `algo.total_steps=20000` = decisions summed over the 4 envs
(sheeprl counts policy_step = decisions x envs, rows.py's unit check), i.e. 5,000 per env;
buffer size 20,000 and no buffer checkpoint (the rows' 500k memmap is sized for 200k runs).
"""
from __future__ import annotations

import json as _json
import os as _os
import re as _re
import subprocess as _sp
import sys as _sys
import time as _time

ENV_ID = "JackWorld-v0"
GYM_ID = "experiments.brain_a.in_world:" + ENV_ID
SMOKE_DECISIONS = 20_000
NUM_ENVS = 4


def make(seed: int = 0, motor_frame: str = "body", render_mode=None, **_ignored):
    """One pilot-world JackEnv (arm K). `render_mode` is accepted and ignored: sheeprl's gym config
    passes render_mode="rgb_array" and vectors-only needs no renderer."""
    from experiments.jack_env import JackEnv, NEEDS_S_TRP, SPAWN_CLEAR_R
    from experiments.pilot_b2 import DIFFICULTY, PILOT_ARMS
    arm = PILOT_ARMS["K"]
    return JackEnv(seed=seed, difficulty=DIFFICULTY, reward=arm["reward"],
                   random_needs=NEEDS_S_TRP, random_layout=True, lazy_contact_audio=True,
                   spawn_clear=SPAWN_CLEAR_R, move_cost_dial=arm["move_cost_dial"],
                   food_floor_only=True, eye_frame="body", motor_frame=motor_frame,
                   native_substep=True)


def _register():
    import gymnasium as gym
    if ENV_ID not in gym.registry:
        gym.register(ENV_ID, entry_point=make)


_register()


def overrides(total: int = SMOKE_DECISIONS, seed: int = 0, motor: str = "body",
              device: str = "cuda", compile_on: bool = True) -> list:
    """rows.BASE with the DMC-only keys swapped for his world. The runner is never edited."""
    from experiments.brain_a.rows import BASE
    drop = ("env.wrapper.", "env.action_repeat=", "algo.compile.enabled=")
    out = [o for o in BASE if not o.startswith(drop)]
    return out + ["env=gym", f"env.id={GYM_ID}", "env.action_repeat=1",
                  "env.max_episode_steps=-1", f"+env.wrapper.motor_frame={motor}",
                  f"algo.compile.enabled={compile_on}",
                  f"seed={seed}", f"algo.total_steps={total}",
                  f"buffer.size={total}", "buffer.checkpoint=False", "buffer.memmap=False",
                  "checkpoint.every=1000000000", "checkpoint.save_last=False",
                  "metric.log_every=1000", f"fabric.accelerator={device}"]


def _scalars(logdir: str) -> dict:
    """tag -> [(step, value)] for Loss/* and Time/* scalars."""
    import glob
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    out = {}
    for f in glob.glob(_os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True):
        ea = EventAccumulator(_os.path.dirname(f), size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            out.setdefault(tag, []).extend((e.step, e.value) for e in ea.Scalars(tag))
    return out


def judge(rec: dict, total: int) -> dict:
    """The four DONE clauses from a run record (rc, cut, last_step, losses, episodes, wall_s)."""
    import math
    losses = rec.get("losses") or {}
    wm = losses.get("Loss/world_model_loss", [])
    finite = all(math.isfinite(v) for vals in losses.values() for _, v in vals)
    ep_ok = all(math.isfinite(r) for _, r, _ in rec.get("episodes", []))
    c = {"steps": rec.get("rc") == 0 and not rec.get("cut") and (rec.get("last_step") or 0) >= 0.9 * total,
         "learns": len({s for s, _ in wm}) >= 2,
         "no_nan": bool(losses) and finite and ep_ok}
    dps = total / rec["wall_s"] if rec.get("wall_s") else None
    return {"clauses": c, "done": all(c.values()), "decisions_per_s": dps}


def run(out_dir: str, total: int = SMOKE_DECISIONS, seed: int = 0, motor: str = "body",
        device: str = "cuda", compile_on: bool = True, deadline_s: float = 3 * 3600,
        cmd_prefix: tuple = None) -> dict:
    from experiments.brain_a.rows import EP_RE
    _os.makedirs(out_dir, exist_ok=True)
    name = f"inworld_s{seed}_{motor}"
    logdir = _os.path.join(out_dir, "logs_" + name)
    cmd = list(cmd_prefix or [_sys.executable, "-m", "sheeprl"]) + overrides(
        total, seed, motor, device, compile_on) + [f"log_root={logdir}"]
    repo = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    env = dict(_os.environ, PYTHONPATH=repo + _os.pathsep + _os.environ.get("PYTHONPATH", ""),
               MUJOCO_GL="disable")
    logf = _os.path.join(out_dir, name + ".log")
    print("RUN", name, " ".join(cmd), flush=True)
    rec = {"name": name, "cmd": cmd, "cut": False, "total": total, "motor": motor, "device": device}
    t0 = _time.time()
    with open(logf, "w") as log:
        p = _sp.Popen(cmd, cwd=out_dir, stdout=log, stderr=_sp.STDOUT, env=env)
        try:
            p.wait(deadline_s)
        except _sp.TimeoutExpired:
            rec["cut"] = True
            p.kill()
            p.wait()
    rec["wall_s"] = round(_time.time() - t0, 1)
    rec["rc"] = p.returncode
    text = open(logf, errors="replace").read()
    rec["episodes"] = [(int(m.group(1)), float(m.group(3)), int(m.group(2)))
                       for m in EP_RE.finditer(text)]
    try:
        sc = _scalars(logdir)
    except Exception as e:
        sc, rec["scalar_error"] = {}, repr(e)[:300]
    rec["losses"] = {k: v for k, v in sc.items() if k.startswith("Loss/")}
    rec["time"] = {k: v[-3:] for k, v in sc.items() if k.startswith("Time/")}
    steps = [s for v in sc.values() for s, _ in v] + [s for s, _, _ in rec["episodes"]]
    rec["last_step"] = max(steps) if steps else None
    if rec["rc"] != 0:
        rec["log_tail"] = text[-3000:]
    rec["verdict"] = judge(rec, total)
    _json.dump(rec, open(_os.path.join(out_dir, "brain_a_inworld.json"), "w"), default=str)
    print("INWORLD", _json.dumps(rec["verdict"]), flush=True)
    return rec


_KERNEL_TAIL = r'''
from experiments.brain_a.in_world import run
_r = run(WORK, device="cuda")
print("RESULT " + json.dumps({"rc": _r["rc"], "verdict": _r["verdict"], "episodes": len(_r["episodes"])}), flush=True)
print("DONE", flush=True)
'''


def kernel_body() -> str:
    """rows.kernel_body's tested preamble (speed._JOB before `def one(`), plus jack_env's pins."""
    from experiments.brain_a import speed
    from experiments.pilot_b2 import KERNEL_PINS
    head = speed._JOB.split("\ndef one(")[0]
    assert "INSTALL" in head and "WORK =" in head, "speed._JOB preamble moved"
    head = (head.replace("__DEV__", repr("cuda")).replace("__RUNS__", "[]")
            .replace("__BASE__", "[]").replace("__PIP__", repr(["sheeprl==0.8.5"] + KERNEL_PINS)))
    return head + "\nprint('VERSIONS', json.dumps(_versions()), flush=True)\n" + _KERNEL_TAIL


OUT = "/data/jack-logs/brain_a/in_world"


def main(argv) -> int:
    """smoke [N]   box, CPU, compile off, N decisions (default 600): the plumbing only
    kernel      Kaggle GPU: the 20k-decision smoke (J14 b), harvested to OUT/inworld.json"""
    _os.makedirs(OUT, exist_ok=True)
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "smoke":
        n = int(argv[2]) if len(argv) > 2 else 600
        r = run(_os.path.join(OUT, "smoke"), total=n, device="cpu", compile_on=False, deadline_s=1800)
        print(_json.dumps({k: v for k, v in r.items() if k not in ("losses", "episodes")}, default=str)[:3000])
        return 0 if r["verdict"]["done"] else 1
    if cmd == "kernel":
        from experiments.gpu import build_job, result_json, submit
        sub = submit(build_job(kernel_body()), prefer="kaggle", est_hours=1.0,
                     timeout_s=4 * 3600, fetch=["brain_a_inworld.json"])
        print("SUBMIT", sub.backend, sub.ok, sub.job_id, sub.message, flush=True)
        if not sub.ok:
            return 1
        dst = _os.path.join(OUT, "inworld.json")
        open(dst, "w").write(_json.dumps(result_json(sub, "brain_a_inworld.json")))
        print("WROTE", dst, flush=True)
        return 0
    print(main.__doc__)
    return 2


if __name__ == "__main__":
    _sys.exit(main(_sys.argv))
