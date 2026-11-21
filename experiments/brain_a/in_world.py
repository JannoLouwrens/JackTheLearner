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

DESIGN NOTE — brain A fed one key per sense (Next-up 3, PUBLISHED_PRIORS §18 item 2; 2026-11-19).
TEXT ONLY: nothing above changes; the frozen A1–A3 rows and this smoke keep the one flat key.
  Today: `JackEnv._obs` concatenates SENSES into one Box(74); sheeprl's gym path wraps a Box as
  {mlp_keys.encoder[0]: obs} and, given >1 mlp key on a Box, WARNS and keeps only the first
  (sheeprl 0.8.5 utils/env.py:192-202). So A sees one key `state`: one decoder head, no per-sense
  loss, no way to hide one sense.
  The keys (widths measured 11-19 on make(0).reset, w.observe()): vision 32 (body-frame eye),
  audio 16, touch 8, proprio 12, needs 6 = 74. Language (cores.MODALITIES, 32) is
  empty today and gets no key until it carries something.
  How, with the runner unedited: a `keys=per_sense` path in `make` returning
  spaces.Dict({k: Box(width_k)}) from `w.observe()` (a Dict env passes sheeprl's wrapper as is),
  and overrides `algo.mlp_keys.encoder=[vision,audio,touch,proprio,needs]` + the same list for
  `algo.mlp_keys.decoder`. sheeprl then builds one encoder MLP over the concatenated keys and one
  decoder head per key (agent.py:842-868, 909-935) — the official Dreamer form.
  The loss, ours beside theirs: dreamer_v3_5.py:155-156 SUMS symlog_mse over keys, each summed
  over its numbers, at one `observation_scale` — a sense's pull is its width (official rssm/agent
  likewise; §18). At equal per-number error vision pulls 32/74 = 43 %, audio 22 %, proprio 16 %, touch 11 %,
  needs 8 % — the sense §18 says must not be hidden gets the least. The width-normalised loss
  (CoRAL 2302.05342; M3L 2311.00924) = each key's sum divided by its width (equal 20 % each) has
  NO knob in sheeprl: it is a one-line change at :156 and therefore a CHANGED RUNNER, a setting
  that differs from Dreamer's recipe, to be listed as such and raced (sum vs width-normalised),
  never assumed. Whole-sense dropout (TORCS 1705.10422) cannot be done env-side without also
  zeroing the decoder's TARGET (the env feeds both); it too needs an encoder-side mask = runner
  change. Hide-at-test (§18 "record does NOT answer") is env-side and needs no edit: zero /
  shuffle / matched-noise one key in the wrapper, eval lives only.
  Order when it is run (after J11, not before): arm keys-sum (no edit) first, its smoke must
  reproduce this file's DONE (1)–(4); keys-norm and dropout only as raced arms beside it.
  BUILT 11-20 (arm keys-sum, no runner edit): `make(keys="per_sense")` / `overrides(keys=...)` /
  `smoke N --per-sense`. Box smoke 1,200 decisions, CPU, compile off: rc 0, sheeprl prints
  "Encoder MLP keys" and "Decoder MLP keys" = [vision, audio, touch, proprio, needs], no
  only-the-first-key warning; STEPS yes, NO NaN yes (all 8 Loss/* finite at step 1200,
  reward_loss 5.537 = the flat smoke's), LEARNS not decidable at one logged step — the same
  reading as the flat smoke, left to a GPU kernel. 2.2 dec/s on a loaded box: no speed claim.
  Gymnasium sorts a Dict's keys alphabetically; sheeprl reads by name, so nothing depends on it.

J14 (d) — THE REPLAY TRAVELS WITH THE CHECKPOINT (2026-11-21). `overrides(ckpt_every=N)` swaps the
three no-checkpoint settings for sheeprl's own `buffer.checkpoint=True`, `checkpoint.every=N`,
`checkpoint.save_last=True` and adds `checkpoint.memmap_buffer=copy`; `resume_from=` resumes. Off
(the default) the list is byte-identical to before (checked against HEAD's module). `resume-test`
(box, CPU, 3 runs, ~2 min): save 400 decisions -> resume to 480 -> buffer per env 100 -> 120 rows,
first 100 bit-identical on every key but `truncated` = PASS; CONTROL fresh 80-decision run = 20
rows, every key differs = FAIL as it must. Tests the BUFFER only (learning_starts 1,024 > both runs:
no gradient step). Found: a resume merges the old run's config, so it logs into the SAVED run's
log_root, not its own — `last_checkpoint` looks in both.
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


KEYS = ("flat", "per_sense")


def per_sense_keys() -> list:
    """The sheeprl mlp keys of the per-sense path, in JackEnv's concatenation order."""
    from experiments.jack_env import SENSES
    return list(SENSES)


def _per_sense(env):
    """Next-up 3 (design note above): the flat Box(74) split back into one Box per sense, as a
    gymnasium Dict, by the widths `w.observe()` reports at construction. JackEnv builds every
    observation (reset, step, the dying body) with `_obs` = concatenate(SENSES), so the split is
    exact: concatenating the Dict's values in SENSES order returns the flat vector bit for bit
    (planted test). Nothing in the world or the runner changes."""
    import gymnasium as gym
    import numpy as np
    from experiments.jack_env import SENSES
    o = env.w.observe()
    widths = [int(np.asarray(o[k]).size) for k in SENSES]
    assert sum(widths) == env.observation_space.shape[0], (widths, env.observation_space.shape)
    cuts = np.cumsum([0] + widths)
    space = gym.spaces.Dict({k: gym.spaces.Box(-np.inf, np.inf, (w,), np.float32)
                             for k, w in zip(SENSES, widths)})

    def split(flat):
        return {k: np.asarray(flat[cuts[i]:cuts[i + 1]], np.float32) for i, k in enumerate(SENSES)}

    return gym.wrappers.TransformObservation(env, split, space)


def make(seed: int = 0, motor_frame: str = "body", render_mode=None, keys: str = "flat",
         **_ignored):
    """One pilot-world JackEnv (arm K). `render_mode` is accepted and ignored: sheeprl's gym config
    passes render_mode="rgb_array" and vectors-only needs no renderer. `keys="per_sense"` hands
    sheeprl a Dict of one Box per sense (vision, audio, touch, proprio, needs) instead of one flat
    Box; the default "flat" is the frozen rows' and the 11-18 smoke's form, unchanged."""
    from experiments.jack_env import JackEnv, NEEDS_S_TRP, SPAWN_CLEAR_R
    from experiments.pilot_b2 import DIFFICULTY, PILOT_ARMS
    if keys not in KEYS:
        raise ValueError(f"keys must be one of {KEYS}, got {keys!r}")
    arm = PILOT_ARMS["K"]
    env = JackEnv(seed=seed, difficulty=DIFFICULTY, reward=arm["reward"],
                  random_needs=NEEDS_S_TRP, random_layout=True, lazy_contact_audio=True,
                  spawn_clear=SPAWN_CLEAR_R, move_cost_dial=arm["move_cost_dial"],
                  food_floor_only=True, eye_frame="body", motor_frame=motor_frame,
                  native_substep=True)
    return _per_sense(env) if keys == "per_sense" else env


def _register():
    import gymnasium as gym
    if ENV_ID not in gym.registry:
        gym.register(ENV_ID, entry_point=make)


_register()


def overrides(total: int = SMOKE_DECISIONS, seed: int = 0, motor: str = "body",
              device: str = "cuda", compile_on: bool = True, keys: str = "flat",
              ckpt_every: int = 0, resume_from: str = None, buffer_size: int = None) -> list:
    """rows.BASE with the DMC-only keys swapped for his world. The runner is never edited.
    keys="per_sense": the encoder AND decoder mlp keys become the five senses (the decoder is
    named explicitly though the preset defaults it to the encoder, so the arm cannot silently
    fall back) and the env wrapper gets `keys=per_sense`."""
    from experiments.brain_a.rows import BASE
    if keys not in KEYS:
        raise ValueError(f"keys must be one of {KEYS}, got {keys!r}")
    drop = ("env.wrapper.", "env.action_repeat=", "algo.compile.enabled=")
    if keys == "per_sense":
        drop = drop + ("algo.mlp_keys.",)
    out = [o for o in BASE if not o.startswith(drop)]
    if keys == "per_sense":
        lst = "[" + ",".join(per_sense_keys()) + "]"
        out += [f"algo.mlp_keys.encoder={lst}", f"algo.mlp_keys.decoder={lst}",
                "+env.wrapper.keys=per_sense"]
    out = out + ["env=gym", f"env.id={GYM_ID}", "env.action_repeat=1",
                 "env.max_episode_steps=-1", f"+env.wrapper.motor_frame={motor}",
                 f"algo.compile.enabled={compile_on}",
                 f"seed={seed}", f"algo.total_steps={total}",
                 f"buffer.size={buffer_size or total}", "buffer.checkpoint=False", "buffer.memmap=False",
                 "checkpoint.every=1000000000", "checkpoint.save_last=False",
                 "metric.log_every=1000", f"fabric.accelerator={device}"]
    if ckpt_every:
        # J14 (d): the checkpoint carries the replay (sheeprl's own options, runner unedited);
        # memmap_buffer=copy so a memmapped buffer travels with its data, not by file reference
        # (a Kaggle kernel's resume is on another node). Off (ckpt_every=0) = the list above, byte for byte.
        swap = {"buffer.checkpoint=False": "buffer.checkpoint=True",
                "checkpoint.every=1000000000": f"checkpoint.every={int(ckpt_every)}",
                "checkpoint.save_last=False": "checkpoint.save_last=True"}
        out = [swap.get(o, o) for o in out] + ["checkpoint.memmap_buffer=copy"]
    if resume_from:
        out.append(f"checkpoint.resume_from={resume_from}")
    return out


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
        cmd_prefix: tuple = None, keys: str = "flat", ckpt_every: int = 0,
        resume_from: str = None, buffer_size: int = None, name: str = None) -> dict:
    from experiments.brain_a.rows import EP_RE
    _os.makedirs(out_dir, exist_ok=True)
    name = name or f"inworld_s{seed}_{motor}" + ("" if keys == "flat" else "_" + keys)
    logdir = _os.path.join(out_dir, "logs_" + name)
    cmd = list(cmd_prefix or [_sys.executable, "-m", "sheeprl"]) + overrides(
        total, seed, motor, device, compile_on, keys, ckpt_every, resume_from,
        buffer_size) + [f"log_root={logdir}"]
    repo = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    env = dict(_os.environ, PYTHONPATH=repo + _os.pathsep + _os.environ.get("PYTHONPATH", ""),
               MUJOCO_GL="disable")
    logf = _os.path.join(out_dir, name + ".log")
    print("RUN", name, " ".join(cmd), flush=True)
    rec = {"name": name, "cmd": cmd, "cut": False, "total": total, "motor": motor, "device": device,
           "keys": keys}
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
    rec["logdir"] = logdir
    _json.dump(rec, open(_os.path.join(out_dir, name + ".json" if (keys != "flat" or ckpt_every
                                                                    or resume_from)
                                       else "brain_a_inworld.json"), "w"), default=str)
    print("INWORLD", _json.dumps(rec["verdict"]), flush=True)
    return rec


def last_checkpoint(logdir: str):
    """(policy_step, ckpt path, its buffer file or None) of the latest `ckpt_<step>_0.ckpt` under logdir."""
    import glob
    best = None
    for c in glob.glob(_os.path.join(logdir, "**", "checkpoint", "ckpt_*_0.ckpt"), recursive=True):
        m = _re.search(r"ckpt_(\d+)_0\.ckpt$", c)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), c)
    if best is None:
        return None
    buf = _os.path.join(_os.path.dirname(_os.path.dirname(best[1])), "checkpoint_buffers",
                        f"ckpt_{best[0]}_buffer_rank_0.pt")
    return best[0], best[1], (buf if _os.path.exists(buf) else None)


def buffer_rows(buf_path):
    """Per env: (rows filled, {key: array of the filled rows, oldest first}) of a saved
    EnvIndependentReplayBuffer. None when the checkpoint carried no buffer. `len()` of a sheeprl
    buffer is its CAPACITY, so filled = capacity if full else `_pos` (read, not assumed)."""
    if not buf_path:
        return None
    import numpy as np
    import torch
    rb = torch.load(buf_path, weights_only=False)
    out = []
    for b in rb.buffer:
        n = b.buffer_size if b.full else b._pos
        assert not b.full, "resume test sizes its buffer so it never wraps"
        out.append((int(n), {k: np.array(b[k][:n]) for k in b.buffer.keys()}))
    return out


def resume_check(saved, resumed, added_per_env: int) -> dict:
    """J14 (d) clause: the resumed run's buffer = the saved one, then `added_per_env` new rows.
    PASS iff every env's filled rows = saved + added AND its first `saved` rows equal the saved
    rows bit for bit on every key except `truncated` (sheeprl's checkpoint writes truncated=1 on
    the last saved row of each env, by design: the env state is not saved)."""
    import numpy as np
    if saved is None or resumed is None:
        return {"pass": False, "why": "a checkpoint carried no buffer", "saved": saved is not None,
                "resumed": resumed is not None}
    per_env = []
    for (ns, rs), (nr, rr) in zip(saved, resumed):
        same = {k: bool(np.array_equal(rs[k], rr[k][:ns])) for k in rs if k != "truncated"}
        per_env.append({"saved": ns, "resumed": nr, "len_ok": nr == ns + added_per_env,
                        "content_ok": all(same.values()), "keys_differ": [k for k, v in same.items() if not v]})
    ok = len(saved) == len(resumed) and all(e["len_ok"] and e["content_ok"] for e in per_env)
    return {"pass": ok, "per_env": per_env}


RT_SAVE, RT_MORE = 400, 80   # decisions summed over the 4 envs: 100 rows/env saved, +20 resumed


def resume_test(out_dir: str, cmd_prefix=None) -> dict:
    """Box, CPU: (A) run RT_SAVE decisions, checkpoint at the end with the replay; (B) resume A's
    checkpoint to RT_SAVE + RT_MORE, checkpoint again; (C, CONTROL, must fail) a FRESH run of
    RT_MORE decisions with the same options — what a resume without the replay amounts to.
    learning_starts (1,024) > both runs: random play, no gradient step — this tests the buffer only."""
    import shutil
    shutil.rmtree(out_dir, ignore_errors=True)
    kw = dict(device="cpu", compile_on=False, deadline_s=1800, cmd_prefix=cmd_prefix,
              ckpt_every=RT_SAVE, buffer_size=4 * RT_SAVE)
    rec = {"save": RT_SAVE, "more": RT_MORE}
    a = run(out_dir, total=RT_SAVE, name="rt_save", **kw)
    ca = last_checkpoint(a["logdir"])
    rec["A"] = {"rc": a["rc"], "ckpt": ca, "wall_s": a["wall_s"]}
    if a["rc"] != 0 or not ca:
        rec["verdict"] = "VOID: the save run did not checkpoint"
        rec["A"]["log_tail"] = a.get("log_tail")
        return rec
    b = run(out_dir, total=RT_SAVE + RT_MORE, name="rt_resume", resume_from=ca[1], **kw)
    cb = last_checkpoint(b["logdir"]) or last_checkpoint(a["logdir"])
    if cb and cb[0] <= ca[0]:
        cb = None
    rec["B"] = {"rc": b["rc"], "ckpt": cb, "wall_s": b["wall_s"], "log_tail": b.get("log_tail")}
    c = run(out_dir, total=RT_MORE, name="rt_control_fresh", **kw)
    cc = last_checkpoint(c["logdir"])
    rec["C"] = {"rc": c["rc"], "ckpt": cc, "wall_s": c["wall_s"]}
    sa = buffer_rows(ca[2])
    per = RT_MORE // NUM_ENVS
    rec["check"] = resume_check(sa, buffer_rows(cb[2]) if cb else None, per)
    rec["control"] = resume_check(sa, buffer_rows(cc[2]) if cc else None, per)
    rec["verdict"] = ("PASS" if rec["check"]["pass"] and not rec["control"]["pass"] else
                      "FAIL" if not rec["control"]["pass"] else "VOID: control passed")
    return rec


_KERNEL_TAIL = r'''
from experiments.brain_a.in_world import run
_r = run(WORK, device="cuda")
print("RESULT " + json.dumps({"rc": _r["rc"], "verdict": _r["verdict"], "episodes": len(_r["episodes"])}), flush=True)
print("DONE", flush=True)
'''


def kernel_out_name(keys: str = "flat") -> str:
    """The json `run` writes on the GPU for `keys` (the flat name predates the per-sense path)."""
    if keys not in KEYS:
        raise ValueError(f"keys must be one of {KEYS}, got {keys!r}")
    return "brain_a_inworld.json" if keys == "flat" else f"inworld_s0_body_{keys}.json"


def kernel_body(keys: str = "flat") -> str:
    """rows.kernel_body's tested preamble (speed._JOB before `def one(`), plus jack_env's pins.
    keys="flat" is byte-identical to the body before the `keys` arg (11-20, tested); any other
    value only adds `keys=` to the one `run` call."""
    from experiments.brain_a import speed
    from experiments.pilot_b2 import KERNEL_PINS
    head = speed._JOB.split("\ndef one(")[0]
    assert "INSTALL" in head and "WORK =" in head, "speed._JOB preamble moved"
    head = (head.replace("__DEV__", repr("cuda")).replace("__RUNS__", "[]")
            .replace("__BASE__", "[]").replace("__PIP__", repr(["sheeprl==0.8.5"] + KERNEL_PINS)))
    kernel_out_name(keys)   # validates keys
    tail = _KERNEL_TAIL if keys == "flat" else _KERNEL_TAIL.replace(
        'run(WORK, device="cuda")', f'run(WORK, device="cuda", keys={keys!r})')
    return head + "\nprint('VERSIONS', json.dumps(_versions()), flush=True)\n" + tail


OUT = "/data/jack-logs/brain_a/in_world"
PROBE_PY = "/data/venvs/jack-a-probe/bin/python"   # the box's sheeprl==0.8.5


def main(argv) -> int:
    """smoke [N] [--per-sense]   box, CPU, compile off, N decisions (default 600): the plumbing only
    kernel [--per-sense]  Kaggle GPU: the 20k-decision smoke (J14 b), harvested to OUT/inworld.json
                          (--per-sense: OUT/inworld_per_sense.json, Next-up 1 item (2))
    resume-test           box, CPU: J14 (d) — save with the replay, resume, buffer equal + a fresh
                          control that must fail -> OUT/resume_test.json"""
    _os.makedirs(OUT, exist_ok=True)
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "smoke":
        try:
            import tensorboard  # noqa: F401  (the judge reads the runner's scalars with it)
        except ImportError:
            if _os.path.exists(PROBE_PY) and _os.path.realpath(_sys.executable) != _os.path.realpath(PROBE_PY):
                # found 11-20: from the project venv the smoke ran but could not read its losses
                _os.execv(PROBE_PY, [PROBE_PY, "-m", "experiments.brain_a.in_world"] + list(argv[1:]))
        keys = "per_sense" if "--per-sense" in argv else "flat"
        rest = [a for a in argv[2:] if not a.startswith("--")]
        n = int(rest[0]) if rest else 600
        prefix = (PROBE_PY, "-m", "sheeprl") if _os.path.exists(PROBE_PY) else None
        r = run(_os.path.join(OUT, "smoke"), total=n, device="cpu", compile_on=False, deadline_s=1800,
                cmd_prefix=prefix, keys=keys)
        print(_json.dumps({k: v for k, v in r.items() if k not in ("losses", "episodes")}, default=str)[:3000])
        return 0 if r["verdict"]["done"] else 1
    if cmd == "resume-test":
        try:
            import tensorboard  # noqa: F401
        except ImportError:
            if _os.path.exists(PROBE_PY) and _os.path.realpath(_sys.executable) != _os.path.realpath(PROBE_PY):
                _os.execv(PROBE_PY, [PROBE_PY, "-m", "experiments.brain_a.in_world"] + list(argv[1:]))
        prefix = (PROBE_PY, "-m", "sheeprl") if _os.path.exists(PROBE_PY) else None
        r = resume_test(_os.path.join(OUT, "resume_test"), prefix)
        _json.dump(r, open(_os.path.join(OUT, "resume_test.json"), "w"), default=str, indent=1)
        print("RESUME_TEST", r["verdict"], _json.dumps({k: r.get(k) for k in ("check", "control")},
                                                       default=str)[:2000], flush=True)
        return 0 if r["verdict"] == "PASS" else 1
    if cmd == "kernel":
        from experiments.gpu import build_job, result_json, submit
        keys = "per_sense" if "--per-sense" in argv else "flat"
        fname = kernel_out_name(keys)
        sub = submit(build_job(kernel_body(keys)), prefer="kaggle", est_hours=1.0,
                     timeout_s=4 * 3600, fetch=[fname])
        print("SUBMIT", sub.backend, sub.ok, sub.job_id, sub.message, flush=True)
        if not sub.ok:
            return 1
        dst = _os.path.join(OUT, "inworld.json" if keys == "flat" else f"inworld_{keys}.json")
        open(dst, "w").write(_json.dumps(result_json(sub, fname)))
        print("WROTE", dst, flush=True)
        return 0
    print(main.__doc__)
    return 2


if __name__ == "__main__":
    _sys.exit(main(_sys.argv))
