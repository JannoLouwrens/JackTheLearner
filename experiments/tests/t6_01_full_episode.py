"""T6.01 — Full episode completes: one life of the shipped companion, N
minutes, no crash, no hang, no non-finite action.

GOAL.md path step 6: "A living Jack: runs for hours, remembers across
sessions, explores unprompted." T6.01 is the first rung of that stage and is
deliberately LIVENESS, NOT CAPABILITY (T6.04's own notes draw that line): the
claim is only that the integrated companion runtime — brain tick, emotional
decay, autonomous monologue, physics, auto-save — survives N_MINUTES of
continuous operation without crashing, hanging, or emitting a non-finite
action. It is also the lift condition of the 2026-09-17 STANDING FREEZE,
which ends when this spec records any verdict at all.

WHAT A "FULL COMPANION SESSION" IS, pinned here so the claim cannot drift:
the per-frame sequence `VirtualWorld.run()` documents — `_update_brain`,
`_update_emotional`, `_update_autonomous`, `_step_physics`,
`_auto_save_tick` — at the shipped 50 fps cadence, with the brain built
exactly as `VirtualWorld.main()` builds it (UnifiedBrainConfig defaults,
pretrained encoders off, `.eval()`), on the shipped EMBODIED scene
(`SCENE_CATALOG["humanoid"]`, 57 actuators). The bodiless default room would
be the cheaper reading of "full episode" that the freeze's own text forbids
buying. Two shipped steps are absent headless, and they are exactly the two
whose own first lines no-op without a display: `_handle_events` (pygame
input) and `_render` (screen blit). pygame is not installed in this venv, so
the session exercised here IS the shipped fallback path, not a trimmed one.

ONE DISCLOSED DIVERGENCE from `main()`: `llm_enabled = False`. `main()`
leaves the config default True, which downloads SmolLM2-1.7B from
HuggingFace at session start. The owner decree (2026-08-09) places the LLM
in Jack's WORLD, out-of-process, never inside him; T6.03 made the same call
for the same reason, and a ladder session must not depend on network fetch.
The monologue runs on its shipped template fallback.

WHY THE HARNESS IS A PARENT/CHILD PAIR, not a loop in this process:
  1. HANG detection from inside a hung loop is impossible — if one call
     blocks forever, no code after it runs. The session runs in a child
     process that writes a heartbeat file every frame; the parent kills it
     and records hangs=1 if the heartbeat goes stale for HANG_S.
  2. A hard crash (segfault, MemoryError) kills only the child; the parent
     records hard_crash=1 with whatever the last heartbeat saw.

WHY THE SWALLOWED-ERROR CHANNEL IS GATED: `VirtualWorld._update_brain`,
`_step_physics`, `_update_emotional` and `_update_autonomous` each wrap
their body in try/except and log failures at logger.debug. A brain that
crashes on every single tick therefore looks, from outside, like a quiet
session. The harness attaches a logging handler to the VirtualWorld logger
and counts those records as crash_frames (brain tick / physics step /
emotional / monologue errors) and sensor_frames (eye render / touch read —
the shipped graceful-degradation path for absent senses, recorded but not
gated as a crash). The registry-declared CONTROL exists to prove this
instrument is alive: a NaN-emitting twin session must register
nonfinite_actions > 0 and a raising twin must register crash_frames > 0, or
the run is VOID — a monitor that reports a faulted session clean measures
nothing (the at-chance-control lesson, LESSONS.md 2026-08-21).

EXPECTED VERDICT ON CURRENT HEAD, pre-registered so the first recording run
cannot be scored as a surprise: FAIL via crash_frames. Measured at
implementation time (2026-10-06, out-of-band 0.5-min smoke, no ledger
write): the brain's action head emits action_dim=17 values and
`humanoid_full.xml` has nu=57 actuators, so `apply_action` — which refuses
width mismatches rather than truncating (T0.06's lesson) — raises on EVERY
frame that produces an action: crash_frames 177 of 177 frames, first_crash
"Physics step error: action width 17 != model.nu 57". The companion cannot
drive his own body. That is a real integration measurement, the first the
Tier-6 rung has produced, and it corroborates the D41 drift findings (two
certified bridges, zero shipped callers) from the runtime side. Do not
repair it by padding or truncating the action; the repair is an
architecture/unison question that routes through the Review. Control twins
verified alive the same sitting: NaN twin nonfinite_actions 1242 > 0, raise
twin crash_frames 2479 > 0 — both planted faults were seen. Peak child RSS
measured 2129 MB (recorded as maxrss_mb every run): the shipped brain +
MuJoCo session transiently exceeds the box's ~1.5 GB guideline, which the
recording run's scheduler should know before it buys the 30-minute session.

ALSO FOUND IMPLEMENTING, recorded here because no instrument watches this
surface: `humanoid_full.xml` has no camera named "eye" (it has left_eye /
right_eye / head_cam), and `_get_eye_image` looks up "eye" specifically — so
the embodied companion session is blind by scene wiring, vision arrives as
None, and the brain runs proprioception+touch only. eye_alive_frames /
eye_sampled are recorded so the ledger shows it; it is not gated here
because sensor fallback is the shipped degradation path and the claim is
liveness. The playground (W0) eye contract (EYE_POS/EYE_XYAXES) postdates
these companion scenes and was never ported.

AND A SECOND ONE, which the first smoke measured the hard way: at main()'s
default 800x600, `mujoco.Renderer` refuses the model's 640x480 default
offscreen framebuffer, `_init_mujoco` catches the exception and FAILS
CLOSED to `mj_model = None` — and the session then runs to completion
WORLDLESS, zero-observation, every log line healthy (measured: completed 1,
173 frames, construction_ok 0, nu 0). The shipped humanoid-scene default is
therefore broken offscreen, and the failure is silent by design. The
construction_ok VOID lane exists because of exactly this: gate what was
BUILT, never what was logged.

VOID lanes (rig, any of these means the run did not test the claim):
  construction_ok != 1          world/brain/humanoid did not build (nu>0,
                                mujoco model present, brain parameters > 0)
  frames < FRAMES_MIN           the loop never actually ran
  action_frames==0 AND
    crash_frames==0             the brain neither acted nor even failed —
                                the session never exercised it
  c_nan_detected != 1 or
    c_raise_detected != 1       the planted faults were not seen: the
                                monitor is dead (registry control)

Claim conjuncts (all must hold, else FAIL):
  completed == 1 and minutes_survived reported (the loop reached N_MINUTES)
  hard_crash == 0   hangs == 0   crash_frames == 0
  nonfinite_actions == 0   nonfinite_sim == 0

Seeds: 1 (the registry's registration; liveness, not an effect estimate).
Budget: CPU_LONG — N_MINUTES=30 plus two 0.5-minute control twins.

Pilot modes (out-of-band, NO ledger write — "running a spec writes the
ledger" is the lesson, so the smoke path never touches run()):
    python -m experiments.tests.t6_01_full_episode smoke [seed] [minutes]
    python -m experiments.tests.t6_01_full_episode nan   [seed]
    python -m experiments.tests.t6_01_full_episode raise [seed]
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]
PY = "/data/venvs/jackthelearner/bin/python"

# ── pre-registered constants (set before any recording run; moving one is a
#    threshold move and is forbidden without a registry-grade reason) ───────
N_MINUTES = 30.0          # the registered session length ("N minutes")
CONTROL_MINUTES = 0.5     # each fault twin runs this long
FAULT_AFTER_S = 5.0       # faults inject after construction has proven alive
HANG_S = 60.0             # heartbeat silence that counts as a hang (target
                          # frame period is 20 ms; 3000x over it is hung)
BUILD_GRACE_S = 180.0     # brain construction time allowed before the FIRST
                          # heartbeat (58M params on 2 CPU threads)
HEARTBEAT_POLL_S = 2.0
FRAMES_MIN = 10           # fewer than this and the loop never really ran
EYE_SAMPLE_EVERY = 50     # canary cadence (frames) for the blind-sensor trap
SCENE = "humanoid"        # the shipped embodied scene; see docstring

IMPL_DEPS = ["VirtualWorld.py", "UnifiedBrain.py", "TaskManager.py",
             "EmotionalState.py", "InnerMonologue.py", "Persistence.py"]

# ── the child session, run in its own process so hangs and hard crashes are
#    observable from outside ────────────────────────────────────────────────
_CHILD = r'''
import json, os, resource, sys, time
cfg = json.loads(sys.argv[1])
sys.path.insert(0, cfg["repo"])

from experiments.render import ensure_gl
ensure_gl()                      # MUST precede any mujoco import (0aaa)

import numpy as np
import torch
torch.set_num_threads(2)

import logging
swallowed = {"crash": 0, "sensor": 0, "first_crash": "", "first_sensor": ""}
_CRASH = ("Brain tick error", "Physics step error",
          "Emotional update error", "Inner monologue error")
_SENSOR = ("Eye render error", "Touch read error", "Render error")

class _Capture(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if msg.startswith(_CRASH):
            swallowed["crash"] += 1
            if not swallowed["first_crash"]:
                swallowed["first_crash"] = msg[:300]
        elif msg.startswith(_SENSOR):
            swallowed["sensor"] += 1
            if not swallowed["first_sensor"]:
                swallowed["first_sensor"] = msg[:300]

_vlog = logging.getLogger("VirtualWorld")
_vlog.setLevel(logging.DEBUG)
_vlog.addHandler(_Capture())

from VirtualWorld import VirtualWorld, WorldConfig, SCENE_CATALOG
from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

seed = int(cfg["seed"])
bc = UnifiedBrainConfig()
bc.use_pretrained_vision = False   # as main() sets them
bc.use_pretrained_audio = False
bc.llm_enabled = False             # disclosed divergence; see spec docstring
torch.manual_seed(seed)
brain = UnifiedBrain(bc)
brain.eval()

# 640x480, NOT main()'s 800x600 default: mujoco.Renderer refuses a size
# larger than the model's offscreen framebuffer, humanoid_full.xml declares
# no <global offwidth>, and MuJoCo's default framebuffer is 640x480 — at
# 800x600 _init_mujoco fails CLOSED and the whole session silently runs
# worldless (measured in the first smoke: nu 0, construction_ok 0). Both
# dimensions are shipped main() flags (--width/--height), not a fork.
wc = WorldConfig(width=640, height=480,
                 scene_xml=SCENE_CATALOG[cfg["scene"]], save_dir=cfg["savedir"])
vw = VirtualWorld(brain, wc)

n_params = sum(p.numel() for p in brain.parameters())
construction_ok = int(vw.mj_model is not None and int(vw.mj_model.nu) > 0
                      and n_params > 0)

def _inject(mode):
    adim = int(bc.action_dim)
    if mode == "nan":
        fake = lambda *a, **kw: {"action": torch.full((1, adim), float("nan"))}
    else:
        def fake(*a, **kw):
            raise RuntimeError("T6.01 injected fault (control twin)")
    if getattr(vw, "task_manager", None) is not None:
        vw.task_manager.tick = fake
    brain.act_with_mood = fake

stats = {"frames": 0, "action_frames": 0, "nonfinite_actions": 0,
         "nonfinite_sim": 0, "eye_alive": 0, "eye_sampled": 0,
         "completed": 0}
period = 1.0 / float(wc.target_fps)
t_start = time.monotonic()
t_end = t_start + float(cfg["minutes"]) * 60.0
fault_pending = cfg["fault"] != "none"

while time.monotonic() < t_end:
    f0 = time.monotonic()
    if fault_pending and (f0 - t_start) >= float(cfg["fault_after_s"]):
        _inject(cfg["fault"])
        fault_pending = False
    now = f0
    vw._update_brain(now)
    vw._update_emotional(now)
    vw._update_autonomous(now)
    vw._step_physics()
    vw._auto_save_tick(now)
    vw.frame_count += 1

    a = vw.current_action
    if a is not None:
        stats["action_frames"] += 1
        if not np.all(np.isfinite(np.asarray(a, dtype=np.float64))):
            stats["nonfinite_actions"] += 1
    if vw.mj_data is not None:
        if not (np.all(np.isfinite(vw.mj_data.qpos))
                and np.all(np.isfinite(vw.mj_data.qvel))):
            stats["nonfinite_sim"] += 1
    if stats["frames"] % int(cfg["eye_every"]) == 0:
        stats["eye_sampled"] += 1
        img = vw._get_eye_image()
        if img is not None and float(img.std()) > 1e-6:
            stats["eye_alive"] += 1
    stats["frames"] += 1

    with open(cfg["heartbeat"], "w") as f:
        f.write(json.dumps({"frames": stats["frames"], "t": time.time()}))

    dt = time.monotonic() - f0
    if dt < period:
        time.sleep(period - dt)

stats["completed"] = 1
elapsed_s = time.monotonic() - t_start
saves = [p for p in os.listdir(cfg["savedir"])] if os.path.isdir(cfg["savedir"]) else []
out = dict(stats)
out.update({
    "minutes_survived": round(elapsed_s / 60.0, 4),
    "crash_frames": swallowed["crash"],
    "sensor_frames": swallowed["sensor"],
    "first_crash": swallowed["first_crash"],
    "first_sensor": swallowed["first_sensor"],
    "construction_ok": construction_ok,
    "nu": int(vw.mj_model.nu) if vw.mj_model is not None else 0,
    "n_params": int(n_params),
    "fps_measured": round(stats["frames"] / max(elapsed_s, 1e-9), 3),
    "autosave_files": len(saves),
    "maxrss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
})
with open(cfg["stats"], "w") as f:
    f.write(json.dumps(out))
'''


def _session(seed: int, minutes: float, fault: str) -> dict:
    """Run one child session under an external watchdog; return its metrics.

    hangs / hard_crash are judged HERE, by the parent, because the child
    cannot report its own hang and may not survive its own crash.
    """
    tmp = tempfile.mkdtemp(prefix=f"t601_{fault}_")
    hb = os.path.join(tmp, "heartbeat.json")
    st = os.path.join(tmp, "stats.json")
    sv = os.path.join(tmp, "saves")
    os.makedirs(sv, exist_ok=True)
    log = os.path.join(tmp, "child.log")
    cfg = {"repo": str(REPO), "seed": seed, "minutes": minutes,
           "fault": fault, "fault_after_s": FAULT_AFTER_S,
           "heartbeat": hb, "stats": st, "savedir": sv,
           "scene": SCENE, "eye_every": EYE_SAMPLE_EVERY}

    out = {"hangs": 0, "hard_crash": 0, "completed": 0, "frames": 0,
           "action_frames": 0, "nonfinite_actions": 0, "nonfinite_sim": 0,
           "crash_frames": 0, "sensor_frames": 0, "construction_ok": 0,
           "minutes_survived": 0.0, "eye_alive": 0, "eye_sampled": 0}

    with open(log, "w") as lf:
        proc = subprocess.Popen([PY, "-c", _CHILD, json.dumps(cfg)],
                                cwd=str(REPO), stdout=lf, stderr=lf,
                                start_new_session=True)
    wall_start = time.time()
    t_start = time.monotonic()
    first_beat_deadline = t_start + BUILD_GRACE_S
    killed_for_hang = False
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            now = time.monotonic()
            if os.path.exists(hb):
                stale = now - os.path.getmtime(hb)
                if stale > HANG_S:
                    killed_for_hang = True
            elif now > first_beat_deadline:
                killed_for_hang = True
            if killed_for_hang:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                rc = None
                break
            time.sleep(HEARTBEAT_POLL_S)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()

    if killed_for_hang:
        out["hangs"] = 1
    elif rc != 0:
        out["hard_crash"] = 1

    if os.path.exists(st):
        try:
            with open(st) as f:
                out.update(json.load(f))
        except Exception:
            out["hard_crash"] = 1  # a stats file it died writing
    elif os.path.exists(hb):
        # the child died mid-session: its last heartbeat bounds how far it got
        try:
            with open(hb) as f:
                beat = json.load(f)
            out["frames"] = int(beat.get("frames", 0))
            out["minutes_survived"] = round(
                max(0.0, os.path.getmtime(hb) - wall_start) / 60.0, 4)
        except Exception:
            pass

    # keep the child log tail for the ledger message surface
    try:
        with open(log) as f:
            tail = f.read()[-400:]
        out["log_tail"] = tail
    except Exception:
        pass
    return out


def _experiment(seed: int) -> dict:
    m = _session(seed, N_MINUTES, "none")
    return m


def _control(seed: int) -> dict:
    """The registry-declared must-be-caught twins: a session made to emit NaN
    actions and a session made to raise on every brain tick. Each must be
    SEEN by the same instruments that judge the claim."""
    a = _session(seed, CONTROL_MINUTES, "nan")
    b = _session(seed, CONTROL_MINUTES, "raise")
    return {
        "c_nan_detected": int(a.get("nonfinite_actions", 0) > 0),
        "c_raise_detected": int(b.get("crash_frames", 0) > 0),
        "c_nan_nonfinite_actions": a.get("nonfinite_actions", 0),
        "c_raise_crash_frames": b.get("crash_frames", 0),
        "c_nan_construction_ok": a.get("construction_ok", 0),
        "c_raise_construction_ok": b.get("construction_ok", 0),
        "c_nan_frames": a.get("frames", 0),
        "c_raise_frames": b.get("frames", 0),
    }


def _check(m: dict, c: dict):
    # ── rig lanes: a run that cannot see is VOID, never a verdict ───────────
    if m.get("construction_ok") != 1:
        return Status.VOID
    if m.get("frames", 0) < FRAMES_MIN:
        return Status.VOID
    if m.get("action_frames", 0) == 0 and m.get("crash_frames", 0) == 0:
        return Status.VOID            # the brain was never exercised at all
    if c.get("c_nan_detected") != 1 or c.get("c_raise_detected") != 1:
        return Status.VOID            # planted faults unseen: monitor is dead

    # ── the claim ────────────────────────────────────────────────────────────
    ok = (m.get("completed") == 1
          and m.get("hard_crash") == 0
          and m.get("hangs") == 0
          and m.get("crash_frames", 1) == 0
          and m.get("nonfinite_actions", 1) == 0
          and m.get("nonfinite_sim", 1) == 0)
    return Status.PASS if ok else Status.FAIL


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T6.01"], _experiment, _check,
                    control_fn=_control, ledger=ledger or Ledger())


if __name__ == "__main__":
    # Out-of-band pilots; NO ledger write on any path here.
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if mode == "smoke":
        minutes = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
        m = _session(seed, minutes, "none")
    elif mode == "nan":
        m = _session(seed, CONTROL_MINUTES, "nan")
    elif mode == "raise":
        m = _session(seed, CONTROL_MINUTES, "raise")
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    for k in sorted(m):
        if k != "log_tail":
            print(f"  {k}: {m[k]}")
    if m.get("log_tail"):
        print("  --- child log tail ---")
        print(m["log_tail"])
