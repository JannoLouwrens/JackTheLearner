"""SO.01 — Jack can be watched: a third-person stream exists and costs what we say it costs.

Serves GOAL.md directly: "I want to watch him figure out the world himself"
(owner, 2026-08-09). This is the FIXTURE half of the spectating commitment —
it does not claim being watched leaves him unchanged (that is SO.04, blocked
behind this); it claims the watching is PHYSICALLY AVAILABLE on this box at a
measured cost, which no spec had ever demonstrated: of 233 specs, none had
produced a frame for a spectator.

THE CLAIM (registry, thresholds are the registry's and do not move):
  - a third-person view of a running life renders at >= 5 fps at 320x240;
  - it is deliverable to the owner WITHOUT a persistent listening service —
    here delivery is a plain file on disk (scp-able), because a standing
    stream server on a tenant-serving box is an OWNER decision the spec's
    notes explicitly refuse to make;
  - the render cost is reported as a fraction of the life's compute budget,
    and FALSIFIED BY that cost pushing the real-time factor below 1.0
    (watching would then be slower than living).

THE LIFE. 10 simulated seconds (2000 steps at dt=0.005) of the Humanoid-v5
body in the playground, driven by PG.8's random-drive idiom (ctrl resampled
every 10 steps in [-0.4, 0.4]). This is the cheap end of a life's envelope
(no water, no learner) and that is the honest direction for THIS claim: the
render cost share can only look LARGER against a cheap life, and the fps
gate does not depend on the life's cost at all.

THE CADENCE IS THE COUPLING. Frames are rendered every 0.2 SIM-seconds, so a
life running at exactly real time would stream at exactly 5 fps. That makes
the two registry gates independent measurements: `render_fps` asks whether
the renderer alone can make frames fast enough at 320x240, and `rtf_stream`
asks whether the life, WITH its stream, still outruns the wall clock.

RIG SIZED BY MEASUREMENT, 2026-09-03, this box (thresholds untouched):
    sim only          rtf 18.6            (0.54 s wall for 10 sim-s)
    render 320x240    75.8 ms/frame       -> 13.2 fps raw
    projected stream  0.54 + 50 x 0.076 s -> rtf ~2.3, delivered ~11.5 fps
    humanoid diff     4.50% of pixels     (identical scenes diff 0.0%)
    repeat render     byte-identical      (mean abs diff 0.0)
So every gate sits with real margin on both sides: a 200 ms/frame renderer
would fail rtf, a camera that cannot see him would fail the control.

THE CONTROL (registry): render the same scene with the humanoid REMOVED —
the frame must measurably change. A renderer whose frames are identical with
and without the subject is producing a background, and a "stream" of
background would pass every fps gate while showing the owner nothing.
DIFF_FRAC_GATE = 0.005 is pre-registered into the measured gap (0.045 with
him vs 0.000 without).

THE TWO RENDER TRAPS (render.py's, binding here): renderers are held in a
module-level list for the process lifetime — a garbage-collected
mujoco.Renderer poisons the shared X display and the NEXT renderer returns
corrupted-but-realistic frames. And a GL context can come up rendering a
uniform frame, which looks exactly like a blind sensor: every stream frame
must show >= 3 distinct colours, and a fixed t=0 canary is re-rendered after
the stream and must match its pre-stream self (measured byte-identical on
this box). Either instrument-death is Status.VOID, never FAIL — a blind
renderer refutes nothing.

WHAT THIS CAMERA IS NOT. The spectator camera (lookat at the spawn point,
3.5 m out, azimuth 135, elevation -15) is THIS FIXTURE's, not the world's:
EYE_POS/EYE_XYAXES/EYE_FOVY — the world contract PG.6 certified — are not
read, not moved, and no visual certificate depends on this pose.
"""

from __future__ import annotations

import os
import time

import numpy as np

from ..render import ensure_gl

ensure_gl()

import mujoco  # noqa: E402  (must follow ensure_gl)

import playground as pg  # noqa: E402

from ..protocol import Ledger, Status, run_spec  # noqa: E402
from ..registry import BY_ID  # noqa: E402

# The world (spawn point, bodies) and the GL path are what this verdict is
# about; a change to either must read as drift.
IMPL_DEPS = ["playground.py", "experiments/render.py"]

# --- registry thresholds (do not move) --------------------------------------
RES_W, RES_H = 320, 240
FPS_GATE = 5.0            # frames per wall second, both raw and delivered
RTF_GATE = 1.0            # sim-seconds per wall-second, WITH the stream

# --- envelope (sized by the pilot above, not thresholds) --------------------
SIM_SECONDS = 10.0
FRAME_EVERY_SIM_S = 0.2   # 5 fps at exactly real time — couples the gates
CTRL_RESAMPLE_STEPS = 10
CTRL_RANGE = 0.4          # PG.8's random-drive amplitude

# --- pre-registered instrument constants ------------------------------------
DIFF_LEVELS = 10          # per-pixel change (0-255) that counts as "changed"
DIFF_FRAC_GATE = 0.005    # in the gap: humanoid 0.045, background 0.000
FRAME_MIN_COLOURS = 3     # render.selftest's aliveness bar
CANARY_MAX_MEAN_DIFF = 0.5  # repeat render measured 0.0; poison is >> this

# Spectator pose — this fixture's, NOT the world contract.
CAM_LOOKAT_Z = 0.8
CAM_DISTANCE = 3.5
CAM_AZIMUTH = 135.0
CAM_ELEVATION = -15.0

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

# A garbage-collected Renderer poisons the shared display (render.py's trap
# #1); everything constructed here lives until the process dies.
_RENDERERS: list = []


def _camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    sp = pg.PlaygroundParams().spawn()   # deterministic, seed-independent
    cam.lookat[:] = [sp[0], sp[1], CAM_LOOKAT_Z]
    cam.distance = CAM_DISTANCE
    cam.azimuth = CAM_AZIMUTH
    cam.elevation = CAM_ELEVATION
    return cam


def _renderer(model) -> mujoco.Renderer:
    r = mujoco.Renderer(model, height=RES_H, width=RES_W)
    _RENDERERS.append(r)
    return r


def _frame(r, data, cam) -> np.ndarray:
    r.update_scene(data, camera=cam)
    return r.render()


def _alive(img: np.ndarray) -> bool:
    return len(np.unique(img.reshape(-1, 3), axis=0)) >= FRAME_MIN_COLOURS


def _drive(rng, model, data, i):
    if i % CTRL_RESAMPLE_STEPS == 0:
        data.ctrl[:] = rng.uniform(-CTRL_RANGE, CTRL_RANGE, model.nu)
    mujoco.mj_step(model, data)


def _experiment(seed: int) -> dict:
    p = pg.PlaygroundParams(seed=seed)
    model, _, _ = pg.make_playground(p, with_water=False, with_humanoid=True)
    dt = float(model.opt.timestep)
    n_steps = int(round(SIM_SECONDS / dt))
    frame_every = int(round(FRAME_EVERY_SIM_S / dt))
    cam = _camera()
    r = _renderer(model)

    # Canary: a state frozen at t=0, rendered before and after the stream.
    # If the display gets poisoned mid-run, this frame moves (trap #2).
    data_c = mujoco.MjData(model)
    mujoco.mj_forward(model, data_c)
    canary_pre = _frame(r, data_c, cam).astype(np.int16)

    # Arm A — the life alone. Same ctrl schedule as the stream arm.
    data_sim = mujoco.MjData(model)
    rng = np.random.RandomState(seed)
    t0 = time.perf_counter()
    for i in range(n_steps):
        _drive(rng, model, data_sim, i)
    sim_wall = time.perf_counter() - t0

    # Arm B — the same life, watched. Wall time runs until the artifact is a
    # file on disk, because "deliverable" is part of the claim.
    data_st = mujoco.MjData(model)
    rng = np.random.RandomState(seed)
    frames = []
    render_s = 0.0
    t0 = time.perf_counter()
    for i in range(n_steps):
        _drive(rng, model, data_st, i)
        if (i + 1) % frame_every == 0:
            tr = time.perf_counter()
            frames.append(_frame(r, data_st, cam).copy())
            render_s += time.perf_counter() - tr
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    path = os.path.join(ARTIFACT_DIR, f"so01_stream_seed{seed}.npy")
    np.save(path, np.stack(frames))
    stream_wall = time.perf_counter() - t0

    canary_post = _frame(r, data_c, cam).astype(np.int16)
    canary_diff = float(np.abs(canary_pre - canary_post).mean())

    n_frames = len(frames)
    delivered_fps = n_frames / stream_wall
    render_fps = n_frames / render_s if render_s > 0 else 0.0
    rtf_stream = SIM_SECONDS / stream_wall
    artifact_mb = os.path.getsize(path) / 1e6

    m = {
        "n_frames": float(n_frames),
        "sim_wall_s": round(sim_wall, 3),
        "stream_wall_s": round(stream_wall, 3),
        "rtf_sim": round(SIM_SECONDS / sim_wall, 2),
        "rtf_stream": round(rtf_stream, 2),
        "delivered_fps": round(delivered_fps, 2),
        "render_fps": round(render_fps, 2),
        "render_ms_per_frame": round(1000.0 * render_s / max(n_frames, 1), 1),
        # The reported number the hypothesis names: what fraction of the
        # watched life's compute the watching bought.
        "cost_share": round(max(0.0, (stream_wall - sim_wall) / stream_wall), 4),
        "artifact_mb": round(artifact_mb, 2),
        "artifact_ok": float(artifact_mb > 0),
        "frames_alive": float(all(_alive(f) for f in frames)),
        "canary_ok": float(canary_diff <= CANARY_MAX_MEAN_DIFF),
        "canary_mean_diff": round(canary_diff, 3),
    }
    m["seed_gates_ok"] = float(
        delivered_fps >= FPS_GATE
        and render_fps >= FPS_GATE
        and rtf_stream >= RTF_GATE
        and m["artifact_ok"] == 1.0)
    return m


def _control(seed: int) -> dict:
    """Remove the subject; the frame must change or the stream is scenery."""
    cam = _camera()
    imgs = {}
    for withh in (True, False):
        model, _, _ = pg.make_playground(
            pg.PlaygroundParams(seed=seed), with_water=False,
            with_humanoid=withh)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        imgs[withh] = _frame(_renderer(model), data, cam).astype(np.int16)
    diff_frac = float(
        (np.abs(imgs[True] - imgs[False]).max(axis=2) > DIFF_LEVELS).mean())
    return {
        "c_diff_frac": round(diff_frac, 4),
        "c_alive_both": float(_alive(imgs[True].astype(np.uint8))
                              and _alive(imgs[False].astype(np.uint8))),
        "c_humanoid_changes_frame": float(diff_frac >= DIFF_FRAC_GATE),
    }


def _check(m: dict, c: dict):
    try:
        alive = (m["frames_alive"], m["canary_ok"], c["c_alive_both"])
    except KeyError:
        return Status.VOID
    # Instrument-dead => VOID, never FAIL: a uniform or poisoned frame is a
    # broken eye, not evidence about the stream's cost.
    if min(alive) < 1.0:
        return Status.VOID
    return (m["seed_gates_ok"] == 1.0
            and m["delivered_fps"] >= FPS_GATE
            and m["render_fps"] >= FPS_GATE
            and m["rtf_stream"] >= RTF_GATE
            and c["c_humanoid_changes_frame"] == 1.0
            and c["c_diff_frac"] >= DIFF_FRAC_GATE)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["SO.01"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
