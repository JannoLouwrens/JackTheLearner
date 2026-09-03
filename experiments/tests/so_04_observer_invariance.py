"""SO.04 — Being watched does not change him.

The spectating commitment's CLAIM half. SO.01 proved the watching is
physically available at a measured cost; this spec proves the watching is
INERT: the same life at the same seed, run once unwatched and once with the
SO.01 spectator stream rendering beside it, must be THE SAME LIFE — state
bit-for-bit identical at every control step, and the policy RNG stream
untouched by the render path. "Kills: any claim measured while the owner was
watching, which under this direction will eventually be most of them."

THE CLAIM (registry): behaviour statistics indistinguishable between a
rendered and an unrendered run at the same seed, and the rendered trajectory
matching the unrendered one bit-for-bit until the first stochastic draw. In
THIS composition the render path performs no stochastic draw at all, so the
claim is taken at its strongest available reading (strengthen-only): the
FULL trajectory must match bit-for-bit — every step's (qpos, qvel, ctrl)
digest equal — and the RNG terminal state must be identical. Behaviour
statistics (final displacement, mean height, mean speed) are reported and
must differ by exactly 0.0; under bit-identity they cannot differ, so the
stats are a redundant witness, not an independent bar.

THE DETECTOR. Per-step blake2b digests of (qpos, qvel, ctrl) bytes — ctrl
included so a diverged resample is caught AT the resample step, not one
physics step later — compared across arms; `first_divergence_step` is the
first index at which they differ, -1 for none. The RNG terminal state
(MT19937 key vector + position) is digested the same way, so an extra draw
that happens to leave the visible trajectory unchanged inside the horizon
still cannot hide.

THE CONTROL (registry, must be caught): deliberately draw ONE value from the
policy RNG inside the render branch. The detector MUST flag it — divergence
detected, located at or after the first render step (the sabotage lives in
the render path; an earlier divergence would be a different bug), and the
RNG digests must mismatch. "A detector that cannot see its own positive
control has measured nothing" (LESSONS). One extra draw shifts every
subsequent ctrl resample, so the first divergent digest is the resample step
immediately after the first frame — measured at exactly step 40 in the
smoke run, = FRAME_EVERY steps, as predicted.

VOID LANES, never FAIL (a blind eye refutes nothing): the watched arm must
actually watch — all EXPECTED_FRAMES frames rendered, every frame alive
(>= 3 distinct colours), and SO.01's t=0 canary byte-stable pre/post stream.
Same lanes on the control's watched arm. Without these, an implementation
whose renderer silently never fired would "pass" the invariance vacuously.

SMOKE RUN (2026-09-03, this box, seed 0, before the registered run): watched
vs unwatched bit_identical over all 2000 steps, rng_match 1, stats_max_diff
0.0; sabotaged arm first_divergence_step 40 == FRAME_EVERY, rng mismatch
caught. Nothing was sized from this — every gate is an exact invariant.

THE LIFE is SO.01's fixture verbatim: 10 sim-s (2000 steps at dt 0.005) of
the playground humanoid, PG.8 random drive (resample every 10 steps in
[-0.4, 0.4]), frames every 0.2 sim-s at 320x240 via SO.01's free spectator
camera — which is this fixture's pose, NOT the world contract (EYE_* is
neither read nor moved). Renderers are held for the process lifetime
(render.py trap #1).
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np

from ..render import ensure_gl

ensure_gl()

import mujoco  # noqa: E402  (must follow ensure_gl)

import playground as pg  # noqa: E402

from ..protocol import Ledger, Status, run_spec  # noqa: E402
from ..registry import BY_ID  # noqa: E402

# The world (spawn, bodies) and the GL path are what this verdict composes
# over; a change to either must read as drift.
IMPL_DEPS = ["playground.py", "experiments/render.py"]

# --- the fixture, SO.01's verbatim ------------------------------------------
RES_W, RES_H = 320, 240
SIM_SECONDS = 10.0
FRAME_EVERY_SIM_S = 0.2
CTRL_RESAMPLE_STEPS = 10
CTRL_RANGE = 0.4

CAM_LOOKAT_Z = 0.8
CAM_DISTANCE = 3.5
CAM_AZIMUTH = 135.0
CAM_ELEVATION = -15.0

# --- pre-registered instrument constants ------------------------------------
FRAME_MIN_COLOURS = 3        # render.selftest's aliveness bar (SO.01)
CANARY_MAX_MEAN_DIFF = 0.5   # repeat render measured 0.0; poison is >> this
DIGEST_BYTES = 16

_RENDERERS: list = []        # held for the process lifetime (render.py trap #1)


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


def _state_digest(data) -> bytes:
    h = hashlib.blake2b(digest_size=DIGEST_BYTES)
    h.update(data.qpos.tobytes())
    h.update(data.qvel.tobytes())
    h.update(data.ctrl.tobytes())
    return h.digest()


def _rng_digest(rng: np.random.RandomState) -> bytes:
    kind, keys, pos, has_gauss, cached = rng.get_state()
    h = hashlib.blake2b(digest_size=DIGEST_BYTES)
    h.update(keys.tobytes())
    h.update(struct.pack("<iid", pos, has_gauss, cached))
    return h.digest()


def _life(seed: int, watch: bool, sabotage: bool = False) -> dict:
    """One life. `watch` adds the SO.01 spectator stream; `sabotage` performs
    the registry control's single deliberate RNG draw inside the render branch.
    Returns per-step digests, the RNG terminal digest, behaviour statistics,
    and (watched only) the stream-aliveness evidence."""
    p = pg.PlaygroundParams(seed=seed)
    model, _, _ = pg.make_playground(p, with_water=False, with_humanoid=True)
    dt = float(model.opt.timestep)
    n_steps = int(round(SIM_SECONDS / dt))
    frame_every = int(round(FRAME_EVERY_SIM_S / dt))

    cam = _camera() if watch else None
    r = _renderer(model) if watch else None
    canary_diff = 0.0
    if watch:
        data_c = mujoco.MjData(model)
        mujoco.mj_forward(model, data_c)
        canary_pre = _frame(r, data_c, cam).astype(np.int16)

    data = mujoco.MjData(model)
    rng = np.random.RandomState(seed)
    digests: list = []
    n_frames = 0
    frames_alive = True
    heights = []
    speeds = []
    for i in range(n_steps):
        if i % CTRL_RESAMPLE_STEPS == 0:
            data.ctrl[:] = rng.uniform(-CTRL_RANGE, CTRL_RANGE, model.nu)
        mujoco.mj_step(model, data)
        digests.append(_state_digest(data))
        heights.append(float(data.qpos[2]))
        speeds.append(float(np.linalg.norm(data.qvel[:3])))
        if watch and (i + 1) % frame_every == 0:
            if sabotage:
                rng.uniform()   # THE deliberate draw in the render path
            f = _frame(r, data, cam)
            n_frames += 1
            frames_alive = frames_alive and _alive(f)

    if watch:
        canary_post = _frame(r, data_c, cam).astype(np.int16)
        canary_diff = float(np.abs(canary_pre - canary_post).mean())

    return {
        "digests": digests,
        "rng_digest": _rng_digest(rng),
        "final_xy": (float(data.qpos[0]), float(data.qpos[1])),
        "mean_height": float(np.mean(heights)),
        "mean_speed": float(np.mean(speeds)),
        "n_frames": n_frames,
        "frames_alive": frames_alive,
        "canary_diff": canary_diff,
        "frame_every": frame_every,
        "n_steps": n_steps,
    }


def _first_divergence(a: list, b: list) -> int:
    for i, (da, db) in enumerate(zip(a, b)):
        if da != db:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


def _stats_max_diff(a: dict, b: dict) -> float:
    diffs = [abs(a["final_xy"][0] - b["final_xy"][0]),
             abs(a["final_xy"][1] - b["final_xy"][1]),
             abs(a["mean_height"] - b["mean_height"]),
             abs(a["mean_speed"] - b["mean_speed"])]
    return float(max(diffs))


def _experiment(seed: int) -> dict:
    unwatched = _life(seed, watch=False)
    watched = _life(seed, watch=True)

    first_div = _first_divergence(unwatched["digests"], watched["digests"])
    bit_identical = first_div == -1
    rng_match = unwatched["rng_digest"] == watched["rng_digest"]
    expected_frames = watched["n_steps"] // watched["frame_every"]

    m = {
        "bit_identical": float(bit_identical),
        "first_divergence_step": float(first_div),
        "rng_match": float(rng_match),
        "stats_max_diff": _stats_max_diff(unwatched, watched),
        "n_frames": float(watched["n_frames"]),
        "n_frames_ok": float(watched["n_frames"] == expected_frames),
        "frames_alive": float(watched["frames_alive"]),
        "canary_ok": float(watched["canary_diff"] <= CANARY_MAX_MEAN_DIFF),
        "canary_mean_diff": round(watched["canary_diff"], 3),
    }
    m["seed_gates_ok"] = float(
        bit_identical and rng_match and m["stats_max_diff"] == 0.0
        and m["n_frames_ok"] == 1.0)
    return m


def _control(seed: int) -> dict:
    """The registry's positive control: one deliberate RNG draw in the render
    path. The detector must catch it — located in the render path — or it
    could not have seen a real observer perturbation either."""
    unwatched = _life(seed, watch=False)
    sabotaged = _life(seed, watch=True, sabotage=True)

    first_div = _first_divergence(unwatched["digests"], sabotaged["digests"])
    detected = first_div != -1
    # The sabotage draw happens at the first render (step index frame_every-1,
    # after that step's digest); the earliest legal divergence is the next
    # ctrl resample, at index frame_every. Earlier = a fault OUTSIDE the
    # render path, which this control cannot claim to have caused.
    in_render_path = detected and first_div >= sabotaged["frame_every"]
    rng_mismatch = unwatched["rng_digest"] != sabotaged["rng_digest"]
    expected_frames = sabotaged["n_steps"] // sabotaged["frame_every"]

    return {
        "c_first_divergence_step": float(first_div),
        "c_detected": float(detected),
        "c_in_render_path": float(in_render_path),
        "c_rng_mismatch": float(rng_mismatch),
        "c_n_frames_ok": float(sabotaged["n_frames"] == expected_frames),
        "c_frames_alive": float(sabotaged["frames_alive"]),
        "c_canary_ok": float(sabotaged["canary_diff"] <= CANARY_MAX_MEAN_DIFF),
        "c_detector_catches": float(detected and in_render_path
                                    and rng_mismatch),
    }


def _check(m: dict, c: dict):
    try:
        alive = (m["frames_alive"], m["canary_ok"], m["n_frames_ok"],
                 c["c_frames_alive"], c["c_canary_ok"], c["c_n_frames_ok"])
    except KeyError:
        return Status.VOID
    # Instrument-dead => VOID, never FAIL: a stream that never rendered, a
    # blind frame or a poisoned display makes the invariance vacuous, not true.
    if min(alive) < 1.0:
        return Status.VOID
    return (m["seed_gates_ok"] == 1.0
            and m["bit_identical"] == 1.0
            and m["rng_match"] == 1.0
            and m["stats_max_diff"] == 0.0
            and c["c_detector_catches"] == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["SO.04"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
