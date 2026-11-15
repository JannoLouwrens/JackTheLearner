"""life_video.py — a video of one life the owner can watch (ORDERS S6 step 1).

One random-action life in `jack_env` (seed 153), filmed OFFLINE: the life is
lived first with poses recorded each decision (qpos, 0.2 s apart), then
replayed into the same model with `mj_forward` only and rendered third-person
under Xvfb + llvmpipe (`experiments/render.py`). Rendering never touches the
world's random stream, so the life is the one `jack_env` would have lived.

Frames: one per decision, thinned evenly so the whole life fits in <= 60 s at
12 fps; 854x480, H.264 (libx264 from the static ffmpeg at /data/stt/bin), to
/data/jack-video/latest.mp4. Run by hand at nice 19; never inside a scored run.

    nice -n 19 /data/venvs/jackthelearner/bin/python scripts/life_video.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from experiments.render import ensure_gl  # noqa: E402

W, H, FPS, MAX_S = 854, 480, 12, 60
FFMPEG = os.environ.get("JACK_FFMPEG", "/data/stt/bin/ffmpeg")


def live(seed: int, cap: int):
    import numpy as np
    from experiments.jack_env import JackEnv
    env = JackEnv(seed=seed)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    snap = lambda: (env.w.data.qpos.copy(), env.w.data.mocap_pos.copy())
    poses = [snap()]
    info, died = {}, False
    for _ in range(cap):
        a = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
        _, _, died, trunc, info = env.step(a)
        if died:      # the world has already respawned him; the last pose kept is the one before
            break
        poses.append(snap())
    return env, poses, info, died


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=153)
    ap.add_argument("--cap", type=int, default=3000, help="max decisions (0.2 s each)")
    ap.add_argument("--out", default="/data/jack-video/latest.mp4")
    a = ap.parse_args()
    ensure_gl(W, H)
    import mujoco
    t0 = time.time()
    env, poses, info, died = live(a.seed, a.cap)
    m, d = env.w.model, mujoco.MjData(env.w.model)
    m.vis.global_.offwidth, m.vis.global_.offheight = max(W, m.vis.global_.offwidth), max(H, m.vis.global_.offheight)
    step = max(1, math.ceil(len(poses) / (FPS * MAX_S)))
    frames = poses[::step]
    r = mujoco.Renderer(m, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation = 2.5, -30.0
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 1           # his body is geom group 3, hidden by default
    bid = env.w.rover_bid
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.mp4")
    ff = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                           "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart", str(tmp)], stdin=subprocess.PIPE)
    for i, (q, mp) in enumerate(frames):
        d.qpos[:] = q
        d.mocap_pos[:] = mp
        mujoco.mj_forward(m, d)
        cam.lookat[:] = d.xpos[bid]
        cam.azimuth = 90.0 + 0.15 * i      # a slow orbit so the room reads as 3-D
        r.update_scene(d, camera=cam, scene_option=opt)
        ff.stdin.write(r.render().tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise SystemExit("ffmpeg failed")
    tmp.replace(out)
    rec = {"seed": a.seed, "decisions": len(poses) - 1, "life_s": round(0.2 * (len(poses) - 1), 1),
           "died": bool(died), "death_cause": info.get("death_cause", ""),
           "meals": int(info.get("meals_this_life", 0) or 0), "frames": len(frames),
           "video_s": round(len(frames) / FPS, 1), "bytes": out.stat().st_size,
           "wall_s": round(time.time() - t0, 1), "out": str(out)}
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
