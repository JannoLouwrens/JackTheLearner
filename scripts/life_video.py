"""life_video.py — a video of one life the owner can watch (ORDERS S6 step 1).

One random-action life in `jack_env` (seed 153), filmed OFFLINE: the life is
lived first with poses recorded each decision (qpos, 0.2 s apart), then
replayed into the same model with `mj_forward` only and rendered third-person
under Xvfb + llvmpipe (`experiments/render.py`). Rendering never touches the
world's random stream, so the life is the one `jack_env` would have lived.

Frames: one per decision, thinned evenly so the whole life fits in <= 60 s at
12 fps; 854x480, H.264 (libx264 from the static ffmpeg at /data/stt/bin), to
/data/jack-video/latest.mp4. Run by hand at nice 19; never inside a scored run.

Step 2: his needs drawn on every frame, and /data/jack-video/latest.json beside
the film (title, summary, facts, events, source) written by TEMPLATE from the
life's own record — no model. Today's world has three needs, energy (fed),
integrity (unhurt) and wetness; it has no thirst or temperature yet, so the
overlay shows the three he has and the summary says so.

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
CAUSE = {"energy": "hunger", "integrity": "his injuries"}
NEEDS = (("Fed", (90, 200, 90)), ("Unhurt", (220, 90, 80)), ("Wet", (80, 150, 230)))


def live(seed: int, cap: int):
    """Live one random-action life; per decision keep pose, needs and meals."""
    import numpy as np
    from experiments.jack_env import JackEnv
    env = JackEnv(seed=seed)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    def snap():
        st = env.w.drives.state
        return {"q": env.w.data.qpos.copy(), "mp": env.w.data.mocap_pos.copy(),
                "h": (float(st.e), float(st.i), float(st.w)),
                "ate": {k: int(v) for k, v in env.w.drives.ate_total.items()}}
    rec = [snap()]
    info, died = {}, False
    for _ in range(cap):
        a = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
        _, _, died, trunc, info = env.step(a)
        if died:      # the world has already respawned him; the last record kept is the one before
            break
        rec.append(snap())
    return env, rec, info, died


def clock(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def events_of(rec, step: int, died: bool, cause: str) -> list:
    """Moments from the record: each meal, each hurt, getting wet, hunger
    thresholds, the end. `t` = seconds into the film, `at` = time in his life."""
    ev = []

    def add(k, label, keep=False):
        ev.append({"t": round((k // step) / FPS, 1), "at": clock(0.2 * k), "label": label, "_keep": keep})
    warned = set()
    for k in range(1, len(rec)):
        (e0, i0, w0), (e1, i1, w1) = rec[k - 1]["h"], rec[k]["h"]
        for food, n in rec[k]["ate"].items():
            if n > rec[k - 1]["ate"].get(food, 0):
                add(k, "Ate", keep=True)
        if i1 < i0 - 0.02:
            add(k, f"Hurt: unhurt {i0:.0%} to {i1:.0%}")
        if w0 < 0.2 <= w1:
            add(k, "Got wet")
        for lvl, word in ((0.5, "Getting hungry"), (0.25, "Very hungry"), (0.1, "Starving")):
            if e0 >= lvl > e1 and lvl not in warned:
                warned.add(lvl)
                add(k, f"{word} (fed {e1:.0%})")
    if died:
        add(len(rec) - 1, f"Died of {CAUSE.get(cause, cause or 'unknown')}", keep=True)
    if len(ev) > 60:                       # keep every meal and the end; thin the rest evenly
        must = [x for x in ev if x["_keep"]][:60]
        rest = [x for x in ev if not x["_keep"]]
        room = 60 - len(must)
        pick = rest[:: max(1, math.ceil(len(rest) / room))][:room] if room > 0 else []
        ev = sorted(must + pick, key=lambda x: x["t"])
    for x in ev:
        x.pop("_keep")
    return ev


def overlay(img, h, life_s: float, banner: str, font, small):
    """Needs as three bars, his age in the life, and the latest moment."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    dr.rounded_rectangle((10, 10, 250, 112), 8, fill=(0, 0, 0, 140))
    for j, ((name, col), v) in enumerate(zip(NEEDS, h)):
        y = 20 + 30 * j
        dr.text((20, y), name, font=small, fill=(255, 255, 255))
        dr.rectangle((92, y + 3, 240, y + 19), outline=(255, 255, 255, 160))
        dr.rectangle((93, y + 4, 93 + int(146 * max(0.0, min(1.0, v))), y + 18), fill=col + (230,))
    dr.rounded_rectangle((W - 150, 10, W - 10, 46), 8, fill=(0, 0, 0, 140))
    dr.text((W - 140, 16), f"age {clock(life_s)}", font=font, fill=(255, 255, 255))
    if banner:
        tw = dr.textlength(banner, font=font)
        dr.rounded_rectangle(((W - tw) / 2 - 12, H - 56, (W + tw) / 2 + 12, H - 16), 8, fill=(0, 0, 0, 160))
        dr.text(((W - tw) / 2, H - 50), banner, font=font, fill=(255, 255, 255))
    return im.tobytes()


def describe(seed, rec, died, cause, events, video_s, commit):
    life_s = 0.2 * (len(rec) - 1)
    meals = sum(rec[-1]["ate"].values()) - sum(rec[0]["ate"].values())
    hurts = sum(1 for x in events if x["label"].startswith("Hurt"))
    e_min = min(r["h"][0] for r in rec)
    i_min = min(r["h"][1] for r in rec)
    w_max = max(r["h"][2] for r in rec)
    end = f"died of {CAUSE.get(cause, cause)}" if died else "was still alive when the film stopped"
    title = f"A life of {clock(life_s)}: he {end}"
    s = [f"He lived {clock(life_s)} (minutes:seconds) and {end}.",
         f"He ate {meals} time{'s' if meals != 1 else ''}" + (f" and was hurt {hurts} time{'s' if hurts != 1 else ''}." if hurts else " and was never hurt."),
         "Every move was random: this film shows the world and his body, not anything he has learned.",
         "His world has three needs today (fed, unhurt, dry); thirst and warmth are not in it yet."]
    facts = [{"label": "lived", "value": clock(life_s)},
             {"label": "how it ended", "value": CAUSE.get(cause, cause) if died else "alive"},
             {"label": "meals", "value": str(meals)},
             {"label": "times hurt", "value": str(hurts)},
             {"label": "lowest fed", "value": f"{e_min:.0%}"},
             {"label": "lowest unhurt", "value": f"{i_min:.0%}"},
             {"label": "wettest", "value": f"{w_max:.0%}"},
             {"label": "film length", "value": clock(video_s)}]
    return {"title": title, "summary": " ".join(s), "facts": facts, "events": events,
            "source": f"random-action life 1 of jack_env seed {seed} · commit {commit}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=153)
    ap.add_argument("--cap", type=int, default=3000, help="max decisions (0.2 s each)")
    ap.add_argument("--out", default="/data/jack-video/latest.mp4")
    a = ap.parse_args()
    ensure_gl(W, H)
    import mujoco
    from PIL import ImageFont
    font, small = ImageFont.load_default(size=22), ImageFont.load_default(size=18)
    t0 = time.time()
    env, rec, info, died = live(a.seed, a.cap)
    cause = info.get("death_cause", "") if died else ""
    m, d = env.w.model, mujoco.MjData(env.w.model)
    m.vis.global_.offwidth, m.vis.global_.offheight = max(W, m.vis.global_.offwidth), max(H, m.vis.global_.offheight)
    step = max(1, math.ceil(len(rec) / (FPS * MAX_S)))
    events = events_of(rec, step, died, cause)
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
    ks = list(range(0, len(rec), step))
    for i, k in enumerate(ks):
        d.qpos[:] = rec[k]["q"]
        d.mocap_pos[:] = rec[k]["mp"]
        mujoco.mj_forward(m, d)
        cam.lookat[:] = d.xpos[bid]
        cam.azimuth = 90.0 + 0.15 * i      # a slow orbit so the room reads as 3-D
        r.update_scene(d, camera=cam, scene_option=opt)
        t = i / FPS
        banner = next((x["label"] for x in reversed(events) if x["t"] <= t < x["t"] + 2.0), "")
        ff.stdin.write(overlay(r.render(), rec[k]["h"], 0.2 * k, banner, font, small))
    ff.stdin.close()
    if ff.wait() != 0:
        raise SystemExit("ffmpeg failed")
    video_s = round(len(ks) / FPS, 1)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    side = describe(a.seed, rec, died, cause, events, video_s, commit)
    tmp.replace(out)
    jtmp = out.with_suffix(".json.part")
    jtmp.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    jtmp.replace(out.with_suffix(".json"))
    print(json.dumps({"seed": a.seed, "decisions": len(rec) - 1, "life_s": round(0.2 * (len(rec) - 1), 1),
                      "died": bool(died), "death_cause": cause, "frames": len(ks), "events": len(events),
                      "video_s": video_s, "bytes": out.stat().st_size,
                      "wall_s": round(time.time() - t0, 1), "out": str(out), "title": side["title"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
