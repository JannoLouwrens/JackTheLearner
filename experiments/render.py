"""Offscreen MuJoCo rendering on this box.

WHY THIS FILE EXISTS. On 2026-08-09 the builder tried to implement PG.6 (the
playground's eyes), found no `libEGL` and no `libOSMesa`, concluded "MuJoCo
fails to render at import", and escalated PG.6 as blocked by the machine. PG.6
was at that moment the largest unblocked lever in the project — `run blocked`
scored it as freeing five specs and blocking seven, including the whole unison
ladder's visual half.

It was never blocked. MuJoCo has THREE backends and only two were tried. This
box has `libGL.so.1`, `libGLX_mesa.so.0` (llvmpipe, software rasteriser),
`mesa-dri-drivers` and `Xvfb` already installed, because WorldTwin renders
headless WebGL globes here. GLX under a virtual display works, needs nothing
installed, and costs ~12 ms/frame at 64x64 — 1000 frames in twelve seconds,
which is far cheaper than the physics that generates them.

The general lesson (docs/LESSONS.md): "the box cannot do X" is a claim about
every path to X, and it is usually made after testing one. Before escalating a
capability as missing, enumerate the ways the capability is normally obtained
and say which ones you tried.

USAGE — call before importing mujoco, because mujoco resolves its GL backend
when the module is first imported:

    from experiments.render import ensure_gl
    ensure_gl()
    import mujoco

`ensure_gl()` is idempotent, safe under concurrent specs (each gets its own
display), and kills its Xvfb at process exit. It raises rather than returning
a broken context: a spec that silently renders black frames would report a
vision failure that is really a setup failure, and that lie is expensive —
PG.6's whole job is to tell us whether vision resolves radius and bearing.
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import time

_STARTED: subprocess.Popen | None = None
_DISPLAY: str | None = None


def _display_is_live(disp: str) -> bool:
    return os.path.exists("/tmp/.X11-unix/X" + disp.lstrip(":").split(".")[0])


def _start_xvfb(width: int, height: int) -> str:
    """Claim a free display number and hold an Xvfb on it."""
    global _STARTED
    last_err = ""
    for n in range(99, 130):
        disp = f":{n}"
        if _display_is_live(disp):
            continue  # another spec owns it; do not share, do not kill
        proc = subprocess.Popen(
            ["Xvfb", disp, "-screen", "0", f"{width}x{height}x24", "-nolisten", "tcp"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        # Xvfb exits ~immediately if the display is taken; wait for the socket.
        for _ in range(80):
            if proc.poll() is not None:
                last_err = (proc.stderr.read() or b"").decode()[-200:]
                break
            if _display_is_live(disp):
                _STARTED = proc
                atexit.register(_stop_xvfb)
                return disp
            time.sleep(0.05)
    raise RuntimeError(
        "could not start Xvfb on any display :99-:129. "
        f"last error: {last_err or 'timed out waiting for the socket'}"
    )


def _stop_xvfb() -> None:
    global _STARTED
    if _STARTED is not None and _STARTED.poll() is None:
        _STARTED.terminate()
        try:
            _STARTED.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _STARTED.kill()
    _STARTED = None


def ensure_gl(width: int = 640, height: int = 480) -> str:
    """Guarantee MuJoCo can create an offscreen GL context. Returns the DISPLAY.

    Prefers an existing usable DISPLAY (so `xvfb-run python ...` keeps working
    and does not get a second, pointless server); otherwise starts its own.
    """
    global _DISPLAY
    if _DISPLAY is not None:
        return _DISPLAY

    disp = os.environ.get("DISPLAY", "")
    if not (disp and _display_is_live(disp)):
        disp = _start_xvfb(width, height)
    os.environ["DISPLAY"] = disp

    # Only force glx if the caller has not chosen a backend. Someone running
    # this on a real GPU box should get egl without editing code.
    if not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = "glx"

    if "mujoco" in sys.modules:
        # Not fatal on every mujoco version, but it is on some, and a silent
        # black-frame render is worse than a loud error.
        raise RuntimeError(
            "ensure_gl() must be called BEFORE `import mujoco` — mujoco binds "
            "its GL backend at import. Move the ensure_gl() call above the "
            "mujoco import in your spec."
        )

    _DISPLAY = disp
    return disp


def selftest() -> dict:
    """Render one frame and prove it is an image, not a blank buffer.

    A GL context that yields a uniform frame is the failure this whole module
    exists to make impossible to miss, so the check is on the PIXELS.
    """
    ensure_gl()
    import numpy as np
    import mujoco

    xml = """<mujoco><worldbody>
      <light pos="0 0 3"/>
      <geom type="plane" size="5 5 .1" rgba=".3 .5 .3 1"/>
      <body pos="0 0 .5"><freejoint/><geom type="box" size=".2 .2 .2" rgba=".9 .2 .2 1"/></body>
    </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, height=120, width=160)
    r.update_scene(d)
    img = r.render()
    colours = len(np.unique(img.reshape(-1, 3), axis=0))
    if colours < 3:
        raise RuntimeError(
            f"GL context produced a near-uniform frame ({colours} distinct "
            "colours) — the renderer is up but is not drawing the scene."
        )
    return {"display": _DISPLAY, "backend": os.environ.get("MUJOCO_GL"),
            "shape": tuple(img.shape), "distinct_colours": int(colours)}


if __name__ == "__main__":
    print(selftest())
