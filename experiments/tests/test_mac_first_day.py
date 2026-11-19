"""Row 11 and row 12 plumbing of scripts/mac_first_day.py (Next-up item 5).

The box has no OpenGL, so row 11's timing loop is exercised with a planted renderer: it must count
exactly the frames it rendered (not the warm-up), report the frame shape, and flag a blank frame.
Control: a renderer that cannot start must be recorded as FAILED with its error, never as a rate.
The launchd plists must parse as plists, name the context, and never ask for Background QoS."""
import importlib.util
import plistlib
import tempfile
import sys
import types
from argparse import Namespace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("mac_first_day", REPO / "scripts" / "mac_first_day.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeRenderer:
    calls = 0

    def __init__(self, model, h, w, blank=False):
        self.h, self.w, self.blank = h, w, blank

    def update_scene(self, data):
        pass

    def render(self):
        _FakeRenderer.calls += 1
        px = np.zeros((self.h, self.w, 3), np.uint8)
        if not self.blank:
            px[0, 0, 0] = 255
        return px

    def close(self):
        pass


def _run_m11(mod, renderer):
    real = sys.modules.get("mujoco")
    sys.modules["mujoco"] = types.SimpleNamespace(Renderer=renderer)
    world = types.SimpleNamespace(w=types.SimpleNamespace(model=object(), data=object()))
    mod._jack_world = lambda: world
    try:
        return mod.m11(Namespace(quick=True, context="launchagent"))
    finally:
        if real is None:
            sys.modules.pop("mujoco", None)
        else:
            sys.modules["mujoco"] = real


def test_render_counts_only_timed_frames():
    mod = _load()
    _FakeRenderer.calls = 0
    rec = _run_m11(mod, _FakeRenderer)
    assert rec["context"] == "launchagent"
    r = rec["render"]
    assert r["status"] == "RAN" and r["frames"] == 200 and r["shape"] == [64, 64, 3]
    assert _FakeRenderer.calls == 210  # 10 warm-up + 200 timed; only the 200 are in the rate
    assert r["frames_per_s"] > 0 and r["nonblank"] is True


def test_blank_frame_is_flagged():
    mod = _load()
    rec = _run_m11(mod, lambda m, h, w: _FakeRenderer(m, h, w, blank=True))
    assert rec["render"]["nonblank"] is False


def test_renderer_that_cannot_start_is_failed_not_zero():
    mod = _load()

    def broken(*_a):
        raise RuntimeError("no OpenGL context")

    rec = _run_m11(mod, broken)
    assert rec["render"]["status"] == "FAILED" and "no OpenGL context" in rec["render"]["error"]
    assert "frames_per_s" not in rec["render"]


def test_launchd_plists():
    mod = _load()
    tmp_path = Path(tempfile.mkdtemp())
    paths = mod.emit_launchd(str(tmp_path / "pl"), str(tmp_path / "res"))
    assert len(paths) == 2
    for p in paths:
        d = plistlib.loads(Path(p).read_bytes())
        ctx = d["Label"].rsplit(".", 1)[1]
        assert ctx in ("launchagent", "launchdaemon")
        args = d["ProgramArguments"]
        assert args[args.index("--context") + 1] == ctx and args[args.index("--only") + 1] == "11"
        assert d["ProcessType"] == "Interactive" and d["RunAtLoad"] is True
        assert ("UserName" in d) == (ctx == "launchdaemon")


def main():
    for t in (test_render_counts_only_timed_frames, test_blank_frame_is_flagged,
              test_renderer_that_cannot_start_is_failed_not_zero, test_launchd_plists):
        t()
        print(f"{t.__name__}: PASS")
    print("test_mac_first_day PASS")


if __name__ == "__main__":
    main()
