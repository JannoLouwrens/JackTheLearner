"""The eye's render-quality contract — what Jack's 64x64 eye pays GL for.

PROVENANCE. PL.00 attempt 1 (FAIL, 2026-08-30) measured the render at ~40 ms
per frame, resolution-independent, and its decomposition put the loop under
LC.02's 5.0 sim-s/real-s floor with NO encoder at all. The renderer bakeoff
(`experiments/tests/pl00_render_bakeoff.py`, 2026-09-07, artifact
/data/pl00_render_bakeoff.json, ordered by the Review's disposition of
`pl02-dependency-on-pl00-verdict-vs-table`) decomposed the 40 ms: a 4096^2
shadow-map pass (~22.7 ms) and 4x MSAA (~12.7 ms) — two full-scene software-GL
passes nobody chose, MuJoCo's defaults, serving an eye that reads 4,096
pixels. Winner by the pre-declared rule: shadows KEPT at 512^2, MSAA off.
Worst-seed loop throughput with the seat-holder encoder live: 8.594 vs the
unmoved 5.0 floor; the heavy ViT reference still fails (0.836), so the floor
still rejects — and now rejects encoders rather than any live eye.

WHAT THIS IS NOT. Not a change to `playground.py`: the world contract (what
exists, where the eye sits — EYE_POS/EYE_XYAXES/EYE_FOVY) is untouched, and
54 test modules declare `playground.py` in IMPL_DEPS. Render quality is a
property of the RENDERER. Existing visual certificates were measured under
the default quality and remain claims about their own runs; whether they
should migrate to this quality is a routed question, not something this
module decides for them. New visual work that wants the affordable eye opts
in by calling this and declaring this file in IMPL_DEPS.

Apply BEFORE constructing `mujoco.Renderer` — the offscreen framebuffers
(including the MSAA buffers) are allocated at construction.
"""
from __future__ import annotations

OFFSAMPLES = 0        # 4x MSAA off: ~12.7 ms/frame under llvmpipe
SHADOWSIZE = 512      # shadow map 4096^2 -> 512^2, pass kept: ~19 ms/frame


def apply_eye_quality(model) -> None:
    """Set the bakeoff-adopted render quality on a MuJoCo model, in place."""
    model.vis.quality.offsamples = OFFSAMPLES
    model.vis.quality.shadowsize = SHADOWSIZE
