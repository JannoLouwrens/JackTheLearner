"""fast_sim.py — J6(2): per-sub-step Python moved out of the decision where the
result is identical. Each piece is a `jack_env` option, default OFF; nothing
here edits `ContactAudio.py` or `w0.py` (so no standing certificate stales).

(a) `lazy_contact_audio` — `ContactAudioSynth.step` with the work it throws
    away removed. The shipped step, every one of the 40 sub-steps of a
    decision, computes `mj_contactForce` and copies `con.pos` for EVERY
    contact (~24 at rest) and only then discards every pair that is already
    touching or inside its refractory window — which is nearly all of them.
    Here pairs are read from MuJoCo's contact arrays in one call, and force and
    position are computed only for a pair that will actually fire, over the
    same contact indices in the same order, so the event's centroid and summed
    force are the same floating-point operations on the same numbers.
    `_prev_pairs` and `_last_fired` evolve exactly as before. Proof:
    `experiments/tests/test_fast_sim.py` (stream identity incl. every audio
    event field, and a control that must differ).

(b) `multiray_vision` — J6(3): `W0._vision`'s 16 `mj_ray` calls replaced by
    ONE `mj_multiRay` over the same origin, directions, geom groups,
    flg_static=1 and bodyexclude=-1, cutoff mjMAXVAL (so the
    "> RAY_MAX_RANGE reads empty" clause stays where it was). The panel's
    noise draw happens per ray in the same order k = 0..15, so the world's
    random stream is consumed identically. MuJoCo 3.2.3 returns the same
    distance and geom id bit for bit (2,400 rays, 0 differences, 11-16).
    Proof: `experiments/tests/test_multiray.py`. Speed: today's eye is
    0.057 ms of a ~20 ms decision, so this is NOT a speed lever for the
    16-ray eye (<0.3%); its value is the batched call §12's retina needs
    (16x16 0.10 ms, 32x32 0.34 ms, 5-7x the loop, §12).

Speed table: in `experiments/tests/test_fast_sim.py`'s docstring.
"""
from __future__ import annotations

import math
import types

import numpy as np

import experiments.w0  # noqa: F401  (puts ContactAudio on the path)
from ContactAudio import REFRACTORY_S


def _lazy_step(self, data) -> None:
    import mujoco
    t = float(data.time)
    n = int(data.ncon)
    by_pair: dict = {}
    if n:
        g = data.contact.geom[:n]
        for i, (a, b) in enumerate(g.tolist()):
            pair = (a, b) if a <= b else (b, a)
            lst = by_pair.get(pair)
            if lst is None:
                by_pair[pair] = [i]
            else:
                lst.append(i)
    prev, last = self._prev_pairs, self._last_fired
    for pair, idx in by_pair.items():
        if pair in prev:
            continue
        if t - last.get(pair, -1e9) < REFRACTORY_S:
            continue
        pts = []
        for i in idx:
            con = data.contact[i]
            f6 = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, i, f6)
            pts.append((np.array(con.pos), abs(float(f6[0]))))
        pos = np.mean([p for p, _ in pts], axis=0)
        force = float(sum(f for _, f in pts))
        last[pair] = t
        self.events.append(self._make_event(t, pair, pos, force))
    self._prev_pairs = set(by_pair)


_MR_HAS_NORMAL = None


def _multi_ray(model, data, pos, vec, groups, gid, dist, n) -> None:
    """`mj_multiRay` under either signature. mujoco >= 3.3 added a `normal`
    out-argument before `nray`; every Kaggle kernel pins 3.17.0 while this box
    runs 3.2.3. Until 11-21 each caller had to remember
    `sim_speed_kcpu._compat_multiray()` first, and `pilot_b2` (J17's row128
    arms) did not: three of four arms would have died at construction on
    Kaggle. The eye now checks the thing itself. Same rays, same outputs; the
    normals are not asked for (None)."""
    global _MR_HAS_NORMAL
    import mujoco
    f = mujoco.mj_multiRay
    if _MR_HAS_NORMAL is None:
        # a caller's installed shim has no docstring and takes the old 11 arguments
        _MR_HAS_NORMAL = "normal:" in (getattr(f, "__doc__", None) or "")
    if _MR_HAS_NORMAL:
        f(model, data, pos, vec, groups, 1, -1, gid, dist, None, n, mujoco.mjMAXVAL)
    else:
        f(model, data, pos, vec, groups, 1, -1, gid, dist, n, mujoco.mjMAXVAL)


def _multiray_vision(self) -> np.ndarray:
    import mujoco
    from experiments import w0 as W
    pos = np.array(self.data.xpos[self.rover_bid], dtype=float)
    pos[2] += W.RAY_Z_OFFSET
    n = W.N_RAYS
    if not hasattr(self, "_mr_vec"):
        self._mr_vec = np.ascontiguousarray(self._ray_dirs, dtype=np.float64).reshape(-1)
        self._mr_gid = np.zeros(n, dtype=np.int32)
        self._mr_dist = np.zeros(n, dtype=np.float64)
    _multi_ray(self.model, self.data, pos, self._mr_vec, W._RAY_GROUPS,
               self._mr_gid, self._mr_dist, n)
    out = np.empty(W.MODALITIES["vision"], dtype=np.float32)
    for k in range(n):
        dist = float(self._mr_dist[k])
        gid = int(self._mr_gid[k])
        if dist < 0 or dist > W.RAY_MAX_RANGE or gid < 0:
            d, t = 1.0, 0.0
        else:
            d = dist / W.RAY_MAX_RANGE
            t = float(self._tex[gid])
            if gid == self.panel_gid:
                amp = min(1.0, 2.5 / max(dist, 1e-6))
                t = 0.5 + amp * (float(self._rng.uniform()) - 0.5)
        out[2 * k] = d
        out[2 * k + 1] = t
    # keep the shipped side effect: _geomid holds the LAST ray's hit
    self._geomid[0] = self._mr_gid[n - 1]
    return out


def apply_multiray_vision(w) -> None:
    """Swap the instance's `_vision` for the one-call version (this W0 only)."""
    w._vision = types.MethodType(_multiray_vision, w)


def apply_lazy_contact_audio(w) -> None:
    """Swap the instance's `synth.step` for the lazy one (this W0 only)."""
    w.synth.step = types.MethodType(_lazy_step, w.synth)


# ── J12 (1): the 128-ray level row, an EYE CHANGE (not an identity) ─────────
# LAB `experiments/eye_streak.py` (bddffac4): one level row of 128 world-frame
# azimuths finds floor food .909 at d0 where today's 16 rays find .528.
# Ray k points at azimuth 2*pi*k/n_az from the same origin as W0's eye, same
# geom groups, flg_static=1, bodyexclude=-1, cutoff mjMAXVAL and the same
# "> RAY_MAX_RANGE reads empty" clause; each ray reads [distance/8 m,
# material] exactly as W0's 16 do, the noise panel included (one world draw
# per panel ray, in ray order). With n_az = 128, rays 8j are today's ray j's
# direction, so on the same body state they read the same numbers.
ROW_EYES = {"row128": 128}


def _row_vision(self) -> np.ndarray:
    import mujoco
    from experiments import w0 as W
    n = self._row_n
    pos = np.array(self.data.xpos[self.rover_bid], dtype=float)
    pos[2] += W.RAY_Z_OFFSET
    if self.__dict__.get("_eye_body"):           # J12 (0): body-frame azimuths
        self._row_vec[:] = _rot_z(self._row_dirs0, body_yaw(self)).reshape(-1)
    _multi_ray(self.model, self.data, pos, self._row_vec, W._RAY_GROUPS,
               self._row_gid, self._row_dist, n)
    dist, gid = self._row_dist, self._row_gid
    miss = (dist < 0) | (dist > W.RAY_MAX_RANGE) | (gid < 0)
    d = np.where(miss, 1.0, dist / W.RAY_MAX_RANGE)
    t = np.where(miss, 0.0, self._tex[np.maximum(gid, 0)])
    if self.panel_gid >= 0:
        # under paint="kind" the draws are still taken (and thrown away), so
        # the world's `_rng` stream — the placebo channel — is the geom
        # paint's, draw for draw.
        for k in np.flatnonzero((~miss) & (gid == self.panel_gid)):
            amp = min(1.0, 2.5 / max(float(dist[k]), 1e-6))
            t[k] = 0.5 + amp * (float(self._rng.uniform()) - 0.5)
    if self.__dict__.get("_paint_kind"):
        # J24 (0): [distance, K 0/1 kind channels] per ray. `_kind_of` is set
        # by set_kind_table; a look before that raises (unknown is not zero).
        out = np.zeros((n, 1 + N_KINDS), dtype=np.float32)
        out[:, 0] = d
        hit = np.flatnonzero(~miss)
        out[hit, 1 + self._kind_of[gid[hit]]] = 1.0
        return out.reshape(-1)
    out = np.empty(2 * n, dtype=np.float32)
    out[0::2] = d
    out[1::2] = t
    return out


# ── J24 (0), ruling 111: ONE LOOK PER KIND (a WORLD-PAINT CHANGE, default off)
# W0 paints every geom with its own code, `((gid * 0.37) % 0.8) + 0.1`
# (w0.py:225): three floor foods = three codes, re-dealt whenever the world's
# geom count moves (23/23 worlds, j24_class_probe.py). No published forager
# was given that (0 of 12, PUBLISHED_PRIORS §28): every one returns a CLASS
# per ray, the same in every episode. paint="kind": the ray's one texture
# number is replaced by N_KINDS 0/1 channels, one per KIND of thing the ray
# HIT, the same in every world, all zero on a miss:
#   food      a geom registered as FLOOR food in this world's drive layer
#             (obj0, obj1 + the dial's extras) — every one the same channel
#   water     `pool_water`
#   obstacle  every other thing a ray can hit: ramp, stairs, ladder, platform,
#             the pool's own walls and floor, blocks, the seesaw, floor
#             objects that are not food in this world, the apple (a food, but
#             not a FLOOR food; it hangs above the level row), shelters, and
#             any geom a mutated world adds
#   wall      the arena's boundary, `wall0`..`wall3`
#   panel     `noise_panel` (a constant look now; its noise was its texture)
# It is a property of the thing hit, as texture was: a food that has been
# eaten and is regrowing keeps its look, as it kept its texture. Nothing says
# which channel feeds him; no reward, need or position enters. His own body
# (group 3) is never hit by a ray, as before.
KINDS = ("food", "water", "obstacle", "wall", "panel")
N_KINDS = len(KINDS)
KIND_WALLS = ("wall0", "wall1", "wall2", "wall3")
PAINTS = ("geom", "kind")


def kind_table(w, floor_foods) -> np.ndarray:
    """geom id -> index into KINDS, for this world as it is built."""
    import mujoco
    m = w.model
    k = np.full(m.ngeom, KINDS.index("obstacle"), dtype=np.int64)
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name == "pool_water":
            k[g] = KINDS.index("water")
        elif name in KIND_WALLS:
            k[g] = KINDS.index("wall")
    if w.panel_gid >= 0:
        k[w.panel_gid] = KINDS.index("panel")
    for name in floor_foods:
        if name in w.drives._food:
            k[int(w.drives._food[name][0])] = KINDS.index("food")
    return k


def apply_kind_paint(w) -> None:
    """Mark this W0's ROW eye as kind-painted (call after apply_row_vision).
    The table itself is set by set_kind_table once the world's floor foods
    are registered."""
    if "_row_vec" not in w.__dict__:
        raise ValueError("paint='kind' is built for the row eye only")
    w._paint_kind = True


def set_kind_table(w, floor_foods) -> None:
    w._kind_of = kind_table(w, floor_foods)


# ── J24 (1): eye="bins16" — closeness bins per kind, from the kind-painted
# row AND NOTHING ELSE (rllab Gather's readings / Safety Gym's lidar bins,
# PUBLISHED_PRIORS §27-28). A pure function of the 128 x (1 + N_KINDS) row:
# per kind, 16 azimuth bins; bin j holds the 8 rays centred on the old eye's
# ray j (row rays 8j-4 .. 8j+3, so "straight ahead" is the middle of bin 0);
# its value = max over those rays painted that kind of (1 - distance/range),
# 0 if none is. Layout: kind-major, [food x16, water x16, obstacle x16,
# wall x16, panel x16] = 80 numbers.
BINS = 16


def bins16(row: np.ndarray) -> np.ndarray:
    r = np.asarray(row, dtype=np.float32).reshape(-1, 1 + N_KINDS)
    n = r.shape[0]
    per = n // BINS
    if n % BINS:
        raise ValueError(f"{n} rays do not fill {BINS} bins")
    close = (1.0 - r[:, :1]) * r[:, 1:]                  # (n, K); 0 where not hit
    close = np.roll(close, per // 2, axis=0)             # ray 8j-4 leads bin j
    return close.reshape(BINS, per, N_KINDS).max(axis=1).T.reshape(-1).astype(np.float32)


def apply_bins16(w) -> None:
    """Wrap this W0's (kind-painted row) observe: vision becomes bins16 of
    it. Call after apply_row_vision + apply_kind_paint."""
    if not w.__dict__.get("_paint_kind"):
        raise ValueError("eye='bins16' needs paint='kind'")
    inner = w.observe

    def observe(self):
        o = inner()
        self._row_last = o["vision"]         # the row it was computed from
        o["vision"] = bins16(o["vision"])
        return o

    w.observe = types.MethodType(observe, w)


def apply_row_vision(w, eye: str = "row128") -> None:
    """Give this W0 the n_az-ray level row. `observe()` keeps its order of
    world draws (vision's panel draws, then the placebo channel) and its width
    check on every other sense; only `vision` is 2*n_az wide. `observe_pure`
    goes through the patched `observe`, so J9's stream restore still holds."""
    n = ROW_EYES[eye]
    az = 2 * np.pi * np.arange(n) / n
    dirs = np.stack([np.cos(az), np.sin(az), np.zeros(n)], -1)
    w._row_n = n
    w._row_vec = np.ascontiguousarray(dirs, dtype=np.float64).reshape(-1)
    w._row_gid = np.zeros(n, dtype=np.int32)
    w._row_dist = np.zeros(n, dtype=np.float64)
    from experiments import w0 as W
    blank = np.zeros(W.MODALITIES["vision"], dtype=np.float32)
    shipped_observe = type(w).observe

    def observe(self):
        v = _row_vision(self)                    # panel draws first, as shipped
        own = self.__dict__.get("_vision")       # multiray_vision may be set
        self._vision = lambda: blank             # no draw, passes the width check
        try:
            o = shipped_observe(self)
        finally:
            if own is None:
                del self._vision
            else:
                self._vision = own
        o["vision"] = v
        return o

    w.observe = types.MethodType(observe, w)


# ── J12 (0), ruling 38: the BODY-FRAME eye (an EYE CHANGE, default off) ─────
# Both eyes above cast ray k at WORLD azimuth 2*pi*k/n — a free compass (LAB
# `find_himself.py`, 99e588f0) that the human-like body will not have. In the
# body frame ray k points at azimuth 2*pi*k/n + yaw, yaw = the heading of the
# rover body's own +x axis on the ground, atan2(xmat[1,0], xmat[0,0]) — the
# SAME yaw his ears already use (`W0._drain_audio`). The rays stay level (a
# tipped body's row does not tip): only the azimuth turns with him. Same
# origin, groups, range, materials, panel draws in ray order. The base
# directions are rotated (c*x - s*y, s*x + c*y), so at yaw exactly 0 the look
# is the world-frame look bit for bit.
def body_yaw(w) -> float:
    m = w.data.xmat[w.rover_bid]
    return math.atan2(float(m[3]), float(m[0]))


def _rot_z(dirs: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    out = dirs.copy()
    out[:, 0] = c * dirs[:, 0] - s * dirs[:, 1]
    out[:, 1] = s * dirs[:, 0] + c * dirs[:, 1]
    return out


def apply_body_frame_eye(w) -> None:
    """Turn this W0's eye (16 rays, or the row if `apply_row_vision` ran
    first) with the body's yaw. Call AFTER multiray/row are applied."""
    w._eye_body = True
    if "_row_vec" in w.__dict__:
        w._row_dirs0 = w._row_vec.reshape(-1, 3).copy()
        return                                   # _row_vision reads the flag
    w._ray_dirs0 = np.array(w._ray_dirs, dtype=float)
    own = w.__dict__.get("_vision")              # multiray_vision may be set
    inner = own if own is not None else types.MethodType(type(w)._vision, w)

    def vision(self):
        rot = _rot_z(self._ray_dirs0, body_yaw(self))
        self._ray_dirs = rot
        mr = self.__dict__.get("_mr_vec")
        if mr is not None:
            mr[:] = rot.reshape(-1)
        return inner()

    w._vision = types.MethodType(vision, w)


# ── J12 (2): the BODY-FRAME motor (a WORLD-INTERFACE CHANGE, default off) ───
# With the eye in the body frame, two compass channels are left: the drive
# (action[6:8] is a force along WORLD x, y) and the two root velocities in
# proprio (WORLD x, y). A body that turns (test_jack_env_eye_frame (d):
# median |yaw| 81 deg under random actions) and is not told its heading cannot
# aim a world-frame push at food seen in a body-frame ray. "body": the drive
# is (forward, left) along the body's own heading and proprio's two root
# velocities are (forward, left) too. Same yaw as the eye and the ears. The
# push is turned ONCE, at the start of the decision (0.2 s): W0's 40 sub-steps
# and their C twin are not touched.
def to_world(vec2, yaw: float):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * vec2[0] - s * vec2[1], s * vec2[0] + c * vec2[1]


def apply_body_frame_proprio(w) -> None:
    """Proprio's last two entries (root vx, vy) read in the body's frame."""
    inner = types.MethodType(type(w)._proprio, w)

    def proprio(self):
        out = inner()
        fx, fy = to_world((float(out[-2]), float(out[-1])), -body_yaw(self))
        out[-2], out[-1] = fx, fy
        return out

    w._proprio = types.MethodType(proprio, w)
