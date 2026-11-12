"""W0.PL — the pool's water does not trap a body on dry floor (World repair, ORDERS J1).

THE FINDING (experiments/pool_lip_contact.py, seed 90, LAB 11-12 ~18:30).
W0's pool has no pit: the infinite floor plane covers it. Yet `Water.apply`
(playground.py:627) (1) OVERWRITES `xfrc_applied[b, :3]` with drag, erasing the
gated drive W0.decide wrote into `xfrc_applied[rover, :2]` that substep, and
(2) counts a body "in the pool" when its bounding sphere dips below z 0 within
`half + r` of the centre — i.e. any body standing on the dry floor over or
beside the pool. Measured: the plinth stalls 0.55-1.27 m into a crossing (open
floor 7.20 m) and cannot back out; with water forces off it travels 4.03-5.89 m.
The pole is trapped too, and food obj1 sits inside the band: a starvation trap
made of a force bug, not of the world.

THE CLAIM. With the SHIPPED `Water.apply`, a body driven straight across the
pool by W0's own gated drive travels, on every crossing and every seed, at
least RATIO_MIN of what the same body travels on the same crossing with water
forces removed entirely. A repair must therefore (a) add to, never overwrite,
the drive, and (b) not apply water to a body the floor is holding up.

DECLARED 2026-11-12 (registration commit), BEFORE the repair and before any
claim seed was read (the diagnosis read seed 90 only):
  Venue     W0 via lf_04's _Life4 (train=False), world seeds 100, 101, 102
            (SEED_OFFSET 100 + runner seed 0..2), bodies pole AND plinth.
  Crossings four, built from the pool's own geometry so every seed has them:
            born BIRTH_OUT m outside each pool side on the side's centre line,
            pushed straight inward (full drive, adhesion action 0) for
            E_DEC decisions. Travel = displacement along the push.
  Arms      SHIPPED  = playground.Water.apply as it stands at run time;
            OFF      = water forces removed (xfrc restored after apply);
            LEGACY   = a frozen copy of the 11-12 apply (overwrite + band).
  Validity  a crossing whose OFF travel < MIN_OFF_TRAVEL m (an object blocks
            it with no water at all) is excluded and REPORTED; a seed with
            fewer than 3 valid crossings per body is VOID.
  Gate      gate_fires(arm) <=> min over seeds x bodies x valid crossings of
            travel_arm / travel_off >= RATIO_MIN.   RATIO_MIN = 0.90.
            The diagnosis' "water added, band kept" arm read 3.84-5.20 vs OFF
            4.03-5.89 (seed 90, plinth): an additive-only repair sits near the
            bar and may fail it — the band must go too.
  PASS      gate fires for SHIPPED AND does NOT fire for LEGACY (must-fail
            control, RUN through the same gate_fires; the 11-12 code read
            ratio ~0.14-0.3 on seed 90).

NOT THIS SPEC: whether he should have a real pit to fall into (a cut floor,
drowning reachable) or a surface-only pool — that is a raced world-design
question for a later spec. This one only requires that water not be a trap
where there is no water.
"""
from __future__ import annotations

import math

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

IMPL_DEPS = ['body_probe.py', 'w0.py', 'playground.py']

SEED_OFFSET = 100
E_DEC = 40
BIRTH_OUT = 1.0
MIN_OFF_TRAVEL = 2.0
RATIO_MIN = 0.90
BODIES = ("pole", "plinth")

_SEEDS: list = []
_CTRL: list = []


def _legacy_apply(self, model, data) -> None:
    """Frozen copy of playground.Water.apply at fe9c917 (the trap)."""
    if not self.enabled:
        return
    for bid in range(1, model.nbody):
        pos = data.xipos[bid]
        r = float(self._radius[bid])
        if r <= 0 or model.body_mass[bid] <= 0:
            continue
        if not (abs(pos[0] - self.x) < self.half + r
                and abs(pos[1] - self.y) < self.half + r
                and pos[2] - r < self.surface_z):
            continue
        d = self.surface_z - pos[2]
        if d >= r:
            frac = 1.0
        elif d <= -r:
            continue
        else:
            h = d + r
            frac = (h * h * (3 * r - h)) / (4.0 * r ** 3)
        vol = (4.0 / 3.0) * math.pi * r ** 3 * frac
        buoy = 1000.0 * vol * 9.81
        vel = data.cvel[bid][3:6]
        speed = float(np.linalg.norm(vel))
        area = math.pi * r * r * frac
        drag = -0.5 * 1000.0 * self.cd * area * speed * vel
        drag = drag - self.c_lin * 1000.0 * vol * vel
        data.xfrc_applied[bid][:3] = drag
        data.xfrc_applied[bid][2] += buoy


def _crossings(wv) -> list:
    o = wv.half + BIRTH_OUT
    return [("E", (wv.x + o, wv.y), (-1.0, 0.0)),
            ("W", (wv.x - o, wv.y), (1.0, 0.0)),
            ("N", (wv.x, wv.y + o), (0.0, -1.0)),
            ("S", (wv.x, wv.y - o), (0.0, 1.0))]


def _travel(body: str, seed: int, arm: str) -> list:
    import types
    from .. import body_probe as B
    import playground as pg
    B.install_body(body)
    try:
        from .lf_03_learner_lives import _calibration
        from .lf_04_living_teaches import _Life4
        j0, alpha, prov = _calibration()
        if j0 is None:
            raise RuntimeError(f"PS.01 calibration refused: {prov}")
        L = _Life4(seed, j0, alpha, train=False)
        w = L.w
        wv = w.water
        if wv is None:
            raise RuntimeError("W0 built without water")
        shipped = pg.Water.apply
        if arm == "legacy":
            wv.apply = types.MethodType(_legacy_apply, wv)
        elif arm == "off":
            def off(self, model, data, _s=shipped):
                keep = np.array(data.xfrc_applied)
                _s(self, model, data)
                data.xfrc_applied[:] = keep
            wv.apply = types.MethodType(off, wv)
        out = []
        for name, xy, u in _crossings(wv):
            w.respawn(at=xy)
            p0 = np.array(w.data.xpos[w.rover_bid][:2])
            a = np.zeros(8)
            a[6:8] = u
            for _ in range(E_DEC):
                w.decide(a)
                if w.died_this_decision:
                    break
            p1 = np.array(w.data.xpos[w.rover_bid][:2])
            out.append((name, float(np.dot(p1 - p0, u))))
        return out
    finally:
        B.install_body("pole")


def _ratios(seed: int, arm: str) -> tuple:
    """(min ratio over valid crossings, n valid, rows) for one seed, both bodies."""
    rows, rs, valid_per_body = [], [], {}
    for body in BODIES:
        t_off = dict(_travel(body, seed, "off"))
        t_arm = dict(_travel(body, seed, arm))
        nv = 0
        for k, off in t_off.items():
            ok = off >= MIN_OFF_TRAVEL
            rows.append(f"{body}:{k} {t_arm[k]:+.2f}/{off:+.2f}{'' if ok else ' (excl)'}")
            if ok:
                nv += 1
                rs.append(t_arm[k] / off)
        valid_per_body[body] = nv
    return (min(rs) if rs else float("nan"), min(valid_per_body.values()), rows)


def gate_fires(ratios: list) -> bool:
    return bool(ratios) and all(np.isfinite(r) for r in ratios) \
        and min(ratios) >= RATIO_MIN


def _experiment(seed: int) -> dict:
    s = seed + SEED_OFFSET
    r, nv, rows = _ratios(s, "shipped")
    m = {"seed_used": s, "min_ratio": r, "valid_min": nv, "rows": "; ".join(rows)}
    _SEEDS.append(m)
    return m


def _control(seed: int) -> dict:
    s = seed + SEED_OFFSET
    r, nv, rows = _ratios(s, "legacy")
    c = {"seed_used": s, "legacy_min_ratio": r, "valid_min": nv,
         "rows": "; ".join(rows)}
    _CTRL.append(c)
    return c


def _check(m: dict, c: dict):
    if not _SEEDS or len(_SEEDS) != len(_CTRL):
        return Status.VOID
    if any(x["valid_min"] < 3 for x in _SEEDS + _CTRL):
        return Status.VOID
    fired = gate_fires([x["min_ratio"] for x in _SEEDS])
    legacy = gate_fires([x["legacy_min_ratio"] for x in _CTRL])
    m["gate_fires"] = 1.0 if fired else 0.0
    c["legacy_gate_fires"] = 1.0 if legacy else 0.0
    return fired and not legacy


def run(ledger: Ledger | None = None):
    _SEEDS.clear()
    _CTRL.clear()
    return run_spec(BY_ID["W0.PL"], _experiment, _check,
                    control_fn=_control, ledger=ledger)
