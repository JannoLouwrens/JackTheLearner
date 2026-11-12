"""LAB — why does a PERFECT map feed only ~1/2 of births? (J1 diagnosis; no ledger row)

`smell_oracle` v3 (ready food at every birth) put the privileged goto_true
follower at .54/.54 on s91/s92: a policy that knows exactly where the nearest
ready food is, pushing straight at it with the full drive, misses half its
50-s births. The v2 probe (s90, 12 births) saw frozen bodies lying on their
side or with hands jammed into the floor. This instrument classifies EVERY
birth so the repair is chosen by a count, not an anecdote.

Per birth (goto_true, food ready at birth, E decisions):
  fed            ate within E
  toppled        upright cosine fell below 0.5 at any decision (first_topple = k)
  righted        after toppling, upright cosine came back above 0.8
  stuck          last 50 decisions moved < 5 cm (alive, not fed)
  partners       non-floor geoms touching the body at the end (wall, ramp...)
  hand_pen       deepest hand-into-ground penetration at the end (m)
  d_end, d_min   distance to the nearest ready floor food

Arms: adhesion 0.5 (action 0, what goto_true used) and adhesion 0 (-1).
Hand-written policy = an instrument, never a part of him.

    [PROBE_BODY=pole|plinth] [PROBE_ORACLE=line|path] nice -n 19 $PY -m experiments.body_probe <n_ep> <E> com|site|foot <adh,..> <seed>...
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

OUT = "/data/tmp/smell_sense/body_probe.jsonl"


DRIVE_Z = None   # body-frame height of the push; None = COM (W0 as is)
DRIVE_SCALE = 1.0  # multiplies W0's gated push (1.0 = W0 as is)
ORACLE = "line"    # "line" = straight at the food (v3); "path" = grid
                   # detour around inflated geometry (see _Planner)
SPAWN_SALT = 0     # !=0 reseeds the birth-position stream (BD.01's twin control)
_ORIG_FRAGMENTS = None
BODY = "pole"      # "pole" = W0 as built; "plinth" = W0.BAL arm C (wide box
                   # foot, mass moved into it), patched here so no certificate moves


def install_body(body: str) -> None:
    """Swap the rover fragments for W0.BAL arm C's plinth (w0bal_probe's
    exact strings). Refuses if the XML moved, so a pole is never run under
    a plinth label."""
    global BODY, _ORIG_FRAGMENTS
    import playground as pg
    if _ORIG_FRAGMENTS is None:
        _ORIG_FRAGMENTS = pg._rover_fragments
    pg._rover_fragments = _ORIG_FRAGMENTS   # one process may race both bodies
    BODY = body
    if body == "pole":
        return
    if body != "plinth":
        raise SystemExit(f"unknown body {body!r}")
    from .w0bal_probe import (_FOOT_PLINTH, _FOOT_SPHERE, _TORSO_HEAVY,
                              _TORSO_LIGHT)
    orig = pg._rover_fragments

    def patched(spawn):
        b, act = orig(spawn)
        if _TORSO_HEAVY not in b or _FOOT_SPHERE not in b:
            raise RuntimeError("plinth patch found nothing to replace")
        return (b.replace(_TORSO_HEAVY, _TORSO_LIGHT)
                 .replace(_FOOT_SPHERE, _FOOT_PLINTH), act)
    pg._rover_fragments = patched


class _PushAt:
    """mujoco shim: before each mj_step, add the torque that moves W0's COM
    push (xfrc_applied[b, :2], already gated by W0.decide) to DRIVE_Z on the
    body axis. Lives here, not in w0.py, so no certificate moves; the race it
    served found the push point irrelevant (see LAB)."""

    def __init__(self, w, z, scale=1.0):
        import mujoco
        self._mj, self._w, self._z, self._k = mujoco, w, z, float(scale)

    def __getattr__(self, k):
        return getattr(self._mj, k)

    def mj_step(self, m, d):
        b = self._w.rover_bid
        d.xfrc_applied[b, :2] *= self._k   # W0 re-sets it every substep
        if self._z is not None:
            mat = np.asarray(d.xmat[b]).reshape(3, 3)
            r = (np.asarray(d.xpos[b]) + mat[:, 2] * float(self._z)
                 - np.asarray(d.xipos[b]))
            d.xfrc_applied[b, 3:6] = np.cross(
                r, np.r_[d.xfrc_applied[b, :2], 0.0])
        self._mj.mj_step(m, d)
        d.xfrc_applied[b, 3:6] = 0.0


INFLATE = {"pole": 0.25, "plinth": 0.55}   # m: body footprint + margin
GRID = 0.1


class _Planner:
    """Privileged detouring map (an instrument, never a part of him): an
    8-connected Dijkstra on a GRID-m occupancy grid. Obstacles are the
    xy-footprints (geom_aabb rotated by geom_xmat) of every non-body geom
    that can touch the rover and is below 1.0 m, plus the pool pit, each
    inflated by INFLATE[BODY]; the target food is excluded. Blocked cells
    are soft (cost 50) so a birth inside an inflated margin can still leave."""

    def __init__(self, w, half):
        import mujoco
        m = w.model
        self.w, self.m, self.d = w, m, w.data
        n = int(round(2 * half / GRID))
        c = (np.arange(n) + 0.5) * GRID - half
        self.X, self.Y = np.meshgrid(c, c, indexing="ij")
        self.n, self.half = n, half
        nm = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        mine = [g for g in range(m.ngeom) if w._body_mask[g]]
        touches = lambda g: any(
            (int(m.geom_contype[g]) & int(m.geom_conaffinity[b]))
            or (int(m.geom_contype[b]) & int(m.geom_conaffinity[g]))
            for b in mine)      # ladder (contype 4) meets the hands
        self.geoms = [g for g in range(m.ngeom)
                      if not w._body_mask[g] and nm(g) != "floor"
                      and (touches(g) or nm(g) == "pool_water")]
        self.r = INFLATE[BODY]

    def occupancy(self, skip):
        m, d = self.m, self.d
        occ = np.zeros_like(self.X, dtype=bool)
        for g in self.geoms:
            if g in skip:
                continue
            R = np.asarray(d.geom_xmat[g]).reshape(3, 3)
            ab = np.asarray(m.geom_aabb[g])
            ctr = np.asarray(d.geom_xpos[g]) + R @ ab[:3]
            hz = float(np.abs(R[2]) @ ab[3:])
            if ctr[2] - hz > 1.0:
                continue                      # overhead (platform, apple)
            dx, dy = self.X - ctr[0], self.Y - ctr[1]
            lx = R[0, 0] * dx + R[1, 0] * dy  # grid point in geom frame (xy)
            ly = R[0, 1] * dx + R[1, 1] * dy
            occ |= ((np.abs(lx) <= ab[3] + self.r)
                    & (np.abs(ly) <= ab[4] + self.r))
        return occ

    def cell(self, p):
        i = np.clip(((np.asarray(p) + self.half) / GRID).astype(int),
                    0, self.n - 1)
        return int(i[0]), int(i[1])

    def path(self, p, q, skip):
        import heapq
        occ = self.occupancy(skip)
        cost = np.where(occ, 50.0, 1.0)
        s, t = self.cell(p), self.cell(q)
        dist = {s: 0.0}
        prev = {}
        h = [(0.0, s)]
        moves = [(a, b, (a * a + b * b) ** 0.5) for a in (-1, 0, 1)
                 for b in (-1, 0, 1) if a or b]
        while h:
            dd, u = heapq.heappop(h)
            if u == t:
                break
            if dd > dist.get(u, 1e18):
                continue
            for a, b, l in moves:
                v = (u[0] + a, u[1] + b)
                if not (0 <= v[0] < self.n and 0 <= v[1] < self.n):
                    continue
                nd = dd + l * cost[v]
                if nd < dist.get(v, 1e18):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(h, (nd, v))
        if t not in dist:
            return None, occ
        pts, u = [], t
        while u != s:
            pts.append(u)
            u = prev[u]
        pts.reverse()
        xy = [np.array([self.X[c], self.Y[c]]) for c in pts]
        blocked_len = sum(bool(occ[c]) for c in pts) * GRID
        return (xy, blocked_len), occ

    @staticmethod
    def steer(p, xy, look=0.5):
        tgt = xy[-1]
        for c in xy:
            if np.linalg.norm(c - p) > look:
                tgt = c
                break
        v = tgt - p
        return v / max(1e-6, float(np.linalg.norm(v)))


def classify(seed: int, n_ep: int, E: int, adh: float) -> list:
    import mujoco
    from . import smell_sense as S
    from . import w0 as W
    if not getattr(W.W0, "_oracle_installed", False):
        S.install("smell", seed)
        W.W0._oracle_installed = True
    from .tests.lf_03_learner_lives import _calibration
    from .tests.lf_04_living_teaches import _Life4
    j0, alpha, prov = _calibration()
    if j0 is None:
        raise SystemExit(f"PS.01 calibration refused: {prov}")
    L = _Life4(seed, j0, alpha, train=False)
    w, dr = L.w, L.w.drives
    m, d = w.model, w.data
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "rover_foot")
    is_box = int(m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_BOX)
    if is_box != (BODY == "plinth"):
        raise RuntimeError(f"body {BODY} not carried by the world")
    if DRIVE_Z is not None or DRIVE_SCALE != 1.0:
        w.mujoco = _PushAt(w, DRIVE_Z, DRIVE_SCALE)
    if SPAWN_SALT:
        w._spawn_rng = np.random.RandomState(seed * 104729 + 7 + SPAWN_SALT)
    w.observe()
    b = w.rover_bid
    floor = {n: g for n, (g, _) in dr._food.items()
             if d.geom_xpos[g][2] <= 0.5}
    body_g = {i for i in range(m.ngeom) if w._body_mask[i]}
    ground_g = {i for i in range(m.ngeom) if w._ground_mask[i]}
    hands = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, h)
             for h in ("handL", "handR")}
    nm = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g)
    idle = np.r_[np.zeros(4), -1.0, -1.0, 0.0, 0.0]
    plan = (_Planner(w, 6.0) if ORACLE == "path" else None)
    out = []
    for ep in range(n_ep):
        while not any(dr.t >= dr._respawn_at[n] for n in floor):
            w.decide(idle)
            if w.died_this_decision:
                w.respawn()
        w.respawn()
        birth_xy = [round(float(x), 2) for x in d.xpos[b][:2]]
        m0 = sum(dr.ate_total.values())
        a = np.zeros(8)
        a[4:6] = adh
        hist, ups = [], []
        fed, died, dmin, d_end = None, False, 1e9, float("nan")
        route, path_len0, blocked0 = None, None, None
        for k in range(E):
            p = np.array(d.xpos[b][:2])
            ready = [np.array(d.geom_xpos[g][:2]) for n, g in floor.items()
                     if dr.t >= dr._respawn_at[n]]
            if ready:
                q = min(ready, key=lambda q: np.linalg.norm(q - p))
                v = q - p
                d_end = float(np.linalg.norm(v))
                dmin = min(dmin, d_end)
                a[6:8] = v / max(1e-6, d_end)
                if plan is not None and d_end > 0.5:
                    if k % 5 == 0 or route is None:
                        tg = {g for n, g in floor.items()
                              if np.allclose(d.geom_xpos[g][:2], q)}
                        res, _ = plan.path(p, q, tg)
                        route = res[0] if res else None
                        if k == 0 and res:
                            path_len0 = round(len(res[0]) * GRID, 2)
                            blocked0 = round(res[1], 2)
                    if route:
                        a[6:8] = plan.steer(p, route)
            else:
                a[6:8] = 0.0
            hist.append(p)
            ups.append(float(d.xmat[b][8]))
            w.decide(a)
            if sum(dr.ate_total.values()) > m0:
                fed = k
                break
            if w.died_this_decision:
                died = True
                break
        ups = np.array(ups)
        low = np.flatnonzero(ups < 0.5)
        first_topple = int(low[0]) if len(low) else None
        righted = bool(first_topple is not None
                       and np.any(ups[first_topple:] > 0.8))
        stuck = bool(fed is None and not died and len(hist) >= 50
                     and np.linalg.norm(hist[-1] - hist[-50]) < 0.05)
        partners, hand_pen = set(), 0.0
        for i in range(int(d.ncon)):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in body_g and g2 not in body_g:
                other, mine = g2, g1
            elif g2 in body_g and g1 not in body_g:
                other, mine = g1, g2
            else:
                continue
            if other not in ground_g or nm(other) != "floor":
                partners.add(nm(other))
            if mine in hands and other in ground_g:
                hand_pen = max(hand_pen, -float(c.dist))
        rec = {"seed": seed, "body": BODY, "oracle": ORACLE,
               "birth_xy": birth_xy, "path_len0": path_len0, "blocked0": blocked0, "adh": adh, "drive_z": DRIVE_Z, "drive_scale": DRIVE_SCALE, "ep": ep, "fed": fed, "died": died,
               "toppled": first_topple is not None,
               "first_topple": first_topple, "righted": righted,
               "up_end": round(float(ups[-1]), 2), "stuck": stuck,
               "partners": sorted(partners), "hand_pen": round(hand_pen, 3),
               "d_min": round(dmin, 2), "d_end": round(d_end, 2),
               "z_end": round(float(d.xpos[b][2]), 2)}
        out.append(rec)
        print(json.dumps(rec), flush=True)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return out


def summary(recs: list) -> dict:
    fail = [r for r in recs if r["fed"] is None]
    fed = [r for r in recs if r["fed"] is not None]
    return {"n": len(recs), "fed": len(fed),
            "fed_toppled": sum(r["toppled"] for r in fed),
            "fail": len(fail),
            "fail_toppled": sum(r["toppled"] for r in fail),
            "fail_toppled_not_righted": sum(r["toppled"] and not r["righted"]
                                            for r in fail),
            "fail_stuck": sum(r["stuck"] for r in fail),
            "fail_upright_stuck": sum(r["stuck"] and r["up_end"] > 0.8
                                      for r in fail),
            "fail_partner": sum(bool(r["partners"]) for r in fail),
            "fail_died": sum(r["died"] for r in fail)}


DRIVE_AT = {"com": None, "site": "ROVER_TORSO_BOT", "foot": "ROVER_FOOT_Z"}


if __name__ == "__main__":
    # body_probe <n_ep> <E> <drive_at:com|site|foot>[@scale] <adh,..> <seed>...
    import torch
    import playground as pg
    torch.set_num_threads(1)
    n_ep, E = int(sys.argv[1]), int(sys.argv[2])
    at, _, sc = sys.argv[3].partition("@")
    DRIVE_SCALE = float(sc) if sc else 1.0
    DRIVE_Z = (None if DRIVE_AT[at] is None
               else float(getattr(pg, DRIVE_AT[at])))
    adhs = [float(x) for x in sys.argv[4].split(",")]
    install_body(__import__("os").environ.get("PROBE_BODY", "pole"))
    ORACLE = __import__("os").environ.get("PROBE_ORACLE", "line")
    if ORACLE not in ("line", "path"):
        raise SystemExit(f"unknown oracle {ORACLE!r}")
    for seed in [int(x) for x in sys.argv[5:]]:
        for adh in adhs:
            t0 = time.perf_counter()
            recs = classify(seed, n_ep, E, adh)
            s = summary(recs)
            s.update(seed=seed, body=BODY, oracle=ORACLE, adh=adh, drive_at=at, drive_scale=DRIVE_SCALE,
                     wall_s=round(time.perf_counter() - t0, 1))
            print("SUMMARY " + json.dumps(s), flush=True)
