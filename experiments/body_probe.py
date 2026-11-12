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

    [PROBE_BODY=pole|plinth] nice -n 19 $PY -m experiments.body_probe <n_ep> <E> com|site|foot <adh,..> <seed>...
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

OUT = "/data/tmp/smell_sense/body_probe.jsonl"


DRIVE_Z = None   # body-frame height of the push; None = COM (W0 as is)
DRIVE_SCALE = 1.0  # multiplies W0's gated push (1.0 = W0 as is)
BODY = "pole"      # "pole" = W0 as built; "plinth" = W0.BAL arm C (wide box
                   # foot, mass moved into it), patched here so no certificate moves


def install_body(body: str) -> None:
    """Swap the rover fragments for W0.BAL arm C's plinth (w0bal_probe's
    exact strings). Refuses if the XML moved, so a pole is never run under
    a plinth label."""
    global BODY
    BODY = body
    if body == "pole":
        return
    if body != "plinth":
        raise SystemExit(f"unknown body {body!r}")
    import playground as pg
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
    out = []
    for ep in range(n_ep):
        while not any(dr.t >= dr._respawn_at[n] for n in floor):
            w.decide(idle)
            if w.died_this_decision:
                w.respawn()
        w.respawn()
        m0 = sum(dr.ate_total.values())
        a = np.zeros(8)
        a[4:6] = adh
        hist, ups = [], []
        fed, died, dmin, d_end = None, False, 1e9, float("nan")
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
        rec = {"seed": seed, "body": BODY, "adh": adh, "drive_z": DRIVE_Z, "drive_scale": DRIVE_SCALE, "ep": ep, "fed": fed, "died": died,
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
    for seed in [int(x) for x in sys.argv[5:]]:
        for adh in adhs:
            t0 = time.perf_counter()
            recs = classify(seed, n_ep, E, adh)
            s = summary(recs)
            s.update(seed=seed, body=BODY, adh=adh, drive_at=at, drive_scale=DRIVE_SCALE,
                     wall_s=round(time.perf_counter() - t0, 1))
            print("SUMMARY " + json.dumps(s), flush=True)
