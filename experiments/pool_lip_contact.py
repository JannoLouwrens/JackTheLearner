"""LAB — what does the plinth touch when it stalls at the pool? (J1 diagnosis; no ledger row)

`pool_lip_geo.py` showed the W0 pool has no pit: the floor PLANE (infinite for
collision) covers it, and poolwall0-3 are buried boxes whose tops sit at
z 0.000. The open question it left: when the plinth stalls by a pool wall,
is the blocking contact a wall FACE (horizontal normal, a ghost edge the box
foot snags on) or nothing at all (the stall is something else)?

This drives the plinth (W0.BAL arm C, via body_probe.install_body) straight
across each wall with W0's own gated drive (full push, adhesion action 0) and
logs every contact between a body geom and a non-floor geom, per decision:
partner, dist, contact-frame normal, position. Crossings: N (across
poolwall2, food obj1's wall), W (across poolwall1), diagonal into poolwall2,
and the obj1 approach. Control: the same push on open floor with no wall in
reach — it must show no partner contact and a full-speed run.

Hand-written pushes = an instrument, never a part of him. mj only; seed 90.

    [PROBE_WATER=as_is|add|off] nice -n 19 $PY -m experiments.pool_lip_contact [seed] [E]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

CROSSINGS = [   # name, birth xy, push direction
    ("N_poolwall2", (2.6, 0.2), (0.0, -1.0)),
    ("W_poolwall1", (0.3, -2.4), (1.0, 0.0)),
    ("diag_poolwall2", (2.0, 0.2), (0.7071, -0.7071)),
    ("obj1_approach", (1.51, 0.4), (0.0, -1.0)),
    ("open_floor_ctrl", (-5.0, 5.0), (1.0, 0.0)),
]


WATER = "as_is"   # "add": water forces added to the drive; "off": no water forces


def run(seed: int = 90, E: int = 40) -> list:
    import mujoco
    from . import body_probe as B
    B.install_body(os.environ.get("PROBE_BODY", "plinth"))
    from .tests.lf_03_learner_lives import _calibration
    from .tests.lf_04_living_teaches import _Life4
    j0, alpha, prov = _calibration()
    if j0 is None:
        raise SystemExit(f"PS.01 calibration refused: {prov}")
    L = _Life4(seed, j0, alpha, train=False)
    w = L.w
    m, d = w.model, w.data
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "rover_foot")
    if (int(m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_BOX)) != (B.BODY == "plinth"):
        raise RuntimeError(f"{B.BODY} not carried by the world")
    nm = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g)
    body_g = {i for i in range(m.ngeom) if w._body_mask[i]}
    floor_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    b = w.rover_bid
    print("drive-gate ground geoms:", sorted(nm(g) for g in w.ground_gids))
    gate = {"on": 0, "n": 0}
    real = w._grounded

    def counted():
        r = real()
        gate["on"] += int(r)
        gate["n"] += 1
        return r
    w._grounded = counted
    wv = w.water
    print("water: centre", (wv.x, wv.y), "half", wv.half, "surface_z", wv.surface_z,
          "rover radius", round(float(wv._radius[b]), 3))
    if WATER != "as_is":
        import types
        orig = type(wv).apply

        def patched(self, model, data):
            keep = np.array(data.xfrc_applied[:, :3])
            orig(self, model, data)
            if WATER == "add":     # the drive (xy) survives; drag/buoyancy
                touched = np.any(data.xfrc_applied[:, :3] != keep, axis=1)
                data.xfrc_applied[touched, :2] += keep[touched, :2]
                # (adding keep's z too re-adds last substep's buoyancy, which
                # W0.decide never clears: it compounds - a probe artefact)
            elif WATER == "off":
                data.xfrc_applied[:, :3] = keep
        wv.apply = types.MethodType(patched, wv)
    wet = {"n": 0}
    out = []
    for name, xy, u in CROSSINGS:
        w.respawn(at=xy)
        a = np.zeros(8)
        a[6:8] = u
        rows = []
        for k in range(E):
            gate["on"] = gate["n"] = 0
            w.decide(a)
            if w.died_this_decision:
                rows.append({"k": k, "died": True})
                break
            mat = np.asarray(d.xmat[b]).reshape(3, 3)
            cons = []
            for i in range(d.ncon):
                c = d.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                if (g1 in body_g) == (g2 in body_g):
                    continue
                other = g2 if g1 in body_g else g1
                if other == floor_g:
                    continue
                n = np.array(c.frame[:3])
                if g1 in body_g:       # normal points geom1 -> geom2; make it
                    n = -n             # point from the partner INTO the body
                cons.append({"mine": nm(g1 if g1 in body_g else g2),
                             "other": nm(other),
                             "dist": round(float(c.dist), 4),
                             "n": [round(float(x), 3) for x in n],
                             "pos": [round(float(x), 3) for x in c.pos]})
            rows.append({"k": k,
                         "wet": bool(wv.in_pool(d.xipos[b], float(wv._radius[b]))),
                         "xy": [round(float(x), 3) for x in d.xpos[b][:2]],
                         "z": round(float(d.xpos[b][2]), 3),
                         "up": round(float(mat[2, 2]), 3),
                         "v": round(float(np.linalg.norm(d.cvel[b][3:5])), 3),
                         "gate": round(gate["on"] / max(1, gate["n"]), 3),
                         "n_floor": sum(1 for i in range(d.ncon)
                                        if floor_g in (int(d.contact[i].geom1),
                                                       int(d.contact[i].geom2))),
                         "cons": cons})
        out.append({"crossing": name, "birth": xy, "u": u, "rows": rows})
    return out


def report(out: list) -> None:
    for c in out:
        rows = [r for r in c["rows"] if "xy" in r]
        if not rows:
            print(c["crossing"], "died at once")
            continue
        p0, p1 = np.array(c["birth"]), np.array(rows[-1]["xy"])
        travel = float(np.dot(p1 - p0, c["u"]))
        hit = [r for r in rows if r["cons"]]
        print(f"{c['crossing']:16s} birth {c['birth']} -> end {rows[-1]['xy']} "
              f"z {rows[-1]['z']} up {rows[-1]['up']} along-push {travel:+.2f} m, "
              f"decisions with a partner contact {len(hit)}/{len(rows)}")
        print("    drive gate per decision:", " ".join(f"{r['gate']:.2f}" for r in rows))
        print("    rover 'in pool'        :", " ".join("W" if r["wet"] else "." for r in rows))
        print("    floor contacts         :", " ".join(str(r['n_floor']) for r in rows))
        seen = {}
        for r in hit:
            for q in r["cons"]:
                key = (q["mine"], q["other"], tuple(np.round(q["n"], 1)))
                seen.setdefault(key, []).append((r["k"], q["dist"], q["pos"]))
        for (mine, other, n), ks in sorted(seen.items(), key=lambda t: t[1][0][0]):
            print(f"    {mine:>10s} x {other:<12s} n {list(n)}  decisions "
                  f"{ks[0][0]}..{ks[-1][0]} ({len(ks)} contacts) "
                  f"dist {min(x[1] for x in ks):+.4f} at {ks[0][2]}")


if __name__ == "__main__":
    os.chdir("/home/opc/jackthelearner")
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    E = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    WATER = os.environ.get("PROBE_WATER", "as_is")
    out = run(seed, E)
    os.makedirs("/data/tmp/poollip", exist_ok=True)
    with open(f"/data/tmp/poollip/contact_{os.environ.get('PROBE_BODY', 'plinth')}_s{seed}_{WATER}.json", "w") as f:
        json.dump(out, f)
    report(out)
