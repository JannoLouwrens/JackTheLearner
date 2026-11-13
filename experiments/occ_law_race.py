"""LAB occlusion-law race (ORDERS J6; no ledger row, nothing registered).

The knife-edge (REVIEW_QUEUE `ne01-occlusion-knife-edge`, NE.01 FAIL RECORD,
J6 11-13): the shipped sky law is 9 head rays (zenith + 8 on a 30 deg cone),
occ moves in ninths, and a sleeping ragdoll cannot hold the 0.5-0.9 band - it
slides out and freezes, or seals (occ -> 1) and cooks. 11-13 00:45 the shelter
night went green, but ok_forager went red: one seed's sleeper slides into
sealed occ ~1 and dies of hyperthermia at ~3540 s of 3600.

The REVIEW row names three runnable arms; this script races the two that do
not need a new playground geometry, against the shipped law, on NE.01's own
probes (`_find_shelter_pose`, `_night`, `_forager` imported unchanged), with a
must-fail control. The world on disk is NOT edited: each arm patches the law
in-process only (D43: no world-edit window is declared; a race is evidence for
the Director/owner, not an edit).

  shipped  - the law as shipped (the null).
  graded   - a view-factor sky: 1 + 8 + 16 + 24 = 49 rays on rings at
             0/15/30/45 deg, each weighted cos(theta) (Lambert view factor of
             a horizontal sky patch), occ = weighted hit fraction. Grades the
             ninths into ~2 % steps; the 45 deg ring widens what "a roof"
             means without moving any geometry.
  damped   - shipped law; while ASLEEP the humanoid's joint (non-root) dof
             damping x DAMP_K = 3 (a sleeper keeps tone; a frictionless ragdoll
             is the anomaly). K is capped by the integrator: the playground
             runs RK4 at dt 5 ms, where damping is EXPLICIT; smoke 11-13 at
             seed 1 gave mjWARN_BADQVEL/QACC at K = 5, 8, 20 and none at 3.
             Tests "the band exists, the body just will not stay in it".
  open     - CONTROL that must FAIL: occ forced 0 (no roof exists). If it
             passes the band rule the rule cannot tell shelter from none.

Per arm x seed (NE.01 seeds 0,1,2 + fresh 3,4 - the fresh pair keeps the
winner from being tuned to the seeds that exposed the fault):
  pose  = _find_shelter_pose (the arm's own law picks the bed)
  night = _night(pose) and _night(OPEN_SPOT), fine physics
  for_  = _forager(pose), 18000 decisions

DECISION RULE, declared before any row is read (11-13 ~04:30 UTC):
  An arm HOLDS on a seed iff  night_shelter alive, occ_mean >= 0.40, cost <= 0.20,
  cost <= 0.5 x open-night cost  (NE.01's ok_night_shelter, verbatim)  AND
  forager alive at 18000 decisions (NE.01's ok_forager, verbatim).
  An arm WINS iff it HOLDS on all 5 seeds and `open` holds on 0 seeds.
  If several win: the one with the lower max over seeds of forager-sleep occ
  std (the knife-edge measure) - recorded, not re-tuned.
  If shipped wins, the knife-edge is not the cause of NE.01's red forager.
  If none wins: the four tables go to ORDERS J6 and the band-geometry arm
  (a playground edit) is the next thing raced.
  Any row whose physics raised a MuJoCo BADQACC/BADQVEL/BADQPOS warning
  (counted as unstable_decisions) does NOT hold: a law that breaks the
  integrator is not a world law.
Nothing here touches a bar: all thresholds are NE.01's own constants.

Usage:
  python -m experiments.occ_law_race ARM SEED OUT.jsonl [--smoke]
  python -m experiments.occ_law_race tally DIR
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ARMS = ("shipped", "graded", "damped", "open")
SEEDS = (0, 1, 2, 3, 4)
DAMP_K = 3.0
RINGS_DEG = (0.0, 15.0, 30.0, 45.0)
RING_N = (1, 8, 16, 24)


def _graded_dirs():
    dirs, w = [], []
    for th, n in zip(RINGS_DEG, RING_N):
        t = math.radians(th)
        for k in range(n):
            a = 2.0 * math.pi * k / n + (math.pi / n if n > 1 else 0.0)
            dirs.append((math.sin(t) * math.cos(a), math.sin(t) * math.sin(a),
                         math.cos(t)))
            w.append(math.cos(t))
    w = np.array(w)
    return np.array(dirs, dtype=np.float64), w / w.sum()


def _patch(arm: str, ne):
    """Install the arm's law into the imported modules, in-process only."""
    from experiments import needs

    if arm == "graded":
        dirs, wts = _graded_dirs()

        def occ(self, model, data):
            if self._mj is None:
                import mujoco
                self._mj = mujoco
            head = int(next(iter(self._mouth_geoms)))
            pnt = np.array(data.geom_xpos[head], dtype=np.float64)
            pnt[2] += needs.SKY_RAY_Z_OFFSET
            s = 0.0
            for k in range(len(dirs)):
                d = self._mj.mj_ray(model, data, pnt, dirs[k], None, 1, -1,
                                    self._geomid)
                g = int(self._geomid[0])
                if d >= 0 and g >= 0 and g not in self._jack_geoms:
                    s += wts[k]
            return float(s)
        needs.NeedLayer._sky_occlusion = occ
    elif arm == "open":
        # the bed is chosen under the SHIPPED law (see run_one), then the
        # roof stops counting: same place, same body, no shelter term
        needs.NeedLayer._sky_occlusion = lambda self, model, data: 0.0
    elif arm == "damped":
        orig = ne._step_decision
        cache = {}

        def step(model, data, water, layer, ctrl, null=None):
            key = id(model)
            if key not in cache:
                from playground import humanoid_index
                bodies = set(humanoid_index(model)["bodies"])
                dofs = np.array([j for j in range(model.nv)
                                 if int(model.dof_bodyid[j]) in bodies
                                 and j >= humanoid_index(model)["dofadr"] + 6])
                cache[key] = (dofs, model.dof_damping[dofs].copy())
            dofs, base = cache[key]
            model.dof_damping[dofs] = base * (DAMP_K if layer.asleep else 1.0)
            return orig(model, data, water, layer, ctrl, null)
        ne._step_decision = step


def run_one(arm: str, seed: int, out: str, smoke: bool = False) -> dict:
    from experiments.tests import ne_01_nobody_survives_by_accident as ne

    if smoke:
        ne.FORAGER_DECISIONS = 300
        ne.NIGHT_DECISIONS = 120
        ne.POSE_REST_DEC = 20
    pose = ne._find_shelter_pose(seed) if arm == "open" else None
    _patch(arm, ne)
    rec = []
    inner = ne._step_decision

    import mujoco
    bad = [0]
    W = mujoco.mjtWarning
    wids = [int(W.mjWARN_BADQACC), int(W.mjWARN_BADQVEL), int(W.mjWARN_BADQPOS)]

    def recorded(model, data, water, layer, ctrl, null=None):
        out = inner(model, data, water, layer, ctrl, null)
        if any(int(data.warning[i].number) > 0 for i in wids):
            bad[0] += 1
        if layer.asleep:
            rec.append(layer.last_occlusion)
        return out
    ne._step_decision = recorded
    t0 = time.process_time()
    if pose is None:
        pose = ne._find_shelter_pose(seed)
    n_sh = ne._night(seed, pose["xy"], 23, coarse=False)
    n_op = ne._night(seed, ne.OPEN_SPOT, 21, coarse=False)
    rec.clear()
    fo = ne._forager(seed, pose["xy"])
    fo["sleep_occ_sd"] = float(np.std(rec)) if rec else 0.0
    fo["sleep_occ_sealed"] = float(np.mean(np.array(rec) >= 0.95)) if rec else 0.0
    holds_night = bad[0] == 0 and bool(n_sh["alive"] == 1 and n_sh["occ_mean"] >= ne.SHELTER_OCC_MIN
                       and n_sh["cost"] <= ne.SHELTER_COST_MAX
                       and n_sh["cost"] <= ne.SHELTER_RATIO_MAX * max(n_op["cost"], 1e-9))
    holds_for = bool(fo["alive"] == 1 and fo["survived_dec"] >= ne.FORAGER_DECISIONS)
    row = {"arm": arm, "seed": seed, "smoke": smoke,
           "pose_xy": [float(v) for v in pose["xy"]], "occ_probe": pose["occ_probe"],
           "occ_max": pose["occ_max_reachable"],
           "night_occ": n_sh["occ_mean"], "night_cost": n_sh["cost"],
           "night_T_dawn": n_sh["T_dawn"], "night_alive": n_sh["alive"],
           "open_cost": n_op["cost"], "open_occ": n_op["occ_mean"],
           "for_alive": fo["alive"], "for_dec": fo["survived_dec"],
           "for_cause": fo["cause"], "for_death_s": fo["death_s"],
           "for_meals": fo["meals"], "for_sleep_occ": fo["sleep_occ_mean"],
           "for_sleep_occ_sd": fo["sleep_occ_sd"],
           "for_sleep_sealed_frac": fo["sleep_occ_sealed"],
           "unstable_decisions": bad[0],
           "holds_night": holds_night, "holds_forager": holds_for,
           "holds": holds_night and holds_for,
           "cpu_s": round(time.process_time() - t0, 1)}
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    return row


def tally(d: str) -> int:
    rows = []
    for p in sorted(Path(d).glob("*.jsonl")):
        rows += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if not r.get("smoke")]
    by = {a: {r["seed"]: r for r in rows if r["arm"] == a} for a in ARMS}
    print(f"{'arm':8s} seed  probe night_occ cost  T_dawn open  for_alive dec   cause        sl_occ sd    HOLDS")
    for a in ARMS:
        for s in SEEDS:
            r = by[a].get(s)
            if r is None:
                print(f"{a:8s} {s}    MISSING")
                continue
            print(f"{a:8s} {s}    {r['occ_probe']:.2f}  {r['night_occ']:.3f}    {r['night_cost']:.3f} "
                  f"{r['night_T_dawn']:.2f}  {r['open_cost']:.3f}  {r['for_alive']}  {r['for_dec']:5d} "
                  f"{r['for_cause']:12s} {r['for_sleep_occ']:.3f} {r['for_sleep_occ_sd']:.3f} {int(r['holds'])}")
    complete = all(len(by[a]) == len(SEEDS) for a in ARMS)
    open_ok = sum(r["holds"] for r in by["open"].values()) == 0
    winners = [a for a in ARMS[:3] if len(by[a]) == len(SEEDS)
               and all(r["holds"] for r in by[a].values())]
    print(f"complete={complete} control_open_fails={open_ok} winners={winners if open_ok else 'VOID (control held)'}")
    if winners and open_ok and len(winners) > 1:
        best = min(winners, key=lambda a: max(r["for_sleep_occ_sd"] for r in by[a].values()))
        print(f"tie-break (min max sleep-occ sd): {best}")
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "tally":
        sys.exit(tally(sys.argv[2]))
    run_one(sys.argv[1], int(sys.argv[2]), sys.argv[3], smoke="--smoke" in sys.argv)
