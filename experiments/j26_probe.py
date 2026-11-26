"""J26: what could the old eye row ever have told a learner? (no learner)

PLAN stage 1, part "Eye", step 4. Prior: PUBLISHED_PRIORS §28 "What the
record does NOT answer"; docs/OUTSIDER.md E1; ruling 111. Nobody has
published a scalar per-object code; our one-seed toy read one-hot 100% by
step 250 and the scalar code 25-36% at 20k.

NOT A LEARNER, NOT A LEDGER SPEC, NEVER GATES J24. A probe: the simulator is
the JUDGE (it labels where the nearest floor food is); a plain regressor
(brain B's own 256-256 MLP shape) asks how much of that label each eye form
carries in worlds it was not trained in.

=============================================================================
FROZEN in this commit (11-26), before any logged decision is read.
=============================================================================
World   the J22/J24 pilot's world (`pilot_b2.make_world`, PILOT_RULE: floor
        food only, body eye, body motor, native substep), eye="row128",
        paint="geom" (today's world — the kind paint changes no world state,
        tests/test_jack_env_paint.py (b), so ONE run logs every form).
Worlds  281..300 (20; the class probe's). TRAIN 281..295, TEST 296..300.
Policy  babble = `world_card.Held(seed, 25)`: a uniform random 8-d action
        held 25 decisions (f4's wander hold). A held random walk, chosen so
        his body travels and the food's bearing changes; reads no obs.
Size    10,000 decisions a world = 200,000. A death or episode end resets
        (env.reset), counted inside the 10k.
Logged  per decision, after the look the policy would act on: per ray
        distance/8 (miss 1), geom id (-1 miss), the row's texture channel
        as the eye returned it (panel noise included); and the JUDGE label:
        nearest FLOOR food (env.floor_foods' geoms) within RAY_MAX_RANGE of
        his eye, its bearing in his body frame (cos, sin) — no label if none.
Inputs  built offline from the same looks:
        R  today's row: [d, texture] x128 (256)
        C  one code for all floor food: R with every floor-food hit's
           texture replaced by 1.0 (outside W0's 0.1-0.9 codes) (256)
        K  kind one-hot per ray: [d, 5 kinds] x128 (768), fast_sim.kind_table
        B  bins16 of K (80), fast_sim.bins16
Probe   MLP 256-256 ReLU -> (cos, sin), Adam 1e-3, batch 256, 20 epochs, MSE,
        on labelled TRAIN decisions; score = R^2 of (cos, sin) on labelled
        TEST decisions (pooled over both outputs). Seeds 941, 942, 943 (init +
        batch order). Control: K with the TRAIN labels permuted -> must read
        R^2 <= 0.05 on TEST (else the probe is leaking; the run is void).
PREDICTIONS (written before the run; §28, OUTSIDER E1):
  P1  R explains < 10% (mean R^2 over 3 seeds < 0.10).
  P2  K and B each explain > 80% (mean R^2 > 0.80); C between.
  If R passes 0.80 the diagnosis behind J24 is WRONG, and WORLD_CARD.md
  § J17 hunt says so plainly. Either way the four numbers + control go there.
TIMING (box, 11-26 07:38, jack-std, nice 19, load 4.6): 300 decisions
  3.89 CPU-s incl. world build -> 200k ~ 0.72 CPU-h. Under the order's 2 h
  kernel; so the logger runs on the BOX, one worker, detached.
=============================================================================

    PYTHONPATH=. /data/venvs/jack-std/bin/python -m experiments.j26_probe time 300
    ... j26_probe log OUTDIR [WORLD ...]      # the logger (box, detached)
    ... j26_probe fit LOOKS OUT.json          # the probe (20 epochs, all looks)
"""
import json
import math
import os
import sys
import time

import numpy as np

WORLDS = tuple(range(281, 301))
TRAIN, TEST = WORLDS[:15], WORLDS[15:]
SEEDS = (941, 942, 943)
PER_WORLD = 10_000
HOLD = 25
FOOD_CODE = 1.0


def make_env(seed: int):
    from experiments.pilot_b2 import PILOT_RULE, make_world
    return make_world(seed, food_floor_only=PILOT_RULE["food_floor_only"],
                      eye_frame=PILOT_RULE["eye_frame"],
                      motor_frame=PILOT_RULE["motor_frame"],
                      native_substep=PILOT_RULE["native_substep"], eye="row128")


def _look(env, foods):
    """(dist row, gid row, texture row, label) for the look just taken."""
    from experiments import fast_sim as FS
    from experiments import w0 as W
    w = env.w
    dist = np.array(w._row_dist, dtype=np.float32)
    gid = np.array(w._row_gid, dtype=np.int16)
    miss = (dist < 0) | (dist > W.RAY_MAX_RANGE) | (gid < 0)
    d = np.where(miss, 1.0, dist / W.RAY_MAX_RANGE).astype(np.float16)
    gid = np.where(miss, -1, gid).astype(np.int16)
    pos = np.array(w.data.xpos[w.rover_bid], dtype=float)
    pos[2] += W.RAY_Z_OFFSET
    best = None
    for g in foods:
        v = np.array(w.data.geom_xpos[g], dtype=float) - pos
        r = math.hypot(v[0], v[1])
        if r <= W.RAY_MAX_RANGE and (best is None or r < best[0]):
            best = (r, v)
    if best is None:
        lab = (np.nan, np.nan)
    else:
        b = math.atan2(best[1][1], best[1][0]) - FS.body_yaw(w)
        lab = (math.cos(b), math.sin(b))
    return d, gid, lab


def _row_offset(o, d) -> int:
    """Where the [d, texture] x128 row sits in the flat observation: the one
    offset whose even entries equal this look's distances (asserted unique)."""
    hits = [k for k in range(0, o.size - 255)
            if np.allclose(o[k:k + 256:2], d.astype(np.float32), atol=2e-3)]
    if len(hits) != 1:
        raise RuntimeError(f"row offset not unique: {hits}")
    return hits[0]


def log_world(seed: int, n: int, out: str | None):
    from experiments import fast_sim as FS
    from experiments import world_card as WC
    env = make_env(seed)
    o, info = env.reset(seed=seed)
    w = env.w
    foods = [int(g) for k, (g, _) in w.drives._food.items() if k in env.floor_foods]
    kind = FS.kind_table(w, env.floor_foods).astype(np.int8)
    pol = WC.Held(seed, HOLD)
    pol.new_life()
    D, G, T, L = [], [], [], []
    off = None
    for _ in range(n):
        o = np.asarray(o, dtype=np.float32)
        d, gid, lab = _look(env, foods)
        if off is None:
            off = _row_offset(o, d)
        D.append(d); G.append(gid); L.append(lab)
        T.append(o[off + 1:off + 256:2].astype(np.float16))
        o, r, term, trunc, info = env.step(pol.act(o))
        if term or trunc:
            o, info = env.reset()
            pol.new_life()
    rec = dict(dist=np.stack(D), gid=np.stack(G), tex=np.stack(T),
               label=np.array(L, dtype=np.float32), kind_of=kind,
               foods=np.array(foods, dtype=np.int16))
    if out:
        np.savez_compressed(os.path.join(out, f"w{seed}.npz"), **rec)
    return rec


# ---- the probe (frozen in the docstring above; nothing here changes it) ----

def load(looks: str, worlds):
    recs = [dict(np.load(os.path.join(looks, f"w{s}.npz"))) for s in worlds]
    return recs


def forms(rec):
    """R, C, K, B inputs and labels for one world's labelled decisions."""
    from experiments import fast_sim as FS
    ok = np.isfinite(rec["label"][:, 0])
    d = rec["dist"][ok].astype(np.float32)
    gid = rec["gid"][ok].astype(np.int64)
    tex = rec["tex"][ok].astype(np.float32)
    n = d.shape[0]
    R = np.stack([d, tex], axis=2).reshape(n, 256)
    food = np.isin(gid, rec["foods"].astype(np.int64))
    texc = np.where(food, FOOD_CODE, tex)
    C = np.stack([d, texc], axis=2).reshape(n, 256)
    kind = np.zeros((n, 128, FS.N_KINDS), dtype=np.float32)
    hit = gid >= 0
    ki = rec["kind_of"].astype(np.int64)[np.where(hit, gid, 0)]
    ii, jj = np.nonzero(hit)
    kind[ii, jj, ki[ii, jj]] = 1.0
    K = np.concatenate([d[:, :, None], kind], axis=2).reshape(n, 128 * (1 + FS.N_KINDS))
    B = np.stack([FS.bins16(k) for k in K])
    return dict(R=R, C=C, K=K, B=B), rec["label"][ok].astype(np.float32)


def stack(recs):
    X, Y = {}, []
    for r in recs:
        f, y = forms(r)
        for k, v in f.items():
            X.setdefault(k, []).append(v)
        Y.append(y)
    return {k: np.concatenate(v) for k, v in X.items()}, np.concatenate(Y)


def probe(Xtr, Ytr, Xte, Yte, seed: int, epochs: int = 20) -> float:
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 256), torch.nn.ReLU(),
                              torch.nn.Linear(256, 256), torch.nn.ReLU(),
                              torch.nn.Linear(256, 2))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    for _ in range(epochs):
        order = torch.from_numpy(rng.permutation(len(Xtr)))
        for i in range(0, len(Xtr), 256):
            b = order[i:i + 256]
            loss = torch.nn.functional.mse_loss(net(xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = net(torch.from_numpy(Xte)).numpy()
    sse = ((p - Yte) ** 2).sum()
    sst = ((Yte - Yte.mean(axis=0)) ** 2).sum()
    return float(1.0 - sse / sst)


def fit(looks: str, out: str, epochs: int = 20, cap: int | None = None):
    import torch
    torch.set_num_threads(1)
    Xtr, Ytr = stack(load(looks, TRAIN))
    Xte, Yte = stack(load(looks, TEST))
    if cap:                                   # smoke only; never the reading
        Xtr = {k: v[:cap] for k, v in Xtr.items()}; Ytr = Ytr[:cap]
    res = dict(n_train=int(len(Ytr)), n_test=int(len(Yte)), epochs=epochs,
               cap=cap, r2={}, control_r2=[])
    for form in ("R", "C", "K", "B"):
        res["r2"][form] = []
        for s in SEEDS:
            t0 = time.process_time()
            r2 = probe(Xtr[form], Ytr, Xte[form], Yte, s, epochs)
            res["r2"][form].append(round(r2, 4))
            print(json.dumps({"form": form, "seed": s, "r2": round(r2, 4),
                              "cpu_s": round(time.process_time() - t0, 1)}), flush=True)
    for s in SEEDS:
        perm = np.random.default_rng(s).permutation(len(Ytr))
        r2 = probe(Xtr["K"], Ytr[perm], Xte["K"], Yte, s, epochs)
        res["control_r2"].append(round(r2, 4))
        print(json.dumps({"form": "K-permuted", "seed": s, "r2": round(r2, 4)}), flush=True)
    m = {k: float(np.mean(v)) for k, v in res["r2"].items()}
    res["mean"] = {k: round(v, 4) for k, v in m.items()}
    res["control_mean"] = round(float(np.mean(res["control_r2"])), 4)
    res["void"] = bool(max(res["control_r2"]) > 0.05)
    res["P1_R_below_0.10"] = bool(m["R"] < 0.10)
    res["P2_K_B_above_0.80"] = bool(m["K"] > 0.80 and m["B"] > 0.80)
    res["R_passes_0.80_diagnosis_wrong"] = bool(m["R"] > 0.80)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res), flush=True)
    return res


def main():
    cmd = sys.argv[1]
    if cmd == "time":
        n = int(sys.argv[2])
        t0 = time.process_time()
        rec = log_world(WORLDS[0], n, None)
        dt = time.process_time() - t0
        lab = np.isfinite(rec["label"][:, 0]).mean()
        print(json.dumps({"decisions": n, "cpu_s": round(dt, 2),
                          "cpu_s_per_decision": round(dt / n, 5),
                          "est_cpu_h_200k": round(dt / n * 200_000 / 3600, 2),
                          "labelled_frac": round(float(lab), 3)}))
    elif cmd == "log":
        out = sys.argv[2]
        os.makedirs(out, exist_ok=True)
        for s in [int(x) for x in sys.argv[3:]] or WORLDS:
            if os.path.exists(os.path.join(out, f"w{s}.npz")):
                continue
            log_world(s, PER_WORLD, out)
            print(f"w{s} done", flush=True)
    elif cmd == "fit":                       # fit LOOKS OUT [EPOCHS CAP] (smoke only)
        fit(sys.argv[2], sys.argv[3], *(int(x) for x in sys.argv[4:6]))
    else:
        raise SystemExit(f"unknown {cmd!r}")


if __name__ == "__main__":
    main()
