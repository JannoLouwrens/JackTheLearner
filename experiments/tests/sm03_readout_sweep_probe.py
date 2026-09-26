"""SM.03 readout sweep — F2's REPAIR, measured on one pre-registered knob.

NOT A SPEC. It writes nothing to the ledger, spends no seed, freezes no gate,
moves no constant, and nothing depends on it. `SM.03` is PILOT-BLOCKED and
`coverage`'s repair note forbids another pilot; this is not one — it renders no
odour field, fits no odour arm, and cannot produce any number the claim is made
of.

WHY IT EXISTS, and whose debt it pays. The Review's 2026-09-12 ruling on
`sm03-heldout-split-saturated` split the row in two: F1 (the saturated held-out
split) is an ARM PICK this desk does not own and which is still open at the
Review, DUE 2026-09-30; F2 (the dead alive-proof, `acc_vis_open` 0.1167 against
`VIS_OPEN_MIN` 0.60 with chance 0.1250) was *"promoted from rider to blocker and
answered by a MEASUREMENT the builder owes"*. That measurement was delivered on
2026-09-13 (`sm03_vis_open_probe.py`) and it DIAGNOSED F2 without repairing it:

    `_make_cnn` ends `AdaptiveAvgPool2d(1) -> Flatten -> Linear(64, N_BINS)`,
    so the 8x8 feature map is GLOBALLY AVERAGED before the classifier while the
    label is the source's BEARING. A global average is translation-invariant, so
    the quantity the label encodes cannot survive the pool. Measured: the
    prediction histogram is a COLUMN, [240, 0, 0, 0, 0, 0, 0, 0], and 0.1167 is
    bin 0's base rate (28/240) — the accuracy of a CONSTANT predictor, carrying
    no information about the venue. Measured on the other side: a closed-form
    LINEAR ridge on 4x-pooled RAW pixels reads the same held-out split at
    **0.9917**. The venue is near-perfectly readable; the readout is blind.

So F2's cause is known and F2's repair has been nobody's for thirteen days. It
is not the F1 arm pick, it is not a bar, and it is not a dispatch — `run()`
keeps refusing on `_GATES_FROZEN` whatever this file finds, so nothing here can
buy a certificate or pre-empt the Review's 09-30 sitting.

--------------------------------------------------------------------------
THE KNOB, AND WHY IT IS ONE KNOB AND NOT AN ARCHITECTURE ZOO
--------------------------------------------------------------------------

The diagnosis names a single mechanism — spatial information destroyed by the
pool — so the repair is swept as a single scalar: **how much spatial resolution
the classifier head is allowed to see.**

    HEAD_POOL_GRID = (1, 2, 4, 8)   # nn.AdaptiveAvgPool2d(p) on the 8x8 map

Everything else in `_make_cnn` is the parent's, unchanged: the same three conv
layers, the same `torch.manual_seed`, the same `_fit_arm` protocol (the spec's
own LR grid on its own 1-in-5 validation split), the same `EPOCHS`, `BATCH` and
`WEIGHT_DECAY`. **`p = 1` IS THE SHIPPED READOUT** and is therefore the
self-validation leg, not a candidate: it must reproduce the pilot's 0.1167.

This shape is deliberate. The 09-13 probe's finding was ALGEBRAIC — a claim that
accuracy is destroyed by translation-invariance — and a one-knob sweep is the
only reading that can falsify it. If bearing accuracy does not rise as spatial
resolution is restored, the algebra was wrong and no readout in this family is
the repair. An architecture zoo could not say that.

Parameter counts, computed not asserted (the conv stack is 27,952 of them):

    p=1    520 head    28,472 total   <- SHIPPED
    p=2  2,056 head    30,008 total
    p=4  8,200 head    36,152 total
    p=8 32,776 head    60,728 total

All four sit UNDER the odour arm's `_make_mlp` at 70,344, so no candidate can be
dismissed — or excused — as the vision arm being handed more capacity than the
nose it is the control for.

--------------------------------------------------------------------------
THE SELECTION RULE — PRE-REGISTERED HERE, BEFORE ANY NUMBER IS READ
--------------------------------------------------------------------------

**Take the `p` that MAXIMISES `vis_open`. Ties (within 1e-9) break to FEWER
parameters.**

That rule is chosen because it is the one that cannot be run-until-pass, and the
reason is worth stating in full because the opposite rule was available and is
the trap:

  - `vis_open` is the ALIVE-PROOF. It is not the claim. The claim is
    `odour_occ >= ODOUR_OCC_MIN` (0.25) **and** `vis_occ <= VIS_OCC_CEIL` (0.22).
  - `vis_open` and `vis_occ` are fitted by the SAME readout. So maximising
    `vis_open` also maximises `vis_occ` — i.e. this rule selects, out of every
    candidate, the one that makes the claim's own occlusion conjunct HARDEST to
    satisfy. The selection runs AGAINST the spec's interest by construction.
  - The rule that would have been illegitimate is *"the cheapest head that
    clears 0.60"*. It reads identically on the alive-proof and it would pick the
    WEAKEST competent readout, i.e. the one that keeps `vis_occ` lowest. That is
    venue-shopping wearing a parsimony argument, and `2^7`'s prohibition — *"the
    venue a certificate is bought on may never be selected after seeing which
    venue is kind"* — is exactly what it would breach.

**PRE-REGISTERED CONSEQUENCE, so it cannot be renegotiated after the numbers
land: if the selected head reads `vis_occ` ABOVE `VIS_OCC_CEIL` 0.22, that is a
FINDING ABOUT THE OCCLUSION PREMISE — the panels do not hide the source from a
competent eye — and it is NOT a licence to pick a weaker head.** It would mean
`SM.03` as designed cannot pass, and that verdict belongs in the row, not in the
readout. No bar moves in either direction on this file's account.

--------------------------------------------------------------------------
THE READING BRANCHES — all three declared before the run
--------------------------------------------------------------------------

  (A) SOME `p` CLEARS `VIS_OPEN_MIN` 0.60. F2's cause is confirmed to be the
      head's spatial resolution and the repair is the winning `p`. Ship it into
      `_make_cnn`; report the winner's `vis_occ` beside it because the claim now
      faces a stricter occlusion test than the one the pilot ran.

  (B) NO `p` CLEARS 0.60, BUT `vis_open` RISES WITH `p`. The algebra is
      confirmed and the family is insufficient — the pool is A cause and not the
      whole cause. ROUTE it (a second readout family, or the ridge itself as the
      alive-proof readout); do NOT ship a head that still fails the leg, and do
      NOT lower `VIS_OPEN_MIN`.

  (C) `vis_open` DOES NOT RISE WITH `p`. The 09-13 probe's algebraic finding is
      REFUTED — translation-invariance was not the binding cause — and this
      file's premise dies with it. Ship NOTHING, say so loudly, and route F2
      back with the refutation attached. A diagnosis that survives thirteen days
      because nobody tested it is worth exactly one test.

`vis_occ` is measured for EVERY `p`, not only the winner, because the ruling
named the second-order effect as something *"the pick must price"* and a single
number at the winner cannot show whether occlusion degrades gracefully or not at
all.

--------------------------------------------------------------------------
SELF-VALIDATION — read this first or read nothing
--------------------------------------------------------------------------

The split is rebuilt from the parent's own `_draw_layout`, in the parent's order,
with the parent's RNG seeding, so positions/headings/labels are the PILOT'S. The
odour and placebo windows are skipped: they are 30 s of puff pre-roll per layout
and they draw from their OWN generators (`lseed`, `placebo_rng`), never from the
layout `rng`, so omitting them cannot shift the layout sequence. `sm03_vis_open_
probe.py` already demonstrated this by reproducing 0.1167 to the last digit.

    if `vis_open[p=1]` != the pilot's 0.1167, this probe is measuring a
    DIFFERENT object and every other number in it is void. Say so and stop.

`vis_occ[p=1]` has the pilot's 0.1167 to reproduce too, and it is a second,
independent check on the same split.

NOTHING IS MOVED. Every geometry, bar, budget and protocol is imported from
`sm_03_nose_reports_occluded` rather than restated, so this file cannot drift
from the spec it diagnoses and cannot silently soften anything. `MIN_SEP_M`,
`VIS_OPEN_MIN`, `VIS_OCC_CEIL`, `N_TRAIN_L`, `SRC_R_RANGE`, `LR_GRID`, `EPOCHS`
and `_GATES_FROZEN` are untouched; `run()` keeps refusing; the pilot is spent
evidence and is not re-run.

RUN IT:
    /data/venvs/jackthelearner/bin/python -m experiments.tests.sm03_readout_sweep_probe

Artifact: /data/sm03_readout_sweep.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import sm_03_nose_reports_occluded as S

ARTIFACT = "/data/sm03_readout_sweep.json"

# The one knob. `nn.AdaptiveAvgPool2d(p)` on the parent's 8x8 feature map:
# p=1 is the shipped global average (SELF-VALIDATION, not a candidate), p=8 is
# the identity, i.e. the full spatial map flattened into the classifier.
HEAD_POOL_GRID = (1, 2, 4, 8)
SHIPPED_POOL = 1

# The parent's recorded pilot readings for the two vision arms on this split.
# Both must reproduce; they are the same number by coincidence of the constant
# readout, and they are checked separately.
PILOT_VIS_OPEN = 0.1167
PILOT_VIS_OCC = 0.1167
REPRO_TOL = 5e-3


def _say(msg: str) -> None:
    print(f"[sm03-sweep] {msg}", flush=True)


def _head_factory(pool: int):
    """The parent's `_make_cnn` with ONE line changed: the head's pool size.

    Written as a closure over `pool` rather than as four hand-copied networks so
    that a future edit to the parent's conv stack cannot silently apply to some
    candidates and not others — the only difference between arms is the knob.
    """
    def make(torch, nn, seed: int, dev):
        torch.manual_seed(seed)
        return nn.Sequential(
            nn.Conv2d(12, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(pool), nn.Flatten(),
            nn.Linear(64 * pool * pool, S.N_BINS)).to(dev)
    return make


def _shipped_is_pool_one(torch) -> bool:
    """Guard: this file claims `p=1` IS the shipped readout. Check it.

    A sweep whose self-validation leg is not actually the incumbent proves
    nothing about the incumbent. Compared by parameter shapes and count, which
    is what distinguishes the heads.
    """
    a = S._make_cnn(torch, torch.nn, 0, "cpu")
    b = _head_factory(SHIPPED_POOL)(torch, torch.nn, 0, "cpu")
    sa = [tuple(p.shape) for p in a.parameters()]
    sb = [tuple(p.shape) for p in b.parameters()]
    return sa == sb


def _both_panoramas(seed: int, n: int, base_offset: int, avoid: list) -> dict:
    """The parent's split, BOTH panoramas, no odour.

    Mirrors `_build_split`'s structure and call order exactly — including
    building a fresh occluded and open world per split, as it does — minus the
    two `_odour_window` calls, which draw from their own generators and so
    cannot shift the layout sequence.
    """
    m_occ, d_occ, r_occ = S._world(True)
    m_open, d_open, r_open = S._world(False)
    rng = np.random.RandomState(seed * 1_000_003 + base_offset)
    X_vo, X_vn, y, pos_list = [], [], [], []
    rejected = 0
    for i in range(n):
        pos, h0, label, rej = S._draw_layout(rng, m_occ, d_occ, avoid)
        rejected += rej
        S._move_source(m_occ, d_occ, pos)
        X_vo.append(S._panorama(r_occ, d_occ, h0))
        S._move_source(m_open, d_open, pos)
        X_vn.append(S._panorama(r_open, d_open, h0))
        y.append(label)
        pos_list.append((pos[0], pos[1]))
        if (i + 1) % 120 == 0:
            _say(f"  ...{i + 1}/{n} layouts (rejected {rejected} so far)")
    # The renderers are returned so the caller can hold them for the process
    # lifetime: a garbage-collected mujoco.Renderer poisons the shared X display
    # and the NEXT one returns corrupted-but-realistic frames (render.py, PG.6).
    return {"vis_occ": np.stack(X_vo), "vis_open": np.stack(X_vn),
            "y": np.asarray(y, dtype=np.int64), "positions": pos_list,
            "rejected": rejected, "_hold": (r_occ, r_open)}


def _confusion(pred: np.ndarray, y: np.ndarray) -> list:
    c = np.zeros((S.N_BINS, S.N_BINS), dtype=int)
    for t, p in zip(y, pred):
        c[int(t), int(p)] += 1
    return c.tolist()


def _pred_hist(torch, net, X, dev) -> list:
    with torch.no_grad():
        out = []
        for j in range(0, X.shape[0], 256):
            out.append(net(X[j:j + 256].to(dev)).argmax(1).cpu().numpy())
    p = np.concatenate(out)
    return np.bincount(p, minlength=S.N_BINS).tolist(), p


def probe(seed: int = S.PILOT_SEED) -> dict:
    from ..render import ensure_gl
    ensure_gl()
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if not _shipped_is_pool_one(torch):
        raise RuntimeError(
            "p=1 is NOT the shipped `_make_cnn` head — the self-validation leg "
            "of this sweep is invalid and the parent must have moved. Read the "
            "parent before trusting any number here.")

    m_can, d_can, r_can = S._world(True)
    colors0, canary0 = S._canary(r_can, m_can, d_can)

    t0 = time.time()
    _say(f"seed {seed}: TRAIN split ({S.N_TRAIN_L} layouts, both panoramas)")
    tr = _both_panoramas(seed, S.N_TRAIN_L, 0, avoid=[])
    _say(f"seed {seed}: TEST split, MIN_SEP_M={S.MIN_SEP_M} exclusion "
         f"({S.N_TEST_L} layouts) — the pilot's split")
    te = _both_panoramas(seed, S.N_TEST_L, 500_000, avoid=tr["positions"])
    _say(f"seed {seed}: splits built in {time.time() - t0:.1f} s")

    T = torch.from_numpy
    ytr, yte = T(tr["y"]), T(te["y"])
    arms = {}
    for pool in HEAD_POOL_GRID:
        make = _head_factory(pool)
        n_par = sum(p.numel()
                    for p in make(torch, torch.nn, 0, "cpu").parameters())
        row = {"pool": pool, "n_params": n_par}
        # vis_open uses the parent's seed+3, vis_occ its seed+2 — the spec's own
        # per-arm seeding, so a candidate is never handed a different draw.
        for cond, arm_seed in (("vis_open", seed + 3), ("vis_occ", seed + 2)):
            ts = time.time()
            acc, lr, net = S._fit_arm(torch, make, arm_seed,
                                      T(tr[cond]), ytr, T(te[cond]), yte, dev)
            fit = S._acc(torch, net, T(tr[cond]), ytr, dev)
            hist, pred = _pred_hist(torch, net, T(te[cond]), dev)
            row[cond] = round(acc, 4)
            row[f"{cond}_train_fit"] = round(fit, 4)
            row[f"{cond}_lr"] = lr
            row[f"{cond}_pred_hist"] = hist
            row[f"{cond}_confusion"] = _confusion(pred, te["y"])
            row[f"{cond}_wall_s"] = round(time.time() - ts, 1)
            _say(f"  pool={pool} {cond}={acc:.4f} (train fit {fit:.4f}, "
                 f"lr {lr}, {row[f'{cond}_wall_s']} s)")
        arms[pool] = row

    # ---- the pre-registered rule, applied mechanically ----
    cands = [p for p in HEAD_POOL_GRID]
    best = max(cands, key=lambda p: (round(arms[p]["vis_open"], 9),
                                     -arms[p]["n_params"]))
    opens = [arms[p]["vis_open"] for p in HEAD_POOL_GRID]
    rises = opens[-1] > opens[0] + 1e-9 and max(opens) > opens[0] + 1e-9
    clears = [p for p in HEAD_POOL_GRID
              if arms[p]["vis_open"] >= S.VIS_OPEN_MIN]
    if clears:
        branch = "A"
    elif rises:
        branch = "B"
    else:
        branch = "C"

    colors1, canary1 = S._canary(r_can, m_can, d_can)
    return {
        "seed": seed,
        "chance": round(S.CHANCE, 6),
        "vis_open_min": S.VIS_OPEN_MIN,
        "vis_occ_ceil": S.VIS_OCC_CEIL,
        "n_train": S.N_TRAIN_L,
        "n_test": int(te["y"].shape[0]),
        "rejected_train": tr["rejected"],
        "rejected_test": te["rejected"],
        "pilot_vis_open": PILOT_VIS_OPEN,
        "pilot_vis_occ": PILOT_VIS_OCC,
        "reproduces_pilot_vis_open": bool(
            abs(arms[SHIPPED_POOL]["vis_open"] - PILOT_VIS_OPEN) < REPRO_TOL),
        "reproduces_pilot_vis_occ": bool(
            abs(arms[SHIPPED_POOL]["vis_occ"] - PILOT_VIS_OCC) < REPRO_TOL),
        "head_pool_grid": list(HEAD_POOL_GRID),
        "arms": {str(k): v for k, v in arms.items()},
        "vis_open_by_pool": {str(p): arms[p]["vis_open"]
                             for p in HEAD_POOL_GRID},
        "vis_occ_by_pool": {str(p): arms[p]["vis_occ"]
                            for p in HEAD_POOL_GRID},
        "selected_pool": best,
        "selected_vis_open": arms[best]["vis_open"],
        "selected_vis_occ": arms[best]["vis_occ"],
        "selected_clears_alive_proof": bool(
            arms[best]["vis_open"] >= S.VIS_OPEN_MIN),
        "selected_breaches_occ_ceil": bool(
            arms[best]["vis_occ"] > S.VIS_OCC_CEIL),
        "pools_clearing_alive_proof": clears,
        "rises_with_pool": bool(rises),
        "branch": branch,
        "canary_colors": colors0, "canary_colors_end": colors1,
        "canary_ok": bool(canary0 == canary1),
        "wall_s": round(time.time() - t0, 1),
    }


def main() -> None:
    out = probe()
    Path(ARTIFACT).write_text(json.dumps(out, indent=1))
    print()
    print("SELF-VALIDATION — p=1 IS the shipped `_make_cnn` head:")
    print(f"  vis_open[p=1] {out['arms']['1']['vis_open']:.4f} vs pilot "
          f"{out['pilot_vis_open']} -> "
          f"{'REPRODUCES' if out['reproduces_pilot_vis_open'] else 'DOES NOT REPRODUCE — read no further'}")
    print(f"  vis_occ [p=1] {out['arms']['1']['vis_occ']:.4f} vs pilot "
          f"{out['pilot_vis_occ']} -> "
          f"{'REPRODUCES' if out['reproduces_pilot_vis_occ'] else 'DOES NOT REPRODUCE — read no further'}")
    print()
    print(f"THE SWEEP (chance {out['chance']:.4f}, VIS_OPEN_MIN "
          f"{out['vis_open_min']}, VIS_OCC_CEIL {out['vis_occ_ceil']}):")
    print("  pool   n_params   vis_open   vis_occ   open_train_fit  pred_hist(open)")
    for p in out["head_pool_grid"]:
        a = out["arms"][str(p)]
        tag = "  <- SHIPPED" if p == SHIPPED_POOL else ""
        print(f"  {p:>4d}   {a['n_params']:>8d}   {a['vis_open']:>8.4f}   "
              f"{a['vis_occ']:>7.4f}   {a['vis_open_train_fit']:>14.4f}  "
              f"{a['vis_open_pred_hist']}{tag}")
    print()
    print(f"  rises with pool           {out['rises_with_pool']}")
    print(f"  pools clearing 0.60       {out['pools_clearing_alive_proof']}")
    print(f"  BRANCH                    {out['branch']}")
    print(f"  SELECTED (argmax vis_open, ties to fewer params)  pool="
          f"{out['selected_pool']}  vis_open={out['selected_vis_open']:.4f}  "
          f"vis_occ={out['selected_vis_occ']:.4f}")
    print(f"  alive-proof cleared       {out['selected_clears_alive_proof']}")
    print(f"  occlusion ceiling BREACHED by the selected head  "
          f"{out['selected_breaches_occ_ceil']}   "
          f"(a FINDING about the premise, never a reason to pick weaker)")
    print(f"  canary {out['canary_colors']} -> {out['canary_colors_end']} "
          f"colours, stable={out['canary_ok']}; wall {out['wall_s']} s")
    print(f"  artifact {ARTIFACT}")


if __name__ == "__main__":
    main()
