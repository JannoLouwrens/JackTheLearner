"""SM.03 vis_open probe — is the alive-proof dead because of the SPLIT, the
VENUE, or the READOUT?

NOT A SPEC. It writes nothing to the ledger, no seed is spent, no gate is
frozen, no constant is moved, and nothing depends on it. `SM.03` is
PILOT-BLOCKED and `coverage`'s own repair note forbids another pilot; this is
not one. It exists because the Review REFUSED to pick `SM.03`'s repair arm on
2026-09-12 (`bc9c5ec`, `sm03-heldout-split-saturated`, DUE 2026-09-15) and
ordered a measurement instead, for a reason worth restating: all three offered
F1 arms act on the held-out split's geometry and **none of them touches `F2`**,
whose failure VOIDs the registered run whichever arm wins.

    F2, as measured by the seed-90 pilot (2026-08-30, /data/sm03_pilot_seed90.json):
        acc_vis_open = 0.1167   against VIS_OPEN_MIN 0.60, with CHANCE = 0.125.
    The instrument-liveness leg of the claim is BELOW chance. Until it is alive,
    `vis_occ` 0.1167 says nothing at all about occlusion.

THE ORDERED QUESTION, verbatim in substance: *is `vis_open` at chance because
the visual observation carries no usable signal at this geometry, or because the
retained test set is too small and too degenerate to measure one?*

WHAT IS REPORTED. The three numbers the ruling names, plus two it did not — and
the two extras are here because without them the third ordered number cannot be
read in the direction the ruling wants to read it. Each is justified at its
line below.

  1. `n_test` retained, and the rejection decomposition.
  2. the per-class confusion of the OPEN-condition readout (8x8, bearing bins).
  3. `vis_open` recomputed on a test split built WITHOUT the `MIN_SEP_M`
     exclusion — the discriminator. An instrument-only reading, NEVER a bar to
     pass, and it is deliberately CONTAMINATED: dropping the exclusion admits
     near-duplicates of training positions, so it is an UPPER BOUND on what the
     shipped readout could extract at this geometry, not an accuracy anyone may
     claim. A bound is exactly what the question needs — if the readout cannot
     clear chance even with leakage allowed, no split geometry saves it.
  4. (EXTRA) the readout's TRAIN fit on its own 480 rows. Free — one forward
     pass on data already in memory. It separates "cannot generalise" from
     "cannot represent the task at all", and those two route to completely
     different desks. A readout at chance ON ITS OWN TRAINING SET is not
     evidence about the venue.
  5. (EXTRA) a spatially-explicit REFERENCE readout (ridge on 4x-pooled raw
     pixels, dual form, closed-form, no optimiser), and a count of
     source-coloured pixels per open panorama. Justification, and it is the
     reason this probe is worth its CPU: `_make_cnn` ends
     `AdaptiveAvgPool2d(1)` -> `Flatten` -> `Linear(64, 8)`, i.e. it GLOBALLY
     AVERAGES over the 8x8 feature map before the classifier, while the label
     is the source's BEARING — a purely spatial quantity. A global average is
     translation-invariant by construction, so within-frame bearing cannot
     survive it; at best "which of the 4 concatenated frames holds the ball"
     survives as channel content, which is 4-way information for an 8-way
     label. **That is a hypothesis about the INSTRUMENT, and if it is right
     then a chance reading in (3) is not a venue fact and must not be routed as
     one.** The reference is a DIAGNOSTIC, explicitly NOT a proposed arm and
     explicitly not this file's pick of the repair: picking is the Review's and
     this file states only what is measurable.

SELF-VALIDATION, and it is the first thing to read in the output. This probe
rebuilds the pilot's split from the parent module's own `_draw_layout` with the
parent's RNG seeding, so the retained positions, headings and labels are the
PILOT'S, not a re-approximation — and it renders only the OPEN panorama (the
odour field, the placebo and the occluded panorama are not needed here and are
the pilot's expensive half). If `cnn_vis_open_excl` does not reproduce the
pilot's 0.1167, the probe is measuring a different object and every other
number in it is void. Say so rather than reading on. (The lesson behind this
paragraph is `lg03_blind_twin_probe.py`'s: an approximation that reorders which
seed is worst turns a probe into a second opinion about a different rig.)

NOTHING IS MOVED. Every geometry, bar and budget is imported from
`sm_03_nose_reports_occluded` rather than restated, so this file cannot drift
from the spec it diagnoses and cannot silently soften anything. `MIN_SEP_M`,
`VIS_OPEN_MIN`, `N_TRAIN_L`, `SRC_R_RANGE` and `_GATES_FROZEN` are untouched;
`run()` keeps refusing; the pilot is spent evidence and is not re-run.

RUN IT:
    /data/venvs/jackthelearner/bin/python -m experiments.tests.sm03_vis_open_probe

Artifact: /data/sm03_vis_open_probe.json

--------------------------------------------------------------------------
PROBE RESULT — RAN 2026-09-13 00:2x-00:3x UTC, seed 90, CPU, 141.2 s,
head 8b6480a (this file, committed BEFORE the run with all three branches
declared). THE THIRD BRANCH FIRED, AND IT IS THE ONE THE RULING DID NOT HAVE.
--------------------------------------------------------------------------

SELF-VALIDATION FIRST: `cnn_vis_open_excl` **0.1167**, the pilot's recorded
`acc_vis_open` **0.1167**, to the last digit. The split is the pilot's and every
number below is about the pilot's object.

    chance 0.1250 | VIS_OPEN_MIN 0.60 | N_BINS 8 | n_train 480 | n_test 240

  (1) `n_test` retained = **240 of 240 asked** — the retention is FIXED by
      `_build_split`'s loop, which redraws until it has `n`. Reject rate 0.9964
      (66,434 rejects) WITH the exclusion vs **0.2258** (70 rejects) without it,
      the second reproducing the pilot's decomposed occlusion-only 0.2405.
      **So the "too small" half of the ordered question is refuted by
      construction, not by measurement: the test set is 240 rows either way.**
      What the exclusion buys is not fewer rows, it is worse ones — nearest
      training position per test row, median **0.2822 m**, max **0.3483 m**,
      against 45°-wide bins that subtend ~1.7 m of arc at 2.2 m radius.

  (2) THE CONFUSION IS A COLUMN, NOT A DIAGONAL. Prediction histogram
      **[240, 0, 0, 0, 0, 0, 0, 0]** — every one of the 240 test rows is
      assigned to bin 0. And then the arithmetic that matters: bin 0's base
      rate in that split is 28/240 = **0.11666…**, which IS the recorded
      0.1167. **`acc_vis_open` was never a chance-level discrimination; it is
      the base rate of whichever bin a CONSTANT readout happens to emit.** It
      carries no information about the venue whatsoever. (Same on the
      no-exclusion split: predictions [240, 0, …], accuracy 0.1042 = 25/240.)

  (3) THE DISCRIMINATOR: `vis_open` WITHOUT the `MIN_SEP_M` exclusion =
      **0.1042**, against **0.1167** with it. Lifting the saturated exclusion
      — and admitting near-duplicates, every test row now within 0.2465 m of a
      training position, median 0.0669 m, minimum 0.0063 m — moves the reading
      **DOWN, by less than one bin's worth of base rate**. Leakage deliberately
      allowed and the readout still cannot use it. **F2 IS NOT A SYMPTOM OF F1.**

  (4) The readout's fit ON ITS OWN 480 TRAINING ROWS = **0.1646** (chance
      0.1250, chosen lr 1e-4 of the spec's own grid). A 28,472-parameter CNN
      does not overfit 480 rows in 40 epochs unless it cannot express the
      mapping at all.

  (5) THE REFERENCE, AND IT IS THE WHOLE ANSWER. Closed-form ridge on 4x-pooled
      raw pixels, lambda swept on the spec's own 1-in-5 validation split
      (chosen 0.1, val 0.9792): **train 1.0000, held-out-WITH-exclusion
      0.9917 (238 of 240), no-exclusion 0.9833.** The source is visible on
      **every** layout (source-coloured pixels per panorama: train mean 7.12,
      min 2, max 16; zero on 0.0000 of layouts). Canary 165 -> 165 colours,
      stable.

**THE ANSWER TO THE ORDERED QUESTION, stated in its own terms.** The visual
observation at this geometry does not merely carry "a usable signal" — it
carries the bearing almost perfectly, and a LINEAR map recovers it at 99.17% on
the very held-out split the pilot called saturated. `vis_open` is dead because
the SHIPPED READOUT is architecturally blind to the quantity its own label
encodes, and for no other reason.

**THE MECHANISM, algebra first because the measurement alone could not separate
"cannot represent" from "cannot optimise".** `_make_cnn` ends
`AdaptiveAvgPool2d(1) -> Flatten -> Linear(64, N_BINS)`: the 8x8 feature map is
GLOBALLY AVERAGED before the classifier. A global average is translation-
invariant, so two layouts that differ only in WHERE the ball sits — which is
exactly what differing bearing bins are — map to near-identical pooled vectors,
up to boundary and shading effects. The one cue that does survive is which of
the 4 channel-concatenated frames holds the ball, i.e. 4-way quadrant
information for an 8-way label (ceiling ~0.5, not 0.125). Measured, not even
that is extracted, and the reason is dilution: ~7 source pixels in 4,096 per
frame is ~0.17% of the average the classifier reads. **Two independent grounds,
one algebraic and one measured, and they agree.**

**WHAT THIS PROBE DOES NOT DO, deliberately.** It does not pick the repair —
that is the Review's, and this file states only what is measurable. But the
measurement does bound the pick: **none of the three offered F1 arms touches
the cause**, which is the same finding the ruling itself made one level down,
now reproduced one level further in. And it warns of the second-order effect
the pick must price: `vis_occ` and `vis_open` share `_make_cnn`, so any repair
to the readout applies to the OCCLUDED arm too and makes the
`vis_occ <= VIS_OCC_CEIL` conjunct HARDER to satisfy, not easier. That is the
legal direction under the never-weaken law and it is the direction that makes
the occlusion claim mean something — but it means the repaired rig must be
expected to be a stricter test of the claim, not a friendlier one.

**F1 IS NOT EXONERATED BY ANY OF THIS — it is independently confirmed from the
other side.** A 99.17% held-out reading on a split whose median test position is
0.28 m from a training position, against bins ~1.7 m of arc wide, is precisely
what "not a held-out sample of the source band" looks like when a competent
readout is pointed at it. F1 and F2 are two faults, not one: F2's cause is the
readout, and F1 remains a real defect in what the claim would be measuring even
after F2 is repaired.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from . import sm_03_nose_reports_occluded as S

ARTIFACT = "/data/sm03_vis_open_probe.json"

# Diagnostic reference only — a ridge on 4x-average-pooled raw pixels, chosen
# because it is closed-form (no optimiser, so nothing about the reading depends
# on a schedule) and SPATIALLY EXPLICIT (flattening preserves position, which is
# the whole quantity the label is about). The pool is what makes the dual solve
# cheap; the grid is swept on the parent's own 1-in-5 internal validation split,
# the same protocol the spec's arms use, so the reference is not hand-tuned.
POOL = 4
RIDGE_GRID = (1e-1, 1e0, 1e1, 1e2, 1e3)

# "source-coloured": the ball is rgba 0.95 0.25 0.10, the floor 0.35 0.42 0.35,
# the panels 0.45 0.40 0.35 — red-dominant by a wide margin is unambiguous.
RED_MIN = 0.50
RED_GAP = 0.25


def _say(msg: str) -> None:
    print(f"[sm03-probe] {msg}", flush=True)


def _open_split(seed: int, n: int, base_offset: int, avoid: list) -> dict:
    """The parent's split, OPEN panorama only.

    `_draw_layout` is the parent's, called in the parent's order with the
    parent's RNG seeding, so positions/headings/labels are byte-for-byte the
    pilot's for the same (seed, n, base_offset, avoid). The odour window, the
    placebo window and the occluded panorama are skipped — they cost the bulk
    of the pilot's 8 minutes and none of them enters `acc_vis_open`.
    """
    m_occ, d_occ, _ = S._world(True)          # occlusion assert lives here
    m_open, d_open, r_open = S._world(False)  # ...the frames come from here
    rng = np.random.RandomState(seed * 1_000_003 + base_offset)
    X, y, pos_list, red = [], [], [], []
    rejected = 0
    for i in range(n):
        pos, h0, label, rej = S._draw_layout(rng, m_occ, d_occ, avoid)
        rejected += rej
        S._move_source(m_open, d_open, pos)
        frame = S._panorama(r_open, d_open, h0)
        X.append(frame)
        y.append(label)
        pos_list.append((pos[0], pos[1]))
        red.append(_red_pixels(frame))
        if (i + 1) % 120 == 0:
            _say(f"  ...{i + 1}/{n} layouts (rejected {rejected} so far)")
    return {"vis_open": np.stack(X), "y": np.asarray(y, dtype=np.int64),
            "positions": pos_list, "rejected": rejected,
            "red_pixels": np.asarray(red, dtype=np.int64)}


def _red_pixels(frame: np.ndarray) -> int:
    """Source-coloured pixels across the 4 frames of one panorama.

    `frame` is (12, IMG, IMG) = 4 frames x RGB. Zero here means the eye cannot
    see the ball at all at this range and resolution, which would make the
    readout question moot — so it is measured rather than assumed.
    """
    f = frame.reshape(4, 3, S.IMG, S.IMG)
    r, g, b = f[:, 0], f[:, 1], f[:, 2]
    return int(np.sum((r > RED_MIN) & (r - g > RED_GAP) & (r - b > RED_GAP)))


def _confusion(pred: np.ndarray, y: np.ndarray) -> list:
    c = np.zeros((S.N_BINS, S.N_BINS), dtype=int)
    for t, p in zip(y, pred):
        c[int(t), int(p)] += 1
    return c.tolist()


def _predict(torch, net, X, dev) -> np.ndarray:
    with torch.no_grad():
        out = []
        for j in range(0, X.shape[0], 256):
            out.append(net(X[j:j + 256].to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def _pooled(X: np.ndarray) -> np.ndarray:
    """(N, 12, IMG, IMG) -> (N, 12*(IMG/POOL)^2), spatial layout PRESERVED."""
    n, c, h, w = X.shape
    p = X.reshape(n, c, h // POOL, POOL, w // POOL, POOL).mean(axis=(3, 5))
    return p.reshape(n, -1).astype(np.float64)


def _ridge_reference(Xtr, ytr, tests: dict) -> dict:
    """Closed-form multi-output ridge on pooled pixels, dual form (n < d).

    Lambda is chosen on the parent's own 1-in-5 internal validation split — the
    spec's LR-selection protocol, so the reference gets no tuning advantage the
    shipped arms did not get.
    """
    A = _pooled(Xtr)
    mu = A.mean(axis=0, keepdims=True)
    A = A - mu
    Y = np.eye(S.N_BINS)[ytr]
    val = np.arange(A.shape[0]) % 5 == 0
    fit = ~val

    def solve(Xf, Yf, lam):
        K = Xf @ Xf.T
        alpha = np.linalg.solve(K + lam * np.eye(K.shape[0]), Yf)
        return Xf.T @ alpha

    best_lam, best = None, -1.0
    for lam in RIDGE_GRID:
        W = solve(A[fit], Y[fit], lam)
        acc = float(np.mean((A[val] @ W).argmax(1) == ytr[val]))
        if acc > best:
            best, best_lam = acc, lam
    W = solve(A, Y, best_lam)
    out = {"lam": best_lam, "val_acc": round(best, 4),
           "train_acc": round(float(np.mean((A @ W).argmax(1) == ytr)), 4)}
    for name, (Xte, yte) in tests.items():
        pred = ((_pooled(Xte) - mu) @ W).argmax(1)
        out[name] = round(float(np.mean(pred == yte)), 4)
        out[f"{name}_confusion"] = _confusion(pred, yte)
    return out


def probe(seed: int = S.PILOT_SEED) -> dict:
    from ..render import ensure_gl
    ensure_gl()
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m_occ, d_occ, r_occ = S._world(True)
    colors0, canary0 = S._canary(r_occ, m_occ, d_occ)

    t0 = time.time()
    _say(f"seed {seed}: TRAIN split ({S.N_TRAIN_L} layouts, open frames only)")
    tr = _open_split(seed, S.N_TRAIN_L, 0, avoid=[])
    _say(f"seed {seed}: TEST split WITH the MIN_SEP_M={S.MIN_SEP_M} exclusion "
         f"({S.N_TEST_L} layouts) — the pilot's split")
    te = _open_split(seed, S.N_TEST_L, 500_000, avoid=tr["positions"])
    _say(f"seed {seed}: TEST split WITHOUT the exclusion ({S.N_TEST_L} "
         f"layouts) — instrument-only, contaminated on purpose")
    te_nx = _open_split(seed, S.N_TEST_L, 500_000, avoid=[])

    min_sep = min(math.hypot(tx - ax, ty - ay)
                  for tx, ty in te["positions"] for ax, ay in tr["positions"])
    min_sep_nx = min(math.hypot(tx - ax, ty - ay)
                     for tx, ty in te_nx["positions"]
                     for ax, ay in tr["positions"])
    # Per-test-position nearest training position: F1 says every retained
    # position sits exactly AT the floor. That is a distribution claim, so the
    # distribution is reported rather than its minimum alone.
    nn_excl = [min(math.hypot(tx - ax, ty - ay) for ax, ay in tr["positions"])
               for tx, ty in te["positions"]]
    nn_noex = [min(math.hypot(tx - ax, ty - ay) for ax, ay in tr["positions"])
               for tx, ty in te_nx["positions"]]

    T = torch.from_numpy
    ytr, yte, yte_nx = T(tr["y"]), T(te["y"]), T(te_nx["y"])
    _say(f"seed {seed}: fitting the SHIPPED vision-open arm "
         f"(_make_cnn, seed {seed + 3}, the spec's own protocol)")
    acc_excl, lr, net = S._fit_arm(torch, S._make_cnn, seed + 3,
                                   T(tr["vis_open"]), ytr,
                                   T(te["vis_open"]), yte, dev)
    acc_noex = S._acc(torch, net, T(te_nx["vis_open"]), yte_nx, dev)
    acc_train = S._acc(torch, net, T(tr["vis_open"]), ytr, dev)
    _say(f"seed {seed}: cnn excl={acc_excl:.4f} noexcl={acc_noex:.4f} "
         f"train_fit={acc_train:.4f} lr={lr}")

    pred_excl = _predict(torch, net, T(te["vis_open"]), dev)
    pred_noex = _predict(torch, net, T(te_nx["vis_open"]), dev)

    _say(f"seed {seed}: spatially-explicit ridge REFERENCE (diagnostic only)")
    ref = _ridge_reference(tr["vis_open"], tr["y"],
                           {"excl": (te["vis_open"], te["y"]),
                            "noexcl": (te_nx["vis_open"], te_nx["y"])})
    _say(f"seed {seed}: ridge excl={ref['excl']:.4f} "
         f"noexcl={ref['noexcl']:.4f} train={ref['train_acc']:.4f} "
         f"lam={ref['lam']}")

    colors1, canary1 = S._canary(r_occ, m_occ, d_occ)
    out = {
        "seed": seed,
        "chance": round(S.CHANCE, 6),
        "vis_open_min": S.VIS_OPEN_MIN,
        "pilot_acc_vis_open": 0.1167,
        "n_train": S.N_TRAIN_L,
        "n_test": int(te["y"].shape[0]),
        "n_test_noexcl": int(te_nx["y"].shape[0]),
        "rejected_train": tr["rejected"],
        "rejected_test_excl": te["rejected"],
        "rejected_test_noexcl": te_nx["rejected"],
        "reject_rate_test_excl": round(
            te["rejected"] / (te["rejected"] + S.N_TEST_L), 4),
        "reject_rate_test_noexcl": round(
            te_nx["rejected"] / (te_nx["rejected"] + S.N_TEST_L), 4),
        "min_sep_excl": round(min_sep, 4),
        "min_sep_noexcl": round(min_sep_nx, 4),
        "nn_dist_excl": {"min": round(float(np.min(nn_excl)), 4),
                         "median": round(float(np.median(nn_excl)), 4),
                         "max": round(float(np.max(nn_excl)), 4),
                         "frac_at_floor": round(float(np.mean(
                             np.asarray(nn_excl) < S.MIN_SEP_M + 1e-3)), 4)},
        "nn_dist_noexcl": {"min": round(float(np.min(nn_noex)), 4),
                           "median": round(float(np.median(nn_noex)), 4),
                           "max": round(float(np.max(nn_noex)), 4),
                           "frac_under_floor": round(float(np.mean(
                               np.asarray(nn_noex) < S.MIN_SEP_M)), 4)},
        "label_hist_train": np.bincount(tr["y"], minlength=S.N_BINS).tolist(),
        "label_hist_test_excl": np.bincount(te["y"],
                                            minlength=S.N_BINS).tolist(),
        "label_hist_test_noexcl": np.bincount(te_nx["y"],
                                              minlength=S.N_BINS).tolist(),
        "cnn_vis_open_excl": round(acc_excl, 4),
        "cnn_vis_open_noexcl": round(acc_noex, 4),
        "cnn_train_fit": round(acc_train, 4),
        "cnn_lr": lr,
        "cnn_confusion_excl": _confusion(pred_excl, te["y"]),
        "cnn_confusion_noexcl": _confusion(pred_noex, te_nx["y"]),
        "cnn_pred_hist_excl": np.bincount(pred_excl,
                                          minlength=S.N_BINS).tolist(),
        "ridge_reference": ref,
        "red_pixels_train": {
            "mean": round(float(np.mean(tr["red_pixels"])), 2),
            "min": int(np.min(tr["red_pixels"])),
            "max": int(np.max(tr["red_pixels"])),
            "frac_zero": round(float(np.mean(tr["red_pixels"] == 0)), 4)},
        "red_pixels_test_excl": {
            "mean": round(float(np.mean(te["red_pixels"])), 2),
            "min": int(np.min(te["red_pixels"])),
            "max": int(np.max(te["red_pixels"])),
            "frac_zero": round(float(np.mean(te["red_pixels"] == 0)), 4)},
        "canary_colors": colors0, "canary_colors_end": colors1,
        "canary_ok": bool(canary0 == canary1),
        "wall_s": round(time.time() - t0, 1),
        "reproduces_pilot": bool(abs(acc_excl - 0.1167) < 5e-3),
    }
    return out


def main() -> None:
    out = probe()
    Path(ARTIFACT).write_text(json.dumps(out, indent=1))
    c = out["chance"]
    print()
    print("SELF-VALIDATION — the pilot's own split, rebuilt open-frames-only:")
    print(f"  cnn_vis_open_excl {out['cnn_vis_open_excl']:.4f} vs pilot "
          f"0.1167 -> {'REPRODUCES' if out['reproduces_pilot'] else 'DOES NOT REPRODUCE — read no further'}")
    print()
    print(f"THE THREE ORDERED NUMBERS (chance {c:.4f}, VIS_OPEN_MIN "
          f"{out['vis_open_min']}):")
    print(f"  1. n_test retained            {out['n_test']} of "
          f"{S.N_TEST_L} asked  (rejects {out['rejected_test_excl']}, "
          f"rate {out['reject_rate_test_excl']})")
    print(f"  2. OPEN-condition confusion   (8x8 below; pred hist "
          f"{out['cnn_pred_hist_excl']})")
    for row in out["cnn_confusion_excl"]:
        print("       " + " ".join(f"{v:>4d}" for v in row))
    print(f"  3. vis_open WITHOUT exclusion {out['cnn_vis_open_noexcl']:.4f}"
          f"   (with exclusion {out['cnn_vis_open_excl']:.4f})")
    print()
    print("THE TWO EXTRAS, which decide how (3) may be read:")
    print(f"  4. cnn TRAIN fit on its own {out['n_train']} rows "
          f"{out['cnn_train_fit']:.4f}")
    r = out["ridge_reference"]
    print(f"  5. spatial ridge REFERENCE    train {r['train_acc']:.4f}  "
          f"excl {r['excl']:.4f}  noexcl {r['noexcl']:.4f}  (lam {r['lam']})")
    print(f"     source-coloured pixels/panorama: train mean "
          f"{out['red_pixels_train']['mean']} "
          f"(min {out['red_pixels_train']['min']}, "
          f"zero on {out['red_pixels_train']['frac_zero']:.4f} of layouts)")
    print()
    print(f"  nearest train position per test row — WITH exclusion: "
          f"{out['nn_dist_excl']}")
    print(f"                                    WITHOUT:            "
          f"{out['nn_dist_noexcl']}")
    print(f"  canary {out['canary_colors']} -> {out['canary_colors_end']} "
          f"colours, stable={out['canary_ok']}; wall {out['wall_s']} s")
    print(f"  artifact {ARTIFACT}")


if __name__ == "__main__":
    main()
