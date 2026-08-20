"""T3.01 — Ablate vision: removing vision measurably hurts a vision task.

HYPOTHESIS (registry). Removing vision measurably hurts a vision-dependent
task. Falsified by: no measurable drop. Null: full system. Kills: the vision
encoder. COVERS: sight (claim) — the first claim spec for the sight
commitment (15 zero-pass commitments as of 2026-08-20; standing rule).

WHAT IS MEASURED, SAID PLAINLY. The vision seat is held by the 0.24M
`PrismaticVisionEncoder` CNN fallback, which T2.03 measured AS SHIPPED
(frozen, never a gradient): its linear probe reads 0.4467-0.4933 — barely
above a random projection of the same pixels (0.36-0.45). Under the
PLASTIC-ONLY decree the seat holder is supposed to LEARN, and this spec is
the first time in this repository the vision encoder receives a gradient.
The full system here = encoder + linear head trained end-to-end on T2.03's
certified task (4-way shape classification through PG.6's certified eye,
chance 0.25); the ablation = the SAME trained system evaluated with vision
removed (every test frame replaced by the per-pixel mean of the TRAIN split
— an information-free input with in-distribution statistics). The claim:
trained accuracy clears its bar AND collapses under the ablation, on every
seed. That makes "vision is load-bearing" a measurement, not a tautology.

WHERE THE TEST CAN FAIL, HONESTLY. The task is vision-only by construction,
so the load-bearing content is the TRAIN gate, not the collapse: FAIL means
the trained system never made vision carry the task (acc_full at or near
where its own frozen features already sit, or no measurable drop because
there was nothing to drop). That is exactly the registry's falsified_by
("no measurable drop") and it kills: an encoder that cannot learn a 4-way
shape task its own frozen features nearly solve linearly has not earned its
245K parameters. A high-full/no-drop outcome (accuracy surviving a constant
input) is unreachable except through a rig defect and is VOID, not FAIL —
a deterministic net fed one constant frame predicts one constant class.

ARMS, per seed (train N=1200, test N=300, T2.03's sizes and split seeds):
  full     encoder (default config CNN, seeded init) + nn.Linear(1024,4),
           trained end-to-end, AdamW wd 1e-4, batch 64, EPOCHS epochs (v2:
           62, raised from 25 by the curves probe — see V2 REPAIR); lr chosen
           from LR_GRID on the every-5th-row train-internal split (VAL_EVERY,
           T2.03's protocol), then retrained at the chosen lr on all 1200.
           Test labels are never consulted before the single test pass.
  ablated  the SAME trained weights, test frames replaced by the train-mean
           frame. Reported diagnostic beside it (not gated): per-frame
           pixel-shuffle ablation, which destroys structure but preserves
           each frame's histogram — prices how much of acc_full is global
           statistics rather than shape.
  ref      MUST-SUCCEED reference (T1.02 / T3.07 lesson): T2.03's registered
           probe procedure (`_probe_acc`, imported — reference, don't copy)
           on the FROZEN seed-init encoder's features. T2.03's PASS certifies
           this arm's band (0.4467-0.4933 across its registered seeds); a rig
           that cannot reproduce it cannot attribute anything -> VOID.
  shuffled CONTROL, must fail: same architecture, same budget, trained on
           permuted labels; TEST accuracy must sit at chance. Clears -> the
           rig leaks episode identity -> VOID, not evidence. Its TRAIN
           accuracy on the shuffled labels is recorded and gated too
           (added 2026-08-20, 23rd audit B1): at-chance test from an arm
           that never fit its own train set is a dead arm, not a control.

PRE-REGISTERED GATES — all exogenous or loaned from T2.03's registered
certificate (nothing calibrated from a pilot; sd at chance with n=300 is
sqrt(.25*.75/300) ~= 0.025):
  RIG (VOID): canary byte-stable AND canary_colors >= 100 (blind-sensor
      trap, T2.03's constants); encoder params in [220K, 270K]; and
      acc_ref >= REF_FLOOR = 0.38 (T2.03 registered worst seed 0.4467 minus
      ~2.3 sd — below that band this rig is not the certified rig).
  TRAIN-ATTRIBUTION (VOID): acc_full >= acc_ref - TRAIN_TOL (0.05) per seed.
      End-to-end training strictly contains the frozen-probe solution (leave
      the encoder at init, fit the head); losing to your own subset is an
      optimisation defect of THIS rig, not evidence about the encoder. The
      lr grid exists to make this gate hard to trip honestly.
  CONTROL (VOID): |acc_shuffled - 0.25| <= SHUFFLE_BAND (0.10) per seed
      (T2.03's registered control read max dev 0.0633 at the same n); and
      acc_shuffled_train >= SHUFFLE_FIT_FLOOR (0.35) per seed — control
      liveness, see the constant's comment (strengthen-only, 2026-08-20).
  CLAIM (every seed, else FAIL):
      acc_full >= MIN_FULL  = 0.45   chance + 8 sd; equals the floor of the
                                     frozen-probe band — a trained encoder
                                     that cannot reach where its own frozen
                                     features sit has learned nothing
      drop     >= MIN_DROP  = 0.15   drop = acc_full - acc_ablated; the
                                     arithmetic floor implied by MIN_FULL
                                     with ablated at chance is 0.20, so 0.15
                                     tolerates ablated a shade above chance
      acc_ablated <= ABL_CEIL = 0.40 chance + 6 sd; a constant input that
                                     still beats this is a rig defect caught
                                     by the high-full/no-drop VOID above

GPU. One submission for the whole spec (module cache, T2.01 pattern). The
job pins mujoco==3.11.0 (wheel-verified 2026-08-20; 3.12.0's sdist reached
PyPI before its wheels and killed two kernels — LESSONS) and never -q's the
install. Science code lives in this module, not the JOB string (T0.16).

DRY-CHECKED verdict paths (python -m experiments.tests.t3_01_ablate_vision
dry): planted pass -> PASS; no-drop / weak-full -> FAIL; planted ref
collapse, control leak, canary drift, param drift, train<ref -> VOID.

V2 REPAIR (2026-08-20, after the attempt-2 VOID; pre-registered by the
curves probe's decision rule, experiments/t3_01_curves_probe.py). Attempt 2
(kernel jack-ladder-1787231872) VOIDed on the train-attribution gate:
acc_full [0.48, 0.39, 0.3833] vs acc_ref [0.4467, 0.4667, 0.4933],
train_vs_ref_min -0.11; seeds 1,2 collapsed a class at 25 epochs. The probe
replayed the exact failed trainings (same init and shuffle seeds) to 100
epochs and applied readings pre-stated before its launch:
  R1 BUDGET fired — at the registered run's chosen LRs the collapsed seeds
     enter the attribution band with all classes alive at epochs 28 (seed 1,
     3e-4) and 31 (seed 2, 1e-3); seed 0 (1e-3) at 24. Dead classes revive;
     nothing plateaus below band (R2 silent).
  R3 did NOT fire either way: the warmstart arm's premise failed (head fit
     on frozen TRAIN features tests at 0.31-0.36, below acc_ref, so it never
     held the band at every epoch) — and joint training IMPROVED it to
     0.69-0.78 everywhere, so there is no destructive-first-gradient
     evidence for the FAIL lane.
  REPAIR, per the rule verbatim: EPOCHS = smallest recorded epoch where all
     seeds clear the band (31), doubled for margin -> 62. Uniform, control
     included; every gate unchanged; the VOID stays in history.
ONE-DIAGNOSTIC CAP (pre-stated at probe dispatch): if this repaired run
VOIDs on attribution again, T3.01 is PARKED per the SM.02 / overseer-B5
rule — the next move would need a new mechanism-level reason, not a lottery.

COVERS: sight (claim).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..gpu import build_job, submit

# T2.03's rig is the substrate: its file hashes into this certificate, so a
# change there stales this claim loudly rather than silently.
IMPL_DEPS = ["playground.py", "UnifiedBrain.py",
             "experiments/tests/pg_6_playground_eyes.py",
             "experiments/tests/t2_03_pretrained_vision.py"]

SEEDS = [0, 1, 2]
PILOT_SEED = 90

# Training budget (pre-registered; per-coordinate reasoning in the docstring).
LR_GRID = (1e-4, 3e-4, 1e-3)
EPOCHS = 62  # v2: was 25; probe R1 — all seeds in band by 31, doubled (see docstring)
BATCH = 64
WEIGHT_DECAY = 1e-4

# Gates (docstring gives every anchor's provenance).
REF_FLOOR = 0.38
TRAIN_TOL = 0.05
MIN_FULL = 0.45
MIN_DROP = 0.15
ABL_CEIL = 0.40
SHUFFLE_BAND = 0.10
# Control-liveness floor (23rd audit, B1; added 2026-08-20, gates only runs
# from here on): the shuffled arm's accuracy on its OWN shuffled train set.
# Chance is 0.25; a same-budget net that cannot exceed chance + 0.10 on data
# it was fitted to did not train, and its at-chance test reading proves
# nothing about leakage.
SHUFFLE_FIT_FLOOR = 0.35


# ── training (runs wherever the job runs) ────────────────────────────────
def _make_model(seed: int, device: str):
    import torch
    from UnifiedBrain import UnifiedBrainConfig, PrismaticVisionEncoder
    torch.manual_seed(seed)
    enc = PrismaticVisionEncoder(UnifiedBrainConfig())
    head = torch.nn.Linear(1024, 4)
    model = torch.nn.Sequential(enc, head).to(device)
    n_params_enc = int(sum(p.numel() for p in enc.parameters()))
    return model, n_params_enc


def _train_one(seed: int, lr_idx: int, imgs, y, device: str,
               epochs: int = EPOCHS):
    """One end-to-end training at LR_GRID[lr_idx]; returns the model.
    Seeded per (seed, lr_idx) so every arm is reproducible independently."""
    import torch
    lr = LR_GRID[lr_idx]
    model, _ = _make_model(seed * 13 + lr_idx, device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed * 1009 + lr_idx)
    n = len(imgs)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = imgs[idx].to(device)
            yb = y[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss (seed {seed}, lr {lr})")
            loss.backward()
            opt.step()
    model.eval()
    return model


def _acc(model, imgs, y, device: str) -> float:
    import torch
    preds = []
    with torch.no_grad():
        for i in range(0, len(imgs), 64):
            preds.append(model(imgs[i:i + 64].to(device)).argmax(1).cpu())
    return float((torch.cat(preds) == y).float().mean())


def _to_tensor(X: np.ndarray):
    import torch
    return torch.from_numpy(X).permute(0, 3, 1, 2).float().div_(255.0)


def remote_run(seeds: list, n_train: int | None = None,
               n_test: int | None = None, epochs: int = EPOCHS,
               lr_grid_idx: tuple = (0, 1, 2)) -> dict:
    """Everything for the given seeds; runs on the GPU VM (or locally, small,
    for the smoke). The reduced-size arguments exist ONLY for the smoke —
    the registered run uses the defaults."""
    import torch
    from .t2_03_pretrained_vision import (_build_dataset, _get_eye,
                                          _probe_acc, N_TRAIN, N_TEST,
                                          MIN_CANARY_COLORS, PARAMS_RANGE,
                                          CLASSES)
    n_train = n_train or N_TRAIN
    n_test = n_test or N_TEST
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "seeds": []}
    for seed in seeds:
        Xtr, ytr, _ = _build_dataset(seed, n_train)
        Xte, yte, _ = _build_dataset(seed + 500_009, n_test)  # T2.03's split
        eye = _get_eye(seed)
        itr, ite = _to_tensor(Xtr), _to_tensor(Xte)
        ttr = torch.from_numpy(ytr)
        tte = torch.from_numpy(yte)

        # ref: frozen seed-init encoder features under T2.03's probe.
        model0, n_params = _make_model(seed, device)
        enc0 = model0[0]
        with torch.no_grad():
            ftr = np.concatenate([enc0(itr[i:i + 16].to(device)).cpu().numpy()
                                  for i in range(0, n_train, 16)])
            fte = np.concatenate([enc0(ite[i:i + 16].to(device)).cpu().numpy()
                                  for i in range(0, n_test, 16)])
        acc_ref, _ = _probe_acc(ftr.astype(np.float32), ytr,
                                fte.astype(np.float32), yte)

        # lr selection on the train-internal split, then retrain on all rows.
        val = np.arange(n_train) % 5 == 0
        fit = ~val
        best_idx, best_val = None, -1.0
        val_accs = {}
        for li in lr_grid_idx:
            m = _train_one(seed, li, itr[fit], ttr[fit], device, epochs)
            va = _acc(m, itr[val], ttr[val], device)
            val_accs[str(LR_GRID[li])] = round(va, 4)
            if va > best_val:
                best_val, best_idx = va, li
        model = _train_one(seed, best_idx, itr, ttr, device, epochs)

        acc_full = _acc(model, ite, tte, device)
        per_class_min = min(
            _acc(model, ite[tte == k], tte[tte == k], device)
            for k in range(len(CLASSES)))

        # ablation: train-mean frame, one constant input per test row.
        mean_frame = itr.mean(0, keepdim=True)
        abl = mean_frame.repeat(n_test, 1, 1, 1)
        acc_abl = _acc(model, abl, tte, device)

        # diagnostic (not gated): per-frame pixel shuffle.
        rng = np.random.RandomState(seed + 271)
        sh = ite.clone().reshape(n_test, 3, -1)
        for i in range(n_test):
            perm = rng.permutation(sh.shape[-1])
            sh[i] = sh[i][:, perm]
        acc_pixshuf = _acc(model, sh.reshape_as(ite), tte, device)

        # control: shuffled-label training, same budget, chosen lr.
        ysh = ytr.copy()
        np.random.RandomState(seed + 41).shuffle(ysh)
        m_sh = _train_one(seed + 7_001, best_idx, itr,
                          torch.from_numpy(ysh), device, epochs)
        acc_shuffled = _acc(m_sh, ite, tte, device)
        # Liveness of the control (23rd audit, B1): a shuffled arm that
        # never trained also reads chance on test, so record whether it fit
        # the shuffled train set it was given — chance there means the
        # control never ran, not that it failed honestly.
        acc_shuffled_train = _acc(m_sh, itr, torch.from_numpy(ysh), device)

        out["seeds"].append({
            "seed": seed, "n_params_enc": n_params,
            "canary_ok": bool(eye.canary() == eye._canary0),
            "canary_colors": eye.canary_colors(),
            "acc_ref": round(acc_ref, 4),
            "lr": LR_GRID[best_idx], "val_accs": val_accs,
            "acc_full": round(acc_full, 4),
            "per_class_min": round(per_class_min, 4),
            "acc_ablated": round(acc_abl, 4),
            "acc_pixshuf": round(acc_pixshuf, 4),
            "acc_shuffled": round(acc_shuffled, 4),
            "acc_shuffled_train": round(acc_shuffled_train, 4),
            "drop": round(acc_full - acc_abl, 4),
        })
    return out


# ── GPU submission (one per spec — module cache, T2.01 pattern) ──────────
JOB = r'''
import os as _o
_o.environ["MUJOCO_GL"] = "egl"   # preamble sets "disabled"; this job renders
import subprocess as _sp, sys as _sys
# Pinned + loud on purpose: 2026-08-20's sdist-before-wheels lesson.
_sp.run([_sys.executable, "-m", "pip", "install", "mujoco==3.11.0"],
        check=True)
import json
from experiments.tests.t3_01_ablate_vision import remote_run
out = remote_run(__SEEDS__)
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "t301.json"), "w"),
          indent=1)
print("DONE", json.dumps(out["seeds"][0]), flush=True)
'''

_CACHE: dict = {}


def _submit(seeds: list) -> dict:
    body = JOB.replace("__SEEDS__", repr(list(seeds)))
    job = build_job(body)
    # Kaggle first: a kernel computes server-side whether or not anyone
    # watches, and the expiring W33 hours are the budget this spec spends.
    res = submit(job, prefer="kaggle",
                 est_hours=round(0.15 + 0.30 * len(seeds), 2),
                 timeout_s=2700 + 1800 * len(seeds),
                 fetch=["t301.json"])
    if not res.ok:
        raise RuntimeError(f"T3.01 job failed on {res.backend}: {res.message}")
    out = json.loads(Path(res.artifacts["t301.json"]).read_text())
    out["backend"] = res.backend
    return out


def pilot():
    """Seed-90 pilot, disjoint from the registered seeds. Prints, records
    nothing; numbers go into the docstring and LOOP_JOURNAL by hand. The
    gates are exogenous and do NOT move on its account — the pilot exists to
    catch rig faults before the registered spend (SM.02 lesson)."""
    out = _submit([PILOT_SEED])
    print(json.dumps(out, indent=1))
    return out


def _experiment(seed: int) -> dict:
    if not _CACHE:
        _CACHE.update(_submit(SEEDS))
    rows = _CACHE["seeds"]
    return {
        "gpu": _CACHE["gpu"], "backend": _CACHE["backend"],
        "acc_full": [r["acc_full"] for r in rows],
        "acc_ablated": [r["acc_ablated"] for r in rows],
        "acc_ref": [r["acc_ref"] for r in rows],
        "acc_pixshuf": [r["acc_pixshuf"] for r in rows],
        "drop_min": min(r["drop"] for r in rows),
        "full_min": min(r["acc_full"] for r in rows),
        "ablated_max": max(r["acc_ablated"] for r in rows),
        "ref_min": min(r["acc_ref"] for r in rows),
        "train_vs_ref_min": min(r["acc_full"] - r["acc_ref"] for r in rows),
        "per_class_min": min(r["per_class_min"] for r in rows),
        "canary_ok_all": all(r["canary_ok"] for r in rows),
        "canary_colors_min": min(r["canary_colors"] for r in rows),
        "n_params_enc": rows[0]["n_params_enc"],
        "lrs": [r["lr"] for r in rows],
    }


def _control(seed: int) -> dict:
    rows = _CACHE["seeds"]
    dev = [abs(r["acc_shuffled"] - 0.25) for r in rows]
    # .get: absent in pre-2026-08-20 artifacts -> reads 0.0 -> VOID, loudly.
    fit = [r.get("acc_shuffled_train", 0.0) for r in rows]
    return {"shuffled_dev_max": round(max(dev), 4),
            "shuffled_fit_min": round(min(fit), 4),
            "acc_shuffled": [r["acc_shuffled"] for r in rows],
            "acc_shuffled_train": fit}


def _check(m: dict, c: dict):
    from .t2_03_pretrained_vision import MIN_CANARY_COLORS, PARAMS_RANGE
    # Rig first: an invalid run is VOID, not evidence about the hypothesis.
    if not m["canary_ok_all"]:
        return Status.VOID          # GL context degraded mid-run
    if m["canary_colors_min"] < MIN_CANARY_COLORS:
        return Status.VOID          # uniform frame == blind sensor
    if not (PARAMS_RANGE[0] <= m["n_params_enc"] <= PARAMS_RANGE[1]):
        return Status.VOID          # the seat holder changed under the test
    if m["ref_min"] < REF_FLOOR:
        return Status.VOID          # rig cannot reproduce T2.03's certificate
    if m["train_vs_ref_min"] < -TRAIN_TOL:
        return Status.VOID          # training lost to its own frozen subset
    if c["shuffled_dev_max"] > SHUFFLE_BAND:
        return Status.VOID          # rig leaks episode identity
    if c["shuffled_fit_min"] < SHUFFLE_FIT_FLOOR:
        return Status.VOID          # shuffled arm never fit its own train
                                    # set: the control did not run (23rd
                                    # audit, B1)
    if m["ablated_max"] > ABL_CEIL:
        return Status.VOID          # a constant input carried the task
    # The claim: trained vision is load-bearing, on every seed.
    return m["full_min"] >= MIN_FULL and m["drop_min"] >= MIN_DROP


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T3.01"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


# ── dry check: every verdict path against fabricated rows ────────────────
def _dry():
    base = dict(canary_ok_all=True, canary_colors_min=2295,
                n_params_enc=244_960, ref_min=0.45, train_vs_ref_min=0.10,
                ablated_max=0.26, full_min=0.72, drop_min=0.46)
    ctrl = {"shuffled_dev_max": 0.05, "shuffled_fit_min": 0.85}
    cases = [
        ("planted pass", base, ctrl, Status.PASS),
        ("no drop -> FAIL", {**base, "drop_min": 0.05, "ablated_max": 0.30,
                             "full_min": 0.46}, ctrl, Status.FAIL),
        ("weak full -> FAIL", {**base, "full_min": 0.40, "drop_min": 0.15,
                               "train_vs_ref_min": -0.05}, ctrl, Status.FAIL),
        ("ref collapse -> VOID", {**base, "ref_min": 0.30}, ctrl, Status.VOID),
        ("train<ref -> VOID", {**base, "train_vs_ref_min": -0.10}, ctrl,
         Status.VOID),
        ("control leak -> VOID", base, {"shuffled_dev_max": 0.20,
                                        "shuffled_fit_min": 0.85},
         Status.VOID),
        ("dead control arm -> VOID", base, {"shuffled_dev_max": 0.05,
                                            "shuffled_fit_min": 0.25},
         Status.VOID),
        ("canary drift -> VOID", {**base, "canary_ok_all": False}, ctrl,
         Status.VOID),
        ("param drift -> VOID", {**base, "n_params_enc": 500_000}, ctrl,
         Status.VOID),
        ("constant frame carries -> VOID", {**base, "ablated_max": 0.55,
                                            "drop_min": 0.17}, ctrl,
         Status.VOID),
    ]
    for name, m, cc, want in cases:
        got = _check(dict(m), dict(cc))
        got = {True: Status.PASS, False: Status.FAIL}.get(got, got)
        assert got == want, f"{name}: wanted {want}, got {got}"
        print(f"  ok: {name}")
    print("DRY OK")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "pilot":
        pilot()
    elif len(sys.argv) > 1 and sys.argv[1] == "dry":
        _dry()
    elif len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # Local: tiny end-to-end run on CPU — plumbing, not learnability.
        out = remote_run([PILOT_SEED], n_train=40, n_test=20, epochs=2,
                         lr_grid_idx=(1,))
        row = out["seeds"][0]
        # the ablated input is constant, so prediction must be constant:
        # accuracy exactly the test share of one class.
        print(json.dumps({"smoke": row, "SMOKE_OK": True}))
    else:
        run()
