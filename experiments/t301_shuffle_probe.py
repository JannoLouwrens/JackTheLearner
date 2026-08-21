"""T3.01 shuffled-control liveness probe — dead arm, or live arm under an
uncalibrated floor?

CONTEXT (attempt 4, kernel jack-ladder-1787256592, 2026-08-20 20:26): the
23rd-audit B1 gate SHUFFLE_FIT_FLOOR (0.35) fired on its FIRST real exercise.
acc_shuffled_train read [0.25, 0.3183, 0.25] for seeds [0, 1, 2] at chosen
lrs [1e-3, 3e-4, 1e-3]. The main arm trained to full_min 0.6333 under the
same code and the same chosen lrs, so the training RIG is demonstrably
alive; the open question is the shuffled ARM. Two readings of exactly
0.2500 on a 1200-row set that the shuffle keeps exactly balanced (300/class)
is the fingerprint of a constant predictor — plausibly an lr-1e-3
optimisation collapse on random labels, NOT a dead rig. Seed 1 (lr 3e-4)
read 0.3183: above chance, learning, below the floor. The floor was derived
a priori (chance + 0.10) and validated only against fabricated rows; it was
never calibrated against what a LIVE same-budget arm actually reads on
random labels, which is a harder objective than the real one (Zhang et al.
2017: random-label fitting needs far more epochs than real labels).

METHOD: exact replay of the registered shuffled arms — same model seed
((seed+7001)*13 + lr_idx), same batch-order generator ((seed+7001)*1009 +
lr_idx), same AdamW/BATCH/WEIGHT_DECAY, same fixed once-shuffled labels
(RandomState(seed+41)) — for ALL THREE grid lrs per seed (the registered
chosen-lr arms are two of the nine), instrumented with per-epoch mean
training loss and periodic shuffled-train accuracy, run to 124 epochs
(2x the registered 62) so "climbing at cutoff" is visible. The first 62
epochs of each replay are the registered arm's trajectory (up to GPU
nondeterminism).

READINGS, stated before launch, per (seed, lr) row:
  loss_init      mean CE on the shuffled train set before any step (~1.386)
  loss_at_62     mean training loss over epoch 62
  fit62, fit124  accuracy on the shuffled train set at epochs 62 and 124
  LIVE(s,lr)     := loss_init - loss_at_62 >= 0.05   (did training move?)
  FIT(s,lr)      := fit62 >= 0.35                    (does it clear the floor?)
  reg_lr         := {0: 1e-3, 1: 3e-4, 2: 1e-3}      (attempt 4's chosen lrs)

PRE-REGISTERED DECISION RULE — the next iteration applies the FIRST matching
branch, top-down, mechanically:
  R0  REPRODUCTION-FAIL: every seed's reg_lr arm is LIVE and FIT.
      Contradicts attempt 4's recorded readings -> the probe measured a
      different run; investigate reproduction, change nothing, dispatch
      nothing.
  R1  CONTROL-OWN-LR: every seed has SOME grid lr with FIT.
      The floor is reachable; the fault is that the control inherits the
      main arm's lr, chosen for real labels. Repair: the control selects
      its OWN lr from the grid by shuffled-train fit (a control given its
      best chance to memorise and still reading chance on dev is STRONGER
      leak evidence, not weaker). SHUFFLE_FIT_FLOOR stays at 0.35.
      Re-run registered T3.01.
  R2  GATE-TO-LOSS-FALL: every seed has some LIVE lr, but some seed's best
      fit62 < 0.35. The floor sits above what a live matched-budget arm
      reads on random labels -> the gate's premise ("cannot exceed
      chance+0.10 => did not train") is measured false. Repair: control
      selects its own lr as in R1, AND the VOID gate's observable becomes
      the direct one — loss_init - loss_at_62 >= 0.05 at the control's
      chosen lr (UB.10's uni_learn_ok, same medicine); acc_shuffled_train
      stays recorded, the floor is demoted to a recorded diagnostic with
      this probe's numbers cited in the commit (law 4: moved loudly, with
      the measurement that moved it). Re-run registered T3.01.
  R3  ESCALATE: some seed has NO live lr at all. That contradicts the main
      arm's demonstrated learning under identical code and optimiser —
      a deeper rig fault. No re-run, no repair by this desk; route to the
      overseer.

Artifact: /data/t301_shuffle_probe.json (written by the local driver after
fetch). Log: /data/tmp/t301_shuffle_probe.log. Budget: W33 Kaggle (~23 h
unspent, expires Sun 2026-08-23), est 0.5 h.

PROBE RESULT (attempt 2, kernel jack-ladder-1787263843, P100, 0.3715 h,
harvested 2026-08-20 22:33 UTC; artifact /data/t301_shuffle_probe.json) —
BRANCH R3 FIRED. All nine (seed, lr) rows, mechanically:

    seed lr    reg  loss_init loss@62 loss@124 fit62  fit124  LIVE FIT
    0    1e-4  .    1.3866    1.3865  1.3769   0.2500 0.3175  no   no
    0    3e-4  .    1.3863    1.3871  1.3871   0.2500 0.2500  no   no
    0    1e-3  REG  1.3863    1.3880  1.3864   0.2500 0.2500  no   no
    1    1e-4  .    1.3872    1.3857  1.3705   0.2575 0.2875  no   no
    1    3e-4  REG  1.3865    1.3689  1.0276   0.3108 0.5367  no   no
    1    1e-3  .    1.3869    1.3868  1.3864   0.2500 0.2500  no   no
    2    1e-4  .    1.3871    1.3866  1.3745   0.2708 0.2800  no   no
    2    3e-4  .    1.3866    1.3872  1.3869   0.2500 0.2500  no   no
    2    1e-3  REG  1.3863    1.3869  1.3864   0.2500 0.2500  no   no

R0 no (no reg arm LIVE+FIT). R1 no (no seed reaches fit62 0.35; best
0.3108). R2 no (its premise "every seed has some LIVE lr" fails — NO row
anywhere clears the 0.05 loss-fall bar by epoch 62; the max is seed 1 at
3e-4, fall 0.0176). R3 YES: every seed lacks a live lr. Per the rule:
ESCALATED TO THE OVERSEER, no re-run, no repair by this desk. T3.01 stays
VOID and undispatchable until the overseer adjudicates.

EVIDENCE THE OVERSEER SHOULD WEIGH, recorded not adjudicated: the R3 branch
was written as "deeper rig fault", but the 124-epoch tails complicate that
reading in both directions. (a) The shuffled arm is NOT uniformly dead:
seed 1 at 3e-4 is a slow learner (loss 1.3865 -> 1.0276, fit124 0.5367 —
it would clear the 0.35 floor at roughly double the registered budget), and
seeds 0/2 at 1e-4 drift down slowly by 124. This matches the docstring's own
Zhang-et-al. premise that random labels fit far slower than real ones, and
suggests every bar this probe pre-registered at epoch 62 (fit floor AND
loss-fall liveness) sits inside the random-label warmup plateau. (b) Yet it
is not purely "slow": at FULL double budget, seeds 0 and 2 still have no lr
fitting shuffled labels above 0.3175 — nine of nine rows at lr >= 3e-4 for
those seeds are bit-flat at 0.2500 while the main real-label arm trains to
full_min 0.6333 under identical code. Seed-1-vs-seeds-0/2 at the SAME lr
(3e-4: one learns to 0.5367, two sit at exactly 0.2500) is the sharpest
anomaly and is not explained by lr choice alone. Whether the repair is a
longer control budget, a loss-fall observable at a calibrated horizon, or a
rig investigation is the overseer's call.

ADJUDICATION (24th audit, 2026-08-21 00:40 UTC, docs/OVERSIGHT.md — appended
per its B1; the R3 finding above is preserved, not rewritten). The overseer
ruled: R3's trigger fired correctly; its attached conclusion ("deeper rig
fault") is REFUTED by this probe's own tail. THERE IS NO RIG FAULT.
  (1) "No seed has a live lr" is an artefact of the LIVE predicate's epoch-62
      horizon: at lr 1e-4 all three seeds are escaping the plateau by epoch
      124 (loss 1.3866->1.3769, 1.3872->1.3705, 1.3871->1.3745; fit off flat
      0.2500 to 0.3175/0.2875/0.2800).
  (2) Every "dead" row sits on the ln 4 = 1.386294 max-entropy fixed point to
      four decimals with fit exactly 0.2500 — the CORRECT pre-memorisation
      behaviour of a live network on random labels (this file's own
      Zhang-et-al. premise), not a broken rig.
  (3) The seed-1-vs-seeds-0/2 "anomaly" dissolves read ACROSS the lr axis:
      a single plateau-escape threshold between 1e-4 and 1e-3, with 3e-4
      straddling it and escape seed-dependent at the boundary.
The substantive finding is R2's diagnosis (the 0.35 floor sits above what a
live matched-budget arm reads on random labels), but R2's repair as written
is NOT licensed: a loss-fall proxy at a recalibrated horizon re-labels
"moved a little" as liveness, with no calibrated relationship to
leak-detection power (memorisation here costs >124 epochs on the single best
row and is unreached by 2 of 3 seeds at ANY tested lr). T3.01 is UN-FROZEN
FOR REDESIGN per the audit's B2 — deterministic train/test hash-disjointness
gate plus a pre-registered fork on the shuffled arm's fate — and is not
cleared to re-run as-is. The redesign record lives in
experiments/tests/t3_01_ablate_vision.py (V3 REDESIGN block).
"""
import json
import os
import sys

# Pin cwd/sys.path to the repo root DERIVED FROM THIS FILE, never hardcoded:
# this module runs in two contexts — detached driver on this box (repo at
# /home/opc/jackthelearner) and remote import on the GPU VM (repo cloned to
# /tmp/jack). Attempt 1 (kernel jack-ladder-1787260513) hardcoded the local
# path and died FileNotFoundError at remote import, before touching the GPU.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_REPO)
sys.path.insert(0, _REPO)

OUT = "/data/t301_shuffle_probe.json"
LIVE_FALL = 0.05
REG_LR = {0: 1e-3, 1: 3e-4, 2: 1e-3}


def remote_probe(seeds=(0, 1, 2), epochs=124, gate_epoch=62,
                 n_train=None) -> dict:
    """Runs on the GPU VM (or locally, tiny, for the smoke)."""
    import numpy as np
    import torch
    from experiments.tests.t3_01_ablate_vision import (
        _make_model, _to_tensor, _acc, LR_GRID, BATCH, WEIGHT_DECAY)
    from experiments.tests.t2_03_pretrained_vision import (
        _build_dataset, N_TRAIN)
    n_train = n_train or N_TRAIN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = torch.nn.CrossEntropyLoss()
    out = {"gpu": (torch.cuda.get_device_name(0) if device == "cuda"
                   else "cpu"),
           "epochs": epochs, "gate_epoch": gate_epoch, "rows": []}
    for seed in seeds:
        Xtr, ytr, _ = _build_dataset(seed, n_train)
        itr = _to_tensor(Xtr)
        ysh = ytr.copy()
        np.random.RandomState(seed + 41).shuffle(ysh)
        tysh = torch.from_numpy(ysh)
        for li in range(len(LR_GRID)):
            # exact replay of _train_one(seed+7001, li, ...)'s seeding
            s2 = seed + 7_001
            model, _ = _make_model(s2 * 13 + li, device)
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=LR_GRID[li],
                                    weight_decay=WEIGHT_DECAY)
            g = torch.Generator().manual_seed(s2 * 1009 + li)
            model.eval()
            with torch.no_grad():
                li0 = float(np.mean([
                    loss_fn(model(itr[i:i + 64].to(device)),
                            tysh[i:i + 64].to(device)).item()
                    for i in range(0, n_train, 64)]))
            model.train()
            n = len(itr)
            loss_curve, acc_curve = [], {}
            for ep in range(1, epochs + 1):
                perm = torch.randperm(n, generator=g)
                ep_losses = []
                for i in range(0, n, BATCH):
                    idx = perm[i:i + BATCH]
                    xb = itr[idx].to(device)
                    yb = tysh[idx].to(device)
                    opt.zero_grad()
                    loss = loss_fn(model(xb), yb)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite loss s{seed} lr{li}")
                    loss.backward()
                    opt.step()
                    ep_losses.append(loss.item())
                loss_curve.append(round(float(np.mean(ep_losses)), 4))
                if ep % 4 == 0 or ep in (1, gate_epoch, epochs):
                    model.eval()
                    acc_curve[str(ep)] = round(_acc(model, itr, tysh,
                                                    device), 4)
                    model.train()
            model.eval()
            row = {"seed": seed, "lr": LR_GRID[li],
                   "is_reg_lr": bool(abs(REG_LR.get(seed, -1) - LR_GRID[li])
                                     < 1e-12),
                   "loss_init": round(li0, 4),
                   "loss_at_gate": loss_curve[min(gate_epoch, epochs) - 1],
                   "loss_final": loss_curve[-1],
                   "fit_gate": acc_curve.get(str(gate_epoch)),
                   "fit_final": acc_curve.get(str(epochs)),
                   "loss_curve": loss_curve, "acc_curve": acc_curve}
            row["live"] = bool(row["loss_init"] - row["loss_at_gate"]
                               >= LIVE_FALL)
            row["fit_ok"] = bool((row["fit_gate"] or 0.0) >= 0.35)
            out["rows"].append(row)
            print(f"seed {seed} lr {LR_GRID[li]:g} reg={row['is_reg_lr']} "
                  f"loss {row['loss_init']}->{row['loss_at_gate']}"
                  f"->{row['loss_final']} fit62={row['fit_gate']} "
                  f"fit{epochs}={row['fit_final']} live={row['live']} "
                  f"fit_ok={row['fit_ok']}", flush=True)
    return out


def _branch(rows: list) -> str:
    """The pre-registered rule, mechanically. First match, top-down."""
    seeds = sorted({r["seed"] for r in rows})
    reg = {s: next(r for r in rows if r["seed"] == s and r["is_reg_lr"])
           for s in seeds}
    if all(reg[s]["live"] and reg[s]["fit_ok"] for s in seeds):
        return "R0-REPRODUCTION-FAIL"
    if all(any(r["fit_ok"] for r in rows if r["seed"] == s) for s in seeds):
        return "R1-CONTROL-OWN-LR"
    if all(any(r["live"] for r in rows if r["seed"] == s) for s in seeds):
        return "R2-GATE-TO-LOSS-FALL"
    return "R3-ESCALATE"


JOB = r'''
import os as _o
_o.environ["MUJOCO_GL"] = "egl"   # preamble sets "disabled"; this job renders
import subprocess as _sp, sys as _sys
_sp.run([_sys.executable, "-m", "pip", "install", "mujoco==3.11.0"],
        check=True)
import json
from experiments.t301_shuffle_probe import remote_probe
out = remote_probe()
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"],
                                 "t301_probe.json"), "w"), indent=1)
print("DONE", flush=True)
'''


def main():
    from pathlib import Path
    from experiments.gpu import build_job, submit
    job = build_job(JOB)
    res = submit(job, prefer="kaggle", est_hours=0.5, timeout_s=3600,
                 fetch=["t301_probe.json"])
    if not res.ok:
        print(f"PROBE FAILED on {res.backend}: {res.message}", flush=True)
        sys.exit(1)
    out = json.loads(Path(res.artifacts["t301_probe.json"]).read_text())
    out["backend"] = res.backend
    out["branch"] = _branch(out["rows"])
    json.dump(out, open(OUT, "w"), indent=1)
    for r in out["rows"]:
        print(f"seed {r['seed']} lr {r['lr']:g} reg={r['is_reg_lr']} "
              f"loss {r['loss_init']}->{r['loss_at_gate']} "
              f"fit62={r['fit_gate']} fit124={r['fit_final']} "
              f"live={r['live']} fit_ok={r['fit_ok']}", flush=True)
    print(f"BRANCH: {out['branch']}  (rule in the module docstring; apply "
          f"verbatim)", flush=True)


if __name__ == "__main__":
    main()
