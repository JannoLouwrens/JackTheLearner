"""T3.01 training-curves probe — WHY did end-to-end training lose to its
own frozen probe? (diagnostic, not a registered test; produces no verdict)

CONTEXT (registered run, kernel jack-ladder-1787231872, 2026-08-20 13:29,
VOID by the pre-registered train-attribution gate): acc_full [0.48, 0.39,
0.3833] vs acc_ref [0.4467, 0.4667, 0.4933] -> train_vs_ref_min -0.11 <
-TRAIN_TOL 0.05. Seeds 1 and 2 collapsed a class (per_class_min 0.0); seed 0
trained healthily and cleared every claim gate. The docstring pre-registered
this lane as "an optimisation defect of THIS rig, not evidence about the
encoder", and SYSTEM.md's path for VOID is: fix the arm, do not decide.

WHAT THIS PROBE MEASURES. Two candidate repairs exist and they are
distinguished by curves the registered run did not record:
  scratch    the registered run's exact full-data training (same init seed
             seed*13+lr_idx, same shuffle generator seed*1009+lr_idx, all
             1200 rows) extended from 25 to 100 epochs, at every grid LR,
             recording per-epoch mean train loss, test acc, per-class min.
             Epochs 1-25 of this arm REPLAY the registered run's trajectory
             (modulo GPU nondeterminism), so the diagnosis is about the run
             that actually failed, not a cousin of it.
  warmstart  the same encoder init the ref arm froze (_make_model(seed)),
             with the head PRE-FITTED on the frozen train features — i.e.
             training STARTS AT the frozen-probe solution the attribution
             gate compares against, making the docstring's "end-to-end
             training strictly contains the frozen-probe solution" literal.
             Then end-to-end at every grid LR, same loop, same recording.

PRE-STATED READINGS (written before launch; the probe decides, we obey):
  R1 BUDGET: if the scratch arm, at the LR the registered run chose (seed 1:
     3e-4, seed 2: 1e-3), reaches test acc >= acc_ref - 0.05 with
     per_class_min > 0 by epoch 100 on the collapsed seeds, the 25-epoch
     budget was the fault. Repair: raise EPOCHS (uniformly, control included)
     to the smallest recorded epoch where ALL seeds clear the band, doubled
     for margin.
  R2 STABILITY: if the scratch curve plateaus below acc_ref - 0.05 (slope of
     the last 20 recorded epochs ~ 0) or the dead class persists at epoch
     100, budget alone cannot repair it — the fault is optimisation from a
     random init, and R3 decides.
  R3 WARM-START: the warmstart arm begins at ~acc_ref by construction. If at
     some single grid LR it stays >= acc_ref - 0.05 at EVERY recorded epoch
     on all three seeds, head-warm-start (LP-FT) is the repair. If joint
     training degrades it below the band at every grid LR, the first
     gradient into the encoder is destructive at this scale — that is
     evidence FOR the registry's falsified_by, and the honest next step is
     to say so at the spec level (FAIL territory), not a stack of repairs.
  DECISION RULE: adopt the cheapest repair among {R1, R3} whose recorded
     curves clear the attribution band on all three seeds; pre-register it
     as the v2 arm (strengthen-only, the VOID stays in history). This probe
     is the ONE diagnostic: if the repaired registered run VOIDs on
     attribution again, park per the SM.02 / overseer-B5 rule — a second
     repair would need a new mechanism-level reason, not a new lottery.

COST. ~18 trainings x 100 epochs on 1200 64x64 frames + 6 head fits; the
registered run's 15 trainings x 25 epochs metered 0.199 h on the P100, so
this prices at ~0.7 h. W33 has ~24.9 h expiring Sunday 08-23. Test labels
are consulted per-epoch here — fine for a diagnostic that produces no
verdict; the v2 registered run keeps the val-split LR-selection protocol.

Run detached:  scripts/launch_detached.sh /data/t3_01_curves.log \
    /data/venvs/jackthelearner/bin/python -m experiments.t3_01_curves_probe
Artifact: /data/t3_01_curves.json (fetched from the kernel's t301_curves.json).
"""

from __future__ import annotations

import json
from pathlib import Path

SEEDS = [0, 1, 2]
EPOCHS_PROBE = 100
OUT_LOCAL = "/data/t3_01_curves.json"


# ── remote science (runs on the GPU VM; imports stay inside) ─────────────
def remote_curves(seeds: list, epochs: int = EPOCHS_PROBE) -> dict:
    import numpy as np
    import torch

    from experiments.tests.t3_01_ablate_vision import (
        BATCH, LR_GRID, WEIGHT_DECAY, _make_model, _to_tensor)
    from experiments.tests.t2_03_pretrained_vision import (
        _build_dataset, _probe_acc, CLASSES, N_TRAIN, N_TEST)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = torch.nn.CrossEntropyLoss()

    def _acc_and_perclass(model, ite, tte):
        preds = []
        with torch.no_grad():
            for i in range(0, len(ite), 64):
                preds.append(model(ite[i:i + 64].to(device)).argmax(1).cpu())
        p = torch.cat(preds)
        acc = float((p == tte).float().mean())
        pc = min(float((p[tte == k] == k).float().mean())
                 for k in range(len(CLASSES)))
        return round(acc, 4), round(pc, 4)

    def _train_recorded(model, itr, ttr, ite, tte, lr, gen_seed):
        opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=WEIGHT_DECAY)
        g = torch.Generator().manual_seed(gen_seed)
        n = len(itr)
        curve = []
        for _ in range(epochs):
            model.train()
            losses = []
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, BATCH):
                idx = perm[i:i + BATCH]
                xb, yb = itr[idx].to(device), ttr[idx].to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at lr {lr}")
                loss.backward()
                opt.step()
                losses.append(float(loss))
            model.eval()
            acc, pc = _acc_and_perclass(model, ite, tte)
            curve.append({"loss": round(sum(losses) / len(losses), 4),
                          "acc": acc, "pc_min": pc})
        return curve

    out = {"gpu": torch.cuda.get_device_name(0) if device == "cuda"
           else "cpu", "epochs": epochs, "seeds": []}
    for seed in seeds:
        Xtr, ytr, _ = _build_dataset(seed, N_TRAIN)
        Xte, yte, _ = _build_dataset(seed + 500_009, N_TEST)
        itr, ite = _to_tensor(Xtr), _to_tensor(Xte)
        ttr, tte = torch.from_numpy(ytr), torch.from_numpy(yte)

        # frozen ref: same encoder init the registered run's ref arm froze.
        model0, _ = _make_model(seed, device)
        enc0 = model0[0]
        with torch.no_grad():
            ftr = np.concatenate(
                [enc0(itr[i:i + 16].to(device)).cpu().numpy()
                 for i in range(0, N_TRAIN, 16)])
            fte = np.concatenate(
                [enc0(ite[i:i + 16].to(device)).cpu().numpy()
                 for i in range(0, N_TEST, 16)])
        acc_ref, _ = _probe_acc(ftr.astype(np.float32), ytr,
                                fte.astype(np.float32), yte)

        # head pre-fit on the frozen features (the warmstart init).
        torch.manual_seed(seed + 90_001)
        head0 = torch.nn.Linear(1024, 4).to(device)
        h_opt = torch.optim.AdamW(head0.parameters(), lr=1e-2,
                                  weight_decay=WEIGHT_DECAY)
        Ftr = torch.from_numpy(ftr.astype(np.float32)).to(device)
        for _ in range(300):
            h_opt.zero_grad()
            l = loss_fn(head0(Ftr), ttr.to(device))
            l.backward()
            h_opt.step()
        head0.eval()
        with torch.no_grad():
            hp = head0(torch.from_numpy(fte.astype(np.float32))
                       .to(device)).argmax(1).cpu()
        head_acc = round(float((hp == tte).float().mean()), 4)

        row = {"seed": seed, "acc_ref": round(acc_ref, 4),
               "head_acc": head_acc, "scratch": {}, "warmstart": {}}
        for li, lr in enumerate(LR_GRID):
            # scratch: the registered run's exact full-data training, longer.
            m, _ = _make_model(seed * 13 + li, device)
            row["scratch"][str(lr)] = _train_recorded(
                m, itr, ttr, ite, tte, lr, seed * 1009 + li)
            # warmstart: ref encoder init + pre-fitted head, end-to-end.
            w, _ = _make_model(seed, device)
            with torch.no_grad():
                w[1].weight.copy_(head0.weight)
                w[1].bias.copy_(head0.bias)
            row["warmstart"][str(lr)] = _train_recorded(
                w, itr, ttr, ite, tte, lr, seed * 2003 + li)
        out["seeds"].append(row)
        print(f"seed {seed} done: ref {row['acc_ref']} head {head_acc}",
              flush=True)
    return out


# ── submission (local side) ──────────────────────────────────────────────
JOB = r'''
import os as _o
_o.environ["MUJOCO_GL"] = "egl"   # preamble sets "disabled"; this job renders
import subprocess as _sp, sys as _sys
# Pinned + loud on purpose: 2026-08-20's sdist-before-wheels lesson.
_sp.run([_sys.executable, "-m", "pip", "install", "mujoco==3.11.0"],
        check=True)
import json
from experiments.t3_01_curves_probe import remote_curves
out = remote_curves([0, 1, 2])
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"],
                                 "t301_curves.json"), "w"), indent=1)
print("DONE", flush=True)
'''


def _summarise(out: dict) -> None:
    band = 0.05
    for row in out["seeds"]:
        ref = row["acc_ref"]
        print(f"seed {row['seed']}: ref {ref} head_fit {row['head_acc']}")
        for arm in ("scratch", "warmstart"):
            for lr, curve in row[arm].items():
                a25 = curve[24]["acc"] if len(curve) > 24 else None
                aN = curve[-1]["acc"]
                pcN = curve[-1]["pc_min"]
                first_ok = next((i + 1 for i, e in enumerate(curve)
                                 if e["acc"] >= ref - band
                                 and e["pc_min"] > 0), None)
                print(f"  {arm:9s} lr {lr}: acc@25 {a25} acc@{len(curve)} "
                      f"{aN} pc_min@end {pcN} first_epoch_in_band {first_ok}")


def main() -> None:
    from experiments.gpu import build_job, submit
    job = build_job(JOB)
    res = submit(job, prefer="kaggle", est_hours=0.9, timeout_s=7200,
                 fetch=["t301_curves.json"])
    if not res.ok:
        raise RuntimeError(f"curves probe failed on {res.backend}: "
                           f"{res.message}")
    out = json.loads(Path(res.artifacts["t301_curves.json"]).read_text())
    out["backend"] = res.backend
    Path(OUT_LOCAL).write_text(json.dumps(out, indent=1))
    print(f"saved {OUT_LOCAL}")
    _summarise(out)


if __name__ == "__main__":
    main()
