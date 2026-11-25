"""Row N7-kws-2's CUDA admission + timing cell (ruling 106; KNOWN_ANSWERS N7-kws-2), Colab T4.

Two things, in this order, in one Colab job — no seed is trained:
  1. ADMISSION ON CUDA: `kws_plumbing.admission` on the T4 (300 clips of the authors' prepared corpus, 30
     relabelled to the label-0 noise branch, 2 epochs x 3 batches, max |dmel| <= 1e-6). On CUDA the noise
     plumbing (ii) is not a no-op, so the box's CPU PASS does not cover it. A FAIL here stops the row.
  2. TIMING: the plumbed pipeline (`kws_plumbing.install`) over N train-word clips, TWO passes with the same
     dataset object — pass 1 decodes (cache misses), pass 2 reads the cache, as epochs 2..200 will. Stages
     as `kws_cuda_timing` (decode / augment / mel / step, `torch.cuda.synchronize` per stage, batch 0 of each
     pass dropped). Projection: seed_h = (pass1 + 199 x pass2) per-clip x 36,923 / 3600 — train only; the
     valid pass (4,445 clips, no backward) is printed as an upper bound by scaling x 41,368 / 36,923.
Fit rule (frozen): one seed fits one session with 20% spare — Kaggle <= 10.0 h, Colab <= 2.4 h.

    $PY -m experiments.kws2_timing submit [N]      # box: Colab T4, writes OUT
"""
from __future__ import annotations

import json
import os
import sys
import time

OUT = "/data/jack-logs/speech_commands/kws2_timing.json"
TRAIN_N = 36_923
VALID_N = 4_445


def time_plumbed(utils, BCResNets, files, noise_dir, n, dev):
    import random

    import numpy as np
    import torch
    from torchvision import transforms

    from experiments import kws_plumbing as KP
    from experiments.kws_cuda_timing import WORDS, _timed
    rng = random.Random(0)
    fs = list(files); rng.shuffle(fs); fs = fs[:n]
    labels = [2 + WORDS.index(os.path.basename(os.path.dirname(f))) for f in fs]
    stats = KP.install(utils)
    try:
        ds = utils.SpeechCommand.__new__(utils.SpeechCommand)
        ds.transform = transforms.Compose([utils.Padding()])
        ds.data_list, ds.labels = fs, labels
        loader = torch.utils.data.DataLoader(ds, batch_size=100, shuffle=True, num_workers=0)
        pre = utils.Preprocess(noise_dir, dev)
        model = BCResNets(8).to(dev)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-3, momentum=0.9)
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
        passes = []
        for p in range(2):
            tot = {k: 0.0 for k in ("decode", "augment", "mel", "step")}
            counted = 0
            for b, (x, y) in enumerate(_timed(loader, tot)):
                if b == 0:
                    tot["decode"] = 0.0
                x, y = x.to(dev), y.to(dev)
                sync(); t = time.perf_counter()
                feat = pre.feature; pre.feature = lambda z: z
                x = pre(x, y, augment=True)
                pre.feature = feat
                sync(); ta = time.perf_counter()
                x = feat(x)
                sync(); tm = time.perf_counter()
                loss = torch.nn.functional.cross_entropy(model(x), y)
                loss.backward(); opt.step(); model.zero_grad()
                sync(); ts = time.perf_counter()
                if b == 0:
                    continue
                counted += len(y)
                tot["augment"] += ta - t; tot["mel"] += tm - ta; tot["step"] += ts - tm
            per = {k: 1e3 * v / counted for k, v in tot.items()}
            passes.append({"pass": p + 1, "clips_counted": counted, "per_clip_wall_ms": per,
                           "epoch_s_projected": sum(per.values()) * TRAIN_N / 1e3,
                           "hits": stats["hits"], "misses": stats["misses"]})
    finally:
        KP.uninstall(utils)
    e1, e2 = passes[0]["epoch_s_projected"], passes[1]["epoch_s_projected"]
    seed_h = (e1 + 199 * e2) / 3600
    return {"arm": "plumbed", "passes": passes, "seed_h_200ep_train": seed_h,
            "seed_h_200ep_upper_with_valid": seed_h * (TRAIN_N + VALID_N) / TRAIN_N,
            "fits_colab_2p4h": seed_h * (TRAIN_N + VALID_N) / TRAIN_N <= 2.4,
            "fits_kaggle_10h": seed_h * (TRAIN_N + VALID_N) / TRAIN_N <= 10.0}


def kernel_main(out_dir: str, n: int, work: str = "/tmp/kws_work") -> dict:
    import torch

    from experiments import kws_plumbing as KP
    from experiments import kws_row
    from experiments.kws_cuda_timing import find_corpus
    os.makedirs(work, exist_ok=True)
    path = os.path.join(out_dir, "kws2_timing.json")
    res = {"commit": kws_row.COMMIT, "n": n, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    src = kws_row.fetch_source(os.path.join(work, "bcresnet_src"))
    bad = kws_row.check_source(src)
    res["source_ok"] = not bad
    if bad:
        res["stopped"] = "SOURCE_MISMATCH %s" % bad
        json.dump(res, open(path, "w")); return res
    res["prepare"] = kws_row.prepare_data(src, work)
    files, noise = find_corpus(work)
    res["corpus"] = {"clips": len(files), "noise_dir": noise}
    json.dump(res, open(path, "w"))
    if res["prepare"]["rc"] or not files or not noise:
        res["stopped"] = "PREPARE_FAILED"
        json.dump(res, open(path, "w")); return res
    sys.path.insert(0, src)
    import utils
    from bcresnet import BCResNets
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yes = [f for f in files if os.path.basename(os.path.dirname(f)) == "yes"]
    corpus_dir = os.path.dirname(os.path.dirname(yes[0]))
    afiles, alabels = KP.pick_clips(corpus_dir, 300)
    res["admission"] = KP.admission(utils, afiles, alabels, noise, dev)
    json.dump(res, open(path, "w"))
    print("ADMISSION", json.dumps(res["admission"]), flush=True)
    if not res["admission"]["ok"]:
        res["stopped"] = "ADMISSION_FAILED_ON_%s" % dev.type
        json.dump(res, open(path, "w")); return res
    res["timing"] = time_plumbed(utils, BCResNets, files, noise, n, dev)
    json.dump(res, open(path, "w"))
    print("TIMING", json.dumps(res["timing"]), flush=True)
    return res


def kernel_body(n: int) -> str:
    from experiments.kws_row import _KERNEL_BODY
    head = _KERNEL_BODY.split("from experiments.kws_row import plan_from_args")[0]
    return head + ("from experiments.kws2_timing import kernel_main\n"
                   "_r = kernel_main(JACK_OUT, %d)\n"
                   "print('RESULT ' + json.dumps({k: _r.get(k) for k in ('stopped', 'gpu', 'torch', 'admission', 'timing')}), flush=True)\n"
                   "print('DONE', flush=True)\n" % n)


def main(argv) -> int:
    if len(argv) > 1 and argv[1] == "submit":
        n = int(argv[2]) if len(argv) > 2 else 5000
        from experiments.gpu import build_job, result_json, submit
        sub = submit(build_job(kernel_body(n)), prefer="colab", est_hours=0.6, timeout_s=3 * 3600,
                     fetch=["kws2_timing.json"])
        print("SUBMIT", sub.backend, sub.ok, sub.job_id, sub.message, flush=True)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        open(OUT, "w").write(json.dumps(result_json(sub, "kws2_timing.json")))
        print("WROTE", OUT, flush=True)
        return 0 if sub.ok else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
