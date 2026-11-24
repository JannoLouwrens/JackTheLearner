"""J14 (c), ruling 79: the ear's timing cell ON A GPU — the box's four stages (`kws_epoch_stages`) on CUDA.

The box read (`c34634a5`): host-side decode + noise loop < 20 s of the T4's 182 s epoch, so the data read
is not the cost. Suspect left: the authors' `Preprocess` copies a whole background-noise file host->device
for every augmented clip (`random.choice(self.background_noise).to(self.device)`, ~4 MB, ~126 GB an epoch).

Two arms, same clips, same seed, the authors' objects (`bcresnet@656f947a`, sha-checked) imported verbatim:
  authors   `Preprocess` as written (noise tensors on the host, copied per clip)
  noisedev  the SAME object with `background_noise` moved to the device once before the loop — the only
            change; `.to(device)` on a device tensor is a no-op, so their code path is untouched.
Every stage is bracketed by `torch.cuda.synchronize()` (wall clock; GPU work is async). Batch 0 of each arm
is warm-up and not counted. Projection = per-clip wall x 36,923 (balanced v2 train split).
Corpus: the authors' own `DownloadDataset` + `SplitDataset` (`kws_row.prepare_data`), clips globbed from
the split's word folders. A pilot-sized timing, no learner trained to any end, no rule judged, no ledger.
Rule applied by the reader (ORDERS J14 (c)): one seed (200 epochs) must fit one session with 20% to spare.

    $PY -m experiments.kws_cuda_timing submit [N]      # box: Colab T4, writes OUT
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
import time

OUT = "/data/jack-logs/speech_commands/kws_cuda_timing.json"
WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
TRAIN_N = 36_923


def find_corpus(work: str):
    base = os.path.join(work, "data", "speech_commands_v0.02")
    noise = sorted(glob.glob(os.path.join(base, "**", "_background_noise_"), recursive=True))
    files = []
    for w in WORDS:
        files += sorted(glob.glob(os.path.join(base, "**", w, "*.wav"), recursive=True))
    return files, (noise[0] if noise else None)


def time_arm(utils, BCResNets, files, noise_dir, n, arm, dev):
    import numpy as np
    import torch
    from torchvision import transforms
    rng = random.Random(0)
    fs = list(files); rng.shuffle(fs); fs = fs[:n]
    labels = [2 + WORDS.index(os.path.basename(os.path.dirname(f))) for f in fs]
    ds = utils.SpeechCommand.__new__(utils.SpeechCommand)
    ds.transform = transforms.Compose([utils.Padding()])
    ds.data_list, ds.labels = fs, labels
    loader = torch.utils.data.DataLoader(ds, batch_size=100, shuffle=True, num_workers=0)
    pre = utils.Preprocess(noise_dir, dev)
    if arm == "noisedev":
        pre.background_noise = [z.to(dev) for z in pre.background_noise]
    model = BCResNets(8).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-3, momentum=0.9)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
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
    epoch = {k: v * TRAIN_N / 1e3 for k, v in per.items()}
    return {"arm": arm, "clips_counted": counted, "per_clip_wall_ms": per, "epoch_s_projected": epoch,
            "epoch_s_total": sum(epoch.values()), "seed_h_200ep": 200 * sum(epoch.values()) / 3600}


def _timed(loader, tot):
    it = iter(loader)
    while True:
        t = time.perf_counter()
        try:
            xy = next(it)
        except StopIteration:
            return
        tot["decode"] += time.perf_counter() - t
        yield xy


def kernel_main(out_dir: str, n: int, work: str = "/tmp/kws_work") -> dict:
    import torch
    from experiments import kws_row
    os.makedirs(work, exist_ok=True)
    path = os.path.join(out_dir, "kws_cuda_timing.json")
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
    res["arms"] = []
    for arm in ("authors", "noisedev"):
        res["arms"].append(time_arm(utils, BCResNets, files, noise, n, arm, dev))
        json.dump(res, open(path, "w"))
        print("ARM", json.dumps(res["arms"][-1]), flush=True)
    return res


def kernel_body(n: int) -> str:
    from experiments.kws_row import _KERNEL_BODY
    head = _KERNEL_BODY.split("from experiments.kws_row import plan_from_args")[0]
    return head + ("from experiments.kws_cuda_timing import kernel_main\n"
                   "_r = kernel_main(JACK_OUT, %d)\n"
                   "print('RESULT ' + json.dumps({k: _r.get(k) for k in ('stopped', 'gpu', 'torch', 'arms')}), flush=True)\n"
                   "print('DONE', flush=True)\n" % n)


def main(argv) -> int:
    if len(argv) > 1 and argv[1] == "submit":
        n = int(argv[2]) if len(argv) > 2 else 5000
        from experiments.gpu import build_job, result_json, submit
        sub = submit(build_job(kernel_body(n)), prefer="colab", est_hours=0.7, timeout_s=3 * 3600,
                     fetch=["kws_cuda_timing.json"])
        print("SUBMIT", sub.backend, sub.ok, sub.job_id, sub.message, flush=True)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        open(OUT, "w").write(json.dumps(result_json(sub, "kws_cuda_timing.json")))
        print("WROTE", OUT, flush=True)
        return 0 if sub.ok else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
