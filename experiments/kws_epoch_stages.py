"""J14 (c), ruling 79: where does the ear's epoch go? Stage-by-stage timing of the AUTHORS' pipeline.

The ear's first kernel (N7-kws, VOID) ran 182 s an epoch on a T4 for a 9,232-weight network: a data-
pipeline cost, not a model cost. This probe times each stage of one training epoch AS THE AUTHORS' CODE
RUNS IT (`Qualcomm-AI-research/bcresnet@656f947a`, pinned + sha-checked by `kws_row.fetch_source`):
the stages are their own objects, imported from their `utils.py`, not re-written:

  decode   `SpeechCommand.__getitem__` = torchaudio.load + Padding, per clip, DataLoader num_workers 0
  augment  `Preprocess.__call__`'s per-clip noise/shift loop (feature step removed by timing it apart)
  mel      `LogMel` (torchaudio MelSpectrogram + log)
  step     BCResNets(8) forward + cross-entropy + backward + SGD step, batch 100

on an N-clip slice of the real Speech Commands v2 corpus (12-class words; `_background_noise_` real).
CPU only, box, nice 19, 1 thread, process CPU time AND wall. A pilot-sized measurement, no learner trained
to any end, no rule judged, no ledger row. Projection to a full epoch = per-clip cost x 36,923 (the
balanced v2 train split, `kws_cost_probe`).

One fact the box cannot measure and is printed, not guessed: on CUDA `Preprocess` moves a whole
background-noise file (`random.choice(self.background_noise).to(self.device)`, ~1M floats, 4 MB) host to
device FOR EVERY AUGMENTED CLIP, before slicing 1 s from it. The byte count per epoch is printed from the
real noise files and the authors' noise probability; its time on a T4 is the Kaggle timing cell's to read.

    PYTHONPATH=/data/tools/torchaudio_lib nice -n 19 $PY -m experiments.kws_epoch_stages [N]
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)
SRC = "/data/tmp/bcres_clone"
CORPUS = "/data/jack_corpora/speech_commands_v2"
WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
TRAIN_N = 36_923          # balanced v2 train split (kws_cost_probe CORRECTED 11-18)
OUT = "/data/jack-logs/speech_commands/kws_epoch_stages.json"


def clock():
    return time.process_time(), time.perf_counter()


def since(t):
    c, w = clock()
    return c - t[0], w - t[1]


def main(n: int) -> dict:
    from experiments import kws_row
    src = kws_row.fetch_source(SRC)
    bad = kws_row.check_source(src)
    assert not bad, "source mismatch %s" % bad
    sys.path.insert(0, src)
    import utils                     # the authors' utils.py, verbatim
    from bcresnet import BCResNets
    from torchvision import transforms

    rng = random.Random(0)
    files = []
    for w in WORDS:
        files += sorted(glob.glob(os.path.join(CORPUS, w, "*.wav")))
    rng.shuffle(files)
    files = files[:n]
    labels = [2 + WORDS.index(os.path.basename(os.path.dirname(f))) for f in files]

    ds = utils.SpeechCommand.__new__(utils.SpeechCommand)          # their __getitem__, our file list
    ds.transform = transforms.Compose([utils.Padding()])
    ds.data_list, ds.labels = files, labels
    loader = torch.utils.data.DataLoader(ds, batch_size=100, shuffle=True, num_workers=0)
    dev = torch.device("cpu")
    pre = utils.Preprocess(os.path.join(CORPUS, "_background_noise_"), dev)
    model = BCResNets(8).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-3, momentum=0.9)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    tot = {k: [0.0, 0.0] for k in ("decode", "augment", "mel", "step")}
    it = iter(loader)
    n_aug = 0
    while True:
        t = clock()
        try:
            x, y = next(it)
        except StopIteration:
            break
        for k, v in zip(("decode",), [since(t)]):
            tot[k][0] += v[0]; tot[k][1] += v[1]
        t = clock()
        feat = pre.feature                     # time the loop and the feature apart
        pre.feature = lambda z: z
        x = pre(x, y, augment=True)
        pre.feature = feat
        d = since(t); tot["augment"][0] += d[0]; tot["augment"][1] += d[1]
        t = clock()
        x = feat(x)
        d = since(t); tot["mel"][0] += d[0]; tot["mel"][1] += d[1]
        t = clock()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward(); opt.step(); model.zero_grad()
        d = since(t); tot["step"][0] += d[0]; tot["step"][1] += d[1]
    noise_len = [int(z.shape[-1]) for z in pre.background_noise]
    # their loop: label 0 (silence) always mixed; others mixed with prob 1 - 0.8 = 0.2 ... no:
    # `if labels[idx] != 0 and (not is_train or random.random() > noise_prob): continue`
    # -> a word clip is SKIPPED when random() > 0.8, i.e. mixed with prob 0.8.
    p_mix = 0.8
    mean_bytes = 4 * float(np.mean(noise_len))
    per_clip = {k: {"cpu_ms": 1e3 * v[0] / n, "wall_ms": 1e3 * v[1] / n} for k, v in tot.items()}
    out = {
        "n_clips": n, "threads": 1, "source_ok": not bad, "torch": torch.__version__,
        "per_clip": per_clip,
        "epoch_cpu_s_projected": {k: v["cpu_ms"] * TRAIN_N / 1e3 for k, v in per_clip.items()},
        "noise_files": len(noise_len), "noise_mean_mb": mean_bytes / 1e6,
        "cuda_noise_copy_gb_per_epoch": TRAIN_N * p_mix * mean_bytes / 1e9,
        "note": "box CPU; step on a T4 is not this number. decode + augment loop are host work on any device.",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
