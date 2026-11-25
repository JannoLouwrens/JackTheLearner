"""Row N7-kws-2's plumbing (ruling 106): the authors' ear pipeline, decoded once and with its noise on the device.

The authors' code (`Qualcomm-AI-research/bcresnet@656f947a`, sha-pinned by `kws_row`) is NOT edited. Two
wrappers are installed on their classes in-process, before `main.py` runs (as the seeding wrapper is):

  (i)  decode cache — `SpeechCommand.__getitem__`: `torchaudio.load(path)` is called once per path and the
       decoded tensor kept in a dict; every later epoch reads the cached tensor (a CLONE, so their
       `Padding` / in-place ops can never write into the cache). Their transform runs after, as written.
  (ii) noise on the device — `Preprocess.__init__`: after their constructor, `background_noise` is moved to
       `self.device` once. Their `__call__` still does `random.choice(...).to(self.device)`, which is then a
       no-op; the random draws (python `random`, numpy) are the same calls in the same order.

Neither touches a random number generator, the model, optimiser, augmentation arithmetic, epochs or band.
Admission (ruling 106, committed before any seed): `admission(...)` feeds the SAME seeded clips through the
authors' `DataLoader` + `Preprocess` (augment on, train mode, their LogMel) with and without the plumbing,
for 2 epochs x 3 batches — epoch 2 is where the cache is READ, so a test of epoch 1 alone would be blind —
and requires max |mel_plain - mel_plumbed| <= 1e-6 on every batch, the labels equal, and the cache hit
on every clip of epoch 2. `tests/test_kws_plumbing.py` runs it on the box (CPU); the Colab timing cell
runs the same function on CUDA, where (ii) is not a no-op.

Ours beside theirs: theirs decodes every clip every epoch (num_workers 0) and copies a ~4 MB noise file
host->device per augmented clip; ours decodes each clip once and copies each noise file once.
"""
from __future__ import annotations

import glob
import os
import random

_ORIG = {}


def install(utils) -> dict:
    """Patch the authors' `utils` module in place. Returns the stats dict (hits / misses)."""
    if "getitem" in _ORIG:
        return _ORIG["stats"]
    import torchaudio
    stats = {"hits": 0, "misses": 0}
    cache: dict = {}
    _ORIG.update(getitem=utils.SpeechCommand.__getitem__, pinit=utils.Preprocess.__init__,
                 stats=stats, cache=cache)

    def __getitem__(self, idx):
        audio_path = self.data_list[idx]
        if audio_path in cache:
            stats["hits"] += 1
        else:
            stats["misses"] += 1
            cache[audio_path] = torchaudio.load(audio_path)[0]
        sample = cache[audio_path].clone()
        if self.transform:
            sample = self.transform(sample)
        return sample, self.labels[idx]

    pinit = _ORIG["pinit"]

    def __init__(self, noise_loc, device, *a, **k):
        pinit(self, noise_loc, device, *a, **k)
        self.background_noise = [z.to(self.device) for z in self.background_noise]

    utils.SpeechCommand.__getitem__ = __getitem__
    utils.Preprocess.__init__ = __init__
    return stats


def uninstall(utils) -> None:
    if "getitem" in _ORIG:
        utils.SpeechCommand.__getitem__ = _ORIG["getitem"]
        utils.Preprocess.__init__ = _ORIG["pinit"]
        _ORIG.clear()


WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


def pick_clips(corpus: str, n: int, seed: int = 0):
    """n word clips (labels 2..11 as the authors' 12-class map) + some label-0 'silence' rows, which take
    the U(0,1) noise branch — both augmentation branches must be exercised."""
    files = []
    for w in WORDS:
        files += [(f, 2 + WORDS.index(w)) for f in sorted(glob.glob(os.path.join(corpus, w, "*.wav")))[:200]]
    rng = random.Random(seed)
    rng.shuffle(files)
    files = files[:n]
    for i in range(0, len(files), 10):          # every 10th clip relabelled silence (0): exercises branch
        files[i] = (files[i][0], 0)
    return [f for f, _ in files], [l for _, l in files]


def _run_arm(utils, files, labels, noise_dir, dev, epochs, batches, seed):
    import numpy as np
    import torch
    from torchvision import transforms
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ds = utils.SpeechCommand.__new__(utils.SpeechCommand)
    ds.transform = transforms.Compose([utils.Padding()])
    ds.data_list, ds.labels = list(files), list(labels)
    loader = torch.utils.data.DataLoader(ds, batch_size=100, shuffle=True, num_workers=0)
    pre = utils.Preprocess(noise_dir, dev)
    out = []
    for _ in range(epochs):
        for b, (x, y) in enumerate(loader):
            if b >= batches:
                break
            x, y = x.to(dev), y.to(dev)
            mel = pre(x, y, augment=True)      # their call: noise/shift loop + LogMel = what main.py feeds
            out.append((mel.detach().cpu().clone(), y.cpu().clone()))
    return out


def admission(utils, files, labels, noise_dir, dev, epochs=2, batches=3, seed=0, tol=1e-6) -> dict:
    plain = _run_arm(utils, files, labels, noise_dir, dev, epochs, batches, seed)
    stats = install(utils)
    try:
        plumbed = _run_arm(utils, files, labels, noise_dir, dev, epochs, batches, seed)
        hits, misses = stats["hits"], stats["misses"]
    finally:
        uninstall(utils)
    diffs = [float((a[0] - b[0]).abs().max()) for a, b in zip(plain, plumbed)]
    lab_eq = all(bool((a[1] == b[1]).all()) for a, b in zip(plain, plumbed))
    n_ep = len(files)
    res = {"device": str(dev), "batches_compared": len(diffs), "max_abs_diff": diffs,
           "labels_equal": lab_eq, "cache_hits": hits, "cache_misses": misses,
           "expected_hits_min": min(n_ep, 100 * batches) * (epochs - 1),
           "tol": tol}
    res["ok"] = (len(diffs) == epochs * batches and lab_eq and max(diffs) <= tol
                 and hits >= res["expected_hits_min"] and misses <= n_ep)
    return res
