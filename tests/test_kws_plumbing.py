"""Row N7-kws-2's admission test (ruling 106), box CPU: the mel batch reaching the net is identical with and
without the plumbing (decode cache + noise on device), over 2 epochs x 3 batches of the authors' pipeline —
and the comparison is shown able to fail (planted: a cache returning a neighbour's clip; a noise copy that
is not the same numbers).

    PYTHONPATH=/data/tools/torchaudio_lib nice -n 19 $PY -m tests.test_kws_plumbing
"""
import importlib
import os
import sys

import torch

from experiments import kws_plumbing as KP
from experiments import kws_row

SRC = "/data/tmp/bcres_clone"
CORPUS = "/data/jack_corpora/speech_commands_v2"
torch.set_num_threads(1)
NOISE = os.path.join(CORPUS, "_background_noise_")


def _utils():
    assert kws_row.check_source(SRC) == [], "authors' source does not match the sha pins"
    sys.path.insert(0, SRC)
    try:
        return importlib.import_module("utils")
    finally:
        sys.path.remove(SRC)


def _clips():
    files, labels = KP.pick_clips(CORPUS, 300)
    assert len(files) == 300 and labels.count(0) == 30
    return files, labels


class _MP:
    """minimal monkeypatch: setattr now, undo() restores."""
    def __init__(self):
        self.undo_list = []

    def setattr(self, obj, name, val):
        self.undo_list.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    def undo(self):
        for obj, name, val in reversed(self.undo_list):
            setattr(obj, name, val)


def test_admission_mel_identical(utils, clips, mp=None):
    res = KP.admission(utils, *clips, NOISE, torch.device("cpu"))
    print(res)
    assert res["batches_compared"] == 6
    assert res["labels_equal"]
    assert max(res["max_abs_diff"]) <= 1e-6
    assert res["cache_hits"] >= 300 and res["cache_misses"] == 300   # epoch 2 read entirely from cache
    assert res["ok"]
    # plumbing removed afterwards: the authors' methods are back
    assert utils.SpeechCommand.__getitem__.__module__ == "utils"


def _planted(monkeypatch, utils, sabotage):
    real = KP.install

    def bad(u):
        st = real(u)
        sabotage(u)
        return st
    monkeypatch.setattr(KP, "install", bad)


def test_planted_cache_wrong_clip_is_caught(monkeypatch, utils, clips):
    def sab(u):
        cache = KP._ORIG["cache"]
        g = u.SpeechCommand.__getitem__

        def wrong(self, idx):                    # after the first epoch, return the NEXT path's audio
            p = self.data_list[idx]
            if p in cache:
                self.data_list[idx], swap = self.data_list[(idx + 1) % len(self.data_list)], p
                try:
                    out = g(self, idx)
                finally:
                    self.data_list[idx] = swap
                return out[0], self.labels[idx]
            return g(self, idx)
        u.SpeechCommand.__getitem__ = wrong
    _planted(monkeypatch, utils, sab)
    res = KP.admission(utils, *clips, NOISE, torch.device("cpu"))
    assert not res["ok"] and max(res["max_abs_diff"][3:]) > 1e-3
    assert max(res["max_abs_diff"][:3]) <= 1e-6              # epoch 1 alone would have been blind


def test_planted_noise_changed_is_caught(monkeypatch, utils, clips):
    def sab(u):
        p = u.Preprocess.__init__

        def init(self, *a, **k):
            p(self, *a, **k)
            self.background_noise = [z * (1 + 1e-3) for z in self.background_noise]
        u.Preprocess.__init__ = init
    _planted(monkeypatch, utils, sab)
    res = KP.admission(utils, *clips, NOISE, torch.device("cpu"))
    assert not res["ok"] and max(res["max_abs_diff"]) > 1e-6


if __name__ == "__main__":
    U, C = _utils(), _clips()
    bad = 0
    for n, f in [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]:
        mp = _MP()
        try:
            f(mp, U, C) if n.startswith("test_planted") else f(U, C)
            print("PASS", n, flush=True)
        except Exception as e:   # noqa: BLE001
            bad += 1
            print("FAIL", n, repr(e), flush=True)
        finally:
            mp.undo()
    sys.exit(1 if bad else 0)
