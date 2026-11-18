"""Cost probe: what does the published BC-ResNet-1 keyword-spotting recipe cost here?

ORDERS Next-up 1, last open idle-slot row: Speech Commands keyword spotting
(PRIORS §7 "Reproduce first", row "ear": 12 classes, speaker-disjoint,
BC-ResNet-1 9.2K weights, 96.6-96.9%). The data is the paper's (N7-sc-task
REPRODUCES); this probe only sizes the LEARNER run, the way
`eco_rl_update_probe.py` and `rl2_cost_probe.py` did. It is a timing probe:
no learner is trained, no rule is judged, no ledger row.

Source (read 11-18, code pinned): `Qualcomm-AI-research/bcresnet@656f947a`
`main.py` — 200 epochs, 5 warmup, SGD lr 0.1 cosine to 0, momentum 0.9,
wd 1e-3, batch 100, num_workers 0; `utils.py` Preprocess — log-mel 40 bins,
n_fft 512, win 480, hop 160 at 16 kHz on 1 s clips (input 1x40x101), noise
mix p 0.8 from `_background_noise_`, SpecAugment off at tau 1. Model =
`experiments/kws_bcresnet/bcresnet.py` + `subspectralnorm.py`, VENDORED
VERBATIM (tau 1 -> BCResNets(8), 12 classes).

What is measured (process CPU time, 1 thread, nice 19):
  step_ms   one training step: forward + loss + backward + SGD at batch 100
  mel_ms    log-mel of 100 clips via torch.stft + a 40-bin mel matrix
            (torchaudio is not in either venv; its MelSpectrogram is the same
            STFT + matmul, so this is a stand-in for the cost, not the values)
Projection: train clips = 12-class train split as the authors' code builds it.
CORRECTED 11-18: `utils.py` SpeechCommand caps `_unknown_` and `_silence_` at
`sample_per_cls_v2` = [3077, 371, 408] (train/valid/test), so the balanced v2
splits are 36,923 / 4,445 / 4,890 (paper e-print, main.tex commented line under
Datasets; 4,890 = our N7-sc-testset count). The first version used the whole
84,843-clip train list and overstated the cost 2.3x -> steps/epoch = ceil(N/100); CPU-h = 200 x
steps/epoch x (step_ms + mel_ms) + eval. Reported, not judged.

    python -m experiments.kws_cost_probe
"""
import importlib.util
import json
import math
import os
import sys
import time

import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
VEND = os.path.join(HERE, "kws_bcresnet")


def load_model(tau=1):
    sys.path.insert(0, VEND)  # bcresnet.py imports subspectralnorm by bare name
    try:
        spec = importlib.util.spec_from_file_location("bcresnet", os.path.join(VEND, "bcresnet.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(VEND)
    return mod.BCResNets(int(tau * 8))


def mel_matrix(n_mels=40, n_fft=512, sr=16000):
    # HTK-style triangular filters (torchaudio's default mel_scale="htk", norm=None)
    def hz2mel(f):
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    nf = n_fft // 2 + 1
    freqs = torch.linspace(0, sr / 2, nf)
    pts = torch.tensor([mel2hz(m) for m in torch.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2).tolist()])
    fb = torch.zeros(nf, n_mels)
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - lo) / (c - lo)
        dn = (hi - freqs) / (hi - c)
        fb[:, i] = torch.clamp(torch.minimum(up, dn), min=0)
    return fb


def main(n_steps=30, warm=5):
    torch.manual_seed(0)
    model = load_model(1)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-3, momentum=0.9)
    lossf = torch.nn.CrossEntropyLoss()
    wav = torch.randn(100, 16000) * 0.1
    y = torch.randint(0, 12, (100,))
    fb = mel_matrix()
    win = torch.hann_window(480)

    def mel():
        s = torch.stft(wav, 512, hop_length=160, win_length=480, window=win, center=True, return_complex=True)
        p = s.abs() ** 2  # (100, 257, 101)
        m = torch.matmul(p.transpose(1, 2), fb).transpose(1, 2)
        return (m + 1e-6).log().unsqueeze(1)  # (100, 1, 40, 101)

    x = mel()
    assert tuple(x.shape) == (100, 1, 40, 101), x.shape
    model.train()

    def step():
        opt.zero_grad()
        loss = lossf(model(x), y)
        loss.backward()
        opt.step()

    for _ in range(warm):
        step()
        mel()
    t = time.process_time()
    for _ in range(n_steps):
        step()
    step_ms = (time.process_time() - t) / n_steps * 1e3
    t = time.process_time()
    for _ in range(n_steps):
        mel()
    mel_ms = (time.process_time() - t) / n_steps * 1e3
    model.eval()
    with torch.no_grad():
        t = time.process_time()
        for _ in range(n_steps):
            model(x)
        eval_ms = (time.process_time() - t) / n_steps * 1e3

    n_train, n_valid = 36923, 4445  # authors' balanced v2 splits
    spe = math.ceil(n_train / 100)
    train_h = 200 * spe * (step_ms + mel_ms) / 3.6e6
    # main.py validates every epoch on the valid split, tests once at the end
    eval_h = 200 * math.ceil(n_valid / 100) * (eval_ms + mel_ms) / 3.6e6
    out = {
        "torch": torch.__version__,
        "params": n_params,
        "step_ms": round(step_ms, 1),
        "mel_ms": round(mel_ms, 1),
        "eval_ms": round(eval_ms, 1),
        "steps_per_epoch": spe,
        "cpu_h_train_200ep": round(train_h, 2),
        "cpu_h_valid_200ep": round(eval_h, 2),
        "cpu_h_per_seed": round(train_h + eval_h, 2),
    }
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
