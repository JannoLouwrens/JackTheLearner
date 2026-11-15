"""LAB — AN EAR TAUGHT BY HIS OWN CLOCK: does predicting the next tick of sound
(the loss a world-model brain already pays) stand in for word labels, and is
his 5-per-second clock too slow for it? (No ledger row; a pilot.)

Seat: ASR (speech -> text), VACANT, arena NONE; touches Audio encoder,
UNDECIDED. PLAN part "Ear", right-hand column: "an ear trained by his brain's
own prediction of sound, with no word labels". END-FIT row served: 3 (words
reach the same brain through his own learned ear); closes none - every weight
here starts random and keeps learning, nothing is frozen in any arm.

THE PRIOR (PUBLISHED_PRIORS.md section 7; abstracts read 11-15 via the arXiv
API). Known: self-supervision helps SMALL keyword models when labels are
scarce - Data2Vec on three compact transformers, +8.22 to +11.18 points
absolute on label-deficient Speech Commands (Bovbjerg & Tan, 2210.01703);
predicting future frames learns phones (APC, 1904.03240: frame grain, 10 ms,
several frames ahead, 360 h). Known too: labels alone reach 90.3% at 125
recordings per word, falling fast below ~50. NOT in the record: whether the
plain next-step prediction a Dreamer-style brain does AT ITS DECISION CLOCK
(0.2 s = 20 frames a tick) teaches an ear at all. Next-frame prediction at
10 ms is solved by copying (why APC looks several frames ahead); at 200 ms
the next tick may be mostly unpredictable and teach only the mean. Biology
hears on two clocks (~25-40 ms and ~200 ms windows). Budget: theirs 100-960
h of speech; ours the ~5.5 h of single words in the pool - below the prior,
but in-domain, which is where Bovbjerg found the gain survives.

THE BET. An ear + small recurrent core (<= 100k weights) that first only
LISTENS (no labels; loss = predict the next tick's log-mel from the core
state) then gets N narrated hearings per word, beats the identical net given
only the N narrated hearings - and it works at his 200 ms clock.

ARMS (one architecture per clock; every arm then trains ALL weights on the
same N labels, same steps, same batches' seed)
  label   HOLDER (the record's recipe): random init, labels only.
  pred    listen first: core state h_t -> next tick's log-mel (MSE).
  wm      listen first: h_t -> this tick's log-mel AND the next tick's
          (what a world model hands him for free: reconstruct + predict).
  shuf    CONTROL THAT MUST FAIL: `pred`, but the "next tick" is taken from
          a different clip in the batch - same updates, no temporal truth.
CLOCKS    200 ms (20 frames a tick, 5 ticks a clip - his decision clock) and
          40 ms (4 frames, 25 ticks - a fast ear loop under the same brain).

DATA. Speech Commands v0.02, official speaker-disjoint lists. Ten words (yes
no up down left right on off stop go). Listening pool: 20,000 training clips
of all 35 words, no labels used. Narrated candidates: 300 per word from the
training list, never in the pool; a seed draws N of them. Validation 100 a
word (official list) picks each arm's own best checkpoint (300 or 1200 label
steps); the number reported is on the whole official test list for the ten
words. 40 log-mel bins, 25 ms window, 10 ms hop; +-100 ms random shift.

THE RULE (written before any seed is run; pilot seeds 0,1,2)
  Primary cell: clock 200 ms, N = 25 hearings per word, test accuracy.
  ALIVE iff  mean(best of pred, wm) - mean(label) >= +5.0 points
        AND  that arm beats label on 3 of 3 seeds
        AND  shuf's mean gain over label is under half of that arm's gain.
  The arm named "best of pred, wm" is chosen on VALIDATION accuracy, not test.
  DEAD otherwise. If the 200 ms clock is dead and 40 ms clears the same three
  clauses, the finding is "an ear needs a faster clock than his decisions"
  and that is what gets entered. N = 5 and 125 are reported, never gated.
  A pilot's ALIVE is not a claim: the claim gate is the same three clauses on
  5 fresh seeds (10-14), run by the workers, bars unmoved.
  Broken-code readings: label at N = 125 under 60% (the net cannot learn
  words at all), or shuf beating label by >= 5 (the pretext leaks labels).

USAGE  python -m experiments.ear_predict cache          # features, once
       python -m experiments.ear_predict run --seed 0   # one seed, all cells
       python -m experiments.ear_predict tally          # the table + the rule
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

ROOT = "/data/jack_corpora/speech_commands_v2"
OUT = "/data/tmp/ear_predict"
WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
SR, N_MELS, WIN, HOP, FRAMES = 16000, 40, 400, 160, 100
POOL_N, CAND_N, VAL_N = 20000, 300, 100
SHIFT = 10                       # frames = 100 ms
CLOCKS = {"200ms": 20, "40ms": 4}
ARMS = ["label", "pred", "wm", "shuf"]
NS = [5, 25, 125]
PRE_EPOCHS, LABEL_STEPS, CKPTS, BATCH = 8, 1200, (300, 1200), 64


# ---------------------------------------------------------------- features
def _mel_fb():
    def h2m(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def m2h(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    n_fft = 512
    pts = m2h(np.linspace(h2m(20.0), h2m(7600.0), N_MELS + 2))
    freqs = np.linspace(0, SR / 2, n_fft // 2 + 1)
    fb = np.zeros((N_MELS, len(freqs)), np.float32)
    for i in range(N_MELS):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    return fb


def _logmel(wavs):
    import torch
    x = torch.from_numpy(wavs)
    spec = torch.stft(x, 512, HOP, WIN, torch.hann_window(WIN), center=True,
                      return_complex=True).abs() ** 2          # B, 257, 101
    mel = torch.from_numpy(_mel_fb()) @ spec
    return torch.log(mel + 1e-6)[:, :, :FRAMES].transpose(1, 2).numpy()  # B, 100, 40


def _load(paths):
    import soundfile as sf
    out = np.zeros((len(paths), SR), np.float32)
    for i, p in enumerate(paths):
        w, sr = sf.read(os.path.join(ROOT, p), dtype="float32")
        assert sr == SR and w.ndim == 1
        out[i, :min(len(w), SR)] = w[:SR]
    feats = [_logmel(out[i:i + 512]) for i in range(0, len(out), 512)]
    return np.concatenate(feats)


def cache():
    os.makedirs(OUT, exist_ok=True)
    rd = lambda f: set(open(os.path.join(ROOT, f)).read().split())
    val, test = rd("validation_list.txt"), rd("testing_list.txt")
    every = sorted(os.path.relpath(p, ROOT) for p in glob.glob(ROOT + "/*/*.wav")
                   if "_background_noise_" not in p)
    train = [p for p in every if p not in val and p not in test]
    rng = np.random.default_rng(0)
    word = lambda p: p.split("/")[0]
    cand, cand_y = [], []
    for k, w in enumerate(WORDS):
        mine = [p for p in train if word(p) == w]
        pick = rng.choice(len(mine), CAND_N, replace=False)
        cand += [mine[i] for i in pick]; cand_y += [k] * CAND_N
    taken = set(cand)
    rest = [p for p in train if p not in taken]
    pool = [rest[i] for i in rng.choice(len(rest), POOL_N, replace=False)]
    v, vy, t, ty = [], [], [], []
    for k, w in enumerate(WORDS):
        mine = sorted(p for p in val if word(p) == w)
        pick = rng.choice(len(mine), VAL_N, replace=False)
        v += [mine[i] for i in pick]; vy += [k] * VAL_N
        mine = sorted(p for p in test if word(p) == w)
        t += mine; ty += [k] * len(mine)
    # speaker-disjointness, checked on the thing itself (speaker id = file prefix)
    spk = lambda ps: {os.path.basename(p).split("_nohash_")[0] for p in ps}
    assert not (spk(pool) | spk(cand)) & spk(t), "train speakers leak into test"
    assert not (spk(pool) | spk(cand)) & spk(v), "train speakers leak into validation"
    F = {n: _load(ps) for n, ps in [("pool", pool), ("cand", cand), ("val", v), ("test", t)]}
    mu, sd = F["pool"].mean((0, 1)), F["pool"].std((0, 1))
    F = {n: ((a - mu) / sd).astype(np.float16) for n, a in F.items()}
    np.savez(os.path.join(OUT, "feat.npz"), cand_y=np.array(cand_y), val_y=np.array(vy),
             test_y=np.array(ty), **F)
    print({n: a.shape for n, a in F.items()})


# ------------------------------------------------------------------- model
def build(T, seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    class Ear(nn.Module):
        def __init__(s):
            super().__init__()
            s.conv = nn.Conv1d(N_MELS, 32, 3, padding=1)       # inside one tick
            s.fc = nn.Linear(32 * T, 64)
            s.core = nn.GRU(64, 64, batch_first=True)
            s.word = nn.Linear(64, len(WORDS))
            s.now = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, T * N_MELS))
            s.nxt = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, T * N_MELS))

        def states(s, x):                                      # x: B, 100, 40
            B = x.shape[0]
            c = x.reshape(B * (FRAMES // T), T, N_MELS).transpose(1, 2)
            e = torch.relu(s.fc(torch.relu(s.conv(c)).flatten(1)))
            h, _ = s.core(e.reshape(B, FRAMES // T, 64))
            return h                                           # B, ticks, 64

        def logits(s, x):
            return s.word(s.states(x).mean(1))

    return Ear()


def _shift(x, rng):
    """Random +-100 ms shift, edge-padded (B, 100, 40) -> same shape."""
    import torch
    k = int(rng.integers(-SHIFT, SHIFT + 1))
    if k == 0:
        return x
    pad = x[:, :1].expand(-1, abs(k), -1) if k > 0 else x[:, -1:].expand(-1, abs(k), -1)
    return torch.cat([pad, x[:, :-k]], 1) if k > 0 else torch.cat([x[:, -k:], pad], 1)


def listen(net, arm, T, pool, seed):
    """The label-free phase. Returns the last epoch's losses (diagnostic)."""
    import torch
    rng = np.random.default_rng(1000 + seed)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    last = {}
    for ep in range(PRE_EPOCHS):
        order = rng.permutation(len(pool)); tot = []
        for i in range(0, len(order) - 127, 128):
            x = _shift(torch.from_numpy(pool[order[i:i + 128]].astype(np.float32)), rng)
            h = net.states(x)
            tk = x.reshape(x.shape[0], FRAMES // T, T * N_MELS)
            tgt = tk[torch.randperm(x.shape[0])] if arm == "shuf" else tk
            loss = ((net.nxt(h[:, :-1]) - tgt[:, 1:]) ** 2).mean()
            if arm == "wm":
                loss = loss + ((net.now(h) - tk) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot.append(float(loss))
        last = {"epoch": ep, "loss": float(np.mean(tot))}
    return last


def _acc(net, X, y):
    import torch
    with torch.no_grad():
        p = torch.cat([net.logits(torch.from_numpy(X[i:i + 1000].astype(np.float32))).argmax(1)
                       for i in range(0, len(X), 1000)])
    return float((p.numpy() == y).mean())


def narrate(net, X, y, F, seed):
    """N labelled hearings per word; ALL weights learn. Returns per-checkpoint val/test."""
    import torch
    rng = np.random.default_rng(2000 + seed)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xt, yt = torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y)
    out = {}
    for step in range(1, LABEL_STEPS + 1):
        idx = torch.from_numpy(rng.integers(0, len(X), min(BATCH, len(X))))
        loss = torch.nn.functional.cross_entropy(net.logits(_shift(Xt[idx], rng)), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step in CKPTS:
            net.eval()
            out[step] = {"val": _acc(net, F["val"], F["val_y"]),
                         "test": _acc(net, F["test"], F["test_y"])}
            net.train()
    best = max(out, key=lambda s: out[s]["val"])
    return {"val": out[best]["val"], "test": out[best]["test"], "ckpt": best, "all": out}


def run(seed):
    import copy
    import torch
    torch.set_num_threads(2)
    F = dict(np.load(os.path.join(OUT, "feat.npz")))
    rng = np.random.default_rng(seed)
    draw = {N: np.concatenate([k * CAND_N + rng.choice(CAND_N, N, replace=False)
                               for k in range(len(WORDS))]) for N in NS}
    path = os.path.join(OUT, f"seed{seed}.jsonl")
    for clock, T in CLOCKS.items():
        for arm in ARMS:
            t0 = time.time()
            base = build(T, seed)                 # same init for every arm of a seed
            pre = listen(base, arm, T, F["pool"], seed) if arm != "label" else {}
            for N in NS:
                net = copy.deepcopy(base)
                r = narrate(net, F["cand"][draw[N]], F["cand_y"][draw[N]], F, seed)
                row = {"seed": seed, "clock": clock, "arm": arm, "N": N, **r, "pre": pre,
                       "params": sum(p.numel() for n, p in net.named_parameters()
                                     if not n.startswith(("now", "nxt"))),
                       "secs": round(time.time() - t0, 1)}
                with open(path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps({k: row[k] for k in ("seed", "clock", "arm", "N", "val", "test", "ckpt", "secs")}),
                      flush=True)


def tally(seeds):
    rows = [json.loads(l) for s in seeds for l in open(os.path.join(OUT, f"seed{s}.jsonl"))]
    cell = lambda c, a, N, k: np.array([r[k] for s in seeds for r in rows
                                        if (r["seed"], r["clock"], r["arm"], r["N"]) == (s, c, a, N)])
    for c in CLOCKS:
        print(f"\nclock {c}   test accuracy, mean over seeds {seeds} (per seed)")
        for N in NS:
            print(f"  N={N:<4}" + "  ".join(
                f"{a} {100 * cell(c, a, N, 'test').mean():5.1f} ({' '.join(f'{100 * v:.1f}' for v in cell(c, a, N, 'test'))})"
                for a in ARMS))
        lab = cell(c, "label", 25, "test")
        ch = max(("pred", "wm"), key=lambda a: cell(c, a, 25, "val").mean())
        g = 100 * (cell(c, ch, 25, "test") - lab)
        gs = 100 * (cell(c, "shuf", 25, "test") - lab)
        ok = g.mean() >= 5.0 and (g > 0).all() and gs.mean() < g.mean() / 2
        print(f"  RULE N=25: challenger (by validation) = {ch}; gain {g.mean():+.1f} "
              f"(per seed {' '.join(f'{v:+.1f}' for v in g)}); shuf gain {gs.mean():+.1f} "
              f"-> {'ALIVE' if ok else 'DEAD'}")
        print(f"  broken-code readings: label N=125 = {100 * cell(c, 'label', 125, 'test').mean():.1f} "
              f"(needs >= 60); shuf gain {gs.mean():+.1f} (needs < 5)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "run", "tally"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    {"cache": cache, "run": lambda: run(a.seed), "tally": lambda: tally(a.seeds)}[a.cmd]()
