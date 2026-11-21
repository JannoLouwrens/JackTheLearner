"""LAB — ONE PARENT'S VOICE: does an ear raised on a single voice understand
strangers, and does hearing that voice "as a bigger or smaller person" close
the gap? (No ledger row; a pilot. Seat: ASR, VACANT, arena NONE. PLAN part
"Ear", right-hand column: "one voice that carries over to other voices".)

END-FIT rows served: 2 (trained here, handed to people — the people have
other voices) and 3 (advised in words through his own learned ear). Closes
none: every weight starts random and keeps learning; nothing pretrained is in
or around any arm.

THE PRIOR (PUBLISHED_PRIORS.md section 7; nothing re-read on the web today).
Known: small ears reach 90-97% on unseen speakers with 125-3,000 recordings
per word from ~2,600 SPEAKERS; "many voices" is in the record's take-as-is
list; synthetic speech alone gives 56.7% with 92 voices. Vocal-tract-length
perturbation (warping the frequency axis; Jaitly & Hinton 2013, from memory)
is a published augmentation — shown with hundreds of speakers already in the
data, ~1 point. NOT in the record: what ONE voice buys on strangers, and
whether the warp stands in for the missing voices. His world has one parent.
Budget: the record's rows need 10^3 recordings per word; one Speech Commands
speaker gives 5. So this is the few-hearings regime (LAB 11-15: many voices,
5 hearings per word = 37%, chance 10%) and says nothing about the 90% regime.

THE BET. At 5 hearings per word, an ear that hears ONE parent and warps the
mel axis of each hearing by a random factor (0.8-1.25: a longer or shorter
vocal tract) beats the same ear on the plain hearings, on strangers.
Kill: under +5.0 points, or the gain is not the warp's (see the control).

DATA. Speech Commands v0.02, ten words (yes no up down left right on off
stop go). Parents: training-list speakers with >= 5 recordings of every one
of the ten words (31 exist), ordered by default_rng(0).permutation of the
sorted ids. RULE parents = the first 12. Smoke parents = numbers 13 and 14,
never in the rule. Test: the whole official test list for the ten words
(4,074 recordings, speakers disjoint from every training speaker — asserted).
Features, net and training are `ear_predict`'s at the 200 ms clock (40
log-mel bins, 70k weights), Adam 1e-3, 600 steps, the whole 50-clip set per
step, +-100 ms shift in EVERY arm. The number is test accuracy at step 600;
no checkpoint is picked on strangers' labels.

ARMS (same initialisation per parent across arms)
  one        the parent's first 5 recordings of each word. The baseline.
  one+warp   THE BET: `one`, each hearing's mel axis resampled by a factor
             drawn log-uniform in [0.8, 1.25] at every step.
  one+noise  CONTROL THAT MUST FAIL: `one` plus Gaussian noise whose size
             (RMS) equals, batch by batch, the size of the change the warp
             would have made. Same disturbance, no vocal tract in it.
  many       REPORTED, the record's condition scaled down: 5 recordings per
             word from 50 different training speakers (not the parent).

THE RULE — DECLARED IN THIS COMMIT, BEFORE ANY RULE PARENT IS RUN (11-21):
  ALIVE iff mean(one+warp) - mean(one) >= +5.0 points over the 12 parents,
        AND one+warp beats one on >= 10 of 12 parents,
        AND one+noise's mean gain over one is under half of the warp's.
  DEAD otherwise. No extension; a miss is a miss.
  VOID (broken code, not a result): `many` under 18%, or `one` under 12%
  (nothing learned; chance is 10%). The bar for `many` was first drafted at
  25% from LAB 11-15's 37%; the smoke on the two spare parents read `many`
  .25 / .27 / .24 at 150, 300 and 600 steps alike (that 37% had 20,000
  clips of statistics and a checkpoint picked on strangers' labels), so the
  draft bar sat ON the healthy reading. Moved before any rule parent ran.
  Smoke, for the record (parents 13, 14; decides nothing): one .182 / .314,
  one+warp .259 / .340, one+noise .221 / .302, many .254 / .246.
  Reported beside it, deciding nothing: the one-to-many gap, and the share
  of it the warp closes.
  A pilot's ALIVE is not a claim. Cannot say, whatever happens: anything at
  the hundreds of hearings a real parent gives; a synthetic (TTS) parent;
  an ear trained inside a brain.

    python -m experiments.ear_one_voice smoke     # parents 13, 14
    python -m experiments.ear_one_voice run       # the 12 rule parents
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from experiments import ear_predict as E

N_HEAR = 5
STEPS = 600
WARP_LO, WARP_HI = 0.8, 1.25
N_RULE = 12
ARMS = ("one", "one+warp", "one+noise", "many")
OUT = "/data/jack-logs/ear_one_voice"


def _lists():
    rd = lambda f: set(open(os.path.join(E.ROOT, f)).read().split())
    val, test = rd("validation_list.txt"), rd("testing_list.txt")
    by = {}                                   # speaker -> word -> [paths]
    for w in E.WORDS:
        for f in sorted(os.listdir(os.path.join(E.ROOT, w))):
            p = f"{w}/{f}"
            if p in val or p in test:
                continue
            by.setdefault(f.split("_nohash_")[0], {}).setdefault(w, []).append(p)
    tests, ty = [], []
    for k, w in enumerate(E.WORDS):
        mine = sorted(p for p in test if p.split("/")[0] == w)
        tests += mine
        ty += [k] * len(mine)
    tspk = {os.path.basename(p).split("_nohash_")[0] for p in tests}
    assert not tspk & set(by), "a training speaker is in the test list"
    elig = sorted(s for s, d in by.items()
                  if all(len(d.get(w, [])) >= N_HEAR for w in E.WORDS))
    order = [elig[i] for i in np.random.default_rng(0).permutation(len(elig))]
    return by, order, tests, np.array(ty)


def _sets(by, parent, seed):
    """(one paths, many paths, labels): 5 per word, word-major order."""
    rng = np.random.default_rng(500 + seed)
    one, many, y = [], [], []
    others = sorted(s for s in by if s != parent)
    used = set()
    for k, w in enumerate(E.WORDS):
        one += by[parent][w][:N_HEAR]
        have = [s for s in others if w in by[s] and s not in used]
        pick = [have[i] for i in rng.choice(len(have), N_HEAR, replace=False)]
        used |= set(pick)
        many += [by[s][w][0] for s in pick]
        y += [k] * N_HEAR
    assert len({os.path.basename(p).split("_nohash_")[0] for p in many}) == len(many)
    return one, many, np.array(y)


def warp(x, alpha):
    """Resample the mel axis: out[..., j] = x[..., j / alpha], linear, edges
    held. x (B, frames, mels) torch; alpha (B,) torch."""
    import torch
    pos = (torch.arange(x.shape[2], dtype=torch.float32)[None, :] / alpha[:, None]
           ).clamp(0, x.shape[2] - 1)
    lo = pos.floor().long().clamp(max=x.shape[2] - 2)
    fr = (pos - lo)[:, None, :]
    idx = lo[:, None, :].expand(-1, x.shape[1], -1)
    return torch.gather(x, 2, idx) * (1 - fr) + torch.gather(x, 2, idx + 1) * fr


def fit(arm, X, y, seed):
    import torch
    net = E.build(E.CLOCKS["200ms"], seed)
    rng = np.random.default_rng(2000 + seed)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xt, yt = torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y)
    for _ in range(STEPS):
        xb = E._shift(Xt, rng)
        if arm in ("one+warp", "one+noise"):
            a = torch.from_numpy(np.exp(rng.uniform(np.log(WARP_LO), np.log(WARP_HI),
                                                    len(xb))).astype(np.float32))
            xw = warp(xb, a)
            if arm == "one+noise":
                rms = float((xw - xb).pow(2).mean().sqrt())
                xw = xb + torch.from_numpy(
                    rng.standard_normal(tuple(xb.shape)).astype(np.float32)) * rms
            xb = xw
        loss = torch.nn.functional.cross_entropy(net.logits(xb), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return net


def run(parents_idx, tag, have=None):
    """`have`: rows of an interrupted run of the SAME rule (every fit is
    seeded by its parent alone, so a row does not depend on the rows before
    it). They are kept, except that the LAST logged parent is fitted again
    whole and its logged rows must reproduce exactly."""
    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.process_time()
    by, order, tests, ty = _lists()
    plan = [(i, order[i]) + _sets(by, order[i], i) for i in parents_idx]
    train_paths = sorted({p for _, _, one, many, _ in plan for p in one + many})
    F = E._load(train_paths)
    mu, sd = F.mean((0, 1)), F.std((0, 1))          # what he heard, nothing else
    at = {p: i for i, p in enumerate(train_paths)}
    Xtest = ((E._load(tests) - mu) / sd).astype(np.float16)
    rows = []
    have = {(r["parent"], r["arm"]): r for r in (have or [])}
    redo = max((k[0] for k in have), default=None)
    for i, spk, one, many, y in plan:
        for arm in ARMS:
            if (i, arm) in have and i != redo:
                rows.append(have[(i, arm)])
                continue
            ps = many if arm == "many" else one
            X = (F[[at[p] for p in ps]] - mu) / sd
            acc = E._acc(fit(arm, X, y, i), Xtest, ty)
            rows.append({"parent": i, "speaker": spk, "arm": arm, "test": acc})
            print(json.dumps(rows[-1]), flush=True)
            if (i, arm) in have:
                assert have[(i, arm)] == rows[-1], ("resume does not reproduce",
                                                    have[(i, arm)], rows[-1])
    M = {a: np.array([r["test"] for r in rows if r["arm"] == a]) for a in ARMS}
    gw, gn = M["one+warp"] - M["one"], M["one+noise"] - M["one"]
    gap = float(M["many"].mean() - M["one"].mean())
    n = len(parents_idx)
    clauses = {"warp gain >= +5.0": float(gw.mean()) >= 0.05,
               f"warp ahead on >= 10 of {n}": int((gw > 0).sum()) >= 10,
               "noise gain < half the warp's": float(gn.mean()) < 0.5 * float(gw.mean())}
    void = []
    if M["many"].mean() < 0.18:
        void.append(f"many {M['many'].mean():.3f} < .18")
    if M["one"].mean() < 0.12:
        void.append(f"one {M['one'].mean():.3f} < .12")
    res = {"tag": tag, "parents": n, "test_clips": int(len(ty)),
           "mean": {a: round(float(v.mean()), 4) for a, v in M.items()},
           "sd": {a: round(float(v.std(ddof=1)), 4) for a, v in M.items()},
           "warp_gain": round(float(gw.mean()), 4), "warp_ahead": int((gw > 0).sum()),
           "noise_gain": round(float(gn.mean()), 4),
           "one_to_many_gap": round(gap, 4),
           "share_closed": round(float(gw.mean()) / gap, 3) if gap > 0 else None,
           "clauses": clauses, "void": void,
           "verdict": "VOID" if void else "ALIVE" if all(clauses.values()) else "DEAD",
           "cpu_s": round(time.process_time() - t0, 1)}
    json.dump({"result": res, "rows": rows}, open(os.path.join(OUT, f"{tag}.json"), "w"),
              indent=1)
    print(json.dumps(res, indent=1))
    return res


def main(argv):
    cmd = argv[0] if argv else ""
    if cmd == "smoke":
        run([12, 13], "smoke")
    elif cmd == "run":
        run(list(range(N_RULE)), "rule")
    elif cmd == "resume":           # argv[1] = the interrupted run's log
        rows = [json.loads(l) for l in open(argv[1]) if l.startswith('{"parent"')]
        run(list(range(N_RULE)), "rule", have=rows)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
