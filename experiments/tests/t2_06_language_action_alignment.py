"""T2.06 — Language-action alignment beats chance.

HYPOTHESIS (registry). Contrastive retrieval of the right action anchor from a
command beats chance and a bag-of-words baseline. Falsified by: at or near
chance (1/n_anchors). Null: chance = 1/8, plus TF-IDF nearest match.
Metric: retrieval_acc.

WHAT THIS MEASURES, SAID PLAINLY. `SemanticActionAnchors` (408,577 params) and
its shipped objective `compute_language_grounding_loss` have never received a
gradient from any caller — the config gates them off with exactly that note.
This is the first test that could fail for the mechanism the module claims:
that contrastive training binds COMMANDS and ACTION SEQUENCES into one anchor
space, so that a command retrieves the anchor its action actually lives at.
The system under claim is the shipped one: `UnifiedBrain(llm_enabled=False,
enable_semantic_anchors=True)` — the PLASTIC fallback text tower (the frozen
LLM lives in Jack's world, not in him) — trained by the shipped loss, tokenized
by the shipped `grounding_fallback_tokens` (single source; T0.16 lesson).

THE HEADLINE METRIC, retrieval_acc, is deliberately CROSS-MODAL: a held-out
(command, action) pair scores 1 only when the command's nearest anchor AND the
action's nearest anchor are BOTH the pair's true category. A text classifier
alone cannot earn it; a motion classifier alone cannot earn it; only an aligned
shared space can. Chance for one side is 1/8; gating the joint event against
1/8 is therefore conservative in the claim's favour being HARDER, not easier.

DATA. No external dataset exists (the CMU URLs 404 — T1.13/T2.04 precedent).
Commands are the shipped ACTION_CATEGORIES synonym phrasings — referenced, not
transcribed. Actions are a committed synthetic family: per category, a fixed
joint mask, per-joint pattern, frequency and envelope (constants drawn once
from RandomState(260814)), plus per-sample random phase, amplitude and noise.
Whether that fixture is LEARNABLE is not assumed: a nearest-centroid reference
arm on z-scored flattened actions must classify held-out actions >= 5 sigma
above chance or the run is VOID (task indicted, not the mechanism — T1.02).

KNOWN CEILING, measured through the shipped path, not patched around: the
fallback vocabulary has 20 words, so 12 of the 33 phrasings (all of "wave",
"sprint", "duck", "go straight"...) tokenize to all-<pad> and are
indistinguishable BY CONSTRUCTION. The language side's ceiling is ~64%, far
above every gate here; T2.07 (held-out phrasings) owns the generalisation
question. Rig gate: >= 6 of 8 categories must have at least one tokenizable
phrasing, else the vocabulary has degenerated and the run is VOID.

THE BAG-OF-WORDS NULL, pre-registered reading of "TF-IDF nearest match": the
zero-training lexical strategy — TF-IDF cosine between the command and each
anchor's NAME ("turn left", "walk", ...), no match -> wrong. The claim must
beat it on the LANGUAGE side per seed: alignment must know more than the words
it was named with (that "sprint" is a run is only in the paired data). The
OTHER reading — nearest TRAIN command — memorises the supervision verbatim and
scores 1.0 on seen phrasings by construction; it is T2.07's null ("memorising
the synonym table"), reported here as context, never gated: a gate no run
could pass is not a threshold, and gating on it here would pre-fail T2.07's
question one tier early.

PRE-REGISTERED GATES (all exogenous, per seed, before first run):
  CLAIM 1  z_binomial(retrieval_acc vs 1/8, n_test) >= 5 on EVERY seed
           (T2.01's sigma precedent; n_test=400 -> acc >= 0.208).
  CLAIM 2  acc_lang > acc_tfidf_name on EVERY seed.
  CONTROL  (registry: shuffled pairing collapses to chance) — a twin trained
           with actions shuffled across samples: its joint retrieval_acc must
           NOT beat 1/8 by 3 sigma on any seed. If it does, the ruler leaks
           and the run is VOID (a control that clears its gate — bakeoff
           lesson). Mechanism note: the shipped loss supervises language ->
           anchor from the label string itself, so the shuffle destroys the
           ACTION side's meaning (and with it the joint metric), which is
           exactly what "pairing collapses to chance" can honestly mean here.
           B3 (24th audit) — the observable distinguishing this twin
           "converged at chance" from "never trained": loss_ctrl_final,
           recorded per seed (the language->anchor loss term survives the
           shuffle, so a LIVE twin's loss falls while its joint acc stays at
           chance; a dead twin's loss does not move), plus the `finite` gate
           which covers both twins. Per-side control accuracies are computed
           but unrecorded — noted, not silently gated.
  RIG -> VOID, not FAIL: non-finite loss; eval not bit-deterministic across
           two passes (dropout scar — asserted, not hoped); reference arm
           below 5 sigma (task indicted); < 6 tokenizable categories;
           unbalanced test partition (50 per category asserted — gate on the
           minimum, ME.11 lesson).

GPU. One submission for the whole spec (module cache; run_spec calls
_experiment per seed — the 5.5-GPU-hour scar). Kaggle: W32's expiring hours
are assigned to GPU work that must not wait (Review 2026-08-14), and a Kaggle
kernel computes server-side even if the local watcher dies. Science code lives
HERE; the JOB string only imports it (T0.16).

COVERS: language (parent) (claim).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..gpu import build_job, submit

# The claim is about the shipped grounding mechanism; its file hashes into
# the certificate.
IMPL_DEPS = ["UnifiedBrain.py"]

SEEDS = [0, 1, 2]

N_TRAIN = 2000
N_TEST_PER_CAT = 50            # x8 categories = 400, balanced by construction
CHUNK, ACT_DIM = 16, 17        # action_encoder's contract: Linear(17*16, ...)
STEPS = 1000
BATCH = 64
LR = 3e-4

CHANCE = 1.0 / 8.0
Z_CLAIM = 5.0                  # per-seed binomial sigma vs chance (claim)
Z_CONTROL = 3.0                # the shuffled twin must stay under this
Z_REFERENCE = 5.0              # the centroid reference must clear this
MIN_TOKENIZABLE_CATS = 6

# ── the committed action family (drawn once; the literal seed is the spec) ──
_S = np.random.RandomState(260814)
_N_CATS = 8
_MASKS = (_S.rand(_N_CATS, ACT_DIM) < 0.45).astype(np.float64)
for _k in range(_N_CATS):                      # no category may be all-silent
    if _MASKS[_k].sum() < 3:
        _MASKS[_k, _S.choice(ACT_DIM, 3, replace=False)] = 1.0
_PATTERNS = _S.randn(_N_CATS, ACT_DIM) / math.sqrt(ACT_DIM)
_JOINT_PHASE = _S.uniform(0, 2 * np.pi, size=(_N_CATS, ACT_DIM))
_FREQS = np.array([1.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 3.5])  # walk run .. wave
_T = np.arange(CHUNK, dtype=np.float64)


def gen_action(cat: int, rng: np.random.RandomState) -> np.ndarray:
    """One (CHUNK, ACT_DIM) sequence for a category: committed mask/pattern/
    frequency + per-sample phase, amplitude and noise. Envelope by kind:
    periodic (walk/run/wave), burst (jump), silence (stand), ramp (turns),
    step-down (crouch)."""
    amp = rng.uniform(0.5, 1.0)
    phase = rng.uniform(0, 2 * np.pi)
    m, w = _MASKS[cat], _PATTERNS[cat]
    base = 0.3 * amp * m * w                             # constant signature
    if cat in (0, 1, 7):                                 # walk, run, wave
        t = 2 * np.pi * _FREQS[cat] * _T / CHUNK
        seq = amp * m * w * np.sin(t[:, None] + phase + _JOINT_PHASE[cat])
    elif cat == 2:                                       # jump: burst
        c = rng.uniform(4, 12)
        seq = amp * m * w * np.exp(-0.5 * ((_T[:, None] - c) / 2.0) ** 2)
    elif cat == 3:                                       # stand: signature only
        seq = np.zeros((CHUNK, ACT_DIM))
    elif cat in (4, 5):                                  # turns: +/- ramp
        sign = 1.0 if cat == 4 else -1.0
        seq = sign * amp * m * np.abs(w) * (_T[:, None] / CHUNK)
    else:                                                # crouch: step-down
        seq = -amp * m * np.abs(w) * (_T[:, None] >= CHUNK // 2)
    return (base + seq + 0.05 * rng.randn(CHUNK, ACT_DIM)).astype(np.float64)


# ── the bag-of-words null: TF-IDF cosine against anchor NAMES, no training ──
def _tfidf_name_predict(phrases: list, names: list) -> np.ndarray:
    docs = [n.replace("_", " ").split() for n in names]
    vocab = sorted({w for d in docs for w in d})
    n_docs = len(docs)
    idf = {w: math.log(n_docs / sum(1 for d in docs if w in d)) + 1.0
           for w in vocab}

    def vec(words):
        v = np.zeros(len(vocab))
        for i, w in enumerate(vocab):
            c = words.count(w)
            if c:
                v[i] = c * idf[w]
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    D = np.stack([vec(d) for d in docs])
    preds = np.empty(len(phrases), dtype=np.int64)
    for i, p in enumerate(phrases):
        sims = D @ vec(p.lower().split())
        preds[i] = int(sims.argmax()) if sims.max() > 0 else -1  # no match
    return preds


def _binom_z(acc: float, n: int, p0: float = CHANCE) -> float:
    return (acc - p0) / math.sqrt(p0 * (1 - p0) / n)


# ── remote entry point (also the local smoke, scaled down in STEPS only —
#    the CONFIG stays production: the T2.04 lesson) ─────────────────────────
def remote_run(seeds: list, n_train: int = N_TRAIN,
               n_test_per_cat: int = N_TEST_PER_CAT,
               steps: int = STEPS) -> dict:
    import torch
    import torch.nn.functional as F
    from UnifiedBrain import (UnifiedBrain, UnifiedBrainConfig,
                              SemanticActionAnchors,
                              compute_language_grounding_loss,
                              grounding_fallback_tokens)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = list(SemanticActionAnchors.ACTION_CATEGORIES.keys())
    syns = SemanticActionAnchors.ACTION_CATEGORIES
    n_cats = len(names)
    assert n_cats == _N_CATS

    # Rig: how many categories have >= 1 phrasing the shipped vocab can see?
    tokenizable = sum(
        1 for n in names
        if any(grounding_fallback_tokens([p]).abs().sum().item() > 0
               for p in syns[n]))

    out = {"gpu": (torch.cuda.get_device_name(0) if device == "cuda"
                   else "cpu"),
           "n_cats_tokenizable": int(tokenizable), "seeds": []}

    def draw(n, rng, balanced=False):
        cats = (np.repeat(np.arange(n_cats), n // n_cats) if balanced
                else rng.randint(0, n_cats, size=n))
        phrases = [syns[names[c]][rng.randint(len(syns[names[c]]))]
                   for c in cats]
        acts = np.stack([gen_action(int(c), rng) for c in cats])
        return cats, phrases, acts

    def train_one(seed_offset, cats, phrases, acts, states, shuffle_seed=None):
        """Build the shipped brain and run the shipped objective. If
        shuffle_seed is set, actions are permuted across samples (the
        control's information-free pairing)."""
        torch.manual_seed(seed_offset)
        cfg = UnifiedBrainConfig(llm_enabled=False,
                                 enable_semantic_anchors=True)
        model = UnifiedBrain(cfg).to(device)
        A = acts
        if shuffle_seed is not None:
            A = acts[np.random.RandomState(shuffle_seed).permutation(len(acts))]
        S = torch.tensor(states, dtype=torch.float32, device=device)
        AT = torch.tensor(A, dtype=torch.float32, device=device)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        g = torch.Generator().manual_seed(seed_offset * 7 + 1)
        model.train()
        last = float("nan")
        finite = True
        for _ in range(steps):
            idx = torch.randint(0, len(S), (BATCH,), generator=g)
            loss, _parts = compute_language_grounding_loss(
                model, S[idx], AT[idx], [phrases[i] for i in idx.tolist()])
            if not bool(torch.isfinite(loss)):
                finite = False
                break
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last = float(loss.item())
        return model, last, finite

    def eval_pairs(model, phrases, acts):
        """(pred_lang, pred_act), each (n,) anchor indices, eval mode."""
        import torch
        model.eval()
        with torch.no_grad():
            toks = grounding_fallback_tokens(phrases).to(device)
            lang = model.language_encoder(toks)
            _sel, probs = model.semantic_anchors(lang)
            pl = probs.argmax(-1).cpu().numpy()
            ae = model.semantic_anchors.encode_actions(
                torch.tensor(acts, dtype=torch.float32, device=device))
            an = F.normalize(model.semantic_anchors.anchors, dim=-1)
            pa = (ae @ an.T).argmax(-1).cpu().numpy()
        return pl, pa

    for seed in seeds:
        rng = np.random.RandomState(int(seed) * 100_003 % 2**32)
        cats_tr, phr_tr, acts_tr = draw(n_train, rng)
        states_tr = 0.1 * rng.randn(n_train, 256)
        n_test = n_test_per_cat * n_cats
        cats_te, phr_te, acts_te = draw(n_test, rng, balanced=True)
        per_cat = np.bincount(cats_te, minlength=n_cats)

        model, loss_f, fin = train_one(int(seed), cats_tr, phr_tr, acts_tr,
                                       states_tr)
        pl1, pa1 = eval_pairs(model, phr_te, acts_te)
        pl2, pa2 = eval_pairs(model, phr_te, acts_te)   # determinism assert
        det = bool(np.array_equal(pl1, pl2) and np.array_equal(pa1, pa2))
        acc_lang = float((pl1 == cats_te).mean())
        acc_act = float((pa1 == cats_te).mean())
        acc_joint = float(((pl1 == cats_te) & (pa1 == cats_te)).mean())

        # control: identical training, actions shuffled across samples
        mc, loss_c, fin_c = train_one(int(seed) + 7, cats_tr, phr_tr,
                                      acts_tr, states_tr,
                                      shuffle_seed=int(seed) + 41)
        cl, ca = eval_pairs(mc, phr_te, acts_te)
        acc_joint_ctrl = float(((cl == cats_te) & (ca == cats_te)).mean())

        # reference arm: nearest centroid, z-scored flat actions (T1.02)
        Ftr = acts_tr.reshape(len(acts_tr), -1)
        Fte = acts_te.reshape(len(acts_te), -1)
        mu, sd = Ftr.mean(0), Ftr.std(0)
        sd[sd < 1e-9] = 1e-9
        Ztr, Zte = (Ftr - mu) / sd, (Fte - mu) / sd
        cent = np.stack([Ztr[cats_tr == k].mean(0) for k in range(n_cats)])
        ref_pred = ((Zte[:, None, :] - cent[None]) ** 2).sum(-1).argmin(-1)
        acc_ref = float((ref_pred == cats_te).mean())

        # the bag-of-words null (and the memorisation context number)
        tf_name = _tfidf_name_predict(phr_te, names)
        acc_tfidf_name = float((tf_name == cats_te).mean())
        train_lookup = {p: int(c) for p, c in zip(phr_tr, cats_tr)}
        acc_tfidf_train = float(np.mean(
            [train_lookup.get(p, -1) == c for p, c in zip(phr_te, cats_te)]))

        out["seeds"].append({
            "seed": int(seed),
            "retrieval_acc": round(acc_joint, 4),
            "acc_lang": round(acc_lang, 4), "acc_act": round(acc_act, 4),
            "acc_joint_ctrl": round(acc_joint_ctrl, 4),
            "acc_ref": round(acc_ref, 4),
            "acc_tfidf_name": round(acc_tfidf_name, 4),
            "acc_tfidf_train": round(acc_tfidf_train, 4),
            "z_joint": round(_binom_z(acc_joint, n_test), 3),
            "z_ctrl": round(_binom_z(acc_joint_ctrl, n_test), 3),
            "z_ref": round(_binom_z(acc_ref, n_test), 3),
            "n_test": int(n_test),
            "min_per_cat": int(per_cat.min()),
            "loss_final": round(loss_f, 5),
            "loss_ctrl_final": round(loss_c, 5),
            "finite": bool(fin and fin_c),
            "det_ok": det,
        })
    return out


# ── GPU submission (one per spec — module cache, T2.01 pattern) ───────────
JOB = r'''
import json, os
from experiments.tests.t2_06_language_action_alignment import remote_run
out = remote_run(__SEEDS__)
json.dump(out, open(os.path.join(os.environ["JACK_OUT"], "t206.json"), "w"),
          indent=1)
print("DONE", json.dumps(out["seeds"][0]), flush=True)
'''

_CACHE: dict = {}


def _submit(seeds: list) -> dict:
    body = JOB.replace("__SEEDS__", repr(list(seeds)))
    job = build_job(body)
    # Sizing (16th-audit rule: measured at the PRODUCTION configuration on the
    # target GPU — probe kernel jack-ladder-1786713772, P100, 2026-08-14:
    # build 1.47 s, train 0.1444 s/step at BATCH 64 with the shipped loss,
    # eval 0.01 s per two passes over 400 rows. Total = STEPS x 2 arms x
    # 3 seeds = 6000 steps x 0.1444 = 866 s + 6 builds x 1.47 + eval ~= 875 s
    # compute + ~300 s clone/setup ~= 0.33 h measured. est_hours 0.4; timeout
    # generous — Kaggle bills the kernel's own metered window, so the cap
    # costs nothing.
    res = submit(job, prefer="kaggle",
                 est_hours=0.4,
                 timeout_s=7200,
                 fetch=["t206.json"])
    if not res.ok:
        raise RuntimeError(f"T2.06 job failed on {res.backend}: {res.message}")
    out = json.loads(Path(res.artifacts["t206.json"]).read_text())
    out["backend"] = res.backend
    return out


def _experiment(seed: int) -> dict:
    if not _CACHE:
        _CACHE.update(_submit(SEEDS))
    rows = _CACHE["seeds"]
    return {
        "gpu": _CACHE["gpu"], "backend": _CACHE["backend"],
        "retrieval_acc": [r["retrieval_acc"] for r in rows],
        "acc_lang": [r["acc_lang"] for r in rows],
        "acc_act": [r["acc_act"] for r in rows],
        "acc_ref": [r["acc_ref"] for r in rows],
        "acc_tfidf_name": [r["acc_tfidf_name"] for r in rows],
        "acc_tfidf_train": [r["acc_tfidf_train"] for r in rows],
        "z_joint_min": min(r["z_joint"] for r in rows),
        "z_ref_min": min(r["z_ref"] for r in rows),
        "n_cats_tokenizable": _CACHE["n_cats_tokenizable"],
        "min_per_cat": min(r["min_per_cat"] for r in rows),
        "beats_chance_all": all(r["z_joint"] >= Z_CLAIM for r in rows),
        "beats_tfidf_all": all(r["acc_lang"] > r["acc_tfidf_name"]
                               for r in rows),
        "ref_ok_all": all(r["z_ref"] >= Z_REFERENCE for r in rows),
        "det_ok_all": all(r["det_ok"] for r in rows),
        "finite_all": all(r["finite"] for r in rows),
        "loss_final_max": max(r["loss_final"] for r in rows),
    }


def _control(seed: int) -> dict:
    rows = _CACHE["seeds"]
    return {
        "acc_joint_ctrl": [r["acc_joint_ctrl"] for r in rows],
        "z_ctrl_max": max(r["z_ctrl"] for r in rows),
        "ctrl_beats_chance_any": any(r["z_ctrl"] >= Z_CONTROL for r in rows),
    }


def _check(m: dict, c: dict):
    # Rig first: an invalid run is VOID, not evidence about the hypothesis.
    if not m["finite_all"]:
        return Status.VOID          # training diverged; nothing was measured
    if not m["det_ok_all"]:
        return Status.VOID          # eval not deterministic (dropout scar)
    if m["n_cats_tokenizable"] < MIN_TOKENIZABLE_CATS:
        return Status.VOID          # the vocabulary degenerated the task
    if m["min_per_cat"] < N_TEST_PER_CAT:
        return Status.VOID          # unbalanced partition (ME.11 lesson)
    if not m["ref_ok_all"]:
        return Status.VOID          # simplest learner fails: task indicted,
                                    # not the mechanism (T1.02 lesson)
    if c["ctrl_beats_chance_any"]:
        return Status.VOID          # information-free pairing beat chance:
                                    # the ruler leaks (bakeoff lesson)
    return bool(m["beats_chance_all"] and m["beats_tfidf_all"])


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T2.06"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # Local, CPU, at the PRODUCTION CONFIG (the T2.04 lesson: a cost or a
        # behaviour measured on a shrunken config is not evidence about the
        # production one). Only counts shrink: steps and dataset. Exercises
        # both training arms, both eval passes, the largest seed's derived
        # RNG, the reference arm, both baselines and the balanced partition.
        import time
        t0 = time.time()
        out = remote_run([2], n_train=240, n_test_per_cat=10, steps=8)
        dt = time.time() - t0
        print(json.dumps(out, indent=1))
        row = out["seeds"][0]
        assert row["finite"] and row["det_ok"], row
        assert row["min_per_cat"] == 10, row
        print(f"SMOKE OK in {dt:.1f}s "
              f"(~{dt / (8 * 2):.2f} s/step upper bound incl. builds+eval)")
    else:
        run()
