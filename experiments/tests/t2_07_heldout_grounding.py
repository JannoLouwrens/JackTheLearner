"""T2.07 — Grounding generalises to held-out phrasings.

HYPOTHESIS (registry). Commands never seen in training map to the right
anchor. Falsified by: accuracy collapses on held-out synonyms. Null:
memorising the ACTION_CATEGORIES synonym table. Metric:
heldout_retrieval_acc. Kills: SemanticActionAnchors as a grounding mechanism.

WHAT THIS MEASURES, SAID PLAINLY. T2.06 proved the shipped grounding path
(UnifiedBrain(llm_enabled=False, enable_semantic_anchors=True), trained by
compute_language_grounding_loss, tokenized by grounding_fallback_tokens) can
bind SEEN phrasings to anchors. T2.07 asks the question T2.06 explicitly
deferred: does the binding COMPOSE — does a phrasing the model never trained
on, but whose words carry category evidence from phrasings it did train on,
map to the right anchor? The system under claim is unchanged; only the
train/eval split of the shipped 33-phrasing table moves, and the split is
COMMITTED HERE, not drawn per seed.

THE HELD-OUT SET IS SMALL BECAUSE THE SHIPPED TABLE IS SMALL, and the
construction is the science, so it is spelled out. The fallback vocabulary
has 19 real words; 14 of the 33 shipped phrasings tokenize to all-<pad> and
are BY CONSTRUCTION indistinguishable from empty input (T2.06's recorded
ceiling). An honest held-out probe must therefore be: (a) NON-EMPTY under the
shipped tokenizer — an all-pad probe tests the vocabulary, not the mechanism;
(b) TOKEN-DISTINCT from every trained phrasing — "stand still" tokenizes
identically to "stand", so holding one out while training the other feeds the
model a bit-identical input and tests memorisation reach, not generalisation
(stand is excluded for exactly this; crouch has one tokenizable phrasing and
wave has none, so neither can split). What survives is FIVE probes over five
categories, committed as HELDOUT below:

    walk forward -> [walk,forward]   trained walk evidence: "walk"; "forward"
                                     also appears in run's "run forward"
    run fast     -> [run,fast]       "run" trained; "fast" in-vocab but
                                     UNSEEN in training (untrained embedding)
    jump in place-> [jump,in,place]  "jump" trained; "in"/"place" unseen
    rotate left / go left  -> [<pad>,left]   "left" trained only in turn_left
    rotate right / go right-> [<pad>,right]  mirror

("rotate left" and "go left" collapse to ONE token sequence; the unit of
evaluation is the UNIQUE TOKEN SEQUENCE — 5 of them — because the model's
prediction is a deterministic function of the sequence, and counting string
duplicates would be pseudoreplication.) Two probes deliberately carry
in-vocab-but-never-trained words: a grounding mechanism derailed by any novel
word it has an embedding slot for does NOT generalise, and that is the kills
clause doing its job — genuine falsification risk, left in on purpose.

THE NULL, as the registry names it: memorising the synonym table. Two
memorisers are computed and must score ZERO on the held-out set BY
CONSTRUCTION (asserted as rig, not gated as claim): exact-string lookup and
exact-token-sequence lookup from the trained phrasings. Construction
violated -> VOID. The stronger lexical comparator — nearest-trained-sequence
by token overlap — is REPORTED as context, never gated (T2.06's
acc_tfidf_train precedent: a comparator the mechanism can at best tie is not
a threshold).

RESOLVABILITY REFERENCE (VOID, not FAIL — T1.02: indict the task, not the
mechanism). A multinomial Naive Bayes over non-pad token counts (Laplace
alpha=1, uniform class prior, committed here) trained on the 26 trained
phrasings must map >= 4 of the 5 held-out sequences correctly, else the split
is not lexically resolvable by even the simplest word-evidence learner and
gating the shipped model on it would be meaningless.

PRE-REGISTERED GATES (exogenous, written before first run):
  CLAIM    heldout_correct >= 4 of 5 unique sequences on EVERY seed
           (language -> anchor argmax vs the true category; exact binomial
           null at chance 1/8: P(>=4/5) = 1.10e-3 per seed. n=5 is the
           ceiling the shipped table imposes; 5-sigma per-seed gates are
           arithmetically unreachable at n=5 and pretending otherwise would
           be theater — the small-n is stated, not hidden).
  CONTROL  (label shuffle — LAW 2's canonical sabotage, made concrete): a
           twin trained with the anchor-label lookup PERMUTED by the fixed
           derangement cat -> (cat+1) mod 8 (instance-level override of
           semantic_anchors.get_anchor_for_label, the single supervision
           site for losses 1 and 2; loss 3 is pairwise and label-free).
           Lexical evidence then points at permuted anchors, so held-out
           accuracy vs TRUE categories must collapse: ctrl_heldout_correct
           < 4 on EVERY seed, else the ruler leaks and the run is VOID.
           Alive-twin proof (LESSONS: an at-chance control must carry proof
           its instrument was alive): the control's loss must fall, gated.
           Diagnostic reported, never gated: ctrl_perm_correct — held-out
           predictions matching the PERMUTED target; high means the
           generalisation machinery works and followed the sabotage.
  RIG -> VOID, not FAIL: non-finite loss either twin; eval not
           bit-deterministic across two passes (dropout scar); < 6
           tokenizable categories (T2.06's vocabulary-degeneration gate);
           held-out construction broken (any probe empty, any probe
           token-identical to a trained sequence, either memoriser > 0);
           NB reference < 4/5; SEEN-FIT: the claim model must map >= n-1 of
           the n unique non-empty TRAINED sequences correctly on every seed
           (a model that cannot fit its own supervision cannot arbitrate
           generalisation — T2.02's principle); loss must fall on every
           seed for BOTH twins (last-100-step mean < first-100-step mean).

GPU. One submission for the whole spec (module cache; run_spec calls
_experiment per seed — the 5.5-GPU-hour scar). Sizing from T2.06's measured
production numbers (probe kernel jack-ladder-1786713772, P100: 0.1444 s/step
at BATCH 64 with the shipped loss): 2 twins x 3 seeds x 1000 steps x 0.1444
~= 866 s + builds + eval + ~300 s clone ~= 0.33 h. est_hours 0.4, timeout
generous (Kaggle bills the kernel's own window). Science lives HERE; the JOB
string only imports it (T0.16).

COVERS: language (parent) (claim), generality (claim).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..gpu import build_job, submit
from .t2_06_language_action_alignment import gen_action

# The claim is about the shipped grounding mechanism, measured through
# T2.06's committed action fixture; both hash into the certificate.
IMPL_DEPS = ["UnifiedBrain.py",
             "experiments/tests/t2_06_language_action_alignment.py"]

SEEDS = [0, 1, 2]

N_TRAIN = 2000
STEPS = 1000
BATCH = 64
LR = 3e-4
LOSS_WIN = 100                 # steps averaged at each end for the fall gate

CLAIM_MIN = 4                  # of the 5 unique held-out sequences, per seed
NB_REF_MIN = 4                 # resolvability floor for the NB reference
MIN_TOKENIZABLE_CATS = 6       # T2.06's vocabulary-degeneration rig gate
N_HELDOUT_UNIQUE = 5           # asserted, not assumed
PERM_SHIFT = 1                 # control derangement: cat -> (cat+1) mod 8

# The committed split: these strings are NEVER shown in training; everything
# else in the shipped table (including its all-pad phrasings) is trainable.
HELDOUT = {
    "walk": ["walk forward"],
    "run": ["run fast"],
    "jump": ["jump in place"],
    "turn_left": ["rotate left", "go left"],
    "turn_right": ["rotate right", "go right"],
}
# Excluded from probing, with reasons (see docstring): stand ("stand still"
# is token-identical to "stand"), crouch (one tokenizable phrasing), wave
# (zero tokenizable phrasings).


def _binom_p_geq(k: int, n: int, p: float) -> float:
    """Exact P(X >= k), X ~ Binomial(n, p) — the null the claim is gated on."""
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                     for i in range(k, n + 1)))


def _nb_reference(train_seqs: list, train_cats: list, probe_seqs: list,
                  n_cats: int) -> list:
    """Multinomial NB over non-pad token ids: Laplace alpha=1, uniform class
    prior (committed). Deterministic argmax (first index wins ties)."""
    vocab_ids = sorted({t for s in train_seqs for t in s if t != 0})
    if not vocab_ids:
        return [-1] * len(probe_seqs)
    col = {t: i for i, t in enumerate(vocab_ids)}
    counts = np.ones((n_cats, len(vocab_ids)))            # Laplace alpha=1
    for s, c in zip(train_seqs, train_cats):
        for t in s:
            if t != 0:
                counts[c, col[t]] += 1
    logp = np.log(counts / counts.sum(1, keepdims=True))
    preds = []
    for s in probe_seqs:
        score = np.zeros(n_cats)
        for t in s:
            if t != 0 and t in col:
                score += logp[:, col[t]]
        preds.append(int(score.argmax()))
    return preds


# ── remote entry point (also the local smoke, scaled down in STEPS only —
#    the CONFIG stays production: the T2.04 lesson) ─────────────────────────
def remote_run(seeds: list, n_train: int = N_TRAIN,
               steps: int = STEPS) -> dict:
    import torch
    from UnifiedBrain import (UnifiedBrain, UnifiedBrainConfig,
                              SemanticActionAnchors,
                              compute_language_grounding_loss,
                              grounding_fallback_tokens)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = list(SemanticActionAnchors.ACTION_CATEGORIES.keys())
    syns = SemanticActionAnchors.ACTION_CATEGORIES
    n_cats = len(names)
    cat_of = {n: i for i, n in enumerate(names)}

    def seq_of(phrase: str) -> tuple:
        return tuple(grounding_fallback_tokens([phrase])[0].tolist())

    # ── the committed split, verified at runtime (rig, recorded) ──────────
    heldout_strings = {p for ps in HELDOUT.values() for p in ps}
    trained = [(p, cat_of[n]) for n in names for p in syns[n]
               if p not in heldout_strings]
    trained_seq_set = {seq_of(p) for p, _ in trained}
    probes = {}                                # unique seq -> true cat idx
    for n, ps in HELDOUT.items():
        for p in ps:
            s = seq_of(p)
            assert probes.get(s, cat_of[n]) == cat_of[n]
            probes[s] = cat_of[n]
    probe_seqs = sorted(probes)                # deterministic order
    probe_cats = [probes[s] for s in probe_seqs]
    construction_ok = (
        len(probe_seqs) == N_HELDOUT_UNIQUE
        and all(sum(s) > 0 for s in probe_seqs)
        and not (set(probe_seqs) & trained_seq_set))

    # the registry's null: memorisers of the trained table, on held-out input
    str_lookup = {p: c for p, c in trained}
    seq_lookup = {seq_of(p): c for p, c in trained}
    mem_str = float(np.mean([                  # exact-string memoriser
        str_lookup.get(p, -1) == cat_of[n]
        for n, ps in HELDOUT.items() for p in ps]))
    mem_seq = float(np.mean([seq_lookup.get(s, -1) == c
                             for s, c in zip(probe_seqs, probe_cats)]))

    # context comparator (never gated): nearest trained sequence by count of
    # shared non-pad tokens, deterministic first-index tiebreak
    def _overlap_nn(s):
        best, arg = -1, -1
        for q, c in zip(sorted(seq_lookup), [seq_lookup[q] for q in
                                             sorted(seq_lookup)]):
            ov = len(set(s) & set(q) - {0})
            if ov > best:
                best, arg = ov, c
        return arg
    nn_correct = int(sum(_overlap_nn(s) == c
                         for s, c in zip(probe_seqs, probe_cats)))

    # resolvability reference (rig): NB over the trained phrasings
    tr_seqs = [seq_of(p) for p, _ in trained]
    tr_cats = [c for _, c in trained]
    nb_correct = int(sum(p == c for p, c in
                         zip(_nb_reference(tr_seqs, tr_cats, probe_seqs,
                                           n_cats), probe_cats)))

    # seen-fit probes: every unique non-empty TRAINED sequence (category is
    # unambiguous — asserted)
    seen_map = {}
    for p, c in trained:
        s = seq_of(p)
        if sum(s) > 0:
            assert seen_map.get(s, c) == c
            seen_map[s] = c
    seen_seqs = sorted(seen_map)
    seen_cats = [seen_map[s] for s in seen_seqs]

    tokenizable = sum(1 for n in names
                      if any(sum(seq_of(p)) > 0 for p in syns[n]))

    out = {"gpu": (torch.cuda.get_device_name(0) if device == "cuda"
                   else "cpu"),
           "n_cats_tokenizable": int(tokenizable),
           "construction_ok": bool(construction_ok),
           "n_heldout_unique": len(probe_seqs),
           "n_seen_unique": len(seen_seqs),
           "mem_str_acc": mem_str, "mem_seq_acc": mem_seq,
           "nn_overlap_correct": nn_correct,
           "nb_ref_correct": nb_correct,
           "null_p_claim": _binom_p_geq(CLAIM_MIN, len(probe_seqs), 1 / 8),
           "seeds": []}

    perm = [(c + PERM_SHIFT) % n_cats for c in range(n_cats)]

    def train_one(seed_offset, phr, cats, acts, states, permute=False):
        torch.manual_seed(seed_offset)
        cfg = UnifiedBrainConfig(llm_enabled=False,
                                 enable_semantic_anchors=True)
        model = UnifiedBrain(cfg).to(device)
        if permute:
            # LAW 2's label shuffle at the single supervision site. The
            # shipped lookup is reproduced then permuted; instance attribute
            # shadows the bound method for losses 1 and 2.
            orig = SemanticActionAnchors.get_anchor_for_label
            sa = model.semantic_anchors
            sa.get_anchor_for_label = lambda l: perm[orig(sa, l)]
        S = torch.tensor(states, dtype=torch.float32, device=device)
        AT = torch.tensor(acts, dtype=torch.float32, device=device)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        g = torch.Generator().manual_seed(seed_offset * 7 + 1)
        model.train()
        losses = []
        finite = True
        for _ in range(steps):
            idx = torch.randint(0, len(S), (BATCH,), generator=g)
            loss, _parts = compute_language_grounding_loss(
                model, S[idx], AT[idx], [phr[i] for i in idx.tolist()])
            if not bool(torch.isfinite(loss)):
                finite = False
                break
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        w = min(LOSS_WIN, max(1, len(losses) // 2))
        first = float(np.mean(losses[:w])) if losses else float("nan")
        last = float(np.mean(losses[-w:])) if losses else float("nan")
        return model, first, last, finite

    def lang_preds(model, phrases):
        model.eval()
        with torch.no_grad():
            toks = grounding_fallback_tokens(phrases).to(device)
            lang = model.language_encoder(toks)
            _sel, probs = model.semantic_anchors(lang)
            return probs.argmax(-1).cpu().numpy()

    probe_strs = [next(p for n, ps in HELDOUT.items() for p in ps
                       if seq_of(p) == s) for s in probe_seqs]
    seen_strs = [next(p for p, _ in trained if seq_of(p) == s)
                 for s in seen_seqs]

    for seed in seeds:
        rng = np.random.RandomState(int(seed) * 100_003 % 2**32)
        cats_tr = rng.randint(0, n_cats, size=n_train)
        by_cat = {}
        for p, c in trained:
            by_cat.setdefault(c, []).append(p)
        phr_tr = [by_cat[int(c)][rng.randint(len(by_cat[int(c)]))]
                  for c in cats_tr]
        acts_tr = np.stack([gen_action(int(c), rng) for c in cats_tr])
        states_tr = 0.1 * rng.randn(n_train, 256)

        model, lf, ll, fin = train_one(int(seed), phr_tr, cats_tr, acts_tr,
                                       states_tr)
        p1 = lang_preds(model, probe_strs)
        p2 = lang_preds(model, probe_strs)             # determinism assert
        s1 = lang_preds(model, seen_strs)
        det = bool(np.array_equal(p1, p2))
        heldout_correct = int((p1 == np.array(probe_cats)).sum())
        seen_correct = int((s1 == np.array(seen_cats)).sum())

        mc, clf, cll, cfin = train_one(int(seed) + 7, phr_tr, cats_tr,
                                       acts_tr, states_tr, permute=True)
        cp = lang_preds(mc, probe_strs)
        ctrl_heldout = int((cp == np.array(probe_cats)).sum())
        ctrl_perm = int((cp == np.array([perm[c] for c in probe_cats])).sum())

        out["seeds"].append({
            "seed": int(seed),
            "heldout_correct": heldout_correct,
            "seen_correct": seen_correct,
            "ctrl_heldout_correct": ctrl_heldout,
            "ctrl_perm_correct": ctrl_perm,
            "probe_preds": [int(x) for x in p1],
            "ctrl_preds": [int(x) for x in cp],
            "loss_first": round(lf, 5), "loss_last": round(ll, 5),
            "loss_ctrl_first": round(clf, 5),
            "loss_ctrl_last": round(cll, 5),
            "finite": bool(fin and cfin),
            "det_ok": det,
        })
    return out


# ── GPU submission (one per spec — module cache, T2.01 pattern) ───────────
JOB = r'''
import json, os
from experiments.tests.t2_07_heldout_grounding import remote_run
out = remote_run(__SEEDS__)
json.dump(out, open(os.path.join(os.environ["JACK_OUT"], "t207.json"), "w"),
          indent=1)
print("DONE", json.dumps(out["seeds"][0]), flush=True)
'''

_CACHE: dict = {}


def _submit(seeds: list) -> dict:
    body = JOB.replace("__SEEDS__", repr(list(seeds)))
    job = build_job(body)
    res = submit(job, prefer="kaggle",
                 est_hours=0.4,
                 timeout_s=7200,
                 fetch=["t207.json"])
    if not res.ok:
        raise RuntimeError(f"T2.07 job failed on {res.backend}: {res.message}")
    out = json.loads(Path(res.artifacts["t207.json"]).read_text())
    out["backend"] = res.backend
    return out


def _experiment(seed: int) -> dict:
    if not _CACHE:
        _CACHE.update(_submit(SEEDS))
    rows = _CACHE["seeds"]
    n_hu = _CACHE["n_heldout_unique"]
    return {
        "gpu": _CACHE["gpu"], "backend": _CACHE["backend"],
        "n_cats_tokenizable": _CACHE["n_cats_tokenizable"],
        "construction_ok": _CACHE["construction_ok"],
        "n_heldout_unique": n_hu,
        "n_seen_unique": _CACHE["n_seen_unique"],
        "mem_str_acc": _CACHE["mem_str_acc"],
        "mem_seq_acc": _CACHE["mem_seq_acc"],
        "nn_overlap_correct": _CACHE["nn_overlap_correct"],
        "nb_ref_correct": _CACHE["nb_ref_correct"],
        "null_p_claim": _CACHE["null_p_claim"],
        "heldout_correct": [r["heldout_correct"] for r in rows],
        "heldout_correct_min": min(r["heldout_correct"] for r in rows),
        "heldout_retrieval_acc": round(
            min(r["heldout_correct"] for r in rows) / n_hu, 4),
        "seen_correct_min": min(r["seen_correct"] for r in rows),
        "probe_preds": [r["probe_preds"] for r in rows],
        "loss_fell_all": all(r["loss_last"] < r["loss_first"] for r in rows),
        "det_ok_all": all(r["det_ok"] for r in rows),
        "finite_all": all(r["finite"] for r in rows),
    }


def _control(seed: int) -> dict:
    rows = _CACHE["seeds"]
    return {
        "ctrl_heldout_correct": [r["ctrl_heldout_correct"] for r in rows],
        "ctrl_heldout_max": max(r["ctrl_heldout_correct"] for r in rows),
        "ctrl_perm_correct": [r["ctrl_perm_correct"] for r in rows],
        "ctrl_loss_fell_all": all(r["loss_ctrl_last"] < r["loss_ctrl_first"]
                                  for r in rows),
    }


def _check(m: dict, c: dict):
    # Rig first: an invalid run is VOID, not evidence about the hypothesis.
    if not m["finite_all"]:
        return Status.VOID          # training diverged; nothing was measured
    if not m["det_ok_all"]:
        return Status.VOID          # eval not deterministic (dropout scar)
    if m["n_cats_tokenizable"] < MIN_TOKENIZABLE_CATS:
        return Status.VOID          # the vocabulary degenerated the task
    if not m["construction_ok"] or m["mem_str_acc"] != 0.0 \
            or m["mem_seq_acc"] != 0.0:
        return Status.VOID          # held-out construction broken: a probe
                                    # is empty, duplicated, or memoriser-
                                    # reachable — the split, not the claim
    if m["nb_ref_correct"] < NB_REF_MIN:
        return Status.VOID          # simplest lexical learner fails: split
                                    # not resolvable — task indicted (T1.02)
    if m["seen_correct_min"] < m["n_seen_unique"] - 1:
        return Status.VOID          # cannot fit its own supervision — a
                                    # non-learner cannot arbitrate (T2.02)
    if not m["loss_fell_all"] or not c["ctrl_loss_fell_all"]:
        return Status.VOID          # a twin never trained; an at-chance
                                    # control must prove it was alive
    if c["ctrl_heldout_max"] >= CLAIM_MIN:
        return Status.VOID          # label-shuffled twin reaches the claim
                                    # bar: the ruler leaks (LAW 2)
    # The claim, on the worst seed.
    return m["heldout_correct_min"] >= CLAIM_MIN


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T2.07"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # Local, CPU, at the PRODUCTION CONFIG (the T2.04 lesson). Only
        # counts shrink: steps and dataset. Exercises both twins, the
        # permuted lookup, both eval passes, the split construction, the
        # memorisers, the NB reference and the determinism assert.
        import time
        t0 = time.time()
        out = remote_run([2], n_train=240, steps=8)
        dt = time.time() - t0
        print(json.dumps(out, indent=1))
        row = out["seeds"][0]
        assert out["construction_ok"], "held-out construction broken"
        assert out["mem_str_acc"] == 0.0 and out["mem_seq_acc"] == 0.0, \
            "a memoriser reached the held-out set"
        assert out["nb_ref_correct"] >= NB_REF_MIN, \
            f"NB reference only {out['nb_ref_correct']}/5 — split not resolvable"
        assert row["finite"] and row["det_ok"], row
        print(f"SMOKE OK in {dt:.1f}s "
              f"(~{dt / (8 * 2):.2f} s/step upper bound incl. builds+eval)")
    else:
        run()
