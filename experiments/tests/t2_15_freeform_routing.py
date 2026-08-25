"""T2.15 — Free-form language routes to the right task.

HYPOTHESIS (registry). Novel paraphrases of known commands map to the correct
command cluster above chance (the LLM->task handoff). Falsified by: held-out
phrasings route at chance. Null: chance routing; bag-of-words retrieval.
Metric: paraphrase_routing_accuracy. Registered note: the verb x object grid
must be designed BEFORE grounding training (CAPABILITIES.md L2) or the
held-out cells cannot exist.

WHAT THIS MEASURES, SAID PLAINLY. The parent LLM lives in Jack's world and
speaks free-form language (GOAL.md); Jack's side of that handoff is a ROUTER:
a phrasing he has never trained on must land on the right task anchor. The
system under claim is the shipped routing path — UnifiedBrain(
llm_enabled=False, enable_semantic_anchors=True), trained by
compute_language_grounding_loss, tokenized by grounding_fallback_tokens
(single source, T0.16) — trained on a verb x modifier GRID designed before
training, exactly as the registry note orders. Routing = language-side anchor
argmax, T2.06's eval.

RELATION TO T2.07, stated because it is load-bearing. T2.07 asked whether the
binding COMPOSES FROM THE SHIPPED SYNONYM TABLE and settled FAIL (attempt 2,
2026-08-21: held-out [2,2,2] of 5 vs a 4/5 bar, NB reference 5/5) — the
shipped table supports only five probes and the mechanism transferred none of
the bar. T2.15 is the different, separately-registered question (registered
with its grid requirement BEFORE T2.07 ran): given training data DESIGNED for
composition — every held-out word carrying evidence from trained phrasings —
does routing generalise? A PASS here does not overturn T2.07 (the shipped
table still does not compose); a FAIL localises the defect in the MECHANISM
rather than the table. Neither outcome re-litigates T2.07, whose redesigns
remain routed through the Review.

THE DESIGNED GRID, committed here (all phrases inside the shipped 20-word
vocabulary — the vocabulary is a property of the shipped system and this spec
measures through it, not around it). Two composition rules, committed before
training: a content verb decides the cluster; the light verb "move" defers to
its modifier. Clusters are 7 of the 8 shipped anchors — WAVE IS AN EXPLICIT
SCOPE EXCLUSION (CAPABILITIES.md C8 precedent): the shipped vocabulary
contains zero words for it, so no in-vocab phrasing can express it; written
down, not silently missing. TRAINED: 32 unique token sequences. HELD-OUT: 16
unique sequences, never trained, each non-empty, each token-distinct from
every trained sequence, every word of each carrying training evidence
(verified at runtime; violation -> VOID, the split is indicted, not the
claim).

SUPERVISION, and why it is overridden for BOTH twins. The shipped
get_anchor_for_label maps by exact-synonym or substring and DEFAULTS TO WALK
on anything unknown — it cannot label a designed grid ("move fast" -> walk).
The grid's own phrase->cluster map is therefore installed at the single
supervision site (instance-level shadowing, T2.07's established machinery for
that exact site). This changes the TRAINING DATA — which the registry note
orders designed — never the learned mechanism: encoders, anchors, losses and
tokenizer are the shipped ones. The label-shuffle control composes its
derangement on top of the same map, so both twins differ by sabotage alone.

THE NULLS, as registered. (1) CHANCE, gated: the claim gate below. Chance is
taken at 1/7, not 1/8, although argmax ranges over all 8 anchors — the
harder reading (T2.06's precedent: conservative in the claim's favour being
HARDER). (2) BAG-OF-WORDS RETRIEVAL, reported per the registered null and
never gated: TF-IDF cosine nearest-TRAINED-phrasing, token-overlap nearest
sequence, and the NB reference are all computed on the held-out set. On a
grid DESIGNED to be lexically resolvable a bag-of-words learner may sit at
the ceiling; a comparator the mechanism can at best tie is not a threshold
(T2.06's acc_tfidf_train precedent). If the mechanism routes WORSE than
bag-of-words, that number is in the row and it is the Review's routing-seat
evidence — T2.07 already proved that outcome possible, which is what makes
this claim falsifiable rather than decorative.

PRE-REGISTERED GATES (exogenous, written before first run):
  CLAIM    heldout_correct >= 12 of the 16 unique held-out sequences on
           EVERY seed (language -> anchor argmax vs the true cluster).
           Exact binomial at chance 1/7: P(>=12/16) = 7.5e-8, beyond the
           one-sided 5-sigma convention (2.87e-7) of T2.01/T2.06, computed
           and recorded in-row as null_p_claim.
  CONTROL  label-shuffle twin (LAW 2): the grid lookup composed with the
           fixed derangement cat -> (cat+1) mod 8 at the single supervision
           site. Held-out accuracy vs TRUE clusters must stay under the
           claim bar on every seed (ctrl_heldout_correct < 12), else the
           ruler leaks and the run is VOID. Alive-twin proof (LESSONS): the
           control's loss must fall, gated. Diagnostic reported, never
           gated: ctrl_perm_correct vs the permuted targets.
  RIG -> VOID, not FAIL: non-finite loss either twin; eval not
           bit-deterministic across two passes (dropout scar); construction
           broken (any held-out probe empty, token-identical to a trained
           sequence, missing word evidence, or either memoriser above 0.0);
           NB reference < 13 of 16 (split not lexically resolvable — indict
           the task, T1.02); SEEN-FIT: the claim model must map >= n-1 of
           the 32 unique trained sequences on every seed (a model that
           cannot fit its supervision cannot arbitrate generalisation,
           T2.02); loss must fall on every seed for BOTH twins; the 8
           shipped category names in their committed order.

GPU. One submission for the whole spec (module cache; run_spec calls
_experiment per seed — the 5.5-GPU-hour scar). Sizing from T2.06's measured
production numbers (probe kernel jack-ladder-1786713772, P100, 0.1444 s/step
at BATCH 64 with the shipped loss): 2 twins x 3 seeds x 1000 steps x 0.1444
~= 866 s + builds + eval + ~300 s clone ~= 0.33 h. est_hours 0.4, timeout
generous (Kaggle bills the kernel's own window). Science lives HERE; the JOB
string only imports it (T0.16).

COVERS: language (parent) (claim).

FAIL RECORD (attempt 2, Kaggle P100, kernel jack-ladder-1787631708, 0.31 h
W34, ran 2026-08-25 04:40 UTC, head 20b8660; harvested by the next unskipped
iteration per the pace-skip contract). Every rig gate green: construction_ok,
memorisers 0.0/0.0, seen-fit 32/32 on every seed, loss fell on both twins,
deterministic eval, NB reference 14/16 (>= 13 bar — the split IS lexically
resolvable), control twin under the bar with falling loss
(ctrl_heldout_correct [2,1,2]). The CLAIM branch alone fired:
heldout_correct [8,9,5] of 16 vs the pre-registered 12/16-per-seed bar —
above chance (1/7 ~ 2.3/16) but under the bar on all three seeds, and on
seed 2 the mechanism (5/16) routes WORSE than both registered bag-of-words
nulls (TF-IDF 11/16, NB 14/16). Read with T2.07 (FAIL, shipped-table
composition), this localises the defect in the MECHANISM: a grid designed
for composition, provably resolvable by token overlap at 14/16, transfers at
most 9/16 through the shipped anchor-argmax router. That is a real
architecture measurement, not a harness fault or a seed lottery. Do NOT
re-dispatch T2.15 unchanged; the routing-seat question is routed to the
Review (REVIEW_QUEUE.md, t215-router-under-lexical-null) with its staleness
bill computed there.
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

# The claim is about the shipped routing mechanism, measured through T2.06's
# committed action fixture; both hash into the certificate.
IMPL_DEPS = ["UnifiedBrain.py",
             "experiments/tests/t2_06_language_action_alignment.py"]

SEEDS = [0, 1, 2]

N_TRAIN = 2000
STEPS = 1000
BATCH = 64
LR = 3e-4
LOSS_WIN = 100                 # steps averaged at each end for the fall gate

CHANCE = 1.0 / 7.0             # conservative: argmax ranges over 8 anchors
CLAIM_MIN = 12                 # of the 16 held-out sequences, per seed
NB_REF_MIN = 13                # resolvability floor for the NB reference
SEEN_FIT_SLACK = 1             # claim model may miss at most 1 trained seq
N_HELDOUT_UNIQUE = 16          # asserted, not assumed
N_TRAINED_UNIQUE = 32          # asserted, not assumed
PERM_SHIFT = 1                 # control derangement: cat -> (cat+1) mod 8

# The committed order of the shipped anchors; violated -> the mechanism under
# claim is not the one this file was written against.
CAT_NAMES = ["walk", "run", "jump", "stand", "turn_left", "turn_right",
             "crouch", "wave"]

# ── THE GRID (designed before training; the two composition rules are in the
#    docstring). Cluster -> phrases. Every word is in the shipped vocabulary.
TRAINED_GRID = {
    "walk":       ["walk", "walk forward", "walk fast", "walk backward",
                   "walk naturally", "move slow", "move naturally"],
    "run":        ["run", "run forward", "run fast", "run backward",
                   "move fast"],
    "jump":       ["jump", "jump up", "jump in place", "jump fast"],
    "stand":      ["stand", "idle", "stop", "stand in place", "stand idle"],
    "turn_left":  ["turn left", "left", "turn left slow", "move left"],
    "turn_right": ["turn right", "right", "turn right slow", "move right"],
    "crouch":     ["crouch", "crouch slow", "crouch in place"],
}
HELDOUT_GRID = {
    "walk":       ["walk slow", "walk in place", "walk forward slow"],
    "run":        ["run slow", "run in place", "run naturally"],
    "jump":       ["jump backward", "jump up fast"],
    "stand":      ["stand naturally", "stop in place", "idle in place"],
    "turn_left":  ["turn left fast", "move left fast"],
    "turn_right": ["turn right fast", "move right fast"],
    "crouch":     ["crouch fast"],
}
N_CLUSTERS = len(TRAINED_GRID)  # 7; wave is the documented scope exclusion


def _binom_p_geq(k: int, n: int, p: float) -> float:
    """Exact P(X >= k), X ~ Binomial(n, p) — the null the claim is gated on."""
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                     for i in range(k, n + 1)))


def _nb_reference(train_seqs: list, train_cats: list, probe_seqs: list,
                  n_cats: int) -> list:
    """Multinomial NB over non-pad token ids: Laplace alpha=1, uniform class
    prior (committed; T2.07's reference). Deterministic argmax."""
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


def _tfidf_retrieval(train_phrases: list, train_cats: list,
                     probe_phrases: list) -> list:
    """The registered bag-of-words null: TF-IDF cosine against the TRAINED
    phrasings, prediction = the nearest phrasing's cluster. Zero training,
    deterministic (first-index tiebreak). Reported, never gated."""
    docs = [p.split() for p in train_phrases]
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
    preds = []
    for p in probe_phrases:
        sims = D @ vec(p.split())
        preds.append(int(train_cats[int(sims.argmax())]) if sims.max() > 0
                     else -1)
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
    n_cats = len(names)
    cat_of = {n: i for i, n in enumerate(names)}
    names_ok = names == CAT_NAMES

    def seq_of(phrase: str) -> tuple:
        return tuple(grounding_fallback_tokens([phrase])[0].tolist())

    # ── the committed grid, verified at runtime (rig, recorded) ───────────
    trained = [(p, cat_of[n]) for n, ps in TRAINED_GRID.items() for p in ps]
    grid_lookup = {p: c for p, c in trained}
    trained_seq_set = {seq_of(p) for p, _ in trained}
    trained_words = {w for p, _ in trained for w in p.split()}

    probes = {}                                # unique seq -> true cat idx
    probes_dup_ok = True
    for n, ps in HELDOUT_GRID.items():
        for p in ps:
            s = seq_of(p)
            if probes.get(s, cat_of[n]) != cat_of[n]:
                probes_dup_ok = False
            probes[s] = cat_of[n]
    probe_seqs = sorted(probes)                # deterministic order
    probe_cats = [probes[s] for s in probe_seqs]
    heldout_words = {w for ps in HELDOUT_GRID.values() for p in ps
                     for w in p.split()}
    construction_ok = (
        names_ok and probes_dup_ok
        and len(trained_seq_set) == N_TRAINED_UNIQUE
        and len(probe_seqs) == N_HELDOUT_UNIQUE
        and all(sum(s) > 0 for s in probe_seqs)
        and not (set(probe_seqs) & trained_seq_set)
        and heldout_words <= trained_words)    # every word carries evidence

    # the registered null's memoriser floor: trained-table lookups must score
    # ZERO on held-out input by construction
    seq_lookup = {seq_of(p): c for p, c in trained}
    mem_str = float(np.mean([grid_lookup.get(p, -1) == cat_of[n]
                             for n, ps in HELDOUT_GRID.items() for p in ps]))
    mem_seq = float(np.mean([seq_lookup.get(s, -1) == c
                             for s, c in zip(probe_seqs, probe_cats)]))

    probe_strs = [next(p for n, ps in HELDOUT_GRID.items() for p in ps
                       if seq_of(p) == s) for s in probe_seqs]

    # the registered bag-of-words null (reported): TF-IDF retrieval, token-
    # overlap NN, and the NB reference (which doubles as the resolvability
    # rig gate)
    tr_phr = [p for p, _ in trained]
    tr_cat = [c for _, c in trained]
    tfidf_correct = int(sum(p == c for p, c in
                            zip(_tfidf_retrieval(tr_phr, tr_cat, probe_strs),
                                probe_cats)))

    def _overlap_nn(s):
        best, arg = -1, -1
        for q in sorted(seq_lookup):
            ov = len(set(s) & set(q) - {0})
            if ov > best:
                best, arg = ov, seq_lookup[q]
        return arg
    nn_correct = int(sum(_overlap_nn(s) == c
                         for s, c in zip(probe_seqs, probe_cats)))

    tr_seqs = [seq_of(p) for p, _ in trained]
    nb_correct = int(sum(p == c for p, c in
                         zip(_nb_reference(tr_seqs, tr_cat, probe_seqs,
                                           n_cats), probe_cats)))

    # seen-fit probes: every unique trained sequence (unambiguous — verified
    # via construction_ok's N_TRAINED_UNIQUE count)
    seen_seqs = sorted(seq_lookup)
    seen_cats = [seq_lookup[s] for s in seen_seqs]
    seen_strs = [next(p for p, _ in trained if seq_of(p) == s)
                 for s in seen_seqs]

    out = {"gpu": (torch.cuda.get_device_name(0) if device == "cuda"
                   else "cpu"),
           "names_ok": bool(names_ok),
           "construction_ok": bool(construction_ok),
           "n_heldout_unique": len(probe_seqs),
           "n_seen_unique": len(seen_seqs),
           "mem_str_acc": mem_str, "mem_seq_acc": mem_seq,
           "tfidf_retrieval_correct": tfidf_correct,
           "nn_overlap_correct": nn_correct,
           "nb_ref_correct": nb_correct,
           "null_p_claim": _binom_p_geq(CLAIM_MIN, len(probe_seqs), CHANCE),
           "seeds": []}

    perm = [(c + PERM_SHIFT) % n_cats for c in range(n_cats)]

    def train_one(seed_offset, phr, cats, acts, states, permute=False):
        torch.manual_seed(seed_offset)
        cfg = UnifiedBrainConfig(llm_enabled=False,
                                 enable_semantic_anchors=True)
        model = UnifiedBrain(cfg).to(device)
        # The grid's own truth map, installed at the single supervision site
        # (instance attribute shadows the bound method — T2.07's machinery).
        # The shipped lookup cannot label a designed grid: it matches only
        # its own synonym table and DEFAULTS TO WALK on anything else.
        sa = model.semantic_anchors
        if permute:
            sa.get_anchor_for_label = lambda l: perm[grid_lookup[l]]
        else:
            sa.get_anchor_for_label = lambda l: grid_lookup[l]
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

    for seed in seeds:
        rng = np.random.RandomState(int(seed) * 100_003 % 2**32)
        cats_tr = rng.randint(0, N_CLUSTERS, size=n_train)  # never wave
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
from experiments.tests.t2_15_freeform_routing import remote_run
out = remote_run(__SEEDS__)
json.dump(out, open(os.path.join(os.environ["JACK_OUT"], "t215.json"), "w"),
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
                 fetch=["t215.json"])
    if not res.ok:
        raise RuntimeError(f"T2.15 job failed on {res.backend}: {res.message}")
    out = json.loads(Path(res.artifacts["t215.json"]).read_text())
    out["backend"] = res.backend
    return out


def _experiment(seed: int) -> dict:
    if not _CACHE:
        _CACHE.update(_submit(SEEDS))
    rows = _CACHE["seeds"]
    n_hu = _CACHE["n_heldout_unique"]
    return {
        "gpu": _CACHE["gpu"], "backend": _CACHE["backend"],
        "names_ok": _CACHE["names_ok"],
        "construction_ok": _CACHE["construction_ok"],
        "n_heldout_unique": n_hu,
        "n_seen_unique": _CACHE["n_seen_unique"],
        "mem_str_acc": _CACHE["mem_str_acc"],
        "mem_seq_acc": _CACHE["mem_seq_acc"],
        "tfidf_retrieval_correct": _CACHE["tfidf_retrieval_correct"],
        "nn_overlap_correct": _CACHE["nn_overlap_correct"],
        "nb_ref_correct": _CACHE["nb_ref_correct"],
        "null_p_claim": _CACHE["null_p_claim"],
        "heldout_correct": [r["heldout_correct"] for r in rows],
        "heldout_correct_min": min(r["heldout_correct"] for r in rows),
        "paraphrase_routing_accuracy": round(
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
    if not m["names_ok"]:
        return Status.VOID          # the shipped anchors are not the ones
                                    # this grid was designed against
    if not m["construction_ok"] or m["mem_str_acc"] != 0.0 \
            or m["mem_seq_acc"] != 0.0:
        return Status.VOID          # grid construction broken: a probe is
                                    # empty, duplicated, evidence-free, or
                                    # memoriser-reachable — the split, not
                                    # the claim
    if m["nb_ref_correct"] < NB_REF_MIN:
        return Status.VOID          # simplest lexical learner fails: split
                                    # not resolvable — task indicted (T1.02)
    if m["seen_correct_min"] < m["n_seen_unique"] - SEEN_FIT_SLACK:
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
    return run_spec(BY_ID["T2.15"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # Local, CPU, at the PRODUCTION CONFIG (the T2.04 lesson). Only
        # counts shrink: steps and dataset. Exercises the grid construction
        # checks, both twins, the supervision override, both eval passes,
        # every reported null and the loss-fall bookkeeping.
        import time
        t0 = time.time()
        out = remote_run([2], n_train=240, steps=8)
        dt = time.time() - t0
        print(json.dumps(out, indent=1))
        row = out["seeds"][0]
        assert out["names_ok"] and out["construction_ok"], out
        assert out["mem_str_acc"] == 0.0 and out["mem_seq_acc"] == 0.0, out
        assert out["n_heldout_unique"] == N_HELDOUT_UNIQUE, out
        assert out["n_seen_unique"] == N_TRAINED_UNIQUE, out
        assert row["finite"] and row["det_ok"], row
        print(f"SMOKE OK in {dt:.1f}s "
              f"(~{dt / (8 * 2):.2f} s/step upper bound incl. builds+eval)")
    else:
        run()
