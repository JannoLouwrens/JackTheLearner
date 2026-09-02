"""LG.00 — Jack knows what his LLM cannot. He is not a puppet.

GOAL.md cites this id verbatim as *"the proof he is a creature and not a
costume"*: strip the diary and the learned core, and his answers about his own
life must COLLAPSE — while his general knowledge survives untouched. He should
be smarter INSIDE his life and dumber outside it. That asymmetry is the claim;
neither half alone is worth anything.

    ARM  jack   frozen SmolLM2-360M + his ATTRIBUTED DIARY (ME.9's store),
                retrieved from the QUESTION TEXT ALONE and pasted into the
                prompt. One pipeline.
    NULL llm    the identical frozen model, the identical scaffold, the
                identical question — no diary. This is LG.01's leg 2 verbatim,
                and its verdicts are READ FROM LG.01's ARTIFACT by content
                hash, not recomputed.

THE READOUT IS LG.01's, BY IMPORT. Each candidate answer is scored as a
CONTINUATION of the prompt by length-normalised log-probability and the arm's
answer is the argmax. No letters, no listed options, no position for bias to
live in — LG.01's VOID RECORD is why, and importing rather than copying is why
that repair cannot rot out of sync here.

WHAT "FULL JACK" IS TODAY, AND WHAT IT IS NOT — declared, not buried. The
registry's hypothesis says *"full Jack (learned core + diary + LLM)"*. There is
no learned core in this project that answers questions in language; building one
to satisfy this sentence would be the disease this repo exists to prevent. So
the arm run here is **diary + LLM**, and the learned core contributes nothing.
That runs the HARD direction for the claim — Jack is given strictly fewer
resources than the hypothesis allows — so it is recorded as a limitation, the
way LG.01 recorded using the 360M parent rather than the 1.7B. The notes' second
dissociation (*"ablate the LLM -> he still ACTS correctly in his world"*) is a
motor claim, is not scored here, and belongs to a successor spec.

WHY THE COMPARISON IS NOT RIGGED, which is the only interesting question about
a retrieval-augmented arm. Four things carry it, and each can fail:

  1. THE QUESTIONS DO NOT LEAK THEIR ANSWERS. LG.01's stripped control scored
     0.2616 against a ceiling of 0.45 and a chance of 0.25 — the wording of
     "Where did Jack find the pale berries?" does not tell you the answer.
     Jack's advantage therefore cannot be an artifact of question wording.
  2. THE FALSE-RECORD CONTROL. `wrong_life` gives the identical pipeline a
     diary from a DIFFERENT life (`_build_life(seed + 100)`) — same generator,
     same vocabulary, same prompt shape, same amount of context, different
     facts. If a plausible-but-false record helps, then the context BLOCK is
     doing the work and the lived record is not. Gate: it may not beat the
     null by more than WRONG_DIARY_MARGIN.
  3. THE GENERAL-KNOWLEDGE CONTROL, which the registry declares: on capitals,
     arithmetic and vocabulary the null must MATCH OR BEAT full Jack. Jack's
     prompt ALWAYS carries a record block — when retrieval abstains it carries
     `NO_RECORD` — so his prompt is never byte-identical to the null's and this
     control is genuinely able to fail. A control that could only ever tie is
     an identity wearing a control's clothes (LESSONS.md, 2026-08-29).
  4. THE SAME CONTROL IN THE OTHER DIRECTION. "Survives untouched" is not
     "loses gracefully": `general_retention` requires Jack keep >= 70% of the
     null's general-knowledge accuracy. The registry's notes name RT-2's
     measured 11-point general-knowledge loss from task-only finetuning as what
     this clause guards.

THE SELECTION TRAP, and how this spec refuses to walk into it. LG.01 retains a
question only when the null is OUTRIGHT WRONG on it (`CHANCE_BAND_HI = 0.0`).
So on the certified set the null scores **0.000 BY CONSTRUCTION, not by
measurement**, and any sigma computed against it would be infinite and
meaningless. This spec therefore reports two readings and gates them
differently:

  - `advantage_life` on the FULL generated set (102 questions/seed, on which
    the null was never selected; it scores its true prior-driven rate). This is
    the registered metric `grounded_knowledge_advantage`, and the >= 3 sigma
    bar is applied HERE, paired per question (McNemar standard error).
  - `jack_acc_certified` on LG.01's certified subset, gated on an ABSOLUTE
    accuracy bar with NO sigma claimed, and with `null_acc_certified` recorded
    at its constructed 0.0 so a reader cannot mistake selection for evidence.

The certification is not re-run and not re-derived: the retained set is read
off the null's own verdicts through LG.01's exact rule, and a seed whose
categories fall below LG.01's `RETAIN_MIN` returns VOID — LG.01's `kills` field
says any LG.00 run scored on an uncertified probe set is void, and this is that
sentence in code.

PASS  — on every seed: the paired advantage on the full life set clears 3
        sigma, Jack clears JACK_LIFE_MIN on the full set AND on the certified
        subset, the null matches-or-beats him on general knowledge, his general
        knowledge is retained, and a false diary does not help.
FAIL  — LLM-alone matches full Jack on his own life (he is a costume), or Jack
        wins on general knowledge too (the test measures scaffolding), or a
        stranger's diary helps as much as his own (the test measures the
        context block).
VOID  — the verdict artifact does not cover these prompts, or the null is not
        demonstrably alive on general knowledge, or the probe set is not
        certified for a seed.

WORST-SEED GATES ARE READ FROM THE RECORDED ROW. `protocol._aggregate` hands
`_check` the mean and population std across seeds, not the per-seed values, so
"on every seed" has to come from somewhere else. Every gated metric is
therefore recorded per seed as an explicit `<key>_s<seed>` key (each seed's
run returns the full per-seed set, identical across runs, so run_spec's
mean/std aggregation carries the values into the row verbatim), and `_check`
is a pure function of (m, c) — no module state. The attempt-3 version read
module-level `_MEMO` instead, so its PASS could not be replayed from its row;
T0.13 attempt 22 flagged the gate keyless AND stale (2026-09-02 16:15), and
this is that repair — the sibling of LG.02's, one commit later. Bars are
byte-identical to the pre-registered ones; only recording and the gate's
input source changed. A row missing a per-seed key the gates need returns
VOID rather than falling back to anything.

VOID RECORD — attempt 1, 2026-08-30 18:47 UTC, and it was MY ESTIMATOR, not the
data. Attempt 1 gated the worst seed by the moment bound `mean -
std*sqrt(n-1)`, which is a valid lower bound on a minimum, and the probe-set
certification gate read `23.0 - 2.160*sqrt(2) = 19.94` against LG.01's
`RETAIN_MIN` of 20 and returned VOID. The per-seed values are **26, 22, 21** —
every seed's probe set is certified, with a true worst-seed margin of one
question. Nothing else in the run was touched by the change: the bound and the
exact value agree on the verdict for all six other gates.

**This repair makes gates LOOSER, which is the direction to distrust** — a
`>=` gate on an exact minimum is easier than the same gate on a lower bound
(LESSONS.md: *"a rig repair that makes your claim EASIER is the one to
distrust"*). Three things are offered against that, and a reader should hold
them to it: no threshold moved (`RETAIN_MIN` is still LG.01's 20, `SIGMA_MIN`
still the registry's 3.0, every other constant byte-identical); the bound was
described as an interface workaround in this docstring BEFORE any number
existed, so the intent being realised is the pre-registered one; and the exact
per-seed table above is published here so the repair can be checked rather than
believed. The VOID row and its `refs/jack/failimpl/LG.00/2026-08-30T18-47-59`
implementation stay in the ledger's history either way.

THE LLM PASS IS OFFLINE AND run() NEVER LOADS A MODEL — LG.01's rule, same
reasons (T0.07: SmolLM2 in-process is a 6.9 GB mistake; Budget.CPU is a
10-minute kill). Precompute with

    python -m experiments.tests.lg_00_not_a_puppet --llm-pass

through `scripts/launch_detached.sh`. Keys are
`sha256(model@revision + scaffold sha + exact prompt + candidate answer)`, so
changing the questions, the retrieval, the context format or the weights makes
every key miss and run() returns VOID rather than scoring against stale
verdicts. The null's LIFE verdicts are looked up in LG.01's artifact first —
identical prompts, identical keys — so the null is not paid for twice and,
more importantly, is provably the same null LG.01 certified against.

COVERS: language (parent) (claim).
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from .lg_01_lived_necessary_probes import (
    CALIBRATION, CATEGORIES, CHANCE_BAND_HI, LLM_ARTIFACT as LG01_ARTIFACT,
    MODEL_ID, N_OPTIONS, RETAIN_MIN, SCAFFOLD_SHA, _build_life, _key, _prompt,
)

REPO = Path(__file__).resolve().parents[2]

# The claim is about the diary Jack reads and the probe set LG.01 certified;
# both hash into this certificate. A change to either invalidates it.
IMPL_DEPS = ["EpisodicMemory.py",
             "experiments/tests/lg_01_lived_necessary_probes.py"]

SEEDS = [0, 1, 2]
ARTIFACT = "/data/lg00_llm_verdicts.json"

# ── retrieval: ME.9's store, cued by the QUESTION TEXT ALONE ────────────────
# LG.01's oracle was handed `channel`, `speaker` and `cue` as metadata, which is
# legitimate for certifying that the record CAN answer a question. It would not
# be legitimate here: an arm told which channel to look in has been given part
# of the answer. So this arm calls `recall(question)` with no provenance filter
# and no cue, and it inherits `EpisodicMemory`'s abstention floor — a question
# whose content words are not in the record returns nothing, which is what makes
# the general-knowledge control able to fail.
TOP_K = 3
NO_RECORD = "(nothing in my record matches)"
# A fixed, far-future clock so recency is 0.0 for every event alike and the
# retrieval is deterministic. Without it `recall` reads the wall clock and the
# ranking is a property of when the test ran.
NOW = 1e9

# ── gates, pre-registered 2026-08-30 BEFORE any LG.00 number existed ─────────
# SIGMA_MIN is the registry's own number, verbatim.
SIGMA_MIN = 3.0
# An absolute floor on Jack, on the full set and on the certified subset. The
# null runs near 0.30 on the full set and chance is 0.25; a creature reading his
# own record should be nowhere near either. Not fitted to anything — no LG.00
# measurement exists at the time this line is written.
JACK_LIFE_MIN = 0.60
# The registry's control, "LLM-alone must MATCH OR BEAT full Jack", made
# numeric. 0.05 is 1.5 questions of 30: one question of thirty is noise, not a
# grounding advantage. Declared here rather than chosen later.
CONTROL_ADV_MAX = 0.05
# "his general knowledge survives untouched" (GOAL.md) as a floor rather than a
# benchmark. RT-2 lost 11 points to task-only finetuning; 70% retention is
# looser than that and deliberately so — this catches destruction, not drift.
GENERAL_RETENTION_MIN = 0.70
# A stranger's diary may not buy more than this over the bare null.
WRONG_DIARY_MARGIN = 0.10
# The null's liveness, LG.01's CALIB_MIN on the same model and scaffold. Below
# this the null cannot answer "what is the capital of France?" and its silence
# on Jack's life proves nothing about Jack.
NULL_LIVE_MIN = 0.50

# ── general knowledge, in the identical scaffold ─────────────────────────────
# The first twelve ARE LG.01's calibration set, verbatim by import, so the
# null's verdicts on them are already in LG.01's artifact and so the liveness
# floor here is literally the one that caught LG.01's dead readout. Eighteen
# more of the same deliberately-easy kind, because a 12-question control is too
# coarse to say "matches or beats" about.
GENERAL = list(CALIBRATION) + [
    ("What is the capital of Japan?", "Tokyo", ["Lisbon", "Oslo", "Cairo"]),
    ("How many legs does a spider have?", "8", ["4", "6", "10"]),
    ("What is 20 minus 8?", "12", ["10", "14", "18"]),
    ("Which planet do we live on?", "Earth", ["Mars", "Venus", "Saturn"]),
    ("What is water made of?", "hydrogen and oxygen",
     ["iron and salt", "sand and lime", "copper and tin"]),
    ("Which instrument has black and white keys?", "a piano",
     ["a drum", "a flute", "a violin"]),
    ("What is the opposite of empty?", "full", ["quiet", "narrow", "slow"]),
    ("How many months are in a year?", "12", ["7", "10", "24"]),
    ("What do cows drink when they are calves?", "milk",
     ["petrol", "vinegar", "ink"]),
    ("Which is the hottest: ice, water or fire?", "fire", ["ice", "water",
     "snow"]),
    ("What language is mainly spoken in Brazil?", "Portuguese",
     ["Swedish", "Korean", "Greek"]),
    ("What is 9 divided by 3?", "3", ["2", "4", "6"]),
    ("Which of these is a metal?", "iron", ["cotton", "leather", "paper"]),
    ("What do plants need from the sun?", "light", ["salt", "iron", "glass"]),
    ("How many sides does a triangle have?", "3", ["2", "4", "5"]),
    ("What is the frozen form of water?", "ice", ["steam", "smoke", "sand"]),
    ("Which one flies?", "a bird", ["a whale", "a worm", "a crab"]),
    ("What is the capital of Italy?", "Rome", ["Athens", "Vienna", "Dublin"]),
]

ARMS = ("null_life", "jack_life", "wrong_life", "null_gen", "jack_gen")


# ─────────────────────────────────────────────────────────────────────────────
# The arm: his record, pasted into his own prompt
# ─────────────────────────────────────────────────────────────────────────────
def _context_lines(mem, question: str) -> list:
    hits = mem.recall(question, top_k=TOP_K, now=NOW)
    if not hits:
        return [NO_RECORD]
    return [h.event.text for h in hits]


def _augment(question: str, lines: list) -> str:
    body = "\n".join(f"- {line}" for line in lines)
    return f"FROM MY OWN RECORD:\n{body}\n\n{question}"


def _pairs(seed: int):
    """Every (arm, question index, prompt, candidate, is_correct) this spec
    scores, for one seed. One list, so the offline pass and run() can never
    disagree about what was asked.
    """
    tmp = Path(tempfile.mkdtemp())
    mem, probes = _build_life(seed, tmp / "life.jsonl")
    # A DIFFERENT life: same generator, same vocabulary, independent draws.
    # +100 keeps it disjoint from every scored seed.
    wrong_mem, _ = _build_life(seed + 100, tmp / "wrong.jsonl")

    out, hits_life = [], 0
    for i, p in enumerate(probes):
        q = p["question"]
        lines = _context_lines(mem, q)
        hits_life += int(lines != [NO_RECORD])
        p_jack = _prompt(_augment(q, lines))
        p_wrong = _prompt(_augment(q, _context_lines(wrong_mem, q)))
        p_null = _prompt(q)                 # LG.01's null prompt, byte-identical
        for opt in p["options"]:
            gold = (opt == p["correct"])
            out.append(("null_life", i, p_null, opt, gold))
            out.append(("jack_life", i, p_jack, opt, gold))
            out.append(("wrong_life", i, p_wrong, opt, gold))

    hits_gen = 0
    for i, (q, correct, distractors) in enumerate(GENERAL):
        lines = _context_lines(mem, q)
        hits_gen += int(lines != [NO_RECORD])
        p_jack = _prompt(_augment(q, lines))
        p_null = _prompt(q)
        for opt in [correct] + list(distractors):
            gold = (opt == correct)
            out.append(("null_gen", i, p_null, opt, gold))
            out.append(("jack_gen", i, p_jack, opt, gold))

    return probes, out, hits_life / len(probes), hits_gen / len(GENERAL)


# ─────────────────────────────────────────────────────────────────────────────
# The offline pass (never called by run())
# ─────────────────────────────────────────────────────────────────────────────
def llm_pass(seeds=None, out_path: str = ARTIFACT) -> dict:
    """Score every prompt with the frozen parent and cache it by content hash.

    Detached and short-lived by design: fp32 SmolLM2-360M peaks near 2.5 GB on
    this box, which is why it must never live inside run(). The scoring is
    LG.01's, line for line, because the null must be the SAME null.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    os.environ.setdefault("HF_HOME", "/data/caches/huggingface")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "right"              # options scored at their own offsets
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, low_cpu_mem_usage=True).eval()
    revision = getattr(model.config, "_commit_hash", None) or "local"

    store = {}
    if Path(out_path).exists():
        store = json.loads(Path(out_path).read_text())
    base = json.loads(Path(LG01_ARTIFACT).read_text()) \
        if Path(LG01_ARTIFACT).exists() else {}
    lg01_model = base.get("_meta", {}).get("model")
    if lg01_model and lg01_model != f"{MODEL_ID}@{revision}":
        # Loud, and NOT worked around: reusing LG.01's verdicts under a
        # different revision would score this claim against a different null.
        print(f"[lg00] WARNING: LG.01 artifact is {lg01_model}, this pass is "
              f"{MODEL_ID}@{revision}; its verdicts will NOT be reused and "
              f"run() will VOID unless they are recomputed here.", flush=True)
    store.setdefault("_meta", {})["model"] = f"{MODEL_ID}@{revision}"
    store["_meta"]["scaffold_sha"] = SCAFFOLD_SHA

    todo = {}
    for seed in (seeds or SEEDS):
        _probes, pairs, _hl, _hg = _pairs(seed)
        for _arm, _i, prompt, option, _gold in pairs:
            k = _key(revision, prompt, option)
            if k not in store and k not in base:
                todo[k] = (prompt, option)
    todo = list(todo.items())
    print(f"[lg00] {len(todo)} (prompt, answer) pairs to score "
          f"({len(base) - 1 if base else 0} reusable from LG.01)", flush=True)

    B = 8
    with torch.no_grad():
        for s in range(0, len(todo), B):
            chunk = todo[s:s + B]
            texts = [p + o for _, (p, o) in chunk]
            n_pre = [len(tok(p).input_ids) for _, (p, _o) in chunk]
            enc = tok(texts, return_tensors="pt", padding=True)
            logp = torch.log_softmax(model(**enc).logits, dim=-1)
            for j, (k, _) in enumerate(chunk):
                ids = enc.input_ids[j]
                n = int(enc.attention_mask[j].sum())
                # Token t is predicted by position t-1. Score the OPTION's
                # tokens only, length-normalised.
                span = range(n_pre[j], n)
                tot = sum(float(logp[j, t - 1, ids[t]]) for t in span)
                store[k] = round(tot / max(1, len(span)), 5)
            if s % (B * 20) == 0:
                print(f"[lg00] {s + len(chunk)}/{len(todo)}", flush=True)
    Path(out_path).write_text(json.dumps(store))
    print(f"[lg00] wrote {out_path} ({len(store) - 1} verdicts)", flush=True)
    return store


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────
def _paired(a: list, b: list):
    """Advantage of `a` over `b` and its McNemar standard error.

    Paired, because both arms answer the SAME questions: only the questions
    they disagree on carry information, and an unpaired binomial se would
    overstate the noise by counting the agreements twice.
    """
    n = len(a)
    b_only = sum(1 for x, y in zip(a, b) if x and not y)
    c_only = sum(1 for x, y in zip(a, b) if y and not x)
    adv = (b_only - c_only) / n
    se = math.sqrt(b_only + c_only) / n
    return adv, se


_MEMO: dict = {}

# The keys each arm returns. _measure computes both sides in one pass; these
# tuples are what partitions its output between experiment and control.
_EXP_KEYS = ("n_life", "n_general", "n_certified", "retained_min_per_category",
             "jack_acc_life", "null_acc_life", "grounded_knowledge_advantage",
             "sigma_life", "jack_acc_certified", "null_acc_certified",
             "retrieval_hit_rate_life", "retrieval_hit_rate_general",
             "verdicts_missing")
_CTL_KEYS = ("null_acc_general", "jack_acc_general", "advantage_general",
             "general_retention", "wrong_acc_life", "wrong_margin",
             "verdicts_missing")


def _flat(seed: int, keys: tuple) -> dict:
    """The seed's own metrics under their plain names (aggregated to mean/std
    across runs, as before) PLUS every seed's value as an explicit
    `<key>_s<seed>` key. The per-seed keys are identical in every run, so
    run_spec's aggregation records them verbatim — that is what makes the
    worst-seed gates in `_check` answerable from the row alone."""
    out = {k: _MEMO[seed][k] for k in keys}
    for s in SEEDS:
        for k in keys:
            out[f"{k}_s{s}"] = _MEMO[s][k]
    return out


def _measure(seed: int) -> dict:
    if seed in _MEMO:
        return _MEMO[seed]
    probes, pairs, hit_life, hit_gen = _pairs(seed)

    store = json.loads(Path(ARTIFACT).read_text()) \
        if Path(ARTIFACT).exists() else {}
    base = json.loads(Path(LG01_ARTIFACT).read_text()) \
        if Path(LG01_ARTIFACT).exists() else {}
    meta = store.get("_meta") or base.get("_meta") or {}
    revision = meta.get("model", "@local").split("@")[-1]

    ranked: dict = {}
    missing = 0
    for arm, i, prompt, option, gold in pairs:
        k = _key(revision, prompt, option)
        s = store.get(k, base.get(k))
        if s is None:
            missing += 1
            continue
        ranked.setdefault((arm, i), []).append((s, gold))

    got = {arm: {} for arm in ARMS}
    for (arm, i), scored in ranked.items():
        if len(scored) != N_OPTIONS:
            missing += 1
            continue
        got[arm][i] = bool(max(scored)[1])       # the arm's argmax was correct?

    n_life, n_gen = len(probes), len(GENERAL)
    life_idx = [i for i in range(n_life)
                if all(i in got[a] for a in ("null_life", "jack_life",
                                             "wrong_life"))]
    gen_idx = [i for i in range(n_gen)
               if all(i in got[a] for a in ("null_gen", "jack_gen"))]
    if len(life_idx) != n_life or len(gen_idx) != n_gen:
        missing += (n_life - len(life_idx)) + (n_gen - len(gen_idx))

    null_l = [got["null_life"].get(i, False) for i in life_idx]
    jack_l = [got["jack_life"].get(i, False) for i in life_idx]
    wrong_l = [got["wrong_life"].get(i, False) for i in life_idx]
    null_g = [got["null_gen"].get(i, False) for i in gen_idx]
    jack_g = [got["jack_gen"].get(i, False) for i in gen_idx]

    def _acc(v):
        return sum(v) / len(v) if v else 0.0

    adv_life, se_life = _paired(jack_l, null_l)
    sigma_life = adv_life / se_life if se_life > 0 else 0.0
    adv_gen, _se_gen = _paired(jack_g, null_g)
    adv_wrong, _se_w = _paired(wrong_l, null_l)

    # LG.01's certification, READ from the null's own verdicts through LG.01's
    # exact retention rule — not re-derived, not re-run.
    retained = [j for j, i in enumerate(life_idx)
                if float(null_l[j]) <= CHANCE_BAND_HI]
    per_cat = {c: 0 for c in CATEGORIES}
    for j in retained:
        per_cat[probes[life_idx[j]]["category"]] += 1
    jack_cert = ([jack_l[j] for j in retained])
    null_cert = ([null_l[j] for j in retained])

    out = {
        "n_life": n_life,
        "n_general": n_gen,
        "n_certified": len(retained),
        "retained_min_per_category": min(per_cat.values()) if retained else 0,
        # ── the claim, on the UNSELECTED full set ──
        "jack_acc_life": round(_acc(jack_l), 4),
        "null_acc_life": round(_acc(null_l), 4),
        "grounded_knowledge_advantage": round(adv_life, 4),
        "sigma_life": round(sigma_life, 3),
        # ── the same reading on LG.01's certified subset; null is 0 by
        #    CONSTRUCTION (retained <=> the null was wrong), so no sigma ──
        "jack_acc_certified": round(_acc(jack_cert), 4),
        "null_acc_certified": round(_acc(null_cert), 4),
        # ── controls ──
        "null_acc_general": round(_acc(null_g), 4),
        "jack_acc_general": round(_acc(jack_g), 4),
        "advantage_general": round(adv_gen, 4),
        "general_retention": round(_acc(jack_g) / _acc(null_g), 4)
        if _acc(null_g) > 0 else 0.0,
        "wrong_acc_life": round(_acc(wrong_l), 4),
        "wrong_margin": round(adv_wrong, 4),
        # ── rig ──
        "retrieval_hit_rate_life": round(hit_life, 4),
        "retrieval_hit_rate_general": round(hit_gen, 4),
        "verdicts_missing": missing,
    }
    _MEMO[seed] = out
    return out


def _experiment(seed: int) -> dict:
    # All seeds are computed (memoized — total work is unchanged) so every
    # run can return the full per-seed key set; see _flat.
    for s in SEEDS:
        _measure(s)
    return _flat(seed, _EXP_KEYS)


def _control(seed: int) -> dict:
    """The two declared controls, both of which must fail to help Jack.

    GENERAL KNOWLEDGE (the registry's): the null must match or beat him, and he
    must not have LOST his general knowledge either.
    A STRANGER'S DIARY: same pipeline, same context shape, another life's facts.
    """
    for s in SEEDS:
        _measure(s)
    return _flat(seed, _CTL_KEYS)


# Every per-seed key the gates consult; a row missing one cannot answer them.
_NEED_M = tuple(f"{k}_s{s}" for k in (
    "verdicts_missing", "retained_min_per_category", "sigma_life",
    "jack_acc_life", "jack_acc_certified") for s in SEEDS)
_NEED_C = tuple(f"{k}_s{s}" for k in (
    "verdicts_missing", "null_acc_general", "advantage_general",
    "general_retention", "wrong_margin") for s in SEEDS)


def _check(m: dict, c: dict):
    """Pure function of the recorded row — no module state, every key a
    static m[...]/c[...] read, all read up front so each is consulted on
    every replay regardless of which gate fires."""
    if any(k not in m for k in _NEED_M) or any(k not in c for k in _NEED_C):
        return Status.VOID          # the record cannot answer the gates
    vm_m = (m["verdicts_missing_s0"], m["verdicts_missing_s1"],
            m["verdicts_missing_s2"])
    vm_c = (c["verdicts_missing_s0"], c["verdicts_missing_s1"],
            c["verdicts_missing_s2"])
    rmc = (m["retained_min_per_category_s0"],
           m["retained_min_per_category_s1"],
           m["retained_min_per_category_s2"])
    sig = (m["sigma_life_s0"], m["sigma_life_s1"], m["sigma_life_s2"])
    jal = (m["jack_acc_life_s0"], m["jack_acc_life_s1"],
           m["jack_acc_life_s2"])
    jac = (m["jack_acc_certified_s0"], m["jack_acc_certified_s1"],
           m["jack_acc_certified_s2"])
    nag = (c["null_acc_general_s0"], c["null_acc_general_s1"],
           c["null_acc_general_s2"])
    adv = (c["advantage_general_s0"], c["advantage_general_s1"],
           c["advantage_general_s2"])
    ret = (c["general_retention_s0"], c["general_retention_s1"],
           c["general_retention_s2"])
    wrm = (c["wrong_margin_s0"], c["wrong_margin_s1"], c["wrong_margin_s2"])
    # ── rig gates: VOID, not FAIL — a run that could not ask the question ──
    if max(vm_m) > 0 or max(vm_c) > 0:
        return Status.VOID       # the artifact does not cover these prompts
    if min(nag) < NULL_LIVE_MIN:
        return Status.VOID       # the null is not demonstrably alive
    if min(rmc) < RETAIN_MIN:
        return Status.VOID       # LG.01's kills: not a certified probe set
    # ── the claim, and both controls, on EVERY seed ──
    return bool(
        min(sig) >= SIGMA_MIN
        and min(jal) >= JACK_LIFE_MIN
        and min(jac) >= JACK_LIFE_MIN
        and max(adv) <= CONTROL_ADV_MAX
        and min(ret) >= GENERAL_RETENTION_MIN
        and max(wrm) <= WRONG_DIARY_MARGIN)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["LG.00"], _experiment, _check,
                    control_fn=_control, ledger=ledger)


if __name__ == "__main__":
    if "--llm-pass" in sys.argv:
        llm_pass()
    else:
        sys.path.insert(0, str(REPO))
        print(json.dumps(_measure(0), indent=2))
