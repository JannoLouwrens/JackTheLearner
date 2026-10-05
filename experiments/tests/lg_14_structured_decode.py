"""LG.14 — Structured decode: his meaning constrains the mouth, not a margin
over its free samples.

THE RULED SUCCESSOR to LG.12's FAIL (`lg12-abstention-knob-has-no-resolution`,
DISPOSITIONED 2026-09-29): arm (b) of the three-way mouth fork. LG.12 measured
that no dominance margin over free samples can buy fidelity — dom spanned
[1.383, 1.826] (sd 0.080) against a bar the pool arithmetic put at 3.74 nats,
so the selector had nothing to select BETWEEN. Structured decode removes the
need for a varying selection quantity by CONSTRAINING the emission instead of
selecting among free samples. LG.10's bars and its FAIL stand untouched; LG.12
stays the provably weaker sibling. Arm (a), a bigger frozen mouth, is REFUSED
(the pool was measured not to be the binding term); arm (c), de-verbatiming
the scaffold, is SEQUENCED BEHIND this spec on price and fires only if this
FAILs.

THE MECHANISM, and what "the intent's structure" buys. The pipeline is
LG.10's, imported rather than copied: the same life, the same trials, the same
verification gate, the same two frozen models, the same length-normalised
scoring, the same seeded softmax draws at the same TEMP. What changes is the
EMISSION SET and (on the held-out conjunct) the CHANNEL the intent travels by:

    emission set   realizations of the ASSERTION STRUCTURE — a frame that
                   asserts one (thing, place) find from his gate-passed state,
                   in any of its wordings. The PHATIC lines cannot realize an
                   assertion frame, so the structure excludes them BY
                   CONSTRUCTION — that is the constraint doing work, not a
                   tuned margin: 26 of LG.10's 55 wrong draws were phatic
                   collapse, and this design makes that failure unreachable
                   while leaving the 29 drift-to-another-memory draws fully
                   reachable. Concretely: LG.10's candidate pool with the
                   meaning-None lines removed. A fabricated fact still enters
                   iff the verification gate leaks (leak_draws counts it).
    what is drawn  a seeded softmax draw over the constrained set, scored by
                   the frozen model under the prompt — the mouth still chooses
                   HOW (which wording) and can still be WRONG about WHAT
                   (distractor fills stay in the set), so match_on_spoken is a
                   real quantity, exactly as the registry's falsified_by
                   requires ("match_on_spoken under 0.90 on any seed/model
                   with the rig alive").

TWO PROMPT FORMS, because the registration's first mandatory conjunct is about
the channel:

    in-scaffold    LG.10's ARM_ASK, unchanged — the canonical intent sentence
                   is verbatim in the prompt (the copying channel LG.12
                   diagnosed). Every (prompt, candidate) verdict here is
                   already in /data/lg10_llm_verdicts.json: ZERO new keys.
    held-out       STRUCT_ASK — the intent reaches the mouth as STRUCTURE
                   (slot values: `thing = {res}; place = {place}`), and the
                   canonical sentence "I found {res} at {place}." appears
                   NOWHERE in the conditioning text. This is the conjunct's
                   reading, disclosed rather than buried: "held-out" is a
                   property of the PROMPT-INTENT pairing — the canonical
                   sentence of the intent (indeed of ANY fact in his life) is
                   absent from everything the decoder is conditioned on, which
                   is enforced by a rig gate (held_out_prompt_clean) that
                   VOIDs the run if any canonical sentence leaks into a
                   held-out or mismatched prompt. The slot VALUES are of
                   course present — the structure carrying his meaning to the
                   mouth is the mechanism under test, not a leak. What the
                   conjunct forbids is the SENTENCE: a decoder that only
                   works when the answer-as-prose is in its prompt has
                   measured the prompt. These trials buy NEW verdict keys
                   (new keys, not a re-key — LG.10/LG.12/LG.13 certificates
                   are untouched; the count is printed by --llm-pass and
                   disclosed in the run's commit).

THE TWO NULLS, both binding (registry, pre-registered 2026-10-01):

    (1) free generation, carried UNMOVED — LG.10's NULL_ASK over LG.10's FULL
        pool (phatics included: the null is the unconstrained mouth), drawn
        with LG.10's exact RNG keys ("lg10null:{seed}:{ti}:{model}:{d}") over
        the cached verdicts, so its numbers are LG.10's row's numbers, not a
        reimplementation. null_match <= NULL_MATCH_MAX (0.35) on both models.
    (2) THE MISMATCHED-CONSTRAINT NULL, the control that MUST fail — the SAME
        decoder (same fills, same draw machinery, same STRUCT_ASK form) with
        its structure drawn from a DIFFERENT trial's intent (report trial
        (i+1) mod N_REPORT, fixed before any number existed). Its
        match-against-THIS-trial's-intent must stay under MATCH_MIN. If it
        ALSO clears MATCH_MIN, the constraint is not carrying meaning — the
        decoder found the right answer through some other channel — and the
        verdict is VOID, not FAIL (law 2; the LG.12 precedent: kills retires
        a proposed mechanism, never a shipped module).

GATES — every bar quoted from its source and byte-unmoved: MATCH_MIN 0.90,
UNANIMITY_MIN 0.90, SWAP_AGREE_MIN 0.90, NULL_MATCH_MAX 0.35, VARIETY_MIN
0.30, LIVENESS_MIN 0.80 (all imported from lg_10), UTTER_MIN 0.50 (imported
from lg_12).

    PASS  — on every seed and BOTH models:
              match_on_spoken                      >= 0.90   (in-scaffold)
              unanimity_on_spoken                  >= 0.90   (in-scaffold)
              swap_agree                           >= 0.90   (in-scaffold)
              match_on_spoken_on_held_out_intents  >= 0.90   CONJUNCT 1
              unanimity_on_held_out                >= 0.90   CONJUNCT 1
              utter_rate                           >= 0.50
              speak_silence == 0.0, leak_draws == 0
              null_match (free generation)         <= 0.35
    FAIL  — any of the above on the wrong side. In particular: the held-out
            conjunct failing while in-scaffold clears is the "decoder measured
            its prompt" branch, a FAIL of this claim, not a smaller pass; and
            utter_rate under 0.50 on any seed is the ME.3 starvation failure,
            read as a determinate FAIL before the remaining rig gates (the
            LG.12 ordering precedent — a declared falsified_by branch may not
            be converted into a non-verdict by a later VOID).
    VOID  — verdicts missing from the artifact; a canonical sentence found in
            a held-out/mismatched prompt (held_out_prompt_clean < 1.0); a
            scorer failing LIVENESS under either prompt form; wording variety
            on spoken trials under VARIETY_MIN on either form (invariance
            never tested); or the mismatched-constraint null clearing
            MATCH_MIN (the constraint carries no meaning).

ABSTENTION: this design has NO abstention path (the registry's control field
says so) — a report trial is unspoken only when the constrained set is EMPTY,
i.e. the verification gate passed nothing, so the utterance floor carries the
aliveness burden the demoted silence control used to carry. Silence trials
(nothing fresh) still bind through speak_silence == 0.0, exactly as in LG.10.

REPORTED, NEVER GATED: swap_agree_held_out (conjunct 1 names match and
unanimity only), gate_rejected_fab_frac, and the per-form match delta is
readable off the row (in-scaffold vs held-out is the falsifier's diagnostic).

THE LLM PASS IS OFFLINE AND run() NEVER LOADS A MODEL — LG.01's rule (T0.07:
in-process SmolLM2 is a 6.9 GB mistake; Budget.CPU is a 10-minute kill).
Precompute with

    python -m experiments.tests.lg_14_structured_decode --llm-pass

through `scripts/launch_detached.sh`. New keys land in LG.10's store
(/data/lg10_llm_verdicts.json) under content hashes — additive, no existing
key is touched. Draws happen inside run(), seeded from (lg14*, seed, trial,
model, draw) strings, so they are deterministic given the artifact and free
given the cache.

UNSATURATED-NULL: declared in the registry block — match_on_spoken is bounded
at 1.0; the free-generation null measured 0.044/0.083 on this exact rig
(LG.12 attempt 1), ~0.82 under the 0.90 bar. The gate is registerable.

COVERS: language (parent) (claim).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from .lg_01_lived_necessary_probes import SCAFFOLD_SHA, _prompt
from .lg_10_jack_chooses_what_to_say import (
    ARM_ASK, ARTIFACT, LIVENESS_MIN, MATCH_MIN, MODELS, MODEL_A, MODEL_B,
    N_FAB, N_REPORT, NULL_ASK, NULL_MATCH_MAX, S_DRAWS, SWAP_AGREE_MIN,
    UNANIMITY_MIN, VARIETY_MIN, _build_trials, _canonical, _core_intent,
    _draw, _key, _modal, _pool, _scramble,
)
from .lg_12_abstaining_mouth import UTTER_MIN

REPO = Path(__file__).resolve().parents[2]

# The diary his intent is drawn from, the scaffold and pipeline the candidates
# are scored under, and the floor this spec quotes — all decide what this row
# means.
IMPL_DEPS = ["EpisodicMemory.py",
             "experiments/tests/lg_01_lived_necessary_probes.py",
             "experiments/tests/lg_10_jack_chooses_what_to_say.py",
             "experiments/tests/lg_12_abstaining_mouth.py"]

SEEDS = [0, 1, 2]

# ── the held-out prompt form: the intent as STRUCTURE, never as a sentence ───
# The canonical form "I found {res} at {place}." must not be reconstructible
# as a substring of this template under any fill; held_out_prompt_clean
# re-checks that at measure time against every canonical sentence of the
# seed's whole life, so the guarantee is measured, not asserted.
STRUCT_ASK = ("You are Jack. Say one thing from your record, in your own "
              "words. Your chosen meaning, as structure: "
              "thing = {res}; place = {place}.")


def _fills(mem, trial) -> list:
    """The constrained emission set: LG.10's gate-filtered pool restricted to
    candidates that realize the assertion structure — i.e. the meaning-None
    (phatic) lines are excluded BY the structure. A leaked fabricated fact
    (meaning ("FAB", ...)) realizes the structure and stays in, so a broken
    verification gate is still measured downstream by leak_draws."""
    return [(u, m) for (u, m) in _pool(mem, trial) if m is not None]


def _struct_prompt(intent) -> str:
    return _prompt(STRUCT_ASK.format(res=intent[0], place=intent[1]))


# ─────────────────────────────────────────────────────────────────────────────
# Keys and prompt enumeration (shared by the offline pass and run())
# ─────────────────────────────────────────────────────────────────────────────
def _trial_prompts(seed: int) -> list:
    """Per report trial: (ti, trial, fills, pool, arm_p, null_p, ho_p, mm_p,
    scram). ONE enumeration, so the offline pass and run() can never disagree.
    Report trials are trials[0..N_REPORT-1] by _build_trials' construction;
    the mismatched constraint for trial i is trial (i+1) mod N_REPORT's
    intent, fixed here before any number existed."""
    mem, trials = _build_trials(seed)
    report = [t for t in trials if t["kind"] == "report"]
    out = []
    for ti, trial in enumerate(trials):
        if trial["kind"] != "report":
            continue
        pool = _pool(mem, trial)
        fills = [(u, m) for (u, m) in pool if m is not None]
        mm_intent = report[(ti + 1) % N_REPORT]["intent"]
        arm_p = _prompt(ARM_ASK.format(intent=_canonical(*trial["intent"])))
        null_p = _prompt(NULL_ASK)
        ho_p = _struct_prompt(trial["intent"])
        mm_p = _struct_prompt(mm_intent)
        rng_s = random.Random(f"lg10scramble:{seed}:{ti}")
        scram = _scramble(_canonical(*trial["intent"]), rng_s)
        out.append((ti, trial, fills, pool, arm_p, null_p, ho_p, mm_p, scram))
    return out, trials


def _prompts_for(seed: int) -> list:
    """Every (prompt, candidate) pair each model must score for one seed. The
    arm/null/scramble-under-arm pairs are LG.10's own keys and will already
    be present in the store (the pass skips present keys); the struct-prompt
    pairs are this spec's NEW keys."""
    ctxs, _trials = _trial_prompts(seed)
    pairs = []
    for ti, trial, fills, pool, arm_p, null_p, ho_p, mm_p, scram in ctxs:
        for u, _m in fills:
            pairs.append((arm_p, u))
            pairs.append((ho_p, u))
            pairs.append((mm_p, u))
        for u, _m in pool:
            pairs.append((null_p, u))
        canon = _canonical(*trial["intent"])
        pairs.append((arm_p, scram))          # cached (LG.10's liveness probe)
        pairs.append((ho_p, scram))           # NEW: liveness under structure
        pairs.append((ho_p, canon))           # in fills already; harmless
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# The offline pass (never called by run()) — LG.10's scoring, line for line
# ─────────────────────────────────────────────────────────────────────────────
def llm_pass(seeds=None, out_path: str = ARTIFACT) -> dict:
    import gc
    import os
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    os.environ.setdefault("HF_HOME", "/data/caches/huggingface")
    store = {}
    if Path(out_path).exists():
        store = json.loads(Path(out_path).read_text())
    store.setdefault("_meta", {}).setdefault("models", {})
    store["_meta"]["scaffold_sha"] = SCAFFOLD_SHA

    all_pairs = []
    for seed in (seeds or SEEDS):
        all_pairs.extend(_prompts_for(seed))

    new_total = 0
    for model_id in MODELS:
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.padding_side = "right"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32, low_cpu_mem_usage=True).eval()
        revision = getattr(model.config, "_commit_hash", None) or "local"
        store["_meta"]["models"][model_id] = revision

        todo = {}
        for prompt, option in all_pairs:
            k = _key(model_id, revision, prompt, option)
            if k not in store:
                todo[k] = (prompt, option)
        todo = list(todo.items())
        new_total += len(todo)
        print(f"[lg14] {model_id}: {len(todo)} NEW pairs to score", flush=True)

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
                    span = range(n_pre[j], n)
                    tot = sum(float(logp[j, t - 1, ids[t]]) for t in span)
                    store[k] = round(tot / max(1, len(span)), 5)
                if s % (B * 20) == 0:
                    print(f"[lg14] {model_id} {s + len(chunk)}/{len(todo)}",
                          flush=True)
        del model, tok
        gc.collect()
        Path(out_path).write_text(json.dumps(store))
    print(f"[lg14] wrote {out_path} ({len(store) - 1} verdicts, "
          f"{new_total} new keys this pass)", flush=True)
    return store


# ─────────────────────────────────────────────────────────────────────────────
# Scoring: deterministic seeded draws over cached verdicts
# ─────────────────────────────────────────────────────────────────────────────
_MEMO: dict = {}


def _score(store, revs, model_id, prompt, cands):
    """[(score, u, m)] for every candidate the store covers; the count of
    absent keys rides back so the VOID gate sees every hole."""
    rev = revs.get(model_id, "local")
    scored, missing = [], 0
    for u, m in cands:
        s = store.get(_key(model_id, rev, prompt, u))
        if s is None:
            missing += 1
        else:
            scored.append((s, u, m))
    return scored, missing


def _measure(seed: int) -> dict:
    if seed in _MEMO:
        return _MEMO[seed]
    ctxs, trials = _trial_prompts(seed)
    store = json.loads(Path(ARTIFACT).read_text()) \
        if Path(ARTIFACT).exists() else {"_meta": {"models": {}}}
    revs = store.get("_meta", {}).get("models", {})

    # The held-out guarantee, measured: no canonical sentence of ANY fact in
    # this life may appear in any held-out or mismatched prompt.
    life_sentences = []
    for t in trials:
        if t["kind"] == "report":
            life_sentences.append(_canonical(*t["intent"]))
            life_sentences += [_canonical(*f) for f in t["distract"]]
            life_sentences += [_canonical(*f) for f in t["fabricated"]]
    prompt_clean = 1.0
    for _ti, _t, _f, _p, _a, _n, ho_p, mm_p, _s in ctxs:
        if any(c in ho_p or c in mm_p for c in life_sentences):
            prompt_clean = 0.0

    missing = 0
    per = {m: dict(match=[], unan=[], variety=[], live_arm=[], live_ho=[],
                   ho_match=[], ho_unan=[], ho_variety=[],
                   mm_match=[], null_match=[], leak=0,
                   modal=[], ho_modal=[]) for m in MODELS}
    n_spoken = 0
    speak_silence = 0
    n_silence = 0
    gate_rejected_fab = 0
    n_fab_offered = 0

    for t in trials:
        if t["kind"] == "silence":
            n_silence += 1
            if _core_intent(t) is not None:
                speak_silence += 1
            continue

    for ti, trial, fills, pool, arm_p, null_p, ho_p, mm_p, scram in ctxs:
        core_ok = _core_intent(trial) == trial["intent"]
        n_fab_offered += N_FAB
        gate_rejected_fab += N_FAB - sum(
            1 for _u, m in pool if isinstance(m, tuple) and m[0] == "FAB")
        if not fills:
            continue              # the mute branch: nothing realizes the
        n_spoken += 1             # structure, so this trial has no emission

        for model_id in MODELS:
            rev = revs.get(model_id, "local")
            arm_sc, m1 = _score(store, revs, model_id, arm_p, fills)
            ho_sc, m2 = _score(store, revs, model_id, ho_p, fills)
            mm_sc, m3 = _score(store, revs, model_id, mm_p, fills)
            null_sc, m4 = _score(store, revs, model_id, null_p, pool)
            canon = _canonical(*trial["intent"])
            s_v_arm = store.get(_key(model_id, rev, arm_p, canon))
            s_s_arm = store.get(_key(model_id, rev, arm_p, scram))
            s_v_ho = store.get(_key(model_id, rev, ho_p, canon))
            s_s_ho = store.get(_key(model_id, rev, ho_p, scram))
            m5 = sum(1 for x in (s_v_arm, s_s_arm, s_v_ho, s_s_ho)
                     if x is None)
            if m1 or m2 or m3 or m4 or m5:
                missing += m1 + m2 + m3 + m4 + m5
                continue
            p = per[model_id]
            p["live_arm"].append(float(s_v_arm > s_s_arm))
            p["live_ho"].append(float(s_v_ho > s_s_ho))

            def draws_of(scored, ns):
                return [_draw(scored, random.Random(
                    f"{ns}:{seed}:{ti}:{model_id}:{d}"))
                    for d in range(S_DRAWS)]

            arm_d = draws_of(arm_sc, "lg14")
            ho_d = draws_of(ho_sc, "lg14ho")
            mm_d = draws_of(mm_sc, "lg14mm")
            null_d = draws_of(null_sc, "lg10null")   # LG.10's keys: UNMOVED

            for draws, mk, uk, vk, modk in (
                    (arm_d, "match", "unan", "variety", "modal"),
                    (ho_d, "ho_match", "ho_unan", "ho_variety", "ho_modal")):
                meanings = [m for _u, m in draws]
                utts = [u for u, _m in draws]
                p[mk].append(
                    (sum(1 for m in meanings if m == trial["intent"])
                     / S_DRAWS) if core_ok else 0.0)
                p[uk].append(float(len(set(meanings)) == 1))
                p[vk].append(float(len(set(utts)) >= 2))
                p[modk].append(_modal(meanings))
            p["mm_match"].append(
                sum(1 for _u, m in mm_d if m == trial["intent"]) / S_DRAWS)
            p["null_match"].append(
                sum(1 for _u, m in null_d if m == trial["intent"]) / S_DRAWS)
            p["leak"] += sum(1 for d in (arm_d + ho_d + mm_d + null_d)
                             if isinstance(d[1], tuple) and d[1][0] == "FAB")

    def _mean(v):
        return sum(v) / len(v) if v else 0.0

    a, b = per[MODEL_A], per[MODEL_B]
    n_pairs = min(len(a["modal"]), len(b["modal"]))
    swap_agree = _mean([float(a["modal"][i] == b["modal"][i])
                        for i in range(n_pairs)])
    n_ho = min(len(a["ho_modal"]), len(b["ho_modal"]))
    swap_agree_ho = _mean([float(a["ho_modal"][i] == b["ho_modal"][i])
                           for i in range(n_ho)])

    out = {
        "n_report": N_REPORT, "n_silence": n_silence,
        "verdicts_missing": missing,
        "held_out_prompt_clean": prompt_clean,
        # ── the claim: in-scaffold structured decode ──
        "match_on_spoken": round(_mean(a["match"]), 4),
        "match_on_spoken_swap": round(_mean(b["match"]), 4),
        "unanimity_on_spoken": round(_mean(a["unan"]), 4),
        "unanimity_on_spoken_swap": round(_mean(b["unan"]), 4),
        "swap_agree": round(swap_agree, 4),
        "utter_rate": round(n_spoken / max(1, N_REPORT), 4),
        # ── MANDATORY CONJUNCT 1: held-out intents ──
        "match_on_spoken_on_held_out_intents": round(_mean(a["ho_match"]), 4),
        "match_on_held_out_swap": round(_mean(b["ho_match"]), 4),
        "unanimity_on_held_out": round(_mean(a["ho_unan"]), 4),
        "unanimity_on_held_out_swap": round(_mean(b["ho_unan"]), 4),
        # ── LG.10's carried gates ──
        "speak_silence": round(speak_silence / max(1, n_silence), 4),
        "leak_draws": a["leak"] + b["leak"],
        # ── aliveness ──
        "variety_on_spoken": round(_mean(a["variety"]), 4),
        "variety_on_held_out": round(_mean(a["ho_variety"]), 4),
        "liveness": round(_mean(a["live_arm"]), 4),
        "liveness_swap": round(_mean(b["live_arm"]), 4),
        "liveness_struct": round(_mean(a["live_ho"]), 4),
        "liveness_struct_swap": round(_mean(b["live_ho"]), 4),
        # ── reported, never gated ──
        "swap_agree_held_out": round(swap_agree_ho, 4),
        "gate_rejected_fab_frac": round(
            gate_rejected_fab / max(1, n_fab_offered), 4),
        # ── the nulls (read by _control) ──
        "null_match": round(_mean(a["null_match"]), 4),
        "null_match_swap": round(_mean(b["null_match"]), 4),
        "mismatch_match": round(_mean(a["mm_match"]), 4),
        "mismatch_match_swap": round(_mean(b["mm_match"]), 4),
    }
    _MEMO[seed] = out
    return out


def _per_seed(key: str) -> list:
    return [_MEMO[s][key] for s in SEEDS]


def _seeds_complete() -> bool:
    return all(s in _MEMO for s in SEEDS)


def _experiment(seed: int) -> dict:
    m = _measure(seed)
    return {k: v for k, v in m.items()
            if not k.startswith(("null_", "mismatch_"))}


def _control(seed: int) -> dict:
    """The two declared nulls: LG.10's free generation carried unmoved, and
    the mismatched-constraint decoder — the condition that MUST fail. Either
    tracking his state collapses the claim (FAIL via null_match's gate; VOID
    via the mismatch gate, per the registration)."""
    m = _measure(seed)
    return {k: v for k, v in m.items()
            if k.startswith(("null_", "mismatch_")) or k == "verdicts_missing"}


def _void(m: dict, reason: str):
    m["void_reason"] = reason
    return Status.VOID


def _check(m: dict, c: dict):
    # ── rig: could the question be asked at all ──
    if not _seeds_complete():
        return _void(m, "not every seed produced a measurement")
    if max(_per_seed("verdicts_missing")) > 0:
        return _void(m, "the cached artifact does not cover these prompts")
    if min(_per_seed("held_out_prompt_clean")) < 1.0:
        return _void(m, "a canonical sentence leaked into a held-out or "
                        "mismatched prompt: the held-out conjunct was never "
                        "tested")
    if min(_per_seed("liveness")) < LIVENESS_MIN \
            or min(_per_seed("liveness_swap")) < LIVENESS_MIN \
            or min(_per_seed("liveness_struct")) < LIVENESS_MIN \
            or min(_per_seed("liveness_struct_swap")) < LIVENESS_MIN:
        return _void(m, "a scorer could not tell prose from its own scramble "
                        "under a prompt form this claim is read on")
    # ── the utterance floor, read BEFORE the remaining rig gates ──
    # A mute mouth has a determinate verdict by the registry's falsified_by
    # (the ME.3 starvation failure); a later VOID may not convert it.
    if min(_per_seed("utter_rate")) < UTTER_MIN:
        return False
    if min(_per_seed("variety_on_spoken")) < VARIETY_MIN \
            or min(_per_seed("variety_on_held_out")) < VARIETY_MIN:
        return _void(m, "wording variety on spoken trials below the floor: "
                        "sampler-invariance was never tested")
    if max(_per_seed("mismatch_match")) >= MATCH_MIN \
            or max(_per_seed("mismatch_match_swap")) >= MATCH_MIN:
        return _void(m, "the mismatched-constraint null also clears "
                        "MATCH_MIN: the constraint is not carrying meaning "
                        "(VOID, not FAIL, per the registration)")
    # ── the claim, both conjuncts, and the free-generation null, every seed ──
    return bool(
        min(_per_seed("match_on_spoken")) >= MATCH_MIN
        and min(_per_seed("match_on_spoken_swap")) >= MATCH_MIN
        and min(_per_seed("unanimity_on_spoken")) >= UNANIMITY_MIN
        and min(_per_seed("unanimity_on_spoken_swap")) >= UNANIMITY_MIN
        and min(_per_seed("swap_agree")) >= SWAP_AGREE_MIN
        and min(_per_seed("match_on_spoken_on_held_out_intents")) >= MATCH_MIN
        and min(_per_seed("match_on_held_out_swap")) >= MATCH_MIN
        and min(_per_seed("unanimity_on_held_out")) >= UNANIMITY_MIN
        and min(_per_seed("unanimity_on_held_out_swap")) >= UNANIMITY_MIN
        and max(_per_seed("speak_silence")) == 0.0
        and max(_per_seed("leak_draws")) == 0
        and max(_per_seed("null_match")) <= NULL_MATCH_MAX
        and max(_per_seed("null_match_swap")) <= NULL_MATCH_MAX)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["LG.14"], _experiment, _check,
                    control_fn=_control, ledger=ledger)


if __name__ == "__main__":
    if "--llm-pass" in sys.argv:
        llm_pass()
    else:
        sys.path.insert(0, str(REPO))
        print(json.dumps(_measure(0), indent=2))
