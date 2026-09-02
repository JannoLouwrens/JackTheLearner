"""LG.10 — Jack chooses what to say; the LLM only chooses how.

GOAL.md: *"The LLM is his mouth, never his mind."* The registry's hypothesis
names three independent measurements and this file implements all three on one
pipeline: (a) same state, different LLM sampling seeds -> same MEANING,
different wording; (b) different state, same LLM -> different meaning; (c)
swap the LLM for a different frozen model -> meaning preserved. The null is
LLM free-generation with no core-selected intent, whose meaning must NOT
track his state; the control is SILENCE — a state with nothing to report must
produce no utterance at all.

THE PIPELINE, exactly the registry notes' second form ("selects among
LLM-proposed phrasings"), because the first form (free generation + parse)
would put the meaning READOUT inside another model. The core's choosing is a
deterministic function of state: the single FRESH fact in his record is the
intent; no fresh fact means intent None and the mouth stays shut. The LLM's
choosing is a seeded softmax draw over a CANDIDATE POOL scored by
length-normalised continuation log-probability (LG.01's v2 readout — no
letters, no positions).

WHY THIS IS FALSIFIABLE AND NOT RIGGED — the only interesting question about
a selection pipeline. The pool is NOT restricted to phrasings of the intent:

    3 phrasings x the INTENT fact          (the meaning the core chose)
    3 phrasings x 4 DISTRACTOR facts       (truthful — other things he could
                                            say; the gate passes them, so the
                                            LLM is free to change the subject)
    2 PHATIC lines                         (assert nothing; fluent attractors
                                            under length-normalisation)
    2 FABRICATED facts (another life's)    (the verification gate must reject
                                            these BEFORE scoring — measured,
                                            not assumed: a broken gate lets
                                            them through and `leak_draws`
                                            counts every time one is drawn)

If the intent conditioning in the prompt does not dominate the model's
fluency preferences, draws land on distractor meanings or phatic lines and
`match` fails. If the model imposes its own preferences over the core's, the
swap changes meaning and `swap_agree` fails. If the scaffold leaks the intent
into the null prompt, the null tracks state and the CONTROL fails. Nothing
here is true by construction except two things, declared rather than buried:
the arm's silence on intent-None states is a property of the intent function
this file implements (measured by executing it, but a sampler over a
non-empty pool cannot abstain, so the ungated mouth's always-speaking is
structural); and meaning DECODE is a table lookup (each pool candidate was
built from a known fact), which is the multiple-choice bargain LG.00/LG.01
already made — the alternative is a second model grading the first.

WHAT A DEAD INSTRUMENT READS. A scorer that returns the same number for every
candidate makes draws uniform: match ~ 3/17 = 0.18 -> FAIL, not PASS — and
before that the LIVENESS gate VOIDs: each model must rank the canonical
intent sentence above a word-scramble of itself on >= LIVENESS_MIN of trials
(24th-audit B3: an instrument must prove it was alive). A sampler with no
freedom reads unanimity 1.0 vacuously, so WORDING VARIETY is an aliveness
gate too: if fewer than VARIETY_MIN of report trials show >= 2 distinct
wordings across draws, sampler-invariance was never tested and the run is
VOID, not PASS.

GATES, pre-registered 2026-09-02 BEFORE any LG.10 number existed. Worst-seed,
read from _MEMO exactly as LG.00 does (the moment-bound VOID of its attempt 1
is why):

    PASS  — on every seed:
              match      >= 0.90 on BOTH models   (a)+(b): drawn meaning ==
                                                   the core's intent
              unanimity  >= 0.90 on BOTH models   (a): all 5 draws of a trial
                                                   agree on meaning
              swap_agree >= 0.90                  (c): per-trial modal meaning
                                                   identical across models
              speak_silence == 0.0                the control: nothing to
                                                   report -> nothing said
              leak_draws == 0                     no fabricated line ever drawn
              null_match <= 0.35 on BOTH models   the null does NOT track
                                                   state (chance ~ 0.18)
    FAIL  — any of the above on the wrong side: the sampler flips meaning,
            the swap flips meaning, a stranger-model's fluency outranks his
            intent, the mouth speaks with nothing to say, or the null tracks
            state (the scaffold leaks, so the arm's tracking proves nothing).
    VOID  — verdicts missing from the artifact (keys are content-hashed:
            change the prompts, the pool, the scaffold or the weights and
            run() refuses stale verdicts), a model failing LIVENESS, or
            wording variety below VARIETY_MIN (invariance never tested).

TEMPERATURE, declared with its rationale because it is the one knob that
could have been fitted: TEMP = 0.25 over length-normalised log-probs. Content
mismatch costs whole nats under a prompt that names the content words, so
cross-meaning gaps should dwarf T; same-meaning rewordings differ by tenths
of a nat, so wording stays live. Chosen from that reasoning on 2026-09-02,
before the artifact existed, and it does not move — a T fitted after seeing
draws would be the seed-lottery in different clothes.

THE SWAP MODEL, and the limitation recorded the way LG.01 recorded the 360M.
MODEL_B is SmolLM2-135M-Instruct: genuinely different frozen weights, same
family and tokenizer. The swap therefore tests weight-independence of
meaning, not family-independence — a cross-family swap is a stronger claim
someone should buy later. Chosen because the 1.7B (cached here) runs ~7 GB
fp32 against this box's tenant-safety ceiling, ~5x the 2.5 GB precedent
LG.00 set. Style differences across models are REPORTED (style_change), not
gated: the hypothesis demands meaning preservation; it does not demand the
two models disagree about wording.

THE LLM PASS IS OFFLINE AND run() NEVER LOADS A MODEL — LG.01's rule, same
reasons (T0.07: in-process SmolLM2 is a 6.9 GB mistake; Budget.CPU is a
10-minute kill). Precompute with

    python -m experiments.tests.lg_10_jack_chooses_what_to_say --llm-pass

through `scripts/launch_detached.sh`. Draws happen inside run(), seeded from
(seed, trial, model, draw) strings, so they are deterministic given the
artifact and free given the cache.

COVERS: language (parent) (claim).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from .lg_01_lived_necessary_probes import (
    PLACES, RESOURCES, SCAFFOLD_SHA, _prompt,
)

REPO = Path(__file__).resolve().parents[2]

# The claim is about the diary his intent is drawn from and the scaffold the
# candidates are scored under; both hash into this certificate.
IMPL_DEPS = ["EpisodicMemory.py",
             "experiments/tests/lg_01_lived_necessary_probes.py"]

SEEDS = [0, 1, 2]
ARTIFACT = "/data/lg10_llm_verdicts.json"

# ── the two frozen mouths ────────────────────────────────────────────────────
MODEL_A = "HuggingFaceTB/SmolLM2-360M-Instruct"   # the parent, settled decisions
MODEL_B = "HuggingFaceTB/SmolLM2-135M-Instruct"   # the swap arm
MODELS = (MODEL_A, MODEL_B)

# ── world sizing, pre-registered ─────────────────────────────────────────────
N_FACTS = 16          # found-facts recorded per life
N_REPORT = 12         # trials with exactly one fresh fact (the intent)
N_SILENCE = 4         # trials with nothing fresh: intent must be None
N_DISTRACT = 4        # truthful other-facts in every trial's pool
N_FAB = 2             # fabricated (wrong-life) facts offered to the gate
N_PHRASINGS = 3
FRESH_AGE = 3600.0    # seconds; younger than this is "fresh"
NOW = 1e9             # fixed clock, LG.00's rule: retrieval must not depend
                      # on when the test ran

# ── gates, pre-registered 2026-09-02 BEFORE any number existed ───────────────
TEMP = 0.25
S_DRAWS = 5
MATCH_MIN = 0.90
UNANIMITY_MIN = 0.90
SWAP_AGREE_MIN = 0.90
NULL_MATCH_MAX = 0.35    # chance is ~3/17 = 0.18 per draw; 0.35 is ~2x chance
VARIETY_MIN = 0.30       # aliveness: below this, invariance was never tested
LIVENESS_MIN = 0.80      # canonical > scramble, per model, per seed

PHATIC = ["Hello there.", "Hmm, let me think."]

ARM_ASK = ("You are Jack. Say one thing from your record, in your own "
           "words. You have decided to tell them: {intent}")
NULL_ASK = ("You are Jack. Say one thing from your record, in your own "
            "words.")


# ─────────────────────────────────────────────────────────────────────────────
# His life, his states, and the candidate pools
# ─────────────────────────────────────────────────────────────────────────────
def _canonical(res: str, place: str) -> str:
    return f"I found {res} at {place}."


def _phrasings(res: str, place: str) -> list:
    """Three wordings of ONE meaning. The first is the canonical sentence, so
    the liveness probe's 'verbatim' key is already in the pool."""
    return [_canonical(res, place),
            f"It was at {place} that I found {res}.",
            f"When I looked, {res} turned up at {place}."]


def _scramble(sentence: str, rng: random.Random) -> str:
    words = sentence.rstrip(".").split()
    while True:
        rng.shuffle(words)
        s = " ".join(words) + "."
        if s != sentence:
            return s


def _gate_ok(mem, res: str, place: str) -> bool:
    """The verification gate, extractive by construction: an utterance may
    assert (res, place) only if ONE retrieved diary event contains both.
    Fabricated pairs retrieve their nearest neighbour, which names one half
    at most. A broken gate here is MEASURED downstream: fabricated lines
    enter the pool and `leak_draws` can count them being drawn."""
    hits = mem.recall(f"found {res} at {place}", top_k=1, now=NOW)
    if not hits:
        return False
    text = hits[0].event.text.lower()
    return res in text and place in text


def _build_trials(seed: int):
    """One life in ME.9's store; N_REPORT states with exactly one fresh fact
    and N_SILENCE states with none. Every fact is this life's RNG's alone."""
    sys.path.insert(0, str(REPO))
    from EpisodicMemory import EpisodicMemory

    rng = random.Random(7000 + seed)
    tmp = Path(tempfile.mkdtemp())
    mem = EpisodicMemory(path=tmp / "life.jsonl")

    facts = list(zip(rng.sample(RESOURCES, N_FACTS),
                     rng.sample(PLACES, N_FACTS)))
    t = NOW - 30 * 24 * 3600.0            # a month ago: everything starts stale
    for res, place in facts:
        mem.record("did", "jack", f"jack found {res} at {place}", t=t)
        t += 30.0

    # Fabricated facts: same vocabulary, ANOTHER life's draws, verified absent
    # from this life so the gate's verdict on them has a known right answer.
    rng_f = random.Random(7500 + seed)
    fabricated = []
    have = set(facts)
    while len(fabricated) < N_REPORT * N_FAB:
        pair = (rng_f.choice(RESOURCES), rng_f.choice(PLACES))
        if pair not in have and pair not in fabricated:
            fabricated.append(pair)

    trials = []
    for i in range(N_REPORT):
        res, place = facts[i]
        others = [f for f in facts if f != (res, place)]
        distract = rng.sample(others, N_DISTRACT)
        fabs = fabricated[i * N_FAB:(i + 1) * N_FAB]
        # The state: his diary, with fact i freshly re-experienced.
        state_ages = {f: (FRESH_AGE / 2 if f == (res, place) else None)
                      for f in facts}
        trials.append(dict(kind="report", intent=(res, place),
                           distract=distract, fabricated=fabs,
                           ages=state_ages))
    for i in range(N_SILENCE):
        trials.append(dict(kind="silence", intent=None, distract=[],
                           fabricated=[], ages={f: None for f in facts}))
    return mem, trials


def _core_intent(trial: dict):
    """The core's choosing — deterministic in the state: the unique fresh
    fact, or None. THIS is the arm's abstention path; if it is wrong, the
    speak_silence gate fails."""
    fresh = [f for f, age in trial["ages"].items()
             if age is not None and age < FRESH_AGE]
    return fresh[0] if len(fresh) == 1 else None


def _pool(mem, trial: dict) -> list:
    """(utterance, meaning) candidates the mouth may pick from, AFTER the
    verification gate. meaning is a fact pair, or None for phatic lines."""
    out = []
    for res, place in [trial["intent"]] + trial["distract"]:
        if _gate_ok(mem, res, place):
            for u in _phrasings(res, place):
                out.append((u, (res, place)))
    for res, place in trial["fabricated"]:
        if _gate_ok(mem, res, place):          # a broken gate shows up HERE
            out.append((_canonical(res, place), ("FAB", res, place)))
    for u in PHATIC:
        out.append((u, None))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Keys and prompt enumeration (shared by the offline pass and run())
# ─────────────────────────────────────────────────────────────────────────────
def _key(model_id: str, revision: str, prompt: str, option: str) -> str:
    return hashlib.sha256(
        f"{model_id}@{revision}\x00{SCAFFOLD_SHA}\x00{prompt}\x00{option}"
        .encode()).hexdigest()


def _prompts_for(seed: int) -> list:
    """Every (prompt, candidate) pair each model must score for one seed.
    One enumeration, so the offline pass and run() can never disagree."""
    mem, trials = _build_trials(seed)
    pairs = []
    for ti, trial in enumerate(trials):
        if trial["kind"] != "report":
            continue                         # the arm never scores silence
        pool = _pool(mem, trial)
        arm_p = _prompt(ARM_ASK.format(intent=_canonical(*trial["intent"])))
        null_p = _prompt(NULL_ASK)
        for u, _m in pool:
            pairs.append((arm_p, u))
            pairs.append((null_p, u))
        # liveness probe: canonical vs its own scramble, arm prompt only
        rng = random.Random(f"lg10scramble:{seed}:{ti}")
        pairs.append((arm_p, _scramble(_canonical(*trial["intent"]), rng)))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# The offline pass (never called by run())
# ─────────────────────────────────────────────────────────────────────────────
def llm_pass(seeds=None, out_path: str = ARTIFACT) -> dict:
    """Score every (prompt, candidate) under BOTH frozen models, sequentially
    so peak RSS is one model at a time (360M fp32 ~2.5 GB, LG.00's measured
    precedent; the 135M is ~0.6 GB). LG.01's scoring, line for line."""
    import gc
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
        print(f"[lg10] {model_id}: {len(todo)} pairs to score", flush=True)

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
                    print(f"[lg10] {model_id} {s + len(chunk)}/{len(todo)}",
                          flush=True)
        del model, tok
        gc.collect()
        Path(out_path).write_text(json.dumps(store))
    print(f"[lg10] wrote {out_path} ({len(store) - 1} verdicts)", flush=True)
    return store


# ─────────────────────────────────────────────────────────────────────────────
# Scoring: deterministic seeded draws over cached verdicts
# ─────────────────────────────────────────────────────────────────────────────
def _draw(scored: list, rng: random.Random):
    """One softmax(logprob/TEMP) draw. scored: [(logprob, utterance, meaning)]."""
    mx = max(s for s, _u, _m in scored)
    ws = [math.exp((s - mx) / TEMP) for s, _u, _m in scored]
    tot = sum(ws)
    r = rng.random() * tot
    acc = 0.0
    for w, (_s, u, m) in zip(ws, scored):
        acc += w
        if r <= acc:
            return u, m
    return scored[-1][1], scored[-1][2]


_MEMO: dict = {}


def _measure(seed: int) -> dict:
    if seed in _MEMO:
        return _MEMO[seed]
    mem, trials = _build_trials(seed)
    store = json.loads(Path(ARTIFACT).read_text()) \
        if Path(ARTIFACT).exists() else {"_meta": {"models": {}}}
    revs = store.get("_meta", {}).get("models", {})

    missing = 0
    per_model = {m: dict(match=[], unan=[], variety=[], live=[],
                         modal=[], modal_utt=[], null_match=[], leak=0)
                 for m in MODELS}
    speak_silence = 0
    n_silence = 0
    gate_rejected_fab = 0
    n_fab_offered = 0

    for ti, trial in enumerate(trials):
        if trial["kind"] == "silence":
            n_silence += 1
            if _core_intent(trial) is not None:
                speak_silence += 1        # the mouth would have been handed
            continue                      # an intent it should not have
        # The core's choosing, executed — not assumed. If it disagrees with
        # the state that was built, that is an ARM fault and every draw of
        # this trial is scored as a meaning miss (never silently repaired).
        core_ok = _core_intent(trial) == trial["intent"]
        pool = _pool(mem, trial)
        n_fab_offered += N_FAB
        gate_rejected_fab += N_FAB - sum(
            1 for _u, m in pool if isinstance(m, tuple) and m[0] == "FAB")
        arm_p = _prompt(ARM_ASK.format(intent=_canonical(*trial["intent"])))
        null_p = _prompt(NULL_ASK)
        rng_s = random.Random(f"lg10scramble:{seed}:{ti}")
        scram = _scramble(_canonical(*trial["intent"]), rng_s)

        for model_id in MODELS:
            rev = revs.get(model_id, "local")
            scored, ok = [], True
            for u, m in pool:
                s = store.get(_key(model_id, rev, arm_p, u))
                if s is None:
                    missing += 1
                    ok = False
                    continue
                scored.append((s, u, m))
            null_scored = []
            for u, m in pool:
                s = store.get(_key(model_id, rev, null_p, u))
                if s is None:
                    missing += 1
                    ok = False
                    continue
                null_scored.append((s, u, m))
            s_verb = store.get(_key(model_id, rev, arm_p,
                                    _canonical(*trial["intent"])))
            s_scram = store.get(_key(model_id, rev, arm_p, scram))
            if s_verb is None or s_scram is None:
                missing += 1
                ok = False
            if not ok:
                continue
            per_model[model_id]["live"].append(float(s_verb > s_scram))

            draws = [_draw(scored,
                           random.Random(f"lg10:{seed}:{ti}:{model_id}:{d}"))
                     for d in range(S_DRAWS)]
            meanings = [m for _u, m in draws]
            utts = [u for u, _m in draws]
            per_model[model_id]["match"].append(
                sum(1 for m in meanings if m == trial["intent"]) / S_DRAWS
                if core_ok else 0.0)
            per_model[model_id]["unan"].append(
                float(len(set(meanings)) == 1))
            per_model[model_id]["variety"].append(
                float(len(set(utts)) >= 2))
            per_model[model_id]["leak"] += sum(
                1 for m in meanings if isinstance(m, tuple) and m[0] == "FAB")
            modal = max(set(meanings), key=meanings.count)
            modal_u = max(set(utts), key=utts.count)
            per_model[model_id]["modal"].append(modal)
            per_model[model_id]["modal_utt"].append(modal_u)

            null_draws = [_draw(null_scored,
                                random.Random(
                                    f"lg10null:{seed}:{ti}:{model_id}:{d}"))
                          for d in range(S_DRAWS)]
            per_model[model_id]["null_match"].append(
                sum(1 for _u, m in null_draws if m == trial["intent"])
                / S_DRAWS)
            per_model[model_id]["leak"] += sum(
                1 for _u, m in null_draws
                if isinstance(m, tuple) and m[0] == "FAB")

    def _mean(v):
        return sum(v) / len(v) if v else 0.0

    a, b = per_model[MODEL_A], per_model[MODEL_B]
    n_pairs = min(len(a["modal"]), len(b["modal"]))
    swap_agree = _mean([float(a["modal"][i] == b["modal"][i])
                        for i in range(n_pairs)])
    style_change = _mean([float(a["modal_utt"][i] != b["modal_utt"][i])
                          for i in range(n_pairs)])

    out = {
        "n_report": N_REPORT, "n_silence": n_silence,
        "verdicts_missing": missing,
        # ── the claim ──
        "meaning_tracks_state_not_model": round(_mean(a["match"]), 4),
        "match_swap": round(_mean(b["match"]), 4),
        "unanimity": round(_mean(a["unan"]), 4),
        "unanimity_swap": round(_mean(b["unan"]), 4),
        "swap_agree": round(swap_agree, 4),
        "speak_silence": round(speak_silence / max(1, n_silence), 4),
        "leak_draws": a["leak"] + b["leak"],
        # ── aliveness ──
        "variety": round(_mean(a["variety"]), 4),
        "liveness": round(_mean(a["live"]), 4),
        "liveness_swap": round(_mean(b["live"]), 4),
        # ── reported, not gated ──
        "style_change": round(style_change, 4),
        "gate_rejected_fab_frac": round(
            gate_rejected_fab / max(1, n_fab_offered), 4),
        # ── the null (read by _control) ──
        "null_match": round(_mean(a["null_match"]), 4),
        "null_match_swap": round(_mean(b["null_match"]), 4),
    }
    _MEMO[seed] = out
    return out


def _per_seed(key: str) -> list:
    return [_MEMO[s][key] for s in SEEDS]


def _seeds_complete() -> bool:
    return all(s in _MEMO for s in SEEDS)


def _experiment(seed: int) -> dict:
    m = _measure(seed)
    return {k: v for k, v in m.items() if not k.startswith("null_")}


def _control(seed: int) -> dict:
    """The declared null: free generation, no core-selected intent, same pool,
    same sampler. Its meaning must NOT track his state — if it does, the
    scaffold leaks state and the arm's tracking proves nothing."""
    m = _measure(seed)
    return {"null_match": m["null_match"],
            "null_match_swap": m["null_match_swap"],
            "verdicts_missing": m["verdicts_missing"]}


def _check(m: dict, c: dict):
    # ── rig gates: VOID, not FAIL — a run that could not ask the question ──
    if not _seeds_complete():
        return Status.VOID
    if max(_per_seed("verdicts_missing")) > 0:
        return Status.VOID      # the artifact does not cover these prompts
    if min(_per_seed("liveness")) < LIVENESS_MIN \
            or min(_per_seed("liveness_swap")) < LIVENESS_MIN:
        return Status.VOID      # a scorer that cannot tell prose from scramble
    if min(_per_seed("variety")) < VARIETY_MIN:
        return Status.VOID      # sampler-invariance was never actually tested
    # ── the claim, and the control, on EVERY seed ──
    return bool(
        min(_per_seed("meaning_tracks_state_not_model")) >= MATCH_MIN
        and min(_per_seed("match_swap")) >= MATCH_MIN
        and min(_per_seed("unanimity")) >= UNANIMITY_MIN
        and min(_per_seed("unanimity_swap")) >= UNANIMITY_MIN
        and min(_per_seed("swap_agree")) >= SWAP_AGREE_MIN
        and max(_per_seed("speak_silence")) == 0.0
        and max(_per_seed("leak_draws")) == 0
        and max(_per_seed("null_match")) <= NULL_MATCH_MAX
        and max(_per_seed("null_match_swap")) <= NULL_MATCH_MAX)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["LG.10"], _experiment, _check,
                    control_fn=_control, ledger=ledger)


if __name__ == "__main__":
    if "--llm-pass" in sys.argv:
        llm_pass()
    else:
        sys.path.insert(0, str(REPO))
        print(json.dumps(_measure(0), indent=2))
