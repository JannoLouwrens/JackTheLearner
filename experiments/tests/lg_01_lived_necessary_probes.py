"""LG.01 — the life-questions are real questions, certified lived-necessary.

LG.00 asks whether full Jack (learned core + diary + frozen LLM) beats
LLM-ALONE on questions about HIS world, HIS body and HIS history. That claim
is only as good as the questions. If a probe is answerable from a language
model's priors, the cell cannot measure grounding AT ANY SCALE — this is
LANGUAGE_GROUNDING.md Finding 1 (2603.19233: libero_object scores 60-100%
REGARDLESS of the prompt). So the instrument gets certified before any arm is
scored on it, which is PG.7's leak probe transported to Q&A.

THE TWO LEGS, both per question and never on average:

  1. THE DIARY-ORACLE ANSWERS IT.  A deterministic EXTRACTIVE reader over
     ME.9's attributed channels: it retrieves one event and picks the option
     that appears IN that event's text. It never generates. >= ORACLE_MIN on
     the retained set, or the question certifies nothing — a probe the record
     cannot answer is not a probe about his life.

  2. THE FROZEN LLM ALONE DOES NOT.  Identical prompt scaffold, no diary, no
     learned core. A question it answers from priors is EXCLUDED and the
     exclusion logged. This is the leg that makes retention FALSIFIABLE
     rather than curated: the author does not get to say which questions are
     lived-necessary, the null does.

Every fact behind every question is drawn by the seed's RNG — which hollow
holds the water, which shoulder hurt after which haul, which of three people
said what about which object. No prior can know them, so an LLM above chance
on such a question is reading the question rather than knowing the answer,
and that is exactly what the exclusion removes.

PASS  — every category retains >= RETAIN_MIN questions on every seed, the
        oracle clears ORACLE_MIN on the retained set, and the control fails.
FAIL  — any category retaining < RETAIN_MIN (the generator cannot produce
        lived-necessary probes and LG.00 is unrunnable until it can), OR the
        oracle below ORACLE_MIN on retained questions.
VOID  — the LLM leg is not demonstrably alive (see CALIBRATION below), or the
        verdict artifact does not cover the questions actually generated.

CONTROL (declared in the registry; must fail): the oracle with the diary
STRIPPED — empty record, identical machinery. Retrieval returns nothing, so
it falls back to scoring the options by token overlap with the QUESTION
ITSELF. That fallback is deliberate and it is what makes the control able to
fail: if a question's wording leaks its own answer, the stripped reader
scores high and the certification is void. A stripped control that could only
ever return zero would be an identity wearing a control's clothes (LESSONS.md
2026-08-29: "a control scored on a gate that mentions the control is a control
that cannot fail").

CALIBRATION — WHY THIS SPEC CANNOT BE PASSED BY A DEAD LLM. The exclusion leg
is permissive in exactly one direction: an LLM that answers nothing correctly
excludes nothing and every question is retained. A broken prompt scaffold, a
wrong chat template, a mis-selected logit — each looks IDENTICAL to "the
questions are beautifully lived-necessary". So the null carries its own
liveness proof: CALIB_N general-knowledge questions (capitals, arithmetic,
vocabulary) in the byte-identical scaffold, which a 360M instruct model
should answer well above chance. Below CALIB_MIN the run is VOID, not PASS.
This is the 24th audit's B3 rule — an at-chance control must carry proof its
instrument was alive — applied to a null whose at-chance reading is the
result the author wants.

VOID RECORD — attempt 1, 2026-08-30 17:38 UTC, and the calibration gate is
what caught it. The v1 readout listed the four options as A/B/C/D and read
the null's answer off the letter logits, with each question scored under four
cyclic rotations to defeat position bias. It returned:

    calib_acc 0.2500 (std 0.0)   llm_mean_frac 0.2525   n_excluded 5 / 102
    oracle 34/34 per category    stripped control at chance

Chance is 0.25 on both. WITHOUT THE CALIBRATION LEG THIS WAS A CLEAN PASS —
every category retaining ~33 of 34, an oracle at ceiling, a control at the
floor. The probe set would have been certified against a null that cannot
answer "what is the capital of France?", and LG.00 would have been scored on
it. That is the whole reason the leg exists, and it paid for itself on the
first run.

THE FAULT, diagnosed rather than guessed. Asked the capital of France with
Paris on the table, SmolLM2-360M's next-token distribution ranks
'D' 17.71 > 'C' 17.37 > 'B' 17.20 > 'A' 16.51 — a monotone preference for the
LAST letter — while 'Paris' itself sits fifth in the same top-5. The model
KNOWS the answer and cannot express it as a letter. Identical behaviour under
the tokenizer's own chat template, so the scaffold was not the fault: the
LETTER READOUT was. And rotation does not rescue it — a purely positional
picker scores exactly 1/N_OPTIONS across N rotations, which is precisely the
0.2525 observed, indistinguishable from an ideal lived-necessary probe set.

THE REPAIR (v2, and it is STRENGTHEN-ONLY). The null is no longer shown the
options at all. Each candidate answer is scored as a CONTINUATION of the bare
question, by length-normalised log-probability, and the null's answer is the
argmax. There is no letter, no listed order, and therefore no position for
bias to live in. This makes the null STRONGER — calibration went 0.2500 ->
0.8333 on the identical questions — and a stronger null excludes MORE
questions and retains fewer, so the repair moves the spec's own bar in the
harder direction. The retention band tightened with it: v1 retained a
question the null got right in one placement of four; v2 retains it only if
the null is outright WRONG (CHANCE_BAND_HI = 0.0).

THE LLM LEG IS OFFLINE, AND run() NEVER LOADS A MODEL. T0.07's throughput
lesson: `UnifiedBrainConfig()` loading SmolLM2 in-process is a 6.9 GB mistake,
and Budget.CPU is a 10-minute kill. So the null's verdicts are precomputed by
`python -m experiments.tests.lg_01_lived_necessary_probes --llm-pass` into
LLM_ARTIFACT, and run() only READS them. The artifact is keyed by
sha256(model revision + scaffold sha + the exact prompt text), so it is
tamper-evident by construction: change the questions, the scaffold, or the
weights and every key changes, run() finds them missing and returns VOID
rather than certifying against stale verdicts. A precomputed null that can be
silently reused is not a null.

WHICH MODEL, AND WHY NOT THE BIGGER ONE. SmolLM2-360M-Instruct — the model
the settled decisions name for dialogue ("frozen and out-of-process, never an
nn.Module submodule"). The 1.7B is also cached on this box and is NOT used
here: T0.07 measured it at 6.9 GB in-process, over this box's ~1.5 GB ceiling
with paying tenants on it. That substitution runs the WEAK direction for this
spec — a weaker null excludes fewer questions and retains more — so it is
recorded here as a limitation rather than buried: LG.01 certifies probes
against the 360M parent Jack actually gets, and if LG.00 is ever run against
a larger parent, its probe set must be re-certified against that parent.

MEASURED ON THIS BOX 2026-08-30, before the design was fixed: fp32 is the
only viable dtype on these ARM cores — 0.40 s/prompt at batch 8, against
6.1 s/prompt for fp16 and worse for bf16 (no native kernels). fp32 weights
are 1.44 GB and the pass peaks near 2.5 GB RSS, which is why it is a
short-lived detached process at nice 19 and not part of run().

COVERS: language (parent) (fixture).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

# The claim is about the certified probe set read by ME.9's attributed store;
# the store's code hashes into this certificate.
IMPL_DEPS = ["EpisodicMemory.py"]

SEEDS = [0, 1, 2]

# ── the frozen parent ────────────────────────────────────────────────────────
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
LLM_ARTIFACT = "/data/lg01_llm_verdicts.json"

# ── gates, pre-registered 2026-08-30 BEFORE any question was scored ──────────
# RETAIN_MIN and ORACLE_MIN are the registry's own numbers, verbatim.
N_OPTIONS = 4
CHANCE = 1.0 / N_OPTIONS               # 0.25
N_GEN = 34                             # generated per category, per seed
RETAIN_MIN = 20                        # registry: ">= 20 questions per category"
ORACLE_MIN = 0.95                      # registry: ">= 0.95 on the retained set"
# v2 readout (see VOID RECORD): the null answers by ranking the OPTIONS, so a
# question is retained only when the null is outright WRONG on it. That is
# strictly stricter than v1's "correct on at most one of four placements".
CHANCE_BAND_HI = 0.0
# The stripped reader guesses among options it cannot retrieve, so it sits at
# CHANCE by construction. With >= 3*RETAIN_MIN = 60 retained questions the
# binomial sd at chance is sqrt(.25*.75/60) = 0.056; 0.45 is 3.6 sd above.
# It is a ceiling a LEAKING question set can exceed, not one noise can trip.
STRIPPED_CEIL = 0.45
# Liveness of the null. Chance is 0.25; a live 360M instruct model must clear
# twice chance on general knowledge in this scaffold or its at-chance reading
# on the probes proves nothing about the probes.
CALIB_MIN = 0.50
CALIB_N = 12

CATEGORIES = ("his_world", "his_body", "his_history")

# ── the vocabulary his life is drawn from ────────────────────────────────────
SHAPES = ["hollow", "fallen", "split", "leaning", "narrow",
          "broad", "crooked", "sunken", "flat", "steep"]
FEATURES = ["stump", "log", "ridge", "rock", "bank",
            "boulder", "hedge", "ford", "gully", "cairn"]
COLOURS = ["pale", "dark", "red", "grey", "green", "amber", "white", "brown"]
STUFFS = ["berries", "flint", "moss", "root", "stones",
          "reeds", "clay", "bark", "shells", "seeds"]
BODYPARTS = ["left knee", "right shoulder", "lower back", "right ankle",
             "left wrist", "neck", "right hip", "left elbow", "ribs", "jaw"]
OBJECTS = ["ladder", "kettle", "compass", "lantern", "shovel",
           "drum", "anchor", "kite", "basket", "sledge", "net", "pole"]
DETAILS = ["cracked", "freshly painted", "smelling of smoke", "missing",
           "too heavy", "leaning sideways", "hollow-sounding", "brand new",
           "stuck fast", "turning green", "soaked through", "half buried"]
SPEAKERS = ["ada", "bruno", "chika"]

PLACES = [f"the {s} {f}" for s in SHAPES for f in FEATURES]        # 100
RESOURCES = [f"the {c} {s}" for c in COLOURS for s in STUFFS]      # 80

# ── the prompt scaffold; its sha is part of every artifact key ───────────────
# Hardcoded rather than taken from the tokenizer's chat template so that a
# transformers upgrade cannot silently change the null's prompt underneath a
# cached verdict.
SCAFFOLD = (
    "<|im_start|>user\n{q}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
SCAFFOLD_SHA = hashlib.sha256(SCAFFOLD.encode()).hexdigest()[:16]

# General knowledge in the identical scaffold. Deliberately easy: this is a
# liveness floor, not a benchmark. (question, correct, [distractors])
CALIBRATION = [
    ("What is the capital of France?", "Paris", ["Madrid", "Rome", "Berlin"]),
    ("What is 7 plus 5?", "12", ["10", "13", "15"]),
    ("Which animal barks?", "a dog", ["a fish", "a snake", "a moth"]),
    ("What colour is fresh snow?", "white", ["black", "purple", "orange"]),
    ("How many days are in a week?", "7", ["5", "10", "12"]),
    ("What is the opposite of hot?", "cold", ["loud", "heavy", "early"]),
    ("Which one is a fruit?", "an apple", ["a hammer", "a river", "a shoe"]),
    ("What do bees make?", "honey", ["bricks", "paper", "glass"]),
    ("What is 3 times 4?", "12", ["7", "9", "14"]),
    ("In which season does snow usually fall?", "winter",
     ["summer", "spring", "harvest"]),
    ("What is the largest ocean on Earth?", "the Pacific",
     ["the Baltic", "the Caspian", "the Aral"]),
    ("What do you use to unlock a door?", "a key",
     ["a spoon", "a candle", "a pillow"]),
]


# ─────────────────────────────────────────────────────────────────────────────
# His life, and the questions it makes answerable
# ─────────────────────────────────────────────────────────────────────────────
def _build_life(seed: int, mem_path):
    """Record one life into ME.9's attributed store and return the probe set.

    Every fact is drawn by `rng`, so the answer to every question is a
    property of THIS life and of nothing else. That is what lived-necessary
    means operationally, and it is why the null should be at chance.
    """
    sys.path.insert(0, str(REPO))
    from EpisodicMemory import EpisodicMemory

    rng = random.Random(1000 + seed)
    mem = EpisodicMemory(path=mem_path)
    t = 2_000_000.0
    probes: list[dict] = []

    def _opts(correct, pool):
        others = [x for x in pool if x != correct]
        return [correct] + rng.sample(others, N_OPTIONS - 1)

    # ── his world: where the things he needs turned out to be ──────────────
    resources = rng.sample(RESOURCES, N_GEN)
    places = rng.sample(PLACES, N_GEN)
    for res, place in zip(resources, places):
        mem.record("did", "jack", f"jack found {res} at {place}", t=t); t += 30.0
        probes.append(dict(category="his_world",
                           question=f"Where did Jack find {res}?",
                           correct=place, options=_opts(place, PLACES),
                           channel="did", speaker="jack", cue=res))

    # ── his body: what his own body did when he used it ────────────────────
    for i in range(N_GEN):
        part = rng.choice(BODYPARTS)
        obj, place = rng.choice(OBJECTS), rng.choice(PLACES)
        deed = f"hauling the {obj} up {place}"
        mem.record("did", "jack",
                   f"jack felt it in his {part} after {deed}", t=t); t += 30.0
        probes.append(dict(category="his_body",
                           question=f"Which part of Jack hurt after {deed}?",
                           correct=part, options=_opts(part, BODYPARTS),
                           channel="did", speaker="jack", cue=deed))

    # ── his history: who told him what, and about which thing ──────────────
    pairs = [(sp, ob) for sp in SPEAKERS for ob in OBJECTS]
    for sp, obj in rng.sample(pairs, N_GEN):
        detail = rng.choice(DETAILS)
        mem.record("heard", sp,
                   f"{sp} told jack the {obj} was {detail}", t=t); t += 30.0
        # Jack answers, so the `said` channel is populated and the retrieval
        # in leg 1 has to survive an interleaved store (ME.9's whole point).
        mem.record("said", "jack",
                   f"jack replied about the {obj}", t=t); t += 30.0
        probes.append(dict(category="his_history",
                           question=f"What did {sp} tell Jack about the {obj}?",
                           correct=detail, options=_opts(detail, DETAILS),
                           channel="heard", speaker=sp, cue=obj))

    # Option ORDER must not encode the answer: shuffle once, deterministically.
    for p in probes:
        rng.shuffle(p["options"])
    return mem, probes


# ─────────────────────────────────────────────────────────────────────────────
# Leg 1 — the extractive diary-oracle
# ─────────────────────────────────────────────────────────────────────────────
def _tok(s: str) -> set:
    return {w for w in s.lower().replace("?", " ").replace(".", " ").split()
            if len(w) > 2}


def _oracle_answer(mem, probe: dict, stripped: bool) -> str | None:
    """Retrieve ONE event and read the answer out of it. Never generates.

    `stripped` is the declared control: the record is empty, so retrieval
    returns nothing and the reader falls back to scoring the options against
    the QUESTION's own words. If a question leaks its answer, this scores.
    """
    hits = []
    if probe["channel"] == "heard":
        hits = mem.what_did_they_tell_me(probe["speaker"], probe["cue"], top_k=1)
    else:
        hits = mem.what_did_i_do(probe["cue"], top_k=1)
    if hits:
        text = (hits[0].event.text if hasattr(hits[0], "event")
                else getattr(hits[0], "text", str(hits[0]))).lower()
        for opt in probe["options"]:
            if opt.lower() in text:
                return opt
        return None                       # retrieved, but says nothing: abstain
    if not stripped:
        return None
    # Stripped fallback: best token overlap with the question itself.
    qt = _tok(probe["question"])
    best, best_n = None, -1
    for opt in probe["options"]:
        n = len(_tok(opt) & qt)
        if n > best_n:
            best, best_n = opt, n
    return best


def _oracle_scores(mem, probes: list, stripped: bool = False) -> list:
    return [_oracle_answer(mem, p, stripped) == p["correct"] for p in probes]


# ─────────────────────────────────────────────────────────────────────────────
# Leg 2 — the frozen LLM alone, read from the offline artifact
# ─────────────────────────────────────────────────────────────────────────────
def _prompt(question: str) -> str:
    return SCAFFOLD.format(q=question)


def _key(revision: str, prompt: str, option: str) -> str:
    """Identifies one (weights, scaffold, question, candidate answer) triple."""
    return hashlib.sha256(
        f"{MODEL_ID}@{revision}\x00{SCAFFOLD_SHA}\x00{prompt}\x00{option}"
        .encode()).hexdigest()


def _all_prompts(probes: list) -> list:
    """Every (question, candidate-answer) pair the null must score, plus the
    calibration set. One list, so the offline pass and run() can never
    disagree about what was asked.

    Yields (kind, question_index, prompt, option_text, is_correct).
    """
    out = []
    for i, p in enumerate(probes):
        for opt in p["options"]:
            out.append(("probe", i, _prompt(p["question"]), opt,
                        opt == p["correct"]))
    for i, (q, correct, distr) in enumerate(CALIBRATION[:CALIB_N]):
        for opt in [correct] + list(distr):
            out.append(("calib", i, _prompt(q), opt, opt == correct))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The offline pass (never called by run())
# ─────────────────────────────────────────────────────────────────────────────
def llm_pass(seeds=None, out_path: str = LLM_ARTIFACT) -> dict:
    """Score every prompt with the frozen parent and cache the letters.

    Short-lived and detached by design: fp32 SmolLM2-360M peaks near 2.5 GB on
    this box, which is why it must never live inside run().
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    os.environ.setdefault("HF_HOME", "/data/caches/huggingface")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # RIGHT padding: the option's tokens are scored at their own absolute
    # positions, counted forward from the prompt's length, so the real tokens
    # must start at index 0. (v1 read the last position and needed the
    # opposite; see the VOID RECORD for why that readout is gone.)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, low_cpu_mem_usage=True).eval()
    revision = getattr(model.config, "_commit_hash", None) or "local"

    store = {}
    if Path(out_path).exists():
        store = json.loads(Path(out_path).read_text())
    store.setdefault("_meta", {})["model"] = f"{MODEL_ID}@{revision}"
    store["_meta"]["scaffold_sha"] = SCAFFOLD_SHA

    todo = {}
    for seed in (seeds or SEEDS):
        tmp = Path(tempfile.mkdtemp())
        _, probes = _build_life(seed, tmp / "life.jsonl")
        for _kind, _i, prompt, option, _gold in _all_prompts(probes):
            k = _key(revision, prompt, option)
            if k not in store:
                todo[k] = (prompt, option)
    todo = list(todo.items())
    print(f"[lg01] {len(todo)} (question, answer) pairs to score", flush=True)

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
                # tokens only, length-normalised so a long correct answer is
                # not penalised against a short wrong one.
                span = range(n_pre[j], n)
                tot = sum(float(logp[j, t - 1, ids[t]]) for t in span)
                store[k] = round(tot / max(1, len(span)), 5)
            if s % (B * 20) == 0:
                print(f"[lg01] {s + len(chunk)}/{len(todo)}", flush=True)
    Path(out_path).write_text(json.dumps(store))
    print(f"[lg01] wrote {out_path} ({len(store) - 1} verdicts)", flush=True)
    return store


# ─────────────────────────────────────────────────────────────────────────────
# The experiment
# ─────────────────────────────────────────────────────────────────────────────
def _score(seed: int, stripped: bool) -> dict:
    tmp = Path(tempfile.mkdtemp())
    mem, probes = _build_life(seed, tmp / "life.jsonl")
    if stripped:
        # The control STRIPS THE RECORD. The probe set is byte-identical --
        # the same questions, generated by the same rng draw -- so the only
        # difference between the arms is whether he lived it.
        sys.path.insert(0, str(REPO))
        from EpisodicMemory import EpisodicMemory
        mem = EpisodicMemory(path=tmp / "stripped.jsonl")

    store = {}
    if Path(LLM_ARTIFACT).exists():
        store = json.loads(Path(LLM_ARTIFACT).read_text())
    revision = (store.get("_meta", {}).get("model", "@local").split("@")[-1])

    # Rank each question's options by the null's length-normalised logprob.
    ranked: dict = {}
    missing = 0
    for kind, i, prompt, option, gold in _all_prompts(probes):
        s = store.get(_key(revision, prompt, option))
        if s is None:
            missing += 1
            continue
        ranked.setdefault((kind, i), []).append((s, gold))
    llm_right = [0] * len(probes)
    calib_right = 0
    for (kind, i), scored in ranked.items():
        if len(scored) != N_OPTIONS:
            missing += 1
            continue
        won = max(scored)[1]                    # the null's pick was correct?
        if kind == "probe":
            llm_right[i] = int(won)
        else:
            calib_right += int(won)

    oracle_ok = _oracle_scores(mem, probes, stripped=stripped)
    retained, excluded, per_cat = [], [], {c: 0 for c in CATEGORIES}
    for i, p in enumerate(probes):
        frac = float(llm_right[i])
        if frac <= CHANCE_BAND_HI:
            retained.append(i)
            per_cat[p["category"]] += 1
        else:
            excluded.append({"q": p["question"], "llm_frac": round(frac, 3)})

    oracle_acc = (sum(oracle_ok[i] for i in retained) / len(retained)
                  if retained else 0.0)
    return {
        "n_generated": len(probes),
        "n_retained": len(retained),
        "retained_his_world": per_cat["his_world"],
        "retained_his_body": per_cat["his_body"],
        "retained_his_history": per_cat["his_history"],
        "retained_min_per_category": min(per_cat.values()),
        "oracle_acc_on_retained": round(oracle_acc, 4),
        "llm_mean_frac": round(sum(llm_right) / len(probes), 4),
        "n_excluded": len(excluded),
        "excluded": excluded[:12],
        "calib_acc": round(calib_right / CALIB_N, 4),
        "verdicts_missing": missing,
    }


def _experiment(seed: int) -> dict:
    return _score(seed, stripped=False)


def _control(seed: int) -> dict:
    """The diary STRIPPED. Same machinery, empty record."""
    m = _score(seed, stripped=True)
    return {"stripped_oracle_acc": m["oracle_acc_on_retained"],
            "stripped_retained": m["n_retained"]}


def _check(m: dict, c: dict):
    # ── rig gates: VOID, not FAIL. A run that could not ask the question ──
    if m["verdicts_missing"] > 0:
        return Status.VOID       # artifact does not cover these questions
    if m["calib_acc"] < CALIB_MIN:
        return Status.VOID       # the null is not demonstrably alive
    # ── the claim ──
    return bool(
        m["retained_min_per_category"] >= RETAIN_MIN
        and m["oracle_acc_on_retained"] >= ORACLE_MIN
        and c["stripped_oracle_acc"] <= STRIPPED_CEIL)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["LG.01"], _experiment, _check,
                    control_fn=_control, ledger=ledger)


if __name__ == "__main__":
    if "--llm-pass" in sys.argv:
        llm_pass()
    else:
        print(json.dumps(_experiment(0), indent=2))
