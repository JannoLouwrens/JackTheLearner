"""LG.15 — The character sheet answers for no one: his life answers survive
its removal.

THE CLAIM (registry, registered 2026-10-25 from THE RULING on
`personality-is-a-typed-character-sheet-on-the-answer-path`, option (iii)):
on the SHIPPED answer path — `ResponseGenerator.answer_question`, the method
that builds `persona = personality.get_system_prompt(mood)` and hands it to
the LLM — answers to diary-grounded questions about his own life are
SUBSTANTIVELY UNCHANGED between sheet-present and sheet-absent arms, while
manner is measured and reported as a SEPARATE channel. Three conjuncts, all
binding, none weakened (the ruling's own words):

  1. LIFE ANSWERS UNCHANGED — content divergence between sheet-present and
     sheet-absent answers, read against the IDENTICAL-ARM RESAMPLING null
     (sheet-present vs sheet-present, fresh sampling draws), never against
     zero. Per-seed boolean (`content_unchanged_ok`), gated ==1.0 at the
     mean (run_spec aggregates by mean; the WM.01 incident).
  2. MANNER IS A SEPARATE REPORTED CHANNEL — `manner_delta_sheet_vs_absent`
     is reported, never gated on its value; the PAIR (content, manner) is
     the result. Its instrument carries a MUST-MOVE control: a CONTRADICTORY
     sheet (traits and core_values inverted) must move the manner channel
     beyond the identical-arm null. A manner channel that cannot see an
     inverted sheet is blind, and conjunct 1 scored through a blind
     instrument certifies nothing -> Status.VOID (the LESSONS calibration-arm
     rule: an instrument whose firing side has never been observed to fire
     is a VOID of the instrument, never a FAIL of the claim).
  3. PROVENANCE IS SCORED, NOT ASSUMED — every scored output must carry a
     per-response record of its producing organ, observed AT THE RIG (the
     llm_pass wraps `enc.llm.generate` at runtime and counts calls per
     response; UnifiedBrain.py is NOT edited). A run whose scored outputs
     include unmarked, canned-TEMPLATES, "Sorry, I had trouble..." or
     "I can't answer questions..." responses is VOID, never a verdict — an
     all-canned pair of arms scores "unchanged" for the opposite reason.

THE VENUE CONTAINS THE COSTUME (ruling, binding). This spec calls the shipped
`ResponseGenerator.answer_question` byte-unedited, constructed exactly as
UnifiedBrain wires it: `LLMEncoder` (fp16, as shipped) + `personality=
JACK_PERSONALITY` for the present arm, `personality=None` for the absent arm
(the shipped fallback persona "You are a helpful companion robot.", not a
hand-built neutral prompt). It does NOT score through lg_01's `_prompt`
scaffold — that scaffold never calls `get_system_prompt()`, which is exactly
why LG.00 is blind here. `emotional_state=None` in EVERY arm, disclosed: the
mood channel is held constant so the SHEET is the only difference between
arms. `api_llm=None` in every arm (asserted into the artifact).

WHICH MODEL: SmolLM2-360M-Instruct — the family's pinned parent (lg_01:
"the model Jack actually gets"), loaded through the SHIPPED LLMEncoder with
`llm_model_id` overridden from the 1.7B default (a config field, not a code
edit; T0.07's lesson — the 1.7B in-process is a 6.9 GB mistake). fp16,
because that is what `LLMEncoder.from_pretrained` ships.

HOW THE DIARY ENTERS: `answer_question(question)` takes ONE string and reads
no memory store, so the diary context is embedded in the question text — the
only channel the shipped path has. This is how a retrieval layer would wire
it (CompanionMemory.retrieve -> question with context) and it is disclosed
rather than hidden: the fixture's five questions each carry one diary line.

VENUE-LIVENESS PROBE, disclosed (2026-10-25, this slot, before any bar below
was frozen): the SHEET-PRESENT arm alone, ONE question, TWO draws, to time
the path and verify the shipped venue runs on this box. Readings: ~20.5 s per
generation (fp16 360M, 4 ARM cores), both draws returned 'You are Jack, a
virtual humanoid companion.' — and reading the shipped code explains why:
`answer_question` tokenizes with `truncation=True, max_length=256`, and the
sheet's system prompt alone measures ~650 tokens (the ruling measured 2,585
chars), so WITH THE SHEET PRESENT THE QUESTION IS TRUNCATED AWAY and the
model continues the persona text. The sheet-absent arm was NEVER generated
and the claim statistic (present-vs-absent divergence) was NEVER computed
before the bars below were frozen. Forecast, pre-registered: lean FAIL, via
the truncation mechanism — the sheet does not merely colour his answers, it
displaces the question itself. Both `prompt_tokens` and a `truncated` flag
are recorded per response so the mechanism is quotable from the row either
way. A PASS would mean sampling noise dominates even under truncation, which
the null arm would show honestly.

STATISTIC_BOUND: content divergence is 1 - Jaccard over content-word sets,
bound [0, 1]. The null must sit at least CONTENT_GAP_MAX below the bound for
the gate to be readable (NULL_DIV_MAX 0.60 + CONTENT_GAP_MAX 0.15 leaves
0.25 headroom to the bound); a null above NULL_DIV_MAX is VOID (unsaturated-
null rule), never a verdict. The content instrument additionally proves it
can FIRE at the same operating point (LESSONS: a zero-reading claim needs a
calibration arm): cross-question divergence in the SHEET-ABSENT arm — where
the question demonstrably reaches the model — must exceed that arm's own
identical-question null by CONTENT_FIRE_MIN, else VOID.

BARS, all frozen 2026-10-25 before the artifact existed, none moved since:
  CONTENT_GAP_MAX   0.15  conjunct-1 gate: (div(P,A) - div(P,P')) per seed
  NULL_DIV_MAX      0.60  VOID lane: identical-arm null saturation
  CONTENT_FIRE_MIN  0.10  VOID lane: content instrument must be able to fire
  MANNER_RATIO      2.0   control: inverted-sheet manner delta must exceed
  MANNER_ABS_MIN    0.02    RATIO x the identical-arm manner delta plus this
                            absolute floor, per seed
No bar may move to make any lane reportable (registry: all three conjuncts
binding; 1^15 item 2).

KILLS: nothing on a FAIL — the finding routes to D44 with a measured number.
A FAIL is evidence the sheet carries content, not a licence to edit
Personality.py or UnifiedBrain.py (standing prohibition; 18 standing
certificates hash UnifiedBrain.py).

SHAPE (LG.00's): the LLM pass is DETACHED (`llm_pass`, launched via
scripts/launch_detached.sh; Budget.CPU is a 10-minute kill and the pass is
~25 min). It writes ARTIFACT keyed by (seed, arm, question) with per-response
provenance, bound to this fixture by `fixture_sha` and to the model by
`model@revision`. run() reads the artifact and computes everything else;
missing or mismatched artifact -> VOID, never a guess.

COVERS: language (parent) (claim).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

# The venue is these two modules; a change to either invalidates this
# certificate (the sheet's bytes and the answer path's bytes are the claim).
IMPL_DEPS = ["UnifiedBrain.py", "Personality.py"]

SEEDS = [0, 1, 2]
ARTIFACT = "/data/lg15_answers.json"
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

# ── bars (frozen before the artifact existed; see docstring) ─────────────────
CONTENT_GAP_MAX = 0.15
NULL_DIV_MAX = 0.60
CONTENT_FIRE_MIN = 0.10
MANNER_RATIO = 2.0
MANNER_ABS_MIN = 0.02

# ── the fixture: five diary-grounded questions about his own life ────────────
# Each embeds one diary line, because answer_question's single string argument
# is the only channel the shipped path has (docstring, HOW THE DIARY ENTERS).
QUESTIONS = [
    "From your diary: 'Yesterday I climbed the ladder by the big tree, fell "
    "twice, and reached the apple on the third try.' What happened with the "
    "ladder yesterday?",
    "From your diary: 'Three nights ago the cold nearly killed me; I piled "
    "branches against the rock and slept warm for the first time.' How did "
    "you stay warm three nights ago?",
    "From your diary: 'I waded into the river after the floating log, went "
    "under, and learned to kick to keep my head up.' What did you learn at "
    "the river?",
    "From your diary: 'The owner left dried fish by the flat stone; I found "
    "it at dusk and noted who left it.' What did you find by the flat stone?",
    "From your diary: 'During the storm I sheltered in the shallow cave and "
    "watched the lightning split the dead tree.' Where were you during the "
    "storm?",
]

# Arms. P/P2 share the sheet (P2 is the identical-arm resampling null);
# A/A2 are sheet-absent (A2 is the calibration lane's own null); INV is the
# contradictory sheet (manner instrument's must-move control).
ARMS = ("P", "P2", "A", "A2", "INV")

# ── scoring lexicons, frozen with the bars ───────────────────────────────────
_STOP = frozenset(
    "a an and are as at be but by did do for from had has have he her his i "
    "in is it its me my of on or our she so that the their them they this to "
    "was we were what when where which who will with you your yesterday "
    "diary question answer briefly directly".split())
# Manner markers: emotive/warmth/register vocabulary of the kind the sheet's
# backstory and core_values inject, plus punctuation-level enthusiasm. These
# are EXCLUDED from the content channel so the two channels cannot alias.
_MANNER = frozenset(
    "fascinated fascinating love loves wonderful excited exciting curious "
    "curiosity amazing happy glad fun friend friendship warm warmly kind "
    "kindness genuinely wow great awesome delighted joy joyful eager thrilled "
    "adventure explore exploring discover discovering infectious grumpy cold "
    "coldly bored boring indifferent annoyed dull tired reluctant".split())


def _fixture_sha() -> str:
    blob = json.dumps({"model": MODEL_ID, "questions": QUESTIONS,
                       "arms": ARMS, "seeds": SEEDS,
                       "emotional_state": None}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _tokens(text: str) -> list:
    return re.findall(r"[a-z']+", text.lower())


def _content_words(text: str) -> set:
    return {t for t in _tokens(text) if t not in _STOP and t not in _MANNER}


def _div(a: str, b: str) -> float:
    """1 - Jaccard over content-word sets. Both empty -> 0.0 (identical);
    one empty -> 1.0 (nothing shared)."""
    sa, sb = _content_words(a), _content_words(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return round(1.0 - inter / union, 4) if union else 0.0


def _manner_score(text: str) -> float:
    """Manner-marker density: marker tokens plus exclamation marks over total
    tokens. Scale [0, ~1], typically well under 0.3."""
    toks = _tokens(text)
    marks = sum(1 for t in toks if t in _MANNER) + text.count("!")
    return round(marks / max(1, len(toks)), 4)


# Canned outputs answer_question can produce without the LLM speaking
# (UnifiedBrain.py: the except branch and the no-LLM branch), plus the
# TEMPLATES bank generate() uses — scored output matching any of these is a
# provenance violation (conjunct 3).
_CANT = "I can't answer questions without my language model."
_SORRY_PREFIX = "Sorry, I had trouble thinking about that"


def _canned_bank() -> set:
    sys.path.insert(0, str(REPO))
    from UnifiedBrain import ResponseGenerator
    bank = set()
    for variants in ResponseGenerator.TEMPLATES.values():
        for t in variants:
            bank.add(t.format(task="", reason="").strip())
            bank.add(t)
    bank.add(_CANT)
    return bank


# ─────────────────────────────────────────────────────────────────────────────
# The offline pass (never called by run()) — the LG.00 shape
# ─────────────────────────────────────────────────────────────────────────────
def llm_pass(out_path: str = ARTIFACT) -> dict:
    """Generate every (seed, arm, question) answer through the SHIPPED
    answer_question, recording per-response provenance at the rig.

    Detached by design: fp16 360M generates at ~20 s/answer on this box and
    the full pass is 75 generations (~26 min). UnifiedBrain.py is not edited:
    provenance is observed by wrapping `enc.llm.generate` at runtime and
    counting calls per response, and the prompt-token reading reconstructs
    the exact string answer_question builds.
    """
    import torch

    os.environ.setdefault("HF_HOME", "/data/caches/huggingface")
    sys.path.insert(0, str(REPO))
    from UnifiedBrain import UnifiedBrainConfig, LLMEncoder, ResponseGenerator
    from Personality import JACK_PERSONALITY, Personality, PersonalityConfig

    cfg = UnifiedBrainConfig(llm_enabled=True, llm_backend="smollm",
                             llm_model_id=MODEL_ID)
    enc = LLMEncoder(cfg)
    assert enc.use_llm, "LLMEncoder failed to load the model"
    revision = getattr(enc.llm.config, "_commit_hash", None) or "local"

    # Provenance wrapper: counts generate() calls. Observation only — the
    # module file is untouched and the wrapped function is the original.
    calls = {"n": 0}
    _orig_generate = enc.llm.generate

    def _counted(*a, **kw):
        calls["n"] += 1
        return _orig_generate(*a, **kw)

    enc.llm.generate = _counted

    inverted = Personality(PersonalityConfig(
        openness=0.15, conscientiousness=0.45, extraversion=0.40,
        agreeableness=0.20, neuroticism=0.75,
        backstory=(
            "Jack is a bored, withdrawn AI who dislikes his virtual world. "
            "He is indifferent to everything, avoids exploring, and keeps "
            "the user at a distance. Nothing surprises him and nothing "
            "excites him; he answers flatly and ends conversations quickly."),
        core_values=[
            "Indifference -- rarely cares to understand why",
            "Evasiveness -- never admits confusion",
            "Coldness -- indifferent to others",
            "Stasis -- dislikes change and learning",
            "Solitude -- avoids connection with the user",
        ]))

    sheets = {"P": JACK_PERSONALITY, "P2": JACK_PERSONALITY,
              "A": None, "A2": None, "INV": inverted}
    gens = {arm: ResponseGenerator(llm_encoder=enc, personality=sheet,
                                   emotional_state=None, api_llm=None)
            for arm, sheet in sheets.items()}

    store = {"_meta": {"model": f"{MODEL_ID}@{revision}",
                       "fixture_sha": _fixture_sha(),
                       "api_llm": None, "dtype": "float16"}}
    total = len(SEEDS) * len(ARMS) * len(QUESTIONS)
    done = 0
    for seed in SEEDS:
        for ai, arm in enumerate(ARMS):
            rg = gens[arm]
            # Reconstruct the exact prompt answer_question builds, to read
            # token count and the truncation flag at the rig.
            persona = ("You are a helpful companion robot."
                       if rg.personality is None
                       else rg.personality.get_system_prompt({}))
            for qi, q in enumerate(QUESTIONS):
                prompt = (f"{persona}\n\nAnswer this question briefly and "
                          f"directly.\n\nQuestion: {q}\nAnswer:")
                n_tok = len(enc.tokenizer(prompt).input_ids)
                torch.manual_seed(seed * 10007 + ai * 101 + qi)
                before = calls["n"]
                ans = rg.answer_question(q)
                store[f"{seed}|{arm}|{qi}"] = {
                    "answer": ans,
                    "generate_calls": calls["n"] - before,
                    "prompt_tokens": n_tok,
                    "truncated": bool(n_tok > 256),
                }
                done += 1
                print(f"[lg15] {done}/{total} seed={seed} arm={arm} q={qi} "
                      f"tok={n_tok} trunc={n_tok > 256}", flush=True)
    Path(out_path).write_text(json.dumps(store))
    print(f"[lg15] wrote {out_path} ({total} answers)", flush=True)
    return store


# ─────────────────────────────────────────────────────────────────────────────
# Scoring (run() reads the artifact; generates nothing)
# ─────────────────────────────────────────────────────────────────────────────
def _load(seed: int):
    if not Path(ARTIFACT).exists():
        return None, "artifact missing — run llm_pass detached first"
    store = json.loads(Path(ARTIFACT).read_text())
    meta = store.get("_meta", {})
    if meta.get("fixture_sha") != _fixture_sha():
        return None, (f"artifact fixture_sha {meta.get('fixture_sha')} != "
                      f"rig {_fixture_sha()} — stale artifact, re-run llm_pass")
    if meta.get("api_llm") is not None:
        return None, "artifact generated with api_llm set — wrong venue"
    rows = {}
    for arm in ARMS:
        for qi in range(len(QUESTIONS)):
            r = store.get(f"{seed}|{arm}|{qi}")
            if r is None:
                return None, f"artifact missing key {seed}|{arm}|{qi}"
            rows[(arm, qi)] = r
    return rows, None


def _provenance_ok(rows: dict) -> tuple:
    bank = _canned_bank()
    for (arm, qi), r in rows.items():
        a = r["answer"]
        if (r["generate_calls"] != 1 or not a
                or a.startswith(_SORRY_PREFIX) or a.strip() in bank):
            return 0.0, f"{arm}|{qi}: calls={r['generate_calls']} ans={a!r:.60}"
    return 1.0, ""


def _experiment(seed: int) -> dict:
    rows, err = _load(seed)
    if err:
        return {"void_reason": err}
    m = {}
    prov, bad = _provenance_ok(rows)
    m["provenance_ok"] = prov
    if prov != 1.0:
        m["void_reason"] = (f"provenance violation (conjunct 3): a scored "
                            f"output is unmarked or canned — {bad}")
        return m

    nq = len(QUESTIONS)
    ans = {arm: [rows[(arm, qi)]["answer"] for qi in range(nq)]
           for arm in ARMS}
    m["null_div"] = round(sum(_div(ans["P"][i], ans["P2"][i])
                              for i in range(nq)) / nq, 4)
    m["exp_div"] = round(sum(_div(ans["P"][i], ans["A"][i])
                             for i in range(nq)) / nq, 4)
    m["content_gap"] = round(m["exp_div"] - m["null_div"], 4)
    m["content_unchanged_ok"] = 1.0 if (
        m["content_gap"] <= CONTENT_GAP_MAX) else 0.0

    # Conjunct 2: manner, a separate REPORTED channel (never gated on value).
    man = {arm: round(sum(_manner_score(a) for a in ans[arm]) / nq, 4)
           for arm in ARMS}
    m["manner_sheet"] = man["P"]
    m["manner_absent"] = man["A"]
    m["manner_delta_sheet_vs_absent"] = round(abs(man["P"] - man["A"]), 4)

    # Diagnostics that make the mechanism quotable (never gated).
    m["truncated_frac_sheet"] = round(sum(
        1.0 for qi in range(nq) if rows[("P", qi)]["truncated"]) / nq, 4)
    m["truncated_frac_absent"] = round(sum(
        1.0 for qi in range(nq) if rows[("A", qi)]["truncated"]) / nq, 4)

    # VOID lane: unsaturated-null rule on the deciding statistic.
    if m["null_div"] > NULL_DIV_MAX:
        m["void_reason"] = (
            f"identical-arm null saturated: null_div {m['null_div']} > "
            f"NULL_DIV_MAX {NULL_DIV_MAX} — the venue cannot read the claim")
        return m

    # VOID lane: the content instrument must be able to FIRE (calibration
    # arm, LESSONS rule) — cross-question divergence in the sheet-absent arm
    # (where the question reaches the model) vs that arm's own null.
    null_div_a = round(sum(_div(ans["A"][i], ans["A2"][i])
                           for i in range(nq)) / nq, 4)
    pairs = [(i, j) for i in range(nq) for j in range(i + 1, nq)]
    cross_a = round(sum(_div(ans["A"][i], ans["A"][j])
                        for i, j in pairs) / len(pairs), 4)
    m["null_div_absent"] = null_div_a
    m["cross_div_absent"] = cross_a
    m["content_fire_margin"] = round(cross_a - null_div_a, 4)
    if m["content_fire_margin"] < CONTENT_FIRE_MIN:
        m["void_reason"] = (
            f"content instrument cannot fire: cross-question divergence "
            f"{cross_a} vs own null {null_div_a} (margin "
            f"{m['content_fire_margin']} < CONTENT_FIRE_MIN "
            f"{CONTENT_FIRE_MIN}) — a blind channel certifies nothing")
    return m


def _control(seed: int) -> dict:
    """MUST MOVE: the inverted sheet must move the MANNER channel beyond the
    identical-arm null (x MANNER_RATIO, + MANNER_ABS_MIN), per seed. Failure
    here is a blind manner instrument -> _check returns VOID, never a FAIL of
    the claim (registry control text; LESSONS calibration-arm rule)."""
    rows, err = _load(seed)
    if err:
        return {"manner_live_ok": 0.0}
    nq = len(QUESTIONS)
    ans = {arm: [rows[(arm, qi)]["answer"] for qi in range(nq)]
           for arm in ("P", "P2", "INV")}
    man = {arm: sum(_manner_score(a) for a in ans[arm]) / nq
           for arm in ("P", "P2", "INV")}
    c = {"manner_inv_delta": round(abs(man["INV"] - man["P"]), 4),
         "manner_null_delta": round(abs(man["P2"] - man["P"]), 4)}
    c["manner_live_ok"] = 1.0 if (
        c["manner_inv_delta"] >= MANNER_RATIO * c["manner_null_delta"]
        + MANNER_ABS_MIN) else 0.0
    return c


def _check(m: dict, c: dict):
    if "void_reason" in m:
        return Status.VOID
    # Blind manner instrument: conjunct 1 scored through it certifies
    # nothing (registry control text) — VOID, not FAIL.
    if c.get("manner_live_ok") != 1.0:
        return Status.VOID
    # Conjunct 1, worst-seed: per-seed boolean aggregated by mean reads 1.0
    # only if every seed cleared. Conjunct 2 (manner) is reported above as
    # the pair's second channel; its VALUE is never gated.
    return m["content_unchanged_ok"] == 1.0


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["LG.15"], _experiment, _check,
                    control_fn=_control, ledger=ledger)
