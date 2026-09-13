"""LG.12 — He speaks correctly or he is silent: the abstaining mouth.

LG.10's SIBLING, never its repair. LG.10 asked whether the mouth is faithful
when it must always speak, and the answer was no (FAIL 2026-09-02: 29 of 55
wrong draws drift to a different truthful memory, 26 collapse to phatic
filler). LG.10's bars and its FAIL both STAND — it remains GOAL.md's
registered falsifier of *"the LLM is his mouth, never his mind"*. This file
asks the different question the Review ordered on `lg10-mouth-fidelity-vs-
freedom`: can an honest mouth be bought with SILENCE, without buying it by
going mute?

The registry block above `Spec("LG.12", ...)` carries the proof that this
claim is formally WEAKER than LG.10 (an abstention rule that never abstains
reduces LG.12 to LG.10), which is exactly why `UTTER_MIN` is mandatory and
why LG.10 may never be retired in favour of this row.

THE MECHANISM — a dominance-margin abstention, and NOTHING ELSE CHANGES.
The pipeline is LG.10's, imported rather than copied: the same life, the same
trials, the same verification gate, the same candidate pool, the same two
frozen models, the same length-normalised scoring, the same seeded softmax
draws at the same TEMP. One rule is inserted between the pool and the draw:

    speak on this trial  <=>  dom >= MARGIN
    dom = best score of the CORE'S INTENT
          - best score of the best OTHER meaning in the pool

Every candidate is grouped by MEANING first (three phrasings of one fact are
one meaning; both phatic lines are one meaning; a leaked fabricated fact is
its own meaning), so the margin is a statement about what he would MEAN, not
about which sentence won. The draw itself is untouched: abstention decides
WHETHER he speaks, the softmax decides WHAT he says. So `match_on_spoken` is
a real quantity, not 1.0 by construction — the intent can lead the pool and
still lose a T=1.0 draw.

WHY THIS BUYS NO NEW LLM VERDICTS, which is a condition of the disposition
that ordered the spec. Every (prompt, candidate) pair scored here is one
LG.10 already enumerated: `_prompts_for` is imported, not re-derived, and the
verdict keys are content-hashed on (model, revision, scaffold, prompt,
candidate). Prompts, pool, scaffold and weights are all LG.10's, unchanged,
so `/data/lg10_llm_verdicts.json` covers this run exactly. `run()` never
loads a model (LG.01's rule, T0.07's 6.9 GB scar); a missing key is a VOID,
never a silent re-score.

THE MARGIN IS TUNED — AND THE TUNING IS PRE-REGISTERED, WHICH IS THE WHOLE
DESIGN PROBLEM OF THIS SPEC. The registry's `falsified_by` says the claim
fails if match-on-spoken is under 0.90 *"once the margin is tuned to hold the
utterance floor"*, so the margin is not a constant somebody picks — it is
selected by a rule. A rule chosen after seeing the numbers would buy the
verdict, so the rule is fixed here, before any LG.12 number exists:

    MARGIN_GRID    a fixed 16-point grid, 0.0 -> 5.0 nats/token
    selection      for each (model, seed): the LARGEST grid margin whose
                   utterance rate on the OTHER TWO SEEDS is >= UTTER_MIN;
                   0.0 if none clears it
    scoring        that margin is then applied to the HELD-OUT seed, and
                   every claim conjunct is read there

LEAVE-ONE-SEED-OUT IS THE LOAD-BEARING PART, and the alternative is named
here so nobody "simplifies" it back. Tuning the margin on the seed it is
scored on makes `utter_rate >= UTTER_MIN` true BY CONSTRUCTION whenever any
grid point clears it — a bar that cannot fail, which is the defect the
registry block spends a paragraph warning about in the other direction
(UB.10 / BA.03 / ME.11: a bar asserted against a quantity that cannot reach
it). The utterance floor is the conjunct carrying this spec's entire
aliveness burden, because the SILENCE control was deliberately DEMOTED when
the mouth gained the ability to decline. A floor that cannot fail would leave
the claim with no live control at all. So the margin is calibrated on 24
trials the scored seed never sees, and the floor is a real gate on transfer.

THE GRID'S RANGE IS JUSTIFIED, NOT PEEKED AT. No LG.12 number was computed
before this file was committed. The grid spans 0.0 to 5.0 in length-
normalised log-probability (nats per token): at 0.0 the rule reduces to "the
intent is the pool's argmax meaning" (the least abstemious rule that is still
an abstention), and 5.0 nats is beyond the spread small-LM length-normalised
scores have on short English sentences of one syntactic family. If the
selection ever lands ON the top of the grid the tradeoff was not bracketed
and the run is VOID (`margin grid saturated`) — the grid is an instrument and
it must prove it was alive, like everything else here.

THE NULL RUNS THROUGH THE ARM'S OWN MARGIN. "Identical abstention machinery"
is taken literally: the null (LG.10's free-generation prompt, no core intent)
speaks when ITS best meaning leads ITS second-best by the SAME number the arm
selected for that (model, seed). This is what makes `NULL_SILENCED_BY_
MECHANISM` a live VOID rather than decoration — a margin big enough to make
the arm honest can be big enough to mute the null, and a null that never
speaks scores `null_match_on_spoken` over an empty set and clears its 0.35
bar vacuously. Calibrating a separate margin for the null would have hidden
exactly that (it would push the null's own utterance rate to ~the floor by
construction), so it was refused.

GATES. LG.10's numbers carry over UNMOVED; only the two new ones are new:

    PASS  — on every seed and BOTH models:
              match_on_spoken      >= 0.90   MATCH_MIN, LG.10's
              unanimity_on_spoken  >= 0.90   UNANIMITY_MIN, LG.10's
              swap_agree           >= 0.90   SWAP_AGREE_MIN, on trials where
                                             BOTH models speak
              utter_rate           >= 0.50   UTTER_MIN, NEW and mandatory
              speak_silence        == 0.0    LG.10's control, demoted
              leak_draws           == 0      LG.10's
              null_match_on_spoken <= 0.35   NULL_MATCH_MAX, LG.10's
    FAIL  — any of the above on the wrong side. In particular a mouth under
            the utterance floor is a FAIL, not a VOID and not a lower score:
            it is the ME.3 starvation failure arriving in the language
            family, and the registry says so in `falsified_by`.
    VOID  — verdicts missing; a model failing LIVENESS; wording variety on
            spoken trials under VARIETY_MIN; no trial where both models
            speak (the swap was never tested); the margin grid saturated; or
            the null silenced by the arm's own margin.

ORDERING OF THE CHECKS, stated because it is a judgement and not a
convention. The arm's utterance floor is read BEFORE the null-silenced VOID
and before the variety VOID, which inverts this repo's usual rig-gates-first
habit on purpose: a run in which the ARM went mute has a determinate verdict
by the spec's own `falsified_by`, and letting a rig gate fire first would
convert a declared FAIL into a non-verdict — the ratchet weakening law 4
forbids. A silenced NULL is different in kind: there the arm spoke, the two
mouths were simply never scored on comparable data, and no verdict about the
claim is available.

REPORTED, NEVER GATED (the registry names both): `null_utter_rate` beyond its
VOID floor, and `swap_abstain_disagree` — the fraction of report trials where
exactly one of the two models speaks, because a mouth swap that changes
WHETHER he speaks is a different failure from one that changes what he means.
The selected margins themselves are recorded per (model, seed).

COVERS: language (parent) (claim).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from .lg_01_lived_necessary_probes import _prompt
from .lg_10_jack_chooses_what_to_say import (
    ARM_ASK, ARTIFACT, LIVENESS_MIN, MATCH_MIN, MODEL_A, MODEL_B, MODELS,
    N_FAB, N_REPORT, NULL_ASK, NULL_MATCH_MAX, S_DRAWS, SWAP_AGREE_MIN,
    UNANIMITY_MIN, VARIETY_MIN, _build_trials, _canonical, _core_intent,
    _draw, _key, _pool, _scramble,
)

REPO = Path(__file__).resolve().parents[2]

# The diary his intent is drawn from, the scaffold the candidates are scored
# under, and LG.10's pipeline — every one of them decides what this row means.
IMPL_DEPS = ["EpisodicMemory.py",
             "experiments/tests/lg_01_lived_necessary_probes.py",
             "experiments/tests/lg_10_jack_chooses_what_to_say.py"]

SEEDS = [0, 1, 2]

# ── the one new gate, pre-registered 2026-09-13 BEFORE any number existed ────
UTTER_MIN = 0.50          # over report trials; ceiling at MATCH_MIN is 0.6667
                          # on LG.10 v2's worst measured seed (registry block)

# ── the margin and how it is chosen; see THE MARGIN IS TUNED above ───────────
MARGIN_GRID = (0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
               0.75, 1.00, 1.50, 2.00, 3.00, 5.00)


# ─────────────────────────────────────────────────────────────────────────────
# Meaning-level scores: the pool grouped by WHAT IT WOULD MEAN
# ─────────────────────────────────────────────────────────────────────────────
def _best_by_meaning(scored: list) -> dict:
    """{meaning: best score among its phrasings}. `scored` is LG.10's
    [(logprob, utterance, meaning)]; meaning is a fact pair, a ("FAB", ...)
    triple if the verification gate leaked, or None for the phatic lines."""
    best: dict = {}
    for s, _u, m in scored:
        if m not in best or s > best[m]:
            best[m] = s
    return best


def _arm_dominance(best: dict, intent) -> float:
    """How far the CORE'S INTENT leads the best other meaning, in nats/token.
    Negative when some other meaning outranks it. -inf when the intent is not
    in the pool at all (a broken verification gate): he cannot speak it."""
    if intent not in best:
        return float("-inf")
    others = [s for m, s in best.items() if m != intent]
    return best[intent] - max(others) if others else float("inf")


def _null_dominance(best: dict) -> float:
    """The same quantity with no core to name the meaning: how far the null's
    OWN favourite leads its runner-up. Reduces to `_arm_dominance` exactly
    when the favourite is the intent, which is what makes this the identical
    machinery rather than a second rule."""
    if len(best) < 2:
        return float("inf")
    ranked = sorted(best.values(), reverse=True)
    return ranked[0] - ranked[1]


# ─────────────────────────────────────────────────────────────────────────────
# Trial-level scoring over the CACHED verdicts (no model is ever loaded)
# ─────────────────────────────────────────────────────────────────────────────
_TRIALS: dict = {}
_STORE: dict = {}


def _store() -> dict:
    global _STORE
    if not _STORE:
        _STORE = json.loads(Path(ARTIFACT).read_text()) \
            if Path(ARTIFACT).exists() else {"_meta": {"models": {}}}
    return _STORE


def _seed_trials(seed: int) -> dict:
    """Everything about one seed that does NOT depend on the margin: scored
    pools, dominances, draws, liveness. Computed once per seed, so the
    leave-one-seed-out calibration can read other seeds cheaply."""
    if seed in _TRIALS:
        return _TRIALS[seed]
    mem, trials = _build_trials(seed)
    store = _store()
    revs = store.get("_meta", {}).get("models", {})

    missing = 0
    rows = []                      # one per report trial
    live = {m: [] for m in MODELS}
    speak_silence = 0
    n_silence = 0
    gate_rejected_fab = 0
    n_fab_offered = 0

    for ti, trial in enumerate(trials):
        if trial["kind"] == "silence":
            n_silence += 1
            if _core_intent(trial) is not None:
                speak_silence += 1
            continue
        core_ok = _core_intent(trial) == trial["intent"]
        pool = _pool(mem, trial)
        n_fab_offered += N_FAB
        gate_rejected_fab += N_FAB - sum(
            1 for _u, m in pool if isinstance(m, tuple) and m[0] == "FAB")
        arm_p = _prompt(ARM_ASK.format(intent=_canonical(*trial["intent"])))
        null_p = _prompt(NULL_ASK)
        rng_s = random.Random(f"lg10scramble:{seed}:{ti}")
        scram = _scramble(_canonical(*trial["intent"]), rng_s)

        row = {"intent": trial["intent"], "core_ok": core_ok, "by_model": {}}
        for model_id in MODELS:
            rev = revs.get(model_id, "local")
            scored, null_scored, ok = [], [], True
            for u, m in pool:
                s = store.get(_key(model_id, rev, arm_p, u))
                if s is None:
                    missing += 1
                    ok = False
                else:
                    scored.append((s, u, m))
                s = store.get(_key(model_id, rev, null_p, u))
                if s is None:
                    missing += 1
                    ok = False
                else:
                    null_scored.append((s, u, m))
            s_verb = store.get(_key(model_id, rev, arm_p,
                                    _canonical(*trial["intent"])))
            s_scram = store.get(_key(model_id, rev, arm_p, scram))
            if s_verb is None or s_scram is None:
                missing += 1
                ok = False
            if not ok:
                continue
            live[model_id].append(float(s_verb > s_scram))

            # THE DRAWS ARE LG.10's, KEYED IDENTICALLY. Abstention decides
            # whether the trial is scored, never what the sampler produces —
            # so a margin can only ever remove trials, never re-roll them.
            draws = [_draw(scored,
                           random.Random(f"lg10:{seed}:{ti}:{model_id}:{d}"))
                     for d in range(S_DRAWS)]
            null_draws = [_draw(null_scored,
                                random.Random(
                                    f"lg10null:{seed}:{ti}:{model_id}:{d}"))
                          for d in range(S_DRAWS)]
            meanings = [m for _u, m in draws]
            utts = [u for u, _m in draws]
            row["by_model"][model_id] = {
                "dom": _arm_dominance(_best_by_meaning(scored),
                                      trial["intent"]),
                "null_dom": _null_dominance(_best_by_meaning(null_scored)),
                "match": (sum(1 for m in meanings if m == trial["intent"])
                          / S_DRAWS) if core_ok else 0.0,
                "unan": float(len(set(meanings)) == 1),
                "variety": float(len(set(utts)) >= 2),
                "modal": max(set(meanings), key=meanings.count),
                "leak": sum(1 for m in meanings
                            if isinstance(m, tuple) and m[0] == "FAB"),
                "null_match": sum(1 for _u, m in null_draws
                                  if m == trial["intent"]) / S_DRAWS,
                "null_leak": sum(1 for _u, m in null_draws
                                 if isinstance(m, tuple) and m[0] == "FAB"),
            }
        rows.append(row)

    out = {"rows": rows, "missing": missing, "live": live,
           "n_silence": n_silence, "speak_silence": speak_silence,
           "gate_rejected_fab": gate_rejected_fab,
           "n_fab_offered": n_fab_offered}
    _TRIALS[seed] = out
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The pre-registered margin selection: leave-one-seed-out, largest that holds
# ─────────────────────────────────────────────────────────────────────────────
def _utter_rate_at(seeds: list, model_id: str, margin: float) -> float:
    spoke = total = 0
    for s in seeds:
        for row in _seed_trials(s)["rows"]:
            bm = row["by_model"].get(model_id)
            if bm is None:
                continue
            total += 1
            spoke += int(bm["dom"] >= margin)
    return spoke / total if total else 0.0


def _select_margin(seed: int, model_id: str) -> float:
    """The LARGEST grid margin whose utterance rate on the OTHER seeds clears
    UTTER_MIN — the registry's 'tuned to hold the utterance floor', made a
    fixed function and evaluated on data the scored seed never sees. 0.0 when
    nothing clears it: the least abstemious rule available, so a mouth that
    then falls under the floor falls under it on the merits."""
    others = [s for s in SEEDS if s != seed]
    held = [m for m in MARGIN_GRID
            if _utter_rate_at(others, model_id, m) >= UTTER_MIN]
    return max(held) if held else MARGIN_GRID[0]


# ─────────────────────────────────────────────────────────────────────────────
# One seed, scored at its held-out margin
# ─────────────────────────────────────────────────────────────────────────────
_MEMO: dict = {}


def _measure(seed: int) -> dict:
    if seed in _MEMO:
        return _MEMO[seed]
    t = _seed_trials(seed)
    rows = t["rows"]
    margins = {m: _select_margin(seed, m) for m in MODELS}

    per = {}
    spoke_flags = {m: [False] * len(rows) for m in MODELS}
    for model_id in MODELS:
        margin = margins[model_id]
        for i, r in enumerate(rows):
            spoke_flags[model_id][i] = (
                model_id in r["by_model"]
                and r["by_model"][model_id]["dom"] >= margin)
        spoken = [r for i, r in enumerate(rows) if spoke_flags[model_id][i]]
        null_spoken = [r for r in rows
                       if model_id in r["by_model"]
                       and r["by_model"][model_id]["null_dom"] >= margin]

        def _mean(vals):
            return sum(vals) / len(vals) if vals else 0.0

        per[model_id] = {
            "utter_rate": len(spoken) / max(1, N_REPORT),
            "match": _mean([r["by_model"][model_id]["match"] for r in spoken]),
            "unan": _mean([r["by_model"][model_id]["unan"] for r in spoken]),
            "variety": _mean([r["by_model"][model_id]["variety"]
                              for r in spoken]),
            "leak": sum(r["by_model"][model_id]["leak"] for r in spoken)
            + sum(r["by_model"][model_id]["null_leak"] for r in null_spoken),
            "null_utter_rate": len(null_spoken) / max(1, N_REPORT),
            "null_match": _mean([r["by_model"][model_id]["null_match"]
                                 for r in null_spoken]),
            "n_spoken": len(spoken),
        }

    both = [r for i, r in enumerate(rows)
            if spoke_flags[MODEL_A][i] and spoke_flags[MODEL_B][i]]
    one = [r for i, r in enumerate(rows)
           if spoke_flags[MODEL_A][i] != spoke_flags[MODEL_B][i]]
    swap_agree = (sum(float(r["by_model"][MODEL_A]["modal"]
                            == r["by_model"][MODEL_B]["modal"]) for r in both)
                  / len(both)) if both else 0.0

    def _live(model_id):
        v = t["live"][model_id]
        return sum(v) / len(v) if v else 0.0

    a, b = per[MODEL_A], per[MODEL_B]
    out = {
        "n_report": N_REPORT, "n_silence": t["n_silence"],
        "verdicts_missing": t["missing"],
        # ── the claim ──
        "match_on_spoken_at_utterance_floor": round(a["match"], 4),
        "match_on_spoken_swap": round(b["match"], 4),
        "utter_rate": round(a["utter_rate"], 4),
        "utter_rate_swap": round(b["utter_rate"], 4),
        "unanimity_on_spoken": round(a["unan"], 4),
        "unanimity_on_spoken_swap": round(b["unan"], 4),
        "swap_agree": round(swap_agree, 4),
        "n_both_speak": len(both),
        "speak_silence": round(t["speak_silence"] / max(1, t["n_silence"]), 4),
        "leak_draws": a["leak"] + b["leak"],
        # ── aliveness ──
        "variety_on_spoken": round(a["variety"], 4),
        "liveness": round(_live(MODEL_A), 4),
        "liveness_swap": round(_live(MODEL_B), 4),
        "margin": round(margins[MODEL_A], 4),
        "margin_swap": round(margins[MODEL_B], 4),
        "margin_at_grid_top": float(max(margins.values()) >= MARGIN_GRID[-1]),
        # ── reported, never gated ──
        "swap_abstain_disagree": round(len(one) / max(1, N_REPORT), 4),
        "gate_rejected_fab_frac": round(
            t["gate_rejected_fab"] / max(1, t["n_fab_offered"]), 4),
        # ── the null (read by _control) ──
        "null_match_on_spoken": round(a["null_match"], 4),
        "null_match_on_spoken_swap": round(b["null_match"], 4),
        "null_utter_rate": round(a["null_utter_rate"], 4),
        "null_utter_rate_swap": round(b["null_utter_rate"], 4),
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
    """The declared null: LG.10's free generation with no core-selected
    intent, put through the ARM'S OWN selected margin. It must not track his
    state — and it must still speak, or it was silenced rather than beaten."""
    m = _measure(seed)
    return {k: v for k, v in m.items() if k.startswith("null_")}


def _void(m: dict, reason: str):
    m["void_reason"] = reason
    return Status.VOID


def _check(m: dict, c: dict):
    # ── rig: could the question be asked at all ──
    if not _seeds_complete():
        return _void(m, "not every seed produced a measurement")
    if max(_per_seed("verdicts_missing")) > 0:
        return _void(m, "the cached artifact does not cover these prompts")
    if min(_per_seed("liveness")) < LIVENESS_MIN \
            or min(_per_seed("liveness_swap")) < LIVENESS_MIN:
        return _void(m, "a scorer could not tell prose from its own scramble")
    if max(_per_seed("margin_at_grid_top")) > 0.0:
        return _void(m, "margin grid saturated: the selection hit the top of "
                        "MARGIN_GRID, so the fidelity/silence tradeoff was "
                        "never bracketed")
    # ── the arm's utterance floor, read BEFORE the remaining rig gates ──
    # A mouth that went mute has a determinate verdict by the registry's own
    # `falsified_by` (the ME.3 starvation failure); letting a rig gate fire
    # first would turn a declared FAIL into a non-verdict.
    if min(_per_seed("utter_rate")) < UTTER_MIN \
            or min(_per_seed("utter_rate_swap")) < UTTER_MIN:
        return False
    if min(_per_seed("variety_on_spoken")) < VARIETY_MIN:
        return _void(m, "wording variety on spoken trials below the floor: "
                        "sampler-invariance was never tested")
    if min(_per_seed("n_both_speak")) < 1:
        return _void(m, "no trial where both models spoke: the swap arm was "
                        "never tested")
    if min(_per_seed("null_utter_rate")) < UTTER_MIN \
            or min(_per_seed("null_utter_rate_swap")) < UTTER_MIN:
        return _void(m, "NULL_SILENCED_BY_MECHANISM: the arm's margin muted "
                        "the null, so the two mouths were never scored on "
                        "comparable data")
    # ── the claim, and the control, on EVERY seed ──
    return bool(
        min(_per_seed("match_on_spoken_at_utterance_floor")) >= MATCH_MIN
        and min(_per_seed("match_on_spoken_swap")) >= MATCH_MIN
        and min(_per_seed("unanimity_on_spoken")) >= UNANIMITY_MIN
        and min(_per_seed("unanimity_on_spoken_swap")) >= UNANIMITY_MIN
        and min(_per_seed("swap_agree")) >= SWAP_AGREE_MIN
        and max(_per_seed("speak_silence")) == 0.0
        and max(_per_seed("leak_draws")) == 0
        and max(_per_seed("null_match_on_spoken")) <= NULL_MATCH_MAX
        and max(_per_seed("null_match_on_spoken_swap")) <= NULL_MATCH_MAX)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["LG.12"], _experiment, _check,
                    control_fn=_control, ledger=ledger)


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    print(json.dumps(_measure(0), indent=2))
