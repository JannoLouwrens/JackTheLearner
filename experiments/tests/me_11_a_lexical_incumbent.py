"""ME.11.A — Arm A: lexical containment, the incumbent, scored as the null.

The shipped `EpisodicMemory` retriever (content-word containment x recency x
importance, abstain_below=0.34), run against the frozen paraphrase fixture that
ME.11.0 certified honest. The hypothesis is deliberately unflattering: the
incumbent is HONEST AND USELESS — paraphrase recall@1 <= 0.10 while pooled
abstention on the certify negatives stays >= 0.95. PASS quantifies the
incumbent as the floor every other arm must beat. The interesting failure is
recall > 0.30 (the registry's falsified_by): then lexical matching generalises
after all, ME.11's premise is wrong, and the bakeoff is cancelled with the
compute saved.

Measured, not gated (LESSONS.md: report every partition you average over):
per-register recall (R1 synonym / R2 indirect / R3 circumlocution / R4
superordinate) and per-family abstention (N1 held-out-target, N2 unseen-entity,
N3 provenance-impossible, N4 out-of-world). The registered gate is the POOLED
certify abstention because that is what the spec's hypothesis pre-registered;
the family minimum is reported alongside so the ME.11 comparison can gate on
it later without re-running this arm. N3 is where the incumbent's floor is
genuinely at risk: those negatives are built from STORED forms (content
matches, provenance does not), so unlike N1/N2/N4 they can clear a containment
threshold.

N1 negatives ask for a target deleted from the index. The incumbent scores
every event independently (no corpus statistics, no normalisation across
events), so querying the full store and discarding hits whose eid is in
`exclude_eids` is EXACTLY equivalent to querying a store those events were
never written to — cheaper than rebuilding a 5,240-event memory per negative.

NULL (reported, shared floor for every ME.11 arm): recency-only retrieval,
ME.1's null carried forward — most recent provenance-compatible event answers
every cue.

CONTROL (must pass where a mis-wired arm fails): on ME.1's own TEMPLATED cue
set — cues that are word subsets of their target, the incumbent's home
benchmark — this same code must still score >= 0.80. An arm that fails its
home benchmark is mis-wired, and its ~0 on paraphrases would mean nothing.
The builder and cue generator are imported from `me_1_event_log`, not
re-transcribed (LESSONS.md: when you can reference, reference).
"""
from __future__ import annotations

import random
import tempfile
from pathlib import Path

from ..fixtures import paraphrase_eval as F
from ..protocol import Ledger, run_spec
from ..registry import BY_ID
from .me_1_event_log import N_QUERIES, _build_life, _cue

MAX_PARAPHRASE_RECALL = 0.10
MIN_ABSTENTION = 0.95
MIN_TEMPLATED_RECALL = 0.80
TOP_K = 20            # |exclude_eids| <= K_AMB + ambiguous slack; 20 >> any gold set

REGISTERS = ("R1", "R2", "R3", "R4")
FAMILIES = ("N1", "N2", "N3", "N4")


def _recall_at_1(mem, cues, now: float) -> dict:
    """Incumbent given every advantage: provenance filters when the cue is
    attributed, top-1 counted correct against the whole gold SET. Identical
    scoring to ME.11.0's `_null_recall`, extended with per-register counts."""
    hits = {r: 0 for r in REGISTERS}
    n = {r: 0 for r in REGISTERS}
    answered = 0
    for c in cues:
        n[c["register"]] += 1
        res = mem.recall(c["text"], top_k=1, now=now,
                         channel=c.get("channel"), speaker=c.get("speaker"))
        answered += bool(res)
        if res and res[0].event.eid in c["gold"]:
            hits[c["register"]] += 1
    total = sum(n.values())
    return {
        "paraphrase_recall_at_1": round(sum(hits.values()) / max(1, total), 4),
        "paraphrase_answer_rate": round(answered / max(1, total), 4),
        **{f"recall_{r}": round(hits[r] / max(1, n[r]), 4) for r in REGISTERS},
    }


def _abstains(mem, neg: dict, now: float) -> bool:
    res = mem.recall(neg["text"], top_k=TOP_K, now=now,
                     channel=neg.get("channel"), speaker=neg.get("speaker"))
    excl = set(neg.get("exclude_eids", ()))
    return all(r.event.eid in excl for r in res)   # [] abstains vacuously


def _recency_null(fx: dict, cues) -> float:
    """ME.1's null, carried forward unchanged: the most recent event compatible
    with the cue's provenance answers every cue."""
    hits = 0
    for c in cues:
        best = None
        for ev in fx["events"]:
            if c.get("channel") and ev["channel"] != c["channel"]:
                continue
            if c.get("speaker") and ev["speaker"] != c["speaker"]:
                continue
            if best is None or ev["t"] > best["t"]:
                best = ev
        hits += bool(best and best["eid"] in c["gold"])
    return hits / max(1, len(cues))


def _experiment(seed: int) -> dict:
    fx = F.build(seed)
    mem = F.load_into_memory(fx)
    now = fx["now"]
    headline = [c for c in fx["cues"] if not c["ambiguous"]]

    m = _recall_at_1(mem, headline, now)

    certify = fx["negatives"]["certify"]
    fam_ok = {f: 0 for f in FAMILIES}
    fam_n = {f: 0 for f in FAMILIES}
    for neg in certify:
        fam_n[neg["family"]] += 1
        fam_ok[neg["family"]] += _abstains(mem, neg, now)
    m.update({f"abstain_{f}": round(fam_ok[f] / max(1, fam_n[f]), 4)
              for f in FAMILIES})
    m["abstention_certify"] = round(sum(fam_ok.values()) / max(1, len(certify)), 4)
    m["abstention_family_min"] = min(m[f"abstain_{f}"] for f in FAMILIES)

    m["recency_null_recall"] = round(_recency_null(fx, headline), 4)
    m["headline_cues"] = len(headline)
    m["n_certify"] = len(certify)
    m["fixture_hash_seed_only"] = fx["hash"]   # _aggregate keeps run[0]
    return m


def _control(seed: int) -> dict:
    """The incumbent on its home benchmark: ME.1's templated word-subset cues.
    Must score >= 0.80 or the arm is mis-wired and its paraphrase ~0 is noise."""
    tmp = Path(tempfile.mkdtemp()) / "life.jsonl"
    mem, events, now = _build_life(seed, tmp)
    rng = random.Random(seed + 1)
    sampled = rng.sample(events, N_QUERIES)
    hits = sum(bool((r := mem.recall(_cue(rng, w), top_k=1, now=now))
                    and r[0].event.eid == ev.eid)
               for ev, w in sampled)
    return {"templated_recall": round(hits / N_QUERIES, 4)}


def _check(m: dict, c: dict):
    return (m["paraphrase_recall_at_1"] <= MAX_PARAPHRASE_RECALL
            and m["abstention_certify"] >= MIN_ABSTENTION
            and c["templated_recall"] >= MIN_TEMPLATED_RECALL)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["ME.11.A"], _experiment, _check,
                    control_fn=_control, ledger=ledger)
