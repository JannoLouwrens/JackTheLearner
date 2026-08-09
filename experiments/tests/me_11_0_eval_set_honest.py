"""ME.11.0 — the paraphrase eval set is honest before anyone is scored.

Every ME.11 arm `depends_on` this spec, so `protocol.blocked_by()` structurally
prevents an encoder from being scored against an unvalidated evaluation set.
That ordering is the whole design: the cheapest way to get a wrong architecture
decision is to measure six arms carefully on a benchmark nobody checked.

Six assertions, each of which can fail:

1. **Lexical disjointness.** No cue shares a content word with any event in its
   own gold set, beyond an explicitly allowed speaker name. Checked with the
   incumbent retriever's OWN tokeniser, so "content word" means exactly what
   the thing being tested means by it.
2. **The lexical null scores ~0.** This follows from (1) by construction, which
   is precisely why it is worth measuring: if containment scores above 0.10 the
   cues have leaked surface form somewhere the token check did not look.
3. **The oracle ceiling is >= 0.95.** An oracle that re-parses each event's
   STORED TEXT back into concepts must answer nearly every cue. Gold comes from
   the generator's bookkeeping and the oracle comes from the text; the two
   disagreeing means the generator does not write what it thinks it writes, and
   every arm would then be scored against a floor effect.
4. **The fixture is frozen.** Two builds at one seed produce one hash. An arm
   carrying a different hash was not compared to the others.
5. **The splits are big enough for the claims that will be made on them.**
   >= 19 positives per provenance stratum (Mondrian conformal minimum at
   alpha = 0.05) and >= 300 tune / >= 300 certify negatives, family-balanced
   (Clopper-Pearson minimum to certify abstention >= 0.95 at 95% confidence).
6. **Gold sets are small enough to be gold, and no register is hollowed out.**
   Every headline cue has |G| <= 3, the AMBIGUOUS partition is non-empty (an
   unexercised partition mechanism is untested code), and every register keeps
   >= 30 headline cues. That last clause is not decoration: two earlier
   generator designs quietly exiled 59 of register R4's 60 cues to AMBIGUOUS,
   and the headline recall would have been a three-register average wearing a
   four-register name.

CONTROL (must pass where the experiment fails): a deliberately LEAKY cue set,
built by deleting words from the target sentence instead of substituting
synonyms, must drive the same lexical null to >= 0.80. A leak detector that
cannot see a planted leak is not a detector — it is a rubber stamp, and it
would certify a broken eval set just as confidently.
"""
from __future__ import annotations

from ..fixtures import paraphrase_eval as F
from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

MAX_NULL_RECALL = 0.10
MIN_ORACLE = 0.95
MIN_LEAKY_RECALL = 0.80
MIN_STRATUM = 19
MIN_SPLIT = 300
MIN_PER_REGISTER = 30


def _null_recall(mem, cues, now: float) -> float:
    """The incumbent retriever, given every advantage: provenance filters when
    the cue is attributed, top-1 counted correct against the whole gold SET."""
    hits = 0
    for c in cues:
        res = mem.recall(c["text"], top_k=1, now=now,
                         channel=c.get("channel"), speaker=c.get("speaker"))
        if res and res[0].event.eid in c["gold"]:
            hits += 1
    return hits / max(1, len(cues))


def _oracle_recall(fx: dict, cues) -> float:
    """Answer each cue by re-deriving concepts FROM THE STORED TEXT.

    Deliberately not the generator's own annotations: this is the second,
    independent path to the truth, and its disagreement with gold is the only
    signal that would catch a generator whose sentences and bookkeeping have
    drifted apart.
    """
    parsed = [F.parse_concepts(e["text"]) for e in fx["events"]]
    hits = 0
    for c in cues:
        best, best_eid = -1, None
        for ev, ann in zip(fx["events"], parsed):
            if not F.satisfies(ann, ev, c["constraints"]):
                continue
            if ev["t"] > best:                    # tie-break by recency
                best, best_eid = ev["t"], ev["eid"]
        if best_eid is not None and best_eid in c["gold"]:
            hits += 1
    return hits / max(1, len(cues))


def _overlap_violations(fx: dict) -> int:
    """A cue may share the speaker's name with its target and nothing else."""
    bad = 0
    for c in fx["cues"] + fx["ambiguous_probes"]:
        allowed = {c["speaker"]} if c.get("speaker") else set()
        q = F._tokens(c["text"]) - allowed
        for eid in c["gold"]:
            if q & F._tokens(fx["events"][eid]["text"]):
                bad += 1
                break
    return bad


def _experiment(seed: int) -> dict:
    fx = F.build(seed)
    rebuilt = F.build(seed)
    mem = F.load_into_memory(fx)
    now = fx["now"]

    headline = [c for c in fx["cues"] if not c["ambiguous"]]
    amb = [c for c in fx["cues"] if c["ambiguous"]] + fx["ambiguous_probes"]

    strata, registers = {}, {r: 0 for r in ("R1", "R2", "R3", "R4")}
    for c in headline:
        strata[c["stratum"]] = strata.get(c["stratum"], 0) + 1
        registers[c["register"]] += 1
    fam = {}
    for split in ("tune", "certify"):
        for n in fx["negatives"][split]:
            fam[(split, n["family"])] = fam.get((split, n["family"]), 0) + 1

    return {
        "events": len(fx["events"]),
        "headline_cues": len(headline),
        "ambiguous_cues": len(amb),
        "overlap_violations": _overlap_violations(fx),
        "lexical_null_recall": round(_null_recall(mem, headline, now), 4),
        "oracle_ceiling": round(_oracle_recall(fx, headline), 4),
        "max_gold_size": max(len(c["gold"]) for c in headline),
        "min_register_cues": min(registers.values()),
        "min_stratum_positives": min(strata.values()),
        "n_tune": len(fx["negatives"]["tune"]),
        "n_certify": len(fx["negatives"]["certify"]),
        "min_family_cell": min(fam.values()),
        "hash_stable": 1.0 if fx["hash"] == rebuilt["hash"] else 0.0,
        "fixture_hash_seed_only": fx["hash"],   # _aggregate keeps run[0]
    }


def _control(seed: int) -> dict:
    """Plant a leak; the same detector must see it."""
    fx = F.build(seed)
    mem = F.load_into_memory(fx)
    return {"leaky_null_recall": round(
        _null_recall(mem, fx["leaky_cues"], fx["now"]), 4)}


def _check(m: dict, c: dict):
    if m["hash_stable"] < 1.0:
        return Status.VOID          # not a refutation: the fixture is not frozen
    return (m["overlap_violations"] == 0
            and m["lexical_null_recall"] <= MAX_NULL_RECALL
            and m["oracle_ceiling"] >= MIN_ORACLE
            and m["max_gold_size"] <= F.K_AMB
            and m["ambiguous_cues"] >= 1
            and m["min_register_cues"] >= MIN_PER_REGISTER
            and m["min_stratum_positives"] >= MIN_STRATUM
            and m["n_tune"] >= MIN_SPLIT and m["n_certify"] >= MIN_SPLIT
            and m["min_family_cell"] >= 1
            and c["leaky_null_recall"] >= MIN_LEAKY_RECALL)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["ME.11.0"], _experiment, _check,
                    control_fn=_control, ledger=ledger)
