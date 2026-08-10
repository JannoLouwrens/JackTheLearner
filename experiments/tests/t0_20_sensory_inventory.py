"""T0.20 — the sensory-inventory audit can see the bad case.

`experiments/senses.py` exists because of OVERSIGHT.md §3.2 (2026-08-10): five
senses GOAL.md calls constitutional had zero specs among 137, and no organ in
this system could report it, because every organ measures the project against
what the project wrote down. An audit whose standard comes from outside the
repository is the fix — and an audit is worth exactly as much as its ability to
report the bad case, which is what this battery establishes.

The control is the organ that FAILED, kept executable: coverage granted by
keyword scan over spec text. That is what the overseer ran by hand, and it is
the version anyone would write first. It has a specific, demonstrable defect —
it matched "voiced" (a struck geom in PG.5's audio spec) and concluded nothing
about voice, in the sense that a grep cannot tell "this spec IS about voice"
from "this spec says the word". Under the decoy registry it reports smell,
taste and voice as covered when none of their specs exist.

No physics, no training, no ledger writes: the fixtures are registry dicts built
in-process, so the numbers hold still while the RULE varies. That is the same
shape as T0.19.
"""
from __future__ import annotations

import re
from dataclasses import replace

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID, LADDER
from ..senses import ABSENT, DEMONSTRATED, INVENTORY, absent, audit

SPEC_ID = "T0.20"

# The five the overseer found at zero. They are the reason this file exists, so
# the battery asserts the inventory still NAMES them however coverage moves.
CONSTITUTIONAL_GAPS_2026_08_10 = ("smell", "taste", "voice", "pain", "temperature")


def _decoy_registry() -> dict:
    """The live registry with SM/TA/VO removed, plus PG.5's real decoy.

    PG.5 is genuine and stays: its notes carry "the voiced geom", which is the
    single hit the overseer's grep returned across 137 specs. Two more decoys
    are synthesised for pain and temperature so the keyword rule has something
    to trip on for senses that have no family to delete.
    """
    reg = {k: v for k, v in BY_ID.items()
           if not k.startswith(("SM.", "TA.", "VO."))}
    donor = BY_ID["T0.01"]
    reg["ZZ.decoy1"] = replace(
        donor, id="ZZ.decoy1",
        title="A decoy that merely mentions an odour and a taste",
        hypothesis="Nothing. It says smell, odour, taste and gustation without "
                   "testing any of them.")
    reg["ZZ.decoy2"] = replace(
        donor, id="ZZ.decoy2",
        title="A decoy that merely mentions pain and temperature",
        hypothesis="Nothing. It says pain, nociception, thermoregulation and "
                   "temperature without testing any of them.")
    return reg


def _keyword_coverage(reg: dict) -> set[str]:
    """THE PRE-GUARD ORGAN, kept executable: coverage by grep over spec text.

    Returns the set of sense keys a keyword scan would call covered. This is
    what OVERSIGHT.md ran, and reproducing it here is what makes the control a
    control rather than a tidied restatement (T0.08 property 5)."""
    covered = set()
    for sense in INVENTORY:
        if not sense.mentions:
            continue
        pat = re.compile(sense.mentions, re.I)
        for sp in reg.values():
            text = " ".join(str(x) for x in
                            (sp.title, sp.hypothesis, sp.notes or ""))
            if pat.search(text):
                covered.add(sense.key)
                break
    return covered


def _probe(coverage_is_keyword: bool) -> dict:
    """Run the six properties under one of the two coverage RULES."""
    failed: list[str] = []

    def cov_absent(reg: dict) -> set[str]:
        """Which senses read ABSENT under the rule under test."""
        if coverage_is_keyword:
            return {s.key for s in INVENTORY} - _keyword_coverage(reg)
        return set(absent(audit(by_id=reg)))

    live = dict(BY_ID)
    decoy = _decoy_registry()

    # P1 — the null: an EMPTY registry must report the whole inventory ABSENT.
    # An audit that reads its own declarations rather than the ladder would
    # report coverage for a repository containing nothing.
    if cov_absent({}) != {s.key for s in INVENTORY}:
        failed.append("p1_empty_registry_all_absent")

    # P2 — the inventory is an OUTSIDE reference and does not shrink. It still
    # names all five senses the overseer found at zero, whatever their coverage
    # is today, and it is not derived from anything the ladder can edit.
    keys = {s.key for s in INVENTORY}
    if not set(CONSTITUTIONAL_GAPS_2026_08_10) <= keys or len(INVENTORY) < 10:
        failed.append("p2_inventory_still_names_the_gaps")

    # P3 — a declared spec id that no longer resolves LOSES its coverage. The
    # failure mode this kills: a family gets renamed or deleted and the audit
    # keeps reporting the sense as covered off its own stale declaration.
    gone = cov_absent(decoy)
    if not {"smell", "taste", "voice"} <= gone:
        failed.append("p3_deleted_family_loses_coverage")

    # P4 — THE LOAD-BEARING ONE. A spec that merely CONTAINS a sense's word
    # buys no coverage. Under the keyword rule the decoys and PG.5's "voiced"
    # make smell, taste and voice read covered while none of their specs exist,
    # which is exactly the artifact that hid the hole for 30 hours.
    mentions_present = _keyword_coverage(decoy)
    if not {"smell", "taste", "voice"} <= mentions_present:
        # The decoy must actually be a decoy, or P4 proves nothing either way.
        failed.append("p4_decoy_is_not_a_decoy")
    elif not {"smell", "taste", "voice"} <= gone:
        failed.append("p4_mention_must_not_grant_coverage")

    # P5 — a broken claim is reported, not silently dropped: the audit lists
    # unresolvable declared ids under `missing` rather than pretending they
    # were never declared.
    missing_reported = sum(len(c.missing) for c in audit(by_id=decoy))
    if missing_reported < 7:  # SM.01/02 TA.01/02/03 VO.01/02 were removed
        failed.append("p5_missing_ids_are_reported")

    # P6 — against the LIVE registry every declared id resolves, and PASS is
    # only ever claimed for a sense some registered spec actually demonstrates.
    led = Ledger()
    live_cov = audit(by_id=live, ledger=led)
    if any(c.missing for c in live_cov):
        failed.append("p6_live_declarations_all_resolve")
    for c in live_cov:
        if c.status == DEMONSTRATED and not all(
                led.status(s) is Status.PASS for s in c.passing):
            failed.append("p6_live_declarations_all_resolve")
            break

    return {
        "properties_checked": float(N_PROPERTIES),
        "properties_failed": float(len(failed)),
        "failed_names": ",".join(failed),
        "inventory_size": float(len(INVENTORY)),
        "senses_absent_live": float(len(absent(audit(by_id=live)))),
        "registry_size": float(len(LADDER)),
    }


N_PROPERTIES = 6


def _experiment(seed: int) -> dict:
    return _probe(coverage_is_keyword=False)


def _control(seed: int) -> dict:
    """Coverage by keyword scan — the organ that failed, kept executable.

    It must break on P3 and P4 and on nothing else that matters: those two are
    the properties that separate "a spec claims this sense" from "a spec says
    this word", which is the entire difference between an audit and a grep."""
    return _probe(coverage_is_keyword=True)


def _check(m: dict, c: dict) -> Status | bool:
    # All six ran AND all six held. Gating on `properties_failed == 0` alone
    # would let a battery that stopped early read as clean (T0.13's own first
    # bug, and T0.19 carries the same guard).
    experiment_clean = (m["properties_failed"] == 0.0
                        and m["properties_checked"] == N_PROPERTIES
                        and c["properties_checked"] == N_PROPERTIES)
    # The control must fail, and fail on THE properties that define the guard.
    control_names = set(str(c.get("failed_names", "")).split(","))
    control_broken = (c["properties_failed"] > 0.0
                      and {"p3_deleted_family_loses_coverage",
                           "p4_mention_must_not_grant_coverage"} <= control_names)
    return bool(experiment_clean and control_broken)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID[SPEC_ID], _experiment, _check, control_fn=_control,
                    ledger=ledger)
