"""T0.29 — the seat tool cannot be discharged by deleting the ring.

`experiments/champions.py` decides whether each seat in Jack's anatomy is
CONTESTABLE — whether the invitation a `BY DEFAULT` marking extends is addressed
to a room that exists. Every audit opens by running it. Until this spec it was
certified by a fixture its own author wrote, the self-certification `SYSTEM.md`'s
first law exists to distrust, and it had already been wrong four times in six
days. It is the second and last of the governance instruments to get a ledger
certificate; `T0.28` did `decisions.py` on 2026-08-30 and is this file's model.

THE KNOWN-POSITIVES ARE RECORDED EVENTS. None is invented.

(a) THE RATCHET, and it is the one the 49th audit called the property a future
agent is most likely to break while "cleaning up". `--check` counted
`ARENA-MISSING` alone. Delete a phantom id from an arena cell and the seat stops
being `ARENA-MISSING` and becomes `NO-ARENA`: the count FALLS, the report prints
a smaller number, and the seat has gone from *uncontested* to *permanently
uncontestable*. The ratchet paid you for the exact repair `champions.py`'s own
docstring forbids in bold. Three seats — ASR, Speaker ID, Language grounding —
were already sitting in that blind spot, invisible to the number that gated the
file. This is the third instrument here with the one-class-ratchet disease
(`coverage.py`, closed by `T0.21` P2; `decisions.py`'s `NO-DEFAULT`, closed by
`T0.28` P9), and the 40th and 47th audits both named it before it was fixed.

(b) CLOSABILITY. `W.6` was withdrawn 2026-08-09, superseded by `NE.08`, and sat
inside the range `W.1`–`W.7` that the World seat cited. Because ranges expand,
one withdrawn id made that seat's ratchet unsatisfiable by any amount of honest
work — and five consecutive audits (44th–48th) relayed *"register `W.1`–`W.7`"*
without anyone noticing a component of it could not be obeyed.

(c) THE QUANTIFIER. `all(v == "NOT_RUN")` discharged a seat the moment ANY arena
spec had run — a `fixture`, a `sensor`, or a VOID, none of which beats an
incumbent. Carried unrepaired by the 43rd, 44th and 45th audits, over a champion
cell reading, in bold, *"DEFAULT, never defended"*.

THE CONTROL is the organ as it stood before 2026-08-29, kept executable, with
all three holes. Each is reconstructed BY DELETION rather than paraphrase
(T0.08 property 5): the ratchet by counting the old class, the closability split
by passing `unregisterable={}` (the branch simply never fires), and the
quantifier by dropping the `UNCONTESTED` rows the old `all(...)` would not have
raised — today's predicate is strictly weaker, so the old verdict set is a
subset of today's and deletion reconstructs it exactly.

THE FIXTURE IS SYNTHETIC ON PURPOSE, for `T0.28`'s reason: a known-positive
pinned to the live document stops being exercised the moment somebody repairs
the document — a guard green because its subject vanished rather than because it
was fixed. P5 and P10 carry the live half.

NO ledger writes, no training, no world: documents are strings built in-process
and the registry is a dict, so the numbers hold still while the RULE varies.
Same shape as T0.19, T0.20, T0.21 and T0.28.

WHAT THIS DOES NOT CERTIFY, stated here so no later reader repeats SYSTEM.md's
mistake in this file's name: seat MARKINGS are still INFERRED — read from a
table column with a prose fallback, never declared. `champions.py`'s own
docstring admits it and calls the repair a `HELD:`/`ARENA:` syntax it is not
permitted to invent unilaterally. No battery over this module can close that,
because the ambiguity is in the document, not in the code. It is owed.
"""
from __future__ import annotations

import contextlib
import io

from ..champions import (BASELINE_ARENA_MISSING, BASELINE_UNFALSIFIABLE, DOC,
                         UNREGISTERABLE, arena_refs, audit, main, parse,
                         resolve, unfalsifiable)
from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

SPEC_ID = "T0.29"

# A claim about champions.py must die when champions.py changes: a certificate
# that survives edits to its own subject is a certificate about nothing (PG.6
# hashing playground.py; T0.21 hashing coverage.py; T0.28 hashing decisions.py).
IMPL_DEPS = ["experiments/champions.py"]

N_PROPERTIES = 10


# ── the fixture. One seat per defect, plus the healthy seats that make a
#    flag-everything scanner fail as loudly as a flag-nothing one. ──

_HEADER = "| seat | champion | held | arena | challenger status |\n|---|---|---|---|---|\n"

_ROWS = [
    # healthy: won its ring outright
    "| Healthy verdict seat | winner | **BY VERDICT** (OK.01) | OK.01 (registered) | a challenger |",
    # healthy: unearned marking, but a real claim spec answered it
    "| Default seat a claim defended | incumbent | **DEFAULT, never defended** | OK.01–OK.02 | a challenger |",
    # (c): the only completion is a declared fixture/sensor — seats nobody
    "| Default seat a fixture answered | incumbent | **DEFAULT, never defended** | OK.04 + OK.05 | a challenger |",
    # (c): the only completion is VOID — not a verdict
    "| Default seat whose run went VOID | incumbent | **DEFAULT, never defended** | OK.06 + OK.03 | a challenger |",
    # (a): every ref dangles, and nothing registered could unseat it
    "| Phantom arena seat | incumbent | **DEFAULT, never defended** | ZZ.00 + ZZ.01 (queued) | a challenger |",
    # (b): a range that swallows a withdrawn id — the W.6 case, mixed with a
    #      merely-unwritten one so both instructions must appear
    "| Seat citing a withdrawn spec | incumbent | **DEFAULT, never defended** | ZZ.00 + W.6 (queued) | a challenger |",
    # names no arena at all: rule 3 is unmeetable, not merely unmet
    "| No arena at all | incumbent | **BY DECREE** (owner) | HR bakeoff (queued) | a challenger |",
    # unearned, arena exists, has never run: a debt in the world, not the file
    "| Uncontested decree seat | incumbent | **BY DECREE** (owner) | OK.03 (registered) | a challenger |",
    # the prose hazard: the word "default" inside a sentence saying nobody holds it
    "| Vacant by default words | **VACANT** — the incumbent by default is nobody | — | OK.01 | a challenger |",
]

_TAIL = """
### DECIDED BY DECREE 2099-01-01: SOMETHING

WHAT STILL RUNS: ZZ.02 (a floor nobody wrote). Cheap, CPU.

### Superseded context: not a decree, must not be parsed as one

This section names ZZ.09 and must contribute no seat and no violation.
"""

DOC_FIXTURE = "\n" + _HEADER + "\n".join(_ROWS) + "\n" + _TAIL

# THE DELETION. The phantom seat's arena cell loses its two dangling ids and
# says only what a tidy-up would leave behind. This is the edit the old ratchet
# rewarded: ARENA-MISSING falls by one, and the seat becomes UNCONTESTABLE.
DOC_DELETED = DOC_FIXTURE.replace("ZZ.00 + ZZ.01 (queued)", "(queued)")


class _S:
    """The only thing `champions.py` reads off a spec is its `notes`."""

    def __init__(self, notes: str = ""):
        self.notes = notes


# OK.04/OK.05 declare support kinds; OK.06 ran but only to VOID.
FIXTURE_BY_ID = {"OK.01": _S("COVERS: smell (claim)"), "OK.02": _S(),
                 "OK.03": _S(), "OK.04": _S("COVERS: smell (fixture)"),
                 "OK.05": _S("COVERS: balance (sensor)"), "OK.06": _S()}
FIXTURE_RAN = {"OK.01": "PASS", "OK.02": "PASS", "OK.03": "NOT_RUN",
               "OK.04": "PASS", "OK.05": "PASS", "OK.06": "VOID"}

# ...and the same registry with ONE phantom written. Registering is the only
# permitted way to lower the ratchet, so the property that it does lower it is
# half of P2's claim — without it, "invariant under deletion" is also satisfied
# by a constant.
#
# `ZZ.01` and not `ZZ.00`, so the drop is exactly one and the assertion can say
# so. `ZZ.00` is cited by TWO seats, and registering it discharges both — which
# is correct behaviour and a weaker property to assert, since `< before` passes
# for an implementation that miscounts by any amount in the right direction.
REGISTERED_BY_ID = dict(FIXTURE_BY_ID, **{"ZZ.01": _S()})


def _audit(doc: str, by_id: dict, ran: dict, *, legacy: bool):
    """The organ under test, or the organ as it stood before 2026-08-29.

    Two of the three holes live here. `unregisterable={}` reconstructs the
    absent closability split exactly — the branch that appends the
    CORRECT-THE-CITATION clause simply never fires. The `UNCONTESTED` filter
    reconstructs the `all(status == "NOT_RUN")` quantifier: today's predicate
    (`not challenger_runs`) is strictly weaker, so the old violation set is a
    subset of today's and removing the difference IS the old set.
    """
    v, seats = audit(doc, by_id, lambda s: ran.get(s, "NOT_RUN"),
                     unregisterable={} if legacy else None)
    if not legacy:
        return v, seats
    by_seat = {s["seat"]: s for s in seats}
    kept = []
    for kind, seat, why in v:
        if kind == "UNCONTESTED":
            status = by_seat[seat].get("arena_status", {})
            if not all(x == "NOT_RUN" for x in status.values()):
                continue  # the old quantifier called this seat defended
        kept.append((kind, seat, why))
    return kept, seats


def _ratchet(violations: list, seats: list, *, legacy: bool) -> int:
    """The number `--check` refuses to let grow. The third hole is here."""
    if legacy:
        return sum(1 for k, _, _ in violations if k == "ARENA-MISSING")
    return len(unfalsifiable(seats))


def _flags(violations: list) -> dict:
    out: dict = {}
    for kind, seat, _why in violations:
        out.setdefault(seat, set()).add(kind)
    return out


def _probe(legacy: bool) -> dict:
    failed: list[str] = []
    L = legacy

    v, seats = _audit(DOC_FIXTURE, FIXTURE_BY_ID, FIXTURE_RAN, legacy=L)
    flags = _flags(v)
    by_seat = {s["seat"]: s for s in seats}

    # P1 — the three defect classes fire, and the healthy seats do NOT. The
    # negative half is not decoration: a scanner that flags everything is as
    # useless as one that flags nothing, and the failure that would hurt most
    # here is indicting a seat that is doing everything the file asks.
    if (flags.get("Phantom arena seat") != {"ARENA-MISSING"}
            or flags.get("No arena at all") != {"NO-ARENA"}
            or flags.get("Uncontested decree seat") != {"UNCONTESTED"}
            or "Healthy verdict seat" in flags
            or "Default seat a claim defended" in flags):
        failed.append("p1_three_classes_fire_and_healthy_seats_do_not")

    # P2 — KNOWN POSITIVE, and the one the 49th audit flagged as most likely to
    # be broken by a future tidy-up. Deleting a seat's phantom arena ids must
    # NOT improve the ratchet: the seat converts ARENA-MISSING -> NO-ARENA and
    # goes from uncontested to uncontestable, which is worse, not better. Both
    # directions are asserted, because "invariant under deletion" alone is also
    # true of a constant: REGISTERING the phantoms must lower it by exactly one.
    vd, sd = _audit(DOC_DELETED, FIXTURE_BY_ID, FIXTURE_RAN, legacy=L)
    vr, sr = _audit(DOC_FIXTURE, REGISTERED_BY_ID, FIXTURE_RAN, legacy=L)
    before = _ratchet(v, seats, legacy=L)
    if (_flags(vd).get("Phantom arena seat") != {"NO-ARENA"}
            or _ratchet(vd, sd, legacy=L) != before
            or _ratchet(vr, sr, legacy=L) != before - 1):
        failed.append("p2_deleting_the_arena_does_not_help")

    # P3 — and the ratchet must be counting the right SEATS, not just the right
    # number. A seat with no runnable arena is unfalsifiable however its refs
    # are spelled; a seat with a live arena beside a phantom one is a citation
    # defect and must stay OUT, or the two conditions collapse and the report
    # stops being able to tell a documentation bug from a dead seat.
    mixed = DOC_FIXTURE.replace("ZZ.00 + ZZ.01 (queued)", "ZZ.00 + OK.01")
    vm, sm = _audit(mixed, FIXTURE_BY_ID, FIXTURE_RAN, legacy=L)
    dead = set(unfalsifiable(seats))
    if ("Phantom arena seat" not in dead
            or "No arena at all" not in dead
            or "Healthy verdict seat" in dead
            or "Phantom arena seat" in set(unfalsifiable(sm))
            or _flags(vm).get("Phantom arena seat") != {"ARENA-MISSING"}):
        failed.append("p3_unfalsifiable_is_not_the_violation_count")

    # P4 — KNOWN POSITIVE (b). A ref the project DECIDED against draws
    # "CORRECT THE CITATION"; a merely-unwritten one draws "REGISTER". The
    # fixture seat cites ONE of each, so a mixed cell must say both — one
    # un-writable id may not excuse the ids that are simply unwritten, and an
    # implementation that reported only the worse half would read as correct on
    # a single-ref seat. The pre-fix organ says "register" about a spec that was
    # withdrawn, which is what five audits relayed.
    why = {seat: w for k, seat, w in v if k == "ARENA-MISSING"}
    withdrawn = why.get("Seat citing a withdrawn spec", "")
    phantom = why.get("Phantom arena seat", "")
    if ("CORRECT THE CITATION" not in withdrawn
            or "REGISTER to discharge" not in withdrawn
            or "ZZ.00" not in withdrawn
            or "CORRECT THE CITATION" in phantom
            or by_seat["Seat citing a withdrawn spec"]["arena_unregisterable"]
            != ["W.6"]
            or by_seat["Phantom arena seat"]["arena_unregisterable"] != []):
        failed.append("p4_unregisterable_is_not_a_todo")

    # P5 — the live decision set, and it is the half a fixture cannot do. Every
    # UNREGISTERABLE entry must cite the record that closed it — an entry is a
    # DECISION, not an opinion, and a ref parked here without a record is how
    # inventory debt gets laundered into "we decided not to". `W.6` and `T2.21`
    # are the two, and neither may resolve in the live registry: if one ever
    # does, the entry is stale and the seat is contestable after all. THAT
    # CLAUSE FIRED FOR REAL on 2026-09-02 (FAIL attempt 1, `81e3b97`): `D1.0`
    # sat in the set from 2026-08-30, was registered and run to VOID while the
    # entry stayed, and the first certificate re-buy caught the contradiction —
    # the pinned set here tracks the decision record, and shrank with it.
    if (set(UNREGISTERABLE) != {"W.6", "T2.21"}
            or any(len(r) < 30 or not any(ch.isdigit() for ch in r)
                   for r in UNREGISTERABLE.values())
            or any(resolve(ref, BY_ID) for ref in UNREGISTERABLE)):
        failed.append("p5_every_decision_names_its_record")

    # P6 — KNOWN POSITIVE (c). A seat is discharged by a CHALLENGER, never by
    # "some arena spec completed". Three distinct false positives, each its own
    # row: a VOID is not a verdict; a declared `fixture`/`sensor` seats nobody;
    # and both are multi-arena seats, because for a ONE-arena seat the old and
    # new quantifiers coincide — which is why every single-arena seat was caught
    # correctly for three audits while the consequential ones were not.
    if (flags.get("Default seat a fixture answered") != {"UNCONTESTED"}
            or flags.get("Default seat whose run went VOID") != {"UNCONTESTED"}
            or by_seat["Default seat a claim defended"]["challenger_runs"]
            != ["OK.01", "OK.02"]
            or by_seat["Default seat a fixture answered"]["challenger_runs"]):
        failed.append("p6_only_a_challenger_discharges")

    # P7 — a cited RANGE is where a phantom hides. `W.1–W.7` is one string
    # naming seven arenas and reporting its endpoints undercounts the hole by
    # five. Zero padding is part of the id (`LC.00` != `LC.0`); the alpha form
    # drops its stem on the right; `PL.*` names a whole family and resolves to
    # every registered member, which for an empty family is nothing; and
    # `D1.0 + T2.21` must NOT parse as a range, or two unrelated seats' arenas
    # merge into a fictional span.
    if (arena_refs("W.1–W.7") != [f"W.{i}" for i in range(1, 8)]
            or arena_refs("LC.00–LC.06") != [f"LC.{i:02d}" for i in range(7)]
            or arena_refs("ME.11.A–F") != [f"ME.11.{c}" for c in "ABCDEF"]
            or arena_refs("D1.0 + T2.21") != ["D1.0", "T2.21"]
            or resolve("ZZ.*", FIXTURE_BY_ID) != []
            or resolve("OK.*", FIXTURE_BY_ID) != sorted(FIXTURE_BY_ID)):
        failed.append("p7_a_range_names_every_arena_in_it")

    # P8 — a decree is a seat. `### DECIDED BY DECREE ...` is not a table row,
    # but it holds a component of Jack by owner fiat and pre-registers its own
    # re-open trigger. A table-only parser reports this file CLEAN on `PL.00` —
    # the single most-cited missing arena in the project, and the one the
    # PLASTIC-ONLY decree's re-open trigger is keyed to. The negative is the
    # other half: a heading that merely contains prose about a decree is not
    # one, so `ZZ.09` must appear nowhere.
    decrees = [s for s in seats if s["kind"] == "decree"]
    if (len(decrees) != 1
            or decrees[0]["arena_missing"] != ["ZZ.02"]
            or flags.get(decrees[0]["seat"]) != {"ARENA-MISSING"}
            or any("ZZ.09" in w for _k, _s, w in v)):
        failed.append("p8_a_decree_outside_the_table_is_a_seat")

    # P9 — markings are INFERRED, so the two known inference hazards are pinned.
    # The `held` COLUMN outranks the champion cell, and the Deliberation row's
    # real prose — "the incumbent by default is nobody" — contains the word
    # `default` inside a sentence denying anyone holds the seat. A VACANT seat
    # can never be UNCONTESTED: there is no holder for an unanswered invitation
    # to protect. Reading that row as BY DEFAULT would invent a champion.
    vacant = by_seat["Vacant by default words"]
    if (vacant["held"] != "VACANT" or "Vacant by default words" in flags
            or by_seat["Healthy verdict seat"]["held"] != "BY VERDICT"
            or by_seat["No arena at all"]["held"] != "BY DECREE"):
        failed.append("p9_vacant_is_not_a_champion")

    # P10 — against the LIVE document. It must parse into a real seat list, both
    # ratchets must sit at or under their baselines, `--check` must exit 0
    # TODAY, and — the direction that matters — it must exit 1 when the
    # unfalsifiable count grows. A ratchet nobody has watched refuse anything is
    # a ratchet nobody has tested; that is this repo's oldest lesson and the
    # reason `champions.py` carries a fixture at all.
    live_v, live_seats = audit(DOC.read_text(), BY_ID,
                               lambda s: Ledger().status(s).value)
    live_dead = unfalsifiable(live_seats)
    live_missing = sum(1 for k, _, _ in live_v if k == "ARENA-MISSING")
    with contextlib.redirect_stdout(io.StringIO()):
        rc_today = main(["--check"])
    grown = [dict(s, arena_present=[]) for s in live_seats]
    if (len(live_seats) < 20
            or len(live_dead) > BASELINE_UNFALSIFIABLE
            or live_missing > BASELINE_ARENA_MISSING
            or rc_today != 0
            or len(unfalsifiable(grown)) <= BASELINE_UNFALSIFIABLE
            or any(seat not in {s["seat"] for s in live_seats}
                   for _k, seat, _w in live_v)):
        failed.append("p10_live_document_is_parsed_and_ratcheted")

    return {
        "properties_checked": float(N_PROPERTIES),
        "properties_failed": float(len(failed)),
        "failed_names": ",".join(failed),
        "live_seats": float(len(live_seats)),
        "live_violations": float(len(live_v)),
        "live_unfalsifiable": float(len(live_dead)),
        "live_arena_missing": float(live_missing),
    }


def _experiment(seed: int) -> dict:
    return _probe(legacy=False)


def _control(seed: int) -> dict:
    """`champions.py` as it stood before 2026-08-29, kept executable.

    Three holes, all real and all carried by multiple audits before repair: a
    ratchet counting `ARENA-MISSING` alone (so deleting a phantom id improved
    it), no closability split (so a withdrawn spec drew the instruction
    "register it", five times), and the `all(status == "NOT_RUN")` challenger
    quantifier (so one fixture or one VOID discharged a multi-arena seat).

    It must fail P2, P4 and P6 — and it must still PASS the rest, which is the
    part that makes it a control rather than a broken import: the disease was
    three specific holes in a mechanism that was otherwise sound, and a control
    that failed everything would prove nothing about which hole this spec
    guards.
    """
    return _probe(legacy=True)


def _check(m: dict, c: dict) -> Status | bool:
    # Every property ran AND every property held, on both arms. Gating on
    # `properties_failed == 0` alone lets a battery that stopped early read as
    # clean — T0.13's own first bug, and every T0.1x/T0.2x battery since.
    experiment_clean = (m["properties_failed"] == 0.0
                        and m["properties_checked"] == N_PROPERTIES
                        and c["properties_checked"] == N_PROPERTIES)
    # The control must fail, and fail on THE properties that name the three
    # holes. A control that fails for some other reason is not the disease
    # reproduced, it is a different bug.
    control_names = set(str(c.get("failed_names", "")).split(","))
    control_broken = (c["properties_failed"] > 0.0
                      and {"p2_deleting_the_arena_does_not_help",
                           "p4_unregisterable_is_not_a_todo",
                           "p6_only_a_challenger_discharges"} <= control_names)
    return bool(experiment_clean and control_broken)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID[SPEC_ID], _experiment, _check, control_fn=_control,
                    ledger=ledger)
