"""T0.28 — the escalation tool can be shown catching a deadlock and a claim-death.

`experiments/decisions.py` stands over eleven pre-registered constitutional
defaults, ten of which fire on the same date. Every audit opens by running it.
Until this spec it was certified by fixtures its own author wrote — the precise
self-certification `SYSTEM.md`'s first law exists to distrust — and it had
already been wrong once, for six days, in the worst direction: `SYSTEM.md` said
*"experiments/decisions.py enforces this"* of the three-part safety clause while
the tool read no default's content at all. A governing document naming an
enforcement is making a capability claim, and law 1 binds it like any other.

WHAT THE KNOWN-POSITIVES ARE. They are events, not inventions. On 2026-08-29
`D8`'s armed default read *"PARK BA.02"* and `BA.02` was the only claim-kind
spec behind `balance`, a commitment the owner listed as constitutional
(`GOAL.md:41`). One unanswered calendar date would have left that commitment
with nothing falsifiable behind it. `BA.03` was registered on the morning of
2026-08-30 — the prescribed repair, *register a successor* — and that, not any
change to the tool, is why the live board is green today.

WHICH IS EXACTLY WHY THE FIXTURE IS SYNTHETIC. Rows are injected through
`audit(rows_for_safety=...)` rather than read from the live ledger, because a
known-positive pinned to the live repository stops being exercised the moment
somebody repairs the repository. That is the same disease one level up:
a guard that is green because its subject vanished rather than because it was
fixed. P5 makes the two silences distinguishable INSIDE the tool; building the
fixture this way makes them distinguishable for the tool itself.

THE CONTROL is the organ as it stood before 2026-08-30, kept executable: an
`audit()` with no safety pass, and a `--check` whose blocking set omits
`NO-DEFAULT`. Both holes were real and both are reconstructed by deletion
rather than by paraphrase (T0.08 property 5) — the safety pass only ever
APPENDS `SAFETY-CLAIM-DEAD` violations, so dropping them from the return value
reproduces the old return value exactly. It must miss the `D8` positive, miss
the both-named case, and exit 0 on a goal-class entry that arms nothing.

NO ledger writes, no training, no world: the documents are strings built
in-process and the rows are dicts, so the numbers hold still while the RULE
varies. Same shape as T0.19, T0.20 and T0.21.

WHAT THIS SPEC DOES NOT CERTIFY, stated so no later reader repeats SYSTEM.md's
mistake in this file's name: two of the three safety clauses — *never edits
GOAL.md*, *never weakens a threshold* — are still enforced by nobody. They are
properties of the COMMIT that fires a default, not of the text that arms it,
and no battery over `decisions.py` can see them. `T0.29` (`champions.py`) is
the companion certificate and is owed.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import io
import re

from ..coverage import _claim_dead
from ..decisions import (BASELINE_UNDECLARED, DOC, audit, blast_radius,
                         check_rc, main, parse)
from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

SPEC_ID = "T0.28"

# A claim about decisions.py must die when decisions.py changes: a certificate
# that survives edits to its own subject is a certificate about nothing (PG.6
# hashing playground.py; T0.21 hashing coverage.py).
IMPL_DEPS = ["experiments/decisions.py"]

N_PROPERTIES = 10

# The pre-2026-08-30 blocking set, verbatim. `NO-DEFAULT` is absent — that is
# the hole, not an abbreviation.
LEGACY_BLOCKING = ("MEANS-ESCALATED", "CLASS", "DATE", "SAFETY-CLAIM-DEAD")

TODAY = _dt.date(2026, 8, 30)

# ── the documents. Each is the smallest thing that carries one defect. ──

# The four parse defects plus one correctly armed entry, and D95 in D1's real
# shape: a header calling an OPTION stale must not read as the DECISION being
# settled.
DOC_PARSE = """
## D90 — an open decision with no declaration at all (OPEN, owner)
## D91 — a means fork sent to the owner instead of to a bakeoff (OPEN, owner)

DECIDE: D91
  class:     means
  default:   ask the owner
  decide_by: 2026-01-01

## D92 — a goal fork with no default (OPEN, owner)

DECIDE: D92
  class:     goal
  decide_by: 2026-01-01

## D93 — correctly armed (OPEN, owner)

DECIDE: D93
  class:     goal
  default:   keep the conservative arm, journal the firing
  decide_by: 2099-01-01

## D94 — RESOLVED, must NOT be reported as open
## D95 — THE OPTION SET IS STALE: an option contradicts a later decree
## D95 — the original question (OPEN, owner)
"""

# `D8` as it actually stood on 2026-08-29, reduced to the sentence that names
# the spec. Wrapped across two physical lines ON PURPOSE: the id lives on the
# continuation line, so a parser that truncates a default at its first line
# computes an empty blast radius and the guard silently never fires. That
# truncation shipped for one run.
DOC_D8 = """
## D80 — the body question (OPEN, owner)

DECIDE: D80
  class:     goal
  default:   Option 1 — PARK the balance claim until a body with directional
             catch authority exists, re-parenting BA.02 behind the
             playground-humanoid line. BA.01 stands untouched.
  decide_by: 2026-08-31
"""

# The same document with the successor ALSO named — the case where registering
# BA.03 is not the repair, because one date still reaches every claim.
DOC_D8_BOTH = DOC_D8.replace("re-parenting BA.02 behind the",
                             "re-parenting BA.02 and BA.03 behind the")

# A means fork whose default names the only two claims behind `smell`. A means
# entry is never armed, so its radius must not be consulted at all — and it
# must block on its own account.
DOC_MEANS = """
## D82 — which nose (OPEN, owner)

DECIDE: D82
  class:     means
  default:   PARK SM.02 and SM.03 both.
  decide_by: 2026-08-31
"""

# A default that names nothing resolvable. `ZZ.99` is a typo, not a reference.
DOC_TYPO = """
## D83 — a default full of typos (OPEN, owner)

DECIDE: D83
  class:     goal
  default:   PARK ZZ.99 and QQ.01 until the world grows traps.
  decide_by: 2026-08-31
"""

# The registry the fixture resolves ids against. Values are unused — only
# membership decides whether an id is a reference (`blast_radius`).
FIXTURE_BY_ID = {i: 1 for i in ("BA.01", "BA.02", "BA.03", "T2.01",
                                "SM.02", "SM.03")}


def _row(commitment: str, kinds: dict, n_pass: int = 0) -> dict:
    """One `coverage.report()` row, reduced to the keys the safety pass reads."""
    return {"commitment": commitment, "n_pass": n_pass, "kinds": kinds,
            "parked": {}, "n_specs": len(kinds), "specs": sorted(kinds)}


# `balance` on 2026-08-29: one sensor, one claim, no PASS.
ROWS_BEFORE = [_row("balance", {"BA.01": "sensor", "BA.02": "claim"})]
# ...after the 08-30 repair: the successor exists and no default names it.
ROWS_REPAIRED = [_row("balance", {"BA.01": "sensor", "BA.02": "claim",
                                  "BA.03": "claim"})]
# ...and the state that must NOT be mistaken for the repair: the claim is gone.
ROWS_VANISHED = [_row("balance", {"BA.01": "sensor"})]


def _violations(text: str, rows, *, safety_enforced: bool) -> list:
    """The organ under test, or the organ as it stood before 2026-08-30.

    The pre-fix `audit()` did not contain the safety loop. That loop only ever
    APPENDS, so deleting its output from the return value reconstructs the old
    behaviour exactly rather than paraphrasing it.
    """
    v, _rows = audit(text, TODAY, rows_for_safety=rows, by_id=FIXTURE_BY_ID)
    if safety_enforced:
        return v
    return [x for x in v if x[0] != "SAFETY-CLAIM-DEAD"]


def _rc(violations: list, *, safety_enforced: bool) -> int:
    """`--check`'s exit code, under today's blocking set or the legacy one."""
    if safety_enforced:
        return check_rc(violations)
    undeclared = sum(1 for k, _, _ in violations if k == "UNDECLARED")
    if undeclared > BASELINE_UNDECLARED:
        return 1
    return 1 if any(v[0] in LEGACY_BLOCKING for v in violations) else 0


def _hazards(text: str, rows, *, safety_enforced: bool) -> list:
    """`(id, commitment, claims)` triples this organ reports for `text`."""
    return [(did, why.split("'")[1], why)
            for kind, did, why in _violations(text, rows,
                                              safety_enforced=safety_enforced)
            if kind == "SAFETY-CLAIM-DEAD"]


def _probe(safety_enforced: bool) -> dict:
    failed: list[str] = []
    S = safety_enforced

    # P1 — the four parse defects and the one correct entry. A scanner that
    # flags everything is as useless as one that flags nothing, so the negative
    # (D93 armed, D94 resolved) is half the property. D95 is D1's real shape:
    # a header calling an OPTION stale is not a resolution of the DECISION.
    v, rows = audit(DOC_PARSE, TODAY, rows_for_safety=[], by_id=FIXTURE_BY_ID)
    kinds = {did: kind for kind, did, _ in v}
    if (kinds.get("D90") != "UNDECLARED"
            or kinds.get("D91") != "MEANS-ESCALATED"
            or kinds.get("D92") != "NO-DEFAULT"
            or kinds.get("D95") != "UNDECLARED"
            or "D93" in kinds or "D94" in kinds
            or [r["id"] for r in rows] != ["D93"]):
        failed.append("p1_parse_defects_are_flagged_and_armed_is_not")

    # P2 — KNOWN POSITIVE, and it is a thing that happened. `D8` on 2026-08-29:
    # its default names BA.02, which was the only claim behind `balance`. The
    # tool must name the decision, the commitment and the claim, and the gate
    # must STOP — a violation the exit code ignores is a violation nobody acts
    # on. The pre-fix organ read no default's content and misses it entirely.
    haz = _hazards(DOC_D8, ROWS_BEFORE, safety_enforced=S)
    if (len(haz) != 1 or haz[0][0] != "D80" or haz[0][1] != "balance"
            or "BA.02" not in haz[0][2]
            or _rc(_violations(DOC_D8, ROWS_BEFORE, safety_enforced=S),
                   safety_enforced=S) != 1):
        failed.append("p2_d8_known_positive_fires")

    # P3 — it must go quiet ONLY for the prescribed repair, and the property is
    # the CONTRAST: the same document, the same default, one registered
    # successor between them — loud before, quiet after. Asserting only the
    # quiet half would be passed by an organ that is quiet about everything,
    # which is precisely the organ this spec's control is.
    if (not _hazards(DOC_D8, ROWS_BEFORE, safety_enforced=S)
            or _hazards(DOC_D8, ROWS_REPAIRED, safety_enforced=S)):
        failed.append("p3_successor_is_the_repair")

    # P4 — and registering a successor is NOT the repair when the same date
    # still reaches it. A default naming BA.02 AND BA.03 fires again: the
    # question is never "does a successor exist", it is "does one calendar
    # event reach every claim". Without this, the prescribed repair could be
    # discharged by writing a spec the default already covers.
    hb = _hazards(DOC_D8_BOTH, ROWS_REPAIRED, safety_enforced=S)
    if len(hb) != 1 or hb[0][0] != "D80" or "BA.03" not in hb[0][2]:
        failed.append("p4_both_named_fires")

    # P5 — THE TWO SILENCES ARE DIFFERENT, and this is the property the
    # existing fixture did not have. Delete the claim instead of succeeding it
    # and the hazard ALSO disappears — because there is nothing left to put at
    # risk. That must not read as health: `coverage._claim_dead` is True on the
    # vanished row and False on the repaired one, so quiet-because-fixed and
    # quiet-because-gone are distinguishable, and the second is `coverage.py`'s
    # red rather than a green here. A guard that cannot tell them apart can be
    # discharged by deleting its subject.
    vanished_quiet = not _hazards(DOC_D8, ROWS_VANISHED, safety_enforced=S)
    if (not vanished_quiet
            or not _claim_dead(ROWS_VANISHED[0])
            or _claim_dead(ROWS_REPAIRED[0])
            or _claim_dead(ROWS_BEFORE[0])):
        failed.append("p5_vanished_is_not_repaired")

    # P6 — a recorded PASS is never put at risk by a calendar. A default cannot
    # un-record a certificate, so a commitment with a passing claim is not a
    # hazard however thoroughly the default names it — and the same row with
    # the PASS removed MUST fire, or "no hazard" is coming from somewhere else
    # entirely. Both directions or neither.
    doc_loco = DOC_D8.replace("BA.02", "T2.01")
    passed = [_row("locomotion", {"T2.01": "claim"}, n_pass=1)]
    unpassed = [_row("locomotion", {"T2.01": "claim"}, n_pass=0)]
    if (_hazards(doc_loco, passed, safety_enforced=S)
            or not _hazards(doc_loco, unpassed, safety_enforced=S)):
        failed.append("p6_a_pass_is_not_at_calendar_risk")

    # P7 — an id that resolves to nothing is a typo, not a reference; counting
    # it would inflate every radius with noise. And the over-approximation runs
    # the other way ON PURPOSE: a resolvable id ANYWHERE in the text is in the
    # radius, including on a wrapped continuation line, because the radius says
    # what a firing could REACH and never what it will do. The continuation
    # half is a real regression — the first parser truncated a default at its
    # first physical line, which would have computed {} for `D8` itself.
    joined = parse(DOC_D8)[0]["D80"]
    if (blast_radius("PARK BA.02 and ZZ.99", FIXTURE_BY_ID) != {"BA.02"}
            or blast_radius(DOC_TYPO, FIXTURE_BY_ID)
            or _hazards(DOC_TYPO, ROWS_BEFORE, safety_enforced=S)
            or "BA.02" not in joined.get("default", "")
            or blast_radius(joined.get("default", ""),
                            FIXTURE_BY_ID) != {"BA.01", "BA.02"}):
        failed.append("p7_typo_is_not_a_reference")

    # P8 — a MEANS fork is never armed, so its blast radius must not be
    # consulted: it is not a thing the owner may leave unanswered, it is a
    # bakeoff somebody has not written. Its default names both claims behind
    # `smell` and must still raise no hazard — while blocking on its own
    # account, in both organs.
    smell = [_row("smell", {"SM.02": "claim", "SM.03": "claim"})]
    mv = _violations(DOC_MEANS, smell, safety_enforced=S)
    if (_hazards(DOC_MEANS, smell, safety_enforced=S)
            or not any(k == "MEANS-ESCALATED" for k, _, _ in mv)
            or _rc(mv, safety_enforced=S) != 1):
        failed.append("p8_means_is_never_armed")

    # P9 — the exit code blocks on every class the report calls fatal, not on
    # one. `NO-DEFAULT` — a goal entry that declares itself and arms nothing —
    # was printed and exited 0 until 2026-08-30: D1's exact disease with a
    # DECIDE block on top. The same one-class-ratchet shape the 40th audit
    # found in `champions.py`. The second direction keeps the fix honest: an
    # UNDECLARED backlog at or below its baseline is a queue, not a wall, and
    # a ratchet that blocked on everything would be no ratchet at all.
    nodefault = _violations(DOC_PARSE, [], safety_enforced=S)
    nodefault = [x for x in nodefault if x[0] == "NO-DEFAULT"]
    backlog = [("UNDECLARED", f"D{i}", "") for i in range(BASELINE_UNDECLARED)]
    if (_rc(nodefault, safety_enforced=S) != 1
            or _rc(backlog, safety_enforced=S) != 0
            or _rc(backlog + [("UNDECLARED", "Dx", "")],
                   safety_enforced=S) != 1):
        failed.append("p9_ratchet_counts_every_class")

    # P10 — against the LIVE document, and this is the half a fixture can
    # never do. It must parse into an armed set with no blocking violation,
    # every armed row must carry an ISO date and a non-empty default, and the
    # report must print each default IN FULL. That last one is not decoration:
    # `main()` sliced defaults at 110 characters while the live ones run
    # 369-1041, so 70-89% of every constitutional clause the owner armed had
    # never appeared in any report anyone read. A clause nobody can read is an
    # unreviewed clause, and the check that would catch a truncation is
    # whether the LONGEST live default's last words survive to stdout.
    live_v, live_rows = audit(DOC.read_text(), TODAY)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main([])
    printed = re.sub(r"\s+", " ", buf.getvalue())
    longest = max((r["default"] for r in live_rows), key=len, default="")
    tail = re.sub(r"\s+", " ", longest)[-60:]
    if (not live_rows
            or any(v[0] in ("SAFETY-CLAIM-DEAD", "MEANS-ESCALATED", "CLASS",
                            "DATE", "NO-DEFAULT") for v in live_v)
            or any(not r["default"] or not isinstance(r["due"], _dt.date)
                   for r in live_rows)
            or len(longest) < 200 or tail not in printed):
        failed.append("p10_live_document_is_armed_and_readable")

    return {
        "properties_checked": float(N_PROPERTIES),
        "properties_failed": float(len(failed)),
        "failed_names": ",".join(failed),
        "live_armed": float(len(live_rows)),
        "live_violations": float(len(live_v)),
        "live_longest_default_chars": float(len(longest)),
        "live_specs_in_radius": float(len(
            {s for r in live_rows for s in blast_radius(r["default"])})),
    }


def _experiment(seed: int) -> dict:
    return _probe(safety_enforced=True)


def _control(seed: int) -> dict:
    """`decisions.py` as it stood before 2026-08-30, kept executable.

    Two holes, both real: `audit()` carried no safety pass, so no default's
    content was ever read; and `--check`'s blocking set omitted `NO-DEFAULT`,
    so a goal-class entry that armed nothing was printed and exited 0. It must
    miss the `D8` known-positive (P2), miss the both-named case (P4) and pass a
    document containing an unarmed escalation (P9).
    """
    return _probe(safety_enforced=False)


def _check(m: dict, c: dict) -> Status | bool:
    # Every property ran AND every property held, on both arms. Gating on
    # `properties_failed == 0` alone lets a battery that stopped early read as
    # clean — T0.13's own first bug, and T0.19/T0.20/T0.21 all carry this guard.
    experiment_clean = (m["properties_failed"] == 0.0
                        and m["properties_checked"] == N_PROPERTIES
                        and c["properties_checked"] == N_PROPERTIES)
    # The control must fail, and fail on THE properties that name the two holes
    # this file closed. A control that fails for some other reason is not the
    # disease reproduced, it is a different bug.
    control_names = set(str(c.get("failed_names", "")).split(","))
    control_broken = (c["properties_failed"] > 0.0
                      and {"p2_d8_known_positive_fires",
                           "p4_both_named_fires",
                           "p9_ratchet_counts_every_class"} <= control_names)
    return bool(experiment_clean and control_broken)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID[SPEC_ID], _experiment, _check, control_fn=_control,
                    ledger=ledger)
