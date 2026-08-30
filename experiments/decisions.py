"""Every open decision is an experiment somebody refused to run.

WHY THIS EXISTS. `D1 — does the 57M trunk stay in the control path?` sat OPEN for
twenty days with `evidence complete` in its own title, blocking 35 specs: the
entire curiosity family, the entire unified-brain family, and six of the seven
Tier 5 claims. Nothing was missing. Three independent runs at matched env-steps
already said a 125K network learns locomotion at 7.11 sigma while the 57M trunk
cannot clear its own 3-sigma gate. The four arms were all runnable. The bakeoff
was never written.

It was never written because SYSTEM.md contradicts itself, and the losing clause
was the important one:

    rule 3   "Decisions are made by bakeoff, never by argument. ... If you find
              yourself reasoning about which approach is better, stop and write
              the bakeoff."
    line 164 "Deleting components, spending money, and architecture calls are
              the owner's - escalate them."

D1 is an architecture call, so line 164 won and rule 3 was cancelled for exactly
the class of question it was written for. The loop obeyed both rules perfectly
and stalled for three weeks.

THE OWNER'S RULING (2026-08-24, verbatim): *"in the future he mustnt get blocked
by anything like this but instead test and try and research both and decide at
end which works better"*.

THE INVARIANT THAT MAKES THAT SAFE, and the whole reason this file is not just
`escalation = off`:

    A measurement may choose among PERMITTED arms.
    It may never choose WHAT IS PERMITTED.

"Which architecture learns better" is a question about MEANS: both arms run, one
pre-registered metric separates them, the loop decides and never asks. "Is a
frozen component permitted inside Jack" is a question about the GOAL: it defines
what winning means, so no experiment can answer it - an experiment that picks
its own success criterion will drift to whatever is cheapest to demonstrate,
which is the precise failure this project's pre-registration discipline exists
to prevent. Those go to the owner, and only those.

AND AN ESCALATION MUST NOT BE ABLE TO DEADLOCK. That was D1's real defect: no
default and no clock, so silence was indistinguishable from "not yet" forever.
Every goal-class decision here carries a DEFAULT and a DECIDE_BY. If the date
passes unanswered the default fires, loudly and in the journal. A default may
only ever pick among ALREADY-PERMITTED actions - it may never edit GOAL.md,
weaken a threshold, or widen what is allowed - so the worst case of an unattended
default is an experiment the owner would have sequenced differently, and the
ledger's history makes that reversible.

THE DECLARATION. Inferring a decision's state from prose is the mistake
`coverage.py` already made and retired: a title is not a claim, and a regex over
a document flatters whoever wrote the regex. So openness and its terms are
DECLARED, at the start of a line, in the same idiom as `COVERS:`:

    DECIDE: D1
      class:     goal
      default:   Bakeoff every constitutionally-permitted arm at matched
                 experience and seat the winner by the recorded margin.
      decide_by: 2026-08-31
      blocks:    T2.01, T2.02

`blocks:` is joined to the live dependency graph - the cost in specs is COMPUTED,
never typed. A hand-written cost is stale the moment the ladder grows, and this
ladder grew 169 -> 179 in one morning.

THE SAFETY CLAUSE, AND HOW MUCH OF IT THIS FILE ACTUALLY ENFORCES (2026-08-30).
`SYSTEM.md` says of the paragraph above: *"experiments/decisions.py enforces
this; the overseer runs it every audit."* Until today that sentence was false.
This file checked that a default EXISTS, that its class is legal, and that its
date parses. It never read the default's CONTENT, so all three safety clauses -
no GOAL.md edit, no weakened threshold, no widening of what is permitted - were
enforced by nobody. Meanwhile every live default hand-asserts its own compliance
in prose ("no threshold moves", "GOAL.md is not touched", "this is a NARROWING
and only a narrowing"). Author self-certification is the exact thing the rest of
this apparatus exists to distrust, and eleven of them fire on 2026-08-31.

What is enforced now, and it is ONE of the three clauses:

    SAFETY-CLAIM-DEAD - a default may not be the single unattended event that
    leaves a GOAL.md commitment with nothing falsifiable behind it.

Computed, not parsed for intent: every spec id NAMED ANYWHERE in a default's
text is that default's BLAST RADIUS - the set of specs whose fate that one
calendar event could reach. If some commitment's every live claim-kind spec
lies inside a single default's blast radius, then that commitment's
falsifiability rests entirely on one date passing unanswered, and
`coverage.py`'s standing rule applies: *"The repair for a red here is to
REGISTER a successor spec, never to unpark or quiet the tool."*

The blast radius deliberately OVER-approximates. It does not attempt to decide
what a default does - only what it could touch - because a scanner that tried to
read intent out of 918 characters of constitutional English would be the regex
that flatters its author, twice warned about above. Over-approximating is the
safe direction: a false positive costs one registry entry, and registering a
successor spec is never the wrong thing to have done.

KNOWN-POSITIVE, historical and real. On 2026-08-29 `D8`'s default read "PARK
BA.02" and `BA.02` was the only claim-kind spec behind `balance` (GOAL.md:41,
constitutional). This check would have fired. It does not fire today because a
builder registered `BA.03` on the morning of 08-30 - the successor, which is
precisely the prescribed repair. A guard whose positive is a thing that actually
happened, and whose green is a thing somebody actually fixed.

WHAT IS STILL NOT ENFORCED, stated here so no later reader repeats SYSTEM.md's
mistake in this file's name: **"never edits GOAL.md" and "never weakens a
threshold" are NOT checked.** Neither is decidable from a DECIDE block - both
are properties of the COMMIT that fires the default, not of the text that arms
it, and the honest place to catch them is a pre-commit check on the firing diff.
Two of three clauses remain on the author's word. Do not write a prose scanner
for them; write the diff check.

AND THIS FILE IS ITSELF A CAPABILITY CLAIM, SO IT IS NOW UNDER THE LEDGER
(2026-08-30, T0.28). For six days `SYSTEM.md` asserted an enforcement this file
did not perform, and the only thing standing behind the code was a fixture its
own author wrote — the exact self-certification law 1 exists to distrust,
applied to the tool that polices eleven constitutional defaults. `T0.28` lifts
`_fixture()` and `_safety_fixture()` into a spec the ladder can see, adds the
directions they did not test (quiet-because-repaired vs quiet-because-vanished;
a PASS is never at calendar risk; a wrapped default's continuation lines are in
the blast radius), and keeps the PRE-2026-08-30 organ executable as the control
so the guard must be shown catching something it once missed. `IMPL_DEPS` pins
this file, so editing it stales the certificate by construction.

    python -m experiments.decisions          # report
    python -m experiments.decisions --check  # ratchet: exit 1 if debt grew
"""
from __future__ import annotations

import datetime as _dt
import re
import sys
import textwrap as _textwrap
from pathlib import Path

DOC = Path(__file__).resolve().parent.parent / "docs" / "DECISIONS_NEEDED.md"

# Start-of-line only. `coverage.py` learned this the expensive way: "nest" inside
# "ho-nest" credited a shelter commitment, so an inline mention must never be
# able to look like a declaration.
_DECIDE = re.compile(r"^DECIDE:\s*([A-Za-z0-9._-]+)\s*$", re.M)
_FIELD = re.compile(r"^\s+(class|default|decide_by|blocks):\s*(.*)$")
_HEADER = re.compile(r"^##\s+(.*)$", re.M)
_DID = re.compile(r"^(D\d+)\b")

# A decision whose every surviving header says RESOLVED is not open. D2 and D5
# both carry live "RESOLVED"/"off your desk" headers above their originals, and
# a scanner that reads only the first header reports them as open forever.
#
# `STALE` IS NOT IN THIS LIST, AND THE FIRST VERSION OF IT WAS. D1's second
# header reads "THE OPTION SET IS STALE: option A contradicts the PLASTIC-ONLY
# decree" — a header about a stale OPTION inside a decision that is emphatically
# open. Matching it silently exonerated the most expensive entry in the file:
# 35 blocked specs, reported as settled, by the very tool written to catch
# exactly that. `coverage.py` learned the same lesson one document over — the
# instrument that finds gaps can have a gap, and it will flatter you. A settled
# marker must name the DECISION's fate, never an entry's freshness.
_SETTLED = re.compile(r"RESOLVED|off your desk|BY THE CALENDAR", re.I)

CLASSES = ("means", "goal")

# Ratchet, not gate. Eight decisions are open today and none carries a default;
# a guard that fails everywhere on day one is one nobody keeps green, and a guard
# nobody keeps green is decoration (LESSONS.md, citations.py precedent). This
# number may SHRINK and may never GROW.
BASELINE_UNDECLARED = 10


def parse(text: str) -> tuple[dict, list]:
    """Return ({id: declaration}, [candidate open decisions from headers])."""
    decls: dict = {}
    lines = text.splitlines()
    for m in _DECIDE.finditer(text):
        did = m.group(1)
        start = text[: m.start()].count("\n") + 1
        d: dict = {"id": did, "line": start}
        last = None
        for ln in lines[start:]:
            f = _FIELD.match(ln)
            if f:
                last = f.group(1)
                d[last] = f.group(2).strip()
                continue
            # An indented line that is not a new key CONTINUES the last one. The
            # first parser silently dropped these, which truncated every wrapped
            # `default:` to its first line — a default that reads as half a
            # sentence is worse than none, because it still looks armed.
            if ln.startswith(" ") and ln.strip() and last:
                d[last] = (d[last] + " " + ln.strip()).strip()
                continue
            if ln.strip() == "":
                continue
            break
        decls[did] = d

    headers: dict = {}
    for m in _HEADER.finditer(text):
        title = m.group(1).strip()
        did = _DID.match(title)
        key = did.group(1) if did else (title.split("(OPEN")[0].strip()[:52]
                                        if "(OPEN" in title else None)
        if key is None:
            continue
        headers.setdefault(key, []).append(title)
    candidates = [k for k, ts in headers.items() if not any(_SETTLED.search(t) for t in ts)]
    return decls, sorted(candidates)


def cost_of(blocks: list[str]) -> tuple[int, list]:
    """Specs freed if EVERY id in `blocks` is fixed. Computed, never typed."""
    try:
        from .protocol import Ledger
        from .registry import LADDER
        from .run import _rank_blockers, _terminal_blockers
    except Exception:
        return 0, []
    led = Ledger()
    _mentions, frees, groups = _rank_blockers(_terminal_blockers(led), led, LADDER)
    got = set(blocks)
    freed: set = set()
    for b in blocks:
        freed |= set(frees.get(b, []))
    for roots, specs in groups.items():          # a pair only counts if the
        if set(roots) <= got:                    # decision covers the whole set
            freed |= set(specs)
    return len(freed), sorted(freed)


# A spec id as it appears inside a default's prose. Same shape as
# `coverage.GOAL_CITATION`, and resolved against `BY_ID` for the same reason
# `champions.py` resolves its arenas: an id that names nothing is not a
# reference, and counting it would inflate every blast radius with typos.
_SPEC_ID = re.compile(r"\b([A-Z]{1,4}[0-9]?\.[0-9]{1,2})\b")


def blast_radius(default_text: str, by_id=None) -> set:
    """Spec ids a default's firing COULD reach — never what it will do.

    Over-approximation is the point; see the module docstring. An id that does
    not resolve in the registry is dropped, not counted.
    """
    if by_id is None:
        try:
            from .registry import BY_ID as by_id
        except Exception:
            return set()
    return {i for i in _SPEC_ID.findall(default_text or "") if i in by_id}


def safety_hazards(decls: dict, candidates: list, rows=None, by_id=None) -> list:
    """`[(decision_id, commitment, [claim ids])]` — commitments whose entire
    live falsifiable surface sits inside one armed default's blast radius.

    Returns [] — not an error — if `coverage.py` or the registry cannot be
    imported. This check is an ADDITION to the ratchet, and a guard that turns
    an unrelated ImportError into a red gate is a guard that gets switched off.
    """
    try:
        from . import coverage as _cov
        if rows is None:
            rows = _cov.report()
    except Exception:
        return []

    radii = {}
    for did in candidates:
        d = decls.get(did) or {}
        if (d.get("class") or "").lower() != "goal" or not d.get("default"):
            continue
        radii[did] = blast_radius(d["default"], by_id)

    out = []
    for r in rows:
        # A commitment with a PASS is not at risk: `_claim_dead` needs BOTH no
        # passing claim AND no live claim declaration, and a default cannot
        # un-record a certificate.
        if r.get("n_pass"):
            continue
        claims = {sid for sid, k in r["kinds"].items() if k == "claim"}
        if not claims:
            continue                      # already claim-dead; coverage owns that red
        for did, hits in radii.items():
            if claims <= hits:
                out.append((did, r["commitment"], sorted(claims)))
    return sorted(out)


def audit(text: str, today: _dt.date, rows_for_safety=None,
          by_id=None) -> tuple[list, list]:
    """Return (violations, rows). A violation blocks; a row is for the report.

    `rows_for_safety`/`by_id` are injection points for the safety pass and
    default to the live `coverage.report()` and registry. They exist so a
    certificate can drive this function on a KNOWN state — `T0.28` replays
    `D8` as it stood on 2026-08-29 — rather than on whatever the repository
    happens to look like on the day the gate runs. A guard whose known-positive
    can only be exercised while the repo is broken is a guard that stops being
    tested the moment somebody fixes the repo.
    """
    decls, candidates = parse(text)
    violations, rows = [], []

    for did in candidates:
        d = decls.get(did)
        if d is None:
            violations.append(("UNDECLARED", did,
                               "open, but declares no DECIDE block — no default, "
                               "no deadline, so silence deadlocks it"))
            continue

        cls = (d.get("class") or "").lower()
        if cls not in CLASSES:
            violations.append(("CLASS", did, f"class must be one of {CLASSES}, got {cls!r}"))
            continue

        if cls == "means":
            violations.append(("MEANS-ESCALATED", did,
                               "a MEANS fork must be settled by bakeoff, not by the "
                               "owner — write the bakeoff and delete this entry"))
            continue

        missing = [k for k in ("default", "decide_by") if not d.get(k)]
        if missing:
            violations.append(("NO-DEFAULT", did,
                               f"goal-class decision missing {', '.join(missing)} — "
                               "an escalation without a default is a deadlock generator"))
            continue

        try:
            due = _dt.date.fromisoformat(d["decide_by"])
        except ValueError:
            violations.append(("DATE", did, f"decide_by not ISO: {d['decide_by']!r}"))
            continue

        blocks = [b.strip() for b in (d.get("blocks") or "").split(",") if b.strip()]
        n, _which = cost_of(blocks)
        rows.append({"id": did, "due": due, "overdue": (today - due).days,
                     "blocks": blocks, "cost": n,
                     "default": d.get("default", "")})

    # The safety clause, last because it needs the whole armed set at once.
    for did, commitment, claims in safety_hazards(decls, candidates,
                                                  rows=rows_for_safety,
                                                  by_id=by_id):
        violations.append(("SAFETY-CLAIM-DEAD", did,
                           f"every live claim spec behind '{commitment}' "
                           f"({', '.join(claims)}) is named in this default — one "
                           "unanswered date would leave a GOAL.md commitment with "
                           "nothing falsifiable behind it. Repair: REGISTER a "
                           "successor claim spec (coverage.py's rule), never park "
                           "or quiet"))
    return violations, rows


# Violation kinds that STOP the gate rather than merely appearing in the
# report. `UNDECLARED` is not here: it is the ratcheted backlog, allowed to
# shrink and never grow (BASELINE_UNDECLARED).
#
# `NO-DEFAULT` WAS NOT HERE EITHER, AND THAT WAS THE HOLE (2026-08-30, T0.28).
# The ratchet counted exactly one class — `UNDECLARED` — and treated four
# others as blocking, so a goal-class entry that declared itself and then
# armed nothing was printed and exited 0. That is D1's exact disease with a
# DECIDE block on top: an escalation with no default and no clock, which this
# whole file exists to make impossible. It is the same one-class-ratchet shape
# the 40th audit found in `champions.py` (`ARENA-MISSING` only) and that
# `T0.21 P2` closed in `coverage.py`. Live count when it was closed: zero, so
# the strengthening cost nothing — which is the only cheap moment to do it.
BLOCKING = ("MEANS-ESCALATED", "CLASS", "DATE", "SAFETY-CLAIM-DEAD",
            "NO-DEFAULT")


def check_rc(violations: list) -> int:
    """The `--check` exit code, as a function of the violations alone.

    Extracted from `main` so the certificate can assert the EXIT CODE rather
    than a re-implementation of it. A test that reproduces the gate's logic
    proves the copy agrees with itself.
    """
    undeclared = sum(1 for k, _, _ in violations if k == "UNDECLARED")
    if undeclared > BASELINE_UNDECLARED:
        return 1
    return 1 if any(v[0] in BLOCKING for v in violations) else 0


def _fixture() -> None:
    """A known-positive this tool must flag, exercising the real code path.

    Every audit tool here carries one, because a scanner that has never been
    shown catching anything is a scanner nobody has tested (LESSONS.md). This
    document contains one of each defect plus one correct entry; the tool must
    flag exactly the three and pass the fourth.
    """
    doc = """
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
    v, rows = audit(doc, _dt.date(2026, 8, 24))
    kinds = {did: kind for kind, did, _ in v}
    assert kinds.get("D90") == "UNDECLARED", kinds
    assert kinds.get("D91") == "MEANS-ESCALATED", kinds
    assert kinds.get("D92") == "NO-DEFAULT", kinds
    assert "D93" not in kinds and "D94" not in kinds, kinds
    # The regression that shipped for one run: a header calling an OPTION stale
    # must not read as the DECISION being settled. D95 is D1's real shape.
    assert kinds.get("D95") == "UNDECLARED", kinds
    assert [r["id"] for r in rows] == ["D93"], rows


def _safety_fixture() -> None:
    """The planted positive for SAFETY-CLAIM-DEAD, on the real code path.

    The shape is not invented: it is `D8` as it actually stood on 2026-08-29,
    when `BA.02` was the only claim-kind spec behind `balance` and `D8`'s
    default read "PARK BA.02". `BA.03` was registered on the morning of
    2026-08-30 and that is why the live check is green — so this fixture is the
    only place the failing state still exists, and without it the guard has
    never been seen to fire.

    Rows are synthetic on purpose. Pinning the fixture to the live ledger would
    make it go quiet the moment the repo is repaired, which is how a guard ends
    up green because its subject vanished rather than because it was fixed.
    """
    decls = {
        "D80": {"class": "goal", "default": "Option 1 - PARK BA.02 until a body "
                                            "with directional catch authority "
                                            "exists. BA.01 stands untouched."},
        "D81": {"class": "goal", "default": "Bakeoff the permitted arms of T2.01 "
                                            "at matched experience."},
        "D82": {"class": "means", "default": "PARK SM.02 and SM.03 both."},
    }
    rows = [
        # the positive: every claim behind it sits inside D80's radius
        {"commitment": "balance", "n_pass": 0,
         "kinds": {"BA.01": "sensor", "BA.02": "claim"}, "parked": {}},
        # a successor exists and is named by no default — the prescribed repair
        {"commitment": "balance-repaired", "n_pass": 0,
         "kinds": {"BA.01": "sensor", "BA.02": "claim", "BA.03": "claim"},
         "parked": {}},
        # a PASS cannot be un-recorded by a calendar, so this is never at risk
        {"commitment": "locomotion", "n_pass": 1,
         "kinds": {"T2.01": "claim"}, "parked": {}},
        # already claim-dead: coverage.py owns that red, not this file
        {"commitment": "empty", "n_pass": 0, "kinds": {"XX.01": "fixture"},
         "parked": {}},
        # a MEANS entry is never armed, so its radius must not be consulted
        {"commitment": "smell", "n_pass": 0,
         "kinds": {"SM.02": "claim", "SM.03": "claim"}, "parked": {}},
    ]
    by_id = {"BA.01": 1, "BA.02": 1, "BA.03": 1, "T2.01": 1,
             "SM.02": 1, "SM.03": 1}
    got = safety_hazards(decls, list(decls), rows=rows, by_id=by_id)
    assert got == [("D80", "balance", ["BA.02"])], got

    # And it must go quiet for the RIGHT reason: register the successor and the
    # same document stops being a hazard. This is the 08-30 repair, replayed.
    rows2 = [dict(rows[0], kinds={"BA.01": "sensor", "BA.02": "claim",
                                  "BA.03": "claim"})]
    assert safety_hazards(decls, list(decls), rows=rows2, by_id=by_id) == []

    # An id that resolves to nothing is a typo, not a reference.
    assert blast_radius("PARK BA.02 and ZZ.99", by_id) == {"BA.02"}


def main(argv: list[str]) -> int:
    _fixture()
    _safety_fixture()
    text = DOC.read_text()
    today = _dt.date.today()
    violations, rows = audit(text, today)

    print(f"\nOpen decisions — {DOC.relative_to(DOC.parent.parent)}\n")
    if rows:
        print("  armed (default fires if unanswered):")
        for r in sorted(rows, key=lambda r: -r["cost"]):
            due = "OVERDUE — DEFAULT IS DUE TO FIRE" if r["overdue"] > 0 else f"due {r['due']}"
            print(f"    {r['id']:<6} costs {r['cost']:3d} specs   {due}")
            print(f"           blocks {', '.join(r['blocks']) or '(nothing declared)'}")
            # NOT `[:110]`. The live defaults run 369-1041 characters, so that
            # slice had never shown 70-89% of any constitutional clause in any
            # report an owner or auditor ever read — including, for four
            # months of report output, the sentences saying what the default
            # would PARK. A default nobody can read is an unreviewed default.
            body = _textwrap.fill(r["default"], 92,
                                  initial_indent="           default: ",
                                  subsequent_indent="                    ")
            print(body if r["default"] else "           default: (none)")
        print()

    if violations:
        print(f"  {len(violations)} decision(s) not armed:")
        for kind, did, why in violations:
            print(f"    [{kind:<15}] {did}")
            print(f"       {why}")
        print()

    undeclared = sum(1 for k, _, _ in violations if k == "UNDECLARED")
    if "--check" in argv:
        rc = check_rc(violations)          # the gate's verdict, computed once
        if undeclared > BASELINE_UNDECLARED:
            print(f"  RATCHET BROKEN: {undeclared} undeclared open decisions, "
                  f"baseline {BASELINE_UNDECLARED}. It may shrink, never grow.\n")
        elif rc:
            blocking = [v for v in violations if v[0] in BLOCKING]
            print(f"  {len(blocking)} hard violation(s) — see above.\n")
        else:
            print(f"  ratchet ok ({undeclared}/{BASELINE_UNDECLARED} undeclared).\n")
        return rc
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
