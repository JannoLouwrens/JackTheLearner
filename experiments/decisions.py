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

    python -m experiments.decisions          # report
    python -m experiments.decisions --check  # ratchet: exit 1 if debt grew
"""
from __future__ import annotations

import datetime as _dt
import re
import sys
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
BASELINE_UNDECLARED = 8


def parse(text: str) -> tuple[dict, list]:
    """Return ({id: declaration}, [candidate open decisions from headers])."""
    decls: dict = {}
    lines = text.splitlines()
    for m in _DECIDE.finditer(text):
        did = m.group(1)
        start = text[: m.start()].count("\n") + 1
        d: dict = {"id": did, "line": start}
        for ln in lines[start:]:
            f = _FIELD.match(ln)
            if not f:
                if ln.strip() == "" or ln.startswith(" "):
                    continue
                break
            key, val = f.group(1), f.group(2).strip()
            d[key] = (d.get(key, "") + " " + val).strip() if key in d else val
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


def audit(text: str, today: _dt.date) -> tuple[list, list]:
    """Return (violations, rows). A violation blocks; a row is for the report."""
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
    return violations, rows


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
"""
    v, rows = audit(doc, _dt.date(2026, 8, 24))
    kinds = {did: kind for kind, did, _ in v}
    assert kinds.get("D90") == "UNDECLARED", kinds
    assert kinds.get("D91") == "MEANS-ESCALATED", kinds
    assert kinds.get("D92") == "NO-DEFAULT", kinds
    assert "D93" not in kinds and "D94" not in kinds, kinds
    assert [r["id"] for r in rows] == ["D93"], rows


def main(argv: list[str]) -> int:
    _fixture()
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
            print(f"           default: {r['default'][:110]}")
        print()

    if violations:
        print(f"  {len(violations)} decision(s) not armed:")
        for kind, did, why in violations:
            print(f"    [{kind:<15}] {did}")
            print(f"       {why}")
        print()

    undeclared = sum(1 for k, _, _ in violations if k == "UNDECLARED")
    if "--check" in argv:
        if undeclared > BASELINE_UNDECLARED:
            print(f"  RATCHET BROKEN: {undeclared} undeclared open decisions, "
                  f"baseline {BASELINE_UNDECLARED}. It may shrink, never grow.\n")
            return 1
        blocking = [v for v in violations if v[0] in ("MEANS-ESCALATED", "CLASS", "DATE")]
        if blocking:
            print(f"  {len(blocking)} hard violation(s) — see above.\n")
            return 1
        print(f"  ratchet ok ({undeclared}/{BASELINE_UNDECLARED} undeclared).\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
