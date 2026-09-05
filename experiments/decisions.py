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

THE SECOND DOCUMENT, AND WHY THIS FILE NOW READS TWO (2026-09-04, 69th audit
B2). Everything above polices `DECISIONS_NEEDED.md` — the file where an
owner-ask has a `class`, a `default` and a `decide_by`. It says nothing about
asks that never reach that file. On 2026-09-03 the Review published its largest
strategic recommendation ("W1 stops being a queue row and becomes the project's
stated stage") into `docs/PROGRESS.md`, which is CURRENT-STATE BY DESIGN: the
next Review rewrote the page and the recommendation was gone, unanswered, at 24
hours old. `decisions --check` printed `ratchet ok (0/10 undeclared)` — true of
the file it read, false of the system. An unrouted owner-ask is strictly worse
than an `UNDECLARED` one: the deadlock is invisible as well as unarmed.

    UNROUTED-OWNER-ASK  - an item under `## FOR THE OWNER` in PROGRESS.md that
                          reaches no entry in DECISIONS_NEEDED/RESOLVED.
    VANISHED-OWNER-ASK  - an item that was on the PREVIOUS committed page, is
                          not on this one, and never reached one either.

TWO THINGS ABOUT THE CLASSIFIER, both of which are the reason it looks blunt.

**It does not try to decide which items are asks.** The audit that ordered this
check hand-classified the same page twice, one paragraph apart, and disagreed
with itself: a "here is the order I will take Sunday in" item counted as an ask
on 2026-09-03 and the same item counted as scheduling — not an ask — on 09-04.
If the author of the rule cannot apply it consistently to four paragraphs, a
regex will not either, and this file has already recorded twice what happens
when a scanner infers a document's intent. So EVERY numbered item is an ask
until something says otherwise, and the something is a DECLARATION:

    NO-DECISION: liveness report, there is nothing here to rule on

Silence is reported; exemption is written down with a reason. That direction is
deliberate and it is the 2026-09-04 lesson one layer up — a two-set
classification asserted on one side is not a partition, and the SILENT side is
the dangerous one. The cost of the blunt rule is that the Review annotates its
own status paragraphs once each; the cost of the clever one is a recommendation
worth a project stage evaporating in 24 hours, which already happened.

**It matches by QUOTATION, not by similarity.** An ask is routed when a
6-token verbatim span of it appears in `DECISIONS_NEEDED.md` or
`DECISIONS_RESOLVED.md`, and it is still on the page when such a span appears in
today's `PROGRESS.md`. That is not a heuristic chosen for convenience: quoting
the recommendation into the decision entry is what the overseer actually DID
when it lifted this exact scar onto the owner's desk as `D21`, blockquote and
commit citation included. The check therefore rewards the repair the system
already performs, and there is no similarity threshold to tune.

AND THE QUOTATION MATCH MUST BE ATTRIBUTABLE, BECAUSE IT SILENCED A LIVE ASK
FOR NO REASON (2026-09-04, 70th audit finding 2, B2). The first version matched
a span anywhere in either document and printed nothing when it hit. Writing
`D21`'s amendment the overseer quoted the Review's docket sentence **as evidence
of a defect**, routing nothing — and `PROGRESS #3` dropped off the `UNROUTED`
list, the count went 3 -> 2, and the ratchet stayed green. It was caught only
because the author re-ran the tool and missed an item it expected. **The only
symptom was a number going down, which every ratchet in this repo is built to
read as good news.** Two changes:

  - the match is ATTRIBUTED, and this is the half that does the work. `main()`
    prints `matched-by: D21 (quoted)` beside every ask it silenced, so a
    spurious match is a line an auditor reads rather than an absence they must
    notice.
  - the match is CONFINED to an ENTRY of those documents rather than to the
    corpus at large, so the two halves of the same test stop disagreeing: the
    cite half already demanded a heading, the quote half took all 400 KB.
    `_entries()` is now the single definition both read.

B2 ordered the confinement to `## D…` entries and called it the load-bearing
half. It is neither — see `_entries`, which records the measurement: that
boundary would not have caught the scar (the offending quotation was inside
`D21`'s own amendment) and it reports the 69th audit's `run blocked`
disposition, an unnumbered but complete `DECISIONS_RESOLVED.md` entry, as
unrouted. Confinement to an entry is worth 17 lines of 5,261. The attribution
is worth the finding.

THE THIRD GUARD, AND IT IS THE SAME HOLE ONE FIELD OVER (2026-09-04, 70th audit
B1). Everything above checks that a declaration has the right SHAPE — a default
exists, a class is legal, a date parses, a blast radius is survivable. None of
it asks whether the default's own action is still AVAILABLE on the day the
default fires. On 2026-09-04 the overseer armed `D21` with

    default:   … the 2026-09-06 FULL Review takes the W1 design as the FIRST
               item on its docket …
    decide_by: 2026-09-11

and nothing printed. The clock fell five days after the event it commanded: on
the morning it became due it would have ordered a Sunday that had already
happened to re-order its docket. That is `D1`'s deadlock with a clock painted on
it — armed, dated, legal in every field, and incapable of doing the thing it
promises. It was caught by hand the next morning, by the organ that wrote it.

    DEFAULT-ACTION-EXPIRED - the default's prose names a date at or before its
                             own `decide_by`, so the action is in the past on
                             every day the default could fire.

THE ARITHMETIC IS THIS FILE'S OWN, not an inference about English. `main()`
marks a row overdue at `(today - decide_by).days > 0`, so **the earliest day a
default can fire is `decide_by + 1`**. A date at or before `decide_by` is
therefore behind the firing on every branch — including the equality case, which
is why the comparison is `<=` and not `<`. Nothing here reads intent; it reads
one date against another under a rule the module already implements.

Dates are mined out of the joined `default:` text the way `blast_radius` mines
spec ids out of it, and the same over-approximation applies for the same reason:
this reports what a default REFERS to, never what it will do. A default may name
a past date as provenance rather than as an action ("(`D15`) fires on
2026-09-05"), and that reads as a finding here. So the class is RATCHETED, not
blocking — baselined at the live reading, allowed to shrink and never to grow —
because the alternative is a scanner that decides which sentences are commands,
and this file has recorded twice what that costs.

THE ATTRIBUTION THAT NARROWING ASKED FOR NOW EXISTS, BY DECLARATION AND NEVER
BY REGEX (2026-09-05, 72nd audit B2). A provenance date can say whose clock it
is, in the same idiom as `COVERS:` and `DECIDE:`:

    ... a pacing decision (`D15`) that fires on 2026-09-05 (CLOCK: D15) ...
    ... silence through 2026-09-08 (CLOCK: consequence) costs ...

A `CLOCK:` marker claims the NEAREST DATE BEFORE IT and that single occurrence
stops counting in the two calendar checks. Per-occurrence on purpose: a blanket
per-entry exemption would be a checkbox nobody has to justify, and a date named
twice with one attribution is still loud at its unattributed mention. This is
the docstring's own refused-regex resolved the honest way — the corpus reads
zero because its author DECLARED which dates are provenance, not because the
instrument was tuned until it agreed.

AND THE FOURTH CLOCK DEFECT IS THE REPAIR'S OWN RESIDUE (same audit, same
commit). `D21`'s `EXPIRED` repair shortened its clock to 2026-09-05 — the only
direction a deadline may move — and thereby created a default whose commanded
event shares its EARLIEST FIRING DAY: `decide_by + 1` is 09-06 and the Review
it commands sits at 06:37 that morning. A six-hour window, and `main()`'s
date-granular arithmetic cannot express an hour, so no instrument would say a
word if the window were missed. So:

    DEFAULT-ACTION-SAME-DAY - the default names its own earliest firing day,
                              so the firing is a RACE against the event it
                              commands, not a certainty before it.

Reported, never blocking and never ratcheted — deliberately weaker than
`EXPIRED`, because landing here is what the prescribed `EXPIRED` repair DOES:
a wall or a ratchet on this class would punish the legal fix for the class
above it. The report line is the whole product: it tells the firing slot to
treat the deadline as having an HOUR.

The bare `MM-DD` form is matched as well as the ISO one, and that is
load-bearing rather than thorough: `D21`'s default wrote its date BOTH ways, and
the defaults in this document routinely write only the short form. A bare
month-day carries no year, so it resolves to whichever of `decide_by`'s year,
the one before or the one after puts it NEAREST `decide_by` — a rule that reads
"12-28" against a `decide_by` of 2027-01-03 as the December just gone rather
than the one eleven months out.

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
import subprocess
import sys
import textwrap as _textwrap
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
DOC = _REPO / "docs" / "DECISIONS_NEEDED.md"
PROGRESS = _REPO / "docs" / "PROGRESS.md"
RESOLVED = _REPO / "docs" / "DECISIONS_RESOLVED.md"

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


def holds(by_id=None, path: Path = DOC) -> dict:
    """{spec_id: "Dxx (decide_by ...)"} — specs an OPEN decision declares in
    `blocks:`, so an instrument can refuse to advertise them as work.

    Scar (2026-09-03): D19's NO-FETCH default held HR.1 blocked-on-disk, the
    Review wrote "do not fetch a corpus to unblock a family" — and `coverage`
    went on printing `cpu<10min <- fillable today: HR.1`, because the block
    lived in prose and no instrument reads prose. Two consecutive journal
    entries had to hand-warn the next iteration off that one line. Same class
    as HR.5→HR.6 (65th audit B1): a blocker written as a sentence is invisible
    to every ranker until it becomes an edge. `blocks:` was already parsed and
    joined to the graph for COST; this is the same field joined for FILLABILITY.

    Tokens are comma-split and validated against the registry — exactly
    `check()`'s idiom — so a prose value like "nothing. T0.27 has no
    dependents" declares no hold (an inline mention must never look like a
    declaration; the _DECIDE regex learned that the expensive way). Only
    decisions still OPEN in the doc hold anything: a resolved decision's
    `blocks:` is history, not a live refusal.

    Fails loud on an unreadable doc: a silent {} would silently re-advertise
    every held spec, which is the optimistic default this repo keeps paying for.
    """
    if by_id is None:
        from .registry import BY_ID as by_id
    decls, open_ids = parse(Path(path).read_text())
    out: dict = {}
    for did in open_ids:
        d = decls.get(did) or {}
        for tok in (d.get("blocks") or "").split(","):
            sid = tok.strip()
            if sid in by_id:
                out[sid] = f"{did} (decide_by {d.get('decide_by') or 'UNDATED'})"
    return out


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


# A date as a default's prose writes one. Two accepted forms and no others:
# the ISO `2026-09-06` and the bare `09-06` this document uses constantly.
#
# BOTH HALVES ARE TWO-DIGIT ON PURPOSE. `\d{2}-\d{2}` with month 01-12 and day
# 01-31 will not match a range written the way ranges are written here ("8-12
# sigma", "1-3 seeds"), and will not match `d10-*`, `2026-W35` or a thousands
# separator. It WILL match a genuine two-digit range like "11-12", and that is
# the over-approximation the ratchet exists to hold — see the module docstring.
_PROSE_DATE = re.compile(
    r"\b(?:(20\d\d)-)?(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")


# The attribution marker (72nd audit B2). Same idiom as `COVERS:`/`DECIDE:`:
# a declaration, uppercase, with a label — `(CLOCK: D15)`, `(CLOCK:
# consequence)`. The label is free-form on purpose: this file must not decide
# which clocks are legitimate, only that the author named one.
_CLOCK = re.compile(r"CLOCK:\s*([A-Za-z0-9._-]+)")


def _resolved_dates(default_text: str, decide_by: _dt.date) -> list:
    """`[(end_pos, date)]` — every date OCCURRENCE in text order.

    A bare `MM-DD` has no year, so it takes whichever of `decide_by.year - 1 /
    +0 / +1` lands NEAREST `decide_by`; an impossible combination (02-30) is
    not a date and is dropped. Positions are kept because `CLOCK:` attribution
    is per-occurrence — see `unattributed_dates`.
    """
    out: list = []
    for m in _PROSE_DATE.finditer(default_text or ""):
        yr, mo, da = m.groups()
        if yr:
            try:
                out.append((m.end(), _dt.date(int(yr), int(mo), int(da))))
            except ValueError:
                continue
            continue
        cands = []
        for y in (decide_by.year - 1, decide_by.year, decide_by.year + 1):
            try:
                cands.append(_dt.date(y, int(mo), int(da)))
            except ValueError:
                continue
        if cands:
            out.append((m.end(),
                        min(cands, key=lambda d: abs((d - decide_by).days))))
    return out


def default_dates(default_text: str, decide_by: _dt.date) -> list:
    """Dates a default's prose names, resolved against its own `decide_by`.

    Sorted, deduplicated, never raising. Same over-approximating contract as
    `blast_radius`: this says what the text REFERS to, never what the default
    will do — attribution does not subtract here, because "what the text
    refers to" includes the dates it disclaims.
    """
    return sorted({d for _, d in _resolved_dates(default_text, decide_by)})


def unattributed_dates(default_text: str, decide_by: _dt.date) -> list:
    """Dates with no `CLOCK:` declaration claiming them — the COMMAND candidates.

    A `CLOCK:` marker claims the nearest date occurrence BEFORE it and only
    that occurrence; a marker with no preceding date claims nothing. The
    over-approximation still runs the safe way: a date named twice with one
    attribution counts, because its unattributed mention is still a command
    candidate, and a marker can never silence more than the one occurrence its
    author put it beside.
    """
    recs = [[pos, d, False] for pos, d in
            _resolved_dates(default_text, decide_by)]
    for cm in _CLOCK.finditer(default_text or ""):
        prior = [r for r in recs if r[0] <= cm.start()]
        if prior:
            prior[-1][2] = True
    return sorted({d for _, d, claimed in recs if not claimed})


def expired_actions(default_text: str, decide_by: _dt.date) -> list:
    """Unattributed dates in a default that are behind its EARLIEST firing.

    `main()` marks a row overdue at `(today - decide_by).days > 0`, so the
    earliest fire is `decide_by + 1` and a date at or before `decide_by` cannot
    be acted on by any firing of this default. The `<=` is that arithmetic, not
    a margin. A date carrying a `CLOCK:` declaration is somebody else's and is
    not consulted — that is the attribution the ratchet's comment promised.
    """
    return [d for d in unattributed_dates(default_text, decide_by)
            if d <= decide_by]


def same_day_actions(default_text: str, decide_by: _dt.date) -> list:
    """Unattributed dates that EQUAL the default's earliest firing day.

    On that day the action is a race, not a certainty: the default fires some
    time after 00:00 and the event it commands happens at some hour of the same
    date, and nothing in this date-granular file can order the two. `D21` is
    the live positive — its commanded Review sits at 06:37 on `decide_by + 1`,
    a six-hour window no instrument could see until this class.
    """
    fire = decide_by + _dt.timedelta(days=1)
    return [d for d in unattributed_dates(default_text, decide_by)
            if d == fire]


# The live reading in the commit that shipped this check, in the
# `BASELINE_UNDECLARED` idiom: it may SHRINK and may never GROW. It shipped at
# 1 — `D22`, whose default cites `D15`'s 2026-09-05 clock as a REASON rather
# than performing anything on it — and shrank to 0 on 2026-09-05 (72nd audit
# B2) when the attribution the old comment called "not built yet" was built
# and `D22` DECLARED both of its dates provenance with `(CLOCK: D15)` and
# `(CLOCK: consequence)`. Shrink by declaration, exactly as prescribed: the
# entry says whose clocks those are, and the instrument was not tuned.
BASELINE_ACTION_EXPIRED = 0


# ── the owner's OTHER desk: `## FOR THE OWNER` in docs/PROGRESS.md ──────────
#
# Start-of-line only, like every declaration in this repo. An item is a numbered
# markdown entry; it runs until the next numbered entry or the end of the
# section. `---` and the next `## ` both close the section, because the Review
# writes both.
_OWNER_HEADING = re.compile(r"^##\s+FOR THE OWNER\s*$", re.M)
_ITEM = re.compile(r"^(\d{1,2})\.\s+(.*)$")
_SECTION_END = re.compile(r"^(##\s|---\s*$)")

# The exemption, and it must carry a reason: an escape hatch nobody has to
# justify is a checkbox, and a checkbox is how `RUNNER_OUTPUTS` and the P10
# partition both went one-sided.
_NO_DECISION = re.compile(r"^\s*NO-DECISION:\s*(\S.*?)\s*$", re.M)

# A decision id as it is cited in prose. `D1`..`D999`; deliberately narrower
# than `_SPEC_ID` so "D" alone or a word starting with D cannot look like a cite.
#
# THE LOOKAHEAD IS LOAD-BEARING AND IT WAS FOUND BY THE FIXTURE'S LIVE HALF.
# `D1.0` is a SPEC — the control-path bakeoff — and `\bD1\b` matches inside it,
# because the `.` is a word boundary. Without `(?!\.\d)` the 2026-09-03 ask
# "`run blocked` cannot see the project's largest unblock" read as ROUTED TO
# `D1`, purely because its prose mentioned `D1.0` twice, and the one true
# positive this check was built for went silently quiet. That is this file's own
# `_DECIDE` scar with the numbers swapped: an inline mention must never be able
# to look like a declaration.
_CITE = re.compile(r"\bD(\d{1,3})\b(?!\.\d)")

# Verbatim span length for the quotation match. Six tokens is long enough that
# ordinary English does not collide (the live corpus is ~400 KB and produces no
# accidental match between any pair of the four live items) and short enough
# that a re-worded paragraph keeping one clause still reads as the same item.
SHINGLE_N = 6

# Ratchets, one per class, in the BASELINE_UNDECLARED idiom: set to the LIVE
# reading on the day the check shipped, may shrink, may never grow. A guard that
# fails everywhere on day one is one nobody keeps green.
#
# UNROUTED was 3 of 4 items on the 2026-09-04 page. Two of the three are the
# Review's standing status paragraphs (Sunday order; organ liveness) and shrink
# the moment it writes `NO-DECISION:` on them; the third is the draft-then-
# ratify recommendation, which is a real unrouted ask — the SAME defect `D21`
# was created for, recurring the next day on the same page.
BASELINE_UNROUTED_ASKS = 3
# VANISHED was 1 (the 09-03 `run blocked` recommendation, which the overseer
# ruled builder-work and the builder implemented, with no durable record on any
# owner-readable page) and is 0 once that disposition is recorded.
BASELINE_VANISHED_ASKS = 0


def _tokens(text: str) -> list:
    """Markdown-blind lowercase word tokens. Backticks, bold, punctuation and
    hyphens all vanish, so `**\\`d10-*\\` gate rows**` and "d10 gate rows" are the
    same six words — the Review reformats constantly and formatting is not
    content."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split()


def _shingles(text: str, n: int = SHINGLE_N) -> set:
    t = _tokens(text)
    return {" ".join(t[i:i + n]) for i in range(len(t) - n + 1)}


def owner_asks(progress_text: str) -> list:
    """The numbered items under `## FOR THE OWNER`, in order.

    `{n, text, lead, exempt, reason, cites, shingles}`. No attempt is made to
    judge which items are questions — see the module docstring; the tool that
    ordered this check could not do it consistently either.
    """
    m = _OWNER_HEADING.search(progress_text or "")
    if not m:
        return []
    lines = progress_text[m.end():].splitlines()
    items: list = []
    cur: dict | None = None
    for ln in lines:
        if _SECTION_END.match(ln):
            break
        head = _ITEM.match(ln)
        if head:
            cur = {"n": int(head.group(1)), "lines": [head.group(2)]}
            items.append(cur)
        elif cur is not None:
            cur["lines"].append(ln)
    out = []
    for it in items:
        text = "\n".join(it["lines"])
        ex = _NO_DECISION.search(text)
        bold = re.search(r"\*\*(.+?)\*\*", text, re.S)
        lead = re.sub(r"\s+", " ", (bold.group(1) if bold else text)).strip()
        out.append({"n": it["n"], "text": text, "lead": lead[:96],
                    "exempt": bool(ex), "reason": ex.group(1) if ex else "",
                    "cites": {f"D{d}" for d in _CITE.findall(text)},
                    "shingles": _shingles(text)})
    return out


def _entries(*texts: str) -> dict:
    """`{key: entry text}` — the decision documents cut into entries.

    A top-level `##` heading opens an entry and the next one closes it; `###`
    sub-headings do not (the live document uses them inside entries). The key is
    the decision id when the heading carries one and the heading text otherwise,
    because **a desk does not need a number** — see the boundary note below.
    Repeated ids accumulate: `D1`, `D2` and `D3` each carry several dated
    correction blocks and every one of them is that decision's desk.

    This is the SINGLE definition both halves of `_reaches_a_desk` read. Before
    2026-09-04 the citation half demanded a heading here while the quotation
    half accepted a span anywhere in either file — one test, two boundaries, and
    the loose one silenced a live ask.

    THE BOUNDARY IS NOT THE ONE THE AUDIT ORDERED, AND THE AUDIT'S OWN EVIDENCE
    IS WHY (builder, 2026-09-04, building 70th audit B2). B2 says *"require the
    matching span to fall inside a `## D…` entry"* and calls it the load-bearing
    half. Both halves of that are checkable and both come out the other way:

      - **It would not have caught the scar.** The quotation that silenced
        `PROGRESS #3` was written INSIDE `D21`'s own `AMENDED` block — a `## D…`
        entry by any reading. Confinement cannot see a misuse that happens
        inside a legitimate desk; only the attribution can, which is why the
        report half is the one that matters.
      - **It reports a real disposition as unrouted.** Measured on the live
        corpus at implementation time: `## D…` confinement moves
        `VANISHED-OWNER-ASK` 0 -> 1 against a baseline of 0 and STOPS the gate,
        and the single positive is `docs/DECISIONS_RESOLVED.md:501` — the 09-03
        `run blocked` ask, ruled NOT-THE-OWNER'S by the 69th audit and recorded
        with its losers and a reversal line. It has no `D` number because it was
        never numbered, and 20 of the 104 entries in these two files are like
        it. The only repairs available for that false positive are to lower a
        shrink-only baseline or to back-number a historical entry to satisfy an
        instrument — the tool editing the record to agree with itself.

    So the confinement is to an ENTRY, not to a NUMBERED entry, and it is worth
    stating what that buys rather than implying more: **17 lines of 5,261.**
    Almost the whole corpus is inside some entry already. The quotation match
    was never mostly loose; it was mostly UNATTRIBUTED, and `owner_ask_silences`
    is the repair.
    """
    out: dict = {}
    for t in texts:
        cur = None
        for line in (t or "").splitlines():
            if line.startswith("##") and not line.startswith("###"):
                mm = _CITE.search(line[:24])
                cur = f"D{mm.group(1)}" if mm else line.lstrip("# ").strip()[:60]
            if cur:
                out[cur] = out.get(cur, "") + line + "\n"
    return out


def _entry_key(key: str) -> tuple:
    """Sort order for attribution: numbered desks first and in numeric order,
    then the unnumbered ones alphabetically. Determinism only — an ask that
    reaches two desks has reached a desk."""
    if re.fullmatch(r"D\d{1,3}", key):
        return (0, int(key[1:]), "")
    return (1, 0, key)


def _decision_ids(*texts: str) -> set:
    """`D` ids that actually have an entry — a heading, at the start of a line.
    An id cited in prose that resolves to no entry is a typo, exactly as an
    unresolvable spec id is in `blast_radius`."""
    return {k for k in _entries(*texts) if re.fullmatch(r"D\d{1,3}", k)}


def _reaches_a_desk(ask: dict, entries: dict) -> tuple | None:
    """`(how, D-id)` when this ask has a durable home, else `None`.

    It cites a decision that exists, or a verbatim span of it is quoted INSIDE
    one — the repair the overseer performed by hand for `D21`. The return value
    names the entry rather than reporting a bare boolean, because the silencing
    is the dangerous direction: an ask that goes quiet for the wrong reason is
    invisible unless the report can say who quieted it.

    Ties are broken by decision number so the attribution is deterministic; an
    ask that reaches two desks has reached a desk, and which one it names first
    is not a fact worth making order-dependent.
    """
    cited = sorted(ask["cites"] & set(entries), key=_entry_key)
    if cited:
        return ("cites", cited[0])
    for did in sorted(entries, key=_entry_key):
        if ask["shingles"] & _shingles(entries[did]):
            return ("quoted", did)
    return None


def owner_ask_silences(progress_text: str, needed_text: str,
                       resolved_text: str = "") -> list:
    """`[(key, lead, how, D-id)]` — every ask this pass did NOT report, and why.

    The report half of the 70th audit's finding 2. `main()` prints these so that
    a quotation made for some unrelated purpose shows up as `matched-by: D21`
    instead of as an item quietly missing from a list. Exempt items are not
    here: a `NO-DECISION:` line is a declaration the Review wrote on purpose and
    is already visible in the page it wrote it on.
    """
    entries = _entries(needed_text, resolved_text)
    out = []
    for a in owner_asks(progress_text):
        if a["exempt"]:
            continue
        hit = _reaches_a_desk(a, entries)
        if hit:
            out.append((f"PROGRESS #{a['n']}", a["lead"], hit[0], hit[1]))
    return out


def owner_ask_findings(progress_text: str, prev_progress_text: str | None,
                       needed_text: str, resolved_text: str = "") -> list:
    """`[(kind, key, why)]` for the two owner-ask classes.

    `prev_progress_text is None` means git could not supply a baseline, and an
    absent baseline must never manufacture a violation — `review_queue.py`'s
    rule for `VANISHED`, for the same reason.
    """
    entries = _entries(needed_text, resolved_text)
    out = []

    for a in owner_asks(progress_text):
        if a["exempt"] or _reaches_a_desk(a, entries):
            continue
        out.append(("UNROUTED-OWNER-ASK", f"PROGRESS #{a['n']}",
                    f"{a['lead']} — on the owner's page, in no decision file: "
                    "no class, no default, no decide_by, and the page is "
                    "rewritten daily. Repair: route it (quoting it, as D21 "
                    "does) or declare `NO-DECISION: <reason>`"))

    if prev_progress_text is None:
        return out
    here = _shingles(progress_text)
    for a in owner_asks(prev_progress_text):
        if a["exempt"] or a["shingles"] & here:
            continue                       # answered-shape, or still on the page
        if _reaches_a_desk(a, entries):
            continue
        out.append(("VANISHED-OWNER-ASK", f"PROGRESS(prev) #{a['n']}",
                    f"{a['lead']} — was on the previous committed page, is not "
                    "on this one, and reached no decision file. An ask that "
                    "rolls off a current-state page is not an ask that was "
                    "answered. Repair: record its disposition in "
                    "DECISIONS_NEEDED.md or DECISIONS_RESOLVED.md"))
    return out


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
          by_id=None, progress_text=None, prev_progress_text=None,
          resolved_text: str = "") -> tuple[list, list]:
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

        # The action must still exist on the day the default fires. Reported,
        # never fatal, and the row survives it: the decision is still armed and
        # its cost is still real — what is broken is the clock, not the arming.
        stale = expired_actions(d["default"], due)
        if stale:
            violations.append((
                "DEFAULT-ACTION-EXPIRED", did,
                f"the default names {', '.join(d.isoformat() for d in stale)} "
                f"but decide_by is {due} and the earliest firing is "
                f"{due + _dt.timedelta(days=1)} — on the day this fires, that "
                "action is in the past. Repair: SHORTEN decide_by (a deadline "
                "may tighten on its own; it may never be lengthened), or, if "
                "the date is not this default's to act on, declare whose it "
                "is with `(CLOCK: <whose>)` beside that date — a provenance "
                "date that says so stops reading as a command"))

        # The race the EXPIRED repair leaves behind: shortening a clock to the
        # day before its action puts the action ON the earliest firing day.
        # Reported, never blocking, never ratcheted — see the class docstring.
        race = same_day_actions(d["default"], due)
        if race:
            violations.append((
                "DEFAULT-ACTION-SAME-DAY", did,
                f"the default names {', '.join(x.isoformat() for x in race)}, "
                f"which is its own earliest firing day ({due} + 1 day) — this "
                "default must fire BEFORE the event it commands, on the same "
                "day. The deadline has an HOUR; nothing date-granular can "
                "enforce it, so the firing slot must know it is in a race"))

        blocks = [b.strip() for b in (d.get("blocks") or "").split(",") if b.strip()]
        n, _which = cost_of(blocks)
        rows.append({"id": did, "due": due, "overdue": (today - due).days,
                     "blocks": blocks, "cost": n,
                     "default": d.get("default", ""), "expired": stale,
                     "same_day": race})

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

    # The owner's other desk. `progress_text is None` means the caller is
    # driving this function on a document pair it built itself (every T0.28
    # fixture, and the pre-2026-09-04 organ), so the pass simply does not run —
    # an absent input must never manufacture a violation.
    if progress_text is not None:
        violations.extend(owner_ask_findings(progress_text, prev_progress_text,
                                             text, resolved_text))
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


# The ratcheted classes: `{kind: baseline}`. Each may shrink and may never grow.
# They are counted rather than blocked because each is a BACKLOG — a queue, not
# a wall — and because a wall here would forbid a legal move (the Review is
# allowed to write a status paragraph; the owner is allowed to be slow).
RATCHETED = {
    "UNDECLARED": BASELINE_UNDECLARED,
    "UNROUTED-OWNER-ASK": BASELINE_UNROUTED_ASKS,
    "VANISHED-OWNER-ASK": BASELINE_VANISHED_ASKS,
    "DEFAULT-ACTION-EXPIRED": BASELINE_ACTION_EXPIRED,
}


def ratchet_debt(violations: list) -> dict:
    """`{kind: (count, baseline)}` for every ratcheted class, always all of
    them — including the zeroes. A counter that only appears when it is nonzero
    cannot be seen to be at floor, and this repo has twice paid for a number
    that was invisible while it was fine."""
    return {k: (sum(1 for kind, _, _ in violations if kind == k), base)
            for k, base in RATCHETED.items()}


def check_rc(violations: list) -> int:
    """The `--check` exit code, as a function of the violations alone.

    Extracted from `main` so the certificate can assert the EXIT CODE rather
    than a re-implementation of it. A test that reproduces the gate's logic
    proves the copy agrees with itself.
    """
    if any(n > base for n, base in ratchet_debt(violations).values()):
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

    # `D21` as the 69th audit actually armed it: an action on 09-06 behind a
    # clock on 09-11. Both spellings of the date are present because the real
    # entry wrote both, and the ISO-only reader would have caught it by luck.
    d21 = """
## D21 — the W1 recommendation, lifted onto the owner's desk (OPEN, owner)

DECIDE: D21
  class:     goal
  default:   the 2026-09-06 FULL Review takes the W1 design as the FIRST item
             on its docket, and `w0-too-shallow` is already dated 09-06 so this
             re-orders a scheduled item and creates no new permission.
  decide_by: 2026-09-11
"""
    kinds21 = {did: kind for kind, did, _ in audit(d21, _dt.date(2026, 9, 4))[0]}
    assert kinds21.get("D21") == "DEFAULT-ACTION-EXPIRED", kinds21
    # ...and the repair the 70th audit performed — SHORTEN the clock so the
    # firing lands the morning before the action — silences EXPIRED and leaves
    # exactly the SAME-DAY race behind, because the shortened clock puts the
    # commanded 09-06 event ON the earliest firing day. That residue is the
    # 72nd audit's finding, and it must be the ONLY thing left.
    short = audit(d21.replace("2026-09-11", "2026-09-05"),
                  _dt.date(2026, 9, 4))[0]
    assert {k for k, _, _ in short} == {"DEFAULT-ACTION-SAME-DAY"}, short
    # The equality case is the same defect: the earliest fire is decide_by + 1,
    # so an action dated ON decide_by has already passed when the clock rings.
    assert expired_actions("act on 2026-09-05", _dt.date(2026, 9, 5))
    assert not expired_actions("act on 2026-09-06", _dt.date(2026, 9, 5))
    # A bare month-day takes the NEAREST year, not decide_by's blindly.
    assert default_dates("by 12-28", _dt.date(2027, 1, 3)) == [_dt.date(2026, 12, 28)]
    # CLOCK attribution (72nd audit B2): a declared provenance date is not a
    # command — and the declaration is per-occurrence, never blanket.
    assert expired_actions("(`D15`) fires on 2026-09-05",
                           _dt.date(2026, 9, 8))
    assert not expired_actions("(`D15`) fires on 2026-09-05 (CLOCK: D15)",
                               _dt.date(2026, 9, 8))
    assert expired_actions("2026-09-05 (CLOCK: D15) and also 2026-09-04",
                           _dt.date(2026, 9, 8)) == [_dt.date(2026, 9, 4)]
    # ...and default_dates still reports the disclaimed date: what the text
    # REFERS to is not narrowed by whose clock it is.
    assert default_dates("2026-09-05 (CLOCK: D15)",
                         _dt.date(2026, 9, 8)) == [_dt.date(2026, 9, 5)]
    # SAME-DAY: equality with the earliest firing day, silenced the same way.
    assert same_day_actions("act on 2026-09-06",
                            _dt.date(2026, 9, 5)) == [_dt.date(2026, 9, 6)]
    assert not same_day_actions("on 2026-09-06 (CLOCK: elsewhere)",
                                _dt.date(2026, 9, 5))
    # A marker with no preceding date claims nothing — it must not eat the
    # date that comes AFTER it.
    assert expired_actions("(CLOCK: D15) then 2026-09-04",
                           _dt.date(2026, 9, 8)) == [_dt.date(2026, 9, 4)]


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


def _previous_page(path: Path) -> str | None:
    """The last committed version of `path` that is not the one on disk now.

    Same channel as `review_queue._prev_revision` and for the same reason: git
    is the one baseline a working-tree edit cannot reach. `None` when git cannot
    say — a brand-new file, no repo, a detached environment — and `None` makes
    the git-baselined class simply not fire.

    "Not the one on disk now" matters: `PROGRESS.md` is rewritten wholesale by
    the Review, so the interesting comparison is against the PREVIOUS Review's
    page, which is HEAD's version when the tree is dirty and HEAD~ when it is
    clean. Taking `HEAD:` unconditionally would compare the page to itself on
    every committed run — a baseline that is always identical is not a baseline.
    """
    try:
        rel = path.relative_to(_REPO).as_posix()
        log = subprocess.run(["git", "log", "-3", "--format=%H", "--", rel],
                             cwd=_REPO, capture_output=True, text=True, timeout=20)
        if log.returncode != 0:
            return None
        here = path.read_text() if path.exists() else ""
        for sha in log.stdout.split():
            r = subprocess.run(["git", "show", f"{sha}:{rel}"], cwd=_REPO,
                               capture_output=True, text=True, timeout=20)
            if r.returncode == 0 and r.stdout != here:
                return r.stdout
    except Exception:
        return None
    return None


def _ask_fixture() -> None:
    """The planted positives for the two owner-ask classes, on the real path.

    Not invented: item 2 IS the 2026-09-03 Review's W1 recommendation, which
    left the page unanswered at 24 hours old and had to be lifted onto the
    owner's desk by hand as `D21`. Both directions are asserted — the loud one
    and the quiet one — because a scanner that has only been seen firing is half
    tested, and this repo's own `RUNNER_OUTPUTS` and `T0.17 P10` scars are both
    one-sided partitions that stayed green while saying nothing.
    """
    yesterday = """
## FOR THE OWNER

1. **Sunday is oversubscribed and here is the order I will take it in.** Six
   rows come due on the same run that owes Part 2.

2. **My recommendation: W1 stops being a queue row and becomes the project's
   stated stage.** This is the strategic fork.

3. **Organ liveness, all green.** builder 06:07, overseer 06:37, no organ is
   silent.
   NO-DECISION: liveness report, nothing here to rule on

---
"""
    today = """
## FOR THE OWNER

1. **Sunday is oversubscribed and here is the order I will take it in.** The
   two gate rows first, then the rest.

2. **The meter's input, measured rather than argued.** See `D20`.

3. **A brand-new ask with nowhere to live.** Nothing points at this one.

## NEXT SECTION
4. **Not an owner ask at all — outside the section.** Must be invisible.
"""
    needed = "## D20 — what should the ceiling count?\n"
    got = owner_ask_findings(today, yesterday, needed)
    kinds = {(k, key.split("#")[1]) for k, key, _ in got}

    # UNROUTED: item 3 only. Item 1 is quoted nowhere but is still on the page —
    # that is not what UNROUTED means, and it fires anyway, so assert it does:
    # a status paragraph the Review has not annotated IS the reported state.
    assert ("UNROUTED-OWNER-ASK", "1") in kinds, got     # unannotated: reported
    assert ("UNROUTED-OWNER-ASK", "3") in kinds, got     # the new one
    assert ("UNROUTED-OWNER-ASK", "2") not in kinds, got  # cites a live D20
    assert ("UNROUTED-OWNER-ASK", "4") not in kinds, got  # outside the section

    # VANISHED: yesterday's item 2 and only it. Item 1 survives by quotation
    # ("oversubscribed and here is the order i will take it in") though the
    # paragraph was rewritten; item 3 declared itself exempt.
    assert ("VANISHED-OWNER-ASK", "2") in kinds, got
    assert not [k for k in kinds if k[0] == "VANISHED-OWNER-ASK" and k[1] != "2"], got

    # And it must go quiet for the RIGHT reason: quoting the vanished ask into a
    # decision document silences it — the repair `D21` actually performed —
    # while deleting the ask from the record does NOT, because the previous page
    # is git and nothing a later edit does can reach it.
    routed = needed + ("> *\"My recommendation: W1 stops being a queue row and "
                       "becomes the project's stated stage.\"*\n")
    assert not [k for k in owner_ask_findings(today, yesterday, routed)
                if k[0] == "VANISHED-OWNER-ASK"], "quoting must silence it"

    # An id cited in prose that resolves to no entry is a typo, not a route —
    # the `blast_radius` rule, applied to decisions instead of specs.
    assert [k for k, _, _ in owner_ask_findings(
        "## FOR THE OWNER\n\n1. **See `D99`.** It does not exist.\n",
        None, needed)] == ["UNROUTED-OWNER-ASK"]

    # AND A SPEC ID IS NOT A DECISION CITE. `D1.0` contains `D1`, `D20` does
    # not contain `D2`. The first of those silenced the live true positive; the
    # second would silence one tomorrow. Both directions, on the real path.
    spec_cite = ("## FOR THE OWNER\n\n1. **`run blocked` cannot see it.** The "
                 "repair runs through `D1.0`; nothing declares that edge.\n")
    assert [k for k, _, _ in owner_ask_findings(
        spec_cite, None, "## D1 — the control path\n")] == ["UNROUTED-OWNER-ASK"]
    assert not owner_ask_findings(
        "## FOR THE OWNER\n\n1. **See `D20`.**\n", None, needed)

    # No baseline must never manufacture a violation (review_queue's rule), and
    # a page with no FOR THE OWNER section is not a page full of hidden asks.
    assert not [k for k, _, _ in owner_ask_findings(today, None, needed)
                if k == "VANISHED-OWNER-ASK"]
    assert owner_asks("## SOMETHING ELSE\n\n1. **x**\n") == []

    # THE SILENCING IS ATTRIBUTED (70th audit B2, and this half is the finding).
    # Item 2 above is quiet because it cites `D20`; the report must be able to
    # SAY so, because the scar's only symptom was a number going down.
    assert [(key, how, did) for key, _lead, how, did in
            owner_ask_silences(today, needed)] == [("PROGRESS #2", "cites",
                                                    "D20")]
    quote = ("> *\"A brand-new ask with nowhere to live. Nothing points at "
             "this one.\"*\n")
    _sil = lambda n: [(k, h, d) for k, _l, h, d in owner_ask_silences(today, n)]
    _unrouted = lambda n: {key.split("#")[1] for k, key, _ in
                           owner_ask_findings(today, None, n)
                           if k == "UNROUTED-OWNER-ASK"}
    assert ("PROGRESS #3", "quoted", "D20") in _sil(needed + quote)
    assert "3" not in _unrouted(needed + quote)

    # AN UNNUMBERED ENTRY IS STILL A DESK, and this is a live positive, not an
    # invention: `DECISIONS_RESOLVED.md:501` rules the 09-03 `run blocked` ask
    # NOT-THE-OWNER'S with its losers and a reversal line, under a heading that
    # carries no `D` number because the ask was never numbered. Confining to
    # `## D…` — which is what B2 ordered — reports that as unrouted, and 20 of
    # the 104 live entries are shaped like it. It is attributed by heading.
    unnumbered = "## `run blocked` — RESOLVED AS NOT-THE-OWNER'S\n" + quote
    assert ("PROGRESS #3", "quoted",
            "`run blocked` — RESOLVED AS NOT-THE-OWNER'S") in _sil(unnumbered)
    assert "3" not in _unrouted(unnumbered)

    # ...but a span in no entry at all reaches nothing. This is the whole of
    # what the confinement buys — 17 lines of the live 5,261 — and it is stated
    # here at its real size rather than at the size B2 assumed.
    assert "3" in _unrouted(quote + needed), "outside every entry is no desk"


def main(argv: list[str]) -> int:
    _fixture()
    _safety_fixture()
    _ask_fixture()
    text = DOC.read_text()
    today = _dt.date.today()
    progress_text = PROGRESS.read_text() if PROGRESS.exists() else None
    resolved_text = RESOLVED.read_text() if RESOLVED.exists() else ""
    violations, rows = audit(
        text, today,
        progress_text=progress_text,
        prev_progress_text=_previous_page(PROGRESS),
        resolved_text=resolved_text)

    print(f"\nOpen decisions — {DOC.relative_to(DOC.parent.parent)}\n")
    if rows:
        print("  armed (default fires if unanswered):")
        for r in sorted(rows, key=lambda r: -r["cost"]):
            due = "OVERDUE — DEFAULT IS DUE TO FIRE" if r["overdue"] > 0 else f"due {r['due']}"
            # A clock defect belongs beside the clock, not only in the
            # violation list — the row is what an owner's eye lands on.
            if r.get("expired"):
                due += "   [ACTION EXPIRED: names " + ", ".join(
                    d.isoformat() for d in r["expired"]) + "]"
            if r.get("same_day"):
                due += "   [SAME-DAY RACE: must fire before its own " + ", ".join(
                    d.isoformat() for d in r["same_day"]) + " event]"
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

    # THE SILENCED HALF, printed because its only symptom was a number going
    # down (70th audit finding 2). Every ask this pass declined to report, and
    # the entry that quieted it. A quotation made as EVIDENCE looks exactly like
    # a quotation made as ROUTING to a shingle set, so the tool cannot tell them
    # apart — but a reader can, in one line, and could not before.
    if progress_text is not None:
        silenced = owner_ask_silences(progress_text, text, resolved_text)
        if silenced:
            print(f"  {len(silenced)} owner-ask(s) reached a desk and are NOT "
                  "reported above — check the attribution:")
            for key, lead, how, did in silenced:
                print(f"    {key:<13} matched-by: {did} ({how})")
                print(f"       {lead}")
            print()

    debt = ratchet_debt(violations)
    if "--check" in argv:
        rc = check_rc(violations)          # the gate's verdict, computed once
        broken = {k: v for k, v in debt.items() if v[0] > v[1]}
        if broken:
            for k, (n, base) in sorted(broken.items()):
                print(f"  RATCHET BROKEN: {n} {k}, baseline {base}. "
                      "It may shrink, never grow.")
            print()
        elif rc:
            blocking = [v for v in violations if v[0] in BLOCKING]
            print(f"  {len(blocking)} hard violation(s) — see above.\n")
        else:
            print("  ratchet ok (" + ", ".join(
                f"{n}/{base} {k.lower()}" for k, (n, base) in debt.items()) + ").\n")
        return rc
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
