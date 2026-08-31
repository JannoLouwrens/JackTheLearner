"""A seat whose arena does not exist is not a contest — it is a title with no ring.

WHY THIS EXISTS. `docs/CHAMPIONS.md` is the file that says who holds each seat in
Jack's anatomy and by what right. Its whole mechanism rests on rule 3 — *"every
seat names its ARENA, the spec that decides it"* — and on the idiom of rule 1:
**held BY VERDICT or BY DEFAULT, and the difference is marked; a default marking
is an open invitation to challengers, not a title.** That is a good mechanism.
The invitation is what makes a default honest: the seat is admitted to be
unearned, and anyone may come take it by running the named spec.

An audit on 2026-08-24 found the invitation is, in several places, addressed to
a room that was never built. Resolving every arena id in the file against
`registry.BY_ID` (179 specs at the time, and growing by the hour) — 8 of the 21
seats name an arena the registry has never heard of:

    Control architecture (D1)   arena `D1.0`, `T2.21`   NEITHER EXISTS
    Curiosity signal            arena `LT.03`/`LT.04`   NEITHER EXISTS
    Language model              arena `LG.00`           DOES NOT EXIST
    Language acquisition        arena `LG.00`           DOES NOT EXIST
    Vision encoder              arena `PL.02`           DOES NOT EXIST
    Audio encoder               arena `PL.*`            THE FAMILY IS EMPTY
    PLASTIC-ONLY decree         `PL.00`, `PL.02`        NEITHER EXISTS
    World                       arena `W.1`–`W.7`       NONE OF THE SEVEN EXISTS

`D1.0` and `T2.21` are the control-architecture bakeoff — fully designed in
`docs/research/D1_CONTROL_ARCHITECTURE.md`, four arms, never registered. `LG.00`
is the anti-puppet falsifier that three governing documents cite as the guard on
the LLM-as-parent decree. `PL.00` is the number the PLASTIC-ONLY decree's own
pre-registered RE-OPEN TRIGGER is keyed to — *"if a from-scratch encoder cannot
hit the PL.00 throughput floor"* — and `PL.00` has no floor, because `PL.00` has
no spec. A re-open trigger keyed to a spec nobody wrote is not a trigger; it is
a sentence about a trigger.

THE FAILURE MODE IS SPECIFIC AND IT IS NOT LAZINESS. Every one of those cells
reads as *work queued*: `(queued)`, `(pending registration)`, `(registered
2026-08-10)`. A reader — human, builder, reviewer, or the field watch aiming a
nomination at a seat by name — sees a named arena and concludes the seat is
contestable. `run blocked` cannot correct them: a spec that was never registered
blocks nothing, ranks nowhere, frees nothing, and fails no gate. It is the exact
shape of `coverage.py`'s scar 2 one document over — **credit nobody audits,
because nobody goes looking for contestability they believe they already have.**
A seat that LOOKS contestable and is not is worse than a seat openly marked
VACANT, because VACANT recruits challengers and a phantom arena repels them.

WHAT THIS FLAGS.

    ARENA-MISSING  the seat names an arena spec id that is not in BY_ID. The
                   headline defect: the invitation names a room with no door.
    NO-ARENA       the seat names no arena spec id at all — `HR bakeoff
                   (queued)`, `LG bakeoff (queued)`. Nothing could ever unseat
                   the holder, so rule 3 is not merely unmet, it is unmeetable.
    UNCONTESTED    held BY DEFAULT or BY DECREE, arena EXISTS, and has never
                   run. Not a defect in the document — a debt in the world. It
                   is the only one of the three a bakeoff can pay off.
    ARENA-UNREACHABLE  the arena EXISTS — every id resolves in BY_ID — and
                   every member that has not yet delivered a verdict is
                   parked, VOID-FORECLOSED, or transitively behind one, so no
                   run that could still happen can ever contest the seat. The
                   54th audit (2026-08-31) found two seats in this state
                   behind a green ratchet: existence and reachability are
                   different questions, and every instrument here had only
                   ever asked the first (LESSONS.md, "An arena that exists is
                   not an arena that can be run"). The repair is a
                   re-parenting or a redesign, never a re-run.

MARKINGS AND ARENAS ARE NOW DECLARED, NOT INFERRED — the repair this docstring
proposed for eight audits and would not take unilaterally (51st audit B2, RANK 2,
which authorises it: *"per-seat `HELD:`/`ARENA:` markers in the same idiom as
`COVERS:` and `DECIDE:`, so the parser resolves instead of inferring from table
structure. Take it."*). `docs/CHAMPIONS.md` carries one line per seat:

    - SEAT: Vision encoder | HELD: BY DEFAULT | ARENA: T2.03, PL.02

A declaration WINS over the inference. The inference is kept beside it
(`held_inferred`, `arena_refs_inferred`) and every disagreement is printed,
because the point is not to replace one guess with a quieter one — it is to make
the places the guess was wrong visible in one column.

WHAT THE INFERENCE ACTUALLY COST, measured the hour the syntax landed. The
PLASTIC-ONLY decree's arena read `PL.* LOUD.* PL.00 PL.02 PG.1`. `LOUD.*` came
from this English sentence in the decree's own body:

    **AND THE FALSIFIER IS NOW BLOCKED, WHICH IS WORTH SAYING OUT LOUD.** `PL.02`

`OUT LOUD.` + `**` is a four-letter family reference to `_FAMILY`, so the seat
was reported ARENA-MISSING, and the 51st audit relayed the repair to the builder
as **"PLASTIC-ONLY (`LOUD.*`: register)"** — an instruction to write a spec
named after an adverb. That is the `W.6` scar exactly (an unsatisfiable
instruction reissued by audits until somebody read the cell), arriving from the
other direction: not a real id that cannot be registered, but a phantom id that
was never a reference at all. `PG.1` was the same class, quieter — it appears in
the decree body as `depends_on=["PG.1", "PL.00"]`, a fact about a dependency,
and it RESOLVES, so it silently padded the seat's arena with a spec that cannot
decide it.

ARENA IDS ARE FACTS ONLY ONCE SOMEBODY SAYS WHICH IDS ARE ARENAS. Undeclared
seats still fall back to extraction from the arena COLUMN (never the champion or
challenger cells), with ranges expanded (`LC.00–LC.06`, `ME.11.A–F`, `W.1–W.7`)
because a range is where a phantom hides. That fallback is now a REPORTED state
(`UNDECLARED`, ratcheted at `BASELINE_UNDECLARED`), not a silent default: the
2026-08-30 lesson one document over is that a state machine which defaults a
missing state reads confident and wrong, and prose-inference is exactly a
missing declaration wearing a value.

`ARENA: NONE` is a DECLARATION, not an absence — it says the seat's ring is
genuinely unbuilt, so `NO-ARENA` for ASR/Speaker ID/Language grounding is now an
asserted fact that a passing prose mention can no longer discharge by accident.

THE DECREE SECTIONS ARE SEATS, and a table-only parser misses the worst case.
`### DECIDED BY DECREE 2026-08-09: PLASTIC ONLY` is not a table row, but it
holds a component of Jack (every encoder's plasticity) by owner decree and
pre-registers its own re-open trigger — that is a seat in everything but
formatting. Parsing only the table would report this file clean on `PL.00`, the
single most-cited missing arena in the project. The `## NOT SEATS` section is
deliberately NOT parsed: it holds the DEFINITION of Jack, which by its own words
no arena may touch, so it is correct for it to name none.

RATCHET, NOT GATE — AND IT TAKES TWO NUMBERS, NOT ONE. `--check` fails if either
baseline below grows. Eight seats are in violation today; a guard that fails
everywhere on day one is one nobody keeps green, and a guard nobody keeps green
is decoration (LESSONS.md; the `citations.py` and `decisions.py` precedent). Each
registration lowers the numbers and the constants follow them down.

WHY TWO. Until 2026-08-30 the ratchet counted `ARENA-MISSING` alone, and that is
not the quantity anybody cares about. Delete a phantom id from an arena cell and
the seat stops being `ARENA-MISSING` and becomes `NO-ARENA` — the count FALLS,
`--check` prints a smaller number, and the seat has gone from *uncontested* to
*permanently uncontestable*. The ratchet rewarded the one repair this file's own
docstring forbids in bold three paragraphs up. So the ratcheted quantity is
`UNFALSIFIABLE`: seats with no runnable arena AT ALL, which is invariant under
that conversion and falls only when a spec is actually registered.
(43rd/44th/45th audits found the same one-class shape in the challenger
quantifier; `decisions.py` had it with `NO-DEFAULT`, closed by `T0.28` P9;
`coverage.py` had it, closed by `T0.21` P2. Third instrument, same disease.)

    python -m experiments.champions          # report
    python -m experiments.champions --check  # ratchet: exit 1 if the debt grew
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

DOC = Path(__file__).resolve().parent.parent / "docs" / "CHAMPIONS.md"

# Eight seats name an arena that is not in the registry, measured 2026-08-24
# against 179 specs: Control architecture (D1.0, T2.21), Curiosity signal
# (LT.03, LT.04), Language model (LG.00), Language acquisition (LG.00), Vision
# encoder (PL.02), World (W.1-W.7), Audio encoder (PL.*), and the PLASTIC-ONLY
# decree (PL.00, PL.02). This number may SHRINK and may never GROW. It shrinks
# by REGISTERING the spec — never by deleting the arena reference, which would
# convert an ARENA-MISSING into a NO-ARENA and make the seat permanently safe
# instead of merely uncontested.
#
# RATCHETED DOWN 8 -> 5, 2026-08-30, against 195 specs. Three seats discharged
# by REGISTERING, which is the only permitted way: Language model and Language
# acquisition (`LG.00` registered `ed2d969`) and World (`W.1`-`W.5`, `W.7`,
# `W.8` registered today, after five consecutive audits asked). A shrink-only
# ratchet that is never re-baselined stops being a ratchet — it would have let
# these three seats regress to phantom arenas in silence, which is the exact
# failure this constant exists to catch. The five that remain: Control
# architecture (D1.0, T2.21), Curiosity signal (LT.03, LT.04), Vision encoder
# (PL.02), Audio encoder (PL.*), and the PLASTIC-ONLY decree (PL.00, PL.02).
#
# RATCHETED DOWN 5 -> 2, 2026-08-31, and this shrink is of a THIRD kind that
# needs its own defence: no spec was registered and no citation was corrected —
# the INSTRUMENT stopped counting a phantom that was never a reference. The
# PLASTIC-ONLY decree's `LOUD.*` came from the prose "WORTH SAYING OUT LOUD."
# and vanished the moment the seat declared its ring (see the module docstring).
# A shrink that comes from the measurement getting more honest is legitimate and
# must be locked in, exactly like one that comes from work: leaving the baseline
# at 5 would have banked three points of slack that no repair paid for. The two
# that remain are real and both are cited by seats that mean them: Control
# architecture (`D1.0`, `T2.21`, UNREGISTERABLE by decision) and Curiosity
# signal (`LT.03`, `LT.04`, unwritten).
BASELINE_ARENA_MISSING = 2

# THE SEATS NOTHING REGISTERED COULD EVER UNSEAT, measured 2026-08-30 against
# 196 specs: Control architecture (`D1.0`, `T2.21` — both UNREGISTERABLE),
# Curiosity signal (`LT.03`, `LT.04`), Audio encoder (`PL.*`, an empty family),
# the PLASTIC-ONLY decree (`PL.*`, `PL.00`, `PL.02`), and the three that name no
# arena at all: ASR, Speaker ID, Language grounding.
#
# THIS IS THE HONEST RATCHET AND `BASELINE_ARENA_MISSING` IS NOT. A seat counts
# here when `arena_present` is empty — no registered spec resolves from its
# arena cell — so the ARENA-MISSING -> NO-ARENA conversion that shrinks the
# other number leaves this one untouched. It falls ONLY by registering a spec
# (or by correcting a citation to a live successor), which is the only repair
# this file has ever endorsed. `Vision encoder` is deliberately NOT here: it
# cites the phantom `PL.02` but also `T2.03`/`T3.01`, which have run — a mixed
# citation is a documentation defect, not an unfalsifiable seat, and collapsing
# the two would hide the difference the split exists to show.
# RATCHETED DOWN 7 -> 5, 2026-08-31. The five listed above are what the tool
# measures today and have been since `PL.00`/`PL.02` were registered on 08-30;
# the constant simply had not followed. By this file's own rule one paragraph
# up — a shrink-only ratchet that is never re-baselined stops being a ratchet —
# banked slack is the same defect whether it comes from work or from delay.
# UNCHANGED by the declaration syntax that landed the same hour, which is the
# safety property that made dropping ids from arena cells permissible at all:
# no seat lost its last real ring, and `--check` is what proves it.
#
# RATCHETED DOWN 5 -> 4, 2026-08-31: registering `LT.01`–`LT.09` (`3688b9e`,
# 54th audit B1) discharged the Curiosity-signal seat the permitted way. The
# morning journal recorded "UNFALSIFIABLE 5→4" and left the constant at 5 for
# three iterations — banked slack, locked in here by the file's own rule. The
# four that remain: Control architecture (D1.0/T2.21, UNREGISTERABLE by
# decision) and the three declared `ARENA: NONE` (ASR, Speaker ID, Language
# grounding).
BASELINE_UNFALSIFIABLE = 4

# THE SEATS WHOSE ARENA EXISTS AND CAN NEVER RUN, measured 2026-08-31 (54th
# audit B4) against 211 specs: **Learning core** (BY DEFAULT; LC.03 is
# VOID-FORECLOSED "no v3, no envelope growth, no re-roll" and LC.04–LC.06 sit
# behind it) and **Fast/slow coupling** (BY DECREE; its whole arena is DP.02 —
# the connectedness test GOAL.md names as the defence against the failure it
# says can happen silently — at DP.02 ← DP.01 ← LC.04 ← LC.03). Both passed
# every existence check this file had while being uncontestable at any budget.
#
# PER `T0.31`'s PRECEDENT THE NEW CLASS JOINS THE TOTAL, not a private zero:
# this constant asserts on UNFALSIFIABLE + ARENA-UNREACHABLE together
# (4 + 2 = 6, measured at the same commit that ratcheted the 5 -> 4 above).
# The composition is the point — registering a spec that remains unreachable
# (the LT.03/LT.04 shape, had they been parented under LC.03) moves a seat
# from one class to the other and the total sees no progress, exactly as the
# ARENA-MISSING -> NO-ARENA conversion taught one level down. The total may
# shrink and may never grow. It shrinks by registering a runnable spec,
# re-parenting an arena member off its foreclosed root, or correcting a
# citation to a live successor — never by parking, foreclosing, or deleting a
# reference. `BASELINE_UNFALSIFIABLE` keeps its own separate assertion above
# so the union check cannot silently trade one class's headroom to the other
# in the growing direction.
BASELINE_UNCONTESTABLE = 6

# SEATS STILL READ BY PROSE INFERENCE. Every seat in the document was declared
# on 2026-08-31, the hour the syntax landed, so this is 0 and a new seat that
# arrives without a declaration turns `--check` red. That is the intended
# strictness: adding a chair is the Review's act, declaring what would unseat
# its holder is the same act, and the gap between them is where `LOUD.*` lived.
# It may shrink and may never grow — and unlike the two ratchets above, it is
# discharged by writing ONE LINE, so a red here is never expensive.
BASELINE_UNDECLARED = 0

# ARENA REFS THAT CAN NEVER BE REGISTERED, and why — the honest cost of closing
# the gap, which this file used to leave the reader to discover by spending the
# iteration (LESSONS.md 2026-08-29: "an instrument that names a gap must also
# say whether the gap is closable"; applied to `coverage.py` that day and not
# to this file until 2026-08-30).
#
# THE SCAR. For five consecutive audits (44th–48th) this module reported the
# World seat as ARENA-MISSING and every one of them relayed the same repair to
# the builder: *register `W.1`–`W.7`*. Six of those seven were registerable and
# were eventually registered. **`W.6` never was and never will be** — it was
# withdrawn 2026-08-09 for conflating three claims and superseded by `NE.08`.
# Because `arena_refs` expands ranges, one withdrawn id inside a cited range
# made that seat's ratchet UNSATISFIABLE BY ANY AMOUNT OF HONEST WORK, and the
# instruction was reissued five times without anybody noticing that a component
# of it could not be obeyed. A ratchet nobody can clear trains its readers to
# skip it — the exact failure `champions.py`'s own docstring warns about for
# phantom arenas, one level up.
#
# ENTRIES ARE DECISIONS, NOT OPINIONS: each names the record that closed it. A
# ref belongs here only when the project has DECIDED not to register it. Do not
# add a ref merely because it is unwritten — that is inventory debt, and the
# correct report for it is "REGISTER to discharge".
UNREGISTERABLE = {
    "W.6":   "withdrawn 2026-08-09, superseded by NE.08 — SURVIVAL_WORLD.md §5",
    "D1.0":  "unregistered BY DECISION 2026-08-13 (`a3b12f6`, choice (b)) — the "
             "WHERE question is D1 and is on the owner's desk",
    "T2.21": "unregistered BY DECISION 2026-08-13 (`a3b12f6`, choice (b)) — same "
             "ruling as D1.0",
}

# A spec id: family, then one or two components that are digits or a single
# capital. Deliberately tight, because the arena cells are dense prose full of
# near-misses. `GOAL.md` and `SYSTEM.md` fail on the lowercase component;
# `0.993`, `4.65` and `2026-08-23` fail on the leading capital; `F9` and `W0`
# fail on the missing dot. The family cap of four characters is what stops
# `DECISIONS_NEEDED.md:599` and `OpenVLA 47.0%` from parsing as ids.
_ID = re.compile(r"\b[A-Z][A-Z0-9]{0,3}\.(?:\d+|[A-Z])(?:\.(?:\d+|[A-Z]))?\b")

# `LC.00–LC.06`, `W.1–W.7`. The backreference forces the SAME family on both
# ends, so `D1.0 + T2.21` cannot be misread as a range across two families.
_NUM_RANGE = re.compile(r"\b([A-Z][A-Z0-9]{0,3})\.(\d+)\s*[–—-]\s*\1\.(\d+)\b")

# `ME.11.A–F` — the right-hand end drops the stem, which is why this needs its
# own pattern rather than a generalisation of the one above.
_ALPHA_RANGE = re.compile(r"\b([A-Z][A-Z0-9]{0,3}\.\d+)\.([A-Z])\s*[–—-]\s*([A-Z])\b")

# `PL.* applies here too` — a reference to a whole family. It resolves to every
# registered member, which for `PL.*` is none: the decree that "eliminated the
# PL.* bakeoff rather than winning it" also eliminated the arena that the seats
# still point at.
_FAMILY = re.compile(r"\b([A-Z][A-Z0-9]{0,3})\.\*")

# Provenance markings, longest and most specific first. VACANT outranks
# everything because the Deliberation row says a reactive-only Jack is the
# "incumbent by default" — the word `default` inside a sentence denying anyone
# holds the seat, which is exactly the prose-reading hazard the docstring warns
# about. `undecided` is not one of the file's five markings but four seats use
# it and it means the same as VACANT in practice; it is reported under its own
# name rather than folded in, because merging a word the file uses with a word
# it defines is how a vocabulary drifts.
_MARKINGS: Sequence[Tuple[str, str]] = (
    ("VACANT", "VACANT"),
    ("BY VERDICT", "BY VERDICT"),
    ("BY ANALYSIS", "BY ANALYSIS"),
    ("BY DECREE", "BY DECREE"),
    ("DECREE", "BY DECREE"),
    ("BY DEFAULT", "BY DEFAULT"),
    ("DEFAULT", "BY DEFAULT"),
    ("UNDECIDED", "UNDECIDED"),
)

HELD_UNEARNED = ("BY DEFAULT", "BY DECREE")

# The markings a DECLARATION may name. Deliberately the canonical right-hand
# side of `_MARKINGS` and nothing else: a declaration is a promise to use the
# file's own vocabulary, so `HELD: default` or `HELD: probably vacant` is a
# violation rather than a near-miss quietly normalised. `UNMARKED` is absent on
# purpose — it is what the INFERENCE returns when it cannot tell, and no author
# may declare that they did not say.
CANONICAL_MARKINGS = tuple(dict.fromkeys(name for _needle, name in _MARKINGS))

# `ARENA: NONE` — the seat's ring is unbuilt, asserted rather than inferred from
# a cell that happened to contain no id.
NO_ARENA_DECLARED = ("NONE", "(NONE)", "-", "—", "–")

# `- SEAT: Vision encoder | HELD: BY DEFAULT | ARENA: T2.03, PL.02`
# Anchored at the start of a line (after list punctuation) so a sentence in this
# file's own prose that happens to contain the word cannot declare a seat —
# `lib_credits.sh`'s rule that a detector on a shared surface must bound itself
# to its own writes, applied to a document that quotes its own syntax.
_DECL_LINE = re.compile(r"^[-*\s]*SEAT:\s*(?P<body>.+)$")

_SEP = re.compile(r"^:?-{2,}:?$")
_DECREE_HEAD = re.compile(r"^###\s+.*\bDECREE\b.*$", re.M)
_ANY_HEAD = re.compile(r"^#{2,3}\s", re.M)


def _clean(cell: str) -> str:
    """Strip markdown emphasis so a marking is not hidden behind `**`."""
    return re.sub(r"[`*_]", "", cell).strip()


def _marking(held: str, champion: str) -> str:
    """The provenance marking for a seat.

    The `held` COLUMN is authoritative. Only when it is empty or `—` does this
    fall back to the champion cell, where several rows carry the marking inline
    (`**VACANT** — prior holder's evidence voided (T0.14)`). Preferring the
    column matters: the champion cell is free prose and contains words like
    "default" and "decree" used descriptively.
    """
    for source in (_clean(held), _clean(champion)):
        if not source or source in {"-", "—", "–"}:
            continue
        up = source.upper()
        for needle, name in _MARKINGS:
            if needle in up:
                return name
        return "UNMARKED"
    return "UNMARKED"


def arena_refs(cell: str) -> List[str]:
    """Every arena reference in an arena cell, ranges expanded, order kept.

    Returns references, not resolutions: `PL.*` survives as `PL.*` so the report
    can say which FORM of reference is dangling. Range expansion is the reason
    this is not a one-line `findall` — `W.1–W.7` is one string naming seven
    arenas, and reporting only its endpoints would undercount the hole by five.
    """
    refs: List[str] = []

    def add(ref: str) -> None:
        if ref not in refs:
            refs.append(ref)

    for fam, lo, hi in _NUM_RANGE.findall(cell):
        # Zero padding is part of the id: `LC.00` and `LC.0` are different
        # strings and only one of them is in the registry.
        width = len(lo)
        for n in range(int(lo), int(hi) + 1):
            add(f"{fam}.{n:0{width}d}")
    for stem, lo, hi in _ALPHA_RANGE.findall(cell):
        for c in range(ord(lo), ord(hi) + 1):
            add(f"{stem}.{chr(c)}")
    for fam in _FAMILY.findall(cell):
        add(f"{fam}.*")
    for sid in _ID.findall(cell):
        add(sid)
    return refs


def declarations(text: str) -> Tuple[Dict[str, dict], List[Tuple[str, str, str]]]:
    """`seat name -> {held, arena_refs, line}` from the `SEAT:` lines, plus problems.

    Problems are returned rather than raised, and each is a violation triple in
    the same shape `audit()` emits, because a malformed declaration is exactly
    as dangerous as a missing one: it reads as *declared* to anyone skimming the
    document. The four:

      DECL-INCOMPLETE   a `SEAT:` line missing `HELD:` or `ARENA:`
      DECL-UNKNOWN-HELD a marking outside CANONICAL_MARKINGS
      DECL-DUPLICATE    two declarations for one seat — a contradiction, and it
                        is NOT resolved by taking the first or the last. Both
                        are dropped and the seat falls back to inference, which
                        keeps it visible in the UNDECLARED list instead of
                        letting an arbitrary tiebreak look authoritative.
      DECL-ORPHAN       (raised in `audit()`, where the seat list exists) a
                        declaration for a seat this file does not contain —
                        a phantom seat, the mirror of a phantom arena.
    """
    decls: Dict[str, dict] = {}
    problems: List[Tuple[str, str, str]] = []
    seen: Dict[str, int] = {}

    for lineno, line in enumerate(text.splitlines(), 1):
        # Matched on the RAW line and cleaned per FIELD, not once over the whole
        # line: `_clean` strips `*`, and the first draft of this parser turned
        # the Audio encoder's declared `ARENA: PL.*` into `PL.`, which resolves
        # to nothing — the syntax invented to stop a seat looking falsifiable
        # when it is not, silently making a falsifiable one look dead. The seat
        # NAME is cleaned (it must match `parse()`'s cleaned cell); the ARENA
        # value is not (a family reference is `*`-significant).
        m = _DECL_LINE.match(line)
        if not m:
            continue
        fields: Dict[str, str] = {}
        for part in m.group("body").split("|"):
            key, sep, val = part.partition(":")
            if sep:
                fields[_clean(key).upper()] = val.strip()
        name = _clean(m.group("body").split("|")[0])
        if "HELD" not in fields or "ARENA" not in fields:
            problems.append(("DECL-INCOMPLETE", name or f"(line {lineno})",
                             "a SEAT: declaration must carry both HELD: and "
                             "ARENA: — a half-declared seat reads as declared"))
            continue
        held = fields["HELD"].upper()
        if held not in CANONICAL_MARKINGS:
            problems.append(("DECL-UNKNOWN-HELD", name,
                             f"HELD: {fields['HELD']!r} is not one of "
                             f"{', '.join(CANONICAL_MARKINGS)}"))
            continue
        arena = fields["ARENA"]
        if arena.strip().upper() in NO_ARENA_DECLARED:
            refs: List[str] = []
        else:
            refs = arena_refs(arena)
            if not refs:
                # `ARENA: the HR bakeoff (queued)` — prose where ids belong. It
                # would parse to zero refs and read exactly like a declared
                # unbuilt ring, which is how a typo (`PL02`) becomes an
                # assertion that nothing could ever unseat the holder. The
                # NONE-token branch above is what makes this distinguishable at
                # all: without it the two cases are one, and a mutation that
                # deletes it cannot be caught (measured — it was the only
                # survivor of this file's mutation battery until this check).
                problems.append(("DECL-EMPTY-ARENA", name,
                                 f"ARENA: {arena!r} names no spec id — write "
                                 f"`ARENA: NONE` to assert the ring is unbuilt, "
                                 f"or name the ids; prose declares nothing"))
                continue
        if name in seen:
            problems.append(("DECL-DUPLICATE", name,
                             f"declared twice (lines {seen[name]} and {lineno}); "
                             "both dropped — a contradiction is not a vote"))
            decls.pop(name, None)
            continue
        seen[name] = lineno
        decls[name] = {"held": held, "arena_refs": refs, "line": lineno,
                       "arena_cell": arena}
    return decls, problems


def apply_declarations(seats: Sequence[dict], decls: Dict[str, dict]) -> List[str]:
    """Declaration wins; the inference is kept beside it. Returns orphan names.

    Every seat gains `declared`, `held_inferred` and `arena_refs_inferred`
    whether or not it is declared, so `main()` can print the disagreement rather
    than the reader having to trust that the switch changed nothing.
    """
    names = {s["seat"] for s in seats if s["kind"] != "malformed"}
    for s in seats:
        s["held_inferred"] = s["held"]
        s["arena_refs_inferred"] = list(s.get("arena_refs", []))
        d = decls.get(s["seat"]) if s["kind"] != "malformed" else None
        s["declared"] = bool(d)
        if d:
            s["held"] = d["held"]
            s["arena_refs"] = list(d["arena_refs"])
    return sorted(n for n in decls if n not in names)


def undeclared(seats: Sequence[dict]) -> List[str]:
    """Seats still read by prose inference. Ratcheted; see BASELINE_UNDECLARED."""
    return [s["seat"] for s in seats
            if s["kind"] != "malformed" and not s.get("declared")]


def declaration_deltas(seats: Sequence[dict]) -> List[Tuple[str, str, str]]:
    """`(seat, what the inference said, what the declaration says)` where they differ.

    This is the payoff of the whole exercise and the reason the inference is not
    simply deleted: a declaration that agrees with the inference everywhere would
    mean the syntax bought nothing, and that claim is checkable only if both
    readings are computed.
    """
    out: List[Tuple[str, str, str]] = []
    for s in seats:
        if not s.get("declared"):
            continue
        was, now = s.get("held_inferred"), s["held"]
        old, new = list(s.get("arena_refs_inferred", [])), list(s["arena_refs"])
        if was != now:
            out.append((s["seat"], f"held {was}", f"held {now}"))
        if old != new:
            dropped = [r for r in old if r not in new]
            added = [r for r in new if r not in old]
            out.append((s["seat"],
                        "arena " + (" ".join(old) or "(none named)"),
                        "arena " + (" ".join(new) or "(none)") +
                        (f"  [-{' -'.join(dropped)}]" if dropped else "") +
                        (f"  [+{' +'.join(added)}]" if added else "")))
    return out


def resolve(ref: str, by_id: dict) -> List[str]:
    """Registered specs a reference names. Empty means the arena does not exist."""
    if ref.endswith(".*"):
        return sorted(i for i in by_id if i.startswith(ref[:-1]))
    return [ref] if ref in by_id else []


def parse(text: str) -> List[dict]:
    """Seats from CHAMPIONS.md: the anatomy table, plus the decree sections.

    Column POSITIONS come from the table's own header row rather than being
    assumed, and they persist across the blank line at CHAMPIONS.md:79 that
    splits the anatomy table into two — the second half (Smell, Taste, Voice,
    Language acquisition) has no header of its own, so a strict per-table
    parser silently drops four seats.
    """
    seats: List[dict] = []
    cols: Dict[str, int] = {}

    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(_SEP.match(_clean(c) or "-") for c in cells):
            continue
        lowered = [_clean(c).lower() for c in cells]
        if "seat" in lowered and "arena" in lowered:
            cols = {name: i for i, name in enumerate(lowered)}
            continue
        if not cols or len(cells) <= max(cols.values()):
            # A row narrower than the header is malformed, not a seat. Reported
            # rather than dropped, on the coverage.py rule that silence about a
            # thing that looks like a claim is the failure mode.
            seats.append({"seat": f"(malformed row at line {lineno})", "line": lineno,
                          "held": "UNMARKED", "champion": line.strip()[:60],
                          "challenger": "", "arena_cell": "", "kind": "malformed"})
            continue
        get = lambda k: cells[cols[k]] if k in cols else ""  # noqa: E731
        seats.append({
            "seat": _clean(get("seat")),
            "line": lineno,
            "champion": _clean(get("champion")),
            "held": _marking(get("held"), get("champion")),
            "arena_cell": get("arena"),
            "challenger": _clean(get("challenger status") or get("challenger")),
            "kind": "seat",
        })

    for m in _DECREE_HEAD.finditer(text):
        nxt = _ANY_HEAD.search(text, m.end())
        body = text[m.end(): nxt.start() if nxt else len(text)]
        seats.append({
            "seat": _clean(m.group(0).lstrip("# ")),
            "line": text[: m.start()].count("\n") + 1,
            "champion": "(the decree itself)",
            "held": "BY DECREE",
            # The WHOLE body is the arena cell here. A decree has no columns, and
            # its arena ids are scattered across its "WHAT STILL RUNS" line and
            # its re-open trigger — which is precisely where PL.00 lives.
            "arena_cell": body,
            "challenger": "the pre-registered re-open trigger",
            "kind": "decree",
        })

    for s in seats:
        s["arena_refs"] = arena_refs(s["arena_cell"])
    return seats


# A ledger status that is a VERDICT. `SYSTEM.md`: a bakeoff arm that fails the
# learning gate returns VOID rather than a confident wrong answer, and the
# instruction is "fix the arm, do not decide" — so a VOID has decided nothing
# and cannot discharge a seat. ERROR and SKIP are not verdicts either.
VERDICTS = ("PASS", "FAIL")

# Kinds that seat no challenger, imported from `coverage.py` rather than
# re-listed: it learned on the 8th-12th audits that a fixture/rule/sensor buys
# no claim credit, and this file cited that scar in its own docstring and then
# reproduced it one document over. When you build an instrument by analogy to
# an existing one, port its SCARS as well as its shape — the fix is an import,
# not a design.
NON_CHALLENGER_KINDS = ("fixture", "rule", "sensor")


def _spec_kinds(by_id: dict) -> Dict[str, set]:
    """`spec id -> {declared COVERS kinds}`, via `coverage.py`'s parser."""
    from .coverage import declarations
    out: Dict[str, set] = {}
    for pairs in declarations(by_id)[0].values():
        for sid, kind in pairs:
            out.setdefault(sid, set()).add(kind)
    return out


def _challenger_runs(arena_status: Dict[str, str], by_id: dict) -> List[str]:
    """Arena specs that actually CONTESTED the seat.

    Three distinct false positives for "defended", each its own line below:
    a spec that has not run; a spec whose only completion is VOID (not a
    verdict); and a spec whose declared kind cannot seat anyone — a fixture
    certifies apparatus, a rule states an admission criterion, a sensor
    reports a channel. None of the three is a challenger beating an incumbent.

    A SPEC WITH NO `COVERS:` MARKER AT ALL COUNTS AS A CHALLENGER, and that is
    the ordered rule (45th audit B4: "its COVERS kind is not fixture, rule or
    sensor"), not an oversight. It is also this filter's remaining soft spot,
    so `main()` prints every seat that rests on kindless arenas rather than
    leaving the reader to assume the discharge was earned: `Learning core` is
    discharged by exactly one such spec, `LC.02` (a throughput feasibility
    gate), which is plausibly the same false positive one layer down. Tightening
    it means DECLARING LC.02's kind, not widening this predicate — a detector
    tuned until it agrees with its maintainer is `coverage.py`'s scar 1.

    NOT filtered here, because it needs a judgement this parser cannot make:
    **the incumbent's own arm is not a contest.** `Episodic retrieval` is held
    BY VERDICT on an arena `ME.11.A-F` whose only run arm is `ME.11.A`,
    "lexical containment, the incumbent, as the null" — a bakeoff where 1 of 6
    arms ran. Detecting that needs the arena to say which arm is the
    incumbent's, which the table does not yet record. Recorded as owed rather
    than silently approximated.
    """
    kinds = _spec_kinds(by_id)
    return sorted(sid for sid, st in arena_status.items()
                  if st in VERDICTS
                  and not (kinds.get(sid, set())
                           and kinds[sid] <= set(NON_CHALLENGER_KINDS)))


def _unrunnable(by_id: dict, status: Callable[[str], str],
                parked_ids: set, foreclosed: set) -> Dict[str, str]:
    """`spec id -> the ROOT that welds it shut`, over every registered spec.

    A spec can never reach a verdict when it is itself parked or
    VOID-FORECLOSED, or when any dependency that has not already PASSed is
    (transitively) in that state. A dep that is merely FAIL or NOT_RUN is a
    queue position, not a wall — T2.01's settled FAIL blocks 36 specs and not
    one of them is *unreachable*, because a redesign could re-run it. Parked
    and foreclosed are different in kind: both are declarations that the next
    unit is NOT a run, so nothing on the runnable graph can ever move them.

    The root id is carried, not just the bit, because the repair lives at the
    root: `DP.02 (behind LC.03)` names the re-parenting question, `DP.02
    (unreachable)` names nothing. A cycle in `depends_on` is a registry bug,
    not proof of unreachability, so the walk treats it as runnable — the safe
    failure direction for a detector whose false positives would indict
    healthy seats.
    """
    memo: Dict[str, Optional[str]] = {}

    def walk(sid: str, stack: tuple) -> Optional[str]:
        if sid in memo:
            return memo[sid]
        if sid in stack:
            return None
        root = None
        if sid in foreclosed or sid in parked_ids:
            root = sid
        else:
            for dep in getattr(by_id.get(sid), "depends_on", None) or ():
                if dep in by_id and status(dep) != "PASS":
                    r = walk(dep, stack + (sid,))
                    if r:
                        root = r
                        break
        memo[sid] = root
        return root

    for sid in by_id:
        walk(sid, ())
    return {sid: r for sid, r in memo.items() if r}


def unreachable_arena(seats: Sequence[dict]) -> List[str]:
    """Seats whose pending arena can never run — the ratchet's second class.

    Reads the flag `audit()` stores rather than recomputing the predicate, so
    this list cannot drift from the violations the report prints — the exact
    reader-drift `run blocked` fell into when VOID split into two states and
    only one of its readers was told (54th audit B2).
    """
    return [s["seat"] for s in seats if s.get("arena_welded")]


def unfalsifiable(seats: Sequence[dict]) -> List[str]:
    """Seats no registered spec could ever unseat — `arena_present` is empty.

    The ratcheted quantity. See `BASELINE_UNFALSIFIABLE`: this is invariant
    under the ARENA-MISSING -> NO-ARENA conversion that deleting an arena
    reference performs, which is exactly why it, and not the violation count,
    is what `--check` may not let grow. Seats must have been through `audit()`,
    which is what sets `arena_present`.
    """
    return [s["seat"] for s in seats
            if s["kind"] != "malformed" and not s.get("arena_present")]


def audit(text: str, by_id: dict, status: Callable[[str], str], *,
          unregisterable: Optional[dict] = None,
          parked_ids: Optional[set] = None,
          foreclosed: Optional[set] = None
          ) -> Tuple[List[Tuple[str, str, str]], List[dict]]:
    """Return (violations, seats). A violation names a seat, never a spec.

    `unregisterable` overrides the module's decision set — the refs the project
    has DECIDED never to register. It is a parameter so the pre-2026-08-30
    organ, which had no closability split and told the builder to "register" a
    withdrawn spec for five consecutive audits, stays executable as a control
    rather than being paraphrased into one (T0.08 property 5; `T0.29`).

    `parked_ids`/`foreclosed` override the reachability roots for the fixture;
    `None` computes them from the real declarations. The foreclosure gate is
    the SAME conjunction as `coverage.queue_depth` and `run blocked` (status
    VOID *and* the module docstring declares) so the three readers cannot
    drift — a declaration without the VOID is a spec with bad manners, and a
    VOID without the declaration is repairable.
    """
    unregisterable = UNREGISTERABLE if unregisterable is None else unregisterable
    if parked_ids is None:
        from .coverage import parked as _parked
        parked_ids = set(_parked(by_id)[0])
    if foreclosed is None:
        from .protocol import void_foreclosed as _vf
        foreclosed = {sid for sid in by_id
                      if status(sid) == "VOID" and _vf(sid)}
    dead_roots = _unrunnable(by_id, status, parked_ids, foreclosed)
    seats = parse(text)
    decls, violations = declarations(text)
    for orphan in apply_declarations(seats, decls):
        violations.append(("DECL-ORPHAN", orphan,
                           "a SEAT: declaration for a seat this file does not "
                           "contain — the seat was renamed or removed and its "
                           "declaration stayed, so it declares nothing"))

    for s in seats:
        if s["kind"] == "malformed":
            violations.append(("MALFORMED-ROW", s["seat"],
                               "a table row this parser could not read as a seat"))
            continue
        refs = s["arena_refs"]
        s["arena_missing"] = [r for r in refs if not resolve(r, by_id)]
        s["arena_present"] = sorted({i for r in refs for i in resolve(r, by_id)})
        s["arena_status"] = {i: status(i) for i in s["arena_present"]}
        # The members still owing a verdict, and which of them can never pay.
        # A VOID member is PENDING (a VOID decided nothing), which is exactly
        # how a foreclosed VOID lands here: still owing, never paying.
        s["arena_pending"] = [i for i, st in s["arena_status"].items()
                              if st not in VERDICTS]
        s["arena_unreachable"] = {i: dead_roots[i] for i in s["arena_pending"]
                                  if i in dead_roots}
        s["arena_pending_dead"] = bool(
            s["arena_pending"]
            and all(i in dead_roots for i in s["arena_pending"]))
        # Ratchet scope: the unearned markings, the same two UNCONTESTED
        # polices — those are the seats where a HOLDER sits without a verdict
        # and the invitation is the only honesty on offer. A VACANT seat with
        # a welded ring is a different defect (nobody can ever WIN it, rather
        # than nobody can ever lose it) and is printed as a note by `main()`
        # instead of counted here — recorded as scope, not overlooked.
        s["arena_welded"] = bool(s["held"] in HELD_UNEARNED
                                 and s["arena_pending_dead"])

        if not refs:
            violations.append(("NO-ARENA", s["seat"],
                               "names no arena spec id at all — nothing that could "
                               "ever be run would unseat the holder"))
        elif s["arena_missing"]:
            # An instrument that names a gap must also say whether the gap is
            # CLOSABLE (LESSONS.md, 2026-08-29 — written for `coverage.py`'s
            # queue depth and never applied here). "Register the spec" is the
            # right repair only for a ref that MAY be registered; for one the
            # project has decided against, it is an instruction nobody can
            # obey, and this file issued exactly that instruction for five
            # consecutive audits. See UNREGISTERABLE.
            s["arena_unregisterable"] = [r for r in s["arena_missing"]
                                         if r in unregisterable]
            reg = [r for r in s["arena_missing"] if r not in unregisterable]
            why = []
            if reg:
                why.append(f"names {', '.join(reg)} — not in the registry; "
                           f"REGISTER to discharge")
            if s["arena_unregisterable"]:
                why.append("names " + "; ".join(
                    f"{r} ({unregisterable[r]})" for r in s["arena_unregisterable"]
                ) + " — NOT registerable, so the repair is to CORRECT THE "
                    "CITATION to the live successor, never to write the spec")
            violations.append(("ARENA-MISSING", s["seat"],
                               " AND ".join(why) +
                               ", so the seat looks contestable and is not"))

        # Only DEFAULT and DECREE seats: those are the two markings the file
        # itself calls unearned ("a DEFAULT champion never won anything";
        # "BY DECREE = an owner decision"). A BY ANALYSIS seat whose arena has
        # not run is in the same position by rule 1's own words — "held on a
        # proof rather than a bakeoff, PENDING ITS ARENA RUN" — but the file
        # states that pending-ness openly, so it is reported as a note below
        # rather than counted as a violation. Flagging what a document already
        # admits trains its readers to skip the flags.
        # QUANTIFY OVER CHALLENGERS, NOT OVER THE ARENA LIST (43rd audit,
        # carried unrepaired by the 44th and 45th). This read
        # `all(v == "NOT_RUN" ...)`, so ONE arena spec having run — any one,
        # for any reason — discharged the whole seat forever. For a one-arena
        # seat the two questions coincide, which is why every single-arena
        # seat was caught correctly and both multi-arena seats were not, and
        # the multi-arena seats are the consequential ones: `Learning core`
        # read ok on four "passing" arenas of which one is a declared `rule`,
        # one a `fixture` and two are feasibility gates, while the three specs
        # that could actually move it were VOID/NOT_RUN/NOT_RUN — over a cell
        # that says, in bold, "DEFAULT, never defended".
        s["challenger_runs"] = _challenger_runs(s["arena_status"], by_id)
        if s["arena_welded"]:
            # Fires whether or not a challenger once ran: Learning core was
            # nominally discharged by LC.02 (a feasibility gate that PASSed),
            # and its four remaining doors are welded all the same. And it
            # SUPPRESSES UNCONTESTED below — "the invitation is real but
            # unanswered" is the cheap reading of a door that cannot open,
            # the exact misreport `run blocked` made of LC.03 (54th audit B2).
            def _tag(i: str) -> str:
                root = s["arena_unreachable"][i]
                if root == i:
                    return (f"{i} (PARKED)" if i in parked_ids
                            else f"{i} (VOID-FORECLOSED)")
                return f"{i} (behind {root})"
            violations.append(("ARENA-UNREACHABLE", s["seat"],
                               f"held {s['held']}; every arena spec still owing "
                               f"a verdict can never run: "
                               + "; ".join(_tag(i) for i in s["arena_pending"])
                               + " — the seat looks contestable and is not; the "
                                 "repair is a re-parenting or a redesign, "
                                 "never a re-run"))
        elif (s["held"] in HELD_UNEARNED and s["arena_present"]
                and not s["challenger_runs"]):
            ran = sorted(i for i, v in s["arena_status"].items()
                         if v not in ("NOT_RUN", None))
            why = (f"the only arena completion(s) — {', '.join(ran)} — buy no "
                   f"contest (a VOID is not a verdict; a fixture/rule/sensor "
                   f"seats no challenger)" if ran else "has never run")
            violations.append(("UNCONTESTED", s["seat"],
                               f"held {s['held']}; arena "
                               f"{', '.join(s['arena_present'])} exists and "
                               f"{why} — the invitation is real but unanswered"))
    return violations, seats


def _fixture() -> None:
    """A known-positive this tool must catch, through the real code path.

    Every audit tool here carries one (LESSONS.md; `decisions.py` precedent): a
    scanner nobody has watched catch something is a scanner nobody has tested.
    This document holds one of each defect plus TWO healthy seats — one BY
    VERDICT, one BY DEFAULT with a run arena — because the failure that would
    hurt most is not a miss, it is flagging a seat that is doing everything the
    file asks. The registry and ledger are synthetic so the fixture keeps its
    verdict when the real ladder grows.
    """
    doc = """
| seat | champion | held | arena | challenger status |
|---|---|---|---|---|
| Healthy verdict seat | thing-that-won | **BY VERDICT** (OK.01) | OK.01 (registered) | a challenger |
| Default seat a claim spec defended | incumbent | **DEFAULT, never defended** | OK.01–OK.02 (registered) | a challenger |
| Default seat only a fixture answered | incumbent | **DEFAULT, never defended** | OK.04 + OK.05 (registered) | a challenger |
| Default seat whose only run went VOID | incumbent | **DEFAULT, never defended** | OK.06 + OK.03 (registered) | a challenger |
| Phantom arena seat | incumbent | **DEFAULT, never defended** | ZZ.00 + ZZ.01 (queued) | a challenger |
| Seat citing a withdrawn spec | incumbent | **DEFAULT, never defended** | ZZ.00 + W.6 (queued) | a challenger |
| Default seat whose ring is foreclosed | incumbent | **DEFAULT, never defended** | OK.07 (registered) | a challenger |
| Default seat behind a foreclosure | incumbent | **DEFAULT, never defended** | OK.08 (registered) | a challenger |
| Default seat contested once, then welded | incumbent | **DEFAULT, never defended** | OK.01 + OK.07 | a challenger |
| Default seat with one live door | incumbent | **DEFAULT, never defended** | OK.07 + OK.03 | a challenger |
| Default seat whose ring is parked | incumbent | **DEFAULT, never defended** | OK.09 (registered) | a challenger |
| Vacant seat behind a foreclosure | **VACANT** — nobody has ever held it | — | OK.08 (registered) | a challenger |
| No arena at all | incumbent | **BY DECREE** (owner) | HR bakeoff (queued) | a challenger |
| Uncontested decree seat | incumbent | **BY DECREE** (owner) | OK.03 (registered) | a challenger |
| Vacant by default words | **VACANT** — the incumbent by default is nobody | — | OK.01 (registered) | a challenger |
| Declared seat whose prose over-reads | incumbent | **BY VERDICT** (OK.01) | OK.01 + OK.02 named in passing, WORTH SAYING OUT LOUD.** | a challenger |
| Declared seat whose ring is unbuilt | incumbent | **BY DECREE** (owner) | OK.03 mentioned but not an arena | a challenger |
| Seat with a half-written declaration | incumbent | **DEFAULT, never defended** | ZZ.00 (queued) | a challenger |
| Declared seat naming a family | incumbent | **BY VERDICT** (OK.01) | OK.04 is the only id in this cell | a challenger |
| Seat declaring prose where ids belong | incumbent | **BY DECREE** (owner) | OK.03 (registered) | a challenger |

- SEAT: Declared seat whose prose over-reads | HELD: BY DEFAULT | ARENA: OK.01
- SEAT: Declared seat whose ring is unbuilt | HELD: BY DECREE | ARENA: NONE
- SEAT: A seat this file does not contain | HELD: VACANT | ARENA: OK.01
- SEAT: Healthy verdict seat | HELD: PROBABLY VACANT | ARENA: OK.01
- SEAT: Seat with a half-written declaration | HELD: BY DEFAULT
- SEAT: Declared seat naming a family | HELD: BY VERDICT | ARENA: OK.*
- SEAT: Seat declaring prose where ids belong | HELD: BY DECREE | ARENA: the HR bakeoff (queued)
- SEAT: Vacant by default words | HELD: VACANT | ARENA: OK.01
- SEAT: Vacant by default words | HELD: BY DECREE | ARENA: ZZ.00

### DECIDED BY DECREE 2099-01-01: SOMETHING

WHAT STILL RUNS: ZZ.02 (a floor nobody wrote). Cheap, CPU.

### Superseded context: not a decree, must not be parsed as one

This section names ZZ.09 and must contribute no seat and no violation.
"""
    class _S:
        def __init__(self, notes="", deps=()):
            self.notes = notes
            self.depends_on = list(deps)

    # OK.04/OK.05 are declared support kinds; OK.06 ran but only to VOID.
    # OK.07 is a foreclosed VOID; OK.08 is runnable-looking but parented under
    # it; OK.09 is parked. The roots arrive as parameters — the real
    # declarations live in docstrings and registry notes this fixture does not
    # have — so the default-computation path is exercised by T0.29's P10
    # against the live documents, not here.
    by_id = {"OK.01": _S("COVERS: smell (claim)"), "OK.02": _S(), "OK.03": _S(),
             "OK.04": _S("COVERS: smell (fixture)"),
             "OK.05": _S("COVERS: balance (sensor)"), "OK.06": _S(),
             "OK.07": _S(), "OK.08": _S(deps=["OK.07"]), "OK.09": _S()}
    ran = {"OK.01": "PASS", "OK.02": "PASS", "OK.03": "NOT_RUN",
           "OK.04": "PASS", "OK.05": "PASS", "OK.06": "VOID",
           "OK.07": "VOID", "OK.08": "NOT_RUN", "OK.09": "NOT_RUN"}
    violations, seats = audit(doc, by_id, lambda sid: ran.get(sid, "NOT_RUN"),
                              parked_ids={"OK.09"}, foreclosed={"OK.07"})
    flagged: Dict[str, set] = {}
    for kind, seat, _why in violations:
        flagged.setdefault(seat, set()).add(kind)

    # THE TWO ROWS THIS GUARD EXISTS FOR (43rd audit). Each is a distinct
    # false positive for "defended" that the old `all(... NOT_RUN)` quantifier
    # scored as healthy, and each row is NAMED for what it is — the previous
    # fixture called such a row "Healthy default seat" and asserted it was not
    # flagged, so the bug was tested-in and blessed by the one battery whose
    # purpose is catching it.
    assert flagged.get("Default seat only a fixture answered") == {"UNCONTESTED"}, flagged
    assert flagged.get("Default seat whose only run went VOID") == {"UNCONTESTED"}, flagged
    assert flagged.get("Phantom arena seat") == {"ARENA-MISSING"}, flagged

    # THE CLOSABILITY SPLIT (2026-08-30). Both seats below are ARENA-MISSING and
    # the old code printed them identically; only one of them can be discharged
    # by writing a spec. If this battery ever passes with the two messages
    # equal, the guard has been removed and the five-audit unsatisfiable
    # instruction is back.
    assert flagged.get("Seat citing a withdrawn spec") == {"ARENA-MISSING"}, flagged
    by_seat = {seat: why for kind, seat, why in violations if kind == "ARENA-MISSING"}
    withdrawn_seat = [s for s in seats if s["seat"] == "Seat citing a withdrawn spec"][0]
    phantom_seat = [s for s in seats if s["seat"] == "Phantom arena seat"][0]
    assert withdrawn_seat["arena_unregisterable"] == ["W.6"], withdrawn_seat
    assert phantom_seat["arena_unregisterable"] == [], phantom_seat
    # The registerable half of a MIXED citation must still say "register" —
    # one un-writable id may not excuse the ids that are merely unwritten.
    assert "ZZ.00" in by_seat["Seat citing a withdrawn spec"], by_seat
    assert "REGISTER to discharge" in by_seat["Seat citing a withdrawn spec"], by_seat
    assert "CORRECT THE CITATION" in by_seat["Seat citing a withdrawn spec"], by_seat
    assert "CORRECT THE CITATION" not in by_seat["Phantom arena seat"], by_seat
    assert flagged.get("No arena at all") == {"NO-ARENA"}, flagged
    assert flagged.get("Uncontested decree seat") == {"UNCONTESTED"}, flagged
    # THE REACHABILITY CLASS (54th audit B4). Four shapes that must fire, one
    # that must not, and one out of scope — each pinned to the real seat it
    # stands in for. A foreclosed ring, a ring parented under one (the
    # Fast/slow shape: DP.02 behind LC.03), a ring parked (the Smell shape,
    # were Smell unearned), and the Learning-core shape: a challenger once ran
    # and every REMAINING door is welded — which must fire even though
    # `challenger_runs` is non-empty, because contestability is about the
    # future. None of the four may ALSO read UNCONTESTED: "real but
    # unanswered" about a door that cannot open is `run blocked`'s LC.03
    # misreport, the exact cheap reading B2 removed one instrument over.
    assert flagged.get("Default seat whose ring is foreclosed") == {"ARENA-UNREACHABLE"}, flagged
    assert flagged.get("Default seat behind a foreclosure") == {"ARENA-UNREACHABLE"}, flagged
    assert flagged.get("Default seat contested once, then welded") == {"ARENA-UNREACHABLE"}, flagged
    assert flagged.get("Default seat whose ring is parked") == {"ARENA-UNREACHABLE"}, flagged
    # One live pending door keeps the seat merely UNCONTESTED — a mixed ring
    # is a debt in the world, not a welded one, and collapsing the two would
    # indict every seat that shares an arena with any foreclosure.
    assert flagged.get("Default seat with one live door") == {"UNCONTESTED"}, flagged
    # Marking scope: a VACANT welded ring is nobody-can-WIN, a different
    # defect, reported as a note rather than counted — and never flagged here.
    assert "Vacant seat behind a foreclosure" not in flagged, flagged
    # The message carries the ROOT, because the repair lives at the root.
    _msg = {seat: why for kind, seat, why in violations
            if kind == "ARENA-UNREACHABLE"}
    assert "VOID-FORECLOSED" in _msg["Default seat whose ring is foreclosed"], _msg
    assert "OK.08 (behind OK.07)" in _msg["Default seat behind a foreclosure"], _msg
    assert "OK.09 (PARKED)" in _msg["Default seat whose ring is parked"], _msg
    _behind = [s for s in seats if s["seat"] == "Default seat behind a foreclosure"][0]
    assert _behind["arena_unreachable"] == {"OK.08": "OK.07"}, _behind
    # ...and the ratchet's second class counts exactly the seats that fired.
    assert set(unreachable_arena(seats)) == {
        "Default seat whose ring is foreclosed",
        "Default seat behind a foreclosure",
        "Default seat contested once, then welded",
        "Default seat whose ring is parked"}, unreachable_arena(seats)
    # The decree section is a seat and its dangling `WHAT STILL RUNS` id is the
    # PL.00 case in miniature: outside the table, cited as live, unregistered.
    decree = [s for s in seats if s["kind"] == "decree"]
    assert len(decree) == 1, [s["seat"] for s in seats]
    assert decree[0]["arena_missing"] == ["ZZ.02"], decree[0]
    assert flagged.get(decree[0]["seat"]) == {"ARENA-MISSING"}, flagged
    # THE DECLARATION SYNTAX (2026-08-31). Six properties, each a way a
    # declaration can be wrong, because a syntax whose malformed cases fall back
    # SILENTLY to prose is worse than no syntax: the reader sees `decl` in the
    # report and stops checking.
    seat_by_name = {s["seat"]: s for s in seats}

    # 1. A declaration NARROWS an over-reading cell, and the over-read is the
    #    real one: `WORTH SAYING OUT LOUD.**` parses as the family `LOUD.*`.
    over = seat_by_name["Declared seat whose prose over-reads"]
    assert "LOUD.*" in over["arena_refs_inferred"], over["arena_refs_inferred"]
    assert over["arena_refs"] == ["OK.01"], over["arena_refs"]
    assert "ARENA-MISSING" not in flagged.get(over["seat"], set()), flagged
    # 2. ...and it overrides the marking column, not only the arena.
    assert over["held_inferred"] == "BY VERDICT" and over["held"] == "BY DEFAULT", over
    # 3. `ARENA: NONE` asserts an unbuilt ring even where the cell names a spec.
    unbuilt = seat_by_name["Declared seat whose ring is unbuilt"]
    assert unbuilt["arena_refs_inferred"] == ["OK.03"], unbuilt
    assert unbuilt["arena_refs"] == [], unbuilt
    assert flagged.get(unbuilt["seat"]) == {"NO-ARENA"}, flagged
    # 4. A declaration for a seat that does not exist is a phantom SEAT, the
    #    mirror of a phantom arena, and is reported rather than ignored.
    assert flagged.get("A seat this file does not contain") == {"DECL-ORPHAN"}, flagged
    # 5. THE THREE MALFORMED CASES ALL FALL BACK TO INFERENCE *AND SAY SO* —
    #    each would otherwise be a seat reading `decl` while declaring nothing.
    #    An unknown marking, a missing ARENA field, and a contradiction.
    assert flagged.get("Healthy verdict seat") == {"DECL-UNKNOWN-HELD"}, flagged
    assert not seat_by_name["Healthy verdict seat"]["declared"], seat_by_name
    half = seat_by_name["Seat with a half-written declaration"]
    assert flagged.get(half["seat"]) == {"ARENA-MISSING", "DECL-INCOMPLETE"}, flagged
    assert not half["declared"] and half["arena_refs"] == ["ZZ.00"], half
    dup = seat_by_name["Vacant by default words"]
    assert flagged.get(dup["seat"]) == {"DECL-DUPLICATE"}, flagged
    # Neither half of the contradiction may be taken: the second says BY DECREE
    # over `ZZ.00`, which would have made this seat ARENA-MISSING on a tiebreak
    # nobody chose. It falls back to the cells and stays in the UNDECLARED list.
    assert not dup["declared"] and dup["held"] == "VACANT", dup
    assert dup["arena_refs"] == ["OK.01"], dup
    # 6. A DECLARED FAMILY REFERENCE SURVIVES THE PARSE. The first draft cleaned
    #    markdown emphasis off the whole declaration line before reading it,
    #    which turned the live `ARENA: PL.*` into `PL.` — a syntax written to
    #    stop seats looking falsifiable when they are not, quietly doing the
    #    reverse to the one seat whose ring is a whole family.
    fam = seat_by_name["Declared seat naming a family"]
    assert fam["arena_refs"] == ["OK.*"], fam["arena_refs"]
    assert fam["arena_present"] == ["OK.01", "OK.02", "OK.03", "OK.04", "OK.05",
                                    "OK.06", "OK.07", "OK.08",
                                    "OK.09"], fam["arena_present"]
    # 7. PROSE IN THE ARENA FIELD DECLARES NOTHING, and must not be readable as
    #    an unbuilt ring — the difference between "no ring exists" and "the
    #    author wrote a sentence" is the whole value of `ARENA: NONE`.
    prose = seat_by_name["Seat declaring prose where ids belong"]
    assert flagged.get(prose["seat"]) == {"DECL-EMPTY-ARENA", "UNCONTESTED"}, flagged
    assert not prose["declared"] and prose["arena_refs"] == ["OK.03"], prose
    # 8. The undeclared list is the fallback made visible.
    still = set(undeclared(seats))
    assert {"Healthy verdict seat", "Phantom arena seat",
            "Vacant by default words"} <= still, still
    assert not ({over["seat"], unbuilt["seat"]} & still), still
    deltas = {(seat, was.split()[0]) for seat, was, _now in declaration_deltas(seats)}
    assert (over["seat"], "arena") in deltas and (over["seat"], "held") in deltas, deltas

    # ...and the healthy seats are untouched. A "Superseded context" heading is
    # not a decree, so ZZ.09 must not appear anywhere.
    for ok in ("Default seat a claim spec defended",):
        assert ok not in flagged, (ok, flagged)
    assert not any("ZZ.09" in w for _k, _s, w in violations), violations
    # The range must expand, or `W.1–W.7` undercounts by five.
    healthy = [s for s in seats
               if s["seat"] == "Default seat a claim spec defended"][0]
    assert healthy["arena_refs"] == ["OK.01", "OK.02"], healthy["arena_refs"]
    # A VACANT seat is never UNCONTESTED — nobody is sitting in it.
    vacant = [s for s in seats if s["seat"] == "Vacant by default words"][0]
    assert vacant["held"] == "VACANT", vacant["held"]


def main(argv: List[str]) -> int:
    _fixture()

    from .protocol import Ledger
    from .registry import BY_ID

    led = Ledger()
    text = DOC.read_text()
    violations, seats = audit(text, BY_ID, lambda sid: led.status(sid).value)

    print(f"\nChampion seats — {DOC.relative_to(DOC.parent.parent)} "
          f"({len(seats)} seats, {len(BY_ID)} specs registered)\n")
    width = min(42, max(len(s["seat"]) for s in seats))
    by_seat: Dict[str, List[str]] = {}
    for kind, seat, _why in violations:
        by_seat.setdefault(seat, []).append(kind)

    for s in seats:
        flags = ", ".join(by_seat.get(s["seat"], [])) or "ok"
        refs = []
        for r in s["arena_refs"]:
            hit = resolve(r, BY_ID)
            refs.append(r if hit else f"{r}!")
        arena = " ".join(refs) or "(none named)"
        src = "decl " if s.get("declared") else "PROSE"
        print(f"  {s['seat'][:width]:{width}}  {src} {s['held']:<11}  {flags}")
        print(f"  {'':{width}}  arena: {arena[:96]}")

    print("\n  ! = named as this seat's arena, absent from the registry.")
    print("  decl = HELD/ARENA declared in the file; PROSE = inferred from the "
          "cells.\n")

    # WHERE THE DECLARATION DISAGREES WITH THE INFERENCE. Printed unconditionally
    # because it is the only evidence that retiring the inference bought
    # anything, and because each line is a claim about this document that a
    # reader can check against the cell in about ten seconds.
    deltas = declaration_deltas(seats)
    if deltas:
        print(f"  the declaration CHANGED the reading on {len(deltas)} "
              f"seat/field(s) — inference on the left:")
        for seat, was, now in deltas:
            print(f"    {seat[:44]:<44} {was[:60]}")
            print(f"    {'':44} -> {now[:80]}")
        print()

    if violations:
        print(f"  {len(violations)} violation(s):")
        for kind, seat, why in violations:
            print(f"    [{kind:<13}] {seat[:width]}")
            print(f"       {why}")
        print()

    # Requirement 4: what the REAL arenas say. A seat whose arena exists is
    # answerable today, and its ledger line is the whole answer — this is the
    # half of the file that works, and printing it keeps the report from
    # reading as an indictment of a mechanism that is mostly sound.
    live: Dict[str, Tuple[str, List[str]]] = {}
    for s in seats:
        for sid, st in s.get("arena_status", {}).items():
            live.setdefault(sid, (st, []))[1].append(s["seat"][:24])
    if live:
        print(f"  arenas that DO exist ({len(live)}), and where they stand:")
        for sid in sorted(live):
            st, who = live[sid]
            print(f"    {sid:<10} {st:<8} {', '.join(sorted(set(who)))}")
        print()

    # The BY ANALYSIS seats: not violations (see audit()), but the file's own
    # rule 1 calls them "pending its arena run", and pending is a promise with
    # a due date only if somebody prints it.
    # Same quantifier as audit()'s, for the same reason: `all(... NOT_RUN)`
    # would call a BY ANALYSIS seat "defended" on one VOID or one fixture.
    pending = [(s["seat"], s["arena_present"]) for s in seats
               if s["held"] == "BY ANALYSIS" and s["arena_present"]
               and not s.get("challenger_runs")]
    # The residual soft spot in _challenger_runs, printed instead of assumed.
    kindless = _spec_kinds(BY_ID)
    weak = [(s["seat"], s["challenger_runs"]) for s in seats
            if s.get("challenger_runs")
            and not any(kindless.get(i) for i in s["challenger_runs"])]
    if weak:
        print("  seats discharged ONLY by arena specs that declare no COVERS "
              "kind — the\n  contest cannot be verified from the registry; "
              "declare the kind to settle it:")
        for seat, runs in weak:
            print(f"    {seat[:44]:<44} {', '.join(runs)}")
        print()

    if pending:
        print("  held BY ANALYSIS with a real arena that has never run "
              "(declared pending, not a violation):")
        for seat, arenas in pending:
            print(f"    {seat[:width]:{width}}  {', '.join(arenas)}")
        print()

    # The closability split. A seat here cannot be discharged by building
    # anything, so reporting it beside the register-me seats — which is what
    # this file did for five audits — costs a builder iteration each time.
    unclosable = [(s["seat"], s["arena_unregisterable"]) for s in seats
                  if s.get("arena_unregisterable")]
    if unclosable:
        print("  ARENA-MISSING that REGISTERING CANNOT FIX — the cited id was "
              "decided against,\n  so the repair is to correct the citation to "
              "the live successor:")
        for seat, refs in unclosable:
            for r in refs:
                print(f"    {seat[:44]:<44} {r:<7} {UNREGISTERABLE[r]}")
        print()

    missing = sum(1 for k, _, _ in violations if k == "ARENA-MISSING")
    dead = unfalsifiable(seats)
    welded = unreachable_arena(seats)

    # The seats nothing could unseat, printed whether or not `--check` is on: it
    # is the number the file's whole mechanism rests on, and until 2026-08-30 no
    # report showed it while three of these seats sat quietly under `NO-ARENA`.
    print(f"  UNFALSIFIABLE — no registered spec resolves from the arena cell, "
          f"so nothing\n  that could be run would unseat the holder "
          f"({len(dead)}/{BASELINE_UNFALSIFIABLE}):")
    for seat in dead:
        print(f"    {seat[:70]}")
    print()

    # The seats whose arena exists and can never run — same disease through the
    # dependency graph instead of the symbol table (54th audit B4). Ratcheted
    # in the TOTAL below.
    print(f"  ARENA-UNREACHABLE — the arena resolves, but every member still "
          f"owing a verdict\n  is parked, VOID-FORECLOSED, or transitively "
          f"behind one ({len(welded)}):")
    for seat in welded:
        s = next(x for x in seats if x["seat"] == seat)
        roots = sorted(set(s["arena_unreachable"].values()))
        print(f"    {seat[:52]:<52} rooted at {', '.join(roots)}")
    if not welded:
        print("    (none)")
    print()

    # Out of the ratchet's scope by marking, printed so the scope is a choice
    # the reader can see: a VACANT/UNDECIDED/BY ANALYSIS seat with a welded
    # ring is a seat nobody can ever WIN.
    unwinnable = [s["seat"] for s in seats
                  if s.get("arena_pending_dead") and not s.get("arena_welded")]
    if unwinnable:
        print("  ...and seats no one can ever WIN — every pending arena member "
              "welded, but no\n  unearned holder to indict (out of the ratchet "
              "by scope, not oversight):")
        for seat in unwinnable:
            print(f"    {seat[:70]}")
        print()

    still = undeclared(seats)
    print(f"  UNDECLARED — no SEAT:/HELD:/ARENA: line, so this report's marking "
          f"and arena for\n  it are a parse of prose "
          f"({len(still)}/{BASELINE_UNDECLARED}):")
    for seat in still:
        print(f"    {seat[:70]}")
    if not still:
        print("    (none — every seat says what would unseat it)")
    print()

    if "--check" in argv:
        if len(still) > BASELINE_UNDECLARED:
            print(f"  RATCHET BROKEN: {len(still)} seat(s) have no declaration, "
                  f"baseline {BASELINE_UNDECLARED}. Add one line per seat to\n"
                  f"  docs/CHAMPIONS.md — `- SEAT: <name> | HELD: <marking> | "
                  f"ARENA: <ids or NONE>`. A seat\n  whose marking is guessed "
                  f"from prose is a seat whose contestability is guessed.\n")
            return 1
        # BOTH ratchets block. Counting only ARENA-MISSING rewarded deleting the
        # arena reference — the repair this file forbids in bold — because that
        # converts the seat to NO-ARENA and the number falls. See
        # BASELINE_UNFALSIFIABLE.
        if len(dead) > BASELINE_UNFALSIFIABLE:
            print(f"  RATCHET BROKEN: {len(dead)} seats have no runnable arena at "
                  f"all, baseline {BASELINE_UNFALSIFIABLE}. It may shrink, never\n"
                  f"  grow — and it shrinks ONLY by registering a spec or "
                  f"correcting a citation to a live successor.\n")
            return 1
        # The TOTAL, per T0.31's precedent — the new class does not get a
        # private zero. Asserted as a sum so a seat converting between the two
        # classes (a spec registered but still unreachable) is neither
        # progress nor regression, exactly like ARENA-MISSING -> NO-ARENA.
        if len(dead) + len(welded) > BASELINE_UNCONTESTABLE:
            print(f"  RATCHET BROKEN: {len(dead)} unfalsifiable + {len(welded)} "
                  f"arena-unreachable seat(s) = {len(dead) + len(welded)}, "
                  f"baseline {BASELINE_UNCONTESTABLE}.\n  The total may shrink, "
                  f"never grow — it shrinks by registering a runnable spec,\n"
                  f"  re-parenting an arena member off its foreclosed root, or "
                  f"correcting a citation —\n  never by parking, foreclosing, or "
                  f"deleting a reference.\n")
            return 1
        if missing > BASELINE_ARENA_MISSING:
            print(f"  RATCHET BROKEN: {missing} seats name a non-existent arena, "
                  f"baseline {BASELINE_ARENA_MISSING}. It may shrink, never grow —\n"
                  f"  and it shrinks by REGISTERING the spec, not by deleting the "
                  f"arena reference.\n")
            return 1
        print(f"  ratchet ok ({missing}/{BASELINE_ARENA_MISSING} seats with a "
              f"phantom arena; {len(dead)}/{BASELINE_UNFALSIFIABLE} "
              f"unfalsifiable;\n  {len(dead)}+{len(welded)}/"
              f"{BASELINE_UNCONTESTABLE} uncontestable in total, arena-"
              f"unreachable included).\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
