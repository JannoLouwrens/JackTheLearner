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

MARKINGS ARE INFERRED HERE, AND THAT IS A KNOWN WEAKNESS. `coverage.py` and
`decisions.py` both retired prose-reading in favour of an explicit marker
(`COVERS:`, `DECIDE:`) after it flattered its author twice. This file cannot do
the same, because CHAMPIONS.md is a human-facing table with no declaration
syntax and this module is not permitted to invent one unilaterally. So `held` is
read from a dedicated table COLUMN — a structural field, not free prose, which
is the strongest thing actually available — and the champion cell is consulted
only as a fallback for rows whose `held` cell is `—`. The durable repair is a
`HELD:`/`ARENA:` declaration per seat, proposed and not taken here; until then
treat the marking column of this report as evidence and the ARENA column as
fact, since arena ids resolve against the registry and cannot be flattered.

ARENA IDS, BY CONTRAST, ARE FACTS. They are extracted from the arena COLUMN only
(never the champion or challenger cells — a challenger cell mentioning `UB.11`
is not an arena), then resolved against the live registry. Ranges are expanded
(`LC.00–LC.06`, `ME.11.A–F`, `W.1–W.7`) because a range is where a phantom
hides: `W.1–W.7` names seven arenas and reads as a whole programme of fidelity
gates; the registry contains no `W.*` spec whatsoever.

THE DECREE SECTIONS ARE SEATS, and a table-only parser misses the worst case.
`### DECIDED BY DECREE 2026-08-09: PLASTIC ONLY` is not a table row, but it
holds a component of Jack (every encoder's plasticity) by owner decree and
pre-registers its own re-open trigger — that is a seat in everything but
formatting. Parsing only the table would report this file clean on `PL.00`, the
single most-cited missing arena in the project. The `## NOT SEATS` section is
deliberately NOT parsed: it holds the DEFINITION of Jack, which by its own words
no arena may touch, so it is correct for it to name none.

RATCHET, NOT GATE. `--check` fails only if ARENA-MISSING grows past the baseline
below. Eight seats are in violation today; a guard that fails everywhere on day
one is one nobody keeps green, and a guard nobody keeps green is decoration
(LESSONS.md; the `citations.py` and `decisions.py` precedent). Each registration
lowers the number and the constant follows it down.

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
BASELINE_ARENA_MISSING = 8

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


def audit(text: str, by_id: dict,
          status: Callable[[str], str]) -> Tuple[List[Tuple[str, str, str]], List[dict]]:
    """Return (violations, seats). A violation names a seat, never a spec."""
    seats = parse(text)
    violations: List[Tuple[str, str, str]] = []

    for s in seats:
        if s["kind"] == "malformed":
            violations.append(("MALFORMED-ROW", s["seat"],
                               "a table row this parser could not read as a seat"))
            continue
        refs = s["arena_refs"]
        s["arena_missing"] = [r for r in refs if not resolve(r, by_id)]
        s["arena_present"] = sorted({i for r in refs for i in resolve(r, by_id)})
        s["arena_status"] = {i: status(i) for i in s["arena_present"]}

        if not refs:
            violations.append(("NO-ARENA", s["seat"],
                               "names no arena spec id at all — nothing that could "
                               "ever be run would unseat the holder"))
        elif s["arena_missing"]:
            violations.append(("ARENA-MISSING", s["seat"],
                               f"names {', '.join(s['arena_missing'])} — not in the "
                               f"registry, so the seat looks contestable and is not"))

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
        if (s["held"] in HELD_UNEARNED and s["arena_present"]
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
| No arena at all | incumbent | **BY DECREE** (owner) | HR bakeoff (queued) | a challenger |
| Uncontested decree seat | incumbent | **BY DECREE** (owner) | OK.03 (registered) | a challenger |
| Vacant by default words | **VACANT** — the incumbent by default is nobody | — | OK.01 (registered) | a challenger |

### DECIDED BY DECREE 2099-01-01: SOMETHING

WHAT STILL RUNS: ZZ.02 (a floor nobody wrote). Cheap, CPU.

### Superseded context: not a decree, must not be parsed as one

This section names ZZ.09 and must contribute no seat and no violation.
"""
    class _S:
        def __init__(self, notes=""):
            self.notes = notes

    # OK.04/OK.05 are declared support kinds; OK.06 ran but only to VOID.
    by_id = {"OK.01": _S("COVERS: smell (claim)"), "OK.02": _S(), "OK.03": _S(),
             "OK.04": _S("COVERS: smell (fixture)"),
             "OK.05": _S("COVERS: balance (sensor)"), "OK.06": _S()}
    ran = {"OK.01": "PASS", "OK.02": "PASS", "OK.03": "NOT_RUN",
           "OK.04": "PASS", "OK.05": "PASS", "OK.06": "VOID"}
    violations, seats = audit(doc, by_id, lambda sid: ran.get(sid, "NOT_RUN"))
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
    assert flagged.get("No arena at all") == {"NO-ARENA"}, flagged
    assert flagged.get("Uncontested decree seat") == {"UNCONTESTED"}, flagged
    # The decree section is a seat and its dangling `WHAT STILL RUNS` id is the
    # PL.00 case in miniature: outside the table, cited as live, unregistered.
    decree = [s for s in seats if s["kind"] == "decree"]
    assert len(decree) == 1, [s["seat"] for s in seats]
    assert decree[0]["arena_missing"] == ["ZZ.02"], decree[0]
    assert flagged.get(decree[0]["seat"]) == {"ARENA-MISSING"}, flagged
    # ...and the healthy seats are untouched. A "Superseded context" heading is
    # not a decree, so ZZ.09 must not appear anywhere.
    for ok in ("Healthy verdict seat", "Default seat a claim spec defended",
               "Vacant by default words"):
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
        print(f"  {s['seat'][:width]:{width}}  {s['held']:<11}  {flags}")
        print(f"  {'':{width}}  arena: {arena[:96]}")

    print("\n  ! = named as this seat's arena, absent from the registry.\n")

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

    missing = sum(1 for k, _, _ in violations if k == "ARENA-MISSING")
    if "--check" in argv:
        if missing > BASELINE_ARENA_MISSING:
            print(f"  RATCHET BROKEN: {missing} seats name a non-existent arena, "
                  f"baseline {BASELINE_ARENA_MISSING}. It may shrink, never grow —\n"
                  f"  and it shrinks by REGISTERING the spec, not by deleting the "
                  f"arena reference.\n")
            return 1
        print(f"  ratchet ok ({missing}/{BASELINE_ARENA_MISSING} seats with a "
              f"phantom arena).\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
