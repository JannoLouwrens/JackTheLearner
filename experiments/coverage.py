"""Does the ladder actually cover what GOAL.md promises?

WHY THIS EXISTS. On 2026-08-10 the ladder held 154 specs and the project had
been running autonomously for days. A hand audit against GOAL.md found FOUR
constitutional commitments — the owner's own words — with **zero** falsifiable
claims behind them:

    "too cold is going to kill him"          -> no thermal spec at all
    "he builds a shelter"                    -> no shelter spec at all
    "he dies, retries, remembers across lives"-> nothing about surviving a death
    damage as something learnable            -> no nociception spec at all

None of that was hidden. Every organ was working: the builder was demonstrating
9-11 specs a day, the overseer was auditing direction, the reviewer was
rewriting weak specs. They were all reasoning about **specs that exist**. A
missing spec has no id, appears in no `run blocked` ranking, blocks nothing, and
fails no gate — it is invisible to every instrument the system owns.

`run status` answers "how much of the ladder is demonstrated". This answers the
question that outranks it: **"is the ladder the right ladder?"**

COVERAGE IS DECLARED, NEVER INFERRED — and that took two scars to learn.

  Scar 1 (a false NEGATIVE, found in a day). `BA.01` was registered
  specifically to close the `balance` hole, titled "He feels himself falling
  before he falls", and the balance regex did not match it. The gap-finder had
  a gap. The tempting repair — adding "fall" to the pattern — is how a detector
  gets tuned until it agrees with its maintainer, so the repair was an explicit
  `COVERS: <commitment>` marker instead: a deliberate statement by the spec's
  author, which cannot be matched by accident.

  Scar 2 (a false POSITIVE, unnoticed for two more days, and worse). The regex
  stayed as a "safety net" that still granted coverage on its own, and it
  granted a lot of it. Measured 2026-08-10, by matching all patterns against
  all 160 titles and reading every hit: the passing spec credited to the
  owner's own image of success, *"he builds a shelter"*, was `ME.11.0`, *"The
  paraphrase eval set is **honest** before anyone is scored"* — `nest` inside
  `ho-nest`. Proprioception's PASS was `PG.3`, *"Ladder is c-**limb**-able"*.
  `dies` matched inside `bo-dies`. `sound` matched *"physically sound"* — sound
  as in valid.

  The two directions are not symmetric. A false negative gets fixed the day its
  author notices their spec is not counted: they are motivated, present, and
  looking straight at it. A false positive is credit nobody audits, because
  **nobody goes looking for coverage they believe they already have**.

So a regex hit is now a NOMINATION — visible work, never coverage. Only a
`COVERS:` declaration counts toward `n_specs`/`n_pass`. A misspelt declaration
is an ERROR, not a silence: `check()` returns nonzero on it, because a marker
that buys nothing while looking like a claim is scar 2 wearing a new hat.

MATCH ON TITLES, NOT ON EVERYTHING. The nomination scan reads titles only.
Searching the whole spec text finds "temperature" inside an unrelated note and
nominates specs that are about something else — measured: the loose search
claimed 2 thermal specs and both were incidental mentions.

HOW IT AVOIDS ROTTING. The commitment list below is hand-maintained, which is
exactly what went stale about `ladder_prompt.md`'s cached counts. Two defences:
`check()` returns a nonzero count that an organ can act on rather than a wall of
prose, and any commitment added to GOAL.md without a line here shows up as a
GOAL.md section this file cannot name — which the overseer is told to look for.
Better would be deriving these from GOAL.md automatically; that is not
attempted, because a regex over prose that silently matches nothing is worse
than a list a human can read and correct.

A PARKED SPEC IS NOT COVERAGE (28th audit, 2026-08-25 — the third scar). At
00:11 that morning the loop retired `SH.01` under its own pre-registered rule
("no ledger row, no envelope growth, no re-roll") — the correct call on the
evidence, and exactly the conduct this system asks for. But `SH.01` was the
ONLY claim-kind spec behind BOTH `shelter/building` and `thermal (kills)`, two
of the four original 2026-08-10 misses that caused this file to exist — and
this tool printed `0 commitment(s) with NO declared spec` and exited 0,
because a parked spec is still a declaration and the ratchet counts
declarations. `smell` had been in the same state via `SM.02` for five days.
The distinction the tool did not draw: **blocked is a queue position; parked
is a retirement.** A spec pre-registered never to run again is not a
falsifiable claim behind a commitment — it is a docstring. So a spec whose
notes carry `PARKED: <YYYY-MM-DD> — <reason>` no longer counts as a
declaration, a commitment with no passing claim and no un-parked claim-kind
declaration prints as claim-dead and `check()` exits 2, and the repair for
that red is to REGISTER A SUCCESSOR SPEC, never to delete the marker or
quiet the tool. Write the marker as its own sentence (the `COVERS:` grammar
consumes to end of sentence, so a marker glued onto a `COVERS:` line would be
swallowed into a malformed declaration). A malformed `PARKED:` — no date, no
reason — is REPORTED, never dropped: an unparseable retirement leaves the
spec silently counting as coverage, which is the false-positive direction,
the one nobody audits.

A CITATION IN GOAL.md IS A PROMISE (29th audit, 2026-08-25 — the fourth
scar). `GOAL.md` cited sixteen spec ids and FIVE did not exist in the
registry — `LG.00`, `GEN.02`, `GEN.03`, `GEN.06`, `GEN.09` — one of them the
test the constitution itself calls "the proof he is a creature and not a
costume". The gap stood open since 2026-08-09 with every organ green,
because this module's unit is the COMMITMENT and `GOAL.md` makes claims one
level finer: `language (parent)` read covered-and-passing while its named
falsifier was never registered. The project had already built this exact
check twice — `champions.py` for `CHAMPIONS.md` arenas, `T0.21` P10 for
docstring markers — and neither generalisation reached the document all the
others defer to. So `goal_citations()` resolves every spec-shaped id in
`GOAL.md` against `BY_ID`; a NEW dangling citation exits 2 (a promise the
constitution just made that the ladder cannot keep), the seeded baseline of
five is standing registration debt reported but not fatal (it is B1(a)'s
work; a permanently red check trains its reader to ignore red), and a
baseline entry that RESOLVES must be removed from the baseline — shrink-only,
enforced at exit 1 like a malformed marker.

Guarded by spec `T0.21`, which feeds it the cases already known to be broken.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# name -> (NOMINATION regex over spec TITLES, why this is constitutional)
#
# These patterns grant NOTHING. They nominate specs that look related so an
# undeclared one reads as work to do. Word boundaries are a cheap partial fix
# for scar 2 — they kill honest->nest, climbable->limb, bodies->dies — but they
# do not fix "physically sound" or "curiosity drives", which is precisely why
# nomination and coverage had to be separated rather than the patterns tuned.
COMMITMENTS: Dict[str, Tuple[str, str]] = {
    "sight":              (r"\b(camera|visual|vision|eye|eyes|see|retina)\b", "every sense a human has"),
    "hearing":            (r"\b(audio|acoustic|sound|hear|hears|heard|binaural|auditory)\b", "every sense a human has"),
    "touch/contact":      (r"\b(touch|tactile|contact)\b", "every sense a human has"),
    "smell":              (r"\b(odour|odor|smell|olfact\w*)\b", "owner named it constitutional"),
    "taste":              (r"\b(taste|gustat\w*|poison)\b", "owner named it constitutional"),
    "voice":              (r"\b(vocal|voice|utterance|speak|signal)\b", "owner: he must have a voice"),
    "balance":            (r"\b(balance|topple|upright|vestibul\w*)\b", "every sense a human has"),
    "proprioception":     (r"\b(propriocept\w*|body schema|limb)\b", "every sense a human has"),
    "thermal (kills)":    (r"\b(thermal|temperature|cold|freez\w*|heat)\b", "owner: too cold/hot KILLS him"),
    "damage/nociception": (r"\b(damage|injur\w*|nocicept\w*|pain)\b", "danger must be learnable before fatal"),
    "hunger/thirst":      (r"\b(hunger|thirst|drink|forage|drive|drives)\b", "owner: permanent human needs"),
    "sleep":              (r"\b(sleep|consolidat\w*|siesta)\b", "biology as oracle"),
    "death & retry":      (r"\b(death|dies|lethal|surviv\w*|statue)\b", "owner: he dies and retries"),
    "memory across lives":(r"\b(across lives|between lives|prior life|erase)\b", "owner: REMEMBERS across lives"),
    "shelter/building":   (r"\b(shelter|build\w*|construct\w*|nest\w*)\b", "owner's own image of success"),
    "tool use":           (r"\b(tool|tools|affordance\w*)\b", "caveman realism"),
    "language (parent)":  (r"\b(language|word|words|grounding|lexic\w*)\b", "LLM as talkative parent"),
    "social/other agents":(r"\b(social|companion|two jacks|second jack)\b", "owner: socialising makes him kind"),
    "curiosity":          (r"\b(curiosity|novelty|exploration|learning progress)\b", "the world is the teacher"),
    "one brain / unison": (r"\b(unison|fused|fusion|binding|cross-modal|shared|one brain)\b", "the constitution itself"),
    "plasticity":         (r"\b(plastic\w*|frozen|does not die|forgetting)\b", "PLASTIC ONLY decree"),
    "generality":         (r"\b(generalis\w*|generaliz\w*|held.out|unseen|transfer\w*)\b", "GEN.00, the final exam"),
    "fast/slow":          (r"\b(deliberat\w*|habit|slow path|lookahead)\b", "owner 2026-08-10"),
}

# `COVERS: a, b` — consumes to end of line, sentence, or string. A spec may
# carry several markers. Names never contain a comma, a period or a semicolon.
# Two guards separate a DECLARATION from a PROSE MENTION, because T0.24's
# notes — "declares NO `COVERS:` commitment" — were read by the bare pattern
# as a malformed declaration named "` commitment", invented from the sentence
# disclaiming one: the marker may not be preceded by a backtick, and the name
# must start with a word character. Either alone stops that artifact; both,
# because a false malformed-declaration report trains its reader to ignore
# the real ones (the LESSONS.md staleness-detector rule).
DECLARATION = re.compile(r"(?<!`)COVERS:\s*(\w[^\n.;]*)", re.I)

# A declaration carries a KIND: `COVERS: curiosity (fixture)`. A missing kind
# is REPORTED like a malformed declaration — it buys nothing.
#
# WHY (Overseer, 8th-10th audits). `n_pass` answered "has this commitment been
# demonstrated", and two passing specs made constitutional commitments read as
# demonstrated when nothing had been: PG.4 passing proved the noisy-TV PANEL
# traps a naive agent — apparatus for a curiosity claim, not one — and LC.01
# passing proved the ADMISSION RULE excludes unbound cores, not that any brain
# binds. With them credited, `curiosity` and `one brain / unison` each read
# 1 pass and the standing zero-pass rule could not see either hole.
#
#   claim   — a capability test that could have failed; the ONLY kind n_pass counts
#   fixture — apparatus a claim will need (a trap, a world property)
#   rule    — a gate/admission criterion enforced on candidates
#   sensor  — an instrument measures/emits a channel; nothing acts on it yet
#
# WHY ABSENT IS AN ERROR AND NOT A DEFAULT (Overseer, 12th audit). v1 of the
# kind mechanism defaulted a kindless declaration to `claim`. The mechanism
# shipped, was applied to 2 of 78 declarations, and the other 76 inherited the
# default — at least ten of them apparatus or sensor-legibility by their own
# titles — so `coverage.py` reported 9 zero-pass commitments when the honest
# figure was 15+, and the standing zero-pass rule steered off the flattered
# list for two days. A default on a field that routes work IS the defect: the
# only safe meaning for silence is a report. (The old defaulting rule stays
# executable via `default_kind=` because T0.21 keeps it as the control that
# must fail.)
#
# Parsing order is load-bearing: canonical names themselves end in parentheses
# — `thermal (kills)`, `language (parent)` — so the full name is looked up
# FIRST and a trailing `(kind)` is stripped only when that fails. An
# unrecognised kind is REPORTED like any malformed declaration, never dropped:
# `(fixure)` reads as a claim to a human and must not silently buy one.
KINDS = ("claim", "fixture", "rule", "sensor")
_KIND = re.compile(r"^(.*\S)\s*\(\s*([\w-]+)\s*\)$")

# `PARKED: 2026-08-25 — reason` — the spec's own decision tree retired it: no
# re-run, no envelope growth, no re-roll. Same anti-prose guard as DECLARATION
# (a backticked mention is discussion, not a retirement). The date and dash are
# REQUIRED: a bare `PARKED: soon` parses as nothing, and a marker that parses
# as nothing leaves the spec counting as coverage — so it is reported like a
# malformed COVERS, loudly, in `bad`.
PARKED_MARK = re.compile(r"(?<!`)PARKED:\s*([^\n]*)")
_PARKED_OK = re.compile(r"^(\d{4}-\d{2}-\d{2})\s*[—–-]\s*(\S.*)$")

_CANON = {k.lower(): k for k in COMMITMENTS}


def parked(by_id: Optional[dict] = None
           ) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """`(spec id -> 'date — reason', [(spec id, malformed marker)])`.

    A spec is parked when its registry notes carry a well-formed
    `PARKED: <YYYY-MM-DD> — <reason>` marker. Malformed markers are the second
    half and they are the point: an unparseable retirement silently leaves the
    spec counting as coverage — the false-positive direction, the one nobody
    goes looking for.
    """
    if by_id is None:
        from .registry import BY_ID
        by_id = BY_ID
    out: Dict[str, str] = {}
    bad: List[Tuple[str, str]] = []
    for sid, spec in by_id.items():
        for raw in PARKED_MARK.findall(str(getattr(spec, "notes", "") or "")):
            m = _PARKED_OK.match(raw.strip())
            if m:
                out.setdefault(sid, f"{m.group(1)} — {m.group(2)}")
            else:
                bad.append((sid, f"PARKED: {raw.strip()!r}  [needs "
                                 f"'PARKED: YYYY-MM-DD — reason']"))
    return out, bad


GOAL_MD = Path(__file__).resolve().parent.parent / "GOAL.md"

# Spec-shaped id: 1-4 capitals, optional tier digit, then .NN — matches T5.03
# and GEN.06 alike. "π0.5" and bare version numbers have no capital prefix and
# do not match; a capitalised non-spec token like "U.S." would dangle LOUDLY,
# which is the safe failure direction for a citation checker.
GOAL_CITATION = re.compile(r"\b([A-Z]{1,4}[0-9]?\.[0-9]{1,2})\b")

# The five citations measured dangling on 2026-08-25 (29th audit). This set
# may ONLY shrink: registering one of these makes `goal_citations()` demand
# its removal here, and a NEW dangler is never added — it is a red exit.
GOAL_DANGLING_BASELINE = frozenset(
    {"LG.00", "GEN.02", "GEN.03", "GEN.06", "GEN.09"})


def goal_citations(text: Optional[str] = None,
                   by_id: Optional[dict] = None,
                   baseline: frozenset = GOAL_DANGLING_BASELINE) -> dict:
    """Resolve every spec-shaped id `GOAL.md` cites against the registry.

    Returns `{"cited", "dangling", "new", "known", "stale_baseline"}` —
    `new` (dangling and NOT in the baseline) is the fatal class: the
    constitution just promised a falsifier nobody registered, the exact hole
    that stood open 16 days. `known` is seeded registration debt (B1(a)).
    `stale_baseline` (baseline entries that now resolve) must be deleted from
    `GOAL_DANGLING_BASELINE` in the same commit that registered them, so the
    baseline only shrinks; leaving one would let the id dangle AGAIN later
    without a red.
    """
    if by_id is None:
        from .registry import BY_ID
        by_id = BY_ID
    if text is None:
        text = GOAL_MD.read_text()
    cited = sorted(set(GOAL_CITATION.findall(text)))
    dangling = {i for i in cited if i not in by_id}
    return {
        "cited": cited,
        "dangling": sorted(dangling),
        "new": sorted(dangling - baseline),
        "known": sorted(dangling & baseline),
        "stale_baseline": sorted(i for i in baseline if i in by_id),
    }


def declarations(by_id: Optional[dict] = None,
                 default_kind: Optional[str] = None
                 ) -> Tuple[Dict[str, List[Tuple[str, str]]],
                            List[Tuple[str, str]]]:
    """Read every spec's `COVERS:` markers.

    Returns `(commitment -> [(spec id, kind)], [(spec id, unrecognised name)])`.
    The second half is the point: a declaration naming a commitment that does
    not exist — carrying a kind that is not one of `KINDS` — or carrying NO
    kind at all — is reported, never dropped and never defaulted. A typo'd
    marker looks exactly like a claim to a human reader and buys exactly
    nothing from this file, which is the false-positive failure this module
    was rewritten to end; a kindless marker silently defaulting to `claim` was
    the same failure one level up (76 of 78 declarations, 12th audit).

    `default_kind` is THE ORGAN THAT FAILED, kept executable: pass `"claim"`
    to get the pre-2026-08-13 defaulting behaviour. Only T0.21's control may
    want that.
    """
    if by_id is None:
        from .registry import BY_ID
        by_id = BY_ID
    declared: Dict[str, List[Tuple[str, str]]] = {k: [] for k in COMMITMENTS}
    bad: List[Tuple[str, str]] = []
    for sid, spec in by_id.items():
        for group in DECLARATION.findall(str(getattr(spec, "notes", "") or "")):
            for raw in group.split(","):
                name = raw.strip()
                if not name:
                    continue
                # Full name first: `thermal (kills)` is a commitment, not a
                # kind annotation. Only an unmatched trailing paren is a kind.
                canon, kind = _CANON.get(name.lower()), None
                if canon is None:
                    m = _KIND.match(name)
                    if m and m.group(2).lower() in KINDS:
                        canon = _CANON.get(m.group(1).strip().lower())
                        kind = m.group(2).lower()
                if canon is None:
                    bad.append((sid, name))
                    continue
                if kind is None:
                    kind = default_kind
                if kind is None:
                    bad.append((sid, f"{name}  [KINDLESS — say (claim), "
                                     f"(fixture), (rule) or (sensor)]"))
                elif sid not in [i for i, _ in declared[canon]]:
                    declared[canon].append((sid, kind))
    return declared, bad


def report(by_id: Optional[dict] = None,
           results: Optional[dict] = None,
           credit_parked: bool = False) -> List[dict]:
    """Coverage (declared) and nominations (regex), never mixed.

    `n_specs`/`n_pass` count DECLARED specs only. `nominations` lists specs a
    pattern matched that have not declared — work to do, not coverage.

    `n_pass` counts passing `claim` declarations ONLY. A passing fixture, rule
    or sensor is real work and is reported in `support_pass` — but apparatus
    demonstrating itself is not the commitment being demonstrated, and merging
    the two is how `curiosity` and `one brain / unison` each read as started
    for three audits while no capability test had ever run.

    A PARKED spec is excluded from `specs`/`kinds`/`n_specs` and reported in
    the row's `parked` map instead — a retirement is not a declaration (28th
    audit). Excluding it from `n_pass` too is the conservative direction: the
    only spec-parking precedents (SH.01, SM.02, UB.10) all concluded WITHOUT a
    ledger PASS, and a marker that could keep credit while retiring the run
    would flatter coverage, the direction nobody audits.

    `credit_parked` is THE ORGAN THAT FAILED, kept executable — the
    pre-2026-08-25 behaviour under which SH.01's retirement left two
    constitutional commitments reading as covered. Only T0.21's control may
    want it (same pattern as `declarations(default_kind=)`).
    """
    if by_id is None:
        from .registry import BY_ID
        by_id = BY_ID
    if results is None:
        results = {}
        p = Path(__file__).resolve().parent / "ledger.json"
        if p.is_file():
            results = json.load(open(p)).get("results", {})
    declared, bad = declarations(by_id)
    parked_map, parked_bad = parked(by_id)
    if credit_parked:
        parked_map = {}
    bad = bad + parked_bad
    out = []
    for name, (pat, why) in COMMITMENTS.items():
        rx = re.compile(pat, re.I)
        all_pairs = [(i, k) for i, k in declared[name] if i in by_id]
        pairs = [(i, k) for i, k in all_pairs if i not in parked_map]
        specs = [i for i, _ in pairs]
        nominated = [s.id for s in by_id.values()
                     if rx.search(s.title) and s.id not in specs]
        # Status alone, deliberately — same call as `senses.py`, same reason:
        # coverage asks whether a commitment was ever demonstrated, not whether
        # the certificate is current. See `Ledger.unsatisfied` for the path
        # where freshness IS load-bearing, and `run stale` for the report.
        passing = [i for i, k in pairs if k == "claim"
                   and results.get(i, {}).get("status") == "PASS"]
        support = {i: k for i, k in pairs if k != "claim"
                   and results.get(i, {}).get("status") == "PASS"}
        out.append({"commitment": name, "why": why, "specs": specs,
                    "kinds": dict(pairs),
                    "parked": {i: k for i, k in all_pairs if i in parked_map},
                    "n_specs": len(specs), "n_pass": len(passing),
                    "support_pass": support,
                    "nominations": nominated, "n_nominated": len(nominated),
                    "bad_declarations": [d for d in bad]})
    return out


def claim_reachability(rows: Optional[List[dict]] = None) -> Dict[str, list]:
    """`commitment -> [(claim spec id, state)]` — the join the 28th audit had
    to compute by hand: `declarations()` × the ledger × the blocker graph.

    States: `PASS`, `RUNNABLE` (every dependency satisfied today), `PARKED`
    (retired by its own decision tree — no path back without a new spec), or
    `blocked<-ROOTS` (a queue position: the terminal blockers its
    unreachability actually rests on). The distinction the states encode is
    the 28th audit's finding: blocked resolves when the blocker does; parked
    resolves never. `run blocked` cannot see the difference and `coverage`
    could not either, so nine of twenty-three commitments sat at zero-passing
    AND zero-runnable with every instrument green.
    """
    from .protocol import Ledger
    from .run import _terminal_blockers
    if rows is None:
        rows = report()
    ledger = Ledger()
    terminal = _terminal_blockers(ledger)
    out: Dict[str, list] = {}
    for r in rows:
        entries = []
        for sid, kind in r["kinds"].items():
            if kind != "claim":
                continue
            res = ledger.results.get(sid)
            status = getattr(getattr(res, "status", None), "name", None)
            if status == "PASS":
                entries.append((sid, "PASS"))
            else:
                roots = terminal.get(sid, set()) - {sid}
                entries.append((sid, "RUNNABLE") if not roots else
                               (sid, "blocked<-" + ",".join(sorted(roots))))
        entries += [(sid, "PARKED") for sid, kind in r["parked"].items()
                    if kind == "claim"]
        out[r["commitment"]] = entries
    return out


def _claim_dead(r: dict) -> bool:
    """No passing claim AND no un-parked claim-kind declaration: nothing this
    commitment promises can currently be falsified by any run. Blocked claims
    do NOT make a commitment claim-dead — blocked is a queue position."""
    return (not r["n_pass"]
            and not any(k == "claim" for k in r["kinds"].values()))


def counts() -> tuple[int, int]:
    """(n_claim_dead, n_malformed) — two different fires, separately
    assertable.

    Claim-dead means no passing claim and zero un-parked claim-kind
    declarations for a constitutional commitment — which includes the original
    zero-declared-specs case, and since the 28th audit also the case where
    every claim spec is PARKED: both are invisible to every other instrument,
    and the reason this module exists. Malformed means a `COVERS:` naming a
    commitment that does not exist, missing its kind, or a `PARKED:` without
    its date — a marker that buys/retires nothing while reading like it does.
    Summing the two fires (the pre-2026-08-14 behaviour) gave them one bell;
    the 17th audit watched the bell ring on a typo and read it as the
    constitutional case.
    """
    rows = report()
    bad = rows[0]["bad_declarations"] if rows else []
    return sum(1 for r in rows if _claim_dead(r)), len(bad)


def check() -> int:
    """Print the audit; exit 2 if any commitment is UNCOVERED or CLAIM-DEAD,
    1 if only malformed declarations exist, 0 clean.

    Uncovered means zero DECLARED specs. Claim-dead means no passing claim and
    no un-parked claim-kind spec — every falsifiable claim it ever had has
    been retired (28th audit: `shelter/building` and `thermal (kills)` both
    went claim-dead in one commit when SH.01 was parked, and this tool exited
    0). "Covered but not passing" is normal — it is a ladder, not a
    scoreboard — so it is reported and not counted. The repair for a red here
    is to REGISTER a successor spec, never to unpark or quiet the tool.
    """
    rows = report()
    bad = rows[0]["bad_declarations"] if rows else []
    reach = claim_reachability(rows)
    parked_notes = parked()[0]
    width = max(len(r["commitment"]) for r in rows)
    uncovered = [r for r in rows if r["n_specs"] == 0 and not r["parked"]]
    dead = [r for r in rows if _claim_dead(r)]
    unproven = [r for r in rows if r["n_specs"] and not r["n_pass"]
                and not _claim_dead(r)]
    print(f"  {'commitment':{width}}  covered (declared)   runnable   nominated")
    n_runnable = {r["commitment"]: sum(1 for _s, st in reach[r["commitment"]]
                                       if st == "RUNNABLE") for r in rows}
    for r in sorted(rows, key=lambda z: (not _claim_dead(z),
                                         z["n_specs"], z["n_pass"])):
        mark = ("NO SPECS" if not r["n_specs"] and not r["parked"] else
                "CLAIM-DEAD (all claim specs parked)" if _claim_dead(r) else
                "none passing" if not r["n_pass"] else "")
        if r["support_pass"]:
            kinds = ", ".join(f"{i} ({k})" for i, k in r["support_pass"].items())
            mark = (mark + f"  [support passing, not credited: {kinds}]").strip()
        print(f"  {r['commitment']:{width}}  {r['n_specs']:>3} specs "
              f"{r['n_pass']:>3} pass   {n_runnable[r['commitment']]:>3} now   "
              f"{r['n_nominated']:>3} nominated   {mark}")
        if _claim_dead(r):
            print(f"  {'':{width}}  ^ {r['why']}")
            for sid, note in sorted(
                    (s, parked_notes.get(s, "")) for s, st in
                    reach[r["commitment"]] if st == "PARKED"):
                print(f"  {'':{width}}    {sid} PARKED {note}")
            if r["nominations"]:
                print(f"  {'':{width}}    nominations (declare or ignore): "
                      f"{', '.join(r['nominations'][:8])}")
        elif not r["n_pass"]:
            # The zero-pass rule is stated over commitments and executed over
            # specs; this line is the join it needs at selection time (28th
            # audit B4): which claim specs could actually move this
            # commitment, and what each is waiting on.
            claims = ", ".join(f"{s} {st}" for s, st in reach[r["commitment"]])
            print(f"  {'':{width}}    claims: {claims}")
    print(f"\n  {len(uncovered)} commitment(s) with NO declared spec, "
          f"{len(dead)} CLAIM-DEAD (no passing claim, every claim spec "
          f"parked),\n  {len(unproven)} with live claim specs but nothing "
          f"passing.")
    if uncovered or dead:
        print("  A commitment with no runnable falsifiable claim is invisible\n"
              "  to `run blocked`, to the overseer, and to every gate. The\n"
              "  repair is to REGISTER a successor spec — parking was the\n"
              "  right call on its evidence; leaving the commitment claim-dead\n"
              "  is the bug, and deleting the PARKED marker would be worse.")
    if bad:
        print(f"  {len(bad)} MALFORMED marker(s) — a typo'd commitment name, "
              f"a missing kind, or a dateless PARKED; none buys anything:")
        for sid, name in bad:
            print(f"      {sid}: {name!r}")
    gc = goal_citations()
    print(f"\n  GOAL.md citations: {len(gc['cited'])} spec ids cited, "
          f"{len(gc['dangling'])} dangling.")
    if gc["new"]:
        print(f"  {len(gc['new'])} NEW dangling citation(s) — the constitution "
              f"names a falsifier nobody registered:\n"
          f"      {', '.join(gc['new'])}\n"
          "  Register the spec (or fix the id in GOAL.md if it is a typo);\n"
          "  never add it to GOAL_DANGLING_BASELINE — that set only shrinks.")
    if gc["known"]:
        print(f"  {len(gc['known'])} known-dangling (seeded 2026-08-25, 29th "
              f"audit; registration debt, B1(a)): {', '.join(gc['known'])}")
    if gc["stale_baseline"]:
        print(f"  {len(gc['stale_baseline'])} baseline entr(y/ies) now RESOLVE "
              f"and must be removed from GOAL_DANGLING_BASELINE: "
              f"{', '.join(gc['stale_baseline'])}")
    print("\n  A nomination is NOT coverage. It is a spec whose title looks\n"
          "  related and whose author has not said so; only `COVERS:` counts.\n"
          "  A PARKED spec is NOT coverage either: a retirement is not a\n"
          "  falsifiable claim, however honest the retiring was.")
    return (2 if (uncovered or dead or gc["new"])
            else (1 if (bad or gc["stale_baseline"]) else 0))


if __name__ == "__main__":
    raise SystemExit(check())
