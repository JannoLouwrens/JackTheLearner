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
    "told world":         (r"\b(told|second.hand|hearsay|anchored|grounded fact|parrot\w*|recit\w*)\b", "owner: the jungle buys MEANING for what he is told"),
    "social/other agents":(r"\b(social|companion|two jacks|second jack)\b", "owner: socialising makes him kind"),
    # `watched`, not `watch\w*`: GEN.02 ("He learns by watching") is Jack
    # watching a TEACHER — a social/other-agents claim, already declared —
    # not the owner spectating him. Verified 2026-09-03: this pattern
    # nominates exactly SO.01 and SO.04, both declared.
    "spectating / being watched": (r"\b(watched|spectat\w*|third.person|stream\w*|observer\w*)\b", "owner: 'I want to watch him figure out the world himself'"),
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

# The citations measured dangling on 2026-08-25 (29th audit; five seeded).
# This set may ONLY shrink: registering one of these makes `goal_citations()`
# demand its removal here, and a NEW dangler is never added — it is a red
# exit. LG.00 registered 2026-08-25 (B1(a)) and removed in the same commit.
# GEN.02/GEN.03/GEN.06/GEN.09 registered 2026-09-01 (Review 08-31 item 6)
# and removed in the same commit — the set is now EMPTY and stays empty:
# any future dangler is `new`, which is a red exit, never a re-seed.
GOAL_DANGLING_BASELINE: frozenset = frozenset()

# The citations measured CITED-BUT-UNRUNNABLE on 2026-09-01 (59th audit B2):
# all three welded behind LC.03 = VOID-FORECLOSED, while GOAL.md:242 describes
# LC.04 as "already testing". Shrink-only, same contract as the dangling
# baseline above: an entry leaves when the citation is revived (upstream
# redesign) or GOAL.md's text stops citing a corpse in the present tense; a
# NEW one is a red exit, never a re-seed. The repair these three await is D10.
GOAL_UNRUNNABLE_BASELINE: frozenset = frozenset({"DP.02", "DP.03", "LC.04"})


def _liveness_state(sids, by_id) -> Dict[str, str]:
    """`{cited id -> 'PARKED' | 'VOID-FORECLOSED' | 'PILOT-BLOCKED' |
    'welded<-ROOTS'}` for every id that RESOLVES but cannot run — the states a
    resolution checker cannot see (59th audit B2). Alive ids are absent."""
    from .protocol import Ledger
    from .run import _terminal_blockers
    ledger = Ledger()
    parked_map, _bad = parked()
    term = _terminal_blockers(ledger)

    def st_name(s):
        res = ledger.results.get(s)
        return getattr(getattr(res, "status", None), "name", None)

    out: Dict[str, str] = {}
    for sid in sids:
        if sid in parked_map:
            out[sid] = "PARKED"
            continue
        fc, _why = foreclosure(sid, status=st_name(sid))
        if fc:
            out[sid] = fc
            continue
        roots = term.get(sid, set()) - {sid}
        if roots and all(root_dead(r, status=st_name(r),
                                   parked_map=parked_map) for r in roots):
            out[sid] = "welded<-" + ",".join(sorted(roots))
    return out


def goal_citations(text: Optional[str] = None,
                   by_id: Optional[dict] = None,
                   baseline: frozenset = GOAL_DANGLING_BASELINE,
                   unrunnable_baseline: frozenset = GOAL_UNRUNNABLE_BASELINE,
                   state_of=None) -> dict:
    """Resolve every spec-shaped id `GOAL.md` cites against the registry —
    for EXISTENCE and for LIVENESS.

    Returns `{"cited", "dangling", "new", "known", "stale_baseline"}` —
    `new` (dangling and NOT in the baseline) is the fatal class: the
    constitution just promised a falsifier nobody registered, the exact hole
    that stood open 16 days. `known` is seeded registration debt (B1(a)).
    `stale_baseline` (baseline entries that now resolve) must be deleted from
    `GOAL_DANGLING_BASELINE` in the same commit that registered them, so the
    baseline only shrinks; leaving one would let the id dangle AGAIN later
    without a red.

    Plus the class resolution cannot see (59th audit B2): `"unrunnable"`
    (`{cited id -> state}` for ids that resolve to a parked, foreclosed or
    welded spec), with `"unrunnable_new"` (red — the constitution cites a
    corpse in the present tense and nobody has said so),
    `"unrunnable_known"` (seeded on `GOAL_UNRUNNABLE_BASELINE`) and
    `"unrunnable_stale_baseline"` (revived — remove from the baseline in the
    same commit). This exists because the checker printed `0 dangling` for
    nine days while GOAL.md:242 said `LC.04` *"is already testing"* of a spec
    welded behind `LC.03` = VOID-FORECLOSED: an id that resolves to a corpse
    is a worse dangling reference than one that resolves to nothing, because
    the nothing-case is the one every checker is built to catch. `state_of`
    is injectable for `_welded_fixture`; the default reads the shared
    predicates (`foreclosure`, `root_dead`, `parked`) so this reader cannot
    drift from `claim_reachability` and `run blocked`.
    """
    if by_id is None:
        from .registry import BY_ID
        by_id = BY_ID
    if text is None:
        text = GOAL_MD.read_text()
    cited = sorted(set(GOAL_CITATION.findall(text)))
    dangling = {i for i in cited if i not in by_id}
    resolved = [i for i in cited if i in by_id]
    if state_of is None:
        unrunnable = _liveness_state(resolved, by_id)
    else:
        unrunnable = {i: s for i in resolved
                      if (s := state_of(i)) is not None}
    return {
        "cited": cited,
        "dangling": sorted(dangling),
        "new": sorted(dangling - baseline),
        "known": sorted(dangling & baseline),
        "stale_baseline": sorted(i for i in baseline if i in by_id),
        "unrunnable": unrunnable,
        "unrunnable_new": sorted(set(unrunnable) - unrunnable_baseline),
        "unrunnable_known": sorted(set(unrunnable) & unrunnable_baseline),
        "unrunnable_stale_baseline": sorted(
            i for i in unrunnable_baseline
            if i in by_id and i not in unrunnable),
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
                    "foreclosed": {
                        i: fk for i, k in pairs if k == "claim"
                        for fk, _w in [foreclosure(
                            i, status=results.get(i, {}).get("status"))]
                        if fk},
                    "bad_declarations": [d for d in bad]})
    return out


def foreclosure(sid: str, status: Optional[str] = None,
                path=None) -> Tuple[Optional[str], Optional[str]]:
    """`("VOID-FORECLOSED", why)` | `("PILOT-BLOCKED", why)` | `(None, None)` —
    THE conjunction for "this spec can never produce evidence", factored into
    one place so `queue_depth` and the CLAIM-DEAD ratchet cannot drift (58th
    audit B1; the `_split_foreclosed` pattern from `404e25a`, taken one step
    further — that helper re-stated the conjunction in `run.py` and documented
    sameness, this one is shared code).

    The two arms are exactly what `queue_depth` computes for its two exclusion
    blocks: a spec is foreclosed when it RAN, VOIDed, and carries an ACCEPTED
    `VOID-FORECLOSED:` declaration (a refused one closes no door), or when it
    is gate-provisional and a run has MEASURED that the pilot's own
    precondition fails (`_PILOT_BLOCKED`, with no contradicting `_PILOT_OWED`
    — the both-case is UNDECLARED, which is queue_depth's routing to decide,
    never a foreclosure). Both repairs are redesigns: neither state can be
    cleared by dispatching anything, which is why each one launders a park if
    an instrument enumerates only `PARKED:`.
    """
    from .protocol import (gates_frozen, module_path_for, pilot_blocked,
                           pilot_owed, void_foreclosed)
    if path is None:
        path = module_path_for(sid)
    if not path:
        return (None, None)
    if status == "VOID":
        why = void_foreclosed(sid, path=path)
        if why:
            return ("VOID-FORECLOSED", why)
    if gates_frozen(sid, path=path) is False:
        why = pilot_blocked(sid, path=path)
        if why and not pilot_owed(sid, path=path):
            return ("PILOT-BLOCKED", why)
    return (None, None)


def root_dead(root: str, status: Optional[str] = None,
              parked_map: Optional[Dict[str, str]] = None) -> Optional[str]:
    """`'PARKED' | 'VOID-FORECLOSED' | 'PILOT-BLOCKED' | None` — can this
    BLOCKER ever resolve? THE shared predicate for the `welded<-ROOTS` state
    (59th audit B1), factored here so `claim_reachability` and `run blocked`
    read the same answer and cannot drift.

    `foreclosure()` above is only ever asked about a spec ITSELF; a retirement
    predicate that is not applied transitively launders itself across exactly
    one dependency edge — `blocked<-LC.03` read as a live queue position for
    nine days while LC.03 was VOID-FORECLOSED and ten specs behind it could
    never run. Blocked resolves when the blocker does; parked and foreclosed
    resolve never, and a root in either state is a closed door whatever the
    dependent's own status says.

    Fails ALIVE: an unknown spec, a missing file, a FAIL, a plain VOID or a
    NOT_RUN all return None — flooding dead through FAIL roots would kill
    every commitment behind T2.01 (the founding blocked-is-alive distinction,
    28th audit). `parked_map` is injectable for callers that already paid for
    `parked()`; `status` is the ledger status name of `root`, if any.
    """
    if parked_map is None:
        parked_map = parked()[0]
    if root in parked_map:
        return "PARKED"
    state, _why = foreclosure(root, status=status)
    return state


# A park's stated release condition, as a DECLARATION inside the PARKED marker
# line: `RELEASE: <spec id>` (repeatable) or `RELEASE: NONE` (this park has no
# release spec — a terminal retirement). Same anti-prose guard as PARKED_MARK.
# The syntax exists because BA.02's marker names five spec ids and only one is
# its release; champions.py already paid for inferring that kind of thing by
# regex (one seat's arena parsed to the words OUT LOUD). Per the 60th-audit
# lesson a declaration syntax converts "badly formatted" into "absent", so the
# parser must count the residue: a parked spec with no RELEASE: at all is
# UNDECLARED-RELEASE, and its prose-named ids are still evaluated (a fallback,
# loudly labelled as one) so an un-migrated row cannot go invisible.
RELEASE_MARK = re.compile(r"(?<!`)RELEASE:\s*([A-Za-z0-9.]+)")

# The park->release pairs measured unreachable when the class first ran
# (62nd audit B3, FINDING 3). Shrink-only, same contract as the other
# baselines: a pair leaves when the release becomes walkable (or the park is
# lifted by a successor/redesign) and must then be removed here; a NEW pair is
# a red exit, never a re-seed. Seeded 2026-09-02 after the class was verified
# RED on the live registry: pre-migration it read 5 pairs (the audit's
# BA.02->LT.08 among them) + 4 UNDECLARED-RELEASE; declaring the four markers'
# own stated releases removed one prose false-positive (T3.10->SM.02, a
# precedent citation) and one successor-of-a-successor (BA.02->BA.03, carried
# by the VOID-FORECLOSED list), leaving these three, all true: a
# constitutional sense parked behind a chain of FAILs priced over free quota,
# and two commitments whose designated successors are themselves
# PILOT-BLOCKED. All three await the w0/redesign desks (09-06, 09-11).
PARK_RELEASE_BASELINE: frozenset = frozenset({
    "BA.02->LT.08", "SH.01->SH.02", "SM.02->SM.03"})


def park_release(by_id: Optional[dict] = None,
                 parked_map: Optional[Dict[str, str]] = None,
                 terminal: Optional[dict] = None,
                 root_status: Optional[Dict[str, str]] = None) -> dict:
    """`{"violations": [(parked sid, release id, state)], "undeclared": [...],
    "none": [...], "declared": {sid: [ids]}}` — is each park's stated release
    condition itself walkable? (62nd audit B3, FINDING 3.)

    THE SCAR: two defaults fired on 2026-09-01 and parked `BA.02` — a
    constitutional sense's only claim — "until the playground-humanoid line",
    which is `LT.08`: blocked by `T2.01`/`T2.02`/the LT chain and priced by
    its own registry note at 43 h per arm-seed against free-compute-only.
    `run blocked` models reachability, `coverage` models claim-liveness, and
    neither joined them to a park's release condition, so a commitment whose
    revival was foreclosed by arithmetic read green on every gate.

    A release is a VIOLATION when it is DANGLING (declared but resolving to
    no registered spec), PARKED, foreclosed (`foreclosure()` — the shared
    conjunction), `welded<-` (every terminal blocker dead — `root_dead()`),
    or `blocked<-` (live roots). Blocked firing here is DELIBERATE and
    different from `claim_reachability`, where blocked-is-alive is founding:
    a claim behind a live FAIL is a queue position, but a PARK behind one is
    a revival path that cannot be walked today, and that composition — legal
    park, unreachable release — is exactly the fact no instrument could
    utter. A `PASS` or runnable-today release is quiet.

    All inputs injectable for `_park_release_fixture`, same idiom as
    `claim_reachability`.
    """
    from .protocol import Ledger
    from .run import _terminal_blockers
    if by_id is None:
        from .registry import BY_ID
        by_id = BY_ID
    if parked_map is None:
        parked_map = parked(by_id)[0]
    ledger = Ledger()
    if terminal is None:
        terminal = _terminal_blockers(ledger)

    def _status_of(sid: str) -> Optional[str]:
        if root_status is not None and sid in root_status:
            return root_status[sid]
        res = ledger.results.get(sid)
        return getattr(getattr(res, "status", None), "name", None)

    out: dict = {"violations": [], "undeclared": [], "none": [],
                 "declared": {}}
    for sid, reason in sorted(parked_map.items()):
        declared = [t.rstrip(".") for t in RELEASE_MARK.findall(reason)]
        if declared:
            out["declared"][sid] = declared
            targets = []
            for t in declared:
                if t == "NONE":
                    continue
                if t not in by_id:
                    # A declaration is an assertion; an unresolvable one is
                    # the loud failure direction, never silently dropped.
                    out["violations"].append((sid, t, "DANGLING"))
                else:
                    targets.append(t)
            if declared == ["NONE"]:
                out["none"].append(sid)
        else:
            # Un-migrated row: no declaration. Counted as residue AND its
            # prose-named ids still evaluated, so the class cannot be quieted
            # by simply never migrating (60th-audit lesson). Only resolvable
            # ids enter — the residue count is the loud half for the rest.
            out["undeclared"].append(sid)
            targets = [t for t in dict.fromkeys(GOAL_CITATION.findall(reason))
                       if t != sid and t in by_id]
        for rid in targets:
            st = _status_of(rid)
            if st == "PASS":
                continue
            state: Optional[str] = ("PARKED" if rid in parked_map
                                    else foreclosure(rid, status=st)[0])
            if state is None:
                roots = terminal.get(rid, set()) - {rid}
                if roots:
                    if all(root_dead(r, status=_status_of(r),
                                     parked_map=parked_map) for r in roots):
                        state = "welded<-" + ",".join(sorted(roots))
                    else:
                        state = "blocked<-" + ",".join(sorted(roots))
            if state:
                out["violations"].append((sid, rid, state))
    return out


def _park_release_fixture() -> List[str]:
    """Known-answer battery for `park_release` (62nd audit B3).

    Planted shapes: the audit's exact case (a declared release behind a LIVE
    FAIL root must fire `blocked<-` — the direction that stays ALIVE in
    `claim_reachability` and deliberately fires HERE), `RELEASE: NONE` quiet,
    a PASS release quiet, a runnable release quiet, a PARKED release firing,
    a PILOT-BLOCKED release firing through the shared `foreclosure()`, a
    DANGLING declaration firing, a welded release (all roots dead), and the
    residue class: an undeclared marker is counted AND its prose ids still
    evaluated, while an undeclared marker naming nothing yields residue only.
    Stubbing idiom of `_welded_fixture`: protocol readers monkeypatched,
    graph and statuses injected, so the CLAUSE is under test.
    """
    by_id = {q: object() for q in
             ("Q.60", "Q.61", "Q.62", "Q.63", "Q.64", "Q.65")}
    parked_map = {
        # The audit's case: declared release blocked by a live FAIL root.
        "Q.50": "2026-09-01 — parked until the body line. RELEASE: Q.60",
        "Q.51": "2026-09-01 — terminal retirement. RELEASE: NONE",
        "Q.52": "2026-09-01 — parked behind a done thing. RELEASE: Q.61",
        "Q.53": "2026-09-01 — parked behind a park. RELEASE: Q.62",
        "Q.62": "2026-08-01 — spent fork, names no release",
        "Q.54": "2026-09-01 — parked. RELEASE: Q.99",
        # Un-migrated prose row: names Q.63 with no declaration.
        "Q.55": "2026-09-01 — re-parented behind Q.63 (the humanoid line)",
        "Q.56": "2026-09-01 — parked. RELEASE: Q.64",
        "Q.57": "2026-09-01 — parked. RELEASE: Q.65",
    }
    terminal = {"Q.60": {"Q.70"}, "Q.64": {"Q.62"}}
    root_status = {"Q.60": "FAIL", "Q.61": "PASS", "Q.62": None,
                   "Q.63": None, "Q.64": None, "Q.65": None, "Q.70": "FAIL"}

    from . import protocol as _proto
    real_vf, real_gf = _proto.void_foreclosed, _proto.gates_frozen
    real_pb, real_po = _proto.pilot_blocked, _proto.pilot_owed
    real_mpf = _proto.module_path_for
    _proto.void_foreclosed = lambda sid, path=None: None
    _proto.gates_frozen = lambda sid, path=None: False
    _proto.pilot_blocked = (lambda sid, path=None:
                            "measured: precondition fails"
                            if sid == "Q.63" else None)
    _proto.pilot_owed = lambda sid, path=None: None
    _proto.module_path_for = lambda sid, strict=False: f"/x/{sid}.py"
    try:
        pr = park_release(by_id=by_id, parked_map=parked_map,
                          terminal=terminal, root_status=root_status)
    finally:
        _proto.void_foreclosed, _proto.gates_frozen = real_vf, real_gf
        _proto.pilot_blocked, _proto.pilot_owed = real_pb, real_po
        _proto.module_path_for = real_mpf

    got = {(s, r): st for s, r, st in pr["violations"]}
    fails = []
    if got.get(("Q.50", "Q.60")) != "blocked<-Q.70":
        fails.append(f"park_release: a park behind a live-blocked release "
                     f"must fire blocked<- (BA.02<-LT.08, the audit's case) — "
                     f"got {got.get(('Q.50', 'Q.60'))!r}")
    if got.get(("Q.53", "Q.62")) != "PARKED":
        fails.append("park_release: a park released by a park must fire")
    if got.get(("Q.54", "Q.99")) != "DANGLING":
        fails.append("park_release: a declared release resolving to no spec "
                     "must fire DANGLING, never be silently dropped")
    if got.get(("Q.55", "Q.63")) != "PILOT-BLOCKED":
        fails.append("park_release: an UNDECLARED row's prose-named release "
                     "must still be evaluated (60th-audit lesson) and reach "
                     "the shared foreclosure() — got "
                     f"{got.get(('Q.55', 'Q.63'))!r}")
    if got.get(("Q.56", "Q.64")) != "welded<-Q.62":
        fails.append("park_release: a release whose every root is dead must "
                     f"fire welded<- — got {got.get(('Q.56', 'Q.64'))!r}")
    for quiet in ("Q.51", "Q.52", "Q.57"):
        if any(s == quiet for s, _r, _st in pr["violations"]):
            fails.append(f"park_release: {quiet} (NONE / PASS-release / "
                         f"runnable-release) must be quiet")
    if sorted(pr["undeclared"]) != ["Q.55", "Q.62"]:
        fails.append(f"park_release: the UNDECLARED-RELEASE residue must "
                     f"count exactly the markers with no RELEASE: — got "
                     f"{sorted(pr['undeclared'])}")
    if pr["none"] != ["Q.51"]:
        fails.append("park_release: RELEASE: NONE must be recorded, not "
                     "counted as undeclared")
    return fails


def claim_reachability(rows: Optional[List[dict]] = None,
                       terminal: Optional[dict] = None,
                       root_status: Optional[Dict[str, str]] = None
                       ) -> Dict[str, list]:
    """`commitment -> [(claim spec id, state)]` — the join the 28th audit had
    to compute by hand: `declarations()` × the ledger × the blocker graph.

    States: `PASS`, `RUNNABLE` (every dependency satisfied today), `PARKED`
    (retired by its own decision tree — no path back without a new spec),
    `FORECLOSED` (declared `VOID-FORECLOSED`, or gate-provisional with a
    measured pilot-block — `foreclosure()`, the same conjunction
    `queue_depth` excludes on, so the two readers cannot drift; 58th audit
    B1), `blocked<-ROOTS` (a queue position: the terminal blockers its
    unreachability actually rests on, at least one of which is LIVE), or
    `welded<-ROOTS` (59th audit B1: every terminal blocker is itself parked
    or foreclosed — `root_dead()` — so no dispatch anywhere can free it; the
    repair is an upstream redesign). The distinction the states encode is
    the 28th audit's finding: blocked resolves when the blocker does; parked
    resolves never. `run blocked` cannot see the difference and `coverage`
    could not either, so nine of twenty-three commitments sat at zero-passing
    AND zero-runnable with every instrument green. FORECLOSED is the same
    finding one invention later: before it existed, a foreclosed spec fell
    through to `RUNNABLE` — the strongest state short of `PASS` — and the
    commitment table contradicted the queue-depth section forty lines below
    it in the same report. WELDED is the same finding once more, transitively:
    a foreclosed ROOT laundered ten dependents back into `blocked<-`, the
    live-queue reading, for nine days. `terminal`/`root_status` are
    injectable for `_welded_fixture`.
    """
    from .protocol import Ledger
    from .run import _terminal_blockers
    if rows is None:
        rows = report()
    ledger = Ledger()
    if terminal is None:
        terminal = _terminal_blockers(ledger)
    parked_map, _bad = parked()

    def _status_of(root: str) -> Optional[str]:
        if root_status is not None and root in root_status:
            return root_status[root]
        res = ledger.results.get(root)
        return getattr(getattr(res, "status", None), "name", None)

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
            elif sid in r.get("foreclosed", {}):
                entries.append((sid, "FORECLOSED"))
            else:
                roots = terminal.get(sid, set()) - {sid}
                if not roots:
                    entries.append((sid, "RUNNABLE"))
                elif all(root_dead(rt, status=_status_of(rt),
                                   parked_map=parked_map) for rt in roots):
                    entries.append(
                        (sid, "welded<-" + ",".join(sorted(roots))))
                else:
                    entries.append(
                        (sid, "blocked<-" + ",".join(sorted(roots))))
        entries += [(sid, "PARKED") for sid, kind in r["parked"].items()
                    if kind == "claim"]
        out[r["commitment"]] = entries
    return out


def _claim_dead(r: dict) -> bool:
    """No passing claim AND every claim-kind declaration is PARKED or
    FORECLOSED: nothing this commitment promises can currently be falsified by
    any run. Blocked claims do NOT make a commitment claim-dead — blocked is a
    queue position that resolves when the blocker does; parked and foreclosed
    resolve never (58th audit B1: `VOID-FORECLOSED` and `PILOT-BLOCKED` are
    retirements that do not spell `PARKED:`, and for two days each one
    laundered a park past the 28th audit's repair — five commitments
    claim-dead in fact, `0 CLAIM-DEAD` reported, rc=0).

    `kinds` has already had PARKED specs removed, so the conjunction here is:
    no PASS, and every surviving claim is in the row's `foreclosed` map —
    which `all()` over an empty list makes subsume the original two cases
    (zero declared claims; every claim parked)."""
    return (not r["n_pass"]
            and all(sid in r.get("foreclosed", {})
                    for sid, k in r["kinds"].items() if k == "claim"))


# ── QUEUE DEPTH — is there anything to SPEND the free quota on? ─────────
#
# WHY THIS EXISTS, and it is a 61-hour scar. Across three consecutive Kaggle
# weeks 8.94 + 22.37 + 29.69 = 61.0 free GPU-hours expired unspent, and four
# documents blamed the loop being dark on the Sunday. `2026-W34` falsified
# that on its own: the builder ran 23 unblocked iterations INSIDE its own GPU
# week, with the full 30 hours available, and dispatched 0.31 of them. Jobs
# completed per week ran 17 -> 23 -> 1. Availability was not the binding
# constraint. INVENTORY was — the shelf of dispatchable specs had been empty
# since 08-25 04:40, 8.4 hours BEFORE the blackout even began.
#
# And no instrument in this repository could say so. `run next` lists specs
# whose DEPENDENCIES pass, which is a different question: 17 of its GPU-cost
# rows were unimplemented, settled, parked or untracked, and it printed them
# all identically. `run blocked` measures what unsticks the ladder. `coverage`
# (above) measures whether the ladder is the right ladder. Nobody measured
# whether the ladder had anything RUNNABLE TODAY, which is the only question a
# perishable weekly quota actually asks.
#
# Same shape as this module's founding scar, one layer up: a missing spec has
# no id and is invisible to every instrument, and so is an empty queue.

# The cost classes MEASURED empty on 2026-08-29, by running this function —
# not inferred from any page's prose. (The first draft of this line seeded
# {gpu<20min, gpu<2h, gpu<8h} from the Review's summary and was wrong in both
# directions: `gpu<20min` holds SM.03 and `gpu<8h` holds T2.02, while the two
# cheap CPU classes were empty and unmentioned. LESSONS: a quantity you can
# read out of the source is not a quantity to estimate.)
#
# Like GOAL_DANGLING_BASELINE this set may ONLY SHRINK: a class that becomes
# non-empty must be deleted from here in the same commit (`stale_baseline`
# demands it), and a class that goes empty and is NOT listed here is a RED,
# because it is new debt. The repair is always to implement a spec — never to
# add a class to this set.
#
# What the red is FOR: `gpu<20min` currently rests on SM.03 alone. When SM.03
# settles, this file exits 2 and says so, which is the standing duty the 45th
# audit and the 08-29 Review both asked for — "an iteration that finds GPU
# queue depth at zero implements a GPU spec before it does anything else" —
# made mechanical instead of written down in one organ's prompt.
# `gpu<2h` LEFT THIS SET 2026-08-30, which is the ratchet doing its job rather
# than a tidy-up: T2.14 was implemented on 08-29 and dispatched to Kaggle on
# 08-30, so the class is no longer empty and `stale_baseline` demanded the
# deletion in the same commit. It may never come back. Note what that costs on
# purpose — when T2.14 settles, `gpu<2h` goes empty again and this file will
# exit 2 rather than quietly re-baseline. That red is the point: the class that
# forfeited 61 free GPU-hours over three weeks is now one that cannot go empty
# in silence.
# `cpu<10min` LEFT THIS SET 2026-09-02, by the same precedent applied
# honestly against a 81-second window: ME.11 was implemented (2e12d1f),
# making the class non-empty and this entry stale, and settled FAIL the same
# hour, emptying it again. `stale_baseline` never fired only because nobody
# ran coverage inside those 81 seconds; leaving the entry would be exactly
# the quiet re-baseline the gpu<2h note forbids. From now on cpu<10min going
# empty reads amber/red like any other class — its refill is the ME.11
# family redesign on the Review's 09-06 desk, not a spec hunt (see
# empty_unfillable's own instruction above).
# `cpu<1min` LEFT THIS SET 2026-09-04 — the cpu<10min shape again, and the
# set is now EMPTY. SO.06's PASS released SO.09 (CPU_FAST) into the class,
# SO.09 was implemented (0476ff9) and PASSED the same hour (attempt 1, 19 s:
# the provisioning accountant refuses C-GIVE, unlogged drops and direct-e
# restores, accepts the clean hand), and the class emptied again. Leaving the
# entry would be the quiet re-baseline the gpu<2h note forbids. From now on
# cpu<1min going empty reads amber/red like any other class — its refills are
# releases (the SO/LG families both feed it), never a spec hunt.
QUEUE_EMPTY_BASELINE = frozenset()

# The unreachable-fraction ratchet (58th audit B3). `run blocked` has printed
# "N of M specs are unreachable" since 08-09 and NO GATE READ IT: the number
# drifted 80/211 (38%, 55th audit, 2026-08-31) -> 85/217 (39%, 58th audit,
# 2026-09-01) and the only reason anyone knew is that two overseers happened
# to print it. A foreclosure that welds a new subtree lands silently.
#
# Same contract as QUEUE_EMPTY_BASELINE, pointed at a count instead of a set:
# SHRINK-ONLY. When the live count falls below this number, the baseline must
# be lowered in the SAME commit (`stale_unreachable_baseline`, amber). When it
# rises above, `check()` goes RED (`unreachable_grew`) and stays red until the
# commit that grew it raises this constant WITH ITS JUSTIFICATION NAMED in the
# comment below and in the commit message — registering a deliberately-blocked
# spec (the GEN.02-09 shape, structural depends_on) is a legitimate reason; a
# reason nobody wrote down is not. The repair for growth without a story is an
# UNBLOCK, never a quiet re-baseline.
#
# Growth log (append a line per raise, newest first; SHRINKS are logged too,
# because a floor that follows the number down silently is a floor nobody can
# audit — see the 2026-09-03 entry, which exists because it did NOT follow):
# NOTE (71st audit B1, 2026-09-04): entries were appended oldest-LAST until
# this commit and are now stored newest-first as the header instructs. Lines
# were moved and nothing was renumbered, reworded or re-justified — so where
# an entry says "the N-entry above" it was written under the old order and
# means the EARLIER entry, which now sits BELOW it.
#   95 @ 2026-09-06 (GROWTH, Review FULL Part 2) — ME.1 was STRENGTHENED and
#     the strengthening FAILED it, which blocks ME.3, ME.5, ME.9 and ME.10
#     behind it and moves the live count 94 -> 95. The justification the
#     ratchet asks for, in full: ME.1's abstention control used cues whose
#     every content word was ABSENT from the store, so a keyword filter
#     passed it and it read a perfect 1.0000 for 29 days. On 2026-09-02
#     ME.11 SETTLED FAIL measuring the HARD version of that same question on
#     this project's retrieval stack — gold masked, topically-similar
#     remainder retained — at abstention 0.877 against 0.95. ME.1 now carries
#     that control (60 events held out of the store, cued for against the 940
#     remaining) at its OWN unchanged 0.95 bar, and measures
#     distractor_abstention 0.0000 +- 0.0 on three seeds against
#     fabricated_abstention 1.0: the store confabulates the nearest neighbour
#     on EVERY absent-target cue. No threshold moved in either direction and
#     cued_recall is unchanged at 0.85. This growth is a measurement becoming
#     honest, not a regression — the four specs behind ME.1 were always
#     resting on a floor calibrated against a control it could not fail.
#     Routed for repair as `me1-similarity-floor-never-abstains` (DUE
#     2026-09-13), so FAIL-UNOWNED returns to its floor of 0.
#   94 @ 2026-09-04 (SHRINK, builder) — SO.06 PASSED (attempt 2, clean tree
#     3ad646e) and released SO.07, SO.08 and SO.09 in one step, so the live
#     count fell 97 -> 94 within the hour the 97 was written. The floor follows
#     the number down in the SAME commit, shrink-only: the 09-03 entry below
#     exists precisely because a floor once did NOT follow, and carried a
#     ceiling one above the truth that would have accepted a silent regression
#     as clean. This is the entry above predicting itself — "it clears the
#     first time SO.06 passes" — and it is logged rather than left implicit,
#     because a floor that moves without a line is a floor nobody can audit.
#   97 @ 2026-09-04 (builder) — the owner's-hands family registered from
#     docs/research/OWNERS_HANDS.md §6 under the INTEGRATION_QUEUE protocol:
#     SO.06 (the provisioning-channel FIXTURE) is RUNNABLE and is the only
#     FRESH dispatch on a board coverage reported empty at every cost class;
#     the other three — SO.07, SO.08, SO.09 — are deliberately blocked behind
#     it, because scoring a provisioning arm against a channel never certified
#     to reach ONLY through the world is the exact defect SO.06 exists to
#     prevent (SO.06's own bit-identity leg: hand present and dropping nothing
#     must be indistinguishable from no hand). The block is `depends_on`, so
#     protocol.blocked_by() enforces it rather than a docstring; it clears the
#     first time SO.06 passes. +3, the GEN.02-09 / SO.04 / LG.03 shape, named
#     here as the ratchet requires. Note what this does NOT buy: SO.09 is
#     `cpu<1min` and blocked, so that class stays no-path-in and stays in
#     QUEUE_EMPTY_BASELINE — registering did not clear it and this entry does
#     not claim it did.
#   94 @ 2026-09-04 (builder) — the LG grounding bakeoff registered from
#     docs/research/LANGUAGE_GROUNDING.md §7 under the INTEGRATION_QUEUE
#     protocol: LG.03 (the cell-certification FIXTURE) is RUNNABLE and refills
#     the cpu<10min class, which coverage reported NEWLY EMPTY and FILL-HELD
#     behind D19; the other three — LG.04, LG.05, LG.06 — are deliberately
#     blocked behind LG.03 because scoring a grounding arm on cells that were
#     never certified language-necessary is the exact defect this family
#     exists to prevent (CAST 2508.13446: the action distribution collapses
#     given the observation alone, so the arm need never read the command).
#     The block is `depends_on`, so protocol.blocked_by() enforces it rather
#     than a docstring; it clears the first time LG.03 passes. +3, the
#     GEN.02-09 / SO.04 shape, named here as the ratchet requires.
#   91 @ 2026-09-03 (builder) — SO.01 PASSED (attempt 2, clean-tree 962c3b9:
#     11.0 fps delivered / rtf 2.2 with the stream), so SO.04's deliberate
#     block behind it cleared exactly as the 91-entry above predicted — "the
#     block clears the first time SO.01 passes". The spectating claim is now
#     RUNNABLE; the floor follows the number down, shrink-only.
#   92 @ 2026-09-03 (builder, 66th audit B1) — LG.11 (THE TOLD WORLD)
#     registered deliberately UNREACHABLE behind LG.00 + LF.01: GOAL.md's
#     third expansion declared itself falsifiable on 2026-08-09 and had no
#     spec for 25 days, invisible because its claim fell in `language
#     (parent)` which reads 3 pass. A truthful red is the deliverable (the
#     GEN.02-09 shape); the LF.01 dep re-parents to the W1 line when the
#     Sunday 09-06 design registers it.
#   91 @ 2026-09-03 (builder) — DIRECTION_AUDIT.md's queue row processed:
#     seven specs registered (LF.01/LF.02, SO.01/SO.02/SO.04, T0.32/T0.33),
#     of which SIX are runnable today — LF.02, SO.01, SO.02, T0.32, T0.33
#     refill the empty cpu<10min class and LF.01 refills cpu<2h — and ONE,
#     SO.04 ("Being watched does not change him"), is deliberately blocked
#     behind SO.01 because an observer-invariance test without a stream to
#     invert is unfalsifiable. Its dep is a fresh RUNNABLE spec, so the block
#     clears the first time SO.01 passes; registering the falsifier beside
#     the capability is the GEN.02-09 shape, named here as the ratchet
#     requires.
#   90 @ 2026-09-03 (65th audit B1) — HR.5 added to HR.6.depends_on. HR.5's
#     registry notes call it "PREREQUISITE FOR HR.6 BEING INFORMATIVE" and it
#     FAILed at 05:25 the same morning (classes_present 1.0 of 4, no kind
#     label, no self flag); the edge was written for HR.8 in the registering
#     commit and forgotten for HR.6 (LESSONS.md, "A prerequisite that lost is
#     worse than a producer that is missing"). HR.6 leaves the runnable set
#     until the fixture repair lands — a truthful red, deliberately bought:
#     the staging valve fires on A2-vs-A0b, but HR.5 predicts A5-ties-
#     everything, which passes that valve and green-lights 3-6 GPU-hours on
#     a question HR.6's own notes call not well-posed.
#   89 @ 2026-09-03 (SHRINK, Review DAILY) — HR.7's PASS (e7badf4) reopened its
#     downstream and the live count fell 90 -> 89. The harvest commit b8f69f4
#     wrote the new reading into ratchet_readings.json and its message says
#     "record unreachable 90 -> 89", but this constant was left at 90, so the
#     ratchet was carrying a floor one above the truth and would have accepted
#     a silent regression back to 90 as clean. The reading file is a log; THIS
#     is the floor. Lowered by the Review under strengthen-only.
#   90 @ 2026-09-03 — HR.1-HR.8 registered from HEARING_BAKEOFF.md (the
#     INTEGRATION_QUEUE's PENDING entry, 5-step protocol followed). Five of
#     the eight (HR.2, HR.3, HR.4, HR.6, HR.8) are structurally blocked
#     behind unimplemented parents in the same family — the GEN.02-09 shape,
#     deliberate depends_on, named here as the ratchet requires. The other
#     three (HR.1, HR.5, HR.7) are RUNNABLE and refill the empty cpu<10min
#     class, which is the point of the registration.
#   85 @ 2026-09-01 — seeded from the 58th audit's own measurement (B3).
UNREACHABLE_BASELINE = 95


def unreachable_ratchet(ledger=None,
                        baseline: int = UNREACHABLE_BASELINE,
                        count_fn=None) -> dict:
    """Compare the live unreachable count against the shrink-only baseline.

    The count comes from `run.unreachable_count` — the SAME union of the SAME
    dependency walk `run blocked` prints, factored so the two readers cannot
    drift (the `_split_foreclosed` pattern). The ranker's own known-answer
    fixture runs first; a ranker that fails it produces a `refused` entry
    instead of a number, because a count from an instrument that flunks its
    fixture is not evidence (LESSONS.md, the at-chance-control rule one level
    up).

    Returns `{"count", "ladder", "baseline", "grown", "stale_baseline",
    "refused"}` — the last three are lists of message strings, empty when
    healthy, truthy for `exit_code` when not.
    """
    out = {"count": None, "ladder": None, "baseline": baseline,
           "grown": [], "stale_baseline": [], "refused": []}
    if count_fn is None:
        def count_fn():
            from .protocol import Ledger
            from .run import _check_ranker, unreachable_count
            led = Ledger() if ledger is None else ledger
            _check_ranker(led)
            return unreachable_count(led)
    try:
        count, ladder = count_fn()
    except RuntimeError as exc:
        out["refused"].append(f"unreachable ratchet: the blocked-ranker "
                              f"failed its own fixture, no count is evidence "
                              f"({exc})")
        return out
    out["count"], out["ladder"] = count, ladder
    if count > baseline:
        out["grown"].append(
            f"unreachable specs GREW: {count} of {ladder} vs baseline "
            f"{baseline}. Growth is permitted only with a named justification "
            f"in the commit that grows it — raise UNREACHABLE_BASELINE there, "
            f"append to its growth log, and say WHY (a deliberately-blocked "
            f"registration is a reason; silence is not). Otherwise the repair "
            f"is an UNBLOCK (`run blocked`).")
    elif count < baseline:
        out["stale_baseline"].append(
            f"unreachable count fell to {count} of {ladder}; "
            f"UNREACHABLE_BASELINE still reads {baseline} and must be "
            f"lowered in the same commit — the ratchet only ratchets if the "
            f"floor follows the number down.")
    return out


def _unreachable_fixture() -> List[str]:
    """Known-answer battery for `unreachable_ratchet` (58th audit B3).

    Four classifications on injected counts — grown, stale, clean, refused —
    each through the REAL function, plus the real counting path run against
    the blocked-ranker's own fixture graph, whose answer is known by hand:
    Y, Z (behind X/W), V (behind the stale PASS), G, Q, N (behind the three
    VOIDs) = 6 unreachable of 12. That half exercises `run.unreachable_count`
    itself, so a drift between the walk and the union fails here by name.
    """
    fails = []
    r = unreachable_ratchet(count_fn=lambda: (100, 217), baseline=85)
    if not r["grown"] or r["stale_baseline"] or r["refused"]:
        fails.append("unreachable: count above baseline must read GROWN, "
                     "alone — a silent weld is the class this exists for")
    r = unreachable_ratchet(count_fn=lambda: (80, 217), baseline=85)
    if not r["stale_baseline"] or r["grown"] or r["refused"]:
        fails.append("unreachable: count below baseline must demand the "
                     "shrink — a floor that does not follow the number down "
                     "is not a ratchet")
    r = unreachable_ratchet(count_fn=lambda: (85, 217), baseline=85)
    if r["grown"] or r["stale_baseline"] or r["refused"]:
        fails.append("unreachable: count at baseline is the healthy state — "
                     "it must be recognisable or the sick ones mean nothing")

    def _broken():
        raise RuntimeError("planted ranker failure")
    r = unreachable_ratchet(count_fn=_broken, baseline=85)
    if not r["refused"] or r["grown"] or r["stale_baseline"] or \
            r["count"] is not None:
        fails.append("unreachable: a ranker that fails its fixture must "
                     "REFUSE the count, not classify it")

    from .run import _fixture_ledger, _ranker_fixture, unreachable_count
    ladder, by_id = _ranker_fixture()
    got = unreachable_count(_fixture_ledger(), ladder=ladder, by_id=by_id)
    # `(6, 15)`: SIX unreachable (Y, Z, V, G, Q, N) of FIFTEEN in the fixture
    # graph. The 6 is the invariant this check exists for and it has NEVER
    # moved; the 15 was 12 until 2026-09-04, when the 69th audit's B3 added
    # three stubs (P, B, C) to `_ranker_fixture` for the repair-edge layer.
    # That edit is exactly the drift this line is built to catch — it fired
    # the same minute, correctly — and the honest response to a known-answer
    # check firing on a KNOWN cause is to re-derive the answer from the new
    # graph and say why here, never to relax the equality to a `>=` or drop
    # the ladder-size half. If the 6 ever moves, the walk really has drifted.
    if got != (6, 15):
        fails.append(f"unreachable: the real counting path read {got} on the "
                     f"ranker fixture graph; the known answer is (6, 15) — "
                     f"the walk and the union have drifted")
    return fails


# ── FAIL-UNOWNED: a settled negative with nobody assigned (72nd audit B1) ──
#
# The scar: `XL.01` — "death does not erase what he learned" — sat FAIL for 17
# days while every instrument printed health, because every reader here is
# keyed to a spec's REACHABILITY (can it be dispatched?) and none to its
# DISPOSITION (who repairs it?). `T2.05` and `T4.02` sat 16 and 15 days on the
# same silence. A settled FAIL leaves the dispatch queue, is not parked, blocks
# little, and was never routed — five readers, five correct behaviours, one
# hole (docs/LESSONS.md, "A negative with no owner is invisible").
#
# GROWTH LOG (shrink-only; every raise needs an entry here and a reason in the
# commit that makes it):
#   2026-09-05  baseline 4: T2.05, T2.15, T4.02, XL.01 (builder, executing
#               72nd audit B1). The audit's own live reading was 3 — it missed
#               T2.15, whose FAIL (2026-08-25, the language memorisation-route
#               measurement) is cited only in SM.03's registry notes; by the
#               same audit's own lesson, a successor's notes are not routing
#               when the successor is PILOT-BLOCKED. T3.07 is excluded by its
#               FAIL-DISPOSED marker (D7's fired default), not by id.
#   2026-09-05  4 -> 0, same slot: all four ROUTED per B4 (REVIEW_QUEUE rows
#               xl01-*, t205-*, t402-*, t215-*, each with DUE 2026-09-13 =
#               next_free_due). Zero is the working state now: every future
#               settled FAIL must be routed or disposed in the same motion
#               that records it, or coverage goes red.
FAIL_UNOWNED_BASELINE = 0

# The readable disposition marker (72nd audit B1: the T3.07 exclusion "must be
# by a readable marker, not by a hardcoded id"). Lives in `Spec.notes` —
# deliberately un-hashed, so writing one costs no certificate re-buys — and
# must name its AUTHORITY (the decision or organ that killed the question) and
# a DATE, or it is reported MALFORMED and excludes nothing: an unparseable
# disposition silently excluding a FAIL is the dateless-PARKED leak, one
# marker over.
FAIL_DISPOSED_RX = re.compile(
    r"FAIL-DISPOSED:\s*(?P<authority>[A-Za-z0-9._-]+)\s+"
    r"(?P<date>\d{4}-\d{2}-\d{2})")


def _owned_by_dued_row(sid: str, queue_doc: str) -> Optional[str]:
    """The strongest clocked-row ownership FORM for `sid`, or None.

    Returns `"queue-row"` when a row about `sid` carries a `DUE:` (a dated
    promise), `"held-on-blocker"` when its only clock is a `BLOCKED-BY:` (a
    named release condition — legal payment per `REVIEW_QUEUE.md`'s own
    contract and `review_queue.py`'s HOLD-WITHOUT-A-CLOCK, but weaker than a
    date: it promises order, not time), and None when `sid` appears in no
    clocked row at all — a bare mention, printed as one.

    A row is "about" `sid` two ways, and both count (74th audit B2):
      - the id appears in the row's `ROUTED:` line or indented body — the
        declaration test is `review_queue._DECL`, not a fresh regex, so the
        two readers cannot drift apart on what a declared clock looks like;
      - the row's own SLUG names it (`t205-…` ↔ `T2.05`): the leading token
        of the row id, matched whole against the spec id lowercased with
        dots dropped. This is how a row actually declares its subject, and
        it is exact — an id cited in another row's flush-left evidence
        paragraph does NOT inherit that row's clock (the `xl01` body names
        `NE.01` and `NE.08`; neither is thereby owned). The whole-token rule
        is the boundary conversion at slug level: `z1` must not be laundered
        by `z10-held`.

    Block boundaries are `review_queue.parse`'s published contract — a
    column-0 `ROUTED:` opens a row, a non-indented line ends its body.
    """
    from . import review_queue as rq
    rx = re.compile(r"(?<![A-Za-z0-9.])" + re.escape(sid) + r"(?!\.?\w)")
    slug = sid.lower().replace(".", "")

    def _forms(block: List[str]) -> set:
        head = rq._ROUTED.match(block[0])
        row_id = head.group(1).split("|")[0].strip() if head else ""
        subject = row_id == slug or row_id.startswith(slug + "-")
        if not subject and not any(rx.search(l) for l in block):
            return set()
        found = set()
        for l in block[1:]:
            d = rq._DECL.match(l.strip())
            if d:
                found.add("queue-row" if d.group(1) == "DUE"
                          else "held-on-blocker")
        return found

    forms: set = set()
    block: List[str] = []
    for raw in queue_doc.splitlines():
        if rq._ROUTED.match(raw):
            if block:
                forms |= _forms(block)
            block = [raw]
            continue
        if not block:
            continue
        if raw.strip() and not raw[:1].isspace():
            forms |= _forms(block)
            block = []
        else:
            block.append(raw)
    if block:
        forms |= _forms(block)
    if "queue-row" in forms:
        return "queue-row"
    if "held-on-blocker" in forms:
        return "held-on-blocker"
    return None


def fail_unowned(by_id: Optional[dict] = None,
                 results: Optional[dict] = None,
                 queue_doc: Optional[str] = None) -> dict:
    """Settled FAILs with NO repair owner — the state no other reader names.

    A spec is FAIL-UNOWNED when ALL of: its ledger row is `FAIL`; its
    `Spec.repaired_by` is empty; its id appears nowhere in
    `docs/REVIEW_QUEUE.md`; and its notes carry no well-formed
    `FAIL-DISPOSED:` marker.

    The ONLY legal repairs, stated here because the tempting ones are the
    banned ones:
      (a) route a `docs/REVIEW_QUEUE.md` row with a `DUE:`,
      (b) declare `repaired_by` on the failed spec,
      (c) an explicit `FAIL-DISPOSED: <authority> <YYYY-MM-DD>` registry
          disposition naming the decision that killed the question (T3.07/D7
          is the model — answered, not orphaned).
    NEVER by deleting the FAIL row, NEVER by re-running for a better number,
    and NEVER by adding to `FAIL_UNOWNED_BASELINE` — that constant only
    shrinks.

    Ownership-by-mention is deliberately generous: any appearance of the id in
    the queue doc counts, header or body, because row bodies cite their
    instrument specs by id (DP.05's repair genuinely rides the
    `w0-too-shallow` row). The generosity is bounded by what the class is for:
    a FAIL about which nobody has written a single queue line still fires,
    which is exactly the XL.01 hole. A prose mention is a weak owner — the
    strong forms are (a)-(c) above — but weak ownership is a routing-quality
    question for the Review, not a hole this counter can see.

    Returns `{"unowned": [ids], "disposed": {id: (authority, date)},
    "malformed": [ids], "owned": {id: form}, "count": int}`. A malformed
    marker COUNTS as unowned.

    `owned` names the FORM of each ownership (73rd audit B2), because on
    2026-09-05 the count went 4 -> 0 in three minutes by routing into a queue
    whose own instrument reads `drain UNBOUNDED` — `AT floor — ok` was true
    and misleading at once. The forms, strongest first: `repaired_by`,
    `disposed`, `queue-row` (a row about the spec — by slug or by in-block
    id — carries a `DUE:`, a dated promise), `held-on-blocker` (74th audit
    B2: the row's only clock is a `BLOCKED-BY:` — a named release condition,
    legal payment per the queue's own contract, weaker than a date because
    it promises order rather than time), `mention-only` (a bare prose
    appearance — the weakest owner, and the map should say so). The count
    itself is untouched: the number stays at floor; the map stops implying
    repair.
    """
    if by_id is None:
        from .registry import BY_ID
        by_id = BY_ID
    if results is None:
        results = {}
        p = Path(__file__).resolve().parent / "ledger.json"
        if p.is_file():
            results = json.load(open(p)).get("results", {})
    if queue_doc is None:
        qp = Path(__file__).resolve().parent.parent / "docs" / "REVIEW_QUEUE.md"
        queue_doc = qp.read_text() if qp.is_file() else ""
    out = {"unowned": [], "disposed": {}, "malformed": [], "owned": {},
           "count": 0}
    for sid in sorted(results):
        if results[sid].get("status") != "FAIL" or sid not in by_id:
            continue
        spec = by_id[sid]
        if getattr(spec, "repaired_by", None):
            out["owned"][sid] = "repaired_by"
            continue
        notes = getattr(spec, "notes", "") or ""
        m = FAIL_DISPOSED_RX.search(notes)
        if m:
            out["disposed"][sid] = (m.group("authority"), m.group("date"))
            out["owned"][sid] = "disposed"
            continue
        if "FAIL-DISPOSED" in notes:
            out["malformed"].append(sid)  # a broken disposition disposes
            # nothing — it falls through to the unowned test below.
        # the clocked-row reader first — it also honours a row's SLUG, which
        # the literal search below cannot see; then the id-boundary mention
        # fallback: `Z.1` must not be satisfied by `Z.11` or `Z.1.B`, but a
        # sentence-final `Z.1.` must count.
        form = _owned_by_dued_row(sid, queue_doc)
        if form:
            out["owned"][sid] = form
            continue
        if re.search(r"(?<![A-Za-z0-9.])" + re.escape(sid) + r"(?!\.?\w)",
                     queue_doc):
            out["owned"][sid] = "mention-only"
            continue
        out["unowned"].append(sid)
    out["count"] = len(out["unowned"])
    return out


def fail_unowned_ratchet(baseline: int = FAIL_UNOWNED_BASELINE,
                         count_fn=None) -> dict:
    """Compare the live FAIL-UNOWNED count against the shrink-only baseline.

    Same contract as `unreachable_ratchet`: `{"count", "unowned", "disposed",
    "malformed", "baseline", "grown", "stale_baseline", "refused"}` — the last
    three are message lists, empty when healthy. A detector that raises
    REFUSES the count rather than classifying it.
    """
    out = {"count": None, "unowned": None, "disposed": None, "malformed": None,
           "baseline": baseline, "grown": [], "stale_baseline": [],
           "refused": []}
    if count_fn is None:
        count_fn = fail_unowned
    try:
        f = count_fn()
    except Exception as exc:
        out["refused"].append(f"fail-unowned ratchet: the detector refused "
                              f"({type(exc).__name__}: {exc}) — no count is "
                              f"evidence")
        return out
    out.update(count=f["count"], unowned=f["unowned"], disposed=f["disposed"],
               malformed=f["malformed"])
    if f["count"] > baseline:
        out["grown"].append(
            f"FAIL-UNOWNED GREW: {f['count']} vs baseline {baseline} "
            f"({', '.join(f['unowned'])}). A new settled FAIL has no repair "
            f"owner. The repair is to ROUTE it — a REVIEW_QUEUE row with a "
            f"DUE:, a declared repaired_by, or an explicit FAIL-DISPOSED "
            f"registry disposition — never to delete the row, re-run for a "
            f"better number, or raise FAIL_UNOWNED_BASELINE.")
    elif f["count"] < baseline:
        out["stale_baseline"].append(
            f"FAIL-UNOWNED fell to {f['count']}; FAIL_UNOWNED_BASELINE still "
            f"reads {baseline} and must be lowered in the same commit — the "
            f"ratchet only ratchets if the floor follows the number down.")
    return out


def _fail_unowned_fixture() -> List[str]:
    """Known-answer battery for `fail_unowned` + its ratchet (72nd audit B1).

    Every exclusion arm is planted alongside the state it must NOT hide, so a
    reader that starts honouring the wrong thing fails here by name. The
    conversion arms are the T0.31 P4/P5/P6 shape: each illegitimate repair —
    a disposition without its date, a queue mention of a LONGER id — must not
    lower the count.
    """
    from types import SimpleNamespace as NS
    fails = []
    by_id = {
        "Z.1": NS(repaired_by=[], notes=""),            # naked FAIL — fires
        "Z.2": NS(repaired_by=["Z.9"], notes=""),       # repaired_by — quiet
        "Z.3": NS(repaired_by=[], notes=""),            # in queue doc — quiet
        "Z.4": NS(repaired_by=[], notes="FAIL-DISPOSED: D7 2026-09-01 — "
                                        "accepted as cosmetics"),  # disposed
        "Z.5": NS(repaired_by=[], notes="FAIL-DISPOSED: someday"),  # malformed
        "Z.6": NS(repaired_by=[], notes=""),            # PASS — out of scope
        "Z.7": NS(repaired_by=[], notes=""),            # dued queue ROW — quiet
        "Z.8": NS(repaired_by=[], notes=""),            # undated row — mention
        "Z.9": NS(repaired_by=[], notes=""),            # slug row + DUE, id
                                                        # only flush-left
        "Z.10": NS(repaired_by=[], notes=""),           # slug row, clock is
                                                        # BLOCKED-BY only
    }
    results = {sid: {"status": "FAIL"} for sid in by_id}
    results["Z.6"] = {"status": "PASS"}
    # Z.3 owned by a body mention; Z.1 must NOT be laundered by `Z.11` or
    # `Z.1.B` appearing (the boundary conversion); sentence-final `Z.3.` is
    # the legitimate mention shape. Z.7 sits inside a ROUTED: block with a
    # DUE: (the strong queue form); Z.8 sits inside a ROUTED: block WITHOUT
    # one, which must read as the weak form — a row that made no dated
    # promise owns nothing more than a sentence does (73rd audit B2).
    # Z.9 and Z.10 are the 74th audit B2 class: Z.9's id appears ONLY in a
    # flush-left evidence paragraph under a DUE:-carrying header whose row
    # SLUG (`z9-...`) declares the subject — the file's real record boundary
    # is the next ROUTED:, not the first flush-left line; Z.10's only clock
    # is a BLOCKED-BY:, which REVIEW_QUEUE.md:43 and review_queue.py's
    # HOLD-WITHOUT-A-CLOCK both accept as legal payment. At slug level the
    # boundary conversion recurs: Z.1 (`z1`) must not be laundered by
    # `z10-held`.
    doc = ("instrumented by Z.11 and Z.1.B; the row's spec is Z.3.\n"
           "ROUTED: z7-row | 2026-09-01 | fixture | OPEN\n"
           "    DUE: 2026-09-13 | repairs Z.7\n"
           "ROUTED: z8-row | 2026-09-01 | fixture | OPEN\n"
           "    mentions Z.8 with no clock attached\n"
           "ROUTED: z9-flushleft-body | 2026-09-01 | fixture | OPEN\n"
           "    DUE: 2026-09-13 | the clock; the subject is named only below\n"
           "The evidence paragraph is flush-left and names Z.9 as the spec\n"
           "this row repairs.\n"
           "ROUTED: z10-held | 2026-09-01 | fixture | HELD for the fixture\n"
           "    BLOCKED-BY: z9-flushleft-body | mentions Z.10; a blocked\n"
           "        hold is legal payment for HOLD-WITHOUT-A-CLOCK\n")
    f = fail_unowned(by_id=by_id, results=results, queue_doc=doc)
    if f["unowned"] != ["Z.1", "Z.5"]:
        fails.append(f"fail-unowned: read {f['unowned']}, expected "
                     f"['Z.1', 'Z.5'] — an exclusion arm is hiding a state "
                     f"or firing on an owned one")
    if f["owned"] != {"Z.2": "repaired_by", "Z.3": "mention-only",
                      "Z.4": "disposed", "Z.7": "queue-row",
                      "Z.8": "mention-only", "Z.9": "queue-row",
                      "Z.10": "held-on-blocker"}:
        fails.append(f"fail-unowned: ownership FORMS misread {f['owned']!r} — "
                     f"the breakdown exists so `AT floor` cannot imply "
                     f"repair, and a wrong form is that implication reborn")
    if f["disposed"] != {"Z.4": ("D7", "2026-09-01")}:
        fails.append("fail-unowned: a well-formed FAIL-DISPOSED marker must "
                     "exclude, visibly, with its authority and date")
    if f["malformed"] != ["Z.5"]:
        fails.append("fail-unowned: a disposition without authority+date "
                     "must be REPORTED malformed, and dispose of nothing")
    r = fail_unowned_ratchet(baseline=1,
                             count_fn=lambda: dict(f, count=2,
                                                   unowned=["A", "B"]))
    if not r["grown"] or r["stale_baseline"] or r["refused"]:
        fails.append("fail-unowned: count above baseline must read GROWN, "
                     "alone — a new orphaned negative is the class this "
                     "exists for")
    r = fail_unowned_ratchet(baseline=3,
                             count_fn=lambda: dict(f, count=2))
    if not r["stale_baseline"] or r["grown"] or r["refused"]:
        fails.append("fail-unowned: count below baseline must demand the "
                     "shrink, or the floor stops following the number down")
    r = fail_unowned_ratchet(baseline=2, count_fn=lambda: dict(f, count=2))
    if r["grown"] or r["stale_baseline"] or r["refused"]:
        fails.append("fail-unowned: count at baseline is the healthy state — "
                     "it must be recognisable or the sick ones mean nothing")

    def _broken():
        raise RuntimeError("planted detector failure")
    r = fail_unowned_ratchet(baseline=2, count_fn=_broken)
    if not r["refused"] or r["grown"] or r["stale_baseline"] or \
            r["count"] is not None:
        fails.append("fail-unowned: a detector that raises must REFUSE the "
                     "count, not classify it")
    return fails


def queue_depth(ledger=None, by_id=None, tracked=None,
                baseline: frozenset = QUEUE_EMPTY_BASELINE,
                held=None, cpu_estimate=None, cpu_remaining_s=None) -> dict:
    """How many specs could actually be DISPATCHED today, by cost class.

    A spec is in the queue when it is **runnable** (every dependency passes),
    **implemented** (a test file exists), **tracked** (git has it — an
    untracked implementation is one `git clean` from gone and `gpu.py:274`'s
    push guard reads `--untracked-files=no`, so it cannot see it), **not
    parked**, and **not settled**.

    SETTLED means the ledger holds a verdict: `PASS` or `FAIL`. `VOID` is NOT
    settled — `SYSTEM.md` is explicit that a VOID decides nothing ("fix the
    arm, do not decide") — so VOIDs count toward depth and are ALSO reported
    separately, because a VOID needs an arm repaired before it is a dispatch
    and a reader who cannot see that would over-count the shelf.

    and **not gate-provisional** — a spec that has declared `_GATES_FROZEN =
    False` refuses its own registered run until a pilot fixes its bars, so it
    is implemented shelf furniture, not a dispatch. That last clause was the
    46th audit's RANK 2: until 2026-08-29 this function counted `SM.03`, and
    `gpu<20min` read 1 while the honest answer was 0 — the instrument built to
    say "the shelf is empty" was itself reporting the shelf as stocked.

    STILL AN UPPER BOUND, and the narrower claim is the honest one:
    `protocol.gates_frozen` detects a DECLARED refusal. A `run()` that refuses
    for some other reason — an unmet precondition, a missing artefact, a raise
    — is invisible here, because nothing in the repo makes that declarable.
    Do not read a non-zero class as "there is work to dispatch" without
    opening the spec.

    Returns `{"depth", "by_class", "void", "excluded", "empty", "new_empty",
    "known_empty", "stale_baseline"}`. `new_empty` is the fatal class.

    AND THE AFFORDABILITY JOIN (75th audit Finding 2 / B3). On 2026-09-05 this
    function printed `SO.08` under "dispatchable TODAY" for the fourteen hours
    `gate_cpu_child` was refusing it — both instruments correct, keyed to
    different facts, reconciled by hand in fourteen consecutive journal
    entries. A row the day meter would refuse right now is marked
    `unaffordable[sid] = "est Ns [provenance] vs Ms remaining"` and `fresh`
    subtracts the UNION of VOID and unaffordable from `depth`. Two scope rules,
    both deliberate: this reads only the meter's INPUTS (`child_estimate_s`
    and `remaining_s` — the same comparison `gate_cpu_child` makes), never its
    `CpuDecision`, so a transient load spike cannot flap the readout; and it
    is an ANNOTATION, not a gate — an unmarked row is still refused loudly at
    dispatch, so the failure direction of a missing budget file (full-day
    remaining, nothing marked) costs a refused attempt, not a false claim.
    `cpu_estimate`/`cpu_remaining_s` are injectable for the fixture only.
    """
    from .protocol import (Budget, Ledger, gates_frozen, module_path_for,
                           pilot_blocked, pilot_harvested, pilot_owed,
                           void_foreclosed, void_foreclosed_refusal)
    if ledger is None:
        ledger = Ledger()
    if by_id is None:
        from .registry import BY_ID
        by_id = BY_ID
    if tracked is None:
        tracked = _tracked_tests()
    if held is None:
        # {spec_id: "Dxx (decide_by ...)"} from OPEN decisions' `blocks:`
        # lists. No try/except: if the decisions doc is unreadable the honest
        # move is to fail loudly, because a silent {} re-advertises every held
        # spec — the optimistic default, the expensive direction here.
        from .decisions import holds
        held = holds(by_id=by_id)
    from .registry import ready

    parked_ids = set(parked(by_id)[0])
    by_class: Dict[str, list] = {b.value: [] for b in Budget}
    excluded: Dict[str, list] = {k: [] for k in
                                 ("unimplemented", "untracked", "parked",
                                  "settled", "gates_provisional",
                                  "void_foreclosed")}
    void: List[str] = []
    foreclosed_why: Dict[str, str] = {}
    foreclosed_refused: Dict[str, str] = {}
    for spec in ready(ledger):
        cls = spec.budget.value
        status = getattr(ledger.status(spec.id), "name", None)
        path = module_path_for(spec.id)
        # A REFUSED declaration is collected for EVERY spec with a file,
        # BEFORE any status-based exit (57th audit B1b). The old site sat
        # inside `if status == "VOID"`, after `gates_frozen is False` had
        # already dropped every never-run spec — so when a wrapped sentence
        # declared gate-provisional LC.07 foreclosed, the refusal that exists
        # "so it is LOUD" was printed by nobody. A bogus declaration on a spec
        # that has not run is exactly the case that occurred.
        if path:
            refusal = void_foreclosed_refusal(spec.id, path=path)
            if refusal:
                foreclosed_refused[spec.id] = refusal
        if spec.id in parked_ids:
            excluded["parked"].append(spec.id)
            continue
        if status in ("PASS", "FAIL"):
            excluded["settled"].append(spec.id)
            continue
        if not path:
            excluded["unimplemented"].append(spec.id)
            continue
        # `module_path_for` answers "does a FILE exist", which is a claim about
        # the filesystem. Git is the claim about the repository, and the GPU
        # backends clone from GitHub: SM.03 sat implemented-but-untracked for
        # 4.5 days while every instrument read it as present.
        if str(Path(path).resolve()) not in tracked:
            excluded["untracked"].append(spec.id)
            continue
        # `is False`, never falsiness: `None` is "does not declare", which is
        # 185 of 187 specs and means NOT APPLICABLE, not "unfrozen".
        if gates_frozen(spec.id, path=path) is False:
            excluded["gates_provisional"].append(spec.id)
            continue
        # VOID-FORECLOSED — the sixth state, and the first one outside the
        # pilot bucket (builder, 2026-08-31, BA.03).
        #
        # Everything above this line is about specs that have never run. The
        # `void` list below is about specs that HAVE, and it carried the same
        # missing state the pilot tri-state kept re-discovering: it is printed
        # as "an arm to repair, not a dispatch", which is the CHEAP reading —
        # fix the arm, re-run, get a verdict — and on the morning this was
        # written that reading was wrong for two of its five members.
        #
        # `BA.03` VOIDed on `seed_rig_ok 0.0` after 3.99 CPU-hours. Six of its
        # seven rig conjuncts were green on every seed; the one that fired was
        # the ceiling gate — the BLIND twin holds 98.9% of the horizon, so the
        # claim has 0.132 s of room and needs 1.336 s. The horizon does not move
        # by re-running. `LC.03` has been concluded since 2026-08-24 by its own
        # pre-registered fork, and this readout has advertised it as a
        # repairable arm every day since.
        #
        # So a VOID is not one state. *The arm failed and a better arm may win*
        # and *the world forecloses the measurement at any envelope* need
        # opposite units of work — a bounded re-run versus a redesign — and read
        # identically in the ledger, whose word for both is the generic "run did
        # not test the claim; not a refutation". The absence of a declaration
        # still defaults to the cheap reading, exactly as `pilot_owed`'s
        # docstring says it must not; the difference is that here the cheap
        # reading is the common one and the author who knows better now has a
        # line to write. Like `pilot_blocked`, it does NOT rescue a class.
        # The conjunction (status VOID and an ACCEPTED declaration) lives in
        # `foreclosure()`, shared with the CLAIM-DEAD ratchet (58th audit B1).
        fkind, why = foreclosure(spec.id, status=status, path=path)
        if fkind == "VOID-FORECLOSED":
            excluded["void_foreclosed"].append(spec.id)
            foreclosed_why[spec.id] = why
            continue
        by_class[cls].append(spec.id)
        if status == "VOID":
            void.append(spec.id)
            # A refused declaration (54th audit B3: no price attached) falls
            # back to the repairable ranking — an unpriced weld closes no
            # door — and its refusal was already collected at the loop head,
            # for every status (57th audit B1b).

    empty = {c for c, ids in by_class.items() if not ids}
    # FILLABLE — can this class be stocked by implementing something TODAY?
    #
    # Found by trying to obey this function's own advice (builder, 2026-08-29).
    # It reported `gpu<20min` NEWLY EMPTY and said "Implement a spec; never
    # baseline the class" — and the class named no spec to implement, so the
    # instruction could not be checked before an hour was spent on it.
    #
    # **AND THE FIRST VERSION OF THIS COMMENT WAS WRONG IN THE INTERESTING
    # DIRECTION, which is why the field is worth having.** It asserted that all
    # ten unimplemented `gpu<20min` specs were blocked and the class was
    # unfillable. That came from a throwaway script whose dependency check was
    # broken; the field, computed from `ready()`, says `gpu<20min` IS fillable
    # today — by `T3.10` — and that the genuinely unfillable class is
    # `cpu<1min`. The instrument caught its author inside ten minutes. That is
    # the whole argument for computing this rather than eyeballing it, and it
    # is LESSONS' "a quantity you can read out of the source is not a quantity
    # to estimate" arriving a second time in the same file.
    #
    # `ready()` already filters to runnable, so `excluded["unimplemented"]` is
    # exactly the set of specs an iteration COULD implement now. Counting them
    # per class turns "implement a spec" from an instruction that may be
    # unexecutable into one the reader can check before spending an hour.
    #
    # The distinction matters because the two states need opposite reactions
    # and read identically today: an empty-and-fillable class is INVENTORY
    # DEBT the builder can clear alone, and an empty-and-unfillable class is
    # STRUCTURAL — the quota at that cost is unspendable until the ladder
    # moves, and no amount of implementing will change it.
    # FILL-HELD — the state `fillable` mislabelled for a day (2026-09-03).
    # An implementable spec named in an OPEN decision's `blocks:` list is not
    # fillable TODAY: the unit it needs is on the decision desk, dated, and
    # "implement a spec" would either waste the hour or walk around the
    # decision (D19/HR.1: the run IS a 338 MB fetch the NO-FETCH default
    # forbids). The hold lived in prose — the Review's "do not fetch a corpus
    # to unblock a family" — and this readout kept printing `fillable today:
    # HR.1` against it, so two journal entries in a row had to hand-warn the
    # next iteration off one line of this instrument. Same disease as the
    # HR.5→HR.6 missing edge (65th audit B1): no instrument reads prose, so
    # the block must be an edge. `blocks:` already existed and was already
    # joined to the graph for cost; `decisions.holds()` joins it here for
    # fillability. Held is NOT unfillable either — the class has a path in,
    # it is just not the builder's — so these classes get their own state
    # below rather than the UNBLOCK misroute.
    fillable: Dict[str, list] = {c: [] for c in by_class}
    fill_held: Dict[str, list] = {c: [] for c in by_class}
    fill_held_why: Dict[str, str] = {}
    for sid in excluded["unimplemented"]:
        spec = by_id.get(sid)
        if spec is None:
            continue
        if sid in held:
            fill_held[spec.budget.value].append(sid)
            fill_held_why[sid] = held[sid]
        else:
            fillable[spec.budget.value].append(sid)
    # PILOT-OWED — the THIRD state, and it was missing (builder, 2026-08-30).
    #
    # Found the same way `fillable` was: by obeying this function's own advice.
    # It reported `gpu<20min` empty and UNFILLABLE, whose printed instruction is
    # "the repair is an UNBLOCK ... do not spend an iteration looking for a spec
    # to write here" — while the only spec at that cost, `DP.04`, was
    # implemented, tracked, runnable, unsettled, unparked and ONE PILOT from
    # dispatchable. The class was not structurally unreachable; it was one
    # bounded CPU unit away, and the instrument routed the builder to `run
    # blocked` instead, which is a different and much larger unit of work.
    #
    # The cause is that a gate-provisional spec falls out of BOTH partitions:
    # `by_class` excludes it (its `run()` refuses, correctly — the 46th audit's
    # rank 2), and `fillable` counts only `unimplemented`, so nothing counts it
    # anywhere. Two states were being used to describe three, and the state that
    # vanished is the CHEAPEST one to repair.
    #
    # This is the queue-depth blind spot one layer up. The instrument exists
    # because W34's 30 free GPU-hours died while every document blamed uptime
    # and no number could say the shelf was empty. Mislabelling "one pilot away"
    # as "unreachable" recreates exactly that: a builder who believes it leaves
    # the class empty for the same reason, holding a correct-looking readout.
    #
    # AND THE THREE-WAY SPLIT NEEDED A FOURTH STATE WITHIN HOURS, from the very
    # spec that motivated it (builder, 2026-08-30, same iteration). This field
    # named `DP.04` as `gpu<20min`'s cheapest repair; `DP.04`'s own
    # pre-registered sizing run then refuted the pilot's PRECONDITION — its
    # claim statistic has no resolution in that world at any affordable
    # envelope — so "run the pilot" would have spent two fresh seeds on a third
    # VOID. `_GATES_FROZEN = False` cannot distinguish *not piloted yet* from
    # *piloting measured not to work*, and those need opposite units of work.
    # `protocol.pilot_blocked` makes the second declarable, with its reason.
    #
    # AND THE THREE-WAY SPLIT STILL HAD A DEFAULT, WHICH IS THE WHOLE DISEASE
    # (builder, 2026-08-30, next iteration). The loop above used to read
    # *gate-provisional AND not declared blocked* as PILOT-OWED — so an author
    # who never declared anything got the CHEAP, ACTIONABLE state for free. On
    # the morning this was found, four of the five gate-provisional specs had
    # already run a pilot that measured the pilot could not succeed — `SH.02`'s
    # headroom VOID, `SM.03`'s two rig faults, `T2.11`'s two control-passes,
    # `DP.04`'s sizing refutation — and three of the four said so IN PROSE in
    # their own docstrings, where no instrument reads. Only `DP.04` had
    # declared it, so this readout advertised BOTH empty GPU classes as one
    # cheap pilot from stocked (`gpu<20min: SM.03`, `gpu<2h: T2.11`) and the
    # 21:45 handoff line repeated it to the next iteration.
    #
    # That is 2026-08-29's lesson — *a repair-class instrument with a missing
    # state defaults it to the expensive reading* — arriving again with the
    # SIGN FLIPPED, and the flipped sign is the worse one: a pessimistic default
    # wastes a reading, an optimistic default spends compute on a run whose own
    # author already recorded it as spent.
    #
    # So the fourth state is UNDECLARED and it is not a synonym for either. It
    # rescues no class, it is never a cheap repair, and `coverage` exits red on
    # it — the same shape `decisions.py` uses for an unarmed default. A spec may
    # sit gate-provisional for as long as its author likes; it may not sit there
    # without saying which of the two units of work it is waiting for.
    #
    # AND PILOT-OWED ITSELF DESCRIBED TWO UNITS OF WORK AN ORDER OF MAGNITUDE
    # APART (builder, 2026-08-30, BA.03 — the fifth state, found the same day as
    # the fourth). "Owed" reads as *spend a bounded CPU run*. But `BA.03`'s
    # seed-90 pilot had already been spent: it ran 13:15-15:00 UTC, completed,
    # and wrote `/data/ba03_pilot_seed90.json`. `_GATES_FROZEN` stayed False,
    # `_PILOT_OWED` went on asserting *"no pilot has been run: the artifact does
    # not exist"* while the artifact existed, and this readout counted BA.03 as
    # pilot-owed shelf furniture for eight hours. The real repair was to read a
    # JSON file and freeze five gates.
    #
    # That is the LOUD half of the same disease the fourth state fixed — a
    # declaration nobody re-checks against the world — and it is worse than a
    # missing state, because the prose was CONFIDENT and FALSE. `pilot_blocked`'s
    # docstring even anticipated it ("the checkable version is what lets a later
    # reader notice the pilot already ran") and then left the noticing to a
    # reader who never came.
    #
    # So PILOT-HARVESTABLE is not another thing for an author to declare. It is
    # `os.path.exists` on the path they already declared, which is why it cannot
    # rot the way the sentence above it did. It says only that the next unit is
    # *harvest and adjudicate* — cheap — and NOT that the pilot succeeded: a
    # completed pilot may still refute its own precondition, and harvesting it
    # into `_PILOT_BLOCKED` is the same act and also clears this state.
    pilot_owed_cls: Dict[str, list] = {c: [] for c in by_class}
    pilot_blocked_cls: Dict[str, list] = {c: [] for c in by_class}
    pilot_harvest_cls: Dict[str, list] = {c: [] for c in by_class}
    pilot_undeclared: List[str] = []
    blocked_why: Dict[str, str] = {}
    owed_why: Dict[str, str] = {}
    harvest_art: Dict[str, str] = {}
    for sid in excluded["gates_provisional"]:
        spec = by_id.get(sid)
        if spec is None:
            continue
        path = module_path_for(sid)
        why = pilot_blocked(sid, path=path)
        owed = pilot_owed(sid, path=path)
        # The blocked-and-not-owed conjunction lives in `foreclosure()`,
        # shared with the CLAIM-DEAD ratchet (58th audit B1); the contradiction
        # routing below stays here because UNDECLARED is queue bookkeeping,
        # not a foreclosure.
        fkind, fwhy = foreclosure(sid, path=path)
        if why and owed:
            # BOTH is a contradiction, not a majority vote. Reading it as either
            # would let the author pick the answer by writing more; it reads
            # UNDECLARED, which is the state that costs them a red exit.
            pilot_undeclared.append(sid)
        elif fkind == "PILOT-BLOCKED":
            blocked_why[sid] = fwhy
            pilot_blocked_cls[spec.budget.value].append(sid)
        elif owed:
            owed_why[sid] = owed
            # A declared artifact that EXISTS refines "owed" into "harvestable".
            # Checked against the filesystem, never against the prose, because
            # it was the prose that was wrong.
            art = pilot_harvested(sid, path=path)
            if art:
                harvest_art[sid] = art
                pilot_harvest_cls[spec.budget.value].append(sid)
            else:
                pilot_owed_cls[spec.budget.value].append(sid)
        else:
            pilot_undeclared.append(sid)
    # UNAFFORDABLE TODAY — the day meter's own comparison (est > remaining),
    # made here on the meter's INPUTS so this readout can never again disagree
    # in silence with the refusal it predicts. Scope matches
    # `cpu_budget.runner_cpu_specs`: gpu classes are metered by T0.12 and
    # `cpu<48h` is the detached lane (`admit_detached`), so neither is this
    # gate's to mark.
    if cpu_estimate is None or cpu_remaining_s is None:
        from .cpu_budget import CpuBudget, child_estimate_s
        if cpu_estimate is None:
            cpu_estimate = child_estimate_s
        if cpu_remaining_s is None:
            cpu_remaining_s = CpuBudget().remaining_s()
    unaffordable: Dict[str, str] = {}
    for cls, ids in by_class.items():
        if not cls.startswith("cpu") or cls == "cpu<48h":
            continue
        for sid in ids:
            est, prov = cpu_estimate(by_id[sid])
            if est > cpu_remaining_s:
                unaffordable[sid] = (f"est {est:.0f}s [{prov}] vs "
                                     f"{cpu_remaining_s:.0f}s remaining")
    depth = sum(len(v) for v in by_class.values())
    return {
        "depth": depth,
        # FRESH subtracts the UNION — a VOID that is also unaffordable is one
        # undispatchable row, not two.
        "fresh": depth - len(set(void) | set(unaffordable)),
        "unaffordable": dict(sorted(unaffordable.items())),
        "cpu_remaining_s": round(float(cpu_remaining_s), 2),
        "by_class": {c: sorted(ids) for c, ids in by_class.items()},
        "void": sorted(void),
        "excluded": {k: sorted(v) for k, v in excluded.items()},
        "empty": sorted(empty),
        "new_empty": sorted(empty - baseline),
        "known_empty": sorted(empty & baseline),
        "stale_baseline": sorted(c for c in baseline if c not in empty),
        "fillable": {c: sorted(ids) for c, ids in fillable.items()},
        "fill_held": {c: sorted(ids) for c, ids in fill_held.items()},
        "fill_held_why": dict(sorted(fill_held_why.items())),
        "pilot_owed": {c: sorted(ids) for c, ids in pilot_owed_cls.items()},
        "pilot_blocked": {c: sorted(ids)
                          for c, ids in pilot_blocked_cls.items()},
        "pilot_blocked_why": dict(sorted(blocked_why.items())),
        "pilot_owed_why": dict(sorted(owed_why.items())),
        "pilot_harvestable": {c: sorted(ids)
                              for c, ids in pilot_harvest_cls.items()},
        "pilot_harvestable_artifact": dict(sorted(harvest_art.items())),
        "pilot_undeclared": sorted(pilot_undeclared),
        # Empty AND no path in: neither a runnable spec to implement NOR a
        # gate-provisional one whose pilot can still succeed. Only then is the
        # repair an unblock. The `pilot_owed` conjunct is the correction —
        # without it this field called a class one bounded CPU unit from
        # stocked "structural". A pilot-BLOCKED spec deliberately does NOT
        # rescue a class from this list: its repair is a redesign, which is the
        # same KIND of work as an unblock, and pretending otherwise would put
        # the optimistic error where the pessimistic one used to be.
        # Reported, never fatal on its own: it is a fact about the ladder's
        # shape, not debt anyone incurred.
        # `pilot_harvest_cls` joins the conjunct for the same reason
        # `pilot_owed_cls` did, only more so: a class whose sole occupant has a
        # completed pilot sitting on disk is not structurally unreachable, it is
        # one file-read from stocked — the cheapest repair this instrument can
        # name, and the one it silently called "unreachable" for eight hours.
        # A FILL-HELD spec rescues its class from "NO path in" the same way a
        # pilot-owed one does — there IS a path, it is just not an implement
        # and not an unblock: it is a dated decision. Leaving held classes in
        # this list would route the builder to `run blocked` for a class whose
        # actual repair fires by calendar (or by the owner) on decide_by.
        "empty_unfillable": sorted(
            c for c in empty
            if not fillable[c] and not pilot_owed_cls[c]
            and not pilot_harvest_cls[c] and not fill_held[c]),
        "void_foreclosed_why": dict(sorted(foreclosed_why.items())),
        "void_foreclosed_refused": dict(sorted(foreclosed_refused.items())),
    }


def _tracked_tests() -> set:
    """Absolute paths of every test file git actually has.

    Shelling out rather than parsing the index: `git ls-files` is the same
    authority `gpu.py` and the push guard answer to, and re-implementing it
    would be a second definition of "tracked" that could disagree with the
    first (LESSONS: two functions computing the same thing is a defect even
    while they agree). A git failure returns the empty set, which reads as
    "nothing is tracked" — the LOUD direction.
    """
    import subprocess
    root = Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "--",
                              "experiments/tests"],
                             capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return set()
    except (OSError, subprocess.SubprocessError):
        return set()
    return {str((root / line).resolve())
            for line in out.stdout.splitlines() if line.strip()}


def _queue_fixture() -> List[str]:
    """Known-answer battery. A scanner nobody has watched catch something is a
    scanner nobody has tested — and the 43rd audit's rule is sharper than
    that: **a guard's fixture must contain the case the guard is FOR**, and a
    fixture row whose label contradicts its assertion is a defect report, not
    a test. So the rows below are named for what they ARE, and the row this
    instrument exists for — an implemented, runnable, untracked spec that
    every other instrument reads as present — asserts that it is EXCLUDED.
    """
    from .protocol import Budget, Status

    class _Spec:
        def __init__(self, sid, budget, notes=""):
            self.id, self.budget, self.notes = sid, budget, notes
            self.depends_on: List[str] = []

    class _Led:
        def __init__(self, st):
            self._st = st

        def status(self, sid):
            return self._st.get(sid)

        def blocked_by(self, spec):
            return []

    # The last column is `gates_frozen`'s answer: None = does not declare
    # (185 of 187 real specs), True = declared and frozen, False = declared
    # provisional. Q.08 and Q.09 are the pair that distinguishes "unfrozen"
    # from "silent" — a fixture with only the False row would pass even if the
    # clause tested falsiness and excluded every non-declaring spec on Earth.
    rows = [
        ("Q.01", Budget.GPU, "", None, True, None),      # the healthy queue row
        ("Q.02", Budget.GPU, "", Status.FAIL, True, None),  # settled: a verdict exists
        ("Q.03", Budget.GPU, "", Status.VOID, True, None),  # VOID is NOT a verdict
        ("Q.04", Budget.GPU, "PARKED: 2026-08-20 — arm redesign owed", None, True, None),
        ("Q.05", Budget.GPU, "", None, False, None),     # implemented but UNTRACKED
        ("Q.06", Budget.CPU, "", None, True, None),      # a different cost class
        ("Q.07", Budget.GPU, "", None, True, None),      # no file: unimplemented
        ("Q.08", Budget.GPU, "", None, True, False),     # gates PROVISIONAL: refuses
        ("Q.09", Budget.GPU, "", None, True, True),      # declared AND frozen: counts
        # The ONLY occupant of its class, and gate-provisional: that class is
        # empty, and its repair is a PILOT, not an unblock. Without this row
        # every empty class in the fixture is empty for the same reason and the
        # three-way split cannot be distinguished from the two-way one.
        ("Q.10", Budget.GPU_SHORT, "", None, True, False),
        # Gate-provisional AND the sole occupant of its class, like Q.10 — but
        # it DECLARES why its pilot cannot succeed. The DP.04 case: running the
        # pilot would spend fresh seeds on a known VOID, so this class is NOT
        # pilot-owed and its repair is a redesign.
        ("Q.11", Budget.CPU_LONG, "", None, True, False),
        # THE FOURTH STATE, and the one that used to be silently free.
        # Gate-provisional and the sole occupant of its class, like Q.10 and
        # Q.11 — and it declares NEITHER. Before 2026-08-30 this row was
        # indistinguishable from Q.10 and its class was advertised as one cheap
        # pilot from stocked. It must now read UNDECLARED and buy nothing.
        ("Q.12", Budget.CPU_DAYS, "", None, True, False),
        # THE CONTRADICTION. Declares BOTH flags, and must read UNDECLARED
        # rather than either one — otherwise an author could pick the answer by
        # writing more, which is the "run-until-pass" shape one surface over.
        # It shares gpu<2h with stocked specs so it exercises the branch
        # without deciding a class.
        ("Q.13", Budget.GPU, "", None, True, False),
        # THE FIFTH STATE. Gate-provisional, sole occupant of its class, and it
        # declares the pilot OWED exactly as Q.10 does — the two rows are
        # identical to every reader of the SOURCE. They differ only in that the
        # artifact Q.14 names EXISTS. The BA.03 case: the run is already spent,
        # so this class is not pilot-owed, it is one file-read from stocked, and
        # reading it as OWED sends a builder to buy a CPU run they already own.
        ("Q.14", Budget.GPU_LONG, "", None, True, False),
        # THE SIXTH STATE, and the first outside the pilot bucket. VOID, gates
        # frozen, sole occupant of its class, and it DECLARES that re-running
        # cannot change the verdict. The BA.03 case: its class must go EMPTY —
        # a foreclosed VOID rescues nothing — and it must NOT appear in `void`,
        # which is printed as "an arm to repair".
        ("Q.15", Budget.CPU_FAST, "", Status.VOID, True, True),
        # THE QUIET DIRECTION, and the one that would be a park with better
        # manners: a spec that declares foreclosure but has NOT VOIDed. Q.03 is
        # its VOID twin. It must keep counting — the declaration describes a
        # recorded verdict, so without one it may not mute anything. Shares
        # gpu<2h with stocked specs so it exercises the branch without
        # deciding a class.
        ("Q.16", Budget.GPU, "", None, True, True),
        # THE REFUSED WELD (54th audit B3). VOID, and it WROTE a foreclosure
        # declaration that validation refused — no `FORECLOSURE ARITHMETIC:` /
        # `BLAST RADIUS:` price attached. It must stay in the queue AND in
        # `void` (an unpriced weld closes no door), and the refusal must
        # survive into the readout — a bare "arm to repair" line about a spec
        # somebody tried to weld is the B2 misroute wearing a new state.
        # Shares gpu<2h with stocked specs so it decides no class.
        ("Q.17", Budget.GPU, "", Status.VOID, True, True),
        # THE PHANTOM ON A SPEC THAT NEVER RAN (57th audit B1). Gates
        # provisional, no verdict, and a refused declaration at its margin —
        # LC.07's exact shape, which the pre-B1 code could not see because
        # refusals were collected only for VOID specs, three exits after
        # `gates_frozen is False` had dropped this one. The refusal must
        # surface even though every status-based branch excludes the spec.
        ("Q.18", Budget.GPU, "", None, True, False),
        # THE HELD ROW (D19/HR.1, 2026-09-03). Unimplemented like Q.07 — so
        # `fillable` is the only reader that could advertise it — and named in
        # an OPEN decision's `blocks:`. It shares cpu<1min with Q.15
        # (foreclosed VOID), a class the pre-held code pinned as
        # empty_unfillable: with the hold it must read FILL-HELD instead,
        # because the class HAS a path in — a dated one, on the decision desk —
        # and "go do `run blocked`" is the wrong routing for a calendar.
        ("Q.19", Budget.CPU_FAST, "", None, True, None),
        # THE AFFORDABILITY TRIPLE (75th audit F2/B3). All three share Q.06's
        # cpu<10min so they decide no class. Q.06 becomes the row the guard is
        # FOR: healthy queue member the day meter would refuse right now —
        # the SO.08 case, where "dispatchable TODAY" was wrong about the one
        # row that mattered for fourteen hours. Q.20 is the affordable
        # control: an estimate under the remaining seconds must NOT be marked,
        # or the annotation is a list, not a comparison. Q.21 is the UNION
        # row: VOID and unaffordable at once — `fresh` must subtract it once.
        ("Q.20", Budget.CPU, "", None, True, None),
        ("Q.21", Budget.CPU, "", Status.VOID, True, None),
    ]
    # `pilot_blocked` / `pilot_owed` answers per spec: a reason string, or None
    # for "does not declare". Q.08 and Q.12 declare neither — Q.12 deliberately,
    # as the undeclared row; Q.08 shares a class with stocked specs, so it
    # exercises the state without deciding a class.
    blocked = {"Q.11": "sizing refuted the pilot's precondition",
               "Q.13": "a run measured the precondition fails"}
    owed = {"Q.10": "the pilot will freeze CEM_K_FIT against its artifact",
            "Q.13": "the pilot will freeze the bars",
            "Q.14": "no pilot has been run: the artifact does not exist"}
    # `pilot_harvested`'s answers: the declared artifact path IF it exists on
    # disk. Q.14 is the BA.03 row — it declares the pilot OWED, in exactly the
    # words BA.03 used, and the file it names is sitting there. Q.10 declares
    # the same state with no artifact, which is the honest owed reading, so the
    # two rows differ ONLY in the filesystem and the fixture can tell whether
    # the split is being read from disk or from prose.
    harvested = {"Q.14": "/data/q14_pilot_seed90.json"}
    # `child_estimate_s`'s answers plus the day's remaining seconds, injected.
    # The DEFAULT is the trap: any spec the clause queries that is not listed
    # here reads 999999s — instantly unaffordable at 100s remaining — so if
    # the scope ever leaks to a gpu row or the detached lane, the
    # exact-equality assertion below names the leak. Q.20's estimate is the
    # only one under the remaining, which is what makes it the control.
    est_stub = {"Q.06": (5000.0, "ENUM (no recorded duration to project from)"),
                "Q.20": (12.0, "MEASURED 0.50s x4 + 10s"),
                "Q.21": (7200.0, "ENUM (no recorded duration to project from)")}
    cpu_est = lambda spec: est_stub.get(spec.id, (999999.0, "LEAKED-QUERY"))
    # `void_foreclosed`'s answers. Q.15 is VOID and declares; Q.16 declares in
    # identical words and has no verdict, so the CLAUSE — not the reader — is
    # what must keep it in the queue.
    foreclosed = {"Q.15": "the blind twin holds 98.9% of the horizon",
                  "Q.16": "the blind twin holds 98.9% of the horizon"}
    # `void_foreclosed_refusal`'s answers. Q.17 (VOID) and Q.18 (never ran,
    # gates provisional) both wrote refused declarations and BOTH must
    # surface — the 57th audit's B1 withdrew the 54th's "no verdict, nothing
    # to refuse" rule after LC.07's phantom sat invisible on exactly that
    # branch. Q.16 has a COMPLETE declaration, so in reality the refusal
    # reader answers None for it; the stub models that.
    refusals = {"Q.17": "declaration REFUSED — missing `BLAST RADIUS:`",
                "Q.18": "declaration REFUSED — missing `FORECLOSURE "
                        "ARITHMETIC:`, `BLAST RADIUS:`"}
    by_id = {sid: _Spec(sid, b, n) for sid, b, n, _s, _t, _g in rows}
    led = _Led({sid: s for sid, _b, _n, s, _t, _g in rows if s is not None})
    tracked = {f"/x/{sid}.py" for sid, _b, _n, _s, t, _g in rows if t}
    frozen = {sid: g for sid, _b, _n, _s, _t, g in rows}

    from . import protocol as _proto
    from . import registry as _reg
    real_ready, real_mpf = _reg.ready, _proto.module_path_for
    real_gf, real_pb = _proto.gates_frozen, _proto.pilot_blocked
    real_po, real_ph = _proto.pilot_owed, _proto.pilot_harvested
    real_vf = _proto.void_foreclosed
    real_vfr = _proto.void_foreclosed_refusal
    _proto.void_foreclosed = lambda sid, path=None: foreclosed.get(sid)
    _proto.void_foreclosed_refusal = lambda sid, path=None: refusals.get(sid)
    _reg.ready = lambda _l: list(by_id.values())
    _proto.module_path_for = lambda sid, strict=False: (
        None if sid in ("Q.07", "Q.19") else f"/x/{sid}.py")
    _proto.gates_frozen = lambda sid, path=None: frozen.get(sid)
    _proto.pilot_blocked = lambda sid, path=None: blocked.get(sid)
    _proto.pilot_owed = lambda sid, path=None: owed.get(sid)
    _proto.pilot_harvested = lambda sid, path=None: harvested.get(sid)
    try:
        q = queue_depth(ledger=led, by_id=by_id, tracked=tracked,
                        baseline=frozenset({"gpu<8h"}),
                        held={"Q.19": "D99 (decide_by 2026-09-14)"},
                        cpu_estimate=cpu_est, cpu_remaining_s=100.0)
        # THE CONTROL: the identical fixture with no hold declared must
        # advertise Q.19 as fillable — otherwise the held state is not being
        # read from the decisions join and the assertion above it proves
        # nothing about the edge.
        q_nohold = queue_depth(ledger=led, by_id=by_id, tracked=tracked,
                               baseline=frozenset({"gpu<8h"}), held={},
                               cpu_estimate=cpu_est, cpu_remaining_s=100.0)
        # THE RICH-DAY CONTROL: same estimates against a fresh day's worth of
        # remaining seconds. Everything must read affordable — otherwise the
        # marks above come from a list, not from the meter's comparison.
        q_rich = queue_depth(ledger=led, by_id=by_id, tracked=tracked,
                             baseline=frozenset({"gpu<8h"}),
                             held={"Q.19": "D99 (decide_by 2026-09-14)"},
                             cpu_estimate=cpu_est, cpu_remaining_s=1e9)
    finally:
        _reg.ready, _proto.module_path_for = real_ready, real_mpf
        _proto.gates_frozen, _proto.pilot_blocked = real_gf, real_pb
        _proto.pilot_owed, _proto.pilot_harvested = real_po, real_ph
        _proto.void_foreclosed = real_vf
        _proto.void_foreclosed_refusal = real_vfr

    fails = []
    if q["by_class"]["gpu<2h"] != ["Q.01", "Q.03", "Q.09", "Q.16", "Q.17"]:
        fails.append(f"gpu<2h queue should be [Q.01, Q.03, Q.09, Q.16, Q.17] "
                     f"(VOID is not a verdict; a spec that declares FROZEN "
                     f"gates counts; a foreclosure declaration without a VOID "
                     f"mutes nothing; a REFUSED one closes no door), got "
                     f"{q['by_class']['gpu<2h']}")
    if q["void"] != ["Q.03", "Q.17", "Q.21"]:
        fails.append(f"VOID must be reported separately, got {q['void']}")
    if q["excluded"]["settled"] != ["Q.02"]:
        fails.append(f"FAIL is settled, got {q['excluded']['settled']}")
    if q["excluded"]["parked"] != ["Q.04"]:
        fails.append(f"parked must not count, got {q['excluded']['parked']}")
    if q["excluded"]["unimplemented"] != ["Q.07", "Q.19"]:
        fails.append(f"a spec with no file is unimplemented, got "
                     f"{q['excluded']['unimplemented']}")
    # THE ROW THIS INSTRUMENT EXISTS FOR.
    if q["excluded"]["untracked"] != ["Q.05"]:
        fails.append(f"an UNTRACKED implementation must be excluded — the "
                     f"SM.03 case — got {q['excluded']['untracked']}")
    # THE ROW THE 46th AUDIT'S RANK 2 EXISTS FOR: runnable, implemented,
    # tracked, unsettled, unparked — and its own `run()` refuses.
    if q["excluded"]["gates_provisional"] != ["Q.08", "Q.10", "Q.11", "Q.12",
                                              "Q.13", "Q.14", "Q.18"]:
        fails.append(f"a spec with PROVISIONAL gates refuses its own "
                     f"registered run and must be excluded — the SM.03 case — "
                     f"got {q['excluded']['gates_provisional']}")
    # THE THREE-WAY SPLIT. gpu<20min is EMPTY and its only occupant Q.10 is
    # gate-provisional: the repair is a PILOT, so it is neither stocked nor
    # structurally unreachable. Before this pair of assertions the field
    # reported it as unfillable and told the builder to go do `run blocked` —
    # the DP.04 case, where the class was one bounded CPU unit from stocked.
    if q["pilot_owed"]["gpu<20min"] != ["Q.10"]:
        fails.append(f"an empty class whose only occupant has provisional "
                     f"gates is PILOT-OWED, got {q['pilot_owed']}")
    if "gpu<20min" in q["empty_unfillable"]:
        fails.append("a pilot-owed class is NOT unfillable: its repair is a "
                     "pilot, not an unblock")
    # THE HELD STATE (D19/HR.1). Q.19 is implementable and named in an open
    # decision's `blocks:` — its class must read FILL-HELD: not advertised as
    # fillable (that exact line earned two journal hand-warnings in one day),
    # not condemned as unfillable (the class has a dated path in), and the
    # decision string must survive to the readout so the reader gets the
    # calendar instead of a bare id.
    if q["fill_held"]["cpu<1min"] != ["Q.19"]:
        fails.append(f"a held spec's class is FILL-HELD, got {q['fill_held']}")
    if "Q.19" in q["fillable"]["cpu<1min"]:
        fails.append("a held spec must NOT be advertised as fillable — that "
                     "is the D19/HR.1 misroute this state exists to stop")
    if "cpu<1min" in q["empty_unfillable"]:
        fails.append("a FILL-HELD class is not unfillable: its path in is the "
                     "decision desk, not `run blocked`")
    if q["fill_held_why"].get("Q.19") != "D99 (decide_by 2026-09-14)":
        fails.append(f"the holding decision must survive into the readout, "
                     f"got {q['fill_held_why']}")
    # ...and the control: strip the hold and the same spec MUST come back as
    # fillable, or the assertion above is not about the decisions edge at all.
    if q_nohold["fillable"]["cpu<1min"] != ["Q.19"]:
        fails.append(f"without a hold Q.19 is plainly fillable, got "
                     f"{q_nohold['fillable']}")
    # (The pre-held pin "a class with nothing at all IS unfillable" lives on
    # at cpu<48h/Q.12; cpu<1min now exercises the held three-way instead.)
    # THE FOURTH STATE. Q.11 is gate-provisional and the sole occupant of
    # cpu<2h, exactly like Q.10 — but it DECLARES that its pilot cannot
    # succeed, so its class must NOT be advertised as pilot-owed. The DP.04
    # case: obeying "run the pilot" would spend fresh seeds on a known VOID.
    if q["pilot_blocked"]["cpu<2h"] != ["Q.11"]:
        fails.append(f"a spec declaring _PILOT_BLOCKED is pilot-BLOCKED, got "
                     f"{q['pilot_blocked']}")
    if q["pilot_owed"]["cpu<2h"]:
        fails.append(f"a pilot-BLOCKED spec must not also read pilot-owed — "
                     f"that is the misroute this state exists to stop, got "
                     f"{q['pilot_owed']['cpu<2h']}")
    if "cpu<2h" not in q["empty_unfillable"]:
        fails.append("a class whose only occupant is pilot-BLOCKED has no "
                     "cheap path in: its repair is a redesign, so it belongs "
                     "in empty_unfillable")
    if q["pilot_blocked_why"].get("Q.11") != blocked["Q.11"]:
        fails.append(f"the blocking REASON must survive into the readout — a "
                     f"blocked pilot without its evidence is a park with "
                     f"better manners, got {q['pilot_blocked_why']}")
    # THE FOURTH STATE — the default this iteration removed. Q.12 is
    # gate-provisional, sole occupant of cpu<48h, and declares NEITHER flag. It
    # must read UNDECLARED, must NOT read pilot-owed, and must NOT rescue its
    # class: every one of these three assertions fails against the code as it
    # stood this morning, which advertised SM.03 and T2.11 as cheap pilots
    # months after their own pilots were spent.
    if q["pilot_undeclared"] != ["Q.08", "Q.12", "Q.13", "Q.18"]:
        fails.append(f"a gate-provisional spec declaring neither flag — or "
                     f"BOTH, which is a contradiction, not a vote — is "
                     f"UNDECLARED, got {q['pilot_undeclared']}")
    if "Q.13" in q["pilot_blocked"]["gpu<2h"] or q["pilot_owed"]["gpu<2h"]:
        fails.append(f"a spec declaring BOTH flags must read neither: letting "
                     f"one win lets the author choose the answer by writing "
                     f"more, got blocked={q['pilot_blocked']['gpu<2h']} "
                     f"owed={q['pilot_owed']['gpu<2h']}")
    if q["pilot_owed"]["cpu<48h"]:
        fails.append(f"an UNDECLARED spec must not default to pilot-owed — "
                     f"that default is what sent a builder to spend a pilot "
                     f"four specs had already spent, got "
                     f"{q['pilot_owed']['cpu<48h']}")
    if "cpu<48h" not in q["empty_unfillable"]:
        fails.append("an UNDECLARED spec rescues no class: until someone says "
                     "which unit of work it waits for, there is no known cheap "
                     "path in")
    # And the owed REASON must survive too, for `pilot_blocked_why`'s reason:
    # "owed" without saying what the pilot would freeze is the claim a later
    # reader cannot check against a pilot that has already run.
    if q["pilot_owed_why"].get("Q.10") != owed["Q.10"]:
        fails.append(f"the owed reason must survive into the readout, got "
                     f"{q['pilot_owed_why']}")
    # THE FIFTH STATE — the BA.03 row. Q.14 declares OWED, like Q.10, and its
    # artifact exists, unlike Q.10's. Every assertion below fails against the
    # code as it stood at 23:00 on 2026-08-30, which read the prose and never
    # the disk.
    if q["pilot_harvestable"]["gpu<8h"] != ["Q.14"]:
        fails.append(f"a spec whose declared pilot artifact EXISTS is "
                     f"HARVESTABLE, not owed: the run is already spent, got "
                     f"{q['pilot_harvestable']}")
    if q["pilot_owed"]["gpu<8h"]:
        fails.append(f"a harvestable pilot must leave PILOT-OWED — otherwise "
                     f"the readout sends a builder to buy a CPU run they "
                     f"already own, got {q['pilot_owed']['gpu<8h']}")
    if "gpu<8h" in q["empty_unfillable"]:
        fails.append("a class whose sole occupant has a completed pilot on "
                     "disk is one file-read from stocked, not structurally "
                     "unreachable")
    if q["pilot_harvestable_artifact"].get("Q.14") != harvested["Q.14"]:
        fails.append(f"the artifact PATH must survive into the readout — "
                     f"'go read something somewhere' is the state this "
                     f"replaces, got {q['pilot_harvestable_artifact']}")
    # And the separator: Q.10 declares the identical state with NO artifact and
    # must stay OWED. Without this row, hardcoding every owed spec to
    # harvestable would pass the four assertions above.
    if q["pilot_owed"]["gpu<20min"] != ["Q.10"] or q["pilot_harvestable"][
            "gpu<20min"]:
        fails.append(f"an owed pilot whose artifact does NOT exist stays owed "
                     f"— that is the honest reading and the split must come "
                     f"from the filesystem, got owed="
                     f"{q['pilot_owed']['gpu<20min']} harvestable="
                     f"{q['pilot_harvestable']['gpu<20min']}")
    # THE SIXTH STATE — the BA.03 row, one bucket over from the pilot states.
    # Q.15 is VOID, gates frozen, tracked, runnable, sole occupant of cpu<1min,
    # and it declares that re-running cannot change the verdict. Against the
    # code as it stood this morning it counted as queue depth AND appeared in
    # `void`, which prints as "an arm to repair, not a dispatch" — the cheap
    # reading, and the wrong one for two of the five live members.
    if q["excluded"]["void_foreclosed"] != ["Q.15"]:
        fails.append(f"a VOID spec declaring VOID-FORECLOSED must be excluded "
                     f"— re-running it cannot change the verdict — got "
                     f"{q['excluded']['void_foreclosed']}")
    if "Q.15" in q["void"] or "Q.15" in q["by_class"]["cpu<1min"]:
        fails.append(f"a foreclosed VOID is not 'an arm to repair' and not "
                     f"queue depth, got void={q['void']} "
                     f"cpu<1min={q['by_class']['cpu<1min']}")
    if q["void_foreclosed_why"].get("Q.15") != foreclosed["Q.15"]:
        fails.append(f"the foreclosure REASON must survive into the readout — "
                     f"without it this is a park on the author's say-so, got "
                     f"{q['void_foreclosed_why']}")
    # ...and it rescues nothing, for `pilot_blocked`'s reason: a redesign is
    # the same KIND of work as an unblock. Since Q.19 joined this class
    # (2026-09-03) the composite reads FILL-HELD — the foreclosed VOID still
    # rescues nothing, but the held implementable spec is a real dated path
    # in, so membership in empty_unfillable is now pinned at cpu<2h (Q.11)
    # and cpu<48h (Q.12) instead. The no-hold control below keeps Q.15's own
    # contribution honest: even with a hold declared nowhere, a foreclosed
    # VOID never puts its class back in the queue.
    if "cpu<1min" in q["empty_unfillable"]:
        fails.append("a FILL-HELD class is not unfillable, whatever else "
                     "shares it: the path in is the decision desk")
    if "Q.15" in q_nohold["by_class"]["cpu<1min"] or "Q.15" in q_nohold["void"]:
        fails.append("a foreclosed VOID must stay excluded with or without "
                     "holds in play")
    # THE QUIET DIRECTION, and the one that would make this a park with better
    # manners. Q.16 declares foreclosure in Q.15's exact words and has NO
    # verdict. The declaration describes a recorded VOID, so without one it may
    # mute nothing — hardcoding "declares -> excluded" passes every assertion
    # above and fails this one.
    if "Q.16" in q["excluded"]["void_foreclosed"]:
        fails.append(f"a foreclosure declaration on a spec that has not VOIDed "
                     f"must not exclude it: a spec may not go quiet ahead of "
                     f"its own evidence, got "
                     f"{q['excluded']['void_foreclosed']}")
    # THE REFUSED WELD (54th audit B3; scope widened by the 57th's B1). Q.17
    # is VOID and wrote a declaration that validation refused. It stays in the
    # queue and in `void` (an unpriced weld closes no door), and its refusal
    # survives into the readout (the loud half — without it the fallback IS
    # the B2 misroute). Q.18 never ran at all — gates provisional, dropped by
    # the earliest status exit — and its refusal must surface anyway: LC.07's
    # phantom lived on exactly the branch the old VOID-only collection could
    # not see.
    if "Q.17" not in q["void"] or "Q.17" not in q["by_class"]["gpu<2h"]:
        fails.append(f"a refused foreclosure stays repairable — got "
                     f"void={q['void']} gpu<2h={q['by_class']['gpu<2h']}")
    if q["void_foreclosed_refused"] != {"Q.17": refusals["Q.17"],
                                        "Q.18": refusals["Q.18"]}:
        fails.append(f"the REFUSAL must survive into the readout for every "
                     f"spec that wrote an unpriced declaration, whatever its "
                     f"status — got {q['void_foreclosed_refused']}")
    if "Q.17" in q["excluded"]["void_foreclosed"]:
        fails.append(f"a refused declaration must not exclude — got "
                     f"{q['excluded']['void_foreclosed']}")
    if q["by_class"]["cpu<10min"] != ["Q.06", "Q.20", "Q.21"]:
        fails.append(f"cost classes must not merge, got {q['by_class']}")
    if q["depth"] != 8:
        fails.append(f"depth should be 8 (Q.17, the refused weld, still "
                     f"counts; an UNAFFORDABLE row is still queue depth — "
                     f"the mark is about today, not about the spec), got "
                     f"{q['depth']}")
    # THE AFFORDABILITY TRIPLE (75th audit F2/B3). Exact equality is the
    # leak-catcher: the stub answers 999999s for any spec it was not told
    # about, so a clause that queries a gpu row or the detached lane marks it
    # and fails here BY NAME.
    want_unaff = {
        "Q.06": "est 5000s [ENUM (no recorded duration to project from)] "
                "vs 100s remaining",
        "Q.21": "est 7200s [ENUM (no recorded duration to project from)] "
                "vs 100s remaining"}
    if q["unaffordable"] != want_unaff:
        fails.append(f"the day meter's refusal must reach this readout with "
                     f"its arithmetic AND its provenance — an [ENUM] estimate "
                     f"is a guess and the reader deserves to see that — and "
                     f"ONLY runner-lane cpu rows may be marked (SO.08 sat "
                     f"under 'dispatchable TODAY' for fourteen refused "
                     f"hours), got {q['unaffordable']}")
    if q["fresh"] != 4:
        fails.append(f"fresh must subtract the UNION of VOID and "
                     f"UNAFFORDABLE from depth (8 - |{{Q.03,Q.17,Q.21}} u "
                     f"{{Q.06,Q.21}}| = 4): Q.21 is both and may not be "
                     f"subtracted twice, got {q['fresh']}")
    # ...and the rich-day control: same estimates, fresh day's remaining.
    # If anything stays marked, the mark came from a list, not a comparison.
    if q_rich["unaffordable"] or q_rich["fresh"] != 5:
        fails.append(f"with a full day remaining nothing is unaffordable and "
                     f"fresh is depth minus VOID alone (8 - 3 = 5), got "
                     f"unaffordable={q_rich['unaffordable']} "
                     f"fresh={q_rich['fresh']}")
    # gpu<20min is empty and NOT in this fixture's baseline: new debt, a red.
    if "gpu<20min" not in q["new_empty"]:
        fails.append(f"an unlisted empty class is new debt, got "
                     f"{q['new_empty']}")
    # gpu<8h is in the baseline and empty: known debt, not a red.
    if q["known_empty"] != ["gpu<8h"]:
        fails.append(f"baselined empty class is known debt, got "
                     f"{q['known_empty']}")
    return fails


def _gates_frozen_fixture() -> List[str]:
    """Known-answer battery for the READER, which `_queue_fixture` cannot test.

    That fixture monkeypatches `gates_frozen` to inject its answers — correctly,
    because it is testing the exclusion CLAUSE — which leaves the AST parse
    itself covered by nothing. Two instruments, two fixtures: a fixture that
    stubs the thing under test has moved the test somewhere else.

    The last row is the one that pays for this function: `SM.03`'s real file on
    disk, read the way the real caller reads it. A reader that is right about
    nine synthetic strings and wrong about the only file it is pointed at has
    told the truth about nothing.
    """
    import tempfile
    from .protocol import gates_frozen, module_path_for

    cases = [
        ("no declaration at all", "X = 1\n", None),
        ("declared frozen", "_GATES_FROZEN = True\n", True),
        ("declared provisional", "_GATES_FROZEN = False\n", False),
        # Python's own answer is the last binding; so is ours.
        ("re-assigned, last wins (True)",
         "_GATES_FROZEN = False\n_GATES_FROZEN = True\n", True),
        ("re-assigned, last wins (False)",
         "_GATES_FROZEN = True\n_GATES_FROZEN = False\n", False),
        # The LOUD direction: cannot be established by reading the source.
        ("non-literal value is not a freeze",
         "import os\n_GATES_FROZEN = os.environ.get('X') == '1'\n", False),
        ("truthy non-True is not a freeze", "_GATES_FROZEN = 1\n", False),
        ("annotated assignment counts", "_GATES_FROZEN: bool = True\n", True),
        ("a syntax error cannot be dispatched", "def (:\n", False),
        # A flag set inside a function is not the module's declaration — it is
        # a local, and reading it as one would let any helper forge a freeze.
        ("function-local assignment is not a declaration",
         "def f():\n    _GATES_FROZEN = True\n", None),
    ]
    fails = []
    with tempfile.TemporaryDirectory() as d:
        for i, (label, src, want) in enumerate(cases):
            p = Path(d) / f"case_{i}.py"
            p.write_text(src)
            got = gates_frozen("X.00", path=p)
            if got is not want:
                fails.append(f"gates_frozen: {label} -> want {want}, got {got}")
        if gates_frozen("X.00", path=Path(d) / "gone.py") is not False:
            fails.append("gates_frozen: an unreadable file must read False "
                         "(the loud direction)")
    if gates_frozen("NO.SUCH.SPEC") is not None:
        fails.append("gates_frozen: an unimplemented spec has no file and no "
                     "declaration -> None, not an accusation")
    # The live files this instrument was built for, read end to end. The
    # assertion is `is not None` — that the reader SEES the idiom — and
    # deliberately not the current value: `SM.02`/`SM.03` are SUPPOSED to flip
    # to True when a pilot freezes their bars, and a fixture that pinned the
    # value would go red on exactly the event it is waiting for.
    for sid in ("SM.02", "SM.03"):
        if module_path_for(sid) and gates_frozen(sid) is None:
            fails.append(f"gates_frozen: {sid} uses the `_GATES_FROZEN` idiom "
                         f"in the tree and the reader read it as 'does not "
                         f"declare' — the 46th audit RANK 2 case, unfixed")
    return fails


def _pilot_blocked_fixture() -> List[str]:
    """Known-answer battery for `protocol.pilot_blocked`, the READER.

    Written because two mutations of that reader — always return `None`, and
    return a reason for every spec — left `_queue_fixture` GREEN (builder,
    2026-08-30). That fixture stubs `pilot_blocked` to test the CLAUSE, which
    is right, and it is exactly the hole `_gates_frozen_fixture` was created
    for one instrument earlier. The same lesson twice in one file: a fixture
    that stubs the thing under test has moved the test somewhere else, so a new
    reader needs its own battery on the day it is written, not after an audit.

    Both directions bite. Reading `None` when a reason is declared re-opens the
    misroute (`DP.04` advertised as a cheap pilot after its own sizing run
    refuted the pilot). Reading a reason when none is declared is worse: it
    would let any spec go quiet without evidence, which is a park nobody voted
    for.
    """
    import tempfile
    from .protocol import module_path_for, pilot_blocked

    R = "sizing refuted the precondition"
    cases = [
        ("no declaration at all", "X = 1\n", None),
        ("a declared reason", f"_PILOT_BLOCKED = {R!r}\n", R),
        # Implicit concatenation is how a reason of any real length is written
        # — DP.04's spans several lines — so it must read as the constant it is.
        ("implicitly concatenated literal",
         "_PILOT_BLOCKED = (\n 'sizing refuted '\n 'the precondition'\n)\n", R),
        ("annotated assignment counts",
         f"_PILOT_BLOCKED: str = {R!r}\n", R),
        ("re-assigned, last wins", f"_PILOT_BLOCKED = 'old'\n"
                                   f"_PILOT_BLOCKED = {R!r}\n", R),
        # THE QUIET DIRECTION, which is the dangerous one here: anything that
        # is not a readable non-empty string is NOT a block. A spec may not go
        # quiet on an expression a reader cannot evaluate.
        ("None is not a block", "_PILOT_BLOCKED = None\n", None),
        ("empty string is not a block", "_PILOT_BLOCKED = ''\n", None),
        ("whitespace is not a block", "_PILOT_BLOCKED = '   '\n", None),
        ("True is not a reason", "_PILOT_BLOCKED = True\n", None),
        ("non-literal value is not a reason",
         "import os\n_PILOT_BLOCKED = os.environ.get('WHY')\n", None),
        ("function-local assignment is not a declaration",
         f"def f():\n    _PILOT_BLOCKED = {R!r}\n", None),
        # A syntax error must NOT mute a spec: `gates_frozen` already fails
        # loud on it (False -> out of the queue), and returning a reason here
        # would let a typo silence a pilot.
        ("a syntax error is not a block", "def (:\n", None),
    ]
    fails = []
    with tempfile.TemporaryDirectory() as d:
        for i, (label, src, want) in enumerate(cases):
            p = Path(d) / f"pb_{i}.py"
            p.write_text(src)
            got = pilot_blocked("X.00", path=p)
            if got != want:
                fails.append(f"pilot_blocked: {label} -> want {want!r}, "
                             f"got {got!r}")
        if pilot_blocked("X.00", path=Path(d) / "gone.py") is not None:
            fails.append("pilot_blocked: a missing file is not a block")
    if pilot_blocked("NO.SUCH.SPEC") is not None:
        fails.append("pilot_blocked: an unimplemented spec has no file and no "
                     "declaration -> None")
    # The live file this instrument was built for, read end to end — the
    # `_gates_frozen_fixture` precedent. Asserted as "a non-empty reason is
    # seen", not as its text, so editing the reason does not go red; the text
    # itself is the author's evidence and lives in the docstring.
    if module_path_for("DP.04") and not pilot_blocked("DP.04"):
        fails.append("pilot_blocked: DP.04 declares `_PILOT_BLOCKED` in the "
                     "tree (SIZING RECORD v1) and the reader missed it — the "
                     "misroute this state exists to stop")
    return fails


def _pilot_harvested_fixture() -> List[str]:
    """Known-answer battery for `protocol.pilot_harvested`, the READER — the
    fifth state's, written the same day as the reader like the two before it.

    The thing this reader must get right is the one thing prose cannot: the
    split is between a declared path that EXISTS and a declared path that does
    not, and BA.03 is the proof that those look identical in source. So every
    case below pairs a real temp file against its absence, and the reader is
    pointed at the filesystem, never at a stub.

    The aliasing cases matter for the same reason `_pilot_owed_fixture`'s do —
    all three readers share `_declared_reason` — with one addition specific to
    this state: `_PILOT_ARTIFACT` is a PATH, not a reason, so a file whose only
    declaration is `_PILOT_OWED` must read `None` here even though the spec is
    unambiguously pilot-owed. Harvestability is a fact about the disk that a
    spec cannot assert about itself.
    """
    import tempfile
    from .protocol import pilot_blocked, pilot_harvested, pilot_owed

    fails = []
    with tempfile.TemporaryDirectory() as d:
        art = Path(d) / "pilot_seed90.json"
        gone = Path(d) / "never_ran.json"
        art.write_text("{}\n")
        cases = [
            ("no declaration at all", "X = 1\n", None),
            # THE PAIR. Identical source shape, opposite answers, and the only
            # difference is on disk — the BA.03 case in two lines.
            ("declared path that exists",
             f"_PILOT_ARTIFACT = {str(art)!r}\n", str(art)),
            ("declared path that does not exist",
             f"_PILOT_ARTIFACT = {str(gone)!r}\n", None),
            ("annotated assignment counts",
             f"_PILOT_ARTIFACT: str = {str(art)!r}\n", str(art)),
            ("re-assigned, last wins",
             f"_PILOT_ARTIFACT = {str(gone)!r}\n"
             f"_PILOT_ARTIFACT = {str(art)!r}\n", str(art)),
            ("None is not a path", "_PILOT_ARTIFACT = None\n", None),
            ("empty string is not a path", "_PILOT_ARTIFACT = ''\n", None),
            ("non-literal value is not a path",
             "import os\n_PILOT_ARTIFACT = os.environ.get('P')\n", None),
            ("function-local assignment is not a declaration",
             f"def f():\n    _PILOT_ARTIFACT = {str(art)!r}\n", None),
            ("a syntax error is not a declaration", "def (:\n", None),
            # A DIRECTORY IS NOT A PILOT. `exists` would say yes here and the
            # next builder would be sent to read something unreadable.
            ("a directory is not an artifact",
             f"_PILOT_ARTIFACT = {str(d)!r}\n", None),
            # THE ALIASING CASES. Three readers, one `_declared_reason`.
            ("a reason is not a path",
             "_PILOT_OWED = 'the pilot will freeze the bars'\n", None),
            ("the blocked flag is not a path",
             "_PILOT_BLOCKED = 'sizing refuted the precondition'\n", None),
        ]
        for i, (label, src, want) in enumerate(cases):
            p = Path(d) / f"ph_{i}.py"
            p.write_text(src)
            got = pilot_harvested("X.00", path=p)
            if got != want:
                fails.append(f"pilot_harvested: {label} -> want {want!r}, "
                             f"got {got!r}")
        # The mirrors: an artifact declaration is neither of the other states.
        p = Path(d) / "ph_x.py"
        p.write_text(f"_PILOT_ARTIFACT = {str(art)!r}\n")
        if pilot_owed("X.00", path=p) is not None:
            fails.append("pilot_owed: `_PILOT_ARTIFACT` is not a reason — an "
                         "artifact path says nothing about which unit of work "
                         "is owed")
        if pilot_blocked("X.00", path=p) is not None:
            fails.append("pilot_blocked: `_PILOT_ARTIFACT` is not a block")
        if pilot_harvested("X.00", path=Path(d) / "gone.py") is not None:
            fails.append("pilot_harvested: a missing spec file declares "
                         "nothing")
    if pilot_harvested("NO.SUCH.SPEC") is not None:
        fails.append("pilot_harvested: an unimplemented spec has no file and "
                     "no declaration -> None")
    return fails


def _class_advice(ids, void, harv, owed, fill, heldc, blk, artifacts,
                  held_why=None) -> str:
    """The advice tail for one cost-class row — what would put a FRESH dispatch
    in it — or `""` when the row already holds one.

    The gate is ZERO-FRESH, not EMPTY (54th audit B5, 2026-08-31). The inline
    version gated on `if ids:`, so a class whose only occupant was VOID printed
    as served: on the day of the audit `cpu<2h` showed one row, `BA.02` (VOID),
    while the computed-and-discarded advice held six implementable specs —
    `LG.02`, `LT.01`, `ME.11.D`, `ME.11.F`, `T3.09`, `UB.14` — two of them
    *Episodic retrieval* champion-arena members. The headline line above the
    table said "only 0 is a FRESH dispatch" honestly; the per-class lines, the
    part actually read when choosing work, hid the repair. Six specs would have
    become visible on 09-01 not because anything was learned but because D8's
    default parking BA.02 emptied the row.

    PILOT-HARVESTABLE IS NAMED FIRST, ahead of PILOT-OWED, because it is
    cheaper still: the CPU run is already spent and its artifact is on disk.
    PILOT-OWED then, because it is the cheapest of the rest and the one this
    readout used to hide behind "NOT FILLABLE".
    """
    if ids and not all(i in void for i in ids):
        return ""
    prefix = "" if not ids else " (no FRESH dispatch here)"
    if harv:
        return (f"  {prefix} <- PILOT ALREADY RAN, HARVEST IT (cheapest repair "
                "of all): "
                + ", ".join(f"{s} -> {artifacts[s]}" for s in harv))
    if owed:
        return (f"  {prefix} <- PILOT OWED (cheapest repair): {', '.join(owed)}"
                + (f"; or implement {', '.join(fill)}" if fill else ""))
    if fill or heldc:
        parts = []
        if fill:
            parts.append(f"fillable today: {', '.join(fill)}")
        if heldc:
            # A held spec is named WITH its decision and date, never as work:
            # printing the bare id is how "fillable today: HR.1" earned two
            # journal hand-warnings in one day.
            hw = held_why or {}
            parts.append("HELD by an open decision (implement NOTHING here): "
                         + ", ".join(f"{s} <- {hw.get(s, 'UNKNOWN DECISION')}"
                                     for s in heldc))
        return f"  {prefix} <- " + "; ".join(parts)
    if blk:
        return (f"  {prefix} <- NOT FILLABLE: pilot BLOCKED on evidence "
                f"({', '.join(blk)}); the repair is a REDESIGN")
    return f"  {prefix} <- NOT FILLABLE: nothing to implement, nothing to pilot"


def _class_advice_fixture() -> List[str]:
    """Known-answer battery for `_class_advice`, the per-class advice tail.

    The known positive is the exact shape the 54th audit caught: a class whose
    only occupant is VOID and whose fillable list is non-empty MUST advertise
    the fill — the pre-B5 gate (`if ids:`) returns `""` there, so this battery
    is red against it. The converse guards the other direction: a row holding a
    fresh dispatch must stay tail-free, or every served class grows noise.
    Discovered and run by T0.21's P12 like every `_*_fixture` here.
    """
    fails: List[str] = []
    cases = [
        # (label, ids, void, harv, owed, fill, held, blk, must_contain)
        ("only-VOID occupant still advertises fill",
         ["BA.X"], ["BA.X"], [], [], ["LT.Y", "ME.Z"], [], [], "fillable today: LT.Y, ME.Z"),
        ("fresh occupant -> no tail",
         ["OK.1", "BA.X"], ["BA.X"], [], [], ["LT.Y"], [], [], None),
        ("empty class advertises fill",
         [], [], [], [], ["LT.Y"], [], [], "fillable today: LT.Y"),
        ("harvest outranks fill on an only-VOID row",
         ["BA.X"], ["BA.X"], ["SM.P"], [], ["LT.Y"], [], [], "HARVEST IT"),
        ("only-VOID, nothing anywhere -> NOT FILLABLE, honestly",
         ["BA.X"], ["BA.X"], [], [], [], [], [], "NOT FILLABLE: nothing"),
        ("empty + blocked -> REDESIGN",
         [], [], [], [], [], [], ["DP.B"], "the repair is a REDESIGN"),
        # THE HELD STATE — the D19/HR.1 row. The exact pre-repair output was
        # "fillable today: HR.1", printed against a standing Review order, so
        # the known positive is that a held spec appears WITH its decision and
        # never inside "fillable today".
        ("held-only class names the decision, not work",
         [], [], [], [], [], ["HR.A"], [],
         "HELD by an open decision (implement NOTHING here): HR.A <- D99 (decide_by 2026-09-14)"),
        ("held must not read as fillable",
         [], [], [], [], ["LT.Y"], ["HR.A"], [], "fillable today: LT.Y; HELD"),
        ("held outranks blocked: the date is the cheaper fact",
         [], [], [], [], [], ["HR.A"], ["DP.B"], "HELD by an open decision"),
    ]
    for label, ids, void, harv, owed, fill, heldc, blk, want in cases:
        got = _class_advice(ids, void, harv, owed, fill, heldc, blk,
                            {"SM.P": "/a.json"},
                            held_why={"HR.A": "D99 (decide_by 2026-09-14)"})
        if want is None:
            if got != "":
                fails.append(f"_class_advice: {label} -> want '', got {got!r}")
        elif want not in got:
            fails.append(f"_class_advice: {label} -> want {want!r} in {got!r}")
    # A held spec must never leak into the "fillable today:" clause itself.
    got = _class_advice([], [], [], [], [], ["HR.A"], [], {},
                        held_why={"HR.A": "D99 (decide_by 2026-09-14)"})
    if "fillable today" in got:
        fails.append(f"_class_advice: a held-only class must not say "
                     f"'fillable today', got {got!r}")
    # An occupied-but-stale row must SAY it holds no fresh dispatch, so the
    # reader cannot mistake the advice for a contradiction of the shown id.
    got = _class_advice(["BA.X"], ["BA.X"], [], [], ["LT.Y"], [], [], {})
    if "no FRESH dispatch here" not in got:
        fails.append(f"_class_advice: only-VOID tail must name itself, got {got!r}")
    return fails


def _void_foreclosed_fixture() -> List[str]:
    """Known-answer battery for `protocol.void_foreclosed`, the READER.

    Written the same day as the reader, per the lesson `_pilot_blocked_fixture`
    records: `_queue_fixture` stubs this function to test the CLAUSE, so it is
    green against a reader that always answers `None` and against one that
    answers for everything. Both mutations are live risks here — the first
    re-opens the misroute (a foreclosed VOID advertised as an arm to repair,
    which is what sent 3.99 CPU-hours of BA.03 into the queue as repairable),
    and the second is worse, because it would let any VOID go quiet without
    evidence. That is a park nobody voted for.

    The declaration is a DOCSTRING line, not a module constant, and the battery
    is where that choice gets its teeth: the whole point is that a VOID spec's
    file carries a certificate, so the declaration must live somewhere
    `run amend --doc-only` can re-stamp. A row below asserts the code idiom does
    NOT work, so a future author who reaches for `_VOID_FORECLOSED = "..."` out
    of habit gets `None` rather than a silently staled ledger row.
    """
    import tempfile
    from .protocol import void_foreclosed, void_foreclosed_refusal

    R = "the blind twin holds 98.9% of the horizon"
    # Every ACCEPTED declaration carries its price (54th audit B3): the
    # multiplier arithmetic and the welded downstream set. `PRICE` is the
    # minimal complete form, appended to every case that must still parse.
    PRICE = ("\n\nFORECLOSURE ARITHMETIC: no multiplier converges\n"
             "\nBLAST RADIUS: none\n")
    cases = [
        ("no docstring at all", "X = 1\n", None),
        ("a docstring with no declaration", '"""Just prose."""\n', None),
        ("a declared, priced reason",
         f'"""Title.\n\nVOID-FORECLOSED: {R}{PRICE}"""\n', R),
        ("the declaration alone, priced",
         f'"""VOID-FORECLOSED: {R}{PRICE}"""\n', R),
        # Continuation by indent — how a reason with its evidence is actually
        # written. `T0.31`'s idiom, so `review_queue.parse` and this agree.
        ("indented continuation lines",
         '"""Title.\n\nVOID-FORECLOSED: the blind twin holds\n'
         f'    98.9% of the horizon{PRICE}"""\n',
         "the blind twin holds 98.9% of the horizon"),
        ("a blank line ends the declaration",
         f'"""VOID-FORECLOSED: {R}\n\n    not part of it{PRICE}"""\n', R),
        ("a line at the margin ends the declaration",
         f'"""VOID-FORECLOSED: {R}\nnot part of it{PRICE}"""\n', R),
        ("re-declared, last wins",
         f'"""VOID-FORECLOSED: old\n\nVOID-FORECLOSED: {R}{PRICE}"""\n', R),
        # THE QUIET DIRECTION. Anything that is not a real reason is not a
        # foreclosure: a spec may not go silent on a keyword.
        ("the bare keyword is not a reason", '"""VOID-FORECLOSED:"""\n', None),
        ("whitespace is not a reason", '"""VOID-FORECLOSED:    """\n', None),
        ("an indented declaration is prose, not a declaration",
         f'"""Title.\n\n    VOID-FORECLOSED: {R}{PRICE}"""\n', None),
        ("a mention inside prose is not a declaration",
         f'"""We should mark it VOID-FORECLOSED: {R}"""\n', None),
        # THE LC.07 PHANTOM (57th audit B1c). A sentence about ANOTHER spec
        # wraps so the keyword lands at this docstring's margin. The anchor
        # rule — first line, or preceded by a blank line — is what rejects it;
        # every genuine declaration stands as its own paragraph.
        ("a wrapped sentence is not a declaration",
         f'"""Title.\n\nThis is not a re-run of the screen (LC.03 is\n'
         f'VOID-FORECLOSED: {R}) and it does not route through it.'
         f'{PRICE}"""\n', None),
        # THE IDIOM ASSERTION. The code form is deliberately NOT read: it would
        # stale the very ledger row the declaration is about.
        ("the module-constant idiom does not declare",
         f'"""Title."""\n_VOID_FORECLOSED = {R!r}\n', None),
        ("a docstring on a function is not the module's",
         f'def f():\n    """VOID-FORECLOSED: {R}{PRICE}"""\n', None),
        ("a syntax error is not a declaration", "def (:\n", None),
        # THE PRICE IS MANDATORY (54th audit B3). Three unpriced declarations
        # in three days welded 10 downstream specs and journalled it as a
        # saving; an unpriced weld is REFUSED and closes no door.
        ("an unpriced declaration is REFUSED",
         f'"""VOID-FORECLOSED: {R}"""\n', None),
        ("missing BLAST RADIUS alone refuses",
         f'"""VOID-FORECLOSED: {R}\n\nFORECLOSURE ARITHMETIC: none '
         'converges\n"""\n', None),
        ("missing FORECLOSURE ARITHMETIC alone refuses",
         f'"""VOID-FORECLOSED: {R}\n\nBLAST RADIUS: none\n"""\n', None),
        ("a bare companion keyword is not a price",
         f'"""VOID-FORECLOSED: {R}\n\nFORECLOSURE ARITHMETIC:\n'
         '\nBLAST RADIUS: none\n"""\n', None),
        ("an INDENTED companion is prose and does not price",
         f'"""VOID-FORECLOSED: {R}\n\n    FORECLOSURE ARITHMETIC: x\n'
         '\nBLAST RADIUS: none\n"""\n', None),
    ]
    # `void_foreclosed_refusal`, the LOUD half: (label, src, must-name,
    # must-NOT-name). `None` for must-name means "no refusal at all" — a spec
    # that never declared, or declared completely, has nothing to be loud
    # about, and a stray companion block welds nothing.
    refusal_cases = [
        ("no declaration -> no refusal", '"""Just prose."""\n', None, ()),
        ("a stray companion alone is not a refusal",
         '"""BLAST RADIUS: none\n"""\n', None, ()),
        ("a priced declaration -> no refusal",
         f'"""VOID-FORECLOSED: {R}{PRICE}"""\n', None, ()),
        ("unpriced -> refusal names BOTH blocks",
         f'"""VOID-FORECLOSED: {R}"""\n',
         ("FORECLOSURE ARITHMETIC:", "BLAST RADIUS:"), ()),
        # LC.07's literal shape: wrapped mid-sentence AND unpriced. The anchor
        # rule says it never declared, so there is nothing to refuse — the
        # repair for a phantom is a reflow, not a price.
        ("a wrapped unpriced sentence is not a refusal either",
         f'"""Title.\n\nprose about another spec (LC.03 is\n'
         f'VOID-FORECLOSED: {R}) continuing the thought."""\n', None, ()),
        ("missing blast alone -> refusal names IT and not the other",
         f'"""VOID-FORECLOSED: {R}\n\nFORECLOSURE ARITHMETIC: none '
         'converges\n"""\n',
         ("BLAST RADIUS:",), ("FORECLOSURE ARITHMETIC:",)),
    ]
    fails = []
    with tempfile.TemporaryDirectory() as d:
        for i, (label, src, want) in enumerate(cases):
            p = Path(d) / f"vf_{i}.py"
            p.write_text(src)
            got = void_foreclosed("X.00", path=p)
            if got != want:
                fails.append(f"void_foreclosed: {label} -> want {want!r}, "
                             f"got {got!r}")
        for i, (label, src, name, not_name) in enumerate(refusal_cases):
            p = Path(d) / f"vfr_{i}.py"
            p.write_text(src)
            got = void_foreclosed_refusal("X.00", path=p)
            if name is None:
                if got is not None:
                    fails.append(f"void_foreclosed_refusal: {label} -> want "
                                 f"None, got {got!r}")
            elif got is None:
                fails.append(f"void_foreclosed_refusal: {label} -> want a "
                             f"refusal, got None")
            else:
                for kw in name:
                    if f"`{kw}`" not in got:
                        fails.append(f"void_foreclosed_refusal: {label} must "
                                     f"name `{kw}`, got {got!r}")
                for kw in not_name:
                    if f"`{kw}`" in got:
                        fails.append(f"void_foreclosed_refusal: {label} must "
                                     f"NOT name `{kw}`, got {got!r}")
        if void_foreclosed("X.00", path=Path(d) / "gone.py") is not None:
            fails.append("void_foreclosed: a missing file declares nothing")
        if void_foreclosed_refusal("X.00", path=Path(d) / "gone.py") is not None:
            fails.append("void_foreclosed_refusal: a missing file refuses "
                         "nothing")
    if void_foreclosed("NO.SUCH.SPEC") is not None:
        fails.append("void_foreclosed: an unimplemented spec has no file and "
                     "no declaration -> None")
    # THE LIVE FILE this instrument was built for, read end to end — the
    # `_gates_frozen_fixture` precedent. `BA.03` VOIDed on 2026-08-31 with six
    # of seven rig conjuncts green; if its declaration ever stops parsing, the
    # readout goes back to calling a four-hour foreclosed run a cheap repair.
    from .protocol import module_path_for
    if module_path_for("BA.03") and not void_foreclosed("BA.03"):
        fails.append("void_foreclosed: BA.03 uses the VOID-FORECLOSED idiom "
                     "in the tree and the reader read it as 'does not "
                     "declare' — the state this reader exists for, unread")
    return fails


def _pilot_owed_fixture() -> List[str]:
    """Known-answer battery for `protocol.pilot_owed`, the READER — written on
    the same day as the reader, per the lesson `_pilot_blocked_fixture` records.

    The two readers now share `_declared_reason`, so this battery is not
    redundant with that one: it is what proves the SHARING did not silently
    alias the two flags. The case that matters most is the cross-flag pair — a
    file declaring only `_PILOT_BLOCKED` must read `None` here, and vice versa,
    because one reader answering for the other's flag would collapse the whole
    four-state split back into the default this iteration removed.
    """
    import tempfile
    from .protocol import module_path_for, pilot_blocked, pilot_owed

    R = "will freeze CEM_K_FIT/N_EVAL against the seed-90 artifact"
    cases = [
        ("no declaration at all", "X = 1\n", None),
        ("a declared reason", f"_PILOT_OWED = {R!r}\n", R),
        ("implicitly concatenated literal",
         "_PILOT_OWED = (\n 'will freeze CEM_K_FIT/N_EVAL '\n"
         " 'against the seed-90 artifact'\n)\n", R),
        ("annotated assignment counts", f"_PILOT_OWED: str = {R!r}\n", R),
        ("re-assigned, last wins",
         f"_PILOT_OWED = 'old'\n_PILOT_OWED = {R!r}\n", R),
        # The quiet direction: anything unreadable is NOT a claim that a pilot
        # can still succeed. Reading one would restore the optimistic default.
        ("None is not a declaration", "_PILOT_OWED = None\n", None),
        ("empty string is not a declaration", "_PILOT_OWED = ''\n", None),
        ("whitespace is not a declaration", "_PILOT_OWED = '   '\n", None),
        ("True is not a reason", "_PILOT_OWED = True\n", None),
        ("non-literal value is not a reason",
         "import os\n_PILOT_OWED = os.environ.get('WHY')\n", None),
        ("function-local assignment is not a declaration",
         f"def f():\n    _PILOT_OWED = {R!r}\n", None),
        ("a syntax error is not a declaration", "def (:\n", None),
        # THE ALIASING CASE, and the reason this battery exists at all.
        ("the other flag is not this one",
         "_PILOT_BLOCKED = 'a run measured the precondition fails'\n", None),
    ]
    fails = []
    with tempfile.TemporaryDirectory() as d:
        for i, (label, src, want) in enumerate(cases):
            p = Path(d) / f"po_{i}.py"
            p.write_text(src)
            got = pilot_owed("X.00", path=p)
            if got != want:
                fails.append(f"pilot_owed: {label} -> want {want!r}, "
                             f"got {got!r}")
        # The mirror of the aliasing case, on the other reader.
        p = Path(d) / "po_x.py"
        p.write_text(f"_PILOT_OWED = {R!r}\n")
        if pilot_blocked("X.00", path=p) is not None:
            fails.append("pilot_blocked: `_PILOT_OWED` is not a block — the "
                         "two readers share `_declared_reason` and must not "
                         "answer for each other's flag")
        if pilot_owed("X.00", path=Path(d) / "gone.py") is not None:
            fails.append("pilot_owed: a missing file is not a declaration")
    if pilot_owed("NO.SUCH.SPEC") is not None:
        fails.append("pilot_owed: an unimplemented spec has no file and no "
                     "declaration -> None")
    # THE STANDING INVARIANT, and the one that keeps the red honest: every
    # gate-provisional spec in the live tree declares exactly one of the two.
    # Asserted here rather than only at the exit code so the failure names the
    # spec, and asserted as "exactly one" so the contradiction case is covered
    # on real files too.
    #
    # PARKED specs are exempt, and the exemption is the point rather than a
    # loophole. A park RETIRES the question — the spec stops being coverage and
    # `queue_depth` drops it before the gate-provisional test is ever reached —
    # so a pilot state is moot and demanding one would flag a fact no instrument
    # reads. Written after this battery caught `T3.10` and `SM.02`, both of
    # which are parked with a registry marker naming the same both-fail branch
    # `_PILOT_BLOCKED` exists to record. The exemption is narrow BECAUSE it
    # keys on the machine-readable marker: `T2.11` carries "PARKED" in its
    # docstring banner and no marker, so it is not exempt — which is right, and
    # is how its missing declaration was found in the first place.
    from .protocol import gates_frozen
    from .registry import BY_ID as _BY_ID
    _parked = set(parked(_BY_ID)[0])
    _examined = []
    for sid in _BY_ID:
        path = module_path_for(sid)
        if (sid in _parked or not path
                or gates_frozen(sid, path=path) is not False):
            continue
        _examined.append(sid)
        b, o = pilot_blocked(sid, path=path), pilot_owed(sid, path=path)
        if bool(b) == bool(o):
            fails.append(
                f"pilot state: {sid} is gate-provisional and declares "
                + ("BOTH `_PILOT_OWED` and `_PILOT_BLOCKED`, which is a "
                   "contradiction" if b else
                   "NEITHER `_PILOT_OWED` nor `_PILOT_BLOCKED` — the absence "
                   "used to default to the cheap repair"))
    # AND THE LOOP MUST HAVE LOOKED AT SOMETHING. A scan whose filter widens to
    # everything reports zero violations and reads identically to a clean tree —
    # mutation, same hour: replacing the park test with `True` left this battery
    # green while checking nothing at all. Five specs are gate-provisional and
    # unparked today; the floor is 1 rather than 5 so that legitimately freezing
    # or parking them is not a red, while deleting the scan is.
    if not _examined:
        fails.append("pilot state: the live scan examined ZERO specs — either "
                     "no spec is gate-provisional (say so by removing this "
                     "check) or its filter has widened to exempt everything, "
                     "which reports clean by checking nothing")
    if set(_examined) & _parked:
        fails.append(f"pilot state: a PARKED spec must be exempt, not "
                     f"examined: {sorted(set(_examined) & _parked)}")
    return fails


def _claim_dead_fixture() -> List[str]:
    """Known-answer battery for the FORECLOSED reachability state (58th audit
    B1) — the CLAIM-DEAD ratchet must see a foreclosure as the retirement it
    is, not as the strongest state short of PASS.

    THE SCAR: on 2026-09-01 `check()` printed `shelter/building … claims:
    SH.02 RUNNABLE` in its commitment table while its own queue-depth section,
    forty lines down, said of the same spec "Do NOT spend seeds on these; the
    repair is a redesign". Five commitments — balance, smell,
    shelter/building, thermal (kills), fast/slow, four of them the owner's own
    2026-08-09 survival directives — had zero passing claims and nothing
    anybody was allowed to run, and this file reported `0 CLAIM-DEAD`, rc=0.
    `VOID-FORECLOSED` and `PILOT-BLOCKED` are retirements that do not spell
    `PARKED:`, so each one laundered a park past the 28th audit's repair.

    Stubbing idiom of `_queue_depth_fixture`: the protocol readers are
    monkeypatched so the CLAUSE is under test, not the parsers (those have
    their own batteries). Both flavours of foreclosure are planted — BA.03's
    shape (VOID + accepted declaration) and SH.02's shape (gate-provisional +
    measured pilot-block) — plus the two directions that must stay ALIVE, per
    the liveness-watch lesson (a monitor that cannot express success cannot
    express failure either).
    """
    from dataclasses import replace

    from .registry import BY_ID
    donor = BY_ID["T0.01"]
    reg = {
        # BA.03's shape: ran, VOIDed, priced declaration accepted.
        "Q.20": replace(donor, id="Q.20", title="Balance is load-bearing",
                        notes="COVERS: balance (claim)"),
        # SH.02's shape: gate-provisional, pilot measured its own
        # precondition failing. Never ran, so no ledger row at all.
        "Q.21": replace(donor, id="Q.21", title="Shelter beats the null",
                        notes="COVERS: shelter/building (claim)"),
        # The audit's exact case: one FORECLOSED claim PLUS one PARKED claim
        # on the same commitment must read CLAIM-DEAD.
        "Q.22": replace(donor, id="Q.22", title="Shelter, first attempt",
                        notes="COVERS: shelter/building (claim). "
                              "PARKED: 2026-08-01 — concluded by its own "
                              "fork."),
        # A live, un-foreclosed claim keeps its commitment alive.
        "Q.23": replace(donor, id="Q.23", title="Hearing binds",
                        notes="COVERS: hearing (claim)"),
        # A PASS is a demonstration; a later foreclosure cannot un-record it.
        "Q.24": replace(donor, id="Q.24", title="Touch grips",
                        notes="COVERS: touch/contact (claim)"),
    }
    results = {"Q.20": {"status": "VOID"}, "Q.24": {"status": "PASS"}}
    foreclosed = {"Q.20": "the blind twin holds 98.9% of the horizon",
                  "Q.24": "stale declaration on a demonstrated spec"}
    blocked = {"Q.21": "the null already holds the roof it was placed under"}
    frozen = {"Q.21": False}

    from . import protocol as _proto
    real_vf, real_gf = _proto.void_foreclosed, _proto.gates_frozen
    real_pb, real_po = _proto.pilot_blocked, _proto.pilot_owed
    real_mpf = _proto.module_path_for
    _proto.void_foreclosed = lambda sid, path=None: foreclosed.get(sid)
    _proto.gates_frozen = lambda sid, path=None: frozen.get(sid)
    _proto.pilot_blocked = lambda sid, path=None: blocked.get(sid)
    _proto.pilot_owed = lambda sid, path=None: None
    _proto.module_path_for = lambda sid, strict=False: f"/x/{sid}.py"
    try:
        rows = report(reg, results)
    finally:
        _proto.void_foreclosed, _proto.gates_frozen = real_vf, real_gf
        _proto.pilot_blocked, _proto.pilot_owed = real_pb, real_po
        _proto.module_path_for = real_mpf

    by_name = {r["commitment"]: r for r in rows}
    bal, shel = by_name["balance"], by_name["shelter/building"]
    hear, touch = by_name["hearing"], by_name["touch/contact"]
    reach = claim_reachability(rows)

    fails = []
    if not _claim_dead(bal):
        fails.append("claim-dead: a commitment whose only claim is "
                     "VOID-FORECLOSED must read CLAIM-DEAD (BA.03's shape) — "
                     "a foreclosure is a retirement that does not spell "
                     "PARKED:")
    if not _claim_dead(shel):
        fails.append("claim-dead: one PILOT-BLOCKED claim plus one PARKED "
                     "claim must read CLAIM-DEAD (SH.02's shape, the 58th "
                     "audit's exact case)")
    if _claim_dead(hear):
        fails.append("claim-dead: a live, un-foreclosed claim keeps its "
                     "commitment ALIVE — the healthy state must be "
                     "recognisable or the sick one means nothing")
    if _claim_dead(touch):
        fails.append("claim-dead: a PASS is a demonstration; a foreclosure "
                     "cannot un-record it")
    if ("Q.20", "FORECLOSED") not in reach["balance"]:
        fails.append(f"reachability: a VOID-FORECLOSED claim must read "
                     f"FORECLOSED, never RUNNABLE — got {reach['balance']}")
    if ("Q.21", "FORECLOSED") not in reach["shelter/building"]:
        fails.append(f"reachability: a PILOT-BLOCKED claim must read "
                     f"FORECLOSED, never RUNNABLE — got "
                     f"{reach['shelter/building']}")
    if ("Q.23", "RUNNABLE") not in reach["hearing"]:
        fails.append(f"reachability: a live claim with no blockers must stay "
                     f"RUNNABLE — got {reach['hearing']}")
    return fails


def _welded_fixture() -> List[str]:
    """Known-answer battery for the `welded<-ROOTS` reachability state (59th
    audit B1) — the retirement predicate must be asked about a spec's
    BLOCKERS, not only about the spec itself.

    THE SCAR: `LC.03` is VOID-FORECLOSED — it resolves never — and for nine
    days `claim_reachability` emitted `blocked<-LC.03` for `DP.02`, the same
    string it emits for a spec waiting on a job that finishes tonight, while
    its own docstring stated the distinction it was failing to draw. Ten specs
    sat in that state and no instrument could utter it: a retirement predicate
    that is not applied transitively launders itself across exactly one
    dependency edge.

    Stubbing idiom of `_claim_dead_fixture`: protocol readers monkeypatched,
    graph and root statuses injected, so the CLAUSE is under test. Planted
    shapes: the audit's exact case (a claim behind a foreclosed root —
    `DP.02<-LC.03`), the direction that must stay ALIVE (a claim behind a
    FAIL root — `BO.01<-DP.05`), a mixed root set (one live root keeps
    `blocked<-`), and the PARKED-root flavour through the shared predicate.
    Per B1.3 a welded claim must NOT tip its commitment CLAIM-DEAD — that
    question is the Review's (09-06), and this battery pins the predicate to
    supplying its input, not its answer.
    """
    from dataclasses import replace

    from .registry import BY_ID
    donor = BY_ID["T0.01"]
    reg = {
        # The audit's exact case: a claim whose ONLY root is foreclosed.
        "Q.30": replace(donor, id="Q.30", title="Habit vs deliberation",
                        notes="COVERS: fast/slow (claim)"),
        # The direction that must stay alive: a FAIL root is a queue
        # position — blocked resolves when the blocker does.
        "Q.31": replace(donor, id="Q.31", title="Brain organisation race",
                        notes="COVERS: fast/slow (claim)"),
        # One live root in the set keeps the whole set blocked<-.
        "Q.32": replace(donor, id="Q.32", title="Trunk lesion dissociates",
                        notes="COVERS: fast/slow (claim)"),
    }
    terminal = {"Q.30": {"Q.90"}, "Q.31": {"Q.91"},
                "Q.32": {"Q.90", "Q.91"}}
    root_status = {"Q.90": "VOID", "Q.91": "FAIL"}
    foreclosed = {"Q.90": "fork (ii) fired; repair is upstream redesign"}

    from . import protocol as _proto
    real_vf, real_gf = _proto.void_foreclosed, _proto.gates_frozen
    real_pb, real_po = _proto.pilot_blocked, _proto.pilot_owed
    real_mpf = _proto.module_path_for
    _proto.void_foreclosed = lambda sid, path=None: foreclosed.get(sid)
    _proto.gates_frozen = lambda sid, path=None: None
    _proto.pilot_blocked = lambda sid, path=None: None
    _proto.pilot_owed = lambda sid, path=None: None
    _proto.module_path_for = lambda sid, strict=False: f"/x/{sid}.py"
    try:
        rows = report(reg, {})
        reach = claim_reachability(rows, terminal=terminal,
                                   root_status=root_status)
        rd = globals().get("root_dead")
        parked_flavour = (rd("Q.92", status=None,
                             parked_map={"Q.92": "2026-08-01 — spent fork"})
                          if rd else None)
        live_flavour = (rd("Q.91", status="FAIL", parked_map={})
                        if rd else "MISSING")
    finally:
        _proto.void_foreclosed, _proto.gates_frozen = real_vf, real_gf
        _proto.pilot_blocked, _proto.pilot_owed = real_pb, real_po
        _proto.module_path_for = real_mpf

    fs = {r["commitment"]: r for r in rows}["fast/slow"]
    states = dict(reach["fast/slow"])

    fails = []
    if states.get("Q.30") != "welded<-Q.90":
        fails.append(f"welded: a claim whose every terminal blocker is "
                     f"foreclosed must read welded<-, never blocked<- "
                     f"(DP.02<-LC.03, the 59th audit's case) — got "
                     f"{states.get('Q.30')!r}")
    if states.get("Q.31") != "blocked<-Q.91":
        fails.append(f"welded: a FAIL root is a LIVE root — blocked<- must "
                     f"survive for it (BO.01<-DP.05) or every commitment "
                     f"behind T2.01 reads dead — got {states.get('Q.31')!r}")
    if states.get("Q.32") != "blocked<-Q.90,Q.91":
        fails.append(f"welded: one live root in the set keeps blocked<- — "
                     f"got {states.get('Q.32')!r}")
    if rd is None:
        fails.append("welded: shared predicate coverage.root_dead is missing "
                     "— the two readers (claim_reachability, run blocked) "
                     "have nothing to share and will drift")
    elif parked_flavour != "PARKED":
        fails.append(f"welded: a PARKED root is a dead root (ME.6<-T2.11's "
                     f"flavour) — root_dead returned {parked_flavour!r}")
    elif live_flavour is not None:
        fails.append(f"welded: root_dead must fail ALIVE (None) on a FAIL "
                     f"root — got {live_flavour!r}")
    if _claim_dead(fs):
        fails.append("welded: a welded claim must NOT tip its commitment "
                     "CLAIM-DEAD — B1 supplies the 09-06 question's input, "
                     "not its answer")

    # 59th audit B2 — the citation flavour. `coverage` printed "0 dangling"
    # while GOAL.md:242 said LC.04 "is already testing" of a spec welded
    # behind a VOID-FORECLOSED root: an id that resolves to a corpse is a
    # worse dangling reference than one that resolves to nothing, because the
    # nothing-case is the one every checker is built to catch.
    cite_by_id = {"LC.94": donor, "Q.30": donor, "Q.31": donor}
    gc = goal_citations(
        text="LC.94 is already testing; Q.30 and Q.31 run beside it.",
        by_id=cite_by_id, baseline=frozenset(),
        unrunnable_baseline=frozenset({"LC.94", "Q.31"}),
        state_of={"LC.94": "welded<-Q.90", "Q.30": "PARKED"}.get)
    if gc.get("unrunnable_known") != ["LC.94"]:
        fails.append(f"cited-but-unrunnable: a baselined unrunnable citation "
                     f"must read KNOWN, not red — got "
                     f"{gc.get('unrunnable_known')!r}")
    if gc.get("unrunnable_new") != ["Q.30"]:
        fails.append(f"cited-but-unrunnable: a NEW citation of a parked/"
                     f"foreclosed/welded spec is the red class — got "
                     f"{gc.get('unrunnable_new')!r}")
    if gc.get("unrunnable_stale_baseline") != ["Q.31"]:
        fails.append(f"cited-but-unrunnable: a baseline entry that is live "
                     f"again must demand its own removal (shrink-only) — got "
                     f"{gc.get('unrunnable_stale_baseline')!r}")
    return fails


def counts() -> tuple[int, int]:
    """(n_claim_dead, n_malformed) — two different fires, separately
    assertable.

    Claim-dead means no passing claim and no claim-kind declaration that could
    still produce evidence — the original zero-declared-specs case, since the
    28th audit the case where every claim spec is PARKED, and since the 58th
    the case where every survivor is FORECLOSED (`VOID-FORECLOSED` /
    `PILOT-BLOCKED`): all invisible to every other instrument, and the reason
    this module exists. Malformed means a `COVERS:` naming a
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
                "CLAIM-DEAD (every claim spec parked or foreclosed)"
                if _claim_dead(r) else
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
            for sid, st in sorted(reach[r["commitment"]]):
                if st == "FORECLOSED":
                    print(f"  {'':{width}}    {sid} FORECLOSED "
                          f"({r.get('foreclosed', {}).get(sid, '')} — the "
                          f"repair is a redesign, never a dispatch)")
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
          f"parked or foreclosed),\n  {len(unproven)} with live claim specs "
          f"but nothing passing.")
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
    if gc["unrunnable"]:
        print(f"  CITED-BUT-UNRUNNABLE: "
              + ", ".join(f"{i} ({s})"
                          for i, s in sorted(gc["unrunnable"].items()))
              + "\n  — each id RESOLVES, so the dangling count above cannot "
              "see it, and each is\n  parked, foreclosed or welded, so the "
              "citation's present tense is false. An id\n  that resolves to "
              "a corpse is a worse dangling reference than one that\n  "
              "resolves to nothing (59th audit B2).")
    if gc["unrunnable_new"]:
        print(f"  {len(gc['unrunnable_new'])} NEW unrunnable citation(s) — "
              f"fix GOAL.md's text or route the revival;\n  never add to "
              f"GOAL_UNRUNNABLE_BASELINE (shrink-only): "
              f"{', '.join(gc['unrunnable_new'])}")
    if gc["unrunnable_stale_baseline"]:
        print(f"  {len(gc['unrunnable_stale_baseline'])} unrunnable-baseline "
              f"entr(y/ies) are LIVE again and must be removed "
              f"from GOAL_UNRUNNABLE_BASELINE: "
              f"{', '.join(gc['unrunnable_stale_baseline'])}")
    print("\n  A nomination is NOT coverage. It is a spec whose title looks\n"
          "  related and whose author has not said so; only `COVERS:` counts.\n"
          "  A PARKED spec is NOT coverage either: a retirement is not a\n"
          "  falsifiable claim, however honest the retiring was.")

    u = unreachable_ratchet()
    fu = fail_unowned_ratchet()
    qf = (_queue_fixture() + _gates_frozen_fixture()
          + _pilot_blocked_fixture() + _pilot_owed_fixture()
          + _pilot_harvested_fixture() + _void_foreclosed_fixture()
          + _claim_dead_fixture() + _welded_fixture() + _exit_code_fixture()
          + _unreachable_fixture() + _park_release_fixture()
          + _fail_unowned_fixture() + u["refused"] + fu["refused"])
    q = queue_depth()
    print(f"\n  QUEUE DEPTH — dispatchable TODAY (runnable, implemented, "
          f"tracked, unparked, unsettled): {q['depth']}"
          + (f", of which {len(q['void'])} VOID" if q["void"] else "")
          + (f", {len(q['unaffordable'])} UNAFFORDABLE TODAY"
             if q["unaffordable"] else "")
          + (f" -> only {q['fresh']} is a FRESH dispatch"
             if q["void"] or q["unaffordable"] else ""))
    for cls, ids in q["by_class"].items():
        if ids or cls in q["empty"]:
            shown = (", ".join(
                sid + (f" !UNAFFORDABLE TODAY ({q['unaffordable'][sid]})"
                       if sid in q["unaffordable"] else "")
                for sid in ids) if ids else "EMPTY")
            tail = _class_advice(ids, q["void"],
                                 q["pilot_harvestable"].get(cls, []),
                                 q["pilot_owed"].get(cls, []),
                                 q["fillable"].get(cls, []),
                                 q["fill_held"].get(cls, []),
                                 q["pilot_blocked"].get(cls, []),
                                 q["pilot_harvestable_artifact"],
                                 held_why=q["fill_held_why"])
            print(f"      {cls:<10} {len(ids):>2}   {shown}{tail}")
    if q["void"]:
        print(f"  of which VOID (an arm to repair, not a dispatch): "
              f"{', '.join(q['void'])}")
    if q["unaffordable"]:
        print(f"  of which UNAFFORDABLE TODAY — the day meter would refuse "
              f"the dispatch right now (its own\n      comparison, est > "
              f"remaining, against {q['cpu_remaining_s']:.0f}s left of the "
              f"day; NOT a verdict on the spec):")
        for sid, why in q["unaffordable"].items():
            print(f"      {sid}: {why}")
        print("  An [ENUM ...] estimate is a class enum typed at "
              "registration, not a measurement — SO.08 sat\n  foreclosed a "
              "full day at ~28,000x its measured cost (75th audit F1). "
              "Before waiting for\n  midnight, ask whether the row's repair "
              "is a SIZING RECORD and an honest re-declaration.")
    ex = q["excluded"]
    print(f"  excluded: {len(ex['unimplemented'])} unimplemented, "
          f"{len(ex['settled'])} settled, {len(ex['parked'])} parked, "
          f"{len(ex['untracked'])} UNTRACKED"
          + (f" ({', '.join(ex['untracked'])})" if ex["untracked"] else "")
          + f", {len(ex['gates_provisional'])} GATES-PROVISIONAL"
          + (f" ({', '.join(ex['gates_provisional'])})"
             if ex["gates_provisional"] else "")
          + f", {len(ex['void_foreclosed'])} VOID-FORECLOSED"
          + (f" ({', '.join(ex['void_foreclosed'])})"
             if ex["void_foreclosed"] else ""))
    if q["void_foreclosed_why"]:
        print(f"  {len(q['void_foreclosed_why'])} spec(s) are VOID-FORECLOSED "
              f"— they RAN, they VOIDed, and the recorded row says\n"
              "      the verdict cannot change at this envelope. Do NOT "
              "re-run; the repair is a redesign:")
        for sid, why in q["void_foreclosed_why"].items():
            print(f"      {sid}: {why}")
        print("  A VOID is two states wearing one word. The ledger says 'run "
              "did not test the claim'\n"
              "  for both the arm that can be fixed and the world that "
              "forecloses the measurement,\n"
              "  and this line used to call both an arm to repair (BA.03, "
              "2026-08-31: 3.99 CPU-hours,\n"
              "  six of seven rig conjuncts green, the blind twin at 98.9% of "
              "the horizon).")
    if q["void_foreclosed_refused"]:
        print(f"  !! {len(q['void_foreclosed_refused'])} spec(s) carry a "
              f"VOID-FORECLOSED declaration that was REFUSED (54th audit B3:\n"
              "      unpriced — no FORECLOSURE ARITHMETIC / BLAST RADIUS), "
              "collected for EVERY status\n      (57th audit B1: the LC.07 "
              "phantom sat on a spec that had never run). A VOID one\n"
              "      ranks as repairable above; the next unit on each is to "
              "REPAIR THE DECLARATION,\n      not to dispatch a re-run:")
        for sid, msg in q["void_foreclosed_refused"].items():
            print(f"      {sid}: {msg}")
    if ex["gates_provisional"]:
        print("  a gate-provisional spec is implemented shelf furniture: its\n"
              "  own run() refuses until a pilot freezes its bars. Run the\n"
              "  pilot and flip `_GATES_FROZEN`, or implement another spec.")
    if q["new_empty"]:
        print(f"  {len(q['new_empty'])} cost class(es) NEWLY EMPTY — nothing "
              f"can be dispatched at this cost:\n"
              f"      {', '.join(q['new_empty'])}\n"
              "  Free weekly quota at an empty class is unspendable however\n"
              "  awake the loop is: that is what cost 61 free GPU-hours over\n"
              "  three weeks. Pilot or implement a spec; never baseline the\n"
              "  class.")
    owed_cls = sorted(c for c in q["empty"] if q["pilot_owed"].get(c))
    if owed_cls:
        print(f"  {len(owed_cls)} empty class(es) are PILOT-OWED, the cheapest "
              f"of the three repairs:\n"
              + "".join(f"      {c:<10} {', '.join(q['pilot_owed'][c])}\n"
                        for c in owed_cls)
              + "  The spec is written, tracked and runnable and its own run()\n"
              "  refuses until a pilot freezes its bars. That is a bounded CPU\n"
              "  unit, NOT an unblock and NOT a new implementation — and it is\n"
              "  the state this readout used to report as NOT FILLABLE.")
    if q["pilot_harvestable_artifact"]:
        print(f"  {len(q['pilot_harvestable_artifact'])} spec(s) are "
              f"PILOT-HARVESTABLE — gate-provisional and still declaring the\n"
              "      pilot OWED, but the artifact they name EXISTS ON DISK. The "
              "run is already spent; the\n"
              "      next unit is to read it and either freeze the gates or "
              "declare `_PILOT_BLOCKED`:")
        for sid, art in q["pilot_harvestable_artifact"].items():
            print(f"      {sid}: {art}")
        print("  A pilot that completed and was never harvested is invisible "
              "to every other state here\n"
              "  (BA.03, 2026-08-30: eight hours, its own prose asserting the "
              "artifact did not exist).")
    if q["pilot_blocked_why"]:
        print(f"  {len(q['pilot_blocked_why'])} spec(s) are PILOT-BLOCKED — "
              f"gate-provisional, but a run has MEASURED that the\n"
              "      pilot's own precondition fails. Do NOT spend seeds on "
              "these; the repair is a redesign:")
        for sid, why in q["pilot_blocked_why"].items():
            print(f"      {sid}: {why}")
    if q["pilot_undeclared"]:
        print(f"  {len(q['pilot_undeclared'])} spec(s) are PILOT-UNDECLARED — "
              f"gate-provisional, declaring NEITHER `_PILOT_OWED`\n"
              f"      nor `_PILOT_BLOCKED` (or both, which is a "
              f"contradiction): {', '.join(q['pilot_undeclared'])}\n"
              "  This is the state that used to default to PILOT-OWED, and\n"
              "  the default sent a builder to spend a pilot that four specs\n"
              "  had already spent. An undeclared spec rescues no cost class\n"
              "  and is never named as a cheap repair. The repair is ONE LINE\n"
              "  in the spec, written by whoever knows: `_PILOT_OWED = \"what\n"
              "  the pilot will freeze\"` or `_PILOT_BLOCKED = \"what a run\n"
              "  measured\"`.")
    held_cls = sorted(c for c in q["empty"] if q["fill_held"].get(c))
    if held_cls:
        print(f"  {len(held_cls)} empty class(es) are FILL-HELD by an open "
              f"decision — an implementable spec exists\n"
              "      and an armed DECIDE block declares it blocked, so "
              "implementing it would waste the hour\n"
              "      or walk around the decision (D19/HR.1: the run IS the "
              "fetch the default forbids):\n"
              + "".join(f"      {c:<10} "
                        + ", ".join(f"{s} <- {q['fill_held_why'][s]}"
                                    for s in q["fill_held"][c]) + "\n"
                        for c in held_cls)
              + "  Not the builder's unit: the repair fires on the decision "
              "desk by decide_by, or the\n"
              "  owner answers early. Until then this class is honestly "
              "closed, and this line — not\n"
              "  a journal hand-warning — is what keeps an iteration from "
              "being routed at it.")
    if q["empty_unfillable"]:
        print(f"  {len(q['empty_unfillable'])} empty class(es) have NO path in "
              f"today — nothing runnable to implement\n"
              f"      and nothing gate-provisional to pilot: "
              f"{', '.join(q['empty_unfillable'])}\n"
              "  Do not spend an iteration looking for a spec to write here.\n"
              "  The repair is an UNBLOCK (`run blocked`), which is a\n"
              "  different unit of work — and the quota at this cost stays\n"
              "  unspendable until the ladder moves, however awake the loop.")
    if q["known_empty"]:
        print(f"  {len(q['known_empty'])} known-empty (baselined 2026-08-29): "
              f"{', '.join(q['known_empty'])} — implementing ONE spec in any "
              f"of\n      these clears it, and it must then leave "
              f"QUEUE_EMPTY_BASELINE.")
    if q["stale_baseline"]:
        print(f"  {len(q['stale_baseline'])} baselined class(es) are NO LONGER "
              f"empty and must be removed from QUEUE_EMPTY_BASELINE: "
              f"{', '.join(q['stale_baseline'])}")
    if u["count"] is not None:
        print(f"\n  UNREACHABLE (`run blocked`'s number, ratcheted — 58th "
              f"audit B3): {u['count']} of {u['ladder']} specs "
              f"({100.0 * u['count'] / u['ladder']:.0f}%), baseline "
              f"{u['baseline']}, shrink-only.")
    for m in u["grown"]:
        print(f"  !! {m}")
    for m in u["stale_baseline"]:
        print(f"  {m}")

    if fu["count"] is not None:
        print(f"\n  FAIL-UNOWNED (72nd audit B1): {fu['count']} settled "
              f"FAIL(s) with NO repair owner — no repaired_by, no "
              f"REVIEW_QUEUE mention,\n      no FAIL-DISPOSED marker — "
              f"baseline {fu['baseline']}, shrink-only"
              + (f": {', '.join(fu['unowned'])}" if fu["unowned"] else "."))
        if fu["disposed"]:
            print(f"      excluded by FAIL-DISPOSED disposition: "
                  + ", ".join(f"{s} <- {a} {d}"
                              for s, (a, d) in sorted(fu["disposed"].items())))
        if fu["unowned"]:
            print("  A FAIL is not disposed by being understood; it is "
                  "disposed by being ROUTED. The legal\n  repairs are a "
                  "REVIEW_QUEUE row with a DUE:, a declared repaired_by, or "
                  "an explicit\n  FAIL-DISPOSED registry disposition — never "
                  "deleting the row, never a re-run for a\n  better number, "
                  "never a baseline raise.")
        if fu["malformed"]:
            print(f"  !! {len(fu['malformed'])} MALFORMED FAIL-DISPOSED "
                  f"marker(s) — no authority+date, so each disposes of\n"
                  f"      nothing and its spec ranks unowned above: "
                  f"{', '.join(fu['malformed'])}")
    for m in fu["grown"]:
        print(f"  !! {m}")
    for m in fu["stale_baseline"]:
        print(f"  {m}")

    pr = park_release()
    pr_pairs = {f"{s}->{r}" for s, r, _st in pr["violations"]}
    pr_new = sorted(pr_pairs - PARK_RELEASE_BASELINE)
    pr_stale = sorted(PARK_RELEASE_BASELINE - pr_pairs)
    if pr["violations"]:
        print(f"\n  PARK-ON-AN-UNREACHABLE-RELEASE (62nd audit B3): "
              f"{len(pr['violations'])} park->release pair(s) whose stated "
              f"revival path cannot be walked today:")
        for s, r, st in pr["violations"]:
            print(f"      {s} -> {r} ({st})")
        print("  Each park was legal and evidence-backed; the defect is "
              "COMPOSITIONAL — no dispatch\n  anywhere revives the commitment "
              "behind these parks. The repair is an upstream\n  redesign or a "
              "Review re-parent, never deleting the PARKED marker.")
    if pr["undeclared"]:
        print(f"  {len(pr['undeclared'])} UNDECLARED-RELEASE park(s) — the "
              f"PARKED marker declares no `RELEASE:`, so any\n      reading "
              f"above for these is a parse of prose. The repair is one line "
              f"by whoever\n      parked it — `RELEASE: <spec id>` or "
              f"`RELEASE: NONE`: {', '.join(pr['undeclared'])}")
    if pr_new:
        print(f"  {len(pr_new)} NEW unreachable-release pair(s) — route the "
              f"redesign or re-parent the park;\n      never add to "
              f"PARK_RELEASE_BASELINE (shrink-only): {', '.join(pr_new)}")
    if pr_stale:
        print(f"  {len(pr_stale)} baseline pair(s) are WALKABLE again and "
              f"must be removed from PARK_RELEASE_BASELINE: "
              f"{', '.join(pr_stale)}")

    if qf:
        print(f"  {len(qf)} QUEUE-FIXTURE FAILURE(S) — the instrument is "
              f"wrong, so its number above is not evidence:")
        for f in qf:
            print(f"      {f}")

    # `pilot_undeclared` is RED, not amber, and the asymmetry is deliberate.
    # An undeclared pilot state is not a missing note: it is the instrument
    # naming a repair it cannot support, and on 2026-08-30 it named two — one
    # of which had already been copied into a handoff and would have been
    # obeyed. Amber would let that stand while the exit code stayed green.
    # It is cheap to clear (one line, by the author, at the moment they know),
    # and it is zero right now, so it never becomes background noise.
    return exit_code(
        red={"uncovered": uncovered, "claim_dead": dead,
             "new_dangling_citation": gc["new"], "new_empty_class": q["new_empty"],
             "new_unrunnable_citation": gc["unrunnable_new"],
             "queue_fixture_failure": qf,
             "unreachable_grew": u["grown"],
             "fail_unowned_grew": fu["grown"],
             "malformed_fail_disposed": fu["malformed"] or [],
             "pilot_undeclared": q["pilot_undeclared"],
             "new_park_release": pr_new,
             "park_release_undeclared": pr["undeclared"]},
        amber={"malformed_declaration": bad,
               "stale_citation_baseline": gc["stale_baseline"],
               "stale_unrunnable_baseline": gc["unrunnable_stale_baseline"],
               "stale_queue_baseline": q["stale_baseline"],
               "stale_unreachable_baseline": u["stale_baseline"],
               "stale_fail_unowned_baseline": fu["stale_baseline"],
               "stale_park_release_baseline": pr_stale})


# RED = the ladder is making a claim it cannot support, or an instrument that
# checks that is itself broken. AMBER = a bookkeeping fact that misleads nobody
# about a capability. Nothing else.
def exit_code(red: dict, amber: dict) -> int:
    """`2` if any red condition is non-empty, else `1` for amber, else `0`.

    Extracted from `check()` on 2026-08-30 because it was untestable where it
    sat. It was a bare conditional expression at the end of a 200-line printer,
    so **no fixture in this repo could assert that any red condition actually
    reaches the exit code** — verified by mutation the same hour: deleting the
    freshly-added `pilot_undeclared` term from that expression left every
    fixture GREEN. A ratchet whose wiring is unasserted is a ratchet that can be
    disconnected in a one-line edit and pass its own tests, which is the disease
    this module exists to catch, one level up.

    Taking DICTS rather than booleans so `_exit_code_fixture` can name the
    condition it is exercising, and so a future reader can enumerate what turns
    this repo's coverage red without reading the printer. Values are truthiness
    only — lists, counts and bools all work.
    """
    if any(red.values()):
        return 2
    return 1 if any(amber.values()) else 0


def _exit_code_fixture() -> List[str]:
    """Known-answer battery for `exit_code` — every red condition on its own.

    The point is the PER-CONDITION loop, not the three-line function: it asserts
    that each named red term is individually load-bearing, which is what the
    mutation showed nothing did. Deleting any single term from `check()`'s call
    now fails here by name.
    """
    RED = ["uncovered", "claim_dead", "new_dangling_citation",
           "new_empty_class", "new_unrunnable_citation",
           "queue_fixture_failure", "unreachable_grew",
           "fail_unowned_grew", "malformed_fail_disposed",
           "pilot_undeclared", "new_park_release",
           "park_release_undeclared"]
    AMBER = ["malformed_declaration", "stale_citation_baseline",
             "stale_unrunnable_baseline",
             "stale_queue_baseline", "stale_unreachable_baseline",
             "stale_fail_unowned_baseline",
             "stale_park_release_baseline"]
    fails = []
    clean_red = {k: [] for k in RED}
    clean_amber = {k: [] for k in AMBER}
    if exit_code(clean_red, clean_amber) != 0:
        fails.append("exit_code: no condition set must be 0")
    for k in RED:
        r = dict(clean_red, **{k: ["x"]})
        if exit_code(r, clean_amber) != 2:
            fails.append(f"exit_code: red `{k}` alone must exit 2 — a red "
                         f"condition that does not reach the exit code is a "
                         f"disconnected ratchet")
        # ...and red must dominate amber, never be averaged with it.
        if exit_code(r, {k2: ["y"] for k2 in AMBER}) != 2:
            fails.append(f"exit_code: red `{k}` must dominate amber")
    for k in AMBER:
        a = dict(clean_amber, **{k: ["x"]})
        if exit_code(clean_red, a) != 1:
            fails.append(f"exit_code: amber `{k}` alone must exit 1")
    # An amber condition may NEVER be silently promoted or demoted: if a future
    # edit moves one of these into `red`, this battery keeps its old meaning
    # visible rather than letting the change pass unremarked.
    if set(RED) & set(AMBER):
        fails.append("exit_code: a condition cannot be both red and amber")

    # THE WIRING, checked STATICALLY — and this half is the one that caught a
    # live hole. The battery above proves `exit_code` behaves; it says nothing
    # about whether `check()` still PASSES it anything real. Mutation, same
    # hour: replacing `"pilot_undeclared": q["pilot_undeclared"]` with a literal
    # `[]` at the call site left every fixture green, and so did doing it to
    # `new_empty_class` — the pre-existing red that has been this file's whole
    # point for two days. Two ratchets, both disconnectable in one line, both
    # passing their own tests.
    #
    # It has to be static because `check()` CALLS this fixture: a dynamic check
    # would recurse into itself. Reading the source is the honest alternative
    # and it is the idiom this repo already uses for `_GATES_FROZEN`.
    #
    # The assertion is deliberately weak in one direction and strong in the
    # other: it does not try to understand the expression, only that the term is
    # present and is not a CONSTANT. A constant is exactly how a term gets
    # disabled while looking wired, and it is the only way that has happened.
    import ast as _ast
    try:
        _tree = _ast.parse(Path(__file__).read_text())
    except (OSError, SyntaxError, ValueError) as exc:      # pragma: no cover
        return fails + [f"exit_code: could not read own source: {exc}"]
    _call = None
    for _fn in _ast.walk(_tree):
        if isinstance(_fn, _ast.FunctionDef) and _fn.name == "check":
            for _n in _ast.walk(_fn):
                if (isinstance(_n, _ast.Call)
                        and getattr(_n.func, "id", None) == "exit_code"):
                    _call = _n
    if _call is None:
        return fails + ["exit_code: `check()` no longer calls `exit_code` — "
                        "the ratchet is disconnected entirely"]
    seen = {}
    for _kw in _call.keywords:
        if isinstance(_kw.value, _ast.Dict):
            for _k, _v in zip(_kw.value.keys, _kw.value.values):
                if isinstance(_k, _ast.Constant):
                    seen[_k.value] = _v
    for name in RED + AMBER:
        if name not in seen:
            fails.append(f"exit_code wiring: `check()` no longer passes "
                         f"`{name}` — a condition dropped from the call site "
                         f"is a ratchet nobody will notice is gone")
        elif isinstance(seen[name], _ast.Constant) or (
                isinstance(seen[name], (_ast.List, _ast.Tuple, _ast.Dict))
                and not getattr(seen[name], "elts", getattr(seen[name],
                                                            "keys", None))):
            fails.append(f"exit_code wiring: `{name}` is passed a CONSTANT at "
                         f"the call site, so the condition can never fire — "
                         f"wired in appearance only")
    return fails


if __name__ == "__main__":
    raise SystemExit(check())
