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

_CANON = {k.lower(): k for k in COMMITMENTS}


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
           results: Optional[dict] = None) -> List[dict]:
    """Coverage (declared) and nominations (regex), never mixed.

    `n_specs`/`n_pass` count DECLARED specs only. `nominations` lists specs a
    pattern matched that have not declared — work to do, not coverage.

    `n_pass` counts passing `claim` declarations ONLY. A passing fixture, rule
    or sensor is real work and is reported in `support_pass` — but apparatus
    demonstrating itself is not the commitment being demonstrated, and merging
    the two is how `curiosity` and `one brain / unison` each read as started
    for three audits while no capability test had ever run.
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
    out = []
    for name, (pat, why) in COMMITMENTS.items():
        rx = re.compile(pat, re.I)
        pairs = [(i, k) for i, k in declared[name] if i in by_id]
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
                    "n_specs": len(specs), "n_pass": len(passing),
                    "support_pass": support,
                    "nominations": nominated, "n_nominated": len(nominated),
                    "bad_declarations": [d for d in bad]})
    return out


def check() -> int:
    """Print the audit; return UNCOVERED commitments + malformed declarations.

    Uncovered means zero DECLARED specs. "Covered but not passing" is normal —
    it is a ladder, not a scoreboard — so it is reported and not counted.
    """
    rows = report()
    bad = rows[0]["bad_declarations"] if rows else []
    width = max(len(r["commitment"]) for r in rows)
    uncovered = [r for r in rows if r["n_specs"] == 0]
    unproven = [r for r in rows if r["n_specs"] and not r["n_pass"]]
    print(f"  {'commitment':{width}}  covered (declared)   nominated")
    for r in sorted(rows, key=lambda z: (z["n_specs"], z["n_pass"])):
        mark = "NO SPECS" if not r["n_specs"] else (
            "none passing" if not r["n_pass"] else "")
        if r["support_pass"]:
            kinds = ", ".join(f"{i} ({k})" for i, k in r["support_pass"].items())
            mark = (mark + f"  [support passing, not credited: {kinds}]").strip()
        print(f"  {r['commitment']:{width}}  {r['n_specs']:>3} specs "
              f"{r['n_pass']:>3} pass   {r['n_nominated']:>3} nominated   {mark}")
        if not r["n_specs"]:
            print(f"  {'':{width}}  ^ {r['why']}")
            if r["nominations"]:
                print(f"  {'':{width}}    nominations (declare or ignore): "
                      f"{', '.join(r['nominations'][:8])}")
    print(f"\n  {len(uncovered)} commitment(s) with NO declared spec, "
          f"{len(unproven)} with specs but nothing passing.")
    if bad:
        print(f"  {len(bad)} MALFORMED declaration(s) — a typo'd commitment "
              f"name or a missing kind; either buys nothing:")
        for sid, name in bad:
            print(f"      {sid}: COVERS: {name!r}")
    if uncovered:
        print("  A commitment with no spec is invisible to `run blocked`, to the\n"
              "  overseer, and to every gate. Register one before demonstrating\n"
              "  anything else — this is the cheapest possible bug to fix and the\n"
              "  most expensive to leave.")
    print("\n  A nomination is NOT coverage. It is a spec whose title looks\n"
          "  related and whose author has not said so; only `COVERS:` counts.")
    return len(uncovered) + len(bad)


if __name__ == "__main__":
    raise SystemExit(1 if check() else 0)
