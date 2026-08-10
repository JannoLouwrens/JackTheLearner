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
DECLARATION = re.compile(r"COVERS:\s*([^\n.;]+)", re.I)

_CANON = {k.lower(): k for k in COMMITMENTS}


def declarations(by_id: Optional[dict] = None
                 ) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
    """Read every spec's `COVERS:` markers.

    Returns `(commitment -> [spec ids], [(spec id, unrecognised name)])`. The
    second half is the point: a declaration naming a commitment that does not
    exist is reported, never dropped. A typo'd marker looks exactly like a
    claim to a human reader and buys exactly nothing from this file, which is
    the false-positive failure this module was rewritten to end.
    """
    if by_id is None:
        from .registry import BY_ID
        by_id = BY_ID
    declared: Dict[str, List[str]] = {k: [] for k in COMMITMENTS}
    bad: List[Tuple[str, str]] = []
    for sid, spec in by_id.items():
        for group in DECLARATION.findall(str(getattr(spec, "notes", "") or "")):
            for raw in group.split(","):
                name = raw.strip()
                if not name:
                    continue
                canon = _CANON.get(name.lower())
                if canon is None:
                    bad.append((sid, name))
                elif sid not in declared[canon]:
                    declared[canon].append(sid)
    return declared, bad


def report(by_id: Optional[dict] = None,
           results: Optional[dict] = None) -> List[dict]:
    """Coverage (declared) and nominations (regex), never mixed.

    `n_specs`/`n_pass` count DECLARED specs only. `nominations` lists specs a
    pattern matched that have not declared — work to do, not coverage.
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
        specs = [i for i in declared[name] if i in by_id]
        nominated = [s.id for s in by_id.values()
                     if rx.search(s.title) and s.id not in specs]
        passing = [i for i in specs if results.get(i, {}).get("status") == "PASS"]
        out.append({"commitment": name, "why": why, "specs": specs,
                    "n_specs": len(specs), "n_pass": len(passing),
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
        print(f"  {len(bad)} MALFORMED declaration(s) — these name no "
              f"commitment and buy nothing:")
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
