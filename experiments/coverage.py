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

HOW IT AVOIDS ROTTING. The commitment list below is hand-maintained, which is
exactly what went stale about `ladder_prompt.md`'s cached counts. Two defences:
`check()` returns a nonzero count that an organ can act on rather than a wall of
prose, and any commitment added to GOAL.md without a line here shows up as a
GOAL.md section this file cannot name — which the overseer is told to look for.
Better would be deriving these from GOAL.md automatically; that is not
attempted, because a regex over prose that silently matches nothing is worse
than a list a human can read and correct.

A SPEC MAY ALSO DECLARE ITSELF. Put `COVERS: <commitment>` in a spec's notes
and it counts, regardless of title. This is not a convenience — it is the fix
for a failure this file committed on its first day: BA.01 was registered
specifically to close the `balance` hole, titled "He feels himself falling
before he falls", and the balance regex (`balance|topple|upright|vestibul`)
did not match it. **The gap-finder had a gap**, and the tempting repair —
adding "fall" to the pattern — is how a detector gets tuned until it agrees
with you. An explicit marker is a deliberate statement by the spec's author and
cannot be matched by accident; the regex stays as a safety net for specs whose
authors never thought about this file.

MATCH ON TITLES, NOT ON EVERYTHING. Searching the whole spec text finds
"temperature" inside an unrelated note and reports coverage that does not exist
— measured: the loose search claimed 2 thermal specs and both were incidental
mentions. A spec covers a commitment when the commitment is what the spec is
ABOUT, and the title is the honest test of that.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

# name -> (regex over spec TITLES, why this is constitutional)
COMMITMENTS: Dict[str, Tuple[str, str]] = {
    "sight":              (r"camera|visual|vision|eye|see\b|retina", "every sense a human has"),
    "hearing":            (r"audio|acoustic|sound|hear|binaural", "every sense a human has"),
    "touch/contact":      (r"touch|tactile|contact", "every sense a human has"),
    "smell":              (r"odour|odor|smell|olfact", "owner named it constitutional"),
    "taste":              (r"taste|gustat|poison", "owner named it constitutional"),
    "voice":              (r"vocal|voice|utterance|speak|signal", "owner: he must have a voice"),
    "balance":            (r"balance|topple|upright|vestibul", "every sense a human has"),
    "proprioception":     (r"propriocept|body schema|limb", "every sense a human has"),
    "thermal (kills)":    (r"thermal|temperature|cold|freez|heat", "owner: too cold/hot KILLS him"),
    "damage/nociception": (r"damage|injur|nocicept|pain", "danger must be learnable before fatal"),
    "hunger/thirst":      (r"hunger|thirst|drink|forage|drive", "owner: permanent human needs"),
    "sleep":              (r"sleep|consolidat|siesta", "biology as oracle"),
    "death & retry":      (r"death|dies|lethal|surviv|statue", "owner: he dies and retries"),
    "memory across lives":(r"across lives|between lives|prior life|erase", "owner: REMEMBERS across lives"),
    "shelter/building":   (r"shelter|build|construct|nest", "owner's own image of success"),
    "tool use":           (r"tool|affordance", "caveman realism"),
    "language (parent)":  (r"language|word|grounding|lexic", "LLM as talkative parent"),
    "social/other agents":(r"social|companion|two jacks|second jack", "owner: socialising makes him kind"),
    "curiosity":          (r"curiosity|novelty|exploration|learning progress", "the world is the teacher"),
    "one brain / unison": (r"unison|fused|binding|cross-modal|shared|one brain", "the constitution itself"),
    "plasticity":         (r"plastic|frozen|does not die|forgetting", "PLASTIC ONLY decree"),
    "generality":         (r"generalis|generaliz|held.out|unseen|transfer", "GEN.00, the final exam"),
    "fast/slow":          (r"deliberat|habit|slow path|lookahead", "owner 2026-08-10"),
}


def report() -> List[dict]:
    from .registry import LADDER
    led = {}
    p = Path(__file__).resolve().parent / "ledger.json"
    if p.is_file():
        led = json.load(open(p)).get("results", {})
    out = []
    for name, (pat, why) in COMMITMENTS.items():
        rx = re.compile(pat, re.I)
        marker = re.compile(r"COVERS:\s*" + re.escape(name), re.I)
        hits = [s.id for s in LADDER
                if rx.search(s.title) or marker.search(str(s.notes or ""))]
        passing = [i for i in hits if led.get(i, {}).get("status") == "PASS"]
        out.append({"commitment": name, "why": why, "specs": hits,
                    "n_specs": len(hits), "n_pass": len(passing)})
    return out


def check() -> int:
    """Print the audit; return the number of UNCOVERED commitments.

    Uncovered means zero specs, which is the only failure this can assert
    without judgement. "Covered but not passing" is normal — it is a ladder,
    not a scoreboard — so it is reported and not counted.
    """
    rows = report()
    width = max(len(r["commitment"]) for r in rows)
    uncovered = [r for r in rows if r["n_specs"] == 0]
    unproven = [r for r in rows if r["n_specs"] and not r["n_pass"]]
    for r in sorted(rows, key=lambda z: (z["n_specs"], z["n_pass"])):
        mark = "NO SPECS" if not r["n_specs"] else (
            "none passing" if not r["n_pass"] else "")
        print(f"  {r['commitment']:{width}}  {r['n_specs']:>3} specs "
              f"{r['n_pass']:>3} pass   {mark}")
        if not r["n_specs"]:
            print(f"  {'':{width}}  ^ {r['why']}")
    print(f"\n  {len(uncovered)} commitment(s) with NO spec, "
          f"{len(unproven)} with specs but nothing passing.")
    if uncovered:
        print("  A commitment with no spec is invisible to `run blocked`, to the\n"
              "  overseer, and to every gate. Register one before demonstrating\n"
              "  anything else — this is the cheapest possible bug to fix and the\n"
              "  most expensive to leave.")
    return len(uncovered)


if __name__ == "__main__":
    raise SystemExit(1 if check() else 0)
