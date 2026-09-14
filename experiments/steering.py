"""Can the orders on the steering pages actually be executed today?

THE SCAR (overseer, 95th audit, `docs/OVERSIGHT.md` RANK 3 and FOR THE BUILDER
item 2 — this module is that item). `docs/PROGRESS.md`'s `FOR THE BUILDER`
block is a live order list written at 06:37 on 2026-09-13. Of its six items,
**two were dead before the ink dried**:

  * item 1 — *"`T2.10` — CPU, ten minutes"* — names a settled FAIL whose own
    docstring records that every scorer this project has measured tops out at
    0.0667 against an unmoving 0.10 bar. A ten-minute re-run buys the row
    already on the ledger.
  * item 3 — *"`D1.0` attempt 3 into W37"* — has been **illegal since 10:05
    the same morning**, when the Review's own Part-2 strengthening turned
    `T1.08` PASS -> FAIL. `D1.0` declares `depends_on: [T2.00, T1.08, T0.09,
    T0.10]`, so the runner would refuse the dispatch.

**Three consecutive builder iterations re-derived that by hand**, from
`coverage`, `blocked` and a docstring, and the fourth (this one) read it off
an audit that also derived it by hand. Every reader needed to answer the
question mechanically already existed — `BY_ID`, `Ledger.unsatisfied`,
`coverage._liveness_state`, `decisions.holds` — and no organ asked them about
the one page the orientation puts in front of every iteration.

This is `coverage.goal_citations` pointed at a different document, and it is
deliberately built in that idiom rather than a new one: resolve every
spec-shaped id a governing page cites, for EXISTENCE and for LIVENESS, using
**the shared predicates** so this reader cannot drift from `run blocked` and
`run coverage` the way two internally-consistent organs always do.

WHAT IT IS NOT, stated first because the sibling row
(`oversight-for-the-builder-has-no-reader`, DUE 2026-09-17) argues correctly
against the other half:

  * **It is not a discharge check.** It asks whether an order *could be*
    executed, never whether it *was*. Discharge is a property of a COMMIT and
    commits do not quote; that is the sibling row's question and it is the
    Review's to rule on.
  * **It is not a gate and carries no floor.** An order can be legitimately
    aspirational — *"`T1.08` is the whole game"* is a true and useful sentence
    about a spec that is FAIL and blocks 45 others. A floor here would forbid
    a legal move, and an unfloored counter that nobody is accountable to is
    the honest shape for a question whose right answer is sometimes non-zero.
  * **It does not read intent.** A cited id may be the object of the order or
    a fact quoted in support of it. The SUBJECT / MENTION split below is a
    heuristic about markdown emphasis and is labelled as one everywhere it is
    printed. The reader locates; the human judges.

WHY THE FALSE-POSITIVE OBJECTION DOES NOT LAND HERE, and it lands hard one row
over. `D27`'s prototype screen flagged 104 of 107 PASS specs and 3 of 12
hand-checks were real; the sibling row's case (1) predicts the same fate for a
VANISHED-ITEM detector, because "this item disappeared" is the normal output
of a page that is rewritten wholesale by design. Legality is not that kind of
question. `T1.08 = FAIL` is not a judgement about whether an audit had moved
on — it is a fact the runner will enforce at the moment of dispatch, and the
reader here reports exactly what the runner would do. Its precision is the
registry's precision.

THE POPULATION PROBLEM IS REAL AND IS NOT SOLVED HERE. `FOR THE BUILDER` items
have no ids; identifying them is heading-and-prose parsing over a page one
organ writes and its successor overwrites. SYSTEM.md named that exact shape on
2026-09-13 — *a checker reading a population somebody else selected* — and the
mitigation there was two independent channels. This reader has one. What keeps
it honest instead is that its VERDICTS come from elsewhere: the page selects
which ids get looked at, and the ledger, the registry and the decision docket
decide what is said about them. A steering page cannot make an illegal order
look legal by rewording itself, only by not mentioning the id at all — and an
order that names no spec is one this reader correctly says nothing about.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent

# The order lists. Both are CURRENT-STATE pages rewritten by their own organ —
# `PROGRESS.md` weekly-and-daily by the Review, `OVERSIGHT.md` every six hours
# by the overseer — which is precisely why an order on either can go stale
# between being written and being read.
STEERING_PAGES = ("docs/PROGRESS.md", "docs/OVERSIGHT.md")

# Start-of-line only, like every declaration in this repo: `OVERSIGHT.md`
# discusses its own `FOR THE BUILDER` section in prose three times and none of
# those may look like a heading. `finditer`, not `search` — a page is allowed
# more than one such block and dropping the second would be a silent hole.
_BUILDER_HEADING = re.compile(r"^##\s+FOR THE BUILDER\s*$", re.M)
_SECTION_END = re.compile(r"^(##\s|---\s*$)")
_ITEM = re.compile(r"^(\d{1,2})\.\s+(.*)$")

# Same shape as `coverage.GOAL_CITATION`, and deliberately the same regex:
# two readers of governing prose disagreeing about what a spec id looks like
# is the drift this file exists to avoid.
SPEC_CITATION = re.compile(r"\b([A-Z]{1,4}[0-9]?\.[0-9]{1,2})\b")

# Most-binding first. An id can be several of these at once (`D1.0` is both
# BLOCKED behind `T1.08` and, were it parked, held); the first match is what
# gets printed, so the order is the claim about which fact stops the work.
VERDICT_ORDER = ("UNKNOWN", "HELD", "DECISION-HELD", "BLOCKED")

# A PROHIBITION whose subject is illegal is the order AGREEING with the
# ledger, not contradicting it, and printing the two identically is how a
# reader earns the fate the sibling row predicts for it.
#
# MEASURED, not asserted (2026-09-14, the live pages at `4a028c1`): 10 items,
# 2 flagged, and one of the two was `PROGRESS.md` item 6 — *"Do not re-run
# `T6.03` until `T2.10` is PASS"* — whose subject is BLOCKED behind exactly
# the spec the sentence names. 50% of this reader's first findings were an
# instrument complaining that a correct prohibition was correct. With the
# rule: 1 flag, 1 real, plus 1 confirmation printed as one.
#
# It is start-anchored prose matching and it is therefore a HEURISTIC, in the
# same declared class as the bold-lead split — a prohibition phrased any other
# way falls back to being flagged, which is the safe direction: the failure
# mode is a false alarm a human dismisses in one line, never a real dead order
# going quiet.
_PROHIBITION = re.compile(r"^\s*(do not|don't|never)\b", re.I)


def builder_items(text: str) -> List[dict]:
    """The numbered items under every `## FOR THE BUILDER` heading, in order.

    `{n, page, text, lead, lead_ids, body_ids}`. `lead` is the item's first
    bold span — the Review and the overseer both write
    `N. **<the order>** <the reasoning>`, so the bold span is where the object
    of the order lives and the rest is support. That is a HEURISTIC about
    house style, not a claim about intent, and every caller labels it as one.
    It is the same `lead` convention `decisions.owner_asks` already uses on
    the facing section of the same file.
    """
    out: List[dict] = []
    for m in _BUILDER_HEADING.finditer(text or ""):
        cur: Optional[dict] = None
        for ln in text[m.end():].splitlines():
            if _SECTION_END.match(ln):
                break
            head = _ITEM.match(ln)
            if head:
                cur = {"n": int(head.group(1)), "lines": [head.group(2)]}
                out.append(cur)
            elif cur is not None:
                cur["lines"].append(ln)
    items = []
    for it in out:
        body = "\n".join(it["lines"])
        bold = re.search(r"\*\*(.+?)\*\*", body, re.S)
        lead = re.sub(r"\s+", " ", (bold.group(1) if bold else
                                    body.splitlines()[0])).strip()
        lead_ids = sorted(set(SPEC_CITATION.findall(lead)))
        items.append({
            "n": it["n"], "text": body, "lead": lead[:110],
            "prohibition": bool(_PROHIBITION.match(lead)),
            "lead_ids": lead_ids,
            "body_ids": sorted(set(SPEC_CITATION.findall(body))
                               - set(lead_ids)),
        })
    return items


def legality(sids, by_id=None, ledger=None, state_of=None,
             held_map=None) -> Dict[str, str]:
    """`{cited id -> reason it cannot be run today}`. Legal ids are absent.

    Four verdicts, each from the organ that owns the question:

      UNKNOWN        not in `BY_ID` — the order names a spec nobody registered
      HELD           `coverage._liveness_state`: PARKED / VOID-FORECLOSED /
                     PILOT-BLOCKED / welded behind dead roots
      DECISION-HELD  `decisions.holds` — an open decision names it
      BLOCKED        `Ledger.unsatisfied` — the runner would refuse it, and
                     this is the one that caught `D1.0` five hours late

    Every argument is injectable so the reader can be checked against a graph
    whose answer is known without markers on disk — the seam `_terminal_
    blockers` named and never got, and which `_next_triage` built in on its
    first day.
    """
    sids = sorted(set(sids))
    if by_id is None:
        from .registry import BY_ID
        by_id = BY_ID
    resolved = [s for s in sids if s in by_id]
    if ledger is None:
        from .protocol import Ledger
        ledger = Ledger()
    if state_of is None:
        from .coverage import _liveness_state
        state_of = _liveness_state(resolved, by_id)
    if held_map is None:
        from .decisions import holds
        held_map = holds()
    out: Dict[str, str] = {}
    for sid in sids:
        if sid not in by_id:
            out[sid] = "UNKNOWN — no such spec in the registry"
            continue
        if sid in state_of:
            out[sid] = f"HELD — {state_of[sid]}"
            continue
        if sid in held_map:
            out[sid] = f"DECISION-HELD — {held_map[sid]}"
            continue
        unmet = ledger.unsatisfied(by_id[sid])
        if unmet:
            out[sid] = "BLOCKED — " + ", ".join(
                f"{d} ({w})" for d, w in unmet)
    return out


def read(pages=STEERING_PAGES, repo: Optional[Path] = None,
         **kw) -> List[dict]:
    """Every steering-page order, with the illegal ids it names.

    One dict per item: `{page, n, lead, subject, mention}`, where `subject`
    and `mention` are `{id -> reason}` over the item's lead span and its body
    respectively. Items naming nothing illegal are still returned — the count
    of clean items is what stops this from being a detector that only ever
    reports bad news, and a page with no orders at all must be visibly
    distinguishable from a page whose orders are all fine.
    """
    repo = _REPO if repo is None else repo
    items: List[dict] = []
    for rel in pages:
        p = repo / rel
        if not p.exists():
            continue
        for it in builder_items(p.read_text()):
            it["page"] = rel
            items.append(it)
    verdicts = legality(
        [s for it in items for s in it["lead_ids"] + it["body_ids"]], **kw)
    for it in items:
        it["subject"] = {s: verdicts[s] for s in it["lead_ids"]
                         if s in verdicts}
        it["mention"] = {s: verdicts[s] for s in it["body_ids"]
                         if s in verdicts}
    return items


def render(items: Optional[List[dict]] = None, indent: str = "  ") -> str:
    """The block printed by `run status` and `run steering`.

    Prints SUBJECT lines always and MENTION lines only under an item that has
    an illegal subject — a mention is context for an order that is already in
    question, and promoting every quoted corpse to a finding is how a reader
    at `D27`'s 104-of-107 rate gets ignored inside a week.
    """
    items = read() if items is None else items
    if not items:
        return (f"{indent}STEERING-PAGE ORDERS — no `## FOR THE BUILDER` "
                f"section found on {', '.join(STEERING_PAGES)}. That is "
                f"itself worth\n{indent}  saying: the pages are the order "
                f"list, and an empty one is a finding, not a clean bill.\n")
    hit = [it for it in items if it["subject"]]
    bad = [it for it in hit if not it["prohibition"]]
    ok = [it for it in hit if it["prohibition"]]
    head = (f"{indent}STEERING-PAGE ORDERS — {len(items)} item(s) on "
            f"{len(set(i['page'] for i in items))} page(s); "
            f"{len(bad)} order(s) name a spec the\n"
            f"{indent}  runner would REFUSE today. Reporting-only, unfloored: "
            f"an order may legitimately be\n"
            f"{indent}  aspirational, and SUBJECT vs MENTION is a heuristic "
            f"about which span is bold, never a\n"
            f"{indent}  reading of intent.\n")
    if not hit:
        return head + (f"{indent}  every order's subject resolves to a spec "
                       f"the runner would accept today.\n")
    lines = [head]
    for it in bad:
        lines.append(f"{indent}  {it['page']} item {it['n']}: {it['lead']}\n")
        for sid, why in sorted(it["subject"].items()):
            lines.append(f"{indent}    SUBJECT  {sid:9s} {why}\n")
        for sid, why in sorted(it["mention"].items()):
            lines.append(f"{indent}    mention  {sid:9s} {why}\n")
    for it in ok:
        lines.append(f"{indent}  CONFIRMED — {it['page']} item {it['n']} is a "
                     f"prohibition and the ledger agrees with it:\n"
                     f"{indent}    {it['lead']}\n")
        for sid, why in sorted(it["subject"].items()):
            lines.append(f"{indent}    subject  {sid:9s} {why}\n")
    return "".join(lines)


# ── the known answer ────────────────────────────────────────────────────────
#
# Verbatim from `docs/PROGRESS.md` at `d44d21a` (Review DAILY, 2026-09-13
# 06:37) — the two items the 95th audit derived dead BY HAND, and the three
# iterations before it derived dead by hand from three different tools. The
# snapshot is frozen on purpose: asserting against the live page would make
# this check pass or fail on what the Review wrote this morning, which is the
# opposite of a control.
_FIXTURE_PAGE = """
## FOR THE BUILDER

Ordered. Items 1-3 are in `ladder_prompt.md` as well; 4-6 are new here.

1. **`T2.10` - CPU, ten minutes, and worth more today than yesterday.** It is
   runnable and FAIL, and it now gates **two certificates**: `T6.03` cannot be
   re-bought and `LF.02` is out of the reachable set behind it.
2. **`D25`'s armed default is due TODAY** - option (iii) FIX THE SEAL.
3. **`D1.0` attempt 3 into W37.** The two-step precondition is satisfied for the
   first time (`7cb00ea`). `T2.01`'s **38** transitively-blocked specs are what
   is behind it.
6. **Do not re-run `T6.03` until `T2.10` is PASS.** It will only return BLOCKED
   and burn a slot.

---

## FOR THE OWNER

4. **This item is on the wrong side of the boundary and must not be read.**
"""


def _check() -> None:
    """Refuse to report from a reader that flunks a known answer.

    Same contract as `_check_next_triage` and `_check_ranker`: a count from an
    instrument that cannot get a fixture right is not evidence.

    THE FIXTURE IS A REPLAY, NOT AN INVENTION — the 2026-09-13 lesson
    (*"replay a remedy's ARITHMETIC, not just whether it fires"*) applied on
    the first day rather than after the first miss. The known answer is not
    "some item is flagged"; it is the exact pair of items a human found by
    hand, with the exact reason for each, and the two items beside them that
    must stay silent.

      item 1  SUBJECT `T2.10`  BLOCKED is WRONG and must not appear -- T2.10's
              deps pass, it is a legal run. Its deadness is a fact about its
              measured ceiling, which no registry reader can see and this one
              must not pretend to. The auditor read the docstring; that is
              still a human's job. Recorded here so nobody later "fixes" this
              reader by teaching it to guess.
      item 3  SUBJECT `D1.0`   BLOCKED behind `T1.08` -- the live miss, five
              hours old when the page shipped, and the one a machine can see.
      item 3  mention `T2.01`  printed only because item 3's subject is bad.
      item 2  clean            names no spec at all; a decision id is not a
              spec id and must not be scraped as one.
      item 6  CONFIRMED        a prohibition whose subject `T6.03` is BLOCKED
              behind the very spec the sentence names. It must NOT render as
              a flag: the ledger is agreeing with the order. This is the live
              false positive that produced `_PROHIBITION`, kept executable so
              deleting the rule turns the fixture red.
      item 4  NOT AN ITEM      it sits under `## FOR THE OWNER`; the section
              boundary is the whole reason `_SECTION_END` exists.
    """
    items = builder_items(_FIXTURE_PAGE)
    got = [(it["n"], it["prohibition"], it["lead_ids"], it["body_ids"])
           for it in items]
    want = [(1, False, ["T2.10"], ["LF.02", "T6.03"]),
            (2, False, [], []),
            (3, False, ["D1.0"], ["T2.01"]),
            # BOTH ids are in the bold span, so both are SUBJECTs — the split
            # is about emphasis and nothing else, and this is what that costs.
            # `T2.10` is legal and simply drops out at the verdict stage; the
            # parser is not allowed to be clever about which of two bolded ids
            # the sentence is "really" about.
            (6, True, ["T2.10", "T6.03"], [])]
    if got != want:
        raise AssertionError(f"steering: item parse flunked its fixture: "
                             f"{got} != {want}")

    by_id = {"T2.10": object(), "D1.0": object(), "T2.01": object(),
             "T6.03": object(), "LF.02": object()}

    class _L:
        def unsatisfied(self, spec):
            if spec is by_id["D1.0"]:
                return [("T1.08", "FAIL")]
            if spec is by_id["T6.03"]:
                return [("T2.10", "FAIL")]
            return []

    verdicts = legality(["T2.10", "D1.0", "T2.01", "T6.03", "LF.02", "ZZ.99"],
                        by_id=by_id, ledger=_L(),
                        state_of={"T2.01": "PARKED"}, held_map={})
    if set(verdicts) != {"D1.0", "T2.01", "T6.03", "ZZ.99"}:
        raise AssertionError(f"steering: legality flunked its fixture: "
                             f"{sorted(verdicts)}")
    if not verdicts["D1.0"].startswith("BLOCKED — T1.08 (FAIL)"):
        raise AssertionError(f"steering: wrong reason for D1.0: "
                             f"{verdicts['D1.0']}")
    if not verdicts["ZZ.99"].startswith("UNKNOWN"):
        raise AssertionError("steering: an unregistered id must read UNKNOWN")

    for it in items:
        it["page"] = "fixture"
    for it in items:
        it["subject"] = {s: verdicts[s] for s in it["lead_ids"]
                         if s in verdicts}
        it["mention"] = {s: verdicts[s] for s in it["body_ids"]
                         if s in verdicts}
    text = render(items, indent="")
    if "item 3" not in text or "D1.0" not in text:
        raise AssertionError("steering: the live miss is not rendered")
    if "item 1" in text:
        raise AssertionError(
            "steering: item 1 was flagged. T2.10's dependencies pass; a "
            "reader that reports it is guessing at a docstring it never read.")
    if "T2.01" not in text:
        raise AssertionError("steering: a mention under a bad subject is "
                             "context and must print")
    if "1 order(s) name a spec the" not in text:
        raise AssertionError(f"steering: the headline count must exclude the "
                             f"confirmed prohibition:\n{text}")
    if "CONFIRMED" not in text or "item 6" not in text:
        raise AssertionError("steering: a prohibition the ledger agrees with "
                             "must be printed, and printed as agreement")


if __name__ == "__main__":  # pragma: no cover
    _check()
    print(render(), end="")
