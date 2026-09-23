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

import datetime as _dt
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent

# The order lists. Both are CURRENT-STATE pages rewritten by their own organ —
# `PROGRESS.md` weekly-and-daily by the Review, `OVERSIGHT.md` every six hours
# by the overseer — which is precisely why an order on either can go stale
# between being written and being read.
STEERING_PAGES = ("docs/PROGRESS.md", "docs/OVERSIGHT.md")

# --- the launcher's hard ceiling (PROGRESS FOR THE BUILDER item 4) ---------
# `ladder_loop.sh` passes this page's ENTIRE TEXT as a single argv to
# `claude -p "$PROMPT"`, so it is bounded by Linux's MAX_ARG_STRLEN. On
# 2026-09-20 at 06:44 a steering rewrite pushed it past that constant and
# every slot from 07:07 died `rc=126` at `execve`, before a token was read —
# nineteen consecutive slots, twenty-three hours, and the builder's whole
# capacity was zero while three liveness instruments read it as healthy.
#
# The Review's repair was a 125000-byte ceiling written into
# `ladder_prompt.md` AS A SENTENCE. That is the failure shape this project
# keeps paying for: a desk rules correctly, writes it down truthfully, commits
# it, and nothing an instrument can see has changed. This is the sentence
# given a reader.
LAUNCH_PAGE = "scripts/ladder_prompt.md"
LAUNCH_CLIFF = 131072           # Linux MAX_ARG_STRLEN. NOT ours to choose:
                                # past it the builder does not read a degraded
                                # prompt, it does not start at all, and the
                                # failure looks like an ordinary rc=126 slot.
LAUNCH_CEILING = 125000         # the Review's self-imposed margin below it.
LAUNCH_GROWTH_DAYS = 21         # window over which the growth rate is fitted.

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


# ── decision deadlines quoted on a steering page (100th audit B1) ───────────
#
# THE SCAR: the duplicate-D30 bug was fixed in the register at ~13:0x on
# 2026-09-18, and at 18:22 the same day the Review's page told the owner the
# decision expired 2026-09-25 — a sentence carried forward from the broken
# register, published five hours AFTER the lesson about exactly this
# ("fixing a register does not fix what was published from it"). The owner
# was handed seven days on a question that expired that night. A written
# lesson did not survive one organ boundary; this reader is the instrument
# form of it, in `run status` where a sitting cannot finish without it.
#
# A decision id, as distinct from a spec id: `D30` yes, the `D1` inside
# `D1.0` no — the lookahead is what keeps the two citation grammars from
# colliding on the same page.
DECISION_CITATION = re.compile(r"\bD(\d{1,3})\b(?!\.\d)")
_ISO = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
# Approximate sentence split over paragraph-flattened prose. A HEURISTIC in
# the same declared class as the bold-lead split: a false boundary costs a
# missed pairing (silent, the safe direction), never an invented one.
_SENT = re.compile(r"(?<=[.!?])\s+")


def open_register(path: Optional[Path] = None) -> Dict[str, str]:
    """{open decision id -> its `decide_by` ISO date}, from the register.

    Only OPEN decisions: a resolved decision's dates are history and a page
    discussing them is narrating, not steering. Only parseable ISO
    `decide_by` values: an undated open decision has no deadline to misquote.
    """
    from .decisions import DOC, parse
    decls, open_ids, _dupes = parse(Path(path or DOC).read_text())
    out: Dict[str, str] = {}
    for did in open_ids:
        db = ((decls.get(did) or {}).get("decide_by") or "").strip()
        if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", db):
            out[did] = db
    return out


def date_mismatches(pages=STEERING_PAGES, repo: Optional[Path] = None,
                    register: Optional[Dict[str, str]] = None) -> List[dict]:
    """Sentences on a steering page that cite an OPEN decision beside a full
    ISO date agreeing with NEITHER its `decide_by` NOR the day after it.

    The day-after allowance is deliberate and asymmetric: an armed default
    FIRES the day after its deadline (`decisions.py` marks overdue at
    `> 0` days), so *"`D30`'s default fires on 2026-09-19"* is a correct
    sentence about a 2026-09-18 deadline and flagging it would put a false
    positive beside the one real finding — the D27 fate. A sentence carrying
    ANY agreeing date is silent, so a page that quotes the wrong date while
    correcting it (as an audit does) stays quiet.

    Whole page, not just `FOR THE BUILDER`: the known positive lived under
    `FOR THE OWNER`. Reporting-only and unfloored, per D27's reasoning — a
    page may cite a date for another reason, and this reader does not read
    intent; the register is the authority it compares against, never the
    judge of why the page said what it said.
    """
    repo = _REPO if repo is None else repo
    if register is None:
        register = open_register()
    out: List[dict] = []
    for rel in pages:
        p = repo / rel
        if p.exists():
            out.extend(text_date_mismatches(p.read_text(), register, rel))
    return out


def text_date_mismatches(text: str, register: Dict[str, str],
                         page: str = "") -> List[dict]:
    """`date_mismatches` over one page's text — the pure half, so the fixture
    can replay the known answer without touching disk."""
    out: List[dict] = []
    seen = set()
    for para in re.split(r"\n\s*\n", text or ""):
        flat = re.sub(r"\s+", " ", para).strip()
        for sent in _SENT.split(flat):
            dates = _ISO.findall(sent)
            if not dates:
                continue
            for m in DECISION_CITATION.finditer(sent):
                did = "D" + m.group(1)
                db = register.get(did)
                if db is None:
                    continue
                fire = (_dt.date.fromisoformat(db)
                        + _dt.timedelta(days=1)).isoformat()
                if any(d in (db, fire) for d in dates):
                    continue
                key = (did, tuple(sorted(set(dates))))
                if key in seen:
                    continue
                seen.add(key)
                out.append({"page": page, "id": did,
                            "page_says": sorted(set(dates)),
                            "register_says": db,
                            "sentence": sent[:160]})
    return out


def render_dates(mis: Optional[List[dict]] = None,
                 indent: str = "  ") -> str:
    """The `STEERING-DATE-MISMATCH` block printed by `run status`."""
    if mis is None:
        mis = date_mismatches()
    if not mis:
        return (f"{indent}STEERING-DATE-MISMATCH — none: every open-decision "
                f"deadline quoted on a steering page\n"
                f"{indent}  agrees with the register (or with its fire day, "
                f"deadline + 1).\n")
    lines = [f"{indent}STEERING-DATE-MISMATCH — {len(mis)} open-decision "
             f"deadline(s) misquoted on a steering page.\n"
             f"{indent}  Reporting-only, unfloored: the register is the "
             f"authority; a page may cite a date\n"
             f"{indent}  for another reason, and this reader does not read "
             f"intent.\n"]
    for m in mis:
        lines.append(f"{indent}    {m['id']}  {m['page']} says "
                     f"{', '.join(m['page_says'])} — register says decide_by "
                     f"{m['register_says']}\n")
        lines.append(f"{indent}      \"{m['sentence']}\"\n")
    return "".join(lines)


# ── metric values quoted on a steering page (builder, 2026-09-23) ───────────
#
# THE SCAR: on 2026-09-22 06:55 (`51c9d13`) the Review armed a midnight
# stop-rule on `me1-similarity-floor-never-abstains` and wrote it onto both
# steering pages quoting `distractor_abstention` **0.0000 ± 0.0** as the live
# reading — the 09-06 routing figure, sixteen days after the repair landed
# and eight days after `ME.1`'s attempt-10 certificate recorded the same
# metric at 1.0. The stop-rule ordered the falsified number routed to the
# owner as an architecture finding, and the discharge evidence sat one
# paragraph below the arming line. Quoted DATES already had this reader
# (100th audit B1); quoted NUMBERS did not, and the ledger — the one
# scoreboard — was never consulted by any organ that copied the figure
# forward. Same contract as the date reader: the ledger is the authority;
# the page may cite a number for another reason; this reader does not read
# intent.
#
# Three declared heuristics, each choosing the silent (missed-pairing)
# direction over the false alarm:
#   * only metric keys containing an underscore — `events` as a bare English
#     word beside an unrelated number is the false-positive shape;
#   * only quoted numbers containing a decimal point — "on 3 seeds" beside a
#     key is prose, and certificate values on these pages are written with
#     their decimals;
#   * a number led by a comparison operator is a BAR, not a reading —
#     "`raw_answer_rate` >= 0.95" describes the gate, and flagging it against
#     the measured 1.0 would put a false positive beside every real finding.
# − is the Unicode minus these pages actually write negatives with —
# reading "−0.2333" as 0.2333 flagged two correct sentences on the first
# live pass of this reader, which is exactly the D27 fate it must not earn.
_NUM = re.compile(r"[-+−]?\d+\.\d+")
_METRIC_WINDOW = 60
# The metric reader ALSO scans the builder's own prompt page — that is where
# today's dead number actually lived (`1^11` item 0), and the page every
# hourly slot reads is the one whose quoted readings most need to be true.
# The date and order readers deliberately do not: `ladder_prompt.md` carries
# superseded blocks as provenance BY DESIGN, and dates/orders inside history
# are narration. Metric quotes are different — the union-agree rule keeps an
# honest historical quote silent whenever the certificate's value appears
# anywhere in the same paragraph, and a paragraph quoting ONLY the dead
# reading is exactly the finding.
# MEASURED on the live pages at first shipping (2026-09-23): 2 findings,
# 1 real (`distractor_abstention` 0.0000 vs the certificate's 1.0 — the
# incident this reader exists for), 1 false (`construction_ok` catching its
# neighbour `memorisers 0.0` inside the window) — a rate a human dismisses
# in one line, recorded here so the next reader knows the error shape.
METRIC_PAGES = STEERING_PAGES + (LAUNCH_PAGE,)


def ledger_metrics(path: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    """{spec id -> {underscore metric key -> value}} from CURRENT entries.

    `metrics` and `control_metrics` both — pages quote either. Numeric values
    only. History is deliberately excluded: a page quoting an old attempt
    writes it beside an arrow to the new one, and the union-agree rule keeps
    such sentences silent without this reader pretending to know which
    attempt a bare number means.
    """
    import json
    p = Path(path) if path else _REPO / "experiments" / "ledger.json"
    entries = json.loads(p.read_text()).get("results", {})
    out: Dict[str, Dict[str, float]] = {}
    for sid, e in entries.items():
        if not isinstance(e, dict):
            continue
        vals: Dict[str, float] = {}
        for src in ("metrics", "control_metrics"):
            for k, v in (e.get(src) or {}).items():
                if (isinstance(v, (int, float)) and not isinstance(v, bool)
                        and "_" in k):
                    vals[k] = float(v)
        if vals:
            out[sid] = vals
    return out


def _rounds_to(quoted: str, value: float) -> bool:
    """Could `quoted` be `value` rounded at the quoted precision?

    `0.344` agrees with 0.343733 (three decimals, tolerance 0.0005);
    `0.0000` does not agree with 1.0. Precision-aware on purpose: an exact
    comparison would flag every honest rounding on the page.
    """
    quoted = quoted.replace("−", "-")
    q = float(quoted)
    dec = len(quoted.split(".")[1]) if "." in quoted else 0
    return abs(q - value) <= 0.5 * 10.0 ** -dec + 1e-12


def metric_mismatches(pages=None, repo: Optional[Path] = None,
                      metrics: Optional[dict] = None) -> List[dict]:
    """Paragraphs on a steering page that name a spec and quote one of that
    spec's ledger metrics beside a number agreeing with NO value the current
    certificate records under that key."""
    repo = _REPO if repo is None else repo
    if pages is None:
        pages = METRIC_PAGES
    if metrics is None:
        metrics = ledger_metrics()
    out: List[dict] = []
    for rel in pages:
        p = repo / rel
        if p.exists():
            out.extend(text_metric_mismatches(p.read_text(), metrics, rel))
    return out


def text_metric_mismatches(text: str, metrics: Dict[str, Dict[str, float]],
                           page: str = "") -> List[dict]:
    """`metric_mismatches` over one page's text — the pure half, so the
    fixture can replay the known answer without touching disk.

    Paragraph-scoped, not sentence-scoped: the live miss quoted its number
    two sentences after naming the spec. Where a paragraph names several
    specs carrying the same key, agreement with ANY of them is silence —
    the reader may not guess whose number the sentence "really" quotes.
    """
    out: List[dict] = []
    seen = set()
    for para in re.split(r"\n\s*\n", text or ""):
        flat = re.sub(r"\s+", " ", para).strip()
        if not flat:
            continue
        sids = sorted(s for s in set(SPEC_CITATION.findall(flat))
                      if s in metrics)
        if not sids:
            continue
        keys: Dict[str, Dict[str, float]] = {}
        for sid in sids:
            for k, v in metrics[sid].items():
                keys.setdefault(k, {})[sid] = v
        for k, by_sid in sorted(keys.items()):
            quoted: List[str] = []
            for m in re.finditer(r"\b%s\b" % re.escape(k), flat):
                window = flat[m.end():m.end() + _METRIC_WINDOW]
                for nm in _NUM.finditer(window):
                    lead = window[:nm.start()].rstrip()[-2:]
                    if lead.endswith("->"):
                        pass            # an arrow TRANSITION: a reading
                    elif any(c in lead for c in "<>="):
                        continue        # a bar, not a reading
                    quoted.append(nm.group(0))
            if not quoted:
                continue
            if any(_rounds_to(q, v)
                   for q in quoted for v in by_sid.values()):
                continue
            dedup = (page, k, tuple(sorted(set(quoted))))
            if dedup in seen:
                continue
            seen.add(dedup)
            out.append({
                "page": page, "key": k,
                "page_says": sorted(set(quoted)),
                "ledger_says": {sid: by_sid[sid] for sid in sorted(by_sid)},
            })
    return out


def render_metrics(mis: Optional[List[dict]] = None,
                   indent: str = "  ") -> str:
    """The `STEERING-METRIC-MISMATCH` block printed by `run status`."""
    if mis is None:
        mis = metric_mismatches()
    if not mis:
        return (f"{indent}STEERING-METRIC-MISMATCH — none: every ledger "
                f"metric quoted beside its spec on a steering\n"
                f"{indent}  page agrees with the current certificate (at the "
                f"quoted precision).\n")
    lines = [f"{indent}STEERING-METRIC-MISMATCH — {len(mis)} quoted "
             f"metric(s) disagree with the live certificate.\n"
             f"{indent}  Reporting-only, unfloored: the ledger is the "
             f"authority; a page may quote a number for\n"
             f"{indent}  another reason (a routing-time figure, a foreign "
             f"attempt), and this reader does not read intent.\n"]
    for m in mis:
        led = ", ".join(f"{sid} {v}" for sid, v in m["ledger_says"].items())
        lines.append(f"{indent}    {m['key']}  {m['page']} says "
                     f"{', '.join(m['page_says'])} — ledger says {led}\n")
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

    # ── the date-mismatch known answer (100th audit B1) ─────────────────────
    # Verbatim from `docs/PROGRESS.md` at `a01837f` (Review DAILY, 2026-09-18
    # 18:22) — the paragraph that told the owner a decision expiring that
    # night expired in seven days, published five hours after the register
    # was repaired and after the lesson about exactly this was written.
    # Frozen on purpose, like `_FIXTURE_PAGE`: the live page will be
    # corrected, and this must keep failing if the reader forgets how to see
    # it. The register here is what `DECISIONS_NEEDED.md` held that night.
    _DATE_FIXTURE = """
**1. `D30` — cited, not re-asked (`decide_by` 2026-09-25), with one new
fact.** The blackout that motivated it has ended — the builder ran 26 commits
today — so the *cost* half of that entry is no longer accruing. What is new
is procedural: `D30`'s own armed default fires on 2026-09-19 at an hour no
date-granular slot can enforce.

**2. `D28` — cited, not re-asked (`decide_by` 2026-09-21), and today is the
fourth piece of evidence.**

Both `D20` and `D30` carry `decide_by: 2026-09-18`. `D24`, which closed
2026-09-12, is not open and says nothing here. `D1.0` attempt 3 waited for
W37 (opened 2026-09-13); a spec id is not a decision id.
"""
    _reg = {"D30": "2026-09-18", "D28": "2026-09-21", "D20": "2026-09-18"}
    mis = text_date_mismatches(_DATE_FIXTURE, _reg, "fixture")
    want_mis = [("D30", ["2026-09-25"], "2026-09-18")]
    got_mis = [(m["id"], m["page_says"], m["register_says"]) for m in mis]
    if got_mis != want_mis:
        raise AssertionError(
            f"steering: date fixture flunked: {got_mis} != {want_mis} — the "
            f"one real misquote must flag; the fire-day sentence (09-19), the "
            f"agreeing citations (D28, D20/D30), the closed decision (D24) "
            f"and the spec id (D1.0) must all stay silent")
    txt = render_dates(mis, indent="")
    if "D30" not in txt or "2026-09-25" not in txt or "2026-09-18" not in txt:
        raise AssertionError("steering: the mismatch render must name the id "
                             "and both dates")
    if "none:" not in render_dates([], indent=""):
        raise AssertionError("steering: a clean read must still print — an "
                             "absent block is indistinguishable from an "
                             "absent check")

    # ── the metric-mismatch known answer (builder, 2026-09-23) ──────────────
    # Paragraph 1 is verbatim from `docs/PROGRESS.md` at `51c9d13` (Review
    # DAILY, 2026-09-22 06:55) — the stop-rule arming that quoted a
    # sixteen-day-dead reading as live and ordered it routed to the owner.
    # Ledger truth at the time of arming AND of this fixture: 1.0.
    # Paragraphs 2-5 are the four sentence shapes that must stay SILENT:
    # the arrow transition (the old number beside the agreeing new one), the
    # honest rounding, the bar quote, and the integer count. Frozen on
    # purpose, like the two fixtures above: the live page will be corrected,
    # and this must keep failing if the reader forgets how to see it.
    _METRIC_FIXTURE = """
0. **`ME.1` TODAY — a stop-rule fires at midnight.** The second branch is a
   real answer, not a failure: `distractor_abstention` reads **0.0000 ± 0.0**
   on 3 seeds at the spec's own unchanged 0.95 bar while
   `fabricated_abstention` is perfect.

`ME.1` a8: `distractor_abstention` 0.0000 -> 1.0000 while `cued_recall` is
0.85 ± 0.0136 — byte-identical to the failing attempt.

`ME.3`'s gain: `aggregation_qa_gain` 0.344 against the registered bar.

One conjunct is strictly harder on `ME.3`: `raw_answer_rate` >= 0.95.

`ME.1`'s `cued_recall` did not move on 3 seeds.

`T3.06`'s `task_cov_vs_random` −0.2333 — the TASK arm explores WORSE than
random.
"""
    _mreg = {
        "ME.1": {"distractor_abstention": 1.0, "fabricated_abstention": 1.0,
                 "cued_recall": 0.85, "cued_recall_std": 0.0136355},
        "ME.3": {"aggregation_qa_gain": 0.343733, "raw_answer_rate": 1.0},
        "T3.06": {"task_cov_vs_random": -0.233333},
    }
    mmis = text_metric_mismatches(_METRIC_FIXTURE, _mreg, "fixture")
    got_mm = [(m["key"], m["page_says"]) for m in mmis]
    want_mm = [("distractor_abstention", ["0.0", "0.0000"])]
    if got_mm != want_mm:
        raise AssertionError(
            f"steering: metric fixture flunked: {got_mm} != {want_mm} — the "
            f"one dead reading must flag; the arrow transition (0.0000 -> "
            f"1.0000, its new number agrees), the rounding (0.344 vs "
            f"0.343733), the bar (>= 0.95 vs measured 1.0) and the bare "
            f"count (on 3 seeds) must all stay silent")
    mtxt = render_metrics(mmis, indent="")
    if "distractor_abstention" not in mtxt or "0.0000" not in mtxt \
            or "1.0" not in mtxt:
        raise AssertionError("steering: the metric mismatch render must name "
                             "the key and both numbers")
    if "none:" not in render_metrics([], indent=""):
        raise AssertionError("steering: a clean metric read must still print "
                             "— an absent block is indistinguishable from an "
                             "absent check")


def launch_size(repo: Optional[Path] = None, days: int = LAUNCH_GROWTH_DAYS,
                _git=None) -> dict:
    """How close is the steering page to the ceiling that stops the builder?

    Returns `{bytes, cliff, ceiling, headroom_cliff, headroom_ceiling,
    per_day, days_to_cliff, days_to_ceiling, samples}`. `per_day` is fitted
    from git — the size at HEAD against the size `days` ago on the same path —
    because a growth rate asserted from memory is the cached number this
    module exists to replace. `None` where git cannot answer (shallow clone,
    file younger than the window); a missing rate prints as unknown rather
    than as zero, since zero growth is the one reading that would falsely
    reassure.
    """
    repo = _REPO if repo is None else repo
    path = repo / LAUNCH_PAGE
    if not path.exists():
        return {"bytes": None}
    size = len(path.read_bytes())

    per_day = samples = None
    try:
        run = _git or (lambda *a: subprocess.run(
            ("git", "-C", str(repo)) + a, capture_output=True, text=True,
            timeout=30).stdout)
        since = run("log", f"--since={days} days ago", "--format=%H",
                    "--", LAUNCH_PAGE).split()
        # The OLDEST commit in the window is the baseline; its PARENT state is
        # not needed — we want the size AT that commit, so the rate is over
        # the span actually observed, not an assumed one.
        if len(since) >= 2:
            old = since[-1]
            blob = run("show", f"{old}:{LAUNCH_PAGE}")
            when = run("log", "-1", "--format=%ct", old).strip()
            now = run("log", "-1", "--format=%ct", since[0]).strip()
            span = (int(now) - int(when)) / 86400.0
            if blob and span > 0.5:
                per_day = (size - len(blob.encode())) / span
                samples = len(since)
    except Exception:                                  # pragma: no cover
        per_day = samples = None                       # git silent -> unknown

    def to_go(limit):
        gap = limit - size
        if per_day is None or per_day <= 0:
            return None
        return gap / per_day

    return {"bytes": size, "cliff": LAUNCH_CLIFF, "ceiling": LAUNCH_CEILING,
            "headroom_cliff": LAUNCH_CLIFF - size,
            "headroom_ceiling": LAUNCH_CEILING - size,
            "per_day": per_day, "samples": samples,
            "days_to_cliff": to_go(LAUNCH_CLIFF),
            "days_to_ceiling": to_go(LAUNCH_CEILING)}


def render_size(st: Optional[dict] = None, indent: str = "  ") -> str:
    """The `STEERING-PAGE SIZE` block printed by `run status`.

    REPORTING-ONLY and UNFLOORED, by the Review's explicit instruction: a
    steering page has legitimate reasons to grow, and a gate here would let an
    instrument refuse the Review's own act. The judgement stays a human's; the
    VISIBILITY is an instrument's. It prints every time, including when there
    is plenty of room — an absent block is indistinguishable from an absent
    check, which is this module's standing rule.
    """
    st = launch_size() if st is None else st
    if st.get("bytes") is None:
        return (f"{indent}STEERING-PAGE SIZE — {LAUNCH_PAGE} not found; the "
                f"launcher's ceiling cannot be read.\n")
    rate = ("unknown" if st["per_day"] is None
            else f"{st['per_day']:+.0f} B/day")
    lines = [
        f"{indent}STEERING-PAGE SIZE — {LAUNCH_PAGE} is {st['bytes']} bytes; "
        f"{st['headroom_cliff']} below the {st['cliff']} EXEC CLIFF "
        f"(MAX_ARG_STRLEN), {st['headroom_ceiling']} below the "
        f"{st['ceiling']} self-imposed ceiling.",
        f"{indent}  Reporting-only, unfloored: a steering page has legitimate "
        f"reasons to grow and a gate here could refuse the Review's own act. "
        f"Past the cliff the builder does not read a degraded prompt — it "
        f"does not start, and the slot looks like an ordinary rc=126.",
        f"{indent}  growth {rate}"
        + (f" over {st['samples']} commit(s)" if st["samples"] else ""),
    ]
    # Keyed off the RATE, not off the derived day counts: a caller that
    # supplies stale day counts with no rate must still read as unknown.
    if st["per_day"] is not None and st.get("days_to_cliff") is not None:
        lines[-1] += (f"; at that rate the cliff is "
                      f"{st['days_to_cliff']:.0f} day(s) away, the ceiling "
                      f"{max(st['days_to_ceiling'], 0):.0f}.")
    else:
        lines[-1] += ("; headroom in DAYS is UNKNOWN — git could not date the "
                      "growth, and unknown is not zero.")
    if st["headroom_ceiling"] < 0:
        lines.append(f"{indent}  !! OVER the self-imposed {st['ceiling']} "
                     f"ceiling. Not a refusal; a fact for whoever edits next.")
    if st["headroom_cliff"] < 0:
        lines.append(f"{indent}  !! OVER THE EXEC CLIFF — the builder cannot "
                     f"launch. This is not a warning, it is the outage.")
    return "\n".join(lines) + "\n"


def _check_size() -> None:
    """Refuse to report from a size reader that flunks the known answer.

    THE FIXTURE IS THE OUTAGE, replayed — same contract as `_check` above.
    The known answers are the two measured states of 2026-09-20: 140331 bytes
    (over the cliff; the builder could not exec) and 85548 (the post-excision
    size the Review verified). A reader that does not call the first an outage
    is the reader that let nineteen slots die.
    """
    over = {"bytes": 140331, "cliff": LAUNCH_CLIFF, "ceiling": LAUNCH_CEILING,
            "headroom_cliff": LAUNCH_CLIFF - 140331,
            "headroom_ceiling": LAUNCH_CEILING - 140331,
            "per_day": 3976.0, "samples": 9,
            "days_to_cliff": (LAUNCH_CLIFF - 140331) / 3976.0,
            "days_to_ceiling": (LAUNCH_CEILING - 140331) / 3976.0}
    txt = render_size(over, indent="")
    if "OVER THE EXEC CLIFF" not in txt or "it is the outage" not in txt:
        raise AssertionError(
            "steering: the 2026-09-20 size (140331) must render as the "
            "outage it was, not as a warning")
    ok = dict(over, bytes=85548, headroom_cliff=LAUNCH_CLIFF - 85548,
              headroom_ceiling=LAUNCH_CEILING - 85548,
              days_to_cliff=(LAUNCH_CLIFF - 85548) / 3976.0,
              days_to_ceiling=(LAUNCH_CEILING - 85548) / 3976.0)
    txt = render_size(ok, indent="")
    if "OVER" in txt:
        raise AssertionError(
            "steering: the post-excision size (85548) is inside both limits "
            "and must not render as a breach")
    if "11 day(s)" not in txt:
        raise AssertionError(
            "steering: headroom must print in DAYS at the measured rate — "
            "(131072-85548)/3976 = 11, the figure the 107th audit derived by "
            f"hand. Got: {txt!r}")
    # Unknown must not read as zero, and must not read as safe.
    txt = render_size(dict(ok, per_day=None, samples=None), indent="")
    if "UNKNOWN" not in txt or "unknown is not zero" not in txt:
        raise AssertionError("steering: an unfitted growth rate must say so")
    # And the live read must actually reach the file.
    if launch_size().get("bytes") is None:
        raise AssertionError(f"steering: {LAUNCH_PAGE} is unreadable")


if __name__ == "__main__":  # pragma: no cover
    _check()
    _check_size()
    print(render(), end="")
    print(render_dates(), end="")
    print(render_size(), end="")
