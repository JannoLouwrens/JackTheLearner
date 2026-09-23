"""Does every finding the field watch publishes have an owner and a clock?

THE SCAR (96th audit, 2026-09-14, and it is the third instance of the same
class). Week 7's field watch produced two findings in one commit at 05:57.
§6b — five ME specs certifying 0.819–0.944 against a 0.95 bar — carried a
formula and a fixture constant; it was diagnosed, repaired and re-bought PASS
by 06:24. §6 — the mandatory collapse diagnostic under the arm holding the
Learning-core seat is computed NOWHERE, so the seat's declared silent-failure
guard was unarmed when `D10` seated it BY VERDICT — got nothing: no queue row,
no `DUE:`, no decision entry, for six hours, until a Review sitting happened
to read the page. `grep -rn "FIELD_WATCH" --include=*.py` returned zero hits:
the one organ that looks OUTWARD wrote its findings to a page no instrument
reads, so they survived on somebody's habit. This project has now paid that
three times — `PROGRESS.md`'s owner-asks (repaired by `UNROUTED-OWNER-ASK`),
`REVIEW_QUEUE.md` (repaired by its own reader, 08-31 B4), and `FIELD_WATCH.md`
(this module).

WHAT IT CHECKS — ownership, never content. The general rule from the audit:
findings are consumed in order of how MECHANISABLE they are, not how much
they MATTER, and the ordering is invisible from inside. The repair is NOT to
judge the science (that is `D27`'s 104-of-107 screen again); it is to
mechanise **whether a finding has reached a desk** — a parse over two files
that says nothing about the finding's content. A finding reaches a desk when
a `REVIEW_QUEUE.md` row or a `DECISIONS_NEEDED.md` entry either CITES it by
section (the live row's own form: *"field watch wk7 §6"*) or QUOTES a
verbatim span of it (the same 6-gram shingle rule `decisions.owner_asks`
uses on the facing problem, imported from there so the two readers cannot
drift apart — but see the threshold below: ONE shared shingle stopped being
enough the day it was measured).

THE QUOTATION THRESHOLD, AND THE MEASUREMENT THAT SET IT (builder,
2026-09-23, executing `fieldwatch-quotation-channel-is-0-for-5`). At one
shared shingle the channel measured 0-true-for-5 routings at n = 5 — every
hit was stock house-style English (*"it is the same shape as"*) or the row's
own slug, because `_queue_chunks` starts each chunk at `ROUTED: <slug>`, so a
finding that NAMES the row it is distinguishing itself from reads as owned by
it. Measured over every (finding x desk-chunk) pair on the live 2026-09-23
corpus, with `ROUTED:` header lines stripped from chunks before shingling:
every spurious pair overlaps by <= 4 shingles (stock phrases 1-2; a row
TITLE mentioned in a finding's body 3-4, bounded because slugs and titles
are ~8-10 words), while every true quotation measured >= 12 (fixture 12,
live 18 and 66). `MIN_QUOTE_OVERLAP = 6` sits between with ~2x margin on
both sides: it demands the verbatim-span equivalent of >= 11 consecutive
words, which is more than any slug or title can carry and less than half the
smallest real quote observed. Post-fix rate on the same corpus: 0 spurious
of 18 sub-threshold pairs suppressed, 1 genuine quotation retained (and
superseded by its own citation). The trade is deliberate and in the safe
direction for a reporting-only counter: a re-worded routing that keeps only
one clause now reads UNROUTED (a human looks at a false red), instead of a
no-owner finding reading ROUTED (nobody looks at a false green — the exact
scar of the 96th audit). The threshold and the stripping both live HERE, not
in `decisions._shingles`: the shared helper also serves the owner-ask
reader, and re-tuning it blind was explicitly forbidden by the routing row.

WHAT A FINDING IS, and why that is a heuristic. Sections whose heading
declares itself one — `## 6. A FINDING IN OUR OWN ARTIFACTS — …` — i.e. any
`##`/`###` heading containing the word "finding", PLUS every `###` subsection
inside such a `##` region (`### 6b — a second, smaller one …` declares no
"finding" in its own words; it is one because the scout filed it under §6).
Each finding owns its span to the next heading, so a parent never inherits
or donates its child's routing.
Nominations (§2) and the watchlist (§3) are deliberately NOT in the
population: they are consumed by the Review's weekly disposition pass (§4
exists to track exactly that), so a reader that flagged them would be
counting a pipeline that already has a reader. The population is selected by
the scout's own headings — SYSTEM.md's "a checker reading a population
somebody else selected" caveat applies and is accepted: the scout cannot make
an unrouted finding look routed by rewording it, only by not calling it a
finding at all, and week 7's own §8 makes finding-headings standing practice.

REPORT-ONLY AND UNFLOORED, by the auditor's explicit instruction (96th audit
FTB 1): `D27`'s reasoning applies unchanged — do not floor a counter whose
false-positive rate has not been measured and written down. The expected
false-positive shape is a finding that was DISCHARGED (repaired in code, or
attached to a row that does not quote it) rather than routed; discharge is a
property of commits, which do not quote, and this reader does not pretend to
see it. Every class is counted and printed (`T0.31`: a reader that counts one
class pays a repair that lowers its own number): total findings, ROUTED by
citation, ROUTED by quotation, UNROUTED.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent

FIELD_WATCH_PAGE = "docs/FIELD_WATCH.md"

# The two desks the 96th audit named. LESSONS.md and commit messages also
# absorb findings, but neither carries an owner or a clock, which is the
# property under test.
DESK_QUEUE = "docs/REVIEW_QUEUE.md"
DESK_DECISIONS = "docs/DECISIONS_NEEDED.md"

# `##` or `###` headings that declare a finding. Case-insensitive on purpose:
# week 7 wrote "A FINDING IN OUR OWN ARTIFACTS" and "a second, smaller one" —
# the marker is the word, not the shouting.
_HEADING = re.compile(r"^(#{2,3})\s+(.*)$")
_FINDING_WORD = re.compile(r"\bfinding\b", re.I)

# "## 6. TITLE" / "### 6b — TITLE" -> section key "§6" / "§6b".
_SECTION_KEY = re.compile(r"^(\d{1,2}[a-z]?)\b")

# The sweep's own week number ("Scout: field watch, week 7.") — the citation
# channel needs it because a desk citing "wk6 §6" is routing LAST sweep's
# finding, and this page is rewritten wholesale every week.
_WEEK = re.compile(r"field watch,?\s+week\s+(\d+)", re.I)

# A queue row opens at column 0; its chunk runs to the next row. Same shape
# as `review_queue._ROUTED`, restated here only because that module's parser
# returns typed fields and this one needs the raw text for quotation checks.
_ROW = re.compile(r"^ROUTED:\s*([a-z0-9][a-z0-9-]*)\s*\|", re.M)

# Stripped from a chunk before the quotation match: the header line carries the
# row's slug, and a finding naming the row it is distinguishing itself FROM
# must not read as quoted BY that row (the measured self-match class).
_ROW_HEADER = re.compile(r"^ROUTED:[^\n]*$", re.M)

# Minimum shared shingles for the quotation channel. Measured 2026-09-23:
# spurious pairs (stock phrases, slug/title mentions) cap at 4 after header
# stripping; true quotations start at 12. See the module docstring.
MIN_QUOTE_OVERLAP = 6


def findings(text: str) -> List[dict]:
    """Finding sections of the current sweep, in order.

    `{key, level, title, text, week}` — `key` is `§6`-style when the heading
    starts with a section number, else the heading text. A `##` finding's own
    span ends at the NEXT heading of any level, so a `### 6b` child inside it
    is its own finding and never inherits (or donates) its parent's routing.
    """
    m = _WEEK.search(text or "")
    week = int(m.group(1)) if m else None
    out: List[dict] = []
    cur: Optional[dict] = None
    in_region = False        # inside a ##-finding's region, until the next ##
    for ln in (text or "").splitlines():
        h = _HEADING.match(ln)
        if h:
            if cur is not None:
                out.append(cur)
                cur = None
            level, title = len(h.group(1)), h.group(2).strip()
            if level == 2:
                in_region = bool(_FINDING_WORD.search(title))
            is_finding = (bool(_FINDING_WORD.search(title))
                          or (level == 3 and in_region))
            if is_finding:
                km = _SECTION_KEY.match(title)
                cur = {"key": f"§{km.group(1)}" if km else title[:40],
                       "level": level,
                       "title": title[:120], "lines": [], "week": week}
            continue
        if cur is not None:
            cur["lines"].append(ln)
    if cur is not None:
        out.append(cur)
    for f in out:
        f["text"] = "\n".join(f.pop("lines"))
    return out


def _queue_chunks(text: str) -> Dict[str, str]:
    """`{row slug: full chunk text}` — a row and everything under it."""
    hits = list(_ROW.finditer(text or ""))
    out: Dict[str, str] = {}
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        # repeated slugs accumulate, like decisions._entries — every dated
        # addendum under a re-used slug is still that row's desk
        out[m.group(1)] = out.get(m.group(1), "") + text[m.start():end]
    return out


def _cites(finding: dict, desk_text: str) -> bool:
    """The live routing form: *"field watch wk7 §6"* (or "week 7 §6").

    Anchored to THIS sweep's week so a stale citation cannot route a fresh
    finding, and `§6` must not be claimed by `§6b` or vice versa.
    """
    if finding["week"] is None or not finding["key"].startswith("§"):
        return False
    num = re.escape(finding["key"][1:])
    wk = finding["week"]
    pat = re.compile(
        rf"(?:wk\s*{wk}|week\s+{wk})[^\n]{{0,40}}§\s*{num}(?![0-9a-z])", re.I)
    return bool(pat.search(desk_text or ""))


def resolve(finds: List[dict], queue_text: str,
            decisions_text: str) -> List[dict]:
    """Adds `routed = (how, where) | None` to each finding.

    Citation beats quotation and queue beats decisions, in that fixed order,
    so the attribution is deterministic (`decisions._reaches_a_desk`'s rule).
    """
    from .decisions import _shingles
    desks: List[Tuple[str, str, str]] = (
        [("queue-row", k, v) for k, v in sorted(_queue_chunks(queue_text).items())]
        + [("decision", k, v) for k, v in sorted(_entries_of(decisions_text).items())])
    quotable = [(kind, key, _shingles(_ROW_HEADER.sub("", text)))
                for kind, key, text in desks]
    for f in finds:
        f["routed"] = None
        for kind, key, text in desks:
            if _cites(f, text):
                f["routed"] = (f"cited by {kind}", key)
                break
        if f["routed"]:
            continue
        sh = _shingles(f["text"])
        for kind, key, desk_sh in quotable:
            if len(sh & desk_sh) >= MIN_QUOTE_OVERLAP:
                f["routed"] = (f"quoted by {kind}", key)
                break
    return finds


def _entries_of(decisions_text: str) -> Dict[str, str]:
    from .decisions import _entries
    return _entries(decisions_text or "")


def read(repo: Optional[Path] = None) -> List[dict]:
    repo = _REPO if repo is None else repo
    page = repo / FIELD_WATCH_PAGE
    if not page.exists():
        return []
    return resolve(findings(page.read_text()),
                   (repo / DESK_QUEUE).read_text()
                   if (repo / DESK_QUEUE).exists() else "",
                   (repo / DESK_DECISIONS).read_text()
                   if (repo / DESK_DECISIONS).exists() else "")


def render(finds: Optional[List[dict]] = None, indent: str = "  ") -> str:
    """The block `run status` prints. Every class counted (`T0.31`)."""
    finds = read() if finds is None else finds
    if not finds:
        return (f"{indent}FIELD-WATCH FINDINGS — no finding-headed section on "
                f"{FIELD_WATCH_PAGE}. Fine for a\n{indent}  sweep that found "
                f"nothing in our own artifacts; the nominations pipeline has "
                f"its own reader.\n")
    cited = [f for f in finds if f["routed"] and f["routed"][0].startswith("cited")]
    quoted = [f for f in finds if f["routed"] and f["routed"][0].startswith("quoted")]
    unrouted = [f for f in finds if not f["routed"]]
    wk = finds[0]["week"]
    head = (f"{indent}FIELD-WATCH FINDINGS — week {wk}: {len(finds)} finding "
            f"section(s); {len(cited)} cited, {len(quoted)} quoted,\n"
            f"{indent}  {len(unrouted)} UNROUTED-FIELD-FINDING. Reporting-only, "
            f"unfloored (96th audit FTB 1 / D27's\n"
            f"{indent}  reasoning): a finding may be legitimately discharged in "
            f"code, which no desk file shows.\n")
    lines = [head]
    for f in finds:
        if f["routed"]:
            how, where = f["routed"]
            lines.append(f"{indent}  {f['key']:5s} ROUTED — {how} "
                         f"`{where}`\n")
        else:
            lines.append(f"{indent}  {f['key']:5s} UNROUTED-FIELD-FINDING — "
                         f"{f['title'][:90]}\n{indent}        on no desk: no "
                         f"queue row cites or quotes it, no decision entry "
                         f"does, and the page\n{indent}        is rewritten "
                         f"weekly. Repair: route it (quoting it) or record "
                         f"its discharge where a\n{indent}        desk can "
                         f"see it.\n")
    return "".join(lines)


# ── the known answer ────────────────────────────────────────────────────────
#
# Verbatim spans from `docs/FIELD_WATCH.md` week 7 (2026-09-14, `9075d58`'s
# subject matter) and from the live `REVIEW_QUEUE.md` row that routed §6 at
# 06:49 the same morning. Frozen on purpose, like `steering._FIXTURE_PAGE`:
# the live §6 IS the scar this module exists for, kept executable so deleting
# either routing rule turns this red.
_FIXTURE_SWEEP = """
**Scout:** field watch, week 7. **Seven days since week 6** (2026-09-07).

## 2. NOMINATIONS

1. **N1 — a nomination is not a finding** and must not be parsed as one.

## 6. A FINDING IN OUR OWN ARTIFACTS — `A4`'s mandatory collapse diagnostic is declared in the governing document and computed nowhere in the repository

> *"Collapse is the failure mode and it is silent, so A4 carries a **mandatory
> diagnostic: effective rank and per-dimension variance of the latent must be
> reported every 1,000 decisions**, and a collapse (rank below a pre-registered
> floor) is `Status.VOID` for A4, not a good loss curve."*

`A4`'s declared VOID condition was never computable.

### 6b — a second, smaller one, free: the abstention conjunct that just shipped is certified below its own bar

A **perfect** abstention run over `m` labelled negatives certifies only
`a_L = γ^(1/m)` at confidence `1 − γ`, so `m ≥ 59` is needed at γ = 0.05.

## 7. What this report does NOT claim

- Findings above are the population; this section's heading word "claim" is
  not "finding" and this section must not be parsed.
"""

_FIXTURE_QUEUE_CITED = """\
ROUTED: a4-mandatory-collapse-diagnostic-is-declared-and-computed-nowhere | 2026-09-14 | `9075d58` (field watch wk7 §6, greps reproduced by the builder at ~06:2x and by this desk at ~07:3x) | OPEN
    DUE: 2026-09-18 | a fork owed by the Review, and it is three-way.
"""

# The quotation channel, isolated: a row that QUOTES §6's blockquote verbatim
# but never writes "wk7 §6".
_FIXTURE_QUEUE_QUOTED = """\
ROUTED: some-other-slug | 2026-09-14 | anywhere | OPEN
    The document promises that effective rank and per-dimension variance of
    the latent must be reported every 1,000 decisions, and nothing does.
"""

# The 0-for-5 scar (routed 2026-09-21, fixed 2026-09-23), executable: a chunk
# that touches §6 through every measured spurious mechanism at once — the
# finding's own words as the row SLUG (4 shingles, dies to header stripping),
# a stock house-style 6-gram, and a short title-length mention (2 shingles) —
# and still MUST NOT route, because none of that is a desk quoting a finding.
_FIXTURE_QUEUE_SPURIOUS = """\
ROUTED: effective-rank-and-per-dimension-variance-of-the-latent | 2026-09-23 | anywhere | OPEN
    The collapse is the failure mode and the diary already knows it; a floor
    that must be reported every 1,000 decisions is someone else's row.
"""


def _check() -> None:
    """Refuse to report from a reader that flunks the known answer.

    The known answer is the 96th audit's scar replayed exactly: two findings
    on the week-7 page, §6 routed by the citation the Review actually wrote,
    §6b on no desk. Plus each channel isolated, and the two populations that
    must stay silent (a nomination; a heading whose word is not "finding").
    """
    finds = findings(_FIXTURE_SWEEP)
    got = [(f["key"], f["level"], f["week"]) for f in finds]
    if got != [("§6", 2, 7), ("§6b", 3, 7)]:
        raise AssertionError(f"fieldwatch: section parse flunked: {got}")
    if "6b" in finds[0]["text"] or "abstention" in finds[0]["text"]:
        raise AssertionError("fieldwatch: §6's span leaked into its child — "
                             "a parent must not inherit its child's routing")

    # both channels dead -> both findings unrouted
    empty = resolve([dict(f) for f in finds], "", "")
    if [f["routed"] for f in empty] != [None, None]:
        raise AssertionError("fieldwatch: empty desks must route nothing")

    # the live scar: §6 routed by citation, §6b on no desk
    live = resolve([dict(f) for f in finds], _FIXTURE_QUEUE_CITED, "")
    if live[0]["routed"] != (
            "cited by queue-row",
            "a4-mandatory-collapse-diagnostic-is-declared-and-computed-nowhere"):
        raise AssertionError(f"fieldwatch: §6 must route via the citation the "
                             f"Review wrote: {live[0]['routed']}")
    if live[1]["routed"] is not None:
        raise AssertionError("fieldwatch: §6b reached no desk and must say so "
                             "— this is the 96th audit's finding, executable")

    # the quotation channel, isolated
    q = resolve([dict(f) for f in finds], _FIXTURE_QUEUE_QUOTED, "")
    if q[0]["routed"] != ("quoted by queue-row", "some-other-slug"):
        raise AssertionError(f"fieldwatch: verbatim quotation must route: "
                             f"{q[0]['routed']}")

    # the 0-for-5 scar: slug + stock phrase + title mention must NOT route,
    # and the fixture must actually exercise both halves of the repair —
    # over threshold with the header, under it (but non-empty) without.
    from .decisions import _shingles
    sh6 = _shingles(finds[0]["text"])
    raw = len(sh6 & _shingles(_FIXTURE_QUEUE_SPURIOUS))
    stripped = len(sh6 & _shingles(_ROW_HEADER.sub("", _FIXTURE_QUEUE_SPURIOUS)))
    if not (raw >= MIN_QUOTE_OVERLAP > stripped > 0):
        raise AssertionError(
            f"fieldwatch: spurious fixture drifted — raw={raw} "
            f"stripped={stripped} vs threshold {MIN_QUOTE_OVERLAP}; it must "
            f"trip the old one-shingle rule and clear neither repair alone")
    fp = resolve([dict(f) for f in finds], _FIXTURE_QUEUE_SPURIOUS, "")
    if fp[0]["routed"] is not None:
        raise AssertionError(
            f"fieldwatch: a slug/stock-phrase chunk routed §6 again — the "
            f"0-for-5 false green is back: {fp[0]['routed']}")

    # a stale citation (wrong week) must not route a fresh finding
    stale = _FIXTURE_QUEUE_CITED.replace("wk7", "wk6")
    s = resolve([dict(f) for f in finds], stale, "")
    if s[0]["routed"] is not None and s[0]["routed"][0].startswith("cited"):
        raise AssertionError("fieldwatch: a wk6 citation routed a wk7 finding")

    text = render(live, indent="")
    if "UNROUTED-FIELD-FINDING" not in text or "§6b" not in text:
        raise AssertionError("fieldwatch: the unrouted class must print")
    if "1 cited" not in text or "0 quoted" not in text:
        raise AssertionError(f"fieldwatch: every class is counted (T0.31):\n"
                             f"{text}")


if __name__ == "__main__":  # pragma: no cover
    _check()
    print(render(), end="")
