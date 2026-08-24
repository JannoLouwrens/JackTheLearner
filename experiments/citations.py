"""Citation-marker audit: a `[V]` must be backed by a resolvable identifier.

WHY THIS EXISTS — the scar, per SYSTEM.md's "no new organ without a scar".

`docs/research/NEEDS_AND_DEATH.md` §1.2 shipped on 2026-08-22 as a table of
"believed primary sources" with **not one of them fetched**, and two of its rows
were live design constants. The citation pass of 2026-08-24 found three quiet
fictions in that one table:

  * the allostasis-evidence row asserted, on Zimmerman et al. (Nature 2016)'s
    authority, a cue-anticipation finding that paper had explicitly tested and
    FAILED to find;
  * "starvation ~3 weeks (Minnesota Starvation Experiment)" cited a study of
    24 weeks of SEMI-starvation with zero deaths, which supports neither that
    number nor any number;
  * the Borbely time constants were credited to Borbely (1982), which states no
    number at all.

None of those three is mechanically detectable — a machine cannot read a paper's
negative results. What IS mechanically detectable is the disease those three grew
in: a research doc where a verification marker is a **typed character** rather
than a **fetched source**. This module makes `[V]` cost something. A claim may
still be wrong; it may no longer be unfalsifiable-by-inspection.

THE RULE, and it is deliberately the weakest rule that still bites:

    Any block bearing `[V]` must carry a resolvable identifier — a DOI, an
    arXiv ID, a PMID/PMCID, an ISBN, or an RFC number — in the same block.

A "block" is one table row (a line starting with `|`) or one blank-line-delimited
paragraph, because that is the unit a reader's eye treats as one citation.

`[V-abs]` (metadata + abstract confirmed, full text NOT read) is reported
separately and is NOT a violation: it is the honest marker for a paywalled
source, and punishing it would push authors back toward the bare `[V]` this
module exists to discourage.

Run:  python -m experiments.citations            # audit, exit 1 on violations
      python -m experiments.citations --self-test # known-answer fixtures
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESEARCH = REPO / "docs" / "research"

# `[V]` exactly — not `[V-abs]`, not `[VOID]`, not `[Verified by ...]`.
VERIFIED_RE = re.compile(r"\[V\](?!-)")
ABSTRACT_ONLY_RE = re.compile(r"\[V-abs\]")

# What counts as resolvable. Each of these can be pasted into a resolver and
# either returns a paper or does not — which is the whole point.
IDENTIFIER_RES = (
    re.compile(r"\b10\.\d{4,9}/\S+"),                  # DOI
    re.compile(r"\barxiv[:\s/]*\d{4}\.\d{4,5}", re.I),  # arXiv, new style
    re.compile(r"\barxiv[:\s/]*[a-z-]+/\d{7}", re.I),   # arXiv, old style
    # A BARE arXiv ID. `UNIFIED_BRAIN_BAKEOFF.md:52` and `LEARNING_CORE.md`
    # both declare this as their convention ("IDs marked [V] were fetched from
    # arxiv.org"), and `2410.16424` resolves exactly as well as `arXiv:2410.16424`
    # does. Recognising it is fitting the instrument to the corpus, not
    # loosening the rule: the shape is specific enough that version strings
    # (2.5.1) and years (2026) do not match it.
    re.compile(r"(?<![\d.])\d{4}\.\d{4,5}(?![\d.])"),
    re.compile(r"\bPMID[:\s]*\d+", re.I),
    re.compile(r"\bPMC\d{5,}"),
    re.compile(r"\bISBN[:\s]*[\d\-Xx]{10,}", re.I),
    re.compile(r"\bRFC\s*\d+", re.I),
)

# A marker written in backticks is a MENTION, not a USE — the doc is talking
# about the convention (a legend, a hygiene note) rather than certifying a
# claim. `[V]` in prose about markers owes no citation; **[V]** attached to a
# claim does. Without this the legend that DEFINES the marker trips the check
# that enforces it.
MENTION_RE = re.compile(r"`\[V\]`")


def has_identifier(text: str) -> bool:
    return any(r.search(text) for r in IDENTIFIER_RES)


def blocks(md: str):
    """Yield (start_line_1indexed, text) for each citation-sized block.

    A table row is its own block; everything else groups into blank-line
    delimited paragraphs. Fenced code is skipped — a `[V]` inside a code fence
    is sample text, not a claim.
    """
    lines = md.splitlines()
    in_fence = False
    para: list[str] = []
    para_start = 0

    def flush():
        nonlocal para, para_start
        if para:
            yield_val = (para_start, "\n".join(para))
            para = []
            return yield_val
        return None

    for i, line in enumerate(lines, start=1):
        if line.lstrip().startswith("```"):
            got = flush()
            if got:
                yield got
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if line.lstrip().startswith("|"):
            got = flush()
            if got:
                yield got
            yield (i, line)
            continue
        if not line.strip():
            got = flush()
            if got:
                yield got
            continue
        if not para:
            para_start = i
        para.append(line)

    got = flush()
    if got:
        yield got


def audit_text(md: str):
    """Return (violations, abstract_only) as lists of (line_no, excerpt)."""
    violations, abstract_only = [], []
    for line_no, text in blocks(md):
        if ABSTRACT_ONLY_RE.search(text):
            abstract_only.append((line_no, excerpt(text)))
        if not VERIFIED_RE.search(text):
            continue
        # Every marker in this block is backticked -> the block discusses the
        # convention rather than invoking it.
        if len(MENTION_RE.findall(text)) == len(VERIFIED_RE.findall(text)):
            continue
        if not has_identifier(text):
            violations.append((line_no, excerpt(text)))
    return violations, abstract_only


def excerpt(text: str, width: int = 110) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


# ------------------------------------------------------------------ ratchet
#
# 74 unbacked markers already existed when this module was written. A guard
# that fails on all of them from day one is a guard nobody can keep green, and
# a guard nobody keeps green is decoration. So this is a RATCHET, not a gate:
# the debt is recorded per file, it may shrink, and it may never grow. Adding
# a bare `[V]` to any research doc from 2026-08-24 onward fails the check.
#
# To tighten: fix markers, re-run `--baseline`, paste the new numbers here.
# Lowering a number is the only edit to this dict that does not need a reason
# in the commit message. RAISING one is weakening a threshold — SYSTEM.md law 4
# applies, and the number must not move without the reason written down.
BASELINE = {
    "D1_CONTROL_ARCHITECTURE.md": 3,
    "FROZEN_VS_PLASTIC.md": 18,
    "HEARING_BAKEOFF.md": 20,
    "LANGUAGE_GROUNDING.md": 1,
    "LEARNING_CORE.md": 20,
    "NEEDS_AND_DEATH.md": 11,
    "UNIFIED_BRAIN_BAKEOFF.md": 1,
}


def audit_repo(paths=None):
    paths = sorted(RESEARCH.glob("*.md")) if paths is None else paths
    all_v, all_a = {}, {}
    for p in paths:
        v, a = audit_text(p.read_text(encoding="utf-8"))
        if v:
            all_v[p] = v
        if a:
            all_a[p] = a
    return all_v, all_a


# ---------------------------------------------------------------- self-test

FIXTURES = [
    # (name, text, expect_violation)
    ("doi backs the marker",
     "| a claim | a source, doi 10.1038/nature18950 | **[V]** |", False),
    ("pmid backs the marker",
     "| a claim | Everson 1989, PMID 2538906 | **[V]** |", False),
    ("arxiv backs the marker",
     "The result holds (arXiv:2109.06780) **[V]**.", False),
    ("BARE MARKER — the disease",
     "| a claim | Believed: Smith et al. (2019) | **[V]** |", True),
    ("bare marker in prose — the disease",
     "This is standard in the literature (Song et al., JNER 2021) **[V]**.", True),
    ("[V-abs] without an identifier is NOT a [V] violation",
     "| a claim | Believed: Smith et al. (2019) | **[V-abs]** |", False),
    ("VOID is not a verification marker",
     "The spec recorded [VOID] and no source is owed.", False),
    ("identifier in a NEIGHBOURING row does not count",
     "| row one | doi 10.1000/x | ok |\n| row two | no source | **[V]** |", True),
    ("a [V] inside a code fence is sample text, not a claim",
     "```\nstatus = '[V]'   # no citation owed here\n```", False),
    ("a bare arXiv ID is an identifier (the corpus's own convention)",
     "**ReViP (2601.16667 [V])** mitigates false completion.", True is False),
    ("a backticked marker is a MENTION — the legend owes no citation",
     "Marker meanings: `[V]` = the primary source was fetched and read.", False),
    ("but a backticked legend PLUS a real bare use still trips",
     "`[V]` means fetched; and the trunk binds senses **[V]**.", True),
    ("a version string is not an arXiv ID",
     "We pin torch 2.5.1 for sm_60 **[V]**.", True),
]


def self_test() -> int:
    bad = 0
    for name, text, expect in FIXTURES:
        v, _ = audit_text(text)
        got = bool(v)
        ok = got == expect
        bad += 0 if ok else 1
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name}"
              f"   (expected violation={expect}, got={got})")
    print(f"\n  {len(FIXTURES) - bad}/{len(FIXTURES)} known-answer fixtures pass")
    return 1 if bad else 0


def main(argv) -> int:
    if "--self-test" in argv:
        return self_test()

    violations, abstract_only = audit_repo()
    n_v = sum(len(x) for x in violations.values())
    n_a = sum(len(x) for x in abstract_only.values())
    verbose = "--list" in argv or "--baseline" in argv

    if verbose and violations:
        print("UNBACKED VERIFICATION MARKERS — `[V]` with no resolvable "
              "identifier in the same block:\n")
        for path, rows in violations.items():
            print(f"  {path.relative_to(REPO)}")
            for line_no, text in rows:
                print(f"    :{line_no}  {text}")
            print()

    if "--baseline" in argv:
        print("  paste into BASELINE:")
        for path, rows in sorted(violations.items()):
            print(f'    "{path.name}": {len(rows)},')
        return 0

    # The ratchet.
    grew, shrank = [], []
    for path in sorted(RESEARCH.glob("*.md")):
        now = len(violations.get(path, []))
        was = BASELINE.get(path.name, 0)
        if now > was:
            grew.append((path.name, was, now))
        elif now < was:
            shrank.append((path.name, was, now))

    print(f"  {n_v} unbacked `[V]` marker(s) against a baseline of "
          f"{sum(BASELINE.values())}; "
          f"{n_a} honest `[V-abs]` (abstract-only) marker(s).")

    for name, was, now in shrank:
        print(f"  BETTER  {name}: {was} -> {now}. Tighten BASELINE "
              f"(`--baseline` prints the new dict).")

    if grew:
        print("\n  RATCHET BROKEN — a research doc gained unbacked `[V]` "
              "markers:\n")
        for name, was, now in grew:
            print(f"    {name}: baseline {was}, now {now}")
        print("\n  A `[V]` asserts a primary source was FETCHED AND READ. "
              "Either record its\n  DOI/PMID/arXiv ID, or downgrade it to "
              "`[V-abs]` (metadata + abstract only)\n  or `[!]` (not "
              "verified). Run with --list to see them.")
        return 1

    print("  RATCHET HELD — no research doc gained an unbacked marker.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
