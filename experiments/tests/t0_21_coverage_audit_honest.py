"""T0.21 — the GOAL.md coverage audit can see both of its bad cases.

`experiments/coverage.py` answers the question that outranks `run status`: is
the ladder the RIGHT ladder? It has now been wrong in both directions.

  Direction 1, a false NEGATIVE: `BA.01` was registered to close the `balance`
  hole and the balance regex did not match its title. Found in a day, because
  its author was looking straight at it.

  Direction 2, a false POSITIVE: the regex kept granting coverage on its own,
  and `shelter/building` — the owner's own image of success — read 4 specs / 1
  PASS off the letters in "ho-NEST", "c-LIMB-able" and "bo-DIES". That survived
  two more days, because nobody goes looking for coverage they believe they
  already have.

The repair is structural rather than a better pattern: a regex hit is a
NOMINATION and only a `COVERS:` declaration is coverage. This battery is what
makes that claim falsifiable, and its P3/P4 are the known-answer test
`docs/LESSONS.md` prescribes — feed the instrument the case you already know is
broken and require it to say so.

The control is the organ that FAILED, kept executable: the pre-2026-08-10
patterns, verbatim, granting coverage by title regex. Under it "The honest
baseline" is shelter coverage and a declared spec with an unrelated title is
not. Note that word boundaries alone would NOT have saved it — PG.1's
"physically sound" still matches `hearing`, and P3 carries that case
deliberately so nobody mistakes the cheap half of the fix for the fix.

No physics, no training, no ledger writes: the fixtures are registry dicts
built in-process, so the numbers hold still while the RULE varies. Same shape
as T0.19 and T0.20.

P11 (28th audit B1) closes the THIRD leak: a PARKED spec counted as coverage.
On 2026-08-25 00:11 the loop retired `SH.01` by its own pre-registered rule —
the right call — and `shelter/building` and `thermal (kills)`, two of the four
original misses this tool exists for, silently lost their last runnable claim
while `coverage.py` printed `0 commitment(s) with NO declared spec` and exited
0. Blocked is a queue position; parked is a retirement; the tool credited
both. P11 requires: a `PARKED:`-marked claim spec buys no coverage and is
reported in the row's `parked` map; a commitment whose only claims are parked
reads claim-dead; a dateless `PARKED:` is REPORTED, never dropped (an
unparseable retirement silently keeps counting as coverage — the direction
nobody audits); a backticked prose mention is neither; and the live registry's
markers are all well-formed. The organ that failed is kept executable as
`report(credit_parked=True)` and must reproduce the leak.

P10 (26th audit B2) extends the repair to the marker's OTHER copy. The house
style also writes `COVERS:` into test-file docstrings, and `declarations()`
reads `Spec.notes` only — so that copy was read by no instrument, and it
rotted invisibly: one file's kind flipped against the registry, one declared
a commitment that does not exist, one kept a family's old name. P10 parses
every docstring under `experiments/tests/` with the registry's own grammar
and requires each marker to be well-formed AND backed, pair-for-pair (kind
included), by its own spec's registry declaration. A convention enforced at
one site and written at two rots at the unenforced site (LESSONS.md,
2026-08-24); this points the checker at the second site.
"""
from __future__ import annotations

import ast
import re
from dataclasses import replace
from pathlib import Path

from ..coverage import (COMMITMENTS, _claim_dead, declarations, parked,
                        report)
from ..protocol import Ledger, Status, module_path_for, run_spec
from ..registry import BY_ID

SPEC_ID = "T0.21"

# The battery is ABOUT coverage.py, so its certificate must die when that file
# changes (same reasoning as PG.6 hashing playground.py: a claim about X that
# survives edits to X is a certificate about nothing).
IMPL_DEPS = ["experiments/coverage.py"]

# The four commitments the 2026-08-10 hand audit found at ZERO specs. They are
# why coverage.py exists, so the battery asserts the list still NAMES them
# however their coverage moves.
CONSTITUTIONAL_GAPS_2026_08_10 = ("thermal (kills)", "shelter/building",
                                  "death & retry", "damage/nociception")

# THE ORGAN THAT FAILED, kept as executable code: the patterns as they stood
# before this file existed, with no word boundaries. Reproducing them verbatim
# is what makes the control a control rather than a tidied restatement
# (T0.08 property 5).
LEGACY_PATTERNS = {
    "touch/contact":      r"touch|tactile|contact",
    "proprioception":     r"propriocept|body schema|limb",
    "death & retry":      r"death|dies|lethal|surviv|statue",
    "shelter/building":   r"shelter|build|construct|nest",
    "hearing":            r"audio|acoustic|sound|hear|binaural",
}


TESTS_DIR = Path(__file__).resolve().parent


def _docstring_covers(source: str) -> tuple[set, list]:
    """`(pairs, bad)` for every `COVERS:` marker in a file's DOCSTRINGS,
    parsed with the SAME grammar `declarations()` applies to registry notes.

    Docstrings only, by AST: a marker inside a code string (this battery's
    own fixtures) is data, not a declaration, and a backticked prose mention
    is already excluded by DECLARATION itself (P5)."""
    texts = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef,
                             ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                texts.append(doc)
    fake = replace(BY_ID["T0.01"], id="ZZ.doc", notes="\n".join(texts))
    dec, bad = declarations({"ZZ.doc": fake})
    return ({(c, kind) for c, lst in dec.items() for _sid, kind in lst},
            bad)


def _docstring_audit(files: dict, by_id: dict,
                     spec_of: dict[str, str]) -> list[str]:
    """One problem per docstring `COVERS:` marker that is malformed or that
    its own spec's registry entry does not declare, pair-for-pair — the kind
    must match too. The registry is the copy `declarations()` reads; a
    docstring marker it does not back is decoration wearing a claim's face.

    `spec_of` maps file name -> spec id. The live caller builds it from
    `module_path_for` — the ONE rule for which file implements a spec —
    never from parsing the file itself. A marker in a file no spec owns
    (a helper module) is pure decoration and flagged as such."""
    declared, _ = declarations(by_id)
    reg_pairs: dict[str, set] = {}
    for c, lst in declared.items():
        for sid, kind in lst:
            reg_pairs.setdefault(sid, set()).add((c, kind))
    problems = []
    for name, source in sorted(files.items()):
        pairs, bad = _docstring_covers(source)
        for _sid, marker in bad:
            problems.append(f"{name}: malformed docstring marker {marker!r}")
        if not pairs:
            continue
        sid = spec_of.get(name)
        if sid is None:
            problems.append(f"{name}: docstring COVERS in a file no spec "
                            "owns")
            continue
        for c, kind in sorted(pairs - reg_pairs.get(sid, set())):
            problems.append(f"{name}: docstring declares '{c} ({kind})' but "
                            f"the registry entry for {sid} does not")
    return problems


def _legacy_coverage(reg: dict, commitment: str) -> list[str]:
    """Coverage by regex over titles — the rule this spec exists to retire."""
    rx = re.compile(LEGACY_PATTERNS[commitment], re.I)
    return [s.id for s in reg.values() if rx.search(s.title)]


def _declared_coverage(reg: dict, commitment: str) -> list[str]:
    """Coverage by declaration — the rule under test."""
    declared, _ = declarations(reg)
    return [i for i, _kind in declared[commitment] if i in reg]


def _fixture() -> dict:
    """The live registry plus two decoys and one honestly-declared outsider."""
    reg = dict(BY_ID)
    donor = BY_ID["T0.01"]
    # D1 — the real 2026-08-10 artifact, reduced to its essence. Its title
    # contains no shelter word a human would name; it contains "ho-NEST".
    reg["ZZ.decoy1"] = replace(
        donor, id="ZZ.decoy1", title="The honest baseline", notes=None)
    # D2 — the case word boundaries do NOT fix: "sound" as in valid, not as in
    # audible. PG.1's real title, so this is not a straw man.
    reg["ZZ.decoy2"] = replace(
        donor, id="ZZ.decoy2",
        title="Playground generates and is physically sound", notes=None)
    # D3 — the false-NEGATIVE case: declares the commitment, and its title says
    # nothing a shelter pattern could match. This is BA.01's situation. (The
    # kind is explicit because P9 made kindless an ERROR; P4's property — a
    # declaration counts however the title reads — is unchanged by it.)
    reg["ZZ.declared"] = replace(
        donor, id="ZZ.declared", title="He keeps the rain off himself",
        notes="A lean-to earns its keep. COVERS: shelter/building (claim)")
    # D4 — a declaration naming a commitment that does not exist. A typo is the
    # false positive wearing a new hat: it reads as a claim and buys nothing.
    reg["ZZ.typo"] = replace(
        donor, id="ZZ.typo", title="Something unrelated",
        notes="COVERS: shelterr")
    # D5 — the real 2026-08-12 artifact: a PROSE MENTION of the marker is not
    # a declaration. T0.24's notes say it deliberately declares NO commitment,
    # and the v1 parser read that sentence as a malformed declaration named
    # "` commitment" — a false positive that makes the malformed-declaration
    # report cry wolf.
    reg["ZZ.prose"] = replace(
        donor, id="ZZ.prose", title="Guards the harness, not a capability",
        notes="Deliberately declares NO `COVERS:` commitment. It guards the "
              "harness, not a capability.")
    return reg


N_PROPERTIES = 12


def _probe(rule_is_regex: bool) -> dict:
    failed: list[str] = []
    cov = _legacy_coverage if rule_is_regex else _declared_coverage
    fix = _fixture()

    # P1 — the null. An EMPTY registry must credit nothing to anything. An
    # audit reading its own commitment list rather than the ladder would report
    # coverage for a repository containing no specs at all.
    if any(cov({}, c) for c in LEGACY_PATTERNS):
        failed.append("p1_empty_registry_covers_nothing")

    # P2 — the commitment list does not shrink to make the ladder look covered.
    # It still names the four holes the hand audit found at zero.
    if (not set(CONSTITUTIONAL_GAPS_2026_08_10) <= set(COMMITMENTS)
            or len(COMMITMENTS) < 20):
        failed.append("p2_commitments_still_name_the_gaps")

    # P3 — KNOWN ANSWER, the false positive. "The honest baseline" must NOT be
    # shelter coverage, and "physically sound" must NOT be hearing coverage.
    # The second is here because word boundaries fix the first and not the
    # second: the cheap half of the fix must not be mistaken for the fix.
    shelter = cov(fix, "shelter/building")
    hearing = cov(fix, "hearing")
    if "ZZ.decoy1" in shelter or "ZZ.decoy2" in hearing:
        failed.append("p3_word_must_not_grant_coverage")

    # P4 — KNOWN ANSWER, the false negative. A spec that DECLARES the
    # commitment counts however its title reads. This is BA.01's case, and a
    # rule that gets P3 right by refusing everything would fail here.
    if "ZZ.declared" not in shelter:
        failed.append("p4_declaration_grants_coverage")

    # P5 — a malformed declaration is REPORTED, not dropped — and a PROSE
    # MENTION of the marker is NEITHER reported NOR credited (D5, T0.24's real
    # sentence). The two directions together are the property: a reporter that
    # drops typos hides claims, and one that reports prose trains its reader
    # to ignore it. Only the declaration rule can even see this; the regex
    # rule never reads a marker, so it is scored as a failure for it, which is
    # honest: an organ that cannot detect the failure mode does not get credit
    # for not having it.
    dec, bad = declarations(fix)
    if (rule_is_regex or ("ZZ.typo", "shelterr") not in bad
            or any(sid == "ZZ.prose" for sid, _ in bad)
            or any(any(sid == "ZZ.prose" for sid, _k in pairs)
                   for pairs in dec.values())):
        failed.append("p5_malformed_declaration_is_reported")

    # P6 — no stale credit. Delete the declaring specs and the coverage goes
    # with them. The failure this kills: a family gets renamed and the audit
    # keeps reporting the commitment as covered off a declaration nobody holds.
    # (Generalised 2026-08-25: this fixture hard-coded ("SH.01","ZZ.declared")
    # as THE declarers of shelter/building; SH.02's registration made the
    # commitment two-declarer and the stale pair failed an honest registry.
    # The property is "remove EVERY credited spec and coverage must vanish" —
    # so the removal set is computed from the rule under test, and registry
    # growth can never stale it again. Semantics unchanged, set strictly
    # larger; the third instance of the cached-list disease, this time inside
    # the guard that exists to catch stale credit.)
    minus = {k: v for k, v in fix.items()
             if k not in set(cov(fix, "shelter/building"))}
    if cov(minus, "shelter/building"):
        failed.append("p6_deleted_spec_loses_coverage")

    # P7 — against the LIVE registry: no malformed declarations anywhere, and
    # every commitment that reports a PASS reports it for a spec that declared
    # itself. A green audit with a typo'd marker in it is not green.
    live_declared, live_bad = declarations(BY_ID)
    if live_bad:
        failed.append("p7_live_declarations_are_well_formed")
    else:
        for row in report():
            if not set(row["specs"]) <= {i for i, _k in
                                         live_declared[row["commitment"]]}:
                failed.append("p7_live_declarations_are_well_formed")
                break

    # P8 — a declaration's KIND decides what a PASS buys (Overseer, items
    # carried across the 8th-10th audits: PG.4 passing made `curiosity` read
    # as demonstrated when what passed was the TRAP; LC.01 did the same to
    # `one brain / unison` off the ADMISSION RULE). Both directions on fixed
    # fake results, plus the two parses that could silently corrupt it: a
    # canonical name that itself ends in parens — `thermal (kills) (claim)` —
    # must still be a claim of that commitment, and a typo'd kind must be
    # REPORTED, not read as a claim. The regex rule reads no markers, so it
    # cannot distinguish apparatus from capability; scored as its failure,
    # same honesty as P5.
    donor = BY_ID["T0.01"]
    kinds_reg = {
        "ZZ.kclaim": replace(donor, id="ZZ.kclaim", title="He keeps rain off",
                             notes="COVERS: shelter/building (claim)"),
        "ZZ.kfix":   replace(donor, id="ZZ.kfix", title="The rain exists",
                             notes="COVERS: shelter/building (fixture)"),
        "ZZ.ktherm": replace(donor, id="ZZ.ktherm", title="Cold is felt",
                             notes="COVERS: thermal (kills) (claim)"),
        "ZZ.kbad":   replace(donor, id="ZZ.kbad", title="Unrelated",
                             notes="COVERS: shelter/building (fixure)"),
    }
    fake_pass = {i: {"status": "PASS"} for i in kinds_reg}
    krows = {r["commitment"]: r for r in report(kinds_reg, fake_pass)}
    kdec, kbad = declarations(kinds_reg)
    shelter_row, therm_row = krows["shelter/building"], krows["thermal (kills)"]
    if (rule_is_regex
            or shelter_row["n_pass"] != 1                       # claim counts,
            or "ZZ.kclaim" not in shelter_row["specs"]
            or shelter_row["support_pass"] != {"ZZ.kfix": "fixture"}  # fixture
            or shelter_row["n_specs"] != 2                      # does not
            or therm_row["n_pass"] != 1                         # paren name OK
            or dict(kdec["thermal (kills)"]) != {"ZZ.ktherm": "claim"}
            or ("ZZ.kbad", "shelter/building (fixure)") not in kbad):
        failed.append("p8_kind_decides_what_a_pass_buys")

    # P9 — a KINDLESS declaration is REPORTED, never silently a `claim`, and
    # an explicit `(claim)` is credited (Overseer, 12th audit). The failure
    # this kills happened in this repo two days after P8 shipped: the kind
    # mechanism was applied to 2 of 78 declarations, the other 76 inherited
    # the implicit default `claim`, and the standing zero-pass rule steered
    # off a coverage report flattered by ten fixtures and sensors counted as
    # capability claims. A default on the field that routes work is the
    # defect; only an error is safe. The control is THE DEFAULTING RULE
    # ITSELF, kept executable via `default_kind="claim"`: under it the
    # kindless marker silently buys a claim and this property must fail.
    nk_reg = {
        "ZZ.nokind": replace(donor, id="ZZ.nokind", title="Rain kept off",
                             notes="COVERS: shelter/building"),
        "ZZ.expl":   replace(donor, id="ZZ.expl", title="Rain kept off, twice",
                             notes="COVERS: shelter/building (claim)"),
    }
    ndec, nbad = declarations(
        nk_reg, default_kind="claim" if rule_is_regex else None)
    npairs = dict(ndec["shelter/building"])
    if (npairs.get("ZZ.expl") != "claim"              # explicit claim credited
            or "ZZ.nokind" in npairs                  # kindless buys nothing
            or not any(sid == "ZZ.nokind" for sid, _ in nbad)):  # and is seen
        failed.append("p9_kindless_declaration_is_reported")

    # P10 — the DOCSTRING copy of a `COVERS:` marker is validated against the
    # registry copy, or it stops being written (26th audit B2). Known answers
    # first — the kind-flip disease (t2_03: docstring `(claim)`, registry
    # `(fixture)`), the unbacked disease (t2_04: a marker its registry entry
    # never made), and the shape that must STAY legal (a matching pair, plus
    # a backticked prose mention) — then the LIVE tests directory must be
    # clean. The regex rule reads no markers, so it is scored as its failure,
    # same honesty as P5/P8.
    live_doc_problems: list[str] = []
    if rule_is_regex:
        failed.append("p10_docstring_covers_match_registry")
    else:
        ka_reg = {"ZZ.dfix": replace(donor, id="ZZ.dfix", title="Apparatus",
                                     notes="COVERS: shelter/building (fixture)")}
        ka_files = {
            "ka_kind_mismatch.py": ('"""COVERS: shelter/building (claim)."""\n'
                                    'SPEC_ID = "ZZ.dfix"\n'),
            "ka_unbacked.py":      ('"""COVERS: hearing (claim)."""\n'
                                    'SPEC_ID = "ZZ.dfix"\n'),
            "ka_clean.py":         ('"""COVERS: shelter/building (fixture).\n'
                                    'A prose mention of `COVERS:` is not a\n'
                                    'declaration."""\n'
                                    'SPEC_ID = "ZZ.dfix"\n'),
        }
        ka = _docstring_audit(ka_files, ka_reg,
                              {"ka_kind_mismatch.py": "ZZ.dfix",
                               "ka_unbacked.py": "ZZ.dfix",
                               "ka_clean.py": "ZZ.dfix"})
        live_spec_of = {}
        for sid in BY_ID:
            path = module_path_for(sid)
            if path is not None:
                live_spec_of[path.name] = sid
        live_doc_problems = _docstring_audit(
            {p.name: p.read_text() for p in sorted(TESTS_DIR.glob("*.py"))},
            BY_ID, live_spec_of)
        if (not any("ka_kind_mismatch" in p for p in ka)
                or not any("ka_unbacked" in p for p in ka)
                or any("ka_clean" in p for p in ka)
                or live_doc_problems):
            failed.append("p10_docstring_covers_match_registry")

    # P11 — a PARKED spec is not coverage (28th audit B1: SH.01's honest,
    # pre-registered retirement left `shelter/building` and `thermal (kills)`
    # with no runnable claim while this tool exited 0). Known answers on a
    # fixed fixture, then the live registry's markers must parse. The regex
    # rule reads no markers, so it is scored as its failure, same honesty as
    # P5/P8/P10. The organ that failed is kept executable —
    # `report(credit_parked=True)` — and must reproduce the leak, or the
    # control controls nothing.
    if rule_is_regex:
        failed.append("p11_parked_is_not_coverage")
    else:
        pk_reg = {
            # The essence of SH.01: a claim spec retired by its own rule.
            "ZZ.pclaim": replace(
                donor, id="ZZ.pclaim", title="He keeps rain off",
                notes="COVERS: shelter/building (claim). "
                      "PARKED: 2026-08-01 — concluded by its own fork."),
            # Apparatus stays live: parking the claim must not touch it.
            "ZZ.pfix": replace(
                donor, id="ZZ.pfix", title="The rain exists",
                notes="COVERS: shelter/building (fixture)"),
            # A dateless marker parses as NOTHING and must be reported: an
            # unparseable retirement silently keeps counting as coverage.
            "ZZ.pbad": replace(
                donor, id="ZZ.pbad", title="Unrelated",
                notes="COVERS: hearing (claim). PARKED: soon"),
            # A backticked prose mention is discussion, not a retirement.
            "ZZ.pprose": replace(
                donor, id="ZZ.pprose", title="Discusses the mechanism",
                notes="A spec may be `PARKED:` by its own decision tree. "
                      "COVERS: hearing (claim)"),
        }
        prows = {r["commitment"]: r for r in report(pk_reg, {})}
        pmap, pbad = parked(pk_reg)
        shelter_p, hearing_p = prows["shelter/building"], prows["hearing"]
        leak = {r["commitment"]: r
                for r in report(pk_reg, {}, credit_parked=True)}
        live_pmap, live_pbad = parked(BY_ID)
        if ("ZZ.pclaim" in shelter_p["specs"]              # parked buys nothing
                or shelter_p["parked"] != {"ZZ.pclaim": "claim"}  # and is seen
                or not _claim_dead(shelter_p)          # commitment reads dead
                or shelter_p["n_specs"] != 1               # fixture stays live
                or _claim_dead(hearing_p)              # live claim ≠ dead
                or not any(sid == "ZZ.pbad" for sid, _ in pbad)  # dateless seen
                or "ZZ.pbad" in pmap                       # ...and not parked
                or "ZZ.pprose" in pmap                     # prose parks nothing
                or any(sid == "ZZ.pprose" for sid, _ in pbad)   # nor cries wolf
                or "ZZ.pclaim" not in leak["shelter/building"]["specs"]  # organ
                or live_pbad):                             # live markers parse
            failed.append("p11_parked_is_not_coverage")

    # P12 — the QUEUE instrument's own known-answer battery, put under the
    # ledger. `coverage.py` grew `queue_depth` on 2026-08-29 to answer "how
    # many specs could be dispatched today", and its fixtures ran in exactly
    # one place: the `__main__` path of `coverage.py`. Nothing re-ran them at
    # `--gate`, nothing recorded them, and no row would have gone red if the
    # instrument silently started answering a different question — which is
    # the shape of every organ failure this spec exists for. The batteries
    # themselves live next to the code they test (they need its internals);
    # what P12 adds is that a ladder run FAILS when they do.
    #
    # It is rule-independent by construction, so the regex control passes it.
    # That is correct and not a weakness: the control's job is to break P3/P4,
    # and `_check` requires exactly those two by name.
    #
    # AND P12 HAD THE HOLE IT WAS WRITTEN TO CLOSE, WITHIN A DAY (builder,
    # 2026-08-31). It imported the two batteries that existed when it was
    # written, BY NAME. `coverage.py` then grew four more — `_pilot_blocked`,
    # `_pilot_owed`, `_pilot_harvested`, `_exit_code` — on 2026-08-30, and every
    # one of them ran only in `__main__` again. A hardcoded list of the guards
    # you have is a guard against the guards you had. So P12 now DISCOVERS
    # them: every module-level `_*_fixture` in `coverage.py` runs here, which
    # covers the next one on the day it is written and needs nobody to remember.
    # The floor assertion is the other half — a battery renamed out of the
    # pattern would otherwise shrink this property to nothing, silently, and
    # `properties_checked` would still read 12.
    from .. import coverage as _cov
    batteries = sorted(n for n in dir(_cov)
                       if n.startswith("_") and n.endswith("_fixture")
                       and callable(getattr(_cov, n)))
    if len(batteries) < 7:
        failed.append("p12_queue_instrument_fixtures_hold")
    for _name in batteries:
        if getattr(_cov, _name)():
            failed.append("p12_queue_instrument_fixtures_hold")
            break

    rows = report()
    return {
        "properties_checked": float(N_PROPERTIES),
        "properties_failed": float(len(failed)),
        "failed_names": ",".join(failed),
        "commitments": float(len(COMMITMENTS)),
        "commitments_uncovered": float(sum(1 for r in rows if not r["n_specs"])),
        "declared_specs_live": float(sum(r["n_specs"] for r in rows)),
        "nominated_not_declared": float(sum(r["n_nominated"] for r in rows)),
        "malformed_declarations_live": float(len(live_bad)),
        "docstring_covers_problems_live": float(len(live_doc_problems)),
    }


def _experiment(seed: int) -> dict:
    return _probe(rule_is_regex=False)


def _control(seed: int) -> dict:
    """Coverage by title regex — the organ that failed, kept executable.

    It must break P3 (it credits "The honest baseline" to shelter and
    "physically sound" to hearing) and P4 (it cannot see a declaration). Those
    two are the whole difference between "a spec is ABOUT this commitment" and
    "a spec contains these letters".
    """
    return _probe(rule_is_regex=True)


def _check(m: dict, c: dict) -> Status | bool:
    # All nine ran AND all nine held. Gating on `properties_failed == 0`
    # alone would let a battery that stopped early read as clean (T0.13's own
    # first bug; T0.19 and T0.20 carry the same guard).
    experiment_clean = (m["properties_failed"] == 0.0
                        and m["properties_checked"] == N_PROPERTIES
                        and c["properties_checked"] == N_PROPERTIES)
    # The control must fail, and fail on THE properties that define the guard.
    control_names = set(str(c.get("failed_names", "")).split(","))
    control_broken = (c["properties_failed"] > 0.0
                      and {"p3_word_must_not_grant_coverage",
                           "p4_declaration_grants_coverage"} <= control_names)
    return bool(experiment_clean and control_broken)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID[SPEC_ID], _experiment, _check, control_fn=_control,
                    ledger=ledger)
