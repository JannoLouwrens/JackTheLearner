"""What does THIS edit cost the scoreboard? — priced BEFORE the commit.

`run stale` answers "which certificates are stale NOW". That is the same
question one commit too late, and on 2026-09-22 the difference was measured.

THE SCAR, in full, because this module may not exist without one. In the
05:1x slot that morning the builder shipped `eb38ae4` — `STEERING-PAGE SIZE`,
a reading added to `experiments/run.py`. `experiments/run.py` is the sole
entry in `T0.36`'s `IMPL_DEPS`, so that commit staled `T0.36`'s **standing
PASS**: a capability certificate about code that had moved. Thirty minutes
later the same slot's journal recorded, about its own work, *"No PASS
certificate staled — every row on the STALE list is FAIL/VOID."* That
sentence was true of the OTHER commit in the slot (`181fbff`, which touched
only a test file and the registry) and false of the slot. Nobody lied and
nobody was careless: `run status` reports staleness truthfully, but only
AFTER the edit, and only to a reader who runs it again after committing. The
author's belief about the author's own blast radius was the only instrument
in the loop, which is the precise form of evidence `SYSTEM.md`'s first law
exists to distrust.

`run blast-radius` already prices a GATE edit against the dependency graph
before it lands. This is its missing twin: the same before-the-edit question
asked of `IMPL_DEPS` — a graph edit is priced, a certificate edit was not.

WHAT IT DELIBERATELY DOES NOT DO.

* It does not REFUSE. Reporting-only and unfloored, for the reason
  `render_size` is: editing an instrument is legitimate and constant work in
  this repo, and a gate here would refuse the Review's own act. The bill is
  not a reason to skip the edit; it is a reason to re-buy in the same slot.
* It prices DECLARED coverage only. A spec that reads a file without naming
  it in `IMPL_DEPS` is invisible here and is UNDER-billed — that population is
  `protocol.undeclared_impl_imports`' subject, not this module's, and the
  render says so rather than implying completeness.
* It never charges twice. A certificate that is ALREADY stale cannot be
  staled again by this edit; it is named in its own bucket with the kind it
  carries, because dropping it would hide a debt and billing it would inflate
  one. That distinction is the one the scar got wrong in BOTH directions in a
  single slot: `T0.21` was stale before the builder touched anything (and was
  correctly re-bought), `T0.36` was staled BY the builder (and was not).
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

from .protocol import Ledger, Status, impl_deps_of, module_path_for, staleness_of
from .registry import BY_ID, LADDER

_REPO = Path(__file__).resolve().parent.parent

#: Staleness kinds that mean "this certificate ALREADY does not describe the
#: code it certifies". `UNVERIFIABLE` and `UNSTAMPED_INTACT` are deliberately
#: absent: they mean the record predates `impl_sha`, i.e. UNKNOWN, and an
#: unknown that renders as already-broken would quietly zero this module's
#: bill on the oldest and least protected rows in the ladder.
ALREADY_KINDS = ("CHANGED", "UNSTAMPED_CHANGED", "UNVERIFIABLE_MOVED", "DIRTY")


def _rel(path) -> str:
    """A repo-relative posix path, whatever the caller handed us."""
    p = Path(path)
    try:
        p = p.resolve().relative_to(_REPO)
    except (ValueError, OSError):
        pass
    return p.as_posix()


def covered_paths(spec_id: str):
    """The files whose bytes this spec's `impl_sha` folds in.

    Returns `(paths, problem)` — the test module itself PLUS everything it
    declares in `IMPL_DEPS`, which is exactly `impl_sha_of`'s input set. Read
    statically through `impl_deps_of`, never by importing the test: the same
    reason `stale_claims` gives, and the same one function, so the writer of a
    sha and this reader of it cannot drift apart.

    `problem` is `impl_deps_of`'s complaint, carried rather than swallowed. A
    spec whose declaration cannot be parsed has UNKNOWN coverage; reporting it
    as "covers nothing" is the silent narrowing `impl_deps_of` already refuses
    one layer down, and here it would read as a zero bill.
    """
    path = module_path_for(spec_id)
    if path is None:
        return frozenset(), "not implemented"
    deps, problem = impl_deps_of(path)
    return frozenset([_rel(path), *(_rel(d) for d in deps)]), problem


def price(paths: Iterable[str], ledger: Optional[Ledger] = None) -> dict:
    """Which certificates would an edit to `paths` stale, and what is the bill?

    Returns `{paths, bill, already, noncert, unknown}`. `bill` is the list that
    costs something: standing **PASS** rows, currently clean, whose declared
    coverage this edit touches. Each entry is
    `{id, status, budget, hits}` — `budget` because the bill is a cost class,
    not a count, and `hits` because a reader must be able to check the charge
    against the declaration rather than trust it.
    """
    ledger = Ledger() if ledger is None else ledger
    want = {_rel(p) for p in paths}
    out = {"paths": sorted(want), "bill": [], "already": [], "noncert": [],
           "unknown": []}
    if not want:
        # Nothing can be covered by nothing, so the scan is skipped rather
        # than run to produce four empty lists — this block prints on the
        # mandated pre-commit step and a clean tree must not pay for it.
        return out
    for spec in LADDER:
        st = ledger.status(spec.id)
        if st is Status.NOT_RUN:
            continue
        covers, problem = covered_paths(spec.id)
        hits = sorted(want & covers)
        if problem and problem != "not implemented":
            # Undecidable, so it is reported whether or not `hits` is empty:
            # the unparseable declaration might name any of these paths.
            out["unknown"].append({"id": spec.id, "status": st.value,
                                   "problem": problem, "hits": hits})
            continue
        if not hits:
            continue
        row = {"id": spec.id, "status": st.value,
               "budget": getattr(spec.budget, "value", str(spec.budget)),
               "hits": hits}
        entry = ledger.results.get(spec.id)
        path = module_path_for(spec.id)
        kinds = [k for k, _ in staleness_of(entry, path)] if entry else []
        stale_now = [k for k in kinds if k in ALREADY_KINDS]
        if st is not Status.PASS:
            row["kinds"] = kinds
            out["noncert"].append(row)
        elif stale_now:
            row["kinds"] = stale_now
            out["already"].append(row)
        else:
            out["bill"].append(row)
    return out


def changed_paths(repo: Optional[Path] = None, _git=None) -> List[str]:
    """What this working tree has actually touched, per git — never per memory.

    Tracked modifications (staged and unstaged, against HEAD) plus untracked
    files, because a new file can be the target of an existing `IMPL_DEPS`
    line. Git silent for any reason returns EMPTY, and the render says the
    list came back empty rather than printing a clean bill — a tool that
    reports "nothing to pay" when it could not ask is the failure mode this
    whole module is about.
    """
    repo = _REPO if repo is None else repo
    run = _git or (lambda *a: subprocess.run(
        ("git", "-C", str(repo)) + a, capture_output=True, text=True,
        timeout=30).stdout)
    try:
        tracked = run("diff", "--name-only", "HEAD").split()
        untracked = run("ls-files", "--others", "--exclude-standard").split()
    except Exception:                                      # pragma: no cover
        return []
    return sorted({*tracked, *untracked})


def render(pr: Optional[dict] = None, indent: str = "  ") -> str:
    """The `STALE-COST` block. Reporting-only, unfloored, and it always prints.

    Always, including for a clean tree with a zero bill: an absent block is
    indistinguishable from an absent check, which is this repo's standing rule
    for every reading of this shape.
    """
    pr = price(changed_paths()) if pr is None else pr
    n = len(pr["bill"])
    lines = [
        f"{indent}STALE-COST — {len(pr['paths'])} changed path(s) priced "
        f"against declared `IMPL_DEPS`; **{n} standing PASS certificate(s) "
        f"would be staled** by this edit.",
        f"{indent}  Reporting-only, unfloored: editing an instrument is "
        f"legitimate work and a gate here would refuse the Review's own act. "
        f"The bill is not a reason to skip the edit — it is the re-buy owed "
        f"in the SAME slot. DECLARED coverage only: a spec that reads a file "
        f"without naming it is UNDER-counted here.",
    ]
    if not pr["paths"]:
        lines.append(f"{indent}  no changed paths (clean tree, or git could "
                     f"not answer — empty is not the same as clean).")
    for r in pr["bill"]:
        lines.append(f"{indent}  BILLED  {r['id']:9} PASS  re-buy costs "
                     f"{r['budget']}  <- {', '.join(r['hits'])}")
    for r in pr["already"]:
        lines.append(f"{indent}  already {r['id']:9} PASS  {'/'.join(r['kinds'])}"
                     f" BEFORE this edit — a debt, but not this edit's bill")
    for r in pr["noncert"]:
        lines.append(f"{indent}  no-cert {r['id']:9} {r['status']:6} covered; "
                     f"no capability claim is lost by staling a non-PASS row")
    for r in pr["unknown"]:
        lines.append(f"{indent}  UNKNOWN {r['id']:9} {r['status']:6} "
                     f"declaration unreadable ({r['problem']}) — coverage "
                     f"cannot be decided, and unknown is not zero")
    if n:
        lines.append(f"{indent}  re-buy after the edit, not before: "
                     + " ".join(f"`run {r['id']}`" for r in pr["bill"]))
    return "\n".join(lines) + "\n"


def _check() -> None:
    """Refuse to report from a pricer that flunks the event it was built for.

    THE FIXTURE IS THE SCAR, on live data. `experiments/run.py` must charge
    `T0.36` — that is `eb38ae4`'s bill, stated by `T0.36`'s own declaration
    and therefore independent of what the tree happens to look like today.
    And it must NOT charge `T0.21`, whose single declared dependency is
    `experiments/coverage.py`: `T0.21` was stale BEFORE that slot and was
    correctly re-bought, `T0.36` was staled BY it and was not. A pricer that
    cannot tell those two apart would have reported the slot clean or
    reported it guilty of everything, and both readings lose the finding.
    """
    covers, problem = covered_paths("T0.36")
    if problem:
        raise AssertionError(f"stale_cost: T0.36's declaration is unreadable "
                             f"({problem}) — the scar fixture cannot be read")
    if "experiments/run.py" not in covers:
        raise AssertionError(
            "stale_cost: `experiments/run.py` is T0.36's sole IMPL_DEPS entry "
            "and staled its PASS on 2026-09-22 (eb38ae4). A pricer that does "
            f"not charge it cannot see the event it exists for. Got {covers}")
    if "experiments/run.py" in covered_paths("T0.21")[0]:
        raise AssertionError(
            "stale_cost: T0.21 declares `experiments/coverage.py`, not "
            "`run.py`. Charging it would make every instrument edit look "
            "equally expensive and the real bill unreadable.")

    # A zero bill must still PRINT, and must say the list may be empty for the
    # wrong reason.
    txt = render({"paths": [], "bill": [], "already": [], "noncert": [],
                  "unknown": []}, indent="")
    if "0 standing PASS" not in txt or "empty is not the same as clean" not in txt:
        raise AssertionError("stale_cost: a clean read must still print, and "
                             "must not call an unanswered git 'clean'")

    # A bill must name the cost class and the re-buy, or it is a number with
    # no action attached.
    billed = {"paths": ["experiments/run.py"],
              "bill": [{"id": "T0.36", "status": "PASS", "budget": "cpu<10min",
                        "hits": ["experiments/run.py"]}],
              "already": [], "noncert": [], "unknown": []}
    txt = render(billed, indent="")
    if "BILLED" not in txt or "cpu<10min" not in txt or "`run T0.36`" not in txt:
        raise AssertionError("stale_cost: a bill must carry its cost class "
                             f"and the re-buy command. Got: {txt!r}")

    # An already-stale certificate is named and NOT billed — no double charge.
    dup = {"paths": ["experiments/coverage.py"], "bill": [],
           "already": [{"id": "T0.21", "status": "PASS", "budget": "cpu<10min",
                        "hits": ["experiments/coverage.py"],
                        "kinds": ["CHANGED"]}],
           "noncert": [], "unknown": []}
    txt = render(dup, indent="")
    if "0 standing PASS" not in txt or "not this edit's bill" not in txt:
        raise AssertionError("stale_cost: an already-stale row may not be "
                             f"charged to this edit. Got: {txt!r}")

    # An unreadable declaration renders as unknown, never as clean.
    unk = {"paths": ["playground.py"], "bill": [], "already": [], "noncert": [],
           "unknown": [{"id": "XX.01", "status": "PASS",
                        "problem": "IMPL_DEPS is not a literal", "hits": []}]}
    if "unknown is not zero" not in render(unk, indent=""):
        raise AssertionError("stale_cost: an unparseable declaration must "
                             "report as UNKNOWN, not as no-coverage")

    # And the live read must actually reach git rather than silently returning
    # the empty list that looks identical to a clean tree.
    if changed_paths(_git=lambda *a: "docs/LOOP_JOURNAL.md\n") != [
            "docs/LOOP_JOURNAL.md"]:
        raise AssertionError("stale_cost: `changed_paths` does not parse git")


if __name__ == "__main__":  # pragma: no cover
    _check()
    print(render(), end="")
