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

THE SECOND SCAR (111th audit, 2026-09-24), because the first version of this
module put `T0.28` in the wrong half of exactly that pair, in the same
direction, two days after being built to tell them apart. "Already stale" was
decided by `staleness_of`, which hashes the file and its declared deps OFF
DISK — and by the time anyone asks what an edit would cost, the edit is on
the disk. So a certificate staled BY the edit read `CHANGED`, `CHANGED` is in
`ALREADY_KINDS`, and the row fell out of `bill` into `already` with the
render string calling it somebody else's debt; `eca5757` quoted that answer
as authority for declining the re-buy. The default invocation made it
unavoidable rather than unlucky: `render()` -> `price(changed_paths())` can
only ever name paths that have already been edited, so in that lane the
`BILLED` branch was reachable only for pre-`impl_sha` rows. The repair is
`kinds_before_edit`: "before" is reconstructed from HEAD's blobs and fed
through the SAME hash (`impl_sha_of`'s overrides), never read off disk.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

from .protocol import (Ledger, Status, blob_sha_at_run, impl_deps_of,
                       impl_sha_of, module_path_for, staleness_of)
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


def _head_blob(rel: str) -> Optional[bytes]:
    """`rel`'s bytes as committed at HEAD; None when HEAD carries no such path.

    None is a REAL answer (absent at HEAD -> `impl_sha_of`'s `missing:` rule,
    which is what "this path did not exist at the pre-edit tree" hashes as).
    A git that cannot answer at all raises instead of returning it, so the
    caller's stated bias decides that case rather than a silent default.
    """
    r = subprocess.run(("git", "-C", str(_REPO), "show", f"HEAD:{rel}"),
                       capture_output=True, timeout=30)
    return r.stdout if r.returncode == 0 else None


def kinds_before_edit(entry, path, want, head_bytes=_head_blob) -> list:
    """Staleness kinds at HEAD-MINUS-THIS-EDIT — the working tree cannot answer.

    (111th audit, RANK 1.) `staleness_of` reads the file and its declared deps
    off disk, and by the time anyone asks what an edit would cost, the edit is
    on the disk — so it answers "is this stale AFTER the edit", which every
    covered certificate is. The question this module owes is whether the row
    was stale BEFORE: for every path in `want` the read is HEAD's blob,
    everything else reads from disk, which IS the pre-edit state for a path
    this edit does not touch. No new hashing — the sha flows through
    `impl_sha_of`'s `file_bytes`/`dep_bytes` overrides, the same one code path
    `tree_reconstructing_sha` uses for the same question about older trees.

    Bias, inherited from `ALREADY_KINDS`' own comment: a pre-edit state that
    cannot be reconstructed (git down, test file itself uncommitted) yields no
    `CHANGED`, so the row stays billable — an unknown that read as
    already-broken would quietly zero the bill.
    """
    kinds = []
    if str(getattr(entry, "commit", "") or "").endswith("+dirty"):
        # A property of the ROW (the runner could not name what it executed),
        # not of the tree — no edit changes it, so it is "already" in any lane.
        kinds.append("DIRTY")
    rel = _rel(path)
    recorded = getattr(entry, "impl_sha", None)
    if recorded:
        try:
            if rel in want:
                fb = head_bytes(rel)
                if fb is None:
                    # The test file itself has no committed state: there is no
                    # pre-edit tree in which this row was clean or stale, and
                    # an undecidable "before" must not zero the bill.
                    return kinds
            else:
                fb = None
            src = fb if fb is not None else Path(path).read_bytes()
            deps, _ = impl_deps_of(path, source=src)
            overrides = {d: head_bytes(d) for d in deps if _rel(d) in want}
            pre = impl_sha_of(path, file_bytes=fb,
                              dep_bytes=overrides or None)
        except Exception:
            return kinds
        if pre is not None and pre != recorded:
            kinds.append("CHANGED")
        return kinds
    # Pre-`impl_sha` rows: every kind `staleness_of` derives there is a row
    # property or comes from COMMIT HISTORY (`deps_moved_since`,
    # `blob_sha_at_run`'s baseline) — except the "now" side of
    # `UNSTAMPED_CHANGED`, which is a disk read. When this edit touches the
    # test file itself, re-ask that one comparison with HEAD's blob: a
    # mismatch that exists only in the working tree is this edit's bill, not
    # an old debt.
    kinds += [k for k, _ in staleness_of(entry, path) if k != "DIRTY"]
    if "UNSTAMPED_CHANGED" in kinds and rel in want:
        try:
            fb = head_bytes(rel)
            base, problem = blob_sha_at_run(path,
                                            getattr(entry, "ran_at", None))
            if fb is not None and not problem \
                    and hashlib.sha256(fb).hexdigest() == base:
                kinds.remove("UNSTAMPED_CHANGED")
        except Exception:
            pass
    return kinds


def price(paths: Iterable[str], ledger: Optional[Ledger] = None,
          _head_bytes=None) -> dict:
    """Which certificates would an edit to `paths` stale, and what is the bill?

    Returns `{paths, bill, already, noncert, unknown}`. `bill` is the list that
    costs something: standing **PASS** rows, clean at HEAD-MINUS-THIS-EDIT,
    whose declared coverage this edit touches. `already` is decided at that
    same pre-edit state — never off disk, where the edit under pricing is
    already sitting (`kinds_before_edit`, the 111th-audit repair). Each entry
    is `{id, status, budget, hits}` — `budget` because the bill is a cost
    class, not a count, and `hits` because a reader must be able to check the
    charge against the declaration rather than trust it.
    """
    ledger = Ledger() if ledger is None else ledger
    want = {_rel(p) for p in paths}
    reader, _cache = (_head_bytes or _head_blob), {}

    def _head(rel):
        # One git read per path per pricing, whatever LADDER's size; a reader
        # that raises stays raising (the bias in `kinds_before_edit` decides).
        if rel not in _cache:
            try:
                _cache[rel] = ("ok", reader(rel))
            except Exception as e:
                _cache[rel] = ("err", e)
        tag, v = _cache[rel]
        if tag == "err":
            raise v
        return v
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
        kinds = kinds_before_edit(entry, path, want, head_bytes=_head) \
            if entry else []
        stale_before = [k for k in kinds if k in ALREADY_KINDS]
        if st is not Status.PASS:
            row["kinds"] = kinds
            out["noncert"].append(row)
        elif stale_before:
            row["kinds"] = stale_before
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

    # THE SECOND SCAR (111th audit): a PASS row, clean at HEAD, whose declared
    # dep carries an UNCOMMITTED edit must land in `bill`, never in `already`.
    # The pre-fix pricer read the dep off disk — i.e. read the very edit it
    # was being asked about — saw CHANGED, and filed `T0.28`'s re-buy as a
    # pre-existing debt; `eca5757` quoted that answer as authority. Replayed
    # exactly, on live declarations: the row's sha is minted against
    # fabricated PRE-EDIT dep bytes, the injected HEAD reader serves those
    # bytes back, and the disk — which the pricer must NOT consult for a path
    # in `want` — plays the edit. The committed pre-fix code fails this case
    # (verified 2026-09-24: bill=[], already=[('T0.21', ['CHANGED'])]).
    t21 = module_path_for("T0.21")
    dep = "experiments/coverage.py"
    pre_dep = b"# coverage.py as it stood before the edit under pricing\n"
    minted = impl_sha_of(t21, dep_bytes={dep: pre_dep})
    if minted is None:
        raise AssertionError("stale_cost: could not mint the scar fixture's "
                             "pre-edit sha for T0.21")
    entry = type("E", (), {"commit": "fixture0", "impl_sha": minted,
                           "ran_at": None})()
    fake = type("L", (), {"results": {"T0.21": entry},
                          "status": lambda self, sid:
                          Status.PASS if sid == "T0.21" else Status.NOT_RUN})()
    head = lambda rel: pre_dep if rel == dep else None
    pr = price([dep], ledger=fake, _head_bytes=head)
    if [r["id"] for r in pr["bill"]] != ["T0.21"] or pr["already"]:
        raise AssertionError(
            "stale_cost: a certificate staled BY the edit under pricing must "
            "be BILLED, not filed as an old debt — the pricer is reading the "
            "edit off disk and calling it the before. Got "
            f"bill={[r['id'] for r in pr['bill']]} "
            f"already={[(r['id'], r['kinds']) for r in pr['already']]}")
    # And the no-double-charge rule under the SAME reader: a row whose sha
    # matches the pre-edit tree of NOTHING was stale before this edit and
    # stays un-billed, exactly as before.
    entry.impl_sha = "0" * 16
    pr = price([dep], ledger=fake, _head_bytes=head)
    if [r["id"] for r in pr["already"]] != ["T0.21"] or pr["bill"]:
        raise AssertionError(
            "stale_cost: a certificate already stale at HEAD-minus-this-edit "
            "must stay in `already`; billing it would charge twice. Got "
            f"bill={[r['id'] for r in pr['bill']]} "
            f"already={[r['id'] for r in pr['already']]}")

    # And the live read must actually reach git rather than silently returning
    # the empty list that looks identical to a clean tree.
    if changed_paths(_git=lambda *a: "docs/LOOP_JOURNAL.md\n") != [
            "docs/LOOP_JOURNAL.md"]:
        raise AssertionError("stale_cost: `changed_paths` does not parse git")


if __name__ == "__main__":  # pragma: no cover
    _check()
    print(render(), end="")
