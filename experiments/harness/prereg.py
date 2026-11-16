"""J4 check #11 — the analysis script is frozen and hashed (PUBLISHED_PRIORS §8).

Source, §8 "Reproduce first" #11: the analysis that will judge a comparison
is fixed, and its hash recorded, BEFORE the runs exist. A comparison whose
code moved after its pre-registration is not believed.

How here:
  * `freeze(path, ...)` writes a pre-registration JSON: the question, the
    arms and their launch manifest (run ids, check #13), W, T, theta,
    n_perm, the permutation seed, and the sha256 of every analysis file
    (`ANALYSIS_FILES`: the outcome code + `compare` in `__init__.py`, the
    floor/ceiling gate in `calibration.py`).
  * `judge(path, runs)` is the ONLY door to a verdict on real arms. It
    refuses (raises `Refused`) when the pre-registration is missing, when
    any analysis file's hash differs from the frozen one, when the
    pre-registration file is not committed and clean in git (so it cannot
    have been written or edited after the runs — `require_committed`
    is turned off only by the tests on temp files), or when the runs do
    not balance the launch manifest (#13). Then it scores each run with
    the frozen W/T/theta and calls `compare` with the frozen n_perm/seed.

Ours beside theirs: §8 states the principle, not a mechanism; the git
"committed and clean" clause is ours (this repo's ledger is git).
`prereg.py` itself is not in the hash: it reads, it does not score; a
change to it cannot move a verdict without moving `compare` or `outcome`.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from experiments.harness import (THETA, Verdict, check_runs_accounted, compare, outcome,
                                 outcome_sustained)

MARKS = ("theta", "sustained")   # ruling 19's mark; ruling 22's (the primary from 11-16)

HERE = Path(__file__).resolve().parent
ANALYSIS_FILES = ("__init__.py", "calibration.py")
REPO = HERE.parent.parent


class Refused(RuntimeError):
    """The comparison will not run: no pre-registration, or the code moved."""


def file_hashes(root: Path = HERE, names=ANALYSIS_FILES) -> dict:
    return {n: hashlib.sha256((Path(root) / n).read_bytes()).hexdigest() for n in names}


def freeze(path, *, question: str, launched: dict, W: int, T: float,
           theta: float = THETA, n_perm: int = 2000, seed: int = 0,
           root: Path = HERE, mark: str = "theta") -> dict:
    if len(launched) != 2:
        raise ValueError("a comparison has exactly two arms (A, B)")
    if mark not in MARKS:
        raise ValueError(f"mark must be one of {MARKS}")
    pre = dict(question=question, arms=list(launched), launched=launched,
               W=W, T=T, theta=theta, n_perm=n_perm, seed=seed, mark=mark,
               analysis_sha256=file_hashes(root))
    Path(path).write_text(json.dumps(pre, indent=1, sort_keys=True) + "\n")
    return pre


def _committed_and_clean(path: Path) -> bool:
    p = Path(path).resolve()
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", str(p)],
                             capture_output=True).returncode == 0
    clean = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", str(p)],
                           capture_output=True).returncode == 0
    return tracked and clean


def judge(path, runs: dict, *, require_committed: bool = True,
          root: Path = HERE) -> Verdict:
    """`runs`: {arm: {run_id: [lives]}}; a life is (decisions, meals) under
    mark "theta", (decisions, capped, intake, drain) under "sustained" (a
    crashed run's lives end with "crash", as both outcome functions read)."""
    path = Path(path)
    if not path.is_file():
        raise Refused(f"no pre-registration at {path}")
    pre = json.loads(path.read_text())
    now = file_hashes(root, tuple(pre["analysis_sha256"]))
    moved = sorted(n for n, h in pre["analysis_sha256"].items() if now[n] != h)
    if moved or set(pre["analysis_sha256"]) != set(ANALYSIS_FILES):
        raise Refused(f"analysis code moved since pre-registration: {moved or 'file set changed'}")
    if require_committed and not _committed_and_clean(path):
        raise Refused(f"pre-registration {path} is not committed and clean in git")
    mark = pre.get("mark", "theta")   # files frozen before 11-16 carry no mark
    if mark == "sustained":
        score = lambda lives: outcome_sustained(lives, pre["W"], pre["T"])  # noqa: E731
    else:
        score = lambda lives: outcome(lives, pre["W"], pre["T"], pre["theta"])  # noqa: E731
    scored = {arm: {rid: score(lives)
                    for rid, lives in runs.get(arm, {}).items()} for arm in set(runs) | set(pre["arms"])}
    check_runs_accounted(pre["launched"], scored)
    a, b = ([scored[arm][rid] for rid in pre["launched"][arm]] for arm in pre["arms"])
    return compare(a, b, n_perm=pre["n_perm"], rng=pre["seed"])
