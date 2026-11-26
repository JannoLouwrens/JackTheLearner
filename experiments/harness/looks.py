"""Look 1 with ten runs home (ORDERS S20, ruling 123).

`compare` refuses fewer than looks[1] runs an arm, yet its look 1 only ever
reads the first looks[0]. Ruling 123: that refusal is plumbing, not design.

This file adds NO statistics. `__init__.py` and `prereg.py` are hashed by
round 1's judge lock and by every pre-registration (`ANALYSIS_FILES`), so
they are not edited: an edit would make those results unreadable. Instead:
  * >= looks[1] runs an arm -> `compare` itself, called unchanged.
  * looks[0] <= n < looks[1] -> each arm is padded to looks[1] with filler
    runs that sit AFTER the real ones, and `compare` runs as frozen. Its look 1
    reads only the first looks[0] runs with the rng drawn first, so p1, the
    alpha and the futility rule are the hashed code's own. A look-1 verdict
    (A_DIFFERS / FUTILE) is returned; if `compare` went on to look 2 its
    answer touched the filler and is thrown away -> `NEEDS_LOOK_2` (never
    NOT_DISTINGUISHED).
  * < looks[0] -> raises, as before.
`judge_looks` is `prereg.judge` the same way: the launched runs not yet home
are filled with "crash" (censored at T) AFTER the first looks[0] launched
ids, which must all be home; every refusal of `judge` still applies.
"""
from __future__ import annotations

import math

from experiments.harness import LOOK_ALPHA, LOOKS, Outcome, Verdict, compare

NEEDS_LOOK_2 = "NEEDS_LOOK_2"
_FILL = Outcome(1.0, False)


def compare_looks(a, b, n_perm: int = 2000, rng=None,
                  alpha: float = LOOK_ALPHA, looks=LOOKS) -> Verdict:
    n1, n2 = looks
    if len(a) >= n2 and len(b) >= n2:
        return compare(a, b, n_perm, rng, alpha, looks)
    if len(a) < n1 or len(b) < n1:
        raise ValueError(f"need {n1} runs per arm for look 1")
    pad = lambda x: list(x[:n2]) + [_FILL] * (n2 - len(x[:n2]))  # noqa: E731
    v = compare(pad(a), pad(b), n_perm, rng, alpha, looks)
    return v if v.look == 1 else Verdict(NEEDS_LOOK_2, 1, v.p1, math.nan)


def judge_looks(path, runs: dict, **kw) -> Verdict:
    """`prereg.judge` that may read look 1 before look 2's runs are home."""
    import json
    from pathlib import Path

    from experiments.harness import prereg
    pre = json.loads(Path(path).read_text()) if Path(path).is_file() else None
    if pre is None:
        return prereg.judge(path, runs, **kw)        # raises its own Refused
    n1, n2 = LOOKS
    filled, short = {}, False
    for arm, ids in pre["launched"].items():
        home = dict(runs.get(arm, {}))
        if len(ids) < n2 or len(home) >= len(ids):
            filled[arm] = home
            continue
        missing_first = [r for r in ids[:n1] if r not in home]
        if missing_first:
            raise prereg.Refused(f"{arm}: look 1 needs the first {n1} launched runs, "
                                 f"not home {missing_first}")
        short = True
        filled[arm] = {**home, **{r: ["crash"] for r in ids if r not in home}}
    for arm in set(runs) - set(filled):
        filled[arm] = runs[arm]                      # check #13 names the extra arm
    v = prereg.judge(path, filled, **kw)
    return Verdict(NEEDS_LOOK_2, 1, v.p1, math.nan) if short and v.look == 2 else v
