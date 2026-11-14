"""BD.01 — a body that stands feeds a perfect path-follower more often than the pole.

THE QUESTION (ORDERS J1). LF.04 asks whether living teaches him; its pilot
lives were spent lying on his side. `experiments/body_probe.py` found why:
today's body (a capsule on a 9-cm ball foot, the "pole") topples with NO drive
at ~decision 10 in 18/20 births and is never righted, so every learner so far
was scored dragged along the floor. W0.BAL arm C's base (the "plinth": a wide
box foot carrying the mass) stands (0-1/20 topples) — but under a straight-line
oracle it fed LESS (15/40 vs 23/40), stuck upright on walls. With a detouring
oracle (privileged grid Dijkstra, `PROBE_ORACLE=path`) the pilot read plinth
12/20 + 18/20 = 30/40 vs pole 7/20 + 16/20 = 23/40 (s90, s94; unpaired logs
/data/tmp/smell_sense/path2/). This spec turns that pilot into a claim before
the body default may move.

The policy is a hand-written privileged instrument (the path oracle), never a
part of him: the question is what CEILING each body gives a perfect map, i.e.
which body a learner should be raced in.

DECLARED 2026-11-12 (this commit), BEFORE ANY CLAIM SEED WAS READ — pilot seeds
were 90 and 94 only:
  Venue    body_probe.classify, ORACLE=path, adhesion 0.5, drive at COM,
           E_DEC = 250 decisions per birth, food ready at birth.
  Births   N_BIRTHS = 40 per body per seed, claim seeds 100, 101, 102
           (SEED_OFFSET 100 + the runner's 0..2). Same world seed and same
           birth-position stream for both bodies; the realised pairing (birth
           xy equal within 5 cm) is REPORTED, never gated — an idle-phase death
           can shift one body's stream.
  Gate     gate_fires(A, B), the SAME function for experiment and control:
             fires  <=>  sum_s fed_A - sum_s fed_B >= MARGIN_BIRTHS
                         AND fed_A - fed_B >= 0 on EVERY seed
                         AND A toppled in <= TOPPLE_MAX of its births (pooled)
           MARGIN_BIRTHS = 15 of 120 (12.5 pp) = 2 sd of a difference of two
           120-birth binomials at p = 0.6 (sqrt(2*120*0.24) = 7.6 births).
           The pilot's effect was +7/40 = 17.5 pp, so a true effect that size
           clears the bar ~3/4 of the time: a FAIL is not a refutation of a
           small effect, it is "not shown at this power" — reported as such.
           TOPPLE_MAX = 0.10: a "standing base" that topples is not one.
  PASS     gate fires for A = plinth, B = pole, AND does NOT fire for the
           control A = plinth, B = plinth twin (spawn stream salted — the same
           body differing ONLY in its birth draws: chance spread must not
           read as a body effect).
  VOID     calibration refused, or a body not carried by the world (raises).

WHAT A PASS DOES NOT SAY. That the plinth is HIS body — biology's answer is a
stable base AND a learned righting reflex; a PASS licenses racing learners in
the plinth (a playground option, pole stays the default until a verdict moves
it). Nor that he will find food: the oracle is privileged. The plinth's known
remaining cap (pool lip 5, ladder 2, ramp 2 of 10 pilot misses) is reported.

COST. ~400 s per 20 births on this box -> ~9 x 800 s = ~2 h CPU single
process; run detached (`scripts/launch_detached.sh`).
"""
from __future__ import annotations

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

IMPL_DEPS = ['body_probe.py', 'w0.py', 'playground.py']

SEED_OFFSET = 100
N_BIRTHS = 40
E_DEC = 250
ADH = 0.5
MARGIN_BIRTHS = 15
TOPPLE_MAX = 0.10
CONTROL_SALT = 1

_CACHE: dict = {}      # seed -> plinth records, shared with _control


def _births(body: str, seed: int, salt: int = 0) -> list:
    from .. import body_probe as P
    P.install_body(body)
    P.ORACLE = "path"
    P.DRIVE_Z, P.DRIVE_SCALE = None, 1.0
    P.SPAWN_SALT = salt
    try:
        return P.classify(seed, N_BIRTHS, E_DEC, ADH)
    finally:
        P.SPAWN_SALT = 0
        P.install_body("pole")


def _counts(recs: list) -> dict:
    return {"fed": sum(r["fed"] is not None for r in recs),
            "toppled": sum(bool(r["toppled"]) for r in recs),
            "n": len(recs)}


def _paired(a: list, b: list) -> int:
    return sum(abs(x["birth_xy"][0] - y["birth_xy"][0]) <= 0.05
               and abs(x["birth_xy"][1] - y["birth_xy"][1]) <= 0.05
               for x, y in zip(a, b))


def _misses(recs: list) -> str:
    from collections import Counter
    c = Counter(p for r in recs if r["fed"] is None
                for p in (r["partners"] or ["open"]))
    return ",".join(f"{k}:{v}" for k, v in c.most_common(6))


def gate_fires(fed_a: list, fed_b: list, top_a: int, n_a: int) -> bool:
    return (sum(fed_a) - sum(fed_b) >= MARGIN_BIRTHS
            and all(x - y >= 0 for x, y in zip(fed_a, fed_b))
            and top_a <= TOPPLE_MAX * n_a)


def _arm_metrics(prefix: str, a: list, b: list) -> dict:
    ca, cb = _counts(a), _counts(b)
    return {f"{prefix}_fed_a": ca["fed"], f"{prefix}_fed_b": cb["fed"],
            f"{prefix}_toppled_a": ca["toppled"],
            f"{prefix}_toppled_b": cb["toppled"],
            f"{prefix}_n": ca["n"], f"{prefix}_paired": _paired(a, b)}


def _experiment(seed: int) -> dict:
    s = seed + SEED_OFFSET
    plinth = _births("plinth", s)
    pole = _births("pole", s)
    _CACHE[s] = plinth
    m = _arm_metrics("race", plinth, pole)
    m["seed_used"] = s
    m["plinth_misses"] = _misses(plinth)
    m["pole_misses"] = _misses(pole)
    _SEEDS.append(m)
    return m


def _control(seed: int) -> dict:
    s = seed + SEED_OFFSET
    plinth = _CACHE.pop(s, None) or _births("plinth", s)
    twin = _births("plinth", s, CONTROL_SALT)
    c = _arm_metrics("twin", plinth, twin)
    c["seed_used"] = s
    _CTRL.append(c)
    return c


_SEEDS: list = []
_CTRL: list = []


def _record_extremes(m: dict, c: dict) -> None:
    """Write onto the row the per-seed sums and worst-seed margin the gate
    judges (the runner aggregates to means), so `run verify` re-derives the
    verdict from the record alone. `_check` reads its arguments, nothing else."""
    for rows, d, p in ((_SEEDS, m, "race"), (_CTRL, c, "twin")):
        d["n_seeds"] = len(rows)
        for k in ("fed_a", "fed_b", "toppled_a", "n"):
            d[f"{p}_{k}_sum"] = sum(r[f"{p}_{k}"] for r in rows)
        d[f"{p}_diff_min"] = min((r[f"{p}_fed_a"] - r[f"{p}_fed_b"]
                                  for r in rows), default=-1)


def _gate(d: dict, p: str) -> bool:
    return gate_fires([d[f"{p}_fed_a_sum"]], [d[f"{p}_fed_b_sum"]],
                      d[f"{p}_toppled_a_sum"], d[f"{p}_n_sum"]) \
        and d[f"{p}_diff_min"] >= 0


def _check(m: dict, c: dict):
    if not m.get("n_seeds") or m["n_seeds"] != c.get("n_seeds"):
        return Status.VOID
    fired, twin_fired = _gate(m, "race"), _gate(c, "twin")
    m["gate_fires"] = 1.0 if fired else 0.0
    c["twin_gate_fires"] = 1.0 if twin_fired else 0.0
    return fired and not twin_fired


def run(ledger: Ledger | None = None):
    _SEEDS.clear()
    _CTRL.clear()

    def check(m, c):
        _record_extremes(m, c)
        return _check(m, c)
    return run_spec(BY_ID["BD.01"], _experiment, check,
                    control_fn=_control, ledger=ledger)
