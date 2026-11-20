"""The forge's round runner (docs/FORGE.md, order F1) — the ladder's rule as code.

A ROUND is a frozen spec (JSON) plus per-run result rows. This module does not
train anything and never says anything about Jack: it only decides WHO CLIMBS
from rung S to M and from M to L. The decisive rung (L) is judged by
`experiments/harness` through its own pre-registration; nothing here replaces it.

The spec (`validate_spec` states every clause):
  round, question, baseline (arm id), twins (never-trains arm ids),
  candidates [{id, source, prior, twin, settings}],
  rungs {S|M|L: {decisions, seeds}},
  outcome {metric, requires_eating: true, metric_sha256, ...},
  budget {core_hours, kernel_hours}

A result row: {arm, rung, seed, score, core_hours, kernel_hours, metric_sha256}
  `arm` is a candidate id, the baseline, or a twin. `score` is the frozen
  metric on that run (higher is better; `experiments/forge/metric.py`).

The ladder's rule (FORGE.md "LADDER"), ours beside the charter's words:
  "the top third"            k = ceil(entrants / 3); into L at most 2. A tie
                             ACROSS the cut is not a win: everyone tied with
                             the first arm below the cut is dropped.
  "beat BOTH the baseline    mean score over the rung's seeds STRICTLY above
   and their never-trains    the baseline's mean and above its own twin's
   twin"                     mean. Equal does not climb.
  the control source         a candidate from any other source must also be
                             strictly above the control source's mean (the
                             mean of the control candidates' means). A
                             candidate equal to plain variation has shown
                             nothing. Control candidates climb by the same
                             rule minus this clause — if plain variation wins,
                             that is the finding. (Our reading; the charter
                             says a source that cannot beat control is not
                             paying.) Above S, if no control candidate
                             climbed, the best control of the rung below
                             runs anyway as the YARDSTICK ("always run the
                             control source at the same budget"): it is
                             compared against and cannot itself climb.
  "a round that runs out     cost of ALL result rows so far > either budget
   stops"                    line -> status STOPPED, nobody climbs.
  crash / NaN                a missing, null or non-finite score scores 0 —
                             it counts against its arm (ruling 19's reading).
  runs analysed = launched   a candidate missing a seed (or its twin missing
                             one) is INCOMPLETE and cannot climb; a duplicate
                             row, an unknown arm, or an incomplete baseline
                             refuses the whole reading.
  the judge is not the       every row's `metric_sha256` and the code on disk
   candidate's to touch      must equal the spec's frozen hash; else Refused.
"""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from experiments.forge.metric import metric_hash

SOURCES = ("published", "inventor", "recombine", "machine", "control")
RUNGS = ("S", "M", "L")
MAX_NEW_CANDIDATES = 8      # FORGE.md fences
MAX_LONG = 2                # FORGE.md: rung L, at most 2 candidates + baseline
REPO = Path(__file__).resolve().parent.parent.parent


class Refused(RuntimeError):
    """The round will not be read: the rule is unfrozen, broken, or the judge moved."""


@dataclass
class Climb:
    round: str
    rung: str                       # the rung that was read
    status: str                     # "OK" | "STOPPED" (over budget)
    climbers: list                  # candidate ids that go to the next rung
    means: dict                     # arm -> mean score at this rung
    reasons: dict                   # candidate -> why it did or did not climb
    spent: dict                     # core_hours, kernel_hours over all rows so far
    control_mean: float = float("nan")
    entrants: list = field(default_factory=list)
    yardstick: str | None = None    # a control arm run for comparison only


def validate_spec(spec: dict) -> None:
    def need(cond, msg):
        if not cond:
            raise Refused(f"round spec: {msg}")

    for k in ("round", "question", "baseline", "twins", "candidates", "rungs", "outcome", "budget"):
        need(k in spec, f"missing '{k}'")
    cands = spec["candidates"]
    ids = [c.get("id") for c in cands]
    need(all(isinstance(i, str) and i for i in ids), "every candidate needs an id")
    reserved = [spec["baseline"], *spec["twins"]]
    need(len(set(ids + reserved)) == len(ids) + len(reserved),
         "candidate, baseline and twin ids must all differ")
    for c in cands:
        need(c.get("source") in SOURCES, f"{c['id']}: source must be one of {SOURCES}")
        need(isinstance(c.get("prior"), str) and c["prior"].strip(),
             f"{c['id']}: no prior (law 3a)")
        need(c.get("twin") in spec["twins"], f"{c['id']}: twin must be one of the round's twins")
    need(any(c["source"] == "control" for c in cands),
         "no control-source candidate (plain variation at the same budget)")
    new = [c for c in cands if c["source"] != "control"]
    need(len(new) <= MAX_NEW_CANDIDATES, f"more than {MAX_NEW_CANDIDATES} new candidates")
    need(set(spec["rungs"]) <= set(RUNGS) and "S" in spec["rungs"], "rungs must be S[, M[, L]]")
    for name, r in spec["rungs"].items():
        need(int(r.get("decisions", 0)) > 0, f"rung {name}: decisions > 0")
        seeds = r.get("seeds", [])
        need(len(seeds) >= 1 and len(set(seeds)) == len(seeds), f"rung {name}: distinct seeds")
    all_seeds = [s for r in spec["rungs"].values() for s in r["seeds"]]
    need(len(set(all_seeds)) == len(all_seeds), "a seed may not be reused across rungs (fresh seeds)")
    out = spec["outcome"]
    need(out.get("requires_eating") is True, "the outcome must require eating")
    need(isinstance(out.get("metric_sha256"), str) and len(out["metric_sha256"]) == 64,
         "outcome.metric_sha256 (the frozen metric code hash) is missing")
    for k in ("core_hours", "kernel_hours"):
        need(isinstance(spec["budget"].get(k), (int, float)) and spec["budget"][k] >= 0,
             f"budget.{k} must be a number")


def _committed_and_clean(path: Path) -> bool:
    p = Path(path).resolve()
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", str(p)],
                             capture_output=True).returncode == 0
    clean = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", str(p)],
                           capture_output=True).returncode == 0
    return tracked and clean


def load_round(round_dir, *, require_committed: bool = True):
    """Read `<round_dir>/spec.json` and `<round_dir>/results.jsonl` (may be absent).
    The spec must be committed and clean in git — so it cannot have been written
    or edited after its runs (turned off only by tests on temp files)."""
    round_dir = Path(round_dir)
    spec_path = round_dir / "spec.json"
    if not spec_path.is_file():
        raise Refused(f"no frozen rule at {spec_path}")
    if require_committed and not _committed_and_clean(spec_path):
        raise Refused(f"{spec_path} is not committed and clean in git")
    spec = json.loads(spec_path.read_text())
    validate_spec(spec)
    res_path = round_dir / "results.jsonl"
    rows = ([json.loads(line) for line in res_path.read_text().splitlines() if line.strip()]
            if res_path.is_file() else [])
    return spec, rows


def _score(row) -> float:
    s = row.get("score")
    if not isinstance(s, (int, float)) or isinstance(s, bool) or not math.isfinite(s):
        return 0.0          # crash / NaN is an outcome and counts against its arm
    return float(s)


def spent(rows) -> dict:
    tot = {"core_hours": 0.0, "kernel_hours": 0.0}
    for r in rows:
        for k in tot:
            v = r.get(k, 0.0)
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise Refused(f"result row has a bad {k}: {r}")
            tot[k] += v
    return tot


def entrants(spec: dict, rows: list, rung: str, *, code_hash: str | None = None) -> list:
    """Who runs at `rung`: every candidate at S; the climbers of the rung below otherwise."""
    if rung == "S":
        return [c["id"] for c in spec["candidates"]]
    below = RUNGS[RUNGS.index(rung) - 1]
    return climb(spec, rows, below, code_hash=code_hash).climbers


def yardstick(spec: dict, rows: list, rung: str, *, code_hash: str | None = None):
    """The control arm that must run at `rung` for comparison only: None at S
    (every control enters) and when a control candidate climbed into `rung`;
    otherwise the best complete control of the rung below (ties: spec order)."""
    if rung == "S":
        return None
    below = climb(spec, rows, RUNGS[RUNGS.index(rung) - 1], code_hash=code_hash)
    ctl = [c["id"] for c in spec["candidates"] if c["source"] == "control"]
    if any(c in below.climbers for c in ctl):
        return None
    ctl = [c for c in ctl if c in below.means]
    return max(ctl, key=lambda c: below.means[c]) if ctl else None


def climb(spec: dict, rows: list, rung: str, *, code_hash: str | None = None) -> Climb:
    """Apply the ladder's rule to the rows of `rung`: who goes to the next rung.
    `code_hash` is the metric code on disk (`metric_hash()`); tests plant another."""
    validate_spec(spec)
    if rung not in spec["rungs"] or rung == "L":
        raise Refused(f"rung {rung!r} is not a climbing rung of this round "
                      "(L is judged by experiments/harness)")
    frozen = spec["outcome"]["metric_sha256"]
    now = metric_hash() if code_hash is None else code_hash
    if now != frozen:
        raise Refused("the metric code on disk differs from the round's frozen hash")
    known = {c["id"] for c in spec["candidates"]} | {spec["baseline"], *spec["twins"]}
    seen = set()
    for r in rows:
        if r.get("metric_sha256") != frozen:
            raise Refused(f"result row scored by other metric code than the frozen one: "
                          f"{r.get('arm')}/{r.get('rung')}/{r.get('seed')}")
        if r.get("arm") not in known or r.get("rung") not in spec["rungs"]:
            raise Refused(f"result row for an arm or rung the rule never named: {r}")
        key = (r["arm"], r["rung"], r.get("seed"))
        if key in seen:
            raise Refused(f"duplicate result row {key}")
        seen.add(key)
        if r.get("seed") not in spec["rungs"][r["rung"]]["seeds"]:
            raise Refused(f"result row on a seed the rule never named: {key}")

    seeds = spec["rungs"][rung]["seeds"]
    by_arm: dict = {}
    for r in rows:
        if r["rung"] == rung:
            by_arm.setdefault(r["arm"], {})[r["seed"]] = _score(r)

    def complete(arm):
        return set(by_arm.get(arm, {})) == set(seeds)

    def mean(arm):
        return sum(by_arm[arm].values()) / len(seeds)

    ins = entrants(spec, rows, rung, code_hash=code_hash)
    cost = spent(rows)
    out = Climb(round=spec["round"], rung=rung, status="OK", climbers=[], means={},
                reasons={}, spent=cost, entrants=list(ins),
                yardstick=yardstick(spec, rows, rung, code_hash=code_hash))
    if any(cost[k] > spec["budget"][k] for k in cost):
        out.status = "STOPPED"
        out.reasons = {c: "round over budget — it stops, it does not borrow" for c in ins}
        return out
    if not ins:
        return out
    if not complete(spec["baseline"]):
        raise Refused(f"the baseline's rung-{rung} runs do not balance the rule's seeds")
    out.means = {a: mean(a) for a in by_arm if complete(a)}
    base = out.means[spec["baseline"]]
    cand = {c["id"]: c for c in spec["candidates"]}
    done = [c for c in ins if complete(c)]
    controls = [c for c in done if cand[c]["source"] == "control"]
    if out.yardstick and complete(out.yardstick):
        controls.append(out.yardstick)
    out.control_mean = (sum(out.means[c] for c in controls) / len(controls)
                        if controls else float("nan"))

    k = min(MAX_LONG, math.ceil(len(ins) / 3)) if RUNGS[RUNGS.index(rung) + 1] == "L" \
        else math.ceil(len(ins) / 3)
    ranked = sorted(done, key=lambda c: -out.means[c])
    cut = out.means[ranked[k]] if len(ranked) > k else -math.inf   # first arm below the cut
    for c in ins:
        if c not in done:
            out.reasons[c] = "INCOMPLETE: runs do not balance the rule's seeds"
            continue
        m, tw = out.means[c], cand[c]["twin"]
        if not m > cut:
            out.reasons[c] = f"not in the top third (cut {cut:.4g})"
        elif not m > base:
            out.reasons[c] = f"does not beat the baseline ({m:.4g} vs {base:.4g})"
        elif not complete(tw):
            out.reasons[c] = f"INCOMPLETE: never-trains twin {tw} is missing runs"
        elif not m > out.means[tw]:
            out.reasons[c] = f"does not beat its never-trains twin ({m:.4g} vs {out.means[tw]:.4g})"
        elif cand[c]["source"] != "control" and not controls:
            out.reasons[c] = "INCOMPLETE: no complete control-source candidate to compare with"
        elif cand[c]["source"] != "control" and not m > out.control_mean:
            out.reasons[c] = (f"does not beat plain variation "
                              f"({m:.4g} vs control mean {out.control_mean:.4g})")
        else:
            out.reasons[c] = "CLIMBS"
            out.climbers.append(c)
    return out


def source_yield(spec: dict, rows: list, *, code_hash: str | None = None) -> dict:
    """LEARN: per source, how many candidates went in at S and how many reached
    M and L. `paying` is False when the source's share reaching M is not
    strictly above the control source's (None while rung S is unread)."""
    cand = {c["id"]: c["source"] for c in spec["candidates"]}
    table = {s: {"in": 0, "M": 0, "L": 0, "paying": None}
             for s in SOURCES if s in cand.values()}
    for c, s in cand.items():
        table[s]["in"] += 1
    read = {}
    for rung, nxt in (("S", "M"), ("M", "L")):
        if rung not in spec["rungs"] or not any(r.get("rung") == rung for r in rows):
            break
        got = climb(spec, rows, rung, code_hash=code_hash)
        read[rung] = got
        for c in got.climbers:
            table[cand[c]]["M" if nxt == "M" else "L"] += 1
    if "S" in read and read["S"].status == "OK":
        ctrl = table["control"]["M"] / table["control"]["in"]
        for s, t in table.items():
            if s != "control":
                t["paying"] = t["M"] / t["in"] > ctrl
    return {"sources": table, "rungs": read}
