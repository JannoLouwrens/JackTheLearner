"""pilot_b_report.py — read the J8 pilot's harvested runs BY THE RULE, mechanically.

ORDERS J8 "done when": the rule, the 20 + 15 runs and the table in
`docs/research/PILOT_B.md` with one of three readings. This file adds no
rule and no threshold: every number it applies is quoted from THE RULE in
`experiments/pilot_b.py` (committed `6f197f94`, before any kernel) and every
outcome is scored by the frozen `experiments.harness` (`outcome_sustained`,
`outcome`, `check_runs_accounted`, `boschloo_p`; hashes in
`docs/research/prereg/pilot_b_vs_{floor,blr0}.json`).

Written 11-16 19:10 UTC with 16 of 35 registered runs harvested and 0 of
them sustained — AFTER data existed. That is why it may only transcribe the
rule: `reading()` is tested on planted logs for each of the rule's four
answers (`tests/test_pilot_b_report.py`), and it REFUSES to print a reading
while any registered run is missing (a partial table is labelled PARTIAL and
decides nothing).

WHAT `prereg.judge` CANNOT DO HERE (found 11-16, before the harvest):
`judge()` ends in `compare()`, ruling 19's two-look test, which needs 20 runs
in EACH arm. The pilot's frozen comparisons have 20 vs 15 (pooled floor) and
20 vs 5 (blr0): `judge` raises "need 20 runs per arm" on both. The rule's
readings never needed `compare` (they are counts against fixed bars), so
this report runs judge's three refusals itself, through prereg's own
functions — analysis hashes unmoved, pre-registration committed and clean,
runs balance the launch manifest (#13) — and then counts. The Boschloo p it
prints is the harness's (`boschloo_p`, two-sided) and is descriptive. The
gap in `judge` is an INBOX line for the system lane, not patched here
(`__init__.py` is hash-pinned by check #1).

A crashed, timed-out or short cell is an outcome (ruling 19): its lives are
passed with "crash" appended — never crossed, never dropped, never re-seeded.

Definitions used in the "reported also" tables (the rule names the
quantities; where it leaves the arithmetic open the choice is stated here):
  meals            floor meals + apples of a life.
  last 50k         lives that END after decision T - 50,000 (incl. the
                   unfinished last life): their meals / their decisions x 1,000.
  early / late     first 5 lives of a run (inside B's 5,000-decision random
                   warm-up) against the last quarter of its finished lives
                   (INBOX 11-16 16:04 ask (a)); "fed" = >= 1 meal.
  first 20/last 20 meals per life over the first and last 20 finished lives.
  cut-off life     the life T interrupted (cause "unfinished"). By the rule it
                   never counts as sustained; it is SHOWN because seed 315's
                   was eating 24 meals at 3.2x its drain when T cut it at
                   6,937 of 9,000 decisions, and no finished-life column sees that.

Usage:  python -m experiments.pilot_b_report            # markdown to stdout
        python -m experiments.pilot_b_report --write    # replace "## Results" in PILOT_B.md
Exit 0 = all 35 registered runs in and a reading printed; 3 = PARTIAL.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import pilot_b as P  # noqa: E402
from experiments.harness import (boschloo_p, check_runs_accounted, outcome,  # noqa: E402
                                 outcome_sustained)
from experiments.harness import prereg  # noqa: E402
from experiments.harness.calibration import FLOOR_MAX_SUCCESSES  # noqa: E402

DOC = REPO / "docs" / "research" / "PILOT_B.md"
PREREG = REPO / "docs" / "research" / "prereg"
T = P.T_RUN
WS = (1, 2, 3)
NEED = 5            # the rule: ">= 5 of 20"
THETA = 5.0         # the rule's secondary mark
T_HALF = 150_000    # the rule: "median crossing <= 150,000, then 150,000"


def rid(arm, seed) -> str:
    return f"{arm}_s{seed}"


REGISTERED = [rid(a, s) for a, s in P.all_cells()]                      # 35
EXPLORE = [rid(a, s) for a in P.EXPLORE_ARMS for s in P.EXPLORE_SEEDS]  # 10


def load(outdir=None) -> tuple:
    """Every cell in every harvested kernel file -> ({run_id: rec}, {run_id: file}).
    A run id returned by two kernel files is an error, not a choice."""
    outdir = Path(outdir or P.OUT)
    runs, src = {}, {}
    for name in P.kernel_groups():
        f = outdir / f"{name}.json"
        if not f.exists():
            continue
        for k, rec in json.loads(f.read_text())["cells"].items():
            if k in runs:
                raise ValueError(f"{k} appears in {src[k]} and {f.name}")
            runs[k], src[k] = rec, f.name
    return runs, src


def crashed(rec: dict) -> bool:
    return bool(rec.get("rc", 0) != 0 or "error" in rec
                or sum(l["decisions"] for l in rec["lives"]) != rec.get("T", T))


def lives_for(rec: dict, mark: str) -> list:
    lv = P.harness_lives(rec, mark)
    return lv + ["crash"] if crashed(rec) else lv


def score(rec: dict) -> dict:
    """{('sustained'|'theta', W): Outcome} for W in 1..3, by the frozen harness."""
    out = {}
    for W in WS:
        out["sustained", W] = outcome_sustained(lives_for(rec, "sustained"), W, T)
        out["theta", W] = outcome(lives_for(rec, "theta"), W, T, THETA)
    return out


def _meals(l) -> int:
    return l["floor_meals"] + l["apples"]


def run_stats(rec: dict) -> dict:
    lives = rec["lives"]
    fin = [l for l in lives if l["cause"] != "unfinished"]
    s = {"lives": len(fin), "crashed": crashed(rec)}
    s["capped"] = sum(l["cause"] == "capped" for l in fin)
    s["causes"] = {c: sum(l["cause"] == c for l in fin)
                   for c in sorted({l["cause"] for l in fin})}
    s["sustained_lives"] = sum(bool(l["sustained"]) for l in fin)
    cum, tail_m, tail_d, first_sus = 0, 0, 0, None
    for l in lives:
        cum += l["decisions"]
        if cum > T - 50_000:
            tail_m += _meals(l)
            tail_d += l["decisions"]
        if first_sus is None and l["sustained"] and l["cause"] == "capped":
            first_sus = cum
    s["meals_per_1k_last50k"] = 1000.0 * tail_m / tail_d if tail_d else float("nan")
    s["first_sustained_at"] = first_sus

    def block(ls):
        if not ls:
            return None
        unfed = [l["seconds"] for l in ls if _meals(l) == 0]
        return {"n": len(ls), "meals": st.fmean(_meals(l) for l in ls),
                "fed": sum(_meals(l) > 0 for l in ls) / len(ls),
                "reward": st.fmean(l["reward"] for l in ls),
                "seconds": st.median(l["seconds"] for l in ls),
                "unfed_s": st.median(unfed) if unfed else float("nan")}
    s["early"] = block(fin[:5])
    s["late"] = block(fin[-max(1, len(fin) // 4):] if fin else [])
    u = lives[-1] if lives and lives[-1]["cause"] == "unfinished" else None
    s["unfinished"] = (None if u is None else
                       (u["decisions"], _meals(u),
                        u["intake"] / u["basal_drain"] if u["basal_drain"] else float("nan")))
    s["first20"] = st.fmean(_meals(l) for l in fin[:20]) if fin else float("nan")
    s["last20"] = st.fmean(_meals(l) for l in fin[-20:]) if fin else float("nan")
    return s


def frozen_checks(runs: dict) -> list:
    """judge()'s three refusals, through prereg's own functions, for both
    pilot pre-registrations. Returns a list of problems (empty = clean)."""
    bad = []
    for name in ("pilot_b_vs_floor", "pilot_b_vs_blr0"):
        path = PREREG / f"{name}.json"
        pre = json.loads(path.read_text())
        now = prereg.file_hashes(names=tuple(pre["analysis_sha256"]))
        moved = sorted(n for n, h in pre["analysis_sha256"].items() if now[n] != h)
        if moved:
            bad.append(f"{name}: analysis code moved since pre-registration: {moved}")
        if not prereg._committed_and_clean(path):
            bad.append(f"{name}: pre-registration not committed and clean in git")
        if pre["T"] != T or pre["mark"] != "sustained" or pre["W"] != 1:
            bad.append(f"{name}: frozen T/W/mark differ from the rule")
        have = {arm: {r: None for r in ids if r in runs} for arm, ids in pre["launched"].items()}
        try:
            check_runs_accounted(pre["launched"], have)
        except ValueError as ex:
            bad.append(f"{name}: #13 {ex}")
    return bad


def reading(scores: dict) -> dict:
    """THE RULE, transcribed. `scores`: {run_id: score(rec)} for exactly the
    35 registered runs; anything else raises (no reading on partial books)."""
    if set(scores) != set(REGISTERED):
        miss = sorted(set(REGISTERED) - set(scores))
        extra = sorted(set(scores) - set(REGISTERED))
        raise ValueError(f"no reading: missing {miss}, unregistered {extra}")
    b = [rid("b", s) for s in P.B_SEEDS]
    floor = {a: [rid(a, s) for s in P.FLOOR_SEEDS] for a in P.FLOOR_ARMS}
    n_b = {W: sum(scores[r]["sustained", W].event for r in b) for W in WS}
    n_theta = {W: sum(scores[r]["theta", W].event for r in b) for W in WS}
    floor_n = {a: sum(scores[r]["sustained", 1].event for r in ids) for a, ids in floor.items()}
    fs = sum(floor_n.values())
    res = {"n_b": n_b, "n_theta": n_theta, "floor_by_arm": floor_n, "floor_successes": fs,
           "floor_gate": fs <= FLOOR_MAX_SUCCESSES, "W": None, "T_decisive": None}
    if not res["floor_gate"]:
        res["reading"] = "VOID"            # the mark is too easy at this rung
    elif n_b[1] >= NEED:
        res["reading"] = "SUSTAINED"
    elif n_theta[3] >= NEED:
        res["reading"] = "EATS BUT STARVES"
    else:
        res["reading"] = "DOES NOT LEARN AT THIS BUDGET"
    # "Fixes": only where the rule fixes something (the rule is silent on
    # EATS BUT STARVES: no W with n_B(W) >= 5 exists there, so none is fixed).
    if res["reading"] == "SUSTAINED":
        W = max(w for w in WS if n_b[w] >= NEED)
        med = st.median(scores[r]["sustained", W].time for r in b
                        if scores[r]["sustained", W].event)
        res.update(W=W, median_crossing=med, T_decisive=T_HALF if med <= T_HALF else T)
    n_floor = sum(len(v) for v in floor.values())
    res["boschloo_b_vs_floor"] = boschloo_p(n_b[1], len(b), fs, n_floor)
    res["boschloo_b_vs_blr0"] = boschloo_p(n_b[1], len(b), floor_n["blr0"], len(floor["blr0"]))
    return res


# ---- markdown ---------------------------------------------------------------
def _f(x, nd=2) -> str:
    return "–" if x is None or x != x else f"{x:.{nd}f}"


def _arm_rows(runs: dict, arm: str, seeds) -> list:
    rows = []
    for s in seeds:
        r = runs.get(rid(arm, s))
        if r is None:
            continue
        x, sc = run_stats(r), score(r)
        e, l = x["early"], x["late"]
        cross = sc["sustained", 1]
        rows.append(
            f"| {arm} | {s} | {x['lives']} | {x['capped']} | {x['sustained_lives']} | "
            f"{int(cross.time) if cross.event else '–'} | {_f(x['first20'])} → {_f(x['last20'])} | "
            f"{_f(e and e['meals'])} → {_f(l and l['meals'])} | "
            f"{_f(e and 100 * e['fed'], 0)} → {_f(l and 100 * l['fed'], 0)} | "
            f"{_f(e and e['unfed_s'], 0)} → {_f(l and l['unfed_s'], 0)} | "
            f"{_f(e and e['reward'], 0)} → {_f(l and l['reward'], 0)} | "
            f"{_f(x['meals_per_1k_last50k'], 3)} | "
            + ("– | " if x["unfinished"] is None else
               f"{x['unfinished'][0]:,} / {x['unfinished'][1]} / {_f(x['unfinished'][2])} | ")
            + f"{' '.join(f'{c} {n}' for c, n in x['causes'].items())}"
            f"{' **CRASH**' if x['crashed'] else ''} |")
    return rows


def _arm_summary(runs: dict, arm: str, seeds) -> str:
    recs = [runs[rid(arm, s)] for s in seeds if rid(arm, s) in runs]
    if not recs:
        return f"| {arm} | 0 of {len(seeds)} | – | – | – | – | – | – | – |"
    xs = [run_stats(r) for r in recs]
    fin = [l for r in recs for l in r["lives"] if l["cause"] != "unfinished"]
    nl = len(fin)
    causes = {c: sum(l["cause"] == c for l in fin) for c in sorted({l["cause"] for l in fin})}
    n1 = sum(score(r)["sustained", 1].event for r in recs)
    return (f"| {arm} | {len(recs)} of {len(seeds)} | {n1} | {nl} ({_f(nl / len(recs), 1)}/run) | "
            f"{_f(100 * sum(x['capped'] for x in xs) / nl if nl else None, 1)} | "
            f"{_f(st.median(l['seconds'] for l in fin) if fin else None, 0)} | "
            f"{_f(st.fmean(_meals(l) for l in fin) if fin else None)} | "
            f"{_f(st.fmean(x['meals_per_1k_last50k'] for x in xs), 3)} | "
            f"{' '.join(f'{c} {n}' for c, n in causes.items())} |")


RUN_HEAD = ("| arm | seed | lives | capped | sustained lives | first sustained at | meals/life first 20 → last 20 | "
            "early → late meals/life | % lives fed | unfed life, median s | reward/life | "
            "meals per 1,000 decisions, last 50k | cut-off life: decisions / meals / intake ÷ drain | ends |\n"
            + "|---" * 14 + "|")
ARM_HEAD = ("| arm | runs in | runs crossing (sustained, W 1) | lives | % capped | life, median s | "
            "meals/life | meals per 1,000, last 50k (mean of runs) | ends |\n" + "|---" * 9 + "|")


def render(runs: dict, src: dict, problems: list) -> tuple:
    """(markdown, complete?)."""
    have = [r for r in REGISTERED if r in runs]
    complete = len(have) == len(REGISTERED) and not problems
    out = ["## Results", ""]
    files = sorted(set(src.values()))
    out.append(f"Generated by `python -m experiments.pilot_b_report` from "
               f"`/data/jack-logs/pilot_b/` ({', '.join(files) or 'no files'}). "
               f"Registered runs in: **{len(have)} of {len(REGISTERED)}**; exploration "
               f"(reported only): {sum(r in runs for r in EXPLORE)} of {len(EXPLORE)}.")
    out.append("")
    if problems:
        out += ["**Books do not balance — no reading:**"] + [f"- {p}" for p in problems] + [""]
    if complete:
        res = reading({r: score(runs[r]) for r in REGISTERED})
        out += [f"### Reading by the rule: **{res['reading']}**", "",
                f"- B runs crossing the sustained mark by T = {T:,}: "
                + ", ".join(f"W = {W}: **{res['n_b'][W]} of 20**" for W in WS)
                + f" (bar: ≥ {NEED} at W = 1).",
                "- B runs crossing θ = 5 meals (secondary): "
                + ", ".join(f"W = {W}: {res['n_theta'][W]} of 20" for W in WS)
                + f" (bar for EATS BUT STARVES: ≥ {NEED} at W = 3).",
                f"- Floor gate: {res['floor_successes']} of 15 floor runs crossed "
                f"({', '.join(f'{a} {n}' for a, n in res['floor_by_arm'].items())}); "
                f"allowed ≤ {FLOOR_MAX_SUCCESSES} → **{'holds' if res['floor_gate'] else 'VOID'}**.",
                f"- Descriptive (harness `boschloo_p`, two-sided): B vs pooled floor "
                f"p = {res['boschloo_b_vs_floor']:.3f}; B vs lr-0 twin p = {res['boschloo_b_vs_blr0']:.3f} "
                f"(harness check #2: the handicapped copy is "
                f"{'told apart' if res['boschloo_b_vs_blr0'] <= 0.05 else 'NOT told apart'} on the mark).",
                (f"- Fixes for the decisive test: W = {res['W']}, T = {res['T_decisive']:,} "
                 f"(median crossing {res['median_crossing']:,.0f})." if res["W"] else
                 "- Fixes: no W is fixed (rule: next are the exploration arms and the "
                 "record's longer budgets, not a decisive test at this T)."),
                "- `prereg.judge` was not the door: its two-look `compare` needs 20 runs in "
                "each arm (floor 15, blr0 5); its three refusals were run here instead "
                "(hashes unmoved, pre-registration committed, #13 books balance).", ""]
    else:
        miss = [r for r in REGISTERED if r not in runs]
        out += ["### PARTIAL — descriptive only, no reading (the rule reads 20 + 15)", "",
                f"Missing: {', '.join(miss)}.", ""]
    out += ["### Per arm", "", ARM_HEAD]
    out.append(_arm_summary(runs, "b", P.B_SEEDS))
    out += [_arm_summary(runs, a, P.FLOOR_SEEDS) for a in P.FLOOR_ARMS]
    out += [_arm_summary(runs, a, P.EXPLORE_SEEDS) + " *(reported only, ruling 28)*"
            for a in P.EXPLORE_ARMS if any(rid(a, s) in runs for s in P.EXPLORE_SEEDS)]
    out += ["", "### Per run", "",
            "Early = a run's first 5 lives (B's random warm-up); late = the last quarter "
            "of its finished lives. A motionless body at this rung lasts ~600 s; a random "
            "one ~300 s — the floor rows below measure both.", "", RUN_HEAD]
    out += _arm_rows(runs, "b", P.B_SEEDS)
    for a in P.FLOOR_ARMS + P.EXPLORE_ARMS:
        out += _arm_rows(runs, a, P.FLOOR_SEEDS)
    out.append("")
    return "\n".join(out), complete


def main(argv) -> int:
    runs, src = load()
    problems = frozen_checks(runs) if all(r in runs for r in REGISTERED) else []
    md, complete = render(runs, src, problems)
    if "--write" in argv:
        text = DOC.read_text()
        DOC.write_text(text[:text.index("## Results")] + md)
        print(f"wrote {DOC} ({'COMPLETE' if complete else 'PARTIAL'})")
    else:
        print(md)
    return 0 if complete else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
