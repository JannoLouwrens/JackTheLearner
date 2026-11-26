#!/usr/bin/env python
"""Render docs/FORGE_SCOREBOARD.md from the forge's rounds (docs/FORGE.md, F1).

    python -m scripts.forge_scoreboard            # write the scoreboard
    python -m scripts.forge_scoreboard --check    # exit 1 if the file is stale
    python -m scripts.forge_scoreboard --dry      # a synthetic round, to stdout only

Reads `experiments/forge/rounds/<round>/{spec.json,results.jsonl[,long.json]}`
and `experiments/forge/one_number.json`. A pure function of those files: no
dates of its own, so `--check` is stable. `long.json` (optional, written after
rung L is judged by `experiments/harness`) is {candidate: "verdict text"}.
The synthetic round of `--dry` never reaches the real scoreboard.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.forge import Refused, load_round, source_yield, spent  # noqa: E402
from experiments.forge.metric import metric_hash  # noqa: E402

ROUNDS = REPO / "experiments" / "forge" / "rounds"
ONE_NUMBER = REPO / "experiments" / "forge" / "one_number.json"
OUT = REPO / "docs" / "FORGE_SCOREBOARD.md"


def render_continuation(round_dir: Path, *, require_committed: bool = True) -> list:
    """A continuation round (ruling 123, `parent` in its spec): rung L only, no
    ladder, so `load_round`/`source_yield` (judge-locked, S-rung rules) cannot
    read it. Shown as its frozen rule; its rows are read by its own reader,
    never by the ladder's yield."""
    from experiments.forge import _committed_and_clean
    from experiments.forge.continuation import validate_round
    sp = round_dir / "spec.json"
    try:
        if require_committed and not _committed_and_clean(sp):
            raise Refused(f"{sp} is not committed and clean in git")
        spec = json.loads(sp.read_text())
        validate_round(spec)
    except Refused as e:
        return [f"### {round_dir.name} — REFUSED", "", f"Not read: {e}", ""]
    bud, L, par = spec["budget"], spec["rungs"]["L"], spec["parent"]
    pr = spec["outcome"].get("primary", {})
    out = [f"### {spec['round']} — {spec['question']}", "",
           f"Continuation of `{par['round']}` (lock `{par['lock']}`; control: {par['control_source']}). "
           f"Baseline `{spec['baseline']}` · rung L {L['decisions']:,} decisions × {len(L['seeds'])} seeds"
           f" · candidates {', '.join(c['id'] for c in spec['candidates'])} · budget "
           f"{bud['core_hours']:g} core-h, {bud['kernel_hours']:g} kernel-h", ""]
    if pr:
        out += [f"Primary: {pr.get('a')} against {pr.get('b')} by `harness.compare`, "
                f"W = {pr.get('W')}, T = {pr.get('T'):,}.", ""]
    res = round_dir / "results.jsonl"
    out += [("Rows present; the verdict is its own reader's, not this board's."
             if res.is_file() else "Rule frozen; no run read yet."), ""]
    return out


def render_round(round_dir: Path, *, require_committed: bool = True) -> list:
    sp = round_dir / "spec.json"
    if sp.is_file() and "parent" in json.loads(sp.read_text()):
        return render_continuation(round_dir, require_committed=require_committed)
    try:
        spec, rows = load_round(round_dir, require_committed=require_committed)
        y = source_yield(spec, rows)
    except Refused as e:
        return [f"### {round_dir.name} — REFUSED", "", f"Not read: {e}", ""]
    cost, bud = spent(rows), spec["budget"]
    out = [f"### {spec['round']} — {spec['question']}", "",
           f"Baseline `{spec['baseline']}` · outcome `{spec['outcome'].get('metric')}` "
           f"(requires eating) · cost {cost['core_hours']:.2f} of {bud['core_hours']:g} core-h, "
           f"{cost['kernel_hours']:.2f} of {bud['kernel_hours']:g} kernel-h", ""]
    if not y["rungs"]:
        out += ["Rule frozen; no rung read yet.", ""]
    for rung, got in y["rungs"].items():
        size = spec["rungs"][rung]
        head = (f"**Rung {rung}** ({size['decisions']:,} decisions × {len(size['seeds'])} seeds): "
                f"{len(got.entrants)} in, {len(got.climbers)} climb")
        if got.status != "OK":
            out += [head + f" — **{got.status}: over budget, the round stops.**", ""]
            continue
        base = got.means.get(spec["baseline"])
        out += [head + f". Baseline {base:.4g}; plain variation (control mean) "
                f"{got.control_mean:.4g}.", "",
                "| arm | source | mean | tiebreak | runs lost | vs baseline | result |",
                "|---|---|---|---|---|---|---|"]
        src = {c["id"]: c["source"] for c in spec["candidates"]}
        lost = {}
        for r in rows:
            if r.get("rung") == rung and r.get("status", "ok") != "ok":
                lost[r["arm"]] = lost.get(r["arm"], 0) + 1

        def line(c, source, result):
            m, tb = got.means.get(c), got.tiebreaks.get(c)
            return (f"| `{c}` | {source} | {'—' if m is None else f'{m:.4g}'} | "
                    f"{'—' if tb is None else f'{tb:.4g}'} | {lost.get(c, 0)} | "
                    f"{'—' if m is None else f'{m - base:+.4g}'} | {result} |")
        out.append(line(spec["baseline"], "baseline", "the thing to beat"))
        out += [line(t, "never trains", "the twin to beat") for t in spec["twins"]]
        for c in sorted(got.entrants, key=lambda c: (-got.means.get(c, float("-inf")),
                                                     -got.tiebreaks.get(c, 0.0))):
            why = got.reasons[c]
            pair = (got.means.get(c), got.tiebreaks.get(c))
            if "equal on the score at the cut" in why and sum(
                    (got.means.get(x), got.tiebreaks.get(x)) == pair for x in got.entrants) == 1:
                why = f"not in the top third (first below the cut, {pair[0]:.4g})"  # tied with nobody
            out.append(line(c, src[c], why))
        if got.yardstick:
            out.append(line(got.yardstick, "control", "yardstick only"))
        tb_name = spec["outcome"].get("tiebreak")
        out += ["", f"Mean = the outcome over the rung's seeds; a lost run (crash, cut, missing) "
                    f"counts 0. Tiebreak{': ' + tb_name if tb_name else ' (none in this rule): 0'}.", ""]
    long_path = round_dir / "long.json"
    if long_path.is_file():
        out += ["**Rung L** (through `experiments/harness`):", ""]
        out += [f"- `{c}`: {v}" for c, v in sorted(json.loads(long_path.read_text()).items())]
        out.append("")
    out += ["| source | in | reached M | reached L | paying (beats plain variation's yield)? |",
            "|---|---|---|---|---|"]
    word = {None: "not read yet", True: "yes", False: "**NO**"}
    for s, t in y["sources"].items():
        out.append(f"| {s} | {t['in']} | {t['M']} | {t['L']} | "
                   f"{'the yardstick' if s == 'control' else word[t['paying']]} |")
    return out + [""]


def render(rounds_dir: Path = ROUNDS, one_number: Path = ONE_NUMBER, *,
           require_committed: bool = True) -> str:
    out = ["# FORGE SCOREBOARD — what the lab's rounds found, and what they cost", "",
           "Written by `scripts/forge_scoreboard.py`; do not edit by hand. Rungs S and M only "
           "decide who climbs. Only rung L, through `experiments/harness`, says anything about Jack.",
           "", "## The one number", "",
           "The current baseline's sustained-life rate at the entry rung, beside its floor.", ""]
    readings = json.loads(one_number.read_text()) if one_number.is_file() else []
    if readings:
        out += ["| date | baseline | sustained-life rate | floor | source |", "|---|---|---|---|---|"]
        out += [f"| {r['date']} | {r['baseline']} | {r['rate']} | {r['floor']} | {r['source']} |"
                for r in readings]
    else:
        out.append("No reading yet.")
    out += ["", "## Rounds", ""]
    dirs = sorted(d for d in rounds_dir.iterdir() if d.is_dir()) if rounds_dir.is_dir() else []
    if not dirs:
        out += ["No round has been frozen yet.", ""]
    for d in dirs:
        out += render_round(d, require_committed=require_committed)
    return "\n".join(out).rstrip() + "\n"


def synthetic_round(root: Path) -> Path:
    """A planted round: one winner, one null, one control. For --dry and the tests."""
    h = metric_hash()
    spec = {"round": "dry", "question": "SYNTHETIC — planted numbers, not a result",
            "baseline": "base", "twins": ["base-lr0"],
            "candidates": [
                {"id": "winner", "source": "published", "prior": "planted", "twin": "base-lr0"},
                {"id": "null", "source": "inventor", "prior": "planted", "twin": "base-lr0"},
                {"id": "ctl", "source": "control", "prior": "plain variation", "twin": "base-lr0"}],
            "rungs": {"S": {"decisions": 50_000, "seeds": [1, 2, 3]},
                      "M": {"decisions": 300_000, "seeds": [11, 12, 13, 14, 15]}},
            "outcome": {"metric": "sustained_rate", "requires_eating": True, "metric_sha256": h},
            "budget": {"core_hours": 10, "kernel_hours": 10}}
    score = {"winner": 0.6, "null": 0.1, "ctl": 0.1, "base": 0.1, "base-lr0": 0.05}
    rows = [dict(arm=a, rung="S", seed=s, score=v, core_hours=0.1, kernel_hours=0.0,
                 metric_sha256=h) for a, v in score.items() for s in (1, 2, 3)]
    d = root / "dry"
    d.mkdir(parents=True)
    (d / "spec.json").write_text(json.dumps(spec, indent=1) + "\n")
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    if a.dry:
        with tempfile.TemporaryDirectory(dir="/data/tmp" if Path("/data/tmp").is_dir() else None) as t:
            print(render(synthetic_round(Path(t)), Path(t) / "none.json", require_committed=False))
        return 0
    text = render()
    if a.check:
        ok = OUT.is_file() and OUT.read_text() == text
        print("FORGE_SCOREBOARD.md is current" if ok else "FORGE_SCOREBOARD.md is STALE")
        return 0 if ok else 1
    OUT.write_text(text)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
