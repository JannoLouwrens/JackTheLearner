"""Row A1-r64 (J28 cell 3): brain A on A1's row at train ratio 64. Rule: docs/research/prereg/a2_ratio64.md
(FROZEN 97762429, before any seed). `rows.py`'s frozen A1-A3 rule is not edited; this module only
builds the arm, its kernel body and its judge.

- Arm: rows.BASE with `algo.replay_ratio=0.0625` (64 / (16 x 64)) and nothing else moved.
- Seeds 20, 21, 22; twin seed 20; no extension.
- HOLDS = every seed inside A1's band at EVERY point, scored on ruling 78's window [P - 30K, P);
  rows.py's (0.9 P, P] printed beside, not read.
"""
import json as _json
import os as _os
import sys as _sys

from experiments.brain_a import rows as R

ROW = "A1"
TAG = "A1r64"
REPLAY_RATIO = "0.0625"
SEEDS = (20, 21, 22)
CONTROL_SEED = 20
EST_HOURS = 1.0
REF_512 = {100_000: (826.3, 843.5, 677.0), 200_000: (973.8, 994.1, 984.3)}   # A1 s0-s2, rows.py window


def base() -> list:
    out = [("algo.replay_ratio=" + REPLAY_RATIO) if o.startswith("algo.replay_ratio=") else o for o in R.BASE]
    assert sum(o.startswith("algo.replay_ratio=") for o in out) == 1
    return out


def score_read(episodes, p: int):
    """Ruling 78's window: episodes ending in [P - 30K, P). None = VOID."""
    xs = [ret for step, ret in episodes if p - 30_000 <= step < p]
    return sum(xs) / len(xs) if xs else None


def judge(per_seed: dict, control_episodes) -> dict:
    out = {"row": TAG, "seeds": sorted(per_seed), "points": {}}
    verdict = "HOLDS"
    if sorted(per_seed) != list(SEEDS) or control_episodes is None:
        return {**out, "verdict": "NOT JUDGED", "why": "seeds or twin missing"}
    for p, (m, sd) in sorted(R.ROWS[ROW]["points"].items()):
        lo, hi = R.band(m, sd)
        sc = {s: score_read(e, p) for s, e in per_seed.items()}
        ctrl = score_read(control_episodes, p)
        rec = {"band": (lo, hi), "scores": sc, "control": ctrl,
               "beside_rows_window": {s: R.score_at(e, p) for s, e in per_seed.items()},
               "beside_a1_512": REF_512[p]}
        if any(v is None for v in sc.values()) or ctrl is None:
            rec["reading"] = "VOID (no episode in window)"
            verdict = "VOID"
        elif ctrl >= lo:
            rec["reading"] = "VOID (twin inside or above band)"
            verdict = "VOID"
        else:
            n_in = sum(lo <= v <= hi for v in sc.values())
            rec["in_band"] = f"{n_in}/{len(sc)}"
            rec["mean_beside"] = sum(sc.values()) / len(sc)
            rec["reading"] = "in band 3/3" if n_in == len(SEEDS) else "MISS"
            if rec["reading"] == "MISS" and verdict == "HOLDS":
                verdict = "MISSES"
        out["points"][p] = rec
    out["verdict"] = verdict
    return out


def judge_files(paths) -> dict:
    per_seed, ctrl, notes = {}, None, []
    for pth in paths:
        for r in _json.load(open(pth))["runs"]:
            ok = "episodes" in r and not r.get("cut") and r.get("rc") == 0 and (r["unit"] or {}).get("ok")
            if not ok:
                notes.append(f"{r['name']}: not complete")
                continue
            eps = [(s, ret) for s, ret, _ in r["episodes"]]
            if r["control"]:
                ctrl = eps
            else:
                per_seed[r["seed"]] = eps
    out = judge(per_seed, ctrl)
    out["notes"] = notes
    return out


_TAIL = r"""
import os
""" + R.ENC_COMPILE_ENV_LINE + r"""
from experiments.brain_a.rows import run_row
from experiments.brain_a.ratio64 import ROW, TAG, CONTROL_SEED, base
_r = run_row(ROW, __SEEDS__, WORK, "cuda", control=__CONTROL__, base=base(), tag=TAG, control_seed=CONTROL_SEED)
os.replace(os.path.join(WORK, "brain_a_rows_" + TAG + ".json"), os.path.join(WORK, __FETCH__))
print("RESULT " + json.dumps({"arm": TAG, "stopped": _r["stopped"],
      "runs": [(x["name"], x.get("rc"), x.get("cut"), len(x.get("episodes", []))) for x in _r["runs"]]}), flush=True)
print("DONE", flush=True)
"""


def fetch_name(seeds, control: bool) -> str:
    return "brain_a_a1r64_" + "_".join(f"s{s}" for s in seeds) + ("_c" if control else "") + ".json"


def kernel_body(seeds, control: bool) -> str:
    """One seed per kernel (the twin rides with seed 20), Kaggle's torch pair pinned as A2-v1S-enc."""
    return R.ENC_TORCH_PIN_LINES + R._kernel_head() + (
        _TAIL.replace("__SEEDS__", repr(list(seeds))).replace("__CONTROL__", repr(control))
        .replace("__FETCH__", repr(fetch_name(seeds, control))))


def main(argv) -> int:
    """kernel SEED [--control]   (Kaggle GPU, one seed a kernel)
    score                      (judge every harvested a1r64 file by the frozen rule)"""
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "kernel":
        from experiments.gpu import build_job, result_json, submit
        seeds = tuple(int(a) for a in argv[2:] if a.isdigit())
        control = "--control" in argv
        if len(seeds) != 1 or seeds[0] not in SEEDS or (control and seeds[0] != CONTROL_SEED):
            print("one seed per kernel from", SEEDS, "; --control only with", CONTROL_SEED)
            return 2
        name = fetch_name(seeds, control)
        sub = submit(build_job(kernel_body(seeds, control)), prefer="kaggle", est_hours=EST_HOURS,
                     timeout_s=R.REMOTE_TIMEOUT_S, fetch=[name])
        print("SUBMIT", sub.backend, sub.ok, sub.job_id, sub.message, flush=True)
        if not sub.ok:
            return 1
        _os.makedirs(R.OUT, exist_ok=True)
        dst = _os.path.join(R.OUT, name.replace("brain_a_", ""))
        open(dst, "w").write(_json.dumps(result_json(sub, name)))
        print("WROTE", dst, flush=True)
        return 0
    if cmd == "score":
        import glob
        paths = sorted(glob.glob(_os.path.join(R.OUT, "a1r64_s*.json")))
        print(_json.dumps(judge_files(paths), indent=1, default=str))
        return 0
    print(main.__doc__)
    return 2


if __name__ == "__main__":
    _sys.exit(main(_sys.argv))
