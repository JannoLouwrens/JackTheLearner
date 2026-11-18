"""J11 item 5 prep: the pilot's kernel packing, its refusal without a rule,
and the child-process path end to end at a plumbing size (T=300, seed 901 —
outside every registered seed range; no learning happens in 300 decisions).
Run with /data/venvs/jack-std/bin/python (sbx lives there), JACK_PIN_CPU=3.

Known-positive / must-fail: pilot_kernel with PILOT_RULE None must raise.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments import pilot_b2 as P  # noqa: E402
from experiments.pilot_b import cells_from_artifacts  # noqa: E402


def main():
    ok = True
    seeds = range(402, 407)
    g = P.pilot_groups(seeds)
    cells = [c for v in g.values() for c in v]
    want = set((a, s) for a in P.PILOT_LEARNERS + P.PILOT_FLOORS for s in seeds)
    good = (len(g) == 10 and len(cells) == 40 and set(cells) == want
            and all(len(v) == 4 for v in g.values()))
    learners_per = {k: sum(a in P.PILOT_LEARNERS for a, _ in v) for k, v in g.items()}
    arms_per = {k: {a for a, _ in v} for k, v in g.items()}
    # no learner arm wholly on one kernel; floors only where <= 2 learners
    spread = all(sum(a in arms for arms in arms_per.values()) >= 4 for a in P.PILOT_LEARNERS)
    floors_ok = all(learners_per[k] <= 2 for k, v in g.items()
                    if any(a in P.PILOT_FLOORS for a, _ in v))
    print(f"packing: {len(g)} kernels, {len(cells)} runs, exact cover {good}, "
          f"learners/kernel {list(learners_per.values())}, arm spread {spread}, floors {floors_ok}")
    ok &= good and spread and floors_ok and set(P.PILOT_ARMS) == set(P.PILOT_LEARNERS + P.PILOT_FLOORS)

    # must-fail: no rule -> refuse (before any import of the submit path)
    try:
        P.pilot_kernel("p0")
        print("refusal: pilot_kernel ran WITHOUT a rule -> BAD")
        ok = False
    except RuntimeError as e:
        print(f"refusal without rule: ok ({str(e)[:60]}...)")
    ok &= P.PILOT_RULE is None

    body = P.pilot_job_body(g["p7"], True, 300, "test")
    compile(body, "job", "exec")
    ok &= "'--food-floor-only'" not in body and "True" in body and "pilot_b2_cells.json" in body
    print("job body compiles: ok")

    # end to end: two children (a learner arm and a floor arm), T=300
    with tempfile.TemporaryDirectory(dir="/data/tmp" if Path("/data/tmp").is_dir() else None) as d:
        cs = [("K", 901), ("held8", 901)]
        r = P.run_pilot_parallel(cs, d, food_floor_only=True, T=300)
        for k, rec in r.items():
            st = rec.get("world", {})
            g1 = (rec["rc"] == 0 and rec["decisions"] >= 300 and rec["pilot_arm"] == k.split("_s")[0]
                  and st.get("food_floor_only") is True
                  and st.get("move_cost_dial") is (k.startswith("K")))
            print(f"child {k}: rc {rec['rc']} decisions {rec.get('decisions')} "
                  f"lives {len(rec['lives'])} world ff={st.get('food_floor_only')} "
                  f"dial={st.get('move_cost_dial')} {'ok' if g1 else 'BAD ' + rec.get('error', '')}")
            ok &= g1
        # reassembly from per-cell files (Kaggle-cap path) finds both
        arts = {p.name: str(p) for p in Path(d).iterdir()}
        a = cells_from_artifacts(arts, cs)
        g2 = a is not None and all(len(a["cells"][f"{x}_s{s}"]["lives"]) == len(r[f"{x}_s{s}"]["lives"])
                                   for x, s in cs)
        print(f"reassembly from cell files: {'ok' if g2 else 'BAD'}")
        ok &= g2
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
