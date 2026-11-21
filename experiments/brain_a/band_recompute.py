"""Row A2's band, recomputed from the official per-seed score file (J14 (b), 11-21).

Reads `scores/dmc_proprio-dreamerv3.json.gz` at danijar/dreamerv3@e01491f (local copy
/data/src/dreamerv3_e01491f, md5 08a6a7b1...) and the harvested A2 / A2-v1S run files. Prints:
  1. the method that reproduces rows.py's published numbers (search over file-point windows);
  2. the file's band at the single 10K bin ending at P (the width of OUR score window);
  3. our seeds scored at both widths, both bin conventions.
  5. the file's individual seeds at the band's own 30K window, beside our v1S seeds scored the
     same way: n, sample SD, the 95% interval for ONE new seed from n file seeds
     (m ± t(n-1) · s · sqrt(1 + 1/n)), and how many file seeds each of ours beats.
Reports only; changes no band and no verdict (rows.py's frozen rule is untouched).
Run: python -m experiments.brain_a.band_recompute"""
import glob, gzip, json, statistics as st

SCORES = "/data/src/dreamerv3_e01491f/scores/dmc_proprio-dreamerv3.json.gz"
RUNS = "/data/jack-logs/brain_a/rows/"
CITED = {("dmc_cartpole_balance", 100_000): (794, 52), ("dmc_cartpole_balance", 200_000): (966, 17),
         ("dmc_cartpole_swingup", 100_000): (478, 20), ("dmc_cartpole_swingup", 200_000): (778, 19),
         ("dmc_ball_in_cup_catch", 200_000): (949, 9)}


def file_seeds(d, task, lo, hi):
    out = []
    for x in (x for x in d if x["task"] == task):
        v = [y for xx, y in zip(x["xs"], x["ys"]) if lo <= xx <= hi]
        out.append(st.mean(v))
    return out


def ours(eps, lo, hi, left_closed):
    v = [r for (e, r, _i) in eps if ((lo <= e < hi) if left_closed else (lo < e <= hi))]
    return st.mean(v) if v else None


def main():
    d = json.load(gzip.open(SCORES))
    print("1. cited (rows.py) vs file: exact point P, and mean of file points {P-20K, P-10K, P}, pop SD")
    for (task, P), (m, s) in CITED.items():
        a, b = file_seeds(d, task, P, P), file_seeds(d, task, P - 20_000, P)
        print(f"   {task} {P//1000}K cited {m}±{s} | at P {st.mean(a):.1f}±{st.pstdev(a):.1f}"
              f" | 3-pt {st.mean(b):.1f}±{st.pstdev(b):.1f}")
    print("2. A2 band (mean ± 2 pop SD) by window width, from the file")
    bands = {}
    for P in (100_000, 200_000):
        for w, (lo, hi) in (("10K", (P, P)), ("30K", (P - 20_000, P))):
            v = file_seeds(d, "dmc_cartpole_swingup", lo, hi)
            m, s = st.mean(v), st.pstdev(v)
            bands[(P, w)] = (m - 2 * s, m + 2 * s)
            print(f"   {P//1000}K width {w}: {m:.1f}±{s:.1f} -> band {m-2*s:.0f}..{m+2*s:.0f}")
    print("3. ours, same width as the band; file bins are np.histogram [lo, hi) labelled by right edge")
    for f in sorted(glob.glob(RUNS + "*.json")):
        r0 = json.load(open(f))
        if r0.get("row") != "A2":
            continue
        for r in r0["runs"]:
            cells = []
            for P in (100_000, 200_000):
                for w, W in (("10K", 10_000), ("30K", 30_000)):
                    for lc in (False, True):
                        sc = ours(r["episodes"], P - W, P, lc)
                        cells.append(f"{P//1000}K/{w}/{'[)' if lc else '(]'} {sc:.1f}" if sc is not None else "-")
            print(f"   {r['name']:18s} " + "  ".join(cells))

    print("4. every row's 3-seed mean: rows.py's own window (0.9P, P] vs the cited band's 30K window (P-30K, P]")
    task_of = {"A1": ("dmc_cartpole_balance", (100_000, 200_000)), "A2": ("dmc_cartpole_swingup", (100_000, 200_000)),
               "A3": ("dmc_ball_in_cup_catch", (200_000,))}
    for f in sorted(glob.glob(RUNS + "*.json")):
        r0 = json.load(open(f))
        row = r0.get("row")
        if row not in task_of:
            continue
        seeds = [r for r in r0["runs"] if not r["control"]]
        if not seeds:
            continue
        task, pts = task_of[row]
        for P in pts:
            m, sd = CITED[(task, P)]
            own = st.mean(ours(r["episodes"], 0.9 * P, P, False) for r in seeds)
            w30 = st.mean(ours(r["episodes"], P - 30_000, P, False) for r in seeds)
            print(f"   {f.split('/')[-1]:16s} {row} n={len(seeds)} {P//1000}K band {m-2*sd}..{min(1000, m+2*sd)}"
                  f"  own-window {own:.1f}  30K-window {w30:.1f}")

    print("5. the file's own seeds at the band's 30K window, and our v1S seeds scored the same way (P-30K, P]")
    T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
    v1s = []
    for f in sorted(glob.glob(RUNS + "v1s_*.json")):
        v1s += [r for r in json.load(open(f)).get("runs", []) if not r["control"] and r["name"].startswith("A2v1S")]
    for P in (100_000, 200_000):
        v = sorted(file_seeds(d, "dmc_cartpole_swingup", P - 20_000, P))
        n, m, s = len(v), st.mean(v), st.stdev(v)
        half = T95[n - 1] * s * (1 + 1 / n) ** 0.5
        print(f"   {P//1000}K file n={n}: " + " ".join(f"{x:.1f}" for x in v)
              + f" | mean {m:.1f} sample SD {s:.1f} | frozen band {m-2*st.pstdev(v):.0f}..{m+2*st.pstdev(v):.0f}"
              + f" | 95% one-new-seed interval {m-half:.0f}..{m+half:.0f}")
        for r in v1s:
            sc = ours(r["episodes"], P - 30_000, P, False)
            print(f"      {r['name']:10s} {sc:.1f}  above {sum(sc > x for x in v)} of {n} file seeds,"
                  f" {(sc - m) / s:+.2f} sample SD, {'inside' if abs(sc - m) <= half else 'OUTSIDE'} the one-seed interval")


if __name__ == "__main__":
    main()
