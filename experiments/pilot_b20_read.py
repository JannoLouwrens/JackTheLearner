"""J32 — the ONE read of `docs/research/prereg/pilot_b20.md` (a SIZING pilot, ruling 141).

Reads the 9 assembled raw files `/data/jack-logs/pilot_b20/chains/b20_s<unit>.json`
that `j24_launch --variant b20` writes (25 cells: B20-bins16 on 2001-2020, its
blind twin on 2001-2005) and prints the prereg's outputs 1-6:

  1. late meals a life per run (lives STARTING >= 500,000; `pilot_b2.j17_late_meals`)
  2. meals a life per 50k window, every run (20 windows)
  3. the blind-twin mean B
  4. the share p = (sighted seeds with late meals > 2 x B) / 20, exact 95 % CI
  5. the per-arm n a race needs for 80 % power, two-sided exact Boschloo at
     0.05, worse arm at p - 0.3 / 0.4 / 0.5 (clipped at 0), by exact
     enumeration; beside it the sighted late-meal spread
  6. "B+K1 still eats at 1M" HOLDS iff >= 15 of 20 sighted seeds end above 2 x B,
     else NOT HELD AT 1M.

Refusals (nothing is read): the pin is not a commit, or `pilot_b20.md` is not
FROZEN at it; a raw file carries another pin; a cell is missing or not whole
(decisions != 1M) and was not declared lost. A LOST run (declared on the
command line, `--lost B20-bins16_s2007`, after the chain's one relaunch) counts
as NOT above the bar and is left out of the spread; declaring a run lost whose
whole record IS home refuses. A sighted run with no finished life starting at
>= 500k (late meals nan) counts as not above the bar. B is the mean over the
blind runs home; no blind run home -> refuses.

    python -m experiments.pilot_b20_read --pin <freeze sha> [--lost ARM_sSEED ...] [--out FILE]
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from experiments import pilot_b2

T = pilot_b2.B20_T
LATE_FROM = 500_000
WINDOW = 50_000
SIGHTED, BLIND = "B20-bins16", "B20-bins16-blind"
SEEDS, BLIND_SEEDS = pilot_b2.B20_SEEDS, pilot_b2.B20_BLIND_SEEDS
HOLD_NEED = 15
GAPS = (0.3, 0.4, 0.5)
ALPHA, POWER = 0.05, 0.80
N_MAX = 150
RAW = Path("/data/jack-logs/pilot_b20/chains")
PREREG = "docs/research/prereg/pilot_b20.md"


class Refused(ValueError):
    pass


# ---------------------------------------------------------------- the freeze
def check_pin(pin: str) -> str:
    from experiments.j24_launch import REPO, frozen
    p = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify", f"{pin}^{{commit}}"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise Refused(f"pilot_b20_read: {pin!r} is not a commit")
    full = p.stdout.strip()
    text = subprocess.run(["git", "-C", str(REPO), "show", f"{full}:{PREREG}"],
                          capture_output=True)
    if text.returncode != 0 or not frozen(text.stdout):
        raise Refused(f"pilot_b20_read: {PREREG} is not FROZEN at {full[:8]}")
    return full


# ---------------------------------------------------------------- the runs
def pack() -> dict:
    from experiments.j24_launch import b20_pack
    return b20_pack()


def load(pin: str, lost=(), root=None) -> dict:
    """{(arm, seed): rec or None (lost)}; refuses anything not whole and not declared lost."""
    root = RAW if root is None else Path(root)
    lost = set(lost)
    names = {f"{a}_s{s}" for u in pack().values() for a, s in u}
    if lost - names:
        raise Refused(f"pilot_b20_read: --lost names no registered run: {sorted(lost - names)}")
    out = {}
    for unit, cl in pack().items():
        f = root / f"b20_s{unit}.json"
        d = json.loads(f.read_text()) if f.exists() else None
        if d is not None and d.get("pin") != pin:
            raise Refused(f"pilot_b20_read: {f.name} carries pin {d.get('pin')}, not the freeze {pin}")
        for arm, seed in cl:
            key = f"{arm}_s{seed}"
            rec = (d or {}).get("cells", {}).get(key)
            whole = (rec is not None and "error" not in rec and rec.get("decisions") == T
                     and sum(l["decisions"] for l in rec["lives"]) == T)
            if key in lost:
                if whole:
                    raise Refused(f"pilot_b20_read: {key} is home whole — it cannot be entered lost")
                out[(arm, seed)] = None
            elif not whole:
                why = "missing" if rec is None else f"not whole ({rec.get('decisions')} decisions)"
                raise Refused(f"pilot_b20_read: {key} {why} — wait, relaunch once, or --lost it")
            else:
                out[(arm, seed)] = rec
    return out


def window_table(rec: dict) -> list:
    """meals a life per 50k window, a life assigned to the window it ENDS in (J24/J30's table)."""
    from experiments import pilot_b_report as R
    rows = [[] for _ in range(T // WINDOW)]
    t = 0
    for life in rec["lives"]:
        t += life["decisions"]
        rows[min((t - 1) // WINDOW, len(rows) - 1)].append(R._meals(life))
    return [round(sum(r) / len(r), 3) if r else None for r in rows]


# ---------------------------------------------------------------- exact Boschloo power
def _log_binom(n: int):
    from scipy.special import gammaln
    k = np.arange(n + 1)
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def boschloo_two_sided(n: int, grid: int = 2001) -> np.ndarray:
    """P[x1, x2]: the two-sided Boschloo p-value (scipy's definition: twice the
    smaller one-sided p, clipped at 1) of every table x1/n vs x2/n, the nuisance
    sup taken on a grid of `grid` points in (0, 1)."""
    from scipy.stats import hypergeom
    x1 = np.arange(n + 1)[:, None]
    x2 = np.arange(n + 1)[None, :]
    # one-sided 'less' statistic (Fisher's p, scipy's layout): small when x1 << x2
    stat = hypergeom.cdf(x1, 2 * n, x1 + x2, n).ravel()
    order = np.argsort(stat, kind="stable")
    s_sorted = stat[order]
    # tables whose stat <= observed * (1 + 1e-13), as scipy's tolerance
    upto = np.searchsorted(s_sorted, stat * (1 + 1e-13), side="right")
    lb = _log_binom(n)
    lb_sum = (lb[:, None] + lb[None, :]).ravel()[order]
    xs = (x1 + x2).ravel()[order]
    p_less = np.zeros(stat.size)
    pis = np.linspace(0, 1, grid + 2)[1:-1]
    for blk in np.array_split(pis, max(1, grid // 100)):
        lp = (lb_sum[:, None] + xs[:, None] * np.log(blk)[None, :]
              + (2 * n - xs)[:, None] * np.log1p(-blk)[None, :])
        cum = np.cumsum(np.exp(lp), axis=0)
        p_less = np.maximum(p_less, cum[upto - 1].max(axis=1))
    p_less = np.minimum(p_less, 1.0).reshape(n + 1, n + 1)
    p_greater = p_less.T            # equal arms: 'greater' is 'less' with the arms swapped
    return np.minimum(2 * np.minimum(p_less, p_greater), 1.0)


def power(n: int, pa: float, pb: float, P=None) -> float:
    from scipy.stats import binom
    P = boschloo_two_sided(n) if P is None else P
    k = np.arange(n + 1)
    w = binom.pmf(k, n, pa)[:, None] * binom.pmf(k, n, pb)[None, :]
    return float(w[P <= ALPHA].sum())


def n_needed(pa: float, pb: float, n_max: int = N_MAX, cache=None):
    """Smallest per-arm n with power >= 80 % (exact enumeration), or None past n_max."""
    if pa - pb <= 0:
        return None
    cache = {} if cache is None else cache
    for n in range(2, n_max + 1):
        if n not in cache:
            cache[n] = boschloo_two_sided(n)
        if power(n, pa, pb, cache[n]) >= POWER:
            return n
    return None


# ---------------------------------------------------------------- THE READ
def read(runs: dict) -> dict:
    from scipy.stats import binomtest
    late = {k: (None if r is None else pilot_b2.j17_late_meals(r, LATE_FROM))
            for k, r in runs.items()}
    blind = [late[(BLIND, s)] for s in BLIND_SEEDS
             if late[(BLIND, s)] is not None and not math.isnan(late[(BLIND, s)])]
    if not blind:
        raise Refused("pilot_b20_read: no blind-twin run home with a late life — B undefined")
    bbar = sum(blind) / len(blind)
    sighted = {s: late[(SIGHTED, s)] for s in SEEDS}
    above = [s for s, v in sighted.items() if v is not None and not math.isnan(v) and v > 2 * bbar]
    k = len(above)
    ci = binomtest(k, len(SEEDS)).proportion_ci(confidence_level=0.95, method="exact")
    p_hat = k / len(SEEDS)
    cache = {}
    n_table = {}
    for g in GAPS:
        pb = max(0.0, p_hat - g)
        n = n_needed(p_hat, pb, cache=cache)
        n_table[str(g)] = {"p_better": p_hat, "p_worse": round(pb, 4),
                           "n_per_arm": n if n is not None else
                           ("no gap" if p_hat - pb <= 0 else f"> {N_MAX}")}
    vals = [v for v in sighted.values() if v is not None and not math.isnan(v)]
    spread = ({"n": len(vals), "mean": float(np.mean(vals)),
               "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
               "min": min(vals), "max": max(vals)} if vals else {"n": 0})
    return {
        "late_meals": {f"{a}_s{s}": ("LOST" if v is None else v) for (a, s), v in late.items()},
        "windows": {f"{a}_s{s}": window_table(r) for (a, s), r in runs.items() if r is not None},
        "blind_mean": bbar, "blind_n": len(blind), "bar": 2 * bbar,
        "above": above, "k": k, "share": p_hat, "share_ci95": [ci.low, ci.high],
        "n_table": n_table, "sighted_spread": spread,
        "lost": sorted(f"{a}_s{s}" for (a, s), r in runs.items() if r is None),
        "nan_sighted": sorted(s for s, v in sighted.items() if v is not None and math.isnan(v)),
        "verdict": "HOLDS" if k >= HOLD_NEED else "NOT HELD AT 1M",
    }


def main(argv=()):
    argv = list(argv)
    if "--pin" not in argv:
        raise Refused("pilot_b20_read needs --pin <pilot_b20.md freeze sha>")
    pin = check_pin(argv[argv.index("--pin") + 1])
    lost = [argv[i + 1] for i, a in enumerate(argv) if a == "--lost"]
    out_file = Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    if out_file is not None and out_file.exists():
        raise Refused(f"pilot_b20_read: {out_file} exists — the pilot is read ONCE")
    out = {"pin": pin, "late_from": LATE_FROM, **read(load(pin, lost))}
    text = json.dumps(out, indent=1, default=str)
    print(text)
    if out_file is not None:
        out_file.write_text(text + "\n")
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
