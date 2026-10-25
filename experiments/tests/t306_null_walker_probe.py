"""T3.06 null-walker probe — did the NULL break, the SURROGATE, or the WORLD?

NOT A SPEC. It writes nothing to the ledger, no seed is spent, no gate is
frozen, no constant is moved, and nothing depends on it. T3.06 is VOID and
stays VOID; this file exists because THE RULING on
`t306-random-arm-breaches-the-analytic-chance-dwell-bound` (Review 2026-10-01,
DUE 2026-10-21) ordered a diagnostic that separates three branches it refused
to pick by argument, with every reading and every branch declared BEFORE any
number exists. Committed before it runs, per the ruling.

THE RECORDED FACT UNDER DIAGNOSIS (T3.06 attempt 2, `1b7e7aa`,
2026-09-22T06:49:16, VOID): the n-aware order-statistic cap
`RANDOM_DWELL_MAX = 0.0185` — derived in `_derive_random_dwell_cap()` from a
memoryless clamped step-map surrogate, alpha 0.01 family-wise over 144 lives,
taken at the WORST admissible goal geometry (a corner) — was breached by the
null walker itself on 2 of 3 seeds: `random_dwell_breach` 0.667,
`random_dwell_worst_life` mean 0.0165 +- 0.0041 across seeds,
`random_dwell` mean 0.00303. A family-wise 1-in-100 guard firing twice in one
family is not a draw to re-roll; one of the derivation's inputs is wrong, and
the ruling names the single unchecked one: the surrogate's MEMORYLESSNESS.
The 0.5% step-length check validates step LENGTH only; the velocity servo's
one-decision lag is exactly the mechanism by which realised displacement at
decision t depends on the action at t-1.

THE THREE ORDERED READINGS, for the `random` arm in the ACTUAL world
(fresh CPU replay — the registered artifacts carry per-life summaries only,
so the per-decision trajectory did not survive and the ruling's fallback
applies; ~144 MuJoCo lives, zero GPU, zero world edits, no `playground.py`
byte):

  1. the EMPIRICAL RESIDENCE-RUN PMF for each life's goal cell, pooled,
     against the pmf the derivation's step 2 computes from the chain for the
     same cells (the same object the cap is a tail of);
  2. the LAG-1 AUTOCORRELATION OF REALISED PER-DECISION DISPLACEMENT
     DIRECTION — the memorylessness the surrogate assumes and nothing has
     ever read;
  3. the EMPIRICAL STATIONARY OCCUPANCY of the goal cell, against step 1's
     chain occupancy (0.002021 centre / 0.002530 corner at the reference
     cells; here computed per actual goal cell).

TWO EXTRAS, each justified or it would not be here:
  4. (EXTRA) the same lag-1 statistic computed on a SIMULATED SURROGATE walk
     (the clamped step map itself, same sub-seed RNG streams, same horizon).
     Justification: the surrogate also clamps at walls, and clamping alone can
     correlate consecutive realised directions even in a memoryless chain —
     so the honest null for reading 2 is the surrogate's own lag-1, not an
     assumed zero. The surrogate sim doubles as a Monte-Carlo cross-check of
     this file's generalised chain code (its occupancy and mean run length
     must reproduce the chain's numbers, or the probe indicts itself).
  5. (EXTRA) a least-squares decomposition of realised displacement d_t onto
     the commanded action vectors (a_t, a_{t-1}). Justification: if reading 2
     fires, this names the MECHANISM — a one-decision servo lag shows up as
     mass on a_{t-1} — and separates it from wall-geometry correlation, which
     loads on neither.

SELF-VALIDATION, the first thing to read in the output, all three mandatory:
  V1. one traced life must reproduce `_life(sub_seed, "random")`'s returned
      (coverage, dwell) EXACTLY — the tracer is the rig or it is nothing;
  V2. the traced seed-0 lives must reproduce the ledger row's
      `per_life_first_seed_only` random columns — the replayed value is
      rounded to the RECORDED precision (REC_DP, the recorder's own
      `round(x, 4)`) and compared EXACTLY, which admits precisely the legal
      rounding and nothing more. The first shipped form compared
      `abs(diff) > 5e-5`, a tolerance equal to 4-dp rounding's maximum legal
      error, so float representation decided the boundary and four correct
      replays read as mismatches (queue row
      `t306-probe-returned-an-undeclared-combination-and-v2-false-alarms`;
      a widened epsilon was refused in advance there) — the
      replay is the recorded run or it is nothing;
  V3. the surrogate sim must land within Monte-Carlo error of the chain's
      occupancy and mean run length — the chain code is the derivation's or
      it is nothing.

THE BRANCHES, declared now so the result cannot be read to taste (bands
declared with them; every number below is printed either way):

  ratio bands: AGREE = ratio in [0.80, 1.25] for pooled-occupancy ratio
      (empirical/chain) and pooled-mean-run ratio; tail ratio at k95 (the
      smallest k with chain P(R >= k) <= 0.05) AGREE = in [0.5, 2.0].
      LONGER/ABOVE = above the band's top.
  lag threshold: NON-ZERO = (lag1_dir_world - lag1_dir_surrogate) > 0.05,
      i.e. the world is more direction-persistent than its own surrogate by
      more than 0.05 mean cosine. Under independence the pooled SE of
      lag1_dir is ~0.001 (≈576k pairs, var 0.5), so 0.05 is not a noise line.

  (A) THE NULL IS RIGHT — occupancy AGREE, run pmf AGREE (mean and k95 tail),
      lag MEMORYLESS. The cap stands; the breach was sampling (noting the
      cap's own construction prices that branch at <= 0.01 family-wise, so
      observing it twice is itself evidence against A); the live question is
      the RESOLUTION arm (48 lives at 16.3 informative), a budget question.
  (B) THE SURROGATE IS WRONG — runs LONGER (mean or tail ABOVE band, or
      occupancy ABOVE band) AND lag NON-ZERO. Repair: re-derive the cap over
      the LAGGED kernel inside `t3_06_ablate_curiosity.py` — bills nothing,
      stales no certificate; `_derive_random_dwell_cap()` is never
      hand-edited, and if the honest re-derivation moves the cap UPWARD that
      is published as a finding, not adopted silently. ALPHA and N_LIVES do
      not move. (The re-derivation is its OWN unit, not this probe's.)
  (C) THE WORLD DIFFERS and it is not the lag — runs LONGER AND lag
      MEMORYLESS. Routes to `w0-too-shallow` / `D33`; the world-edit window
      is DECLINED and nothing here re-opens it.
  ANY OTHER COMBINATION (runs SHORTER, occupancy BELOW band, lag non-zero
      with agreeing runs, disagreeing sub-readings...) is reported verbatim
      as UNDECLARED and goes back to the row unpicked. A probe that forces
      every outcome into three boxes is choosing, not measuring.

Artifact: /data/t306_null_walker_probe.json (off-repo scratch, like every
probe before it). Runtime ~11 min CPU, single process.
"""
from __future__ import annotations

import json
import math
import sys
import time

BAND_LO, BAND_HI = 0.80, 1.25     # occupancy + mean-run agreement band
TAIL_LO, TAIL_HI = 0.50, 2.00     # k95 tail agreement band
LAG_NONZERO = 0.05                # world minus surrogate mean-cosine excess
ARTIFACT = "/data/t306_null_walker_probe.json"

# V2's comparison precision IS the recorder's: T3.06 writes every
# per_life_first_seed_only column through round(x, 4)
# (t3_06_ablate_curiosity.py, the per_life_first_seed_only literal). If the
# recorder's precision ever changes, this constant must change with it or V2
# goes blind/false-positive again.
REC_DP = 4


def _v2_match(replayed: float, recorded: float) -> bool:
    """Round the replayed value to the recorded precision, compare exactly.

    Strictly stronger than any epsilon band: the only admitted difference is
    the recorder's own rounding. round() returns the nearest double to the
    decimal result, which is the same double json.load parses from the
    ledger's literal, so == is exact here, not a float-tolerance in disguise.
    """
    return round(replayed, REC_DP) == recorded


def _v2_selftest() -> list:
    """Known answers for _v2_match, run before anything expensive.

    The four passes are the four boundary rows the 2026-10-21 run flagged as
    false alarms (artifact self_validation.V2_replay_is_recorded_run), each a
    legal 4-dp rounding whose float diff computed to one ULP over 5e-5. The
    failures are genuine half-point mismatches that no repair may admit.
    """
    cases = [  # (replayed, recorded, must_match)
        (0.6983471074380165, 0.6983, True),   # sub_seed 200 cov
        (0.00075,            0.0008, True),   # sub_seed 200 dwell (boundary)
        (0.5041322314049587, 0.5041, True),   # sub_seed 291 cov
        (0.01975,            0.0198, True),   # sub_seed 291 dwell (boundary)
        (0.6983471074380165, 0.6984, False),  # real mismatch, one half-point up
        (0.00075,            0.0007, False),  # real mismatch, rounding the
                                              # other way — round() must not
                                              # be allowed to match both
        (0.01975,            0.0199, False),  # real mismatch above the legal
    ]
    return [(r, c, want) for r, c, want in cases if _v2_match(r, c) != want]


def main() -> None:
    import numpy as np

    from experiments.tests.t3_06_ablate_curiosity import (
        _ACTIONS, _CAP_SUBGRID, _DECISION_STEP, _cell, _goal_cell, _life,
        _sub_seeds, CELL_M, GRID_LO, GRID_HI, GRID_N, N_CELLS, N_DECISIONS,
        SPEED, SUBSTEPS)
    from experiments.registry import BY_ID

    t0 = time.time()
    bad = _v2_selftest()
    if bad:
        print(f"V2 SELFTEST FAILED on {bad} — the comparator cannot "
              "demonstrate its own known answers; refusing to run.")
        sys.exit(1)
    print("V2 selftest: 7/7 known answers (4 legal roundings match, "
          "3 real mismatches fire)", flush=True)
    seeds = list(range(BY_ID["T3.06"].seeds))          # [0, 1, 2]
    report: dict = {"readings": {}, "self_validation": {}, "branch": None}

    # ---- the chain, generalised to any cell (the derivation's steps 1-2) ---
    m = GRID_N * _CAP_SUBGRID
    step = CELL_M / _CAP_SUBGRID
    xs = GRID_LO + (np.arange(m) + 0.5) * step

    def axis_map(d):
        v = np.clip(xs + d, GRID_LO, GRID_HI)
        return np.clip(((v - GRID_LO) / step).astype(int), 0, m - 1)

    maps = [(axis_map(_DECISION_STEP * cx), axis_map(_DECISION_STEP * cy))
            for cx, cy in _ACTIONS]

    def push(v):
        out = np.zeros((m, m))
        for jx, jy in maps:                            # rows = y, cols = x
            np.add.at(out, (jy[:, None], jx[None, :]), v / len(_ACTIONS))
        return out

    pi = np.full((m, m), 1.0 / (m * m))
    for _ in range(8000):
        nxt = push(pi)
        if np.abs(nxt - pi).sum() < 1e-14:
            pi = nxt
            break
        pi = nxt
    pi /= pi.sum()

    cell_of = np.repeat(np.arange(GRID_N), _CAP_SUBGRID)

    def chain_cell(goal: int) -> dict:
        """Stationary occupancy, entry rate and residence-run pmf for one cell."""
        cx, cy = goal % GRID_N, goal // GRID_N
        inside = (cell_of[:, None] == cy) & (cell_of[None, :] == cx)
        occ = float(pi[inside].sum())
        entry = np.where(inside, push(np.where(inside, 0.0, pi)), 0.0)
        rate = float(entry.sum())
        v, surv = entry / rate, [1.0]
        for _ in range(400):
            v = np.where(inside, push(v), 0.0)
            surv.append(float(v.sum()))
            if surv[-1] < 1e-16:
                break
        surv = np.array(surv)
        pmf = np.zeros(len(surv))
        pmf[1:] = surv[:-1] - surv[1:]
        er = float((np.arange(len(pmf)) * pmf).sum())
        kac = occ / rate
        return {"occ": occ, "rate": rate, "pmf": pmf, "er": er, "kac": kac}

    # ---- the traced replay of the rig's own random arm ---------------------
    def life_traced(sub_seed: int) -> dict:
        import mujoco
        from experiments.tests.pg_4_noisy_tv import _build
        model, data, _pg, _rb, (ax, ay) = _build()
        rng = np.random.RandomState((sub_seed * 104729 + 7) % (2 ** 32 - 1))
        goal = _goal_cell(sub_seed)
        pos = [(float(data.qpos[-2]), float(data.qpos[-1]))]
        cells, acts, visited, in_goal = [], [], set(), 0
        for _t in range(N_DECISIONS):
            s = _cell(*pos[-1])
            visited.add(s)
            a = int(rng.randint(len(_ACTIONS)))        # the rig's only draw
            acts.append(a)
            data.ctrl[ax] = SPEED * _ACTIONS[a][0]
            data.ctrl[ay] = SPEED * _ACTIONS[a][1]
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)
            p = (float(data.qpos[-2]), float(data.qpos[-1]))
            pos.append(p)
            s2 = _cell(*p)
            cells.append(s2)
            if s2 == goal:
                in_goal += 1
        return {"goal": goal, "cov": len(visited) / N_CELLS,
                "dwell": in_goal / N_DECISIONS,
                "pos": np.array(pos), "cells": np.array(cells),
                "acts": np.array(acts)}

    def runs_of(cells: "np.ndarray", goal: int) -> list:
        out, cur = [], 0
        for c in cells:
            if c == goal:
                cur += 1
            elif cur:
                out.append(cur)
                cur = 0
        if cur:
            out.append(cur)
        return out

    # ---- V1: the tracer IS the rig -----------------------------------------
    v1_seed = _sub_seeds(0)[0]
    ref_cov, ref_dwell = _life(v1_seed, "random")
    tr = life_traced(v1_seed)
    v1_ok = (abs(tr["cov"] - ref_cov) < 1e-12 and
             abs(tr["dwell"] - ref_dwell) < 1e-12)
    report["self_validation"]["V1_tracer_is_rig"] = {
        "sub_seed": v1_seed, "ok": bool(v1_ok),
        "traced": [tr["cov"], tr["dwell"]], "rig": [ref_cov, ref_dwell]}
    print(f"V1 tracer==rig on sub_seed {v1_seed}: {v1_ok} "
          f"(traced {tr['cov']:.4f}/{tr['dwell']:.4f} vs rig "
          f"{ref_cov:.4f}/{ref_dwell:.4f})", flush=True)
    if not v1_ok:
        report["branch"] = "PROBE-INVALID: V1 failed"
        json.dump(report, open(ARTIFACT, "w"), indent=1, default=float)
        sys.exit(1)

    # ---- the replay: 144 traced lives --------------------------------------
    lives: dict = {}
    for sd in seeds:
        for ss in _sub_seeds(sd):
            lives[(sd, ss)] = life_traced(ss)
        print(f"seed {sd} replayed ({time.time()-t0:.0f}s)", flush=True)

    # ---- V2: the replay IS the recorded run --------------------------------
    led = json.load(open("experiments/ledger.json"))
    row = led["results"]["T3.06"]
    rec = row["metrics"]["per_life_first_seed_only"]
    cols = row["metrics"]["per_life_cols"].split()
    i_ss, i_rc, i_rd = (cols.index("sub_seed"), cols.index("random_cov"),
                        cols.index("random_dwell"))
    v2_bad = []
    for r in rec:
        ss = int(r[i_ss])
        lf = lives[(0, ss)]
        if (not _v2_match(lf["cov"], r[i_rc]) or
                not _v2_match(lf["dwell"], r[i_rd])):
            v2_bad.append([ss, lf["cov"], r[i_rc], lf["dwell"], r[i_rd]])
    report["self_validation"]["V2_replay_is_recorded_run"] = {
        "n_rows": len(rec), "mismatches": v2_bad, "ok": not v2_bad}
    print(f"V2 replay==ledger seed0: {not v2_bad} "
          f"({len(rec)} rows, {len(v2_bad)} mismatches)", flush=True)

    # recorded-breach reproduction (per-seed worst dwell vs the cap)
    per_seed_worst = {sd: max(lives[(sd, ss)]["dwell"]
                              for ss in _sub_seeds(sd)) for sd in seeds}
    report["per_seed_worst_dwell"] = per_seed_worst
    print("per-seed worst random dwell:", per_seed_worst, flush=True)

    # ---- chain predictions for the actual goal cells -----------------------
    chain_cache: dict = {}
    for lf in lives.values():
        g = lf["goal"]
        if g not in chain_cache:
            chain_cache[g] = chain_cell(g)
    print(f"chain computed for {len(chain_cache)} distinct goal cells "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- READING 3: occupancy ----------------------------------------------
    emp_goal_decisions = sum(lf["dwell"] * N_DECISIONS for lf in lives.values())
    chain_goal_decisions = sum(chain_cache[lf["goal"]]["occ"] * N_DECISIONS
                               for lf in lives.values())
    occ_ratio = emp_goal_decisions / chain_goal_decisions
    report["readings"]["occupancy"] = {
        "empirical_goal_decisions": emp_goal_decisions,
        "chain_goal_decisions": chain_goal_decisions, "ratio": occ_ratio}

    # ---- READING 1: residence-run pmf --------------------------------------
    all_runs, chain_mix = [], np.zeros(401)
    for lf in lives.values():
        all_runs += runs_of(lf["cells"], lf["goal"])
        ch = chain_cache[lf["goal"]]
        w = ch["rate"] * N_DECISIONS                  # expected entries/life
        pmf = ch["pmf"]
        chain_mix[:len(pmf)] += w * pmf
    n_runs = len(all_runs)
    exp_runs = float(chain_mix.sum())
    chain_mix_pmf = chain_mix / chain_mix.sum()
    er_chain = float((np.arange(len(chain_mix_pmf)) * chain_mix_pmf).sum())
    er_emp = float(np.mean(all_runs)) if all_runs else 0.0
    surv_chain = 1.0 - np.cumsum(chain_mix_pmf)
    k95 = int(np.flatnonzero(surv_chain <= 0.05)[0]) + 1   # P(R >= k95) <= .05
    p_tail_chain = float(chain_mix_pmf[k95:].sum())
    p_tail_emp = (sum(1 for r in all_runs if r >= k95) / n_runs
                  if n_runs else 0.0)
    run_ratio = er_emp / er_chain if er_chain else float("inf")
    tail_ratio = p_tail_emp / p_tail_chain if p_tail_chain else float("inf")
    report["readings"]["residence_runs"] = {
        "n_runs_empirical": n_runs, "n_runs_chain_expected": exp_runs,
        "mean_run_empirical": er_emp, "mean_run_chain": er_chain,
        "run_ratio": run_ratio, "k95": k95,
        "p_tail_empirical": p_tail_emp, "p_tail_chain": p_tail_chain,
        "tail_ratio": tail_ratio,
        "empirical_run_hist": {str(k): all_runs.count(k)
                               for k in sorted(set(all_runs))}}

    # ---- READING 2 + EXTRA 4: lag-1, world vs surrogate --------------------
    def lag1_dir(deltas: "np.ndarray") -> float:
        nrm = np.linalg.norm(deltas, axis=1)
        ok = (nrm[1:] > 1e-6) & (nrm[:-1] > 1e-6)
        u = deltas / np.maximum(nrm, 1e-12)[:, None]
        return float((u[1:][ok] * u[:-1][ok]).sum(axis=1).mean())

    world_lags, step_lens, coef_t, coef_t1 = [], [], [], []
    for lf in lives.values():
        d = np.diff(lf["pos"], axis=0)
        world_lags.append(lag1_dir(d))
        step_lens.append(float(np.linalg.norm(d, axis=1).mean()))
        av = np.array([_ACTIONS[a] for a in lf["acts"]])   # commanded dirs
        X = np.hstack([av[1:], av[:-1]])                   # [a_t, a_t-1]
        beta, *_ = np.linalg.lstsq(X, d[1:], rcond=None)
        coef_t.append(float(np.abs(beta[:2]).mean()))
        coef_t1.append(float(np.abs(beta[2:]).mean()))
    lag1_world = float(np.mean(world_lags))

    sur_lags, sur_runs, sur_goal_dec = [], [], 0.0
    for (sd, ss), lf in lives.items():
        rng = np.random.RandomState((ss * 104729 + 7) % (2 ** 32 - 1))
        p = np.zeros((N_DECISIONS + 1, 2))
        for t in range(N_DECISIONS):
            a = _ACTIONS[int(rng.randint(len(_ACTIONS)))]
            p[t + 1] = np.clip(p[t] + _DECISION_STEP * np.array(a),
                               GRID_LO, GRID_HI)
        sur_lags.append(lag1_dir(np.diff(p, axis=0)))
        scells = np.array([_cell(x, y) for x, y in p[1:]])
        sur_runs += runs_of(scells, lf["goal"])
        sur_goal_dec += float((scells == lf["goal"]).sum())
    lag1_sur = float(np.mean(sur_lags))
    lag_excess = lag1_world - lag1_sur
    report["readings"]["lag1"] = {
        "world": lag1_world, "surrogate": lag1_sur, "excess": lag_excess,
        "mean_realised_step_m": float(np.mean(step_lens)),
        "lstsq_mass_on_a_t": float(np.mean(coef_t)),
        "lstsq_mass_on_a_t_minus_1": float(np.mean(coef_t1))}

    # ---- V3: the chain code reproduces its own surrogate -------------------
    sur_occ_ratio = sur_goal_dec / chain_goal_decisions
    sur_er = float(np.mean(sur_runs)) if sur_runs else 0.0
    v3_ok = 0.9 < sur_occ_ratio < 1.1 and 0.9 < sur_er / er_chain < 1.1
    report["self_validation"]["V3_chain_matches_surrogate_sim"] = {
        "occ_ratio": sur_occ_ratio, "mean_run_sim": sur_er,
        "mean_run_chain": er_chain, "ok": bool(v3_ok)}

    # ---- the branch, mechanically ------------------------------------------
    occ_agree = BAND_LO <= occ_ratio <= BAND_HI
    occ_above = occ_ratio > BAND_HI
    run_agree = (BAND_LO <= run_ratio <= BAND_HI and
                 TAIL_LO <= tail_ratio <= TAIL_HI)
    run_longer = run_ratio > BAND_HI or tail_ratio > TAIL_HI
    lag_nonzero = lag_excess > LAG_NONZERO
    if occ_agree and run_agree and not lag_nonzero:
        branch = "A: null right, breach was sampling"
    elif (run_longer or occ_above) and lag_nonzero:
        branch = "B: surrogate wrong (lagged kernel) — null-model repair"
    elif (run_longer or occ_above) and not lag_nonzero:
        branch = "C: world differs and it is not the lag"
    else:
        branch = "UNDECLARED combination — back to the row unpicked"
    report["branch"] = branch
    report["bands"] = {"BAND": [BAND_LO, BAND_HI], "TAIL": [TAIL_LO, TAIL_HI],
                       "LAG_NONZERO": LAG_NONZERO}
    report["wall_s"] = time.time() - t0

    json.dump(report, open(ARTIFACT, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in report.items()
                      if k != "readings"}, indent=1, default=float))
    print("READING occupancy ratio:", round(occ_ratio, 4))
    print("READING mean-run ratio:", round(run_ratio, 4),
          "tail ratio @k>=%d:" % k95, round(tail_ratio, 4))
    print("READING lag1 world/surrogate/excess:",
          round(lag1_world, 4), round(lag1_sur, 4), round(lag_excess, 4))
    print("BRANCH:", branch)
    print("artifact:", ARTIFACT)


if __name__ == "__main__":
    main()
