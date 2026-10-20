"""LC.08 — the Learning-core seat's binding objective, paired on/off.

THE QUESTION (field watch week 8 §6; routed as
`lc03-five-controls-never-switch-off-the-term-a4-is-named-for`; RULED
2026-09-29, re-dated 2026-10-10, DUE 2026-10-21). `A4` (`wm-latent`) holds the
Learning-core seat on lg_margin_null t = 4.64 and lg_margin_twin t = 4.00. The
machinery it is NAMED for — the `latent_pred` head, 149,312 params, 17.3% of
the arm — is switched off by NO control in the screen that seated it: all five
switch it off only as part of switching everything off. So the seat's margins
are equally consistent with an RSSM actor-critic having produced the entire
number. `D37` fired 2026-10-05 to (iii) — the declared collapse diagnostic is
settled BY HOLD and never built — which makes this comparator THE ONLY LIVE
MEASUREMENT the seat will ever get (the ruling's own pre-registered condition,
and it fired upward).

THE TWO CONJUNCTS, from THE RULING verbatim in substance, neither droppable to
make a result reportable:

  1. PAIRED, not merely adjacent — identical seeds (0/1/2), identical step
     budget (N_STEPS=400,000 decisions AND W_CLOCK=17,280 core-s, whichever is
     later), identical venue (lethal W0, e0=1.0, constant explore_std 0.3,
     LC.02's committed train_ratio, the same record flags inside the measured
     clock), so the only declared difference is the dropped term. Enforced
     structurally: the envelope is READ FROM the seated run's own artifacts
     (`experiments/artifacts/lc03_curves_seed{N}.json`), the calibration comes
     through `borrow_metrics` from PS.01/LC.02 exactly as LC.03's did, the
     rig constants are imported from the LC.03 module, and the comparator's
     param count must equal the artifact's recorded 861,545 to the digit.
  2. SCORED ON THE SEAT'S OWN STATISTICS — lg_margin_null and lg_margin_twin,
     the two numbers D10 seated A4 on, paired per seed against the SAME
     recorded null_random and wm-latent/twin life histories, never a proxy
     chosen afterwards.

THE DECISION RULE, pre-registered before any number exists. Let seat_mean and
seat_std (ddof=0 — the house t's own denominator) be the seated run's margins
recomputed from the artifacts and verified against the recorded row's
aggregates, pinned with provenance in SEAT_RECORDED (a live ledger read of
another spec's row is forbidden, T0.22 P9; the pin is fail-safe — see the
constant's comment). VOID on mismatch: pairing targets that do not
reconstruct pair nothing.

  PASS (load-bearing):  (seat_mean - comp_mean) > 1.0 * seat_std on BOTH
                        statistics — dropping the term costs more than the
                        seat's own seed noise on both numbers it was bought on.
  FAIL (otherwise):     the comparator reproduces one or both margins within
                        the seated arm's own seed spread. When BOTH reproduce,
                        `deadweight_reading` = 1.0 is recorded: the ruling's
                        named finding that 17.3% of the seat is dead weight —
                        a reportable result about the seat, never a null. A
                        mixed or inverted outcome (comparator ABOVE the seat)
                        is also FAIL, with the per-statistic deficits on the
                        row telling the story; no reading beyond the recorded
                        arithmetic is claimed for it.

STATISTIC_BOUND: none. The deciding statistic (difference of paired margins)
is unbounded in both directions; its anchor (the seated mean) sits 4.6/4.0
house-sigma from zero. The null is nowhere near a bound, so the
UNSATURATED-NULL RULE's registration condition is met and declared.

FORECAST, pre-registered with its arithmetic (ps09 idiom): lean FAIL with the
dead-weight reading. ppo-needs — the screen's pure actor-critic — recorded
mean lg_margin_null 67.4 in the same venue, ABOVE this spec's reproduce cutoff
(100.1 - 37.3 = 62.8), so an actor-critic has already produced a
reproducing-range number here once; and the untrained twin (-0.83) proves the
seat's margin came from training, not that it came from the binding term.
Either branch is a result the ruling priced the 14.4 core-h for.

THE MECHANISM — zero the gradient, keep the graph. `_NoBindWMLatent` computes
the full `binding_loss` (per-key leave-one-out latent attribution + KL, the
seated arm's own code path) and multiplies the total by zero: backward
traverses the same graph, per-update compute stays matched to the seated run
inside the W_CLOCK budget, and the term's gradient contribution is exactly
zero (Adam with a zero grad and zero state takes a zero step). The smoke
asserts both halves: `latent_pred`/`post`-via-KL receive no update from the
binding path, and the architecture is init-identical to `build_arm
("wm-latent")` under the rig's own `torch.manual_seed(9000 + seed)`.

VENUE IDENTITY, verified at registration rather than assumed: zero commits to
playground.py / experiments/w0.py / survival.py / cores.py / drives.py since
2026-08-20 (`git log --since=2026-08-23` empty; last rig commit `3fa7d61`);
the seated run executed 2026-08-21 04:22 -> 2026-08-23 21:11. The artifacts
are on-box and gitignored; their integrity gate is the replay: the
recomputed seated aggregates must reproduce the pinned recorded aggregates
(SEAT_RECORDED; LC.03 attempt 3, ran_at 2026-08-23T21:11:17) to 1e-4 or the
run is VOID.

WHAT THIS SPEC DOES NOT DO: it re-runs nothing of LC.03, moves no LC.03
verdict (the screen's VOID stands), renames no metric, and does not pre-empt
`T1.08` — THE RULING places it explicitly behind the pipeline repair in
priority; it is bought because it is cheap (14.4 core-h, 2.7% of the LC.07
re-run the seat cannot afford) and decisive, not because it is urgent.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ..cores import WorldModelCore
from ..protocol import Ledger, Status, borrow_metrics, run_spec
from ..registry import BY_ID
from .. import survival as _survival
from ..survival import run_survival
from .lc_02_throughput_floor import TRAIN_RATIOS, committed_ratio
from .lc_03_survival_screening import (EXPLORE_STD_LC03, N_LIVES_MIN,
                                       TRACE_EVERY, _panel_dwell)

IMPL_DEPS = ["experiments/survival.py", "experiments/cores.py",
             "experiments/w0.py", "experiments/drives.py", "playground.py"]

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO / "experiments" / "artifacts"

SEAT = "wm-latent"              # the seated arm, as recorded
ARM = "wm-latent-nobind"        # this spec's comparator
SPREAD_MULT = 1.0               # the ruling's own cutoff: one seated seed std
TOL_REPLAY = 1e-4               # artifact -> recorded-aggregate reconstruction
TOL_IDENTITY = 1e-9             # control: scoring the seat's own data

# The recorded v2 row's wm-latent aggregates, PINNED as the replay anchor
# (LC.03 attempt 3, ran_at 2026-08-23T21:11:17, read from
# experiments/ledger.json at registration, commit da9ece0). A LIVE ledger read
# here is forbidden by T0.22 P9 — no test may read another spec's metrics off
# the ledger directly — and `borrow_metrics` correctly REFUSES this source
# (VOID, P6), so the pin is the only honest anchor. It is also fail-safe in
# the one drift direction that exists: LC.03 is terminally settled under a
# standing do-not-re-dispatch order, and the only event that could move its
# row (a forbidden re-run) would regenerate the artifacts, whose recomputed
# aggregates would then MISMATCH these pins and VOID this spec's run rather
# than silently re-pairing it. Stds are ddof=0, the house t's own denominator.
SEAT_RECORDED = {
    f"{SEAT}/lg_margin_null": 100.101537,
    f"{SEAT}/lg_margin_null_std": 37.325548,
    f"{SEAT}/lg_margin_twin": 97.659971,
    f"{SEAT}/lg_margin_twin_std": 42.252817,
    f"{SEAT}/life_gain": 96.831082,
}


# ── the comparator core ─────────────────────────────────────────────────────
class _NoBindWMLatent(WorldModelCore):
    """A4's architecture byte-for-byte — same __init__, same RNG draw order,
    latent_pred present and parameter-counted — with the binding objective's
    gradient contribution zeroed. `total * 0.0` on purpose, not an early
    return: backward traverses the seated arm's own graph, so per-update
    compute stays matched inside the W_CLOCK budget, and every grad from this
    term is exactly zero."""

    binding_term = "ZEROED (LC.08 comparator): computed, then multiplied by 0"

    def binding_loss(self, obs, dropped=()):
        total, per = super().binding_loss(obs, dropped)
        return total * 0.0, per


_REAL_BUILD_ARM = _survival.build_arm


def _patched_build_arm(name: str, with_placebo: bool = True):
    if name == ARM:
        return _NoBindWMLatent(with_placebo, decoder=False)
    return _REAL_BUILD_ARM(name, with_placebo)


def _install_arm():
    """survival.run_survival resolves arms through its module-level
    `build_arm`; this points that name at the one-arm extension above, in THIS
    process (run() uses spawn workers, so each worker installs its own).
    Delegation for every other name is exact — the rig is not edited."""
    _survival.build_arm = _patched_build_arm


# ── pairing targets, recomputed never pasted ────────────────────────────────
def _gain(spans) -> float:
    """LC.03's life_gain, verbatim: final-third minus first-third mean span."""
    third = len(spans) // 3
    return (float(np.mean(spans[-third:]) - np.mean(spans[:third]))
            if third >= 1 else 0.0)


def _load_artifact(seed: int) -> Tuple[Optional[dict], str]:
    p = ARTIFACTS / f"lc03_curves_seed{seed}.json"
    if not p.exists():
        return None, f"{p.name} missing (on-box, gitignored — this venue only)"
    try:
        art = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return None, f"{p.name} unreadable: {e}"
    runs = art.get("runs", {})
    for k in ("null_random", SEAT, f"{SEAT}/twin"):
        if k not in runs or not runs[k].get("life_spans"):
            return None, f"{p.name} carries no life_spans for {k!r}"
    return art, ""


def _borrow() -> Tuple[Optional[dict], dict]:
    """PS.01's j0/alpha and LC.02's committed wm-latent ratio — LC.03's own
    calibration path, scoped to the one arm this spec runs."""
    b1 = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    if not b1.ok:
        return None, dict(b1.provenance, refusal=b1.refusal)
    keys = [f"{SEAT}/clears@{r}" for r in TRAIN_RATIOS]
    b2 = borrow_metrics("LC.02", keys)
    if not b2.ok:
        return None, dict(b2.provenance, refusal=b2.refusal)
    ratio = committed_ratio(b2.values, SEAT)
    if ratio is None:
        return None, dict(b2.provenance,
                          refusal=f"LC.02 committed no ratio for {SEAT}")
    return ({"j0": b1.values["j0_ms"], "alpha": b1.values["alpha"],
             "ratio": float(ratio)}, dict(b2.provenance))


def _venue_echo(n_steps: int, w_clock: float, e0: float) -> str:
    """Derived from the artifact values the run genuinely loaded — never
    copied from the declaration (protocol.Spec.venue's own rule)."""
    return (f"W0 lethal v2 screening envelope | axis: binding objective "
            f"off, all else as seated | {int(n_steps)} decisions, "
            f"{w_clock:g} core-s, e0 {e0:g}, paired to lc03_curves seeds 0-2")


def _pairing(seed: int, art: dict) -> dict:
    """Per-seed pairing targets plus the seat replay, from the artifact."""
    runs = art["runs"]
    null_lg = _gain(runs["null_random"]["life_spans"])
    twin_lg = _gain(runs[f"{SEAT}/twin"]["life_spans"])
    seat_lg = _gain(runs[SEAT]["life_spans"])
    return {
        "paired_null_life_gain": null_lg,
        "paired_twin_life_gain": twin_lg,
        "seat_life_gain_replayed": seat_lg,
        "seat_lg_margin_null_replayed": seat_lg - null_lg,
        "seat_lg_margin_twin_replayed": seat_lg - twin_lg,
        "seat_params_recorded": float(runs[SEAT].get("params", 0.0)),
        "venue_n_steps": float(art["n_steps"]),
        "venue_w_clock": float(art["w_clock_core_s"]),
        "venue_e0": float(art["e0"]),
    }


# ── the experiment ──────────────────────────────────────────────────────────
def _experiment(seed: int, n_steps: Optional[int] = None,
                w_clock: Optional[float] = "artifact",
                write_artifact: bool = True) -> dict:
    """One seed: load the seated run's artifact, re-derive the pairing
    targets, run the l_bind-dropped arm at the artifact's own envelope, score
    on the seat's statistics. `n_steps`/`w_clock` overrides exist for the
    smoke only — the registered run always inherits the artifact's envelope."""
    m: dict = {}
    art, why = _load_artifact(seed)
    if art is None:
        m["artifact_ok"] = 0.0
        m["artifact_refusal"] = why
        return m
    m["artifact_ok"] = 1.0
    m.update(_pairing(seed, art))

    cal, prov = _borrow()
    m.update(prov)
    if cal is None:
        m["borrowed_ok"] = 0.0
        return m
    m["borrowed_ok"] = 1.0
    m["train_ratio"] = cal["ratio"]

    env_steps = int(art["n_steps"]) if n_steps is None else int(n_steps)
    env_clock = (float(art["w_clock_core_s"]) if w_clock == "artifact"
                 else w_clock)
    env_e0 = float(art["e0"])
    m["venue_used"] = _venue_echo(art["n_steps"], art["w_clock_core_s"],
                                  art["e0"])

    _install_arm()
    r = run_survival(seed, j0=cal["j0"], alpha=cal["alpha"], e0=env_e0,
                     policy="core", arm=ARM, train=True,
                     train_ratio=cal["ratio"], n_decisions=env_steps,
                     record_xy=True, record_transitions=TRACE_EVERY,
                     min_core_s=env_clock, explore_std=EXPLORE_STD_LC03)

    m["life_gain"] = r["life_gain"]
    m["mean_life_s"] = r["mean_life_s"]
    m["n_lives"] = r["n_lives"]
    m["lives_ok"] = float(r["n_lives"] >= N_LIVES_MIN)
    m["lg_margin_null"] = r["life_gain"] - m["paired_null_life_gain"]
    m["lg_margin_twin"] = r["life_gain"] - m["paired_twin_life_gain"]
    m["needs_rise"] = r["needs_ok_final_third"] - r["needs_ok_first_third"]
    m["physics_finite"] = r["physics_finite"]
    m["params"] = r.get("params", 0.0)
    m["params_match"] = float(m["params"] == m["seat_params_recorded"])
    m["decisions"] = r["decisions"]
    m["core_s"] = r["process_time_s"]
    m["optimiser_steps"] = r["optimiser_steps"]
    m["panel_dwell"] = _panel_dwell(r["xy"], _panel_pos(seed, cal))

    if write_artifact:
        ARTIFACTS.mkdir(exist_ok=True)
        keep = {k: r[k] for k in ("life_spans", "life_ends",
                                  "deaths_at_decision", "optimiser_steps",
                                  "decisions", "sim_seconds",
                                  "process_time_s", "reward_sum")}
        keep["params"] = r.get("params", 0.0)
        (ARTIFACTS / f"lc08_curves_seed{seed}.json").write_text(
            json.dumps({"seed": seed, "e0": env_e0, "n_steps": env_steps,
                        "w_clock_core_s": env_clock, "arm": ARM,
                        "runs": {ARM: keep}}, indent=1))
    return m


def _panel_pos(seed: int, cal: dict):
    from ..w0 import W0
    w = W0(seed=seed, j0=cal["j0"], alpha=cal["alpha"])
    xy = (np.array(w.model.geom_pos[w.panel_gid][:2], dtype=float)
          if w.panel_gid >= 0 else None)
    del w
    return xy


# ── the control: the seat's own data through the identical scoring path ────
def _control(seed: int) -> dict:
    """comparator := seat. Must land exactly on the recorded margins with
    both deficits zero, and the claim branch must NOT fire on it."""
    c: dict = {}
    art, why = _load_artifact(seed)
    if art is None:
        c["artifact_ok"] = 0.0
        c["artifact_refusal"] = why
        return c
    c["artifact_ok"] = 1.0
    c.update(_pairing(seed, art))
    seat_lg = c["seat_life_gain_replayed"]
    c["life_gain"] = seat_lg
    c["lg_margin_null"] = seat_lg - c["paired_null_life_gain"]
    c["lg_margin_twin"] = seat_lg - c["paired_twin_life_gain"]
    return c


# ── the pre-registered check ────────────────────────────────────────────────
def _void(m: dict, reason: str):
    m["void_reason"] = reason
    return Status.VOID


def _deficits(m: dict) -> Tuple[float, float, float, float]:
    dn = m.get("seat_lg_margin_null_replayed", 0.0) - m.get("lg_margin_null", 0.0)
    dt = m.get("seat_lg_margin_twin_replayed", 0.0) - m.get("lg_margin_twin", 0.0)
    sn = m.get("seat_lg_margin_null_replayed_std", 0.0)
    st = m.get("seat_lg_margin_twin_replayed_std", 0.0)
    return dn, dt, sn, st


def _fires(m: dict) -> bool:
    dn, dt, sn, st = _deficits(m)
    return dn > SPREAD_MULT * sn and dt > SPREAD_MULT * st


def _check(m: dict, c: dict):
    # ── instrument validity ────────────────────────────────────────────
    if m.get("artifact_ok", 0.0) != 1.0 or c.get("artifact_ok", 0.0) != 1.0:
        return _void(m, "seated artifacts missing or unreadable")
    if m.get("borrowed_ok", 0.0) != 1.0:
        return _void(m, "uncalibrated borrow")       # refuses, never refutes
    for k in ("venue_n_steps", "venue_w_clock", "venue_e0"):
        if m.get(f"{k}_std", 0.0) != 0.0:
            return _void(m, f"venue inconsistent across seeds: {k}")

    # ── replay gate: the pairing targets must reconstruct the pinned
    # recorded aggregates (see SEAT_RECORDED's provenance comment) ─────
    for ours, theirs in (
            ("seat_lg_margin_null_replayed", f"{SEAT}/lg_margin_null"),
            ("seat_lg_margin_null_replayed_std", f"{SEAT}/lg_margin_null_std"),
            ("seat_lg_margin_twin_replayed", f"{SEAT}/lg_margin_twin"),
            ("seat_lg_margin_twin_replayed_std", f"{SEAT}/lg_margin_twin_std"),
            ("seat_life_gain_replayed", f"{SEAT}/life_gain")):
        rec = SEAT_RECORDED[theirs]
        if abs(m.get(ours, float("nan")) - rec) > TOL_REPLAY:
            return _void(m, f"pairing targets do not reconstruct: {ours} "
                            f"{m.get(ours)} vs recorded {rec}")

    # ── comparator run validity ────────────────────────────────────────
    if m.get("physics_finite", 0.0) != 1.0:
        return _void(m, "non-finite physics")
    if m.get("params_match", 0.0) != 1.0:
        return _void(m, f"architecture not identical to the seated arm "
                        f"(params {m.get('params')} vs recorded "
                        f"{m.get('seat_params_recorded')})")
    if m.get("lives_ok", 0.0) != 1.0:
        return _void(m, "comparator under 12 lives — the venue did not "
                        "produce a scorable run, not an arm verdict")

    # ── the control must sit exactly on the seat and must NOT fire ────
    for a, b in (("lg_margin_null", "seat_lg_margin_null_replayed"),
                 ("lg_margin_twin", "seat_lg_margin_twin_replayed")):
        if abs(c.get(a, float("nan")) - c.get(b, float("nan"))) > TOL_IDENTITY:
            return _void(m, "control: scoring path did not reproduce the "
                            f"seat from its own data ({a})")
    if _fires(c):
        return _void(m, "control: claim fired on the seat's own data")

    # ── the claim, exactly as pre-registered ───────────────────────────
    dn, dt, sn, st = _deficits(m)
    m["deficit_null"] = dn
    m["deficit_twin"] = dt
    m["deadweight_reading"] = float(dn <= SPREAD_MULT * sn
                                    and dt <= SPREAD_MULT * st)
    if _fires(m):
        return True
    return False


def run(ledger: Ledger | None = None):
    """3 seeds over 3 single-threaded nice-19 workers, memoised into run_spec
    (LC.03's own pattern — run_spec calls fn once per seed and 4.8 core-h of
    work must not happen twice)."""
    import multiprocessing as mp

    spec = BY_ID["LC.08"]
    seeds = list(range(spec.seeds))
    ctx = mp.get_context("spawn")
    with ctx.Pool(3, initializer=_worker_init) as pool:
        exp = dict(zip(seeds, pool.map(_experiment, seeds)))
    ctl = {s: _control(s) for s in seeds}          # artifact reads, cheap
    return run_spec(spec, lambda s: exp[s], _check,
                    control_fn=lambda s: ctl[s], ledger=ledger or Ledger())


def _worker_init():
    import os
    torch.set_num_threads(1)
    if os.nice(0) < 19:
        os.nice(19 - os.nice(0))


# ── smoke: mechanics only, records NOTHING ──────────────────────────────────
def _smoke():
    """Asserts the four things the pairing stands on, at a toy envelope:
    init identity with the seated arm, the zeroed gradient (latent_pred does
    not move while the actor does), the scoring identity of the control, and
    the replay of the recorded row from the artifacts. Prints JSON."""
    torch.set_num_threads(2)
    t0 = time.time()
    out: dict = {}

    # 1. init identity under the rig's own seeding.
    torch.manual_seed(9_000)
    a = _REAL_BUILD_ARM(SEAT)
    torch.manual_seed(9_000)
    b = _patched_build_arm(ARM)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys(), "state dict keys differ"
    assert all(torch.equal(sa[k], sb[k]) for k in sa), "init weights differ"
    out["init_identical"] = 1.0
    out["params"] = float(sum(p.numel() for p in b.parameters()))

    # 2. the zeroed gradient: one lc_update step; latent_pred must not move,
    #    the actor must.
    from ..cores import lc_update, lc_targets, make_optimizer, LC_BATCH
    gen = torch.Generator().manual_seed(7)
    batch = {k: torch.randn(LC_BATCH, d, generator=gen)
             for k, d in b.dims.items()}
    before_lp = {k: v.clone() for k, v in b.latent_pred.state_dict().items()}
    before_actor = {k: v.clone() for k, v in b.actor.state_dict().items()}
    opt = make_optimizer(b)
    loss = lc_update(b, batch, lc_targets(gen), opt)
    assert math.isfinite(loss), "non-finite loss"
    after_lp = b.latent_pred.state_dict()
    after_actor = b.actor.state_dict()
    assert all(torch.equal(before_lp[k], after_lp[k]) for k in before_lp), \
        "latent_pred moved — the binding term is NOT off"
    assert any(not torch.equal(before_actor[k], after_actor[k])
               for k in before_actor), "actor did not move — nothing trained"
    out["binding_term_off"] = 1.0

    # 3. + 4. control identity and the replay gate, on the real artifacts.
    for seed in (0, 1, 2):
        c = _control(seed)
        assert c.get("artifact_ok") == 1.0, c.get("artifact_refusal")
        assert abs(c["lg_margin_null"]
                   - c["seat_lg_margin_null_replayed"]) <= TOL_IDENTITY
    per = [_pairing(s, _load_artifact(s)[0]) for s in (0, 1, 2)]
    for ours, theirs in (("seat_lg_margin_null_replayed",
                          f"{SEAT}/lg_margin_null"),
                         ("seat_lg_margin_twin_replayed",
                          f"{SEAT}/lg_margin_twin")):
        vals = [p[ours] for p in per]
        mean = sum(vals) / 3
        assert abs(mean - SEAT_RECORDED[theirs]) <= TOL_REPLAY, \
            f"replay mismatch on {theirs}"
    out["replay_ok"] = 1.0

    # 5. a toy comparator run end to end through the patched rig.
    m = _experiment(0, n_steps=400, w_clock=None, write_artifact=False)
    assert m.get("borrowed_ok") == 1.0, m
    assert m.get("physics_finite") == 1.0, m
    assert m.get("params_match") == 1.0, m
    out["toy_run_ok"] = 1.0
    out["elapsed_s"] = round(time.time() - t0, 1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    _smoke()
