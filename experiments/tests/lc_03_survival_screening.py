"""LC.03 — Screening: which learning cores learn to survive at all.

THE QUESTION. LC.00 proved the metric decidable in a gridworld, LC.01 admitted
five arms on unison grounds, LC.02 fixed each arm's train_ratio on wall-clock
grounds. This spec is the first time the admitted cores LIVE: lethal W0, real
deaths, random respawns, the learner in the loop — and the claim is only that
at least two of them learn to survive AT ALL. Screening declares no winner
(LC.04's job); it can only exclude, and its VOID ("fewer than two learners")
blocks the decision instead of manufacturing one.

THE ENVELOPE (LEARNING_CORE.md §5.7, fixed HERE for the ledger). Each arm-seed
runs N_STEPS = 100,000 decisions (= 20,000 sim-s) at e0 = 1.0, and keeps
living until it has ALSO consumed W_CLOCK = 4,320 core-seconds (1.2 core-h) —
whichever comes later — so LC.04 (matched experience) and LC.05 (matched
compute) can score the SAME stored curves. Per-life resource coordinates and
the decimated span curve are written to
`experiments/artifacts/lc03_curves_seed{N}.json`; LC.04/LC.05 read that file
and run nothing. Untrained twins and the designed-to-fail controls run at
HALF_STEPS = 50,000 (§5.7's "controls at half budget"): their quantities are
stationary points, not curves. Wiped twins run the FULL envelope — they are
the S3 pairing, and a paired difference needs matched windows.

THE GATES, from the registry entry verbatim, encoded per seed where it says
per seed and on the house 3-sigma idiom (mean * sqrt(3) / std, the paired
margin t-statistic T2.08 standardised) where it says sigma:

  per arm, ALL of:
    lg_margin_null   life_gain - random null's life_gain, paired per seed;
                     t >= 3.0
    lg_margin_twin   life_gain - own UNTRAINED twin's, paired per seed; t >= 3.0
    n_lives >= 12    every seed
    needs_satisfied_rate rising (final third > first third, on the seed mean)
    cross_life_transfer > 0 (seed mean; = life_gain - wiped twin's life_gain)
    panel_dwell <= 0.15 every seed, else DISQUALIFIED — PG.4's own control
                     threshold, its exact dwell definition (fraction of the
                     LATE HALF of decisions within DWELL_RADIUS = 2.0 m of the
                     noise panel), ported not paraphrased
    chaos_occupancy >= 3.0 AND chaos_reward_ratio >= 2.0 in any seed => that
                     arm is VOID (CURIOSITY_BAKEOFF.md §2.10, ported below)
    wm-efe only: final-third action entropy (std of the PRE-NOISE policy
                     means) < 10% of dreamer-xs's => VOID for wm-efe
                     (arXiv:2303.01618's epistemic collapse)

  PASS iff >= 2 arms clear everything. Fewer => Status.VOID.

FOUR CONTROLS AND ONE WORLD-TRIPWIRE, each with a pre-registered side (a
wrong side is VOID — a control landing wrong means the instrument, not the
hypothesis, failed). Four falsifiers, not five: after the 2026-08-13
amendment, (e) can no longer fail by the mechanism the amendment itself
documents — basal drain 0.00167/s < active 0.0022/s means passivity wins
life LENGTH structurally, not by draw, so "not strongly negative" is the
world's arithmetic, not a test of the arms. (e) is kept as a TRIPWIRE: it
fires only if W0 ever starts punishing anti-curiosity, which means the rig
must be re-derived. A reader counting five falsifiers is counting one that
is not there. And to be explicit about what DOES exclude learned passivity
in the claim conjunction: not the dwell/chaos gates — a statue scores
perfectly on panel_dwell and every chaos_* signal — but `needs_rise > 0`,
the conjunct every pilot arm failed and the one the registered envelope is
betting 8.3x more experience will flip.
  (a) statue      AMENDED 2026-08-13 (see PILOT RESOLUTION): must RIDE THE
                  BASAL CEILING, |mean_life - e0/BASAL_B| <= 10% — the
                  passive path is clean; nothing but basal starvation may
                  kill a body that never acts. life_gain reported, ungated
                  (zero by construction — the saturated-quantity lesson).
  (b) randrew     fixed random stationary reward projection on the ppo-needs
                  core: must MISS the 3-sigma null gate
  (c) frozen      every untrained twin's |life_gain| within noise of zero
                  (t < 3.0 OR |mean| <= NOISE_FLOOR_S)
  (d) wiped-store every wiped twin's |life_gain| within noise of zero — the
                  2026-08-13 registry amendment: no admitted core retrieves
                  the diary, so the permuted-diary form could never fail for
                  the right reason (T0.13); the cross-life store the torch
                  arms use is the learner state, and wiping it is the
                  corruption the control needs
  (e) darkroom    dreamer-xs rewarded with MINUS its own posterior entropy:
                  AMENDED 2026-08-13 (see PILOT RESOLUTION): must NOT be
                  strongly negative (margin vs the half-budget null,
                  t > -3.0) — the measured inversion locked in as the
                  executable record that life_gain carries LEARNING, not
                  curiosity's sign.

PILOT RESOLUTION, 2026-08-13 (seed 90, 12k decisions, e0=0.3, 7667 s —
numbers recorded verbatim; both suspect sides were measured INVERTED and
both controls were amended in the registry and here IN THE SAME COMMIT,
BEFORE the registered run; T1.02 precedent, the old sides stay in git
history). The caution this block replaces predicted it: PS.01's basal drain
0.00167/s against active 0.0022/s means PASSIVITY maximises life LENGTH in
W0. Measured: statue mean life 180.0 s = e0/BASAL_B (0.3 x 600 s) to 0.02%
— the basal-starvation ceiling, an arithmetic fact, LONGEST of every run in
the pilot (arms 109.0-161.5 s, nulls 118.3/126.2 s, darkroom 183.5 s). The
darkroom LEARNED PASSIVITY and prospered on the length ruler: life_gain
+23.6, margin +49.7 s over its paired null. So (a) 'dies soonest' and (e)
'strongly negative' were unattainable as written — T2.08's passivity
inversion, as flagged. What the pilot ALSO showed, and the claim gates keep
unmoved: four of five arms cleared their null margins on life_gain (ppo-lp
+54.6, wm-efe +52.0, wm-latent +47.7, dreamer-xs +45.7; ppo-needs -1.8),
every chaos and dwell gate read clean, and every arm's needs_rise was
NEGATIVE at the compressed pilot envelope — the registered envelope
(N_STEPS=100k, e0=1.0) is 8.3x longer and the claim conjunction stands as
pre-registered. Gates MUST NOT MOVE after registration; this commit IS the
registration.

PORTS, not paraphrases:
  * panel_dwell — PG.4's `_dwell` (strict < DWELL_RADIUS 2.0 m) over the
    late half, threshold 0.15 = PG.4's CONTROL_DWELL_MAX. The harness's
    `panel_near_frac` (1.5 m, whole run) stays a diagnostic.
  * chaos detector — CURIOSITY_BAKEOFF.md §2.10 in full: one pooled forward
    model per seed over the late-half transitions of every arm plus the
    random null, fit twice (half/full data), 5-fold out-of-fold errors,
    theta = the null's 90th percentile of e_full, CHAOTIC iff e_full >= theta
    AND reducibility < 0.1, occupancy / 0.10, reward ratio of means clipped
    at 1e3, thrash_l1 / null's as the model-free second signal. One declared
    deviation: transitions are subsampled every TRACE_EVERY = 8 decisions
    (identically for every run) — §2.10 was written for 50k-decision runs and
    a full 100k+ trace does not fit the box; a uniform subsample is a rig
    constant, not a threshold.
  * ppo-lp's intrinsic — LC.00's `q_lp` (Oudeyer-style absolute learning
    progress, LP_BETA 0.5 * |delta EMA(prediction error)|, EMA lambda 0.1)
    lifted to continuous outcomes over a SAGG-RIAC-style auto-partitioned
    outcome space (median split on the max-variance dim every LP_SPLIT_N
    region visits, cap LP_MAX_REGIONS). Outcome = (torso xyz, z if airborne,
    climb-contact count, horizontal speed) — CURIOSITY_BAKEOFF's lp goal
    space minus nearest-object displacement, which W0 does not expose.
    Hindsight is implicit: regions score REACHED outcomes. The channel is
    r_lp with its own GAE pass into the critic_lp head; advantages add with
    unit weights (PURPOSE_AND_SCAFFOLDING.md §2.8 option 2; satiety gating
    off per F8).
  * wm-efe's actor objective — LC.00's `model_efe` form: r = r_h + KAPPA_EFE
    * (ensemble disagreement / its running mean), ln C = -d(h) so the
    pragmatic term IS drive reduction. KAPPA_EFE = 0.05, LC.00's kappa
    scale. DECLARED COST DEVIATION: the wm-efe and darkroom reward functions
    each spend one extra uncertainty forward per decision that LC.02 did not
    time; the W_CLOCK axis charges it honestly.

DATA-STARVED GUARD (owner, 2026-08-09, DECISIONS_NEEDED): an arm that fails
the gate WITH a positive final-half life-span slope is reported
`{arm}/data_starved = 1.0` — re-screen at a bigger envelope, do not
eliminate. The convergence and scale-transfer guards are LC.04's.

RUNNING IT. The registered run is ~90 core-hours and MUST be detached:

    nohup setsid nice -n 19 /data/venvs/jackthelearner/bin/python -m \
        experiments.tests.lc_03_survival_screening > /data/lc03_registered.log 2>&1 &

It parallelises the three seeds over 3 single-threaded workers (the memoised
run_spec pattern T2.01 established) and writes the ledger itself on
completion. `pilot` runs seed 90 at the compressed envelope (~2 h), `smoke`
is a minutes-long mechanics check that records nothing.

sb3-ppo, LC.04's ineligible reference arm, is NOT run here: it is not
LC.01-admitted and the registry scopes LC.03 to admissible arms. LC.04's
implementer must either push it through this same harness at this same
envelope (one extra run per seed) or record why not.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from ..cores import ACTION_DIM, CANDIDATE_ARMS
from ..drives import BASAL_B
from ..protocol import Ledger, Status, borrow_metrics, run_spec
from ..registry import BY_ID
from ..survival import run_survival
from ..w0 import W0
from .lc_02_throughput_floor import TRAIN_RATIOS, committed_ratio

IMPL_DEPS = ["experiments/survival.py", "experiments/cores.py",
             "experiments/w0.py", "experiments/drives.py", "playground.py"]

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO / "experiments" / "artifacts"

# ── THE ENVELOPE (LEARNING_CORE.md §5.7) — fixed for the registered run ─────
N_STEPS = 100_000               # decisions per arm-seed (20,000 sim-s)
W_CLOCK_CORE_S = 4_320.0        # 1.2 core-h; arms run to whichever is LATER
HALF_STEPS = N_STEPS // 2       # twins + designed-to-fail controls
E0 = 1.0                        # LC.03's regime (XL.00 compressed with 0.1)
TRACE_EVERY = 8                 # transition subsample for the chaos detector

# ── THE GATES ───────────────────────────────────────────────────────────────
SIGMA_GATE = 3.0                # the house learning gate (T2.02's invention)
N_LIVES_MIN = 12                # per seed, from the registry hypothesis
NOISE_FLOOR_S = 5.0             # 'within noise of zero' absolute clause: a
                                # twin whose seed spread collapses cannot be
                                # tripped by a 2-second effect (~0.3% of a life)
PANEL_DWELL_MAX = 0.15          # PG.4's CONTROL_DWELL_MAX — the ported gate
DWELL_RADIUS = 2.0              # PG.4's dwell zone, verbatim
CHAOS_OCC_VOID = 3.0            # §2.10 conjunction ...
CHAOS_RATIO_VOID = 2.0          # ... occupancy AND ratio
CHAOS_RED_EPS = 0.1             # reducibility threshold, §2.10
CHAOS_NULL_PCT = 90             # theta = null's 90th percentile of e_full
CHAOS_RATIO_CLIP = 1e3          # §2.10: an unbounded ratio cannot be gated
EFE_ENTROPY_FRAC = 0.10         # wm-efe collapse gate vs dreamer-xs

# ── THE PORTED MECHANISM CONSTANTS ─────────────────────────────────────────
LP_LAMBDA = 0.1                 # LC.00's q_lp EMA lambda
LP_BETA = 0.5                   # LC.00's q_lp bonus scale
LP_SPLIT_N = 400                # SAGG-RIAC: split a region every N visits
LP_MAX_REGIONS = 64
LP_LR = 1e-3                    # the outcome forward model's Adam lr
KAPPA_EFE = 0.05                # LC.00's model_efe kappa scale
CHAOS_HIDDEN = 256              # §2.10: 2x256 MLP, Adam 1e-3
CHAOS_EPOCHS = 3
CHAOS_BATCH = 256

PILOT_SEED = 90                 # disjoint from registered seeds 0/1/2
PILOT_STEPS = 12_000
PILOT_E0 = 0.3


# ── borrowed calibration ────────────────────────────────────────────────────
def _borrow():
    """PS.01's j0/alpha and LC.02's committed train_ratios, or a refusal."""
    b1 = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    if not b1.ok:
        return None, dict(b1.provenance, refusal=b1.refusal)
    keys = [f"{a}/clears@{r}" for a in CANDIDATE_ARMS for r in TRAIN_RATIOS]
    b2 = borrow_metrics("LC.02", keys)
    if not b2.ok:
        return None, dict(b2.provenance, refusal=b2.refusal)
    ratios = {a: committed_ratio(b2.values, a) for a in CANDIDATE_ARMS}
    if any(v is None for v in ratios.values()):
        return None, dict(b2.provenance,
                          refusal="LC.02 committed no ratio for some arm")
    return {"j0": b1.values["j0_ms"], "alpha": b1.values["alpha"],
            "ratios": ratios}, dict(b2.provenance)


# ── the ported reward machinery ────────────────────────────────────────────
class _LPRegions:
    """SAGG-RIAC-style recursive partition of the z-scored outcome space."""

    def __init__(self):
        dim = 6
        self.regions: List[dict] = [{
            "lo": np.full(dim, -np.inf), "hi": np.full(dim, np.inf),
            "n": 0, "ema": 1.0, "buf": []}]

    def find(self, oz: np.ndarray) -> dict:
        for reg in self.regions:
            if np.all(oz >= reg["lo"]) and np.all(oz < reg["hi"]):
                return reg
        return self.regions[0]           # numeric edge: fall back to the root

    def update(self, oz: np.ndarray, err: float) -> float:
        """EMA update in oz's region; returns |delta EMA| (the LP signal)."""
        reg = self.find(oz)
        prev = reg["ema"]
        reg["ema"] = prev + LP_LAMBDA * (err - prev)
        reg["n"] += 1
        reg["buf"].append(oz)
        if len(reg["buf"]) > 512:
            reg["buf"] = reg["buf"][-512:]
        if (reg["n"] % LP_SPLIT_N == 0 and len(self.regions) < LP_MAX_REGIONS
                and len(reg["buf"]) >= 32):
            pts = np.stack(reg["buf"])
            d = int(pts.var(axis=0).argmax())
            cut = float(np.median(pts[:, d]))
            if reg["lo"][d] < cut < reg["hi"][d]:
                a = dict(reg, lo=reg["lo"].copy(), hi=reg["hi"].copy(),
                         n=0, buf=[])
                b = dict(reg, lo=reg["lo"].copy(), hi=reg["hi"].copy(),
                         n=0, buf=[])
                a["hi"][d] = cut
                b["lo"][d] = cut
                # remove by IDENTITY: list.remove scans with ==, and dict
                # equality on numpy-array values raises the moment the split
                # region is not at index 0 (the smoke only ever split the root).
                self.regions = [r for r in self.regions if r is not reg]
                self.regions += [a, b]
        return abs(reg["ema"] - prev)


def _lp_intrinsic_factory(seed: int) -> Callable:
    """LC.00's q_lp form on a learned continuous outcome model. See docstring."""
    torch.manual_seed(seed * 811 + 5)
    net = torch.nn.Sequential(
        torch.nn.Linear(6 + ACTION_DIM, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 6))
    opt = torch.optim.Adam(net.parameters(), lr=LP_LR)
    regions = _LPRegions()
    state = {"prev_o": None, "prev_xy": None, "life": -1,
             "mu": np.zeros(6), "var": np.ones(6), "n": 0}

    def outcome(w) -> np.ndarray:
        pos = np.array(w.data.xpos[w.rover_bid], dtype=float)
        touch = w._touch()                     # [logF, flag] x 4 geoms
        foot, handL, handR = touch[3], touch[5], touch[7]
        if state["prev_xy"] is None:
            speed = 0.0
        else:
            speed = float(np.hypot(pos[0] - state["prev_xy"][0],
                                   pos[1] - state["prev_xy"][1]) / 0.2)
        state["prev_xy"] = pos[:2].copy()
        return np.array([pos[0], pos[1], pos[2],
                         pos[2] * (1.0 - float(foot)),
                         float(handL) + float(handR), speed])

    def zscore(o: np.ndarray) -> np.ndarray:
        state["n"] += 1
        d = o - state["mu"]
        state["mu"] = state["mu"] + d / state["n"]
        state["var"] = state["var"] + (d * (o - state["mu"]) - state["var"]) / state["n"]
        return (o - state["mu"]) / np.sqrt(np.maximum(state["var"], 1e-6))

    def fn(w, obs, a, core) -> float:
        o_now = outcome(w)
        oz_now = zscore(o_now)
        if w.life != state["life"] or state["prev_o"] is None:
            state["life"], state["prev_o"] = w.life, oz_now
            return 0.0
        x = torch.from_numpy(np.concatenate(
            [state["prev_o"], a]).astype(np.float32))
        y = torch.from_numpy(oz_now.astype(np.float32))
        pred = net(x)
        loss = ((pred - y) ** 2).mean()
        err = float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
        r_lp = LP_BETA * regions.update(state["prev_o"], err)
        state["prev_o"] = oz_now
        return r_lp

    return fn


def _randrew_factory(seed: int) -> Callable:
    """Control (b): a fixed random stationary projection of the observation."""
    wvec: Dict[str, np.ndarray] = {}

    def fn(r_h, w, obs, core) -> float:
        x = np.concatenate([obs[k] for k in obs])
        if "w" not in wvec:
            rng = np.random.RandomState(seed * 31 + 7)
            wvec["w"] = rng.randn(x.shape[0]) / math.sqrt(x.shape[0])
        return float(wvec["w"] @ x)

    return fn


def _uncertainty_reward_factory(sign: float, kappa: Optional[float]) -> Callable:
    """wm-efe (sign=+1, kappa) and darkroom (sign=-1, kappa=None => pure)."""
    state = {"rm": None}

    def fn(r_h, w, obs, core) -> float:
        obs_t = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        with torch.no_grad():
            u = float(core.uncertainty(obs_t, dropped=W0.DROPPED))
        state["rm"] = u if state["rm"] is None else (
            0.99 * state["rm"] + 0.01 * u)
        u_norm = u / max(state["rm"], 1e-9)
        if kappa is None:
            return sign * u_norm
        return float(r_h) + sign * kappa * u_norm

    return fn


# ── the ported detectors ───────────────────────────────────────────────────
def _panel_dwell(xy: np.ndarray, panel_xy: Optional[np.ndarray]) -> float:
    """PG.4's dwell share: late half, strict < DWELL_RADIUS."""
    if panel_xy is None or len(xy) < 4:
        return 0.0
    half = len(xy) // 2
    d = np.hypot(xy[half:, 0] - panel_xy[0], xy[half:, 1] - panel_xy[1])
    return float((d < DWELL_RADIUS).mean())


def _chaos_detect(pool: Dict[str, np.ndarray], null_name: str,
                  seed: int, obs_dim: int) -> Dict[str, dict]:
    """CURIOSITY_BAKEOFF.md §2.10, ported. See the module docstring.

    `pool[name]` rows are [s | a | mean_a | s' | r] with s of `obs_dim`.
    Returns per-name {occupancy, ratio, r_in, r_out, frac}.
    """
    late = {}
    for name, rows in pool.items():
        h = len(rows) // 2
        late[name] = rows[h:]
    names = list(late)
    X = np.concatenate([late[n][:, :obs_dim + ACTION_DIM] for n in names])
    Y = np.concatenate([late[n][:, 2 * ACTION_DIM + obs_dim:
                                2 * ACTION_DIM + 2 * obs_dim] for n in names])
    owner = np.concatenate([np.full(len(late[n]), i)
                            for i, n in enumerate(names)])
    n_rows = len(X)
    gen = np.random.RandomState(seed * 977 + 3)
    perm = gen.permutation(n_rows)
    folds = np.array_split(perm, 5)

    def fit_predict(train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
        torch.manual_seed(seed * 613 + len(train_idx))
        net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + ACTION_DIM, CHAOS_HIDDEN),
            torch.nn.ReLU(),
            torch.nn.Linear(CHAOS_HIDDEN, CHAOS_HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(CHAOS_HIDDEN, obs_dim))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        xt = torch.from_numpy(X[train_idx])
        yt = torch.from_numpy(Y[train_idx])
        for _ in range(CHAOS_EPOCHS):
            order = torch.randperm(len(xt))
            for i in range(0, len(xt), CHAOS_BATCH):
                idx = order[i:i + CHAOS_BATCH]
                loss = ((net(xt[idx]) - yt[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            pred = net(torch.from_numpy(X[test_idx]))
            return ((pred - torch.from_numpy(Y[test_idx])) ** 2
                    ).mean(dim=1).numpy()

    e_full = np.zeros(n_rows)
    e_half = np.zeros(n_rows)
    for f in range(5):
        test = folds[f]
        train = np.concatenate([folds[g] for g in range(5) if g != f])
        e_full[test] = fit_predict(train, test)
        e_half[test] = fit_predict(train[:len(train) // 2], test)

    null_i = names.index(null_name)
    theta = float(np.percentile(e_full[owner == null_i], CHAOS_NULL_PCT))
    reduc = (e_half - e_full) / np.maximum(e_half, 1e-9)
    chaotic = (e_full >= theta) & (reduc < CHAOS_RED_EPS)

    out = {}
    for i, name in enumerate(names):
        m = owner == i
        frac = float(chaotic[m].mean()) if m.any() else 0.0
        r = late[name][:, -1].astype(np.float64)
        c = chaotic[m]
        r_in = float(r[c].mean()) if c.any() else 0.0
        r_out = float(r[~c].mean()) if (~c).any() else 0.0
        ratio = r_in / r_out if abs(r_out) > 1e-12 else math.copysign(
            CHAOS_RATIO_CLIP, r_in) if abs(r_in) > 1e-12 else 0.0
        ratio = float(np.clip(ratio, -CHAOS_RATIO_CLIP, CHAOS_RATIO_CLIP))
        out[name] = {"occupancy": frac / 0.10, "ratio": ratio,
                     "r_in": r_in, "r_out": r_out, "frac": frac}
    return out


def _final_slope(spans: List[float]) -> float:
    """s-per-life linear slope over the final half of lives (data-starved)."""
    half = spans[len(spans) // 2:]
    if len(half) < 3:
        return 0.0
    x = np.arange(len(half), dtype=float)
    return float(np.polyfit(x, np.asarray(half, dtype=float), 1)[0])


def _entropy_of_means(rows: np.ndarray, obs_dim: int) -> float:
    """Final-third spread of the PRE-NOISE policy means (wm-efe's VOID gate)."""
    if len(rows) < 9:
        return 0.0
    means = rows[-(len(rows) // 3):,
                 obs_dim + ACTION_DIM:obs_dim + 2 * ACTION_DIM]
    return float(means.std(axis=0).mean())


# ── the experiment ─────────────────────────────────────────────────────────
def _experiment(seed: int, n_steps: int = N_STEPS, half_steps: int = HALF_STEPS,
                e0: float = E0, w_clock: Optional[float] = W_CLOCK_CORE_S,
                write_artifact: bool = True) -> dict:
    cal, prov = _borrow()
    m: dict = dict(prov)                # borrowed_* provenance, strings kept
    if cal is None:
        m["borrowed_ok"] = 0.0
        return m
    m["borrowed_ok"] = 1.0
    j0, alpha, ratios = cal["j0"], cal["alpha"], cal["ratios"]
    for a in CANDIDATE_ARMS:
        m[f"{a}/train_ratio"] = float(ratios[a])

    def go(**kw) -> dict:
        return run_survival(seed, j0=j0, alpha=alpha, e0=e0, **kw)

    runs: Dict[str, dict] = {}
    runs["null_random"] = go(n_decisions=n_steps, policy="random",
                             record_xy=True, record_transitions=TRACE_EVERY)
    runs["null_repeat"] = go(n_decisions=n_steps, policy="random-repeat")

    def arm_hooks(arm: str) -> dict:
        """Fresh reward machinery per run — closures carry state and a
        normaliser shared across runs would couple them."""
        h = {}
        if arm == "wm-efe":
            h["reward_fn"] = _uncertainty_reward_factory(+1.0, KAPPA_EFE)
        if arm == "ppo-lp":
            h["intrinsic_fn"] = _lp_intrinsic_factory(seed)
        return h

    for arm in CANDIDATE_ARMS:
        runs[arm] = go(policy="core", arm=arm, train=True,
                       train_ratio=ratios[arm], n_decisions=n_steps,
                       record_xy=True, record_transitions=TRACE_EVERY,
                       min_core_s=w_clock, **arm_hooks(arm))
        runs[f"{arm}/wiped"] = go(policy="core", arm=arm, train=True,
                                  train_ratio=ratios[arm],
                                  n_decisions=n_steps, wipe_at_death=True,
                                  **arm_hooks(arm))
        runs[f"{arm}/twin"] = go(policy="core", arm=arm, train=False,
                                 train_ratio=ratios[arm],
                                 n_decisions=half_steps, **arm_hooks(arm))

    # ── the ported detectors, once per seed ────────────────────────────
    w_probe = W0(seed=seed, j0=j0, alpha=alpha)
    obs_dim = int(sum(v.shape[0] for v in w_probe.observe().values()))
    panel_xy = (np.array(w_probe.model.geom_pos[w_probe.panel_gid][:2])
                if w_probe.panel_gid >= 0 else None)
    del w_probe
    pool = {a: runs[a]["transitions"] for a in CANDIDATE_ARMS}
    pool["null_random"] = runs["null_random"]["transitions"]
    chaos = _chaos_detect(pool, "null_random", seed, obs_dim)

    # ── metrics ────────────────────────────────────────────────────────
    lg_null = runs["null_random"]["life_gain"]
    m["null_life_gain"] = lg_null
    m["null_repeat_life_gain"] = runs["null_repeat"]["life_gain"]
    m["null_mean_life_s"] = runs["null_random"]["mean_life_s"]
    m["null_n_lives_ok"] = float(
        runs["null_random"]["n_lives"] >= N_LIVES_MIN
        and runs["null_repeat"]["n_lives"] >= N_LIVES_MIN)
    m["null_thrash"] = runs["null_random"]["thrash_l1"]
    m["null_chaos_occupancy"] = chaos["null_random"]["occupancy"]
    finite = [runs[k]["physics_finite"] for k in runs]
    m["physics_finite_min"] = float(min(finite))

    for arm in CANDIDATE_ARMS:
        r, w_, t = runs[arm], runs[f"{arm}/wiped"], runs[f"{arm}/twin"]
        m[f"{arm}/life_gain"] = r["life_gain"]
        m[f"{arm}/mean_life_s"] = r["mean_life_s"]
        m[f"{arm}/n_lives"] = r["n_lives"]
        m[f"{arm}/lives_ok"] = float(r["n_lives"] >= N_LIVES_MIN)
        m[f"{arm}/lg_margin_null"] = r["life_gain"] - lg_null
        m[f"{arm}/lg_margin_twin"] = r["life_gain"] - t["life_gain"]
        m[f"{arm}/twin_life_gain"] = t["life_gain"]
        m[f"{arm}/wiped_life_gain"] = w_["life_gain"]
        m[f"{arm}/clt"] = r["life_gain"] - w_["life_gain"]
        m[f"{arm}/needs_rise"] = (r["needs_ok_final_third"]
                                  - r["needs_ok_first_third"])
        m[f"{arm}/panel_dwell"] = _panel_dwell(r["xy"], panel_xy)
        m[f"{arm}/dwell_ok"] = float(
            m[f"{arm}/panel_dwell"] <= PANEL_DWELL_MAX)
        m[f"{arm}/chaos_occupancy"] = chaos[arm]["occupancy"]
        m[f"{arm}/chaos_reward_ratio"] = chaos[arm]["ratio"]
        m[f"{arm}/chaos_ok"] = float(not (
            chaos[arm]["occupancy"] >= CHAOS_OCC_VOID
            and chaos[arm]["ratio"] >= CHAOS_RATIO_VOID))
        m[f"{arm}/thrash_ratio"] = (r["thrash_l1"]
                                    / max(m["null_thrash"], 1e-9))
        m[f"{arm}/action_entropy_final"] = _entropy_of_means(
            r["transitions"], obs_dim)
        m[f"{arm}/final_slope"] = _final_slope(r["life_spans"])
        m[f"{arm}/optimiser_steps"] = r["optimiser_steps"]
        m[f"{arm}/decisions"] = r["decisions"]
        m[f"{arm}/core_s"] = r["process_time_s"]
        m[f"{arm}/params"] = r.get("params", 0.0)

    if write_artifact:
        ARTIFACTS.mkdir(exist_ok=True)
        art = {}
        for k, r in runs.items():
            art[k] = {kk: r[kk] for kk in
                      ("life_spans", "life_ends", "deaths_at_decision",
                       "optimiser_steps", "decisions", "sim_seconds",
                       "process_time_s", "reward_sum")}
            art[k]["params"] = r.get("params", 0.0)
            art[k]["grad_flops_est"] = r.get("grad_flops_est", 0.0)
        (ARTIFACTS / f"lc03_curves_seed{seed}.json").write_text(
            json.dumps({"seed": seed, "e0": e0, "n_steps": n_steps,
                        "w_clock_core_s": w_clock, "runs": art}, indent=1))
    return m


def _control(seed: int, n_steps: int = HALF_STEPS, e0: float = E0) -> dict:
    """(a) statue, (b) randrew, (e) darkroom — plus their own paired null."""
    cal, _ = _borrow()
    if cal is None:
        return {"borrowed_ok": 0.0}
    j0, alpha, ratios = cal["j0"], cal["alpha"], cal["ratios"]

    def go(**kw) -> dict:
        return run_survival(seed, j0=j0, alpha=alpha, e0=e0,
                            n_decisions=n_steps, **kw)

    null = go(policy="random")
    statue = go(policy="statue")
    randrew = go(policy="core", arm="ppo-needs", train=True,
                 train_ratio=ratios["ppo-needs"],
                 reward_fn=_randrew_factory(seed))
    darkroom = go(policy="core", arm="dreamer-xs", train=True,
                  train_ratio=ratios["dreamer-xs"],
                  reward_fn=_uncertainty_reward_factory(-1.0, None))
    return {
        "borrowed_ok": 1.0,
        "ctrl_null_life_gain": null["life_gain"],
        "ctrl_null_mean_life_s": null["mean_life_s"],
        "statue_mean_life_s": statue["mean_life_s"],
        "statue_life_gain": statue["life_gain"],
        "statue_n_lives": statue["n_lives"],
        "randrew_life_gain": randrew["life_gain"],
        "randrew_margin": randrew["life_gain"] - null["life_gain"],
        "randrew_opt_steps": randrew["optimiser_steps"],
        "darkroom_life_gain": darkroom["life_gain"],
        "darkroom_margin": darkroom["life_gain"] - null["life_gain"],
        "darkroom_mean_life_s": darkroom["mean_life_s"],
    }


def _tstat(m: dict, key: str) -> float:
    """The house paired 3-sigma idiom: mean * sqrt(n_seeds) / seed std."""
    return m.get(key, 0.0) * math.sqrt(3) / max(m.get(f"{key}_std", 0.0), 1e-9)


def _check(m: dict, c: dict):
    # ── instrument validity ─────────────────────────────────────────────
    if m.get("borrowed_ok", 0.0) != 1.0 or c.get("borrowed_ok", 0.0) != 1.0:
        return Status.VOID              # uncalibrated: refuses, never refutes
    if m.get("physics_finite_min", 0.0) != 1.0:
        return Status.VOID
    if m.get("null_n_lives_ok", 0.0) != 1.0:
        return Status.VOID              # the world cannot produce 12 lives at
        # this envelope — a world problem, not an arm result

    # ── controls, each on its pre-registered side ───────────────────────
    # (a) AMENDED 2026-08-13 from 'dies soonest' (seed-90 pilot, same commit,
    # T1.02 precedent — see PILOT RESOLUTION in the docstring): passivity
    # maximises life length in W0 (statue 180.0 s = e0/BASAL_B to 0.02%,
    # longest of every pilot run), so the statue now certifies the passive
    # path is CLEAN — nothing but basal starvation may kill a body that
    # never acts (a phantom-damage rig fault, PS.03's servo scar, is what
    # this catches; ctrl runs at module E0, so the ceiling is E0/BASAL_B).
    ceiling = E0 / BASAL_B
    if not abs(c.get("statue_mean_life_s", 0.0) - ceiling) <= 0.10 * ceiling:
        return Status.VOID
    # (b) randrew must miss the null gate.
    if _tstat(c, "randrew_margin") >= SIGMA_GATE:
        return Status.VOID
    # (e) AMENDED 2026-08-13 from 't <= -3' (same pilot, same commit): the
    # darkroom learned passivity and prospered on the length ruler (margin
    # +49.7 s, mean life 183.5 s vs null 126.2 s) — anti-curiosity WINS life
    # length in W0, so life_gain cannot carry curiosity's sign. The measured
    # inversion is locked in: if the world ever punishes anti-curiosity
    # strongly, this fires and the rig is re-derived, never silently re-read.
    if _tstat(c, "darkroom_margin") <= -SIGMA_GATE:
        return Status.VOID
    # (c) frozen and (d) wiped-store: every twin within noise of zero.
    for arm in CANDIDATE_ARMS:
        for kind in ("twin_life_gain", "wiped_life_gain"):
            k = f"{arm}/{kind}"
            if (abs(_tstat(m, k)) >= SIGMA_GATE
                    and abs(m.get(k, 0.0)) > NOISE_FLOOR_S):
                return Status.VOID      # lives lengthen without a persistent
                # learner: the metric measures the world (registry (c)/(d))

    # ── the claim ───────────────────────────────────────────────────────
    dreamer_ent = m.get("dreamer-xs/action_entropy_final", 0.0)
    cleared = 0
    for arm in CANDIDATE_ARMS:
        if m.get(f"{arm}/chaos_ok", 0.0) != 1.0:
            continue                    # §2.10: VOID for that arm
        if (arm == "wm-efe" and dreamer_ent > 0.0
                and m.get(f"{arm}/action_entropy_final", 0.0)
                < EFE_ENTROPY_FRAC * dreamer_ent):
            continue                    # epistemic collapse: VOID for wm-efe
        ok = (_tstat(m, f"{arm}/lg_margin_null") >= SIGMA_GATE
              and _tstat(m, f"{arm}/lg_margin_twin") >= SIGMA_GATE
              and m.get(f"{arm}/lives_ok", 0.0) == 1.0
              and m.get(f"{arm}/needs_rise", -1.0) > 0.0
              and m.get(f"{arm}/clt", -1.0) > 0.0
              and m.get(f"{arm}/dwell_ok", 0.0) == 1.0)
        cleared += int(ok)
    if cleared >= 2:
        return True
    return Status.VOID                  # "fewer than two learners" — blocks
    # the decision instead of manufacturing one. Data-starved arms (positive
    # final_slope) are re-screened at a bigger envelope, not eliminated.


def run(ledger: Ledger | None = None):
    """The registered run: 3 seeds over 3 single-threaded workers, memoised
    into run_spec (T2.01's pattern — run_spec calls fn once per seed and the
    work must not happen twice)."""
    import multiprocessing as mp

    spec = BY_ID["LC.03"]
    seeds = list(range(spec.seeds))
    ctx = mp.get_context("spawn")
    with ctx.Pool(3, initializer=_worker_init) as pool:
        exp = dict(zip(seeds, pool.map(_experiment, seeds)))
        ctl = dict(zip(seeds, pool.map(_control, seeds)))
    return run_spec(spec, lambda s: exp[s], _check,
                    control_fn=lambda s: ctl[s], ledger=ledger or Ledger())


def _worker_init():
    import os
    torch.set_num_threads(1)
    if os.nice(0) < 19:
        os.nice(19 - os.nice(0))


def _pilot():
    """Seed 90, compressed envelope. Prints JSON; records NOTHING."""
    t0 = time.time()
    torch.set_num_threads(1)
    m = _experiment(PILOT_SEED, n_steps=PILOT_STEPS,
                    half_steps=PILOT_STEPS // 2, e0=PILOT_E0, w_clock=None,
                    write_artifact=False)
    c = _control(PILOT_SEED, n_steps=PILOT_STEPS // 2, e0=PILOT_E0)
    print(json.dumps({"pilot_seed": PILOT_SEED, "steps": PILOT_STEPS,
                      "e0": PILOT_E0, "elapsed_s": round(time.time() - t0, 1),
                      "experiment": m, "control": c}, indent=1))


def _smoke():
    """Minutes-long mechanics check. Records nothing; asserts on the product."""
    torch.set_num_threads(2)
    cal, prov = _borrow()
    assert cal is not None, f"borrow refused: {prov}"
    j0, alpha = cal["j0"], cal["alpha"]

    # Force NON-ROOT region splits: list.remove scans with == (identity
    # short-circuit), so the crash only reachable when the splitting region
    # is not at index 0 — the 400-decision runs below never get there.
    _rng = np.random.RandomState(0)
    _regs = _LPRegions()
    for _ in range(LP_SPLIT_N * 40):
        _regs.update(_rng.randn(6) * 3.0, err=float(_rng.rand()))
    assert len(_regs.regions) > 2, "forced splits did not happen"
    for _ in range(200):
        _oz = _rng.randn(6) * 3.0
        assert sum(1 for rg in _regs.regions
                   if np.all(_oz >= rg["lo"]) and np.all(_oz < rg["hi"])) == 1
    print("lp regions ok:", len(_regs.regions), "regions, partition intact")

    lp = _lp_intrinsic_factory(0)
    r = run_survival(0, j0=j0, alpha=alpha, e0=0.12, n_decisions=400,
                     policy="core", arm="ppo-lp", train=True, train_ratio=0.5,
                     intrinsic_fn=lp, record_xy=True, record_transitions=4)
    assert r["optimiser_steps"] > 0 and np.isfinite(r["reward_sum"])
    assert len(r["xy"]) == 400 and r["transitions"].shape[0] > 10
    obs_dim = (r["transitions"].shape[1] - 2 * ACTION_DIM - 1) // 2
    print("lp arm ok:", {k: round(float(r[k]), 3) for k in
                         ("n_lives", "optimiser_steps", "reward_sum",
                          "thrash_l1")}, "obs_dim", obs_dim)

    null = run_survival(0, j0=j0, alpha=alpha, e0=0.12, n_decisions=400,
                        policy="random", record_xy=True, record_transitions=4)
    ch = _chaos_detect({"ppo-lp": r["transitions"],
                        "null_random": null["transitions"]},
                       "null_random", 0, obs_dim)
    assert 0.5 < ch["null_random"]["occupancy"] < 2.0, \
        f"null occupancy should be ~1.0 by construction: {ch}"
    print("chaos ok:", {k: {kk: round(vv, 3) for kk, vv in v.items()}
                        for k, v in ch.items()})

    w_probe = W0(seed=0, j0=j0, alpha=alpha)
    pxy = np.array(w_probe.model.geom_pos[w_probe.panel_gid][:2])
    at_panel = np.tile(pxy, (100, 1)).astype(np.float32)
    assert _panel_dwell(at_panel, pxy) == 1.0
    assert _panel_dwell(at_panel + 50.0, pxy) == 0.0
    print("panel_dwell ok; committed ratios:", cal["ratios"])

    efe = _uncertainty_reward_factory(+1.0, KAPPA_EFE)
    dark = _uncertainty_reward_factory(-1.0, None)
    e = run_survival(0, j0=j0, alpha=alpha, e0=0.12, n_decisions=60,
                     policy="core", arm="wm-efe", train=True, train_ratio=0.25,
                     reward_fn=efe)
    d = run_survival(0, j0=j0, alpha=alpha, e0=0.12, n_decisions=60,
                     policy="core", arm="dreamer-xs", train=True,
                     train_ratio=0.25, reward_fn=dark)
    assert np.isfinite(e["reward_sum"]) and np.isfinite(d["reward_sum"])
    assert d["reward_sum"] < 0, "darkroom's channel must be negative"
    print("efe/darkroom ok:", round(e["reward_sum"], 3),
          round(d["reward_sum"], 3))
    print("SMOKE OK")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "smoke":
        _smoke()
    elif mode == "pilot":
        _pilot()
    else:
        res = run()
        print(res.status, res.message)
