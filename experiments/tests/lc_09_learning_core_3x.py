"""LC.09 — the learning-core screen at 3x, with the collapse guard computed.

Runner for the spec registered 2026-11-13 (registry_expansion.py, ORDERS J3).
Every number below is copied from the registered spec; if this file and the
spec disagree, the spec wins and this file is the bug.

STATUS 2026-11-13 18:30: the COLLAPSE GUARD is built and self-tested here; the
claim run is NOT — `run()` refuses, naming why. Two things stand between this
file and a claim seed, and the second is a finding about Jack, not the rig:

  1. the `wm-latent+ring` arm does not exist in `cores.py`/`survival.py`
     (RingMemory.py is seated by WM.02 but never wired into a living core);
  2. THE WORLD-MODEL CORES HAVE NO DYNAMICS. `WorldModelCore.rssm()` feeds the
     GRU a ZERO action and a ZERO hidden state on every call
     (`cores.py` rssm: `h0 = zeros`, `a0 = zeros`), and wm-latent's binding
     loss predicts the CURRENT frame's own detached latent, not a future one.
     Measured (`probe` below, and the 18:20 check in the commit): after 20
     `lc_update` steps the gradient on the GRU's action columns AND on
     `weight_hh` is exactly 0.0 in dreamer-xs, wm-efe and wm-latent. Their
     "world model" is a feed-forward encoder; no weight that reads an action
     or a previous state ever moves. The seated Learning-core (wm-latent,
     LC.03 v2) therefore holds its seat as a model of nothing that changes.

     Consequence for THIS spec: the guard is RIGHT and the cores are wrong.
     Measured by `probe` (18:30, W0 seed 0, 300 random-policy decisions,
     5 init seeds 9000-9004): Delta_5 on untrained cores = -0.0061, -0.0079,
     -0.0081, -0.0080, -0.0073 — every one <= 0 (identical across the three
     arms: same construction order, same init). Training can never move the
     action columns, so the ActSWM clause will VOID every world-model arm,
     the seat included: a true positive, predictable from structure — the
     seated model ignores what he does, which is the failure ActSWM names.
     The clause stays unmoved. The repair is in Jack's code: a world-model
     core that carries its state across decisions and predicts the NEXT
     latent from (state_t, a_t), raced against these as a new arm (re-
     registration = strengthening: one more arm, every gate unmoved).

THE GUARD (LEARNING_CORE.md §5.4; spec `control` text). For every latent arm,
every GUARD_EVERY decisions, on the last GUARD_BUF observations:
  rank      RankMe effective rank (Garrido et al., arXiv:2210.02885):
            exp(entropy of the normalised singular values) of the centred
            shared-state matrix the actor reads.
  std       mean per-dimension std of that matrix (reported).
  Delta_5   cos(roll under the recorded actions, true future state) -
            cos(roll under all-zero torque, same target), averaged over the
            buffered 5-step windows that cross no death.
  The ROLL, declared here because the spec says "roll" and the cores define
  none: from the posterior state at t, deter_{k} = gru([z_t, a_k], deter_{k-1})
  for k = 1..5 with z_t (the last observed latent) held, and the rolled
  state = [deter_5 | prior(deter_5)]; the target is the core's own
  `rssm(obs_{t+5})["state"]`. Identical for all three world-model arms.
  VOID the arm (not clearer, not loser) iff over its final third
  mean Delta_5 <= 0 OR mean rank < RANK_FRAC x state dim OR mean action
  gain < ACT_MARGIN (or unmeasurable).

RE-REGISTERED 2026-11-13 20:15 (strengthening, no clause removed):
  Delta_5 > 0 cannot fail for an arm that learned nothing — the untrained
  seated core reads +.0147 from a filtered start (eda11bf), and Delta_5 FELL
  (.015 -> .007-.0085) while wm-dyn learned what his actions do (06a4502).
  Added: act_gain = gap_shuf(K=5) of the arm - gap_shuf(K=5) of its own
  UNTRAINED init twin, same buffered windows (`act_gap`, i.e.
  cores_dyn.action_share_k: own open-loop roll, z fed back from next_pred).
  Must reach ACT_MARGIN = .03 of z_{t+5}'s variance over the final third.
  Planted known-positives in the selftest: untrained dyn arm vs its twin
  reads exactly 0 and trips; the seated wm-latent (no next-latent head)
  trips as unmeasurable. Paired gains from 06a4502's probe (300 updates,
  held-out): +.149/+.101/+.083 — the clause can pass a learner.
"""
from __future__ import annotations

import json
import math
import sys
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

N_DECISIONS = 1_200_000          # 3x LC.03 v2's 400k
SEED_OFFSET = 140                # claim seeds 140, 141, 142
ARMS = ("ppo-needs", "ppo-lp", "dreamer-xs", "wm-efe", "wm-latent",
        "wm-latent+ring")
LATENT_ARMS = ("dreamer-xs", "wm-efe", "wm-latent", "wm-latent+ring")
SIGMA_GATE = 3.0                 # LC.03's two gates, unmoved
N_LIVES_MIN = 12
LEAD_SIGMA = 1.5                 # LC.04's seat bar
GUARD_EVERY = 1_000              # decisions between guard readings
GUARD_BUF = 256                  # observations per reading
ROLL_K = 5                       # ActSWM's Delta_5
RANK_FRAC = 0.10                 # VOID if mean rank < .10 x dim
ACT_K = 5                        # action-gain roll length (re-reg 11-13 20:15)
ACT_MARGIN = 0.03                # VOID unless final-third mean of
                                 # gap_shuf(trained) - gap_shuf(init twin)
                                 # >= .03 of z_{t+K}'s variance. Declared from
                                 # cores_dyn probes (06a4502): untrained
                                 # gap_shuf -.028/+.007/-.014 (sd ~.018),
                                 # trained @300 updates +.055/+.108/+.135.


# ── the guard's arithmetic: pure functions ────────────────────────────────
def rankme(z: np.ndarray, eps: float = 1e-7) -> float:
    """RankMe effective rank of a (n, d) matrix, centred."""
    z = np.asarray(z, dtype=np.float64)
    z = z - z.mean(0, keepdims=True)
    s = np.linalg.svd(z, compute_uv=False)
    if s.sum() <= eps:
        return 0.0
    p = s / s.sum() + eps
    return float(np.exp(-(p * np.log(p)).sum()))


def per_dim_std(z: np.ndarray) -> float:
    return float(np.asarray(z, dtype=np.float64).std(0).mean())


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1)


@torch.no_grad()
def roll(core, obs_t: Dict[str, torch.Tensor], acts: torch.Tensor,
         dropped=("language",)) -> torch.Tensor:
    """The declared roll (docstring). acts: (n, ROLL_K, ACTION_DIM)."""
    s = core.rssm(obs_t, dropped)
    z, deter = s["z"], s["deter"]
    for k in range(acts.shape[1]):
        deter = core.gru(torch.cat([z, acts[:, k]], -1), deter)
    prior = core._probs(core.prior(deter)).reshape(deter.shape[0], -1)
    return torch.cat([deter, prior], -1)


@torch.no_grad()
def delta5(core, obs_t, acts, obs_t5, dropped=("language",)) -> float:
    target = core.rssm(obs_t5, dropped)["state"]
    c_act = _cos(roll(core, obs_t, acts, dropped), target)
    c_zero = _cos(roll(core, obs_t, torch.zeros_like(acts), dropped), target)
    return float((c_act - c_zero).mean())


def guard_void(readings: List[dict], dim: int) -> Optional[str]:
    """The spec's VOID rule over the final third of an arm's readings.
    None = the guard passes; a string names the clause that fired."""
    if not readings:
        return "no guard readings"
    tail = readings[-max(1, len(readings) // 3):]
    d5 = float(np.mean([r["delta5"] for r in tail]))
    rk = float(np.mean([r["rank"] for r in tail]))
    if d5 <= 0.0:
        return f"collapse guard: final-third Delta_5 {d5:.4g} <= 0"
    if rk < RANK_FRAC * dim:
        return f"collapse guard: final-third rank {rk:.2f} < {RANK_FRAC} x {dim}"
    gains = [r.get("act_gain") for r in tail]
    if any(g is None or not math.isfinite(g) for g in gains):
        # Unknown is not zero (LESSONS_CORE 11): an arm with no action-
        # conditioned prediction of its own future cannot show it models
        # what he does, so it cannot clear.
        return ("collapse guard: action gain unmeasurable (arm has no "
                "next-latent prediction / filtered sequence)")
    ag = float(np.mean(gains))
    if ag < ACT_MARGIN:
        return (f"collapse guard: final-third action gain {ag:.4g} < "
                f"{ACT_MARGIN} over the untrained init twin")
    return None


@torch.no_grad()
def act_gap(core, hist, starts) -> Optional[float]:
    """gap_shuf of `cores_dyn.action_share_k` (K=ACT_K, the arm's OWN open-
    loop roll, z fed back from next_pred): the share of z_{t+K}'s variance
    the arm credits to the actions it was told vs another window's. None
    if the arm has no next-latent head or sequence filter."""
    if not (hasattr(core, "next_pred") and hasattr(core, "filter_seq")):
        return None
    from ..cores_dyn import action_share_k
    return float(action_share_k(core, hist, starts, K=ACT_K)["gap_shuf"])


# ── the guard in a living run, from outside survival.py ──────────────────
class GuardRecorder:
    """Rides `run_survival`'s `reward_fn` hook (called once per decision with
    (r_h, w, obs, core), after w.decide). Returns r_h unchanged or passes it
    to an inner reward_fn (wm-efe's). On first call it wraps the world
    instance's `decide` to record the executed action — runner-side, so no
    hashed file changes. Windows crossing a death are dropped."""

    def __init__(self, inner: Optional[Callable] = None,
                 every: int = GUARD_EVERY, buf: int = GUARD_BUF,
                 init_core=None):
        self.inner, self.every, self.buf = inner, every, buf
        # The arm's UNTRAINED twin for the action-gain clause. The runner
        # passes the twin built from the arm's init seed; if it does not,
        # the core is copied at the recorder's first call (before any
        # update the recorder has seen).
        self.init_core = init_core
        self.k = 0
        self.hist: List[tuple] = []          # (obs, action, life_id)
        self.readings: List[dict] = []
        self.life = 0
        self._w = None

    def _wrap(self, w):
        orig = w.decide

        def decide(a):
            self._last_a = np.asarray(a, dtype=np.float32).copy()
            return orig(a)
        w.decide = decide
        self._w = w

    def __call__(self, r_h, w, obs, core):
        if self._w is not w:
            self._wrap(w)
            if self.init_core is None:
                import copy
                self.init_core = copy.deepcopy(core).eval()
            r_out = self.inner(r_h, w, obs, core) if self.inner else r_h
            return r_out                     # the first action predates the wrap
        self.hist.append((obs, self._last_a, self.life))
        if w.died_this_decision:
            self.life += 1
        if len(self.hist) > self.buf + ROLL_K:
            self.hist.pop(0)
        self.k += 1
        if self.k % self.every == 0 and len(self.hist) > ROLL_K + 8:
            self.readings.append(self.read(core))
        return self.inner(r_h, w, obs, core) if self.inner else r_h

    @torch.no_grad()
    def read(self, core) -> dict:
        def batch(idx):
            return {kk: torch.from_numpy(np.stack([self.hist[i][0][kk]
                                                   for i in idx]))
                    for kk in self.hist[0][0]}
        n = len(self.hist)
        s = core.shared_state(batch(range(n)), dropped=("language",)).numpy()
        out = {"k": self.k, "rank": rankme(s), "std": per_dim_std(s),
               "dim": int(s.shape[1])}
        starts = [i for i in range(n - ROLL_K)
                  if self.hist[i][2] == self.hist[i + ROLL_K][2]]
        if starts and hasattr(core, "gru"):
            acts = torch.from_numpy(np.stack(
                [np.stack([self.hist[i + j][1] for j in range(ROLL_K)])
                 for i in starts]))
            out["delta5"] = delta5(core, batch(starts), acts,
                                   batch([i + ROLL_K for i in starts]))
        else:
            out["delta5"] = float("nan")
        from ..cores_dyn import SEQ
        a_starts = [i for i in range(SEQ, n - ACT_K)
                    if self.hist[i - SEQ][2] == self.hist[i + ACT_K][2]]
        g_now = act_gap(core, self.hist, a_starts) if a_starts else None
        g_init = (act_gap(self.init_core, self.hist, a_starts)
                  if a_starts and self.init_core is not None else None)
        out["act_gap"], out["act_gap_init"] = g_now, g_init
        out["act_gain"] = (None if g_now is None or g_init is None
                           else g_now - g_init)
        return out


# ── the registered run: refuses until the arm exists ─────────────────────
def run(ledger=None):
    raise RuntimeError(
        "LC.09 not runnable: (1) wm-latent+ring is not built in cores.py/"
        "survival.py; (2) the world-model arms never train an action- or "
        "state-reading weight (GRU action cols and weight_hh grad == 0), so "
        "the guard's Delta_5 clause must VOID them (untrained Delta_5 -0.006..-0.008). See the docstring "
        "and ORDERS J3. Nothing recorded.")


# ── selftest: planted latents through the real guard code ────────────────
def _selftest() -> dict:
    rng = np.random.RandomState(0)
    res = {}
    full = rng.randn(256, 64)
    rank1 = np.outer(rng.randn(256), rng.randn(64))
    res["rankme_full_high"] = rankme(full) > 0.5 * 64
    res["rankme_rank1_low"] = rankme(rank1) < 1.5
    res["rankme_constant_zero"] = rankme(np.ones((50, 8))) == 0.0
    good = [{"delta5": 0.02, "rank": 40.0, "act_gain": 0.10}] * 9
    res["healthy_passes"] = guard_void(good, 64) is None
    res["planted_rank1_trips"] = "rank" in (guard_void(
        [{"delta5": 0.02, "rank": rankme(rank1), "act_gain": 0.10}] * 9, 64) or "")
    res["delta_nonpos_trips"] = "Delta_5" in (guard_void(
        good[:6] + [{"delta5": -0.01, "rank": 40.0, "act_gain": 0.10}] * 3, 64) or "")
    res["early_collapse_forgiven"] = guard_void(
        [{"delta5": -1.0, "rank": 1.0, "act_gain": -1.0}] * 6 + good[:3], 64) is None
    # the action-gain clause (re-registered 11-13 20:15)
    res["act_gain_below_margin_trips"] = "action gain" in (guard_void(
        good[:6] + [{"delta5": 0.02, "rank": 40.0,
                     "act_gain": ACT_MARGIN - 1e-3}] * 3, 64) or "")
    res["act_gain_at_margin_passes"] = guard_void(
        [{"delta5": 0.02, "rank": 40.0, "act_gain": ACT_MARGIN}] * 9, 64) is None
    res["act_gain_missing_trips"] = "unmeasurable" in (guard_void(
        good[:6] + [{"delta5": 0.02, "rank": 40.0, "act_gain": None}] * 3, 64) or "")
    res["empty_is_void"] = guard_void([], 64) is not None
    # the recorder on a real world-model core, synthetic observations
    from ..cores import build_arm, ACTION_DIM
    torch.manual_seed(0)
    core = build_arm("wm-latent").eval()

    class _W:                                  # minimal world: decide + death flag
        died_this_decision = False

        def decide(self, a):
            pass
    w, g = _W(), GuardRecorder(every=50, buf=64)
    for t in range(201):
        obs = {k: rng.randn(d).astype(np.float32) for k, d in core.dims.items()}
        w.decide(rng.uniform(-1, 1, ACTION_DIM))
        assert g(0.5, w, obs, core) == 0.5      # reward passes through unchanged
    res["recorder_reads"] = len(g.readings) == 4
    r = g.readings[-1]
    res["recorder_dim_is_state"] = r["dim"] == core.state_dim
    res["recorder_delta5_finite"] = math.isfinite(r["delta5"])
    # known-positive through the real recorder: the seated wm-latent has no
    # next-latent head -> act_gain None -> the guard VOIDs it
    res["seated_act_gain_none"] = r["act_gain"] is None
    res["seated_guard_trips"] = "unmeasurable" in (guard_void(
        [{**rr, "rank": 40.0, "delta5": 0.02} for rr in g.readings], 64) or "")
    # an UNTRAINED dynamics arm vs its own init twin reads gain exactly 0
    # and must trip: the clause cannot pass an arm that learned nothing
    from ..cores_dyn import DynWorldCore
    torch.manual_seed(0)
    dyn = DynWorldCore().eval()
    w2, g2 = _W(), GuardRecorder(every=50, buf=64)
    for t in range(201):
        obs = {k: rng.randn(d).astype(np.float32) for k, d in dyn.dims.items()}
        w2.decide(rng.uniform(-1, 1, ACTION_DIM))
        g2(0.0, w2, obs, dyn)
    res["dyn_untrained_gain_zero"] = all(
        rr["act_gain"] == 0.0 for rr in g2.readings) and len(g2.readings) == 4
    res["dyn_untrained_trips"] = "action gain" in (guard_void(
        [{**rr, "rank": 40.0, "delta5": 0.02} for rr in g2.readings], 64) or "")
    res["ok"] = all(bool(v) for v in res.values())
    return res


# ── probe: the structural finding, measured (records nothing) ────────────
def _probe(n_init: int = 5, n_dec: int = 300) -> dict:
    """(a) gradient reaching the action/recurrent weights after real
    lc_update steps; (b) Delta_5 on untrained cores over real W0 random-policy
    observations, across init seeds — what the guard reads when nothing in
    the dynamics path has ever learned."""
    from ..cores import build_arm, make_optimizer, lc_update, lc_targets, LATENT
    from ..survival import random_action
    from ..w0 import W0
    from .lc_03_survival_screening import _borrow
    torch.set_num_threads(1)
    cal, prov = _borrow()
    assert cal is not None, prov
    out = {"grad": {}, "delta5_untrained": {}}
    for arm in ("dreamer-xs", "wm-efe", "wm-latent"):
        torch.manual_seed(0)
        c = build_arm(arm)
        opt = make_optimizer(c)
        gen = torch.Generator().manual_seed(1)
        b = {k: torch.randn(16, d, generator=gen) for k, d in c.dims.items()}
        for _ in range(20):
            lc_update(c, b, lc_targets(gen), opt, dropped=("language",))
        gi = c.gru.weight_ih.grad
        out["grad"][arm] = {
            "action_cols_max": float(gi[:, LATENT:].abs().max()),
            "obs_cols_max": float(gi[:, :LATENT].abs().max()),
            "weight_hh_max": float(c.gru.weight_hh.grad.abs().max())}
    w = W0(seed=0, j0=cal["j0"], alpha=cal["alpha"], lethal=True)
    rng = np.random.RandomState(0)
    hist = []
    for _ in range(n_dec):
        obs = w.observe()
        a = random_action(rng).astype(np.float32)
        w.decide(a)
        hist.append((obs, a, w.died_this_decision))
    for arm in ("dreamer-xs", "wm-efe", "wm-latent"):
        vals = []
        for s in range(n_init):
            torch.manual_seed(9_000 + s)
            c = build_arm(arm).eval()
            g = GuardRecorder()
            g.hist = [(o, a, 0) for o, a, _ in hist[-(GUARD_BUF + ROLL_K):]]
            vals.append(round(g.read(c)["delta5"], 6))
        out["delta5_untrained"][arm] = vals
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "selftest":
        print(json.dumps(_selftest(), indent=1))
    elif mode == "probe":
        print(json.dumps(_probe(), indent=1))
    else:
        raise SystemExit("usage: python -m experiments.tests.lc_09_learning_core_3x selftest|probe")
