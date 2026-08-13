"""survival.py — the shared survival loop for the LC screening/arbitration specs.

LC.03, LC.04 and LC.05 all run the same thing: an arm lives lives in a lethal
W0 and the spec reads `life_gain` off the run. This module is that loop, in ONE
place, for the same reason `cores.lc_update` is: *"two kernels re-implementing
one operation is the defect"* (LESSONS.md). LC.02 timed `lc_update`; this file
feeds it REAL targets instead of `lc_targets`' noise, and nothing else about
the update changes — so LC.02's train_ratio certificates still describe the
thing that runs here.

WHAT THIS MODULE IS NOT. It is not a claim and it is not the spec. Every gate
(the 3-sigma learning gates, S1-S5, the vetoes) lives in the LC.03 test file
and is pre-registered there before the recorded run. Constants below marked
PRE-REG CANDIDATE are proposals: the committed LC.03 test fixes them, and
changing one after LC.03's recorded run means a new spec version, not an edit.

STILL OWED BY THE SPEC LAYER, deliberately not invented here (each needs its
reference implementation ported, not paraphrased):
  * panel_dwell — PG.4's detector, its exact definition and 0.15 threshold.
    This file reports `panel_near_frac` (fraction of decisions with the rover
    inside PANEL_NEAR_M of the noise panel) as a DIAGNOSTIC ONLY; gating on it
    would be gating on a paraphrase.
  * chaos_occupancy / chaos_reward_ratio — CURIOSITY_BAKEOFF.md §2.10, reused
    unchanged, not re-derived.
  * ppo-lp's learning-progress intrinsic (CURIOSITY_BAKEOFF's `lp` arm
    unchanged) — the `reward_fn` hook and the `value_lp` target slot are here;
    the LP machinery itself is the spec's to port.
  * the shuffled-diary control's retrieval semantics (LC.03 control (d)) — an
    open design question recorded in LOOP_JOURNAL: none of the cores READS the
    diary yet, so what "permuted before retrieval" permutes must be settled
    before that control can fail for the right reason.

THE REWARD CHANNEL (F8 — fixed and identical for every arm). `homeo-dr`:
r = d(h_t) - d(h_{t+1}), the plain drive reduction LC.00 used, never clipped
or one-sided. Death is termination: the terminal transition charges
d_before - D_DEATH, where D_DEATH is the deviation of the LEAST-deviant state
any death can have (one need at its floor, the others at setpoint) — a lower
bound, computed from `drives.drive`, never hand-tuned. Arms that REINTERPRET
the channel (EFE's log-preference, darkroom's entropy objective, randrew's
projection) do so through `reward_fn`, which sees the same r_h everyone gets.

PERSISTENCE (W0-3). What crosses death is the learner: weights, optimiser,
replay. `wipe_at_death=True` reinitialises all of it at every death from the
same init seed — S3's `cross_life_transfer` is the paired difference between
a run and its wiped twin, exactly LC.00's "tables persist across lives" made
falsifiable for torch arms. The diary rows W0 writes cross death regardless
(that is XL.00's certificate, not ours).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from . import drives
from .cores import (ACTION_DIM, LC_BATCH, Core, build_arm, lc_update,
                    make_optimizer, n_params)
from .w0 import W0, SIM_S_PER_DECISION, random_action

# ── PRE-REG CANDIDATES — the LC.03 test file fixes these ────────────────────
GAMMA = 0.95            # <1 is load-bearing (NEEDS_AND_DEATH.md 0.2(f))
GAE_LAMBDA = 0.9        # F9's measured PPO configuration
SEG = 64                # decisions per GAE segment (targets recomputed here)
EXPLORE_STD = (0.5, 0.1)  # F9: initial action std ~0.5, decayed linearly
HOLD_K = 5              # random-repeat null: hold each action 5 decisions
REPLAY_CAP = 256        # LC.02's ring size, kept so the timed shape holds
PANEL_NEAR_M = 1.5      # diagnostic only — see the docstring
D_DEATH = drives.drive(0.0, 1.0, 0.0)   # terminal deviation, lower bound

POLICIES = ("core", "random", "random-repeat", "statue")


class _TargetRing:
    """Transitions WITH their GAE targets, ready for `lc_update`.

    Rows only enter once their segment closes (targets need the future), so a
    sampled batch is always fully labelled. Same ring shape LC.02 timed.
    """

    def __init__(self, obs: Dict[str, np.ndarray], cap: int = REPLAY_CAP):
        self.cap = cap
        self.keys = list(obs)
        self.obs = {k: torch.zeros(cap, obs[k].shape[0]) for k in self.keys}
        self.action = torch.zeros(cap, ACTION_DIM)
        self.value = torch.zeros(cap, 1)
        self.value_lp = torch.zeros(cap, 1)
        self.advantage = torch.zeros(cap)
        self.n = 0

    def push(self, obs, action, value_target, advantage, value_lp=0.0):
        i = self.n % self.cap
        for k in self.keys:
            self.obs[k][i] = obs[k]
        self.action[i] = action
        self.value[i, 0] = value_target
        self.value_lp[i, 0] = value_lp
        self.advantage[i] = advantage
        self.n += 1

    def sample(self, gen, size: int = LC_BATCH):
        hi = min(self.n, self.cap)
        idx = torch.randint(0, hi, (size,), generator=gen)
        batch = {k: self.obs[k][idx] for k in self.keys}
        targets = {"value": self.value[idx], "value_lp": self.value_lp[idx],
                   "action": self.action[idx], "advantage": self.advantage[idx]}
        return batch, targets


def _gae(rewards: List[float], values: List[float], boot: float,
         terminal: bool) -> tuple:
    """Advantages and value targets for one closed segment.

    `boot` is V(s_T) when the segment was cut by length, 0.0 when it was cut
    by death — plain termination, no hand-tuned penalty (LC.00's rule).
    Advantages are normalised per segment (F9's normalised targets).
    """
    n = len(rewards)
    adv = np.zeros(n)
    last = 0.0
    v_next = 0.0 if terminal else boot
    for t in reversed(range(n)):
        delta = rewards[t] + GAMMA * v_next - values[t]
        last = delta + GAMMA * GAE_LAMBDA * last
        adv[t] = last
        v_next = values[t]
    targets = adv + np.asarray(values)
    if n > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, targets


def _updates_due(decisions: int, train_ratio: float) -> int:
    """LC.02's cadence, verbatim — the certified semantics, not a paraphrase."""
    if train_ratio >= 1.0:
        return int(train_ratio)
    return 1 if (decisions % int(round(1.0 / train_ratio)) == 0) else 0


def run_survival(seed: int, *, j0: float, alpha: float,
                 n_decisions: int,
                 policy: str = "core",
                 arm: Optional[str] = None,
                 train: bool = True,
                 train_ratio: float = 0.25,
                 e0: float = 1.0,
                 reward_fn: Optional[Callable] = None,
                 wipe_at_death: bool = False,
                 explore_std: tuple = EXPLORE_STD,
                 diary=None) -> dict:
    """One arm-seed (or null/control) run: `n_decisions` of lethal W0.

    policy      "core" (needs `arm`), "random", "random-repeat", "statue".
    train       False = the untrained twin / frozen control: same core, same
                exploration schedule, optimiser NEVER stepped.
    reward_fn   (r_h, w, obs_t, core) -> float. None = homeo-dr unchanged.
                This is where randrew/darkroom/EFE reinterpret the channel.
    wipe_at_death  reinitialise core+optimiser+replay from the init seed at
                every death (S3's wiped twin). Weights are the cross-life
                store for these arms; wiping them is wiping what crosses.
    e0          starting/respawn energy. 1.0 is LC.03's regime; XL.00 used
                0.1 to force fast deaths and specs may do the same to pilot.

    Returns per-run facts only — no verdicts. Spec layer gates.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy {policy!r} not in {POLICIES}")
    if policy == "core" and arm is None:
        raise ValueError("policy='core' needs arm=")

    w = W0(seed=seed, j0=j0, alpha=alpha, lethal=True, diary=diary)
    if e0 != 1.0:
        w.drives.state = drives.DriveState(e=e0)
    rng = np.random.RandomState(seed * 6553 + 11)
    gen = torch.Generator().manual_seed(seed * 7919 + 3)
    init_seed = 9_000 + seed

    core = opt = ring = None

    def _fresh_learner():
        torch.manual_seed(init_seed)
        c = build_arm(arm)
        c.eval()                                        # F1
        return c, (make_optimizer(c) if train else None)

    if policy == "core":
        core, opt = _fresh_learner()
        ring = _TargetRing(w.observe())

    # panel position for the diagnostic (a mutated world may drop it)
    panel_xy = None
    if w.panel_gid >= 0:
        panel_xy = np.array(w.model.geom_pos[w.panel_gid][:2], dtype=float)

    seg_obs: list = []
    seg_act: list = []
    seg_rew: list = []
    seg_val: list = []
    held_action, held_left = None, 0
    optimiser_steps = 0
    reward_sum = 0.0
    needs_ok: List[int] = []          # per decision: d(h) < d_setpoint_half
    act_log: List[np.ndarray] = []
    panel_near = 0
    d_half = drives.drive(0.0, 1.0, 0.0) / 2.0   # halfway to a dead need
    t_cpu0, t_wall0 = time.process_time(), time.perf_counter()

    def _close_segment(boot: float, terminal: bool):
        nonlocal seg_obs, seg_act, seg_rew, seg_val
        if not seg_rew:
            return
        adv, tgt = _gae(seg_rew, seg_val, boot, terminal)
        for i in range(len(seg_rew)):
            ring.push(seg_obs[i], seg_act[i], float(tgt[i]), float(adv[i]))
        seg_obs, seg_act, seg_rew, seg_val = [], [], [], []

    for k in range(n_decisions):
        obs = w.observe()
        if policy == "statue":
            a = np.zeros(ACTION_DIM)
        elif policy == "random":
            a = random_action(rng)
        elif policy == "random-repeat":
            if held_left == 0:
                held_action, held_left = random_action(rng), HOLD_K
            a, held_left = held_action, held_left - 1
        else:
            obs_t = {kk: torch.from_numpy(v).unsqueeze(0)
                     for kk, v in obs.items()}
            with torch.no_grad():
                s = core.shared_state(obs_t, dropped=W0.DROPPED)
                mean = core.act(obs_t, s).squeeze(0).numpy()
                v = float(core.critic(s))
            frac = k / max(1, n_decisions - 1)
            std = explore_std[0] + (explore_std[1] - explore_std[0]) * frac
            a = np.clip(mean + std * rng.randn(ACTION_DIM), -1.0, 1.0)

        d_before = w.drives.state.d()
        w.decide(a)
        died = w.died_this_decision
        d_after = D_DEATH if died else w.drives.state.d()
        r = d_before - d_after
        if reward_fn is not None:
            r = float(reward_fn(r, w, obs, core))
        reward_sum += r
        needs_ok.append(int(d_after < d_half))
        act_log.append(a)
        if panel_xy is not None:
            xy = np.array(w.data.xpos[w.rover_bid][:2], dtype=float)
            panel_near += int(np.linalg.norm(xy - panel_xy) <= PANEL_NEAR_M)

        if policy == "core":
            seg_obs.append({kk: torch.from_numpy(v.copy())
                            for kk, v in obs.items()})
            seg_act.append(torch.from_numpy(a.astype(np.float32)))
            seg_rew.append(r)
            seg_val.append(v)
            if died or len(seg_rew) >= SEG:
                _close_segment(boot=v, terminal=died)
            if train and ring.n >= LC_BATCH:
                for _ in range(_updates_due(w.decisions, train_ratio)):
                    batch, targets = ring.sample(gen)
                    lc_update(core, batch, targets, opt, dropped=W0.DROPPED)
                    optimiser_steps += 1

        if died:
            if e0 != 1.0:
                w.drives.state = drives.DriveState(e=e0)
            if wipe_at_death and policy == "core":
                core, opt = _fresh_learner()
                ring = _TargetRing(obs)
                seg_obs, seg_act, seg_rew, seg_val = [], [], [], []

    spans = list(w.life_lengths)
    third = len(spans) // 3
    life_gain = (float(np.mean(spans[-third:]) - np.mean(spans[:third]))
                 if third >= 1 else 0.0)
    n = len(needs_ok)
    acts = np.asarray(act_log)
    out = {
        "life_gain": life_gain,
        "n_lives": float(len(spans)),
        "mean_life_s": float(np.mean(spans)) if spans else 0.0,
        "life_spans": [round(float(s), 3) for s in spans],
        "deaths_at_decision": [],     # filled below from sim clock
        "reward_sum": float(reward_sum),
        "needs_ok_first_third": (float(np.mean(needs_ok[:n // 3]))
                                 if n >= 3 else 0.0),
        "needs_ok_final_third": (float(np.mean(needs_ok[-(n // 3):]))
                                 if n >= 3 else 0.0),
        "action_std_final_third": (float(acts[-(len(acts) // 3):].std())
                                   if len(acts) >= 3 else 0.0),
        "panel_near_frac": float(panel_near / max(1, n)),
        "decisions": float(w.decisions),
        "sim_seconds": float(w.sim_seconds),
        "optimiser_steps": float(optimiser_steps),
        "process_time_s": float(time.process_time() - t_cpu0),
        "wall_s": float(time.perf_counter() - t_wall0),
        "physics_finite": float(bool(np.all(np.isfinite(w.data.qpos))
                                     and np.all(np.isfinite(w.data.qvel)))),
    }
    # F6: the curve, decimated to <=200 points — (decisions consumed, span).
    ends = np.cumsum([s / SIM_S_PER_DECISION for s in spans])
    step = max(1, len(spans) // 200)
    out["deaths_at_decision"] = [round(float(d), 1) for d in ends[::step]]
    if core is not None:
        out["params"] = float(n_params(core))
    return out


def _smoke() -> None:
    """`python -m experiments.survival` — does the loop hold together?

    NOT a spec and records nothing. Short lives (e0 low) so deaths happen in
    seconds; asserts on the product (lives ended, updates stepped, all finite)
    per "silence is not success".
    """
    from .protocol import borrow_metrics
    b = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    if b.values is None:
        raise SystemExit(f"PS.01 unavailable: {b.provenance}")
    j0, alpha = b.values["j0_ms"], b.values["alpha"]

    null = run_survival(0, j0=j0, alpha=alpha, n_decisions=550,
                        policy="random", e0=0.12)
    print("random:", {k: null[k] for k in
                      ("n_lives", "mean_life_s", "life_gain", "decisions")})
    assert null["n_lives"] >= 2, "no deaths at e0=0.12 — drives not draining?"
    assert null["physics_finite"] == 1.0

    armed = run_survival(0, j0=j0, alpha=alpha, n_decisions=700,
                         policy="core", arm="ppo-needs", train=True,
                         train_ratio=0.5, e0=0.12)
    print("ppo-needs:", {k: armed[k] for k in
                         ("n_lives", "optimiser_steps", "life_gain",
                          "reward_sum", "process_time_s")})
    assert armed["optimiser_steps"] > 0, "the learner never stepped"
    assert armed["n_lives"] >= 1
    assert np.isfinite(armed["reward_sum"])

    twin = run_survival(0, j0=j0, alpha=alpha, n_decisions=120,
                        policy="core", arm="ppo-needs", train=False,
                        train_ratio=0.5, e0=0.12)
    assert twin["optimiser_steps"] == 0.0, "the frozen twin stepped"
    print("twin:", {k: twin[k] for k in ("n_lives", "optimiser_steps")})
    print("SMOKE OK")


if __name__ == "__main__":
    _smoke()
