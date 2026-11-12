"""LF.04 collapse diagnostic (ORDERS J1) — a MEASUREMENT, not a spec.

Writes no ledger row. The seed-90 pilot learner hit the statue's 600 s
ceiling at lives 3-7, ate once in life 8, then collapsed to 119-400 s lives
(lives 9-14) while his frozen twins drifted UP on exploration decay alone.
That forecasts an LF.04 FAIL; this script asks WHY, by replaying the very same
learner (same seed, same _Life4, deterministic) with per-life readings:

  cause            what killed him (W0's last_death_cause)
  act_abs          mean |policy mean action| (0 = statue; ~1 = saturated)
  act_sat          fraction of action dims with |mean| > 0.9
  path_m           metres travelled by the rover this life
  v_mean, v_sd     critic value at decision time
  rew_mean         mean per-decision reward (d_before - d_after)
  loss_mean, loss_max   lc_update loss over the life's optimiser steps
  dtheta           ||theta - theta0|| at life end
  nonfinite        cumulative non-finite losses

Hypotheses this can separate (decided by the readings, not argued):
  H1 learned stillness then a reward/value blow-up (loss_max, v_sd jump at 8-9)
  H2 policy saturation (act_sat -> 1: he runs flat out and starves faster)
  H3 a new death cause appears after the meal (cause changes at life 9)

VARIANTS (the 11-12 policy-loss race; H2 was confirmed and traced to
cores.py's l_pi = ||mu - a||^2 * adv, unbounded below for adv < 0 with mu
unsquashed). Each challenger changes ONLY the policy term; world, seed, core,
critic, binding loss, cadence and exploration are the pilot's:
  base       cores.lc_update as committed (bit-exact replay of the pilot)
  tanh       mu = tanh(actor(s)) in rollout AND loss: the loss is bounded,
             but a negative advantage still pushes mu away from a
  awr        advantage-weighted regression: weight = min(exp(adv/AWR_BETA),
             AWR_WMAX) >= 0, so every action is pulled TOWARD, never pushed
             away; mu unsquashed (it regresses onto clipped actions)
  tanh-awr   both
Not a spec: it picks which bounded loss LF.04 should be re-run with; the
claim is LF.04's, on seeds 100-102, against his frozen twin.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

VARIANTS = ("base", "tanh", "awr", "tanh-awr")
AWR_BETA = 1.0     # advantages are per-segment normalised, so beta=1 (Peng 2019)
AWR_WMAX = 20.0    # Peng 2019's weight clip


def _awr_update(core, batch, targets, opt, *, dropped=()):
    """cores.lc_update with ONLY the policy term replaced (AWR)."""
    import torch
    import torch.nn.functional as F

    from . import cores
    was_training = core.training
    core.train()
    try:
        l_bind, _ = core.binding_loss(batch, dropped)
        z = core.shared_state(batch, dropped)
        l_v = F.mse_loss(core.critic(z), targets["value"])
        if getattr(core, "lp", False):
            l_v = l_v + F.mse_loss(core.critic_lp(z), targets["value_lp"])
        a = core.act(batch, z)
        w = torch.clamp(torch.exp(targets["advantage"] / AWR_BETA),
                        max=AWR_WMAX)
        l_pi = (((a - targets["action"]) ** 2).mean(-1) * w).mean()
        loss = l_bind + cores.VALUE_COEF * l_v + l_pi
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return float(loss.detach())
    finally:
        core.train(was_training)


def main(seed: int = 90, n_lives: int = 12,
         out_path: str = "/data/tmp/lf04/diag_s90.jsonl",
         variant: str = "base") -> None:
    import torch

    from . import cores
    from .tests.lf_03_learner_lives import _calibration, _flat
    from .tests.lf_04_living_teaches import LIFE_CAP, _Life4

    torch.set_num_threads(1)
    j0, alpha, prov = _calibration()
    if j0 is None:
        raise SystemExit(f"PS.01 calibration refused: {prov}")
    L = _Life4(seed, j0, alpha, train=True)
    w = L.w
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}; one of {VARIANTS}")
    if variant.startswith("tanh"):
        raw_act = L.core.act
        L.core.act = lambda obs, s: torch.tanh(raw_act(obs, s))
    if variant.endswith("awr"):
        cores.lc_update = _awr_update

    acc = {"act": [], "val": [], "loss": [], "rew0": len(L.seg["rew"])}
    core_act = L.core.act

    def act_hook(obs, s):
        out = core_act(obs, s)
        if not torch.is_grad_enabled():
            acc["act"].append(out.detach().squeeze(0).numpy().copy())
        return out

    def critic_hook(_mod, _inp, out):
        if not torch.is_grad_enabled():
            acc["val"].append(float(out.reshape(-1)[0]))

    L.core.act = act_hook
    L.core.critic.register_forward_hook(critic_hook)
    real_update = cores.lc_update

    def update_hook(*a, **kw):
        loss = real_update(*a, **kw)
        acc["loss"].append(float(loss))
        return loss

    cores.lc_update = update_hook

    rews: list = []
    real_close = L._close_segment

    def close_hook(boot, terminal):
        rews.extend(L.seg["rew"])
        real_close(boot, terminal)

    L._close_segment = close_hook

    def pos():
        p = w.data.xpos[w.rover_bid]
        return np.array([float(p[0]), float(p[1])])

    t0 = time.perf_counter()
    with open(out_path, "w") as out:
        for life in range(n_lives):
            acc["act"].clear(); acc["val"].clear(); acc["loss"].clear()
            rews.clear()
            path, p_prev, k_life = 0.0, pos(), 0
            ate0 = int(sum(w.drives.ate_total.values()))
            while True:
                L.step()
                k_life += 1
                died = w.died_this_decision
                if not died:
                    p = pos()
                    path += float(np.linalg.norm(p - p_prev))
                    p_prev = p
                if died or k_life >= LIFE_CAP:
                    break
            acts = np.array(acc["act"]) if acc["act"] else np.zeros((1, 1))
            span = (float(w.life_lengths[-1]) if died
                    else float(w.sim_seconds - w._life_started_at))
            row = {
                "life": life, "sim_s": round(span, 1), "decisions": k_life,
                "censored": int(not died),
                "cause": w.last_death_cause if died else "",
                "ate": int(sum(w.drives.ate_total.values())) - ate0,
                "act_abs": round(float(np.abs(acts).mean()), 4),
                "act_sat": round(float((np.abs(acts) > 0.9).mean()), 4),
                "path_m": round(path, 2),
                "v_mean": round(float(np.mean(acc["val"])), 4) if acc["val"] else None,
                "v_sd": round(float(np.std(acc["val"])), 4) if acc["val"] else None,
                "rew_mean": round(float(np.mean(rews)), 6) if rews else None,
                "loss_mean": round(float(np.mean(acc["loss"])), 5) if acc["loss"] else None,
                "loss_max": round(float(np.max(acc["loss"])), 5) if acc["loss"] else None,
                "opt_steps": L.opt_steps,
                "dtheta": round(float((_flat(L.core) - L.theta0).norm()), 4),
                "nonfinite": L.nonfinite_loss,
                "wall_s": round(time.perf_counter() - t0, 1),
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            if not died:
                break
            if life == 0:
                # first life vs the pilot's: a replay that diverges is not
                # the pilot's learner and its readings must not be read as his
                print(f"life0 sim_s={row['sim_s']} (pilot 421.2)", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    variant = a[2] if len(a) > 2 else "base"
    main(int(a[0]) if a else 90, int(a[1]) if len(a) > 1 else 12,
         out_path=(a[3] if len(a) > 3 else
                   f"/data/tmp/lf04/diag_s90{'' if variant == 'base' else '_' + variant}.jsonl"),
         variant=variant)
