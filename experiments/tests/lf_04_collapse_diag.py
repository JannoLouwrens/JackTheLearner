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
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np


def main(seed: int = 90, n_lives: int = 12,
         out_path: str = "/data/tmp/lf04/diag_s90.jsonl") -> None:
    import torch

    from .. import cores
    from .lf_03_learner_lives import _calibration, _flat
    from .lf_04_living_teaches import LIFE_CAP, _Life4

    torch.set_num_threads(1)
    j0, alpha, prov = _calibration()
    if j0 is None:
        raise SystemExit(f"PS.01 calibration refused: {prov}")
    L = _Life4(seed, j0, alpha, train=True)
    w = L.w

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
    main(int(a[0]) if a else 90, int(a[1]) if len(a) > 1 else 12)
