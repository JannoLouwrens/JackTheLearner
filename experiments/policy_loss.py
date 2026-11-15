"""Policy-loss challengers to `cores.lc_update`'s committed policy term.

WHY A SEPARATE MODULE. `cores.lc_update`'s policy term
`||mu - a||^2 * adv` is unbounded below for adv < 0 with mu unsquashed; the
11-12 LF.04 collapse diagnostic traced the actor's runaway (|mu| 1.8 -> 19.3,
every dim saturated) to it, and the policy-loss race on the s90 pilot
(experiments/lf04_collapse_diag.py, 12 lives/arm) found AWR ends the collapse
(saturation 0.97 -> 0.00) where tanh-squashing alone does not (0.96).
SEATED 11-15 BY TALLY: the J1 policy-loss race (policy_loss_race.py, seeds
150-152) seated tanh-awr in `cores.py` (POLICY_SEAT). "awr" is now bit-exact to
`cores.lc_update`, and "regress" is the CHALLENGER (on the seated tanh mean).

Editing `cores.py` would stale three standing PASSes (LC.02, LC.08, LF.03 —
0.78 CPU-h by `run stale-cost`, 11-12) for a change their runs never take, so
the challenger lives here and reaches `cores` only through its constants and
the binding/critic/actor methods every core already exposes.

`lc_update_with(core, batch, targets, opt, policy_loss=...)` is
`cores.lc_update` with ONLY the policy term chosen by name:
  "regress"  the pre-seat term ||mu - a||^2 * adv  (now a challenger)
  "awr"      the SEATED term (bit-exact to lc_update; `selftest()` proves it
             on a cloned core): advantage-weighted regression (Peng et al. 2019,
             arXiv:1910.00177): weight = min(exp(adv / AWR_BETA), AWR_WMAX)
             >= 0, so every taken action is pulled TOWARD, never pushed away.
"""
from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn.functional as F

from . import cores

POLICY_LOSSES = ("regress", "awr")
AWR_BETA = cores.AWR_BETA     # one source: the seat's constants
AWR_WMAX = cores.AWR_WMAX
SEATED = "awr"                # the name lc_update now computes


def policy_weight(advantage: torch.Tensor, policy_loss: str) -> torch.Tensor:
    if policy_loss == "regress":
        return advantage
    if policy_loss == "awr":
        return torch.clamp(torch.exp(advantage / AWR_BETA), max=AWR_WMAX)
    raise KeyError(f"unknown policy_loss {policy_loss!r}; one of {POLICY_LOSSES}")


def lc_update_with(core: cores.Core, batch: Dict[str, torch.Tensor],
                   targets: Dict[str, torch.Tensor], opt, *,
                   dropped: Iterable[str] = (),
                   policy_loss: str = SEATED) -> float:
    """`cores.lc_update`, line for line, with the policy weight chosen by name.
    F1 (train() only inside, eval state restored) is kept exactly."""
    was_training = core.training
    core.train()
    try:
        l_bind, _ = core.binding_loss(batch, dropped)
        z = core.shared_state(batch, dropped)
        l_v = F.mse_loss(core.critic(z), targets["value"])
        if getattr(core, "lp", False):
            l_v = l_v + F.mse_loss(core.critic_lp(z), targets["value_lp"])
        a = core.act(batch, z)
        w = policy_weight(targets["advantage"], policy_loss)
        l_pi = (((a - targets["action"]) ** 2).mean(-1) * w).mean()
        loss = l_bind + cores.VALUE_COEF * l_v + l_pi
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return float(loss.detach())
    finally:
        core.train(was_training)


def selftest(arm: str = "wm-latent", steps: int = 5) -> dict:
    """SEATED ("awr") must equal cores.lc_update bit-exactly (loss AND weights),
    and "regress" must differ — a control that must fail, so the check can fire."""
    import copy

    out = {}
    for pl in POLICY_LOSSES:
        torch.manual_seed(0)
        ref = cores.build_arm(arm)
        dut = copy.deepcopy(ref)
        o_ref, o_dut = cores.make_optimizer(ref), cores.make_optimizer(dut)
        gen = torch.Generator().manual_seed(1)
        same_loss = True
        for i in range(steps):
            batch = cores.make_batch(i, n=cores.LC_BATCH)
            tg = cores.lc_targets(gen)
            l1 = cores.lc_update(ref, batch, tg, o_ref)
            l2 = lc_update_with(dut, batch, tg, o_dut, policy_loss=pl)
            same_loss &= (l1 == l2)
        same_w = all(torch.equal(p, q) for p, q in
                     zip(ref.parameters(), dut.parameters()))
        out[pl] = {"same_loss": bool(same_loss), "same_weights": bool(same_w)}
    out["ok"] = (out[SEATED]["same_loss"] and out[SEATED]["same_weights"]
                 and not out["regress"]["same_weights"])
    return out


if __name__ == "__main__":
    import json
    r = selftest()
    print(json.dumps(r))
    raise SystemExit(0 if r["ok"] else 1)
