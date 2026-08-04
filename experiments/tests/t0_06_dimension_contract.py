"""T0.06 — the sim's actuator count and the policy's action width must agree,
and a mismatch must RAISE rather than be silently written.

`VirtualWorld` assigns the policy's output into `mj_data.ctrl` with no width
check. NumPy will happily broadcast or truncate, so a policy emitting the wrong
number of actions does not crash — it drives the wrong joints with the wrong
values, forever, and every locomotion number after that is meaningless. This is
the cheapest possible test and it guards the most expensive class of result.

Decision D2 standardises on Gymnasium Humanoid-v5: nu = 17.

Control: attempting the assignment with a deliberately wrong width MUST raise.
If it does not, the runtime has no contract at all and the test says so.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]
EXPECTED_NU = 17          # Humanoid-v5, per decision D2


def _env():
    os.environ.setdefault("MUJOCO_GL", "disabled")   # physics only; no renderer on this box
    sys.path.insert(0, str(REPO))
    import gymnasium as gym
    return gym.make("Humanoid-v5")


def _experiment(seed: int) -> dict:
    import numpy as np
    e = _env()
    u = e.unwrapped
    nu = int(u.model.nu)
    act_dim = int(e.action_space.shape[0])

    # A correctly-sized action must be assignable and must actually land.
    from VirtualWorld import apply_action
    good = np.full(nu, 0.123, dtype=np.float64)
    apply_action(u.data, u.model, good)
    landed = float(np.abs(u.data.ctrl - good).max())

    e.reset(seed=seed)
    for _ in range(5):
        _, r, *_ = e.step(e.action_space.sample())

    return {
        "nu": nu,
        "action_space_dim": act_dim,
        "expected_nu": EXPECTED_NU,
        "ctrl_write_error": landed,
        "obs_dim": int(e.observation_space.shape[0]),
        "steps_ok": 5,
        "last_reward": round(float(r), 4),
    }


def _control(seed: int) -> dict:
    """Every wrong width must be REFUSED by the runtime's own writer.

    Raw NumPy assignment is not a sufficient guard: nu-1 and nu+1 raise, but
    width 1 is silently broadcast across all 17 actuators (measured). And the
    original runtime used min(len(action), nu), which truncated without
    complaint. So this exercises VirtualWorld.apply_action, the actual write
    path, not NumPy's incidental behaviour.
    """
    import numpy as np
    from VirtualWorld import apply_action
    e = _env()
    u = e.unwrapped
    nu = int(u.model.nu)
    wrongs = (nu - 1, nu + 1, 1, 2 * nu)
    raised = 0
    for wrong in wrongs:
        try:
            apply_action(u.data, u.model, np.zeros(wrong, dtype=np.float64))
        except ValueError:
            raised += 1
    # A NaN of the correct width must also be refused — it would propagate
    # silently into the physics and poison an entire run.
    try:
        apply_action(u.data, u.model, np.full(nu, np.nan))
    except ValueError:
        raised += 1
    return {"wrong_widths_tried": len(wrongs) + 1, "raised": raised}


def _check(m: dict, c: dict) -> bool:
    return (m["nu"] == EXPECTED_NU
            and m["action_space_dim"] == EXPECTED_NU
            and m["ctrl_write_error"] == 0.0
            and c["raised"] == c["wrong_widths_tried"])


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.06"], _experiment, _check, ledger=ledger, control_fn=_control)
