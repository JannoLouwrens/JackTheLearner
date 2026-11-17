"""The recurrent arms' row adapter (ORDERS J3 (4), 11-17): learning reaches the
part that makes each arm recurrent, and the never-trains control moves nothing.

  wm-dyn          the GRU's ACTION input columns (the seated arms get exactly
                  0.0 gradient there — cores_dyn docstring, ef335cf) and
                  weight_hh must move under training;
  wm-latent+ring  the RingMemory's parameters (angle, write, head) must move.
Control (must fail the "moved" check): the same cell with train=False leaves
every parameter byte-identical. Row 1 (Pendulum), seed 0, 400 steps.
"""
import torch

from experiments.cores import ACTION_DIM, LATENT
from experiments.inhouse_rows import run_cell_rec


def _moved(arm, train):
    torch.manual_seed(9_000)
    out = run_cell_rec(arm, 1, 0, train=train, steps=400, return_core=True)
    core = out["core"]
    from experiments.cores_dyn import DynWorldCore
    from experiments.cores_ring import RingWorldCore
    torch.manual_seed(9_000)               # the init run_cell_rec built from
    fresh = (DynWorldCore(with_placebo=False) if arm == "wm-dyn"
             else RingWorldCore(with_placebo=False))
    a, b = core.state_dict(), fresh.state_dict()
    if arm == "wm-dyn":
        keys = {"gru.weight_ih[action]": lambda d: d["gru.weight_ih"][:, LATENT:LATENT + ACTION_DIM],
                "gru.weight_hh": lambda d: d["gru.weight_hh"]}
    else:
        keys = {k: (lambda d, k=k: d[k]) for k in a if k.startswith("ring.")}
    return out, {k: float((f(a) - f(b)).abs().max()) for k, f in keys.items()}


def test_rec_arms_learn_their_recurrence_and_control_holds():
    for arm in ("wm-dyn", "wm-latent+ring"):
        out, d = _moved(arm, True)
        assert out["opt_steps"] > 0
        assert all(v > 0 for v in d.values()), (arm, d)
        print(arm, "trained", out["opt_steps"], "steps, max |dw|:", d)
        out, d = _moved(arm, False)
        assert out["opt_steps"] == 0
        assert all(v == 0 for v in d.values()), (arm, "control moved", d)
        print(arm, "control: 0 steps, max |dw|:", d)


if __name__ == "__main__":
    test_rec_arms_learn_their_recurrence_and_control_holds()
    print("PASS")
