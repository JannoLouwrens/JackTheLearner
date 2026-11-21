"""J14 (b): parameter counts of A2-v1S as built vs the matched-encoder arm (`v1s_diff.MATCHED_ENC`).

Builds sheeprl's own agent (build_agent, CPU, 32-true) for cartpole_swingup's proprio obs (5 dims,
1 action) and counts parameters. No training, no speed claim (speed is read only by a timing cell).
Run: /data/venvs/jack-a-probe/bin/python -m experiments.brain_a.v1s_params
11-21 reading: encoder 266,752 -> 4,209,664; decoder 1,053,189 -> 5,782,533; world model
7,493,893 -> 16,428,293 (x2.19); actor/critic unchanged."""
import numpy as np, gymnasium
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from lightning import Fabric
from sheeprl.utils.utils import dotdict
from sheeprl.algos.dreamer_v3.agent import build_agent
from experiments.brain_a import rows, v1s_diff
def count(extra):
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=list(rows.BASE_V1S) + list(extra))
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    fab = Fabric(accelerator="cpu", devices=1, precision="32-true")
    obs = gymnasium.spaces.Dict({"state": gymnasium.spaces.Box(-np.inf, np.inf, (5,), np.float32)})
    wm, actor, critic, tcrit, _ = build_agent(fab, [1], True, cfg, obs)
    n = lambda m: sum(p.numel() for p in m.parameters())
    return {"encoder": n(wm.encoder), "decoder": n(wm.observation_model), "world_model": n(wm),
            "actor": n(actor), "critic": n(critic)}

if __name__ == "__main__":
    a = count(()); b = count(v1s_diff.MATCHED_ENC)
    for k in a: print(f"{k:12s} v1S {a[k]:>11,}  matched {b[k]:>11,}  x{b[k]/a[k]:.2f}")
