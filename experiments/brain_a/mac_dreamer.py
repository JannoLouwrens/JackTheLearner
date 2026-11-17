"""mac_dreamer.py — the Mac first-day rows 9 and 10 for brain A's runner, automated (ORDERS "Next up" 2).

Source of the two rows: PUBLISHED_PRIORS.md §10 "Reproduce first" — (9) "Dreamer steps/s on JAX, sheeprl,
sheeprl-MPS at 1M then 12M, same script on T4"; (10) "2,000 sheeprl steps CPU vs MPS (any divergence bars
MPS, PyTorch issue 204359)". The runner is J14's pin, `sheeprl==0.8.5`, algorithm `dreamer_v3_5`
(experiments/brain_a/README.md). Code only, never weights: every agent here starts from a fresh init.

The runner is NOT edited and not re-implemented. This file composes sheeprl's own configuration (the same
overrides as `speed.py`: cartpole-swingup vectors, 4 envs, the `_1M` or `_12M` preset), builds the agent with
sheeprl's own `build_agent`, optimizer and `Moments`, and calls sheeprl's own `train()` (one gradient step of
world model + actor + critic on one batch) and `update_target_critic()` exactly as its `main()` does. What it
leaves out is the environment and the replay buffer: the batches are FIXED synthetic data, so that the same
numbers go into the CPU run and the MPS run (the row-10 requirement "fixed seed and data").

Ours beside theirs (beside sheeprl's `main()`):
| setting            | sheeprl main()                       | here                                          |
|--------------------|--------------------------------------|-----------------------------------------------|
| observations       | the env's (cartpole: 5 numbers)      | a dummy 30-number vector (§10 long form)      |
| actions            | the env's (cartpole: 1, continuous)  | 6 continuous (the rover's arm count order)    |
| batches            | sampled from the replay buffer       | K = 16 fixed batches (numpy seed), cycled     |
| replay context     | latents written back to the buffer   | context latents held at zeros (a fresh buffer)|
| everything else    | batch 16 x 64 (+1 context), lr, eps, losses, target-critic schedule — unchanged     |
A gradient step here is the same arithmetic as one of main()'s; the replay-ratio bookkeeping and the env
stepping are not timed (row 9 is "grad steps/s", §10 long form).

Row 9 (`speed`): build on DEVICE, `--warm` steps (default 500, the long form's warm-up), then `--timed`
steps with a device synchronise at the end; prints gradient steps/s. Metrics aggregation off.

Row 10 (`check`) — the RULE, written before any Mac run (11-17):
  init: one fresh agent built on the CPU with torch seed S; every run below starts from its state_dicts.
  runs, `--steps` gradient steps each (default 2,000), the same K batches in the same order:
    REF   cpu, sampling seed S            REP   cpu, seed S again (must equal REF exactly, else VOID)
    FL1   cpu, seed S+1  } the noise floor: same weights, same data, other sampling noise (the agent
    FL2   cpu, seed S+2  }                  draws latents and imagined actions at every step)
    CTRL  cpu, seed S, the reward head's gradient zeroed by a hook (a planted silent-kernel bug)
    TEST  mps, seed S
  recorded every step: every loss/metric the runner aggregates, and the gradient norm of each of the
  world model, actor and critic (the long form's "loss curves and per-module gradient norms").
  distance of a run to REF, per series: the mean over 10 windows (each steps/10 long) of
  |window mean (run) - window mean (REF)|. floor = mean of FL1's and FL2's distances.
  ratio = distance / floor (a floor of 0 with distance 0 counts 0; with distance > 0 counts inf).
  TEST PASSES iff every series is finite and every ratio <= 2.0. Any non-finite value or any ratio > 2
  is DIVERGENCE and bars MPS from decisive runs (§10).
  VOID unless REP equals REF bit for bit AND CTRL shows a ratio > 2 on at least one series (the check
  can fail). On a machine without MPS, TEST is SKIPPED and the four CPU runs prove the apparatus.

    python -m experiments.brain_a.mac_dreamer speed --device cpu|mps [--size 1M|12M] [--warm N] [--timed N]
    python -m experiments.brain_a.mac_dreamer check [--steps 2000] [--size 1M] [--seed 0] [--out FILE]
Each prints one JSON line last. Needs a Python with `sheeprl==0.8.5` (on the box: /data/venvs/jack-a-probe).

Box smoke 11-17 18:20-18:45 (Ampere, nice 19, 1 thread, 1M preset = 641,641 weights; a smoke of the
apparatus, not a Mac reading):
  speed cpu: 0.53 grad steps/s (1,902 ms a step; warm 5, timed 10). §10's scaled guess for a fast M6 core
    on PyTorch CPU is 3-10 steps/s.
  check --steps 30 (5 CPU runs, ~67 s each): 18 series; REP == REF bit for bit; CTRL caught (worst
    Loss/reward_loss ratio 314 > 2) -> APPARATUS_OK; TEST SKIPPED (no MPS). First smoke was VOID: sheeprl's
    aggregator silently dropped every unregistered loss key (3 series left, CTRL ratio 1.27) -> Collector.
  Disclosed risk, before any Mac run: at 30 steps some floors are ~1e-6 (reward, policy loss, actor grad
  norm — the sampling seed barely moves them early). If a 2,000-step floor stays that small, float32
  rounding on MPS alone can exceed 2x it; a DIVERGENCE there is still a DIVERGENCE under this rule, and
  the series file shows which series and by how much.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OBS_DIM, ACT_DIM, K_BATCHES = 30, 6, 16
RATIO_MAX, WINDOWS = 2.0, 10


def compose_cfg(device: str, size: str = "1M"):
    os.environ.setdefault("MUJOCO_GL", "disable")
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf
    from sheeprl.utils.utils import dotdict

    from experiments.brain_a.speed import BASE
    ovr = [o for o in BASE if not o.startswith("algo=")] + [
        f"algo=dreamer_v3_5_{size}", "algo.mlp_keys.decoder=[state]", f"fabric.accelerator={device}"]
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        c = compose("config", overrides=ovr)
    return dotdict(OmegaConf.to_container(c, resolve=True, throw_on_missing=True))


class Collector:
    """Stands in for sheeprl's MetricAggregator inside train(): keeps every reported scalar of the last step."""
    disabled = False

    def __init__(self):
        self.vals = {}

    def update(self, name, value):
        self.vals[name] = float(value.detach().float().mean()) if hasattr(value, "detach") else float(value)

    def compute(self):
        return dict(self.vals)

    def reset(self):
        self.vals = {}


class Agent:
    """sheeprl's agent, optimizer and Moments, built as `dreamer_v3_5.main()` builds them."""

    def __init__(self, device: str, size: str, states=None, metrics: bool = False):
        import gymnasium as gym
        import hydra
        import numpy as np
        from lightning.fabric import Fabric
        from sheeprl.algos.dreamer_v3_5.agent import build_agent
        from sheeprl.algos.dreamer_v3_5.utils import Moments

        self.cfg = cfg = compose_cfg(device, size)
        self.fabric = Fabric(accelerator=device, devices=1, num_nodes=1, precision=cfg.fabric.precision)
        self.obs_space = gym.spaces.Dict({"state": gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)})
        self.actions_dim = (ACT_DIM,)
        st = states or {}
        self.wm, self.actor, self.critic, self.target, _ = build_agent(
            self.fabric, self.actions_dim, True, cfg, self.obs_space,
            st.get("world_model"), st.get("actor"), st.get("critic"), st.get("target_critic"))
        opt = hydra.utils.instantiate(
            cfg.algo.optimizer,
            params=[*self.wm.parameters(), *self.actor.parameters(), *self.critic.parameters()],
            _convert_="all")
        self.opt = self.fabric.setup_optimizers(opt)
        m = cfg.algo.actor.moments
        self.moments = Moments(m.decay, m.max, m.percentile.low, m.percentile.high)
        # train() reports through an aggregator's update(); sheeprl's own drops every key not pre-registered
        # (a box smoke recorded 0 loss series that way), so a collector that keeps every series stands in
        self.agg = Collector() if metrics else None
        self.step_n = 0

    def states(self):
        def sd(m):
            m = getattr(m, "module", m)
            return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        return {"world_model": sd(self.wm), "actor": sd(self.actor), "critic": sd(self.critic),
                "target_critic": sd(self.target)}

    def batches(self, seed: int):
        """K fixed batches, numpy-seeded, in the buffer's dtypes, moved to this agent's device."""
        import numpy as np
        import torch
        cfg = self.cfg
        T = cfg.algo.per_rank_sequence_length + cfg.algo.replay_context
        B = cfg.algo.per_rank_batch_size
        rss = cfg.algo.world_model.recurrent_model.recurrent_state_size
        sto = cfg.algo.world_model.stochastic_size
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(K_BATCHES):
            d = {
                "state": rng.standard_normal((T, B, OBS_DIM)).astype(np.float32),
                "actions": rng.uniform(-1, 1, (T, B, ACT_DIM)).astype(np.float32),
                "rewards": rng.standard_normal((T, B, 1)).astype(np.float32),
                "terminated": (rng.random((T, B, 1)) < 0.01).astype(np.float32),
                "truncated": np.zeros((T, B, 1), np.float32),
                "is_first": (rng.random((T, B, 1)) < 0.02).astype(np.float32),
            }
            t = {k: torch.as_tensor(v, device=self.fabric.device) for k, v in d.items()}
            if cfg.algo.replay_context > 0:
                t["deter"] = torch.zeros((T, B, rss), dtype=torch.float16, device=self.fabric.device)
                t["stoch"] = torch.zeros((T, B, sto), dtype=torch.uint8, device=self.fabric.device)
            out.append(t)
        return out

    def step(self, data):
        from sheeprl.algos.dreamer_v3_5.dreamer_v3_5 import train, update_target_critic
        cfg = self.cfg
        train(self.fabric, cfg, self.wm, self.actor, self.critic, self.target, self.opt, self.moments,
              data, self.agg, self.actions_dim)
        if self.step_n % cfg.algo.critic.per_rank_target_network_update_freq == 0:
            update_target_critic(self.critic, self.target, cfg.algo.critic.tau)
        self.step_n += 1

    def grad_norms(self):
        import torch
        out = {}
        for name, mod in (("world_model", self.wm), ("actor", self.actor), ("critic", self.critic)):
            g = [p.grad.detach().double().norm() for p in mod.parameters() if p.grad is not None]
            out[f"gradnorm/{name}"] = float(torch.stack(g).norm()) if g else 0.0
        return out


def _sync(device):
    import torch
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _device_ok(device):
    import torch
    if device == "mps":
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if device == "cuda":
        return torch.cuda.is_available()
    return True


def speed(device="cpu", size="1M", warm=500, timed=500, seed=0):
    import torch
    if not _device_ok(device):
        return {"status": "SKIPPED", "reason": f"{device} not available"}
    torch.manual_seed(seed)
    a = Agent(device, size)
    data = a.batches(seed + 1)
    t0 = time.perf_counter()
    for i in range(warm):
        a.step(data[i % K_BATCHES])
    _sync(device)
    t1 = time.perf_counter()
    for i in range(timed):
        a.step(data[(warm + i) % K_BATCHES])
    _sync(device)
    t2 = time.perf_counter()
    n = sum(p.numel() for m in (a.wm, a.actor, a.critic) for p in m.parameters())
    return {"status": "RAN", "device": device, "size": size, "params": int(n), "warm": warm, "timed": timed,
            "warm_s": round(t1 - t0, 2), "timed_s": round(t2 - t1, 2),
            "grad_steps_per_s": round(timed / (t2 - t1), 3), "ms_per_step": round(1e3 * (t2 - t1) / timed, 2),
            "torch": torch.__version__, "threads": torch.get_num_threads()}


def _run(device, size, states, data_seed, sample_seed, steps, planted=False):
    import torch
    a = Agent(device, size, states=states, metrics=True)
    if planted:  # the planted bug: the reward head learns nothing (a kernel that silently returns zeros)
        for p in a.wm.reward_model.parameters():
            p.register_hook(lambda g: torch.zeros_like(g))
    data = a.batches(data_seed)
    torch.manual_seed(sample_seed)
    series = {}
    for i in range(steps):
        a.step(data[i % K_BATCHES])
        row = {k: float(v) for k, v in a.agg.compute().items() if _scalar(v)}
        a.agg.reset()
        row.update(a.grad_norms())
        for k, v in row.items():
            series.setdefault(k, []).append(v)
    return series


def _scalar(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError, RuntimeError):
        return False


def _windows(x, w):
    return [sum(x[i * w:(i + 1) * w]) / w for i in range(len(x) // w)]


def distance(run, ref, steps):
    w = max(1, steps // WINDOWS)
    out = {}
    for k, r in ref.items():
        x = run.get(k)
        if x is None or len(x) != len(r):
            out[k] = math.inf
            continue
        if not all(math.isfinite(v) for v in x):
            out[k] = math.inf
            continue
        a, b = _windows(x, w), _windows(r, w)
        out[k] = sum(abs(p - q) for p, q in zip(a, b)) / len(a)
    return out


def ratios(d, floor):
    out = {}
    for k, v in d.items():
        f = floor.get(k, 0.0)
        out[k] = (0.0 if v == 0 else math.inf) if f == 0 else v / f
    return out


def judge(series: dict, steps: int) -> dict:
    """series: name -> {metric: [values]} for REF, REP, FL1, FL2, CTRL and optionally TEST. Pure; tested."""
    ref = series["REF"]
    rep_identical = all(series["REP"].get(k) == v for k, v in ref.items())
    d1, d2 = distance(series["FL1"], ref, steps), distance(series["FL2"], ref, steps)
    floor = {k: (d1[k] + d2[k]) / 2 for k in ref}
    rc = ratios(distance(series["CTRL"], ref, steps), floor)
    ctrl_fails = any(v > RATIO_MAX for v in rc.values())
    out = {"steps": steps, "series": len(ref), "rep_identical": rep_identical,
           "ctrl_worst": _worst(rc), "ctrl_fails": ctrl_fails,
           "floor": {k: round(v, 6) for k, v in floor.items()}}
    if not (rep_identical and ctrl_fails):
        out["verdict"] = "VOID"
        return out
    if "TEST" not in series:
        out["verdict"] = "APPARATUS_OK (no TEST run)"
        return out
    rt = ratios(distance(series["TEST"], ref, steps), floor)
    out["test_worst"] = _worst(rt)
    out["test_over"] = sorted(k for k, v in rt.items() if not v <= RATIO_MAX)
    out["verdict"] = "PASS" if not out["test_over"] else "DIVERGENCE"
    return out


def _worst(r):
    k = max(r, key=lambda k: r[k])
    return [k, r[k] if math.isfinite(r[k]) else "inf"]


def check(steps=2000, size="1M", seed=0, test_device="mps"):
    import torch
    torch.manual_seed(seed)
    states = Agent("cpu", size).states()
    plan = [("REF", "cpu", seed, False), ("REP", "cpu", seed, False), ("FL1", "cpu", seed + 1, False),
            ("FL2", "cpu", seed + 2, False), ("CTRL", "cpu", seed, True)]
    skipped = None
    if _device_ok(test_device):
        plan.append(("TEST", test_device, seed, False))
    else:
        skipped = f"{test_device} not available: TEST SKIPPED"
    series, secs = {}, {}
    for name, dev, s, planted in plan:
        t0 = time.perf_counter()
        series[name] = _run(dev, size, states, seed + 100, s, steps, planted)
        secs[name] = round(time.perf_counter() - t0, 1)
    out = judge(series, steps)
    out.update({"status": "RAN", "size": size, "seed": seed, "seconds": secs, "torch": torch.__version__})
    if skipped:
        out["skipped"] = skipped
    return out, series


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["speed", "check"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--size", default="1M", choices=["1M", "12M"])
    ap.add_argument("--warm", type=int, default=500)
    ap.add_argument("--timed", type=int, default=500)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-device", default="mps")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS"):   # mac_first_day caps workers in code, never by priority
        import torch
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    if a.what == "speed":
        rec = speed(a.device, a.size, a.warm, a.timed, a.seed)
    else:
        rec, series = check(a.steps, a.size, a.seed, a.test_device)
        if a.out:
            Path(a.out).write_text(json.dumps({"verdict": rec, "series": series}))
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
