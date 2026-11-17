"""MLX port step 0.2b: the OFFICIAL DreamerV3 `debug` preset on a dummy 74-in / 8-out world.

Source: danijar/dreamerv3 @ e01491fad6434b2245a3b8ca201dd7faedcc458c (MIT), code only, no weights.
Report: docs/research/PUBLISHED_EVIDENCE_2026-11-15_PORT.md, build-plan row 0.2 —
"run the official `debug` preset for 200 steps on a dummy 74-in, 8-out world; both libraries
import; official losses logged; library versions recorded".

What this file does — nothing inside the agent is touched:
  * `Dummy74`: an `embodied.Env` with Jack's interface shape — obs `vector` float32 (74,),
    act `action` float32 (8,) in [-1, 1], 100-step episodes, deterministic obs (a fixed ramp
    plus the step count / 100 in slot 0), reward 0 except +1 on even steps. It is plumbing, not
    a task: 0.2 only asks that the official step runs and logs losses.
  * registers it by wrapping `dreamerv3.main.make_env` (suite `jack74`) and calls the
    official `main()` with `--configs defaults debug` (debug: jax cpu, debug=True,
    batch 8 x 10, envs 4, train_ratio 8, tiny widths), `--task jack74_v0`,
    `--run.steps 200`, logger jsonl only (the encoder/decoder take every non-reward obs key, here `vector`).
  * then reads the official `metrics.jsonl` and prints every `train/loss/*` series plus the
    library versions as one JSON line (also written to `<logdir>/official_debug.json`).

Settings that differ from the official run: logger outputs [jsonl] (scope dropped — no viewer
here), `jax.compute_dtype float32` (debug preset leaves bfloat16) (the report's golden-file rule: CPU, 32-bit,
compiler as-is), `run.steps 200`. Everything else is the published `debug` preset.

Not in this file (later steps): file-noise patch and the G1 two-run identity (step 0.4).

    /data/venvs/jack-dv3/bin/python -m experiments.brain_a.official_debug \
        --src /data/src/dreamerv3_e01491f --logdir /data/jack-logs/mlx/official_debug_r1
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

PIN = "e01491fad6434b2245a3b8ca201dd7faedcc458c"


def build_env_class():
    import elements
    import embodied
    import numpy as np

    class Dummy74(embodied.Env):
        def __init__(self, task, length=100):
            del task
            self.length = length
            self.count = 0
            self.done = False
            self._ramp = np.linspace(-1.0, 1.0, 74).astype(np.float32)

        @property
        def obs_space(self):
            return {
                "vector": elements.Space(np.float32, (74,)),
                "reward": elements.Space(np.float32),
                "is_first": elements.Space(bool),
                "is_last": elements.Space(bool),
                "is_terminal": elements.Space(bool),
            }

        @property
        def act_space(self):
            return {
                "reset": elements.Space(bool),
                "action": elements.Space(np.float32, (8,), -1.0, 1.0),
            }

        def step(self, action):
            if action.pop("reset") or self.done:
                self.count = 0
                self.done = False
                return self._obs(0.0, is_first=True)
            self.count += 1
            self.done = self.count >= self.length
            return self._obs(float(self.count % 2 == 0), is_last=self.done, is_terminal=self.done)

        def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
            v = self._ramp.copy()
            v[0] = self.count / self.length
            return dict(vector=v, reward=np.float32(reward), is_first=is_first,
                        is_last=is_last, is_terminal=is_terminal)

    return Dummy74


def versions():
    out = {"dreamerv3": PIN, "python": sys.version.split()[0]}
    import importlib.metadata as md
    for name in ("jax", "jaxlib", "numpy", "elements", "ninjax", "portal", "granular", "optax", "chex"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="checkout of danijar/dreamerv3 at the pin")
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    src = pathlib.Path(a.src).resolve()
    sys.path.insert(0, str(src))
    from dreamerv3 import main as dv3main  # noqa: E402  (official module)

    Dummy74 = build_env_class()
    official_make_env = dv3main.make_env

    def make_env(config, index, **overrides):
        if config.task.split("_", 1)[0] == "jack74":
            return dv3main.wrap_env(Dummy74(config.task), config)
        return official_make_env(config, index, **overrides)

    dv3main.make_env = make_env  # main() binds make_env at call time via its module global

    logdir = pathlib.Path(a.logdir)
    flags = [
        "--configs", "defaults", "debug",
        "--task", "jack74_v0",
        "--logdir", str(logdir),
        "--seed", str(a.seed),
        "--run.steps", str(a.steps),
        "--logger.outputs", "jsonl",
        "--run.log_every", "-1",
        "--jax.compute_dtype", "float32",
    ]
    dv3main.main(flags)

    losses: dict[str, list] = {}
    for line in (logdir / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        for k, v in row.items():
            if k.startswith("train/loss/"):
                losses.setdefault(k, []).append([row.get("step"), v])
    rec = {"versions": versions(), "steps": a.steps, "seed": a.seed, "losses": losses,
           "n_loss_series": len(losses),
           "finite": all(v == v and abs(v) != float("inf") for s in losses.values() for _, v in s)}
    (logdir / "official_debug.json").write_text(json.dumps(rec, indent=1))
    print("OFFICIAL_DEBUG", json.dumps({k: rec[k] for k in ("versions", "n_loss_series", "finite")}))
    for k, s in sorted(losses.items()):
        print(f"  {k}: first {s[0][1]:.5g} @ {s[0][0]}  last {s[-1][1]:.5g} @ {s[-1][0]}  (n={len(s)})")
    return 0 if losses and rec["finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
