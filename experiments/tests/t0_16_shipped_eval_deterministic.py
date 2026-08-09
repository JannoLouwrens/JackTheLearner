"""T0.16 — the evaluation a spec SHIPS must be deterministic, not the one the pipeline owns.

T0.14 closed the most expensive bug in this project: 36 `nn.Dropout` modules at
p=0.1 were live during rollout, during the PPO update, and during "deterministic"
evaluation, because `TrainingPipeline` never called `.eval()` or `.train()`. It
PASSes. It fixed `collect_rollout_vec` and `rl_update`.

It could not fix its consumers. T2.01 and T2.02 each carry their own
`eval_policy()` inside a `JOB` string — source that is shipped to a GPU VM and
never imported here — and both reached straight for
`tp.model(tp.project_obs(...))` with no mode call at all. So:

  * the **untrained control** evaluated in train mode, because a fresh
    `nn.Module` defaults to `training=True`;
  * the **trained arm** evaluated in train mode, because `rl_update` correctly
    leaves it there and nothing turned it off afterwards.

Measured at that exact call site on the real 57M net, 2026-08-09: two forwards of
one identical state differ by **103.6%** of the policy mean's own magnitude. The
~13 GPU-hours of T2.01/T2.02 re-runs whose whole purpose was to remove the
dropout confound would have reintroduced it, and the resulting numbers would
have entered the D1 architecture decision looking like evidence.

This is `docs/LESSONS.md`'s "a guard built by fixing one file leaves the file
that motivated it unfixed", at composition scale — T0.14 cannot see across a
process boundary into a kernel string. So T0.16 tests the shipped text itself.

Three properties, each able to fail on its own terms:

  1. STATIC — every locomotion kernel's `eval_policy` calls
     `tp.act_deterministic` and contains no bare `tp.model(` forward. Modules
     whose source cannot be extracted are COUNTED, not skipped: a scan that read
     nothing and a clean scan are the same number otherwise (T0.13's lesson).
  2. RUNTIME MODE — replay the real call order (untrained eval, rollout, PPO
     update, trained eval), driving the **extracted shipped source** against a
     stub env, and observe `model.training` from inside the forward via a hook.
     It must be False at both evaluation points, and the surrounding mode must be
     restored on exit so T0.14's two invariants still hold.
  3. RUNTIME DETERMINISM — with the running normaliser frozen, the shipped path
     must return bit-identical actions for one identical observation.

CONTROL: the PRE-FIX evaluation body, copied verbatim from T2.01 v4, run through
the identical harness. It MUST observe train mode at both points and MUST drift.
Without it this spec would pass on the broken code it exists to catch.

Freezing the normaliser is isolation, not leniency — it is applied to the
control too. It did surface a second, independent issue, reported here as
`normaliser_mutated_by_eval` and deliberately NOT gated: `normalize_obs` updates
the running mean/var on every call, so evaluation observations enter the
statistics the next training update uses. That is pre-existing behaviour shared
by every recorded locomotion run; changing it would move T2.01's numbers for a
reason unrelated to this bug, so it is measured and escalated rather than fixed
in passing.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

MAX_SHIPPED_DRIFT = 0.0     # bit-identity, not "small"
MIN_CONTROL_DRIFT = 1e-3    # the pre-fix path must be visibly non-deterministic

# The kernels whose evaluation path produces a number the ladder records.
LOCOMOTION_KERNELS = ["t2_01_locomotion_vs_random", "t2_02_mlp_showdown"]

# Verbatim from T2.01 v4 (commit 10e3aef), the body that ran on every recorded
# locomotion result. Kept here as the known-positive fixture: if this stops
# failing, the harness has stopped measuring.
PREFIX_EVAL_BODY = """
def eval_policy(tp, episodes, random_actions=False):
    env = tp.make_env()
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            if random_actions:
                act = env.action_space.sample()
            else:
                with torch.no_grad():
                    on = tp.normalize_obs(obs)
                    ot = torch.tensor(on, dtype=torch.float32,
                                      device=tp.device).unsqueeze(0)
                    out = tp.model(tp.project_obs(ot))
                    act = tp.policy_mean(out)[0].cpu().numpy()
            act = np.clip(act, env.action_space.low, env.action_space.high)
            obs, r, term, trunc, _ = env.step(act)
            total += float(r)
            done = term or trunc
        returns.append(total)
    env.close()
    return returns
"""

_EVAL_SRC = re.compile(
    r"^def eval_policy\(.*?(?=^\S|\Z)", re.MULTILINE | re.DOTALL)


def _calls_attr(src: str, attr: str) -> bool:
    """Does this source CALL `<something>.<attr>(...)`? Parsed, not grepped.

    The first version of this check was `"tp.model(" in src`, and it read the
    explanatory comment above the fix — which names the old broken call — as
    the broken call itself. A substring match cannot tell code from prose, and
    the failure mode is the dangerous direction: the guard reports the bug it
    just fixed, forever, so its verdict stops meaning anything.
    """
    import ast
    tree = ast.parse(src)
    return any(isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute)
               and n.func.attr == attr
               for n in ast.walk(tree))


def _shipped_eval_source(mod_name: str) -> str | None:
    """Pull `eval_policy` out of a spec's JOB string — the text that ships.

    Reading the module's own `JOB` rather than a copy is the whole point: a
    tidied restatement would pass while the shipped kernel stayed broken.
    """
    import importlib
    mod = importlib.import_module(f"experiments.tests.{mod_name}")
    job = getattr(mod, "JOB", None)
    if not isinstance(job, str):
        return None
    m = _EVAL_SRC.search(job)
    return m.group(0) if m else None


class _StubEnv:
    """Returns ONE fixed observation forever, ends after `steps` steps.

    A constant observation is what makes drift attributable: with the normaliser
    frozen, every action in the episode is the model's answer to an identical
    input, so any spread between them is the model being non-deterministic.
    """

    def __init__(self, obs_dim: int, act_dim: int, steps: int = 4):
        import numpy as np
        from types import SimpleNamespace
        self._np = np
        self._obs = np.zeros(obs_dim, dtype=np.float64)
        self._n = steps
        self._i = 0
        self.actions: list = []
        self.action_space = SimpleNamespace(
            low=np.full(act_dim, -0.4), high=np.full(act_dim, 0.4),
            sample=lambda: np.zeros(act_dim))

    def reset(self):
        self._i = 0
        return self._obs, {}

    def step(self, act):
        self.actions.append(self._np.array(act, dtype=self._np.float64, copy=True))
        self._i += 1
        return self._obs, 0.0, self._i >= self._n, False, {}

    def close(self):
        pass


def _freeze_normaliser(tp):
    """Make normalize_obs pure for the duration of a probe.

    Applied identically to the experiment and the control. Also returns the
    observed mutation of the LIVE normaliser, which is a separate finding.
    """
    import numpy as np

    def pure(obs_raw):
        obs = np.asarray(obs_raw, dtype=np.float64)
        std = (tp.obs_var + 1e-8).sqrt()
        import torch
        t = torch.tensor(obs, dtype=torch.float32, device=tp.device)
        return (t - tp.obs_mean).div(std).clamp(-10, 10).cpu().numpy().astype(np.float32)

    tp.normalize_obs = pure


def _probe(tp, eval_src: str, label: str) -> dict:
    """Run one shipped/pre-fix eval body against the stub env and watch it.

    The mode is read by a forward hook on the model rather than by inspecting
    `tp.model.training` around the call, because what matters is the mode AT THE
    FORWARD — a path that flips the mode back before returning would otherwise
    look clean.
    """
    import numpy as np
    import torch

    modes: list = []
    h = tp.model.register_forward_pre_hook(
        lambda mod, inp: modes.append(bool(mod.training)))

    env = _StubEnv(int(tp.obs_mean.shape[0]), int(tp.config.action_dim))
    ns = {"torch": torch, "np": np}
    exec(eval_src, ns)
    orig_make_env, tp.make_env = tp.make_env, lambda *a, **k: env
    outer_mode_before = tp.model.training
    try:
        ns["eval_policy"](tp, 1)
    finally:
        tp.make_env = orig_make_env
        h.remove()

    acts = np.stack(env.actions) if env.actions else np.zeros((1, 1))
    scale = max(float(np.abs(acts).mean()), 1e-9)
    return {
        f"{label}_forwards": len(modes),
        f"{label}_any_train_mode": bool(any(modes)),
        f"{label}_drift": float(np.abs(acts - acts[0]).max()) / scale,
        f"{label}_mode_restored": tp.model.training == outer_mode_before,
    }


def _harness(seed: int, eval_src: str):
    """Replay the exact call order T2.01/T2.02 perform, probing at both evals."""
    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch
    from TrainingPipeline import TrainingPipeline, PipelineConfig

    torch.manual_seed(seed)
    np.random.seed(seed)
    tp = TrainingPipeline(PipelineConfig())
    tp.make_optimizer(phase=3)
    _freeze_normaliser(tp)

    # POINT A — untrained control eval, on a model nobody has touched.
    out = _probe(tp, eval_src, "untrained")

    # The real loop in between. A minimal rollout + update is enough: what
    # matters is the mode rl_update leaves behind, not how much it learned.
    envs = tp.make_vec_envs(2)
    try:
        buf = tp.collect_rollout_vec(envs, n_steps=8)
        tp.rl_update(buf)
    finally:
        envs.close()
    _freeze_normaliser(tp)          # collect_rollout_vec rebinds nothing, but
                                    # rl_update may have; re-assert the isolation
    out["train_mode_after_update"] = bool(tp.model.training)

    # POINT B — trained eval, immediately after the update, as the kernels do.
    out.update(_probe(tp, eval_src, "trained"))
    return tp, out


def _experiment(seed: int) -> dict:
    src_ok, unreadable, uses_guard, raw_forward = {}, 0, 0, 0
    for name in LOCOMOTION_KERNELS:
        src = _shipped_eval_source(name)
        if src is None:
            unreadable += 1
            continue
        src_ok[name] = src
        uses_guard += int(_calls_attr(src, "act_deterministic"))
        raw_forward += int(_calls_attr(src, "model"))

    # The runtime probe runs the source T2.01 actually ships.
    shipped = src_ok.get("t2_01_locomotion_vs_random", PREFIX_EVAL_BODY)
    tp, m = _harness(seed, shipped)

    # Separate, ungated finding, measured against the CLASS method (the probe
    # froze only the instance binding): a single evaluation observation advances
    # the running statistics that the next PPO update normalises against. Real,
    # pre-existing, shared by every recorded locomotion run, and out of scope
    # here — reported so it is on the record rather than fixed in passing.
    import numpy as np
    from TrainingPipeline import TrainingPipeline as _TP
    before = int(tp.obs_count)
    _TP.normalize_obs(tp, np.zeros(int(tp.obs_mean.shape[0]), dtype=np.float32))
    m["normaliser_mutated_by_eval"] = int(tp.obs_count) != before

    m.update({
        "kernels_scanned": len(LOCOMOTION_KERNELS),
        "unreadable_kernels": unreadable,
        "kernels_using_act_deterministic": uses_guard,
        "kernels_with_raw_forward": raw_forward,
        "max_shipped_eval_drift": round(
            max(m["untrained_drift"], m["trained_drift"]), 9),
    })
    return m


def _control(seed: int) -> dict:
    """The pre-fix body, same harness. It MUST evaluate in train mode and drift."""
    _tp, m = _harness(seed, PREFIX_EVAL_BODY)
    return {
        "prefix_untrained_train_mode": m["untrained_any_train_mode"],
        "prefix_trained_train_mode": m["trained_any_train_mode"],
        "prefix_max_drift": round(
            max(m["untrained_drift"], m["trained_drift"]), 6),
    }


def _check(m: dict, c: dict) -> bool:
    return (
        # 1. static: the shipped kernels reference the guarded path
        m["unreadable_kernels"] == 0
        and m["kernels_using_act_deterministic"] == m["kernels_scanned"]
        and m["kernels_with_raw_forward"] == 0
        # 2. runtime mode, at BOTH evaluation points, read at the forward
        and m["untrained_forwards"] > 0 and m["trained_forwards"] > 0
        and not m["untrained_any_train_mode"]
        and not m["trained_any_train_mode"]
        # T0.14's invariants must survive the new path
        and m["train_mode_after_update"]
        and m["untrained_mode_restored"] and m["trained_mode_restored"]
        # 3. runtime determinism of the shipped path
        and m["max_shipped_eval_drift"] <= MAX_SHIPPED_DRIFT
        # CONTROL: the pre-fix body must fail both, or nothing is being measured
        and c["prefix_untrained_train_mode"] and c["prefix_trained_train_mode"]
        and c["prefix_max_drift"] > MIN_CONTROL_DRIFT
    )


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.16"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
