"""ME.8 — working memory survives restarts: kill the process mid-thought.

GOAL.md: "He remembers the ladder" is episodic memory (ME.1-4); this is the
other half — what Jack is holding in mind RIGHT NOW must survive the host
dying under him, because on free-tier compute the host WILL die under him.
docs/research/MEMORY.md 3.2 picked GRU recurrence for exactly this reason:
the whole within-episode memory is one small tensor, checkpointable to disk
every step. WorkingMemory.py is the substrate.

The task makes memory necessary by construction: a cue (1 of 8) is shown at
step 0 and NEVER AGAIN; after ~30 steps of pure-noise distractors the agent
must name the cue. Anything it answers with after a restart can only have
come from the state file.

The teeth:

  1. A REAL kill, not a simulated one. The episode runs in a child process
     that checkpoints wm.state after every step; the parent SIGKILLs it at a
     drifting mid-episode step (T0.05 discipline — the atomic write is what
     makes this survivable). Every child must die by SIGKILL (rc -9); a
     child that finished politely proves nothing and fails the run.
  2. A FRESH process restores wm.state, plays out only the remaining
     distractor steps, and must name the cue: resume_acc >= 0.90.
  3. The null: the same fresh process with the state ZEROED (position kept,
     memory dropped) must collapse toward the 1/8 base rate — the spec's
     "post-restart behavior equals a zeroed-state agent" is the falsifier,
     so resume - zeroed >= 0.50 and zeroed <= 0.45.

CONTROL (must fail): finish episode i from episode j's restored state. The
agent must answer with j's cue (match_restored >= 0.80) and its accuracy on
i's true cue must collapse (<= 0.30). If it could still name i's cue, the
second half of the episode leaks the answer and the test never measured
memory at all.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import numpy as np

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]
PY = "/data/venvs/jackthelearner/bin/python"

C = 8                     # cue classes -> base rate 0.125
N_NOISE = 4               # distractor channels, N(0,1) every step after the cue
OBS_DIM = C + N_NOISE + 1  # + query flag on the final step
T = 30                    # cue at step 0, answer read at step T-1
HIDDEN = 32
BATCH = 64
MAX_ITERS = 1000
LR = 5e-3
N_EP = 16                 # evaluation episodes (one kill each)
KILL_MIN, KILL_MAX = 10, 20   # SIGKILL target drifts across this window

MIN_HOLDOUT = 0.95        # the trained model must actually solve the task
MIN_RESUME = 0.90
MAX_ZEROED = 0.45         # ~4 sigma above 0.125 at 16 episodes
MIN_GAP = 0.50            # resume_vs_zeroed, the registered metric
MAX_CROSS_TRUE = 0.30     # control: true-cue accuracy must collapse
MIN_MATCH_RESTORED = 0.80  # control: answers must follow the restored state

_CACHE: dict = {}         # seed -> artifacts, shared between _experiment and _control


def _build_wm():
    from WorkingMemory import WorkingMemory
    return WorkingMemory(OBS_DIM, C, HIDDEN)


def _episode_obs(seed: int, ep: int, cue: int) -> np.ndarray:
    """Deterministic episode: killed process and fresh process must agree on
    every distractor without communicating."""
    rng = np.random.default_rng(100003 * seed + ep + 7)
    obs = np.zeros((T, OBS_DIM), dtype=np.float32)
    obs[0, cue] = 1.0
    obs[1:, C:C + N_NOISE] = rng.standard_normal((T - 1, N_NOISE)).astype(np.float32)
    obs[T - 1, -1] = 1.0
    return obs


def _train(seed: int, tmp: Path) -> float:
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    sys.path.insert(0, str(REPO))
    wm = _build_wm()
    opt = torch.optim.Adam(wm.parameters(), lr=LR)

    def batch_obs(n):
        cues = torch.randint(0, C, (n,))
        obs = torch.zeros(T, n, OBS_DIM)
        obs[0, torch.arange(n), cues] = 1.0
        obs[1:, :, C:C + N_NOISE] = torch.randn(T - 1, n, N_NOISE)
        obs[T - 1, :, -1] = 1.0
        return obs, cues

    streak = 0
    for _ in range(MAX_ITERS):
        obs, cues = batch_obs(BATCH)
        wm.reset(BATCH)
        for t in range(T):
            logits = wm.step(obs[t])
        loss = F.cross_entropy(logits, cues)
        opt.zero_grad()
        loss.backward()
        opt.step()
        streak = streak + 1 if (logits.argmax(1) == cues).float().mean() == 1.0 else 0
        if streak >= 25:
            break

    with torch.no_grad():
        obs, cues = batch_obs(256)
        wm.reset(256)
        for t in range(T):
            logits = wm.step(obs[t])
        holdout = (logits.argmax(1) == cues).float().mean().item()
    torch.save(wm.state_dict(), tmp / "weights.pt")
    return holdout


# Runs one episode up to (never including) the query step, checkpointing
# wm.state after every step, until the parent SIGKILLs it. Reaching exit(3)
# means the kill missed — the parent counts that against killed_frac.
PHASE_A = textwrap.dedent("""
    import sys, time
    repo, weights, state_path, seed, ep, cue = (
        sys.argv[1], sys.argv[2], sys.argv[3],
        int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
    sys.path.insert(0, repo)
    import torch
    torch.set_num_threads(1)
    from experiments.tests.me_8_working_memory import _build_wm, _episode_obs, T
    wm = _build_wm()
    wm.load_state_dict(torch.load(weights, weights_only=True))
    obs = _episode_obs(seed, ep, cue)
    wm.reset()
    with torch.no_grad():
        for t in range(T - 1):
            wm.step(torch.from_numpy(obs[t]))
            wm.checkpoint(state_path)
            time.sleep(0.025)
    sys.exit(3)
""")

# A fresh process: restore (or zero, or cross-restore) and finish the episode.
PHASE_B = textwrap.dedent("""
    import json, sys
    repo, weights, states_dir, mode, out_json, seed = (
        sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
        int(sys.argv[6]))
    cues = json.loads(sys.argv[7])
    sys.path.insert(0, repo)
    import torch
    torch.set_num_threads(1)
    from experiments.tests.me_8_working_memory import _build_wm, _episode_obs, T
    wm = _build_wm()
    wm.load_state_dict(torch.load(weights, weights_only=True))
    preds = []
    for ep in range(len(cues)):
        src = (ep + 1) % len(cues) if mode == "cross" else ep
        start = wm.restore(f"{states_dir}/ep{src}.state")
        if mode == "zeroed":
            wm.reset()
            wm.step_idx = start   # same position in the episode, no memory
        obs = _episode_obs(seed, ep, cues[ep])
        with torch.no_grad():
            for t in range(start, T):
                logits = wm.step(torch.from_numpy(obs[t]))
        preds.append(int(logits.argmax()))
    json.dump(preds, open(out_json, "w"))
""")


def _kill_phase(seed: int, tmp: Path, cues: list) -> dict:
    """SIGKILL a live episode per evaluation episode; keep what hit the disk."""
    import torch
    script = tmp / "phase_a.py"
    script.write_text(PHASE_A)
    rng = np.random.default_rng(seed + 31)
    killed, ckpt_steps = 0, []
    for ep in range(N_EP):
        target = int(rng.integers(KILL_MIN, KILL_MAX + 1))
        state = tmp / f"ep{ep}.state"
        proc = subprocess.Popen(
            [PY, str(script), str(REPO), str(tmp / "weights.pt"), str(state),
             str(seed), str(ep), str(cues[ep])],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.time() + 120
            while time.time() < deadline:
                try:
                    if torch.load(state, weights_only=True)["step"] >= target:
                        break
                except (FileNotFoundError, EOFError, RuntimeError):
                    pass
                time.sleep(0.005)
            os.kill(proc.pid, signal.SIGKILL)
        finally:
            proc.wait()
        if proc.returncode == -signal.SIGKILL:
            killed += 1
        ckpt_steps.append(torch.load(state, weights_only=True)["step"])
    return {"killed_frac": killed / N_EP,
            "mean_ckpt_step": float(np.mean(ckpt_steps))}


def _finish(tmp: Path, mode: str, seed: int, cues: list) -> list:
    script = tmp / "phase_b.py"
    script.write_text(PHASE_B)
    out = tmp / f"preds_{mode}.json"
    subprocess.run(
        [PY, str(script), str(REPO), str(tmp / "weights.pt"), str(tmp), mode,
         str(out), str(seed), json.dumps(cues)],
        check=True, timeout=300,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return json.loads(out.read_text())


def _eval_cues(seed: int) -> list:
    """Consecutive cues distinct (incl. the wrap pair) so the cross-restore
    control is never accidentally correct."""
    rng = np.random.default_rng(seed + 13)
    cues = [int(rng.integers(0, C))]
    for i in range(1, N_EP):
        avoid = {cues[-1]} | ({cues[0]} if i == N_EP - 1 else set())
        cues.append(int(rng.choice([c for c in range(C) if c not in avoid])))
    return cues


def _pipeline(seed: int) -> dict:
    if seed in _CACHE:
        return _CACHE[seed]
    tmp = Path(tempfile.mkdtemp(prefix=f"me8_s{seed}_"))
    holdout = _train(seed, tmp)
    cues = _eval_cues(seed)
    kill = _kill_phase(seed, tmp, cues)
    _CACHE[seed] = {"tmp": tmp, "cues": cues, "holdout": holdout, **kill}
    return _CACHE[seed]


def _experiment(seed: int) -> dict:
    art = _pipeline(seed)
    cues = art["cues"]
    resume = _finish(art["tmp"], "resume", seed, cues)
    zeroed = _finish(art["tmp"], "zeroed", seed, cues)
    resume_acc = float(np.mean([p == c for p, c in zip(resume, cues)]))
    zeroed_acc = float(np.mean([p == c for p, c in zip(zeroed, cues)]))
    return {
        "holdout_acc": round(art["holdout"], 4),
        "killed_frac": art["killed_frac"],
        "mean_ckpt_step": round(art["mean_ckpt_step"], 2),
        "resume_acc": round(resume_acc, 4),
        "zeroed_acc": round(zeroed_acc, 4),
        "resume_vs_zeroed": round(resume_acc - zeroed_acc, 4),
    }


def _control(seed: int) -> dict:
    """Finish episode i from episode j's state: the answer must FOLLOW THE
    FILE (j's cue), not the episode — else the env leaks the answer."""
    art = _pipeline(seed)
    cues = art["cues"]
    try:
        cross = _finish(art["tmp"], "cross", seed, cues)
        restored_cues = [cues[(ep + 1) % N_EP] for ep in range(N_EP)]
        return {
            "acc_true_cue": round(float(np.mean(
                [p == c for p, c in zip(cross, cues)])), 4),
            "match_restored": round(float(np.mean(
                [p == c for p, c in zip(cross, restored_cues)])), 4),
        }
    finally:
        shutil.rmtree(art["tmp"], ignore_errors=True)
        _CACHE.pop(seed, None)


def _check(m: dict, c: dict) -> bool:
    return (m["holdout_acc"] >= MIN_HOLDOUT
            and m["killed_frac"] == 1.0
            and m["resume_acc"] >= MIN_RESUME
            and m["zeroed_acc"] <= MAX_ZEROED
            and m["resume_vs_zeroed"] >= MIN_GAP
            and c["acc_true_cue"] <= MAX_CROSS_TRUE
            and c["match_restored"] >= MIN_MATCH_RESTORED)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["ME.8"], _experiment, _check,
                    control_fn=_control, ledger=ledger)
