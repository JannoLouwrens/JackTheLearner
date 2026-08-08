"""WorkingMemory — the GRU hidden state that survives restarts (wm.state).

GOAL.md: "What he learned yesterday ... persists on disk, inspectable, across
restarts." EpisodicMemory is the diary; THIS is the moment — what Jack is
holding in mind mid-episode. docs/research/MEMORY.md 3.2 chose GRU recurrence
for it deliberately: O(1) per step, microseconds on ARM, and the entire
within-episode memory is one small tensor that can be checkpointed to disk
EVERY step. That makes it the only working memory that natively survives a
process kill — a transformer's KV cache or a context window dies with the
process unless you persist the whole prompt.

The contract (ME.8 is the ledger test; nothing here is claimed beyond it):

  - step(obs) advances one timestep and returns the head's logits over
    outputs. State lives in self.h (batch, hidden) plus self.step_idx.
  - checkpoint(path) atomically persists {h, step} — write to a tmp file then
    os.replace, the discipline T0.05 proved: a SIGKILL landing mid-write must
    never leave a corrupt wm.state for the next process to load.
  - restore(path) puts a FRESH process exactly where the dead one stopped.
  - reset() zeroes the state: the memoryless null that ME.8's zeroed-restart
    baseline is built from.

Weights and state are separate on purpose. Weights are the skill — saved
rarely, via state_dict like any module. wm.state is the moment — saved every
step, a few KB. Conflating them is how working memory ends up living only in
RAM and dying with the process.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn


class WorkingMemory(nn.Module):
    def __init__(self, obs_dim: int, n_out: int, hidden: int = 32):
        super().__init__()
        self.obs_dim, self.n_out, self.hidden = obs_dim, n_out, hidden
        self.cell = nn.GRUCell(obs_dim, hidden)
        # Update-gate bias +1 so the cell RETAINS by default — the GRU analogue
        # of LSTM forget-bias init (Jozefowicz et al. 2015). Without it,
        # latching a step-0 cue across 30 distractor steps fails to train at
        # some seeds (seed 2 sat at chance for 1000 iters).
        with torch.no_grad():
            for b in (self.cell.bias_ih, self.cell.bias_hh):
                b[hidden:2 * hidden] += 1.0
        self.head = nn.Linear(hidden, n_out)
        self.h = torch.zeros(1, hidden)   # plain attribute: state, not weights
        self.step_idx = 0

    def reset(self, batch: int = 1) -> None:
        self.h = torch.zeros(batch, self.hidden)
        self.step_idx = 0

    def step(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if self.h.shape[0] != obs.shape[0]:
            self.reset(obs.shape[0])
        self.h = self.cell(obs, self.h)
        self.step_idx += 1
        return self.head(self.h)

    def checkpoint(self, path: str) -> None:
        tmp = f"{path}.tmp"
        torch.save({"h": self.h.detach().clone(), "step": self.step_idx}, tmp)
        os.replace(tmp, path)

    def restore(self, path: str) -> int:
        snap = torch.load(path, weights_only=True)
        self.h = snap["h"]
        self.step_idx = int(snap["step"])
        return self.step_idx
