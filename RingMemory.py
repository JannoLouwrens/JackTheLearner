"""RingMemory — a working memory whose events ROTATE what is held.

The challenger for the Working-memory seat (`WM.02`; the incumbent is
`WorkingMemory.py`'s GRU buffer, which FAILs `WM.01` on two of three seeds).

The idea is the head-direction circuit's: a held state is not overwritten by
each event, it is TURNED by it. The state is K planes (hidden = 2K floats);
every step each plane is rotated by an angle that is a learned linear function
of the observation, then a learned gate may blend in an absolute write:

    h <- R(theta(obs)) h                    relative update — composition
    h <- (1 - g(obs)) h + g(obs) tanh(w(obs))   absolute write, gate biased shut

A rotation preserves the norm of what is held, so a cue survives any number of
distractor steps without a retain-bias trick and its gradient neither vanishes
nor explodes through time; and composing events is composing rotations, which
is the one thing a gated additive cell has to discover the hard way.

Nothing here is given: angles, gate, write, the resting state `h0` and the
head are all learned from scratch. Same contract as `WorkingMemory`
(`step` / `reset` / `checkpoint` / `restore`), same state size, so it can be
dropped into ME.8 and LF.02's rigs if it takes the seat.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn


class RingMemory(nn.Module):
    def __init__(self, obs_dim: int, n_out: int, hidden: int = 32,
                 rotate: bool = True):
        super().__init__()
        assert hidden % 2 == 0, "state is K planes: hidden must be even"
        self.obs_dim, self.n_out, self.hidden = obs_dim, n_out, hidden
        self.k = hidden // 2
        self.ang = nn.Linear(obs_dim, self.k)
        self.write = nn.Linear(obs_dim, hidden)
        self.gate = nn.Linear(obs_dim, 1)
        self.h0 = nn.Parameter(torch.randn(hidden) * 0.5)   # the resting state
        self.head = nn.Linear(hidden, n_out)
        with torch.no_grad():
            self.ang.bias.zero_()        # no event -> no turn, at init
            self.gate.bias.fill_(-3.0)   # hold by default
        # `rotate=False` is WM.02's ablation control: identical module with
        # the turn switched off and its weights frozen at zero.
        self.rotate = rotate
        if not rotate:
            with torch.no_grad():
                self.ang.weight.zero_()
            for p in self.ang.parameters():
                p.requires_grad_(False)
        self.h = self.h0.detach().unsqueeze(0).clone()   # state, not weights
        self.step_idx = 0

    def reset(self, batch: int = 1) -> None:
        self.h = self.h0.unsqueeze(0).expand(batch, -1)
        self.step_idx = 0

    def step(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if self.h.shape[0] != obs.shape[0]:
            self.reset(obs.shape[0])
        h = self.h
        if self.rotate:
            th = self.ang(obs)
            c, s = th.cos(), th.sin()
            a, b = h[:, :self.k], h[:, self.k:]
            h = torch.cat([c * a - s * b, s * a + c * b], dim=-1)
        g = torch.sigmoid(self.gate(obs))
        self.h = (1 - g) * h + g * torch.tanh(self.write(obs))
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
