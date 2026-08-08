"""PG.4 — the noisy-TV panel must actually trap naive curiosity.

GOAL.md stakes everything on curiosity that is REAL — and the classic way fake
curiosity dies is the noisy TV (Burda et al., arXiv:1808.04355): a source of
irreducible stochasticity gives a prediction-error agent infinite "novelty",
so it stares forever and learns nothing. The playground carries a noise panel
as a MANDATORY fixture for exactly this reason (playground.py:24). But a trap
that has never caught anything certifies nothing: before any later spec can
claim "his curiosity does NOT get trapped" (CU.3), this spec must show the
trap WORKS — a deliberately naive prediction-error agent fixates on the panel
— and that the dwell-time metric every curiosity claim must report actually
measures that fixation.

The rig: a rover in the real playground MJCF observing through a 24-ray
"retina" cast at panel height (mujoco.mj_ray against the actual geoms). Each
ray reports [normalized distance, texture value]. Texture is a fixed per-geom
value — except rays that land on `noise_panel`, which return a fresh uniform
sample every decision step: the re-randomizing texture, realized at the
observation level. The naive agent is the textbook construction the noisy-TV
argument is ABOUT: an online forward model (MLP predicting the next retina
from retina+action) whose prediction error is the intrinsic reward of a
tabular Q-learner over floor cells. Nothing anywhere rewards the panel; only
irreducible surprise does.

Declared rig conveniences (this spec certifies the TRAP, not navigation): the
rover is a non-colliding velocity-controlled slider (contype 0, no z dof) —
obstacles are visual-only for it. That is not a weakness but the point: with
trivial, fully learnable movement dynamics and a static world, the ONLY
irreducible prediction error in the noisy arm is the panel. If the agent
still fails to fixate, the trap is broken; if the static-texture control
fixates anyway, the dwell metric is measuring geometry, not noise.

Pre-registered claims (thresholds set from a seed-7 pilot — dwell 0.62,
null 0.17, control 0.001, ratio 3.2 — BEFORE any official seed ran):
  trap    the ICM agent's dwell share near the panel (last half of a 20k-step
          life) reaches >= 0.4 — discovery time varies by seed, fixation must
          still dominate the converged half
  null    a uniform random walk's dwell share stays <= 0.2, and the ICM
          agent beats it by >= 0.25 absolute
  cause   late intrinsic reward inside the dwell zone is >= 2x the reward
          outside — the fixation is FED by panel surprise, not floor-plan bias
CONTROL (must fail to fixate): the identical agent, panel texture static.
Every other geom, ray, weight and hyperparameter unchanged. Its dwell share
must stay <= 0.15 — if it fixates without noise, dwell measures geometry and
the fixture cannot certify any curiosity claim.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

PANEL_XY = (0.0, 5.9)           # playground.py: pos="0 {a-0.1} 1.2", a=6.0
DWELL_RADIUS = 2.0              # m — the dwell zone every curiosity claim reports
N_RAYS = 32
RAY_Z = 1.2                     # panel centre height
MAX_RANGE = 17.0                # > room diagonal: every ray lands on a surface
R_RESOLVE = 2.5                 # visual acuity: texture noise is fully resolved
                                # within this range and blurs toward the mean
                                # beyond it — a TV across the room is a shimmer,
                                # not a firehose. Keeps the trap LOCAL (the thing
                                # the dwell metric measures) instead of letting
                                # the agent farm surprise from mid-room, which a
                                # pilot measured: parked at 5.7 m with 36% map
                                # coverage vs the static arm's 74%.
N_DECISIONS = 20_000            # one life; sim dt 0.005 x 40 substeps = 0.2 s/decision
SUBSTEPS = 40                   # 0.3 m per decision at full speed — the rover
                                # must be able to CROSS the arena within a life
SPEED = 1.5                     # m/s commanded
CELL = 1.0                      # Q-state grid resolution over [-5.5, 5.5]^2
GAMMA = 0.95
Q_LR = 0.2
Q_INIT = 3.0                    # optimistic init: the naive agent sweeps the map
                                # because everywhere starts promising; only the
                                # panel keeps refreshing its value (the trap)
EPS_HI, EPS_LO = 1.0, 0.10      # linear decay over the first third, then fixed

ICM_DWELL_MIN = 0.40            # pre-registered (see docstring: set from a
NULL_DWELL_MAX = 0.20           # seed-7 pilot before any official seed ran)
MARGIN_MIN = 0.25
PANEL_REWARD_RATIO_MIN = 2.0
CONTROL_DWELL_MAX = 0.15        # the static-texture control must NOT fixate

_ACTIONS = [(0.0, 0.0)] + [
    (math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)) for k in range(8)
]


def _build():
    """Playground + rover. Returns (model, data, panel_gid, rover_bid, act_ids)."""
    sys.path.insert(0, str(REPO))
    import mujoco
    from playground import PlaygroundParams, build_mjcf

    # n_objects=0: free-floating clutter would add its own (learnable) dynamics
    # noise; the trap claim wants the panel to be the sole stochastic source.
    p = PlaygroundParams(seed=0, n_objects=0)
    xml = build_mjcf(p)
    rover = (
        '<body name="rover" pos="0 0 0.15">'
        '<joint name="rover_x" type="slide" axis="1 0 0" range="-5.5 5.5" damping="1"/>'
        '<joint name="rover_y" type="slide" axis="0 1 0" range="-5.5 5.5" damping="1"/>'
        '<geom name="rover" type="sphere" size="0.15" mass="1" contype="0" '
        'conaffinity="0" rgba="0.9 0.6 0.1 1"/></body>')
    actuators = (
        '<actuator>'
        '<velocity name="vx" joint="rover_x" kv="200" ctrlrange="-2 2" forcerange="-400 400"/>'
        '<velocity name="vy" joint="rover_y" kv="200" ctrlrange="-2 2" forcerange="-400 400"/>'
        '</actuator>')
    xml = xml.replace("</worldbody>", rover + "\n  </worldbody>")
    xml = xml.replace("</mujoco>", actuators + "\n</mujoco>")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return (model, data, model.geom("noise_panel").id, model.body("rover").id,
            (model.actuator("vx").id, model.actuator("vy").id))


class _Retina:
    """24 horizontal rays at panel height; per ray [distance, texture].

    Texture is a deterministic per-geom value except the noise panel, which is
    resampled uniformly every call while `noisy` — the noisy TV itself.
    """

    def __init__(self, model, panel_gid: int, rover_bid: int, noisy: bool, rng):
        import numpy as np
        self.model, self.panel = model, panel_gid
        self.exclude, self.noisy, self.rng = rover_bid, noisy, rng
        self.dirs = np.array(
            [[math.cos(2 * math.pi * k / N_RAYS),
              math.sin(2 * math.pi * k / N_RAYS), 0.0] for k in range(N_RAYS)])
        self._geomid = np.zeros(1, dtype=np.int32)
        # Stable, arbitrary per-geom texture values in [0.1, 0.9].
        self.tex = ((np.arange(model.ngeom) * 0.37) % 0.8) + 0.1

    def observe(self, data) -> "tuple":
        """Returns (obs vector, n_rays_on_panel)."""
        import mujoco
        import numpy as np
        x, y = float(data.qpos[-2]), float(data.qpos[-1])
        vx, vy = float(data.qvel[-2]), float(data.qvel[-1])
        pnt = np.array([x, y, RAY_Z])
        obs = np.empty(4 + 2 * N_RAYS, dtype=np.float32)
        obs[:4] = (x / 6.0, y / 6.0, vx / 2.0, vy / 2.0)
        hits = 0
        for k in range(N_RAYS):
            dist = mujoco.mj_ray(self.model, data, pnt, self.dirs[k], None, 1,
                                 self.exclude, self._geomid)
            gid = int(self._geomid[0])
            if dist < 0 or dist > MAX_RANGE:
                d, t = 1.0, 0.0
            elif gid == self.panel:
                hits += 1
                d = dist / MAX_RANGE
                if self.noisy:
                    amp = min(1.0, R_RESOLVE / max(dist, 1e-6))
                    t = 0.5 + amp * (float(self.rng.uniform()) - 0.5)
                else:
                    t = 0.55
            else:
                d, t = dist / MAX_RANGE, float(self.tex[gid])
            obs[4 + 2 * k] = d
            obs[5 + 2 * k] = t
        return obs, hits


def _dwell(x: float, y: float) -> bool:
    return math.hypot(x - PANEL_XY[0], y - PANEL_XY[1]) < DWELL_RADIUS


def _cell(x: float, y: float) -> int:
    cx = min(10, max(0, int((x + 5.5) / CELL)))
    cy = min(10, max(0, int((y + 5.5) / CELL)))
    return cy * 11 + cx


def _run_agent(seed: int, policy: str, noisy_panel: bool,
               n_decisions: int = N_DECISIONS) -> dict:
    """One life in the playground. policy: 'icm' or 'random'."""
    import mujoco
    import numpy as np

    model, data, panel_gid, rover_bid, (ax, ay) = _build()
    env_rng = np.random.RandomState(seed * 7919 + 13)
    agent_rng = np.random.RandomState(seed * 104729 + 7)
    retina = _Retina(model, panel_gid, rover_bid, noisy_panel, env_rng)

    fwd = opt = None
    if policy == "icm":
        import torch
        torch.manual_seed(seed)
        obs_dim = 4 + 2 * N_RAYS
        fwd = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + len(_ACTIONS), 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, obs_dim))
        opt = torch.optim.Adam(fwd.parameters(), lr=1e-3)
        buf_x, buf_y = [], []
        q = np.full((121, len(_ACTIONS)), Q_INIT)

    obs, _ = retina.observe(data)
    half = n_decisions // 2
    dwell_late = 0
    dwell_q = [0, 0, 0, 0]
    panel_hits_dwell, dwell_steps = 0, 0
    reward_in, n_in, reward_out, n_out = 0.0, 0, 0.0, 0
    visited = set()

    for t in range(n_decisions):
        x, y = float(data.qpos[-2]), float(data.qpos[-1])
        s = _cell(x, y)
        visited.add(s)
        if policy == "random":
            a = int(agent_rng.randint(len(_ACTIONS)))
        else:
            eps = max(EPS_LO,
                      EPS_HI - (EPS_HI - EPS_LO) * t / (n_decisions // 3))
            if agent_rng.uniform() < eps:
                a = int(agent_rng.randint(len(_ACTIONS)))
            else:
                best = np.flatnonzero(q[s] >= q[s].max() - 1e-12)
                a = int(best[agent_rng.randint(len(best))])

        data.ctrl[ax] = SPEED * _ACTIONS[a][0]
        data.ctrl[ay] = SPEED * _ACTIONS[a][1]
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        obs2, hits = retina.observe(data)
        x2, y2 = float(data.qpos[-2]), float(data.qpos[-1])
        in_dwell = _dwell(x2, y2)

        if policy == "icm":
            import torch
            with torch.no_grad():
                inp = torch.cat([torch.from_numpy(obs),
                                 torch.eye(len(_ACTIONS))[a]])
                r = float(((fwd(inp) - torch.from_numpy(obs2)) ** 2).sum())
            s2 = _cell(x2, y2)
            q[s, a] += Q_LR * (r + GAMMA * q[s2].max() - q[s, a])
            # Full-history replay: with a short window the model forgets the
            # far side of the map, and revisits masquerade as novelty — the
            # agent then chases its own forgetting instead of the panel.
            buf_x.append(inp.numpy())
            buf_y.append(obs2)
            idx = agent_rng.randint(len(buf_x), size=min(64, len(buf_x)))
            bx = torch.from_numpy(np.stack([buf_x[i] for i in idx]))
            by = torch.from_numpy(np.stack([buf_y[i] for i in idx]))
            loss = ((fwd(bx) - by) ** 2).sum(dim=1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if t >= half:
                if in_dwell:
                    reward_in += r; n_in += 1
                else:
                    reward_out += r; n_out += 1

        if in_dwell:
            dwell_q[min(3, 4 * t // n_decisions)] += 1
            if t >= half:
                dwell_late += 1
                panel_hits_dwell += hits
                dwell_steps += 1
        obs = obs2

    q_len = max(1, n_decisions // 4)
    out = {
        "dwell_share": round(dwell_late / half, 4),
        "dwell_by_quarter": "/".join(f"{d / q_len:.2f}" for d in dwell_q),
        "visited_cell_frac": round(len(visited) / 121, 4),
        "final_dist_to_panel": round(
            math.hypot(float(data.qpos[-2]) - PANEL_XY[0],
                       float(data.qpos[-1]) - PANEL_XY[1]), 3),
        "rays_on_panel_while_dwelling": round(
            panel_hits_dwell / max(1, dwell_steps), 2),
    }
    if policy == "icm":
        r_in = reward_in / max(1, n_in)
        r_out = reward_out / max(1, n_out)
        out["late_reward_in_zone"] = round(r_in, 4)
        out["late_reward_out_zone"] = round(r_out, 4)
        out["panel_reward_ratio"] = round(r_in / max(1e-9, r_out), 2)
    return out


def _experiment(seed: int) -> dict:
    icm = _run_agent(seed, "icm", noisy_panel=True)
    rnd = _run_agent(seed, "random", noisy_panel=True)
    return {
        "icm_dwell_share": icm["dwell_share"],
        "null_dwell_share": rnd["dwell_share"],
        "dwell_margin": round(icm["dwell_share"] - rnd["dwell_share"], 4),
        "panel_reward_ratio": icm["panel_reward_ratio"],
        "late_reward_in_zone": icm["late_reward_in_zone"],
        "late_reward_out_zone": icm["late_reward_out_zone"],
        "rays_on_panel_while_dwelling": icm["rays_on_panel_while_dwelling"],
        "icm_visited_cell_frac": icm["visited_cell_frac"],
        "icm_final_dist_to_panel": icm["final_dist_to_panel"],
    }


def _control(seed: int) -> dict:
    """Identical ICM agent, static panel texture: must NOT fixate."""
    icm = _run_agent(seed, "icm", noisy_panel=False)
    return {
        "icm_dwell_share": icm["dwell_share"],
        "panel_reward_ratio": icm["panel_reward_ratio"],
        "icm_visited_cell_frac": icm["visited_cell_frac"],
        "icm_final_dist_to_panel": icm["final_dist_to_panel"],
    }


def _check(m: dict, c: dict) -> bool:
    return (m["icm_dwell_share"] >= ICM_DWELL_MIN
            and m["null_dwell_share"] <= NULL_DWELL_MAX
            and m["dwell_margin"] >= MARGIN_MIN
            and m["panel_reward_ratio"] >= PANEL_REWARD_RATIO_MIN
            and m["rays_on_panel_while_dwelling"] > 0
            and c["icm_dwell_share"] <= CONTROL_DWELL_MAX)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["PG.4"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
