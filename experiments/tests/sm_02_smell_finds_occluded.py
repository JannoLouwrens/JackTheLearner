"""SM.02 — Smell finds what vision cannot see.

HYPOTHESIS (registry, unchanged). A Jack with the odour modality reaches
OCCLUDED food in fewer simulated seconds than an identical no-smell twin, AND
shows no significant advantage when the same food is in plain sight.

THE CONDITIONAL IS THE CLAIM. A test that only measured the occluded condition
could not distinguish "smell works" from "an extra input channel helped"
(capacity, a distance cue, a leak). So the registered metric is
`occluded_minus_visible_advantage`: the advantage must APPEAR when a wall is in
the way and VANISH when it is not, which is the measured shape of the audio
result the spec's notes cite (ManiWAV: audio pays under occlusion and
approximately nothing otherwise).

THE RIG, and what it deliberately is not. There is NO locomotion in this rig —
T2.01 is unresolved, and a claim about a sense must not be hostage to walking
(TA.02's precedent). The agent is a point nose at Z_NOSE with a heading,
moving at SPEED with three actions (forward / turn left / turn right) at the
world's own sniff rate (odour.SNIFF_HZ — the mammalian band, and the rate the
bilateral sensor is built for). Movement respects the world's solid geometry
via the same `mj_ray` the odour occlusion uses: a step into a wall is a step
not taken. Everything the agent can know arrives through three declared
channels, identical in dimension across every arm:

  position   [x/ARENA, y/ARENA, sin h, cos h] — needed so the no-smell twin
             CAN learn a search route; without it the twin is structurally
             incapable of search and the comparison measures the handicap,
             not the sense.
  sight      omnidirectional line-of-sight to the food point (the same
             `odour.line_of_sight` SM.01 certified as "light does not pass"):
             [seen, sin bearing, cos bearing, 1/(1+d)], zeros when blocked.
             Omnidirectional ON PURPOSE: the claim is about occlusion, not
             gaze control, and a fair visible condition wants vision at its
             strongest.
  smell      the bilateral OdourSensor (2*C + C = 12 floats), squashed by
             tanh(ODOUR_GAIN * conc) / tanh(DERIV_GAIN * dconc) so whiffs at
             2-5 m land mid-range (probe: mean conc 0.30 at 2 m downwind,
             0.08 at 5 m). Per arm this channel carries the REAL field, ZEROS,
             matched NOISE, or a WRONG-SOURCE field — never a different shape.

THE GEOMETRY, probed before writing (2026-08-20, seed 0, 300 random clear
points). Four candidate food sites on the south band, each inside a
`build_mjcf(shelters=...)` wind-break (SH.01's substrate — three 1 m walls,
open to -y, i.e. toward the south wall 1.5 m away):

    occluded world:  LOS-visible fraction 0.000 / 0.000 / 0.000 / 0.000
    open world:      LOS-visible fraction 0.890 / 0.910 / 0.887 / 0.867
    plume (wind +x): whiff fraction 0.56 at +2 m, 0.38 at +5 m on the band;
                     0.00 upwind, 0.00 crosswind 2 m north

So with shelters the food is invisible from essentially everywhere an agent
can stand, while its plume is smellable in a downwind cone; with the shelters
removed the same coordinates are visible from ~90% of the room. Food location
varies per episode over the four sites — with a FIXED location the no-smell
twin would memorise a route from its position channel and the advantage of
knowing WHICH site would be zero by construction.

THE ARMS (each its own DQN, identical nets, identical budgets):
  smell/occ      the claim arm: real field, shelters present.
  nosmell/occ    the twin: smell dims are zeros. Must learn to SEARCH
                 (tour the mouths, use sight through each opening).
  placebo/occ    control (b), MUST FAIL: the smell dims carry the real
                 field sampled at an IID RANDOM pose each sniff — matched
                 marginal statistics by construction, zero mutual
                 information with the agent's own pose.
  shuffled/occ   control (a), MUST FAIL: a real, spatially coherent plume
                 whose source sits at a WRONG site (a different episode's
                 layout, resampled to never equal the true site).
  smell/vis      \\ the must-pass control: shelters removed, same sites,
  nosmell/vis    /  the twins must be statistically indistinguishable.
  random         both conditions, untrained: the learning gate's floor.

WHY THE LEARNING GATES ARE SHAPED THIS WAY (T2.02's law: a non-learner cannot
arbitrate). smell/vis and nosmell/vis beating random/vis certifies the
learner+rig can learn AT ALL (sight is the easy channel). nosmell/occ beating
random/occ certifies occluded search is learnable WITHOUT smell — if it is
not, occluded time differences are noise between two flailing agents: VOID.
Given those three, a smell/occ arm that still loses is a FINDING about the
sense (the registry's kills clause), not a broken arm — so no learning gate
shields the claim arm itself.

────────────────────────────────────────────────────────────────────────────
PROVISIONAL GATES — NOT YET FROZEN. `run()` REFUSES the registered run while
`_GATES_FROZEN` is False. The pilot (seeds 90-92, disjoint from the
registered 0-2, full production size, on the GPU) must price:
  * the between-seed and within-seed spread of every time-to-food mean,
  * the resolving power of the advantage gates against that spread
    (OVERSIGHT B3: a gate must clear the instrument's own std by the margin
    intended, priced BEFORE the recording run, not after),
  * timeout rates (a clipped mean compresses true advantages),
  * the whiff-fraction and LOS tripwires on production seeds.
The next iteration replaces the constants below with the pilot's numbers,
records the pilot table here, sets `_GATES_FROZEN = True`, commits, and only
then dispatches `scripts/dispatch.sh SM.02`.

PILOT 1 (2026-08-20, kernel jack-ladder-1787185633, P100, seeds 90-92,
n_train=400, n_eval=48 — /data/sm02_pilot.json). VERDICT: the rig's own
learning gates fail on ALL THREE seeds — gates cannot be frozen from a
non-learning pilot, and a registered run would have recorded VOID.

    t_mean (s) / timeout_frac      seed 90        seed 91        seed 92
    smell/occ                      58.56 / 0.96   51.95 / 0.83   60.00 / 1.00
    nosmell/occ                    60.00 / 1.00   60.00 / 1.00   60.00 / 1.00
    placebo/occ                    60.00 / 1.00   60.00 / 1.00   60.00 / 1.00
    shuffled/occ                   60.00 / 1.00   60.00 / 1.00   58.53 / 0.96
    smell/vis                      60.00 / 1.00   60.00 / 1.00   60.00 / 1.00
    nosmell/vis                    60.00 / 1.00   41.45 / 0.65   58.86 / 0.98
    random/vis                     60.00 / 1.00   58.91 / 0.94   59.60 / 0.98
    rig tripwires: occ_hidden 1.00/1.00/1.00, vis_seen 0.89/0.89/0.88,
    smell/occ whiff 0.12/0.09/0.11, det_ok 1.0 everywhere.

DIAGNOSIS (probed locally the same day, before any code change). A scripted
turn-to-bearing policy solves the VISIBLE condition in 18-20 s (timeout
0.12-0.19) on the same rig the DQN times out on, and a straight 3-step walk
enters a shelter mouth and reaches food — so the apparatus is sound and the
failure is the learner's: reward is terminal-only (+1 on reach, -1/300 per
step) and the random-policy reach rate is ~0-2 % per episode, so 400 training
episodes carry almost no positive signal. Seed 91's nosmell/vis (41.45 s) and
smell/occ (51.95 s) show the budget sits exactly at the learnability edge.

REPAIR, pre-registered before pilot 2: potential-based reward shaping
(Ng, Harada & Russell 1999), TRAINING ONLY, identical in every arm:
    r' = r + RL_GAMMA * phi(s') - phi(s),   phi(terminal) = 0
Potential-based shaping provably preserves optimal policies FOR ANY phi; it
changes what is LEARNABLE at this budget, not what is optimal. It is computed
from the true food position in every arm alike, so no arm's OBSERVATIONS gain
anything — the twins differ only in the smell channel, exactly as before —
and eval remains raw unshaped time-to-food. The shuffled/placebo must-fail
controls keep their meaning: their smell channels still carry zero/wrong
information and their shaping is identical to the twin's.

THE POTENTIAL MUST RESPECT THE WALLS (CPU check, LESSONS rule, 2026-08-20,
seed 90, full budget, /data/sm02_learnability_{vis,occ}.json). The first
repair used phi = -euclid_dist(s, food)/ARENA and the pre-dispatch CPU check
caught what pilot 2 would have paid 2 GPU hours to learn:
    nosmell/vis  trained 42.4 s vs random 59.2 s  (ratio 0.72 — learning)
    nosmell/occ  trained 60.0 s vs random 60.0 s  (ratio 1.00 — NOTHING)
Diagnosis: occluded food sits inside a three-walled shelter, so the Euclidean
potential has its steepest descent INTO the shelter's back wall — the shaping
itself builds a local minimum exactly where the wall is, and greedy descent
pins the agent there. The fix keeps the identical-in-every-arm shaping but
makes the potential geometry-aware:
    phi(s) = -geodesic_dist(s, food)/ARENA
where geodesic_dist is a Dijkstra distance field on a GRID_STEP lattice whose
edge passability is decided by THE SAME mj_ray + R_AGENT test step() moves
by — descent of this potential can never dead-end against geometry the agent
cannot cross. CPU verification at full budget (seed 90) before dispatch:
    [PENDING — numbers land here from the check before any dispatch]
────────────────────────────────────────────────────────────────────────────

COVERS: smell (claim).
"""

from __future__ import annotations

import fcntl
import heapq
import json
import math
import os
import time

import numpy as np

import mujoco

import playground as pg

from .. import odour
from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..gpu import build_job, result_json, submit

# The claim is about the odour field in the playground's geometry — both hash
# into the certificate (SM.01's convention).
IMPL_DEPS = ["playground.py", "experiments/odour.py"]

SEEDS = [0, 1, 2]

# ── rig geometry, pre-registered ─────────────────────────────────────────
# Four sites on the clutter-free south band; every shelter opens toward the
# south wall 1.5 m away, which is what makes occlusion total (probe above).
SITES = ((-4.5, -4.5), (-1.5, -4.5), (1.5, -4.5), (4.5, -4.5))
SHELTERS = tuple((f"site{i}", float(x), float(y))
                 for i, (x, y) in enumerate(SITES))
Z_NOSE = 0.5                   # below the 1.0 m shelter walls
Z_FOOD = 0.30
WIND = (1.0, 0.0, 0.0)         # along the band; SM.01's convention
DT = 1.0 / odour.SNIFF_HZ      # 0.2 s — decisions at the sniff rate
SPEED = 1.0                    # m/s; forward step = 0.2 m
TURN = math.radians(30.0)
T_MAX_S = 60.0
MAX_STEPS = int(round(T_MAX_S / DT))     # 300
R_REACH = 0.5                  # xy metres; the mouth line is ~0.49 from centre
R_AGENT = 0.25                 # collision margin ahead of the nose
PREROLL_S = 40.0               # plume spin-up before t=0 (500-puff steady state)
ARENA = 6.0                    # wall half-extent (playground contract)
POS_CLAMP = 5.7
SPAWN_BOUND = 5.2
SPAWN_SITE_CLEAR = 1.5         # never spawn on top of a site
ODOUR_GAIN = 3.0               # tanh gain on concentrations (probe: 0.08-0.30)
DERIV_GAIN = 0.5               # tanh gain on the derivative channel
WHIFF_THR_MULT = 10.0          # a whiff is > this * NOISE_SIGMA (SM.01's line)
GRID_STEP = 0.2                # geodesic lattice pitch = one forward step

# Spawn exclusion rectangles (xlo, xhi, ylo, yhi): the declared fixtures a
# point should not materialise inside. At Z_NOSE most clutter is below the
# nose plane; these are the fixtures tall enough to matter, plus the pool pit.
EXCLUDE = (
    (-3.8, -1.2, 0.9, 3.1),    # ramp
    (1.7, 3.9, 1.4, 3.0),      # stairs
    (-0.9, 0.9, -3.2, -1.8),   # ladder + platform
    (0.9, 4.3, -4.1, -0.7),    # pool pit
)

N_ACTIONS = 3                  # forward, turn left, turn right
OBS_DIM = 8 + odour.OBS_DIM    # 4 pose + 4 sight + 12 smell = 20

# ── training/eval protocol, pre-registered ───────────────────────────────
N_TRAIN = 400                  # episodes per arm
N_EVAL = 48                    # greedy episodes per arm; sites round-robin
EPS_DECAY_EP = 300             # epsilon 1.0 -> RL_EPS_END across these
EVAL_SEED_BASE = 500_000       # eval episode seeds, disjoint from training

# The DQN, constants standard and declared (TA.02's convention).
RL_GAMMA = 0.99
RL_LR = 1e-3
RL_BATCH = 64
RL_BUFFER = 100_000
RL_TARGET_SYNC = 1000
RL_EPS_END = 0.05
RL_HIDDEN = 64

# ── PROVISIONAL GATES (see the banner in the docstring) ──────────────────
_GATES_FROZEN = False          # run() refuses the registered run until True

OCC_ADV_GATE = 6.0             # pooled t_nosmell_occ - t_smell_occ, seconds
SEED_OCC_ADV_FLOOR = 2.0       # ...and per seed at least this
VIS_ADV_CEIL = 2.0             # |pooled visible advantage| must sit under
CTRL_ADV_CEIL = 2.0            # placebo and shuffled occluded advantage ceiling
LEARN_VIS_FRAC = 0.6           # t_vis(arm) <= this * t_vis(random)
LEARN_OCC_FRAC = 0.85          # t_nosmell_occ <= this * t_random_occ
OCC_HIDDEN_MIN = 0.95          # LOS tripwire: occluded food unseen from >= this
VIS_SEEN_MIN = 0.80            # ...and open food seen from >= this
WHIFF_MIN = 0.02               # smell/occ eval steps with a whiff, at least


# ── the world ────────────────────────────────────────────────────────────
class _Rig:
    """One condition's world: geometry only, never stepped. The playground is
    static in this rig; the only dynamics are the plume's, and `PuffField`
    owns those."""

    def __init__(self, seed: int, occluded: bool):
        p = pg.PlaygroundParams(seed=seed)
        xml = pg.build_mjcf(p, shelters=SHELTERS if occluded else ())
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.occluded = occluded
        self._geo_edges = None         # lazy: built on first geodesic() call
        self._geo_dist: dict = {}      # site_idx -> Dijkstra distance field

    def _geo_build(self):
        """Passability graph on a GRID_STEP lattice at Z_NOSE. An edge exists
        exactly when the SAME mj_ray + R_AGENT test that gates step() clears
        it, so descent of the resulting distance field can never dead-end
        against geometry the agent cannot cross."""
        n = int(round(2 * POS_CLAMP / GRID_STEP)) + 1
        xs = np.linspace(-POS_CLAMP, POS_CLAMP, n)
        edges: list = [[] for _ in range(n * n)]
        gid = np.zeros(1, dtype=np.int32)
        for i in range(n):
            for j in range(n):
                p = np.array([xs[i], xs[j], Z_NOSE])
                for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
                    i2, j2 = i + di, j + dj
                    if not (0 <= i2 < n and 0 <= j2 < n):
                        continue
                    seg = math.hypot(di, dj) * GRID_STEP
                    v = np.array([di * GRID_STEP / seg,
                                  dj * GRID_STEP / seg, 0.0])
                    hit = mujoco.mj_ray(self.model, self.data, p, v,
                                        None, 1, -1, gid)
                    if 0.0 <= hit < seg + R_AGENT:
                        continue
                    a, b = i * n + j, i2 * n + j2
                    edges[a].append((b, seg))
                    edges[b].append((a, seg))
        self._geo_n, self._geo_xs, self._geo_edges = n, xs, edges

    def geodesic(self, site_idx: int, pos) -> float:
        """Walkable distance (m) from pos to SITES[site_idx] on the lattice;
        Euclidean fallback for the (unreachable-cell) corner case."""
        if self._geo_edges is None:
            self._geo_build()
        if site_idx not in self._geo_dist:
            n = self._geo_n
            sx, sy = SITES[site_idx]
            src = (int(round((sx + POS_CLAMP) / GRID_STEP)) * n
                   + int(round((sy + POS_CLAMP) / GRID_STEP)))
            dist = np.full(n * n, np.inf)
            dist[src] = 0.0
            heap = [(0.0, src)]
            while heap:
                d, u = heapq.heappop(heap)
                if d > dist[u]:
                    continue
                for v2, w in self._geo_edges[u]:
                    nd = d + w
                    if nd < dist[v2]:
                        dist[v2] = nd
                        heapq.heappush(heap, (nd, v2))
            self._geo_dist[site_idx] = dist
        n = self._geo_n
        i = int(np.clip(round((pos[0] + POS_CLAMP) / GRID_STEP), 0, n - 1))
        j = int(np.clip(round((pos[1] + POS_CLAMP) / GRID_STEP), 0, n - 1))
        d = float(self._geo_dist[site_idx][i * n + j])
        if not math.isfinite(d):
            sx, sy = SITES[site_idx]
            return math.hypot(pos[0] - sx, pos[1] - sy)
        return d

    def los_tripwire(self, n: int = 300, seed: int = 7) -> float:
        """Fraction of random clear poses with line of sight to a site,
        averaged over sites. The design probe, kept as a live gate."""
        rng = np.random.RandomState(seed)
        pts = []
        while len(pts) < n:
            x, y = rng.uniform(-SPAWN_BOUND, SPAWN_BOUND, 2)
            if min(math.hypot(x - sx, y - sy) for sx, sy in SITES) \
                    < SPAWN_SITE_CLEAR:
                continue
            pts.append((x, y))
        fracs = []
        for sx, sy in SITES:
            vis = [odour.line_of_sight(self.model, self.data,
                                       (x, y, Z_NOSE), (sx, sy, Z_FOOD))[0]
                   for x, y in pts]
            fracs.append(float(np.mean(vis)))
        return float(np.mean(fracs))


class _Episode:
    """One food-finding episode: a site, a plume, a spawn, a clock."""

    def __init__(self, rig: _Rig, site_idx: int, ep_seed: int, arm: str):
        self.rig = rig
        self.arm = arm
        self.rng = np.random.RandomState(ep_seed)
        sx, sy = SITES[site_idx]
        self.food = np.array([sx, sy, Z_FOOD])
        self.site_idx = site_idx

        self.field = None
        self.sensor = None
        if arm in ("smell", "placebo"):
            src = odour.Source("food0", "food", (sx, sy, Z_FOOD))
            self.field = odour.PuffField([src], wind=WIND, seed=ep_seed,
                                         los=True)
        elif arm == "shuffled":
            wrong = int(self.rng.choice(
                [i for i in range(len(SITES)) if i != site_idx]))
            wx, wy = SITES[wrong]
            src = odour.Source("food0", "food", (wx, wy, Z_FOOD))
            self.field = odour.PuffField([src], wind=WIND, seed=ep_seed,
                                         los=True)
        if self.field is not None:
            for _ in range(int(round(PREROLL_S / DT))):
                self.field.step(DT)
            self.sensor = odour.OdourSensor(self.field)

        self.pos = self._spawn()
        self.heading = float(self.rng.uniform(-math.pi, math.pi))
        self.t = 0.0
        self.whiffs = 0
        self.sniffs = 0

    def _spawn(self) -> np.ndarray:
        for _ in range(1000):
            x, y = self.rng.uniform(-SPAWN_BOUND, SPAWN_BOUND, 2)
            if min(math.hypot(x - sx, y - sy) for sx, sy in SITES) \
                    < SPAWN_SITE_CLEAR:
                continue
            if any(xlo <= x <= xhi and ylo <= y <= yhi
                   for xlo, xhi, ylo, yhi in EXCLUDE):
                continue
            return np.array([x, y, Z_NOSE])
        raise RuntimeError("no clear spawn in 1000 draws")

    def obs(self) -> np.ndarray:
        o = np.zeros(OBS_DIM, dtype=np.float32)
        x, y = float(self.pos[0]), float(self.pos[1])
        h = self.heading
        o[0], o[1] = x / ARENA, y / ARENA
        o[2], o[3] = math.sin(h), math.cos(h)

        clear, _ = odour.line_of_sight(self.rig.model, self.rig.data,
                                       self.pos, self.food)
        if clear:
            b = math.atan2(self.food[1] - y, self.food[0] - x) - h
            d = math.hypot(self.food[0] - x, self.food[1] - y)
            o[4:8] = (1.0, math.sin(b), math.cos(b), 1.0 / (1.0 + d))

        if self.sensor is not None:
            if self.arm == "placebo":
                # matched-statistics noise: the REAL field at an iid random
                # pose — zero mutual information with the agent's own pose
                rp = np.array([self.rng.uniform(-SPAWN_BOUND, SPAWN_BOUND),
                               self.rng.uniform(-SPAWN_BOUND, SPAWN_BOUND),
                               Z_NOSE])
                rh = float(self.rng.uniform(-math.pi, math.pi))
                raw = self.sensor.obs(rp, rh, self.t, model=self.rig.model,
                                      data=self.rig.data, rng=self.rng)
            else:
                raw = self.sensor.obs(self.pos, h, self.t,
                                      model=self.rig.model,
                                      data=self.rig.data, rng=self.rng)
            o[8:16] = np.tanh(ODOUR_GAIN * raw[:8])
            o[16:20] = np.tanh(DERIV_GAIN * raw[8:])
            fc = odour.CHANNEL_INDEX["food"]
            self.sniffs += 1
            if max(raw[fc], raw[odour.C + fc]) \
                    > WHIFF_THR_MULT * odour.NOISE_SIGMA:
                self.whiffs += 1
        return o

    def step(self, action: int) -> bool:
        """Apply one decision; returns True when the food is reached."""
        if action == 1:
            self.heading += TURN
        elif action == 2:
            self.heading -= TURN
        else:
            dirv = np.array([math.cos(self.heading),
                             math.sin(self.heading), 0.0])
            step_len = SPEED * DT
            gid = np.zeros(1, dtype=np.int32)
            hit = mujoco.mj_ray(self.rig.model, self.rig.data, self.pos,
                                dirv, None, 1, -1, gid)
            if not (0.0 <= hit < step_len + R_AGENT):
                self.pos = self.pos + dirv * step_len
                self.pos[:2] = np.clip(self.pos[:2], -POS_CLAMP, POS_CLAMP)
        if self.field is not None:
            self.field.step(DT)
        self.t += DT
        return self.dist() < R_REACH

    def dist(self) -> float:
        """xy distance to the food — the reach test's argument."""
        return math.hypot(self.pos[0] - self.food[0],
                          self.pos[1] - self.food[1])

    def phi(self) -> float:
        """Shaping potential (docstring REPAIR): negative geodesic distance
        to the TRUE food, identical in every arm, never in any observation."""
        return -self.rig.geodesic(self.site_idx, self.pos) / ARENA


# ── the learner (TA.02's DQN, navigation-shaped) ─────────────────────────
# Declared per-arm stream offsets: str.hash() is process-randomised and would
# make a "seeded" run unrepeatable.
ARM_OFFSET = {"smell": 11, "nosmell": 23, "placebo": 37, "shuffled": 41}


def _make_net(torch, nn, dev):
    return nn.Sequential(
        nn.Linear(OBS_DIM, RL_HIDDEN), nn.ReLU(),
        nn.Linear(RL_HIDDEN, RL_HIDDEN), nn.ReLU(),
        nn.Linear(RL_HIDDEN, N_ACTIONS)).to(dev)


def _train_arm(rig: _Rig, arm: str, seed: int, n_train: int,
               torch, nn, dev):
    torch.manual_seed(seed * 131 + ARM_OFFSET[arm])
    net, target = _make_net(torch, nn, dev), _make_net(torch, nn, dev)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=RL_LR)
    buf, gstep = [], 0
    rng = np.random.RandomState(seed * 7079 + ARM_OFFSET[arm])

    def train_step():
        nonlocal gstep
        if len(buf) >= RL_BATCH:
            idx = rng.randint(0, len(buf), RL_BATCH)
            o = torch.tensor(np.stack([buf[i][0] for i in idx]),
                             dtype=torch.float32, device=dev)
            a = torch.tensor([buf[i][1] for i in idx], device=dev)
            r = torch.tensor([buf[i][2] for i in idx],
                             dtype=torch.float32, device=dev)
            o2 = torch.tensor(np.stack([buf[i][3] for i in idx]),
                              dtype=torch.float32, device=dev)
            d = torch.tensor([float(buf[i][4]) for i in idx],
                             dtype=torch.float32, device=dev)
            with torch.no_grad():
                y = r + RL_GAMMA * (1.0 - d) * target(o2).max(1).values
            q = net(o).gather(1, a.view(-1, 1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(q, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        gstep += 1
        if gstep % RL_TARGET_SYNC == 0:
            target.load_state_dict(net.state_dict())

    for ep in range(n_train):
        eps = max(RL_EPS_END,
                  1.0 - (1.0 - RL_EPS_END) * ep / EPS_DECAY_EP)
        e = _Episode(rig, site_idx=int(rng.randint(len(SITES))),
                     ep_seed=seed * 100_003 % (2**31) + ep, arm=arm)
        obs = e.obs()
        phi = e.phi()
        for _ in range(MAX_STEPS):
            if rng.rand() < eps:
                act = int(rng.randint(N_ACTIONS))
            else:
                with torch.no_grad():
                    q = net(torch.tensor(obs, dtype=torch.float32,
                                         device=dev).unsqueeze(0))
                act = int(q.argmax(1).item())
            reached = e.step(act)
            # potential-based shaping (docstring REPAIR): training only,
            # identical in every arm, phi(terminal) = 0
            phi2 = 0.0 if reached else e.phi()
            r = ((1.0 if reached else -1.0 / MAX_STEPS)
                 + RL_GAMMA * phi2 - phi)
            phi = phi2
            obs2 = e.obs()
            if len(buf) >= RL_BUFFER:
                buf.pop(0)
            buf.append((obs, act, r, obs2, reached))
            train_step()
            obs = obs2
            if reached:
                break
    return net


def _eval_arm(rig: _Rig, net, arm: str, seed: int, n_eval: int,
              torch=None, dev=None, rng_random=None) -> dict:
    """Greedy (or random, when net is None) times-to-food. Sites round-robin
    so every site is scored equally; episode seeds disjoint from training."""
    times, whiff_frac = [], []
    det_ok = True
    for i in range(n_eval):
        e = _Episode(rig, site_idx=i % len(SITES),
                     ep_seed=seed * 100_003 % (2**31) + EVAL_SEED_BASE + i,
                     arm=arm)
        obs = e.obs()
        t = T_MAX_S
        for k in range(MAX_STEPS):
            if net is None:
                act = int(rng_random.randint(N_ACTIONS))
            else:
                with torch.no_grad():
                    ot = torch.tensor(obs, dtype=torch.float32,
                                      device=dev).unsqueeze(0)
                    q1, q2 = net(ot), net(ot)
                if not torch.equal(q1, q2):    # the .eval() lesson, asserted
                    det_ok = False
                act = int(q1.argmax(1).item())
            if e.step(act):
                t = (k + 1) * DT
                break
            obs = e.obs()
        times.append(t)
        if e.sniffs:
            whiff_frac.append(e.whiffs / e.sniffs)
    return {
        "t_mean": float(np.mean(times)),
        "t_median": float(np.median(times)),
        "t_std": float(np.std(times)),
        "timeout_frac": float(np.mean([t >= T_MAX_S for t in times])),
        "whiff_frac": float(np.mean(whiff_frac)) if whiff_frac else 0.0,
        "det_ok": float(det_ok),
    }


# ── one seed, remotely ───────────────────────────────────────────────────
# (arm, condition) pairs: which get trained, and where each is evaluated.
TRAINED = (("smell", "occ"), ("nosmell", "occ"), ("placebo", "occ"),
           ("shuffled", "occ"), ("smell", "vis"), ("nosmell", "vis"))


def _one_seed(seed: int, n_train: int, n_eval: int) -> dict:
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rigs = {"occ": _Rig(seed, occluded=True),
            "vis": _Rig(seed, occluded=False)}
    row: dict = {"seed": int(seed),
                 "occ_hidden_frac": 1.0 - rigs["occ"].los_tripwire(),
                 "vis_seen_frac": rigs["vis"].los_tripwire()}

    for arm, cond in TRAINED:
        t0 = time.time()
        net = _train_arm(rigs[cond], arm, seed, n_train, torch, nn, dev)
        ev = _eval_arm(rigs[cond], net, arm, seed, n_eval,
                       torch=torch, dev=dev)
        for k, v in ev.items():
            row[f"{arm}_{cond}_{k}"] = v
        row[f"{arm}_{cond}_train_s"] = round(time.time() - t0, 1)

    rng = np.random.RandomState(seed * 41)
    for cond in ("occ", "vis"):
        ev = _eval_arm(rigs[cond], None, "nosmell", seed, n_eval,
                       rng_random=rng)
        for k, v in ev.items():
            row[f"random_{cond}_{k}"] = v
    return row


def remote_run(seeds: list, n_train: int = N_TRAIN,
               n_eval: int = N_EVAL) -> dict:
    out = {"seeds": [], "gpu": "cpu",
           "n_train": int(n_train), "n_eval": int(n_eval)}
    try:
        import torch
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    for seed in seeds:
        out["seeds"].append(_one_seed(int(seed), n_train, n_eval))
        print(f"seed {seed} done", flush=True)
    return out


# ── remote plumbing (TA.02's shape) ──────────────────────────────────────
JOB = r'''
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q", "mujoco"], check=True)
import json, os
from experiments.tests.sm_02_smell_finds_occluded import remote_run
out = remote_run(__SEEDS__, n_train=__NTRAIN__, n_eval=__NEVAL__)
json.dump(out, open(os.path.join(os.environ["JACK_OUT"], "sm202.json"), "w"),
          indent=1)
print("DONE", flush=True)
'''

_CACHE: dict = {}


def _submit(seeds: list, n_train: int = N_TRAIN,
            n_eval: int = N_EVAL, est_hours: float = 2.0,
            timeout_s: int = 16200) -> dict:
    # Sizing: local smoke measured ~2.9 ms/decision (field + rays + DQN update
    # on 4 shared ARM cores). Worst case, every episode timing out: 6 arms x
    # 400 eps x 300 steps x 3 seeds ~ 2.2 h here, less on Kaggle, and much
    # less once policies start reaching food. timeout_s is sized to the worst
    # case plus margin (the budget lesson: multiply by seeds AND arms).
    body = (JOB.replace("__SEEDS__", repr(list(seeds)))
               .replace("__NTRAIN__", str(int(n_train)))
               .replace("__NEVAL__", str(int(n_eval))))
    job = build_job(body)
    r = submit(job, prefer="kaggle", est_hours=est_hours,
               timeout_s=timeout_s, fetch=["sm202.json"])
    if not r.ok:
        raise RuntimeError(f"GPU submission failed: {r.message}")
    data = result_json(r, "sm202.json")
    data["backend"] = r.backend
    return data


def _row(seed: int) -> dict:
    if not _CACHE:
        _CACHE.update(_submit(SEEDS))
    for row in _CACHE["seeds"]:
        if row["seed"] == seed:
            return row
    raise KeyError(f"seed {seed} missing from remote result")


# ── the ledger interface ─────────────────────────────────────────────────
def _experiment(seed: int) -> dict:
    r = _row(seed)
    m = {k: v for k, v in r.items() if k != "seed"}
    m["gpu"] = _CACHE.get("gpu", "?")
    m["backend"] = _CACHE.get("backend", "?")

    # the two advantages and the registered metric, per seed; seed means of
    # these are the pooled values run_spec hands _check
    m["occ_advantage_s"] = m["nosmell_occ_t_mean"] - m["smell_occ_t_mean"]
    m["vis_advantage_s"] = m["nosmell_vis_t_mean"] - m["smell_vis_t_mean"]
    m["occluded_minus_visible_advantage"] = (m["occ_advantage_s"]
                                             - m["vis_advantage_s"])

    m["rig_ok"] = float(
        m["occ_hidden_frac"] >= OCC_HIDDEN_MIN
        and m["vis_seen_frac"] >= VIS_SEEN_MIN
        and m["smell_occ_whiff_frac"] >= WHIFF_MIN
        and all(m[f"{a}_{c}_det_ok"] == 1.0 for a, c in TRAINED))
    m["learn_ok"] = float(
        m["smell_vis_t_mean"] <= LEARN_VIS_FRAC * m["random_vis_t_mean"]
        and m["nosmell_vis_t_mean"] <= LEARN_VIS_FRAC * m["random_vis_t_mean"]
        and m["nosmell_occ_t_mean"] <= LEARN_OCC_FRAC * m["random_occ_t_mean"])
    m["seed_floor_ok"] = float(m["occ_advantage_s"] >= SEED_OCC_ADV_FLOOR)
    return m


def _control(seed: int) -> dict:
    r = _row(seed)
    c = {
        "placebo_occ_t_mean": r["placebo_occ_t_mean"],
        "shuffled_occ_t_mean": r["shuffled_occ_t_mean"],
        "nosmell_occ_t_mean_c": r["nosmell_occ_t_mean"],
        "vis_advantage_s_c": (r["nosmell_vis_t_mean"]
                              - r["smell_vis_t_mean"]),
    }
    # controls (a) and (b): the advantage a broken channel buys over the twin
    c["placebo_advantage_s"] = (r["nosmell_occ_t_mean"]
                                - r["placebo_occ_t_mean"])
    c["shuffled_advantage_s"] = (r["nosmell_occ_t_mean"]
                                 - r["shuffled_occ_t_mean"])
    c["c_placebo_ok"] = float(c["placebo_advantage_s"] <= CTRL_ADV_CEIL)
    c["c_shuffled_ok"] = float(c["shuffled_advantage_s"] <= CTRL_ADV_CEIL)
    # the must-pass control: visible twins indistinguishable
    c["c_visible_ok"] = float(abs(c["vis_advantage_s_c"]) <= VIS_ADV_CEIL)
    return c


def _check(m: dict, c: dict):
    # the rig and the learning gates: an invalid apparatus is not evidence
    if m.get("rig_ok", 0.0) != 1.0:
        return Status.VOID
    if m.get("learn_ok", 0.0) != 1.0:
        return Status.VOID

    # must-fail controls cleared -> the advantage is not the odour
    # information -> the claim's mechanism is refuted: FAIL (registry kills)
    if c.get("c_placebo_ok", 0.0) != 1.0:
        return False
    if c.get("c_shuffled_ok", 0.0) != 1.0:
        return False
    # the must-pass side: an advantage that survives plain sight is a leak
    # (the registry's falsified_by, second clause)
    if c.get("c_visible_ok", 0.0) != 1.0:
        return False

    return bool(m.get("seed_floor_ok", 0.0) == 1.0
                and m.get("occ_advantage_s", 0.0) >= OCC_ADV_GATE
                and abs(m.get("vis_advantage_s", 99.0)) <= VIS_ADV_CEIL
                and m.get("occluded_minus_visible_advantage", 0.0)
                >= OCC_ADV_GATE - VIS_ADV_CEIL)


def run(ledger: Ledger | None = None):
    if not _GATES_FROZEN:
        raise RuntimeError(
            "SM.02's gates are PROVISIONAL. Run the pilot (seeds 90-92, "
            "python -m experiments.tests.sm_02_smell_finds_occluded pilot), "
            "freeze the gates from its numbers, set _GATES_FROZEN = True, "
            "commit, and only then dispatch the registered run.")
    return run_spec(BY_ID["SM.02"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


# ── pilot / smoke CLI ────────────────────────────────────────────────────
GPU_LOCK = "/tmp/jack-ladder-gpu.lock"


def _pilot():
    """Full-size pilot on disjoint seeds 90-92, on the GPU, holding the
    project's GPU lock (dispatch.sh's resource — one kernel at a time)."""
    lock = open(GPU_LOCK, "a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"REFUSING: {GPU_LOCK} is held; one GPU run at a "
                         "time. Retry when the holder finishes.")
    # A pilot runs outside run_spec, so JACK_SPEC_ID is unset and its receipt
    # would read spec:"" — 27 of 49 receipts were unattributable that way
    # (overseer 20th-audit B2). Name the spec and the phase so pilot spend is
    # summable separately from registered spend.
    os.environ["JACK_SPEC_ID"] = "SM.02"
    os.environ["JACK_SPEC_PHASE"] = "pilot"
    out = _submit([90, 91, 92])
    path = "/data/sm02_pilot.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"pilot result written to {path}")
    for row in out["seeds"]:
        print(json.dumps(row, indent=1))


def _smoke():
    """Tiny local end-to-end of the remote entry point (CPU, 1 seed)."""
    out = remote_run([90], n_train=20, n_eval=8)
    print(json.dumps(out["seeds"][0], indent=1))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "pilot":
        _pilot()
    elif len(sys.argv) > 1 and sys.argv[1] == "smoke":
        _smoke()
    else:
        print("usage: python -m experiments.tests.sm_02_smell_finds_occluded"
              " {pilot|smoke}")
