"""TA.02 — Conditioned taste aversion: learning from ONE exposure.

HYPOTHESIS (registry, unchanged). After exactly ONE ingestion of the toxic
plant followed by delayed illness, Jack avoids that plant on the next
encounter above a pre-registered rate, and the aversion PERSISTS ACROSS A
DEATH via the diary('s persistence boundary — see THE DEATH GATE below).

WHAT THIS WOULD BE IF IT PASSES. `FROZEN_VS_PLASTIC.md` §8.4 searched the RL
and ALife literatures and found no agent implementing a taste-specific
one-trial associator with a long eligibility window and latent inhibition.
The substrate is `experiments/aversion.py` (routed windows, content
addressing, extinction by re-exposure only); the fixture under it is TA.01's
(visually-identical twins, declared dose-response, DELAY_S derived from rat
CTA as a fraction of the starvation horizon). This spec is the BEHAVIOURAL
claim on top: one exposure, next-encounter avoidance, death-crossing.

THE RIG, and what it deliberately is not. An event-driven taste-decision
session at the world's own metabolism: encounters present one plant
individual; the agent licks (the taste percept precedes ingestion), a FIXED
pre-registered decision rule maps aversion to consume/reject, consumption
schedules the toxin exactly as `plants.Toxin` declares. There is NO
locomotion in this rig — T2.01 is unresolved, and a claim about taste must
not be hostage to walking. Every constant of the world is imported live
(`plants.DELAY_S`, `plants.Q_FIRST`, `drives.BASAL_B`, `drives.NU_FLOORFOOD`);
the constants declared HERE are rig geometry (encounter spacing, probe
counts) and the decision threshold, all pre-registered below.

    ENC spacing 120 s > TASTE_MAX_S (100 s): each acquisition's eligibility
    trace resolves before the next encounter, so Kwok-Boakes overshadowing —
    correct behaviour in the wild — is excluded from the APPARATUS by
    spacing, not by code that fights the mechanism.

PROBES ARE DECISIONS, NOT MEALS (Bernstein 1978's choice test; extinction
test trials in the CTA literature). A probe presentation records the
consume/reject decision and has no consequence — because a consumed toxic
probe would poison, re-teach, and partially recover later probes, turning a
one-trial claim into a several-trial one mid-measurement. The lick of a probe
is not observed into the trace (dose 0 writes nothing and extinguishes
nothing, so this is exact, not approximate).

THE GARCIA 2x2, exactly. Cue probes present AV_CUE with WATER (the zero
taste — kernel to any stored taste < 1e-5), because 1966 tested
"bright-noisy water" against "saccharin water". A cue probe that carried a
taste the agent had eaten would be rejected FOR THE TASTE, and the control
would fail for the wrong reason (caught in this spec's design pass, recorded
here so nobody 'simplifies' it back).

THE DEATH GATE. XL.00 certifies the death/respawn machinery and that the
DIARY crosses; this spec gates that the AVERSION VALUES cross the same
boundary with the same contract (`FastPath.to_jsonable`: values serialise,
eligibility traces are working memory and die with the body — §8.4: "stored
in the diary ... AND as an aversion value ... so it crosses death like every
other diary entry"). Death itself is the world's starvation arithmetic:
t_death = (e0 + eaten) / BASAL_B, the same 1/b XL.00 certified against two
charges. The session declares its boundary at e = 0, drives' own clip floor.

THE STANDARD-RL NULL (the GPU half). A DQN — the standard discounted TD
learner: gamma 0.99, replay, target net, epsilon-greedy — lives in the SAME
world stepped at the world's decision length (0.2 s, TA.01's DECISION_S
precedent), sees the SAME lick (the taste is in its observation; what it
lacks is the fast path, not the information), and is given the SAME single
exposure: life 0 is the claim arm's exact timeline, the acquisition
ingestion is scripted (matched experience — a null that happened to reject
the acquisition would have refused the trial, not failed it), the agent
trains on that life's own experience to the probe point, and is probed
greedily. With gamma = 0.99 the discount across DELAY_S = 150 decisions is
0.99^150 = 0.22, diluted across ~800 no-op transitions per replay sweep —
the credit does not arrive in one life. The null must show NO one-trial
discrimination. Its learning gate (VOID, not FAIL, if missed): the same
learner over N_LONG lives must at least learn to eat (immediate reward) —
a null that cannot learn anything cannot refute anything (T2.02's law).
The long run also reports whether toxic discrimination EVER emerges and
after how many exposures — a diagnostic, deliberately ungated: §8.4's claim
is about ONE trial, not about the asymptote.

THE ARMS (fast path unless said otherwise):
  claim      one toxic ingestion at t=10, illness per the declared curve,
             probe1 (20 toxic + 20 safe fresh individuals) in the same life,
             death by starvation, probe2 across the boundary.
  swap       control (a), MUST NOT AVERT THE CUE: bright-noisy-safe meal,
             illness administered at DELAY_S (LiCl analogue, magnitude
             matched to the toxic first dose). Cue probes must be consumed.
             The taste half (safe taste IS averted by illness) is reported
             as the 1966 positive half, not gated.
  shuffled   control (b), MUST FAIL: the acquisition writes a scrambled
             vector into the trace (sensor swap). The scramble is rejection-
             sampled >= 0.6 from BOTH type means — a "random" vector that
             lands on the poison is not a shuffle, it is the answer with
             noise on it (same rule as excluding the identity permutation).
  placebo    control (c), MUST FAIL: the acting channel carries matched-
             statistics noise (TA.01's placebo convention). Matched noise
             draws cluster (E[L2] ~ 0.18, same as same-type taste pairs), so
             an illness produces BLANKET rejection — the honest failure is
             NO DISCRIMINATION, and that is the gate. The blanket rejection
             itself is reported: a noise tongue starves you.
  shock      control (d), MUST PASS: bright-noisy-safe meal, shock at
             +0.4 s (< EXTERO_MAX_S). Cue probes must be REJECTED and the
             eaten taste must NOT be (the Garcia-Koelling double
             dissociation, both halves, gated).
  naive      the base rate: a fresh store must consume everything (rig
             tripwire — VOID if the decision rule itself is broken).
  rl         the standard-RL null above (one-trial gate + learning gate).

PRE-REGISTERED GATES — frozen from the 2026-08-19 pilot (seeds 90-95 for the
arms, 4000 stored-draw simulations for the aggregation, all disjoint from
the registered 0-2; pilot order per the SH.01/XL.01 lessons: must-PASS
control first, then tripwires and distributions, then every gated control,
claim arms last).

THE AGGREGATION WAS PRICED, AND IT CHANGED THE DESIGN — the XL.01 lesson
applied at design time instead of paid for afterwards. False alarms inside a
seed are CORRELATED through the single stored acquisition vector: a toxic
individual whose taste noise leans toward the safe mean smears onto every
safe probe of that seed at once, so per-seed rate gates at 0.90 have a
measured 1-5% per-seed false-fail (pilot seed 93 scored safe-consume 0.75
from exactly this lottery; 4000-draw simulation: P(per-seed safe < 0.9) =
0.05-0.12 across candidate thresholds). More probes cannot fix a bad stored
draw. The DISCRIMINATION statistic (avoid_toxic - avoid_safe) is robust to
the lottery (both rates move together), and pooling across seeds (equal K,
so the seed mean IS the pooled rate — what run_spec's aggregation already
hands _check) buys the rest:

  ACT_THRESH   0.002. At this threshold, 4000-draw simulation:
               per-seed avoid_toxic  P(< 0.9) = 2.7e-3, P(< 0.8) ~ 1e-3
               pooled-3 avoid_toxic  P(< 0.9) = 1.5e-4
               pooled-3 disc         P(< 0.6) = 2.5e-4
               Shock-cue aversion is DI_FIRST exactly (101x threshold);
               water vs any stored taste < 1e-5 of it.
  SEED_AVOID_FLOOR  0.80 per seed per phase (>= 16/20 everywhere — the
               claim must hold in every world; the floor is set where the
               lottery's 1-in-1000 tail sits, not where the mean sits).
  POOL_AVOID_GATE   0.90 pooled per phase (the headline strength).
  POOL_DISC_GATE    0.60 pooled per phase. The discriminating check the
               XL.01 lesson demands: the placebo arm's disc pools to 0.0
               (measured, every pilot seed), so the pooled form still fails
               the failure it exists to catch, by the full 0.6 margin.
  CTRL_CEIL    0.10 (<= 2/20) — swap cue, shuffled, shock taste: the
               mechanisms are exact zeros in the pilot (routing, content
               addressing); the ceiling absorbs nothing and catches leaks.
  DISC_CEIL    0.25 — placebo |avoid_toxic - avoid_safe| (pilot: 0.0).
  RL_DISC_CEIL 0.40 — the null's one-trial discrimination must sit under
               the claim's pooled floor by 1.5x (greedy argmax at one life
               of experience).
  RL_EAT_GATE  0.60 — the learning gate: long-trained safe consumption
               (smoke at 8 lives: 1.0; the gate sits 1.6x under that and
               above cointoss).
Total pre-registered false-fail budget across every claim gate and phase:
< 1% per run — priced, not hoped.

COVERS: taste (claim).
"""

from __future__ import annotations

import json

import numpy as np

from .. import aversion, drives, plants
from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..gpu import build_job, submit

# The claim is about the fast path in the world's chemistry and metabolism —
# all three hash into the certificate.
IMPL_DEPS = ["experiments/aversion.py", "experiments/plants.py",
             "experiments/drives.py"]

SEEDS = [0, 1, 2]

# ── rig geometry, pre-registered ─────────────────────────────────────────
DECISION_S = 0.2               # the world's decision length (TA.01 precedent)
ACQ_T = 10.0                   # the one exposure
ENC_S = 120.0                  # encounter spacing; > TASTE_MAX_S by design
PROBE1_T = 170.0               # illness ([40,60]) over, trace (<=110) resolved
K_PROBE = 20                   # presentations per category per phase
PROBE_GAP_S = 1.0              # pure queries; spacing is bookkeeping only
E0 = 0.5                       # starting charge: death at ~307 s, after probe1
SHOCK_DELAY_S = 0.4            # < aversion.EXTERO_MAX_S = 0.8
AV_CUE = (0.9, 0.1, 0.8, 0.3)  # the bright-noisy signature; a literal, like
                               # TA.01's canary — it must not move with a rng
SHUFFLE_MIN_DIST = 0.6         # rejection floor for the scrambled vector

# One severity for the illness insult and the shock: the toxic first dose's
# own declared total. Matched by construction, not tuned.
DI_FIRST = plants.illness_total(plants.Q_FIRST, plants.TOXIC.potency)

# ── the gates (see docstring for the pilot that froze them) ──────────────
ACT_THRESH = 0.002
SEED_AVOID_FLOOR = 0.80        # per seed, per phase: holds in every world
POOL_AVOID_GATE = 0.90         # pooled (= seed mean at equal K), per phase
POOL_DISC_GATE = 0.60          # pooled avoid_toxic - avoid_safe, per phase
SHOCK_CUE_GATE = 0.90          # must-PASS side; mechanism is exact (101x)
CTRL_CEIL = 0.10
DISC_CEIL = 0.25
RL_DISC_CEIL = 0.40
RL_EAT_GATE = 0.60

# ── the RL null's constants, all standard, all declared ──────────────────
RL_GAMMA = 0.99
RL_LR = 1e-3
RL_BATCH = 64
RL_BUFFER = 100_000
RL_TARGET_SYNC = 1000
RL_EPS_END = 0.05
RL_HIDDEN = 64
N_LONG_LIVES = 150             # the learning-gate / diagnostic run
RL_EPS_DECAY_LIVES = 50        # epsilon 1.0 -> RL_EPS_END across these


# ── the shared decision rule (fixed, identical in every arm) ─────────────
def _decide(fp: aversion.FastPath, lick, av, channel: str = "taste") -> bool:
    """True = consume. Reject when any acquired aversion clears ACT_THRESH."""
    a_taste = fp.aversion(channel, lick)
    a_cue = fp.aversion("extero", np.asarray(av, float))
    return max(a_taste, a_cue) < ACT_THRESH


def _consume(fp: aversion.FastPath, tox: plants.Toxin, t: float, plant,
             lick, av, channel: str = "taste") -> float:
    """One real ingestion: trace both channels, schedule the toxin, eat."""
    fp.observe(channel, lick, t, plants.Q_FIRST)
    fp.observe("extero", np.asarray(av, float), t, 1.0)
    tox.ingest(t, plant, plants.Q_FIRST)
    return plants.Q_FIRST * drives.NU_FLOORFOOD


def _probe(fp: aversion.FastPath, rng, channel: str = "taste") -> dict:
    """K_PROBE fresh toxic + K_PROBE fresh safe presentations, decisions only.

    Returns rates plus the aversion headroom (min over toxic probes, max over
    safe probes) so the record carries the margin, not only the verdict.
    """
    a_tox, a_safe = [], []
    for _ in range(K_PROBE):
        lick_t = (plants.TOXIC.taste(rng) if channel == "taste"
                  else _placebo_lick(rng))
        lick_s = (plants.SAFE.taste(rng) if channel == "taste"
                  else _placebo_lick(rng))
        a_tox.append(fp.aversion(channel, lick_t))
        a_safe.append(fp.aversion(channel, lick_s))
    avoid_tox = float(np.mean([a >= ACT_THRESH for a in a_tox]))
    consume_safe = float(np.mean([a < ACT_THRESH for a in a_safe]))
    return {"avoid_toxic": avoid_tox, "consume_safe": consume_safe,
            "min_toxic_aversion": float(min(a_tox)),
            "max_safe_aversion": float(max(a_safe))}


def _cue_probe(fp: aversion.FastPath, rng) -> float:
    """K_PROBE bright-noisy-WATER presentations; returns the avoid rate."""
    water = np.zeros(plants.TASTE_DIM)
    return float(np.mean([not _decide(fp, water, AV_CUE)
                          for _ in range(K_PROBE)]))


def _taste_probe(fp: aversion.FastPath, rng, plant) -> float:
    """K_PROBE plain presentations of `plant`'s taste; returns the avoid rate."""
    return float(np.mean([not _decide(fp, plant.taste(rng), np.zeros(4))
                          for _ in range(K_PROBE)]))


def _placebo_lick(rng) -> np.ndarray:
    """TA.01's placebo convention: matched mean, matched sigma, zero signal."""
    mu = np.mean([t.taste_mu for t in plants.TYPES], axis=0)
    return np.clip(mu + rng.normal(0.0, plants.TASTE_SIGMA, plants.TASTE_DIM),
                   0.0, 1.0)


def _shuffle_vec(rng) -> np.ndarray:
    """A scrambled taste, rejection-sampled away from BOTH type means."""
    for _ in range(1000):
        r = rng.rand(plants.TASTE_DIM)
        if all(np.linalg.norm(r - np.asarray(p.taste_mu)) >= SHUFFLE_MIN_DIST
               for p in plants.TYPES):
            return r
    raise RuntimeError("no admissible shuffle vector in 1000 draws")


# ── the fast-path arms ────────────────────────────────────────────────────
def _arm_claim(seed: int) -> dict:
    """One toxic meal, illness, probe1, starvation death, probe2 across it."""
    rng = np.random.RandomState(seed * 9209 + 11)
    fp, tox = aversion.FastPath(), plants.Toxin()

    # base rate first: a fresh store must consume everything it is shown
    naive = _probe(fp, rng)
    base_ok = (naive["avoid_toxic"] == 0.0 and naive["consume_safe"] == 1.0)

    lick0 = plants.TOXIC.taste(rng)
    consumed = _decide(fp, lick0, np.zeros(4))
    eaten = _consume(fp, tox, ACQ_T, plants.TOXIC, lick0, np.zeros(4)) \
        if consumed else 0.0
    # the illness, felt at onset, bound once with the episode's own total
    written = fp.insult("illness", ACQ_T + plants.DELAY_S, DI_FIRST) \
        if consumed else 0.0
    fp.resolve(PROBE1_T)

    p1 = _probe(fp, rng)

    # the death boundary: starvation arithmetic, then the crossing contract
    t_death = (E0 + eaten) / drives.BASAL_B
    probe1_end = PROBE1_T + 2.0 * K_PROBE * PROBE_GAP_S
    fp2 = aversion.FastPath.from_jsonable(json.loads(json.dumps(
        fp.to_jsonable())))
    traces_dead = all(len(c.trace) == 0 for c in fp2.channels.values())
    store_crossed = any(len(c.store) > 0 for c in fp2.channels.values())
    p2 = _probe(fp2, rng)

    return {
        "base_rate_ok": float(base_ok),
        "acq_consumed": float(consumed),
        "acq_written": float(written),
        "one_trial_avoidance_rate": p1["avoid_toxic"],
        "safe_consume_rate": p1["consume_safe"],
        "min_toxic_aversion": p1["min_toxic_aversion"],
        "max_safe_aversion": p1["max_safe_aversion"],
        "death_after_probe1": float(t_death > probe1_end),
        "t_death_s": float(t_death),
        "traces_dead_after_death": float(traces_dead),
        "store_crossed_death": float(store_crossed),
        "postdeath_avoidance_rate": p2["avoid_toxic"],
        "postdeath_safe_consume": p2["consume_safe"],
    }


def _arm_swap(seed: int) -> dict:
    """(a) bright-noisy-safe meal + administered illness: cue must survive."""
    rng = np.random.RandomState(seed * 7127 + 5)
    fp, tox = aversion.FastPath(), plants.Toxin()
    lick0 = plants.SAFE.taste(rng)
    consumed = _decide(fp, lick0, AV_CUE)
    if consumed:
        _consume(fp, tox, ACQ_T, plants.SAFE, lick0, AV_CUE)
        fp.insult("illness", ACQ_T + plants.DELAY_S, DI_FIRST)
    fp.resolve(PROBE1_T)
    return {
        "swap_consumed": float(consumed),
        "swap_cue_avoid": _cue_probe(fp, rng),
        # the 1966 positive half — sick rats DID avert the taste (reported,
        # not gated: it is the claim arm's mechanism on another taste)
        "swap_taste_avoid": _taste_probe(fp, rng, plants.SAFE),
    }


def _arm_shuffled(seed: int) -> dict:
    """(b) illness bound to a scrambled vector: the poison must survive."""
    rng = np.random.RandomState(seed * 6949 + 3)
    fp, tox = aversion.FastPath(), plants.Toxin()
    scr = _shuffle_vec(rng)
    consumed = _decide(fp, scr, np.zeros(4))
    if consumed:
        _consume(fp, tox, ACQ_T, plants.TOXIC, scr, np.zeros(4))
        fp.insult("illness", ACQ_T + plants.DELAY_S, DI_FIRST)
    fp.resolve(PROBE1_T)
    return {
        "shuffled_consumed": float(consumed),
        "shuffled_toxic_avoid": _taste_probe(fp, rng, plants.TOXIC),
        "shuffled_safe_avoid": _taste_probe(fp, rng, plants.SAFE),
    }


def _arm_placebo(seed: int) -> dict:
    """(c) the acting channel is matched noise: no discrimination possible."""
    rng = np.random.RandomState(seed * 5881 + 7)
    fp, tox = aversion.FastPath(), plants.Toxin()
    fp.add_channel("placebo", dim=plants.TASTE_DIM, window="taste",
                   max_s=aversion.TASTE_MAX_S, route_from=("illness",))
    lick0 = _placebo_lick(rng)
    consumed = _decide(fp, lick0, np.zeros(4), channel="placebo")
    if consumed:
        _consume(fp, tox, ACQ_T, plants.TOXIC, lick0, np.zeros(4),
                 channel="placebo")
        fp.insult("illness", ACQ_T + plants.DELAY_S, DI_FIRST)
    fp.resolve(PROBE1_T)
    p = _probe(fp, rng, channel="placebo")
    # in the placebo channel "toxic" and "safe" presentations are BOTH fresh
    # noise; the two rates estimate the same quantity, which is the point
    return {
        "placebo_consumed": float(consumed),
        "placebo_toxic_avoid": p["avoid_toxic"],
        "placebo_safe_avoid": 1.0 - p["consume_safe"],
        "placebo_disc": abs(p["avoid_toxic"] - (1.0 - p["consume_safe"])),
        "placebo_blanket": p["avoid_toxic"],   # the cost of a noise tongue
    }


def _arm_shock(seed: int) -> dict:
    """(d) MUST PASS: shock -> cue averted, taste NOT (both 1966 halves)."""
    rng = np.random.RandomState(seed * 4801 + 13)
    fp, tox = aversion.FastPath(), plants.Toxin()
    lick0 = plants.SAFE.taste(rng)
    consumed = _decide(fp, lick0, AV_CUE)
    if consumed:
        _consume(fp, tox, ACQ_T, plants.SAFE, lick0, AV_CUE)
        fp.insult("shock", ACQ_T + SHOCK_DELAY_S, DI_FIRST)
    fp.resolve(PROBE1_T)
    return {
        "shock_consumed": float(consumed),
        "shock_cue_avoid": _cue_probe(fp, rng),
        "shock_taste_avoid": _taste_probe(fp, rng, plants.SAFE),
    }


# ── the standard-RL null ─────────────────────────────────────────────────
class _RLWorld:
    """The same session, stepped at DECISION_S, consequences as reward.

    obs = [lick(5), av(4), sick(1), e(1)]; lick/av are zero except at an
    encounter step. action 1 = consume (meaningful only at encounters).
    reward = food energy gained - integrity lost this step: the world's own
    consequences, expressed in the only language standard RL speaks.
    """

    OBS_DIM = plants.TASTE_DIM + 4 + 2
    ENC_TYPES_P_TOXIC = 0.5

    def __init__(self, seed: int, schedule: str = "natural"):
        self.rng = np.random.RandomState(seed * 3557 + 29)
        self.schedule = schedule
        self.reset()

    def reset(self):
        self.t, self.e = 0.0, E0
        self.tox = plants.Toxin()
        self.k_enc = 0
        self.done = False
        self._roll_enc_type()
        return self._obs()

    def _enc_time(self, k: int) -> float:
        return ACQ_T + k * ENC_S

    def _at_encounter(self) -> bool:
        # the encounter occupies exactly one decision step
        te = self._enc_time(self.k_enc)
        if self.schedule == "one_trial" and self.k_enc > 0:
            return False
        return te <= self.t < te + DECISION_S

    def _obs(self) -> np.ndarray:
        o = np.zeros(self.OBS_DIM, dtype=np.float32)
        if self._at_encounter():
            plant = self._current_plant()
            o[:plants.TASTE_DIM] = plant.taste(self.rng)
        o[-2] = 1.0 if self.tox.rate(self.t) > 0.0 else 0.0
        o[-1] = self.e
        return o

    def _current_plant(self):
        if self.schedule == "one_trial":
            return plants.TOXIC
        return plants.TOXIC if self._enc_type_toxic else plants.SAFE

    def step(self, action: int):
        reward = 0.0
        if self._at_encounter():
            if action == 1:
                plant = self._current_plant()
                self.tox.ingest(self.t, plant, plants.Q_FIRST)
                gain = plants.Q_FIRST * drives.NU_FLOORFOOD
                self.e = min(1.0, self.e + gain)
                reward += gain
            self.k_enc += 1
            self._roll_enc_type()
        di = self.tox.rate(self.t) * DECISION_S
        reward -= di
        self.e -= drives.BASAL_B * DECISION_S
        self.t += DECISION_S
        if self.e <= 0.0:
            self.done = True
        return self._obs(), float(reward), self.done

    def _roll_enc_type(self):
        self._enc_type_toxic = bool(self.rng.rand() < self.ENC_TYPES_P_TOXIC)


def _rl_null(seed: int, n_long_lives: int = N_LONG_LIVES) -> dict:
    """The one-trial gate and the learning gate, one DQN each."""
    import torch
    import torch.nn as nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def make_net():
        return nn.Sequential(
            nn.Linear(_RLWorld.OBS_DIM, RL_HIDDEN), nn.ReLU(),
            nn.Linear(RL_HIDDEN, RL_HIDDEN), nn.ReLU(),
            nn.Linear(RL_HIDDEN, 2)).to(dev)

    def greedy(net, obs) -> int:
        with torch.no_grad():
            q = net(torch.tensor(obs, dtype=torch.float32,
                                 device=dev).unsqueeze(0))
        return int(q.argmax(1).item())

    def probe_rates(net, rng, e_now: float) -> tuple:
        """Greedy decisions on K_PROBE fresh toxic and safe encounter obs."""
        def enc_obs(plant):
            o = np.zeros(_RLWorld.OBS_DIM, dtype=np.float32)
            o[:plants.TASTE_DIM] = plant.taste(rng)
            o[-1] = e_now
            return o
        avoid_t = float(np.mean([greedy(net, enc_obs(plants.TOXIC)) == 0
                                 for _ in range(K_PROBE)]))
        avoid_s = float(np.mean([greedy(net, enc_obs(plants.SAFE)) == 0
                                 for _ in range(K_PROBE)]))
        return avoid_t, avoid_s

    def train(net, target, opt, buf, rng, gstep: int) -> int:
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
        return gstep

    rng = np.random.RandomState(seed * 2503 + 41)

    # ── the one-trial arm: the claim arm's life 0, matched exactly ──────
    torch.manual_seed(seed * 17 + 1)
    net, target = make_net(), make_net()
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=RL_LR)
    buf, gstep = [], 0
    w = _RLWorld(seed, schedule="one_trial")
    obs = w.reset()
    n_steps_one = int(PROBE1_T / DECISION_S)
    ate_one = False
    for _ in range(n_steps_one):
        at_enc = w._at_encounter()
        # scripted acquisition: matched experience, not a choice
        act = 1 if at_enc else 0
        if at_enc:
            ate_one = True
        obs2, r, done = w.step(act)
        buf.append((obs, act, r, obs2, done))
        gstep = train(net, target, opt, buf, rng, gstep)
        obs = obs2
        if done:
            break
    rl_avoid_t1, rl_avoid_s1 = probe_rates(net, rng, w.e)

    # ── the long arm: the learning gate + the asymptote diagnostic ──────
    torch.manual_seed(seed * 17 + 2)
    net, target = make_net(), make_net()
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=RL_LR)
    buf, gstep = [], 0
    toxic_meals = 0
    curve = []
    for life in range(n_long_lives):
        eps = max(RL_EPS_END,
                  1.0 - (1.0 - RL_EPS_END) * life / RL_EPS_DECAY_LIVES)
        w = _RLWorld(seed * 1009 + life, schedule="natural")
        obs = w.reset()
        while not w.done:
            if rng.rand() < eps:
                act = int(rng.randint(2))
            else:
                act = greedy(net, obs)
            if w._at_encounter() and act == 1 and w._enc_type_toxic:
                toxic_meals += 1
            obs2, r, done = w.step(act)
            if len(buf) >= RL_BUFFER:
                buf.pop(0)
            buf.append((obs, act, r, obs2, done))
            gstep = train(net, target, opt, buf, rng, gstep)
            obs = obs2
        if (life + 1) % max(1, n_long_lives // 6) == 0:
            at, asf = probe_rates(net, rng, 0.3)
            curve.append({"life": life + 1, "avoid_toxic": at,
                          "avoid_safe": asf,
                          "toxic_meals": toxic_meals})
    rl_avoid_tL, rl_avoid_sL = probe_rates(net, rng, 0.3)

    return {
        "rl_ate_acquisition": float(ate_one),
        "rl_avoid_toxic_one": rl_avoid_t1,
        "rl_avoid_safe_one": rl_avoid_s1,
        "rl_disc_one": rl_avoid_t1 - rl_avoid_s1,
        "rl_avoid_toxic_long": rl_avoid_tL,
        "rl_avoid_safe_long": rl_avoid_sL,
        "rl_disc_long": rl_avoid_tL - rl_avoid_sL,
        "rl_safe_consume_long": 1.0 - rl_avoid_sL,
        "rl_toxic_meals_long": float(toxic_meals),
        "rl_curve": curve,
        "rl_finite": float(np.isfinite(
            [rl_avoid_t1, rl_avoid_s1, rl_avoid_tL, rl_avoid_sL]).all()),
    }


# ── remote entry point (also the local smoke, scaled down) ───────────────
def remote_run(seeds: list, n_long_lives: int = N_LONG_LIVES) -> dict:
    out = {"seeds": [], "gpu": "cpu"}
    try:
        import torch
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    for seed in seeds:
        row = {"seed": int(seed)}
        row.update(_arm_claim(seed))
        row.update(_arm_swap(seed))
        row.update(_arm_shuffled(seed))
        row.update(_arm_placebo(seed))
        row.update(_arm_shock(seed))
        row.update(_rl_null(seed, n_long_lives=n_long_lives))
        out["seeds"].append(row)
    return out


JOB = r'''
import json, os
from experiments.tests.ta_02_one_trial_aversion import remote_run
out = remote_run(__SEEDS__)
json.dump(out, open(os.path.join(os.environ["JACK_OUT"], "ta202.json"), "w"),
          indent=1)
print("DONE", flush=True)
'''

_CACHE: dict = {}


def _submit(seeds: list) -> dict:
    body = JOB.replace("__SEEDS__", repr(list(seeds)))
    job = build_job(body)
    r = submit(job, prefer="kaggle", est_hours=0.75, timeout_s=5400,
               fetch=["ta202.json"])
    if not r.ok:
        raise RuntimeError(f"GPU submission failed: {r.message}")
    data = json.loads(r.artifacts["ta202.json"])
    data["backend"] = r.backend
    return data


def _row(seed: int) -> dict:
    if not _CACHE:
        _CACHE.update(_submit(SEEDS))
    for row in _CACHE["seeds"]:
        if row["seed"] == seed:
            return row
    raise KeyError(f"seed {seed} missing from remote result")


def _experiment(seed: int) -> dict:
    r = _row(seed)
    m = {k: r[k] for k in (
        "base_rate_ok", "acq_consumed", "acq_written",
        "one_trial_avoidance_rate", "safe_consume_rate",
        "min_toxic_aversion", "max_safe_aversion",
        "death_after_probe1", "t_death_s",
        "traces_dead_after_death", "store_crossed_death",
        "postdeath_avoidance_rate", "postdeath_safe_consume",
        "rl_ate_acquisition", "rl_avoid_toxic_one", "rl_avoid_safe_one",
        "rl_disc_one", "rl_avoid_toxic_long", "rl_avoid_safe_long",
        "rl_disc_long", "rl_safe_consume_long", "rl_toxic_meals_long",
        "rl_finite")}
    m["gpu"] = _CACHE.get("gpu", "?")
    m["backend"] = _CACHE.get("backend", "?")
    # the discrimination statistic, per seed; its seed MEAN is the pooled
    # value (equal K per seed), which is exactly what run_spec hands _check
    m["one_trial_disc"] = (m["one_trial_avoidance_rate"]
                           - (1.0 - m["safe_consume_rate"]))
    m["postdeath_disc"] = (m["postdeath_avoidance_rate"]
                           - (1.0 - m["postdeath_safe_consume"]))
    # rig validity and the per-seed floors, inside the seed (XL.00's rule:
    # the mean of these booleans is 1.0 iff every seed cleared — run_spec
    # averages, law 2 does not)
    m["rig_ok"] = float(
        m["base_rate_ok"] == 1.0 and m["acq_consumed"] == 1.0
        and m["acq_written"] > 0.0 and m["death_after_probe1"] == 1.0
        and m["traces_dead_after_death"] == 1.0
        and m["store_crossed_death"] == 1.0
        and m["rl_ate_acquisition"] == 1.0 and m["rl_finite"] == 1.0)
    m["seed_floor_ok"] = float(
        m["one_trial_avoidance_rate"] >= SEED_AVOID_FLOOR
        and m["postdeath_avoidance_rate"] >= SEED_AVOID_FLOOR
        and m["rl_disc_one"] <= RL_DISC_CEIL)
    m["rl_learner_alive"] = float(m["rl_safe_consume_long"] >= RL_EAT_GATE)
    return m


def _control(seed: int) -> dict:
    r = _row(seed)
    c = {k: r[k] for k in (
        "swap_consumed", "swap_cue_avoid", "swap_taste_avoid",
        "shuffled_consumed", "shuffled_toxic_avoid", "shuffled_safe_avoid",
        "placebo_consumed", "placebo_toxic_avoid", "placebo_safe_avoid",
        "placebo_disc", "placebo_blanket",
        "shock_consumed", "shock_cue_avoid", "shock_taste_avoid")}
    c["c_acquired_ok"] = float(
        c["swap_consumed"] == 1.0 and c["shuffled_consumed"] == 1.0
        and c["placebo_consumed"] == 1.0 and c["shock_consumed"] == 1.0)
    c["c_swap_ok"] = float(c["swap_cue_avoid"] <= CTRL_CEIL)
    c["c_shuffled_ok"] = float(c["shuffled_toxic_avoid"] <= CTRL_CEIL
                               and c["shuffled_safe_avoid"] <= CTRL_CEIL)
    c["c_placebo_ok"] = float(c["placebo_disc"] <= DISC_CEIL)
    c["c_shock_ok"] = float(c["shock_cue_avoid"] >= SHOCK_CUE_GATE
                            and c["shock_taste_avoid"] <= CTRL_CEIL)
    return c


def _check(m: dict, c: dict):
    # ── the rig: an invalid run is not evidence ─────────────────────────
    if m.get("rig_ok", 0.0) != 1.0 or c.get("c_acquired_ok", 0.0) != 1.0:
        return Status.VOID
    # the null's learning gate: a learner that never learned to EAT cannot
    # refute one-trial learning (T2.02's law — VOID, not a free pass)
    if m.get("rl_learner_alive", 0.0) != 1.0:
        return Status.VOID

    # ── the controls, each on its declared side, EVERY seed ────────────
    # (a)-(c) acquiring a selective aversion would mean the mechanism is a
    # generic one-shot memoriser — that FALSIFIES the fast path (registry
    # `kills`), so a cleared must-fail control is FAIL here, not VOID: the
    # apparatus measured exactly what it claims to measure and the claim lost.
    if c.get("c_swap_ok", 0.0) != 1.0:
        return False
    if c.get("c_shuffled_ok", 0.0) != 1.0:
        return False
    if c.get("c_placebo_ok", 0.0) != 1.0:
        return False
    # (d) must PASS: if nothing one-shot works on the extero path, the
    # routing design is refuted and (a)'s null proves nothing
    if c.get("c_shock_ok", 0.0) != 1.0:
        return False

    # ── the claim: per-seed floors AND the pooled gates ─────────────────
    # `m` values are seed means; at equal K per seed a mean of rates IS the
    # pooled rate, so the pooled gates read them directly (the aggregation
    # the pilot priced — see the docstring's ACT_THRESH block).
    return bool(m.get("seed_floor_ok", 0.0) == 1.0
                and m.get("one_trial_avoidance_rate", 0.0) >= POOL_AVOID_GATE
                and m.get("postdeath_avoidance_rate", 0.0) >= POOL_AVOID_GATE
                and m.get("one_trial_disc", 0.0) >= POOL_DISC_GATE
                and m.get("postdeath_disc", 0.0) >= POOL_DISC_GATE)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["TA.02"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


# ── pilot / smoke CLI ────────────────────────────────────────────────────
def _pilot():
    """Pilot order per the SH.01/XL.01 lessons: must-PASS control first,
    then tripwires and distributions (aggregation priced), then every gated
    control, claim arms LAST. Disjoint seeds 90-95."""
    seeds = range(90, 96)

    print("== (d) shock — the must-PASS control, FIRST ==")
    for s in seeds:
        print(f"  seed {s}: {_arm_shock(s)}")

    print("== rig tripwires: naive base rate ==")
    for s in seeds:
        fp = aversion.FastPath()
        p = _probe(fp, np.random.RandomState(s))
        print(f"  seed {s}: naive avoid_toxic={p['avoid_toxic']}, "
              f"consume_safe={p['consume_safe']}")

    print("== headroom: aversion distributions (200k draws) ==")
    rng = np.random.RandomState(4242)
    n = 200_000
    stored = np.stack([plants.TOXIC.taste(rng) for _ in range(n // 100)])
    a_tox, a_safe = [], []
    for i in range(n // 100):
        for _ in range(2):
            a_tox.append(DI_FIRST * plants.Q_FIRST
                         * aversion.kernel(stored[i], plants.TOXIC.taste(rng)))
            a_safe.append(DI_FIRST * plants.Q_FIRST
                          * aversion.kernel(stored[i], plants.SAFE.taste(rng)))
    a_tox, a_safe = np.asarray(a_tox), np.asarray(a_safe)
    print(f"  toxic-probe aversion:  p5={np.percentile(a_tox, 5):.5f} "
          f"p1={np.percentile(a_tox, 1):.5f} min={a_tox.min():.5f}")
    print(f"  safe-probe aversion:   p95={np.percentile(a_safe, 95):.6f} "
          f"p99={np.percentile(a_safe, 99):.6f} max={a_safe.max():.6f}")
    for th in (0.002, 0.003, 0.004, 0.005):
        miss = float((a_tox < th).mean())
        fa = float((a_safe >= th).mean())
        # per-seed gate failure at K_PROBE with >= 3 misses (rate < 0.9)
        from math import comb
        pfail_m = sum(comb(K_PROBE, k) * miss**k * (1 - miss)**(K_PROBE - k)
                      for k in range(3, K_PROBE + 1))
        pfail_f = sum(comb(K_PROBE, k) * fa**k * (1 - fa)**(K_PROBE - k)
                      for k in range(3, K_PROBE + 1))
        print(f"  theta={th}: miss={miss:.2e} fa={fa:.2e} "
              f"P(seed fails avoid)={pfail_m:.2e} "
              f"P(seed fails safe)={pfail_f:.2e}")
    print(f"  shock cue aversion = DI_FIRST = {DI_FIRST:.4f} "
          f"({DI_FIRST / ACT_THRESH:.0f}x threshold)")

    print("== (a) swap ==")
    for s in seeds:
        print(f"  seed {s}: {_arm_swap(s)}")
    print("== (b) shuffled ==")
    for s in seeds:
        print(f"  seed {s}: {_arm_shuffled(s)}")
    print("== (c) placebo ==")
    for s in seeds:
        print(f"  seed {s}: {_arm_placebo(s)}")

    print("== claim arm, LAST ==")
    for s in seeds:
        r = _arm_claim(s)
        print(f"  seed {s}: avoid1={r['one_trial_avoidance_rate']} "
              f"safe1={r['safe_consume_rate']} "
              f"avoid2={r['postdeath_avoidance_rate']} "
              f"safe2={r['postdeath_safe_consume']} "
              f"minTox={r['min_toxic_aversion']:.5f} "
              f"maxSafe={r['max_safe_aversion']:.6f} "
              f"death_ok={r['death_after_probe1']}")


def _smoke():
    """Tiny local end-to-end of the remote entry point (CPU, 1 seed)."""
    out = remote_run([90], n_long_lives=8)
    row = out["seeds"][0]
    keep = {k: v for k, v in row.items() if k != "rl_curve"}
    print(json.dumps(keep, indent=1))
    print("curve:", row["rl_curve"])


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "pilot":
        _pilot()
    elif len(sys.argv) > 1 and sys.argv[1] == "smoke":
        _smoke()
    else:
        print("usage: python -m experiments.tests.ta_02_one_trial_aversion "
              "{pilot|smoke}")
