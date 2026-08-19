"""aversion.py — the taste fast path: one exposure, one update, one grudge.

Built for `TA.02` (conditioned taste aversion from ONE exposure). This module
is the ASSOCIATOR that `plants.py` deliberately refuses to carry — plants.py
line: "The associator is TA.02's, and it must be ablatable." It decides
nothing about behaviour: it stores cue traces, binds insults to them, and
answers "how averse am I to this cue". The behavioural loop (encounter,
sample, reject/consume) is the TEST's, exactly as SH.01's episodic loop owns
its world stepping.

WHY A FAST PATH EXISTS AT ALL (FROZEN_VS_PLASTIC.md §8.4, verified sources).
With gamma < 1 and an illness delay of DELAY_S = 30 s (150 decision steps at
0.2 s), the discounted return carries no usable signal back to the eating
decision — and one sample is not a gradient. Biology solved this with a
privileged channel: one taste-illness pairing produces months of aversion
(Garcia, Kimeldorf & Koelling 1955; delay tolerance 1-6 h reliable, Smith &
Roll 1967; Riley, Hempel & Clasen 2018). This is one of the two named,
ablatable fast paths §9.4 budgets for. If TA.02 passes it is, per §8.4's
literature sweep, the first RL-adjacent implementation of a taste-specific
one-trial associator with a long eligibility window and latent inhibition.

THE SELECTIVITY IS TWO MECHANISMS, NOT ONE, AND BOTH ARE DECLARED HERE.
Garcia & Koelling 1966 (read via Domjan 2015, doi:10.46867/ijcp.2015.28.01.08
— the 1966 primary is paywalled; §8.4 records this) found a double
dissociation: sick rats avoided the taste and not the audiovisual cue;
shocked rats avoided the audiovisual cue and not the taste.

  * WINDOWS alone cannot produce that: the taste was still inside its long
    eligibility window when the shock arrived, so a window-only mechanism
    would have shocked rats averse to the taste too. They were not.
  * ROUTING alone (illness->taste, shock->extero, hardcoded) would make the
    cue-consequence swap control pass by construction with no window to
    test, and an hours-delayed illness would bind to a taste eaten seconds
    ago and one eaten a day ago identically. It does not (Smith & Roll 1967:
    the delay gradient is real).

So: each CHANNEL carries an eligibility window scaled to its biology, and
each INSULT KIND is routed to consequence-congruent channels. TA.02's
controls each grip one mechanism: the cue-consequence swap (a) tests
routing; the must-PASS shock control (d) tests that the extero path works
(so (a)'s null is not "nothing works"); the shuffled-taste control (b) tests
content addressing; the placebo channel (c) tests that the capability comes
from the information in the channel, not from the machinery around it —
a placebo channel is WIRED like taste (same routing, same window) and must
still buy nothing, because noise identifies no plant.

TIME SCALING — same derivation as plants.DELAY_S, gated by TA.01 as a
FRACTION of the starvation horizon (1 / drives.BASAL_B = 600 s), so these
track the world if metabolism is recalibrated:

  taste reliable window   1-6 h  of a ~72 h rat starvation horizon -> 8.3%
                          -> TASTE_RELIABLE_S = 50 s  (weight 1.0 inside)
  taste marginal edge     12 h   (Smith & Roll: "only marginally
                          significant") -> 16.7% -> TASTE_MAX_S = 100 s
                          (weight falls linearly 1 -> 0 across 50-100 s)
  extero window           non-taste conditioning fails past ~1 min; as a
                          RATIO to the taste window (1 min : 1-6 h is
                          1/60-1/360) -> 50/60 ~ 0.8 s. EXTERO_MAX_S = 0.8 s
                          = 4 decision steps: spans an immediate cue->shock
                          pairing, cannot span DELAY_S = 30 s. The absolute
                          rescaling (1 min / 72 h * 600 s = 0.14 s) would be
                          below one decision step, i.e. unrepresentable; the
                          ratio form is the one that preserves the CONTRAST
                          the mechanism exists for, and this choice is
                          declared rather than hidden.

CONTENT ADDRESSING. Aversion is keyed on the cue vector, not the plant id —
the store never learns "toxic plant", it learns "this taste". Gaussian
kernel on L2 distance, SIM_SIGMA = 0.15, derived from the fixture it serves:
same-type taste pairs sit at E[L2] ~ 0.19 (TASTE_SIGMA 0.06 x sqrt(2) x
sqrt(5)), the SAFE-TOXIC mean separation is 0.457 (dominated by bitter,
0.10 vs 0.55). k(0.19) = 0.45, k(0.457) = 0.0097 — a 46x same-type vs
cross-type credit ratio, so an aversion to the toxic taste generalises over
toxic individuals and does not smear onto the safe twin. If plants.py's
taste geometry changes, TA.02's IMPL_DEPS goes stale and this derivation is
re-checked, not trusted.

LATENT INHIBITION (De la Casa & Lubow 1995): familiarity weakens acquisition
as a function of amount consumed. Learning multiplier 1 / (1 + LI_RATE *
familiarity), familiarity = kernel-weighted sum of doses previously ingested
uneventfully. A novel taste (familiarity 0) learns at full rate; a staple
eaten safely ten times learns at ~1/4 rate (LI_RATE = 0.3, 1/(1+3) = 0.25).

EXTINCTION IS RE-EXPOSURE, NEVER A CLOCK (Rosenblum et al. 2003: retention
is months, decay is driven by safe re-exposure). Nothing in this module
decays with wall time. When a trace event's window expires with no congruent
insult having bound to it, matching store entries weaken by EXT_RATE *
kernel * dose. `resolve(t)` performs this bookkeeping and the caller must
call it (the caller owns time, exactly as with plants.Toxin).

OVERSHADOWING COMES FREE (Kwok & Boakes): an insult binds to EVERY live
trace event in its routed channels, weighted by w(delay) * dose share — an
agent that ate three novel things before falling ill smears credit across
all three. That is correct behaviour, not a bug, and TA.02 does not gate on
it; it is noted so nobody "fixes" it.

CROSSING DEATH. `to_jsonable()` / `from_jsonable()` round-trip the store
exactly (traces are NOT serialised: an eligibility trace is working memory
and dies with the body; the aversion VALUES are the diary-adjacent organ
that persists). TA.02's death-persistence gate rides XL.00's respawn
machinery plus this pair.

Self-test: `python -m experiments.aversion` (also the permanent smoke).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import drives

# ── windows, derived (docstring carries the derivation) ─────────────────
_HORIZON_S = 1.0 / drives.BASAL_B          # 600 s resting starvation horizon
TASTE_RELIABLE_FRAC = 0.083                # 6 h / 72 h
TASTE_MAX_FRAC = 0.167                     # 12 h / 72 h
TASTE_RELIABLE_S = TASTE_RELIABLE_FRAC * _HORIZON_S     # ~50 s
TASTE_MAX_S = TASTE_MAX_FRAC * _HORIZON_S               # ~100 s
EXTERO_MAX_S = 0.8                         # ratio-derived; 4 decision steps

# ── learning constants, pre-registered ──────────────────────────────────
SIM_SIGMA = 0.15        # kernel width; derivation in docstring, from plants.py
LI_RATE = 0.3           # latent inhibition: 1/(1 + LI_RATE * familiarity)
EXT_RATE = 0.25         # extinction per uneventful re-exposure, kernel-weighted
GAIN = 1.0              # insult magnitude -> strength, identity by declaration


def kernel(a: np.ndarray, b: np.ndarray) -> float:
    """Similarity credit between two cue vectors: exp(-0.5 (|a-b|/sigma)^2)."""
    d = float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))
    return float(np.exp(-0.5 * (d / SIM_SIGMA) ** 2))


def taste_weight(delay_s: float) -> float:
    """Eligibility weight of a taste trace event at `delay_s` before an insult.

    1.0 through the reliable band, linear to 0.0 at the marginal edge —
    Smith & Roll 1967's shape at this world's scale. Negative delay (event
    after the insult) is 0: backward conditioning is not modelled.
    """
    if delay_s < 0.0 or delay_s >= TASTE_MAX_S:
        return 0.0
    if delay_s <= TASTE_RELIABLE_S:
        return 1.0
    return float((TASTE_MAX_S - delay_s) / (TASTE_MAX_S - TASTE_RELIABLE_S))


def extero_weight(delay_s: float) -> float:
    """Box window: a cue is either recent enough to blame or it is gone."""
    return 1.0 if 0.0 <= delay_s < EXTERO_MAX_S else 0.0


_WEIGHTS = {"taste": taste_weight, "extero": extero_weight}


@dataclass
class _TraceEvent:
    vec: np.ndarray
    t: float
    dose: float
    bound: bool = False      # a congruent insult credited this event


@dataclass
class _Entry:
    """One acquired aversion: a cue vector and how much it is hated."""
    vec: np.ndarray
    strength: float


class Channel:
    """One cue modality: an eligibility trace plus an aversion store.

    `window` names the weight function ("taste" or "extero") rather than
    holding one, so a Channel round-trips through JSON without pickling code.
    """

    def __init__(self, name: str, dim: int, window: str, max_s: float,
                 trace_cap: int = 32) -> None:
        if window not in _WEIGHTS:
            raise ValueError(f"unknown window {window!r}")
        self.name, self.dim, self.window, self.max_s = name, dim, window, max_s
        self.trace_cap = trace_cap
        self.trace: List[_TraceEvent] = []
        self.store: List[_Entry] = []
        self.familiarity: List[Tuple[np.ndarray, float]] = []   # (vec, dose)

    # -- working memory ---------------------------------------------------
    def observe(self, vec: Sequence[float], t: float, dose: float) -> None:
        v = np.asarray(vec, dtype=float)
        if v.shape != (self.dim,):
            raise ValueError(f"{self.name}: cue shape {v.shape} != ({self.dim},)")
        self.trace.append(_TraceEvent(vec=v, t=float(t), dose=float(dose)))
        if len(self.trace) > self.trace_cap:
            self.trace.pop(0)

    def _weight(self, delay_s: float) -> float:
        return _WEIGHTS[self.window](delay_s)

    # -- the one-shot update ----------------------------------------------
    def bind(self, t: float, magnitude: float) -> float:
        """Credit `magnitude` of insult onto every live trace event.

        Returns the total strength written (0.0 if the trace held nothing
        live — the caller may treat that as "insult went unexplained").
        Latent inhibition gates each event by its cue's prior safe history.
        """
        written = 0.0
        for ev in self.trace:
            w = self._weight(t - ev.t)
            if w <= 0.0:
                continue
            li = 1.0 / (1.0 + LI_RATE * self._familiar(ev.vec))
            s = GAIN * magnitude * w * li * ev.dose
            if s <= 0.0:
                continue
            self.store.append(_Entry(vec=ev.vec.copy(), strength=float(s)))
            ev.bound = True
            written += s
        return float(written)

    # -- extinction by uneventful re-exposure -------------------------------
    def resolve(self, t: float) -> None:
        """Retire expired trace events; uneventful ones extinguish matches.

        An event whose window has fully elapsed without a congruent insult
        is evidence the cue was safe THIS time: every store entry weakens by
        EXT_RATE * kernel * dose. Bound events retire silently. Also books
        the safe exposure into the familiarity ledger (latent inhibition).
        """
        live: List[_TraceEvent] = []
        for ev in self.trace:
            if self._weight(t - ev.t) > 0.0 or t < ev.t:
                live.append(ev)
                continue
            if not ev.bound:
                for e in self.store:
                    e.strength *= 1.0 - EXT_RATE * kernel(e.vec, ev.vec) * min(ev.dose, 1.0)
                self.familiarity.append((ev.vec.copy(), ev.dose))
        self.trace = live
        self.store = [e for e in self.store if e.strength > 1e-6]

    # -- queries -------------------------------------------------------------
    def aversion(self, vec: Sequence[float]) -> float:
        """How averse to this cue: kernel-weighted max over acquired entries.

        Max, not sum — ten writes about one bad meal are one grudge, and
        overlapping entries from a single smeared binding must not stack
        into a stronger aversion than the insult that wrote them.
        """
        v = np.asarray(vec, dtype=float)
        return max((e.strength * kernel(e.vec, v) for e in self.store), default=0.0)

    def _familiar(self, vec: np.ndarray) -> float:
        return float(sum(d * kernel(v, vec) for v, d in self.familiarity))

    # -- persistence (the part that crosses death) ---------------------------
    def to_jsonable(self) -> dict:
        return {
            "name": self.name, "dim": self.dim, "window": self.window,
            "max_s": self.max_s, "trace_cap": self.trace_cap,
            "store": [{"vec": e.vec.tolist(), "strength": e.strength}
                      for e in self.store],
            "familiarity": [{"vec": v.tolist(), "dose": d}
                            for v, d in self.familiarity],
        }

    @classmethod
    def from_jsonable(cls, d: dict) -> "Channel":
        ch = cls(d["name"], int(d["dim"]), d["window"], float(d["max_s"]),
                 trace_cap=int(d["trace_cap"]))
        ch.store = [_Entry(vec=np.asarray(e["vec"], float),
                           strength=float(e["strength"])) for e in d["store"]]
        ch.familiarity = [(np.asarray(f["vec"], float), float(f["dose"]))
                          for f in d["familiarity"]]
        return ch


class FastPath:
    """The routed associator. Channels are declared, insults are routed.

    Default wiring is the §8.4 design; TA.02's placebo control adds a third
    channel wired EXACTLY like taste (same window, same routing) whose
    content is noise — the machinery must not mind, and must buy nothing.

        fp = FastPath()                          # taste + extero
        fp.add_channel("placebo", dim=5, window="taste",
                       max_s=TASTE_MAX_S, route_from=("illness",))
        fp.observe("taste", t_vec, t=12.0, dose=0.15)
        fp.insult("illness", t=42.0, magnitude=0.20)
        fp.aversion("taste", t_vec)              # > 0 now
        fp.resolve(t)                            # caller owns time
    """

    def __init__(self) -> None:
        self.channels: Dict[str, Channel] = {}
        self.routing: Dict[str, Tuple[str, ...]] = {}
        self.add_channel("taste", dim=5, window="taste", max_s=TASTE_MAX_S,
                         route_from=("illness",))
        self.add_channel("extero", dim=4, window="extero", max_s=EXTERO_MAX_S,
                         route_from=("shock",))

    def add_channel(self, name: str, dim: int, window: str, max_s: float,
                    route_from: Tuple[str, ...] = ()) -> None:
        self.channels[name] = Channel(name, dim, window, max_s)
        for kind in route_from:
            self.routing[kind] = tuple(self.routing.get(kind, ())) + (name,)

    def observe(self, channel: str, vec: Sequence[float], t: float,
                dose: float) -> None:
        self.channels[channel].observe(vec, t, dose)

    def insult(self, kind: str, t: float, magnitude: float) -> float:
        """One insult, one update, routed. Returns total strength written."""
        if kind not in self.routing:
            raise ValueError(f"unrouted insult kind {kind!r}")
        return float(sum(self.channels[c].bind(t, magnitude)
                         for c in self.routing[kind]))

    def resolve(self, t: float) -> None:
        for ch in self.channels.values():
            ch.resolve(t)

    def aversion(self, channel: str, vec: Sequence[float]) -> float:
        return self.channels[channel].aversion(vec)

    def to_jsonable(self) -> dict:
        return {"channels": {k: c.to_jsonable() for k, c in self.channels.items()},
                "routing": {k: list(v) for k, v in self.routing.items()}}

    @classmethod
    def from_jsonable(cls, d: dict) -> "FastPath":
        fp = cls.__new__(cls)
        fp.channels = {k: Channel.from_jsonable(c)
                       for k, c in d["channels"].items()}
        fp.routing = {k: tuple(v) for k, v in d["routing"].items()}
        return fp


# ── smoke — permanent, run by `python -m experiments.aversion` ──────────
def _smoke() -> None:
    import json
    from . import plants

    rng = np.random.RandomState(0)
    toxic_t = plants.TOXIC.taste(rng)
    safe_t = plants.SAFE.taste(rng)

    # 1. THE CLAIM'S SHAPE: one ingestion, illness at DELAY_S, aversion to the
    # toxic taste generalising over toxic individuals, not to the safe twin.
    fp = FastPath()
    fp.observe("taste", toxic_t, t=10.0, dose=plants.Q_FIRST)
    di = plants.illness_total(plants.Q_FIRST, plants.TOXIC.potency)
    written = fp.insult("illness", t=10.0 + plants.DELAY_S, magnitude=di)
    assert written > 0.0, "illness inside the reliable window must bind"
    a_tox = fp.aversion("taste", plants.TOXIC.taste(rng))   # fresh individual
    a_safe = fp.aversion("taste", plants.SAFE.taste(rng))
    assert a_tox > 10.0 * a_safe > 0.0 or a_safe == 0.0, (a_tox, a_safe)
    assert a_tox > 0.005, f"one-shot aversion too weak to act on: {a_tox}"

    # 2. ROUTING (control (a)'s mechanism): an AV cue in the extero trace at
    # ingestion time must NOT be blamed for a DELAY_S-late illness — illness
    # never reads extero, and 30 s is 37x the extero window anyway.
    fp2 = FastPath()
    av = np.array([1.0, 0.0, 0.5, 0.2])
    fp2.observe("extero", av, t=10.0, dose=1.0)
    fp2.insult("illness", t=10.0 + plants.DELAY_S, magnitude=di)
    assert fp2.aversion("extero", av) == 0.0

    # 3. THE MUST-PASS PATH (control (d)'s mechanism): AV cue then fast shock
    # binds; and the taste sitting in its long window must NOT catch shock
    # blame (the Garcia-Koelling double dissociation, both halves).
    fp3 = FastPath()
    fp3.observe("taste", toxic_t, t=10.0, dose=plants.Q_FIRST)
    fp3.observe("extero", av, t=10.0, dose=1.0)
    fp3.insult("shock", t=10.4, magnitude=0.5)
    assert fp3.aversion("extero", av) > 0.0
    assert fp3.aversion("taste", toxic_t) == 0.0

    # 4. WINDOW EDGE: beyond the marginal edge nothing binds (weight hits 0).
    fp4 = FastPath()
    fp4.observe("taste", toxic_t, t=0.0, dose=1.0)
    assert fp4.insult("illness", t=TASTE_MAX_S + 1.0, magnitude=1.0) == 0.0

    # 5. CONTENT ADDRESSING (control (b)'s mechanism): illness bound to a
    # random taste far from both types must not transfer to the real poison.
    fp5 = FastPath()
    shuffled = np.array([0.05, 0.95, 0.9, 0.9, 0.9])
    fp5.observe("taste", shuffled, t=10.0, dose=plants.Q_FIRST)
    fp5.insult("illness", t=40.0, magnitude=di)
    assert fp5.aversion("taste", toxic_t) < 1e-4 * fp5.aversion("taste", shuffled)

    # 6. LATENT INHIBITION: ten safe full portions of the safe taste, then a
    # (false-alarm) illness pairing — acquisition must be well below novel.
    fp6 = FastPath()
    for k in range(10):
        fp6.observe("taste", plants.SAFE.taste(rng), t=200.0 * k, dose=1.0)
        fp6.resolve(200.0 * k + TASTE_MAX_S + 1.0)
    fp6.observe("taste", safe_t, t=5000.0, dose=plants.Q_FIRST)
    w_familiar = fp6.insult("illness", t=5030.0, magnitude=di)
    fp7 = FastPath()
    fp7.observe("taste", safe_t, t=5000.0, dose=plants.Q_FIRST)
    w_novel = fp7.insult("illness", t=5030.0, magnitude=di)
    assert w_familiar < 0.5 * w_novel, (w_familiar, w_novel)

    # 7. EXTINCTION by safe re-exposure, and by NOTHING else: strength holds
    # over dead time, then falls with each uneventful re-taste.
    fp8 = FastPath()
    fp8.observe("taste", toxic_t, t=10.0, dose=plants.Q_FIRST)
    fp8.insult("illness", t=40.0, magnitude=di)
    a0 = fp8.aversion("taste", toxic_t)
    fp8.resolve(100000.0)                       # months of nothing
    assert fp8.aversion("taste", toxic_t) == a0, "no decay by clock"
    for k in range(6):
        fp8.observe("taste", toxic_t, t=200000.0 + 200.0 * k, dose=plants.Q_FIRST)
        fp8.resolve(200000.0 + 200.0 * k + TASTE_MAX_S + 1.0)
    a1 = fp8.aversion("taste", toxic_t)
    assert a1 < 0.9 * a0, (a0, a1)

    # 8. DEATH ROUND-TRIP: values cross, working memory does not.
    fp9 = FastPath()
    fp9.observe("taste", toxic_t, t=10.0, dose=plants.Q_FIRST)
    fp9.insult("illness", t=40.0, magnitude=di)
    fp9.observe("taste", safe_t, t=41.0, dose=1.0)      # un-resolved trace
    blob = json.dumps(fp9.to_jsonable())
    fp10 = FastPath.from_jsonable(json.loads(blob))
    assert fp10.aversion("taste", toxic_t) == fp9.aversion("taste", toxic_t)
    assert all(len(c.trace) == 0 for c in fp10.channels.values())

    # 9. PLACEBO WIRING (control (c)'s harness): a noise channel wired exactly
    # like taste binds mechanically — the discriminating failure TA.02 gates
    # is behavioural (noise identifies no plant), so here we only assert the
    # machinery accepts the wiring and the aversion does not read plant type.
    fp11 = FastPath()
    fp11.add_channel("placebo", dim=5, window="taste", max_s=TASTE_MAX_S,
                     route_from=("illness",))
    noise = rng.rand(5)
    fp11.observe("placebo", noise, t=10.0, dose=plants.Q_FIRST)
    fp11.insult("illness", t=40.0, magnitude=di)
    assert fp11.aversion("placebo", noise) > 0.0
    assert fp11.aversion("placebo", rng.rand(5)) < fp11.aversion("placebo", noise)

    print("aversion.py smoke: 9/9 properties hold")
    print(f"  windows: taste reliable {TASTE_RELIABLE_S:.0f}s / max "
          f"{TASTE_MAX_S:.0f}s, extero {EXTERO_MAX_S}s; DELAY_S {plants.DELAY_S}s")
    print(f"  one-shot aversion at toxic taste: {a_tox:.4f} "
          f"(safe twin: {a_safe:.6f}, ratio {a_tox / max(a_safe, 1e-12):.0f}x)")


if __name__ == "__main__":
    _smoke()
