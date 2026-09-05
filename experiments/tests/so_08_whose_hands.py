"""SO.08 — The diary records WHOSE hands, and he acts on it.

GOAL.md's sentence: *"His diary records who left it — so gratitude, like
trust, has somewhere real to grow."* LG.02 proved trust in a VOICE can be
earned from the attributed diary; this spec proves the same for HANDS — two
donors of different reliability leave visually indistinguishable gifts, and
Jack's approach rate to a newly dropped object must diverge by donor, with the
divergence running THROUGH the attributed diary: strip or shuffle the donor
field and it must collapse.

THE WORLD. N_ROUNDS rounds; each round one donor (they alternate) drops one
gift — a plants.py plant, the certified unreliable gift: two types drawn from
ONE visual distribution (TA.01's certificate), distinct only in the toxin. The
good donor's gifts are SAFE at GOOD_P; the bad donor's at BAD_P. Jack sees the
drop and who made it (SO.06's channel: the placement is logged and becomes
perceptible; here that is a "saw" diary event with the donor as speaker — the
attribution GOAL.md asks for). He then decides whether to APPROACH and eat.
The gift's value is discovered ONLY by eating it, on plants.py's declared
dose-response and DELAY — a neophobic first bite (Q_FIRST, felt and survivable
by TA.01's own gate), with the malaise arriving DELAY_S later through the real
`Toxin` schedule, never as an annotation. A cautious policy gathers no
evidence at all: an unapproached gift writes no outcome, which is the
registered difference between this venue and LG.02's (an uttered claim is
verified whether or not you follow it; an object is not).

THE MECHANISM IS LG.02's, IMPORTED — the T2.16/LG.04-A4 rule, named in the
registry: one Laplace posterior over verified outcomes, shared implementation,
never re-derived. `_trust` is imported from `lg_02_liar_loses_him` and used
byte-identically (posterior, join, WINDOW, PRIOR all LG.02's). What SO.08 owns
is only the venue's grammar: `_AsClaims` is a READ-ONLY view of the diary that
presents each EATEN gift to `_trust` in the event shape it joins — the gift as
the donor's implicit claim ("this is safe food"), the felt outcome as Jack's
own finding. Strip or shuffle the donor field at RECORD time and the view
carries the damage straight into the imported mechanism; nothing downstream
can repair it, because nothing downstream knows the truth.

WHAT IS TRUE BY CONSTRUCTION, declared rather than buried (LG.02's clause,
inherited): the approach DECISION RULE (probability = posterior mean) is
designed, not learned. What is measured is that the posterior it reads can
only be computed from attributed lived experience — the null and the shuffle
are the proof: identical rule, identical world stream, attribution removed or
scrambled, divergence gone.

THE NULL (registry: "Donor-stripped diary, same events, same counts — LG.02's
null, reused"): the donor field is scrubbed at record time (speaker
"someone"), one pooled posterior drives every approach, and whole-life
|divergence by true donor| must stay under COLLAPSE_MAX. Its aliveness band:
pooled trust must land in NULL_TRUST_BAND — a null sitting at the untouched
prior with no evidence joined is a dead instrument, and dead instruments VOID
(24th-audit B3).

THE CONTROLS (registry, both mandatory):
  SHUFFLE — content intact, source permuted at record time (a permutation of
  the true donor sequence, so per-donor event counts are exact). Divergence by
  TRUE donor must collapse under COLLAPSE_MAX (Johnson, Hashtroudi & Lindsay
  1993's source-monitoring dissociation made a control). Aliveness: both
  shuffled-label posteriors must land in SHUF_TRUST_BAND — the join ran on
  scrambled sources, it did not silently die.
  EQUAL-DONORS — two donors of IDENTICAL reliability (both GOOD_P): whole-life
  |divergence| must stay under COLLAPSE_MAX, and both posteriors must clear
  EQ_TRUST_MIN (a detector that reports a difference where none exists is
  broken; one whose posteriors never moved off the prior measured nothing).

THE SO.07 LESSON, carried as a RIG GATE rather than prose (its VOID root
cause: fixture constants frozen from one venue did not transfer to another).
This venue's credit assignment assumes one round is long enough to feel one
gift's whole malaise: ROUND_S > DELAY_S + ILL_WINDOW_S. That inequality is
plants.py's constants against this file's, so it is computed at run time and
recorded (`round_window_ok`); if plants.py is ever recalibrated past it, the
run VOIDs loudly instead of smearing credit across rounds (Kwok & Boakes
overshadowing, which this venue does not model).

GATES, pre-registered 2026-09-05 BEFORE any SO.08 number existed — bars from
binomial arithmetic, not previews. Expected claim divergence: the good donor's
posterior plateaus near (0.9*W+1)/(W+2) ~ 0.87 at WINDOW=30 while the bad
donor's self-suppresses near 0.15 (he stops sampling what he distrusts, so its
evidence stays thin — that asymmetry is the venue's point), giving expected
last-quarter divergence ~0.72 with per-seed sd ~0.09 at 30 decisions/donor;
MIN_DIV 0.40 sits >3 sigma below. Collapse legs are whole-life (120
decisions/donor): the noisiest (shuffle/null, p~0.5) has sd ~0.065, so
COLLAPSE_MAX 0.20 sits ~3 sigma. NULL_TRUST_BAND (0.35,0.65) is ~2.9 sigma for
an alternating-fed window (sd ~0.05, LG.02's own arithmetic); the shuffled
windows are iid-mixed (sd ~0.086), so SHUF_TRUST_BAND is the wider
(0.25,0.75) at ~2.9 sigma. EQ_TRUST_MIN 0.70 is ~3.4 sigma below the 0.875
plateau (window sd ~0.05). Reliability bands (0.75,1.0)/(0.0,0.25) sit >5
sigma from realized 0.9/0.1 at 120 drops. Worst-seed, read from the RECORDED
ROW: every gated metric is recorded per seed as `<key>_s<seed>` (LG.02's
repaired pattern) and `_check` is a pure function of (m, c).

    VOID  — the record is missing a per-seed key the gates need; the round
            window cannot contain one malaise (round_window_ok != 1); an eaten
            gift with no felt outcome or an outcome with no eaten gift
            (verify_complete != 1); a donor was not systematic (realized
            safe-rate outside its band, claim and equal lives); the null's
            pooled trust outside NULL_TRUST_BAND; a shuffled posterior outside
            SHUF_TRUST_BAND; an equal-donors posterior under EQ_TRUST_MIN
            (dead instruments VOID, they do not testify).
    FAIL  — worst-seed last-quarter divergence < MIN_DIV; cross-seed
            mean - 3*sd <= 0; first-encounter trust not exactly PRIOR for both
            donors (scripted divergence — the registry's "initialised" kill);
            attributed recall of who-left-what below ATTRIB_MIN (divergence
            with attribution broken is a leak); the stripped null, the
            shuffle, or the equal-donors life diverging past COLLAPSE_MAX.
    PASS  — all of the above on the right side, every seed.
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from .. import plants
from .lg_02_liar_loses_him import PRIOR, WINDOW, _trust

REPO = Path(__file__).resolve().parents[2]

# The posterior/join is lg_02's and the outcome machinery is plants.py's —
# both hash into this certificate. EpisodicMemory is the diary under all of it.
IMPL_DEPS = ["EpisodicMemory.py", "experiments/plants.py",
             "experiments/tests/lg_02_liar_loses_him.py"]

SEEDS = (0, 1, 2)
DONORS = ("asha", "brom")        # asha reliable, brom not (claim life);
                                 # both reliable in the equal-donors life
N_ROUNDS = 240                   # alternating donors -> 120 drops each
ROUND_S = 60.0                   # must exceed DELAY_S + ILL_WINDOW_S (gated)
DT = 0.2                         # W0's decision length: malaise is FELT at the
                                 # cadence he lives at, not read off a table
GOOD_P, BAD_P = 0.9, 0.1         # P(the drop is the SAFE type)

LAST_Q = 180                     # rounds >= this form the "end of life" read
MIN_DIV = 0.40                   # worst-seed lastq approach-rate divergence
COLLAPSE_MAX = 0.20              # null/shuffle/equal, whole life, |divergence|
NULL_TRUST_BAND = (0.35, 0.65)   # pooled posterior actually moved (alternating
                                 # feed: sd ~0.05 -> ~2.9 sigma)
SHUF_TRUST_BAND = (0.25, 0.75)   # iid-mixed windows are noisier (sd ~0.086)
EQ_TRUST_MIN = 0.70              # equal-donors posteriors off the prior
REL_GOOD_BAND = (0.75, 1.0)      # realized safe-rate, reliable donor's drops
REL_BAD_BAND = (0.0, 0.25)       # realized safe-rate, unreliable donor's drops
ATTRIB_MIN = 0.90                # diary recall of WHOSE hands left the gift
N_ATTRIB_Q = 20


class _Ev:
    """The minimal event shape `_trust` reads."""
    __slots__ = ("channel", "speaker", "text")

    def __init__(self, channel: str, speaker: str, text: str) -> None:
        self.channel, self.speaker, self.text = channel, speaker, text


class _AsClaims:
    """Read-only view: the diary, in the grammar LG.02's `_trust` joins.

    Each EATEN gift becomes two synthetic events — the donor's implicit claim
    ("the gift is at the safe site") and Jack's felt finding ("the safe/toxic
    site") — so the imported posterior scores hit = (claimed == found) exactly
    as it does for LG.02's advisors. Built from `mem.events` alone: whatever
    attribution was written at record time (true donor, "someone", or a
    shuffled label) is what the mechanism sees. Unapproached gifts have no
    outcome event and therefore never enter — evidence only by eating.
    """

    __slots__ = ("events",)

    def __init__(self, mem) -> None:
        felt: dict[int, str] = {}
        for e in mem.events:
            if e.channel == "did" and " felt " in e.text:
                felt[int(e.text.rsplit(" ", 1)[-1])] = (
                    "safe" if " fine " in e.text else "toxic")
        evs: list[_Ev] = []
        for e in mem.events:
            if e.channel == "saw" and " left a gift " in e.text:
                r = int(e.text.rsplit(" ", 1)[-1])
                if r in felt:
                    evs.append(_Ev("heard", e.speaker,
                                   f"{e.speaker} promised the safe site "
                                   f"in round {r}"))
                    evs.append(_Ev("did", "jack",
                                   f"jack felt the {felt[r]} site "
                                   f"in round {r}"))
        self.events = evs


def _drop_text(name: str, r: int) -> str:
    return f"{name} left a gift at the clearing in round {r}"


def _live(seed: int, mem_path, mode: str):
    """One life. mode: claim | null | shuffle | equal. Returns per-round
    records the rig scores; the agent inside sees only the diary."""
    sys.path.insert(0, str(REPO))
    from EpisodicMemory import EpisodicMemory

    window_ok = ROUND_S > plants.DELAY_S + plants.ILL_WINDOW_S
    rng_world = random.Random(seed)          # identical across arms
    rng_agent = random.Random(seed + 101)
    p_safe = {DONORS[0]: GOOD_P,
              DONORS[1]: GOOD_P if mode == "equal" else BAD_P}
    labels = list(DONORS[r % 2] for r in range(N_ROUNDS))
    if mode == "shuffle":
        recorded = list(labels)
        random.Random(seed + 202).shuffle(recorded)   # counts preserved: 120/120
    else:
        recorded = labels

    mem = EpisodicMemory(path=mem_path)
    tox = plants.Toxin()
    t0 = 3_000_000.0
    rows = []
    first_trust: dict[str, float] = {}
    for r in range(N_ROUNDS):
        donor = labels[r]
        t_r = t0 + r * ROUND_S
        safe = rng_world.random() < p_safe[donor]
        plant = plants.SAFE if safe else plants.TOXIC
        name = "someone" if mode == "null" else recorded[r]

        mem.record("saw", name, _drop_text(name, r), t=t_r)
        p = _trust(_AsClaims(mem), None if mode == "null" else name)
        if donor not in first_trust:
            first_trust[donor] = p
        approach = rng_agent.random() < p

        ate = False
        if approach and window_ok:
            ate = True
            t_eat = t_r + 1.0
            mem.record("did", "jack", f"jack ate the gift in round {r}",
                       t=t_eat)
            tox.ingest(t_eat, plant, plants.Q_FIRST)
            # Feel the round out at decision cadence: the delay is lived
            # through the real schedule, not looked up.
            sick = False
            t = t_eat
            while t < t_r + ROUND_S - 1.0:
                if tox.rate(t) > 0.0:
                    sick = True
                t += DT
            word = "sick" if sick else "fine"
            mem.record("did", "jack",
                       f"jack felt {word} after the gift in round {r}",
                       t=t_r + ROUND_S - 1.0)
        rows.append({"r": r, "donor": donor, "recorded": name, "safe": safe,
                     "approach": approach, "ate": ate, "trust": p})
    return mem, rows, first_trust, t0 + N_ROUNDS * ROUND_S, float(window_ok)


def _approach_rate(rows, donor, lo=0, hi=N_ROUNDS):
    picks = [x["approach"] for x in rows
             if x["donor"] == donor and lo <= x["r"] < hi]
    return sum(picks) / max(1, len(picks))


def _safe_rate(rows, donor):
    drops = [x["safe"] for x in rows if x["donor"] == donor]
    return sum(drops) / max(1, len(drops))


def _measure(seed: int) -> dict:
    tmp = Path(tempfile.mkdtemp())
    good, bad = DONORS

    # ── the claim life ──
    mem, rows, first, now, window_ok = _live(seed, tmp / "life.jsonl", "claim")
    ar_g = _approach_rate(rows, good, LAST_Q)
    ar_b = _approach_rate(rows, bad, LAST_Q)

    # Attribution intact: WHOSE hands must come back from the diary with the
    # right name on it, through ME.9's provenance-filtered recall.
    rng_q = random.Random(seed + 7)
    qs = rng_q.sample(rows, N_ATTRIB_Q)
    hits = 0
    for x in qs:
        res = mem.recall(_drop_text(x["donor"], x["r"]), top_k=1,
                         channel="saw", speaker=x["donor"], now=now)
        if res:
            got = res[0].event
            hits += (got.speaker.lower() == x["donor"]
                     and got.text.endswith(f"in round {x['r']}"))
    attrib_acc = hits / N_ATTRIB_Q

    # Join completeness: every eaten gift felt, nothing felt but uneaten.
    n_ate = sum(1 for e in mem.events
                if e.channel == "did" and " ate the gift " in e.text)
    n_felt = sum(1 for e in mem.events
                 if e.channel == "did" and " felt " in e.text)

    # ── the stripped null (same world stream, no record of whose hands) ──
    _, rows_n, _, _, _ = _live(seed, tmp / "null.jsonl", "null")
    ndiv = _approach_rate(rows_n, good) - _approach_rate(rows_n, bad)
    null_trust_end = rows_n[-1]["trust"]

    return {
        "div_lastq": round(ar_g - ar_b, 4),
        "approach_good_lastq": round(ar_g, 4),
        "approach_bad_lastq": round(ar_b, 4),
        "rel_good": round(_safe_rate(rows, good), 4),
        "rel_bad": round(_safe_rate(rows, bad), 4),
        "first_trust_good": first[good],
        "first_trust_bad": first[bad],
        "attrib_acc": round(attrib_acc, 4),
        "verify_complete": float(n_felt == n_ate),
        "round_window_ok": window_ok,
        "n_eaten_good": float(sum(1 for x in rows
                                  if x["donor"] == good and x["ate"])),
        "n_eaten_bad": float(sum(1 for x in rows
                                 if x["donor"] == bad and x["ate"])),
        "null_abs_div": round(abs(ndiv), 4),
        "null_trust_end": round(null_trust_end, 4),
    }


def _measure_controls(seed: int) -> dict:
    tmp = Path(tempfile.mkdtemp())
    good, bad = DONORS

    # ── SHUFFLE: content intact, source permuted at record time ──
    mem_s, rows_s, _, _, _ = _live(seed, tmp / "shuffle.jsonl", "shuffle")
    sdiv = _approach_rate(rows_s, good) - _approach_rate(rows_s, bad)
    view = _AsClaims(mem_s)
    st_a, st_b = _trust(view, good), _trust(view, bad)

    # ── EQUAL-DONORS: identical reliability, divergence must not appear ──
    mem_e, rows_e, _, _, _ = _live(seed, tmp / "equal.jsonl", "equal")
    ediv = _approach_rate(rows_e, good) - _approach_rate(rows_e, bad)
    view_e = _AsClaims(mem_e)
    et_a, et_b = _trust(view_e, good), _trust(view_e, bad)

    return {
        "shuf_abs_div": round(abs(sdiv), 4),
        "shuf_trust_a_end": round(st_a, 4),
        "shuf_trust_b_end": round(st_b, 4),
        "eq_abs_div": round(abs(ediv), 4),
        "eq_rel_a": round(_safe_rate(rows_e, good), 4),
        "eq_rel_b": round(_safe_rate(rows_e, bad), 4),
        "eq_trust_a_end": round(et_a, 4),
        "eq_trust_b_end": round(et_b, 4),
    }


_MEMO: dict = {}
_CTL_MEMO: dict = {}


def _flat(memo: dict, seed: int) -> dict:
    """The seed's own metrics under their plain names PLUS every seed's value
    as an explicit `<key>_s<seed>` key — identical in every run, so run_spec's
    aggregation records them verbatim and `_check`'s worst-seed gates are
    answerable from the row alone (LG.02's repaired pattern)."""
    out = dict(memo[seed])
    for s in SEEDS:
        for k, v in memo[s].items():
            out[f"{k}_s{s}"] = v
    return out


def _experiment(seed: int) -> dict:
    for s in SEEDS:
        if s not in _MEMO:
            _MEMO[s] = _measure(s)
    return _flat(_MEMO, seed)


def _control(seed: int) -> dict:
    for s in SEEDS:
        if s not in _CTL_MEMO:
            _CTL_MEMO[s] = _measure_controls(s)
    return _flat(_CTL_MEMO, seed)


def _in_band(vals, band) -> bool:
    return all(band[0] <= v <= band[1] for v in vals)


_NEED_M = tuple(f"{k}_s{s}" for k in (
    "div_lastq", "rel_good", "rel_bad", "first_trust_good", "first_trust_bad",
    "attrib_acc", "verify_complete", "round_window_ok", "null_abs_div",
    "null_trust_end") for s in SEEDS)
_NEED_C = tuple(f"{k}_s{s}" for k in (
    "shuf_abs_div", "shuf_trust_a_end", "shuf_trust_b_end", "eq_abs_div",
    "eq_rel_a", "eq_rel_b", "eq_trust_a_end", "eq_trust_b_end") for s in SEEDS)


def _check(m: dict, c: dict):
    """Pure function of the recorded row — no module state; every key read up
    front so each is consulted on every replay regardless of which gate
    fires."""
    if any(k not in m for k in _NEED_M) or any(k not in c for k in _NEED_C):
        return Status.VOID          # the record cannot answer the gates
    divs = tuple(m[f"div_lastq_s{s}"] for s in SEEDS)
    rel_g = tuple(m[f"rel_good_s{s}"] for s in SEEDS)
    rel_b = tuple(m[f"rel_bad_s{s}"] for s in SEEDS)
    ft_g = tuple(m[f"first_trust_good_s{s}"] for s in SEEDS)
    ft_b = tuple(m[f"first_trust_bad_s{s}"] for s in SEEDS)
    att = tuple(m[f"attrib_acc_s{s}"] for s in SEEDS)
    verify = tuple(m[f"verify_complete_s{s}"] for s in SEEDS)
    windows = tuple(m[f"round_window_ok_s{s}"] for s in SEEDS)
    ndiv = tuple(m[f"null_abs_div_s{s}"] for s in SEEDS)
    ntr = tuple(m[f"null_trust_end_s{s}"] for s in SEEDS)
    sdiv = tuple(c[f"shuf_abs_div_s{s}"] for s in SEEDS)
    str_a = tuple(c[f"shuf_trust_a_end_s{s}"] for s in SEEDS)
    str_b = tuple(c[f"shuf_trust_b_end_s{s}"] for s in SEEDS)
    ediv = tuple(c[f"eq_abs_div_s{s}"] for s in SEEDS)
    erel_a = tuple(c[f"eq_rel_a_s{s}"] for s in SEEDS)
    erel_b = tuple(c[f"eq_rel_b_s{s}"] for s in SEEDS)
    etr_a = tuple(c[f"eq_trust_a_end_s{s}"] for s in SEEDS)
    etr_b = tuple(c[f"eq_trust_b_end_s{s}"] for s in SEEDS)
    # ── rig gates: VOID, not FAIL — a run that could not ask the question ──
    if min(windows) != 1.0:
        return Status.VOID          # one round cannot contain one malaise
    if min(verify) != 1.0:
        return Status.VOID          # an outcome with nothing to join to
    if not (_in_band(rel_g, REL_GOOD_BAND) and _in_band(rel_b, REL_BAD_BAND)
            and _in_band(erel_a, REL_GOOD_BAND)
            and _in_band(erel_b, REL_GOOD_BAND)):
        return Status.VOID          # a donor was not systematic
    if not _in_band(ntr, NULL_TRUST_BAND):
        return Status.VOID          # the stripped join never actually ran
    if not (_in_band(str_a, SHUF_TRUST_BAND)
            and _in_band(str_b, SHUF_TRUST_BAND)):
        return Status.VOID          # the shuffled join never actually ran
    if min(etr_a) < EQ_TRUST_MIN or min(etr_b) < EQ_TRUST_MIN:
        return Status.VOID          # the equal-donors posteriors never moved
    # ── the claim, its guards, the null and both controls, EVERY seed ──
    mean = sum(divs) / len(divs)
    sd = (sum((d - mean) ** 2 for d in divs) / len(divs)) ** 0.5
    return bool(
        min(divs) >= MIN_DIV
        and mean - 3 * sd > 0
        and all(v == PRIOR for v in ft_g)
        and all(v == PRIOR for v in ft_b)
        and min(att) >= ATTRIB_MIN
        and max(ndiv) <= COLLAPSE_MAX
        and max(sdiv) <= COLLAPSE_MAX
        and max(ediv) <= COLLAPSE_MAX)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["SO.08"], _experiment, _check,
                    control_fn=_control, ledger=ledger)


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        # Crash-freedom only: a 6-round toy life per mode, metric VALUES not
        # printed — the gates above were registered from arithmetic, not
        # previews.
        globals().update(N_ROUNDS=6, LAST_Q=4, N_ATTRIB_Q=3)
        tmp = Path(tempfile.mkdtemp())
        for mode in ("claim", "null", "shuffle", "equal"):
            _live(0, tmp / f"smoke_{mode}.jsonl", mode)
        print("smoke: 4 toy lives completed without error")
    else:
        print(json.dumps({"experiment": _measure(0),
                          "control": _measure_controls(0)}, indent=2))
