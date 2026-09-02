"""LG.02 — Trust is earned by track record: the liar loses him.

THE LIAR TEST, owner-designed 2026-08-09, verbatim in the registry: two
advisors speak into his world, one systematically truthful, one systematically
false, every piece of advice verifiable by his own subsequent experience. His
advice-following must DIVERGE by advisor track record — with attribution
intact, through ME.9's channels — and the divergence must be EARNED, never
initialised.

THE WORLD. N_ROUNDS foraging rounds; each round the food is at one of two
sites and the round's advisor (they alternate) claims a site. The truthful
advisor's claim matches the world at TRUTH_P; the liar's at LIE_P. Jack picks
a site — the claimed one if he follows, the other if not — and ends every
round knowing where the food actually was (he searches; an empty site sends
him to the other), so every claim is verified by his OWN experience, which is
what makes track record computable without an annotator. The diary records
the claim (channel "heard", speaker = the advisor) and the finding (channel
"did", speaker "jack") as raw observations: no event says "this advice was
false" — the JOIN of who-claimed-what with what-he-found is the only place
veracity exists, and that join runs through the attributed diary.

THE MECHANISM — one code path for claim, null and control, because a
control-specific mechanism is a control that cannot fail. At each decision,
trust in the current speaker is recomputed FROM THE DIARY: retrieve that
speaker's past claims (ME.9's channel+speaker filter), join each to his own
recorded finding for that round, take the last WINDOW verified claims, and
score (hits + 1) / (n + 2) — a Laplace posterior whose prior is exactly 0.5
for every speaker with no evidence. Follow with probability = trust. Nothing
is keyed on an advisor's name; nothing is initialised apart; the window is
what lets trust MIGRATE when the world changes who is honest.

THE KILLS-CLAUSE GUARD, measured rather than asserted: the registry kills any
scripted-trust design ("if trust must be initialised, annotated or hard-coded
to diverge, the emergence claim is dead"). So the trust each advisor holds at
first encounter is captured and must equal PRIOR (0.5) EXACTLY, both
advisors, every seed. A mechanism that starts them apart fails here before
its divergence is worth anything.

THE NULL (declared in the registry; lives in _experiment like ME.9's pooled
null): ATTRIBUTION STRIPPED AT RECORD TIME — the heard events carry speaker
"someone" and the advisor's name is scrubbed from the text, so there is no
record of who said what. The same mechanism then computes ONE pooled track
record (~50% true by construction, since the two advisors alternate) and must
treat both advisors identically: |whole-life divergence| <= NULL_DIV_MAX. If
the stripped agent diverges, speaker identity leaks outside the attributed
diary and the arm's divergence measures the leak, not trust. The null carries
its own aliveness band: its pooled trust must land in NULL_TRUST_BAND,
proving the join machinery ran and only attribution was missing — a null
whose trust sits at the untouched prior is a dead instrument, and dead
instruments VOID (24th-audit B3).

THE CONTROL (owner-designed, declared in the registry): THE SWAP. The
advisors exchange roles at SWAP_ROUND. Trust must MIGRATE — over the last
quarter, follow-rate(newly-truthful) - follow-rate(newly-lying) >=
MIN_MIGRATE — or the mechanism was tracking voices, not veracity. And it must
ALSO have been right before the swap (quarter-2 divergence toward the
originally-truthful >= MIN_PRESWAP), or a mechanism that ignores the first
half entirely could fake migration.

WHAT IS TRUE BY CONSTRUCTION, declared rather than buried: the follow
DECISION RULE (probability = posterior mean) is designed, not learned — what
is measured is that the POSTERIOR it reads can only be computed from
attributed lived experience. The null is the proof: identical rule, identical
world, identical evidence stream, attribution removed, divergence gone.

GATES, pre-registered 2026-09-02 BEFORE any LG.02 number existed (bars from
binomial arithmetic, not previews: expected divergence ~0.75 with per-seed sd
~0.09, so MIN_DIV 0.40 is >3 sigma below expectation; the null's whole-life
divergence sd is ~0.065, so NULL_DIV_MAX 0.20 sits at 3 sigma). Worst-seed,
read from the RECORDED ROW: every gated metric is recorded per seed as an
explicit `<key>_s<seed>` key (each seed's run returns the full per-seed set,
identical across runs, so run_spec's mean/std aggregation carries the values
into the row verbatim), and `_check` is a pure function of (m, c) — no module
state. The attempt-1 version read module-level _MEMO/_CTL_MEMO instead, so
its PASS could not be replayed from its row; T0.13 attempt 22 flagged the
gate keyless AND stale (2026-09-02 16:15), and this is that repair. Bars are
byte-identical to the pre-registered ones; only recording and the gate's
input source changed.

    VOID  — a rig that could not ask the question: the record is missing a
            per-seed key the gates need (a row that cannot answer is not a
            refutation); an
            advisor was not systematic (realized truth-rate outside
            TRUTH_BAND / LIE_BAND, per half in the swap life); verification
            incomplete (a claim with no finding to join); or the null's
            pooled trust outside NULL_TRUST_BAND (the join never ran).
    FAIL  — worst-seed last-quarter divergence < MIN_DIV; or cross-seed
            mean - 3*sd <= 0; or first-encounter trust not exactly PRIOR for
            both advisors (scripted divergence); or attributed recall of the
            advice below ATTRIB_MIN (divergence with attribution broken is
            a leak, per the registry's falsified_by); or the stripped null
            diverges past NULL_DIV_MAX; or the swap life fails to migrate
            (MIN_MIGRATE) or was not right pre-swap (MIN_PRESWAP).
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

REPO = Path(__file__).resolve().parents[2]

IMPL_DEPS = ["EpisodicMemory.py"]

SEEDS = (0, 1, 2)
SITES = ("north", "south")
ADVISORS = ("mira", "tovan")     # truthful-first in the claim life; swap life
                                 # makes mira the first-half truthful voice
N_ROUNDS = 240                   # alternating speakers -> 120 claims each
SWAP_ROUND = 120                 # control life only
TRUTH_P, LIE_P = 0.9, 0.1
WINDOW = 30                      # verified claims per speaker considered
PRIOR = 0.5                      # (0+1)/(0+2): the only trust a stranger gets

LAST_Q = 180                     # rounds >= this form the "end of life" read
MIN_DIV = 0.40                   # worst-seed lastq follow-rate divergence
NULL_DIV_MAX = 0.20              # stripped agent, whole life, |divergence|
NULL_TRUST_BAND = (0.35, 0.65)   # pooled posterior must have actually moved
                                 # off nothing: alternating 0.9/0.1 -> ~0.5
MIN_MIGRATE = 0.40               # swap life, lastq, toward newly-truthful
MIN_PRESWAP = 0.40               # swap life, rounds 60..120, pre-swap trust
TRUTH_BAND = (0.75, 1.0)         # realized claim accuracy, truthful voice
LIE_BAND = (0.0, 0.25)           # realized claim accuracy, lying voice
ATTRIB_MIN = 0.90                # ME.9-channel recall of who advised what
N_ATTRIB_Q = 20


def _other(site: str) -> str:
    return SITES[1 - SITES.index(site)]


def _claim_text(name: str, site: str, r: int) -> str:
    return f"{name} claimed the food is at the {site} site in round {r}"


def _finding_text(site: str, r: int) -> str:
    return f"jack searched and found the food at the {site} site in round {r}"


def _parse_round(text: str) -> int:
    return int(text.rsplit(" ", 1)[-1])


def _parse_site(text: str) -> str:
    return text.split(" site", 1)[0].rsplit(" ", 1)[-1]


def _trust(mem, speaker: str | None, window: int = WINDOW) -> float:
    """Posterior that this speaker's next claim is true, computed ONLY from
    the diary: their past claims (ME.9's channel+speaker filter — or every
    heard event when attribution is stripped and there is no speaker to ask
    for) joined to jack's own recorded findings. Laplace (hits+1)/(n+2), so a
    stranger scores PRIOR exactly."""
    findings = {}
    for e in mem.events:
        if e.channel == "did":
            findings[_parse_round(e.text)] = _parse_site(e.text)
    verified = []
    for e in mem.events:
        if e.channel != "heard":
            continue
        if speaker is not None and e.speaker.lower() != speaker:
            continue
        r = _parse_round(e.text)
        if r in findings:
            verified.append((r, _parse_site(e.text) == findings[r]))
    verified.sort()
    tail = [hit for _, hit in verified[-window:]]
    return (sum(tail) + 1) / (len(tail) + 2)


def _live(seed: int, mem_path, stripped: bool = False, swap: bool = False):
    """One life. Returns per-round records the rig scores; the agent inside
    sees only the diary."""
    sys.path.insert(0, str(REPO))
    from EpisodicMemory import EpisodicMemory

    rng_world = random.Random(seed)          # identical across arms
    rng_agent = random.Random(seed + 101)
    mem = EpisodicMemory(path=mem_path)
    t0 = 3_000_000.0
    rows = []
    first_trust = {}
    for r in range(N_ROUNDS):
        advisor = ADVISORS[r % 2]
        truthful_now = (advisor == ADVISORS[0]) ^ (swap and r >= SWAP_ROUND)
        food = rng_world.choice(SITES)
        accurate = rng_world.random() < (TRUTH_P if truthful_now else LIE_P)
        claimed = food if accurate else _other(food)

        p = _trust(mem, None if stripped else advisor)
        if advisor not in first_trust:
            first_trust[advisor] = p
        follow = rng_agent.random() < p

        name = "someone" if stripped else advisor
        mem.record("heard", name, _claim_text(name, claimed, r), t=t0 + r * 60.0)
        # He searches; an empty site sends him to the other, so the round
        # always ends with the true site in the diary — his own experience,
        # not an annotation.
        mem.record("did", "jack", _finding_text(food, r), t=t0 + r * 60.0 + 30.0)
        rows.append({"r": r, "advisor": advisor, "truthful": truthful_now,
                     "accurate": accurate, "claimed": claimed, "follow": follow,
                     "trust": p})
    return mem, rows, first_trust, t0 + N_ROUNDS * 60.0


def _follow_rate(rows, advisor, lo=0, hi=N_ROUNDS):
    picks = [x["follow"] for x in rows if x["advisor"] == advisor and lo <= x["r"] < hi]
    return sum(picks) / max(1, len(picks))


def _truth_rate(rows, advisor, lo=0, hi=N_ROUNDS):
    hits = [x["accurate"] for x in rows if x["advisor"] == advisor and lo <= x["r"] < hi]
    return sum(hits) / max(1, len(hits))


def _measure(seed: int) -> dict:
    tmp = Path(tempfile.mkdtemp())
    truthful, liar = ADVISORS

    # ── the claim life ──
    mem, rows, first, now = _live(seed, tmp / "life.jsonl")
    fr_t = _follow_rate(rows, truthful, LAST_Q)
    fr_l = _follow_rate(rows, liar, LAST_Q)

    # Attribution intact, through ME.9's channels: sampled advice must come
    # back from the diary with the right voice on it.
    rng_q = random.Random(seed + 7)
    qs = rng_q.sample(rows, N_ATTRIB_Q)
    hits = 0
    for x in qs:
        res = mem.what_did_they_tell_me(
            x["advisor"], f"food is at the {x['claimed']} site in round {x['r']}",
            top_k=1, now=now)
        if res:
            got = res[0].event
            hits += (got.speaker.lower() == x["advisor"]
                     and got.text.endswith(f"in round {x['r']}"))
    attrib_acc = hits / N_ATTRIB_Q

    # Verification completeness: every claim joinable to a finding.
    n_findings = sum(1 for e in mem.events if e.channel == "did")

    # ── the stripped null (same world stream, no record of who said what) ──
    mem_n, rows_n, _, _ = _live(seed, tmp / "null.jsonl", stripped=True)
    ndiv = _follow_rate(rows_n, truthful) - _follow_rate(rows_n, liar)
    null_trust_end = rows_n[-1]["trust"]

    return {
        "div_lastq": round(fr_t - fr_l, 4),
        "follow_truthful_lastq": round(fr_t, 4),
        "follow_liar_lastq": round(fr_l, 4),
        "truth_rate_truthful": round(_truth_rate(rows, truthful), 4),
        "truth_rate_liar": round(_truth_rate(rows, liar), 4),
        "first_trust_truthful": first[truthful],
        "first_trust_liar": first[liar],
        "attrib_acc": round(attrib_acc, 4),
        "verify_complete": float(n_findings == N_ROUNDS),
        "null_abs_div": round(abs(ndiv), 4),
        "null_trust_end": round(null_trust_end, 4),
    }


_MEMO: dict = {}
_CTL_MEMO: dict = {}


def _flat(memo: dict, seed: int) -> dict:
    """The seed's own metrics under their plain names (aggregated to mean/std
    across runs, as before) PLUS every seed's value as an explicit
    `<key>_s<seed>` key. The per-seed keys are identical in every run, so
    run_spec's aggregation records them verbatim — that is what makes the
    worst-seed gates in `_check` answerable from the row alone."""
    out = dict(memo[seed])
    for s in SEEDS:
        for k, v in memo[s].items():
            out[f"{k}_s{s}"] = v
    return out


def _experiment(seed: int) -> dict:
    # All seeds are computed (memoized — total work is unchanged) so every
    # run can return the full per-seed key set; see _flat.
    for s in SEEDS:
        if s not in _MEMO:
            _MEMO[s] = _measure(s)
    return _flat(_MEMO, seed)


def _measure_swap(seed: int) -> dict:
    """THE SWAP: roles exchange at SWAP_ROUND. Trust must migrate to the
    newly-truthful voice, and must have been with the originally-truthful one
    before the swap — same mechanism, same code path, nothing reset."""
    tmp = Path(tempfile.mkdtemp())
    a, b = ADVISORS                      # a truthful first half; b second
    _, rows, _, _ = _live(seed, tmp / "swap.jsonl", swap=True)
    return {
        "ctl_div_lastq": round(_follow_rate(rows, b, LAST_Q)
                               - _follow_rate(rows, a, LAST_Q), 4),
        "ctl_div_q2": round(_follow_rate(rows, a, 60, SWAP_ROUND)
                            - _follow_rate(rows, b, 60, SWAP_ROUND), 4),
        "ctl_truth_a_h1": round(_truth_rate(rows, a, 0, SWAP_ROUND), 4),
        "ctl_truth_a_h2": round(_truth_rate(rows, a, SWAP_ROUND), 4),
        "ctl_truth_b_h1": round(_truth_rate(rows, b, 0, SWAP_ROUND), 4),
        "ctl_truth_b_h2": round(_truth_rate(rows, b, SWAP_ROUND), 4),
    }


def _control(seed: int) -> dict:
    for s in SEEDS:
        if s not in _CTL_MEMO:
            _CTL_MEMO[s] = _measure_swap(s)
    return _flat(_CTL_MEMO, seed)


def _in_band(vals, band) -> bool:
    return all(band[0] <= v <= band[1] for v in vals)


# Every per-seed key the gates consult; a row missing one cannot answer them.
_NEED_M = tuple(f"{k}_s{s}" for k in (
    "div_lastq", "verify_complete", "truth_rate_truthful", "truth_rate_liar",
    "first_trust_truthful", "first_trust_liar", "attrib_acc", "null_abs_div",
    "null_trust_end") for s in SEEDS)
_NEED_C = tuple(f"{k}_s{s}" for k in (
    "ctl_div_lastq", "ctl_div_q2", "ctl_truth_a_h1", "ctl_truth_a_h2",
    "ctl_truth_b_h1", "ctl_truth_b_h2") for s in SEEDS)


def _check(m: dict, c: dict):
    """Pure function of the recorded row — no module state, every key a
    static m[...]/c[...] read, all read up front so each is consulted on
    every replay regardless of which gate fires."""
    if any(k not in m for k in _NEED_M) or any(k not in c for k in _NEED_C):
        return Status.VOID          # the record cannot answer the gates
    divs = (m["div_lastq_s0"], m["div_lastq_s1"], m["div_lastq_s2"])
    verify = (m["verify_complete_s0"], m["verify_complete_s1"],
              m["verify_complete_s2"])
    tr_t = (m["truth_rate_truthful_s0"], m["truth_rate_truthful_s1"],
            m["truth_rate_truthful_s2"])
    tr_l = (m["truth_rate_liar_s0"], m["truth_rate_liar_s1"],
            m["truth_rate_liar_s2"])
    ft_t = (m["first_trust_truthful_s0"], m["first_trust_truthful_s1"],
            m["first_trust_truthful_s2"])
    ft_l = (m["first_trust_liar_s0"], m["first_trust_liar_s1"],
            m["first_trust_liar_s2"])
    att = (m["attrib_acc_s0"], m["attrib_acc_s1"], m["attrib_acc_s2"])
    ndiv = (m["null_abs_div_s0"], m["null_abs_div_s1"], m["null_abs_div_s2"])
    ntr = (m["null_trust_end_s0"], m["null_trust_end_s1"],
           m["null_trust_end_s2"])
    ctl_div = (c["ctl_div_lastq_s0"], c["ctl_div_lastq_s1"],
               c["ctl_div_lastq_s2"])
    ctl_q2 = (c["ctl_div_q2_s0"], c["ctl_div_q2_s1"], c["ctl_div_q2_s2"])
    a_h1 = (c["ctl_truth_a_h1_s0"], c["ctl_truth_a_h1_s1"],
            c["ctl_truth_a_h1_s2"])
    a_h2 = (c["ctl_truth_a_h2_s0"], c["ctl_truth_a_h2_s1"],
            c["ctl_truth_a_h2_s2"])
    b_h1 = (c["ctl_truth_b_h1_s0"], c["ctl_truth_b_h1_s1"],
            c["ctl_truth_b_h1_s2"])
    b_h2 = (c["ctl_truth_b_h2_s0"], c["ctl_truth_b_h2_s1"],
            c["ctl_truth_b_h2_s2"])
    # ── rig gates: VOID, not FAIL — a run that could not ask the question ──
    if min(verify) != 1.0:
        return Status.VOID          # a claim with nothing to join to
    if not (_in_band(tr_t, TRUTH_BAND) and _in_band(tr_l, LIE_BAND)
            and _in_band(a_h1, TRUTH_BAND) and _in_band(a_h2, LIE_BAND)
            and _in_band(b_h1, LIE_BAND) and _in_band(b_h2, TRUTH_BAND)):
        return Status.VOID          # an advisor was not systematic
    if not _in_band(ntr, NULL_TRUST_BAND):
        return Status.VOID          # the stripped join never actually ran
    # ── the claim, its guards, the null and the swap, on EVERY seed ──
    mean = sum(divs) / len(divs)
    sd = (sum((d - mean) ** 2 for d in divs) / len(divs)) ** 0.5
    return bool(
        min(divs) >= MIN_DIV
        and mean - 3 * sd > 0
        and all(v == PRIOR for v in ft_t)
        and all(v == PRIOR for v in ft_l)
        and min(att) >= ATTRIB_MIN
        and max(ndiv) <= NULL_DIV_MAX
        and min(ctl_div) >= MIN_MIGRATE
        and min(ctl_q2) >= MIN_PRESWAP)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["LG.02"], _experiment, _check,
                    control_fn=_control, ledger=ledger)


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        # Crash-freedom only: a 6-round toy life, metric VALUES not printed —
        # the gates above were registered from arithmetic, not previews.
        globals().update(N_ROUNDS=6, SWAP_ROUND=3, LAST_Q=4, N_ATTRIB_Q=3)
        tmp = Path(tempfile.mkdtemp())
        for kw in ({}, {"stripped": True}, {"swap": True}):
            _live(0, tmp / f"smoke_{len(kw)}.jsonl", **kw)
        print("smoke: 3 toy lives completed without error")
    else:
        print(json.dumps({"experiment": _measure(0),
                          "control": _measure_swap(0)}, indent=2))
