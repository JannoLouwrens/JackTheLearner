"""ME.9 — he remembers what he hears, says, and does, and WHO is who.

Owner directive, 2026-08-07, verbatim in GOAL.md: "he must also remember what
he hears says and does so when people interact with him." The operative word
is ATTRIBUTED. "What did Ada tell me about the ladder" and "what did I tell
Ada about the ladder" contain the same content words; only provenance
separates them. A memory that answers both from the same pooled soup is a
text-similarity engine wearing a diary's clothes.

Three interleaved speakers (ada, bruno, chika) converse with Jack about
OVERLAPPING topics — every topic is discussed by at least two parties, so
content alone cannot identify the source. Questions come in three forms:
  heard:  "what did <speaker> tell me about <topic>"
  said:   "what did I say about <topic>"
  did:    "what did I do with <topic>"
Correct = top-1 retrieval is an event with the right channel AND speaker AND
topic. >= 80% per channel, per spec.

NULL (must fail the attribution questions): the same events with provenance
STRIPPED (everything pooled, channel/speaker ignored at query time). It can
match topics fine; it cannot know who said what — its attribution accuracy
should collapse toward the share of events that happen to have the right
provenance by luck.

CONTROL (must invert): a swapped-provenance store — Jack's lines relabelled
as the speaker's and vice versa. If accuracy SURVIVES the swap, the test
never used provenance and is broken; the spec demands swapped accuracy fall
below half the true accuracy.

STRENGTHENED 2026-08-30 (Review, Part 2) — WHY THE 1.0000 +- 0.0000 OF
2026-08-08 WAS NOT WORTH WHAT IT LOOKED WORTH.

`what_did_they_tell_me(speaker, cue)` is `recall(cue, channel="heard",
speaker=speaker)`: the provenance is a HARD FILTER applied before scoring.
So of the three conjuncts the scorer above counts —

    got.channel == channel  and  got.speaker == speaker  and  topic in got.text

— the first two are true of every candidate the filter can possibly return.
They cannot fail. The only falsifiable conjunct was the third, and the
recorded metric was therefore "does lexical containment find the right topic
among ~40 pre-filtered events", reported under the name attributed_recall.
Perfect accuracy with zero seed variance was the fingerprint: a claim about
whether interleaving confuses provenance, measured by a quantity that is
partly an identity.

Both existing references have the same shape. The pooled NULL fails because
it has no filter, and the swap CONTROL fails because the filter comes back
EMPTY — each dies of the filter's absence or misdirection, and neither can
tell a good scorer from a coin flip once the filter is in place.

So the missing reference is the third corner: provenance KEPT, scoring
STRIPPED. `trivial=True` answers every question with the most recent event
inside the identical (channel, speaker) filter, scored by the identical
predicate. Its accuracy is what the dict filter alone buys; the headline
minus it (`scoring_margin`) is what retrieval earned. Three conjuncts were
added and nothing was relaxed: the trivial reference must stay at/below
MAX_TRIVIAL, the scoring margin must clear MIN_SCORING_MARGIN, and the
pooled null must now lose by MIN_POOLED_MARGIN rather than merely sit under
the pass bar. (LESSONS.md, 2026-08-30: "a null you can beat is not enough —
check that a trivial reference does not annihilate the task first"; and
2026-08-29: "a control scored on a gate that mentions the control is a
control that cannot fail".)

STRENGTHENED 2026-09-06 (78th audit B2 / Review FTB 3) — ME.11's distractor
question, asked on THIS spec's harness: what does the store answer when the
attribution question has no answer? A censored twin of the life is built with
the identical RNG stream, except that for each askable speaker a seeded set of
topics is withheld from their HEARD channel (the draws still happen, so every
other event is byte-identical). The censored topics remain in the corpus —
other speakers mention them, jack says and does them — so the cue's content
word is KNOWN to the store; only the (speaker, topic, heard) combination is
absent. `what_did_they_tell_me(speaker, topic_cue)` must return NOTHING for
>= MIN_DISTRACTOR_ABST of the censored pairs: answering with that speaker's
nearest other remark is confabulated attribution, the exact failure ME.11
measured at 12.29% on this stack. Aliveness: a censored pair is excluded from
the denominator if the speaker's retained heard events still carry the topic
(impossible by construction, checked anyway) or if the topic word fell out of
the retained corpus entirely (then the cue is ME.1's easy all-unknown case,
not this control); both counts are recorded and the denominator has a floor.
Nothing above moved.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

# The store under test. Undeclared until 2026-09-06 (78th audit B2).
IMPL_DEPS = ["EpisodicMemory.py"]

SPEAKERS = ["ada", "bruno", "chika"]
TOPICS = ["ladder", "kettle", "pond", "compass", "orchard", "lantern",
          "bridge", "shovel", "kite", "cellar", "anchor", "drum"]
DETAILS = ["being cracked", "needing paint", "smelling of smoke", "going missing",
           "being too heavy", "leaning sideways", "sounding hollow",
           "being brand new", "getting stuck", "turning green"]
N_ROUNDS = 40           # per speaker; total events ~ 3 speakers * 3 channels * rounds
N_Q_PER_CHANNEL = 40
MIN_ACC = 0.80
# Added by the Review 2026-08-30, Part 2 (strengthen-only; nothing above moved).
# `1/len(TOPICS)` is what a cue-blind pick inside the filter scores by
# construction; the bar sits a little above it so a mildly-informative trivial
# reference still fails, and the margin gate is what the headline must EARN.
MAX_TRIVIAL = 0.25
MIN_SCORING_MARGIN = 0.50
MIN_POOLED_MARGIN = 0.40
# Added 2026-09-06 (78th audit B2 / Review FTB 3; strengthen-only). ME.1's own
# abstention bar, applied to censored attribution questions on this harness.
N_CENSOR_TOPICS = 5          # per askable speaker -> 15 censored pairs
MIN_DISTRACTOR_EVAL = 9      # aliveness: below this the control has gone quiet
MIN_DISTRACTOR_ABST = 0.95


def _build(seed: int, mem_path, swap: bool = False, censor: dict | None = None):
    """`censor` maps speaker -> topics whose HEARD events are withheld from the
    store. Every draw still happens, so the retained events are byte-identical
    (same text, same t) to the uncensored life — ME.1's held-out discipline."""
    sys.path.insert(0, str(REPO))
    import random
    from EpisodicMemory import EpisodicMemory

    censor = censor or {}
    rng = random.Random(seed)
    mem = EpisodicMemory(path=mem_path)
    t0 = 2_000_000.0
    truth = {"heard": [], "said": [], "did": []}
    i = 0
    for _ in range(N_ROUNDS):
        for sp in SPEAKERS:
            topic, detail = rng.choice(TOPICS), rng.choice(DETAILS)
            heard_sp, said_sp = (("jack", sp) if swap else (sp, "jack"))
            if topic not in censor.get(sp, ()):
                ev = mem.record("heard", heard_sp,
                                f"{sp} mentioned the {topic} {detail}",
                                t=t0 + i * 30.0)
                truth["heard"].append((ev, sp, topic))
            i += 1

            topic2, detail2 = rng.choice(TOPICS), rng.choice(DETAILS)
            ev = mem.record("said", said_sp,
                            f"jack replied that the {topic2} was {detail2}",
                            t=t0 + i * 30.0); i += 1
            truth["said"].append((ev, "jack", topic2))

            topic3 = rng.choice(TOPICS)
            ev = mem.record("did", "jack" if not swap else sp,
                            f"jack inspected the {topic3} afterwards",
                            t=t0 + i * 30.0); i += 1
            truth["did"].append((ev, "jack", topic3))
    return mem, truth, t0 + i * 30.0


def _provenance_of(channel: str, speaker: str):
    """The (channel, speaker) pair the query API filters on for each question
    form. Named once so the trivial reference below cannot drift from it."""
    return (channel, speaker if channel == "heard" else "jack")


def _ask(mem, truth, seed: int, now: float,
         pooled: bool = False, trivial: bool = False) -> dict:
    """Score the three question forms.

    `pooled`   — provenance STRIPPED (the spec's null): one flat recall over
                 the whole log. It cannot answer attribution at all.
    `trivial`  — provenance KEPT, SCORING stripped (added by the Review
                 2026-08-30): inside the very filter the real query uses,
                 answer with the MOST RECENT event and ignore the cue. See
                 the module docstring — this is the reference that makes the
                 headline number mean something.
    """
    import random
    rng = random.Random(seed + 7)
    acc = {}
    for channel in ("heard", "said", "did"):
        qs = rng.sample(truth[channel], min(N_Q_PER_CHANNEL, len(truth[channel])))
        hits = 0
        for ev, speaker, topic in qs:
            if pooled:
                res = mem.recall(f"the {topic}", top_k=1, now=now)
            elif trivial:
                ch, sp = _provenance_of(channel, speaker)
                cands = [e for e in mem.events
                         if e.channel == ch and e.speaker.lower() == sp]
                res = [_Triv(max(cands, key=lambda e: e.t))] if cands else []
            elif channel == "heard":
                res = mem.what_did_they_tell_me(speaker, f"the {topic}", top_k=1, now=now)
            elif channel == "said":
                res = mem.what_did_i_say(f"the {topic}", top_k=1, now=now)
            else:
                res = mem.what_did_i_do(f"the {topic}", top_k=1, now=now)
            if res:
                got = res[0].event
                hits += (got.channel == channel
                         and got.speaker.lower() == speaker
                         and topic in got.text)
        acc[channel] = hits / max(1, len(qs))
    return acc


class _Triv:
    """Minimal stand-in for a Recall so the trivial reference is scored by the
    IDENTICAL three-conjunct predicate above — a separate scoring path is how
    a reference quietly stops being comparable."""
    __slots__ = ("event",)

    def __init__(self, event):
        self.event = event


def _distractor_abstention(seed: int) -> dict:
    """ME.11's control on this harness: censor a seeded set of topics out of
    each askable speaker's heard channel, then ask that speaker about exactly
    those topics. The topic words stay in the corpus (other speakers mention
    them; jack says and does them), so the floor sees KNOWN evidence the
    filtered events lack — answering is confabulated attribution."""
    import random
    from EpisodicMemory import _tokens

    tmp = Path(tempfile.mkdtemp())
    rng = random.Random(seed + 13)
    censor = {s: set(rng.sample(TOPICS, N_CENSOR_TOPICS)) for s in SPEAKERS}
    mem, _, now = _build(seed, tmp / "censored.jsonl", censor=censor)

    vocab: set = set()
    heard_topics = {s: set() for s in SPEAKERS}
    for e in mem.events:
        tok = _tokens(e.text)
        vocab |= tok
        if e.channel == "heard" and e.speaker.lower() in heard_topics:
            heard_topics[e.speaker.lower()] |= tok & set(TOPICS)

    abstained = evaluated = excluded = 0
    for s in SPEAKERS:
        for topic in sorted(censor[s]):
            # A retained heard event carrying the topic makes a hit correct
            # retrieval (impossible by construction, checked anyway); a topic
            # gone from the corpus is ME.1's easy all-unknown cue, not this
            # control. Both leave the denominator, visibly.
            if topic in heard_topics[s] or topic not in vocab:
                excluded += 1
                continue
            evaluated += 1
            abstained += not mem.what_did_they_tell_me(
                s, f"the {topic}", top_k=1, now=now)
    return {
        "distractor_abstention": round(abstained / evaluated, 4) if evaluated
                                 else 0.0,
        "distractor_evaluated": evaluated,
        "distractor_excluded": excluded,
    }


def _experiment(seed: int) -> dict:
    tmp = Path(tempfile.mkdtemp())
    mem, truth, now = _build(seed, tmp / "life.jsonl")
    acc = _ask(mem, truth, seed, now)
    pooled = _ask(mem, truth, seed, now, pooled=True)
    triv = _ask(mem, truth, seed, now, trivial=True)
    dis = _distractor_abstention(seed)
    return {
        **dis,
        "events": len(mem),
        "acc_heard": round(acc["heard"], 4),
        "acc_said": round(acc["said"], 4),
        "acc_did": round(acc["did"], 4),
        "pooled_null_worst": round(min(pooled.values()), 4),
        "pooled_null_mean": round(sum(pooled.values()) / 3, 4),
        # Provenance kept, scoring stripped. The gap between this and the
        # headline is everything the retrieval SCORER contributes; the
        # headline minus this is the only part the dict filter cannot claim.
        "filtered_recency_worst": round(max(triv.values()), 4),
        "filtered_recency_mean": round(sum(triv.values()) / 3, 4),
        "scoring_margin": round(sum(acc.values()) / 3 - sum(triv.values()) / 3, 4),
    }


def _control(seed: int) -> dict:
    """Swapped provenance must gut the accuracy. Same content, wrong authors."""
    tmp = Path(tempfile.mkdtemp())
    mem, truth, now = _build(seed, tmp / "swapped.jsonl", swap=True)
    acc = _ask(mem, truth, seed, now)
    return {"swapped_mean_acc": round(sum(acc.values()) / 3, 4)}


def _check(m: dict, c: dict) -> bool:
    true_mean = (m["acc_heard"] + m["acc_said"] + m["acc_did"]) / 3
    return (m["acc_heard"] >= MIN_ACC
            and m["acc_said"] >= MIN_ACC
            and m["acc_did"] >= MIN_ACC
            and m["pooled_null_mean"] < MIN_ACC
            and c["swapped_mean_acc"] < true_mean / 2
            # Added 2026-08-30 (Review, Part 2). Strengthen-only: every gate
            # above is unchanged and these are additional conjuncts.
            and m["filtered_recency_worst"] <= MAX_TRIVIAL
            and m["scoring_margin"] >= MIN_SCORING_MARGIN
            and true_mean - m["pooled_null_mean"] >= MIN_POOLED_MARGIN
            # Added 2026-09-06 (78th audit B2 / Review FTB 3; strengthen-only):
            # the censored-attribution abstention question, with an aliveness
            # floor so a control that stops evaluating cannot pass by silence.
            and m["distractor_evaluated"] >= MIN_DISTRACTOR_EVAL
            and m["distractor_abstention"] >= MIN_DISTRACTOR_ABST)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["ME.9"], _experiment, _check, control_fn=_control, ledger=ledger)
