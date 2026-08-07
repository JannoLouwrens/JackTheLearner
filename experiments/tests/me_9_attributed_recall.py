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
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

SPEAKERS = ["ada", "bruno", "chika"]
TOPICS = ["ladder", "kettle", "pond", "compass", "orchard", "lantern",
          "bridge", "shovel", "kite", "cellar", "anchor", "drum"]
DETAILS = ["being cracked", "needing paint", "smelling of smoke", "going missing",
           "being too heavy", "leaning sideways", "sounding hollow",
           "being brand new", "getting stuck", "turning green"]
N_ROUNDS = 40           # per speaker; total events ~ 3 speakers * 3 channels * rounds
N_Q_PER_CHANNEL = 40
MIN_ACC = 0.80


def _build(seed: int, mem_path, swap: bool = False):
    sys.path.insert(0, str(REPO))
    import random
    from EpisodicMemory import EpisodicMemory

    rng = random.Random(seed)
    mem = EpisodicMemory(path=mem_path)
    t0 = 2_000_000.0
    truth = {"heard": [], "said": [], "did": []}
    i = 0
    for _ in range(N_ROUNDS):
        for sp in SPEAKERS:
            topic, detail = rng.choice(TOPICS), rng.choice(DETAILS)
            heard_sp, said_sp = (("jack", sp) if swap else (sp, "jack"))
            ev = mem.record("heard", heard_sp,
                            f"{sp} mentioned the {topic} {detail}",
                            t=t0 + i * 30.0); i += 1
            truth["heard"].append((ev, sp, topic))

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


def _ask(mem, truth, seed: int, now: float, pooled: bool = False) -> dict:
    import random
    rng = random.Random(seed + 7)
    acc = {}
    for channel in ("heard", "said", "did"):
        qs = rng.sample(truth[channel], min(N_Q_PER_CHANNEL, len(truth[channel])))
        hits = 0
        for ev, speaker, topic in qs:
            if pooled:
                res = mem.recall(f"the {topic}", top_k=1, now=now)
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


def _experiment(seed: int) -> dict:
    tmp = Path(tempfile.mkdtemp())
    mem, truth, now = _build(seed, tmp / "life.jsonl")
    acc = _ask(mem, truth, seed, now)
    pooled = _ask(mem, truth, seed, now, pooled=True)
    return {
        "events": len(mem),
        "acc_heard": round(acc["heard"], 4),
        "acc_said": round(acc["said"], 4),
        "acc_did": round(acc["did"], 4),
        "pooled_null_worst": round(min(pooled.values()), 4),
        "pooled_null_mean": round(sum(pooled.values()) / 3, 4),
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
            and c["swapped_mean_acc"] < true_mean / 2)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["ME.9"], _experiment, _check, control_fn=_control, ledger=ledger)
