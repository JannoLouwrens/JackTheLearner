"""EpisodicMemory — the attributed event log Jack lives out of.

GOAL.md, Memory section: when people interact with him he remembers what he
HEARD, what he SAID, and what he DID — attributed, per person — while the same
lived experience also distils into general skill. This module is the first
half of that sentence: the diary. The distillation half lives in the training
pipeline, and ME.10 holds the two apart (wipe the diary, the skill survives;
revert the weights, the diary survives).

Design, and where each choice comes from (docs/research/MEMORY.md):

  Append-only JSONL on disk   GOAL.md demands memory that "persists on disk,
                              inspectable, across restarts". A human can read
                              the file. Nothing is ever rewritten in place;
                              the log is the ground truth of what happened.

  Provenance on every event   channel ("heard"/"said"/"did"/"saw") + speaker.
                              "What did I tell you" and "what did you tell me"
                              differ ONLY in provenance — a memory that drops
                              it cannot answer either honestly (ME.9's
                              swapped-provenance control exists to catch
                              exactly that).

  recency x importance x      Park et al. 2023 (Generative Agents) scoring.
  similarity retrieval        Any single term fails alone: recency-only is
                              ME.1's null baseline, similarity-only forgets
                              that yesterday's identical breakfast is not
                              last year's.

  Abstention threshold        A query nothing matches must return NOTHING.
                              Confabulating the nearest neighbour of a
                              fabricated event is worse than silence — it is
                              false memory presented with confidence (ME.1's
                              control kills it).

No LLM, no torch: retrieval must work on this CPU-only box with zero models
loaded, so similarity is lexical (content-word Jaccard with an IDF-style
down-weighting of common words). An embedding hook can be layered on later;
the CONTRACT (score, abstain, attribute) is what the ladder tests.
"""
from __future__ import annotations

import json
import math
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, List, Optional

CHANNELS = ("heard", "said", "did", "saw")

# Words too common to identify an event. Kept tiny and boring on purpose —
# a real IDF table would need a corpus, and the point is only to stop "the"
# from matching everything.
_STOP = frozenset(
    "a an the and or but of to in on at for with by is was are were be been "
    "i you he she it we they me him her us them my your his its our their "
    "this that these those there here about into over under again then than "
    "what who when where why how did do does done doing said says tell told "
    "jack".split()
)


def _tokens(text: str) -> set:
    norm = unicodedata.normalize("NFKD", text.lower())
    words = "".join(c if c.isalnum() else " " for c in norm).split()
    return {w for w in words if w not in _STOP and len(w) > 1}


@dataclass
class Event:
    t: float                    # wall-clock seconds; sim time may ride along in meta
    channel: str                # heard | said | did | saw
    speaker: str                # who produced it: a person's name, or "jack"
    text: str                   # what happened, plainly
    importance: float = 1.0     # 0..10; how much it mattered when it happened
    meta: dict = field(default_factory=dict)
    eid: int = -1               # assigned by the store; position in the log

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class Recall:
    event: Event
    score: float
    similarity: float


class EpisodicMemory:
    """The event log with scored, abstaining, provenance-aware retrieval."""

    def __init__(self, path: Optional[Path] = None,
                 half_life_s: float = 6 * 3600.0,
                 w_recency: float = 1.0, w_importance: float = 1.0,
                 w_similarity: float = 10.0,
                 abstain_below: float = 0.34):
        """abstain_below is a SIMILARITY floor, not a score floor: an event can
        be recent and important and still be the wrong answer. 0.34 means at
        least roughly a third of the query's content words must appear in the
        event before any recency or importance is allowed to speak.

        w_similarity is deliberately an order of magnitude above the other
        weights: content match is the PRIMARY KEY, recency and importance are
        tie-breakers among equally matching events. With the weights near
        parity, a fresher event matching 2 of 3 cue words outvoted the right
        event matching all 3 — measured on ME.1, recall fell from 0.85 to
        0.70 exactly this way. At 10x, a one-word similarity gap (>= 0.25 of
        a typical cue) is worth more than the entire recency+importance range
        (2.0), so the wrong-but-recent event can no longer win.
        """
        self.path = Path(path) if path else None
        self.half_life_s = half_life_s
        self.w = (w_recency, w_importance, w_similarity)
        self.abstain_below = abstain_below
        self.events: List[Event] = []
        self._tok: List[set] = []
        if self.path and self.path.exists():
            self._load()

    # ── the log ─────────────────────────────────────────────────────────
    def record(self, channel: str, speaker: str, text: str,
               importance: float = 1.0, t: Optional[float] = None,
               meta: Optional[dict] = None) -> Event:
        if channel not in CHANNELS:
            raise ValueError(f"channel {channel!r} not in {CHANNELS}")
        ev = Event(t=float(t if t is not None else time.time()),
                   channel=channel, speaker=speaker, text=text,
                   importance=float(importance), meta=meta or {},
                   eid=len(self.events))
        self.events.append(ev)
        self._tok.append(_tokens(text))
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(ev.to_json() + "\n")
        return ev

    def _load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            d.pop("eid", None)
            ev = Event(**d)
            ev.eid = len(self.events)
            self.events.append(ev)
            self._tok.append(_tokens(ev.text))

    # ── retrieval ───────────────────────────────────────────────────────
    def recall(self, query: str, top_k: int = 3,
               channel: Optional[str] = None,
               speaker: Optional[str] = None,
               now: Optional[float] = None) -> List[Recall]:
        """Top-k events by recency x importance x similarity.

        channel/speaker restrict PROVENANCE, which is how attribution
        questions are asked: "what did Ada tell me" is
        recall(q, channel="heard", speaker="Ada"); "what did I say" is
        recall(q, channel="said", speaker="jack"). Returns [] when nothing
        clears the similarity floor — abstention is a first-class answer.
        """
        if not self.events:
            return []
        now = float(now if now is not None else time.time())
        q = _tokens(query)
        if not q:
            return []
        wr, wi, ws = self.w
        out: List[Recall] = []
        for ev, tok in zip(self.events, self._tok):
            if channel is not None and ev.channel != channel:
                continue
            if speaker is not None and ev.speaker.lower() != speaker.lower():
                continue
            inter = len(q & tok)
            if inter == 0:
                continue
            # CONTAINMENT, not Jaccard: how much of the QUERY the event covers.
            # A cue is a fragment of the event it asks about — "the ladder"
            # against "ada mentioned the ladder being cracked" is a full match
            # of the question, not a 20% match of the sentence. Jaccard made
            # every short attribution question abstain (ME.9 measured 0.0
            # across the board); containment keeps the abstention floor
            # meaningful for fabricated content while letting terse, natural
            # questions through.
            sim = inter / len(q)
            if sim < self.abstain_below:
                continue
            age = max(0.0, now - ev.t)
            recency = math.pow(0.5, age / self.half_life_s)
            importance = min(max(ev.importance, 0.0), 10.0) / 10.0
            score = wr * recency + wi * importance + ws * sim
            out.append(Recall(ev, score, sim))
        out.sort(key=lambda r: r.score, reverse=True)
        return out[:top_k]

    # ── attribution helpers (ME.9's vocabulary) ─────────────────────────
    def what_did_they_tell_me(self, speaker: str, query: str,
                              top_k: int = 3, now: Optional[float] = None):
        return self.recall(query, top_k, channel="heard", speaker=speaker, now=now)

    def what_did_i_say(self, query: str, top_k: int = 3,
                       now: Optional[float] = None):
        return self.recall(query, top_k, channel="said", speaker="jack", now=now)

    def what_did_i_do(self, query: str, top_k: int = 3,
                      now: Optional[float] = None):
        return self.recall(query, top_k, channel="did", speaker="jack", now=now)

    def __len__(self) -> int:
        return len(self.events)
