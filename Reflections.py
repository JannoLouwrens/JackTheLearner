"""Reflections — consolidated beliefs distilled from the episodic log.

GOAL.md, Memory section: keeping the record and learning from it are separate
stores. EpisodicMemory is the record; this module is the first consolidation
consumer of it (docs/research/MEMORY.md 2.2's rule: EVERY episodic table must
have a consolidation consumer). It answers the question raw retrieval cannot:
not "what happened that one time" but "what USUALLY happens" — the Generative
Agents reflection-tree idea (2304.03442) reduced to what runs on this box.

Design, and why:

  Statistical, no LLM        The box is CPU-only with zero models loaded.
                             Consolidation is per-speaker content-token
                             frequency over the whole log: "ada often involved
                             with pond (52 of 244 events)". Crude, but it
                             carries exactly the information top-k raw
                             retrieval cannot — a count over ALL events, in
                             one line's worth of tokens. ME.3 tests precisely
                             that trade.

  Counts live in the text    A reflection states its own evidence ("52 of 244
                             events"). Any reader — human or mechanical — can
                             weigh it without trusting hidden state, and the
                             ledger's reader parses the same words a person
                             would read.

  Source-linked              Every reflection keeps the eids it was distilled
                             from. A belief that cannot say which events back
                             it is a rumour, and rumours are how the wrong
                             agent's reflections sneak in (ME.3's control).

  Re-derived, not patched    consolidate() rewrites the store from the log
                             each time. The log is ground truth; beliefs are a
                             cache of it. Atomic tmp+replace write, the same
                             pattern the ledger uses, so a SIGKILL cannot
                             leave half a belief system on disk.

  Same retrieval contract    recall() uses the same containment similarity and
                             abstention floor as EpisodicMemory — a query
                             nothing matches returns NOTHING. Ties break on
                             evidence count: the better-supported belief wins.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

from EpisodicMemory import EpisodicMemory, _tokens


@dataclass
class Reflection:
    speaker: str                # whose habit this is
    token: str                  # the recurring content word
    count: int                  # events of theirs containing it
    n_events: int               # their total events at consolidation time
    text: str                   # the belief, stated plainly with its evidence
    sources: List[int] = field(default_factory=list)   # eids it distils
    t: float = 0.0              # consolidation time

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class Reflections:
    """Per-speaker habit beliefs, re-derived from an EpisodicMemory log."""

    MAX_SOURCES = 20            # keep the link inspectable, not exhaustive

    def __init__(self, path: Optional[Path] = None,
                 abstain_below: float = 0.34):
        self.path = Path(path) if path else None
        self.abstain_below = abstain_below
        self.beliefs: List[Reflection] = []
        self._tok: List[set] = []
        if self.path and self.path.exists():
            self._load()

    # ── consolidation ───────────────────────────────────────────────────
    def consolidate(self, mem: EpisodicMemory, top_m: int = 16,
                    min_count: int = 3, now: float = 0.0) -> List[Reflection]:
        """Distil the whole log into at most top_m beliefs per speaker.

        A token becomes a belief when it recurs (>= min_count) in one
        speaker's events. min_count guards against enshrining a coincidence;
        top_m keeps the store a summary rather than a second log.
        """
        by_speaker: dict = {}
        for ev, tok in zip(mem.events, mem._tok):
            n, counts, srcs = by_speaker.setdefault(ev.speaker,
                                                    [0, Counter(), {}])
            by_speaker[ev.speaker][0] = n + 1
            for w in tok:
                counts[w] += 1
                srcs.setdefault(w, []).append(ev.eid)

        self.beliefs, self._tok = [], []
        for speaker in sorted(by_speaker):
            n_events, counts, srcs = by_speaker[speaker]
            for token, count in counts.most_common(top_m):
                if count < min_count:
                    break
                text = (f"{speaker} often involved with {token} "
                        f"({count} of {n_events} events)")
                self.beliefs.append(Reflection(
                    speaker=speaker, token=token, count=count,
                    n_events=n_events, text=text,
                    sources=srcs[token][:self.MAX_SOURCES], t=now))
                self._tok.append(_tokens(text))
        self._write()
        return self.beliefs

    # ── persistence ─────────────────────────────────────────────────────
    def _write(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for b in self.beliefs:
                fh.write(b.to_json() + "\n")
        os.replace(tmp, self.path)

    def _load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            b = Reflection(**json.loads(line))
            self.beliefs.append(b)
            self._tok.append(_tokens(b.text))

    # ── retrieval ───────────────────────────────────────────────────────
    def recall(self, query: str, top_k: int = 5) -> List[Reflection]:
        """Beliefs by containment similarity, ties broken by evidence count.

        Same abstention contract as the event log: below the floor, silence.
        """
        q = _tokens(query)
        if not q:
            return []
        scored = []
        for b, tok in zip(self.beliefs, self._tok):
            sim = len(q & tok) / len(q)
            if sim < self.abstain_below:
                continue
            scored.append((sim, b.count, b))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [b for _, _, b in scored[:top_k]]

    def __len__(self) -> int:
        return len(self.beliefs)
