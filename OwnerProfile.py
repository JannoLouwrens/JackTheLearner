"""OwnerProfile — what Jack knows about his owner, in a file.

GOAL.md, Memory section: "He remembers the ladder. He remembers you." The
EpisodicMemory diary answers WHAT HAPPENED; this module answers WHAT THE
OWNER WANTS — the distilled, current-state view of stated preferences that a
raw event log cannot give you without re-reading a lifetime of chatter.

The ME.2 contract, verbatim from the spec:
  - a preference stated once is honoured NEXT SESSION (so it must survive a
    process restart — the store is profile.json on disk, human-readable);
  - a later contradiction SUPERSEDES it (the current value is the latest
    statement; the history is kept, append-style, for inspection);
  - WIPE the file and the knowledge must be gone (the control: adherence
    falls to a no-memory agent's base rate, proving the memory lives in the
    file and not in weights, code, or an in-process cache).

Extraction is deliberately modest: a handful of first-person preference
shapes ("i prefer the X Y", "from now on make it the X Y", ...) parsed with
regexes. No LLM — this box runs CPU-only with zero models loaded, and what
the ladder tests is the CONTRACT (persist, supersede, vanish-on-wipe), not
open-domain language understanding. A model-backed extractor can replace
`parse_preference` later without touching the contract. Statements that do
not match a shape are ignored — mentioning a teal mug is not preferring it.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

# First-person preference shapes. Each yields (value, topic).
# Ordering matters only for readability; matches are mutually exclusive in
# practice because all require the "i ..." stem.
_SHAPES = [
    re.compile(r"\bi (?:really )?prefer the (\w+) (\w+)\b"),
    re.compile(r"\bi'?d rather have the (\w+) (\w+)\b"),
    re.compile(r"\bfrom now on,? (?:make it|use) the (\w+) (\w+)\b"),
    re.compile(r"\bactually,? i want the (\w+) (\w+) instead\b"),
    re.compile(r"\bi like the (\w+) (\w+) best\b"),
]


def parse_preference(text: str) -> Optional[Tuple[str, str]]:
    """Return (topic, value) if the utterance states a preference, else None."""
    low = text.lower()
    for pat in _SHAPES:
        m = pat.search(low)
        if m:
            value, topic = m.group(1), m.group(2)
            return topic, value
    return None


class OwnerProfile:
    """Owner preferences: latest statement wins, everything lives in a file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.topics: dict = {}
        if self.path.exists():
            self.topics = json.loads(self.path.read_text(encoding="utf-8"))

    # ── ingestion ───────────────────────────────────────────────────────
    def ingest(self, text: str, t: float) -> Optional[Tuple[str, str]]:
        """Feed one OWNER utterance. Caller is responsible for attribution —
        only the owner's own words belong here (EpisodicMemory's speaker
        field is how you know). Returns (topic, value) if a preference was
        extracted and recorded, else None."""
        parsed = parse_preference(text)
        if parsed is None:
            return None
        topic, value = parsed
        entry = self.topics.setdefault(topic, {"value": None, "t": -1.0,
                                               "history": []})
        # Supersede: strictly-later statements win; ties keep the incumbent
        # (a re-statement of the same value is the common tie case anyway).
        if t > entry["t"]:
            entry["value"], entry["t"] = value, float(t)
        entry["history"].append({"value": value, "t": float(t)})
        self._save()
        return topic, value

    # ── queries ─────────────────────────────────────────────────────────
    def preference(self, topic: str) -> Optional[str]:
        entry = self.topics.get(topic)
        return entry["value"] if entry else None

    def choose(self, topic: str, options: List[str], rng) -> str:
        """Pick among options: the stated preference if one applies, else a
        uniform draw — which IS the no-memory base rate the ledger's wipe
        control measures against."""
        pref = self.preference(topic)
        if pref is not None and pref in options:
            return pref
        return rng.choice(options)

    # ── persistence ─────────────────────────────────────────────────────
    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.topics, indent=1, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, self.path)

    def __len__(self) -> int:
        return len(self.topics)
