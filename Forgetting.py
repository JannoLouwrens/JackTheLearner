"""Forgetting — the bounded working store that keeps what matters.

GOAL.md, Memory section: what Jack learned yesterday persists on disk,
inspectable, across restarts. But a store that only ever grows is not a
memory system — docs/research/MEMORY.md 4.1 collects the evidence that
full-context reading LOSES to selective retrieval at only ~50 sessions.
So something must be forgotten, and WHICH things are forgotten is a policy
with a failure mode: FIFO forgets by age and silently discards the old fact
you rely on every day. ME.4 is the ladder test that this module's policy
beats that null.

EpisodicMemory stays the append-only diary — ground truth, never rewritten.
This module is the complementary BOUNDED working set: a fixed budget of
traces competing to stay, under the policy MEMORY.md 4.2 pre-registers:

  Ebbinghaus decay           MemoryBank (2305.10250): a trace's retention
                             decays exponentially with time since last
                             access, with a stability that GROWS with each
                             recall — strength(now) =
                             exp(-(now - last_access) / (tau * S)),
                             S = 1 + n_recalls. A trace recalled twenty
                             times decays twenty times slower.

  Reinforce-on-recall        recall(reinforce=True) bumps the winning
                             trace's stability and refreshes its access
                             time. Use IS the signal for what matters; no
                             curator assigns retention by hand.

  Supersede, never edit      Zep-style: a new trace recorded under the same
                             key INVALIDATES prior traces with that key.
                             The old trace is not rewritten — it is marked
                             invalid, excluded from recall, and becomes the
                             first candidate for eviction. This must
                             override reinforcement: a fact repeated a
                             hundred times is still wrong the moment its
                             owner corrects it (ME.2 learned this for
                             preferences; ME.4's control re-checks it here,
                             because reinforcement actively fights the
                             update — the stale trace is the strong one).

  Eviction at fixed budget   When the store exceeds its budget: superseded
                             traces go first, then the weakest strength,
                             oldest breaking ties. FIFO (`policy="fifo"`)
                             is the same store evicting purely by age — it
                             exists to be the honest null, not to be used.

  Similarity outranks        Retrieval sorts by (similarity, strength):
  strength                   containment similarity is the primary key,
                             reinforced strength only breaks ties among
                             equally-matching traces. EpisodicMemory
                             measured why (ME.1): with strength allowed to
                             outvote content, a wrong-but-familiar trace
                             beats the right one. Same abstention floor as
                             the diary: no match, no answer.

No LLM, no torch: the contract (decay, reinforce, supersede, evict, abstain)
is what the ladder tests, on this CPU-only box with zero models loaded.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from EpisodicMemory import _tokens

POLICIES = ("ebbinghaus", "fifo")


@dataclass
class Trace:
    t: float                    # when it was recorded
    text: str                   # the content, plainly
    key: Optional[str] = None   # identity for supersession ("ada:spyglass")
    importance: float = 1.0
    valid: bool = True          # False once superseded — excluded from recall
    n_recalls: int = 0
    last_access: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class ForgettingMemory:
    """A fixed-budget trace store with a use-it-or-lose-it retention policy."""

    def __init__(self, budget: Optional[int] = None,
                 policy: str = "ebbinghaus", supersede: bool = True,
                 tau_s: float = 3600.0, path: Optional[Path] = None,
                 abstain_below: float = 0.34):
        if policy not in POLICIES:
            raise ValueError(f"policy {policy!r} not in {POLICIES}")
        self.budget = budget
        self.policy = policy
        self.supersede = supersede
        self.tau_s = tau_s
        self.path = Path(path) if path else None
        self.abstain_below = abstain_below
        self.traces: List[Trace] = []
        self._tok: List[set] = []
        if self.path and self.path.exists():
            self._load()

    # ── retention ───────────────────────────────────────────────────────
    def strength(self, tr: Trace, now: float) -> float:
        """Ebbinghaus retention: exp(-age_since_access / (tau * S)),
        S = 1 + n_recalls. Fresh or recently-recalled -> ~1; a never-recalled
        trace an afternoon old -> ~0."""
        s = 1.0 + tr.n_recalls
        return math.exp(-max(0.0, now - tr.last_access) / (self.tau_s * s))

    # ── the store ───────────────────────────────────────────────────────
    def record(self, text: str, t: float, key: Optional[str] = None,
               importance: float = 1.0) -> Trace:
        if self.supersede and key is not None:
            for tr in self.traces:
                if tr.valid and tr.key == key:
                    tr.valid = False          # invalidated, not rewritten
        tr = Trace(t=float(t), text=text, key=key,
                   importance=float(importance), last_access=float(t))
        self.traces.append(tr)
        self._tok.append(_tokens(text))
        if self.budget is not None:
            while len(self.traces) > self.budget:
                self._evict(now=float(t))
        return tr

    def _evict(self, now: float) -> None:
        if self.policy == "fifo":
            idx = 0                            # pure insertion order: the null
        else:
            # Superseded first (valid=False sorts before True), then the
            # weakest retention, oldest breaking ties.
            idx = min(range(len(self.traces)),
                      key=lambda i: (self.traces[i].valid,
                                     self.strength(self.traces[i], now),
                                     self.traces[i].t))
        del self.traces[idx]
        del self._tok[idx]

    # ── retrieval ───────────────────────────────────────────────────────
    def recall(self, query: str, now: float, top_k: int = 1,
               reinforce: bool = False) -> List[Trace]:
        """Valid traces ranked by (containment similarity, reinforced
        strength). Returns [] below the abstention floor. reinforce=True
        bumps the winner — recall is what keeps a trace alive."""
        q = _tokens(query)
        if not q or not self.traces:
            return []
        scored = []
        for i, (tr, tok) in enumerate(zip(self.traces, self._tok)):
            if not tr.valid:
                continue
            inter = len(q & tok)
            if inter == 0:
                continue
            sim = inter / len(q)
            if sim < self.abstain_below:
                continue
            rank = (1.0 + tr.n_recalls) * self.strength(tr, now)
            scored.append((sim, rank, i))
        scored.sort(reverse=True)
        if reinforce and scored:
            winner = self.traces[scored[0][2]]
            winner.n_recalls += 1
            winner.last_access = float(now)
        return [self.traces[i] for _, _, i in scored[:top_k]]

    # ── persistence ─────────────────────────────────────────────────────
    def save(self) -> None:
        """Snapshot the live store. Eviction and reinforcement mutate state,
        so unlike the diary this is a rewritten snapshot — atomic tmp+replace,
        the same pattern the ledger uses."""
        if not self.path:
            raise ValueError("no path configured")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for tr in self.traces:
                fh.write(tr.to_json() + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    def _load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            tr = Trace(**json.loads(line))
            self.traces.append(tr)
            self._tok.append(_tokens(tr.text))

    def __len__(self) -> int:
        return len(self.traces)
