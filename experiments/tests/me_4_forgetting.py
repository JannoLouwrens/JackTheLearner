"""ME.4 — forgetting keeps what matters: decay+reinforce+supersede vs FIFO.

The claim (docs/research/MEMORY.md 4.2; MemoryBank 2305.10250, Zep): at a
FIXED store budget, evicting by Ebbinghaus retention — where each recall
grows a trace's stability — keeps frequently-referenced OLD facts that FIFO
throws away, and supersession keeps updated facts CURRENT even when the
stale version is the heavily-reinforced one.

Setup: a 1,200-event stream against a 150-trace budget. 24 durable facts
("ada stores the spyglass in the cellar") are stated in the first 200
events and then referenced (recalled) roughly every 50 events for the rest
of the life — old by position, alive by use. Everything else is one-off
filler that is never referenced again. 8 of the facts are UPDATED mid-life
(same key, new value) after dozens of reinforcing recalls have made the old
version the strongest trace in the store — the exact case where
reinforcement fights the truth.

Four stores ingest the identical stream (facts recorded with their key,
filler without; references reinforce where the policy allows):

  smart     : budget 150, ebbinghaus + reinforce + supersede. Answers are
              read back from a snapshot RELOADED FROM DISK.
  fifo      : budget 150, evict-oldest. The null.
  unbounded : no eviction — the ceiling; the gap to it is what forgetting
              costs.
  no-supersede (control): smart minus supersession only.

Questions, answered by top-1 recall + extracting the place named in the
returned trace (abstention counts as wrong):

  retention set: the 16 never-updated facts. FIFO evicted their statements
              ~850 events ago, so it must collapse toward zero; smart must
              keep them. retention_vs_fifo is this gap.
  update set : the 8 updated facts — the CURRENT value must come back.

CONTROL (must fail): the no-supersede store answers the update questions.
Its stale, reinforced traces outrank the fresh corrections — post-update
references keep reinforcing the WRONG one, a self-deepening rut — so its
stale-answer rate must be >=0.5. If it is not, the updates never actually
conflicted with anything and the update questions test nothing.
"""
from __future__ import annotations

import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

T_EVENTS = 1200
BUDGET = 150
DT = 60.0                     # one event a minute; a 20-hour life
N_FACTS = 24
N_UPDATED = 8
STATE_WINDOW = (0, 200)       # facts stated early -> FIFO must lose them
UPDATE_WINDOW = (550, 850)    # after ~dozens of reinforcements of the old value
REF_PERIOD = 50               # a fact is referenced ~every 50 events
JITTER = 10

MIN_RETENTION = 0.90          # smart keeps referenced old facts
MIN_GAP = 0.40                # retention_vs_fifo (expected ~0.9: FIFO -> ~0)
MIN_UPDATE = 0.90             # smart answers updates with the CURRENT value
MAX_STALE = 0.10
MIN_CONTROL_STALE = 0.50      # no-supersede MUST answer stale, else no conflict

PEOPLE = ["ada", "bruno", "chika", "dora"]
# Fact items are disjoint from filler objects so ME.4 measures the retention
# policy, not retrieval disambiguation — ME.1 already owns that claim.
FACT_ITEMS = ["spyglass", "chisel", "flute", "sextant", "quill", "abacus",
              "trowel", "mallet", "sickle", "bellows", "stirrup", "gimlet",
              "awl", "adze", "churn", "loom", "crucible", "tongs", "vise",
              "plumb", "gouge", "rasp", "spindle", "cleat"]
PLACES = ["pond", "ramp", "platform", "meadow", "shed", "gate", "bridge",
          "cellar", "orchard", "quarry", "dock", "tower", "trail", "garden"]
OBJECTS = ["kettle", "ladder", "apple", "lantern", "hammer", "compass",
           "bucket", "rope", "mirror", "whistle", "anchor", "basket"]
ACTIONS = ["carried", "dropped", "painted", "repaired", "buried", "balanced",
           "measured", "cleaned", "stacked", "traded", "borrowed", "hid"]
COLOURS = ["copper", "crimson", "olive", "violet", "amber", "teal", "ivory",
           "slate", "coral", "bronze"]


def _cue(f) -> str:
    return f"{f['person']} stores {f['item']}"


def _build_life(seed: int, stores: dict, reinforce: set):
    """Stream the identical life into every store. Returns the facts (with
    current and superseded values) and end-of-life time."""
    import random
    rng = random.Random(seed)

    facts = [{"person": rng.choice(PEOPLE), "item": FACT_ITEMS[i],
              "value": rng.choice(PLACES), "old": None}
             for i in range(N_FACTS)]
    state_slots = rng.sample(range(*STATE_WINDOW), N_FACTS)
    stmt = {slot: i for i, slot in enumerate(state_slots)}
    upd_slots = rng.sample(range(*UPDATE_WINDOW), N_UPDATED)
    upd = {slot: i for slot, i in zip(upd_slots,
                                      rng.sample(range(N_FACTS), N_UPDATED))}

    refs = defaultdict(list)
    for i in range(N_FACTS):
        nxt = state_slots[i] + REF_PERIOD + rng.randint(-JITTER, JITTER)
        while nxt < T_EVENTS:
            refs[nxt].append(i)
            nxt += REF_PERIOD + rng.randint(-JITTER, JITTER)

    t0 = 1_000_000.0
    for step in range(T_EVENTS):
        now = t0 + step * DT
        if step in stmt or step in upd:
            i = stmt.get(step, upd.get(step))
            f = facts[i]
            if step in upd:
                f["old"] = f["value"]
                f["value"] = rng.choice([p for p in PLACES
                                         if p != f["value"]])
            text = f"{f['person']} stores the {f['item']} in the {f['value']}"
            for st in stores.values():
                st.record(text, t=now, key=f"{f['person']}:{f['item']}")
        else:
            text = (f"{rng.choice(PEOPLE)} {rng.choice(ACTIONS)} the "
                    f"{rng.choice(COLOURS)} {rng.choice(OBJECTS)} near the "
                    f"{rng.choice(PLACES)}")
            for st in stores.values():
                st.record(text, t=now)
        for i in refs.get(step, ()):
            for name in reinforce:
                stores[name].recall(_cue(facts[i]), now=now, reinforce=True)
    return facts, t0 + T_EVENTS * DT


def _answer(store, facts, indices, now):
    """Accuracy and stale rate over one question set: top-1 recall, extract
    the place the trace names. Abstention counts as wrong."""
    sys.path.insert(0, str(REPO))
    from EpisodicMemory import _tokens
    hits = stale = 0
    for i in indices:
        f = facts[i]
        got = store.recall(_cue(f), now=now, top_k=1)
        if not got:
            continue
        tok = _tokens(got[0].text)
        val = next((p for p in PLACES if p in tok), None)
        if val == f["value"]:
            hits += 1
        elif f["old"] is not None and val == f["old"]:
            stale += 1
    n = len(indices)
    return hits / n, stale / n


def _experiment(seed: int) -> dict:
    sys.path.insert(0, str(REPO))
    from Forgetting import ForgettingMemory

    tmp = Path(tempfile.mkdtemp())
    stores = {
        "smart": ForgettingMemory(budget=BUDGET, path=tmp / "working.jsonl"),
        "fifo": ForgettingMemory(budget=BUDGET, policy="fifo",
                                 supersede=False),
        "unbounded": ForgettingMemory(budget=None),
    }
    facts, now = _build_life(seed, stores, reinforce={"smart", "unbounded"})

    stores["smart"].save()
    reloaded = ForgettingMemory(budget=BUDGET, path=tmp / "working.jsonl")

    keep = [i for i, f in enumerate(facts) if f["old"] is None]
    updated = [i for i, f in enumerate(facts) if f["old"] is not None]

    r_smart, _ = _answer(reloaded, facts, keep, now)
    r_fifo, _ = _answer(stores["fifo"], facts, keep, now)
    r_unb, _ = _answer(stores["unbounded"], facts, keep, now)
    u_acc, u_stale = _answer(reloaded, facts, updated, now)

    return {
        "retention_acc": round(r_smart, 4),
        "fifo_retention_acc": round(r_fifo, 4),
        "unbounded_retention_acc": round(r_unb, 4),
        "retention_vs_fifo": round(r_smart - r_fifo, 4),
        "update_acc": round(u_acc, 4),
        "stale_rate": round(u_stale, 4),
        "smart_size": len(reloaded),
        "fifo_size": len(stores["fifo"]),
        "unbounded_size": len(stores["unbounded"]),
        "n_retention_q": len(keep),
        "n_update_q": len(updated),
    }


def _control(seed: int) -> dict:
    """Same life, same machinery, supersession OFF: updates must go stale."""
    sys.path.insert(0, str(REPO))
    from Forgetting import ForgettingMemory

    stores = {"nosup": ForgettingMemory(budget=BUDGET, supersede=False)}
    facts, now = _build_life(seed, stores, reinforce={"nosup"})
    updated = [i for i, f in enumerate(facts) if f["old"] is not None]
    u_acc, u_stale = _answer(stores["nosup"], facts, updated, now)
    return {"nosup_update_acc": round(u_acc, 4),
            "nosup_stale_rate": round(u_stale, 4)}


def _check(m: dict, c: dict) -> bool:
    return (m["retention_acc"] >= MIN_RETENTION
            and m["retention_vs_fifo"] >= MIN_GAP
            and m["unbounded_retention_acc"] >= m["retention_acc"]  # ceiling honesty
            and m["update_acc"] >= MIN_UPDATE
            and m["stale_rate"] <= MAX_STALE
            and m["smart_size"] <= BUDGET                           # budget honesty
            and m["fifo_size"] <= BUDGET
            and c["nosup_stale_rate"] >= MIN_CONTROL_STALE
            and c["nosup_update_acc"] < m["update_acc"])


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["ME.4"], _experiment, _check,
                    control_fn=_control, ledger=ledger)
