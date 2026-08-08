"""ME.5 — retrieval survives growth: 100 -> 100k events, one life, four decades.

ME.1 proved cued recall beats recency at 1,000 events. This spec asks what
happens over a LIFETIME: the store grows three orders of magnitude and every
property that made the log worth having must be re-earned at each decade —
against 1000x the distractor mass, with the same abstention floor, off the
same on-disk file, at a latency a live agent could tolerate.

One synthetic life is grown to 100,000 events (~69 days at one event/minute).
At each decade (100, 1k, 10k, 100k) we measure, with `now` at the head of the
log so old events carry ~zero recency:

  1. UNIQUE cues — every event holds a distinct (object, place, colour,
     action) 4-tuple drawn without replacement from disjoint word pools, so a
     cue carrying all four content words identifies exactly one event. The
     hand-picked oracle ceiling is therefore 1.0 BY CONSTRUCTION, and the gap
     between measured precision@1 and 1.0 is pure retrieval degradation:
     score crowding, tie-break leakage, tokenisation collisions. Gate: >=0.95
     at EVERY decade.
  2. AMBIGUOUS cues — only 3 of the 4 words. At 100 events that subset is
     usually still unique; at 100k, ~4-8 events share it and exact-eid
     precision MUST fall toward 1/n_competitors. That curve, reported per
     decade, is the honest degradation measurement the spec asks for. Gates:
     exact-eid precision stays above the recency null at every decade, and
     the top-1 answer must MATCH all three cue words >=0.95 at every decade —
     under crowding, retrieval may return a different genuinely-matching
     occasion, but never a fresher non-match (the "similarity is the primary
     key" claim, empirically re-verified at 100k where tens of thousands of
     partial matches compete).
  3. Abstention — fabricated cues from disjoint vocabulary must still return
     nothing (>=0.95) at every decade; false-memory pressure grows with N.
  4. Latency — mean recall wall-time at 100k events must stay under 1s/query
     on this 4-core ARM box, or the log is not usable live.
  5. Disk roundtrip at scale — reload the ~14 MB JSONL into a fresh store at
     100k and re-run the unique cues: precision must hold (>=0.95).

NULL (must lose at every decade): recency-only retrieval — answer every cue
with the newest event. Its precision is ~1/N and collapses with growth; the
experiment's whole claim is that scored retrieval does not.
"""
from __future__ import annotations

import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

DECADES = [100, 1_000, 10_000, 100_000]
N_QUERIES = 100          # per query class, per decade
N_FABRICATED = 30        # per decade
MIN_PRECISION = 0.95     # unique cues, and reload at 100k
MIN_MATCH = 0.95         # ambiguous top-1 must still match all 3 cue words
MIN_ABSTENTION = 0.95
MAX_LATENCY_MS = 1000.0  # mean per query at 100k events

# Pairwise-disjoint pools; 40*30*20*20 = 480,000 distinct 4-tuples, so 100k
# events can each carry a unique tuple. No word appears in _STOP, in another
# pool, in a speaker name, or in the cue/event filler ("thing", "near").
OBJECTS = ["kettle", "ladder", "apple", "lantern", "hammer", "compass",
           "bucket", "rope", "mirror", "whistle", "anchor", "basket", "drum",
           "kite", "shovel", "candle", "bell", "net", "flag", "chain",
           "barrel", "saddle", "plough", "quilt", "spade", "crate", "fiddle",
           "goblet", "helmet", "inkpot", "jug", "loom", "mallet", "needle",
           "oar", "pulley", "rake", "sickle", "trowel", "vane"]
PLACES = ["pond", "ramp", "platform", "meadow", "shed", "gate", "bridge",
          "cellar", "orchard", "quarry", "dock", "tower", "trail", "garden",
          "mill", "stable", "granary", "wharf", "paddock", "terrace",
          "courtyard", "hedge", "windbreak", "spring", "hollow", "ridge",
          "thicket", "byre", "kiln", "sluice"]
COLOURS = ["copper", "crimson", "olive", "violet", "amber", "teal", "ivory",
           "slate", "coral", "bronze", "indigo", "maroon", "ochre", "pewter",
           "russet", "sable", "scarlet", "sepia", "turquoise", "umber"]
ACTIONS = ["carried", "dropped", "painted", "repaired", "buried", "balanced",
           "measured", "cleaned", "stacked", "traded", "borrowed", "hid",
           "polished", "weighed", "hoisted", "mended", "sharpened", "soaked",
           "wrapped", "hauled"]
SPEAKERS = ["ada", "bruno", "chika", "jack"]

FAB_OBJECTS = ["zeppelin", "harmonica", "telescope", "cauldron", "typewriter",
               "gramophone", "sundial", "abacus", "monocle", "tapestry"]
FAB_PLACES = ["volcano", "lighthouse", "catacomb", "glacier", "bazaar",
              "observatory", "vineyard", "citadel"]

T0 = 1_000_000.0
DT = 60.0


def _tuple_stream(seed: int):
    """The life's event tuples: 100k DISTINCT 4-tuples, seeded. Shared by the
    experiment and the recency null so both answer identical questions."""
    rng = random.Random(seed)
    n_o, n_p, n_c, n_a = len(OBJECTS), len(PLACES), len(COLOURS), len(ACTIONS)
    idxs = rng.sample(range(n_o * n_p * n_c * n_a), DECADES[-1])
    tuples = []
    for ix in idxs:
        ix, o = divmod(ix, n_o)
        ix, p = divmod(ix, n_p)
        a, c = divmod(ix, n_c)
        tuples.append((OBJECTS[o], PLACES[p], COLOURS[c], ACTIONS[a]))
    return rng, tuples


def _queries(seed: int, decade: int, n: int):
    """Which events are asked about at this decade, and the cue word orders.
    Seeded by (seed, decade) so experiment and null sample identically."""
    rng = random.Random(seed * 7919 + decade)
    picks = rng.sample(range(n), N_QUERIES)
    orders = [rng.sample(range(4), 4) for _ in picks]      # unique-cue scramble
    drops = [rng.randrange(4) for _ in picks]              # ambiguous: word to drop
    fabs = [(rng.choice(FAB_OBJECTS), rng.choice(FAB_PLACES))
            for _ in range(N_FABRICATED)]
    return picks, orders, drops, fabs


def _unique_cue(words, order) -> str:
    w = [words[i] for i in order]
    return f"the thing about the {w[0]} and the {w[1]} and the {w[2]} {w[3]}"


def _ambiguous_cue(words, drop) -> str:
    kept = [w for i, w in enumerate(words) if i != drop]
    return f"the thing about the {kept[0]} and the {kept[1]} {kept[2]}"


def _experiment(seed: int) -> dict:
    sys.path.insert(0, str(REPO))
    from EpisodicMemory import EpisodicMemory

    tmp = Path(tempfile.mkdtemp()) / "life.jsonl"
    rng, tuples = _tuple_stream(seed)
    mem = EpisodicMemory(path=tmp)
    out: dict = {}

    n_done = 0
    for decade in DECADES:
        for i in range(n_done, decade):
            obj, place, colour, act = tuples[i]
            speaker = rng.choice(SPEAKERS)
            channel = ("said" if speaker == "jack" and rng.random() < 0.5
                       else "did" if speaker == "jack"
                       else "heard")
            mem.record(channel, speaker,
                       f"{speaker} {act} the {colour} {obj} near the {place}",
                       importance=rng.uniform(0.5, 5.0), t=T0 + i * DT)
        n_done = decade
        now = T0 + n_done * DT

        picks, orders, drops, fabs = _queries(seed, decade, n_done)
        u_hits = a_hits = a_matches = 0
        t_q0 = time.perf_counter()
        for qi, order, drop in zip(picks, orders, drops):
            words = tuples[qi]
            res = mem.recall(_unique_cue(words, order), top_k=1, now=now)
            u_hits += bool(res and res[0].event.eid == qi)
            res = mem.recall(_ambiguous_cue(words, drop), top_k=1, now=now)
            a_hits += bool(res and res[0].event.eid == qi)
            if res:
                kept = {w for i, w in enumerate(words) if i != drop}
                a_matches += kept <= set(res[0].event.text.split())
        lat_ms = (time.perf_counter() - t_q0) * 1000 / (2 * N_QUERIES)
        abstained = sum(
            not mem.recall(f"the thing about the {fo} and the {fp}",
                           top_k=1, now=now)
            for fo, fp in fabs)

        tag = str(decade)
        out[f"u_p1_{tag}"] = round(u_hits / N_QUERIES, 4)
        out[f"a_p1_{tag}"] = round(a_hits / N_QUERIES, 4)
        out[f"a_match_{tag}"] = round(a_matches / N_QUERIES, 4)
        out[f"abstain_{tag}"] = round(abstained / N_FABRICATED, 4)
        out[f"lat_ms_{tag}"] = round(lat_ms, 2)

    # Disk roundtrip at full scale: a fresh process would see this file.
    del mem
    mem2 = EpisodicMemory(path=tmp)
    now = T0 + n_done * DT
    picks, orders, _, _ = _queries(seed, DECADES[-1], n_done)
    re_hits = sum(
        bool((r := mem2.recall(_unique_cue(tuples[qi], order), top_k=1,
                               now=now)) and r[0].event.eid == qi)
        for qi, order in zip(picks, orders))
    out["u_p1_reload_100000"] = round(re_hits / N_QUERIES, 4)
    out["events"] = len(mem2)
    shutil.rmtree(tmp.parent, ignore_errors=True)
    return out


def _control(seed: int) -> dict:
    """Recency-only null: the newest event answers every cue. Its precision is
    ~1/N by construction and must sit below the experiment at every decade.
    Uses the identical seeded query sample, so it answers the same questions."""
    out: dict = {}
    for decade in DECADES:
        picks, _, _, _ = _queries(seed, decade, decade)
        hits = sum(qi == decade - 1 for qi in picks)
        out[f"rec_p1_{decade}"] = round(hits / N_QUERIES, 4)
    return out


def _check(m: dict, c: dict) -> bool:
    for d in DECADES:
        tag = str(d)
        if not (m[f"u_p1_{tag}"] >= MIN_PRECISION
                and m[f"a_match_{tag}"] >= MIN_MATCH
                and m[f"abstain_{tag}"] >= MIN_ABSTENTION
                and m[f"u_p1_{tag}"] > c[f"rec_p1_{tag}"]
                and m[f"a_p1_{tag}"] > c[f"rec_p1_{tag}"]):
            return False
    return (m["u_p1_reload_100000"] >= MIN_PRECISION
            and m["lat_ms_100000"] <= MAX_LATENCY_MS)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["ME.5"], _experiment, _check,
                    control_fn=_control, ledger=ledger)
