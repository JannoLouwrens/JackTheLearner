"""The paraphrase evaluation fixture — shared, frozen, and hostile to its author.

Designed in `docs/research/MEMORY_RETRIEVAL_BAKEOFF.md` §2 and built here so
that ME.11 and every ME.11.x arm are scored on IDENTICAL data, verified honest
by ME.11.0 *before* any arm is allowed to run.

The whole point of this file is that its most important properties are
MACHINE-CHECKED rather than asserted in prose:

* **Lexical disjointness.** Every content concept has one *stored* form (what
  goes in the diary) and a set of *cue* forms (what a person would actually
  say). The two pools are globally disjoint, verified at import by
  `_assert_vocab_disjoint()`, so a cue can share no content word with its
  target. Corollary, and the reason ME.11.0 exists: the incumbent
  lexical-containment retriever must score ~0 on this cue set BY CONSTRUCTION.
  If it does not, the eval set has leaked and every arm's number is worthless.

* **Gold is a SET, derived mechanically.** `G(cue)` is every event whose
  concept tuple satisfies the cue's concept constraints, computed from the
  generator's own bindings — never hand-written after seeing an arm's output.
  A single-gold label on a question with two correct answers is a test bug
  scored as a model failure, and it silently depresses every arm equally.

* **Two independent paths to the truth.** Gold comes from the generator's
  bookkeeping; the ORACLE re-derives each event's concepts by PARSING ITS
  STORED TEXT. If the generator ever writes a sentence that disagrees with what
  it recorded it wrote, the oracle ceiling drops and ME.11.0 fails.

* **Frozen by hash.** `build()` is a pure function of `seed`; the sha256 of the
  whole fixture goes into every arm's ledger entry. Two arms carrying different
  hashes were not compared.

Structure of one seed's fixture:
  - a life of 5,000 events, of which 40 are *targets* (one per object, with a
    predicate and a provenance) each surrounded by 5 distractors
    (2 same-object/different-predicate, 2 same-predicate/different-object,
    1 identical-text/wrong-provenance) and the rest disjoint-vocabulary filler;
  - 160 headline cues, 40 in each of four registers (R1 synonym, R2 indirect
    attributed question, R3 circumlocution, R4 superordinate);
  - 12 deliberately ambiguous probes, which exercise the |G| > k_amb partition;
  - 600 adversarial negatives in four families (N1 held-out-target,
    N2 unseen-entity, N3 provenance-impossible, N4 out-of-world), split
    300 tune / 300 certify, family-balanced.

Nothing here imports torch, touches the network, or costs a GPU second.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from EpisodicMemory import _tokens  # noqa: E402  the null's OWN tokeniser

# ── vocabulary ───────────────────────────────────────────────────────────
# stored form -> the cue forms a person would use instead. Longest cue form is
# the circumlocution (register R3); the pools must never intersect.
OBJECTS = {
    "ladder":  ["rungs", "climbing frame", "the thing you climb"],
    "kettle":  ["teapot", "brew pot"],
    "pond":    ["water", "still pool"],
    "compass": ["navigator", "bearing dial"],
    "orchard": ["grove", "apple trees"],
    "lantern": ["lamp", "flame holder"],
    "bridge":  ["crossing", "narrow span"],
    "shovel":  ["spade", "digging tool"],
    "kite":    ["flyer", "wind toy"],
    "cellar":  ["basement", "underground room"],
    "anchor":  ["mooring", "boat weight"],
    "drum":    ["percussion", "skin tom"],
    "gate":    ["barrier", "swing panel"],
    "rope":    ["cord", "coiled twine"],
    "trowel":  ["hand fork", "planting scoop"],
    "bell":    ["chime", "brass ringer"],
    "crate":   ["packing box", "wooden bin"],
    "hose":    ["garden pipe", "watering tube"],
    "mirror":  ["reflector", "looking glass"],
    "bench":   ["seat plank", "resting perch"],
    "fence":   ["palings", "boundary rail"],
    "awning":  ["canopy", "shade sheet"],
    "wagon":   ["handcart", "pull trolley"],
    "flute":   ["woodwind", "tin whistle"],
    "easel":   ["painting stand", "canvas rack"],
    "kiln":    ["firing oven", "pottery furnace"],
    "saddle":  ["riding tack", "horse pad"],
    "quilt":   ["bed cover", "patch blanket"],
    "lever":   ["prying bar", "throw handle"],
    "basket":  ["woven hamper", "carry pannier"],
    "clock":   ["timepiece", "hour face"],
    "ladle":   ["broth dipper", "serving spoon"],
    "shutter": ["window flap", "louvre blind"],
    "pulley":  ["block tackle", "lifting wheel"],
    "pitcher": ["jug", "pouring vessel"],
    "harrow":  ["soil rake", "tilling comb"],
    "lectern": ["book podium", "speech desk"],
    "birdcage": ["aviary", "wire enclosure"],
    "stile":   ["field step", "wall notch"],
    "beacon":  ["signal tower", "warning fire"],
}
PREDICATES = {
    "cracked":   (["broken", "split", "fractured"], "faulty"),
    "stuck":     (["jammed", "wedged"], "faulty"),
    "leaning":   (["tilted", "slanted"], "faulty"),
    "missing":   (["gone", "vanished"], "faulty"),
    "repainted": (["recoloured", "freshly coated"], "altered"),
    "new":       (["unused", "pristine"], "altered"),
    "green":     (["verdant", "mossy"], "altered"),
    "heavy":     (["weighty", "unliftable"], "unwieldy"),
    "noisy":     (["loud", "clattering"], "unwieldy"),
    "hollow":    (["resonant", "empty within"], "unwieldy"),
}
CLASS_CUES = {"faulty": "faulty", "altered": "altered", "unwieldy": "unwieldy"}

SITES = ["shed", "porch", "lane", "hedge", "barn", "yard"]

FILLER_OBJECTS = ["cauldron", "trellis", "mailbox", "sundial", "hammock",
                  "birdbath", "wheelbarrow", "gramophone", "footstool",
                  "weathervane", "doorknob", "teacup", "bookcase", "sledge"]
FILLER_PREDICATES = ["dusty", "warm", "folded", "polished", "damp", "crooked",
                     "faded", "sticky", "humming", "frozen"]

# N2's entity substitutions: objects that are NEVER recorded, anywhere.
UNSEEN_OBJECTS = ["trampoline", "harpsichord", "snowplough", "periscope"]
# N4's out-of-world material: another world entirely.
OUT_OF_WORLD = ["wedding cake", "tax return", "submarine hatch",
                "opera ticket", "glacier survey", "dental appointment"]

PEOPLE = ["ada", "bruno", "chika"]
CHANNELS = ["heard", "said", "did"]

N_TARGETS = 40              # one per object; 40 x 4 registers = 160 headline cues
LIFE_EVENTS = 5000
N_AMBIGUOUS = 12
N_NEG_PER_FAMILY = 150
K_AMB = 3                   # |G| above this -> AMBIGUOUS partition
DT = 30.0


def _pool(*phrases) -> set:
    out = set()
    for p in phrases:
        out |= _tokens(p)
    return out


def _assert_vocab_disjoint() -> None:
    """The invariant the whole eval set rests on, checked at import.

    A single shared token between a stored form and a cue form would hand the
    lexical null a free hit and quietly turn ME.11 into a measurement of that
    accident. Cheap to check; catastrophic to miss.
    """
    stored = _pool(*OBJECTS, *PREDICATES, *SITES,
                   *FILLER_OBJECTS, *FILLER_PREDICATES)
    cue = _pool(*[c for cs in OBJECTS.values() for c in cs],
                *[c for cs, _ in PREDICATES.values() for c in cs],
                *CLASS_CUES.values(), *UNSEEN_OBJECTS, *OUT_OF_WORLD)
    clash = stored & cue
    if clash:
        raise AssertionError(f"stored/cue vocabulary overlap: {sorted(clash)}")
    seen_unseen = _pool(*UNSEEN_OBJECTS) & stored
    if seen_unseen:
        raise AssertionError(f"'unseen' objects are recorded: {sorted(seen_unseen)}")
    # A token shared by two concepts' CUE forms would make the cue itself
    # ambiguous — the arm would be right and we would score it wrong.
    groups = ([(f"obj:{o}", cs) for o, cs in OBJECTS.items()]
              + [(f"pred:{p}", cs) for p, (cs, _) in PREDICATES.items()]
              + [(f"class:{k}", [v]) for k, v in CLASS_CUES.items()]
              + [("unseen", UNSEEN_OBJECTS), ("outofworld", OUT_OF_WORLD)])
    owner: dict = {}
    for name, forms in groups:
        for tok in _pool(*forms):
            if owner.setdefault(tok, name) != name:
                raise AssertionError(
                    f"cue token {tok!r} claimed by {owner[tok]} and {name}")


_assert_vocab_disjoint()


def _text(obj: str, pred: str, site: str) -> str:
    return f"the {obj} was {pred} near the {site}"


# ── the reverse path: parse a stored sentence back into concepts ─────────
_STORED_OBJ = {tuple(_tokens(o))[0]: o for o in OBJECTS}
_STORED_PRED = {tuple(_tokens(p))[0]: p for p in PREDICATES}


def parse_concepts(text: str) -> dict:
    """Recover (object, predicate) from a diary sentence, WITHOUT consulting
    the generator's bookkeeping. The oracle scores with this; gold is built
    from the bookkeeping. The two disagreeing is a fixture bug, and ME.11.0's
    oracle-ceiling assertion is what turns that bug into a red ledger entry."""
    toks = _tokens(text)
    obj = next((_STORED_OBJ[t] for t in toks if t in _STORED_OBJ), None)
    pred = next((_STORED_PRED[t] for t in toks if t in _STORED_PRED), None)
    return {"obj": obj, "pred": pred}


def satisfies(ann: dict, ev: dict, con: dict) -> bool:
    """Does one annotated event satisfy a cue's concept constraints?"""
    if "obj" in con and ann.get("obj") != con["obj"]:
        return False
    if "pred" in con and ann.get("pred") != con["pred"]:
        return False
    if "pred_class" in con:
        p = ann.get("pred")
        if p is None or PREDICATES[p][1] != con["pred_class"]:
            return False
    if "speaker" in con and ev["speaker"] != con["speaker"]:
        return False
    if "channel" in con and ev["channel"] != con["channel"]:
        return False
    return True


# ── the generator ────────────────────────────────────────────────────────
def build(seed: int) -> dict:
    """A complete, deterministic evaluation fixture for one seed."""
    rng = random.Random(1000 + seed)
    events: list[dict] = []
    anns: list[dict] = []

    def rec(channel, speaker, text, obj=None, pred=None, importance=1.0):
        eid = len(events)
        events.append({"eid": eid, "t": eid * DT, "channel": channel,
                       "speaker": speaker, "text": text,
                       "importance": importance})
        anns.append({"obj": obj, "pred": pred})
        return eid

    # --- targets: ONE per object, each with its five distractors ----------
    # Register R4's constraint is (object, predicate CLASS), so anything else
    # recorded about that object with a same-class predicate is a genuine
    # second answer and the cue lands in the AMBIGUOUS partition. Two weaker
    # designs were measured and rejected: unique (object, predicate) tuples
    # left R4 with 1 of 60 headline cues, and unique (object, class) tuples
    # left it with 7, because another target's D1 distractor re-records the
    # same object under a same-class predicate. One target per object, with
    # distractors drawn from other classes, is the design that makes every
    # register survive its own gold-set arithmetic.
    objs = list(OBJECTS)
    rng.shuffle(objs)
    chosen = [(o, rng.choice(list(PREDICATES))) for o in objs[:N_TARGETS]]
    target_pred = dict(chosen)
    targets = []
    for i, (obj, pred) in enumerate(chosen):
        channel = CHANNELS[i % 3]
        speaker = rng.choice(PEOPLE) if channel == "heard" else "jack"
        targets.append({"obj": obj, "pred": pred, "channel": channel,
                        "speaker": speaker, "site": rng.choice(SITES),
                        "eid": None, "d3_eid": None})

    # Interleave targets/distractors with filler so recency cannot be a cue.
    slots = list(range(LIFE_EVENTS))
    rng.shuffle(slots)
    plan: dict[int, tuple] = {}
    used = 0
    for ti, tg in enumerate(targets):
        # A distractor may never become a correct answer to some OTHER cue.
        # D1 (same object) takes another predicate CLASS, so it cannot answer
        # this target's R4. D2 (same predicate) goes only to objects whose own
        # target is in a different class, so it cannot answer THEIR R4 either.
        klass = PREDICATES[tg["pred"]][1]
        others_p = [p for p, (_, k) in PREDICATES.items() if k != klass]
        others_o = [o for o in objs[:N_TARGETS]
                    if o != tg["obj"]
                    and PREDICATES[target_pred[o]][1] != klass]
        plan[slots[used]] = ("target", ti); used += 1
        for p in rng.sample(others_p, 2):                     # D1
            plan[slots[used]] = ("d1", ti, p); used += 1
        for o in rng.sample(others_o, 2):                     # D2
            plan[slots[used]] = ("d2", ti, o); used += 1
        plan[slots[used]] = ("d3", ti); used += 1             # D3

    for k in range(LIFE_EVENTS):
        job = plan.get(k)
        if job is None:
            o, p = rng.choice(FILLER_OBJECTS), rng.choice(FILLER_PREDICATES)
            rec(rng.choice(CHANNELS),
                rng.choice(PEOPLE + ["jack"]), _text(o, p, rng.choice(SITES)))
            continue
        kind, ti = job[0], job[1]
        tg = targets[ti]
        if kind == "target":
            tg["eid"] = rec(tg["channel"], tg["speaker"],
                            _text(tg["obj"], tg["pred"], tg["site"]),
                            tg["obj"], tg["pred"])
        elif kind == "d1":                     # same object, other predicate
            rec(tg["channel"], tg["speaker"],
                _text(tg["obj"], job[2], rng.choice(SITES)), tg["obj"], job[2])
        elif kind == "d2":                     # same predicate, other object
            rec(tg["channel"], tg["speaker"],
                _text(job[2], tg["pred"], rng.choice(SITES)), job[2], tg["pred"])
        else:                                  # D3: right content, wrong source
            wrong_ch = next(c for c in CHANNELS if c != tg["channel"])
            wrong_sp = "jack" if wrong_ch != "heard" else next(
                s for s in PEOPLE if s != tg["speaker"])
            tg["d3_eid"] = rec(wrong_ch, wrong_sp,
                               _text(tg["obj"], tg["pred"], tg["site"]),
                               tg["obj"], tg["pred"])

    def gold(con: dict) -> list[int]:
        return [e["eid"] for e, a in zip(events, anns) if satisfies(a, e, con)]

    # --- cues: four registers over the same 60 targets ------------------
    cues = []
    for ti, tg in enumerate(targets):
        ocues, (pcues, pclass) = OBJECTS[tg["obj"]], PREDICATES[tg["pred"]]
        o_short = rng.choice(ocues[:-1]) if len(ocues) > 1 else ocues[0]
        o_long, pc = ocues[-1], rng.choice(pcues)
        base = {"obj": tg["obj"], "pred": tg["pred"]}
        attributed = {**base, "speaker": tg["speaker"], "channel": tg["channel"]}
        for reg, text, con in (
            ("R1", f"the {pc} {o_short}", base),
            ("R2", f"what did {tg['speaker']} say was {pc} about the {o_short}",
             attributed),
            ("R3", f"you know {o_long}, it turned out {pc}", base),
            ("R4", f"was there anything {CLASS_CUES[pclass]} about the {o_short}",
             {"obj": tg["obj"], "pred_class": pclass}),
        ):
            g = gold(con)
            cues.append({"register": reg, "text": text, "constraints": con,
                         "gold": g, "target_eid": tg["eid"], "ti": ti,
                         "stratum": tg["channel"],
                         "speaker": tg["speaker"] if reg == "R2" else None,
                         "channel": tg["channel"] if reg == "R2" else None,
                         "ambiguous": len(g) > K_AMB})

    # Deliberately unanswerable-by-uniqueness probes: the |G| > K_AMB path
    # must be exercised, or the partition mechanism is untested code.
    ambiguous = []
    for i in range(N_AMBIGUOUS):
        pclass = list(CLASS_CUES)[i % len(CLASS_CUES)]
        con = {"pred_class": pclass}
        ambiguous.append({"register": "AMB", "constraints": con,
                          "text": f"did anyone mention something {CLASS_CUES[pclass]}",
                          "gold": gold(con), "stratum": "any",
                          "speaker": None, "channel": None, "ambiguous": True})

    # --- negatives: adversarial, four families -------------------------
    negatives = []
    heard_cues = [c for c in cues
                  if c["stratum"] == "heard" and "pred" in c["constraints"]]
    for i in range(N_NEG_PER_FAMILY):
        c = cues[rng.randrange(len(cues))]
        negatives.append({"family": "N1", "text": c["text"],
                          "exclude_eids": list(c["gold"]),
                          "speaker": c["speaker"], "channel": c["channel"]})

        c = cues[rng.randrange(len(cues))]
        unseen = UNSEEN_OBJECTS[i % len(UNSEEN_OBJECTS)]
        oc = OBJECTS[c["constraints"]["obj"]]
        swapped = c["text"]
        for form in sorted(oc, key=len, reverse=True):
            if form in swapped:
                swapped = swapped.replace(form, unseen)
                break
        negatives.append({"family": "N2", "text": swapped, "exclude_eids": [],
                          "speaker": c["speaker"], "channel": c["channel"]})

        # N3 must be genuinely unanswerable: verify no event with this content
        # was ever recorded from the wrong speaker before emitting it. Unique
        # target tuples do NOT guarantee this — another target's D1/D2 can
        # re-record the same (object, predicate) pair under a different voice.
        for _ in range(50):
            c = heard_cues[rng.randrange(len(heard_cues))]
            tg = targets[c["ti"]]
            wrong = rng.choice([s for s in PEOPLE if s != tg["speaker"]])
            con = {"obj": tg["obj"], "pred": tg["pred"], "speaker": wrong,
                   "channel": "heard"}
            if not gold(con):
                break
        else:
            raise AssertionError("could not build an unanswerable N3 negative")
        negatives.append({"family": "N3", "exclude_eids": [],
                          "text": f"what did {wrong} say was "
                                  f"{PREDICATES[tg['pred']][0][0]} about the "
                                  f"{OBJECTS[tg['obj']][0]}",
                          "speaker": wrong, "channel": "heard"})

        thing = OUT_OF_WORLD[i % len(OUT_OF_WORLD)]
        negatives.append({"family": "N4", "text": f"what was said about the {thing}",
                          "exclude_eids": [], "speaker": None, "channel": None})

    # STRATIFIED split. Shuffling then cutting in half gave family cells of
    # 68-82 instead of 75/75; a certify split that is short on N1 (the hardest
    # family) would certify abstention on the easy negatives.
    rng.shuffle(negatives)
    tune, certify = [], []
    for f in ("N1", "N2", "N3", "N4"):
        fam = [n for n in negatives if n["family"] == f]
        tune += fam[:len(fam) // 2]
        certify += fam[len(fam) // 2:]
    rng.shuffle(tune)
    rng.shuffle(certify)

    # --- the leaky control: cues made by DELETING words from the target --
    # If the leak detector cannot see a planted leak it is not a detector.
    leaky = [{"text": " ".join(sorted(_tokens(events[c["target_eid"]]["text"]))),
              "gold": c["gold"], "speaker": c["speaker"], "channel": c["channel"]}
             for c in cues if not c["ambiguous"]]

    fx = {"seed": seed, "events": events, "anns": anns, "targets": targets,
          "cues": cues, "ambiguous_probes": ambiguous, "leaky_cues": leaky,
          "negatives": {"tune": tune, "certify": certify},
          "now": len(events) * DT + DT}
    fx["hash"] = fixture_hash(fx)
    return fx


def fixture_hash(fx: dict) -> str:
    """Frozen identity of an eval set. Two arms carrying different hashes were
    scored on different data and may not be compared, however similar the
    numbers look."""
    payload = {k: fx[k] for k in ("seed", "events", "cues", "ambiguous_probes",
                                  "negatives", "leaky_cues")}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_into_memory(fx: dict, path=None):
    """Materialise the fixture as a real `EpisodicMemory`, so every arm reads
    the store through the same interface the live agent uses."""
    from EpisodicMemory import EpisodicMemory
    mem = EpisodicMemory(path=path)
    for e in fx["events"]:
        mem.record(e["channel"], e["speaker"], e["text"],
                   importance=e["importance"], t=e["t"])
    return mem
