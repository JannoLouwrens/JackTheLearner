"""ME.10 — keeps the memory AND learns the general skill (double dissociation).

GOAL.md's memory section says lived experience must live in TWO stores with
separate failure modes: the verbatim diary (EpisodicMemory, ME.1/ME.9) and
general skill distilled into weights (McClelland et al. 1995's complementary
learning systems — the hippocampus/cortex split; docs/research/MEMORY.md).
This spec holds them apart: distillation must not eat the diary, the diary
must not be secretly doing the skill's job, and each ablation must destroy
exactly its own capability.

The life: Jack drops 84 of the 120 possible {colour} x {object} things into
the pond and watches each sink or float, every episode recorded verbatim in
the diary among 240 filler events. The world's rule is COMPOSITIONAL: each
colour and each object carries a hidden bit, outcome = colour_bit XOR
object_bit. XOR is load-bearing — knowing only the colour (or only the
object) predicts at exactly chance, so retrieving a similar episode from the
diary answers a NOVEL pair at 0.5, while a distilled net that recovered the
bits answers it near 1.0. The skill is measured ONLY on the 36 held-out
pairs the diary has never seen; the diary is measured on cued recall of the
84 episodes that actually happened.

Distillation literally replays the diary: training pairs are parsed from the
store's "did" events, not from a side channel — wipe the diary BEFORE
distillation and there would be nothing to learn from.

NULL (must lose): the same store with no weight update — the untrained init
predicts held-out outcomes at chance. Its gap to the trained net is what
distillation buys.

CONTROL (each ablation kills exactly its own capability):
  wipe the diary   -> cued recall collapses to abstention, held-out skill
                      unchanged (the skill was in the weights);
  revert the weights -> held-out skill back to chance, cued recall
                      unchanged (the diary was on disk).
Either ablation killing BOTH means one store was masquerading as two.

STRENGTHENED 2026-09-06 (78th audit B2 / Review FTB 3) — ME.11's distractor
question on the diary: `store_on_heldout` already demanded the diary not do
the skill's JOB (outcome readout pinned to chance by XOR), but it never asked
whether the diary ANSWERS at all. A held-out cue names a colour the diary saw
~8 times and an object it saw ~7 times, in a pairing it never lived —
returning the nearest lived episode as the answer is the confabulation ME.11
measured, whatever outcome it happens to name. The diary must ABSTAIN on
>= MIN_DISTRACTOR_ABST of the 36 held-out pairs. Every pair is evaluable by
construction (each lived episode names exactly one pair, so no seen event
carries both cue words), which the code checks and records rather than
assumes. Nothing above moved.
"""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

# The diary under test. Undeclared until 2026-09-06 (78th audit B2).
IMPL_DEPS = ["EpisodicMemory.py"]

N_FILLER = 240
HIDDEN = 32
LR = 5e-3
WEIGHT_DECAY = 2e-3
EPOCHS = 6000

MIN_RECALL = 0.90            # diary answers "what happened when..." on lived pairs
MAX_RECALL_DROP = 0.02       # distillation must not eat the diary
MIN_SKILL = 0.85             # trained net on held-out pairs (compositional rule)
MAX_CHANCE_SKILL = 0.65      # untrained init / diary-on-held-out / reverted net
MIN_SKILL_GAIN = 0.25        # trained minus untrained on held-out
MAX_WIPE_RECALL = 0.05       # empty diary must abstain, not confabulate
MAX_ABLATION_DRIFT = 0.02    # the capability an ablation should NOT touch
# Added 2026-09-06 (78th audit B2 / Review FTB 3; strengthen-only).
MIN_DISTRACTOR_ABST = 0.95   # abstention on never-lived pair cues
MIN_DISTRACTOR_EVAL = 30     # aliveness floor (36 held pairs by construction)

COLOURS = ["copper", "crimson", "olive", "violet", "amber", "teal", "ivory",
           "slate", "coral", "bronze"]
OBJECTS = ["kettle", "ladder", "apple", "lantern", "hammer", "compass",
           "bucket", "rope", "mirror", "whistle", "anchor", "basket"]
N_SEEN = 84                  # of the 120 pairs; 36 held out for the skill
OUTCOMES = ("sank", "floated")

FILLER_PEOPLE = ["ada", "bruno", "chika", "dora"]
FILLER_TOPICS = ["weather", "harvest", "market", "festival", "song", "story",
                 "riddle", "journey", "recipe", "garden", "letter", "dance"]
FILLER_PLACES = ["meadow", "orchard", "bridge", "tower", "dock", "trail"]


def _world(seed: int):
    """Hidden bits, the seen/held split, and a shuffled slot order for when
    each episode happens inside the filler stream. All from one seeded RNG so
    _experiment and _control rebuild the identical life."""
    rng = random.Random(seed)
    cbits = {c: rng.randint(0, 1) for c in COLOURS}
    obits = {o: rng.randint(0, 1) for o in OBJECTS}
    pairs = [(c, o) for c in COLOURS for o in OBJECTS]
    rng.shuffle(pairs)
    seen, held = pairs[:N_SEEN], pairs[N_SEEN:]
    slots = rng.sample(range(N_SEEN + N_FILLER), N_SEEN)
    return rng, cbits, obits, seen, held, slots


def _outcome(cbits, obits, pair) -> str:
    return OUTCOMES[cbits[pair[0]] ^ obits[pair[1]]]


def _live(seed: int, mem_path: Path):
    """Stream the life into a diary on disk. Returns (mem, world, now)."""
    sys.path.insert(0, str(REPO))
    from EpisodicMemory import EpisodicMemory

    rng, cbits, obits, seen, held, slots = _world(seed)
    episode_at = {slot: i for i, slot in enumerate(slots)}
    mem = EpisodicMemory(path=mem_path)
    t0 = 1_000_000.0
    for step in range(N_SEEN + N_FILLER):
        now = t0 + step * 60.0
        if step in episode_at:
            c, o = seen[episode_at[step]]
            mem.record("did", "jack",
                       f"jack dropped the {c} {o} into the pond and it "
                       f"{_outcome(cbits, obits, (c, o))}", t=now)
        else:
            mem.record("heard", rng.choice(FILLER_PEOPLE),
                       f"{rng.choice(FILLER_PEOPLE)} shared a "
                       f"{rng.choice(FILLER_TOPICS)} about the "
                       f"{rng.choice(FILLER_TOPICS)} near the "
                       f"{rng.choice(FILLER_PLACES)}", t=now)
    return mem, (cbits, obits, seen, held), t0 + (N_SEEN + N_FILLER) * 60.0


def _features(pairs):
    import torch
    x = torch.zeros(len(pairs), len(COLOURS) + len(OBJECTS))
    for i, (c, o) in enumerate(pairs):
        x[i, COLOURS.index(c)] = 1.0
        x[i, len(COLOURS) + OBJECTS.index(o)] = 1.0
    return x


def _distill_from_diary(mem):
    """Parse (pair, outcome) training data out of the diary's 'did' events.
    The store IS the dataset — nothing reaches the weights except through it."""
    data = []
    for ev in mem.events:
        if ev.channel != "did":
            continue
        words = ev.text.split()
        c = next(w for w in words if w in COLOURS)
        o = next(w for w in words if w in OBJECTS)
        y = next(w for w in words if w in OUTCOMES)
        data.append(((c, o), OUTCOMES.index(y)))
    return data


def _train(data, seed: int):
    """Distil the replayed episodes into a small MLP. Returns (net, init)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(len(COLOURS) + len(OBJECTS), HIDDEN),
                        nn.Tanh(), nn.Linear(HIDDEN, 2))
    init = {k: v.clone() for k, v in net.state_dict().items()}
    x = _features([p for p, _ in data])
    y = torch.tensor([lbl for _, lbl in data])
    opt = torch.optim.Adam(net.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)
    lossf = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        opt.zero_grad()
        lossf(net(x), y).backward()
        opt.step()
    return net, init


def _skill_acc(net, pairs, cbits, obits) -> float:
    import torch
    with torch.no_grad():
        pred = net(_features(pairs)).argmax(1)
    truth = [OUTCOMES.index(_outcome(cbits, obits, p)) for p in pairs]
    return sum(int(p == t) for p, t in zip(pred.tolist(), truth)) / len(pairs)


def _recall_acc(mem, pairs, cbits, obits, now) -> float:
    """Cued recall: 'jack dropped the copper anchor' -> does the top trace
    name the true outcome? Abstention counts as wrong."""
    hits = 0
    for c, o in pairs:
        got = mem.recall(f"jack dropped the {c} {o}", top_k=1, now=now)
        if got and _outcome(cbits, obits, (c, o)) in got[0].event.text.split():
            hits += 1
    return hits / len(pairs)


def _experiment(seed: int) -> dict:
    tmp = Path(tempfile.mkdtemp()) / "diary.jsonl"
    mem, (cbits, obits, seen, held), now = _live(seed, tmp)

    recall_pre = _recall_acc(mem, seen, cbits, obits, now)

    data = _distill_from_diary(mem)
    net, init = _train(data, seed)
    skill = _skill_acc(net, held, cbits, obits)

    import torch.nn as nn
    null_net = nn.Sequential(nn.Linear(len(COLOURS) + len(OBJECTS), HIDDEN),
                             nn.Tanh(), nn.Linear(HIDDEN, 2))
    null_net.load_state_dict(init)
    null_skill = _skill_acc(null_net, held, cbits, obits)

    recall_post = _recall_acc(mem, seen, cbits, obits, now)
    # The diary asked the skill's question: retrieve for a NEVER-lived pair,
    # read out whatever outcome comes back. XOR pins this to chance.
    store_held = _recall_acc(mem, held, cbits, obits, now)

    # Distractor conjunct (2026-09-06): the diary must return NOTHING for a
    # never-lived pair, not its nearest lived episode. Evaluability is checked
    # per pair rather than assumed: no seen event may carry both cue words.
    d_abst = d_eval = 0
    for c_, o_ in held:
        if any(c_ in ev.text.split() and o_ in ev.text.split()
               for ev in mem.events):
            continue
        d_eval += 1
        d_abst += not mem.recall(f"jack dropped the {c_} {o_}", top_k=1, now=now)

    return {
        "distractor_abstention": round(d_abst / d_eval, 4) if d_eval else 0.0,
        "distractor_evaluated": d_eval,
        "distractor_excluded": len(held) - d_eval,
        "recall_pre": round(recall_pre, 4),
        "recall_post": round(recall_post, 4),
        "skill_heldout": round(skill, 4),
        "null_skill_heldout": round(null_skill, 4),
        "skill_gain": round(skill - null_skill, 4),
        "store_on_heldout": round(store_held, 4),
        "recall_kept_x_skill_gained": round(recall_post * (skill - null_skill), 4),
        "n_train_episodes": len(data),
        "n_heldout": len(held),
    }


def _control(seed: int) -> dict:
    """Rebuild the identical life and distilled net, then ablate each store
    and measure BOTH capabilities after each ablation."""
    sys.path.insert(0, str(REPO))
    from EpisodicMemory import EpisodicMemory

    tmp = Path(tempfile.mkdtemp()) / "diary.jsonl"
    mem, (cbits, obits, seen, held), now = _live(seed, tmp)
    net, init = _train(_distill_from_diary(mem), seed)

    # The two ablations are counterfactual BRANCHES of one life, so snapshot
    # the intact diary from disk now — the wipe below deletes the file, and
    # the revert branch must answer from a diary the revert did not touch.
    intact = EpisodicMemory(path=tmp)

    # Ablation 1: wipe the diary. File deleted, store reloaded, weights kept.
    tmp.unlink()
    wiped = EpisodicMemory(path=tmp)
    wipe_recall = _recall_acc(wiped, seen, cbits, obits, now)
    wipe_skill = _skill_acc(net, held, cbits, obits)

    # Ablation 2: revert the weights. The diary snapshot is untouched.
    net.load_state_dict(init)
    revert_skill = _skill_acc(net, held, cbits, obits)
    revert_recall = _recall_acc(intact, seen, cbits, obits, now)

    return {
        "wipe_recall": round(wipe_recall, 4),
        "wipe_skill": round(wipe_skill, 4),
        "revert_skill": round(revert_skill, 4),
        "revert_recall": round(revert_recall, 4),
    }


def _check(m: dict, c: dict) -> bool:
    return (
        # the diary works, and distillation did not eat it
        m["recall_pre"] >= MIN_RECALL
        and m["recall_post"] >= m["recall_pre"] - MAX_RECALL_DROP
        # the skill is real, general, and beats the no-distillation null
        and m["skill_heldout"] >= MIN_SKILL
        and m["null_skill_heldout"] <= MAX_CHANCE_SKILL
        and m["skill_gain"] >= MIN_SKILL_GAIN
        # the diary cannot do the skill's job on never-lived pairs
        and m["store_on_heldout"] <= MAX_CHANCE_SKILL
        # Added 2026-09-06 (78th audit B2 / Review FTB 3; strengthen-only):
        # and it must ABSTAIN on them, not answer with the nearest episode.
        and m["distractor_evaluated"] >= MIN_DISTRACTOR_EVAL
        and m["distractor_abstention"] >= MIN_DISTRACTOR_ABST
        # wipe kills recall, and ONLY recall
        and c["wipe_recall"] <= MAX_WIPE_RECALL
        and c["wipe_skill"] >= m["skill_heldout"] - MAX_ABLATION_DRIFT
        # revert kills the skill, and ONLY the skill
        and c["revert_skill"] <= MAX_CHANCE_SKILL
        and c["revert_recall"] >= m["recall_post"] - MAX_ABLATION_DRIFT
    )


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["ME.10"], _experiment, _check,
                    control_fn=_control, ledger=ledger)
