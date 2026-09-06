"""T2.12 — the emotion model produces distinguishable states.

If EmotionalState is to be an input modality (its mood embedding conditions the
brain), the PAD trajectory must carry information about what actually happened.
Different lives must feel different: a week of praise and success must leave a
different PAD signature than a week of failure and scolding, and both must
differ from a week of novelty or a week of nothing.

This is a genuine risk, not a formality. In EmotionalState.update() the OCC
event deltas never touch pad_vector directly — the ONLY pathway from an event
into the state is through an UNTRAINED, randomly-initialised GRU
(EmotionalState.py:611, "total_delta = gru_delta"). If that random transform
squashes or ignores its event input, every life produces the same drift and
the modality is dead weight. That is exactly what this spec would catch, and
exactly what its kills-clause says: EmotionalState as an input modality.

Design:
  4 event regimes, each a per-step distribution over EventType:
    thriving   — TASK_SUCCESS .25, USER_PRAISE .10, SKILL_LEARNED .05
    struggling — TASK_FAILURE .25, USER_SCOLD .10, DAMAGE .05
    exploring  — NOVELTY .40
    neglected  — BOREDOM_TICK .50
  (remainder of each step: no event). 40 trajectories per regime, 120 steps
  each, module reset() between trajectories, biological noise left ON at its
  shipped value. Features per trajectory: per-dim mean and std of the PAD
  trace (6-dim). Classifier: nearest class centroid on z-scored features,
  even trajectories train / odd trajectories test. Chance = 1/4.

NULL (must lose): a clamped random walk from the same baseline whose per-step
increment std is matched per-dim to the real trajectories, carrying the same
regime labels it never saw — separability ~ chance by construction.
CONTROL (must fail): the real trajectories with train labels shuffled —
centroids become meaningless, accuracy ~ chance. Kills a classifier that
cheats (e.g. exploits trajectory index).

3 seeds; each seed re-draws the GRU init, the event streams, and the noise.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

# The implementation under test. Undeclared until 2026-09-06 (78th audit
# finding 1.1; grandfather set shrunk here).
IMPL_DEPS = ['EmotionalState.py']

REPO = Path(__file__).resolve().parents[2]

N_TRAJ = 40          # per regime
T_STEPS = 120
N_TEST = N_TRAJ // 2  # odd-indexed trajectories

CHANCE = 0.25
MIN_ACC = 0.80
MAX_NULL_ACC = 0.45
MIN_MARGIN = 0.30
MAX_SHUFFLED_ACC = 0.45

REGIMES = [
    ("thriving",   [("TASK_SUCCESS", 0.25), ("USER_PRAISE", 0.10),
                    ("SKILL_LEARNED", 0.05)]),
    ("struggling", [("TASK_FAILURE", 0.25), ("USER_SCOLD", 0.10),
                    ("DAMAGE", 0.05)]),
    ("exploring",  [("NOVELTY", 0.40)]),
    ("neglected",  [("BOREDOM_TICK", 0.50)]),
]


def _sample_event(rng, table, event_enum):
    u = rng.random()
    acc = 0.0
    for name, p in table:
        acc += p
        if u < acc:
            return event_enum[name]
    return None


def _trajectories(seed: int):
    """All real PAD trajectories: {regime_idx: [[(p,a,d) x T_STEPS] x N_TRAJ]}."""
    sys.path.insert(0, str(REPO))
    import torch
    from EmotionalState import EmotionalState, EmotionalConfig, EventType

    torch.manual_seed(seed)
    emo = EmotionalState(EmotionalConfig())
    baseline = [emo.baseline[i].item() for i in range(3)]

    trajs = {}
    with torch.no_grad():
        for ri, (_, table) in enumerate(REGIMES):
            trajs[ri] = []
            for ti in range(N_TRAJ):
                emo.reset()
                rng = random.Random(seed * 100_000 + ri * 1_000 + ti)
                torch.manual_seed(seed * 100_000 + ri * 1_000 + ti)
                tr = []
                for _ in range(T_STEPS):
                    pad = emo.update(event_type=_sample_event(rng, table, EventType),
                                     dt=1.0)
                    tr.append((pad[0].item(), pad[1].item(), pad[2].item()))
                trajs[ri].append(tr)
    return trajs, baseline


def _features(tr):
    """Per-dim mean and std over the trajectory -> 6-dim feature vector."""
    n = len(tr)
    means = [sum(s[d] for s in tr) / n for d in range(3)]
    stds = [(sum((s[d] - means[d]) ** 2 for s in tr) / n) ** 0.5
            for d in range(3)]
    return means + stds


def _centroid_accuracy(trajs, shuffle_seed=None):
    """Nearest-centroid held-out accuracy. Even trajectories train, odd test.

    If shuffle_seed is given, TRAIN labels are shuffled (the control): the
    centroids no longer correspond to regimes and accuracy must fall to chance.
    """
    train = [(ri, _features(tr)) for ri, trs in trajs.items()
             for i, tr in enumerate(trs) if i % 2 == 0]
    test = [(ri, _features(tr)) for ri, trs in trajs.items()
            for i, tr in enumerate(trs) if i % 2 == 1]

    if shuffle_seed is not None:
        labels = [ri for ri, _ in train]
        random.Random(shuffle_seed).shuffle(labels)
        train = [(l, f) for l, (_, f) in zip(labels, train)]

    nfeat = len(train[0][1])
    mu = [sum(f[j] for _, f in train) / len(train) for j in range(nfeat)]
    sd = [max(1e-9, (sum((f[j] - mu[j]) ** 2 for _, f in train)
                     / len(train)) ** 0.5) for j in range(nfeat)]
    z = lambda f: [(f[j] - mu[j]) / sd[j] for j in range(nfeat)]

    cents = {}
    for ri in trajs:
        fs = [z(f) for l, f in train if l == ri]
        cents[ri] = [sum(f[j] for f in fs) / len(fs) for j in range(nfeat)]

    hits = 0
    for ri, f in test:
        zf = z(f)
        pred = min(cents, key=lambda c: sum((zf[j] - cents[c][j]) ** 2
                                            for j in range(nfeat)))
        hits += pred == ri
    return hits / len(test)


def _matched_walks(trajs, baseline, seed: int):
    """One clamped random walk per real trajectory, per-dim increment std
    matched to the real data, same regime labels it never saw."""
    inc_sd = []
    for d in range(3):
        incs = [tr[t][d] - tr[t - 1][d]
                for trs in trajs.values() for tr in trs
                for t in range(1, len(tr))]
        m = sum(incs) / len(incs)
        inc_sd.append((sum((x - m) ** 2 for x in incs) / len(incs)) ** 0.5)

    walks = {}
    for ri, trs in trajs.items():
        walks[ri] = []
        for ti in range(len(trs)):
            rng = random.Random(seed * 200_000 + ri * 1_000 + ti)
            state = list(baseline)
            tr = []
            for _ in range(T_STEPS):
                state = [max(-1.0, min(1.0, state[d] + rng.gauss(0.0, inc_sd[d])))
                         for d in range(3)]
                tr.append(tuple(state))
            walks[ri].append(tr)
    return walks


def _experiment(seed: int) -> dict:
    trajs, _ = _trajectories(seed)
    acc = _centroid_accuracy(trajs)
    # Diagnostic: mean pleasure of the two valence-opposed regimes.
    mean_p = lambda ri: sum(_features(tr)[0] for tr in trajs[ri]) / N_TRAJ
    return {
        "separability_acc": round(acc, 4),
        "chance": CHANCE,
        "mean_pleasure_thriving": round(mean_p(0), 4),
        "mean_pleasure_struggling": round(mean_p(1), 4),
    }


def _control(seed: int) -> dict:
    trajs, baseline = _trajectories(seed)
    walks = _matched_walks(trajs, baseline, seed)
    return {
        "randomwalk_acc": round(_centroid_accuracy(walks), 4),
        "shuffled_label_acc": round(_centroid_accuracy(trajs, shuffle_seed=seed + 7),
                                    4),
    }


def _check(m: dict, c: dict) -> bool:
    return (m["separability_acc"] >= MIN_ACC
            and c["randomwalk_acc"] < MAX_NULL_ACC
            and m["separability_acc"] - c["randomwalk_acc"] >= MIN_MARGIN
            and c["shuffled_label_acc"] < MAX_SHUFFLED_ACC)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T2.12"], _experiment, _check,
                    control_fn=_control, ledger=ledger)
