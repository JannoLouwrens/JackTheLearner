"""T1.13 — are the language/motion pairs real, or are we training on noise?

This is the cheapest test in the ladder and possibly the most important, because
it checks the DATA rather than the model. Every grounding claim in Tier 2 and
Tier 5 rests on the assumption that when a sample is labelled "walk forward", the
motion attached to it is a walk. If that assumption is false, a grounding module
can still train, still converge, and still report a falling loss — it will simply
have learned the marginal distribution of the six label strings. Nothing about
the loss curve would look wrong.

What the code actually does, MoCapLoader.py:703-706, when no BVH files are found:

    obs, actions = self._get_synthetic_sample()      # a pure sinusoid
    synthetic_labels = ["walk forward", "run forward", "stand still", ...]
    label = synthetic_labels[np.random.randint(len(synthetic_labels))]

The motion is a sinusoid with random amplitude and phase, and the label is drawn
UNIFORMLY AT RANDOM and stapled to it. "walk forward" and "turn left" are
therefore attached to statistically identical motion. The mutual information
between language and movement is zero by construction.

So this test asks two questions the loss curve cannot:

  1. IS THE MOTION REAL?  A pure sinusoid puts nearly all its spectral energy in
     one frequency bin. Real motion capture does not — a human walk has harmonics,
     asymmetry between limbs, and noise. So: FFT each joint trajectory and measure
     the fraction of energy in the dominant non-DC bin.

  2. DOES THE LANGUAGE MEAN ANYTHING?  Group samples by label and compute an
     F-ratio: between-label variance over within-label variance. If labels track
     motion, samples sharing a label resemble each other more than they resemble
     samples with other labels, and F > 1. The CONTROL shuffles the labels; if the
     real F is no better than the shuffled F, the words were decorative.

Falsifying it is the useful outcome. A FAIL here does not mean the model is
broken, it means Tier 2's grounding specs must not be run until there is real
data — and it converts "the MoCap URLs 404 and the loader fabricates" from a
review comment into a number that blocks the ladder.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

# The implementation under test. Undeclared until 2026-09-06 (78th audit
# finding 1.1; grandfather set shrunk here).
IMPL_DEPS = ['MoCapLoader.py']

REPO = Path(__file__).resolve().parents[2]

N_SAMPLES = 240
# Pre-registered, before running.
MAX_SINUSOID_FRACTION = 0.60   # above this, the trajectory is a generated tone
MAX_SYNTHETIC_SHARE = 0.10     # at most 10% of samples may look synthetic
MIN_F_RATIO_ADVANTAGE = 1.5    # real labels must beat shuffled labels by this


def _spectral_purity(traj) -> float:
    """Fraction of AC spectral energy in the single strongest frequency bin.

    A pure sinusoid -> ~1.0. Real motion spreads energy across harmonics.
    """
    import numpy as np
    x = np.asarray(traj, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] < 4:
        return 0.0
    spec = np.abs(np.fft.rfft(x, axis=0)) ** 2
    spec = spec[1:]                       # drop DC
    total = spec.sum()
    if total <= 0:
        return 0.0
    # Per joint, then averaged: a per-joint tone is what the generator produces.
    return float((spec.max(axis=0) / np.maximum(spec.sum(axis=0), 1e-12)).mean())


def _f_ratio(feats, labels) -> float:
    """Between-label variance / within-label variance.

    ~1.0 means labels carry no information about the motion.
    """
    import numpy as np
    feats = np.asarray(feats, dtype=np.float64)
    groups = {}
    for f, l in zip(feats, labels):
        groups.setdefault(l, []).append(f)
    if len(groups) < 2:
        return 0.0
    grand = feats.mean(axis=0)
    between = within = 0.0
    for g in groups.values():
        g = np.asarray(g)
        between += len(g) * float(((g.mean(axis=0) - grand) ** 2).sum())
        within += float(((g - g.mean(axis=0)) ** 2).sum())
    k, n = len(groups), len(feats)
    if n <= k or within <= 0:
        return 0.0
    return (between / (k - 1)) / (within / (n - k))


def _collect(seed: int):
    """Pull N samples from the loader EXACTLY as training would."""
    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch
    from MoCapLoader import MoCapConfig, MoCapDataset

    np.random.seed(seed)
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = MoCapDataset(MoCapConfig())

    feats, labels, purity = [], [], []
    for i in range(min(N_SAMPLES, max(len(ds), 1))):
        obs, actions, label = ds[i % max(len(ds), 1)]
        a = actions.numpy() if hasattr(actions, "numpy") else np.asarray(actions)
        purity.append(_spectral_purity(a))
        # A motion signature: per-joint mean, spread and range. Enough for two
        # different gaits to look different, and cheap.
        feats.append(np.concatenate([a.mean(0), a.std(0), a.max(0) - a.min(0)]))
        labels.append(label)
    return np.asarray(feats), labels, np.asarray(purity), len(ds)


def _experiment(seed: int) -> dict:
    import numpy as np
    feats, labels, purity, n = _collect(seed)
    synth_share = float((purity > MAX_SINUSOID_FRACTION).mean())
    return {
        "dataset_len": n,
        "samples_checked": len(labels),
        "distinct_labels": len(set(labels)),
        "mean_spectral_purity": round(float(purity.mean()), 4),
        "synthetic_share": round(synth_share, 4),
        "real_f_ratio": round(_f_ratio(feats, labels), 4),
        "_feats": feats.tolist(),
        "_labels": labels,
    }


def _control(seed: int) -> dict:
    """Shuffle the labels. If nothing changes, the labels were never signal."""
    import numpy as np
    feats, labels, _, _ = _collect(seed)
    rng = np.random.RandomState(seed + 31)
    shuffled = list(labels)
    rng.shuffle(shuffled)
    return {"shuffled_f_ratio": round(_f_ratio(feats, shuffled), 4)}


def _check(m: dict, c: dict) -> bool:
    m.pop("_feats", None)
    m.pop("_labels", None)
    adv = m["real_f_ratio"] / max(c["shuffled_f_ratio"], 1e-9)
    m["label_signal_advantage"] = round(adv, 3)

    reasons = []
    if m["dataset_len"] == 0:
        reasons.append("the dataset is EMPTY — every sample is fabricated on demand")
    if m["synthetic_share"] > MAX_SYNTHETIC_SHARE:
        reasons.append(
            f"{m['synthetic_share']:.0%} of trajectories are single-tone "
            f"(spectral purity > {MAX_SINUSOID_FRACTION}) — generated, not captured")
    if adv < MIN_F_RATIO_ADVANTAGE:
        reasons.append(
            f"labels carry no motion signal: F={m['real_f_ratio']:.3f} real vs "
            f"{c['shuffled_f_ratio']:.3f} shuffled ({adv:.2f}x, need "
            f"{MIN_F_RATIO_ADVANTAGE}x)")
    if reasons:
        m["verdict"] = ("Grounding data is not real. " + "; ".join(reasons)
                        + ". Tier 2 grounding specs must NOT run on this data — a "
                          "loss curve computed here would fall while learning nothing.")
        return False
    return True


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T1.13"], _experiment, _check, control_fn=_control, ledger=ledger)
