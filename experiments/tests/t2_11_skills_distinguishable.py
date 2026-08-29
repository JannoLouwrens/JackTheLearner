"""T2.11 — skills are distinguishable, measured by someone who did not train them.

`UnifiedBrain.SkillDiscovery` (DIAYN, arXiv:1802.06070) has been in this
repository since before the ladder existed and has never received a gradient in
a registered experiment. Its docstring says "the robot learns walking, jumping,
turning just from wanting to be distinguishable!" — the exact shape of the
README-says-Working disease SYSTEM.md was written to kill. This spec is the test
that could have failed, and its `kills` field is that class, by name.

THE VACUITY THIS SPEC EXISTS TO AVOID, stated before the design because the
design is downstream of it. DIAYN trains a discriminator q(z|s) to recover the
skill from the state and pays the policy log q(z|s) - log p(z) for making that
easy. So "the discriminator recovers the skill" is TRAINING FIT and proves
nothing: it is the objective, read back. Three separate ways this test could
score high while measuring nothing, and what each forces in the design:

  (1) SCORING WITH THE DISCRIMINATOR. Circular. -> The verdict is read by an
      INDEPENDENT classifier: different architecture, different init, different
      optimiser, different input representation (see FEATURES), trained on
      rollouts the DIAYN policy never trained on and evaluated on a further
      disjoint set.
  (2) A DETERMINISTIC POLICY. If eval rollouts are greedy, MuJoCo is
      deterministic and every rollout of skill z is byte-identical, so the
      classifier's "held-out" set is its training set and 100% is arithmetic.
      -> Eval rollouts carry EVAL_EPS exploration on private RNG streams, and
      `no_leak` gates the actual feature-hash overlap at zero rather than
      trusting that argument.
  (3) ANY PER-SKILL LOTTERY. n private policies with n private random seeds
      produce n systematically different walks whether or not a mutual
      information was ever maximised. This is the one that matters, and it is
      why the claim is NOT "above chance": chance is the registered null and it
      is kept, but the binding gate is a MARGIN over two twins that have the
      identical machinery and no MI (see THE ARMS).

THE ARMS — matched world, matched learner, matched budget, matched number of
gradient steps; they differ in one thing, what the discriminator is trained on:
  diayn     the claim. SkillDiscovery's own `get_discriminator_loss` on
            (state, true skill), its own `compute_diayn_reward` paid to the
            policy. The mechanism under test, imported from UnifiedBrain.py —
            not reimplemented, so a FAIL falls on the shipped component.
  shuffled  THE CONTROL, and it must fail. Identical in every respect except
            that the skill labels are permuted inside each discriminator batch.
            Same nets, same optimiser, same number of steps, same reward
            pathway, same reward magnitudes — the MI is destroyed and nothing
            else is. If skills still come out distinguishable here, this rig
            measures the lottery of (2)/(3) and not DIAYN.
  zero      r = 0.0 to the identical learner: the FLOOR instrument. With
            Q_INIT = 0 its tables never leave zero, so its action choice is a
            uniform tie-break — a random walk, per skill, from an identical
            start. It bounds what "distinguishable" costs when nothing is
            learned at all.

SAID OUT LOUD, because a control that cannot fail is decoration: `zero`'s
failure is close to ENTAILED (exchangeable random walks cannot carry a label
across disjoint rollouts). `shuffled` is NOT entailed — its policy is trained,
it chases a live reward signal of the same magnitude as the claim arm's, and
its skills could genuinely differentiate by chasing different noise. That is
where the discriminating work is done, and `_check` reads `shuffled` as the
declared control for exactly that reason. `zero` is reported and gated only as
a floor.

WHY Q_INIT = 0 HERE, against PG.4's optimistic 3.0. PG.4/T2.09 want a frontier
sweep; this spec wants specialisation, and an optimistic init pays every policy
to leave wherever it is — it fights the thing being measured. Both twins take
the identical init, so the comparison is matched; the constant is not inherited
from PG.4 and is not claimed to be.

FEATURES — what the independent classifier reads, and why it is not the retina.
The discriminator reads the 68-d retina (position, velocity, 32 rays x
[distance, texture]) as its `state_latent`. The classifier reads a purely
KINEMATIC trajectory summary: the normalised visitation histogram over PG.4's
121 floor cells, plus the (x, y) at 8 evenly-spaced checkpoints. Two reasons.
First, independence: a classifier reading the same features as the objective
inherits its blind spots. Second, it is what DIAYN's own docstring claims —
"skills are distinguishable by the STATES THEY VISIT" — so the verdict is
scored against the component's advertised claim rather than a friendlier one.

THE WORLD is PG.4's, imported not copied: same MJCF (n_objects=0), same
non-colliding velocity rover, same 32-ray retina, same 1 m cell grid. The panel
is STATIC (`noisy=False`) throughout — an irreducibly stochastic percept is
T2.09's subject, and it is settled there; letting it in here would confound a
skill's identity with what it happened to see on the TV.

PRE-REGISTERED GATES — PROVISIONAL. `_GATES_FROZEN = False` and `run()` REFUSES
until a pilot on seeds disjoint from the registered ones freezes every bar
below. The numbers written here are placeholders anchored on mechanism, not
measurements; the pilot may move them once, in the open, before the first
registered seed is drawn.

  RIG (violated -> VOID, not FAIL: the apparatus did not ask the question)
    instrument_alive   the label-SHUFFLED classifier's TRAIN accuracy >=
                       SHUFFLE_FIT_FLOOR. LESSONS, "An at-chance control must
                       carry proof its instrument was alive": a classifier with
                       no capacity reads at chance on everything, which would
                       make every at-chance control below pass for the worst
                       possible reason. This is that proof, and it is a rig
                       gate rather than a claim gate because a dead classifier
                       is a broken instrument, not a refuted hypothesis.
    instrument_honest  the label-shuffled classifier's HELD-OUT accuracy <=
                       CHANCE + SHUFFLE_BAND. If a classifier trained on
                       permuted labels still scores on true held-out labels,
                       something outside the label is carrying the signal.
    no_leak            zero feature-hash collisions between the classifier's
                       train and held-out rollout sets, on every arm. See
                       vacuity (2); T3.01's structural leak gate, same idiom.
    body_moved         every arm's mean rollout coverage > 0 and `zero`'s mean
                       coverage >= FLOOR_COVERAGE. The rover moves in this
                       world under the null policy — without it, "skills are
                       indistinguishable" could mean "nothing went anywhere".

  CLAIM (WORST registered seed, never the mean; all four must hold)
    above_chance   diayn held-out accuracy >= CHANCE + ABOVE_CHANCE_MIN. The
                   registered null (`Chance = 1/n_skills`), kept and not
                   weakened — but it is the weakest of the four.
    beats_shuffled diayn - shuffled held-out accuracy >= MARGIN_MIN. THE GATE
                   THAT DECIDES THIS SPEC. It is the one gate applied to the
                   run only: a control cannot be a margin ahead of itself, and
                   wiring it to a constant for the control would make the
                   control unable to fail (`_fold_control` says this in code).
    beats_zero     diayn - zero held-out accuracy >= MARGIN_MIN.
    per_class      min per-skill recall >= PER_CLASS_MIN. Without it, two
                   legible skills out of eight and six indistinguishable ones
                   clear an aggregate bar — "the MI objective collapsed" (the
                   registered `falsified_by`) is exactly what partial collapse
                   looks like, and an aggregate cannot see it.

  CONTROL (must fail the CLAIM gates): `shuffled`, scored on the identical
  claim gates in the same run.

WHAT THIS SPEC DOES NOT TEST, so its `kills` is scoped honestly. The policy
here is tabular, one Q table per skill over 121 cells — so
`SkillDiscovery.skill_embedding` and `get_skill_embedding` (the neural
conditioning path) are NOT exercised. What is exercised, and what a FAIL
therefore falls on, is DIAYN's objective as this repository ships it: the
discriminator, `get_discriminator_loss`, and `compute_diayn_reward`. A FAIL
here says the shipped objective does not produce distinguishable behaviour in
this world; it does not say a skill-conditioned neural policy could not. That
distinction belongs in the record before the run, not after it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ..gpu import build_job, submit
from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from .pg_4_noisy_tv import (
    _ACTIONS, _Retina, _build, _cell, GAMMA, N_RAYS, Q_LR, SPEED, SUBSTEPS,
)

# The world, PG.4's rig, and the component under test all hash in: this spec's
# verdict is about SkillDiscovery in THIS arena, and a change to either must
# stale the certificate loudly.
IMPL_DEPS = ["playground.py", "experiments/tests/pg_4_noisy_tv.py",
             "UnifiedBrain.py"]

_GATES_FROZEN = False           # PROVISIONAL — pilot, freeze, then run.

SEEDS = [0, 1, 2]               # registered; the registry declares 3 seeds
PILOT_SEEDS = (7, 90)           # disjoint from the registered set, and spent
                                # once used — the same two T2.09 piloted on.

# ── the rig ──────────────────────────────────────────────────────────────
N_SKILLS = 8                    # chance = 0.125. Eight is what an 11x11 arena
                                # can plausibly partition; SkillDiscovery's own
                                # default of 50 would put chance at 0.02 and
                                # make every bar below a coin-flip on 121 cells.
N_CELLS = 121                   # PG.4's floor grid
OBS_DIM = 4 + 2 * N_RAYS        # 68 — what the discriminator sees
EPISODE_LEN = 200               # decisions; 0.3 m/decision at full speed, so
                                # ~60 m of travel across an 11 m arena
TRAIN_EPISODES = 200            # 25 per skill, z sampled uniformly
EVAL_ROLLOUTS = 16              # per skill, per split (train / held-out)
EVAL_EPS = 0.10                 # PG.4's EPS_LO: eval is not deterministic, see
                                # vacuity (2)
EPS_HI, EPS_LO = 1.0, 0.10      # training exploration, decayed over the first
                                # third of the episodes
Q_INIT = 0.0                    # NOT PG.4's 3.0 — see WHY Q_INIT = 0 above
DISC_STEPS_PER_EP = 8           # discriminator gradient steps per episode
DISC_BATCH = 128
DISC_LR = 1e-3
CHECKPOINTS = 8                 # (x, y) samples in the trajectory feature
FEAT_DIM = N_CELLS + 2 * CHECKPOINTS

# The independent classifier
CLF_HIDDEN = 64
CLF_LR = 3e-3
CLF_EPOCHS = 300

_ARMS = ("diayn", "shuffled", "zero")
_CLAIM_ARM = "diayn"
_CONTROL_ARM = "shuffled"

# ── PROVISIONAL bars ─────────────────────────────────────────────────────
CHANCE = 1.0 / N_SKILLS         # 0.125 — the registered null baseline
ABOVE_CHANCE_MIN = 0.15         # placeholder: ~2x chance
MARGIN_MIN = 0.15               # placeholder: the deciding gate
PER_CLASS_MIN = 0.20            # placeholder: > chance for EVERY skill
SHUFFLE_FIT_FLOOR = 0.60        # placeholder: the classifier can fit 8 labels
SHUFFLE_BAND = 0.10             # placeholder: shuffled held-out <= 0.225
FLOOR_COVERAGE = 0.05           # placeholder: the random-walk twin moves


# ── one arm, one seed ────────────────────────────────────────────────────
def _feature(cells: list, xy: list) -> "object":
    """Trajectory -> the classifier's input. Kinematic only; see FEATURES."""
    import numpy as np
    f = np.zeros(FEAT_DIM, dtype="float32")
    for c in cells:
        f[c] += 1.0
    f[:N_CELLS] /= max(1, len(cells))
    step = max(1, len(xy) // CHECKPOINTS)
    for i in range(CHECKPOINTS):
        px, py = xy[min(len(xy) - 1, i * step)]
        f[N_CELLS + 2 * i] = px / 6.0
        f[N_CELLS + 2 * i + 1] = py / 6.0
    return f


def _rollout(model, data, retina, q, z: int, rng, eps: float) -> dict:
    """One episode under skill z's table. Returns the transitions and the
    kinematic trace. Always starts from the same reset state, so no skill can
    be identified by where it began."""
    import mujoco
    import numpy as np

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    ax, ay = retina.act_ids
    obs, _ = retina.observe(data)
    cells, xy, states, trans = [], [], [], []
    for _ in range(EPISODE_LEN):
        x, y = float(data.qpos[-2]), float(data.qpos[-1])
        s = _cell(x, y)
        cells.append(s)
        xy.append((x, y))
        states.append(obs)
        if rng.uniform() < eps:
            a = int(rng.randint(len(_ACTIONS)))
        else:
            best = np.flatnonzero(q[s] >= q[s].max() - 1e-12)
            a = int(best[rng.randint(len(best))])
        data.ctrl[ax] = SPEED * _ACTIONS[a][0]
        data.ctrl[ay] = SPEED * _ACTIONS[a][1]
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        obs2, _ = retina.observe(data)
        x2, y2 = float(data.qpos[-2]), float(data.qpos[-1])
        trans.append((s, a, _cell(x2, y2)))
        obs = obs2
    return {"cells": cells, "xy": xy, "states": states, "trans": trans,
            "z": z, "coverage": len(set(cells)) / N_CELLS}


def _train_arm(seed: int, arm: str) -> dict:
    """Train one arm to convergence-of-budget and return its frozen tables.

    Every arm runs the identical loop and the identical number of gradient
    steps. `arm` selects ONLY (a) whether the discriminator sees true or
    permuted labels and (b) whether the DIAYN reward reaches the policy.
    """
    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from UnifiedBrain import SkillDiscovery, UnifiedBrainConfig

    # `_ARMS.index`, never `hash(arm)`: Python randomises string hashing per
    # process, so a hash-seeded net would make this rig non-reproducible across
    # runs while every other seed in it stayed fixed — the quietest possible
    # determinism bug (T0.02's subject).
    torch.manual_seed(seed * 31 + _ARMS.index(arm))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = UnifiedBrainConfig(latent_dim=OBS_DIM)
    disc = SkillDiscovery(cfg, num_skills=N_SKILLS).to(dev)
    opt = torch.optim.Adam(disc.parameters(), lr=DISC_LR)

    model, data, panel_gid, rover_bid, act_ids = _build()
    retina = _Retina(model, panel_gid, rover_bid, False,
                     np.random.RandomState(seed * 7919 + 13))
    retina.act_ids = act_ids
    rng = np.random.RandomState(seed * 104729 + 7)

    q = np.full((N_SKILLS, N_CELLS, len(_ACTIONS)), Q_INIT)
    buf_s, buf_z = [], []
    loss_first = loss_last = None
    r_abs_sum, r_n = 0.0, 0

    for ep in range(TRAIN_EPISODES):
        z = int(rng.randint(N_SKILLS))
        eps = max(EPS_LO, EPS_HI - (EPS_HI - EPS_LO)
                  * ep / max(1, TRAIN_EPISODES // 3))
        roll = _rollout(model, data, retina, q[z], z, rng, eps)

        st = torch.as_tensor(np.stack(roll["states"]), device=dev)
        zt = torch.full((len(roll["states"]),), z, dtype=torch.long, device=dev)
        if arm == "zero":
            rewards = np.zeros(len(roll["trans"]), dtype="float32")
        else:
            with torch.no_grad():
                r, _info = disc.compute_diayn_reward(st, zt)
            rewards = r.cpu().numpy().astype("float32")
        r_abs_sum += float(np.abs(rewards).sum()); r_n += len(rewards)
        # DIAYN pays for the state ARRIVED IN, so transition t is paid
        # r(states[t+1]); the final transition reuses the last state it has.
        arrive = np.concatenate([rewards[1:], rewards[-1:]])

        # Q update, replayed over the episode's own transitions.
        for t, (s, a, s2) in enumerate(roll["trans"]):
            q[z, s, a] += Q_LR * (arrive[t] + GAMMA * q[z, s2].max()
                                  - q[z, s, a])

        buf_s.extend(roll["states"])
        buf_z.extend([z] * len(roll["states"]))

        # Discriminator: identical step count on every arm, including `zero`.
        # `shuffled` permutes the labels INSIDE the batch — same gradients
        # flowing, same magnitudes, no mutual information.
        for _ in range(DISC_STEPS_PER_EP):
            idx = rng.randint(len(buf_s), size=min(DISC_BATCH, len(buf_s)))
            bs = torch.as_tensor(np.stack([buf_s[i] for i in idx]), device=dev)
            bz = np.array([buf_z[i] for i in idx])
            if arm == "shuffled":
                bz = bz[rng.permutation(len(bz))]
            bzt = torch.as_tensor(bz, dtype=torch.long, device=dev)
            loss = disc.get_discriminator_loss(bs, bzt)
            opt.zero_grad(); loss.backward(); opt.step()
            if loss_first is None:
                loss_first = float(loss.detach())
            loss_last = float(loss.detach())

    return {"q": q, "model": model, "data": data, "retina": retina,
            "disc_loss_first": loss_first, "disc_loss_last": loss_last,
            "mean_abs_reward": r_abs_sum / max(1, r_n)}


def _eval_arm(seed: int, arm: str, trained: dict) -> dict:
    """Freeze the tables, draw two DISJOINT rollout sets per skill, and hand
    them to a classifier that had no part in training anything."""
    import numpy as np
    import torch

    q, model, data = trained["q"], trained["model"], trained["data"]
    retina = trained["retina"]
    feats = {"train": [], "test": []}
    labels = {"train": [], "test": []}
    covs = []
    cent = np.zeros((N_SKILLS, 2))
    for z in range(N_SKILLS):
        for split, base in (("train", 500_000), ("test", 900_000)):
            for k in range(EVAL_ROLLOUTS):
                rr = np.random.RandomState(base + seed * 10_000 + z * 100 + k)
                roll = _rollout(model, data, retina, q[z], z, rr, EVAL_EPS)
                feats[split].append(_feature(roll["cells"], roll["xy"]))
                labels[split].append(z)
                covs.append(roll["coverage"])
                if split == "test":
                    cent[z] += np.mean(roll["xy"], axis=0) / EVAL_ROLLOUTS

    xtr = np.stack(feats["train"]); ytr = np.array(labels["train"])
    xte = np.stack(feats["test"]); yte = np.array(labels["test"])

    # Structural leak gate: an identical feature in both splits would make
    # "held-out" a lie. Hashed on the exact bytes, T3.01's idiom.
    htr = {hash(v.tobytes()) for v in xtr}
    hte = {hash(v.tobytes()) for v in xte}
    overlap = len(htr & hte)

    def _fit(x, y, y_eval_true, seed_off: int) -> tuple:
        torch.manual_seed(seed * 977 + seed_off)
        clf = torch.nn.Sequential(
            torch.nn.Linear(FEAT_DIM, CLF_HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(CLF_HIDDEN, N_SKILLS))
        o = torch.optim.Adam(clf.parameters(), lr=CLF_LR)
        xt = torch.as_tensor(x); yt = torch.as_tensor(y, dtype=torch.long)
        for _ in range(CLF_EPOCHS):
            loss = torch.nn.functional.cross_entropy(clf(xt), yt)
            o.zero_grad(); loss.backward(); o.step()
        with torch.no_grad():
            tr_acc = float((clf(xt).argmax(1) == yt).float().mean())
            pred = clf(torch.as_tensor(xte)).argmax(1).numpy()
        acc = float((pred == y_eval_true).mean())
        per = [float((pred[y_eval_true == z] == z).mean())
               for z in range(N_SKILLS)]
        return tr_acc, acc, per

    tr_acc, acc, per = _fit(xtr, ytr, yte, 0)
    # The at-chance control for the INSTRUMENT: same classifier, permuted
    # training labels, scored against the true held-out labels.
    perm = np.random.RandomState(seed * 13 + 5).permutation(len(ytr))
    sh_tr_acc, sh_acc, _ = _fit(xtr, ytr[perm], yte, 1)

    d = np.array([[np.hypot(*(cent[i] - cent[j])) for j in range(N_SKILLS)]
                  for i in range(N_SKILLS)])
    off = d[~np.eye(N_SKILLS, dtype=bool)]

    return {
        "arm": arm,
        "heldout_acc": round(acc, 4),
        "train_acc": round(tr_acc, 4),
        "per_class_min": round(min(per), 4),
        "per_class": [round(p, 4) for p in per],
        "shuffled_clf_train_acc": round(sh_tr_acc, 4),
        "shuffled_clf_heldout_acc": round(sh_acc, 4),
        "hash_overlap": overlap,
        "mean_coverage": round(float(np.mean(covs)), 4),
        "centroid_sep_mean": round(float(off.mean()), 3),
        "disc_loss_first": round(trained["disc_loss_first"] or 0.0, 4),
        "disc_loss_last": round(trained["disc_loss_last"] or 0.0, 4),
        "mean_abs_reward": round(trained["mean_abs_reward"], 4),
    }


_ARM_CACHE: dict = {}


def _arm(seed: int, arm: str) -> dict:
    key = (seed, arm)
    if key not in _ARM_CACHE:
        _ARM_CACHE[key] = _eval_arm(seed, arm, _train_arm(seed, arm))
    return _ARM_CACHE[key]


def remote_run(seeds: list) -> dict:
    """Every arm this spec reads, for every seed. Runs on the GPU VM."""
    out = {"gpu": "cpu", "n_skills": N_SKILLS, "seeds": []}
    try:
        import torch
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    for seed in seeds:
        row = {"seed": seed}
        for arm in _ARMS:
            row[arm] = _arm(seed, arm)
        out["seeds"].append(row)
    return out


# ── GPU submission (one per spec — module cache, T2.01 pattern) ──────────
JOB = r'''
import subprocess as _s, sys as _y, os as _o
_s.run([_y.executable, "-m", "pip", "install", "-q", "mujoco"], check=True)
import json
from experiments.tests.t2_11_skills_distinguishable import remote_run
out = remote_run(__SEEDS__)
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "t211.json"), "w"),
          indent=1)
print("DONE", out["gpu"], flush=True)
'''

_CACHE: dict = {}


def _submit(seeds: list) -> dict:
    body = JOB.replace("__SEEDS__", repr(list(seeds)))
    job = build_job(body)
    # PLACEHOLDER until the pilot measures it. `est_hours` feeds the GPU budget
    # ledger and the watcher timeout, so it is deliberately left as a number
    # this file cannot honour silently: `_GATES_FROZEN` is False and `run()`
    # refuses, so no submission can be made against a guessed cost. The pilot
    # writes the measured seconds-per-seed here in the same commit that flips
    # the freeze (T2.19's rule: calibrate, never guess).
    est_hours = round(0.10 + _SEC_PER_SEED / 3600.0 * len(seeds), 3)
    timeout_s = int(est_hours * 3600 * 1.5) + 900
    res = submit(job, prefer="kaggle", est_hours=est_hours,
                 timeout_s=timeout_s, fetch=["t211.json"])
    if not res.ok:
        raise RuntimeError(f"T2.11 job failed on {res.backend}: {res.message}")
    out = json.loads(Path(res.artifacts["t211.json"]).read_text())
    out["backend"] = res.backend
    return out


_SEC_PER_SEED = 1200.0          # PROVISIONAL — the pilot replaces this


# ── the reading ──────────────────────────────────────────────────────────
def _seed_view(row: dict) -> dict:
    cl, ctl, zero = row[_CLAIM_ARM], row[_CONTROL_ARM], row["zero"]
    return {
        "seed": row["seed"],
        "claim_acc": cl["heldout_acc"],
        "claim_per_class_min": cl["per_class_min"],
        "margin_vs_shuffled": round(cl["heldout_acc"] - ctl["heldout_acc"], 4),
        "margin_vs_zero": round(cl["heldout_acc"] - zero["heldout_acc"], 4),
        # rig
        "shuffle_clf_fit": min(row[a]["shuffled_clf_train_acc"] for a in _ARMS),
        "shuffle_clf_heldout": max(row[a]["shuffled_clf_heldout_acc"]
                                   for a in _ARMS),
        "hash_overlap_max": max(row[a]["hash_overlap"] for a in _ARMS),
        "zero_coverage": zero["mean_coverage"],
        "min_coverage": min(row[a]["mean_coverage"] for a in _ARMS),
        # reported, ungated
        "ctrl_acc": ctl["heldout_acc"],
        "ctrl_per_class_min": ctl["per_class_min"],
        "zero_acc": zero["heldout_acc"],
        "claim_centroid_sep": cl["centroid_sep_mean"],
        "ctrl_centroid_sep": ctl["centroid_sep_mean"],
        "claim_disc_loss_first": cl["disc_loss_first"],
        "claim_disc_loss_last": cl["disc_loss_last"],
        "ctrl_disc_loss_last": ctl["disc_loss_last"],
        "claim_mean_abs_reward": cl["mean_abs_reward"],
        "ctrl_mean_abs_reward": ctl["mean_abs_reward"],
    }


def _fold(rows: list) -> dict:
    """Per-seed rows -> the numbers the gates read: WORST registered seed.

    Never a mean. `run_spec._aggregate` means whatever it is handed, so a spec
    whose gates are worst-case must fold before it returns — T2.09's scar,
    where a docstring saying "worst of N" was scored on the mean until `_fold`
    was written.
    """
    v = [_seed_view(r) for r in rows]

    def w(key, hi: bool):
        return (max if hi else min)(x[key] for x in v)

    return {
        "n_seeds": float(len(v)),
        "chance": CHANCE,
        # claim, worst seed
        "claim_acc": w("claim_acc", False),
        "claim_per_class_min": w("claim_per_class_min", False),
        "margin_vs_shuffled": w("margin_vs_shuffled", False),
        "margin_vs_zero": w("margin_vs_zero", False),
        # rig, worst seed
        "shuffle_clf_fit": w("shuffle_clf_fit", False),
        "shuffle_clf_heldout": w("shuffle_clf_heldout", True),
        "hash_overlap_max": w("hash_overlap_max", True),
        "zero_coverage": w("zero_coverage", False),
        "min_coverage": w("min_coverage", False),
        # reported
        "ctrl_acc": w("ctrl_acc", True),
        "zero_acc": w("zero_acc", True),
        "claim_centroid_sep": w("claim_centroid_sep", False),
        "ctrl_centroid_sep": w("ctrl_centroid_sep", True),
        "claim_disc_loss_last": w("claim_disc_loss_last", True),
        "ctrl_disc_loss_last": w("ctrl_disc_loss_last", False),
        "claim_mean_abs_reward": w("claim_mean_abs_reward", False),
        "ctrl_mean_abs_reward": w("ctrl_mean_abs_reward", False),
        "per_seed": [[x["seed"], x["claim_acc"], x["ctrl_acc"], x["zero_acc"],
                      x["claim_per_class_min"], x["margin_vs_shuffled"],
                      x["hash_overlap_max"]] for x in v],
        "per_seed_cols": ("seed claim_acc ctrl_acc zero_acc claim_per_class_min"
                          " margin_vs_shuffled hash_overlap"),
    }


def _fold_control(rows: list) -> dict:
    """`shuffled` scored on the SHARED claim gates, over the same seeds.

    "Shared" is the load-bearing word, and it is why `_claim_holds` is stated
    over three gates rather than four. `beats_shuffled` cannot be asked of
    `shuffled` — a control cannot be a margin ahead of itself, and scoring it
    with that gate wired to 0.0 would make `not _claim_holds(c)` a tautology,
    i.e. a control that cannot fail. So the fourth gate is applied to the run
    ONLY, in `_check`, and the control faces the three that are satisfiable by
    it: it clears them if its skills really are legible, above chance, in every
    class, by a margin over the same random-walk floor.
    """
    v = [_seed_view(r) for r in rows]
    return {
        "claim_acc": min(x["ctrl_acc"] for x in v),
        "claim_per_class_min": min(x["ctrl_per_class_min"] for x in v),
        "margin_vs_zero": min(round(x["ctrl_acc"] - x["zero_acc"], 4)
                              for x in v),
        "ctrl_per_seed_acc": [x["ctrl_acc"] for x in v],
    }


def _experiment(seed: int) -> dict:
    """`seed` is ignored: one submission runs every seed and `_fold` reduces
    them to the worst. run_spec calls this once per registered seed and means
    identical dicts, so the recorded numbers are the fold."""
    if not _CACHE:
        _CACHE.update(_submit(SEEDS))
    m = _fold(_CACHE["seeds"])
    m["gpu"] = _CACHE["gpu"]
    m["backend"] = _CACHE.get("backend", "?")
    return m


def _control(seed: int) -> dict:
    return _fold_control(_CACHE["seeds"])


def _claim_holds(m: dict) -> bool:
    """The three claim gates BOTH the run and the control can be scored on.

    `beats_shuffled` is deliberately absent — see `_fold_control` for why a
    control asked to beat itself is a control that cannot fail.
    """
    return (m["claim_acc"] >= CHANCE + ABOVE_CHANCE_MIN
            and m["claim_per_class_min"] >= PER_CLASS_MIN
            and m["margin_vs_zero"] >= MARGIN_MIN)


def _check(m: dict, c: dict):
    # A dead instrument or a leaking split is an APPARATUS outcome, not a
    # refutation: FAIL would fire this spec's `kills` field — deleting a
    # shipped component — off a run that never asked the question.
    rig = (m["shuffle_clf_fit"] >= SHUFFLE_FIT_FLOOR
           and m["shuffle_clf_heldout"] <= CHANCE + SHUFFLE_BAND
           and m["hash_overlap_max"] == 0
           and m["min_coverage"] > 0.0
           and m["zero_coverage"] >= FLOOR_COVERAGE)
    if not rig:
        return Status.VOID
    return bool(_claim_holds(m)
                and m["margin_vs_shuffled"] >= MARGIN_MIN
                and not _claim_holds(c))


def run(ledger: Ledger | None = None):
    if not _GATES_FROZEN:
        raise RuntimeError(
            "T2.11 gates are provisional — pilot first, freeze the bars in "
            "this file, then run (SM.02's _GATES_FROZEN idiom).")
    return run_spec(BY_ID["T2.11"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":  # pilot: python -m experiments.tests.t2_11_... SEED OUT
    import time
    _seed = int(sys.argv[1])
    _out = sys.argv[2]
    _t0 = time.time()
    _res = remote_run([_seed])
    _res["wall_s"] = round(time.time() - _t0, 1)
    Path(_out).write_text(json.dumps(_res, indent=1))
    print("PILOT DONE", _out, _res["wall_s"], "s", flush=True)
