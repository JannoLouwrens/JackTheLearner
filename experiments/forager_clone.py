"""forager_clone.py — J15: the learnability proof, part 2 (ORDERS J15).

PLAN stage 1, part "Learning to eat", line "a copy of it inside his own
network, and a learner given the direction to food". Step 2 -> 3. Prior:
behaviour cloning a scripted policy into a small MLP is a known-answer task;
a learner handed the bearing to food is the record's standard "is it the eye
or the learner" control. Nothing here is a part of him: the forager is an
adult's ruler, the clone is a probe of whether HIS network shape can hold it.

ORDERS: "Not before J12 (2) has a body-frame row. Until then: write the two
rules and the data collector only." This file is that and nothing else —
no clone is trained, no E2c run starts, until J12 gate 2 is in WORLD_CARD.
If gate 2 FAILS at d0 this order becomes "say why" and E2b/E2c do not run.

=============================================================================
RULE E2b — the forager copied into brain B's actor network.
Written 2026-11-18 09:3x UTC, before any data is collected or any clone runs.
=============================================================================
Teacher  gate 2's row f4: `eye_forager.EyeForager` unchanged, in
         `eye_forager.make_env(seed, 0.75, row="f4")` (body-frame row128 eye,
         body-frame motor, lazy_contact_audio, native_substep, d0).
Data     `collect`: the teacher's own lives on TRAINING seeds 701..740 (fresh:
         no row, pilot or LAB has used 7xx), 3 lives a seed, life cap 9,000;
         every decision stores (row = o[:256], teacher action a[0:8]) plus,
         as a DIAGNOSTIC ONLY (never a clone input), whether the teacher held
         a target that decision.
Student  brain B's actor shape as the J11 pilot builds it (SBX SAC
         MlpPolicy, net_arch [256, 256] library default): an MLP 256 -> 256
         -> 256 -> 8, tanh output, trained from scratch by mean-squared error
         on the teacher's actions. Input: the body-frame row (256 numbers)
         ONLY — no proprio, no needs, no memory (ORDERS). Adam 1e-3, batch
         256, 20 epochs, 10% held out for the loss report; 3 training
         seeds (clone seeds 0, 1, 2) — the clone is judged per clone seed.
Test     each clone drives the body on FRESH seeds 741..760, d0, 10 lives a
         seed, cap 9,000 — gate 2's bookkeeping (`eye_forager.run_seed`'s
         loop) with the clone as the policy.
Bar      UNMOVED, gate 2's: mean floor meals/life >= 5.0 AND > b4's mean on
         the same seed (white random, rerun on 741..760), on >= 18 of 20
         seeds. E2b PASSES iff >= 2 of the 3 clone seeds pass.
Control  MUST FAIL: the same student trained on the same data with the 128
         (distance, texture) pairs of every input permuted by a fresh
         permutation each sample (the g4 twin's corruption, at training and
         at test). If it passes on >= 2 of 3 clone seeds, E2b is VOID.
Disclosed before the run: the teacher is STATEFUL (a dead-reckoned target,
         bans, cooldowns, a wander hold of 25 decisions); a memoryless clone
         of it can only copy the part of its policy the current row
         determines. A miss with the target-held fraction high is "the
         forager's memory, not the eye" — and says brain B needs memory
         (§5's recurrent rows), not that the eye is blind. That reading is
         written now so a miss cannot be explained afterwards.
Extension  NONE.

=============================================================================
RULE E2c — brain B given the direction to food.
=============================================================================
World    the J11 pilot's arm-K world (food_floor_only, spawn_clear 1.55, d0,
         drain ramp K, body-frame eye AND motor — the default proposed in
         INBOX 11-18 07:40 unless the Director rules otherwise first),
         native_substep on.
Learner  the pilot's B0 SAC unchanged, its observation extended by TWO
         numbers: the bearing (body frame, radians / pi) and the distance
         (metres / 10, clipped to 1) to the nearest uneaten floor food,
         computed from the simulator — a handed sense, scored-and-
         ineligible, never his.
Size     T = 300,000 decisions, 5 seeds (seeds 801..805), J11's frozen
         readings (sustained / eats but starves / slides) and J11's floor
         clause, read from PILOT_B2.md's rule unchanged.
Control  MUST FAIL: the same arm with the two numbers replaced by the bearing
         and distance of a random fixed point re-drawn each life.
Reading  pass (sustained on >= 3 of 5, control < 3) = the task is learnable
         by B when the direction is given -> a miss of B0 in J11 is the eye.
         miss = B cannot learn the task even told where food is -> the
         learner or the world, not the eye. Reported, decides no seat.
Extension  NONE. E2c's code (the two-number wrapper) is written when J11's
         rule is frozen — it inherits that rule's readings verbatim.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import eye_forager as EF  # noqa: E402

TRAIN_SEEDS = tuple(range(701, 741))
TEST_SEEDS = tuple(range(741, 761))
TRAIN_LIVES = 3
D0 = 0.75
N_IN = 2 * EF.N_ROW                       # the row: 128 (distance, texture)
OUT = Path("/data/jack-logs/forager_clone")


def collect_seed(seed: int, lives: int = TRAIN_LIVES, cap: int = 9000,
                 d: float = D0) -> dict:
    """The teacher f4's own lives on one training seed: (row, action, held)."""
    env = EF.make_env(seed, d, row="f4")
    o, info = env.reset(seed=seed)
    pol = EF.make_policy("f4", seed)
    X, A, H, meals = [], [], [], []
    for _life in range(lives):
        pol.new_life()
        ate0 = dict(info["ate_total"])
        n, term = 0, False
        while n < cap:
            row = np.asarray(o[:N_IN], dtype=np.float32).copy()
            a = np.asarray(pol.act(o), dtype=np.float32)
            X.append(row)
            A.append(a.copy())
            H.append(pol.target is not None)
            o, _r, term, _tr, info = env.step(a)
            n += 1
            if term:
                break
        ate = info["ate_total"]
        meals.append(int(ate.get("obj0", 0) - ate0.get("obj0", 0)
                         + ate.get("obj1", 0) - ate0.get("obj1", 0)))
        if not term:
            env.w.respawn()
            env._pending_obs = None
        o, info = env.reset()
    return {"x": np.stack(X), "a": np.stack(A), "held": np.array(H, bool),
            "meals": np.array(meals, int), "seed": seed}


def collect(seeds=TRAIN_SEEDS, out: Path = OUT, lives: int = TRAIN_LIVES) -> list:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in seeds:
        r = collect_seed(s, lives)
        p = out / f"teacher_s{s}.npz"
        np.savez_compressed(p, **r)
        print(f"seed {s}: {len(r['x'])} decisions, meals/life {r['meals'].tolist()}, "
              f"target held {r['held'].mean():.3f}", flush=True)
        paths.append(p)
    return paths


def main(argv):
    if not argv or argv[0] not in ("collect", "smoke"):
        print("usage: forager_clone.py collect [seed ...] | smoke")
        return 2
    if argv[0] == "smoke":
        r = collect_seed(153, lives=1, cap=300)
        print(f"smoke: x {r['x'].shape} a {r['a'].shape} held {r['held'].mean():.3f}")
        return 0
    seeds = tuple(int(s) for s in argv[1:]) or TRAIN_SEEDS
    collect(seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
