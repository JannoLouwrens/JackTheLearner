"""T3.10 — Trunk knowledge survives action training.

HYPOTHESIS (registry). "Linear probes on frozen-trunk features (object class,
color, spatial relation) hold constant through action training AND semantic-task
success tracks probe quality." FALSIFIED BY: "Probes drift (gradient leak — a
bug), or probes hold while semantic tasks sit at chance (knowledge not reaching
the action head — architecture flaw)." NULL: "Probes on a random-weight trunk."
CONTROL: "Deliberately unfreezing the trunk must reproduce the drift."
METRIC: probe_drift.

WHAT IS ACTUALLY UNDER TEST, AND WHAT IS NOT — read this before citing any
number here in a D1 argument. The registry note calls this "the cheapest direct
evidence for/against decision D1". D1 asks where the **57M multimodal
backbone** belongs in the control path. This spec does NOT train that backbone:
it measures **the vision trunk that is actually seated** — the ~245K
`PrismaticVisionEncoder` CNN fallback, `UnifiedBrain.py:627`, the seat holder
`CHAMPIONS.md` marks "DEFAULT, never defended". What generalises from here is
the MECHANISM (does a frozen upstream representation keep its knowledge through
action training, and does that knowledge reach a learned head); what does NOT
generalise is any number, because a 245K CNN is not a 57M transformer. A run of
this spec is one wall of D1's room, not the room.

PLASTIC-ONLY CAUTION (decree 2026-08-09, GOAL.md:76; T2.03's precedent). A PASS
here does not seat a frozen trunk inside Jack — frozen components inside him are
constitutionally barred, and D1's armed default STRIKES its option A for exactly
that reason. The `frozen` arm is a MEASUREMENT arm. Its value is that it isolates
the question "is the knowledge reachable?" from "is the knowledge destroyed?",
which the plastic arm confounds by construction. The plastic arm is here too,
and it is the CONTROL.

────────────────────────────────────────────────────────────────────────────
THE RIG

The world is PG.6's certified eye (EYE_POS/EYE_XYAXES/EYE_FOVY) in an empty
playground (`n_objects=0`, no water) with one probe body whose geom TYPE, SIZE
and RGBA are edited in the compiled model per episode — T2.03's technique, so no
`playground.py` certificate goes stale under this spec. Rendered at PG.6's
certified RES=96.

Three labels, drawn INDEPENDENTLY and balanced EXACTLY by construction
(`i % 4`, `(i // 4) % 4`, `(i // 16) % 2` with n a multiple of 32, so every one
of the 32 cells appears equally often and no label can be read off another):

  shape   4-way: sphere / box / cylinder / capsule       ("object class")
  colour  4-way: one of four fixed rgba                   ("color")
  near    2-way: distance drawn from [2.2,2.6] or [3.2,3.6] ("spatial relation")

`near` is a spatial relation between Jack's eye and the object — "is it within
reach" — and it is deliberately the one used INSTEAD of left/right bearing:
bearing is confounded here, because the eye sees a different piece of the fixed
playground on the left than on the right, so a bearing probe could read the
BACKGROUND and score high while knowing nothing about the object. Bearing is
still drawn (±22°, PG.6's certified band) but as NUISANCE, never as a label.
`SIZE_RANGE` is narrowed to (0.10, 0.14) — uniformly across all four classes, so
it leaks no shape information — which leaves apparent angular size dominated by
distance. That is honest: monocular near/far IS apparent size.

THE SEMANTIC TASK. 8-way, `action = shape * 2 + near`. It needs shape AND the
spatial relation and it needs NOTHING ELSE. **Colour is a deliberate distractor:
no gradient in phase A ever asks for it.** That is what makes the control sharp
— an unfrozen trunk has every reason to discard colour, and the colour probe is
where its drift should show first.

THE PHASES, per internal seed:

  P (perception)  trunk + three linear heads trained on shape/colour/near.
                  This INSTALLS the knowledge whose survival is the subject.
                  Checkpoint θ_P; SHA-256 over its parameter bytes.
  probe θ_P       ridge probe (T2.03's `_Probe`, imported) per target →
                  `probe_before`. Same on a random-init trunk → the registry's
                  NULL, `probe_random`.
  A (action)      three arms, all from the SAME θ_P where applicable, all
                  training the SAME action-head architecture for the SAME steps:
                    frozen     trunk .eval(), requires_grad_(False), only the
                               head in the optimiser.
                    unfrozen   CONTROL: trunk trains at the head's own lr. A
                               deliberately weak trunk lr would be a control
                               engineered not to fire.
                    randtrunk  NULL for reachability: a random-init trunk,
                               frozen, same head, same steps.
  probe again     `probe_after_frozen`, `probe_after_unfrozen`.

`probe_drift_<arm> = max over the three targets of |after − before|`. It is an
ABSOLUTE difference on purpose: a probe that gets BETTER under unfrozen action
training has still moved, and movement is all the control has to prove. The
signed per-target numbers are reported beside it so a reader can see which way
each went.

THE NULL IS NOT ONE GATE, BECAUSE THE THREE TARGETS ARE NOT ONE KIND OF THING —
and the local smoke is what showed it. `colour` is very nearly free: mean RGB
survives a global average pool, so a RANDOM-weight trunk reads it well and no
amount of phase-P training can add a margin over that. Gating "knowledge exists"
as one margin over all three targets would therefore VOID every honest run. The
two roles are separated instead:

  shape, near   the knowledge the action head MUST use. These carry the
                registry's null: each must beat the random-weight trunk.
  colour        the distractor. It carries no null — it is required only to be
                HIGH before phase A, so that there is room for the control's
                drift to appear in it at all.

WHY THE FROZEN ARM'S DRIFT IS A RECEIPT AND NOT THE CLAIM — stated because
LESSONS ("A worst-case instrument gated on the SEED MEAN…", and the T3.06
exception) requires a tautological gate to name itself. Under a CORRECT freeze
the trunk's parameters are bit-identical afterwards, so its features are
identical, so `probe_drift_frozen` is exactly 0.0 and the gate cannot fail. It is
kept because the failure it detects is real and is NOT a gradient: a trunk left
in `.train()` mode, a BatchNorm running-stat update, an optimiser handed
`model.parameters()` instead of `head.parameters()` — all move features with
`requires_grad=False` everywhere. `frozen_params_identical` is the same receipt
stated in bytes. THE LOAD-BEARING CONJUNCT IS REACHABILITY: does the frozen
trunk's knowledge beat a random trunk's at driving the head? That is the branch
of `falsified_by` no receipt can satisfy, and it is where this spec can lose.

────────────────────────────────────────────────────────────────────────────
GATES — FROZEN AT THIS COMMIT, EXOGENOUS OR RELATIVE, NO PILOT NUMBERS IN THEM

Every bar below is either a chance level, a multiple of chance, a relative
margin against an arm measured in the same run, or a multiple of the binomial
noise of the test split — never a fraction of an observed effect. **Not one of
them has moved, and none may.** `_GATES_FROZEN` is nevertheless **False**: see
the PILOT RECORD below, which found the gates unsatisfiable-by-arithmetic rather
than merely unmet, and the repair pre-registered under it. `run()` refuses until
a pilot shows the repair worked, because a dispatch whose VOID is known in
advance is not a measurement — it is 20 GPU-minutes spent to be told something
this file already says.

PILOT RECORD — seed 90, Colab **Tesla T4**, 2026-08-30T02:19Z, head `ea99989`,
one submission, artifact fetched. It cost ~6 minutes and it stopped a registered
run that could not have succeeded. **THE RIG IS ALIVE ON A GPU**, and every
receipt held:

  canary_ok true, canary_colors 1362, mean_tries 1.56
  n_params_trunk 244960                     ← the 245K seat holder, unchanged
  frozen_params_identical true, probe_drift_frozen **0.0**  ← the freeze is real
  probe_drift_unfrozen **0.1875** (≥0.10)   ← THE CONTROL FIRES: unfreezing
                                               moves the probes, so a zero in
                                               the frozen arm means something
  action_acc_unfrozen 0.3867 (≥0.25)        ← the 8-way task is learnable
  action_acc_frozen 0.2826, action_acc_randtrunk 0.1250 (= chance exactly)
  reach_margin **0.1576** (≥0.10)           ← the load-bearing conjunct CLEARED

**AND THE ONE THING THAT DID NOT HOLD IS THE PREMISE.** Probe accuracies,
random-weight trunk vs after phase P:

    target    random    after phase P     max margin the null leaves
    shape     0.4193    0.3633            +0.5807   gate reachable
    colour    0.9245    0.7122            +0.0755   IMPOSSIBLE
    near      0.9427    0.6849            +0.0573   IMPOSSIBLE

Two findings, and the second is worth more than this spec:

  1. **`near` and `colour` are free from a random projection.** 0.94 and 0.92
     against chances of 0.50 and 0.25. So `min over {shape, near} of
     (before − random) >= 0.15` demanded `probe_before["near"] >= 1.09`. The
     gate was unsatisfiable at commit time, by any trunk, at any budget — and
     the local toy smoke did not catch it because at n=32 its standard error is
     ±0.09, wider than the 0.15 margin it was being used to price. Generalised
     into `null_admissible` above so it cannot recur here, and into LESSONS.md
     so it cannot recur elsewhere.
  2. **Supervised training made the seated trunk a WORSE linear feature
     extractor on all three targets** (shape 0.4193 → 0.3633, colour 0.9245 →
     0.7122, near 0.9427 → 0.6849), while `action_acc_randtrunk` landed on
     0.1250 — chance to four figures. That is a real measurement about the
     vision seat and it corroborates T2.03 from the opposite direction: T2.03
     found the never-trained encoder is a structured random projection; this
     finds that training it lightly makes it a poorer one. It is NOT yet
     evidence about `frozen vs plastic`, because phase P was under-trained —
     `final_perception_loss` 2.2246 against a chance sum of 3.4655 (ln4 + ln4 +
     ln2) is a run that had started to learn and stopped, and an under-trained
     trunk cannot be said to have tested whether knowledge survives.

REPAIR 1, PRE-REGISTERED HERE BEFORE IT RUNS, AND IT IS THE ONLY ONE ALLOWED
(the SM.02 / UB.10 one-diagnostic cap; two specs have now reached a both-fail
branch and in both the cap stopped a plausible third recipe):
  (a) `EPOCHS_P` 40 → 150. An apparatus repair, not a bar: the loss above says
      phase P had not converged, and nobody would claim an under-trained trunk
      tested this hypothesis. No gate moves.
  (b) `null_admissible` becomes mechanism, as specified above. No gate moves;
      the rule states in advance how an impossible conjunct is handled instead
      of leaving it to be discovered per-spec.
THE FORK, and whichever branch fires is recorded rather than argued:
  (i)  the next pilot clears +0.15 on at least one admissible task target →
       set `_GATES_FROZEN = True`, dispatch the registered run, no further
       repairs.
  (ii) it does not → **T3.10 PARKS** with finding 2 above as its result, and the
       redesign question goes to the Review, NOT to a third recipe: *what probe
       target can a 128-d globally-pooled bottleneck learn that its random
       initialisation cannot already read?* Note the shape of the trap before
       answering it — colour and apparent size are low-order image statistics
       that survive any random projection, so the honest candidates are
       relational or compositional, and this world may not be able to pose one.

Binomial σ at n_test = 768 is sqrt(0.25/768) = 0.018. Read every "σ" below as
that number.

  RIG → VOID (an invalid run is not evidence about the hypothesis)
    canary byte-stable across the run, and ≥100 distinct colours in the canary
      frame (a GL context can come up rendering a uniform frame and look exactly
      like a blind sensor — PG.6's measured lesson)
    trunk parameter count in [220K, 270K] — the spec names the 245K seat holder;
      a different number means the seat holder changed under the test
    KNOWLEDGE EXISTS: min over seeds and over the NULL-ADMISSIBLE task targets
      of (probe_before − probe_random) ≥ 0.15 (≈8σ). If phase P installed
      nothing the trunk did not already have, there is no knowledge whose
      survival could be measured, and a drift of zero is vacuous rather than
      reassuring. Colour is excluded by role for the reason given above.
    AT LEAST ONE TASK TARGET IS NULL-ADMISSIBLE. A target is null-admissible
      when `probe_random[t] <= 1 − 0.15`, i.e. when the margin the gate demands
      is arithmetically available at all. This rule is the 2026-08-30 pilot made
      mechanical and it is the reason the constant above is a rule rather than a
      list: at n_test=768 the random-weight trunk read `near` at **0.9427** and
      `colour` at **0.9245**, so a gate demanding +0.15 over the null on those
      targets could not be cleared by ANY trunk, at any training budget, ever.
      A conjunct whose null has already won is not a conjunct — it is an
      arithmetic impossibility wearing a threshold's clothes. It is excluded
      BY RULE, computed from the run's own null before the claim is evaluated,
      and the exclusion is recorded in `null_admissible` so a reader sees which
      conjuncts actually carried the claim. If NONE of the task targets is
      admissible, that is VOID and it is a finding about the trunk: this world
      poses no probe target the seat holder could learn that its random
      initialisation does not already read.
    THE DISTRACTOR HAS HEADROOM: `probe_before["colour"]` ≥ 0.50 (2× chance) on
      every seed. Below chance + the control's own 0.10 floor the control could
      not show a colour drop even if unfreezing destroyed the information, and a
      gate whose failure mode is arithmetic tells you nothing about the world.
    THE DRIFT INSTRUMENT IS ALIVE: `probe_drift_unfrozen` ≥ 0.10 (≈5σ) on EVERY
      seed. This is the registry's control and it is a VOID gate, not a FAIL
      gate: if unfreezing does not move the probes, the frozen arm's zero proves
      nothing about freezing — it proves the probe cannot see. (LESSONS: "An
      at-chance control must carry proof its instrument was alive"; the same
      rule inverted for a zero-drift result.)
    THE TASK IS LEARNABLE: `action_acc_unfrozen` ≥ 0.25 (2× chance) on every
      seed. If nobody can learn the 8-way task, the reachability conjunct is
      measuring the task, not the architecture.

  CLAIM → PASS iff all four
    frozen_params_identical on every seed                        (receipt)
    probe_drift_frozen ≤ 0.02 on every seed — under 1σ           (receipt/leak)
    action_acc_frozen ≥ action_acc_randtrunk + 0.10 (≈5σ) on every seed
                                                    ← THE LOAD-BEARING CONJUNCT
    action_acc_frozen ≥ 0.25 (2× chance 0.125) on every seed

The two FAIL branches map onto `falsified_by` one-to-one: a frozen arm that
drifts is the gradient-leak bug; a frozen arm that holds its probes while its
action accuracy sits at or below the random trunk's is knowledge that does not
reach the head — the architecture flaw, and the finding D1 would actually want.

SEEDS. The registry says `seeds=1`, and that is honoured literally: `run_spec`
calls `_experiment(0)` once. Three INTERNAL seeds [0,1,2] run inside that one
call and are folded to their WORST CASE there — T1.08's and T2.09's idiom, and
today's LESSONS entry is the reason it is not done the other way: `protocol.py`
`_aggregate` takes the MEAN across spec-level seeds before `_check` sees
anything, so a per-seed worst case declared at the spec level would be averaged
away. Folding inside `_experiment` keeps "every seed clears it" literal. It also
costs one clone and one queue wait instead of three.

GPU. One submission for the whole spec (module cache — the T2.01 pattern;
`run_spec` calls `_experiment` once per seed and would otherwise pay per seed).
The job clones the pushed repo, sets MUJOCO_GL=egl, pip-installs mujoco, and
imports THIS module: science code lives here where the guards can see it, never
inside the JOB string (T0.16 lesson).

NO `COVERS:` LINE, deliberately. `T3.10`'s registry entry declares none, and a
docstring that declared one would be a claim the registry cannot see — the exact
drift `T0.21 P10` was built to catch, one day after it caught T2.19's.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..gpu import build_job, submit
# The probe body and the shape draws are T2.03's, reused rather than copied so
# the two specs cannot drift apart; the world and the seat-holding encoder are
# the claim's other two halves. All three hash into the certificate.
from .t2_03_pretrained_vision import _Probe, _rand_quat

IMPL_DEPS = ["playground.py", "UnifiedBrain.py",
             "experiments/tests/pg_6_playground_eyes.py",
             "experiments/tests/t2_03_pretrained_vision.py"]

# False, and NOT because a bar is un-piloted: the 2026-08-30 pilot found two of
# the three gated conjuncts unsatisfiable by arithmetic (see PILOT RECORD).
# run() refuses until a pilot shows REPAIR 1 worked. No gate has moved or may.
_GATES_FROZEN = False

INNER_SEEDS = [0, 1, 2]         # folded to the worst case inside _experiment
PILOT_SEED = 90                 # disjoint; a RIG check, it cannot move a gate

RES = 96                        # PG.6's certified resolution
N_TRAIN, N_TEST = 2048, 768     # both multiples of 32 → exact balance
CELL = 32                       # 4 shapes x 4 colours x 2 distance bands

SHAPES = ("sphere", "box", "cylinder", "capsule")
COLOURS = ((0.85, 0.20, 0.15), (0.20, 0.55, 0.85),
           (0.30, 0.70, 0.25), (0.85, 0.75, 0.20))
N_ACTIONS = len(SHAPES) * 2                    # shape x near
CHANCE_ACTION = 1.0 / N_ACTIONS                # 0.125

DIST_NEAR = (2.2, 2.6)          # inside PG.6's certified distance band, with a
DIST_FAR = (3.2, 3.6)           # gap so the label has no boundary ambiguity
SIZE_RANGE = (0.10, 0.14)       # narrow, uniform over classes: no shape leak
IN_FOV_MAX = 22.0               # deg, PG.6's certified bearing band (NUISANCE)

L2_GRID = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)
VAL_EVERY = 5                   # every 5th train row selects l2

# EPOCHS_P 40 -> 150 is REPAIR 1(a): the pilot's final_perception_loss was 2.2246
# against a chance sum of 3.4655, i.e. phase P had started to learn and stopped.
# An apparatus repair, pre-registered in the docstring; no gate moves with it.
EPOCHS_P, EPOCHS_A = 150, 40
BATCH = 64
LR = 1e-3

# ── gates, frozen at this commit ─────────────────────────────────────────
MIN_CANARY_COLORS = 100
PARAMS_RANGE = (220_000, 270_000)
KNOWLEDGE_MARGIN = 0.15         # ~8 sigma, on TASK_TARGETS only; see docstring
COLOUR_HEADROOM = 0.50          # 2x chance: room for the control to show a drop
CONTROL_DRIFT_MIN = 0.10        # ~5 sigma; else the probe cannot see drift
TASK_LEARNABLE_FLOOR = 2 * CHANCE_ACTION        # 0.25
DRIFT_MAX = 0.02                # under 1 sigma
REACH_MARGIN = 0.10             # ~5 sigma over the random-trunk null


# ── the world, rendered (runs wherever the job runs) ─────────────────────
_EYES: dict = {}


def _get_eye(seed: int):
    """One compiled playground + renderer per seed, held for the process
    lifetime: a garbage-collected `mujoco.Renderer` poisons the shared GL
    context and the NEXT renderer returns corrupted-but-realistic frames with
    no error (PG.6's measured lesson)."""
    if seed not in _EYES:
        _EYES[seed] = _SceneEye(seed)
    return _EYES[seed]


class _SceneEye:
    """PG.6's certified eye pointed at a probe body whose type, size and colour
    are edited in the compiled model. `geom_rbound` is maintained by hand —
    `mj_forward` does not recompute it and `mj_ray` prunes on it."""

    def __init__(self, seed: int):
        # pg_6 FIRST: its module-level ensure_gl() must precede any mujoco
        # import in this process. On a GPU VM with MUJOCO_GL=egl it is a no-op.
        from . import pg_6_playground_eyes as P6
        import mujoco
        import playground as pg
        self._mujoco, self._P6 = mujoco, P6
        params = pg.PlaygroundParams(seed=seed, n_objects=0)
        self.model, self.data, _ = pg.make_playground(
            params, with_water=False, probe_objects=(("probe0", 0.0, 0.0, 0.10),))
        self.gid = self.model.geom("probe0").id
        self.bid = self.model.body("probe0").id
        self.qadr = self.model.jnt_qposadr[self.model.body_jntadr[self.bid]]
        self.r = mujoco.Renderer(self.model, height=RES, width=RES)
        self._canary0 = None
        self._canary0 = self.canary()

    # -- canary ----------------------------------------------------------
    _CANARY = dict(bearing_deg=7.0, dist=3.0, cls="sphere", size3=(0.12, 0.0, 0.0),
                   quat=(1.0, 0.0, 0.0, 0.0), rbound=0.12, rgb=COLOURS[0])

    def canary(self) -> float:
        return float(np.round(
            self._frame_raw(**self._CANARY).astype(np.float64).sum(), 3))

    def canary_colors(self) -> int:
        f = self._frame_raw(**self._CANARY)
        return int(len(np.unique(f.reshape(-1, 3), axis=0)))

    # -- scene editing ---------------------------------------------------
    def _set_geom(self, cls, size3, rbound, rgb):
        m, mj = self.model, self._mujoco
        m.geom_type[self.gid] = {
            "sphere": mj.mjtGeom.mjGEOM_SPHERE, "box": mj.mjtGeom.mjGEOM_BOX,
            "cylinder": mj.mjtGeom.mjGEOM_CYLINDER,
            "capsule": mj.mjtGeom.mjGEOM_CAPSULE}[cls]
        m.geom_size[self.gid] = size3
        m.geom_rbound[self.gid] = rbound
        m.geom_rgba[self.gid] = (rgb[0], rgb[1], rgb[2], 1.0)

    def _place(self, bearing_deg, dist, rbound, quat):
        # PG.6's _place maps (bearing, dist) to world xy and uses its third
        # argument only for z = arg + 0.02, so passing rbound keeps every
        # orientation of every shape clear of the floor.
        x, y, z = self._P6._place(bearing_deg, dist, rbound)
        q = self.qadr
        self.data.qpos[q:q + 3] = (x, y, z)
        self.data.qpos[q + 3:q + 7] = quat
        self.data.qvel[:] = 0.0
        self._mujoco.mj_forward(self.model, self.data)
        return x, y, z

    def _frame_raw(self, bearing_deg, dist, cls, size3, quat, rbound, rgb):
        self._set_geom(cls, size3, rbound, rgb)
        self._place(bearing_deg, dist, rbound, quat)
        self.r.update_scene(self.data, camera="eye")
        return self.r.render()                       # uint8 (RES, RES, 3)

    def unoccluded(self, bearing_deg, dist, cls, size3, quat, rbound, rgb) -> bool:
        """PG.6's registered visibility rule: ONE centre ray, so partially
        occluded episodes stay in — they are the hard ones."""
        self._set_geom(cls, size3, rbound, rgb)
        x, y, z = self._place(bearing_deg, dist, rbound, quat)
        origin = np.asarray(self._P6._eye_frame()[0], dtype=np.float64)
        vec = np.array([x, y, z], dtype=np.float64) - origin
        vec /= np.linalg.norm(vec)
        gid = np.zeros(1, dtype=np.int32)
        self._mujoco.mj_ray(self.model, self.data, origin, vec, None, 1, -1, gid)
        return int(gid[0]) == self.gid


def _draw_shape(cls: str, rng):
    """(size3, rbound) for one episode. The SCALE draw is identical for every
    class — narrowed relative to T2.03 so apparent size tracks distance — while
    the aspect draws give box/cylinder/capsule real outline variety."""
    s = rng.uniform(*SIZE_RANGE)
    if cls == "sphere":
        return (s, 0.0, 0.0), s
    if cls == "box":
        hx, hy, hz = s, s * rng.uniform(0.6, 1.4), s * rng.uniform(0.6, 1.4)
        return (hx, hy, hz), math.sqrt(hx * hx + hy * hy + hz * hz)
    if cls == "cylinder":
        h = s * rng.uniform(0.8, 2.0)
        return (s, h, 0.0), math.sqrt(s * s + h * h)
    h = s * rng.uniform(0.8, 2.0)                    # capsule
    r = s * rng.uniform(0.6, 1.0)
    return (r, h, 0.0), r + h


def _build_dataset(seed: int, n: int):
    """n frames with exactly balanced (shape, colour, near) over all 32 cells.

    Returns (X uint8 [n,RES,RES,3], labels dict of int arrays, mean rejection
    tries). The rng seed is taken mod 2**32 because numpy refuses larger seeds
    and the TEST split's derived value overflows — T2.03 met that exact line for
    the first time 131 s into a Kaggle kernel.
    """
    assert n % CELL == 0, f"n={n} must be a multiple of {CELL} to balance"
    eye = _get_eye(seed)
    rng = np.random.RandomState((seed * 100_003 + n) % 2**32)
    X = np.empty((n, RES, RES, 3), dtype=np.uint8)
    y_shape = np.empty(n, dtype=np.int64)
    y_colour = np.empty(n, dtype=np.int64)
    y_near = np.empty(n, dtype=np.int64)
    tries_total = 0
    for i in range(n):
        si, ci, ni = i % 4, (i // 4) % 4, (i // 16) % 2
        cls, rgb = SHAPES[si], COLOURS[ci]
        band = DIST_NEAR if ni == 1 else DIST_FAR
        for k in range(200):
            b = rng.uniform(0.0, IN_FOV_MAX) * (1 if rng.rand() < 0.5 else -1)
            d = rng.uniform(*band)
            size3, rbound = _draw_shape(cls, rng)
            quat = _rand_quat(rng)
            if eye.unoccluded(b, d, cls, size3, quat, rbound, rgb):
                break
        else:
            raise RuntimeError(f"no unoccluded {cls} in 200 draws (seed {seed})")
        tries_total += k + 1
        X[i] = eye._frame_raw(b, d, cls, size3, quat, rbound, rgb)
        y_shape[i], y_colour[i], y_near[i] = si, ci, ni
    labels = {"shape": y_shape, "colour": y_colour, "near": y_near}
    return X, labels, tries_total / n


TARGETS = (("shape", len(SHAPES)), ("colour", len(COLOURS)), ("near", 2))
# The two the semantic task needs, and therefore the two the registry's
# random-weight null is asked about. `colour` is the distractor (see docstring).
TASK_TARGETS = ("shape", "near")
DISTRACTOR = "colour"


def _action_labels(labels: dict) -> np.ndarray:
    """The semantic task: shape x near, 8-way, exactly balanced. Colour is
    NEVER read — that is what makes it the distractor the control needs."""
    return labels["shape"] * 2 + labels["near"]


# ── the probe: T2.03's ridge, one Gram shared across the l2 grid ─────────
def _probe_all(Xtr: np.ndarray, ltr: dict, Xte: np.ndarray, lte: dict) -> dict:
    """Accuracy per target off ONE pair of Gram matrices. l2 is selected on a
    deterministic every-5th-row split of TRAIN; test labels are never seen."""
    val = np.arange(len(Xtr)) % VAL_EVERY == 0
    fit = ~val
    inner, full = _Probe(Xtr[fit]), _Probe(Xtr)
    out = {}
    for name, k in TARGETS:
        ytr, yte = ltr[name], lte[name]
        best_l2, best_acc = None, -1.0
        for l2 in L2_GRID:
            pred = inner.predict_classes(ytr[fit], Xtr[val], l2, k)
            acc = float((pred == ytr[val]).mean())
            if acc > best_acc:
                best_acc, best_l2 = acc, l2
        pred = full.predict_classes(ytr, Xte, best_l2, k)
        out[name] = round(float((pred == yte).mean()), 4)
    return out


# ── trunk, heads, training ───────────────────────────────────────────────
def _make_trunk(seed: int, device):
    """The seat holder exactly as shipped: `PrismaticVisionEncoder` on its CNN
    path (`use_pretrained_vision=False`), ~245K params."""
    import torch
    from UnifiedBrain import UnifiedBrainConfig, PrismaticVisionEncoder
    torch.manual_seed(seed)
    return PrismaticVisionEncoder(UnifiedBrainConfig()).to(device)


def _param_sha(module) -> str:
    h = hashlib.sha256()
    for _, p in sorted(module.state_dict().items()):
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _features(trunk, X: np.ndarray, device, bs: int = 128) -> np.ndarray:
    """Always eval() + no_grad: extraction must never itself perturb the trunk."""
    import torch
    was_training = trunk.training
    trunk.eval()
    imgs = torch.from_numpy(X).permute(0, 3, 1, 2).float().div_(255.0)
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            outs.append(trunk(imgs[i:i + bs].to(device)).float().cpu())
    trunk.train(was_training)
    return torch.cat(outs).numpy().astype(np.float32)


def _train_perception(trunk, X, labels, device, seed, epochs=EPOCHS_P):
    """Phase P: install the knowledge. Trunk + three linear heads, summed CE."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed + 7717)
    heads = nn.ModuleDict({n: nn.Linear(1024, k) for n, k in TARGETS}).to(device)
    opt = torch.optim.Adam(list(trunk.parameters()) + list(heads.parameters()),
                           lr=LR)
    imgs = torch.from_numpy(X).permute(0, 3, 1, 2).float().div_(255.0)
    ys = {n: torch.from_numpy(labels[n]) for n, _ in TARGETS}
    n = len(X)
    g = torch.Generator().manual_seed(seed + 31)
    trunk.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            f = trunk(imgs[idx].to(device))
            loss = sum(nn.functional.cross_entropy(heads[nm](f), ys[nm][idx].to(device))
                       for nm, _ in TARGETS)
            opt.zero_grad(); loss.backward(); opt.step()
    trunk.eval()
    return float(loss.item())


def _train_action(trunk, X, y, Xte, yte, device, seed, unfreeze: bool,
                  epochs=EPOCHS_A):
    """Phase A. `unfreeze=False` is the arm the hypothesis is about; True is the
    registry's control, and it trains the trunk at the head's OWN lr — a control
    given a deliberately small trunk lr is a control engineered not to fire.

    The head is identical in both arms, seeded identically, and sees the same
    number of optimiser steps, so the only difference between the arms is
    whether the trunk is in the graph.
    """
    import torch
    import torch.nn as nn
    torch.manual_seed(seed + 9001)
    head = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, N_ACTIONS)).to(device)
    if unfreeze:
        trunk.requires_grad_(True); trunk.train()
        opt = torch.optim.Adam(list(trunk.parameters()) + list(head.parameters()), lr=LR)
    else:
        trunk.requires_grad_(False); trunk.eval()
        opt = torch.optim.Adam(head.parameters(), lr=LR)
    imgs = torch.from_numpy(X).permute(0, 3, 1, 2).float().div_(255.0)
    ys = torch.from_numpy(y)
    n = len(X)
    g = torch.Generator().manual_seed(seed + 131)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            if unfreeze:
                f = trunk(imgs[idx].to(device))
            else:
                with torch.no_grad():
                    f = trunk(imgs[idx].to(device))
            loss = nn.functional.cross_entropy(head(f), ys[idx].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    trunk.eval(); head.eval()
    # held-out semantic-task accuracy
    imte = torch.from_numpy(Xte).permute(0, 3, 1, 2).float().div_(255.0)
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), 128):
            preds.append(head(trunk(imte[i:i + 128].to(device))).argmax(1).cpu())
    acc = float((torch.cat(preds).numpy() == yte).mean())
    return round(acc, 4)


# ── one internal seed, end to end ────────────────────────────────────────
def _one_seed(seed: int, device, n_train=N_TRAIN, n_test=N_TEST,
              epochs=None) -> dict:
    """`epochs` is the SMOKE path only and is threaded as an argument rather
    than assigned to the module globals: a smoke that mutates EPOCHS_P would
    leave a shrunken training budget behind for anything else in the process,
    and the run would still look ordinary in the record."""
    import copy
    ep_p = EPOCHS_P if epochs is None else epochs
    ep_a = EPOCHS_A if epochs is None else epochs

    Xtr, ltr, tries = _build_dataset(seed, n_train)
    Xte, lte, _ = _build_dataset(seed + 500_009, n_test)   # disjoint episode rng
    eye = _get_eye(seed)
    ya_tr, ya_te = _action_labels(ltr), _action_labels(lte)

    # NULL: a random-weight trunk, never given phase P.
    rand_trunk = _make_trunk(seed + 4242, device)
    rand_trunk.eval()
    probe_random = _probe_all(_features(rand_trunk, Xtr, device), ltr,
                              _features(rand_trunk, Xte, device), lte)

    # PHASE P — install the knowledge.
    trunk = _make_trunk(seed, device)
    n_params = int(sum(p.numel() for p in trunk.parameters()))
    final_p_loss = _train_perception(trunk, Xtr, ltr, device, seed, ep_p)
    theta_p = copy.deepcopy(trunk.state_dict())
    sha_p = _param_sha(trunk)
    probe_before = _probe_all(_features(trunk, Xtr, device), ltr,
                              _features(trunk, Xte, device), lte)

    # PHASE A — three arms, each starting from its own copy of theta_P.
    frozen = _make_trunk(seed, device); frozen.load_state_dict(theta_p)
    acc_frozen = _train_action(frozen, Xtr, ya_tr, Xte, ya_te, device, seed,
                               False, ep_a)
    sha_frozen_after = _param_sha(frozen)
    probe_after_frozen = _probe_all(_features(frozen, Xtr, device), ltr,
                                    _features(frozen, Xte, device), lte)

    unfrozen = _make_trunk(seed, device); unfrozen.load_state_dict(theta_p)
    acc_unfrozen = _train_action(unfrozen, Xtr, ya_tr, Xte, ya_te, device, seed,
                                 True, ep_a)
    probe_after_unfrozen = _probe_all(_features(unfrozen, Xtr, device), ltr,
                                      _features(unfrozen, Xte, device), lte)

    rand_frozen = _make_trunk(seed + 4242, device)
    acc_randtrunk = _train_action(rand_frozen, Xtr, ya_tr, Xte, ya_te, device,
                                  seed, False, ep_a)

    def drift(after):
        return round(max(abs(after[n] - probe_before[n]) for n, _ in TARGETS), 4)

    def signed(after):
        return {n: round(after[n] - probe_before[n], 4) for n, _ in TARGETS}

    # A task target is NULL-ADMISSIBLE when the margin the knowledge gate
    # demands is arithmetically available over this run's own null. Computed
    # BEFORE `probe_before` is compared to anything, so it cannot be tuned by
    # what the claim needs; see the docstring's PILOT RECORD for the run that
    # made this necessary (random-weight `near` = 0.9427, gate = +0.15).
    admissible = [n for n in TASK_TARGETS
                  if probe_random[n] <= 1.0 - KNOWLEDGE_MARGIN]

    return {
        "seed": seed,
        "canary_ok": bool(eye.canary() == eye._canary0),
        "canary_colors": eye.canary_colors(),
        "mean_tries": round(tries, 2),
        "n_params_trunk": n_params,
        "final_perception_loss": round(final_p_loss, 4),
        "probe_random": probe_random,
        "probe_before": probe_before,
        "probe_after_frozen": probe_after_frozen,
        "probe_after_unfrozen": probe_after_unfrozen,
        # The registry's null, asked only about the knowledge the head needs AND
        # only about the targets on which the margin it demands is arithmetically
        # available. `admissible` is computed from THIS run's null, before the
        # claim is looked at, and is reported so a reader can see which conjuncts
        # carried the claim rather than having to trust that all three did.
        "null_admissible": admissible,
        "n_null_admissible": len(admissible),
        "knowledge_margin_min": round(
            min([probe_before[n] - probe_random[n] for n in admissible],
                default=-1.0), 4),
        "colour_margin_over_random": round(
            probe_before[DISTRACTOR] - probe_random[DISTRACTOR], 4),  # reported
        "probe_before_colour": probe_before[DISTRACTOR],
        "probe_drift_frozen": drift(probe_after_frozen),
        "probe_drift_unfrozen": drift(probe_after_unfrozen),
        "signed_delta_unfrozen": signed(probe_after_unfrozen),
        "colour_drop_unfrozen": round(
            probe_before[DISTRACTOR] - probe_after_unfrozen[DISTRACTOR], 4),
        "frozen_params_identical": bool(sha_p == sha_frozen_after),
        "action_acc_frozen": acc_frozen,
        "action_acc_unfrozen": acc_unfrozen,
        "action_acc_randtrunk": acc_randtrunk,
        "reach_margin": round(acc_frozen - acc_randtrunk, 4),
    }


def remote_run(seeds: list, n_train=N_TRAIN, n_test=N_TEST, epochs=None) -> dict:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "seeds": []}
    for s in seeds:
        out["seeds"].append(_one_seed(s, device, n_train, n_test, epochs))
    return out


# ── GPU submission (one per spec — module cache, T2.01 pattern) ──────────
JOB = r'''
import os as _o
_o.environ["MUJOCO_GL"] = "egl"   # the preamble sets "disabled"; this job renders
import subprocess as _sp, sys as _sys
_sp.run([_sys.executable, "-m", "pip", "install", "-q", "mujoco"], check=True)
import json
from experiments.tests.t3_10_trunk_knowledge_survives import remote_run
out = remote_run(__SEEDS__)
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "t310.json"), "w"),
          indent=1)
print("DONE", json.dumps(out["seeds"][0]), flush=True)
'''

_CACHE: dict = {}


def _submit(seeds: list) -> dict:
    body = JOB.replace("__SEEDS__", repr(list(seeds)))
    job = build_job(body)
    # Colab first: this is a GPU_SHORT spec and Kaggle's 30 h/week is the scarce
    # resource (SYSTEM.md). One submission runs every internal seed serially, so
    # the timeout is sized by seed count — LESSONS: multiply by seeds before
    # sizing any budget or timeout.
    res = submit(job, prefer="colab",
                 est_hours=round(0.06 + 0.09 * len(seeds), 2),
                 timeout_s=900 + 700 * len(seeds),
                 fetch=["t310.json"])
    if not res.ok:
        raise RuntimeError(f"T3.10 job failed on {res.backend}: {res.message}")
    out = json.loads(Path(res.artifacts["t310.json"]).read_text())
    out["backend"] = res.backend
    return out


def pilot():
    """Seed-90 RIG check, disjoint from the internal seeds. It records nothing
    and it cannot move a gate — every bar in this file is exogenous or relative
    and was frozen at the commit that introduced it."""
    out = _submit([PILOT_SEED])
    print(json.dumps(out, indent=1))
    return out


# ── the spec ─────────────────────────────────────────────────────────────
def _experiment(seed: int) -> dict:
    """Folded to the WORST internal seed HERE, not at the spec level:
    `protocol.py:_aggregate` means over spec-level seeds before `_check` runs,
    which would turn every worst case below into an average (LESSONS,
    2026-08-30)."""
    if not _CACHE:
        _CACHE.update(_submit(INNER_SEEDS))
    rows = _CACHE["seeds"]
    return {
        "gpu": _CACHE["gpu"], "backend": _CACHE["backend"],
        "n_inner_seeds": len(rows),
        "canary_ok_all": all(r["canary_ok"] for r in rows),
        "canary_colors_min": min(r["canary_colors"] for r in rows),
        "n_params_trunk": rows[0]["n_params_trunk"],
        "knowledge_margin_min": min(r["knowledge_margin_min"] for r in rows),
        "n_null_admissible_min": min(r["n_null_admissible"] for r in rows),
        "probe_before_colour_min": min(r["probe_before_colour"] for r in rows),
        "probe_drift_frozen_max": max(r["probe_drift_frozen"] for r in rows),
        "frozen_params_identical_all": all(r["frozen_params_identical"] for r in rows),
        "action_acc_frozen_min": min(r["action_acc_frozen"] for r in rows),
        "action_acc_randtrunk_max": max(r["action_acc_randtrunk"] for r in rows),
        "reach_margin_min": min(r["reach_margin"] for r in rows),
        # reported, not gated — the audit trail for a reader who is not me.
        # Reported, not gated. NAMED for what `_aggregate` will actually keep:
        # a non-numeric field survives as runs[0][k], the FIRST spec-level
        # seed's value, with no marker (LESSONS 2026-08-30). Here spec.seeds is
        # 1, so runs[0] IS the only run and this carries all three INTERNAL
        # seeds intact — the name states the condition rather than assuming it.
        "per_inner_seed_from_run0": [{k: r[k] for k in (
            "seed", "probe_random", "probe_before", "probe_after_frozen",
            "probe_after_unfrozen", "probe_drift_frozen", "probe_drift_unfrozen",
            "signed_delta_unfrozen", "colour_drop_unfrozen",
            "colour_margin_over_random", "action_acc_frozen",
            "action_acc_unfrozen", "action_acc_randtrunk",
            "final_perception_loss", "mean_tries")}
            for r in rows],
    }


def _control(seed: int) -> dict:
    """The registry's control: "Deliberately unfreezing the trunk must reproduce
    the drift." Worst case = the seed where the instrument is LEAST alive."""
    rows = _CACHE["seeds"]
    return {
        "probe_drift_unfrozen_min": min(r["probe_drift_unfrozen"] for r in rows),
        "colour_drop_unfrozen_min": min(r["colour_drop_unfrozen"] for r in rows),
        "action_acc_unfrozen_min": min(r["action_acc_unfrozen"] for r in rows),
        "probe_drift_unfrozen": [r["probe_drift_unfrozen"] for r in rows],
    }


def _check(m: dict, c: dict):
    # ── RIG first: an invalid run is not evidence about the hypothesis ──
    if not m["canary_ok_all"]:
        return Status.VOID                      # GL context degraded mid-run
    if m["canary_colors_min"] < MIN_CANARY_COLORS:
        return Status.VOID                      # uniform frame == blind sensor
    if not (PARAMS_RANGE[0] <= m["n_params_trunk"] <= PARAMS_RANGE[1]):
        return Status.VOID                      # the seat holder changed
    if m["n_null_admissible_min"] < 1:
        # No task target leaves room for the margin the gate demands: this world
        # poses nothing the seat holder could learn that its random init does not
        # already read. A finding about the trunk, and not evidence either way
        # about the hypothesis — so VOID, and PARK per the docstring's fork (ii).
        return Status.VOID
    if m["knowledge_margin_min"] < KNOWLEDGE_MARGIN:
        return Status.VOID                      # nothing was installed to survive
    if m["probe_before_colour_min"] < COLOUR_HEADROOM:
        return Status.VOID                      # no room for the control to show
    if c["probe_drift_unfrozen_min"] < CONTROL_DRIFT_MIN:
        return Status.VOID                      # the drift instrument is not alive
    if c["action_acc_unfrozen_min"] < TASK_LEARNABLE_FLOOR:
        return Status.VOID                      # nobody can learn the task
    # ── the claim ──────────────────────────────────────────────────────
    return (m["frozen_params_identical_all"]                     # receipt
            and m["probe_drift_frozen_max"] <= DRIFT_MAX         # receipt / leak
            and m["reach_margin_min"] >= REACH_MARGIN            # LOAD-BEARING
            and m["action_acc_frozen_min"] >= TASK_LEARNABLE_FLOOR)


def _selftest() -> int:
    """Known-answer test for `_check`: every gate must FIRE on a planted
    violation, not merely stay quiet on a clean row. Written because a gate that
    has only ever been shown passing is a gate of unknown state — the same
    reason `_control` exists one level down. Runs in milliseconds, no GPU, no
    render; `T0.21`-style, and it is the only thing in this file that can be
    checked without spending a dispatch.
    """
    clean_m = {
        "canary_ok_all": True, "canary_colors_min": 1343,
        "n_params_trunk": 244960, "knowledge_margin_min": 0.30,
        "n_null_admissible_min": 2,
        "probe_before_colour_min": 0.97, "probe_drift_frozen_max": 0.0,
        "frozen_params_identical_all": True, "action_acc_frozen_min": 0.62,
        "action_acc_randtrunk_max": 0.34, "reach_margin_min": 0.28,
    }
    clean_c = {"probe_drift_unfrozen_min": 0.19,
               "colour_drop_unfrozen_min": 0.14,
               "action_acc_unfrozen_min": 0.70}
    cases = [
        # (label, metric overrides, control overrides, expected _check result)
        ("clean row PASSes", {}, {}, True),
        ("GL degraded mid-run", {"canary_ok_all": False}, {}, Status.VOID),
        ("uniform frame == blind sensor", {"canary_colors_min": 3}, {}, Status.VOID),
        ("seat holder changed", {"n_params_trunk": 57_000_000}, {}, Status.VOID),
        ("nothing installed to survive", {"knowledge_margin_min": 0.02}, {}, Status.VOID),
        # The 2026-08-30 pilot, replayed as a fixture: every task target's null
        # already wins, so `knowledge_margin_min` is the sentinel -1.0 and the
        # admissibility gate — not the margin gate — must be what fires.
        ("no task target is null-admissible",
         {"n_null_admissible_min": 0, "knowledge_margin_min": -1.0}, {}, Status.VOID),
        ("distractor has no headroom", {"probe_before_colour_min": 0.30}, {}, Status.VOID),
        ("drift instrument dead", {}, {"probe_drift_unfrozen_min": 0.01}, Status.VOID),
        ("task not learnable at all", {}, {"action_acc_unfrozen_min": 0.13}, Status.VOID),
        # the two falsified_by branches, which must FAIL and not VOID
        ("gradient leak: probes drifted", {"probe_drift_frozen_max": 0.11}, {}, False),
        ("gradient leak: bytes moved", {"frozen_params_identical_all": False}, {}, False),
        ("knowledge does not reach the head",
         {"reach_margin_min": 0.01, "action_acc_frozen_min": 0.35}, {}, False),
        ("head sits at chance", {"action_acc_frozen_min": 0.13}, {}, False),
    ]
    fails = []
    for label, dm, dc, want in cases:
        got = _check({**clean_m, **dm}, {**clean_c, **dc})
        if got is not want:
            fails.append(f"{label}: _check returned {got!r}, expected {want!r}")
    # A VOID must never be reachable from a clean rig — i.e. the rig gates and
    # the claim gates must not be the same gate wearing two names.
    if _check(clean_m, {**clean_c, "probe_drift_unfrozen_min": 0.10}) is not True:
        fails.append("control exactly AT its floor must not VOID (>= is the bar)")
    for f in fails:
        print("FAIL", f)
    print(f"selftest: {len(cases) + 1 - len(fails)}/{len(cases) + 1} checks passed")
    return 1 if fails else 0


def run(ledger: Ledger | None = None):
    if not _GATES_FROZEN:
        raise RuntimeError(
            "T3.10 refuses: no gate is un-piloted — the 2026-08-30 T4 pilot "
            "found two of three gated conjuncts UNSATISFIABLE BY ARITHMETIC "
            "(random-weight probes read near 0.9427 and colour 0.9245 against "
            "a +0.15 margin). REPAIR 1 (EPOCHS_P 150, null_admissible as a "
            "rule) is in the file and UNPILOTED. Run `pilot`, then take fork "
            "(i) _GATES_FROZEN = True, or fork (ii) PARK — both pre-registered "
            "in the docstring. Do NOT dispatch the registered run first: its "
            "VOID is already known.")
    return run_spec(BY_ID["T3.10"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "selftest":
        sys.exit(_selftest())
    elif arg == "pilot":
        pilot()
    elif arg == "smoke":
        # Local CPU: the whole pipeline at toy size. Proves the rig, the label
        # balance, the freeze receipt and the probe plumbing; proves NOTHING
        # about the gates, and its numbers never enter this file.
        out = remote_run([PILOT_SEED], n_train=CELL * 2, n_test=CELL, epochs=1)
        print(json.dumps(out, indent=1))
    elif arg == "render-smoke":
        # Cheapest possible check: render a balanced mini-batch, no torch.
        X, lab, tries = _build_dataset(PILOT_SEED, CELL)
        eye = _get_eye(PILOT_SEED)
        print(json.dumps({
            "frames": list(X.shape), "mean_tries": tries,
            "shape_counts": np.bincount(lab["shape"]).tolist(),
            "colour_counts": np.bincount(lab["colour"]).tolist(),
            "near_counts": np.bincount(lab["near"]).tolist(),
            "canary_ok": bool(eye.canary() == eye._canary0),
            "canary_colors": eye.canary_colors(),
            "distinct_frame_sums": len({int(f.sum()) for f in X}),
        }))
    else:
        run()
