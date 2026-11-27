# eye_geom_500k — the old paint's sighted cell at 500k, beside J24's twins (ORDERS Next-up 3)

**STATUS: DRAFT 2026-11-27 00:30 UTC — NOT FROZEN, NOTHING RUN.** Ruling 111
stops every learner seed on the per-geom paint until a sitting orders one; this
text exists so a sitting can decide with the design in front of it. A sitting
may change any term below before freezing; after a freeze, no term moves.

## Question
WORLD_CARD § J17 hunt, read-first sentence (ruling 121): "the toy shows the old
code is slow to read, not impossible — whether a brain could read it at 500k+ is
untested." This row tests exactly that: does brain B, given the OLD eye on the
OLD paint (`eye="row128"`, `paint` default = per-geom, each floor food its own
texture code re-dealt per world), eat from its eye at 500k decisions — J24's
size, J24's seeds, J24's learner?

## Prior — why this row could lose either way
- Against: J26 held-out R² of food bearing from today's row is **−0.135** (POST-HOC on
  looks with food in sight, −0.169): a plain regressor reads nothing from it in a
  new world. Every J17 row on this paint (h0–h4, 100k, one seed) ate no better
  than its blind twin. 0 of 12 published foragers use a per-object code (§28).
- For: the J24 toy (ruling 121, C MISSED) — the old scalar code reached 72–85%
  recall by 200k in the toy, ~800× slower than the kind look, but not zero. 500k is
  2.5× the toy's 200k.
- So a NOT SHOWN here would support "the per-object code is not a learnable eye at
  this size"; an EATS would say the toy's slow read carries into his world, and
  J17's misses were too short, not impossible.

## Arms
| cell | fed | role |
|---|---|---|
| `K-row128` | `J17_ARMS["K-row128"]` exactly (`pilot_b2.py`): row128 eye, per-geom paint, 256 vision numbers + the non-eye numbers | candidate |
| `K-row128-blind` | `J17_ARMS["K-row128-blind"]`: the same width, every vision number 0 | own twin, MUST NOT beat sighted |
| J24's `K-bins16-blind`, `K-bins16-shuffled` (seeds 941–943, already running) | their own raw files, re-used, not re-run | the order's twins ("beside J24's twins"), printed and read |

Disclosed difference: J24's twins have a 122-number observation, this cell 256+;
a blind twin carries no eye information either way, so J24's blind twin is the
same zero-information arm at a different input width. That width difference is
why the own blind twin is proposed too (one more cell a seed). The order named
only J24's twins; **a sitting chooses: own twin added (6 cells) or J24's twins only
(3 cells).** Default proposed here: add it — a twin of a different width is a
proxy (LESSONS_CORE 8).

## Learner and world
J24's exactly (`prereg/eye_bins16.md` § Learner and world), except `eye="row128"`
and the paint left at its default (per-geom). sbx 0.28.0 SAC, MlpPolicy LN 2×256,
lr 3e-4, batch 256, tau 0.005, γ 0.995, warmup 5,000, train_freq 32 /
gradient_steps 32, gSDE, `ent_coef="auto"` (ruling 64), buffer = T. World =
`PILOT_RULE` (mujoco 3.17.0; eye_frame body, motor_frame body, food_floor_only,
lazy_contact_audio, native_substep; reward `dr100-eat`, life cap 9,000).

## Seeds, size, score — J24's
- Seeds **941, 942, 943**; **500,000 decisions** a cell.
- Score: late meals a life (lives starting ≥ 250,000, ruling 65), `j17_late_meals`;
  `j17_onset` and critic rulers beside, deciding nothing (ruling 102).

## Rule (proposed — J24's bar, unchanged)
- **EATS** = sighted above EVERY twin read (own blind + J24's blind + J24's shuffled,
  or J24's two only if the sitting drops the own twin) on 3/3 seeds AND sighted mean
  ≥ 2 × the highest twin mean.
- **No extension.** (J24's 1.5M carry is J24's; this row exists to test 500k.)
- Else **NOT SHOWN AT 500k ON THE PER-GEOM PAINT** — a row, not a door.
- A cell not home whole is re-run from its last whole carry; never read short.

## Compute (estimate — a timing cell is owed before launch, ruling 99)
row128's observation is ~2× bins16's width; J24 timed 26.48 decisions/s (slowest
cell). If row128 runs ≥ 20/s, a cell is ≤ 7 h → one kernel a seed holding two
cells ≈ 1 segment. 3 kernels ≈ 21 kernel-h of Kaggle CPU — queued behind J24 and
J18 in ruling 85's rank. No GPU.

## Settings that differ from the published foragers
Per-object code (no published forager uses one — the point of the row); one
environment; 500k not 1–3e6; 3 seeds; our body and world.
