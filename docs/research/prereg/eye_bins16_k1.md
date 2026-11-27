# eye_bins16_k1 — J24's nine cells again, on the forge's reset learner (K1)

**STATUS: DRAFT 2026-11-27 05:40 UTC — NOT FROZEN, NOTHING RUN; reset knob + arms built 11-27 (§ Owed).**
J24's branch on NOT SHOWN says "nothing more is launched; the sitting rules the
next ask"; its last state names two: the per-ray one-hot arm, and/or "the same
nine cells on round 1-L's winning learner, pre-registered fresh". This is the
second, written so a sitting can rule with the design in front of it. A sitting
may change any term; after a freeze, none moves.

## Question
J24 (`eye_bins16.md`, pin `03e6285d`, read `/data/jack-logs/j24/read.json`) read
NOT SHOWN AT 500k: late meals sighted 0.229 vs blind 0.887 vs shuffled 2.338,
above both 0/3. Beside it (deciding nothing): every sighted cell out-ate both
twins EARLY (s941 8.78 meals a life in a 50k window) and then unlearned to ~0;
drain per decision identical in all cells, action std flat — it stopped finding
food (`j24_collapse_diag.py`). The same window table over forge rung M shows
plain SAC collapsing 4/5 there too, K1 (resets) 0/5, K2 (TQC) 0/5
(`j24_collapse_vs_forge.py`). So the named suspect is the learner, not the eye.
**This row asks: with the learner change that did not collapse in rung M, does
brain B eat from the kind eye at 500k?**

## Prior — why it could lose either way
- For: rung M late-half sustained rate K1 0.84 vs baseline 0.059 (lock
  `2eb2c0f4`); the record's best-supported fix for plasticity loss (Nikishin 2022
  2205.07802, DMC IQM 501 → 656; Nauman 2024 2403.00514). J24's early windows show
  the sighted learner DID find food before it collapsed.
- Against: rung M is the old paint and the old eye (E-food), not bins16; a reset
  every 50k costs a dip after each reset, and the late window (≥ 250k) holds five
  resets. The record's foragers need 1–3e6 steps (§28); 500k is below that.
- A NOT SHOWN here would move the suspect off the learner and back onto size or
  the eye; an EATS says J24's miss was the collapse.

## Learner — the only change from J24
J24's learner exactly (`eye_bins16.md` § Learner and world: sbx 0.28.0 SAC,
MlpPolicy LN 2×256, lr 3e-4, batch 256, tau 0.005, γ 0.995, warmup 5,000,
train_freq 32 / gradient_steps 32, gSDE, `ent_coef="auto"`, buffer = T) **plus
K1's change from `forge/rounds/r1L/spec.json`: full network resets every 50,000
decisions, replay and temperature kept** — reset points 50k…450k (9) in a 500k
cell. All three cells carry it (a twin must share the learner, LESSONS_CORE 8).
Open for the sitting: if round 1-L's keep rule names K2, swap the change for
K2's settings; the default here is K1 because the keep rule's tie goes to K1 and
look 1 is not read yet. **Waiting for look 1 costs ≥ 6 h; drafting on K1 costs
nothing if K2 wins (one line changes before the freeze).**

## Arms, world, seeds, size, score — J24's
- Cells `K1-bins16`, `K1-bins16-blind`, `K1-bins16-shuffled` = `J24_ARMS` with
  the reset change (a NEW dict, J24's byte-identical).
- World `PILOT_RULE` + `paint="kind"`, `eye="bins16"`; seeds **941, 942, 943**
  (J24's — so the reset change is the only difference seed for seed);
  **500,000 decisions**; score = late meals a life (lives starting ≥ 250,000,
  ruling 65), `j17_late_meals`; `j17_onset`, the 50k window table and per-reset
  dips beside, deciding nothing.
- Beside, deciding nothing: J24's own nine numbers (no resets), re-read, not re-run.

## Rule (proposed — J24's bar, unchanged)
- **EATS** = sighted above BOTH twins on 3/3 seeds AND sighted mean ≥ 2 × the
  higher twin mean.
- **Extension, stated:** above both on ≥ 2/3 but not EATS → the nine cells
  carried to 1.5M (resets continue every 50k), read at lives starting ≥ 750,000.
- Else **NOT SHOWN AT 500k ON K1** — a row, not a door.
- A cell not home whole is re-run from its last whole carry; never read short.

## Owed before a freeze (build, not judge)
- ~~`pilot_b2.run_pilot_cell` takes no `param_resets`~~ BUILT 11-27 ~05:50:
  `pilot_b.make_b/run_b(param_resets=)` hands sbx's own `param_resets` (forge
  K1's path; None = unchanged, no key), `pilot_b2.J24K1_ARMS` (K1-bins16,
  -blind, -shuffled = J24's dicts + resets 50k…450k; J24_ARMS byte-identical).
  Planted `tests/test_j24k1_resets.py` 3/3 under jack-std (reset fires at the
  named count, a no-reset twin does not). Declared stale-cost 0 certificates.
- ~~chain/launcher wiring~~ BUILT 11-27 ~06:50: `j24_chain` runs the K1 arms
  (resets handed to `make_b`, `param_resets_done` in carry + record, a carry
  whose count disagrees with the model is refused); `j24_launch --variant k1`
  = this file's freeze guard, K1 arms, `/data/jack-logs/j24k1/chains`, label
  `j24k1`; default `j24` unchanged (`tests/test_j24_launch.py` new test PASS).
- ~~The carry must keep the reset schedule's position~~ PROVEN 11-27: K1-bins16-
  shuffled with resets moved to 160/300/420 (one per segment), cut + carried in
  3 processes == unbroken (every life row, trace row, actor weight; 3 resets
  in both); planted: K's actor ≠ K1's, a carry rewritten to 0 resets REFUSED
  (`tests/test_j24_chain.py`; J24's three K identities re-PASS on the edit).
- Timing (ruling 99), box-measured, not a kernel (J24's timing kernel is 20k,
  under the first reset): one reset costs ≈ 9 s once (1000-decision chunks
  21.2–22.2 s without, 30.3–30.9 s with a reset; Pendulum, one core) → 9
  resets ≈ 80 s on a ≈ 5.2 h cell, < 0.5 %. 500k stays 1 segment.
- Read: `python -m experiments.j24_read --variant k1 --pin <this file's freeze
  sha>` — J24's `verdict()` unchanged, arms `K1-bins16*`, raw
  `/data/jack-logs/j24k1/chains/j24k1_s<seed>.json`; refuses unless this file is
  FROZEN at the pin and every raw file carries it (`tests/test_j24_read_k1.py`).

## Compute
3 Kaggle CPU kernels × ~1 segment ≈ 16–21 kernel-h, no GPU; queued behind J18's
tranche 1 (runs to ~11:00–20:00 11-27) on the two free sessions.

## Settings that differ from the published foragers
One environment, not many; 500k not 1–3e6; 3 seeds; our body and world; resets
every 50k decisions (Nikishin resets every 2e5 steps on DMC, PUBLISHED_PRIORS §19 l.741 — ours is
the forge's interval, carried unchanged, not tuned here).
