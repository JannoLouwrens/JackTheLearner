# ORDERS — what the workers do. Written by the Director.
Director sitting: 2026-11-16 06:41 UTC

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5). **One state+journal commit per slot,
not one per step.** Machine problems go under INBOX, one line each. Rulings
22–27 cited here are in `docs/DECISIONS_RESOLVED.md` under 2026-11-16 06:41.

**`docs/PLAN.md` binds every order (owner, 11-15).** Research → combine →
reproduce → test the last 10% → put it in him. We are in **stage 1: the
instrument.** `docs/research/PUBLISHED_PRIORS.md` §1–§12 is the reference —
**read your order's section at the start of every slot and follow its exact
versions and numbers where it is sharper than this page.**
**Where stage 1 stands (11-16).** Done: the world test card (J1), the dial
and the record's setup (J2), the harness's six learner-free checks (J4),
brain B on its two SAC rows. **Measured, not argued:** at d = 1 (today's
world) nothing that uses only his senses sustains a life — the best
hand-written forager harvests 26% of the 70% a life needs. That is a
NEGATIVE first answer to "can the task be learned from his senses at d = 1";
nobody tunes a third forager against that bar (ruling 23). The entry rung is
**d0 = 0.75** (ruling 22); no rung below 0.5 is ever used (the dial feeds a
statue there). The ready-list in PLAN ("before the Mac arrives") is stage
1's deadline.
**Stopped (rulings 10, 20):** races between home-made learners outside J3(4)
and J8; new forager cells; world-rule races; `D1.0` and `T2.01` runs; new
specs with no published prior. A question with no prior is a `RESEARCH:`
line in INBOX, never a run.
**Where things run.** Box CPU: day ceiling 57,600 s, ≤ 2 computing detached
jobs (pollers do not count, ≤ 6). Runs < 15 CPU-min may run on the box at
nice 19; everything longer goes to Kaggle CPU kernels (`prefer="kaggle-cpu"`,
cap 5, use 4, free). Kaggle GPU has **0 h** until Sat 11-21 00:00 UTC (W46
spent 33.8 of 30); Colab T4 is open through `gpu.py` (`prefer="colab"`).
**`/` has 3.7 GB free and workers refuse under 3:** nothing new is written
under `/home` or `/tmp` — environments, downloads, logs and scratch go under
`/data` only; a slot that finds `/` under 3.3 GB says so in INBOX first.
**World files** (`w0.py`, `needs.py`, `survival.py`, `drives.py`): changed
only through `jack_env.py` options whose default is today's world, proven by
`test_jack_env.py` — except J9's one hook (ruling 25).
**A smoke for a multi-life row runs the registered life count on one seed.**
**Receipts:** the slot that launches a kernel commits
`experiments/gpu_submissions.jsonl` and `cpu_budget.json` by name in its
state commit — they must not sit dirty.

---

## JACK — build the creature, his world and his curriculum (max 7 open, ranked)

*Order of work: **J8** → **J3** → **J9** → **J5** → **J6** → J10 → J7. A slot
waiting on a kernel advances the next order. **The 07:07 and 19:07
strong-worker slots take J5 first — this binds even when kernels are waiting
to be harvested (harvest ≤ 5 minutes, then J5)** — then J8. A slot that can
advance none takes "Next up" item 1.*

### J8 — Brain B lives in his world: the 20-run pilot at the entry rung (the first result that needs no Mac)
part: Learning to eat + His brain · stage: 1 (ready-list item 4) · step: 4 (pilot of the last 10%; B passed step 3 on its SAC rows — ruling 26) · prior: §1 "Take as-is" and "Ours beside theirs", §5, §8, ruling 19 as strengthened by ruling 22
prior: the record predicts B eats reliably after 100,000–200,000 decisions with large replay, a 40 s horizon, sustained exploration and payment on eating; published SAC needed 100k–1M on comparable foraging. B reproduced Pendulum (−302.0 in [−314.7, −282.1], 15 seeds) and sparse MountainCar (5/5 > 90). The plain learner that shows the task learnable does not exist yet at any rung: this pilot IS that question.
**Do.** (1) `experiments/pilot_b.py`: `brain_b` on `jack_env` with
`difficulty=0.75`, `reward="dr100-eat"`, `random_needs`, `random_layout`,
death = terminated, cap = truncated; observations normalised (running
statistics, frozen at evaluation — say so in the docstring); §1's settings
(replay holds the whole run; discount for a 40 s horizon; the exploration
that passed row 3b). "Ours beside theirs" table. (2) Box smoke, one seed,
20,000 decisions: decisions per second WITH learning, written down. (3) RULE
committed before any kernel: T = 300,000 decisions per run (if the smoke
says 4 runs do not fit one 9-hour kernel, the rule says the T that does and
why — for every arm); 20 seeds of B; floor arms 5 seeds each — held-8
random, do-nothing, B with learning rate 0; ceiling = J1's row e. Per life
log (decisions, floor meals, seconds lived, capped or cause of death). Read
through `experiments/harness` (`prereg.freeze` first). FIRST add ruling
22's sustained mark to the harness as a second outcome (life capped AND
intake ≥ basal drain; edge-case tests like #10's) — primary = that mark; the old 5-meal mark reported beside it; floor arms
≤ 1 success in total or VOID. (4) Kaggle CPU, 4 runs per kernel. (5) Pay
harness checks #2 and #4–#9 where B's logs allow, rows in `HARNESS_CHECKS.md`.
(6) After the 20 + 15 are launched, two REPORTED-ONLY exploration arms, 5
seeds each, same T (ruling 28; FIELD_WATCH N1, arXiv:2611.11868 — 30 seeds,
sparse tasks: gSDE learned SLOWER than white noise, z −0.75; pink +0.35):
plain white noise, and pink noise applied only when the action is chosen
(never inside the update). They choose nothing; they say whether row 3b's
exploration is the right default for the decisive test.
No seat moves on a pilot; it fixes W and T for the decisive test.
serves: M2→M3 · PLAN ready-list "B reproduced, then piloted in his world" · philosophy 1, 4.
**done when:** the rule, the 20 + 15 runs and the table are in `docs/research/PILOT_B.md` with one of three readings: sustained / eats but starves / does not learn at this budget.
state: (1)-(3) DONE `6f197f94` (pilot_b.py, harness sustained mark, rule); (4) builder `5d66b69b`; smoke 40.2 decisions/s wall, 25.9/CPU-s → T 300,000 stands.
state: 11-16 09:24 prereg FROZEN `f0dc9fd7`: `docs/research/prereg/pilot_b_vs_{blr0,floor}.json` (mark sustained, W 1, T 300k; manifest asserted = kernel_groups' 35 runs).
state: k0 HARVESTED 14:15 (kaggle 4.6 h, 18 dec/CPU-s): seeds 301-304, 474 lives, **0 sustained**, meals/life first 20 → last 20 ≈ 0.7 → 0.2, every death energy (descriptive; verdict waits for 20+15). k1-k3 RUNNING. LC.02 re-bought clean `fbd47305` (default_cal works) → k4 LAUNCHED 14:50 jack-cpu-1794840642, RUNNING.
state: harness #4 PASS on the box: seed 301 ×2, 20k decisions → 8/8 lives identical field for field (row in HARNESS_CHECKS.md); 38.05 decisions/s.
state: 15:23 Kaggle: k1-k4 all RUNNING (k1-k3 at 6.0 h vs k0's 4.6 h — slower, watch the 9 h limit ~18:24). next: as k1-k3 free (≤4 sessions), launch f0-f3 one per minute (`pilot_b kernel --name fN` via launch_detached); harvest kN.json; then (6) register the two exploration arms.

### J3 — Brain B from published code reproduces published numbers (§5's rows)
part: His brain · stage: 1 · step: 2→3 (combine, reproduce) · prior: §5
**Why.** No learner here has ever reproduced a published result.
**Do.** (1) A separate environment `/data/venvs/jack-std` (Python 3.12 is at
`~/.local/bin/python3.12`) with §5's pins exactly; the project venv is not
touched. (2) `experiments/brain_b.py`: SBX `SAC` with
`policy_kwargs=dict(layer_norm=True)` and §5's settings — NOT a rewrite.
Docstring = source, versions, "ours beside theirs". (3) §5's rows, rule
committed first, 5 seeds each, the metric exactly as §5 states it (**ruling
18:** these replace the old "−157, pass ≥ −200"): row 1 Pendulum SAC — the
unmodified library first, then ours — accept a 5-seed mean in −314.7 to
−282.1; row 3 sparse MountainCarContinuous, gSDE recipe in PyTorch SB3,
> 90 on ≥ 3 of 5 seeds, and any SBX or pink-noise variant passes the same
row before it is used; rows 2 and 4 (PPO reference). Table →
`docs/research/KNOWN_ANSWERS.md` (task, steps, published, ours, seeds). A
miss means the build is broken: fix it, do not go on. Rows 5–7 (DMC,
recurrent) are next week's. (4) **The owner's fair chance (11-15 19:31):
ALL seven in-house arms** (`ppo-needs`, `ppo-lp`, `dreamer-xs`, `wm-efe`,
`wm-latent`, `wm-latent+ring`, `wm-dyn`) plus the seated `cores.py` tanh-AWR,
each through one thin adapter, on rows 1 and 3. Rule committed first
(ruling 24): an arm PASSES a row if it reaches the row's published band
within 3× the published learner's steps on ≥ 3 of 5 seeds, with its own
never-trains control failing; one Kaggle CPU kernel per arm. A failing arm
is set aside with its numbers, not deleted, and may return repaired.
serves: M3 · PLAN stage 1 "each brain reproduces published numbers" · philosophy 3, 4.
**done when:** `KNOWN_ANSWERS.md` has rows 1–4 inside their ranges and one row pair per in-house arm, pass or fail.
state: (1)(2) DONE `1d12bb9a`. Row 1 PASSED (1b, 15 seeds: ours −302.0); row 2b REPRODUCES (−199.5); row 3 / 3b REPRODUCE 5/5 (`5d794fba`).
state: row 4 / 4b MISSED (15-seed 72.6 ± 25.7; seeds solve then collapse). Stack hunt: published stack (py3.9, SB3 1.7.0, gym 0.21) has no behavioural source diff; row 4c on it VOID on its per-seed control (lr-0 seed 0 = 98.0), data 82.3 ± 11.6 / 20, misses 4b's band itself (KNOWN_ANSWERS §Row 4c).
state: row 4d at 40/arm (rule's one extension) = **MISSES** (11-16 14:15): ours 72.6 ± 19.0 (9/40 < 60) vs published code 73.4 ± 28.5 (6/40); HL −3.75, 90% CI [−9.1, +0.6] (width 9.7, sized for 6.5); MW p .16. The published code on this box: 60 seeds, 76.4, 7/60 collapse — it does not meet its own 87.5 ± 4.2. KNOWN_ANSWERS §Row 4d extension; INBOX asks the Director to close or re-register row 4.
state: (4) RULE `3054fbe6` `experiments/inhouse_rows.py`: arm core + seated update through one adapter; rows 1 (60k) and 3 (150k), 5 seeds + frozen-twin ctrl; smoke ppo-needs 5.7 ms/step, wm-latent 15.9 ms/step (≈2.4 / ≈7 CPU-h per arm).
state: next: `inhouse_rows run … ppo-needs` is ≈2.4 CPU-h → Kaggle CPU kernel (not the box: > 15 CPU-min), after J8's k4/f0–f3 hold their sessions.

### J9 — What death looks like: the dying body's last observation (before brain A touches his world)
part: His world (the interface) · stage: 1 (ready-list item 3) · step: 2 (combine) · prior: §5 "next observation at episode ends", §8 self-check 8; INBOX code review 11-15 17:02
**Why.** On the decision he dies, `jack_env.step()` returns the REBORN
body's senses with `terminated=True`. Harmless to B; a world model would
learn that a full, relocated body is what death looks like.
**Do (ruling 25 permits this one edit to `w0.py`).** (1) A hook that
captures the dying body's observation before the respawn WITHOUT drawing
from the world's random stream (the placebo and noise-panel draws must not
be repeated: reuse the values already drawn for that decision, or a
side-effect-free observe). Default behaviour byte-identical. (2) `jack_env`
option `terminal_obs="dying"` (default today's), returning that observation
on the terminal step and the reborn one from `reset()`. (3) Proofs: the J1
identity test passes with the option off AND on (the world's stream is
untouched); a planted test shows the terminal observation has the dying
body's energy and place. (4) Move the two imports inside `step()` to module
top. (5) `run stale` on a scratch worktree: name in INBOX what the `w0.py`
edit stales; regate re-buys.
serves: M3 · PLAN ready-list "the interface correct at death" · philosophy 4.
**done when:** the three proofs pass and the stale list is in INBOX.
state: DONE `cfb49f25` 11-16 12:40: `W0.capture_dying_obs` + `W0.observe_pure()` (rng state saved/restored) and `JackEnv(terminal_obs="dying")`; default byte-identical; the two step() imports moved to module top.
state: proofs `experiments/tests/test_jack_env_terminal.py` PASS 3/3: J1 identity vs run_survival 12/12 OFF and ON (controls differ); ON stream == OFF on every non-terminal step + reset (6 deaths); planted: ON terminal proprio+needs == draw-free spy on respawn, e 0.0 at death vs 0.05 reborn, control (OFF terminal) fails 3/3.
state: stale list in INBOX (9 certificates; LC.02 among them blocks default_cal() until re-bought).

### J5 — Brain A (Dreamer): source pinned, speed measured on a T4 (the plan's pacing item)
part: His brain · stage: 1 · step: 2 (combine) · prior: §2, §9 row 7
**Why.** A sets the date of the whole plan: it cannot train on a free CPU
at the published rate, and nobody has published its speed on a T4.
**Do.** (1) `experiments/brain_a/README.md`: reference = official
`danijar/dreamerv3` at `e01491f`; runner = sheeprl 0.8.5 `dreamer_v3_5`
(`_1M`, `_12M`) — not sheeprl's older `dreamer_v3`, not dreamerv3-torch
(§2 says why); licence; code only, never weights. (2) On a **Colab T4 now**
(Kaggle GPU opens 11-21): the runner's smallest preset unmodified on DMC
cartpole-swingup — decisions per second, the train ratio it achieves,
wall-clock per 100k decisions; the same on one Kaggle CPU kernel to put a
number on "a CPU cannot". (3) From those: the T4-hours for §2's first three
known-answer rows (estimate on record: 25–60) and for one 1M-decision life
run, written beside the estimates. (4) A's "ours beside theirs" table (none
changed yet). Do not wire it to his world.
serves: M3, M5 · PLAN stage 1 lines 3–4 · objective 3 (the GPU-able order) · philosophy 2, 3.
**done when:** the pin, the two speed figures and the two cost lines are committed.
state: DONE `ca4dcf00` 11-16 15:40. (1) pin in README. (2) T4 uncompiled 2.64 grad steps/s = 5.28 dec/s; **compiled 25.9 = 51.8 dec/s = 0.54 h/100k** (9.8×, inside the record's 20–50); bf16 4.17 dec/s (slower); 3 runs/T4 5.86 summed (2 vCPUs limit). Kaggle CPU 1.71 dec/s.
state: (3) at compiled speed: 3 known-answer rows × 3 seeds ≈ 7 T4-h (5 seeds ≈ 12); one 1M-decision life ≈ 5.4 T4-h of learning, ≈ 13 serial with his 28 ms/decision simulator → J6 now sets A's price.
state: next for A (a new order, not J5): the §2 rows with `algo.compile.enabled=True` when Kaggle GPU opens 11-21. Scratch venv `/data/venvs/jack-a-probe` may be deleted.

### J6 — Make the simulator fast, once, and say what it costs to re-certify
part: His world (the instrument's speed) · stage: 1 · step: 2 (combine) · prior: §4 "Take as-is"; INBOX measurement 11-15 15:55
**Why.** 38.8 decisions/s means 1M decisions ≈ 7 h. §4: the loop, not
physics, is the bottleneck; predicted 500–2,000 per kernel.
**Do, each a `jack_env` option, default OFF, measured alone and together
(decisions per CPU-second, box and one Kaggle CPU kernel, table in the
docstring):** (1) a 4-process vector (changes nothing in the world — do it
first); (2) all sub-steps of a decision in one native call, per-sub-step
Python work moved to once per decision where the result is identical; (3)
rays by `mj_multiRay`; (4) `integrator="implicitfast"`. For each, the same
3-seed identity test as J1: identical, or the measured difference (the
integrator IS a world change: integrity .879 → .972 in one life). Then list
which standing world certificates each non-identical option would stale
(`run stale` on a scratch worktree). The Director rules on the switch with
that list; do not flip any default.
serves: M2→M3 · PLAN stage 1 line 1, "honest limits" 1 · philosophy 5, 6.
**done when:** the speed table and the re-certification list are committed.
state: open — ordered 11-15 16:57. 11-16: item (3) now also serves his first eye (§12: a ray retina from the same one call) — measure 16 rays AND a 16×16 and 32×32 [depth, material] grid.
state: (1) DONE `7f3af75d` 11-16 13:50: `jack_env.make_vec(n, seeds=…)` (AsyncVectorEnv, SAME_STEP autoreset, terminal obs in info["final_obs"]); proof `experiments/tests/test_jack_env_vec.py` PASS: 3 seeds × 300 steps, 144 ends, streams == single env exactly (reborn + dying 6/6), controls differ 2/2. World untouched → stales nothing (`run stale`: 0 jack_env lines).
state: box speed at load 3.8: single 35.3 dec/wall-s, vec2 56.2 (1.59×), per CPU-s unchanged (35.9 vs 33.3). Hand cal j0=alpha=1.0 (default_cal refuses while LC.02 stale; identity is cal-independent).
state: profile `experiments/sim_profile.py` (11-16 14:15, box): 28 ms/decision = mj_step×40 under the shipped **RK4** 15–16 ms (54–58%) + Python 12.6 ms (ContactAudio ~27%, drives.substep ~12%). Zero-Python ceiling only 61–66 /CPU-s (1.9×); implicitfast physics 3.9–4.2 ms (RK4 costs 3.9×) → ~60 with today's Python, ~240 ceiling with both. §4's 500–2,000 needs (2) AND (4).
state: (2a) DONE `403def65` 11-16 15:40: `lazy_contact_audio` (experiments/fast_sim.py; ContactAudio.py/w0.py untouched) — force+pos only for a pair that fires. Proof `tests/test_fast_sim.py` 3/3 PASS (144 ends, 29,805 audio events equal field for field), control differs; box 40.8 → 51.8 dec/CPU-s (1.27×). Stales nothing.
state: next: (4) implicitfast measured alone + identity diff (a world change: report the difference, list stales on a scratch worktree), then (3) mj_multiRay; drives.substep is already vectorised (low yield). 4-process row rides the first free Kaggle CPU kernel after J8.

### J10 — The walking layer starts now, by reproducing a published walk (owner, 11-15 19:18)
part: Body and movement · stage: 1 side job (stage 3's part, independent of the brain) · step: 3 (reproduce) · prior: §4 "Reproduce first", "Take as-is"
**Why.** The owner asked whether walking is in the right place in the plan.
The record: walking is a known recipe and needs no brain choice; `T2.01` is
re-earned this way, never by more one-world attempts.
**Do, one per slot, Kaggle CPU only, rule first, table row in
`KNOWN_ANSWERS.md`.** (1) §4's Gymnasium Humanoid floor numbers (random and
standing-still returns; seconds). (2) §4's 2018 CPU recipe — helping forces
that are withdrawn — with the UNMODIFIED published learner and settings on
the published task, one kernel, accept inside §4's band, never-trains
control must fail. (3) Not before (2) reproduces: the same recipe on our
humanoid. The MJX tutorial humanoid waits for the Kaggle T4 on 11-21.
Design note for stage 3 (owner's correction 19:21): inside him the layer
KEEPS LEARNING (§4 arm c); the frozen layer (arm b) is a scored reference only.
serves: M4 · PLAN stage 3 "a separate, fast walking layer is built and checked in parallel from now" · philosophy 1, 2.
**done when:** rows (1) and (2) are in `KNOWN_ANSWERS.md`, pass or fail.
state: open — ordered 11-16 06:41.

### J7 — Close what is in flight (cut down 11-16: two days of state lines bought one harvest)
part: His world, needs and lessons · stage: 1 · step: repairs (PLAN "continues")
**Do, one per slot, each ≤ 20 minutes of a slot.** (a) Harvest whichever
forager kernels have exited (logs `/data/tmp/forager3_*.log`); **launch no
more**; append to `docs/research/FORAGER_RACE.md` under "PILOT — learners
not known-answer-checked, below every published foraging budget"; no seat
moves; when all have exited remove `/data/tmp/wt_forager3`. (c) `NE.01` on
the shipped world under the fatigue edge probe (`eaa9302`), one run. (e)
`run W1.04` once; a FAIL at ~53 s is the honest reading and is followed by
nothing. (`PG.6` and `PS.02` v3 moved to "Next up" 8 — they need a detached
launch and a design step, not a leftover slot.)
serves: M2 · philosophy 4 (a failure is fixed, not filed).
**done when:** nothing is in flight and `NE.01`, `W1.04` each have a new ledger row.
state: (a) ppo-lp s153/s154 harvested 11-15 15:38 (floor meals 13/17 over 60 lives, apples 0). (c) clause applied `eaa9302`, not run.

**Paused or closed by the record (ruling 20):** `D1.0` (no more GPU to the
57M arena; cached units stay in `/data/d10_units/873e861e…`), `T2.01` (re-
earned through J10), `LF.06`, `LC.09`, `CR.01`, `LF.04` — "to earn again" on
the brain stage 2 chooses.
**Next up, in order (ruling 27 ranks the owner's asks):** 1. **MLX port,
stage 0 only** (owner 11-15 19:18; `PUBLISHED_EVIDENCE_2026-11-15_PORT.md`
steps 1–3): golden files from the official code and gate G0, on Kaggle CPU
— becomes an order the day J5's speed figures land; a failed G0 stops the
port. 2. **Mac first-day script** (§10's twelve measurements, one command,
no Linux-only calls) — due 12-15. 3. **His first eye, step 2:** the ray
retina of §12 as a `jack_env` option after J6(3). 4. A's known-answer rows
on the T4 from 11-21 (after J5), with the sheeprl-vs-official line diff.
5. In-house arms that passed J3(4) join the pilot: same seeds, length and
rule as J8, each twice (own settings; §1's settings). 6. Idle-slot
reproductions (§3, §6, §7; one per slot): the 8×8 shaping grid, Dyna maze,
RL² bandits, Speech Commands keyword spotting (corpus already at
`/data/jack_corpora/speech_commands_v2`). 7. The clone (E2b) and the cheat
learner (E2c) at d0 — the learnability proof J1 did not give. 8. `PG.6`
detached; `PS.02` v3 (ruling 12). 9. The dial's success gate. 10. Diary
retrieval keyed on learned state (§6), stage 3.

---

## SYSTEM — work on the machine (max 3 open)

*Order of work: **S7** → **S9** → **S6** (step 3).*

### S7 — Another session's uncommitted files must not kill a run
part: (machine) · stage: — · step: —
**Why.** Twice on 11-15 dashboard files nobody imports tripped the tree-wide
dirty guard: `D1.0` lost a 5.9 GPU-hour slice, the forager cells launched
from a hand-made worktree. Stage 1 launches many small kernels beside a live
owner session.
**Do, one commit each, each with a planted test.** (1) The submit-time guard
in `gpu.py` refuses only dirt inside the spec's import closure plus
`experiments/` core (`offending_dirt` computes the list); dirt under
`dashboard/`, `docs/`, `scripts/dashboard_status.py` is named in the row,
not fatal. A planted dirty imported file must still refuse. (2)
`scripts/launch_detached.sh` takes `JACK_REPO`. (3) `JACK_REUSE_KERNEL`
applies to the first `submit()` only and is refused when a second body asks
for the same slug. (4) `scripts/test_lane_guard.sh` passes at clean HEAD
while two detached jobs are alive.
serves: philosophy 4, 5 · objective 3.
**done when:** the four planted tests pass.
state: 11-16 (1) DONE `36e20081`: guard refuses only dirt the VM could load (under `experiments/`, or named by experiment code); dashboard/status-script dirt goes into the attempt row as `foreign_dirt`. Planted test `experiments/tests/test_gpu_dirt_guard.py` 4/4.
state: 11-16 (2) DONE `9a322df6`: `JACK_REPO=<worktree> launch_detached.sh` runs the child in the worktree; a non-worktree is refused; CPU day stays main's file. Planted `scripts/test_launch_detached_repo.sh` 3/3 (2/3 FAIL on the old launcher).
state: caveat for (2): `run.py` inside a worktree still writes that worktree's `ledger.json` — the receipt symlink J1 used is still by hand.
state: next: (3) `JACK_REUSE_KERNEL` first submit only; (4) `test_lane_guard.sh` with two live jobs.

### S6 — A video the owner can watch (owner request 11-12) — split so one slot finishes one step
part: (machine — he can be known) · stage: — · step: —
**Step 1 (one slot, done when a file exists):** `scripts/life_video.py`
renders OFFLINE with `experiments/render.py` (Xvfb + llvmpipe) one random-
action life of ≤ 60 s from `jack_env` (seed 153) — third-person view, 480p,
~12 frames a second, H.264 mp4 ≤ 5 MB — to `/data/jack-video/latest.mp4`.
Run by hand at nice 19; no cron, no service; never inside a scored run. A
rough clip today beats a finished one next week.
**Step 2:** hunger / thirst / temperature overlaid; `/data/jack-video/latest.json`
beside it: `title`; `summary` (2–4 sentences from a TEMPLATE over the life's
own record — how long he lived, what he ate and drank, what killed him; no
model); `facts` (≤ 8 `{label, value}`); `events` (≤ 60 `{t, at, label}`: `t`
seconds into the video, `at` the time in his life); `source` (life, seed,
commit). The dashboard's Watch tab reads both.
**Step 3:** the most recent RECORDED life, no picking (record poses if none are).
serves: M8 · philosophy 1 (he can be known).
**done when:** step 1: the mp4 exists and plays. Order done: steps 2–3.
state: steps 1–2 DONE (`070cea18`, `10b72b29`): seed 153 random life, 4:54, died of hunger, 1 meal
→ 41 s mp4 (1.66 MB, ~105 s wall, nice 19) with fed/unhurt/wet bars, age and a moment banner;
`latest.json` beside it (title, template summary, 8 facts, 6 events). World has no thirst or
temperature yet — the overlay shows the three needs it has and the summary says so.
Next: step 3 (most recent RECORDED life; poses already replayed per decision).

### S9 — Every settled FAIL gets a named fate (the queue that owned them is closed)
part: (machine — nothing faked) · stage: — · step: —
**Why.** `coverage` exits 2: closing the review queue removed
the "owner" of 33 settled FAILs. They are owned by THIS order until each has
a fate; a FAIL nobody owns is a failure filed, not fixed (philosophy 4).
**Do, ≤ 8 ids per slot, each one of three, never a baseline raise, never a
re-run for a better number.** (i) `repaired_by` in the registry when an open
order or roadmap milestone will re-earn it (name it). (ii) A `FAIL-DISPOSED:
<authority> <date> — reason` note (format as `registry.py:847`) when a
ruling already answered the question — e.g. ruling 10/20 for "to earn again
on the chosen brain", ruling 20 for one-world walking. (iii) A one-line ask
in INBOX when it is a live defect that needs a JACK order. The ids: DP.05,
HR.1, HR.5, LG.10, LG.12, LG.14, LT.02, ME.11, ME.11.B, ME.11.C, ME.11.D,
PS.05, PS.06, PS.08, T0.07, T0.17, T0.23, T0.27, T0.32, T0.35, T1.08, T1.11,
T2.05, T2.07, T2.10, T2.11, T2.15, T3.09, T4.02, W.2, W1.00, WM.01, XL.01.
In the first slot also lower the floors `coverage` says fell (UNREACHABLE
115 → 101, PASS-ON-DEAD-DEPENDENCY 6 → 4; FAIL-UNOWNED 1 → 0 only when the
id list above is empty) with `run ratchets record` — `coverage` stays 2 until then.
serves: philosophy 4 · the auditor's exit codes.
**done when:** `coverage` exits 0 with this order's id list deleted.
state: open — ordered 11-16 06:47.

**Next up:** (0) **S8** — a seat held BY RECORD in `champions --check` (ruling 11; text in git at `b953e9db`; no build is a candidate until S8's known-answer rows exist, so it waits behind S9). (1) `coverage` prints where each passing CLAIM was earned (in
him / bench / parent rig). (2) `gpu.py` reattach does its own pin-swap
proof; watcher timeout from RUNNING. (3) `BD.01`, `LC.08` budget re-class
(ruling 14). (4) **S4** — six PASSes gated on a seed average that admits a
failing seed. (5) `T0.13`'s dead clauses. (6) `LC.02`'s salt replay excludes
wall-clock metrics. (7) D52 scout retry. (8) `XL.00` re-buy adds a planted positive its negative claim must catch (queue row, 11-16). `T0.33`→`T0.34`, `T0.12`→`T0.24`
re-buys are regate's on a clean tree.

---

## INBOX — one line each, workers add, the Director empties
- 11-16 14:15 JACK: regate SKIPS LC.02 every 2 h ("measured 648 s against the label's 600 s — not cheap until re-labelled; stays with the loop"), so the J9-stale LC.02 that gates `default_cal()` (and J8's k4/f0–f3) had no owner; I re-bought it on the box detached. The label (or regate's 8% tolerance) wants a system-lane look.
- 11-16 14:15 JACK: J3 row 4 — the published SB3 code itself, 60 seeds on this box, gives 76.4 with 7/60 collapses against the store's 87.5 ± 4.2 (n 10); ±5 equivalence is unreachable at 40/arm (CI width 9.7). Rule: close row 4 as "reference does not reproduce its own number", or re-register at n ≈ 100/arm (~3 Kaggle-CPU-h)?
- 11-16 12:42 JACK: J9's `w0.py` edit (`cfb49f25`) stales 9 certificates — PASS: LC.02, BA.01, LF.02, LF.03, PS.03, W0.DIAG, W1.02, XL.00; FAIL: PS.02 (scratch-worktree `run stale` diff, 48 → 57). **LC.02 stale makes `default_cal()` refuse ("calibration unborrowable"), and `pilot_b` builds `JackEnv` without j0/alpha — so J8's k4/f0–f3 must NOT be launched until regate has re-bought LC.02 (~5 CPU-min; `run stale` clear of it), or they crash at birth on Kaggle.** I did not re-buy LC.02 myself: it is a throughput floor and the box is carrying row 4d's 2 workers.
- 11-16 12:10 JACK: J3 row 4d (60 cells × ~60 CPU-s ≈ 60 CPU-min total, each cell < 2 min) runs on the box although the header says runs > 15 CPU-min go to Kaggle CPU — all 4 Kaggle sessions are held by J8 k0–k3 and J8 queues 5 more; rule the unit (whole run vs cell) if this was wrong.

- 11-16 07:13 OWNER'S SESSION — disk: I removed `~/.cache/puccinialin` (1.5 GB: a Rust build toolchain that a package install pulled in on 11-15 19:24; re-downloadable, nothing was using it). Root went from 3.7 to about 5.2 GB free. `~/.cache/uv` (0.7 GB) and `~/.cache/pip` are left for the owner's answer. To stop this recurring: new environments should set `UV_CACHE_DIR`, `PIP_CACHE_DIR` and `XDG_CACHE_HOME` under `/data/caches/` (the launcher already sets `XDG_CACHE_HOME` for lanes; detached and Kaggle-prep installs on the box do not inherit it), and prefer wheels (`--only-binary=:all:`) so no source build pulls a compiler toolchain onto the small disk.
- 11-16 06:44 DIRECTOR → RESEARCHER: `RESEARCH:` enter FIELD_WATCH N2 (OdorSpaces, arXiv:2611.05738: a multi-point nose, start-distance dial, 30M steps) as the Smell row of `PUBLISHED_PRIORS.md`; our three smell kills become pilots below the published budget. No run — stage 3. Also trim `FIELD_WATCH.md` under its 12 KB cap (12,963 B) at the next sweep.

## DONE (kept one day)

- **J1 — the world test card** (`c65848cd`): seven rows, three answers in `WORLD_CARD.md`. Privileged forager 200/200 lives at the cap; sensor-only foragers 6/20 and 13/20 against a bar of 18 — NEGATIVE at d = 1, recorded as such.
- **J2 — the dial and the record's setup** (`9178993c`): reward, truncation, dial, random needs and layout, each proven; entry-rung table at d = 1/.75/.5/.25/0. Ruling 22 names d0 = 0.75.
- **J4 — the harness's learner-free checks** (`b79cca33`): #1, #3, #10–#13 PASS; #2, #4–#9 owed with B (J8 item 5). Ruling 22 adds the sustained mark beside θ = 5.
- **S6 steps 1–2** (`070cea18`, `10b72b29`): a 41 s video of one life with his needs drawn on it, and `latest.json`.
- **Researcher: his eye** (`9f9357bf`): `PUBLISHED_PRIORS.md` §12 → "Next up" 3, J6(3).
