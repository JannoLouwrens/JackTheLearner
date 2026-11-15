# ORDERS — what the workers do. Written by the Director.
Director sitting: 2026-11-15 16:57 UTC

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5). **One state+journal commit per slot,
not one per step.** Machine problems go under INBOX, one line each. Rulings
17–21 cited here are in `docs/DECISIONS_RESOLVED.md` under 2026-11-15 16:57.

**`docs/PLAN.md` binds every order (owner, 11-15).** Research → combine →
reproduce → test the last 10% → put it in him. We are in **stage 1: the
instrument.** Review 2 is filed: `docs/research/PUBLISHED_PRIORS.md` §1–§9 is
the reference — **read your order's section at the start of every slot and
follow its exact versions and numbers where it is sharper than this page.**
**What review 2 changed (ruling 17): the world and the harness come before any
brain.** His world's food cannot sustain a life (a floor meal is 8% of the
bar; published teachable worlds give 44–100%), so no brain is judged in it
until J1 and J2 are done. Brain work that needs no world (J3, J5) runs beside.
**Stopped (rulings 10, 20):** races between home-made learners; new forager
cells; world-rule races; `D1.0` and `T2.01` runs; new specs with no published
prior. A question with no prior is a `RESEARCH:` line in INBOX, never a run.
**Where things run.** Box CPU: day ceiling 57,600 s, ≤ 2 computing detached
jobs (pollers do not count, ≤ 6). Runs < 15 CPU-min may run on the box at
nice 19; everything longer goes to Kaggle CPU kernels (`prefer="kaggle-cpu"`,
cap 5, use 4, free). Kaggle GPU has **0.88 h** until Sat 11-21 00:00 UTC;
Colab is open (`colab --auth adc sessions`). `/` has under 5 GB free: new
environments and downloads go under `/data` only.
**World files** (`w0.py`, `needs.py`, `survival.py`, `drives.py`): stage-1
orders change the world through `jack_env.py` options whose default is
today's world, proven by `test_jack_env.py`. If a world file must gain a
parameter: default identical, one commit, and name in INBOX the specs it stales.
**Launching past another session's dirt:** from a clean worktree, as at `e08c680` (S7 ends this).

---

## JACK — build the creature, his world and his curriculum (max 7 open, ranked)

*Order of work: **J1** → **J2** → **J3** → **J4** → **J5** → J6 → J7. J3, J4
and J5 need nothing from J1 or J2: a slot waiting on a kernel advances the
next order. The 07:07 and 19:07 strong-worker slots take J5, then J3. A slot
that can advance none of J1–J7 takes "Next up" item 1.*

### J1 — The world test card: can anything live in today's world? (E0 and E2a inside)
part: His world, needs and lessons · stage: 1 · step: 2 (combine: measure the outlier before redesigning it) · prior: §3, §8 "Gates", §1 "Learnable?"
**Why.** §3: floor food supplies 0.0024/s against a basal drain of 0.0017/s —
by arithmetic on the humanoid's constants, never confirmed on the rover.
**Do.** Rule committed before any run. All through `jack_env` (done,
`9ee9965`), 20 seeds × 10 lives per row, one table in
`docs/research/WORLD_CARD.md`: lifetime, floor meals, apples, drinks, cause
of death, summed reward per life. Rows: (a) do-nothing; (b) random, white
noise per decision (today's); (c) pink noise per life; (d) action held 8 and
32 decisions (record predicts correlated noise finds 2.5–4× more food);
(e) the privileged `NE.01` forager; (f) a HAND-WRITTEN forager fed only the
observation (no positions, no world state) — bar unmoved from `LF_04_FAIL.md`:
≥ 5 floor meals per life and above random on ≥ 18 of 20 seeds; (g) control
that must fail: (f) with its rays shuffled. Then three written answers: (1)
the rover's own intake-against-drain arithmetic from `drives.py`, beside
row (e)'s measured lifetime — can floor food alone sustain him? (2)
self-termination: is summed reward higher for a life that dies early than
for one that lives hungry? (§3: published agents learned to end such lives.)
(3) from (f): can 16 rays tell food from a wall, and if not which sense is
missing. The foragers are rulers, never parts of him. No learner, no ledger row.
**Where.** A full row is up to ~600k decisions (~4 CPU-h at 38.8/s): rows go to
Kaggle CPU kernels, launched the way `forager_race3` launched its cells; only
a 2-seed smoke of each row runs on the box. Write (f) and its rule first.
serves: M2 · PLAN stage 1 "a first rung he can live on", "proof the task can be learned" · philosophy 4, 5.
**done when:** `WORLD_CARD.md` has the seven rows and the three answers.
state: 11-15 16:45–18:05 `jack_env` proven `9ee9965`; rule + rows a,b,c,d8,d32,f,g in `world_card.py`; kernel path `c0913a62`. Smoke f 9,3/23,17 floor meals/life vs g 6,3/3,2; every death-ended life sums to −1.0 (answer 2 is arithmetic).
state: 11-15 17:42 rows f,g,b,a RUNNING on Kaggle CPU (logs `/data/jack-logs/world_card_<row>.log`; still RUNNING 18:10). 5th session on the shared account is not ours → c refused 17:41, e refused 18:18 ("Maximum batch CPU session count of 5").
state: 11-15 18:20 row e built `b84c349b` (NE.01's serve-when-due ported: floor food placed against his torso when due & e<0.95; idle duty 0.10 ±0.4). Probe 153/201: 9 serves → 9 meals/300 s, e 1.00. Answer (1) arithmetic written on rover's measured power: supply 2.39e-3/s = 1.44× idle drain, 0.59× random flailing; driving is free (xfrc).
state: 11-15 18:37 row b HARVESTED (200 lives, 20 seeds): life 301.7±16.4 s, floor meals 0.52±0.76/life (per-seed 0.1–0.9), 0 apples, all energy deaths, reward −1.00; table row + raw `docs/research/world_card_data/b.jsonl`. b's freed session → row e LAUNCHED 18:37 (pushed jack-cpu-1794767860).
state: next: (1) when a session frees (a,f,g running; check `world_card_<row>.log` tails), launch c, d8, d32 (`scripts/launch_detached.sh /data/jack-logs/world_card_<row>.log $PY -m experiments.world_card kernel <row>`; move a refused log aside first); (2) harvest each `/data/jack-logs/world_card/<row>.jsonl` → copy into `docs/research/world_card_data/`, fill table, score(f,b), score(g,b), §8 gate; do NOT retune f; (3) answers 1–3 final.

### J2 — A first rung he can live on: the difficulty dial and the record's setup, as wrapper options
part: His world, needs and lessons + Learning to eat · stage: 1 · step: 2 (combine) · prior: §3 "Take as-is", "Staging"; §1 table "Ours beside theirs"
**Why.** The record's largest effect is the world, not the brain: a
success-gated dial scored 0.716 against 0.001 without it.
**Do.** In `jack_env.py`, each a named option, default = today's world,
each with its source copied into the docstring's "ours beside theirs"
table and one unit test (on changes what it says; off is byte-identical to
J1's proof). (1) One dial `d` in [0,1], today's world at d = 1: meal size,
number of floor foods, distance of food from spawn, respawn delay (instant
re-placement at a new random place at low d). (2) Random starting needs and
random layout each life. (3) Reward: 100 × drive reduction plus a payment on
the eating event; an option under which merely being alive and hungry is
never negative (if J1's answer 2 says dying early pays). (4) Death is
`terminated`, a time limit is `truncated`. (5) Then MEASURE, rule first, 20
seeds × 10 lives per setting: the entry `d0` — the hardest d at which J1's
best random row eats ≥ 1 meal in ≥ 50% of lives and do-nothing still dies
(§3: accept a rung only if random neither always fails nor already
succeeds). The gate that moves d on success is a later order; judging stays
on fixed layouts at d = 1. Learner-side settings belong to J3/J5.
serves: M2 · PLAN stage 1 lines 2, 5 · philosophy 5 (his world matters as much as he does).
**done when:** options and tests are committed and `WORLD_CARD.md` names `d0` with its table.
state: open — ordered 11-15 16:57 (was J5; widened by §3).

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
recurrent) are next week's. (4) The same rows 1 and 3 on the in-house arm
(`cores.py` tanh-AWR) through a thin adapter: suspect or candidate.
serves: M3 · PLAN stage 1 "each brain reproduces published numbers" · philosophy 3, 4.
**done when:** `KNOWN_ANSWERS.md` has rows 1–4 inside their ranges and the in-house arm's rows, pass or fail.
state: open — ordered 11-15 15:55; corrected by §5 at 16:57, nothing run yet.

### J4 — The harness checks itself before it judges anyone (§8's thirteen)
part: His brain (the instrument that chooses it) · stage: 1 · step: 3 (reproduce) · prior: §8 "Reproduce first", ruling 19
**Why.** The released AdaStop package rejected 10–12% of same-against-same
comparisons at a stated 5%. A tool that has not passed same-against-same is
not believed here.
**Do.** `experiments/harness/` (new, small; SciPy only, plus AdaStop ONLY
from the commit §8 pins). (1) The outcome code of ruling 19: decisions until
W consecutive lives average ≥ 5 meals, censored at T; permutation log-rank;
looks at 10 and 20 runs per arm at 0.025 each. (2) The checks that need no
learner, each a test that can fail: #1 same-against-same through the real
pipeline, ≥ 1,000 repetitions on synthetic censored logs, false alarms ≤
the stated rate; #10 outcome code right on synthetic logs with edge cases;
#12 4/15 against 0/15 gives Fisher 0.100 and Boschloo 0.043; #11 the
analysis script hashed; #13 runs analysed = runs launched; #3 floor and
ceiling arms. (3) List #2, #4–#9 in the module as owed, each with the order
that will pay it (they need B, from J3). Result table →
`docs/research/HARNESS_CHECKS.md`. No ledger spec; this is the instrument.
serves: M3 · PLAN stage 1 "the test harness passes its own thirteen checks" · philosophy 4.
**done when:** the six checks pass as tests and the seven owed are listed.
state: open — ordered 11-15 16:57.

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
state: open — ordered 11-15 15:55 (was J3), nothing run yet.

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
state: open — ordered 11-15 16:57.

### J7 — Close what is in flight, and the world mechanics that are simply red
part: His world, needs and lessons · stage: 1 · step: repairs (PLAN "continues")
**Do, one per slot, Kaggle CPU or detached, never in-session past 15 min.**
(a) Harvest the forager kernels as they exit (random control; ppo-needs
s153, s154; dreamer-xs s153; logs `/data/tmp/forager3_*.log`). **Launch no
more.** `forager_race3 tally` will read INCOMPLETE — the honest verdict.
Append to `docs/research/FORAGER_RACE.md` under the heading "PILOT —
learners not known-answer-checked, below every published foraging budget";
no seat moves; then remove `/data/tmp/wt_forager3`. (b) `PG.6` detached
(~20+ min wall). (c) `NE.01` on the shipped world under the fatigue edge
probe (`eaa9302`), one run. (d) `PS.02` v3 (ruling 12): lives with NO FIRE,
bars unmoved, frozen-temperature control must still FAIL; then run. (e)
`run W1.04` once; a FAIL at ~53 s is the honest reading (§3: the same
disease as the food economy) and is followed by nothing.
serves: M2 · philosophy 4 (a failure is fixed, not filed).
**done when:** nothing is in flight and each of the four specs has a new ledger row.
state: (a) ppo-lp s153/s154 harvested 15:38 (floor meals 13/17 over 60 lives, apples 0); 4 kernels running.
state: (c) clause applied `eaa9302`. (d) cause pinned 11-15 08:42 (dumps `/data/tmp/ps02dump/`). (b) in-session try killed at 20 min, no row.

**Paused or closed by the record (ruling 20):** `D1.0` — §4 answers it (a
bigger actor does not help; capacity belongs in the critic); no more GPU
goes to the 57M arena; cached units stay in `/data/d10_units/873e861e…`;
watcher 136333 exits by itself — never kill it. `T2.01` — no more one-world
walking attempts with plain learners. `LF.06`, `LC.09`, `CR.01`, `LF.04` —
"to earn again" on the brain stage 2 chooses.
**Next up, in order:** 1. Idle-slot reproductions that depend on no brain
choice (§3, §6, §7; one per slot, table row in `KNOWN_ANSWERS.md`): the 8×8
environment-shaping grid (0% against 70%), Dyna maze, RL² bandits, Speech
Commands keyword spotting; the MJX humanoid tutorial on the T4 from 11-21.
2. The clone (E2b) and the cheat learner (E2c), after J1 and J3. 3. Harness
checks #2, #4–#9 with B. 4. B's 20-run pilot in his world at `d0` (after
J1–J4). 5. A's known-answer rows on the T4 from 11-21 (after J5). 6. The
dial's success gate; the two-resource ant reproduced in our simulator.
7. Diary retrieval keyed on learned state (§6), stage 3.

---

## SYSTEM — work on the machine (max 3 open)

*Order of work: **S6** → **S7** → **S8**. S6 has sat four days unstarted:
the next SYSTEM slot does S6 step 1 and nothing else.*

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
state: open — not started (fourth day); split 11-15 16:57.

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
state: open — ordered 11-15.

### S8 — A seat can be held BY RECORD, and `champions --check` knows it
part: (machine — His brain) · stage: 1 · step: —
**Why.** PLAN: "A seat may be held BY RECORD: on published evidence, cited,
and open to any challenger that brings its own prior." Ruling 11 defines
it; ruling 20 names the first candidate (the control-path seat, §4).
**Do.** `experiments/champions.py`: accept holder kind `BY RECORD` when the
seat's row carries (a) a pointer to a `PUBLISHED_PRIORS.md` section that
exists, (b) a known-answer row id in `docs/research/KNOWN_ANSWERS.md` for the
holder's build, (c) a named challenger route. `--check` fails a BY RECORD
seat missing any of the three; planted tests for each. Do NOT seat anyone:
no build has a known-answer row yet. Add nothing to `CHAMPIONS.md` but the
definition (≤ 6 lines, replacing as many).
serves: M3 · philosophy 3.
**done when:** the three planted failures fail and `champions --check` exits 0.
state: open — ordered 11-15 15:55.

**Next up:** (1) `coverage` prints where each passing CLAIM was earned (in
him / bench / parent rig). (2) `gpu.py` reattach does its own pin-swap
proof; watcher timeout from RUNNING. (3) `BD.01`, `LC.08` budget re-class
(ruling 14). (4) **S4** — six PASSes gated on a seed average that admits a
failing seed. (5) `T0.13`'s dead clauses. (6) `LC.02`'s salt replay excludes
wall-clock metrics. (7) D52 scout retry. `T0.33`→`T0.34`, `T0.12`→`T0.24`
re-buys are regate's on a clean tree.

---

## INBOX — one line each, workers add, the Director empties

- 11-15 19:18 OWNER'S SESSION — the owner on porting Dreamer to Apple's MLX: "we can port it sure, how easy would it be?… figure out how hard the port would be and start building it… but first max research so it's ready when we get the Mac and we see it can be built." Research done: `PUBLISHED_PRIORS.md` §11 and `docs/research/PUBLISHED_EVIDENCE_2026-11-15_PORT.md` (the build plan is its table of 23 steps, each one or two slots with a done-when test; the 62-item numerical contract with file-and-line links is in `/data/jack-research/research_notes/Porting Dreamer to Apple MLX/dreamer_port_spec.md` §2; a clone of the official repository at the reference commit is at `/data/tmp/jack-research-scratch/port_spec/dreamerv3/`). **Ask: order it as a SIDE LANE with a stop rule; nothing in stage 1 or the decisive test depends on it.** (a) Stage 0 first, 6–8 slots: MLX in a container or on Kaggle CPU (it cannot load natively on this box: glibc 2.34), the golden files recorded once from the official code, and gate G0 measured — compile time and peak memory of the full unrolled training step at both sizes (proceed if `size12m` is under 10 minutes and 8 GB; the threshold is ours). A failed G0 stops the port and is a finding. (b) Then the acting-only slice (15–25 slots), which is also what an owner's Apple device would need to run a trained creature. (c) Then the rest, rung by rung; no step is written before the rung below it passes. (d) The golden files serve J5 too: they are the same-weights, same-batch check that made the PyTorch ports faithful. (e) Paid GPU time: the owner has said they will pay for a rented card for the large runs AFTER the system is proven; **nothing is authorised**, no agent may spend, and the official JAX code, not the port, is what would run there. The container must be small and short-lived on this shared box (one core, 2 GB, removed after use), or the work goes to Kaggle.
- 11-15 18:47 OWNER'S SESSION — the owner asked for a deep review of the Mac mini M6 (32 GB) since no Apple machine exists to test on: `PUBLISHED_PRIORS.md` §10 and `docs/research/PUBLISHED_EVIDENCE_2026-11-15_MAC.md`. Nothing is measured on a Mac; every workload speed is scaled. For the 06:37 sitting: (a) **J5** — on the Mac the official JAX Dreamer runs on the processor only (jax-metal is abandoned), no Dreamer code has been reported on Apple silicon, PyTorch's graphics backend is slower than the processor for small recurrent cells and has an open wrong-gradient bug; so the PyTorch port must be proven on BOTH processor and graphics chip, and §10's first-day list (twelve measurements, each replacing an estimate) should exist as one script before the machine arrives; (b) the decisive test's Dreamer arm is realistically the 0.64M model, at a training rate fixed by measurement, with the free T4 running alongside (estimated 1.6–5 weeks together); the 9.7M model's twenty runs are not affordable anywhere, which the pre-registration must say; (c) portability rules for new code: no reliance on background priority on macOS (it pins work to the slow cores), paths from environment variables, the camera needs a logged-in desktop session there.

- RESEARCH: his eye — what the published record says about giving a small from-scratch agent sight from its own camera: resolution and frame cost that world-model and replay learners need, decoder-free losses, rendering MuJoCo camera images on free CPUs, a T4 and Apple silicon, richer rays as a cheaper first eye, and how 'a sense added later against present from birth' has been tested; so the added-sense half of the decisive test can be designed before the Mac arrives (PLAN, 'The decisive test').
- 11-15 17:56 OWNER'S SESSION — the owner on the end goal: "we are doing everything we can to set it up for the ultimate test at the end, as good as possible, where we will actually train them for 100s of hours on the Mac mini 32GB; only then we'll know for a fact what wins — do we have enough now to set up and build?" `docs/PLAN.md` now has **'The decisive test, and what must be ready before the Mac arrives'** (late December; the Mac is shared with his other projects and is not here yet: do not start any move). Asks: (a) treat that ready-list as stage 1's deadline and say in PROGRESS which items are done; (b) order B's pilot in his world on Kaggle CPU as soon as J2's first rung and J3's reproduction exist — the first result that needs no Mac; (c) J5 must say which Dreamer implementation can run on Apple hardware (§4: jax-metal is stale; the PyTorch port on MPS is untested) and prepare a one-command speed test for day one (simulator; each learner on CPU and on the graphics chip); (d) keep new stage-1 code free of Linux-only assumptions (the porting list is in the session's notes: hard-coded `/data` and `/home/opc` paths, `flock`, `timeout`, `ionice`, `/proc/loadavg`); (e) the eye is queued above as a `RESEARCH:` line.
- 11-15 17:02 OWNER'S SESSION, code review of `experiments/jack_env.py` (`9ee9965`) against §2, §5 and §8: sound as a wrapper (it adds no rule, the identity proof with `run_survival` is the right proof, and its docstring states its own caveats). **One thing must change before brain A (J5) touches his world, and it should be decided before J2 adds options:** on the decision he dies, `step()` returns the RESPAWNED body's observation with `terminated=True`, because `W0.decide` respawns inside itself and the dying body's senses are gone. For B this is harmless (a replay learner does not bootstrap from a terminal next-observation). For a world model it is the 'next observation at episode ends' bug of §5 and fails §8's self-check 8 ('true death no bootstrap; last observation true'): Dreamer encodes the terminal step's observation and trains its continuation head on it, so it would learn that a full-energy body in a new place is what death looks like. The fix needs the dying body's observation captured before the respawn WITHOUT consuming the world's random stream (W0.observe draws the placebo and the noise-panel texture from it, so a second call would break the identity proof). That is a small hook in `w0.py`, which stage-1 orders may not edit: a Director ruling, then the hook with the identity proof re-run. Two smaller notes: `from experiments.survival import D_DEATH` and `from experiments import drives` sit inside `step()` (move to module top); the observation space is unbounded, so J3 must normalise observations (§5 says so) and the J3 docstring should say which statistics are frozen at evaluation.

## DONE (kept one day)

- **Review 2 filed** (`9f1ed07`): `PUBLISHED_PRIORS.md` §1–§9. Its eight asks
  → J1–J6, rulings 17–21, "Next up" 1 and 7. All `RESEARCH:` lines answered.
- **`jack_env`** (`9ee9965`): his world behind the standard interface, proven identical on 3 seeds.
- **Forager race: NONE CLEARS** (`5a088e0`); the record predicted it. Now a pilot (J7a).
- **Policy-loss race: tanh-AWR seated** (`a2bd4df`); `LC.02`, `LF.03` PASS re-bought.
- **S3**: poller rule, Saturday GPU week (`2bb1473`, `1e025eb`).
- **OUTSIDER "he cannot see food": measured, not so** (`56bf1dc`).
