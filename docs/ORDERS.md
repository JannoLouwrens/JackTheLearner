# ORDERS — what the workers do. Written by the Director.
Director sitting: 2026-11-15 15:55 UTC

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5). **One state+journal commit per slot,
not one per step.** Machine problems go under INBOX, one line each. Rulings
10–16 cited here are in `docs/DECISIONS_RESOLVED.md` under 2026-11-15 15:55.

**`docs/PLAN.md` binds every order (owner, 11-15).** Research → combine →
reproduce → test the last 10% → put it in him. We are in **stage 1: the
instrument** — before any brain is judged, the thing that judges must be
right. Every order names `part:`, `stage:`, `step:`; the reference for what
the record says is `docs/research/PUBLISHED_PRIORS.md` (§1 today; review 2
adds sections tonight — **re-read it at the start of every slot and follow
its exact versions and numbers where it is sharper than this page**).
**Stopped (ruling 10):** races between home-made learners; new forager cells;
world-rule races (humid, DRINK, W1 energy clock); `D1.0` GPU spending; new
specs with no published prior. No in-house arm (`cores.py`) is a candidate
until it passes J2's known-answer checks. A question with no prior is a
`RESEARCH:` line in INBOX for the Researcher, never a run.
**Continues:** harvesting what is running, repairs, tests of world mechanics.
**Where things run.** Box CPU: day ceiling 57,600 s, detached ≤ 34,560 s and
≤ 2 computing detached jobs (pollers < 0.10 cores do not count, ≤ 6). Short
known-answer runs (< 15 CPU-min) may run on the box at nice 19; everything
longer goes to Kaggle CPU kernels (`prefer="kaggle-cpu"`, cap 5, use 4,
free). Kaggle GPU has **0.88 h** left until Sat 11-21 00:00 UTC; Colab is
open (`colab --auth adc sessions`). World files (`w0.py`, `needs.py`,
`survival.py`) are not edited by stage-1 orders: the setup lives in the wrapper.
**Launching past another session's dirt:** from a clean worktree, as J1 did at `e08c680` (S7 ends this).

---

## JACK — build the creature, his world and his curriculum (max 7 open, ranked)

*Order of work: **J1** → **J2** → **J3** → **J4** → **J5** (J2 and J3 need
nothing from J1 — start them in the same slot if J1 is waiting on anything)
→ J6 → J7. The 07:07 and 19:07 strong-worker slots take J3 if J1 is done.
Waiting on a kernel is not work: advance the next order in the same slot.*

### J1 — His world behind the standard interface, speed measured, and the noise probe (E0)
part: Learning to eat · stage: 1 · step: 2 (combine) · prior: `PUBLISHED_PRIORS.md` §1
**Why.** Every published learner plugs into Gymnasium. Until W0 is a
`gymnasium.Env`, neither brain of the plan can touch his world.
**Do.** (1) `experiments/jack_env.py`: W0 + his needs + his current senses as
a `gymnasium.Env` (1.1.1 is installed) — continuous `Box` action, flat `Box`
observation, one step = one 0.2 s decision, `terminated` on death, `info`
carries meals, drinks, cause of death. It WRAPS `w0.py`/`survival.py`; it
edits neither. Default options reproduce today's world exactly. (2) Proof it
is the same world: `gymnasium.utils.env_checker.check_env` passes, and a
fixed-seed random-action life through the wrapper matches the same life
through the existing loop (same death time and meal count, 3 seeds). (3)
Speed: decisions per second, one env and a 4-process vector, on the box and
on one Kaggle CPU kernel — written in the docstring (1M decisions ≈ 10 h is
today's guess). (4) **E0, rule committed before any run:** random policies
only — white noise per decision (today's), pink noise per life, action held
8 and 32 decisions — × 20 seeds × 10 lives: food contacts and meals per life.
Record's prediction: correlated noise finds 2.5–4× more food (an uncalibrated
toy figure; this calibrates it). No learner, no seat, no ledger row.
serves: M3→M7 · PLAN stage 1 line 1 · philosophy 3, 5.
**done when:** (2) and (3) are in the module docstring and E0's table is in
`docs/research/E0_NOISE_PROBE.md`.
state: open — ordered 11-15 15:55.

### J2 — Brain B from published code reproduces published numbers
part: His brain · stage: 1 · step: 2→3 (combine, reproduce) · prior: §1 "B" and "Known-answer checks"
**Why.** No learner here has ever reproduced a published result; B (a SAC-
family replay learner) is the record's cheap control and the first to check.
**Do.** `experiments/brain_b.py`: SAC from Stable-Baselines3 (2.7.1 is
installed) — NOT a rewrite. Docstring = the design note: source and version,
and an "ours beside theirs" table (replay = whole run, batch 256, 1 update
per decision, LayerNorm in actor and critic via `policy_kwargs`, observation
normalisation, pink or gSDE noise as an option). Known-answer rows, 5 seeds
each, rule committed first: Pendulum-v1 ≈ −157 by 20k steps (pass: mean final
return ≥ −200); MountainCarContinuous-v0 88–95 by 50k with persistent noise
(pass: ≥ 4 of 5 seeds ≥ 85) and, beside it, white noise (record predicts ~0 —
if white noise also solves it the check is too easy: say so); the five probe
environments. Table → `docs/research/KNOWN_ANSWERS.md` (task, steps, published,
ours). A miss means our build is broken: fix it, do not go on. Then run the
SAME gate on the seated in-house arm (`cores.py` tanh-AWR) through a thin
adapter — its row decides whether it is a suspect or a candidate.
serves: M3 · PLAN stage 1 lines 2–3 · philosophy 3, 4.
**done when:** `KNOWN_ANSWERS.md` has B's rows passing and the in-house arm's rows, pass or fail.
state: open — ordered 11-15 15:55.

### J3 — Brain A (Dreamer) from the official code runs, and its speed is measured (E3)
part: His brain · stage: 1 · step: 2 (combine) · prior: §1 "A", E3; review 2 "building A" when it lands
**Why.** A is the record's recommended whole brain. Nobody has published its
speed on a T4 or a CPU, and our old "dreamer-xs" had no action input and no
imagination — it was not a Dreamer.
**Do.** (1) Choose the source and pin its commit in `experiments/brain_a/README.md`:
the official `danijar/dreamerv3` unless review 2 names a verified port;
licence noted; code only, never weights. (2) Run its smallest preset
(`size1m`) unmodified on one small standard task on a **Colab T4** and on one
Kaggle CPU kernel: decisions per second and wall-clock for 100k decisions.
(3) Write the "ours beside theirs" table: every setting we would change for
his world, with the reason (none changed yet). (4) A's gate, declared now:
match B's `KNOWN_ANSWERS.md` score on the same small tasks, and one seed's
outputs equal the official code's. Do not wire it to his world yet.
serves: M3, M5 · PLAN stage 1 lines 2–3 · objective 3 (the only GPU-able order) · philosophy 2, 3.
**done when:** the pinned source, the two speed figures and the gate are committed.
state: open — ordered 11-15 15:55.

### J4 — Can the task be learned with his senses? The rays-only forager (E2a)
part: Learning to eat · stage: 1 · step: 4 (the record does not answer it) · prior: §1 "Learnable?"
**Why.** "Can 16 distance rays tell food from a wall?" is answered nowhere in
print. `NE.01`'s forager eats 24–27 meals but reads the world's state.
**Do.** Needs J1. Rule committed first. A HAND-WRITTEN forager fed only
`jack_env`'s observation (what he senses — no positions, no world state),
20 seeds × 10 lives, beside the privileged `NE.01` forager and the random
floor. Bar (from `LF_04_FAIL.md`, unmoved): ≥ 5 floor meals per life, mean
over seeds, and above random on ≥ 18 of 20 seeds. Control that must fail: the
same forager with its rays shuffled. It is a ruler, never a part of him.
PASS → the task is learnable from his senses. FAIL → the finding is "his
senses cannot do it"; write which sense is missing (the Director orders the
sense, not a brain). The clone (E2b) and the cheat learner (E2c) follow when
J2 is done — the Director orders them then.
serves: M2, M5 · PLAN stage 1 line 5 · philosophy 4.
**done when:** `docs/research/LEARNABLE.md` has the table and the verdict.
state: open — ordered 11-15 15:55; waits on J1 step 1.

### J5 — The setup the record supports, as switches on the wrapper
part: Learning to eat · stage: 1 · step: 2 (combine) · prior: §1 table "Ours beside theirs"
**Why.** The record's largest effects are in the setup, not the brain:
curriculum 0.716 vs 0.001; tiny replay 0.04–0.07 of normal.
**Do.** Needs J1. In `jack_env.py`, each a named option, default OFF, each
row of the "ours beside theirs" table copied into the docstring with its
source: reward = 100 × drive reduction plus a payment on the eating event;
energy randomised at birth; food placed near first with the distance a
parameter (the stage rule is J-later, not here); a time-limit `truncated`.
Learner-side settings (discount 0.995, replay, noise) belong to J2/J3, not
here. One unit test per option: on changes what it says, off is
byte-identical to J1's proof. No learner runs, no race.
serves: M2 · PLAN stage 1 line 4 · philosophy 5.
**done when:** the options and their tests are committed.
state: open — ordered 11-15 15:55; waits on J1 step 1.

### J6 — Harvest the forager pilots, then stop
part: Learning to eat · stage: 1 · step: — (a pilot of unchecked learners; ruling 10)
**Do.** Four Kaggle CPU kernels are running (random control both seeds;
ppo-needs s153, s154; dreamer-xs s153; logs `/data/tmp/forager3_*.log`).
Harvest each as it exits. **Launch no more** — not dreamer-xs s154, not
`wm-latent+ring`. Run `forager_race3 tally` on what exists; it will read
INCOMPLETE — that is the honest verdict. Append to
`docs/research/FORAGER_RACE.md`: the per-cell table, the heading "PILOT —
learners not known-answer-checked, 40k–160k decisions, below every published
foraging budget", and ppo-lp's one rising seed as a lead, not a result. No
seat moves. Remove `/data/tmp/wt_forager3` afterwards.
serves: PLAN "what stops today" · philosophy 4.
**done when:** the four are harvested, the page says PILOT, nothing is in flight.
state: 11-15 15:38 ppo-lp s153/s154 harvested (floor meals 13/17 over 60 lives, apples 0; rho −.006 p=.52 / .25 p=.025).

### J7 — World mechanics that are simply red or stale (`NE.01`, `PS.02`, `W1.04`, `PG.6`)
part: His world, needs and lessons · stage: 1 · step: repairs (PLAN "continues")
**Do, one per slot, Kaggle CPU or detached, never in-session past 15 min.**
(a) `NE.01` on the shipped world under the fatigue edge probe (`eaa9302`) —
one run, verdict written; no arm is re-tallied (those races are stopped).
(b) `PS.02` (ruling 12): re-register v3 — the probe's lives are run with NO
FIRE in the world, so cold alone decides each death; every bar unmoved; the
frozen-temperature control must still FAIL; v2's reading stays on the row,
reporting-only. Then run it. (c) `run W1.04` once (pollers per state of
11-15, `f6c6177`); a FAIL at ~53 s is the honest reading and is not followed
by a clock race. (d) `PG.6` detached (needs ~20+ min wall).
`T2.03` and `W1.01`'s re-pin wait: GPU, and ruling 13.
serves: M2 · philosophy 4 (a failure is fixed, not filed).
**done when:** each of the four has a new ledger row.
state: (a) clause applied `eaa9302`, smoke: spread_f .81, frozen control 0.0. (b) cause pinned 11-15 08:42 (s2 life 0, fire prolongs it; dumps `/data/tmp/ps02dump/`). (d) in-session try killed at 20 min, no row.

**Paused by the plan (ruling 10), nothing lost:** `D1.0` (units ref,
aprime:0, aprime:1 cached in `/data/d10_units/873e861e…`; watcher 136333 is
left to exit by itself — never kill it; its next submit is refused by the
0.88 h meter) until review 2's body section lands; `LF.06`, `LC.09`, `CR.01`,
`LF.04` re-runs — "to earn again" on the brain stage 2 chooses.
**Next up, in order:** 1. E2b clone and E2c cheat learner (after J2, J4).
2. E1 for brain A (after J3). 3. The stage-2 prediction and rule, written
from the record, and the 20-seed pilot of B in his world (after J1, J2, J5).
4. Simulator speed (PLAN honest limits). 5. `W.5` fire and the advice gate —
after review 2's world and words sections. 6. Inside/around sweep: bench
proofs re-credited as parent-rig results.

---

## SYSTEM — work on the machine (max 3 open)

*Order of work: **S6** (the owner has waited four days) → **S7** → **S8**.*

### S6 — A video the owner can watch (owner request 11-12) — FIRST SYSTEM SLOT, NO EXCEPTIONS
part: (machine — he can be known) · stage: — · step: —
**Do.** `scripts/life_video.py` renders OFFLINE from the most recent recorded
life — never inside a scored run — with `experiments/render.py` (Xvfb +
llvmpipe): third-person view, hunger / thirst / temperature overlaid,
phone-playable H.264 mp4, **aim ≤ 5 MB** (480p, ~12 frames a second, under a
minute; ceiling 20 MB) at `/data/jack-video/latest.mp4`. The most recent
life, no picking. Run by hand at nice 19; no cron, no service. If no life is
recorded with poses, record one s90 life first (detached, ≤ 600 CPU-s). A
rough 30-second clip today beats a finished one next week.
Beside it write `/data/jack-video/latest.json`: `title`; `summary` (2–4 plain
sentences made by a TEMPLATE from the life's own record: how long he lived,
what he ate and drank, where he sheltered, what killed him — no model,
nothing the record does not show); `facts` (≤ 8 of `{label, value}`);
`events` (≤ 60 of `{t, at, label}` — `t` seconds into the VIDEO, `at` the
time in his life as text — one per meal, drink, fall, shelter entry, and the
death); `source` (life, seed, commit). The dashboard's Watch tab reads both.
serves: M8 · philosophy 1 (he can be known).
**done when:** the mp4 and json exist, the film plays, and it is the latest recorded life.
state: open — not started (fourth day).

### S7 — Another session's uncommitted files must not kill a run
part: (machine) · stage: — · step: —
**Why.** Twice on 11-15 dashboard files nobody imports tripped the tree-wide
dirty guard: `D1.0` lost a 5.9 GPU-hour slice, J1 launched from a hand-made
worktree. Stage 1 will launch many small kernels beside a live owner session.
**Do, one commit each, each with a planted test.** (1) The submit-time guard
in `gpu.py` refuses only dirt inside the spec's import closure plus
`experiments/` core (`offending_dirt` already computes the list); dirt under
`dashboard/`, `docs/`, `scripts/dashboard_status.py` is named in the row,
not fatal. A planted dirty imported file must still refuse. (2)
`scripts/launch_detached.sh` takes `JACK_REPO`. (3) `JACK_REUSE_KERNEL`
applies to the first `submit()` only and is refused when a second body asks
for the same slug. (4) `scripts/test_lane_guard.sh` passes at clean HEAD
while two detached jobs are alive.
serves: philosophy 4, 5 (the loop faster and truer) · objective 3.
**done when:** the four planted tests pass.
state: open — ordered 11-15 (S7 of 06:40 narrowed to what stage 1 needs).

### S8 — A seat can be held BY RECORD, and `champions --check` knows it
part: (machine — His brain) · stage: 1 · step: —
**Why.** PLAN: "A seat may be held BY RECORD: on published evidence, cited,
and open to any challenger that brings its own prior." Ruling 11 defines it.
**Do.** `experiments/champions.py`: accept holder kind `BY RECORD` when the
seat's row carries (a) a pointer to a `PUBLISHED_PRIORS.md` section that
exists, (b) a known-answer row id in `docs/research/KNOWN_ANSWERS.md` for the
holder's build, (c) a named challenger route. `--check` fails a BY RECORD
seat missing any of the three; planted tests for each. Do NOT seat anyone:
no build has a known-answer row yet. Add nothing to `CHAMPIONS.md` but the
definition (≤ 6 lines, replacing as many).
serves: M3 · philosophy 3 (holds its seat only until something beats it).
**done when:** the three planted failures fail and `champions --check` exits 0.
state: open — ordered 11-15 15:55.

**Next up:** (1) `coverage` prints where each passing CLAIM was earned (in
him / bench / parent rig) — the dashboard already computes it. (2) `gpu.py`
reattach does its own pin-swap proof; watcher timeout from RUNNING. (3)
`BD.01`, `LC.08` budget re-class (ruling 14). (4) **S4** — six PASSes gated
on a seed average that admits a failing seed. (5) `T0.13`'s dead clauses.
(6) `LC.02`'s salt replay excludes wall-clock metrics. (7) D52 scout retry.
S3's leftovers (`T0.33`→`T0.34`, `T0.12`→`T0.24` re-buys) are regate's on a clean tree.

---

## INBOX — one line each, workers add, the Director empties

- 11-15 16:37 OWNER'S SESSION, review 2 part 6 landed: **`PUBLISHED_PRIORS.md` §5 (building brain B and the PPO reference)**. It corrects two things in **J2** before any run, both from the source of the numbers, not from a result: (a) the Pendulum row. J2's '≈ −157 by 20k, pass ≥ −200' came (through §1) from the RL Zoo table, which is ONE run per entry; the public multi-seed record for the same configuration is the mean return of all 100 training episodes over 20,000 steps, −298.4 ± 10.3 (13 seeds), accept a 5-seed mean in −314.7 to −282.1. Use §5's rows 1–6 with their stated metric; a bar set on a one-run number would fail a correct build or pass a wrong one. Director: please rule the J2 rows to §5's at the next sitting; workers: per the ORDERS header, §5's numbers are the sharper ones. (b) SB3's SAC does not expose LayerNorm through `policy_kwargs`; SBX's does. The LayerNorm arm is SBX `SAC`; the sparse-task gSDE recipe has its public record in PyTorch SB3. The standard-task checks want their own Python 3.12 environment with §5's pins (the project venv is 3.9 with SB3 2.7.1 and must keep the MuJoCo his world is certified on).
- 11-15 16:30 OWNER'S SESSION, review 2 part 5 landed: **`PUBLISHED_PRIORS.md` §4 (body, movement, simulation speed)** answers the `RESEARCH:` line on the control path (deleted). For the next sitting: (a) **`D1.0`** — the record already answers its question in the direction its void attempts pointed: a bigger actor does not help or hurts, capacity belongs in the critic or world model (SimBa, BRO); if a verdict of our own is still wanted, §4 gives a clean, checkable version (DeepMind Control humanoid-walk, pass mark 600, known published scores) in place of the present arena; a candidate for a seat held BY RECORD under ruling 11. (b) **`T2.01`** failing since August is what published one-world learners do below about 1M steps; walking from scratch without mocap needs either thousands of parallel copies (minutes on a T4) or the 2018 CPU recipe with helping forces that are withdrawn. (c) **A 0.2 s brain cannot balance a humanoid by itself**: every published slow brain hands over to a 30–200 per second reflex controller. (d) **J1 speed:** the rover loop is the bottleneck, not physics (85,000 steps/s for a stock car on this box against our 2,800); the recipe is implicitfast, all sub-steps in one native call, native ray casts, several lives per process; predicted 500–2,000 decisions/s per kernel, to be measured.

- 11-15 16:25 OWNER'S SESSION, review 2 parts 2–4 landed. **`PUBLISHED_PRIORS.md` §3 (world, needs, staging)** answers the `RESEARCH:` line on need clocks (deleted) and half of the one on fire, cooking and shelter (kept until the advice half is filed). Its headline for J5, J7 and the next sitting: **our world is the outlier among worlds that taught agents.** One floor meal is 8% of the bar (published first meals: 44–100%, or 10% only with dense, sensed, instantly re-placed food); two floor foods on 66.9 s timers supply 0.0024/s against a basal 0.0017/s, so floor food alone sustains about a fifth-time activity level and a sustained life needs the ladder apple, the hardest skill, first. The record's fix is a difficulty dial (meal size, count, distance, respawn delay as functions of a success-gated ceiling, today's world near the top), random layouts and needs each life, rays that report what they hit, and judging only on fixed unassisted layouts. It is a world redesign under the plan's steps (combine, then reproduce the two-resource ant result in our simulator), not a switch for J5 to flip; W1's 53-second lives are the same disease. Memory/sleep and ear/words/advice notes are in `/data/jack-research/research_notes/Published evidence for building Jack/`; their sections follow tonight.
- 11-15 16:08 OWNER'S SESSION, review 2 part 1 landed: `PUBLISHED_PRIORS.md` §2 (building brain A). For **J3**: reference = official `danijar/dreamerv3` at `e01491f`; runner = sheeprl 0.8.5 `dreamer_v3_5` (`dreamer_v3_5_1M` / `_12M`), not sheeprl's older `dreamer_v3` nor dreamerv3-torch, whose default actor cannot learn from death; the known-answer rows are six proprioceptive DMC tasks with published mean ± SD (needs `dm_control`), and they cost an estimated 25–60 T4-hours for the first three, so A's reproduction spans next week's GPU quota and Colab. A CPU cannot run the published train ratio. Also note §2's EVAAA line for J5: under drive-reduction reward with termination, published agents learned to end the episode when food could not be found.

- 11-15 15:55 OWNER'S SESSION, for J1 step 3 (speed), measured read-only on the box (1 core, nice 19, seed 153, random actions, no file changed): the world ALONE runs 33–38 decisions/s, so the simulator, not the learner, bounds every stage-1 and stage-2 run. Where the time goes per 0.2 s decision (40 substeps): `mj_step` ≈ 55% (0.39 ms each; the model is on the **RK4 integrator**, `opt.integrator = 1`: 2,800 steps/cpu-s raw, against 10,600 for `implicitfast` and 10,700 for Euler with the option flipped in memory, a 3.8× difference); `ContactAudio.step` ≈ 25%; `drives.substep` ≈ 13%. Flipping the integrator in memory took the whole loop from 38 to 63 decisions/cpu-s (1.65×; Python per-substep work then dominates). **It is a world change, not a free switch:** one 400-decision random life ended at integrity .879 under RK4 and .972 under both other integrators (path 68.8 m vs 62.0 m), so the world's own certificates would have to be re-bought, and plan step 1 comes first (review 2's body-and-simulation section tonight covers what MuJoCo's own documentation recommends; ask the Researcher if it does not settle it). The switch that changes nothing in the world is J1's 4-process vector: about 4× per kernel. Nothing here is a verdict; it is where J1's speed table should look.
- RESEARCH: fire, cooking and shelter learned from scratch, and advice in words to a learning agent — the nearest published tests and their controls, so `W.5` and the advice gate can be registered with a prior.

## DONE (kept one day)

- **Forager race: NONE CLEARS** (`5a088e0`, 7 arms × 2 seeds × 15 lives; best
  late floor meals 0.4/0.2 vs bar 5). The record predicted it: 40k decisions
  is below every published foraging budget. Now a pilot (J6).
- **Policy-loss race: tanh-AWR seated** (`a2bd4df`); `LC.02`, `LF.03` PASS re-bought.
- **`W1.02` PASS** att 11 under the smaller injection (`d1b0880`).
- **S3**: `W1.00` re-classed, poller rule, Saturday GPU week (`6623eb0`, `2bb1473`, `1e025eb`).
- **OUTSIDER "he cannot see food": measured, not so** — rays hit floor food
  on ~20% of random decisions (`56bf1dc`); its other four faults are the
  record's table, fixed by J2/J5.
- **INBOX of 11-15 (21 lines)** → this page, rulings 10–16, three RESEARCH lines.
