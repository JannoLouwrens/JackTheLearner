# ORDERS — what the workers do. Written by the Director.
Director sitting: 2026-11-28 09:26 UTC

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5, **each ≤ 4 lines of text — replace,
do not append; history is in git**). One state+journal commit per slot.
Machine problems go under INBOX, one line each. Rulings 137–141 cited here
are in `docs/DECISIONS_RESOLVED.md` under 2026-11-28 09:26; 131–136 under
11-27 06:41.

**`docs/PLAN.md` binds every order (owner, 11-15).** Research → combine →
reproduce → test the last 10% → put it in him. We are in **stage 1: the
instrument.** `docs/research/PUBLISHED_PRIORS.md` is the reference — read
your order's section (only that section) at the start of every slot.
**Where stage 1 stands (11-28 morning).** **J30 READ: EATS** — brain B
with resets on the kind eye ate 27.41 late meals a life against its blind
and shuffled twins' 3.34 / 2.36, 3 of 3 seeds: a learner eats from his own
eye for the first time, and J24's miss was the learner's collapse, not the
eye. **Four more results came home overnight and nobody has read them**
(every agent was stopped by the usage gate 00:37–09:07): round 1-L's first
five kernels (J18 look 1), B-7's four memory seeds (J29), the ear's third
seed and twin (J14), J28's three GPU cells. **Read them FIRST, each once,
by its sealed reader** — then fill the three Kaggle CPU sessions standing
empty (J32).
**Never idle on a question (ruling 127).** (a) Every order carries
`branches:` — what to do on each likely outcome; do it, do not ask. (b) A
missed gate stops THAT ORDER's next step, never the lane: write the miss,
take the next order, then Next-up. (c) A question that only chooses between
options an order already sized, and moves no pass line, clause, control or
exemption (ruling 114 stands), is written to INBOX with YOUR choice first:
`- QUESTION (lane, order): <decision> | options: <A / B> | default: <what
I do if unanswered>`; `scripts/answer.sh` answers between sittings; if
nothing answers in 3 hours your choice is the ruling — act on it and say so
in the state line. (d) A slot may end "nothing to do" only when every order
is waiting on a running kernel AND Next-up is empty; say which kernels.
**A free session is a finding (ruling 135).** A JACK slot that sees fewer
than 5 Kaggle CPU sessions running, and an order whose rule is frozen or
one freeze away, launches it — a draft the lane itself wrote, with J-order
arms, seeds, bar and twins unchanged and a prior in the file, may be frozen
by the worker under ruling 127(c) (INBOX line with the choice, 3 hours).
**A line is not the worker's to move (ruling 114).** After a read has
failed, nobody but a Director sitting changes a pass line, a clause, a
control or an exemption.
**Usage.** The shared weekly meter reads 13% all models (resets Thu 12-03
09:00 UTC); it read 100% from 11-28 00:37 to 09:07 and stopped every agent. Every multi-segment job stays a SELF-ADVANCING CHAIN (ruling
109); raw results land where the next slot can read them and the state
line says where.
**One world (ruling 76): mujoco 3.17.0.** Every row about his world runs in
`/data/venvs/jack-std` (box) or a kernel pinned to 3.17.0. Learner tests
(`sbx`, `optax`) run under jack-std, not the project venv.
**Every new run in his world passes** `eye_frame="body"`,
`motor_frame="body"`, `food_floor_only=True`, `lazy_contact_audio=True`,
`native_substep=True`; brain B uses `ent_coef="auto"` (ruling 64); "eats"
reads the LATE half of the run (ruling 65). An eye row is read by eating —
late meals a life against its own blind and shuffled twins, seed by seed;
critic rulers are printed beside it and decide nothing (ruling 102).
Learner runs in his world only through J32 and the forge's frozen rounds;
every NEW run in his world passes `terminal_obs="dying"` (ruling 138).
**Stopped (rulings 10, 20, 23, 36, 59, 80, 92, 101, 107, 111, 113, 131):**
home-made learners; forager cells on the 16-ray compass eye; world-rule
races; `D1.0`, `T2.01`, smell runs; the needs-hidden row; ANY learner seed
on the per-geom paint; a per-ray one-hot eye arm (J26: it tells no more
than the bins, 0.585 vs 0.606); a learner residual on the hand reflex; the
MLX port; new specs with no published prior. A question with no prior is a
`RESEARCH:` line in INBOX.
**Kaggle CPU: 5 sessions IN ALL (ruling 85).** Rank: (0) any kernel
< 1 h; (1) J18 round 1-L (tranche 1's last kernels, then tranche 2 if
look 1 asks for it); (2) J32's pilot chains. Launch from a clean
worktree ff'd to a PUSHED commit (its own worktree per job, `JACK_REPO=`);
copy its receipts into main's `gpu_submissions.jsonl` on harvest; do not
remove a worktree while a watcher lives. Every JACK slot's first act:
`/data/venvs/kaggle/bin/kaggle kernels status` on each running slug —
harvest what is home, refill what is free.
**GPU (rulings 96, 124).** The Kaggle account is shared: OUR meter counts
only our jobs, and **Kaggle's refusal is the truth** (a refusal now closes
the week in `gpu_budget.json`, `59160055`). W48 (30 h, ends Sat 12-05
00:00 UTC): the ear and J28's cells are home, A's seed 10 is home, 11 is
running, 12 follows by itself (pid 3886689) — about 23 h in all; the
spare ≈ 7 h is J33's. A refusal ends the week's GPU plan; nothing retries
in a loop. Colab: one try per job, no loop. No first launch of a GPU job or
of a NEW learner shape on a CPU kernel without a timing cell (rulings 86, 99).
**The box (rulings 32, 53).** Day ceiling 57,600 CPU-s, ≤ 2 computing
detached jobs. < 15 CPU-min → box, nice 19. The box is shared (never touch
other projects). `pgrep -af` the same command line before launching.
**Disks:** `/` 6.2 GB free (workers refuse under 3); `/data` 27 GB free. Scratch,
clones, caches under `/data` only; the slot that harvests a round deletes
its worktree's caches (not raw results) once the receipts are copied.
**World files** (`w0.py`, `needs.py`, `survival.py`, `drives.py`): changed
only through `jack_env.py` options whose default is today's world.
**Known-answer rows (rulings 33, 70, 94):** seeds AND at most one extension
stated BEFORE the first seed. A miss is a miss.
**Receipts:** the slot that launches a kernel commits
`experiments/gpu_submissions.jsonl`, `gpu_budget.json` and `cpu_budget.json`
by name.

---

## JACK — build the creature, his world and his curriculum (max 7 open, ranked)

*Order of work: the four reads, each once, in ONE slot if they fit —
**J18** (look 1) → **J29** (B-7) → **J14** (ear score, A's seed 10) →
**J28** (harvest) — then **J32** (fill the empty CPU sessions) → **J33**
(GPU, behind A's seed 12) → **J31** (text) → Next-up → exit. A slot
waiting on a kernel advances the next order.*

### J18 — Forge round 1-L: READ LOOK 1 NOW — g1-0…g1-4 are whole
part: Learning to eat + The science lab · stage: 1 · step: 4 · prior: §19–§22; `docs/FORGE.md`; rulings 61, 65, 69, 83, 105, 112, 122, 123, 130, 135
prior: rung M (300k, 5 seeds, lock `2eb2c0f4`): K1 resets 0.84, K2 TQC 0.82, baseline 0.059, twin 0.017. Record: resets + LN critic best-supported (Nauman 2024); nobody has published a small agent still improving at the end of a long life (§20). J30 beside: K1 on his own eye still climbing at 450k.
**Frozen (do not reopen):** rule `rounds/r1L/spec.json` `52033df2`; judge
lock `9d3feb7e` (n_perm 2000, seed 0, ruling 130); PRIMARY K1 against
baseline E-food on `outcome_sustained` W = 2, T = 1M, two looks at 10 and
20 runs an arm, 0.025 each; KEEP RULE: if a look says K1 differs, B's keep
setting is whichever of K1/K2 has the higher late-half mean at 1M, closer
than K1's own seed range → K1. A lost run is a crash, never replaced.
**Do.** (1) `python -m experiments.forge.r1L_read --tranches 1` ONCE
(jack-std), `L_g1-0…g1-4.json` in `/data/jack-logs/forge/r1L/raw/` (last
home 11-28 08:48). (2) Write the verdict, check #13's books and the beside
rates into `docs/FORGE.md` and the scoreboard. (3) Chains B (pid 237409,
g1-5 → g1-7) and C (pid 1449092, g1-6) keep running: K2 and the twin are
BESIDE and are read when home.
branches: A_DIFFERS → the keep setting by the frozen rule goes into `stage2_a_vs_b.md` and J32's draft; if K2's runs are not home yet the keep rule waits for them and J32 uses K1 meanwhile (the tie rule's own answer). · FUTILE → written as read; B's stage-2 setting stays K1 BY RECORD + J30; INBOX one line. · NEEDS_LOOK_2 → launch tranche 2 (seeds 1431–1440, K1 + baseline) yourself from pin `495a766d` on free sessions, claim files as chain C's, `JACK_WATCHER=cpu`. · the reader refuses (lock, hash, a missing run) → the refusal text into the state line and INBOX; do not edit the reader or the lock.
serves: M3, M7 ("he keeps learning") · philosophy 1, 3, 4.
**done when:** look 1's verdict is in `FORGE.md`; then tranche 2 runs or the round is closed with the keep setting named.
state: 11-28 09:25 **look 1 REFUSED by the sealed reader**: g1-0…g1-4 all whole (K1+E-food 1421–1430), but `r1L_read --tranches 1` wants the WHOLE tranche — missing E-food-lr0 1421–1422 and K2 1421–1430 (g1-5…g1-7; B and C chains RUNNING, wrap pids 237409/1449092; chain A done, its g1-4/g1-6 refused as claimed). Reader locked → not edited; INBOX QUESTION. Raw not opened, no rates printed. Output `/data/jack-logs/forge/r1L/reads/look1.txt`.
state: 11-28 09:26 (Director) g1-0…g1-4 WHOLE (13:59, 12:53, 03:32, 05:41, 08:48), raw not opened. Chain A ended after g1-2; B and C alive per `pgrep`. 3 of 5 CPU sessions free.
state: 11-27 03:35 **JUDGE LOCK `9d3feb7e`** committed alone at `85ca3d4e` before any raw file; both readers `check_lock` first. Rule FROZEN `52033df2` 11-26 20:27; launched 20:30 from pin `495a766d`; chain C 11-27 14:15 with g1-4 + g1-6 claimed (`d1361856`).

### J29 — Brain B with a memory: READ B-7 NOW — the four seeds are home
part: His brain (B) · stage: 1 · step: 3 (reproduce) · prior: §5 row 7, §6; `KNOWN_ANSWERS.md` row B-7 (FROZEN `c848d888`); `prereg/stage2_a_vs_b.md` row `B-mem`; ruling 136
prior: published — Ni et al. (arXiv 2110.05038): LSTM-SAC on velocity-only Pendulum ≈ −170 by 40k–50k steps (band −190.9…−154.5, 4 seeds); memoryless SAC cannot solve it. Ours: twin `B7flat` −1490.2 (fails as required); same learner with the full observation −148 / −143 (the twin's miss is the missing memory).
**Do.** `python -m experiments.b7_read` ONCE (reader sealed `127a6bc4`;
`b7mem_s{0..3}.json` in `/data/jack-logs/j29/b7mem/`, home 11-27 20:00 –
11-28 01:59). Row into `KNOWN_ANSWERS.md` B-7 exactly as read; copy the
receipts (the chains ran from MAIN's tree).
branches: REPRODUCES → `stage2_a_vs_b.md` (draft) names `B-mem` = this arm; INBOX one line. · MISS → a miss, written with the four seed scores and the curve; the arm is a suspect, NOT a stage-2 arm; INBOX — the sitting rules (no extension exists). · VOID (a seed LOST) → written VOID with the segment log; INBOX; nothing is re-bought by a worker.
serves: M3 (stage-2 ready-list: "a control brain that has a memory arm") · philosophy 3, 4.
**done when:** B-7's row is written from the sealed read.
state: 11-28 09:23 **(6) READ — REPRODUCES, DONE `054a7ce1`**: B7mem s0/s1/s2/s3 −174.14/−170.45/−162.05/−183.88, arm **−172.63** in −190.9…−154.5; twin −1490.23 fails. Row B-7 written in KNOWN_ANSWERS; `stage2_a_vs_b.md` (draft) names `brain_b_mem.py` as B-mem's core. Shared arm not run (beside, decides nothing). Output `/data/jack-logs/j29/b7mem/b7_read.json`.
state: 11-28 09:26 (Director) all four raw files present, unread. Built `befb7848` (port of `pomdp-baselines@e7c19c32`, MIT), launcher `e0f5ea7e`, launched 11-27 14:26 at pin `4caae6c3`.

### J14 — The ear's row is complete: SCORE IT; brain A's seeds at the published size come home by themselves
part: Ear + His brain (A) · stage: 1 · step: 3 (reproduce) · prior: §2, §7; `prereg/a2_size_s_enc.md` (FROZEN + addendum `381b83b9`); `KNOWN_ANSWERS.md` rows A2-v1S-enc, N7-kws-2; rulings 67, 78, 79, 96, 124
prior: Ear: BC-ResNet-1 published 96.9 ± 0.30 (band 96.3–97.5); ours s0 97.137, s1 97.055. A: A1, A3 PASS, A2 MISS under band; A2-v1S-enc is the arm at the published size, 5.0 T4-h a seed.
**Do.** (1) Ear: `kws2_s2.json` (01:44) and `kws2_twin.json` (03:06) are
in `/data/jack-logs/kws_row2/`. Copy the receipts from worktree
`/data/wt/jack-kws-w48`; `score2` by the frozen rule into
`KNOWN_ANSWERS.md` N7-kws-2 — a miss is a miss, and the twin must fail.
(2) A: `v1s_enc_s10_c.json` is home (05:26); seed 11 runs (kernel
jack-ladder-1795843580, home ≈ 10:30), seed 12 follows from pid 3886689.
When all three are home: copy receipts from `/data/wt/jack-a2enc`, `rows`
score into `KNOWN_ANSWERS.md` A2-v1S-enc by the frozen rule, once.
(3) Note under SYSTEM Next-up (v) that no doc refused these submits.
branches: ear in band on 3/3 and twin fails → the ear's row REPRODUCES; PLAN "Ear" goes to step 4 at the next sitting — INBOX one line. · any other ear read → written as read. · A in band → gate 2 of `stage2_a_vs_b.md` reads met (draft text). · A misses → a miss; J33's row is the next evidence, nothing is re-run. · a seed's submit refused by Kaggle → written; the week is closed; no retry.
serves: M3, M5 · PLAN "brain A reproduced", "Ear" step 3 · standing objective 3 · philosophy 3, 4.
**done when:** N7-kws-2 is scored; A2-v1S-enc is scored on its three seeds.
state: 11-28 09:30 **(b) N7-kws-2 READ — PASS** (`054a7ce1`): s0/s1/s2 97.137/97.055/96.646, mean 96.946 sd 0.263 in 96.3–97.5; twin 8.569% ≤ 15. (a) A's seed 10 (+twin) HOME 05:26 `v1s_enc_s10_c.json`, seed 11 RUNNING (jack-ladder-1795843580), pid 3886689 alive. NEXT: when s10–s12 home → `rows` score into A2-v1S-enc (s10 not yet scored).
state: 11-28 09:26 (Director) timers fired as armed: A seed 10 00:01 → home 05:26, seed 11 launched 05:26; ear chain done 03:06 (`w48_launch.log`). Nothing scored.

### J28 — The three cheap GPU cells ran: HARVEST them
part: Body and movement (cell 1) + His brain (cells 2, 3) · stage: 1 · step: 1 → 3 · prior: §29, §2 (RESEARCH 11-27 16:12: A1 at ratio 64 predicted to MISS at both points, ≈ 420 / ≈ 500); `prereg/a2_ratio64.md` FROZEN `97762429`; rulings 119, 124, 126
**Do.** `/data/jack-logs/j28/w48_launch.log` says the chain ended rc 0 at
08:18 (mjx → fp16 → r64 s20+twin → s21 → s22). Copy receipts from
`/data/wt/jack-j28-w48`. (1) MJX cell: did JAX run on the GPU Kaggle
gave, steps a second, memory → §29 as "ours" (or the error text). (2)
fp16: seconds per 1k decisions, both → §2 as "ours". (3)
`python -m experiments.brain_a.ratio64 score` ONCE → the row as read.
branches: cell 1 ran → INBOX "walkerwalk: JAX runs" (the sitting freezes `w1_walkerwalk.md`); failed → error into §29 + a `RESEARCH:` line "walker venue". · ratio 64 HOLDS → re-cost `stage2_a_vs_b.md`'s table at ratio 64 (draft). · MISSES → written as a miss beside the record's prediction; the table stays; brain A's affordable size is then J33's question.
serves: M5 (body), M3 (brain A affordable on free GPU) · standing objective 3 · philosophy 3, 6.
**done when:** the three results are in their files.
state: 11-28 09:35 **DONE `44e5c6f6`**, all three home. Cell 3 ratio 64 **MISSES** (200K 282.3/275.9/316.1 vs 932; twin 288.2 — no learning seen in ≈ 6k grad steps), `a2_ratio64.md` READ. Cell 1 FAILED: 2×T4 seen by jax 0.11.1, but brax 0.14.2 calls removed `jax.device_put_replicated` → §29 'ours' + RESEARCH line. Cell 2: fp32 165.3 s/1k; 16-mixed crashed (Lightning AMP vs sheeprl grad clipping) → `stage2_a_vs_b.md` item 2.
state: 11-28 09:26 (Director) chain done 08:18, `a1r64_s2{0,1,2}.json` in `/data/jack-logs/brain_a/rows/`; nothing harvested.

### J32 — The pilot stage 2 is sized by: twenty lives-long runs of brain B on his own eye, 1M decisions
part: Learning to eat + His brain (B) · stage: 1 → 2 · step: 4 · prior: §8 (test method: n from a 20-run pilot), §21, §28; `WORLD_CARD.md` § J30; `prereg/stage2_a_vs_b.md` arms table; rulings 135, 138, 141
prior: published — foragers eat at 1e6–3e6 steps (§28); resets keep a replay learner learning (Nikishin 2022; Nauman 2024); no small agent is published still improving at the end of a long life (§20). Ours: B+K1 on the kind eye at 500k EATS 3/3 (27.41 vs 3.34 / 2.36), still climbing at 450k; B reproduces reacher-hard; hand forager 20/20.
**Why.** PLAN stage 2: "the number of runs is set by a 20-run pilot", and
"we should know whether anything learns to eat before the long runs
start". J30 says something does, at 500k on 3 seeds. Nobody knows the
spread over 20 seeds or whether it holds to 1M — and three free sessions
stand empty.
**Ruled (141).** This is a SIZING pilot: it seats nothing and no arm
"wins". Fixed here, not the worker's to move: arm = J30's sighted cell
exactly (B, `ent_coef="auto"`, K1 resets every 50k, `paint="kind"`,
`eye="bins16"`, entry rung d0) with `terminal_obs="dying"`; 1M decisions;
seeds 2001–2020; the blind twin on seeds 2001–2005. Outputs: late meals a
life per seed (lives starting ≥ 500k), the 50k window table, the share of
seeds above twice the blind twins' mean, and from that share the n a race
needs (§8's formula) — plus ONE falsifiable line: **"B+K1 still eats at
1M" holds iff ≥ 15 of 20 seeds end above twice the blind-twin mean**;
fewer is written NOT HELD AT 1M and stage 2's B is in question.
**Do.** (1) Draft `prereg/pilot_b20.md` from the lines above + the prior
(no other choice in it). (2) Timing: J30's cells are the timing cell (same
learner shape) — say kernels and hours. (3) Freeze under ruling 135 (your
INBOX line; this order IS the arms/seeds/bar). (4) Launch self-advancing
chains on the free sessions behind J18's rank, own worktree, `cpu` watcher.
branches: `terminal_obs="dying"` changes the J30 cell's trace beyond the death step (check on one short cell) → say so in the prereg; it stays `dying`. · J18 names K2 as keep before the freeze → still K1 here (J30's cell); K2 is a stage-2 draft matter. · a kernel dies → relaunch that chain once from its last whole carry; a lost seed is entered lost.
serves: M3 (stage 2 sized by evidence), M7 · philosophy 1, 3, 4, 6.
**done when:** `pilot_b20.md` is frozen and chains run with paths in a state line; then the one read.

### J33 — Brain A at the authors' OWN current preset: one known-answer row on W48's spare GPU
part: His brain (A) · stage: 1 · step: 3 (reproduce) · prior: §2 as re-checked 11-27 16:12 (`fde24942`) — read it first; `KNOWN_ANSWERS.md` A2; rulings 86, 96, 124, 139
prior: published — the official `dmc_proprio` preset is now size1m at train ratio 1024, repeat 1; the v2 code re-run independently (MR.Q, 10 seeds) reads cartpole_swingup 736 [591, 838] at 500K. Ours: 1m at ratio 512 is a pairing nobody published — A2 MISSED its (2023, v1) band at 100K, 393.3 vs 438–518. A1, A3 pass.
**Ruled (139).** A2's row stays a MISS. This is a NEW row, `A2-off`.
**Do.** (1) Draft `prereg/a2_official.md`: cartpole_swingup, the official
preset unchanged, band = MR.Q's interval at the frames it was read at
(take the exact frames/interval from §2's source, not from this order),
seeds 30–32, twin (lr 0) on seed 30, no extension, a lost seed is lost.
If one seed to the band's frame count costs more than 3 T4-h, read at the
EARLIEST frame count the source tabulates and say so before any seed.
(2) Timing cell ≤ 15 min (ruling 86) once A's seed 12 has launched or
finished. (3) Freeze (ruling 135, INBOX line), then one chain, seeds
30 → 31 → 32, stopping at Kaggle's refusal.
branches: the source tabulates no interval at an affordable frame count → do not invent a band: `RESEARCH:` line, order waits. · timing says 3 seeds exceed the spare → run what fits in seed order; the row is read only when 3 are home (next week finishes it). · refused → written; no retry.
serves: M3 (PLAN "brain A reproduced") · standing objective 3 · philosophy 3, 4, 6.
**done when:** `a2_official.md` is frozen with its timing and the chain runs or is refused.

### J31 — Stage 2's rule made freezable: memory in the race, his own death, and every OPEN line closed or priced
part: His brain · stage: 1 (ready-list: "the rule of the race written down, sized for the cloud") · step: 4 (pre-registration text; nothing runs) · prior: §8, §9 (battery; memory rows A 0.65 / 0.59 vs B 0.04 / 0.00), §6; rulings 137, 138
**Ruled (137, owner's question 11-28).** Eating food he can see does not
choose his brain on its own. The verdict section of
`prereg/stage2_a_vs_b.md` must say: (a) no brain takes the seat on the
visible-food race alone — the battery's memory and delayed-reward rows are
read BESIDE it in the same verdict, with B-mem beside B; (b) the race's
own world carries ONE memory condition, drafted with its prior and a
control that must fail (a memoryless arm must not pass it by sight).
**Do (text).** (1) Write (a) into the verdict rule. (2) Draft (b): pick
the form the record supports best from §6/§9 (food that goes out of
sight behind a wall; a remembered place across a life; a delay between
cue and meal), with the published number for each brain class, the
must-fail control, and its cost in kernels. No published form fits his
world → a `RESEARCH:` line, and say so. (3) `terminal_obs="dying"` for
every arm, in the draft and in `pilot_a.md` (ruling 138). (4) Fill what
today's reads settle: B's keep setting (J18), `B-mem` (J29), ratio (J28),
gate 2 (J14). (5) List every remaining `OPEN:` with the one cheap row that
answers it. Still a DRAFT — a sitting freezes it.
branches: a read is not home → leave that OPEN, named. · (b) needs a new world option → name the `jack_env.py` option and its default (today's world), build nothing.
serves: M3 · philosophy 3, 4, 7 (the brain he keeps must carry memory and words, not only eat).
**done when:** the draft carries (a), (b), `dying`, and a short OPEN list.

**Lane `forge`** reads `docs/FORGE.md`; its work is J18. Lane `lab`: a
candidate for brain B enters through `docs/FORGE_POOL.md` with its prior;
probes of other seats stay in `LAB.md`. **The hindsight eye (ruling 140):**
bins16 holds the eye's stage-1 seat BY VERDICT (J30); the hindsight eye is
raced against it only when the lab's below-ceiling ruler exists on
`paint="kind"` and has been shown able to FAIL (a yoked twin loses on it,
a random-imprint table does not pass) — ruling 134 stands. That ruler is
the lab's next unit (box-minutes, no learner). Note for it: J30's blind
twin also climbs late (2.6–8.0 meals a life) — the ruler must separate
sight from that.
**Parked or closed:** the MLX port (ruling 113). `D1.0`, `T2.01`, `LF.06`,
`LC.09`, `CR.01`, `LF.04` — "to earn again" on the chosen brain. `NE.01`,
`PS.02`/`.05`/`.06`/`.08`, the place key, the needs-hidden row → stage 3.
`W1.04` → Sunday's sitting. `prereg/eye_geom_500k.md` stays a draft
(ruling 111). `prereg/w1_walkerwalk.md` waits for J28's cell 1 and a
sitting.
**Next up, in order (text or box < 15 CPU-min; do ALL of these before any slot ends idle):**
1. `docs/research/STAGE1_READY.md`: bring every line up to date from
today's reads (pointers only).
2. HandedSense at death (ruling 138): add an option (default = today's
behaviour, so r1L's lock is untouched) under which the handed food channel
on the dying step is the dying body's; planted test beside
`tests/test_death_interface.py`. No run uses it yet.
3. `tests/test_j24k1_resets.py`: add the sys.path line so it runs by path.
4. `WORLD_CARD.md` § J30: one paragraph, BESIDE — what the touch columns
say the blind twin does to eat 3 meals a life (walls? chance contact?).

---

## SYSTEM — work on the machine (max 3 open)

*Order of work: **S20** (closes with J18's read) → **S22** → **S17** → Next-up. A fix a RUNNING
Jack kernel needs goes first (ruling 54).*

### S20 — The harness reads its first look when ten runs are home; round 1-L gets a reader
part: (machine — the judge of stage 1 and 2; card line 4) · stage: 1 · step: —
**Do (ruling 123).** (1) `harness.compare` raises "need 20 runs per arm"
even to take look 1. Change nothing of the test: with ≥ `looks[0]` and
< `looks[1]` runs an arm it computes look 1 exactly as today and returns
A_DIFFERS or FUTILE if look 1 says so, else a new verdict `NEEDS_LOOK_2`
(never NOT_DISTINGUISHED); with ≥ `looks[1]` it behaves bit for bit as
today. `prereg.judge` passes it through. Planted tests: same seeds, 10
runs then 20 → the same look-1 p; a planted look-1 winner stops at 10; a
null pair at 10 says NEEDS_LOOK_2; < 10 still raises. Re-run
`same_vs_same` — its false-alarm figures must not move. (2) A reader for
round r1L (`r1_read` stays locked to r1): the harness verdict for K1 vs
baseline, check #13's books, and beside it rung M's rate per arm under the
frozen metric hash. It refuses a spec whose hash differs from the freeze.
serves: PLAN stage 1 "the harness passes its own checks" · J18 · philosophy 4, 5.
**done when:** the planted tests pass, `same_vs_same` is unchanged, and the reader runs on a planted r1L folder.
state: (1)+(2) built `968da0b8`, in a NEW `harness/looks.py`, not by editing `compare`: `__init__.py`/`prereg.py` are hashed by r1's judge lock and every prereg, so editing them would make those results unreadable. `compare_looks` pads AFTER the real runs and calls the frozen `compare`, so look 1 is the hashed code; hash 382cf304 unchanged, so `same_vs_same` cannot move (not re-run: same code, same hash).
state: Reader `python -m experiments.forge.r1L_read --tranches 1` (JACK runs it when tranche 1 lands, ≈ 01:00); planted 4/4 + 4/4. The spec names no permutation seed: the reader uses n_perm 2000, seed 0, and prints them. Close S20 when JACK's first real read runs.

### S22 — The usage meter stopped every agent for 8.5 hours on a reading that contradicts itself: record it, do not touch the stop
part: (machine — slots lost to budget; card line 5) · stage: — · step: —
**Why.** 11-27 06:41 the gating line read 15% (resets Dec 2 12:00). At
11-28 00:37 it read 100% with the SAME reset; at 09:07 it read 13% with a
reset of Dec 3 09:00 — a week that "reset" four days before the date it
printed. Nine slots, the audit and the 06:37 sitting were refused; four
results sat unread. `usage_attribution`'s METER-NON-MONOTONE only fires
when the percentage falls inside one printed reset window; this fell
across a window that moved.
**Do.** (1) In `scripts/usage_attribution.py` (reporting only): also flag a
reading that falls by > 20 points while the earlier reading's own printed
reset time is still in the future, whatever the new window says; count
the slots refused between. Planted test: the 11-28 sequence fires, an
honest reset at its printed time does not. (2) One paragraph in
`docs/OWNER_UPDATES.md` with the three readings and the slots lost.
**Never** touch the stop (95%, the owner's), the pace switch or `.paused`.
branches: the log shows the 100% was a real cap on the shared account (other projects) → say that with the lines; (1) still lands.
serves: philosophy 5, 6 (the loop's lost hours are counted truthfully).
**done when:** the planted test passes and the owner paragraph is written.

### S17 — A failed Colab submit must not leave an orphan session
part: (machine — the free GPU; standing objective 3) · stage: — · step: —
**Do.** `gpu.run_on_colab`: after a run that failed before a session was
recorded locally (the assign POST timed out at 120 s and the server
assigned the VM anyway — 3 of 5 submits on 11-24), list assignments and
unassign those with no local session (`state.client.unassign`, as done by
hand twice), then retry once. Planted test with a fake client: a timed-out
assign that did assign is unassigned; a live local session is never
touched. Try CLI 0.8.0 in a scratch venv under `/data` only if the
timeout survives; no change to the pinned preamble.
serves: philosophy 5, 6.
**done when:** the planted test passes and the next real Colab submit's log shows no "session busy" refusal after a failed one.
state: built `f0fb4f3c`: after a failed run `run_on_colab` unassigns server VMs with no local session and retries ONCE, only if one was released; a plain job failure is not retried; a "session busy" refusal never reaps (holder may be mid-assign). Planted 6/6 (`python -m tests.test_gpu_colab_reap`); live glue lists the account (0 assignments, none touched). CLI 0.8.0 not tried (no timeout seen since).
state: open until the next real Colab submit after a failed one shows no busy refusal — whoever makes it notes the log line here.
branches: Colab still 503 at the next real submit → leave the code, note the date; no retry loop.

**Next up:** (u) the forge's chain B declares `JACK_WATCHER=gpu` while polling a `jack-cpu-` kernel (INBOX 11-27 07:23): make `r1_run chain` default to `cpu`; running chains untouched. (v) S18: whoever makes the next real ear/A submit notes here that no doc refused it (ruling 125 ratifies the doc-only half; S7(1)’s import closure rules the rest). (w) S16 stays wired (`bd43f84b`, `a426fe45`): whoever sees the first real `SKIPPED-IDLE` line in `ladder.log` notes it here; three jack slots RUN with 5/5 sessions RUNNING and no ORDERS change → read the pre-check's reason and fix that cause. (1) S13 (a):
reproduce `scripts/test_lane_guard.sh`'s FAILs of 11-20 on a clean tree.
(2) **S9** as written 11-20 (`git show 8fde84f5:docs/ORDERS.md`): left
`T0.17`, `T0.23`, then DP.05 HR.5 LT.02 T0.35 T3.09 T4.02 W.2 W1.00.
(3) `XL.00` (the rebirth test) re-buy with a planted positive — stale
80 d, in front of 19 specs. (4) `gpu.py`: torch pin 2.6.0+cu126 in the
Kaggle preamble. (5) **S6** step 3 — the video of the latest recorded
life. (6) Ruling 91: `T0.04` gains the world-state clause; every ledger
row records its mujoco version. (7) **S8**, **S4**, `T0.13`'s dead
clauses, `verify` VENUE 266 vs 256. (8) The 4 kept tmp folders (2.8 GB).

---

## INBOX — one line each, workers add, the Director empties
(emptied 2026-11-28 09:26 — SESSION 09:40 memory-in-the-race → ruling 137, J31; JACK 07:23 /data → eased, 27 GB free; JACK 07:23 watcher label + test path → SYSTEM Next-up (u), JACK Next-up 3; FORGE 14:20 → J18; JACK 14:25 J30 EATS → ruling 140, J32; RESEARCH 16:15 → ruling 139, J33, J28's prior; JACK 16:40 death interface → ruling 138, J31 (3), Next-up 2.)
- RESEARCH: (11-28 09:40 SESSION, owner asked "will he taste") Taste as a sense for Jack: what published embodied learners use a taste or gustatory signal (good vs bad food, spoiled food that sickens later, learned food aversion as in conditioned taste aversion), how it is encoded, what it costs to learn, and whether it helps foraging over sight and touch alone. Note what the Inventor's "remember the colour of what I just ate" already covers. Answer as a PUBLISHED_PRIORS section; mark measured vs inferred.
- QUESTION (JACK, J18): look 1 — the sealed `r1L_read --tranches 1` refuses until the beside arms (K2 ×10, twin ×2, g1-5…g1-7) are home, though the primary (K1 vs E-food, g1-0…g1-4) is whole | options: A wait for g1-5…g1-7 (≈ +1 day) / B unlock S20 to read the primary arms alone | default: A, wait; nothing opened meanwhile
- RESEARCH (JACK, J28 cell 1): walker venue — which jax/brax/playground triple runs Playground PPO on Kaggle's T4 image (brax 0.14.2 + jax 0.11.1 breaks on `device_put_replicated`; a second CUDA PJRT plugin is also registered). §29 'ours'.
- JACK 11-28: ratio-64 Dreamer stayed at the never-trains level on all three seeds (≈ 6k updates) — the raw keeps only first-step losses, so 'world model not fitted' vs 'actor never moved' cannot be told; a sitting may want last-step scalars kept.

## DONE (kept one day)

- **J30** (`e406b65f`): **EATS** — B+K1 resets on the kind eye, late meals a life sighted 27.41 / blind 3.34 / shuffled 2.36, 3/3 seeds above both; same seeds, twins and bar as J24 (which stays NOT SHOWN AT 500k for plain SAC). Rows in `WORLD_CARD.md` § J30, `KNOWN_ANSWERS.md` J30.
- **S21** (`bff35c96`, `f2e7cc58`): `T0.01` judged on the fastest of three fresh-process imports, line unmoved; PASS; its five dependents re-bought.
- SYSTEM Next-up v2 (`59160055`): a Kaggle quota refusal closes the week in `gpu_budget.json`.
