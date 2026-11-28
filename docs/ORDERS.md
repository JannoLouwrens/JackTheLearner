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
eye. Overnight (every agent was stopped by the usage gate 00:37–09:07) four
more results came home; three were read at 09:24–09:26: **B-7 REPRODUCES**
(brain B's memory arm −172.63, in the published band; its memoryless twin
−1490), **the ear's row PASSES** (96.946, twin 8.6%), **ratio 64 MISSES**
(brain A at an eighth of the training: 282 / 276 / 316 against a band of
932 — as the record predicted), and the body's MJX cell and the fp16 cell
did not run (install faults, named). Round 1-L's look 1 is NOT read: its sealed reader waits for the beside
arms, and the seal holds (ruling 142). **First: bring g1-7 forward (J18),
then fill the other Kaggle CPU sessions standing empty (J32).**
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

*Order of work: **J18** (launch chain D for g1-7; the read waits) → **J32** (fill
the empty CPU sessions) → **J33** (GPU, behind A's seed 12) → **J14**
(score when seed 12 is home) → **J31** (text) → Next-up → exit. A slot
waiting on a kernel advances the next order.*

### J18 — Forge round 1-L: look 1 waits for the beside arms — bring g1-7 forward, then read once
part: Learning to eat + The science lab · stage: 1 · step: 4 · prior: §19–§22; `docs/FORGE.md`; rulings 61, 65, 69, 83, 105, 112, 122, 123, 130, 135
prior: rung M (300k, 5 seeds, lock `2eb2c0f4`): K1 resets 0.84, K2 TQC 0.82, baseline 0.059, twin 0.017. Record: resets + LN critic best-supported (Nauman 2024); nobody has published a small agent still improving at the end of a long life (§20). J30 beside: K1 on his own eye still climbing at 450k.
**Frozen (do not reopen):** rule `rounds/r1L/spec.json` `52033df2`; judge
lock `9d3feb7e` (n_perm 2000, seed 0, ruling 130); PRIMARY K1 against
baseline E-food on `outcome_sustained` W = 2, T = 1M, two looks at 10 and
20 runs an arm, 0.025 each; KEEP RULE: if a look says K1 differs, B's keep
setting is whichever of K1/K2 has the higher late-half mean at 1M, closer
than K1's own seed range → K1. A lost run is a crash, never replaced.
**Ruled (142, the worker's QUESTION 09:26 — answer A, and shorten the
wait).** The sealed reader refuses look 1 until the BESIDE arms (K2 ×10,
twin ×2: g1-5, g1-6, g1-7) are home. The seal was set before any raw file
(ruling 130) and five raw files are now home: nobody unlocks or edits the
reader, S20 or the lock. Look 1 waits.
**Do.** (1) Shorten it: g1-7 sits BEHIND g1-5 in chain B. Launch chain D
for g1-7 alone on a free session NOW, exactly as chain C was launched
(`--claim D`, claim = the segment-0 file chain B's running code skips,
`d1361856`; pin `495a766d`; `JACK_WATCHER=cpu`). Not certain B will skip
it → do not launch; a seed run twice is worse than a day's wait. (2) When
g1-5…g1-7 are whole: `python -m experiments.forge.r1L_read --tranches 1`
ONCE (jack-std). (3) Verdict, check #13's books and the beside rates into
`docs/FORGE.md` and the scoreboard.
branches: A_DIFFERS → the keep setting by the frozen rule goes into `stage2_a_vs_b.md` and J32's draft; if K2's runs are not home yet the keep rule waits for them and J32 uses K1 meanwhile (the tie rule's own answer). · FUTILE → written as read; B's stage-2 setting stays K1 BY RECORD + J30; INBOX one line. · NEEDS_LOOK_2 → launch tranche 2 (seeds 1431–1440, K1 + baseline) yourself from pin `495a766d` on free sessions, claim files as chain C's, `JACK_WATCHER=cpu`. · the reader refuses for any OTHER reason (lock, hash) → the refusal text into the state line and INBOX; do not edit the reader or the lock.
serves: M3, M7 ("he keeps learning") · philosophy 1, 3, 4.
**done when:** look 1's verdict is in `FORGE.md`; then tranche 2 runs or the round is closed with the keep setting named.
state: 11-28 17:10 (JACK) chain D brought g1-7 seg 0 home 16:49 (jack-cpu-1795858719 COMPLETE) and pushed seg 1 itself (jack-cpu-1795884569 RUNNING); chain B (g1-5 seg 1, -1795870396) and C (g1-6 seg 1, -1795881767) RUNNING. All three beside arms now on their LAST segment. 11 receipts from worktrees forge-c/-d, b20, a2enc copied into main. Look 1 ≈ 11-29 morning. Reader not run (seal).
state: 11-27 03:35 **JUDGE LOCK `9d3feb7e`** committed alone at `85ca3d4e` before any raw file; both readers `check_lock` first. Rule FROZEN `52033df2` 11-26 20:27; launched 20:30 from pin `495a766d`; chain C 11-27 14:15 with g1-4 + g1-6 claimed (`d1361856`).

### J14 — Brain A at the published size: seeds 10–12 come home by themselves; score them once
part: His brain (A) · stage: 1 · step: 3 (reproduce) · prior: §2; `prereg/a2_size_s_enc.md` (FROZEN + addendum `381b83b9`); `KNOWN_ANSWERS.md` row A2-v1S-enc; rulings 67, 78, 79, 96, 124
prior: A1, A3 PASS, A2 MISS under band; A2-v1S-enc is the arm at the published size, 5.0 T4-h a seed. (The ear half of this order is DONE — see DONE.)
**Do.** `v1s_enc_s10_c.json` is home (05:26); seed 11 runs (kernel
jack-ladder-1795843580, home ≈ 10:30), seed 12 follows from pid 3886689
(log `/data/jack-logs/a2_enc/sat_launch.log`). When all three are home:
copy receipts from `/data/wt/jack-a2enc`, `rows` score into
`KNOWN_ANSWERS.md` A2-v1S-enc by the frozen rule, ONCE.
branches: in band → gate 2 of `stage2_a_vs_b.md` reads met (draft text); INBOX one line. · a miss → a miss, written; J33's row is the next evidence, nothing is re-run. · a submit refused by Kaggle → written; the week is closed; no retry; the row waits for W49.
serves: M3 · PLAN "brain A reproduced" · standing objective 3 · philosophy 3, 4.
**done when:** A2-v1S-enc is scored on its three seeds.
state: 11-28 17:25 (JACK) seeds 10, 11 home; seed 12 (`jack-ladder-1795867241`) still QUEUED on Kaggle's GPU queue since 12:00 (≈ 5.4 h, past the 11-21 precedent's 4.7 h) beside 5 RUNNING CPU sessions — no action, no resubmit (order forbids a retry). Not scored (needs all three).

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
state: 11-28 15:35 (JACK) **FROZEN + LAUNCHED.** `prereg/pilot_b20.md` FROZEN at `01d1857a` (commit alone, 15:35, freeze-intent line unanswered; reader sha256 `983a9d84` unchanged). Test control re-pinned off HEAD first (`63faceac`). Worktree `/data/wt/jack-b20` at `01d1857a`, JACK_WATCHER=cpu, pin `01d1857a904b297eea06e0f55b2f61939afae713`.
state: chain a = units 1–5 (pid 2860178, kernel jack-cpu-1795880127 RUNNING, log `/data/jack-logs/pilot_b20/chains/chain_b20a.log`); chain b = units 6–9 (pid 2860519, jack-cpu-1795880141 RUNNING, `chain_b20b.log`). Out `/data/jack-logs/pilot_b20/chains/`. Draft pin refused by `job` (checked). ≈ 2 segments a unit → a ≈ 3.1 days, b ≈ 2.5 days → read ≈ 12-01. NEXT: a died kernel → relaunch that chain ONCE from its last whole carry; when all 9 units home: `pilot_b20_read --pin 01d1857a… --out /data/jack-logs/pilot_b20/read.json` ONCE (jack-std).

### J33 — Brain A at the authors' OWN current preset: one known-answer row on W48's spare GPU
part: His brain (A) + Body and movement (cell 0) · stage: 1 · step: 3 (reproduce) · prior: §2 as re-checked 11-27 16:12 (`fde24942`) — read it first; `KNOWN_ANSWERS.md` A2; rulings 86, 96, 124, 139
prior: published — the official `dmc_proprio` preset is now size1m at train ratio 1024, repeat 1; the v2 code re-run independently (MR.Q, 10 seeds) reads cartpole_swingup 736 [591, 838] at 500K. Ours: 1m at ratio 512 is a pairing nobody published — A2 MISSED its (2023, v1) band at 100K, 393.3 vs 438–518. A1, A3 pass.
**Ruled (139).** A2's row stays a MISS. This is a NEW row, `A2-off`.
**Do.** (0) The body's cell again — a failure fixed, not filed: J28's
MJX cell died at install (brax 0.14.2 calls `jax.device_put_replicated`,
gone in jax 0.11.1; a second CUDA plugin is also registered). The
researcher has the question (INBOX `RESEARCH` walker venue). When §29
names a jax / brax / playground triple: pin exactly that in the cell's
preamble, planted test that the preamble carries the pin, re-run
`experiments/body/mjx_timing.py` once (10 GPU-min) → three numbers into
§29 as "ours". No answer yet → skip (0), do (1)–(3); do not guess pins on
GPU time. A2-off's kernel also keeps LAST-step losses beside the scores
(J28's raw kept only first-step ones, so its flat seeds cannot say why).
(1) Draft `prereg/a2_official.md`: cartpole_swingup, the official
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
state: 11-28 12:30 (JACK) BRANCH FIRED — no band to read: (1) drafted `prereg/a2_official.md` as DRAFT-BLOCKED. MR.Q's 736 [591, 838] is the 10-seed-MEAN CI at 500K agent steps ONLY, from commit `251910d` whose `dmc_proprio` = size12m / ratio 512 — not today's 1m / 1024 / repeat 1 (no score anywhere). MR.Q's repo ships per-step curves for MR.Q only, not DreamerV3. Cost to the one tabulated point ≈ 12–23 T4-h (251910d) or ≈ 10.7 T4-h (current) a seed, > 3.
state: No seed, timing cell or freeze. INBOX `RESEARCH:` line 12:30. (0) skipped (§29 has no pin triple). Seen: A's seed 11 home 12:00, seed 12 pushed 12:00 (`jack-ladder-1795867241`) — J14's.

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
today's reads settle: B's keep setting (J18), `B-mem` = the B-7 arm
(REPRODUCES), ratio 64 MISSES so the cost table stays at ratio 512,
gate 2 (J14 / J33). (5) List every remaining `OPEN:` with the one cheap row that
answers it. Still a DRAFT — a sitting freezes it.
branches: a read is not home → leave that OPEN, named. · (b) needs a new world option → name the `jack_env.py` option and its default (today's world), build nothing.
serves: M3 · philosophy 3, 4, 7 (the brain he keeps must carry memory and words, not only eat).
**done when:** the draft carries (a), (b), `dying`, and a short OPEN list.
state: 11-28 13:15 (JACK) **DONE** `5b0b1e73` (+ `bdc21d5a`): (2) memory condition = the dark-eye probe — Animal-AI L7 lights-out on the passive T-Maze's one axis (D 8/16/32/64); wall form and remembered-place form rejected by record (POPGym MLP wins navigation; Mirowski weights carry fixed places). Must-fail floor: memoryless B must not beat a keep-heading script at D ≥ 16, else VOID. Option `eye_blackout` NAMED, not built; validation row ≈ 20 kernel-h.
state: (5) the OPEN table at the draft's end: 10 lines, each with its one row; still open by reads not home — B's keep (J18), gate 2 (J14 seed 12). Still a DRAFT; a sitting freezes it.

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
(ruling 111). `prereg/w1_walkerwalk.md` waits for J33's cell 0 and a
sitting. The fp16 cell is closed unrun (Lightning's AMP refuses the
port's gradient clipping): a precision change is a difference from the
published recipe and is not chased before brain A reproduces.
**Next up, in order (text or box < 15 CPU-min; do ALL of these before any slot ends idle):**
0. Copy the receipts still sitting in worktrees `/data/wt/jack-kws-w48`
and `/data/wt/jack-j28-w48` into main's `gpu_submissions.jsonl` (the 09:26
slot ran out before doing it); commit by name.
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

*Order of work: **S20** (closes with J18's read) → **S17** → Next-up. A fix a RUNNING
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
- RESEARCH 16:15 (answered SESSION 11-28 brain A ceilings → PUBLISHED_PRIORS §31): the record says A's two measured, cheap ceilings are the 64-step replay window (R2I: dependencies > 64 steps score ZERO; `brain_a/rows.py:42`) and FIFO replay eviction (Continual-Dreamer: FIFO forgets by 2–4M steps, reservoir fixes; `in_world.py:173`). Change: enter §31's forge lines A-M1 (FIFO + reservoir), A-M2 (TBTT + window 256), A-M3 (Curious Replay), A-M4 (actor-only slow-copy reset) + window-64 control into FORGE_POOL.md for brain A rounds; never a full reset of A's world model; Dynalang/R2I/Director/growth are out of budget or need pretraining.
(emptied 2026-11-28 09:26 — SESSION 09:40 memory-in-the-race → ruling 137, J31; JACK 07:23 /data → eased, 27 GB free; JACK 07:23 watcher label + test path → SYSTEM Next-up (u), JACK Next-up 3; FORGE 14:20 → J18; JACK 14:25 J30 EATS → ruling 140, J32; RESEARCH 16:15 → ruling 139, J33, J28's prior; JACK 16:40 death interface → ruling 138, J31 (3), Next-up 2.)
- RESEARCH: (11-28 09:40 SESSION, owner asked "will he taste") Taste as a sense for Jack: what published embodied learners use a taste or gustatory signal (good vs bad food, spoiled food that sickens later, learned food aversion as in conditioned taste aversion), how it is encoded, what it costs to learn, and whether it helps foraging over sight and touch alone. Note what the Inventor's "remember the colour of what I just ate" already covers. Answer as a PUBLISHED_PRIORS section; mark measured vs inferred.
- RESEARCH (JACK, J28 cell 1): walker venue — which jax/brax/playground triple runs Playground PPO on Kaggle's T4 image (brax 0.14.2 + jax 0.11.1 breaks on `device_put_replicated`; a second CUDA PJRT plugin is also registered). §29 'ours'.
- JACK (J32, ruling 135/127(c)) FREEZE INTENT 12:35: `prereg/pilot_b20.md` freezes from 15:35 UTC unless answered — arms/seeds/bar/twins exactly ruling 141's; the only worker choices are the reader's handling (draft § Reader): lost run declared after one relaunch counts NOT above and leaves the spread; nan late meals counts NOT above; B̄ over blind runs home (none → refuse); n = smallest ≤ 150 with exact power ≥ 80 %.
- RESEARCH (JACK, J33 12:30): A2-off has no readable band — MR.Q's DreamerV3 736 [591, 838] is a 10-seed-mean CI at 500K agent steps only, from `251910d` (dmc_proprio = size12m, ratio 512), not the current 1m/1024/repeat-1 preset; no per-step DreamerV3 curve is released (MRQ repo results/ hold MR.Q's own). Wanted: any published DreamerV3 (official code, any 2024+ preset) cartpole_swingup score WITH an interval or per-seed values at ≤ 200K frames, or a per-seed curve file; else the row needs a re-sized order (≈ 11–23 T4-h a seed to 1M frames). See `prereg/a2_official.md`.
- JACK 13:55: JACK Next-up 0 and 3 are already discharged (both W48 worktrees' receipts are verbatim in main's gpu_submissions.jsonl, 0 missing lines; test_j24k1_resets has its sys.path line since `91066452`, runs under jack-std) — strike them; Next-up 2 done 12:50, 4 done 13:55.
- JACK 14:40: Next-up 1 done — `STAGE1_READY.md` up to date from 11-28's reads. Finding for a sitting: 5a's certificate exists (`docs/research/HARNESS_CHECKS.md`) and reads 10 of 13; #2 (lr-0 copy at planned power), #6 (resume = uninterrupted) and #9 (probes on/off) are OWED since 11-16, and #6 looks payable now on the bit-equal carries J29/r1L already run — no order holds it.
(11-28 09:28: QUESTION J18 → ruling 142, answer A + chain D; JACK ratio-64 losses → J33 keeps last-step scalars.)

## DONE (kept one day)

- **S22** (`b4de7ce0`): `usage_attribution` now flags METER-WINDOW-MOVED (a > 20-point fall while the earlier reading's own reset is still ahead), planted 11-28 shape fires / honest reset does not. Live: 6 such falls in 5 weeks. The 11-27/28 100% was a real cap: the meter climbed steadily 16 → 100 over 11-27 (69% of this week's points not this project), then the account's week restarted 4 days early; 41 slots refused 11-27 20:07 → 11-28 08:37. Owner paragraph written.
- **J29** (`054a7ce1`): **B-7 REPRODUCES** — brain B's memory arm (port of Ni's recurrent SAC) −172.63 on velocity-only Pendulum (seeds −174.14 / −170.45 / −162.05 / −183.88; band −190.9…−154.5); memoryless twin −1490.23 fails as required. `B-mem` is a stage-2 arm.
- **J14 ear** (`054a7ce1`): **N7-kws-2 PASS** — mean 96.946, sd 0.263 (s2 96.646), published 96.9 ± 0.30; twin 8.569%. The ear's architecture reproduces; his own ear (no labels) is stage 3.
- **J28** (`44e5c6f6`): ratio 64 **MISSES** (200K: 282.3 / 275.9 / 316.1 vs band 932; twin 288.2) — predicted by the record; MJX cell failed at install (→ J33 cell 0); fp16 crashed (closed unrun); fp32 165.3 s per 1k decisions.
- **J30** (`e406b65f`): **EATS** — B+K1 resets on the kind eye, late meals a life sighted 27.41 / blind 3.34 / shuffled 2.36, 3/3 seeds above both; same seeds, twins and bar as J24 (which stays NOT SHOWN AT 500k for plain SAC). Rows in `WORLD_CARD.md` § J30, `KNOWN_ANSWERS.md` J30.
- **S21** (`bff35c96`, `f2e7cc58`): `T0.01` judged on the fastest of three fresh-process imports, line unmoved; PASS; its five dependents re-bought.
- SYSTEM Next-up v2 (`59160055`): a Kaggle quota refusal closes the week in `gpu_budget.json`.
