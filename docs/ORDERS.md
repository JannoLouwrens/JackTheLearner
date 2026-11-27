# ORDERS — what the workers do. Written by the Director.
Director sitting: 2026-11-26 19:14 UTC

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5, **each ≤ 4 lines of text — replace,
do not append; history is in git**). One state+journal commit per slot.
Machine problems go under INBOX, one line each. Rulings 121–128 cited here
are in `docs/DECISIONS_RESOLVED.md` under 2026-11-26 19:14; 111–120 under
11-26 06:40.

**`docs/PLAN.md` binds every order (owner, 11-15).** Research → combine →
reproduce → test the last 10% → put it in him. We are in **stage 1: the
instrument.** `docs/research/PUBLISHED_PRIORS.md` is the reference — read
your order's section (only that section) at the start of every slot.
**Where stage 1 stands (11-26 evening).** The world has its second paint
(one look per KIND, `paint="kind"`), the 16-bin eye is built on it, and a
hand-written forager eats from those bins alone (20/20 seeds, 25.5 meals a
life; shuffled control 0/20). Both questions that held the lane since 10:14
are RULED below: **J24 freezes and launches its nine cells now (ruling
121); J18 freezes round 1-L on the harness's own two-look design and
launches its first ten seeds (ruling 122).** Nothing in this file waits on
the Director.
**Never idle on a question (ruling 127).** (a) Every order carries
`branches:` — what to do on each likely outcome; do it, do not ask. (b) A
missed gate stops THAT ORDER's next step, never the lane: write the miss,
take the next order, then Next-up. (c) A question that only chooses between
options an order already sized, and moves no pass line, clause, control or
exemption (ruling 114 stands), is written to INBOX with YOUR choice first;
if nothing answers in 3 hours, your choice is the ruling — act on it and
say so in the state line. Write it in the shape the owner's answering poll
reads (`scripts/answer.sh`, being installed by the owner's session):
`- QUESTION (lane, order): <decision> | options: <A / B> | default: <what
I do if unanswered>` — then carry on with other work. (d) A slot may end "nothing to do" only when
every order is waiting on a running kernel AND Next-up is empty; say which
kernels.
**A line is not the worker's to move (ruling 114).** After a read has
failed, nobody but a Director sitting changes a pass line, a clause, a
control or an exemption.
**Usage.** The shared weekly meter reads 9% (resets Wed 12-02 12:00 UTC).
Every multi-segment job stays a SELF-ADVANCING CHAIN (ruling 109); raw
results land where the next slot can read them and the state line says
where.
**One world (ruling 76): mujoco 3.17.0.** Every row about his world runs in
`/data/venvs/jack-std` (box) or a kernel pinned to 3.17.0. Learner tests
(`sbx`, `optax`) run under jack-std, not the project venv.
**Every new run in his world passes** `eye_frame="body"`,
`motor_frame="body"`, `food_floor_only=True`, `lazy_contact_audio=True`,
`native_substep=True`; brain B uses `ent_coef="auto"` (ruling 64); "eats"
reads the LATE half of the run (ruling 65). An eye row is read by eating —
late meals a life against its own blind and shuffled twins, seed by seed;
critic rulers are printed beside it and decide nothing (ruling 102).
Learner runs in his world only through J24 and the forge's frozen rounds.
**Stopped (rulings 10, 20, 23, 36, 59, 80, 92, 101, 107, 111, 113):**
home-made learners; forager cells on the 16-ray compass eye; world-rule
races; `D1.0`, `T2.01`, smell runs; the needs-hidden row; ANY learner seed
on the per-geom paint; a learner residual on the hand reflex; the MLX
port; new specs with no published prior. A question with no prior is a
`RESEARCH:` line in INBOX.
**Kaggle CPU: 5 sessions IN ALL (ruling 85).** Rank: (0) any kernel
< 1 h; (1) J24's three seed chains; (2) J18 round 1-L — the other two now,
and every session J24 frees. Launch from a clean worktree ff'd to a PUSHED
commit (its own worktree per job, `JACK_REPO=`); copy its receipts into
main's `gpu_submissions.jsonl` on harvest; do not remove a worktree while a
watcher lives. Every JACK slot's first act: `/data/venvs/kaggle/bin/kaggle
kernels status` on each running slug — harvest what is home, refill what
is free.
**GPU (rulings 96, 124).** The Kaggle account is shared: OUR meter counts
only our jobs, and **Kaggle's refusal is the truth** (W47 closed full on
11-26 with our meter at 28.38 h). W48 opens Sat 11-28 00:00 UTC with 30 h;
plan 22 of them: A's seeds 10–12 (15 h) → ear s2 + twin (3 h) → J28's
cells (4 h). A refusal ends that week's GPU plan; nothing retries in a
loop. Free Colab T4 answers 503 since 11-25 — one try per job, no loop.
No first launch of a GPU job or of a NEW learner shape on a CPU kernel
without a timing cell (rulings 86, 99).
**The box (rulings 32, 53).** Day ceiling 57,600 CPU-s, ≤ 2 computing
detached jobs. < 15 CPU-min → box, nice 19. The box is shared (never touch
other projects). `pgrep -af` the same command line before launching.
**Disks:** `/` 5.1 GB free (workers refuse under 3); `/data` 15 GB free —
**it fell 8 GB in a day.** Scratch, clones, caches under `/data` only;
the slot that harvests a round deletes its worktree's caches (not raw
results) once the receipts are copied.
**World files** (`w0.py`, `needs.py`, `survival.py`, `drives.py`): changed
only through `jack_env.py` options whose default is today's world.
**Known-answer rows (rulings 33, 70, 94):** seeds AND at most one extension
stated BEFORE the first seed. A miss is a miss.
**Receipts:** the slot that launches a kernel commits
`experiments/gpu_submissions.jsonl`, `gpu_budget.json` and `cpu_budget.json`
by name.

---

## JACK — build the creature, his world and his curriculum (max 7 open, ranked)

*Order of work: **J24** (freeze + launch, three sessions) → **J18** (freeze
+ launch tranche 1 on the other two) → **J14** (harvest when the Saturday
timers fire) → **J28** (text and prereg now; cells queue behind the ear on
W48) → Next-up → exit. A slot waiting on a kernel advances the next order.*

### J24 — Does he eat from an eye that sees KINDS? Freeze the rule, launch the nine cells
part: Learning to eat + Eye + His world · stage: 1 · step: 4 (test the last 10%) · prior: §28 (read it first), §27, §12; `WORLD_CARD.md` § J17 hunt + § J20; rulings 101–104, 111, 121
prior: published — 0 of 12 foragers' rays return a per-object code; all return a CLASS (rllab Gather closeness arrays; ML-Agents one-hot tag; Safety Gym 16 bins per class). Even so: FoodCollector reward 10 at 2M, SAC PointGoal1 27.0 at 3e6, a TD3 navigator 13% at 100k.
prior: ours beside theirs — class per ray now ours too (`paint="kind"`); theirs 1e6–3e6 steps, ours 500k with one stated extension to 1.5M. Reproduced: B yes (reacher-hard 933.8, J23). Learnable by a plain rule: hand forager on the 16 food bins 20/20, mean 25.5 meals a life, shuffled control 0/20 (11-26). Toy: kind look read at step 250 on 45/45 runs.
**Ruled (121).** The toy row stands as written: K HOLDS, **C MISSED** — the
old scalar code is NOT unreadable in the toy (72–85% recall by 200k, about
800× slower than the kind look). That miss is a wrong prediction about the
OLD paint; it is not a condition on the new one. Step (3) proceeds on K +
the forager. Write one sentence at the top of `WORLD_CARD.md` § J17 hunt:
"rows since 11-14 read NOT SHOWN ON THIS PAINT; the toy shows the old code
is slow to read, not impossible — whether a brain could read it at 500k+
is untested" (this replaces Next-up item 3's wording).
**Do.** (3) Commit the freeze of `docs/research/prereg/eye_bins16.md`
exactly as drafted: brain B LIBRARY arm, flat MLP, J22 pilot settings;
cells sighted / blind / shuffled-bins; seeds 941–943; 500k decisions;
score = late meals a life. **EATS = sighted above both twins on 3/3 seeds
AND sighted mean ≥ 2 × the higher twin mean.** Extension, stated: above
both on ≥ 2/3 but not EATS → the same nine cells carried to 1.5M. Else
**NOT SHOWN AT 500k — a row, not a door.** (5) Push, then
`launch_detached.sh … python -m experiments.j24_launch chain --pin <freeze sha>`
— three chains, one seed's three cells a kernel. Read with
`j17_late_meals`; `j17_onset` beside.
branches: EATS → write the row in `WORLD_CARD.md` and `KNOWN_ANSWERS.md`, copy the bar into `pilot_a.md`, INBOX one line (the lab's hindsight eye is raced next sitting). · ≥ 2/3 above both, not EATS → launch the 1.5M carry yourself from the same pin, no question. · else → write NOT SHOWN AT 500k with the nine numbers; nothing more is launched; the sitting rules the per-ray one-hot arm. · a kernel dies → relaunch that seed's chain once from its last whole carry; twice → INBOX with the log path, the other seeds run on.
serves: M3 (the earliest unfinished milestone) · PLAN "Learning to eat", "Eye" · philosophy 1, 3, 4.
**done when:** `eye_bins16.md` is frozen, the nine cells run as chains with raw-file paths in a state line; then the read by the frozen rule.
state: (0)–(2) DONE 11-26: paint proof PASS `366486d3`; toy K HOLDS / C MISSED (ruled 121); forager PASS 20/20 (mean 25.5, shuffled 0/20). (4) timing PASS: slowest 26.48/s, 500k ≈ 5.2 h = 1 segment (`/data/jack-logs/j24/timing_kernel.json`).
state: (3) FROZEN 19:20 `03e6285d` (exactly as drafted, ruling 121); WORLD_CARD § J17 hunt read-first sentence written.
state: (5) LAUNCHED 19:23–19:24 from pin `03e6285d…` — three chains (pids 164441 / 164667 / 165170, JACK_WATCHER=gpu), Kaggle CPU kernels jack-cpu-1795721024 (s941) / -1795721034 (s942) / -1795721080 (s943), all RUNNING 19:26. Logs `/data/jack-logs/j24/chains/chain_s94{1,2,3}.log`; segments → `chains/segments/`, raw → `chains/j24_s94{1,2,3}.json`.
state: **DONE 11-27 02:45 `2758215c` — NOT SHOWN AT 500k.** Nine cells whole (pin `03e6285d`), `python -m experiments.j24_read` → `/data/jack-logs/j24/read.json`: late meals sighted 0.23/0.31/0.14 (mean 0.229), blind 0.73/1.01/0.92 (0.887), shuffled 2.46/2.30/2.26 (2.338); above both 0/3. Row in `WORLD_CARD.md` § J24. Nothing more launched (branch).
state: beside (decides nothing): all three sighted cells eat early then unlearn (s941 4.89 → 8.78 → 1.93 → 0.11 meals/life by 50k window; s942 6.78 → … → 0.05); twins flat. Same shape as no-reset SAC in forge rung M (collapse 4/5; K1 resets / K2 TQC 0/5). For the sitting: per-ray one-hot arm (branch), and/or the same nine cells on round 1-L's winning learner, pre-registered fresh.

### J18 — Forge round 1-L: who still learns at 1M — on the harness's own two looks
part: Learning to eat + The science lab · stage: 1 · step: 4 · prior: §19–§22; `docs/FORGE.md`; rulings 61, 65, 69, 83, 105, 112, 122, 123
prior: rung M (300k, 5 seeds, lock `2eb2c0f4`): K1 resets 0.84, K2 TQC 0.82, baseline 0.059, twin 0.017. Record: resets + LN critic best-supported (Nauman 2024); TQC; nobody has published a small agent still improving at the end of a long life (§20). Both arms reproduce in the library; rung M shows the task learnable.
**Ruled (122).** The judge is NOT bent to ten runs. The harness's design is
two looks, 10 then 20 runs an arm, 0.025 each; round 1-L pre-registers
**20 seeds an arm (1421–1440)** and runs them in two tranches.
**Do.** Freeze `experiments/forge/rounds/r1L/spec.json` before the first
seed (round 1's files untouched):
- PRIMARY: **K1 against baseline E-food**, `harness.compare`, its own
  outcome (decisions until W sustained), unchanged. Tranche 1 = seeds
  1421–1430 of both arms, 1M decisions.
- BESIDE IT, deciding nothing by test: **K2** on seeds 1421–1430 and the
  **never-trains twin** on 1421–1422; for every run rung M's late-half
  rate (`metric.py`, the lock) and the long-life reading (rate in the last
  200k against 300k–500k).
- KEEP RULE, frozen now: if look 1 or 2 says K1 differs from baseline, B's
  keep setting for stage 2 is whichever of K1/K2 has the higher late-half
  mean at 1M; closer than K1's own seed range → K1 (the record's).
- Budget, its own: tranche 1 = 32 runs = 8 kernels ≈ 140 of 180 kernel-h.
  Tranche 2 (seeds 1431–1440, K1 + baseline, 180 kernel-h) only if look 1
  is neither A_DIFFERS nor FUTILE.
`validate_spec` (the forge's own file, not the judge): teach it a
continuation round — rung L only, `parent: r1` with the parent's lock and
control source named — with planted tests (a round with no parent and no S
rung still refuses). You may do that; you may NOT edit
`experiments/harness/` or `r1_read` (S20 does). The launch does not wait
for S20; the read does. Launch from ONE pushed pin as self-advancing
chains on the two free sessions, K1 + baseline kernels first.
branches: look 1 A_DIFFERS or FUTILE → the round is read; write the scoreboard and the keep setting into `stage2_a_vs_b.md` (draft). · needs look 2 → launch tranche 2 yourself. · a run lost → it is entered as a crash, never replaced (check #13). · S20 not landed when the runs are home → rates are printed, the verdict waits for S20; take the next order.
serves: M3, M7 ("he keeps learning") · philosophy 1, 3, 4.
**done when:** `r1L/spec.json` is frozen and tranche 1's chains run with log paths in a state line; then the harness verdict at look 1.
state: 11-27 04:40 tranche 1 segment 0 HOME both chains (g1-0 03:46, g1-1 03:49 → `r1L/raw/segments/L_g1-{0,1}_seg0.json`; all 4 runs a kernel cut + carried, none lost); segment 1 kernels jack-cpu-1795751166 (A) / -1795751359 (B) RUNNING per Kaggle 04:40; wrap pids alive. Segments not opened (judge reads only whole runs). NEXT unchanged: g1-0…g1-4 whole → `r1L_read --tranches 1` once.
state: 11-27 03:35 **JUDGE LOCK `9d3feb7e`** (ruling 130), committed alone at `85ca3d4e` before any raw file: harness incl. looks.py (__init__ 382cf304 unchanged), registry, forge __init__/metric, spec 52033df2, r1L_read, r1L_beside, continuation, r1_read. Both readers now `check_lock` before opening raw (tests 4/4, 10/10, looks_fa 6/6). Kernels A/B RUNNING (Kaggle status 03:40, 7h08m; wrap pids 237214/237409 alive), no raw file home.
state: 11-26 20:12 per-arm tranche seeds in `r1_run` (`e99c1b78`; rule `f618bba5` in `forge/continuation.py`, out of judge-locked `forge/__init__.py`). 11-26 20:27 **(2) FROZEN `52033df2`** `rounds/r1L/spec.json`: parent r1 lock 2eb2c0f4, K1+K2, tranches exactly `_r1l_like()`, PRIMARY `harness.compare(K1, E-food)` on `outcome_sustained` **W = 2** (E2c's frozen W, unchanged) T = 1M, K1 resets 50k…950k (19), keep rule + 180 kernel-h a tranche. `forge_scoreboard` renders a parent round via `validate_round` (board re-rendered). Tests: r1_run (new frozen-shape test), r1_spec 8/8, round 27/27, read 13/13.
state: 11-26 12:30 RUNNER READY FOR ANY ROUND (commit below): `r1_run --round <name>` (`set_round`) moves rule, raw folder `/data/jack-logs/forge/<round>/raw`, cell files, kernel summary, pin check and child `cell --round`; refuses a missing/mis-named round; r1 paths unchanged. Tests 30/30 under jack-std (2 new). `r1_read` NOT touched (judge-locked, hard-codes r1) — reading 1-L needs a reader decision with the ruling.
state: (3) LAUNCHED 11-26 20:30 tranche 1 from pin `495a766d` (rule `52033df2`), beside J24's three (order of work: "the other two"): chain A g1-0→g1-2→g1-4→g1-6 (pid 237214, kernel jack-cpu-1795725017), chain B g1-1→g1-3→g1-5→g1-7 (pid 237409, jack-cpu-1795725031), both RUNNING 20:31; JACK_WATCHER=gpu. Logs `/data/jack-logs/forge/r1L/chains/chain_{A,B}.log`, raw → `/data/jack-logs/forge/r1L/raw/`. A kernel ≈ 7.5–18.3 h (≤ 3 segments); K1+baseline (g1-0…g1-4) home first ≈ 3 kernels deep a chain. NEXT: watch chains (a lost run = a crash, never replaced); when g1-0…g1-4 are home → look 1 needs a 1-L reader (S20; `r1_read` locked) — print rates meanwhile.

### J14 — Brain A at the published size, and the ear's last seed
part: His brain (A) + Ear · stage: 1 · step: 3 (reproduce) · prior: §2, §7; `prereg/a2_size_s_enc.md` (FROZEN + timing addendum `381b83b9`); `KNOWN_ANSWERS.md` rows A1–A3, A2-v1S, N7-kws-2; rulings 67, 78, 79, 86, 96, 103, 106, 117
prior: A1 PASS, A3 PASS, A2 MISS under band. A2-v1S-enc: 5.0 T4-h a seed. Ear: BC-ResNet-1 published 96.9 ± 0.30; N7-kws-2 s0 97.137, s1 97.055 (band 96.3–97.5), s2 and the twin (control) outstanding.
**Do.** (b) **Ear s2 goes on KAGGLE GPU now** (ruling 117): W47 has 1.62 h
that expire Sat 00:00; a seed is 1.38 h (≤ 1.54 h with valid). The "20% to
spare" in the old order was about one Colab session's length, not the
weekly quota — it does not apply. Stop the Colab retry loop first (`kill`
its pid by number, never `pkill -f`), so there is no double submit. A
quota refusal or a cut run costs nothing: then s2 + the twin run on W48
after A's chain is launched. When s2 and the twin are home: `score2` by
the frozen rule into `KNOWN_ANSWERS.md` N7-kws-2 — a miss is a miss.
(a) **A's seeds 10–12: arm the launcher TODAY.** A detached script (a
poller, not a computing job; `JACK_WATCHER=gpu`; chdir + sys.path pinned)
that sleeps until Sat 11-28 00:01 UTC, ff's `/data/wt/jack-clean` to
origin, and launches ONE self-advancing chain: seed 10 → 11 → 12, one a
session (`rows.py` A2-v1S-enc). Dry-run its launch line today with the
date faked; log `/data/jack-logs/a2_enc/sat_launch.log`. No worker need be
awake on Saturday.
serves: M3, M5 · PLAN "brain A reproduced"; "Ear" step 3 · standing objective 3 · philosophy 3, 4, 6.
**done when:** s2 is launched on Kaggle or refused with the reason; the Saturday launcher is alive with its dry-run in a state line; later, N7-kws-2 scored and A's three seeds home.
state: (a) ARMED 11-26 10:12 (`4cd8fff2`): `scripts/a2_enc_sat_launch.py` pid 3886689 (JACK_WATCHER=gpu) sleeps to Sat 11-28 00:01 UTC, makes its OWN worktree `/data/wt/jack-a2enc` at origin/main (jack-clean has the forge's receipt dirt), then `rows enc-kernel 10 --control` → 11 → 12, one Kaggle GPU session each, stops on a refusal. New: twin on seed 10 per the prereg (`control_seed`; A2 rows keep 0), test PASS. Dry-run (date faked) `/data/jack-logs/a2_enc/sat_launch_dryrun.log`; live log `sat_launch.log`. Harvest: `rows` files `v1s_enc_s1{0,1,2}*.json` in `/data/jack-logs/brain_a/rows/`, copy receipts from the worktree.
state: (b) REFUSED 11-26 07:26 by Kaggle (W47 quota full; meter said 1.62 h; log `/data/jack-logs/kws_row2/kaggle_s2.log`); Colab retry pid 3514759 killed. s0 97.137, s1 97.055 home. ARMED 17:24 (commit below): `scripts/kws_w48_launch.py` pid 29379 (JACK_WATCHER=gpu, log `/data/jack-logs/kws_row2/w48_launch.log`) fires `kws_row kaggle2 s2 twin` from its own worktree `/data/wt/jack-kws-w48` 15 min after A's seed-10 LAUNCH line in `sat_launch.log`, or Sat 02:01 UTC at the latest; test 5/5, dry-run `w48_launch_dryrun.log`. NEXT: harvest `kws2_s2.json`/`kws2_twin.json` in `/data/jack-logs/kws_row2/`, copy the worktree's receipts, `score2` by the frozen rule into KNOWN_ANSWERS N7-kws-2.
state: (c) DONE 11-25: `pilot_a.md` carries J24's arms/bar, 0.87 struck. Still a draft.
branches: a Saturday timer's submit is refused → write the refusal; the other timer still fires; nothing retries (ruling 124). · s2 + twin home → `score2` by the frozen rule, a miss is a miss. · A's seed home → `rows` score into KNOWN_ANSWERS A2-v1S-enc; the chain goes on by itself.

### J28 — Spend W48's spare GPU on three cheap cells the record left open
part: Body and movement (cell 1) + His brain (cells 2, 3) · stage: 1 · step: 1 → 3 · prior: §29 (walkers; RESEARCHER 11-26 16:14), §1–§2, `prereg/stage2_a_vs_b.md` "OPEN"; rulings 119, 124, 126
prior: published walkers — 150–200M PPO steps on 8,192 copies with PD joint targets, ≈ 1–2 h a seed on a T4 (inferred from one T4 figure); JAX fails on a P100 convolution (jax#33224); MJX has no PGS solver. Dreamer at training ratio 64 on proprio: no published number at our size — ours to find out; A2 at ratio 512 MISSED its band, A1/A3 passed.
**Do.** Text and prereg today, cells behind the ear on W48 (one launcher
that waits for `kws2_twin.json` or Sun 11-29 00:01 UTC, modelled on
`scripts/kws_w48_launch.py`, own worktree, `JACK_WATCHER=gpu`, dry-run
with the date faked):
1. **MJX timing cell, 10 minutes:** on the GPU Kaggle gives (ask T4; print
   what came), Playground `HumanoidStand`, PPO, 8,192 copies — does JAX
   run, steps a second, memory. No verdict; three numbers into
   `PUBLISHED_PRIORS.md` §29 as "ours".
2. **fp16 timing cell, 20 minutes:** brain A size S with mixed precision
   on/off — seconds per 1k decisions, both.
3. **Ratio-64 row:** freeze `prereg/a2_ratio64.md` BEFORE the run — task
   and band are A1's (the row A already passes), ratio 64 instead of 512,
   seeds 20–22, no extension; "holds" = in A1's band on 3/3. ≈ 1 T4-h a
   seed.
No walker is built: his GPU body is a new world (ruling 76) and a stage-3
design; PD joint targets hold that seat BY RECORD (ruling 126).
branches: cell 1 fails on the GPU given → the error text into §29, RESEARCH line "walker venue"; cells 2–3 still run. · quota refused → stop, write it (ruling 124). · ratio 64 holds → `stage2_a_vs_b.md` table re-costed at ratio 64 (draft); misses → written as a miss, the table stays.
serves: M5 (body), M3 (brain A affordable on free GPU) · standing objective 3 · philosophy 3, 6.
**done when:** the launcher is alive with its dry-run in a state line and `a2_ratio64.md` is frozen; later the three results are in their files.
state: 11-26 21:2x cell 3 prereg **FROZEN `97762429`** `prereg/a2_ratio64.md`: A1 cartpole_balance, size 1m, replay_ratio 0.0625 (= 64), seeds 20–22, twin 20, no extension, HOLDS = 3/3 seeds in A1's band at 100K AND 200K, read on ruling 78's [P−30K, P) (rows.py window beside). Debt said before any seed: A1 at 512 had s2 677 < 690 at 100K → the 100K point cannot separate ratio from noise; 200K can.
state: arm + kernel + judge built `adae3ce3` (`experiments/brain_a/ratio64.py`, rows.py untouched): `python -m experiments.brain_a.ratio64 kernel 20 --control` / `kernel 21` / `kernel 22`, `… score`. Tests 5/5. Nothing submitted.
state: 11-26 22:40 cells 1+2 BUILT: `experiments/body/mjx_timing.py` (`2d28314a`, Playground's own HumanoidStand PPO config, 10M steps, GPU kind printed first) and `experiments/brain_a/fp16_timing.py` (`07217ae4`, A2-v1S-enc s10, only `fabric.precision` 32-true→16-mixed, sec/1k decisions). Launcher `scripts/j28_w48_launch.py` (`a52398a9`): twin home or Sun 00:01 → mjx → fp16 → r64 s20+twin → s21 → s22, refusal stops (not cell 1). Tests 3+1+2 PASS; dry-run `/data/jack-logs/j28/w48_launch_dryrun.log`.
state: **ARMED 11-27 02:39** (J24's chains ended, cap freed): `scripts/j28_w48_launch.py` pid 688795 (wrap 688785, JACK_WATCHER=gpu), log `/data/jack-logs/j28/w48_launch.log` — opens Sat 11-28 00:01, waits for `kws2_twin.json` or Sun 11-29 00:01, then mjx → fp16 → r64 s20+twin → s21 → s22. NEXT: harvest the three results into §29 / `a2_ratio64` score.
state: 11-27 04:25 cell 1 FIXED `54e0a7dc` (RESEARCH 04:11, re-checked on the 0.2.0 wheel): `mujoco_playground` is a PyPI 404 → installs `playground==0.2.0`; env forced `config_overrides={"impl": "jax"}` (0.2.0 humanoid defaults to warp); num_envs = config's 2,048, not 8,192. Prints VERSIONS + IMPL. Planted test fails on old body; launcher pid 688795 alive, ff's to origin at fire → picks it up.

**Lane `forge`** reads `docs/FORGE.md`; its work is J18. Lane `lab`: a
candidate for brain B enters through `docs/FORGE_POOL.md` with its prior;
probes of other seats stay in `LAB.md`. The lab's hindsight eye (rulings
84, 98, 107): ENTERED, waits for J24's read. Lab's next unit, box-minutes,
no learner: on `paint="kind"`, does his own live table find WHICH kind
feeds him (first imprint), same judge worlds and kill bars as 11-22. Then: enter
the scout's N2 (resets WITHOUT layer norm, arXiv 2610.42100) in
`FORGE_POOL.md` with its prior, for round 2 (ruling 129).
**Parked or closed:** the MLX port (ruling 113). `D1.0`, `T2.01`, `LF.06`,
`LC.09`, `CR.01`, `LF.04` — "to earn again" on the chosen brain. `NE.01`,
`PS.02`/`.05`/`.06`/`.08`, the place key, the needs-hidden row → stage 3.
`W1.04` → Sunday's sitting. Any learner seed on the per-geom paint stays
stopped (ruling 111) until a sitting orders one.
**Next up, in order (text or box < 15 CPU-min; do ALL of these before any slot ends idle):**
1. J26 follow-up, from the existing log only: the four held-out R² again
on the looks where at least one ray hits food (18–23% had none). Marked
POST-HOC in `WORLD_CARD.md`; it changes no line and no verdict.
2. `KNOWN_ANSWERS.md`: draft row B-7 (velocity-only Pendulum, recurrent
SAC, Ni 2021) — task, steps, band WITH its source page/figure; if the band
can only be read off a figure say so and add a `RESEARCH:` INBOX line.
Draft only; nothing run.
3. Draft (text, NOT frozen, nothing run) `prereg/eye_geom_500k.md`: the
old paint's sighted cell on seeds 941–943 at 500k beside J24's twins —
the row that would test "slow, not impossible". A sitting decides.
4. `scripts/` Mac first-day files: one line at the top "PARKED 11-26,
ruling 113" (if not done).
5. `/data` fell 8 GB in a day: `du -sh /data/* /data/jack-logs/* | sort -h
| tail -15` into one INBOX line; delete nothing but pip/wheel caches in
worktrees whose watcher is dead.

---

## SYSTEM — work on the machine (max 3 open)

*Order of work: **S20** → **S17** → Next-up. A fix a RUNNING Jack kernel
needs goes first (ruling 54).*

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

**Next up:** (v) S18: whoever makes the next real ear/A submit notes here that no doc refused it (ruling 125 ratifies the doc-only half; S7(1)’s import closure rules the rest). (v2) `budget.afford`: a Kaggle quota refusal marks that week FULL in `gpu_budget.json` (`closed_by: kaggle`), so no later job is planned into it (ruling 124); planted test. (w) S16 stays wired (`bd43f84b`, `a426fe45`): whoever sees the first real `SKIPPED-IDLE` line in `ladder.log` notes it here; three jack slots RUN with 5/5 sessions RUNNING and no ORDERS change → read the pre-check's reason and fix that cause. (0) `tests/test_gpu_accelerator.py` FAILs `tpu_own_weekly_line`
on HEAD: fix by cause or name why the test is wrong. (1) S13 (a):
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
(emptied 2026-11-26 19:14 — 10:14 → ruling 122/123, J18, S20; 10:45 + 11:40 → ruling 121, J24; 07:30 → ruling 124; 16:14 → ruling 126, J28; 15:10 → ruling 125; 19:15 SESSION → rulings 127, 128.)
- LAB 23:35 (own_kind, rule `07d2e7b7`): on `paint="kind"` his own table finds food (live 73 meals vs babble 12, first imprint on food 7/8) but is DEAD BY ITS RULE — the yoked twin eats 57 (bars: ahead 6/8 and 2x; got 3/8, 1.3x). With five kinds, random imprint + unlearning converges and both hit the regrow ceiling: 11-22's ruler cannot fail on this eye. If the hindsight eye is raced on bins16 after J24, it needs a no-unlearning twin and a below-ceiling measure (first meal → fifth). Beside, not read: a carried table eats from birth 74 vs told 83 (first meal ~decision 54 vs ~943), one world of 8 lost to a wrong table. K9 + C4 (scout N2) are in `FORGE_POOL.md` for round 2.
- RESEARCH 04:11 (answers Director 19:14, W-1's band; §30): **the paper's number is 795.5 at 88.47M steps** (2502.08844 Fig. 9 p. 14, vector-read; seeds 626–851; still 99 at 59M — a late jump); the 115 is SLM-Lab's own PPO, which never learned. But NOBODY outside the authors reproduces it: Playground's own `train_jax_ppo.py` gets ~50 at 105M (issue #171, won't-fix; the paper used Brax `learner.py`), and that is the path our cell uses. Change: (1) **J28 cell 1 as armed dies at install** — `mujoco_playground` is not a PyPI name (404, checked); needs `playground==0.2.0` (the last JAX release; 1.x is Warp/torch) AND `registry.load(name, config_overrides={"impl": "jax"})` (0.2.0 defaults to warp, which a P100 cannot run) — fix `experiments/body/mjx_timing.py:34,42` before the launcher fires (Sat/Sun); config copies are 2,048, not 8,192. (2) Do not freeze W-1 on HumanoidStand: band `WalkerWalk` PPO instead (paper ~968, SLM-Lab 945 — two programs agree; every seed ≥ 900), humanoid run beside it unbanded.
- JACK 00:15 (Next-up 5): **/data at 14 GB free, under the 15 GB floor.** Largest: stt 43G, starwright 31G, jack_corpora 15G, jack-data 12G (kit 8.2, cmu 3.1), recorder 11G, kiln 9.4G, caches 7.1G (huggingface 4.6, align 1.6), venvs 5.3G; /data/wt only 374M, jack-logs < 1G a dir. Files > 50 MB written in the last 24 h: /data/llm/models Qwen3-4B (3.3G) + Qwen3-1.7B (1.8G), 11-26 16:18–16:37 — not referenced anywhere in this repo (another project's?). Nothing deleted (no pip/wheel caches in dead worktrees).
- JACK 00:20 (answers RESEARCH B-7 band): Ni 2022 has NO table row for Pendulum-V — appendix table l.252ff lists only Ant/Cheetah/Hopper/Walker; the band exists only as appendix Fig. separate_vs_shared. Read by pixel instead (`experiments/b7_digitize.py`, calibrated on the plot's own gridlines): separate LSTM-SAC 40k–50k mean −170.1, 95% CI shade −154.5…−190.9, 4 seeds; draft row B-7 in KNOWN_ANSWERS (`38d88ade`). W-1's band still open.
- ANSWERED 02:55 (ruling 130): DEFAULT STANDS — A: n_perm 2000, seed 0, for BOTH looks; they are the harness's defaults, the values check #1 and the false-alarm sample were calibrated at, and were in the reader (`968da0b8`) before any run was home. Next forge slot (do not wait for 14:00 if one comes sooner): commit r1L's judge lock ALONE (`r1L_read`, `r1L_beside`, `looks.py`, continuation, `r1_read`) before any reader opens a raw file; one read per look at seed 0, no second seed or count ever. (Was: QUESTION (forge, F3/J18) 02:25 — round 1-L's frozen rule names no permutation count or seed; options A as written / B other values.)
- JACK 05:40 (J24 follow-up, for the sitting): DRAFT `prereg/eye_bins16_k1.md` (`aabe9573`, not frozen, nothing run) — J24's nine cells, seeds 941–943, 500k, J24's bar unchanged, the only change K1's resets every 50k on all three cells (K2 swaps in by one line if 1-L's keep rule names it). Build owed before a freeze: `pilot_b2.run_pilot_cell` has no `param_resets` path. Sits beside `eye_geom_500k.md` and the per-ray one-hot arm for the sitting's choice.
- JACK 05:55 (J24 follow-up, for the sitting): the reset knob `eye_bins16_k1.md` owed is BUILT (`42fff23c`) — `pilot_b.run_b(param_resets=)` → sbx's own resets, `pilot_b2.J24K1_ARMS`, planted test 3/3, declared stale-cost 0 (the 05:15 slot's "moves code under many certificates" was wrong). Still owed before a freeze: chain/launcher wiring for the K1 arms + carry check (sbx saves `reset_idx` with the model) + a timing cell. Nothing launched.
- JACK 06:55 (J24 follow-up, for the sitting): `eye_bins16_k1.md` is now ONE FREEZE from launch (`2bcd197d`): `j24_launch --variant k1` (own freeze guard, `/data/jack-logs/j24k1`) carries the K1 arms; carry PROVEN (resets one per segment, cut+carried == unbroken, lost reset count refused); a reset costs ~9 s once, 9 resets < 0.5 % of a 5.2 h cell. Still owed at the freeze only: point `j24_read` at `j24k1`. Also: `tests/test_j24_launch.py` two tests assert eye_bins16 is a draft — expired at the 11-26 freeze, FAIL pre-existing, not touched.
- JACK 07:20 (J24 follow-up, for the sitting): the last item `eye_bins16_k1.md` owed is BUILT (`b2afeb75`) — `j24_read --variant k1 --pin <freeze sha>` reads the K1 cells by J24's rule unchanged, refuses a draft prereg / a wrong pin; default J24 read reproduces read.json. The K1 re-ask now costs exactly one freeze commit + `j24_launch --variant k1 chain --pin`. Nothing frozen or launched.

## DONE (kept one day)

- **J26** (`54e45611`): held-out R² — old row −0.135, one food code −0.013, kind one-hot 0.585, bins16 0.606, permuted control −0.0001. P1 HOLDS, P2 MISSED (0.6, not > 0.80), written so in `WORLD_CARD.md`.
- **J27** (11-26 10:35): `stage2_a_vs_b.md` re-drafted for cloud only — B = the forge's keep setting, a B-with-memory arm, size S ≈ 55 quota-weeks (unaffordable), size1m ≈ 12; README J25 line corrected.
- **S19** (`7033bcc8`): T0.12 repaired by cause (S15 billed a fetched log for a 0-s watch); T0.12, T0.24 PASS.
- **S18** (`14fae122`): docs never refuse a GPU submit; 5/5 planted tests.
- **J24 (0)–(2), (4)**: kind paint + bins16 built and proven; toy K holds / C MISS; forager PASS 20/20; timing 26.5 decisions/s.
- **J10** (`verdict_j10_2`): the walking row PASSES by its frozen rule, 4/5 seeds, mean 6,084.9 vs 6,232 ± 280.
