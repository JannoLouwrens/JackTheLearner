# ORDERS — what the workers do. Written by the Director.
Director sitting: 2026-11-27 06:41 UTC

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5, **each ≤ 4 lines of text — replace,
do not append; history is in git**). One state+journal commit per slot.
Machine problems go under INBOX, one line each. Rulings 131–136 cited here
are in `docs/DECISIONS_RESOLVED.md` under 2026-11-27 06:41; 121–130 under
11-26 19:14.

**`docs/PLAN.md` binds every order (owner, 11-15).** Research → combine →
reproduce → test the last 10% → put it in him. We are in **stage 1: the
instrument.** `docs/research/PUBLISHED_PRIORS.md` is the reference — read
your order's section (only that section) at the start of every slot.
**Where stage 1 stands (11-27 morning).** J24 is read by its frozen rule:
**NOT SHOWN AT 500k** — the learner on the kind eye ate LESS than its blind
and shuffled twins (0.23 vs 0.89 vs 2.34 late meals a life, 0/3 seeds).
Beside it: every sighted cell ate well early (up to 8.8 meals a life) and
then unlearned it — the same collapse plain SAC shows in the forge's rung
M (4/5), which resets (K1) and TQC (K2) do not (0/5). So the suspect is
the learner, and the re-ask is built and one freeze from launch. **Ruled
(131): J30 freezes and launches it NOW on the three Kaggle CPU sessions
that have stood empty since 02:45.** J24's row stays NOT SHOWN.
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
**Usage.** The shared weekly meter reads 15% all models (resets Wed 12-02
12:00 UTC). Every multi-segment job stays a SELF-ADVANCING CHAIN (ruling
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
Learner runs in his world only through J30 and the forge's frozen rounds.
**Stopped (rulings 10, 20, 23, 36, 59, 80, 92, 101, 107, 111, 113, 131):**
home-made learners; forager cells on the 16-ray compass eye; world-rule
races; `D1.0`, `T2.01`, smell runs; the needs-hidden row; ANY learner seed
on the per-geom paint; a per-ray one-hot eye arm (J26: it tells no more
than the bins, 0.585 vs 0.606); a learner residual on the hand reflex; the
MLX port; new specs with no published prior. A question with no prior is a
`RESEARCH:` line in INBOX.
**Kaggle CPU: 5 sessions IN ALL (ruling 85).** Rank: (0) any kernel
< 1 h; (1) J30's three seed chains; (2) J18 round 1-L. Launch from a clean
worktree ff'd to a PUSHED commit (its own worktree per job, `JACK_REPO=`);
copy its receipts into main's `gpu_submissions.jsonl` on harvest; do not
remove a worktree while a watcher lives. Every JACK slot's first act:
`/data/venvs/kaggle/bin/kaggle kernels status` on each running slug —
harvest what is home, refill what is free.
**GPU (rulings 96, 124).** The Kaggle account is shared: OUR meter counts
only our jobs, and **Kaggle's refusal is the truth.** W48 opens Sat 11-28
00:00 UTC with 30 h; three timers are armed for ≈ 22.5 of them (pids
3886689 A's seeds, 29379 ear, 688795 J28). A refusal ends that week's GPU
plan; nothing retries in a loop. Free Colab T4 answers 503 since 11-25 —
one try per job, no loop. No first launch of a GPU job or of a NEW learner
shape on a CPU kernel without a timing cell (rulings 86, 99).
**The box (rulings 32, 53).** Day ceiling 57,600 CPU-s, ≤ 2 computing
detached jobs. < 15 CPU-min → box, nice 19. The box is shared (never touch
other projects). `pgrep -af` the same command line before launching.
**Disks:** `/` 5.1 GB free (workers refuse under 3); `/data` 16 GB free,
90% full — the growth is other projects' (11-27 00:15), not ours. Scratch,
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

*Order of work: **J30** (freeze + launch NOW, three sessions) → **J18**
(watch; look 1 when g1-0…g1-4 are whole) → **J29** (box and text, no
kernel) → **J14** / **J28** (harvest when Saturday's timers fire) →
Next-up → exit. A slot waiting on a kernel advances the next order.*

### J30 — J24 asked again on the learner that does not collapse: freeze `eye_bins16_k1`, launch the nine cells
part: Learning to eat + Eye · stage: 1 · step: 4 (test the last 10%) · prior: `docs/research/prereg/eye_bins16_k1.md` § Prior (read it first); §21 (win-then-lose), §28; `WORLD_CARD.md` § J24; rulings 121, 131
prior: published — resets are the record's best-supported cure for a learner that stops learning (Nikishin 2022, DMC IQM 501 → 656; Nauman 2024); the record's foragers eat at 1e6–3e6 steps (§28), ours asks at 500k with one stated extension to 1.5M.
prior: ours beside theirs — rung M (old eye, 300k, 5 seeds): K1 resets 0.84, plain SAC 0.059 and collapsing 4/5, K1 0/5. J24 (plain SAC, kind eye): sighted eats early (s941 8.78 meals a life) then unlearns to ≈ 0. B reproduced (reacher-hard 933.8). Learnable by a plain rule: hand forager on the food bins 20/20.
**Ruled (131).** J24's row stays NOT SHOWN AT 500k — nothing about it is
re-read. This is a NEW row with J24's seeds, size, twins and bar
unchanged; the only change is K1's resets every 50k on ALL three cells.
K1, not K2: the keep rule's tie goes to K1 and look 1 is a day away.
**Do.** (1) In this slot only, bounded 45 minutes: add per-kind contact
counts (touches of food / each other kind per life) to the life row as
BESIDE columns — J24's collapse could not say what he approaches instead.
Keep it only if J24's three K identities, the carry test and the reset
test all re-pass; otherwise revert and go on without it. (2) Set
`eye_bins16_k1.md` STATUS to FROZEN, exactly as drafted (rule § unchanged),
commit that file alone, push. (3) `launch_detached.sh … python -m
experiments.j24_launch --variant k1 chain --pin <freeze sha>` — three
chains, one seed's three cells a kernel. (4) (the two expired launcher tests: DONE `8924b588`.) (5) Read once, when all nine are whole:
`python -m experiments.j24_read --variant k1 --pin <sha>`.
branches: EATS → row in `WORLD_CARD.md` + `KNOWN_ANSWERS.md`, the bar copied into `pilot_a.md`, INBOX one line (the hindsight eye is raced next sitting under ruling 134). · ≥ 2/3 above both, not EATS → launch the 1.5M carry yourself from the same pin. · else → write NOT SHOWN AT 500k ON K1 with the nine numbers and the 50k window table (did the collapse go away or not — say which); launch nothing; the sitting chooses between size (1.5M) and the eye. · a kernel dies → relaunch that seed's chain once from its last whole carry; twice → INBOX with the log path.
serves: M3 (the earliest unfinished milestone) · PLAN "Learning to eat", "Eye" · philosophy 1, 3, 4 (a failure fixed, not filed).
**done when:** the prereg is frozen, three chains run with raw-file paths in a state line; then the read by the frozen rule.
state: 11-27 07:23 **(1) KEPT `5bda4a6e`**: life rows on the kind paint carry BESIDE `touch` {food, water, obstacle, wall, panel} = decisions ending in contact (`pilot_b.PilotLives._count_touch`, read-only; same cell before/after has identical actor, trace, rows). J24's three K identities, K1 carry, reset 3/3, launcher 10/10, reader 3/3 re-PASS; `tests/test_j30_touch.py` (forager that ate → food 3883; still body → 0).
state: **(2) FROZEN `21a45857`** (`eye_bins16_k1.md`, STATUS line only, committed alone, pushed). **(3) LAUNCHED 07:21–07:23** from worktree `/data/wt/jack-j30` at that pin, JACK_WATCHER=cpu: s941 pid 1149300 kernel jack-cpu-1795764111 · s942 pid 1149715 -1795764132 · s943 pid 1150532 -1795764177, all RUNNING per Kaggle 07:23 (5/5 sessions with J18's two). Logs `/data/jack-logs/j24k1/chains/chain_s94{1,2,3}.log`; raw → `/data/jack-logs/j24k1/chains/j24k1_s94{1,2,3}.json`. One segment each, ≈ 5–7.5 h → home ≈ 12:30–15:00 UTC.
state: NEXT (5): when all three raw files exist, ONCE: `/data/venvs/jack-std/bin/python -m experiments.j24_read --variant k1 --pin 21a45857f644e9920599def71ad9d07cb6a2d0f2`, then the branch. Harvest: the three attempt receipts are already in main's `gpu_submissions.jsonl`; copy only the worktree's RESULT rows. Beside the read: `j24_collapse_diag` per raw file (window table) and the `touch` columns. Do not remove the worktree while a watcher lives.

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
**Added 11-27 (ruling 135).** Look 1 needs g1-0…g1-4 whole; g1-4 sits
third in chain A (≈ Sat evening). When J30's three sessions come free
(≈ 6 h after its launch): read `r1_run chain` — if a second chain can
take g1-4 → g1-6 and g1-5 → g1-7 with NO chance that chain A/B also
launches them (a claim file both chains honour, planted test: a claimed
name is skipped, an unclaimed one is not), build that and launch chains
C and D from the same pin `495a766d`. If it cannot be made certain in one
slot, leave the chains as they are — a seed run twice is worse than a
day's wait (check #13: one run a seed, never a choice between two).
**done when:** `r1L/spec.json` is frozen and tranche 1's chains run with log paths in a state line; then the harness verdict at look 1.
state: 11-27 04:40 tranche 1 segment 0 HOME both chains (g1-0 03:46, g1-1 03:49 → `r1L/raw/segments/L_g1-{0,1}_seg0.json`; all 4 runs a kernel cut + carried, none lost); segment 1 kernels jack-cpu-1795751166 (A) / -1795751359 (B) RUNNING per Kaggle 04:40; wrap pids alive. Segments not opened (judge reads only whole runs). NEXT unchanged: g1-0…g1-4 whole → `r1L_read --tranches 1` once.
state: 11-27 03:35 **JUDGE LOCK `9d3feb7e`** (ruling 130), committed alone at `85ca3d4e` before any raw file: harness incl. looks.py (__init__ 382cf304 unchanged), registry, forge __init__/metric, spec 52033df2, r1L_read, r1L_beside, continuation, r1_read. Both readers now `check_lock` before opening raw (tests 4/4, 10/10, looks_fa 6/6). Kernels A/B RUNNING (Kaggle status 03:40, 7h08m; wrap pids 237214/237409 alive), no raw file home.
state: 11-26 20:12 per-arm tranche seeds in `r1_run` (`e99c1b78`; rule `f618bba5` in `forge/continuation.py`, out of judge-locked `forge/__init__.py`). 11-26 20:27 **(2) FROZEN `52033df2`** `rounds/r1L/spec.json`: parent r1 lock 2eb2c0f4, K1+K2, tranches exactly `_r1l_like()`, PRIMARY `harness.compare(K1, E-food)` on `outcome_sustained` **W = 2** (E2c's frozen W, unchanged) T = 1M, K1 resets 50k…950k (19), keep rule + 180 kernel-h a tranche. `forge_scoreboard` renders a parent round via `validate_round` (board re-rendered). Tests: r1_run (new frozen-shape test), r1_spec 8/8, round 27/27, read 13/13.
state: (3) LAUNCHED 11-26 20:30 tranche 1 from pin `495a766d` (rule `52033df2`), beside J24's three (order of work: "the other two"): chain A g1-0→g1-2→g1-4→g1-6 (pid 237214, kernel jack-cpu-1795725017), chain B g1-1→g1-3→g1-5→g1-7 (pid 237409, jack-cpu-1795725031), both RUNNING 20:31; JACK_WATCHER=gpu. Logs `/data/jack-logs/forge/r1L/chains/chain_{A,B}.log`, raw → `/data/jack-logs/forge/r1L/raw/`. A kernel ≈ 7.5–18.3 h (≤ 3 segments); K1+baseline (g1-0…g1-4) home first ≈ 3 kernels deep a chain. NEXT: watch chains (a lost run = a crash, never replaced); when g1-0…g1-4 are home → look 1 needs a 1-L reader (S20; `r1_read` locked) — print rates meanwhile.

### J29 — Brain B with a memory: the control stage 2 needs, from published code
part: His brain (B) · stage: 1 · step: 2 (combine) → 3 (reproduce) · prior: §5 row 7 (read it first), §6; `KNOWN_ANSWERS.md` draft row B-7 (`38d88ade`); `prereg/stage2_a_vs_b.md` row `B-mem`; ruling 136
prior: published — Ni et al. (arXiv 2110.05038, "Recurrent model-free RL can be a strong baseline"; LSTM-128, separate actor/critic encoders, whole-episode sequences): LSTM-SAC on velocity-only Pendulum reaches about −170 by 40k–50k steps (4 seeds, band read off the appendix figure by pixel: −154.5…−190.9); a memoryless SAC cannot solve it. Ours: no recurrent arm exists; B (memoryless) reproduces reacher-hard.
**Why.** Stage 2 races a world model (A) against B. Without a B that has a
memory, "A wins" could mean only "A remembers". The arm must exist and
reproduce a published number before the race is frozen.
**Do.** (1) Name the published code to build from (the authors'
`pomdp-baselines`, or a maintained library's recurrent SAC — check what
installs under jack-std; code only, no weights) and write the design note
in the module docstring with the "ours beside theirs" table. (2) Build
`experiments/brain_b_mem.py` beside `brain_b` (B untouched). (3) Time one
cell on the box (velocity-only Pendulum, 2k steps, nice 19) and write
seconds per 1k steps and the cost of 50k. (4) Freeze row B-7 in
`KNOWN_ANSWERS.md` BEFORE the first seed: Ni's configuration, 51,000
steps, seeds 0–3, pass = mean over 40k–50k inside the pixel-read band;
twin = memoryless B on the same task must end outside it; the
shared-encoder arm beside it, expected worse (decides nothing); no
extension. (5) Run it: box if
4 seeds fit the day ceiling, else one Kaggle CPU kernel at rank (0)/(2).
branches: no library installs cleanly → vendor the authors' recurrent actor/critic as code with its licence line, say so in the note. · B-7 in band → row written; `stage2_a_vs_b.md` (draft) names the arm. · miss → a miss, written; the arm is a suspect — INBOX with the curve, the sitting rules. · the twin passes → the task is not a memory test as built: stop, INBOX.
serves: M3 (stage-2 ready-list: "a control brain that has a memory arm") · philosophy 3, 4.
**done when:** the design note and timing are in a state line; then B-7 frozen; then its four seeds and twin read.
state: 11-27 08:45 **(1)+(2) BUILT `befb7848`** `brain_b_mem.py`: torch/gymnasium PORT of `pomdp-baselines@e7c19c32` recurrent SAC (MIT; no library has one), Ni's config, ours-beside-theirs table in the docstring. **(3) TIMED** box 1 thread 1.62 CPU-s/update ≈ 22.5 CPU-h a 51k seed (does not fit the box); 2 threads same CPU-s, wall 0.9 s/update (`/data/jack-logs/j29/timing_box*.json`).
state: 11-27 09:30 **(4) B-7 RULE FROZEN `c848d888`** (before any seed): seeds 0–3, 51k, arm score = 4-seed mean of eval returns at 40k–50k (11 points), REPRODUCES iff in −190.9…−154.5; twin `B7flat` = memoryless Brain B (`brain_b.py` ours, Zoo Pendulum settings) on the same Pendulum-V, VOID if ≥ −190.9; shared arm beside; no extension; a lost seed = VOID, never replaced.
state: **Run apparatus built:** `run` loop in their learner order (`19036ff0`: 1,000 init updates, eval every 1k steps 2k–51k, curve rewritten each eval) + **carry** `--carry/--stop-at` (`8b98a469`), bit-equal over two cuts, planted RNG drop caught; tests 6/6 `python -m tests.test_brain_b_mem` (jack-std).
state: NEXT (5): Kaggle 5/5 busy until J30's kernels home (≈ 12:30–15:00). Then a Kaggle CPU chain: per seed `python -m experiments.brain_b_mem run --seed S --threads 4 --out … --carry … --stop-at <steps fitting ~11 h>`, carry across kernels (watcher kind cpu). Still owed before launch: the `B7flat` twin runner (brain_b SBX on PendulumV, same eval schedule/score) and the chain wiring; read once all 4+4 are whole with `b7_seed_score`.

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
probes of other seats stay in `LAB.md`. The lab's hindsight eye: ENTERED,
waits for J30's read; when raced it carries a no-unlearning twin and a
below-ceiling measure (first meal → fifth) — the 11-22 ruler cannot fail
on the kind eye (ruling 134). Lab's next unit, box-minutes, no learner:
build that below-ceiling ruler on `paint="kind"` and show it CAN fail (a
yoked twin must lose on it, a random-imprint table must not pass).
**Parked or closed:** the MLX port (ruling 113). `D1.0`, `T2.01`, `LF.06`,
`LC.09`, `CR.01`, `LF.04` — "to earn again" on the chosen brain. `NE.01`,
`PS.02`/`.05`/`.06`/`.08`, the place key, the needs-hidden row → stage 3.
`W1.04` → Sunday's sitting. `prereg/eye_geom_500k.md` stays a draft: any
learner seed on the per-geom paint is stopped (ruling 111) until J30 is
read.
**Next up, in order (text or box < 15 CPU-min; do ALL of these before any slot ends idle):**
1. Draft (text, NOT frozen) `prereg/w1_walkerwalk.md` (ruling 133): the
body's known-answer row is banded on Playground `WalkerWalk` PPO (paper
≈ 968, SLM-Lab 945, every seed ≥ 900 — §30), humanoid beside it unbanded;
seeds, steps, the one extension, the twin. A sitting freezes it after
J28's cell 1 says JAX runs on Kaggle's GPU.
2. Stage-1 ready-list, one page `docs/research/STAGE1_READY.md`: each
line of PLAN's ready-list with the commit, row or file that proves it, or
"NOT YET — what is missing". No new claims; pointers only.
3. "The interface correct at death" (ready-list): one planted box test —
the observation returned on the dying step is the dying body's, not the
reborn one's — for the wrapper J30 and the forge use. If it exists, name
it in item 2 and skip.
4. `WORLD_CARD.md` § J24: add the 50k window table for all nine cells
(from `j24_collapse_diag.py`), marked BESIDE.

---

## SYSTEM — work on the machine (max 3 open)

*Order of work: **S21** → **S20** → **S17** → Next-up. A fix a RUNNING
Jack kernel needs goes first (ruling 54).*

### S21 — The ladder's root test went red from a slow clock: measure the import, not the neighbours
part: (machine — the ledger's truth; card line 4) · stage: — · step: —
**Why.** `T0.01` ("repo imports clean") has been FAIL since 11-26 12:44:
all 13 modules import, but `UnifiedBrain` took 5.4 s against the 5.0 s
line while the whole run took 9.0 s (1.71 s the day before) — the shared
box was busy. Five PASS rows now rest on a dead dependency and the
re-checker will not re-buy them. Nobody owned it (OVERSIGHT 11-27 § 1).
**Ruled (132).** The 5.0 s line does not move and nobody re-rolls until
the clock is kind. The line exists to catch an import that DOES WORK
(a download, a scan). So: each module's import is timed in a fresh
process THREE times and the FASTEST is judged against 5.0 s; the 1-minute
load average is recorded in the row.
**Do.** (1) Change `experiments/tests/t0_01_imports.py` so. (2) Planted
tests first: a module that sleeps 6 s at import FAILS (all three are
slow); one slowed once by a planted stall passes with all three times
printed. (3) One named run, load printed. (4) If it passes, let the
re-checker re-buy the five rows under it (name them in the state line).
(5) One line each, no change: is `T0.07` (CPU throughput, FAIL since
10-23) the same cause? why did the 11-26 10:46 sweep "run but record
nothing" on five specs?
branches: still > 5.0 s on the fastest of three → it is a real slow import: find what `UnifiedBrain` does at import and fix THAT, by cause. · the planted sleeper passes → the change is wrong; revert.
serves: philosophy 4 (a red root test repaired by cause, not by retry), 5.
**done when:** planted tests pass, `T0.01` is re-bought with its load in the row, and `coverage` no longer prints FAIL-UNOWNED for it.
state: (1)-(3) built `bff35c96`: each module timed in a fresh process, tries stop at the first under 5.0 s (verdict = fastest of three, 1/3 the wall); planted `python -m tests.test_t0_01_slow_clock`: 6 s sleeper 6.0/6.0/6.0 FAILS, once-stalled 6.0/0.01 passes, broken import reported.
state: (3) T0.01 PASS 11-27 09:10 (`f2e7cc58`), slowest module 1.58 s, load_1m 1.45 in the row; coverage FAIL-UNOWNED 0. (4) Left to the re-checker: T0.02, T0.06, T0.20, T0.21, T1.03.
state: (5) T0.07: same family, not same rule — its FAIL is policy_no_llm_rel_std 0.272 > 0.20 (repeats disagree on a busy box); fastest-of-N does not apply to a spread check; unchanged. 11-26 10:46: runner's CPU gate refused all five (day budget 20k/57.6k s, so load > 6.0); regate now prints the REFUSED reason instead of '(no verdict line)'.
state: DONE when the re-checker has re-bought the five.

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
(emptied 2026-11-27 06:41 — LAB 23:35 → ruling 134; RESEARCH 04:11 → ruling 133, Next-up 1 (cell 1 already fixed `54e0a7dc`); JACK 00:15 → PROGRESS, for the owner; JACK 00:20 → J29; ANSWERED 02:55 → kept, ruling 130 stands; JACK 05:40–07:20 → ruling 131, J30.)
- JACK 07:23 (J30): `/data` is at 15.06 GB free — 62 MB above the 15 GB floor. J30's worktree cost 49 MB and its raw files are MBs; the growth is not ours (as 00:15), but the next harvest of any size will cross the floor unless something outside this repo is cleared — the owner's.
- JACK 07:23 (machine, no change made): the forge's chains A/B declare `JACK_WATCHER=gpu` though they poll `jack-cpu-` kernels, so the gpu poller count read 5 of 6 and three more gpu-declared chains would have been refused. J30's three are declared `cpu` (what `cpu_budget` itself calls a jack-cpu poller). Also `tests/test_j24k1_resets.py` cannot be run by path (no sys.path line; ran it with `PYTHONPATH=.`).

## DONE (kept one day)

- **J24** (`2758215c`): **NOT SHOWN AT 500k** by its frozen rule — late meals a life sighted 0.229, blind 0.887, shuffled 2.338; above both 0/3. Beside: sighted eats early (up to 8.78) then unlearns; plain SAC collapses the same way in rung M. Re-asked on K1 as J30.
- **J26** (`54e45611`): held-out R² — old row −0.135, kind one-hot 0.585, bins16 0.606; P1 HOLDS, P2 MISSED. Post-hoc on looks with food in sight: 0.700 / 0.726, still under 0.80.
- **J27** (11-26): `stage2_a_vs_b.md` re-drafted for cloud only.
- Auditor 11-27 § 2 (permutation count and seed): ruling 130, judge lock `9d3feb7e` before any raw file. § 3 (J28 launcher): armed 02:39.
