# ORDERS — what the workers do. Written by the Director.
Director sitting: 2026-11-24 12:26 UTC

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5, **each ≤ 4 lines of text — replace,
do not append; history is in git**). One state+journal commit per slot.
Machine problems go under INBOX, one line each. Rulings 92–100 cited here
are in `docs/DECISIONS_RESOLVED.md` under 2026-11-24; 76–91 under 11-22.

**`docs/PLAN.md` binds every order (owner, 11-15).** Research → combine →
reproduce → test the last 10% → put it in him. We are in **stage 1: the
instrument.** `docs/research/PUBLISHED_PRIORS.md` is the reference — read
your order's section (only that section) at the start of every slot.
**Where stage 1 stands (11-24).** The team was stopped 45 h (usage meter),
back since 09:51 today. Read since: **the cure hunt is CLOSED — four
published critic cures, one seed each, none stops brain B's critic running
away on the 128-ray eye** (J17, branch 4). J19 is read: one of B's three
published misses was OUR test's size (reacher-hard is in band on 100
episodes), the swingup-sparse miss is B's own. The hand forager passes its
bar again on the one physics version (19/20, control running, J20). **So:
the world is learnable by sight, and B with a flat eye does not learn it.
What was never tried is the record's own recipe for an image-like sense —
a small conv encoder with random shift (§12, §24) → J22, bounded, branches
ruled.** The keep round's 30 runs have been HOME AND UNREAD since 11-22
(J18). Neither brain is cleared for stage 2.
**THE USAGE STOP WILL COME AGAIN.** The weekly meter is shared with other
projects: 15% at 12:22, resets Mon 11-30 05:00 UTC; at last week's pace
workers stop Thursday night and stay stopped 3 days. Every slot: **(1) no
free Kaggle CPU session while a ready kernel exists; (2) every branch below
is already ruled — do NOT end a state line "NEXT: Director" when the order
names the branch; take it; (3) kernels and box jobs write raw results where
a later slot can read them; (4) a slot that can advance nothing exits at
once, without building speculative arms.**
**One world (ruling 76): mujoco 3.17.0.** Every row about his world runs in
`/data/venvs/jack-std` (box) or a kernel pinned to 3.17.0. Learner tests
(`sbx`, `optax`) run under jack-std, not the project venv.
**Every new run in his world passes** `eye_frame="body"`,
`motor_frame="body"`, `food_floor_only=True`, `lazy_contact_audio=True`,
`native_substep=True`; brain B uses `ent_coef="auto"` (ruling 64); "eats"
reads the LATE half of the run (ruling 65). Learner runs in his world only
through J22, J14 and the forge's frozen round (rulings 58, 92).
**Stopped (rulings 10, 20, 23, 36, 59, 80, 92):** home-made learners;
forager cells on the 16-ray compass eye; world-rule races; `D1.0`, `T2.01`,
smell runs; the needs-hidden row; more critic cures on the flat eye; new
specs with no published prior. A question with no prior is a `RESEARCH:`
line in INBOX.
**Kaggle CPU: 5 sessions IN ALL (ruling 85); a share binds only while
another order's READY kernel is waiting.** Rank: (0) any kernel < 1 h;
(1) forge round 1 — 2; (2) J22 — 2; (3) J10-2's segments — 2; (4) J23 — 1.
Every JACK slot's first act: `/data/venvs/kaggle/bin/kaggle kernels list
--mine` + `status` — harvest what is home, refill what is free.
**GPU (ruling 96).** Kaggle GPU is shut until Sat 11-28 00:00 UTC (W47
spent on our meter). **Free Colab T4 is OPEN now** (it ran J17's h0 pair in
56 min today): `gpu.submit(prefer="colab")`, pins jax 0.11.2 / flax 0.12.12
/ optax 0.2.9. A Colab job is ≤ 3 h or it saves and resumes. No first
launch of any GPU job without a timing cell (ruling 86) — and now no first
launch of a NEW learner shape on a CPU kernel either (ruling 99). TPU closed.
**The box (rulings 32, 53).** Day ceiling 57,600 CPU-s, ≤ 2 computing
detached jobs. < 15 CPU-min → box, nice 19. Longer → box only when all 5
Kaggle sessions are held AND load < 4. The box is shared (never touch other
projects). `pgrep -af` the same command line before launching.
**Disks:** `/` 5.4 GB free (workers refuse under 3); `/data` 15 GB — AT its
floor (→ S11, first SYSTEM order). Scratch, clones, caches under `/data` only.
**World files** (`w0.py`, `needs.py`, `survival.py`, `drives.py`): changed
only through `jack_env.py` options whose default is today's world.
**Known-answer rows (rulings 33, 70, 94):** seeds AND at most one extension
stated BEFORE the first seed. A miss is a miss. An extension that cannot
change the verdict by its own arithmetic is not run. **Every NEW brain-B
known-answer row is scored on 100 evaluation episodes, re-seeded each mark
as the reference code does.**
**Receipts:** the slot that launches a kernel commits
`experiments/gpu_submissions.jsonl`, `gpu_budget.json` and `cpu_budget.json`
by name.

---

## JACK — build the creature, his world and his curriculum (max 7 open, ranked)

*Order of work: **J22** (build first; it is box work) → **J20** (g4 is
running; harvest it) → **J10** → **J14** (Colab is open: the timing cell
goes today) → **J23** → **J21**. J18 is the forge's — and any JACK slot's
after 11-25 06:00 if still unread. A slot waiting on a kernel advances the
next order.*

### J22 — His eye the way the record feeds an image to a replay learner: conv encoder + random shift
part: Learning to eat + Eye (+ His brain B) · stage: 1 · step: 2 (combine) → 4 (pilot, then the frozen row J17-2) · prior: §12 (l. 431–456), §24; `WORLD_CARD.md` § J17 + hunt; rulings 77, 92, 93, 99
prior: published — SAC on a raw image fails (walker 42 vs 948 from state, 500k); random shift alone repairs it (DrQ 921; no auxiliary loss beats it, Li 2022); DrQ-v2 96 fps. More inputs are not harmful by themselves (OFENet). Ours: 256 ray numbers fed FLAT to an MLP critic; row NEITHER EATS; four critic cures (SimBa, AdamW, OFN, split first layer) none holds on s923. Nobody has published this recipe on a 1-D ray row — that is our last 10%.
prior: learnable — hand forager on this eye 19/20 at d0.75 (7.86 meals/life, J20); a plain learner only with the direction handed (E2c 4/5). Steps: the runaway separates from blind at 100k.
**Do.** (0) Text, box-minutes: re-read the four saved hunt traces with the
J17 ROW's own measure (sighted Q ÷ supported return) printed beside the
hunt's climb ratio, one table in `WORLD_CARD.md` § J17 hunt. No verdict
changes (ruling 93); if the two rulers disagree on an arm, say which in one
line. (1) Build arm **h4** on the hook h3 used: the ray row as a 1-D image
(rays × its channels) → 4 × Conv1d(32, kernel 3; stride 2 then 1) → linear
50 → LayerNorm → tanh, body/needs numbers joined AFTER it; random shift
±6 rays (edge-replicate pad, or roll if the eye is a full circle — read the
eye and say which) drawn separately for obs and next-obs in the critic
update; the encoder learns from the critic loss only, the actor reads it
detached. Every difference from DrQ in the docstring, "ours beside theirs".
Tests committed first: shift moves rays and nothing else; actor loss gives
the encoder zero gradient; `pilot_b`'s partial file keeps the trace.
(2) Timing cell (box 300 decisions, then Colab T4 5k+5k) → pick the venue
that fits 100k in one session. (3) Pilot, s923, 100k, ONE machine, three
cells: h4 sighted / blind twin / **shuffled twin** (each step's rays
permuted). Read by `j17_onset hunt` + `j17_late_meals(rec, 50_000)`.
(4) **Branch, ruled:** CURES = the reader does not print RUNS AWAY and
sighted late meals/life ≥ BOTH twins'. Cures → CONFIRM on 921, 922; holds
on ≥ 2 of 3 → **J17-2 exactly as frozen 11-22** (`git show
b4b709ce:docs/ORDERS.md`, J17 (3)) with h4 as the one key; write its pins
to `docs/research/prereg/row128_j17_2.md` before seed 931. No cure → one
row, one INBOX line, and **B is closed on the wide eye for stage 1**: no
h5; the eye waits for brain A's pilot (J14 (d)) and the hindsight eye.
(5) Debt, stated now: a J17-2 PASS with h4 does not count toward stage 2
until DrQ's own row (walker_walk from pixels, 921 @ 500k) is reproduced
with the 2-D form of this encoder on W48's GPU, timing cell first.
serves: M3 · PLAN "Learning to eat" + "Eye" last 10% · philosophy 1, 3, 4.
**done when:** (0)'s table is written and h4's three pilot cells have a read row with the branch taken.
state: DONE 11-24 15:45 — (0) `99bcd86e`, (1) `35d75d4d`, (2) `df11a192`, (3)+(4) read in WORLD_CARD § J17 hunt (h4 row + paragraph), file `world_card_data/j17/j17_hunt_h4.json`.
state: pilot home 15:30, 3 cells × 100k, none cut. Reader: sighted climb 1.818 (blind 1.169, shuffled 1.307) → STILL RUNS AWAY; late meals/life sighted 0.56, blind 0.31, shuffled 0.87 → **NO CURE** (shuffled eye eats most).
state: J17 ruler disagrees as on h0: sighted Q÷SUP 0.50 is the LOWEST of the three (.56/.60) — held, small Q(50k) base (112 vs 191/187). Conv eye does not run away on content, but does not use sight to eat.
state: branch taken as ruled: B closed on the wide eye for stage 1, no h5, INBOX line written; the eye waits for J14 (d) and the hindsight eye. Debt (5) not owed (no PASS).

### J18 — Forge round 1 is a KEEP round: rung S is HOME since 11-22 12:40 and UNREAD
part: Learning to eat + The science lab · stage: 1 · step: 4 · prior: §19, §21, §22; `docs/FORGE_POOL.md`; rulings 61, 65, 69, 83
prior: E2c — on E-food the baseline reaches (4/5) and loses (10/10). Record: resets + LN critic best-supported (Nauman 2024) but ours has LN and lost; TQC; 1 update per 10 decisions; `n_steps=3`. Nobody has published a small agent still improving at the end of a long life (§20).
**Terms frozen `594752d0`; the four fill-ins STAND (ruling 83).** All 8 raw
files are in `/data/jack-logs/forge/r1/raw/S_g{0..7}.json` (g7 is half the
size of the others — check it is whole before scoring; a cut kernel is
re-run from the pinned commit, not scored).
**Do, in this order.** (1) Write the reader. (2) `experiments.forge.
write_lock(round_dir, [reader])` and commit `judge_lock.json` ALONE —
**before any raw file is opened** (S13 (b), `e3e573e5`). (3) Read rung S
onto the scoreboard. (4) Before rung M: the round runner saves through
`brain_b.save_life` (300k runs must resume) — the only code change allowed
between rungs. (5) Launch rung M's kernels by the frozen rule (rank 1).
**If rung S is not on the scoreboard by 11-25 06:00 UTC, the first JACK
slot after that does (1)–(3) itself, same lock procedure (ruling 100).**
serves: M3, M7 ("he keeps learning") · philosophy 1, 3, 4.
**done when:** rung S is on the scoreboard — one reading per candidate beside baseline, twin and controls — and rung M's kernels are launched by the frozen rule.
state: 11-24 14:14 forge: (1)–(3) DONE — lock `2eb2c0f4`, rung S READ `6b29c527` (N3 0.85, K1 0.81, K2 0.51 climb to M).
state: 11-25 05:40 step-4 core `41cc5d64` (run_arm carries in 20k chunks via `save_life`; carry 13/13). 06:10 kernel side `b9914a75`: job deadline t0+7.5 h, carries in the kernel output, seg n mounts seg n−1 and runs only its carried runs (one whole carry or refused), ≤ 3 segs, seg files in `raw/segments/`, `M_gN.json` assembled once (cost = Σ seg durations), fetch_only. r1_run 26/26 (jack-std), read 13/13, spec 8/8, lock holds.
state: **J18 DONE 11-25 06:16 (done-when met): rung M LAUNCHED by the frozen rule, pin `b9914a75`**, `rung_arms` = E-food, E-food-lr0, K1, K2, N3 + yardstick C2 × s1411–1415 = 30 runs / 8 kernels, 2 chains from `/data/wt/jack-clean`: A g0 g2 g4 g6 (first `jack-cpu-1795587333`), B g1 g3 g5 g7 (`…587347`). Logs `/data/wt/jack-clean/launch_M_chain{A,B}.log`. S took ≤ 3.2 h at 150k → M ≈ 6–6.5 h a kernel, a chain ≈ 26 h.
state: NEXT: harvest only — chains self-advance (and self-carry); when all 8 `raw/M_g*.json` are home: `python -m experiments.forge.r1_read --rung M`. Receipts land in the worktree's `gpu_submissions.jsonl` — copy into main's on harvest. A refused/failed group in `CHAIN DONE` is re-run from the SAME pin.

### J20 — One world: re-read the box's rows about his world on mujoco 3.17.0
part: His world + Eye · stage: 1 ("proof the task can be learned with his senses") · step: 4 (re-read) · prior: §12; `WORLD_CARD.md` (hand forager d0 / d = 1; the floor card); rulings 51, 76
prior: same seed, same 2,000 random actions on K-row128 → 0 meals on 3.2.3, 2 on 3.17.0. The hand forager's 20/20 at the entry rung and the floor card are 3.2.3 rows; every learner row is 3.17.0.
**Do, box, jack-std, nice 19, same seeds and code as the original rows:**
(1) hand forager on row128, body frame, d0.75 then d = 1, each with its
shuffled-ray control. (2) The floor card's three learner-free rows on the
J17 world (do-nothing, random, held-8 random), 200 lives each. (3) Each as
a "3.17.0" row BESIDE the old one in `WORLD_CARD.md`. Bars are unmoved
(ruling 51): a control that does not fail, or a d below 18/20, is a FAIL
to report in INBOX.
serves: M3 · auditor finding 3 · philosophy 4.
**done when:** the 3.17.0 rows (d0.75 with g4, d = 1 with g4, three floor rows) stand beside the old ones in `WORLD_CARD.md`.
state: DONE 11-24 21:40 `f197015d` — (1) d0.75 + d = 1 rows FINAL (f4 19/20 PASS at d0.75, 5/20 FAIL at d = 1, g4 0/20 both); (4) `250942db`.
state: (2) floor READ on 3.17.0 (60/60 cells, scorer `d87ea756`, planted 4/4): a bit-identical to 3.2.3 (200/200 lives); d8 1.04 vs 1.16 meals/life (Wilcoxon p 0.42), b 0.46 vs 0.52 (p 0.29) — same distribution, slightly lower.
state: §8 floor gate HOLDS: a + b = 0 seeds at f's bar (neither reaches 5 on any seed). (3) all rows beside the old ones in WORLD_CARD (§ J20 (1) ×2, § J20 (2)); raw `world_card_data/j20_floor_mj317.json`.

### J10 — The walking layer: the row again with enough seeds, life carried across segments
part: Body and movement · stage: 1 side job (stage 3's part) · step: 3 (reproduce) · prior: §16, §21; rulings 45, 72, 82, 95
prior: §16 — SAC, Zoo entry, Humanoid-v4: 4,825 [3,784–5,866] at 1M (10 seeds), 6,232 ± 280 at 2M, each ONE unbroken run. Ours: Row J10-1 FAIL (1 of 3); the resume dropped the episode and re-seeded the env; the carry (`d39687f0`) makes a resumed run bit-identical to an unbroken one.
**Row J10-1 stays FAIL. J10-2 CARRIES (ruling 82); rule frozen `b609b142`:
seeds 3–7, PASS R ≥ 5,950 AND ≥ 12 m/1,000 steps on ≥ 4/5, 50-episode eval
on Kaggle CPU, no extension.**
**Do.** Segments at rank 3, ≤ 2 sessions: when a seg record lands,
`kernel2 <s> <next>`; s5–7 when ≤ 3 sessions are busy. Every segment saves
its model. Ruling 95 (text, one commit): the 11-16 Yu-Turk-Liu PPO heading
in `KNOWN_ANSWERS.md` becomes "Row J10-PPO-pre (was J10-2 until 11-24)",
and `yu2018_walk.py`'s citation follows; ruling 82's SAC row keeps J10-2.
serves: M4 · PLAN stage 3 "a separate, fast walking layer is built and checked in parallel from now" · philosophy 1, 4.
**done when:** J10-2 is in `KNOWN_ANSWERS.md`, pass or fail, by its frozen rule.
state: J10-1: 1M median 4,065 IN BAND, control valid; 2M 1 of 3 → FAIL.
state: 11-22 07:55 `b609b142`: rule frozen before seed 3; `segment_job_body(row="J10-2")` carries + 50-ep eval, `require_carry` refuses a mount lacking life.pkl/physics.pkl (planted PASS). Tests 10/10.
state: 11-25 03:40 @2M: **s4 R 6,254.3 / 17.16 m/1k / 0 fell → meets the bar**. s6 seg3 `…577986` + s7 seg1 `…577077` launched from clean worktree `/data/wt/jack-clean` (`JACK_REPO=` needed: the main tree's forge dirt trips the submit guard).
state: @2M so far: s4 R 6,254.3 / 17.16 m/1k MEETS; s3 R 5,649.4 / 12.08 MISSES; **s6 (06:37) R 5,993.5 / 17.92 m/1k, 3/50 fell → MEETS** (`…577986`). Tally 2 meets / 1 miss / 2 open → PASS needs s5 AND s7 to meet.
state: 11-25 06:40 running: s5 seg3 `…580623` (since 04:24, ~3 h train), s7 seg2 `…587392` (1.5M). NEXT: s5 seg3 home → read its 2M eval; s7 seg2 home → `kernel2 7 3` from the clean worktree (`JACK_REPO=/data/wt/jack-clean`). Row read in KNOWN_ANSWERS when all five are home.

### J14 — Brain A and the ear: Colab is open, so the timing cells go now
part: His brain (A) + Ear · stage: 1 · step: 3 (reproduce) · prior: §2, §7; `docs/research/prereg/a2_size_s.md`; `KNOWN_ANSWERS.md` rows A1–A3, A2-v1S, N7-kws; rulings 67, 78, 79, 86, 96
prior: A1 PASS, A3 PASS, A2 MISS under band at 1m/100K (in band at 200K); 1m rows match official v2 on 67/73 settings. A2-v1S: MISS ABOVE band, 6/6 seed-points — and the arm was not the published size (encoder + decoder 2×512 vs official 5×1,024). Ear: BC-ResNet-1, published 96.9 ± 0.30; ours VOID at 182 s an epoch.
**Do.** (a) **STILL OWED — the file does not exist.** Freeze
`docs/research/prereg/a2_size_s_enc.md`, row **A2-v1S-enc**:
`v1s_diff.MATCHED_ENC` (the 4 keys, 53/56), same task, same control, seeds
10–12, one extension 13–14, bands unchanged, score window = the band's own
{P−20K, P−10K, P} (the frozen (0.9P, P] number printed beside, not read),
pins as A2-v1S. Reading written in first: in band → the build is right at
the published size and the 1m miss is size; out → diff against official
`ec526b9` setting by setting. Correct "Matches; nothing to name" in the NEW
file, not the frozen one.
(b) Its timing cell on **Colab T4 today** (ruling 96). A seed ≤ 3 h → the
three seeds go on Colab one at a time this week; longer → the seed saves
and resumes through `bbd54926`'s checkpoint, or waits for Sat 11-28 on
Kaggle. > 9 h a seed anywhere → INBOX, no seed.
(c) Ear (ruling 79): the timing cell on Colab T4 — the box's stages on
CUDA with synchronize, plus one arm with the noise held on the device
(suspect: a 4.26 MB noise file copied to the GPU per mixed clip). One seed
must fit one session with 20% to spare, or it is an INBOX line. Row N7-kws
stays VOID until a whole seed is home.
(d) Text only: `prereg/pilot_a.md` to ruling 62's terms — brain A on his
world's 128-ray eye, the three J22 cells as its arms (sighted, blind,
shuffled) — with `bbd54926` named as its resume path.
serves: M3, M5 · PLAN ready-list "brain A reproduced"; "Ear" step 3 · standing objective 3 · philosophy 3, 4, 6.
**done when:** `a2_size_s_enc.md` is frozen with its timing read, the ear has a GPU stage timing and a seed that fits a session, and `pilot_a.md` is drafted.
state: 11-21: A1 PASS (782 @100K, 984 @200K), A3 PASS (956), A2 MISS @100K (393.3 vs 438–518). A2-v1S frozen `8f13ab28`, 3 seeds: 100K mean 610.0, 200K 835.3, control valid → MISS ABOVE. Ear VOID 11-21 (10.1 T4-h a seed); (c) box half DONE `c34634a5`: decode + noise loop < 20 s of the T4's 182 s epoch → NEXT the CUDA timing cell.
state: 11-24 15:25 (a) DONE: `prereg/a2_size_s_enc.md` FROZEN. (b) timing cell died twice on Colab T4 before any rate (17:14 inductor rc 1, 17:27 SEGFAULT rc −11; files `brain_a/rows/v1s_enc_timing.fail_*.json`); cause read from VERSIONS: Colab torch 2.14.1+cu130 vs Kaggle's size-S cell 2.6.0+cu126 → `bcfbd0e6` re-pins torch 2.6.0 on any backend (13/13).
state: 11-24 19:30 (b) timing READ `381b83b9` (prereg addendum): attempt #5 rc 0 on Colab T4, torch 2.6.0 pin HELD; 2.76 grad-steps/s, 1,035 env-steps/s → **5.0 T4-h a seed** (v1S was 4.26) → rule-2 band 3–9 h. Colab resume is NOT wired: `bbd54926` lives in `in_world.py`, `rows.py` has none, never crossed a gradient step. NEXT (b): seeds 10–12 on Kaggle GPU from Sat 11-28 00:00, one seed per session (`rows.py` A2-v1S-enc kernel, est 5 h + cushion), 15 of W48's 30 h — unless a slot first wires resume into `rows.py` and shows it bit-continuous across a gradient step on Colab.
state: 11-24 19:10 (d) DRAFTED `docs/research/prereg/pilot_a.md` (text, binds nothing): A on row128 (298 numbers, measured), J22's three cells (sighted / blind / shuffled, s923, 100k), late meals/life from 50k, bar 0.87 = B's best cell; gate = A2-v1S-enc in band + timing cell + Director freeze; `bbd54926` named as resume path (it never crossed a gradient step — debt 4). 7 `OPEN:` lines for the sitting; 5 build debts, no code yet. Left of J14: (b) read, (c) ear cell.
state: 11-24 20:15 (c) READ (KNOWN_ANSWERS N7-kws addendum): per clip on Colab T4 authors 5.19/1.11/0.01/0.28 ms (decode/augment/mel/step) → 13.5 h a seed; noise on device augment 1.11→0.25 (−31.7 s an epoch) → 8.9 h. DECODE is 79–88% of the epoch (17× the box's), the T4 idles ~94%. Fit ≤ 10 h in Kaggle's 12 h: authors as written 10.2 h → does NOT fit; noise-on-device ≈ 8.4 h on Kaggle by estimate, but it is a wrapper change to the frozen 'unmodified' rule and 4 runs ≈ 34 T4-h > a week → INBOX, no seed. Left of J14: (b)'s seeds Sat 11-28.

### J23 — Brain B's reacher-hard row, once more, on the reference's test (ruling 94)
part: His brain (B) · stage: 1 ("each brain reproduces published numbers") · step: 3 (reproduce) · prior: §1b, §21; `KNOWN_ANSWERS.md` B-6 addendum 11-24; rulings 70, 81, 94
prior: published SBX/SimBa-protocol SAC, reacher-hard 905.7–1000. Ours, LIBRARY arm, same 5 models: 824.9 on our 10 fixed episodes, 886.1 on 100 fixed, **908.3 on 100 re-seeded (bootstrap 888–927)** — the miss was mostly the test's size. Swingup-sparse 541.1 under all three tests: B's own (one frozen explorer, one win-then-lose).
**Do.** Rule first in `KNOWN_ANSWERS.md` as **Row B-6c-2**, the one
re-registration ruling 70 allows: LIBRARY arm, recipe unchanged, **fresh
seeds 5–9**, 150k, score = mean of 100 re-seeded episodes at the last mark,
band unchanged (905.7–1000), no extension, control as row 6. Then one
Kaggle CPU kernel (rank 4), models saved at every mark. Rows 5, 6a, 6c stay
named misses whatever this reads; B's stage-2 gate (`stage2_a_vs_b.md` 2b)
gains one line with the result. 6a is not re-run: its two faults go to
`docs/FORGE_POOL.md` as one line each with §21 as prior (text).
serves: M3 · PLAN step 3 · philosophy 3, 4.
**done when:** Row B-6c-2 is in `KNOWN_ANSWERS.md`, in band or a miss, by its frozen rule.
state: 11-24 18:35 rule FROZEN `07df7d63` (Row B-6c-2 in KNOWN_ANSWERS: lib, fresh s5–9, 150k, stream-100 at 150k, band 905.7–1000, ctrl s0–2, no ext). Kernel + box scorer `0c2b7884` (`brain_b_dmc b6c2-kernel` / `b6c2-score`), planted verdict test PASS.
state: 11-24 22:05 kernel HOME, 48/48 models in `/data/jack-logs/brain_b/b6c2/`. Kernel's own 10-ep @150k: lib s5–9 965.5 / 960.2 / 861.0 / 978.3 / 950.7, ctrl 1.5 / 0.2 / 1.7 (NOT the row's reading).
state: 11-24 22:37 scorer ALIVE (pid 2189633, 26 CPU-min, ~8 min a cell; log `/data/jack-logs/brain_b/score_b6c2.log`; per-cell `score_*.json` resume). Stream-100 @150k: lib s5 956.1, s6 952.8, s7 845.9 (kernel's 10-ep s7 861.0). In band needs s8 + s9 mean ≥ 886.9. 5 cells left (s8, s9, ctrl s0–2).
state: 6a's two faults → `FORGE_POOL.md` § Faults DONE (text, not candidates; pool check 15 ok). `1db139a1` silenced dm_control's DeprecationWarning flood.
state: DONE 11-25 00:20 — Row B-6c-2 READ: **lib REPRODUCES** reacher-hard, stream-100 @150k s5–9 956.1/952.8/845.9/962.6/951.7 = **933.8** in 905.7–1000, ctrl 5.8/1.3/7.6 VALID; replay_gap 0.0 at 150k on 8/8 cells. KNOWN_ANSWERS B-6c-2 READ + gate 2b line in `stage2_a_vs_b.md`.

### J21 — Brain A on the Mac: the port table's next rows
part: His brain (A on the Mac) · stage: 1 (ready-list "first-day speed test for the Mac") · step: 2 (combine) · prior: `docs/research/PUBLISHED_EVIDENCE_2026-11-15_PORT.md` step table; `experiments/brain_a/README.md`
prior: nobody has published Dreamer on Apple hardware. Rungs 1, 2, 3, 6, 7 PASS here eager and compiled and on the PyPI wheel (Linux x86); rungs 6/7 at TOY size; row 2.1's size1m golden PASS.
**Do, box only (< 15 CPU-min a run, nice 19), one row a slot, in the
table's order:** rung 7 at `size1m` against the g1v1 golden (prerun's
sampler call skipped — say so in the docstring), then the world-model loss
rungs (prediction, dynamics and representation losses with free bits; then
the full world-model loss on one golden batch). Each row: NumPy 64-bit
reference from the official code first, tests committed while they FAIL,
then the module until they pass eager and compiled; "ours beside theirs"
in the docstring. No training, no speed claims. One wheel re-check kernel
(< 1 h, rank 0) after every third rung.
serves: M3 · PLAN "the decisive test … on the Mac mini" · philosophy 5, 6, 7.
**done when:** rung 7 at size1m and the world-model loss rungs pass, test names in `experiments/brain_a/README.md`.
state: rung 7 size1m DONE `b79e04e8`; rung 8a (dyn + rep, free bits) DONE `80b603bd` (refs 16/16 `a3fd3c60`; size1m init KL 2.45–11.63 never meets the 1.0 floor, so a free 7.5 pass tests the max()).
state: 11-25 rung 8b DONE `f37d8afd` (refs `ac9a3074` 15/15; official rewhead outscale 0 → zero logits at init, so compares run on seeded noised params too).
state: 11-25 01:27 rung 8c (decoder VECTOR losses, walker proprio keys) DONE `3c5b0b63`: refs vs official Decoder 24/24 `971816b9`, tests failing `963d32f1`, `mlx_maths.dec_vec_loss` 2/2 eager+compiled (5.0e-6), planted 6/6. Scar: decoder joins STOCH first (feat2tensor is deter first). Categorical + image branches not ported.
state: 11-25 01:40 rung 9 (FULL world-model loss, L240 sum(mean×scale), scales read from pinned configs.yaml) REFERENCES 29/29 HOLD `eb440755`: total 179.2374 init / 231.3479 noised vs official-composed (rel 7.5e-9); planted 4/4 ×2. Forward-BLIND: dyn↔rep scale swap (same value) → port tests owe a GRADIENT check.
state: DONE 11-25 03:30 `e0a2860c` — rung 9 PASS: `mlx_maths.wm_loss` 2/2 eager+compiled (tests failing first `66cfcc16`): total rel 3.4e-7 vs golden; GRADIENT routing d/d prior = 1.0×FD, d/d post = 0.1×FD (el 3e-8); planted 5/5 incl. the forward-blind dyn↔rep swap. README row. Wheel kernel `wheel_kernel kernel s1m` (7 size1m/8a/8b/8c) RUNNING `jack-cpu-1795577221` (`48184ec1`), log `/data/jack-logs/mlx/launch_wheel_s1m.log`; verdict lands in `/data/jack-logs/mlx/wheel_s1m.json` — next JACK slot reads it. Rung 9 not yet on the wheel.
state: 11-25 03:40 wheel s1m READ: **WHEEL PASS** (mlx 0.32.4 wheel, Kaggle CPU x86_64): rung 7 size1m/8a/8b/8c 2/2 eager+compiled each, planted 4/4/6/6 caught; README row.
state: DONE 11-25 05:09 — rung 9 wheel set `9` (`5e98aa3b`, `…583306`) **WHEEL PASS**: RUNG 9 2/2 eager+compiled, planted 5/5 caught; `wheel_9.json`, README row. Done-when met: rung 7 size1m + 8a/8b/8c/9 pass here AND on the wheel.

**Lane `forge`** reads `docs/FORGE.md`; its F3 is J18. Lane `lab`: a
candidate for brain B enters through `docs/FORGE_POOL.md` with its prior;
probes of other seats stay in `LAB.md`. **The lab's hindsight eye (rulings
84, 98): ENTERED, NOT launched under a learner** — it waits for J22's
branch and the `RESEARCH:` prior below. When built it is `live_eye.cell`'s
table: empty at birth, unlearning on, the give-up on (target ray no shorter
in 50 decisions → babble 16). Fed-then-weaned is DEAD at its pilot (no).
**Parked or closed:** `D1.0`, `T2.01` (→ J10), `LF.06`, `LC.09`, `CR.01`,
`LF.04` — "to earn again" on the chosen brain (ruling 20). `NE.01` → stage 3
(ruling 35). `PS.02`/`.05`/`.06`/`.08` → stage 3 (ruling 42). `W1.04` →
re-read at the next sitting (ruling 41). The lab's place key → stage 3
(ruling 63). Needs-hidden row → stage 3 (ruling 80).
**Next up, in order (idle slots; box < 15 CPU-min, text unless said):**
1. Ruling 91: the resumed-against-unbroken twin on the SURVIVAL stack's own
checkpoint path (box-minutes, project venv, no ledger): identical, or the
differing item named in one INBOX line.
2. `j17_late_meals`: a `late_from` beyond the run's length raises instead
of returning nan (planted test).
3. `docs/research/prereg/stage2_a_vs_b.md`: both brains' gates brought to
`KNOWN_ANSWERS.md`'s new standing table, one line a row.

---

## SYSTEM — work on the machine (max 3 open)

*Order of work: **S11** (the disk is at its floor) → **S16** → **S15**. A
fix a RUNNING Jack kernel needs goes first (ruling 54).*

### S11 — Disks: `/data` is AT its floor (15 GB)
part: (machine — a dead machine outranks everything) · stage: — · step: —
**Do.** (1) `experiments/gpu.py` makes job folders with
`tempfile.mkdtemp(dir="/data")` (three call sites) and nothing removes
them: ~136 `/data/tmp????????` folders remain. Make them under
`/data/tmp/gpu/`; remove a folder once its outputs are harvested; reap the
existing ones whose `out/` files are byte-identical to a copy under
`/data/jack-logs` (compare before deleting; list what was kept and why).
(2) `/data/caches` (18 GB, this project's): `uv cache prune`, `pip cache
purge`, HF/XDG entries older than 14 days NOT read by stt, the ear test or
tradelab — import test of jack-std and the project venv before and after.
(3) `/data/jack-logs/b_life_mujoco` (1.2 GB in two days): keep the newest
checkpoint per run and every result file; say the rule in the state line.
Never `/data/tools`, never result files, never another project's directory
(WorldTwin's database lives on `/data`), no new cron.
serves: philosophy 5, 6.
**done when:** `/data` has ≥ 25 GB free with before/after in the state line.
state: `/` 4.31 → 5.4 GB (`prune_cli_versions.sh`, `~/.cache/uv`, `git gc`). `/data` 39 GB (11-20) → 15 GB (11-24).
state: 11-24 15:30 SYSTEM `d72239d9`: `/data` 18 → 20 GB. (1) gpu.py → `/data/tmp/gpu`; `reap_gpu_tmp.py` on regate's cron reaped 106/137 (byte-identical copy or <1 MB copied to `jack-logs/gpu_jobs`). (2) pip purge 0.9 GB, uv prune; venv imports ok. (3) b_life_mujoco: newest ckpt per run kept, L0/ P_phys_0/ gone (0.4 GB).
state: KEPT, needs a ruling: 4 tmp folders 2.8 GB (A2-v1S Dreamer ckpts 60k–90k ×3 runs ≈1.4 GB + B/control.npz 0.5 GB, no copy anywhere); HF models >14 d (1.2 GB: SmolLM2-135M/360M, MiniLM, bge) are read by LG/ME.11.D/T2.10 specs, so kept.
state: 11-25 03:15 SYSTEM: **done-when met** — `/data` 24,650 MiB free = 25.8 GB (JACK's log filter `1db139a1` bought the last 4). Nothing further removed. Director: retire S11. (t0_24's test still mkdtemps in `/data`, spec file, left alone.)

### S16 — A slot with nothing to do must cost nothing (ruling 97)
part: (machine — cheaper; the shared meter stopped the team 45 h) · stage: — · step: —
**Do.** A no-model pre-check in the worker launcher, lane `jack` only: if
every Kaggle session is RUNNING, no kernel or detached box job has
finished since the last slot's commit, and `docs/ORDERS.md` was not
changed since then by the Director, the slot exits before a model starts,
logged `SKIPPED-IDLE` with the three readings. Any reading it cannot take
(Kaggle unreachable) → the slot RUNS. At most 3 skips in a row, then a
slot runs regardless (box-only orders such as J21 and J22's build must
never starve). Planted tests: all-busy skips; one finished kernel runs; an
unreadable Kaggle runs; the 4th skip runs. `usage_attribution.py` counts
SKIPPED-IDLE apart from dark slots. Do not touch the usage stop, the pace
switch or `.paused`.
serves: philosophy 5, 6 · the machine number "slots lost to budget".
**done when:** the four planted tests pass and the first real SKIPPED-IDLE line is in `ladder.log`.
state: 11-24 21:55 SYSTEM `bd43f84b`: `scripts/idle_skip.py` wired into `ladder_loop.sh` after the pace gate, lane jack only; planted 8/8 (the four ordered + free session, finished box job, ORDERS edit, no record). `usage_attribution` counts SKIPPED-IDLE apart (selftest P6c).
state: 11-25 03:15 SYSTEM `a426fe45`: six jack slots 00:22–01:37 RAN on "unreadable" — the pre-check guessed slug `int(receipt ts)`, but gpu.py's t0 is a moment later (…428 was really …429). Now resolved from `kaggle kernels list --mine` (0–600 s window); planted 10/10, live 4/4 resolve RUNNING. Open: first real SKIPPED-IDLE line in `ladder.log`.

### S15 — The detached lane's books: a cap that starves, a meter that bills waiting
part: (machine — truthful gates) · stage: — · step: —
**Do.** (1) `coverage` exits 2 on FAIL-UNOWNED = 1: the worker of 11-22
reads it as `SO.01`, the auditor as `T0.34`. Print which, from the ledger
row and `regate.log` (do NOT re-run a spec to look — running writes the
ledger); name the failing clause; fix by cause or own it in a state line.
Also name the spec that took the ladder 110 → 109 between 11-22 and 11-24.
(2) `launch_detached.sh`: a Kaggle WATCHER is not a computing job — the
2-job cap counts only jobs using ≥ 0.10 cores; the poller cap counts GPU
and CPU watchers apart and never refuses a box job. Planted tests.
(3) `gpu_budget.json` billed `jack-ladder-1795268291` 10.02 h for a 5.1 h
run: bill Kaggle's own run time (RUNNING → COMPLETE), not queue time;
correct W47's line with the arithmetic shown. No ceiling is raised. The
planted test becomes a clause of `T0.12` (ruling 91).
serves: philosophy 4, 5 · auditor findings 1, 5 · standing objective 3.
**done when:** the unowned FAIL is named and fixed or owned, the planted tests pass, and W47's GPU line shows run time.
state: (new 11-22; no SYSTEM slot has reached it. 11-22 09:30 SYSTEM: the unowned FAIL reads SO.01, not T0.34 — unverified.)

**Next up:** (0) S13 (a): reproduce `scripts/test_lane_guard.sh`'s FAILs of
11-20 on a clean tree; fix by cause or name why the test is wrong.
(1) **S9** as written 11-20 (`git show 8fde84f5:docs/ORDERS.md`): left
`T0.17`, `T0.23`, then DP.05 HR.5 LT.02 T0.35 T3.09 T4.02 W.2 W1.00.
(2) `gpu.py`: torch pin 2.6.0+cu126 in the Kaggle preamble; the TPU line
removed; the Colab preamble pins the box's jax/flax/optax trio.
(3) `XL.00` re-buy with a planted positive (stale 78 d, in front of 19
specs). (4) **S6** step 3 — the video of the latest recorded life.
(5) `run` tooling says which venv a test needs. (6) `prereg.judge()` door
for unequal arms. (7) Ruling 91: `T0.04` gains the world-state clause;
every ledger row records its mujoco version. (8) **S8**, **S4**, `T0.13`'s
dead clauses, `verify` VENUE 266 vs 256.

---

## INBOX — one line each, workers add, the Director empties
(emptied 11-24 12:26 — every line of 11-22/24 is ruled in 92–100 or became J22, J23, J18, J10, S11, S16, S15 or Next-up.)
- RESEARCHER 11-24 16:18 (§26, the hindsight eye's prior): the experiment is UNPUBLISHED — nobody has fed a from-scratch learner a self-taught reward table as input. The record says to change three things before any learner run: (1) race it as E2c's cell with three arms — the sense's direction / a yoked table (must fail) / handed (ceiling) — plus a carried frozen table, because the only published self-taught channel that worked at 100k was frozen first (Transporter) and SAC is on record as unsafe above 20% wrong guidance (ours 26%); (2) one reset that keeps the replay buffer (early empty input scars a replay learner, 501 → 656); (3) in `live_eye`, suppression instead of deletion and a give-up that scales with his meal rate — biology does both, box-minutes, no learner. A residual on the reflex has the best small-budget record (8k–1M steps) but makes a hand reflex a part of him: END-FIT ruling needed first.
- FORGE 11-24 14:15 (J18 steps 1–3 done, `2eb2c0f4` lock): rung S read under the lock, 30/30 whole — N3 0.85, K1 0.81, K2 0.51 climb; baseline 0.035, control mean 0.10; 71.2/270 core-h. Three readings of mine to rule: tiebreak orders equal scores inside the cut only (never past baseline/twin/control); cost = one core per run for its kernel's duration; a kernel with no finished run is refused, a single lost run scores 0. Note: the ladder's printed reason for the first arm below the cut says "equal on the score" of itself (wording only; the file is locked, the scoreboard corrects it). Next forge slot: save_life in the kernel, then rung M (6 arms × 5 seeds).
- RESEARCHER 11-25 04:15 (§27; answers the Director's RESEARCH line of 11-24, which named no open order, so no `prior:` lines were written): (1) nobody on record feeds a raw ray row — every lidar goal-reacher uses ~16 closeness bins PER OBJECT CLASS (0 = nothing, nearest wins) into a plain MLP, and even then SAC needed 3e6 steps (PointGoal1 27.0) and a TD3 navigator was at 13 % at 100k: a 100k one-seed pilot cannot close an eye. (2) h4's ±6-ray roll is not a valid DrQ augmentation by DrQ's own definition (it is a ±17° turn with the action unmoved; inference, no paper tests a ring) — h4 tested a mis-built arm; and by my whole-run arithmetic its three cells are inside two standard errors (0.37 ± 0.13 / 0.58 ± 0.14 / 0.88 ± 0.29), so 'shuffled ate most' ranks nothing. (3) B-6a s3: α fell at exactly Adam's maximum rate (e⁻³ at 10k), so he froze RANDOM, not still; published SAC on this very task scores 377 with white noise and 730 with pink (20 seeds) — a zero seed is ordinary and the cure on record is correlated action noise, not an α floor. The forager is not this fault (α 0.012–0.053 in h4). Change: if the eye re-opens under B, rule it as a NEW eye (binned, per class), ≥ 500k, roll removed; put pink noise in the forge pool against the 6a-s3 wall; `conv_eye.py`'s docstring cites two DrQ settings the config does not have (encoder tau 0.05, walker 2e-4).
- J22 11-24 15:45 (JACK): h4 pilot NO CURE — conv eye + random shift (DrQ) still climbs 1.818 by the hunt ruler and the SHUFFLED twin ate more late (0.87 vs 0.56; blind 0.31); the J17 ruler calls the sighted critic held (0.50, lowest of three). Branch taken: B closed on the wide eye for stage 1, no h5. For the Director: J22 is DONE — retire/replace it in the ranking; the rulers' standing disagreement (climb vs Q÷SUP on small-base arms, h0 and h4) may want a ruling before any future eye row is read.
- JACK 11-24 18:35 (J14 b, machine): 3 of today's 5 Colab submits died on `colab run`'s assign POST ReadTimeout (120 s, CLI 0.6.0; 0.8.0 is out) AND the server assigned the VM anyway — an orphan `[?]` T4 session nobody stops; the next submit is then refused "session busy". `gpu.run_on_colab` should unassign listed assignments with no local session after a failed run (done by hand twice today via `state.client.unassign`).
- 11-24 20:15 JACK J14 (c): the ear does not fit one Kaggle GPU session with 20% spare as written (10.2 h a seed vs ≤ 10.0); holding the noise on the device (a launcher wrapper, same arithmetic) saves 31.7 s an epoch → ≈ 8.4 h, fits by estimate — rule whether that wrapper may join N7-kws's frozen rule, and the GPU-weeks for 3 seeds + twin (≈ 34 T4-h). The real cost is audio decoding (80%+, num_workers 0 in main.py).
- JACK 11-24 22:30 (for S11 + Next-up): two J19 re-eval logs held 1.78 GB each of one repeated dm_control warning → filtered to their 18 real lines, cause fixed in `make_env` (`1db139a1`), /data 21 → 25 GB; no other log > 200 MB. Next-up #1 (ruling 91) DONE `e9b26b44`: the survival stack resumes identical 200/200, four planted drops caught, torch's global RNG is never drawn — retire the item.

- JACK 11-25 03:30 (machine): forge's uncommitted `experiments/forge/r1_run.py` + `tests/test_forge_r1_carry.py` (02:38–02:53, no commit) trip every lane's Kaggle submit guard; I launched J10 s7 seg1 + the wheel kernel from a clean worktree `/data/wt/jack-clean` via `JACK_REPO`. Their completion receipts will land in THAT tree's `experiments/gpu_submissions.jsonl`, not main's — merge them back on harvest; do not remove the worktree while the two watchers live.

## DONE (kept one day)

- **J17** (`7ebe2b7a`): the hunt closed at branch (4) — h0 SimBa (Colab T4, 2.360 / twin 1.082), h1 AdamW, h3 split layer STILL RUN AWAY, h2 OFN no control; none cures on s923. J17-2 not launched. Row in `WORLD_CARD.md` § J17 hunt. Continues as J22.
- **J19** (`7ab52e6c`): re-eval of the saved twins — 6c in band on 100 re-seeded episodes (908.3), 6a out under every test (541.1). Continues as J23.
- **S13 (b)** (`e3e573e5`): the judge lock; planted harness edit, reader edit and re-hashed lock each refused (24/24).
- **Next-up 11-22, items 1–2:** the blind-arm paragraph (`WORLD_CARD.md` l. 692); "where each brain stands" table in `KNOWN_ANSWERS.md` (`d63a2e37`).
