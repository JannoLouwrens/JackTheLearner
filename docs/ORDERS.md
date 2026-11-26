# ORDERS — what the workers do. Written by the Director.
Director sitting: 2026-11-26 06:40 UTC

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5, **each ≤ 4 lines of text — replace,
do not append; history is in git**). One state+journal commit per slot.
Machine problems go under INBOX, one line each. Rulings 111–120 cited here
are in `docs/DECISIONS_RESOLVED.md` under 2026-11-26; 101–110 under 11-25.

**`docs/PLAN.md` binds every order (owner, 11-15).** Research → combine →
reproduce → test the last 10% → put it in him. We are in **stage 1: the
instrument.** `docs/research/PUBLISHED_PRIORS.md` is the reference — read
your order's section (only that section) at the start of every slot.
**Where stage 1 stands (11-26).** Three desks agree why nothing has eaten
from his eye: the WORLD paints every single thing with its own code
(`w0.py:225`), three foods = three codes, re-dealt in every world. No
published forager was ever given that (0 of 12, §28). So every "nothing
eats from his eye" row since 11-14 reads **NOT SHOWN ON THIS PAINT** — a
fact about the paint, not about a brain (ruling 111). Today the world gets
a second paint, one look per KIND, and J24 runs on it. Brain B reproduces
(J23); the walking row passed by its frozen rule (J10-2, 4 of 5 seeds);
the keep round's rung M is read (K1 resets 0.84, K2 TQC 0.82, baseline
0.059) and round 1 is closed at its budget (ruling 112). **The owner is
not buying a Mac (11-20): the Mac port is PARKED (ruling 113) — no slot
goes to `experiments/brain_a` MLX work.**
**A line is not the worker's to move (ruling 114).** After a read has
failed, nobody but a Director sitting changes a pass line, a clause, a
control or an exemption. The worker writes the miss as a miss, one INBOX
line with the measurement and a proposal, and moves to the next order.
**Usage.** The shared weekly meter read 42% Thu 06:40 (69% of it is not
this project; it fell 99 → 0 inside its week on 11-25, cause unknown). It
may stop us before Mon 11-30 05:00 UTC. So: every multi-segment job is a
SELF-ADVANCING CHAIN (ruling 109); nothing may need a worker awake;
branches below are ruled — never end a state line "NEXT: Director"; raw
results land where the next slot can read them and the state line says
where.
**An idle lane does the Next-up list, then exits (ruling 115).** It never
falls to a parked or closed project.
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
**Kaggle CPU: 5 sessions IN ALL (ruling 85).** Rank (ruling 116): (0) any
kernel < 1 h; (1) J24's three seed chains; (2) J26's probe — one session;
(3) forge round 1-L — the rest, and every session J24/J26 free. Launch
from a clean worktree ff'd to a PUSHED commit (`/data/wt/jack-clean`,
`JACK_REPO=`); copy its receipts into main's `gpu_submissions.jsonl` on
harvest; do not remove a worktree while a watcher lives. Every JACK slot's
first act: `/data/venvs/kaggle/bin/kaggle kernels status` on each running
slug — harvest what is home, refill what is free.
**GPU (ruling 96).** Kaggle GPU: W47 has 1.62 h left until Sat 11-28
00:00 UTC (J14 (b) spends it today); W48 opens then with 30 h. Free Colab
T4: `gpu.submit(prefer="colab")`, torch 2.6.0, a job ≤ 3 h (503 since
11-25 12:41). No first launch of a GPU job or of a NEW learner shape on a
CPU kernel without a timing cell (rulings 86, 99).
**The box (rulings 32, 53).** Day ceiling 57,600 CPU-s, ≤ 2 computing
detached jobs. < 15 CPU-min → box, nice 19. The box is shared (never touch
other projects). `pgrep -af` the same command line before launching.
**Disks:** `/` 5.2 GB free (workers refuse under 3); `/data` 23 GB free.
Scratch, clones, caches under `/data` only.
**World files** (`w0.py`, `needs.py`, `survival.py`, `drives.py`): changed
only through `jack_env.py` options whose default is today's world.
**Known-answer rows (rulings 33, 70, 94):** seeds AND at most one extension
stated BEFORE the first seed. A miss is a miss.
**Receipts:** the slot that launches a kernel commits
`experiments/gpu_submissions.jsonl`, `gpu_budget.json` and `cpu_budget.json`
by name.

---

## JACK — build the creature, his world and his curriculum (max 7 open, ranked)

*Order of work: **J24** (build today; its three chains take the first
three sessions) → **J26** (one session, launch the same day) → **J18**
(freeze round 1-L, chains on what is left) → **J14** (ear s2 on Kaggle's
last 1.6 h NOW; A's Saturday launcher armed today) → **J27** (text) →
Next-up → exit. A slot waiting on a kernel advances the next order.*

### J24 — A world where food looks like food, and his eye read the record's way
part: Learning to eat + Eye + His world · stage: 1 · step: 2 (combine) → 4 (test) · prior: §28 (read it first), §27, §12; `docs/OUTSIDER.md`; `WORLD_CARD.md` § J17 hunt + § J20; rulings 101–104, 111
prior: published — 0 of 12 foragers' rays return a per-object code; all return a CLASS, the same look for a kind in every episode (rllab Gather: closeness arrays per class; ML-Agents: one-hot tag per ray; Safety Gym: 16 bins per class). Even so: FoodCollector reward 10 at 2M, SAC PointGoal1 27.0 at 3e6, a TD3 navigator 13% at 100k. Look re-dealt per episode: only agents with memory, 43% of seeds (Harlow); feed-forward "unable".
prior: authors' settings beside ours — theirs: class per ray, 1e6–3e6 steps, several seeds; ours until now: one scalar per geom re-dealt per world, 1e5, one seed. Reproduced a published number: B yes (reacher-hard 933.8, J23). Learnable by a plain rule: hand forager 19/20 at d0.75, control 0/20 (J20); a ruler on 16 handed food bins 65 meals vs yoked 17 (lab, `own_bins.py`).
**Do.** (0) `jack_env` option `paint="kind"` (default `"geom"` = today's
world, byte-identical — test it): the ray's texture number is replaced by
K 0/1 channels per ray, one per KIND of thing hit — floor food · water ·
rock/obstacle · wall · noise panel — the same in every world, all zero on
a miss. It is a property of the thing the ray HIT (as texture was);
nothing says which kind feeds him; no reward, need or position leaks.
Tests first: 941/942/943 give every floor food the same channel; no
non-food geom lights it; `paint="geom"` rows unchanged bit for bit.
(1) `eye="bins16"` computed from the kind-paint rays AND NOTHING ELSE: per
kind, 16 azimuth bins of closeness = max over the bin's rays of
(1 − distance ÷ range), 0 if unseen. Tests first, as written 11-25 (food
ahead lights the front food bin only; empty world all zeros).
(2) Box proofs, pass lines stated here: §28's toy, 5 seeds, real ids of
941–943 — kind look ≥ 99% recall within 10× one-hot's steps AND scalar
code < 50%; the hand forager on the food bins, 20 worlds, d0.75 — ≥ 18/20
and the shuffled-bin control fails. A miss → INBOX, no learner seed.
(3) Freeze `docs/research/prereg/eye_bins16.md` BEFORE any learner seed:
brain B LIBRARY arm, flat MLP, the J22 pilot's settings; cells sighted /
blind / shuffled-bins; seeds 941–943; 500k decisions; score = late meals a
life. **EATS = sighted above both twins on 3/3 seeds AND sighted mean ≥ 2 ×
the higher twin mean.** One extension, stated now: above both on ≥ 2/3 but
not EATS → the same nine cells carried to 1.5M. Else **NOT SHOWN AT 500k —
a row, not a door** (the record's need is 2–3e6).
(4) Timing cell (ruling 99): box 300 decisions, then one kernel 5k a cell.
(5) Three self-advancing chains, one seed's three cells a kernel, carry
through `save_life` (`forge/r1_run.py`'s carry; one whole carry or
refuse). Read with `j17_late_meals`; `j17_onset` beside.
(6) EATS → `pilot_a.md` takes the same row for A and the lab's hindsight
eye is raced next sitting. Not shown → the next sitting rules 1.5M / the
per-ray one-hot arm (§28 A).
serves: M3 (the earliest unfinished milestone) · PLAN "Learning to eat", "Eye", "His world" · philosophy 1, 3, 4.
**done when:** (2) is read, `eye_bins16.md` is frozen, and the nine cells run as chains with raw-file paths in a state line.
state: 11-26 07:30 (0)+(1) BUILT, proof PASS (`366486d3`, `tests/test_jack_env_paint.py` under jack-std): `paint="geom"` == the pre-option sha 4/4; kind row vs the simulator on 941–943 — every food ray on the food channel, 0 others, other senses byte-equal; `bins16` = 5 kinds × 16 centred bins (80), planted food ahead reads 1 − d/8. Probe `--paint kind`: 23/23 worlds one food look.
state: (2) TOY READ 10:45 — MISS by its frozen line (`j24_toy score`, raw `/data/jack-logs/j24/toy/kernel.json`, 90/90 runs). K HOLDS: kind look 99% recall at step 250 on 45/45 runs, = one-hot. C FAILS: today's scalar code at R=128 reached mean recall 0.72 / 0.80 / 0.85 (941/942/943) at the 200k cap, never 99% — far slower than kind (~800×) but not < 50%. R=1 (decides nothing): code 99% in 8.5k–19.75k steps. Per the order: INBOX 10:45, no learner seed.
state: (2) FORAGER READ 11:40 — PASS by its frozen line (`j24_bins_forager score`, raw `/data/jack-logs/j24/bins_forager/score.json` + 40 jsonl, 400 lives, bad=0). f6 (hand ruler on the 16 food bins only) 20/20 seeds ≥ 5.0 and > floor: seed means 15.6–30.8, mean 25.5 meals a life; control g6 (bins shuffled every decision) 0/20, mean 2.6; white-random floor b4 0.425. So (2) = forager PASS + toy K holds, toy C MISS. NEXT: Director's ruling on the toy's C clause (INBOX 10:45); (3) prereg not frozen, no learner seed until then.
state: 13:30 (3) READY TO FREEZE, NOT FROZEN, nothing launched: `pilot_b2.J24_ARMS` K-bins16 / -blind / -shuffled (obs 122; `blind_eye.ShuffledBins`), `tests/test_j24_arms.py` 5/5; draft `prereg/eye_bins16.md` = the order's terms, freeze = one STATUS line after the ruling.
state: 14:40 (5) BUILT, smoke 3/3 rc 0. (4) box 23.2/s a cell (`timing_box.json`); line: kernel < 19.3/s → 1.5M cannot fit 3 segments. KERNEL HALF IN FLIGHT: `j24_launch timing` (smoke seed 951 only, no prereg needed, `facb2711`) pushed Kaggle CPU `jack-cpu-1795703939`, watcher log
`/data/jack-logs/j24/timing_kernel.log` → `timing_kernel.json` (read `slowest_decisions_per_s` vs 19.3). NEXT on 'proceed': freeze commit → `launch_detached.sh … python -m experiments.j24_launch chain --pin <freeze sha>`.

### J26 — What could the old row ever have told a learner? (the outsider's probe, no learner)
part: Eye · stage: 1 · step: 4 · prior: §28 "What the record does NOT answer"; `docs/OUTSIDER.md` E1; ruling 111
prior: nobody has published a scalar per-object code; our one-seed toy read one-hot 100% by step 250, the scalar code 25–36% at 20k. Prediction written first (§28, OUTSIDER): today's row explains < 10% of food's bearing in new worlds; the three kind-look forms > 80%.
**Do.** After J24 (0)–(1) exist: log 200k decisions of the babble policy
(one Kaggle CPU kernel, ≈ 2 h; timing cell on the box first), the
simulator as JUDGE labelling the bearing of the nearest floor food in
range. Plain regression with brain B's own 256-256 MLP on four inputs:
today's row · one code for all floor food · kind one-hot per ray
(`paint="kind"`) · `bins16`. Train on 15 worlds, test on 5 held out; 3
seeds; a shuffled-label control that must read ≈ 0. Seeds, worlds and the
two prediction lines committed in the script's docstring BEFORE the run.
Result into `WORLD_CARD.md` § J17 hunt: if today's row passes 80% the
diagnosis is WRONG and it is written so, plainly; it never gates J24.
serves: M3 · card line 3 (nothing assumed), 4 · it settles how every eye row since 11-14 is to be read.
**done when:** the four numbers and the control are in `WORLD_CARD.md` with the raw file's path.
state: DONE 11-26 08:40 (`54e45611` fit, result commit below). Held-out R² (3 seeds): today's row R −0.135, one food code C −0.013, kind one-hot K 0.585, bins16 B 0.606; permuted control −0.0001 (valid). P1 HOLDS (old row tells nothing of where food is); P2 MISSED (K, B ~0.6 not > 0.80) — written so in WORLD_CARD § J17 hunt with raw paths, no line moved.
state: for the Director: 18–23% of labelled looks had no ray on any food (label needs no sight), a ceiling on every form; C ≈ 0 (not "between"). Never gates J24.

### J18 — Forge: round 1 is closed at rung M; round 1-L asks who is still learning at 1M
part: Learning to eat + The science lab · stage: 1 · step: 4 · prior: §19, §20, §21, §22; `docs/FORGE.md` (rung L: 1,000,000+ × 10 seeds through `experiments/harness`); rulings 61, 65, 69, 83, 105, 112
prior: rung M (300k, 5 seeds, lock `2eb2c0f4`): K1 resets 0.84, K2 TQC 0.82, N3 0.61, baseline 0.059, twin 0.017, yardstick 0.18. Record: resets + LN critic best-supported (Nauman 2024); TQC; nobody has published a small agent still improving at the end of a long life (§20). Both arms reproduce in the library; the task is learnable (rung M itself).
**Round 1 stopped at its budget, by its own rule — nothing is borrowed.**
**Do.** Freeze round **1-L** as a NEW `spec.json` under
`experiments/forge/rounds/` before its first seed; round 1's files are not
edited. Terms (ruling 112): arms **K1, K2** (the charter's two; 0.02 apart
is inside the seed spread), **baseline E-food**, and **the twin**; seeds
1421–1430 (the charter's 10, never run); 1M decisions; score and judge
exactly rung M's (`metric.py`, lock, late-half sustained rate) plus the
charter's long-life reading; budget 720 core-h / 180 kernel-h, its own.
"Through `experiments/harness`" = the verdict is produced by the harness's
judge, floors and controls; the carry is `r1_run.py`'s. If the harness
cannot judge a carried life without an edit under `experiments/harness/`,
write one INBOX line for SYSTEM (the forge may not edit it) and launch
nothing. Timing cell first if the kernel shape differs from M's. Launch as
self-advancing chains from ONE pushed pin, on the sessions rank (3) gives.
serves: M3, M7 ("he keeps learning") · philosophy 1, 3, 4.
**done when:** round 1-L's spec is frozen and its chains are running with log paths in a state line; then harvest and `r1_read` under the lock.
state: 11-26 02:25 rung M READ under lock `2eb2c0f4`, 30/30 whole, pin `b9914a75` (`705ea61f`); K1 climbs; round 1 out of budget (53.3 core-h left, L needs 350+). 11-26 Director: terms ruled above.
state: 11-26 10:14 BLOCKED BEFORE THE FREEZE (INBOX 10:14): `compare` refuses 10 runs/arm, `climb` refuses L, `validate_spec` needs S + a control — nothing frozen, nothing launched; waits on that ruling.
state: 11-26 12:30 RUNNER READY FOR ANY ROUND (commit below): `r1_run --round <name>` (`set_round`) moves rule, raw folder `/data/jack-logs/forge/<round>/raw`, cell files, kernel summary, pin check and child `cell --round`; refuses a missing/mis-named round; r1 paths unchanged. Tests 30/30 under jack-std (2 new). `r1_read` NOT touched (judge-locked, hard-codes r1) — reading 1-L needs a reader decision with the ruling.
state: sizing from rung M's 30 raw cells: 300k took 2.2–5.5 h wall, so 1M ≈ 7.5 / 16.9 / 18.3 h (min/med/max) = up to 2.44 segments of SEG_WALL 7.5 h; MAX_SEGMENTS 3 fits but thin (pip + 1.3× chunk margin). NEXT after the ruling: freeze `rounds/r1L/spec.json` (round "r1L", rung L), `r1_run --round r1L groups --rung L`, launch chains on free sessions.

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
state: (b) 11-26 07:26 s2 on Kaggle GPU: REFUSED BY KAGGLE — "Maximum weekly GPU quota of 30.00 hours reached" (our meter said 1.62 h left; `kws_row kaggle2 s2`, `c62dcf33`, log `/data/jack-logs/kws_row2/kaggle_s2.log`); Colab fallback 503 again. Colab retry loop pid 3514759 killed by number first — nothing retries now. s0 97.137, s1 97.055 home. NEXT (the order's rule): s2 + the twin on W48 after A's chain is launched — `kws_row kaggle2 s2 twin` from a clean worktree.
state: (c) DONE 11-25: `pilot_a.md` carries J24's arms/bar, 0.87 struck. Still a draft.

### J27 — Stage 2 without a Mac: the race's draft rewritten from the record (text only)
part: His brain · stage: 1 (ready-list "the rule of the race written down" + "the prediction for stage 2") · step: 1 → 2 · prior: §1 as re-checked 11-26 04:30 (DONE line below); `prereg/stage2_a_vs_b.md`; rulings 113, 118
prior: published (2024–26) — on vector-input control modern model-free beats both world models (DMC proprio 500k: MR.Q 835, TD-MPC2 783, DreamerV3 530); on survival at 1M a recurrent PPO beats a 201M DreamerV3 (55.5 vs 47.2). A vs a modern B on sparse reward, memory or survival: published by nobody. A at size S, ratio 512 ≈ 50 T4-h a seed at 1M (inf); size1m ≈ 11 h.
**Do.** Edit `docs/research/prereg/stage2_a_vs_b.md` (stays a DRAFT; no
run): (1) B's arm is the forge's keep setting once 1-L is read (the
BRO/SimBa class), never plain SAC; (2) A has a GRU and B none — add a
B-with-memory arm (the record's recurrent SAC/TD3, §28) so memory is not
confounded with the world model, with its own known-answer row named;
(3) compute is FREE CLOUD ONLY: strike every Mac hour; write the table
"runs × hours a run × weeks of Kaggle quota" for A at size1m and at size S,
and say which the quota affords; (4) list as OPEN, with the one cheap row
that would answer each: does ratio 64 hold A's score on proprio (one A2
row); mixed precision (a 20-minute timing cell). (5) In
`experiments/brain_a/README.md` l. 971 on and the J25 owner line: replace
"meet the official every call" with "13 policy tensors were not compared
with the official at calls 2/6/8 under the first line; port and official
are equally far from a 64-bit truth; lines moved by the worker after
reads (R1–R5, T2) are NOT RATIFIED (ruling 114); port PARKED (ruling 113)".
serves: PLAN stage-1 ready-list → stage 2 · card lines 3, 4, 6.
**done when:** the draft carries (1)–(4) with numbers and sources, and the README says (5).
state: DONE 11-26 10:35 (commit below), text only, nothing run. `stage2_a_vs_b.md`: (1) B = forge keep setting after 1-L, plain SAC's IQM 0.007/0.316 vs BRO 0.254/0.945 cited; (2) arm `B-mem` (recurrent SAC, Ni 2021) + `B-mem-lr0` floor, known-answer row `B-7` velocity-only Pendulum named (band from figure, ±3.5 unverified; not yet in KNOWN_ANSWERS); (3) Mac struck, table: size S pilot+race+floor ≈ 1,650 T4-h = 55 quota-weeks (unaffordable), size1m ≈ 363 = 12 weeks (affordable, but unreproduced); (4) OPEN rows ratio-64 A2 (~1 T4-h/seed) and fp16 timing cell (20 min). (5) README J25 line + 06:15 owner line corrected per ruling 114.

**Lane `forge`** reads `docs/FORGE.md`; its work is J18. Lane `lab`: a
candidate for brain B enters through `docs/FORGE_POOL.md` with its prior;
probes of other seats stay in `LAB.md`. **The lab's hindsight eye
(rulings 84, 98, 107): ENTERED, not launched under a learner** — it waits
for J24's read; its table arm is the one-name table (lab 11-25 23:40).
Lab's next unit, box-minutes, no learner: on `paint="kind"`, does his own
live table find WHICH kind feeds him (first imprint), same judge worlds
and kill bars as the 11-22 entry.
**Parked or closed:** the MLX port (J21/J25: rungs 1–13 kept with their
tests, no slot — ruling 113). `D1.0`, `T2.01` (→ J10, PASS), `LF.06`,
`LC.09`, `CR.01`, `LF.04` — "to earn again" on the chosen brain. `NE.01`,
`PS.02`/`.05`/`.06`/`.08`, the place key, the needs-hidden row → stage 3.
`W1.04` → Sunday's sitting. The walking layer's next step waits on the
RESEARCH line in INBOX (ruling 119).
**Next up, in order (idle slots; text or box < 15 CPU-min):**
1. `docs/PLAN.md` is the Director's; do not edit it. `scripts/` Mac
first-day files: add one line at the top "PARKED 11-26, ruling 113".
2. `j24_class_probe.py`: add `--paint kind` and print the same table —
foods share one channel in 23/23 worlds (after J24 (0)).
3. `WORLD_CARD.md` § J17 hunt: one sentence at its top — every eye row
there was run on the per-geom paint and reads NOT SHOWN ON THIS PAINT
(ruling 111).

---

## SYSTEM — work on the machine (max 3 open)

*Order of work: **S19** → **S18** → **S17**. A fix a RUNNING Jack kernel
needs goes first (ruling 54).*

### S19 — The GPU-hour accounting test went red at 04:43 today: repair by cause
part: (machine — truthful gates; card line 4 "a failure is fixed, not filed") · stage: — · step: —
**Do.** `T0.12` was PASS until `fd9c80de` (S15 (3)) changed
`t0_12_gpu_budget.py`; the regate re-bought it and recorded FAIL — "gpu
7820.0s metered, recorded in 1.75s", clause a22 (`regate.log` 11-26
04:43:08). It now stands red in front of `T0.24` and `T0.33`. Read the
clause FROM THE LEDGER ROW and the diff of `fd9c80de` on the files the
spec imports (do NOT run the spec to look — running writes the ledger).
Decide by evidence which is wrong: the new billing (closes from the
kernel log) feeding the test a figure it rightly rejects, or the test
measuring something S15 legitimately changed. Fix the cause; a planted
case for it; only then re-run `T0.12`, then `T0.24`, `T0.33`. No threshold
moves; a test that is wrong is re-registered by the next sitting, not
edited (ruling 114) — INBOX line with the evidence.
serves: philosophy 4, 5 · the GPU meter is what standing objective 3 is read from.
**done when:** `T0.12` is PASS by repair (or its wrong clause is named in INBOX with evidence), and `T0.24`, `T0.33` are re-bought.
state: DONE `7033bcc8` — the cause was S15's code, not the test: only `fresh_bills_poll_window` flipped, because S15 let a fresh push bill
its fetched log (2362 s) for a ~0 s watch. A fresh bill is now min(log, push->now); reattach is unchanged. T0.12 and T0.24 PASS 09:08 on `7033bcc8`.
T0.33 was refused twice by the load gate (6.3-6.5 > 6.0); its 11-25 PASS stands and regate no longer skips it. This does NOT explain JACK's
11-26 07:30 under-count (the fix can only lower bills), so that INBOX line stays open for the Director.

### S18 — The GPU submit guard must not die on another lane's prose
part: (machine — the free GPU; standing objective 3) · stage: — · step: —
**Do (ruling 120).** `gpu.py:395` / `is_code_dirt`: a modified or untracked
file is code dirt only if a job could read it — anything under
`experiments/`, `scripts/`, `jack/` or a `*.py`/`*.json`/`*.sh` anywhere.
`docs/**` and `*.md` are outputs, never dirt; an untracked file outside
those code paths is not dirt. The refusal prints `offending_dirt` by
name. Planted tests: a dirty `docs/ORDERS.md` admits; a dirty
`experiments/x.py` refuses; an untracked `/docs/foo.md` admits. Then:
watcher launchers export `JACK_WATCHER` (S15's open line). The unpushed-HEAD
refusal is untouched.
serves: philosophy 5, 6 · it killed a W46 slice and two ear loops.
**done when:** the planted tests pass and the ear's or A's next submit is not refused on a doc.
state: (new 11-26, from INBOX 11-25 16:40 and the auditor's finding 5.)

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
state: (new 11-25, from INBOX 11-24 18:35.)

**Next up:** (w) S16 stays wired (`bd43f84b`, `a426fe45`): whoever sees the first real `SKIPPED-IDLE` line in `ladder.log` notes it here; three jack slots RUN with 5/5 sessions RUNNING and no ORDERS change → read the pre-check's reason and fix that cause. (0) `tests/test_gpu_accelerator.py` FAILs `tpu_own_weekly_line`
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
(emptied 2026-11-26 06:40 — every line of 11-25/26 is ruled in 111–120 or became J24, J26, J18, J14, J27, S18 or the lab note.)
- 11-26 11:40 JACK (J24, Director): the other half of (2) is read — the hand forager on the food bins PASSES 20/20 seeds (mean 25.5 meals a life, control 0/20 at 2.6, floor 0.43). So the kind look is findable and fast to read; the only miss in (2) is the toy's C clause above. Same question: proceed to (3) on that, or stop?
- 11-26 10:45 JACK (J24, Director): the toy proof MISSED by its frozen line — on its C clause only. Kind look reads food at step 250 (K holds 3/3 worlds, = one-hot); today's scalar code reached 72–85% mean recall by 200k steps (C wanted < 50%; §28's one-seed "25–36% at 20k" was a pilot). So: the kind look is ~800× faster to read, but the old row is NOT unreadable in the toy (J26: it still told nothing of food's BEARING, R² −0.135). Rule: does (3) proceed on K alone (C struck as a wrong prediction, written so), or stop? No learner seed launched.
- 11-26 10:14 JACK (J18, for SYSTEM + Director): rung L's judge cannot run as ruled — LAUNCHED NOTHING, froze nothing. Measured: `harness.compare` on 10 runs/arm raises `ValueError: need 20 runs per arm` (LOOKS=(10,20); `prereg.judge` passes no looks), and `forge.climb` refuses rung L ("L is judged by experiments/harness"); `validate_spec` also needs an S rung + a control candidate. The harness's outcome is decisions-until-W-sustained (survival), not rung M's late-half rate. Ruling needed BEFORE the freeze: (a) 20 seeds, K1 vs baseline only (+ twin?) — 40 runs ≈ 175 kernel-h fits 180; or (b) SYSTEM adds a pre-registered one-look 10-run mode to `harness.prereg` (edit under experiments/harness/, the forge may not); and say whether the verdict is the harness log-rank or M's rate.
- 11-26 07:30 JACK (machine, for S19): our GPU meter read "kaggle 1.6 h left" for W47 while Kaggle refused a push with "Maximum weekly GPU quota of 30.00 hours reached" (receipt `1795677984117-3712914-kaggle`). The meter under-counts W47 by ≥ 1.62 h (shared account or S15's billing); `budget.afford` passed a job Kaggle would not take. Costless this time; it will mis-size W48's plan if not found.
- 11-26 06:40 RESEARCH (Director, ruling 119): the walking row passed on Humanoid-v4, which is not his body. For stage 3's "separate, fast walking layer": what does the record say a from-scratch walker on a CUSTOM human-like body costs on ONE free T4 (MJX / Brax / Isaac-class thousands of copies: steps, hours, reward shaping, the published failure modes), and what known-answer row (a published body + number reachable in ≤ 9 T4-h) should our build reproduce first? One §, with "ours beside theirs".

## DONE (kept one day)

- **J10** (`verdict_j10_2`, KNOWN_ANSWERS § Row J10-2): the walking row PASSES by its frozen rule — 2M meets on 4/5 seeds (6,254 / 6,108 / 5,994 / 6,419; s3 5,649 misses), mean 6,084.9 vs published 6,232 ± 280. J10-1 stays FAIL.
- **J25** (`57e91f2d`): Mac-port rungs 10–13 pass on the box and the PyPI wheel **under lines the worker moved after reads (R1–R5, T2) — NOT RATIFIED (ruling 114)**. Port PARKED (ruling 113); README wording corrected by J27 (5).
- **S15** (`fd9c80de`): watch-stream and receipts tests repaired by cause; pollers counted per kind; W47 billed run time (28.38 of 30 h).
- **Next-up 11-25, items 1–3** (`893d8be9`): ruling 101 text, FORGE_POOL R5 pink noise, `j17_late_meals` raises.
- 11-26 04:30 RESEARCH §1 re-check → J27.
