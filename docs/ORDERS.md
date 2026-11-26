# ORDERS — what the workers do. Written by the Director.
Director sitting: 2026-11-25 06:42 UTC

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5, **each ≤ 4 lines of text — replace,
do not append; history is in git**). One state+journal commit per slot.
Machine problems go under INBOX, one line each. Rulings 101–110 cited here
are in `docs/DECISIONS_RESOLVED.md` under 2026-11-25; 92–100 under 11-24.

**`docs/PLAN.md` binds every order (owner, 11-15).** Research → combine →
reproduce → test the last 10% → put it in him. We are in **stage 1: the
instrument.** `docs/research/PUBLISHED_PRIORS.md` is the reference — read
your order's section (only that section) at the start of every slot.
**Where stage 1 stands (11-25).** Brain B now reproduces reacher-hard on
fresh seeds (933.8 in 905.7–1000, J23). The hand forager passes on the one
physics version and its control fails (J20). The keep round's rung S is
read and rung M (30 runs of 300k) is running on two self-advancing chains
(J18). The Mac port holds through the full world-model loss (J21). **What
is NOT settled: whether a learner can eat from his eye.** The conv-eye
pilot (J22) was one seed at 100k whose two eyeless twins differ by more
than the effect, and its shift was not a valid augmentation on a ring —
it closes nothing (ruling 101). The record builds a ray sense differently:
closeness bins per object class into a plain MLP, read at 500k–3M steps
(§27) → **J24, first in the lane.** Neither brain is cleared for stage 2.
**THE USAGE STOP IS NEAR.** The shared weekly meter read 58% at 06:37
(15% a day ago); the stop lands between tonight and Thursday and holds
until Mon 11-30 05:00 UTC. So, every slot: **(1) no free Kaggle CPU
session while a ready kernel exists; (2) every multi-segment job is
launched as a SELF-ADVANCING CHAIN (the forge's chain pattern, ruling
109) — nothing may need a worker awake to take its next segment; (3)
branches below are ruled — never end a state line "NEXT: Director";
(4) raw results land where Monday's first slot can read them, and the
state line says where; (5) a slot that can advance nothing exits at once.**
**One world (ruling 76): mujoco 3.17.0.** Every row about his world runs in
`/data/venvs/jack-std` (box) or a kernel pinned to 3.17.0. Learner tests
(`sbx`, `optax`) run under jack-std, not the project venv.
**Every new run in his world passes** `eye_frame="body"`,
`motor_frame="body"`, `food_floor_only=True`, `lazy_contact_audio=True`,
`native_substep=True`; brain B uses `ent_coef="auto"` (ruling 64); "eats"
reads the LATE half of the run (ruling 65). **An eye row is read by
eating — late meals a life against its own blind and shuffled twins, seed
by seed; critic rulers are printed beside it and decide nothing (ruling
102).** Learner runs in his world only through J24, J14 (d) and the
forge's frozen round.
**Stopped (rulings 10, 20, 23, 36, 59, 80, 92, 101, 107):** home-made
learners; forager cells on the 16-ray compass eye; world-rule races;
`D1.0`, `T2.01`, smell runs; the needs-hidden row; more critic cures or
encoders on the RAW ray row; a learner residual on the hand reflex (a
scripted reflex inside him — card line 2); new specs with no published
prior. A question with no prior is a `RESEARCH:` line in INBOX.
**Kaggle CPU: 5 sessions IN ALL (ruling 85).** Rank (ruling 108): (0) any
kernel < 1 h; (1) forge rung M — 2 chains; (2) J24 — the first 2 sessions
that free; (3) J10's remaining segments — what is left. Launch from the
clean worktree `/data/wt/jack-clean` (`JACK_REPO=`), ff'd to a pushed
commit; copy its receipts into main's `gpu_submissions.jsonl` on harvest;
do not remove the worktree while a watcher lives. Every JACK slot's first
act: `/data/venvs/kaggle/bin/kaggle kernels status` on each running slug
— harvest what is home, refill what is free.
**GPU (ruling 96).** Kaggle GPU opens Sat 11-28 00:00 UTC (W48, 30 h).
Free Colab T4 is open now: `gpu.submit(prefer="colab")`, torch pinned
2.6.0 (`bcfbd0e6`), a job ≤ 3 h. No first launch of a GPU job or of a NEW
learner shape on a CPU kernel without a timing cell (rulings 86, 99).
**The box (rulings 32, 53).** Day ceiling 57,600 CPU-s, ≤ 2 computing
detached jobs. < 15 CPU-min → box, nice 19. The box is shared (never touch
other projects). `pgrep -af` the same command line before launching.
**Disks:** `/` 5.4 GB free (workers refuse under 3); `/data` 24 GB free.
Scratch, clones, caches under `/data` only.
**World files** (`w0.py`, `needs.py`, `survival.py`, `drives.py`): changed
only through `jack_env.py` options whose default is today's world.
**Known-answer rows (rulings 33, 70, 94):** seeds AND at most one extension
stated BEFORE the first seed. A miss is a miss. Every NEW brain-B
known-answer row is scored on 100 re-seeded evaluation episodes.
**Receipts:** the slot that launches a kernel commits
`experiments/gpu_submissions.jsonl`, `gpu_budget.json` and `cpu_budget.json`
by name.

---

## JACK — build the creature, his world and his curriculum (max 7 open, ranked)

*Order of work: **J24** (build now — two sessions free about 14:00 UTC
today and its kernels must be READY) → **J18** (harvest) → **J10**
(harvest + chain s7) → **J14** (ear cell on Colab today; A's seeds
Saturday) → **J25** (box only, when nothing above can move). A slot
waiting on a kernel advances the next order.*

### J24 — His eye the way the record builds a ray sense: closeness bins per class, at the record's length
part: Learning to eat + Eye (+ His brain B) · stage: 1 · step: 2 (combine) → 4 (test) · prior: §27 (read it first), §12, §24; `WORLD_CARD.md` § J17 hunt + § J20; rulings 101–104
prior: published — every lidar goal-reacher on record feeds ~16 closeness bins PER OBJECT CLASS (0 = nothing, nearest wins) to a plain MLP; even so SAC took 3e6 steps (PointGoal1 27.0) and a TD3 navigator stood at 13% at 100k. Nobody feeds a raw ray row. Ours so far: 128 raw rays (flat, then conv), 100k, one seed — sighted 0.56 late meals a life, blind 0.31, shuffled 0.87: unresolved, not a verdict.
prior: authors' necessary settings beside ours — theirs: bins per class, 1e6–3e6 steps, several seeds; ours until now: raw rays, 1e5, one seed. Reproduced a published number: B yes (reacher-hard 933.8, J23). Learnable by a plain rule: hand forager 19/20 at d0.75, 7.9 meals a life, shuffled control 0/20 (J20).
prior: §28 (11-25, for the stop at (1)) — no published forager's ray returns a per-object code (0 of 12 worlds: class arrays or one-hot tags, the same look for a kind in every episode); where look→reward is re-drawn per episode only agents with memory have learned it (Harlow task: LSTM, 43% of seeds; feed-forward "unable"), and a memoryless MLP is on record nowhere. Known-answer: none published for a scalar-coded ray; our one-seed toy reads one-hot 100% by step 250, today's code 25–36% at 20k. Learnable from today's row by a memoryless learner: NO by the record (inf); with one look per kind: yes at 2M+ (FoodCollector), unknown at 500k.
**Do.** (1) `jack_env` option `eye="bins16"` (default = today's eye):
computed from the row128 rays AND NOTHING ELSE — for each object class the
ray row itself tells apart (read the eye; say in the docstring which
channel carries class), 16 azimuth bins of closeness = max over the bin's
rays of (1 − distance ÷ range), 0 when the class is not seen. If the ray
row carries no class signal, stop: one INBOX line. Tests committed first:
food dead ahead lights the front food bin and no other class; an empty
world is all zeros; nothing reads simulator state a ray did not return.
(2) Learnable proof, box, jack-std: the hand forager steering on the food
bins, 20 worlds, d0.75, with a shuffled-bin control. Bars unmoved (≥ 18/20;
the control must fail). A miss → INBOX, no learner run.
(3) Freeze `docs/research/prereg/eye_bins16.md` BEFORE any learner seed:
brain B LIBRARY arm, the J22 pilot's settings with the flat MLP; cells
sighted / blind / shuffled-bins; **seeds 941–943; 500k decisions each**;
score = late meals a life over the last half. **EATS = sighted above both
its twins on 3/3 seeds AND the sighted mean ≥ 2 × the higher twin mean.**
One extension, stated now: sighted above both twins on ≥ 2/3 but not EATS
→ the same nine cells carried on to 1.5M. Anything else reads **NOT SHOWN
AT 500k — a row, not a door** (the record's own need is 3e6).
(4) Timing cell (ruling 99): box 300 decisions, then one kernel 5k per
cell. (5) Launch as self-advancing chains carrying through `save_life`
(reuse `forge/r1_run.py`'s carry — deadline 7.5 h, one whole carry or
refuse), one seed's three cells per kernel, on the first 2 free sessions.
Read with `j17_late_meals`; `j17_onset` printed beside (ruling 102).
(6) EATS → the same row for brain A is `pilot_a.md`'s next line, and
J17-2 is re-frozen on this eye at the next sitting. Not shown → the row
stands as written and the next sitting rules the 1.5M question.
serves: M3 (the earliest unfinished milestone) · PLAN "Learning to eat" + "Eye" last 10% · philosophy 1, 3, 4.
**done when:** (2) is read, `eye_bins16.md` is frozen, and the nine cells are running as chains with their raw-file paths in a state line.
state: 11-25 07:10 (1) STOPPED BY ITS OWN STOP CLAUSE — the ray row carries NO class signal. `experiments/j24_class_probe.py` (box, jack-std, 23 pilot worlds incl. 941–943): texture = a hash of the geom ID (`w0.py:225`); each world's 3 floor foods read 3 DIFFERENT values among 29–34 other hittable geoms; 941 (.49 .86 .43), 942 (.86 .43 .80), 943 (.43 .80 .37); in 23/23 worlds another world's food value is a non-food geom here.
state: So no `eye="bins16"`, no forager proof, no prereg, no learner seed, no kernel taken (INBOX line 11-25 07:10). Steps (2)–(6) wait on a ruling that gives the eye a class to bin; until then the two sessions J24 was ranked for fall to J10's remaining segment (rank 3).

### J18 — Forge round 1, rung M: 30 runs of 300k are on two chains — harvest and read
part: Learning to eat + The science lab · stage: 1 · step: 4 · prior: §19, §21, §22; `docs/FORGE_POOL.md`; rulings 61, 65, 69, 83, 105
prior: E2c — on E-food the baseline reaches (4/5) and loses (10/10). Record: resets + LN critic best-supported (Nauman 2024); TQC; `n_steps=3`. Nobody has published a small agent still improving at the end of a long life (§20). Rung S (150k, read under lock `2eb2c0f4`): N3 0.85, K1 0.81, K2 0.51 against baseline 0.035, control mean 0.10.
**Terms frozen `594752d0`; the forge's three readings of 11-24 STAND
(ruling 105).** Rung M = E-food, E-food-lr0, K1, K2, N3 + yardstick C2 ×
s1411–1415, pin `b9914a75`, chains A (g0 g2 g4 g6) and B (g1 g3 g5 g7)
from `/data/wt/jack-clean`; logs `launch_M_chain{A,B}.log` there.
**Do.** Harvest only: the chains advance and carry themselves (≈ 26 h a
chain, home about Thu 11-26 08:00–10:00). A refused or failed group in
`CHAIN DONE` is re-run from the SAME pin. When all eight
`/data/jack-logs/forge/r1/raw/M_g*.json` are home: `python -m
experiments.forge.r1_read --rung M` under the lock, scoreboard updated,
receipts copied into main. Then rung L by the frozen rule, as chains. No
code change between rungs except what the frozen terms allow. If a chain
dies (its watcher gone and its last kernel COMPLETE with groups left),
restart it from the next unrun group, same pin, and say so.
serves: M3, M7 ("he keeps learning") · philosophy 1, 3, 4.
**done when:** rung M is on the scoreboard — one reading per candidate beside baseline, twin and yardstick — and rung L's kernels are launched or the round is closed by its rule.
state: 11-24 14:14 rung S READ `6b29c527` under lock `2eb2c0f4`. 11-25 06:16 rung M LAUNCHED (`41cc5d64` carry 13/13, `b9914a75` kernel side 26/26). Home: g0, g1, g3.
state: 11-25 14:38 chain B DONE with **g5, g7 FAILED in 5 s**: `gpu.repo_preamble` clones `--depth 50` and checks out the box HEAD; `/data/wt/jack-clean` sits AT the pin, now 51 commits behind main → `reference is not a tree` (kernel log `…617284`). Not the round's code.
state: g5+g7 RE-RUN from the SAME pin via a clean worktree at main `76422e0a` (`/data/wt/jack-forge-rerun`; forge/gpu/brain_b/env byte-identical to the pin; the job's own fetch-the-pin fallback takes over), log `launch_M_chainB_rerun.log` there, first kernel `…617532`.
state: 11-25 15:23 g4+g6 RE-RUN NOW (not waiting for chain A to fail them ~16:30): same pin, launched from main (HEAD's box code = pin's except the preamble fix `a1893299`), log `/data/wt/jack-forge-rerun/launch_M_chainA_rerun.log`, kernel `…620207` RUNNING past the clone. Chain A's own g4/g6 later: fail at clone (writes nothing) or Refused once home — expect `failed` in its CHAIN DONE, harmless. Home: g0 g1 g3; running g2 (A), g5→g7, g4→g6. Read rung M when all 8 home (≈ Thu).

### J10 — The walking layer: the row with enough seeds, life carried across segments
part: Body and movement · stage: 1 side job (stage 3's part) · step: 3 (reproduce) · prior: §16, §21; rulings 45, 72, 82, 95, 108
prior: §16 — SAC, Zoo entry, Humanoid-v4: 4,825 [3,784–5,866] at 1M (10 seeds), 6,232 ± 280 at 2M, each ONE unbroken run. Ours: Row J10-1 FAIL (1 of 3); the carry (`d39687f0`) makes a resumed run bit-identical to an unbroken one.
**Row J10-1 stays FAIL. J10-2: rule frozen `b609b142`: seeds 3–7, PASS
R ≥ 5,950 AND ≥ 12 m/1,000 steps on ≥ 4/5, 50-episode eval, no extension.**
**Do.** (1) Harvest s5 seg3 (`…580623`) — read its 2M eval (s6 is read: it
meets; tally 2 meet / 1 miss / 2 open). (2) s7 seg2 (`…587392`) → seg3 must
not wait for a worker: start a detached chain watcher for it today (the
forge's pattern; `pgrep` first) so `kernel2 7 3` goes the moment seg2 is
home. (3) When all five are home the row is read by its frozen rule into
`KNOWN_ANSWERS.md` — **a FAIL is written as plainly as a PASS** (s3 has
missed; one more miss and the row fails). Ruling 95's rename, if not yet
done, in the same commit. (4) A FAIL is not re-registered here: the next
sitting rules what the walking layer's stage-3 entry needs.
serves: M4 · PLAN stage 3 "a separate, fast walking layer is built and checked in parallel from now" · philosophy 1, 4.
**done when:** J10-2 is in `KNOWN_ANSWERS.md`, pass or fail, by its frozen rule.
state: **DONE 11-25 14:30 — J10-2 PASS by its frozen rule** (`j10_2_read` → verdict_j10_2): 2M meets on 4/5 (s4 6,254.3/17.16, s5 6,107.9/17.26, s6 5,993.5/17.92, **s7 6,419.2/20.81** `…600347`); s3 5,649.4 misses. 1M median 5,219.8 IN BAND. 2M mean 6,084.9 vs published 6,232 ± 280.
state: Written into KNOWN_ANSWERS §Row J10-2 (+ ruling 95 rename: the PPO pre-flight is "Row J10-PPO-pre"); s7 receipts copied from the worktree into main. Per (4) the next sitting rules the walking layer's stage-3 entry (Humanoid-v4, not his body).

### J14 — Brain A at the published size, and the ear with its plumbing fixed
part: His brain (A) + Ear · stage: 1 · step: 3 (reproduce) · prior: §2, §7; `prereg/a2_size_s_enc.md` (FROZEN + timing addendum `381b83b9`); `KNOWN_ANSWERS.md` rows A1–A3, A2-v1S, N7-kws (+ addendum 11-24); rulings 67, 78, 79, 86, 96, 103, 106
prior: A1 PASS, A3 PASS, A2 MISS under band at 1m/100K. A2-v1S: MISS ABOVE band — and the arm was not the published size. A2-v1S-enc: 5.0 T4-h a seed. Ear: BC-ResNet-1, published 96.9 ± 0.30; ours VOID — 13.5 h a seed on a T4 that idles 94%: decoding audio is 79–88% of the epoch (`num_workers 0`), the 9,000-weight net is not the cost.
**Do.** (a) **A's seeds 10–12** go on Kaggle GPU from Sat 11-28 00:00, one
seed a session, as ONE self-advancing chain launched by the first slot
that is awake then (or Monday's first, if the stop holds). Before then,
box work: wire the checkpoint resume into `rows.py` and show a resumed run
bit-continuous across a gradient step (test first) — with it a seed may
instead run on Colab in ≤ 3 h pieces this week.
(b) **Ear, ruling 106 — re-registered ONCE as Row N7-kws-2**, rule frozen
in `KNOWN_ANSWERS.md` before any seed: the authors' model, optimiser,
augmentation arithmetic, epochs and band UNCHANGED; plumbing only —
(i) every clip decoded once into a waveform cache, (ii) the noise held on
the device. Admission test, committed first: with one seed, the mel batch
fed to the net is identical (max abs diff ≤ 1e-6) with and without the
plumbing, for 3 batches. Then the timing cell on Colab T4 today. A seed
that fits one session with 20% to spare → 3 seeds + the control on Colab
or W48, as a chain. Does not fit → INBOX with the stage table, no seed.
(c) Text: `prereg/pilot_a.md` — strike the 0.87 bar (it was a shuffled
eye's number, ruling 103); A's arms and bar become J24's: bins16 eye,
sighted / blind / shuffled, EATS as J24 defines it. Still a draft.
serves: M3, M5 · PLAN ready-list "brain A reproduced"; "Ear" step 3 · standing objective 3 · philosophy 3, 4, 6.
**done when:** A2-v1S-enc's three seeds are launched or home; N7-kws-2 is frozen with its admission test passing and its timing read; `pilot_a.md` carries no B number as a bar.
state: (a) seeds 10–12 whole on Kaggle from Sat 11-28 00:00, one seed a session (`rows.py` A2-v1S-enc, 5.0 T4-h a seed, `381b83b9`). Colab-in-pieces CLOSED: sheeprl resume not bit-continuous (`j14_grad_resume_probe`), cause measured `0ac058e1` — no env state in the checkpoint by design, only a runner edit could fix it.
state: (c) DONE in text 11-25 07:35: `pilot_a.md` carries J24's arms/bar, 0.87 struck; gate 5 (J24's eye) NOT MET. Still a draft.
state: (b) N7-kws-2 FROZEN + ADMITTED `b1db7b5d` (Δmel 0.0 box + Colab T4); timing `5053a28d`: 1.38 h a seed, ≤ 1.54 h with valid → fits Colab 2.4 h.
state: 11-25 09:39 (b) chain RELAUNCHED (pid 2744743, log `/data/jack-logs/kws_row2/chain.log`; the 09:24 one stopped: assign POST ReadTimeout 120 s, log kept as `chain_0924_stop.log`). The timeout left an orphan T4 `[?]` (S17's bug again); `colab stop` cannot see it, server had released it by 09:38. Relaunch ran from MAIN (cwd not set; code at HEAD = `be507986` for experiments/ scripts/), s0 on VM `ladder-1795599545` at 09:39.
state: 11-25 13:25 **s1 HOME: test 97.055** (s0 97.137; band 96.3–97.5; `score2` NOT JUDGED, missing s2, twin). s2 on Colab assign 503 since 12:41 (Kaggle 0.0 h till Sat); detached retry loop `/data/jack-logs/kws_row2/retry_chain.sh` re-runs the chain every 30 min, max 12.
state: 11-26 02:38 old loop (pid 3298770) at try 11/12, Colab still 503; gives up ≈ 03:25. FOLLOW-ON QUEUED: `retry_after.sh` (pid 3514759, log `kws_row2/chain_retry2.log`) waits for it to exit, then runs the same 12-try loop on today's ration (no double submit). NEXT: harvest s2, twin → `score2` → KNOWN_ANSWERS N7-kws-2; if it too gives up (≈ 09:30), Kaggle GPU returns Sat.

### J25 — Brain A on the Mac: the behaviour-learning rungs
part: His brain (A on the Mac) · stage: 1 (ready-list "first-day speed test for the Mac") · step: 2 (combine) · prior: `docs/research/PUBLISHED_EVIDENCE_2026-11-15_PORT.md` step table; `experiments/brain_a/README.md` (rungs 1–9 PASS here and on the PyPI wheel)
prior: nobody has published Dreamer on Apple hardware. Ours: rungs 1–9 match the official code eager and compiled, to 3.4e-7 on the full world-model loss, gradients routed as the official ones (J21).
**Do, box only (< 15 CPU-min a run, nice 19), one rung a slot, in the
table's order:** rung 10 imagination rollout + λ-returns; rung 11 the
critic loss (two-hot, slow target); rung 12 the actor loss (return
normalisation by percentiles, entropy); rung 13 one WHOLE optimiser step
of the official optimiser on one golden batch (parameters after the step
match). Each: NumPy 64-bit reference from the official code first, tests
committed while they FAIL, then the module until they pass eager and
compiled, with a GRADIENT check where the forward value is blind (rung
9's scar) and planted bugs caught; "ours beside theirs" in the docstring.
One wheel kernel (< 1 h, rank 0) after rung 12 and after 13. No training,
no speed claims.
serves: M3 · PLAN "the decisive test … on the Mac mini" · philosophy 5, 6, 7.
**done when:** rungs 10–13 pass here and on the wheel, test names in `experiments/brain_a/README.md`.
state: **DONE 11-26 06:15 — rungs 10–13 PASS box + WHEEL.** Wheel 13 re-read under ruling T2 (`872a99af`, kernel `jack-cpu-1795670064`): 11/11 runs PASS, goldens sha256 = box; 13b-grads pol on the official's adv meets the published 1e-4 gate at calls 2/6/8 (global rel 2.6e-6/2.4e-6/2.5e-6), plants 8/8, control fails 4/4.
state: Receipt `/data/jack-logs/mlx/wheel_13.json`; written into `experiments/brain_a/README.md` (rung 13 wheel line). Limit: x86_64 PyPI wheel, not the Mac's arm64 binary; no training, no speed claim. Owner's no-Mac line (INBOX 11-25 11:58) still awaits the Director.

**Lane `forge`** reads `docs/FORGE.md`; its work is J18. Lane `lab`: a
candidate for brain B enters through `docs/FORGE_POOL.md` with its prior;
probes of other seats stay in `LAB.md`. **The lab's hindsight eye
(rulings 84, 98, 107): ENTERED, not launched under a learner** — it waits
for J24's read. Lab's next unit, box-minutes, no learner (§26 (3)): in
`live_eye`, suppression instead of deletion, and a give-up that scales
with his own meal rate; same judge worlds and kill bars as the 11-22
entry. When it is raced it is E2c's cell with §26's arms (the sense's
direction / a yoked table that must fail / handed / a carried frozen
table) and one reset that keeps the replay.
**Parked or closed:** `D1.0`, `T2.01` (→ J10), `LF.06`, `LC.09`, `CR.01`,
`LF.04` — "to earn again" on the chosen brain (ruling 20). `NE.01`,
`PS.02`/`.05`/`.06`/`.08`, the place key, the needs-hidden row → stage 3
(rulings 35, 42, 63, 80). `W1.04` → Sunday's sitting.
**Next up, in order (idle slots; text or box < 15 CPU-min):**
1. Ruling 101, text: `WORLD_CARD.md` § J17 hunt's h4 paragraph and
`conv_eye.py`'s docstring — "NOT SHOWN (one seed, 100k, twins 0.31 / 0.87;
a ±6-ray roll is a turn, not DrQ's shift)", never "does not learn"; remove
the two DrQ settings the config does not have (encoder tau 0.05, walker
2e-4).
2. `docs/FORGE_POOL.md`: one candidate line — pink (correlated) action
noise against the 6a-s3 wall, prior §27 (3) (377 white vs 730 pink, 20
seeds). Text; it enters at the next round.
3. `j17_late_meals`: a `late_from` beyond the run's length raises instead
of returning nan (planted test), if not already done.

---

## SYSTEM — work on the machine (max 3 open)

*Order of work: ~~S15~~ (DONE `fd9c80de`) →
**S17** → **S16**. A fix a RUNNING Jack kernel needs goes first (ruling 54).*

### S15 — DONE `fd9c80de` — Two tests are recorded FAIL, not stale: name them, repair them
part: (machine — truthful gates; card line 4 "a failure is fixed, not filed") · stage: — · step: —
**Do, in this order — (1) is today's.** (1) The ledger says FAIL for the
watch-stream test (re-run 11-22 08:49: under five frames a second at a
cost share of 0.95 — the ladder's lost point, 110 → 109, and END-FIT row
"Watched while he struggles"'s first gate) and for the detached lane's
receipts test (11-21 02:48, `detached_receipts_ok 0.0`). For each: print
the spec id and the failing clause FROM THE LEDGER ROW and `regate.log`
(do NOT run a spec to look — running writes the ledger); find the cause
(what changed in the stream path / the receipts path since its last
PASS: `git log` on the files the spec imports); fix the cause; only then
re-run the spec, and then the dependent "being watched does not change
him" spec, so `coverage` stops printing PASS-ON-DEAD-DEPENDENCY 5 vs 4.
A cause that cannot be fixed in a slot → a state line naming it, and the
spec stays FAIL. No threshold moves.
(2) `launch_detached.sh`: a Kaggle WATCHER is not a computing job — the
2-job cap counts only jobs using ≥ 0.10 cores; the poller cap counts GPU
and CPU watchers apart and never refuses a box job. Planted tests.
(3) `gpu_budget.json` billed `jack-ladder-1795268291` 10.02 h for a 5.1 h
run: bill Kaggle's own run time (RUNNING → COMPLETE), not queue time;
correct W47's line with the arithmetic shown. No ceiling is raised.
serves: philosophy 4, 5 · END-FIT row 5 · auditor finding 1 (twice).
**done when:** both FAILs are named with their clause and cause, each is PASS by repair or owned in a state line, `coverage` no longer exits 2 on the dead dependency, the planted tests pass, and W47's GPU line shows run time.
state: 11-25 09:15 SYSTEM `bb8c9106`: (1) DONE. SO.01 (watch stream: delivered 4.69 < 5 fps, code unchanged since PASS) — cause: software shadow + reflection passes were 2/3 of a frame, box load doubled it; spectator drops both (94→31 ms, control diff 0.035 ≥ 0.005), now ~20–24 fps. T0.34 (`detached_receipts_ok 0`, launcher_ok false) — launcher refused in ~1.5 s, reason discarded by the test; now recorded, a live seat/load refusal is VOID. SO.01, T0.34, SO.04 PASS; `coverage` exits 0.
state: 11-25 21:10 SYSTEM `cac00782`: (2) DONE. The 0.10-core busy/poller split already stood; now pollers count per kind (gpu/cpu, 6 each — ruling 5's 6 kept per kind, Director may set others) and bind only a launch with `JACK_WATCHER=gpu|cpu`; a box job is never refused by it. Live kind = declared, else `jack-cpu-` in its log (both forge chains read cpu). T0.33, T0.34 PASS (planted: own-kind refused, other kind + box admitted).
state: 11-26 03:20 SYSTEM `fd9c80de`: (3) DONE — **S15 DONE.** Kaggle's own log for `…268291` spans 18,310 s = 5.0862 h; billed 10.0189 h; 4.9327 h of queue refunded on its row (reason kept). W47 kaggle used 33.31 → 28.38 of 30 (1.62 h free). Bills now close from the kernel log always, else from first RUNNING; planted 4/4.
state: left open for the Director: watcher launchers should export `JACK_WATCHER` (undeclared watchers escape the cap, never block); other W47 kernels were not re-checked for queue time.

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

### S16 — A slot with nothing to do must cost nothing (ruling 97)
part: (machine — cheaper; the shared meter is the scarce thing) · stage: — · step: —
**Do.** Built and wired (`bd43f84b`, slug fix `a426fe45`). Open: the first
real `SKIPPED-IDLE` line in `ladder.log`. With 5/5 sessions running and
chains self-advancing, slots between harvests should now skip — if three
consecutive jack slots RUN with all five sessions RUNNING and no ORDERS
change, read the pre-check's printed reason and fix that cause (planted
test for it). Never touch the usage stop, the pace switch or `.paused`.
serves: philosophy 5, 6 · the machine number "slots lost to budget".
**done when:** the first real SKIPPED-IDLE line is in `ladder.log`.
state: 11-24 21:55 SYSTEM `bd43f84b`: `scripts/idle_skip.py` wired into `ladder_loop.sh` after the pace gate, lane jack only; planted 8/8 (the four ordered + free session, finished box job, ORDERS edit, no record). `usage_attribution` counts SKIPPED-IDLE apart (selftest P6c).
state: 11-25 03:15 SYSTEM `a426fe45`: six jack slots 00:22–01:37 RAN on "unreadable" — the pre-check guessed slug `int(receipt ts)`, but gpu.py's t0 is a moment later (…428 was really …429). Now resolved from `kaggle kernels list --mine` (0–600 s window); planted 10/10, live 4/4 resolve RUNNING. Open: first real SKIPPED-IDLE line in `ladder.log`.

**Next up:** (0) S13 (a): reproduce `scripts/test_lane_guard.sh`'s FAILs of
11-20 on a clean tree; fix by cause or name why the test is wrong.
(1) **S9** as written 11-20 (`git show 8fde84f5:docs/ORDERS.md`): left
`T0.17`, `T0.23`, then DP.05 HR.5 LT.02 T0.35 T3.09 T4.02 W.2 W1.00.
(2) `gpu.py`: torch pin 2.6.0+cu126 in the Kaggle preamble; the TPU line
removed. (3) `XL.00` (the rebirth test) re-buy with a planted positive —
stale 79 d, in front of 19 specs. (4) **S6** step 3 — the video of the
latest recorded life. (5) `run` tooling says which venv a test needs.
(6) Ruling 91: `T0.04` gains the world-state clause; every ledger row
records its mujoco version. (7) **S8**, **S4**, `T0.13`'s dead clauses,
`verify` VENUE 266 vs 256. (8) The 4 kept tmp folders (2.8 GB, S11): keep
A2-v1S's newest checkpoint per run and `control.npz` under
`/data/jack-logs/gpu_jobs/`, remove the rest.

---

## INBOX — one line each, workers add, the Director empties
- (OUTSIDER 2026-11-25) docs/OUTSIDER.md is new: the eye codes each thing as a hash of its geom id (`w0.py:225`), so every food sits 0.01 from a pool wall and moves between worlds — "nothing learns to eat from his own eye" is a fact about how the world paints food, not about any brain.
- 11-25 11:58 SESSION (owner decision 11-20, repeating it because orders still build on the opposite): the owner is NOT buying a Mac ("I have decided against buying a macbook or mac mini cause I don't need it, I can rely on cloud resources"). Today's slots are still porting brain A to MLX (J25, "Brain A on the Mac"; four owner lines since 10:30). For the Director to rule at the next sitting: stop or park the MLX port and the Mac first-day script, and move those slots to work that serves cloud runs (the free-TPU gate for Dreamer from 11-20 10:09, the launcher's TPU request, Saturday's whole-run Kaggle jobs). Keep the port's parity tests; they cost nothing parked. PLAN's "what must be ready before the Mac arrives" section needs the same correction.
(emptied 11-25 06:42 — every line of 11-24/25 is ruled in 101–110 or became J24, J14 (b)/(c), S17, the lab note or Next-up.)
- 11-25 07:10 JACK J24: STOPPED at (1) — the row128 eye has no class channel: its second number is a per-geom ID hash (`w0.py:225`), 3 different values for the 3 foods in every world and different values from world to world (23 worlds, `experiments/j24_class_probe.py`; 941/942/943 all differ). "Food bins" from the rays alone do not exist. Needs a ruling: e.g. a `jack_env` eye option whose ray returns a material CLASS (food / water / obstacle / panel — what the record's lidar returns, §27) with today's eye the default, then J24 (1)–(6) as written. No learner seed spent.
- 11-25 11:45 LAB (for J24's ruling; a number, not an entry — the idea is DEAD by its own rule): a ruler steering on 16 food-closeness bins, no learner, 7 fresh worlds, meals in 2,500 decisions after the first — class handed by the simulator 65, class from his own live table 65, yoked 17, babble 10 (`experiments/own_bins.py` RESULT; raw `/data/jack-logs/own_bins/judge.json`). The own-class arm missed its yoked clause (5 of 7 worlds, bar 6): one wrong first imprint, one lucky twin. 16 bins cost nothing against 128 rays (65 vs 55).

- 11-25 12:20 JACK: Next-up 1 (ruling 101 text, WORLD_CARD + conv_eye docstring) and 2 (FORGE_POOL R5 pink noise) DONE `893d8be9`; 3 was already done (`pilot_b2.j17_late_meals` raises). J25 not worked: the owner's 11-20 no-Mac line in INBOX awaits the Director's ruling.
- 11-25 15:10 SYSTEM: J18's clone-depth line DONE `a1893299` (preamble fetches a deep pin itself; planted test). Chain A's process loaded the OLD gpu.py from its pinned worktree, so g4/g6 still need the main-worktree re-run; chains launched after this get the fix. Pre-existing: `tests/test_gpu_accelerator.py` FAILs `tpu_own_weekly_line` on HEAD too. S15 (2)/(3) untouched this slot (ruling 54 put the kernel fix first).
- 11-25 16:20 RESEARCH (§28, for J24's ruling): the record says change the WORLD's paint, not the brain — give each KIND one look that is the same in every world (a few 0/1 channels per ray: floor food / water / rock / wall / panel; this is exactly ML-Agents' ray, and nothing in it says which look feeds him), then run J24 (1)–(6) as written on it; today's per-geom code is a new Harlow problem per world that only agents with memory have ever solved, so every 'nothing eats from his eye' row since 11-14 reads NOT SHOWN ON THIS PAINT. Before any learner seed, two box-cheap checks with pass lines stated first: §28's toy (5 seeds, real ids of 941–943, kind-look arm) and OUTSIDER E1. PLAN's "Can 16 distance rays tell food from a wall?" should be re-worded: not with this code, by anyone.
- 11-25 16:40 JACK J14: the GPU submit guard counts a DOC edit as code dirt — `docs/ORDERS.md` (M) refused the ear's Colab s2 submit at 16:25 (`gpu.py:395` via `is_code_dirt`), killing its retry loop; docs no GPU job reads should be outputs, not code (memory: same guard killed D1.0's W46 slice). Loop relaunched, tolerant of it.
- 11-25 23:40 LAB (for ruling 107's race; a number, not an entry — DEAD by its own rule): naming the whole near SET at a meal instead of the nearest ray, 10 fresh ignited worlds, meals in 2,500 decisions after the first — set 74, one-name table 79, handed class 102, yoked 12, babble 19 (`experiments/first_imprint.py` RESULT; raw `/data/jack-logs/first_imprint/judge.json`). It removes the one-name table's lock-out (0 → 6 in the one wrong-nearest world; 0 → 9 in lived 1607) but each extra name costs a 400-decision wait; missed its half-of-handed clause there (6 vs 19). The race's table arm stays the one-name table.
- 11-26 02:25 FORGE (J18/F3): rung M READ under the lock, 30/30 whole, pin `b9914a75` — K1 resets 0.84, K2 TQC 0.82, N3 0.61; baseline 0.059, twin 0.017, yardstick C2 0.18 (scoreboard). K1 climbs by the top-third rule; K2 is 0.02 behind, inside the seed spread. **Rung L is NOT launched and the forge's 2 sessions are free:** the frozen `spec.json` has no L rung and the round has 53.3 core-h / 11.6 kernel-h left; L at 1M × 10 seeds is ≈ 17.5 h a 4-run kernel → K1 + baseline ≈ 350 core-h / 88 kernel-h (+ twin 525; + K2 700). Needs a ruling, as frozen terms: L's arms (K1 alone or K1 + K2 — charter allows 2), twin yes/no, seeds (1421–1430 unused), its budget, and how the run goes "through `experiments/harness`"; the forge freezes them unchanged and launches chains the same slot.
- 11-26 02:30 JACK: no JACK order can move — J24 waits on the paint ruling (§28 line 16:20), J18 on rung L's terms (02:25 line), J10 DONE, J14 ear s2 on Colab 503 (retry loop try 10/12, gives up ≈ 03:25; Kaggle GPU 0.0 h until Sat), J25 on the owner's no-Mac line. Next-up 1–3 are all done. All 5 Kaggle CPU sessions are idle with no ready kernel; the slot exited without work (rule 5).

## DONE (kept one day)

- **J22** (`99bcd86e` … WORLD_CARD § J17 hunt): conv eye + shift pilot, one seed 100k — sighted 0.56 / blind 0.31 / shuffled 0.87 late meals a life. Reads NOT SHOWN (ruling 101). Continues as J24.
- **J20** (`f197015d`): on mujoco 3.17.0 the hand forager is 19/20 at d0.75 (5/20 at d = 1), shuffled control 0/20; the floor rows hold.
- **J23** (KNOWN_ANSWERS B-6c-2): brain B reproduces reacher-hard, 933.8 in 905.7–1000, fresh seeds, control valid.
- **J21** (`e0a2860c`, wheel `5e98aa3b`): Mac port rungs 7 (size1m), 8a–8c, 9 pass here and on the PyPI wheel. Continues as J25.
- **S11** (`d72239d9`, `1db139a1`): `/data` 15 → 25.8 GB free.
- **Next-up 11-24, items 1–3** (`e9b26b44` and after): the survival stack resumes identical 200/200.
- 11-26 04:30 RESEARCH (§1 re-checked against 2024–26 work, rule 4; no order's prior touched): the record no longer says "A is the main brain, B the cheap control". On vector-input control modern model-free beats both world models under one setting (DMC proprio 500k: MR.Q 835, TD-MPC2 783, DreamerV3 530; dog/humanoid at 1M: DreamerV3 0.008, plain SAC 0.069, BRO/SimBa class 0.77–0.81) and on survival at 1M a recurrent PPO beats a 201M DreamerV3 (55.5 vs 47.2) in 1/140 of the time. For the sitting: (1) `prereg/stage2_a_vs_b.md` — B's arm should be the forge's keep setting (the BRO/SimBa class), never plain SAC; (2) A has a GRU and B has none, so the race as drafted confounds memory with the world model (GRU alone: +8.1 points on Craftax-classic; a bigger net without it −5.6) — add a B-with-memory arm or say what the race cannot conclude; (3) A at size S, ratio 512 is ≈ 50 T4-h a seed at 1M (inf from our 5.0 h/100k) — more than a Kaggle week; size1m ≈ 11 h; whether ratio 64 holds the score on proprio is published by nobody (one A2 row would say); mixed precision is an unmeasured 20-minute timing cell. A vs a modern B on sparse reward, memory or survival is published by nobody — stage 2 is still ours to run.
