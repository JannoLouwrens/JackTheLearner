# ORDERS — what the workers do. Written by the Director.
Director sitting: 2026-11-22 06:42 UTC

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5, **each ≤ 4 lines of text — replace,
do not append; history is in git**). One state+journal commit per slot.
Machine problems go under INBOX, one line each. Rulings 76–91 cited here are
in `docs/DECISIONS_RESOLVED.md` under 2026-11-22.

**`docs/PLAN.md` binds every order (owner, 11-15).** Research → combine →
reproduce → test the last 10% → put it in him. We are in **stage 1: the
instrument.** `docs/research/PUBLISHED_PRIORS.md` is the reference — read
your order's section (only that section) at the start of every slot.
**Where stage 1 stands (11-22).** J17 is read, 20 of 20: **NEITHER eye
eats. The wide eye helps by its width, not its sight; a learner that trains
on real sight eats 3–8× LESS than one that never trains; B's critic runs
away on eye content in 10/10 sighted runs, 0/10 blind.** So the wall is not
the eye, it is **brain B's critic on a wide sense** (→ J17 hunt). And on
published tasks the LIBRARY recipe itself now misses 3 of its last 3 rows
on our apparatus (→ J19), and brain A's size-S arm was not the published
size (→ J14). **Neither brain is cleared for stage 2 today.** Keeping what
he learned is forge round 1, running (J18).
**THE USAGE STOP IS ABOUT A DAY AWAY (ruling 88).** The shared weekly meter
read 63% at 06:37 and climbs ~1.5 points an hour; workers stop at the
owner's number and stay stopped until Thu 11-26 09:00 UTC. So, every slot
until then: **(1) no free Kaggle CPU session while a ready kernel exists;
(2) every branch below is already ruled — do NOT end a state line "NEXT:
Director" when the order names the branch; take it; (3) kernels and box
jobs write their raw results where a slot on Thursday can read them; (4) a
slot that can advance nothing exits without building speculative arms.**
**One world (ruling 76): mujoco 3.17.0.** Every row about his world runs in
`/data/venvs/jack-std` (box) or a Kaggle kernel pinned to 3.17.0. A box row
made on 3.2.3 is not set beside a learner row until re-read (J20). Learner
tests (`sbx`, `optax`) run under jack-std, not the project venv.
**Every new run in his world passes** `eye_frame="body"`,
`motor_frame="body"`, `food_floor_only=True`, `lazy_contact_audio=True`,
`native_substep=True`; brain B uses `ent_coef="auto"` (ruling 64); "eats"
reads the LATE half of the run (ruling 65). Learner runs in his world only
through J17, J14 and the forge's frozen round (ruling 58).
**Stopped (rulings 10, 20, 23, 36, 59, 80):** home-made learners; forager
cells on the 16-ray compass eye; world-rule races; `D1.0`, `T2.01`, smell
runs; the needs-hidden row (closed, FAIL twice); new specs with no
published prior. A question with no prior is a `RESEARCH:` line in INBOX.
**Kaggle CPU: 5 sessions IN ALL (ruling 85). "Cap" below is a share, and a
share binds only while another order's READY kernel is waiting — 3 running
is NOT full.** Rank: (0) any kernel < 1 h; (1) forge round 1 — 2; (2) J17's
hunt and J17-2 — up to 3; (3) J19 — 2; (4) J10-2's segments — 2. Every JACK
slot's first act: `/data/venvs/kaggle/bin/kaggle kernels list --mine` +
`status` — harvest what is home, refill what is free.
**The box (rulings 32, 53).** Day ceiling 57,600 CPU-s, ≤ 2 computing
detached jobs. < 15 CPU-min → box, nice 19. Longer → box only when all 5
Kaggle sessions are held AND load < 4; else Kaggle. The box is shared
(never touch other projects). `pgrep -af` the same command line before
launching.
**Kaggle GPU: W47 is spent (33.1 of 30 on our meter; ~1.6 h real). Nothing
launches on a GPU before W48 (Sat 11-28 00:00 UTC) except J14's timing
cell. No first launch of any GPU job without a timing cell (ruling 86).**
TPU closed (ruling 71).
**Disks:** `/` 5.4 GB free (workers refuse under 3); `/data` 16 GB (floor
15 → S11). Scratch, clones, caches under `/data` only.
**World files** (`w0.py`, `needs.py`, `survival.py`, `drives.py`): changed
only through `jack_env.py` options whose default is today's world.
**Known-answer rows (rulings 33, 70):** seeds AND at most one extension
stated BEFORE the first seed. A miss is a miss. **An extension that cannot
change the verdict by its own arithmetic is not run; the line says so
(rulings 78, 81).**
**Receipts:** the slot that launches a kernel commits
`experiments/gpu_submissions.jsonl` and `cpu_budget.json` by name.

---

## JACK — build the creature, his world and his curriculum (max 7 open, ranked)

*Order of work: **J17** (harvest h0/h2, launch h3, take the branch) →
**J19** (two Kaggle CPU sessions are free NOW) → **J10** → **J20** →
**J14** → **J21**. J18 is the forge's. A slot waiting on a kernel advances
the next order.*

### J17 — Brain B's critic runs away on a wide sense: finish the hunt, then the row again
part: Learning to eat (+ His brain B) · stage: 1 · step: 4 (pilot, then a frozen row) · prior: §19, §24; `WORLD_CARD.md` § J17; rulings 66, 77
prior: published — no case of a dominant block of normalised vector inputs making a SAC critic run away (§24); the cures on record act on the critic's features (SimBa; LN + weight decay, Lyle 2024; unit-ball feature norm, Hussing 2024) or give each sense its own encoder (Lee 2018, DreamerV3). Ours: J17 row NEITHER EATS; sighted Q ÷ supported 1.2–2.2 (10/10), blind 0.4–0.8.
prior: learnable by a plain learner — only with the direction handed (E2c 4/5). B's known-answer side is itself open (J19). Steps: runaway separates from blind at 100k (5/5 vs 5/5), not before.
**The hunt is BOUNDED (ruling 77): h0 SimBa (Kaggle), h1 AdamW (read: STILL
RUNS AWAY), h2 OFN (box), h3 split first layer (built). Seed 923, 100k,
each beside its own blind twin on the same machine, read by `j17_onset
hunt`. No h4 — a fifth arm is an INBOX line.**
**Do.** (1) Harvest h0 and h2 the hour they land; launch h3 (Kaggle if a
session is free — both arms in one kernel — else the box as h2 ran).
(2) **Branch, already ruled:** an arm CURES on s923 when the reader does
not print RUNS AWAY and its sighted late meals/life ≥ its blind twin's.
If one or more cure → the one with the smallest sighted Q climb goes to
CONFIRM: the same pair on seeds 921 and 922, one Kaggle CPU kernel a seed.
(3) **J17-2, FROZEN NOW, launched only if CONFIRM holds on ≥ 2 of the 3
seeds:** J17's rule and world exactly, plus the ONE curing key on every
learning arm; arms K-row128 / K-row128-blind / K-row128-shuffled (each
step's ray values permuted: width and statistics kept, content destroyed) /
K-rays16; seeds 931–935, 300k; primary = late meals/life, row128 over
BLIND **and** over SHUFFLED, each MWU U ≥ 21/25 and ≥ 4/5 pairs; verdict
SIGHT / WIDTH / NEITHER; no extension. Write those pins into
`docs/research/prereg/row128_j17_2.md` and commit before seed 931.
(4) If NO arm cures: stop. One row in `WORLD_CARD.md` § J17 ("four
published critic cures, one seed each, none holds"), one INBOX line, and
J19 becomes the lead — B's fault is hunted where a published curve exists.
serves: M3 · PLAN "Learning to eat" last 10% · philosophy 1, 3, 4.
**done when:** all four hunt arms have a read row, and either J17-2's four arm rows are in `WORLD_CARD.md` with its verdict, or branch (4)'s row is written.
state: 11-21 J17 row DONE (`98e22036`/`59bbdf90`/`9b3ea35a`): NEITHER EATS; row128 > rays16 (U 22/25) but < blind (U 0/25). Blind arm = 256 constant zeros (no live inputs), so content vs live width stays open.
state: 11-24 11:45 **h0 SimBa READ** on Colab T4 (`ladder-1795516779`, 100k both, rc 0): Q climb sighted 2.360 / twin 1.082 → **STILL RUNS AWAY**; late meals/life 1.13 / 0.93 (sighted Q below its twin's at both marks — a ratio climbing from a small base, as in h2; caveat in WORLD_CARD).
state: **Hunt closed at branch (4): four published critic cures, one seed each, none holds** (h0/h1/h3 RUNS AWAY, h2 NO CONTROL). Row written in `WORLD_CARD.md` § J17 hunt; J17-2 NOT launched; INBOX line sent. The order's own text says J19 becomes the lead, but J19 is DONE (`7ab52e6c`).
state: NEXT: the Director rules what follows (J17 done-when is met by the branch (4) row). Until then, workers go to J10.

### J19 — DONE `7ab52e6c` — The library recipe misses published numbers here: find out why, where a published curve says what right looks like
part: His brain (B) · stage: 1 ("each brain reproduces published numbers") · step: 3 (reproduce) · prior: §1b, §21; `KNOWN_ANSWERS.md` rows B-5, B-6a, B-6b, B-6c + B-6 addendum; rulings 68, 81
prior: published SBX/SimBa-protocol SAC bands — walker (row 5) 841.7–864.3, swingup_sparse 726.9–826.1, reacher-hard 905.7–1000, reacher-easy 929.9–1000. Ours, LIBRARY arm: 833.8 / 541.1 / 824.9 MISS, reacher-easy 953.7 REPRODUCES. Shape each time: 3–4 of 5 seeds match the paper, one never finds the reward or scores zeros at eval. No wiring fault (`ecc33343`); three unlisted differences: update order, α-loss form, eval re-seeded each mark (theirs) vs one fixed target set (ours).
**The extensions are NOT run (ruling 81: each needs fresh seeds averaging
above every seed seen). Rows 5, 6a, 6c stay named MISSES. This is a cause
twin, not a third try at the row — it cannot turn a miss into a pass.**
**Do.** (1) Two Kaggle CPU kernels NOW (rank 3), `brain_b_dmc` LIBRARY arm
only, same 5 seeds, same steps: 6a swingup_sparse and 6c reacher-hard, with
TWO additions and nothing else — save the model at every eval mark, and
log per 10k steps critic loss, mean Q, α (J17's trace keys). (2) When home,
on the box (jack-std): re-evaluate every saved model on 100 episodes two
ways — our fixed target set, and re-seeded each mark as the paper's code.
(3) Reading, written into `KNOWN_ANSWERS.md` B-6 BEFORE (2) runs: mean in
band under the re-seeded 100 → the miss is OUR EVAL (fix the eval, row may
be re-registered once, ruling 70); still out with one seed flat from the
start → an exploration draw (report its Q/critic trace beside J17's
sighted shape: same climb or not); a seed that learned then fell → the
J10/E2c win-then-lose shape, on a published task — say so in one line.
(4) `docs/research/prereg/stage2_a_vs_b.md` (draft): B's gate lists rows
5, 6a, 6c as named misses and says B does not enter stage 2 until this
order's reading is written.
serves: M3 · PLAN step 3 ("a learner that fails is a suspect, not a contestant") · auditor finding 2 · philosophy 3, 4.
**done when:** the two kernels' re-evaluation table and its one-line reading are in `KNOWN_ANSWERS.md`, and the stage-2 draft carries B's gate.
state: **DONE `7ab52e6c` (11-24 10:30).** Re-eval table + reading in `KNOWN_ANSWERS.md` B-6 addendum 11-24; gate in `stage2_a_vs_b.md` 2b (11-22). replay_gap 0 on 10/10.
state: 6c reacher-hard @150k: kernel-10 824.9 / fixed-100 886.1 / stream-100 **908.3** → reading 1 OUR EVAL by 2.6 pts (bootstrap 888–927); mostly the eval's SIZE (s4 5/10 zeros = 22–26/100). Re-registering 6c on a 100-ep stream eval is allowed once (ruling 70) — Director's call, INBOX.
state: 6a swingup_sparse: 541.1 under all three evals → reading 2 (s3 frozen explorer, α→0 before any reward, not J17's climb) AND reading 3 (s1 772.2 @125k → 407.7). Either alone leaves the mean out (676.4 / 614.0): B's to cure, not the eval's.

### J18 — Forge round 1 is a KEEP round (lane `forge` runs it; rung S is running)
part: Learning to eat + The science lab · stage: 1 · step: 4 · prior: §19, §21, §22; `docs/FORGE_POOL.md`; rulings 61, 65, 69, 83
prior: E2c — on E-food the baseline reaches (4/5) and loses (10/10). Record: resets + LN critic best-supported (Nauman 2024) but ours has LN and lost; TQC (Kuznetsov 2020); 1 update per 10 decisions and ±1 a food (the one published SAC forager); `n_steps=3` (Rainbow, D4PG). Nobody has published a small agent still improving at the end of a long life (§20).
**Terms frozen `594752d0`; the forge's four fill-ins (M = 5 seeds, 270
core-h, K1 resets every 50k, one never-trains twin) STAND (ruling 83).**
Rung S: 30 runs, two chains, 2 Kaggle sessions (rank 1).
**Ruling 83: the rung RUNS; only ACCEPTANCE waits for the tripwire (S13
(b)). If S13 (b) is not committed when the forge's 14:00 slot starts, the
forge builds the hash check itself inside its reader — with the planted
edited-judge test, and the reader's own file in the hash list, pinned at
the commit BEFORE the first raw file is opened — and SYSTEM's next slot
reviews it. Raw files are not opened before that commit.** Before rung M
launches, the round runner saves through `brain_b.save_life` (300k runs
must be resumable); that is the only code change allowed between rungs.
serves: M3, M7 ("he keeps learning") · philosophy 1, 3, 4.
**done when:** rung S is on the scoreboard — one reading per candidate beside baseline, twin and controls — and rung M's kernels are launched by the frozen rule.
state: 11-22 02:25 forge: kernel built + tested `bf1f5f7d` (baseline = E2c's E-food bit for bit); rung S RUNNING since 02:20 in two chains (`launch_S_chain{A,B}.log`), 3 kernels COMPLETE, 2 RUNNING at 06:39; each kernel checks out one pinned commit (`forge.code_head`).

### J10 — The walking layer: the row again with enough seeds, life carried across segments
part: Body and movement · stage: 1 side job (stage 3's part) · step: 3 (reproduce) · prior: §16, §21; rulings 45, 72, 82
prior: §16 — SAC, Zoo entry, Humanoid-v4: 4,825 [3,784–5,866] at 1M (10 seeds), 6,232 ± 280 at 2M, each ONE unbroken run. Ours: Row J10-1 FAIL (1 of 3); (a) READ — the learner's state resumes exactly, but each resume dropped the episode in progress and re-seeded the env; the carry (`carry_life=True`, `d39687f0`) makes a resumed run bit-identical to an unbroken one (toy, 1 seed).
**Row J10-1 stays FAIL, with (a)'s named difference beside it. Ruling 82:
J10-2 CARRIES — the published run is unbroken, so the faithful copy is the
one a resume cannot be told from.**
**Do.** (1) Rule text first, `humanoid_sac_row.py` as J10-2: the frozen
recipe (no cures), `carry_life=True`, fresh seeds 3–7, PASS R ≥ 5,950 AND
≥ 12 m/1,000 steps on ≥ 4/5, eval = 50 episodes on ONE stated machine
(Kaggle CPU), 1M median band and control as before, no extension. (2)
`segment_job_body` passes the carry and REFUSES to start a segment > 0
whose mounted input lacks `life.pkl` or `physics.pkl` (planted test).
(3) Segments at rank 4, ≤ 2 sessions; every segment saves its model (it
does) so a collapse can be re-evaluated.
serves: M4 · PLAN stage 3 "a separate, fast walking layer is built and checked in parallel from now" · philosophy 1, 4.
**done when:** J10-2 is in `KNOWN_ANSWERS.md`, pass or fail, by its frozen rule.
state: J10-1: 1M median 4,065 IN BAND, control valid; 2M 1 of 3 → FAIL. s1 collapsed in its LAST segment (α .009→.024, Q rising while return fell); two re-seeded forks both KEPT.
state: (a) READ 11-21 08:45 (`j10_resume_twin.py`): control bit-identical; twin differs from the resume row — SB3 drops the episode in progress, `train_segment` re-seeds the env. Addendum in `KNOWN_ANSWERS.md`.
state: 20:35 `d39687f0` carry built, default off: `brain_b.save_life` + `physics.pkl` (MjData + env RNG). Twin RC bit-identical on every item at T=1,000 and 1,010. `b6b2da1d`: `save_life` itself now carries MuJoCo physics (Hopper twin PASS) — `_save_physics` is redundant beside it, left.
state: 11-22 07:55 `b609b142`: (1)+(2) DONE — `RULE_J10_2` frozen in `humanoid_sac_row.py` before seed 3; `segment_job_body(row="J10-2")` carries + 50-ep eval, `require_carry` refuses a mount lacking life.pkl/physics.pkl (planted PASS, J10-1 text unchanged). Tests 10/10.
state: 11-24 10:09 **seg0 LAUNCHED for s3 (`jack-cpu-1795514966`) and s4 (`jack-cpu-1795514981`)**, Kaggle CPU, row J10-2 (carry on), watchers detached (logs `/data/jack-logs/j10/launch_j10_2_s{3,4}_seg0.log`; records → J10_2_OUT `s<k>_seg0.json`). ~10.5 h a segment. Row stub in `KNOWN_ANSWERS.md` (id clash with 11-16 Yu-Turk-Liu row → INBOX). NEXT: when a seg0 record lands, `kernel2 <s> 1`; s5–7 when ≤ 3 sessions busy (cap 2 for J10).

### J20 — One world: re-read the box's rows about his world on mujoco 3.17.0
part: His world + Eye · stage: 1 ("proof the task can be learned with his senses") · step: 4 (re-read) · prior: §12; `WORLD_CARD.md` (hand forager d0 / d = 1; the floor card); rulings 51, 76
prior: same seed, same 2,000 random actions on K-row128 → 0 meals on 3.2.3, 2 on 3.17.0 (INBOX 11-21 07:32). The hand forager's 20/20 at the entry rung — ROADMAP M3's "learnable from his senses" — and the floor card are 3.2.3 rows; every learner row is 3.17.0.
**Do, box, jack-std, nice 19, same seeds and same code as the original
rows, nothing else changed:** (1) hand forager on row128, body frame, d0:
20 worlds + its shuffled-ray control; then d = 1. (2) The floor card's
three learner-free rows on the J17 world (do-nothing, random, held-8
random), 200 lives each. (3) Each as a "3.17.0" row BESIDE the old one in
`WORLD_CARD.md` — the old row is not edited. Bars are unmoved (ruling 51):
if d0 no longer reads ≥ 18/20, that is a FAIL to report in INBOX, and
ROADMAP's line changes at the next sitting. (4) `eye_streak.py:173` and
`find_himself.py:132` call the 3.2.3 ray signature: route them through
the eye's own compat, as `59bbdf90` did.
serves: M3 · auditor finding 3 · philosophy 4 (a box number and a Kaggle number may share a table only in one world).
**done when:** the 3.17.0 rows stand beside the old ones in `WORLD_CARD.md` and the two callers pass on both versions.
state: 11-22 08:26 (1) d0.75 RUNNING on the box [09:30: 8/40 cells home after ~1 h → ~5 h wall, not 3; load 4.3, /data 16 G — g4 cannot go beside it]: `experiments/j20_reread.py` (`5c90a078`; ruler + bar unchanged, resumable) `run 0.75 f4 b4`, 40 cells, 2 workers, log `/data/jack-logs/j20/launch_j20_gate2_d075_f4b4.log`, ~3 h wall (12 ms a decision). g4 (shuffled, must fail) NOT launched: day's detached cap (34,560 s) left 15 k s.
state: (4) DONE `250942db`: both callers via `fast_sim._multi_ray`; `tests/test_j20_ray_callers.py` PASS on 3.2.3 (bit for bit vs old) and 3.17.0 (old FAILS = control).
state: NEXT: `launch_detached.sh /data/jack-logs/j20/launch_j20_gate2_d075_g4.log /data/venvs/jack-std/bin/python -m experiments.j20_reread run 0.75 g4` (JACK_DETACHED_EST_S≈6000) when the cap admits it; when all 60 cells home `j20_reread score 0.75` → 3.17.0 row beside gate 2's in WORLD_CARD; then d = 1, then (2).

### J14 — Brain A and the ear: make W48's GPU hours count before they open
part: His brain (A) + Ear · stage: 1 · step: 3 (reproduce) · prior: §2, §7; `docs/research/prereg/a2_size_s.md`; `KNOWN_ANSWERS.md` rows A1–A3, A2-v1S, N7-kws; rulings 67, 78, 79, 86
prior: A1 PASS, A3 PASS, A2 MISS under band at 1m/100K (in band at 200K); 1m rows match official v2 on 67/73 settings, 0 undisclosed. A2-v1S: MISS ABOVE band, 6/6 seed-points — and the arm was not the published size (encoder + decoder 2×512 vs official 5×1,024). Ear: BC-ResNet-1, published 96.9 ± 0.30; ours VOID at 182 s an epoch.
**Do — no GPU seed this week:**
(a) **A2-v1S is closed as MISS (above); its extension is not run (ruling
78).** Freeze the matched arm as `docs/research/prereg/a2_size_s_enc.md`,
row **A2-v1S-enc**: `v1s_diff.MATCHED_ENC` (the 4 keys, 53/56), same task,
same control, seeds 10–12, one extension 13–14, bands unchanged, **score
window = the band's own window {P−20K, P−10K, P}** (the frozen (0.9P, P]
number printed beside, not read), pins as A2-v1S. Reading written in
first: in band → the build is right at the published size and the 1m miss
is size; out → diff against official `ec526b9` setting by setting. Correct
the "Matches; nothing to name" claim in the NEW file, not the frozen one.
(b) Its timing cell is the FIRST GPU act: this week if S15's meter fix
shows real hours left, else Sat 11-28 00:00. > 9 h a seed → INBOX, no seed.
(c) **The ear gets no GPU hour until its epoch time is explained (ruling
79).** A 9,000-weight network at 182 s an epoch on a T4 is a data-pipeline
cost. Box-minutes: time one epoch's stages (decode, mel, augment, forward)
on a 2,000-clip slice; cache the features once if that is where the time
is; then a Kaggle CPU timing cell (rank 0). One seed must fit one kernel
with 20% to spare, CPU or GPU, or it is an INBOX line. Row N7-kws stays
VOID.
(d) Text only: `prereg/pilot_a.md` to ruling 62's terms, with (d) of
11-21's checkpoint option named as its resume path.
serves: M3, M5 · PLAN ready-list "brain A reproduced"; "Ear" step 3 · standing objective 3 · philosophy 3, 4, 6.
**done when:** `a2_size_s_enc.md` is frozen with its kernel built, the ear's epoch has a stage-by-stage timing and a seed that fits a kernel, and `pilot_a.md` is drafted.
state: 11-21: A1 PASS (782 @100K, 984 @200K), A3 PASS (956), A2 MISS @100K (393.3 vs 438–518). A2-v1S frozen `8f13ab28`, 3 seeds: 100K mean 610.0, 200K 835.3, control valid → MISS ABOVE.
state: ear HOME 11-21 13:28 = VOID: 10.1 T4-h a seed (guess 1.5), cut at epoch 199/200, no test accuracy.
state: NH-2 SCORED: R1 FAIL (gmq@5000 0.79 > 0.70), R2 FAIL (bimodal, 4/10 seeds 46–90) → the row is CLOSED (ruling 80). Checkpoint-with-replay option DONE `bbd54926` (buffer only; no gradient step in the test).
state: 11-22 08:45 (c) box half DONE `c34634a5` (`kws_epoch_stages.py`, authors' objects, 2,000 real clips, 1 thread, CPU s per 36,923-clip epoch): decode 8.2 (cold 14), noise loop 4.9, log-mel 22.6, step 303. Decode + loop < 20 s of the T4's 182 s → not the file read. Suspect: `Preprocess` copies a whole 4.26 MB noise file to the GPU per mixed clip = 126 GB an epoch. NEXT: Kaggle timing cell (rank 0) — same stages on CUDA with synchronize, + one arm with the noise held on the device; then size one seed.

### J21 — Brain A on the Mac: the port table's next rows
part: His brain (A on the Mac) · stage: 1 (ready-list "first-day speed test for the Mac") · step: 2 (combine) · prior: `docs/research/PUBLISHED_EVIDENCE_2026-11-15_PORT.md` step table; `experiments/brain_a/README.md`
prior: nobody has published Dreamer on Apple hardware. Rungs 1, 2, 3, 6, 7 PASS here eager and compiled and on the PyPI wheel (Linux x86); rungs 6/7 at TOY size; row 2.1's size1m golden PASS.
**Do, box only (< 15 CPU-min a run, nice 19), one row a slot, in the
table's order:** rung 7 at `size1m` against the g1v1 golden (prerun's
sampler call skipped — say so in the docstring), then the world-model loss
rungs (prediction, dynamics and representation losses with free bits;
then the full world-model loss on one golden batch). Each row as before:
NumPy 64-bit reference from the official code first, tests committed
while they FAIL, then the module until they pass eager and compiled; "ours
beside theirs" in the docstring. No training, no speed claims. One wheel
re-check kernel (< 1 h, rank 0) after every third rung.
serves: M3 · PLAN "the decisive test … on the Mac mini" — every Mac hour an hour of the real test · philosophy 5, 6, 7.
**done when:** rung 7 at size1m and the world-model loss rungs pass, test names in `experiments/brain_a/README.md`.
state: (new 11-22; continues J16, DONE `75bd15b5`, wheel PASS `jack-cpu-1795286363`.)

**Lane `forge`** reads `docs/FORGE.md`; its F3 is J18. Lane `lab`: a
candidate for brain B enters through `docs/FORGE_POOL.md` with its prior;
probes of other seats stay in `LAB.md`. **The lab's hindsight eye (ruling
84): ENTERED as J17's next candidate, NOT launched** — it waits for the
hunt's branch and for a `RESEARCH:` prior (below).
**Parked or closed:** `D1.0`, `T2.01` (→ J10), `LF.06`, `LC.09`, `CR.01`,
`LF.04` — "to earn again" on the chosen brain (ruling 20). `NE.01` → stage 3
(ruling 35). `PS.02`/`.05`/`.06`/`.08` → stage 3 (ruling 42). `W1.04` →
re-read 11-25 (ruling 41). The lab's place key → stage 3 (ruling 63).
Needs-hidden row → stage 3 (ruling 80).
**Next up, in order (idle slots; box < 15 CPU-min, text unless said):**
1. `WORLD_CARD.md` § J17: one paragraph, plain words, that the blind arm
fed constant zeros (no width-matched noise control) — J17-2's shuffled arm
is the answer.
2. `KNOWN_ANSWERS.md`: one table "where each brain stands against the
record" — every row of A and B, in band / named miss / void, one line each.
3. The forge runner saving through `save_life` (J18, before rung M) if the
forge has not done it.
4. Ruling 91: the resumed-against-unbroken twin on the SURVIVAL stack's
own checkpoint path (box-minutes, project venv, no ledger): identical, or
the differing item named in one INBOX line.

---

## SYSTEM — work on the machine (max 3 open)

*Order of work: **S13 (b) FIRST** (the keep round's results sit unread
without it) → **S15** → **S11**. A fix a RUNNING Jack kernel needs goes
first (ruling 54).*

### S13 — The judge lock's tripwire (forge F5), then the lane-guard test
part: (machine — nothing faked; FORGE.md "Fences") · stage: — · step: —
**Do.** (b) FIRST, THIS SLOT: hashes of the judge files
(`experiments/harness/`, `registry.py`, the frozen rule files a forge
round names, the round's `spec.json`, the round's reader) recorded at the
round's freeze commit and checked by the forge round runner/reader before
it accepts a result; a planted edited judge is REFUSED (test committed).
If the forge already built it (ruling 83), review that instead: run its
planted test, add a second plant of your own, one state line. (a) After:
reproduce `scripts/test_lane_guard.sh`'s FAILs of 11-20 on a clean tree;
fix by cause or name why the test is wrong. Real separation (own user) is
NOT ordered.
serves: philosophy 4 · J18.
**done when:** the planted judge edit is refused, and `test_lane_guard.sh` exits 0 or its wrong clauses are named.
state: 11-22 09:30 (b) DONE `e3e573e5`: `experiments.forge.write_lock(round_dir, [reader])` hashes harness, registry, ladder rule, metric, the round's files + the reader's own file at HEAD → `<round>/judge_lock.json` (commit it alone, once); `check_lock` refuses a moved byte or a re-hashed lock, and `load_round` runs it whenever `results.jsonl` exists. Planted: harness edit, reader edit, re-hashed lock — each refused, committed or not (tests.test_forge_round 24/24).
state: Forge: write the lock at your reader's commit, before opening raw files; no hash check of your own needed (ruling 83). r1 has no lock yet. (a) lane-guard test next.

### S15 — The detached lane's books: a FAILED receipts test, a cap that starves, a meter that bills waiting
part: (machine — truthful gates) · stage: — · step: —
**Do.** (1) `T0.34` ("the detached lane writes its own receipts") went
PASS → FAIL at the regate of 11-21 02:48 and `coverage` exits 2
(FAIL-UNOWNED). **You own it.** First step: name the failing clause (read
the ledger row and `regate.log`; do NOT re-run the spec to look — running
writes the ledger). Fix by cause; re-buy on a clean tree. (2)
`launch_detached.sh`: a Kaggle WATCHER is not a computing job — the
2-job cap counts only jobs using ≥ 0.10 cores; the poller cap counts GPU
and CPU watchers apart and never refuses a box job ("0 busy detached jobs
alive" must launch). Planted tests for each. (3) `gpu_budget.json` billed
`jack-ladder-1795268291` 10.02 h for a 5.1 h run: bill Kaggle's own run
time (RUNNING → COMPLETE), not queue time; correct W47's line with the
arithmetic shown; `dispatch.sh` pre-flight prints the correction. No
ceiling is raised. That planted test becomes a clause of `T0.12` (ruling
91: debit = run time; the old meter must FAIL it).
serves: philosophy 4, 5 · auditor findings 1, 5 · standing objective 3.
**done when:** `T0.34` is PASS again by a fix (or its FAIL is explained and owned in a state line), the three planted tests pass, and W47's GPU line shows run time.
state: (new 11-22.)

### S11 — Disks: `/data` is at its floor
part: (machine — a dead machine outranks everything) · stage: — · step: —
**Do.** `/data` has 16 GB free against a 15 GB floor; `/data/caches` holds
18 GB and is THIS project's (uv, pip, XDG). `uv cache prune`, `pip cache
purge`, HF/XDG entries older than 14 days — first show that no venv under
`/data/venvs` breaks (import test of jack-std and the project venv before
and after). Never `/data/tools`, never `/data/jack-logs` results, never
another project's directory (WorldTwin's database lives on `/data`), no
new cron. State line: free GB before/after. `/` (c): unchanged — re-run
`prune_cli_versions.sh` only when `ps` shows no 2.1.325/326/329 claude.
serves: philosophy 5, 6.
**done when:** `/data` has ≥ 25 GB free with before/after in the state line.
state: `/` 4.31 → 5.4 GB (`prune_cli_versions.sh`, `~/.cache/uv`, `git gc`); the rest is held by the owner's 6 open sessions (~1.1 GB). FIELD_WATCH cap + prompt lines DONE `a072b554`.

**Next up:** (0) **S9** as written 11-20 (`git show 8fde84f5:docs/ORDERS.md`):
left `T0.17`, `T0.23`, then DP.05 HR.5 LT.02 T0.35 T3.09 T4.02 W.2 W1.00.
(1) `gpu.py`: torch pin 2.6.0+cu126 in the Kaggle preamble; the TPU line
removed (ruling 71); pushes count RUNNING kernels first. (2) **S6** step 3
— the video of the latest recorded life. (3) `XL.00` re-buy with a planted
positive (stale 76 d, in front of 19 specs). (4) `run` tooling says which
venv a test needs (no venv has `pytest`). (5) `prereg.judge()` door for
unequal arms. (6) `run status`'s STEERING parser reads this page's
headings. (7) Ruling 91: `T0.04` gains the world-state clause from
`b_life_mujoco_test` (planted no-physics save must differ); every ledger
row records its mujoco version. (8) **S8**, **S4**, `T0.13`'s dead clauses, `verify` VENUE 266
vs 256; `coverage` baseline entries DP.02, DP.03, LC.04 live again.

---

## INBOX — one line each, workers add, the Director empties
- 11-24 11:50 JACK J17: hunt CLOSED at branch (4) — h0 SimBa on Colab T4 reads STILL RUNS AWAY (2.360 / twin 1.082; late meals 1.13 / 0.93), so none of the four published critic cures holds on s923. The order's fallback lead J19 is already DONE → J17 needs a ruling on what follows. Also: in h0 and h2 the sighted Q sits below its twin's and the ratio climbs from a small base, and `j17_late_meals`'s default late_from (150k) returns nan on a 100k hunt (must pass 50_000) — the reader's ratio may be the wrong ruler for bounded critics.
- 11-24 10:35 JACK: J19 DONE — 6c reacher-hard reads in band (908.3 ≥ 905.7) under a 100-episode reference-style eval, out on our 10-episode one (824.9). Ruling 70 allows one re-registration of 6c on that eval; ordering it is yours. 6a's miss is the learner (frozen explorer + win-then-lose), not the eval.
- 11-24 09:55 SESSION: the team stood idle from 11-22 ~13:00 to 11-24 09:51 (144 slots stopped at the 100% weekly meter; the owner signed the box in to another account at 09:51, meter now 8%, resets 11-30 05:00). The meter is shared with other projects on this box, so it will fill again. Two asks. (1) FIRST THING this slot: harvest what finished on Kaggle while we were stopped — forge round 1 rung S (launched 11-22 02:25, never read), the SimBa/eye fixes, walking segments — before launching anything new; outputs older than Kaggle keeps them are lost. (2) For the Director to rule, lane system to build: many worker slots in 11-20..22 ended "every free machine is busy, nothing could start" and still cost a full session. Add a no-model pre-check to the launcher: if all Kaggle sessions are RUNNING, nothing is ready to harvest, and no order has box-only work, the slot exits before a model is started (logged as SKIPPED-IDLE, with a planted test). That stretches the shared allowance without removing any work.
- 11-22 11:40 JACK: `pilot_b` partial file (`write_partial`, pilot_b.py:276) keeps lives only — a hunt cell killed by the 8.5 h cut loses its whole Q/critic/α trace (h0 SimBa: 8.5 h of Kaggle CPU, zero readable marks). Also nothing prices an arm's rate before a launch; SimBa ran 26× slower than the MLP it replaced.
- 11-22 09:40 SESSION (machine, /data at 90% — 17 GB free, was 39 GB on 11-20; the platform monitor alerts the owner above 85%): (1) `experiments/gpu.py` makes its job folders with `tempfile.mkdtemp(dir="/data")` (lines 458, 800, 1172) and nothing removes them — 392 `/data/tmp????????` folders since 11-11, 6.3 GB, each a kernel.py + `out/` (some hold 239 MB .npz files that also exist under `/data/jack-logs`). For lane system: make them under `/data/tmp/gpu/`, remove a folder once its outputs are harvested, and reap the existing ones that are harvested (I deleted none — I cannot tell which `out/` is the only copy). (2) `/data/jack-logs/b_life_mujoco` grew 1.2 GB in two days; say what its retention is. (3) Not Jack's, for the owner's eyes only: `starwright/tmp` +4.2 GB and `tradelab` +3.5 GB in the same two days; caches under `/data/caches` (hf 9 GB, huggingface 4.6 GB) were all read within two days by stt, the ear test and tradelab, so they are in use and stay.
(emptied 11-22 06:42 — every line of 11-21/22 is ruled in 76–88 or became J17, J19, J20, J14, J10, S13, S15, S11 or Next-up.)
- RESEARCH (Director, 11-22): what is published on a sense built from OUTCOME — features of what was in view when reward arrived, used as the learner's input (reward-conditioned salience, successor/landmark features, "what was at my mouth when I ate")? Numbers, sizes, steps; and whether any small agent learned to reach food that way from scratch. For the lab's hindsight eye (ruling 84).

- LAB 11-22 11:29 (for ruling 84's candidate, `experiments/live_eye.py`, rule `f3654d00`, raw `/data/jack-logs/live_eye/judge.json`): the hindsight table filled FROM EMPTY lights itself — 8 fresh worlds, meals in the 2,500 decisions after the first babbled meal: live 70 vs babble 12 vs yoked 27 (ahead 8/8, 6/8). It must carry UNLEARNING (a tabled colour stood at unfed for 400 decisions is dropped): without it 2 of 8 births lock onto a wrong first imprint (1 and 0 meals). Patience must outlast the 50 s regrow clock (10 s forgot real food). Proposal: when ruling 84's arm is built, build it as `live_eye.cell`'s table (empty at birth, extinction on), not the carried 215-meal table. Hand reflex only; no learner was run.
- SYSTEM 11-22 09:30: `coverage` exits 2 on FAIL-UNOWNED = 1 (SO.01, not T0.34 as S15 says) — bookkeeping, not acted on.
- JACK 11-22 09:50: `KNOWN_ANSWERS.md` has TWO "Row J10-2" headings — the 11-16 Yu, Turk & Liu PPO pre-flight (cited by `yu2018_walk.py`) and ruling 82's carried SAC row (stub added this slot, marked). One needs a new id.
- 11-24 10:35 JACK J17: J17's state said Kaggle W48 GPU hours open Mon 11-23; the meter resets SATURDAY (`gpu.week_key`, ruling 9) and W47 shows 33.1 h > 30, so Kaggle GPU is shut until Sat 11-28 00:00 — h0's GPU timing went to free Colab T4 instead. J14 (b) already had the right date.
- LAB 11-24 11:30 (for ruling 84's arm; `experiments/fed_once.py`, dev worlds only, no judge spent): (1) the hindsight reflex can WEDGE — one life steered 2,983 decisions straight at a real, tabled food it never reached; unlearning never fired because he was never "near". Any arm built on `live_eye.cell` needs a give-up (the file has one: target ray no shorter in 50 decisions → babble 16). (2) The imprint "nearest ray at the bite" depends on how he arrived: food on 74% of wild meals, 8 of 15 when a parent sets him beside the food — so feeding a newborn near food (the dial's d = 0 too: 1 of 4) mis-names it. Fed-then-weaned read DEAD at the pilot (ahead of born 1 of 6).

## DONE (kept one day)

- **J17's row** (`9b3ea35a`, `WORLD_CARD.md` § J17): 20/20, NEITHER EATS — width, not sight; the critic runs away on eye content.
- **J16** (`75bd15b5`): rungs 3, 6, 7 PASS eager + compiled; wheel PASS; row 2.1 size1m golden PASS. Continues as J21.
- **J14 of 11-21:** A1–A3 scored; A2-v1S 3 seeds MISS above; NH-2 FAIL, FAIL; ear VOID; A's checkpoint-with-replay option `bbd54926`.
- **Brain B row 6:** 6a MISS, 6c MISS, 6b REPRODUCES (lib 953.7, ours 958.1) → J19.
- **S14** (`e3f9a55d`): usage refusals carry the meter line; 0 slots were refused on a contradicted reading.
- **Next-up 11-21:** stage-2 draft's late-half text; `pilot_b.trace` std; B's stop-resume twin PASS (+ MuJoCo physics carried, `b6b2da1d`); the lab's taste-eye read DEAD; one-voice ear DEAD (for that warp only, §23).
