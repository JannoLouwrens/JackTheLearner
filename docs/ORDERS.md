# ORDERS — what the workers do. Written by the Director.
Director sitting: 2026-11-19 06:41 UTC

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5, **each ≤ 4 lines of text — replace,
do not append; history is in git**). One state+journal commit per slot.
Machine problems go under INBOX, one line each. Rulings 49–56 cited here are
in `docs/DECISIONS_RESOLVED.md` under 2026-11-19 06:41.

**`docs/PLAN.md` binds every order (owner, 11-15).** Research → combine →
reproduce → test the last 10% → put it in him. We are in **stage 1: the
instrument.** `docs/research/PUBLISHED_PRIORS.md` §1–§18 is the reference —
read your order's section (only that section) at the start of every slot.
**Where stage 1 stands (11-19).** The first learner seeds in the repaired
world are running: pilot 2's rule was frozen at 04:35 (`e2c3c378`) and
p0–p2 are on Kaggle. **Ruling 49: his motor is body-frame, like his eye** —
he pushes forward along where he faces and feels his velocity in his own
frame. The ground is END-FIT row 1 (the human-like body carries no compass)
and gate 2's row h4 (body eye + world motor: 0 of 20), not either floor
card. The rule frozen on the worker's written default STANDS; nothing is
re-drafted. On the body-frame eye the hand forager passes at the entry rung
(20/20, shuffled 0/20) and FAILS at the full rung (3/20 against 18).
**Ruling 51: "learnable from his senses" is shown at d0 = 0.75 only, by a
hand forager that is a lower bound; stage 1 and stage 2 are judged at the
entry rung; d = 1 is the dial's work in stage 3.** No learner has yet eaten.
**Every new run in his world passes** `eye_frame="body"`,
`motor_frame="body"`, `food_floor_only=True`, `lazy_contact_audio=True`,
`native_substep=True`. The integrator is not switched (ruling 40).
**Stopped (rulings 10, 20, 23, 36):** races between home-made learners
outside J3; forager cells on the 16-ray compass eye; world-rule races;
`D1.0`, `T2.01`, smell runs; new specs with no published prior. A question
with no prior is a `RESEARCH:` line in INBOX.
**Kaggle CPU: 4 sessions, RANKED.** A free session goes to the first READY
kernel: (0) any kernel estimated < 1 h; (1) J11's p-runs — **J11 holds at
most 3 sessions while another ready kernel waits**; the 4th rotates: (2)
J15's f5 gate; (3) J3 — one session; (4) J10's next segment; (5) J15
`kernel-full` (only after ruling 50's conditions); (6) brain B row 5
(`brain_b_dmc.py`, ~2 h). `W1.04` is PARKED to 11-25 (ruling 41).
**The box (rulings 32, 53).** Day ceiling 57,600 CPU-s, ≤ 2 computing
detached jobs. Whole run < 15 CPU-min → box, nice 19. 15–60 CPU-min → box
only when all 4 Kaggle sessions are held AND load < 4, one worker. > 60 →
Kaggle or it waits. **The box is shared: yesterday 1 slot in 5 was refused
on load from other projects (never touch them).** A launch refused on load
retries inside its slot (bounded), then the NEXT slot takes it first — a
launch is never dropped because one slot aborted.
**Kaggle GPU: 0 h until Sat 11-21 00:00 UTC**, then 30 h, order in J14.
**`/` has 5.3 GB free, workers refuse under 3:** scratch, clones and caches
under `/data` only (`UV_CACHE_DIR`, `PIP_CACHE_DIR`, `XDG_CACHE_HOME` under
`/data/caches/`, `--only-binary=:all:`).
**World files** (`w0.py`, `needs.py`, `survival.py`, `drives.py`): changed
only through `jack_env.py` options whose default is today's world.
**Known-answer rows (ruling 33):** seeds AND at most one extension stated
BEFORE the first seed runs. A miss after that is a miss.
**Receipts:** the slot that launches a kernel commits
`experiments/gpu_submissions.jsonl` and `cpu_budget.json` by name.

---

## JACK — build the creature, his world and his curriculum (max 7 open, ranked)

*Order of work: **J11** → **J15 (1)–(2)** → **J14 (from Sat 00:00 it goes
FIRST until A1 is launched)** → **J3** → **J16** → **J10**. A slot waiting
on a kernel advances the next order. A slot that can advance none takes
"Next up" item 1.*

### J11 — Pilot 2: can a plain learner eat at the entry rung once the record's cures are in?
part: Learning to eat + His world · stage: 1 (ready-list item 4) · step: 4 (pilot) · prior: §13 (read it whole), §1, §8; rulings 30, 38, 39, 49
prior: the record predicts the lie-still slide wherever moving costs energy (Kwiatkowski 2023; Hwangbo 2019; Hollenstein 2022 with SAC) and its proven cure: ramp the cost in. Every published eater lived where moving drains nothing extra and ate after 100k–1M decisions. Ours beside theirs: movement drain 2–3× basal, +100 on eating, 300k decisions.
prior: B reproduced its two SAC rows. No plain learner has shown the task learnable at any rung — this pilot is that question. Predicted: arm K eats inside 300k; arm P stops the slide, meals near random; arm G unknown.
**RULED — do not re-decide:** the rule frozen at `e2c3c378` (`PILOT_B2.md`
§3, `PILOT_RULE`) stands as written: body eye, **body motor (ruling 49)**,
floors from the m-card (same frame as the arms), W = 2, seeds 402–406.
**Do.** (4) p3..p9 into each Kaggle session as it frees
(`scripts/launch_detached.sh /data/jack-logs/pilot_b2/launch_pN.log $PY -m
experiments.pilot_b2 pilot-kernel --name pN`). A cut or errored run is
re-launched under the rule's own cut clause, never dropped. (5) When all 10
are home: `pilot_b2 table` + `pilot_b2 readings` → `PILOT_B2.md` §4 — the
reading is the tool's, not re-derived by hand — and ONE INBOX line: the
reading per arm. Reported-only: it picks the SETUP for the decisive test;
no seat moves. (6) Whatever it reads, the next learner run is J15's E2c on
arm K's world; do not launch any other learner run on the pilot's result
before the Director's ruling.
serves: M3 · PLAN ready-list "B piloted in his world" · philosophy 3, 4, 7.
**done when:** `PILOT_B2.md` has the frozen rule, 40 runs and one reading per arm.
state: (3) RULE FROZEN 11-19 04:35 `e2c3c378` (`PILOT_B2.md` §3 + `PILOT_RULE`): seeds 402–406, W = 2, U 0.64 k0-bm / 0.80 k1-bm → floor(U) = 0, native_substep, body eye. **Motor frame body by the 07:40 written default** (no ruling landed); a `world` ruling re-drafts on seeds 407+ with the f-card arms.
state: 11-19 13:35 `d3815ce6` §3c (reported, no run; p0–p2 home, 0 cut/crash): movement energy = E0+intake−basal per energy death. Control K (moving free) reads ~0. B0/P/G: movement falls Q2→Q4 in 6/6 cells (25–55 %, ~0.21→0.12 × basal) while meals fall — the slide; entropy does not override the cost; P did not stop it. 13:22 all 5 sessions still RUNNING.
state: 11-19 12:12 p5 LAUNCHED `jack-cpu-1795090316` (`launch_p5.log`) into p2's session; p3 `…82879`, p4 `…82927` RUNNING since 10:08. J11 holds 3; the 4th went to J3 (rank 3). Launches were held ~4 min by the 2-detached cap (LAB's place_key fits).
state: `pilot_b2 table` PARTIAL 8/40 (reported, NOT the reading): 0 sustained lives in any arm; every learner's meals/life falls early→late (K 0.90→0.14, B0 0.30→0.12); K-lr0 (no learning) 1.40→0.62 is the best late row so far.
state: 11-19 12:45 `7d3b18fc` §3b (reported, no run): fixed ent_coef 0.1 × 8 action dims → entropy bonus ≤ 0.555/decision vs task reward −0.036 (meal-less K) / +0.031 (K-lr0 meals) — 15–18×; ≈ 111 discounted vs +100 a meal (Haarnoja 2018: too-small reward scale → near-uniform policy). E2c's trace tests it: std/saturation rising = confirmed.
next: p6..p9 as sessions free (p3/p4 ~17:30, p5 ~19:30); then `table` + `readings` → §4.

### J15 — The learnability proof, part 2: a forager his network can copy, then a learner given the direction
part: Learning to eat · stage: 1 (stage-1 line "a copy of it inside his own network, and a learner given the direction to food") · step: 2 → 3 · prior: §17 (read it whole); J12; ruling 50
prior: §17 — plain regression copies a teacher only where the student's own input determines the teacher's action (25 trajectories → 77–95% of expert); where the teacher knows more the student learns its average: 0.88 → 0.00–0.11 (Weihs 2021); MSE on a random-choosing script 0.67 → 0 (BeT 2022). Ours: f3/f4 holds a target on 0.551 of training decisions (measured, seeds 701–712; NOT 0.993) and for the other 45% steers along a random ray held 25 decisions — a coin the row cannot see.
prior: E2c — expected first reliable reach 50k–200k (inference from reacher-class state SAC, dense reward); survival by repeated eating at 300k is on NO record (nearest: SAC PointGoal1 27.0 at 1M, shaped), so a miss at 300k is weak evidence. Learnable: unknown.
**RULED (50) — E2b as frozen (`760ae70a`) is NOT RUN:** `kernel-full` takes
no session on it. The record predicts its miss for the teacher's coin, not
his eye. The stop clause meant the ENTRY rung: gate 2 passed at d0, so this
order stands. Nothing has been read from seeds 741–760.
**Do, rule first (ruling 33), in this order.**
(1) **A ruler the row determines — `f5`.** In `eye_forager.py`: a hand
forager whose every action is a function of the CURRENT body-frame row128
only — no remembered target, no random draw, no counter (when no food is in
the row: a fixed function of the row, e.g. steer to the widest open gap).
Planted test: the same row twice → the same action; a shuffled-order replay
of one life's rows → identical actions.
(2) **Gate 2 again on f5, frozen before the run:** same bar (≥ 5 floor
meals/life on ≥ 18 of 20), d0 only, **fresh seeds 281–300**, shuffled twin
must miss, no extension. One Kaggle kernel (rank 2). If f5 MISSES: write
"a memoryless function of this row does not meet the bar at d0 (k of 20) —
finding food on his eye needs memory" in `WORLD_CARD.md`; E2b closes as
"not clonable by a memoryless net, by measurement"; go to (4).
(3) If f5 passes: amend `forager_clone.py`'s docstring BEFORE collecting —
teacher = f5, the 0.551 reading and ruling 50 written in, everything else
(train 701–740, test 741–760, clone ≥ 2 of 3, permuted twin must fail)
unchanged; re-collect in-kernel, new manifest; then `kernel-full` (rank 5).
(4) **E2c, after J11's forty runs are home** (it uses J11's readings):
`handed_sense` with **`encoding="cossin"`**; brain B on arm K's world, 300k,
seeds 801–805; controls BOTH `zero` (primary: the field's) and
`random_point` (reported; it leaks odometry — say so); the rule says before
seed 801 that a miss at 300k is weak evidence. Amend the frozen E2c text
with these three lines before any run; 15 runs = 15 sessions at J11's rank.
serves: M3 · PLAN stage 1 "proof that the task can be learned with his senses" · philosophy 3, 4.
**done when:** f5's gate row, E2b's row or its closure, and E2c's row are in `WORLD_CARD.md`, each with its control.
state: 11-19 12:10 (2) GATE 3 MISSES `a8f41c0b`: f5 5/20 seeds (3.57 meals/life), shuffled g5 0/20 (2.00), b5 0.50. 71 % of f5's eats in the 21 capped (parked) lives; the bout counter cannot split one-food meals (eats = bouts, gap 5 < regrow). WORLD_CARD J15 (2) + E2b CLOSED for this teacher; (3) does not run.
state: (4) E2c rule AMENDED `35a6c2d2` (cossin; zero primary, random_point reported; 300k miss = weak). Kernel CODED `37e8cf32`: `pilot_b2 e2c-kernel e0..e3` (15 runs, 4 a kernel), refuses until J11's 40 are home. FOUND + fixed `HandedSense` under PilotLives (raised on env.w; would have stranded `_truncated`); tests PASS.
state: (1) f5 = `f5_act(row)` / `RowForager` (`eye_forager.py`, `test_f5` 5/5): memoryless row128 ruler; gate 3 ran `jack-cpu-1795073402` at `9882df51` on seeds 281–300, frozen before any cell.
state: 11-19 12:30 (4) E2c TRACED before seed 801: `run_b(trace=True)` (E2C_ARMS only; J11's arms/kernels untouched) logs every 5k decisions the action mean/std/saturation + SAC actor/critic loss, ent_coef. Reads only: `test_pilot_b_trace` lives identical on/off, other-seed control differs; e2c test 3/3. Why: J11 partial 12/40 — every learner ends BELOW K-lr0 (INBOX), and the pilot logs nothing to say why.
next: E2c waits for J11's 40 (p6..p9 still to launch). Then `scripts/launch_detached.sh /data/jack-logs/pilot_b2/launch_eN.log $PY -m experiments.pilot_b2 e2c-kernel eN` for e0..e3 at J11's rank; score with J11's frozen readings. Done when E2c's row is in WORLD_CARD.

### J14 — The free GPU week (opens Sat 11-21 00:00 UTC): brain A's rows, A in his world, the ear's row
part: His brain (A) + Ear · stage: 1 · step: 3 (reproduce) · prior: §2, §9 row 7, §7; J5 (`ca4dcf00`: 51.8 dec/s on a T4); rulings 33, 47, 48, 52
prior: §2's three rows at the early points of the published curves, ~2.6 T4-h each. A has reproduced nothing yet. Ear: BC-ResNet-1 (9.2K weights), Speech Commands v2 12-class, published 96.9 ± 0.30 over 5 seeds; ours: model vendored verbatim, 9,232 weights, corpus and official test set reproduce.
**Everything before Saturday is DONE** (rules frozen, kernels tested).
**From Sat 11-21 00:00 UTC, pushed HEAD, in this order (ruling 52):**
A1 → A2 → A3 (`scripts/launch_detached.sh
/data/jack-logs/brain_a/rows/launch_A1.log $PY -m experiments.brain_a.rows
kernel A1`) → A in his world (`… -m experiments.brain_a.in_world kernel`,
body eye + body motor, ruling 49) → the ear (`… -m experiments.kws_row
kernel s0 s1 s2 twin`) → extension seeds only if a rule calls them.
**A1 goes from the FIRST slot after 00:00 that runs; a slot refused on load
or a failed push means the next slot launches it — every JACK slot on
Saturday checks "is A1 launched?" before anything else.** Budget 30 h; a
quota refusal is final for the week → Colab through `gpu.py`. Brain B's
row 7 gets NO GPU hour (ruling 52: dropped as mis-cited at source).
serves: M3, M5 · PLAN ready-list "brain A reproduced … and smoke-tested in his world"; part "Ear" 2 → 3 · objective 3 · philosophy 2, 3.
**done when:** A1–A3 and the ear row are in `KNOWN_ANSWERS.md`, in band or a named miss, and A's smoke line.
state: A rows: rule frozen `286d6451` (A1 cartpole_balance, A2 swingup, A3 ball_in_cup; seeds 0–2, ext 3–4; never-trains twin below band else VOID); kernels + tests `31286dac`; control corrected to replay_ratio 0.0. Nothing left for A1–A3 before Saturday.
state: (a) DONE `57574dc7`: rule frozen in `experiments/kws_row.py` (authors' clone sha256-pinned, band 96.3–97.5, seeds 0–2 + ext 3–4, counts 36923/4445/4890 or stop, shuffled twin ≤ 15.0%); box smoke ran the real main.py 200 epochs both arms, guard fired; `test_kws_row` PASS. T4 cost a GUESS (1.5 h/run) — first kernel measures it. Ear on 11-21: `scripts/launch_detached.sh /data/jack-logs/kws_row/launch_k1.log $PY -m experiments.kws_row kernel s0 s1 s2 twin`.
state: (b) DONE `388c2582`: `experiments/brain_a/in_world.py` — rows' BASE unmodified, env swapped via sheeprl `env=gym` + gymnasium `module:id` (no runner edit); pilot world arm K, body eye, body motor (INBOX default; `--motor world` exists). DONE rule in docstring: steps / ≥2 logged loss steps / all finite / dec/s. Box 1,200-decision smoke: rc 0, env ~290 dec/s, losses finite at the one logged step (reward 5.537 vs §2 5.541); the learns clause is left to the kernel.

### J3 — The home-made brains' fair chance on the published rows (owner 11-15)
part: His brain · stage: 1 · step: 3 (reproduce) · prior: §15, §5; rulings 24, 33, 44
prior: §15 — all seven arms are ONE on-policy AWR learner with different auxiliary losses. Tuned PPO itself misses row 1's 60k rule (reaches at ~60,700). Measured so far: five arms FAIL both rows with controls holding; `dreamer-xs` ≈ `wm-latent` (−359 vs −364), so the gap is the shared learner, not the home-made losses. `wm-efe` is `dreamer-xs` on these rows (115 tensors identical).
prior: no in-house arm has reproduced a row; B has (rows 1b, 2b, 3, 3b).
**Do.** (4) Last plain arm: `wm-latent+ring` (`scripts/launch_detached.sh
/data/jack-logs/j3_inhouse/launch_ARM.log $PY -m experiments.inhouse_rows
kernel wm-latent+ring`), rule `3054fbe6` unchanged. (5) **The repaired
entry — the fair chance:** `… kernel wm-latent@repair` (`94406999`: noise
held 32 decisions, γ 0.99, fresh seeds 5–9, same verdict clause;
never-trains control must fail). If it passes, the other arms get the same
repair, one session each. If it fails, every arm is set aside with its
numbers on record (PLAN: "may return if it is repaired") and J3 closes.
serves: M3 · PLAN "a fair chance for our own brains" · philosophy 3, 4.
**done when:** `KNOWN_ANSWERS.md` has `wm-latent+ring`'s row pair and the repaired entry's row, pass or fail.
state: 11-19 02:23 `wm-dyn` HARVESTED (`bc08a6ef`) → KNOWN_ANSWERS Row J3-4c: FAIL both rows, controls 0/5. Row 1 best windows −547…−874 (learns −1300→−650…−1130, 300 eps); row 3 0 flag reaches. Below wm-latent on row 1.
state: 11-19 04:40 `dreamer-xs` HARVESTED → KNOWN_ANSWERS Row J3-4d: FAIL both rows, controls 0/5. Row 1 best −392.7, −398.9, −340.0, −286.2, −378.5 → 1/5 reach (first in-house seed in band); row 3 0. (6) read: equal to wm-latent (mean −359.2 vs −364.4). Corrected J3-4b: wm-latent s0 last-10 −839.6.
state: 11-19 05:45 (5) WIRED `94406999` (no session free; p0/p1/p2/J10 s1 all RUNNING): arm `wm-latent@repair` = same core/loop, two settings changed — noise direction held 32 decisions (std schedule kept), GAE γ 0.99; fresh seeds 5–9; plain cells byte-identical; test 3/3.
state: 11-19 12:12 (4) `wm-latent+ring` LAUNCHED `jack-cpu-1795090393` at `71324fae` (log `/data/jack-logs/j3_inhouse/launch_wm-latent+ring.log`, est 3 h) into gate 3's freed session (rank 3). First try died on the dirty-tree guard (my uncommitted edit); re-launched after the commit. MY retry loop then launched a DUPLICATE at 12:19 (`jack-cpu-1795090752`, still running on Kaggle, ~3 h wasted); its local watcher was killed (PIDs 510947/510938) so only `…090393` writes `wm-latent+ring.json`. The launch log was truncated by the duplicate: the push receipt for `…090393` is in `gpu_submissions.jsonl`.
next: harvest `wm-latent+ring` → KNOWN_ANSWERS row; then `wm-latent@repair` in the next free rank-3 session.

### J16 — Brain A on the Mac: the port's rung 1 (G0 passed 11-19)
part: His brain (A on the Mac) · stage: 1 (ready-list "first-day speed test for the Mac") · step: 2 (combine) · prior: `docs/research/PUBLISHED_EVIDENCE_2026-11-15_PORT.md` (its step table, rows 0.4–1.3; gate G1); `experiments/brain_a/README.md`
prior: nobody has published Dreamer on Apple hardware; the official code cannot use the Mac's graphics chip. G0 PASS (12m compiled first step 216 s, 1.8 GB). x86 MLX timings are 7.4× off the box's ARM core — no speed is read anywhere but the Mac.
**Do, box only (< 15 CPU-min a run, nice 19), the report's steps unchanged
and in its order.** (a) Row 0.4: the golden generator on the official code's
`debug` preset; **gate G1 — two runs give identical files; if not, stop and
write why.** (b) Row 1.2: rung-1 tests against NumPy 64-bit references,
committed while they FAIL. (c) Row 1.3: the maths module (squashing, bins,
two-hot with mirrored-difference decode, distributions) until rung 1
passes eager and compiled. Each module's docstring carries the "ours beside
theirs" table. No training, no speed claims.
serves: M3 · PLAN "the decisive test … on the Mac mini" — every Mac hour an hour of the real test · philosophy 5, 6, 7.
**done when:** G1's result and rung 1's pass are in `experiments/brain_a/README.md` with the test names.
state: **DONE `07a65c3d`** (11-19 08:25). (a) G1 met by 0.4e (IDENTICAL 1,400/1,400), in README. (b) `rung1.py` tests: refs vs official 25/25, stub 0/9, planted 9/9 caught.
state: (c) `mlx_maths.py` filled: **rung 1 PASS 9/9 eager AND compiled** (mlx 0.32.4, box); worst bins 1.6e-7, decode 5.7e-7 (cond), grads ≤ 1.6e-7; uniform/mirrored decode exactly 0 both modes; planted still 9/9. Test names in README §J16 (c).
state: FOUND: first fill FAILED bins (1.92e-6) — the official grid is what XLA:CPU emits, −20·fma(−i, f32(1/127), 1): 0/128 misses vs 41 for `mx.linspace`. Parity = the compiler's output, read from optimized HLO. Decode deviates on purpose: (p3 − p1[::-1])·b3.
next: (report's row after 1.3) the same `rung1.py` on the PyPI MLX wheel, then rung 2 — for the Director to order.

### J10 — The walking layer: reproduce a published walk (owner 11-15)
part: Body and movement · stage: 1 side job (stage 3's part) · step: 3 (reproduce) · prior: §16, §4; ruling 45
prior: §16 — the cheapest published walk: SAC, Zoo entry, Humanoid-v4 — 4,825 [3,784–5,866] at 1M (10 seeds), 6,232 ± 280 at 2M. A return near 5,000 is STANDING: a walk row must also demand forward metres. The 2018 helper-force paper states no seeds or success rate.
**Do, Kaggle only (rank 4).** (2′) The frozen row
(`experiments/humanoid_sac_row.py`): seeds 0–2 in 500k segments —
`kernel <seed> <segment>`; harvest s1 seg 0, then `kernel 0 1` (the first
real mount test — if `resolve_prev_dir` fails fast, fix and say), `kernel
1 1`, seed 2 after. (2) Yu 2018 stays third; no full run before (2′) has a
row. (3) Our humanoid: not before a row above reproduces. Inside him the
layer KEEPS LEARNING (owner 11-15 19:21).
serves: M4 · PLAN stage 3 "a separate, fast walking layer is built and checked in parallel from now" · philosophy 1, 2.
**done when:** row (2′) is in `KNOWN_ANSWERS.md`, pass or fail.
state: (2′) RULE FROZEN `experiments/humanoid_sac_row.py`: Zoo entry, seeds 0–2, PASS R ≥ 5,950 AND ≥ 12 m/1,000 steps on ≥ 2/3, 1M median in [3,784, 5,866], late-start control < 500 else VOID; resume = 500k segments, model + replay (float64, ~6.1 GB) carried; box resume_check OK (`23e375be`).
state: s0 seg 0 HOME (500k, 490k updates; output holds model.zip + replay.pkl); s1 seg 0 `jack-cpu-1795059827` HOME 07:08 (500k hit, 490k updates, 11,774 s train; `j10/sac_row/s1_seg0.json`). `resolve_prev_dir` (`27d5c364`) finds the segment-1 mount or fails fast (test 6/6).
state: 11-19 06:25 (2) `411ec079`: Yu rule's "2000 N" corrected to SPD gains (N/m sideways, N per m/s forward; human_walker.py:175-212); `stage_exits` per-stage log THRESHOLD / CAP / CUT wired into the smoke body; FOUND reverted iterations draw 20k samples the cap never counts — reported as headers × 20k. Verdicts must name "k of n stages advanced on the cap". Test 4/4.
state: 11-19 13:08 `kernel 0 1` TRIED into a 6th session → refused by Kaggle (cap 5, `jack-cpu-1795093735`, 0 s billed, ERROR row); nothing ran, s0 seg0's kernel output still the mount source.
next: `scripts/launch_detached.sh /data/jack-logs/j10/launch_s0_seg1.log $PY -m experiments.humanoid_sac_row kernel 0 1` when a session frees (J3's duplicate `…090752` ends ~15:20 → that session is rank 4's under the J11-holds-3 rule), then `kernel 1 1`.

**Parked or closed:** `D1.0`, `T2.01` (→ J10), `LF.06`, `LC.09`, `CR.01`,
`LF.04` — "to earn again" on the chosen brain (ruling 20). `NE.01` → stage 3
(ruling 35). `PS.02`/`.05`/`.06`/`.08` → stage 3 (ruling 42). `W1.04` →
re-read 11-25 (ruling 41). Brain B row 7 → dropped (ruling 52).
**Next up, in order (idle slots; box < 15 CPU-min unless said):**
1. The dial race's missing parts (`docs/research/prereg/
dial_gated_vs_uniform.md`): the eval-lives hook and the sham `success_fn`,
each with a planted test. Code only; it runs after J11.
2. **Needs-hidden known-answer (§18 item 4):** rule text + kernel body for
Dulberg's 4-need grid with DQN, 30k steps — the one sense ablation with a
published expected FAILURE. Ruling 33's form; no run before the rule.
   state: 11-19 RULE FROZEN + kernel body `5464c5b8` (`experiments/needs_grid_row.py`; read from the e-print 2204.06608, not the summariser): 6 cells × seeds 0–9 (dqn/gmq @ anneal 5000 and 1, hidden@5000, frozen control > 5.0 else VOID); bands from the figures. Grid is 9×9 in a wall ring, not 8×8; the paper has NO needs-hidden arm (R3 is TEST, EVAAA's prediction). `test_needs_grid_row` 4/4.
   next: box smoke 117 s / 1,500 DQN steps → a GPU kernel (~2 T4-h for 60 lives), not CPU. Launch `$PY -m experiments.needs_grid_row kernel <cells…>` when the Director gives it a GPU slot; `score` judges.
3. Brain A fed one key per sense (§18 item 2): a design note in
`brain_a/in_world.py`'s docstring — the keys, the width-normalised loss,
ours beside theirs. No code change to the frozen A1–A3 rows.
   state: 11-19 10:45 NOTE WRITTEN in `brain_a/in_world.py` docstring (text only; rows + smoke untouched). Widths measured: vision 32, audio 16, touch 8, proprio 12, needs 6 = 74. sheeprl 0.8.5 keeps only the FIRST mlp key on a Box env → per-sense needs a Dict-space `make`; then one decoder head per key with no runner edit. Its loss SUMS keys by width (needs pull 8 %); width-normalised loss and whole-sense dropout both need a runner change → raced arms, not defaults. Hide-at-test is env-side, no edit.
   next: after J11 — `keys=per_sense` path in `make` + smoke against this file's DONE (1)–(4).
4. `row128` vs 16 rays under B (body frame): rule text only, after J11.
5. Stage-2 rule, first draft from `STAGE2_PREDICTION.md` + J11's readings
→ `docs/research/prereg/stage2_a_vs_b.md` (text; the Director freezes it).
6. In-house arms that pass J3 (5) join a pilot under J11's winning setup.

---

## SYSTEM — work on the machine (max 3 open)

*Order of work: **S9** — and nothing else in this lane until ruling 43's
fates are executed, except a fix a RUNNING Jack kernel needs (ruling 54) →
**S11** → **S6** (step 3).*

### S9 — Every settled FAIL gets a named fate; the two red ratchets get an owner
part: (machine — nothing faked) · stage: — · step: —
**Do, ≤ 8 ids per slot; never a baseline raise, never a re-run for a better
number.** Fates: (i) `repaired_by` naming the order or milestone; (ii)
`FAIL-DISPOSED: <authority> <date> — reason` (marker BEFORE any `COVERS:`
text); (iii) an INBOX ask. **Ruled today (ruling 43):** `PS.05`, `PS.06`,
`PS.08` → (ii) `Director ruling 42 2026-11-18 — the generic probe is the
defect (measured `441ac2f1`); a new probe is pre-registered at the first
stage-3 sitting or 2027-01-15`. `T0.17` → (i): demote `T2.14`'s row by the
registry's own drift mechanism (its PASS was bought under claim words since
amended; it is not re-run on the old core), then T0.17 re-earns. `T0.23` →
(i) repaired by this order: **no conjunct changes**; make `run ratchets`
green honestly — attribute the 6 `gpu_unattributed_jobs` above the floor (or
name each that cannot be and why), record the fallen floor
`live_unauditable_pairs` 22 → 17 — then T0.23 re-earns. `T0.32` → (i): it
stays FAIL; none of the 16 impls is grandfathered; each gets
`rtf.require_feasible` in the commit that next re-runs it (a line in
`LESSONS_CORE.md`). Left: DP.05 HR.5 LT.02 T0.35 T3.09 T4.02 W.2 W1.00 WM.01
XL.01 (the last two are the ones `coverage` names unowned today).
serves: philosophy 4 · the auditor's exit codes.
**done when:** `coverage` and `run ratchets` exit 0 with this id list empty.
state: slots 1–3: 18 fated by (ii); floors UNREACHABLE 115→107, PASS-ON-DEAD-DEPENDENCY 6→4; `coverage` exits 0; FAIL-UNOWNED floor 1 → 0 waits for the empty list.
state: 11-19 Director: untaken 27 h; `run ratchets` still exits 2 on exactly ruling 43's three items (`fail_unowned` 2 > 1; `gpu_unattributed_jobs` 27 > 21; `live_unauditable_pairs` 17 < 22). Start with ruling 43's six ids. 06:58: `WM.01` and `XL.01` read FAIL-UNOWNED until this order named them (`coverage` exits 0 again); give each a real fate (`XL.01` waits on `PS.02`, stage 3), then lower both floors in one commit: FAIL_UNOWNED 1 → 0, UNREACHABLE 107 → 104.

### S11 — Two steering files are over their caps; the root disk leaks at the source
part: (machine — bounded steering) · stage: — · step: —
**Do, one slot each, no finding dropped.** (a) `docs/FIELD_WATCH.md` is
14,549 B against 12 KB: `scripts/field_watch_prompt.md` gets one line —
"the page is rewritten inside 12 KB each sweep; NO-ACTION is one line per
seat"; trim today's page to the cap by moving nothing live. (b)
`scripts/research_prompt.md` line 50: the file cap of 70 KB is replaced by
"≤ 6 KB a section; workers read one section" (ruling 55) — a prompt edit
that removes words, adds none. (c) Root disk (5.3 GB, falling 0.1–0.3 a
day): the weekly prune of old CLI copies in
`~/.local/share/claude/versions/` that keeps the launcher's target and
anything a running process has open — with a planted test; run by hand
first, no new cron.
serves: philosophy 5 · standing objective 4 · auditor bookkeeping line 8.
**done when:** FIELD_WATCH ≤ 12 KB, the two prompt lines changed, and the prune has run once with its before/after free GB in the state line.
state: (new 11-19.)

### S6 — A video the owner can watch (owner request 11-12): step 3 left
part: (machine — he can be known) · stage: — · step: —
**Step 3:** `scripts/life_video.py` renders the most recent RECORDED life,
no picking (record poses if none are); `/data/jack-video/latest.mp4` +
`latest.json` as steps 1–2 made them. By hand, nice 19, no cron, never
inside a scored run.
serves: M8 · philosophy 1 (he can be known).
**done when:** the mp4 of the latest recorded life exists and plays.
state: steps 1–2 DONE (`070cea18`, `10b72b29`): seed 153 random life, 41 s mp4, overlay + summary json.

**Next up:** (0) `prereg.judge()` door for unequal arms — before the
decisive test's rule. (1) A detached job still `queued` on Kaggle after 2 h
gives up its box seat and writes one INBOX line (ruling 41). (2) `run
status`'s STEERING-PAGE ORDERS parser reads this page's `### J`/`### S`
headings. (3) **S8** — a seat held BY RECORD in `champions --check`. (4)
`verify` VENUE undeclared 266 vs baseline 256. (5) `XL.00` re-buy with a
planted positive (stale 73 d, in front of 19 specs). (6) `run.py` inside a
worktree writes that worktree's ledger. (7) **S4** — six PASSes gated on a
seed average that admits a failing seed. (8) `T0.13`'s dead clauses;
`BD.01`/`LC.08` re-class.

---

## INBOX — one line each, workers add, the Director empties
(emptied 11-19 06:41 — every line of 11-18/19 is ruled in 49–56 or became J15, J16, S11, Next-up 1–3.)
- 11-19 07:30 JACK J15: f5 cannot know which texture is food (geom ids, so textures, shift per world) and never gives a thing up — it parks; a seed can pass gate 3 by sitting on one regrowing food. If gate 3 misses, the order's closure sentence over-reads one hand ruler: propose "THIS memoryless hand rule does not meet the bar (k of 20)". Also disclosed in the rule: seed 281's world was built once to print its geom table (no policy stepped).
- 11-19 08:46 JACK Next-up 1 DONE `95e2e4cb` (all 4 Kaggle sessions RUNNING, J16 done): `sham_success_fn` + eval-lives hook (`eval_plan`/`plan_sha`/`eval_lives`/`make_eval_callback`/`final_rate`) in `dial_curriculum.py`; `test_dial_race_parts` PASS 4/4 (eval on vs off: B's params, replay, VecNormalize identical; planted leaky eval caught). Strike item 1; the hook needs jack-std (sbx).
- 11-19 10:30 JACK Next-up 2 DONE `5464c5b8` (all 4 Kaggle sessions busy): Dulberg needs-grid rule + kernel body frozen; it is a GPU kernel (~2 T4-h; 55 min per CPU life measured) — needs a GPU slot from the Director (J14's week or a free session). §18's Dulberg line was summariser error (8×8, "4→8 needs"); corrected from the e-print.
- 11-19 12:20 JACK J15: gate 3 MISSES — f5 5/20 (3.57 meals/life; shuffled 0/20, random 0.50); 71 % of its eats from 21 parked lives. Wrote the closure for THIS hand rule only (the 07:30 line's wording), E2b closed, E2c amended + kernel coded (waits for J11's 40). Machine: `world_card.BOUT_GAP` = 5 decisions is shorter than food regrowth, so `floor_bouts` = `floor_eats_all` on every row — the 'meals from one food' measure the rule promised cannot be read; a per-food eat count per life would.
- 11-19 12:30 JACK J11 partial (12/40, reported, NOT the reading): every learner arm ends BELOW its untrained twin in the same world — late meals/life K 0.14, KP 0.14 vs K-lr0 1.01; last-50k meals/1000 K 0.02/0.10 vs K-lr0 0.23/0.56 — and K's reward/life FALLS with learning (−19→−94, 2→−75). K/KP lives do not lengthen (~550 s flat), so in the free-movement world this is not the lie-still slide; B0/P/G lives do lengthen 276→~500 s (the slide signature). The 5k warm-up covers 1–3 of the "early 5 lives" (post-warm-up lives tell the same story). The pilot saves no loss/entropy trace, so the cause cannot be read from it; E2c now logs one (J15 state). For the Director before J11's ruling: a learner made worse by learning is an apparatus question (reward scale vs fixed ent_coef 0.1, the critic), not only a world one.
- 11-19 12:40 JACK: a worker-written retry loop around `launch_detached.sh` cannot tell that the same kernel was already launched by hand, so it launched J3's `wm-latent+ring` twice (one Kaggle session wasted ~3 h). The launcher could refuse a second live watcher with the same command line.
- 11-19 12:45 JACK J11: one priced cause for 'learners end below K-lr0' (`PILOT_B2.md` §3b): SAC's fixed ent_coef 0.1 makes the entropy bonus up to ~15–18× the task reward per decision (≈ a meal's worth of discounted value everywhere). If E2c's trace shows action std rising, the ruling J11 (6) waits on should consider a raced arm with ent_coef "auto" or a reward scale — not a change to the frozen rule.
- 11-19 13:10 JACK: MEASURED the Kaggle CPU cap is **5** batch sessions, not 4 — 5 ran at once all afternoon (p3, p4, p5, J3 ×2), and a 6th push (J10 `kernel 0 1`, `jack-cpu-1795093735`) was refused: "Maximum batch CPU session count of 5 reached". The 4-session ranking leaves one session idle whenever no duplicate is running; Director to re-rank for 5.
- 11-19 13:10 JACK (machine, for SYSTEM): `gpu.py` logs that refused push as `rc=0` and then raises the unclear "job produced no result: no artifact named 'segment.json'" — the push error text (`Kernel push error: …`) should be read as a refusal and named in the error.

- 11-19 13:35 JACK J11 §3c (12/40, reported, no run): where moving costs energy (B0, P, G) learning CUTS movement work in 6/6 cells (25–55 %) while meals fall — the predicted slide, and P's positive-only reward did not stop it in this partial; control K (moving free) reads ~0, so §3b's entropy cause stays open only for K's world and only E2c's trace can read it.
- 11-19 13:45 JACK: third JACK slot running with nothing to advance. All 5 Kaggle CPU sessions are RUNNING (p3, p4, p5, J3 ×2; checked with `kernels status`); W46's GPU quota is spent (33.8 h of 30) until Sat; Next-up 1–3 are DONE (I re-checked `95e2e4cb`) and 4–6 say "after J11". The first session frees ~15:20 (J3's duplicate → J10 `kernel 0 1`), then p3/p4 ~17:30. Slots before then idle unless the Director arms a box-only JACK item (<15 CPU-min, not gated on J11).

## DONE (kept one day)

- **J12 — his eye, body-frame** (`7d547777`): `eye_frame="body"` and `motor_frame="body"` built and proven; gate 2 on fresh seeds: d0 PASS 20/20 (7.53 meals/life, shuffled 0/20), d = 1 FAIL 3/20 (3.92 meals/life); body eye + world motor 0/20. Compass-eye rows labelled "ceiling only".
- **J13 — MLX gate G0 PASS** (`bc08a6ef`): 12m compiled first step 216 s < 600, peak 1.8 GB < 8, compiled == eager. → J16.
- **S10 — regate re-buys ≤ 900 s certificates** (`5a11800f`).
- **J11 (1)–(3):** f-card and m-card complete (16,000 learner-free lives), rule frozen `e2c3c378`, readings coded before data `ebb59a38`.
- **J3 (4):** `wm-dyn`, `dreamer-xs` FAIL both rows, controls hold (KNOWN_ANSWERS J3-4c, J3-4d).
- **Idle slots 11-18/19:** `STAGE2_PREDICTION.md`; brain B rows 5–6 rule + kernels (`0bf84989`); dial race rule text; Mac rows 11–12 runnable (`25a13d60`); `STAGE3_NOTES.md`; PRIORS §17, §18.
