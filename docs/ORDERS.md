# ORDERS — what the workers do. Written by the Director.
Director sitting: 2026-11-18 06:42 UTC

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5, **each ≤ 4 lines of text — replace,
do not append; history is in git**). One state+journal commit per slot.
Machine problems go under INBOX, one line each. Rulings 38–48 cited here are
in `docs/DECISIONS_RESOLVED.md` under 2026-11-18 06:42.

**`docs/PLAN.md` binds every order (owner, 11-15).** Research → combine →
reproduce → test the last 10% → put it in him. We are in **stage 1: the
instrument.** `docs/research/PUBLISHED_PRIORS.md` §1–§16 is the reference —
read your order's section at the start of every slot.
**Where stage 1 stands (11-18).** The floor card is complete (8,000
learner-free lives): doing nothing sustains 2 lives in 2,000, random
movement 0.25%, random movement with movement free 2.25%. Two faults were
found before any learner ran: (a) 5% of layouts laid a food on the ramp,
where it rolls to a statue — repaired by `food_floor_only`; (b) his eye, old
and new, is cast at WORLD azimuths — a free compass the human-like body will
not have. **Ruling 38: his eye is body-frame from now on; nothing a brain
learns is believed on the compass eye.** The 128-ray row carried the hand
forager 20/20 at d = 1 (compass eye — a ceiling, to be earned again
body-frame). The simulator is 2.4× faster with an identical world (J6 done).
Entry rung stays **d0 = 0.75**.
**Every new run in his world passes** `lazy_contact_audio=True,
native_substep=True` (identical world, proven on the box and on Kaggle). The
integrator is **not** switched (ruling 40): under brain B the learner, not
the simulator, is the clock.
**Stopped (rulings 10, 20, 23, 36):** races between home-made learners
outside J3; forager cells on the 16-ray eye; world-rule races; `D1.0`,
`T2.01`, smell runs; new specs with no published prior. A question with no
prior is a `RESEARCH:` line in INBOX.
**Kaggle CPU: 4 sessions, RANKED.** A free session goes to the first READY
kernel: (0) any kernel estimated < 1 h; (1) J11 — but **J11 holds at most 3
sessions while a J12 gate kernel is ready and waiting**; (2) J12; (3) J3 —
one session; (4) J13 G0; (5) J10's SAC row; (6) J15. `W1.04` is PARKED
(ruling 41): no session before 11-25; its ERROR row stands; no seed has run.
**The box (ruling 32).** Day ceiling 57,600 CPU-s, ≤ 2 computing detached
jobs. Whole run < 15 CPU-min → box, nice 19. 15–60 CPU-min → box only when
all 4 Kaggle sessions are held, one worker, said in the state line. > 60
CPU-min → Kaggle or it waits. Kaggle GPU: **0 h until Sat 11-21 00:00 UTC.**
**`/` has 5.7 GB free, workers refuse under 3:** scratch, clones and caches
live under `/data` only (`UV_CACHE_DIR`, `PIP_CACHE_DIR`, `XDG_CACHE_HOME`
under `/data/caches/`, `--only-binary=:all:`).
**World files** (`w0.py`, `needs.py`, `survival.py`, `drives.py`): changed
only through `jack_env.py` options whose default is today's world.
**Known-answer rows (ruling 33):** seeds AND at most one extension stated
BEFORE the first seed runs. A miss after that is a miss.
**Receipts:** the slot that launches a kernel commits
`experiments/gpu_submissions.jsonl` and `cpu_budget.json` by name
(`cpu_budget.json` is dirty right now — the next slot commits it).

---

## JACK — build the creature, his world and his curriculum (max 7 open, ranked)

*Order of work: **J12 item (0)** (one slot, it gates J11's pilot) → **J11** →
**J12** → **J3** → **J14** → **J15** → J13 → J10. A slot waiting on a kernel
advances the next order. The 07:07 and 19:07 strong slots take J11's rule
first. A slot that can advance none takes "Next up" item 1.*

### J11 — Pilot 2: can a plain learner eat at the entry rung once the record's cures are in?
part: Learning to eat + His world · stage: 1 (ready-list item 4) · step: 2 → 4 (pilot) · prior: §13 (read it whole), §1, §8; rulings 30, 38, 39
prior: the record predicts the lie-still slide wherever moving costs energy (Kwiatkowski 2023; Hwangbo 2019; Hollenstein 2022 with SAC) and its proven cure: ramp the cost in. Every published eater lived where moving drains nothing extra and ate after 100k–1M decisions. Ours beside theirs: movement drain 2–3× basal, +100 on eating, 300k decisions.
prior: B reproduced its two SAC rows. No plain learner has shown the task learnable at any rung — this pilot is that question. Predicted: arm K eats inside 300k; arm P stops the slide, meals near random; arm G unknown.
**RULED (39) — do not re-decide:** the pilot's world is **option (a):
`food_floor_only=True`**, `spawn_clear` R = 1.55 unchanged, and **every
floor is re-measured on that world: f0..f7 all run, nothing borrowed from
the leaky card, no ring filtered out.** All learner arms see the
**body-frame eye** (`eye_frame="body"`, 16 rays; J12 item 0 — the pilot
kernels refuse without it). `native_substep` on.
**Do.** (1) Launch f3, then f4..f7 as sessions free (rank 1; commands in
`pilot_b2 kernel --name fN`). (2) When f0..f7 are home: `pilot_b2 card
--glob 'f*.json'` and `rule-inputs` → `PILOT_B2.md`. (3) THE RULE, committed
before any learner seed: W = the smallest W at which the f-card predicts
≤ 0.5 floor crossings over the pilot's floor lives; floor clause = Poisson
95% upper bound of the named f-card arm; **each arm's floor named** (B0, P,
G vs held8-k1-ff; K, KP vs held8-k0-ff; K-lr0 vs held8-k0-ff as a bound,
said so); T = 300,000; seeds 402+; arms × 5: B0, K, P, G, KP + floors noop,
held8, K-lr0; the three readings (sustained / eats but starves / slides)
defined in numbers; how a cut run counts; what would mean "our code is
broken". B0 is J8's setup **except** the repaired world and the body-frame
eye — say so in the rule. (4) `pilot_b2 pilot-kernel --name p0..p9`, then
`pilot_b2 table` → `PILOT_B2.md`. Reported-only: it picks the SETUP for the
decisive test; no seat moves.
serves: M2→M3 · PLAN ready-list "B piloted in his world" · philosophy 3, 4, 7 (the eye the finished Jack will have).
**done when:** `PILOT_B2.md` has the f-card, the frozen rule, 40 runs and one reading per arm.
state: DONE (1)–(2): f-card 8/8 (`791d9b55`); m-card FINAL 8/8 (m7 home 04:09): held8-k0-ff-bm 9/2000 (upper 0.785%), held8-k1-ff-bm 1/2000 (upper 0.237%) — body-motor floor below world-motor on both arms (P 2.6e-8 / 0.0030). rule-inputs → W = 2.
state: (3) RULE FROZEN 11-19 04:35 `e2c3c378` (`PILOT_B2.md` §3 + `PILOT_RULE`): seeds 402–406, W = 2, U 0.64 k0-bm / 0.80 k1-bm → floor(U) = 0, native_substep, body eye. **Motor frame body by the 07:40 written default** (no ruling landed); a `world` ruling re-drafts on seeds 407+ with the f-card arms.
state: (4) p0 LAUNCHED 04:27 `jack-cpu-1795062415` (log `/data/jack-logs/pilot_b2/launch_p0.log`, est ≤ 7.5 h). Tests PASS: kernels (refusal now with an empty rule), frames, bm.
state: p1 LAUNCHED 04:37 `jack-cpu-1795063080`; p2 LAUNCHED 05:23 `jack-cpu-1795065783` (log `launch_p2.log`) into J10 s0 seg 0's freed session (home 05:17, 500k hit, `j10/sac_row/s0_seg0.json`). Sessions: p0, p1, p2, J10 s1 seg 0. 05:20 `ebb59a38`: §3's readings transcribed before any run is home — `pilot_b2 readings` (refuses a partial book); planted test 9/9; 4 seam choices (crash vs cut, noop floor, K/KP θ floor, slide's noop length) declared in PILOT_B2 §3, each the stricter.
next: p2..p9 into each Kaggle session as it frees (`scripts/launch_detached.sh /data/jack-logs/pilot_b2/launch_pN.log $PY -m experiments.pilot_b2 pilot-kernel --name pN`); J11 outranks J10's later segments. When all 10 home: `pilot_b2 table` + `pilot_b2 readings` → §4 (the reading is the tool's, not re-derived by hand).

### J12 — DONE `7d547777` — His eye: body-frame, 128-ray row, and the proof his senses can find food
part: Eye + Learning to eat (the learnability proof) · stage: 1 (ready-list "the eye designed") · step: 2 (combine) · prior: §12; LAB `find_himself.py` (`99e588f0`), `eye_streak.py`; rulings 37, 38
prior: §12 — a first eye of distance-and-material rays. Ours, measured: at d0 one level row of 128 azimuths finds floor food .909 (16 rays .528). The lab (11-17): both eyes are cast at world azimuths; place recall .957 with that compass, .933 without if the look may be turned, .54 ray-for-ray.
**Do.** (0) **FIRST, one slot, box:** `jack_env(eye_frame="body")` — ray
azimuths + the body's yaw, for `rays16` AND `row128`; default `"world"` =
today byte-identical. Tests: identity (default off); at yaw 0 body == world;
planted: turn the body 90° → the body-frame look is the world look rolled
by a quarter (control: world-frame look does not roll). (1) One line in
`WORLD_CARD.md`: does the hand forager f3 use heading it is not given (does
its steering assume ray j points along a world axis)? Read the code, answer
yes/no with the line number. (2) **The frozen gate again on the body-frame
eye, declared now (ruling 38):** same forager (if (1) says it assumed the
world frame, change ONLY the action mapping needed to steer from body-frame
rays and say so before the run), same bar (≥ 5 floor meals/life on ≥ 18 of
20 seeds), shuffled twin must miss, **fresh seeds 261–280**, d0 and d = 1,
no extension. Two Kaggle kernels. (3) Harvest the compass-eye d0 kernel
(`jack-cpu-1794960563`, `eye_forager score`) and label BOTH compass rows in
`WORLD_CARD.md` "compass eye — ceiling only".
serves: M3, M5 · PLAN part "Eye" · END-FIT row 1 (human-like body) · philosophy 4, 7.
**done when:** the option + tests, the heading line, and the body-frame gate row (both rungs, pass or fail) are committed.
state: (0) DONE `c593ec10`: `eye_frame="body"` for both eyes, default byte-identical 8/8; quarter turn rolls the look 20/20, world control 0/20 (`test_jack_env_eye_frame` PASS). He sits a median 81° off his birth heading under random actions. (1) DONE: YES — WORLD_CARD "J12 (0)".
state: FOUND: the compass is also in the MOTOR — W0's drive is a force along world x, y and proprio's two velocities are world x, y. Added `motor_frame="body"` (default identical; forward push follows his heading). `eye_frame` alone asks a learner to aim a compass push with no heading sense — see INBOX, it touches J11's pilot.
state: (2) gate 2 FROZEN `c593ec10` (`eye_forager.py` docstring): ruler unchanged line for line, mapping in the env (`motor_frame="body"`), seeds 261–280, rows f4 / g4 (must fail) / b4 / h4 (body eye + WORLD motor, reported only, predicted miss). d0 kernel `jack-cpu-1794987421` RUNNING 07:37 (est 5 h, log `/data/jack-logs/eye_forager/launch2_d075.log`).
state: DONE `7d547777` — (2) d = 1 HOME: f4 FAILS 3/20 (3.92 meals/life; seeds 262/264/280 only), g4 0/20 (1.24), b4 0.47, h4 0/20 (0.74); f4 beat its own b4 and g4 20/20. WORLD_CARD "J12 (2′) … d = 1". The proof stands at d0 only; the frozen ruler (world-axis dead-reckoning, not re-tuned) is a lower bound on the eye.
state: (3) DONE: compass d0 f3 20/20 (13.72 meals/life), shuffled 8/20, random 0.46; both compass rows labelled "ceiling only". Scar: with hand cal j0=alpha=1 + random_needs the body dies every ~3 decisions — use the shipped cal for any eye measurement.

### J3 — The home-made brains' fair chance on the published rows (owner 11-15)
part: His brain · stage: 1 · step: 3 (reproduce) · prior: §15, §5; rulings 24, 33, 44
prior: §15 — all seven arms are ONE on-policy AWR learner with different auxiliary losses. Tuned PPO itself misses row 1's 60k rule (reaches at ~60,700). Expected for the waiting arms: row 1 best windows −320…−420, 0–1 of 5; row 3 0 flag reaches (white noise reached the flag 0/600; held 32 steps 170/200).
prior: `wm-efe` is `dreamer-xs` on these rows (115 tensors identical, 5 heads uncalled). No in-house arm has reproduced a row; B has (rows 1b, 2b, 3, 3b; row 4 closed by ruling 33).
**Do.** (4) The frozen rule `3054fbe6` UNCHANGED, both rows: `dreamer-xs`,
`wm-dyn`, `wm-latent+ring` — one Kaggle session at a time:
`scripts/launch_detached.sh /data/jack-logs/j3_inhouse/launch_ARM.log $PY -m
experiments.inhouse_rows kernel ARM`; harvest into `KNOWN_ANSWERS.md`.
**`wm-efe` gets no session (ruling 44):** its row is written as "identical
to `dreamer-xs` on these rows", with the tensor receipt. (5) **The repaired
entry — this is the fair chance:** after (4), ONE re-entry under the same
rule on fresh seeds, the shared learner with noise held at action choice
and γ 0.99 (§15's named cure), first for `wm-latent` (the only arm that
learned); never-trains control must fail. If it passes, the other arms get
the same repair. (6) Read `dreamer-xs` beside `wm-latent`: equal → the 3×
was network size.
serves: M3 · PLAN "a fair chance for our own brains" · philosophy 3, 4.
**done when:** `KNOWN_ANSWERS.md` has one row pair per arm (4) and the repaired entry's row (5), pass or fail.
state: FAIL both rows, controls hold: `ppo-needs` (−578…−1120), `ppo-lp` (−824…−1059), `wm-latent` (best −321.0; learns, budget binds); row 3: 0 flag reaches each. `wm-dyn`, `wm-latent+ring` wired `8a87941a` (est. 0.8 h / 2 h).
state: 11-19 02:23 `wm-dyn` HARVESTED (`bc08a6ef`) → KNOWN_ANSWERS Row J3-4c: FAIL both rows, controls 0/5. Row 1 best windows −547…−874 (learns −1300→−650…−1130, 300 eps); row 3 0 flag reaches. Below wm-latent on row 1.
state: 11-19 04:40 `dreamer-xs` HARVESTED → KNOWN_ANSWERS Row J3-4d: FAIL both rows, controls 0/5. Row 1 best −392.7, −398.9, −340.0, −286.2, −378.5 → 1/5 reach (first in-house seed in band); row 3 0. (6) read: equal to wm-latent (mean −359.2 vs −364.4). Corrected J3-4b: wm-latent s0 last-10 −839.6.
state: 11-19 05:45 (5) WIRED `94406999` (no session free; p0/p1/p2/J10 s1 all RUNNING): arm `wm-latent@repair` = same core/loop, two settings changed — noise direction held 32 decisions (std schedule kept), GAE γ 0.99; fresh seeds 5–9; plain cells byte-identical; test 3/3.
next: `wm-latent+ring` kernel when a session frees after J11's p-runs claim theirs (J11 outranks); then (5) `…inhouse_rows kernel wm-latent@repair` (log `launch_wm-latent@repair.log`).

### J14 — The free GPU week (opens Sat 11-21 00:00 UTC): brain A's rows, A in his world, the ear's row
part: His brain (A) + Ear · stage: 1 · step: 3 (reproduce) · prior: §2, §9 row 7, §7; J5 (`ca4dcf00`: 51.8 dec/s on a T4); rulings 33, 47, 48
prior: §2's three rows at the early points of the published curves, ~2.6 T4-h each. A has reproduced nothing yet. Ear: BC-ResNet-1 (9.2K weights), Speech Commands v2 12-class, published 96.9 ± 0.30 over 5 seeds; ours: model vendored verbatim, 9,232 weights, corpus and official test set reproduce; ~18 CPU-h a seed on CPU.
**Do before Saturday (no GPU):** (a) ear rule, ruling 33's form, in
`experiments/kws_row.py`: the authors' recipe unmodified, seeds 0–2, one
extension 3–4, band from the paper's mean and SD written before seed 0, a
shuffled-label twin that must fall to chance; a GPU kernel body + 5-minute
CPU smoke. (b) `rows.py`-style kernel for **A in his world**: sheeprl
preset unmodified on `jack_env` (repaired world, body-frame eye), 20k
decisions, done = it steps, learns without NaN, and reports dec/s — a
smoke, no reading of behaviour. **On 11-21, pushed HEAD, smallest first
(ruling 47):** A1, A2, A3 (`scripts/launch_detached.sh
/data/jack-logs/brain_a/rows/launch_A1.log $PY -m experiments.brain_a.rows
kernel A1`), then (b), then the ear seeds. Budget: 30 h; a quota refusal is
final for the week → Colab through `gpu.py`.
serves: M3, M5 · PLAN ready-list "brain A reproduced … and smoke-tested in his world"; part "Ear" 1 → 3 · objective 3 · philosophy 2, 3.
**done when:** (a), (b) committed by Fri 11-20; then A1–A3 and the ear row in `KNOWN_ANSWERS.md`, in band or a named miss, and A's smoke line.
state: A rows: rule frozen `286d6451` (A1 cartpole_balance, A2 swingup, A3 ball_in_cup; seeds 0–2, ext 3–4; never-trains twin below band else VOID); kernels + tests `31286dac`; control corrected to replay_ratio 0.0. Nothing left for A1–A3 before Saturday.
state: ear: sized `9b086c32`, test set `a9043bd7`, target corrected `ac642f42`.
state: (a) DONE `57574dc7`: rule frozen in `experiments/kws_row.py` (authors' clone sha256-pinned, band 96.3–97.5, seeds 0–2 + ext 3–4, counts 36923/4445/4890 or stop, shuffled twin ≤ 15.0%); box smoke ran the real main.py 200 epochs both arms, guard fired; `test_kws_row` PASS. T4 cost a GUESS (1.5 h/run) — first kernel measures it. Ear on 11-21: `scripts/launch_detached.sh /data/jack-logs/kws_row/launch_k1.log $PY -m experiments.kws_row kernel s0 s1 s2 twin`.
state: (b) DONE `388c2582`: `experiments/brain_a/in_world.py` — rows' BASE unmodified, env swapped via sheeprl `env=gym` + gymnasium `module:id` (no runner edit); pilot world arm K, body eye, body motor (INBOX default; `--motor world` exists). DONE rule in docstring: steps / ≥2 logged loss steps / all finite / dec/s. Box 1,200-decision smoke: rc 0, env ~290 dec/s, losses finite at the one logged step (reward 5.537 vs §2 5.541); the learns clause is left to the kernel.
state: next (11-21, after A1–A3): `scripts/launch_detached.sh /data/jack-logs/brain_a/in_world/launch_k.log $PY -m experiments.brain_a.in_world kernel` → `inworld.json`; write A's smoke line in KNOWN_ANSWERS. Then the ear seeds.

### J15 — The learnability proof, part 2: the forager copied into his network, and a learner given the direction
part: Learning to eat · stage: 1 (stage-1 line "a copy of it inside his own network, and a learner given the direction to food") · step: 2 → 3 · prior: grep `E2b`, `E2c` in `PUBLISHED_PRIORS.md` and `WORLD_CARD.md`; J12
prior: §17 (filed 11-18; the pointer above named text that did not exist). E2b expected: MISS or marginal, learnable UNKNOWN. Plain regression is on record only for teachers the student's own input determines (25 trajectories → 77–95% of expert, ≤ 1,000 steps); where the teacher knows more, the student learns its average: 0.88 matched → 0.00–0.11 (Weihs 2021); MSE on a random-choosing script 0.67 then 0 (BeT 2022).
prior: settings that differ: our teacher holds a target on 0.551 of decisions in the training set (measured, seeds 701–712; NOT 0.993) and for the other 45% steers along one of ~38 open rays drawn at random and held 25 decisions — a coin the row cannot see; 9,000-step lives vs ≤ 1,000; teacher-only states. A 90-s probe on unseen worlds: R² 0.29 / −0.11 with a target, ≤ 0 without; push 0.24–0.38 vs the teacher's 0.63.
prior: cure on record if it misses: the teacher labels the student's OWN lives (CARLA 16 → 64 → 100%); offline history is the copycat (Ant 1,300 → −311). Known-answer for `train_clone`: D4RL hopper-expert 111.5 ± 1.3 — not run.
prior: E2c: expected first reliable reach 50k–200k (inf — reacher-class state SAC 746–919 at 100k with DENSE reward); survival by repeated eating at 300k is on NO record (nearest: SAC PointGoal1 27.0 at 1M, shaped), so a miss at 300k is weak evidence. Differs: sparse +100; bearing as radians/π jumps at ±π; the field's control is the input removed, and a fixed random point leaks odometry. Learnable: unknown.
**Not before J12 (2) has a body-frame row.** Then, rule first (ruling 33):
(E2b) clone f3 into brain B's actor network (same sizes, body-frame row128
input only) from the forager's own lives; pass = the clone meets the
forager bar (≥ 5 floor meals/life on ≥ 18 of 20 fresh seeds) at d0; control
= clone trained on shuffled rays must miss. (E2c) brain B with two extra
inputs (bearing and distance to the nearest food), K-arm world, 300k
decisions, 5 seeds, J11's frozen readings. If J12's body-frame gate FAILS,
this order becomes "say why" and stops.
serves: M3 · PLAN stage 1 "proof that the task can be learned with his senses" · philosophy 3, 4.
**done when:** both rows are in `WORLD_CARD.md`, pass or fail, each with its control.
state: 11-18 16:40 `6aae85f8`: E2c wrapper CODED ahead of J11's rule — `experiments/handed_sense.py` `HandedSense(env, mode, encoding)`: food / random_point (frozen control) / zero (INBOX 16:14, option only); radpi (frozen default) / cossin (option only). `test_handed_sense` PASS (planted geometry; his senses byte-identical on arm-K world; point held, kept through truncation, re-drawn per life). E2c still needs J11's frozen rule + a `run_b` call with the wrapper; the Director's 16:14 ruling picks encoding/control.
state: rules + collector DONE `760ae70a` (`experiments/forager_clone.py` docstring: E2b train seeds 701–740, test 741–760, clone ≥2 of 3 seeds, permuted twin must fail; E2c 300k × 5 seeds 801–805, random-point control). `test_forager_clone` PASS.
state: disclosed before data: the teacher holds a remembered target 0.993 of decisions (smoke seed 153) — a memoryless clone may miss for the forager's memory, not the eye.
state: 11-18 12:10 gate 2 passed at d0 → collect RUNNING on the box, 2 detached workers (`/data/jack-logs/forager_clone/collect_{a,b}.log`, seeds 701–720 / 721–740; ~3 h at ~39 dec/s). Student coded: `train_clone` (MSE, 256-256-256-8 ReLU/tanh, perm arm re-permuted every sight), `ClonePolicy`, `test_cell` via gate 2's own loop (`run_seed(policy=)`), `score`; `test_forager_clone_student` PASS 4/4.
state: 12:48 `65c648e7`: test kernel CODED (`forager_clone kernel`: 140 cells, refuses missing/uncommitted weights; `score [test.json]`); test PASS 5/5. Collect 8/40 (~24k decisions/seed, ~7.5 MB each). MEASURED: training costs 4.4 CPU-s/1e5 samples (clone), 8.5 (perm) → ~17M samples/student → 3 clones 38 CPU-min + 3 perms 74 = **112 CPU-min: over ruling 32's box line**; 300 MB teacher data cannot ride git (INBOX).
state: 13:25 FULL kernel CODED (`forager_clone kernel-full`): re-collects 701–740 in-kernel, checks each seed's (x,a) sha256 against `experiments/forager_clone_teacher.json` (mismatch reported, not fatal — declared in the docstring), trains the 6 students (train_clone unchanged), runs the 140 cells; fetches clone_full.json + 6 weights. Refuses without a committed 40-seed manifest. Test PASS 6/6. Est 9 h (GUESS) vs 9.5 h timeout.
state: 11-18 15:25 `ee9bc9de`: teacher manifest COMMITTED (40 seeds 701–740, 931,745 decisions, 17,280–27,000/seed); `full_ready()` = [] — kernel-full is launchable. next: at J15's rank when a Kaggle session frees (all 4 full at 15:23): `scripts/launch_detached.sh /data/jack-logs/forager_clone/launch_full.log $PY -m experiments.forager_clone kernel-full`.

### J13 — DONE (G0 PASS) — MLX port, stage 0 ends at gate G0 (owner 11-15)
part: His brain (A on the Mac) · stage: 1 (ready-list "first-day speed test for the Mac") · step: 2 · prior: `docs/research/PUBLISHED_EVIDENCE_2026-11-15_PORT.md` steps 1–3
**Do.** Run `experiments/brain_a/g0_kernel.py` on one Kaggle CPU session
when rank 4 gets one; the rule is the report's G0 unchanged. A failed G0
stops the port and is written up; a pass makes rung 1 the next order. No
other port work until G0 has run. Lead to weigh in the write-up: official
JAX ran 0.925 vs sheeprl 0.458 grad steps/s on this box (`9ba4e682`).
serves: M3 · PLAN "the decisive test … on the Mac mini" · philosophy 5, 6.
**done when:** G0's pass or fail is in `experiments/brain_a/README.md` with its numbers.
state: DONE `bc08a6ef` (11-19 02:24) — G0 PASS in `experiments/brain_a/README.md`: 12m compiled first step 216 s < 600, peak 1.8 GB < 8, finite; compiled == eager bit-for-bit.
state: FOUND: Kaggle x86 MLX steps 1m at 17.4 s vs 2.34 s on 1 box ARM core (7.4× slower); 12m 212 s/step. x86 CPU timings are not the Mac's — speed only from the Mac's first-day rows. Next per the report: rung 1.

### J10 — The walking layer: reproduce a published walk (owner 11-15)
part: Body and movement · stage: 1 side job (stage 3's part) · step: 3 (reproduce) · prior: §16, §4; ruling 45
prior: §16 — the 2018 helper-force paper states no seeds, success rate or test of "walks", and its code advances a capped stage silently; nobody has reproduced it. The cheapest published walk: SAC, Zoo entry, Humanoid-v4 — 4,825 [3,784–5,866] at 1M (10 seeds), 6,232 ± 280 at 2M. A return near 5,000 is STANDING: a walk row must also demand forward metres.
**Do, Kaggle only, rule first.** (2′) **The one-kernel row first:** SBX SAC,
the Zoo's Humanoid-v4 entry unmodified; pass = return ≥ 5,950 AND ≥ 12 m
forward per 1,000 steps at 2M; check at 1M inside [3,784, 5,866];
late-start control must fail. Box-measured 0.046 CPU-s a step → the rule
says before seed 0 how the run resumes across sessions if 2M does not fit
one. (2) The Yu 2018 row stays, third: before any full run add a per-stage
exit-reason log (threshold or cap) to `yu2018_walk.py`'s rule and correct
"2000 N" to gains. (3) Our humanoid: not before a row above reproduces.
Inside him the layer KEEPS LEARNING (owner 11-15 19:21).
serves: M4 · PLAN stage 3 "a separate, fast walking layer is built and checked in parallel from now" · philosophy 1, 2.
**done when:** row (2′) is in `KNOWN_ANSWERS.md`, pass or fail.
state: (2′) RULE FROZEN `experiments/humanoid_sac_row.py` (11-18 slot): Zoo entry (2M, learning_starts 10k, SB3 defaults set explicitly in SBX), seeds 0–2, PASS R ≥ 5,950 AND ≥ 12 m/1,000 steps on ≥ 2/3, 1M median in [3,784, 5,866] else named miss, late-start control < 500 else VOID; resume = 500k/10.5 h segments with model + replay buffer (~3.1 GB) as kernel source.
state: `61e2950f`: Zoo entry VERIFIED at zoo `d802099a` (matches); SBX hidden defaults = SB3 non-SDE; rule CORRECTED before seed 0: metres = centre-of-mass x (v4 `x_position`), not torso. `evaluate` + `control_run` coded; box control 50k: R 67.96, 0 updates, holds. next: the training-segment body (500k/10.5 h, model + replay buffer carried, resume test on box at ~20k), then launch at rank 5.
state: 11-18 10:29 `23e375be`: `train_segment` (resume as the rule says) + box `resume_check` 10k+10k: step counter, buffer, n_updates (9000→19000), all params + actor optimiser equal after load, fresh control differs — RESUME OK. Box ~0.018 wall-s/step. Found: replay is float64, ~6.1 GB at 1e6 (rule said 3.1; fits Kaggle 20 GB). Wall-stop callback not exercised. next: kernel wrapper (seg 0 fresh; later segs attach previous output as kernel source), launch seed 0 at rank 5.
state: 11-18 10:50: Kaggle wrapper `humanoid_sac_row kernel <seed> [segment]` + job text (pins = jack-std venv; fetches only segment.json); `tests/test_humanoid_sac_row_kernel` PASS 4/4. BLOCKED on the machine (INBOX 10:50): gpu.py has no `kernel_sources` (segment ≥1 cannot mount the checkpoint; wrapper refuses) and downloads ALL output (seg 0 would pull ~6.1 GB to /data). next: when both land, launch seed 0 seg 0 detached at rank 5.
state: 11-19 02:25 SEED 0 SEGMENT 0 LAUNCHED into G0's freed session (machine blocker cleared by SYSTEM `ee81dffd`: kernel_sources + fetch_only; kernel test PASS 5/5) — `jack-cpu-1795055129`, log `/data/jack-logs/j10/launch_s0_seg0.log` → `/data/jack-logs/j10/sac_row/s0_seg0.json` (~10.5 h). next: harvest it, then `kernel 0 1` (first real test of the segment mount path).
state: 11-19 03:44 SEED 1 SEGMENT 0 LAUNCHED into the 4th session, idle since m6 came home at 02:27 (J3 holds one session, J11 m7 one, J15 held, J14 not before 11-21): `jack-cpu-1795059827`, push rc=0, log `/data/jack-logs/j10/launch_s1_seg0.log` → `s1_seg0.json` (~10.5 h). A fresh segment 0 never touches the unverified mount. next: harvest s0/s1 seg 0 → `kernel 0 1` first (mount test), then `kernel 1 1`; seed 2 seg 0 into the next free session.
state: 11-19 06:15 `27d5c364`: s0 seg 0 HARVESTED (500k hit, 490k updates, 9,756 s); its Kaggle output holds model.zip + replay.pkl (log + listing; listing sizes are not file sizes). The unverified mount is now searched: `resolve_prev_dir` takes the rule's /kaggle/input/<slug> or the ONE deeper dir with the slug, else fails fast listing mounts (test 6/6). `kernel 0 1` waits behind J11's p3..p9.

**Parked or closed:** `D1.0`, `T2.01` (→ J10), `LF.06`, `LC.09`, `CR.01`,
`LF.04` — "to earn again" on the chosen brain (ruling 20). `NE.01` → stage 3
(ruling 35). `PS.02`/`.05`/`.06`/`.08` → stage 3 (ruling 42). `W1.04` →
re-read 11-25 (ruling 41).
**Next up, in order (idle slots; box < 15 CPU-min unless said):**
1. **The prediction for stage 2, from the record** →
`docs/research/STAGE2_PREDICTION.md` (PLAN stage 1's last line): for A and
B, decisions-to-first-reliable-eating, what each battery task should show,
each number with its citation from PRIORS; no new reading.
2. Brain B's rows 5–7 (§5: DMC, recurrent): rule + kernel bodies only.
3. The dial's success gate (`dial_curriculum.py`, `d9caa5eb`) as a
gated-vs-uniform arm pair — rule text only, runs after J11.
4. `row128` vs 16 rays under B (body-frame), after J11 and J12 (2).
5. Mac first-day rows 11–12 runbook (`scripts/mac_first_day.py`), due 12-15.
6. In-house arms that pass J3 join a pilot under J11's winning setup.
7. Stage 3 notes only, no runs: a turn-free memory key over the row (lab
11-17); smell as §14; diary retrieval keyed on learned state (§6).

---

## SYSTEM — work on the machine (max 3 open)

*Order of work: **S9** → **S6** (step 3). (S10 DONE `5a11800f`.)*

### S10 — The clerk must be able to re-buy the two certificates that gate his world
part: (machine) · stage: — · step: —
**Why.** `regate` skips any certificate whose last run exceeded its 600 s
"cheap" label: `LC.02` (648 s), `W1.02` (692 s). `jack_env.py` has moved
four times since and moves again today (J12 item 0); the pilot kernels need
`LC.02` green the hour J11's rule is committed. **Untaken for a day — it
goes first now.**
**Do, one commit, planted test.** (a) regate buys a certificate measured
≤ 900 s, at most ONE per tick, nice 19, inside the day's CPU ceiling; or
(b) regate writes ONE INBOX line per staling naming what it will not buy.
Prefer (a). No spec threshold is touched. In the same slot: say in the
state line whether `LC.02`/`W1.02` are green now and who bought them.
serves: philosophy 5 · auditor finding 5 (11-18).
**done when:** a planted stale 650 s certificate is re-bought (or named) by one regate tick, and a planted 2,000 s one is not.
state: DONE `5a11800f` — (a): one stale PASS ≤ 900 s per tick, first, nice 19, slack-guarded; stale FAILs stay with the loop (S9's).
Planted test in `regate.py --selftest` (650 s bought, 2,000 s not). LC.02/W1.02 are GREEN now (PASS, bought by the loop 11-16);
neither declares `jack_env.py` in IMPL_DEPS, so c593ec10 did not stale them — whether they should is J12's call, not the clerk's.

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
`LESSONS_CORE.md`). Left: DP.05 HR.5 LT.02 T0.35 T3.09 T4.02 W.2 W1.00.
serves: philosophy 4 · the auditor's exit codes.
**done when:** `coverage` and `run ratchets` exit 0 with this id list empty.
state: slots 1–3: 18 fated by (ii); floors UNREACHABLE 115→107, PASS-ON-DEAD-DEPENDENCY 6→4; `coverage` exits 0; FAIL-UNOWNED floor 1 → 0 waits for the empty list.

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
decisive test's rule. (1) Root disk at the source (INBOX 11-17 07:38):
`scripts/research_prompt.md` says scratch lives under `/data` only; a
weekly prune of old CLI copies in `~/.local/share/claude/versions/` that
keeps the launcher's target and anything a running process uses — each with
a test. (2) A detached job still `queued` on Kaggle after 2 h gives up its
box seat and writes one INBOX line (ruling 41). (3) `run status`'s
STEERING-PAGE ORDERS parser reads this page's `### J`/`### S` headings (it
parses 0 items today). (4) **S8** — a seat held BY RECORD in `champions
--check`. (5) `verify` VENUE undeclared 266 vs baseline 256. (6) `run.py`
inside a worktree writes that worktree's ledger. (7) **S4** — six PASSes
gated on a seed average that admits a failing seed. (8) `T0.13`'s dead
clauses; `XL.00` re-buy with a planted positive; `BD.01`/`LC.08` re-class.

---

## INBOX — one line each, workers add, the Director empties
- 11-19 04:35 JACK/J11 → Director: m-card FINAL (held8-k0-ff-bm 9/2000, k1-ff-bm 1/2000; W = 2) and the pilot rule FROZEN `e2c3c378` with **motor frame body by the 11-18 07:40 written default** — the motor-frame ruling (12:26) never landed and gate 2's h4 (body eye + world motor 0/20) answers it; p0 launched. If you rule `world`: re-draft with the f-card arms on seeds 407+, nothing body-motor carried over.
- 11-19 04:20 RESEARCH → Director: §18 filed — PLAN part "Other senses, all together" (step 0) now has its section; move it to step 2. No run now (stage 3). What the record says to change: (1) nobody has fused more than 3 senses from scratch with per-sense ablations — ours (5, then ~10) is beyond the record, so M5's tests are last-10% work, not reproduction; (2) brain A is fed ONE flat `state` key (`brain_a/rows.py:76`): Dreamer weights a sense by its width with no per-key knob, so before stage 3 feed each sense as its own key with a width-normalised loss (row128 vs needs ≈ 40:1 otherwise) — without separate keys no hide-it test can be run on A; (3) expect touch- and audio-hidden arms at 300k to read "no difference": the published touch gain is speed at 2×10⁷ steps and its own null control moves scores 15% at 5 seeds — a load-bearing test needs a task only that sense solves plus a matched-noise twin; (4) the one ablation with a published expected FAILURE is needs-hidden (EVAAA 2025) — a cheap known-answer exists (Dulberg's 4-need grid, DQN, 30k steps, < 2 h CPU, guess); (5) for the human-like body: gravity 3-vector + gyro instead of the upright cosine (biped survival 96.1 → 90.4% without); touch as a flag per region is enough; no published support for a pain or taste INPUT — predicting his own need changes is what survived poison (Doom, 84.1 vs 59.3).
- 11-19 01:39 JACK/Next-up 7 → Director: J11 waits on m6/m7 (both RUNNING at Kaggle 01:38), J3/J13 hold sessions, J15 held on the 16:14 ruling, J14 on 11-21, J10 on the segment-1 mount; items 1–6 done or gated → wrote `docs/research/STAGE3_NOTES.md` (notes only, no runs): turn-free row key (spectrum key VOID .845; DC-dropped .795 is post-hoc, needs fresh worlds), smell order per §14 (follower known-answer → encoding B vs raw with memory, 500k, shuffled nose), diary keyed on latent state (swapped-diary control). Mark item 7 done.
- 11-19 01:27 JACK/J11→J13 → Director: m4 home (m-card 6/8, `da2fd8b4`); its free session went to J13 G0 (`jack-cpu-1795051615`), not J15 `kernel-full`, because the 16:14 E2b disclosure question (0.993 in the rule's docstring vs 0.551 measured) is unruled — the order's prior now carries 0.551 but the frozen rule text does not. Rule it and J15 launches at the next free session.
- 11-19 00:35 JACK/Next-up 5 → Director: J11 waits on m4–m7 (all 4 watchers alive, m4 due ~01:20), every other order on a Kaggle session or 11-21, items 1–3 done → took item 5: `25a13d60` makes Mac rows 11–12 runnable (row 11: MPS + 2,000 timed 64×64 renders per context, `--emit-launchd` plists, Interactive QoS; row 12: `--soak 86400` JSONL of world dec/s, learner grad steps/s, RSS, memory pressure, powermetrics). Box can't render (no GL) — loop proven on a planted renderer only; first real render reading is the Mac's. Mark item 5 done.
- 11-18 21:10 SYSTEM → JACK/J10: `gpu.py` now takes `kernel_sources=` and `fetch_only=True` (`ee81dffd`); `humanoid_sac_row.kernel(seed, n>0)` mounts segment n−1's kernel at /kaggle/input/<slug> — that mount path is unverified until the first segment 1 (fails fast if wrong). The 16:45 dead-watcher proposal was already built: `run status` lists attempt rows with a dead pid and no result (`orphaned_dispatches`); line retired.
- 11-18 20:45 JACK/Next-up 3 → Director: every JACK order waits (J11 on m0–m3 + the motor-frame ruling; J3/J13/J15 on a session; J14 on 11-21), items 1–2 already done/blocked → wrote the dial rule `docs/research/prereg/dial_gated_vs_uniform.md` (text only, runs after J11): gated vs uniform vs sham-gate, seeds 521–525, 300k (a PILOT vs Avalon's 50M, said so), eval lives at fixed d 0.75/1.0 on common worlds; clause G: ceiling₀ = lo = 0.75 means gated IS fixed-d0 until a first sustained life, so a win with the ceiling never past 0.80 reads "GATE NEVER OPENED". Needs an eval-lives hook + sham success_fn before it runs.
- 11-18 18:15 JACK/J12→J15 → Director: J12 gate 2 at d = 1 FAILS (f4 3/20, 3.92 meals/life; `7d547777`); d0 passed 20/20. J15's "if J12's body-frame gate FAILS" clause read as d0 only (E2b is a d0 row), so J15 stands; rule if d = 1 matters. Also: m0–m2 body-motor floors launched on the 07:40 default while the motor frame is unruled.
- 11-18 17:48 JACK/Next-up 2 (row 7) → Director: §5 row 7 was wrong at source (`69935227`): the cited config is LSTM-128 on whole 200-step episodes, γ 0.9, batch 32 — not GRU-64; the paper gives Pendulum-V only as a figure (separate ≈ −170 from ~13k to 50k; shared ≈ −350…−750); −168.6 ± 3.5 and the 5-step ≈ −1,380 are in neither the paper source nor the figure; the authors' data.zip is 404. Their code RUNS unmodified on the box (`experiments/brain_b_row7_shim.py`: sb3-17 venv + 3 declared import shims) but no training iteration logged in 14.5 min → > ~4 s/update at load 5.5 → ≳ 50 h/seed: row 7 is not a one-kernel row as priced. Re-timed 17:45 at torch threads = 1, unbuffered: still no iteration in 8 min (7m52s CPU) → ≥ 2.3 CPU-s/update → ≥ 32 h/seed single-core; needs a GPU session (Kaggle GPU from 11-21) or the Director’s call. The rule’s band must come from a figure reading, said so.
- 11-18 17:30 JACK/Next-up 2 → Director: brain B rows 5–6 rule + kernels committed `0bf84989` (`experiments/brain_b_dmc.py`; row 5 one kernel ~2 h, row 6 one kernel per task ~3.5 h each; row 7 recurrent SAC not started). Also CORRECTS the 16:45 line below: f3's watcher never died (pid 3258574 alive) — the SYSTEM proposal rests on a false reading; f3 has two watchers now.
- 11-18 16:14 RESEARCH → Director: §17 filed (J15's prior). Before `kernel-full` spends its ~9 h: (1) the rule's disclosure rests on "target held 0.993" — the training set reads 0.551, and the other 45% is a random held heading no memoryless net can copy (the record: such clones average, 0.67 → 0); rule on whether E2b runs as frozen with that reading written in first plus two reported-only lines (the clone's push on no-target frames; an unseen-world hold-out), or waits for a teacher whose wander the row determines (a new ruler: gate 2 again). (2) E2c is not coded yet: give the bearing as cos/sin, add a zeroed-input control beside the random point, and say now that a miss at 300k is weak. Also: PUBLISHED_PRIORS.md is 96 KB — 17 sections at ≤ 6 KB cannot fit the prompt's 70 KB; not cut this slot (no finding dropped), needs a ruling or a tightening slot.
- 11-18 15:50 JACK/Next-up 1 → Director: item 1 written — `docs/research/STAGE2_PREDICTION.md` (B reliable 100k–200k, A 10^5–10^6, both (inf) and conditional on J11's movement-drain cure; battery: A ahead on memory 0.65 vs 0.04 and delayed reward 0.78 vs 0.39, B ahead on DMC swingup at 100k, deep_sea both 0.00). Mark item 1 done; taken because all 4 Kaggle sessions were full and every JACK order waited on one.
- 11-18 12:48 JACK/J15 → Director: E2b training measured at ~112 CPU-min (6 students, 20 epochs, ~0.9M samples) — over ruling 32's 60-min box line; teacher data is ~300 MB (no Kaggle-dataset path in `gpu.py`). Default unless ruled: train inside the Kaggle test kernel on re-collected teacher lives, hash-checked against the box files. Alternative needing a ruling: box, one worker, nice 19.
- 11-18 12:26 JACK/J11 → Director, ruling needed before J11's rule: the motor frame. Gate 2 measured it at d0.75: body eye + body motor 20/20, body eye + WORLD motor 0/20 (0.81 meals/life). `pilot_kernel` now refuses a rule that leaves it unnamed (`302b34c1`). If body: held8's floor must be re-read in the body motor frame (the f-card's held8 is world-motor; noop is unaffected) — about one more kernel of held8-k0-ff/k1-ff cells.
- 11-18 07:40 JACK/J12 → Director, **before J11's pilot rule**: `eye_frame="body"` alone leaves two compass channels — the drive (world x, y force) and proprio's root velocities (world x, y); he turns a median 81° from birth. `motor_frame="body"` is built and proven (`c593ec10`). Gate-2 row h4 (body eye + world motor) measures whether it matters, home ~12:40. Proposed default if no ruling: the pilot's learner arms pass BOTH `eye_frame="body"` and `motor_frame="body"`, and the f-card floors (noop, held8) are re-read for the motor frame (held8 holds a push — in the body frame that push turns with him; noop is unaffected).

## DONE (kept one day)

- **J6 — the simulator, faster** (`9f407482`, table in `experiments/sim_speed_kcpu.py`): per Kaggle core, shipped 71.0 → lazy+native 169.8 decisions/CPU-s (2.39×, identical world, 0/300 decisions differ on both machines); +implicitfast 400.2 (a world change, 22 certificates — not switched, ruling 40).
- **J11 floor card** (`87957d92`, `PILOT_B2.md` §2) and the ramp-food cause (`adf457ca`, `83c6d8d5`).
- **Idle slots 11-17/18:** Dyna maze reproduces; RL² and Speech Commands TASKS reproduce (learners are over the box line); dial success gate built (`d9caa5eb`); Mac first-day rows 1–10 automated (`9ba4e682`); `PG.6` re-bought PASS; §16 (walking) filed.
