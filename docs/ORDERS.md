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
state: DONE — options `move_cost_dial`, `spawn_clear`, `food_floor_only` (`4978e989`, `adf457ca`); c-card 16/16 (leaky world, kept as history, NOT the rule's input); arms, packer, table tool coded and tested (`1eb7af3f`, `2d6808f0`, `1970d4a8`); f4..f7 groups (`31f5e5ac`).
state: 11-18 15:23 — f0, f1, f2, f4, f5 HOME. Card: noop-k1-ff 0/1500 (upper 0.20%); held8-k0-ff 31/1485 = 2.09% (upper 2.82%; one cell cut); held8-k1-ff 3/1000 = 0.30% (upper 0.775%, 8 worlds). f7 LAUNCHED `jack-cpu-1795015395` into f5's session. f3 (since 09:23), f6, f7 RUNNING — all 4 Kaggle sessions full (with J12 d=1). next: when f3/f6/f7 home, `card` + `rule-inputs`; the rule still waits on the motor-frame ruling.
state: for the rule writer: seed 401 was stepped once (start at 402); K-lr0 has no card twin (SBX `use_sde` holds a draw 32 decisions).
state: 13:45 `ca3f560a`: body-motor floor twins built ahead of the ruling — `held8-k0-ff-bm`, `held8-k1-ff-bm`, groups m0..m7 (4 cells = 2 worlds × both arms; `kernel --name mN`); `test_pilot_b2_bm` PASS 4/4. Full twin = 8 kernels; if ruled world, never run. 13:37: nothing new home (f3/f4/f5 RUNNING), collect 27/40 holds the detached cap → f6 still blocked.
state: 11-18 12:26 `302b34c1`: the pilot code never passed the eye frame (a rule to the letter would have run the compass eye). Now `PILOT_RULE = {seeds, food_floor_only, eye_frame, motor_frame, rule}`; `pilot_kernel` REFUSES without eye_frame='body' or a named motor_frame; frames reach every child's env, floors included (`test_pilot_b2_frames` PASS). The rule must name the motor frame (INBOX 07:40; gate 2's h4 says body eye + world motor 0/20). If body: the f-card's held8 ran world-motor and is not that floor. f3/f4/f5 RUNNING (f3 ~2 h to go at f2's 5.3 h); f6, f7 wait for sessions.

### J12 — His eye: body-frame, 128-ray row, and the proof his senses can find food
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
state: (2) d0.75 HOME `67c51727`: f4 PASSES 20/20 (7.53 meals/life, min 5.6), g4 0/20 (2.17), b4 0.45, h4 0/20 (0.81) → WORLD_CARD "J12 (2′) — gate 2's reading at d = 0.75". 14:38: d = 1 kernel LAUNCHED `jack-cpu-1795012722` (log `launch2_d100.log`, est ~5 h like d0.75). next: when home, `eye_forager score2` then the d = 1 row in WORLD_CARD.
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
state: next: `dreamer-xs` kernel when rank 3 gets a session.

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
prior: the hand forager on the 128-ray row eats 8.8 meals a life at d = 1 (compass eye). Behaviour cloning of a scripted policy into a small MLP is a known-answer task; a learner handed the bearing to food is the record's standard "is it the eye or the learner" control.
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
state: rules + collector DONE `760ae70a` (`experiments/forager_clone.py` docstring: E2b train seeds 701–740, test 741–760, clone ≥2 of 3 seeds, permuted twin must fail; E2c 300k × 5 seeds 801–805, random-point control). `test_forager_clone` PASS.
state: disclosed before data: the teacher holds a remembered target 0.993 of decisions (smoke seed 153) — a memoryless clone may miss for the forager's memory, not the eye.
state: 11-18 12:10 gate 2 passed at d0 → collect RUNNING on the box, 2 detached workers (`/data/jack-logs/forager_clone/collect_{a,b}.log`, seeds 701–720 / 721–740; ~3 h at ~39 dec/s). Student coded: `train_clone` (MSE, 256-256-256-8 ReLU/tanh, perm arm re-permuted every sight), `ClonePolicy`, `test_cell` via gate 2's own loop (`run_seed(policy=)`), `score`; `test_forager_clone_student` PASS 4/4.
state: 12:48 `65c648e7`: test kernel CODED (`forager_clone kernel`: 140 cells, refuses missing/uncommitted weights; `score [test.json]`); test PASS 5/5. Collect 8/40 (~24k decisions/seed, ~7.5 MB each). MEASURED: training costs 4.4 CPU-s/1e5 samples (clone), 8.5 (perm) → ~17M samples/student → 3 clones 38 CPU-min + 3 perms 74 = **112 CPU-min: over ruling 32's box line**; 300 MB teacher data cannot ride git (INBOX).
state: 13:25 FULL kernel CODED (`forager_clone kernel-full`): re-collects 701–740 in-kernel, checks each seed's (x,a) sha256 against `experiments/forager_clone_teacher.json` (mismatch reported, not fatal — declared in the docstring), trains the 6 students (train_clone unchanged), runs the 140 cells; fetches clone_full.json + 6 weights. Refuses without a committed 40-seed manifest. Test PASS 6/6. Est 9 h (GUESS) vs 9.5 h timeout.
state: 11-18 15:25 `ee9bc9de`: teacher manifest COMMITTED (40 seeds 701–740, 931,745 decisions, 17,280–27,000/seed); `full_ready()` = [] — kernel-full is launchable. next: at J15's rank when a Kaggle session frees (all 4 full at 15:23): `scripts/launch_detached.sh /data/jack-logs/forager_clone/launch_full.log $PY -m experiments.forager_clone kernel-full`.

### J13 — MLX port, stage 0 ends at gate G0 (owner 11-15)
part: His brain (A on the Mac) · stage: 1 (ready-list "first-day speed test for the Mac") · step: 2 · prior: `docs/research/PUBLISHED_EVIDENCE_2026-11-15_PORT.md` steps 1–3
**Do.** Run `experiments/brain_a/g0_kernel.py` on one Kaggle CPU session
when rank 4 gets one; the rule is the report's G0 unchanged. A failed G0
stops the port and is written up; a pass makes rung 1 the next order. No
other port work until G0 has run. Lead to weigh in the write-up: official
JAX ran 0.925 vs sheeprl 0.458 grad steps/s on this box (`9ba4e682`).
serves: M3 · PLAN "the decisive test … on the Mac mini" · philosophy 5, 6.
**done when:** G0's pass or fail is in `experiments/brain_a/README.md` with its numbers.
state: G0 kernel READY; never run — three days waiting at rank 4. If it has not run by Fri 11-20 it takes rank 1 for one session.

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
state: (1) DONE: Humanoid-v5 floor reproduces (random 109.9, zero 198.5). Yu pinned `b50478f8`, rule `experiments/yu2018_walk.py`; smoke never run.
state: (2′) RULE FROZEN `experiments/humanoid_sac_row.py` (11-18 slot): Zoo entry (2M, learning_starts 10k, SB3 defaults set explicitly in SBX), seeds 0–2, PASS R ≥ 5,950 AND ≥ 12 m/1,000 steps on ≥ 2/3, 1M median in [3,784, 5,866] else named miss, late-start control < 500 else VOID; resume = 500k/10.5 h segments with model + replay buffer (~3.1 GB) as kernel source.
state: `61e2950f`: Zoo entry VERIFIED at zoo `d802099a` (matches); SBX hidden defaults = SB3 non-SDE; rule CORRECTED before seed 0: metres = centre-of-mass x (v4 `x_position`), not torso. `evaluate` + `control_run` coded; box control 50k: R 67.96, 0 updates, holds. next: the training-segment body (500k/10.5 h, model + replay buffer carried, resume test on box at ~20k), then launch at rank 5.
state: 11-18 10:29 `23e375be`: `train_segment` (resume as the rule says) + box `resume_check` 10k+10k: step counter, buffer, n_updates (9000→19000), all params + actor optimiser equal after load, fresh control differs — RESUME OK. Box ~0.018 wall-s/step. Found: replay is float64, ~6.1 GB at 1e6 (rule said 3.1; fits Kaggle 20 GB). Wall-stop callback not exercised. next: kernel wrapper (seg 0 fresh; later segs attach previous output as kernel source), launch seed 0 at rank 5.
state: 11-18 10:50: Kaggle wrapper `humanoid_sac_row kernel <seed> [segment]` + job text (pins = jack-std venv; fetches only segment.json); `tests/test_humanoid_sac_row_kernel` PASS 4/4. BLOCKED on the machine (INBOX 10:50): gpu.py has no `kernel_sources` (segment ≥1 cannot mount the checkpoint; wrapper refuses) and downloads ALL output (seg 0 would pull ~6.1 GB to /data). next: when both land, launch seed 0 seg 0 detached at rank 5.

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
- 11-18 15:50 JACK/Next-up 1 → Director: item 1 written — `docs/research/STAGE2_PREDICTION.md` (B reliable 100k–200k, A 10^5–10^6, both (inf) and conditional on J11's movement-drain cure; battery: A ahead on memory 0.65 vs 0.04 and delayed reward 0.78 vs 0.39, B ahead on DMC swingup at 100k, deep_sea both 0.00). Mark item 1 done; taken because all 4 Kaggle sessions were full and every JACK order waited on one.
- 11-18 12:48 JACK/J15 → Director: E2b training measured at ~112 CPU-min (6 students, 20 epochs, ~0.9M samples) — over ruling 32's 60-min box line; teacher data is ~300 MB (no Kaggle-dataset path in `gpu.py`). Default unless ruled: train inside the Kaggle test kernel on re-collected teacher lives, hash-checked against the box files. Alternative needing a ruling: box, one worker, nice 19.
- 11-18 12:26 JACK/J11 → Director, ruling needed before J11's rule: the motor frame. Gate 2 measured it at d0.75: body eye + body motor 20/20, body eye + WORLD motor 0/20 (0.81 meals/life). `pilot_kernel` now refuses a rule that leaves it unnamed (`302b34c1`). If body: held8's floor must be re-read in the body motor frame (the f-card's held8 is world-motor; noop is unaffected) — about one more kernel of held8-k0-ff/k1-ff cells.
- 11-18 10:50 JACK/J10 → SYSTEM: `experiments/gpu.py` `run_on_kaggle` needs (a) an optional `kernel_sources=["user/slug"]` written into kernel-metadata.json (J10's 2M SAC row resumes from the previous kernel's model+replay output; mounted under /kaggle/input/<slug>), and (b) `kernels output --file-pattern` built from `fetch`, else every segment downloads its ~6.1 GB checkpoint onto /data. `humanoid_sac_row.kernel` refuses segment ≥ 1 until (a).
- 11-18 07:40 JACK/J12 → Director, **before J11's pilot rule**: `eye_frame="body"` alone leaves two compass channels — the drive (world x, y force) and proprio's root velocities (world x, y); he turns a median 81° from birth. `motor_frame="body"` is built and proven (`c593ec10`). Gate-2 row h4 (body eye + world motor) measures whether it matters, home ~12:40. Proposed default if no ruling: the pilot's learner arms pass BOTH `eye_frame="body"` and `motor_frame="body"`, and the f-card floors (noop, held8) are re-read for the motor frame (held8 holds a push — in the body frame that push turns with him; noop is unaffected).

## DONE (kept one day)

- **J6 — the simulator, faster** (`9f407482`, table in `experiments/sim_speed_kcpu.py`): per Kaggle core, shipped 71.0 → lazy+native 169.8 decisions/CPU-s (2.39×, identical world, 0/300 decisions differ on both machines); +implicitfast 400.2 (a world change, 22 certificates — not switched, ruling 40).
- **J11 floor card** (`87957d92`, `PILOT_B2.md` §2) and the ramp-food cause (`adf457ca`, `83c6d8d5`).
- **Idle slots 11-17/18:** Dyna maze reproduces; RL² and Speech Commands TASKS reproduce (learners are over the box line); dial success gate built (`d9caa5eb`); Mac first-day rows 1–10 automated (`9ba4e682`); `PG.6` re-bought PASS; §16 (walking) filed.
