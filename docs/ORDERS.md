# ORDERS — what the workers do. Written by the Director.
Director sitting: 2026-11-17 06:50 UTC

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5). **One state+journal commit per slot,
not one per step.** Machine problems go under INBOX, one line each. Rulings
30–37 cited here are in `docs/DECISIONS_RESOLVED.md` under 2026-11-17 06:50.

**`docs/PLAN.md` binds every order (owner, 11-15).** Research → combine →
reproduce → test the last 10% → put it in him. We are in **stage 1: the
instrument.** `docs/research/PUBLISHED_PRIORS.md` §1–§14 is the reference —
**read your order's section at the start of every slot and follow its exact
versions and numbers where it is sharper than this page.**
**Where stage 1 stands (11-17).** Brain B lived 2,350 lives in his world
(J8, 35 + 10 runs) and the reading is **VOID with a negative inside it: 0
sustained lives where doing nothing predicts 7.8; learning made him eat
less** (he learned to lie still — the record's expected result where moving
costs energy, §13). The VOID itself came from two rules that disagreed (a
body born touching food is fed by contact). **Nothing of J8 is re-run.** The
next pilot (J11) is a new rule on fresh seeds, written from the record and
from a learner-free floor measurement — never from B's 35 runs (ruling 30).
The lab found his 16-ray eye finds floor food barely half the time and a
128-ray row nine times in ten (J12). Entry rung stays **d0 = 0.75** (ruling
22); no rung below 0.5. PLAN's ready-list ("before the Mac arrives") is the
deadline.
**Stopped (rulings 10, 20, 23):** races between home-made learners outside
J3(4); forager cells on the OLD eye; world-rule races; `D1.0`, `T2.01` runs;
new specs with no published prior; any smell run (§14 is filed for stage 3,
ruling 36). A question with no prior is a `RESEARCH:` line in INBOX.
**Kaggle CPU is the scarce thing: 4 sessions, RANKED (ruling 31).** A slot
that finds a free session gives it to the first READY kernel in this order:
(0) any kernel estimated < 1 h (J6's speed column; J12's gate if it is one);
(1) J11; (2) J12; (3) J3(4) — one session; (4) MLX G0 (J13); (5) J10's
smoke; (6) `W1.04`'s remaining seeds. `W1.04` (in flight from J7, now
closed) **yields between seeds** whenever anything ranked above it waits; its
row is harvested by whichever slot sees it exit (`run status | grep W1.04`).
Reported-only arms never hold a session something ranked above waits for.
**The box (ruling 32).** Day ceiling 57,600 CPU-s, ≤ 2 computing detached
jobs. The unit is the WHOLE RUN, not the cell: < 15 CPU-min → box, nice 19.
15–60 CPU-min → box only when all 4 Kaggle sessions are held, ONE worker,
nice 19, said in the state line. > 60 CPU-min → Kaggle or it waits. Kaggle
GPU has **0 h** until Sat 11-21 00:00 UTC; Colab T4 through `gpu.py`.
**`/` has 4.1 GB free and workers refuse under 3:** nothing new under
`/home` or `/tmp`. Every install on the box sets `UV_CACHE_DIR`,
`PIP_CACHE_DIR`, `XDG_CACHE_HOME` under `/data/caches/` and uses
`--only-binary=:all:`.
**World files** (`w0.py`, `needs.py`, `survival.py`, `drives.py`): changed
only through `jack_env.py` options whose default is today's world, proven by
`test_jack_env.py`.
**Known-answer rows (ruling 33):** a row's rule states its seeds AND at most
one extension (size, fresh seeds) BEFORE the first seed runs. A miss after
that is a miss. No second look that the first rule did not name.
**A smoke for a multi-life row runs the registered life count on one seed.**
**Receipts:** the slot that launches a kernel commits
`experiments/gpu_submissions.jsonl` and `cpu_budget.json` by name in its
state commit — **both sit dirty right now; the next slot commits them.**

---

## JACK — build the creature, his world and his curriculum (max 7 open, ranked)

*Order of work: **J11** → **J12** → **J6** → **J3** → **J14** → J13 → J10.
A slot waiting on a kernel advances the next order. **The 07:07 and 19:07
strong-worker slots take J11 first, then J14.** A slot that can advance none
takes "Next up" item 1.*

### J11 — Pilot 2: can a plain learner eat at the entry rung once the record's cures are in? (new rule, fresh seeds)
part: Learning to eat + His world · stage: 1 (ready-list item 4) · step: 2 (combine the cures as options) → 4 (pilot) · prior: §13 (read it whole), §1, §8; ruling 30
prior: the record predicts the lie-still slide wherever moving costs energy (Kwiatkowski 2023: optimum at γ ≈ 0.99 is v = 0; Hwangbo 2019; Hollenstein 2022 with SAC) and names the cure proven on it: ramp the cost in (3 locomotion papers, PPO, 10^7–10^8 steps). An alive bonus and white noise failed there; white, pink and gSDE all failed here (J8 item 6).
prior: every published eater (ant, EVAAA, Crafter, Avalon, ForageWorld) lived where moving drains nothing extra; they ate after 100k–1M decisions. Ours beside theirs: movement drain (theirs none; ours 2–3× basal), +100 on eating (the ant pays none), 300k decisions.
prior: B reproduced its two SAC rows (step 3 done). No plain learner has yet shown the task learnable at any rung — this pilot is still that question. Predicted by the record: arm K learns to eat inside 300k; arm P stops the slide but meals stay near random; arm G unknown.
**Do.** (1) Options in `jack_env`, default = today's world, each with an
identity test (off = byte-identical) and a planted test (on = the effect):
`move_cost_dial=True` — `drives.KAPPA` scaled by the dial so movement costs
nothing extra at d0 and the full KAPPA at d = 1 (linear in (d − d0)/(1 − d0));
`spawn_clear=R` — birth and rebirth never within R of a food, R = the
contact-eating reach (cause a: a statue fed by contact — a world rule, and
it is repaired FIRST because it is learner-free); reward `dr100-eat-pos` and
discount 0.999 if §13 needs code for them. (2) **Floor card, learner-free,
box or one short kernel, BEFORE any rule:** do-nothing and held-8 random,
≥ 2,000 lives each, at d0 with `spawn_clear` on, with and without
`move_cost_dial`: crossings of the sustained mark at W = 1, 2, 3, life
lengths, meals per life. (3) RULE committed before any learner runs, seeds
**401+** (never 301–345): W = the smallest W at which the floor card
predicts ≤ 0.5 floor crossings over the pilot's planned floor lives; floor
clause = crossings ≤ the Poisson 95% upper bound of the card's rate (not
"≤ 1 in total"); T = 300,000; arms, 5 seeds each, all with `spawn_clear`
and `lazy_contact_audio`: **B0** J8's setup unchanged; **K** `move_cost_dial`;
**P** `dr100-eat-pos`; **G** discount 0.999; **KP** K + P; floors (do-nothing,
held-8, K with learning rate 0) 5 each. Primary = sustained lives per arm
beside the floor; also the J8 early/late table (meals per life first 5 vs
last quarter; unfed life length). Reported-only: it picks the SETUP the
decisive test uses, no seat moves. (4) Before any box run of B fix or
explain INBOX 11-16 19:40 (`_single_thread` ran 2.4 cores). (5) Kaggle CPU,
4 runs a kernel, `pilot_b_report`-style table → `docs/research/PILOT_B2.md`.
serves: M2→M3 · PLAN ready-list "B piloted in his world" · philosophy 3, 4 (a rule that could fail, written before the data).
**done when:** `PILOT_B2.md` has the floor card, the frozen rule, 40 runs and one reading per arm: sustained / eats but starves / slides.
state: (1) DONE 11-17 07:29 `4978e989`: `move_cost_dial`, `spawn_clear` in jack_env, default off; `tests/test_jack_env_cures.py` PASS (identity 3/3 seeds; at d0 extra drain 3.7e-17 vs 5.2–5.9e-4/decision, control fails; 225 cleared spawns ≥ 1.58 m, control 75/75 inside; do-nothing at d=0: 0 meals vs 2,545). `dr100-eat-pos` exists; discount is a learner setting (no env code). R = `SPAWN_CLEAR_R` 1.55 m, measured learner-free (180 do-nothing lives, seeds 381–382): the body tips and slides ≤ 1.07 m; 5 contact meals, all from spawns ≤ 0.88 m, 0/123 from ≥ 0.98 m.
state: (2) FLOOR CARD fixed in `experiments/pilot_b2.py` docstring (8 kernels c0..c7). c0..c3 Kaggle RUNNING at 13:10 (5.7/5.0 h); watchers time out ~17:00/17:40, Kaggle's own cap ~19:29/20:08. c4..c7 carry `POOL_DEADLINE_S` 8.5 h. Box noop-k1 smoke 11-17 13:10: lives 2,576/2,860 decisions (estimate held) — the overrun is Kaggle's 4-worker throughput (~2× one process, J6 vec4), not life length.
state: RECOVERY 11-17 13:20: a c0..c3 kernel whose watcher times out is NOT lost — `cells_from_artifacts` assembles the card from its per-cell files (written every 10 lives); `test_pilot_b2_recover` PASS. Harvest a timed-out one once Kaggle shows it terminal: `JACK_REUSE_KERNEL=jannolouwrens/jack-cpu-<id> $PY -m experiments.pilot_b2 kernel --name cN` (ids: c0 1794900564, c1 1794900599, c2 1794902904, c3 1794902959 — check against launch_cN.log).
state: next: as each frees a session launch the next of c4..c7 (`scripts/launch_detached.sh /data/jack-logs/pilot_b2/launch_cN.log $PY -m experiments.pilot_b2 kernel --name cN`). All 8 in → `pilot_b2 card` (cut cells shown) → `PILOT_B2.md`; ONLY THEN the rule (3).
state: (4) DONE 11-17 13:40 `3b054813`: the extra cores are jax 0.11's `tf_XLAEigen` pool (4 threads, 46% of CPU), which ignores XLA_FLAGS even set from start; it follows the affinity mask. Box runs of B use `JACK_PIN_CPU=<cpu>` (or `_single_thread(pin_cpu=)`): 1.00 cores, 9% less CPU, 1.46× wall, learned weights + lives bit-identical. Kaggle's 4-runs-a-kernel pools oversubscribe the same way (4×4 threads) — unmeasured there. c0..c3 still RUNNING 13:22.

### J12 — His eye, step 2: the 128-ray row as an option, and the proof his senses can find food
part: Eye + Learning to eat (the learnability proof) · stage: 1 (ready-list "the eye designed"; stage-1 line "a hand-written forager that uses only what he senses") · step: 2 (combine) · prior: §12; LAB `experiments/eye_streak.py` (`bddffac4`); ruling 37
prior: §12 — a first eye of distance-and-material rays, 0.1–0.3 ms a look. Ours, measured 11-16: at d0 a 16×16 grid finds floor food no better than today's 16 rays (.517 vs .528); ONE level row of 128 azimuths finds it .909 (row64 .856, row256 .925; blind .19, shuffled .21). The old eye failed the forager bar twice (6 and 13 of 20; ruling 23).
**Do.** (1) `jack_env(eye="row128")`: 128 azimuths × [distance, material] in
one `mj_multiRay` call (builds on `fast_sim.apply_multiray_vision`), default
OFF = today's 16 rays byte-identical (identity test), cost per look written
down. (2) Gate FROZEN before any run (ruling 37 lifts ruling 23 for the NEW
eye only): world-card row f2's forager, changed only in that it reads the
new eye; 20 fresh seeds; the unmoved bar — ≥ 5 floor meals per life on ≥ 18
of 20 seeds; twin with the rays shuffled must miss; run at d0 AND at d = 1,
both reported, the bar judged at each. One extension clause or none, stated
first. (3) Row in `WORLD_CARD.md`. Pass or fail it is an adult-taught
ceiling: it says the food is findable from the sense, not that a brain
learns to. Not yet: `row128` under brain B — that is an arm pair once J11
shows B moves at all.
serves: M3, M5 · PLAN parts "Eye" (1 → 2) and "Learning to eat" last-10% question "can 16 rays tell food from a wall?" · philosophy 3, 4.
**done when:** the option, its two tests and the gate's row (both rungs) are committed.
state: (1) DONE 11-17 07:58 `9e4031bb`: `jack_env(eye="row128")` (fast_sim.apply_row_vision, one mj_multiRay, vision 256 wide; default "rays16" untouched). `tests/test_jack_env_eye.py` PASS: identity 4/4, row ray 8j == old ray j on 4/4 seeds (control 8j+4 differs), any ray on floor food .287 → .905 at d 0.75 (400 decisions), 0.089 vs 0.058 CPU-ms a look (+0.15% of a decision).
state: scar: with hand cal j0=alpha=1 + random_needs the body dies every ~3 decisions and never settles to eye height — NO eye sees food; use the shipped cal for any eye measurement.
state: (2) GATE FROZEN 11-17 08:25 `4f819fee` (WORLD_CARD.md last section, `experiments/eye_forager.py`): f3 = f2 reading row128 (blob rule kept in angles), g3 shuffled twin, b3 random; seeds 241–260 × 10 lives; d0 and d=1; bar unmoved; NO extension. `test_eye_forager` PASS: 0/12,000 actions differ from HandForager2 at 16 rays, nudged control differs. Smoke d0 seed 153: f3 capped 19 meals, g3 capped 10 (g3 at d0 may pass → f3 VOID there; disclosed), b3 3.
state: next: (2) run = 2 Kaggle kernels, ~4.7 h worst each, rank 2 → launch only after J11's c4..c7 have sessions: `scripts/launch_detached.sh /data/jack-logs/eye_forager/launch_d075.log $PY -m experiments.eye_forager kernel 0.75` (and `1.0`); harvest `eye_forager score /data/jack-logs/eye_forager/gate_d*.json` → (3) the row in WORLD_CARD.md.

### J6 — Make the simulator fast, once, and say what it costs to re-certify
part: His world (the instrument's speed) · stage: 1 · step: 2 (combine) · prior: §4 "Take as-is"; INBOX measurement 11-15 15:55
**Why.** 38.8 decisions/s means 1M decisions ≈ 7 h. §4: the loop, not
physics, is the bottleneck; predicted 500–2,000 per kernel.
**Do, each a `jack_env` option, default OFF, measured alone and together
(decisions per CPU-second, box and one Kaggle CPU kernel, table in the
docstring):** (1) a 4-process vector (changes nothing in the world — do it
first); (2) all sub-steps of a decision in one native call, per-sub-step
Python work moved to once per decision where the result is identical; (3)
rays by `mj_multiRay`; (4) `integrator="implicitfast"`. For each, the same
3-seed identity test as J1: identical, or the measured difference (the
integrator IS a world change: integrity .879 → .972 in one life). Then list
which standing world certificates each non-identical option would stale
(`run stale` on a scratch worktree). The Director rules on the switch with
that list; do not flip any default.
serves: M2→M3 · PLAN stage 1 line 1, "honest limits" 1 · philosophy 5, 6.
**Today (11-17):** the Kaggle column died twice (`status=error`; read the kernel's own log with `/data/venvs/kaggle/bin/kaggle kernels output`, fix, relaunch — it is a < 1 h kernel and goes FIRST in the queue). Then item (2b), not yet done: the 40 sub-steps of a decision in one native call (RK4 `mj_step`×40 is 54–58% of a decision). **Ruling 34:** every new run passes `lazy_contact_audio=True` (identical world); the integrator is NOT switched for J11 — it is ruled once, when this table is complete, before the decisive test's rule is written.
**done when:** the speed table (box + Kaggle columns, 2b included or measured-impossible) and the re-certification list are committed.
state: (1) vec `7f3af75d` (1.59× wall); (2a) lazy audio `403def65` identical 1.24×; (3) multiray `11-16` identical, not a lever; (4) implicitfast = WORLD CHANGE 1.65× (re-cert list 22 certs in `tests/test_integrator.py` docstring). Kaggle column `dc0e2b05` (`sim_speed_kcpu.py` docstring).
state: (2b) DONE on the box 11-17 12:45 `e7c5011c`: `jack_env(native_substep=True)` (needs lazy audio; four C twins, `native_substep.py`). `tests/test_native_substep_env.py` PASS: 0/300 differing decisions × seeds 153-155 × {dying e0=0.05 (2 deaths/seed), J11 pilot + both cures (truncations), reset(seed) rebuild, implicitfast}; dt+1ulp control differs 273-287/300. Stales nothing (identical; default OFF).
state: box row (`sim_speed_kcpu.py` docstring, T=300): shipped 49.0, lazy 60.6, lazy+native 93.7 (1.91×), lazy+implicitfast 120.0, lazy+native+implicitfast 281.3 dec/CPU-s (5.74×). Finding: piece tests' fma_norm=False control changes no decision (too weak for whole-env).
state: next (last item): Kaggle column for native — KERNEL READY 11-17 13:00 `ff0b11b7` (`sim_speed_kcpu kernel` now runs `native_identity(300)` first: A+D × seeds 153-155, ≥1 death each, dt+1ulp control must differ; else box-only). Box pre-check under Kaggle pins (mujoco 3.17): C compiles, 0/300 × 6 pairs, control 85/300. Launch when a session frees (rank 0, ~1 h): `scripts/launch_detached.sh /data/jack-logs/j6/launch_native.log $PY -m experiments.sim_speed_kcpu kernel`; harvest `/data/jack-logs/j6/kcpu_speed.json` into the docstring → J6 DONE.

### J3 — Brain B from published code reproduces published numbers (§5's rows)
part: His brain · stage: 1 · step: 2→3 (combine, reproduce) · prior: §5
**Why.** No learner here has ever reproduced a published result.
**Do.** (1) A separate environment `/data/venvs/jack-std` (Python 3.12 is at
`~/.local/bin/python3.12`) with §5's pins exactly; the project venv is not
touched. (2) `experiments/brain_b.py`: SBX `SAC` with
`policy_kwargs=dict(layer_norm=True)` and §5's settings — NOT a rewrite.
Docstring = source, versions, "ours beside theirs". (3) §5's rows, rule
committed first, 5 seeds each, the metric exactly as §5 states it (**ruling
18:** these replace the old "−157, pass ≥ −200"): row 1 Pendulum SAC — the
unmodified library first, then ours — accept a 5-seed mean in −314.7 to
−282.1; row 3 sparse MountainCarContinuous, gSDE recipe in PyTorch SB3,
> 90 on ≥ 3 of 5 seeds, and any SBX or pink-noise variant passes the same
row before it is used; rows 2 and 4 (PPO reference). Table →
`docs/research/KNOWN_ANSWERS.md` (task, steps, published, ours, seeds). A
miss means the build is broken: fix it, do not go on. Rows 5–7 (DMC,
recurrent) are next week's. (4) **The owner's fair chance (11-15 19:31):
ALL seven in-house arms** (`ppo-needs`, `ppo-lp`, `dreamer-xs`, `wm-efe`,
`wm-latent`, `wm-latent+ring`, `wm-dyn`) plus the seated `cores.py` tanh-AWR,
each through one thin adapter, on rows 1 and 3. Rule committed first
(ruling 24): an arm PASSES a row if it reaches the row's published band
within 3× the published learner's steps on ≥ 3 of 5 seeds, with its own
never-trains control failing; one Kaggle CPU kernel per arm. A failing arm
is set aside with its numbers, not deleted, and may return repaired.
serves: M3 · PLAN stage 1 "each brain reproduces published numbers" · philosophy 3, 4.
**done when:** `KNOWN_ANSWERS.md` has row 4's closing line (ruling 33) and one row pair per in-house arm, pass or fail. **At most ONE Kaggle session at a time for these arms** (two only when nothing ranked above waits); never on the box (ruling 32).
state: (1)(2) DONE `1d12bb9a`. Rows 1b, 2b, 3, 3b REPRODUCE. Row 4 MISSES through 4d at 40/arm (ours 72.6 vs published code 73.4 on this box; HL CI [−9.1, +0.6]); **row 4 CLOSED by ruling 33** — write the closing line in KNOWN_ANSWERS, run nothing more on it.
state: (4) RULE `3054fbe6` `experiments/inhouse_rows.py`; KERNEL READY `40635f1b` (`inhouse_rows kernel ARM`, 20 cells, 4 workers, pins gymnasium==1.1.1).
state: 11-17 01:30 **ppo-needs (the seated tanh-AWR learner) FAILS both rows, controls hold** (box run, 20/20 cells): row 1 best windows −578…−1120 vs −314.7 (0/5; rising every window — too slow); row 3 0/5, and **0 flag reaches in 750 training episodes** — it learned to stop paying action cost (−23 → −1.1); i.i.d. Gaussian exploration never samples the reward (B passed only with gSDE). Set aside with numbers, KNOWN_ANSWERS §Row J3-4.
state: 11-17 08:50 `dc0e2b05` **wm-latent and ppo-lp FAIL both rows, controls 0/5** (KNOWN_ANSWERS §Row J3-4b): wm-latent row 1 best windows −321.0…−397.2 vs −314.7 (last 10 eps −231…−335: learns, the 3× budget binds); ppo-lp −824…−1059; row 3 0 flag reaches in 1,500 eps each. The row-3 rise −22 → −1.3 is in the never-trained controls too (noise schedule) — J3-4's "learned to stop paying" corrected.
state: next: dreamer-xs, then wm-efe, `inhouse_rows kernel ARM`, one Kaggle CPU session (rank 3, after J11 c4..c7 and J12's two kernels); harvest into KNOWN_ANSWERS. wm-dyn, wm-latent+ring need the adapter wired first.

### J14 — Brain A's known-answer rows: everything ready so the free GPU is spent from Sat 11-21 00:00 UTC
part: His brain · stage: 1 · step: 3 (reproduce) · prior: §2 (rows, bands, settings), §9 row 7; J5's speed (`ca4dcf00`: compiled 51.8 dec/s on a T4, 0.54 h per 100k)
prior: §2's first three known-answer rows at the early points of the published curves; estimate 7 T4-h at 3 seeds, 12 at 5. A has reproduced nothing yet; B's rows are done.
**Do, before Saturday (no GPU needed).** (1) `experiments/brain_a/rows.py`:
sheeprl 0.8.5 `dreamer_v3_5` smallest preset, UNMODIFIED, with
`algo.compile.enabled=True`, on §2's three tasks; a never-trains twin per
row. (2) The rule, committed first (ruling 33's form): seeds, band per row
exactly as §2 states it, the one extension. (3) The sheeprl-vs-official
line diff (`danijar/dreamerv3@e01491f`) into the README's "ours beside
theirs". (4) A 5-minute CPU smoke that the kernel body imports and steps.
(5) **On 11-21:** launch through `gpu.py` (`prefer="kaggle"`), push first
(GPU jobs clone from GitHub); rows → `KNOWN_ANSWERS.md`.
serves: M3 · PLAN ready-list "brain A reproduced at the early points of its published curves on the free GPU" · objective 3 (spend the free GPU) · philosophy 2, 3.
**done when:** (1)–(4) committed by Fri 11-20; then the three rows, in band or a named miss.
state: (2) RULE FROZEN 11-17 13:45 `286d6451` `experiments/brain_a/rows.py`: A1 cartpole_balance 100K/200K (690–898, 932–1000), A2 cartpole_swingup (438–518, 740–816), A3 ball_in_cup_catch 200K (931–967); seeds 0–2, one extension 3–4; never-trains twin must land below band else VOID. (3) DONE 11-16 (README line diff).
state: (1)(4) DONE 11-17 14:10 `31286dac`: kernel = `rows.py kernel A1|A2|A3` (one Kaggle GPU kernel a row: control first, then seeds 0–2, file rewritten per run, 9 h in-kernel cut; `kernel ROW --ext` = seeds 3–4; `score ROW` merges and judges). Per-episode returns from the runner's stdout line (rew_avg is an interval mean). `experiments/tests/test_brain_a_rows.py` PASS 5/5.
state: box smoke (`rows.py smoke`, A2 control, CPU, 15 s): unit check POSITIVE — first episode at policy_step=2000 on all 4 envs (env steps = 2 × policy_step). CONTROL CORRECTED before any seed: `learning_starts=total+1` still trained ~2 gradient steps on the last iteration (runner floors to iterations; replay_ratio 0.001); now `2×total` → replay_ratio 0.0, no loss logged. Stronger, no reading could have moved.
state: next: nothing left before Saturday. On 11-21 00:00 UTC, pushed HEAD: `scripts/launch_detached.sh /data/jack-logs/brain_a/rows/launch_A1.log $PY -m experiments.brain_a.rows kernel A1` (then A2, A3; ~2.6 T4-h each, est. in EST_HOURS). Harvest: `rows.py score A1` → KNOWN_ANSWERS. The kernel needs `sheeprl` only on Kaggle (pip in the body); `score` runs in the project venv.

### J13 — MLX port, stage 0 ends at gate G0 (owner 11-15 19:18)
part: His brain (A on the Mac) · stage: 1 (ready-list "first-day speed test for the Mac") · step: 2 (combine) · prior: `docs/research/PUBLISHED_EVIDENCE_2026-11-15_PORT.md` steps 1–3; `experiments/brain_a/README.md` §MLX port
**Why.** Nobody has published Dreamer on Apple hardware; a port that cannot
match the official code tensor for tensor must be known before the Mac comes.
**Do.** Steps 0.1–0.4e are DONE in idle slots (golden file frozen
`2c7c9516`, per-tensor gradient gate `port_gate.py`, selftest 8/8). What is
left: run `experiments/brain_a/g0_kernel.py` on one Kaggle CPU session when
the queue gives it one; the rule is the report's G0 unchanged. **A failed
G0 stops the port** and is written up; a pass makes rung 1 the next order.
No more idle-slot port work until G0 has run.
serves: M3 · PLAN "the decisive test … on the Mac mini" · philosophy 5, 6.
**done when:** G0's pass or fail is in the README with its numbers.
state: G0 kernel READY `experiments/brain_a/g0_kernel.py`; waits for a session (queue rank 4).

### J10 — The walking layer starts now, by reproducing a published walk (owner, 11-15 19:18)
part: Body and movement · stage: 1 side job (stage 3's part, independent of the brain) · step: 3 (reproduce) · prior: §4 "Reproduce first", "Take as-is"
**Why.** The owner asked whether walking is in the right place in the plan.
The record: walking is a known recipe and needs no brain choice; `T2.01` is
re-earned this way, never by more one-world attempts.
**Do, one per slot, Kaggle CPU only, rule first, table row in
`KNOWN_ANSWERS.md`.** (1) §4's Gymnasium Humanoid floor numbers (random and
standing-still returns; seconds). (2) §4's 2018 CPU recipe — helping forces
that are withdrawn — with the UNMODIFIED published learner and settings on
the published task, one kernel, accept inside §4's band, never-trains
control must fail. (3) Not before (2) reproduces: the same recipe on our
humanoid. The MJX tutorial humanoid waits for the Kaggle T4 on 11-21.
Design note for stage 3 (owner's correction 19:21): inside him the layer
KEEPS LEARNING (§4 arm c); the frozen layer (arm b) is a scored reference only.
serves: M4 · PLAN stage 3 "a separate, fast walking layer is built and checked in parallel from now" · philosophy 1, 2.
**done when:** rows (1) and (2) are in `KNOWN_ANSWERS.md`, pass or fail.
state: (1) DONE 11-16 17:40 (rule `c4dfd21c`, `experiments/humanoid_floor.py`): Humanoid-v5 floor, 100 eps each — random 109.9 (SD 30.3) in 23.47 steps, zero 198.5 (SD 3.2) in 40.3 steps → REPRODUCES §4's line (which is an earlier on-box measurement, not a paper; its ± is the per-episode SD). Ran on the box (3 CPU-s) because all 4 Kaggle CPU sessions held J8. Row J10-1 in KNOWN_ANSWERS.
state: (2) PINNED 11-16 17:55: `VincentYu68/SymmetryCurriculumLocomotion@b50478f8`; DART 6.3.0 + pydart2 from source, TF1 (py3.7), MPI (KNOWN_ANSWERS §Row J10-2).
state: (2) RULE COMMITTED 11-16 18:30 `experiments/yu2018_walk.py`. CORRECTION: the script's 5 M is a PER-STAGE cap (~22 force stages, each ends early on success) → total ≈ paper's 10–30 M, so it does NOT fit one kernel; full run = chained kernels resumed from the authors' 10-iter saves. 8 MPI ranks oversubscribed on 4 cores (keeps the 20k batch; `-np 4` dropped). Accept ≥ 9/10 unaided 500-step rollouts at ≥ 0.75 m/s; iteration-0 control must fail; seeds 0-2.
state: next: when the queue (header) gives it a session, `launch_detached.sh /data/jack-logs/j10/launch_smoke.log $PY -m experiments.yu2018_walk smoke` (~1.5 h: build stages + 20-min slice → s/iteration); then register the full run (kernels = ⌈1,500 × s/it ÷ 11 h⌉, resume wrapper).

**Paused or closed by the record (ruling 20):** `D1.0`, `T2.01` (re-earned
through J10), `LF.06`, `LC.09`, `CR.01`, `LF.04` — "to earn again" on the
brain stage 2 chooses. `NE.01` parked to stage 3 (ruling 35).
**Next up, in order:** 1. **Idle-slot reproductions** (§3, §6, §7; one per
slot, box < 15 CPU-min): the 8×8 shaping grid, Dyna maze, RL² bandits,
Speech Commands keyword spotting (corpus at
`/data/jack_corpora/speech_commands_v2`), each with ruling 33's rule first.
2. **Mac first-day script** — `scripts/mac_first_day.py` (`61cb3bda`) covers
§10's rows 1–8 and 11; automate 9–10 once J14's pins are known; due 12-15.
3. In-house arms that pass J3(4) join a pilot under J11's winning setup,
same seeds and rule. 4. The clone (E2b) and the cheat learner (E2c) at d0,
after J12's gate. 5. `row128` vs 16 rays as an arm pair under B (after J11
and J12). 6. `PG.6` detached; `PS.02` v3 (ruling 12). 7. The dial's success
gate. 8. Smell, stage 3 only, as §14 says (re-encoded nose, memory, ≥ 500k
decisions, shuffled-nose control). 9. Diary retrieval keyed on learned
state (§6), stage 3.

---

## SYSTEM — work on the machine (max 3 open)

*Order of work: **S9** → **S10** → **S6** (step 3).*

### S9 — Every settled FAIL gets a named fate (the queue that owned them is closed)
part: (machine — nothing faked) · stage: — · step: —
**Why.** `coverage` exits 2: closing the review queue removed
the "owner" of 33 settled FAILs. They are owned by THIS order until each has
a fate; a FAIL nobody owns is a failure filed, not fixed (philosophy 4).
**Do, ≤ 8 ids per slot, each one of three, never a baseline raise, never a
re-run for a better number.** (i) `repaired_by` in the registry when an open
order or roadmap milestone will re-earn it (name it). (ii) A `FAIL-DISPOSED:
<authority> <date> — reason` note (format as `registry.py:847`) when a
ruling already answered the question — e.g. ruling 10/20 for "to earn again
on the chosen brain", ruling 20 for one-world walking. (iii) A one-line ask
in INBOX when it is a live defect that needs a JACK order. The ids: DP.05,
HR.5, LG.10, LG.12, LG.14, LT.02,
PS.05, PS.06, PS.08, T0.07, T0.17, T0.23, T0.27, T0.32, T0.35, T1.08,
T2.05, T2.07, T2.11, T2.15, T3.09, T4.02, W.2, W1.00, WM.01, XL.01.
In the first slot also lower the floors `coverage` says fell (UNREACHABLE
115 → 101, PASS-ON-DEAD-DEPENDENCY 6 → 4; FAIL-UNOWNED 1 → 0 only when the
id list above is empty) with `run ratchets record` — `coverage` stays 2 until then.
serves: philosophy 4 · the auditor's exit codes.
**done when:** `coverage` exits 0 with this order's id list deleted.
**Added 11-17 (ruling 35):** `NE.01` joins the list with its fate already ruled — (ii) `FAIL-DISPOSED: Director ruling 35 2026-11-17 — heat is a stage-3 need (PLAN stage 3 "eat, drink, keep warm"); re-read at the first stage-3 sitting or 2027-01-15, whichever is first`. Also lower the three floors the auditor names (UNREACHABLE 115 → 107, PASS-ON-DEAD-DEPENDENCY 6 → 4, FAIL-UNOWNED when the list is empty).
state: slot 1 (11-17) — 8 fated by (ii): NE.01 (ruling-35), ME.11/.B/.C/.D + T2.10 (ruling-20, diary retrieval →
stage 3), HR.1 (D54), T1.11 (D41). Floors lowered: UNREACHABLE 115→107, PASS-ON-DEAD 6→4 (recorded).
The marker's authority must be ONE token (`ruling-35`, not "Director ruling 35" — that parses MALFORMED).
26 ids left; FAIL-UNOWNED floor 1→0 waits for the empty list. Next slot: the T1/T2 old-core FAILs.

### S10 — The clerk must be able to re-buy the two certificates that gate his world
part: (machine) · stage: — · step: —
**Why.** `regate` skips any certificate whose last run exceeded its 600 s
"cheap" label: `LC.02` (648 s) and `W1.02` (692 s), every tick. After J9's
world edit `LC.02` (which `default_cal()` needs before any pilot can be
built) and `W1.02` had no owner; `W1.04` sat blocked 8 hours. J11's options
and any J6 switch will stale them again.
**Do, one commit, planted test.** The smallest of: (a) regate buys a
certificate measured ≤ 900 s, at most ONE such per tick, nice 19, inside
the day's CPU ceiling; or (b) regate writes ONE INBOX line naming each
stale certificate it will not buy (once per staling, not per tick). Prefer
(a). Do not touch any spec's thresholds; the label is a cost label only.
serves: philosophy 5 (the loop faster and truer) · auditor finding 3 (11-17).
**done when:** a planted stale 650 s certificate is re-bought (or named) by one regate tick, and a planted 2,000 s one is not bought.
state: open — ordered 11-17 06:50.

### S6 — A video the owner can watch (owner request 11-12) — split so one slot finishes one step
part: (machine — he can be known) · stage: — · step: —
**Step 1 (one slot, done when a file exists):** `scripts/life_video.py`
renders OFFLINE with `experiments/render.py` (Xvfb + llvmpipe) one random-
action life of ≤ 60 s from `jack_env` (seed 153) — third-person view, 480p,
~12 frames a second, H.264 mp4 ≤ 5 MB — to `/data/jack-video/latest.mp4`.
Run by hand at nice 19; no cron, no service; never inside a scored run. A
rough clip today beats a finished one next week.
**Step 2:** hunger / thirst / temperature overlaid; `/data/jack-video/latest.json`
beside it: `title`; `summary` (2–4 sentences from a TEMPLATE over the life's
own record — how long he lived, what he ate and drank, what killed him; no
model); `facts` (≤ 8 `{label, value}`); `events` (≤ 60 `{t, at, label}`: `t`
seconds into the video, `at` the time in his life); `source` (life, seed,
commit). The dashboard's Watch tab reads both.
**Step 3:** the most recent RECORDED life, no picking (record poses if none are).
serves: M8 · philosophy 1 (he can be known).
**done when:** step 1: the mp4 exists and plays. Order done: steps 2–3.
state: steps 1–2 DONE (`070cea18`, `10b72b29`): seed 153 random life, 4:54, died of hunger, 1 meal
→ 41 s mp4 (1.66 MB, ~105 s wall, nice 19) with fed/unhurt/wet bars, age and a moment banner;
`latest.json` beside it (title, template summary, 8 facts, 6 events). World has no thirst or
temperature yet — the overlay shows the three needs it has and the summary says so.
Next: step 3 (most recent RECORDED life; poses already replayed per decision).

**Next up:** (0) `prereg.judge()` gets a door for unequal arms (a new
function; `__init__.py`'s pinned hash and check #1 re-bought once,
deliberately) — needed before the decisive test's calibration arms, not
before J11. (1) **S8** — a seat held BY RECORD in `champions --check`
(ruling 11; text in git at `b953e9db`). (2) `run.py` inside a worktree
writes that worktree's ledger (S7's open caveat). (3) `coverage` prints
where each passing CLAIM was earned. (4) `gpu.py` reattach pin-swap proof;
watcher timeout from RUNNING. (5) `BD.01`, `LC.08` budget re-class (ruling
14). (6) **S4** — six PASSes gated on a seed average that admits a failing
seed. (7) `T0.13`'s dead clauses. (8) D52 scout retry. (9) `XL.00` re-buy
adds a planted positive. `T0.33`→`T0.34`, `T0.12`→`T0.24` re-buys are
regate's on a clean tree.

---

## INBOX — one line each, workers add, the Director empties
- 11-17 SYSTEM (bookkeeping, not acted on): `run ratchets` still exits 2 on two floors no order owns — gpu_unattributed_jobs 27 above floor 21, live_unauditable_pairs 17 below floor 22.
- 11-17 07:38 SESSION (machine): root disk 4.1 → 6.4 GB free; three causes cleared by hand, two need fixing at the source by the system lane, each with a test. (1) The research desk cloned sources into `/tmp/j8p1` (499 MB, 11-16 16:06) although the header says scratch lives on `/data` only — moved to `/data/tmp/j8p1`, a link left at the old path; `scripts/research_prompt.md` should state the rule in its own words. (2) The CLI keeps every old copy in `~/.local/share/claude/versions/` (about 350 MB per update; five were there) — I removed the three no process was running; a weekly prune that keeps the launcher's target and any copy a running process uses belongs with the tmp reaper. (3) The owner's tunnel logs had no rotation (833 MB since August) — a rule now exists in `/etc/logrotate.d/xray`; not the team's to touch. Seen and left alone: `/tmp/aud_rev_*` worktrees from 11-11 (30 MB each).
- 11-17 07:31 JACK: `launch_detached.sh` refused J11's third Kaggle kernel ("2 busy detached jobs alive") while the two "busy" jobs were Kaggle pollers 40 s old (pip/push phase) and W1.04's wrap — a fresh poller counts as a busy box job; a free Kaggle CPU session went unused this slot.
- 11-17 07:31 JACK: `jack_env.py` moved (`4978e989`, two default-off options, identity proven) — `LC.02`/`W1.02` and any certificate with jack_env in IMPL_DEPS will read stale (S10).
- 11-17 07:31 JACK: free sessions seen at 07:29: J3 wm-latent, ppo-lp and J6 speed kernel 3 all COMPLETE on Kaggle, unharvested (their orders' next slots); J11 took two of the three.

## DONE (kept one day)

- **J8 — brain B's first pilot** (`72d38623`, `docs/research/PILOT_B.md`): 35 + 10 runs, 2,350 + 1,286 lives. VOID by its own floor clause (5 of 15 floor runs crossed, ≤ 1 allowed); inside it, B 0 sustained against 7.8 expected from doing nothing; no exploration scheme stops the slide. → J11.
- **J9 — the dying body's last observation** (`cfb49f25`): `terminal_obs="dying"`, three proofs pass.
- **J5 — brain A pinned and timed** (`ca4dcf00`): compiled 51.8 decisions/s on a T4; a 1M-decision life ≈ 5.4 T4-h of learning. → J14.
- **J7 — closed** : forager kernels harvested (pilot, no learner learned), `NE.01` re-run FAIL (parked, ruling 35), `W1.04` in flight (header).
- **S7 — another session's files cannot kill a run** (`36e20081`, `9a322df6`, `8d7388a1`, `c80d13b0`): four planted tests pass.
- **Idle slots:** MLX port 0.1–0.4e → J13; `scripts/mac_first_day.py` (`61cb3bda`); lab's eye probe → J12; Researcher §13 (movement cost) → J11, §14 (smell) → ruling 36.
