# ORDERS — what the workers do. Written by the Director.
Director sitting: 2026-11-12 06:40 UTC

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5). Machine problems go under INBOX, one
line each. The Director rewrites this file daily; git keeps the history.
Rulings cited here are in `docs/DECISIONS_RESOLVED.md` under the same date.

**GPU today.** Kaggle's 30 h week is spent (the account is shared with another
project); every push fails over to Colab in seconds at 0 h. **Colab runs ONE
session per account** — `D1.0` holds it now (watcher 1973610, 13 units of
≤ 1.8 h). Never start a second Colab job beside a live one (`colab ls` first).
A `state:` line may say "in flight" only with a ledger row or a live backend
status behind it. A waiter that only sleeps and then calls `dispatch.sh` may be
launched with plain `setsid` (ruling 11-12) — name its log in `state:`.

---

## JACK — build the creature, his world and his curriculum (max 7 open, ranked)

*Order of work: **J8 first** (25 CPU-minutes, seats a vacant seat). Then **J1**
— the creature order, CPU-only, waits on nothing. J2 and J4 are GPU: read
their logs every slot, act only on an exit. Then J6, J7, J5. A CPU order and a
GPU order may be advanced in the same slot.*

### J8 — Run the working-memory challenger (`WM.02`)
**Why.** The Working-memory seat is VACANT (the GRU buffer fails `WM.01`). The
lab's ring memory piloted 1.00/1.00/1.00/0.87/1.00 against GRU 0.55 on pilot
seeds; `WM.02` is registered on claim seeds 100–104 with two must-fail controls
and has never been run.
**Do.** `$PY -m experiments.run WM.02` (CPU, ~25 min; gates are in the spec
docstring — do not touch them). PASS → seat it BY VERDICT in
`docs/CHAMPIONS.md`, GRU kept as challenger. FAIL → mark it DEAD in
`docs/LAB.md` with the failing clause.
serves: M3 · philosophy 3 (a seat raced, not assumed).
**done when:** `WM.02` has a ledger verdict and the seat or the lab page says so.
state: **DONE 11-12** — WM.02 PASS (seeds 100–104, run at 2103829, hash-salt clean): ring 0.902±0.196 vs GRU 0.579, null 0.347;
mean lead +0.323 (bar 0.10), every seed earns/not-worse; shuffled 0.127, no-rotate margin −0.047, so both controls failed.
Seated BY VERDICT in CHAMPIONS (TRIGGER WM.01, WM.02; GRU kept as challenger); LAB entry marked SEATED.
Open, not this order: a non-commutative register, and ME.8/LF.02 with the ring in place of the GRU.

### J1 — Does living teach him? (`LF.04`) — the next creature gate
**Why.** `LF.03` PASSed: the learner lives, dies five times, resumes, and his
weights move. It says nothing about whether he gets BETTER: 2.3 meals per five
lives, spans 314–547 sim-s. Card line 1: "he learns everything himself by
living, dying and remembering." Nothing tests that on the whole creature.
**Do.** Register `LF.04` on `LF.03`'s harness (`lf_03_learner_lives.py`): the
seated core (`wm-latent`) vs **his own frozen twin** (same init, same worlds,
weights never updated), ≥ 3 seeds, N lives each. Claim: late-life survival
time (mean of the last third of lives) exceeds the twin's, paired, **on every
seed**, and exceeds his own first third. Controls that must fail, each RUN:
twin-vs-twin (gate must not fire); learner with rewards shuffled across steps.
Pilot ONE seed first to pick N and the margin from the twin's own life-to-life
spread — declare both before the claim seeds are read (pilot seed ≠ claim
seeds). If it FAILs, that is the finding: the order's next step is then "why
does the seated core not learn in lethal W0" (`LC.00`–`LC.02` pass on easier
venues) — diagnose, do not soften.
serves: M1→M2/M7 · philosophy 1 (learns by living), 4.
**done when:** `LF.04` is registered with both controls and has a verdict.
state: 11-12 LF.04 (7446b86) + BD.01 (c38683a) REGISTERED, NOT RUN: cpu<48h is closed by D20/D39 (SYSTEM 11-13 INBOX) - Director picks cpu<2h re-reg or S1.
Learner gap (LAB): oracle feeds ~95 % of births; smell dose2 DEAD; gamma .99 DEAD; 11-13 03:35 WEAN-WITH-SMELL DEAD by declared rule:
weaned meals nursed 1/9/1 vs null 2/0/3 (1 of 3); fed lives 1/1/1 vs 2/0/3; s92's 9 = one life born 0.70 m. Nursing feeds him (18/5/6 vs
decoy 0) but r climbs to 0.9-1.69 m and falls back to 0.5 - he never holds approach past ~1 m. Gap = credit over the walk, not birth distance.
Next: LF.04 when the Director re-budgets it; learner side: J5's CR.00 staged-widening pilot is the last world-side arm (harness 10de6fc).

### J4 — Run the one-brain contest (`UB.10`) on a venue that can tell arms apart
**Why.** Six fusion arms incl. A5, experts + router ("if A5 wins, one brain is
the wrong shape"); VOID 59 days on a saturated venue. Frees 4 specs. Yesterday's
pilot showed matched noise sd 2.5 makes every arm ineligible; a 4-venue sweep
with a selection rule declared before reading is queued.
**Do.** The sweep runs when Colab is free (waiter 1992023, or by hand if the
waiter is gone — it goes AHEAD of any `D1.0` re-dispatch). On "SWEEP EXIT 0":
apply `_select_venue` exactly as declared; set `VENUE_NOISE_SD`/`VENUE_VISION_SD`;
commit the pilot record; dispatch `UB.10`. If NO venue qualifies, do not add
venues by eye: write the four readings here and the Director rules. The
train-loss 4e-4 vs test 0.7 gap (nets memorise the fixed draw) goes in the
commit as a known property of the venue, not a thing to tune.
serves: M3 · philosophy 3 · standing objective 3.
**done when:** `UB.10` has a WINNER/TIE verdict, or a VOID with a different,
recorded cause and the next repair named.
state: 11-13 05:10 SWEEP HARVESTED (Colab, seed 90): _select_venue as declared -> (2.5 audio, 0.0 vision), margin .0719; (2,0) .0125;
(2,2)/(2,1) fail on A4 vslot .644/.750. All 6 arms eligible (slot A0 .866..A3 .925). Declared + pilot record committed 3fab32e
(foreclosure fixture rows now carry venue_vision_sd; smoke OK, live guard CLEAR). UB.10 DISPATCHED 05:09: watcher pid 2541605,
log /data/tmp/dispatch_ub_10.log, seeds 0-2, 1.25 GPU-h, tries Kaggle then Colab. Never kill the watcher.
Next: on the watcher's exit read the ledger row; WINNER/TIE -> done; VOID -> name the cause from the row, do not re-dispatch unchanged.

### J6 — The W0 world rules that are red: `NE.01`, `W.1`, `W.2`
**Why.** `NE.01` (needs are a real control problem) fails 2 of ~12 clauses —
`ok_night_shelter`, `ok_spread` — and frees 7 specs. `W.1` (temperature obeys
the published heat balance) and `W.2` (needs are a conserved ledger that can
kill) have been FAIL 73 days; M2's done-when names them, and the shelter clause
and the heat balance are very likely one fault.
**Do.** For each failing clause find the cause: world calibration (shelter
thermal balance, need spread across deaths, a leak in the ledger) or a clause
that cannot fire. Fix the world or the apparatus; never the bar. Start with
`W.1` + `ok_night_shelter` together. A world change re-stales `LF.03` — re-run
it after (minutes).
serves: M2 · philosophy 1 (a world that can kill him, honestly), 4.
**done when:** `NE.01`, `W.1`, `W.2` each have a new attempt with every
formerly-red clause explained in the commit. (D53 files `SO.07`'s VOID here too:
the body cannot reach privileged food on 2 of 3 recording worlds.)
state: 11-13 00:45 NE.01 (56c6519) FAIL: ok_night_shelter GREEN 3/3; red ok_spread 1/3 (3-drop flail probe truncates i; drop+rest spans .65)
and ok_forager 2/3 (a sleeper slides into sealed occ~1, hyperthermia at ~3540/3600 s - the knife-edge). No world edit (D43 HOLD stands).
11-13 04:40 occlusion-law RACE written + LAUNCHED (experiments/occ_law_race.py/.sh, pid 2503265, log /data/tmp/occrace/race.log, rows
/data/tmp/occrace/*.jsonl): shipped vs graded (49-ray cos-weighted cone) vs damped sleeper (x3; x5/x8/x20 broke RK4) vs open (must fail),
seeds 0-4, NE.01's own probes + bars, rule in docstring. Waits for CR.00, ~2 h on 2 workers. Next: on "OCCRACE EXIT" apply the rule; then drop probe.
11-13 04:45 W.2 DIAGNOSED (ledger 08-30, no run): a/b/d GREEN (integrator 1e-13, ledger resid 0, tau 700/160 exact); red = (c) only: deaths
hunger 1700 s vs 25200 (k=72, ratio 14.8), thirst 570 vs 3600 (6.3), tol 2x. Cause = needs.py B_E 1/1800, B_W 1/450: needs run ~10x past the declared clock. Same class as W.1(b) - Director ruling (INBOX 04:45).
### J7 — The deeper world (W1): `W1.00` FAIL, `W1.04` VOID
**Why.** Owner, 2026-11-11: the world and the curriculum are "as important as
jack himself". W0 is too shallow: lookahead does not pay in it (`DP.05`), a
ridge regression beats the world model (`T2.05`). Fast/slow (`BO.01`) cannot be
raced until thinking ahead is worth something. `W1.01`, `W1.02` already pass.
**Do.** Repair `W1.00` and `W1.04`; then register `DP.05`'s question in W1.
serves: M2 · philosophy 5 (his world matters as much as he does).
**done when:** both have new verdicts and `DP.05`-in-W1 is registered or run.
state: 11-13 01:45 W1.04 VOID = float-clock apparatus fault, REPAIRED 8495d56 (not run); expect control green + claim FAIL (c).
11-13 02:40 W1.00 RE-REGISTERED (attempt 2, not run): n 3 -> 20 on FRESH seeds 300..319, t made exact (sqrt(n-1)/std_pop, was 1.22x
generous at n=3), cpu<2h; no bar/control/channel touched. Sized from attempt 1: dwell d=0.93/seed, E[t] 4.2 at n=20 (power ~.87),
white2-white floor SE .0019 vs bar .0062. Smoke at seed 319 OK (all 4 procs, finite). ~192 core-min, ~65 min on 3 workers.
Next: after wean-smell exits, run W1.04 then W1.00 (`launch_detached.sh … experiments.run W1.00`), one at a time - box is shared.

### J5 — The curriculum, as a cook-off
**Why.** Owner, 2026-11-11: *"the training curriculum … as important as jack
himself, put as much research into that."* Today there is none: he is dropped
into W0 and scored. `CU.1`–`CU.7` have never run; curriculum has no seat.
**Do.** (1) Research pass, cited, to `docs/research/CURRICULUM.md` (≤ 15 KB):
how children and animals are staged; learning-progress curricula, goal
babbling, teacher-proposed goals, unsupervised environment design. (2) Add a
Curriculum seat to `docs/CHAMPIONS.md`, holder VACANT. (3) Register ONE
cook-off: no curriculum (the null) vs hand-staged vs self-chosen by learning
progress vs teacher-proposed — same core, world, lives; metric = `LF.04`'s
(survival gain at matched experience); control: a shuffled-stage curriculum
must not beat the staged one. Pilot before sizing. Arms stage the WORLD and
what the adults say — never his actions.
serves: M2 · philosophy 3, 5.
**done when:** the doc and the seat exist; the cook-off is registered with its
control and has a pilot number.
state: 11-13 01:40 steps (1)+(2) DONE: docs/research/CURRICULUM.md + Curriculum seat VACANT, ARENA: NONE (champions --check rc 0).
11-13 02:40 CR.00 harness 10de6fc (null / staged .6,1.2,2.4,4.0,U / shuffled same blocks; uniform EXAM; fed-life frac + median span).
11-13 03:38 LAUNCHED detached: experiments/cr00_pilot_race.sh (log /data/tmp/cr00_race.log, rows /data/tmp/cr00/*_s89.jsonl):
waits for WEAN_SMELL exit, then 3 arms s89 train 40000 exam 20000 no smell in parallel (~60-90 min), then tally. 11-13 05:15 PILOT HARVESTED (s89, exam 8-9 lives/arm): fed null 5/8, staged 2/8,
shuffled 0/9; median span 516/493/439 (sd 49/19/42) - staged does NOT beat null on the pilot. Next: set N + margin from null's spread, register CR.01.

**Next up, in this order, as slots free:**
1. **The diary in his decisions** (opens when `LF.04` has a verdict): today
   the diary crosses death as luggage — nothing in the learner's forward path
   reads it. Cook-off: core with retrieved diary rows in its input vs without,
   wiped-diary twin as the control that must lose (M7; card line 1 "remembers").
2. **Inside/around sweep** — the 2026-11-11 ruling: re-credit `LG.00`, `HR.4`,
   `T3.08`, `LG.10/12/13/14`, `ME.11.C/D/E` as *parent-rig* results in coverage
   and CHAMPIONS; delete the paid `AudioListener._transcribe_api`; strike
   `PG.6`'s "frozen tower" kill-text; re-run `LG.00` with the learned core in
   the loop.
3. **Learned ear** — a plastic speech ear vs a no-ASR null on a small
   vocabulary (M5/M6). `HR.1`, `HR.5` (voice corpus, playground sounds) are
   repaired under this order when it opens — unowned until then BY DECISION.
4. **Judge admission gate** (FIELD_WATCH N1, arXiv:2611.13596): before any
   adult's score counts, it must accept a real success from Jack's world AND
   reject the same episode with the deciding event removed; a yes-to-all judge
   must fail. Opens with J5's teacher arm, which is its first user.
5. **Sense added later vs from birth** (M5); `XL.01` at proper power; `LT.02`
   (unowned BY DECISION until curiosity is raced under J5); `ME.11.B/F`.

---

## SYSTEM — work on the machine (max 3 open)

### S5 — A finished Colab run must not be lost to its download
**Why.** Three runs have computed everything and then lost the result when the
kept-session download lost the VM (`T1.08` ×2, `T1.14` attempt 2: 1.10 GPU-h).
`gpu.py` keeps a 400-char stdout tail; only `T1.14`/`t108_probe` have the
`JACKRESULT` stdout fallback. **`D1.0` is running 13 units without it.** Colab
also drops the CLI's stderr ("exit 1" was a busy session).
**Do.** In `experiments/gpu.py` only: (1) `submit` persists each job's full
stdout under `/data` and, when the artifact download fails, parses the last
`JACKRESULT` line generically; job templates print it. (2) Record the colab
CLI's stderr in the result row; say "session busy" when that is the cause.
(3) `dispatch_guard` refuses a Colab-first dispatch while `colab ls` shows a
kept session. Extend `T0.12`/`T0.11` with a planted lost-download and a planted
busy session. Net lines in `gpu.py` should not grow by more than ~60. The live
`D1.0` watcher keeps its loaded code — do not restart it.
serves: M3/M4 (every GPU order) · philosophy 5, 6 (free hours not wasted).
**done when:** a test loses the download and the verdict still lands from
stdout; a test plants a busy session and the dispatch refuses with the reason.
state: **DONE 11-12 d9c94ba** — fetching Colab jobs are wrapped in try/finally that prints each fetch file as a `JACKFILE` stdout line;
a failed download is recovered from it (generic, all specs, no spec edits). `colab sessions` read before each run: held -> "session busy",
never runs. Full stdout/stderr kept in /data/jack-logs/gpu_stdout/, result row has stderr_tail. T0.12 PASS clean (props 13-14, pre-S5
control fails all 3). gpu.py +72 net. Not done: dispatch_guard itself unchanged (the refusal sits in run_on_colab, which every path calls).
S1 opens (same file now free).

### S3 — Guards that are red by design teach everyone to ignore red
**Why.** `coverage` exits 2 on a shrink-only UNREACHABLE counter that moves
every time an order registers a spec; its FAIL-UNOWNED check names the closed
REVIEW_QUEUE as the repair; `run verify` exits 2 on the standing `T0.18`
KeyError. `ratchets record` rewrites all 8 moved counters at once, so J1 could
not bank `pass_on_dead_dependency` 7 → 5 without blessing 7 strangers.
**Do.** (0) Give `ratchets record` an `--only <counter>`; record
`pass_on_dead_dependency` = 5 and `unreachable` = 93 with it (both earned:
`T1.14` PASS `73665ca`, ruling 11-12). (1) Move the by-design-red counters out
of the exit code (still printed). (2) Point FAIL-UNOWNED at this file (owned =
named in an open order or in "Next up"). (3) Fix the `T0.18` KeyError.
(4) `run status`'s STEERING-PAGE reader parses 0 items from 3 pages — aim it
at this file's `### ` orders or delete it. (5) `T0.28` went FAIL 11-12 00:52 on
`p10_live_document_is_armed_and_readable`: no decision is armed today (all
defaults fired), and p10 reads an empty register as a broken one. Make it tell
empty from unreadable; a planted unreadable page must still fail. Nothing that guards a threshold, a
control or a seed count leaves the exit code. Smaller, not bigger.
serves: philosophy 4, 5 (a truthful machine).
**done when:** on a clean tree `coverage` and `run verify` exit 0, and each
still exits non-zero when a test plants a PASS-on-FAIL-dependency.
state: 11-12 SYSTEM: (0) `ratchets record --only a,b` DONE; nothing banked, the ruling's numbers are stale (live pass_on_dead_dependency 9, unreachable 96; INBOX).
(3) DONE: verify judges a row whose gate grew a key since by the gate at the row's commit (T0.18@7ffd961 re-derives PASS); a gate
raising for any other reason stays unevaluable. `run verify` exits 0 (was 2). Re-running T0.18 can't fix it: BLOCKED by T0.13 writes no row.
Net 0 lines (T0.18-specific prose cut). Open: (1) coverage exit code, (2) FAIL-UNOWNED -> ORDERS, (4) STEERING reader, (5) T0.28 p10, planted-dep test.

### S4 — Six standing PASSes are gated on a seed AVERAGE that admits a failing seed
**Why.** `run worst-seed-audit`: `ME.10` (diary vs skill), `NE.00`, `LT.01`,
`PG.4`, `PS.02`, `T2.20` each pass a bar on the mean over seeds while the
recorded mean and spread allow one seed on the failing side (`ME.10`
`skill_gain`: mean 0.370, a seed may sit at 0.237 against a bar of 0.25). "At
least 3 seeds" means each seed. Ordered 2026-09-30, never done.
**Do.** For each of the six: gate the flagged key per seed (record `_min` /
`_max` folds and gate on them; bars byte-unmoved), check `depends_on` first,
re-run. Start with `ME.10` (CPU, minutes). A spec that turns red stays red:
write its id under INBOX and the Director orders the repair.
serves: philosophy 4 (several seeds, nothing faked) · M7, M2 (their roots).
**done when:** the audit's LANE B lists none of the six, each with a verdict
from a per-seed gate.
state: open — not started.

**Next up:** (1) **S1 — free CPU off this box**: a CPU-only Kaggle kernel mode
that charges no GPU quota; pilot one W0 survival run; sim-seconds per real
second vs this box (`LC.07` is priced at 526 h here). Opens when S5 closes
(same file). (2) `T0.13`'s 13 gate clauses that can never fire — arm or remove
with reasons.

---

## INBOX — one line each, workers add, the Director empties
- **S1 PILOT MEASURED 11-12 07:35 (owner session, private kernel `jack-cpu-pilot-1794467678`, CPU-only, `enable_gpu:false`):** a Kaggle CPU push is ACCEPTED with the weekly GPU quota spent. Machine: Xeon 2.2 GHz, 2 cores / 4 threads, 31 GB. `run_survival` wm-latent train, 600 decisions, 1 torch thread: 1 proc 28.0 dec/s; 2 procs 34.2+33.8 = 68.0 total; 4 procs 4×15.4 = 61.6 total (no gain past 2 — pack TWO learners per session, not four). Random policy: 1 proc 67.6; 4 procs 112.6 total. Same script on this box: 20.5 learner, 49.4 random. So one learner run of 4M decisions ≈ 33 h = three 12-h sessions with resume; `LC.07` whole ≈ 200–270 session-hours (rough; mix of arms not priced). Unmeasured: how many CPU sessions Kaggle runs side by side. Needs `pip install mujoco` in the kernel and the https repo URL.
- **OWNER PRIORITY 11-12 (via session): "the scale thing is a big one … when will we run it bigger scale against the others".** The Learning-core seat rests on a 1× single-learner screen and `LC.07` has been VENUE-UNAFFORDABLE since 09-12. Director: (1) promote S1 (CPU-only Kaggle kernels, no GPU quota) from "next up" to an open order and price `LC.07` on it; (2) consider registering a NEW intermediate spec (e.g. ~3×, all five arms plus ring-memory if seated, collapse diagnostic of LEARNING_CORE §5.4 implemented, weights persisted) as its own claim — it does NOT replace or shrink `LC.07`, whose ~10× ratio stays untouched; (3) say in PROGRESS when a scale result can honestly be expected. Free compute only.
- **OWNER REQUEST 11-12 (via session): a video he can watch.** Make a system order: after a life is lived, write a short watchable video (third-person view of the body in his world, needs/hunger overlaid, phone-playable mp4) to a fixed path under `/data`, refreshed whenever a newer life exists, and say in `docs/PROGRESS.md` where it is. Use `experiments/render.py` (Xvfb + llvmpipe); render OFFLINE from a recorded life, never inside a scored run; low priority CPU, no background service. Show him as he is — no cherry-picked life.
- **11-12 07:10 JACK: both GPU backends refuse; J2/J4 are stalled.** Kaggle: "Maximum weekly GPU quota of 30.00 hours reached" (shared account, W45). Colab: a bare hello-world `colab run --gpu T4` fails at `POST /tun/m/assign … variant=GPU`: **503 Service Unavailable** (our W45 colab 4.41 h). gpu.py reports this only as "exit 1" (S5 item 2). Also: the UB.10 sweep waiter (pid 1992023) was a hand-chained nohup that inherited the launching slot's JACK_ITER_DEADLINE and refused Colab with "187s left". Chained GPU waits need launch_detached/dispatch.sh, which strip it. Re-launched the sweep by hand at 07:08 via launch_detached; it was refused by both backends (/data/tmp/ub10_sweep_rerun.log). D1.0 attempt 3 = ERROR, a lost download after 1.49 GPU-h (2103829). Next GPU slot: probe `colab run` first.
- **11-13 SYSTEM -> Director (replaces J1's 08:45/18:05 lines): the cpu<48h refusal is NOT a bug, it is D20/D39.** D20 closed `cpu<48h` to registered specs in both lanes and banned new registrations there, held by prose only; `LC.08` (10-20), `LF.04` and `BD.01` (11-12) were registered into it anyway. The gate's message said "is the detached lane (launch_detached.sh)" and sent J1 there twice; it now names D20/D39 and the two exits. Director must pick one: (a) re-register LF.04/BD.01 in `cpu<2h` - a never-run 3-seed cpu<2h spec projects at its 54000 s kill allowance, so it admits only on a day with <= 3600 s used (launch just after local midnight, before detached lab work); or (b) run them on S1 (Kaggle CPU). Not done here: re-budgeting is a registration act. Also: T0.33's `slack_predicts_refusal` reads False under today's live state (unrelated to this change; noted, not chased).
- **11-12 11:50 LAB → J1/J5/J6 (World): he cannot sense floor food before touching it.** Seed-90 W0, all 584 legal poses: his 16 rays leave horizontally at z 0.70 m, floor food tops out at ~0.33 m — rays hitting floor food 0/584 at every distance; `observe()` has no smell and proprio has no x,y. Weaning pilot (`experiments/wean_pilot.py`, 13k decisions/arm, AWR): born ≤ 0.55 m he ate 3 lives of 3, born at 1.07/1.64 m he missed; after uniform births resumed nursed 0 meals = null 0 = decoy control 0. `LF.04`'s margin needs foraging, and foraging here is blind bumping: a World repair (smell into `W0.observe`, or downward rays — raced) comes before any `LF.04` re-run or J5 staging. Details `docs/LAB.md`.

- **11-12 SYSTEM S3:** ruling 2's counts no longer hold. `pass_on_dead_dependency` reads 9, not 5 — new: `DP.00`, `LC.08`, `LF.03`, `XL.00` <- `LC.02` FAIL, `T0.18`/`T0.19` <- `T0.13`, `PL.00`, `T1.12`, `LF.02`. `unreachable` reads 96, not 93. Nothing recorded; `--only` is ready when the Director names the values to bank.

- **11-12 19:40 JACK J1 [FIXED d05c997, diagnostic moved to experiments/lf04_collapse_diag.py]:** `run next` (and any LF.04 run) RAISES: "LF.04 has 2 implementations: lf_04_collapse_diag.py, lf_04_living_teaches.py" — the tracked diagnostic (9df00c2/ffe45e5) sits in experiments/tests/ under the spec's prefix. Move/rename it out of the `lf_04_` namespace (check its launch scripts first); not done here.
- **11-12 20:25 JACK J6 -> Director: W.1 (b) vs the world's clock is a design conflict, not a bug.** W.1's notes and W.7's registered control say thermal physics runs at 1x (a 70 kg body's tau = 4.74 h "cannot be sped up by a clock convention"); W0 ships TAU_T = 240 s (~71x compressed) so a 400 s night can chill him at all. Under W.7 the shipped world IS the naive-compression variant. Compressing (b) would pass it at 33.80 but certify what W.7 exists to forbid, so I did not. Rule: re-register W.7/W.1 for a compressed thermal clock, or slow the world's thermal tau (and lose cold nights). Also: J4's state still names waiter 1992023 - it died 06:53 (inherited JACK_ITER_DEADLINE, see 07:10); my 20:07 relaunch via launch_detached was refused at load 6.5 (/data/tmp/ub10_sweep2.log).
- **11-12 23:35 LAB → J1/J5: run the weaning-WITH-smell race (LAB's last entry; rule declared, nothing read).** `scripts/launch_detached.sh /data/tmp/wean_smell/run.log experiments/wean_smell_race.sh` (mkdir /data/tmp/wean_smell first) — REFUSED tonight by the CPU day budget (57,600 s used), so launch after 00:00 UTC. 6 runs (nursed/decoy × s91–93, 52k decisions, nurse 26k), 1 worker until J1's gamma99 race exits then 2; ~4–5 h. Harvest on "WEAN_SMELL EXIT": `$PY experiments/wean_smell_tally.py`, apply the rule in the script header (nursed must out-eat decoy AND the dose2 null 2/0/3 on ≥ 2 of 3 seeds, weaned lives only), write ALIVE/DEAD in LAB. Also 11-12 LAB: bout (pink/OU) exploration DEAD at stage A — fed births white 4/40, pink 6/40, though pink roams 1.7× wider; range is not the gap.
- **11-13 00:55 JACK: wean-with-smell race LAUNCHED** (INBOX 23:35): launch_detached pid 2432985, log /data/tmp/wean_smell/run.log; it idles until gamma99 exits (~01:40), then 2 workers ~4-5 h. Note: the meter bills its idle wait as wall clock. Harvest on "WEAN_SMELL EXIT" with wean_smell_tally.py.

- **11-13 04:25 JACK J4:** a Colab probe with the bare `colab run` prompts for OAuth and looks like "auth broken"; gpu.py uses `/data/venvs/colab/bin/colab --auth adc`, which works. Probe with the latter, or the 503/quota story gets mis-retold.

- **11-13 04:45 JACK J6 -> Director: W.2's only red clause, (c), is the world's needs clock, not a leak.** a/b/d PASS (closed forms 1e-13, conservation resid 0 over 10 days, sleep taus exact). (c): starvation 1700 s vs 21 d/72 = 25200 s (ratio 14.8), dehydration 570 s vs 3 d/72 = 3600 s (6.3); bar is a 2x factor. `needs.py` B_E=1/1800, B_W=1/450 compress needs ~1070x/455x against the clock's declared 72x; human hunger:thirst is 7, W0's is 3.0. No single declared k is honest unless the world declares it: k in [~530, ~910] would pass both, but picking it from this reading fits the bar to the data, so I did not re-register. Rule with W.1(b) (20:25 line), as one question: (i) slow B_E/B_W to k=72 (lives ~10x longer, LF/CR costs x10; world edit, D43 HOLD) or (ii) declare a separate needs compression in W.7/W.2 from a design reason stated before any re-run.

## DONE (kept one day)

- **`LF.03` PASS** (`6a92fd4`, was J3): the learner lived 5 deaths × 3 seeds,
  weights moving, diary carried, bit-exact resume; frozen twin and wiped diary
  both flagged. Roadmap M1 met. Integration only — J1 asks the next question.
- **`T1.14` PASS** (`73665ca`, was J1): SNR 51.89, seed-ignoring arm run and
  rejected. `T2.03`/`T2.14` stand on a proven floor again.
- **S2** (`4749711`): Kaggle no-session = shared quota spent; push refusal and
  404 now fail over at 0 h; 3.01 h refunded.
