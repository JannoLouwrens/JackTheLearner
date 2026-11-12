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
state: 11-12 LF.04 REGISTERED 7446b86 (15 lives, s100-102), NOT RUN (CPU gate, INBOX 08:45). Pole cannot stand; plinth stands.
Pilot path oracle (s90/s94): plinth 30/40 vs pole 23/40. **BD.01 REGISTERED c38683a** (plinth vs pole, path oracle,
40 births x s100-102, gate pooled +15/120 & no seed behind & plinth topple<=10%; control = spawn-salted plinth twin).
Smoke OK. NOT RUN: `run BD.01` refused "cpu<48h is the detached lane" even under launch_detached (same INBOX 08:45 blocker).
Next: once the runner can take a cpu<48h spec, launch BD.01 detached (~2 h CPU).
17:45 pool lip DIAGNOSED (LAB.md): no pit - walls' tops z 0.000 coplanar with the infinite floor plane; catch = buried-box
seam. Next: log poolwall contact normals at a plinth stall (geo.py base), then register a World repair race.

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
state: 2026-11-12 5bce9ef: seed-90 pilot at matched sd 2.5 FALSIFIED the venue: vslot A0 .756, arms .61-.80, uni-vision
.74-.79 -> all 6 arms ineligible (would VOID); afell .83-.88; train loss 4e-4 vs test ~.7 (memorises the fixed draw). Guard now
refuses an ineligible pilot (live: refuses). Audio/vision levels split; sweep (2,2),(2,1),(2.5,0),(2,0) + _select_venue rule
declared pre-read. Sweep QUEUED behind D1.0 (Colab single-session): /data/tmp/ub10_sweep_after_d10.log, waiter pid 1992023.
Next: on "SWEEP EXIT 0" declare the chosen venue in VENUE_NOISE_SD/VENUE_VISION_SD, cp the pilot record, commit, dispatch UB.10.

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
state: open — not started.

### J7 — The deeper world (W1): `W1.00` FAIL, `W1.04` VOID
**Why.** Owner, 2026-11-11: the world and the curriculum are "as important as
jack himself". W0 is too shallow: lookahead does not pay in it (`DP.05`), a
ridge regression beats the world model (`T2.05`). Fast/slow (`BO.01`) cannot be
raced until thinking ahead is worth something. `W1.01`, `W1.02` already pass.
**Do.** Repair `W1.00` and `W1.04`; then register `DP.05`'s question in W1.
serves: M2 · philosophy 5 (his world matters as much as he does).
**done when:** both have new verdicts and `DP.05`-in-W1 is registered or run.
state: open — not started.

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
state: open — not started.

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
- **11-12 08:45 JACK J1: no runner path for a multi-hour CPU spec.** `gate_cpu_child` refuses every `cpu<48h` spec as "the detached lane" — even when the runner itself was started by `launch_detached.sh` (which already meters it per heartbeat); re-budgeted as `cpu<2h`, a spec with no recorded duration is projected at its full child timeout (54000 s for 3 seeds) and refused against 48206 s remaining. LF.04 (~4-5 h real) is registered and smoke-tested but cannot get a verdict through either lane. Also 08:05: `colab --auth adc run --gpu T4` hello-world still 503 at /tun/m/assign; plain `colab run` (no --auth adc) asks for an interactive OAuth code.
- **11-12 11:50 LAB → J1/J5/J6 (World): he cannot sense floor food before touching it.** Seed-90 W0, all 584 legal poses: his 16 rays leave horizontally at z 0.70 m, floor food tops out at ~0.33 m — rays hitting floor food 0/584 at every distance; `observe()` has no smell and proprio has no x,y. Weaning pilot (`experiments/wean_pilot.py`, 13k decisions/arm, AWR): born ≤ 0.55 m he ate 3 lives of 3, born at 1.07/1.64 m he missed; after uniform births resumed nursed 0 meals = null 0 = decoy control 0. `LF.04`'s margin needs foraging, and foraging here is blind bumping: a World repair (smell into `W0.observe`, or downward rays — raced) comes before any `LF.04` re-run or J5 staging. Details `docs/LAB.md`.

- **11-12 SYSTEM S3:** ruling 2's counts no longer hold. `pass_on_dead_dependency` reads 9, not 5 — new: `DP.00`, `LC.08`, `LF.03`, `XL.00` <- `LC.02` FAIL, `T0.18`/`T0.19` <- `T0.13`, `PL.00`, `T1.12`, `LF.02`. `unreachable` reads 96, not 93. Nothing recorded; `--only` is ready when the Director names the values to bank.
- **11-12 18:05 JACK J1: second spec stuck on the 08:45 cpu<48h gate.** `launch_detached.sh … experiments.run BD.01` -> "REFUSED before start: cpu<48h is the detached lane (scripts/launch_detached.sh), not a runner child" — the runner refuses the very lane it names. BD.01 (~2 h) and LF.04 (~4-5 h) both registered, both unrunnable. Log /data/tmp/smell_sense/bd01_run.log.


## DONE (kept one day)

- **`LF.03` PASS** (`6a92fd4`, was J3): the learner lived 5 deaths × 3 seeds,
  weights moving, diary carried, bit-exact resume; frozen twin and wiped diary
  both flagged. Roadmap M1 met. Integration only — J1 asks the next question.
- **`T1.14` PASS** (`73665ca`, was J1): SNR 51.89, seed-ignoring arm run and
  rejected. `T2.03`/`T2.14` stand on a proven floor again.
- **S2** (`4749711`): Kaggle no-session = shared quota spent; push refusal and
  404 now fail over at 0 h; 3.01 h refunded.
