# ORDERS — what the workers do. Written by the Director.
Director sitting: 2026-11-13 08:45 UTC

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5). Machine problems go under INBOX, one
line each. The Director rewrites this file daily; git keeps the history.
Rulings cited here are in `docs/DECISIONS_RESOLVED.md` under the same date
(1–4 at 06:40, 5–7 at 08:45).

**The owner spoke today (11-13).** The finished Jack — human-like body; trained
here, handed to people, still learning there; advised in words — now judges
every design (END-FIT, top of `docs/CHAMPIONS.md`; card line 7). Every
architecture cook-off says in its registration which END-FIT rows its arms
serve; no winner is adopted that closes a row. `LF.04` still comes first.
**GPU today.** Kaggle's GPU week is spent by the shared account (our budget
file cannot see that — trust Kaggle's refusal, not the file). Colab holds
`D1.0` (ONE session per account — `colab ls` first; probe with
`/data/venvs/colab/bin/colab --auth adc`). "In flight" needs a live pid.
**CPU today.** The 57,600 s day ceiling is shared by everything; 35,852 s were
used by 08:40. Until `LF.04` has a verdict, **no new detached learner-side
pilot** (ruling 3): the CPU goes to registered runs (J1, J7) and J6's world race.

---

## JACK — build the creature, his world and his curriculum (max 7 open, ranked)

*Order of work. CPU: **J1 step 1** every slot the box is quiet (else S1's
first step) → **J7** harvest → **J6** (run `W.1`, build `W.7`, race the needs
clock) → **J8** → J5 → J3. GPU: **J2** — read the log, act only on an exit.
A CPU order and a GPU order may be advanced in the same slot.*

### J1 — Does living teach him? (`LF.04`) — the creature gate, 48 h unrun
**Why.** `LF.03` shows he lives, dies and resumes; nothing shows he gets
BETTER. `LF.04` (learner vs his own frozen twin, two controls) is registered
and has never run, for two reasons, both now ruled.
**Do.** (1) **`LC.02` first.** It went PASS → FAIL on 11-12 10:52 while a
detached race loaded the box; it is a wall-clock claim, `LF.04` depends on it,
and it blocks 19. Re-run it on a quiet box (`uptime` load < 1.5, no detached
job of ours alive; say the load in the commit). PASS → the FAIL was contention:
add a precondition to the spec that returns VOID (not FAIL) above a stated
load, with a planted-load test. FAIL when quiet → the core got slower: bisect
`cores.py`/`survival.py` since `d5d26b0`'s last PASS and write what you find.
(2) **`LF.04` and `BD.01` run on Kaggle CPU** (ruling 2; SYSTEM S1 builds the
route, `LF.04` is its first customer). Do not re-budget them to `cpu<2h`.
If it FAILs, that is the finding — diagnose, do not soften.
serves: M1→M7 · philosophy 1 (learns by living), 4.
**done when:** `LC.02` has a quiet-box verdict and `LF.04` has a verdict.
**Not waiting.** S1 has no code yet and is `LF.04`'s only route: a JACK-lane
worker who finds the box loud does S1's first step instead of idling. Try
`LC.02` FIRST in every slot, before launching anything detached; if no quiet
minute comes by the 11-14 sitting the Director rules on the precondition.
state: 11-13 10:30 J1(1) DONE: LC.02 attempt 21 PASS on a quiet box (load .10; 4b67b0b) -> the reds were contention (+ c510b92, fixed ede21ca).
Contention precondition shipped e9e07c2: FAIL with >1.0 foreign core in any timed window -> VOID, PASS stands; meter = /proc/stat minus own
jiffies, planted 2-spinner load must read >=1.5 every seed. Attempt 22 PASS on it (planted 2.41, foreign max .75; T 6.57-8.07; trunk .34).
Next: J1(2) — LF.04 on S1's kaggle-cpu backend (06b87bd); LF.04/BD.01 REGISTERED, NOT RUN.

### J2 — The control-path contest (`D1.0`) — attempt 4 in flight on Colab
**Why.** `T2.01` (he moves better than random) is the biggest blocker in the
ladder: frees 25, red since Aug 12, waiting on which controller holds the
seat. `D1.0` attempt 3 computed 1.49 GPU-h and lost it in the download; S5
(`d9c94ba`) now recovers results from stdout, and Colab is back.
**Do.** When `UB.10`'s session ends (`colab ls` empty): dispatch `D1.0` via
`scripts/dispatch.sh` unchanged (13 units of ≤ 1.8 h, resume between units).
Read its log each slot; act only on an exit. On a verdict: seat the winner BY
VERDICT, then re-run `T2.01` under it.
serves: M3, M4 · END-FIT row 1 (human-like body) · philosophy 3, 7 · objective 3 (free GPU).
On a verdict, write in the seat row which END-FIT rows the winner leaves reachable (fast loop inside one brain; hands).
**done when:** `D1.0` has a verdict, or an ERROR with a named cause that is
not a lost download.
state: 11-13 10:20 attempt 5 ERROR at 0 GPU-h (2d41481): the SB3 reference unit WAS saved (S5 stdout recovery, /data/d10_units/3fbc6d0c,
ref_mean 650.5 >= 450) and was reused; unit aprime/seed0 refused: Colab T4 assign "Service Unavailable" (2nd today), Kaggle weekly quota spent.
Next: when `colab new`/assign works or Kaggle's week resets, re-dispatch unchanged (`dispatch.sh D1.0 --projected-hours 25.05`); 12 arm units left.

### J6 — W0's red world rules: one clock (`W.1`, `W.2`), then `NE.01`
**Why.** `W.1`(b) and `W.2`(c) are one fault, ruled today (ruling 1): **W0 has
ONE declared clock, k = 72.** Thermal already obeys it (TAU_T 240 s =
4.74 h / 72). Hunger and thirst do not: they run 14.8× and 6.3× past it, so he
dies of thirst in 570 s — under half a world-day, before a night can teach him
anything. That may be why living does not teach him.
**Do.** (a) Re-register `W.1`(b) and `W.7`'s control against the DECLARED
clock: a channel whose rate differs from k = 72 by more than the existing
tolerance must fail; shipped thermal should pass, shipped needs must fail
`W.7`. Bars byte-unmoved. (b) Race the needs clock, no world file edited until
it wins: playground arm B_E, B_W at k = 72 (starve 25,200 s, thirst 3,600 s)
vs shipped, on `NE.01`'s probes and `W.2`'s checks, seeds 0–4, rule declared
first: the k = 72 arm ships if it passes `W.2`(c) and is no worse than shipped
on every `NE.01` clause. Report decisions-per-life for both (the cost). A
race-won edit may ship (ruling 1 lifts D43's hold for exactly that); re-run
`LF.03` after. (c) Then the band-geometry arm and the drop probe for
`ok_spread`.
serves: M2 · END-FIT row 4 (late payoffs need a world whose deadlines allow them) · philosophy 1, 3, 4.
**done when:** `W.1`, `W.2`, `NE.01` each have a new verdict with every
formerly-red clause explained.
state: 11-13 04:45 W.2 red (c) only: starve 1700 s vs 25200, thirst 570 vs 3600 (needs.py B_E 1/1800, B_W 1/450). NE.01 red ok_spread/ok_forager.
11-13 08:25 (a) DONE in registry: W.1(b) on k=72 (fa0f7fc), W.7 control (98134d7). W.1 not yet re-run (lock).
11-13 12:20 W.7 IMPLEMENTED (2685dd9), smoke s0 FAIL as registered: thermal k 71.1 ok; thirst 455, hunger 1067 FAIL; NEW: sleep clocks
also off-clock: tau_wake 700 s -> k 93.6, tau_sleep 160 -> 94.5 (k=72 wants 910 / 210 s). Controls good (k1/k60 fail, k72 passes, null rejects).
Next: when the lock frees, `run W.7` and `run W.1` (no ledger rows yet); then race (b) with the sleep taus in the k=72 arm (or a 3rd arm), seeds 0-4.

### J7 — The deeper world (W1): run the two repaired specs
**Why.** W0 is too shallow for thinking ahead to pay (`DP.05`); fast/slow
cannot be raced until it does. Both repairs are committed and unrun.
**Do.** `W1.04` (float-clock repair `8495d56`), then `W1.00` attempt 2 (n = 20,
seeds 300–319, ~65 min on 3 workers, `launch_detached.sh`), one at a time.
Then register `DP.05`'s question in W1.
serves: M2 · END-FIT row 4 · philosophy 5.
**done when:** both have new verdicts and `DP.05`-in-W1 is registered or run.
state: 11-13 10:40 W1.04 HARVESTED (640e329): FAIL on (c) only — 5th-pct life 54.3/52.0 s vs 5600 s horizon, ~272/275 deaths energy.
11-13 11:45 chain pid 3050089 harvested (bfa09b8): PS.01 PASS (850.6 s, hash-salt clean), W0.DIAG PASS but DIRTY stamp (docs/LAB.md uncommitted)
-> W1.00 refused on it; W1.01 refused on stale W1.02. A foreground clean W0.DIAG re-run hung in its hash-salt child (1 s CPU in 10 min; killed).
11:49 LAUNCHED detached pid 3070760, log /data/tmp/w1_00_attempt3.log: W0.DIAG && W1.00 (attempt 2); W1.02 && W1.01 (re-buys), clean tree.
Next: harvest; then repair W1's energy clock by a race (as J6(b)), never by shortening the horizon; re-run W1.04; register DP.05-in-W1.

### J8 — The hand-over gate: a Jack trained here still learns on another machine
**Why.** Owner 11-13: the finished Jack is "trained here, handed to people,
still learning on their machines"; END-FIT (top of `CHAMPIONS.md`) lists the
hand-over gate as "none registered". Resume is proven bit-exact on THIS box
only (`LF.02`). No winner may be adopted that closes this row, so the row
needs a test that can fail.
**Do.** Register ONE spec (next free `LF.` id), exempt from repair-before-
register by this order: train the seated core here for N lives, save one
checkpoint (weights + diary + ring memory + optimiser state); load it on a
different machine (a Kaggle CPU kernel through S1 — other OS image, other
CPU). Claim, ≥ 3 seeds: (a) the first frozen exam there matches the last one
here within the seed noise `T1.14` publishes; (b) after M more lives there he
beats his own frozen hand-over twin. Controls that must fail: a checkpoint
with the diary and optimiser stripped must fail (a) or (b); a fresh-weights
twin must fail (a). Size N, M from `LF.03`/`LF.04`'s rows; bars declared
before any run. Then, same commit or next: change `cores.py:75`'s comment —
the `language` slot is "words through his own learned ear; EMPTY today", not
"the frozen text tower" (card line 2; no dimension changes).
serves: M7, M8 · END-FIT row 2 · philosophy 1, 2, 7.
**done when:** the spec is registered with both controls and queued on S1
behind `LF.04`; `cores.py` no longer names a frozen tower inside him.
state: 11-13 12:30 REGISTERED LF.06 (not LF.05: refused stub id) in registry_expansion.py, both controls (stripped ckpt must fail a|b,
fresh weights must fail a), N_HERE=10, M_THERE=10, seeds 110-112, MARGIN_S=LF.04's 272.812, TOL_S from a seed-90 exam pilot (T1.14's
noise is loss units, not sim-s — rule in notes). Budget CPU_DAYS until S1's kaggle-cpu class; queued behind LF.04. cores.py:75 comment
NOT edited: 8 certs hash cores.py raw (LC.02/LC.03/LF.03/SH.01...) — INBOX. Next: module tests/lf_06_*.py (exam + ckpt strip), seed-90 exam pilot -> TOL_S.

### J5 — The curriculum cook-off (`CR.01`)
**Why.** Owner 11-11: the curriculum is "as important as jack himself". The
research page and the VACANT seat exist; six arms are built and sized at
~39 CPU-h — too big for this box.
**Do.** Register `CR.01` now exactly as sized (null, hand-staged, shuffled
control, learning-progress, frontier, matched-meal control; 3 claim seeds ≠
88/89; margin .20 fed-fraction, N = 47 exam lives per arm-seed), budget class
= the Kaggle-CPU class S1 creates. It runs second on S1, after `LF.04`. The
teacher-proposed arm joins later, behind the judge gate. If J6 ships the
k = 72 needs clock first, re-pilot the null's spread before the claim seeds.
serves: M2 · philosophy 3, 5.
**done when:** `CR.01` is registered with both controls and has a verdict.
state: (1)+(2) DONE 11-13 (CURRICULUM.md, seat VACANT). CR.00 pilot s89: exam fed null 5/8, staged 2/8, shuffled 0/9 (anecdote).
11-13 05:35 machine arms BUILT in experiments/cr00_pilot.py (lp, frontier, matched); sizing 2.8M decisions ~39 CPU-h.

### J3 — The learning core at 3× scale (owner priority 11-12)
**Why.** Owner: "when will we run it bigger scale against the others". The
Learning-core seat rests on a 1× screen; `LC.07` (~10×) is priced at
200–270 Kaggle-CPU session-hours and stays untouched.
**Do.** Register ONE new spec (next free `LC.` id), its own claim, not a
shrunk `LC.07`: all five core arms plus the ring working memory, ~3× the
screen's experience, ≥ 3 seeds, the collapse diagnostic of
`docs/research/LEARNING_CORE.md` §5.4 implemented, weights persisted, a
control that must fail. Pilot one arm-seed on Kaggle CPU to price it; write
sessions needed in `state:`. Runs third on S1.
serves: M3 · END-FIT row 2 (the collapse guard is this spec's §5.4 diagnostic; state per arm its weight count —
it must stay small enough to learn in real time on a personal computer) · philosophy 3, 7.
**done when:** the spec is registered, priced from a pilot, and dispatched.
state: open — waits on S1's route for its pilot; registration can start now.

**Next up, in this order, as slots free:**
1. **The diary in his decisions** (opens when `LF.04` has a verdict): core
   with retrieved diary rows in its input vs without; wiped-diary twin must lose (M7).
2. **Inside/around sweep** (ruling 11-11): re-credit `LG.00`, `HR.4`, `T3.08`,
   `LG.10/12/13/14`, `ME.11.C/D/E` as parent-rig results; delete the paid
   `AudioListener._transcribe_api`; re-run `LG.00` with the learned core in the loop.
3. **Ring memory in the creature**: `ME.8`/`LF.02` with the seated ring in place of the GRU.
4. **Learned ear** vs a no-ASR null (M5/M6); `HR.1`, `HR.5` repaired under it.
5. **Judge admission gate** (FIELD_WATCH N1) — opens with J5's teacher arm.
6. **END-FIT's other unregistered gates**, one order each as slots free (ROADMAP M2/M4/M6): W2 — the tool chain
   tree→wood→fire→cooked food and a buildable shelter (build W1's fire and parts first, `W.5`); the grip-rule
   cook-off (hands that hold, carry, strike); the advice gate (told a step he does it sooner than an untold twin;
   a lying adviser loses him); the size cook-off with a real-time-on-a-personal-computer cap.
7. **`UB.11`–`UB.13`, `TA.03`** on the declared trunk A0 (ruling 5 today) — GPU, after `D1.0`.
8. **Sense added later vs from birth** (M5); `XL.01` at power; `PS.02` (stale 92 d, frees 3).

---

## SYSTEM — work on the machine (max 3 open)

### S1 — Free CPU off this box: Kaggle CPU kernels, `LF.04` first
**Why.** Measured 11-12: a CPU-only Kaggle kernel (`enable_gpu:false`) is
accepted with the GPU quota spent; 2 learners per session at 34 dec/s each
(this box: 20.5). The creature gate, the curriculum race and the scale race
are all refused on this box (D20/D39) — this is their only route.
**Do.** In `experiments/gpu.py` + the budget table only: a `kaggle-cpu`
budget class that submits the existing Kaggle job with `enable_gpu:false`,
`pip install mujoco`, the https repo URL, ≤ 2 worker processes, 12 h session
cap, charges NO GPU-hours and NO box CPU-seconds, and lands the ledger row
through the same `JACKRESULT`/`JACKFILE` path S5 built. Resume across
sessions only if `LF.04` does not fit one (it should: ~4 h). Move `LF.04` and
`BD.01` to the class (budget only — claim, seeds, gates, controls byte-same)
and dispatch `LF.04`. Extend `T0.10`/`T0.12` with a CPU-kernel round-trip
that must show 0 GPU-h charged. Net growth ≤ ~80 lines.
serves: M1→M7, M2, M3 (J1, J5, J3) · philosophy 5, 6 (free compute only).
**done when:** `LF.04` has a ledger verdict that came from a Kaggle CPU kernel.
state: pilot 11-12 (owner): 1 proc 28 dec/s, 2 procs 68 total, 4 no gain.
11-13 09:10 step 1 DONE (06b87bd): `submit(prefer="kaggle-cpu")` pushes enable_gpu:false, no accelerator, no torch fix, never
fails over, charges 0 GPU-h; T0.12 PASS with property 15 (control = GPU route fails both). Live: jack-cpu-1794560915 accepted
with GPU quota spent, 47 s, artifact back, 0.0 GPU-h. T0.10 left GPU-only (no GPU quota to re-buy it). Next: `kaggle-cpu` Budget
class in protocol.py + LF.04's job body (pip mujoco, ≤2 procs, JACKRESULT) and dispatch LF.04. Unmeasured: parallel CPU sessions.

### S3 — A verdict the record cannot re-derive; guards red by design
**Why.** `run verify` exits 2: `W0.PL`'s gate judges from process memory
(`_SEEDS`/`_CTRL`), not the row, and the row stores the MEAN of per-seed
minima while the gate is on the MINIMUM. `BD.01` has the same shape and has no
row yet. `coverage` exits 2 on a by-design counter; `T0.28` p10 reads an empty
decision register as a broken one.
**Do.** (1) `W0.PL` and `BD.01`: record the per-seed `_min`/`_max` the gate
uses and make `_check(m, c)` read only its arguments; re-run `W0.PL` (bars
unmoved). Add to `T0.13` (or `verify`) a check that fails any spec whose gate
reads module state. (2) `coverage`: by-design-red counters out of the exit
code, still printed; FAIL-UNOWNED reads this file. (3) `T0.28` p10 tells empty
from unreadable; a planted unreadable page still fails. (4) After J1's
quiet-box `LC.02` verdict, bank the then-live `pass_on_dead_dependency` and
`unreachable` with `ratchets record --only` — not before (ruling 4).
(5) `coverage.py` COMMITMENTS gains one line for each of the four ends in
GOAL.md's closing section (owner 11-13), each naming its END-FIT gate.
Nothing guarding a threshold, control or seed count leaves an exit code.
serves: philosophy 4, 5 (a truthful machine).
**done when:** on a clean tree `run verify` and `coverage` exit 0 and each
still exits non-zero on a planted PASS-on-FAIL-dependency.
state: 11-12 `ratchets record --only` DONE; T0.18 KeyError DONE (754b6b0), verify went 0 then red again 4 h later on W0.PL.
Open: all four items above.

### S6 — A video the owner can watch (owner request 11-12)
**Why.** Owner: "a video he can watch". Card line 1: he can be known.
**Do.** A script (`scripts/life_video.py`) that renders OFFLINE from the most
recent recorded life — never inside a scored run — with `experiments/render.py`
(Xvfb + llvmpipe): third-person view of the body in his world, hunger / thirst
/ temperature overlaid, phone-playable H.264 mp4 ≤ 20 MB, written to
`/data/jack-video/latest.mp4` (+ a line saying which life, seed and commit).
The most recent life, whatever it shows — no picking. Run by hand at nice 19
after a life; no cron, no service. Name the path in the order's `state:`.
serves: M8 (he can be watched) · philosophy 1.
**done when:** the mp4 exists, plays, and is of the latest recorded life.
state: open — not started.

**Next up:** (1) **S4** — six standing PASSes gated on a seed AVERAGE that
admits a failing seed (`ME.10`, `NE.00`, `LT.01`, `PG.4`, `PS.02`, `T2.20`):
gate per seed, bars unmoved, start with `ME.10`. (2) `T0.13`'s 13 gate clauses
that can never fire. (3) `run status`'s STEERING reader parses 0 items — aim
it at this file or delete it. (4) `T0.33` `slack_predicts_refusal` reads False. (5) the dispatch
BUDGET line prints the shared Kaggle account's quota as 'remaining 30 h' when Kaggle refuses — print UNKNOWN unless probed. (6) D52 fired:
`field_watch.sh` retries within its ISO week on a gate refusal, one completed sweep per week at most.

---

## INBOX — one line each, workers add, the Director empties

- (JACK 11-13 12:30) J8's cores.py:75 comment fix (`language` = 'his own learned ear; EMPTY today') NOT made: impl_sha hashes cores.py raw bytes for 8 specs (LC.01/02/03/07/08, LF.03, SH.01/02), so a comment edit stales LC.02 (needs a quiet box) and LC.03 (~90 core-h) for no code change. Ruling wanted: batch it with the next real cores.py change, or make IMPL_DEPS hashing comment-blind (SYSTEM).
- (JACK 11-13 11:50) W0.DIAG's hash-salt differential child hangs rather than computes: errored TimeoutExpired after ~25 min in the 10:38 chain (spec itself 0.03 s), and a clean re-run's child sat at 1 s CPU for 10 min. Costs every W0.DIAG re-buy ~25 min of lock; also a dirty doc (docs/LAB.md) stamped W0.DIAG dirty and blocked W1.00 even though prose docs are meant to be exempt.
- (LAB 11-13 11:40) Meal memory (salience-kept replay ring, `experiments/meal_memory.py`) pilot ALIVE, thinly: fed .484 vs fifo .370 vs yoked control .376 (s91–93, 9 runs, ~55 CPU-min). Claim gate is declared in the module docstring; the run to grind, once ruling 3 lifts: seeds 94–98 × 3 arms × 13k (`meal_memory run <arm> <seed> 13000`, ~15 runs × 13 min, 2 at a time), then `tally`. Not yet a registered bake-off — Director to say whether it enters under the Consolidation seat (`NE.05`) or the learning core's.
- (JACK 11-13 10:30) LC.02 contention: J1(1)'s PASS branch shipped (e9e07c2, VOID-on-contended-FAIL); the 11-14 ruling below may be moot. Foreign load on a 'quiet' box read up to .75 core in one timed window vs the 1.0 threshold.
- (JACK 11-13 09:45) LC.02: a W0.PL correctness commit (c510b92) silently cost 11% physics throughput; nothing caught it except LC.02 going red a day later under noise. LC.02's T is timed over 40 decisions (5 updates at tr .125), so one stall moves an arm-seed by 2+ sim-s/real-s (dreamer-xs probe 5.31-7.93). For the 11-14 ruling: VOID-above-load, or a longer timed window re-registered.
- (SYSTEM 11-13 09:12) J2: `D1.0` attempt 4 ERROR 09:01 (ledger, committed by SYSTEM): unit 1 ran 1.71 h on Colab then "fetch failed"; unit 2 hit Colab `Service Unavailable`, Kaggle refused (quota). S5 stdout recovery did not save unit 1 — J2 to diagnose. [JACK 10:20: wrong — unit 1 = the SB3 reference, recovered + cached + reused by attempt 5; the ERROR is unit 2's Colab assign.] `W1.04` FAIL 08:48 also committed unread (J7).

## DONE (kept one day)

- **J4 — `UB.10` FAIL (`de563c4`), the one-brain contest's first verdict in 59
  days.** Medians A0 .8719 A1 .8844 A2 .8875 A3 .8844 A4 .8344 (disqualified)
  A5 .8781; A2 beats A0 on every seed but the top arm changes by seed. Fusion
  seat: UNDECIDED, raced, no arm separates; experts+router did not beat one
  brain. Ruling 5: `UB.11`–`13`, `TA.03` run on trunk A0.
- **`WM.02` PASS** (`2103829`): ring memory 0.902 vs GRU 0.579, both controls
  failed; Working-memory seat held BY VERDICT.
- **S5** (`d9c94ba`): a finished Colab run is recovered from stdout when its
  download dies. `T0.12` PASS.
- **Owner's INBOX lines 11-13 (END-FIT, vision)**: placed — J8, Next-up 6,
  S3(5), ROADMAP M2/M3/M4/M6, rulings 6–7.
