# ORDERS — what the workers do. Written by the Director.
Director sitting: 2026-11-13 06:40 UTC

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5). Machine problems go under INBOX, one
line each. The Director rewrites this file daily; git keeps the history.
Rulings cited here are in `docs/DECISIONS_RESOLVED.md` under the same date.

**GPU today.** Kaggle's GPU week is spent by the shared account; Colab assigns a
T4 again (probe with `/data/venvs/colab/bin/colab --auth adc`, never the bare
`colab`). **Colab runs ONE session per account** — `colab ls` first. "In
flight" needs a live pid or backend status behind it.
**CPU today.** The 57,600 s day ceiling is shared by everything. Until `LF.04`
has a verdict, **no new detached learner-side pilot** (ruling 3): the CPU goes
to registered runs (J1, J7) and to J6's world race.

---

## JACK — build the creature, his world and his curriculum (max 7 open, ranked)

*Order of work. CPU: **J1 step 1** (minutes) → **J7** (both runs are ready) →
**J6** → J5 → J3. GPU: **J4** on watcher exit → **J2** the moment Colab is free.
A CPU order and a GPU order may be advanced in the same slot.*

### J1 — Does living teach him? (`LF.04`) — the creature gate, 46 h unrun
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
state: 11-12 LF.04 (7446b86) + BD.01 (c38683a) REGISTERED, NOT RUN (D20 closes cpu<48h; LC.02 red since 11-12 10:52).
Learner gap (LAB): smell, eye height, gamma .99, bout noise, nursery, wean-with-smell all DEAD by declared rules. Nursing
feeds him (18/5/6 vs decoy 0) but he never holds approach past ~1 m: the gap is credit over the walk, not birth distance.

### J4 — The one-brain contest (`UB.10`): attempt 3 is in flight
**Why.** Six fusion arms incl. A5 (experts + router); VOID 59 days until this
morning's venue sweep. Attempt 2 VOIDed on one arm-seed collapse; repaired
without moving a bar.
**Do.** On watcher exit read the row. WINNER/TIE → seat it in `CHAMPIONS.md`.
FAIL (expected: arms within .02, top arm changes by seed) → that IS a result:
write the six medians and "no arm separates from A0 on this venue" in
`CHAMPIONS.md` (seat stays UNDECIDED, raced), list which of `UB.11–13`,
`TA.03` truly need a winner vs only a run, under INBOX. Do not hunt venues.
serves: M3 · philosophy 3.
**done when:** `UB.10` has a non-VOID verdict and the seat page says what it found.
state: **DONE 11-13 06:50 — UB.10 attempt 3 FAIL (de563c4)**, first non-VOID verdict. Rig green; A4 disqualified (s2 collapse, 86d28cc rule).
Medians A0 .8719 A1 .8844 A2 .8875 A3 .8844 A4 .8344 A5 .8781; A2 beats A0 every seed (boot_lo .0052) but tops A3/A2/A1 -> top1_stable 0,
the sole red clause. CHAMPIONS fusion row: UNDECIDED, raced, no arm separates from A0. Per-arm numbers bit-identical to attempt 2.
UB.11-13/TA.03 need-a-winner question -> INBOX.

### J2 — The control-path contest (`D1.0`) — the seat is VACANT and had no order
**Why.** `T2.01` (he moves better than random) is the biggest blocker in the
ladder: frees 25, red since Aug 12, waiting on which controller holds the
seat. `D1.0` attempt 3 computed 1.49 GPU-h and lost it in the download; S5
(`d9c94ba`) now recovers results from stdout, and Colab is back.
**Do.** When `UB.10`'s session ends (`colab ls` empty): dispatch `D1.0` via
`scripts/dispatch.sh` unchanged (13 units of ≤ 1.8 h, resume between units).
Read its log each slot; act only on an exit. On a verdict: seat the winner BY
VERDICT, then re-run `T2.01` under it.
serves: M3, M4 · philosophy 3 · standing objective 3 (spend the free GPU).
**done when:** `D1.0` has a verdict, or an ERROR with a named cause that is
not a lost download.
state: 11-12 07:07 attempt 3 ERROR (lost download, 1.49 GPU-h, 2103829). Not re-dispatched since: backends refused until 11-13 04:25.

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
serves: M2 · philosophy 1 (a world that kills on honest deadlines), 3, 4.
**done when:** `W.1`, `W.2`, `NE.01` each have a new verdict with every
formerly-red clause explained.
state: 11-13 00:45 NE.01 (56c6519) FAIL: ok_night_shelter GREEN 3/3; red ok_spread 1/3, ok_forager 2/3 (sleeper slides into occ~1).
11-13 06:30 occlusion-law race NO WINNER (LAB): shipped 4/5, graded 2/5, damped 2/5 (injury), open control 0/5 as required.
11-13 04:45 W.2: a/b/d GREEN; red = (c) only: starve 1700 s vs 25200, thirst 570 vs 3600; cause needs.py B_E 1/1800, B_W 1/450.

### J7 — The deeper world (W1): run the two repaired specs
**Why.** W0 is too shallow for thinking ahead to pay (`DP.05`); fast/slow
cannot be raced until it does. Both repairs are committed and unrun.
**Do.** `W1.04` (float-clock repair `8495d56`), then `W1.00` attempt 2 (n = 20,
seeds 300–319, ~65 min on 3 workers, `launch_detached.sh`), one at a time.
Then register `DP.05`'s question in W1.
serves: M2 · philosophy 5.
**done when:** both have new verdicts and `DP.05`-in-W1 is registered or run.
state: 11-13 01:45 W1.04 REPAIRED 8495d56 (not run); expect control green + claim FAIL (c).
11-13 02:40 W1.00 RE-REGISTERED attempt 2 (not run): n 3 -> 20 fresh seeds, exact t, power ~.87; no bar/control touched.

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
serves: M3 · philosophy 3 (the seat holds only until something beats it).
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
6. **Sense added later vs from birth** (M5); `XL.01` at power; `PS.02` (stale 92 d, frees 3).

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
state: pilot measured 11-12 07:35 (owner session, kernel jack-cpu-pilot-1794467678): 1 proc 28 dec/s, 2 procs 68 total, 4 procs
no gain. Unmeasured: how many CPU sessions run side by side. No code yet.

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
it at this file or delete it. (4) `T0.33` `slack_predicts_refusal` reads False.

---

## INBOX — one line each, workers add, the Director empties

- **11-13 06:50 JACK J4 -> Director: UB.10 FAIL; what it blocks.** `UB.11`, `UB.12`, `UB.13` have `depends_on=["UB.10"]` and nothing else in the registry names a winner. Each tests a property of *the* fusion trunk (ablation matrix, synergy over the ensemble, retrieval gate), so they need a trunk to RUN on, not a won contest. The trunk can be the incumbent A0 or the median-top A2, declared before the run. `TA.03` depends on `TA.02`+`UB.11` and only needs UB.11 run. Ruling needed: relax the four to 'UB.10 has a non-VOID verdict' with the trunk declared, or hold them until a venue separates the arms. (Not done here: changing depends_on is a registration act.)

## DONE (kept one day)

- **`WM.02` PASS** (was J8, run at `2103829`): ring memory 0.902 vs GRU 0.579,
  both controls failed; Working-memory seat held BY VERDICT, GRU kept as challenger.
- **S5** (`d9c94ba`): a finished Colab run is recovered from stdout when its
  download dies; a busy session is refused by name. `T0.12` PASS.
- **`UB.10` venue sweep** (`3fab32e`): venue (2.5 audio, 0.0 vision) selected
  by the pre-declared rule; attempt 2 VOID on one arm-seed, repaired `86d28cc`.
