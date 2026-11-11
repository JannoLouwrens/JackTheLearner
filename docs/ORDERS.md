# ORDERS — what the workers do. Written by the Director.
Director sitting: 2026-11-11 21:58 UTC

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5). Machine problems go under INBOX, one
line each. The Director rewrites this file daily; git keeps the history.
Rulings cited here are in `docs/DECISIONS_RESOLVED.md` under the same date.

**GPU routing until S2 lands:** Kaggle gave J1's kernel no session for 70
minutes and billed 1.17 h for it. Send every job of ≤ 2 h to **Colab**; split
longer jobs into ≤ 2 h pieces (one arm × one seed each) rather than wait.
**A `state:` line may say "in flight" only with a ledger row or a live backend
status behind it** — a watcher pid is not evidence.

---

## JACK — build the creature, his world and his curriculum (max 7 open, ranked)

*Order of work: **J3 first, always** — it is the earliest milestone, CPU-only,
and waits on nothing. J1 is a ten-minute harvest when its run lands. Then J2
and J4 (GPU, the week's free hours expire Sunday), J6, J7, J5. A CPU order and
a GPU order may be advanced in the same slot.*

### J3 — One whole life with the learner in the loop (`LF.03`) — the creature gate
**Why.** Nothing has ever run end to end. `LF.01`/`LF.02` already compose world
+ needs + diary + working memory around a *scripted* forager. Put the learner
where the script is. The auditor's test of this whole reorganisation: does
`LF.03` exist by the weekend.
**Do.** Register `LF.03`: the seated core (`wm-latent`, `experiments/cores.py`)
driving the body in lethal W0 across ≥ 5 deaths — needs, every W0 sense live,
diary written and carried across death, save-and-resume once mid-life
(`LF.02`'s mechanism), no crash or NaN, weights measurably changing. The checkpoint must carry the
learner's weights, optimiser and replay (`LF.02` certifies only the scripted
forager's stores). Controls
that must fail, each actually RUN: a frozen twin (weights do not change →
flagged), a wiped diary (nothing crosses death → flagged). An integration
claim, not a performance claim — do not gate on how well he does. Pilot ONE
life first to size it; repair `LF.01` (VOID) only if it is in the path.
serves: M1 · philosophy 1 (the creature), 4 (controls that must fail).
**done when:** `LF.03` is registered with both controls and has a verdict.
state: REGISTERED + implemented at d6c59a9 (`tests/lf_03_learner_lives.py`). Pilot, one
natural life (seed 0, e0=1.0): 503 sim-s ≈ 2,515 decisions, 141 s wall, 860 opt steps.
Smoke: 5-store resume (learner weights+Adam+replay) bit-exact over 50 decisions.
Registered run launched detached 22:14 UTC, AWAITING LF.03 (pid 1793344, log
/data/jack-logs/lf03_run1.log), ~60–75 min. Next: harvest the ledger row; a FAIL is fixed.

### J1 — Harvest `T1.14`, then give it a control that is run, not described
**Why.** `T1.14` (seed noise is measured; SNR ≥ 3) now carries 45 specs and
two standing PASSes (`T2.03`, `T2.14`). Its run failed over to Colab at 21:40
(`ladder-1794433248`). Its "seed-ignoring arm shows zero spread" control is
only a property of its own three arms (`t1_14_*.py:68-72`) — never demonstrated.
**Do.** (1) Read the `T1.14` LEDGER row (not this file). No row and no live
Colab session → re-dispatch to Colab. (2) Add a real seed-ignoring arm (same
seed fed to all three runs → spread must read ~0, and the SNR gate must then
FAIL on it); re-run once. A strengthening; the SNR bar is unmoved. (3) Fix the
stale comments in `t2_14_imitation_mocap.py:6,170-173` (still name `T1.08`).
(4) **Do not `ratchets record` `pass_on_dead_dependency` 7 → 5 until `T1.14`
has a PASS row.** If it does not pass, write here that `T2.03`/`T2.14` stand on
an unproven floor.
serves: M3/M4 (unblocks `D1.1`, `T2.01`) · philosophy 4.
**done when:** `T1.14` has a verdict from a run that includes the seed-ignoring
arm; `T1.08` still FAIL with 7.0 unmoved.
state: steps registry+re-parent DONE at 4818a2c. Kaggle attempt died "queued"
after 70 min (no session ever existed); Colab attempt started 21:40, pid
1755193. No ledger row as of 21:58.

### J2 — Settle who controls the body (`D1.1`), then re-run "he can move" (`T2.01`)
**Why.** `D1.0` spent 17.6 GPU-hours: small plastic head **506**, MLP 415, the
two 57M-trunk arms 344–350, SB3 reference 518 — VOID only because the null
(random actions) was wrong: an untrained net stands still and out-scores
flailing. `T2.01` has been FAIL since Aug 12 running the losing arm.
**Do.** Register `D1.1` NOW (registration needs no GPU and not J1): same four
arms, same budget, learning gate = trained vs **its own untrained twin**,
paired, ≥ 3σ; margin rule unchanged. Build it as independent ≤ 2 h jobs (arm ×
seed) so Colab can carry it while Kaggle is unexplained. Dispatch when `T1.14`
has a PASS row. Seat the winner in `docs/CHAMPIONS.md` BY VERDICT, losers kept
as challengers. Then run `T2.01` under the seated controller, bars unmoved.
serves: M3, M4 · philosophy 3 (raced, not assumed), standing objective 3 (GPU).
**done when:** `D1.1` has a verdict, the control seat names its holder by
verdict, `T2.01` has an attempt dated after it.
state: open — registration unblocked today; dispatch waits on `T1.14` PASS.

### J4 — Run the one-brain contest (`UB.10`) on the idle GPU
**Why.** The fusion bakeoff — six arms including A5, experts + router ("if A5
wins, one brain is the wrong shape") — has sat VOID for 59 days. `gpu<2h`,
frees 4 specs. It is how "how unified is the brain" stays raced, not assumed.
**Do.** Read the VOID's reason in the ledger row; repair the apparatus (not
the bar); pilot on CPU; dispatch to **Colab**. Does not depend on `T1.14`.
serves: M3 · philosophy 3 · standing objective 3.
**done when:** `UB.10` has a WINNER/TIE verdict, or a VOID with a different,
recorded cause and the next repair named.
state: open — not started.

### J6 — The W0 world rules that are red: `NE.01`, `W.1`, `W.2`
**Why.** `NE.01` (needs are a real control problem) fails 2 of ~12 clauses —
`ok_night_shelter`, `ok_spread` — and frees 7 specs, the most of any blocker.
`W.1` (temperature obeys the published heat balance) and `W.2` (needs are a
conserved ledger that can kill) have been FAIL 73 days with no owner; M2's
done-when names them, and the shelter clause and the heat balance are very
likely one fault.
**Do.** For each failing clause find the cause: world calibration (shelter
thermal balance, need spread across deaths, a leak in the ledger) or a clause
that cannot fire. Fix the world or the apparatus; never the bar. Start with
`W.1` + `ok_night_shelter` together.
serves: M2 · philosophy 1 (a world that can kill him, honestly), 4.
**done when:** `NE.01`, `W.1`, `W.2` each have a new attempt with every
formerly-red clause explained in the commit. (D53, fired today, files `SO.07`'s
VOID here too: the body cannot reach privileged food on 2 of 3 recording worlds.)
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
progress vs teacher-proposed — same core, world, lives; metric = survival gain
at matched experience; control: a shuffled-stage curriculum must not beat the
staged one. Pilot before sizing. Arms stage the WORLD and what the adults say —
never his actions.
serves: M2 · philosophy 3, 5.
**done when:** the doc and the seat exist; the cook-off is registered with its
control and has a pilot number.
state: open — not started.

**Next up, in this order, as slots free:**
1. **Inside/around sweep** — execute the 2026-11-11 Director ruling: re-credit
   `LG.00`, `HR.4`, `T3.08`, `LG.10/12/13/14`, `ME.11.C/D/E` as *parent-rig*
   results (not Jack's capability) in coverage and CHAMPIONS; delete the paid
   `AudioListener._transcribe_api`; strike `PG.6`'s "frozen tower" kill-text;
   after `LF.03`, re-run `LG.00` with the learned core in the loop (GOAL.md's
   "strip the core → answers collapse" half has never been tested).
2. **Learned ear** — a plastic speech ear vs a no-ASR null on a small
   vocabulary, `HR.2`'s winner as scored-but-ineligible reference (M5/M6).
3. **Sense added later vs from birth** — the owner's cook-off (M5), judged by
   the hide-it / fake-sense / combination tests.
4. `XL.01` at proper power (bar unmoved); `ME.11.B/F`; `LC.07` at scale via S1.

---

## SYSTEM — work on the machine (max 3 open)

### S2 — The GPU poller bills for kernels that never ran, and hides why
**Why.** `experiments/gpu.py:994-1020`: the poll reads only stdout; Kaggle's
"404 no such kernel" goes to stderr, so a kernel with no session polls as
"queued" until the full timeout and is charged for all of it (1.17 h today).
The push's own output is dropped when rc = 0, so the cause was never written.
The same Kaggle account also runs another project's `next-l1-icsi-*` kernels.
**Do.** (1) Treat rc≠0 / 404 from the status call as failure after ≤ 5 minutes
and fail over. (2) Always log the push's stdout+stderr. (3) Charge 0 for a
kernel that never had a session; re-label today's 1.17 h through the existing
charge path with the reason. (4) Push one trivial kernel and write down what
Kaggle says — quota, concurrency, or something else.
serves: M3/M4 (every GPU order) · philosophy 5 (cheaper, truer), 6.
**done when:** a test feeds the poller a 404 and sees failover inside 5 min
with zero charge; the cause of today's no-session is written in `state:`.
state: open — not started.

### S4 — Six standing PASSes are gated on a seed AVERAGE that admits a failing seed
**Why.** `run worst-seed-audit` (reporting-only, read by nobody for weeks):
`ME.10` (diary vs skill — GOAL.md's "memory makes it him"), `NE.00`, `LT.01`,
`PG.4`, `PS.02`, `T2.20` each pass a bar on the mean over seeds while the
recorded mean and spread allow one seed on the failing side (`ME.10`
`skill_gain`: mean 0.370, a seed may sit at 0.237 against a bar of 0.25). "At
least 3 seeds" means each seed, not their average. Ordered 2026-09-30, never done.
**Do.** For each of the six: gate the flagged key per seed (record `_min` /
`_max` folds and gate on them; bars byte-unmoved), check `depends_on` first,
re-run. Start with `ME.10` (CPU, minutes). A spec that turns red stays red:
write its id under INBOX and the Director orders the repair.
serves: philosophy 4 (several seeds, nothing faked) · M7, M2 (their roots).
**done when:** the audit's LANE B lists none of the six, each with a verdict
from a per-seed gate.
state: open — not started.

### S3 — Guards that are red by design teach everyone to ignore red
**Why.** `coverage` exits 2 on a shrink-only UNREACHABLE counter that moves
every time an order registers a spec; its FAIL-UNOWNED check names the closed
REVIEW_QUEUE as the repair; `run verify` exits 2 on a venue baseline and the
standing `T0.18` KeyError. Prompts now say "ignore these" — a red nobody reads.
**Do.** Move those counters out of the exit code (still printed). Point
FAIL-UNOWNED at `docs/ORDERS.md` (owned = named in an open order). Fix the
`T0.18` KeyError. `run status`'s STEERING-PAGE reader parses 0 items from 3
pages — aim it at this file's orders or delete it. Nothing that guards a threshold, a control or a seed count
leaves the exit code. Smaller, not bigger.
serves: philosophy 4, 5 (a truthful machine).
**done when:** on a clean tree `coverage` and `run verify` exit 0, and each
still exits non-zero when a test plants a PASS-on-FAIL-dependency.
state: open — not started.

**Next up:** (1) **S1 — free CPU off this box**, opens the day S2 closes (same
file, `experiments/gpu.py`): a CPU-only Kaggle kernel mode that charges no GPU
quota; pilot one W0 survival run; write down sim-seconds per real second
against this box and how many sessions run at once (`LC.03` took 2.7 days
here; `LC.07` is priced at 526 h). (2) `T0.13`'s 13 gate clauses that can never fire (`LT.01` 3, `SO.02`
5, `T0.17` 3, `T0.33` 2) — arm or remove with reasons (needs `T0.08` re-bought
clean first; the 22:43 clerk sweep should do that — check `regate.log`).

---

## INBOX — one line each, workers add, the Director empties

- LAB 2026-11-11: run `WM.02` (`python -m experiments.run WM.02`, CPU, ~25 min, 5 seeds; gates in the spec docstring, do not touch them) — ring-memory challenger for the VACANT Working-memory seat; pilot 1.00/1.00/1.00/0.87/1.00 vs GRU 0.55. PASS → seat it BY VERDICT in CHAMPIONS; FAIL → mark it DEAD in `docs/LAB.md`.

## DONE (kept one day)

- **S2-old — megabyte files split** (`f2e23fe`): 4.57 MB → 422 KB, history
  verbatim in `docs/archive/`. Ruling: `DECISIONS_NEEDED.md`'s cap is met when
  it holds only OPEN decisions; the four overdue defaults fired today.
- **J1 steps 1–2** (`4818a2c`): `T1.14` registered, five dependents re-parented.
