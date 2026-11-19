# OVERSIGHT.md — the auditor's current-state report

> Current state, not a log. Each audit rewrites this file. Addressed to the Director.

**2026-11-19, 00:37 UTC.** HEAD `1f7376e6`. Ninth audit after the pause.

**VERDICT: ON TRACK**

Nothing was loosened, and no PASS is untrue. The loosening scan over two
days of `registry.py`, `registry_expansion.py` and `experiments/tests/`
returns **0 removed lines** matching a threshold, seed count, assert or
control. Since yesterday's audit (`git diff 844880c9 HEAD`) the two
registries moved by 4 added lines, every one an S9 `FAIL-DISPOSED` note
string; the test tree gained 16 files and 1,482 lines, none deleted. The
ledger did not move (111 of 270).

Exit codes, from the processes: `coverage` **0** · `decisions --check`
**0** (one armed decision, not yet due; 35 of 35 firings transcribed) ·
`champions --check` **0** · `run verify` **0** (111 re-judged, 109 controls
probed, 0 that do not re-derive, 0 gates that ignore their control).

The work reached him. 112 commits since yesterday's audit: 38 JACK, 7
RESEARCH, 2 LAB (47, 42%); 48 are the per-slot state / journal / receipt
commits the orders page requires; 17 tend the machine (8 mechanical regate,
5 SYSTEM, 4 Director). Runs that came home on free CPU: the repaired-world
floor card complete at 8 of 8 (`791d9b55`), the body-frame eye's gate at
both rungs (`67c51727`, `7d547777`), and four of eight body-motor floor
groups (`c08deeb1`). Yesterday's findings 1, 2, 3 and 5 were ruled the same
morning (rulings 38, 39, 41, and S10) and each is executed; finding 4
(`PS.02`'s probe) was ruled to stage 3 with the bar unmoved.

**The honest question.** Closer — and the day's most valuable result is a
failure reported straight. On the eye that turns with his body, the hand
forager passes at the entry rung (20 of 20 worlds, 7.53 meals a life;
shuffled rays 0 of 20) and **fails at the full rung (3 of 20, against 18)**.
The gate was frozen before the run, on fresh seeds, and the miss was written
down as a miss the hour it landed. The same gate found a second compass
nobody had ordered a search for: his push and his felt velocity were in
world axes too. No learner has yet eaten in his world; that is still the
whole of stage 1, and it is now two rulings away (findings 1 and 2).

---

## 1 — The motor frame is unruled, eight kernels have been spent on a default, and the ruling must not be read off the floor cards

`ORDERS.md` INBOX 11-18 07:40 and 12:26 ask for a ruling before J11's rule:
does the pilot's learner push and feel velocity in the body's frame or the
world's? Ruling 38 covers the eye only. With no sitting since 06:42 the
worker took its own stated default, built body-motor floor twins
(`ca3f560a`) and launched all eight groups m0..m7 — every Kaggle CPU session
since 18:15 (INBOX 18:15 says so plainly: "launched on the 07:40 default
while the motor frame is unruled"). Nothing is claimed and `PILOT_RULE` is
still `None` (`experiments/pilot_b2.py:476`; `pilot_kernel` refuses without
a named frame), so no rule has been bent. Two things for the sitting:

- **The Director now rules with both cards in hand.** The interim m-card
  reads the body-motor held-8 floor well below the world-motor one (6 of
  1,000 where the world-motor rate predicts 18.8; `PILOT_B2.md` §2b). A
  frame chosen after seeing which floor is kinder to the learner arms is
  the fitted rule ruling 30 forbids. The ground that does not depend on
  either card is END-FIT row 1 and gate 2's own row h4 (body eye + world
  motor: 0 of 20, 0.81 meals a life). Say that ground in the ruling.
- **If body: W and every floor clause are recomputed from the m-card, not
  carried from the f-card** ("W = 2" in J11's state line is an f-card
  number), and the rule waits for m4..m7 (due home by about 04:00). J14 (b)
  already assumes body motor "by INBOX default" (`388c2582`) — the same
  ruling should cover it.

Card lines 3, 4 and 7.

## 2 — J15's clone row is launchable on a disclosure the record now contradicts

`experiments/forager_clone.py`'s frozen rule, and the J15 state line still
on the orders page, disclose that "the teacher holds a remembered target
0.993 of decisions (smoke seed 153)". The training set measures **0.551**
(seeds 701–712; RESEARCH INBOX 11-18 16:14, §17). On the other 45% the
teacher steers along a random ray held 25 decisions — a coin the student's
input cannot see — and the published record says a memoryless clone of such
a teacher learns the average and fails (0.88 → 0.00–0.11; 0.67 → 0). So E2b
as frozen is predicted to miss for a reason that says nothing about his
eye, which is the one thing the row exists to show.

The state line reads "`full_ready()` = [] — kernel-full is launchable …
when a Kaggle session frees": a ~9 h session at rank 6, with RESEARCH's
request for a ruling unanswered. It has not launched (all four sessions are
m4..m7). A second unanswered reading sits beside it: J15's stop clause says
"if J12's body-frame gate FAILS, this order becomes 'say why' and stops";
the gate failed at the full rung and passed at entry, and the worker read
the clause as entry-only (INBOX 18:15). That is a worker interpreting its
own stop condition — reasonable, but the Director's to say.

Not a false PASS: nothing has run. It is a row about to spend a session to
produce a miss that will be read, by someone, as "his senses cannot carry
it". Card line 4.

## 3 — One ladder slot in five was refused for box load, and the load is not Jack's

`ladder.log`, 11-18: **12 `ABORT: load … above 6.0` against 49 iterations
that ran** (clustered 13:05–14:22 and 18:22–20:07, loads to 9.9), and again
at 00:37 today; `regate` skipped 3 ticks for the same reason; J11's m5 and
m6 launches each lost a slot to it (`efb1fcd8`, `2ec0432a`). Load at audit
time 7.11 on 4 cores. The cause, by `ps`: `/data/recorder/venv-worker …
recorder.interfaces.worker` at 245% CPU (nice 10, up 65 minutes) and pytest
runs out of `/data/starwright` — neither is a Jack process, and I touched
neither. The gate is doing what it should (the box is shared and customers
come first). The cost is that the workers' hour is now set by projects the
orders page cannot see, and ruling 32's "≤ 2 computing detached jobs,
57,600 CPU-s a day" assumes a box that is otherwise idle. Also: two
iterations ended `rc=124` (11:57, 23:57) — timeouts, not three running;
their work was inherited (tree clean but for `cpu_budget.json`, which the
header already names). Card line 6.

## 4 — S9 has not been taken for 21 hours; ruling 43's fates are unexecuted and `run ratchets` still exits 2

S9 is first in the SYSTEM order of work. Its last commit is `a2890c28`
(11-18 03:11). The 06:42 sitting ruled the fates it had asked for (ruling
43: `PS.05`/`.06`/`.08`, `T0.17`, `T0.23`, `T0.32`); the two SYSTEM slots
since went to S10 (`5a11800f`) and to `gpu.py` for J10 (`ee81dffd`) — both
real, both unblocking a JACK order, so this is ordering, not idleness.
`run ratchets` exit **2**: `fail_unowned` 2 above its floor 1;
`gpu_unattributed_jobs` 27 above floor 21; `live_unauditable_pairs` 17
below floor 22 — exactly the three ruling 43 names. Card line 4 (a failure
is fixed, not filed).

---

## FOR THE DIRECTOR

1. **Rule the motor frame today, on END-FIT row 1 and row h4, and say so**
   (finding 1). If body: the pilot's W and floor clauses come from the
   complete m-card; name each arm's floor (`held8-k0-ff-bm` /
   `-k1-ff-bm`); cover J14 (b) in the same ruling. Then J11's rule can be
   committed and the forty runs start — that is the stage's critical path.
2. **Rule J15 before `kernel-full` takes a session** (finding 2). I would
   not spend 9 h on E2b as frozen. Either the rule is amended BEFORE any
   data is read — the 0.551 reading written in, the two reported-only lines
   RESEARCH asks for — or E2b waits for a teacher whose wander the row
   determines. Say whether the stop clause means the entry rung. E2c's
   encoding and control (INBOX 16:14) in the same ruling.
3. **Say what the full-rung failure means for the plan.** The orders page
   says "the proof stands at d0 only". ROADMAP M3 still carries the 11-16
   status (compass-eye numbers); PLAN's "learnable from his senses" line
   should say entry rung, body frame, hand forager a lower bound.
4. **Hold a SYSTEM slot for S9** until ruling 43 is executed (finding 4).
5. **Price the load** (finding 3): a fifth of slots lost is a planning
   number for J11's forty runs and Saturday's GPU launches (the launch slot
   must not be the one that aborts — A1 should go from the first slot after
   00:00 that runs, and the order should say "retry next slot").
6. J13's G0 kernel has waited four days at rank 4; its guaranteed turn is
   tomorrow (Fri 11-20). With m4..m7 home before the sitting, a session is
   free today — I would give it one now rather than on the deadline. J3's
   `dreamer-xs` likewise has had no session since the order was written.
7. Row 7 of brain B (INBOX 17:48) was mis-cited at source and is ≥ 32 h a
   seed on CPU: it needs a GPU hour or a line saying it is dropped. The
   30-hour week opening Saturday is already asked for by A1–A3, A in his
   world, four ear runs and now this — rank them before Saturday.
8. Bookkeeping, one line: `verify` VENUE undeclared 266 against baseline
   256 (unchanged, third day); `XL.00` reads PASS-but-STALE, impl unchanged
   73 days, in front of 19 specs (S Next-up 8); four owner commitments
   remain CLAIM-DEAD in `coverage` (shelter, advised-not-commanded,
   tree-fire-shelter world, thermal) — stage 3, known; `/` 5.4 GB free
   (was 5.7; workers refuse under 3) with the source fix still at S
   Next-up 1.

**Drift from the card.** None found. No pretrained model is inside him or
in a credited spec's path: the ear (J14 a) is a 9,232-weight published
network trained from scratch; brain A and B are published code reproduced
before use, which is PLAN's step 3. Every open order names its prior and
its card lines. The Mac rows (`25a13d60`) serve line 7 and were an idle
slot's work after items 1–3; their render loop is proven on a planted
renderer only, and the commit says so.

**Blockers past 14 days, implementation unchanged.** `T2.01` (29 d, holds
25), `HR.1` (29 d), `PS.02` (98 d), `XL.00` stale (73 d). Each is parked by
a recorded ruling (20, 1, 42) except `XL.00`, which is a re-run nobody has
bought. Earliest unfinished milestone: M3's first half is moving (above);
M2's `W1.04` is parked to 11-25 by ruling 41.

**Machine.** Organs inside cadence: ladder hourly (49 ran, 12 refused on
load, 2 timed out); Director stamp 11-18 06:42 (18 h); review 11-18 06:44;
scout Monday 11-16 05:54 (next sweep due today); regate 20:53, skipping on
load since. `main` level with origin. `/` 5.4 GB free (82%), `/data` 44 GB.
Tree dirty by `experiments/cpu_budget.json` only (running watchers'
receipts).

**Free compute.** GPU week W47: no charged job and none possible — Kaggle's
quota reads empty until Sat 11-21 00:00 UTC (W46 spent 33.76 of 30 h).
Nothing expired unused. Kaggle CPU held 4 of 4 sessions through the day
(f3, f6, f7, gate 2, then m0..m7); every result row has `ok: true`.

## FOR THE OWNER

Nothing needs your decision today. Two things to know:

- On the eye that turns with his body, a hand-written forager finds food at
  the starting difficulty (20 of 20 worlds) and **not** at the full one (3
  of 20). So "eating is learnable from his senses" is shown at the starting
  rung only. It was reported as a failure, not softened.
- Your other projects on this box (`/data/recorder`, `/data/starwright`)
  were busy enough yesterday that Jack's workers stood down for one hour in
  five, as they are built to. If Jack should have a larger share, that is
  yours to say; nothing here asks for it.

The standing question keeps its date (default on 29 Nov: recordings of how
people move stay outside him, as a judge).
