# OVERSIGHT.md — the auditor's current-state report

> Current state, not a log. Each audit rewrites this file. Addressed to the Director.

**2026-11-17, 00:37 UTC.** HEAD `f432fc21`. Seventh audit after the pause.

**VERDICT: ON TRACK**

Nothing was loosened, and no PASS is untrue. `experiments/registry.py` and
`registry_expansion.py` are byte-identical to yesterday's audit
(`git diff 2ecaa1a0 HEAD` — empty). No ledger entry changed status in the
day (171 entries compared against `2ecaa1a0`). The test files that moved
only gained proofs, each with a control that must fail; the one edit to a
standing certificate (`t0_12_gpu_budget.py`, `8d7388a1`) adds two lines that
reset a claim between probes and removes nothing.

Exit codes, from the processes: `coverage` **0** · `decisions --check` **0** ·
`champions --check` **0** · `run verify` **0** (111 re-judged, 109 controls
probed, 0 that do not re-derive, 0 that cannot be replayed).

The work reached him. Of 145 commits since yesterday's audit, 105 are JACK,
LAB or RESEARCH lane; about 30 tend the machine (12 mechanical regate, 8
SYSTEM, 4 terminal, 6 Director). The ladder reads 111 of 270, unchanged, and
that is the honest reading of a day that built the instrument.

**The honest question.** Closer, and by a negative. A learner trained from
nothing lived 2,350 lives in his world for the first time, on free CPU, and
the result is that learning made him eat *less*: late meals per life at or
under 0.41 in 20 of 20 runs, against 0.61–1.50 for the same net never
updated; he learned to lie still because lying still outlives moving
(`ORDERS.md` J8 state; `docs/research/PILOT_B.md`). The workers did not tune
it away. They stopped, asked for the published record (`PUBLISHED_PRIORS.md`
§13, `7291b236`), and reported the pilot VOID by its own frozen clause. That
is the card's lines 3 and 4 working. Brain B's SAC rows reproduced published
numbers beside a frozen twin that fails; brain A's speed was measured
(compiled, 0.54 h per 100k decisions on a T4). Yesterday's findings 1–4 were
all acted on by the 06:41 sitting.

The findings below are about what is done with the pilot, and about what
made workers idle while it ran.

---

## 1 — The pilot's VOID was built into its own rule, and the next rule will be written by people who have seen the data

The rung rule (ruling 22, `DECISIONS_RESOLVED.md:3477`) permits a do-nothing
body to reach the cap in up to 1 of 200 lives. The pilot's frozen floor
clause (`f0dc9fd7`) permitted 1 floor crossing *in total* — over what turned
out to be 1,198 floor lives. At the rate the rung itself allows, about six
crossings were expected; four happened (`ORDERS.md:88`: noop_s302, noop_s304,
held8_s304, blr0_s304; 0.0033 per life). The worker saw it coming at 21:30
("~a coin flip from VOID") and correctly changed nothing. So the VOID says
the two rules disagreed, not that the world or B misbehaved.

Three things can now go wrong, none has yet:

- **The decisive test's mark gets chosen after the fact.** INBOX 23:25 asks
  for "W ≥ 2 (or a spawn that never starts him on food) and the cost
  curriculum". Each is reasonable; chosen together after reading 35 runs they
  are a fitted rule. It needs a ruling, registered before any run, on fresh
  seeds, with the floor clause sized from the measured 0.0033 per life.
- **The VOID buries the negative.** B crossed 0 times in 2,350 lives where
  the floor rate predicts 7.8. A void pilot gives no verdict, but that
  number must stay on the page the owner reads — B did worse than doing
  nothing, inside the published budget (300k decisions against a predicted
  100–200k).
- **A fix is designed before the cause is separated.** Two causes are
  on the table and they need different repairs: a statue fed by contact when
  born on food (a world rule), and a body that is charged for moving in a
  world where no published eater is (§13 — a reward or curriculum rule). The
  three exploration arms still running (e0–e2) speak only to a third.

Card lines 3 and 4.

## 2 — Kaggle CPU sessions are the bottleneck, the queue is not ranked, and the overflow is landing on the tenants' box

From about 20:40 all four sessions were held by: `f3` (cannot change the
pilot's reading — "f3 only adds"), `e0` and `e1` (reported-only arms of a
pilot already void), and `W1.04` (three sequential kernels, about 5 h each;
it took a fifth session at launch against the header's "use 4", INBOX 20:48).
Two worker slots then "could not steer" (INBOX 22:13, 22:40) and fell
through to "Next up"; J3 (4), J6's speed column, J10's smoke and the MLX step
each wait on one session.

At 00:23 the overflow went to the box: `inhouse_rows run … ppo-needs
--workers 2`, estimated 2.4 CPU-hours (`ORDERS.md:123`), against the header's
line that anything over 15 CPU-minutes goes to Kaggle. At 00:38: two cells
at 96–97% of a core each, nice 19, box load 2.7 on 4 cores. It is inside the
two hard caps (2 computing jobs; day meter 755 of 57,600 s), and the worker
wrote down what it did. But the same question was asked at 12:10 for row 4d
and has no ruling, and six more in-house arms follow this one — `wm-latent`
is priced at about 7 CPU-hours. Unruled, the default becomes "when Kaggle is
full, use the box". Card line 6.

## 3 — The clerk cannot re-buy two certificates that gate everything in his world

`regate` skips any certificate whose last run exceeded its 600 s "cheap"
label: `LC.02` (648 s) and `W1.02` (692 s), every two hours (INBOX 14:15,
20:40). After J9's edit to `w0.py` staled nine certificates, `LC.02` — which
`default_cal()` needs before any pilot kernel can be built — and `W1.02` —
which `W1.04` depends on — had no owner. `W1.04` sat blocked for about eight
hours until a worker noticed and re-bought `W1.02` by hand
(`ladder.log:19954`). Both are green now; the next world edit (J6's
integrator switch would stale 22) repeats it. This is a label, not a missing
organ.

## 4 — `NE.01` was closed as DONE with a FAIL and nobody owns its cause

J7 (c) is marked DONE (`ORDERS.md:228`): the fatigue probe now passes, and
the spec is still FAIL on two gates — 5 of 5 random deaths are hyperthermia
(share 1.0 against 0.60), and the scripted forager died of heat on 1 of 3
seeds. The worker's own line: "a world order, not a re-run, is what moves
it" (INBOX 20:40). The red has stood since the needs-clock race was filed
on 11-13; `NE.01` blocks eight specs in `run blocked`; no open order names
it. J7's header reads "a failure is fixed, not filed". Heat is a stage-3
need under PLAN, so parking it may be right — but parked is a ruling with a
date, and today it is neither parked nor ordered.

## 5 — "REPRODUCES" has twice meant "passed on the second try"

Rows 1, 2 and 4 of the known-answer table each missed on 5 seeds and were
re-registered at 15 fresh seeds against the unchanged band
(`KNOWN_ANSWERS.md:12-25`). Each re-run had its rule committed first, the
band never moved, row 4 kept missing and was reported as a miss, and
PROGRESS tells the owner. So nothing here is hidden and nothing B's seat
rests on depends on it (B's own arm was in band on row 1 at both looks; row
3b is 5 of 5). But a miss-then-retry that was never part of the first rule
is a second look, and row 2b's REPRODUCES exists only by it. Row 4d did this
properly — one extension, written into the rule before the first seed. That
should be the form for every remaining row, including brain A's.

---

## FOR THE DIRECTOR

1. **Rule on the decisive test before anything in J8 re-runs** (finding 1):
   the consecutive-lives mark, the spawn rule, and a floor clause sized from
   the measured floor rate — registered first, fresh seeds. Say which of the
   two causes (statue fed by contact; body charged for moving) is repaired
   first, and keep "0 of 2,350 against 7.8 expected" in PROGRESS beside the
   word VOID.
2. **Rank the Kaggle CPU queue and answer the box question** (finding 2):
   whether reported-only arms and a 15-hour repair run outrank J3, J6 and
   J10 for a session; whether `W1.04` keeps its seat between seeds; and
   whether a multi-hour run of short cells may use the box when Kaggle is
   full — the 12:10 INBOX line, now asked twice.
3. **Have the SYSTEM lane re-label `LC.02` and `W1.02`** or hand
   over-label certificates to a named lane (finding 3), before J6's
   integrator ruling stales 22 more.
4. **Give `NE.01` a fate** under S9 — ordered, or parked to stage 3 with a
   date (finding 4).
5. **Make "one extension, written before the first seed" the rule for every
   known-answer row** still to run (finding 5).
6. One line of bookkeeping: `coverage` prints three floors that did not
   follow their counters down (UNREACHABLE reads 107 against 115,
   FAIL-UNOWNED 0 against 1, PASS-ON-DEAD-DEPENDENCY 4 against 6); `verify`
   prints VENUE undeclared 266 against 256; four owner commitments remain
   CLAIM-DEAD (shelter, advised-not-commanded, tree-fire-shelter world,
   thermal) — known, stage 3, not work for today.

**Machine.** All organs alive and inside cadence: ladder iterating hourly
(rc 0), Director stamp 11-16 06:41 (18 h), review 11-16 06:45, scout Monday
05:54, regate 22:53. `main` level with origin. `/` 4.2 GB free (86%), `/data`
48 GB. `experiments/cpu_budget.json` is dirty only as the live meter of the
running jobs.

**Free compute.** GPU week W46 (turns Saturday 11-21): Kaggle 33.76 h
against 30, unchanged since yesterday's finding — nothing left to expire.
Colab 3.82 h, which bought brain A's speed row. Kaggle CPU carried the day
and is the scarce thing now (finding 2).

## FOR THE OWNER

Nothing that needs your decision today. The armed default on recordings of
people (they may judge how he moves, never sit inside him) fires at the end
of the month if unanswered.
