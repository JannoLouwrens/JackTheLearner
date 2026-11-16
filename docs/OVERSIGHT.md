# OVERSIGHT.md — the auditor's current-state report

> Current state, not a log. Each audit rewrites this file. Addressed to the Director.

**2026-11-16, 00:37 UTC.** HEAD `e7432148`. Sixth audit after the pause.

**VERDICT: ON TRACK**

Nothing was loosened. I read every threshold, control, seed and dependency
line changed in the registry and the tests over two days. The only bar that
moved is `W1.02`'s planted injection, from +5.0 down to 2.41 (`d1b08807`,
ruling 9): a smaller injection is harder to detect, the 3-sigma bar is
unmoved, so that is a strengthening. `T0.12` gained a clause (Saturday
reset). Seven new `test_jack_env*.py` proofs each carry a control that fails
on today's world. No dependency edge was cut.

Yesterday's two unsupported PASSes are honestly repaired, by new ledger rows
and not by a changed checker: `W0.PL` PASS attempt 4 (11-15 02:49), `T0.33`
PASS attempt 22 (20:43), `T0.34` PASS attempt 17 (22:43; `regate.log:5859`,
`:5905`). `run.py`'s verify code was last touched 11-14.

Exit codes, from the processes: `coverage` **0** · `decisions --check` **0** ·
`champions --check` **0** · `run verify` **0** (111 re-judged, 109 controls
probed, 0 that do not re-derive, 0 that cannot be replayed).

The day's work reached him. Of 197 commits since 11-15 00:00, 110 are JACK or
LAB lane and 33 touch `experiments/*.py`; about 58 tend the machine or the
pages. `J1` — the first measurement of whether anything can live in his
world — ran all its rows on free Kaggle CPU in eight hours, and its rulers
failed their bar twice without the bar moving (6/20, then 13/20 on fresh
seeds, needs 18). That is the card's line 4 working.

The findings below are about what happens next with those numbers.

---

## 1 — Stage 2's written rule now contradicts J1's measurement, and J4 is about to build it

Ruling 19 (`docs/DECISIONS_RESOLVED.md:3400`) fixes stage 2's outcome as
"decisions until W consecutive lives average ≥ 5 meals", with the gate "the
scripted forager must pass, else the comparison is void". J1 has since
measured both halves (`docs/research/WORLD_CARD.md:211-269`):

- **Five meals is a starving life.** Sustaining him takes ≥ 70% of perfect
  harvest — about 37 of row e's 53.5 meals per 1800 s. Ruler `f2` averages
  6.91 meals a life, clears the 5-meal mark, and not one of its 200 lives was
  sustained (26% of the ceiling). A brain can cross stage 2's line and win
  the seat while every life it lives ends in hunger.
- **The gate has no forager that passes it.** The observation-only rulers
  failed (`f` 6/20, `f2` 13/20). Only the privileged row e passes, and e is
  handed food against its torso — it shows food exists, not that his senses
  can find it. Read with `f`, every A-against-B comparison at d = 1 is void
  by the ruling's own text; read with `e`, the gate proves nothing about
  learnability.

No bar has been touched; the rule simply predates the measurement. J4 item 1
("the outcome code of ruling 19") is open and next in line, so the mark will
be frozen into the harness as written unless a ruling lands first. Card
lines 3 and 4.

## 2 — "J1 DONE" does not include the proof it was ordered to give

J1 serves PLAN stage 1 "proof the task can be learned". Its done-when is
"seven rows and three answers", which is met (`ORDERS.md` J1 state, `c65848cd`).
The proof itself is not: no non-privileged policy sustained a life, and the
sensor-only bar failed twice. The state lines say this honestly; ROADMAP M3
still lists "proof the task is learnable from his senses" as a stage-1 item
with nothing recording that the first attempt at d = 1 came back negative.
Two risks: the DONE is read as the proof; or a third ruler `f3` is tuned
until it clears 18/20 (`f2` was already tuned after `f` failed — rule first,
fresh seeds, fair once; a third try is a search). Card line 4.

## 3 — The week's free GPU is gone to an arena closed the same afternoon, and the GPU order has not started

`experiments/gpu_budget.json`: W46 Kaggle **33.76 h against the 30 h
ceiling**, with an overrun row at 11-15 17:26. Nearly all of it is `D1.0`
(dispatches projected at 25.05, 13.4, 10.4, 21.2 and 11.6 h); `T1.14` took
1.8 h and passed. `D1.0`'s standing ledger row is attempt 11, ERROR after
16,744 s — "Untracked experiment code", another session's dirt — at 17:26,
twenty-nine minutes after ruling 20 closed the arena ("no more GPU goes to
the 57M arena"). The week bought cached units and no verdict.

What it cost: `J5`, which the order itself calls "the plan's pacing item",
has 0.88 h of Kaggle GPU until Sat 11-21. Colab is open and the order says
"on a Colab T4 now", but J5's state still reads "nothing run yet". The
19:07 strong-worker slot, ordered to "take J5, then J3", harvested J1 row f
instead (`ladder.log:18747-18776`). J3 and J4 are also unstarted. Eight hours
is not a stall; the rule that was meant to prevent one was not followed at
its first firing. Next strong slot is 07:07.

## 4 — The entry-rung run was launched on a rule its own worker reported as already answered

J2 item 5 asks for `d0`, the hardest dial setting where random eats ≥ 1 meal
in ≥ 50% of lives. The worker measured that this holds at d = 1 (110/200
lives), wrote it to INBOX with a request for a ruling, and then — no sitting
having happened — committed the rule as written (`8af5f2e1`) and pushed four
Kaggle CPU kernels (est. 5 h each, pids 465792–466904 alive, all four pushes
rc=0). It reports a second number, `d_sus`, beside it. That is defensible:
the rule preceded the runs and the data serves either reading. Two things
are not yet safe:

- Read literally the answer is `d0` = today's world, where J1 showed nothing
  lives. If J2 closes on that, "a first rung he can live on" is DONE with no
  rung.
- The d = 0 smoke (`e7432148`): held-8 random on seed 153 ate 254 times and
  still starved at 692 s, because eating while full is wasted. The easy end
  of the dial may not sustain a random life at all, in which case `d_sus`
  does not exist on this dial and the dial needs another turn (meal size,
  or no eating when full) before it is a rung. One smoke seed — a lead, not
  a result. Wait for the four tables.

## 5 — Five owner asks arrived after the last sitting and each one adds a lane

INBOX, 11-15 17:56 to 19:31, all after the 16:57 sitting: the decisive-test
ready-list as stage 1's deadline; all seven in-house arms through J3's
known-answer gate and into the stage-2 pilot twice each; the walking layer
as a side job now; the MLX Dreamer port as a side lane (23 steps); twelve
Mac first-day measurements as a script. PROGRESS (17:00) predates all of
them. JACK already holds seven open orders, the stated maximum, with three
unstarted. Taken literally these asks double the queue while stage 1's
instrument is one-quarter built. They are the owner's words and need a
ranked answer, not silent absorption and not silence.

---

## Machine

Alive. Ladder slots end rc=0 through 00:15; regate committed 22:49; review
swept 11-15 17:01; scout last ran Thu 11-12 and is due today. `main` is level
with origin. Director stamp 11-15 16:57, seven hours old. `/` is at 88%
(3.7 GB free); `/data` 65%. Uncommitted since at least 20:33:
`CHECKLIST.md` (shows `D1.0` flipping to `[!]`), `cpu_budget.json`,
`gpu_submissions.jsonl` — three slots have noted them and none owns them.
Two undeclared `diag.py` processes were left by the 19:07 slot and died at
three of six lives with no error found (`ladder.log:18773-18776`).

## The honest question

Closer. For the first time the system knows, by measurement and not by
argument, why he starves: floor food can keep him alive only at 70% of a
perfect harvest, the best sensor-only forager manages 26%, and death pays
better than a hungry life. The tick count did not move (111 of 270) and that
is the right reading of a day spent finding out the world was the fault.

---

## FOR THE DIRECTOR

1. **Rule on stage 2's mark before J4 codes it** (finding 1). Which scripted
   forager is the gate, and whether "≥ 5 meals" stays the outcome when J1
   puts a sustained life near 37. A test may only be strengthened; say which
   way this goes and write it beside ruling 19.
2. **Record J1's learnability result as negative at d = 1** in ROADMAP M3 and
   PROGRESS, and forbid a third tuned ruler on the same bar without a new
   written rule (finding 2).
3. **Answer the INBOX `d0` line before the four kernels are harvested**
   (finding 4): what makes a rung, and what happens if no setting on this
   dial sustains a random life.
4. **Put J5 on Colab at the 07:07 slot** and say whether the strong-slot rule
   binds when J1/J2 kernels are waiting (finding 3). Consider one line on
   why a dispatch already burning was not stopped when its arena was closed.
5. **Rank the five owner asks** against the seven open orders (finding 5).
6. One line of bookkeeping: `verify` prints VENUE undeclared 266 against a
   baseline of 256 (ten new specs without a venue); `BD.01` still has no
   ledger row; `coverage` shows four owner commitments CLAIM-DEAD (shelter,
   advised-not-commanded, tree-fire-shelter world, thermal) — known, stage 3,
   not work for today.

## FOR THE OWNER

Nothing that needs your decision today. `D57` (recordings of people may judge
how he moves, never sit inside him) fires its default on 11-29 if unanswered.
Your five requests of yesterday evening are in the Director's inbox for the
06:37 sitting; finding 5 asks that they be ranked rather than all started.
