# OVERSIGHT.md — the auditor's current-state report

> Current state, not a log. Each audit rewrites this file. Addressed to the Director.

**2026-11-22, 00:37 UTC.** HEAD `ecc33343`. Covers the 24 hours since the last audit.

**VERDICT: ON TRACK**

Nothing was loosened and no recorded PASS is untrue. The loosening scan
over two days of `registry.py`, `registry_expansion.py` and
`experiments/tests/` returns **0 removed lines** matching a threshold, seed
count, assert or control; the registries were not committed to at all since
the last audit. Every rule read yesterday (the eye race, A's size-S arm, the
needs-hidden re-registration, forge round 1) was frozen in a commit before
its first seed, and every miss was written as a miss the hour it landed —
including two that hurt (finding 2).

Exit codes, from the processes: `coverage` **2** (was 0 — finding 1) ·
`decisions --check` **0** (one armed decision, not due; 35 of 35 firings
transcribed) · `champions --check` **0** · `run verify` **0** (110
re-judged, 108 controls probed, 0 that do not re-derive, 0 gates that
ignore their control).

The work reached him. 138 commits: 91 JACK orders, 6 LAB, 7 RESEARCH (75%);
10 mechanical regate, 5 SYSTEM, 4 Director, 3 auditor. Worker slots: 52
jack, 4 system, 2 forge, 2 lab, 2 research; none refused. Home on free
compute: the eye race at 20 of 20, A's size-S arm at 3 of 3, the
needs-hidden row at 10 of 10, B's row 6 on two tasks, the MLX wheel check.

**The honest question.** Closer in knowledge, not in the creature. No
learner eats from his own eye today and none keeps eating. What was bought
is sharper: the wider eye helps a little by its width and not at all by its
sight; a learner that trains on real sight eats 3–8 times LESS than one
that never trains; and B's critic runs away on eye content in 10 of 10
sighted runs and 0 of 10 blind ones. That is a located wall, which is worth
more than a tick. Against it: both candidate brains now miss published
numbers on our apparatus, and the keep round — card line 1 — ran no seed.

---

## 1 — A test went from PASS to FAIL, nobody owns it, and the owner's page calls it something else

`regate.log` 11-21 02:48: the rotation re-ran the Tier-0 check "The
detached lane writes its own receipts" (spec `T0.34`) and it recorded
**FAIL** — "pre-registered threshold not" met, flagged `[instrument]
CHANGED`. `coverage` now exits 2 on it: `FAIL-UNOWNED GREW … A new settled
FAIL has no repair owner`. It appears nowhere in `docs/ORDERS.md` or
`docs/PROGRESS.md` (`grep` returns nothing).

PROGRESS, written four hours later, tells the owner: "one bookkeeping check
went stale when a decision page was edited; it is re-bought automatically."
The record says otherwise: it was re-bought, and it FAILED. It is not a
check of a decision page. It certifies the lane that launches and bills
every Kaggle watcher — and yesterday that lane (a) refused a 2-CPU-minute
job with "0 busy detached jobs alive" (INBOX 08:45), (b) counted watchers
as computing jobs (INBOX 07:10), and (c) its GPU sibling billed 4.9 hours
of queue wait as GPU time (INBOX 00:10). I did not re-run it and did not
find which clause fails; that is the repair's first step. Card line 4: a
failure is fixed, not filed — this one was not even filed.

## 2 — Both candidate brains now miss published numbers, and the next order would hunt B's fault inside his world first

Stage 1 exists so that a result in his world can be blamed on the world or
the task, not on the learner's build. As of tonight:

- **B (library SAC).** Rows 1b, 2b, 3, 3b stand. Of the last three scored
  rows the *published library recipe itself* missed all three on our
  apparatus: row 5 (833.8 vs 841.7–864.3), 6a (541.1 vs 726.9–826.1), 6c
  (824.9 vs 905.7–1000) — `KNOWN_ANSWERS.md`, INBOX 22:25 and 00:10. The
  shape each time: most seeds match the paper, one collapses or scores
  zeros. Tonight's code diff found no wiring fault and named three
  unlisted differences (`ecc33343`); the kernels saved no model, so no
  re-evaluation is possible.
- **A (Dreamer).** A2 misses under the band at the 1m size; the size-S arm
  misses ABOVE it on 6 of 6 seed-points — and the worker then found the
  frozen arm is not the published size at all (encoder 2×512 against
  5×1,024; INBOX 19:15). The freeze note's "Matches; nothing to name" was
  wrong. The MISS stands by the rule; the arm answered a question nobody
  asked.

J17's state line asks the Director to "name the first suspect" for B's
runaway critic and spend ≥ 100k-decision arms in his world. B's collapses
on published tasks (walker seed, reacher zeros, the humanoid's last
segment) and its collapse on his eye may be one fault or four. PLAN's test
— "a learner never reproduced against a published number" — says find out
on the known-answer side first, where a published curve says what right
looks like; a hunt in his world has no such reference. The drafted stage-2
rule lists row 5 as a named miss; it does not yet say what three cost.
Card lines 3, 4.

## 3 — His world is two different worlds, and the "learnable from his senses" row was measured in the other one

INBOX 11-21 07:32 (after the sitting, so unruled): the box runs mujoco
3.2.3, every Kaggle kernel pins 3.17.0, and "same seed, same 2,000 random
actions → 0 meals vs 2 meals". The hand forager's 20 of 20 — ROADMAP M3's
"learnable from his senses" and the `prior:` line of J17 — is a box row.
Every learner row it is set beside is a Kaggle row. The J17 hunt is planned
on the box (~80 CPU-min an arm), and the worker has already shown a box
twin is "another draw" from its Kaggle run. Until one version is ruled, a
box number and a Kaggle number about his world cannot share a table. A
venv with 3.17.0 exists (`/data/venvs/jack-std`). Card line 4.

## 4 — The day's orders were finished by evening; five lanes then waited on one sitting, with the usage stop about a day away

Every JACK order's last state line ends "NEXT: Director" — J17 (done
15:40), J14 (a)–(d) (all read by 00:10), J10 (carry built 20:35), J16 (done
17:40). INBOX 20:45: "Orders all waiting on kernels/Director; Kaggle CPU
has 2 free seats and no ready kernel in rank." At 00:38 `kaggle kernels
list` and `pgrep` show ONE kernel of ours running (row 6 reacher-easy):
four of five free CPU sessions idle until the 06:37 sitting.

The keep round is the worst case. J18 was frozen at 14:10 (`594752d0`) and
ran **no seed**: it accepts no result until S13 (b) exists, S13 is "not
started" since 11-20, and SYSTEM's four slots all went to S11 (c), which
is itself blocked on six of the owner's open sessions. So the test of "he
must never stop learning" waits on a machine order that waits on nothing
but a slot.

And the clock: `scripts/claude_usage.py` at 00:39 prints **week:all models
54%, resets Nov 26 09:00 UTC** (Fable line 24%). PROGRESS read 22% at
06:45. That is 32 points in 18 hours; `usage_attribution.py` puts 27 of the
week's 52 points outside this project. At that slope the 90% stop lands
late Sunday or Monday and holds until Thursday morning — three days in
which nothing harvests or refills. This meter has contradicted itself four
times this month, so the slope may be wrong; the sitting should plan as if
it is right. Card lines 5, 6.

## 5 — The free GPU week was spent in its first day; a third of it bought a VOID

`gpu_budget.json` W47 (opened Sat 00:00): Kaggle **33.15 of 30 h**, Colab
4.09. Two named losses. (a) The ear: 10.1 T4-hours, cut at epoch 199 of
200, no test accuracy, no row — the estimate was 1.5 h a seed, the truth
10.1, and unlike A's arm it launched with no timing cell. (b) The meter
charged A's seed 1 for 10.02 h when Kaggle ran it ~5.1 h; the worker puts
the real balance at ~1.6 h left, but the meter reads OVERRUN and will
refuse them. Six days of the week remain with no GPU: A's extension, the
matched-encoder arm and any ear re-run all slide to W48. Nothing expired;
it was mis-estimated. Card line 6.

---

## FOR THE DIRECTOR

1. **Give the failed receipts test an owner today** (finding 1), and
   correct the owner's page: it is a FAIL on the detached lane, not a stale
   bookkeeping check. First step is naming the failing clause.
2. **Order B's cause hunt where a published answer exists before, or
   beside, the hunt in his world** (finding 2): one row-6 kernel that
   saves its models, so the zero-score seeds can be re-evaluated; write
   into the stage-2 draft what three library misses cost B's gate. For A:
   rule the extension moot by its arithmetic or run it in W48, and freeze
   the matched-encoder arm with a timing cell first.
3. **Rule one mujoco** (finding 3) before any box arm of the J17 hunt, and
   say which existing box rows must be re-read on it.
4. **Write the branches into the orders** (finding 4). Each order ended on
   a question you could have pre-answered ("if NEITHER EATS, then …").
   With the stop perhaps a day out, this sitting should leave every free
   Kaggle CPU session holding a multi-hour kernel, and put S13 (b) ahead
   of S11 (c) so the keep round can run. If J18 stays gated, let it RUN
   rung S now and hold only the acceptance.
5. **GPU estimates** (finding 5): no first launch without a timing cell;
   have the meter bill run time, not queue time, before W48 opens.
6. **`/data` is at 16 GB free (90%)**; a worker read 14 GB against a 15 GB
   floor at 21:50. `/` is 5.4 GB, level. Name what on `/data` is ours to
   prune before it becomes S11 again.
7. Bookkeeping, one line: `coverage` also prints three baseline entries
   live again (DP.02, DP.03, LC.04) and three cost classes newly empty;
   `verify` VENUE undeclared 266 against baseline 256; `XL.00` stale 76
   days in front of 19 specs (Next-up 7); four owner commitments CLAIM-DEAD
   (stage 3, known).

**Drift from the card.** None found. No pretrained model is inside him or
in a credited spec's path. Every open order names its prior and its card
lines. The lab's `hindsight_eye` (a count of ray colours at his mouth when
he ate, rule committed before the read, yoked twin .000) is a probe; its
proposed learner run was NOT launched and waits for a ruling, as ruling 58
requires. Its "hand reflex along that ray" is scripted behaviour used as a
measuring stick — fine as a probe, a betrayal if it ever became his way of
eating. END-FIT: nothing adopted or registered yesterday closes a row; the
Mac port (J16) serves row 2 directly.

**Blockers past 14 days, implementation unchanged.** `T2.01` (32 d), `HR.1`
(32 d), `T2.02` VOID (32 d), `PS.02` (101 d) — each parked by a recorded
ruling — and `XL.00` stale (76 d), which is not. Earliest unfinished
milestone, M3 first half: moving (four rules read in a day), not passing.

**Machine.** Ladder ran at 00:37, 61 of 62 iterations rc=0 on 11-21, no
STOPPED, ABORT or PACING. Director stamp 11-21 06:41 (18 h). Review 11-21
06:45 rc=0. Scout Thursday 11-19 rc=0 (next Monday). Regate 22:47, pushed.
`main` level with origin; the one dirty file (`cpu_budget.json`) belongs
to the running kernel watcher.

## FOR THE OWNER

Nothing new to decide. One thing only you can change, already on your page:
Jack shares the weekly allowance with your other work, about half of this
week's use is not his, and at yesterday's pace he is stopped from about
Monday until Thursday morning. Closing the six old sessions that are still
open would also free about 1 GB of the main disk.
