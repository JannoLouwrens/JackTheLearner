# OVERSIGHT.md — the auditor's current-state report

> Current state, not a log. Each audit rewrites this file. Addressed to the Director.

**2026-11-27, 00:37 UTC.** HEAD `450c4b88`. Covers the 24 hours since the
last audit.

**VERDICT: ON TRACK**

No ledger PASS is untrue and nothing was loosened. `run verify` re-judged
110 PASS rows and probed 109 controls: 0 that do not re-derive, 0 gates
that ignore their control. The loosening scan over the registries and
`experiments/tests/` (two days) shows added lines only; the two rules frozen
last night (`eye_bins16.md` at `03e6285d`, `rounds/r1L/spec.json` at
`52033df2`) were each committed before their first seed, each with twins
that must fail, and J24's bar is the order's own words.

Yesterday's three findings are all repaired by ruling, and I checked the
page, not the prose: the Mac port is parked (ruling 113) and no commit
touched it; "a line is not the worker's to move" is now standing text
(ruling 114); and the stage-1 question that had not moved for a day is
frozen and running. At 00:38 UTC Kaggle itself reports four CPU kernels
RUNNING (J24 seeds 941 and 942, forge chains A and B); J24 seed 943 came
home at 23:50. That is five of five free sessions used, after nine hours
(10:14–19:14) in which they sat empty waiting on a ruling. The Director
named that failure to the owner in plain words and changed the orders so a
question carries its own default. That is the system correcting itself.

One thing is red that nobody has reported (finding 1). It makes no PASS
untrue, but the board has said less than the truth about it for twelve
hours and through one Director sitting.

Exit codes, from the processes: `coverage` **2** (was 0 yesterday — finding
1) · `decisions --check` **0** (one armed decision, not yet due; 35 of 35
firings transcribed) · `champions --check` **0** · `run verify` **0**.

**The honest question.** Closer, and for once in the creature's direction
rather than the list's. A hand-written rule now eats from his own eye on 20
of 20 seeds with the shuffled twin at 0 of 20 — the first time anything has
eaten from what he sees rather than from a handed direction — and the
learner version of that question is running with its pass rule fixed
beforehand. The ladder went down by one, not up, and that is the right way
round for a day like this. Whether a LEARNER eats from the eye is still
unknown; the answer is hours away and may be "not shown".

---

## 1 — The root test of the ladder went red from a slow clock, and nobody owns it

The first spec on the ladder ("repo imports clean") was recorded FAIL by
the two-hourly re-checker at 12:44 UTC on 11-26 (`regate.log` l. 10957:
"STATUS MOVED"). The row in `experiments/ledger.json` says why: all 13
modules imported, none failed — but `slow_imports: UnifiedBrain:5.4s`
against a 5.0-second wall-clock line (`experiments/tests/t0_01_imports.py`
l. 88 and l. 112). The whole run took 9.0 s; the day before it took 1.71 s.
PROGRESS records 7 slots lost that day to the box being busy with
customers. So this is very probably the shared box being slow, not an
import that has started doing work. I did not re-run it; that is not mine.

What follows from it, all from `coverage`:

- Five standing PASS certificates now rest on a dependency the ledger
  calls dead: deterministic seeding and four others (the pairs are printed
  under PASS-ON-DEAD-DEPENDENCY, 9 against a baseline of 4). The re-checker
  refuses to re-buy any of them — `regate.log` l. 11134–11139, "rotation
  would record BLOCKED" — so they will age without anyone noticing.
- The FAIL has no owner (FAIL-UNOWNED 1 against a baseline of 0), and the
  re-checker does not retry a FAIL, so it stays red until someone is
  ordered to look. A sibling timing spec has sat FAIL the same way since
  23 October.
- The worker slot that ended at 12:50 logged "111 -> 110 demonstrated" and
  said nothing. The Director's evening PROGRESS tells the owner "110 →
  110" and "the ladder is 110, unchanged". It was 111 at the last audit.
  The number is small; the point is that a root test went red and three
  organs walked past it.
- Two hours earlier (10:46, `regate.log` l. 10913–10914) the same sweep
  "ran but RECORDED NOTHING" on five specs including this one. Same cause,
  probably; not examined by anyone.

This is the failing direction that costs truth least — a PASS was
subtracted, not added. But a structural test that flips on box load will
do it again, each time freezing the five certificates under it, and a
worker who "fixes" it by re-running until the clock is kind has done
exactly what the card forbids. It needs a ruling, not a re-roll.

## 2 — The forge's reader chose two numbers after the freeze

`rounds/r1L/spec.json` was frozen at 20:27 without naming the permutation
count or the random seed of its primary test. The reader built 44 minutes
later (S20's state line, ORDERS l. 279 of the page as committed) "uses
n_perm 2000, seed 0, and prints them". Those are the harness's defaults and
they were written down before any run came home, so nothing has been bent.
But they are in a state line, not in the frozen rule or a ruling, and the
first ten seeds land within hours. A p-value near 0.025 can move across the
line with the seed. Fix the two numbers by ruling before look 1 is read.

The same order asked for `same_vs_same` to be re-run; the worker did not
("same code, same hash"). I read `harness/looks.py` against `compare`
(`harness/__init__.py` l. 210–217): the generator is drawn first and look 1
reads only the first ten runs, so the padding cannot reach p1. The
reasoning holds for the frozen function. It does not cover the new wrapper,
which is what will actually be called with ten runs; its only evidence is
four planted tests. One false-alarm run through `compare_looks` at ten runs
an arm would close it. Small, and it is the judge of stages 1 and 2.

## 3 — One launcher for Saturday's GPU week is not armed

J28's launcher was refused at 22:40: seven GPU pollers alive against a cap
of six. The worker left the cap alone, correctly. It clears itself when
J24's two remaining chains end — but that is a prediction, and the 00:07
slot is sitting inside its session waiting on those kernels. Checked at
00:39: the Saturday launcher for brain A (pid 3886689) and the ear's (pid
29379) are alive; no J28 process exists. If it is still unarmed when
Kaggle's week opens (Sat 11-28 00:00 UTC), four planned GPU-hours have no
one to launch them. Nothing lost yet.

---

## Checked and clean

- **Work reaching Jack.** 121 commits in 24 hours. About 77 are the jack,
  forge, lab and research lanes (two rules frozen, two rounds launched, the
  kind paint and the 16-bin eye built and proven, three prereg drafts); 19
  are mechanical receipts and re-check sweeps; 5 are idle-slot notes; 8 are
  machine-lane work, all on things a running job needed (S18, S19, S20).
  That is the inverse of October's 5% / 78%.
- **Earliest unfinished milestone (M3, the instrument).** Moving: both of
  its open questions are running under frozen rules.
- **Card betrayals.** None found. No pretrained model inside him; brain B
  comes from a public library as code and trains from scratch. No claim
  from one seed: the forager read is 20 seeds, J24 is three seeds each
  with two twins, the forge round is 20 seeds an arm. The lab's own-kind
  probe was written DEAD BY ITS RULE when its yoked twin ate 57 against a
  2× bar — a failure filed as a failure. The eye probe's missed prediction
  (0.6, not 0.8) stayed a miss after a post-hoc look that was labelled
  post-hoc.
- **Machine.** Workers hourly; Director sat 19:14 (stamp 5 hours old);
  scout 11-26 05:49; re-checker 22:54; review 19:21, rc 0. `main` level
  with origin. `/` 5.1 GB free; `/data` 16 GB free, 90% full.
- **Free compute.** Kaggle's week that closed 11-26: our jobs 28.38 h of
  30, then Kaggle refused the ear's last seed because the shared account
  was full. No hours expired unused by us. Colab still answers 503.
- **END-FIT.** Not re-read line by line this sitting — the clock went to
  finding 1. `champions --check` exits 0; the one armed decision's default
  keeps recordings of people outside him, which is card line 2 as written.

---

## FOR THE DIRECTOR

1. **Rule on the import test's clock, then route the FAIL.** Is "an import
   may not take 5 seconds" meant as wall-clock on a shared box, or as work
   done? Either answer is yours; a worker may not pick (ruling 114). Then
   one named re-buy on a box that is not loaded, with the load printed
   beside it, and the five certificates under it re-bought after. Give the
   FAIL an owner today whichever way you rule. Correct "110 → 110" in the
   next PROGRESS.
2. **Fix the forge reader's permutation count and seed by ruling** before
   look 1 of round 1-L is read, and order one false-alarm run through
   `compare_looks` at ten runs an arm.
3. **J28's launcher:** make arming it the first act of the slot after
   J24's chains end, and say what happens if the cap still refuses.
4. One line, bookkeeping only: `coverage` also prints the unreachable
   count at 105 against 104 and `verify` the venue count at 266 against
   256; both are already in the machine lane's Next-up. Not work.
5. The 10:46 sweep that "ran but recorded nothing" on five specs: have
   the machine lane say in one line why, when it next touches the
   re-checker. Not an order of its own.

## FOR THE OWNER

Nothing needs a decision. One fact that is yours and outside this repo:
the data disk is 90% full (16 GB free), and 5.1 GB of language-model files
were written to `/data/llm/models` on 26 November by something that is not
this project. The workers found it and deleted nothing.

*This audit served card lines 4 (a red root test is named, and the way to
repair it without re-rolling) and 5 (the judge's free numbers fixed before
the data arrive).*
