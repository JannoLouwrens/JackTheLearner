# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Findings are ranked by how much damage they do to the
> trustworthiness of the ledger, not by how interesting they are.

**2026-09-21 06:37–07:0x UTC — the 107th audit.** Running concurrently with the
Review's DAILY sitting, as it does every morning; every instrument in this
report was re-run against the tree immediately before committing, and the
Review's four commits from this sitting (`785f921`…`0056ea5`) are inside the
readings.

---

## VERDICT: INTEGRITY RISK — not in the ledger, which is the cleanest it has audited in weeks, but in the organ that fills it: the builder could not start a single iteration for 23 hours, and all three instruments that watch its liveness read a crashed slot as a healthy one

The ledger itself is sound. 109 PASS, every commit present in git, every
implementation on disk, **0 verdicts that no longer re-derive, 0 gates that
ignore their control**, and the only threshold that moved this window moved
**up**, with exogenous arithmetic. Section 1 and section 2 are clean and I say
so plainly.

The risk is one layer out. Between 2026-09-20 07:07 and 2026-09-21 06:07 the
builder loop started six slots and **executed none of them**, thirteen more were
covered by the usage pace gate, and the project's own dark-slot counter printed
`0 dark slots` throughout — including right now, at the moment I write this,
with the builder twenty-three hours idle. A ladder whose scoreboard is honest
and whose climber cannot leave the ground is not ON TRACK, and the reason no
number went red is the finding, not the outage.

---

## RANK 1 — THE BUILDER COULD NOT EXEC FOR 23 HOURS. The cause is one file crossing one kernel constant, the file was pushed over it by the Review's own Sunday steering rewrite, and the repair landed at 06:41 this morning from a third organ that happened to look

### The measurement, from `/data/jack-logs/ladder.log`

```
2026-09-20T06:10:38  iteration end rc=0     <- the last iteration that ever ran
2026-09-20T07:07:24  iteration end rc=126   ladder_loop.sh: line 278:
                                            /usr/bin/nice: Argument list too long
2026-09-20T08:07:24  iteration end rc=126   (same)
2026-09-20T09:07:23  iteration end rc=126   (same)
2026-09-20T10:07:24  iteration end rc=126   (same)
2026-09-20T11:07:23  iteration end rc=126   (same)
2026-09-20T12:07 .. 2026-09-21T05:07        17 consecutive PACING skips
2026-09-21T06:07:22  iteration end rc=126   (same)
```

Six `rc=126` slots, thirteen pace-gate skips between and after them, **zero
iterations, zero spec attempts, zero ledger events in 23 hours** — against 7 of
7 slots on the morning it died and a demonstrated 9–11 settlements a day in a
good week.

### The cause, derived and then confirmed by experiment

`scripts/ladder_loop.sh:270` reads the entire steering page into a shell
variable, and lines 278–282 pass it as a **single `argv` string**:

```sh
PROMPT=$(cat "$REPO/scripts/ladder_prompt.md")
...
  nice -n 19 ionice -c3 env ... timeout 50m claude -p "$PROMPT" \
```

Linux caps a single argument at `MAX_ARG_STRLEN` = 32 pages = **131072 bytes**.
Measured on this box against `/bin/true`, so nothing was spent:

| bytes | result | what it is |
|---|---|---|
| 129855 | OK | `ladder_prompt.md` at `c124fad4`, 2026-09-19 06:39 |
| 131000 | OK | |
| **131072** | **E2BIG** | the kernel constant, exactly where it should be |
| 139002 | E2BIG | `af21fe0`, 2026-09-20 **06:44** |
| 140331 | E2BIG | `87c8f04`, 2026-09-20 06:48, and HEAD until this morning |

`git cat-file -s` on every revision of the file dates the crossing to the
minute. **The 09-19 steering rewrite left 1217 bytes of headroom. The Sunday
FULL's `af21fe0` at 06:44 spent all of it and 7930 bytes more.** The 06:07 slot
on 09-20 read the file at 129855 bytes and ran normally, ending 06:10. The 07:07
slot was the first to read it after 06:44, and it is the first `rc=126`. There
is no gap in that chain and no second candidate.

### What this cost, priced rather than asserted

- **23 hours, 19 slots, 0 iterations, 0 ledger events.**
- **`2026-W38` opened 09-20 with 30 free Kaggle GPU-hours expiring Saturday
  2026-09-26, and 0.00 h drawn.** For the first time in three weeks a **legal
  buyer existed** — the `T2.06` `gpu<20min` re-buy the Sunday FULL created and
  ordered as its own builder item 3 — and the only organ that may spend it could
  not start. W37 expired with ~24.75 of 30 hours unspent for want of a buyer;
  W38 was on the same path for the opposite reason.
- **Five ordered builder items undeliverable**: `BA.03` (c), `T3.06` (b)+(a),
  the `T2.06` re-buy with its `UNREACHABLE_BASELINE` 98→96 restoration, the
  `T1.08` trigger declaration, and the two 09-21 dispositions.
- **`D27`'s armed default went OVERDUE.** The builder had handed itself the
  firing in its own journal — *"the first slot on 09-21 fires D27's default if
  still armed"* — and the first slot on 09-21 was one of the dead ones. See
  RANK 3.

### The repair, and who made it — recorded as theirs, not mine

I did not fix this and I am not permitted to. **The Review's DAILY sitting
reached the same cause from the same log independently at 06:41 UTC**
(`b8e807b`, roughly fifteen minutes before I finished my own derivation) and
trimmed `scripts/ladder_prompt.md` from **140331 → 85548 bytes**, then added
ITEM 0 at 06:42 (`a7233b2`) telling the builder the work below it is *unread*,
not stale. I verified the repair by exec rather than by reading it: the live
file now passes as a single argv under `nice`. **The builder is expected to run
at 07:07 and the outage is closed.**

Two things follow that the repair does not touch, and they are why this is
RANK 1 rather than a closed incident:

1. **The fix has a computable expiry.** The page has grown 52850 → 140331 bytes
   since 08-29: **~3976 bytes/day, monotone, 22 consecutive days.** Today's
   43835 bytes of headroom is **about eleven days** — roughly 2026-10-02 — and
   nothing in this repository will say a word when it arrives. Routed as `D34`
   with the durable repair (stdin instead of argv) as the live question.
2. **It was found by luck.** No instrument pointed at it. A desk read a log
   during a sitting that had no reason to send it there. That is RANK 2.

---

## RANK 2 — THREE INSTRUMENTS WATCH THE BUILDER'S LIVENESS AND ALL THREE READ A CRASHED SLOT AS A HEALTHY ONE. The one built for exactly this fired its default two days ago

This outlives the outage, so it ranks above the queue and above `D27`.

**1. `scripts/usage_attribution.py:164` — the dark-slot streak.** It counts
trailing `PACING:` lines and **breaks on any line whose first four characters
are digits**:

```python
if "PACING:" in line:
    streak += 1
elif line[:4].isdigit():       # a real slot line ends the streak
    break
```

`2026-09-20T07:07:24+00:00 iteration end rc=126` satisfies that test. So five
consecutive slots that died before reading a token printed **`0 dark slots`** at
12:07. Re-run just now, with the builder 23 hours idle:

```
$ python scripts/usage_attribution.py --line
... builder 20 (41%), desks 1 (2%), both 0 (0%), NOT THIS PROJECT 27 (56%); 0 dark slots
```

**The comment is the bug.** A slot line is not evidence a slot ran; it is
evidence a slot *started*. The counter measures **skipped**, and this project
has been reading it as **dark**. Its own test `P6` pins the defect in place —
*"a loop that just ran is 0 dark slots"* — asserting the behaviour that hides
the failure.

**And the part that makes this a finding about governance and not just a
regex.** `D30`'s armed default fired on **2026-09-19** to build this reading,
for an entry titled *"The builder has been dark for 18 consecutive hourly
slots."* It was **two days old** when the builder went dark in the one way it
cannot see. The instrument is not wrong about what it measures — it is that a
liveness check which cannot distinguish *chose not to run* from *could not run*
is measuring intent, not life, and this project spent a fired default to build
it without anyone noticing the gap.

**2. `scripts/overseer.sh:112` — the no-op gate's condition (2).**

```sh
ITER=$(awk -v ts="$LAST_TS" '$1 > ts && /iteration start/' "$LOGDIR/ladder.log" | wc -l)
```

It counts `iteration start` lines. A slot that dies at `execve` writes one. This
organ's own skip logic would therefore have read yesterday's dead builder as
busy. It fails toward *more* oversight so it cost nothing here, but it is the
same mistaken premise in a second file.

**3. The Review's Part 2.5 organ-liveness paragraph.** `PROGRESS.md` reads
*"Ladder 06:10 — alive, 7 of 7 hourly slots ran."* That was **true when
written** and I am not indicting it: the sitting was at 06:37 and the first
failure was thirty minutes later, caused by a commit the same sitting had not
yet made. It is in this list because it is counted the same way — from slot
lines — and would have said the same thing at 13:00.

**Nothing in this repository asks whether an iteration produced anything.** The
one number that would have caught this in an hour is trivially available and
uncomputed: **consecutive slots ending `rc != 0`**, or the wall-clock gap since
the last `rc=0`. Routed below.

---

## RANK 3 — `D27`'s ARMED DEFAULT WENT OVERDUE AND I FIRED IT. Two organs ordered the same firing within ten minutes and the builder must not fire it twice

`decisions --check` at the top of this audit:

```
D27    costs   0 specs   OVERDUE — DEFAULT IS DUE TO FIRE
```

**The owner did not rule by 2026-09-20, so the pre-registered default fired** —
option **(i) BUILD THE SCREEN, REPORTING ONLY**: a summarisation-aware
`metric_recorded_but_unread` reading in `run status`, **unfloored** until its
false-positive rate is measured and written down. Options (ii) and (iii) were
not taken, for the reasons the entry pre-registered. **Reversal: delete one
function from `experiments/coverage.py`** — and until that function exists there
is nothing to reverse, which is the honest status of this firing today.

It fired **one day late, and the lateness has a named cause rather than an
excuse**: the builder had handed itself this act, and the slot it handed it to
could not exec. That is RANK 1.

**THE COLLISION, and it is the thing to act on.** The Review's ITEM 0
(`a7233b2`, 06:42) orders the **builder** to fire this same default at 07:07. I
fired it at 06:5x. Both orders are correct in isolation and together they
produce a double firing, which in the record is indistinguishable from one
default fired twice under two different readings. The builder must not fire it
again — see FOR THE BUILDER item 1.

Per the `D22` precedent an overseer may fire and the builder transcribes. What
is owed by the builder's next live slot: the reading itself, **its false-positive
rate measured** (the entry's own text binds the firing to this — *"the counter
gets floored or deleted once it exists"*), and the transcription into
`docs/DECISIONS_RESOLVED.md`. The builder's own 09-14 evidence addendum splits
the risk and should govern build order: the **ledger-only half** measured clean
on 840 pairs and re-found the one known true positive; the **bar-pairing half**
is where the prototype's 104-of-107 false-positive rate lives. Build the first,
measure it, do not ship the second on faith.

**`D28` does NOT fire today.** `decide_by` 2026-09-21, and `decisions.py` marks
overdue at `> 0` days, so its earliest legal firing is 2026-09-22. What I did
fire today is `D28`'s **reclassification**, armed by this organ on 2026-09-19
with a stated firing date of today — `class: goal` → `class: conduct`, one word,
`decide_by` untouched, default untouched, authority `SYSTEM.md` class 3 as
amended on the owner's own question, and *sitting order* is that clause's first
named example. `decisions --check` now reads `[CONDUCT-DESK] D28`, so the change
landed where the instrument reads it and not only where a human would.
**Reversal: change one word back.** Recorded honestly in the entry: the reclass
disposes not one row, and `D28`'s default fires 09-22 regardless, so it buys
days rather than capability.

---

## RANK 4 — the queue broke 5 more promises overnight, and 4 of the 5 were dated onto the Sunday by the desk that was sitting that Sunday

`run review-queue`: **17 VIOLATIONS, all OVERDUE**, up from the 12 the Sunday
FULL ended on. `review_queue_violations` 12 → 17, and `run status` calls the
movement **CLOCK** — correct, no commit is to blame for the rise.

But the composition is not neutral. The five that broke at midnight:

| row | promised | dated by |
|---|---|---|
| `t310-anticorrelated-gates` | 2026-09-20 | *"THIRD SLIP"* by its own text |
| `w1-cold-is-not-lethal-at-night` | 2026-09-20 | Review DAILY 09-08 |
| `gates-that-measure-something-other-than-what-they-say` | 2026-09-20 | *"dated onto a SUNDAY on purpose"* |
| `lg03-teacher-does-not-cap-the-twin` | 2026-09-20 | *"dated onto the SAME Sunday"* |
| `ub10-part1-premise-false-marginals-are-what-saturate` | 2026-09-20 | Review |

**Four of the five named 2026-09-20 deliberately, and the Sunday FULL sat on
2026-09-20 and disposed none of them.** The desk scheduled work onto the day it
would be at the desk, arrived, spent the sitting on the stop-rule bundle and
Part 2, and let its own four promises break at midnight. That is not the
builder's outage and it is not the clock: it is a desk dating work onto a day it
had already committed elsewhere. The instrument already warns about this shape —
it flags `2026-09-24` as **AMBER: 7 rows against a measured capacity of 6/cycle**
and names `lt02-...` as `DATED ONTO A FULL DAY`. The same pile is being built
again, three days out.

Throughput, re-read against the tree at commit time — the Review routed two more
rows during this sitting, so these are higher than the numbers I opened with:
**arrived 11 (1.57/cycle), disposed 7 (1.00/cycle), designed 6 (0.86/cycle —
still live, still ageing), drain UNBOUNDED, 53 live rows of 70 routed, arrivals
exceeding disposals by 4 over the window.** The desk routed faster this morning
than it disposed, which is not a fault — routing is the correct repair for a
finding — but it is exactly what `D28` measures, and `D28`'s default fires
tomorrow.

---

## The rest of the audit, section by section

### §1 — Integrity of the ledger: CLEAN, and it is the best reading in weeks

`run verify` re-judged **108 PASS entries from the record alone and probed 106
controls**:

```
verdicts that no longer re-derive      0
gates that IGNORE their control        0
controls declared but never run        2   T0.01, T0.10
gates that could not be replayed       0
entries that could not be audited      0
controls run but NOT declared          0 / 0 budget
```

Independently, across all 109 PASS rows: **0 whose `commit` is missing from
git**, **0 with no `commit` field at all**, **0 whose implementation path does
not exist.** The two flagged rows — `T0.01`, `T0.10` — are Tier-0 existence
claims with no control to delete; long-standing, disclosed by the tool itself,
not new and not hidden. **No finding.**

### §2 — Thresholds and controls over time: NO LOOSENING, and I looked hard

`git log -p --since="7 days ago"` over `registry.py`, `registry_expansion.py`
and `experiments/tests/` covers 22 commits. The only threshold to move is
**`T2.06`'s `MARGIN_LANG`, and it moved UP**: `acc_lang > acc_tfidf_name` — a
strict `>` that would have certified a margin of 0.0001 on a comparison whose
standard error is ~0.035 — became `>= 0.07`, derived exogenously from `n` and
the binomial (`2*sqrt(2*0.25/400) = 0.0707`) rather than from the run, with the
recorded margins disclosed so the derivation is checkable without them. No
conjunct dropped, comparison unchanged. **This is a strengthening and a good
one.**

Every removed-line candidate in the window belongs to `LT.01`'s C2 → C2'
rewrite, which the 106th audit examined and resolved: the 0.6 m bar is
byte-identical on both sides. The rest of the window is new specs arriving with
new bars — `PS.05`, `PS.06`, `PS.08`, `PS.09`, `LT.02` — not old bars moving.
**No `_check` gained an `or`. No seed count fell. No control was deleted or
weakened. No assertion was removed.** No finding, and saying so is the result.

### §3 — Drift from the goal

**What the builder worked on in its last live day (09-19 22:0x → 09-20 06:10),
since it worked on nothing after that:** the lane-guard declaration
(`JACK_DETACHED_LANE`), four fixture cases in `test_lane_guard.sh`, three
journal reconstruction entries for the unjournalled 09-19 slots, and two
`LESSONS.md` occurrence records. Then six verify-and-end slots that correctly
refused to manufacture work from an empty board.

**Which GOAL.md sentence does that serve?** *"Everything below is that one
sentence, unpacked… or protects the honesty of watching what happens when the
three meet."* Conduct and instrument work sits in that fourth clause and is
legitimately in scope — the builder said so itself (*"conduct and instrument
work claims nothing about Jack"*) and did not stamp a render. It is not drift.
It is also not brain, body or world, and **the render has not moved since
2026-09-19: 109/253, six consecutive days.**

**The converse and harder question — what has no passing spec at all.**
`commitments_uncovered = 0`, at floor, so nothing is *uncovered*. But coverage
is not demonstration, and the picture underneath the floor is worse than the
floor suggests:

- **CLAIM-DEAD, every claim spec parked or foreclosed: smell, balance,
  shelter/building, thermal.** Four commitments the owner named constitutional,
  with no runnable path in. `claim_dead = 4`, unchanged since 09-19.
- **`touch`, `tool use`, `told world`, `proprioception`, `sleep`, `fast/slow`,
  `hunger/thirst` — 0 passing, 0 runnable now.** Most are blocked behind
  `T1.08`/`T2.01`.
- **Curiosity: 12 specs, 2 pass.** **One brain / unison: 27 specs, 1 pass.**
  These are the two claims GOAL.md leans on hardest and they are the thinnest
  per spec on the board.
- **Generality: byte-identical for a third consecutive week** — 14 barriers, 4
  registered, **0 implemented, 0 run, 0 PASS** on 09-06, 09-13 and 09-20. This
  is `goal_unrunnable = 7`, red since 09-05, **16 days**, and it is the
  standing `coverage` EXIT 2.
- **`far`, `tiring`, `heavy`, `worth-it` — 1 spec each, 0 pass, all RUNNABLE,
  all FAILED first-ever this week.** That is the best news on this page and I
  return to it in §8.

### §4 — Is the builder alive and productive?

**No, and that is RANK 1.** 0 iterations in 23 hours; 6 slots `rc=126`; 13
pace-gate skips; PASS delta **0**. Repaired at 06:41 by the Review, expected
live at 07:07. One further fact worth naming: the 06:07 slot today printed
`REFUSING fable — week:Fable 100% is at or past the 95% model floor` **and then
hit the same `rc=126` anyway**, so the model-floor refusal path falls into the
identical broken exec. `week:all models` is at **70%**, under the 90% stop;
`week:Fable` is **exhausted at 100%**, so the chain falls to `opus sonnet`.

### §5 — Compute honesty

`gpu_hours_no_verdict = 48.42 h TOTAL`, unchanged — dominated by **`D1.0` at
33.78 h across 2 attempts with 0 verdicts**, plus 6.32 h across 21 unattributed
jobs (`gpu_unattributed_jobs = 21`, at its declared floor). No new spend to
audit: **`2026-W38` has drawn 0.00 h of its 30**, W37 closed at 5.25 h of 30.
The waste this week is not misspent hours, it is **unspendable** ones — and for
the first and only time in three weeks the reason was not "no legal buyer" but
"the buyer existed and the builder could not start". **5 days remain on W38's
30 hours; they expire Saturday 2026-09-26.**

### §6 — Stuck decisions

Acted on, above: `D27` fired (RANK 3), `D28` reclassified (RANK 3), `D34`
routed (RANK 1). Armed and pending: `D28` (fires 09-22), `D29` (09-22), `D32`
(09-24), `D34` (09-24), `D31` (09-25). `D33` is `[CONDUCT-DESK]` — the Review's
to execute and report, not the owner's, and it is the Review's own entry about
its own inability to produce the W1 design.

**Nothing on the register has enough evidence to be decided that is not already
armed, and no owner decision was quietly acted on without being recorded.**
`decisions --check` ratchet: **0/10 undeclared, 0/3 unrouted-owner-ask, 0/0
vanished-owner-ask, 0/0 default-action-expired, 0/0 firing-diff** — all at
floor, verified after my edits.

One honest disclosure about my own firing: because I may not write
`docs/DECISIONS_RESOLVED.md`, `firing_coverage` will not list `D27` as declared
until the builder transcribes it. That reading is **reported, not ratcheted**,
by the tool's own design, and it is stated in the entry rather than left to be
discovered.

**`docs/PROGRESS.md` `FOR THE OWNER`, read this audit as the prompt requires**,
and `decisions --check` agrees at `0 unrouted-owner-ask / 0 vanished-owner-ask`.
Six items, five marked NO-DECISION. The live one is **`D33`** and the Review's
own recommendation on it is *"move the W1 world design to the BUILDER, under
this desk's review rather than its authorship"* — correctly **not** taken by its
armed default, because a default may not reassign authority `D22` settled. It
sits with the owner. I have nothing to add to it that the Review has not already
written against itself, and I am not re-routing it.

### §7 — Bakeoff hygiene

`champions --check` EXIT 0, every class at its declared floor: **2/3
unfalsifiable, 2+1/4 uncontestable, 4/4 unwinnable, 2/2 unverified verdicts, 3/3
trigger debt, 1/1 kindless discharges, 0/0 phantom arenas, 0/0 undeclared
seats** — every seat now says what would unseat it, which is a real improvement
held from previous audits.

The two `UNVERIFIED VERDICTS` remain the standing sore: **Learning core held BY
VERDICT off `LC.03`, which is a VOID**, and **World held BY VERDICT with no
deciding run named at all.** Both are already on the register — `D29` (fires
09-22) and `D33` respectively. `champions_trigger_debt = 3` unchanged since
09-03, `champions_unwinnable = 4` at floor since 09-13. **No decision was made
without a learning gate, no VOID was treated as a verdict this window, and no
winner was chosen inside the noise margin.** No new finding.

---

## §8 — The honest summary, answered directly

**Are we closer to a curious humanoid that climbs the ladder than we were
yesterday? No — yesterday produced nothing at all. Are we closer than we were a
week ago? Yes, and for the first time in a while the answer is about Jack rather
than about the instruments.**

The week before this outage, this project measured something it had only ever
asserted: that **its world does not charge him for anything.** `PS.05` (far),
`PS.06` (tiring), `PS.08` (heavy), `PS.09` (worth-it) were registered, run and
**failed honestly**, four first-ever FAILs in one day, and the finding is that
he can wander as far as he likes, exert himself as hard as he likes and carry
whatever he likes and nothing bills him. GOAL.md says *"the needs ARE the
curriculum"* and *"cold nights teach shelter-building the way no scripted lesson
can"*; those four FAILs are this project's first **measurement** that the
curriculum is not there to be taught. That is worth more than a green tick and
the rate cannot show it — a re-buy and a first-ever verdict both count as one
event, and only one of them is about the creature.

So the honest state is a pair of facts that do not cancel:

**The measurement apparatus is in the best shape I have audited.** Zero
re-derivation failures across 108 certificates. Zero gates ignoring their
control. Zero loosening in seven days, and the one threshold that moved moved up
on exogenous arithmetic. Four commitments went from asserted to falsified in a
day.

**And the creature has not moved in six days.** 109/253 since 09-19. Four
constitutional commitments are CLAIM-DEAD with no runnable path. Generality has
read byte-identical for three weeks. Curiosity is 2-of-12 and one brain / unison
is 1-of-27. Seven independent instruments now say `W0` is too shallow to resolve
anything above it, the repair for all seven is one world design, that design has
been ordered by an armed default and promised on three consecutive Sundays, and
the desk that owns it routed `D33` yesterday to ask whether it can produce it at
all.

**The sharpest thing I can say about this audit is that the system spent 23
hours unable to work and its own instruments reported it as fine.** The ledger
was never at risk — nothing false was written, because nothing was written. But
a project whose liveness check measures *intent* rather than *life*, and which
found the outage because a desk happened to read a log, is one where the next
failure of this shape is equally invisible. The ladder is honest. What it needs
now is not another instrument pointed at the ladder; it is a world that charges
him for being alive, and an hour in which the builder can actually run.

---

## FOR THE BUILDER

**Item 1 is time-critical and supersedes ITEM 0 of your steering page in one
specific respect. Read it before you act on anything.**

1. **DO NOT FIRE `D27`'s DEFAULT. IT IS ALREADY FIRED.** `ladder_prompt.md`
   ITEM 0 (`a7233b2`, 06:42) orders you to fire it; I fired it at 06:5x in this
   commit, with the required wording and the reversal, because it was OVERDUE
   and the slot that was supposed to fire it could not exec. Both orders were
   right when written. A default that fires twice is indistinguishable in the
   record from one that fired under two readings — so **check
   `docs/DECISIONS_NEEDED.md` for the `## D27 — RESOLVED BY ARMED DEFAULT`
   heading before you act**, and if it is there (it is), your job is the
   *remainder*, item 2.

2. **Discharge `D27`'s firing.** (a) Build the `metric_recorded_but_unread`
   reading in `run status`, **unfloored**. (b) **Measure its false-positive rate
   and write the number down** — the entry binds the firing to this and an
   unfloored counter nobody is accountable to is the exact failure it was
   written against. (c) Transcribe the entry into `docs/DECISIONS_RESOLVED.md`
   under a `## D27 — RESOLVED BY ARMED DEFAULT` heading, which is what
   `firing_coverage` reads. Build order from your own 09-14 addendum: the
   **ledger-only half** first (measured clean on 840 pairs, re-found the one
   known true positive); the **bar-pairing half** is where 104-of-107 lives —
   do not ship it on faith.

3. **Build the liveness reading nothing in this repo has: consecutive slots
   ending `rc != 0`, and hours since the last `rc=0`.** RANK 2. Today
   `usage_attribution.py:164` counts trailing `PACING:` lines and breaks on any
   line starting with four digits, so `iteration end rc=126` reads as a healthy
   slot — it printed `0 dark slots` through five dead ones and prints `0` right
   now. The repair is **not** to widen the existing streak: `dark_slots` measures
   *skipped* and that is a real and separate quantity. Add a second reading
   beside it, name it for what it is (`failed_slots` / `hours_since_rc0`), and
   put it where `pace_gate` already prints the streak so a dead launcher is
   visible in the same sentence as a paced one. **Its test `P6` currently asserts
   the behaviour that hid this** — *"a real slot line ends the streak"* — so fix
   the assertion in the same motion or the next reader inherits the premise.
   While you are there: `overseer.sh:112` counts `iteration start` lines for the
   same purpose and has the same mistaken premise; it fails safe, so name it,
   do not gate on it.

4. **Restore `UNREACHABLE_BASELINE` 98 → 96 in the same commit as the `T2.06`
   re-buy**, whichever way the run falls. That obligation was written into the
   growth log by the Review on 09-20 and is unchanged by the outage.

5. **`W38` has 5 days and 30 unspent GPU-hours, and one legal buyer** — the
   `T2.06` `gpu<20min` re-buy, the first in three weeks. It was ordered before
   the outage and is still ordered. If it FAILs, record the FAIL and **do not
   touch `MARGIN_LANG`**.

6. **Everything else on `ladder_prompt.md` `1^10` stands and is UNREAD, not
   stale** — the Review's ITEM 0 is right about that and I confirm it from the
   log: nobody skipped that work, the launcher refused before a token was read.
   `BA.03` (c), `T3.06` (b)-then-(a), the `T1.08` trigger declaration, the two
   09-21 dispositions in order and not folded.

---

## FOR THE OWNER

**1. `D34` — NEW, routed today, and it is the item I would put in front of you
first.** Your builder could not start for 23 hours because one steering file
crossed one kernel constant (`MAX_ARG_STRLEN`, 131072 bytes) and
`ladder_loop.sh` passes that file as a single command-line argument. Nineteen
slots, zero iterations, zero ledger events, and **30 free GPU-hours expiring
Saturday with a legal buyer and no organ able to spend them.** It is **fixed as
of 06:41 this morning** — the Review trimmed the file 140331 → 85548 bytes and I
verified the exec — so nothing is broken while you read this. What I am asking
you to rule on is the *durable* repair, option (i): feed the prompt on **stdin**
instead of `argv`, which removes the ceiling rather than moving it. The reason
not to simply let the trim stand is arithmetic: the page has grown ~3976
bytes/day for 22 consecutive days, so today's headroom is **about eleven days**,
and the next crossing will look exactly like this one. I did not fire option (i)
as a default because an unverified stdin change can fail *silently* with an
empty prompt — replacing a loud failure with a quiet one — and because changing
how the builder is launched touches the same lane `D32` is already open on, and
a desk should not settle that by silence while you hold it. `decide_by`
2026-09-24.

**2. NO-DECISION — the fact underneath `D34`, reported because it is about what
your instruments can see and there is nothing in it to rule on.** Three
instruments watch the builder's liveness and **all three read a crashed slot as
a healthy one.** The dark-slot counter printed `0 dark slots` through five dead
slots and prints `0` right now with the builder 23 hours idle, because it counts
*paced* skips and breaks on any line that looks like a slot. `D30`'s armed
default fired on **2026-09-19** to build that very counter, for an entry titled
*"The builder has been dark for 18 consecutive hourly slots"* — it was two days
old when the builder went dark in the one way it cannot see. The outage was
found because a desk read a log it had no instrument telling it to read. I have
ordered the missing reading from the builder; I am telling you because the
pattern — a fired default that buys a number which does not cover the case it
was bought for — is one you have now paid for twice, and it is not visible from
any ratchet.

**3. `D28` — RECLASSIFIED today, reported not asked, and here is the reversal.**
`class: goal` → `class: conduct`, executing a notice this desk armed on
2026-09-19 with today's date on it. One word; `decide_by`, options and default
all untouched; authority is `SYSTEM.md` class 3 as you amended it on 09-17, and
*sitting order* is that clause's first named example. **Reversal: change one word
back, or rule `D28` yourself at any time.** Stated plainly because it flatters
nobody: this disposes **not one queue row**, and `D28`'s own default fires
2026-09-22 regardless. It buys days, not capability. `D29` and `D31`
deliberately stay on your desk — `D29` would touch an ARCHITECTURE seat's
marking, which `SYSTEM.md` makes never a desk's by fiat, and `D31` would have a
default invent a budget number on a shared box with paying tenants.

**4. `D27` — FIRED today, one day late, off your desk.** *The owner did not rule
by 2026-09-20, so the pre-registered default fired*: option (i), the
reporting-only unread-metric screen, **unfloored** until its false-positive rate
is measured. It is late only because the slot that was meant to fire it could
not start. **Reversal: delete one function from `experiments/coverage.py`.** Your
option (ii) — raise Part 2's sample — remains yours to rule at any time; the
Review's own evidence from yesterday cuts against it and it said so.

**5. NO-DECISION — the six-day render, reported as a trend rather than a
complaint.** 109/253 since 2026-09-19, and this audit found **zero** ledger
defects and **zero** loosening in seven days. The apparatus is the healthiest it
has been. Four constitutional commitments (smell, balance, shelter, thermal)
remain CLAIM-DEAD with no runnable path; generality has read byte-identical for
three weeks; `goal_unrunnable = 7` has been red for 16 days. The one genuinely
good thing to report about Jack is that four claims about whether his world
charges him for distance, exertion, mass or effort were written, run and
**failed honestly** last week — the first measurement, rather than assumption,
that the curriculum GOAL.md relies on is not there yet. Every instrument now
agrees the bottleneck is `W0`, the repair is one world design, and that design
is `D33`, which is already in front of you.
