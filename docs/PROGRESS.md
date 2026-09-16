# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: **DAILY** (Part 1 over the last 24 h, and Part 2.5 in full). Part 2 is
> deliberately skipped — tests are re-examined on Sundays.

**2026-09-16 06:37–07:3x UTC — DAILY.** Window: the 24 hours since yesterday's
DAILY sat down.

*The one sentence: **the builder produced nothing for the second consecutive
day — forty-one dark slots, zero settle events, zero commits — and this desk
spent the silence paying the four promises it deliberately let break at
midnight, which turned out to contain a real finding: `LG.00`'s "life" is
synthesised by a random number generator, and four of the seven primitives
`GOAL.md` says survival earns have no entry in the register built to notice
exactly that.***

> This page is the RECEIPT for commits that already exist. Every act below was
> committed as it was made, before this page was written: `81fdaba`, `34116ca`,
> `d521384`, `36d120c`, `39bb22b`, `a07f6ee`, `8e92ab4`.

---

## The numbers

| | today (24 h) | yesterday (09-15, DAILY) |
|---|---|---|
| demonstrated / registry | **108 / 249** | 108 / 249 |
| pass rate | **43.4%** | 43.4% |
| net demonstrated, 24 h | **0** | 0 |
| **settle events, 24 h** | **0** | 1 |
| commits, 24 h | **15 — and not one of them the builder's** | 26 |
| rework rate (attempt > 1) | **78.4%** | 78.4% |
| standing VOID / FAIL / BLOCKED | 14 / 25 / 1 | 14 / 25 / 1 |
| unreachable (shrink-only floor) | **97** — AT floor | 97 |
| stale certificates | **12 + 1 pre-`impl_sha`** (`T1.07` still deliberately staled) | 12 + 1 |
| **consecutive dark builder slots** | **41** | 18 |
| `week:all models` | **72%** against a **44%** line at 29% of the week | 37% vs 35% |
| …of which **NOT THIS PROJECT** | **28 of 37 points — 75%** | 75% |
| GPU, live week `2026-W37` | 3.4899 charged, **26.51 free of 30**, expires **Sat 09-19 — 3 days** | same, 4 days |
| queue violations | **0 → 4 at midnight → 0** (all four ACTED, none re-dated) | 0 → 6 → 0 |
| live queue rows | **47** (31 OPEN / 3 HELD / 13 DISPOSITIONED) | 50 |
| `review_queue_net_arrivals` | **6**, MOVED **−2** and earned (clock +1, act −3) | 8 |

**Goodhart check: unavailable for the second day, and for the second day that
is the finding.** Rate cannot be read against registry growth when neither
moved. The sharper number is the one below them: **zero settle events in
twenty-four hours.** Not one spec on this ladder returned a verdict of any
kind — not a PASS, not a FAIL, not a re-buy. The last settlement was
`T1.08 FAIL`, 2026-09-13. **The ladder has been motionless for three days**, and
the only reason the commit count is not zero is that three auditing organs kept
writing about the stillness.

---

## Part 1 — the state of progress, last 24 h only

### Produce, thrash, or stall? Neither — it was switched off, for the second day

`2026-09-14T13:07` through `2026-09-16T06:07` is forty-one consecutive
`PACING: ... skipping` lines. The gate is firing correctly on its stated rule;
the rule reads `week:all models`, which stands at **72%** against a pace line of
**44%** at 29% of the week elapsed. The gap is **28 points and widening**: the
line rises at a fixed 0.3869 pts/h, and 75% of the meter's consumption is not
this project's. **Yesterday's arithmetic predicted this and was right** — the
builder does not come back this week by waiting. It is `D30`, it is on the
owner's desk, and this page will not re-forecast it.

Two costs became instrument-visible in this window rather than merely predicted:

- **`D19`'s armed default is now two days overdue and unfired.** `run decisions`
  prints `OVERDUE — DEFAULT IS DUE TO FIRE`. The organ that fires it is the one
  being skipped. Armed defaults exist because `D1` deadlocked twenty days
  blocking thirty-eight specs; a deadline conditional on a spend gate rebuilds
  that deadlock through a side door.
- **`w1-world-edit-window` (DUE 09-18) is now the gate on the project's only
  remaining world branch** — see the disposition below — and the work behind it
  is builder registration. That is new today and it is the second such cost.

### The day's real find, and it came out of a promise this desk broke

Yesterday this desk left four rows standing on their due date and wrote: *"If
they break at midnight the break is mine and it will be reported as mine."*
They broke. `review_queue_violations` read **4** at 06:37. **All four are paid
this morning — ACTED, not one of them re-dated** — and the first of them
contained something worth the sitting.

`told-world-has-no-rung` sub-question (a) asked whether `LG.11`'s matched fact
sets can be built before W1 exists. Yesterday's desk **refused to answer it and
named the decisive test in advance**: *"settling it requires reading how `LG.00`
SOURCES its life corpus — whether from lived `W0` episodes or from synthesised
diary entries — and a wrong answer there licenses a told-world rung built on a
corpus nobody lived."*

**The reading was done. The corpus is synthesised.** `lg_00_not_a_puppet.py:156`
imports `_build_life` from `LG.01`, and `lg_01_*.py:229` is an RNG draw over
word pools written straight into `EpisodicMemory` —
`rng.sample(RESOURCES, N_GEN)`, `mem.record("did", "jack", f"jack found {res} at
{place}")`. No simulator steps. No body. No life.

This does **not** touch `LG.00`'s PASS, and the distinction matters: `LG.00`
claims Jack's *diary* beats an LLM without it, and that claim is true, measured
at `grounded_knowledge_advantage` 0.5327 with a false-diary control at −0.2000.
What it is not is a certificate that anything in the diary was **lived**.
Yesterday's desk flagged `LG.00` as *"a counter-example in the making"* for
sub-question (a). It does not survive the test yesterday's desk wrote for it,
and it is refused on that test.

**Then the second ground, which is larger than the row.** `GOAL.md:186-188`
justifies the entire survival programme in one sentence — *"Survival earns him
the primitives that make anything else mean something — hot, heavy, far,
tiring, dangerous, worth-it, that-person-lied."* Measured against `coverage`'s
commitment register and the ledger:

| primitive | commitment | certified lived? |
|---|---|---|
| dangerous | `damage/nociception` | **YES** — `PS.03` PASS |
| that-person-lied | `social/other agents` | **YES** — `LG.02` PASS |
| hot | `thermal (kills)` | **NO — CLAIM-DEAD**, 0 pass |
| heavy | **no commitment entry** | no registered claim |
| far | **no commitment entry** | no registered claim |
| tiring | **no commitment entry** | no registered claim |
| worth-it | **no commitment entry** | no registered claim |

**Two of seven.** And the two that landed are the two least usable for this
design — `LG.02` is a social inference, `PS.03` is nociception; `LG.11`'s
hypothesis names five sensorimotor primitives and gets one of them. A few dozen
anchored facts cannot be drawn from one usable primitive, and a set drawn from
uncertified ones makes `LG.11`'s own control **vacuous**: the stripped agent
would show no gap because there was no lived ground to strip, and the run would
read FAIL-by-construction against a claim never tested.

**This is the 2026-08-09 shape recurring inside the file written to prevent
it.** The mechanism is not a bug — `coverage` parses BOLDED commitments and
these seven live in a prose sentence — which is precisely why no instrument
here could report them. Routed as its own row (DUE 09-18) and ordered to the
builder as steering item 5; the instrument half is monotone (adding a
commitment can only raise `claim_dead`, never lower it) and needs no ruling,
and the question of whether to commit to claims or correct `GOAL.md`'s words is
explicitly not the builder's.

### The other three promises, and one of them was the wrong question for three cycles

- **`reparenting-the-welded-fifteen` + `goal-cites-four-specs-that-resolve-to-
  corpses`**, answered as one question (`34116ca`) after **three** broken dates
  — 09-06, 09-10, 09-15; the fourth would have triggered the standing stop-rule.
  The fifteen were recomputed live at registry 249 rather than quoted from the
  08-31 walk at 211: `LC.03` → 8, `UB.10` → 5, `T3.06` → 2. **The answer is that
  no re-parent is owed.** A re-parent repairs a *wrong parent*; all three weld
  roots are the other case — a *right* parent with a VOID run — so the repair is
  a SUCCESSOR SPEC (`SM.02→SM.03`, `BA.02→BA.03`, `SH.01→SH.02`, three worked
  precedents), which leaves every dependent's `depends_on` untouched. Editing
  fifteen dependency entries to point somewhere reachable would shrink
  `unreachable` (floor 97) by **re-labelling**, which `T0.31` was gated to
  forbid. And the row was dated three times against `W1` registration, **which
  was never its blocker** — the three repairs already have owners (`UB.10`'s arm
  redesign is mine, `T3.06`'s is its own row, `LC.03`'s successor is downstream
  of `D29`). The four `GEN` ids are DECLINED as a design question and re-parented
  to `D24`'s ruling, on the row's own words. Deleting their `GOAL.md` citations
  would clear `coverage` rc=2 in one edit and is refused; `goal_unrunnable`
  stays **7**.
- **`w100-honest-null-does-not-rescue-pile-a`** (`d521384`). Its carry-back half
  was already discharged on the `w0-too-shallow` row on 09-06 and is **not
  credited twice**. What it still owed was the ordering consequence: **the W1
  ordering stops being contingent and becomes unconditional.** Before `W1.00`
  the programme was hedged — if Pile A dissolved, W0 was fine and the repair was
  in our scoring. `W1.00` fired the immaterial branch, so the hedge is gone.
  `W1.01` (passivity dies) first, because it measures W0's headroom directly and
  can still falsify the whole programme cheaply; `W1.03` second; `W1.04` last and
  already held there by its own falsifier. **Honest limit kept rather than
  rounded: Pile A is closed on SEVEN of eight margins.** The eighth (`dwell`,
  `T3.06`) reads CANNOT TELL — `f_dw` 0.0082 exceeds that margin's own std
  0.0062 and the spec's guard excludes it. Nothing above depends on the eighth.

The contrast between the two dispositions is the rule worth keeping: **re-parent
when the edge is false, write a successor when the run failed.** `LG.11` is the
one spec in the family with a false edge, and it is not among the fifteen.

### The frontier, recomputed

Unchanged in shape, two days older: **`LT.01` FAIL frees 7** (impl unchanged
**15 d**), **`NE.01` FAIL frees 7** (**22 d**), **`UB.10` VOID frees 4** —
and today's disposition sharpens that one: `UB.10` is not a graph problem, it
needs a successor arm, and that design is **my** outstanding debt. **`T1.08`
FAIL frees 3 and blocks 45.** `run next` reads **0 fresh of 44**. The cheapest
ladder-moving unit on the board is still `T1.07`'s owed ~0.47 GPU-h re-buy,
still unbought, its certificate still standing deliberately staled.

### Effort vs. goal

Fifteen commits, **zero from the builder**. Seven are this desk's, and all seven
served the queue, the steering page and the ratchets — governance, not
capability. Eight are yesterday's Review and the 97th audit. **Nothing in this
window served the creature, and there was nothing legal that could have.** The
honest accounting is that a second consecutive day of this project's output was
produced entirely by organs auditing a silence, which is the 2026-08-28 shape —
*an organ whose only remaining subject is the organ next door* — arriving again
with a different cause. Today at least the desk's own subject was the ladder:
four of its dispositions were about specs, not about organs.

---

## Part 2.5 — steering maintenance

**1. Priorities — AMENDED, not replaced (`8e92ab4`), and the restraint is the
point.** `1^8` has **never been read by its addressee** — 41 skips since it was
written. Re-ordering an unread block is churn, not steering, so items 1–4 keep
their order and their perishability framing untouched. One correction in place:
item 1 said `D19` was *"overdue since 00:00 today"* and "today" was 09-15;
absolute date substituted, and it is two days overdue now. Two zero-GPU units
appended **below** the perishable head, both earned by this morning's
dispositions: (5) add the four missing primitives to `coverage`'s commitment
register, with the expected `claim_dead` 4 → up to 8 declared in advance as the
gap becoming visible rather than a regression; (6) the `W1.01`/`W1.03`
registration order recorded in advance and **explicitly not yet permitted**,
so it is not re-derived and not discovered late. `run steering`: **13 orders, 0
naming a spec the runner would refuse.**

**2. Field watch — nothing new, and nothing owed.** wk7 landed 09-14 05:57,
was consumed in full the same morning, and `FIELD_WATCH.md` is unchanged since.
`run status` reads **2 finding sections, 2 cited, 0 UNROUTED-FIELD-FINDING**.
Next sweep Monday 2026-09-21.

**3. Seat staleness.** `champions --check` **EXIT 0**. **`World` still declares
no deciding run and no `TRIGGER:` while held BY VERDICT — thirteen days**, the
oldest seat finding on the board and still nobody's dated row. `Learning core`
holds 3 trigger debts plus the `A4` diagnostic debt (`D29`, DUE 09-18, mine).
`Fast/slow coupling` stays welded behind `LC.03` — and today's disposition says
what that weld actually needs: a successor, not surgery. 4 seats unwinnable by
construction (**Episodic retrieval, Language grounding, Smell, Body schema**),
at the declared floor.

**4. Organ liveness.**

| organ | cadence | last fire | verdict |
|---|---|---|---|
| builder | hourly | **last real iteration 2026-09-14T11:14** | **process alive, output zero — 41 consecutive `PACING` skips, 43 h.** The gate is firing correctly on a meter three-quarters of which is not ours |
| overseer | 6-hourly | 06:37 today (98th, sitting concurrently) | alive; its 12:37/18:37/00:37 slots paced out, only the `D15` clause-(c) exempt slot ran |
| field watch | Mondays | 09-14 05:57 (wk7) | alive, on cadence, consumed same morning |
| review | daily / Sun FULL | this run | alive; not pace-gated, which is why every disposition above is mine and carries no excuse |

**5. The queue — four broken promises paid, and for once none re-dated.**
`review_queue_violations` went **0 → 4** at midnight: exactly the four rows
yesterday's sitting left standing as its own bill, on a date it named out loud.
All four are **ACTED** this morning with executing commits, and **none was
moved to a later date** — the first sitting in this file's history to clear its
entire overdue class by answering it. `review_queue_net_arrivals` **8 → 6**,
MOVED −2 and earned (four ACTED against one new row routed; the +1 is the
sliding window and is nobody's act). `--check` **EXIT 0, 0 violations, no amber
date.**

**And the one row due TODAY that is builder work is re-dated before it breaks**
(`39bb22b`): `pass-certificates-are-not-re-evaluated-when-a-dependency-falls`
is *"a small instrument owed by the builder"* by its own first line, and its
owner is provably dark — moved to **09-22**, the first Monday *after* the 09-21
meter reset (09-21 itself deliberately not used: a reset at the start of a week
is not a slot completed inside it). **The three other rows due today —
`t108-noise-floor-is-quoted-by-nobody`, `t211-diayn-metric-cannot-separate-mi-
from-noise`, `five-commitments-are-claim-dead-behind-foreclosures` — are all
desk debt and all three stay on today's date.** If they break at midnight the
break is mine, exactly as this morning's four were.

---

## The honest paragraph

No numbers. We are not closer, and for the second day we were not busier
either — the ladder did not move at all, and nothing that happened in this
repository happened to Jack. What did happen is that a desk, forced to pay four
promises it had let break on purpose, found that one of the things this project
believes about itself is not true: there is a spec about knowledge that has been
lived, standing on a dependency whose life is a random number generator drawing
words out of a list. Nobody lied. The spec that supplies the corpus says
candidly what it does and claims only what it measures; the error was in the
reading of it by the spec downstream, and it would have been discovered at run
time as a control that could not fail. That is the good news and it is small.
The larger thing found this morning is quieter and has been sitting still for
five weeks: the sentence that justifies the entire survival programme names
seven things survival is supposed to earn, and four of them are not written
anywhere an instrument can see. We built a machine in August whose whole purpose
was to notice when the goal names a capability nothing tests, and it cannot see
these four, because it reads bold text and the goal wrote them in a sentence.
The week's most important step toward Jack is that one of those blind spots is
now a row with a date on it. The most concerning drift is unchanged from
yesterday and has doubled in size: this project's output is not a function of
its science or its ladder, and on the evidence of two silent days the thing most
likely to be discovered next is another sentence nobody could measure — because
a desk with nothing to audit but its own files will keep finding them, and a
creature is not built out of findings about files.

---

## FOR THE BUILDER

Ordered, and items 1–6 are all live in `ladder_prompt.md` `1^8`/`2^8`.

1. **Fire `D19`'s NO-FETCH default — two days overdue.** Minutes, zero GPU,
   pre-registered wording. Unchanged from yesterday except that it is worse.
2. **The `T1.08` colab repair, §9d, step (a) spends nothing.** Whole-stdout
   capture first, then one dispatch. **No third dispatch under the unchanged
   mechanism.** Abandoning the lane and reporting single-backend is a legitimate
   outcome. Carry the three-seed disclosure. Unchanged.
3. **`T1.07`'s re-buy, ~0.47 GPU-h, funded by hours that expire Saturday.**
   Unchanged, and now three days from expiry rather than four.
4. **NEW — add `heavy`, `far`, `tiring`, `worth-it` to `coverage.py`'s
   commitment register.** Monotone in the safe direction; no ruling needed.
   **Declare in the commit message that `claim_dead` 4 → up to 8 is the gap
   becoming visible, not a regression.** `T0.21` owes a re-stamp. Do not
   propose anything about `GOAL.md`'s wording — that half is mine.
5. **NEW — `W1.01`/`W1.03` registration order is recorded and NOT yet
   permitted.** `w1-world-edit-window` (DUE 09-18, mine) is the gate. Read the
   order so you do not re-derive it; register nothing until it rules.
6. **When a slot pages out, say so in the journal.** Second asking. This desk
   has now re-dated six of your rows across two mornings on cause inferred from
   a log tail at 06:40. It can keep doing that; it should not have to.
7. **Do not pre-empt the `A4` disposition (`D29`, DUE 09-18), the `T2.10`
   repair design, or `UB.10`'s successor arm.** All three are mine, and today's
   weld-root answer made the third one's shape explicit.

---

## FOR THE OWNER

**1. `D30` — cited, not re-asked (`decide_by` 2026-09-25).** The builder is now
dark **41 consecutive slots** rather than 18, and the two costs yesterday
forecast are now instrument-visible rather than predicted: `D19`'s armed default
is two days overdue and unfired, and `w1-world-edit-window`'s registration — the
gate on the project's only remaining world branch after this morning's
`w100` answer — has no awake executor. 26.51 free GPU-hours now expire in three
days with a legal, authorised buyer. The recommendation on the entry is
unchanged and the desk still recommends **against its own default**. Nothing
new is asked here.

**2. `D28` — cited, not re-asked (`decide_by` 2026-09-21), and the amendment is
now three-for-three.** For the third consecutive day I did not spend my first
act on the overdue class as option (a) as written would require — except that
today the overdue class *was* the right first act, because its four rows were my
own broken promises rather than someone else's backlog. That is the amendment
restated from the other side: *"overdue first UNLESS a perishable resource is
the reason"* would have produced the correct order today and the wrong one on
the two days before it. One amendment, three pieces of evidence, unchanged ask.

**3. NO-DECISION: liveness report, nothing here to rule on.** Three organs alive
and on cadence; the builder's process is alive and its output is zero for the
second day. Field watch wk7 fully consumed, 0 UNROUTED-FIELD-FINDING, next sweep
Mon 09-21. `champions --check` EXIT 0. `review_queue --check` EXIT 0, 0
violations. `decisions --check` EXIT 0. `run steering` 13 orders, 0 illegal.

**4. NO-DECISION: the day's acts, declared because they are mine to do.** Four
broken promises paid by ANSWERING them, none re-dated — the first clean sweep of
an overdue class in this file's history, and it was an overdue class this desk
created on purpose yesterday. One new queue row routed (the four unregistered
primitives). One builder-owned row due today re-dated before it could break;
three desk-owned rows due today deliberately left standing as my bill. One
steering block amended rather than replaced, because its addressee has not read
it. One ratchet recorded and earned. **No threshold moved in any direction. No
ledger row was written by hand. No spec was re-run. No GOAL.md citation was
deleted, though deleting four would have cleared `coverage` rc=2 in one edit.**

**5. A finding you should see before Sunday's Completeness Audit reaches it, and
it needs no ruling from you today. NO-DECISION: reported now because it is the
kind of thing that vanishes, not because there is a fork in it yet.**
`GOAL.md:186-188` names seven primitives survival is supposed to earn — *hot,
heavy, far, tiring, dangerous, worth-it, that-person-lied*. **Two are certified
by a passing claim. One is CLAIM-DEAD. Four — `heavy`, `far`, `tiring`,
`worth-it` — have no entry in `coverage.py`'s commitment register at all**, so
no instrument in this repo can report them missing. This is the 2026-08-09
smell/taste/voice shape recurring *inside* the file built to prevent it, and the
mechanism is mundane: `coverage` parses bolded commitments, and these seven live
in a prose sentence. The instrument repair is ordered to the builder and is
monotone. **The fork — do we commit to falsifiable claims for those four, or
correct `GOAL.md`'s sentence to name only what we intend to test? — is real and
it is yours, but it is not ripe today**: a default may not narrow what this
project has promised itself, so it cannot be armed the usual way, and the
honest sequence is to make the gap visible first and count it. It is on the
queue as `goal-187-names-seven-primitives-four-have-no-commitment` (DUE 09-18),
and if it resolves toward correcting the words it will reach you as a `D` with
the recommendation quoted verbatim.
