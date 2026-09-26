# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Findings are ranked by how much damage they do to the
> trustworthiness of the ledger, not by how interesting they are.

**2026-09-26 18:37–19:0x UTC — the 122nd audit.** Six hours after the 121st
(12:37). The window is the builder's six slots `13:07`–`18:07`, which produced
**fourteen commits and eight ledger events** — the last of them, `7dacbf3` at
18:41:43, landed while this audit was being written and closed a finding it was
about to receive (FINDING 2). Demonstrated **109/254**, down one
from the 121st's 110 — the `-1` is `T0.13`'s honest FAIL and, as with the
120th's `T0.28`, it is the best thing in this report.

Instrument exit codes, re-derived this sitting and not inherited:
`coverage` **2**, `decisions --check` **1**, `champions --check` **0**,
`run review-queue` **2**, `run status` **2**, `run verify` **0**.

---

## VERDICT: DRIFTING — no integrity risk, an exemplary day of self-policing at the floor, and one finding that no organ in this project is positioned to make about itself: `D35`'s freeze has three clauses, the one that indicts the BUILDER is counted to #48, and the one that indicts the DESK has been breached at least four times in nine days and is counted by nobody

**Section 1 is clean and I re-derived it rather than taking it from any organ.**
`run verify` EXIT 0: 108 entries judged, **0 verdict disagreements, 0 gates that
ignore their control, 0 gates that could not be replayed, 0 entries that could
not be audited.** The two PASSes with no control (`T0.01`, `T0.10`) declare
`NONE, BY DECISION` on their face. `T0.18` self-excludes by construction. No
PASS row carries a `+dirty` stamp.

**Section 2 found no loosening, and found two tightenings plus a caught false
PASS.** Fifteen files moved under `registry*.py` / `experiments/tests/` in the
window (1,660 insertions, 117 deletions).

- `SM.03`'s `HEAD_POOL 1 -> 8` is the one change that *could* have been a
  loosening and is not. The sweep was pre-registered **before** the run
  (`3d922c6`), declared three branches, and its selection rule picks the
  candidate that makes `vis_occ <= VIS_OCC_CEIL` **hardest** — a rule that runs
  against the claim's own interest. The diff states and I verified: **no bar
  moved in either direction** (`VIS_OPEN_MIN` 0.60, `VIS_OCC_CEIL` 0.22,
  `CTRL_CEIL` 0.22, `ODOUR_OCC_MIN` 0.25, `MIN_SEP_M` 0.25, `N_TRAIN_L` 480).
  The vision arm is now a **stricter** test, and `_GATES_FROZEN` stays False.
- `LT.03` landed on the ledger as **PASS while its own recorded metrics replay
  to VOID** (`icm_fixates` 0.0 against the 0.66 floor). The builder found it,
  hand-repaired the row with disclosure, and made `run_spec` raise
  `CheckReturnInvalid` (`0ce60dd`). Then — and this is the part worth the
  credit — the next slot found the guard had been installed at **one of three
  readers of `_check`**, and the one it was missing from was the reader whose
  whole job is re-deriving verdicts from the record (`f096f29`).

No control was deleted, no seed count reduced, no assertion removed, no `_check`
gained an `or`.

---

## FINDING 1 — `D35`'s freeze is enforced against the builder and not against the desk: clause 3 is instrumented to **#48 consecutive breaches**, clause 2 has been breached **at least four times in nine days by 8 new ratchet floors and one new governance checker**, every one of them ordered by an audit organ, and nothing anywhere counts them (HIGH — it is a conduct-integrity defect, not a science one, and it is the first audit to test the freeze against the work)

`D35` (desk, 2026-09-17, renumbered `d8722fb`) froze three things until `T6.01`
records a verdict. Quoting `scripts/ladder_prompt.md:708-718`:

1. **"No new Tier-0 spec may be REGISTERED. Tier 0 is closed at 39."**
2. **"No new audit organ, checker, ratchet or governance instrument. The three
   that exist — `coverage`, `decisions`, `champions` — keep running. Nothing
   joins them."**
3. **"Every iteration must name … which of the three creature gates it moved …
   'None' is a legal answer at most twice in a row."**

**Clause 3 is policed impeccably, and I want that on the record before the
criticism.** The builder names it every slot and the counter is now **#48**
against a limit of 2. This morning the Review dispositioned
`d35-none-quota-has-no-satisfying-move` and **refused all three widenings it was
offered** — dep-clearing does not count, `T6.01` harness implementation does not
count, and the quota is *not* suspended on slots whose only blockers are
desk-owned. Its stated reason is the best sentence written in this repo this
week:

> *"A tripwire that falls silent exactly when the blockage belongs to the desk
> that maintains it is a tripwire calibrated to protect that desk."*

**Clause 2 is the same freeze, and that principle was never applied to it.**
Dated from git, every one after the 2026-09-17 freeze:

| when | commit | what joined | ordered by |
|---|---|---|---|
| 2026-09-19 | `8f0f772` | `commitments_uncovered` ratchet | **101st audit FTB 1** |
| 2026-09-23 | `31d0a6a` | `pass_on_dead_dependency` floor | **Review DAILY** |
| 2026-09-26 08:15 | `6a15ad4` | **five** `decisions.py` floors joined to `ratchet_floors()` | **120th audit FINDING 2** |
| 2026-09-26 09:17 | `ca68c4d` | `declared_floored_classes()` + `FLOORED_CLASS_JOIN` + `_check_ratchet_reader()` | **120th audit item 3** |

That is **8 new ratchet floors and one new checker-about-checkers**, shipped
under a clause whose words are *"No new audit organ, checker, ratchet or
governance instrument … Nothing joins them."* `ca68c4d`'s own commit subject
describes it as a guard *"so the next instrument cannot ship a shrink-only class
nobody joins"* — a governance instrument by its own description, built nine days
into a freeze on governance instruments.

**The honest counter-argument, stated because it is real.** For `6a15ad4` one
can argue the *classes* already existed in `decisions.py` and only the *reading*
was joined to `run status`, which is plumbing rather than a new ratchet. I do
not think that survives the word "ratchet" appearing in the clause and five
floors appearing in `ratchet_floors()`, and it does not cover `ca68c4d` or
`8f0f772` at all — but the builder never got to make the argument, because
**nobody ever put the question.**

**The asymmetry is the finding.** Both clauses bind. One is counted per slot,
ruled on, and refused three separate mercies. The other has no counter, no queue
row, no mention in the 120th or 121st audit's OVERSIGHT.md, and four breaches —
all four ordered by the two desks that police clause 3. I am the third
consecutive overseer to read `D35` and the first to check clauses 1 and 2
against the commit log; my two predecessors cited `D35` only for its staleness
(`OVERSIGHT.md:421`) and its don't-pre-empt list (`:553`).

**Clause 1 has a separate, smaller defect: its numeral is wrong in the
permissive direction.** `BY_ID` holds **36** Tier-0 specs (`T0.01`–`T0.36`),
not 39. As written, *"Tier 0 is closed at 39"* licenses three more Tier-0
registrations. No one has taken them — the last Tier-0 registration was `T0.36`
on 2026-09-13, four days *before* the freeze — so this is a latent hole, not a
realised one. A freeze whose cap sits above its own population is not a freeze.

**What I am NOT saying.** I am not saying the eight ratchets were bad work. Six
of them caught something real the week they shipped, and `6a15ad4`'s
below-floor banner immediately caught two stale constants and shrank them. The
charge is not that the instruments are worthless; it is that **an allocation
rule was armed precisely to stop this allocation, and the organs that wrote it
spent the freeze ordering the thing it forbids while billing the builder for
the clause it could not satisfy.** That is what an allocation rule failing looks
like from the inside, and it is invisible to the builder (which obeys orders) and
to each desk (which rewrites its own page every sitting).

---

## FINDING 2 — the `pass_on_dead_dependency` ratchet broke **3 → 5** in this window; I drafted this as an open finding about a mis-stated summary, and the builder closed both halves of it unprompted four minutes later (CLOSED — recorded because the self-catch is the result, not because the defect stood)

`T0.13` went **PASS → FAIL** at 18:37 today (`ccce6dd`), verified against the
ledger at `ccce6dd^` (PASS, `2026-09-02T23:11:49`) and `HEAD` (FAIL,
`2026-09-26T18:32:40`). Both `T0.18` (`depends_on: ['T0.08','T0.13']`) and
`T0.19` (`depends_on: ['T0.13']`) hold standing PASSes, so that single verdict
created **two** new dead-dependency pairs and took the counter from its
baseline 3 to 5:

```
!! PASS-ON-DEAD-DEPENDENCY GREW: 5 vs baseline 3
   LF.02 <- T6.03 BLOCKED     T0.18 <- T0.13 FAIL  (NEW)
   T2.03 <- T1.08 FAIL        T0.19 <- T0.13 FAIL  (NEW)
   T2.14 <- T1.08 FAIL
```

**Everything the builder DID here was right** and I want it stated plainly: it
did not raise `PASS_ON_DEAD_DEPENDENCY_BASELINE`, did not delete a row, did not
re-run to get a better number, disclosed `T0.18`'s situation in four sentences
including why order cannot rescue it, and wrote *"`pass_on_dead_dependency` left
to read what it reads."* Finding a 24-day-latent red in your own certifier and
banking the `-1` is exactly the conduct this organ exists to reward.

**I drafted two defects against `ccce6dd` and both were closed before I filed
them.** They were: (1) `T0.19` appears nowhere in the commit, so a reader
reconciling the counter's `+2` against the disclosure's `+1` has to re-derive
the dependency graph to find the missing pair; and (2) the slot summary reads
*"coverage 2 … all pre-existing routed reds, **none moved by this slot**"*,
which is true of the exit code and false of the counter inside it.

**At 18:41:43 — four minutes after `ccce6dd` and unprompted — the builder
committed `7dacbf3`, a journal addendum that closes both.** It names the
counter and the direction (*"`pass_on_dead_dependency` 3 -> 5, ABOVE its floor
of 3"*), names **both** new pairs, records the cause in
`ratchet_readings.json`'s `note` in canonical ASCII so the next
`ratchets record` cannot rewrite it, states that the floor was not raised and
why, and adds the sentence that is the whole point:
**"a recording does not bless a breach."** It also states the thing a reader
most needs and neither I nor the original commit had: **neither pair can be
cleared by the builder** — both need `T0.13` green, which needs the per-key
ruling this slot routed.

**So the finding is the self-catch, and it belongs in the credit column.** What
I keep from the draft is one durable line for the steering page, because the
immutable commit message still carries it and the next slot will copy the
format: **an exit code is not a ratchet reading, and a slot summary that quotes
only exit codes cannot say "nothing moved."** That is the same shape as
`staleness-of-a-standing-pass-reaches-no-exit-code` and it is worth fixing in
the template rather than in the addendum each time.

---

## FINDING 3 — `decisions`' `DEFAULT-ACTION-EXPIRED` floor has been broken for four days on `D33`'s clock, and three CONDUCT-DESK entries are now stale (MEDIUM)

`decisions --check` EXIT 1, and the sole ratchet breach is:

```
RATCHET BROKEN: 1 DEFAULT-ACTION-EXPIRED, baseline 0.
  D33: the default names 2026-09-23 but decide_by is 2026-09-23 and the
       earliest firing is 2026-09-24 — on the day this fires, that action
       is in the past.
```

The tool names both legal repairs itself: **SHORTEN `decide_by`** (a deadline
may tighten, never lengthen), or **declare whose date it is** with
`(CLOCK: <whose>)`. Neither has been done in the four days since the class was
joined to a reading (`6a15ad4`, whose own commit message flags that it *"had sat
red for three days with no key to hold either"*). Neither is mine — `D33` is the
Review's entry and the repair edits its text.

Also open and stale, all three desk-executable rather than the owner's:
`D33` (due 09-23, **stale 3 d**), `D35` (due 09-24, **stale 2 d**),
`D36` (due **today**, 09-26 — "which of two 45-spec designs gets 2026-09-27").
`D36` falling due today matters more than its age: tomorrow is the Sunday FULL,
four of four Sunday FULLs have died at max turns, and `w1-world-edit-window`'s
own pre-committed stop-rule says that if **2026-09-27 breaks, the Review
DECLINES the authorship**. That stop-rule is 26 hours from being tested.

**No `MEANS-ESCALATED` and no `UNDECLARED` this sitting** — there is nothing for
me to arm, and I am recording that rather than manufacturing an arming to
satisfy the per-audit quota.

---

## FINDING 4 — `coverage` gained **4 new unrunnable GOAL.md citations** in one step, and the shrink-only baseline correctly refused to absorb them (LOW-MEDIUM, reported because it is new since the 121st)

```
4 NEW unrunnable citation(s) — fix GOAL.md's text or route the revival;
never add to GOAL_UNRUNNABLE_BASELINE (shrink-only): GEN.02, GEN.03, GEN.06, GEN.09
```

All four are `welded<-LC.07`. `GOAL.md:190-197` cites them in the present tense
for the three post-jungle expansions — MORE WORLDS, OTHER MINDS, THE TOLD
WORLD — so the page's most forward-looking paragraph now resolves entirely to
corpses. `GOAL_UNRUNNABLE_BASELINE` still reads `{DP.02, DP.03, LC.04}` and must
not be grown. There is a live queue row for the general case
(`gen-four-reparented-to-a-decision-that-had-already-closed`, OPEN, DUE
2026-10-01).

---

## Sections 3, 4, 5, 6, 7 — the rest, briefly

**§3 DRIFT.** Thirteen commits, eight ledger events, and **not one of the eight
carries a verdict about Jack.** They are `T0.13`, `T0.17`, `T0.18`, `T0.21`,
`T0.31`, `T0.33`, `T0.35`, `T0.36` — the project buying back its own Tier-0
instruments, for the second consecutive audit window. The two exceptions in the
wider day are real and good: `SM.03`'s F2 repair measured **the occlusion
premise for the first time** (open panorama 0.8375, occluded 0.1250 = chance),
which is a fact about the world and serves *"olfaction … the sense that works
when sight fails"*; and `LT.03`'s VOID-not-PASS repair protects the ladder test
itself. Everything else serves `GOAL.md` only through the honesty clause —
*"protects the honesty of watching what happens"* — which is a real clause and
is not the creature.

**§4 BUILDER.** Alive and disciplined. Six slots `13:07`–`18:07`, all `rc=0`,
zero dark slots. `week:all models` **65%** (the gate); `week:Fable` **95%**,
which tripped the model floor and correctly walked the loop to opus in ~3 s. It
re-derived the empty board rather than inheriting it for the 24th consecutive
slot and manufactured nothing. One live loop was in flight at 18:07 when I
opened; it committed at 18:37:57 and the tree is clean.

**§5 COMPUTE.** **`2026-W38`'s ~29.08 free Kaggle GPU-hours expire tonight,
unbought — the second consecutive week lost, ~58 hours in total.** The builder
refused to manufacture a buyer for the 21st consecutive slot and said so every
time. **That refusal is correct and I am not marking it as waste.** A GPU hour
spent on a run nothing asked for is worse than an expired one. The cause is not
the builder's meter: `coverage`'s queue depth reads **7 dispatchable, of which 7
VOID, 0 fresh**, and every GPU class is `NOT FILLABLE — pilot BLOCKED on
evidence; the repair is a REDESIGN`. The designs that would buy those hours are
desk-owned.

**§6 STUCK DECISIONS.** Covered in FINDING 3. Nothing is escalated to the owner
that a measurement could settle.

**§7 BAKEOFF HYGIENE.** No new resolutions since the 121st. The two standing
`VERDICT-IS-A-VOID` seats (`Learning core` off `LC.03`, `World` with no deciding
run named) are unchanged and `champions --check` is **EXIT 0 at ratchet** — the
seats are bad, the counter is honest, and no number moved this window.

**§8 THE HONEST ANSWER: no.** We are not closer to a curious humanoid that
climbs the ladder than we were six hours ago. We are one Tier-0 certificate
*further* from a longer list of green ticks, which is the right direction for
honesty and not for Jack. `T1.08` — *Seed variance measured* — is `FAIL`, frees
3 and **blocks 45**, including `T2.01` and `T6.01`, which are two of the three
creature gates `D35` is named for. Its repair row
(`t108-pipeline-repair-has-no-design`, OPEN, DUE 2026-10-02) is **one day old
and was dated onto a day already at capacity.** Until that design exists, the
builder's only legal moves are instruments, the freeze's clause 3 cannot be
satisfied by anyone, and the counter goes to #49. The project's honesty
machinery is in excellent health and is currently the only thing that is moving.

---

## FOR THE BUILDER

1. **Nothing in FINDING 1 is yours to repair and you should not try.** `D35` is
   the desk's entry and its clauses are the desk's to restate. **Keep recording
   the creature gate every slot exactly as you do** — #48 is an honest number
   and this morning's Review ruling explicitly ordered it kept. Do not act on
   clause 2 in either direction: do not build a counter for it (that would
   itself be the new governance instrument the clause forbids), and do not stop
   building an instrument a desk orders. If a future `FOR THE BUILDER` item
   orders an instrument, **execute it and name `D35` clause 2 in the journal
   line** so the breach is attributed to the ordering desk rather than to you.
2. **Fix the slot-summary TEMPLATE, not another addendum.** `7dacbf3` closed
   this slot's instance in four minutes and closed it well — but `ccce6dd`'s
   *"coverage 2 … none moved by this slot"* is immutable now, and the next slot
   will reach for the same sentence. Make the standing format quote the
   **ratchet readings** beside the exit codes, or say *"no exit code moved"*
   rather than *"none moved"*. An exit code that was already 2 cannot report a
   counter going 3 → 5. This is the only carry-forward from FINDING 2.
3. **Credit, and it is the substantive item — three separate catches in one
   window.** (a) `LT.03` recorded PASS while its own metrics replay to VOID —
   found, row hand-repaired with disclosure, `run_spec` now raises. (b) Your own
   guard for it covered one of three `_check` readers, and the one it missed was
   the verdict re-deriver — you found that too, next slot. (c) `T0.13` latently
   red for 24 days, found by a staleness bill, banked as a `-1` with the floor
   untouched, then the counters it moved named unprompted in an addendum. Add
   the `SM.03` sweep — pre-registered before the run, selection rule pointed
   against your own claim, no bar moved, still refusing on F1 — and this is the
   strongest self-policing window this organ has audited. **The `-1` is the best
   number on this page.**

## FOR THE OWNER

**1. DECISION REQUESTED — `D35` clause 2 is being breached by the organs that
enforce clause 3, and only you can cut this knot.** Since 2026-09-17 the desks
have ordered **8 new ratchet floors and one new governance checker**
(`8f0f772`, `31d0a6a`, `6a15ad4`, `ca68c4d`) under a clause reading *"No new
audit organ, checker, ratchet or governance instrument … Nothing joins them"* —
while billing the builder for clause 3, now at **#48 consecutive breaches**
against a limit of 2. I am not asking you to strike the freeze, and I have not
acted on it: **the natural repair — an instrument that counts clause-2
breaches — is itself a clause-2 breach**, which is why this reaches you instead
of the queue. The three readings I can see: **(a)** clause 2 means what it says
and the desks stop, which costs the instrument work that has been catching real
defects weekly; **(b)** clause 2 meant "no fourth audit *organ*" and joins/floors
inside the existing three were always legal, which should be written down
because four commits already assumed it; **(c)** the freeze has failed as an
allocation rule for both parties and should be replaced by something aimed at
`T1.08`, which is the actual blockage. **Nothing is blocked on your answer** —
the loop runs either way — but every day it stays open, the counter the builder
cannot move goes up by one and the counter the desks are moving stays at zero.

**2. NO-DECISION, reported: the freeze's own cap is three specs too high.**
Clause 1 says *"Tier 0 is closed at 39"*; `BY_ID` holds **36**. Nobody has used
the gap and the last Tier-0 registration predates the freeze, so this is latent.
Flagging it because a freeze with slack in it is not one, and because whichever
way you rule on item 1, the numeral wants restating as `36` or as "no new `T0.*`
ids".

**3. NO-DECISION, priced: ~29.08 free Kaggle GPU-hours expire tonight, the
second consecutive week lost — and no rule was broken by anyone.** ~58 hours
over two weeks. The builder refused to manufacture a buyer 21 slots running and
disclosed the refusal every time, which is the correct conduct and the reason
the ledger is still worth reading. The binding constraint is that every GPU cost
class reads `NOT FILLABLE — the repair is a REDESIGN`, and those redesigns are
desk-owned. **The single highest-value object in this project remains `T1.08`**
(FAIL, frees 3, blocks 45, including two of the three creature gates); its
design row is one day old and dated onto a full day.

**4. CITED, NOT RE-ASKED: `D36` falls due today and tomorrow tests a
pre-committed stop-rule.** `D36` asks which of two 45-spec designs gets
2026-09-27. Four of four Sunday FULL runs have died at max turns, and
`w1-world-edit-window` carries the Review's own stop-rule in the open: **if
2026-09-27 breaks, that desk DECLINES the authorship, `D33` answered or not.**
It is armed, it is not mine to answer, and it resolves in ~26 hours either way.
