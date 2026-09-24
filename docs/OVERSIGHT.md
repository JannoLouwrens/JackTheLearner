# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Findings are ranked by how much damage they do to the
> trustworthiness of the ledger, not by how interesting they are.

**2026-09-24 06:37–07:0x UTC — the 112th audit.** Six hours after the 111th
(00:37). The window is the builder's six live slots 01:07–06:07 (`8762982`,
`9001e6a`, `2accfbb`, `6681c9e`, `3511624`, `71f9d53`, `2b58c90`, `6a7038d`) and
the Review's 06:37 DAILY, which ran **concurrently with this audit** and
committed five times while I was reading (`746865e`, `94ab003`, `ad44e13`,
`f47bcc5`, `f7abc08`). **Every instrument in this report was re-run against the
tree AFTER the Review's last commit**, and the readings below are the post-Review
ones.

---

## VERDICT: ON TRACK — but the entry that gates this project's largest unblock cannot resolve itself in either direction, and the instrument class built for exactly that shape is blind to it by one `continue`

The ledger is sound and I say so with numbers rather than adjectives. `run
verify` re-judges **110 PASS entries: 0 verdicts that no longer re-derive, 0
gates that ignore their control, 0 that could not be replayed.** All 110 PASS
commits still resolve in git. **Not one threshold moved in the loosening
direction in seven days** — the only two constants that moved at all moved the
hard way (`RANDOM_DWELL_MAX` 0.02 → 0.0185, `N_PROPERTIES` 18 → 19 → 20), zero
assertions were deleted against seven added, and no `_check` gained an `or`.
Section 2 has nothing to report and that is a real result.

Both organs did their jobs. The builder ran six clean slots, **manufactured
nothing** against a genuinely empty board, and discharged all four of the 111th
audit's code items. The Review disposed **all six OVERDUE rows** in one sitting —
three `ACTED` against commits already on disk, three re-dated with causes that
are new today — taking `review_queue_violations` **6 → 0**.

What I found is one defect in `experiments/decisions.py`, the module behind this
organ's SECOND mandated check. `DEFAULT-ACTION-EXPIRED` exists because `D21`
armed a clock whose commanded action was already in the past. Its one live
successor — `D33`, the entry `D21`'s own failure produced — is in **exactly that
state today**, and the check never runs on it, because the `class == "conduct"`
branch `continue`s seventeen lines above where the check sits. `decisions
--check` prints **`0/0 default-action-expired`** over it. **The 110th audit found
that same `continue` yesterday and wrote a lesson about it; the repair it ordered
fixed the one symptom it had looked at and left the branch's exemption from every
other check intact.**

---

## RANK 1 — `DEFAULT-ACTION-EXPIRED` CANNOT SEE A CONDUCT ENTRY, AND `D33` — THE ENTRY GATING THE WORLD EDIT — HAS BEEN IN THAT STATE SINCE MIDNIGHT WITH A GREEN `0/0` PRINTED OVER IT

### The defect, in the module's own control flow

`experiments/decisions.py`, inside the per-entry loop:

| line | code | effect |
|---|---|---|
| **1290** | `if cls == "conduct":` | conduct branch entered |
| 1291–1307 | compute `overdue`, append `CONDUCT-DESK` | the only thing a conduct entry is ever checked for |
| **1309** | `continue` | **leaves the loop body** |
| 1311 | `NO-DEFAULT` check | never reached for conduct |
| **1326** | `stale = expired_actions(d["default"], due)` | **never reached for conduct** |
| 1342 | `race = same_day_actions(...)` | never reached for conduct |

So for any entry with `class: conduct`, the default's prose is **never compared
to its own `decide_by`**. The class is ratcheted (`RATCHETED[
"DEFAULT-ACTION-EXPIRED"] = BASELINE_ACTION_EXPIRED = 0`, `decisions.py:664`,
`:1446`) and it reads `0/0` — at floor, green, on the morning the condition is
live.

### The live instance, demonstrated with the module's own function

`D33` (`docs/DECISIONS_NEEDED.md:81`, `class: conduct`, `decide_by: 2026-09-23`)
pre-registers `default: (i) RE-DATE ONCE MORE, TO 2026-09-23, AND CHANGE NOTHING
ELSE.` I ran the module's own detector against the module's own parse of the
live file, read-only:

```
id D33  class='conduct'  decide_by='2026-09-23'
earliest firing day  : 2026-09-24   (today is 2026-09-24)
expired_actions()    -> [datetime.date(2026, 9, 23)]
same_day_actions()   -> []
```

**`expired_actions` returns the violation.** The module knows. `main()` never
asks it, because line 1290 returned first. By the file's own stated arithmetic
(`decisions.py:230` — *"the earliest day a default can fire is `decide_by + 1`"*),
`D33`'s default names an action that is in the past on **every** day it could
ever fire. That is the class docstring's founding scar verbatim: *"armed, dated,
legal in every field, and incapable of doing the thing it promises."*

### This branch was already found once, yesterday — and the repair fixed the symptom it saw and left the exemption standing

This is the part that makes it RANK 1 rather than a missed corner.
**`docs/LESSONS.md:16541` is a lesson about this exact branch, written by the
110th audit on 2026-09-23:** *"A checker's early `continue` makes a whole class
structurally incapable of going red — and the comment on that branch will say the
opposite."* Its live instance was `D33`. Its ordered repair was *"usually one
comparison, not a new class"*, the builder shipped exactly that the same day
(`eca5757` — `CONDUCT-DESK` now computes `STALE by N day(s)` against today), and
the repair is correct: the line now varies, which is what the audit asked for.

**But the `continue` is still there, three lines below the comparison that was
added.** The 110th audit asked *"what does this branch PRINT, and what could make
its output CHANGE?"* — and got a true answer. It did not ask **what does this
branch SKIP**, and the answer to that is every check between line 1311 and the
end of the loop body: `NO-DEFAULT`, `DEFAULT-ACTION-EXPIRED`, and the same-day
race. So the class was measured once, repaired once, and its structural exemption
survived the repair intact. `D33` — the same entry, one day later — is now sitting
in the half that was never looked at. **A lesson about an early `continue` was
written on the morning a different consequence of the same `continue` came due.**

### Why the fixture could not catch it either

**The fixture cannot catch it either.** `T0.28` is the certificate that
`decisions.py` "detects every defect it claims to detect" over 20 properties —
the gate under this organ's second mandated check. Its `DEFAULT-ACTION-EXPIRED`
property (`decisions.py:1573–1584`) builds its `D21` replay with **`class:
goal`** and asserts on that. The conduct lane is exercised only by P19, the soft
conduct *advisory*. **No property in this repo asserts what a conduct entry's
expired default should do**, so the blind spot passes its own test suite — which
is the disease `exit_code` was extracted in `coverage.py` to stop, one module
over.

**`D33` is the single highest-leverage entry on any desk.** Its own `blocks:`
field: `W1.01`/`W1.03` registration, `SH.02`'s adopted arm (b),
`ba03-vestibular-channel-is-never-load-bearing-under-one-kick`, and
`ne01-occlusion-knife-edge` + `water-apply-phantom-force` (both `HELD` on it) —
and behind those the world edit, which is the upstream of all four CLAIM-DEAD
constitutional commitments in §3 below. It cannot be fired (the action is dead),
it has not been resolved (`grep D33 docs/DECISIONS_RESOLVED.md` → nothing), and
its `decide_by` has not moved. **It is a deadlock with a clock painted on it, and
the counter for that state reads zero.**

**It has already changed a real decision, today.** `f47bcc5` (06:4x) re-dated
`w0-too-shallow` — whose execution debt IS the `W1.01`/`W1.03`/`W1.04`
registration, 18 days unregistered — to **2026-10-01**, on the stated ground
that handing it to the idle builder *"would pre-empt the D33 ruling this desk
itself asked for."* And `docs/PROGRESS.md` FOR THE OWNER 2 reports that `D33`'s
default *"has fired for a fourth time."* **A conduct entry has no armed default
to fire**; `decisions.py` does not arm it and does not fire it — it prints
`CONDUCT-DESK … execute it, report it, do not ask`. No firing block was written
anywhere, and the desk's actual re-dates were to **09-27** and **10-01**, not to
the 09-23 the default names.

**I want to be exact about what is and is not wrong here.** The Review's *acts*
were legal and well disclosed: a desk re-dating its own rows in the open, with
written causes and a stop-rule against itself, is permitted and was done
carefully. What is broken is the machinery around it — a guard that cannot fire,
a fixture that cannot catch it, and a vocabulary ("the default fired") describing
something that structurally did not happen and left no record. Nothing in the
ledger is unsound as a result. What is unsound is the belief that this entry is
on a clock.

---

## RANK 2 — `T4.06`'s WINNER RESTS ON **TWO** ANCHOR-RELATIVE CONJUNCTS; THE READER BUILT YESTERDAY TO REPORT THEM DESCRIBES **ONE**, BY CONSTRUCTION — AND ADOPTION IS TOMORROW

`T4.06` (PASS attempt 1, the window's only capability event, 109 → 110) is the
fusion-balancing bakeoff. Its pre-registered winner rule
(`experiments/tests/t4_06_fusion_balancing_bakeoff.py:91–99`) has four conjuncts.
**Two of them are measured against the incumbent arm, not against an exogenous
bar:**

```
513:    r2_ok    = s["min_r2"] > bar_r2          # (2) strictly exceeds the incumbent
514:    loss_ok  = s["eval_loss_mean"] <= inc_eval  # (3) non-strict — ZERO required margin
```

`experiments/resolution.py` shipped **yesterday** (`3e9cf42`) to print exactly
these margins. Its table is keyed one-per-spec and says so:
`resolution.py:169` — *"**One entry per spec**, copied from the spec's own
pre-registration."* `ANCHOR_CONJUNCTS["T4.06"]` names `min_modality_latent_r2`
and nothing else, and the `UNDESCRIBED` lane fires only when
`ANCHOR_CONJUNCTS.get(sid) is None` (`:224`) — **per row, never per conjunct**.
A row with one of N conjuncts described therefore reports as fully described.

### The conjunct nobody prints, computed from the committed row

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| incumbent `eval_loss` | 0.4842 | 0.4943 | 0.4900 | **0.4895** |
| `loss_reweight` (winner) | 0.4831 | 0.4936 | 0.4891 | **0.4886** |

Margin **+0.0009** against the incumbent's **own seed-to-seed spread of 0.0101**
— **8.9% of it.** For comparison, the disclosed conjunct (2) was +0.0187 against
a spread of 0.2699 — **6.9%**. **Both deciding anchor-relative conjuncts on this
certificate sit under 10% of the anchor's own noise, and `run status` prints one
of them.** The certificate's amendment caveat (`f7900b5`, ledger `amended` block)
names conjunct (2) only.

Two things keep this RANK 2 rather than RANK 1, and both are to the builder's
credit. **The verdict does not depend on either weak conjunct.** The decisive
work is done by the *exogenous* 10× ratio gate: `ratio_worst` 29.83 → **2.45**,
every seed dominating, a 12× result the Review pre-registered and did not move.
And conjunct (3)'s paired-seed agreement is **3/3 improving** (−0.0011/−0.0007/
−0.0009), better than conjunct (2)'s 2/3; its refutations were decisive
(`grad_norm` −71% of spread, `modality_dropout` −181%), so it did discriminate.

But `<=` at zero required margin is the same shape the Review **strengthened out
of `T2.06` four days earlier** — `acad758`, 09-20: *"CLAIM 2 gets an exogenous
margin — a strict `>` was deciding at zero margin."* The same shape shipped into
`T4.06` on 09-23 and the reader built the same day cannot see it. **`T4.06`'s
adoption is dated 2026-09-25 — tomorrow.** Whoever adopts should see both
numbers, not one.

---

## RANK 3 — THE FOUR `GEN` CITATIONS STILL HAVE NO OWNER: 23rd DAY OF `coverage` rc=2, AND THE SITTING THAT WAS ASKED TO RE-OWN THEM RAN AND DID NOT

Carried from the 111th audit's RANK 2, re-verified against the post-Review tree
rather than restated. `coverage` still prints:

```
4 NEW unrunnable citation(s) — fix GOAL.md's text or route the revival;
never add to GOAL_UNRUNNABLE_BASELINE (shrink-only): GEN.02, GEN.03, GEN.06, GEN.09
```

`GOAL.md` cites all four in the present tense as generality claims being tested.
All four are `welded<-LC.07`, whose venue `D24` declared **unaffordable** on
09-12. The row that held them (`goal-cites-four-specs-that-resolve-to-corpses`)
re-parented them on **09-16** to *"whoever closes `D24` inherits these four"* —
four days after `D24` closed — and was stamped **ACTED**, which is terminal.

**What is new today:** the 111th audit put this in `FOR THE REVIEW` at 00:37 with
the words *"what is owed is a live parent."* The Review's 06:37 sitting ran, made
five commits, and **no `GEN` row was routed, no `DECLINED` written, and no owner
entry opened** — `grep -i "gen\.0[2369]" docs/REVIEW_QUEUE.md` returns only the
two historical mentions. This is a fair call on the Review's part (its sitting
spent its clock on the OVERDUE sweep, which was the higher-ranked order), so I
carry it rather than indict it — but the number is now **23 days red with nobody
holding the repair**, and it is the second morning it has been ordered.

---

## The audit, item by item

### 1. Integrity of the ledger — NO FINDINGS, and the scan is stated so it can be checked

I scanned **all 110 PASS rows** independently of `run verify`: implementation
present in `experiments/tests/`, recorded `commit` still resolving via
`git cat-file -e <sha>^{commit}`, `Spec.control` declared, and `control_metrics`
present on the row.

- **110/110 commits resolve in git.** 0 dangling.
- **108/110 carry a declared control AND recorded `control_metrics`.**
- **2 carry `control: NONE, BY DECISION`** — `T0.01` (an import either raises or
  it does not) and `T0.10` (a remote service returning real bytes is its own
  falsifier), both under the 52nd audit's B5 ruling, both with the reason written
  into the `control` field itself rather than left blank. `run verify` reports the
  same two under *"PASSes with NO control at all"* and I agree with the ruling.
- `run verify` EXIT 0: **0 verdicts that no longer re-derive, 0 gates that IGNORE
  their control, 0 that could not be replayed, 0 controls run but undeclared.**
- **0 PASS rows carry a `dirty_files` stamp.**

`run status` EXIT 0 with **zero PASS rows in the STALE block** — the 111th
audit's `T0.28` bill was paid at 01:10 (`2accfbb`, attempt 22, 56.34 s, clean at
`6ebfa9e`, `impl_sha 3543b4eebff2abe7` matching the live tree).

### 2. Thresholds and controls over seven days — NO LOOSENING. Reported plainly.

I diffed `registry.py`, `registry_expansion.py` and `experiments/tests/` over
seven days and extracted every constant appearing on **both** sides of the diff
with a changed value. Four, and not one is a loosening:

| constant | from | to | direction | justification |
|---|---|---|---|---|
| `RANDOM_DWELL_MAX` | 0.02 | **0.0185** | **TIGHTER** | `875caf6` — cap DERIVED from n rather than typed off a 16-life pilot |
| `N_PROPERTIES` | 18 → 19 | **20** | **TIGHTER** | more properties asserted (`1f32522`, `eca5757`, `9001e6a`) |
| `CORPUS_ROOT` | LibriSpeech | VCTK | venue, not a bar | `9663f23`, declared **before** the run with three pre-registered outcomes |
| `MANIFEST` | LibriSpeech | VCTK | venue, not a bar | same commit |

**0 assertions removed; 7 added.** No `_check` gained an `or`. No seed count
reduced (`SEEDS = [0,1,2]` everywhere touched). `955b9ef` amended `HR.1`'s
registry text **after** attempt 4 recorded FAIL — I read that diff line by line
because a post-hoc hypothesis edit is the shape I most distrust, and the claim
"no bar, seed, probe or control touched" holds: `LEAK_EXCESS` 0.05,
`CONTROL_FLOOR_EXCESS` 0.15, 17 features, 3 seeds all unmoved; the amendment
marks itself *"recorded AFTER attempt 4"* on its face, keeps the original notes
verbatim below, and moved `spec_sha` so the change is visible. Its one code
change made the headline `min_channel_leak_margin` **be** the worst-seed value
`_check` gates on, instead of the seed-mean that had been reported under a `min_`
name — a reporting **tightening** of 12×. `HR.1` is FAIL; no certificate rests on
any of it.

Also verified: `9001e6a`'s self-certified claim *"stale_cost.py sits in no spec's
IMPL_DEPS"* — `grep -r stale_cost experiments/tests/` is empty. It holds.

### 3. Drift from the goal — no drift outward; the hole is inward and unmoved

Everything the builder touched in the window traces to a GOAL.md sentence:
`T0.28`'s re-buy and `stale_cost`'s pre-edit pricer serve *"protects the honesty
of watching what happens when the three meet"*; `T4.06` serves *"all of it
processed together in ONE model … every sense is load-bearing"* directly. **No
work in the window serves no sentence.** Six of the eight commits are journal
lines for slots that correctly produced nothing, which is conduct, not drift.

The converse question is the one that matters and the answer has not improved.
**Four constitutional commitments have no live falsifiable claim at all** —
`smell`, `balance`, `shelter/building`, `thermal (kills)` — every claim spec
behind them PARKED or FORECLOSED on honest evidence. `NO-LIVE-PATH` reads **7
distinct commitments/seats** with no way in. And the commitments GOAL.md warns
are *"most likely to be quietly neglected"* read: **curiosity 2 passing of 12
specs; one brain / unison 1 of 28; hearing 1 of 14; sleep 0 of 5; fast/slow 0 of
8.**

One number deserves stating because it is easy to misread the week as progress:
`T4.06` is credited by `coverage` as **support, not as a claim** (`COVERS: one
brain / unison (rule)`). The ladder went 109 → 110 and the commitment it touches
still has **1 passing claim of 28 specs.** That declaration is honest and
conservative, and I am not asking for it to be changed.

### 4. Is the builder alive and productive? — alive, honest, and correctly idle

**6 iterations in the last 6 h, 6 × `rc=0`, 0 failed slots, `dark_slots 0`,
`hours_since_rc0 0.55`.** Over 24 h: **5 ledger events** — `T4.06` PASS a1
(10:46), `HR.1` FAIL a4 (17:11), `T0.36` PASS a20, `T0.21` PASS a22, `T0.28` PASS
a22 — PASS delta **+1** (109 → 110), three of the five being re-buys of
certificates staled by instrument edits. The 26-slot blackout the 107th–111st
audits tracked is **over**: `week:all models` **19%** at 11% of the week elapsed.

Seven consecutive slots journalled "the board is honestly empty," which is
exactly the claim an overseer should distrust. **I re-derived it and it is true.**
`run next`: **0 fresh** of 48 ready (33 carrying a settled verdict, 15 held).
`coverage`'s queue depth: **4 dispatchable, all 4 VOID → 0 fresh**, and every one
of the seven cost classes is `NOT FILLABLE`:

```
cpu<1min 0 EMPTY · cpu<10min 0 EMPTY · cpu<2h 3 (LF.01,PL.02,SO.07 — all VOID)
cpu<48h 0 EMPTY · gpu<20min 0 EMPTY · gpu<2h 1 (UB.10, VOID) · gpu<8h 0 EMPTY
```

Five of the seven say *"pilot BLOCKED on evidence; the repair is a REDESIGN."*
**A redesign is the Review's unit of work, not the builder's.** The builder is
not the constraint and refusing to manufacture a dispatch seven times running was
the correct call each time.

### 5. Compute honesty — clean this week; the loss is perishable, not spent

`2026-W38`: **0.9176 h drawn of 30 free Kaggle hours, 2 jobs** — and both bought
something. `T4.06` drew **0.4387 h** against a declared projection of 0.75 h and
produced a PASS. **~29.08 h expire at the Sunday reset, end of Saturday
2026-09-26 — two days out, with no legal buyer** (see §4: two of three GPU
classes empty with no path in, the third holding only a VOID). Historical
unattributed spend is at floor and unchanged: `gpu_unattributed_jobs` **21**
(6.32 h), `gpu_hours_no_verdict` **48.42 h total**, of which **`D1.0` alone is
33.78 h across 2 attempts and 0 verdicts**, behind `T1.08`. No overruns recorded.
**Nothing was spent this week with nothing to show for it.**

### 6. Stuck decisions — RANK 1 is here; plus three entries on the owner's desk that block no spec

`decisions --check` EXIT 0, `ratchet ok (0/10 undeclared, 0/3 unrouted-owner-ask,
0/0 vanished-owner-ask, 0/0 default-action-expired, 0/0 firing-diff)`.

- **`MEANS-ESCALATED`: 0.** No fork a measurement could settle is on the owner's
  desk. The `D1` disease is absent.
- **`UNDECLARED`: 0.** Nothing to arm this audit — the ratchet is at floor and I
  am not inventing an entry to have something to arm.
- **`OVERDUE — DEFAULT IS DUE TO FIRE`: 0.** Nothing to fire.
- **Armed: `D31` (09-25), `D32` (TODAY), `D34` (TODAY).** All three fall due
  within two days and **all three block zero spec ids.** Each is flagged
  `CONDUCT-MISFILED?` by the instrument: *"class `goal` but blocks no spec id — if
  this is about how the ORGANS work rather than what Jack must BECOME, reclass to
  `conduct`."* Each default's own text says it *"buys VISIBILITY and nothing
  else."* I am **not** re-lodging this as a finding: the **101st audit** already
  made it (*"the owner's desk is 100% items our own instrument calls the desks'
  paperwork"*), and re-discovering a standing condition as news is its own
  failure. It is in §FOR THE OWNER as a date warning, not a new indictment.
- **`D33`: `CONDUCT-DESK`, STALE by 1 day — RANK 1.**

Was any owner decision quietly acted on without being recorded? **One, and it is
the RANK 1 finding**: `D33` is described in `docs/PROGRESS.md` as having had its
default *"fire for a fourth time"* with no firing block, no
`DECISIONS_RESOLVED.md` entry, no `decide_by` change, and re-dates to 09-27 and
10-01 rather than the 09-23 the default names. Nothing illegitimate was *done* —
the underlying re-dates are a desk's own permitted move — but the record does not
match the vocabulary.

### 7. Bakeoff hygiene — one winner inside the noise, disclosed once and reported once; see RANK 2

`docs/DECISIONS_RESOLVED.md`, 33 entries. The most recent is `D29`, resolved by
armed default 09-23 — `(iii) RECORD THE DEBT`, the weakest option on its list,
chosen because its premise had died under its deadline. Correct handling of a
bad-premise firing, and `champions --check`'s `UNVERIFIED-VERDICTS` was verified
2/2 before and after so the firing paid itself no greener number. **No VOID is
treated as a verdict in any entry I could find**, and the Learning-core seat's
`LC.03=VOID` is carried openly as `VERDICT-IS-A-VOID` in `champions --check`
rather than laundered.

The one winner-inside-the-noise is `T4.06`'s, and **`T4.06`'s own docstring says
so, at length, against its own certificate** (`t4_06…py:44–62`: *"the
latent-recovery conjunct certified the winner INSIDE the anchor's noise and must
not be quoted as demonstrated"*). That is the right behaviour and I want it on
the record as such. RANK 2 is not that the margin is small — it is that a
**second** conjunct of the same kind on the same row is reported by nobody.

### The architecture can still lose — `champions --check` EXIT 0, ratchet at floor, and two seats frozen

`ratchet ok (0/0 phantom arena; 2/3 unfalsifiable; 2+1/4 uncontestable;
4/4 unwinnable; 2/2 unverified verdicts; 3/3 trigger debt; 1/1 kindless)`. The
`ARENA-MISSING` class this organ's prompt records at **8 seats** is now **0** —
that ratchet was paid down by registration, not by deletion, which is the repair
working as designed.

What has not moved: `champions_trigger_debt` **3 since 09-03 (21 days)** and
`champions_unwinnable` **4 since 09-13 (11 days)**. The **World** seat is still
held **BY VERDICT** — the file's strongest marking — with **no `VERDICT:` and no
`TRIGGER:` declared at all**, so the seat behind this project's largest standing
scientific result cannot be contested by any evidence. `Fast/slow coupling` is
`ARENA-UNREACHABLE` rooted at `LC.03`. Both belong to the `D33` docket and I am
not acting on them for the same reason the last three audits gave: acting would
pre-empt the authorship question `D33` is supposed to settle. **That reason is
now itself suspect — see RANK 1 — which is why RANK 1 is RANK 1.**

### Did the routed work move? — yes, more than on any day this month

`run review-queue` EXIT **0** (was EXIT 2 with 6 OVERDUE when this audit opened).
**36 OPEN, 3 HELD, 15 DISPOSITIONED, 22 ACTED, 0 DECLINED of 76 routed.**
Throughput over 7 cycles: arrived 15, **disposed 8 (1.14/cycle, up from 5)**,
designed 8. `drain` still **UNBOUNDED** with **54 live rows** and oldest live
31 d. `DECLINE` remains **unused across all 76 routed rows** — the honest
asterisk the Review itself keeps writing.

`IMMINENT` reads **13 live dated rows due on or before the next cycle against a
measured capacity of 6**, with 7 rows sharing **09-25** — one of which is
`T4.06`'s adoption (RANK 2). The tool calls that a metric and not a violation and
I agree; it is printed here because it is the day RANK 2 lands on.

---

## FOR THE BUILDER

1. **RANK 1's repair, and price it before you make it.** In
   `experiments/decisions.py`, run the `DEFAULT-ACTION-EXPIRED` and
   `same_day_actions` checks for **every** class — either move them above the
   `if cls == "conduct":` branch at line 1290, or run them before the class
   fan-out. **Leave `CONDUCT-DESK` exactly as it is**; it is correct and it is
   the 110th audit's repair working. `decisions.py` is `T0.28`'s **sole**
   `IMPL_DEPS` entry, so this stales a standing PASS: **price it in the pre-edit
   lane and name the bill in the commit message**, the way `9001e6a` did and
   `eca5757` did not. Read `docs/LESSONS.md:16541` and its 09-24 UPDATE first —
   that lesson is about this same `continue`, and the repair you shipped for it on
   09-23 was correct and left the exemption standing. **While you are in there,
   enumerate every violation kind appended after line 1309 and justify the
   conduct exemption per kind, or move the class-independent ones above the
   fan-out.** `NO-DEFAULT` is also unreachable for conduct today; say in the
   commit whether that one is deliberate.
2. **Expect `decisions --check` to go to EXIT 1 with `default-action-expired
   1/0`, and DO NOT repair it by touching `BASELINE_ACTION_EXPIRED`.** That
   constant is 0 and shrink-only. The live 1 is `D33` and **its discharge is the
   Review re-arming `D33`, not a baseline bump.** Raising it would be the exact
   move `T0.31` exists to forbid — a repair that lowers its own number. Say in
   the commit message that the new red is real and whose it is.
3. **Add the property that would have caught it.** One case is enough and it is
   the one the class docstring already writes in prose: an entry with
   `class: conduct` whose `default` names a date `<=` its own `decide_by` must be
   reported `DEFAULT-ACTION-EXPIRED`. The existing `D21` replay
   (`decisions.py:1573`) is `class: goal` — copy it and change one word. **Prove
   it load-bearing by running the committed pre-fix code against the new
   fixture** (the standard `5b18cd3` set and the 111th audit's item 3 repeated).
4. **RANK 2's repair, reporting-only.** In `experiments/resolution.py` make
   `ANCHOR_CONJUNCTS[sid]` a **list** of descriptors and fire `UNDESCRIBED`
   **per undescribed anchor-relative conjunct**, not per row — the comment at
   `:169` (*"One entry per spec"*) is the defect, written down. Add `T4.06`'s
   conjunct (3): `stat="eval_loss_per_seed"`, `decided_at="mean"`,
   `higher_is_better=False`, `label="eval_loss_mean"`, and mark it **non-strict
   (`<=`, zero required margin)** in the printed line, because that is the part a
   reader cannot infer. **No cutoff, no verdict, no exit-code change, no
   ratchet** — same contract the module already declares. Extend
   `resolution.py`'s selftest to assert both T4.06 conjuncts render.
5. **Do not touch `T4.06`, its gates, or its certificate.** The `<=` on line 514
   is **pre-registered** (docstring line 94) and pre-registration is binding even
   when the margin turns out thin. Changing it now would be moving a bar after
   the numbers were read. If it should be an exogenous margin next time, that is
   a design question for the Review, on the `T2.06` precedent (`acad758`).
6. **Do not touch `GOAL.md`, `GOAL_UNRUNNABLE_BASELINE`, or the four `GEN`
   citations.** RANK 3 is the Review's to re-own. Deleting a citation would clear
   `coverage`'s red in one edit and that is the `champions.py` prohibition
   verbatim.
7. **Still not yours to pre-empt:** `A4`, `T2.10`, `SO.07`, `SO.10`, `T1.08`'s
   pipeline repair, `UB.10`'s successor arm, the world-edit window, the `lc03`
   seat row, the `t306` venue row, the `W1.01`/`W1.03`/`W1.04` registration,
   `T4.06`'s adoption, `HR.1`'s family fate, and the `GEN` group's re-ownership.
   `BA.03` (c) stays refused. **Today's `PROGRESS.md` FOR THE BUILDER is a clean
   page and item 2 there (the 09-25 `WAITS-ON:` unlock) is correctly dated — do
   not start it early.** Items 1–4 above are additive to that page, not a
   replacement for it.

---

## FOR THE REVIEW (read at your next sitting)

**`D33` cannot be fired and has not been resolved, and your page says it fired.**
`class: conduct`, `decide_by: 2026-09-23`, and its default orders a re-date **to
2026-09-23** — a date that was already past on the first morning the default
could act. I proved it with your own module's `expired_actions()`; the check
never runs because the conduct branch returns first (RANK 1, repair ordered to
the builder). `docs/PROGRESS.md` FOR THE OWNER 2 says the default *"has fired for
a fourth time"*; there is no firing block, no `DECISIONS_RESOLVED.md` entry, the
`decide_by` has not moved, and your actual re-dates were **09-27** and **10-01**.
**Your acts were legal and well disclosed — the stop-rule you armed against
yourself on `w1-world-edit-window` is the best thing on that page.** What is owed
is the record: either **re-arm `D33`** with a `decide_by` and a default whose
action lies in the future (a deadline may tighten, never lengthen), or **execute
your own 09-23 addendum** (`d9f568f`), which already told you the cheap limb —
*"hold this desk to registering `W1.01`/`W1.03`/`W1.04` itself"* — and priced it.
The one thing that is not available is a fourth instalment described as a firing.

**And notice what that costs.** `w0-too-shallow` is now dated **2026-10-01** on
the ground that the builder taking it *"would pre-empt the D33 ruling."* If
`D33` has no live clock, that ground does not hold up — the registration is
deferred behind a ruling that cannot arrive by silence. **Eighteen days
unregistered, six idle builder slots this morning, four CLAIM-DEAD constitutional
commitments downstream.** You do not need the owner to un-block this; your own
addendum says so.

**Before you adopt `T4.06` on 09-25, read BOTH anchor-relative conjuncts, not
one.** Conjunct (2) `min_modality_latent_r2`: +0.0187 = **6.9%** of the anchor's
own seed spread, 1 of 3 seeds regressing — disclosed on the certificate and in
`run status`. Conjunct (3) `eval_loss_mean`: **`<=`, zero required margin**,
+0.0009 = **8.9%** of the anchor's own spread (0.0101), 3/3 paired seeds
improving — **printed by nothing.** The demonstrated result is the exogenous one:
`ratio_worst` **29.83 → 2.45** against the unmoved 10× gate, every seed. An
adoption that cites `loss_reweight CERTIFIED` as evidence about latent recovery
**or** about task loss would be quoting two numbers your own design pre-registered
against being read that way. Consider whether conjunct (3) should carry an
exogenous margin in the next bakeoff, on your own `T2.06` precedent
(`acad758`, 09-20: *"a strict `>` was deciding at zero margin"*).

**The `GEN` group still has no owner — second morning, 23rd day.** The 111th
audit asked for a **live parent**: a row that ages, an honest `DECLINED` with the
reason, or a named owner entry. Your sitting spent its clock on the OVERDUE
sweep, which was correctly ranked higher, and I am not indicting the choice. But
`coverage` has been rc=2 on `GEN.02`/`GEN.03`/`GEN.06`/`GEN.09` for 23 days, the
row that held them is terminal and will never be re-read, and neither refusal you
made on it (not a fourth date, not a deleted citation) was wrong. One line
creates the parent.

**The World seat is still uncontestable.** `champions_trigger_debt` **3** since
09-03, `champions_unwinnable` **4** since 09-13. Held **BY VERDICT** with no
`VERDICT:` and no `TRIGGER:` declared. Same docket as `D33`.

---

## FOR THE OWNER

**1. NO-DECISION: the instrument that stops decisions rotting on your desk has a
blind spot, and the one entry sitting in it is the one holding up his world.**
`decisions.py` checks whether a pre-registered default names an action that is
already in the past — the check exists because `D21` did exactly that last month.
`D33` does it too: its default says *"re-date once more, to 2026-09-23"*, and
2026-09-23 was yesterday. **The module's own function confirms the violation when
I call it by hand; the program never calls it, because entries marked as the
desks' own paperwork skip that check on a single line of control flow.** The
counter for it prints **0**. So `D33` cannot resolve by your silence (the action
is dead), has not been resolved by anyone, and its deadline has not moved —
while this morning it was cited as the reason to defer the world-edit
registration to **2026-10-01**. **Nothing in the ledger is unsound and no
threshold moved.** The repair is three small edits, ordered to the builder as
items 1–3, and it will make one of your instruments turn **amber on purpose** —
that is the correct outcome and I have forbidden the builder from making it green
by moving the baseline. **Nothing to rule.**

**2. `D32` and `D34` fall due TODAY; `D31` tomorrow. All three block zero
specs.** If unanswered, `D32`'s and `D34`'s pre-registered defaults become due to
fire on **2026-09-25** and `D31`'s on **09-26**. Each default's own text says it
*"buys VISIBILITY and nothing else"*, none moves a threshold, none edits
`GOAL.md`, none spends GPU. **My recommendations are unchanged in substance from
the two audits before this one and I am not re-arguing them: let all three
fire.** `D34`'s underlying defect is the one worth your eye —
`scripts/ladder_prompt.md` is **88,330 B against a 131,072 B exec cliff**, and
past the cliff your builder does not read a shortened prompt, it does not start,
and the slot looks like an ordinary `rc=126`. The instrument flags all three as
things our own tooling calls the desks' paperwork rather than questions about
what Jack must become; the **101st audit** already made that point and I am
citing it rather than making it again.

**3. NO-DECISION: the perishable GPU clock, reported because `D30` requires it —
two days to zero.** `2026-W38` holds 30 free Kaggle GPU-hours; **0.9176 drawn,
~29.08 h expire at end of Saturday 2026-09-26.** The one
dependency-satisfied buyer this project produced in three weeks was spent on
09-23 (`T4.06`, 0.4387 h against a 0.75 h projection, PASS) and there is no
second one: two of three GPU cost classes are empty with **no path in**, the
third holds only a VOID. **Neither your builder nor I will invent a dispatch to
spend them**, and that judgement is now consistent across five audits. The
reasoning, restated rather than the conclusion: a run whose result nothing may
claim converts free hours into a ledger row someone has to explain later, and
**48.42 GPU-hours already sit against specs holding no verdict — 33.78 of them
`D1.0`'s alone.**

**4. The standing indictment, twentieth day, and it is the only number on this
page I would ask you to remember.** Four of your constitutional commitments have
**no live falsifiable claim at all** — smell, balance, shelter/building, and *too
cold kills him* — every claim spec behind them parked or foreclosed on honest
evidence, and `NO-LIVE-PATH` reads **7 distinct commitments or seats** with no way
in. Your builder ran six clean slots this morning at 19% weekly usage and could
touch **none** of them: `run next` shows **zero fresh dispatches** and **all seven
cost classes are unfillable**, five of them because the repair is a redesign. Every
one of those holes traces to the same place — a world that charges nothing for
cold, distance, mass or exertion — and the world edit that would change it **was
designed eighteen days ago, has never been registered, and was re-dated this
morning to 2026-10-01 behind a decision that cannot arrive by silence.** The
desk that owns it disposed of **8 rows against 15 arrivals** this week — its best
week this month, and still its own instrument calls the drain **UNBOUNDED** with
54 live rows and `DECLINED` unused in all 76.

**The honest summary, asked directly.** Are we closer to a curious humanoid that
climbs the ladder than yesterday, or only to a longer list of green ticks? **We
are marginally closer, and less than the +1 suggests.** `T4.06` is a real
measurement about the unified brain — touch's grip on the fusion boundary went
from 29.8× to 2.45× against an exogenous gate nobody moved — and that is the
first thing in a while that is about the creature rather than about the
instruments. But `coverage` credits it as **support, not a claim**: the
`one brain / unison` commitment still has **1 passing claim of 28 specs**, and
curiosity **2 of 12**. Six of yesterday's eight builder commits were honest
accounts of having nothing legal to do. **The bottleneck is not the builder's
speed or the compute; it is that the world he lives in still cannot charge him
for anything, and the single unit of work that fixes that has now been deferred
five times by the only desk allowed to write it.** Nothing is being faked — the
refusals are all correct and all disclosed, which is why this reads ON TRACK
rather than DRIFTING. But a ladder whose next rung has been eighteen days from
existing is not a ladder anyone is climbing.
