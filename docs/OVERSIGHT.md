# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Findings are ranked by how much damage they do to the
> trustworthiness of the ledger, not by how interesting they are.

**2026-09-25 18:37–19:0x UTC — the 118th audit.** Six hours after the 117th
(12:37). The window is the builder's six slots `13:07`–`18:07`, which produced
**nine commits, one ledger event (`LT.02` PASS, 13:25:19), one new registered
spec (`LT.03`) and the first in-flight run of the constitutional ladder test in
this project's history.**

---

## VERDICT: ON TRACK — and this is the first audit in three weeks that can say it about the creature rather than about the instruments

The ledger is clean and got cleaner. All three of the 117th audit's `FOR THE
BUILDER` orders were executed in the first two slots of the window, verified
against their commits rather than their claims. The watch item my predecessor
could not settle — whether `LT.02`'s re-pointed `C1` was a silent loosening — is
now settleable because the row exists, and **it is not a loosening**; the
working is at Section 2. `110 → 111` demonstrated. And at the moment I write
this, pid `42934` is burning 104% of a core on `LT.03` — *"if there is a ladder
with an apple on top, he must try to climb the ladder, fall, and learn from
falling, purely out of curiosity"* — which is `GOAL.md`'s own image of success,
registered, pilot-validated and running for the first time ever.

The one finding that matters is not in the ladder. It is that **the organ whose
whole job is to tell the owner what happened today has not written a page since
2026-09-24 06:45, its own staleness guard exempted it at 06:57 on a rule that
expired at 07:45 and is never re-checked, and five substantive rulings — one of
which produced today's only PASS — have no owner-facing report anywhere.**

---

## FINDING 1 — `docs/PROGRESS.md` is 36 hours old, carries no staleness marking, and the rule that exempted it expired eleven hours ago (HIGH — this is the D15 disease, exactly)

### 1a. The fact, from disk

`git log -1 -- docs/PROGRESS.md` returns **`f7abc08`, 2026-09-24 06:45:13**. The
file on disk is byte-identical to the 09-24 page. It is **35 h 52 m old** as I
write.

Today's Review **did sit**, and productively: five acts committed between
`06:38` and `06:52`, every one of them to `docs/REVIEW_QUEUE.md` —

| commit | time | act |
|---|---|---|
| `a208c40` | 06:38 | OVERDUE act 1 — `pl02-eye-gate` ACTED `d361b10`, stop-rule refused on a false premise |
| `2c1d1c0` | 06:42 | **THE PS-FAMILY LEGIBILITY RULING** — disposes `ps05`+`ps06`+`ps08`+`ps09` in one act, "and it is a STRENGTHEN" |
| `5f88609` | 06:44 | **`lt02` RULED (a)** — a body-carried stochastic noise source, found SPEC-LOCAL |
| `ee3826d` | 06:47 | `t215` DISPOSITIONED, FAIL-DISPOSED marker REFUSED — OVERDUE 0 of 7 |
| `46d4b60` | 06:52 | MINT `t108-pipeline-repair-has-no-design` |

Then it died. `review.log:939-942`:

```
2026-09-25T06:37:04  review start — mode DAILY, model opus, 20m / 120 turns
Execution error
2026-09-25T06:57:11  docs/PROGRESS.md untouched by this rc=124 run and only
                     24h old (cadence allows 25h) — still current, not stamping
2026-09-25T06:57:11  wrote the INCOMPLETE trend row for 2026-09-25
2026-09-25T06:57:11  sweep end rc=124 — 0
```

### 1b. The guard was RIGHT at 06:57 and has been WRONG since 07:45

That decision was correct when it was made: the page was 24 h 12 m old against a
25 h cadence. **But the banner decision is taken once, at the moment of death,
and nothing ever re-evaluates it.** The page crossed 25 h at **07:45** and has
carried no marking for the eleven hours since. The 09-23 death did better —
`51efe92` *"run exited rc=124 mid-report — docs/PROGRESS.md sealed as a draft"* —
so a reader could see the hole. Today a reader cannot: the page opens
`**2026-09-24 06:3x–06:5x UTC — DAILY**` and reads as a finished report.

`PROGRESS_LOG.md` *does* carry the labelled hole (`| 2026-09-25 | DAILY | — | — |
— | — | INCOMPLETE …`), and its own text says *"any sealed draft is bannered in
docs/PROGRESS.md."* Today there is no sealed draft and no banner, so the log's
pointer resolves to a page that denies the gap.

### 1c. What the owner is being shown, and why it is now actively wrong

My own prompt orders me to read this page's `FOR THE BUILDER` and `FOR THE
OWNER` every audit. Read today, the live copy tells the owner:

- **FTB 2:** *"Tomorrow (09-25) the `WAITS-ON:` disposition unlocks… **Do not
  start it early**."* — that tomorrow is today and the work was done at 15:13.
- **FOR THE OWNER 1:** *"~29.08 hours expire **Saturday 2026-09-26, two days
  out**"* — it is one day out.
- **FOR THE OWNER 2:** *"`D33` … is now **one day past** its `decide_by`"* — it
  is two.
- **The headline:** *"THE BUILDER IS BACK AND HAS NOTHING TO DO."*

That last one is the damaging sentence. The day it describes had **+1 PASS, a
new Tier-5 spec registered, a pilot validated and the ladder test launched.**
The owner's only current-state view of this project says the builder is idle, on
the day it was not.

**And the five rulings above have no owner-facing report at all.** The
PS-family legibility ruling was executed by the builder three hours later
(`a313122`) and the `lt02` arm-(a) ruling is what produced today's PASS — the
two most consequential desk acts of the week exist only as queue-row stamps.
This is `D15` verbatim: the page is current-state by design, so an unwritten
page is not a late report, it is a **deleted** one.

**Not routed as a new decision.** `D33` already asks whether this desk can
produce its designs inside a 20-minute sitting, `D36` asks which design gets
Sunday, and `review.sh` records **eight max-turns deaths across the three
organs** (four of four Sunday FULLs, plus 09-22, 09-23 and now 09-25 DAILY).
Opening a fourth entry on the same capacity question would be noise. What IS
new and is routed below is the narrow mechanical gap: **the staleness exemption
is evaluated once and never re-checked.**

---

## FINDING 2 — `decisions --check` is still RED on a broken ratchet, and the organ that owns the repair sat this morning and did not take it (MEDIUM, inherited, now aged)

```
RATCHET BROKEN: 1 DEFAULT-ACTION-EXPIRED, baseline 0. It may shrink, never grow.
```

Unchanged from the 117th audit. `D33`'s default names **2026-09-23**, its
`decide_by` is **2026-09-23**, earliest legal firing **2026-09-24** — the
ordered action is in the past on the day it fires. The two named repairs
(**shorten `decide_by`**, or declare **`(CLOCK: <whose>)`**) are the Review's to
make and neither was taken in this morning's five acts.

What is new is only the ageing: `D33` now reads `CONDUCT-DESK … STALE by 2
day(s)`, and `D35` joins it at **STALE by 1**. `D36` (filed 09-25) is due
tomorrow. All three are desk-executable and say so — *execute it, report it, do
not ask.*

`run status` still prints the compounding cross-organ error, and it has now
grown a second arm — **my predecessor's**:

```
STEERING-DATE-MISMATCH — 2 open-decision deadline(s) misquoted on a steering page.
  D33  docs/PROGRESS.md  says 2026-09-27 — register says decide_by 2026-09-23
  D33  docs/OVERSIGHT.md says 2026-09-20 — register says decide_by 2026-09-23
```

The second arm is **my predecessor's own page**: the 117th audit named `D33`'s
FULL-sitting filing date in prose beside the entry, and the reader — which does
not read intent — could not tell a provenance date from a deadline. It was
right to flag it. **This page states the deadline once and does not restate the
filing date in any form: `D33`'s authoritative `decide_by` is `2026-09-23`, and
it is two days stale.** That arm should clear when this file is committed,
leaving `PROGRESS.md`'s `2026-09-27` as the only one — and that one matters,
because it tells the owner the wrong deadline for an entry about this desk's own
capacity to deliver.

---

## FINDING 3 — `LT.03` is in flight, healthy, and its single most likely outcome is VOID on a control that has never been observed to fire in a humanoid body (MEDIUM — a watch item, not a violation)

### 3a. The run is genuinely alive — verified, not inherited

```
42914  SNs  01:14:05  cpu_budget wrap LT.03 …          (setsid leader)
42931  SN   01:14:05    experiments.run LT.03
42934  RNl  01:14:04  CPU-time 01:17:32  104%  m.run(Ledger())
```

`declared_pids` shows `42914` admitted at `17:23:59` on the sanctioned
`launch_detached` lane and `42934` declared as `run_spec LT.03` at `17:24:00`.
CPU-time **exceeds** wall-time at 104%, so it is computing, not blocked. The
static 1,107-byte log is buffered stdout. **The 117th audit's FINDING 1 is
fully discharged: the correct lane was used on the first try.**

### 3b. The gate that decides its fate

`lt_03_ladder_test.py:850`:

```python
if m.get("icm_fixates", 0.0) < 0.66:
    return _void(m, "the ICM control did not fixate — the panel trap is "
                    "not live in this rig, so dwell <= 0.15 is untested")
```

`icm_fixates` is `float(icm.panel_dwell > 0.4)`, needed in ≥2 of 3 seeds. It is
checked **before** the claim, so if the ICM control does not fixate, `LT.03`
returns **VOID** — not PASS, not FAIL. Five CPU-hours, no verdict, and the
`kills` clause (*"the pivot is decided by this result"*) does not fire either
way.

### 3c. Why that outcome is the live risk

Every measurement this project owns of an ICM agent's panel dwell **in a
ragdoll/humanoid body** reads zero:

| source | seeds | `panel_dwell` (ICM) |
|---|---|---|
| `LT.02` attempt 1 (`f16ee9b`, 09-19, full envelope) | 3 | **0.0**, std 0.0 |
| `LT.02` attempt 2 (`e4ec371`, 09-25, 795.61 s) | 3 | **0.0**, std 0.0 |
| `LT.03` pilot (seed 90, 800 dec/arm) | 1 | **0.0** |

The only reading above the 0.4 bar anywhere in the repo is `LT.02`'s control
block — `panel_dwell 0.666667` — and that is **PG.4's rover**, a different
apparatus. `LT.03`'s own control clause names this gap in so many words:
*"proving the trap is live HERE, not only in PG.4's rover."*

### 3d. The builder's disclosure is accurate and should be said so

I went looking for an overstatement here and did not find one. The commit
message compresses to *"the trap is live in THIS rig"*, which on its own would
be too strong — but the durable artifact, the `PILOT RECORD` in the docstring,
draws the exact line:

> *"The trap is live in THIS rig; **whether icm follows the gradient into it over
> 25,000 decisions is control (1)'s question and stays the run's to answer.**"*

That is the honest statement. The teleport probe proved the panel is
*perceivable* (5 rays stochastic at 1.0 m, every non-panel geom exactly 0.0;
1 ray still reading it at 10.9 m) — a sensor result, correctly not sold as a
behavioural one. **Section 2 finds nothing wrong here.** What I am recording is
the *cost shape*: the full envelope is 25,000 decisions/arm against the pilot's
800, so the question genuinely needs the run — and the project should know
before the harvest that a VOID is the modal outcome, so nobody reads it as
evidence about curiosity.

### 3e. The cost class is correctly read

`cpu<2h` is enforced per-seed, not per-experiment: `run.py:4306` derives the
child kill from `rtf.spec_child_timeout_seconds(_spec)`, and the builder's
stated arithmetic (`7200 × 3 seeds × 2 slack = 43,200 s`) matches the
machinery. `projected_full_seed_s 6219` ≈ **1.73 h/seed** is under the label.
`Budget.CPU_LONG` stands and the envelope was frozen as declared (10 × 2500),
**not grown to fit** — which is the failure mode this check exists for.

---

## Section 1 — integrity of the ledger

**Clean, and re-derived this audit rather than inherited.** 111 PASS rows. Every
PASS commit still resolves in git (0 missing). Every PASS whose spec declares a
`control` carries `control_metrics` or the declared-`NONE` form (`T0.01`,
`T0.10`). `run status` EXIT 0.

`LT.02`'s new row was checked row-by-row because it is the window's only ledger
event: `status PASS`, `attempt 2`, `commit e4ec371` (exists), `duration_s
795.61`, `seeds [0,1,2]`, `impl_sha b7e355fb21d13f7f`, `dirty_files null` —
**clean stamp**, and attempt 1's FAIL preserved intact under `history` with its
own metrics rather than overwritten. `supersedes_fail` points at `f16e…`.

Standing reporting-only debts, all previously reported and none moved by
today's work: 2 DIRTY STAMPS (`T6.03`, `PL.02`), 16 STALE CLAIMS, 1 stale
pre-`impl_sha` (`T2.02`), 4 UNBACKED CERTIFICATES, `pass_on_dead_dependency = 3`
**at floor**.

**One ratchet floor moved DOWN and it is the good direction:** `1594e8e`
lowered `UNREACHABLE_BASELINE` **96 → 95** because `LT.02`'s PASS regained a
spec — *the floor follows the count down*. `unreachable = 95` vs floor 95, **AT
floor**. `T0.21` was re-bought in the same slot (`b186249`, PASS attempt 23,
9.95 s) to pay the staleness that edit created. That is the ratchet working as
designed: it shrank, and the shrink was paid for.

`review_queue_net_arrivals = 12` and `review_queue_piled_on = 4` are unchanged
since the 117th audit. `review_queue_violations` **0**, forms `{}`.

## Section 2 — thresholds and controls over time

**No findings — and one inherited watch item is now CLOSED CLEAN.**

**The `LT.02` `C1` re-pointing was not a loosening.** The 117th audit could not
settle this because no row existed; it exists now. Attempt 1 gated
`chaos_occupancy_icm >= 3.0` and measured **0.145** → FAIL (attempt 1's own
recorded value, preserved in the row's `history` and quoted verbatim by the
registry's GUARD; attempt 2 re-measures the same arm at `0.1425`, which is what
the live certificate carries — `STEERING-METRIC-MISMATCH` flags the pair and
names "a foreign attempt" as the legitimate reason, which this is). Attempt 2
gates `chaos_occupancy_icmnoise >= 3.0` and measures **6.41**. The bar did not
move.
What moved is the arm under it, which is the harder question, and four things
make it honest rather than a re-point-until-green:

1. **The falsified claim is preserved and re-reported every run**, in the
   registry hypothesis itself: *"GUARD: attempt 1 measured noise-free body chaos
   REDUCIBLE (occupancy 0.145 vs 3.0); that stays falsified … and a PASS
   certifies the DETECTOR, never that his own body is a noise trap."*
2. **Attempt 1's number is still a live recorded metric** —
   `chaos_detector_separation_reducible = 0.0592` sits on the PASS row.
3. **The scope narrowed rather than widened.** `LT.02` now claims only that the
   detector works against a by-construction irreducible source. A positive
   control is *supposed* to be guaranteed; the discriminating half is the
   negative control, and it holds: scripted climber `chaos_occupancy 0.0833`
   against a `<= 1.0` bar.
4. **The independent-agreement control ran and fired:** `c["panel_dwell"]
   0.6667 > 0.4` AND `c["chaos_occupancy"] 6.198 >= 3.0`, two detectors agreeing
   on a known positive.

**`PS.08` part 2 (`a313122`) is a TIGHTENING, verified against the diff, not the
message.** The commit claims *"NO bar moves"* and the diff bears it out:
`ACC_MIN 0.70`, `SHUF_ACC_MAX 0.60`, `CONTROL_ACC_MAX 0.60`,
`CONTROL_MARGIN_MIN 0.10` all byte-identical. The change makes the amputation
control **rate-blind** — its rows are now frozen at trip start, before contact —
because attempt 1 measured the control at `0.708 ± 0.156`, *above* the probe's
`0.583`. A control the amputated channel can beat is not a control; deleting the
leak is the repair. And it was checked that it cannot rescue anything: replayed
offline, `probe_bal_acc 0.583 < ACC_MIN 0.70` fails under either control.

`e5e627b` promotes `T2.15`'s tf-idf null from REPORTED to **GATED** — a new
conjunct in `_check`, strictly more to satisfy.

Seven days of `git log -p` over `registry.py`, `registry_expansion.py` and
`experiments/tests/`: **no threshold moved in the loosening direction, no
control deleted or weakened, no `_check` gained an `or`, no seed count reduced,
no assertion removed.** Every numeric move in the window is an addition
(`LT.03`'s frozen bars) or a tightening.

## Section 3 — drift from the goal

**No drift, and the strongest goal-trace this organ has written.**

- **`LT.02` PASS** → *"he must try to climb the ladder, fall, and learn from
  falling, purely out of curiosity"* (`GOAL.md:31-33`). It is the detector that
  stops `PG.4`'s noisy-TV trap from eating the curiosity signal. It is now
  measured.
- **`LT.03` registered + launched** → the same sentence, directly. This is not a
  spec that serves the ladder image; it **is** the ladder image, registered as a
  Tier-5 falsifiable claim with `kills` pre-committed (*"if it fails, GOAL.md's
  ladder image needs a goal/skill layer … and that pivot is decided by this
  result, not by preference"*). The static apparatus audit — every arm's source
  scanned for `ladder/rung/platform/rise/climb` before any physics runs — is the
  right kind of paranoia for a claim this load-bearing.
- **`PS.08` part 2** → *"a few dozen concepts learned by consequence — hot,
  heavy, far, tiring, dangerous, worth-it"* (`GOAL.md:187`).
- **`T2.15`, `T0.21`, the baseline drop** → instrument work, correctly labelled
  as such.

**The converse question, unimproved.** `coverage` EXIT 0 with
`commitments_uncovered = 0` **at floor** — but **4 CLAIM-DEAD** commitments
(smell, balance, thermal-kills, shelter/building), **13 with live claim specs
and nothing passing**, and `goal_unrunnable = 7` **unchanged since 09-05, twenty
days**. Curiosity: 12 specs, 2 passes. `one brain / unison`: **28 specs, 1
pass** — the single worst ratio on the board and the commitment `GOAL.md` argues
hardest for. `LT.03` does not touch it.

## Section 4 — is the builder alive and productive?

**Alive, disciplined, and this is its best six hours in three weeks.**
**19 iterations since midnight, 19 of 19 `rc=0`**, no aborts, no pause, no
credit exhaustion (`week:all models` **51%** at the 18:0x open, well under the
90% stop; the Fable line reads 73% and is not the gate). **PASS delta over 24 h:
110 → 111.**

The six slots in my window, each traced to its commit:

| slot | commit(s) | what landed |
|---|---|---|
| 13:0x | `e4ec371` | 117th audit items 2+3 — four missing journal lines, LESSONS occurrence record; `LT.02` launched **in the foreground** |
| 14:0x | `d377874`, `e5e627b` | `LT.02` PASS harvested (13:25:19, clean stamp); `T2.15` null promoted to GATED |
| 15:0x | `1594e8e`, `b186249`, `a313122`, `c64ba51` | baseline 96→95 + `T0.21` re-buy paid; `PS.08` part 2 |
| 16:0x | `601abb8` | `LT.03` implemented, ~700 lines, `run()` refuses until pilot |
| 17:0x | `c1114ae`, `753030e` | pilot validated, envelope frozen, **registered run launched on the sanctioned lane** |
| 18:0x | `afa0d02` | mid-flight verification; board re-derived empty |

**All three 117th-audit `FOR THE BUILDER` orders were executed, verified against
commits.** Item 1 (land `LT.02` in the foreground) — done at 13:25. Items 2+3
(the four missing journal lines and the LESSONS occurrence record) — `e4ec371`,
reconstructed from `ladder.log` and `declared_pids` rather than smoothed. Item 4
(do not fix `LT.02`'s science) — no bar moved. Item 5 (do not pre-empt) — `D31`,
`D33`/`D35`/`D36`, the `W1.01/03/04` registration and `UB.10`'s arm choice all
untouched; `BA.03 (c)` stands REFUSED on the record.

**Two pieces of conduct worth naming.** The 18:0x slot chased down a dirty
`experiments/cpu_budget.json` that could have poisoned the `LT.03` stamp and
**verified in `protocol.py` source** that it sits in `RUNNER_OUTPUTS` and is
excluded — rather than assuming, and rather than committing it. And the slot
re-derived the board empty (`run next`: 1 fresh of 48, `LT.03` itself) instead
of inheriting the previous slot's claim of it. That habit is what made the
117th audit's items a ten-minute discharge.

## Section 5 — compute honesty

**Reconciles.** `2026-W38` holds **0.9176 h drawn of 30**, so **~29.08 free
Kaggle GPU-hours expire tomorrow, Saturday 2026-09-26.** The builder has now
refused to manufacture a buyer **28 consecutive times** and it remains the right
call: `coverage` shows every GPU cost class EMPTY or NOT FILLABLE. An unasked-for
GPU run is worse than an expired hour.

Today's only spend is CPU and it is accounted: `LT.03` has consumed ~1.3
core-hours of a projected ~5.2 h against a day budget reading `5674 s of
57600 s used` at audit time. `37 cpu spec(s) currently unaffordable until
midnight` is the day-scoped counter behaving as designed, not a regression.

The standing debt is unchanged and unmoved since 09-18:
`gpu_hours_no_verdict` **TOTAL 48.42 h**, of which **`D1.0` alone is 33.78 h
across 2 attempts for 0 verdicts**, and `UNATTRIBUTED` **6.32 h across 21 jobs**
(at its declared floor of 21).

## Section 6 — stuck decisions

**Zero `MEANS-ESCALATED`.** Nothing a measurement could settle is sitting on the
owner's desk.

**Zero `UNDECLARED`, so the standing "arm one per audit" duty has no object
today, and that is the honest report rather than an omission.** The five unarmed
items are three `CONDUCT-DESK` (`D33`, `D35`, `D36` — desk-executable, not the
owner's), one `DEFAULT-ACTION-EXPIRED` (`D33`, FINDING 2) and one soft
`CONDUCT-MISFILED?` (`D31`).

**`D31` is the owner's until midnight and I have not touched it.** Its
`decide_by` is **2026-09-25** — today — so the earliest legal firing is
2026-09-26, and the 117th audit already claimed that slot. **I re-affirm the
booking so it cannot be lost: the overseer slot at or after 2026-09-26 00:37 UTC
owes `D31`'s armed default (i) MARK BUT DO NOT CAP**, journalled with the words
*"the owner did not rule by 2026-09-25, so the pre-registered default fired"*,
reversal = delete one `if` from `experiments/gpu.py`. A ruling from the owner
overtakes it — check `decisions` first (the `D19` lesson).

`D32` and `D34` were fired by this desk at 00:4x today and are transcribed in
`DECISIONS_RESOLVED.md` at lines 8327 and 8403 — checked, both present, so
nothing was acted on without being recorded. The two `PROGRESS.md` owner-asks
are correctly matched to `D22` and `D31` and exempted; **0 UNROUTED, 0
VANISHED** — though see FINDING 1 for why that reading is about a page from
yesterday.

## Section 7 — bakeoff hygiene

**No findings, and the one live bakeoff is honest about its own weakness.**
`T4.06` remains the only bakeoff verdict, and `run status`'s `ANCHOR-DECIDED
CONJUNCTS` block prints its margins rather than burying them: `loss_reweight`
certified at **+0.0187 = 6.9% of the incumbent's own seed spread 0.2699, with
one of three seeds regressing.** That is inside the noise by any ordinary
reading, `f7900b5`'s `STATISTIC_BOUND` arrival note says so on the certificate's
face, and its adoption is routed to the live `t402` row and has **not** been
taken. A weak result correctly labelled weak and correctly not acted on.

`LT.03` deliberately does **not** use `run_bakeoff` — *"it VOIDs on a sub-gate
arm, and the icm control is REQUIRED to fail"* — and defers arbitration to
`LT.04`. That is the right separation of screening from ranking.

## Section 8 — the honest summary

**Yes. For the first time in a long while, genuinely yes — and with one
qualification I will not let the yes swallow.**

`110 → 111` is a small number and it is not why. The reason is that at 17:23:59
this project launched a registered, pre-registered, pilot-validated, control-
bearing experiment whose hypothesis is the sentence the owner wrote at the top
of `GOAL.md`: a humanoid, with reward identically zero, climbing a ladder
because it wants to. It has a `kills` clause that fires a real pivot. It has a
static audit that raises before any physics if an arm's reward code so much as
mentions the word *rung*. Its envelope was frozen from measurement and **not
grown to fit**. And the desk that ruled its dependency did so at 06:44 this
morning. That is the whole machine — Review, builder, ladder, budget — working
in the direction it was built for, in one day.

Three weeks of this organ's pages have been about instruments: launch lanes,
journal gaps, broken clocks, a backlog with no drain. Today the instruments got
out of the way.

The qualification: **it may well come back VOID**, on a control nobody has seen
fire outside PG.4's rover (FINDING 3), and if it does, the honest reading is
*the trap is not behaviourally live in a humanoid body* — a finding about the
rig, not about curiosity, and not a rung. Nobody should be surprised into
treating it as one.

And the thing that did not work today is the one that tells the owner any of
this. The Review sat, ruled five times, produced today's PASS by ruling — and
then died at its wall clock for the fifth time in ten days, leaving the owner's
only current-state page reading **"THE BUILDER IS BACK AND HAS NOTHING TO DO"**
on the day the ladder test launched. The science is ahead of the reporting, and
that is a much better problem than the reverse. It is still a problem.

---

## FOR THE BUILDER

1. **CREDIT, and it is the substantive item.** Six slots, nine commits, every
   one of the 117th audit's orders discharged in the first two, the correct
   detached lane used on the first try after four failures yesterday, the
   envelope frozen from measurement instead of grown to fit, and the
   `cpu_budget.json` dirty-stamp worry run down in `protocol.py` source rather
   than assumed. Keep re-deriving rather than inheriting.
2. **At the harvest (~22:40+), read a `VOID` on `icm_fixates` as a rig finding
   and write it down as one.** `lt_03_ladder_test.py:850` VOIDs before the claim
   is evaluated if ICM panel dwell is not `> 0.4` in ≥2 of 3 seeds. Every
   humanoid-body reading this repo owns is **0.0** (`LT.02` attempt 1 and
   attempt 2, 3 seeds each, std 0.0; the `LT.03` pilot at 800 dec/arm). The only
   `> 0.4` reading anywhere is `LT.02`'s control at `0.6667`, which is **PG.4's
   rover**. If it VOIDs, the sentence for the journal is *"the panel trap is not
   behaviourally live in this body"* — **not** anything about curiosity, and
   **not** a reason to touch `CONTROL_DWELL_MIN`. Report the measured
   `icm_dwell` per seed either way; that number is the useful output of a VOID.
3. **Do not relaunch and do not end a slot on a guess about it.** `42914` is a
   setsid leader on the sanctioned lane; verify CPU-time growth on `42934` and
   stay off it. If the row lands, replay `_check` offline and commit it as
   found — a `FAIL` fires the `kills` clause (goal/skill-layer pivot) and a
   `VOID` names its lane. Do not re-roll seeds.
4. **Do not "fix" `LT.02`'s science.** Its PASS is clean and its scope is
   deliberately narrow: it certifies the DETECTOR against a by-construction
   irreducible source, and the registry's own GUARD keeps attempt 1's
   *noise-free body chaos is REDUCIBLE* finding falsified and re-reported. If
   you quote `LT.02` anywhere, quote that scope with it.
5. **Not yours, do not pre-empt:** `D31` (fires at the ≥00:37 overseer slot
   tomorrow — see FOR THE OWNER 2), `D33`/`D35`/`D36`, the `W1.01`/`W1.03`/
   `W1.04` registration, `T1.08`'s pipeline repair, `UB.10`'s arm choice, the
   `t402` adoption, `A4`, `T2.10`, `SO.07`, `SO.10`. `hash-salt-lottery-in-a-
   gated-metric` becomes legal tomorrow, 09-26.

## FOR THE REVIEW

6. **Your page has not been written since 2026-09-24 06:45 and carries no
   marking.** `review.sh` correctly declined to banner it at 06:57 (*"only 24h
   old, cadence allows 25h"*) — **but that exemption is evaluated once, at the
   moment of death, and never re-checked.** The page went stale at 07:45 and has
   read as a finished report for eleven hours. The mechanical repair is a
   producer-side staleness check that runs on *read* rather than on *death* —
   e.g. `review.sh` (or the organ-liveness half in
   `scripts/lib_liveness.sh:review_liveness`) stamping the banner when
   `PROGRESS.md` crosses its own 25 h cadence, not only when a run dies inside
   it. **This is the concrete repair; the capacity question behind it is already
   `D33`/`D36` and is not re-asked here.**
7. **This morning's five acts have no owner-facing report.** `a208c40`,
   `2c1d1c0` (the PS-family legibility ruling — the builder executed it at
   15:22), `5f88609` (the `lt02` arm-(a) ruling — it produced today's only
   PASS), `ee3826d`, `46d4b60`. They exist only as queue-row stamps. Whatever
   else the next sitting does, those five and today's `LT.02` PASS and `LT.03`
   launch belong on the page before anything older does.
8. **`D33` still breaks a ratchet with a floor of 0 and only you can repair
   it.** `DEFAULT-ACTION-EXPIRED`: default names 2026-09-23, `decide_by` is
   2026-09-23, earliest firing 2026-09-24. Take a named repair — **shorten
   `decide_by`**, or **declare `(CLOCK: <whose>)`** — deleting the date is not
   one of them. It is now `STALE by 2 days` as `CONDUCT-DESK`, with `D35` at
   **STALE by 1** beside it. And fix `PROGRESS.md`'s `D33` date while you are in
   there: it says **2026-09-27**, the register says **2026-09-23**, and the
   register is the authority.
9. **The queue's drain is still UNBOUNDED and today's pile broke.** 36 OPEN /
   3 HELD / 59 live; arrivals exceed disposals by **12** over the trailing 7
   days (21 arrived, 9 disposed, 14 DISPOSITIONED — which is not disposal). **8
   rows were dated onto 2026-09-25 against a measured capacity of 6**, plus
   `2026-09-27` (7) and `2026-10-02` (7) amber. `review-queue` prints **2026-10-03
   as the next date with room**. Your own pre-committed stop-rule on
   `w1-world-edit-window` lands on **2026-09-27**, the same Sunday `D36` is
   about.

## FOR THE OWNER

**1. NO-DECISION — the good news first, and it is real.** At 17:23:59 today the
project launched `LT.03`, *"THE LADDER TEST: curiosity alone climbs the
ladder"* — a registered, pilot-validated, three-control Tier-5 experiment whose
hypothesis is the image you wrote into `GOAL.md`: a humanoid figuring out a
ladder with the environment's reward set to exactly zero. It is running now
(~5.2 h, expected ~22:40 UTC). It has a pre-committed `kills` clause that fires
a real architectural pivot if it fails, and a static audit that refuses to run
at all if any arm's reward code references the apparatus. **Priced honestly: its
most likely single outcome is VOID** — one of its controls requires the naive
curiosity agent to fixate on the noise panel, and in a humanoid body that agent
has measured exactly 0.0 dwell in every run this project has ever made
(6 seeds across two `LT.02` attempts, plus the `LT.03` pilot). A VOID would be a
finding about the rig, not about Jack, and would cost ~5 CPU-hours to learn. I
am telling you the odds before the result rather than after.

**2. NO-DECISION: `D31` is yours until midnight tonight and fires tomorrow.**
Its `decide_by` is **2026-09-25**, so nothing may fire before 2026-09-26. If you
have not ruled by then, the overseer slot at or after **2026-09-26 00:37 UTC**
will fire its armed default **(i) MARK BUT DO NOT CAP** — a per-job overrun mark
and stderr print in `experiments/gpu.py` when billed hours exceed the declared
estimate. It caps nothing, refuses no dispatch and moves no threshold; it buys
visibility and nothing else, which is why option (ii) GIVE COLAB A CEILING is on
your desk and not in the default. Reversal is one `if`.

**3. `D33` — CITED, NOT RE-ASKED, and now two days past its `decide_by`.** The
narrow ask is unchanged: rule that the world EDIT is IMPLEMENTATION and was
never the Review's to hold under `D22`. The cost line is the only thing new —
`W1.01`/`W1.03`/`W1.04` are now **nineteen days unregistered**, and
`review-queue` prints both of `w0-too-shallow`'s ordered specs as **NOT
REGISTERED** row by row. `D35` and `D36` are also live and are desk-executable,
not yours.

**4. The perishable price, stated for the last time this week.** `2026-W38`
holds **0.9176 h drawn of 30 — ~29.08 free Kaggle GPU-hours expire TOMORROW,
Saturday 2026-09-26**, and there is no legal buyer for them: every GPU cost
class in `coverage` is EMPTY or NOT FILLABLE. The builder has refused to
manufacture one **28 consecutive times** and should keep refusing. A GPU hour
spent on a run nothing asked for is worse than an expired one.

**5. NO-DECISION, and the one thing I would want you to see: the reporting organ
missed its page today, on the best day it has had.** The Review sat at 06:37,
ruled five times in fourteen minutes — including the ruling that produced
today's only PASS — and then died at its 20-minute wall for the fifth time in
ten days. Its staleness guard correctly exempted `docs/PROGRESS.md` at 06:57
(24 h old against a 25 h cadence), that exemption expired at 07:45, and nothing
re-checks it. So the only current-state page you have says **"THE BUILDER IS
BACK AND HAS NOTHING TO DO"** — written yesterday, about yesterday — on the day
the ladder test launched. I have routed the mechanical repair to the Review
(FOR THE REVIEW 6) and have deliberately **not** opened a fourth decision on the
capacity question behind it, because `D33` and `D36` already ask it and a third
entry would be noise. I am flagging it here only so that, if you read
`PROGRESS.md` this week, you know what it is missing.
