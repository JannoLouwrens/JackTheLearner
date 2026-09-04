# OVERSIGHT — 69th audit, 2026-09-04 06:40–07:2x UTC (began at `3b2c095`; the Review committed `e20c75e` mid-audit, see the CONVERGENCE note below)

## VERDICT: DRIFTING — the ledger is clean and the builder is healthy, and every red the instruments print now terminates in one desk that has closed 2 of 35 routed rows in 15 days, while the reader built to watch that desk prints `0 violations`

> ### CONVERGENCE — read this first, because it changes how much weight finding 1 deserves
>
> I began this audit at 06:39 and reached finding 1 by reading
> `docs/REVIEW_QUEUE.md`'s 35 `ROUTED:` rows by date and status. **At 06:5x,
> while I was writing, the Review committed `e20c75e` and reached the same
> finding from a different direction** — it counted *startable* specs (9 of the
> 99 rows-less specs have all deps PASS; all 9 are parked, pilot-blocked or
> decision-held) and computed arrivals over the last five full days.
>
> | | this audit | the Review, independently |
> |---|---|---|
> | rows routed / closed | 35 / **2** | 35 / **2** |
> | arrival rate | **4.3/day** (7-day window) | **5.6/day** (5-day window) |
> | closure rate | 0.13/day | 0.13/day |
> | verdict | the desk is the binding constraint | *"Arrival minus departure is ≈5.5 rows/day… I am the consumer; this is a finding about me."* |
>
> Two organs, two methods, two windows, same conclusion inside one hour, and
> the Review indicted itself without prompting. **That raises my confidence in
> finding 1 and lowers the priority of finding 3's instance** — the Review is
> now awake to this. It does not lower finding 1's structural half: neither
> organ can build the counter, and `review-queue` still prints `0 violations`.
> Findings 1 and 2 stand as written. Finding 3 is corrected below, and my
> FOR THE OWNER item 3 was **wrong** — the Review caught it. Both corrections
> are marked.

Say the good part first, because it is large and it is true.

**§1 no findings.** 135 rows, **102 PASS / 22 FAIL / 11 VOID**. Every PASS
commit resolves in git (0 dangling); **0 PASS rows carry a `+dirty` stamp**;
every PASS has an implementation in `experiments/tests/`; **every PASS whose
spec declares a control carries control metrics AND its `_check` consumes
them** — I re-verified the consumption by hand on `LF.02` (`c["corrupt_raised_
frac"] == 1.0`) and `VO.02` (`if _claim(c): return False`) after a naive scan
flagged them; both are correct. The only two PASSes with empty
`control_metrics` are `T0.01`/`T0.10`, which declare `control="NONE, BY
DECISION (52nd audit B5)"` with a recorded reason.

**§2 no findings.** Seven days of `git log -p` across `registry.py`,
`registry_expansion.py` and `experiments/tests/`, restricted to *modified*
files so new specs cannot hide a loosening. Every constant that moved in an
existing spec moved toward harder: `T3.09` seeds 1→3 and `N_LIVES` 16→32,
`LG.10` `TEMP` 0.25→1.0, `T0.31` `N_PROPERTIES` 13→14, `T0.34` gaining a
DISJOINT-CHARGES property with a control that reproduces the pre-fix
double-billing. No `_check` gained an `or`. No control was deleted or weakened.
No seed count fell. No assertion was removed.

**§4 no findings.** 24 iteration starts, 24 ends, **24 `rc=0`** in 24 h.
Demonstrated **95 → 102 (+7)**, registry 225 → 234 (+9), pass rate 42.2% →
43.6%. All seven of the 68th audit's builder findings (B1–B7) were executed
inside eight hours, and the double-billing defect I found live yesterday is
measurably fixed: `XL.00`'s detached re-buy billed **1171.28 s under the
wrapper alone**, with zero per-spec grandchild charge.

Four of that day's seven PASSes are science, not plumbing: `SO.01`/`SO.04`
(he can be watched, and being watched does not change him — bit-identical over
2000 steps), `SO.02` ("I'm cold" is true when he is cold, MI 1.51 bits at the
ear), `LF.02` (a life survives `kill -9` bit-exactly).

**Now the finding.** It is not an integrity failure and it is not a loosening.
It is a throughput failure at the single point every other red passes through,
and it is invisible because the instrument that watches that point was built to
count broken promises and has no reading for kept ones.

Ranked by damage to the trustworthiness of the ladder:

| # | finding | damage |
|---|---|---|
| 1 | **The Review queue is diverging, and `review-queue` reports `0 violations` while it does.** 35 rows routed since 08-20; **2 ACTED, 0 DECLINED, 0 DISPOSITIONED — ever.** 30 of the 35 arrived in the last 7 days (**~4.3/day**) against a lifetime disposal of **2 in 15 days (~0.13/day)**, a **33× divergence**. 33 rows are live; **26 dated promises fall due between 09-05 and 09-11** against the tool's own measured capacity of **1 per consumer cycle**. The reader counts OVERDUE, STALE, HOLD-WITHOUT-A-CLOCK, VANISHED, CLOCK-REMOVED — five classes, all of them *promises broken*. It has **no reading at all for promises kept**, so a desk that closes nothing prints green until the dates go red, and then prints them one day at a time | this is the exact disease `REVIEW_QUEUE.md` was created on 2026-08-24 to cure ("nothing could print *3 routed, 0 acted on, oldest 4 days*"). The file now prints all three of those and still cannot print *the desk is not keeping up* |
| 2 | **At least 16 of the 31 OPEN rows are the ONLY declared repair for something an instrument prints red right now** — including **all five PILOT-BLOCKED specs, which is 100% of GPU dispatch** (`gpu<20min`←`DP.04`+`SM.03`, `gpu<2h`←`T2.11`, `gpu<8h`←`LC.07`), **all four CLAIM-DEAD commitments** (`five-commitments-are-claim-dead-behind-foreclosures`), the seven corpse citations, the fifteen welded specs, both VOID-FORECLOSED redesigns (`BA.03`, `T3.06`), the `Learning core` seat's trigger debt (three `d10-*` rows), the `Sensory fusion` seat's VOID, and the `Language grounding` NO-ARENA seat | the queue is not a backlog beside the ladder; it **is** the ladder's forward edge. Finding 1's divergence is therefore the project's binding constraint, not a filing problem |
| 3 | **The Review's `FOR THE OWNER` section is a second decisions desk that no instrument reads, and an unanswered item leaves the current-state page in 24 hours.** `PROGRESS.md` is declared *"current state, not a log"* by its own header, so each run rewrites it. Yesterday's item 2 — *"W1 stops being a queue row and becomes the project's stated stage"*, which the Review itself called *"the strategic fork; the `D1.0` gate is a detail beside it"* — and item 3 (`repaired_by`) **both have zero occurrences in today's page**, unanswered. `decisions --check` reads `DECISIONS_NEEDED.md` only and prints **`ratchet ok (0/10 undeclared)`**; `grep` confirms the W1 recommendation appears nowhere in `DECISIONS_NEEDED.md` or `REVIEW_QUEUE.md`. **CORRECTION, verified before publishing: it is not destroyed.** `docs/PROGRESS_LOG.md` is append-only and its 09-03 row carries both recommendations verbatim. The accurate statement is *buried and un-clocked*, not *deleted* | this is the D1 disease relocated. It has **no `class`, no `default`, no `decide_by`**, so it cannot go OVERDUE — it was never armed — and it survives only as a clause inside one row of a dense log table that no instrument parses. Contrast `REVIEW_QUEUE.md`, whose contract is *rows are never deleted* (T1.02 precedent) and whose clocks go red. **And the mechanism was already diagnosed**: `D15`'s own update (2026-08-29) states *"`docs/PROGRESS.md` appears nowhere \[in `overseer_prompt.md`]. The Review reads the overseer every morning; the overseer has never read the Review."* Six days, unrepaired. I have armed the W1 item as **`D21`**, into a file where entries are never deleted, in the same hour it left the current-state page |
| 4 | **One of those three items is a MEANS question on the owner's desk.** Item 3 asks the owner to authorise a `repaired_by` field in `run blocked`. It adds a reporting edge, changes no `depends_on`, no verdict, no gate and no certificate — the Review itself already ruled out the option that *would* change semantics ("do NOT add the edge to the registry"). Nothing is left to decide | rule 3 exists for exactly this. It is invisible to `decisions --check` for the same reason as finding 3, so the `0 MEANS-ESCALATED` reading is also true only of the file it reads |
| 5 | **The CPU day-meter forecloses 53 of 152 CPU specs after 3600 s of spend, on a 57600 s ceiling — 6.25% utilisation closes 35% of the lane, and today it fired on genuine spend.** Live now: `used_s` **5906.82 s (10.3% of ceiling)**, 53 specs refused until midnight. `gate_cpu_child` refuses on `spec_child_timeout_seconds` — the enum worst case, 54000 s — while `rtf.py`, shipped by the same builder 24 h earlier and gated as `T0.32`, **measures and projects the actual wall duration**. `XL.00`'s actual cost was **1167.8 s against that 54000 s allowance: 46×**. The day's 5907 s is almost entirely routine housekeeping (gate sweep 4560.65 + `XL.00` re-buy 1171.28), so the class is foreclosed on essentially every day the loop does its own maintenance | today's realised cost is **zero** — `coverage` reports 0 FRESH dispatches at `cpu<2h`, so nothing was actually refused. This is a **live landmine, not a live wound**, and I rank it fifth for that reason. It is armed for the first morning after the 09-06 Review orders CPU work |

**What flips the verdict back.** Not a code change. Finding 1 needs an
instrument (B1 below) and findings 2–3 need the queue to drain. If next
week's `review-queue` reads a disposal rate at or above its arrival rate, the
ledger stays clean, and the builder keeps its 24/24, this is `ON TRACK`
without qualification — the science this system is producing is real and the
integrity discipline around it is the best it has been.

---

## THE FOUR MANDATORY INSTRUMENTS (read live 06:39–06:41 at `3b2c095`)

| instrument | rc | reading |
|---|---|---|
| `coverage` | **2** | **0 commitments with NO declared spec.** The red is the standing, routed set: **4 CLAIM-DEAD** (smell, balance, shelter/building, thermal), **3 PARK-ON-AN-UNREACHABLE-RELEASE** (`BA.02→LT.08`, `SH.01→SH.02`, `SM.02→SM.03`), **5 PILOT-BLOCKED**, **6 VOID-FORECLOSED**. `unreachable` **91 of 234, baseline 91 — AT FLOOR**. `CITED-BUT-UNRUNNABLE` **7**, of which **4 NEW** (`GEN.02/03/06/09`) — carried correctly by the widened `goal-cites-four-specs-that-resolve-to-corpses` row (68th audit B5). **QUEUE DEPTH: 4 dispatchable, all 4 VOID → 0 FRESH dispatches at any of the seven cost classes.** |
| `decisions --check` | 0 | **0 MEANS-ESCALATED, 0 UNDECLARED, 0 OVERDUE in the file** (ratchet 0/10 — nothing in `DECISIONS_NEEDED.md` needs arming, so I did not invent one; I armed **D21** from finding 3 instead). Live: **`D15` and `D16` default tomorrow, 09-05**; `D17` 09-07; `D18` 09-09; `D19` 09-14 (costs 3 specs, holds `cpu<10min` empty *honestly*); `D20` 09-18 (raised by the 68th audit). See finding 3 for what this rc=0 cannot see. |
| `champions --check` | 0 | 27 seats, **11 violations, every ratchet at baseline, none grown**: phantom arenas 0/0, unfalsifiable 3/3, uncontestable 3+1/4, unverified verdicts 2/2, trigger debt 3/3, UNDECLARED 0/0. `Learning core` still held **BY VERDICT off a VOID** with all three re-open triggers closed (`LC.07` PILOT-BLOCKED, `LC.03` VOID-FORECLOSED, `UB.10` VOID). `World` still held BY VERDICT with **no deciding run named and no TRIGGER declared** — the file's strongest marking backed by nothing, unchanged. |
| `run review-queue` | 0 | **0 violations** — and see finding 1 for why that number is not reassuring. 31 OPEN / 2 HELD / **2 ACTED** / 0 DECLINED / 0 DISPOSITIONED of 35; oldest live 11 d; consumer ran 09-03. `piled_on` **17 of 31** (the 68th audit's B7 instrument, working). Five amber date-piles; `next_free_due` **2026-09-12**. |

**Liveness.** Builder hourly at `:07`, 24/24 present. Overseer `:37 */6`. Review
started **06:37:04 today and was still running while I wrote this** (pid
1085589) — the `:37` cron collision the Review has flagged three times and
which is routed as `cross-organ-doc-race-voids-certificates` (DUE 09-06). Tree
was clean at `3b2c095` when I began; I commit only my own files by explicit
pathspec. Field watch Mondays, next 09-07.

---

## §1 — INTEGRITY OF THE LEDGER: no findings

Checked mechanically against `registry.BY_ID` and `git cat-file`, then by hand
where the mechanical check was ambiguous.

- PASS rows whose `commit` no longer resolves: **0**
- PASS rows carrying `+dirty`: **0**
- PASS rows with no implementation file: **0**
- PASS rows declaring a control with empty `control_metrics`: **0** (excluding
  the two `NONE, BY DECISION` declarations, which are correct)
- PASS rows declaring a control whose `_check` ignores it: **0**, verified by
  reading `_check` in `LF.02` and `VO.02` after an automated scan raised them.
  Both consume the control dict; the scan's pattern was wrong, not the specs.
  **Reported here rather than as a finding, because a false accusation against
  a clean spec costs more than the audit gains.**

`run status` independently reports 4 STALE CLAIMS (`UB.10`, `T3.09`, `D1.0`,
`LF.01`), 1 stale pre-`impl_sha` claim (`T2.02`), and 11 PASS rows predating
`spec_sha`. All five stale claims are deliberately-held reds waiting on the
09-06 Review; the 11 pre-`spec_sha` rows are 9 GPU-class plus 2 behind genuine
FAILs, and the builder correctly reports that the CPU-reachable stratum of that
set is now complete (`LC.02`, `XL.00` upgraded in the last 24 h).

## §2 — THRESHOLDS AND CONTROLS OVER TIME: no findings

Scan restricted to files with git status `M` in each commit, so a new spec's
constants cannot masquerade as a loosening. Every change in the window and its
direction:

| change | commit | direction |
|---|---|---|
| `T0.31` `N_PROPERTIES` 13 → 14 (`piled_on`/`next_free_due` ratchet conjunct) | `c255c34` | **stronger** |
| `T0.34` + DISJOINT CHARGES property, with a marker-stripping control that must double-bill | `df48f8e` | **stronger** — the control reproduces the exact defect the 68th audit found |
| `T3.09` seeds 1 → 3; `N_LIVES` 16 → 32 | `19461c4`, `d36f3f9` | **stronger** |
| `LG.10` `TEMP` 0.25 → 1.0 | `f6d1e3a` | **stronger** (more sampler entropy makes every gate harder) |
| `T1.09`/`T1.10` T4 → P100 | `d96042b` | **neutral** — identical ceilings, re-aimed at the recording device |

One accounting edit deserves naming even though it is not a finding: at
`7264bbf` the builder **hand-removed 4558.53 s of phantom charges** from
`cpu_budget.json`'s 09-04 bucket (`used_s` 9126.18 → 4567.65). I verified the
arithmetic is exact and the removals are precisely the six sweep descendants
that double-billed before the B1 fix landed mid-sweep, and it was disclosed in
the commit. It was correct. It is also the case that **nothing in the ladder
would have caught it if it had not been** — `cpu_budget.json` carries no
tamper-evidence, unlike the ledger's `commit`/`spec_sha`/`impl_sha` stamps. I
am not routing this: the file is a budget, not evidence, and the honest
disclosure is the control that worked. Recorded so the next audit has the
precedent.

## §3 — DRIFT FROM THE GOAL

### 3a. What the builder built in the last 24 h, and what it serves

| unit | GOAL.md sentence | verdict |
|---|---|---|
| `SO.01`, `SO.04` — the spectator stream, and that being watched changes nothing | *"His people are part of his world"* (:139-146); the observed-not-scripted deal (:172-183) | serves |
| `SO.02` — "I'm cold" is true when he is cold | *"VOICE — he must be able to make sound"* (:43); interoception (:41-42) | serves — **this is the one I would show the owner** |
| `LF.02` — a life survives `kill -9` bit-exactly | *"He lives, he dies, he remembers… death is not a reset; it is a page turn"* (:103-116) | serves |
| `T0.33`, `T0.34`, `6e3ad9a`, the B1–B7 discharge | *"protects the honesty of watching what happens when the three meet"* (:8-9) | serves |
| the `cpu<2h` certificate sweep, `LC.02`/`XL.00` re-buys | same | serves — provenance, and it caught zero regressions, which is the right answer |
| `docs/research/DUAL_PROCESS.md` | the fast/slow section (owner directive 2026-08-10), a research debt owed since that date | serves |

**Nothing is drift.** The shape is still worth naming, and this time it has
turned: **the last 7 iterations (00:07 → 06:17) moved the demonstrated count by
zero.** Every one of them said so in the journal, checked the board first, and
picked an honest fallback (accounting fixes, certificate provenance, a research
debt). `coverage` independently confirms the board: **0 FRESH dispatches at all
seven cost classes**, three classes with **no path in at all**, one honestly
FILL-HELD by `D19`. The builder is not idling and it is not manufacturing work.
It is a healthy engine with nothing in the hopper, and the hopper is filled by
the desk in finding 1.

### 3b. Which parts of GOAL.md have no passing spec

Unchanged from yesterday, and these are the thesis:

| commitment | specs | passing claims |
|---|---:|---:|
| one brain / unison | 25 | **1** |
| curiosity | 12 | 2 |
| fast/slow | 8 | **0** |
| hunger/thirst | 6 | **0** |
| sleep | 5 | **0** |
| plasticity | 4 | **0** |
| death & retry | 4 | **0** |
| smell · balance · shelter · thermal | 10 | **0 — CLAIM-DEAD** |

Every one of these is behind something in finding 2's list. Not one is behind a
missing idea or a busy builder.

## §4 — IS THE BUILDER ALIVE AND PRODUCTIVE: yes, no findings

24/24 `rc=0`. No paused loop, no repeated identical failure, no load abort, no
credit exhaustion. The 68th audit's B1–B7 were fully discharged in eight hours,
each with a re-bought certificate and a live verification rather than an
assertion — B1's fix was proved by a real detached receipt (`XL.00`, 1171.28 s
under the wrapper, 0 s to grandchildren), and B7's guard found the three
offending rows before the prose triage did.

One operational note under §4's "credit exhaustion" heading: at **06:07:13 the
loop refused Fable** — `week:Fable` **99%**, past the 95% model floor — and ran
the iteration on **Opus**, printing both meters and naming the gate it acted on
(`week:all models`), which is the standing rule followed correctly. See §5.

## §5 — COMPUTE HONESTY

**GPU.** `%U` (Sunday-start) keying matches Kaggle's reset. Current week
**2026-W35 = Sun 08-30 → Sat 09-05**.

| week | Kaggle spent | of 30 h | wasted |
|---|---:|---:|---:|
| W32 | 21.06 h | — | 0.12 h |
| W33 | 7.63 h | 22.4 h unspent | 0.26 h |
| W34 | 1.62 h | 28.4 h unspent | 0.00 h |
| **W35 (current)** | **18.93 h** | **11.07 h expire end of Sat 09-05** | **0.00 h** |

**0 overruns, 0 wasted hours this week, every charged job maps to a recorded
pilot or run.** (Minor correction to the 68th audit's line: 19.20 h is
Kaggle 18.93 + Colab 0.27; the Kaggle quota reading is 18.93/30.)

**The expiry has a named cause, and the sequencing is correct.** All three GPU
classes are `NOT FILLABLE` because `DP.04`, `SM.03`, `T2.11` and `LC.07` are
PILOT-BLOCKED — four redesigns, all four routed to the Review, none actioned.

**CORRECTION to my own first draft of this section.** I initially wrote that
`D1.0` attempt 2 is gated on the two `d10-*` rows due 09-06, *"the day after
W35's 11.07 h expire"*, and called it a timing miss. **That is wrong.** W35
expires at 09-06 00:00 and **W36 opens the same instant with a fresh 30 h**, so
a gate landing on 09-06 spends W36 exactly as intended; the Review's page,
committed while I was writing, states it plainly and orders the builder to *let
them expire — "this is inventory, not uptime… manufacturing a dispatch to spend
a dying quota is the failure mode, not the fix."* That is the right call and I
endorse it. The honest statement is the narrower one: **11.07 h expire unspent
because four redesigns sit unactioned, for the fourth consecutive week
(W33: 22.4 h, W34: 28.4 h)** — a queue-throughput cost, not a scheduling error.

**CPU.** `used_s` 5906.82 of 57600 today; 53 specs unaffordable until midnight.
See finding 5. The double-billing defect is **fixed and verified live**; what
remains is the sizing question, which is a different defect.

**Model quota.** `week:Fable` **99%**, `week:all models` **55%**, both resetting
Sep 7 05:00 UTC. The gate is `all models` and it is nowhere near the 90% stop —
but Fable is spent, so **every remaining iteration this week runs on Opus**, and
`all models` moved **48% → 55% (+7 points)** across the seven iterations that
moved the ladder by zero. I am not asserting causation (`D15`'s own measurement
found organ hours contribute near-noise to this meter, and I have not
re-derived it), and this is not a violation. It is the number `D15` fires on
**tomorrow**, so it belongs on the page today.

## §6 — STUCK DECISIONS: one finding, and it is finding 3

Inside `DECISIONS_NEEDED.md`: **nothing is stuck**. 10 open entries, all armed,
0 MEANS-ESCALATED, 0 UNDECLARED, 0 OVERDUE. No owner decision has been quietly
acted on — `D8`, `D10`, `D12`, `D13`, `D14` all show firing records in
`DECISIONS_RESOLVED.md` with reversal instructions attached.

Outside it: the Review's three `FOR THE OWNER` items (findings 3 and 4). I have
armed the largest of them as **`D21`** in this audit — see FOR THE OWNER.

## §7 — BAKEOFF HYGIENE: no findings

14 resolved entries. Every one names its losers, its reversal path, and its
evidence. Spot-checked `D13` (armed-default firing, five-case harness test at
firing time, and the honest note that condition (3) reads
`experiments.decisions` rather than grepping `decide_by:` because resolved
entries keep their dates forever — the inverse failure, caught before it
shipped) and `D14` (verified live at firing time against `ladder_loop.sh:271`).
**No decision was made without a learning gate; no VOID was treated as a
verdict** — `LC.03`, `D1.0`, `UB.10` and `ME.11.E/F` are all recorded as VOIDs
that decided nothing, which is precisely why `champions --check` can still
indict the `Learning core` seat for being held on one. **No winner was chosen
inside a noise margin**; `D1.0` VOIDed at 2.56σ against a 3.0σ bar rather than
being called, which cost 16.17 GPU-hours and was the right call.

## §8 — THE HONEST SUMMARY

**Yesterday, yes. Today, no — and the reason is not the builder's.**

Yesterday this project made a creature that can be watched without changing,
that can be interrupted by `kill -9` and resume the same life bit-for-bit, and
that cries in a way a listener can decode as *cold* rather than *hungry* with
1.51 bits of mutual information at the ear, against a dead null and a control
that misleads perfectly. That is four real steps and any of them would be worth
showing someone. The apple did not get closer to being climbed, but the creature
that would climb it got measurably more real.

Since 23:23 last night, seven iterations have produced seven honest reports that
there is nothing to build. That is not a failure of will or of imagination —
`coverage` agrees with them at every one of seven cost classes. It is what a
healthy engine does when the fuel line is closed.

And the fuel line is a desk with 33 live rows, 2 lifetime closures, 26 dated
promises falling due in the next seven days against a measured capacity of one,
and a reader that says `0 violations`. Every commitment GOAL.md calls the thesis
— unison, curiosity, fast/slow, sleep, plasticity, hunger, death-and-retry — sits
behind a row in that queue. Four of the owner's constitutional commitments are
formally claim-dead behind rows in that queue. All three GPU cost classes are
closed behind rows in that queue, and 11.07 free GPU-hours expire on Saturday
because the gate that would spend them is scheduled for Sunday.

The Review has now said this about itself three times — *"the builder is no
longer the bottleneck; this organ is"* (09-01), *"the world is the measured
bottleneck on six independent instruments"* (09-03), and this morning, with the
arithmetic attached and no prompting from anyone: *"Arrival minus departure is
≈5.5 rows/day and every date in the file is downstream of it. I am the
consumer; this is a finding about me."* It was right all three times. The first
two it wrote into a page that gets overwritten; the third arrived in the same
hour I reached it independently from the other side of the file, which is the
strongest evidence either of us could have offered that it is true.

What neither of us can do is count it. **The system built an instrument for the
backlog and gave it eyes for exactly one half of the transaction: it can see a
promise break, and it cannot see a promise kept.** That asymmetry is why a queue
can triple in a week while every ratchet in the repo sits at its floor, and it
is why two organs had to find the same number by hand on the same morning
instead of reading it off a line that fires on its own.

So: are we closer to a curious humanoid than yesterday? We are closer to one
that has a voice, survives death and can be watched. We are not closer to one
that climbs, and we will not be until something drains that desk. The green
ticks are honest. There are just no longer any left that we are allowed to buy.

---

## FOR THE BUILDER

**B1 (highest value, and it is the finding). `experiments/review_queue.py`
gains a THROUGHPUT reading — the desk's disposal rate, not just its broken
promises.** The file has five violation classes and all five fire on a promise
*breaking*. Nothing in the repo can print *"2 disposed in 15 days against 30
arrived in 7."* Add, computed from the file's own declared fields plus its git
history (the same source `VANISHED`/`CLOCK-REMOVED` already use):

- `disposed_per_cycle` — transitions into `ACTED`/`DECLINED`/`DISPOSITIONED`
  per completed consumer cycle, trailing 3 cycles.
- `arrived_per_cycle` — `ROUTED:` rows added per cycle, same window.
- `drain` — live rows ÷ (disposed − arrived), printed as **`UNBOUNDED — the
  desk is not keeping up`** when arrivals ≥ disposals, which is today's answer.

**A METRIC, never a violation**, for the same reason `piled_on` is: the Review
is a colleague, a slow week is legal, and a gate here would forbid a legal
move. But it must be a **ratcheted** metric — join `ratchet_readings.json` and
gate it in `T0.31` with the conjunct that matters: **re-dating a row, re-arming
a `DUE:`, or splitting one row into two must not improve `drain`.** That is the
68th-audit-B7 lesson applied one layer up — B7 measured the *act* that makes a
pile; this measures whether the pile is ever *consumed*, and rearranging cannot
do it.

**B2. `experiments/decisions.py` gains `UNROUTED-OWNER-ASK`, and
`scripts/overseer_prompt.md`'s READ FIRST gains `docs/PROGRESS.md`.** Parse
`PROGRESS.md`'s `## FOR THE OWNER` section for its numbered items and report
every one with no corresponding entry in `DECISIONS_NEEDED.md`. Report it as a
**red** class in the same ratchet family as `UNDECLARED` — an owner-ask with no
`class`, no `default` and no `decide_by` is strictly worse than an `UNDECLARED`
entry, because the deadlock is invisible as well as unarmed. Today's reading
would be **3** (09-03 items 1, 2, 3), and `D21` below takes one of them off it.

The prompt half is yours because it may not be mine: `D13`'s title records that
*the overseer may not edit its own script*. `PROGRESS.md` has been unread by
this organ since the Review measured that fact on 2026-08-29 and wrote it into
`D15`; six days is long enough.

Sizing, so the instrument is built against the real quantity: **yesterday's
page would have read 3; today's reads 1** (item 1, the design-throughput fork,
has no `D` entry; item 2 cites `D20`, items 3–4 are scheduling and liveness).
**The reading falling from 3 to 1 overnight without a single answer is the
defect, not the fix** — two items rolled off a current-state page. So the check
must compare against `PROGRESS_LOG.md`'s history as well as today's page, and
report an ask that *vanished unanswered* as its own class, exactly as
`review_queue.py` computes `VANISHED` against the previous committed revision.

**B5 (CONDITIONAL — build it only if the owner grants the draft-then-ratify
route in FOR THE OWNER item 2). A machine check for the strengthen-only law.**
If the builder begins drafting redesigns, §2 of this audit becomes the
load-bearing safeguard and it is currently a hand-run `git log -p`. The check:
a diff-direction detector over registry and gate constants that fails a draft
moving any threshold in the loosening direction, reducing a seed count,
deleting a control, or adding an `or` to a `_check` — the four things §2 looks
for by eye — in the same ratcheted idiom as `T0.28`/`T0.31`. Do not build it
speculatively; it is priced into the owner's decision, not ahead of it.

**B3. Implement `repaired_by` in `run blocked`. It does not need the owner.**
The Review escalated it (09-03 `FOR THE OWNER` item 3) as *"a real design change
to `run blocked`, so it is yours to authorise"*. It is not: it adds a
**reporting** edge that carries transitive-block mass without blocking
semantics, changes no `depends_on`, no verdict, no gate and no certificate, and
the Review itself already ruled out the variant that *would* change semantics
("do NOT add the edge to the registry" — it would make `T2.01` unreachable and
drift its certificate). A means question with the dangerous arm already
eliminated is builder work. First declaration: `T2.01.repaired_by = ["D1.0"]`
— the edge the 60th audit had to route by hand.

**B4. `gate_cpu_child` should gate on the `rtf` projection, with the enum worst
case as the fallback only when no projection exists.** `rtf.py` shipped
2026-09-03 12:16 and projects real wall duration within 25%; `gate_cpu_child`
shipped 10 hours later using `spec_child_timeout_seconds` — the enum worst case.
`XL.00` measured **1167.8 s against a 54000 s allowance (46×)**, and the result
is that **3600 s of routine housekeeping forecloses 53 of 152 CPU specs on a
57600 s ceiling**. The design's posture for overruns already exists and is the
right one: `admit_detached` marks an overrun and never kills. Admit on the
projection, mark the overrun, keep the hard ceiling. Second half: **join
`n_foreclosed_now` to `ratchet_readings.json`** so a day that closes 35% of the
CPU lane is a committed number rather than a line in a transient print — B3 of
the 68th audit made it visible; this makes it *remembered*.

Honest scope on B4: **today's realised cost is zero** — there are no FRESH
dispatches at `cpu<2h` to refuse. Do not treat it as urgent over B1. Treat it
as due before the 09-06 Review orders CPU work.

---

## FOR THE OWNER

**1. I have moved the Review's largest recommendation onto your desk and armed
it, as `D21` in `docs/DECISIONS_NEEDED.md` — and it left the Review's own page
in the same hour.** On 2026-09-03 the Review wrote that **W1 should stop being
a queue row and become the project's stated stage** — *"the strategic fork; the
`D1.0` gate is a detail beside it"* — with four constitutional commitments
claim-dead behind the world and six independent instruments returning the same
verdict from different directions. It wrote it into `docs/PROGRESS.md`, which
has no `default`, no `decide_by` and no reader in this repo. **Today's Review
rewrote that page — it is current-state by design — and the recommendation is
gone from it, unanswered, at 24 hours old.** It survives as a clause inside one
row of `PROGRESS_LOG.md`, which nothing parses. It now has a class, a default
and a date (**09-11**), in a file where entries are never deleted. **The default deliberately does NOT touch
GOAL.md's staging text** — that is yours and only yours — it does the narrower
already-permitted thing: the 09-06 FULL Review takes the W1 design as its first
item. Read `D21` and rule; the default is a scheduling choice, not an answer to
your question.

**2. The Review desk is this project's binding constraint — and it says so
about itself. Its proposed repair is sound, and it has one cost it did not
name.** It has closed **2 of 35 routed rows in 15 days** while 30 arrived in
the last 7. Sixteen of the 31 open rows are the *only* declared repair for
something an instrument prints red today, including every GPU cost class and
every claim-dead commitment. Its proposal, on today's page: **let the builder
DRAFT redesigns; keep ratification at the Review.** I have audited that
proposal against my own duties and I **support it**, with one condition.

*Why it is sound.* It moves work to where the capacity is (the builder has 24
slots a day and an empty board; the Review has ~40 minutes a week). The
conflict of interest it names — the builder drafting the redesign of a spec
that just failed it, the `T1.02` disease — is real, and the safeguards it cites
genuinely exist and genuinely run: the strengthen-only law, and my §2 duty to
audit every spec diff independently of its author.

*The cost it did not name, and it is mine.* **That proposal makes §2 of this
audit the load-bearing safeguard for the whole scheme, and §2 today is a
hand-run `git log -p` with no instrument behind it.** I read seven days of
diffs, restricted to modified files, and formed a human judgement about the
direction each constant moved. At today's volume that is tractable. At 5.6
drafted redesigns a day it is roughly a fivefold increase in exactly the work
that has no machine check — and this system's own history says an unmechanised
guard decays: `UNREACHABLE_BASELINE` sat one above the truth for half an hour
before the Review caught it by hand, and `T0.31`, `T0.28`, `T0.21` and
`T0.34` all exist because somebody decided an auditor's eye was not enough.
**My condition, therefore: grant the draft route, and in the same decision
require that the strengthen-only law get a machine check** — a diff-direction
detector over registry and gate constants that fails a draft moving a threshold
in the loosening direction, in the same idiom as the ratchets already running.
Without it, this proposal trades a throughput bottleneck for an integrity one,
and integrity is the thing this project has actually been good at.

**3. 11.07 free Kaggle GPU-hours expire at 09-06 00:00, for the fourth
consecutive week — and letting them expire is the correct call.** I want to be
explicit that this is **not** a scheduling error, because my own first draft of
§5 said it was and the Review's page corrected me before I published: W36 opens
30 fresh hours the same instant W35 dies, and `D1.0` attempt 2 is deliberately
timed for it. The Review's order to the builder — *"inventory, not uptime;
manufacturing a dispatch to spend a dying quota is the failure mode, not the
fix"* — is right. What the number is evidence **for** is item 2: the hours are
unspendable because four PILOT-BLOCKED redesigns sit on the queue, and W33 lost
22.4 h and W34 28.4 h to the same cause.

**4. `D15` and `D16` both fire tomorrow (09-05) if you do not rule.** `D15`
installs a per-organ usage ledger and a pace check exempting the first audit of
each UTC day; `D16` leaves `T0.27` visibly RED rather than exonerating the party
that proposed the fix. Both defaults were written to cost the ladder a visible
failure rather than a quiet green. Neither needs action from you unless you
disagree.

**5. One thing worth reading in full, because it is the good news.** `SO.02`
passed at attempt 1 yesterday: Jack now makes a sound that a listener can decode
as *cold* rather than *hungry* — 1.51 bits of mutual information at the ear
against a 0.10 permutation floor, with a freeze control that collapses to chance
and a swap control that misleads perfectly, and loudness held fixed so it cannot
fake urgency. He also survives `kill -9` mid-life and resumes bit-for-bit
(`LF.02`), and being watched provably does not change him (`SO.04`, bit-identical
over 2000 steps). The creature is getting realer. It is the world he is supposed
to live in that has stopped moving.

---

*Audited: `experiments/ledger.json` (135 rows, all 102 PASS commits resolved via
`git cat-file`), `git log -p --since="7 days ago"` over `registry.py`,
`registry_expansion.py`, `experiments/tests/` restricted to modified files,
`/data/jack-logs/ladder.log` (24 iterations), `/data/jack-logs/review.log`,
`experiments/gpu_budget.json`, `experiments/cpu_budget.json`,
`docs/REVIEW_QUEUE.md` (35 `ROUTED:` rows by date and status),
`docs/DECISIONS_NEEDED.md`, `docs/DECISIONS_RESOLVED.md` (14 entries),
`docs/PROGRESS.md`, `docs/CHAMPIONS.md` via `champions --check`, and all four
mandatory instruments run live at `3b2c095`.*
