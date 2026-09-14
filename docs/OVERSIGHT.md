# OVERSIGHT.md — the overseer's current-state report

> Rewritten every audit. Current state, not a log. The running history is the
> git log of this file.
>
> **95th audit — 2026-09-14, 00:38–01:0x UTC.** Read at HEAD `958c5ec`, clean
> tree. The builder is awake and between slots (00:07 slot ended `rc=0` at
> 00:12, having fired `D25`'s armed default at 00:10). 27 iterations in 24 h,
> 26 `rc=0`, one `rc=124` at 09:57. `demonstrated` 106 → **108** over the 24 h,
> and **108 → 108 across the last seven consecutive iterations**. The Sunday
> FULL ran yesterday at 06:37 and `docs/PROGRESS.md` still carries it; the next
> Review sitting is in ~6 hours.

## VERDICT: DRIFTING

**The clean part first, and it is most of the audit. I checked it mechanically,
not by reading anyone's report.**

- **Section 1, integrity.** 108 PASS rows. **Zero** without an implementation
  in `experiments/tests/`. **Zero** without a declared `control`. **386**
  distinct commits referenced across every `history` entry, **zero dangling**
  after stripping `+dirty`.
- **Section 2, thresholds.** `git diff --stat 7fa3da6..HEAD --
  experiments/registry.py experiments/registry_expansion.py experiments/tests/`
  is **empty**. Not one spec, test or registry byte changed in the 6 h since the
  94th audit. Over the 7-day window the only constants that moved moved
  *harder* and each was pre-registered with its reachability before the re-run
  (`MAX_SPREAD_RATIO` 6.0 and `MAX_HELDOUT_CV_PCT` 7.0 as **new** conjuncts;
  `T6.03`'s five additive conjuncts). **Nothing loosened. No finding.**
- **Section 5, compute.** Kaggle week `2026-W37` (Sunday-start `%Y-W%U`, which
  is correct — `gpu.py:396` records the ISO-week bug it replaced): **0.82 h of
  30 spent, 29.18 h free.** No overruns recorded, ever. `gpu_hours_no_verdict`
  TOTAL 48.42 h, unchanged; `D1.0` remains the single largest waste at 33.78 h
  across 2 attempts for 0 verdicts, and `gpu_unattributed_jobs` is 21, **AT** its
  floor. Nothing new was wasted — nothing was spent at all.
- **Section 6, decisions.** Three open (`D19` 09-14, `D20` 09-18, `D27` 09-20),
  **all armed**. `UNDECLARED` 0/10, `MEANS-ESCALATED` empty, `OVERDUE` empty,
  `UNROUTED-OWNER-ASK` 0/3, `VANISHED-OWNER-ASK` 0/0, `FIRING-DIFF` 0/0. Nothing
  to arm and nothing to fire. **`D19` is due TODAY** and becomes fireable at
  00:00 on 09-15 under `decisions.py`'s `(today − decide_by).days > 0` rule.
- **Section 7, bakeoff hygiene.** `D25` fired correctly at 00:10 with the
  required wording, a reversal path, and — see RANK 5 — a deviation from its own
  letter that is *declared, measured from git, and adversarially tested*. No
  VOID treated as a verdict, no winner inside a noise margin, no owner-decision
  acted on without a record.
- **`coverage`'s highest-priority check is GREEN: 0 commitments with NO declared
  spec.** EXIT 2 is on `claim_dead` 4 and the FILL-HELD/no-path classes — all
  owned, all routed, all unchanged.
- **`champions --check` rc=0, every class at or inside its ratchet**, and 0/0
  UNDECLARED seats: every seat now says what would unseat it.

**Why DRIFTING and not ON TRACK, in one sentence:** a ratchet counter built
after the day this queue tripled while every number sat at its floor now moves
**by the calendar alone, in both directions, on a file nobody has touched** —
and it is the counter watching the one desk that broke **13 promises at midnight**
and whose backlog has no projected end.

**Why not INTEGRITY RISK:** no capability claim on this ledger is false, nothing
loosened, every hard number I recomputed agreed with the record, and the two
worst facts below (the broken promises; the GPU lane closing) are *correctly
reported* by the instruments that own them. The defect is in what a reader can
be made to *miss*, not in what anyone asserted.

---

## RANK 1 — `review_queue_net_arrivals` moves with the CALENDAR, is bannered as an ACT, and can go silent on the exact scar it was built for

`run status` printed this at 00:12:

```
review_queue_net_arrivals = 8  !! MOVED -3 since 2026-09-13 (was 11). Say so in
  your report; if a committed change justifies it, `run ratchets record` in that
  commit.
```

The builder read it honestly — *"Neither is justified by anything I committed,
so I recorded nothing."* It was right, and the reason is worse than a one-off.

**The measurement. `docs/REVIEW_QUEUE.md` has not changed since `7e04382`
(09-13 19:19). I held those bytes fixed and moved only the date:**

```
SAME FILE BYTES, only the calendar moves:
  as-of 2026-09-11   arrived 25  disposed 8   net_arrivals 17
  as-of 2026-09-12   arrived 23  disposed 8   net_arrivals 15
  as-of 2026-09-13   arrived 18  disposed 7   net_arrivals 11
  as-of 2026-09-14   arrived 15  disposed 7   net_arrivals  8
  as-of 2026-09-15   arrived 13  disposed 6   net_arrivals  7
  as-of 2026-09-16   arrived 13  disposed 5   net_arrivals  8
  as-of 2026-09-17   arrived 13  disposed 5   net_arrivals  8
  as-of 2026-09-18   arrived 13  disposed 2   net_arrivals 11
```

Eight days, **zero acts**: −2, −4, −3, −1, **+1**, 0, **+3**. Six `!! MOVED`
banners, in **both** directions, on a dead file. `throughput()` is measured
against `_revision_before(today − THROUGHPUT_WINDOW_DAYS)`, so every sunrise
retires a day of history out of the window and the number changes.

**A pre-registered, falsifiable prediction, so this is not an argument:** if
nobody commits to `docs/REVIEW_QUEUE.md` between now and then, `run status` will
banner **`review_queue_net_arrivals !! MOVED +3`** on or about **2026-09-18** —
the *"the desk fell that far behind"* direction — with nothing whatsoever behind
it. If that does not happen, this finding is wrong and I want it struck.

**Why this is more than noise, and it is the part that matters.** The counter's
own docstring says why it exists:

> *"a queue tripling while every ratchet sat at its floor is what happened on
> 2026-09-04."*

The clock's daily contribution is **±1 to 4**. The desk's real daily routing is
**0 to 3**. Signal and noise are the same magnitude, so the two cancel: had the
Review routed 3 rows yesterday, today's −3 clock drift would have rendered
**UNCHANGED** and the counter built for 09-04 would have reproduced 09-04's
blindness exactly. That is not a hypothetical shape — it is arithmetic on the
numbers above.

**This is a sibling of the 84th audit's scar and the repair's predicate cannot
reach it.** `run.py:1499` reads `DAY_SCOPED_COUNTERS = ("cpu_foreclosed_now",)`
— a **one-element tuple** — and its class is *"a point-in-time reading of a
meter that RESETS at 00:00 UTC"*. `net_arrivals` does not reset; its **baseline
revision slides**. Same defect (a delta that is the calendar, not an act),
different mechanism, and so it sits outside the guard. This is verbatim the
lesson the builder itself wrote at 22:16 yesterday (`caa4257`): *"when you sweep
a class the READER transfers and the REPAIR does not."* Two hours later the
clock rolled and produced another member.

**Suppression is the wrong repair and I am saying so before anyone reaches for
it.** Adding `net_arrivals` to `DAY_SCOPED_COUNTERS` would suppress *every*
cross-day comparison — and this counter is only ever read once a day, so that
silences the real movement too. The decomposition is available and I derived it
above: recompute the counter with **today's bytes against the recorded reading's
baseline date**; the difference is the **clock** component and the residual is
the **act** component. Routed as B1.

---

## RANK 2 — 13 promises broke at midnight, the first violations this queue has ever carried, and the desk that owns them is measurably insolvent

`review_queue_violations` **0 → 13**. It had been 0 since 2026-09-03 and 0 for
the queue's whole recorded life.

```
13 VIOLATION(S) — OVERDUE 13     (all promised 2026-09-13, all 1 d old)
  w0-too-shallow · w1-world-edit-window · t215-router-under-lexical-null
  sh02-null-saturation · two-eyes-one-certified · ba03-null-saturates-the-horizon
  t306-matched-magnitude-noise-buys-coverage · lt01-c2-body-cannot-rise
  cross-organ-doc-race-voids-certificates · xl01-death-and-retry-...
  t205-world-model-loses-to-the-ridge-reference · t402-touch-drowns-audio-...
  t215-heldout-language-routing-diagnosis-is-filed-behind-a-pilot-blocked-wall
```

**Nobody was deceived and nobody lied — which is why this is RANK 2 and not
RANK 1.** The 93rd audit's `IMMINENT` reading forecast it precisely; the Review
wrote on 09-13 *"I said for a week it would not clear, and it did not"*; the
builder reported the movement at 00:12 and correctly declined to record it as
its own act. Every instrument did its job. **The desk simply cannot pay.**

**The insolvency, quantified from `review-queue`'s own THROUGHPUT block:**

| | reading |
|---|---|
| live rows | **49** (33 OPEN, 3 HELD, 13 DISPOSITIONED) of 59 routed |
| oldest live | 21 d |
| arrivals, trailing 7 d | **15** (2.14/cycle) |
| disposals, trailing 7 d | **7** (1.00/cycle) |
| *designs* (DISPOSITIONED, still ageing, not drain) | 10 (1.43/cycle) |
| drain | **UNBOUNDED — no projected end** |
| measured one-cycle capacity | **6** |
| due on or before the next cycle | **19**, of which **13 cannot be discharged** |
| settled FAILs whose ONLY owner is a row here (`D23`) | **23** |

That last line is the one that reaches the ledger. `fail_unowned` reads **0, AT
its floor** — a reassuring green — and `coverage` already prints the caveat in
full: *"a queue-row owner is a dated promise, not a repair; read the queue's own
drain before calling it handled."* As of midnight, 23 settled FAILs are owned by
a desk that just defaulted on 13 dated promises. **Ownership is nominal.** The
instrument says so out loud, which is the system working; the fact remains true.

**The Review has self-diagnosed this for seven days and its behaviour has not
changed.** `PROGRESS_LOG` 09-07, its own words: *"the bottleneck is not compute,
not credits and not the builder — 40 live queue rows, UNBOUNDED drain."* Live
rows have gone **40 → 49** since. On 09-13, knowing 14 rows were due that day
against a capacity of 6, it routed **3 more**. Each routing was individually
correct — `D23` establishes that routing *is* the right repair for an orphaned
FAIL — which is exactly why this is not a behaviour anyone can be argued out of.
It is a capacity fork, and I have escalated it as **`D28`**.

**One note on the honest repairs.** The reader offers three: ACT, DECLINE, or
re-date with a reason. `next_free_due` is **2026-09-18**. Re-dating 13 rows onto
one cycle with a capacity of 6 rebuilds the pile and breaks it again; **DECLINE
is an honest disposition and is underused** — 0 of 59 rows have ever been
declined. That is the desk's call and not mine, and I am naming it rather than
ordering it.

---

## RANK 3 — the live steering page carries two orders that cannot be executed, and the builder rederived that by hand on three consecutive slots

`docs/PROGRESS.md` is the Review's current-state page and its `FOR THE BUILDER`
block is a live order list. Written 06:37 yesterday; still at HEAD. Of six items,
4 and 5 were executed, 2 (`D25`) fired, 6 is a prohibition — and **two are dead**:

**Item 1 — *"`T2.10` — CPU, ten minutes, and worth more today than yesterday."***
I checked the spec rather than the page. `t2_10_retrieval_vs_recency.py:47,80-81`
carries its own reachability table:

```
MIN_PARA_MARGIN = 0.10   # the bar
arm D  all-MiniLM-L6-v2   0.0667 +- 0.0147   <- family best
D var  bge-small          0.0667
```

Every encoder this project has measured tops out at **0.0667 against an unmoving
0.10**. A ten-minute re-run returns the FAIL already on the row. The unit is an
`ME.11`-class retrieval redesign, not a dispatch. The bar is right and must not
move; the *order* is wrong about what the work is.

**Item 3 — *"`D1.0` attempt 3 into W37"* — has been illegal since 10:05 the same
morning.** `D1.0` declares `depends_on: [T2.00, T1.08, T0.09, T0.10]`, and the
Review's own Part-2 strengthening turned `T1.08` **PASS → FAIL** at
`heldout_cv_pct` 40.006 vs the 7.0 bar it had armed four hours earlier. The
strengthening was *correct* — it is exactly the class of finding this project
should be making. What no page states is the bill.

**And that bill is much larger than any page has priced it.** `run blocked`:

```
T1.08 = FAIL  frees 3  (blocks 45)  [impl unchanged 0 d]  — Seed variance measured
      frees:  D1.0, T2.01, T2.02
      also blocks (co-requisite): BA.02, CU.1-7, LT.08, ME.7, T2.13, T2.16-18,
      T3.02, T3.04, T3.05, T4.01, T4.04, T4.05, T5.01-05, T5.07-09, T6.01,
      T6.02, T6.04, T6.05, UB.1-8, UB.15, UB.16
```

**45 specs** — more than the next three terminal blockers combined (`LT.01` 9,
`NE.01` 8, `UB.10` 5). `T2.01`, which `PROGRESS.md` calls *"the largest single
unblock in the project for five weeks"* and *"one dispatch away from moving"*,
is now itself unreachable behind `T1.08`. `unreachable` went 93 → 94 → **97**.

**To be fair to everyone: this IS on the record.** The builder signed it in
`deda088`'s commit subject — *"…and it makes `T1.08` the project's largest
blocker"* — and the repair is routed twice with a clock
(`t108-bar-set-from-n1-is-now-the-projects-largest-blocker`, DUE 09-16). It is
on the record **in git**, and on **no page a reader consults**. The 94th audit
priced the same event as *"three certificates in a reporting lag"* and used that
pricing to argue down its own verdict. I think that under-priced it: the
certificates were the symptom; the GPU critical path closing was the event.

**What it cost, in hours.** The builder spent turns at 19:07, 20:07 and 21:07
deriving by hand that the board was empty and that items 1 and 3 were dead. Its
own 21:07 note: *"Three iterations today derived 'the board is empty' by hand."*
The page rewrites itself at 06:37, so these two orders self-heal in six hours —
**the mechanism does not.** An order names a spec id literally; nothing resolves
those ids against the live ledger between rewrites. There is already a routed
sibling for the *discharge* half (`oversight-for-the-builder-has-no-reader`, DUE
09-17); this is the **legality** half. Routed as B2.

---

## RANK 4 — the builder is not drifting and not idle. It is STARVED, and seven straight iterations of instrument-on-instrument work is what starvation looks like here

**Section 3, what the builder worked on in the last 24 h.** 18 commits since the
94th audit, **zero** touching a spec, a test or the registry:

| work | GOAL.md / SYSTEM.md sentence it serves |
|---|---|
| `D25`'s armed default fired; `lib_seal.sh` tail receipt | *"protects the honesty of watching what happens"* — a seal that defamed a complete page |
| `run next` TRIAGE lanes (0 fresh · 28 settled · 16 held) | *"a decision machine that cannot fool itself"* |
| `run blocked` reads `decisions.holds()` (HR.1 ranked 5th as a fresh unblock while `coverage` called it forbidden) | same |
| `unbacked_certificates` + `-> root` attribution | same |
| `T0.36` re-bought ×4 (32.3–32.8 s, 7/7 properties, 0 overstated roots) | certificate maintenance forced by the above |
| 4 lessons + journal entries | *"methodological memory"* |

**None of it is drift.** SYSTEM.md is explicit: *"Any session that makes the
machine better at catching its own errors has done the whole job even if no spec
passed."* Every one of these found a real defect in a tool shipped 1–4 hours
earlier by the same loop, each was red-first with the wrong versions *measured*
rather than asserted, and each failing version broke on a **different** conjunct.
That is good work by this project's own standard.

**But the same file carries the corollary:** *"when the machine is sufficient,
PROVE it by throughput — the guard against polishing the machine instead of
running it."* So here is the throughput, plainly:

- **`demonstrated` 108 → 108 across seven consecutive iterations** (18:07 → 00:07).
- **0.82 of 30 Kaggle GPU-hours spent this week. 29.18 free.** Three iterations
  in a row closed with a variant of *"W37's ~29 free hours still have no legal
  buyer."*
- **`run next`: 0 fresh · 28 settled · 16 held.** Every one of the 9 `NOT_RUN`
  specs is PARKED, PILOT-BLOCKED or decision-HELD.
- Of 76 GPU specs, 26 have satisfied dependencies and **not one is both legal
  and load-bearing**: the PASSes are certificates, the NOT_RUNs are all held, and
  every FAIL is settled and routed. The single highest-value GPU unit on the
  board is `T1.08`'s own repair — and that is a fixture redesign owed by the desk
  in RANK 2, dated 09-16.
- **`GENERALITY.md`: 14 barriers, 4 registered, 0 RUN, 0 PASS** — byte-identical
  for two weeks. The stated destination received zero seconds of compute again.

**The honest causal chain, and it is one sentence:** the builder cannot reach
Jack because every unit that would move Jack is owed as a redesign by a desk with
UNBOUNDED drain, so it spends its hours on the only work it can legally reach —
its own instruments — and does that work well enough that the starvation reads
like diligence. **Four of this project's 7 CLAIM-DEAD/no-live-path commitments
(smell, balance, thermal, shelter) and all 23 undrained FAILs sit behind the same
desk.** This is RANK 2 restated as its cost to Jack.

---

## RANK 5 — `D25` fired with a declared deviation from its own letter. I checked it adversarially and it holds. Recorded because a firing that deviates is exactly where an overseer should look

The default's letter: *"if this run committed `docs/PROGRESS.md` **AND** appended
its `PROGRESS_LOG` row."* The builder replayed that against the 2026-09-06
incident the decision was written about, found the **first conjunct false of that
very run** (`cf18320` shows `lib_seal.sh` committing the page the agent left
dirty), and implemented **the receipt alone**.

**Receipt-alone is strictly BROADER than the letter** — it stamps *CHECKLIST
COMPLETE* on runs the two-conjunct form would have called a draft. That is a
loosening relative to a pre-registered text, decided and certified by the agent
executing it, in one commit. I looked hard at it for that reason. It holds:

- the reason is **measured from git**, not argued;
- the self-certification trap is **closed and tested** — `review.sh` writes its
  own dead-run fallback row, which would have let the organ certify itself;
  `lib_seal.sh` refuses any candidate containing `INCOMPLETE` and
  `test_lib_liveness.sh` asserts it directly, with 14 new assertions across all
  four cases;
- the banner is careful about what it does **not** claim (`UNAUDITED` still said;
  nothing after the receipt claimed to have run);
- `--firing-check WORKTREE` EXIT 0 — no `GOAL.md` edit, no numeric bar moved;
- the deviation is declared in three places and the reversal is one line.

**No violation. One residual, named rather than called closed:** *"in this branch
the page is the run's complete product and is committed in the same breath"* is
an argument, not a measurement — it is the one load-bearing sentence in the
record that is not backed by a number. And **one owed act**: `D25` is the only
one of the last five fired defaults with **0** entries in
`docs/DECISIONS_RESOLVED.md` (`D17`/`D22`/`D23`/`D26` each have exactly 1). It
fired 30 minutes ago and `D22`'s record shows transcription is normally a later
act — it is owed, not late. B4 so it does not roll off.

---

## Everything else I ran, with no finding

- **`champions --check` rc=0, all eight classes at or inside ratchet.** 2/3
  unfalsifiable, 4/4 unwinnable (AT floor), 2/2 unverified verdicts, 3/3 trigger
  debt, 1/1 kindless discharges, **0/0 UNDECLARED**. Unchanged. The `World` seat
  is now **11 days** held BY VERDICT with no deciding run and no `TRIGGER:` — the
  oldest seat finding on the board, already the Review's and already stated by it.
- **Section 5 detail.** `%Y-W%U` (Sunday-start) is the correct week key and its
  comment records the ISO bug it replaced. `overruns: []`. No dispatch in 24 h,
  so no new unattributed hours.
- **Usage meter.** `week:all models` **86%** at 00:07 against the 90% stop, with
  `pace_gate`'s line at 89. `week:Fable` pinned 100%, so every slot since 19:07
  correctly refused Fable and walked to Opus and said so. The week resets
  ~04:59 today. Healthy and correctly reported.
- **`decisions --firing-check`** clean across all 26 firings in repo history.

---

## FOR THE BUILDER

Ordered. B1 is the durable repair and the only one I would call urgent.

1. **Split `review_queue_net_arrivals`'s delta into CLOCK and ACT — do not
   suppress it.** `run status` currently banners a calendar artefact as an act,
   and worse, a real routing can be cancelled by the clock into `UNCHANGED`.
   The decomposition needs no new data: `throughput()` already takes the
   baseline, so recompute the counter **with today's document bytes against the
   recorded reading's own `at` date**. That value minus the recorded value is the
   **clock** component; the live value minus that is the **act** component. Print
   `MOVED -3 (clock -3, act 0)`. My eight-day table above is a ready-made
   fixture: with the file frozen, `act` must read **0** on every one of those
   days while `clock` reads −2/−4/−3/−1/+1/0/+3.
   **Do NOT add it to `DAY_SCOPED_COUNTERS`** — it is read once a day, so
   suppressing cross-day comparison suppresses everything. And the sweep the
   22:16 lesson demands: `run.py:1499` is a 1-tuple against a class that now has
   at least 2 members. Name **every** counter whose value can move with no commit
   in the commit that fixes the first — I checked and `review_queue_total`,
   `review_queue_piled_on` and the `*_violations` classes are date-sensitive too,
   though `violations` moving by the clock is a **real event** (a promise
   breaking) and must keep bannering. The distinction to encode is not
   "clock-driven" but **"is the moving thing an event or a window?"**
2. **A legality reader for the steering pages' order lists.** `PROGRESS.md`'s
   `FOR THE BUILDER` items 1 and 3 have been un-executable since 10:05 on the day
   they were written, and three iterations burned turns discovering that by hand.
   The ids are literal, so this is mechanical: resolve every spec id named in
   `PROGRESS.md`'s and `OVERSIGHT.md`'s builder sections against `BY_ID` and
   `Ledger.unsatisfied`, and print any that name a spec which is held, parked, or
   has an unsatisfied dependency. **Reporting-only and unfloored** — an order can
   be legitimately aspirational, and a gate here would forbid a legal move. This
   is the sibling of `oversight-for-the-builder-has-no-reader` (DUE 09-17): that
   row asks whether an order was *discharged*, this asks whether it *could be*.
   Attach it to that row rather than opening a new one if the desk prefers —
   `review_queue_net_arrivals` is bannered either way, and RANK 1 is about to
   make that phrase mean something again.
3. **`T1.08` is the whole game and it is a fixture redesign, not a dispatch.**
   45 specs, 29.18 free GPU-hours, and `[impl unchanged 0 d]`. It is routed and
   dated 09-16 and I am not re-dating it — I am recording here that it is the
   single highest-value unit on this board by a factor of five, so that when the
   desk triages on 09-16 it is triaging against that number and not against a row
   title. **`heldout_cv_pct ≤ 7.0` does not move.** The measured 40.006 is the
   finding; the repair is to the fixture's seed variance, never to the bar.
4. **Transcribe `D25` to `docs/DECISIONS_RESOLVED.md`.** It is the only one of
   the last five fired defaults with zero entries there. One-line item, noted so
   it does not roll off.

## FOR THE OWNER

**1. `D28` — NEW, routed today, `class: goal`, `decide_by` 2026-09-21.** *The
Review desk is structurally insolvent and the repair costs something only you may
spend.* The evidence is RANK 2 in full: arrivals 15 vs disposals 7 over the
trailing week, drain **UNBOUNDED**, live rows 40 → 49 in seven days, **13 dated
promises broken at midnight** — the first in this queue's history — and **23
settled FAILs whose only repair owner is a row on that desk**, while
`fail_unowned` reads a reassuring 0. The desk has diagnosed itself correctly for
seven days and its behaviour has not changed, because each individual routing is
the *correct* act under `D23`.

Why this is yours and not a bakeoff (rule 3 checked explicitly): the arms are
(i) more Review wall clock, (ii) fewer routings, (iii) a second consumer, (iv)
the builder drains its own queue. Arm (i) spends the **shared all-models usage
meter**, whose exhaustion took every organ dark for 4.3 days in August, and
(iii)/(iv) reallocate design authority that `D22` placed with the Review by armed
default. Every arm turns on what is *permitted*, not on what *works* — which is
the narrow case SYSTEM.md still reserves for you. The full entry is in
`docs/DECISIONS_NEEDED.md` with the numbers attached.

The armed default, so silence cannot deadlock it: **(a) OVERDUE FIRST.** The
Review's daily sitting spends its first act disposing the OVERDUE class before
routing anything new. It picks only already-permitted actions (a desk may order
its own work), moves no threshold, edits no `GOAL.md` text, deletes no row,
declines nothing on anyone's behalf, takes no new permission, and is reversible
by deleting one sentence from the Review's prompt. Its price, stated: on a day
with both a broken promise and a fresh finding, the finding gets routed later in
the same sitting.

**2. `D19` is due TODAY** (09-14) and becomes fireable at 00:00 on 09-15. Cited,
not re-asked. Its default is NO FETCH, which leaves hearing's four claim specs
visibly red rather than quietly worked around. It costs 3 specs.

**3. NO-DECISION — the thing I would want you to see even if you read nothing
else.** `T1.08` blocks **45 specs**. That is the largest blocker number ever
recorded on this board, more than the next three combined, and it appeared
yesterday as the *correct* consequence of the Review finding that a noise-floor
spec had never gated its own noise floor. The project got more honest and
simultaneously lost its GPU critical path, with 29 free hours in the pot. Both
halves are true, both are good news about the method, and **neither is on any
page you would read** — the first is in a commit subject, the second is a
sentence in a log. That gap is what RANK 3's B2 is for.

**4. NO-DECISION — the builder.** 27 iterations, 26 clean, one `rc=124`. It
refused to pre-empt `D25`'s clock by three hours to fill an empty slot, refused
to manufacture a GPU dispatch into 29 free hours, refused two of its own steering
page's orders on measured grounds and said why, and found a defect in each of its
own last four instruments within hours of shipping them. Nothing here needs you.
It is starved, not idle, and the starvation is upstream of it in `D28`.
