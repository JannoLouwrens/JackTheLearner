# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: **DAILY** (Part 1 + Part 2.5. Part 2, the anatomy audit and the
> completeness audit are Sunday work and did not run.)

**2026-09-11 06:37–07:0x UTC — DAILY.** Window: the last 24 hours
(2026-09-10 06:37 → 2026-09-11 06:5x).

*The one sentence: **yesterday I told the owner and the builder that the day's
real finding was a one-week date error in the builder's steering; today the
finding is that I was the one who was wrong, the builder had it right, and the
correction I wrote into its live priority block was the error.***

> This page is the RECEIPT for commits that already exist. Every act below was
> committed as it was made, before this page was written: `4bd81ec`, `04d8b69`,
> `5e39771`, `57f67e6`, `3410b77`.

---

## The numbers

| | today | yesterday |
|---|---|---|
| demonstrated / registry | **108 / 245** | 108 / 245 |
| pass rate | **44.1%** | 44.1% |
| net demonstrated | **0** (3rd consecutive) | 0 (2nd) |
| rework rate | 77.6% | 77.6% |
| unreachable (shrink-only) | 93 of 245 (38%) | 93 |
| ledger settlements in 24 h | **0** | 0 |
| commits in 24 h | 19 — **0 from the builder** | 6 — 0 from the builder |
| consecutive skipped builder slots | **70** | 46 |
| `week:all models` | **72%** (line 63%) | 68% (line 54%) |
| `week:Fable` | **100%** | 100% |
| GPU, live week `2026-W36` | **17.72 h of 30 charged, 12.28 left** | *mis-stated as "W37: 0.00 of 30"* |
| queue violations | 0 | 0 |
| live queue rows | 38 | 39 |

Fourth day at 108/245. No builder commit since 2026-09-08T08:23.

---

## Part 1 — did the builder produce, thrash, or stall?

**Still none of the three: it was switched off, and today it is day three.**
`pace_gate` has skipped 70 consecutive hourly slots. What is new is not the
outage — it is that **the two headline findings this desk and the overseer
published about the outage have both now been withdrawn, within an hour of each
other, by their own authors.**

### The one that is mine, and it reached the builder's steering

Yesterday's page called this *"the day's real finding"*: the priority block said
`attempt 3 goes to W37 (opens 09-13)`, and I rewrote it to say **"W37 IS THIS
WEEK AND IT CLOSES SUNDAY 09-13"**, told the builder *"do not defer the
attempt-3 dispatch"*, and committed it into the file the builder navigates by
(`fd2101d`).

**That was wrong. `experiments/gpu.py::_week()` keys the GPU budget by
`%Y-W%U`, and `%U` weeks start SUNDAY** — deliberately. Its own docstring says
why: Kaggle's quota resets on Sunday, and the original ISO `%G-W%V` was removed
because it *"kept charging Sunday's runs to the exhausted week, so the tracker
refused jobs for the entire first day of every fresh Kaggle quota."*

```
  2026-W36   Sun 2026-09-06 -> Sat 2026-09-12     <- TODAY (Fri 09-11)
  2026-W37   Sun 2026-09-13 -> Sat 2026-09-19     <- opens Sunday, as written
```

In the only namespace the spending is accounted in, **W37 opens on 09-13,
exactly as the builder wrote it.** I read the label in the ISO calendar instead
of the tracker's. Withdrawn and the original text restored verbatim (`4bd81ec`).

**The builder's sentence carried its own arithmetic and I overwrote it.** It
read *"W37 (opens 09-13) — W36 has ~12.4 GPU-h left against attempt 2's measured
17.61 h, so it does not fit and must not be squeezed."* That is correct in every
part. My edit **kept that sentence and pasted a block contradicting it directly
underneath**, leaving the builder's live steering self-contradictory for 24
hours. Nothing was spent on it only because the builder never woke to read it.
That is luck, not process.

### The false alarm it generated, which ran through four documents in three days

| source | the claim |
|---|---|
| `PROGRESS.md`, 09-10 | *"W37 free GPU-hours charged **0.00 of 30**"*, in the numbers table |
| `fd2101d`, 09-10 | *"no `2026-W37` key at all — 0.00 of 30 … the week already 3 days gone"* → **into the builder's steering** |
| 87th audit, 09-10 | *"0.00 of 30 … expiring Sunday 2026-09-13 — before any projected wake"* |
| 88th audit, 09-11 | *"before W37's free GPU quota expires on Sunday 2026-09-13"* — carried forward uncorrected, one hour before this page |

**The key is absent because the week has not started.** Absence meant *not yet*
and four documents read it as *unspent*. The live numbers, from the tracker's
own accessor: key `2026-W36`, kaggle **17.7238 h of `KAGGLE_WEEKLY_HOURS=30.0`,
12.28 h remaining, expiring end of SATURDAY 09-12** — a day earlier than anyone
said, and a pot of 12.28 h rather than 30. **W36 at 17.72 h is the second-best
of the last six weeks** (W31 37.46, W32 21.06, W33 7.63, W34 1.62, W35 18.93).
It is not an allocation dying unspent; it was spent before the blackout began.

### The one that is the overseer's, found independently and better

The 88th audit, running concurrently and reading a different file, withdrew both
organs' **drain forecasts** — its own 49-day figure and my *"~6 days;
`pace_gate` never releases the builder at all"*. The gap closed **14 → 9 points
in the 24 h after both were published**, against my 2.3/day and its 0.29/day.
Its diagnosis is exactly right and I adopt it: we each sampled ONE day's draw
from a quantity whose own record reads **+21, +36, +9, +4** points/day, and
published the remainder as a rate. Its replacement — quote the **flat-meter
bound with its assumption stated**, never a point forecast — is the correct
instrument, and its re-specified B2 (print the bound and the spread, print no
single release date) is better than the 87th's B4 it replaces.

**Neither organ found the other's error. Each found its own.** That is the only
encouraging structural fact on this page.

**The release bound, stated as a bound:** meter 72%, elapsed 58%, line 63%, flat
at 72% for 15.5 h. Flat meter → **09-12T08:40**; at 09-10's measured external
draw → 09-12T18:40; at its measured total → 09-12T23:40; week resets
09-14T05:00. **Under every rate measured since the blackout began except the two
worst single days, the builder wakes on 09-12 — and W37's fresh 30 GPU-hours
open the next morning.** The builder wakes into a full quota, not a dying one.

---

## The day's science — `PL.02`'s eye gate, RULED

`pl02-eye-gate-reads-the-encoder-not-the-eye` was due today, was priced, and had
already been re-dated once by this desk as a **refusal** rather than a capacity
slip. Ruled (`5e39771`): **the pre-registered eye-aliveness VOID gate reads the
RAW-PIXEL ridge, not `U_A`'s 64-d features.** `EYE_RADIUS_R2_MIN` unmoved at
0.80, same VOID semantics, measured on the run's own probe episodes. `r2_ua` is
**not deleted** — it stays a first-class recorded metric on the ledger row.

**The decisive reason is one neither the 82nd audit's B4 nor the row stated, and
unlike their two arguments it is algebraic rather than interpretive:**

> `r2_ua` is the **SUBTRAHEND in the claim's own effect size** — the spec
> computes `R_pl = r2_pl − r2_ua`. A VOID gate demanding `r2_ua ≥ 0.80` demands
> a near-saturated baseline *before the run is allowed to count*, capping the
> largest reshaping gain the spec can ever report at **≤ 0.20**, against an
> observed **0.94**. It does not test whether the eye is alive; it algebraically
> suppresses the quantity it was added to guard, and is un-clearable by
> construction in the exact regime the claim exists to test.

**Both holes the old gate covered are already closed by instruments that exist,
so no redundant conjunct was invented:** dead channel — B4's actual worry — by
the new referent (a blind eye cannot produce the measured 0.9327/0.9438 pixel
ridge); audio leakage by the spec's **own declared `SHUFFLED` derangement
control**, measured clean (`shuffled_R` −0.0023, CI excluding zero from below,
`control_reshapes_too` 0).

**The honest cost, stated because it is the direction I refused on 09-09:** this
unblocks the sole registered falsifier of `GOAL.md`'s PLASTIC-ONLY decree, and
the run it unblocks already reads in the direction that *supports* the decree.
The convenience has not changed since 09-09; the argument has. The subtrahend
point holds whichever way `PL.02` falls. **A desk that refuses a correct ruling
because the correct ruling is convenient has not avoided bias — it has inverted
it.** Nothing weakened: no threshold moves, the control is untouched, the
registered run stays blocked behind a PASSING smoke, staleness bill zero.

---

## Part 2.5 — steering maintenance

**1. Priorities — the block was repaired, not reconciled.** Today's edit removed
damage this desk put there yesterday; the `1'''''`/`2'''''` units are otherwise
unchanged and still unspent at 70 skipped slots. The withdrawal note tells the
builder plainly that its own plan was right, restates the live W36 numbers
(12.28 h left, expiring Sat 09-12, against 17.61 h needed), and repeats that the
probe-and-gate precondition binds and an unchanged re-dispatch stays forbidden.

**2. Field watch — nothing to consume.** Last sweep 2026-09-07 (wk6, `3b68b7d`),
consumed in full by the 09-07 Review. Next sweep Monday 09-14. Unchanged.

**3. Seat staleness — carried, nothing new.** `Learning core` holds 3 trigger
debts (`LC.07` PILOT-BLOCKED, `LC.03` VOID-FORECLOSED, `UB.10` VOID), `World`
still reads no deciding run and no `TRIGGER:` declared, `Fast/slow coupling` is
welded behind `LC.03`, `D1` (Control architecture) remains VACANT. No seat's
arena context changed in 24 h — nothing ran. The missing language-**routing**
seat found yesterday stays routed to Sunday's anatomy audit.

**4. Organ liveness — all four alive, and one apparent silence is by design.**

| organ | cadence | last fire | verdict |
|---|---|---|---|
| builder | hourly | 06:07 today | alive; **70 of 70 slots fired and refused** |
| overseer | 6-hourly | 06:37 today, 88th audit `e112a34` | alive; **the once-daily rhythm is BY DESIGN** — `overseer.sh` pacing exempts exactly one audit per UTC day at/after the Review slot and `pace_gate`s the other three. One commit per day here is the gate working, not an organ going quiet |
| field watch | Mondays | 09-07, consumed | alive; next 09-14 |
| review | daily / Sun FULL | this run | alive |

**A live cross-organ write race, observed rather than inferred.** The 88th audit
held `docs/LESSONS.md` dirty in the working tree while this run was writing. I
did not touch it, and the `LESSONS.md` entry both of today's corrections deserve
is therefore **deferred to the next run, not dropped** — recorded in
`04d8b69` so it cannot evaporate. This is the live queue row
`cross-organ-doc-race-voids-certificates` (DUE 09-13) happening in real time.

---

## Dispositions committed this morning

**Four rows due. Four handled. Queue 0 violations before and after.**

- **`pl02-eye-gate-reads-the-encoder-not-the-eye` → RULED, DISPOSITIONED
  (`5e39771`), DUE 09-14.** Above. DISPOSITIONED and not ACTED deliberately: the
  ruling exists, the spec edit does not, and `ACTED` means *executed* and must
  name an executing commit.
- **`me1-similarity-floor-never-abstains` → 09-14 (`57f67e6`).** The 09-07
  pull-forward moved it EARLIER on the argument that *"09-13 carries ten rows
  while 09-11 carried one"* — **desk** capacity, for work that is entirely the
  **builder's** ("one harness edit plus a re-run"). It was dated onto a day the
  builder could not work by a desk counting its own load. Both `ME.1` and `ME.3`
  FAILs stand; the 0.95 bar does not move.
- **`t310-anticorrelated-gates` → 09-20 (`57f67e6`). Third slip, and I repaired
  the pattern rather than the date.** The 09-07 note already conceded the
  question *"needs the frozen-vs-plastic evidence re-read"* — FULL-sized work,
  now dated onto a DAILY three times and failed three times. **That is not three
  bad mornings; it is a row filed against the wrong kind of sitting.** Onto a
  FULL, and 09-20 not 09-13 because 09-13 already carries 14 against a capacity
  of 6. A longer slip than either previous one, not dressed as sequencing.
- **`five-commitments-are-claim-dead-behind-foreclosures` → 09-16 (`57f67e6`),
  on a dependency that is checkable rather than asserted.** Its own text names
  *"the re-parenting outcome"* as an input; that input is
  `reparenting-the-welded-fifteen`, bundled yesterday to DUE 09-15. Deciding
  today would mean deciding blind to the registry surgery that determines what
  the five commitments can be re-parented onto. The CLAIM-DEAD ratchet stays RED
  at 4 and `coverage` keeps exiting rc=2 until it is acted on.

**Ratchets, said out loud as the tool requires:** `review_queue_net_arrivals`
**6** (MOVED −23 since 09-08) and `review_queue_piled_on` **8** (MOVED +2).
**Neither moved by today's re-datings** — 09-14, 09-16 and 09-20 all sit under
the measured capacity of 6. The −23 is the drain of yesterday's seven-row
sitting; the +2 predates this run.

---

## The frontier

**`D1.0`'s twin-spread probe** is still the most important unblocked unit
(forward passes only, both branches pre-registered, DUE 09-14), standing in
front of `T2.01` — frees 35 / blocks 38, the largest single unblock in the
project. Untouched; nothing ran.

**The frontier's constraint is unchanged, but it is no longer compounded.**
Yesterday I reported the gate *plus* a date error that would have wasted the
release. The date error was mine, it is withdrawn, and what remains is the gate
alone — which, on the corrected bound, releases the builder on 09-12 into a
fresh 30-hour GPU quota opening 09-13. The builder's own deferral plan was
aimed at that window before either of us interfered with it.

---

## The honest paragraph

No numbers. Jack is exactly what he was on Monday, for the third day, and this
desk spent the third day discovering that what it told the owner and the builder
was wrong. Yesterday's page named the drift as *this project's errors are
migrating out of its instruments and into its prose* — and then, from the same
desk inside the same day, produced two more instances of it, one of which I
committed into the file the builder steers by while believing I was rescuing it.
The pattern both errors share is worth more than either: a number was read out of
the correct file and interpreted in the wrong frame. A drain rate sampled from
one window of a quantity that varies six-fold. A week label read in the ISO
calendar when its own module keys it Sunday-start, and says so in a docstring
written precisely because somebody got this wrong before. No instrument was
wrong. No threshold moved. Every ratchet held, the ledger is sound, the gates
bind — and all of that was true yesterday too, while the prose steering the
creature's only productive organ said the opposite of the truth. The week's
single most important step toward Jack is still the one made before this window.
The most concerning drift is the same sentence I wrote yesterday, now with the
author's name attached to it: the things that hurt us this week were a date, a
rate, a marker and a seat — and the one artefact this project produces that
nothing audits is the one I produce. The encouraging fact, and it is real: two
organs, reading different files with no coordination, each caught their own
error inside a day. Nobody covered for anybody. That is what oversight is
supposed to look like, and it is the reason both errors cost a page instead of
a dispatch.

---

## FOR THE BUILDER

**Your first act on waking is still a measurement, not a spec** — append one
line to `docs/LOOP_JOURNAL.md` recording how many consecutive slots you skipped
and the `week:all models` reading that released you. It is **70** and counting.
Nothing in this system counts a dark slot. Then, in order:

1. **YOUR W37 PLAN WAS RIGHT AND I BROKE IT. IT IS RESTORED — READ THE
   WITHDRAWAL, NOT THE CORRECTION.** My 09-10 edit (`fd2101d`) is withdrawn
   (`4bd81ec`). `W37` opens **Sunday 09-13**, as you wrote, because
   `gpu.py::_week()` keys by `%U` (Sunday-start). The live week is **`W36`,
   17.72 h charged of 30, 12.28 h left, expiring end of SATURDAY 09-12** — and
   attempt 2's measured **17.61 h does not fit in 12.28 h**, which is exactly
   what you said. **Do not scrape the attempt out of W36.** The precondition is
   unchanged and still binds under every branch: twin-spread result on the row,
   successor gate committed in a non-dispatch commit, and only then a dispatch.
   An unchanged re-dispatch stays forbidden.

2. **Three overdue armed defaults are yours to fire, and the queue is one
   longer than yesterday** — `D22` (overdue 09-09), `D18` (overdue 09-10), `D26`
   (overdue 09-11). `D26`'s default is `(iv) MEASURE ONLY`; both of the urgency
   arguments it was escalated on have now been withdrawn, which changes nothing
   about whether the default fires. Use the required journal wording: *"the
   owner did not rule by <date>, so the pre-registered default fired."*

3. **`PL.02`'s eye gate is RULED — implement it** (`5e39771`, DUE 09-14).
   Rebind the VOID condition to a raw-pixel radius ridge R² ≥ 0.80
   (`EYE_RADIUS_R2_MIN` unmoved) measured **on the run's own probe episodes**,
   not inherited from the seed-90 probe. Keep `r2_ua` as a first-class recorded
   metric on the ledger row. Then a smoke; the registered run stays blocked
   until the smoke PASSES, exactly as before.

4. **`W1.04` still gains conjunct (c) before you register it** — carried
   unchanged from yesterday. Register from the amended design, not the 09-06
   text. It stales nothing.

---

## FOR THE OWNER

**1. `D26` — both reasons you were asked to hurry have been withdrawn, by the
two organs that raised them, within an hour of each other. The recommendation
is unchanged.** Already routed; see `D26` and today's premise-correction
addendum (`04d8b69`). My recommendation stays **(i) ATTRIBUTE THE LINE**, on the
structural argument — `pace_gate` rations a shared meter with no attribution and
no ordering, and that recurs every time an external consumer draws, regardless of
how long *this* blackout lasts. What is withdrawn is the urgency: the 88th audit
withdrew the drain forecast (my "~6 days / never", its "49 days") as unsupported,
and this desk withdrew the GPU loss (the "0.00 of 30 dying Sunday" that appeared
in four documents, including one of mine in your builder's steering). On the
corrected numbers the override buys roughly **one day**, not six, and W37's fresh
30 hours open the morning after the builder is forecast to wake. **The overdue
default `(iv) MEASURE ONLY` is unaffected by any of this and should still fire.**
Nothing here argues for deciding `D26` differently — only for deciding it calmly.

**2. NO-DECISION: the withdrawal of my own steering edit, reported because it is
expensive and because it is mine to fix, not yours to rule on.** Yesterday I
committed a false correction into the file the builder navigates by, on a finding
I led this page with. It is withdrawn and the builder's original text is restored
verbatim. It cost nothing only because the builder has been dark for 70 slots and
never read it. I am telling you rather than asking you because there is no fork
here — but you should know that **the steering file has now been wrong in two
successive directions in two days, both times by this desk**, and that the
`LESSONS.md` entry it deserves is deferred rather than written because the
overseer held that file dirty mid-run (recorded in `04d8b69`, and it will be
written by the next run that finds the file free).

**3. NO-DECISION: the docket, announced rather than asked about.** Four rows due,
four handled, 0 violations before and after. One ruled, three re-dated on three
distinct reasons — and only one of those three was capacity. Sunday 09-13 still
goes to **14 rows against a measured capacity of 6**; I said a week ago it will
not clear and I have dated nothing new onto it. `review_queue_net_arrivals` is
**6** (MOVED −23) and `review_queue_piled_on` **8** (MOVED +2); neither moved by
today's acts.

**4. NO-DECISION: liveness report, nothing here to rule on.** All four organs
alive. The overseer's once-daily commit rhythm is its own pacing working as
designed, not silence — worth stating because "one audit a day from a 6-hourly
organ" is exactly the shape a dead organ would also make, and the distinction is
in `overseer.sh`, not in the log.

**5. `D22` and `D18` remain OVERDUE and unfired, and `D26` joins them today.**
Repeated verbatim rather than dropped, because a `VANISHED-OWNER-ASK` is the
scar this section exists to prevent. `D22`'s own default is **(i) THE RULE
STANDS**, which denies my own ask, and I withdrew the ask on 09-10 because its
premise is doubly falsified. If nobody fires it, it should be fired as (i).
