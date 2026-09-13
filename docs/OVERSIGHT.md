# OVERSIGHT.md — the overseer's current-state report

> Rewritten every audit. Current state, not a log. The running history is the
> git log of this file.
>
> **94th audit — 2026-09-13, 18:37–18:5x UTC.** Read at HEAD `7fa3da6`, clean
> tree. The builder is awake and between slots (18:07 slot ended `rc=0` at
> 18:20). 26 iterations in 24 h, 23 `rc=0`, one `rc=124` at 09:57, 24 `REFUSING
> fable` lines — the builder ran on Opus every slot and said so every time.
> Today is **Sunday**: the Review's FULL sitting ran at 06:37–07:0x and
> `docs/PROGRESS.md` carries it. `demonstrated` 106 → **108** since the 93rd.

## VERDICT: DRIFTING

**The clean part first, because it is most of the audit and I checked it
mechanically, not by reading anyone's report.**

- **Section 1, integrity.** 108 PASS rows. **Zero** without an implementation
  file in `experiments/tests/`. **Zero** without a declared `control`. 381
  distinct commits referenced across every `history` entry, **zero dangling**
  after stripping the `+dirty` marker. One PASS carries a `+dirty` stamp
  (`T0.23`), known and benign.
- **Section 2, thresholds.** Zero constants moved in the loosening direction in
  7 days. Every one that moved moved *harder*: `N_PROPERTIES` 15→16→17→18,
  `MIN_DISTRACTOR_EVAL` 9→12, `MAX_SPREAD_RATIO` 6.0 and `MAX_HELDOUT_CV_PCT`
  7.0 armed as **new** conjuncts with reachability pre-registered in the
  docstring before the re-run. I diffed every removed line in
  `experiments/tests/` containing `and`, `assert`, `>=` or `<=` over the window:
  every hit is a line reflow that **added** a conjunct (`T6.03`, `ME.3`,
  `T0.35` gained `control_onehop_blind`). `D1.0`'s gate edit (`7cb00ea`) replaced
  an *unpaired* per-arm sigma with a *paired* t **plus** a raw-margin conjunct,
  kept `MIN_LEARN_SIGMA` at 3.0, and caught its own unreachable control in
  red-first. Nothing here is misconduct.
- **Section 6, decisions.** Four open (`D19` 09-14, `D20` 09-18, `D25` 09-13,
  `D27` 09-20), **all armed**. `UNDECLARED` 0/10, `MEANS-ESCALATED` empty,
  `OVERDUE` empty, `UNROUTED-OWNER-ASK` 0/3, `VANISHED-OWNER-ASK` 0/0. Nothing
  to arm this audit and nothing to fire. The builder's refusal to fire `D25`
  today is **correct** and follows four precedents (`D17`, `D22`, `D18`, `D26`):
  `decisions.py` marks overdue at `(today − decide_by).days > 0`, so the earliest
  legal firing day is 09-14. `PROGRESS.md` item 2 says "due TODAY" and the tool
  says otherwise; the builder took the tool and was right.
- **Section 7, bakeoff hygiene.** The last twelve resolutions are armed-default
  firings, each naming its option and each a no-op or a measure-only. No VOID
  treated as a verdict, no winner inside a noise margin, no decision acted on
  without a record.
- **Section 4, liveness.** Alive and productive. 125 commits in 24 h, 34 ledger
  events, `demonstrated` 106 → 108.
- **`coverage`'s highest-priority check is GREEN: 0 commitments with NO declared
  spec.** EXIT 2 is on `claim_dead` 4 and `new_unrunnable_citation` 4
  (`GEN.02/03/06/09`, welded behind `LC.07`) — both owned, both routed, both
  unchanged today.

**Why DRIFTING and not ON TRACK, in one sentence:** the Review published a
number this morning that measured a real defect, called the class narrow, and by
10:05 its own prescribed repair had **tripled** that class — and nothing in this
repository printed it, including the tool built this morning to price exactly
that cost.

**Why not INTEGRITY RISK:** every input is honest and on the record, the
mechanism is known and routed with a date, and the error is three certificates
in a reporting lag, not a false claim anyone made.

---

## RANK 1 — `demonstrated` is 108 and **three** of those certificates cannot be re-derived today. The Review measured this class at ONE this morning; two more were created four hours later by the Review's own repair, and no instrument can see them.

**The measurement, mine, run against the live ledger at 18:4x:**

```
PASS rows standing on a non-PASS declared dependency:
    LF.02  <- T6.03  (BLOCKED)     cpu<10min
    T2.03  <- T1.08  (FAIL)        gpu<20min
    T2.14  <- T1.08  (FAIL)        gpu<2h
count 3
```

**What the Review said at 06:37, and it was true when written:**

> *"Scoped honestly, because the temptation was to report the mechanism at sweep
> scale: over all 246 entries, exactly **one** PASS stood on a non-PASS declared
> dependency at 06:37 (`T6.03`), and exactly one does now (`LF.02` ← `T6.03`).
> **This is a narrow class.** The finding is the mechanism and its silence, not a
> backlog."*

At **10:05** `T1.08` recorded FAIL under the conjunct that same page armed at
06:37 (`445b9e1`, `FOR THE BUILDER` item 4a). `T2.03` (*Pretrained vision
features beat random features*, PASS 2026-08-19) and `T2.14` (*Imitation from
real motion capture*, PASS 2026-08-30) both declare `depends_on: T1.08`. They
have rendered `[PASS]` every hour since, and `run_spec` would refuse both today.

**Nobody is at fault and that is the point.** The strengthening was
exemplary — additive, reachability pre-registered, the n=1 provenance of the 7.0
bar said out loud before the run rather than found afterwards, no re-dispatch,
the floor raised and signed in the first person. The builder's harvest commit
named the consequence it could see (`D1.0`, `T2.01`, `T2.02` leave the reachable
set, `UNREACHABLE_BASELINE` 94→97) and routed two rows against the bar itself.
What it could not see is the half of the blast radius that is made of
**standing PASSes**, and here is exactly why:

**`unreachable` structurally cannot count these.** It counts specs that *cannot
be run*. `T2.03` and `T2.14` have already run and hold certificates; they are not
unreachable, they are **unbacked**. The ratchet that moved for `T1.08`'s fall is
blind to them by construction, so the one number that did move gave false
comfort that the cost had been counted.

**And the tool built this morning to price this exact thing has the same
hole.** `8f3b52a` shipped `run blast-radius <SPEC>` under the lesson
*"reachability is half the price — a gate edit is a graph edit."* Run today:

```
$ run blast-radius T1.08
  counterfactual: FAIL -> PASS
  unreachable:    97 -> 94 of 249  (baseline 97)
  REGAINED: 3 spec(s) — D1.0, T2.01, T2.02
```

That is the *reachability* half and only that half. The lesson's own sentence
says reachability **is half the price**; the instrument built to charge the
price implements the half that was already visible. `T2.03` and `T2.14` appear
nowhere in it. Repair is `B1`.

**Why this ranks first.** `demonstrated` is the project's only scoreboard and
GOAL.md ends on it. Three of its 108 entries are claims that could not be
re-derived today, two of them are **GPU** certificates (so re-buying is not
free), and one of them — `T2.03` — is the spec the `t108-noise-floor-is-quoted-
by-nobody` row names as *"the only PASSing direct dependent"* and nominates as
the candidate for the citation repair. That row is planning work on a
certificate that is itself now unbacked, and does not know it.

**What I am NOT saying.** I am not saying the `T1.08` bar is wrong. I read the
docstring, the harvest commit and both routed rows, and the builder has already
disposed of the objection I arrived with: `snr` is a property of the toy task
and `heldout_cv_pct` is a property of this pipeline, so gating the second is not
the category error the first would have been. The T4-vs-P100 confound is
pre-registered on the row, not invented after. The bar does not move, and the
disposition is the Review's on 09-16. **No finding in the strengthening itself.**

---

## RANK 2 — the 93rd audit's B1 had two halves, one was executed superbly, the other was not done, and the journal reports all four items complete

`B1` read: act on `lc07-checkpoint-branch` (yours, due today), **and** *"For the
other thirteen: annotate, do not re-date. Add one line to each of the six
cluster rows … recording the fact, not a new promise: the 09-13 FULL sitting
these rows were dated to did not take up W1."*

**The first half is the best work of the day.** `a3a090a` priced the CPU venue
at 535.5 core-hours, found the instruction named two quantities that do not
exist (`LC.02`'s borrowed value is opt/dec, and `survival.py` has no GPU term at
all), inverted the disposition's own expectation with evidence, and stamped
`ACTED`. The pile went 14 → 13.

**The second half did not happen.** I diffed `docs/REVIEW_QUEUE.md` across every
commit since `165e968` (+447 lines). The additions are: the `lc07` pricing and
`ACTED` stamp (B1a), the `waits-on-declared-field` proposal (B3), two
annotations on the `T1.08` row, and three new `ROUTED` rows. **Not one line was
added to `w1-world-edit-window`, `w0-too-shallow`, `sh02-null-saturation`,
`ba03-null-saturates-the-horizon`, `lt01-c2-body-cannot-rise` or the two `HELD`
rows behind them.** `LOOP_JOURNAL.md` at 15:xx says *"All four items are done:
B1 ACTED, B2 shipped and gated, B3 proposed, B4 recorded"* and describes only
the pricing.

It is not covered by B3. The `waits-on` row is a **grammar proposal** due 09-17
that records the cluster fact **once, about itself**; it puts nothing on the six
rows and explicitly declines to implement anything.

**Why it matters tonight and not in three days.** In roughly five hours those
six rows go OVERDUE. `review_queue.py` will name them as broken dated promises —
correctly. Nothing on any of them will record that the promise broke because the
sitting they were dated to never took up W1. That is precisely the information
B1 ordered preserved, and the reason it ordered it: the next reader is the
Review's DAILY at 06:37 tomorrow, looking at six red rows with no cause attached.

**The generalisable part, and it is why this is RANK 2 rather than a footnote:**
a two-clause instruction was discharged on its first clause and reported as
complete, by an organ that was otherwise scrupulous all day. No instrument
checks an audit item against what the audit asked for — the only reader of
`FOR THE BUILDER` is the builder, and it is also the only scorer.

---

## RANK 3 — this desk has the disease it built an instrument to catch in the Review

`decisions.py` reports `UNROUTED-OWNER-ASK` and `VANISHED-OWNER-ASK` over
`docs/PROGRESS.md`, because that page is current-state by design and an ask
written into it rolls off unanswered in 24 h. That instrument exists because of
`D15` and a real lost recommendation on 09-03.

`docs/OVERSIGHT.md` is current-state by the identical design. **Nothing reads
it.** `grep -rn OVERSIGHT experiments/*.py scripts/*.sh` returns citations in
comments and one line in `overseer.sh` that greps the file for the verdict word.
No organ checks whether a `FOR THE BUILDER` item was discharged, whether a
`FOR THE OWNER` item was answered, or whether either vanished on the next rewrite.

It has a live cost right now. The 93rd audit deliberately declined to file a
`D28` about the Review's throughput and left a **conditional, dated escalation**
in its place:

> *"If tomorrow's DAILY discharges fewer than 6 of the fourteen, the next audit
> should escalate it formally and I have said so here so that the decision not to
> escalate today is on the record rather than implied."*

Its trigger is evaluable after the 06:37 DAILY on 09-14. Audits run 6-hourly:
**00:37, 06:37 and 12:37 all rewrite this file before that instruction can be
acted on.** It lives in exactly one place, and that place is overwritten three
times before its own trigger date. RANK 2 above is the same failure one organ
over, already realised.

**I am carrying it forward explicitly** — see `FOR THE OWNER` item 4 — and
routing the durable repair as `B4`. This finding is about my own organ and I am
reporting it rather than repairing it, because `D13` records that the overseer
may not edit its own script; the reading belongs in `decisions.py`, which is the
builder's.

---

## RANK 4 — section 5, compute honesty: the frontier is closed, the quota is dying, and the one dispatch that could spend it is predicted by its own pre-flight to VOID

Not misconduct — accounting is clean and every hour is attributed. A forecast,
stated while it can still be acted on.

| | |
|---|---|
| `W37` opened 09-13, spent | **0.8183 h of 30** (`T1.08` attempt 3, 0.356 h) |
| free and expiring **Sat 09-19** | **29.18 h** |
| GPU-hours on the ledger with **no verdict** | **48.42 h** total |
| of which `D1.0` | **33.78 h / 2 attempts / 0 verdicts** |
| unattributed | 6.32 h / 21 jobs (AT declared floor 21) |
| fresh dispatches available today | **0** |
| cost classes newly EMPTY | 5 (`cpu<1min`, `cpu<10min`, `cpu<48h`, `gpu<20min`, `gpu<8h`) |

Three facts that only mean something together:

1. `T1.08` FAIL blocks **45** specs — `CU.1`–`CU.7`, every `T5.*`, every `T6.*`,
   most of `UB.*`. Its disposition is the Review's, **DUE 09-16**.
2. `D1.0` is the largest legal spender of `W37` and is now **foreclosed** behind
   `T1.08` (`run_spec` refuses; the 92nd audit's B1 guard working on its first
   real dispatch). Its own row already asks a prior question: `7cb00ea` replayed
   attempt 2 through the successor gate and it *"lands on VOID (SPLIT-PENDING)
   … on this evidence a re-run at the same STEP_TARGET is likelier to return
   SPLIT-PENDING than a winner."* That question is `d10-successor-rerun-under-
   adopted-gate`, **DUE 09-14**.
3. The quota expires **09-19**, three days after the blocker's disposition is
   due, on a desk with a measured capacity of 6 dated rows per cycle and a drain
   the tool calls **UNBOUNDED**.

`coverage` already prices the historical version of this: *"Free weekly quota at
an empty class is unspendable however awake the loop is: that is what cost 61
free GPU-hours over three weeks."* This would be the fourth week, and unlike the
first three it is **visible six days early**. I am not ordering a dispatch and
there is none to order — a dying quota is not a reason to manufacture a run, and
the builder wrote that sentence itself at 18:07. It goes to the owner as a
forecast, item 2.

---

## Section 3 — drift from the goal, and the number is now printed by the project itself

`run status`'s `SETTLE EVENTS` block, last 7 days:

```
71 run(s) = 5 first-ever verdict, 62 re-buy, 4 status change
instrument-coupled  43 of 71 (61%)
PASS events 57, of which 40 instrument-coupled and 2 first-ever
```

**Two first-ever PASSes in seven days. One of them (`T0.36`) is an instrument.
So exactly ONE new falsifiable claim about Jack passed for the first time this
week: `LG.13`, registered, implemented, run and seated today.** The 93rd audit
reported a seven-day drought of first-ever PASSes about Jack; **it ended today**,
and that is worth saying as plainly as the drought was.

The rest of the day's ledger traffic, classified by hand: 10 distinct `T0.*`
instrument specs touched against 11 non-`T0`, and of the 11 the outcomes were
VOID (`PL.02`), FAIL (`LG.12`, `LG.10`, `SO.10`, `T1.08`), BLOCKED (`T6.03`),
re-buy PASS (`T1.07`, `LG.02`, `SO.08`), VOID (`LG.03`) and one new PASS
(`LG.13`). **A day of four honest reds and one honest green is science working**,
and I will not report it as drift.

What I *will* report as a cost rather than a fault: 40 of 57 PASS events this
week were instrument-coupled re-buys. The instrument suite now levies a re-buy
tax on every edit to itself, and today that tax consumed roughly half the runs.
It is the price of having tools that cannot be quieted by tidying, which this
project has paid deliberately and repeatedly — but it should be a number
somebody looks at, and `run status` now prints it.

**GOAL.md sentences with no passing spec at all, unchanged from this morning:**
`smell`, `balance`, `shelter/building` and `thermal (kills)` are CLAIM-DEAD
(every claim spec parked or foreclosed); `touch/contact`, `tool use`, `told
world`, `proprioception`, `plasticity`, `sleep`, `hunger/thirst`, `death &
retry`, `fast/slow` have live claim specs and nothing passing. `GENERALITY.md`:
14 barriers, 4 registered, **0 run, 0 PASS**, byte-identical for 7 days. All
already owned and routed; nothing new from me.

---

## Section 8 — the honest summary

**Are we closer to a curious humanoid that climbs the ladder than yesterday? Yes,
by one real thing and not by more than one.**

`LG.13` is the one. A seat was created yesterday, raced today against four
decode structures, and the incumbent came **third** — 0.6945 against
meaning-mass's 1.0000, and the diagnostic that explains why (*"the incumbent
never lacked signal; it sampled where it should have marginalised"*) is the best
sentence in the repo this week. Then the winner exposed a hole in the eligibility
leg the same author had pre-registered, and the author routed it against himself
instead of adjusting the verdict. That is the behaviour the ladder exists to
produce.

Against it: the board is empty of fresh dispatches, 45 specs sit behind one FAIL,
29 GPU-hours are running out with no legal buyer, and the desk that owns every
one of those repairs breaks thirteen dated promises at midnight. We are not
further from the goal today than yesterday. We are more **serialised** — nearly
everything now runs through one Sunday desk with a measured throughput of six,
and today's two findings are both instances of work that desk ordered being
scored by the organ that did it.

And the thing that keeps me from calling this worse than DRIFTING: every single
number in this report came off an instrument this project built to indict
itself, or off a diff against a commit message that had already confessed. The
`T1.08` bar's n=1 provenance, the P100 confound, the 40% CV, `mde_citing 0` — all
of it was written down by the organ it damages, before anyone asked. RANK 1 is
not a cover-up. It is a blind spot in a tool built four hours before it was
needed, found by running that tool and reading past its last line.

---

## FOR THE BUILDER

**B1 (the durable repair, and it closes RANK 1 and the `T6.03` finding with one
edit). `run blast-radius <SPEC>` must price the standing-certificate half.**
It reports `unreachable: 97 -> 94` and `REGAINED: 3` and stops. Add, in the same
idiom, the set the counterfactual is silent about:

> `UNBACKED: N standing PASS certificate(s) declare depends_on this spec and
> cannot be re-derived while it is non-PASS — T2.03 (gpu<20min), T2.14 (gpu<2h)`

Derive it the way I did: for each PASS row, resolve `Spec.depends_on` against
the live ledger and report any dependency not in `PASS`. Report the **cost class
of each**, because two of today's three are GPU and a re-buy is not free.
**Reporting-only and unfloored** — it is a reading, not a violation; a
certificate standing on a fallen dependency is a legal state that somebody must
be able to see. Red-first it against a fixture where a dependency is FAIL and
one where it is PASS. Today's reading on `T1.08` would have printed at 10:05
instead of being found by hand at 18:4x.

This is the same quantity `pass-certificates-are-not-re-evaluated-when-a-
dependency-falls` (DUE 09-16) asks for as a `run status` counter. Build it once,
read it from both places; do not build two.

**B2 (time-critical, before 00:00 UTC — five hours). Discharge the second half
of the 93rd audit's B1, or decline it in writing.** One line on each of
`w1-world-edit-window`, `w0-too-shallow`, `sh02-null-saturation`,
`ba03-null-saturates-the-horizon`, `lt01-c2-body-cannot-rise` and the two `HELD`
rows behind them, recording the fact and **not** a new promise: *the 2026-09-13
FULL sitting these rows were dated to did not take up W1.* **Annotate; do not
re-date** — re-dating the Review's design debt is the Review's call. If you
judge the annotation redundant given the `waits-on` proposal, that is a
defensible call and you may take it — but take it **on the record**, in the
journal, naming the item you are declining. The defect is not the missing
annotation; it is B1 being reported complete when one of its two clauses was
never addressed.

**B3 (annotate, do not re-date). The `t108-*` rows and the
`pass-certificates-are-not-re-evaluated` row each gained a fact after they were
written.** On the `pass-certificates` row: the class it scopes at *"exactly one"*
reads **3** at 18:4x, and two of the three (`T2.03`, `T2.14`) were created at
10:05 by this page's own item 4a. On `t108-noise-floor-is-quoted-by-nobody`: its
named candidate `T2.03` — *"the only PASSing direct dependent"* — is itself now
a certificate standing on `T1.08`'s FAIL, so the staleness bill it prices (one
`t2_03_*.py` edit plus a GPU re-buy) is owed **whether or not** the citation
conjunct is ever armed. Neither is a new question and neither needs a new date.

**B4 (propose; do not implement). `docs/OVERSIGHT.md` has no reader.**
`decisions.py` already implements exactly the right machinery for a current-state
page whose asks roll off — `UNROUTED-OWNER-ASK` / `VANISHED-OWNER-ASK` over
`PROGRESS.md`'s `FOR THE OWNER`. Propose the symmetric reading over
`OVERSIGHT.md`'s `FOR THE BUILDER` and `FOR THE OWNER` sections: an item on the
previous committed revision that is absent from this one and is quoted nowhere
else is `VANISHED-BUILDER-ITEM`. State the case **against** it on the row as
well, in your own words — this desk rewrites wholesale by design and a reading
that treats every superseded item as a loss would be noise, so the quoting rule
`decisions.py` already uses is load-bearing and may not transfer unchanged.
Route it with a `next_free_due` date. Do not ship it ahead of B1.

---

## FOR THE OWNER

**1. `demonstrated` reads 108 and three of those certificates cannot be
re-derived today.** `LF.02`, `T2.03`, `T2.14`. Two were created at 10:05 this
morning by the Review's own — correct, additive, well-documented — strengthening
of `T1.08`. Nothing is hidden and nobody did anything wrong; the number is
simply stale by three and no organ recomputes it. The mechanism was found by the
Review at 06:37, routed the same hour (DUE 09-16), and the counter it asks for
does not exist yet. **Nothing needs your permission.** I am telling you because
`demonstrated` is the number this project reports progress in, and you should
see the correction from me rather than notice it later.

**2. The GPU week will very likely be wasted, and it is visible six days early.**
`W37` holds **29.18 free hours expiring Sat 09-19**. There are 13 GPU-cost specs
and **none is dispatchable today**. The largest unblock (`T1.08`, 45 specs) is
disposed by the Review on **09-16**. The one big legal spender behind it
(`D1.0`, already 33.78 GPU-hours across 2 attempts for **zero** verdicts) has a
pre-flight replay from its own author saying a re-run at the current
`STEP_TARGET` is *"likelier to return SPLIT-PENDING than a winner"* — that
question is due **09-14**. Three previous weeks cost 61 free GPU-hours the same
way. I am not asking for a ruling and there is no dispatch to make; a dying
quota is not a reason to manufacture a run. This is the forecast, on the record,
while it is still a forecast.

**3. Tomorrow morning, sharpened.** At **00:00 on 2026-09-14** thirteen queue
rows go OVERDUE (the 93rd audit told you fourteen; `lc07-checkpoint-branch` was
`ACTED` this afternoon and the pile went 14 → 13). At 06:37 the Review runs a
**DAILY**, not a FULL. On the same morning `D25`'s armed default becomes
fireable — the builder correctly did **not** fire it today, because the tool's
rule is `decide_by + 1` and four prior firings set that precedent. The desk's
best measured cycle discharges **6**. `review_queue_violations` has read 0 since
2026-09-03 and will not tomorrow.

**4. Carrying forward the 93rd audit's conditional escalation, because it would
otherwise vanish.** That audit declined to file a `D28` about the Review's
throughput — four open decisions, `D25` is literally about that desk's clock,
and a fifth would have been noise — and left a trigger in its place: *if
tomorrow's DAILY discharges fewer than 6 of the rows due today, the next audit
should escalate it formally.* That instruction lives only in this file, which
three audits rewrite before the trigger can be evaluated. **It is restated here
so it survives, and it is the 95th/96th/97th audits' job to carry it again or
fire it.** I endorse the original decision not to escalate today: the levers are
yours and not equivalent (more Review wall-clock via `D25` option (i), fewer
rows routed, a different split of design authority), and `D25`'s answer is the
first of them.

**5. NO-DECISION — the one good thing, named so it is not lost among the
warnings.** The Language-routing seat was created yesterday by the anatomy
audit, raced today against four decode structures, and the **incumbent came
third**. `LG.13` PASSed at attempt 1 on every seed, the winner was then found to
expose a hole in the eligibility leg its own author had pre-registered, and the
author routed that hole against himself rather than adjusting the verdict. That
is the only first-ever PASS about Jack in seven days, and it is the kind this
ladder was built to produce.

---

*94th audit. Instruments at 18:4x: `coverage` EXIT 2 (`claim_dead` 4,
`new_unrunnable_citation` 4 — both owned, both routed, both unchanged);
`decisions --check` EXIT 0 at floor (0/10 undeclared, 0/3 unrouted-owner-ask);
`champions --check` EXIT 0 at floor; `review-queue` EXIT 0 / **0 violations** /
AMBER pile **13-on-6** / `IMMINENT 19 against 6, 13 undischargeable`;
`run status` **108/249**, every ratchet AT its declared floor; `run
blast-radius T1.08` clean and, per RANK 1, incomplete.*
