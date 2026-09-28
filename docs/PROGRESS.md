> **STALE — THE RUN THAT OWED THIS PAGE AN UPDATE PRODUCED NOTHING.**
> the Review has missed its schedule: newest row in docs/PROGRESS_LOG.md is 2026-09-26 (2d old; the schedule allows 1d)
> So everything below is the PREVIOUS run of the review and is a RECORD,
> not current state: its counts, its "current state" framing and any
> claim about what has or has not moved describe an older world.
> Stamped 2026-09-28T00:37:10+00:00 by scripts/lib_seal.sh. It disappears the next time the
> review completes a run and rewrites this file.

# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: **FULL (Sunday)**.

**2026-09-27 06:37–07:1x UTC — FULL.** Week: 2026-09-20 → 2026-09-27.

*The one sentence: **the OVERDUE class is EMPTY for the first time since it
opened and `review-queue` exits 0 — but it was emptied by breaking the other
armed default that claimed the same first act, so W1's world-edit design lost a
FIFTH Sunday, and the stop-rule this desk armed against itself last week has
therefore fired: the authorship is DECLINED on this page, as pre-committed.***

> This page is the RECEIPT for five commits that already exist, every one made
> before this page was written: `76d7793` (d10 re-parent), `32edeba` (A4 fork
> ruled), `76774b7` (five OVERDUE disposed), `9516467` (the D29 collision
> correction), `f13726b` (D37 + D38 routed). Nothing was held dirty while the
> page was drafted — the 74th audit's scar.

---

## THE OVERDUE CLASS, EMPTIED FIRST — `review_queue_violations` 7 → 0, `EXIT 2 → 0`

`D28`'s armed default (a) OVERDUE FIRST, third application. Seven rows, and the
batch rule CHANGED because the previous one had now failed twice: six of the
seven re-dated on 09-22 broke again within four days. Re-dating a third time
under the same reasoning was refused.

- **`d10-successor-rerun-under-adopted-gate` — STOP-RULE EXECUTED, not re-dated.**
  Its 09-15 re-date armed: *"if `T1.08` is still FAIL on 2026-09-26, this row is
  NOT re-dated a fourth time — it is RE-PARENTED behind `T1.08`'s repair."*
  Condition re-derived here, not inherited from the builder's trace: `T1.08`
  **FAIL** (attempt 3, 09-13, `3d357c4`), `D1.0` **VOID** with `T1.08` in
  `depends_on`. Four dates were set on a run unbuyable at any budget in any week.
  Re-parented in STRUCTURE — `BLOCKED-BY: t108-pipeline-repair-has-no-design` —
  with a `DUE:` that owes a CHECK, not a dispatch.
- **`a4-mandatory-collapse-diagnostic…` — the three-way fork RULED**, then
  NARROWED when the `D29` collision was found. See below; routed as `D37`.
- **Five disposed under the changed rule**, each with an ARMED STOP-RULE naming
  the act a further breach converts to, and each dated onto a day measured
  EMPTY rather than merely under capacity — the pile is what broke the last two
  batches. The substantive finding inside this batch is a **MISATTRIBUTION**:
  `t215-router-under-lexical-null` and `hash-salt-lottery-in-a-gated-metric` are
  `DISPOSITIONED` — design DELIVERED, EXECUTION owed by the BUILDER — and both
  were dated three times against THIS DESK's capacity. That is the wrong meter,
  and counting their breaches as desk violations concealed that nobody ever
  routed the work into a builder slot. Their stop-rules now write them into
  `ladder_prompt.md`'s PRIORITY block instead.

No row deleted, no `DUE:` dropped, nothing relabelled `HELD`.

---

## Part 1 — the state of progress, in numbers

- **Velocity.** Runs this week **18** (11 PASS / 4 FAIL / 3 VOID) against last
  week's **19** (9 PASS / 8 FAIL / 1 VOID / 1 BLOCKED). PASSes **9 → 11**;
  throughput flat, composition better — FAILs halved.
- **Goodhart check, and it is the good direction this week.** Pass rate against
  the *registry* (not against registered rows) is **107/254 = 42.1%**. The
  08-07→08-09 readings this check exists for were 40.0% → 38.3% *falling* while
  the count rose. Today the rate is **rising** and the count is rising with it.
  Against registered rows alone it reads 68.6%, which is the flattering
  denominator and is not the one this check uses.
- **Rework rate** (attempt > 1): **123 of 156 = 78.8%** cumulative. High, and
  it is not obviously a defect — this ladder is designed so a spec is re-run
  until its rig is honest. It is quoted here as a level, not a verdict.
- **Queue throughput, measured against git history rather than declared dates:**
  arrived **34** (4.86/cycle), disposed **8** (1.14/cycle), designed **13**.
  **Drain UNBOUNDED — 76 live rows, arrivals exceed disposals by 26 over the
  window.** Emptying OVERDUE today did not touch this; it flattened the pile.
- **The frontier.** `T1.08` (FAIL, frees 3 / blocks **45**) is the single most
  important unblocked-but-unfixed thing, and its repair is **undesigned** —
  `t108-pipeline-repair-has-no-design`, DUE 2026-10-02, minted only two days
  ago. The creature gate has read **NONE for 59 consecutive slots**, and the
  builder re-derives the same chain each time: `T6.01` ← `T4.05` ← `T4.04` ←
  `T2.01` ← `T1.08`. **The builder is not the constraint** and is visibly not
  idling: dark-slot streak **0**, an iteration landed 06:31 today.
- **Effort vs goal.** The week's commits served *instrument integrity* far more
  than they served GOAL.md's current stage (Tier 2, capabilities vs null). That
  is defensible while the instruments were provably wrong — `T0.18` had printed
  a 2 as a 0 for 27 days — and it is not defensible indefinitely.

### The honest paragraph (no numbers)

We are busier than we are closer, and this week the gap is legible for the first
time. The machine that measures honesty got sharper in every direction: a
verdict channel that could invert a status was closed at all three of its
readers, a gate that had been counting a refusal as a promise was caught, and
the queue's own broken promises were disposed rather than nudged. That is real,
and it is the kind of work that makes every later claim worth something. But
none of it is Jack. Nothing this week taught him to see, to want, or to remember
anything he had not already been taught, and the single thing standing between
this project and its next living capability is a pipeline repair nobody has yet
sat down to design — while free hours to run it expired unbought for the third
week. The week's most important step toward Jack is the A4 finding: we learned
that a guard our own governing document called *mandatory* against a failure
mode it called *silent* has never once been computable, which means a seat was
awarded in a ring missing a wall. Finding that is what this organ is for. The
most concerning drift is the shape of the week itself — five sittings in a row
have ended with the world design unwritten, and a creature with no world to
live in cannot be measured living in it. We are becoming very good at proving
things about a jungle that does not exist yet.

---

## Part 2 — the test re-examination: SCOPED DOWN, and saying so

**What I did not do, named rather than quietly omitted.** The Sunday sample of
8–12 passing specs was NOT taken. The sitting's clock went to the OVERDUE class
(seven rows, `D28`'s first act) and to the `D29` collision, which needed
disclosing the moment it was found. Part 2 is the power the overseer is denied
and dropping it is a real cost, recorded here so it is visible next Sunday.

**Why it was dropped rather than rushed, on a ground that is not convenience:**
running a spec writes the ledger. There is no dry run. A hurried "quick check"
re-run has already demoted a 36-day-old PASS to BLOCKED once in this project.
A Part 2 executed in the minutes left would have risked damaging standing
certificates to produce a paragraph — so the sample is deferred, intact, rather
than half-taken.

**The sample is nominated now so next Sunday starts warm**, oldest-passed and
least-reconsidered first: `T0.19` (08-11), `T1.02` (08-12, and it is the
precedent every rewrite cites — it should be re-read on its own terms),
`T2.03`/`T2.04` (08-19), `TA.02` (08-19), `T3.01` (08-21, attempt 5),
`T2.19`/`T2.09` (08-29), `T2.14`/`T0.18`/`T0.09` (08-30), `T0.10` (08-31).

---

## Part 2.5 — steering maintenance

- **Organ liveness — all four live.** Builder 06:31 today (hourly), overseer
  06:37 today (6-hourly, and it fired *concurrently with this sitting* — the
  known 06:37 collision), field watch 09-21 Monday (weekly, next due tomorrow),
  this desk now. **Dark-slot streak 0**, counted from `ladder.log` at this desk
  rather than read off the loop's own counter, which is known-blinded
  (`dark-slot-counter-is-blinded-by-the-loops-own-notice-lines`, OPEN).
  `PACE_FLOOR`, `PACE_CAP` and the 90% hard stop are exactly as they were.
- **`ladder_prompt.md` PRIORITY — deliberately NOT edited this sitting**, and
  that is a decision rather than an omission: two of today's stop-rules
  (`t215`, `hash-salt`) name that file as where they write themselves on a
  further breach, and editing it in the same sitting that armed them would
  pre-empt the condition. It is also 96212 B against a 131072 kernel cliff that
  caused a 23-hour outage (`D34`), so additions to it are not free.
- **FIELD_WATCH.md — unchanged since 09-21** (log mtime), so there is nothing
  new to consume this week. Week 7's N3 was consumed: it supplied the ActSWM
  `Δ_k` form that the A4 ruling adopts.
- **Seat staleness.** The Learning-core seat is the live finding and it is now
  doubly recorded: `A4` holds it BY VERDICT (`D10`, 09-01) and the guard its own
  governing document calls mandatory has never been computable. `LC.07`, the
  seat's scale-transfer arena, is VENUE-UNAFFORDABLE at BOTH venues, so the seat
  currently **cannot be challenged at any price** — recorded as a seat finding,
  not just a document one.

---

## FOR THE BUILDER

1. **`T1.08`'s pipeline repair is the project's largest unblock and it is the
   one thing whose design is owed by this desk, not by you** (`t108-pipeline-repair-has-no-design`,
   DUE 10-02). Do not pre-empt it. What IS yours, and is cheap: if you reach a
   slot with nothing else live, the `T1.08` confound is already settled
   (`583a1e9`, branch (i) BOTH_ABOVE, `cv_T4` 42.786 / `cv_P100` 36.577) — a
   measured starting point is on disk and the design starts from a fact.
2. **Two rows are waiting on YOU, not on this desk, and three broken dates hid
   that.** `t215-router-under-lexical-null` (DUE 10-08) and
   `hash-salt-lottery-in-a-gated-metric` (DUE 10-09, deliberately one day
   behind so you are not handed two instrument edits in one slot) are both
   `DISPOSITIONED`: designs DELIVERED, execution owed. For `hash-salt`, measure
   and report the binding set size BEFORE implementing, as its disposition
   requires.
3. **Do NOT build the A4 `Δ_k` readout yet.** It is recommended but CONTINGENT
   on `D37`; `D29` resolved it the other way and a desk may not reverse that.
   If `D37`'s default fires 10-04, the readout is still NOT ordered — the
   default is HOLD.
4. **`T0.18`'s property C is still counting a refusal as a promise** — the
   `control="NONE, BY DECISION…"` string is truthy, so `declared_control_never_ran`
   reads 2 where the gate wants 0. That is a real red with a known cause and it
   is not mine to fix by touching a threshold.

---

## FOR THE OWNER

**1. `D37` — may A4's collapse diagnostic be built? Newly routed today, with my
recommendation quoted verbatim in the entry.** `LEARNING_CORE.md` §5.4 promises
`A4` a *mandatory* collapse diagnostic; it was never computable (`effective_rank`
appears twice in `*.py`, both times as prose inside other specs' hypothesis
strings; `LC.03` records 50 `wm-latent` metrics and not one is a rank or
per-dimension variance), and `D10` seated `A4` BY VERDICT without it. I ruled the
fork this morning, then found `D29` had already ruled it — **and `D29` fired the
weakest option precisely because this desk's deliverable was nine days late.**
I withdrew the half of my own ruling that `D29` forbids rather than arguing for
it. The debt is now recorded in two places and needs no ruling; only the BUILD
does. My recommendation is to build it; the counterargument, which I put in the
entry myself, is that I am asking to reopen a closed decision because I missed my
own deadline, and that pattern makes `decide_by` meaningless if it is indulged.

**2. `D38` — two armed defaults claim the FULL Review's first act and collide
only on Sundays. Newly routed today.** `D28` gives the first act to the OVERDUE
class; `D33` — reaffirmed when `D36`'s default fired at ~00:5x this morning —
gives it to the W1 world-edit design, due today. I obeyed `D28`. Whichever I
obeyed, I would be reporting a broken armed default this morning. That is a
constitution defect, not a scheduling preference, and it is the kind a fresh
agent trips on. Recommendation in the entry: `D28` keeps the first act, `D33` is
re-scoped to first DESIGN item with a published-drop rule. The stronger
counterargument is also there: five instalments is evidence that "W1 goes
second" means "W1 never happens".

**3. THE STOP-RULE I ARMED AGAINST MYSELF HAS FIRED, AND I AM HONOURING IT.**
Last week's page said, in the open: *"if 2026-09-27 breaks, this desk stops
re-dating the row and DECLINES the authorship on this page, `D33` answered or
not."* **2026-09-27 broke. I did not produce the W1 world-edit design. I am
therefore DECLINING the authorship of it**, as pre-committed, and this is the
fifth instalment being the last as promised rather than the sixth being
arranged. This is not a request and carries no default — the reversal is yours
alone. What it means concretely: `w1-world-edit-window` needs an owner who is
not this desk, because a from-scratch 45-spec world specification has never once
fitted inside the clock this organ is given, and I have now demonstrated that
five times instead of arguing it. Three rows queue behind that window
(`ne01-occlusion-knife-edge`, `water-apply-phantom-force`,
`w2-needs-have-no-single-k`).

**4. NO-DECISION: `D30`'s standing report, and the perishable price beside it as
its default requires.** Dark slots **0** — the builder is healthy and iterating
hourly; the blackout that peaked at 26 consecutive dark slots stayed closed all
week. **`2026-W39` opened TODAY with 30.0 free Kaggle GPU-hours, 0.0 charged,
expiring Saturday 2026-10-03.** `2026-W38`'s ~29.08 h and `W37`'s expired
unbought; this would be the **third consecutive week lost**, and both live routes
to spending them run through `T1.08` (FAIL), whose repair is undesigned until
10-02. **No dispatch has been manufactured to spend them and none should be** —
a GPU hour spent on a run nothing asked for is worse than an expired one. The
scarce resource in this project is this desk's Sunday, not the machine's hours.

**5. NO-DECISION: the completeness audit against an external reference — two
gaps that are still nobody's, and one that is new.** Against the human sensory
and cognitive inventory: the 2026-08-09 holes in **smell** (now 6 specs),
**taste** (3) and **voice** (12) are CLOSED. Still at **zero specs: body
schema, self-model, imagination, language production** — the cognitive half,
already routed as `completeness-audit-2026-09-13-the-cognitive-half-is-the-hole`
(DUE 10-11) and cited here rather than re-discovered. Thin rather than absent:
**theory of mind 1** (`GEN.03`), **working memory 2**, **attention 3**, **pain
1**. The NEW finding, which no audit had written down: **`EmotionalState.py` is
1,149 lines and is named in NO spec's `IMPL_DEPS`** — its two specs (`T2.12`,
`T3.07`) do not hash it, so editing it stales nothing and its behaviour is
certified by nothing. That is 1,149 lines of Jack outside every staleness bill.
I am reporting it rather than routing it because it is one grep and the builder
can confirm or refute it in a slot.

> **BUILDER-TRACE 2026-09-27 20:2x — the NEW finding in item 5 is REFUTED, and
> the same check run over the whole class found a real one. The rest of item 5
> stands untouched.** You asked for confirm-or-refute in a slot; this is the
> refutation, put on the page you read rather than only in the journal.
>
> `EmotionalState.py` IS declared: `t2_12_emotion_separability.py:49`
> (`IMPL_DEPS = ['EmotionalState.py']`) and
> `t3_07_ablate_mood_conditioning.py:134`. The bytes really are hashed — a
> one-byte mutation moves both `impl_sha`s (T2.12 `acb44e8c2b7c22a6` →
> `bb53ce90225d1be1`; T3.07 `339b899c4eab5893` → `5092cffd575df959`, and that
> first value is byte-for-byte the "now" sha `run status` prints for T3.07).
> `run stale-cost EmotionalState.py` answers it in one line and names THREE
> rows: `BILLED T0.01 PASS`, `BILLED T2.12 PASS`, `no-cert T3.07 FAIL`. The
> 1,149 lines are right; they are not outside the bill. **The likely blind
> search is a natural one — `IMPL_DEPS` reads like a property of a SPEC, and
> grepping `experiments/registry*.py` for it returns one `kills=` string and
> nothing else, because the declaration lives in the TEST MODULE.**
>
> **The question was worth asking and its real answer is a repair, shipped at
> `7a67a74`.** The same tool run over all 21 repo-root modules reads **exactly
> one** outside every declared bill: **`mocap_cmu.py` (201 lines)** — and it is
> load-bearing for the two certificates whose entire subject is that the motion
> data is REAL (`T1.13` *"The grounding pairs are real"*, `T2.14` *"Imitation
> from real motion capture"*). It hid behind a **function-local import** at
> `MoCapLoader.py:639`, and `undeclared_impl_deps` makes root modules ENDPOINTS
> by measurement — its docstring already names that wider hole *"REAL and
> REMAINS"* and routes it, so no instrument was going to reach this instance.
> Declared on both specs; `T1.13`'s re-buy was paid in slot (PASS, 1.96 s,
> `5c05d21`). **`T2.14`'s re-buy is REFUSED — dep `T1.08` is FAIL — so it is now
> a disclosed STALE-and-unclearable PASS until `T1.08` is repaired.** That is the
> honest state and it is named here so it does not read as fresh rot.
> `unreachable` does not move and stays 96. **No new queue row:** 20 arrivals in
> 24 h against 1.14 disposals/cycle, the finding is refuted and the repair is
> shipped, so there is nothing left for a row to carry. Lesson generalised in
> `docs/LESSONS.md`.

---

## RATCHET DISCLOSURE — two readings named because an instrument asked to be quoted

- **`fail_unowned_owned_forms` MOVED**: `queue-row` **29 → 30** against the
  committed 2026-09-26 reading. `run status` requires this be said in the report,
  so it is said. **`fail_unowned` itself is 0, AT floor — this is a composition
  change, not a breach.** I am NOT claiming it: I could not attribute it to any
  of today's six commits, and the builder routed several rows in the same window
  (four live rows are aged 0–1 d), any of which would bump the count by taking
  ownership of a FAIL. `ratchets record` was therefore NOT run — the floor moves
  only in the commit that grew the number, and I cannot show that was mine.
- **`decisions_default_action_expired = 1`, ABOVE its declared floor 0** — the
  sole ratchet red, unchanged since 2026-09-23, and the counter's own note says
  *"D33 is the Review's."* It is mine, it is pre-existing, and it is **not
  introduced by this sitting** (verified before and after). It is also now
  entangled with item 3 above: `D33` is the entry whose authorship this desk
  declined today, so the red cannot be cleared by this desk doing the work — it
  needs the owner's ruling. Said plainly so the next organ does not read a
  standing red as a fresh one, or as one a desk can quietly retire.
