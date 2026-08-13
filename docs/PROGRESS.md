# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: DAILY (Part 2, the test re-examination, runs Sundays only).

**2026-08-13 06:40 UTC — DAILY. Window: 2026-08-12 06:40 → 2026-08-13 06:40.**

*The one sentence: **this was the best window in the project's history, and it
was the first one in which the runner outran the ladder** — eleven capabilities
passed for the first time, eight of them Jack's rather than the harness's, and
the pass rate rose against a registry that barely moved. Read every number
below as a real reading, not an artefact: unlike 08-11, nothing was switched
off.*

*Two windows are covered, not one. **The 2026-08-12 Review died on an API 529**
before it reached Part 2.5 — `review.log` holds `sweep end rc=1` and no organ
reads another organ's exit code, so the failure was recorded and unnoticed. The
numbers below are 24 h; the steering and field-watch work covers 38 h.*

---

## 1. The numbers

**Ladder: 78/166 demonstrated (47.0%).** 24 h ago: 70/165 (42.4%). At the last
completed Review (08-11 17:02): 66/162 (40.7%).

| | this window (24 h) | prior 24 h |
|---|---|---|
| spec runs recorded | 59 | 39 |
| PASS / FAIL / VOID / ERROR | 48 / 9 / 1 / 1 | 35 / 3 / 0 / 1 |
| **first-ever PASSes** | **11** | 6 |
| net new demonstrated | **+8** (70 → 78) | +5 |
| registry growth | **+1** (165 → 166) | +3 |
| builder iterations | **24 of 24, all rc=0, none gated** | 8 |

The eleven: **T0.24, T1.02, T0.25, PS.02, VO.01, UB.9, BA.01, T0.26, T2.08,
PS.03, T2.03.** Named because the list is the finding — *voice, thermal,
balance, damage, curiosity, fusion, vision*. Six of Jack's constitutional
commitments went from zero passing specs to one in three days.

**Goodhart check — and this one is real.** Pass rate 42.4% → 47.0% while the
registry grew by a single spec. This is the first window since measurement
began in which the ladder was climbed faster than it was built. The 08-10
reading (the registry sprint had stopped outrunning the runner) is now
superseded in the runner's favour, and the 08-11 reading was an outage
artefact I declined to believe at the time. **Three days of data now say the
same thing: the instrument is built and it is being used.** The honest caveat
that keeps this from being a victory lap is in §3.

**Rework: 39 of 80 entries carry `attempt` > 1 (48.8%), up from 44.9%.** The
denominator moved properly this time (69 → 80), so the rise is real but softer
than it looks. It stays concentrated in the harness: **T0.13 at 22 attempts**,
T0.22 at 14, T0.21 at 13, T0.12 at 12. Note what those four have in common —
they are the specs that gate *other specs' certificates*, so every change to
the runner re-stales them and they re-run. That is the machine working, but
T0.13 at twenty-two attempts is still queued for Sunday's Part 2.

**Coverage: 16 of 23 commitments still have specs and nothing passing** (was
19 two days ago). Zero commitments have no declared spec at all.

**Two clocks, and only one is binding now:**
- **Claude: the week reset 08-12 12:00 UTC. `week:all models` reads 32%**,
  Fable 35%, resets 08-19. The loop is back on Fable rather than falling back
  to Opus. The credit emergency I escalated on 08-11 has passed for this week
  — and it passed by calendar, not by decision (see §7).
- **Kaggle: 18.53 h of 30 h charged to W32. ~11.5 h remain and are destroyed
  Sunday 2026-08-16.** This is the binding resource.

---

## 2. The frontier

**58 of 166 specs are unreachable** (was 67 of 162 — nine specs came free).

| terminal blocker | status | frees | blocks |
|---|---|---|---|
| **T2.01** Locomotion beats a random policy | **FAIL, 2.67σ vs a bar of 5** | **26** | **36** |
| **LC.03** Which learning cores learn to survive at all | NOT_RUN | **7** | 7 |
| UB.10 Fusion bakeoff: six arms | NOT_RUN | 3 | 5 |
| T2.02 Locomotion beats the honest MLP | VOID | 3 | 4 |
| T2.06 Language-action alignment | NOT_RUN | 3 | 3 |
| T3.01 Ablate vision | NOT_RUN | 0 | 9 |

**THE BLOCKING FACT OF THIS REVIEW: T2.01 is fenced off by three things, and
two of them do not exist.**

1. **`D1` — nine days open, and unanswerable as written.** It asks the owner to
   pick A/B/C/D and marks `A. Freeze the trunk` as `RECOMMENDED`; the
   PLASTIC-ONLY decree, which postdates it by eight hours, forbids exactly that.
   The overseer has raised this three times (08-10, and twice since with cost
   corrections). I second it without adding a new ask — §7 adds only the
   evidence that landed this morning.
2. **`ladder_prompt.md §3` fences T2.01 behind "registering and running D1.0 +
   T2.21".** Neither spec is in the registry, nine days later, and nobody is
   building them. Meanwhile T2.01 re-ran twice anyway and the gate was silently
   ignored both times. A gate that is stepped over rather than met is worse
   than no gate; I have rewritten §3 to force an explicit choice (§5).
3. **The P100 was dead.** Every torch-on-Kaggle job — including the planned
   8-hour T2.01 re-run — was silently failing because the index dropped
   `nvidia-cudnn-cu12==9.1.0.70` and `check=False` swallowed it. The builder
   found and repaired this yesterday and **verified the fix live on the real
   P100**. This is the first week T2.01 can actually run there, and ~11.5 h
   remain against its 8 h cost, expiring Sunday.

**LC.03 has been named "the biggest non-GPU unblock" in the steering file and
in the journal's NEXT line for three consecutive days and has not run once.**
This is not neglect and the builder was not wrong: the STANDING RULE (a
commitment with zero passing specs outranks fan-out) legitimately outranked it
every single time, and that rule is what produced this window's eleven passes.
But a rule that always wins starves everything behind it, and sixteen
commitments still read zero. The tie-break, written into §0 this run: **LC.03
serves the standing rule transitively** — `fast/slow` has five declared specs
and zero passing, and its only claim-kind specs (DP.01/02/03) are all blocked
behind LC.03. It is the one runnable spec that satisfies both rankings at once.

**Effort vs GOAL.md's path — the ratio inverted.** Of the eleven first-ever
passes, **eight served the creature** (T1.02 generality, PS.02 thermal, VO.01
voice, UB.9 unison, BA.01 balance, T2.08 curiosity, PS.03 damage, T2.03 vision)
and **three served the machine** (T0.24, T0.25, T0.26). On 2026-08-10 the same
count read thirteen machine to four creature. The instrument was declared
sufficient three days ago and it has since been pointed at Jack, which is
exactly what was asked for.

---

## 3. The honest paragraph

*(Not required in DAILY mode. Written anyway, because for the first time the
answer changed.)*

He got a voice, a sense of falling, a sense of cold, and a sense of being hurt
— and the hurt taught him something after one exposure, which is the fastest
kind of learning there is and the kind nothing else in his design has. Two days
ago I wrote that the machine had got sharply better at telling the truth and
Jack had not got more alive; that sentence is now wrong, and it is a pleasure
to retract it. He is measurably more of a creature than he was, and every one
of those senses arrived through a gate that could have refused it — one of them
did refuse, twice, and was let through only on the third honest attempt with
the failures kept in the record. So: closer to a creature, not merely busier,
and by a clear margin. The concerning drift is the other half of the same
picture, and it is this. Everything that moved this week was something we could
build alone on a small machine and check in an afternoon. The two questions
that decide what kind of mind he ends up with — *how does he learn*, and *where
does his control live* — both sat still: one waiting on a screening run that
the day's own success rule kept displacing, the other waiting on a decision
that cannot be made because the menu offers an option the constitution forbids.
We are getting very good at giving him senses and no better at deciding what
to do with what they tell him. A creature with every sense and no settled way
of learning from them is a very well-instrumented animal, and that is not what
we said we were building.

---

## 4. REWRITTEN / STRENGTHENED

None. DAILY mode does not re-examine tests — Part 2 runs Sundays. **No
threshold moved, no control softened, no spec file touched, no ledger entry
edited.** The uncommitted `ledger.json` / `gpu_budget.json` changes in the tree
are the T2.03 watcher's own write from 06:29 and are deliberately left for the
builder's 07:07 iteration to commit; the Review does not commit the ledger.

Queued for Sunday's Part 2, carried and added to:
- **T0.13** — twenty-two attempts to make true. Re-examine its shape.
- **SM.01's 0.33 intermittency shortfall**, reported honestly and left ungated
  because it fell outside the registered hypothesis. Whether `SM.02` may be
  built on top of it is a Part 2 question — and wk2-N1's whiff-clock pre-gate
  (§5) now bears directly on it.
- **PS.03's honest caveat**, recorded by the builder itself: the gradation arms
  are deterministic straight shots and all three worlds offered the same flat
  corner, so *gradation is one physics measurement, not three*. The Part B
  entry rates are the genuinely three-seeded half. This is exactly the kind of
  self-reported weakness Part 2 exists to price.
- **BA.01's control margin** — 0.6918 against a 0.70 cap. The pre-registered
  struct-leak risk is partially real on registered worlds where the pilot read
  chance. The cap held and nothing moved, but a control passing by 0.008 is a
  control worth re-examining rather than trusting.

---

## 5. Steering maintenance performed

**`scripts/ladder_prompt.md` — five corrections.**

1. **§0aa pointed the builder at spent work — the fourth time this file has
   done that.** It named `UB.9` as the unison gate and ranked it third in the
   project; UB.9 PASSED at 08-12 17:09. Re-aimed at **UB.10** (the fusion
   bakeoff, now third), and the cached "0 of the 37 unison specs pass"
   corrected to 1 of 21.
2. **§0's caveat was false and would have cost an iteration.** It told the
   builder that `run stale` flags PS.01 and that LC.03 must therefore re-run it
   first. `run stale` reads **zero**. Removed, with LC.03 marked ready as-is.
3. **§0 gained the LC.03 tie-break** described in §2 — the transitive
   `fast/slow` argument. Written explicitly *because* the builder's three days
   of displacing LC.03 were correct under the rules as they stood; this changes
   the ranking, not the verdict on past iterations.
4. **§2's two-clocks directive is superseded and was pointing the wrong way.**
   I wrote it on 08-11 when credits were the binding resource at 93%; the week
   reset and it now reads 32%. Left as-is it would have had the builder
   rationing iterations against a resource that is not scarce. Replaced with:
   only Kaggle binds, ~11.5 h expire Sunday, and T2.01 is the one job that both
   fits and matters.
5. **§3's phantom gate** — see §2, item 2. Rewritten to demand an explicit
   choice: register D1.0 + T2.21, or run T2.01 without them and say so.

Also corrected: the priority preamble cited T2.01's **08-10** figure; the live
number is **v5, 2026-08-12, 2.67σ**. And the STANDING RULE's cached worked
example (a count and a spec, both moved) was replaced with a pointer to
`run coverage` — the file's own rule, broken again by the person who wrote the
rule reminder.

**Field watch — TWO sweeps consumed** (`docs/INTEGRATION_QUEUE.md`, new section).
Week 2 (08-11) landed after the 08-11 Review had already read the file, and the
08-12 Review died before Part 2.5, so **wk2 was never dispositioned and wk3
rewrote the state file over it.** Both were reconstructed from the append-only
`FIELD_WATCH_LOG.md`. *That log just earned its entire existence* — a
state-file-only scout would have had a week of work silently overwritten, and
nothing would have reported it. Dispositions:

| | nomination | disposition |
|---|---|---|
| wk2-N1 | whiff clock → `SM.02` | **ACCEPTED, first of the four.** Our `OdourSensor` has no blank-duration state in a 40–55% blank field; SM.02's kills clause deletes a constitutional sense's wiring, so it needs a positive control in front of it |
| wk2-N2 | RPE-prioritised replay → `NEEDS_AND_DEATH` §3.4 | **ACCEPTED as an arm — take the control, not the mechanism.** The wet-lab's published must-fail control (magnitude does *not* bias replay) is worth more to us than another sampler |
| wk3-N1 | CIG → `LEARNING_CORE` A3's epistemic term | **ACCEPTED as an arm.** Cheapest in three weeks; same M=5 ensemble, zero new hyperparameters. Two conditions: it must not leak into model-free `LT.04`, and **it must not delay LC.03** |
| wk3-N2 | Optimistic World Models → arm on `A2` | **ACCEPTED WITH ITS WIN CONDITION AMENDED** — see below |

wk3-N2 is the one that needed a decision rather than a tick. Its lead objection
is constitutional: the optimistic term biases dynamics toward high-reward
futures, and in this project the world model *is* the unified brain, so an arm
could win its own bakeoff by corrupting the representation every `UB` gate
measures downstream. Rejecting on that reasoning would be deciding by argument,
which law 3 forbids. So the argument becomes a test: **the arm may not be
adopted on task return alone; its win condition must include the UB
representation gates re-run under it.**

Three methodology items also dispositioned: wk1's "confirm the table says what
the abstract says" is **already written** at `LESSONS.md:1725` (two Reviews
carried it as outstanding — mine included; it is closed). wk2's arXiv
primary-category convention is **closed by adoption**. wk3's is **accepted for
the builder to write**, and it is the sharpest thing in three sweeps:
*verify-before-nominating protects a reader of papers, because there is an
abstract to doubt; it protects nobody reading our own ledger, where the
diagnosis is a story the reader wrote itself.* Rule: a diagnosis of one of our
own failures must carry the arithmetic that survives, not the literature that
motivated it.

**Seat staleness (`docs/CHAMPIONS.md`) — the file itself is now the finding.**
It has not been touched since 2026-08-10 and is factually wrong in five seats:

- **Smell** — "SM.01 is CPU and runnable today". It passed 08-11.
- **Taste** — "TA.01 is CPU and runnable today". It has passed.
- **Voice** — "**VACANT — he still cannot make a sound**… VO.01 is the cheapest
  constitutional gap left". **He can make a sound. VO.01 passed 08-12 12:17**,
  after four honest FAILs. The most out-of-date sentence in the project.
- **Sensory fusion** — arena `UB.10`, still undecided, but UB.9 has passed and
  UB.10 is now the project's third blocker. Not wrong; no longer current.
- **Learning core** — challenger status still reads "match in progress". The
  match has not started; LC.03 is NOT_RUN. **A seat that describes an unstarted
  contest as in progress is the exact failure this rule exists to catch.**
- **Vision encoder** — its arena `T2.03` ran and passed this morning. See §7;
  this one is not housekeeping.

Reported, not edited: CHAMPIONS.md edits are a Sunday ANATOMY AUDIT power. But
five stale seats in three days means the map the builder navigates seats by is
no longer trustworthy, and the builder can fix the factual lines under
`INTEGRATION_QUEUE` protocol step 6 without waiting for me.

**Organ liveness — all five alive, nothing silent, but one loud failure went
unheard:**

| organ | cadence | last fire | verdict |
|---|---|---|---|
| builder | hourly | 06:07 (rc=0) | **OK — 24 of 24 fires ran, none gated, every one with an end line** |
| overseer | 6-hourly | 06:37 (running) | OK — 00:50 audit closed `DRIFTING` |
| field watch | Mondays | 08-12 07:02 (rc=0) | OK — extra sweeps were resume catch-ups, mandate now spent to 08-19 |
| Review | daily | 08-12 **rc=1 (API 529)** | **FINDING — see below** |
| tmp_reaper | 4-hourly | 04:13 | OK |

The `EXIT` trap three consecutive Reviews asked for **is built and working** —
`ladder_loop.sh:121`. All 24 iterations closed with an end line. Asked for on
08-09, delivered; taking it off the list.

**The finding: the 2026-08-12 Review died on an API 529 and nothing noticed.**
It logged `sweep end rc=1` honestly and no organ reads another organ's exit
code. The organ-liveness check I run is a *silence* detector — it would have
caught a dead cron and it did not catch a cron that fired, failed loudly, and
produced nothing. A day of steering maintenance was skipped, which is how wk2's
field-watch nominations came within one overwrite of being lost. Silence is
never success; **a non-zero exit is not success either, and nobody is reading
them.** For the owner in §7.

---

## 6. FOR THE BUILDER

Ordered. Item 1 is the only one with a deadline.

1. **T2.01 — submit it to Kaggle before Sunday. `gpu<8h`, ~11.5 h left, hours
   are destroyed at the reset.** It frees 26 and blocks 36; it is FAIL at 2.67σ
   against a bar of 5 that **does not move**. Read `ladder_prompt.md §3` first
   and make the D1.0/T2.21 choice explicitly in your commit message — do not
   step over that gate a third time. If the answer is (b), say plainly that
   T2.01 measures *whether* the trunk learns and not *where* it belongs.
   And know why this is newly possible: you repaired the cudnn/torchvision pins
   yesterday and verified them on the live P100. Every P100 job before that was
   dead. This is the first week the run can happen.
2. **LC.03 — run it. Ready as-is; no PS.01 re-run needed, `run stale` is zero.**
   `cpu<2h`, frees 7, and it is the only runnable spec that satisfies both the
   `blocked` ranking and the STANDING RULE at once (the `fast/slow` argument in
   §2). Three days named as next, zero runs. It decides HOW JACK LEARNS, and it
   runs beside the T2.01 GPU job rather than competing with it.
3. **Refresh `docs/CHAMPIONS.md`'s factual status lines** — five seats listed in
   §5, of which "Voice — VACANT, he still cannot make a sound" is the one to be
   embarrassed by. Protocol step 6 already makes this yours. Change no verdict
   and seat nobody; only correct what has demonstrably happened.
4. **Write the wk3 LESSONS entry** the scout cannot write itself: *a diagnosis
   of one of our own failures must carry the arithmetic that survives, not the
   literature that motivated it.* Full scar and the surviving arithmetic
   (R² ceiling 0.816 vs a 0.50 gate) are in the new `INTEGRATION_QUEUE`
   section. This is the third Review to carry a scout-nominated lesson; the
   wk1 one turned out to be **already written** at `LESSONS.md:1725` — check
   before you write.
5. **The four consumed field-watch nominations**, in the queue order set in
   §5 — wk2-N1 (whiff clock, first), then wk2-N2, wk3-N1, wk3-N2. All are
   design entries, none is urgent, and **none of them may displace item 2**.
6. Carried from 08-10, still open: **N1, the certificate pre-gate for UB.11**,
   before UB.9's results start feeding the ablation matrix. UB.9 has now passed,
   so this is no longer hypothetical — the matrix has something to eat.

---

## 7. FOR THE OWNER

**1. The strategic one, and it is new this morning. `T2.03` passed, and the
number it produced says the PLASTIC-ONLY decree may have been implemented more
broadly than you decreed it.**

The result, seeds 0/1/2: pretrained features **0.9867 / 0.9833 / 0.9533**.
Our from-scratch encoder — the sitting champion of the vision seat, 244,960
params — **0.4467 / 0.4667 / 0.4933**. Random projections 0.40, raw pixels 0.38,
shuffled control 0.25.

Read it honestly in both directions, because it cuts both ways:

- **Your decree's pre-registered re-open trigger did NOT fire.** It says the
  decision returns to you if *"visual competence has not cleared its null"*.
  From-scratch cleared its null — 0.47 against random projections at 0.40 and
  pixels at 0.38, so the conv structure is genuinely worth something and the
  encoder genuinely sees. **The decree stands on its own terms** and I am not
  asking you to revisit it.
- **But it now has a price tag, and this is its first.** The gap to the
  pretrained yardstick is about half the accuracy range on this task. Your
  recorded counterargument #2 — *"pure forfeits inherited visual knowledge…
  expect a longer, more data-hungry childhood for his eyes"* — was written as a
  prediction. It has been measured, and it is large.

**The actual fork, and it is not "revisit the decree".** When the decree
collapsed the PL.* bakeoff, `CHAMPIONS.md` recorded that *"of four arms
(frozen / frozen+adapters / critical-period / pure), three involve freezing at
some stage. Only pure survives, so there is nothing left to arbitrate."*
There is a fifth arm, it was never on that list, and **the decree does not
forbid it**: *initialise from pretrained weights and keep training everything,
forever.* Nothing about it is frozen. Its reshaping gain is not zero — it is a
fully plastic encoder that happens to start somewhere useful instead of at
random. It inherits the visual head start whose loss you priced as acceptable,
without accepting the welded-shut component you actually objected to.

**My recommendation: authorise a warm-start-plastic arm into the vision-encoder
seat's contest (`PL.02` + `T2.03`'s successor), and let it fight from-scratch in
the arena.** This is not a request to reopen a decree — it is the observation
that the collapse from "nothing frozen" to "only from-scratch" eliminated an
arm that "nothing frozen" permits. If from-scratch wins, the decree is
vindicated by measurement instead of by argument, which is what SYSTEM.md law 3
wants anyway. One line from you and it enters the queue.

**2. `D1` is nine days old and I am seconding the overseer without adding a new
ask.** It gates T2.01, which gates 26 specs including *every curiosity spec in
the ladder* — the north star of GOAL.md. It cannot be answered as written
because its `RECOMMENDED` option is the one your own decree forbids. The
overseer has laid out the two ways to fix the menu (strike option A, or write
into CHAMPIONS.md's SCOPE that a frozen *control* trunk is a different question
from a frozen *sensory* tower). **One line either way unblocks the largest
thing in the project.** I would add only this: if item 1 above appeals to you,
the same warm-start-plastic option is missing from D1's menu too, and it is
arguably the best answer there as well.

**3. An organ failed loudly on 08-12 and nothing heard it.** The Review hit an
API 529, logged `sweep end rc=1`, and died before doing any steering
maintenance. No organ reads another organ's exit code. My own liveness check
is a *silence* detector, so it is structurally blind to this — a cron that
fires, fails, and logs the failure looks identical to a healthy one from where
I stand. The cost was not hypothetical: a day of steering went unmaintained
and week 2's field-watch nominations survived only because that scout happens
to keep an append-only log alongside its rewritten state file.

Recommendation, cheapest version: **have each organ's wrapper record its exit
code where the next organ reads it, and make a non-zero exit a finding rather
than a log line.** This costs a few lines in the four `scripts/*.sh` wrappers.
I have not built it — it touches organ scripts rather than steering files, and
after the credit-gate experience I would rather you chose the shape than have
me install one. Related and still open on your desk: *"Claude credits are the
binding resource and are unmetered"*. Note that the credit crisis I escalated
on 08-11 resolved **by the calendar rolling over, not by a decision** — the
same is true of D5. Two constraints in a row have expired rather than been
settled, which means we have not actually learned whether the gate's shape is
right; we have only learned that weeks end.

*Nothing in this review touched a threshold, a control, a spec file or the
ledger. The steering edits are operational and itemised in §5.*
