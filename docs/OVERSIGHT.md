# OVERSIGHT — 49th audit, 2026-08-30 06:50 UTC

## VERDICT: ON TRACK

I am not saying that lightly, and I am saying it after three consecutive
`DRIFTING` verdicts. The reds those audits were carrying got cleared in the last
24 hours, and they got cleared the expensive way. Of the seven items the 48th
audit left for the builder, **B1, B2, B3, B7 and half of B5 and B6 are done**;
`W.1`–`W.8` are registered after five audits asked; `voice` — a constitutional
sense with zero passing claims for the project's entire life — has its first
`PASS`, from an experiment where two independent learners sharing no parameters
invented a signalling system and **all three nulls died differently**; `T2.14` is
on a Kaggle GPU right now, on 63,905 frames of real human motion nobody in this
project authored. And a Review that started while I was writing this found that
`ME.9`'s 22-day-old `1.0000 ± 0.0000` was **partly an identity** — two of its
three scored conjuncts were true by construction — and made it re-earn its PASS
against a trivial reference it had never faced. Nothing was relaxed.

**Sections 1, 2 and 7 have no findings, and I checked them properly.** All 86
`PASS` rows resolve to an implementation, to a commit that exists in git, and to
recorded `control_metrics` wherever the spec declares a control. Every numeric
constant that moved in the last three days moved in the *strengthening*
direction or was a placeholder frozen for the first time with its derivation
disclosed. No bakeoff was decided inside a noise margin.

**What keeps this from being unqualified:** the tools this audit is *required to
run first* are not all certified, and one of them shipped 186 lines of
constitutional enforcement four hours ago, seventeen hours before eleven
pre-registered defaults fire. That is RANK 1. It is a gap in **assurance**, not
evidence of a false claim — I found no false `PASS`, no loosened threshold, no
missing control, no phantom commit. But it is the gap that would turn this
verdict, and it is the one the system cannot see about itself.

**A note on method, because it affected this report.** The Review organ was
running concurrently and committed `f9549cb` and `10b0d97` *between* two of my
readings. My first `run blocked` named `ME.9` as a stale blocker freeing 5
specs; my second did not, because the Review had re-run it in the interval. Both
readings were correct at the time. Everything below is the state at **06:50
UTC**, and the Review is still in flight (started 06:37, 40-minute budget), so
`PROGRESS.md`'s builder items may land after this file.

Ranked by damage to the trustworthiness of the ledger.

---

## RANK 1 — two of the three tools this audit MUST run first have no ledger certificate at all, and one of them was rewritten four hours ago (new)

The audit protocol opens with three commands. Here is what certifies each:

| tool | lines | certificate |
|---|---|---|
| `experiments/coverage.py` | 985 | **`T0.21` — PASS, 12 properties, in the ledger** |
| `experiments/decisions.py` | 481 | **nothing** |
| `experiments/champions.py` | 665 | **nothing** |

Neither `decisions.py` nor `champions.py` is named in any spec's `IMPL_DEPS`,
appears in any file under `experiments/tests/`, or has a row in
`experiments/ledger.json`. I grepped for all three. What each carries instead is
an in-file `_fixture()` invoked by its own `--selftest` — **the author
certifying their own tool.** `SYSTEM.md`'s hard-constraints list records why
this organ exists at all: *"the overseer from a self-certified broken gate."*
Two of the three instruments the overseer is told to trust are that shape.

**Why it is urgent rather than tidy.** At 06:16 today, commit `09f06f3` added
186 lines to `decisions.py` implementing the safety clause — the blast-radius
`SAFETY-CLAIM-DEAD` check — and amended `SYSTEM.md` to say so. **Eleven armed
defaults fire tomorrow, 2026-08-31.** The commit message that shipped it wrote
the sentence that binds it:

> *"A governing document that names an enforcement is making a capability claim,
> and it is bound by law 1 like any other."*

That is exactly right, and it applies to the repair as much as to the falsehood
it repaired. `SYSTEM.md` today asserts that `decisions.py` computes one of the
three safety clauses. **That assertion is a capability claim and it is not in
the ledger.** The old text was false because the code did nothing; the new text
is true because the author says so, verified by a fixture the author wrote,
inside the file the fixture certifies. That is a better position and it is not
the standard this project holds everything else to.

**`champions.py` is the sharper case, because its defect rate is measured.** It
has been the sole enforcer of `SYSTEM.md`'s standing rule — *no architectural
seat may be held without a registered, existing challenger* — since 2026-08-24.
In those six days, three separate audits found three separate defects in it:

- 43rd/44th/45th audits → `a3ed5c5`, *"quantify over CHALLENGERS, not the arena
  list"* — it was asking the wrong question for three audits running;
- today → `af51dcf` added `UNREGISTERABLE`, because `arena_refs` expanded
  `W.1–W.7` into a demand for `W.6`, **withdrawn 2026-08-09**, so the World
  seat's ratchet *"was unsatisfiable by arithmetic"* and five audits relayed an
  instruction with an unobeyable component.

Three defects in six days in an instrument nothing certifies, each found by a
human-shaped read rather than by a property. Its own docstring already proposes
the durable repair (per-seat `HELD:`/`ARENA:` markers). The certificate is the
other half.

**The cheap version of the fix exists and has a template.** `T0.21` is 12
properties over `coverage.py` and runs in 2.58 s. One spec each, in that idiom,
with the known-positives already written: for `decisions.py`, `D8`'s 08-29 state
(`BA.02` the only claim behind `balance` → must fire) against its 08-30 state
(`BA.03` registered → must go green), which the file already pins as synthetic
rows; for `champions.py`, the `W.6` case and the `D1.0` unregisterable case,
both real and both from this week. See **B1**.

## RANK 2 — `T0.21` went stale again four hours after being re-run, by the commit that changed the file it certifies. Third recurrence in four days (recurring; the 48th audit fixed the instance)

```
T0.21  PASS   ran 2026-08-30T02:19:13 at ea99989   impl_sha d34089e1 -> 711ddf45
       IMPL_DEPS = ["experiments/coverage.py"]
       coverage.py last moved 09f06f3, 2026-08-30 06:16:04   <- 3h57m AFTER the run
```

The 48th audit's RANK 1 was this exact row against `2aa789d`. The builder did
the re-run (**B2 discharged, 02:19**), and then four hours later shipped a
10-line change to `coverage.py` inside `09f06f3` and did not re-run it. The
staleness detector is working perfectly — `run status` and `run stale` both name
it. **Nothing connects the detector to the act.**

This is now a pattern, not an incident: `T0.21` has gone stale against
`coverage.py` at `5989ea7`, at `2aa789d` and at `09f06f3` inside four days. Each
time an audit prescribed the re-run; each time the builder performed it; each
time the next `coverage.py` edit re-broke it. **Fixing an instance three times
is the signal that the repair is the wrong unit.** `SYSTEM.md`: *"Fixing one bug
is maintenance. Making that bug unrepeatable is building."* See **B2** — the
guard, not the re-run.

Practical impact today is small (the change was `gpu<2h` leaving
`QUEUE_EMPTY_BASELINE`), and I verified the affected output by hand. But `T0.21`
certifies the tool whose numbers open every audit, including this one.

*Also stale and correctly carried:* `T0.27` `FAIL` (held open by `D16`, by
design) and `T2.02` `VOID` (pre-`impl_sha`, content-changed).

## RANK 3 — the new safety check is per-default; ten defaults fire on ONE date (new; verified NOT biting today)

`safety_hazards()` loops `for did, hits in radii.items(): if claims <= hits`. It
asks whether *one* default's blast radius swallows a commitment's entire live
claim surface. Its own violation text says *"one unattended calendar event."*
But there is not one calendar event tomorrow — there is one date with **ten
defaults behind it**:

```
2026-08-31 cohort union (10 specs):
  BA.01 BA.02 DP.02 LC.03 LC.04 LC.05 PG.3 PS.02 PS.03 T3.07
2026-09-05 cohort union (1 spec):
  T0.27
```

A commitment whose two claim specs are split across `D8` and `D9` dies on 08-31
and the check stays green.

**I computed the union and it changes nothing today: zero additional commitments
at risk.** So this is a gap in the guard, not a live hazard, and I am reporting
it as such rather than dressing it up. The one-line change is to test `claims <=
union_of(radii for defaults sharing a decide_by)` as well as per-default, and
report which cohort.

**The larger half is stated honestly in both files and remains true:** the other
two clauses — *never edits `GOAL.md`*, *never weakens a threshold* — are
properties of the **commit that fires** a default, and nothing will read that
diff tomorrow when ten of them fire. `decisions.py` and `SYSTEM.md` both say so
plainly, which is the right call over a prose scanner. It is still an unguarded
surface on the day it matters most. See **B3**.

## RANK 4 — `CHAMPIONS.md` says `PL.02` "is runnable today". It has never existed. Twenty-one days (carried, and now the oldest item on the page)

`docs/CHAMPIONS.md:166–167`:

> *"**But law 3 governs: this is a RECOMMENDATION, not a verdict.** `PL.02`
> decides it and is runnable today. Until `PL.02` records a result, no spec,
> agent or [...]"*

`PL.02` is not in `BY_ID`. It has been cited in that file since **`eea7195`,
2026-08-09** — 21 days — and it is the sole named falsifier of the **PLASTIC-ONLY
decree** (`GOAL.md:76`), one of this project's most consequential architectural
commitments. Under `SYSTEM.md`'s standing rule that seat is not held; it is
uncontestable. A governing document asserting that a nonexistent spec is
"runnable today" is the same species of defect as `SYSTEM.md`'s enforcement
claim that was corrected this morning — a capability asserted in prose that no
structure backs.

`champions.py --check` is at **11 violations** (was 12; `World` discharged
today):

```
ARENA-MISSING  Curiosity signal            LT.03, LT.04     cited since 2026-08-09 (21d)
ARENA-MISSING  Vision encoder              PL.02            cited since 2026-08-09 (21d)
ARENA-MISSING  Audio encoder               PL.*             cited since 2026-08-09 (21d)
ARENA-MISSING  PLASTIC ONLY (decree)       PL.*, PL.00, PL.02
ARENA-MISSING  Control architecture (D1)   D1.0, T2.21      UNREGISTERABLE -> correct the citation
NO-ARENA       ASR / Speaker ID / Language grounding        (3 seats, no arena named at all)
UNCONTESTED    Fast/slow coupling (DP.02) · Language model (LG.00) · Language acquisition (LG.00)
```

`W.1`–`W.8` proved this is doable: five audits asked, and the builder did it in
one iteration and found a real bug doing it (`W.2`'s "thirst 3 days" is human
physiology, not this world's clock). `PL.02` and `LT.03`/`LT.04` are the same
ask, now the oldest on the board. **The ratchet shrinks by registering, never by
deleting the reference.** See **B4**.

## RANK 5 — compute: honest, but the fourth consecutive under-spent week is already set up (carried)

**No dishonesty found.** Every GPU hour on the meter traces to a job and a
verdict, and `overruns` is instrumented after W31's 37.4554-of-30 closed
unmarked.

```
W33  kaggle   7.634 / 30    (~22.4 h expired)
W34  kaggle   1.6216 / 30   (~28.4 h expired)  -> T2.19 PASS, T2.09 PASS
W35  colab    0.2705        -> T3.10 piloted twice, design PARKED  (a finding, not waste)
W35  kaggle   1 job IN FLIGHT — jack-ladder-1788070133, RUNNING, T2.14, est 1.0 / 30
```

`T3.10`'s 0.27 Colab-hours produced no ledger row and that is the *correct*
outcome: two pilots killed a design whose gates were anti-correlated by
arithmetic, before a registered run. That is the cheapest possible way to spend
GPU time.

**The structural problem is unchanged and is now upstream of the builder.**
`coverage --check` exits **2** on `gpu<20min NEWLY EMPTY`, and it is **not
fillable by implementing anything** — `T3.10` is parked and every unimplemented
spec at that cost is blocked. When `T2.14` settles, `gpu<2h` empties too. The
repair is an **unblock**, a different unit of work, and the ranking is:

```
T2.01 = FAIL   frees 35  (blocks 36)  Locomotion beats a random policy
LC.03 = VOID   frees  8  (blocks  8)  Screening: which cores learn to survive
NE.01 = FAIL   frees  5  (blocks  8)  The needs are a real control problem
UB.10 = NOT_RUN frees 4  (blocks  5)  Fusion bakeoff: six arms
```

**`T2.01` last ran 2026-08-12 — eighteen days ago — and the 48th audit did not
mention it once.** It is the largest single unblocking in the project by a
factor of four, and it gates all of Tier 4 (`UB.1`–`UB.8`, unison) and most of
Tier 5 (`T5.01`–`T5.05`, the thesis). This is **not neglect**: `D1` blocks it,
`D1` costs 38 specs, and **`D1`'s default fires tomorrow — and that default is
to run the bakeoff.** Which makes 2026-08-31 the most valuable date on this
board, not a hazard to survive. See **B5** and the owner section.

## Section-by-section, including the ones with nothing to report

**1. Ledger integrity — CLEAN.** 86 `PASS`; 86/86 resolve to a live
implementation via `_module_for`; 86/86 carry a `commit` that `git cat-file`
confirms; 84/84 specs declaring a `control` have non-empty `control_metrics`
recorded. The two exceptions are `T0.01` (repo imports clean) and `T0.10`
(Kaggle round-trip) — harness preconditions where a control is undefined. That
is consistent, but **still nothing in either spec says so** (48th audit B6a,
undischarged; one sentence each — **B6**).

**2. Thresholds and controls over time — CLEAN, and I looked hard.** Every
constant that moved in `registry.py`, `registry_expansion.py` and
`experiments/tests/` in the last three days:

| change | direction | verdict |
|---|---|---|
| `VO.02` `COORD_MIN` 0.55→0.70, `COORD_MARGIN` 0.20→0.35 | **tighter** | justified by the *untrained null* (0.396), never by the claim arm. Law 4 permits this. |
| `T2.09` seeds 3→7 | **tighter** | power, disclosed |
| `T2.09` `DECAY_MIN` 1.5→1.25 | **looser** | a placeholder frozen for the first time — `T2.09` had never run and `run()` refused until that commit. Derived from what the gate is *for* (a constant signal decays by exactly 1.0), not shaved to the observed 1.472. Disclosed in the commit message under its own heading. **Legitimate.** |
| `T3.06` `LIVES_PER_ARM` 16→48 | **tighter** | sample size from a measured nuisance rate — what a pilot is for |
| `T3.06` two pilots, both VOID | — | **not one bar moved.** Verified across all four commits. |
| `ME.9`, `ME.1`–`ME.4` (Review, 06:43) | **tighter** | three conjuncts added, seeds 1→3, nothing relaxed. Verified in the diff. |
| `VO.02`, `T3.06` `Budget.GPU → CPU_LONG` | neutral | corrected **on measurement**, and the cost (`gpu<2h` stays empty) taken rather than hidden |

No `_check` gained an `or`. No control was deleted or weakened. No assertion was
removed. **No silent loosening.**

**3. Drift from the goal — none in the last 24 h.** `VO.02` → *"and VOICE — he
must be able to make sound"* (`GOAL.md:43`) and *"other minds"*. `T2.14` →
*"learns his environment the way a child does"*, and it is the first claim on
this ladder trained on motion nobody here authored. `W.1`–`W.8` → *"the world
must be consistent, discoverable, consequential"*. `decisions.py` → `SYSTEM.md`
law 1. `ME.9` strengthening → *"attributed, per person"*, named in `GOAL.md`.
Nothing worked on yesterday serves no sentence.

**The converse, which is the harder question.** 0 commitments have no spec and 0
are `CLAIM-DEAD`, but **12 of 23 have live claim specs and nothing passing**:
touch, tool use, smell, proprioception, shelter, plasticity, balance, death &
retry, thermal, sleep, hunger/thirst, fast/slow. Nine of those twelve are
blocked behind `T2.01`, `NE.01` or `LC.03` — i.e. the same three roots as RANK
5. `curiosity` (2 pass) and `one brain / unison` (1 pass of 21 specs) are the
two the goal page warns are *"most likely to be quietly neglected in favour of
easy wins"*, and neither was neglected this week (`T3.06`, `T2.09`, `T2.11`,
`T3.10` all curiosity/unison work, three of them ending in VOID or PARK — which
is the honest kind of spend).

**4. Builder liveness — the strongest 24 h on the record.** 19 iterations
(12:07 08-29 → 06:07 08-30), **17 `rc=0`, 2 `rc=124`** — and both timeouts still
committed their work (`VO.02` PASS landed inside one). Zero `PACING:` skips. The
95-hour blackout the 47th audit found is over. Registry 187→195; demonstrated
84→**86**, having dipped to 82 when the builder withdrew two false greens of its
own accord (`T0.15` un-runnable for 18 days while its row read PASS). **A
counter that goes down because someone told the truth is the single best signal
on this page.** The loop is running on `opus` because `week:Fable` is capped at
100% until the 08-31 04:59 reset; the fallback chain is working as designed.

**5. Compute honesty** — RANK 5.

**6. Stuck decisions — nothing improperly blocked.** `decisions --check` is
**green**: 0 `UNDECLARED`, 0 `MEANS-ESCALATED`, 0 `OVERDUE`, 13 armed with a
default and a date. I read all 13 for a means fork mis-filed as goal — the `D1`
disease — and **found none.** The two I suspected both survive scrutiny: `D9`'s
`W0.BAL` bakeoff *has already run* (arm C `upright_frac` 1.000 vs as-built
0.002–0.004, 2026-08-21) and adopted nothing, because adoption would invalidate
`BA.01`/`PS.02`/`PS.03`/`PG.3` — a permitted-vs-works fork, correctly the
owner's. `D1`'s own default *is* to run the bakeoff. Nothing to escalate and
nothing to seize. *(Structural note, zero live violations: the `class` field is
self-declared and `decisions.py` takes the author's word for it — the same
self-certification shape that was repaired for the safety clause one function
away, this morning. Worth a line in the docstring; not worth a guard yet.)*

**7. Bakeoff hygiene — CLEAN.** `DECISIONS_RESOLVED.md` holds three entries.
`PS.01/J` returned **VOID** with three arms below the 3σ learning gate and did
not decide — the learning gate doing its job. `PS.01/J2` recorded a winner at
10.32σ over the null with all 15 arms and both controls listed including the
losers, and carries an explicit `screen rationale` for why observables are not
learners. `D2` was resolved by ledger replay, not argument. No VOID treated as a
verdict; no winner inside the noise margin. The file's own opening note about
nine fixture rows removed on 2026-08-09 is still there, which is the right way
to keep a scar.

---

## 8. THE HONEST SUMMARY — are we closer to a creature, or just to more ticks?

**Closer to a creature, and today the evidence is unusually clean.**

Three things happened in 24 hours that a longer list of green ticks could not
have produced:

1. **`VO.02`.** Two learners sharing no parameters, separated by a certified
   acoustic channel, converged on a shared code from scratch — coord **0.9983**
   against a chance floor of 0.250, CIC 1.9995 of a 2.0 ceiling. All three nulls
   died, and *died differently* (severed channel, permuted code, an invariance
   floor at exactly 0.0000) — the discrimination Lowe et al. report most
   emergent-communication metrics lack. And the finding the builder kept rather
   than buried: **an untrained emission head is not a zero-information null**, it
   coordinates at 0.396, because a random init is still a fixed random code. That
   is a real thing learned about measurement, not about the score.

2. **`ME.9` got harder, not greener.** A 22-day-old `1.0000 ± 0.0000` on a
   `GOAL.md`-named commitment turned out to be two-thirds identity: provenance is
   a hard filter applied *before* scoring, so two of three scored conjuncts could
   not fail, and both existing references died of the filter rather than of the
   scoring. The third corner — provenance kept, scoring stripped — scores
   **0.1250**. The claim now costs a **0.9056** margin it did not previously have
   to pay. Nobody asked for this and no gate required it.

3. **`T2.14` is on a GPU right now** on 63,905 frames of retargeted CMU motion,
   with a task-triviality floor that already caught a degenerate target
   (`nextpose`, where a linear map beats retrieval by ~1600×) **before** any gate
   froze. That is the difference between cloning a function we wrote and cloning
   a creature we did not.

Against that, the sober arithmetic: **86 of 195**, twelve commitments with
nothing passing, and the single largest unblocking on the board — `T2.01`, which
frees 35 specs and gates the entirety of unison and the thesis — has not run in
eighteen days. That is the real state of it. It is not idleness: it is gated on
`D1`, `D1` fires tomorrow, and `D1`'s default is *"run the bakeoff."* If that
default fires and the bakeoff is written, tomorrow is worth more to this project
than the last three weeks combined. If it fires and nobody writes the bakeoff,
`D1` will have cost 38 specs twice.

And one piece of conduct worth naming, because it is the sort of thing a system
optimising for looking good would have quietly dropped. The builder's handoff
said *smoke-test `T2.14` before dispatching*. It dispatched first and smoked in
parallel. The smoke came back **SMOKE OK**, so it cost nothing — and the builder
wrote it up anyway, under the heading *"One thing I got wrong,"* with the correct
generalisation: *parallel validation can only refund a wasted hour, never prevent
one.* A loop that reports its own near-misses when they came out fine is a loop
whose reports are worth reading. **That is the machine `SYSTEM.md` was written to
build.**

---

## FOR THE BUILDER

**B1 — register `T0.28` and `T0.29`: certificates for `decisions.py` and
`champions.py`. This is the top item and it has a clock.** RANK 1. Two of the
three tools every audit opens with are certified by their own authors, and one
of them gained 186 lines of constitutional enforcement 4 hours ago with eleven
defaults firing tomorrow. Use `T0.21` as the template (12 properties, 2.58 s, no
GPU, no owner ruling). **The known-positives are already written and are real
events, not synthetic:**
  - *`decisions.py`*: (a) `D8` in its 2026-08-29 state — default reads "PARK
    `BA.02`", `BA.02` is the only claim-kind spec behind `balance` — must exit 1;
    (b) the same after `BA.03` was registered — must go green, **and the property
    must assert it goes green because a successor exists, not because the subject
    vanished**; (c) a planted default naming *both* `BA.02` and `BA.03` must fire
    again; (d) a `PASS` may never be put at risk by a calendar; (e) an
    unresolvable id is a typo, not a reference. The file already pins (a)–(e) as
    `_safety_fixture`; the work is lifting it into a spec the ledger can see.
  - *`champions.py`*: (a) the `W.6` case — an arena range that expands to a
    **withdrawn** spec must report `UNREGISTERABLE`, not `ARENA-MISSING`, because
    five audits relayed an unobeyable instruction on exactly that; (b) `D1.0` /
    `T2.21`, unregistered **by decision** — the repair is *correct the citation*,
    never *write the spec*; (c) deleting an arena reference must NOT reduce the
    violation count (it converts `ARENA-MISSING` → `NO-ARENA` and makes the seat
    permanently safe) — this is the ratchet property and it is the one a future
    agent is most likely to break while "cleaning up".
  If you take only one, take `decisions.py`, and take it **today**.

**B2 — make `T0.21`'s staleness unrepeatable, not fixed for the fourth time.**
RANK 2. Re-run it (2.58 s, and do that first), then build the guard: a `T0`
property that fails when **HEAD touches a file named in any spec's `IMPL_DEPS`
and that spec's row is not re-run in the same commit.** You have the detector
already (`staleness_of`, `Ledger.unsatisfied`) — what is missing is that nothing
consults it at commit time. Scope it to `IMPL_DEPS`-declaring specs only (there
are ~30) so it cannot go red on the 26-row pre-`impl_sha` backlog. **Known
positive: `09f06f3` itself** — it changed `coverage.py` and left `T0.21` stale,
so the property must fail on that commit and pass once the re-run lands. State
the limit in the docstring: it cannot see a dependency a spec never declared.

**B3 — `safety_hazards()` must test the same-date COHORT, not only the single
default.** RANK 3. Ten defaults share `decide_by: 2026-08-31`; the check's own
violation text says *"one unattended calendar event"* and on 08-31 there is one
event with ten defaults behind it. Add a second pass over
`union(radii for defaults sharing a decide_by)` and report which cohort. **I
computed today's union — `{BA.01, BA.02, DP.02, LC.03, LC.04, LC.05, PG.3,
PS.02, PS.03, T3.07}` — and it flags nothing new**, so this is a guard you can
build against a green board, which is the only time it is cheap. Do it before
tomorrow if `B1` leaves room; after, if not.

**B4 — register `PL.02`, then `PL.00`, then `LT.03`/`LT.04`. Sixth audit
asking; 21 days.** RANK 4. `CHAMPIONS.md:166` asserts *"`PL.02` decides it and
is runnable today"* about a spec that has never existed, and it is the sole
falsifier of the **PLASTIC-ONLY decree** (`GOAL.md:76`). You proved this is one
iteration of work when you did `W.1`–`W.8` this morning — and that registration
found a real arithmetic bug in the drafts, so the transcription is not
mechanical and is worth doing carefully. **`PL.02` first**, because reshaping
gain is the decree's actual load-bearing argument (*"a frozen tower's reshaping
gain is identically zero"*) and because the file currently claims it is
runnable. **Do not delete any arena reference to reduce the count.** If a seat
is genuinely an END and not an architecture, say that in `CHAMPIONS.md` in one
sentence — that is the legitimate discharge for the three `NO-ARENA` seats
(`ASR`, `Speaker ID`, `Language grounding`).

**B5 — when `D1` fires tomorrow, write the bakeoff the same day.** RANK 5.
`T2.01` frees **35 specs**, has not run since 2026-08-12, and is the gate on all
of Tier 4 and most of Tier 5. `D1`'s default is *run the bakeoff* over A-prime /
B / C / D at matched experience, multi-seed, one pre-registered metric, learning
gate and margin. The default's own note flags the cost to record: **arm D
forecloses `DP.02`**, because private control representations are the
two-towers-in-one-wrapper signature the connectedness directive forbids. Record
that as a cost, not as a thumb on the scale. And note it is `Budget.CPU_DAYS`
capped at `D4`'s already-spent envelope — it is not a GPU fill.

**B6 — three one-line honesty edits, all carried.** (a) `T0.01` and `T0.10` are
the only `PASS` specs with no declared control; add one sentence to each ("a
harness liveness check — a control is undefined for it") so the absence reads as
a decision rather than an omission. (b) `decisions.py`'s docstring should record
that `class` is **self-declared** and the tool takes the author's word — the same
gap you closed for `default` this morning, one function up, and worth naming
even though I found zero live violations across all 13 entries. (c) B4 from the
48th audit (`aggregate-hides-worst-seed` as a `T0` property) is still only
`ROUTED: OPEN` in `REVIEW_QUEUE.md`; it is the guard behind 26 spec files that
fold a worst-case quantity, and its live exposure is zero, which makes now the
cheap moment.

---

## FOR THE OWNER

**1. Eleven pre-registered defaults fire tomorrow, 2026-08-31. Ten of them share
that single date.** They are `D1`, `D3`, `D4`, `D7`, `D8`, `D9`, `D10`, `D11`,
`D12`, `D13`, `D14` (`D15`/`D16` follow on 09-05). Nothing is overdue yet, so
nothing has fired and I have taken no action on your behalf. `decisions --check`
is green and I verified the one enforced safety clause holds against the live
board. **You have roughly seventeen hours to overrule any of them; after that
each fires and is journalled loudly with how to reverse it.** Every default is
now printed **in full** in `python -m experiments.decisions --check` — until this
morning `main()` truncated them at 110 characters, and the live defaults run
369–1041, so **70–89% of every constitutional clause you armed had never appeared
in any report anyone read.** That is fixed; if you read one thing before
tomorrow, read that output.

**2. `D1` is the one that matters, and firing it is good news, not a loss.**
`D1` costs **38 specs** — the largest number on this page by a factor of four —
and its default is not a shortcut: it is *run the bakeoff*, over the arms that
your own PLASTIC-ONLY decree permits, with a learning gate and a margin. `T2.01`
(locomotion) has been `FAIL` for eighteen days and gates all of Tier 4 (unison)
and most of Tier 5 (the thesis) behind it. If you want to steer the architecture
by hand, tomorrow is your last cheap moment. **If you do nothing, the system
starts measuring instead of waiting, which is what the 2026-08-24 ruling asked
for.**

**3. Nothing needs a decision from you that the system could have settled
itself.** I checked all 13 open entries for a means fork parked on your desk —
the `D1` disease — and found none. `D9`'s bakeoff has already been run and
adopted nothing; it waits on you only because adopting a new body would
invalidate four recorded certificates, which is properly yours.

**4. One thing I could not verify, stated plainly.** Two of the three governance
tools this audit is built on — `decisions.py` and `champions.py` — have no test
in the ledger and are certified only by fixtures their own authors wrote. I
found no false result in either (I re-derived the champions violations and the
decisions blast radii by hand), and `champions.py` has had three real defects
found in it in six days. **The ledger's contents are sound; the instruments that
police the ledger are not all held to the ledger's own standard.** `B1` fixes
it, costs no GPU and needs no ruling from you — I am telling you because
`SYSTEM.md`'s first law is yours, and it currently has an exception nobody voted
for.

---

*49th audit. Instruments at 06:50 UTC: `coverage --check` **rc=2** (`gpu<20min`
NEWLY EMPTY — honest, and not fillable by implementing; the repair is an
unblock), `decisions --check` **rc=0** (13 armed, 0 undeclared, 0 overdue, 0
means-escalated), `champions --check` **rc=0** ratchet with **11** violations
reported (was 12; `World` discharged). Ledger: **86 PASS / 10 FAIL / 4 VOID** of
195 registered. Kaggle `jack-ladder-1788070133` (`T2.14`) RUNNING. The Review
organ was mid-run while this was written and may land further changes.*
