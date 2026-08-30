# OVERSIGHT — 51st audit, 2026-08-30 18:45 UTC

## VERDICT: ON TRACK — and one armed default, firing tomorrow, would have deleted today

Sections 1, 2 and 7 have **no findings**, checked mechanically rather than by
reading commit messages. All 91 `PASS` rows resolve to an implementation that
calls `run_spec`, to a commit that exists in git, and to recorded
`control_metrics` wherever a control is declared. Over seven days exactly one
numeric bar moved downward (`DECAY_MIN` 1.5 → 1.25, adjudicated by the 50th
audit and still correct); no control was deleted, no seed count was reduced
(the ME family went 1 → 3), no `_check` gained an admitting disjunction.
`DECISIONS_RESOLVED.md` contains no decision made inside its own noise margin
and no VOID treated as a verdict.

Of the three mandatory ratchets: `decisions` `rc=0` (**0 UNDECLARED, 0
MEANS-ESCALATED** — there is nothing left to arm, and I say so rather than
manufacture an entry), `champions` `rc=0` at 9 standing violations against its
baseline, `coverage` **exits 2** on `2 cost class(es) NEWLY EMPTY —
gpu<20min, gpu<2h`. That red is honest and is §5.

The builder ran **24 iterations in 24 h, 21 at `rc=0`, 3 at `rc=124`**, and
moved the ladder **84 → 91 PASS of 200**. It recorded four registered verdicts
today: `W.1` FAIL, `W.2` FAIL, `PL.00` FAIL, `LG.01` PASS.

**The finding that outranks everything else is not about the ledger. It is
about whether the builder exists next week.** `week:Fable` has been at 100%
for the entire day, and **19 of 19 iterations today ran on the Opus fallback**
— including every one that produced a claim. `D14`'s armed default fires
tomorrow and, implemented from the words in its own `DECIDE:` block, aborts
the slot instead of falling back. Applied to today it would have produced
**zero iterations, zero verdicts, zero PASS movement**. Its own cost line says
*"this option buys honesty, not throughput"*; the throughput it costs was
never named, because on 2026-08-26 when it was armed the builder was at zero
iterations and there was nothing to lose. There is now.

---

## 1. Integrity of the ledger — NO FINDINGS

108 rows: **91 PASS, 13 FAIL, 4 VOID**, of 200 registered specs.

Checked mechanically over every `PASS` row (`/tmp/audit2.py`, re-derivable):

| check | result |
|---|---|
| `commit` resolves in git (`cat-file -e`) | **0 phantom commits** of 91 |
| an implementation exists in `experiments/tests/` | **91 of 91** |
| implementation passes `control_fn=` where a control is declared | **89 of 89** |
| `control_metrics` recorded when a control is declared | **89 of 89** |

The two `PASS` rows with no declared control are `T0.01` (repo imports clean)
and `T0.10` (Kaggle round-trip). Both carry `control=None` and
`null_baseline="n/a — structural precondition."` in the registry, neither
asserts a capability, and the one-line docstring repair is still the 49th
audit's B6(a). Unchanged, still costs nothing, still not re-ranked.

Carried and unchanged from the 50th: `T2.02`'s VOID row is stale by content
(the test file at HEAD differs from the blob that stood at its `ran_at`); 18
entries predate `impl_sha` and 57 `PASS` rows predate `spec_sha`. A re-run
upgrades each. None of these is a claim standing on nothing; they are stamps
that cannot answer a question the record now knows how to ask.

**I looked hardest at today's only new PASS, because it is load-bearing.**
`LG.01` certifies the probe set that `LG.00` — GOAL.md's *"proof he is a
creature and not a costume"* — will be scored on. It went VOID at
`calib_acc 0.2500` (exact chance), was diagnosed to a letter-readout failure
in the 360M parent, repaired to continuation scoring, and re-run to
`calib_acc 0.8333`. **The repair is strengthen-only and I verified it as
such**: `CHANCE_BAND_HI` moved 0.25 → **0.0** (a question is now retained only
if the null is outright wrong, not "wrong in 3 of 4 placements"), while
`RETAIN_MIN` 20, `ORACLE_MIN` 0.95 and `CALIB_MIN` 0.50 are untouched. A
stronger null retains fewer questions, so the spec's own bar moved in the
harder direction. No violation.

**One limitation, recorded rather than ranked as a defect.** `LG.01`'s
calibration leg is **12 questions**, deterministic, and `calib_acc_std` is
`0.0` across all three seeds — the seeds vary the generated life, not the
null, so multi-seed protection does not reach this leg at all. It is the only
gate separating `LG.01`-PASS from `LG.01`-VOID (without it, attempt 1 was a
clean PASS against a dead null). At 10/12 versus a 0.50 bar that is a real
signal (one-sided binomial *p* ≈ 0.019), but it is thin for something a Tier-4
constitutional claim rests on. `LG.00` has already widened its own liveness
floor to **30** general questions of which 12 are shared, so the successor is
not certified against the same 12 alone — the builder saw this and fixed it in
the same day. Recorded here so the thinness of `LG.01`'s own leg is on the
record, not as work owed.

## 2. Thresholds and controls over time — NO VIOLATIONS

Every named numeric constant in `registry.py`, `registry_expansion.py`,
`experiments/tests/`, `protocol.py` and `bakeoff.py` that has both a `-` and a
`+` form in the last seven days — i.e. every constant that *changed value*
rather than being introduced:

| spec | constant | move | direction |
|---|---|---|---|
| VO.02 | `COORD_MARGIN` | 0.20 → **0.35** | tightened |
| LG.01 | `CHANCE_BAND_HI` | 0.25 → **0.0** | tightened |
| LG.01 | `N_ORDER` | 4 → **removed** | readout replaced, see §1 |
| T2.11 | `_SEC_PER_SEED` | 1200.0 → **355.0** | a cost estimate, not a gate; measured |
| T2.09 | `DECAY_MIN` | 1.5 → **1.25** | **loosened — declared, adjudicated** |

`DECAY_MIN` is the same one the 50th audit cleared: a rig bar, not a claim
bar, on a spec whose `run()` refused until the freeze commit, derived from
what the gate is *for* rather than shaved to the observed minimum, and stated
in `44f24c41` under the heading `ONE BAR MOVED, DOWNWARD, IN THE OPEN`. I
re-checked and did not move it, per the 50th's B4(a).

**One thing that looked like a violation and is not**, stated because the next
audit will hit it too: `f9549cb` shows `- control="Reflections generated from
ANOTHER agent's log must hurt."` in `ME.3`. The control was **not** deleted —
the line was reflowed to add `seeds=3` and a `notes` field, and the control
survives verbatim two lines below. The same commit raised `ME.1`, `ME.3` and
`ME.4` from **1 seed to 3** and recorded that the single seed had been
reporting `ME.1`'s null *"a factor of two kinder than it is"*. That is a
strengthening that cost a headline, done voluntarily, 22 days after the row
went green.

Zero `control_fn` removals in seven days. Zero seed reductions.

## 3. Drift from the goal — none in what was built; the gap is unchanged and now has a number attached

Every unit the builder touched in 24 h traces to a GOAL.md sentence:

| unit | GOAL.md sentence it serves |
|---|---|
| `W.1`, `W.2` (FAIL) | *"the world must be **consistent** … so rules are learnable"*; *"too cold kills him"* |
| `PL.00` (FAIL), `PL.02` registered | *"PLASTIC ONLY — nothing inside him is frozen"* (GOAL.md:76) |
| `LG.01` (PASS), `LG.00` implemented | *"strip the diary and the learned core, and his answers about his own life must COLLAPSE"* |
| `BA.03` implemented | *"proprioception & balance"* in the sensory inventory |
| `T0.28`–`T0.30`, T0 re-runs | the harness itself — SYSTEM.md's *"is the machine better than I found it?"* |

**No drift.** Nothing in the last day serves no sentence.

**The converse, which is the real answer to this section.** `coverage` reports
**12 commitments with a live claim spec and nothing passing**: touch/contact,
tool use, smell, proprioception, shelter/building, balance, death & retry,
taste-adjacent thermal (kills), plasticity, sleep, hunger/thirst, fast/slow.
The three GOAL.md names as most likely to be quietly neglected stand at:
**curiosity 2 passing of 12 specs**, **one brain / unison 1 of 22**,
**learning-by-living — the entire `NE` family (8 specs) blocked behind `NE.01`
FAIL**.

**And today the character of one of those gaps changed, which is worth saying
precisely.** `W.1` and `W.2` are the first specs this project has ever run
that measure the **world** rather than Jack. They found that W0's shivering
gain (33.33 W/C) exceeds its dry conductance (14.29 W/C), so a body that does
nothing parks at 34.000 °C forever, and the ambient that would actually kill
it is **exactly 0.0 °C against a world whose night is 20 °C**. GOAL.md's
sentence is *"too cold kills him, too hot kills him"*. The world as built
**cannot kill him by cold**, and that is now measured rather than suspected —
it is also a direct quantitative account of `SH.02`'s saturated null and
`SH.01`'s `ORACLE_CANNOT`, which two exhaustive geometries failed to reach.

I want to be careful about what this does and does not cost. It does **not**
reduce reachability: under `D2`'s BLOCK semantics `NOT_RUN` already blocked, so
`W.3`/`W.4`/`W.5`/`W.7`/`W.8` and `PL.02` were unreachable yesterday too. 74 of
200 specs are unreachable and that number did not move. What changed is the
**kind** of repair owed: `W.3` — the registered spec for *"cold kills, and
shelter is why it does not"* — went from *"nobody has run its dependency"*
(cost: one iteration) to *"its dependency ran and the world failed"* (cost: a
change to `needs.py`, which is a design decision nobody has taken). Same for
`PL.02`, the sole falsifier of the PLASTIC-ONLY decree, which became
unreachable within eight hours of being registered to discharge a
`champions.py` violation.

**No instrument distinguishes those two states.** `run blocked` prints
`W.1 = FAIL frees 1` immediately below `T2.11 = NOT_RUN frees 1`, ranked by
specs-freed, in identical shape. Specs-freed is the right axis for scheduling
and the wrong axis for knowing what sort of work is owed. Today four of the
fourteen terminal blockers are FAILs of the world itself (`W.1`, `W.2`,
`NE.01`, `DP.05`) and five are VOID rigs. Filed as **B3**.

The builder's own conduct on `PL.02` is the right precedent and I want it on
the record: it noticed that editing `PL.02`'s dependency an hour after `PL.00`
produced an inconvenient FAIL *"is the shape of a weakening"*, and routed it to
the Review with three costed arms instead of taking it. That is law 4 obeyed
where obeying it was expensive.

## 4. Is the builder alive and productive? — YES, and entirely on the fallback

Window 2026-08-29T18:07 → 2026-08-30T18:07:

| | |
|---|---|
| iterations started | **24** |
| ended `rc=0` | **21** |
| ended `rc=124` (50-min timeout) | **3** (00:57, 05:57, 12:57 — 12.5%) |
| PASS delta | **84 → 91** (+7) |
| iterations that ran on the primary model | **0 of 24** |

**`week:Fable` has read 100% since before 00:07 today.** Every iteration
logged `LIMITED on fable (credits or session) — falling back to opus` and ran
a full unit on Opus. The fallback chain is not a degraded mode here; it is the
only mode, and it produced the entire day's science: `W.1`, `W.2`, `PL.00`,
`LG.01`, the `W.1`–`W.8` registration, `PL.00`/`PL.02` registration, `BA.03`,
`LG.00`'s implementation and pre-registration, and eight T0 regression stamps.

Current meters, read at 18:39 UTC: `week:all models` **84%**, resets
**Aug 31 05:00 UTC** (~10 h). The 90% hard stop is ~6 points away and the
observed burn is ~1 pt/iteration, so the builder will likely stop itself for
the last few hours before the reset. That is the gate working as designed and
I am not flagging it.

**The 12.5% timeout rate is worth one line.** Three lost 50-minute iterations
in 24 h. The inheritance mechanism exists and works (`T3.06` attempt 1 was
inherited from a timed-out iteration and replayed offline, `dd4d3f9`), so
these are not silent losses — but three in a day is the highest rate in the
last week and nothing tracks it as a rate.

## 5. Compute honesty — the queue is the constraint, and W35 is FRESH

`gpu_budget.json`, by budget week (`%U`, Sunday-start, matching Kaggle's own
reset):

| week | kaggle h used | of 30 | jobs | failed |
|---|---|---|---|---|
| 2026-W32 | 16.61 | 55% | 17 | 4 |
| 2026-W33 | 7.89 | 26% | 22 | 4 |
| 2026-W34 | 1.62 | 5% | 4 | 0 |
| **2026-W35** | **1.01** | **3%** | 3 | 0 |

**Today (Sunday 2026-08-30) is day 1 of W35, not its last day** — I checked
`Gpu._week()` uses `%U`, so W35 runs 08-30 → 09-05 and there are **~29 free
hours and six days left**. The 50th audit's figure was right. No emergency.

**There is no waste to find: there are no GPU hours spent without a ledger
entry.** The problem is the opposite one and it is structural. `coverage`'s
queue block:

```
gpu<20min   0   EMPTY   <- NOT FILLABLE: no runnable spec to implement
gpu<2h      0   EMPTY   <- NOT FILLABLE: no runnable spec to implement
gpu<8h      1   T2.02   (VOID — an arm to repair, not a dispatch)
```

All four dispatchable specs in the whole project are VOID. Both empty GPU
classes are `NOT FILLABLE` — every unimplemented spec at that cost is blocked
upstream, so writing a new spec cannot clear them. The only repair is an
unblock, and the top unblock is `T2.01` (frees 35), which is blocked on `D1`,
which fires tomorrow. This is the same chain that cost ~59 free GPU-hours
across W32–W34 and it is unchanged.

`coverage` exits **2** on `q["new_empty"]` (`coverage.py:979`) — the GPU red.
The `cpu<10min` stale-baseline line is only worth exit **1** and is not what is
red. The builder declined to edit `QUEUE_EMPTY_BASELINE` this afternoon because
implementing and running `LG.01` in the same hour moved the count by zero, and
routed it to the Review instead. That is correct and I am not re-ranking it.

## 6. Stuck decisions — NOTHING IS STUCK, and nothing is left to arm

`decisions --check`: **0 UNDECLARED, 0 MEANS-ESCALATED, 0 OVERDUE.** All 15
open entries carry a `DECIDE:` block with a default and a date. My standing
duty is *"arm at least one per audit"* and **there is nothing to arm** — the
ratchet reached its floor on 2026-08-26 and the remaining entries are all
armed. Inventing one to look useful would be the opposite of the job.

Fourteen defaults fire in the next 36 hours: **D1, D3, D4, D7, D8, D9, D10,
D11, D12, D13, D14 on 2026-08-31**, **D15, D16 on 09-05**, **D17 on 09-07**.

The assurance gap the 50th audit carried is unchanged and I confirm it:
`MEANS-ESCALATED` fires only when an author types `class: means`, and **15 of
15 open entries type `goal`**. It has never fired on live data. I read all
fifteen; I am not claiming any is misfiled. It remains a gap in assurance, not
evidence of a false escalation — and `SYSTEM.md` now honestly records that two
of the three safety clauses on defaults are enforced by nobody.

**My one substantive finding in this section is D14, and it is §"FOR THE
OWNER" 1 below.** Evidence appended to `DECISIONS_NEEDED.md` this audit.

## 7. Bakeoff hygiene — NO FINDINGS

`DECISIONS_RESOLVED.md` holds three entries.

- **`PS.01/J` — VOID**, recorded as VOID and *not* used to decide anything:
  three arms below the 3.0σ learning gate. A VOID correctly refused a verdict.
- **`PS.01/J2` — WINNER `impact_speed`**, 10.32σ over null, beating the
  runner-up by **2.66σ**. Outside the noise margin. The `screen` gate mode
  carries a written rationale for why observables are not learners, and it
  names the T2.02 ambiguity it is exempt from.
- **`D2` — WINNER BLOCK**, decided by ledger replay rather than `run_bakeoff`,
  with the method justified in the entry (no seed noise, no null, no training
  that could have failed), the loser recorded with what survives of it, and a
  **re-open trigger** naming the exact quantity the trade rests on.

No decision inside its noise margin. No VOID promoted to a verdict. No
decision made without a learning gate where one applies.

## 8. The honest summary — closer to a creature, or just to more green ticks?

**Closer to a creature, and by an unusual route: three of today's four
verdicts were FAILs, and they are the most valuable thing the project produced
this week.**

`LG.01` is the green tick, and it is a real one — the same frozen 360M parent
is alive at **0.833** on the world's general knowledge and at **chance 0.271**
on questions about Jack's world, body and history. That contrast is the
measured form of GOAL.md's *"he should be smarter inside his life and dumber
outside it"*, and it is what makes `LG.00` scoreable at all.

But the thing that actually moved the project is that **the world went under
the ladder for the first time.** For twenty-two days, five separate
instruments (`LC.03` darkroom, `LC.03 v2`, `DP.05`, `SH.01 ORACLE_CANNOT`,
`SH.02` headroom) pointed at W0 as the bottleneck and none of them could say
so as a claim. `W.1` and `W.2` now do, with arithmetic: the world cannot kill
him by cold, it has no single time-compression factor (implied *k* ranges 72 to
864 across subsystems, and no choice of *k* repairs it because the *ratios*
between needs are wrong too), and its shelter is a hyperthermia trap by day.

GOAL.md says *"the needs ARE the curriculum"* and *"cold nights teach
shelter-building the way no scripted lesson can."* We have now measured that
the cold nights do not teach, because they do not bite. That is a refutation of
a constitutional sentence — **about the world, not about Jack** — and it is
exactly the kind of finding the ladder exists to produce. It is also the
reason curiosity sits at 2/12 and unison at 1/22: a world with no
consequences cannot grade an explorer.

What keeps this from being unqualified: nobody has taken the decision about
what to do with `needs.py`, `W.3` is blocked behind those two FAILs, and the
one thing that would keep the builder producing at this rate is an armed
default that would have stopped it 24 times today.

---

## FOR THE BUILDER

**B1 — RANK 1. When `D14`'s default fires tomorrow, implement the 95% check on
the model that WILL ACTUALLY RUN, not on the primary — and record the measured
cost of the strict reading either way.**

`D14`'s default names *"a pre-flight check in `scripts/ladder_loop.sh` before
`run_claude`, at a 95% floor on the loop model's own weekly line."* The literal
reading aborts the slot when `week:Fable` ≥ 95%. **Applied to today, that is 19
aborts, 0 verdicts, 0 PASS movement** — because `week:Fable` read 100% for the
entire day and every iteration ran on the Opus fallback. The measured cost of
the literal reading, from today alone:

| what the literal reading would have deleted | |
|---|---|
| iterations | 19 of 19 |
| registered verdicts | `W.1` FAIL, `W.2` FAIL, `PL.00` FAIL, `LG.01` PASS |
| ladder movement | 84 → 91 |
| specs registered | `W.1`–`W.8`, `PL.00`, `PL.02`, `BA.03` |
| the PLASTIC-ONLY decree's first-ever falsifier | `PL.02` |

The other reading — apply the check to the model selected *after* the fallback
chain resolves, and abort only when every model in the chain is exhausted — is
**not a widening**: running on Opus after `LIMITED on fable` is already
permitted, is current behaviour, and is what shipped every claim today. It also
preserves the entry's stated purpose. `D14`'s point 2 complains the switch runs
*"with nothing recording that as an event"*; that premise is already
substantially false — `ladder_loop.sh:241` logs `LIMITED on ${MODEL} — falling
back to ${FB}` and every iteration journal opens by naming the model.

Concretely: put the check inside the `for FB in $FALLBACK_MODELS` loop
(`ladder_loop.sh:238`), not before `run_claude "$MODEL"` at line 228. **Say in
the commit message which reading you took and why**, and if you take the
literal one, log an `ABORT` line that names the throughput it just refused.
Do not decide this by argument if the owner rules first — see FOR THE OWNER 1.

**B2 — RANK 2. `champions.py` still has no declaration syntax, and this is the
eighth audit carrying its consequences.** The tool's own docstring proposes the
durable repair: per-seat `HELD:` / `ARENA:` markers in the same idiom as
`COVERS:` and `DECIDE:`, so the parser resolves instead of inferring from table
structure. Take it. It is the one change that makes 9 standing violations
mechanically true or mechanically false rather than a guess. Today's 9:

- `ARENA-MISSING` ×3 — `Curiosity signal` (`LT.03`, `LT.04`: **register**),
  PLASTIC-ONLY (`LOUD.*`: **register**), `Control architecture (D1)`
  (`D1.0`, `T2.21`: decided against on `a3b12f6`, so the repair is to **correct
  the citation** to the live successor, never to write the spec).
- `NO-ARENA` ×3 — `ASR`, `Speaker ID`, `Language grounding`. The legitimate
  discharge for the first two is one sentence saying the seat is an END, not an
  architecture. `Language grounding (word → lived skill)` is **not** an END —
  GOAL.md makes it a falsifiable claim — and it should name `LG.00` now that
  `LG.00` exists.
- `UNCONTESTED` ×3 — `Fast/slow coupling` (`DP.02`), `Language model` and
  `Language acquisition` (both `LG.00`). **Two of these three discharge the
  moment `LG.00` runs**, which is your next unit. Note it in the commit.

**Do not delete an arena reference to reduce a count** — that converts
`ARENA-MISSING` into `NO-ARENA` and makes the seat permanently safe.

**B3 — RANK 3. `run blocked` ranks by specs-freed and cannot say what kind of
repair is owed.** It prints `W.1 = FAIL frees 1` in the same shape as
`T2.11 = NOT_RUN frees 1`. One costs an iteration; the other costs a change to
the world's physics that nobody has decided to make. Add the blocker's status
and a repair class to each row — `re-run` (NOT_RUN), `fix the rig` (VOID),
`the measurement refuted the design` (FAIL) — and total them. Today that reads
**4 world-FAILs, 5 VOID rigs, 5 unrun** among fourteen terminal blockers, and
no instrument in this project can currently print that sentence. Scar: `W.3`,
the registered spec for GOAL.md's *"too cold kills him"*, changed repair class
today and nothing announced it.

**B4 — RANK 4. The ~1.5 GB memory ceiling is a CONDUCT constant and it was
exceeded twice today, declared but not enforced.** `SYSTEM.md`'s hard
constraints say *"Stay at `nice 19`, under ~1.5 GB RAM"* — class 3, fixed, not
up for measurement. `lg_00_not_a_puppet.py:275` and
`lg_01_lived_necessary_probes.py:121` both declare *"peaks near 2.5 GB"*, and I
measured the live `--llm-pass` at **2.36 GB RSS** at 18:39 with 1.5 GB of swap
in use on a box serving paying tenants. The declaration is honest and the
mitigations are real (detached, short-lived, `nice 19`, fp32 chosen on a
measured 15× throughput argument). The inconsistency is that the same ceiling
was cited as the reason to **reject** the 1.7B parent (6.9 GB) and then not
applied to the 360M that was chosen. Either escalate the ceiling with the
measurement attached, or cap the pass (smaller batch / sharded load) to fit it.
A CONDUCT constant that a docstring may opt out of is not a constant.

**B5 — RANK 5, carried from the 49th (B6a) and the 50th (B6).** `T0.01` and
`T0.10` are the only `PASS` specs with no declared control. Add a sentence to
each spec's `control` field so the absence reads as a decision. I re-verified
both are honest. Bookkeeping, one line each.

**B6 — RANK 6, carried from the 50th (B4), unserved.** `44f24c41`'s claim that
`T2.09`'s seed-selection formula *"reads only the null and the rig instruments
— never the claim arm's dwell, fed-ratio, coverage or margin"* is false as a
summary: `t2_09_*.py:583-589` gates on `claim_static_reward_q1`,
`claim_static_decay` and `exposure_frac_of_random`. **Live effect is zero** —
all three exclusions fired on `trap_dwell` — so **do not move `DECAY_MIN`**;
re-fitting it now would be the real violation. Fix the *sentence*. And cap or
contextualise seed 1's `trap_ratio` of **953,594,661,617.28** (a vanished
denominator, not a spectacular trap) so a ledger reader need not re-derive it.

## FOR THE OWNER

**1. One question, and you have about ten hours. `D14` fires tomorrow and its
default, read literally, stops the builder entirely whenever its primary model
is exhausted — which today was all day.**

You do not need to understand the mechanism to answer. The facts:

> `week:Fable` — the model the builder is configured to use — read **100% for
> all 24 hours** of today. The loop fell back to Opus every single time and
> **ran normally**: it moved the ladder 84 → 91, recorded four verdicts, and
> measured the world's physics for the first time in the project's history.
>
> `D14`'s default, armed on 2026-08-26 when the builder was producing nothing,
> instructs the loop to **abort the hour instead of falling back**. Applied to
> today: **19 aborts, nothing produced.**

The default is not unsafe — it only ever refuses more, so it does not breach
the rule that a default may never widen what is allowed, and it is reversible
by reverting one commit. It is simply expensive in a way its own cost line does
not say: it reads *"this option buys honesty, not throughput"*, written when
there was no throughput to lose.

**If you want the builder to keep falling back, say so in one line** and the
loop will implement the check on the model that actually runs. **If you want
the strict abort**, say that instead and it will be implemented literally, with
the cost logged each time it fires. **If you say nothing, the default fires as
written** — which is the whole point of arming it, and I am not going to
pretend otherwise. Evidence appended to `docs/DECISIONS_NEEDED.md` under
`D14 — EVIDENCE UPDATE 2026-08-30`.

**2. Thirteen other defaults fire tomorrow and I am not asking you about any of
them.** `D1` is the one that matters — 38 specs, `T2.01` alone frees 35 — and
its default **changes nothing you decreed**: PLASTIC-ONLY stands verbatim, the
option that would have narrowed it is struck as unconstitutional, and the
remaining arms go to a bakeoff the loop runs itself. You only need to act if
you want the decree *narrowed*, which is the branch no experiment may take for
you. This is unchanged from the 50th audit and I have nothing to add.

**3. A finding about your world, not about Jack, and it is the most important
thing measured this week.** You wrote *"he must eat, drink, sleep, stay warm —
too cold kills him, too hot kills him"* and *"cold nights teach shelter-building
the way no scripted lesson can."*

The loop measured W0 today and **the cold nights do not bite**. The body's
shivering response is stronger than its heat loss, so it parks at 34.0 °C and
stays there; the ambient temperature that would actually kill it is **0.0 °C**,
and the world's night is **20 °C**. A night in the open is survivable
indefinitely by a body that does nothing at all. Separately, the world has no
single time-compression factor — hunger runs 12× too slow relative to the day,
thirst 6×, and the *ratio* between them is wrong too, so no single correction
repairs it.

This is not a bug report; it is why `SH.01` and `SH.02` could not find their
claims from two exhaustive geometries, why curiosity has 2 passing specs of 12,
and why the survival world cannot yet grade an explorer. **Nothing is asked of
you** — repairing `needs.py` is builder work under rule 3, and the loop has
already routed it. I am telling you because it is your sentence that the
measurement contradicts, and you should hear that from the audit rather than
find it in a commit message.
