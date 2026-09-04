# OVERSIGHT — 70th audit, 2026-09-04 12:38–13:0x UTC (at `a4f5b8f`)

## VERDICT: DRIFTING — the ledger is the cleanest this organ has measured it, and the decision desk armed a default yesterday that could never have fired in time to do the thing it promised

Say the clean part first, because it is large, it is true, and it is the result
that took the most work to establish.

**Section 1 and section 2 are clean, and I checked them properly rather than
sampling.** All **102** PASS rows: every `commit` still resolves in git, none
is `+dirty`, every spec resolves in `BY_ID`, every one declares a `control`,
every one carries non-empty `control_metrics`, and every one has an
implementation file that `module_path_for` resolves. Over the trailing 7 days
the builder made **126 commits** touching `registry.py`, `registry_expansion.py`
and `experiments/tests/` — 30,715 insertions against 144 deletions — and a
mechanical diff of every `CONSTANT = number` in those files found **exactly one
changed constant, in the strengthening direction**: `T0.21 N_PROPERTIES`
11 → 12. Every removed `control=` / `seeds=` / `falsified_by=` line was replaced
by a stronger one in the same commit — `ME.4`'s control gained a paraphrase-venue
aliveness conjunct, `T2.15`'s seeds went 3 → 7, three ME specs went seeds 1 → 3
with the re-measured nulls recorded, `CU.3` gained a magnitude-matched permuted
control it never had. **No finding in section 2. That is a real result and it is
the third clean week in a row.**

And the three PASSes bought yesterday evening are the best-composed the ledger
has taken in a while: `SO.02` (he says "I'm cold" and it is true of his own
body), `SO.04` (watched and unwatched are bit-identical), `LF.02` (a life
survives `kill -9` exactly). Those are three different kinds of true.

Now the findings, ranked by damage to the trustworthiness of the machine.

---

## 1. `D21`'s default was unfireable, and the defect is arithmetic — the clock fell five days after the event the default commands. THIS ORGAN WROTE IT YESTERDAY. (REPAIRED, and the instrument gap routed)

`D21` is the entry that lifted the Review's W1 recommendation onto the owner's
desk. It was armed on **2026-09-04** — yesterday's audit, this organ — with:

    default:   … the 2026-09-06 FULL Review takes the W1 design as the FIRST
               item on its docket, ahead of the two `d10-*` gate rows …
    decide_by: 2026-09-11

A default fires only when its date passes unanswered. **This one becomes due
five days after the sitting it instructs**, and the next FULL Review after
09-11 is 09-13. On the day it fired it would have ordered a Sunday that had
already happened to re-order its docket. A default that cannot perform its own
action on the day it fires is not a weak default — it is the `D1` deadlock with
a clock painted on it, and `D1` cost this project twenty days and 38 blocked
specs.

**It is not a hypothetical.** `docs/PROGRESS.md`'s 2026-09-04 `FOR THE OWNER`
item 3 puts the pair of `d10-*` gate rows ahead of the world row on Sunday's
docket — precisely the ordering `D21`'s default exists to override. So as of
today the desk is scheduled to do the opposite of the default, and the clock
would not have fired in time to say so. The whole entry would have aged out
having achieved nothing, which is the exact failure `D21` was created to end,
recurring inside `D21` itself.

**No instrument catches this class.** `decisions.py` checks that a default
EXISTS, that its `class` is legal, that its `decide_by` parses as ISO, and
(since 08-30) that firing it cannot leave a GOAL.md commitment claim-dead. It
never asks whether the default's own action is still AVAILABLE at `decide_by`.
Nine of the ten armed defaults name a dated or event-bound action; this was the
first one where the two dates crossed, and nothing would have printed.

**Repaired here, in the only direction a deadline may move on its own:**
`D21.decide_by` **2026-09-11 → 2026-09-05**, with a dated `AMENDED` block that
states the defect, the shortening, and its cost to the owner (a seven-day window
becomes one). Shortening tightens; it widens nothing, moves no threshold, and
the default text and option set are untouched. `decisions.py` marks overdue at
`(today - decide_by).days > 0`, so the default now becomes due on the morning of
**2026-09-06** — in time, but only just, which is why B3 below pins the hour.

## 2. The owner-ask reader can be silenced by a quotation made for an unrelated purpose — found live, on this audit, by this audit

`decisions.py`'s `UNROUTED-OWNER-ASK` marks a Review item as routed when a
6-token verbatim span of it appears anywhere in `DECISIONS_NEEDED.md` or
`DECISIONS_RESOLVED.md`. The docstring defends this well: it rewards the repair
the system actually performs and has no similarity threshold to tune.

**It also has no attribution.** Writing the `D21` amendment above, I quoted the
Review's docket sentence — *"the two `d10-*` gate rows first … then
`w0-too-shallow`"* — as **evidence of the defect**, with no intention of routing
anything. `PROGRESS #3` immediately dropped off the `UNROUTED` list. The count
went 3 → 2 and the ratchet stayed green, and the ask had reached no desk at all.

I caught it because I re-ran the tool after the edit and the item I expected was
missing. **Nothing in the system would have caught it.** A quote inside a queue
row, a lesson, a resolved entry, or an unrelated decision silences an ask
identically, and the only visible symptom is a number going down — which every
ratchet in this repo is built to read as good news.

Repaired locally by paraphrasing the sentence (and the paraphrase says why, in
the file, so the next editor does not re-introduce it). The durable repair is
B2: make the match **attributable** — print `matched-by: D21` beside a silenced
ask, and require the matching span to fall inside a `## D…` entry rather than
anywhere in the corpus.

## 3. The CPU day meter has 3600 s of slack, only certificate churn has ever spent it, and it has foreclosed the entire never-run `cpu<2h` population for 22 of today's 24 hours

The arithmetic, computed live at 12:4x:

    CPU_DAY_CEILING_S                       57 600 s
    enum child estimate, cpu<2h             54 000 s   ← 93.75% of the whole day
    ------------------------------------------------
    slack before the class forecloses        3 600 s   (one hour)

    billed today (experiments/cpu_budget.json)  6 111.81 s
      detached:gate_sweep_cpu2h.log             4 560.65   certificate sweep
      detached:rebuy_xl00.log                   1 171.28   re-buy
      LC.02 140.29 · T0.28 137.35 · T0.34 48.49 · T0.17 17.09
      T0.21 17.04 · T0.33 8.27 · T0.27 5.95 · T0.31 5.40      re-buys / re-stamps
    ------------------------------------------------
    remaining                                  51 488.19 s   < 54 000 → class closed

**Every line item is a re-buy or a re-stamp sweep. Not one second bought a new
measurement about Jack** — the Review said exactly this yesterday about the
meter's first day (5,906.8 s, same shape). What the Review did not compute is
the consequence: **the single detached gate sweep, 4,560.65 s, exceeds the whole
day's slack on its own.** It finished around 01:30. From that moment all **39**
never-run `cpu<2h` specs were foreclosed until midnight — 22.5 of 24 hours — and
the same was true yesterday.

**And the projection cannot help the specs that need it.** `child_estimate_s`
returns a MEASURED estimate only when the ledger already carries a duration for
that spec; a never-run spec has none by definition and always pays the enum. The
69th audit's B4 improved the gate from 53 foreclosed to 36 by adopting measured
costs — a good repair — and the builder recorded honestly in `9c5e74a` that *"all
36 are specs never run"*. That residual is not a tuning gap. It is structural:
**the gate is hardest, permanently, on exactly the population the ladder needs
to move**, and it is spent by our own audit-and-re-buy churn.

**Cost today: zero.** No dispatch was attempted (the board is empty), and there
is no `cpu-refused` line in `ladder.log` — the 68th audit's B4 trace is armed
and silent, correctly. **Cost the day the 09-06 design unblocks anything: the
whole class**, on any day whose maintenance sweep has already run, which is
every day. This is the `pace_gate` shape the Review named and did not compute —
a throttle regulating the builder against a quantity our own churn generates —
and `pace_gate` cost 66 dark hours in August before anyone saw it. It is one
week old. It is not costing anything yet. It should not be three weeks old
before the number is on the owner's desk, so it is FOR THE OWNER item 2 and it
is `D20`'s input, measured rather than argued.

Distinct from the routed row `cpu48h-class-self-forecloses-the-day-meter`
(DUE 09-08), which is about `cpu<48h` and the detached lane. This is `cpu<2h`
and the runner lane, and no row owns it.

## 4. The desk that gates everything is diverging, the Review indicted itself for it, and the ask is now ROUTED rather than on a page that is rewritten every morning

`run review-queue`, trailing 7 days / 7 consumer cycles:

    arrived   30  (4.29/cycle)      33 live rows, oldest 11 days
    disposed   1  (0.14/cycle)      2 ACTED of 35 routed, 0 DECLINED
    designed   0                    DISPOSITIONED is not disposal
    drain     UNBOUNDED             net +29 over the window
    2026-09-06 carries 6 rows against a measured capacity of 1/cycle

`coverage` says the same thing without being asked: **0 FRESH dispatches at any
of the seven cost classes**, and every non-fillable class names the same reason
— *"the repair is a REDESIGN"*. Of the 9 specs whose dependencies all PASS, 8
are parked or pilot-blocked behind this desk and 1 (`HR.1`) is held by `D19`.
**None waits on the builder. None waits on compute.**

The Review published this yesterday, indicted itself in its own words (*"I am
the consumer. This is a finding about me."*), and asked the owner for
draft-then-ratify. It wrote it into `docs/PROGRESS.md`, which is current-state
by design. `decisions --check` printed it as `UNROUTED-OWNER-ASK` — the check
built for this the day before yesterday, catching its second instance within
twenty-four hours, on the same page, for a different recommendation.

**Routed here as `D22`**, quoting the ask verbatim, class `goal`, default
(i) THE RULE STANDS, `decide_by` **2026-09-08**. The rule-3 test was applied
rather than assumed: no bakeoff can settle who is *permitted* to draft a
redesign — that is CONDUCT, class 3, and `SYSTEM.md` says escalate exactly when
the fork turns on what is permitted rather than on what works. `UNROUTED` is now
**2/3** and shrinking; the remaining two are the Review's own status paragraphs,
whose `NO-DECISION:` repair is already written into `scripts/review_prompt.md`
and will land on its next run.

## 5. The rest of the audit, stated plainly

**Section 3 — drift.** In the last 24 h: 3 claim PASSes about Jack (`SO.02`,
`SO.04`, `LF.02` — voice, spectating, memory-across-lives, all with GOAL.md
sentences behind them), 2 machine PASSes (`T0.33`/`T0.34`, the CPU accountant —
GOAL.md's *"protects the honesty of watching what happens"*), a
`DUAL_PROCESS.md` research sweep owed since 08-10, `LANGUAGE_GROUNDING.md`
§2.2–§11 with 10 verified citations, and `LG.03`–`LG.06` registered under the
5-step cross-check (registry 234 → 238). **Nothing served no GOAL.md sentence.
No drift found.**

The converse, which is the harder question, and the number is uncomfortable:
of the **102** PASS rows, **33 are Tier 0** and 13 are Tier 1 — 45% of the
scoreboard is the harness — **20 are fixtures, sensors or rules**, and only
**14 declare `COVERS: … (claim)`** against a GOAL.md commitment. `coverage`
still reports **4 CLAIM-DEAD** commitments (smell, balance, shelter/building,
thermal), **9** with live claim specs and nothing passing, and **7**
CITED-BUT-UNRUNNABLE GOAL.md citations of which **4 are new** (`GEN.02`,
`GEN.03`, `GEN.06`, `GEN.09`, all welded behind `LC.07`) — routed as
`goal-cites-four-specs-that-resolve-to-corpses`, DUE 09-10.

**Section 4 — the builder is alive and it is not the problem.** 13 iterations
2026-09-04 00:07–12:19, **13 × rc=0**, zero `PACING:` skips (last 08-29), load
0.00–1.07, 11 GB free, tree clean and pushed every iteration. **PASS delta 0**;
the ledger has not moved since 2026-09-03T23:23. `REFUSING fable — week:Fable
99%` on every start with the run taken on Opus against `week:all models` 57% —
the correct meter, printed with both readings, exactly as `D14` requires. The
builder also *declined* to build a `MANDATORY`-clause scanner and recorded why.
That is the right instinct on an empty board and it deserves saying.

**Section 5 — compute honesty.** Kaggle `2026-W35`: **18.93 h of 30 used, ~11.07
h remaining, expiring 2026-09-06 00:00.** Expired unspent in the three prior
weeks: W32 ~8.8 h, W33 ~22.1 h, W34 ~28.4 h — **~59 free GPU-hours burnt by the
calendar in three weeks**, and W35 is on course to add ~11. Zero GPU hours were
spent in the last 24 h and **no GPU hours anywhere in the budget lack a ledger
row**. This is inventory, not waste-by-negligence: every GPU-runnable spec is a
settled FAIL whose re-run is a seed lottery, or parked. The Review is right that
manufacturing a dispatch into a dying quota is the failure mode. It is still the
fourth consecutive week and it belongs in front of the owner as a number.

**Section 6 — stuck decisions.** `decisions --check`: **0 MEANS-ESCALATED, 0
UNDECLARED, 0 OVERDUE.** Nothing on the owner's desk that a measurement could
settle. Nothing was quietly acted on without being recorded — the reverse
happened, well: the 69th audit's `repaired_by` question was ruled NOT-THE-OWNER'S
and written into `DECISIONS_RESOLVED.md` with its three losers and a reversal
line. **Three defaults now come due on 2026-09-05/06** — `D15`, `D16`, `D21` —
see B3.

**Section 7 — bakeoff hygiene.** One standing violation, unmoved and correctly
reported by `champions --check`: the **Learning core** seat is held **BY
VERDICT** — the strongest marking in `CHAMPIONS.md` — off `LC.03`, which is a
**VOID**. `SYSTEM.md` is unambiguous that a VOID decided nothing ("fix the arm,
do not decide"), and all three of that seat's pre-registered re-open triggers
are now closed doors (`LC.07` PILOT-BLOCKED, `LC.03` VOID-FORECLOSED, `UB.10`
VOID). The **World** seat is held BY VERDICT with no deciding run named and no
trigger declared at all. These are at the ratchet's baseline, both are routed
(`d10-*` rows, DUE 09-06), and neither moved today. They remain the oldest
integrity debt on the board: **two of this project's most consequential seats
rest on a marking nothing bought.**

**Ratchets.** coverage rc=2 (known red) · decisions rc=0, unrouted **3 → 2** ·
champions rc=0, 11 violations all at baseline · review-queue rc=0, 0 violations ·
`unreachable` 94/238 at its declared floor · review liveness OK (DAILY 09-04,
FULL 08-31, next FULL 09-06 inside cadence).

---

## FOR THE BUILDER

**B1 — `decisions.py` gains `DEFAULT-ACTION-EXPIRED`, and it is the finding
this audit paid for.** A default whose prose names a date EARLIER than its own
`decide_by` cannot perform its action on the day it fires. Parse dates out of
the `default:` text (the field is already fully joined across continuation
lines, and `blast_radius` already mines that same text for spec ids, so the
machinery exists); report any that precede `decide_by`. Baseline it at the live
reading in the shipping commit, shrink-only, in the `BASELINE_UNDECLARED` idiom.
The scar to cite is `D21`, armed 2026-09-04 by the overseer with a 09-06 action
and a 09-11 clock, caught by hand the next morning. **This is the third time a
decision-desk guard has shipped checking the FORM of a declaration and not its
CONTENT** — `decisions.py` itself said so in 2026-08-30's correction ("the tool
checked that a default EXISTS… it never read the default's content"), and this
is the same hole one field over.

**B2 — make the owner-ask quotation match ATTRIBUTABLE.** Two changes, and the
second is the load-bearing one: (a) when an ask is silenced by a shingle match,
print `matched-by: <D-id or file:line>` beside it in the report, so a spurious
match is visible instead of appearing as a number going down; (b) require the
matching span to fall inside a `## D…` entry, not merely anywhere in
`DECISIONS_NEEDED.md` / `DECISIONS_RESOLVED.md`. Live evidence, produced by
this audit at 12:5x: quoting the Review's docket sentence inside `D21`'s
amendment — as evidence, routing nothing — dropped `PROGRESS #3` off the
`UNROUTED` list and moved the count 3 → 2 with the ratchet green. Add the
regression to `T0.28`, whose claim text already owns the owner-ask classes:
*a quote that routes nothing must not silence anything*.

**B3 — three defaults come due, and one of them has an HOUR, not just a date.**
- `D21` (**decide_by 2026-09-05, so due to fire the morning of 09-06**): its
  action is to set the 09-06 FULL Review's docket order. **Fire it in a
  00:xx–05:xx iteration on 09-06, before the Review's ~06:37 run**, or it misses
  the sitting a second time and the shortening in B1's scar bought nothing. Fire
  it by recording the firing in `DECISIONS_RESOLVED.md` in the standing idiom
  ("the owner did not rule by 2026-09-05, so the pre-registered default fired")
  AND by writing the mandated order onto the `w0-too-shallow` row in
  `docs/REVIEW_QUEUE.md`, which is the docket the Review actually reads. Prose
  in a journal is not a docket (`D19`/`HR.1` and `HR.5`/`HR.6` both cost an
  iteration each to learn that).
- `D15` (decide_by 2026-09-05): its default edits `scripts/overseer.sh` and adds
  a usage ledger to every organ script. **`D13` forbids this organ from editing
  its own script, so this firing is yours and nobody has written that down on a
  dated row until now.**
- `D16` (decide_by 2026-09-05): default (b) is a no-op — `T0.27` stays RED. Fire
  it by recording it; change no code.

**B4 — the `cpu<2h` slack is 3600 s and it belongs in `run status`.** Finding 3.
`cpu_foreclosed_now` is printed as a bare count (39) with no floor and no
denominator, and it GROWS with the registry — three specs registered today moved
it 36 → 39 and nothing went amber. Print the arithmetic that makes it
actionable: `slack = CPU_DAY_CEILING_S − max(child_estimate_s over the live
cpu<2h population)` and today's `used_s` against it, so the line reads *"1 h of
slack, 1.70 h spent, class closed since ~01:30, all 39 never-run"* rather than a
number. Do **not** move `CPU_DAY_CEILING_S` — the ceiling is `D20`'s and the
owner's, this is instrumentation only, and a default may not widen a limit.

**B5 — carried forward from the 69th, unchanged and still not yours:** do not
route `D22`. It is an ask about the builder's own authority, the Review raised
it, this organ routed it, and the owner rules it. Standing prohibitions all
still stand: no unchanged `D1.0` re-dispatch; `HR.1`–`HR.4` D19-held to 09-14,
no corpus fetch; `HR.6` behind `HR.5`; `LF.01` attempt 2 waits for the 09-09
design; no third increment of the CPU accountant (B4 is a print statement in
`run status`, not a fourth organ — if it grows past that, stop and say so); let
`W35`'s ~11 free hours expire.

---

## FOR THE OWNER

**1. The Review has asked you for the biggest change to how this project works
since August, and it is now `D22` on your desk with a default and a clock
(decide_by 2026-09-08).** The ask, in its words: *"let the builder DRAFT
redesigns; keep ratification here."* The evidence is not in dispute and both
organs reached it independently: the design desk has closed **2 of 35 routed
rows in 15 days**, the trailing week is **30 arrived against 1 disposed**, the
drain is unbounded, and **all 9 startable specs sit behind it while the builder
runs 13 clean iterations a day with an empty board**. The default if you are
silent is **(i) the rule stands** — that is the only legal default, because
(iii) would widen what the builder is permitted to do and a default may never
widen. Silence is therefore safe and it is not free: it costs roughly 17 more
net queue rows by 09-08 at the measured rate.

You should have my view on the record, because the safeguard the proposal leans
on is *this organ*. Its §2 duty is to audit every spec diff independently of its
author, and it works: 126 commits and 7 days of diffs this week produced one
changed constant, in the strengthening direction. What I cannot tell you is
whether it holds under a load it has never seen — today it catches drift the
builder produces incidentally; under (iii) it would audit redesigns the builder
authored *on purpose*, which is a different adversary. That is the real question
in this fork and no tool I own can settle it.

**2. `D20`'s input, and it is now measured on both lanes rather than one.** The
CPU day meter's second full day billed **6,111.81 s across ten line items and
all ten are certificate re-buys or re-stamp sweeps** — zero seconds of new
science, exactly as yesterday. New today, and this is the part the Review did
not compute: the day ceiling is **57,600 s** and a never-run `cpu<2h` spec costs
**54,000 s** at admission, so the class has **one hour of slack per day** — and
this morning's single detached certificate sweep spent 4,560 s of it by 01:30.
**All 39 never-run `cpu<2h` specs have been foreclosed for 22 of the last 24
hours, by our own bookkeeping.** It cost nothing today because the board is
empty. It costs the entire class the day the 09-06 design unblocks anything. I
am deliberately not recommending a number — every recommendation available to me
points at loosening a ceiling and that is not my direction to push. The pattern
is `pace_gate`'s, which cost 66 dark hours before anyone saw it, and this is
week one.

**3. Free GPU quota expired unspent for a fourth week.** W32 ~8.8 h, W33 ~22.1 h,
W34 ~28.4 h — **~59 hours gone by calendar** — and `2026-W35` will add ~11 more
at 00:00 on 09-06. Nobody did anything wrong: every GPU-runnable spec is a
settled FAIL whose re-run is a seed lottery, or parked, and manufacturing a
dispatch to spend a dying quota is the failure mode rather than the fix. It is
reported as a number and not as a fault, for the fourth week, because at some
point the fact that the free compute cannot be spent *is* the finding.

**4. Two of Jack's most consequential seats are held by markings nothing
bought,** unchanged for a fifth day and correctly reported by the instrument
every time. The **Learning core** is held BY VERDICT off `LC.03`, which returned
**VOID** — and `SYSTEM.md` says a VOID decides nothing — with all three of its
re-open triggers now closed doors. The **World** seat is held BY VERDICT with no
deciding run named and no re-open trigger declared at all. Both are routed onto
Sunday. I raise them here because the file's strongest marking resting on a
non-verdict is the one kind of drift that makes the whole scoreboard mean less,
and because it has now survived five audits that each reported it correctly.

---

## The honest summary — are we closer to a curious humanoid, or to a longer list of green ticks?

**Yesterday, genuinely closer.** He said something about his own body that could
have been false and was not. He was shown to be the same animal watched and
unwatched, which is what makes it legitimate for anyone to look at him at all.
A life of his was killed outright and resumed bit-exactly, which is what turns a
life from something that runs into something that can accumulate. Three claims,
three commitments that were reading zero or one, none of them a fixture.

**Today, neither.** Thirteen clean iterations, zero PASSes, and the ledger has
not moved in thirteen hours — not because the builder faltered (it delivered a
research sweep owed since August and four new specs under a cross-check that
caught a missing control in its own drafts) but because there is nothing it is
permitted to run. Forty-five percent of the scoreboard is the harness. Fourteen
of 102 PASSes are claims about Jack against a GOAL.md commitment. Four
constitutional commitments have no runnable falsifiable claim at all, and not
one of them died because Jack failed to learn — they died because the world he
lives in cannot host the measurement.

**And the thing I am most uneasy about is not any of that.** It is that this
organ armed a default yesterday whose clock fell five days after the event it
commanded, published it as the repair for exactly this failure mode, and no
instrument in the system would have said a word. The machine is very good at
catching what the builder does. It is markedly worse at catching what its own
governance writes, because governance is prose and prose is what everything here
is built to distrust — and every guard we have added to that desk so far has
checked the FORM of the declaration rather than what it says. That is now three
for three. B1 fixes this instance. The pattern is the finding.

The list of green ticks did not get longer today. Whether we got closer depends
entirely on a desk that closed one row this week, and the ask that would change
that is on your desk with a clock on it for the first time.
