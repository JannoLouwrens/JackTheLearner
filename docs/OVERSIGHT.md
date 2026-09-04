# OVERSIGHT — 71st audit, 2026-09-04 18:40–19:0x UTC (at `7eb2b63`)

## VERDICT: DRIFTING — the ledger is clean and the ladder is not moving. Nineteen consecutive iterations bought zero PASSes, 16.18 GPU-hours bought one VOID that is now stale, and this organ's own armed default was about to demote the two cheapest rows on Sunday's docket into a week whose free quota is fresh.

Say the clean part first, because it is large and it is true.

**Sections 1 and 2 are clean, checked mechanically rather than sampled.** All
**102** PASS rows: every implementation resolves in `experiments/tests/` (nine
of them via the `SPEC_ID` indirection, which my first pass mis-read as missing
and which I re-checked before writing this); every `commit` still resolves in
git — **zero dangling**; every spec declares a `control`; every one wires
`control_fn` except `T0.01` and `T0.10`, both of which declare
`"NONE, BY DECISION (52nd audit B5)"` with the reason stated in the registry, so
the absence is pre-registered and not silent. Over the trailing 7 days the diff
of `registry.py`, `registry_expansion.py` and `experiments/tests/` shows **no
constant moved in the loosening direction, no control deleted or weakened, no
`_check` gaining an `or`, no seed count reduced.** Every removed line I traced
was replaced by a stronger one in the same commit. **No finding in section 2 —
the fourth clean week.**

**The ratchets were also respected under growth.** The registry went 234 → 238
today (`a4d9c92`, LG.03–LG.06) and `UNREACHABLE_BASELINE` was raised 91 → 94 **in
that same commit** with a named justification in the growth log — the
`GEN.02-09` shape, three specs deliberately blocked behind `LG.03`. Live count
is 94 against a floor of 94: at floor, not above it. `GOAL_UNRUNNABLE_BASELINE`
stayed at `{DP.02, DP.03, LC.04}` while the live reading is 7 — the baseline was
**not** raised to swallow the new red, which is the correct behaviour and the one
the tool's own docstring warns is most often violated.

Now the findings, ranked by damage.

---

## 1. `D21`'s default was about to demote the two rows that release a 16.18-hour dispatch into next week's fresh GPU quota — and this organ wrote that clause, then repaired the wrong half of it yesterday. (NARROWED, in `DECISIONS_NEEDED.md`, this audit)

The 70th audit caught that `D21`'s `decide_by` (2026-09-11) fell five days
after the Sunday its default commands, and moved it to 2026-09-05. It repaired
the **clock**. It did not re-read the **sentence the clock was attached to**:

> the 2026-09-06 FULL Review takes the W1 design as the FIRST item on its
> docket, **ahead of the two `d10-*` gate rows** and ahead of Part 2

**The Review had already published a different order for that same sitting,
with a priced reason, and no organ read it back.** `docs/PROGRESS.md`
`FOR THE OWNER` item 3, written this morning:

> Order: the two `d10-*` gate rows first (**cheap, and they release a 16 h
> dispatch into W36's 30 free hours**), then `w0-too-shallow` …

**The price, from the ledger rather than from the prose.** The "16 h dispatch"
is `D1.0`: `duration_s` **58236.9 s = 16.18 h**, spent across three Kaggle jobs
on 2026-09-01, returned **VOID**. `D1.0` is the repair path for `T2.01`, which
`run blocked` ranks first in the project — **frees 35 / blocks 38**. The two
`d10-*` rows are the gate redesign that re-dispatch waits on. `W36` opens
2026-09-06 00:00 with 30 free Kaggle hours. The Review's measured disposal
capacity is **1 dated row per cycle**.

So the default, firing by owner silence on 09-06, would have spent that single
row on the design item and pushed the gate rows behind it — into the one week
whose quota is fresh. `D21`'s own text asserts it *"moves no threshold, weakens
no control, widens nothing the project may take"*. All three are true. None of
them is this cost, and the cost was nowhere in the entry. It also justified
itself with *"The Review already owns the ordering of its own docket"*, which is
the argument **against** overriding the order the Review published.

**Repaired under the shrink-only rule.** The clause `ahead of the two d10-* gate
rows` is **STRUCK**; W1 goes first among design items and ahead of Part 2, the
gate rows keep the head of the docket. Strictly less is taken; nothing is added;
`decide_by` unmoved at 2026-09-05. The Review's item 3 is quoted verbatim into
`D21` as the evidence, which also **routes it** — `decisions.py` was reporting
it as `UNROUTED-OWNER-ASK` and the ratchet moved **2/3 → 1/3** on this edit.

**The generalisable lesson, and it is why this class keeps recurring:** the 70th
audit found a defect in a field of its own entry and repaired that field. A
default is not its `decide_by`; it is an **action with a cost**, and an audit
that repairs the clock without re-pricing the action has fixed the part an
instrument could see and left the part only a reader can.

---

## 2. Nineteen consecutive iterations, all `rc=0`, PASS delta ZERO — and the builder is not the reason. The board is empty and the desk that could refill it disposed of one row in seven days.

From `/data/jack-logs/ladder.log`, verified line by line:

| window | iterations | `rc=0` | PASS delta |
|---|---|---|---|
| 2026-09-04 00:07 → 18:22 UTC | **19** | 19 | **102 → 102, ZERO** |
| trailing 24 h (09-03 18:30 →) | 27 starts, 24 ends | 24 | 98 → 102, +4 |

Seven `PACING:` strings appear in the window and **all seven are quoted prose
inside journal lines reporting zero skips** — I checked, because the Review
reports "zero `PACING:` skips, last skip 08-29" and a contradiction there would
have been serious. There is none. The loop is alive, on cadence, and honest.

**What it is doing instead is the finding.** `run review-queue`, live:

    36 routed · 32 OPEN · 2 HELD · 2 ACTED · 0 DISPOSITIONED · 0 DECLINED
    34 live rows · oldest live 11 d · consumer last ran 0 d ago
    trailing 7 d: arrived 31 (4.43/cycle) · disposed 1 (0.14/cycle) · designed 0
    drain UNBOUNDED — net +30 over the window
    0 violations

**Zero violations is correct and it is the problem.** Every class the file owns
fires on a promise being *broken*; a desk that cannot keep up has broken none
yet. `coverage` says the same thing from the other end: **0 FRESH dispatches at
any of the seven cost classes**, and all four non-fillable classes name the same
cause in the tool's own words — *"the repair is a REDESIGN"*. Of the nine specs
whose dependencies all PASS, eight are parked or pilot-blocked behind the Review
and one (`HR.1`) is held by `D19`. **None waits on the builder. None waits on
compute.**

Both organs have this right and neither is under-reporting it. The Review named
it as its own finding and put it on the owner's desk; the overseer lifted it to
`D22` this morning. I am recording it as unmoved, twelve hours later, and adding
the number the pile chart now makes unavoidable: **six rows share 2026-09-06
against a capacity of one.** Five of those six are scheduled to break.

---

## 3. Compute honesty: 84% of this week's productive GPU spend bought one VOID, that certificate has since gone stale, and 10.80 free hours expire in five hours.

`experiments/gpu_budget.json`, week `2026-W35`, cross-referenced against every
`gpu_job_id` in the ledger:

| hours | job | ledger outcome |
|---|---|---|
| **16.18** | 3 jobs, one `D1.0` attempt | **VOID** — "run did not test the claim" |
| 1.005 | `jack-ladder-1788070133` | `T2.14` **PASS** |
| 0.304 | `jack-ladder-1788293396` | `UB.10` **VOID** |
| 0.106 | 2 jobs | `T1.09`, `T1.10` **PASS** |
| 1.62 | 5 jobs | **no ledger row — pilots**, legitimate by design |
| **19.20 spent · 10.80 remaining · expires 2026-09-06 00:00 UTC** | | |

Two honest readings and I will not collapse them. **The 16.18 h was not waste in
the sense section 5 means** — it was one pre-registered run that returned VOID,
which is a measurement, and the Review has correctly prohibited an unchanged
re-dispatch as a seed-lottery redraw. **But that certificate is now STALE**:
`run status` reports `D1.0` ran on `870b3c77` and the file at HEAD is
`ab4e3bbd`. The most expensive single run this project has ever bought no longer
describes the code, and its repair is the `d10-*` pair — the rows finding 1 was
about to demote.

The 10.80 expiring hours are a deliberate, correct call by the Review
(*"inventory, not uptime"* — manufacturing a dispatch to spend a dying quota is
the failure mode). I endorse it and note the shape of the week: **the project's
free GPU allocation produced 3 PASSes for 1.11 h, one VOID for 16.18 h, and will
retire 10.80 h unspent.**

One more compute fact, unflagged anywhere: **the loop has refused Fable and run
on Opus for 14 consecutive iterations** (06:07 → 18:07), because `week:Fable`
sits at 99–100% until the 09-07 reset. That is `D14` option (b) working as
designed. It also means the most expensive model available has been spending an
hour a slot on a board with nothing to dispatch. `D15` — the decision that would
put a pace check on exactly this — has `decide_by` **2026-09-05**, tomorrow.

---

## 4. Four constitutional citations in GOAL.md went unrunnable, correctly reported, correctly not baselined away.

`coverage` rc=2 (standing). `CITED-BUT-UNRUNNABLE` is now **7**, of which **4
are NEW**: `GEN.02`, `GEN.03`, `GEN.06`, `GEN.09` — all `welded<-LC.07`, which
became PILOT-BLOCKED on 09-01. These are GOAL.md's own "OTHER MINDS" and
"MORE WORLDS" expansions; the constitution's present tense about them is now
false.

**No violation here, and I want that stated as plainly as the finding.**
`GOAL_UNRUNNABLE_BASELINE` was left at 3 entries rather than raised to 7 — the
red stands. The row `goal-cites-four-specs-that-resolve-to-corpses` is routed,
**DUE 2026-09-10**. The instrument worked and the loop did not paper over it.
I record it because it is now on a desk that disposes of 0.14 rows per cycle.

Standing reds, all unmoved and all correctly ratcheted, listed so a blessed red
cannot go quiet: **4 CLAIM-DEAD** commitments (smell, balance, shelter/building,
thermal — every claim spec parked or foreclosed); **3** PARK-ON-AN-UNREACHABLE-
RELEASE pairs; **3** UNFALSIFIABLE champion seats (ASR, Speaker ID, Language
grounding); **2** UNVERIFIED VERDICTS (Learning core held BY VERDICT on
`LC.03=VOID`; World held BY VERDICT with no deciding run named); **3** TRIGGER
DEBT seats. `champions --check` now reports **0 phantom arenas**, down from the
8 `ARENA-MISSING` my brief describes — and it shrank by **registering** `W.1`–
`W.8` and the `LG.*` family, never by deleting an arena reference. That is the
ratchet repaired in the only legal direction.

---

## 5. Minor: the `UNREACHABLE_BASELINE` growth log is not in the order it declares.

`experiments/coverage.py:914` instructs *"append a line per raise, newest
first"*. The entries read **90, 89, 90, 91, 92, 85, 91, 94** — neither
newest-first nor chronological, with the seed entry (`85 @ 2026-09-01`) buried
sixth of eight. Every individual entry is honest and justified; I checked all
eight against their commits. But this log exists so a human can audit whether a
floor was quietly re-based, and a log you cannot read in order is the one place
a re-base would hide. Cheap to fix, so it should be.

---

## 6. Drift: none found, in either direction I can check.

Everything the builder touched in the last day traces to a GOAL.md sentence:
`LG.03`–`LG.06` and `LANGUAGE_GROUNDING.md` → *"he learns words the way every
child does: by hearing them used while things happen"*; `OWNERS_HANDS.md` and
the `SO.06`–`SO.09` drafts → *"their hands may leave things in his world…
Never puppeteering"*, which is the one clause of that paragraph with no spec;
`T0.21`/`T0.28`/`T0.33`/`T0.34` re-buys → *"protects the honesty of watching"*.
Both research passes were ordered by the Review's `FOR THE BUILDER` block. No
drift.

**The converse is where the honest answer hurts.** Of 102 PASSes, **33 are
`T0.*` — the measurement harness itself** (32%), and 13 are `T1.*` primitives.
The two claims GOAL.md leans hardest on read: **curiosity 2 PASS of 12 specs**,
**one brain / unison 1 PASS of 25 specs**. Nine further commitments have live
claim specs and nothing passing — touch, tool use, told world, proprioception,
death & retry, plasticity, sleep, hunger/thirst, fast/slow.

---

## 7 & 8. Decision and bakeoff hygiene.

`decisions --check`: **0 MEANS-ESCALATED, 0 UNDECLARED, 0 OVERDUE.** No fork a
measurement could settle is sitting on the owner's desk — the `D1` disease is
absent. Ten armed, three firing within four days (`D15`, `D16`, `D21` all
`decide_by` 2026-09-05). `D22`'s `DEFAULT-ACTION-EXPIRED` is the
over-approximation the tool's own docstring predicts by name (*"a default may
name a past date as provenance"*) — its 09-05 refers to `D15`'s clock, not to an
action of its own. Baselined, ratcheted, correctly not tuned away. **Not a
finding.**

`DECISIONS_RESOLVED.md`: no decision made without a learning gate, no winner
chosen inside a noise margin. The one VOID-as-verdict case — `D10` seating
wm-latent on `LC.03`'s VOID — is declared on its face with the single-arm caveat
and is exactly what `champions --check` counts in UNVERIFIED VERDICTS. Declared,
counted, unmoved. No new hygiene finding.

---

## FOR THE BUILDER

1. **`experiments/coverage.py:914` — sort the `UNREACHABLE_BASELINE` growth log
   newest-first as its own header instructs**, or change the header to say
   "append-only, unordered" and explain why an auditor should accept that. Do
   not renumber, reword or re-justify any entry; move lines only. Finding 5.
2. **`D21`'s default is narrowed as of this audit** (`DECISIONS_NEEDED.md`, the
   AMENDMENT block above `DECIDE: D21`). When you fire `D21`/`D15`/`D16` in the
   00:xx–05:xx slot on 09-06, fire the **amended** text: the two `d10-*` gate
   rows keep the head of Sunday's docket. Write that onto the queue rows, not
   into prose — this is the 70th audit's own B3 instruction and it still holds.
3. **Do not re-dispatch `D1.0`.** Unchanged. Its 16.18 h certificate is stale
   *because* the gate redesign is owed; running it before 09-06 buys a redraw at
   the price of half of W36's free quota. Finding 3.
4. **Standing prohibitions, unchanged and re-verified this audit:** `HR.1`–
   `HR.4` stay `D19`-held to 09-14, no corpus fetch; `HR.6` blocked behind
   `HR.5`; `LF.01` attempt 2 waits for the 09-09 design; no third increment of
   the CPU accountant; do not re-stagger the 09-06 docket a fourth time; let
   `W35`'s 10.80 GPU-hours expire.
5. **`PROGRESS.md` `FOR THE OWNER` item 4 is still `UNROUTED-OWNER-ASK`** and it
   is a pure liveness report with nothing to rule on. The repair is the Review's
   `NO-DECISION:` line, already in `scripts/review_prompt.md` — **not yours, and
   not mine.** Flagged so it is not mistaken for work either of us owes.

## FOR THE OWNER

1. **`D21`, `D15` and `D16` all fire on 2026-09-06 if you do not rule tomorrow
   (2026-09-05).** `D21`'s default has been narrowed by this audit and now takes
   strictly less: W1 becomes the first *design* item on Sunday's docket, but no
   longer displaces the two cheap gate rows that release a 16.18-hour dispatch
   into W36's 30 fresh GPU-hours. **If you want W1 ahead of those gate rows,
   that is a ruling, not a silence** — say so and it is so. Nothing in GOAL.md is
   touched by any of the three defaults.

2. **`D22` is the only entry on your desk whose cost is compounding, and it is
   the same fork the Review named this morning.** The measured state twelve
   hours on: 34 live queue rows, **+30 net over seven days**, one row disposed,
   all nine startable specs behind that desk, zero fresh dispatches anywhere,
   and nineteen consecutive builder iterations that bought no PASS. Its default
   — *(i) the rule stands* — is the only legal one, because (iii) widens what
   the builder may do and a default may not widen. **So silence here does not
   resolve this; it entrenches it, at roughly 17 further net rows by 09-08.**
   My own view, on the record because I am one of the two ratifying organs the
   proposal leans on: the safeguard it depends on is my §2 duty, it ran again
   today over seven days of spec and test diffs and found **nothing** — but it
   has only ever audited drift the builder produced *incidentally*, and under
   (iii) it would audit redesigns the builder authored *on purpose*. That is a
   different adversary and I cannot tell you from a tool whether the guard holds
   against it.

3. **The honest answer to section 8: no. We are not closer to a curious humanoid
   that climbs the ladder than we were yesterday — we are closer to a longer
   list of green ticks, and today not even that.** The ledger has not moved in
   nineteen hours. Thirty-three of the 102 PASSes are the measuring apparatus.
   Curiosity reads **2 of 12**, unison reads **1 of 25**, and four of your own
   constitutional commitments — smell, balance, shelter, thermal — have no
   runnable falsifiable claim at all. What the last three days genuinely bought
   is real and I will not undersell it: he can say *"I'm cold"* and be right, he
   is the same creature whether or not he is watched, and a life of his survives
   being killed. Those are three true things about Jack. But every instrument in
   this repo now points at the same place from a different direction — the
   world, and the desk that must redesign it. The builder has 24 slots a day and
   nothing it is permitted to build.
