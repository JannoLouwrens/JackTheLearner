# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Findings are ranked by how much damage they do to the
> trustworthiness of the ledger, not by how interesting they are.

**2026-09-30 06:37–07:0x UTC — the 131st audit.** **Twenty-four hours after the
130th, not six.** The three scheduled sittings in between (`12:37`, `18:37`,
`00:37`) were each **paced out** — that is RANK 2 and it is why this window is a
day wide. The window is the builder's slots `07:07` (2026-09-29) through
`06:07` (today) and demonstrated moved **107 → 107**.

Instrument exit codes, re-derived this sitting from source and not inherited:
`coverage` **2**, `decisions --check` **1**, `champions --check` **0**,
`run status` **2**, `run review-queue` **2**.

Ratchet delta, quoted as a DELTA before anything is recorded, from `run status`'s
own SLOT LINE: **4 MOVED** (`review_queue_net_arrivals` 31 → 27,
`review_queue_piled_on` 3 → 2, `review_queue_violation_forms`
`{HOLD-ON-A-RESOLVED-BLOCKER:9}` → `{HOLD-ON-A-RESOLVED-BLOCKER:9, OVERDUE:6}`,
`review_queue_violations` 9 → 15); no counter refused to compute; floors
**3 ABOVE** (`decisions_default_action_expired`, `pass_on_dead_dependency`,
`unreachable`), 0 BELOW, 0 UNVERIFIED. *(On the re-run immediately before this
commit the same line read `review_queue_violations` 9 → **10** and
`review_queue_violation_forms` `{HOLD:8, OVERDUE:2}` — the Review desk is
disposing rows as I write; the floors and the exit codes did not move.)*

**DISCLOSURE — the Review DAILY is sitting CONCURRENTLY and its numbers moved
under me mid-audit**, exactly as at the 130th; the collision is `37 */6` against
`37 6`, 100 % on every morning sitting. It committed `773a52c` (*"t211 ruling
stamped ACTED"*) at `06:40` and `53b6802` (*"the world-edit price corrected at
all three LIVE sites"*) at `06:42` while I was reading, and the tree carried
`M docs/REVIEW_QUEUE.md` throughout. The queue read **16 = 7 OVERDUE + 9 HOLD**
at my open, **15 = 6 + 9** twenty minutes later, and **11 = 3 + 8** on the
re-run immediately before this commit — the desk has taken its OVERDUE class
from 7 to 3 and released one hold in the time I have been reading. **The queue
numbers below are the FIRST reading unless marked; I quote all three rather
than one, and they were falling while I wrote them.** I committed by name.

**What I did NOT do, named rather than omitted.** I re-ran no spec (the
no-dry-run rule). I armed no decision: `decisions_undeclared` reads **0** —
there is nothing armable on the register — and manufacturing an entry to satisfy
the per-audit arming quota is the disease the quota exists to prevent. I did
**not** restart, resume or unpause the builder loop: `.usage-resumed` is the
owner's file and writing it is outside what this organ may do. I fired no
default.

---

## VERDICT: DRIFTING — and for the first time in nine audits the builder is not healthy. It has been DARK for 23 consecutive hourly slots on a weekly meter of which **74 % was spent by something that is not this project**; the pace line does not release it until **Saturday 2026-10-03 ~23:00 UTC**, which is after this week's 30 free Kaggle GPU-hours expire; the project's own current-state page still says *"Builder health: 0 dark slots"*; and the reporting channel that an armed default installed for exactly this event **did not report, because the organ that carries it died `rc=124` before reaching its own last line.**

---

## RANK 1 — the builder has been dark for 23 hours and every page in this repository says it is healthy. `D30`'s armed default made the dark streak a STANDING owner finding; its only carrier is a paragraph in the Review's prompt; the Review died `rc=124` yesterday morning before rewriting `PROGRESS.md`; so the blackout's first day was reported by nobody. (HIGH — the project's current-state reporting is false about its own liveness)

### The measurement, taken this sitting and quoted from no one

Last builder iteration: **2026-09-29T07:07 → `rc=0` at 07:18** (`e39664c`).
Every slot since is a pacing skip. From `/data/jack-logs/ladder.log`:

    2026-09-29T08:07  'week:all models' 36% at 16% of the week (line 36%) — skipping
    ...
    2026-09-30T06:07  'week:all models' 78% at 29% of the week (line 44%) — skipping
                      22 consecutive dark slot(s); 23.4 h since the last rc=0

**23 consecutive dark slots** at the time of this sitting (`usage_attribution
--line`, run live). The meter rose **36 % → 78 %, forty-two points, during the
twenty-two hours in which this project executed nothing at all.**

The attribution, run live rather than quoted from the frozen skip line:

    of this week's 78 shared point(s): builder 19 (24%), desks 0 (0%),
    both 1 (1%), NOT THIS PROJECT 58 (74%)

**Fifty-eight of seventy-eight points — 74 % — were spent by a consumer that is
not JackTheLearner.** The builder spent 19 and is stopped by 78.

### When it comes back, computed rather than guessed

`pace_gate` is a pure function of the clock:
`allow = 25 + ((90-25)*elapsed + 99)/100`, zero variance
(`scripts/lib_usage.sh`). At `pct = 78` the builder runs again when
`allow ≥ 79`, i.e. at `elapsed ≥ 82 %` of the week. The week reset Monday
2026-09-28 05:00 UTC, so:

> **The builder's next legal slot is 2026-10-03 ~23:07 UTC — Saturday night —
> and only if `week:all models` does not rise one further point.**

This week's **30.0 free Kaggle GPU-hours expire on Saturday 2026-10-03**
(`2026-W39` does not appear in `gpu_budget.json`'s `weeks` map at all — **0.00 h
spent**). The loop wakes at the end of the expiry day or after it. **W39 is the
fourth consecutive week lost, and this one is lost by arithmetic, not by
chance.** `W37` 5.25 h, `W38` 0.92 h, `W39` 0.00 h of 30.

And the hard stop is close: the meter is **12 points from the 90 % ceiling**
that stops *every* organ including this one, having moved 42 points in 22 hours.
It has been flat at 78 for the last three readings, so I do not project a time;
I record the distance.

### Why this is RANK 1 and not merely bad luck

**`pace_gate`'s own docstring is a forecast of this exact event, and the
forecast has now failed on its stated success criterion.** It was written on
2026-08-24 against two prior blackouts:

    W32  loop went dark Fri 08-14 15:07   8.82 of 30 Kaggle GPU-h expired unspent
    W33  loop went dark Fri 08-21 12:07  22.11 of 30 Kaggle GPU-h expired unspent

    "PACE_FLOOR=25 buys the week's opening burst; by Friday the line is ~62%,
     Sunday ~81%, and the loop is still awake when the GPU quota expires."

**This week the loop went dark on WEDNESDAY at 08:07** — a day and a half
earlier than either blackout the fix was built to prevent — for the same
measured cause the docstring names: *"the loop is stopped by consumption it
does not control, and being the only consumer with a gate, it is the one that
starves."*

**And this has happened before, on the record, with a number attached.**
`D30` (2026-09-15, Review DAILY): *"The builder has been dark for 18
consecutive hourly slots on a meter three-quarters of which this project did
not spend, and 26.51 free GPU-hours expire on Saturday with nobody awake to
dispatch them."* Its reading then: `total 37, builder 7 (18%), NOT THIS
PROJECT 28 (75%)`. Today: `total 78, builder 19 (24%), NOT THIS PROJECT 58
(74%)`. **Same disease, same share, 23 slots instead of 18, and 30 hours
expiring instead of 26.51.**

### The part that is an integrity finding rather than a compute finding

`D30` RESOLVED BY ARMED DEFAULT 2026-09-19, default **(v) REPORT THE STREAK,
GATE NOTHING, RELAX NOTHING**:

> *"One paragraph added to `scripts/review_prompt.md` Part 2.5 §4: a builder
> dark streak past 2× its hourly cadence is a STANDING FOR THE OWNER finding,
> counted from `ladder.log` and printed beside the week's GPU-expiry forecast
> … so a blackout is visible on day one with its perishable cost priced in the
> same sentence."*

**Day one has now happened and it was visible to nobody**, because the default's
entire mechanism is a paragraph in ONE organ's prompt, and that organ's
2026-09-29 sitting **died `rc=124` at 06:57 before rewriting `docs/PROGRESS.md`**
(`/data/jack-logs/review.log`; disclosed by the builder at `6422eea`). The
consequence, which is the finding:

> **`docs/PROGRESS.md` — the Review's current-state page, the page the owner
> reads — says today: *"Builder health: 0 dark slots. Fourteen consecutive
> `rc=0` slots; the blackout that peaked at 26 stayed closed. The builder is not
> the constraint."*** It is dated 2026-09-28 and it is 47 hours old. My own
> launcher stamped it STALE at 06:37 this morning (`3fcad58`), which is the
> system working — but the STALE banner says *"this is an older world"*, not
> *"the builder is down and has been for a day"*.

The 130th audit (mine, yesterday 06:52) reported **"0 dark slots, 6 of 6 builder
slots `rc=0`"** and it was *true when written*: the blackout began at 08:07, one
hour and fifteen minutes after that audit committed. Nothing in it is wrong.
**What is wrong is that nothing looked again for twenty-four hours** — see
RANK 2.

### The mechanical hole underneath all of it

I grepped for the number. **`dark_slots` appears nowhere in `experiments/`** —
not in a spec, not in `ratchet_readings.json` (0 matches), not in any exit code.
Its only two writers in this repository are `scripts/usage_attribution.py` and
`scripts/overseer.sh`. So the streak is reported by exactly two channels:

1. **the pacing skip line in `ladder.log`** — written by the organ that is being
   skipped, which the library's own comment already concedes is
   *"the one fault this gate cannot report about itself"*; and
2. **the Review's prose page** — which was dead for this one.

A number with no ratchet key, no floor and no exit code is a number that can go
to 23 without anything going red. Every other standing red in this project
(`unreachable`, `pass_on_dead_dependency`, `live_unauditable_pairs` as of
yesterday) is floored precisely so it cannot. **This one is not.** Repair priced
in FOR THE BUILDER 1.

### One honest limitation of the attribution instrument, recorded not repaired

`usage_attribution` buckets the spans between consecutive marks in
`/data/jack-logs/usage_ledger.jsonl`, and only a *running* organ writes a mark.
The last mark before today was `2026-09-29T07:18 pct=35`. So for the whole
blackout the skip line printed a **frozen** `35 shared point(s) … NOT THIS
PROJECT 15 (42%)` — 22 identical lines — and only became `78 … 58 (74%)` when
the first organ woke at 06:37 today and wrote a mark. This is correct
behaviour (unknown is not zero, and the gap is attributed once it is bounded),
and it is also the reason **the 42-point rise was invisible to every one of the
22 slots that decided to skip on it.** I am not calling this a defect; I am
recording that the instrument cannot see a blackout from inside one.

---

## RANK 2 — three of the four scheduled audits in the last 24 h did not happen. The overseer is paced by the same shared meter it exists to report on, keeps exactly one exempt sitting a day, and therefore degrades from 6-hourly to daily **precisely when something is going wrong**. (MEDIUM-HIGH — oversight cadence, and it is this organ's own blind spot)

From `/data/jack-logs/overseer.log`, verbatim:

    2026-09-29T12:37  PACING: 'week:all models' 46% ... 5 consecutive dark slot(s) — skipping
    2026-09-29T12:37  audit slot paced (D15 clause c, 76th B2 slotting)
    2026-09-29T18:37  PACING: ... 66% ... 11 consecutive dark slot(s) — skipping
    2026-09-30T00:37  PACING: ... 75% ... 17 consecutive dark slot(s) — skipping
    2026-09-30T06:37  audit start — model opus, 452056e

**Each of those three skip lines printed the dark-slot streak — 5, then 11, then
17 — and then declined to run the organ that would have acted on it.** The
exemption is *"the first completed audit at or after the Review's 6:37 slot"*
(`D15` clause c), so during a blackout there is exactly **one** audit a day, and
it lands in the same six-minute window as the Review desk it collides with.

This is not a bug in `D15` and I am not proposing to widen my own exemption —
that would be an organ voting itself more budget, and `D26` already refused the
loosening direction for the builder. It is a **stated structural property that
nobody has written down**: the three organs that could notice a starvation
event are gated on the resource being starved, and the fourth (`regate`, `43
*/2`) is deliberately ungated but only ever re-buys documentation certificates.
`regate` ran on time all night — twelve ticks, every one *"nothing cheap is
stale"*. The only lane that was awake had nothing to look at.

Consequence for this page: my window is 24 h wide, not 6, and **the 107 → 107 in
the header covers a full day, not a morning.**

---

## RANK 3 — the window served no `GOAL.md` sentence at all, because one slot ran in it. (MEDIUM — drift, and the builder is not its cause)

**What the builder worked on in the last day: one slot.** `07:07` on 2026-09-29
(`e39664c`) executed the 130th audit's FOR THE BUILDER 2 —
`UNAUDITABLE_PAIRS_BASELINE = 22`, floored shrink-only and joined to
`ratchet_live` / `ratchet_floors` / the `FLOORED_CLASS` scan and pin. I verified
it at source, not from the commit message: the constant is in
`experiments/protocol.py` beside `audit_supersedes_fail`, live reading **22, AT
floor**. It serves the first principle's fourth clause — *protects the honesty
of watching* — and nothing else. **No slot in this window served a sentence
about the brain, the body or the world**, because there were no other slots.

The 130th's FOR THE BUILDER 1 (stop re-arming the regate forecast chain) was
also adopted, verifiably: the `07:07` commit gave the `06:43` tick one journal
line instead of a headline, and there has been no chain since — because there
has been no slot since.

**The converse, from `coverage` EXIT 2 and unchanged in a day:** **0
commitments with NO declared spec** (floor held), **3 CLAIM-DEAD** (smell,
shelter/building, thermal-kills — every claim spec parked or foreclosed) and
**14 more with live claim specs and nothing passing** — **17 of the owner's
constitutional commitments with zero passing claims**, including *too cold
kills him*, *he builds a shelter*, touch, tool use, proprioception, sleep,
plasticity, fast/slow and the told world. `NO-LIVE-PATH` stands at **6**
distinct commitments/seats; the repair for every one is a **registration**,
never an unpark and never a deletion. `GOAL.md` cites 16 spec ids, **0
dangling**, with `CITED-BUT-UNRUNNABLE` at **7** (`DP.02`, `DP.03`, `GEN.02`,
`GEN.03`, `GEN.06`, `GEN.09`, `LC.04`) — owned by the OPEN queue row
`gen-four-reparented-to-a-decision-that-had-already-closed`, DUE 2026-10-01.

**Creature gate: NONE**, re-derived rather than quoted:
`T6.01 ← T4.05 ← T4.04 ← T2.01 ← T1.08 (FAIL)`. `T1.08`'s pipeline-repair design
is the Review's, `DUE 2026-10-02`. **The builder will not be awake on 10-02 to
execute it.** That is the sentence in this report I would most like to be wrong
about.

---

## The mandated sections, with the findings above not repeated

**1. Integrity of the ledger — CLEAN, re-derived not inherited.** All **107**
standing PASS rows carry a `commit` field (**0 empty**); the **54** distinct
commits behind them all still exist in git (`git cat-file -e` over the distinct
set, **0 missing**); every PASS id resolves in `BY_ID` (254 registered). The
control half: `experiments/verify` **EXIT 0** — no PASS in this ledger declares
a control that its test never called. Ledger totals **PASS 107 / FAIL 32 /
VOID 16 / BLOCKED 1** = 156 rows with a verdict against 254 registered →
**42.1 %**, byte-identical to yesterday, which is what a day with one slot in it
looks like. Standing classes unchanged and reported honestly by `run status`:
2 DIRTY STAMPS (`T6.03`, `PL.02`, both refused with written reasons), 5 UNBACKED
CERTIFICATES (legal, reporting-only), 1 DELIBERATELY-RED GATE (`T0.27`).
`T0.27`'s `live_violations` still reads **3** — the 130th's RANK 1 — and the
130th's repair for the class around it (`live_unauditable_pairs`) is now floored
at 22, verified above.

**2. Thresholds and controls over seven days — NO SILENT LOOSENING FOUND, and
the check is a fresh one.** In my own 24-hour window the answer is trivial and I
verified rather than assumed it: **no spec, test or registry file was edited by
anyone** — the window's only commits are one ratchet/protocol change
(`e39664c`), two mechanical regate sweeps, the Review's doc acts, and my own
launcher's STALE stamp. Over the full seven days I re-ran the loosening grep
(`_MIN`/`_MAX`/`_FLOOR`/`_CEIL`/seed counts/comparison operators/` or `) across
`registry.py`, `registry_expansion.py` and `experiments/tests/`, and I
deliberately spot-checked **the one diff in the window that has the exact shape
my mandate names as most serious and that neither the 129th nor the 130th
examined**: `f047060` (2026-09-27), which adds an **`or` to a `_check`-side
expression** —

    - chaos = int(occ >= CHAOS_OCC and ratio >= CHAOS_RATIO)
    + chaos = int(occ >= CHAOS_OCC and (not defined or ratio >= CHAOS_RATIO))

**It is a strengthening, and I confirmed the direction in source rather than
from the commit message** (`lt_03_ladder_test.py:829-830`): `chaos` marks an arm
as self-chaos-CONTAMINATED, so `not defined → chaos = 1` moves a candidate from
`live` to VOID. It cannot rescue a verdict. The companion edit
`< REWARD_RATIO_MIN` → `not (>= REWARD_RATIO_MIN)` closes a NaN-passes-silently
hole in the same direction. `REWARD_RATIO_MIN 2.0`, `CHAOS_RATIO 2.0`,
`CHAOS_OCC 3.0` all byte-unchanged. **And the builder priced its own cost
honestly in the same commit** — `LT.02` counterfactual `PASS → FAIL`,
`unreachable 95 → 96` — which is where the standing above-floor `unreachable`
reading comes from. That is the behaviour this section exists to find the
absence of. **No findings in section 2.** Stated plainly because it is true.

**3. Drift from the goal.** Covered in RANK 3.

**4. Builder liveness — this is RANK 1 and the answer is NO.** 1 iteration start
in 24 h against a cadence of 24; **1 ended `rc=0`** (100 % of those that ran);
**23 consecutive dark slots**; PASS delta **107 → 107**. `lost_iterations.log`
is 0 bytes and no slot failed — **this is not a crash, a credit exhaustion or a
paused loop nobody resumed; it is a gate doing exactly what it was written to
do on a meter this project does not control.** The distinction matters for the
repair: there is nothing to restart. The last slot that ran read `run next` at
0 fresh of 51 — the seventeenth consecutive verified-empty board — and
manufactured nothing, which is now the ninth consecutive audit at which I can
say so.

**5. Compute honesty — the worst reading this organ has recorded.**
Re-derived from `gpu_budget.json`'s own per-week records: **`2026-W39` does not
appear in the `weeks` map at all — zero jobs, 0.00 h — against 30.0 free Kaggle
GPU-hours expiring Saturday 2026-10-03.** With `W37` at 5.25 h and `W38` at
0.92 h this is the **fourth consecutive week substantially lost**, and the first
one lost to a *computable* cause: the pace line does not release the builder
until Saturday ~23:00 (RANK 1). Every cost class in `coverage` reads `NOT
FILLABLE`; **3 are empty with no path in** (`cpu<1min`, `gpu<20min`, `gpu<8h`).
Queue depth dispatchable today: **6, of which 6 VOID → 0 FRESH dispatches.**
Both live routes run through `T1.08` (FAIL), whose repair design is due 10-02 —
a date on which no builder slot will run. **No dispatch has been manufactured
and none should be; there is no organ awake to manufacture one.** Standing waste
unchanged: `gpu_hours_no_verdict` TOTAL **48.42 h**, `D1.0` alone holding
**33.78 h across 2 attempts and 0 verdicts**; `gpu_unattributed_jobs = 21`, AT
floor.

**6. Stuck decisions.** `decisions --check` EXIT 1. **No `MEANS-ESCALATED`** —
nothing a measurement could settle sits on the owner's desk; the `D1` disease is
absent. **`decisions_undeclared` = 0, at floor** — nothing was armable and I
armed nothing rather than invent an entry. One armed: `D37`, due 10-04, default
(iii) HOLD, `costs 0 specs` and the entry says so itself. Five not armed: `D33`
(CONDUCT-DESK, stale **7 d**), `D35` (CONDUCT-DESK, stale **6 d**), `D38`
(CONDUCT-DESK, due 10-04); `D33` additionally `DEFAULT-ACTION-EXPIRED` — the one
class above floor, **1 vs 0, unchanged since 2026-09-23**; `D37` flagged
`CONDUCT-MISFILED?`. `decisions_unrouted_owner_ask` and
`decisions_vanished_owner_ask` both **0**, at floor — the `D15` disease this
organ measured on 2026-08-29 stays closed, and `PROGRESS.md`'s single owner-ask
is correctly attributed to `D22`/`D33`. `decisions_firing_diff = 0`: **no owner
decision was quietly acted on without being recorded.** One decision was
resolved-by-default and has now been tested by events and found not to work —
`D30` — and because it is resolved rather than open it appears in no instrument;
I have appended the evidence to its entry rather than reopening it (see FOR THE
OWNER 1).

**7. Bakeoff hygiene — `champions --check` EXIT 0, every class at floor, none
new.** **Learning core** held `BY VERDICT` off `LC.03` (a VOID) with
`VERDICT-IS-A-VOID` **and** `TRIGGER-UNREACHABLE` — that is this section's *"a
VOID treated as a verdict"*, printed rather than hidden, and `D37` is the live
entry on it. Also standing: **World** `VERDICT-UNDECLARED`/`TRIGGER-UNDECLARED`;
**Fast/slow coupling** `ARENA-UNREACHABLE`/`TRIGGER-UNREACHABLE`; 2 `NO-ARENA`
(ASR, Speaker ID); 2 `UNCONTESTED` (Vision encoder, PLASTIC ONLY, both turning
on the already-dated `PL.02`). **`ARENA-MISSING` remains 0** — my own standing
prompt still says 8, and those seats were repaired by REGISTERING specs, never
by deleting arena references. Two seats moved yesterday under the Review's own
hand (`1ed5f1f`): Language routing's basis NARROWED, Person model ruled VACANT
with a tie-break owed — both legal, both recorded, neither a winner chosen
inside a noise margin. On *"a winner chosen inside the noise margin"* the
standing answer is unchanged: `T4.06`'s `loss_reweight` is certified at
**+0.0187 = 6.9 % of the incumbent's own 0.2699 seed spread, with one seed of
three regressing**, disclosed on the certificate (`f7900b5`), the adoption the
Review's, and the field watch's week-9 §6 attacking the same statistic from the
other side while its draft stays sealed `rc=124`. Not mine to rule; recorded so
it is not lost.

**8. THE HONEST SUMMARY — are we closer to a curious humanoid that climbs the
ladder than we were yesterday?**

**No, and this time something did go wrong.**

For eight consecutive audits the answer to this question was *"no, and nobody
did anything wrong"* — a healthy builder verifying an empty board, hour after
hour, while the two design debts that would refill it sat at a desk that sits
for twenty minutes a day. That report was accurate and it was also, I now think,
sedating. It trained three organs to read *"0 dark slots, `rc=0`, nothing
manufactured"* as the shape of a good day, and the shape of a good day and the
shape of a stopped machine are, on a page, nearly identical: both produce no
ledger rows.

Yesterday morning the machine stopped. Not crashed — **stopped**, politely, by
its own pacing gate, on a shared weekly meter of which it had spent 24 % and
someone else had spent 74 %. It will not start again until Saturday night, by an
arithmetic anyone can check in one line. Thirty free GPU-hours expire before
then, for the fourth week running. The `T1.08` design that would refill the
board is due on Thursday to a builder that will not wake until Saturday. And
during the twenty-three hours this was happening: the Review's current-state
page said *"0 dark slots … the builder is not the constraint"*; three scheduled
audits read the streak off their own skip line and declined to run; and the one
number that describes the event reaches no ratchet, no floor and no exit code
anywhere in `experiments/`.

That is the finding, and it is not about compute. **This project can measure a
6.9 %-of-seed-spread margin on a fusion bakeoff, can date twenty-two historical
`impl_sha` gaps to the hour, and floored a shrink-only constant yesterday so a
coverage channel could never move silently again — and it could not tell itself
that its builder had been switched off for a day.** The instruments are
extraordinary and they are all pointed inward at the ladder. The one thing none
of them watches is whether anybody is climbing it.

Set against the ladder, nothing moved: 107 → 107. Seventeen of the owner's
constitutional commitments still have no passing claim. Eight live rows still
read `HOLD-ON-A-RESOLVED-BLOCKER` (nine at my open) behind a world whose
authorship was formally DECLINED and has no owner. Of 31 resolved decisions,
30 were resolved by armed
default and **one, ever, by the owner** (`D19`, 2026-09-17) — and the one
default that was armed against *this* failure was tested this week and did not
fire, because it was written as a paragraph in a prompt instead of a number in
a checker.

The one thing moving, again, is the Review desk: it stamped `t211` ACTED at
06:40 (un-parking `T2.11` after 32 days) and corrected the world-edit price at
three live sites at 06:42, while I wrote this. It is the only organ in the
project that is not gated on the meter that has stopped the other one.

---

## FOR THE BUILDER

Read this when you wake. **Both items are cheap and neither is urgent relative
to whatever the Review has put on your board** — if `1^15`'s six DISPOSITIONED
rows or the `T1.08` design are live when you read this, they outrank both of
these, and say so rather than doing these first.

1. **Give the dark-slot streak a ratchet key, a floor and an exit code.**
   RANK 1's mechanical hole, and it is the smallest possible version of the
   repair. `dark_slots` appears **nowhere in `experiments/`** — I grepped: 0
   hits in `*.py`, 0 in `ratchet_readings.json`, no exit code. Its only writers
   are `scripts/usage_attribution.py` and `scripts/overseer.sh`, i.e. the organs
   that are themselves being skipped, plus a Review prose page that was dead for
   this one. Concretely: compute the consecutive-dark-slot count from
   `/data/jack-logs/ladder.log` in the same idiom as
   `UNAUDITABLE_PAIRS_BASELINE` (which you shipped yesterday) — a live reading
   in `ratchet_live`, a **floor of 0**, joined to `ratchet_floors`, the
   `FLOORED_CLASS` scan and the self-check pin — so a streak past `2×` the
   hourly cadence turns `run status` red and stays red until it clears.
   **Gate nothing else, relax nothing, and do not touch `PACE_FLOOR`,
   `PACE_CAP`, the pace line or the 90 % stop** — those are `D26`/`D30` and the
   owner's, and a builder that widens its own budget is the one move this repair
   must not be able to become. This is REPORTING, in the same spirit as `D30`'s
   fired default (v), moved out of a prompt and into a checker where it cannot
   die with a sitting.

   **Disclose the `D35` clause-2 tension in the commit, as you did for
   `UNAUDITABLE_PAIRS_BASELINE` at `e39664c`.** This is a floor on an existing
   checker for a truthfulness repair — the same class the owner is being asked
   about in FOR THE OWNER 3(b) — and it should not ship pretending the freeze
   does not touch it.

2. **When you wake, your first journal line should be the outage, measured, not
   the board.** You will find `run next` empty and `rc=0` available and the
   temptation will be to report a normal slot. The honest line is *"this is the
   first slot in N hours; the pace line held me out from 2026-09-29T08:07; the
   meter read 78 % of which this project spent 24 %; W39's 30 GPU-hours expired
   unspent."* Pull the numbers from `ladder.log` and
   `usage_attribution.py --line` rather than from this page — by then it will be
   an older world.

## FOR THE OWNER

**1. PERISHABLE AND NEW — your builder has been switched off for a day, it will
not wake until Saturday night, and 74 % of the meter that switched it off was
not spent by this project. One file you own reverses it today.** This is
`D30`'s question, asked a second time with worse numbers; I have appended the
measurement to `D30`'s entry in `docs/DECISIONS_NEEDED.md` as an EVIDENCE
ADDENDUM rather than reopening a resolved decision or firing anything.

- **Measured:** 23 consecutive dark slots; last `rc=0` 2026-09-29T07:18;
  `week:all models` **78 %**; `usage_attribution --line`: *builder 19 (24 %),
  desks 0, both 1, **NOT THIS PROJECT 58 (74 %)***.
- **Computed, not estimated:** `allow = 25 + ((90-25)·elapsed + 99)/100`, so at
  78 % the loop resumes at `elapsed ≥ 82 %` = **2026-10-03 ~23:07 UTC**, and
  only if the meter does not rise one further point. It is **12 points from the
  90 % hard stop** that would also stop this organ and the Review.
- **The perishable cost:** `2026-W39` has **no entry at all** in
  `gpu_budget.json` — 0.00 of 30.0 free Kaggle GPU-hours, expiring Saturday.
  Fourth consecutive week (`W37` 5.25 h, `W38` 0.92 h).
- **The two things you can do, both already on the record and neither available
  to any organ here.** (i) Write `.usage-resumed` with a ceiling and an expiry —
  it is the only thing that lifts the gate, it is yours alone, and it expires at
  the weekly reset by design so it cannot become a deletion of the limit. (ii)
  Rule `D30`'s option (i) — *pace against this project's OWN attributed spend
  rather than the shared total*. The Review recommended (i) on 2026-09-15 and it
  remains quoted and open at no cost; it did not fire as a default **for the
  correct reason** — a default may not loosen a gate (`D26`'s reasoning) — which
  means it can only ever reach you by your hand. **Two blackouts have now been
  bought with that rule.**
- **What I am not asking for.** I am not asking to widen my own pacing
  exemption, and I did not take one. RANK 2 records that three of four audits in
  this window were paced out; the honest repair for that is your ruling above,
  not an organ voting itself more budget.

**2. `D33` — who authors the W1 world edit? Seven days past its deadline, and
neither organ can clear the red.** Unchanged from the 126th through 130th
audits, repeated because nine live rows are behind it.
`decisions_default_action_expired` reads **1** against floor **0**, unchanged
since 2026-09-23. `D33`'s own default is *"re-date once more, to 2026-09-23"* —
an act in the past on the earliest day it could fire — and the Review's formal
**DECLINE** of the W1 authorship (the first `DECLINED` in 114 routed rows)
forecloses re-dating entirely. The instrument names two legal repairs — shorten
`decide_by` (a deadline may tighten, never lengthen) or declare whose date it is
with `(CLOCK: <whose>)` — and **neither is available to either organ**; `D13`
bars me from editing the register's rulings. **Nine** live rows read
`HOLD-ON-A-RESOLVED-BLOCKER` at my open and **eight** on my last re-run, two of
them (`ne01-occlusion-knife-edge`, `water-apply-phantom-force`) **37 days old
with no `DUE:` at all**. The Review
recommends option (ii) — the builder drafts under Review — and says it may not
carve that exception out of `D22`, which is your ruling. A stop-rule fires
2026-10-09. **New this morning:** the Review corrected the world-edit's price at
all three live sites (`53b6802`) and records that *"D33's owner ruling was
priced 60 % low"* — so the cost figure in front of you when you last read this
entry was wrong in the direction that made it look cheaper.

**3. `D35`'s repairs, unchanged from the 116th, 125th–130th audits, and clause
(b) now blocks a third thing.** (a) a reachable release condition — `T6.01` sits
behind `T4.05 ← T4.04 ← T2.01 ← T1.08` (FAIL), so the freeze cannot end by any
act available to anyone; (b) a clause-2 exemption for truthfulness repairs and
floors on EXISTING checkers — this already covered the unfloored
`no_control_specs` and yesterday's `live_unauditable_pairs`, and **FOR THE
BUILDER 1 above is now a third instance of exactly that class**; (c) confirm the
freeze is meant to be unbounded; (d) say whether clause 2's *"audit organ,
checker or ratchet"* covers autonomous **actors**. `D35` has been
`CONDUCT-DESK`-stale for 6 days.

**4. Your headline number is still scheduled to fall by one on 2026-10-04, by
cron, and it is an artefact.** `T0.28` requires at least one *armed* decision to
exist in `DECISIONS_NEEDED.md` and currently passes on `live_armed = 1.0` from
the single entry `D37`, whose `decide_by` is 2026-10-04. Ruling `D37` subtracts
one from the demonstrated count within about two hours. That is the
certificate's shape, not a loss; the forecast is written on the queue row. (My
append in item 1 re-stales `T0.28`'s certificate; `regate` is deliberately
ungated on usage, ran twelve times on schedule through the blackout, and re-buys
it at the next `:43` tick at a cost of ~47 s.)

**5. NO-DECISION, liveness — and this line has changed for the first time in a
fortnight.** Builder: **23 consecutive dark slots, 0 failed slots, 0 bytes in
`lost_iterations.log`** — stopped, not broken. Overseer: **1 of 4 scheduled
sittings ran**. Review: sitting now, and its previous sitting died `rc=124`
before rewriting its own page, which is why `docs/PROGRESS.md` is 47 h old and
carries a STALE banner my launcher wrote at 06:37. Field watch: last ran Monday
06:07, also `rc=124`, its draft sealed and deliberately not consumed. `regate`:
alive, on time, twelve ticks, nothing to buy. **Three of five organs have died
or been silenced in the last two days and the ladder did not move.**
