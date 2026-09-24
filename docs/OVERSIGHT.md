# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Findings are ranked by how much damage they do to the
> trustworthiness of the ledger, not by how interesting they are.

**2026-09-24 12:37–13:0x UTC — the 113th audit.** Six hours after the 112th
(06:37). The window is the builder's six live slots 07:1x–12:0x (`5c5146e`,
`7c8fc7b`, `b278f6a`, `c7b6f33`, `c1f093d`, `f25f9f6`, `d8722fb`, `a7712c4`,
`6595e08`, `bf9ac33`, `090b175`) — including the desk's own Tier-0 freeze
(`f25f9f6`, 08:34) — and no Review sitting (the 06:37 DAILY predates this
window; the 09:22 and 12:22 `--retry` polls no-opped, correctly, because the
sitting was held rather than deferred).

---

## VERDICT: INTEGRITY RISK — the top line of the second mandated instrument is a standing order to overturn an owner ruling, and it has been armed for ten days

Say the good part first and with numbers, because it is true and it is most of
the report. **`run verify` re-judges 109 PASS entries: 0 verdicts that no
longer re-derive, 0 gates that ignore their control, 0 that could not be
replayed, 0 that could not be audited.** Every PASS commit still resolves in
git. **Not one threshold moved in any direction in this window** — no file
under `experiments/registry*.py` or `experiments/tests/` was touched at all
between 07:23 and 12:37; the 112th audit cleared the full seven days before
that. Section 2 has nothing to report and I am saying so plainly rather than
inventing a concern.

The risk is not in the ledger. It is in the decision record, and it is this:

**`experiments.decisions --check` prints, as its first and highest-cost line:**

```
D19    costs   3 specs   OVERDUE — DEFAULT IS DUE TO FIRE
       blocks HR.1, HR.2, HR.3, HR.4
       default: NO FETCH. The builder does not download corpora or models to /data ...
```

**The owner answered D19.** The ruling is in `docs/DECISIONS_NEEDED.md`,
verbatim: *"yes may download anything to /data"* — recorded with its own price
paragraph, and explicitly: *"The armed default was NO FETCH; it is superseded,
unfired."* It was acted on. `/data/jack_corpora/librispeech` and
`/data/jack_corpora/vctk` exist on disk. `HR.1` ran three further attempts
against them and recorded two measured FAILs (`467cf1b` 09-18, `623b16d` and
`1f5575e` 09-23), and `registry_expansion.py` says *"Fetched under D19"* in
its own provenance line.

**My charter's instruction for this class is unconditional: *"the owner did not
answer by the date they were given. Fire the default."*** Firing it would
forbid the fetches that already happened, re-block `HR.2`/`HR.3`/`HR.4`, and
retroactively de-venue two honestly-measured FAILs — by instrument, against an
explicit owner decision, in the one class my own page says to act on **without
asking**. Nothing stopped that in ten days except that ten consecutive audits
happened not to pull the trigger, and none of them wrote down why.

That is the whole finding: **not a wrong number, an armed one.**

---

## FINDING 1 — D19 reads OVERDUE because the owner's ruling was filed under a different decision's heading (CRITICAL)

**The mechanism, traced rather than guessed.** `decisions.py:333-345` settles a
decision by scanning its **headers** for `_SETTLED` (`RESOLVED|off your desk|BY
THE CALENDAR`). The D19 ruling has no header of its own. It was appended into
the file at the tail of the **`## D35 — CONDUCT: the Tier-0 freeze`** section
(heading at line 4344), immediately above the retained `## D19` entry at line
4430 — which still carries its original, intact `DECIDE: D19 / default: NO
FETCH / decide_by: 2026-09-14` block at line 4455, preserved deliberately under
the file's own *"Superseded entry retained below, as this file's convention
requires"* rule.

So the parser sees exactly what the page told it to see: an armed default, ten
days past its date, costing three specs. The convention that keeps the record
honest (retain the superseded entry) and the parser that keeps it live
(headers-only settlement) are each correct alone and produce a false reading
together. **No instrument in this repo can see the difference between a
decision nobody answered and one whose answer was filed one heading too high.**

**Damage if fired:** three specs re-blocked, two measured FAILs de-venued, and
an owner ruling reversed by an organ that was following its instructions.
**Damage already done:** none that I can find — I checked the three `HR.1`
attempts and all three ran *after* the ruling and cite it. This is a trap, not
a wound, and it is worth reporting loudly precisely because it is still only a
trap.

**I did not fire it and I did not extend the deadline** — my charter forbids
the second (*"a deadline that moves when it is reached is the deadlock it
replaced"*) and the facts forbid the first. There is a third path neither my
page nor the instrument names: **record that it was ANSWERED.** I have appended
a dated guard to `DECISIONS_NEEDED.md` under D19 so that the next organ to read
that line — the 13:07 builder slot, or tomorrow's 06:37 Review — does not fire
it before the filing repair lands. The repair itself is a documented change and
is routed to the builder below.

---

## FINDING 2 — the ladder has not moved in 25 hours, across 26 clean slots, and not one of 17 ratchets measures that

```
2026-09-23T11:13 .. 2026-09-24T12:10   26 consecutive slots   rc=0   110 -> 110
```

Every one of those iterations ended `rc=0`. Every one re-derived the board and
found it honestly empty. **The builder is behaving correctly and the project is
stationary, and those are the same fact, not two.** The last movement was
`T4.06` at 09-23 11:09 (109 → 110); every ledger commit since has been a `T0.*`
instrument re-buy, which by this repo's own convention *"claims nothing about
Jack."*

**What no organ can see about this.** I enumerated all 17 entries in
`ratchet_readings.json`. Every one counts a **defect class** —
`commitments_uncovered`, `fail_unowned`, `unreachable`, `review_queue_violations`,
`champions_trigger_debt`, `pass_on_dead_dependency`, and eleven more. All are
shrink-only. **Not one measures motion.** `D30`'s dark-slot counter counts slots
that failed to *run*; these slots ran perfectly. So the entire instrument suite
reads green-or-known-red while `demonstrated` is flat for 25 hours and 29 GPU
hours perish, and the builder's own success criterion — *board empty, nothing
manufactured* — is **satisfied by the deadlock**.

This is the coverage lesson in a new place. A missing spec has no id and is
invisible to every instrument; **a ladder that has stopped moving has no defect
class and is invisible the same way.** The only organ that notices is a human
reading a report, every six hours, by hand.

**Credit where it is owed, because this is the honest half:** the desk measured
the same disease from another angle this morning and said it better than I
would have. `f25f9f6`, 08:34: *"there is no Jack. 108 certified parts and no
creature."* Commits over 30 days: **465 docs/scripts, 352 experiment
infrastructure, 258 touching a spec or the registry — under a third about
Jack.** Registry 169 → 249 (+80) against passes 84 → 108 (+24): *"the
denominator is winning."* That diagnosis is correct and it is the reason D35
exists.

---

## FINDING 3 — seven dated promises break together at midnight, no desk can sit before then, and one of them fires a DECLINE on work that is provably finished

**The arithmetic, and it is arithmetic rather than a forecast.** Seven live
rows carry `DUE: 2026-09-24`. The Review's DAILY sitting is `37 6 * * *` —
next at 09-25 06:37, **after** midnight. The `--retry` polls at 15:22 and 18:22
cannot substitute: `scripts/review.sh:24` holds a sitting *"AT MOST ONCE if and
only if it has"* been deferred, and today's 06:37 sitting was **held**
(`/data/jack-logs/review-last-sitting` = `2026-09-24`). The overseer sits at
18:37 and 00:37 and **cannot dispose queue rows.** No organ with the authority
to stamp these rows sits between now and 00:00.

So `review_queue_violations` goes **0 → 7** tonight, against a baseline of 0 set
2026-09-22. `review-queue` currently prints `0 violations / EXIT 0` and flags
`2026-09-24  7  !! AMBER: pile` — it reads *now*, and nothing in it says *"and
nobody sits before these break."*

**The row that matters is `pl02-eye-gate-reads-the-encoder-not-the-eye`**
(REVIEW_QUEUE line 5829). It carries a STOP-RULE in the desk's own words:

> *"if this date breaks too, the row is DECLINED and the finding goes to the
> owner — a promise renewed four times is not a promise."*

**I verified the underlying debt against commits rather than the journal, and
it is fully paid:**

| ordered | delivered | evidence |
|---|---|---|
| spec edit re-aiming the VOID gate to the raw-pixel ridge | ✅ 09-12 | `a4132c8`, `pl_02_reshaping_gain.py` +79/−8, `EYE_RADIUS_R2_MIN` **0.80 unmoved** |
| smoke | ✅ 09-12 | `c150187`, `r2_raw_pixel` **0.924963** vs 0.80 |
| registered run | ✅ ×2 | `PL.02` attempt 2 on the ledger, VOID, arm-attributed (FROZEN voids `learn_ok`) |

So the DECLINE that fires tonight would tell the owner the desk cannot produce
this, when the builder produced it twelve days ago and only the stamp is
missing. **PL.02 is the sole registered falsifier of GOAL.md's PLASTIC-ONLY
decree.** A false DECLINE on that row is the most expensive single
mis-statement available in this file tonight, and it costs one line to prevent.

---

## FINDING 4 — D35's rule 3 has no satisfying move, and the builder is right about that

The freeze requires every iteration to name which of `T2.01`, `XL.01` or
`T6.01` it moved, *"None" legal at most twice running*. I re-derived all three
independently of the builder's four derivations:

- **`T2.01`** — settled FAIL (a1 08-12). Both repair lanes desk-owned and
  prohibited to the builder by name.
- **`XL.01`** — settled FAIL (a2 08-19). Successor `NE.08` sits behind `T6.03`
  BLOCKED ← `T2.10` FAIL; that repair is the Review's.
- **`T6.01`** — no ledger row, no test file; dependency `T4.05` has no row
  either, and `T4.05 ← T4.04 ← T2.01`.

**All three are downstream of `T1.08`** — `FAIL`, *frees 3, blocks 45*, impl
unchanged 6 days, by far the largest lever in the system (next is `NE.01` at
7). The freeze names the symptom correctly and does not name `T1.08`, which the
builder may not repair. So rule 3 asks the builder for something the dependency
graph forbids, and since 11:07 **every slot records a further violation into a
counter with no bound and no satisfying move.**

The builder's conduct here is the best thing in the window and should be said
plainly: it predicted the breach one slot *before* it fired, routed it as
`d35-none-quota-has-no-satisfying-move` with the derivation attached, wrote the
FIRED record **once** with an explicit instruction not to duplicate it per
slot, touched no `DECIDE` field, and did not manufacture or re-label a unit of
work to dodge it. That is exactly right. **D35's own `decide_by` is today.**

---

## FINDING 5 — two live ratchet breaks, both real, both desk-owned

1. **`DEFAULT-ACTION-EXPIRED` = 1, baseline 0** (`decisions --check`, EXIT 1) —
   `D33`. Armed on purpose by the 112th audit's repair (`5c5146e`: the conduct
   `continue` no longer exempts the action-date checks) and certified onto
   `T0.28` (`7c8fc7b`, `live_expired_actions 1.0`). D33 is `CONDUCT-DESK` and
   `STALE by 1 day`. **The Review sat at 06:37 today and did not discharge it**,
   and a CONDUCT-DESK entry is the desk's to execute and report, never to ask.
2. **`goal_unrunnable`** (`coverage`, EXIT 2) — 4 NEW GOAL.md citations resolve
   to corpses: `GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`, all welded behind
   `LC.07`, against a shrink-only baseline of 7. `GOAL.md`'s present tense is
   false for each.

Neither is the builder's; it left both alone, correctly, and said so.

---

## The audit, section by section

**1. Integrity of the ledger — CLEAN.** `run verify`: 109 re-judged, 107
controls probed, **0** non-re-deriving verdicts, **0** gates ignoring their
control, **0** unreplayable, **0** unauditable. 2 PASSes with no control at all
(`T0.01`, `T0.10`) — existence claims whose gate was never shown capable of
reporting the bad case; long-known, structural, unchanged. 1 self-exclusion
(`T0.18`), correct by construction.

**2. Thresholds and controls — NOTHING TO REPORT, and that is a result.** No
file under `experiments/registry*.py` or `experiments/tests/` was modified in
this window. The three code commits (`5c5146e`, `7c8fc7b`, `b278f6a`) touch
`decisions.py`, `resolution.py` and the ledger/budget, all discharging the
112th audit's own items, all with a re-buy priced in the same commit. The one
semantic change I checked in the prior week moved the **hard** way:
`registry_expansion.py` narrowed HR.1's cross-mic control from controlling for
occasion to *"controls for EQUIPMENT and NOT for occasion"* — an honest
admission that a control proves less than claimed, which is the direction that
costs the claimant.

**3. Drift from the goal — no drift; a stall.** Eight of eleven commits in the
window are journal entries and freeze bookkeeping. The three code commits serve
audit hygiene, not a GOAL.md sentence. Under ordinary circumstances I would
call that drift and say so. I do not, because the board was genuinely empty
every time it was re-derived and manufacturing a dispatch would have been
worse. The converse question is the damning one: **5 of GOAL.md's own
constitutional commitments are CLAIM-DEAD** — *smell*, *balance*,
*shelter/building*, *thermal (kills)* — every claim spec parked or foreclosed,
with 3 `PARK-ON-AN-UNREACHABLE-RELEASE` pairs whose stated revival path cannot
be walked today (`BA.02→LT.08`, `SH.01→SH.02`, `SM.02→SM.03`). Curiosity: 12
specs, 2 pass. One brain / unison: 28 specs, **1** pass. Fast/slow: 8 specs,
**0** pass. Those are the claims my charter warns are quietly neglected in
favour of easy wins, and they are.

**4. Builder alive and productive — alive, disciplined, unproductive, in that
order.** 13 iterations today, **13 `rc=0`, 0 dark slots**, PASS delta **0**.
Meters read and named correctly every slot (`week:all models` 36%, the gate;
Fable 47%; week 47% elapsed against a ~55% pace line). Hygiene clean:
`lost_iterations.log` 0 bytes, no undeclared pids, `/data` 68 G free. No
repeated identical failure, no paused loop, no credit exhaustion.

**5. Compute honesty — nothing wasted in the window, and a standing debt.**
`2026-W38`: **0.9176 of 30 h drawn; ~29.08 h expire Saturday 2026-09-26** with
no legal buyer. The builder refused to manufacture a dispatch against them and
named that refusal in every slot — **that is the correct call**; a GPU hour
spent on a run nothing asked for is worse than an expired one. The standing
waste is historical and already recorded: `gpu_hours_no_verdict` shows **`D1.0`
33.78 h across 2 attempts for 0 verdicts** and `UNATTRIBUTED` 6.32 h across 21
jobs.

**6. Stuck decisions — 1 OVERDUE (Finding 1), 0 MEANS-ESCALATED, 0
UNDECLARED.** I checked the class census directly: nothing is sitting on the
owner's desk that a measurement could settle, and **there is no `UNDECLARED`
entry to arm this audit.** My charter says arm at least one per audit; the
honest report is that the class is empty, and I would rather say that than
manufacture an arming to satisfy a quota. The remaining six unarmed entries are
`CONDUCT-DESK` (D33, D35) and `CONDUCT-MISFILED?` (D31, D32, D34) — the latter
a soft routing question, not a blocker. `D32`, `D34` and `D35` all carry
`decide_by: 2026-09-24`: **today.**

**7. Bakeoff hygiene — one caveat already self-reported, correctly.** `T4.06`
passed on attempt 1 with `loss_reweight` the sole winner, but its own
`STATISTIC_BOUND` note (`f7900b5`) records that conjunct (2)'s margin is
**0.0187 = 6.9% of the incumbent's own seed spread 0.2699, with one of three
seeds regressing.** The desk wrote down that the latent-recovery conjunct is
not to be quoted as demonstrated, and the ratio result is what is
demonstrated. That is a winner inside the noise margin, *disclosed by the organ
that owned it, before any auditor asked*. No VOID is being treated as a
verdict; `champions --check` EXIT 0 with the ratchet intact, though it carries
2 unverified verdicts (`Learning core` LC.03=VOID, `World` no deciding run) and
3 trigger-debt seats.

**8. The honest summary — no.** We are not closer to a curious humanoid that
climbs the ladder than we were yesterday. We are closer by **zero specs**, and
the twenty-six slots that produced that zero were each individually correct.
The instruments are in excellent health and the thing they instrument has not
moved in a day. The desk saw this at 08:34 and reached for a freeze; the freeze
is aimed at the right disease and its tripwire has no satisfying move, because
the actual obstruction — `T1.08`, blocking 45 specs, impl unchanged 6 days — is
owned by a desk whose own drain reads **UNBOUNDED** against 28 FAILs it holds.
**The bottleneck is no longer the builder's meter, and it is no longer even the
builder. It is that the only organ permitted to unblock the ladder sits once a
day and disposes six rows a cycle against a queue that takes eight tomorrow.**

---

## FOR THE BUILDER

1. **DO NOT FIRE D19'S DEFAULT.** `decisions --check` will keep printing
   `D19 costs 3 specs OVERDUE — DEFAULT IS DUE TO FIRE` until the filing is
   repaired. **The owner answered it** — *"yes may download anything to
   /data"* — and the corpora are on disk and already carry three `HR.1`
   attempts. Firing `NO FETCH` would reverse an owner ruling. A dated guard is
   now appended under D19 in `DECISIONS_NEEDED.md`; read it before acting on
   that line.
2. **The filing repair, and it is documentation only — no code, no threshold,
   no ledger row.** Give the ruling its own header so the parser can settle it:
   insert `## D19 — RESOLVED BY THE OWNER 2026-09-17: "yes may download anything
   to /data"` immediately above the ruling text that currently sits at the tail
   of the `## D35` section (`docs/DECISIONS_NEEDED.md`, above line 4430), and
   add the matching `RESOLVED` entry to `docs/DECISIONS_RESOLVED.md`. Keep the
   superseded `## D19` entry and its `DECIDE:` block exactly where they are —
   the retention convention is right; it is the *missing header* that is the
   defect. Verify with `decisions --check`: D19 must leave the armed list.
   Re-buy `T0.28` (its `IMPL_DEPS` reads this page) and price the re-buy in the
   same commit.
3. **Then consider the durable repair, and describe it before building it** —
   note that D35 rule 2 forbids new checkers, so this is a *report*, not a
   licence: `decisions.py:333-345` settles on headers alone. A ruling recorded
   in prose under a neighbouring heading is invisible to it. The narrow fix is
   to also scan a decision's own `DECIDE:` block region for an explicit
   `SUPERSEDED`/`RESOLVED` marker. **Do not build it under the freeze** — put
   the design in the queue and let a desk rule whether it is exempt.
4. **Do not fire, extend, or touch D33's `DEFAULT-ACTION-EXPIRED` red.** It is
   `CONDUCT-DESK`, it is the Review's, and the 112th audit armed it on purpose.
   Leave `BASELINE_ACTION_EXPIRED` alone.
5. **Standing prohibitions unchanged and still binding:** `T1.08`'s pipeline
   repair, `A4`, `T2.10`, `SO.07`, `SO.10`, `UB.10`'s successor arm,
   `W1.01`/`W1.03`/`W1.04` registration, the world-edit window, the `lc03` seat
   row, the `t306` venue row, and the GEN citations. The 09-25 `WAITS-ON:` unit
   unlocks tomorrow — **do not start it early.**
6. **Keep answering the D35 gate NONE and keep counting it in the journal.**
   Do not duplicate the FIRED record. Do not manufacture or re-label a unit of
   work to make the count reset — the breach being visible is the only value it
   has left, and item 4 of this report says in writing that the quota has no
   satisfying move that is yours to take.

---

## FOR THE OWNER

**1. D19 is ANSWERED and the file does not know it — one line fixes it, and
until it does, an instrument is ordering your ruling reversed.** You ruled *"yes
may download anything to /data"* on the hearing corpora. Because that ruling was
written under the `D35` heading instead of its own, `decisions.py` still reads
D19 as an unanswered decision whose `NO FETCH` default is ten days overdue — and
it prints that as the highest-cost line in the report every organ reads. Nothing
has fired it. The repair is routed to the builder as a filing fix and needs no
decision from you. **What would help: confirm the ruling stands as recorded
("anything, to `/data`", with the builder's self-imposed 15 GB-free floor and
the tenant-safety constraint unchanged).** `/data` is at 33 G used / 68 G free
today, well clear of the floor.

**2. D35's rule 3 is unsatisfiable as written, and it is your one-line strike.**
The freeze (filed at the desk 08:34 today, `decide_by` **today**) demands every
builder iteration move `T2.01`, `XL.01` or `T6.01`. All three are settled FAILs
or unimplemented behind `T1.08`, which blocks 45 specs and which the builder is
prohibited from repairing. The quota breached at 11:07 on the freeze's own first
day and every slot since records another violation. **The freeze's diagnosis is
right and I would not strike the whole thing** — *"465 docs/scripts, 352
infrastructure, 258 touching a spec"* over 30 days is the real disease, and
rules 1 and 2 (Tier 0 closed at 39, no new audit organs) bite correctly. **Rule
3 is the part that cannot be obeyed.** The minimum amendment: let a slot
discharge the quota by naming and advancing the *blocker* of a creature gate —
today, `T1.08` — rather than the gate itself. That converts an unbounded
violation counter into the unblock the ladder actually needs.

**3. NO-DECISION, reported because the price perishes: ~29.08 of `2026-W38`'s
30 free Kaggle GPU-hours expire Saturday 2026-09-26 with no legal buyer.**
0.9176 h drawn. The builder refused to manufacture a dispatch against them in
all thirteen slots today and named the refusal each time. **That is the right
call and I am endorsing it, not flagging it** — but the reason no buyer exists
is Finding 2: there is no runnable creature spec to spend them on.

**4. NO-DECISION, and it is the number I would want you to see: the ladder has
not moved in 25 hours across 26 clean slots — `110 -> 110` — and no instrument
in this repo can go red for that.** All 17 ratchets count defect classes; none
counts motion. Every organ is healthy, every check passes or fails in a known
and owned way, and the project stood still. `T1.08` is the lever: FAIL, frees
3, **blocks 45**, implementation unchanged for 6 days, held in a queue whose
own drain reads UNBOUNDED against 28 FAILs. If you want one thing moved this
week, move that.

**5. `D33` is one day past its `decide_by` and unanswered, and it is now also a
broken ratchet** (`DEFAULT-ACTION-EXPIRED` 1, baseline 0). It asks whether the
Review is capable of producing the W1 world-edit design at all; `W1.01`,
`W1.03` and `W1.04` are now **nineteen days** unregistered, and the builder
that would execute them has been idle for 26 slots. The desk's own revised
recommendation asks for less than the original and still stands.

**6. Seven dated promises break at midnight tonight and no desk can sit before
then.** One of them — `pl02-eye-gate-reads-the-encoder-not-the-eye` — carries a
stop-rule that routes a **DECLINE to you** on its fourth break. I verified the
work is **done** (`a4132c8` spec edit with `EYE_RADIUS_R2_MIN` 0.80 unmoved,
`c150187` smoke at 0.9250, `PL.02` attempt 2 VOID on the ledger). PL.02 is the
sole registered falsifier of GOAL.md's PLASTIC-ONLY decree. **If that DECLINE
reaches you tomorrow, it is false** — the row needs a stamp, not a decision, and
the stamp is the Review's first act at 06:37.
