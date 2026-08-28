# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: DAILY (Part 2, the test re-examination, runs Sundays only).

**2026-08-28 06:43 UTC — DAILY. Window: 2026-08-27 06:40 → 2026-08-28 06:43.**

*The one sentence: **the builder has been dark for sixty-six consecutive slots,
and in the last three days four organs have published eight forecasts of the
moment it wakes — three have come due, all three were wrong, and not one of
them, right or wrong, would have changed a single action anybody took.***

---

## 1. The numbers

**Ladder: 84/187 demonstrated (44.9%).** Fourth consecutive day on which not one
figure in this table has moved.

| | this window | previous |
|---|---|---|
| demonstrated | **84** | 84 |
| registered | **187** | 187 |
| rate | **44.9%** | 44.9% |
| net new PASS | **+0** | +0 |
| rework (attempt > 1) | 62.5% (60/96) | 62.5% |
| ledger totals | 84 PASS / 9 FAIL / 3 VOID | same |
| runnable now | 34 | 34 |
| unreachable | 69 of 187 | 69 |

**Runs in the window: zero.** Last builder iteration **2026-08-25 12:23**; last
ledger row of any kind `T0.21` PASS **08-25 10:14**; last first-ever claim PASS
`T3.01` **08-21 01:28 — 7.2 days**. `ladder.log` carries **66 consecutive
`PACING:` lines** and nothing else. All four commits in the window are overseer
audits (37th–40th), all **DRIFTING**.

**The frontier, recomputed live and unchanged.** `T2.01` (FAIL, 2.67σ against a
5σ bar) frees **35**, blocks 36. Behind it `LC.03` 8, `NE.01` 8, `UB.10` 4,
`T2.02` 3, `LG.01` 3. None waits on compute. **Is the builder working on the
frontier? It has not been asked to do anything for three days.**

**Goodhart check: still not applicable, still the point.** `coverage` exits 0,
`0 CLAIM-DEAD`, 14 of 23 commitments carry a live claim spec with nothing
passing. GOAL citations 16 cited / 4 dangling (at baseline). Champions ratchet
6/8; decisions ratchet 0/10 with 11 armed entries, every one due 08-31.

**Meters at 06:07:** `week:all models` **69%** at 58% elapsed, pace line 63 →
skip. `week:Fable` **100%** until 08-31 04:59. Kaggle W34: **0.31 of 30 h
spent.**

---

## 2. THE FINDING — eight forecasts, three days, four organs, and not one
## decision that turned on any of them

Since 08-26 the following predictions of the moment `pace_gate` releases the
builder have been published, each by an organ that had read its predecessor:

| made | organ | predicted release | outcome |
|---|---|---|---|
| 08-26 06:37 | 33rd audit | **08-27 05:07**, "no owner action needed" | **wrong** |
| 08-26 12:37 | 34th audit | gap "cannot close" | **wrong, other way** (gap 12 → 6) |
| 08-27 00:37 | 36th audit | **08-28 05:07** | **wrong** |
| 08-27 06:46 | Review (this desk) | **08-28 06:20** | **wrong** |
| 08-27 12:45 | 38th audit | 08-29 08:00 | pending |
| 08-27 18:47 | 39th audit | 08-29 10:00 best / never worst | pending |
| 08-28 00:46 | 40th audit | 08-30 04:00 | pending |
| 08-28 06:47 | 41st audit | 08-30 21:00 | pending |

*(The ninth arrived while I was writing this. I note it without irony: the 41st
audit derived **08-30 21:00** from a 7.0 pts/day burn, which is exactly this
desk's r = 7.0 branch in §3 to the hour — two organs reaching the same number
from the same data independently. It does not change the conclusion, because it
still lands after the Kaggle expiry. It did also re-assert a per-audit price —
"one Opus run is ≈ 1.4 pts" — which is the fifth instance of the estimate
falsified below, and the reason B5 exists.)*

Every resolved forecast was wrong and **every one was optimistic**; the
predicted date has moved monotonically later, roughly one day per day. That is
not four organs each being careless. It is one method being repeated: each
forecast extrapolates the meter's *most recent local slope*, and the meter moves
in flats of ten-to-fourteen hours punctuated by jumps — so a forecast written at
the end of a flat always reads a burn rate near zero. Mine was written after
fourteen flat hours and predicted this morning.

**The part that matters more than the errors: not one of these forecasts was
decision-relevant.** Under every branch — releases tomorrow, releases Sunday,
never releases — the builder's available action was identical, because the
builder is the thing being skipped and cannot act either way. Three days and
roughly ten Opus sessions have gone into predicting an event nobody could
prepare for differently. **I am not writing the ninth forecast.** What follows
instead is a conditional with its falsifier attached, and the reframe that
makes the whole question small.

**The mechanism question, re-tested out of sample, and it replicates.**
Yesterday this page reported that 75% of the meter's rise over 42 hours fell in
hours when this box issued zero requests. I re-ran that join on the *following*
24 hours — data that did not exist when the claim was made:

| 08-27 06:00 → 08-28 06:00 | hours | on-box requests | output tokens | Δ all-models |
|---|---|---|---|---|
| hours containing an organ session | 5 | 384 | **444,251** | **+2** |
| hours with **zero** on-box requests | 19 | **0** | 0 | **+5** |

**71% of the rise, out of sample, in hours when this box ran nothing** — against
75% in sample. Four full Opus audits and a Review, 444K output tokens, moved the
meter by at most two points *in total*. The gate throttling the builder is
driven from off this box; that now has a prediction and a confirmation behind
it, not a correlation.

**This falsifies the 40th audit's RANK 3 in passing.** It priced one Opus audit
at *"≈ 1.2 pts ≈ 3.1 hours of postponed builder wake-time"* and escalated D13 on
that basis. Measured on the same window it was written in, the audits' entire
contribution is bounded at 2 points across four of them. The price is the fourth
attempt to model this meter to be falsified inside eight days — and the reason
it keeps recurring is structural: *"do not model the meter"* is written in
`ladder_prompt.md`, which is **the builder's file**, and the organ that keeps
making the error never opens it. **That belongs in `docs/LESSONS.md`, which all
four organs read** — routed as B5 below rather than written by me, because a
lesson is the builder's ratchet and I would be putting words in its file.

---

## 3. THE REFRAME — the question everyone has been forecasting is worth less
## than the question nobody has asked

**Will the gate open in time to save W34's 29.69 free Kaggle hours?** Stated as
a falsifiable conditional rather than a date: the line rises 9.29 pts/day; the
gate needs a 6-point deficit closed before **08-30 00:00 UTC**; that requires
exogenous burn below **5.84 pts/day.**

| window | measured burn |
|---|---|
| last 48 h | 8.5 pts/day |
| last 24 h | 7.0 pts/day |
| last 12 h | 6.0 pts/day |
| last 6 h | *4.0 pts/day* |

**Every window of twelve hours or more exceeds the threshold.** The only window
that clears it is six hours long and contains a single integer increment — which
is precisely the local-flat extrapolation that produced the four wrong forecasts
above, so I decline to lean on it. **W34's hours should be treated as sunk**,
and if I am wrong the error is in the safe direction: the builder gets compute
it was not counting on.

**And here is why that matters far less than three days of organ output implies.
GPU weeks are keyed `%Y-W%U` — Sunday-start (`gpu.py:_week`). `2026-W35` opens
Sunday 08-30 00:00 UTC with a fresh 30 hours. The meter resets 08-31 04:59 with
the pace line back at its 25% floor, which admits the builder immediately.** So:

- Under **every** rate measured this week, the builder's next iteration lands in
  **W35**, not W34 — even the optimistic 6 pts/day branch releases it at
  ~08-30 02:00, two hours *after* the new allocation opens.
- The builder therefore wakes, with near-certainty and **with no owner action at
  all**, into a **full free GPU allocation and six days to spend it.**

The thing actually at stake this week is at most a slice of a dying quota. The
thing that decides *next* week — does the builder dispatch inside the first 48
hours of W35, or drift to Friday like the last three weeks — is entirely within
our control, has had zero attention, and has cost **61 of 90 free GPU-hours in
three consecutive weeks** (W32 8.94 unspent, W33 22.37, W34 29.69), every time
by the same route: the loop was dark on the Sunday. **That is the instruction I
have put in front of the waking builder**, replacing a head block whose deadline
expires before it can be read.

---

## 4. A small finding, and it is a green tick that says "clean"

`gpu_budget.json` records `2026-W31: kaggle 37.4554` against a
`KAGGLE_WEEKLY_HOURS = 30.0` hard ceiling — **7.46 hours over**, with
`"overruns": []`, zero `charged_jobs` rows for that week, and no
`opening_balances` entry (the mechanism built for exactly this covers W32 only).
It is almost certainly an artifact of the 2026-08-08 ISO→`%U` key migration
rather than a real overspend, and it changes no decision today.

What it does change is a word. Four consecutive audits have reported §5 as
*"accounting clean, `overruns: []`, books reconcile"* — while the same file
holds a week 25% over a ceiling whose enforcement check only ever runs at charge
time, so migrated hours pass under it unexamined. **`overruns: []` means "no
overrun was recorded", not "no overrun occurred",** and §5 has been reading the
first as the second. Routed as B6.

---

## 5. Steering maintenance (Part 2.5) — done

**1. `scripts/ladder_prompt.md` — three edits, all in the builder's own file.**
- **The dated priority head block is replaced.** The 08-27 version ordered
  *commit SM.03 → dispatch → build* against "W34's hours expire 08-30" — a
  deadline that, per §3, expires before the builder can read it. The new block
  gives a **branch the builder evaluates itself** (`time.strftime('%Y-W%U')`
  against `gpu_budget.json`) instead of a date I predict, with the W35 arm
  spelled out: dispatch inside the first 48 hours, do not chase W34, do not
  spend an iteration writing its post-mortem.
- **Added, in place of a ninth forecast:** the eight-forecast record, the
  out-of-sample replication, and the instruction *do not write the ninth*.
- **Two factual corrections to the Kaggle block, both of which overstated the
  scar in this desk's favour:** it said W32's and W33's *"whole allocations died
  unspent"* — recomputed from `weeks{}`, the losses are partial (21.06 and 7.63
  hours *were* spent), the run is three weeks not two, and the correct total is
  61.0 of 90. It also dated W34's opening to 08-24; `%U` opens it Sunday 08-23.

**2. `docs/FIELD_WATCH.md` — nothing owed.** Unchanged since sweep wk4
(`474061d`, 08-24); all three nominations dispositioned in INTEGRATION_QUEUE on
08-25 (wk4-N1 ACCEPTED as an A4 variant, N2/N3 REJECTED with re-open triggers).
Next sweep Mon 08-31.

**3. Seat staleness — no seat has moved, because nothing has run.** *Learning
core* PENDING D10 (armed 08-31); *Vision encoder* contested with T3.01 as its
defence; *Sensory fusion* PARKED with the `UB.10` arm redesign owed by this desk
**08-30**. Ratchet steady at 6/8 phantom arenas — and per the 40th audit that
ratchet counts only `ARENA-MISSING`, so it would read *perfect* after one markdown
deletion. Builder item B4.

**4. Organ liveness — all four fire on time; one of them does nothing.**

| organ | cadence | last fire | verdict |
|---|---|---|---|
| overseer | 6 h | 08-28 06:37 | live |
| field watch | Mon 05:37 | 08-24 05:54 | live (next 08-31) |
| review | daily 06:37 | 08-28 06:37 | live (this) |
| builder | hourly | 08-28 06:07 | **fires on time; 66 slots, 0 work** |

`lost_iterations.log` is still 0 bytes — correct, since no slot has *attempted*
a model since the fallback repair landed. It has not yet been tested and must
not be trusted until it has (B3).

**5. The cron collision stopped being a cost argument and became a realized
fault — this morning, between these two organs, and the overseer caught it.**
`37 */6` (overseer) and `37 6 * * *` (review) fire in the same minute daily;
three audits reported it as *two Opus sessions on a shared meter*, a cost
argument yesterday's measurement undercuts. What actually happened is different:
**this Review staged its three files while the 41st audit was running, and the
audit's first commit (`af37b21`) contained six files instead of three** — the
Review's page, its log row, and its `ladder_prompt.md` edits, none of which the
overseer authored. For a few minutes `git log -- docs/PROGRESS.md` credited this
desk's page to an overseer audit, with **the auditor and the audited in one
commit**, against an overseer whose §2 duty is to audit Review diffs
independently.

**It was detected and reversed by the overseer, unprompted, within minutes** —
`reset --soft`, foreign paths unstaged, recommitted as `a2722ac` with its own
three files, nothing pushed, no content touched. I verified this independently
against the reflog rather than taking the report's word: `a2722ac` is clean, and
all of this desk's work came back to the tree intact. **That is the system
working**, and it belongs on this page as prominently as the fault.

**Its diagnosis is also sharper than the one I first wrote here, and I am
adopting it:** a named-pathspec `git add` does *not* protect you, because
`git commit` writes the **whole index** — only `git commit -- <paths>` is safe.
`ladder_loop.sh:166` already does exactly that (`git commit … -- $HARVEST_PATHS`),
under an `add -A` ban written after an owner-side sweep on 08-24. **So the
correct form exists in the one organ that is a shell script, and the two organs
that commit from an agent session are ungoverned by it** — and because cron
fires them in the same minute every morning, a shared-index collision is not an
accident, it is *scheduled daily*. Second attribution incident here, after
`9449a1b`'s block signed "Review, 2026-08-24" when no Review had run; the
standing note applies verbatim — *an organ that can be quoted by another organ's
name is an organ whose independence is decorative.*

Routed as **B0**, which is the overseer's own B8 and this desk's independently —
we found it from opposite ends of the same commit, which is the one form of
agreement between auditors that is worth something. Owner fork 5 is the cron
line, which no organ may edit. This commit is the Review's, made after the
reversal, so the record is attributable.

---

## 6. The honest paragraph, no numbers

We are not closer, and this week the shape of *how* we are not closer changed
into something worse than idleness. For three days this system has been fully
awake, fully instrumented, scrupulously honest, and pointed entirely at itself.
Every organ read its predecessor, found a real flaw in it, corrected it
carefully, and published — and the corrections were true, and they were about
each other. The machinery of falsification has been running beautifully with
nothing but its own output in the chamber. Nobody lied, nobody weakened a bar,
nobody manufactured a result; the failure is subtler and it is the one this desk
exists to name: an organ whose only remaining subject is the organ next door is
no longer doing science, however rigorous it is being. The week's single most
important step toward Jack was made by nobody, because the only organ that can
take one has not been permitted to act since Tuesday. The most concerning drift
is that we have become very good at watching ourselves not build him — and that
the audit trail of doing nothing is now indistinguishable, in volume, tone and
apparent rigour, from the audit trail of doing everything.

---

## REWRITTEN / STRENGTHENED

**None — DAILY mode. Part 2 runs Sunday 08-30**, together with the anatomy
audit, the completeness audit, and the `UB.10` arm redesign owed that day.

---

## FOR THE BUILDER — ordered

**B0. `git commit -- <paths>`, in every organ that commits from a session.**
This is the overseer's B8 and mine, found independently from opposite ends of
the same commit (§5.5). The subtlety that makes it worth code rather than prose:
**`git add <named-paths>` does not protect you — `git commit` writes the whole
index**, so a second organ's staged files ride along. `ladder_loop.sh:166`
already has the correct form; `overseer.sh`, `review.sh` and `field_watch.sh`
delegate committing to their agent session, where nothing enforces it. Minimum
fix: a pathspec in every organ prompt's commit instruction. Better fix, since
prose bans get forgotten and this one was already written down once: a
pre-commit hook refusing any commit that touches another organ's output file
(`PROGRESS*.md` → Review, `OVERSIGHT.md` → overseer, `FIELD_WATCH*.md` → scout)
unless that organ is the author. Cheap, CPU-free, and it closes an incident
class that has now fired twice.

**B1. Commit `experiments/tests/sm_03_nose_reports_occluded.py` before anything
else.** Fourth day, seventh organ asking. 32 KB, the only runnable claim spec
for *smell*, and the only spec of 187 whose implementation git has never seen —
one `git clean` from gone. Then re-launch its pilot with
`scripts/launch_detached.sh`; the 08-25 pilot died with its session and its log
is 0 bytes, so the numbers in that iteration's summary were never produced.
I again decline to sweep it into a Review commit (`c0afded` bans exactly that).

**B2. Dispatch inside the first 48 hours of your GPU week.** Read
`time.strftime('%Y-W%U')` and `gpu_budget.json` before you rank anything; the
new head block in `ladder_prompt.md` gives both arms. If you are in W35, W34 is
sunk — do not chase it, and do not spend an iteration on its post-mortem.

**B3. Verify the fallback repair rather than trusting it.** Your first slot will
refuse on Fable if it lands before 08-31 04:59. Expected: a `LIMITED on fable —
falling back to opus` line and a real iteration. If instead you see `rc=1` in
three seconds with `lost_iterations.log` still at 0 bytes, `model_limited()`
missed a fourth wording — say so loudly, do not quietly re-run.

**B4. Baseline the ratchets that only count one class.** (40th audit, RANK 1.)
`champions.py:449` counts `ARENA-MISSING` only, so deleting 13 phantom ids takes
the tool to a *perfect* ratchet while five seats — including the plastic-only
decree and the World seat — become permanently unfalsifiable. `decisions.py` has
the same shape with `NO-DEFAULT`. The repair precedent is in the repo:
`coverage.py` had this exposure and `T0.21 P2` closes it. No test in
`experiments/tests/` imports either tool.

**B5. Move "do not model the meter" into `docs/LESSONS.md`.** It currently lives
only in `ladder_prompt.md`, which the auditing organs never open — which is why
four separate attempts to price organ-hours against `week:all models` have been
published and falsified in eight days, the most recent yesterday. The
generalised form: *a lesson written in one organ's prompt is not a lesson the
system has learned.* Yours to write; I am not putting words in your file.

**B6. Make `overruns: []` mean what §5 reads it as meaning.** See §4:
`2026-W31` records 37.46 h against a 30.0 ceiling with an empty overruns list,
because the check runs at charge time and migrated hours never pass under it.
Add a standing invariant (any `weeks{}` entry over `KAGGLE_WEEKLY_HOURS` with no
`overruns` row and no `opening_balances` entry is a violation) and a known-answer
property in `T0.21`.

**B7. Make the orphan class unrepeatable — a registry×index join.** Carried,
third day. No instrument checks that a registered, non-PARKED spec's
implementation is *tracked in git*. Add it to `coverage.py` beside
`goal_citations()` (untracked implementation exits 2, shrink-only baseline). Pair
with the 36th audit's `gpu.py:274` fix — `--untracked-files=no` makes an
untracked file invisible to the one guard whose job is refusing a job whose code
isn't the code being tested. CPU-free, and four findings have come through this
hole.

**B8. Report skip streaks.** Carried. `ladder_loop.sh` should count consecutive
`PACING:` slots and past 6 emit a `PACE-STREAK n` line plus a marker the overseer
reads. Sixty-six have now passed and no instrument can say so.

**B9. `SH.02` implementation** — tier 2, CPU_LONG, deps all PASS, no owner gate,
no GPU. Still the only claim-kind work on the board that needs nothing from
anybody.

---

## FOR THE OWNER — strategic forks only

**1. The GPU ask is now smaller than it looked, and I am downgrading it.** For
three days this desk and the overseer have escalated `JACK_NO_PACE=1` as urgent
because 29.69 free GPU-hours were about to expire. Per §3 those hours are sunk
under every burn rate measured this week, and the builder wakes into a **fresh
full allocation with six days to spend it** at the 08-31 reset **with no action
from you at all**. Setting `JACK_NO_PACE=1` today would still buy a real ~42-hour
window and it costs you nothing (the 90% hard stop is untouched and has never
fired this week — 21 points of headroom remain), **so it is still worth doing if
you are reading this today.** But it is no longer the emergency three days of
organ output made it sound, and I would rather correct that than keep the
urgency because it got your attention.

**2. The one that is genuinely yours, and it outranks the GPU question.**
`pace_gate` regulates the builder against a pool whose movement is now
**measured twice, on independent windows, to be ~70–75% exogenous to this box.**
It has one call site — the builder, the only organ that writes to the ledger —
while the three Opus organs that produced every document in the last three days
are ungated on the same meter. Its own source comment diagnoses this
("*being the only consumer with a gate, it is the one that starves*") nine lines
before installing the gate. Cost to date: 66 slots, three days, zero ledger rows.
**Recommendation: suspend it — `JACK_NO_PACE=1` permanently, or delete
`pace_gate` — and let the 90% hard stop be the only limit.** I have not touched
it: rule 4 binds, I cannot show the counterfactual, and the 37th audit routed
the redesign to a bakeoff rather than an argument. But the bakeoff has three
arms sitting implemented in one file while the question is settled by essay, and
that is law 3's own disease pointed at our machinery.

**3. Eleven decisions default-fire on 2026-08-31 — one date, one hour.** `D1`
costs 38 specs, `D10` 8, `D4` 8; `D7` costs 0 and fires alongside them. Two of
them (`D13`, `D14`) exist *solely* to unblock the builder this week and fire the
day the week ends. And per the 40th audit, on 09-01 there will be 11 OVERDUE rows
covering 54 specs and `decisions --check` will **exit 0** — `overdue` is a row
field, never a violation. Recommendation unchanged: **answer D1 and D10, or
re-arm both past the W1 design** (owed by this desk 08-30).

**4. Unchanged, and still the one that matters most: build W1, do not patch W0.**
Four independent instruments say the world is too thin to be worth learning.
Design owed by this desk **2026-08-30**.

**5. The cron collision is now a correctness fault, not a cost one — please fix
the line.** `37 */6` and `37 6 * * *` fire in the same minute daily. This
morning that produced a commit containing both organs' work (§5.5); the overseer
caught and reversed it unprompted, so nothing was lost, and both of us have
routed the code fix. But the shared index remains, the collision is scheduled
every morning, and the next one may not be noticed — the two organs happened to
be the two that read each other's commits. Three audits asked for this line as a
*spend* argument, which yesterday's measurement undercut, and leaving it was
reasonable on those grounds. It is now the mechanism by which the auditor and
the audited land in one commit. **Free fix: `37 3,9,15,21 * * *` for the
overseer.** Cron is outside every organ's mandate, so this one can only be yours.

**6. A question about us, not about the machine.** In the last 72 hours this
system produced eleven documents and zero measurements, and this page is the
twelfth. Every one was honest and most contained a real finding — about another
organ. The auditors are working exactly as designed; what is missing is the only
thing they audit. If the pace question in fork 2 is not resolved, the honest
recommendation is not to add another instrument but to **cut audit cadence to
match the rate at which there is anything to audit** — I say that against my own
organ, and I would apply it to the Review before the overseer, since the overseer
is the one that has produced the findings.
