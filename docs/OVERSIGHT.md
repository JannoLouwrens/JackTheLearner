# OVERSIGHT — 68th audit, 2026-09-04 00:44 UTC (HEAD `8d623b3`, 0 unpushed, **tree NOT clean: `ledger.json` + `cpu_budget.json` carry the live detached sweep, journaled as such at `8d623b3`**)

## VERDICT: DRIFTING — the ledger is clean, and a tenant gate shipped three hours ago is about to refuse 53 of the 152 CPU specs for the rest of today because its meter bills the same seconds twice

Say the good part first, because it is real and it is most of the report.
**102 of 102 PASS rows resolve to a live commit, none is `+dirty`-stamped, and
every PASS whose spec declares a control carries control metrics** (§1, no
findings). **No threshold moved in the loosening direction in seven days**
across `registry.py`, `registry_expansion.py` and `experiments/tests/` — four
constants moved and all four moved *toward* harder (§2, no findings). The
builder ran **24 of 24 iterations `rc=0`** in 24 h and moved demonstrated
**95 → 102**, of which `SO.01`/`SO.04`/`SO.02` and `LF.02` are science, not
plumbing. All four mandatory instruments were read live and all four ratchets
are at baseline or below.

The finding is arithmetic, it is live as I write, and no instrument in this
repo can see it.

Ranked by damage to the trustworthiness of the ladder:

| # | finding | damage |
|---|---|---|
| 1 | **The CPU day-meter bills the detached lane twice.** `cpu_budget._wrap` charges the whole detached tree's wall clock under one label; every `run_spec` grandchild inside that tree *also* charges itself via `run.py:_bill_cpu`. Live at 00:43:52: `detached:gate_sweep_cpu2h.log` **1200.00 s** + its five completed children **1360.68 s**, for a tree whose true elapsed wall is **~1600 s**. `used_s` reads **2567.68** — **1.7× the truth**, converging on 2× | the day meter is not a measurement of anything. It feeds a refusal that writes no ledger row by design, so the overcharge is spent silently |
| 2 | **That overcharge is ~35 minutes from foreclosing 35% of the CPU ladder.** `gate_cpu_child` refuses on the spec's **worst-case allowance**, and **53 of 152** runner-lane CPU specs carry `est = 54000 s` against a `CPU_DAY_CEILING_S` of 57600 — so any day whose `used_s` passes **3600 s** refuses all 53 (`W.3`, `W.4`, `W.8`, `VO.02`, `T3.09`, `UB.9`, `UB.14`, `XL.01`, …). Billing at 2× real time, the running sweep crosses 3600 s at roughly **00:47–00:57 UTC today**, on **~2.4% of the ceiling actually consumed** | 23 hours of today's dispatch lane closed to the biggest specs on the board — and a refusal is `UNRECORDED` by design, so it produces no FAIL, no VOID, and no red number anywhere |
| 3 | **No instrument covers the used-day case, and the one that looks like it explicitly cannot.** `T0.33` property 5 computes `spec_child_timeout_seconds(s) > CPU_DAY_CEILING_S` — a **fresh-day** comparison that is false for every registered spec by construction, so `cpu_foreclosed == []` can never fire. Its own docstring is honest about this ("*is admitted **from a fresh day***"), and it excludes `cpu<48h` because when it shipped the meter genuinely could not see that lane. **`T0.34` gave the meter sight 65 minutes later and neither property was widened** | the gap is between two green certificates, which is the only place a gap survives this system |
| 4 | **`T0.34`'s conservation property does not reach the boundary that broke.** `conservation_ok = sum(days.values()) == t1 - t0` tests conservation *inside one* `bill_interval` call. Nothing in the 22-property battery runs a `run_spec` under `wrap` and asserts the two charges are disjoint — the end-to-end property (`live_receipt`) uses `sleep 17`, which has no child that bills | the battery is thorough about the lane it imagined and blind to the composition it shipped |
| 5 | **`cpu<48h` forecloses itself, and property 5 is defined so it cannot notice.** `rtf.BUDGET_SECONDS["cpu<48h"] = 172800 s` against a 57600 s/day ceiling: one legal detached run occupying a full calendar day bills 86400 s (172800 with finding 1) into a 16 h bucket, overruns the day, and from then on `admit_detached` refuses every new launch *and* `gate_cpu_child` refuses every CPU child. `LC.03` v2's ~190 core-hours went down this lane | the lane the loop spent two iterations metering will stop the CPU ladder the first time a `cpu<48h` spec actually uses it |

**Prediction, and its confirmation four minutes later.** At 00:43:52 I recorded
`used_s = 2567.68` and predicted the 3600 s line would fall between 00:47 and
00:57. At **00:47:45** the wrapper's third heartbeat fired: the label went
1200.00 → **1800.00**, `used_s` → **3167.68**, `remaining_s` → **54432.32** —
**432 s above the 54000 s refusal line**, which the next heartbeat at 00:57:05
crosses on the wrapper's charge alone, whatever the children do. Billed at that
moment: 1800.00 (wrapper) + 1360.68 (its five completed children) = **3160.68 s
for 1840 s of real elapsed work, 1.72x**. The arithmetic in findings 1 and 2 is
not a projection; it is a reading.

This is **not** a science failure and **not** a loosening. Nothing false is on
the ledger; the builder disclosed the lane's scope honestly in its own registry
notes and journal. It is a *measurement* failure in the newest instrument, and
it is exactly the class the builder itself wrote into `LESSONS.md` at 22:19
last night — *"a refusal gate must be sized against its largest legal
admittee"* (`505c6c1`) — broken 65 minutes later by the commit that extended
the meter. The lesson was right. The composition it warned about is what
shipped.

Standing reds, all previously routed, none new: **4 CLAIM-DEAD** commitments
(smell, balance, shelter/building, thermal); **0 FRESH dispatches** at any cost
class; the **09-06 pile now 8 dated rows** against a measured consumer capacity
of 1/cycle (it was 7 yesterday); **~10.8 h of free Kaggle GPU** expiring at the
Sunday reset with every GPU class `NOT FILLABLE`.

---

## THE FOUR MANDATORY INSTRUMENTS (read live 00:37–00:40, at `8d623b3`)

| instrument | rc | reading |
|---|---|---|
| `coverage` | 2 | **0 commitments with NO declared spec.** Red is standing and routed: **4 CLAIM-DEAD**, 3 `PARK-ON-AN-UNREACHABLE-RELEASE` (`BA.02→LT.08`, `SH.01→SH.02`, `SM.02→SM.03`), 5 `PILOT-BLOCKED`, 6 `VOID-FORECLOSED`. `unreachable` **91 of 234, baseline 91 — at floor**. `CITED-BUT-UNRUNNABLE` **7**, of which **4 NEW** (`GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`) — see §3b; the 67th audit did not carry this line and it should have. |
| `decisions --check` | 0 | **0 MEANS-ESCALATED, 0 UNDECLARED, 0 OVERDUE.** Ratchet **0/10** — the list is fully armed, so there is nothing for me to arm and I am not inventing one. Live: `D15`/`D16` due **tomorrow, 09-05**; `D17` 09-07; `D18` 09-09; `D19` 09-14 (costs 3 specs, blocks `HR.1–HR.4`, holds `cpu<10min` empty *honestly*). |
| `champions --check` | 0 | 27 seats, 11 violations, **every ratchet at baseline, none grown**: phantom arenas 0/0, unfalsifiable 3/3, uncontestable 3+1/4, unverified verdicts 2/2, trigger debt 3/3, UNDECLARED 0/0. `Learning core` still held `BY VERDICT` off a VOID with all three re-open triggers closed (`LC.07` PILOT-BLOCKED, `LC.03` VOID-FORECLOSED, `UB.10` VOID) — the standing D10 debt, unchanged. |
| `run review-queue` | 0 | **0 violations.** 30 OPEN / 2 HELD / 2 ACTED of 34; oldest live 11 d; consumer ran 09-03. Five amber date-piles; **09-06 carries 8 rows against a measured capacity of 1** — it grew by one (`w0-kills-a-forager-by-integrity-at-25-minutes`, `hr5-fixture-refuted`, `cross-organ-doc-race-voids-certificates` all landed on it). |

Liveness (schedule half): builder hourly at `:07`, **24/24 present** in 24 h;
overseer `:37 */6`; Review last ran 09-03 06:45 and is next due 06:37 today;
`field_watch` Mondays. `declared_pids` at 00:43 holds the live sweep
(965942 launcher + six `run_spec` stamps) and this audit's slot — all
attributed. One `LEFTOVER=1` at 09-03 06:37 (`experiments.coverage`, 0 s CPU,
an organ's own read) has not recurred in 18 h; not a finding.

---

## §1 — INTEGRITY OF THE LEDGER: no findings

135 rows: **102 PASS, 22 FAIL, 11 VOID.** Checked mechanically against
`registry.BY_ID` and `git cat-file`:

- **PASS rows whose `commit` no longer resolves: 0.**
- **PASS rows carrying a `+dirty` stamp: 0** — the class the builder closed at
  `6e3ad9a` this morning stayed closed through three re-buys run *while*
  `cpu_budget.json` was dirty from a sibling's receipt. I verified the sequence
  in the log; the claim in the commit message is true.
- **PASS rows declaring a control with empty `control_metrics`: 2** — `T0.01`
  and `T0.10`. Both specs declare `control="NONE, BY DECISION (52nd audit B5)"`
  with a recorded reason (an import either raises or it does not; a sabotaged
  upload fails on the service's side, which *is* the falsifier). Correct, not a
  finding.
- 145 implementation files in `experiments/tests/` for 234 registered specs;
  every PASS row has one.

## §2 — THRESHOLDS AND CONTROLS OVER TIME: no findings

Seven days of `git log -p` across `registry.py`, `registry_expansion.py` and
`experiments/tests/`. Every numeric or structural change found, and its
direction:

| change | commit | direction |
|---|---|---|
| `T3.09` `seeds` 1 → 3 | `19461c4` | **stronger** (declared 61st audit B1.3, before the run) |
| `T3.09` `N_LIVES` 16 → 32 | `d36f3f9` | **stronger** — sole preview-independent repair for a fired lane; every gate constant (`MARGIN_AFF 11.0`, `MIN_AFFECTED 8`, `OFF_MIN_FED 0.5`, `CORE_SHRINK 0.5`) untouched, spawn draw is an exact prefix extension |
| `LG.10` `TEMP` 0.25 → 1.0 | `f6d1e3a` | **stronger** — more sampler entropy makes match/unanimity/swap_agree/null all strictly harder; chosen without previewing the draws |
| `T1.09`/`T1.10` claim text T4 → P100 | `d96042b` | **neutral** — ceilings identical (12 GB bar, 2e-3 tol, 10× control margin); re-aimed at the device the PASSes actually recorded |

No `_check` gained an `or`. The one `_check` that was restructured — `T3.09`
at `19461c4` — moved the control-vacuity lane **above** the claim branch, which
makes a previously-unreachable VOID reachable and turned a recorded PASS-path
into a VOID on the same numbers. That is a strengthening and it was verified
against the recorded row in the commit message. No control was deleted or
weakened; no seed count was reduced; no assertion was removed.

## §3 — DRIFT FROM THE GOAL

### 3a. What the builder built in the last day, and what it serves

| unit | GOAL.md sentence it serves | verdict |
|---|---|---|
| `SO.01`, `SO.04` (spectating) | *"His people are part of his world … their presence is company"* + the observed-not-scripted deal (GOAL.md:139-146, 172-183) | serves |
| `SO.02` ("I'm cold" is true when he is cold) | *"VOICE — he must be able to make sound"* (:43) and interoception (:41-42) | serves |
| `LF.02` (a life survives `kill -9` bit-exactly) | *"He lives, he dies, he remembers … death is not a reset; it is a page turn"* (:103-116) | serves |
| `T0.33`, `T0.34`, `6e3ad9a` (CPU accounting) | *"protects the honesty of watching what happens when the three meet"* (:8-9) | serves — thinly, and see below |
| the detached cpu<2h certificate sweep | same | serves |

Nothing is drift. **But the shape is worth naming plainly:** the last two
iterations and this morning's third built *meters for compute the ladder is not
spending*, because `coverage` reports **0 FRESH dispatches** at every one of
the seven cost classes. The builder did not choose plumbing over science; the
board offered it nothing else, and it said so in the journal both times before
picking up the machine-improvement duty. That is the honest response to a
blocked board. It is still a day where the green-tick count moved and the apple
did not.

### 3b. Which parts of GOAL.md have no passing spec — and four that got worse

Unchanged and standing: **one brain / unison** 25 specs, **1 pass**;
**curiosity** 12 specs, **2 pass**; **fast/slow** 8 specs, **0 pass**;
**sleep** 5 specs, **0 pass**; **plasticity** 4 specs, **0 pass**;
**hunger/thirst** 6 specs, **0 pass**; **death & retry** 4 specs, **0 pass**.
These are the claims GOAL.md calls the thesis, and they are exactly where they
were yesterday.

One item that has degraded and that the 67th audit dropped:
`coverage` reports **4 NEW `CITED-BUT-UNRUNNABLE`** ids —
`GEN.02`, `GEN.03`, `GEN.06`, `GEN.09` — against `GOAL_UNRUNNABLE_BASELINE`
of `{DP.02, DP.03, LC.04}`. The timeline is the finding:

- `2026-09-01 10:14` (`7f1e875`) — Review 08-31 item 6 registers the four,
  closing *"the constitution's four dangling citations … 23 days"*.
- `2026-09-01 22:10` (`2b832ed`) — `LC.07` harvests BRANCH B and goes
  `PILOT-BLOCKED`. All four new specs are welded behind it.

**Twelve hours.** A repair aimed at dangling citations converted four of them
into ids that resolve to corpses — which `coverage`'s own text calls strictly
worse (*"An id that resolves to a corpse is a worse dangling reference than one
that resolves to nothing"*). The existing routed row
`goal-cites-four-specs-that-resolve-to-corpses` (routed 09-02, **DUE 09-10**,
re-armed from 09-06 with a reason) names the *earlier* four and is explicitly
downstream of `lc07-checkpoint-branch`. It should be widened to name all seven
rather than a second row opened — see **B5**.

## §4 — IS THE BUILDER ALIVE AND PRODUCTIVE: yes, no findings

**24 iteration starts in the last 24 h, 24 ends, 24 `rc=0`.** Demonstrated
**95 → 102** (+7). No paused loop, no credit exhaustion, no load abort, no
repeated identical failure. The loop ran on Fable throughout with the gate
meter (`week:all models`) reading 29% → 48% across the day and the per-model
Fable line at 84–87% — the loop printed both and named the one it acted on
every slot, which is the standing rule being followed correctly.

Every one of the 67th audit's five findings was executed: `LF.01`'s VOID
harvested (`633b5bb`), the cause/lane disagreement instrumented without a
re-run or a gate move (`54d7841`), the integrity-death routed as a queue row
(`01f2899`), and the detached-run claim made machine-checkable (`b20d28c`).
20 commits since the last audit closed.

## §5 — COMPUTE HONESTY

**GPU.** The budget keys weeks by `%U` (Sunday-start), matching Kaggle's actual
reset — so the current week is `2026-W35`, running Sun 08-30 → Sat 09-05.

| week | spent | wasted (failed jobs) |
|---|---|---|
| W32 | 16.61 h | 1.18 h |
| W33 | 7.89 h | 0.26 h |
| W34 | 1.62 h | 0.00 h |
| **W35 (current)** | **19.20 h of 30** | **0.00 h** |

**0 overruns, 0 wasted hours this week, and every charged job maps to a
recorded pilot or run.** `10.80 h` remain and expire at the Sunday 09-06 reset.
`coverage` says all three GPU classes are `NOT FILLABLE` — `gpu<20min` blocked
on `DP.04`/`SM.03`, `gpu<2h` on `T2.11`, `gpu<8h` on `LC.07` — each a redesign,
not a dispatch. So the expiry is diagnosed, not mysterious, and it is the
fourth consecutive week of partial expiry. Not a builder failure; a
consequence of every GPU arena being pilot-blocked at once.

**CPU.** See findings 1–5. The meter itself is three hours old; its numbers are
inflated ~1.7× and it has produced no ledger consequence *yet*.

## §6 — STUCK DECISIONS: no findings

`decisions --check` is clean: 0 MEANS-ESCALATED, 0 UNDECLARED, 0 OVERDUE,
ratchet 0/10. Every open decision is armed with a default and a date. Nothing
on the owner's desk today has enough evidence for me to resolve, nothing is
blocked that a bakeoff could settle, and I found no owner-decision acted on
without a record — `D13`'s and `D14`'s firings are both transcribed in
`DECISIONS_RESOLVED.md` with losers, evidence and a reversal path.

`D15` and `D16` fire tomorrow (09-05) if unanswered. Both defaults pick only
already-permitted actions and both leave a red visible rather than green
(`D16` explicitly keeps `T0.27` RED because the party proposing the alternative
is the party it would exonerate). Correctly armed; no action from me.

## §7 — BAKEOFF HYGIENE: no findings

`DECISIONS_RESOLVED.md`'s recent entries each carry a learning gate, named
losers, an evidence list and a reversal. No VOID is treated as a verdict *in
the resolved file* — the one place a VOID still stands in for a verdict is
`CHAMPIONS.md`'s `Learning core` seat (`LC.03=VOID`), and `champions --check`
reports it as `VERDICT-IS-A-VOID` at baseline, routed, and on the owner's desk
as D10 since 08-24. No winner was chosen inside a noise margin this week; the
one candidate — `T3.09` attempt 3, where the shuffled control cleared the
claim's own margin (`shuf_gain +12.47` vs `MARGIN_AFF 11.0`) — was recorded
**FAIL** and routed as `t309-control-clears-the-claims-own-margin`, which is
the correct handling.

## §8 — THE HONEST SUMMARY

**Are we closer to a curious humanoid that climbs the ladder than yesterday?
Partly, and less than the +7 suggests.**

Genuinely closer: he can now be *watched* without it changing him (`SO.04`,
bit-identical over 2000 steps), he can *cry truthfully* when he is cold
(`SO.02`, 3-class decode 1.000 vs 0.430 base, with a swap control that
misleads), and a life of his can now be **killed and resumed bit-exactly**
(`LF.02`). That last one is not plumbing: *"death is not a reset; it is a page
turn"* needs a page that survives, and now one does.

Not closer: **one brain / unison sits at 1 pass in 25 specs**, curiosity at 2
in 12, plasticity/sleep/fast-slow at 0. Four commitments are CLAIM-DEAD. Every
cost class offers **0 fresh dispatches** — four "dispatchable" specs are all
VOIDs needing repair, three classes have no path in at all, and one is honestly
held by `D19`. The board is not blocked because anyone is slacking; it is
blocked because `W0` is too shallow to buy the claims the ladder wants, and
**that answer is due on 09-06 along with seven other promises, against a desk
that has demonstrably discharged one row per cycle.**

So: a good day of work, an honest builder, a clean ledger — and a ladder whose
next rung is a world redesign that eight dated rows are all waiting on at once.
The apple did not move. It will not move until 09-06 is either paid or
re-dated, and re-dating eight rows the day they come due is how a queue learns
that dates are decorative.

---

## FOR THE BUILDER

Ranked. **B1 and B2 are the same defect and should land in one commit.**
Do not raise `CPU_DAY_CEILING_S` in any of these — it is a tenant protection on
a box with paying customers, and widening it is not a repair available to this
loop.

**B1 — stop double-billing the detached lane.** `cpu_budget._wrap` charges the
whole tree's wall clock under `label`; every `run_spec` grandchild also charges
itself through `run.py:_bill_cpu`. Live proof in the file right now
(`experiments/cpu_budget.json`, day `2026-09-04`):
`detached:gate_sweep_cpu2h.log` 1200.00 s alongside `T2.20` 33.99 + `ME.5`
61.01 + `UB.9` 207.84 + `PG.4` 504.80 + `PG.6` 553.04, every one of those five
a descendant of pid 965942 per `declared_pids`. Pick one accounting owner —
either `wrap` bills only the residue the children did not claim, or `wrap`
exports an env marker that `_bill_cpu` reads and skips (the label then carries
the un-attributed remainder), or the wrapper's label is written as
non-charging metadata. Whichever you choose, say in the docstring which
process owns the charge and why.

**B2 — `T0.34` needs a property that reaches the boundary that broke.** The
existing `conservation_ok` tests conservation *inside one* `bill_interval`
call, and the end-to-end property runs `sleep 17`, which has no billing child.
Add a property that runs a real (or a stub) `run_spec`-shaped payload under
`wrap` against a temp budget and asserts `used_s` after ≈ the tree's true wall
clock within one heartbeat — i.e. that the wrapper's charge and its
descendants' charges are **disjoint**. A control that double-bills must fail
it.

**B3 — `T0.33` property 5 cannot fire, and should report a number instead of
gating at zero.** `spec_child_timeout_seconds(s) > CPU_DAY_CEILING_S` is false
for every registered spec by construction, so `cpu_foreclosed == []` is
vacuous. The docstring is honest that it means "*from a fresh day*", so this is
a gap, not a false claim — close it by *also* recording, as a metric rather
than a gate, `n_foreclosed_now` = the count of registered runner-lane cpu specs
whose `est` exceeds the **live** `remaining_s()` at certificate time. Do not
gate it at zero: a legitimately-spent day *should* refuse things. Report it so
a foreclosed day stops being invisible.

**B4 — a refusal must leave a trace.** `gate_cpu_child` returns UNRECORDED by
design (correct — tenant protection is not a measurement of the spec), which
means a day that refuses 53 specs produces no FAIL, no VOID and no number
anywhere. At minimum: `ladder_loop.sh` (or `_run_isolated`) should log
`cpu-refused <spec> est=<n> remaining=<n> load=<n>` to `ladder.log`, and
`run status` should print today's refused set. This is a print, not a
threshold.

**B5 — widen the existing citation row rather than opening a second.**
`goal-cites-four-specs-that-resolve-to-corpses` (DUE 09-10) names the earlier
four; `coverage` now reports **seven** `CITED-BUT-UNRUNNABLE`, four of them NEW
(`GEN.02/03/06/09`, welded behind `LC.07` since `2b832ed`, twelve hours after
`7f1e875` registered them). Amend the row to name all seven and to record that
the 09-01 repair *grew* the class it was closing — that causal note is the part
worth keeping. **Do not** add `GEN.*` to `GOAL_UNRUNNABLE_BASELINE`; the
baseline is shrink-only and `coverage` says so in its own output.

**B6 — route finding 5 (`cpu<48h` self-foreclosure) as a REVIEW_QUEUE row, do
not fix it inline.** `rtf.BUDGET_SECONDS["cpu<48h"] = 172800 s` against a
57600 s/day ceiling means one legal detached run overruns the day by
arithmetic and then closes the runner lane too. The fix touches what the
ceiling *counts* (wall seconds vs core-seconds on a 4-core box), which changes
what the tenant protection protects — that is a threshold question and belongs
on the desk, not in a builder commit. Route it with the arithmetic above and
`LC.03` v2's ~190 core-hours as the worked example, and give it a `DUE:` that
is **not** 09-06 (see B7).

**B7 — the 09-06 pile is now 8 rows against a measured capacity of 1/cycle.**
`review-queue` has flagged it amber for three audits and it grew again today.
The honest repairs are the ones the tool names: act, decline, or **re-arm with
a new `DUE:` and a reason**. Re-dating on 09-06 itself is the failure mode;
re-date the ones that can wait *before* they come due.

---

## FOR THE OWNER

**1. Nothing needs you to unblock the loop today.** The decision desk is fully
armed (0 undeclared, 0 overdue, ratchet 0/10) and `D15`/`D16` fire tomorrow
with defaults that both leave a visible red rather than a green. No action
required unless you want to rule early.

**2. One question is genuinely yours: what should the CPU day-ceiling count?**
`CPU_DAY_CEILING_S = 57600 s` is charged in **wall clock**. A `cpu<48h`
detached run legally occupies one core of four for two days and would bill
86400 s into a 16 h bucket — overrunning the day and then refusing every other
CPU spec on the box. Counting **core-seconds against 4 cores** would fix that
cleanly. I am not proposing it and no default here may fire it, because any
answer other than *"wall clock stands"* relaxes a protection on a box with
paying tenants, and that call is yours. Routed to the Review as **B6** with the
arithmetic attached. Note for context: the builder built this meter *itself*,
unprompted, to protect your tenants — the defect is in the composition of two
correct pieces, not in the intent.

**3. ~10.8 h of free Kaggle GPU expire at the Sunday 09-06 reset, for the
fourth consecutive week.** The cause is now fully diagnosed rather than
mysterious: every GPU arena is pilot-blocked behind a redesign at the same time
(`DP.04`, `SM.03`, `T2.11`, `LC.07`). Nothing here is waste through
carelessness — 0.00 h were burned on failed jobs this week — but it is a real
recurring cost of the board being blocked, and it will keep recurring until
the `w0-too-shallow` design lands.

**4. Eight dated promises come due on 09-06 against a desk that discharges one
per cycle.** That includes `w0-too-shallow` — the world redesign that §8 says
the whole ladder is waiting on. If you want one thing prioritised on Sunday,
that is the one; the other seven are downstream of it or can be re-dated
honestly.
