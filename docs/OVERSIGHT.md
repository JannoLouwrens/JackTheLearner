# OVERSIGHT — 42nd audit, 2026-08-28 12:45 UTC

## VERDICT: DRIFTING — **`week:Fable` rose 66% → 100% during 72 hours in which not one Fable request was made anywhere on this box.** The last `claude-fable-5` call in any transcript is `2026-08-25T12:23:27.661Z` — the same second the builder's final iteration ended. `D14`'s armed default is justified by that meter, fires in 3 days, and would refuse every builder iteration from midweek onward, permanently.

**State.** `HEAD` is `95d39c8` (today's Review). **Zero builder commits since
2026-08-25 10:14:58**; last builder iteration ended **2026-08-25 12:23:33 —
72.4 hours ago**; **72 consecutive `PACING: … skipping` slots and nothing else**
in `ladder.log`. **84 PASS / 187 registered (44.9%)**, unmoved for 9 days. The
untracked `experiments/tests/sm_03_nose_reports_occluded.py` (710 lines) is still
the only thing in the working tree. Meters at 12:07: `week:all models` **72%**
(the gate) at **61%** of the week, line **65%**; `week:Fable` **100%**. Both
reset **2026-08-31 05:00 UTC**. Kaggle W34: **0.3111 h** charged, **29.6889 h**
expiring **Sun 2026-08-30 00:00 UTC — 35 hours from now**.

**The three constitutional gates are green, and I re-ran all three myself.**
`coverage` exit 0: 0 commitments with no declared spec, 0 CLAIM-DEAD, 4 known
dangling GOAL citations at baseline. `decisions --check` exit 0: **0/10
undeclared**, no `MEANS-ESCALATED`, no `OVERDUE`. `champions --check` exit 0:
ratchet **6/8** — it shrank by two since the 40th, by registering the `LG`
family, which is the correct direction and the correct mechanism.

---

## RANK 1 — `week:Fable` is not measuring Fable, and an armed default rests on it

`D14` — raised by the 34th audit, reasserted by four audits since — says the
builder's own model meter is the real constraint, and that the burn comes from
outside this project:

> *"all 12 points of Fable burned in the six hours to 12:07 came from outside
> jackthelearner, with the builder at zero iterations."*

**I checked that against the request log rather than the percentage.** Every
Claude request on this box writes an assistant record carrying its `model` and
`usage` into `~/.claude/projects/*/*.jsonl`. Summed across **all** project
directories, with no threshold:

| date | `claude-fable-5` output tok | `claude-opus-5` output tok |
|---|---|---|
| 2026-08-24 | 1,831,575 | 805,990 |
| 2026-08-25 | 800,639 | 583,033 |
| 2026-08-26 | **0** | 564,334 |
| 2026-08-27 | **0** | 593,308 |
| 2026-08-28 → 12:44 | **0** | 471,138 |

**The last Fable request anywhere on this box is `2026-08-25T12:23:27.661Z`** —
`ladder.log` records `iteration end rc=0` at `12:23:33`, six seconds later. There
has not been a Fable request since, from this project or any other.

Yet across that silence the meter climbed, in lockstep with the shared one:

| time | `week:all models` | `week:Fable` | Fable requests in window |
|---|---|---|---|
| 08-25 13:07 | 38% | 66% | 0 |
| 08-25 22:07 | 47% | 77% | 0 |
| 08-26 04:07 | 52% | 86% | 0 |
| 08-26 10:07 | 55% | 93% | 0 |
| 08-26 16:07 | 62% | **100%** | 0 |

**At least 34 of Fable's 100 points were added with zero Fable requests
recorded.** Two explanations survive, and I cannot separate them from inside
this repo:

- **(a)** `week:Fable` is not an independent spend meter — it tracks the shared
  pool, offset by this project's genuine Fable spend of 08-24/08-25. The
  correlation above is exact and monotone, which favours this.
- **(b)** a consumer with no transcript on this box (a container, another
  machine) uses Fable and only Fable. Possible; it must also have stopped being
  visible at the moment our builder did.

**Why this is RANK 1 rather than trivia.** `D14`'s pre-registered default is
armed and fires **2026-08-31**:

> *"Option (b) — the LOUD REFUSAL … a pre-flight check in `scripts/ladder_loop.sh`
> before `run_claude`, at a 95% floor on the loop model's own weekly line. This is
> a NARROWING and only a narrowing: it refuses strictly more than the 90% stop
> already refuses."*

That description is exactly correct and is exactly the problem. Under branch
**(a)**, Fable's line is pinned by consumption the builder does not make and
cannot reduce; the tracking offset puts Fable at ~95% whenever all-models is
near ~65%, which is where we are on a Friday. A ≥95% pre-flight would then abort
**every** iteration from roughly midweek onward, every week, and it would do so
*before* `pace_gate` — converting a 72-hour outage into a standing one. Under
branch **(b)**, the same default hands an unnamed external party a unilateral,
silent, permanent off-switch for this project's only productive organ.

**Both branches make option (b) unsafe, and the entry contains no measurement
that distinguishes them.** This is not an argument that the default is wrong on
the merits — it is that the evidence attached to it does not say what it is read
as saying. Recorded as an evidence update in `docs/DECISIONS_NEEDED.md`; I have
**not** touched `D14`'s options, default, or `decide_by`, which are not mine to
move.

---

## RANK 2 — the audit organs are now the entire consumption of the meter that gates the builder

The 38th audit found that `pace_gate` has one call site and the Opus organs are
ungated. That was a structural argument. Here it is as a measurement.

On **2026-08-26** the builder ran **zero** iterations. Total recorded output
across every project directory that day: **564,334 tokens, 100% `claude-opus-5`,
100% inside `-home-opc-jackthelearner`.** The crontab has exactly five Opus
organ runs a day (`overseer.sh` at `37 */6`, `review.sh` at `37 6`). **All of
that day's burn was the overseer and the Review.** 08-28 to 12:44 is the same
shape: 471,138 tokens, all in-project Opus, builder iterations zero.

Converting against the gate's own units — `week:all models` rose ~5.8 points/day
on ~530k output tokens/day:

- **~1.06–1.15 points per organ run** (this refines the 41st's ~1.4 estimate,
  which the Review flagged as repeatedly falsified; the token side gives it
  directly rather than by subtraction).
- The pace line recovers at **0.387 pts/h = 9.29 pts/day**.
- **The audit organs consume 62% of the pace line's own daily recovery rate.**

Five audits a day, observing a builder that five audits a day are the reason
cannot run. The 08-26 column is the clean experiment: the only consumer was the
instrument.

**The forecast, recomputed from the 12:07 reading.** At the measured 7.0 pts/day
the gate releases at **2026-08-31 02:38 UTC** — and `usage_gate`'s 90% hard stop
fires at essentially the same instant (pct 90.1). At the full 72-hour rate
(11.5 pts/day) the 90% stop fires first, **2026-08-30 02:38**, and the gate never
opens at all. Only at ≤5 pts/day — below anything observed — does it open
earlier, 08-30 01:38.

So the sharpest available statement is no longer a date, it is a structure:
**`pace_gate` will hand back the budget it conserved with at most 2.4 hours of
week remaining**, into a meter that does not roll over, having spent the week
holding it for a builder that got ≤2 of 168 slots. The margin the 41st audit put
at 8 hours is, one reading later, **2.4 hours**. It has shrunk every reading.

---

## The audit, section by section

**§1 — Integrity of the ledger: CLEAN.** 96 rows, **84 PASS**. Every PASS row
resolves to a spec in `BY_ID` (0 orphans). Every PASS `commit` still exists in
git (0 missing, checked with `git cat-file -e`). Every PASS has an
implementation file in `experiments/tests/` (0 missing). Two PASSes declare no
`control` — `T0.01` *"Repo imports clean"* and `T0.10` *"Kaggle job round-trip"* —
both existence claims where a control is not meaningful, the standing §1.2 note,
unchanged and correct.

**§2 — Thresholds and controls over 7 days: CLEAN, no silent loosening.** The
window holds 18 commits, all 08-24/08-25. Every registry change is an *addition*
carrying its own `control=` / `null_baseline=` / `falsified_by=` (`f0cb81d`
SH.02+SM.03, `ed2d969` the LG family, `50baf1d`/`20b8660` T2.15). Two hunks read
as deletions and neither is a loosening:

- `0345f0d` removes `falsified_by="… That arm is EXCLUDED from LC.03/LC.04 — not
  scored and beaten, excluded"`. That **widens** what may be refuted, under the
  owner's 2026-08-24 ruling now written into `SYSTEM.md`. A premise became
  contestable; nothing became easier to pass.
- `b624d78` removes `N_PROPERTIES = 10` — `T0.21` grew to 11 properties and its
  P6 stopped hard-coding `SH.01`. Strengthening.

No threshold moved loose, no control deleted or weakened, no `_check` gained an
`or`, no seed count cut, no assertion removed. The most recent claim result in
the window is `T2.15`'s **FAIL** — held-out [8,9,5]/16 against a 12/16
pre-registered bar, recorded as a FAIL with the mechanism localised rather than
re-tuned. That is the ladder working.

**§3 — Drift: none, because there was no work.** The builder worked on nothing
in the last day, and nothing in the three days before it. That is absence, not
drift, and it should not be scored as compliance. The converse question is the
one that bites: of 23 commitments, **14 carry live claim specs with nothing
passing**. Named specifically because GOAL.md names them: **curiosity 12 specs /
1 pass**, **one brain / unison 21 specs / 1 pass**, **fast-slow 8 specs / 0
pass**, **touch, tool use, proprioception, plasticity, sleep, social 0 pass
each**. The ladder is honest about this — it is not hiding it — but four days of
green gates over a static 84 is the exact shape of *"a longer list of green
ticks"*.

**§4 — Is the builder alive? No.** 72 iterations attempted in 72 hours, 72
skipped by `pace_gate`, 0 reached `run_claude`, PASS delta **+0**.
`lost_iterations.log` is **0 bytes** — correctly, since a pace skip is not a
lost iteration. Nothing is paused, nothing crashed, no credit exhaustion was
hit. The loop is healthy and doing exactly what it was told.

**§5 — Compute honesty: spend is clean, non-spend is the waste.** Every charged
job in `gpu_budget.json` maps to a real outcome. W34's only charge — **0.3111 h**
for `T2.15` — produced a real ledger row (a FAIL), with both receipts committed.
**No GPU hours were spent without a ledger entry to show for them.** The failure
is the other direction: **29.6889 of 30 Kaggle hours expire 2026-08-30 00:00
UTC**, 26.6 hours before the earliest credible gate release. That is the fourth
consecutive week: **8.82 → 22.11 → 29.69**, now ~60.6 hours cumulative, on a
project whose standing rule is free compute only.

**§6 — Stuck decisions.** 11 armed, all `decide_by: 2026-08-31`, ratchet 0/10
undeclared. Nothing is `MEANS-ESCALATED` — no fork a measurement could settle is
sitting on the owner's desk today. Nothing is `OVERDUE`. Nothing was quietly
acted on without record. The 41st's four defective defaults (D8, D10, D3, D4)
remain defective and remain builder items — unactioned solely because no builder
has run. **`D13`/`D14`'s deadline still falls after the harm it is armed
against** (39th audit's finding, now 35 hours from the harm), and my RANK 1 adds
that `D14`'s evidence does not support its default.

**§7 — Bakeoff hygiene: CLEAN.** `DECISIONS_RESOLVED.md` shows VOID handled
correctly throughout: `PS.01/J` returned VOID because three arms fell below the
3.0σ learning gate and was **not** converted into a winner; `PS.01/J2` re-ran and
produced `impact_speed` on a real margin; `D2` established by ledger replay that
a VOID **blocks** its dependents, with the property made executable as `T0.08`
P6. No decision was made without a learning gate, no VOID was treated as a
verdict, no winner was chosen inside the noise margin.

**§8 — The honest summary.** No. We are not closer to a curious humanoid than we
were on 2026-08-25, and we are not closer to a longer list of green ticks either
— the list has not moved in nine days. What has grown is the quality of the
description of why it has not moved: four audits and two Reviews in 72 hours,
~1.6M output tokens, producing an increasingly precise account of an outage that
each one made ~1.1 points worse. The instruments are excellent and they are now
the load. The single most valuable thing that could happen to this project this
week is for the audit cadence to drop and one builder iteration to run.

---

## FOR THE BUILDER

**B1. `D14`'s default must not fire as written. RANK 1.** Amending a default
*toward* safety is a tightening the ratchet permits. The measurement is in
`DECISIONS_NEEDED.md` under today's evidence update. Minimum repair: gate the
pre-flight on **`max(all models, loop model)`** (D14's own option (c)) rather
than on the loop model alone — it is equally a narrowing, it is monotone against
the 90% stop, and it cannot be pinned by a meter that moves without requests.
Do **not** ship a bare ≥95% floor on `week:Fable`.

**B2. Before trusting any per-model line, verify it against the request log.**
A ten-line check, and it is the guard that would have caught this: sum
`output_tokens` per `model` per day from `~/.claude/projects/*/*.jsonl` and
assert that a per-model percentage **cannot rise across a window with zero
requests for that model**. Ship it beside `claude_usage.py` and print the
request count next to every per-model percentage the loop logs. `pace_gate`
currently prints `week:Fable 100% (not the gate)` hourly; make it print
`week:Fable 100% (0 requests in 72 h)`.

**B3. `SY.01`, the three-arm pace-gate bakeoff** — still unwritten after five
audits ordered it, and now with the arms measurable. **A** gate as shipped;
**B** `JACK_NO_PACE=1`; **C** `pace_gate` added to `overseer.sh` / `review.sh` /
`field_watch.sh` beside the `usage_gate` line each already has. Today's §RANK 2
gives arm C its predicted effect directly: the organs are 62% of the line's
recovery rate, so gating them is worth ~5.8 pts/day of builder wake-time. Rule 3
governs; this is not an escalation.

**B4. Carry forward, unchanged and unactioned because no builder ran:** the
41st's B0–B2 and B4 (read the eleven defaults in full; fix D8 before it fires;
make `decisions.py` read and print the field it certifies; `OVERDUE` as a
violation with a `fired:` marker), the 40th's B1/B2 (`champions.py` must ratchet
`NO-ARENA` and the sum; `decisions.py` must ratchet `NO-DEFAULT`), and `T0.23` as
the ladder-side sibling of `T0.21` covering all of it.

**B5. Commit `experiments/tests/sm_03_nose_reports_occluded.py`** — named
pathspec, per the `add -A` ban — and **re-run the pilot for real numbers before
freezing any gate.** Third audit asking. The 08-25 handoff said the pilot was
"running full-size on seed 90 (pid 1552865, ~667 MB, healthy)"; pid 1552865 is
gone, and **no artifact exists anywhere under `/data` newer than 08-25**. The
numbers in that handoff were never produced.

**B6. `rc=0` must stop meaning "I launched something."** Fifth sighting, and B5
is today's instance. Assert the artifact is non-empty ~10 s after a detached
launch and record the assertion in the handoff line.

## FOR THE OWNER

**One thing is new, and it changes an entry already on your desk.**

`D14` asks you to rule on which meter should govern the builder, and its
supporting evidence says the builder's own Fable line is being burned by
consumption from outside the project. **Measured from the request logs, no Fable
request has been made anywhere on this box since 2026-08-25 12:23:27** — the
second the builder stopped — while the Fable percentage rose from 66% to 100%.
Whatever `week:Fable` is reporting, it is not this week's Fable spend.

You do not need to rule on that. What matters is that **`D14`'s default fires on
2026-08-31 and installs a pre-flight abort keyed to that meter.** If the meter
tracks the shared pool (which the data favours), that abort refuses every
iteration from midweek onward, permanently — a strictly-narrowing guard,
correctly described as such, resting on a number that does not mean what it is
read to mean. Builder item **B1** repairs it before the date by switching to
`D14`'s own option (c), `max(all models, loop model)`, which is equally a
narrowing and cannot be pinned this way. **One line from you if you would rather
option (b) stand as written.**

**The standing ask is unchanged and the clock is now 35 hours.** Rule `D13`/`D14`,
or say the free hours may go. `D14`'s option (d) — a bounded `JACK_NO_PACE`
window — remains the only option that saves this week's GPU hours, and it remains
yours alone because it suspends a throttle and no default may widen what is
allowed.

**The number that should set the priority:** **8.82 → 22.11 → 29.69** free
GPU-hours expired unspent in three consecutive weeks, **~60.6 hours**, on a
project whose standing rule is free compute only — while in the same window the
watching organs spent ~1.6M output tokens describing it.
