# OVERSIGHT — 44th audit, 2026-08-29 00:45 UTC

## VERDICT: DRIFTING — **the pacing organ, built on 2026-08-24 to stop free GPU hours expiring unspent, is presiding over the worst such loss on its own table, and it is the sole cause.** Across 84 consecutive hourly slots the meter never once reached 71%, let alone the 90% stop; every one of those 84 iterations was refused by `pace_gate` alone. W34's **29.6889 of 30** free Kaggle hours expire in **23.2 hours**, and on the observed trajectory the builder cannot wake for **52**.

**State.** `HEAD` is `ba087a9` (the 43rd audit). **Zero builder commits since
2026-08-25 10:14:58.** Last builder iteration ended **2026-08-25 12:23:33 —
84.4 hours ago**. **84 consecutive `PACING: … skipping` slots** and not one other
line in `ladder.log` since `08-25 13:07`. **84 PASS / 187 registered (44.9%)**.
`experiments/tests/sm_03_nose_reports_occluded.py` (32 KB, `Aug 25 12:20`) is
still untracked and still the only thing in the working tree. Meters at 00:07:
`week:all models` **73%** (the gate) at **69%** of the week, line **70%**;
`week:Fable` **100%**.

**The three constitutional gates are green and I re-ran all three.** `coverage`
exit 0 — 0 commitments with no declared spec, 0 CLAIM-DEAD, 4 known-dangling
GOAL citations at baseline (`GEN.02/03/06/09`). `decisions --check` exit 0 —
**0/10 undeclared**, no `MEANS-ESCALATED`, no `OVERDUE`; nothing to arm.
`champions --check` exit 0 — ratchet **6/8**, 12 violations, byte-identical to
the 43rd audit because no builder commit has landed since.

---

## RANK 1 — the pace gate is now the *cause* of the outage it was built to prevent, and the loss is worse than either baseline week

`scripts/lib_usage.sh:34-60` states the organ's purpose and its own
pre-registered success criterion, in its own words:

```
#     week   loop went dark        dark for   Kaggle GPU-h expired unspent
#     W32    Fri 08-14 15:07          ~4.5 d   8.82 of 30
#     W33    Fri 08-21 12:07          ~2.7 d  22.11 of 30
#
#   PACE_FLOOR=25 buys the week's opening burst; by Friday the line is ~62%,
#   Sunday ~81%, and the loop is still awake when the GPU quota expires.
```

**The first full week under the fix, measured, appended to its own table:**

| week | loop went dark | dark for | Kaggle GPU-h expired unspent |
|---|---|---|---|
| W32 (pre-fix) | Fri 08-14 15:07 | ~4.5 d | 8.82 of 30 |
| W33 (pre-fix) | Fri 08-21 12:07 | ~2.7 d | 22.11 of 30 |
| **W34/35 (post-fix)** | **Tue 08-25 13:07** | **≥5.7 d** | **29.6889 of 30** |

Every column is worse. The blackout began **two and a half days earlier in the
week** than the Friday disease the organ was written to cure, and will cost
**3.4× W32's** and **1.34× W33's** expired GPU hours.

**It is not the 90% stop doing this.** I extracted all 84 skip lines. The
maximum `week:all models` reading across the entire blackout is **73%**.
`usage_gate` returns 0 below 90 unconditionally — so **all 84 iterations would
have run under the owner's rule alone.** `pace_gate` is the whole of the
refusal.

**And the line is losing the race.** From the log, endpoint-exact:

```
2026-08-25T13:07  meter 38   line 38     <- blackout begins, meter touches line
2026-08-26T00:07  meter 51   line 42
2026-08-27T00:07  meter 62   line 51
2026-08-28T00:07  meter 68   line 61
2026-08-29T00:07  meter 73   line 70
```

Over 84 slots / 83.0 h the **meter rose 35 points and the line rose 32**. The
gap did not close — it **widened from 0 to 3**. Least-squares over all 84 points:
meter **0.3753 pts/h**, line **0.3876 pts/h**; the builder's structural headroom
is **+0.0123 pts/h**, so a 3-point gap needs **243 hours** to clear. The weekly
meter resets at **2026-08-31 ~04:12 UTC — 52 hours away.** The builder therefore
does not run again this week.

**Kaggle resets on a different clock.** `Budget._week()` uses `%U`
(Sunday-start), so W34 closes **Sun 2026-08-30 00:00 GMT — 23.2 hours from
now**, **29 hours before** the earliest possible builder wake. `remaining()`
confirms **29.6889 h** unspent against one 0.3111 h charge.

**Who spent the 35 points is unknown, and that is the second half of the
finding.** The builder ran zero iterations across the window, so ≥35 of the
65-point pace band (`PACE_CAP 90 − PACE_FLOOR 25`) went to something the gate
does not meter. Two known contributors, neither measured:

- **The three Opus oversight organs are ungated by design.** `overseer.sh:45`,
  `review.sh:29`, `field_watch.sh:31` each call `usage_gate` and **not**
  `pace_gate`. Fourteen overseer audits and three Reviews ran inside this
  blackout, producing six commits and zero specs.
- **The pool is shared with the owner's interactive sessions** —
  `lib_usage.sh:47` says so explicitly.

`ladder_loop.sh:174` justifies the asymmetry by **run count**: *"Builder ONLY:
it is ~82% of all organ runs (168/wk against the overseer's 28, review's 7,
field watch's 1), so pacing it captures nearly all the benefit."* That
reasoning counts runs, not tokens. A pace-skipped builder slot costs four
`claude -p /usage` reads (`usage_gate`, then `--week-elapsed`, `--pct` and
`--model Fable --pct` inside `pace_gate`); a full Opus audit costs a session.
The measurement above refutes the premise: with the builder at 82% of runs and
**0% of spend**, the pool still consumed 54% of the entire weekly band.

**No instrument in this repo attributes meter spend to an organ.** I looked.
That is why this took 44 audits to see: the gate reads one aggregate number and
converts it into builder downtime one-for-one, and nothing anywhere records who
moved it. The pacing docstring's own diagnosis — *"the loop is stopped by
consumption it does not control, and being the only consumer with a gate, it is
the one that starves"* — is still exactly true. The fix did not address it; it
sharpened it, because a line at 25%+ starves earlier than a ceiling at 90%.

---

## RANK 2 — the 43rd audit's B1 is unimplemented, so the contestability check still cannot see an undefended seat

Restated, not re-litigated: `champions.py:317` tests
`all(v == "NOT_RUN" for v in s["arena_status"].values())`, so one arena spec
having run — including the **incumbent's own arm** — discharges the whole seat.
`Learning core` (BY DEFAULT, cell reads *"DEFAULT, never defended"*) prints `ok`
on `LC.00`/`LC.01`/`LC.02`/`PS.01` — a decidability precondition, an admission
rule, a feasibility gate and a fixture — while `LC.03` is VOID and `LC.04`, the
cell's own *"actual match"*, is NOT_RUN. `Episodic retrieval` is held **BY
VERDICT** on `ME.11.A` PASS, which the registry calls *"lexical containment, the
incumbent, as the null"*; `ME.11.B–F` have never been built. The tool's
`_fixture()` still asserts the false negative as must-pass.

**New this audit:** the builder has had 84 hours of wall-clock and zero
runnable hours to fix it, and cannot get any before the meter reset. B1 is
carried forward verbatim and is now sequenced *behind* RANK 1, because a repair
nobody can run is not a repair.

---

## RANK 3 — `SM.03` has been untracked for 4.5 days and its pilot artefact does not exist

`experiments/tests/sm_03_nose_reports_occluded.py`, 32086 bytes, mtime
`Aug 25 12:20`. It is the only runnable claim spec for the `smell` commitment
and the only registered `GPU_SHORT` candidate for the 29.69 hours expiring
tomorrow. I re-confirmed both facts the 43rd audit reported:
`/data/sm03_pilot_seed90.json` does not exist, and `pid 1552865` is gone. The
iteration that logged `rc=0` on a promised pilot is 4.5 days old and the promise
was never kept. While untracked the file is invisible to `dispatch.sh`'s
`--untracked-files=no` push guard and one `git clean` from gone.

---

## The audit, section by section

**1. Integrity of the ledger — clean, all 84 re-checked from scratch.** 84 PASS
/ 9 FAIL / 3 VOID; 96 ledger rows, **0 ids absent from `BY_ID`**. Every PASS's
`commit` resolves in git (**84/84**). **82/84 declare a control and carry
recorded `control_metrics`**; the two that do not are `T0.01` and `T0.10`,
Tier-0 harness specs. I added a check no prior audit ran: **no PASS has
`control_metrics` identical to its `metrics`** (0/82), and **55 of 82 share no
metric key at all** between experiment and control — the controls are separate
computations, not the same code path relabelled. `run status`: 1 stale-by-content
claim (`T2.02`, a VOID), 27 pre-`impl_sha` entries verified byte-identical, 0
unanswerable. **No findings.**

**2. Thresholds and controls over 7 days — no findings, and it is a real
result.** The last commit touching `registry*.py` or `experiments/tests/` is
`ed2d969` on 2026-08-25; nothing in four days. I re-scanned all sixteen commits
in the window. The one apparent removal —
`- depends_on=["DP.00", "VO.01"]` in `ed2d969` — is a **tightening**: the line
was reflowed to `depends_on=["DP.00", "VO.01", "LG.00"]`, adding a dependency,
exactly as the commit message states. Everything else moves in the tightening or
neutral direction with a measurement in its message (`20b8660` added a control
declaration after `UndeclaredControl` refused a dispatch; `7951f45` made the
coverage ratchet go red; `b624d78` generalised T0.21 P6; `f5d8f1c`/`78699b9` are
FAIL harvests that record the failure). **Not one loosening.**

**3. Drift from the goal.** The builder did nothing in the last day or the last
four, so there is nothing to test for drift; the last six commits are five
audits and a Review — the oversight organs describing an outage RANK 1 shows
they are helping to fund. On the converse question, `coverage` names **14
commitments with live claim specs and nothing passing**, and GOAL.md's load-
bearing ones are among them: **`fast/slow`** (8 specs, 0 passing),
**`sleep`** (4, 0), **`plasticity`** (2, 0), **`proprioception`** (2, 0),
**`shelter/building`**, **`thermal (kills)`**, **`smell`**, **`voice`**,
**`balance`**, **`tool use`**, **`touch/contact`**, **`social`**,
**`hunger/thirst`**, **`death & retry`**. **`curiosity`** stands at 12 specs / 1
claim passing and **`one brain / unison`** at 21 / 1.

**4. Builder alive and productive.** Iterations in the last 24 h: **0**. `rc=0`
in the last 24 h: **0**. PASS delta 24 h: **0**; 7 days: **+2**, both Tier 0.
The last Tier ≥ 2 PASS is `T3.01`, **2026-08-21T01:28:42 — 8.0 days ago.** The
loop is not crashed: cron fired at 00:07 today and every hour before it. It is
being refused by `pace_gate`. See RANK 1.

**5. Compute honesty — no waste found; the waste is unspent.** Every charged job
in `gpu_budget.json` maps to a submission row. W34's single charge (0.3111 h,
`jack-ladder-1787631708`, `T2.15`) produced a real ledger FAIL that was
harvested in `f5d8f1c`. The `2026-W32:kaggle` 6.3849 h opening balance is still
honestly labelled unattributable and not lowered. **29.6889 h expire in 23.2 h**
with no dispatchable spec (RANK 3) and no awake builder (RANK 1). One cosmetic
defect worth a line: that opening balance's `labelled_at` field reads
`"2026-08-14T07:1x builder, per overseer B2"` — a truncated timestamp with prose
spliced in. It affects nothing computed; it should be a clean ISO stamp.

**6. Stuck decisions — nothing to act on, and I looked for the D1 shape.** 11
armed entries, **0 undeclared**, 0 `MEANS-ESCALATED`, 0 `OVERDUE`. Nothing on the
owner's desk that a bakeoff could settle today: `D10` and `D4` both gate on runs
the builder cannot launch. No owner-decision was quietly acted on without
record. **One new entry appended this audit — `D15`, armed** (see FOR THE OWNER);
the 41st audit's finding that four of the armed defaults pick actions outside the
already-permitted set stands and is not mine to re-file.

**7. Bakeoff hygiene — no finding in `DECISIONS_RESOLVED.md`.** Three entries:
`PS.01/J` recorded as VOID and never read as a verdict; `PS.01/J2` WINNER
`impact_speed`, 2.66σ over the runner-up and 10.32σ over the null, with the
eleven gate-eliminated arms named and a cost column; `D2` resolved by ledger
replay with its property made executable (T0.08 P6). No winner inside a noise
margin, no VOID promoted. The hygiene defect remains outside that file — the
`BY VERDICT` marking on `Episodic retrieval` whose arena never ran (RANK 2).

**8. The honest summary.** No — and for a new reason. For a week the answer was
"the ladder has not moved"; today the answer is that **the machine is spending
its binding resource on watching itself**. Fourteen audits and three Reviews ran
inside this blackout. They were correct, they were adversarial, and they consumed
part of the same 65-point band that would otherwise have funded roughly thirty
builder iterations — while the organ that can actually write to the ledger was
refused 84 times by a gate that never once saw the meter above 73%. `SYSTEM.md`
says a session that makes the machine better at catching its own errors has done
the whole job even if no spec passed. That clause was written to defend building
guards over chasing green ticks. It was not written to license an oversight layer
that outspends the builder infinity-to-one, and RANK 1 is the first measurement
this project has that can tell the two apart.

---

## FOR THE BUILDER

Sequenced for the first iteration after the meter reset (~2026-08-31 04:12 UTC),
because nothing can run before it.

**B1 (highest priority; ~20 lines; CPU-only). Make the pace gate incapable of
outlasting the resource it protects, and stop it starving on spend it cannot
see.** Two parts, both strictly inside the owner's 90% stop:

- **A starvation release.** Track consecutive pace skips (a counter file beside
  `$LOST`). After **24** consecutive skips, `pace_gate` returns 0 for one
  iteration and resets the counter, logging
  `PACE RELEASE: N consecutive skips, running one iteration under the 90% stop
  (meter X%)`. This never runs above 90 — `usage_gate` still rules first — so it
  authorises nothing new; it bounds a blackout that is currently unbounded. Pick
  the constant from the measurement in RANK 1 and put the number in the message.
- **Attribution, so the next audit does not have to infer.** Every organ script
  (`ladder_loop.sh`, `overseer.sh`, `review.sh`, `field_watch.sh`) already reads
  the meter at entry. Append `{organ, ts, pct, model_pct, phase}` to
  `/data/jack-logs/usage_ledger.jsonl` at start **and** end of each run. Thirty-
  five points vanished this week and no instrument in this repo can say where —
  that is the scar this organ is built on, and it is named and dated.

**B2 (carried verbatim from the 43rd audit, still owed). Fix the quantifier in
`experiments/champions.py` and the fixture that certifies it.** An arena spec
counts as a *challenger run* only if its ledger status is `PASS` or `FAIL` (a
**VOID is not a verdict**) **and** its registry `COVERS:` kind is not `fixture`,
`rule` or `sensor` (import `coverage.py`'s parser, do not re-implement). Change
`all(v == "NOT_RUN" …)` to `not challenger_runs`, and extend the check to
`BY VERDICT` / `BY ANALYSIS` under a distinct flag so the existing ratchet
baseline is undisturbed. **Fix `_fixture()` in the same commit** — rename
`Healthy default seat` to what it is and assert it *is* flagged; add a genuinely
healthy default seat and a `BY VERDICT` seat whose only run arm is the
incumbent's. Expect new flags on `Learning core`, `Vision encoder`, `Episodic
retrieval`, probably `Sensory fusion` and `Taste`; add a second shrink-only
baseline constant set from the measured count, in the same commit, with the count
in the message. **Do not repair this by editing `CHAMPIONS.md` markings.**

**B3 (cheap, losing value every hour). Commit
`experiments/tests/sm_03_nose_reports_occluded.py`.** Untracked 4.5 days, the
only runnable claim spec for `smell`, invisible to the GPU push guard while
untracked. Commit it with its pilot state stated honestly — *implementation only,
pilot never completed, gates not frozen* — and do **not** dispatch on unfrozen
gates.

**B4 (carried from the 30th audit, third victim now). `harvest_bookkeeping` /
`ladder_loop.sh`: before writing `iteration end rc=0`, verify that any background
work the iteration claims is live has (a) a live pid and (b) a non-empty declared
artefact, and log a distinct nonzero outcome naming the orphan if not.** The rule
is in `LESSONS.md`; nothing implements it.

**B5 (one line). `experiments/gpu_budget.json`'s `2026-W32:kaggle` opening
balance carries `labelled_at: "2026-08-14T07:1x builder, per overseer B2"`** — a
truncated timestamp with prose in the field. Nothing computes on it; make it a
clean ISO stamp and move the prose into `reason`.

---

## FOR THE OWNER

**1. Your oversight layer is outspending your builder, and I am part of it.**
Measured over 84 hourly slots in which the builder ran nothing: 35 of the
65-point weekly pace band was consumed anyway, the pace line lost ground rather
than gaining it, and the builder cannot wake before the meter reset on 08-31.
Four overseer audits a day plus a daily Review, all on Opus, all exempt from the
gate that stops the builder. I have appended **`D15`** to
`docs/DECISIONS_NEEDED.md`, armed, `decide_by 2026-09-05`, with the measurement
attached. **Its default keeps one full-strength audit per day and puts the other
three behind the same pace line as the builder** — spend-reducing only,
authorising nothing new. The counterargument is recorded beside it, because it is
real: `ladder_loop.sh` deliberately left the oversight organs ungated so that
*"the machinery that catches drift keeps the plain 90% gate at full strength"*,
and this audit is the evidence that the machinery does catch things. One line
from you overrides the default either way.

**2. 29.6889 of 30 free Kaggle hours expire tomorrow, Sun 2026-08-30 00:00
GMT.** That is the third consecutive week, and the largest loss of the three. No
decision of yours is blocking it; the builder simply cannot be awake, and the
one dispatchable candidate (`SM.03`) is untracked with unfrozen gates. Recorded
so the number is not lost, not as a request.

**3. All eleven armed defaults fire on 2026-08-31 — the same day the meter
resets.** The first audit after the reset will be required to fire eleven
defaults at once, four of which the 41st audit found pick actions outside the
already-permitted set, at the exact moment the builder wakes with 84 hours of
carried work. If you intend to rule on any of them — `D1` alone costs **38
specs** — the next two days are when it matters. `D14`'s evidence problem
(42nd audit: its meter rose 34 points during 72 hours of zero requests on that
model) is unchanged and its repair is routed as a builder item.
