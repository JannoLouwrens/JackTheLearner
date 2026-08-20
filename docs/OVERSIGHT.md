# OVERSIGHT — 22nd audit, 2026-08-20 12:40 UTC

## VERDICT: ON TRACK — the ledger is honest, but the guard that polices amend-after-a-verdict watches only one of the two verdicts that invite amendment

Every hard integrity check passes, and section 2 is a genuine null result:
**no silent loosening in seven days**, verified by parsing constants rather
than reading diffs. The day's five work items all trace to a GOAL.md sentence.
T2.05 landed the FAIL its own docstring predicted three commits before it ran —
anti-run-until-pass working exactly as designed, and the best single piece of
evidence in this audit that the machine is not fooling itself.

The finding that outranks the rest is in the machinery, not the science:

> **`supersedes_fail` — the artifact that makes "a threshold moved after a bad
> verdict" auditable by a non-author — fires only when the previous verdict is
> `FAIL`. `VOID` produces nothing.** VOID is the verdict `SYSTEM.md` explicitly
> tells the builder to respond to by *changing the arm* ("VOID: fix the arm, do
> not decide"). The guard was built for the lane where redesign is discouraged
> and left open the lane where redesign is doctrine. Three post-machinery
> VOID→verdict transitions exist; none carries an artifact; one of them
> (`PS.02`) sits under a **live credited PASS**.

Nothing has actually been laundered. I checked all three, and each was honest
in its commit message. That is precisely the problem this project's own
constitution names: *"Disclosure is a property of the agent; this makes it a
property of the ledger"* (`protocol.py:1180`). In the VOID lane it is still
only a property of the agent.

---

## 0. Coverage — is the ladder the RIGHT ladder?

`python -m experiments.coverage` exits **0**. **Zero commitments with no
declared spec.** The tool's own guard holds.

**15 of 23 commitments have specs but nothing passing** — unchanged from the
21st audit. The eight carrying a claim-kind PASS: damage/nociception, taste,
memory-across-lives, generality, language (parent), hearing, curiosity,
one-brain/unison.

| commitment | declared specs | claim PASS |
|---|---|---|
| one brain / all senses in unison | **21** | **1** |
| curiosity | **12** | **1** |
| fast/slow | 6 | **0** |
| sight | 5 | **0** (3 support specs pass) |

Sight moved today in the only way it could without a run: `T3.01` (ablate
vision) is now **implemented** — the first of the 15 zero-pass commitments to
gain a runnable declared claim spec. It has not been dispatched. Sight's
claim-PASS count is still 0 and will stay 0 until it runs.

---

## 1. Integrity of the ledger — CLEAN on every hard check

82 PASS. Mechanical checks, all run against the live ledger:

| check | result |
|---|---|
| PASS whose `commit` no longer exists in git | **0 / 82** |
| PASS with no resolvable implementation (`module_path_for(strict=True)`) | **0 / 82** |
| PASS recorded from a dirty tree | **0 / 82** |
| PASS declaring a control with no `_control()` in the file | **0 / 80** |
| PASS whose `_check` never reads its control argument (AST, 2nd formal) | **0 / 80** |
| PASS with a declared control but **empty** `control_metrics` | **0 / 80** |
| PASS with fewer seeds than its spec declares | **0 / 82** |
| **stale PASS** (test changed after the run that recorded it) | **0** |

The two PASSes with no control at all are `T0.01` and `T0.10`, which declare
none by design — unchanged and correctly excluded, as in the last four audits.

`run verify` re-judges **81 PASS entries from the record alone**: 0 verdicts
that fail to re-derive, 0 gates ignoring their control, 0 controls declared but
never run, 0 unreplayable, 0 unauditable.

Four stale non-PASS entries (`T3.07` FAIL, `T4.02` FAIL, `LC.03` VOID, `BA.02`
VOID) plus `T2.02`'s pre-`impl_sha` VOID. **None is a PASS**, so no capability
claim rests on stale code.

### 1.1 RANK 1 — the amend-after-VOID path has no artifact and no auditor

`protocol.py:518` writes the `supersedes_fail` artifact under exactly one
condition:

```python
if prev and prev.get("status") == Status.FAIL.value \
        and prev.get("ran_at") != r.ran_at:
```

and `audit_supersedes_fail` (`protocol.py:1227`) skips every non-FAIL history
entry:

```python
if e.get("status") != Status.FAIL.value:
    continue
```

So a VOID that is superseded after its gates were redesigned leaves **no
machine-readable pairing and is never audited**. Measured on the live ledger,
post-machinery (the artifact landed 2026-08-13):

| spec | transition | impl_sha moved | artifact | current status |
|---|---|---|---|---|
| `PS.02` | VOID 08-19T08:18 → **PASS** 08-19T08:40 | no (`362ee283…` → `362ee283…`) | **none** | **PASS (live claim)** |
| `T2.05` | VOID 08-14T12:04 → ERROR 08-20T11:34 | **yes** (`df366939…` → `6f8ea6f8…`) | **none** | FAIL |
| `BA.02` | VOID 08-13T17:41 → VOID 08-14T01:07 | **yes** (`c6503b29…` → `b697bfda…`) | **none** | VOID |
| `XL.01` | FAIL 08-19T16:19 → FAIL 08-19T18:35 | **yes** | **YES** | FAIL |

The FAIL lane demonstrably works — `XL.01` is the proof. The VOID lane is
0 for 3.

**I verified each of the three is honest.** `PS.02`'s VOID was a staleness
displacement re-run at an identical `impl_sha` (`643637f`); `T2.05`'s redesign
is strengthen-only (§2 below); `BA.02` is parked under D8. The finding is not
that something was hidden. It is that **the record cannot tell you that**, and
`SYSTEM.md`'s whole premise is that the machine must not depend on the
optimism of its author.

This is exactly the shape the 15th audit named — *"the staleness detector reads
green over an empty domain"* — one lane further on.

### 1.2 RANK 2 — an intervening ERROR silently severs the FAIL→PASS pairing

Two independent mechanisms both pair a verdict with **the immediately
preceding row only**:

- the recorder: `prev = on_disk.get(rid)` (`protocol.py:482`), the single
  previous top-level entry;
- the auditor: `zip(seq, seq[1:])` over history, with `if e["impl_sha"] ==
  nxt["impl_sha"]: continue` (`protocol.py:1232`).

Concrete failure, using only events that occurred **today**:

1. A spec records `FAIL` at `impl_sha` A.
2. A GPU kernel dies — `ERROR` recorded, still at `impl_sha` A. *(This happened
   three times in the last 24 h: `T2.05` twice on the mujoco wheel gap,
   `TA.02` once on the artifact-path read.)*
3. The builder moves a gate and re-runs → `PASS` at `impl_sha` B.

The recorder sees `prev.status == ERROR` and writes no artifact. The auditor
pairs (FAIL, ERROR), finds `impl_sha` **equal**, and `continue`s — never
checked. It then reaches (ERROR, PASS) and skips it because the left side is
not a FAIL. **The amendment from A to B is invisible to both.**

`T2.05` is the live demonstration of the severed shape today: its chain is
`VOID → ERROR → ERROR → FAIL`. Dead kernels are not an exotic event here; they
are the most common event in the GPU path.

### 1.3 The auditor has still never checked a live pair

`audit_supersedes_fail(ledger, repo_root='.')` → **violations 0, checked_pairs
0, unauditable_pairs 27.** Third audit in a row reporting 0 checked (18th audit
RANK 2, 21st audit). This is currently *honest* — the auditor ignores records
whose current status is not PASS, and `XL.01`'s genuine post-`impl_sha`
amendment sits under a FAIL, so it is correctly excluded. But taken with §1.1
and §1.2, the standing characterisation is: **the guard has never fired, cannot
see VOID, and can be severed by a dead kernel.**

---

## 2. Thresholds and controls over time — NO SILENT LOOSENING

Method, so this null result is checkable rather than asserted: I parsed every
module-level constant assignment in `registry.py`, `registry_expansion.py` and
`experiments/tests/` at both sides of all **41 commits** in the window and
diffed the *values*, then separately AST-diffed every `_check` and `_control`
body for `or`-count, `assert`-count and length, and regex-diffed every `Spec(…)`
declaration for `seeds=` and `control=`.

**Exactly one pre-existing constant moved in seven days:**

| commit | change | verdict |
|---|---|---|
| `265e683` | `ALIEN_MIN_DIST` 1.5 → **2.0** | **STRENGTHENED** (control pushed further away), message quotes the pilot min_dists that made it constructible |

Everything else flagged was a **new** constant (`CTRL_TOL`, `GRID_STEP`,
`EXPLORE_STD_LC03`, `BA02_TILT0_LOG10_DEG`, `IMPL_DEPS`, XL.01's fixture
geometry) or the `JOB` install string (the two mujoco pin commits).

**No `_check` gained an `or`. No assertion was deleted. No `seeds=` changed.
No `control=` was set to None.**

### 2.1 T2.05's redesign, examined closely — strengthening on every axis

This is the only claim-gate redesign in the window and it deserves the working,
because it *removed a VOID gate* and it landed a verdict change:

- **Null strengthened.** `mse_persist` → `min(mse_persist, mse_mean)`. On the
  measured data that is 0.824 rather than 1.092, so the claim bar tightened
  from `≤ 0.874` to `≤ 0.659` — a **25 % harder** target.
- **A gate was ADDED to the conjunction**, not removed: `wm_beats_ridge_all`
  (`mse_wm ≤ mse_ridge` every seed), joined by `and`.
- **The removed gate cannot launder a PASS.** `persist_informative_all` VOIDed
  runs where persistence lost to the mean. Under `min()` the null is now the
  *stronger* of the two, so the condition it guarded against is arithmetically
  impossible to exploit. Its only effect is to let an honest **FAIL** be
  recorded where the run previously could only say "measurement invalid" —
  which is a stronger statement against the claim, not a weaker one.
- **`CTRL_TOL = 0.98` is a new 2 % control tolerance** — the one genuinely
  permissive constant introduced. It was registered in the commit *before* the
  run, justified against the "bar finer than one quantum" lesson, and **it was
  not load-bearing**: measured `mse_shuffled` [0.8244, 0.8637, 0.9160] sits
  *above* `mse_null` [0.8240, 0.8601, 0.9142] on every seed, so
  `shuffled_beats_null = 0.0` with or without it.
- **The docstring predicted FAIL before the run, and the run recorded FAIL.**
  Predicted wm 0.178–0.231 vs ridge 0.114–0.131; measured wm [0.1783, 0.1956,
  0.2312] vs ridge [0.1143, 0.1168, 0.1309]. `wm_beats_ridge_all = 0.0`.

That is the anti-run-until-pass discipline working, and it is the strongest
positive finding in this audit.

### 2.2 LC.03's control inversion — re-checked, and now in flight

`87590a4` (08-13) inverted control side (e) from `darkroom_margin ≤ -SIGMA_GATE`
to `> -SIGMA_GATE` and replaced side (a) "statue dies soonest" with "statue
rides the basal ceiling ±10 %". The 21st audit examined this and ruled it
justified; I re-read the diff and agree — both sides were pre-flagged as
suspect *before* the pilot, amended in the registering commit with the pilot
numbers quoted, old sides left in git history, claim gates untouched.

One forward-looking note, because the situation has changed since that ruling:
**LC.03's registered run is executing right now** under those amended controls
(pid 92854, launched 10:19 UTC, ~15 h). When it lands, side (e) is a rig
tripwire rather than a must-fail control, and the curiosity burden rests on the
dwell/chaos gates. That is what was registered and it is not a loosening — but
if LC.03 returns PASS, the reviewer should read the dwell/chaos evidence as the
load-bearing part and not credit the darkroom as a disconfirming control.

---

## 3. Drift from the goal — NONE in the work

Five work items since 00:00 UTC, each against the GOAL.md sentence it serves:

| work | GOAL.md sentence |
|---|---|
| `SM.02` implemented, piloted (1.56 GPU-h), three repairs, **PARKED on measurement** | "SMELL… olfaction finds food, fire and decay at a distance and through occlusion — the sense that works when sight fails" (:41, :45–48) |
| Reattach guard `b062ccd` (`kernel_sha256`, refuse-on-mismatch at zero billing) | "Really learning, not appearing to learn" (:57); SYSTEM.md law 1 |
| `LC.03` rig re-derived (constant `explore_std`), food-quantum residual resolved by replay probe, registered run launched | "one interconnected brain"; "learns his world by living in it" (:11, :21) |
| `T2.05` redesigned, dispatched, **FAIL landed** | fast/slow axis 1 — "whether lookahead earns its keep" (:241–244) |
| `T3.01` implemented — ablate vision | "we PROVE each one is load-bearing — ablate a sense, something measurable must degrade" (:54) |

**No drift.** Nothing this day serves no sentence.

The converse remains the standing gap, and it is a fifth audit running: 15 of
23 commitments have nothing passing; unison 21 specs / 1 claim PASS; curiosity
12 / 1. This is not neglect — the 20th audit established most are *not
implemented* rather than lost, and today's `T3.01` is the loop working the
right list. It is restated because §8 turns on it.

---

## 4. Is the builder alive and productive? — ALIVE; zero ledger movement today

**25 iterations in the last 24 h. 24 ended `rc=0`; one `rc=124`** (timeout,
08-19T14:57). No paused loop, no credit exhaustion, no load aborts.

**PASS delta over 24 h: 81 → 82** (+1, `TA.02` at 08-19T23:16).
**PASS delta since 00:00 today: 82 → 82 across 13 iterations.**

That zero is honest rather than idle, and I checked it rather than assuming:
the day produced one park on measurement (`SM.02`), one honest FAIL (`T2.05`),
one new guard (reattach), one spec implemented (`T3.01`), and one 15-hour
registered run currently executing. Four of those cannot move the PASS count by
construction.

**The 21st audit's RANK 1 is CLOSED, and closed the right way.** It found three
iterations waiting on dead background jobs. The builder shipped
`scripts/launch_detached.sh` — a *mechanical* guard, not advisory: pinned cwd,
`setsid`, `2>&1` into the log, and at 15 s the process must be alive and the log
non-empty or it exits 1 with the log tail. LC.03's launch used it. B2 (silent
iterations) is closed too: every iteration since 08:23 wrote a journal summary.
B5 was obeyed to the letter — the checks came back negative again and the
builder stopped rather than attempting a fourth repair.

**Liveness verified, not assumed:** LC.03's three workers read 2 h 19 m CPU time
each at 12:37, up from 1 h 27 m at 11:49 and 1 h 56 m at 12:18. It is computing.

---

## 5. Compute honesty — W33 fully attributed

**W33 Kaggle: 4.566 h charged (ok) + 0.2049 h failed. 25.23 h remain of 30, and
they expire Sunday 2026-08-23 — about 59 hours away.**

Every W33 job is attributable:

| job | hours | what it bought |
|---|---|---|
| `1787185633` | 1.5583 | SM.02 pilot → the non-learning verdict that parked SM.02 |
| `1787173733` | 0.9599 | T2.04 PASS |
| `1787226047` | 0.9187 | T2.05 FAIL |
| `1787178802` | 0.5101 | TA.02 PASS |
| `1787170366` | 0.3220 | T2.03 PASS |
| `1787124880` | 0.2970 | T2.06 PASS |
| `1787166872` / `1787225429` / `1787225777` | 0.2049 | three failed kernels, all recorded as ERROR rows |

The SM.02 pilot is the only W33 job with no ledger row, and it is correctly
attributed — the 20th audit's B2 backfill wrote an `attribution` receipt line
joining `attempt_id` to `SM.02/pilot`. It bought a real negative. **No waste.**

**13.99 h across 11 jobs carry neither a ledger row nor a receipt attribution —
all of them W32 or earlier**, predating the spec-attribution machinery
(2026-08-14). The two largest (5.5798 h + 5.5786 h = 11.16 h) are T2.01's two
runs, attributed in prose at `DECISIONS_NEEDED.md:1455`. This is a closed
historical gap, not a live leak: the machinery-era record is complete.

`overruns: []` still under-reports — W31 holds 37.46 h against a 30 h ceiling
(19th audit finding, unchanged; the backfill never crossed the charge path that
would have fired the detector).

---

## 6. Stuck decisions

**D1 — "Does the 57M trunk stay in the control path?" — open since 2026-08-09,
eleven days.** It blocks `T2.01` (*Locomotion beats a random policy*), which has
read **FAIL since 2026-08-12**, eight days with no attempt 3. Cost updated at
00:45 today by the 20th audit; I am **not** appending a fresh restatement 12
hours later — that would be noise, not evidence. The arithmetic has moved
slightly and belongs in FOR THE OWNER below.

Nothing else on the desk has gained decisive evidence since the last audit, and
nothing blocked there is resolvable by a bakeoff the system could run itself.
**No owner-decision was quietly acted on:** the PLASTIC-ONLY decree is intact in
`GOAL.md:76`, and `T3.01` is built on a *plastic* `PrismaticVisionEncoder`
receiving its first-ever gradient — the decree honoured, not routed around.

---

## 7. Bakeoff hygiene — CLEAN

Three entries in `DECISIONS_RESOLVED.md`:

- **`PS.01/J` — VOID, recorded as VOID.** Three arms below the 3.0σ learning
  gate; the file says so and declares no winner. A VOID was *not* treated as a
  verdict — this is the exact behaviour T2.02 invented the gate for.
- **`PS.01/J2` — WINNER `impact_speed`.** Beats the runner-up by **2.66σ**
  against a declared `margin_sigma` default of **1.5** (`bakeoff.py:152`), and
  clears the null by 10.32σ. Outside the noise margin. Runs under
  `gate_mode: screen` with a written rationale explaining why these arms are
  deterministic observables over identical cached rollouts rather than
  learners — the exemption is argued, not assumed. Losers recorded.
- **`D2` — WINNER BLOCK**, resolved by ledger replay rather than `run_bakeoff`,
  with the method justified (no seeds, no null, no training that could fail),
  exposure 0 vs 9 measured on a natural experiment the ledger had already run,
  loser recorded, and a **re-open trigger** stated. Textbook.

**No decision made without a learning gate. No VOID treated as a verdict. No
winner chosen inside the noise margin.**

---

## 8. The honest summary — are we closer to a creature?

**Closer to a system that cannot fool itself: yes, measurably. Closer to Jack:
barely, and not today.**

The case for the day is real and I do not want to undersell it. `T2.05` is the
single best artifact this audit found: a spec whose docstring said *"this will
FAIL"* three commits before it ran, which then spent 0.92 GPU-hours to record
exactly that FAIL, on gates that had been tightened 25 % in the same commit.
A system that spends real compute to publish its own negative result is doing
the thing this repository was built to do. `SM.02` parked on measurement after
three honest repairs is the same instinct. The reattach guard closed a path by
which a local edit could have been laundered into a certificate.

But the ledger reads **82 PASS, unchanged for thirteen hours**, and the four
capabilities the goal actually turns on are where they were yesterday: unison
1 claim of 21 specs, curiosity 1 of 12, sight 0 of 5, fast/slow 0 of 6.
`T2.05`'s FAIL means the shipped world model loses K-step prediction to a single
ridge regression — an honest and useful finding for `LC.04`, and also a plain
statement that the imagination half of fast/slow is not yet there. `XL.01` says
the carried diary bought nothing on fresh worlds. `T2.01` says we still cannot
demonstrate that he moves.

GOAL.md's own standard is *"climbing the ladder on attempt 40 after falling on
attempts 1–39, without anyone telling him to."* Today the system got better at
proving it is not lying about attempt 40. It did not get closer to attempt 1.

That is not drift and it is not dishonesty — it is what an eleven-day-old owner
decision does to a project whose locomotion spec is red. The machine is sound.
It is pointed at a wall.

---

## FOR THE BUILDER

**B1 (RANK 1). Extend the supersede artifact to VOID.** In
`protocol.py:518`, the condition is `prev.get("status") == Status.FAIL.value`.
Widen it to `in (Status.FAIL.value, Status.VOID.value)` and write the artifact
under a name that carries the source verdict (e.g. keep `supersedes_fail` for
FAIL and add `supersedes_void`, or add a `"status"` key inside the existing
dict — either is fine, but *do not* silently reuse the FAIL name for a VOID,
because §1.1's table is exactly the read an auditor needs). Mirror it in
`audit_supersedes_fail` (`protocol.py:1227`): `if e.get("status") not in
(FAIL, VOID): continue`.

Rationale, in one line: `SYSTEM.md` tells you that a VOID means *"fix the arm,
do not decide"* — so VOID is the verdict that **doctrinally** precedes a
redesign, and it is the one lane with no artifact. Live instances: `PS.02`
VOID→PASS (under a credited PASS today), `T2.05` VOID→…→FAIL with `impl_sha`
moved, `BA.02` VOID→VOID with `impl_sha` moved. All three are honest; none is
provable from the record.

Add the property to `T0.27` in the same commit (it is at 8 properties; this is
P9): plant a VOID→PASS pair with a moved threshold and assert the auditor flags
it, plus a known-negative VOID→PASS at an identical `impl_sha` (`PS.02`'s real
shape) that must **not** flag.

**B2 (RANK 2). Make the pairing survive an intervening ERROR.** Both the
recorder (`prev = on_disk.get(rid)`, `protocol.py:482`) and the auditor
(`zip(seq, seq[1:])`, `protocol.py:1226`) pair a verdict with the immediately
preceding row only. An `ERROR` between a FAIL and its amended re-run defeats
both — and the auditor's `if e["impl_sha"] == nxt["impl_sha"]: continue` makes
it worse, because a dead kernel records the *unchanged* `impl_sha`, so the pair
is skipped as "same code re-run" and the real amendment is never examined.

Fix: when selecting the prior verdict to pair against, **skip rows whose status
is `ERROR`** (an ERROR is an infrastructure event, not a verdict on the
hypothesis — the same reasoning that already excludes it from `unsatisfied`).
Walk back through history to the last row carrying a real verdict. Do this in
both places. `T2.05`'s live chain `VOID → ERROR → ERROR → FAIL` is your
fixture; you produced three ERROR rows in the last 24 h, so this is a common
path, not a corner case.

**B3.** `T2.05`'s FAIL (`ran_at` 12:35:59, kernel `1787226047`, 0.9187 h)
landed in the working tree **after** the 12:18 iteration ended — ledger,
`gpu_budget.json` and `gpu_submissions.jsonl` are all modified and uncommitted
right now. Your journal already hands this to the next iteration, so this is a
note rather than a finding: commit it before anything else, and state in the
message that the FAIL was **predicted in the docstring before the run**. That
prediction is the most valuable thing in the row and it will be invisible to a
later reader who sees only a red verdict.

**B4 (credit, and one thing to preserve).** `launch_detached.sh` is the right
answer to the 21st audit's B1 — mechanical, not advisory, and it verifies the
*product* (alive **and** log non-empty) rather than the exit code. Two things
worth keeping: the `WARN` on alive-but-silent correctly refuses to call
alive-but-silent a death, and the header line makes "log non-empty" meaningful
against a buffering payload. Consider making `dispatch.sh` and any future
detached launcher route through it, so the guard cannot be bypassed by habit.

**B5.** When `LC.03` lands (~01:20 UTC, pid 92854, workers verified climbing),
remember its control side (e) is now a rig tripwire, not a must-fail control
(§2.2). If it reads PASS, the dwell/chaos gates carry the curiosity burden —
say so explicitly in the harvest commit, so no later reader credits the
darkroom as a disconfirming control it is no longer configured to be.

**B6.** 25.23 of 30 W33 Kaggle hours expire **Sunday 2026-08-23**, ~59 h away.
`T3.01`'s pilot plus registered run is the queued spender and is the right one —
it is sight's first claim spec. Do not manufacture a dispatch to burn quota;
the 10:24 iteration got this exactly right when it said so out loud.

---

## FOR THE OWNER

**One thing, and it is the same thing, now eleven days old.**

**D1 — does the 57M trunk stay in the control path?** It was raised
2026-08-09. It blocks `T2.01`, *Locomotion beats a random policy*, which has
read **FAIL since 2026-08-12** — eight days, no attempt 3. The fork is
unchanged and is the whole of what is owed:

> **(i)** strike option A (freeze the trunk) — the PLASTIC-ONLY decree of
> 2026-08-09 (`GOAL.md:76`) stands as written; or
> **(ii)** keep option A available and narrow the decree's scope, saying where.

Please do not answer *"do what the measurements say"* — a trigger earlier in
`DECISIONS_NEEDED.md` would read that as option A, and the 15th audit
established this is a constitutional question about what the decree admits, not
a measurement question a bakeoff can settle.

**The updated arithmetic:** 25.23 of 30 W33 Kaggle hours expire **Sunday
2026-08-23**, about 59 hours from now. `T2.01` billed 5.58 h on each of its two
previous runs. **The expiring quota fits it four times over**, and neither
compute nor credits is the reason it has not run. One line from you is.

I am not asking you to hurry a constitutional decision, and I take no side. I
am reporting that the project has now certified taste, smell's fixture, damage,
thermal sensing, balance sensing, voice, vision features, language-action
alignment and behaviour cloning — 82 PASS — for a creature that cannot yet be
shown to walk. GOAL.md's standard for learning is *"climbing the ladder on
attempt 40 after falling on attempts 1–39."* Climbing requires moving.

*Nothing new appended to `DECISIONS_NEEDED.md` this audit: the 20th audit
restated D1 twelve hours ago with the same fork, and repeating it hourly would
turn the evidence file into noise.*
