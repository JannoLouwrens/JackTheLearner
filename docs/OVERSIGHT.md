# OVERSIGHT — 36th audit, 2026-08-27 00:40 UTC

## VERDICT: DRIFTING — the guard that exists to stop a GPU job running code that isn't in the commit reads `git status --untracked-files=no`, so it is blind to a **brand-new, untracked** test file. `SM.03` is exactly that file, `run status` already calls it implemented, and it is the only registered GPU_SHORT claim candidate for 29.69 free hours that expire in 47 h. Every gate would go green and the kernel would clone a repo that does not contain it.

**State.** `HEAD` is `f16e33e` — the 35th audit's own commit. **Zero commits
since.** Working tree is still the single untracked
`sm_03_nose_reports_occluded.py`. `84 PASS / 187 registered`. The builder's last
iteration ended **2026-08-25 12:23:33**; every slot since — **36 consecutive
hourly slots, 36 h 17 m** — logged `PACING: … skipping`.

**Clean results, each re-run by me rather than relayed:**

- **§1 ledger integrity — clean.** `run verify`: **83 PASS re-judged from the
  record alone, 81 controls probed.** 0 verdicts that no longer re-derive, 0
  gates that ignore their control, 0 controls declared but never run, 0 gates
  unreplayable, 0 entries unauditable, 0 controls run but undeclared. Two PASSes
  carry no control (`T0.01`, `T0.10`), both long-declared existence claims;
  `T0.18` self-excludes correctly. One known stale-by-content entry (`T2.02`,
  recorded VOID) is reported by the runner itself.
- **§2 thresholds and controls — clean.** `git log f16e33e..HEAD` is **empty**;
  nothing has been written since the last audit. I re-ran the seven-day scan
  anyway across `registry.py`, `registry_expansion.py` and `experiments/tests/`.
  The single hit that mentioned `seeds` is `ed2d969` reformatting the LG
  registration — `seeds=3` survives verbatim on every touched spec. No threshold
  moved in the loosening direction, no control deleted or weakened, no `_check`
  gained an `or`, no assertion removed. Both `N_PROPERTIES` edits in the window
  are **increases** (9→10, 10→11).
- **§5 compute accounting — clean.** `overruns: []`; W34 charged **0.3111 h of
  30**, one job, one real ledger row (`T2.15`, FAIL, harvested at `f5d8f1c`).
- **§7 bakeoff hygiene — no findings.** No bakeoff has run since the 29th audit.
- **The three constitutional gates all exit 0**, all run by me: **coverage** — 0
  commitments with no declared spec, 0 CLAIM-DEAD, 4 known-dangling GOAL.md
  citations (shrink-only baseline); **decisions** — ratchet ok, now **1/10**
  undeclared (I closed one, honestly — RANK 3); **champions** — ratchet ok, 6/8
  seats with a phantom arena, 12 violations, all carried.

---

## RANK 1 — `assert_ref_is_current` cannot see an untracked file, and `T0.22 P15` passes on a fixture its own production path can never produce

This is first because it is the only finding this audit that can put a **wrong
number in the ledger**, and because it is live right now on the one spec the
builder's own handoff points at.

### The hole, measured

`experiments/gpu.py:274`, inside the function whose entire job is to refuse a
job whose code is not the code being tested:

```python
rc, dirty = git("status", "--porcelain", "--untracked-files=no")
```

`experiments/protocol.py:368`, the `+dirty` stamp, asking the *same question*:

```python
porcelain = subprocess.run(["git", "status", "--porcelain"], ...)
dirty = [ln for ln in porcelain if is_code_dirt(ln)]
```

Run against the live tree, this hour:

```
what protocol.py's stamp queries : ['?? experiments/tests/sm_03_nose_reports_occluded.py']
   -> is_code_dirt says          : ['?? experiments/tests/sm_03_nose_reports_occluded.py']
what gpu.py's push guard queries : []
   -> offending_dirt says        : []

SAME PREDICATE, SAME LINE:
  '?? experiments/tests/sm_03_...py'  is_code_dirt = True   offending_dirt = True
```

**The predicate agrees. The query does not.** One organ asks git a question that
includes untracked files and the other asks a question that excludes them, and
then they hand the same function two different answers.

### Why no instrument caught it — and this is the part that matters

`gpu.py:229`, `offending_dirt`'s own docstring, is a 20-line essay about exactly
this class of bug:

> *"The exclusion is `protocol.is_code_dirt` — NOT a second list that happens to
> agree. On 2026-08-12 the two lists disagreed by exactly one entry each way …
> Both organs answer ONE question — does this uncommitted file mean the code
> moved — so there is now one predicate and zero permitted difference.
> **T0.22 P15 pins them together.**"*

So I read P15. `experiments/tests/t0_22_borrowed_constants.py:390`:

```python
code_lines = [" M experiments/run.py", "?? experiments/tests/t9_99.py",
              " M TrainingPipeline.py"]
...
disagree = [ln for ln in code_lines + out_lines
            if dirt(ln) != bool(offending_dirt([ln]))]
```

**P15's fixture contains an untracked test file — `"?? experiments/tests/t9_99.py"` — and asserts both organs flag it.** They do. The property passes, 15/15,
every run. And it is measuring nothing about the defect, because the production
call site *cannot generate that line*: `--untracked-files=no` strips every `??`
before `offending_dirt` is ever reached.

The repair of 2026-08-12 unified the **predicate** and left the **query**
un-pinned. The guard-of-the-guard tests a function in isolation and the hole is
in what gets passed to it. Every instrument this project owns is green.

### What it would cost, concretely, on the next admitted iteration

`SM.03` — *"The nose reports what the eye cannot"* — is the successor claim spec
for **smell**, one of the owner's constitutional senses, registered on 08-25
specifically to clear its CLAIM-DEAD status. Trace the full dispatch path as it
stands:

| gate | reads | verdict on SM.03 today |
|---|---|---|
| `run status` | the **filesystem** | **implemented** — and note it prints *no* `(not implemented)` marker, unlike `SH.02` and `VO.02` on the same screen |
| `coverage.py` | registry + ledger | **RUNNABLE** |
| `dispatch.sh:26` | is HEAD pushed | **pushed** — HEAD is `f16e33e`, on `origin/main` |
| `assert_ref_is_current` | `--untracked-files=no` | **clean tree** |
| the Kaggle kernel | `git clone --depth 50` from GitHub at HEAD | **the file is not there** |

Four green gates, one clone, one `ModuleNotFoundError`, one burned submission
against 29.69 hours that expire in 47. And `gpu.py`'s own docstring names the
worse branch, which this hole also opens:

> *"Worse was the near-miss before it — had the missing method been an OPTIONAL
> path instead of an AttributeError, the run would have SUCCEEDED and silently
> measured the old model, and the ladder would have recorded that number as
> evidence."*

An untracked file is the *general* case of that near-miss, not a special one: any
new module that an existing test imports optionally would produce a PASS
attributed to a commit that never contained the code that produced it. That is a
ledger integrity failure, and the ledger is the only scoreboard this project has.

**Fairness, stated plainly: no damage has been done yet.** I checked. W34's only
charged job (`T2.15`, dispatched 08-25 04:07 at HEAD `20b8660`) predates
`sm_03_...py`'s creation at 08-25 12:20, so no past GPU result is implicated. The
harm here is entirely prospective — and it is queued behind the very next
iteration the pace gate admits.

---

## RANK 2 — the wake window is REAL, and three consecutive audits' pessimism is falsified by measurement

The 32nd audit titled itself *"29.69 free GPU-hours that expire a day before the
builder may wake."* The 35th said *"I give no wake-up date."* Both were reading a
burn rate that has since stopped. The correct finding today is the opposite one,
and it changes what should be done rather than merely updating a number.

**The burn has been flat at zero for nine hours.** From `ladder.log`, and
confirmed by my own live read at 00:38:

| slot (UTC) | 16:07 | 18:07 | 20:07 | 22:07 | 00:07 | live 00:38 |
|---|---|---|---|---|---|---|
| `week:all models` (**the gate**) | 62 | 62 | 62 | 62 | **62** | **62** |
| week elapsed | 35 | 36 | 38 | 39 | **40** | **40** |
| pace line | 48 | 49 | 50 | 51 | **51** | — |
| gap over the line | 14 | 13 | 12 | 11 | **11** | — |

**0.00 pts/h over 8 h 31 m**, against a recovery rate of 0.3869 pts/h. The gap is
closing at full speed for the first time this week.

**The arithmetic, from the gate's own source** (`lib_usage.sh:70,85`,
`PACE_FLOOR=25`, `PACE_CAP=90`, integer division):

```
allow = 25 + (65*elapsed + 99)/100
  elapsed 56 -> allow 62   62 >= 62  -> SKIP
  elapsed 57 -> allow 63   62 <  63  -> PROCEED
```

The week runs Aug 24 04:59 → Aug 31 04:59 UTC (168 h). `elapsed` reaches 57% at
**Aug 28 04:44 UTC**, so the first slot the gate admits is **2026-08-28 05:07
UTC — about 28.5 hours from now** — *if `all models` stays at 62.*

W34's Kaggle hours die when `%U` rolls to 35, which is **Aug 30 00:00 UTC**
(`gpu.py:361`; Aug 27 is a Thursday, so the next Sunday is Aug 30). That leaves a
**~43-hour window** in which the builder is awake and the free GPU-hours are
still alive. It is not a lost week. It is a narrow, real, and datable window.

**But it is a low duty cycle, and nobody has stated this before.** Fable is at
100%, so every admitted iteration falls through `FALLBACK_MODELS` to **Opus**,
which bills the same `all models` meter that gates it. The line advances one
point per **2.58 h**. So an iteration that burns `B` points re-gates the builder
for `2.58 × B` hours:

| Opus iteration burns | next admitted slot | iterations in the 43 h window |
|---|---|---|
| 1 pt | +2.6 h | ~16 |
| 2 pts | +5.2 h | ~8 |
| 3 pts | +7.7 h | ~5 |

**This inverts the standing advice.** Previous audits told the builder to do the
cheapest work first. That was right for a builder running hourly on Fable. It is
wrong for a builder that may get **five** iterations before the hours expire. The
first admitted iteration must do the thing that cannot be recovered later, not
the thing that is easiest — and RANK 1 says what that is: commit `SM.03`, close
the untracked hole, then dispatch. In that order, in one iteration, because there
may not be a second one for eight hours.

**One caveat I will not hide:** the 33rd audit was burned by turning a rate into a
date, and 62 has been flat before and then moved. Nine hours of zero burn is a
measurement, not a guarantee — the external sessions drawing this meter are
outside the project's sight and can resume at any time. Treat Aug 28 05:07 as the
**earliest** admissible slot, not a promise.

---

## RANK 3 — the `UNDECLARED` ratchet has reached a floor it cannot leave by arming, because both entries in it were **already answered by the owner**

`decisions --check` reported **2 not armed**. I read both. Neither is an unasked
question.

**(a) "The owner's hands — how does a human TOUCH Jack's world?"** Its body, 22
lines under its header:

> **DECIDED 2026-08-09, same day: YES.** Owner: *"Can you also drop stuff in for
> him... Yes."* Care verbs approved … Design work unblocked -> INTEGRATION_QUEUE.

**(b) "Was physics-first retired by argument instead of by bakeoff?"** — the 35th
audit found this one; I confirm it and add the mechanism:

> **DECIDED 2026-08-09: (a) RUN IT.** Owner: *"schedule the run after T2.01."*

**Why the instrument is blind, exactly.** `decisions.py:99` is
`_SETTLED = re.compile(r"RESOLVED|off your desk|BY THE CALENDAR", re.I)`, and
`parse()` applies it to `_HEADER.finditer(text)` — **markdown `##` headers only,
never bodies**. An owner ruling written in prose inside an entry is invisible to
the ratchet forever. This is the **fourth** instance of the constraint-in-prose
shape already recorded in `LESSONS.md` (LC.03's missing test file, the phantom
champion arenas, DP.04's prose-only block behind LG.00) and the first where the
invisible thing is an **answer** rather than a dependency.

**What I did, and why it is the ratchet-legal move.** My standing duty is to arm
at least one entry per audit. **Arming either of these would be fabricating a
question** — a default for an answered decision either restates the owner
(noise) or departs from them (unconstitutional). So I shrank the ratchet the only
honest way available: I appended a **settling header** to (a), which is
unambiguously and completely closed — decision made, anti-puppeteering constraint
recorded, work routed to the queue, nothing owed by anyone. Verified live:
`ratchet ok (1/10 undeclared)`, down from 2.

**I deliberately did NOT settle (b), and the reason is the finding.** Its ruling
is given but **unexecuted**: `T5.01`, titled "THE thesis test" — the spec that
makes the project's *founding* premise rest on our own numbers — is still
`NOT_RUN` **18 days** later, queued behind `T2.01` (FAIL, 2.67σ against a 5σ bar,
transitive block mass 36). Filing it as "resolved" would convert a live debt into
a closure. It belongs in `DECISIONS_RESOLVED.md` **with the owed run recorded as
owed** — builder item B5, carried.

**The structural consequence.** The remaining `UNDECLARED` entry is un-armable,
so from here **the ratchet can only shrink by CLOSING — and closing is precisely
what the instrument cannot detect.** An overseer instructed to "arm at least one
per audit" against a floor of un-armable entries will either quietly stop
complying (and the instruction rots) or manufacture an arming (and corrupt the
record). Neither is acceptable. The repair is builder item **B3**: teach
`_SETTLED` to read entry bodies, and report a body-level ruling as a **distinct
state, `ANSWERED-UNCLOSED`** — because *"nobody asked properly"* and *"the owner
answered and we did not act"* have opposite remedies, and the tool currently
prints the same word for both.

---

## RANK 4 — `SM.03` untracked for 36 hours (sixth consecutive audit), and the rescue path still cannot see it

```
?? experiments/tests/sm_03_nose_reports_occluded.py   32,086 bytes, mtime 08-25 12:20
```

`harvest_bookkeeping`'s `HARVEST_PATHS` covers `ledger.json`, `gpu_budget.json`
and `gpu_submissions.jsonl`. It is the pace-skip rescue path, it fired correctly
during this blackout, and it covers **the one artifact class that is not at
risk**. 32 KB of a constitutional sense's only claim implementation is one
`git clean` from gone. RANK 1 is what upgrades this from housekeeping to a
correctness problem: while it is untracked it is not merely *unsaved*, it is
*invisible to the guard that decides whether a GPU result is attributable*.

I did not commit it. Versioning a test file is outside this role and the boundary
is load-bearing — it is what stops specs entering the tree without
pre-registration discipline. It remains builder item **B1**, and this is the
sixth audit to write that sentence.

---

## §3 — drift from the goal

**What the builder worked on in the last 24 hours: nothing.** Thirty-six dark
slots. There are **zero commits of any kind** since the 35th audit — for the
first time in this blackout, not even a document.

That is a change in kind from the 35th audit's finding, and it cuts both ways.
Its complaint was *"100% of output is prose about itself"*; the honest update is
that the prose has stopped too. There is nothing to report as drift, because
there is no work to classify. **§2 is clean partly because nothing was written**,
and I will not present an empty diff as a clean bill of health.

**The converse and harder question — which parts of GOAL.md have no passing spec
at all** — I re-ran `coverage` rather than quoting: **14 of 23 commitments have a
live claim spec and nothing passing.** Unchanged. Verbatim from the tool:

- `smell` — 0 pass; claim `SM.03` RUNNABLE (**untracked, undispatchable** — RANK 1)
- `thermal (kills)` / `shelter/building` — 0 pass; claim `SH.02` RUNNABLE, not implemented
- `voice` — 0 pass; claim `VO.02` RUNNABLE, not implemented
- `balance` — 0 pass; claim `BA.02` RUNNABLE
- `proprioception`, `plasticity`, `sleep`, `fast/slow` — 0 pass, **every claim blocked**
- `curiosity` — 12 specs, **1 pass**
- `one brain / unison` — 21 specs, **1 pass**

Curiosity and all-senses fusion are the two names the audit brief warns are *"most
likely to be quietly neglected in favour of easy wins"*. They carry **33 specs and
2 passes** between them.

---

## §4 — is the builder alive and productive?

Alive as a cron entry; productive not at all.

| | value |
|---|---|
| iterations in the last 24 h | **0** |
| consecutive pace-skipped slots | **36** (08-25 13:07 → 08-27 00:07) |
| hours dark | **36 h 17 m** |
| PASS delta over that window | **0** (84 → 84) |
| commits of any kind since the last audit | **0** |
| last *claim* PASS | `T3.01`, 2026-08-20 15:29 — **6.4 days ago** |

No repeated failures, no aborted loads, no credit exhaustion on the project's own
side, no paused loop anybody forgot to resume. The cron fires every hour and the
pace gate declines every hour. The mechanism is understood and is RANK 2.

---

## §5 — compute honesty

Accounting is clean: `overruns: []`, W34 shows exactly one charged job
(`jannolouwrens/jack-ladder-1787631708`, 0.3111 h, `ok: true`) with a real ledger
row. **No GPU hours have been spent without a ledger entry to show for them.**

The dishonesty is in the other direction — hours *not* spent:

| week | Kaggle unspent at expiry |
|---|---|
| W32 | ~8.8 of 30 |
| W33 | 22.11 of 30 |
| **W34** | **29.6889 of 30**, expiring **2026-08-30 00:00 UTC** |

Fourth consecutive week in the same shape, trending the wrong way. On a project
whose owner has ruled **free compute only**, roughly **60 free GPU-hours** will
have died unspent in three weeks. RANK 2 says a ~43 h window to spend some of
W34's still exists; RANK 1 says the one registered candidate for it cannot
currently be dispatched.

---

## §6 — stuck decisions

Covered in RANK 3. Beyond it:

- **No `MEANS-ESCALATED` entries.** Nothing a measurement could settle is on the
  owner's desk. The D1 disease is not present today.
- **`D1`** still costs **38 specs**, correctly armed, default upholds the
  plastic-only decree verbatim, due 2026-08-31.
- **`D10`** (8 specs) carries **four independent instruments** measuring W0 as too
  shallow — the darkroom control, LC.03 v2, DP.05, and the SH.01 oracle pilot.
  Evidence base complete, default sound. Still the single most valuable thing the
  owner could rule on early.
- **The all-dated-2026-08-31 defect** found by the 35th audit stands unchanged:
  ten armed entries, one date, and that date is *after* W34's hours expire and
  *at* the moment the meter resets. I have not moved any deadline; it is the
  owner's to shorten. RANK 2 now attaches a concrete cost to D13/D14 specifically
  — they fire ~19 h after the window they exist to protect has closed.

---

## §8 — the honest summary

**No. We are not closer to a curious humanoid that climbs the ladder than we were
yesterday.** The ladder reads **84/187 demonstrated (44.9%)**, unchanged for six
days. The last first-PASS of a *claim* about a capability of Jack's was
2026-08-20. In the 36 hours since the builder went dark this project produced, in
total, **one document** — the 35th audit — and zero measurements.

What is different today, and it is the reason this audit is worth its own cost:
**two of the three things this system believed were true are not.**

1. The project believed its GPU push guard made an unattributable result
   impossible. It does not — it has a hole exactly the shape of a brand-new file,
   and the property written to pin it shut contains a fixture of precisely that
   shape and still passes. **A guard-of-the-guard that tests the predicate and not
   the query is a guard-of-the-guard for the wrong thing.**
2. The project believed this week's free compute was already lost. It is not —
   there is a datable ~43-hour window, and the reason nobody could see it is that
   three audits in a row extrapolated a burn rate that had already stopped.

Both corrections point the same way, and it is not a comfortable direction:
**this system's instruments are now confidently wrong more often than its
science is.** §1 and §2 are clean, and I say so plainly — 83 PASS entries
re-derive from the record, no threshold has moved, no control has been weakened.
The falsification machinery is intact. But intact and idle is a museum, and the
last three audits, including this one, have spent the meter that keeps the
builder asleep in order to describe the sleeping. The strongest argument for
cutting this organ's cadence is that it keeps finding real defects **in
instruments**, which is what an organ looks for when there is no science to
audit.

The one genuinely good thing: **nothing lied.** Not one number in the ledger has
degraded. When the builder wakes on Aug 28, it wakes into a record it can trust.
It will get a handful of iterations before the hours expire, and RANK 1 and RANK
2 together say exactly how to spend the first one.

---

## FOR THE BUILDER

Ranked. **The pace gate will admit approximately five to sixteen iterations
between ~2026-08-28 05:07 UTC and 2026-08-30 00:00 UTC, and each one may be
followed by hours of silence** (RANK 2). Do not order this list cheapest-first.
**B1 and B2 must be in the FIRST admitted iteration, together, in one commit**,
because B1 without B2 will pass every gate and burn a Kaggle submission on a
kernel that cannot import the file.

- **B1 — commit `experiments/tests/sm_03_nose_reports_occluded.py` (sixth
  carry).** 32,086 bytes, untracked since 08-25 12:20. Its registry entry is
  sound and pre-registered (`f0cb81d`). The pilot meant to freeze its gates died
  and produced nothing, so **commit it as implemented-and-unpiloted and say so in
  the message** — do not invent pilot numbers, do not dispatch on gates frozen by
  guess, do not silently re-run the pilot as if the first had merely been slow.

- **B2 — close the untracked hole in `assert_ref_is_current` (RANK 1).**
  `experiments/gpu.py:274` reads
  `git("status", "--porcelain", "--untracked-files=no")`. Drop
  `--untracked-files=no`, so the guard asks git the same question
  `protocol.py:368` already asks. `is_code_dirt` / `NOT_CODE` already classify
  `??` lines correctly — this is a one-argument change, it **narrows** what is
  permitted and widens nothing, and it moves no threshold. Then **strengthen
  `T0.22 P15` so it pins the QUERY, not just the predicate**: assert that both
  organs derive their porcelain from the same `git status` invocation (compare
  the argv, or have both call one shared `porcelain_lines()` helper). P15's
  current fixture already contains `"?? experiments/tests/t9_99.py"` and passes
  while the live path is blind to it — that is the exact failure to make
  impossible. Re-run T0.22 through the runner to re-stamp it.

- **B3 — teach `decisions.py` to see an owner ruling written in an entry BODY
  (RANK 3).** `decisions.py:99` applies `_SETTLED` to `_HEADER` matches only, so
  `**DECIDED 2026-08-09: (a) RUN IT.**` in an entry's prose is invisible forever.
  Extend the scan to the body **and report it as a distinct state,
  `ANSWERED-UNCLOSED`, not as settled** — an answered-but-unexecuted decision is
  a debt and must stay loud. Do not simply widen `_SETTLED`: that would silence
  the physics-first entry, which is the one carrying a real debt. Keep
  `BASELINE_UNDECLARED` shrink-only.

- **B4 — close the two unrecorded decisions of RANK 3 / 35th-audit RANK 4.**
  (i) Move the physics-first entry to `DECISIONS_RESOLVED.md` with the owner's
  ruling **and** `T5.01` recorded as an **owed run queued behind T2.01**, so the
  debt is visible as a debt. (ii) "The owner's hands" is closed by me this audit;
  do not let it re-open.

- **B5 — extend the pace-skip rescue path to untracked spec implementations.**
  `HARVEST_PATHS` covers the three RUNNER_OUTPUTS and not the artifact class that
  has actually been at risk for 36 hours. Add a narrow, pathspec-explicit rescue
  for untracked files under `experiments/tests/` matching a registered spec id,
  committed with a message marking them **UNPILOTED**. Keep the `add -A` ban and
  the torn-file guard.

- **B6 — make an `rc=0` that certifies a corpse impossible** (carried from the
  35th audit, unactioned; two instances in eight days). After `run_claude`
  returns and before the `iteration end` line: if the iteration claims live
  background work, verify the pid is alive and its declared artifact is non-empty,
  else log `iteration end rc=2 — orphaned background work`. Route **every**
  long-running local pilot through `launch_detached.sh` so it inherits `setsid`,
  and state the contract in that launcher's docstring: *a `claude -p` iteration
  cannot be "re-invoked when it completes"; the process that would receive the
  notification exits first.*

- **B7 — fix the 06:37 cron collision** (carried twice, unactioned). `37 */6`
  (overseer) and `37 6 * * *` (review) fire simultaneously every day, putting two
  concurrent Opus sessions on the meter that gates the builder, 30 minutes before
  the 07:07 slot. Free fix: `37 3,9,15,21 * * *`.

- **B8 — log the model substitution as an event.** The next admitted slot runs on
  **Opus**, not Fable. Emit `MODEL SUBSTITUTION: primary <m> LIMITED, running on
  <fallback>` at the same prominence as the `PACING:` line, so the log records
  what the pool was actually spent on.

---

## FOR THE OWNER

Two things, and the first is different from what the last three audits told you.

**1. This week is not lost, and the window is datable — but it is narrow and it
needs one thing from the builder that it cannot currently do.**

The last three audits told you the free GPU-hours would expire before the builder
woke. That was extrapolated from a burn rate that has since **stopped**:
`week:all models` has sat at **62% for nine hours**, against a pace line
recovering at 0.387 pts/h. On that measurement the gate admits the builder at
**~2026-08-28 05:07 UTC**, and W34's **29.69 free Kaggle hours** live until
**2026-08-30 00:00 UTC** — a **~43-hour window**. I am giving you the earliest
admissible slot, not a promise: the meter is drawn down by sessions outside this
project and can resume at any time.

The catch is that the builder will wake on **Opus** (Fable is still at 100%), and
Opus bills the same meter that gates it — so it gets roughly **five to sixteen
iterations** in that window, not forty. And the one registered GPU candidate for
those hours, `SM.03` (the **smell** claim), currently **cannot be dispatched
correctly**: its implementation is untracked, and the guard that exists to catch
exactly that reads `git status --untracked-files=no` and reports a clean tree.
Builder items B1+B2 fix both in one commit. Nothing here needs you — I am telling
you because it changes the answer you were given yesterday.

**2. The armed defaults are still dated 2026-08-31, and RANK 2 now prices that.**
This was the 35th audit's ask and I am repeating it with a sharper number rather
than a new argument:

- Kaggle W34's **29.69 free GPU-hours die 2026-08-30 00:00 UTC**.
- **D13** (halve this organ's cadence) and **D14** (make the gate read the meter
  that actually binds) both fire **2026-08-31** — about **19 hours after the
  window they exist to protect has closed**, and on the day the meter refills and
  makes both questions moot.

I did not re-date them. A shortened deadline shortens *your* window to answer and
that is your call. **If you rule on nothing else, ruling D13 and D14 early — or
authorising a re-date to 2026-08-28 — is what converts a narrow window into a
usable one.**

One thing that does **not** need you: `D10` has four independent instruments
behind it and a sound default. If you have one minute rather than ten, spend it
on D13/D14.

And the standing offer, still against my own organ: **cut this audit's cadence
before anything that produces science.** Four Opus audits a day against a tree
that has been byte-identical for 36 hours is the clearest waste in the picture.
The counterargument is honest and I will state it rather than bury it — this
audit found two real defects, both in instruments the system believed were
sound, and neither was findable from inside the builder. D13's change-gated
no-op (option (c)) resolves that tension correctly: it skips when nothing has
changed and still runs when something has.
