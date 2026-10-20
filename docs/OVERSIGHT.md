# OVERSIGHT.md — the overseer's current-state report

> Current state, not a log. Each audit rewrites this file.

**2026-10-20 06:37–07:1x UTC — 154th audit.** Second audit slot of the day to be
*permitted*: `18:37` (10-19) and `00:37` (10-20) both read
`STOPPED at 95% weekly usage`. The Review's `06:37` DAILY started at the same
instant as this audit and was still sitting — and committing — while this was
written, so every `review-queue` count below carries the clock at which I read
it. The queue moved under me four times during the audit: **OVERDUE 20 (06:38)
→ 19 → 17 → 16 (06:49)**, as the Review disposed rows in `98e61e4`, `b6a4973`,
`d8cf347`, `938fa2e`. That fall is the Review working correctly and I am not
claiming credit for it or against it; the reading my findings are argued on is
the **06:38** one, and the live direction is **down**.

True exit codes this audit, taken from the processes and not from a pipeline:
`coverage` **2** · `decisions --check` **0** · `champions --check` **0** ·
`run review-queue` **2** · `run status` **2**.

## VERDICT: INTEGRITY RISK

**The gate that paused this project 18 times in the last fifteen hours was
acting on a number that cannot be true.** Not "a number I disagree with" — a
number that violates an invariant the meter itself declares. `week:all models`
is a cumulative weekly percentage inside a window the CLI prints. That window is
**2026-10-19 05:00 → 2026-10-26 05:00 UTC** (`claude_usage.reset_epoch`, 15% of
the week elapsed at audit time). Inside that single unchanged window the gate
read:

```
10-19 07:07   7%   usage_ledger.jsonl   builder start
10-19 15:22  42%   usage_ledger.jsonl   review start   <- last true-looking read
10-19 16:07  95%   ladder.log           STOPPED
10-19 18:22  95%   review.log           DEFERRED
10-19 18:37  95%   overseer.log         STOPPED
10-20 00:37  95%   overseer.log         STOPPED
10-19 19:07 … 10-20 06:07  95% x 12     ladder.log     STOPPED, hourly
10-20 06:37   0%   usage_ledger.jsonl   review start + overseer start
10-20 06:44   2%   my own read          `resets Oct 26, 4:59am (UTC)` — same window
```

A cumulative meter does not go 42 → 95 → 0 → 2 inside one window. At least one
of those readings is false, and the 95% plateau is the one that spent the
project: **15 builder slots, 2 overseer audits and 1 Review retry poll were
refused on it.** The same shape fired twelve hours earlier and nobody caught it
either: at `10-19 06:37`, 1.6 h into a *fresh* window, the meter read **93%**,
and at `07:07` it read **7%** — the 06:37 read was the previous week's value
served past its own reset, and it cost the Review its `06:37` DAILY sitting on
the first day of a full budget. The builder saw that pair, wrote *"the shared
meter read 93% at 06:07 and 6% at 07:07 — off-box movement, read not modelled"*,
and charitably filed it as someone else's spend. It is not off-box movement. It
is the gate's input being wrong twice in one day.

**Why it survived two episodes and three audits.**
`lib_usage.sh:usage_ledger()` records `pct` and `model_pct` and **not the
`resets` instant** — so the one invariant that proves the fault (pct may not
fall while `resets` holds) is uncomputable from the record. And
`usage_ledger()` is only called when an organ *starts or ends*; a gate-refused
slot starts nothing, so **all 18 false readings exist only as English prose in
three log files.** The 153rd audit quoted `STOPPED at 90% / 93%` as fact in its
own opening paragraph. Neither it nor the Review had anything to check it
against.

**The Review reached the firing independently this morning, and that matters to
how this finding should be read — §A.4.** Its `06:37` DAILY (`522db77`,
committed 06:50, mid-audit) corrected yesterday's pacer attribution, established
that the **90% hard stop** is what fired, found that **nothing in the repo
records the firing**, and routed+executed that as `D50`. That is the same fault
my B1 addresses and the Review got there first; I am not re-reporting it as a
discovery. **What it then concluded is the part this audit falsifies.**

Ranked by damage to the trustworthiness of the record:

1. **The usage gate acted 18 times on a provably invalid meter reading, the
   project cannot detect this class at all, and the Review has just written
   "the blackout's cause is gone" onto the current-state page on the strength of
   the very reading that proves the meter wrong.** There was no weekly reset:
   the window has six days to run. §A, §A.4.
2. **`D46` — the live entry asking what to do about this very gate — is argued
   on a premise this audit falsifies**, is `class conduct` / `decide_by
   2026-10-17` (3 days stale), and `decisions --check` tells the desk to execute
   it rather than ask. Evidence addendum appended. §A.3.
3. **`scripts/ladder_prompt.md` is 1,721 B OVER its own 125,000 B ceiling and
   4,351 B below the 131,072 B EXEC CLIFF, growing +1,077 B/day — 4 days.** Past
   it the builder does not start and the slot is indistinguishable from `rc=126`.
   Three organs edit this file routinely. §C.
4. **The certificate that guarantees §1 of this audit holds a PASS on a FAIL
   dependency, and has for 24 days.** `T0.18` *"Every PASS is re-derivable from
   the record, and every control is read"* declares `depends_on [T0.08, T0.13]`;
   `T0.13` *"No gate in the ladder is decorative"* is FAIL. §1.
5. **Two more weeks of free GPU died unbought — and the honest finding is still
   not the hours.** §5.

**§2 is clean and I want that said plainly: not one threshold moved in the
loosening direction in seven days.** §1 is clean on its own terms too. The
integrity risk in the verdict is not in the ledger's content; it is that the
machine which decides whether the project may *run* is taking dictation from an
unvalidated external scalar, and has now demonstrably been wrong twice in
twenty-four hours.

---

## §A — RANK 1: the gate's input is false, and nothing in the repo can say so

### A.1 The measurement

Re-derived at source this audit, not inherited:

- `scripts/claude_usage.py:read()` parses `claude -p /usage`. `reset_epoch()`
  turns the printed reset instant into an epoch. I ran it: `resets Oct 26,
  05:00 UTC`, so the window opened **2026-10-19 05:00 UTC** and
  `--week-elapsed` prints **15**.
- `scripts/lib_usage.sh:_usage_pct()` reads live every call, with one 5 s retry
  **only** when the output is unparseable. There is no local cache, so the flat
  95% across 15 hourly reads was the CLI itself reporting 95% each time — this
  is not a stale variable on our side.
- `usage_gate()` compares that scalar to 90 and, with no `.usage-resumed`
  present (verified absent), returns 1. That is correct behaviour given its
  input. **The defect is upstream of the policy, and the policy has no way to
  doubt it.**

Two readings of the drop are available and I will not pretend to settle it from
this box: either the meter misreported, or the account's weekly ceiling was
raised mid-window (which would lower the percentage without lowering absolute
spend). **The repair is the same under both** — and under both, 18 organ slots
were refused on a number that is false by the time anyone audits it.

### A.2 The cost, counted off the logs this sitting

| | |
|---|---|
| builder slots refused `STOPPED at 95%` | **15** (10-19 16:07 → 10-20 06:07, hourly) |
| overseer audits refused | **2** (10-19 18:37, 10-20 00:37) |
| Review polls refused | **1** (10-19 18:22) — plus `06:37` lost to the 93% read |
| `dark_slots` ratchet | **0 → 15**, and **ABOVE its declared floor of 0** |
| `review_queue` violations | committed reading **8** → **20 at 06:38**, i.e. **OVERDUE +11 in one day**; live **16** at 06:49 as the Review drains it |

The queue number is the one that matters, because it is the measured
*consequence* and it is attributed correctly by the instrument itself:
`run status` prints `OVERDUE +11 — CLOCK: the calendar reached a date; a real
event with a real owner, and no commit is to blame for the rise.` The owner of
that date was the 10-19 Review DAILY, which had five chances to sit and took
none: `06:37` deferred on the **93% false read**, `09:22` and `12:22` aborted on
genuine tenant load (7.64, 6.44), `15:22` refused at the door by a **genuine**
session limit, `18:22` deferred on the **95% false read**. **Two of the five
were lost to data this audit can prove wrong.** The 10-18 Sunday FULL had driven
OVERDUE to 0 and said so to the owner; it is 17+ two days later.

### A.3 What I did with it, and the line I did not cross

Appended a **`D46` EVIDENCE ADDENDUM** to `docs/DECISIONS_NEEDED.md` (same idiom
as `D40`'s three addenda, two of them written by this organ): not a re-opening,
not a new decision, nothing fired, no `decide_by` moved, `PACE_FLOOR` /
`PACE_CAP` / the 90% stop untouched. `D46` asks the owner about a gate "that has
held the builder dark for 119 slots" and prices it as a *consumption* problem.
That premise is now partly false, and the correction's direction is the same one
the 144th audit had to make to `D40`: **the instrument's error blames the
builder.**

I did not write the repair. The code half is specified for the builder below and
is deliberately split so that no part of it can loosen the stop.

### A.4 The Review closed this finding four minutes before I opened it, on the strength of the reading that proves it

`docs/PROGRESS.md`, written by today's DAILY and now the project's current-state
page, says:

> *"as of this morning `usage_ledger.jsonl` reads `pct: 0` for both organ
> starts — **the usage week has reset and the gate has RELEASED.** The
> blackout's cause is gone; its debt is not."*

**There was no reset.** `claude_usage.reset_epoch()` resolves the live window to
`2026-10-26 05:00 UTC` and `--week-elapsed` prints **15**, so the window opened
**2026-10-19 05:00** and has **six days left**. The `95%` at `06:07` and the
`0%` at `06:37` are **thirty minutes apart inside the same window.** The gate
did not release because the week turned; the meter changed its mind.

So the one sentence on the current-state page that tells the next reader the
fault is finished is the fault itself, misread as its own cure. **The
consequence is specific and near: the 95% can recur any hour of the next six
days, and the page now says the cause is gone.** The Review's *operational*
advice is unaffected and correct — the gate is genuinely open right now, because
the meter genuinely reads 2–3% — so item 0's "spend the perishable week" is
right, for the wrong reason. It is the causal claim that must not stand.

**This is the second time in twenty-four hours that an organ explained an
impossible reading by assuming the meter was honest.** The builder wrote
*"off-box movement, read not modelled"* for the `93% → 7%` pair at 07:07
yesterday; the Review wrote *"the week has reset"* for the `95% → 0%` pair at
06:37 today. Both inferences are charitable, both are the natural reading, and
both preserve the meter's truthfulness at the cost of the finding. Neither organ
was careless — **the invariant that refutes both is one subtraction, and nothing
in the repo was recording the operand.**

---

## §1 — Integrity of the ledger: CLEAN, with one structural caveat

Checked independently rather than by reading `T0.18`'s certificate, for the
reason in §1b. All 160 rows, 107 PASS:

| check | result |
|---|---|
| PASS rows whose `commit` no longer resolves in git | **0 / 107** |
| PASS rows whose impl module will not resolve via `run._module_for` | **0 / 107** |
| PASS rows missing `impl_sha` | **0 / 107** |
| PASS rows with no declared `control` | **0 / 107** |
| PASS rows with empty `control_metrics` | **2** — `T0.01`, `T0.10` |

Both exceptions are adjudicated, not holes: each declares a typed
`protocol.NoControlByDecision` (a *falsy* `str` subclass, so it cannot be
mistaken for a control by any reader) carrying the 52nd audit's B5 reasoning —
`T0.01` "an import either raises or it does not"; `T0.10` "a sabotaged upload
fails on the service's side, which is the falsifier itself". The 10-18 Review
sweep reached the same verdict on `T0.10` independently. **No PASS in this
ledger is a claim without evidence.**

### §1b — the caveat, and it is the reason I did the above by hand

`T0.18` — *"Every PASS is re-derivable from the record, and every control is
read"* — is the instrument that would normally discharge this section. It holds
**PASS, attempt 6, ran 2026-08-30**, and declares `depends_on [T0.08, T0.13]`.
`T0.13` — *"No gate in the ladder is decorative"* — has been **FAIL since
2026-09-27**, at 109 gates with **28 disarmed conjunct keys** against a
certificate last bought at 90 gates. `T0.19` sits on the same FAIL.

`coverage` **exits 2** and names all six pairs of
`pass_on_dead_dependency = 6`, **a shrink-only ratchet sitting at double its
declared floor of 3** since 10-04:

```
LF.02 <- T6.03 BLOCKED     T0.18 <- T0.13 FAIL     T0.19 <- T0.13 FAIL
T1.12 <- T1.11 FAIL        T2.03 <- T1.08 FAIL     T2.14 <- T1.08 FAIL
```

Two of those deserve naming beyond the count. **`T2.14`'s own PASS rests on
`T1.08` FAIL** — and `T2.14` is the spec whose `task_not_trivial_all` guard the
10-18 Review ordered ported into `T2.05` as the project's single most important
unblocked item. The port is right and §2 confirms it is strictly strengthening;
it is nonetheless being copied *from* a certificate standing on a dead
dependency, and `run status` already flags the order as
`SUBJECT T2.14 BLOCKED — T1.08 (FAIL)`. **`T2.03` is the second**, and it is a
Tier-2 vs-null claim. `coverage` also exits 2 on `unreachable = 96 vs baseline
95` — pre-existing, documented in the counter's own growth log as an honest
demotion's downstream (`LT.02`), and correctly not papered over by raising the
floor.
This is known, noted in the counter's own growth log, and owned by
`t013-latently-red-28-disarmed-keys` (OPEN, 24 d, DUE **2026-10-23** — I checked
for a phantom OVERDUE here and there isn't one; the row carries two `DUE:` lines
and the reader correctly takes the later). It is also *structurally* stuck: a
re-run of `T0.18` records BLOCKED on `T0.13` and never reaches its own `_check`,
so the certificate cannot be re-bought until the per-key adjudication lands.

**My finding is not that the row is late. It is that for 24 days the project's
ledger-integrity guarantee has rested on a certificate whose own declared
dependency says the gates might be decorative — and that no audit in that window
re-derived §1 by hand.** This one did. It came back clean. That is worth
knowing, and it is not a substitute for `T0.13`.

---

## §2 — Thresholds and controls over seven days: NO FINDINGS

`git log -p --since="7 days ago" -- experiments/registry.py
experiments/registry_expansion.py experiments/tests/` — five commits. Every one
read at the diff, not the message:

- **`b6069dc`** `T2.05`: ports `T2.14`'s `task_not_trivial_all` +
  `floor_selftest_ok`. **ADDS two VOID gates**, adds a known-answer self-test
  that must *fire* on a linear target and *pass* on a non-linear one, reports
  `mse_nonlinear` / `nonlinear_headroom` without gating on them.
  `TRIVIAL_FLOOR = 0.05` copied **unchanged**; `WM_RATIO` stays 0.8, `CTRL_TOL`
  stays 0.98, the claim conjunct is byte-untouched. Strictly stronger. This was
  the 10-18 Review's FTB item 1, **executed 6 days before its 10-25 due date.**
- **`1ee9c76`** `rtf.require_feasible` adopted in `LT.03`/`SO.07`/`PL.02`: adds
  a standing pre-dispatch refusal that routes an infeasible run to a VOID lane
  (`rtf_admitted 0.0`), never to FAIL. Adds a gate; removes none.
- **`1ee9c76`** also re-declares `W1.02` `Budget.CPU_LONG → Budget.CPU`
  (`cpu<2h → cpu<10min`). **This is the only change in seven days that could be
  read as loosening, and it is not**: a *smaller* declared class refuses more,
  not less, and the commit carries the arithmetic (0.030 s/decision × 28,000
  decisions ≈ 840 core-s/experiment-seed — 2.8× over `cpu<1min`'s 300 s, 2.1×
  margin inside `cpu<10min`'s 1,800 s). `seeds=3` unchanged.
- **`fbf58c2`** repairs `D49`'s illegal register class and `T0.22`'s P9
  over-match; the P9 narrowing *reports* what it exempts (`own_row_reads`)
  instead of hiding it, and the legacy control still fails P9 (0 vs 12).
- **`87fc826`** `LG.03`: comment-only, verified by
  `git diff -U0 | grep -v '^[+-]\s*#'` returning nothing.

No numeric threshold moved in the loosening direction. No control deleted or
weakened. No `_check` gained an `or`. No seed count reduced. No assertion
removed. **Section 2 is clean, and that is a real result, not a blank.**

---

## §3 — Drift from the goal

**What the builder worked on in its 6 live slots (10-19 07:07 → 13:29), and what
each serves:** `T0.28`/`T0.22` red repairs, the regate dirty-tree guard, the
`T2.05` triviality-floor port, five `+dirty` certificate re-buys, the
`crossing_a_class_slack` fix, `D41`/`D42` transcription, the `rc=0`-mid-wait
routing. Every one of these traces to GOAL.md's *"protects the honesty of
watching what happens when the three meet"* clause. **None of them traces to
building the brain, the body, or the world.** The builder says so itself, every
slot, in the same words: *"Creature gate: NONE"* — three consecutive slots, with
the forced derivation re-run each time (`T2.01` behind the prohibited `T1.08`,
`XL.01` at the desk, `T6.01` blocked through `T4.04`/`T4.05`).

This is not drift in the dishonest sense. It is worse in a quieter way: **the
builder-legal moves toward Jack are exhausted, the organs know it, they say it
out loud every hour, and the system's response is to keep producing excellent
instruments.** The 10-18 Review named the same thing — *"we are busier, not
closer"* — and its Goodhart check is the sharpest sentence any organ has written
this month.

**The converse, which is the harder question.** From `coverage.py` this audit —
0 commitments with no spec at all (the 2026-08-10 scar stays closed), but:

| GOAL.md commitment | specs | PASSING claims |
|---|---|---|
| smell · shelter/building · thermal ("too cold kills him") | 2 / 2 / 4 | **0 — CLAIM-DEAD**, every claim spec PARKED or FORECLOSED |
| plasticity | 4 | **0** |
| sleep | 5 | **0** |
| fast/slow (the owner's 2026-08-10 directive) | 8 | **0** |
| one brain / unison | 28 | **1** |
| curiosity | 12 | **2** |
| told world | 1 | **0** |
| tool use | 1 | **0**, and **no champion seat** |
| self-model | 1 | **0**, and **no seat distinct from Person model** |

**14 commitments have live claim specs and nothing passing.** Tier 5 — the file
literally headed *"THE CLAIMS"* — is **0 of 7 implemented**. The three
CLAIM-DEAD commitments are the owner's own image of success ("too cold kills
him", "he builds a shelter"), their named queue row
`five-commitments-are-claim-dead-behind-foreclosures` is **DECLINED**, and the
`D44`-adjacent firing block at the tail of `DECISIONS_NEEDED.md` already records
that no instrument can *ask* for the repair because a missing successor spec has
no id. That block's reversal condition is one sentence from the owner naming one
commitment. It is the cheapest unblocking act available to anyone in this
system and it is not mine to take.

`champions --check` exits 0 with **ARENA-MISSING at 0** — note for the next
reader that the overseer prompt's *"8 seats today"* is a dated example and reads
0 now. Standing: `NO-ARENA` on ASR and Speaker ID, `UNCONTESTED` on Vision
encoder and PLASTIC-ONLY, trigger debt 3/3 (Learning core, Fast/slow coupling,
World). The 10-18 Review committed to adding the **tool use** and **self-model**
seats as "next Sunday's first act" (10-25) and gave a good reason for not doing
it in a timed tail — a malformed `CHAMPIONS.md` breaks its only reader. Held to
that date, not re-routed.

---

## §4 — Is the builder alive and productive?

24 h to 06:44, all 24 scheduled slots accounted for:

| outcome | n |
|---|---|
| `rc=0` | **6** (07:07, 08:07, 09:07, 10:07, 12:07, 13:07) |
| `rc=1` lost to a genuine session limit | 1 (15:07 — every model in the chain) |
| `ABORT: load` — tenants first, correct behaviour | 2 (11:07, 14:07) |
| `STOPPED at 95%` — **the false reading** | **15** |

**PASS delta: 105 → 107 (+2)**, both booked in the 07:07 slot, both `T0.*`
instrument re-buys (`T0.28`, `T0.22`). `107/257 = 41.6%`. The builder is not
broken, is not refusing, and `failed_slots` is 1. Its six live slots were
genuinely good work. It then lost the next fifteen to §A.

**Organ liveness.** Field watch: `FIELD_WATCH.md` last moved **2026-10-12**;
its weekly slot is `37 5 * * 1`, so 10-19 was due and the log's last line is
10-19 05:37 — needs the next Monday to confirm, and the 153rd audit's finding
that the Review reports it live while it is dark is **not** reproduced today.
Review: STALE banner still on `docs/PROGRESS.md` from 10-19 12:37, and the
`06:37` sitting of today is in flight as this is written. Overseer: 2 of the
last 4 slots refused by §A.

---

## §5 — Compute honesty

`experiments/gpu_budget.json`, summed by week this audit:

```
W35 19.201 h   W36 17.727 h   W37 5.249 h   W38 0.918 h   W39 1.072 h
W40 1.971 h    W41 0.000 h    W42 0.000 h  (0 charged jobs; last job is W40)
```

**`2026-W41`'s entire 30.0 free Kaggle hours expired unbought.** `2026-W42`
holds 30.0 with **0.000 drawn, expiring Saturday 2026-10-24 — 4 days.** Across
the last five weeks the project drew **3.96 h of ~150 available**. Two
`overruns` are recorded for W40 (0.65 vs 0.50 est; 1.05 vs 0.50 est) — both
honestly booked, both small.

**And the honest part, which the builder states every slot and I confirm: there
is no designed buyer.** No spec is waiting on a GPU; the hours would very likely
have expired even with every organ awake. **The waste is not the hours. It is
that this is the sixth consecutive week nobody has been able to decide whether
to spend them** — which is exactly what `D30`/`D40` were raised about and
exactly what `D40`'s own firing recorded. Reported, not re-routed: a seventh copy
of this entry on the owner's desk would be noise.

---

## §6 — Stuck decisions

`decisions --check` **EXIT 0**, ratchet clean: `0/0 undeclared, 0/0
unrouted-owner-ask, 0/0 vanished-owner-ask, 0/0 default-action-expired, 0/0
firing-diff`; **34 of 34 firings transcribed** to `DECISIONS_RESOLVED.md`.

- **`MEANS-ESCALATED`: none.** No fork a measurement could settle is sitting on
  the owner's desk. The `D1` disease is not present.
- **`UNDECLARED`: none** — so there is nothing to arm this audit, and I am not
  manufacturing an entry to satisfy the quota. The ratchet reads `0/0`.
- **`OVERDUE — DEFAULT DUE TO FIRE`: none.** `D44` is armed, `decide_by
  2026-10-24`, not yet due. I re-read its default and confirm it is MONOTONE and
  picks only among already-permitted actions.
- **Six `CONDUCT-DESK` entries are stale**: `D33` (27 d), `D35` (26 d), `D38`
  (16 d), `D40` (10 d), `D39` (5 d), `D46` (3 d). The tool's own words are
  *"desk-executable, not the owner's — execute it, report it, do not ask."*
  These are not blocked on the owner and must stop being counted as if they
  were. `D46` is the live one and §A.3 corrects its premise.
- **`PROGRESS.md` owner-asks**: all 5 reached a desk and are correctly
  attributed (`D46`, `D47`, `D48`, `D30`). No `UNROUTED-OWNER-ASK`, no
  `VANISHED-OWNER-ASK`. The 2026-09-03 failure mode that put this page in my
  prompt is not recurring.
- **Nothing acted on without being recorded**, checked the other direction:
  `D41`/`D42` were transcribed same-day (`fadf712`) and the builder self-caught
  one wrong number in its own draft (`e091864`).
- **`D50` was routed by the Review mid-audit** (`938fa2e`, act 4 — a silently
  unparsed declared field). For about eleven minutes `decisions --check` read
  `RATCHET BROKEN: 1 UNDECLARED, baseline 0`; it now reads `CONDUCT-DESK` and
  the ratchet is back to `0/0`. I checked whether my own addendum caused it — it
  did not; the window is the Review's entry between its heading and its class
  line. **Recorded because it is a live instance of the cross-organ doc race
  that `cross-organ-doc-race-voids-certificates` was routed for: a ratchet that
  reads BROKEN for eleven minutes will eventually be read in that window by an
  organ that then reports a violation nobody committed.**

## §7 — Bakeoff hygiene

`docs/DECISIONS_RESOLVED.md` against the firing record: no decision made
without a learning gate; no VOID treated as a verdict; `champions --check`
flags `VERDICT-IS-A-VOID` on the **Learning core** seat, which is the honest
marking of exactly that risk rather than a hidden instance of it. One item to
keep visible rather than call a violation: `T4.06`'s recorded anchor-relative
margins show `loss_reweight` CERTIFIED at **+6.9%** and **+8.9% of the anchor's
own seed spread** — inside the noise by any ordinary reading. `run status` prints
it as REPORTING-ONLY under the SO.10 vacancy precedent, which is defensible
because the spec pre-registered an exogenous gate rather than an anchor-relative
one. **But a winner chosen at 7% of seed spread is the shape §7 exists to catch,
and it should be named in CHAMPIONS.md as such rather than left for the next
reader to re-derive.** Carried to FOR THE BUILDER as a reporting item only — I
am not touching the verdict.

---

## §C — the exec cliff, because it can silence the builder with no error

`run status`, this audit: `scripts/ladder_prompt.md` is **126,721 B**, **4,351 B
below the 131,072 B `MAX_ARG_STRLEN` EXEC CLIFF**, and **1,721 B OVER the
125,000 B self-imposed ceiling**, growing **+1,077 B/day over 15 commits → the
cliff in 4 days**. Past it the builder does not read a degraded prompt: it does
not start, and the slot is indistinguishable from any other `rc=126`.

The 10-18 Review measured this by walking into it — its `1^23` priority block
left **782 B** of margin — then reverted its own edit, correctly, and reported
it. Its recommendation was *"treat the 125,000 ceiling as a hard refusal rather
than a reporting line."* **One thing nobody has said about that
recommendation: the page is already past 125,000, so implemented literally it
would refuse every organ immediately, including the archival commit that is the
actual repair.** The ordering therefore matters and is not optional: archive
first, then arm the refusal. Named here so the next editor of that file — any of
three organs, routinely — does not discover the cliff by crossing it.

---

## FOR THE BUILDER

**Scope note, so B1/B2 are not read as a second copy of `D50`.** `D50`
(routed and executed by today's Review) makes the 90% stop's firing a
**reported** event and standing desk practice — prose, written by a desk that
sits. B1/B2 make it a **checkable** one: the field that lets a program contradict
the meter, and the reading that does. `D50` would not have caught today's
misdiagnosis, because a desk reporting "the stop fired and then released" is
exactly what happened. Do B1/B2 *as well*, not *instead*.

**B1. Record what the gate acted on. Two fields, no policy change.** In
`scripts/lib_usage.sh:usage_ledger()`, add the weekly **`resets` string** beside
`pct` (it is already in the same `claude_usage.py` output the function parses —
no extra CLI call), and emit a row on the **refusal** path of `usage_gate()` and
the skip path of `pace_gate()` with `"phase":"refused"` / `"phase":"skipped"`.
Today a gate-refused slot writes nothing, which is why all 18 false readings in
§A exist only as English. Keep the existing contract: unreadable writes `null`,
a failed write is swallowed, and this never blocks a run.

**B2. Then make the invariant checkable, and only report it.** With B1 landed, a
weekly `pct` that **decreases while `resets` is unchanged** proves at least one
of the pair false. Add that as a reading in `usage_attribution.py` —
`METER-NON-MONOTONE`, with both timestamps and both values — **reporting-only
and unfloored**, in the idiom of D26's armed default (`MEASURE ONLY, GATE
NOTHING, RELAX NOTHING`). It must not gate, must not refuse, must not move
`PACE_FLOOR`, `PACE_CAP` or the 90% stop.

**B3. Require a STOP to be measured twice, and be honest that this is for the
blink class only.** `_usage_pct()` already re-reads once when the output is
unparseable. Extend that same retry to the case where a reading would cross the
90% stop: two agreeing reads before pausing every agent. This refuses strictly
nothing new — if both read ≥90 it stops exactly as today — so it cannot loosen
the gate. **It would NOT have saved the 15 slots in §A**, because that reading
persisted across 15 reads an hour apart. Do not let it be reported as the fix
for §A; it is the fix for the two-in-500 blink class `lib_usage.sh` already
documents.

**B4. `T0.22`'s guard comment.** Carried from the 10-18 Review's FTB 3 and still
live: the comment says "All nine ran" while `N_PROPERTIES = 15`. No effect on
the gate. Ride it along with the next edit that bills `T0.22` anyway — do not
stale a certificate for a comment.

**B5. Name `T4.06`'s margin where a reader will meet it.** §7: add one line to
`docs/CHAMPIONS.md` beside the Sensory-fusion / fusion-balancing seat recording
that `loss_reweight` was certified at **+6.9% / +8.9% of the anchor's own seed
spread**. Reporting only. **Do not re-run it, do not touch the verdict, do not
move the gate** — the point is that the next reader should not have to
re-derive the margin to know how thin it is.

**B6. `bare-git-commit-writes-the-whole-shared-index` fired on this audit, and
the evidence is in the register.** My `D46` evidence addendum was written to
`docs/DECISIONS_NEEDED.md` at 06:46 and was **uncommitted**. The Review's
`522db77` (06:50:42) swept it in. Nothing was lost — the text is in HEAD and
intact — but **137 lines of overseer-authored decision-register prose are now
attributed to a Review commit whose subject describes the usage blackout.**
Provenance in that file is load-bearing: who wrote an addendum determines
whether it is a desk act or an ask on the owner, and `decisions.py` attributes
by entry, not by author. The row is already OPEN and OVERDUE with the repair
specified (commit by pathspec, never a bare `git commit`); this is a dated,
reproducible instance to attach to it. I committed this audit by explicit
pathspec for the same reason.

**Not for the builder, said so it is not picked up by mistake:** the archival of
`ladder_prompt.md` (§C) belongs to its existing queue row; the `T0.13` per-key
adjudication (§1b) is explicitly not the builder's and is blocked behind the
FAIL itself; and the policy half of §A is on `D46`.

## FOR THE OWNER

**1. The number that has been pausing your project is not reliable, and this is
the first audit able to prove it.** Window `2026-10-19 05:00 → 2026-10-26 05:00`
UTC; readings inside it went **7% → 42% → 95% (15 reads over 14 h) → 0% → 2%**.
A weekly cumulative meter cannot fall inside its own window. **Cost in the last
fifteen hours: 15 builder slots, 2 overseer audits, 1 Review poll — and
`review-queue`'s OVERDUE class went 0 → 17+ in two days**, because the Review
sitting that owned those dates lost two of its five chances to sit to this same
bad data. Full trace in §A with every log line and timestamp.

**1b. Your Review desk found the same firing this morning, independently, and
then wrote that it was over.** It corrected yesterday's misattribution, proved
the 90% stop was the gate, found that nothing records the firing, and routed and
discharged that as `D50` — good work, and it got there before this audit did.
But it read today's `0%` as a weekly reset and wrote **"the blackout's cause is
gone"** onto the current-state page. The window has six days left; the 95% and
the 0% are half an hour apart inside it. **The cause is not gone, it is
unexplained, and the page now tells the next reader not to look.** §A.4.

**2. `D46` is the entry about this gate, and its premise is now partly wrong.**
I appended an EVIDENCE ADDENDUM (nothing fired, no `decide_by` moved, no
threshold touched, same idiom as `D40`'s). `D46` prices the blackout as
consumption. Part of it is a misread, and the error direction is the one the
144th audit already had to correct on `D40`: **it blames the builder.** Note
also that `decisions --check` classes `D46` as `CONDUCT-DESK` — *"desk-
executable, not the owner's"* — so if you are waiting to be asked, you are
waiting on something the desks have been told to execute. It is 3 days past its
`decide_by` with five other conduct entries stale behind it (up to 27 days).

**3. The one question only you can answer is the policy half, and it is one
sentence.** §A's repair is split so nothing can loosen the 90% stop by accident:
the builder gets recording and reporting (B1, B2), which makes the fault visible
but recovers no slots. **What is left is: when the meter says STOP and the
system can show the reading is invalid, may the organs run?** Today the answer
is no, by silence, and that silence cost 18 slots in fifteen hours. I have not
armed a default for this, because every option I can construct either widens
what the builder may spend or invents a second limit you did not set — both
forbidden to a desk default. It needs you.

**4. Three of your own named commitments still have no passing claim, and the
instruments cannot ask for the repair.** smell, shelter/building, and thermal
("too cold kills him") are **CLAIM-DEAD**: every claim spec PARKED or
FORECLOSED, the named queue row DECLINED, and a missing successor spec has no id
so nothing can block on it. The reversal is one sentence naming **one**
commitment — the desks' standing recommendation is **thermal**, because it is
load-bearing for the other two and its successor needs no world edit. Carried
forward unchanged, not re-asked.

**5. Four days to W42's GPU expiry, and the hours are still not the argument.**
30.0 free Kaggle hours, 0.000 drawn, expiring Saturday 2026-10-24; W41's 30.0
expired unbought; 3.96 h drawn of ~150 across five weeks. There is no spec
waiting on a GPU, so they would likely have expired regardless — §5. The sixth
consecutive week of being unable to decide is the finding.

---

## §8 — The honest summary

**No. We are not closer to a curious humanoid that climbs the ladder than we
were yesterday, and this week we were not closer than we were a month ago
either.**

The ladder moved 105 → 107. Both of those are Tier-0 instrument self-tests
re-bought after the regate lane surfaced them red — they measure the measuring
apparatus. Tier 5, the tier whose heading is *"THE CLAIMS"*, is **0 of 7
implemented**. `fast/slow` — an explicit owner directive — is 0 of 8. Plasticity
0 of 4, sleep 0 of 5, unison 1 of 28, curiosity 2 of 12. Three commitments the
owner wrote in his own words are CLAIM-DEAD. The builder has now printed
*"Creature gate: NONE"* for three consecutive live slots, having re-derived it
each time rather than inheriting it, and it is right.

What *did* happen is real and I will not sneer at it: a dirty-tree guard shipped
five days early, a triviality floor ported six days early that converts a
43-day-old unattributable FAIL into an attributable one, two latent reds caught
and repaired within hours, and a seven-day window with not one threshold moved
in the loosening direction. This project's instruments are in better health than
most research codebases ever reach. **That is precisely the problem to name: the
honest verdict is that we have built an extraordinary apparatus for measuring a
creature we are no longer building.** Six live builder slots in twenty-four
hours, all six spent on the apparatus, fifteen more refused by a number that
was false.

The 10-18 Review wrote *"the rate is now measuring the re-buy lane, not the
ladder."* That is still true, and this audit adds the sharper version: **the
constraint is no longer capability, scheduling, or compute. It is that the
builder-legal path to Jack is exhausted and has been for three slots, every
organ knows it and says so hourly, and the two acts that would re-open it — one
sentence naming a CLAIM-DEAD commitment, and a ruling on the gate in §A — are
both on a desk that has six stale conduct entries and one human.**
