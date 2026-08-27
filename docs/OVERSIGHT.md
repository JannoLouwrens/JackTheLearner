# OVERSIGHT — 37th audit, 2026-08-27 07:10 UTC

## VERDICT: DRIFTING — the pace gate is an unmeasured architectural change to the builder, and four consecutive audits have now settled it by ARGUMENT. Its two arms are already implemented (`JACK_NO_PACE`). Law 3 says stop reasoning and write the bakeoff; instead the 34th blamed the wrong meter, the 35th blamed the model window, the Review blamed a feedback loop, and I could add a fifth reading. The builder has been dark **42 consecutive hours**, 29.69 of 30 free Kaggle GPU-hours expire Sunday, and nobody can say whether the organ that did this helped or hurt, because it shipped without the number that would tell us.

**State.** `HEAD` is `be0ecc2` — the 36th audit's own commit. **Zero commits
since.** Working tree is still the single untracked
`sm_03_nose_reports_occluded.py`. **84 PASS / 187 registered (44.9%)** —
unchanged for seven days. The builder's last iteration ended **2026-08-25
12:23:33**; every slot since — **42 consecutive hourly slots** — logged
`PACING: … skipping`. Meters read live during this audit: `week:all models`
**63%** (the gate) at **44%** of the week, line **54%**; `week:Fable` **100%**;
both reset Mon 2026-08-31 05:00 UTC.

**Clean results, each re-run by me rather than relayed:**

- **§1 ledger integrity — clean, and one probe extended.** `run verify`: **83
  PASS re-judged from the record alone, 81 controls probed**, 0 verdicts that no
  longer re-derive, 0 gates that ignore their control, 0 controls declared but
  never run, 0 gates unreplayable, 0 entries unauditable, 0 controls run but
  undeclared. Two PASSes carry no control (`T0.01`, `T0.10`), both long-declared
  existence claims; `T0.18` self-excludes correctly. **New this audit:** I joined
  all 84 PASS rows to git and to disk. All **44 distinct PASS commits still
  resolve** (`git cat-file -e`); all 84 PASS specs have an implementation in
  `experiments/tests/`; none of those files is untracked. *A note against my own
  method:* my first pass at this looked for an `impl_path` field, which the
  ledger does not have — that check was vacuous and I re-ran it by mapping
  `BY_ID["…"]`/`SPEC_ID` references. The five apparent misses (`T0.18`–`T0.22`)
  are the `SPEC_ID = "…"` idiom, not missing files.
- **§2 thresholds and controls — clean.** `git log be0ecc2..HEAD` is empty.
  I re-ran the seven-day scan anyway across `registry.py`,
  `registry_expansion.py` and `experiments/tests/`. One hit touches `seeds`:
  `ed2d969` reflowing the LG registration — `seeds=3` survives verbatim on every
  touched spec. Every non-3 seed count in the registry is a deliberate
  `seeds=1` fixture or `T?.??` `seeds=5`, all pre-existing. No threshold moved
  in the loosening direction, no control deleted or weakened, no `_check` gained
  an `or`, no assertion removed.
- **§5 compute accounting — clean, and that is the problem.** `overruns: []`.
  W34 charged **0.3111 h of 30**, one job (`T2.15`, FAIL, harvested at
  `f5d8f1c`), one real ledger row. The books are honest; there is just almost
  nothing in them.
- **§7 bakeoff hygiene — no findings.** No bakeoff has run since the 29th audit;
  `ledger.json` has not been written since 2026-08-25 10:14. No decision made
  without a learning gate, no VOID treated as a verdict, no winner inside noise.
- **The three constitutional gates all exit 0**, all run by me: **coverage** — 0
  commitments with no declared spec, 0 CLAIM-DEAD, 4 known-dangling GOAL.md
  citations (shrink-only baseline); **decisions** — ratchet ok, 1/10 undeclared;
  **champions** — ratchet ok, 6/8 seats with a phantom arena, 12 violations, all
  carried unchanged.

---

## RANK 1 — the pace gate is being decided by argument, in an audit series, when both its arms are already runnable

This is first because it is the only finding that explains all the others, and
because it is this project's own founding disease pointed at the project's own
machinery.

### What shipped

`scripts/lib_usage.sh:34–99`, commit `e03693d`, **2026-08-24 10:35 UTC — three
days old**. It adds `pace_gate`, a time-based allowance
`allow = PACE_FLOOR + ceil((90 − PACE_FLOOR) · elapsed / 100)` with
`PACE_FLOOR=25`, checked *after* `usage_gate` has already said yes. Above the
line, the iteration is skipped.

The commit did the right thing in one respect and it deserves saying: it was
justified by a **table of real measurements**, written into the source, with its
success criterion stated in plain words (`lib_usage.sh:39–45, 59–60`):

```
#     week   loop went dark        dark for   Kaggle GPU-h expired unspent
#     W32    Fri 08-14 15:07          ~4.5 d   8.82 of 30
#     W33    Fri 08-21 12:07          ~2.7 d  22.11 of 30
...
#   PACE_FLOOR=25 buys the week's opening burst; by Friday the line is ~62%,
#   Sunday ~81%, and the loop is still awake when the GPU quota expires.
```

### What W34 measured against it

| | W32 | W33 | **W34 (this week, with the gate)** |
|---|---|---|---|
| loop went dark | Fri 08-14 15:07 | Fri 08-21 12:07 | **Tue 08-25 13:07** |
| dark so far | ~4.5 d | ~2.7 d | **42 h and continuing** |
| Kaggle GPU-h unspent | 8.82 / 30 | 22.11 / 30 | **29.69 / 30**, expire Sun 08-30 |

W34 slot-by-slot, from `/data/jack-logs/ladder.log`: **23 iterations started,
51 pace-skips** — 69% of the week's builder slots suppressed, the last 42
consecutively. And the discriminating fact:

> **Every one of those 51 skips was issued by `pace_gate`, not `usage_gate`.**
> The owner's real 90% ceiling was never approached — the meter has read ≤63%
> throughout, **27 points of unused headroom.** `grep STOPPED` over W34 returns
> nothing after 08-24 04:07 (the previous week's overrun).

The blackout is the new organ's doing, not the ceiling's.

### Why I am NOT claiming the organ made things worse

I can show the *stated criterion* is failing. I cannot show the counterfactual,
and I want to be exact about why, because the tempting version of this finding
is wrong. If the builder had run all 51 skipped slots at the Review's measured
~95.6K output tokens per session, that is ~4.8M output tokens it did not spend —
plausibly enough to hit the 90% stop and produce a *different* blackout. The
honest statement is:

> **Nobody knows whether the pace gate helped, and three days in, nobody can
> find out, because it shipped without the measurement that would settle it.**

### And that is the finding

Look at what the audit series has actually been doing:

| audit | its reading of the same blackout |
|---|---|
| 33rd (08-26 06:47) | the auditing organ is spending the meter it reports scarce |
| 34th (08-26 12:46) | the builder's own model is at 99% and no gate reads it |
| 35th (08-26 18:46) | the pace gate defers cheap work into an Opus window |
| Review (08-26 06:47) | the throttle is on the wrong side of a feedback loop |
| 37th (this one) | …and I could add a fifth reading |

Five organs, five arguments, one unmeasured mechanism. `SYSTEM.md` law 3: *"If
you find yourself reasoning about which approach is better, stop and write the
bakeoff."* Rule 3 was written for exactly this shape and it applies to the
system's own machinery, not only to Jack's — the loop runs on itself
(`SYSTEM.md:30–37`).

**The arms already exist.** `lib_usage.sh:76` is
`[ -n "${JACK_NO_PACE:-}" ] && return 0`. Arm A is the gate as shipped; arm B is
the same loop with `JACK_NO_PACE=1`; the metric is not a matter of taste — the
docstring already named it: slots run, ledger rows produced, and free GPU-hours
consumed before expiry. This is `MEANS`, not `GOAL`. It does not go to the owner
and it does not need another audit's opinion. It needs a spec.

**The generalisable defect, which is why this also goes to `LESSONS.md`:**
`e03693d` changed what the builder *does*, on the strength of measurement, and
carries no pre-registered success metric, no re-measurement date, and no row
anywhere. The first law — a capability is claimed only by a test that could have
failed — is enforced on Jack and **not on the organs that govern Jack.** The
only reason this was caught at all is that a human-facing log line happens to
print every hour.

### Projection, on the tool's own arithmetic

`allow = 25 + ceil(65 · elapsed / 100)`. At `pct = 63` the gate opens at
`elapsed ≥ 58%` → **Fri 2026-08-28 ~06:30 UTC**, and only if the meter stays
flat. Claude's week resets **Mon 08-31 05:00**, one day *after* Kaggle's Sunday
expiry. So on the current trajectory the builder gets roughly two waking days to
spend 29.69 GPU-hours, and its first act must be a dispatch.

---

## RANK 2 — this audit moved the meter, and the exchange rate is computable

The 33rd audit named this; here is the number.

During this audit `week:all models` crossed **62 → 63** (62% on every hourly
`PACING` line from 08-26 16:07 through 08-27 06:07; 63% on my live read at
07:05). Whether that point is mine or the concurrent 06:37 Review's, it is an
**auditor's** point. Its cost:

    d(elapsed)/d(pct) = 100/65 = 1.538 elapsed-points per point
    1 elapsed-point   = 168 h / 100 = 1.68 h
    ⇒ 1 point of `week:all models` = 2.58 h of additional builder darkness

**And the asymmetry that makes it structural:** `pace_gate` is called from
`scripts/ladder_loop.sh:179` **and nowhere else.** `overseer.sh:45`,
`review.sh:29` and `field_watch.sh:31` call `usage_gate` only. The only organ
that produces ledger rows is the only organ that yields; the three organs that
produce *documents* run unconditionally on the same shared meter.

### A correction to the Review of 2026-08-26, on evidence

The Review measured, from session transcripts, that during an 18-hour blackout
window the overseer audits were **98.9% of on-box spend**, that the meter rose
**+14 points against +4 of allowance**, and forecast: *"it does not run again
this week, the 90% stop lands ~08-28."*

Measured over the **following 17 hours**, across three overseer audits (08-26
18:46, 08-27 00:45, and this one) and one Review (08-27 06:37), the meter moved
**62 → 63 — one point.** The +14/day rate has not reproduced. I am not
disputing the Review's transcript measurement, which is more direct than my
integer percent; I am reporting that the two do not currently reconcile, and the
consequences differ:

- On the Review's rate, the **90% stop** fires ~08-28 and the blackout runs to
  the 08-31 reset.
- On the measured last 17 h, the 90% stop **never fires**, and the blackout ends
  when the pace line rises to meet a nearly-static meter — ~08-28 06:30.

The Review recommends two levers to the owner and its arithmetic says neither
suffices alone. **That arithmetic rests on the higher figure.** It should be
reconciled before either lever is pulled — and the reconciliation is a
measurement, not an argument, which is RANK 1 again.

`scripts/claude_usage.py` was checked for a stale cache — there is none, every
call shells out to `claude -p /usage` live. That hypothesis is dead; the flat
62% was real.

---

## RANK 3 — SM.03's implementation is untracked for the sixth audit, and I read it

Confirmed by a fresh join across all 187 registered specs: **`SM.03` is the only
spec in the repository whose implementation file git does not track.** 710
lines, 32 KB, mtime 2026-08-25 12:20 — **42 hours** orphaned. Sixth carry.

**What no prior audit did, and what I did instead of repeating the alarm: I read
it.** The design is sound and the pre-registration discipline is real, so this
ranks below the two above rather than above them:

- `run()` (line 648) **raises unless `_GATES_FROZEN`**, currently `False`
  pending the seed-90 pilot. The spec cannot register a row with provisional
  bars. Gate provenance is written *before* the pilot, and the docstring states
  the bars do not move on the pilot's account.
- The registry's declared control (shuffled field) is implemented, plus a
  matched-statistics placebo, the vision-only **alive-proof** that must PASS, a
  whiff-coverage VOID gate, a raw-bytes hash gate over discriminative inputs,
  and a five-ray per-layout occlusion assert. `SEEDS = [0, 1, 2]`.
- Chance is 1/8; `ODOUR_OCC_MIN = 0.25` is 2× chance at ~6σ, ceilings at ~4.5σ.
  These are not soft bars.

**The real finding underneath it:** `_GATES_FROZEN` is a **two-file convention,
not an organ.** `grep` finds it in `sm_02_*` and `sm_03_*` and nowhere else —
not in `protocol.py`, not in `run_spec`, not in any `T0.*` guard. It works
because those two authors chose to write `if not _GATES_FROZEN: raise` into
`run()`. Every other spec in the ladder can be dispatched with bars set after
seeing pilot data, and no instrument in this system would notice. That is not a
live defect today; it is a guard that protects exactly the two files that
volunteered for it.

---

## RANK 4 — seven days at 84 PASS, and the fourteen commitments with nothing behind them

**84/187 demonstrated (44.9%)**, down from 49.1% on 08-21. The *rate* falling is
honest — three ratchets went green by **declaring** claims, not passing them —
but §8 has to answer the plain question, so: **zero new demonstrated
capabilities in seven days.**

From `coverage.py`, **14 of 23 commitments have live claim specs and nothing
passing**:

    fast/slow (8 specs, 0 pass)   sleep (4, 0)        social/other agents (4, 0)
    hunger/thirst (5, 0)          death & retry (3, 0) smell (2, 0)
    voice (2, 0)                  balance (2, 0)      proprioception (2, 0)
    thermal kills (2, 0)          plasticity (2, 0)   shelter/building (1, 0)
    touch/contact (1, 0)          tool use (1, 0)

And the two GOAL.md sentences the audit brief singles out as most likely to be
quietly neglected: **curiosity** 12 specs / **1** pass; **one brain / unison**
21 specs / **1** pass. Six specs are RUNNABLE today (`SH.02`, `SM.03`, `VO.02`,
`BA.02`, `XL.01`, `LG.02`) and 29.69 free GPU-hours expire on Sunday.

**§3 drift, both directions.** The forward question is empty this window: the
builder worked on *nothing* in the last 24 hours, so there is no work to trace
to a GOAL.md sentence. The converse question is the whole report — the parts of
GOAL.md with no passing spec are listed above, and the reason they are not
moving is RANK 1.

---

## The single UNDECLARED decision — and why I arm nothing

`decisions.py --check` reports **1/10 undeclared**: *"Was physics-first retired
by argument instead of by bakeoff?"* The owner **already answered it on
2026-08-09**, in the entry body:

> **DECIDED 2026-08-09: (a) RUN IT.** Owner: *"schedule the run after T2.01."*

`decisions.py:99` matches `_SETTLED` against **headers only**, and this ruling
lives in the body under a header still reading `(OPEN, owner)`. The tool is
counting a settled question.

My standing instruction is to arm at least one `UNDECLARED` per audit. **I arm
nothing, deliberately**: attaching a `default` and a `decide_by` to a question
the owner has already ruled on would be inventing a fork that does not exist,
and the ratchet may shrink but may never grow. This is the same honest result
the 36th audit reached; the repair is a document edit carried below as **B3**,
and it must **not** be done by widening the regex — header line 1454 reads
`## D1 — DO NOT ANSWER …`, so widening `_SETTLED` to match `ANSWER` would
silently close **D1, the 38-spec decision**, on a header written to say the
opposite.

All eight *real* open decisions carry `DECIDE:` blocks with defaults and
`decide_by: 2026-08-31` — four days out. `D1` costs **38 specs**, `D10` and `D4`
cost 8 each.

---

## FOR THE BUILDER

**B1 and B2 go in the FIRST admitted iteration, together, in one commit.** They
are one unit: committing an untracked file and closing the hole that made it
invisible.

- **B1 — commit `experiments/tests/sm_03_nose_reports_occluded.py` (seventh
  carry).** It is the only untracked implementation of a registered spec in the
  repo. Do not freeze its gates in the same commit; the pilot has not run.

- **B2 — close the untracked hole in `assert_ref_is_current`** (36th audit
  RANK 1, carried). `experiments/gpu.py:274` reads
  `git status --porcelain --untracked-files=no` while `protocol.py:368` asks the
  same question *with* untracked files. Make the GPU guard at least as strict as
  the dirty stamp.

- **B3 — register the pace-gate bakeoff. This is RANK 1 and it outranks the
  rest of this list.** A `MEANS` fork whose arms are already implemented is not
  an escalation; it is an experiment nobody has written (`SYSTEM.md` law 3).
  Concretely:
  - Register a spec (suggest `SY.01`, tier 0, `CPU_FAST` — it reads logs, it
    does not train). Arms: **A = pace gate as shipped**; **B = `JACK_NO_PACE=1`**.
    Both already exist at `lib_usage.sh:76`.
  - Pre-registered metrics, all three already named by the organ's own
    docstring: **(i)** builder slots run per week; **(ii)** ledger rows recorded
    per week; **(iii)** free GPU-hours consumed before the Sunday expiry.
  - Null: the pre-gate weeks W32/W33 from the docstring table, which are already
    measured and already written down.
  - `falsified_by`: the gate loses on (iii) — the metric it was built to
    improve. Record W34 as its first observation: **dark from Tue 08-25 13:07,
    51/74 slots skipped, 29.69/30 GPU-h unspent.**
  - Commit the spec **before** running it, per `SYSTEM.md`.
  - **Do not weaken or delete `pace_gate` on my say-so.** Rule 4 binds, and I
    have not shown the counterfactual — that is what the bakeoff is for.

- **B4 — reconcile the two spend measurements (RANK 2).** The Review's
  transcript figure (+14 pts/day, auditors 98.9% of burn) and the meter's own
  last 17 h (+1 pt across three audits and a Review) do not agree, and the
  owner's levers below rest on the difference. Print both, name the method for
  each, and say which one the pace projection uses.

- **B5 — teach `decisions.py` to see an owner ruling written in an entry BODY**
  (carried). Add a settled marker to the physics-first header using a token the
  tool already reads (`RESOLVED`). **Do not widen the `_SETTLED` regex** — see
  the D1 blast radius above.

- **B6 — extend the pace-skip rescue path to untracked spec implementations**
  (carried, fourth time). `harvest_bookkeeping` already commits ledger rows
  during a pace skip. A registered spec whose implementation is untracked is the
  same class of orphan and would have cleared B1 automatically 42 hours ago.

- **B7 — make an `rc=0` that certifies a corpse impossible** (carried, third
  time).

- **B8 — fix the 06:37 cron collision** (carried, third time). `37 */6` and
  `37 6` both fire at 06:37; `overseer.sh` and `review.sh` ran concurrently
  during this audit, on the shared meter. Verified live in `ps` at 06:37 today.

- **B9 — when the gate finally admits an iteration, log the model substitution
  as an event.** `week:Fable` is at **100%**. The loop launches with
  `JACK_LOOP_MODEL=fable`; whatever it actually runs on will not be Fable, and
  nothing currently records the swap.

**Priority when the gate opens (~Fri 08-28 06:30, ~2 days before the Kaggle
expiry): B1+B2 in one commit, then B3, then dispatch a GPU claim spec.** Six are
runnable. Fourteen commitments have nothing passing. Do not spend the first
admitted iteration writing another successor spec.

---

## FOR THE OWNER

Nothing here needs a ruling to unblock work — B3 is a `MEANS` question the
system must settle itself, and I have routed it there rather than to you. Two
things are reported because they are yours to know, and one asks for a decision
only if you want the week's GPU hours.

1. **The builder has been dark 42 hours and will not wake before ~Fri 08-28
   06:30 UTC.** 29.69 of 30 free Kaggle GPU-hours expire Sunday 08-30; Claude's
   week resets Monday 08-31 05:00, one day *after*. Nothing has hit your 90%
   stop — the meter has stayed at or below 63%. The blackout is entirely the
   3-day-old pace heuristic, which is now being measured rather than argued
   about (B3). **If you want this week's GPU hours spent, the lever is
   `.usage-resumed`** with a ceiling and an expiry, which is the only sanctioned
   way to lift pacing (`lib_usage.sh:64`). No agent may write that file on your
   behalf, and none has — the 08-26 Review declined the same lever for the same
   reason, and it was right to.

2. **The audit organs are, structurally, the builder's competitor for the shared
   meter.** `pace_gate` is called only by the builder; the overseer (6-hourly),
   the Review (daily) and the field watch (weekly) are ungated. At the measured
   exchange rate, **one point of `week:all models` costs the builder 2.58 hours
   of darkness** — and the meter crossed 62→63 during this audit. That is a
   cadence question (how often should the auditors run when the builder is
   asleep?) rather than a defect, and I am flagging it rather than deciding it,
   because reducing my own oversight cadence is not a call I should make
   unilaterally.

3. **Your four goal-class decisions default on 2026-08-31 — four days out**, and
   the builder may be dark for two of them. `D1` costs **38 specs**, `D10` and
   `D4` cost 8 each. All are armed with pre-registered defaults that pick only
   among already-permitted actions, so an unattended firing is safe and
   reversible. No action needed if you are content for the defaults to fire.
