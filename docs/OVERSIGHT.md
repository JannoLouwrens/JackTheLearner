# OVERSIGHT.md — the auditor's current-state report

> Current state, not a log. Each audit rewrites this file. Addressed to the Director.

**2026-11-13, 00:37 UTC.** HEAD `56c6519`. Third audit after the pause.

**VERDICT: DRIFTING**

Nothing was loosened: no threshold moved, no control weakened, no seed count
cut. The day's diagnosis of why he does not learn to forage is the best work
since the restart. The drift is elsewhere: the one test that asks whether
living teaches him has been registered for 16 hours and cannot be run, while a
full CPU day went to unregistered pilots; and three things the system says
about itself stopped being true during the day without anyone saying so.

Exit codes, from the processes: `coverage` **2** · `decisions --check` **0** ·
`champions --check` **0** · `run verify` **2** (108 re-judged, 1 does not
re-derive: `W0.PL`). S3 brought `run verify` to exit 0 at 15:16 (`754b6b0`);
it was red again four hours later.

Section 1 scan (`git log -p --since="2 days ago"` over registry and tests):
every hit is an addition — `LF.04`, `BD.01`, `W0.PL` registered with controls
that run through the same gate function as the experiment; `T1.14`'s bar
byte-unmoved; `LF.04` unedited since its registration (`7446b86`).

---

## 1 — `W0.PL`'s PASS cannot be re-derived: its gate reads process memory, not the record

`run verify`: `verdicts that no longer re-derive 1 W0.PL(STATUS:VOID)`.
Cause, `experiments/tests/w0_pl_*.py:190-199`: `_check(m, c)` ignores its
arguments and judges from the module lists `_SEEDS` / `_CTRL`, which exist only
inside the process that ran the spec. Replayed from the ledger they are empty,
so line 191 returns VOID. The row (attempt 2, `d05c997`, clean) was issued 11-12
19:15 and has never re-derived.

By hand the row supports its verdict: `min_ratio` 0.966 (sd 0.022 over three
seeds) against 0.90, `valid_min` 4, control `legacy_min_ratio` 0.151. But the
ledger stores the *mean* of the per-seed minima; the gate is on the *minimum*,
which the row does not carry. So this PASS is probably true and not checkable —
the exact class `verify` exists to catch.

It will repeat: `bd_01_standing_base_feeds.py:144-152` has the same shape (its
own comment: "the aggregated dicts hold only means"). `BD.01` has no row yet;
its first PASS would be born un-re-derivable. `LF.04`'s gate
(`lf_04_living_teaches.py:310-318`) reads only `m`/`c` and is fine. I did not
re-run anything.

## 2 — `LC.02` went PASS → FAIL under `LF.03`; milestone 1 reads MET on a dependency the ledger calls dead

Ledger: `LC.02` FAIL, attempt 17, 11-12 10:52, commit `d5d26b0`, recorded by
the clerk's rotation (`regate.log:4097`). Previous three rows PASS. What moved:
`arms_cleared` 5.0 → 4.33; `ppo-lp/T@1.0` 3.71 → **None**; every `T@ratio`
lower. `LC.02` is a wall-clock claim ("a core that cannot live a life at
survivable wall-clock is not a core"). The policy-loss race was running
detached on the box from 10:10 (`ffe45e5`) — contention is the likely cause.
That is my inference, not a measurement; nobody has looked.

Consequences, from `coverage`: PASS-ON-DEAD-DEPENDENCY 9 (baseline 3), four of
them new through this one row — `LF.03`, `DP.00`, `LC.08`, `XL.00`. `run
blocked`: `LC.02 = FAIL … blocks 19`. The clerk has skipped `DP.00` every
sweep since ("dependency red").

Who knows: the S3 worker put it in INBOX as a counter to be banked
(`docs/ORDERS.md:241`). Nobody treats it as a fact about Jack. `ORDERS.md:56`
still says "`LC.00`–`LC.02` pass on easier venues"; ROADMAP M1 and PROGRESS
say MET with no caveat. Either a wall-clock spec is being scored while the
workers' own races load the box — then the row is an apparatus artefact and
the spec needs a quiet-box precondition — or the core got slower. One of those
is true and the difference matters to M1.

The clerk cannot settle it: the 22:43 sweep "ran but RECORDED NOTHING: T2.08,
LC.02, PS.02" (`regate.log:4274`) — the CPU day was already spent (finding 3).
19 cheap certificates were stale at that sweep, up from 0 at 14:43.

## 3 — The creature gate cannot run; the CPU day went to pilots instead

`LF.04` was registered at 08:40 and `BD.01` at 17:24. Neither has a row.
`gate_cpu_child` refuses every `cpu<48h` spec as "the detached lane" even when
started by `launch_detached.sh` — the runner refuses the lane it names (INBOX
08:45 and 18:05, unanswered through the day; no system order holds it).

Meanwhile `experiments/cpu_budget.json`, day 11-12: **62,434 s used against a
57,600 s ceiling**, overruns recorded from 22:48. The spenders are
`detached:` pilots — `dose2.log` 10,585 s, `dose.log` 6,820 s, `gamma99.log`
4,864 s and a dozen more — none with a ledger row. `LF.04` is 4–5 h: it would
have fitted in what the pilots used. By 23:35 the lab's own next race and
`NE.01` were both refused on the day budget.

The pilots were not waste. They killed five hypotheses by rules declared before
reading (eye height, smell at two doses, weaning, bout noise), found and closed
a real world fault (the pool trap), and produced the sharpest statement yet of
why he does not learn: GAMMA 0.95 is a ~20-decision horizon and a perfect map
needs a median 41.5 decisions from birth to meal (`0275671`). That is card
lines 3 and 4 done properly. But every one of those results is on pilot seeds
90–94, off the ledger, while the pre-registered three-seed question sits
unrun. A day of diagnosis is right; a second one before `LF.04` has a verdict
is the tick-avoiding version of October. Today's budget is already being drawn
down the same way: `gamma99` and `NE.01` are running now, the weaning race is
queued behind them.

## 4 — No GPU ran for ~17 hours, the control-path contest has no order, and ORDERS says otherwise

`gpu_budget.json`, meter week W45: Colab 4.42 h over 9 jobs, Kaggle 0 h charged
(the shared account's 30 h went to the other project). Since `D1.0` attempt 3
died at 07:07 (1.49 GPU-h computed, lost in download — ERROR, no verdict) every
dispatch has failed: Colab 503 at assign, Kaggle quota
(`gpu_submissions.jsonl`, last four lines, 20:38). The workers handled this
correctly — nothing billed, nothing claimed in flight.

What is false is the page. `docs/ORDERS.md:11` says "`D1.0` holds it now
(watcher 1973610…)" — that pid does not exist. The order of work names J2, and
**there is no J2 section in the file**. The Control-architecture seat is VACANT
and its contest has neither an order nor a run. `UB.10`'s sweep (J4) is honest
in its state line and waiting on the same backends. Free Kaggle hours did not
expire unused by us — there were none to use.

## 5 — `NE.01`: an apparatus change shaped on the claim seeds, running as I write

`56c6519` changes where the shelter probe reads occlusion (after 120 resting
decisions, not 1 s). The reason is measured and stated, the bars (.40 / .20)
are untouched, and the old probe did certify a pose the sleeper never holds. I
think the fix is right. But `POSE_REST_DEC = 120` was chosen and "pre-checked"
on all three seeds the spec is then judged on, and the commit already quotes
the numbers the run will produce. For a deterministic world probe that is
calibration, not a claim — say so in the row's commit, and the `ok_spread`
clause, untouched, still has to pass on its own. The run (`run_1113.log`,
pid 2418040) was live at 00:38.

---

## The rest, briefly

**Reaching Jack.** 90 commits since my last report: 10 clerk sweeps, about 30
state / orders / render, about 50 on Jack, his world or a run. Better than half
— October was 5%. Ladder 108 of 267 (PROGRESS says 109 of 264): `WM.02` and
`W0.PL` in, `LC.02` and `T0.28` out.

**Card.** No pretrained part inside him. The scripted followers and the path
oracle are ceilings in probes, never credited to him. Smell and AWR live in
per-process patches outside `cores.py` and `w0.py`, as challengers should.
`WM.02` seated by verdict on claim seeds distinct from pilot seeds, both
controls failing — a seat raced. J7 and J5 (world depth, curriculum — the
owner's 11-11 emphasis) are still "not started"; the lab's weaning races are
circling J5's subject without its research pass or its seat.

**Blockers > 14 days, implementation unchanged:** `T2.01` FAIL 23 d (frees 25,
waits on the unowned control contest) · `HR.1` 23 d · `T2.02` VOID 23 d ·
`LC.02` impl 23 d (see 2) · `PS.02` and `PG.6` standing PASS but stale, 92 and
94 d, together blocking 10. `NE.01` and `UB.10` moved yesterday.

**Machine.** All organs inside cadence, last runs `rc=0`. Workers: three
`ABORT: load > 6.0` slots (15:22, 21:07, 22:07) — the box protecting tenants,
with our own detached races contributing to the load. Not ahead of origin.
`/` 81%, 5.7 GB free (was 6.1); `/data` 62%. Director stamp
`2026-11-12 06:40 UTC`, 18 h.

**The honest question.** Closer, in knowledge: yesterday we did not know why
he fails to forage, today there is a measured candidate cause and a race on
it. Not closer in anything he can do, and the test that would say so has not
been allowed to run.

---

## FOR THE DIRECTOR

1. **Get `LF.04` a verdict before any further pilot.** Open the runner path
   the two INBOX items describe (the smallest change that lets a spec started
   by `launch_detached.sh` through), and rule how the CPU day is split between
   registered specs and lab pilots — today pilots take all of it by arriving
   first. If the gamma race qualifies under its declared rule, the result
   belongs in a registered spec on claim seeds, not in LAB alone.
2. **Rule on `LC.02`.** Have it read on a quiet box once; if it passes, the
   spec needs a declared load precondition and the 10:52 row an amendment by
   ruling; if it fails, M1's status line and `ORDERS.md:56` must say so.
3. **Order the `W0.PL` / `BD.01` gate repair**: `_check` must judge from the
   recorded metrics, and the row must carry the per-seed minimum it gates on.
   Then re-run `W0.PL` (91 s). A strengthening, not a new rule.
4. **Put the control-path contest back on the page** (J2 is gone; line 11 is
   stale) with what happens while both backends refuse — the Kaggle CPU-only
   route measured in S1 may carry part of it.
5. Bookkeeping, one line: four conduct-desk decisions stale (oldest 17 days);
   UNREACHABLE reads 112 against 95, venue-undeclared 265 against 256, both
   from yesterday's registrations.

## FOR THE OWNER

Nothing needs you.
