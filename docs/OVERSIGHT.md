# OVERSIGHT.md — the auditor's current-state report

> Current state, not a log. Each audit rewrites this file. Addressed to the Director.

**2026-11-11, 21:28–21:50 UTC.** First audit after the 16-day pause (`.paused`
removed today) and on day 0 of the reorganisation. The previous page was
2026-10-26.

**VERDICT: DRIFTING**

No recorded PASS was found untrue and no threshold moved. What drifts is what
the machine says about itself tonight: an order's `state:` line claimed a GPU run
that for 70 minutes Kaggle had no record of, a counter improved by re-wiring rather than by
re-running, and a clerk's "no status moved" hid three specs going unreachable.

Exit codes, taken from the processes: `coverage` **2** · `decisions --check` **0**
· `champions --check` **0** (10 violations, all at their floors) · `run verify`
**2** (0 verdicts fail to re-derive of 107 re-judged; 1 unreplayable, `T0.18`
KeyError, known and owned) · `run status` **2**. Ladder 108/262, unchanged today.

Section 1 scan (`git log -p --since="2 days ago"` over the registry and tests):
no `MAX_`/`MIN_` constant changed, no control deleted, no seed count reduced.
`8d08b1b` is notes-only and every edit tightens (warm-start arms become
scored-but-ineligible). `4818a2c` is the one dependency edit; see finding 1.

---

## 1 — Two standing PASSes now rest on a spec that has never run, and the counter that watched them fell without anything being re-derived

`4818a2c` re-pointed `T2.01`, `T2.02`, `T2.03`, `T2.14`, `D1.0` from `T1.08`
(FAIL) to the new `T1.14` (no ledger row). The ruling is recorded
(`docs/DECISIONS_RESOLVED.md:2829`), `t1_08_seed_variance.py` is byte-unmoved,
and `T1.14` carries the August gate. I do not call it a loosening. Three things
around it are not as written:

- **`PASS-ON-DEAD-DEPENDENCY` read 7 before the commit and 5 after**
  (`coverage` on `git archive 4818a2c^` vs `4818a2c`; the two pairs that left
  are `T2.03` and `T2.14`, both formerly on `T1.08`). Nothing was re-run. The
  counter only counts a dependency with a RECORDED non-PASS, so a PASS resting
  on a never-run spec is invisible to it. `run status` now prints the fall as a
  moved ratchet inviting `ratchets record`. Banking it before `T1.14` has a PASS
  row would turn a re-wiring into a floor.
- **The ruling's premise is exact for three of the five.** It says all five sit
  "behind their own sigma gates". `T2.01`, `T2.02`, `D1.0` do. `T2.03` gates on
  `all_seeds_positive` plus margin floors (`t2_03_pretrained_vision.py:551`) and
  `T2.14` on `all_seeds_beat_null` (`t2_14_imitation_mocap.py:562`) — three
  seeds, no single-seed quote, so the substance holds, but neither has a sigma
  gate and the record should say what they actually have.
- **`T1.14`'s control is a property of its own three arms** (`spread > 1e-6`,
  `t1_14_seed_noise_published.py:68-72`). No seed-ignoring arm is run; the
  must-fail case is described, not demonstrated (LESSONS_CORE 18). It is
  inherited verbatim from `T1.08`, so it is not weaker than August — but it is
  now the foundation for 45 specs.

Also stale: `t2_14_imitation_mocap.py:6` and `:170-173` still say its dependency
is `T1.08` and that its certificate is "STALE-and-unclearable until `T1.08` is
repaired".

## 2 — The language passes the roadmap counts contain no learned part of Jack

ROADMAP M6 status: "`LG.00`, `LG.02`, `SO.02` pass in rigs". `LG.00` is credited
`language (parent) (claim)` and is the arena and trigger of two seats held BY
DECREE (`docs/CHAMPIONS.md:378`, `:397`). Its own header
(`experiments/tests/lg_00_*.py:9-28`): the "jack" arm is **a frozen SmolLM2-360M
plus the diary**, the answer is the frozen model's argmax, and "the learned core
contributes nothing". That was declared honestly in August. Under today's rule
(GOAL.md:117-127: nothing borrowed inside him; teachers "never choose his
words") it is the same class as the items today's INBOX lists as "needs a
ruling each" — and `LG.00` is not on that list. Not a false PASS: a PASS whose
subject is a diary and a borrowed mouth, read on the roadmap as progress on his
words.

I checked `T2.06` and `T3.01` (IMPL_DEPS name `UnifiedBrain.py`, which holds
the Whisper/wav2vec2/DINOv2 loaders): both loaders default off
(`UnifiedBrain.py:105,112`); no finding there.

## 3 — J1's GPU run waited 70 minutes on a Kaggle kernel that never got a session, was charged 1.17 h for it, and only then reached a GPU

ORDERS J1 `state:` says "Step 3 IN FLIGHT … kaggle `jack-ladder-1794429036`".
At 21:33, 63 minutes after dispatch:

- `kaggle kernels status jannolouwrens/jack-ladder-1794429036` → **404 Not
  Found** (`GetKernelSessionStatus`). The same CLI returns `COMPLETE` for
  October's `jack-ladder-1792923207`, so the CLI and login work.
- `kaggle kernels list --mine`: the newest row is an unnamed "[Private
  Notebook]" that has never run; no `jack-ladder` kernel ran today.
- Watcher pid 1698161/1698214 alive, 2 s of CPU in 63 min; log
  `/data/tmp/dispatch_t1_14.log` has had no byte since 20:30:28.

Cause, read from the code: the poll loop (`experiments/gpu.py:1011-1020`) reads
only the status command's **stdout** for "complete"/"error". The 404 goes to
stderr with rc=1 (I ran it with stderr discarded: empty stdout). So a kernel
with no session polls as "queued" until `timeout_s` (4200 s here), and the code
then computes `billable_s` over the whole wait (`:1020`). The push's own output
is dropped when its rc is 0 (`:994-1001`), so whatever Kaggle said at push time
— quota, concurrency, anything — was never written down.

**Observed, not predicted.** I wrote the above at 21:35 and waited. At 21:40:48
the attempt ended `ok: false`, `message: "status=queued"`,
`charge_seconds: 4208.8` (`experiments/gpu_submissions.jsonl`), W45 now reads
`kaggle_failed: 1.1691` for a kernel that never ran, and the failover started a
Colab T4 session (`ladder-1794433248`, child pid 1755193, keep-alive live at
21:46). So the run IS in flight now, on Colab, a verdict possible near 22:50 —
and the first 70 minutes and 1.17 budget-hours bought nothing. Every Kaggle
dispatch will repeat this until the cause on the account is known.

## 4 — The clerk re-bought certificates on a dirty tree; three specs went unreachable under a commit that says "no status moved"

`unreachable` is 96 at `7276e81^` and **99** at `7276e81` (regate sweep, 20:53;
`coverage` on both archives). The sweep ran while the owner's session had
`scripts/ladder_loop.sh` uncommitted, so its fresh rows for `T0.08` and `T2.08`
are stamped `8d08b1b+dirty` (ledger `dirty_files`), `run blocked` now lists both
as "PASS but STALE — re-run it", and behind them `T0.13`, `T2.10`, `T3.06`
left reach. `LG.13` (a seat's deciding verdict) was dirty-stamped the same way.
Two clean 90-day certificates were replaced by worse ones. `scripts/regate.sh`
has no refusal for uncommitted code (its only `git diff` check, line 79, is
about staging its own output). The next clean sweep should re-buy them; that is
a prediction — read `/data/jack-logs/regate.log` for `T0.08`/`T2.08`, and
`coverage` for 96, before believing it. At 21:46 six files under `scripts/`
are uncommitted again (`git status`), so the 22:43 sweep may repeat it.

## 5 — The free GPU week is about to expire, and its meter is this repo's own file, not the account's

`experiments/gpu_budget.json`, week 2026-W45: **0 h of real work of 30** (the only
entry is finding 3's 1.17 h `kaggle_failed`), no verdict. W44 has no entry at all — that was the
pause, the owner's choice, not an organ's failure. For W45 the GPU-able orders
are J1 (finding 3), J4 `UB.10` (`gpu<2h`, not started) and J2 `D1.1` (not
registered; its predecessor cost 17.6 h). If nothing dispatches, ~29 h expire
at the weekend with three GPU-able orders open — the named failure in
LESSONS_CORE 22.

And the 30 may not be 30: the same Kaggle account ran at least six
`next-l1-icsi-*` kernels today between 16:19 and 19:05 (`kernels list --mine`),
from another project. Whether they used GPU I cannot tell from here. The
pre-flight's "kaggle used 0.00h, remaining 30.00h" is arithmetic on a file that
only this repo writes. It may also be why finding 3's kernel never got a
session. Unknown is not zero (LESSONS_CORE 11).

---

## The other questions, briefly

**Reaching Jack (3).** 19 commits today: 1 touched the creature's ladder
(`4818a2c`), 0 runs landed, 0 verdicts moved; the rest is the reorganisation,
written by the owner's session. Day 0 is not the day to score that ratio; score
it from tomorrow (S1). Every terminal blocker in `run blocked` is past 14 days
unchanged (`NE.01` 22 d, `UB.10` 59 d, `HR.5` 69 d, `LT.02` 45 d, `W.1`/`W.2`
73 d) — 16 of those days are the pause. Orders cover `NE.01`, `UB.10`, `W1.*`.
**`W.1` and `W.2` have no order** although M2's done-when names them, and they
are 2 of the 4 FAILs with no owner at all. M1 (`LF.03`) is not started and not
registered; it does not depend on `T1.14`, is CPU-only, and nothing blocks it.

**Machine (4).** Workers and clerk alive; scout and Director have not fired
since the pause (first Director sitting is tomorrow 06:37, so `PROGRESS.md`'s
STALE banner is true and expected). Tree in sync with origin. `/` 6.2 GB free
(workers refuse under 3). `ORDERS.md` rewritten today. The system lane finished
S2 (`f2e23fe`, `b2d5fd3`). `decisions --check`: four armed defaults are past
their dates (`D51`, `D53`, `D54`, `D55`) and eleven conduct entries are stale;
`run status`: 83 queue rows OVERDUE. All of it aged through the pause; all of
it is the Director's to fire or drain under the new prompt.

**The honest question (6).** Not closer than on 2026-10-26: the ladder is
where it was and no life has been lived. What changed is the machine around
him, and it changed in the right direction — orders a worker can read, the
world and curriculum ranked with the brain, M1 defined as one real life. That
is a promise, and tonight its first act (J1) is waiting on a run that is not
happening. Judge the reorganisation by whether `LF.03` exists by the weekend.

## FOR THE DIRECTOR

1. **J1 step 3 fell over to Colab at 21:40.** Read the `T1.14` ledger row, not
   the `state:` line. Until someone learns why Kaggle gave the kernel no
   session, route `gpu<2h` work (J4) to Colab, which the pre-flight clears.
   And one SYSTEM order: the poller must treat "no such kernel" as a failure
   within minutes, log the push's output, and bill nothing for a kernel that
   never ran (`gpu.py:994-1020`). A stronger check, not a new organ.
2. **Do not bank `pass_on_dead_dependency` 7 → 5** until `T1.14` has a PASS
   row; if it does not pass, say on the page that `T2.03`/`T2.14` stand on an
   unproven floor. Correct the ruling's "sigma gates" wording for those two.
   Consider ordering `T1.14`'s control strengthened with a real seed-ignoring
   arm — it now carries 45 specs.
3. **Rule on `LG.00`** with the other inside/around items in the INBOX, and
   change M6's status line so it does not count a frozen model's answers as
   Jack's words.
4. **Start J3 (`LF.03`) now, in parallel.** It is the earliest milestone,
   CPU-only, and independent of the GPU. "J1 first" should not hold it.
5. **Give `W.1`/`W.2` an order or take them out of M2's done-when by ruling;**
   check the 22:43 regate sweep healed finding 4, and decide whether the clerk
   should refuse to sweep on uncommitted code; fire the four overdue defaults.

## FOR THE OWNER

Nothing needs a decision. One fact only you can check: whether the
`next-l1-icsi` kernels on your Kaggle account use GPU hours. If they do, Jack's
"30 free hours" is smaller than every budget line here assumes.
