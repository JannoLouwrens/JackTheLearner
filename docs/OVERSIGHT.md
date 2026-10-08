# OVERSIGHT.md — the overseer's current-state report

> Current state, not a log. Each audit rewrites this file.

**2026-10-08 18:37 UTC — 146th audit.**

## VERDICT: INTEGRITY RISK

**A Tier-0 certificate's implementation became permanently unreconstructable
today, and the cause was an overseer leaving its own legal work uncommitted.**
The 145th audit ran at 12:46, wrote a correct 123-line evidence addendum to
`docs/DECISIONS_NEEDED.md` under `D40`, and never committed it. `lib_seal.sh`
then stamped `docs/OVERSIGHT.md` STALE with the words *"exited rc=1 **without
writing a word**"* — which is false, and the file it wrote was still sitting
dirty in the tree. Two regate sweeps then ran across that dirty tree at 14:44
and 16:45. `T0.28` carries `docs/DECISIONS_NEEDED.md` in its staleness bill, so
its re-bought PASS at `95c0a7b` recorded: *"the code that ran was never
committed, reconstructs from no commit and was not preserved … **it cannot be
recovered by anyone**."* `T0.28` is *"The escalation tool can be shown catching
a deadlock and a claim-death"* — the certificate that guarantees the organ
watching the owner's desk works. It was dirtied by an uncommitted edit to the
owner's desk. **This audit's first act is to commit that addendum**, which is
what makes the next regate able to buy a clean stamp.

**The ledger itself is clean and that is the valuable half.** 107 standing PASS
rows, every one resolving in `BY_ID`; no `MEANS-ESCALATED`, no `UNDECLARED`, no
`OVERDUE` default; `decisions --check` exits 0 with its whole ratchet at floor
and 31 of 31 firings transcribed. **Section 2 is clean: in seven days not one
numeric bar moved in the loosening direction.** `MAX_HELDOUT_CV_PCT` 7.0 is
byte-unmoved across both `T1.08` dispatches, including the one that came back
FAIL at 14.666.

**Instruments, every one re-run immediately BEFORE this file was committed**
(the hourly builder slot fires at `:07`, so a reading quoted from the top of a
sitting is the default failure here): `coverage` **2**, `decisions --check`
**0**, `champions --check` **0**, `run status` **2**, `run review-queue` **2**.
Ratchet floors: **3 ABOVE** (`dark_slots` 6 vs 0, `pass_on_dead_dependency` 6
vs 3, `unreachable` 96 vs 95), 0 BELOW, 0 UNVERIFIED. `ratchets vs committed
readings (HEAD): 9 MOVED`.

> Correction, stated rather than quietly fixed: I first read `review-queue` and
> `champions` as exiting 0, because I had piped each into `head` and taken `$?`
> from the pipe. Re-run unpiped, `champions` is genuinely 0 and
> **`review-queue` is 2** — see RANK 6. The readings above are the unpiped ones.

---

## RANK 1 — an uncommitted overseer edit destroyed a Tier-0 certificate's
## reconstructability, and the organ built to notice recorded the opposite

Timeline, every item from `git log` and `/data/jack-logs/ladder.log`:

```
12:32:21  d668a65  builder slot ends rc=0 (T1.08 Step 2b harvested + redispatched)
12:45:03  4e4d973  regate sweep
12:46:25  f87fbed  overseer: OVERSIGHT.md stamped STALE — "exited rc=1 without
                   writing a word"   <-- FALSE
~12:4x             145th audit writes a 123-line D40 addendum to
                   docs/DECISIONS_NEEDED.md and does NOT commit it
14:44:37  95c0a7b  regate sweep  <-- ran with that file dirty
16:45:14  14b76e7  regate sweep  <-- ran with that file dirty
18:37              this audit finds the addendum still uncommitted, 6 h old
```

`run status` on the resulting row, verbatim:

> `T0.28  recorded PASS; ran from a modified tree at 95c0a7b; the code that ran
> was never committed, reconstructs from no commit and was not preserved (1
> uncommitted code file(s) at run time, INCLUDING this spec's own
> docs/DECISIONS_NEEDED.md) — it cannot be recovered by anyone. Re-run it from a
> clean tree.`

Three separate things are wrong here and they compound:

**(a) `lib_seal.sh` cannot see work done outside the file it guards.** The seal
asks "was `OVERSIGHT.md` rewritten?" and reports "produced nothing" when the
answer is no. The overseer is explicitly permitted to write `OVERSIGHT.md`,
`DECISIONS_NEEDED.md` and `LESSONS.md`. A run that writes two of the three and
dies gets recorded as having written none. That is not a stale page; it is a
**false statement about another organ's output**, generated mechanically, and it
will recur on every truncated run. My own standing lesson is to audit the
predecessor's stated reason for not acting — here the wrong reason is written by
a script, so it would have gone on being wrong indefinitely.

**(b) `DECISIONS_NEEDED.md` is in a certificate's staleness bill, so overseer
prose is load-bearing code for `T0.28`.** The per-spec doc-staleness repair made
`OVERSIGHT.md` and `LESSONS.md` `PROSE_DOCS`-exempt so the overseer could write
them mid-run. `DECISIONS_NEEDED.md` was not exempted — correctly, because
`T0.28` really does read it. The consequence nobody wrote down: **the one page
the overseer is required to append to is the one page it cannot leave dirty**,
and nothing warns it.

**(c) The damage is to the escalation tool's own certificate.** Of all 107 PASS
rows, the one made unreconstructable is `T0.28`, which exists to prove
`decisions.py` can be *shown* catching a deadlock. The instrument this audit
leans on hardest for section 6 now has a PASS whose implementation nobody can
reproduce.

**What I did about it.** I verified the inherited addendum against source before
adopting it (`scripts/lib_usage.sh:128` and `:130`; 19 log lines; `.usage-resumed`
absent) and found its substance correct and one of its numbers stale — see RANK 4.
It is committed with this report, corrected and attributed to the 145th. That
clears the dirt; `T0.28`'s row still needs a re-run from a clean tree, which is
the builder's.

---

## RANK 2 — 25 consecutive hourly slots produced no builder work, the PASS delta
## is zero, and the counter that should hold the 19-slot hard stop reads 6

The 90 % stop **fired for the first time in this project's history**, and
released itself without the owner:

```
2026-10-07T12:07:04  STOPPED at  90% weekly usage — all agents paused until the owner resumes
2026-10-07T13:07:04  STOPPED at  99% ...
2026-10-07T14:07:03  STOPPED at 100% ...   (100% for sixteen further slots)
2026-10-08T06:07:03  STOPPED at 100% ...   <- the 19th and last
2026-10-08T07:07:09  iteration start — 107/257 demonstrated, model fable, load 2.07
2026-10-08T07:22:52  iteration end rc=0 — 107 -> 107 demonstrated
```

- **19 consecutive stopped slots**, 10-07T12:07 → 10-08T06:07. `.usage-resumed`
  is absent from disk and was absent throughout; the log holds **zero** `RESUMED
  BY OWNER` lines in the window and its most recent is **2026-08-12T11:07**,
  eight weeks before this firing. The release was the meter reset (100 % → 17 %).
- Then **four more slots lost** — 08:07, 09:07, 10:07, 11:07, each `rc=1` after
  ~20 s — to a per-session limit, a third mechanism again.
- Then **six consecutive pace-skips**, 13:07 → 18:07, at `week:all models` 42–48 %
  against a line of 28–29 %.

**Last 24 h: 25 slots, 2 iterations actually ran, demonstrated 107 → 107.** Both
runs that ran were bookkeeping and regate, not a claim.

`dark_slots` = **6**, ABOVE its declared floor of 0, `MOVED +6 since 2026-09-30`.
The number is a *trailing streak*, so it measures the six pace-skips since 13:07
and has **no memory of the 19-slot hard stop at all** — it read `0 dark slots` in
the 13:07 line, one slot after the stop ended, because the 12:32 `rc=0` zeroed it.
A 25-slot blackout inside 24 hours reaches no exit code and no page. The routed
row `dark-slot-counter-is-blinded-by-the-loops-own-notice-lines` (DISPOSITIONED,
`DUE 2026-10-10`) is about notice lines; **this is a second, independent blindness
in the same counter** and it is the one that cost 19 slots.

---

## RANK 3 — the stop's own log line is false about the code that prints it
## (inherited from the 145th audit, re-verified at source by me)

`scripts/lib_usage.sh`, `usage_gate()`:

```sh
128:  if [ "$pct" -lt 90 ]; then return 0; fi          # <-- the whole latch
130:  local f="$REPO/.usage-resumed"
131:  if [ -f "$f" ]; then ... fi                      # <-- only reached ABOVE 90
147:  "$say_fn" "STOPPED at ${pct}% weekly usage — all agents paused until the owner resumes"
```

`.usage-resumed` is read **only on the branch where `pct >= 90`**. It is a
*ceiling override for continuing to operate above 90 %*, not a resume latch.
Below 90 the function returns 0 unconditionally. So *"paused until the owner
resumes"* is false: the meter reset resumes it, which is exactly what happened at
07:07 today. The realised cost of this mechanism's first firing is 19 slots, its
maximum is one usage week, and **the owner is not on the critical path of
either** — which is the opposite of what this entry, `D30`'s standing report and
`docs/PROGRESS.md:431` all currently tell the owner. Both repairs are
message-only and routed to the builder below; neither can change what
`usage_gate` returns, and **nothing here is a reason to touch the 90 constant.**

---

## RANK 4 — I reconciled a number two desks disagree on, and the 144th audit's
## was the wrong one

`D40`'s 144th-audit addendum quotes the Kaggle draw as **W37 5.249 h**;
`docs/PROGRESS.md:331` quotes **W37 1.379 h**; the 145th audit silently switched
to 1.379 without saying it was correcting anything. Re-derived from
`experiments/gpu_budget.json`:

| basis | W37 |
|---|---|
| all jobs, all backends, ok and failed | 5.2493 h |
| all backends, ok only | 4.412 h |
| **kaggle only, ok only** — what the 30 h free quota meters | **1.379 h** |

**`PROGRESS.md` is right and the 144th audit's own addendum was wrong for the
claim it was supporting** (it folds in two Colab jobs and one failed Kaggle job).
Stated plainly because the error was in this organ's output, not the Review's,
and because a silent correction by the next audit is how a wrong number survives.

On the same corrected basis, re-derived live for `D40`'s `decide_by 2026-10-10`:
**`2026-W40` has 1.971 h drawn of 30 free Kaggle GPU-hours; 28.03 h expire
Saturday 2026-10-10.** The 145th's addendum said 0.9252 h / ~29.07 h; that was
true at 12:4x and job `…1791461828` (1.0458 h) landed afterwards. The addendum is
committed with that figure corrected in place.

---

## RANK 5 — the pacing line's denominator ran backwards inside nine hours

From the `PACING:` lines, same gate, same day:

```
13:07  'week:all models' 42% at 15% of the week (line 35%)
14:07  'week:all models' 42% at 16% of the week (line 36%)
15:07  'week:all models' 32% at  4% of the week (line 28%)   <-- elapsed fell 16 -> 4
16:07  'week:all models' 37% at  4% of the week (line 28%)
18:07  'week:all models' 48% at  5% of the week (line 29%)
```

**Two meter resets in nine hours** (100 % → 17 % at ~07:00, then elapsed 16 % → 4 %
at ~15:00). `pace_gate`'s line is a function of week-elapsed, so a non-monotone
denominator moved the line 36 % → 28 % — *tighter* — with no change in policy and
nothing recording that it happened. Reporting-only and in the conservative
direction, so it is not an integrity finding; it is a finding that the window
`claude_usage.py` reports is not the week the gate's arithmetic assumes it is.

Beside it, the shared-pool attribution: *"of this week's 27 shared point(s):
builder 8 (29 %), desks 4 (14 %), **NOT THIS PROJECT 15 (55 %)**"*. The majority of
the pool this project is paced against is another project's spend — already
routed and ACTED as `builder-blackout-is-paced-by-another-projects-usage`, noted
here only because it is what the six pace-skips above were actually paying for.

---

## RANK 6 — seven queue rows are exempt from ageing behind a blocker that was
## REFUSED, and two of them are the claim-dead commitments

`run review-queue` exits **2** on **7 violations, all one class:
`HOLD-ON-A-RESOLVED-BLOCKER`**. Every one of the seven is held behind
`w1-world-edit-window`, and that row is **DECLINED** — in the tool's own words,
*"the window was abandoned, not opened — the blocker was refused, and this hold
now waits on nothing that will ever move."*

```
ne01-occlusion-knife-edge                                  HELD  45 d
water-apply-phantom-force                                  HELD  45 d
sh02-null-saturation                                       DISPOSITIONED 39 d
w1-cold-is-not-lethal-at-night                             OPEN  39 d
w2-needs-have-no-single-k                                  OPEN  39 d
hr5-fixture-refuted                                        HELD  35 d
ba03-vestibular-channel-is-never-load-bearing-under-one-kick  OPEN 18 d
```

`HELD` exempts a row from ageing, which is only legal while it pays with a live
`BLOCKED-BY:`. These seven stopped paying when the window was declined, so the
bundling rule has become the place rows go to die — exactly what the class was
built to catch. **Why it compounds today's other findings:**
`w1-cold-is-not-lethal-at-night`, `w2-needs-have-no-single-k` and
`sh02-null-saturation` are the thermal, needs and shelter rows — i.e. three of
the four parked behind the **claim-dead** commitments in §3. The commitments have
no live falsifier *and* their repair rows are frozen behind a refusal.

Already routed as `seven-rows-are-held-behind-a-refused-window-and-a-moot-decision`
(OPEN, `DUE 2026-10-11`), so this is the Review's work and not an unowned hole —
reported because it is a standing red of seven that the rest of this page's
arithmetic depends on. Throughput, for honesty about whether the desk can absorb
it: **49 OPEN, 3 HELD, 87 live rows, 2.86 disposed/cycle against 2.57
arriving — a measured drain of 304 cycles.** A slow week is legal and this is a
metric, not a violation; 304 cycles is not.

---

## 1. Integrity of the ledger

**107 PASS, 35 FAIL, 17 VOID, 1 BLOCKED, 0 NOT_RUN/ERROR of 257.** Every PASS
resolves in `BY_ID`. No PASS was found whose declared control went unrun. What
the record does carry, all of it from `run status` rather than my reading:

- **3 DIRTY STAMPS** — `T6.03` (BLOCKED, predates `impl_sha`), **`T0.28` (PASS,
  new today — RANK 1)**, `PL.02` (VOID, dirt outside its declared bill).
- **13 STALE CLAIMS** where a path inside `impl_sha` moved after the run, plus
  `T2.02` stale by content pre-`impl_sha`. Six of the thirteen moved only a
  *sibling* file, which is the hash's name and not a claim about the spec.
- **6 UNBACKED CERTIFICATES** — standing PASS rows whose dependency is not
  satisfied today, so `run_spec` would refuse to re-derive them: `LF.02` (root
  `T2.10` FAIL), `T0.18`, `T0.19` (`T0.13`), `T1.12` (`T1.11`), `T2.03`, `T2.14`
  (`T1.08`). Legal, reporting-only, and the honest reading of "107 demonstrated".
- **`pass_on_dead_dependency` = 6 vs floor 3**, `MOVED +1 since 2026-09-26`. Six
  standing PASS certificates rest on a RECORDED non-PASS dependency — the claim
  the board renders could not be re-derived today. Nobody raised the floor and
  nobody may: the repair is a re-run, of the dependency or of the dependent.
- **`SETTLE EVENTS`, last 7 days: 76 runs = 6 first-ever verdicts, 67 re-buys, 3
  status changes — and 68 of 76 (89 %) instrument-coupled.** This project's own
  tool edits are what staled the certificates it then spent a week re-buying.
  Four first-ever PASSes in seven days, of which three (`T0.31`, `T0.21`,
  `T0.28`) are audit organs and one (`W1.01` *"Passivity dies"*) is about Jack.

## 2. Thresholds and controls, over seven days — NO FINDINGS, and that is real

`git log -p --since="7 days ago" -- experiments/registry.py
experiments/registry_expansion.py experiments/tests/`: **not one numeric bar moved
in the loosening direction.** Specifically:

- `MAX_HELDOUT_CV_PCT` **7.0 byte-unmoved** across both `T1.08` Step 2b
  dispatches, including attempt 4's honest FAIL at `heldout_cv_pct` 14.666 — the
  exact place a loosening would have paid, and the commit says so in advance
  (*"`MAX_HELDOUT_CV_PCT` 7.0 is byte-unmoved; a STILL-FAIL is an …"*).
- `SEEDS = [0, 1, 2]` appears as an **addition** (a newly implemented spec), not a
  reduction anywhere.
- The one substantive change to a dispatch: `prefer="colab"` → `prefer="kaggle"`,
  `est_hours` 0.3 → 0.5, `timeout_s` 3000 → 4200. A venue and budget change, in
  the generous direction, touching no bar.
- No control deleted, no `_check` conjunct removed, no `or` added to a decision.

## 3. Drift from the goal

**What the builder did in the last 24 h:** two iterations. 07:07 → `rc=0` (15 min,
the T1.08 Step 2b harvest-and-redispatch); 12:07 → `rc=0` (25 min, bookkeeping +
the pace-skip detached-ledger commit). Everything else was a stop, an `rc=1` or a
pace-skip.

**Does it trace to GOAL.md?** Yes, and it is the right unit. `T1.08` *"Seed
variance measured"* serves *"every capability claimed only by an experiment that
could have failed … at ≥3 seeds where the claim is about learning"*, and it is
this project's largest single blocker — `T2.03` and `T2.14`'s certificates and 19
more specs sit behind it. **No drift found in the last day.**

**The converse, which is the harder question, re-derived independently rather than
quoted:** `git log --since="7 days ago"` = **165 commits; ZERO touched
`UnifiedBrain.py`, `TrainingPipeline.py`, `playground.py` or `EpisodicMemory.py`.**
The 144th audit measured this over its own window at 203 commits and the Review
measured 249 on 10-04. Three independent windows, same answer: **the creature has
not been touched in over a week.** Every commit was instrument, desk page, ledger
row or gate.

**And which commitments have no passing spec at all** (`coverage`, exit 2):

- **3 CLAIM-DEAD** — `smell`, `thermal (kills)`, `shelter/building`. Every claim
  spec parked or foreclosed; the repair is a REGISTRATION, and it is `D42` on the
  owner's desk, armed, `decide_by 2026-10-18`.
- **14 commitments with live claim specs but nothing passing**, including
  `touch/contact`, `tool use`, `told world`, `proprioception`, `fast/slow`,
  `sleep`, and the four jungle primitives `heavy`/`far`/`tiring`/`worth-it`.
- **`NO-LIVE-PATH` = 6** distinct commitments/seats with no live path at all.
- **`CITED-BUT-UNRUNNABLE` = 7** GOAL.md citations resolving to a parked,
  foreclosed or welded spec — **4 NEW** (`GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`).
  GOAL.md's §187 cites all four in the present tense and all four are welded
  behind `LC.07`. An id that resolves to a corpse is worse than one that resolves
  to nothing.
- **Queue depth: 7 dispatchable today, of which 7 VOID — 0 FRESH dispatches.**
  Three cost classes are NEWLY EMPTY with no path in (`cpu<1min`, `gpu<20min`,
  `gpu<8h`). `unreachable` = **96 of 257 (37 %)**, ABOVE floor 95.

Curiosity: `LT.03` *"THE LADDER TEST: curiosity alone climbs the ladder"* is VOID,
`LT.02` is FAIL, `T3.06` is VOID-FORECLOSED, `LT.04`–`LT.09` unimplemented. **The
ladder-and-apple standard that names this project has no live falsifier today.**

## 4. Is the builder alive and productive?

Alive, barely, and not productive. See RANK 2: 25 slots, 2 runs, PASS delta 0.
Not a paused loop nobody resumed and not credit exhaustion in the ordinary sense
— three *different* refusal mechanisms (`usage_gate` 90 %, a per-session limit,
`pace_gate`) each took a share of the same 24 hours, and only the middle one
leaves an `rc`. No repeated identical failure, no abort on load (loads 0.07–2.07
against a live ceiling).

## 5. Compute honesty

- **`2026-W40`: 1.971 h drawn of 30 free Kaggle hours — 28.03 h expire Saturday
  2026-10-10.** Fourth consecutive near-empty week (W37 1.379, W38 0.918,
  W39 1.072, W40 1.971 — **~114 h unbought in four weeks**, on the corrected
  kaggle-ok basis of RANK 4).
- **Unlike the previous three, W40's hours have a buyer that is buying.** All
  three W40 jobs are `ok:true` and all three are `T1.08`: `…1791217029` 0.2708 h,
  `…1791444041` 0.6544 h, `…1791461828` 1.0458 h. Attempt 4 came back an honest
  FAIL (`heldout_cv_pct` 14.666 vs 7.0) and attempt 5 landed. **Nothing was
  manufactured to spend the quota and nothing should be** — but the 19 slots the
  hard stop took were taken from exactly this work.
- **`gpu_hours_no_verdict` TOTAL 51.47 h**, up 1.98 h. The standing waste is
  unchanged and it is one row: **`D1.0` — 33.78 h across 2 attempts, 0
  verdicts.** `UNATTRIBUTED` 6.32 h across 21 jobs, at its floor of 21.
  `T1.08` now reads 2.33 h / 4 attempts / 3 verdicts — one attempt still owes a
  verdict, which is attempt 5, harvested at 12:32 and committed at 14:07.

## 6. Stuck decisions — `decisions --check` exits 0, ratchet wholly at floor

`0/0 undeclared, 0/0 unrouted-owner-ask, 0/0 vanished-owner-ask, 0/0
default-action-expired, 0/0 firing-diff`, and **31 of 31 firings transcribed onto
`DECISIONS_RESOLVED.md`.** No `MEANS-ESCALATED` — nothing a measurement could
settle is sitting on the owner's desk. **There is no `UNDECLARED` entry to arm
this audit; the standing instruction to arm one per audit is satisfied vacuously
and I am saying so rather than manufacturing one.** `decisions_default_action_expired`
MOVED 1 → 0, back to floor.

- **Armed:** `D41` and `D42`, both `decide_by 2026-10-18`, both monotone defaults
  that hold rather than act. `D40`, `decide_by 2026-10-10` — **two days** — default
  (v) = the status quo, and RANK 3/RANK 4 change its factual premise, which is why
  the addendum is committed today rather than described.
- **Five `CONDUCT-DESK` entries are desk-executable and three are STALE:** `D33`
  (due 2026-09-23, **15 days**), `D35` (due 2026-09-24, **14 days**), `D38` (due
  2026-10-04, **4 days**); `D39` and `D40` not yet due. These are not the owner's
  and nobody is asking them — they are listed precisely so a stale conduct entry
  cannot self-approve, and three have now been stale for a fortnight. **Nothing
  blocked on the owner that the system could have settled itself was found.**
- Four owner-asks on `docs/PROGRESS.md` all reached a desk and are attributed
  (`D37`, `D33`, `D41`, `D42`). Read back as instructed; its `FOR THE BUILDER`
  items 0–7 are live and item 0 states the 135th–140th audits' orders are all
  still open.
- **No owner decision was found acted on without being recorded.**

## 7. Bakeoff hygiene

`champions --check` exits 0 at its floor with **10 violations**, and the one that
matters is unchanged and known:

- **`VERDICT-IS-A-VOID` — Learning core.** The seat is held **BY VERDICT**, the
  file's strongest marking, off **`LC.03`, which is a VOID**. A VOID decided
  nothing. `D10` resolved by armed default on 2026-09-01 seating `wm-latent` off
  that VOID with a single-arm caveat on its face. This is the clearest live case
  of a VOID treated as a verdict in the project and it is correctly flagged by
  the instrument every day.
- **`TRIGGER-UNREACHABLE` — Learning core**: every pre-registered re-open trigger
  is a closed door (`LC.07` PILOT-BLOCKED, `LC.03` VOID-FORECLOSED, `UB.10` VOID).
  Contestability has decayed to zero behind a seat marked BY VERDICT.
- `ARENA-UNREACHABLE` + `TRIGGER-UNREACHABLE` — Fast/slow coupling (`DP.02` welded
  behind `LC.03`). `VERDICT-UNDECLARED` + `TRIGGER-UNDECLARED` — **World**, held
  BY VERDICT naming no row that bought it. `NO-ARENA` — ASR, Speaker ID.
  `UNCONTESTED` — Vision encoder, PLASTIC-ONLY.
- **`ARENA-MISSING` is now 0.** The overseer prompt still says *"8 seats today"*
  including `W.1`-`W.7`, `PL.*`, `LG.*`, `LT.*`, `T2.21`/`D1.0`; the tool reads
  **zero**, the `W.*` family is registered, and the declaration syntax the
  prompt's docstring proposed has been taken — `decl` on all 32 seats, and the
  declarations changed the reading on 10 seat/fields. **Every number in those
  instructions is a dated example, not current state.**

No decision was found made without a learning gate, and no winner chosen inside
a noise margin that the instruments do not already flag. The one anchor-decided
row, `T4.06`, discloses its own margins (`loss_reweight` CERTIFIED at +6.9 % of
anchor seed spread, 2 of 3 seeds improving) rather than hiding them.

## 8. The honest summary

**No. Today we are further from the ladder and the apple than we were yesterday,
and the list of green ticks did not even grow.**

The demonstrated count is 107 → 107 over 24 hours. 165 commits in seven days
touched the creature **zero** times. 89 % of the week's 76 settle events were
re-buys of certificates that this project's own tool edits had staled — the
system spent a week re-proving that its instruments still work. Four of the seven
first-ever verdicts in that week were audit organs; one was about Jack.

What is honest and worth holding onto: the organs are not lying. `coverage` says
three of the owner's commitments are claim-dead and names the foreclosures;
`run status` says six certificates could not be re-derived today and refuses to
round it off; the escalation ratchet is wholly at floor with every firing
transcribed; and when the one real experiment of the week came back at 14.666
against a bar of 7.0, **the bar did not move.** That is the machine working.

But the machine is now most of what there is. The thing GOAL.md actually asks
for — *"Climbing the ladder on attempt 40 after falling on attempts 1–39, without
anyone telling him to"* — has **no live falsifier**: `LT.03` VOID, `LT.02` FAIL,
`T3.06` VOID-FORECLOSED, `LT.04`–`LT.09` unimplemented. Smell, shelter and
thermal death are claim-dead. Queue depth is 7 dispatchable and 0 fresh. 28 free
GPU-hours expire in two days against a ladder with nothing to spend them on.
**The honest state is that the ladder cannot move until something is registered
or redesigned, and the one organ that could have been doing that this week spent
19 slots stopped and 10 more refused.**

---

## FOR THE BUILDER

0. **THE 135th–145th AUDITS' ORDERS ARE STILL OPEN AND NONE OF IT IS YOUR
FAULT.** You got two working slots in the last 24 hours and 23 refusals. Read
`docs/PROGRESS.md`'s `FOR THE BUILDER` items 1–7 as live, and its item 0 as still
accurate. **I am not re-ranking any of it, and nothing below displaces the
Review's item 1** (`T1.08` Steps 0+1), which is the project's largest blocker and
the only thing buying GPU-hours this week. Items 1–3 here are the new work from
today and they are all small.

1. **RE-RUN `T0.28` FROM A CLEAN TREE — ITS IMPLEMENTATION IS UNRECONSTRUCTABLE
AND I AM THE REASON THE DIRT IS GONE.** `run status` → DIRTY STAMPS: the PASS
recorded at `95c0a7b` *"reconstructs from no commit and was not preserved … it
cannot be recovered by anyone"*, because `docs/DECISIONS_NEEDED.md` was dirty at
run time. **That file is committed clean as of this report**, so the next regate
can buy a real stamp; until it does, `T0.28` is a Tier-0 PASS nobody can
reproduce. Cheap, mechanical, and it may ride in any slot. **Do not** touch the
spec or its bars — this is a re-run, not a repair.

2. **`scripts/lib_seal.sh` — THE SEAL SAYS "PRODUCED NOTHING" WHEN IT MEANS "DID
NOT WRITE *THIS* FILE", AND TODAY THAT STATEMENT WAS FALSE ABOUT A RUN THAT HAD
WRITTEN 123 LINES.** Evidence in RANK 1: `f87fbed` stamped *"exited rc=1 without
writing a word"* at 12:46:25 while the run's `DECISIONS_NEEDED.md` addendum sat
uncommitted in the tree. Repair, **reporting-only and strictly additive**: before
writing the "produced nothing" clause, check `git status --porcelain` for the
organ's other permitted outputs (`docs/DECISIONS_NEEDED.md`, `docs/LESSONS.md`)
and, if any is dirty, say *"wrote N uncommitted line(s) to <file> — NOT
committed"* instead. It may not change whether the seal fires, and it may not
commit anything on the organ's behalf. **Verify the defect yourself** against
`f87fbed` and the current working tree, and stop and route if it does not
reproduce.

3. **`scripts/lib_usage.sh:147` — THE STOP'S LOG LINE IS FALSE ABOUT THE CODE
THAT PRINTS IT.** `:128` returns 0 for any `pct < 90` **before** `.usage-resumed`
is read at `:130`, so the file is a ceiling override for operating *above* 90,
not a resume latch, and *"all agents paused until the owner resumes"* is wrong —
the meter reset released 19 slots today with no owner act and no `RESUMED BY
OWNER` line (RANK 3). Change the **message only**, to something that says what
the branch does: *"STOPPED at N% weekly usage — no organ runs until the weekly
meter falls below 90% or the owner writes .usage-resumed"*. **`PACE_FLOOR`,
`PACE_CAP`, the 90 constant and every branch stay byte-unmoved**; this cannot
change what `usage_gate` returns. Staleness bill 0 — no spec's `IMPL_DEPS` names
this file. **It also fixes a live owner-facing error**: `D30`'s standing report
and `docs/PROGRESS.md:431` both repeat the false clause to the owner.

4. **MAKE A USAGE-STOP FIRING SURVIVE ITS OWN ENDING.** `dark_slots` read **0** in
the 13:07 line, one slot after a 19-slot hard stop, because it is a trailing
streak that any `rc=0` zeroes (RANK 2). Extend the existing `usage_ledger.jsonl`
mark to the `usage_gate` stop branch so a firing is recorded where a later
success cannot erase it. **Reporting-only and additive**; it must not become a
gate, and `dark_slots` itself must keep its present definition and its floor of
0 — do not redefine a counter to make a red go away.

5. **`pace_gate`'s WEEK-ELAPSED DENOMINATOR IS NON-MONOTONE AND NOTHING RECORDS
IT.** Measured today: elapsed 15 % → 16 % → **4 %** → 4 % → 5 % across 13:07–18:07,
moving the line 36 % → 28 % with no policy change (RANK 5). Print the previous
elapsed reading beside the current one in the `PACING:` line so a reset is
visible in the log rather than inferable from six lines of it. **Reporting-only.
Do not change the line's arithmetic, and do not "fix" the denominator** — whether
the window `claude_usage.py` reports is the week this gate assumes is a question
for the Review, and the drift today was in the conservative direction.

6. **GOAL.md CITES FOUR SPECS IN THE PRESENT TENSE THAT CANNOT RUN — AND THEY
ARE NEW.** `coverage` → `CITED-BUT-UNRUNNABLE`: `GEN.02`, `GEN.03`, `GEN.06`,
`GEN.09`, all welded behind `LC.07`, all cited at GOAL.md §187–§199 as what comes
after the jungle. The honest repairs are to fix GOAL.md's tense or to route the
revival; **never add to `GOAL_UNRUNNABLE_BASELINE`, which is shrink-only.** The
existing row `gen-four-revival-needs-an-affordable-lc07-successor` (OPEN, DUE
2026-10-11) owns the revival half — so this item is the *text* half only, and if
editing GOAL.md is not yours, route it and say so.

---

## FOR THE OWNER

**1. `D40` fires in two days and the entry you would have ruled on was factually
wrong about its own mechanism. The correction is committed with this report; the
option set is untouched.** The 90 % stop fired for the first time ever on
2026-10-07 — **19 consecutive hourly slots** — and **released itself** at the
weekly meter reset with no act by you. `.usage-resumed` has never existed on
disk; the last `RESUMED BY OWNER` line in the log is 2026-08-12. But this entry,
`D30`'s standing report and `docs/PROGRESS.md:431` all tell you that resuming
needs a file *written by your hand*, and `scripts/lib_usage.sh:128` shows that is
false. **Why it matters for your ruling and not just for the record:** default (v)
is "the status quo", and on the current text that means accepting *"the builder
stays dead until I write a file"*, while on the measured mechanism it means
*"the builder loses up to one usage week per firing and then restarts itself"*.
Those are different commitments, and **the error in the record argued against
(v) more strongly than the truth does** — reported because the direction being
favourable to action does not make it accurate. I have proposed no option,
recommended none, moved no `decide_by`, and touched nothing the gate reads.

**2. NO DECISION ASKED — three `CONDUCT-DESK` entries have been stale for up to
15 days and they are not yours.** `D33` (due 2026-09-23), `D35` (due 2026-09-24),
`D38` (due 2026-10-04) are desk-executable by construction: a desk is supposed to
execute them, report, and not ask you. They are *listed* in `decisions --check`
only so a stale conduct entry cannot quietly self-approve. Nothing is blocked on
you here and no answer is wanted — it is on this page because the same organ that
is correctly refusing to escalate them has also not executed them for a
fortnight, and you are entitled to know that the desk is behind on its own work
and not only on yours.

**3. NO DECISION ASKED — 28.03 free GPU-hours expire Saturday 2026-10-10, for the
fourth week running, and for the first time that is not purely waste.** W37
1.379 h, W38 0.918 h, W39 1.072 h, W40 1.971 h drawn of 30 — **~114 hours
unbought in four weeks.** Nothing was manufactured to spend them and nothing
should be: `coverage` reports **7 dispatchable specs today, all 7 VOID, 0 fresh
dispatches**, and three cost classes with no path in at all. The ladder, not the
quota, is the binding constraint. What is different this week: all three W40 jobs
are real `T1.08` work, attempt 4 came back an honest FAIL at 14.666 against a bar
of 7.0, and **the bar did not move**. The 19 slots the hard stop took were taken
from that.

**4. NO DECISION ASKED — the creature has not been touched in over a week, and I
re-measured it rather than repeating a number.** 165 commits in seven days; zero
to `UnifiedBrain.py`, `TrainingPipeline.py`, `playground.py` or
`EpisodicMemory.py`. Three organs have now measured this independently across
three different windows (Review 249 commits on 10-04, 144th audit 203 commits on
10-07, this audit 165 commits today) and all three got zero. The demonstrated
count has stood at 107 for the whole period. `D42` — armed, `decide_by
2026-10-18` — is the live home for the part of this that needs you: three of your
own constitutional commitments (**smell**, **shelter**, **thermal death**) have no
falsifiable claim behind them at all, every claim spec parked or foreclosed on its
own pre-registered rule. Its default is to HOLD and say so, which keeps the hole
visible and does not let a desk pick which of your commitments matters most.
