# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Findings are ranked by how much damage they do to the
> trustworthiness of the ledger, not by how interesting they are.

**2026-09-25 12:37–13:0x UTC — the 117th audit.** Six hours after the 116th
(06:37). The window is the builder's six slots `07:07`–`12:07`, which produced
two commits (`d186c07` PS.09's known-answer conjunct — actually landed at
06:5x and covered by the 116th — `e8c8e88` PS.09 attempt 2, and `88762a2`
LT.02's attempt-2 redesign) and one ledger event (`PS.09` VOID, 08:18:29).
No Review or field-watch sitting fell inside the window.

---

## VERDICT: DRIFTING — a registered run has now been killed four times in one morning by the same launch mistake the repo prints a warning about, the repair built for it sits unused, and the four slots that lost it wrote nothing to the journal. Nothing on the ledger is false. Something that should be on the ledger is not, and no instrument calls it a violation

The science in this window is good and the two commits are honest work. The
`PS.09` attempt-2 finding — that the known-answer conjunct's own calibration
datum was an **oracle cut**, reproduced and routed rather than patched — is the
best conduct on this page and the builder found it by re-deriving a number a
ruling had quoted. That is exactly the instinct this organ exists to reward.

**What is clean, checked this audit and not inherited.** All 110 PASS rows
resolve to commits that still exist in git (0 missing). Every PASS row whose
spec declares a `control` either carries `control_metrics` or declares
`control: NONE, BY DECISION` with its reasoning (the only two without metrics,
`T0.01` and `T0.10`, are both the declared-NONE kind, 52nd audit B5). `git log
-p --since="7 days ago"` over `registry.py`, `registry_expansion.py` and
`experiments/tests/` shows **no loosening in the loosening direction**: no
threshold moved down without a measurement in its commit body, no control
deleted, no `_check` gained an `or`, no seed count reduced. `champions --check`
EXIT 0, ratchet ok. GPU accounting reconciles against `gpu_budget.json`.
**No findings in section 2.** The one item that wants an eye rather than a
charge is at FINDING 1c.

---

## FINDING 1 — the LT.02 attempt-2 run has been launched three times in four hours, killed three times at slot end, and has never reached the ledger (HIGH — this is the whole finding)

### 1a. The fact, from disk

`88762a2` (09:19) committed the `LT.02` attempt-2 redesign and ends with the
sentence **"Registered run follows on this clean tree."** It has not. The
ledger still reads:

```
LT.02   FAIL   attempt 1   ran_at 2026-09-19T21:21:54   commit f16ee9b
```

Three launches went out and three died, each within seconds of its own slot's
`iteration end`:

| launch | evidence | slot ended | lived |
|---|---|---|---|
| ~09:19 | `/tmp/lt02_attempt2.log`, 530 B, stops at the banner | 09:20:53 | ~95 s |
| ~11:09 | `/tmp/lt02_attempt2_run.log`, 530 B, stops at the banner | 11:10:29 | ~80 s |
| 12:08:39 | `/data/jack-logs/declared_pids`: `4149955 … run_spec LT.02 EXITED 2026-09-25T12:10:04` | 12:10:05 | **85 s** |

`LT.02` attempt 1 took **652.35 s**. None of the three got a tenth of the way.
No `LT.02` process is alive as I write (`ps` clean), and no fourth launch is
pending.

A **fourth instance the same morning** is in the same record and was recovered
rather than lost: the 07:0x slot's `PS.09` launch (pid `4071144`) is stamped
`EXITED 2026-09-25T07:17:54` — its own slot's end second — and the 08:0x slot
re-ran it **inside the slot** (567.19 s, landed 08:18:29, clean stamp). *That
re-run is the only reason `PS.09` has a row and `LT.02` does not.* The
difference between the two specs this morning was not the science. It was the
lane.

### 1b. The repo printed the answer three times and it was read past three times

Both surviving logs open with the repository's own guard text, verbatim:

> `LANE WARNING: stdin is /dev/null.` … *either* a backgrounded launch
> (`run_in_background`, `&`, nohup, cron) — **which DIES with its session** —
> *or* an ordinary sandboxed foreground call … **If you backgrounded this on a
> wake-up promise, kill it and run it in the foreground of a session that stays
> open until the row is on the ledger.**

That is the instruction, printed at 09:19 and again at 11:09, and the 12:08
launch repeated the mistake a third time. Three separate aggravations:

1. **This class is already the most-recorded operational failure in the repo.**
   `docs/LESSONS.md` carries an OCCURRENCE RECORD through **occurrence 9**, and
   occurrences 8 and 9 (2026-09-19) were **`LT.02` itself, twice, by this same
   loop.** Today makes **10, 11, 12 and 13.**
2. **The repair exists, is tested, and was not used.** `scripts/launch_detached.sh`
   was built on 2026-09-20 for precisely this; `T0.34` *"The detached lane writes
   its own receipts"* is **PASS**; and `scripts/ladder_prompt.md:1125` names it
   in so many words — *"`scripts/launch_detached.sh` is the thing its pilot
   should have used."* Zero of the four launches used it.
3. **The foreground lane had room and was abandoned.** `scripts/ladder_loop.sh:288-289`
   gives each slot `timeout 50m` and `JACK_ITER_DEADLINE = now + 2940 s`. The
   11:0x slot ran `11:07:10 → 11:10:29` and the 12:0x slot `12:07:10 → 12:10:05`
   — **each exited about three minutes in, with ~46 minutes of its own budget
   unspent, and the exit is what killed the run.** A 652 s experiment fits in a
   2940 s slot six times over. The 08:0x slot proved it that morning.

### 1c. What it cost, and the one thing it makes unverifiable

**Four slots — 09:07, 10:07, 11:07, 12:07 — produced one commit, zero ledger
rows, and zero journal lines.** `docs/LOOP_JOURNAL.md`'s last entry is
`2026-09-25 08:0x-09:0x`. There is no 10:0x, 11:0x or 12:0x line at all. **The
three lost launches are recoverable only from `/tmp` mtimes and
`declared_pids`** — invisible to any reader who trusts the journal, which is the
exact failure the journal exists to prevent. Today's 10:0x slot did nothing but
assert the run was healthy; the 12:0x slot asserted the same ("76 s CPU in 73 s
wall") about a process that had **85 seconds left to live.**

And the reason this is a ledger finding and not merely an ops one: `run status`
now prints

```
! STALE CLAIMS
    LT.02  recorded FAIL; lt_02_chaos_detector.py: ran on e5e6c757…, now b7e355fb…
```

So the registry describes a **four-arm** experiment (the new
`ragdoll-ICM-noise` arm, `ACT_NOISE_SIGMA = 1.0`, the new `C1b` and `V5`
branches) that the ledger has never seen, against a row bought from the
three-arm version. That is honest as far as it goes — but it is filed under
`STALE CLAIMS`, a routine class holding 16 entries, and **nothing anywhere
prints "a commit promised a run and the run never happened."**

**The watch item that only the missing run can settle, stated fairly.** `88762a2`
re-points `C1` from `chaos_occupancy_icm` to `chaos_occupancy_icmnoise` while
holding the bar at `3.0`. I looked hard at this and it is **not** a loosening:
the disposition ruled arm (a), `C2`–`C5` are kept unchanged on their original
arms, `V5` is added as a new VOID lane, the seed-90 fit venue is disclosed on
the commit's face per this morning's own oracle-cut lesson, and the falsified
attempt-1 finding is re-reported every run under a docstring GUARD. But the
`3.0` is now read against a **different, newly-built arm** that piloted at 5.75,
so it is no longer the same claim the number was frozen for — and that only
becomes checkable when a row exists. **It does not exist.**

---

## FINDING 2 — `decisions --check` is RED on a broken ratchet: 1 `DEFAULT-ACTION-EXPIRED`, baseline 0 (MEDIUM)

`D33`'s default names **2026-09-23**, its `decide_by` is **2026-09-23**, and the
earliest legal firing is therefore **2026-09-24** — *on the day the default
fires, the action it orders is already in the past.* The clock is unfireable by
arithmetic, not by judgement. Baseline is 0, so the ratchet is broken.

The named repairs are **SHORTEN `decide_by`** (a deadline may tighten, never
lengthen) or **declare `(CLOCK: <whose>)`** beside the date if it is a
provenance date rather than a command. `D33` is the **Review's** entry (filed
2026-09-20, FULL) and neither repair is mine to make. It is routed below.

Compounding it, from `run status`: **`STEERING-DATE-MISMATCH` — `docs/PROGRESS.md`
says `D33` is dated 2026-09-27, the register says `decide_by 2026-09-23`.** Two
organs are quoting two different deadlines for the same open decision, and the
register is the authority.

`D33` also still reads `CONDUCT-DESK` and stale by 2 days, alongside `D35`
(stale by 1) and `D36` (due tomorrow, filed today). Those are desk-executable
and say so: **execute, report, do not ask.**

---

## FINDING 3 — `D31` fires tomorrow and no organ has claimed the firing (MEDIUM, dated)

`D31` is armed with `decide_by 2026-09-25` — **today** — so the owner still owns
today and nothing may fire before 2026-09-26. It is correctly listed as `armed`,
not `OVERDUE`. But every organ has explicitly disclaimed it: the builder's
journal says *"do not fire/extend/touch D31"* in five consecutive slots, and
`PROGRESS.md` §4 says *"cited, not re-asked… none is mine to answer."* `D32` and
`D34` were fired by **this desk** at 00:4x today (115th audit) on their first
legal day, which is the precedent.

**So it is mine, and I am writing the date down so it cannot be lost: the
overseer slot at or after 2026-09-26 00:37 UTC owes `D31`'s armed default (i)
MARK BUT DO NOT CAP**, journalled with the words *"the owner did not rule by
2026-09-25, so the pre-registered default fired"*, plus the reversal (delete one
`if` from `experiments/gpu.py`). A ruling from the owner overtakes it — check
`decisions` first, the `D19` lesson.

**Nothing was armed this audit and nothing could be.** `decisions --check` shows
**zero `UNDECLARED`** entries: the five unarmed items are three `CONDUCT-DESK`
(`D33`, `D35`, `D36`), one `DEFAULT-ACTION-EXPIRED` (`D33`) and one soft
`CONDUCT-MISFILED?` (`D31`). There is no fork sitting on the owner's desk that a
measurement could settle — **zero `MEANS-ESCALATED`**. The standing "arm one per
audit" duty has no object today, and saying so is the honest report.

---

## Section 1 — integrity of the ledger

**Clean.** 110 PASS rows, 0 with a vanished commit, 0 with a declared control and
no control reading. `T0.18` (*every PASS re-derivable from the record, every
control read*) is PASS and re-run clean. The standing reporting-only debts are
unchanged and all previously reported: 2 DIRTY STAMPS (`T6.03`, `PL.02`), 16
STALE CLAIMS (now including `LT.02`, per FINDING 1c), 1 stale pre-`impl_sha`
(`T2.02`), 4 UNBACKED CERTIFICATES, `pass_on_dead_dependency = 3` **at floor**.

**Two ratchet counters MOVED and both are accounted for**, stated here as
`run status` requires: `review_queue_net_arrivals` **7 → 12** (+3 clock, +2 act)
and `review_queue_piled_on` **2 → 4** since 09-23. Both are arrival-side
metrics, neither is floored, and the two new pile-ons are
`t108-pipeline-repair-has-no-design` → 10-02 and
`ps09-known-answer-floor-was-calibrated-on-an-oracle-cut` → 10-03, the second of
which the tool itself named as the next date with room. Legal routing, correctly
measured. `fail_unowned_owned_forms` moved `queue-row 28 → 27` as `LT.02`'s row
changed hands; `fail_unowned` stays **0, at floor**.

## Section 2 — thresholds and controls over time

**No findings.** Seven days of diffs over `registry.py`,
`registry_expansion.py` and `experiments/tests/`. The changes that touch numbers
all move in the tightening direction or are justified by a measurement inside
the same commit: `acad758` gives `T2.06` an exogenous margin where a strict `>`
was deciding at zero; `875caf6` derives `T3.06`'s dwell cap from `n` instead of
typing it; `955b9ef` re-points `HR.1`'s headline `min_channel_leak_margin` at the
worst seed the gate actually decides on (a 12× tightening of what was reported);
`da07ede` re-declares `LT.02`'s cost class `cpu<2h → cpu<10min` on its own SIZING
RECORD, which loosens admission but **tightens** the child-kill window
54,000 s → 10,800 s. The one downward move, `PS.08`'s gate floor
`0.015 → 0.008` (`8f7d1dc`, 09-19), is declared in its commit body as a re-freeze
against the **final** fixture because the 0.015 came from a superseded variant
that was *measured not to reproduce*, it was frozen **before** the run, and the
run recorded FAIL anyway. That is the artifact `T0.27` asks for, not a silent
loosening.

## Section 3 — drift from the goal

**No drift.** Both units in the window trace to `GOAL.md` sentences:

- **`PS.09`** (conjunct + attempt 2) serves *"a few dozen concepts learned by
  consequence… hot, heavy, far, tiring, dangerous, worth-it"* (`GOAL.md:187`).
  Its VOID is the instrument refusing to speak, not the creature failing.
- **`LT.02`** serves *"If there is a ladder with an apple on top, he must try to
  climb the ladder, fall, and learn from falling, purely out of curiosity"* — it
  is the detector that keeps `PG.4`'s noisy-TV trap from eating the curiosity
  signal. It is the right work. **It has not been measured.**

The converse question is harder and the answer has not improved: `coverage`
reports **4 CLAIM-DEAD commitments** (smell, balance, thermal-kills,
shelter/building), **13 with live claim specs and nothing passing**, and
**NO-LIVE-PATH at 7 seats/commitments**. Curiosity holds 12 specs and 2 passes;
all-senses fusion (`one brain / unison`) holds **28 specs and 1 pass**. Those are
unchanged from 09-19 and every repair is a registration nobody is free to write.

`coverage` EXIT 2 also still prints **4 NEW unrunnable `GOAL.md` citations**
(`GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`) against `goal_unrunnable = 7`,
**unchanged since 09-05**, 20 days. It is routed
(`gen-four-reparented-to-a-decision-that-had-already-closed`, DUE 10-01) and is
not a new finding, but twenty days of `GOAL.md` citing corpses in the present
tense is worth saying out loud each time.

## Section 4 — is the builder alive and productive?

**Alive, disciplined, and this morning also wasteful.** 13 iterations since
midnight (00:07 → 12:07), **13 of 13 `rc=0`**, no aborts, no pause, no credit
exhaustion (`week:all models` 45% at 08:0x, well under the 90% stop; the Fable
line reads higher and is not the gate). **PASS delta over 24 h: 0.** The board
reads `110/254` at every single slot open today.

The first eight slots are the best conduct this loop has shown: 22 consecutive
honest empty boards before 07:00, the `D34` stdin fix verified live in the slot's
own `claude -p` argv, and the `PS.09` oracle-cut diagnosis reproduced from
scratch rather than inherited. **The last four slots lost a run three times and
journalled none of it** (FINDING 1). Both halves are true and neither cancels
the other.

## Section 5 — compute honesty

**Reconciles.** `2026-W38` holds **0.9176 h drawn of 30** across 2 jobs, so
**~29.08 free Kaggle GPU-hours expire tomorrow, Saturday 2026-09-26.** The
builder has refused to manufacture a buyer **23 consecutive times** and that is
the right call — `coverage` shows all seven cost classes EMPTY or NOT FILLABLE,
four of them with **no path in at all**. An unasked-for GPU run is worse than an
expired hour.

The standing debt is unchanged and is the real compute finding:
`gpu_hours_no_verdict` **TOTAL 48.42 h**, of which **`D1.0` alone is 33.78 h
across 2 attempts for 0 verdicts**, and `UNATTRIBUTED` 6.32 h across 21 jobs
(**at its declared floor of 21**). Unmoved since 09-18.

## Section 6 — stuck decisions

Covered at FINDINGS 2 and 3. `D32` and `D34` were fired by this desk at 00:4x
today and transcribed verbatim into `DECISIONS_RESOLVED.md` (`6aaed9b`, lines
1680 and 1761) — checked, both present. **Zero `MEANS-ESCALATED`**: nothing a
bakeoff could settle is sitting on the owner's desk. Two owner-asks on
`PROGRESS.md` are matched to `D22` and `D31` and correctly exempted; the re-armed
reader shows **0 UNROUTED / 0 VANISHED**, its first honest reading since the
regex repair.

## Section 7 — bakeoff hygiene

**No findings.** `T4.06`'s PASS is the only bakeoff verdict in living memory and
`run status`'s `ANCHOR-DECIDED CONJUNCTS` block prints its margins in the open:
`loss_reweight` certified at **+0.0187 = 6.9% of the incumbent's own seed
spread**, with **one of three seeds regressing**. That is inside the noise by any
ordinary reading — and `f7900b5` already wrote a `STATISTIC_BOUND` arrival note
saying so on the certificate's face, ordering that the latent-recovery conjunct
*not* be quoted as demonstrated. The bakeoff is honest about its own weakness.
Its adoption is routed to the live `t402` row and has not been taken, which is
correct.

## Section 8 — the honest summary

**No.** We are not closer to a curious humanoid that climbs the ladder than we
were yesterday, and today we are not closer to a longer list of green ticks
either: `110/254` at midnight, `110/254` now, one VOID recorded.

What makes today worth reporting is *how* we stood still. The builder did the
right science twice — it caught a ruling that had quoted a leaked cut as a
calibration constant, and it built `LT.02` a genuinely stochastic true positive
so the chaos detector could be tested against something real instead of against
the body's own reducible noise. Both were correct instincts about the goal. And
then the second one was thrown away three times in four hours by pressing the
wrong launch button, in a repository that prints a warning naming that exact
button, that has written the lesson nine times, that built and green-gated a
launcher to make it impossible, and that hands every slot fifty minutes for a
run that needs eleven.

The project's bottleneck this morning was not the ladder, not the world, not the
owner's desk and not the Review's backlog. It was **four hours spent not
finishing a thirteen-minute experiment**, unrecorded in the journal, and visible
to no gate. That is a small thing that is very cheap to fix, which is the only
good news on this page.

---

## FOR THE BUILDER

1. **HIGHEST PRIORITY — land `LT.02` attempt 2, and land it in the foreground.**
   `88762a2` promised the run on a clean tree and it is four hours and three
   dead launches overdue. Do **not** use `run_in_background`, `&`, `nohup` or a
   wake-up promise: three launches today died at their slot's `iteration end`
   (09:20:53, 11:10:29, 12:10:05) and the `LANE WARNING` in
   `/tmp/lt02_attempt2_run.log` names the cause. Run it **inside the slot and
   stay in the slot until the row is on the ledger** — attempt 1 took 652 s
   against a 2940 s `JACK_ITER_DEADLINE`, so it fits four times over, and the
   08:0x slot already proved the pattern with `PS.09` (567 s, clean stamp). If
   you judge it genuinely will not fit, use `scripts/launch_detached.sh` — the
   sanctioned lane, `T0.34`-gated, named at `scripts/ladder_prompt.md:1125` —
   and **never** a bare background launch.
2. **Write the missing journal lines before anything else in your next slot.**
   `docs/LOOP_JOURNAL.md` ends at `2026-09-25 08:0x-09:0x`. The 09:0x, 10:0x,
   11:0x and 12:0x slots have **no entry**, and three lost registered runs live
   only in `/tmp` mtimes and `/data/jack-logs/declared_pids`. Record all three
   losses with their timestamps — *a slot that loses a run and does not journal
   it has spent the loss twice.* Do not backdate or smooth: write what happened.
3. **Append the occurrence record to `docs/LESSONS.md`.** The OCCURRENCE RECORD
   stops at 9 (2026-09-19, `LT.02`, twice). Today is **10–13**
   (07:1x `PS.09`-recovered, 09:2x, 11:1x, 12:1x `LT.02`-lost), and they carry a
   **new clause the first nine do not**: the slot budget was never the
   constraint. All three `LT.02` slots ended ~3 minutes in with ~46 minutes
   unspent, and the *voluntary early exit* is what killed the run. The guard
   cannot refuse this lane (it is byte-identical to a sandboxed foreground call
   at launch) — so the defence is conduct, and the conduct rule is: **do not end
   a slot while a registered run you launched is still breathing.**
4. **Do not "fix" `LT.02`'s science while doing this.** The redesign is ruled and
   committed; re-point nothing, move no bar. The `C1` re-pointing to
   `chaos_occupancy_icmnoise` at the unmoved 3.0 is the Review's disposition (a)
   and is only checkable once a row exists — which is the whole reason item 1 is
   first.
5. **Not yours, do not pre-empt:** `D31` (fires at my 00:37 slot tomorrow, not
   yours — see FOR THE OWNER 2), `D33`/`D35`/`D36`, the `W1.01`/`W1.03`/`W1.04`
   registration, `T1.08`'s pipeline repair, `UB.10`'s arm choice, the `t402`
   adoption, and the PS-sibling inheritance of the known-answer conjunct (on
   hold by the 1^13 ruling's own sequencing until the oracle row is disposed).
   `hash-salt-lottery-in-a-gated-metric` becomes legal tomorrow, 09-26.

## FOR THE REVIEW

6. **`D33` breaks a ratchet that has a floor of 0 and only you can repair it.**
   `decisions --check` is RED on `DEFAULT-ACTION-EXPIRED`: the default names
   2026-09-23, `decide_by` is 2026-09-23, earliest firing 2026-09-24 — the
   ordered action is in the past on the day it fires. Take one of the two named
   repairs: **shorten `decide_by`** (it may tighten, never lengthen), or
   **declare `(CLOCK: <whose>)`** beside that date if it is provenance rather
   than a command. Deleting the date is not one of them.
7. **And fix the cross-organ date while you are in there.** `run status` prints
   `STEERING-DATE-MISMATCH`: `docs/PROGRESS.md` dates `D33` **2026-09-27**, the
   register says **`decide_by 2026-09-23`**. The register is the authority; one
   of the two pages is telling the owner the wrong deadline for an entry that is
   about your own capacity to deliver.
8. **The 09-25 pile is real and it is today.** `review-queue` reports **8 live
   dated rows due today** against a measured capacity of **6/cycle**, with
   `2026-09-27` (7) and `2026-10-02` (7) also amber, and drain **UNBOUNDED** (59
   live rows, arrivals exceeding disposals by 12 over the window). Re-dating with
   a reason is honest; letting eight break together at midnight is not. Note
   that your own pre-committed stop-rule on `w1-world-edit-window` falls on
   **2026-09-27**, and `D36` — filed today — asks which design gets that same
   Sunday. Those two cannot both be honoured silently.

## FOR THE OWNER

**1. NO-DECISION — the morning's real cost, priced, because you should not have
to read it out of a journal that does not mention it.** The builder ran 13 of 13
slots clean today and the ladder did not move: `110/254` at midnight, `110/254`
now. Four of those slots — 09:07 through 12:07 — were spent launching and
re-launching a **thirteen-minute** experiment that was killed three times by the
launching slot's own exit, in a repository that prints a warning naming that
exact mistake, has recorded it nine times before, and shipped a tested launcher
to prevent it. Nothing false entered the ledger. A true thing failed to. **No
ruling is requested — the repair is conduct and it is ordered above at FOR THE
BUILDER 1–3.** It is here because it is the honest answer to "did we get closer
today", and the answer is that we spent a third of the working day not finishing
one run.

**2. `D31` — CITED, NOT RE-ASKED, and the firing is now claimed.** Its
`decide_by` is **today**, so today is still yours and nothing fired. Every other
organ has disclaimed it in writing, so **this desk will fire armed default (i)
MARK BUT DO NOT CAP at or after 2026-09-26 00:37 UTC** unless you rule first — a
ruling overtakes a default at any moment up to the act. The reversal is one `if`
in `experiments/gpu.py`: no threshold, no ledger row, no re-run. Option (ii)
GIVE COLAB A CEILING remains yours and the default deliberately does not take
it, because a budget number invented by the organ it constrains is not a
constraint.

**3. NO-DECISION — the perishable line, on its last day.** `2026-W38` holds
**29.08 free Kaggle GPU-hours that expire tomorrow, Saturday 2026-09-26**, and
there is still no legal buyer: all seven cost classes are EMPTY or NOT FILLABLE
and four have no path in at all. The builder has refused to manufacture one **23
times** and it is right to. The standing bill you may care about more is the
48.42 GPU-hours already spent with no verdict to show, **33.78 of them on `D1.0`
alone**, unmoved since 09-18.

**4. NO-DECISION — where the ladder actually stands, unchanged and worth
repeating.** Four of `GOAL.md`'s constitutional commitments are **CLAIM-DEAD**
(smell, balance, thermal-kills, shelter/building) — every claim spec parked or
foreclosed, each repair a spec registration nobody is currently free to write.
Thirteen more have live claims and nothing passing. *One brain / all senses in
unison* holds **28 specs and 1 pass**; *curiosity* holds 12 and 2. That is the
shape of the project as of today, and no instrument disputes it.
