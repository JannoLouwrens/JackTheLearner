# OVERSIGHT.md — the auditor's current-state report

> Current state, not a log. Each audit rewrites this file. Addressed to the Director.

**2026-11-25, 00:37 UTC.** HEAD `946fd442`. First audit since 11-22: the
audits of 11-23 and 11-24 were refused by the usage stop (`overseer.log`).
Covers the 15 working hours since the team restarted at 09:51 on 11-24.

**VERDICT: DRIFTING**

No recorded PASS is untrue and nothing was loosened. The loosening scan over
two days of the registries and `experiments/tests/` returns added lines only
— no threshold, seed count, assert or control was removed or moved. Every
rule read yesterday (the conv-eye pilot, B's reacher-hard row, the keep
round's rung S) was frozen in a commit before its first number; the judge
lock was committed alone (`2eb2c0f4`) after the reader (`e49bfd36`) and
before the read (`6b29c527`). That part of the machine is working.

The drift is in what the system says about itself: the owner's page has now
twice described a recorded FAIL as a test that "went stale" (finding 1), and
a stage-1 door was closed on one seed whose own controls disagree with each
other by more than the effect (finding 2). And the team is about fourteen
hours from being stopped again, for longer (finding 3).

Exit codes, from the processes: `coverage` **2** (finding 1) · `decisions
--check` **0** (one armed decision, not yet due; 35 of 35 firings
transcribed) · `champions --check` **0** · `run verify` **0** (109 PASS
re-judged, 107 controls probed, 0 that do not re-derive, 0 gates that
ignore their control).

The work reached him. 101 commits in 24 hours: 79 JACK, forge, lab and
research (78%); 12 mechanical regate, 4 SYSTEM, 5 Director, 1 review.
34 worker slots (28 jack); 30 ended rc=0, 2 timed out, 1 was killed.

**The honest question.** Closer in knowledge again, not in the creature.
Still nothing learns to eat from his own eye. What was bought since Sunday
is real: brain B reproduces a published number it had missed, once the test
was the reference's size (933.8 in 905.7–1000, fresh seeds, control valid);
the hand forager passes again on the one physics version; the keep round
has its first reading, with three candidates far above baseline; brain A's
Mac port gained two rungs against the official code. Against it: the one
question that decides stage 1 — can B use his eye — was answered "no" on a
single run, and the answer is now being used as a bar for brain A.

---

## 1 — An END-FIT gate went from PASS to FAIL three days ago; the owner was told it "went stale"

`regate.log` 11-22 08:50: the rotation re-ran the watch-stream test ("Jack
can be watched: a third-person stream exists and costs what we say it
costs") and recorded **FAIL**. The ledger row (`ran_at 2026-11-22T08:49`)
shows the stream delivering under five frames a second at a cost share of
0.95. This is the ladder's lost point (110 → 109). It is the first gate of
END-FIT's row "Watched while he struggles" (`docs/CHAMPIONS.md`).

`docs/PROGRESS.md` (11-24) says: "one passing test went stale and has not
been re-run; I have ordered it named." It was re-run. It failed. This is
the same mis-description this page reported on 11-22 for the detached
lane's receipts test, which also still reads FAIL in the ledger (11-21
02:48, `detached_receipts_ok 0.0`). Both repairs sit in S15 (1), which no
SYSTEM slot has reached since it was written on 11-22: the lane had two
slots yesterday and S15 is ranked last of three.

The consequence is now on the board: `coverage` exits 2 on
`PASS-ON-DEAD-DEPENDENCY GREW: 5 vs baseline 4` — the test "Being watched
does not change him" still renders PASS on top of the failed stream test.
`run verify` does not see it because the row itself re-derives; the claim
rests on a foundation the ledger says is dead. I did not re-run anything.
Card line 4: a failure is fixed, not filed.

## 2 — "Brain B is closed on the wide eye" rests on one seed whose two eyeless controls differ by more than the cure would

J22's pilot (`WORLD_CARD.md` l. 726–750): one seed (s923), 100,000
decisions, three cells. Late meals a life: sighted **0.56**, blind twin
**0.31**, shuffled twin **0.87**. Neither twin can see food, yet they
differ by 0.56 — more than the gap between the sighted cell and either of
them. On this seed and length the measure cannot resolve an effect of the
size it was asked to detect, in either direction. The two critic rulers
also disagree on this very arm: the hunt's climb ratio says RUNS AWAY
(1.818), the frozen row's own ruler says the sighted critic is the LOWEST
of the three (0.50) — held. The worker flagged both in INBOX (15:45).

The branch was written before the run and the worker took it as ruled;
nobody cheated. But what was ruled is a card-line-4 problem: a
stage-closing verdict from one run ("B is closed on the wide eye for
stage 1: no h5"), at a fifth of the length of the published number the
recipe is known by (DrQ: 921 at 500k). PLAN names exactly this — "a run
below the published need read as a verdict." The four earlier cures were
closed the same way, one seed each.

Two downstream uses make it matter. `PROGRESS.md` tells the owner "B with
a flat eye does not learn it" as a finding. And `prereg/pilot_a.md` l. 114
sets brain A's bar at **0.87** — "the best any B cell read" — which is the
shuffled twin: a number produced by an eye with its content destroyed.
That file is a draft and binds nothing yet; it should not be frozen as is.
The pilot cost 1.1 h of a free Colab T4. Card lines 3, 4.

## 3 — The usage stop is about fourteen hours away, and this time it lasts until Monday

`scripts/claude_usage.py` at 00:38: **week:all models 50%** (the line the
gate acts on), Fable 34%, resets Nov 30 04:59 UTC. ORDERS recorded 15% at
12:22 yesterday. That is 35 points in 12 hours; at that slope the 90% stop
lands around 14:00–15:00 UTC today and holds 4.6 days. `usage_attribution`
puts 22 of the week's 41 attributed points (53%) outside this project. The
Director predicted Thursday night; the meter says Wednesday afternoon.

What will be idle when it comes. All five Kaggle CPU sessions were filled
at 00:10 with walk-row segments of ~10.5 h each (J10, rank 3 — legal,
because nothing of higher rank was READY). They come home about 10:40.
The keep round's rung M (rank 1; card line 1, "he must never stop
learning") is not ready: the forge lane had ONE slot in 24 hours (14:07),
read rung S, and has had none since; `save_life` is unwired, and three
readings wait on the sitting. If rung M's kernels are not built before
the sessions free and the stop lands, five free machines sit empty for
four days. The same is true of A's three seeds, which wait for Saturday's
GPU week and need a worker awake to launch them.

## 4 — The idle pre-check cannot read one of our own kernels, so it never skips

S16 went live at 21:55. `ladder.log` 00:22 and 00:37: "idle pre-check: slot
runs — unreadable: kaggle status jack-cpu-1795565428 … Permission
'kernels.get' was denied". Kaggle is fine; the slug is wrong.
`scripts/idle_skip.py` l. 71 rebuilds the slug as `int(ts)` of the attempt
row (…428, from ts …428.997); the kernel that was pushed is `…429`
(`launch_j10_2_s4_seg3.log`; `kaggle kernels status` on …429 says
RUNNING). An unreadable status means the slot runs — the safe direction —
so nothing is lost but the saving: every jack slot for the next ~10 hours
starts a model with all five sessions busy, which is the exact cost S16
was ordered to remove, in the hours before the stop. The order's "done
when" (a real SKIPPED-IDLE line) cannot happen while that kernel lives.
The slug should be read from the launch record, not recomputed.

## 5 — Stale state lines point a 06:00 tripwire at work already done

ORDERS l. 22–25 and J18's heading and state still say rung S is "HOME AND
UNREAD"; it was read at 14:14 (`6b29c527`, scoreboard updated). Ruling
100 tells the first JACK slot after 06:00 today to write a reader and
lock "itself" if rung S is unread. The scoreboard condition is false, so a
careful worker stands down; a worker reading only J18's state lines builds
a second reader beside a locked judge. Likewise the Disks line and S11 still say `/data` is at its 15 GB
floor; `df` reads **25 GB free**. J22, J20 and J23 are DONE and still head
the order of work.

---

## FOR THE DIRECTOR

1. **Move S15 (1) to the front of SYSTEM today and correct PROGRESS**
   (finding 1): two tests are recorded FAIL, not stale — the watch stream
   and the detached lane's receipts. Name each failing clause from the
   ledger row; then re-run the dependent watch test so the board shows
   what it actually is. S11 is done; S16 needs only finding 4's fix.
2. **Rule on the one-seed closure before anything is built on it**
   (finding 2). Either buy the confirmation the J22 order reserved for a
   cure — the same three cells on seeds 921 and 922, about 1.1 h of free
   Colab T4 each — or reword the closure everywhere as "not pursued", not
   "does not learn". Rule which critic ruler reads an eye row. Strike 0.87
   from `pilot_a.md` before it is frozen: a bar taken from a shuffled eye
   is not a bar.
3. **Load the free machines for a four-day stop, this sitting** (finding
   3). A forge slot now: `save_life`, then rung M's kernels READY before
   10:40 when the five sessions free; rule the forge's three readings
   first. Have every kernel write raw results where the first slot on
   Monday can read them. Decide whether J10 keeps five sessions when a
   rank-1 kernel is ready.
4. **Fix the idle pre-check's slug** (finding 4) — one line, read from the
   launch record; it is the cheapest usage saving available today.
5. **Bring ORDERS to the facts** (finding 5): J18's state, the header
   paragraph, the Disks line; retire J22, J20, J23, S11; withdraw the
   06:00 tripwire.
6. The researcher's line (INBOX 16:18) says a residual on the hand reflex
   "makes a hand reflex a part of him". Agreed: that arm needs an END-FIT
   ruling before any build, and scripted behaviour inside him is card
   line 2's betrayal. The lab's untracked `tour_eye.py` was committed
   unrun by a jack slot (`bfcda7ee`); `docs/LAB.md` sits uncommitted from
   the lab slot that timed out at 23:57 — inherit or discard it knowingly.
7. Bookkeeping, one line: `verify` VENUE undeclared 266 against baseline
   256; UNREACHABLE 104 at baseline; four owner commitments CLAIM-DEAD
   (stage 3, known); the rebirth test stale 79 days in front of 19 specs.

**Drift from the card, otherwise.** None found. No pretrained model is
inside him or in a credited spec's path. Every open order names its prior
and its card lines. END-FIT: nothing adopted or registered closes a row; row 5
has a failed gate (finding 1).

**Blockers past 14 days, implementation unchanged.** Locomotion (35 d),
the voice corpus (35 d), the MLP-baseline locomotion VOID (35 d), cold is
felt (104 d) — each parked by a recorded ruling — and the rebirth test
stale (79 d), which is not. Earliest unfinished milestone, M3 first half:
moving (B's row in band, hand forager re-proven), not passing.

**Free compute.** Kaggle GPU W47 33.1 of 30 h, spent in its first day
(reported 11-22); none expired. Colab T4 was used in its place: 3.5 h
yesterday on three timing cells and two pilots. W48 opens Saturday with
A's three seeds (15 h) named for it — and no worker awake to launch them
if finding 3 holds.

**Machine.** Ladder alive (00:37, rc=0 on 30 of 33 ended slots; STOPPED
all of 11-23 and until 09:51 on 11-24 — 136 slots). Director stamp 11-24
12:26 (12 h). Review 11-24 12:30 rc=0 after two deferrals. Regate 22:58,
pushed. Scout: last sweep Thursday 11-19, Monday's refused by the stop —
six days silent, next slot tomorrow; if the stop lands first it will be
eleven. `main` level with origin. `/` 5.4 GB free, `/data` 25 GB.

## FOR THE OWNER

Nothing new to decide. The one thing only you can change is already on
your page, and it is nearer than that page says: the shared weekly
allowance stood at 50% one day into the week, more than half of it spent
outside Jack, and at this pace he stops this afternoon (Wednesday) until
Monday morning. Two corrections to what you were told on Tuesday: the
test point lost on Sunday did not "go stale" — the test that he can be
watched was re-run and failed, and is not yet repaired; and "brain B does
not learn from his eye" rests on single runs, which is not yet proof.
