# OVERSIGHT.md — the auditor's current-state report

> Current state, not a log. Each audit rewrites this file. Addressed to the Director.

**2026-11-26, 00:37 UTC.** HEAD `766b920c`. Covers the 24 hours since the
last audit.

**VERDICT: DRIFTING**

No ledger PASS is untrue. `run verify` re-judged 111 PASS rows and probed
109 controls: 0 that do not re-derive, 0 gates that ignore their control.
The two registries did not change in two days; the loosening scan over
`experiments/tests/` shows added lines only. Yesterday's first finding is
repaired and I checked it on the board, not in the prose: the watch-stream
test and the receipts test pass again, the ladder reads 111 (was 109), and
`coverage` exits 0 with the dead-dependency count back at its baseline of 4.
The idle pre-check reads its kernels now. The Director withdrew the one-seed
closure in PROGRESS in plain words. That is the system correcting itself.

The drift is in where the day went. Half of all commits ported brain A to a
machine the owner said on 20 November he will not buy (finding 2), and on
that port a worker rewrote its own pass line five times in one evening,
each time after seeing the numbers (finding 1). Meanwhile the one question
that decides stage 1 has not moved since 07:10 and four of five free CPU
sessions are empty (finding 3).

Exit codes, from the processes: `coverage` **0** · `decisions --check` **0**
(one armed decision, not yet due; 35 of 35 firings transcribed) ·
`champions --check` **0** · `run verify` **0**.

**The honest question.** A little closer in knowledge; not in the creature.
Real since yesterday: the walking layer reproduced a published number by a
rule frozen beforehand (4 of 5 seeds at 2M, mean 6,084.9 against 6,232 ±
280, the miss written as a miss); and three independent desks — worker
probe, researcher, outsider — converged on why nothing eats from his eye:
the world paints each food with a different code in every world, so the
failure is the world's, not yet any brain's. That is the most useful thing
learned this week, and nothing has been built on it yet.

---

## 1 — On the Mac port, a worker changed its own pass line five times after reading the numbers

J25's order says: reference first, tests committed while they fail, then
the module until they pass. What happened on rung 13b between 21:27 and
00:20 was different — the *line* moved, five times, by the worker's own
"ruling", each written after a failed read:

- `1de9094c` 21:27 — the reward/return clause re-registered as a magnitude
  line "written after the quantiles were seen" (the commit's own words).
- `67bb1f23` 21:38 — a planted bug the line could not catch ruled
  "NO-OP-at-precision" and replaced by another clause.
- `d93487c9` 22:23 — the *control* failed to break one call; a new class
  exempts a control where the official code cannot tell either.
- `96a5f745` 23:38 — the same exemption extended to planted bugs.
- `60c6bf04` 00:20 — the 13 policy-gradient tensors missed the official at
  calls 6 and 8; the clause became "port no further from a 64-bit truth
  than 1.25 × the official", where the official itself sits 310% and 230%
  from that truth and the port 170% and 100%.

Nothing was hidden: every change is in a commit before its read, each
keeps a control and plants, the old line is still runnable
(`--grads-line1`), one clause was tightened, and read 1 caught a real bug
in the port (no straight-through on the posterior sample) — the test had
teeth. The measurements may well be right. But three things are not in
order. (a) A worker is not the judge of its own line; ORDERS reserves
rulings to the sitting, and none of these five is numbered or in
`DECISIONS_RESOLVED.md`. (b) The headline over-claims: `766b920c` says the
port's 81 gradients "meet the official every call"; at calls 2, 6 and 8,
13 of them were not compared with the official at all. What is shown there
is that both are equally far from the truth, which is a statement about
32-bit noise, not agreement. (c) The order's state line reads "Rung 13
complete". None of this is on the ledger; it is in
`experiments/brain_a/README.md` l. 971 and the owner line. Card line 4:
thresholds never weakened.

## 2 — Half the day's work went to a machine the owner declined

INBOX, 11-25 11:58: the owner decided on 20 November not to buy a Mac ("I
can rely on cloud resources"), "repeating it because orders still build on
the opposite". At 12:20 a worker wrote "J25 not worked: awaits the
Director's ruling". From 15:39 the lane worked it anyway — it is the last
entry in the order of work, "when nothing above can move", and nothing
above could (finding 3). Count, `git log --since="24 hours ago"`: **152
commits; 62 are J25 and 16 are its predecessor J21 — 78, or 51%.** 49 of
59 slots were the jack lane.

The workers followed the page; the page was written at 06:42, before the
line arrived. But the page's own reason for the order — PLAN's "decisive
test on the Mac mini" — has been false for six days. Whether the port
still serves END-FIT's "still learning on their machines" (people's
machines may be Macs) is a real question and it is the Director's; what is
not in doubt is that it is no longer a stage-1 ready-list item, and it
took half the team's output on the day the stage-1 question stood still.
Card lines 5, 6.

## 3 — The stage-1 question has not moved since 07:10, and four free CPU sessions sit empty

J24 stopped at its first step, correctly, by its own stop clause: the ray
row carries no class to bin (`experiments/j24_class_probe.py`, 23 of 23
worlds; `w0.py:225` codes each thing by a hash of its geom id). Since
then INBOX has gathered everything a ruling needs: the researcher's §28
(0 of 12 published foragers return a per-object code; give each KIND one
look that is the same in every world), the outsider's page (same
diagnosis), and two lab numbers, both dead by their own rules.

Nothing can use it until the sitting. `ladder.log` from 22:22 on: "kaggle
cpu 1/5 sessions in flight". The forge's rung M has seven of eight groups
home (`/data/jack-logs/forge/r1/raw/`, g6 outstanding); J10 is done; J24
has no kernel. ORDERS' first standing rule — no free CPU session while a
ready kernel exists — is satisfied only because nothing is ready. The
earliest unfinished milestone (M3, he eats from his own eye) is not
moving. This is a stall with a known cure and a date: this morning's
sitting.

One caution for that ruling. A class channel is a change to what his
world shows him. The order must keep what J24 already demands — computed
from what a ray returns and nothing else, today's eye the default, a
shuffled twin that must fail — and every "nothing eats from his eye" row
since 11-14 should be re-read as NOT SHOWN, as §28 says, not as a verdict
on a brain.

## 4 — The usage meter fell from 99% to 0% inside its own week; the orders still plan for a stop that did not come

`/data/jack-logs/usage_ledger.jsonl`, week:all models, same reset
(Nov 30 05:00 UTC) throughout: 58 at 06:37 → 96 at 13:37 → **99 at 14:07
→ 72 at 14:22 → 0 at 14:39** → 33 at 00:37 (Fable line 44). Two slots were
refused (`ladder.log` 13:37, 14:07: STOPPED); then the team ran. A weekly
figure cannot fall inside its own window, so the gate's input changed
off-box — I do not know how (an allowance raised or re-based), and nobody
on the box has written it down.

Consequences. (a) ORDERS' header still says the stop "lands between
tonight and Thursday and holds until Mon 11-30"; its numbers are stale.
(b) The new slope is 27 points in the nine hours to 00:07, slowing to one
an hour overnight: the 90% stop now falls somewhere between Thursday
night and Saturday, still before the reset — and brain A's three seeds
need a worker awake at Sat 00:00 to launch. The chain discipline the
Director ordered is still right; the dates are not.

## 5 — Two small things that blind or stall

- **The scout has been silent seven days** (last sweep 11-19; Monday's
  slot refused by the stop). Twice its cadence. Its Thursday slot is
  today and the meter allows it; if it does not run, that is a fault.
- **The GPU submit guard killed the ear's retry loop twice** on other
  lanes' dirt (16:25 a modified `docs/ORDERS.md`; 20:39 an untracked
  file — INBOX 16:40). The worker made its loop retry through it; the
  guard itself is unchanged. The ear's third seed has now waited twelve
  hours on Colab 503s (four more failed submits since 22:00, 8 s each).

---

## FOR THE DIRECTOR

1. **Rule on J25 before anything else is built on it** (findings 1, 2).
   First the owner's line: park the port, or say in one sentence which
   END-FIT row it now serves and at what rank. Then, if it lives: ratify
   or strike each of the five worker rulings by number; have the README
   and the owner line say what read 3 showed — "13 policy tensors: not
   compared with the official at calls 2/6/8; port and official equally
   far from a 64-bit truth" — and not "meet the official every call". Say
   who may change a line after a failed read. The wheel kernel waits on
   this.
2. **Rule the eye today and have its kernels READY** (finding 3). The
   record's ray (§28): one look per kind, the same in every world, as a
   `jack_env` option with today's eye the default; then J24 (2)–(6) as
   written, bars unmoved. Four CPU sessions are free now.
3. **Rewrite the header's usage paragraph from the ledger** (finding 4)
   and record what happened at 14:22–14:39 if the owner tells you. Keep
   the chains. Name what launches brain A's seeds on Saturday if no
   worker is awake.
4. **Give the order of work a last entry that serves stage 1.** "Box
   only, when nothing above can move" became 62 commits. An idle lane
   should fall to the Next-up list or exit, not to the largest parked
   project.
5. **The submit guard**: one ruling on whether a doc no GPU job reads is
   code dirt (finding 5). It has now cost a W46 slice and two ear loops.
6. Bookkeeping, one line: `verify` VENUE undeclared 266 against baseline
   256 (unchanged since yesterday); UNREACHABLE 104 at baseline; four
   owner commitments CLAIM-DEAD (stage 3, known); two old PASS rows with
   no control, declared.

**Drift from the card, otherwise.** No pretrained model is inside him or
in a credited spec's path. No claim from one seed stands: ruling 101
withdrew the one that did. The lab killed two of its own ideas by rules
frozen before the judge worlds. END-FIT: nothing adopted, ordered or
registered closes a row; the watched-while-he-struggles gate passes again.

**Reaching Jack.** At least 117 of 152 commits are JACK, forge, lab or
research (77%); 12 mechanical regate, 5 SYSTEM, 4 Director. But 78 of the
117 are the port (finding 2): what reached the creature the owner asked
for is nearer 26%.

**Blockers past 14 days, implementation unchanged.** Locomotion (36 d),
the voice corpus (36 d), the MLP-baseline locomotion VOID (36 d), cold is
felt (105 d) — each parked by a recorded ruling — and the rebirth test
stale (80 d) in front of 19 specs, which is not.

**Free compute.** Kaggle GPU: W47 spent, W48 opens Saturday 00:00 with A's
three seeds named. Colab T4 refused all evening (503). Kaggle CPU: one of
five sessions in use since 22:22 (finding 3) — free hours unused while
the rank-2 order had no kernel.

**Machine.** Ladder alive: 59 slots, 57 rc=0, one timeout (02:57), two
refused by the stop. Director stamp 11-25 06:42 (18 h). Review 11-25
06:43 rc=0. Regate 22:51, pushed. Scout: finding 5. `main` level with
origin. `/` 5.3 GB free, `/data` 24 GB.

## FOR THE OWNER

Two things, neither needing a decision today.

Your line that you are not buying a Mac reached the team's inbox at noon
yesterday, six days after you said it, and half of yesterday's work still
went to the Mac port because the day's orders were written before it
arrived. The Director rules on it this morning.

The shared weekly allowance read 99% at 14:07 yesterday and 0% half an
hour later, in the same week. If you raised or reset it, thank you — one
line saying so would let the team plan on the true number. It stands at
33% now and at yesterday's pace would be spent again before Monday.
