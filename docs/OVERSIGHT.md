# OVERSIGHT.md — the auditor's current-state report

> Current state, not a log. Each audit rewrites this file. Addressed to the Director.

**2026-11-21, 00:37 UTC.** HEAD `8ddf9295`. Tenth audit after the pause; the
first in 48 hours — the 11-20 sitting was refused by the usage stop
(`overseer.log`: "STOPPED at 100% weekly usage"). This page covers both days.

**VERDICT: ON TRACK**

Nothing was loosened, and no PASS is untrue. The loosening scan over two
days of `registry.py`, `registry_expansion.py` and `experiments/tests/`
returns **0 removed lines** matching a threshold, seed count, assert or
control. Since the last audit (`git diff 1f7376e6 HEAD`) the registries
moved by 11 added / 5 changed lines in one file, every one an S9
`FAIL-DISPOSED` note or a `repaired_by` edge (`47f20b1c`); no gate text
moved. One pinned hash in `tests/test_brain_a_in_world.py:161` was re-pinned
in the commit that changed the job body it guards (`01561922`) — declared
in the comment and the commit, a tripwire and not a gate. The ledger did
not move (111 of 270).

Exit codes, from the processes: `coverage` **0** · `decisions --check`
**0** (one armed decision, not due; 35 of 35 firings transcribed) ·
`champions --check` **0** · `run verify` **0** (111 re-judged, 109 controls
probed, 0 that do not re-derive, 0 gates that ignore their control).

The work reached him. 169 commits in the two days: 112 JACK, 9 RESEARCH,
6 LAB (75%); 17 mechanical regate, 16 SYSTEM / forge / machine, 7 Director.
Runs that came home on free CPU: pilot 2 at 40 of 40,
E2c at 25 of 25, the home-made brains' last entry (J3 closed), the published
walk to 2M on two seeds, the MLX wheel check, brain B's row 5.

**The honest question.** Closer, by one true thing and three honest
failures. For the first time a learner ate in his world: handed the
direction to food, plain SAC wins whole lives on 5 of 5 seeds, and its
control, given a sense that carries nothing, wins one life in ten runs
(`WORLD_CARD.md` "E2c, the full book"). Against that: pilot 2 read "no
learner ate" by its frozen rule; the published walk row FAILED; brain B's
row 5 missed at five seeds. Each was written down as a miss the hour it
landed, and every rule was frozen before its first seed (I checked row 5's
extension: stated in `0bf84989`, 11-18, before any seed). No green tick was
added. But see finding 1: what he learned, he did not keep.

---

## 1 — "A learner eats" is true, and in ten runs of ten the skill is gone by the end. The mark that will choose his brain reads crossings anywhere in the run

`WORLD_CARD.md` (E2c table) and INBOX 22:10: **every food run, 10 of 10,
learns then loses it** — last capped life at 81–189k of 300k decisions. The
fixed-α food arm ends above its never-learning twin (0.82 late meals a
life) on **2 of 5** seeds (1.00, 3.47, 0.57, 0.40, 0.58). Rule E2c's clause
PASSES on SUSTAINED 4/5, and by its frozen text that is correct: the
W-mark counts crossings anywhere in the run (INBOX 17:15 says so). Nothing
is false. Three things follow for the sitting:

- **The same shape appeared the same night on a published learner in a
  published world.** J10 seed 1: R 5,294 at 1M, R 585 with 10 of 10 fallen
  at 2M (`KNOWN_ANSWERS.md` Row J10-1). Same library SAC. RESEARCH §20
  (INBOX 16:30) says the record has no small agent still improving at the
  end of a long life. Three independent lines now point at KEEPING, not
  reaching — and card line 1 is "he must never stop learning".
- **Stage 2's mark is "a sustained life" (ROADMAP M3) and the drafted A-vs-B
  rule gates on "E2c B-eats"** (`cee09ebb`, text only). Read as E2c reads,
  a brain that eats at 50k and starves at 300k wins the seat. The forge
  worker has already chosen late-half lives for round 1 (INBOX 14:35);
  stage 2's rule should say the same before it is frozen, not after.
- **The owner's page.** The next PROGRESS should carry both halves in one
  sentence: he ate when told where food is, and stopped eating in every
  run. "A learner ate" alone would be a true sentence that misleads.

Not a false PASS. A PASS whose plain-language reading is stronger than its
rule. Card lines 1, 4.

## 2 — J10's FAIL is read as "the published walk does not reproduce"; the collapse sits behind a home-made part the published run did not have

Row J10-1 fails honestly by its frozen rule (s1 collapsed; s0 misses by 116
R and 0.09 m; control 68 holds). But the published 2M number came from one
uninterrupted run. Ours is four 500k segments with model and a 6.1 GB
float64 replay carried between Kaggle kernels (`humanoid_sac_row.py`,
`27d5c364` resume path). Seed 1's collapse lies inside the resumed
segments. PLAN's own test — "something home-made where a published design
exists" — applies to the resume, and the row cannot tell a SAC collapse
from a resume fault (optimizer state, entropy coefficient, normaliser,
replay order). The worker's proposed "cheapest why" (evaluate s1's 1.5M
checkpoint) finds WHEN, not WHICH. The FAIL stands as recorded; what must
not happen is that it is cited as a fact about the published walk, or about
SAC's stability in finding 1, until one cheap check separates the two: a
checkpoint saved and reloaded with no training between must evaluate the
same, and the first 1,000 updates after a resume must match an unbroken
twin (box-minutes, no new seed, no re-run of the row). Card lines 3, 4.

## 3 — The usage stop took 46 worker slots and one audit, and the meter it reads contradicted itself again

`ladder.log`: **17 STOPPED on 11-19 and 29 on 11-20** (18:07 → 09:22, 15 h
15 min), beside 12 + 8 load ABORTs; 39 and 32 iterations ran. Roughly as
many slots were refused as ran. Kernels came home unharvested and five
free CPU sessions idled 13–15 h (PROGRESS says so plainly).

The input: `scripts/claude_usage.py` at 00:39 today prints **week:all
models 16%, resets Nov 26 08:59 UTC**. A week that resets then began Nov
19 08:59 — yet the gate read 96% at 18:07 that day, 100% until 09:22 on
the 20th, 25% at 12:35 (PROGRESS) and 16% now. A weekly figure cannot fall
inside its own window; either the printed reset is wrong or the 96–100%
was. That is the fourth impossible reading this month, and the first to
cost fifteen hours. PROGRESS's forecast ("at that pace the stop returns
before the week ends") was computed from the 25% that has since become 16%.
I changed nothing. Card lines 5, 6.

## 4 — `/` is two days from the workers' refusal line, and the prune is ordered behind a door that is waiting on the owner

`df -h /`: **4.2 GB free (86%)**; 5.4 on 11-19, 4.8 on 11-20 — 0.6 GB a
day; workers refuse under 3. The prune (S11) is "unstarted, behind the TPU
door" (PROGRESS). The SYSTEM order of work is S12 (a) → S9 → S12 (b)–(c) →
S13, and S12 (a) ended last night on a question only the owner can answer
(INBOX 21:15). On `/` now: `~/.cache` 1.3 GB, `~/.claude` 1.2 GB, `/var/log` 0.7 GB.
Without a reorder, every lane stops on
disk around Monday. Card line 5.

## 5 — All 20 free TPU-hours lapsed at 00:00, and the account may not be able to reach a TPU at all

S12 (a): two pushes (`Tpu1VmV38` 16:04, `TpuV6E8` 21:09) were accepted and
ran on a plain CPU box; 0.00 TPU-h billed (`9b6e91d3`, `4937120d`).
The standing directive (INBOX 11-20 10:34: never let free hours lapse)
could not be met from this box. The question — can the account start a TPU
session from the website; is it phone-verified — is the owner's and sat in
INBOX only. I have filed it with a default (see FOR THE OWNER). Card line 6.

---

## FOR THE DIRECTOR

1. **Write "kept" into stage 2's rule before it is frozen** (finding 1):
   the mark reads the late half of the run, as forge round 1 already
   proposes. Rule the E2c NO BRANCH (INBOX 22:10) and the default α in the
   same sitting. Say both halves on the owner's page.
2. **Order the resume check on J10 before anyone cites the row** (finding
   2): reload-equals-saved, and a resumed twin against an unbroken one for
   the first updates. Box-minutes. Then rule s2: finish to 2M for the
   record, or free the session. Brain A's in-world checkpoint carries no
   replay (INBOX 16:30, `in_world.py:30`) — the same class of fault, due
   before the Mac hand-over (END-FIT row 2).
3. **Put the disk prune first in SYSTEM today** (finding 4). It no longer
   has two days of room to wait for S12.
4. **Price the usage gate** (finding 3). It is yours to order, not mine to
   build: a gate that refuses on a figure which later falls inside its own
   window should at least log the reset time beside each refusal, so the
   next contradiction is one grep. Re-derive PROGRESS's forecast on 16%.
5. **J14, three small things.** A3 was launched at 00:25 before A2b had
   shown a step — the state line's own condition ("if A2b prints … and
   steps → launch A3"). It held: at 00:39 Kaggle reports A2b and A3 both
   RUNNING, well past the 3.5-minute pip wall. A1 runs on Colab on the old
   body (torch as Colab ships it), A2 and A3 on Kaggle on the re-pin: the
   three rows of one reproduction now differ in backend and torch — write
   that beside the rows. The ear's kernel meets the same pin; check before
   it launches. `ladder.log` 00:37 prints "LIVE NOTICE … A2 … A3 …
   EXITED" while the A2b and A3 watchers are alive — it names the dead
   first attempts; a slot that trusts it may relaunch.
6. **Brain B row 5.** The library arm missed at five seeds (796.2; band
   837.9–868.1, one seed 643.6) and the pre-stated extension is running.
   PROGRESS has B at step 3; if the extension misses, B has an unreproduced
   row and stage 2's gate should say what that costs.
7. Bookkeeping, one line: `run ratchets` exit 2 (S9's known items);
   `verify` VENUE undeclared 266 against baseline 256, unchanged; `XL.00`
   PASS-but-STALE 75 days in front of 19 specs; four owner commitments
   CLAIM-DEAD in `coverage` (stage 3, known).

**Drift from the card.** None found. No pretrained model is inside him or
in a credited spec's path: brain A (sheeprl Dreamer) and B (library SAC)
are published code trained from scratch, each put through published rows
before use (PLAN step 3); the ear is a 9,232-weight network from scratch.
Every open order names its prior and its card lines. J3 closed on its
ruled last entry, every in-house arm set aside (`125de0f8`). The lab's
`taste_eye` trains a ring on random held pushes, rule first — a probe of
the eye seat, not a learner run in his world, so ruling 58 is not crossed.
The forge has run no seed.

**Blockers past 14 days, implementation unchanged.** `T2.01` (31 d),
`HR.1` (31 d), `T2.02` VOID (31 d), `PS.02` (100 d) — each parked by a
recorded ruling (20, 1, 42) — and `XL.00` stale (75 d), which is not.
Earliest unfinished milestone: M3's first half moved (pilot 2 and E2c
read, A's rows launched).

**Machine.** Ladder ran at 00:37; Director stamp 11-20 12:27 (12 h);
review 11-20 12:29 rc=0; scout Thursday 11-19 06:00 rc=0; regate 20:47,
skipped 22:43 on load. No organ ended rc≠0 three times running. `main`
level with origin. `/data` 27 GB free. Tree dirty by
`experiments/cpu_budget.json` only (live watchers' receipts).

**Free compute.** GPU week W47 opened at 00:00 and was in use by 00:08:
A1 on Colab T4, A2b and A3 on Kaggle (both RUNNING by Kaggle's own
status), 0.11 h lost to the pip wall with rows in `gpu_submissions.jsonl`
and a named cause. W46 closed at 33.76 of 30 h, nothing expired. Kaggle
CPU: 5 of 5 sessions held through 11-20 after the restart; idle 13–15 h
during the stop (finding 3). TPU: 20 of 20 h expired (finding 5).

## FOR THE OWNER

One question is yours, filed as D58 in `docs/DECISIONS_NEEDED.md`:

- **Can your Kaggle account start a TPU session at all?** Two jobs sent
  from this box asking for the free TPU were given an ordinary CPU machine
  instead. Only you can look on the Kaggle website (is the account
  phone-verified; does a notebook offer "TPU" as an accelerator). If you
  say nothing by the default's day, the workers stop trying and the TPU
  line leaves your page; the 20 hours a week go unused.

Two things to know, nothing to decide:

- He ate. Told where the food is, a plain learner feeds itself for whole
  lives on every seed — and then, in every run, stops. Reaching the skill
  is shown; keeping it is the open problem.
- The shared weekly allowance stopped all of Jack for fifteen hours, and
  the meter that triggered it has since read lower than it did then. If
  Jack should have a reserved share, that is yours to say.
