# PROGRESS — for the owner, on a phone

Written 2026-11-11 22:05 UTC by the Director (Sunday-style FULL sitting, the
first under the new organisation). Rewritten every day; git keeps the old ones.

## What Jack can do today

Nothing new since 26 October — the agents were paused for 16 days and came
back today. What was already true:

- A small learning brain (`wm-latent`) learns to survive in the first world
  (W0), and carries what it learned across deaths. Four rival brains did not.
- A hand-scripted forager can live a full life there with needs, a diary and
  save-and-resume. **The learning brain has never been put in that seat.**
- The diary and learned skill are separate memories; wiping one leaves the
  other. (One of its test bars is being tightened — see below.)
- He cannot yet move well, hear with his own ear, or use words. The "language"
  passes on the books are a diary plus a ready-made language model speaking —
  I ruled today that those count as his parent's work, not his.

The test ladder reads **108 of 262** demonstrated. Unchanged today.

## Where he stands on the route (9 milestones)

1. **One whole life** with the learning brain in control — not started. Top order.
2. **A world and a curriculum that can teach** — deeper world half built; three
   world-rule tests red; no curriculum yet.
3. **Brain shape decided by contest** — no seat moved; two contests wait on GPU.
4. **A body that moves** — failing since August; the fix is queued behind 3.
5. **Every sense in one brain** — five senses in as summaries; no learned ear.
6. **Words** — nothing of his own yet.
7–9. Keeps learning / a living Jack / beyond the jungle — not started.

## What moved since yesterday

- The reorganisation went live: workers now read one page of ranked orders.
- A worker split an over-strict seed-noise test (`T1.08`) so it no longer
  blocks 45 others; the replacement (`T1.14`) is **running on a free Colab GPU
  right now** — no result yet.
- A worker shrank three megabyte-sized log files to a size agents can read.
- Today I: rewrote the orders; made "one whole life" the first order and told
  workers not to wait on anything for it; gave two orphaned world tests an
  owner; settled 8 stale decisions (4 by their pre-agreed defaults — none
  moved a pass mark); cleared 10 of 65 rows of old backlog.

## What did not move

- No test result landed. No part of Jack's code changed. Day 0 — judge from tomorrow.
- The first GPU job waited 70 minutes on Kaggle for a session that never
  existed, and our own meter charged 1.17 hours for it. A bug in our poller
  (it does not notice "no such job"). Repair ordered (S2); short jobs go to
  Colab meanwhile.

## What the workers are on next

1. `LF.03` — one whole life: the learning brain living, dying five times,
   remembering, saved and resumed mid-life. Runs on this box, needs no GPU.
2. Collect the `T1.14` result, then give it a stronger control.
3. `D1.1` — the rerun of "which controller moves the body", with a fair
   comparison this time; then retry "he can move".
4. `UB.10` — the contest on how unified his brain should be (stuck 59 days).
5. Fix the red world rules (heat balance, needs ledger, night shelter).
6. Finish the deeper world; then the curriculum contest.

Machine work: fix the GPU poller; make six passing tests hold on **every**
seed rather than on the average (I found six passes, including the
diary-vs-skill one, where the average clears the bar but one run may not —
bars unchanged, tests re-run); stop guard tools being red by design.

## The four machine numbers

| | |
|---|---|
| Points per worker run | 0 today (108 → 108; ~3 worker runs since restart) |
| Slots lost to budget | 0 since restart |
| Free GPU-hours expired | last week: all 30 (the pause). This week: 0 used of 30, ~3.5 days left |
| Oldest unrepaired blocker | 73 days (`W.1`, `W.2` — world heat and needs rules); now owned by order J6 |

## Dead or stuck

- **Usage meter cannot be trusted tonight.** It read 76% at 21:33 and 0–2% at
  21:53 inside the same week. One of those is false. The 95% stop acts on
  whatever it reads; I did not touch it. If the 76% was the true one, the
  agents may go dark before the meter resets on 13 Nov 04:00 UTC.
- **Kaggle** gave our job no session, cause unknown (S2 will find out).
- **Scout** (reads new papers) has not run since the pause; its page is 91 KB
  against a 12 KB cap and will be rewritten at its next run.
- The auditor's run was cut off by its clock but its report was complete.
- Pushes work. Disk: 6.1 GB free on the system disk (workers stop under 3).
- 55 old backlog rows and 6 stale process decisions remain; I take 10 and 3 a day.

## The honest paragraph

Closer to a creature, or just busier? This week: busier, in the right
direction. The ladder has not moved since 26 October and no life has been
lived by the learning brain. The week's biggest step is not a result — it is
that "one whole life" is now defined as a concrete test on the brain that
actually learns, and is the first thing every worker sees. The worst drift is
old and I am naming it: for two months the machine studied itself (its queue,
its meters, its own paperwork) while the creature stood still, and some
passes on the books were about a borrowed language model rather than him.
Judge the new set-up by one thing: whether `LF.03` exists by the weekend.

Prompts: I changed none today. All steering files are inside their caps
except the scout's page (above).

## FOR THE OWNER

Nothing needs a decision. One thing only you can check: your Kaggle account
ran several `next-l1-icsi-*` jobs today from another project. If those use
GPU, Jack's "30 free hours a week" is smaller than our meter assumes.
