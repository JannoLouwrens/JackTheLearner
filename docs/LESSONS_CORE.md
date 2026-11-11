# LESSONS_CORE — the rules, short enough to actually be read

`docs/LESSONS.md` is 1.4 MB. Nothing can read it at the start of an iteration,
so for two months it was written to and not learned from — a lessons file
nobody can load is a diary. **This page is the part that binds.** It is capped
at 15 KB; adding a rule means merging or cutting another. The long file stays
as the archive — `grep` it when you hit something that smells familiar.

## About evidence

1. **A capability is claimed only by a PASS in `experiments/ledger.json`.** Not
   a loss curve, not a docstring, not "it ran". Only the runner writes the
   ledger.
2. **A control that also passes means the test measures nothing.** Every spec
   carries a control that must FAIL. When a control fails alongside the
   experiment, that localises the bug — it is a gift.
3. **VOID is not FAIL.** A run that could not test the claim refutes nothing.
4. **A PASS carries the world it was earned in.** `DP.00` passed in a 12×12
   gridworld and was quoted for a month as a fact about Jack's world.
5. **Never one seed.** Report mean and spread; a number from a single run is an
   anecdote. And size the sample to the claim: a 3-seed estimate of a
   coefficient of variation cannot certify a 7% bar (a process sitting exactly
   on the bar passes only ~63% of the time).
6. **Never weaken a pre-registered threshold, loosen a control, or delete a
   failing test.** If the *experiment* is wrong, strengthen or re-register it
   and say why in the commit; the old version stays in history.
7. **Pilot before you pay.** Size a multi-hour run from a measured probe, and
   declare the thresholds before the registered run, not after.

## The disease this project keeps catching: a control that reads a proxy

Directory mtime for "in use". A shared log's tail for "this run". A spec's
title for "covers". A header's tone for "closed". A dict key for "one
decision". Each reported success while watching the wrong thing.

8. **Check the thing itself.** If a check reads something *about* the thing,
   ask what happens when the two disagree.
9. **Every audit tool carries a known-positive it must catch**, through the
   real code path. A checker that has never been seen firing is untested.
10. **Declarations beat inference.** `COVERS:`, `DECIDE:`, `IMPL_DEPS` — parse
    what was declared at the start of a line, never guess from prose.
11. **Unknown is not zero.** A limit that cannot read its meter refuses to run;
    a smoothing heuristic that cannot read its input steps aside.

## About sharing a box and a tree

12. **Bound yourself to your own writes.** Explicit paths in `git add`, never
    `git add -A` — other organs and the owner work in this tree. `ps` for a
    live peer before committing. Never `pkill -f <pattern>`: the pattern
    matches your own shell.
13. **This box serves paying customers.** Never restart the Docker daemon;
    `nice 19`; ≤ 2 threads; stay under ~1.5 GB RAM; refuse to start below 3 GB
    free on `/`; never take `/data` below 15 GB free.
14. **Durable work must not report through an ephemeral reporter.** Multi-hour
    runs go through `scripts/launch_detached.sh` / `scripts/dispatch.sh`; their
    liveness is worker CPU time, not log bytes. Stream logs — a log that only
    appears on success cannot report a failure.
15. **GPU backends clone from GitHub.** Push before you dispatch.

## About deciding

16. **A measurement chooses among architectures; it never chooses the ends.**
    ENDS (what Jack must become) are the owner's. ARCHITECTURE (how he is
    built) is always raced, never assumed — no seat without a registered,
    existing challenger. CONDUCT (how the organs work) the desks amend
    themselves and report.
17. **A fork whose arms can both be run is an experiment, not an escalation.**
    Only what a measurement cannot answer goes to the owner, and then with a
    default and a deadline so silence cannot deadlock it.
18. **An assumption that cannot lose is not a finding.** An excluded
    alternative, an alternative cast as the must-fail control, a property test
    standing in for a contest, a falsifier that was never registered — all four
    happened here.

## About the organisation — the lessons of August to October 2026

These are measured, not opinions (`docs/research/BOTTLENECK_AUDIT_2026-11.md`).

19. **A verdict is the middle of the loop, not the end.** RESEARCH → TEST →
    IMPLEMENT → TEST → **FIX**. By October every runnable spec already had a
    verdict and *every* terminal blocker's implementation had been untouched
    for 22–89 days. The ladder ran out of things to test and was full of things
    to fix, and the organisation only knew how to test. **A FAIL you leave
    behind is not a result, it is a debt.**
20. **Repair before you register.** 76 of the 118 first-ever passes came in the
    first two weeks; the next ten weeks added 42 while the registry grew to 261
    and governance code to ~20,000 lines against ~1,600 lines of creature.
21. **System work is bounded by the clock, not by judgment.** Left to judgment
    it went from 34% of commits to ~78%. The system lane gets its cron slots
    and no others.
22. **Free compute that expires is the cheapest thing we own.** ~200 free
    Kaggle GPU-hours expired unused in seven weeks. If a GPU-able order exists
    and the week ends with hours left, that is a failure to report.
23. **Steering must stay small enough to obey.** Orders live in
    `docs/ORDERS.md`, bounded. History lives in git. A prompt that narrates its
    own past is 29,000 tokens of reasons not to start.
24. **Rigor must not eat the answer.** `D1.0` measured the control question
    cleanly (small plastic head 506, the 57M trunk 344–350, paired t = 20
    against its own untrained twin) and was recorded VOID over the wrong null.
    When a VOID's data plainly answers the question, re-register with the right
    null and run it again — do not leave the answer in the bin.
