# META_AUDIT — the Claude machine audited (2026-08-09)

The four organs audit Jack's repo. This audits the machine itself: loop,
organs, prompts, orchestrator. Method: every finding verified by command
before assertion; one suspicion was refuted by its own check and is reported
as such.

## Verdict: SOUND, with five confirmed defects (all fixed today) and two
## structural risks (one mitigated, one escalated to the owner)

| # | finding | evidence | action |
|---|---|---|---|
| 1 | REFUTED: "builder avoids hard frontier work" | PG.8 implemented & PASSING — Jack is in the playground | none; suspicion withdrawn |
| 2 | Pass rate fell while count rose (40.0%→38.3%) | ledger + git | Review now runs an explicit Goodhart check weekly |
| 3 | Prompt cached stale state + contradictory priorities, twice | grep | prompts no longer cache facts; lesson recorded |
| 4 | Newest 2 organs had no load/disk guards on a tenant box | grep -c = 0 | guards added, same thresholds as overseer |
| 5 | The machine's own schedule (crontab) was not in git | git ls-files | scripts/crontab.txt, committed |
| 6 | Credits unmetered; 4 exhaustion events in one day | ladder.log | escalated: DECISIONS_NEEDED; audit.py (zero-credit checks) queued |
| 7 | Organ boundaries enforced by prose, not mechanism | SYSTEM_DESIGN P1-6/P1-7 unimplemented | audit.py queued to builder; worktree containment remains open |
| 8 | Constitution accretes contradictions (curiosity vs needs) | GOAL.md text | reconciled; Review now scans constitution coherence weekly |

## The residual risk that cannot be closed from inside

Nobody audits the orchestrator (the conversation-level Claude) in real time.
Mitigations, honest but partial: every decision lands in git where the
overseer and Review audit it after the fact; owner directives now enter with
their recorded counterargument (SYSTEM.md norm); the owner reads this file.
The orchestrator's claims to the owner remain unverified at the moment of
utterance — which is why this machine's rule stands: trust the ledger over
any narrative, including this one.
