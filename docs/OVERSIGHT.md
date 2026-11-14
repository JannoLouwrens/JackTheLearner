# OVERSIGHT.md — the auditor's current-state report

> Current state, not a log. Each audit rewrites this file. Addressed to the Director.

**2026-11-14, 00:37 UTC.** HEAD `314ad46`. Fourth audit after the pause.

**VERDICT: DRIFTING**

Nothing was loosened. The scan over registry and tests for the last two days
shows additions and strengthenings only: `LC.02`'s contention precondition
(VOID, never PASS, on a loud box; planted-load meter), `LC.09`'s collapse guard
gaining a third disqualifying clause before any run, `CR.01` and `LF.06`
registered with controls that must fail. `LC.02`'s edit last night moved no
bar and no status line.

The drift is the same as yesterday, one day older: the test that asks whether
living teaches him (`LF.04`) has still never run, and it is now blocked from
two sides. Meanwhile the certificate everything under it needs was passing at
10:29 and is not passing now, because of our own edit.

Exit codes, from the processes: `coverage` **2** · `decisions --check` **0** ·
`champions --check` **0** · `run verify` **2** (105 re-judged, 1 does not
re-derive).

---

## 1 — `W0.PL`'s PASS still cannot be re-derived (second day)

`run verify`: `verdicts that no longer re-derive 1 W0.PL(STATUS:VOID)`. The
row is unchanged (attempt 2, `d05c997`, issued 11-12 19:15). The cause is the
one reported yesterday: the gate judges from lists held in process memory, and
the row stores the mean of per-seed minima while the gate is on the minimum.
The repair is ordered (`docs/ORDERS.md:222-229`) and its state line reads
"Open … needs a quiet-box W0.PL re-run" (`ORDERS.md:243`). `BD.01` has the
same gate shape and still no row. Until the repair lands, this is a PASS on
the ledger that the record alone cannot support.

## 2 — The creature gate is blocked twice, and milestone 1 reads MET on a stale certificate

`run blocked`: `LC.02 = VOID … blocks 22` (including `LF.04`, `LF.06`,
`LC.09`) and `T2.10 = FAIL … blocks 4` (including `LF.04`, `LF.06`).

- **`LC.02`.** It passed on a quiet box at 10:29 on 11-13 (`2ceaaef`,
  foreign load .75 cores). At 22:39 commit `4969084` edited its test file so
  it could also time `LC.09`'s two recurrent challengers; the worker's own
  summary says "LC.02 is now stale and needs one re-run". The re-run at 00:08
  came back VOID (`3f3e121`): foreign load 1.92 cores against a 1.0 limit,
  from another project's speech job and a root monitor. The ladder count went
  106 → 105. That edit served J3, the last order in the Director's ranking;
  it undid step 1 of J1, the first. The alternative — a Director ruling on the
  two arms' training ratio — is named in the same worker's summary.
- **The foreign job is not ours to stop.** At 00:38 `ps` shows
  `/data/stt/live-cookoff/…/proof_r10.py` at 92% of a core, 55 minutes old.
  `LC.02` now passes only when another project is idle. Under that load the
  seated brain itself read 4.97 against the 5.0 floor (2 of 3 seeds).
- **`T2.10`.** `LF.04 ← LF.03 ← LF.02 ← T6.03 ← T2.10 FAIL`. The workers asked
  for a ruling in INBOX at 14:25 and again at 19:55 (`ORDERS.md:276`, `:286`);
  the Director last sat at 08:45, so it is unanswered.
- **Milestone 1.** `docs/ROADMAP.md:25` says MET on `LF.03` PASS. `run stale`
  lists `LF.03`, `LF.02` and `XL.00` as PASS-but-stale, and `coverage` lists
  `LF.03 ← LC.02 VOID` and `LF.02 ← T6.03 BLOCKED` among 12
  pass-on-dead-dependency pairs (baseline 6). `LF.03` cannot be re-bought
  until both blockers clear. PROGRESS tells the owner M1 is met with no caveat.

`LF.04` was registered 11-12 08:40. It is about 64 hours unrun.

## 3 — The CPU day overran again and the clerk recorded nothing for six hours

`experiments/cpu_budget.json`, day 11-13: 59,599 s used against the 57,600 s
ceiling by 15:37, the overrun from `detached:needs_clock_race.log` (J6's
ordered race). From then to midnight every registered run was refused: J1's
`XL.00` re-run (`ORDERS.md:285`), and the clerk's sweeps at 16:43, 20:43 and
22:43 each "ran but RECORDED NOTHING" on the same eight specs
(`regate.log:4661`, `:4769`, `:4828`). Stale cheap certificates: 19 at
yesterday's audit, 32 now. I did not see the first sweep of the new day; the
next audit should read whether those eight cleared.

Today started the same way. At 00:11 the heat race went up detached with two
workers; at 00:15 `W1.02` added three, and load reached 6.7 on a 4-core box
that serves paying tenants (the worker killed its own run after 10 minutes,
`ORDERS.md` J7 state). `LC.02`'s run also peaked at 2,053 MB against the
1,536 MB ceiling (`ladder.log`, 00:26). Load was 4.1 at 00:37.

## 4 — Last week's free Kaggle GPU went entirely to another project

`experiments/gpu_budget.json`, week W45: Kaggle 0.00 h for Jack, 15 dispatches
refused; Colab 7.15 h in 16 jobs, 7 of them failed. The refund notes give the
cause: Kaggle answered "Maximum weekly GPU quota of 30.00 hours reached" and
the account is shared with kernels named `next-l1-icsi-*`. `T1.14` (blocks 46)
and `D1.0` waited through it. The quota reset at 00:00 today; `T1.14` is
RUNNING — I checked Kaggle itself, kernel `jack-ladder-1794615228`. `D1.0`
needs 25.05 of the 30 hours, so one busy week on the other project stalls it
again. This is the owner's to split; see FOR THE OWNER.

One more thing the budget file gets wrong: `gpu.py:495` keys weeks from
Sunday, but Kaggle reset on Saturday 00:00 UTC, so today's `T1.14` hours are
being filed under the week that was refused.

## 5 — Where the work went

136 commits in 24 hours; 43 touched experiment code or tests and 19 the
ledger; 4 touched scripts or the runner. That is far from October's 5%. But
inside the Jack lane the largest share (about 26 commits) went to J3 — arms
for `LC.09`, which cannot run until `LC.02` passes — because J1 was waiting on
a ruling and the CPU day was spent. The lab's sense-lookahead pilot ran on
Kaggle CPU with a rule declared first and a shuffled control; it died by that
rule (two seeds, a kill, not a claim). That is honest work and it used no box
CPU.

**The honest question.** Slightly closer in knowledge: the world's heat
balance fault is found and being raced, and the hand-over file is measured
(19.6 MB). Not closer in evidence: nothing yet shows he improves by living,
and the ladder went down by one.

---

## FOR THE DIRECTOR

1. **Rule on `T2.10` today**, on the record. If `LF.02`'s dependency on a
   language-memory paraphrase claim is wrong, cut it by a written ruling; if
   it is right, order the `T2.10` repair first. Either way J1's chain needs an
   answer the workers cannot give themselves.
2. **Rule on `LC.02`'s venue.** A wall-clock claim that passes only when
   another project is idle will keep flipping. Options I can see: time it on
   Kaggle CPU through S1, or keep the 10:29 quiet-box verdict as the standing
   one for unchanged code. And say that a passing certificate on J1's path is
   not edited for a lower-ranked order without your ruling.
3. **Caveat milestone 1** in ROADMAP and PROGRESS until `LF.03` is re-bought.
4. **Split the CPU day** so a registered run and the clerk are never refused
   by a detached race; cap detached workers so two launches cannot take the
   box to load 6.7. Both days' overruns came from detached jobs.
5. **Land the `W0.PL` / `BD.01` gate repair** already ordered — it needs the
   same quiet minute as `LC.02`, so sequence them together.
6. Bookkeeping, one line: UNREACHABLE reads 118 against 115, venue-undeclared
   266 against 256, three unrunnable-baseline entries live again; champions
   prints the END-FIT table as malformed rows 51–57; the GPU week key starts
   on the wrong day.

## FOR THE OWNER

One item. The free Kaggle GPU week is shared between Jack and your other
project, and last week the other project used all 30 hours; Jack's GPU work
waited three days. The split is yours. Default if you say nothing, by
2026-11-21: no split — Jack runs its smallest jobs first, accepts a refusal,
and overflows to Colab. Filed as the newest entry in
`docs/DECISIONS_NEEDED.md`.

Philosophy served: 4 (nothing faked), 5 (the loop), 6 (limits).
