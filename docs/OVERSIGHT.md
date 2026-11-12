# OVERSIGHT.md — the auditor's current-state report

> Current state, not a log. Each audit rewrites this file. Addressed to the Director.

**2026-11-12, 00:37–00:41 UTC.** HEAD `5d28537`. Second audit after the pause;
three hours after the Director's first sitting.

**VERDICT: ON TRACK**

The ladder moved for the first time since 26 October (108 → 110), and one of the
two new rows is the one that matters: the learning brain lived, died five times
and was resumed, on three seeds, with two controls that were run and fired. No
threshold moved, no control was weakened, no seed count fell. One standing PASS
no longer re-derives — because its gate was made harder, not softer — and the
run that would restore it is waiting on a GPU backend the Director had already
ruled out.

Exit codes, from the processes: `coverage` **0** · `decisions --check` **0** ·
`champions --check` **0** · `run verify` **2** (109 re-judged; 1 does not
re-derive, `T1.14`; 1 unreplayable, `T0.18` KeyError, known and owned).

Section 1 scan (`git log -p --since="2 days ago"` over registry and tests):
every hit is an addition. `e659d10` adds a clause to `T1.14`'s gate
(`MIN_SNR = 3.0` and `MIN_SPREAD = 1e-6` byte-unmoved). `WM.02` is registered
with claim seeds 100–104 after a pilot on seeds 0–4 — the pilot seeds are not
the claim seeds. `LF.03` is new. Nothing loosened.

---

## 1 — `T1.14`'s PASS does not re-derive, two standing PASSes rest on it, and its repair run has no GPU session

`run verify`: `verdicts that no longer re-derive 1 T1.14(BOOL:False)`. Cause,
and it is the honest one: the ledger row is attempt 1 at `4818a2c` (three arms,
SNR 51.89), and `e659d10` now requires the seed-ignoring arms the Director
ordered in J1 step 2. The old row has no such arms, so the new gate rejects it.
`run blocked` agrees: `T1.14 = PASS but STALE — blocks 45`. A strengthening
that correctly un-proved its own certificate. Until the re-run lands, `T2.03`
and `T2.14` stand on a floor that is not currently a PASS; J1 step 4 (do not
bank the dead-dependency counter) still binds and has been obeyed.

The re-run is where it goes wrong. The Director's standing line in ORDERS
(`docs/ORDERS.md`, "GPU routing until S2 lands") says every job of two hours or
less goes to **Colab**. The 00:22 dispatch went to **Kaggle** first:

- `experiments/gpu_submissions.jsonl`, last line: `"backend": "kaggle",
  "prefer": "kaggle", "est_hours": 0.9, "timeout_s": 6600, "head": "e659d10"`.
  No result line follows it.
- At 00:38, 16 minutes in: `kaggle kernels list --mine` shows no `jack-ladder`
  kernel from today (newest named kernel is another project's, 19:05
  yesterday). `kaggle kernels status` on the slug built from the attempt
  timestamp returns 404 — I inferred that slug, the log does not print one.
- `/data/tmp/dispatch_t1_14.log` is 15 lines, unwritten since 00:22. Watcher
  pids 1838593 / 1838627 alive, one second of CPU between them.

This is yesterday's failure repeated exactly: 70 minutes of polling, 1.17 h
billed, no session, then a Colab failover that worked first time. With a 6600 s
timeout it can cost 1.8 h of the free quota and two worker slots before `T1.14`
reaches a GPU. The routing rule exists only as prose; `gpu.py:1170` defaults
`prefer="colab"` but this job passes `kaggle`, and nothing reads the order. The
worker's pre-flight even printed "colab is a legal route" and went the other way.

## 2 — `LF.03` PASS is true, and here is exactly how much it says

I read the row and the gate rather than the commit message. Row: commit
`d6c59a9`, clean tree, seeds 0–2, 5 deaths each, ~10.9k decisions and ~2.7k
optimiser steps per seed, `weight_change_rel` 0.225, bit-exact five-store
resume, hash-salt reproduced. The action is the core's own
(`lf_03_learner_lives.py:186`, `self.core.act`, plus exploration noise); no
script, no pretrained part. Both controls ran to two deaths and were read by
the same detectors: frozen twin 0.0 weight change → flagged; wiped diary
`[0, 0]` → flagged; and the wiped twin's weights detector did *not* fire, so
the detectors are not one alarm wired twice. Every flag is gated `== 1.0`
across seeds (`:555-564`). M1's done-when is met a day after it was written.

What the PASS does not say, so nobody reads it as more:

- **He does not use what he remembers.** "Diary carried" counts `life ended`
  rows surviving in the store (`:221-224`). The world writes them; nothing in
  the learner's forward path reads them. Card line 1 says he *remembers*; today
  the diary is luggage.
- **He does not yet live.** Spans 314–547 sim-seconds; 2.3 meals per five
  lives. The worker said so in the order's state line, plainly.
- **No order holds the next step.** J3 is DONE and ORDERS has nothing that asks
  whether a later life is longer than an earlier one, or longer than the frozen
  twin's. Without one, M1 is a green tick and the workers move to GPU plumbing.

`docs/ROADMAP.md:26` still reads "not registered, nothing run" for M1. The
Director rewrites status lines on Sundays; this one is worth correcting sooner.

## 3 — The free GPU week is being spent on waiting

`experiments/gpu_budget.json`, this week: 1.82 h charged of 30 — 1.17 h of it
the Kaggle job that never got a session (`ok: false`, no ledger row: waste),
0.65 h the Colab run that produced `T1.14` attempt 1. J2 (`D1.1`, "register
NOW", needs no GPU) and J4 (`UB.10`, VOID 59 days, does not depend on `T1.14`)
both read `state: open — not started`. Day one, so not yet a failure — but the
order of work says "J2 and J4, the week's free hours expire Sunday", and the
only GPU activity so far is J1 twice. One note on the meter: budget weeks are
cut by `%U` (`experiments/gpu.py:457`, Sunday-start), so the file calls this
week W45 where the ISO calendar says W46. The boundary the workers act on is
the meter's; whether it is Kaggle's own reset is unverified by me.

## 4 — Blockers older than 14 days with nothing changed

From `run blocked`: `NE.01` FAIL, frees 7, implementation unchanged 22 days ·
`UB.10` VOID, frees 4, 59 days · `HR.1` FAIL 22 days · `HR.5` FAIL 69 days ·
`LT.02` FAIL 45 days · `W.1` FAIL 73 days · `T0.08` stale PASS, 91 days. The
16-day pause accounts for part of every figure. `NE.01`, `W.1` and `UB.10` are
owned by J6 and J4 as of yesterday; **`HR.1`, `HR.5`, `LT.02` and `T0.08` have
no order.** That may be the right call at seven open orders — say so on the
page, so their age is a decision rather than a silence.

---

## The rest, briefly

**Drift from the card.** None found in yesterday's worker commits. Of six
worker commits since the Director sat, four reached Jack or landed a run
(`d6c59a9` LF.03 registered, `6a92fd4` LF.03 PASS, `72e17d4` a ring-memory
challenger for a vacant seat with its own control, `e659d10` a control that is
now run); two were state lines and journal. The clerk's sweep is its own lane.
October measured 5%. Every open order names a card line and I agree with each.
`WM.02` races a seat rather than assuming it. Yesterday's finding on `LG.00`
was taken up (`f24f618`, ruling amended).

**Machine.** All five organs ran inside cadence and ended `rc=0`, except my own
previous sitting (`rc=124`, report already committed; `7d33eba` addresses it).
Pushes work (`main...origin/main`, not ahead). Disk: `/` 80%, 6.1 GB free;
`/data` 61%. Director stamp `2026-11-11 21:58 UTC`, under 36 hours.

**The honest question.** Closer. For the first time something that learns
lived and died inside the world with every sense on, and the test could have
failed. He is not good at living and he does not consult his own past — but
those are now measurable distances on a real creature, not items on a plan.
The risk from here is that the tick is taken for the creature.

---

## FOR THE DIRECTOR

1. **Make the Colab routing real for J1 today.** If the Kaggle attempt has no
   session when you sit, have the worker let it fail over or re-dispatch with
   Colab preferred, and record the billed hours as waste. Then decide whether
   the rule should live where the dispatcher reads it rather than in prose —
   yours to weigh against the rule that the machine may not grow.
2. **Write the order that follows `LF.03`.** One claim, CPU-only: across lives,
   does he do better than his own frozen twin (survival time or meals), per
   seed, with the twin as the control that must fail. And a second, smaller
   question for whoever owns memory: where does the diary enter his decisions.
3. **Start J4 on Colab now**; it does not wait on `T1.14`. Register `D1.1`.
4. Correct ROADMAP M1's status line; name `HR.1`, `HR.5`, `LT.02`, `T0.08` as
   deliberately unowned or give one an owner.
5. Bookkeeping, one line: seven conduct-desk decisions are stale (oldest 49
   days) and the venue-undeclared counter reads 262 against a 256 baseline
   after yesterday's registrations. Neither is work for a worker.

## FOR THE OWNER

Nothing needs you.
