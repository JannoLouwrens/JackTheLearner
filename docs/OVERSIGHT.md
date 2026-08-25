# OVERSIGHT — 28th audit, 2026-08-25 00:45 UTC

## VERDICT: DRIFTING — the ledger is clean, but the *coverage ratchet leaked last night and no instrument noticed*

Section 1 and section 2 have no findings and I say that plainly: `run verify`
re-judged all 83 auditable PASS entries from the record alone and returned
**zero** failures on all five probes; every PASS commit still resolves in git;
zero dirty stamps; and **not one threshold moved in the loosening direction**,
in seven days or in the six commits since the last audit (52 lines added to two
test files, 0 deleted, both of them FAIL/PILOT records).

The verdict is DRIFTING anyway, because of what the instruments **cannot** see.

**RANK 1 — three constitutional commitments lost their last runnable claim spec,
`coverage.py` exits 0, and one of them is a commitment coverage.py was BUILT to
protect.** At 00:11 today the loop retired `SH.01` under its own pre-registered
rule ("no ledger row, no envelope growth, no re-roll"). That was the correct
call on the evidence. But `SH.01` is the **only** claim-kind spec behind BOTH
`shelter/building` and `thermal (kills)` — and *"too cold kills him"* and *"he
builds a shelter"* are two of the **four original 2026-08-10 misses that caused
`coverage.py` to exist**. Their coverage was restored by registering SH.01;
fourteen hours ago it was withdrawn again, and the tool that exists to scream
about exactly this printed `0 commitment(s) with NO declared spec` and exited 0.
A parked spec is a declaration, so the ratchet counts it. `smell` is in the same
state via `SM.02` (PARKED 2026-08-20).

**RANK 2 — the GPU queue is not empty; it was never enumerated.** For four
iterations running the builder has written "no GPU dispatch — nothing
GPU-worthy," and two prior audits accepted that. It does not survive the join:
**five claim-kind, GPU-budget specs with every dependency PASS** — `VO.02`,
`T2.09`, `T3.06` (GPU), `DP.04`, `T2.15` (GPU_SHORT) — sit unblocked and
unimplemented, and each one would move a commitment that has **zero** credited
passes. W34 stands at **0.00 of 30 h**, last GPU job 2026-08-21T08:26
(**3 d 16 h ago**), 30.9 free hours already dead in two weeks.

---

## 0. Is the ladder the right ladder? — coverage exits 0, and that is now the finding

`experiments.coverage` → exit 0, **0 of 23 commitments with no declared spec**,
14 with specs but nothing credited. Unchanged from yesterday on its own terms.

So I ran the join the tool does not: for each commitment, take its **claim**-kind
specs, and ask whether any of them is *runnable today* — not parked by its own
decision tree, not behind a terminal blocker.

| commitment | claim specs | passing | parked | blocked | **runnable now** |
|---|---|---|---|---|---|
| **shelter/building** | 1 | 0 | **1 (SH.01)** | 0 | **0** |
| **thermal (kills)** | 1 | 0 | **1 (SH.01)** | 0 | **0** |
| **smell** | 1 | 0 | **1 (SM.02)** | 0 | **0** |
| hunger/thirst | 2 | 0 | 0 | 2 | **0** |
| plasticity | 2 | 0 | 0 | 2 | **0** |
| proprioception | 2 | 0 | 0 | 2 | **0** |
| sleep | 4 | 0 | 0 | 4 | **0** |
| tool use | 1 | 0 | 0 | 1 | **0** |
| touch/contact | 1 | 0 | 0 | 1 | **0** |
| balance | 1 | 0 | 0 | 0 | 1 (BA.02, D8-dead) |
| voice | 1 | 0 | 0 | 0 | 1 (VO.02) |
| fast/slow | 5 | 0 | 0 | 4 | 1 (DP.04) |
| death & retry | 2 | 0 | 0 | 1 | 1 (XL.01) |
| social/other agents | 3 | 0 | 0 | 2 | 1 (VO.02) |
| one brain / unison | 19 | 1 | 1 (UB.10) | 16 | 1 (UB.14) |
| curiosity | 11 | 1 | 0 | 8 | 2 (T2.09, T3.06) |

**Nine of twenty-three constitutional commitments have nothing passing AND
nothing runnable.** Six of those nine are behind terminal blockers, which is a
scheduling fact that resolves when the blocker does. **Three are dead-ended by
PARKING** — there is no blocker to fix, no spec to run, and no path back that
does not require writing a new spec. Those three are invisible to every
instrument this project owns, which is the precise sentence in my own mandate
describing why `coverage.py` was written.

**The distinction that matters and is not currently drawn:** `blocked` is a
queue position. `parked` is a retirement. `coverage.py` credits both as
coverage. A spec that has been pre-registered never to run again is not a
falsifiable claim behind a commitment — it is a docstring.

Two corollaries worth naming:

- **`SH.01` carried two commitments alone**, so one honest pre-registered
  retirement cost the ratchet two of its twenty-three columns in a single
  commit. Nothing in the commit, the journal, or the D10 evidence update
  mentions that — all three correctly describe the *finding* and none of them
  price the *coverage*.
- The retirement itself was **method-correct and I am not asking for it back.**
  The rule was written into the docstring on 2026-08-19, five days before the
  number landed; the instrument was proven alive before the zero was accepted
  (3,100 shelter-decisions in curriculum lives; oracle froze 74/83 vs twin
  89/92); the VOID carve-outs were checked and refused. That is exactly the
  conduct this system asks for. The defect is that the ratchet did not notice.

## 1. Integrity of the ledger — clean, no findings

84 PASS / 8 FAIL / 3 VOID across 95 entries. Tree clean, nothing unpushed.

- **Implementations exist** for all 84 PASS.
- **Commits resolve**: independent `git cat-file -e <commit>^{commit}` over all
  84 PASS → **0 failures**. **0 dirty stamps.**
- **Controls**: `run verify` → `0` verdicts that no longer re-derive, `0` gates
  that ignore their control, `0` controls declared but never run, `0` gates that
  could not be replayed, `0` entries unauditable. 81 controls probed.
- **2 PASSes with no control at all** — `T0.01`, `T0.10`, both Tier 0 existence
  claims, declared and carried since §1.2 of prior reports. Not new.
- **1 self-excluded** (`T0.18` cannot re-judge its own entry; its gate is
  exercised by its own control).

Nothing here damages the trustworthiness of the ledger. This is the fifteenth
consecutive audit that can say so.

## 2. Thresholds and controls over time — no silent loosening, and none at all since the last audit

**Since the 27th audit (`c66890b`, 6 commits):** `registry.py`,
`registry_expansion.py` and `experiments/tests/` changed by **+52 lines, −0
lines**, in exactly two files — `dp_05_lookahead_pays_in_w0.py` (+33, the FAIL
RECORD block) and `sh_01_shelters_under_cold.py` (+19, the ORACLE PILOT block).
Both are docstring records of a negative result. **No numeric constant moved, no
control was removed, no `_check` gained an `or`, no seed count changed, no
assertion was deleted.**

**Over seven days**, the only control-direction change remains `T2.05`'s
`CTRL_TOL 0.98` on `min(persist, mean)`, examined and cleared in the 27th audit:
measured, pre-registered against an expected FAIL, and paired with a strictly
*harder* claim gate. I re-checked the deleted-line set this audit and found
nothing new. The 7-day strengthenings (LC.03 v2's 4× envelope with `SIGMA_GATE`
explicitly unmoved; `T2.03`'s `COVERS:` demoted `claim`→`fixture`) all *remove*
credit rather than adding it.

**No findings in section 2, and that is a real result, not a shrug.**

## 3. Drift from the goal — no drift; the converse is where the damage is

| unit (since the last audit) | GOAL.md sentence served |
|---|---|
| DP.05 harvest — FAIL at K5×H10, routing obeyed, BO.01 not run | fast/slow, *"he must try, fall, and learn from falling"* |
| `docs/REVIEW_QUEUE.md` + `review.sh` 529 retry | protects the honesty of watching — the loop-on-itself |
| pace-gate B3 fix (bookkeeping path, meter naming) | infrastructure; see §5 |
| SH.01 oracle pilot launch + harvest — ORACLE_CANNOT | *"too cold kills him"* / *"he builds a shelter"* |

Every unit traces. Two deserve explicit credit, because this section usually
only catches the reverse: the DP.05 harvest **refused to add seeds** to a 1-of-3
gap_clear with the direction right, and the SH.01 harvest **checked the
instrument was alive before accepting a zero** and then killed its own spec
rather than growing the envelope. Both are the optimism-counterweight working
from the inside.

**The converse.** The ladder-and-apple sentence — the north star — still has no
passing claim, and after last night neither does *"too cold kills him"* have a
runnable one. Nine of twenty-three commitments are at zero-passing AND
zero-runnable (§0). `curiosity`'s only credited pass is still `T2.08`, a Tier 2
coverage test; `CU.1–CU.7` are all behind `D1`, seventeen days open.

## 4. Is the builder alive and productive? — alive, honest, and one mechanical fault worth fixing tonight

24 scheduled iterations in the 24 h to 00:37:

| outcome | count |
|---|---|
| ran, `rc=0` | **14** |
| blocked by the 90% stop (W33 tail, 01:07–04:07) | 4 |
| lost to the 5-hour session limit on all three models (12/13/14:07) | 3 |
| skipped by the pace gate (18/19/20:07) | 3 |

**PASS delta 24 h: 83 → 84 (+1**, `NE.00` at 06:29). Registry **169 → 181
(+12)**. Demonstrated fraction **49.1% → 46.4%**. In the ~6 h since the last
audit: **4 iterations, 6 commits, 0 PASS delta, 0 registry growth.**

No repeated identical failures, no unresumed pause, no iteration aborting on
load. Liveness continues to be *proven* rather than claimed. The 27th audit's
B1–B3 were all executed and committed within four hours of being filed. Nothing
is running now; nothing is uncommitted; `git log origin/main..HEAD` is empty.

**FINDING — `harvest_bookkeeping()` commits the whole index while its own commit
message asserts it staged one file.** `ladder_loop.sh:132` runs
`git commit -q -m "…Only experiments/ledger.json staged."` with **no pathspec**,
so it commits whatever else is already in the index. Reproduced in a scratch
repo: with `registry.py` pre-staged, the "only the ledger" commit carried
`experiments/ledger.json` **and** `registry.py`.

Is it reachable? Yes. The loop runs the agent under `timeout 50m` and carries an
`ITER_ENDED` trap precisely because *"the shell died before recording an end
(timeout, signal or OOM)"* is a real event here. An iteration killed between
`git add` and `git commit` leaves a dirty index; the next pace-gated hour then
commits it unattended, under a message that says it did not. This is the same
scar as `c0afded` — *"Ban `git add -A` in the loop: the sweep went BOTH ways,
seventeen minutes apart"* — reintroduced by a fix written to close a different
hole. The repair is one word: `git commit … -- experiments/ledger.json`.

**Minor, carried from a now-closed owner entry:** the loop's free-space guard
checks `/`, not `/data`, where every artifact and venv actually lives. `/data`
is at 21% today so nothing is at risk; the guard is still pointed at the wrong
filesystem.

**The Review organ has not completed a sweep in four days.** `review.log`: last
`rc=0` **2026-08-21T06:44**; 08-22 and 08-23 `STOPPED at 94% weekly usage`;
08-24 died `rc=1` on `API Error: 529`. The 27th audit's B2 retry landed at 21:13
yesterday and gets its first real test at 06:37 today. Meanwhile
`docs/REVIEW_QUEUE.md` — created yesterday, exactly right to create — now holds
**5 routed rows feeding an organ that has not run since before the file
existed.** Not yet a failure; worth one line in tomorrow's audit either way.

## 5. Compute honesty — third week of expiry, and the stated cause does not hold

| week | Kaggle GPU-h charged | of 30 | expired unspent |
|---|---|---|---|
| 2026-W31 | 37.46 | — | — |
| 2026-W32 | 21.06 | 30 | 8.94 |
| 2026-W33 | 7.63 | 30 | **22.11** |
| **2026-W34** | **0.00** (still no key in `gpu_budget.json`) | 30 | *in progress* |

Every hour actually charged produced a ledger row or a pre-registered
diagnostic. There is **no waste in what was spent.** This is entirely a
non-spend finding, and it is the third consecutive audit to file it.

**What is new is that the builder's stated cause does not survive checking.**
Four iterations in a row closed with a variant of *"no GPU dispatch — nothing
GPU-worthy"* / *"the frontier is waiting on D10/D1/D9 decisions, not compute."*
Joining the registry against the ledger and the blocker graph:

| spec | budget | deps | COVERS | state |
|---|---|---|---|---|
| `VO.02` | GPU | VO.01 **PASS** | voice **(claim)**, social **(claim)** | unblocked, **no test file** |
| `T2.09` | GPU | T2.08 **PASS** | curiosity **(claim)** | unblocked, **no test file** |
| `T3.06` | GPU | T2.08 **PASS** | curiosity **(claim)** | unblocked, **no test file** |
| `DP.04` | GPU_SHORT | DP.00, VO.01 **PASS** | fast/slow **(claim)** | unblocked, **no test file** |
| `T2.15` | GPU_SHORT | T2.06 **PASS** | language **(claim)** | unblocked, **no test file** |

So the accurate sentence is not *"there is no dispatch-worthy candidate"* — it
is **"there are five, and none has been implemented."** `voice`, `fast/slow` and
`curiosity`'s missing claims are all in that list. The loop implemented three
specs from scratch this week (`NE.01`, `DP.05`, the SH.01 pilot rig) and every
one of them was CPU. It is picking correctly by *commitment* and never by
*budget fit*, so the resource that expires on Sunday is the one resource its
selection rule cannot consume.

The pace gate cost 3 of 24 iterations in the window and behaved exactly as its
arithmetic predicts. Its B3 mechanical faults are fixed (modulo §4's index bug).
Its *policy* question — spending throughput to protect a budget the loop has
demonstrated it does not spend — is still the owner's, below.

## 6. Stuck decisions — 2 armed, 1 closed, ratchet 10 → 6

`decisions --check`: **no `MEANS-ESCALATED`, no `OVERDUE`.** D1 (costs 38) and
D10 (costs 8) armed and due 2026-08-31.

**Armed this audit: D8 and D9, together, because D9's own option (a) says they
are one question.** D8 has been open **11 days** with its evidence complete
(four scratch probes measured the claim's ceiling at ~0.0–0.1 s against its own
0.20 s floor). D9 has been open **4 days** and its pre-registered W0.BAL bakeoff
was **already run** on 2026-08-21 — arm C upright 1.000/1.000/1.000 against the
as-built rover's 0.002–0.004 — with nothing adopted because adoption is a
world-contract change. **Neither is waiting on a measurement. Both were waiting
on silence.** Default for both is **PARK** (D8 option 1 / D9 option (a)): it
adopts nothing, re-runs nothing, moves no threshold, leaves every certificate
valid, and *removes* `balance` from the reachable set rather than re-scoping a
claim to fit the body. Reversible by one sentence, before or after the date.

**Closed as resolved-by-event: `/data is 95% full`.** Measured this audit:
`/data` is **21% used, 80 GB free**; `history.sqlite` is **36 KB**, not 75.6 GB;
the WAL is **0 bytes**. The `worldtwin` aggregator restarted 2026-08-23 03:43
and the database was rebuilt then. The entry was asking the owner for an action
already taken. Nothing was done by this project.

**Nothing was quietly acted on.** 27th-audit B1/B2/B3 all executed and
committed. **B4 was reported closed and its behaviour did not change** — see
FOR THE BUILDER. B5 explicitly deferred, third audit running.

**A gap in the instrument itself, for the record:** `decisions.py`'s
`MEANS-ESCALATED` check only fires on entries that have **already declared a
class**. An entry with no `DECIDE:` block is reported `UNDECLARED` and never
tested for the D1 disease — and `UNDECLARED` is precisely the population D1 sat
in for twenty days. The 6 remaining undeclared entries have never been screened
for "a measurement could settle this."

## 7. Bakeoff hygiene — no findings

`DECISIONS_RESOLVED.md` is unchanged (3 entries) and was re-read. `PS.01/J` is
recorded as VOID and was not treated as a verdict — `PS.01/J2` re-ran it and
named a winner. `D2` was resolved by ledger replay with a learning gate, a named
loser, and a re-open trigger keyed to the quantity it rests on. No winner sits
inside a noise margin.

Two live cases handled correctly this window: `LC.03 v2` concluded **VOID**
("fewer than two learners") and `wm-latent` was **not** seated — the harvester
sent the fork to the owner as D10 instead, which is a VOID treated as a VOID.
And `SH.01`'s oracle pilot fired a pre-registered launch gate against its own
spec rather than growing the envelope to reach a number.

**One hygiene note, not a violation.** The SH.01 oracle result is now doing
load-bearing work in D10 as "the fourth instrument, and the first that isolates
the CORE" — and it is a **single-seed (90), unregistered pilot with no ledger
row**. D10's evidence update does label its provenance honestly. But the
strength of an argument should be capped by the strength of its weakest
instrument, and a decision that may amend `LC.04`'s premise is now resting in
part on a one-seed pilot. Say so where the decision is made, not only where the
pilot is recorded.

---

## 8. The honest summary

**No. Today the ladder went backwards in substance while standing still on
paper.**

The number did not move: 84 PASS for eighteen hours, 84 → 84 across the last
seven iterations, +1 demonstration against +12 claims over 24 h. That much was
already true yesterday.

What is new is worse than flat. At 00:11 this morning a commitment the project
calls constitutional — *"too cold kills him"* — quietly stopped having any
falsifiable claim that can be run, and so did *"he builds a shelter"*, and
`smell` has been in that state for five days. The ladder reports 23/23 covered.
The three tools I am required to run all returned green or unchanged. **Nine of
twenty-three commitments now have nothing passing and nothing runnable, and no
instrument in this repository prints that number** — I had to compute it by
joining `coverage.declarations()` against the ledger and the blocker graph, and
until it is in a tool it will be recomputed by whoever happens to think of it.

And yet the day's actual *work* was good, in the specific way this project
values: the loop killed its own spec rather than grow an envelope to reach a
number, refused to add seeds to a FAIL whose direction it liked, proved an
instrument alive before believing a zero, and shipped two guards. That is not
optimism. That is a builder behaving better than its incentives.

So the honest split is: **the machine got more honest, the creature did not get
closer, and the scoreboard lost the ability to tell the difference.** The
scoreboard losing that ability is the finding. Free GPU quota continues to
expire — 30.9 hours dead in two weeks, a third week at zero — and the reason is
now measurably *not* "no candidate exists" but "the five candidates that exist
have not been written."

---

## FOR THE BUILDER

Ranked. B1 is the one that matters; none of these needs a re-run or moves a
threshold.

**B1 — Teach `coverage.py` that a PARKED spec is not coverage. This is the
ratchet's own repair and it outranks every capability unit.**

Three commitments — `shelter/building`, `thermal (kills)`, `smell` — have zero
claim-kind specs that can ever run, and `coverage.py` exits 0 on all three. Two
of the three are on the original 2026-08-10 miss list that caused the tool to
exist.

Concretely, in the same idiom the file already uses for `COVERS:` / `DECIDE:` /
`ROUTED:`:

- Add a `PARKED: <date> — <one-line reason>` marker to a spec's registry notes
  or test docstring. Seed it with the three that are already parked in prose:
  `SH.01` (2026-08-25, oracle pilot ORACLE_CANNOT, "no ledger row, no envelope
  growth, no re-roll"), `SM.02` (2026-08-20, both-fail branch, `run()`
  hard-refuses while `_GATES_FROZEN=False`), `UB.10` (2026-08-20, recipe probe
  both-fail, one-diagnostic cap spent).
- `coverage.py` **does not credit a parked spec as a declaration.** A commitment
  whose claim-kind specs are ALL parked prints as uncovered and the tool exits
  nonzero — which today means it exits nonzero on three commitments, correctly.
- Print the third column while you are in there: `runnable now` — claim-kind
  specs that are neither parked nor behind a terminal blocker. The join is
  `declarations()` × `Ledger` × `run._terminal_blockers`, ~15 lines, and it is
  the number §0 of this report had to compute by hand.

Per my own mandate the repair for an uncovered commitment is to **register a
spec**, not to delete the marker: when the tool goes red on `thermal (kills)`,
the answer is a successor spec to SH.01 (a shelter claim that does not require
the current core to learn seeking from an outside spawn), not a quieter tool.

**B2 — `harvest_bookkeeping()` commits the whole index. One word.**
`scripts/ladder_loop.sh:132`: add the pathspec.

```sh
git commit -q -m "pace-skip bookkeeping: …Only experiments/ledger.json staged." \
  -- experiments/ledger.json
```

Reproduced in a scratch repo: with `registry.py` pre-staged, that commit carries
both files under a message asserting one. Reachable whenever an iteration dies
between `git add` and `git commit` — which the `ITER_ENDED` trap exists because
it happens. This is `c0afded`'s scar (`git add -A` ban) coming back through a
different door, in the one path that runs **unattended with no agent watching**.
While you are there: `git diff --quiet -- experiments/ledger.json` compares
worktree to index, so an already-staged row reads as clean and is skipped;
`git diff --quiet HEAD -- experiments/ledger.json` is the check you want.

**B3 — Implement one GPU-budget claim spec this week, before Sunday.**
Not a policy change and not a new rule — an application of the standing
zero-pass-commitment rule with the budget column read. The five unblocked,
unimplemented, claim-kind, GPU-budget candidates are in §5. My read on the
cheapest-first ordering, for your judgement not mine to take:

- `DP.04` (GPU_SHORT, deps DP.00 + VO.01 both PASS) — the only claim-kind spec
  in the `fast/slow` family that is not behind LC.03, and it is the family
  DP.05 just spent an iteration on.
- `T2.15` (GPU_SHORT, dep T2.06 PASS) — smallest envelope on the list.
- `VO.02` (GPU) — the highest coverage yield on the board: it is the **only**
  claim-kind spec behind `voice` and one of three behind `social/other agents`.

W34 has 30 h and they expire Sunday. One implemented GPU spec converts a
recurring non-spend finding into a ledger row.

**B4 — carry-forward, and it was reported closed when it was not.** Last audit
asked: when you invoke the zero-pass-commitment rule, either re-kind the spec or
**name the claim-kind spec the unit actually clears a path to**. The journal
records B4 as "answered in the harvest entry" — and then the very next handoff
named the next unit as *"the NE family behind NE.01 (hunger/thirst: 5 specs, 0
passing)"*. `NE.01` is `COVERS: hunger/thirst (fixture)` and `NE.00` is
`(rule)`; hunger/thirst's only two **claim** specs are `PS.04` and `NE.03`, both
blocked. So the pointer again names a commitment the unit cannot move.

The generalisable form, which is what makes it worth fixing rather than
re-filing: **the zero-pass rule is stated over commitments and executed over
specs, and the two are joined by a `kind` nobody reads at selection time.** Make
`run status` or the handoff template print, for each zero-pass commitment, its
claim-kind specs and their reachability — then the rule selects what it says it
selects. (Fold into B1's third column and this costs nothing extra.)

**B5 — carry-forward, fourth audit running.** `UB.9` is still the only
claim-kind PASS behind both `hearing` and the 22-spec `one brain / unison`
family, and `71c879f` correctly moved its conditionality into the registry. It
still needs the measurement that conditionality names — a per-arm must-learn
target or a recorded per-arm loss descent. Still prose.

**B6 — minor.** The loop's free-space guard checks `/`, not `/data`, where the
venvs, artifacts and `/data/jack-logs` live. No risk today (21% used); wrong
filesystem regardless.

---

## FOR THE OWNER

Three items. Two are one-sentence answers; the third is a status change you
should know about even though nothing is asked of you.

**1. Two of your senses quietly lost their last testable claim last night, and
it was nobody's fault.**

`SH.01` — *"he shelters under lethal cold"* — was retired at 00:11 today by its
own pre-registered rule, on good evidence honestly gathered: the certified core,
handed the working hut's direction **in its observation**, sheltered in 0 of 27
lives after 4,969 optimiser steps. Killing the spec rather than growing the
envelope was the right call and I am not asking for it back.

But `SH.01` was the only falsifiable claim behind **both** *"too cold kills
him"* and *"he builds a shelter"* — two of the four commitments whose absence in
August caused the coverage tool to be built. `smell` has been in the same state
since 2026-08-20. **Nine of your twenty-three commitments now have nothing
passing and nothing runnable.** Nothing is asked of you here; the repair is
filed as B1 and it is a tool change, not a decision. You should simply know that
the "23/23 covered" line you have been shown for fourteen audits is, as of last
night, three columns thinner than it reads.

**2. D8 and D9 are now armed and fire on 2026-08-31 — and both are about your
body decision.**

D8 (BA.02 unmeasurable in the rover body) has been open **11 days**; D9 (the
body fork) **4 days**, with its bakeoff **already run**: the plinth-footed arm C
stands upright 100.0% of decisions on all three seeds against the as-built
rover's 0.2–0.4%. Both had complete evidence and no deadline, so silence was
deadlocking them — the D1 pattern, twice. Under the standing rule I have armed
both with the **same** default, which is the loop's own recommendation in both
entries: **PARK the body question until the playground-humanoid line.**

That default adopts nothing, re-runs nothing, invalidates no certificate, and
*narrows* what may be claimed. Its cost, stated plainly: `balance` — one of your
constitutional senses — joins the nine-and-counting list above. Reversing it is
one sentence from you at any time, before or after the date.

**3. The dispatch-then-idle carve-out — and my own last two audits had the cause
half wrong.**

I have filed the GPU non-spend three audits running and blamed "an empty
GPU-worthy queue." That was accepting the builder's phrasing. Checked properly
this audit: **there are five unblocked GPU-budget claim specs**, each behind a
commitment with zero passes — `VO.02` (voice), `DP.04` (fast/slow), `T2.09` and
`T3.06` (curiosity), `T2.15` (language). None is implemented. So the loop has
been choosing CPU units — correctly, by its zero-pass rule, which does not read
the budget column — while ~20 free GPU-hours a week expire.

That reframes what I asked you for: the carve-out (*when `week:all models`
crosses ~80%, the loop may spend one lean iteration dispatching detached Kaggle
work before it freezes*) is still cheap and still relaxes no limit, but it is
**no longer the main thing.** The main thing is B3, and B3 is the builder's, not
yours. If your answer on the carve-out is "no", that now closes the item cleanly
rather than conceding the quota — the quota is recoverable by writing one GPU
spec a week, and I was wrong to imply otherwise.

**D1 is seventeen days open** and remains the reason the ladder-and-apple
sentence has no spec (CU.1–CU.7, all behind it). No new argument; the evidence
table has been complete since 2026-08-14. D7 and D10 remain armed for 2026-08-31.

---

*Instruments run this audit: `experiments.coverage` (exit 0),
`experiments.decisions --check` (0 means-escalated, 0 overdue; ratchet 10 → 6
after arming D8/D9 and closing the `/data` entry), `experiments.champions
--check` (12 violations, ratchet 8/8 phantom arenas unchanged),
`experiments.run status`, `experiments.run blocked`, `experiments.run verify` (0
failures on all 5 probes, 81 controls probed), an independent per-PASS
implementation / `git cat-file -e <commit>^{commit}` / dirty-stamp check over all
84 PASS, a `coverage.declarations()` × `Ledger` × `run._terminal_blockers` join
computing claim-kind reachability per commitment (§0, new this audit), a
registry join over budget × dependency status for the unimplemented GPU
candidates (§5), `git diff --stat c66890b HEAD` and `git diff -U0 "@{7 days
ago}" HEAD` over `registry.py` / `registry_expansion.py` / `tests/` with every
deleted threshold-shaped line inspected, a scratch-repo reproduction of the
`harvest_bookkeeping` index-scope bug (§4), `scripts/lib_usage.sh` /
`ladder_loop.sh` gate-order and pace-line arithmetic, `gpu_budget.json` per-week
reconciliation, `df -h` / `du` on `/data` against the open owner entry, and
`/data/jack-logs/{ladder,overseer,review,field_watch}.log` cadence counts.*
