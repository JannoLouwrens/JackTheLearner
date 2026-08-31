# OVERSIGHT — 52nd audit, 2026-08-31 00:45 UTC (HEAD `901f7fc`)

## VERDICT: DRIFTING — the ledger is clean; the organ that owns the measured bottleneck has been dead for two days, and eleven defaults fire into it tomorrow

Sections 1, 2 and 7 have **no findings**, checked mechanically. All 92 `PASS`
rows resolve to an implementation, to a commit that exists in git, and to
recorded `control_metrics` wherever a control is declared; **zero** rows carry a
`+dirty` stamp. Over seven days exactly one numeric bar moved downward
(`DECAY_MIN` 1.5 → 1.25, adjudicated by the 50th audit and still correct); eight
moved upward. No control was deleted, no seed count reduced, no `_check` gained
an admitting disjunction. `DECISIONS_RESOLVED.md` is clean.

The three mandatory ratchets: `decisions` `rc=0` (**0 UNDECLARED, 0
MEANS-ESCALATED, 0 OVERDUE** — nothing left to arm, and I say so rather than
manufacture an entry), `champions` `rc=0` at 8 standing violations against
today's freshly-ratcheted baseline, `coverage` **exits 2** on the pre-existing
`2 cost class(es) NEWLY EMPTY — gpu<20min, gpu<2h`. That red is honest and it is
§5.

The builder ran **25 iterations in 24 h, 22 at `rc=0`, 3 at `rc=124`**, and moved
the ladder **85 → 92 PASS of 200**. Its conduct on thresholds was strengthen-only
throughout, including twice when strengthening cost it a headline.

**Three findings, and none of them is about a spec.**

1. **The Review has produced nothing since 2026-08-29 06:48.** Its
   **Sunday FULL** run — the only mode that does Part 2, and the run that owed
   the `w0-too-shallow` world design — started 2026-08-30 06:37, hit
   `Error: Reached max turns (60)` after **11 minutes of a 40-minute budget**,
   and exited having written **nothing**: no `PROGRESS.md`, no `PROGRESS_LOG.md`
   line, no `REVIEW_QUEUE.md` disposition, no commit. `seal_output` correctly did
   not fire, because it stamps a *dirty* output file and this run never made one.
   **No organ announced it.** `docs/PROGRESS.md` today still opens
   *"2026-08-29 … Ladder: 84/187 (44.9%). Fifth consecutive day on which not one
   figure in this table has moved"* — during the most productive 48 hours in the
   project's history, and with nothing on the page saying it is stale. This is
   the 27th audit's own lesson recurring on the same organ: *"a queue whose
   consumer is silently dead reads exactly like a queue that is empty."*

2. **A stray process burned a full core for 75 minutes on a box that serves
   paying tenants, and I killed it.** PID 3749514,
   `/data/venvs/jackthelearner/bin/python -c "x=0 / while 1: x+=1"`, started
   2026-08-30 23:24:47, orphaned to `ppid 1`, `cwd` in this repo — a manual
   verification aid for the `_cpu_fraction` fix (`1296ca3`), never cleaned up.
   `/proc` accounting at kill time: **453,199 utime ticks = 4,532 s = 1.26
   core-hours** of a 4-core shared box, at 25% of total capacity, alongside
   `BA.03`'s registered 3-seed run. The 00:07 iteration printed a **full `ps`
   dump** to confirm BA.03 was alive and did not see the second 99.7% python in
   its own venv two lines away. SYSTEM.md's *"leave no process running"* is a
   CONDUCT constant enforced by nothing but voluntary prose in an iteration
   report.

3. **Eleven defaults fire in under 24 hours, and one of them is the one the 41st
   audit said the owner should rule on rather than be ruled for.** `D10`'s
   default seats the learning core **BY VERDICT** off `LC.03`, whose ledger
   status is `VOID`. That was raised on 2026-08-28; its default text is
   byte-identical today. Three of the four defects the 41st audit found *have*
   been repaired (see §6) — `D10` is the one that has not.

---

## 1. Integrity of the ledger — NO FINDINGS

109 rows: **92 PASS, 13 FAIL, 4 VOID**, of 200 registered specs.

Checked mechanically over every `PASS` row (`/tmp/audit52.py`, re-derivable):

| check | result |
|---|---|
| `commit` resolves in git (`cat-file -e`) | **0 phantom** of 92 |
| `commit` carries a `+dirty` stamp | **0** of 92 (and 0 across all 109 rows) |
| spec resolves in `BY_ID` | **92 of 92** |
| `control_metrics` recorded where a control is declared | **90 of 90** |
| implementation passes `control_fn=` | **90 of 90** (only `t0_01`, `t0_10` and a probe script lack one) |

**I looked hardest at `LG.00`, because the 22:07 iteration flagged it and it is
the load-bearing claim of the day.** It is clean, and the mechanism worked
exactly as designed: attempt 1 ran at `8faff43+dirty` at 18:47, was recorded
**VOID** with the message *"run did not test the claim; not a refutation"*, and
its failing implementation was preserved at
`refs/jack/failimpl/LG.00/2026-08-30T18-47-59`. The recorded `PASS` is attempt 2
at clean commit `6c008d9`. A dirty tree produced a VOID and an archived artifact,
not a claim. `T0.27`'s live-violation count going 1 → 2 was that VOID, and it has
since resolved back to 1.

The two `PASS` rows with no declared control remain `T0.01` (repo imports) and
`T0.10` (Kaggle round-trip), both `control=None` with
`null_baseline="n/a — structural precondition"`, neither asserting a capability.
Carried since the 49th audit as a one-line docstring repair; still costs nothing;
still not re-ranked.

**One finding worth stating, because it bears on this report's own
trustworthiness.** Four certificates are `STALE` by `impl_sha`, and **two of them
certify two of the three tools this audit is required to run first**:

| spec | certifies | ran at | HEAD | why stale |
|---|---|---|---|---|
| `T0.21` | `experiments/coverage.py` | `de1de26a…` | `c42077be…` | `coverage.py` changed at `1c41cb5` (23:23) |
| `T0.29` | `experiments/champions.py` | `d4dbb911…` | `af35d0de…` | `champions.py` changed at `901f7fc` (00:24) |
| `T0.17` | `protocol.py` | `29f2a9be…` | `bb758901…` | |
| `T0.27` | `protocol.py` | `4e6a843e…` | `4bb2094b…` | FAIL row, held RED by `D16` |

`T0.28` (which certifies `decisions.py`) is **not** stale — `decisions.py` is
unchanged since `0c7e36b`, the commit T0.28 ran at. So the escalation tool's
certificate is live and the coverage and seat tools' are not. The builder could
not re-run them: `BA.03`'s registered run has held `/tmp/jack-ladder.lock` since
23:16 and will until ~05:15Z. It replayed `T0.29`'s ten properties offline and
reported 0 failures — that is diligent, and it is not a ledger fact. I ran both
tools anyway and report their output above; the caveat belongs on the record
rather than in a footnote.

Carried and unchanged: `T2.02`'s VOID row is stale by content; 18 entries predate
`impl_sha` (17 verified byte-identical, 1 stale); 57 `PASS` rows predate
`spec_sha`. None is a claim standing on nothing.

## 2. Thresholds and controls over time — NO VIOLATIONS

Every named constant in `registry.py`, `registry_expansion.py`,
`experiments/tests/`, `protocol.py` and `bakeoff.py` that has both a `-` and a
`+` form in seven days — i.e. every constant that *changed value* rather than
being introduced (272 constants were newly introduced by new specs; none of those
is a move):

| spec | constant | move | direction |
|---|---|---|---|
| VO.02 | `COORD_MARGIN` | 0.20 → **0.35** | tightened |
| VO.02 | `COORD_MIN` | 0.55 → **0.70** | tightened |
| T2.19 | `UNI_MIN` | 0.8 → **0.90** | tightened |
| LG.01 | `CHANCE_BAND_HI` | 0.25 → **0.0** | tightened |
| T0.21 / T0.29 / T0.13 | `N_PROPERTIES` | 9→10, 10→11, 11→12 | tightened (three files) |
| BA.03 | `N_EVAL` | 48 → **120** | strengthened, derived from measured sigma |
| DP.04 | `LIFE_CAP` | 200 → **400** | envelope widened, then refuted by its own sizing run |
| T3.06 | `LIVES_PER_ARM` | 16 → **48** | strengthened |
| T2.11 | `_SEC_PER_SEED` | 1200 → 355 | a **cost estimate**, not a gate; measured |
| T2.09 | `DECAY_MIN` | 1.5 → **1.25** | **loosened — declared, adjudicated, do not re-fit** |

`DECAY_MIN` is the same move the 50th and 51st audits cleared: a rig bar, not a
claim bar, on a spec whose `run()` refused until the freeze commit, derived from
what the gate is *for*, and declared in `44f24c41` under the heading
`ONE BAR MOVED, DOWNWARD, IN THE OPEN`. I re-verified and did not touch it.

**Zero `control_fn` removals. Zero seed reductions. Zero assertions removed.**

Two moves in the last six hours are worth naming as the *opposite* of a
violation, because both cost the builder something:

- `BA.03` added `HEADROOM_MIN_MULT = 2.0` after finding its blind twin sat at
  88.6% of the horizon — a new gate, chosen on principle, added to a spec that
  was about to run and whose seed-90 pilot already read **negative** (gain
  −0.2375 s). It dispatched anyway and recorded that it had forecast a FAIL
  *before* the run, so the outcome cannot be narrated afterwards.
- `PL.02`'s dependency was **not** edited an hour after `PL.00` produced an
  inconvenient FAIL; the builder wrote that doing so *"is the shape of a
  weakening"* and routed it. Law 4 obeyed where obeying it was expensive.

## 3. Drift from the goal — none in what was built

Every unit since the 51st audit (18:45 → 00:24, six iterations) traces to a
GOAL.md or SYSTEM.md sentence:

| unit | sentence it serves |
|---|---|
| `LG.00` **PASS** (0.739 in-life vs 0.271 null; 0.533 vs 0.733 out-of-life) | *"he should be smarter inside his life and dumber outside it — that asymmetry is the proof he is a creature and not a costume"* |
| `DP.04` pilot VOID + sizing record | *"fast and slow, in one brain"* — and the honest finding is that the world has no resolution for it |
| `BA.03` harvest, gates frozen, 3-seed run dispatched | *"proprioception & balance"* in the sensory inventory |
| `D14` implemented, `test_lib_usage.sh` (31 assertions) | SYSTEM.md conduct: the three gates that decide whether any organ runs had **zero** tests |
| `queue_depth` states 3→5, `_cpu_fraction` fix, `CHAMPIONS.md` declaration syntax | SYSTEM.md *"is the machine better than I found it?"* |

**No drift.** Nothing in the last day serves no sentence.

**The converse, which is the real answer to this section, is unchanged from the
51st and I have nothing new to add to it.** `coverage` reports **12 commitments
with a live claim spec and nothing passing**: touch/contact, tool use, smell,
proprioception, shelter/building, balance, death & retry, thermal (kills),
plasticity, sleep, hunger/thirst, fast/slow. The three GOAL.md names as most
likely to be quietly neglected stand at **curiosity 2 passing of 12**,
**one brain / unison 1 of 22**, and **learning-by-living — the entire `NE` family
(8 specs) blocked behind `NE.01` FAIL**. `voice` gained its first passing claim
(`VO.02`) and `language (parent)` its second (`LG.00`) this week; those are the
only two of the twelve that have moved.

## 4. Is the builder alive and productive? — YES, and the model gate is working

Window 2026-08-30T00:07 → 2026-08-31T00:24:

| | |
|---|---|
| iterations started | **25** |
| ended `rc=0` | **22** |
| ended `rc=124` (50-min timeout) | **3** (00:57, 05:57, 12:57 — 12%) |
| PASS delta | **85 → 92** (+7) |
| registered verdicts | `W.1` FAIL, `W.2` FAIL, `PL.00` FAIL, `LG.01` PASS, `LG.00` PASS, `T0.28`/`T0.29`/`T0.30` PASS |

**`D14`'s (b-effective) reading is live and behaving as specified.** From 21:07
onward every slot logs
`REFUSING fable — week:Fable 100% is at or past the 95% model floor` and then
runs the unit on the Opus fallback. Four such slots, four `rc=0`, four commits.
The literal reading would have produced four aborts. The switch to the literal
reading is one crontab variable (`JACK_MODEL_READING=literal`) and is documented
in `DECISIONS_NEEDED.md`.

**Zero PASS movement in the last six iterations, and it is explained, not
alarming.** 19:07–22:07 produced a refutation (`DP.04`'s sizing) and machine
repairs, neither of which mints a ledger row; 23:07 and 00:07 were **lock-bound**
— `BA.03`'s registered 3-seed run has held `/tmp/jack-ladder.lock` since 23:16
and lands ~05:15Z, so no spec can run and no row can move until then. I confirmed
the run is alive: pid 3747299, 99.6% CPU, 242 MB RSS, `nice 19`, 1h23 elapsed.

The 12% timeout rate is the same as yesterday's and still nothing tracks it as a
rate. Inheritance works (`T3.06` attempt 1, `dd4d3f9`), so these are not silent
losses.

## 5. Compute honesty — no waste; the queue is the constraint, and its owner is the dead organ

`gpu_budget.json` by budget week (`%U`, Sunday-start, matching Kaggle's reset):

| week | used | of 30 | expired unspent |
|---|---|---|---|
| 2026-W32 | 16.61 h | 55% | 13.39 h |
| 2026-W33 | 7.89 h | 26% | 22.11 h |
| 2026-W34 | 1.62 h | 5% | **28.38 h** |
| **2026-W35** | **1.28 h** | **4%** | 28.72 h **still live** — W35 runs 08-30 → 09-05, five days left |

`overruns: []`. **There are no GPU hours spent without a ledger entry to show for
them.** The problem is the mirror image and it is now three weeks old: **63.9
free hours have expired unspent** since W32 (the W32 figure carries the known
6.38 h unattributable opening balance, which over-states spend and therefore
under-states the loss — the honest number is a floor).

The reason is inventory, and today it has a name:

```
gpu<20min   0  EMPTY  <- NOT FILLABLE: pilot BLOCKED on evidence (DP.04, SM.03); the repair is a REDESIGN
gpu<2h      0  EMPTY  <- NOT FILLABLE: pilot BLOCKED on evidence (T2.11);        the repair is a REDESIGN
gpu<8h      1  T2.02  (VOID — an arm to repair, not a dispatch)
```

**Four of the five gate-provisional specs have now run a pilot that measured the
pilot cannot succeed** — `SH.02`'s headroom VOID (twin, privileged oracle and
control all hold the roof at exactly 1.0000 against `HEADROOM_MAX` 0.85),
`SM.03`'s 8.5×-oversubscribed held-out split with a dead alive-proof
(`vis_open` 0.1167 vs 0.60), `T2.11`'s label-permuted control beating the claim
arm on both pilots, `DP.04`'s quantisation refutation (mean lifespan quantised at
6.25 steps against `MIN_GAIN` 5.0; 0 of 3072 lives ended between the old cap and
the new one). Each is declared `pilot_blocked` in code — a real repair, shipped
yesterday, that stops the next builder spending seeds on a fifth VOID.

**And every one of those four repairs is routed to the same desk: the Review.**
So is `w0-too-shallow`, the world-redesign row whose design was **owed by the
Review of 2026-08-30**. That Review died (§ headline 1). The Review's queue also
gained three new `OPEN` rows on 08-30 alone —
`w1-cold-is-not-lethal-at-night`, `w2-needs-have-no-single-k`,
`dp04-lifespan-has-no-resolution`. **Six rows OPEN, the oldest seven days, the
consumer dead for two, and nothing in the repo prints that sentence.**

This is the join no instrument makes: `coverage` correctly says the repair is a
redesign and names the destination; `REVIEW_QUEUE.md` correctly holds the rows;
nothing asks whether the destination is running. W35's 28.7 free hours are
unspendable until it does.

## 6. Stuck decisions — nothing is stuck, nothing was acted on unrecorded, and one default is still unsafe

`decisions --check`: **0 UNDECLARED, 0 MEANS-ESCALATED, 0 OVERDUE.** All 14 open
entries carry a `DECIDE:` block with a default and a date. My standing duty is
*"arm at least one per audit"* and **there is nothing to arm** — the ratchet
reached its floor on 2026-08-26. Inventing an entry to look useful would be the
opposite of the job.

**Eleven defaults are dated `2026-08-31` and today is 2026-08-31.** `decisions.py`
computes `overdue = (today - due).days` and reports `OVERDUE` only at `> 0`, so
they read `armed` for the next 23 hours and become due to fire at 00:00 on
2026-09-01: **D1** (costs 38 specs), **D4** and **D10** (8 each), **D3, D7, D8,
D9, D11, D12, D13, D14**. `D15`/`D16` follow on 09-05, `D17` on 09-07.

**Was any owner decision quietly acted on? No — and I checked the one that looks
like it was.** `D14` was implemented by the builder at ~20:4x on 08-30, eight
hours before its deadline and in the (b-effective) rather than the literal
reading. It is recorded, prominently, at `DECISIONS_NEEDED.md:3836` as
*"an implementation record, not a resolution"*, with the `DECIDE:` block
untouched, the measured table that chose the reading, the limitation (only Fable
has a per-model line, so the all-exhausted abort is currently unreachable), and
the one-line reversal. That is the correct shape.

**Three of the four defects the 41st audit found in the defaults are repaired:**

- **D8 — repaired.** Firing it would have taken `coverage` CLAIM-DEAD on
  `balance` (0 → 1). `BA.03` was registered on 08-30 (`1bf1eac`) explicitly
  *"BEFORE D8 fires, so parking BA.02 costs the ratchet nothing"*. `coverage`
  today reads `balance … claims: BA.02 RUNNABLE, BA.03 RUNNABLE`, and
  `decisions --check` prints no `SAFETY-CLAIM-DEAD`. Confirmed by running it.
- **The clause itself — partly repaired.** `decisions.py` now *computes* one of
  SYSTEM.md's three safety clauses instead of trusting the author's prose, and
  `SYSTEM.md` was corrected in the same window to say plainly that the other two
  remain enforced by nobody. A governing document that named an enforcement it
  did not have now says so. That is the right direction.
- **D3 / D4 — unchanged, and I am not re-ranking them.** Both are narrowings
  measured against practice rather than permission. The 41st audit named the
  shape (*"an escalation ignored long enough becomes a default that legalises the
  thing that was escalated"*) and it stands; neither widens what is permitted
  relative to what the loop is already doing, which is the clause that governs.

**`D10` is not repaired, and it is the one with a name on it.** Its default seats
`wm-latent` on the learning-core seat **BY VERDICT** — the strongest marking in
`CHAMPIONS.md` — off `LC.03`, whose ledger status is `VOID`. `SYSTEM.md`:
*"VOID: an arm failed the learning gate; fix the arm, do not decide"* and
*"two non-learners cannot arbitrate an architecture"*. After firing,
`champions --check` will print `Learning core … BY VERDICT ok`, because it reads
the marking and cannot ask whether the verdict was earned.

**In fairness to the entry, three mitigations are real and belong here:** a
CHAMPIONS seat is explicitly *"a CHAMPION, not a constitution"* and is unseated
by any winning challenger; `LC.04`–`LC.06` exist in the registry as live arena;
and the default itself keeps the owner's scale-transfer guard binding **before
adoption**, so this seats a champion, it does not adopt a core. The defect is
therefore narrower than "a VOID becomes a capability claim" — it is that the
*label* will overstate what was measured, on a seat whose only reader is a tool
that cannot see the difference. Precedent that this matters: the `World` seat is
already held `BY VERDICT` with a rematch trigger that pointed at seven specs
nobody had written.

Appended to `DECISIONS_NEEDED.md` this audit as
**`D10 — UNREPAIRED WITH HOURS TO GO (52nd overseer audit)`**.

**And the ownership gap the 41st audit named is still open, one day out.** Nine
of the eleven defaults require a write the overseer may not make (a spec, a
script, a registry entry, `DECISIONS_RESOLVED.md`). Nothing in `scripts/` runs
`experiments.decisions` at all — `grep -rn decisions scripts/*.sh` returns
nothing — so the deadlines are read only by an agent that happens to be prompted
to read them. The builder's prompt does name `run decisions --check` and knows
`D1`/`D10` are armed for today, so this is not unowned; it is unassigned. Tomorrow
it becomes eleven-at-once.

## 7. Bakeoff hygiene — NO FINDINGS

`DECISIONS_RESOLVED.md` holds three entries, re-checked in full:

- **`PS.01/J` — VOID**, recorded as VOID and used to decide nothing: all arms
  below the 3.0σ learning gate. A VOID correctly refused a verdict.
- **`PS.01/J2` — WINNER `impact_speed`**, 10.32σ over null, beating the runner-up
  by **2.66σ** — outside the noise margin. The `screen` gate mode carries a
  written rationale for why deterministic observables are not learners, and names
  the `T2.02` ambiguity it is exempt from.
- **`D2` — WINNER BLOCK**, decided by ledger replay rather than `run_bakeoff`,
  with the method justified (no seed noise, no null, no training that could have
  failed), exposure measured at 9 vs 0, the loser recorded with what survives of
  it, and a re-open trigger naming the exact quantity the trade rests on.

No decision inside its noise margin. No VOID promoted to a verdict. No decision
made without a learning gate where one applies.

## 8. The honest summary — closer to a creature, or just to more green ticks?

**Closer to a creature. `LG.00` is the reason, and it is the best single result
this project has produced.**

GOAL.md's test for whether Jack is a costume is: *"strip the diary and the
learned core, and his answers about his own life must COLLAPSE — while his
general knowledge survives untouched. He should be smarter inside his life and
dumber outside it."* That is now measured, on a probe set that was itself
certified first (`LG.01`, so the questions are provably about *his* life and not
answerable from the world): **0.739 vs a 0.271 null inside his life, 0.533 vs a
0.733 null outside it.** Both directions, in one run. The asymmetry GOAL.md names
as *"the proof he is a creature and not a costume"* exists as a number.

**And the honest counterweight is that the ladder is now bottlenecked on
something no spec can fix.** Seven independent instruments now say W0 is too
shallow to grade a learner — `LC.03`'s darkroom, `LC.03 v2`'s one-learner-in-five,
`DP.05`'s FAIL, `SH.01`'s `ORACLE_CANNOT`, `SH.02`'s saturated headroom, `W.1`
and `W.2`'s measured physics, and now `DP.04`'s lifespan quantisation. Four of the
five gate-provisional specs are pilot-blocked on it; both empty GPU cost classes
are unfillable because of it; curiosity sits at 2 passing of 12 and unison at 1 of
22 because a world with no consequences cannot grade an explorer.

**The repair for all of it is one design decision, it is owed by the Review, and
the Review has been dead since Saturday morning without anyone noticing.** That
is why this audit reads DRIFTING and not ON TRACK. The builder is doing
everything right — twenty-two clean iterations, seven new PASS, two voluntary
strengthenings, a refutation of its own pilot's premise, and a refusal to edit a
dependency an hour after it produced an inconvenient FAIL. It is doing it while
the one desk that can unblock it is not open, and no instrument in this system
can print that sentence.

We are closer to a creature. We are also, for the third consecutive week, about
to let ~29 free GPU-hours expire — not because the loop is asleep, but because
the shelf is bare and the person who restocks it never came in.

---

## FOR THE BUILDER

**B1 — RANK 1. Nothing watches whether the Review ran. Build the schedule-side
liveness check, because the artifact-side one provably cannot see this failure.**

The scar, precisely: `review.sh` started `2026-08-30T06:37:03` in **FULL** mode,
died at `Error: Reached max turns (60)` at `06:48:03` — **11 minutes into a
40-minute `TMOUT`** — and wrote nothing. `seal_output` behaved correctly and did
nothing, because `lib_seal.sh` requires the output file to be *dirty*
(`git status --porcelain -- "$file"`) and a run that dies before writing leaves it
clean. So the seal covers death-after-writing and this was death-before-writing:
**one scar's repair assumed the other scar's opposite, for the second time in the
same file.**

Consequences to state in whatever you write: `docs/PROGRESS.md` still presents
2026-08-29 data as current state (`84/187`, *"fifth consecutive day on which not
one figure has moved"*) while the ladder is `92/200`; `PROGRESS_LOG.md`'s last row
is 08-29; Part 2 (the test re-examination) is **Sundays only** and is now missed
until 2026-09-06; and `w0-too-shallow`'s design, owed by that exact run, is still
`OPEN`.

Two repairs, both cheap, and take **both**:

(a) **A liveness assertion keyed to the SCHEDULE, not to the file.** Something —
the overseer's own script is the natural home, since it runs 4×/day and takes no
lock — must assert *"`PROGRESS_LOG.md` gained a row within the last 25 h"* and
*"the last row within 8 days has `mode FULL`"*, and shout when it has not. The
27th audit already wrote the rule as a corollary and it was never built:
*"an organ that is the destination of routed work must have liveness watched by
something other than itself."*

(b) **Extend the seal to the clean-file case.** If a run exits non-zero and its
output file is *clean*, that file is now **stale** — stamp a one-line banner on
it naming the run that failed and the date, and commit it. A current-state page
that silently describes a two-day-old world is the same disease as the 49th
audit's uncommitted draft, in the other direction.

**And fix the cause while you are there:** `--max-turns 60` is hard-coded
identically for `DAILY` (`TMOUT=20m`) and `FULL` (`TMOUT=40m`) at
`review.sh:42/48/57`. The mode that does twice the work gets twice the clock and
the same turn budget, so **FULL is the mode structurally most likely to die this
way** — and it is the mode that owns the world redesign. Scale `--max-turns` with
`MODE`. `overseer.sh` has the same shape (`--max-turns 60` under `timeout 25m`)
and has died this way twice; there have been **7 max-turns deaths across the
three organs** (`ladder` 4, `overseer` 2, `review` 1).

**B2 — RANK 2. `SYSTEM.md`'s "leave no process running" is enforced by nothing.
Add the check to `ladder_loop.sh`'s exit path.**

I found and killed PID 3749514 — `python -c "x=0 / while 1: x+=1"`, orphaned to
`ppid 1`, `cwd=/home/opc/jackthelearner`, started 23:24:47, **1.26 core-hours**
of a 4-core box shared with paying tenants, at `nice 19`. It was a verification
aid for the `_cpu_fraction` fix in `1296ca3` and nothing in the system could see
it: `tmp_reaper.sh` reaps scratch *directories* and explicitly avoids processes;
`ladder_loop.sh` has no process check on any exit path; and the hygiene claim
that appears in most iteration reports (*"no leftover compute — the only `pgrep`
match is the grep's own shell"*) is **voluntary prose**. Both iterations that
straddled this one omitted that sentence, which is honest and is also exactly why
prose cannot be the guard.

The sharpest part: the 00:07 iteration printed a **full `ps` dump** to prove
`BA.03` was alive, and did not notice a second `/data/venvs/jackthelearner`
python at 99.7% CPU in the same output. **A liveness check that scans for one
known pid cannot see an unknown one** — it is looking for presence, not for
excess.

Concretely: at the end of `harvest_bookkeeping`/the exit path, snapshot
`pgrep -u opc -f '/data/venvs/jackthelearner'` before `run_claude` and after,
log any pid present in the second set and absent from the first that is not the
declared detached run, and refuse to log `iteration end rc=0` silently when one
exists. Do not auto-kill a pid you cannot attribute — a detached registered run
is legitimate and must survive; name it and let the next reader decide.

**B3 — RANK 3. Re-run `T0.21` and `T0.29` the moment `BA.03` frees the lock
(~05:15Z).** They certify `coverage.py` and `champions.py`, both of which you
changed after their last run, and both of which the overseer is required to
trust before anything else in the audit. Your offline replay of `T0.29`'s ten
properties (0 failed) is good practice and is not a ledger row. `T0.17` is stale
in the same window. This is bookkeeping, it is owed, and it is rank 3 only
because the lock makes it impossible until then.

**B4 — RANK 4, and it is a reporting gap not a defect.** Nothing in this repo
can print *"6 rows OPEN in `REVIEW_QUEUE.md`, oldest 7 days, consumer last ran
2 days ago"*. The queue file was built (27th audit B2) so the backlog would stop
being invisible; it succeeded at holding the rows and never gained a `--check`.
Given that four of five gate-provisional specs and both empty GPU classes now
route their repair here, a `grep '^ROUTED:'`-shaped counter with an age column
and the consumer's last-run date is the single highest-leverage instrument you
could add this week. Scar: `w0-too-shallow` was due from a run that did not
happen and no number anywhere went red.

**B5 — RANK 5, carried from the 49th (B6a), 50th (B6), 51st (B5).** `T0.01` and
`T0.10` are the only `PASS` specs with no declared control. Add a sentence to each
spec's `control` field so the absence reads as a decision. I re-verified both are
honest. One line each.

**B6 — RANK 6, carried from the 50th (B4) and 51st (B6), unserved.**
`44f24c41`'s claim that `T2.09`'s seed-selection formula *"reads only the null
and the rig instruments"* is false as a summary — `t2_09_*.py:583-589` gates on
`claim_static_reward_q1`, `claim_static_decay` and `exposure_frac_of_random`.
**Live effect is zero** (all three exclusions fired on `trap_dwell`), so **do not
move `DECAY_MIN`**; re-fitting it now would be the real violation. Fix the
*sentence*. And cap or contextualise seed 1's `trap_ratio` of
**953,594,661,617.28** — a vanished denominator, not a spectacular trap.

## FOR THE OWNER

**1. One decision is worth your time before tomorrow, and it is `D10`.**

You do not need to know what `wm-latent` is. The facts:

> `LC.03` — the bakeoff that was supposed to choose Jack's learning core — ran
> and returned **VOID**: only one of its arms demonstrably learned, and this
> project's own rule is *"two non-learners cannot arbitrate an architecture; fix
> the arm, do not decide."*
>
> `D10`'s pre-registered default, which fires tomorrow if you say nothing, seats
> that single surviving arm as the learning core **"BY VERDICT"** — the strongest
> marking the system has. Not "provisionally", not "by default". By verdict, off
> a run that refused to give one.

**What it does not cost you:** the seat is a champion, not a constitution — any
future arm that beats it takes it, and `LC.04`–`LC.06` already exist as the ring.
The default also keeps your scale-transfer guard binding before anything is
actually *adopted*. So this is reversible and it is not an adoption.

**What it does cost:** every instrument downstream will read the seat as settled
by measurement, because the tool that checks seats reads the label and cannot ask
whether the verdict was earned. Your own standing rule is that this project
claims nothing it has not measured.

**If you want it seated with an honest label**, one line — *"seat it, mark it
BY DEFAULT not BY VERDICT"* — and the loop does exactly that; nothing else in the
default changes. **If you want the arm fixed first**, say so and `LC.03` stays
CONCLUDED with 8 specs blocked behind it, which is the cost you would be
accepting. **If you say nothing, the default fires as written**, which is the
point of arming it, and I am not going to pretend otherwise. Evidence appended to
`docs/DECISIONS_NEEDED.md` under `D10 — UNREPAIRED WITH HOURS TO GO`.

**2. Ten other defaults fire tomorrow and I am not asking you about any of them.**
`D1` is the largest (38 specs) and its default **changes nothing you decreed** —
PLASTIC-ONLY stands verbatim, the option that would have narrowed it is struck as
unconstitutional, and the remaining arms go to a bakeoff the loop runs itself.
`D8`'s hazard was found and repaired by the builder before the deadline (it wrote
the successor spec `BA.03` so that parking `BA.02` costs the coverage ratchet
nothing — that is exactly the conduct this system is supposed to produce).
`D14` is already built and running. You only need to act on `D1` if you want the
decree *narrowed*, which is the one branch no experiment may take for you.

**3. A process failure you should know about, because it is costing you free
compute and it is not the loop's fault.**

The **Review** — the weekly desk that owns world-design decisions — has not
produced anything since Saturday morning. Its most important run of the week,
the Sunday full review, started and died after eleven minutes on a technical
limit, having written nothing, and **no part of this system noticed for two
days.** The page you would open to see project status,
`docs/PROGRESS.md`, still shows Friday's numbers and says the ladder has not
moved in five days. It has moved from 85 to 92 since then.

This matters beyond tidiness because **that desk is now the only thing standing
between the project and its bottleneck.** Seven separate measurements now say
Jack's world is too shallow to teach him — the cold does not kill, the needs do
not bite, lifespans do not vary enough to measure a decision. Four experiments
are frozen waiting on a redesign of that world, both GPU work-queues are empty
because of it, and ~29 hours of free Kaggle GPU time will expire on Saturday for
the third consecutive week. All of that unblocks on one design decision that was
due from the run that died.

**Nothing is asked of you** — the repair is builder work and it is ranked B1
above. I am telling you because "the ladder has not moved" and "the desk that
unblocks the ladder has not opened" look identical from the outside, and only one
of them is true.
