# OVERSIGHT.md — the overseer's current-state report

> Rewritten every audit. Current state, not a log. The running history is the
> git log of this file.
>
> **90th audit — 2026-09-12, 18:37–19:0x UTC.** Read at HEAD `a6bc589`. **The
> builder woke at 17:07 after 104 dark slots and is STILL RUNNING as I write**
> (`ladder_loop.sh` pid 3659175, started 18:07, 14 commits since 17:09, two test
> files dirty in the tree at the moment I read them). Every number below is a
> snapshot of a moving tree and says so where it matters.

## VERDICT: ON TRACK

And the qualifier is the whole finding: **on track on the MACHINE, stalled on
the LADDER.** `demonstrated` has read 108/245 for five consecutive days. Today's
entire output was meta-work — five armed defaults fired, two counters recorded,
one gate tightened, one spec re-run to a second VOID. Under SYSTEM.md that is a
complete day's work (*"any session that makes the machine better at catching its
own errors has done the whole job even if no spec passed"*), and it earned that
reading honestly: within 30 minutes of implementing a Review ruling the builder
**measured that the ruling's stated mechanism was false**, kept the tightening
anyway because the one law binds in one direction, and routed the refutation.
That is the loop running on itself, correctly, without being asked.

**I found no integrity fault.** Section 1 is clean on all 108 PASS rows; section
2 is clean across seven days and 25 commits. What I found are **two instrument
blind spots** — places where a live tool prints a number that is true of its
input and false about the world — and one **re-measurement** of a known,
routed, disclosed weakness that nobody had ever actually counted.

---

## RANK 1 — `LG.03` is now un-verdictable at its registered envelope on TWO independent gates, it carries no foreclosure declaration, and so `coverage` still advertises it as the sole dispatchable unit in `cpu<10min`

**The evidence is determinism, and it is exact.** Attempt 1 (`a2ff63c`,
2026-09-04) and attempt 2 (`1bd42dc`, today 18:23) ran the same three seeds
0/1/2 through the same `_experiment`. Compare the two rows in `ledger.json`
(attempt 1 is preserved in `history[0]`):

| metric | attempt 1 | attempt 2 |
|---|---|---|
| `blind_calib_rate` | 0.583333 ± 0.311805 | **0.583333 ± 0.311805** |
| `planner_reach_mean` | 0.754167 ± 0.0212459 | **0.754167 ± 0.0212459** |
| `stripped_both_rate` (control) | 0.0541667 ± 0.0117851 | **0.0541667 ± 0.0117851** |
| `excluded_seed0_only` | 13-cell string | **byte-identical** |

Every shared field reproduces to the last digit. The only new fields are the two
the amendment added. **This spec is deterministic at fixed seeds**, so a third
dispatch at this envelope buys the identical VOID for another ~727 CPU-s.

And it now VOIDs on **two** conjuncts, not one, and the builder's own probe
(`75a5544`, 18:40) measured the per-seed join that proves both are closed:

| seed | `planner_calib_reach` | `blind_calib_rate` |
|---|---|---|
| 0 | 1.00 | **0.50** |
| 1 | 0.75 | 1.00 |
| 2 | 0.75 | 0.25 |

- Gate 1, `planner_calib_reach` ≥ `PLANNER_CALIB_MIN` 1.0: mean 0.8333. Closed.
- Gate 2, `blind_calib_rate` ≥ `CALIB_MIN` 0.75: mean 0.5833. Closed. **And
  seed 0 — the one seed whose teacher is perfect, i.e. the only seed that
  survives gate 1 — reads the twin at 0.50.** So "fix the venue and the
  liveness proof arrives" is refuted by the run's own numbers.

**Nothing in this is a violation and I am not calling it one.** The tightening
is real, the cost was stated up front in the commit message (*"expected to VOID
on MORE seeds, not fewer"*), the run was committed as found, and the builder
routed the mechanism refutation itself. **The defect is that the repair class is
not DECLARED, so three instruments read it wrong.**

`protocol.void_foreclosed`'s own docstring names this exact misroute: the
`queue_depth` void list is printed as *"an arm to repair, not a dispatch"*,
which it calls *"the CHEAP reading, and it was wrong for two of its five
members."* LG.03 is now the third.

**I tested the counterfactual read-only rather than asserting it** (monkeypatched
`void_foreclosed` in memory; no file touched):

- `coverage.queue_depth()` drops **7 → 6**, LG.03 leaves the repairable-VOID
  list, and **`cpu<10min` loses its only occupant** and reads EMPTY — which is
  the honest state of that cost class today.
- `champions --check` moves the **Language grounding (word → lived skill)** seat
  onto the *"seats no one can ever WIN — every pending arena member welded"*
  list, where it is not today. Its arena is `LG.04, LG.05, LG.06`
  (`CHAMPIONS.md:320`) and all three `depends_on` LG.03, so the whole grounding
  arena is welded behind a spec that cannot reach a verdict.
  **Stated against my own first guess: no champions ratchet counter moves** —
  the seat is `HELD: UNDECIDED`, so there is no unearned holder to indict and it
  is scoped out of `ARENA-UNREACHABLE`. This is a visibility repair, not a
  ratchet repair, and I say so because I expected the opposite and checked.

The declaration is well-formed and cheap: `FORECLOSURE ARITHMETIC:` is the two
per-seed vectors above; `BLAST RADIUS:` is LG.04/LG.05/LG.06. It goes through
`run amend LG.03 --doc-only`, so no certificate stales. Routed as **B1**.

---

## RANK 2 — the sweep `aggregate-hides-worst-seed` asked for has now been run across all 108 PASS rows, and the answer is ONE standing exception

`docs/REVIEW_QUEUE.md:1170` (`aggregate-hides-worst-seed`, OPEN 13 days, DUE
2026-09-18) says the population that could carry the mean-hides-worst-seed bug
is large — *"26 spec files fold a `worst`/`_lo`/`_hi` quantity and 89 lines read
a `_std`, so the population that could carry this bug is large and nothing
mechanical distinguishes a correct gate from a wrong one."* Its option (c) is a
static audit of every `_check`. **Nobody had run it.** `docs/LESSONS.md:6660`
even gives the mechanical tell: *"a gated metric with `_std > 0` compared
straight to a bar was met by a mean."*

I ran it, read-only, from the ledger and the ASTs: for every multi-seed PASS
row, every `_check` comparison of `m["k"]` against a resolvable numeric
constant, where the recorded `k_std > 0`, bounded by the exact n=3 extreme-value
result `|x_i − μ| ≤ σ√2` the T3.06 row already derived.

**The result, and it is mostly good news:**

| spec | conjunct | mean ± std | bar | worst admissible seed | reading |
|---|---|---|---|---|---|
| **PG.4** | `icm_dwell_share` | 0.666667 ± 0.471405 | ≥ 0.40 | **0.0** | **BREACHED, provably** |
| **PG.4** | `dwell_margin` | 0.605267 ± 0.445437 | ≥ 0.25 | ≤ 0 | **BREACHED** |
| **PG.4** | `panel_reward_ratio` | 6.411e8 ± 4.534e8 | ≥ 2.0 | ≈ 0 | **BREACHED** |
| **PG.4** | `rays_on_panel_while_dwelling` | 7.4767 ± 5.2868 | > 0 | 0 | **BREACHED** |
| LG.01 | `retained_min_per_category` | 23 ± 2.1602 | ≥ 20 | 20 or 21 | safe, by integer arithmetic |
| ME.10 | `skill_gain` | 0.37037 ± 0.094423 | ≥ 0.25 | 0.2368 | **unresolvable from the record** |
| T2.08 | `coverage_margin` | 0.0544 ± 0.0187 | ≥ 0.05 | — | guarded: `margin_floor` + paired t-stat (idiom 3) |
| PS.02, VO.01 | several | — | — | — | guarded: `seed_gates_ok == 1.0` (idiom 2) |
| ME.9, PG.8, T2.03, T2.14, T2.19, T3.01, W0.DIAG | 13 conjuncts | — | — | — | safe: `_std = 0.0`, or bound clears the bar |

Two hits I discarded as **my own parse artefacts**, named so nobody re-finds
them: `PL.00 physics_travel` and `W0.DIAG jit_delta_up` are `if m[k] < BAR:
return VOID` guards, which my AST reader took with the sense inverted.

**The PG.4 proof, because "provably" is a strong word.** `icm_dwell_share` is a
share in [0,1]. From mean 2/3 and population std √2/3: Σx = 2 and Σx² = 2. On
[0,1], x² ≤ x with equality only at 0 and 1, so Σx² = Σx forces **every seed to
be exactly 0 or 1** — the vector is `{1, 1, 0}`, uniquely. `rays_on_panel`'s
std/mean is 0.70711 = √2/2 exactly, the `{a, a, 0}` signature on the same seed;
`panel_reward_ratio` solves to `{9.617e8, 9.617e8, ≈0}`. So on one of PG.4's
three seeds, **four of its five experiment conjuncts fail** and the row says
PASS.

**And this was DISCLOSED — I am reproducing, not discovering.** The PG.4 commit
(`4a4afb3`, 2026-08-10) ends: *"Honest caveat: per-seed dwell is (1.0, 1.0, 0.0)
— one seed never discovered the panel in its 20k-step life; PASS is on the
pre-registered aggregate-mean protocol uniform across the ladder. CU.3 should
use dwell distributions, not means."* `LESSONS.md:6630` names the same vector.
Nothing was hidden and nothing was cheated.

**What IS a finding is where the caveat lives.** It lives in a git commit
message from 33 days ago and in a lesson. It does not live on the ledger row,
and `run status` prints `[PASS] PG.4` with no mark. PG.4 is apparatus under
T2.08 (curiosity's only headline PASS) and is `depends_on` by **CU.3, LT.01,
LT.02, LT.03** — and LT.03/LT.04 are the *Curiosity signal* arena in
`CHAMPIONS.md`. The one commitment GOAL.md calls the north star rests on a
fixture whose trap demonstrably did not fire on a third of its seeds.

**What this buys the routed row.** It re-prices its own three arms with a
number it did not have: arm (b) — make `_aggregate` refuse to flatten
worst-case keys — was costed as *"the one that will break existing specs, which
is the point and the cost."* **The measured cost is one spec: PG.4.** ME.10 is
the only other row where the record cannot answer, and that is a one-run
question. Routed as **B2**; I am attaching evidence to the Review's row, not
opening a competing one.

---

## RANK 3 — the blackout ended, and the thing it was blocking cleared in 16 minutes

104 consecutive skipped slots (2026-09-08T08:23 → 2026-09-12T16:07), ended
17:07 when `week:all models` 76% fell under a pace line of 77%. The builder
counted the streak itself from `ladder.log` rather than inheriting either
oversight desk's stale 94 — which is the measurement five audits and two Reviews
asked for, and it corrected both of us.

**The overdue-armed-default queue is EMPTY for the first time since 2026-09-08.**
All five cleared between 17:09 and 17:25: `D22`'s record completed, `D18`,
`D24`, `D23`, `D26` fired. `decisions --check` now reads *3 armed, 0 overdue,
ratchet ok (0/10 undeclared, 0/3 unrouted-owner-ask, 0/0 vanished-owner-ask)*.

Two of those firings are worth recording because they show the mechanism working
better than its own text:

- **`D18` fired on a premise that was already false.** Both code changes it
  ordered had landed on 09-03 in `a071d91`, before the entry was armed, and the
  shipped code was *stronger* than the order. What actually fired was the
  never-executed second word, REPORT: 8 of 69 own-peak rows breach the ceiling,
  median 6.4× *under* it. `D18` posed a binary from one sample; executing its
  literal words would have been wrong about 61 of 69 rows. The builder wrote the
  lesson (`726a5fc`): *an armed default's text is a snapshot of the world when it
  was ARMED.*
- **`D26` option (i) did not fire** — the option both desks recommend, and the
  one that would have released the builder sooner — because a default may not
  loosen a gate. The builder took the narrower (iv). That is the safety clause
  working against the builder's own interest, unsupervised.

---

## The audit, section by section

### 1. Integrity of the ledger — NO FINDINGS

108 PASS rows. Checked mechanically, not by sampling:

- **Commits:** all 108 `commit` fields resolve with `git cat-file -e
  <sha>^{commit}`. Zero dangling.
- **Controls declared:** all 108 specs declare a non-empty `control` in the
  registry. Zero undeclared.
- **Controls RUN:** 106 of 108 carry non-empty `control_metrics`. The two that
  do not are `T0.01` (repo imports clean) and `T0.10` (Kaggle round-trip), and
  both registry entries read `"NONE, BY DECISION (52nd audit B5)"` with the
  reason — an import either raises or it does not; there is no sabotage whose
  survival would prove the test vacuous. Correctly dispositioned, no finding.
- `run status` reports the standing staleness set unchanged: 5 rows with a moved
  `impl_sha` (D1.0, XL.01, LG.10, LF.01, SO.07 — all FAIL/VOID, none a PASS),
  1 pre-`impl_sha` content-stale (T2.02, VOID), 10 PASS rows predating
  `spec_sha`. All printed, none silent.

### 2. Thresholds and controls over seven days — NO FINDINGS

25 commits touched `registry.py`, `registry_expansion.py` or `experiments/tests/`.
Every numeric constant that moved, and its direction:

| constant | move | direction | justified? |
|---|---|---|---|
| `T0.31 N_PROPERTIES` | 15 → 16 → 17 | **tighten** (properties added) | yes, red-first receipts in both commits |
| `ME.9 MIN_DISTRACTOR_EVAL` | 9 → 12 | **tighten** | yes, arithmetic at the constant (80% of a 15-pair pool) |
| `LG.03 PLANNER_CALIB_MIN` | new, 1.0 | **tighten** (new VOID conjunct) | yes, cost stated as "more VOIDs, not fewer" |
| `ME.1 distractor_abstention` | conjunct replaced by per-tag conjuncts | **tighten** | yes |

No threshold moved in the loosening direction. No control was deleted or
weakened. The only `_check` that gained an `or` is `if m["canary_ok"] < 1.0 or
m["learn_ok"] < 1.0` — an `or` inside a **VOID guard**, which adds ways to void,
not ways to pass. No seed count was reduced. The one `-` line matching a deleted
assertion (`and m["distractor_abstention"] >= MIN_ABSTENTION`) was replaced in
the same hunk by a strictly stronger per-tag form.

**Saying it plainly because the brief asks for it: section 2 is clean, and that
is a real result on a week in which a gate was edited and re-run.**

### 3. Drift from the goal — none, and the converse is the problem

Everything the builder did in the last 24 h traces:

| work | GOAL.md / SYSTEM.md sentence it serves |
|---|---|
| D22/D18/D24/D23/D26 fired | SYSTEM.md rule 3: *"an escalation may not deadlock"* |
| `pace_gate` attribution instrument (D26) | *"free compute only"* — honest accounting of the binding resource |
| `FAIL-OWNED-BUT-UNDRAINED` counter (D23) | *"a red ladder that tells the truth is worth more than a green one that does not"* |
| LG.03 gate tightening + re-run | *"the command cells are language-necessary"* → GOAL.md *"he learns words the way every child does"* |
| the mechanism refutation + lesson | SYSTEM.md: *"the loop mutates the system that hosts it"* |
| `review_queue` counters recorded | the routed-work ratchet |

**Zero drift. The converse is where the damage is.** `coverage` exits `rc=2`:

- **0 commitments with NO declared spec** — the 2026-08-10 hole stays closed.
- **4 CLAIM-DEAD** (every claim spec parked or foreclosed): **smell**, **balance**,
  **shelter/building**, **thermal (kills)**. Two of those are the owner's own
  words. Unchanged for nine days. Routed as
  `five-commitments-are-claim-dead-behind-foreclosures` (DUE 09-16).
- **9 commitments with live claim specs and nothing passing**, including
  touch, tool use, told world, proprioception, sleep, hunger/thirst, death &
  retry, fast/slow.
- **`fast/slow`: 8 specs, 0 passing, 4 welded behind `LC.03`, 1 foreclosed.**
- **4 NEW unrunnable GOAL.md citations** — GEN.02, GEN.03, GEN.06, GEN.09, all
  `welded<-LC.07`. GOAL.md's present tense cites specs that resolve to corpses.

Curiosity, all-senses fusion and learning-by-living are exactly the three the
brief warns are quietly neglected, and the reading is: curiosity has 2 passing
of 12 (and RANK 2 is about one of them); one-brain/unison has **1 passing of
27**; memory-across-lives has 1 of 9.

### 4. Is the builder alive and productive? — alive, newly

24 hourly slots in the last 24 h. **23 skipped by `pace_gate`; 1 executed.**

- 17:07 → `rc=0` at 17:26, `108 → 108 demonstrated`, 9 commits.
- 18:07 → **still running at the time of this read** (38 min in), 5 commits so
  far, two test files dirty in the tree.
- Model: `week:Fable` pinned at 100% against `MODEL_FLOOR=95`, so both slots
  refused Fable and walked the chain to **Opus**, billed to the shared meter.
  The builder recorded that substitution this time; four audits had asked.

PASS delta over 24 h: **0**. Over five days: **0**. `net demonstrated` has read
0 for five consecutive days. Not a fault today — the ladder's only fresh
dispatch lane is one spec deep (see RANK 1) — but it is the number section 8
has to answer to.

### 5. Compute honesty — one loss, correctly taken

- **GPU, `2026-W36`:** 17.7238 h charged of 30. **12.2762 h expire tonight,
  unspent.** No Kaggle job has been charged since 09-08. This is the loss the
  89th audit forecast and it is the correct outcome: attempt 3 of `D1.0`
  measures 17.61 h and does not fit, and its precondition (twin-spread result on
  the row → successor gate in a non-dispatch commit → dispatch) is still
  unwalked. **The builder had four hours awake tonight and did not scrape it.**
  `2026-W37` opens tomorrow with a fresh 30 h.
- **CPU today:** 747.49 s — `LG.03` 728.86 (the attempt-2 re-run), `T0.21`
  18.63 (a certificate re-buy). Billed, attributed, in `cpu_budget.json`.
- **The standing line, unmoved:** `gpu_hours_no_verdict` TOTAL **48.07 h**, of
  which **`D1.0` 33.78 h across 2 attempts for 0 ledger verdicts** — still the
  project's single largest compute line — and **6.32 h across 21 UNATTRIBUTED
  jobs**, at its declared floor of 21.

### 6. Stuck decisions — 3 armed, 0 overdue, 0 MEANS-ESCALATED

First empty overdue queue since 2026-09-08. Live:

| id | decide_by | costs | note |
|---|---|---|---|
| **D25** | **2026-09-13 — TOMORROW** | 0 specs | `lib_seal.sh` banners a Sunday FULL that committed everything with the same "UNVERIFIED" it gives one that committed nothing. Default (iii) FIX THE SEAL is the only legal one of the three and is monotone. |
| D19 | 2026-09-14 | 3 specs | blocks HR.1–HR.4; default NO FETCH |
| D20 | 2026-09-18 | 0 specs | default (i) WALL STANDS |

`decisions --check` EXIT 0, ratchet ok on all four classes. **No
`MEANS-ESCALATED` — nothing that a measurement could settle is on the owner's
desk.** And the D25 entry records something worth keeping: it reached that desk
only after `decisions.py` **refused** an earlier D25 as `MEANS-ESCALATED` and
the builder ran the bakeoff instead (`ab857b1`). The D1 disease has an
antibody now and it has been observed firing.

**Nothing acted on without record.** I checked the five firings against
`DECISIONS_NEEDED.md`: each carries the required wording *"the owner did not
rule by &lt;date&gt;, so the pre-registered default fired"* and a reversal.

### 7. Bakeoff hygiene — one standing red, correctly printed

`champions --check` EXIT 0, ratchet ok on all five classes. The one entry that
matters here is **`Learning core` held BY VERDICT off `LC.03 = VOID`** — a VOID
treated as a verdict, which section 7 asks about by name. It is **not hidden**:
it is counted in `UNVERIFIED VERDICTS (2/2)`, it is counted in `TRIGGER DEBT
(3/3)`, `D10`'s resolution carries the single-arm caveat on its face, and the
seat's arena was labelled `VENUE-UNAFFORDABLE` (~526 h against 30 h/week) by
`D24` firing four hours ago. A verdict this thin being visible in three
counters is the machine working. **It is still a seat held by a run that did not
test the claim, and it should not become normal.**

Also standing and printed: 2/3 `UNFALSIFIABLE` seats (ASR, Speaker ID), 1
`ARENA-UNREACHABLE` (Fast/slow coupling, rooted at LC.03), 1
`KINDLESS DISCHARGE` (LF.02).

### 8. The honest summary

**Are we closer to a curious humanoid that climbs the ladder than yesterday?
No. Are we closer to a longer list of green ticks? Also no — the list did not
move either.** Five days at 108/245, and today's work produced one more VOID.

**But this was not a wasted day, and saying it was would be the dishonest
answer.** The system did the thing it exists to do: a ruling came down from one
organ, a second organ implemented it, measured it, **found the ruling's stated
mechanism was false — traced to a swapped pair of seed labels in a probe
docstring** — kept the repair because it was a tightening and the one law binds
one way, refused to remove the conjunct that its own finding weakened the
argument for, and routed the refutation with a lesson. No human asked for any of
that. On the ladder-and-apple standard we did not move; on the standard that
makes the ladder worth trusting, we did.

**The thing I would say to the owner if I could say one sentence:** the machine
is now measurably better at catching itself than the ladder is at growing, and
the binding constraint on Jack is no longer honesty — it is that **four of your
constitutional commitments have no living falsifiable claim, and the fresh
dispatch queue is one spec deep.** More instrument is not what buys the next
rung.

---

## FOR THE BUILDER

**B1 — Declare `LG.03`'s foreclosure. Highest priority, ~10 minutes, no
compute, no certificate staled.** (RANK 1.) Add to the module docstring at the
left margin, via `run amend LG.03 --doc-only`:

```
VOID-FORECLOSED: both gates are closed at this envelope and the run is
    deterministic — attempt 1 (a2ff63c) and attempt 2 (1bd42dc) reproduce every
    shared metric to the last digit. A third dispatch buys the identical VOID.
FORECLOSURE ARITHMETIC: per-seed, measured by lg03_blind_twin_probe.py 2026-09-12 —
    planner_calib_reach {1.00, 0.75, 0.75} vs PLANNER_CALIB_MIN 1.0 (mean 0.8333),
    and blind_calib_rate {0.50, 1.00, 0.25} vs CALIB_MIN 0.75 (mean 0.5833).
    Seed 0 is the ONLY seed that survives gate 1, and its twin reads 0.50 — so
    repairing the venue does not deliver the liveness proof. No multiplier on
    seeds or steps moves either: the calibration cell is fixed at
    approach@sorted(objs)[0] and the servo's reach on it is a property of W0.
    The repair is a FIXTURE redesign, already named by the amendment itself.
BLAST RADIUS: LG.04, LG.05, LG.06 (all depends_on LG.03) — which is the entire
    declared ARENA of the CHAMPIONS.md `Language grounding (word -> lived skill)`
    seat, CHAMPIONS.md:320.
```

Verified read-only before ordering it: `coverage.queue_depth()` goes 7 → 6, LG.03
leaves the repairable-VOID list, and **`cpu<10min` correctly reads EMPTY**;
`champions --check` moves the Language-grounding seat onto the *"seats no one
can ever WIN"* list. **No ratchet counter moves in either tool** — I checked, and
I expected the champions ratchet to move and it does not. This is a visibility
repair. Route the fixture redesign onto the existing `w0-too-shallow` family,
not a new row; the amendment already names it as an observation-side instrument.

**B2 — Attach the sweep to `aggregate-hides-worst-seed` (DUE 09-18), and do not
open a second row.** (RANK 2.) The row asks for exactly this and has been OPEN
13 days without it. Record these three findings on the row:

1. The sweep is runnable today from the ledger alone and needs no new machinery:
   for each multi-seed PASS row, each `_check` comparison of `m["k"]` against a
   resolvable constant where `k_std > 0`, bounded by `|x_i − μ| ≤ σ√2` at n=3.
2. **The measured breakage cost of arm (b) is ONE spec: `PG.4`.** Six of the
   nine extremum-gated PASS specs record `_std = 0.0` (already folded); T2.08
   uses idiom 3, PS.02/VO.01 use idiom 2; LG.01 is safe by integer arithmetic.
   **`ME.10 skill_gain` (mean 0.37037 ± 0.094423, bar 0.25, worst admissible
   0.2368) is the one row the record cannot answer** — one re-run settles it.
3. `PG.4`'s `icm_dwell_share` is provably `{1, 1, 0}` from the row alone
   (Σx = Σx² = 2 on [0,1] forces every seed to 0 or 1), and `rays_on_panel`
   and `panel_reward_ratio` carry the `{a, a, 0}` signature on the same seed.
   **Four of five conjuncts fail on one seed and the row says PASS.**

**B3 — `PG.4`'s caveat belongs on the row, not only in `4a4afb3`.** Whatever
the Review decides about the recorder, the disclosure that exists today lives in
a commit message from 33 days ago; `run status` prints `[PASS] PG.4` unmarked
and four specs depend on it. This is the `T0.27` `DELIBERATELY-RED GATES` idiom
one notch over — a PASS that needs its reading printed beside it. Cheapest
honest version: name the per-seed vector in the module docstring so `coverage`
and any future reader see it without `git log`.

**B4 — `D1.0`'s twin-spread probe, and nothing ahead of it once B1–B3 are
done.** Carried unchanged and now six days untouched. Forward passes only, both
branches pre-registered on the row, no GPU quota needed, no owner ruling
pending: **frees 35 / blocks 38, the largest single unblock in the project.**
`W37` opens tomorrow with a fresh 30 h; the precondition still binds under every
branch — twin-spread result on the row → successor gate committed in a
**non-dispatch** commit → then a dispatch. An unchanged re-dispatch stays
forbidden.

**B5 — `PL.02`'s eye gate is RULED and still unimplemented** (`5e39771`, DUE
09-14, carried from the 89th audit B5 and the Review's item 3). Raw-pixel radius
ridge R² ≥ 0.80, `EYE_RADIUS_R2_MIN` unmoved, measured on the run's own probe
episodes; `r2_ua` stays a first-class recorded metric. Then a smoke; the
registered run stays blocked until the smoke passes.

**B6 — `W1.04` still gains conjunct (c) before you register it.** Carried
unchanged from 09-10 and now three audits old. Register from the amended design,
not the 09-06 text.

**Not repeated as items because you already did them today:** B1–B6 of the 89th
audit are discharged — five defaults fired, the `allow(elapsed=100)` endpoint
in the pacing line, `git commit --only` observed on every commit in this slot,
and both `review_queue` counters recorded in `f57eb81`. That is the first time
in four days a whole FOR THE BUILDER block cleared.

---

## FOR THE OWNER

**1. `D25` is due TOMORROW, 2026-09-13, and it is the only clock that is red
inside 24 hours.** It asks whether `lib_seal.sh` may go on telling a measured
falsehood: it reads only `rc != 0`, so a Sunday Review that committed its whole
page, its dispositions and its log row gets the same *"THIS IS A DRAFT, NOT A
FINDING … UNVERIFIED"* banner as one that committed nothing. The cost is already
realised — the builder spent a full day executing seven of nine items off a page
formally marked unverified, and was right to. **The default (iii) FIX THE SEAL,
BUY NOTHING is the only legal one of the three**: (i) spends credits against the
shared meter by silence, and that meter's exhaustion is what took every organ
dark for 4.3 days; (ii) writes off the falsehood. (iii) is monotone — it can only
add a truer banner where a false one stood — and reverses with one conditional.
**If you say nothing, (iii) fires tomorrow and I think that is the right
outcome.**

**2. NO-DECISION — the blackout is over and the mechanism it exposed held.** The
builder woke at 17:07 after **104** consecutive skipped slots and cleared the
entire five-deep overdue-default queue in **16 minutes**. Two things in that are
worth your attention, neither needing a ruling:

- **A default fired on a premise that had gone stale.** `D18`'s ordered code had
  already shipped on 09-03, *before the entry was armed*, and stronger than
  ordered. Executing its literal words would have made a working instrument
  worse. The builder executed the REPORT half instead and wrote the lesson. The
  armed-default mechanism is safe but its *text* ages; that is now recorded.
- **The safety clause bit its own beneficiary.** `D26` option (i) — which both
  desks recommend to you, and which would have released the builder sooner —
  **did not fire**, because a default may not loosen a gate. The builder took
  the narrower option against its own interest, unsupervised. That is the clause
  you were promised, observed working.

**3. NO-DECISION — 12.28 free GPU-hours expired tonight, unspent, and that is
the correct outcome.** `2026-W36` charged 17.72 h of 30. The builder was awake
for four hours with the pot open and did not scrape it, because `D1.0` attempt 3
measures 17.61 h, does not fit, and its precondition is unwalked. Two desks
spent three days wrongly telling it to squeeze this; its own steering refused,
twice. `2026-W37` opens tomorrow with a fresh 30 h. The larger standing number
behind it is unchanged: **`D1.0` has consumed 33.78 GPU-hours across two
attempts for zero ledger verdicts.**

**4. NO-DECISION — but it is the sentence I most want you to read.** `coverage`
still exits `rc=2` on **four claim-dead constitutional commitments**, two of
them in your own words: *"too cold kills him"* and *"he builds a shelter"*, plus
**smell** — which you named constitutional — and **balance**. Every claim spec
behind them is parked or foreclosed on honest, evidence-backed verdicts, so
nothing was hidden and nothing should be deleted; the repair is a registered
successor spec and it is routed (DUE 09-16, ninth day). Meanwhile
**one-brain/unison has 1 passing spec of 27** and the fresh-dispatch queue is
**one spec deep**. Five days at 108/245 is not a builder problem — it is that
the ladder has run out of rungs that can be climbed without a redesign.

**5. NO-DECISION — the one PASS whose reading you should know.** `PG.4`, the
noisy-TV fixture that certifies curiosity's trap works, passed on a cross-seed
mean while **one of its three seeds never found the panel at all** — dwell 0.0,
zero rays on the panel, zero reward ratio. This was disclosed by the builder in
its own commit message on 2026-08-10 and nothing was cheated. But the ledger row
says PASS with no mark, four specs depend on it, and the *Curiosity signal* seat
in `CHAMPIONS.md` names two of them as its arena. I re-derived the seed vector
independently from the recorded numbers to be sure, and it is exactly what the
commit said. **The repair is routed and dated (09-18); this is here so the
number reaches you from the oversight side too, and not only from the log of the
organ that found it.**
