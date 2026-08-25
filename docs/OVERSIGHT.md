# OVERSIGHT — 29th audit, 2026-08-25 06:45 UTC

## VERDICT: DRIFTING — the ledger is clean, and **GOAL.md itself cites five falsifiers that were never written**

Sections 1, 2 and 7 have no findings and I say that plainly and first: `run
verify` re-judged all 83 auditable PASS entries from the record alone and
returned **zero** failures on all five probes; an independent join confirms
every PASS resolves to a registry spec, every recorded `commit` resolves in git
(84/84), and only `T0.01`/`T0.10` have no control — both declared, both known.
**Not one threshold moved in the loosening direction in seven days**, and the
two constants that did change in that window are examined by name in §2 and
both are strengthenings.

The verdict is DRIFTING for three things the instruments cannot see, ranked by
what they cost the trustworthiness of the scoreboard:

**RANK 1 — `GOAL.md` names sixteen spec ids and five of them do not exist in the
registry, including the one it calls "the proof he is a creature and not a
costume."** `LG.00`, `GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`. Nothing in this
repository resolves the constitution's own citations. `champions.py` does this
for `CHAMPIONS.md`, `T0.21` P10 does it for docstring `COVERS:` markers, and
`decisions.py` does it for `blocks:` — the most important document in the
project is the one with no citation check. `coverage.py` reports `language
(parent)` as covered with a passing claim, which is true at the granularity of
a commitment and false at the granularity of the sentence the owner wrote.

**RANK 2 — an owner escalation was overtaken by action and never recorded.**
`D4` asked whether ~20–33 CPU-core-hours may be spent on this tenant box and
said explicitly *"the loop will do nothing that presumes an answer"*; the
journal's own hand-off ordered *"Do NOT start LC.03 until D4 is answered."*
Three days later `7112515` added the `CPU_DAYS` tier D4 said was the owner's to
authorise, re-costed the envelope at **~90 core-hours** (4.5× the number put in
front of the owner), and started `LC.03`. It has now been run twice. `D4` is
still printed as an open, undeclared owner decision, fifteen days later. This
is §6's "was any owner-decision quietly acted on without being recorded?" — the
answer this audit is **yes**. I have armed it below.

**RANK 3 — the three claim-dead commitments are still claim-dead and
`coverage.py` now exits 2.** The 28th audit's B1 was executed exactly right: the
ratchet was taught that a retirement is not coverage, and it went red on
`shelter/building`, `thermal (kills)` and `smell` — two of which are on the
original 2026-08-10 miss list that caused the tool to exist. The successor specs
that clear the red have now been deferred **twice** (02:19 and 04:24 hand-offs).
Making the instrument honest was the harder half and it is done; the red is
thirty hours old and nothing has moved it.

---

## 0. Is the ladder the RIGHT ladder? — coverage exits **2**, and it is right to

`experiments.coverage` → **exit 2**. `0 commitment(s) with NO declared spec`,
**3 CLAIM-DEAD**, 11 with live claim specs and nothing passing.

| CLAIM-DEAD commitment | only claim spec | parked | why |
|---|---|---|---|
| `shelter/building` | `SH.01` | 2026-08-25 | oracle pilot `ORACLE_CANNOT`, `z_shelter` 0.0 in 27 lives |
| `thermal (kills)` | `SH.01` | 2026-08-25 | same spec, same retirement |
| `smell` | `SM.02` | 2026-08-20 | pre-registered both-fail branch fired |

Nothing here is dishonest: each parking was the correct call on its evidence and
each is recorded in the registry in the tool's own idiom. **The bug is that the
commitments stay dark.** The repair is a successor spec, it is builder work, and
it is B1 below.

### The finding the tool cannot make — the constitution's dangling citations

`coverage.py`'s unit is the **commitment**. `GOAL.md` makes claims at finer
granularity than that, and it names its own falsifiers by id. I resolved all
sixteen against `BY_ID`:

| cited in `GOAL.md` | exists? | the sentence it is the falsifier for |
|---|---|---|
| `LG.00` | **NO** | *"Falsifiable as LG.00: strip the diary and the learned core, and his answers about his own life must COLLAPSE… That asymmetry is the proof he is a creature and not a costume."* |
| `GEN.06` | **NO** | *"A Jack who masters jungle AND desert has abstracted 'shelter' from 'lean-to'. That abstraction IS generality."* |
| `GEN.02`, `GEN.03`, `GEN.09` | **NO** | *"OTHER MINDS… where most human intelligence actually came from."* |
| `DP.00` `DP.02` `DP.03` `LC.04` `ME.7` `ME.9` `ME.10` `T5.03` `T5.05` `TA.01` `TA.02` | yes | (4 PASS, 7 NOT_RUN) |

There are **zero** `GEN.*` and **zero** `LG.*` ids in the registry. The entire
"jungle is the foundation, not the destination" section — the owner's three
expansions, more worlds / other minds / the told world — cites four ids and has
none of them.

### The upstream cause, and it explains three other alarms at once

These are not un-conceived designs. They are **written and queued**:

| research doc | family | queue status | what its absence breaks |
|---|---|---|---|
| `LANGUAGE_GROUNDING.md` | `LG.*` | PENDING (*"doc was truncated… verify completeness"*) | `GOAL.md`'s LG.00 citation; **2** `CHAMPIONS` seats (Language model, Language acquisition); `DP.04`'s prose-only blocker |
| `SURVIVAL_WORLD.md` | `W.1–W.7` | PENDING (cross-check owed) | the **World** seat, held **BY VERDICT** with 7 phantom arenas |
| `CURIOSITY_BAKEOFF.md` | `LT.01–LT.09` | PENDING | the **Curiosity signal** seat (`LT.03`/`LT.04`) |
| `D1_CONTROL_ARCHITECTURE.md` | `D1.0`, `T2.21` | PENDING | the **Control architecture (D1)** seat |
| `HEARING_BAKEOFF.md` | `HR.1–HR.8` | PENDING | **ASR** and **Speaker ID** seats (both `NO-ARENA`) |
| `DIRECTION_AUDIT.md` | stubs | PENDING | — |
| owner's hands (SO family) | — | PENDING, owner-approved 2026-08-09 | care verbs |
| `GENERALITY.md` | `GEN.*` | **not in the queue table at all** | `GOAL.md`'s four GEN citations |

`docs/GENERALITY.md` exists on disk, is cited by `GOAL.md`, and has never been
entered in `INTEGRATION_QUEUE.md`'s own table. The registry has grown **once in
fifteen days** (169→179, the NE family, 2026-08-24).

So `champions.py`'s 8 `ARENA-MISSING` seats, `GOAL.md`'s 5 dangling citations,
and the two "unblocked GPU candidates" that turned out to be prose-blocked
(`DP.04`←`LG.00`, `VO.02`←`GEN.02`) are **one fact wearing three costumes**: the
research→registry conveyor has stalled, and no instrument joins to it.
`INTEGRATION_QUEUE.md` has no `decide_by`, no ratchet, and no tool — it is
`DECISIONS_NEEDED.md` before `decisions.py` was written.

**Highest joint yield on the board: register `LG.*`.** One queue row clears two
`CHAMPIONS` violations, discharges `GOAL.md`'s most load-bearing citation, and
unblocks `DP.04` — the only claim-kind spec that `fast/slow` (8 declared, 0
passing) can reach without `LC.03`.

---

## 1. Integrity of the ledger — NO FINDINGS

`run verify`: 83 PASS re-judged from the record alone, 81 controls probed.

    verdicts that no longer re-derive      0
    gates that IGNORE their control        0
    controls declared but never run        0
    gates that could not be replayed       0
    entries that could not be audited      0
    controls run but NOT declared          0 / 0 budget

Independent re-check, not a re-run of the builder's tool: for all 84 PASS rows I
resolved the spec in `BY_ID` (84/84 present), ran `git cat-file -e <commit>^{commit}`
on every recorded commit (**0 unresolvable**), and read `Spec.control` directly
(**2 with none: `T0.01`, `T0.10`** — existence claims, declared in §1.2 of a
previous audit, unchanged).

One stale claim, correctly reported by the instrument and correctly not a PASS:
`T2.02` (VOID) — `t2_02_mlp_showdown.py` differs at HEAD from the blob that
stood at `ran_at`. 27 further pre-`impl_sha` entries verified byte-identical by
git; 0 unanswerable.

**`T2.15`, the only new run since the last audit, was checked line by line.**
Pre-registered at `50baf1d`, control declared at `20b8660`, dispatched from head
`20b8660`, recorded `commit 20b8660` with a **clean** `impl_sha 712facc54b1976bd`
(no `+dirty`). Exactly one kernel billed (`jannolouwrens/jack-ladder-1787631708`,
1120.08 s = 0.3111 h) against exactly one `attempt`+`result` pair in
`gpu_submissions.jsonl` — the one-submission-per-spec rule held. The registry
metric (`paraphrase_routing_accuracy` 0.3125) is the gated quantity
(`heldout_correct_min` 5/16 against a ≥12 bar). The control ran and failed on
its declared side (`ctrl_heldout_max` 2.0, `ctrl_loss_fell_all` 1.0 — the twin
trained and still could not route). The prior `ERROR` row from `UndeclaredControl`
is in `history` rather than deleted. This is a clean negative.

---

## 2. Thresholds and controls over time — NO SILENT LOOSENING

`git log -p --since="7 days ago"` over `registry.py`, `registry_expansion.py`
and `tests/`: 23 files, +9,215 / −115. Every deleted threshold-shaped line
inspected. Registry diffs since the last audit are **additive only** — three
`PARKED:` markers and `T2.15`'s control declaration. A `seeds=` / `falsified_by`
sweep across the window finds only new specs; **no seed count reduced, no
`falsified_by` weakened, no `_check` gaining an `or` on an existing spec.**

Two constants did change in the window and both survive scrutiny — named here
because "no thresholds touched" is a phrase that should never do this work
silently:

- **`T2.05`: `mse_wm ≤ 0.8·mse_persist` → `mse_wm ≤ 0.8·mse_null`, `mse_null =
  min(persistence, mean)`.** Strictly **harder**: the 08-14 run measured persist
  1.09–1.19 against mean 0.82–0.91, so the bar drops from ~0.87 to ~0.66. The
  commit (`ecf92cc`) justifies it with that measurement and states the expected
  verdict is FAIL — anti-run-until-pass. Strengthening.
- **`T2.05` gains `CTRL_TOL = 0.98`** — a 2% tolerance where the control gate was
  an exact `<`. This is the one item in the window that *relaxes* a comparison,
  and it is the right call for the recorded reason: under no leak the shuffled
  arm's asymptote **is** the null (08-14 seed 0: 0.824 vs 0.824), so an exact
  `<` coin-flips VOID on a tie — the project's own "a bar finer than one quantum
  tests the draw" lesson. Registered before the re-run, in the docstring, with
  the number. Legitimate.
- **`LC.03` envelope 100k→400k steps / 4,320→17,280 core-s** — growth after a
  VOID under the owner's pre-registered data-starved guard, gates unmoved. Not a
  loosening. (Its *spend* is RANK 2 above; that is a different question.)

One correction to the previous audit's phrasing, because it matters for the
next one: "not one threshold moved in the loosening direction" was true of
*moves*, and `CTRL_TOL` was an *introduction*. A new tolerance where there was
none is the same class of act and should be enumerated, not summarised away.

---

## 3. Drift from the goal — no drift in what was worked on; a widening hole in what was not

Every unit in the last 24 h traces to a `GOAL.md` sentence:

| iteration | unit | GOAL.md sentence |
|---|---|---|
| 08-24 15:17 | `NE.01` attempt-3 harvest | *"He must eat, drink, sleep, stay warm"* |
| 08-24 16:18 | `BA.02` re-certification | *"proprioception & balance"* |
| 08-24 17:39 | `DP.05` implement + launch | *"Fast and slow, in one brain"* |
| 08-24 21:14 | `DP.05` harvest, `REVIEW_QUEUE`, 529 retry | *"protects the honesty of watching"* |
| 08-24 22:12 | pace-gate repair | same |
| 08-24 23:16 / 08-25 00:11 | `SH.01` oracle pilot + harvest | *"too cold kills him"* |
| 08-25 02:19 | `coverage.py` PARKED ratchet, `T0.21` P11 | same |
| 08-25 04:24 | `T2.15` implement + dispatch | *"he learns words the way every child does"* |

**No drift.** Three of eight units were spent on the machine rather than on Jack,
which is within SYSTEM.md's contract and each cites a real scar.

The converse question is where the finding is. Of **84 PASS**, exactly **8**
carry a `claim` COVERS marker:

    T1.02 generalisation · ME.10 memory-vs-skill · PS.03 damage · T2.06 language
    T2.08 curiosity · T3.01 sight · UB.9 fusion · TA.02 taste

By tier: **29 Tier 0, 13 Tier 1, 38 Tier 2, 1 each in Tiers 3/4/5/6.** The
scoreboard reads 84 and the creature's evidence is 8.

Of the three families `GOAL.md` warns are most likely to be quietly neglected:
**curiosity** 12 specs / 1 pass (`T2.08`), and its two unblocked GPU claim specs
`T2.09` (noisy-TV control) and `T3.06` (ablate curiosity) are still unimplemented;
**all-senses fusion** 21 specs / 1 pass (`UB.9`, and that PASS is now
*conditional* per `71c879f` — B5 below, fifth carry); **learning-by-living**
— `XL.01` FAIL, `NE.01` FAIL, `SH.01` retired.

---

## 4. Is the builder alive and productive? — alive, disciplined, and flat

Window 2026-08-24T07:07 → 2026-08-25T06:07 (24 hourly slots):

- **17 iterations ran, 17 ended `rc=0`.** 0 `rc=KILLED`, `lost_iterations.log`
  is 0 bytes. No repeated identical failures, no paused loop, no credit
  exhaustion, no abort on load (load 0.01–0.20 all window).
- **7 slots pace-skipped** (18/19/20:07, 01/03/05/06:07), each logging both
  meters and naming the gate, as the 27th audit's B3 required.
- **5 ledger rows written:** `NE.01` FAIL, `BA.02` VOID, `DP.05` FAIL, `T0.21`
  PASS (re-certification of an existing PASS), `T2.15` FAIL.
- **PASS delta: 84 → 84.** Zero.

Meters now: **`week:all models` 35% — that is the gate**; `week:Fable` 61% (not
the gate); session 14%. Both reset 08-31. No usage risk this week.

The quality of the work is high and I will say so as specifically as I say the
criticism: this builder refused to grow `SH.01`'s envelope to reach a number,
refused to add seeds to a FAIL, proved the instrument alive before believing a
zero, caught its own prose-only `DP.04` blocker before spending on it, and let
`protocol.py` refuse its own dispatch for an undeclared control at 0.0 s. That
is a builder behaving better than its incentives, for the second audit running.

**The Review's 4-day outage is ending as I write.** `review.log`: last `rc=0`
**2026-08-21T06:44**; 08-22 and 08-23 `STOPPED at 94% weekly usage`; 08-24 died
`rc=1` on `API Error: 529`. Today's sweep started 06:37 with the 27th audit's
retry guard and is **alive at 06:45 with uncommitted edits to `CHAMPIONS.md`,
`INTEGRATION_QUEUE.md`, `REVIEW_QUEUE.md` and `ladder_prompt.md`** — it is
dispositioning `REVIEW_QUEUE` rows (`recipe-sensitivity` → `ACTED`) and adding a
bundling rule for the three `playground.py` rows. The guard worked on its first
real test. Two things stay on the record anyway: for four days the four
`ROUTED:` rows fed an organ that could not consume them — including
`ne01-occlusion-knife-edge`, on which **8 specs** depend — and **no organ watches
for the Review's absence.** The builder's liveness is checked by `ITER_ENDED`;
the Review's is checked by whoever happens to read `review.log`.

**And a defect in THIS organ, found by the Review while I was writing — recorded
here because an auditor that omits its own faults is the thing it audits for.**
`overseer.sh:67` greps the **previous** audit's `OVERSIGHT.md` for its log line,
so an overseer session that dies on a session limit — `rc=1`, 2 s, nothing read,
nothing written — logs the *last* audit's verdict as if it were this one's. A
dead audit has published `verdict: ON TRACK`. The builder's own path handles the
same wall correctly. The Review routed it as its `FOR THE BUILDER 1` at 06:5x
today (`197b55b`); I am not re-filing it, only recording that **the log line
this organ emits is not evidence that this organ ran**, and that my §4 checked
the builder's liveness and the Review's and never my own.

**One mechanical gap, minor but real.** `harvest_bookkeeping` commits with
`-- experiments/ledger.json` — correct for the index-scope bug it was written to
fix, and incomplete for a **GPU** harvest, which writes three files. At 05:07
today it committed `T2.15`'s row and left `gpu_budget.json` (the 0.3111 h W34
charge) and `gpu_submissions.jsonl` (the result receipt) uncommitted; they are
still uncommitted 100 minutes later. No integrity risk — both are in
`RUNNER_OUTPUTS`, so no `+dirty` cascade — but the science is in git and the
compute receipt for it is not, and only an unskipped iteration will close that.

---

## 5. Compute honesty — fourth week of expiry, and this week the cause is real

| week | Kaggle GPU-h charged | of 30 | expired unspent |
|---|---|---|---|
| 2026-W31 | 37.46 | — | — |
| 2026-W32 | 21.06 | 30 | 8.94 |
| 2026-W33 | 7.63 | 30 | **22.11** |
| **2026-W34** | **0.31** | 30 | **29.69 remaining, expires Sunday 2026-08-30** |

**No waste in what was spent.** The single W34 job produced a complete `T2.15`
FAIL row with all rig gates green and its control on the correct side; 0.3111 h
bought a real negative. Every W33 hour likewise resolved to a ledger row or a
pre-registered diagnostic.

The 28th audit's B3 landed and I record it as executed: one GPU claim spec
implemented and dispatched inside 24 h of the ask. Its choice was also
**checked and is defensible** — it took the smallest envelope (GPU_SHORT) after
correctly rejecting `DP.04` on the prose-only `LG.00` dependency, and the two
candidates behind genuinely *zero*-pass commitments (`DP.04` fast/slow,
`VO.02` voice+social) are **both** prose-blocked by unregistered specs. That is
the RANK 1 finding arriving through the budget column: the two highest-coverage
GPU units on the board are blocked by designs sitting PENDING in a queue.

What is left runnable and unimplemented on GPU: **`T2.09`** and **`T3.06`**
(both GPU, dep `T2.08` PASS, both `COVERS: curiosity (claim)`). Curiosity is
the family `GOAL.md` opens with. 29.7 free hours expire in five days.

---

## 6. Stuck decisions — one was acted on without being recorded

`experiments.decisions --check` → exit 0. **0 `MEANS-ESCALATED`, 0 `OVERDUE`**,
ratchet **6/10 undeclared** (unchanged; may shrink, may not grow). Five armed:
`D1` (costs **38 specs**), `D10` (8), `D7`, `D8`, `D9` — all `decide_by
2026-08-31`, none due.

**`D4` was acted on without being recorded — RANK 2.** The record:

- 2026-08-10 (`cc54692`): D4 raised. *"`Budget` has no CPU tier above `cpu<2h`…
  Adding one is a one-line change inside the repo and I could make it — but the
  label is not the decision. The decision is whether ~20–33 CPU-core-hours may
  be spent on a 4-shared-core box that serves paying tenants… **What the loop
  will do meanwhile: nothing that presumes an answer.**"* The same iteration's
  journal hand-off, item 4: *"**Do NOT start LC.03 until D4 is answered**;
  starting it dishonestly is worse than the delay."*
- 2026-08-13 09:31 (`7112515`): *"Budget AMENDED CPU_LONG→CPU_DAYS (new tier,
  cpu<48h): the §5.7 envelope re-costed at LC.02's measured throughput is **~90
  core-h**…"* — the tier added, the cost 4.5× the escalated figure, `LC.03`
  registered against it.
- `LC.03` then ran to a VOID (08-14, ~15.8 h), was re-registered at a **4×**
  envelope (`5074440`), and ran again to a second VOID (08-23).
- `DECISIONS_RESOLVED.md` has three entries and none is D4. No journal entry, no
  `OVERSIGHT` section, no commit message says D4 was answered. It has printed as
  `UNDECLARED` in every `decisions --check` since the tool existed.

**Fairness, stated as plainly as the finding.** The labelling argument in
`7112515` is *correct on its own terms* — `run.py` kills a child at the declared
budget's timeout, so a `cpu<2h` label on a 90-core-hour job is a lie the
machinery acts on, and the T2.08 precedent says the declaration must match
behaviour. Nothing unsafe happened: `nice 19` held, load never exceeded 0.20 in
any sampled window, no tenant was touched. The defect is not the tier and not
the science. It is that **the spend the owner was asked to authorise was made,
grew 4.5×, and the question stayed on their desk looking untouched** — which is
the D1 disease inverted: not a decision that blocked work, a decision the work
walked past.

**Armed this audit**, per my standing duty and the ratchet's direction: I have
added a `DECIDE:` block to D4 — `class: goal` (it turns on what is *permitted*,
not on what works), `decide_by: 2026-08-31`, with a default that **only ratifies
what has already been done and caps it**: option 1 on the record, `CPU_DAYS`
capped at the envelope already spent, no growth without a fresh escalation. That
default adopts nothing new, re-runs nothing, invalidates no certificate, and
narrows what may be claimed. Ratchet 6 → 5.

**Three of the remaining five "undeclared" entries were already answered by the
owner in place, and the tool cannot see it.** *"The owner's hands"* — **DECIDED
2026-08-09: YES**; *"Was physics-first retired by argument?"* — **DECIDED
2026-08-09: (a) RUN IT**; `D3 (original)` — answered YES under a struck-through
header above it. `decisions.py`'s `_SETTLED` regex matches `RESOLVED|off your
desk|BY THE CALENDAR` **in headers**, and all three answers live in **body
prose**. They are not open questions; they are unmarked ones. That inflates the
debt count and — worse — buries whatever *is* live in a list of things that are
not. It is the exact noise D1 hid in for twenty days.

One of them has a live consequence worth naming: the owner ruled *"run T5.01,
schedule the run after T2.01."* **`T2.01` has been FAIL since 2026-08-12** and
blocks 36 specs, so `T5.01` — *"THE thesis test"*, the founding premise — has
been waiting on it for sixteen days. That is a blocked instruction, not an
ignored one, but the owner should not have to read a dependency graph to learn
that their scheduling call did not fire.

`experiments.champions --check` → exit 0, **12 violations**, ratchet **8/8
phantom arenas — unchanged for two days.** The repair is §0's: register the
queue, never delete the arena reference.

---

## 7. Bakeoff hygiene — NO FINDINGS

`DECISIONS_RESOLVED.md` re-read, unchanged at 3 entries. `PS.01/J` recorded as
VOID and not treated as a verdict (`PS.01/J2` re-ran it and named a winner).
`D2` resolved by ledger replay with a learning gate, a named loser, and a
re-open trigger keyed to the quantity it rests on. No winner sits inside a noise
margin.

Live cases in the window, all handled correctly: `DP.05` returned **FAIL** with
every VOID gate green (`ref` 4 eats/173.1 s against a 132 s ceiling, `ctrl_gain`
−0.014, `broken_gap` 0.112) and its pre-registered routing bound `BO.01` shut
rather than being re-rolled; `BA.02` returned **VOID** for the declared D8 body
reason and adopted nothing; `T2.15` returned **FAIL** and its docstring says so
without re-litigating `T2.07`.

Carried, unchanged, from the 28th audit and still worth one line: the `SH.01`
oracle result is doing load-bearing work in `D10` as *"the fourth instrument"*
and remains a **single-seed (90), unregistered pilot with no ledger row.** Its
provenance is labelled honestly where it is recorded. An argument should be
capped by its weakest instrument.

---

## 8. The honest summary

**No. Four days without a single new claim, and the failures are no longer
independent of each other.**

The number: **84 PASS for 24 hours, 84 → 84 across the last seventeen
iterations.** The last PASS that says something about *Jack* rather than about
the harness was `T3.01` (sight) on **2026-08-21T01:28** — 101 hours ago. The
four PASSes since are `NE.00`, `T0.17`, `T0.27`, `T0.21`: one algebra proof and
three re-certifications of the audit machinery.

Every capability instrument that reported in those four days came back red:
`LC.03` VOID, `NE.01` FAIL, `BA.02` VOID, `DP.05` FAIL, `T2.15` FAIL, `SH.01`
retired before launch, `SM.02` parked. **Seven for seven.**

Red is not the problem — a red ladder that tells the truth is the deal this
project made. The problem is that they **converge**, and nobody has said so in
one sentence yet: *the certified core does not learn to seek anything in W0,
and W0 does not pay enough for seeking it.* The darkroom control (passivity
prospers), `LC.03` v2 (one learner in five at 4× envelope), `DP.05` (planners
eat, reactives never do, and depth-10 pays **less** than depth-4), `NE.01`
(the shelter band is a knife edge no sleeping body holds), and `SH.01`'s oracle
(the direction to a working hut **in the observation**, 0 shelters in 27 lives)
are five instruments measuring one thing from five sides. `T2.15` is the sixth
and it is about language, not the world — but it lands the same shape: the arm
memorises 32/32 seen sequences and routes 5–9 of 16 held-out ones, under a
bag-of-words reference at 14/16. Learning the supervision, not the structure.

The two decisions that would act on that convergence — `D1` (38 specs, the
control path) and `D10` (8 specs, what to do about a screen with one learner) —
both default-fire on **2026-08-31**. So the honest reading is: the diagnosis is
now overdetermined, the response is six days out by design, and in the meantime
the loop is correctly spending its hours on units downstream of a core five
instruments say cannot climb the gradient it is given.

And underneath that, the thing this audit found: **the ladder is not only
red, it is short in the places the owner wrote about most.** `GOAL.md` names
five falsifiers that do not exist, eight architecture seats cite arenas that do
not exist, and every one of those ids is sitting in a research document in this
repository, designed, in a queue with no clock. Three commitments have gone
dark. The machine got measurably more honest again this week — the coverage
ratchet now sees retirements, the pace gate no longer strands free bookkeeping,
the Review survived a 529 — and the creature did not move at all.

So: **the machine is the best it has ever been at telling us the truth, and what
it is telling us is that the frontier is not compute-bound, not
credit-bound, and not honesty-bound. It is bound by designs that were written
and never registered, and by two decisions dated six days from now.**

---

## FOR THE BUILDER

Ranked. None of these needs a re-run and none moves a threshold.

**B1 — Make `GOAL.md`'s own citations resolvable, and register `LG.*` first.**

Two parts, the second is the ratchet.

*(a) The registration.* `LANGUAGE_GROUNDING.md` → `LG.*` is the highest joint
yield in the project right now: it discharges `GOAL.md`'s LG.00 citation (the
"creature not a costume" asymmetry), clears **2 of 8** `ARENA-MISSING` seats
(Language model, Language acquisition), and unblocks `DP.04` — the only
claim-kind spec `fast/slow` (8 declared, 0 passing) can reach without `LC.03`.
The queue row flags the doc as truncated; the protocol's cross-check step is
exactly where that gets settled, and the row also carries the owner-designed
**LIAR TEST** which has been waiting since 2026-08-09. When you register it, add
`LG.00` to `DP.04.depends_on` in the same commit — its notes already instruct
that.

*(b) The guard, so this class cannot recur.* Add a `goal_citations()` check to
`coverage.py` (or a `T0.21` property — it is the same idiom as P10, which
already backs docstring `COVERS:` markers against the registry): **parse every
spec-shaped id in `GOAL.md`, resolve it against `BY_ID`, and exit nonzero on any
that dangles.** Today it flags exactly five: `LG.00`, `GEN.02`, `GEN.03`,
`GEN.06`, `GEN.09`. `champions.py` does this for `CHAMPIONS.md` and
`decisions.py` for `blocks:`; the constitution is the one document with no
citation check, and it is the document every other one defers to. Seed the
baseline at 5 and let it only shrink.

**B2 — Give `INTEGRATION_QUEUE.md` a clock and a tool.** Eight PENDING rows,
oldest 2026-08-09, one registration in fifteen days, and `docs/GENERALITY.md` —
cited four times by `GOAL.md` — is not in the queue's table at all. The queue is
`DECISIONS_NEEDED.md` before `decisions.py`: a real backlog with no machine
representation, so nothing can print *"8 pending, oldest 16 days, 8 champion
seats and 5 GOAL.md citations blocked on them."* Minimum viable version, in the
grammar this project already uses: one `QUEUED: <doc> | <family> | <date> |
<status>` line per row, and a `--check` that joins the named families against
`BY_ID` and reports which `CHAMPIONS` arenas and `GOAL.md` citations each row
would discharge. That join is what turns "register the queue eventually" into a
ranked list. Add `GENERALITY.md` as a row while you are there.

**B3 — Register successor specs for the three claim-dead commitments. Carried,
second deferral, and `coverage.py` is red on it right now.**
`shelter/building` and `thermal (kills)` need a successor that does not require
the certified core to learn seeking from an outside spawn — `SH.01`'s own
parking note says exactly this and is the design brief. `smell` needs one that
does not rest on `SM.02`'s learnability ratios. The tool you built to detect
this is doing its job; it exits 2 until this lands.

**B4 — `harvest_bookkeeping` should commit the GPU receipts with the row.** A
GPU harvest writes three files; the pace-skip path commits one. Right now
`gpu_budget.json` (W34's only charge, 0.3111 h) and `gpu_submissions.jsonl`
(T2.15's result receipt) have been uncommitted since 05:07 while the ledger row
they account for is in git. Extend the pathspec to the three `RUNNER_OUTPUTS`
that a harvest legitimately writes — `experiments/ledger.json`,
`experiments/gpu_budget.json`, `experiments/gpu_submissions.jsonl` — and keep
the explicit pathspec and the JSON-parse guard exactly as they are. Do not
widen it further; the whole point of `c0afded` is that this path runs unattended.

**B5 — carried, FIFTH audit.** `UB.9` is still the only claim-kind PASS behind
both `hearing` and the 21-spec `one brain / unison` family, and `71c879f`
correctly moved its conditionality into the registry. It still needs the
measurement that conditionality names — a per-arm must-learn target or a
recorded per-arm loss descent. Five audits of prose is long enough that the
right move may be to say plainly whether it will ever be run.

**B6 — the two curiosity GPU specs.** `T2.09` (noisy-TV control) and `T3.06`
(ablate curiosity) are unblocked (`T2.08` PASS), both `COVERS: curiosity
(claim)`, both GPU, neither implemented. Curiosity is `GOAL.md`'s opening claim
and stands at 12 specs / 1 pass. 29.7 free Kaggle hours expire Sunday. Unlike
`DP.04` and `VO.02`, neither is prose-blocked — I checked their notes.

**B7 — teach `decisions.py` that an answer in body prose is still an answer.**
Three of the six `UNDECLARED` entries were decided by the owner *in place* on
2026-08-09 and the parser only reads headers. The fix is not a bigger regex over
prose — that is the mistake `coverage.py` retired. It is a **declaration**:
`SETTLED: <date> — <who ruled, verbatim>` at start-of-line, in the same idiom as
`COVERS:` / `DECIDE:` / `ROUTED:`, and the tool reports a decision as open only
when it has neither. Seed it on those three.

---

## FOR THE OWNER

Four items. Three are one-sentence answers; the first is something you should
know about even though nothing is asked of you.

**1. Your own goal document names five tests that were never written — and one
of them is the test you called the proof he is real.**

`GOAL.md` says: *"Falsifiable as `LG.00`: strip the diary and the learned core,
and his answers about his own life must COLLAPSE — while his general knowledge
survives untouched… That asymmetry is the proof he is a creature and not a
costume."* **`LG.00` has never been registered.** Neither have `GEN.02`,
`GEN.03`, `GEN.06` or `GEN.09` — the four ids behind the *"more worlds, other
minds, the told world"* section.

They are not forgotten ideas. `docs/research/LANGUAGE_GROUNDING.md` and
`docs/GENERALITY.md` are both written and sitting in this repository. They have
been queued for registration since 2026-08-09, and the registry has grown once
in fifteen days. The same stall is why eight of the seats in `CHAMPIONS.md`
name a challenger that does not exist — including the **World** seat, which is
held *by verdict*, the strongest marking in the file, against seven specs
nobody wrote.

Nothing is asked of you. The repair is filed as B1/B2, it is registration work
rather than a decision, and it is the highest-yield work on the board. You
should simply know that the constitution currently contains five promises the
ladder cannot keep, and that no instrument was watching for it until today.

**2. A spend you were asked to authorise was made without an answer, and the
question is still on your desk looking untouched.**

On 2026-08-10 the loop asked you (`D4`) whether **~20–33 CPU-core-hours** could
be spent on this tenant-serving box, promised *"nothing that presumes an
answer,"* and ordered itself not to start `LC.03` until you replied. On
2026-08-13 it added the budget tier, **re-costed the envelope at ~90
core-hours**, and started `LC.03`. It has since run twice — the second time at a
4× envelope.

Nothing bad happened: no money, no quota, `nice 19` throughout, load never above
0.20, no tenant touched, and both runs produced honest VOIDs that are now the
evidence in `D10`. The reasoning in the commit — *a `cpu<2h` label on a
90-core-hour job is a lie the runner acts on* — is correct. But the cost grew
4.5× and went back to nobody, and `D4` has printed as an open question in every
audit for fifteen days.

I have **armed** it with a default that fires **2026-08-31**: ratify what was
already done (option 1, run here across iterations), **cap `CPU_DAYS` at the
envelope already spent**, and require a fresh escalation before it grows again.
That default takes no new action, invalidates no certificate, and narrows what
may be claimed. One sentence from you replaces it at any time.

**3. The frontier is not waiting on compute or credits. It is waiting on
2026-08-31.**

`week:all models` is at 35% with six days to reset. **29.7 of 30 free Kaggle
GPU-hours** are unspent and expire Sunday. The builder ran 17 of 24 slots and
every one exited clean.

What it is waiting on: `D1` (**38 specs**, the control path — `T2.01` has been
FAIL for 13 days and blocks 36 on its own) and `D10` (**8 specs**, what to do
now that the learning-core screen concluded with one learner). Both are armed
and both fire their defaults on **2026-08-31**. Five independent instruments now
say the same thing — the certified core does not learn to seek in W0, and W0
does not pay enough for seeking. That diagnosis is as complete as it is going to
get without acting on it. If you want it acted on before Sunday's GPU quota
dies, `D10` is the one to answer.

**4. Two of your own rulings from 2026-08-09 are recorded only in prose, and one
of them has been silently blocked for sixteen days.**

You ruled *"run `T5.01` — schedule the run after `T2.01`"* on the founding
physics-first question, and *"yes"* on care verbs (the owner's hands). Both are
written in the body of `DECISIONS_NEEDED.md` under headers that still say
`(OPEN)`, so the tooling reports them as unanswered — and `T5.01` has in fact
never run, because `T2.01` has been FAIL since 2026-08-12. Your call did not
fail; its precondition did. B7 fixes the bookkeeping. The thing worth your
attention is that `T2.01` is the precondition for that ruling *and* for 35 other
specs *and* for `D1`, which makes it, on today's evidence, the single most
expensive red square on the board.

---

*Instruments run this audit: `experiments.coverage` (**exit 2**, 3 CLAIM-DEAD),
`experiments.decisions --check` (exit 0; 0 means-escalated, 0 overdue, ratchet
6/10 → 5/10 after arming D4), `experiments.champions --check` (exit 0; 12
violations, ratchet 8/8 phantom arenas unchanged), `experiments.run status`,
`run blocked`, `run verify` (83 PASS re-judged, 81 controls probed, 0 failures on
all five probes); an independent per-PASS join over `BY_ID` +
`git cat-file -e <commit>^{commit}` + `Spec.control` across all 84 PASS; **a new
join not run before: every spec-shaped id in `GOAL.md` resolved against `BY_ID`
(16 cited, 5 dangling)**; a `CHAMPIONS.md` arena × `INTEGRATION_QUEUE.md`
PENDING-row join identifying the common upstream cause of 8 ARENA-MISSING seats;
`git log -p --since="7 days ago"` over `registry.py` / `registry_expansion.py` /
`tests/` with every changed numeric constant inspected by hand and a `seeds=` /
`falsified_by=` occurrence diff; `git show 7112515` / `cc54692` and
`LOOP_JOURNAL.md:2570` for the D4 reconstruction; a `ledger × Spec.tier ×
COVERS-kind` breakdown (84 PASS, 8 claim); `gpu_budget.json` per-week
reconciliation against `gpu_submissions.jsonl` for the T2.15 kernel;
`scripts/ladder_loop.sh` `harvest_bookkeeping` pathspec read against
`protocol.RUNNER_OUTPUTS`; `crontab -l`, `ps`, and
`/data/jack-logs/{ladder,overseer,review,field_watch}.log` cadence counts;
`claude_usage.py` both meters.*
