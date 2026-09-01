# OVERSIGHT — 58th audit, 2026-09-01 12:45 UTC (HEAD `566f840`, tree dirty: 2 GPU bookkeeping files, written 12:19 by the D1.0 watcher)

## VERDICT: DRIFTING — the ladder is clean and nothing was weakened, but the FIRST instrument this audit is told to trust reports `0 CLAIM-DEAD` while five of the owner's own commitments have nothing anybody is allowed to run behind them

Sections 1, 2, 6 and 7 have **no violations**, checked mechanically rather than
asserted, and section 5 has none either. Saying that plainly is the point: it is
what makes FINDING 1 worth acting on instead of reading as noise.

The 57th audit's FINDING 1 (`LC.07`'s wrapped-sentence phantom foreclosure) is
**CLOSED** — repaired at `ff9917a`, verified live: `coverage` reports 6
VOID-FORECLOSED specs and `LC.07` is not among them.

---

## THE FOUR MANDATORY INSTRUMENTS

| instrument | rc | reading |
|---|---|---|
| `coverage` | **0 (GREEN)** | 23 commitments, **0 with no declared spec, 0 CLAIM-DEAD**, 0 dangling GOAL.md citations. **And the `0 CLAIM-DEAD` is false in fact — FINDING 1** |
| `decisions --check` | 0 | 0/10 UNDECLARED, no `MEANS-ESCALATED`, no `OVERDUE`. 3 armed: D15/D16 due 09-05, D17 due 09-07 |
| `champions --check` | 0 | 27 seats, ratchet ok — 0 phantom arenas, 3/4 unfalsifiable, 3+1/6 uncontestable, all baselined |
| `run review-queue` | 0 | 11 OPEN / 2 HELD / 2 ACTED of 15; oldest live 8 d; consumer ran today; **0 violations** |
| `review_liveness` | 0 | schedule half green, no banner |

`coverage` went **green** for the first time in days (the four `GEN`
registrations at `7f1e875` cleared the last dangling citations, and the empty
classes were re-baselined). That is the finding: it went green on the day the
thing it exists to catch got worse.

**Nothing to arm this audit.** `decisions` reports 0/10 UNDECLARED, so the
standing "arm at least one per audit" duty has nothing to bite on. The ratchet
did not grow.

---

## FINDING 1 (top rank) — `coverage.py`'s CLAIM-DEAD ratchet cannot see a foreclosure, so five commitments are claim-dead in fact and green in the instrument

### The mechanism, in two lines of code

`experiments/coverage.py:437`:

```python
def _claim_dead(r: dict) -> bool:
    return (not r["n_pass"]
            and not any(k == "claim" for k in r["kinds"].values()))
```

`kinds` has already had **PARKED** specs removed (28th audit, the third scar) —
and nothing else. `claim_reachability()` (`coverage.py:398`) has exactly four
states: `PASS`, `RUNNABLE`, `PARKED`, `blocked<-ROOTS`. **There is no state for
`VOID-FORECLOSED` or `PILOT-BLOCKED`.** A foreclosure is a retirement that does
not spell `PARKED:`, so it launders a park — and worse, it lands in `RUNNABLE`,
the strongest state short of `PASS`.

### The consequence: the same report contradicts itself

`coverage` prints, in its commitment table:

```
shelter/building   2 specs  0 pass  1 now   claims: SH.02 RUNNABLE, SH.01 PARKED
```

and forty lines later, in its own queue-depth section:

```
4 spec(s) are PILOT-BLOCKED ... Do NOT spend seeds on these; the repair is a redesign:
    SH.02: ... The repair is a REDESIGN ... routed to the Review.
```

Seven rows are in this state (computed against the tool's own declared sets —
VOID-FORECLOSED `{BA.03, LC.03, ME.11.E, ME.11.F, T3.06, UB.14}`, PILOT-BLOCKED
`{DP.04, SH.02, SM.03, T2.11}`):

| commitment | spec | coverage says | actually |
|---|---|---|---|
| balance | `BA.03` | RUNNABLE | VOID-FORECLOSED |
| smell | `SM.03` | RUNNABLE | PILOT-BLOCKED |
| shelter/building | `SH.02` | RUNNABLE | PILOT-BLOCKED |
| thermal (kills) | `SH.02` | RUNNABLE | PILOT-BLOCKED |
| fast/slow | `DP.04` | RUNNABLE | PILOT-BLOCKED |
| curiosity | `T3.06` | RUNNABLE | VOID-FORECLOSED |
| one brain / unison | `UB.14` | RUNNABLE | VOID-FORECLOSED |

### Five commitments are claim-dead in fact

Discount the do-not-run specs and these five have **zero passing claims and
nothing anybody may run**:

| commitment | every claim spec |
|---|---|
| **balance** | `BA.03` VOID-FORECLOSED, `BA.02` PARKED |
| **smell** | `SM.03` PILOT-BLOCKED, `SM.02` PARKED |
| **shelter/building** | `SH.02` PILOT-BLOCKED, `SH.01` PARKED |
| **thermal (kills)** | `SH.02` PILOT-BLOCKED, `SH.01` PARKED |
| **fast/slow** | `DP.04` PILOT-BLOCKED; `DP.01`/`DP.02`/`DP.03` blocked←`LC.03` (itself VOID-FORECLOSED); `BO.01` blocked←`DP.05` FAIL |

Note `SH.02` is the sole live claim for **two** commitments, so one pilot-blocked
spec carries thermal and shelter together.

**Four of the five are the owner's own 2026-08-09 survival directives** — *"too
cold kills him"*, *"he builds a shelter"*, smell as a named sense
(GOAL.md:41–48), and balance — and the fifth is the 2026-08-10 fast/slow
directive. These are, almost exactly, the commitments the 2026-08-10 miss was
about. The failure mode has simply moved one step downstream: then the spec did
not exist; now it exists, has an id, appears in `run next`, and cannot produce
evidence.

### And it was already found, in the same audit that fixed the identical hole next door

The **54th audit** (2026-08-31, `2e769d5`), §3.d, wrote verbatim:

> `coverage` reports 0 commitments with no declared spec and 0 CLAIM-DEAD, but
> `CLAIM-DEAD` counts *parked*, not *unreachable*. Under the reachability
> reading, **fast/slow** is materially claim-dead.

That same audit turned the identical insight into a builder item — **for the
other tool**: *"B4 — `champions.py` must count REACHABILITY, not just
existence"* — which shipped at `78aad78` as `ARENA-UNREACHABLE` and is working
today. **It never wrote the matching item for `coverage.py`.** The observation
sat in prose, the 55th/56th/57th audits did not carry it, and in two days it
went from **1 commitment to 5**. A finding that lives only in a paragraph is a
finding the system cannot act on — which is the `wk4-N3` scar in a different
file.

### The repair must ADD a class, never convert one

Per the standing rule this repo already paid for three times: the CLAIM-DEAD
total must go **up by 5**, not sideways. The repair for the five commitments is
registration or re-parenting, never quieting the row. Precise instructions in
**FOR THE BUILDER B1**.

---

## FINDING 2 — 39% of the ladder is unreachable, it is drifting upward, and no ratchet counts it

`run blocked`: **85 of 217 specs are unreachable.** The 55th audit (08-31)
measured 80 of 211 (38%); today it is 85 of 217 (39%).

One FAIL dominates and it is the right one to be spending on: **`T2.01`
(*Locomotion beats a random policy*) blocks 38** — the whole `CU` curiosity
family, nine `UB` unison specs, six Tier 5 claims. After it: `NE.01` frees 8,
`LT.01` frees 7, `UB.10` frees 4, `LC.07` frees 4.

Four instruments ratchet in this repo — `coverage` on uncovered commitments,
`decisions` on undeclared, `champions` on phantom arenas, `review_queue` on
violations. **None counts the unreachable fraction.** A foreclosure that welds a
new subtree lands silently; the only reason anyone knows the number moved is
that two overseers happened to print it. This is FINDING 1's raw form: the
constitutional consequence is F1, the number itself is F2.

---

## 1. Integrity of the ledger — NO FINDINGS

119 rows, **93 PASS**. Checked mechanically, every PASS row:

- **93/93 resolve to an implementation** via `run._module_for`. Zero missing.
- **0 commits missing from git** (`git cat-file -e` on every `commit` field).
- **0 stamped `+dirty`.** The dirty-stamp block that `run status` carried
  yesterday was cleared at `cde56c8`/`ff9917a` by honest re-buys.
- **0 specs declaring no control.** Two carry empty `control_metrics` —
  `T0.01` and `T0.10` — and both declare `CONTROL: NONE, BY DECISION (52nd audit
  B5)` with a stated reason (an import either raises or does not; a sabotaged
  upload fails on the service's side, which *is* the falsifier). Honest, not a
  gap.
- **21 rows carry `amended` entries; every one is an `impl_sha`/`hardware`
  doc-only reconstruction** with a `proof` field and a named author (`26th-audit-B2`,
  `54th-audit-B3`, `T0.14`, …). **No status was flipped by an amend.**

**The one standing red is `T0.27` (FAIL, 2 live violations)** and it is red *by
decision*, not by neglect. I re-ran `audit_supersedes_fail` independently: the
violations are `LG.00` (VOID stamped `8faff43+dirty`) and `T0.17` (FAIL stamped
`d84101e+dirty`). `T0.17`'s is permanently unrecoverable (`7ffd961` proved no
committed tree reconstructs it — the class is closed going forward, this
instance cannot be repaired backward). `LG.00`'s turns on whether a verified
preserved manifest at `refs/jack/failimpl/` counts as an equal artifact, which
is **D16, armed, due 2026-09-05**. 7 pairs checked, 24 historically unauditable
(pre-`impl_sha`, correctly read as a gap rather than a violation).

## 2. Thresholds and controls over 7 days — NO FINDINGS, and this one was worth checking hard

81 commits touched `registry.py` / `registry_expansion.py` / `tests/`. I paired
every same-name numeric constant across every one of them.

**10 constants moved. Nine tightened:**

| commit | constant | move | file |
|---|---|---|---|
| `9e7cc86` | `N_EVAL` | 48 → **120** | `ba_03` |
| `1653104` | `LIVES_PER_ARM` | 16 → **48** | `t3_06` |
| `bf947a1` | `LIVES_PER_ARM` | 4 → **16** | `t3_06` |
| `2c90fc9` | `STEPS` | 300 → **500** | `t2_19` |
| `2a31a8e` | `N_DECISIONS` | 3200 → **4800** | `w0_diag` |
| `5465fc5` | `COORD_MIN` | 0.55 → **0.70** | `vo_02` |
| `5465fc5` | `COORD_MARGIN` | 0.20 → **0.35** | `vo_02` |
| `5989ea7` | `N_PROPERTIES` | 11 → **12** | `t0_21` |
| `bf64a85` | `N_PROPERTIES` | 11 → **12** | `t0_31` |

**One moved in the loosening direction, and it is legitimate:** `DECAY_MIN`
1.5 → **1.25** (`44f24c4`, `t2_09_noisy_tv_control.py`). It is a `# PILOT`
placeholder being frozen for the first time on a spec whose `run()` **refused
until that commit** — not a registered threshold weakened. The commit message
carries the measurement (seed 90's claim-arm static decay read 1.472, so the
placeholder would have discarded a live decaying signal as dead) and sets the
bar from principle rather than from the observation (*"a constant signal decays
by exactly 1.0"* — 1.25 is not shaved to the minimum). The **same commit raised
seeds 3 → 7** and replaced a mean-over-seeds fold with worst-informative-seed.
Net strengthening.

**Controls:** two `control=` lines were removed; both were replaced by strictly
stronger text in the same hunk — `T2.10` gained the paraphrase venue plus a
leaky-cue aliveness floor (`468772e`), `ME.9` gained `seeds=1 → 3` with the
control's own spread recorded as the largest number in its record (`f9549cb`).

**Seeds:** no seed count was reduced anywhere in 7 days.

**`_check` gaining an `or`:** 24 added disjunctions inside gate functions. I read
all 24. **Every one widens a VOID/aliveness guard** — `seed_alive_ok != 1.0 or
control_alive_ok != 1.0`, `force_calib_ok`, `verdicts_missing > 0` — i.e. each
makes *claiming* harder, not easier. None touches a claim gate.

## 3. Drift from the goal — no drift in what was done; the converse is FINDING 1

**Every unit in the last 24 h traces to a GOAL.md sentence:**

| unit | serves |
|---|---|
| `D1.0` implemented, envelope frozen, 3-kernel run dispatched | *"Give him a brain, a body, and a world"* — the control-path bakeoff aimed at `T2.01`, which blocks 38 |
| `LC.07` implemented | the owner's ~10× scale-transfer guard — *"research and testing at EVERY SINGLE STAGE"* |
| `UB.14` implemented, run, VOID-FORECLOSED | *"all senses in unison… ablate a sense, something measurable must degrade"* — an honest negative on the venue |
| `LT.01` implemented, FAIL | *"a ladder with an apple on top… climb, fall, learn from falling"* — the north star's own arena |
| `T2.10` conjunct + honest re-buy RED | *"Really learning, not appearing to learn"* — a capability left the board because we asked harder |
| `ME.11.E`/`.F` VOID-FORECLOSED | *"Memory makes it him"* — settling a family by arithmetic instead of re-buying it |
| `GEN.02/03/06/09` registered | *"the jungle is the foundation, not the destination"* — the constitution's four dangling citations resolve |
| eleven armed defaults fired | *"protects the honesty of watching what happens"* |
| `UB.10` unparked, `ACTED`/`DISPOSITIONED` split, queue provenance | instruments |

**The converse — what has no passing spec at all.** 12 of 23 commitments have
live claim specs and nothing passing; five of those are FINDING 1's
claim-dead-in-fact. The three the instruction names specifically:

- **Curiosity** — 2 PASS (`T2.08`, `T2.09`), but the ablation that would prove
  curiosity *earns its parameters* (`T3.06`) is VOID-FORECLOSED, and `CU.1`–`CU.7`
  are all blocked←`T2.01`.
- **All-senses fusion** — 23 specs, **1 PASS** (`UB.9`). `UB.10` (frees 4) is
  the live unit; `UB.14` foreclosed on the venue; nine more blocked←`T2.01`.
- **Learning by living** — `W.1` FAIL, `W.2` FAIL, `LT.01` FAIL, `DP.05` FAIL,
  `SH.02`/`DP.04`/`SM.03` pilot-blocked, `LC.03` foreclosed. Nine independent
  instruments now say the same thing about W0, and the `w0-too-shallow` row
  (OPEN, 8 d, DUE 2026-09-06) is where that lands.

## 4. Is the builder alive and productive? — alive, disciplined, and net −1

- **25 iterations in 24 h, 25 ended `rc=0`.** No paused loop, no credit
  exhaustion, no aborts on load. Load average 0.08; 16.4 GB available; only the
  D1.0 watcher (pid 4187660, 10 h 31 m) and this audit are running.
- **PASS delta: 94 → 93 (−1).** Registry 211 → 217. The single loss is
  `T2.10` re-bought RED under the strictly-harder paraphrase conjunct
  (`b4805ac`) — **a gain in ledger trustworthiness recorded as a loss on the
  scoreboard**, which is the system working.
- **All 13 of today's iterations ended `93 -> 93`.** The last five self-describe
  as "hold slots" behind the D1.0 GPU lock and spent their unit on bookkeeping.
  I checked whether that is idling: it is not. `coverage` shows `cpu<1min`,
  `cpu<10min`, `cpu<2h`, `cpu<48h` and `gpu<20min` all empty, three of them with
  **no path in at all** — nothing to implement, nothing to pilot. The CPU lane is
  genuinely dry. That is FINDING 2, not a builder fault.
- **Model discipline clean.** The gate `week:all models` read 23→29% across the
  day and was named in every slot as the meter acted on, per D14.
- Two `LEFTOVER=1` events (08-31 19:12, 09-01 04:27); both transient audit-side
  processes, both gone. No leak now.

## 5. Compute honesty — NO FINDINGS

**Kaggle week `2026-W35` (resets Sunday 2026-09-06): 11.62 h spent of 30,
≈18.4 h remaining.** Colab 0.27 h.

**D1.0 is 10.61 h of that 11.62 h (91%)** — pilot 0.50 h + kernel 1 (4.08 h,
`ok`) + kernel 2 (6.03 h, `ok`, harvested 12:19). Kernel 3 dispatched 12:19 UTC
at HEAD `566f840`, est 8.6 h, timeout 32,000 s; watcher verified alive.

**Zero ledger rows for those 10.61 h — and that is correct, not waste.**
`_KERNEL_SPLIT` is by arm (`("aprime","d_mlp")`, `("b_split",)`, `("c_e2e",)`),
so the verdict lands at kernel 3's harvest. This is the best-targeted GPU spend
the project has made: D1.0 is the bakeoff for `T2.01`, the single spec blocking
38. The only other W35 spend, `T2.14` at 1.005 h, is PASS.

**The two prior weeks under-spent badly** — W33 7.63 h, W34 1.62 h of 30 each.
That is the 61-hour scar; `D15` is armed against it, due 09-05.

**Attribution, measured and reported honestly:** 22.87 of 42.76 recorded
charge-hours (53.5%) carry an empty `spec` field in `gpu_submissions.jsonl`.
The headline is misleading on its own and I will not report it as a live
problem: the gap is overwhelmingly historical — ISO-week 33 holds 18.88 h of it,
against **0.50 h in the current week** (the D1.0 pilot, trivially identifiable
from the commit record). It is shrinking on its own. `T0.12` claims only *"every
GPU run debits a weekly budget file; the ladder refuses to launch past quota"* —
which is true — and never claimed per-spec attribution, so there is no green
certificate standing over a blind ledger here.

**The two uncommitted files** (`gpu_budget.json`, `gpu_submissions.jsonl`) were
written by the watcher at 12:19, 26 minutes before this audit, and carry kernel
2's 6.03 h charge. The 13:0x iteration commits them, exactly as `e1543e5` did
for kernel 1. Normal cycle.

## 6. Stuck decisions — NO FINDINGS

- **No `MEANS-ESCALATED`.** Nothing a measurement could settle is on the owner's
  desk. The D1 disease is absent.
- **No `OVERDUE`.** Three armed and live: `D15`, `D16` (due 09-05), `D17` (due
  09-07).
- **Eleven defaults fired 09-01 00:13–00:37** (D1, D3, D4, D7, D8, D9, D10, D11,
  D12, D13, D14). I spot-checked D1, D4, D7, D10 against the safety clause: each
  is **strictly narrowing**, none edits `GOAL.md`, none moves a threshold, each
  records its losers and a reversal path. D1 struck its own option A as
  unconstitutional rather than taking it. D4 explicitly struck option 3
  ("cut the envelope") because *"law 4 forbids buying hours by weakening a
  gate"*.
- **Nothing acted on without a record.** The only 7-day change to `GOAL.md` or
  `SYSTEM.md` is `09f06f3`, which **removed a false enforcement claim** from
  SYSTEM.md ("decisions.py enforces this" when it enforced `len(default)>0`) —
  a strengthening, landed 14 hours before eleven defaults fired.

## 7. Bakeoff hygiene — no violation, one thing worth naming

**`D10` seats `wm-latent` on the Learning-core seat off a bakeoff that returned
VOID** — `LC.03` v2, *"fewer than two learners (1 cleared)"* — while SYSTEM.md's
own decision table says `VOID : an arm failed the learning gate; fix the arm, do
not decide`.

It is not hidden and it is not a violation. The seat is marked **BY VERDICT
(single-arm)** with *"the verdict is a one-learner screen, not a won bakeoff"*
on its face in `CHAMPIONS.md`; **adoption is still gated** behind `LC.07` (~10×
scale transfer, registered in the same commit, `depends_on` LC.00–LC.02/PS.01/
XL.00 all PASS, so the seat is contestable today) and then the standing unison
gates; three re-open triggers are pre-registered. Four of five arms missed the
3σ null gate with every control on its pre-registered side, so *"the screen IS
the arbitration when it returns exactly one"* is a defensible reading, and it
was armed on 08-24 with the owner given until 08-31 to veto.

I name it because it is **the only place in the record where a VOID produced a
seat**, and the entire thing that keeps it honest is `LC.07` actually running.
If `LC.07` is ever quietly deferred, that caveat becomes a champion with no won
contest behind it. Watch, do not act.

No winner was chosen inside a noise margin; no VOID elsewhere is treated as a
verdict.

## 8. The honest summary — closer to a trustworthy machine, not closer to Jack

**93 PASS. 43 of them (46%) are `T0.*`/`T1.*` harness and primitive specs. Only
12 are in Tiers 2–6, the capability ladder proper.**

And the Tier 2–4 headline claims read: `T2.01` *locomotion beats random* **FAIL**
· `T2.05` *world model beats constant prediction* **FAIL** · `T2.07` *grounding
generalises* **FAIL** · `T2.10` *memory beats recency* **FAIL** · `T4.02` *no
modality collapse* **FAIL** · `T3.06` *ablate curiosity* **VOID-FORECLOSED**.
Jack cannot walk, his world model loses to *"next state = current state"*, his
grounding does not generalise, his memory does not beat recency, and one
modality dominates the gradients.

What today actually bought: a two-meaning status token given teeth (it had
parked a frees-4 spec for seven days), four constitutional citations turned into
real specs, one capability honestly removed from the board, a queue row marked
after a day of reading "to register", and 10.6 GPU-hours committed to the one
bakeoff that could unblock 38 specs. **Nine of those ten items are about the
machine. The tenth is about Jack and it has not landed yet.**

So, directly: **are we closer to a curious humanoid that climbs the ladder than
yesterday? No.** Are we closer to a machine that will not lie to us about it?
Yes, measurably — the net −1 is the proof, and the loop took it without
flinching.

**DRIFTING, not ON TRACK**, because the instrument ranked first in this audit's
own instructions reports `0 CLAIM-DEAD` on a day when five of the owner's
commitments have nothing runnable behind them, and because that was written down
two days ago and not acted on. **Not INTEGRITY RISK**, because the ledger itself
is clean: zero violations across sections 1 and 2 under mechanical checks, no
threshold weakened, no control removed without a stronger replacement, nothing
silently amended.

---

# FOR THE BUILDER

**B1 (top priority) — `coverage.py` must count REACHABILITY in CLAIM-DEAD, the
way `champions.py` already does in `ARENA-UNREACHABLE`.** This is the 54th
audit's own §3.d insight, shipped for one tool and never for the other.

1. `claim_reachability()` (`coverage.py:398`) gains a fifth state,
   **`FORECLOSED`**, returned when the spec is declared `VOID-FORECLOSED`
   (`protocol.void_foreclosed`) **or** is gate-provisional with a measured
   pilot-blocked record. Use the **same conjunction `queue_depth` already
   computes** for its PILOT-BLOCKED / VOID-FORECLOSED blocks, factored into one
   helper so the two readers cannot drift — exactly the pattern `_split_foreclosed`
   established against `queue_depth` in `404e25a`.
2. `_claim_dead(r)` (`coverage.py:437`) gains a third condition: a commitment is
   also claim-dead when it has no PASS and every claim-kind declaration is
   `PARKED` **or** `FORECLOSED`.
3. **ADD the class, do not convert it.** The CLAIM-DEAD total must go **0 → 5**.
   This is the exact tidy-up-lowers-its-own-number shape that `T0.31` P4/P5/P6
   exist to catch; if the number does not rise by 5, the patch is wrong.
   Expected: balance, smell, shelter/building, thermal (kills), fast/slow.
4. **Known-answer fixture**, in the `_void_foreclosed_fixture` /
   `_pilot_blocked_fixture` idiom: a synthetic commitment with one `FORECLOSED`
   claim and one `PARKED` claim must read CLAIM-DEAD, and the **current** code
   must be red against it (teeth verified, not asserted).
5. Re-buy `T0.21` from a clean tree — its certificate covers `coverage.py` and
   will decay by `IMPL_DEPS` as designed.

**Do not repair the five commitments by deleting or quieting anything.** The
only honest repairs are registering a successor spec or re-parenting the
foreclosed one — the same rule that governs `ARENA-MISSING`.

**B2 — route a `REVIEW_QUEUE` row for the five, and bundle it.** Suggested id
`five-commitments-are-claim-dead-behind-foreclosures`, `DUE: 2026-09-06` so it
rides the window that `w0-too-shallow`, `ba03-null-saturates-the-horizon`,
`t306-matched-magnitude-noise-buys-coverage` and `reparenting-the-welded-fifteen`
already share. Four of the five (balance, smell, shelter, thermal) are downstream
of the same W0 venue findings, so they sequence into **one** edit window — which
is what the bundling rule is for. Staleness bill: compute it, but note `BA.03`,
`SH.02`, `SM.03` have no PASS certificates, so the bill is likely small.

**B3 — nothing ratchets the unreachable fraction.** `run blocked` prints *"85 of
217 specs are unreachable"* and no gate reads it. Add a shrink-only baselined
counter in the `QUEUE_EMPTY_BASELINE` idiom: growth is permitted only with a
named justification in the commit that grows it. Evidence for the baseline:
80/211 (38%) on 2026-08-31 (55th audit), 85/217 (39%) today. Without this, a
foreclosure that welds a new subtree lands silently and only an overseer who
happens to print the number can see it.

**B4 (carried, low) — `gpu_submissions.jsonl` should never write an empty
`spec`.** `_dispatch_pilot` in `d1_0_control_path_bakeoff.py` wrote `spec: ""`
for the 0.50 h pilot on 09-01. The historical 22.87 h gap is not worth
back-filling, but new rows should not add to it: pass the spec id (with a
`:pilot` suffix, as `T2.04:probe` already does).

---

# FOR THE OWNER

**1. Five of your commitments have nothing runnable behind them, and the
instrument built to catch exactly this reports zero.**

> *too cold kills him* · *he builds a shelter* · *smell* · *balance* ·
> *fast and slow in one brain*

This is not neglect. Each has a spec, each spec was built and run or piloted,
and each hit the same wall: **W0, the world as built, cannot grade the claim.**
`SH.02`'s pilot record says it outright — *"the null already holds the roof it
was placed under and no choice can show above it… this is D10 evidence that W0
is the bottleneck."* This is now the ninth independent line of evidence pointing
at the world, and it is the strongest one yet, because it is stated in the units
of your own constitution rather than in spec ids. The `w0-too-shallow` Review
row is where the redesign lands, **DUE 2026-09-06**.

**2. Three armed defaults fire this week unless you rule.** `D15` and `D16` on
**2026-09-05**, `D17` on **2026-09-07**. `D16` deliberately chooses to leave
`T0.27` permanently RED rather than take the option that would make the ladder
green, *"because the party proposing (c) is the party it would exonerate."* I
think that is the right call, and you should know it is being made in your
silence rather than by you.

**3. Kaggle: 18.4 h of 30 remain, resetting Sunday 2026-09-06.** 10.6 h are
already committed to `D1.0`, the control-path bakeoff, with its third and final
kernel in flight. `D1.0` is the only thing on the board that can unblock
`T2.01`, and `T2.01` blocks **38 specs** — the entire curiosity family, nine
unison specs, six of seven Tier 5 claims. If it lands, next week is the first in
three with a real dispatch queue. If it VOIDs, the ladder has neither CPU work
nor GPU work, and the answer is the world redesign in item 1.

**4. The honest one.** 93 green ticks, 43 of them harness. Twelve in the
capability ladder proper. The ladder got more trustworthy today and Jack did not
get closer to climbing anything. That is the correct trade for one day. It is
not the correct trade for a month, and `D1.0` landing this week is what decides
which one we are in.
