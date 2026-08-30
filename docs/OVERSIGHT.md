# OVERSIGHT — 48th audit, 2026-08-30 00:55 UTC

## VERDICT: DRIFTING

Not for dishonesty and not for idleness — both are in better shape than at any
point this week. **12 of 12 builder iterations on 08-29 ended `rc=0`, zero
`PACING:` skips since 12:07, and the 13th is running as I write** (started
00:07, 35 min into a 50 min budget, two commits already landed). Two of
yesterday's units ended in outcomes that bought nothing green — `T2.11` PARKED
on a control that beat the claim arm, `T3.06` VOID on both pilot families — and
in neither case did a bar move. I checked `T3.06`'s every constant across all
four of its commits: **not one threshold changed**, through two pilots that both
saw numbers. That is the four laws working.

The drift is that **every red the instruments raise is being carried, not
cleared**, and the ones being carried are all the same species: work that has to
be *written* rather than *run*. `coverage --check` exits **2**. `champions
--check` prints **12 violations**. `W.1`–`W.7` have now gone unregistered
through five consecutive audits and `CHAMPIONS.md` has not been edited in five
days. `decisions.py` still enforces none of the safety clause `SYSTEM.md` says
it enforces — unchanged since 2026-08-24, across three audits — and **11
pre-registered defaults fire in about 23 hours.** One of them takes the ladder
from `0 CLAIM-DEAD` to `1` on a sense the owner listed as constitutional.

And W35's **30 free Kaggle hours opened 55 minutes ago against a GPU shelf with
zero dispatchable specs on it.** That is the fourth consecutive week — but for
the first time the audit lands on day 1 of the window rather than eight hours
from expiry, and three of the fills are one iteration each.

Ranked by damage to the trustworthiness of the ledger.

---

## RANK 1 — the instrument that certifies `coverage.py` has a stale PASS, and `coverage.py` is step one of this audit (new)

`run status` reports three stale rows. Two are real staleness against a moved
dependency, and the previous audit missed one of them:

```
T0.21  PASS   ran 2026-08-29T15:13:19 at 5989ea7   impl_sha 80d03e8e -> d34089e1
       IMPL_DEPS = ["experiments/coverage.py"]
       coverage.py last moved 2aa789d, 2026-08-29 20:18:55   <- 5h05m AFTER the run

T0.12  PASS   ran 2026-08-20T01:22:46 at b062ccd   impl_sha c5456665 -> 85582f28
       IMPL_DEPS = ["experiments/gpu.py"]
       gpu.py last moved cbc64a6, 2026-08-29 12:22:02        <- 9 days AFTER the run

T0.27  FAIL   ran 2026-08-29T14:18:28 at 09008cc   impl_sha db4d1476 -> 97844470
```

**Why `T0.21` is the one that matters.** It is the spec whose whole job is *"the
coverage audit is honest"* — the 12-property certificate behind the tool this
audit is instructed to run first, before any other question. `coverage.py`
gained `fillable` and `empty_unfillable` at 20:18 last night, in the same
iteration that implemented `T3.06`, and `T0.21` has not been re-run since. So
the first number in this report — `QUEUE DEPTH ... 1 FRESH dispatch`,
`gpu<20min EMPTY <- fillable today: T3.10` — comes out of code that no passing
certificate covers.

**I verified the new logic by hand rather than trusting it,** and it is correct:
`T3.10` is `Budget.GPU_SHORT`, unimplemented, `depends_on=["T2.03"]` which is
PASS — it genuinely fills `gpu<20min`. `T2.14` (deps `T1.13`, `T1.08`, both
PASS) and `VO.02` (dep `VO.01`, PASS) genuinely fill `gpu<2h`. The instrument is
telling the truth. It just has no certificate saying so, which is the state
`T0.15` was in for eighteen days.

The precedent is on the record and was set by this builder eleven hours earlier:
`19df4b1` re-ran `T0.17`/`T0.20` *"after the coverage.py change — the sibling
instruments that import it are unbroken."* The same courtesy was not paid to the
instrument that owns the file. Both re-runs are seconds of CPU (`T0.21`'s last
run: 2.51 s). See **B2**.

*One cosmetic correction while I was in there, so it does not read as a
rollback later:* `6a5f626`'s message says *"T0.21 attempt 23 PASS"*; the row it
wrote is `attempt 22`. The row is the latest and the run is real — the message
is off by one.

## RANK 2 — a dated ladder regression fires in ~23 hours, and the repair its own tool prescribes has not been written (new framing; the 47th raised the hazard, not the fix)

`D8` and `D9` both fire **2026-08-31**. Both defaults park `BA.02`.

```
coverage: balance   2 specs  0 pass  1 now   claims: BA.02 RUNNABLE
                    [support passing, not credited: BA.01 (sensor)]
```

`BA.01` is a *sensor*-kind declaration. `BA.02` is the **only claim-kind spec
behind `balance`**, which `GOAL.md:41` lists in the constitutional sensory
inventory. When those defaults fire, `coverage --check` goes from `0
CLAIM-DEAD` to `1`.

**The 47th audit called this default unsafe. On re-reading `D8`/`D9` in full, I
do not think that is the right charge, and the distinction matters.** The armer
anticipated exactly this and wrote it down (`DECISIONS_NEEDED.md:2333`): *"If
these defaults fire, `balance` joins `shelter/building`, `smell`, `thermal
(kills)` and six others as a commitment with nothing passing and nothing
runnable — 10 of 23. That is the honest state either way; parking makes it
VISIBLE."* That is a defensible reading, and both defaults are genuinely
narrowing — neither edits `GOAL.md`, weakens a threshold, or widens what may be
claimed.

**The defect is not the default. It is that the successor is missing.**
`coverage.py`'s own docstring states the rule: *"The repair for a red here is to
REGISTER a successor spec, never to unpark or quiet the tool."* Parking `BA.02`
with a balance successor registered costs nothing and keeps the ratchet at zero.
Parking it without one spends the ratchet to buy visibility the ladder already
had. Nobody has written the successor, and it is the cheapest item on this
page — one registry entry, no run, no GPU, no owner ruling. See **B1**.

## RANK 3 — W35 opened with 30 free GPU-hours and a shelf with nothing on it; this time the audit is on day 1, not the last night

`gpu_budget.json`, by week:

| week | Kaggle charged | of 30 | forfeited at expiry |
|---|---|---|---|
| W32 | 16.61 h (+6.38 h unattributable opening balance) | | ~8.8 h |
| W33 | 7.89 h | | **~22.1 h** |
| W34 | **1.62 h** | | **~28.4 h**, expired 2026-08-30 00:00 UTC |
| W35 | 0.00 h — opened 00:00 UTC today | | 30 h at risk |

**Cumulatively ~59 forfeited free GPU-hours over three weeks.** As the 47th
audit found and I confirm: there is no *waste* here. Every charged hour has a
ledger row behind it — `T2.09`'s carries `gpu_job_id
jannolouwrens/jack-ladder-1788031002`, `compute_s 3307.82`, and the budget file
carries the 0.9188 h charge. The loss is entirely the other failure: nothing was
implemented to spend the hours on.

The shelf right now, verified spec-by-spec against the registry rather than
taken from the tool:

```
gpu<20min   EMPTY   fillable by T3.10  (GPU_SHORT, dep T2.03 PASS, unimplemented)
gpu<2h      EMPTY   fillable by T2.14  (GPU, deps T1.13/T1.08 PASS, unimplemented)
                                VO.02  (GPU, dep VO.01 PASS, unimplemented)
gpu<8h      T2.02   VOID — an arm to repair, not a dispatch
```

**Zero fresh GPU dispatches exist in the project.** The single fresh dispatch
anywhere is `T3.06`, and last night's iteration correctly re-priced it
`Budget.GPU -> CPU_LONG` on measured wall time (~4.2 s/life, no gradients in the
rig) — which is the honest call and also removes the last spec nominally
stocking a GPU class. The commit says so in as many words: *"leaving it GPU
would have this spec stock the `gpu<2h` queue class it can never honestly spend
a Kaggle hour on."*

The difference from the last three weeks is timing, and it is the whole point:
this audit lands **on the first day of a seven-day window**, not eight hours
before it closes. Three specs are one implementation iteration each. See **B3**.

## RANK 4 — `decisions.py` enforces none of the clause `SYSTEM.md` says it enforces, with 11 defaults firing tomorrow

`git log -- experiments/decisions.py`: last touched **2026-08-24 10:54** (`d97c33f`).
The 41st audit (08-28) filed this. The 47th (08-29) filed it as **B2**. Nothing
has moved in six days, and the clock has:

```
armed, firing 2026-08-31 (11):  D1 (costs 38 specs) · D10 (8) · D4 (8) · D3
                                D7 · D8 · D9 · D11 · D12 · D13 · D14
armed, firing 2026-09-05 (2):   D15 · D16
ratchet: 0/10 UNDECLARED · 0 MEANS-ESCALATED
```

`SYSTEM.md:126–133` says a default *"may only pick among already-permitted
actions … `experiments/decisions.py` enforces this."* `audit()` checks
`UNDECLARED`, `CLASS`, `MEANS-ESCALATED`, `NO-DEFAULT` and `DATE`, and touches
`default` exactly once — as a non-empty-string test. The invariant is prose in a
docstring, enforced by nobody, on the day before eleven of them fire.

`D13` still names `SY.01` as the experiment that would settle it, and `SY.01`
still occurs **nowhere in the repository except inside `D13` and inside the last
two audits' write-ups of `D13`.** It is not in `BY_ID` (187 specs). `champions.py`
resolves arena ids against the registry; `decisions.py` resolves nothing.

## RANK 5 — architecture: the ratchet has not moved, and `CHAMPIONS.md` has not been touched in five days

`champions --check`: **12 violations, 6/8 seats with a phantom arena** —
identical to the 47th audit. `git log -- docs/CHAMPIONS.md`: last edit
**2026-08-25**.

- **`ARENA-MISSING` (5 seats):** World (held **BY VERDICT**, the strongest
  marking in the file, rematch trigger aimed at `W.1`–`W.7`, none of which
  exist); PLASTIC-ONLY decree (`PL.*`, `PL.00`, `PL.02`); Curiosity signal
  (`LT.03`, `LT.04`); Control architecture (`D1.0`, `T2.21`); Audio encoder
  (`PL.*`). Vision encoder names `PL.02`.
- **`NO-ARENA` (3):** ASR, Speaker ID, Language grounding — no spec id named at
  all, so nothing that could ever be run would unseat the holder.
- **`UNCONTESTED` (3):** Fast/slow coupling and Language model / Language
  acquisition, held BY DECREE with real arenas (`DP.02`, `LG.00`) that have
  never run.

This is the **fifth consecutive audit** to report `W.1`–`W.7` (the 47th recorded
it as the fourth). See the lesson appended to `LESSONS.md` today: nothing in
this project ranks registration debt, which is why it is the one category that
survives every audit intact.

---

## Section-by-section

**1. Ledger integrity — clean, and I widened the check rather than repeat the
last one.** Over all 98 rows (85 PASS / 10 FAIL / 3 VOID):

```
PASS rows whose `commit` no longer resolves in git ......... 0 / 85
PASS rows carrying a `+dirty` commit stamp ................. 0 / 85
PASS specs with no implementation in experiments/tests/ .... 0 / 85
PASS specs with an empty `null_baseline` ................... 0 / 85
PASS specs declaring no `control` .......................... 2  (T0.01, T0.10)
PASS rows with empty `control_metrics` ..................... 2  (the same two)
```

The two exceptions are **self-consistent** — `T0.01` (repo imports clean) and
`T0.10` (Kaggle round-trip) declare no control and record none, so there is no
case anywhere of a control that was declared and never run. Both are harness
liveness checks where a control is arguably meaningless; neither says so in its
spec text, which is the 47th audit's **B6** and still open (**B6** below).

`audit_supersedes_fail` over the live ledger: **1 violation, 5 checked_pairs, 24
unauditable_pairs** — byte-identical to yesterday. `T0.17`'s
`2026-08-29T13:14:23` FAIL at `d84101e+dirty` remains the single violation and
remains, on the commit-message evidence, a half-written test rather than a moved
threshold. **66 of 383 recorded runs carry a dirty stamp**, which is why
`unauditable` is five times `checked`. `T0.27` stays **RED**, correctly, as
`D16`'s armed default requires be reported here.

**2. Thresholds and controls — one bar moved downward in seven days, I replayed
the counterfactual, and it did not change the verdict.**

Seven days of `git log -p` over `registry.py`, `registry_expansion.py` and
`experiments/tests/` turns up exactly one movement in the loosening direction,
and it is disclosed at length in the file it lives in.

**`T2.09`, `DECAY_MIN` 1.5 → 1.25**, moved at the pilot freeze (`44f24c4`,
19:16 on 08-29 — after the 47th audit was written, which is why it is new here).
The docstring's justification: the bar is a placeholder being frozen for the
first time on a spec that had never run, and 1.25 is anchored on the mechanism
(a constant or dead signal has decay identically 1.0) rather than shaved to the
observed minimum.

**Half of that argument is exogenous and half is not, and it should be said
plainly.** The floor (1.0) is a property of the mechanism. The interpolation
endpoint (seed 90's observed 1.472) is a draw. Had seed 90 read 1.1, the same
reasoning would have produced a bar of 1.05. That is a bar that bends to the
pilot, however well-narrated.

**So I replayed the registered run against the unmoved bar, from the recorded
`per_seed` table.** With `DECAY_MIN = 1.5`, seed 0 (`static_decay 1.424`) drops
out of the informative set; live seeds become {1, 3, 6}, `n_informative = 3`,
which still meets `MIN_LIVE_SEEDS = 3`, so no VOID. Re-folding every gate over
that subset:

```
claim_dwell        max(0.0126, 0.0504, 0.0148) = 0.0504  <= 0.20   PASS
claim_fed_ratio    max(0.889,  1.400,  1.223)  = 1.400   <= 1.50   PASS
coverage_frac      min(1.0,    1.0,    1.0)    = 1.000   >= 0.80   PASS
trap_dwell         min(1.0,    0.7137, 0.8926) = 0.7137  >= 0.40   PASS
trap_fed_ratio     min(9.5e11, 8.321,  2.864)  = 2.864   >= 2.00   PASS
exposure           min(0.9169, 0.9257, 1.0012) = 0.9169  >= 0.50   PASS
claim_static_decay min(1.597,  2.762,  2.265)  = 1.597   >= 1.50   PASS
dwell_margin       min(0.9874, 0.6633, 0.8778) = 0.6633  >= 0.15   PASS
control (must fail) max(1.0, 0.7137, 0.8926)   = 1.000   >  0.20   FAILS ✓
```

**`T2.09` PASSES at the unmoved bar.** The move cost the run one informative
seed and nothing else, exactly as the docstring claimed (*"would have … cost a
seed for nothing"*). The finding is that the claim survives its own bar's
history — which is the only thing that settles a moved threshold.

Everything else moved in the tightening or truth-correcting direction and was
measured in its commit message. **`T3.06` is the model case and deserves to be
named:** I diffed every module-level constant across all four of its commits
(`2aa789d` → `17e6c4d` → `bf947a1` → `1653104`), through two pilots that both
returned numbers, and the only things that changed were `LIVES_PER_ARM`
(4 → 16 → 48, a sample size) and `_GATES_FROZEN`. `DELTA_MIN`, `TASK_DWELL_MIN`,
`MIN_INFORMATIVE_LIVES`, `RANDOM_DWELL_MAX`, the random-coverage band, the
spread and t-stat factors: **untouched**. The commit's claim of *"NOT ONE BAR
MOVED"* is exactly true.

*One readability trap left behind:* `TASK_DWELL_MIN = 0.10` still carries the
inline comment `# placeholder: …` in a file whose gates are now FROZEN. A frozen
bar that calls itself a placeholder invites the next reader to move it (**B6**).

**2b. I bounded last night's own lesson, because the builder left it
unbounded.** `LESSONS.md` gained *"A worst-case instrument gated on the SEED
MEAN is not a worst-case instrument"* at 00:25 today, and the `REVIEW_QUEUE`
entry says *"26 spec files fold a worst-case quantity and nothing mechanical
tells a correct gate from a wrong one."* That reads as an open exposure across
the ledger, so I measured it.

Scanning every `_check` for metric keys shaped `worst|_min|_max|n_*`, restricted
to multi-seed specs that are PASS and do **not** fold inside `_experiment`:
**8 specs.** For each, the recorder emits `<key>_std`, and for n=3 with ddof=0
no seed deviates from the mean by more than `sqrt(2)·std` — so the worst
possible seed is computable from the ledger alone.

```
ME.11.0  T2.03  T2.06  T3.01  T6.03  PS.01   — every gate key std = 0.0
                                               (seed-invariant; the mean IS the worst seed)
PG.8   settle_qvel_max  0.01936 ± 0.00694  worst seed <= 0.0292  vs bar 0.50   17x headroom
VO.01  mute_r2_max     -0.26616 ± 0.12373  worst seed <= -0.0912 vs bar 0.05   clears
```

**Live exposure on the ledger today: zero.** No PASS row is carried by a
worst-case gate that a per-seed value would have failed. Stated with its own
limit, because the bound is only as good as the scan: I keyed on the *name
shape* of metric keys, so a worst-case quantity named `dwell_lo` or `slowest`
would be missed, and the number is "zero among the 8 name-matched multi-seed
PASS specs", not "zero across all 26 files". The durable version is a T0
property, not an audit — **B4**.

**3. Drift from the goal — none in what was built; the gap is in what was not.**

Thirteen iterations since 12:07 on 08-29. The eight since the 47th audit:

| iteration | work | GOAL.md sentence served |
|---|---|---|
| 18:07 | `T2.19` PASS harvested (83→84); `T2.09` implemented | *"components must EARN their parameters"*; *"he explores because he wants to"* |
| 19:07 | `T2.09` gates FROZEN from the pilots, worst-informative-seed fold, dispatched into W34 | curiosity — and it discharged the 47th's **B1** with 5 h to spare |
| 20:07 | **`T2.09` PASS (84→85)**; `T3.06` implemented | *"he explores because he wants to"* — the noisy TV does not capture the signal |
| 21:07 | `T2.11` implemented (DIAYN vs an independent classifier); pilot: **the control passed** | *"really learning, not appearing to learn"* |
| 22:07 | `T2.11` repair implemented and re-piloted (shared skill-conditioned policy) | same |
| 23:07 | **`T2.11` PARKED** — the repair worked and the outcome did not move | law 2, taken at the cost of a dispatch |
| 00:07 | `T3.06` v2 pre-registered, then piloted: **both families VOID**, gates frozen, no bar moved | curiosity ablation — *"every component ablated; dead weight deleted"* |

**No drift. Every unit traces to a GOAL.md sentence,** and three of the seven
ended in an outcome that produced no green tick at all. The ratio has also
flipped in the right direction since the 47th complained about it: **six of
these eight built Jack, two built instruments** — the inverse of the
five-instruments-to-two ratio the last audit measured.

The converse question, which is where the real answer is. `coverage` reports
**0 commitments with no declared spec** — the ratchet holds — and **14 with live
claim specs and nothing passing**: proprioception, plasticity, sleep (4 specs),
fast/slow (8), hunger/thirst (5), death-and-retry, social, thermal, shelter,
voice, balance, smell, tool use, touch. The three GOAL.md flags most at risk of
quiet neglect sit where they sat yesterday: **curiosity 12 specs / 2 pass**
(up one — `T2.09`), **one brain and unison 21 specs / 1 pass**, **generality
4 / 1**. Twenty-six of the 85 demonstrated rungs are Tier 0 + Tier 1 harness.

**4. Builder alive and productive — the best day in a fortnight.**

```
2026-08-29:  12 iterations started, 12 ended rc=0, 12 PACING skips (all before 12:07)
2026-08-30:   1 iteration started 00:07, ALIVE at 00:55 (pid 2833493, 35 min into `timeout 50m`)
              two commits already landed this iteration: bf947a1 (00:13), 1653104 (00:25)
PASS delta over 24 h:  84 -> 82 -> 83 -> 84 -> 85   net +1
```

No paused loop, no repeated identical failure, no orphaned process (`pgrep
python` shows only tenant services), load nominal, `/data` at 21%, tree clean,
`HEAD == origin/main`. Every iteration since 12:07 yesterday fell back
`fable -> opus` in ~3 s because `week:Fable` is capped until 08-31 04:59 — the
silent-fallback behaviour `D14` exists to rule on, still firing 12 times a day
and still costing the shared meter it was built to protect.

**5. Compute honesty — no waste; the loss is unspent, not misspent.** See RANK 3
for the table. Every charged hour maps to a ledger row; `T2.09`'s row carries
its `gpu_job_id`, `compute_s 3307.82` and `hardware remote/Tesla P100`, and the
budget file carries the matching 0.9188 h charge. Two artifacts written last
night (`/data/t3_06_pilot_v2.json` 00:22, 5.2 KB) are present and were harvested
in-iteration. **W35: 30 h available, 0 h charged, 0 fresh GPU dispatches
implemented.**

**6. Stuck decisions.** `0 UNDECLARED`, `0 MEANS-ESCALATED`, **nothing
OVERDUE** — every armed default is dated 08-31 or 09-05 and none has ever been
extended (the 47th audit checked all 211 `decide_by` values in git history; I
re-confirmed the live file shows only those two dates). Nothing has been quietly
acted on. The live problems are RANK 2 (`D8`/`D9` fire without their successor
spec), RANK 4 (the invariant is unenforced and `SY.01` is a phantom), and the
structural one: **`D1` has been armed and blocking 38 specs for six days, and
its default is due in 23 hours.** Under `SYSTEM.md` rule 3 the bakeoff that
would settle it was always the loop's to write, and it has not been written; the
default will now settle by clock what a measurement was supposed to settle. That
is a legal outcome, and it is not the one rule 3 was for.

**7. Bakeoff hygiene — no findings.** `DECISIONS_RESOLVED.md` is unchanged and
reads clean: `PS.01/J` VOID on three arms below the 3.0-sigma learning gate,
re-run as `PS.01/J2` to a real winner (`impact_speed`, 2.66 sigma over the
runner-up, 10.32 over the null, with a stated `screen rationale` for why
observables may be screened rather than raced); `D2` settled by ledger replay
with its loser and a re-open trigger recorded. No VOID is treated as a verdict,
no winner sits inside the noise margin.

The uncomfortable true thing about this section: **the file still has three
entries, one of which is a VOID and one of which was not a bakeoff.** `SYSTEM.md`'s
third law — *decisions are made by bakeoff, never by argument* — has been
exercised on essentially one real question in the project's history, while
`LC.03` sits VOID, `LC.04` NOT_RUN, `DP.02` NOT_RUN, `LG.00` NOT_RUN and `D1`
prepares to resolve itself by calendar.

**8. The honest summary — closer to a curious humanoid, or just to a longer list
of green ticks?**

**Yesterday: genuinely closer, and by an unusual route.** `T2.09` is a real
finding about curiosity — an ensemble-disagreement signal that does *not* get
captured by a noisy TV, where the ICM control fixates completely (`claim_dwell
1.0` against the claim arm's 0.078, margin 0.6633 on the worst informative seed)
and the spec's `kills: ICM alone` fires exactly as registered. That is
`GOAL.md`'s *"he explores because he wants to"* moved from assertion to
measurement.

But the better evidence is the two units that produced **no** tick. `T2.11` was
parked after its control scored 0.8984 against the claim's 0.7812 — a shipped
`SkillDiscovery` class whose docstring claims Jack "learns walking, jumping,
turning" is now, correctly, on the record as undemonstrated rather than
quietly green. `T3.06` VOIDed both pilot families and froze without moving a
bar. **A system that reports 85 instead of 87 because two things did not
survive their controls is a system whose 85 means something.** That is the
project's actual product.

**Over the week: still no.** The ladder moved **83 → 85 in seven days**, on a
board of 187. Fourteen of the owner's own commitments have live claims and
nothing passing. The three claims that *are* the thesis — curiosity, unison,
generality — stand at 2, 1 and 1 PASS. Roughly 59 free GPU-hours have expired
against an empty shelf over three weeks and week four opened an hour ago in the
same state.

**And the specific shape of the drift is worth naming, because it is not
laziness.** The builder cleared a great deal yesterday: two false greens, an
18-day-dead spec, a GPU dispatch, a park, a VOID, four lessons. Every one of
those was work it could *run*. What it did not touch — `W.1`–`W.7`, `PL.00`,
`PL.02`, `LT.03`, `LT.04`, `SY.01`, `D1.0`, `T2.21`, a balance successor — is
work it would have had to *write*. An unregistered spec has no id, sits in no
cost class, blocks nothing, fails no gate, and appears nowhere in the queue
instrument that now (correctly, usefully) drives unit selection. **The queue
made implementation debt legible and, by doing so, made registration debt the
only kind of work nothing ranks.** Lesson appended.

---

## FOR THE BUILDER

**B1 — register a `balance` successor claim TODAY; `D8`/`D9` fire in ~23 hours.**
This is the cheapest item on the page and it has a deadline. When those defaults
park `BA.02`, `balance` — `GOAL.md:41`, constitutional — becomes the ladder's
first `CLAIM-DEAD` commitment and `coverage --check` goes red on it. `coverage.py`'s
own docstring names the repair: *"REGISTER a successor spec, never unpark or
quiet the tool."* One registry entry, no implementation, no run, no GPU, no
owner ruling required. Scope it to what a body without directional catch
authority *can* be asked — `D8`'s four probes say the catch cannot be measured
in the rover, which is a statement about catching, not about balance sensing.
If you conclude no honest balance claim is registrable before a playground
humanoid exists, **say that in `DECISIONS_NEEDED.md` under `D8` in one
sentence** so the CLAIM-DEAD is a recorded decision rather than a side effect.
Do not park anything to make this easier.

**B2 — re-run `T0.21` and `T0.12`. Two invocations, ~5 seconds of CPU.**
`T0.21`'s PASS predates `coverage.py`'s 20:18 change by five hours, and
`coverage.py` is the tool the overseer is instructed to run before anything
else; `T0.12`'s PASS predates `gpu.py`'s push-guard change by nine days. You set
the precedent yourself at `19df4b1` for the siblings — the owner of the file did
not get the same treatment. If either now fails, that is the finding and it is
worth far more than the re-run cost.

**B3 — implement ONE GPU spec this week, and do it early.** `gpu<20min` is
`NEWLY EMPTY` and is why `coverage --check` exits 2. The fills, verified against
the registry:
  - **`T3.10`** — `Budget.GPU_SHORT`, dep `T2.03` PASS. Cheapest; clears the red.
  - **`T2.14`** — `Budget.GPU`, deps `T1.13`/`T1.08` PASS.
  - **`VO.02`** — `Budget.GPU`, dep `VO.01` PASS. Also the only live claim
    behind **voice**, and it discharges a `VACANT` champion seat.
30 free hours expire 2026-09-06 00:00 UTC. Three weeks of these have gone
unspent; the difference this week is that you have seven days, not eight hours.
**`VO.02` is the one I would take** — it is the only candidate that buys a GPU
dispatch, a constitutional commitment's first claim, and a contested seat with
one iteration.

**B4 — make last night's lesson mechanical, and use the bound in §2b as its
known-answer test.** You routed `aggregate-hides-worst-seed` to `REVIEW_QUEUE`
with an unbounded staleness bill; the live exposure is **zero across the 8
name-matched multi-seed PASS specs**, computed from the recorded `_std` values
(`PG.8` 17x headroom, `VO.01` clears at the provable worst seed, the other six
seed-invariant). So this is a *guard*, not a repair, and it can be built without
touching a single certificate. Minimum executable version, as a T0 property: for
every spec with `seeds > 1`, every `_check` key matching the worst-case shape
must either be produced by an in-`_experiment` fold or be bounded by
`m[k] ∓ 1.5·m[k+"_std"]` — and the property must fail loudly on a planted
violation, not merely pass on today's clean ledger. State the scan's limit in
its docstring: **name-shape matching misses a worst case called `dwell_lo`.**

**B5 — register `W.1`–`W.7`. Fifth consecutive audit.** The World seat is held
**BY VERDICT** — the strongest marking in `CHAMPIONS.md` — with a rematch
trigger aimed at seven specs that do not exist, and the file has not been edited
in five days. `PL.00`/`PL.02` (the plastic-only decree, `GOAL.md:76`) and
`LT.03`/`LT.04` (the curiosity signal) are next by consequence. `SY.01` is the
same disease inside `D13`. **The ratchet shrinks by REGISTERING, never by
deleting the reference** — deleting turns `ARENA-MISSING` into `NO-ARENA` and
makes the seat permanently safe, which is the opposite of the repair. If you
take only one: `W.1`, because the seat holding it is marked BY VERDICT.

**B6 — two one-line honesty edits, both carried from the 47th.** (a) `T0.01` and
`T0.10` are the only PASS specs with no declared control and no
`control_metrics`; that is *consistent*, not an omission, but nothing in either
spec says so — one sentence each ("a harness liveness check; a control is
undefined for it") closes it permanently. (b) `T3.06`'s `TASK_DWELL_MIN = 0.10`
still carries the inline comment `# placeholder:` in a file whose gates are now
`_GATES_FROZEN = True`. A frozen bar that calls itself a placeholder is an
invitation to move it.

**B7 — `decisions.py`, carried from the 41st and 47th, and now urgent by the
clock.** In priority order, and each is a known-positive today:
  1. **A default whose firing moves a `coverage.report()` row from a live claim
     to CLAIM-DEAD is a violation.** Computable now; `D8`/`D9` are the positive
     and they fire tomorrow.
  2. An `arena:` field in the `DECIDE:` block, resolved against `BY_ID`, in
     `champions.py`'s idiom. Positive: `D13` names `SY.01`, absent from 187 specs.
  3. Print the default in full. `main()` truncates at `[:110]`; the live
     defaults run 369–1041 characters, so 70–89% of every constitutional clause
     has never appeared in any report — including the ones firing tomorrow.

---

## FOR THE OWNER

**O1 — eleven pre-registered defaults fire tomorrow, 2026-08-31.** `D1` costs
**38 specs**, `D10` 8, `D4` 8; `D3`, `D7`, `D8`, `D9`, `D11`, `D12`, `D13`,
`D14` cost meter, GPU-hours and honesty rather than specs. **No deadline in this
project has ever been extended** — I re-confirmed the live file carries only
`2026-08-31` and `2026-09-05`. The clock is honest and today is the last day on
which a ruling beats it. If you want any of these decided rather than
defaulted, this is the window.

**O2 — one of them will turn the coverage ratchet red, and the fix is a
builder's, not yours.** `D8`/`D9` park `BA.02`, the only live falsifiable claim
behind **balance** (`GOAL.md:41`). I want to be fairer to the default than the
last audit was: the armer disclosed this cost explicitly and both options are
strictly narrowing, so the default is *legal*. What is missing is the successor
spec `coverage.py`'s own rule requires, and I have filed that as **B1** with a
deadline rather than escalating it to you. **You need do nothing here unless you
want `BA.02` kept alive** — one sentence naming `D8` option 2 or 3 reverses it
at any time, before or after tomorrow, at the cost of a registry re-parent and
no re-run.

**O3 — `D1` is about to be settled by a calendar, and it was supposed to be
settled by an experiment.** It has been armed and blocking **38 specs** since
2026-08-24. `SYSTEM.md` rule 3 — your own ruling of that date, *"in the future
he mustnt get blocked by anything like this but instead test and try and
research both and decide at end which works better"* — makes writing its bakeoff
the loop's job, not yours. Nobody wrote it. Tomorrow the default fires and the
plastic-only decree stands by clock. **That is a legal outcome and it is not the
one the rule was for**, and you should know it happened that way. No action
needed; recorded so the record is accurate.

**O4 — the fourth week of free GPU-hours started an hour ago, and the cause has
not changed.** ~59 hours forfeited over W32–W34, ~28.4 of them last week. The
cause is not waste: every charged hour has a ledger row. It is that **zero GPU
specs are implemented and dispatchable**, and three (`T3.10`, `T2.14`, `VO.02`)
are one builder-iteration away each. Filed as **B3** with a recommendation. The
one thing that would make this *your* problem again is `D13`/`D14`: they fire
tomorrow, 29 hours after the resource they were written to protect already
died — see the 47th audit's O3, unchanged.

**O5 — for information, no action, as `D16`'s armed default requires.** `T0.27`
stays **RED**. The live audit reads 1 violation, 5 checked pairs, 24 unauditable
— unchanged from yesterday. The violation is `T0.17`'s FAIL at a commit that
does not exist; the cause was a half-written test disclosed in its own commit
message the same minute, not a moved threshold. The structural repair (a policy
distinguishing a development FAIL from a refutation) is the 47th audit's **B4**
and remains open.
