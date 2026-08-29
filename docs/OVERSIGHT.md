# OVERSIGHT — 47th audit, 2026-08-29 18:40 UTC

## VERDICT: DRIFTING

The ledger is telling the truth and told it twice today at its own expense
(84 -> 82 -> 84, two false greens taken down by the builder). `T2.19` is the
best-evidenced capability claim added in a fortnight. That is the good news and
it is real.

The drift is elsewhere and it is arithmetic: **the builder was dark for 95
consecutive hours** (2026-08-25T13:07 -> 2026-08-29T11:07, 104 of 134 scheduled
slots skipped since 08-22 = 78%), it woke at 12:07 today **because a Claude
meter fell below a line, not because anybody decided anything**, and W34's
~29.3 free GPU-hours expire in about five hours with **zero fresh dispatches**
in the whole project. That is the third consecutive week of expiry and roughly
**60.6 forfeited free GPU-hours**. Meanwhile 14 of GOAL.md's own commitments
have live claim specs and **nothing passing** — thermal, plasticity, sleep,
death-and-retry, social, hunger/thirst, fast/slow, proprioception, voice,
balance among them.

Ranked by damage to the trustworthiness of the ledger.

---

## RANK 1 — the ledger cannot tell a refutation from a half-written test, and `T0.27` is red because of it (today)

`T0.27` (*a threshold moved after a FAIL leaves an artifact*) is **FAIL**,
attempt 17, at `09008cc`. Running its auditor against the live ledger myself:

```
audit_supersedes_fail(ledger["results"])
  violations       1
  checked_pairs    5
  unauditable_pairs 24
```

The single violation is dated **today**:

```
T0.17  FAIL  d84101e+dirty  2026-08-29T13:14:23  impl 072ea7a4d7
T0.17  PASS  d84101e+dirty  2026-08-29T13:15:07  impl 3656fcac07   <- 44 s later, code moved
T0.17  PASS  be60c3d        2026-08-29T13:16:00
```

A PASS superseded a FAIL at a changed `impl_sha`, and **the failing
implementation exists in no commit** — the exact T2.08 shape `T0.27` was built
to make impossible, reproduced inside the audit-instrument family on the same
day the guard was being extended.

**But read the cause before you read the crime.** `be60c3d`'s message says it
plainly: *"T0.17 PASS 4.93s (was FAIL on the round-trip sub-property before the
BLOCKED path was stamped)"*. That FAIL was not a refutation. It was a new
sub-property failing against production code that had not been repaired yet —
a build error with a verdict attached. Nobody moved a threshold. The disclosure
is honest and the red is honest.

**The finding is the structural one, and I do not think any prior audit has
stated it:** `run_spec` writes every run, so an iteration of *writing* a test is
recorded with the same status, the same generic message (`"pre-registered
threshold not met"`) and the same history slot as an iteration of *refuting*
one. There are **66 dirty-stamped runs** in the ledger. Every guard built on
"what happened after the FAIL" — `supersedes_fail`, `supersedes_void`, `T0.27`
itself — is therefore reasoning over a population it cannot clean, which is why
`unauditable_pairs` (24) is five times `checked_pairs` (5) and why `T0.27` will
stay red until someone states a policy rather than fixes a row.

Precedent notes, so this is not read as worse than it is: `cb3a301` shipped a
dirty-tree warning *at* 13:20, six minutes after the offending run, and `D16`
was raised the same hour with default (b) — *the red stands, `T0.27` is not
touched*. The system caught itself. It has not yet given itself a way not to
repeat it. See **B4**.

## RANK 2 — 11 defaults fire in two days, and `decisions.py` still enforces none of the clause that makes firing safe

`decisions --check`: **0 UNDECLARED, 0 MEANS-ESCALATED** — the ratchet is at
zero and has never been extended (I checked every version of the file in git:
**211 `decide_by` values in history, all of them `2026-08-31` or `2026-09-05`**.
No deadline has ever moved. That is a clean result and worth saying).

What is not clean: **eleven defaults fire on 2026-08-31**, including `D1`
(costs **38 specs**), `D10` (8) and `D4` (8). `SYSTEM.md:126-133` says a default
*"may only pick among already-permitted actions … `experiments/decisions.py`
enforces this"*. Reading `audit()` at `experiments/decisions.py:170-211`, it
checks `UNDECLARED`, `CLASS`, `MEANS-ESCALATED`, `NO-DEFAULT`, `DATE` — and
touches `default` exactly once, as a non-empty-string test. **The safety
invariant is prose in a docstring, enforced by nobody.** The 41st audit found
this on 2026-08-28; it is unchanged 36 hours later, with two days on the clock.

I re-verified the 41st audit's known positive rather than take it on trust.
`D8`'s default parks `BA.02`. `coverage` reports `balance` as *2 specs, 0 pass,
`claims: BA.02 RUNNABLE`* — `BA.02` is the **only live claim-kind spec** behind
a sense GOAL.md:41 lists as constitutional. Firing `D8` takes the ladder from
`0 CLAIM-DEAD` to `1`, on a sense the owner named. **A default that removes the
last falsifiable claim behind a constitutional commitment is a narrowing of what
the ladder can ever show, and that is the thing the clause forbids.** It is also
mechanically computable today, which makes it the right first guard (**B2**).

**And the `class` field is self-declared with nothing checking it.** `audit()`
flags `MEANS-ESCALATED` when the entry *says* `means`; four characters of typing
files any fork on the goal side forever. Live example in RANK 3.

## RANK 3 — a decision names the experiment that would settle it, and that experiment does not exist (new)

`D13`'s own text: *"`SY.01` (the three-arm pace-gate bakeoff, arm C = pace-gate
the auditors) is the instrument that would settle it, and it is still
unwritten."*

```
'SY.01' in BY_ID           ->  False      (187 specs)
grep -rn 'SY\.01' .        ->  1 hit, docs/DECISIONS_NEEDED.md:3021
```

`SY.01` occurs **exactly once in the entire repository — inside the decision
entry that says it would settle the question.** This is the `D1` disease and the
`CHAMPIONS.md` `ARENA-MISSING` disease at the same time: a fork sitting on the
owner's desk whose falsifier is named, costed, and never registered. `D13`'s
arms are both implemented in `scripts/ladder_loop.sh` (37th audit) — under
SYSTEM.md rule 3 that is *"an experiment somebody has not written yet"*, not an
escalation.

`champions.py` resolves every seat's arena against `BY_ID` and has driven that
ratchet 8 -> 6. **`decisions.py` resolves nothing.** A decision may name a
phantom instrument indefinitely and every instrument in this project will report
it as correctly armed. See **B3**.

## RANK 4 — the builder woke by the calendar, exactly as the 39th audit forecast

Measured from `ladder.log`, not from the loop's own account of itself:

| day | iterations started | `PACING:` skips |
|---|---|---|
| 08-24 | 16 | 9 |
| 08-25 | 7 | 17 |
| 08-26 | **0** | 24 |
| 08-27 | **0** | 24 |
| 08-28 | **0** | 24 |
| 08-29 | 7 (from 12:07) | 12 |

**Longest unbroken skip streak: 95 hours, 2026-08-25T13:07:11 ->
2026-08-29T11:07:10.** Since 08-22: 30 iterations, 104 skips, **78% of
scheduled builder time discarded**.

The 39th audit (2026-08-27) wrote: *"On 2026-08-31 both Claude meters reset, the
pace gate opens on its own, the builder wakes without anyone deciding anything."*
It happened two days early and for the same reason — `week:all models` fell from
62% to 75% against a week 79% elapsed, i.e. the line overtook the meter. **That
forecast is now CONFIRMED, not pending.** `D13` and `D14` will fire on 08-31
into a symptom that has already cleared, and will appear to have worked.

The gating meter is not driven by this box: five independent builder
observations now record a full Opus iteration moving `week:all models` **zero
points**, and the 42nd audit found `week:Fable` rising 66% -> 100% across 72
hours in which zero Fable requests were made here. The builder is throttled by
somebody else's consumption. That is a fact for the owner (**O3**), not
something an agent may fix by editing a gate.

## RANK 5 — architecture: the ratchet is shrinking, the World seat is not

`champions --check`: **12 violations, 6/8 seats with a phantom arena** (was 8 —
credit where due, `a3ed5c5` fixed the quantifier and two seats were discharged).

Still open, and this is the fourth consecutive audit to say so: the **World**
seat is held **BY VERDICT** — the strongest marking in the file — with a rematch
trigger pointing at `W.1`–`W.7`, **none of which exist**. Also missing:
`PL.*`/`PL.00`/`PL.02` (the plastic-only decree, GOAL.md:76), `LT.03`/`LT.04`
(curiosity signal), `D1.0`/`T2.21` (control architecture). Three seats
(ASR, Speaker ID, Language grounding) name no arena at all. Three
(Fast/slow coupling, Language model, Language acquisition) are `UNCONTESTED`
with real arenas — `DP.02`, `LG.00` — that have never run.

---

## Section-by-section

**1. Ledger integrity — clean apart from RANK 1.** All **84 PASS rows** have a
`commit` that still resolves in git (0 dangling). Every PASS spec declares a
control and carries populated `control_metrics` **except `T0.01`** (repo imports
clean) and **`T0.10`** (Kaggle round-trip) — both harness liveness checks where
a control is arguably meaningless, but neither says so in its spec text. Worth
one sentence each rather than a silent exemption.

`T2.19` audited in full, since it is today's only capability delta:

```
status PASS   commit 2c90fc9   seeds [0,1,2]   798.4 s   kaggle P100
spec_sha 75ccf4f5ea78aae6      impl_sha c7bc3cd825811813
CLAIM    bimodal_success_ratio 98.995 >= 10    flow worst seed 0.7734 >= 0.60
CONTROL  uni_min 1.0 >= 0.90 | uni_gap 0.0 <= 0.10 | untrained_max 0.0 <= 0.05
         shuf_mult 466.5 >= 10 | reg_shared_pass_bimodal 1.0 >= 0.90
```

Every control ran, every one landed on its pre-registered side, three seeds, and
the pilot seed (**90**) is disjoint from the registered seeds (**0/1/2**) — the
bars were frozen on held-out data. This is how it is supposed to look.

**2. Thresholds and controls — no silent loosening. Reported plainly because it
is true.** Seven days of `git log -p` over `registry.py`,
`registry_expansion.py` and `experiments/tests/` shows no threshold moved in the
loosening direction, no control deleted, no `_check` gaining an `or`, no seed
count reduced. The movements found were all in the tightening or
truth-correcting direction and all measured in their commit messages:
`SHUF_MULT` 2.0 -> 10.0, `RATIO_MIN` 3.0 -> 10.0, `UNI_MIN` 0.8 -> 0.90,
`est_hours` re-priced 0.0004 -> 3e-5 h/step against a measured 635 s.

One judgment call, disclosed and worth naming without accusing: `T2.19`'s
`STEPS` moved **300 -> 500 after the pilot**, in the direction that helps the
claim (flow bimodal success 0.836@400, 0.867@500), and `FLOW_MIN = 0.60` was
frozen from that same seed-90 run. Budget and bar were co-chosen from one
observation. The justification is on the record and is a good one — the
regression arm is pinned at 0.000 from step 100 through 1200, so extra steps buy
the null nothing — and the registered run cleared the bar at 1.29x on the worst
of three unseen seeds. It stands. It is simply the one place in this PASS where
a reader trusts a judgment rather than a pre-registration.

**3. Drift from the goal.** Seven builder iterations today, all `rc=0`:

| iteration | work | GOAL.md sentence served |
|---|---|---|
| 12:07 | commit SM.03 impl (untracked 4.5 d); `champions.py` quantifier; `queue_depth` | *"protects the honesty of watching what happens"* — system, not Jack |
| 13:07 | `spec_sha`; `T0.15` un-runnable 18 days found; **84 -> 82** | honesty; two false greens removed |
| 14:07 | `T0.13` — 5 rig-guards made falsifiable; **82 -> 83** | honesty |
| 15:07 | `queue_depth` sees a refusing spec; `T0.21` P12 | honesty |
| 16:07 | **T2.19 implemented + pilot dispatched** | *"components must EARN their parameters or be deleted"* |
| 17:07 | T2.19 bars frozen, registered run dispatched | same |
| 18:07 | **T2.19 PASS (83 -> 84)**; **T2.09 implemented** | flow head earns its place; *"he explores because he wants to"* |

**No drift.** Every unit traces to a GOAL.md sentence. But note the ratio: **five
of seven iterations built instruments and two built Jack**, and the two that
built Jack are the two the queue instrument made unarguable. SYSTEM.md's own
guard applies — *"when the machine is sufficient, PROVE it by throughput."*

The converse, which is the harder question. `coverage` reports **0 commitments
with no spec** (the ratchet holds) but **14 with live claim specs and nothing
passing**: proprioception, plasticity, sleep (4 specs), fast/slow (8 specs),
hunger/thirst (5), death-and-retry, social, thermal, voice, balance. And the
three GOAL.md flags most likely to be quietly neglected: **curiosity 12 specs /
1 pass**, **one-brain-and-unison 21 specs / 1 pass**, **generality 4 / 1**. The
ladder is 187 rungs and the parts that *are* the goal sit at one PASS apiece.
26 of the 84 demonstrated rungs are Tier 0 + Tier 1 harness.

**4. Builder alive and productive.** Alive since 12:07 today; 7/7 `rc=0`; net
PASS delta over 24 h **+0** (84 -> 82 -> 84) — and the dip is a credit, not a
debit: `T0.15` had been un-runnable for 18 days while its row read PASS, and it
was the builder that found it. Both `T2.09` pilots completed cleanly
(`/data/t2_09_pilot_seed90.json` 18:26, `seed7.json` 18:30); no orphaned
processes, no dead watchers, load 0.06. Before 12:07: see RANK 4.

**5. Compute honesty — no waste, and that is the problem.** `gpu_budget.json`:

| week | Kaggle charged | failed | unspent at expiry |
|---|---|---|---|
| W32 | 21.06 h (+6.38 h unattributable opening balance) | 0.12 h | ~8.8 h |
| W33 | 7.63 h | 0.26 h | **~22.1 h** |
| W34 | 0.70 h | 0.00 h | **~29.3 h, expiring 2026-08-30 00:00 UTC** |

Every charged hour has a ledger row behind it; `T2.19`'s row carries its
`gpu_job_id` (`jack-ladder-1788023542`) and the budget file carries the charge.
**There is no waste of the kind this section was written to find.** The loss is
the opposite failure: hours that were never spent because nothing was
implemented to spend them on. `queue_depth` now reads **3 dispatchable, all
VOID -> 0 fresh dispatches**, with `gpu<20min` NEWLY EMPTY *because T2.19
passed*. Inventory, not uptime, is the binding constraint — and it is the one
thing an hour of builder time reliably fixes.

**6. Stuck decisions.** Nothing is MEANS-ESCALATED by the tool's reckoning and
nothing is UNDECLARED (0/10). Nothing has been quietly acted on: `D3`'s entry
explicitly records that 146 pushes happened under no stated limit and proposes to
*fence* the observed practice rather than pretend it did not occur — that is the
honest handling. The live problems are RANK 2 (the invariant is unenforced and
`D8`'s default is unsafe as written) and RANK 3 (`SY.01` does not exist).

**7. Bakeoff hygiene — no findings.** `DECISIONS_RESOLVED.md` reads clean.
`PS.01/J` is the model case: three arms below the 3.0-sigma learning gate ->
**VOID**, explicitly *"an arm that has not demonstrably learned cannot arbitrate
the decision"*, then `PS.01/J2` re-run to a real winner (`impact_speed`). No
VOID is treated as a verdict. `D2` was settled by ledger replay rather than
bakeoff, correctly — it is a property question about `Status.VOID` semantics,
not a competition, and it carries a re-open trigger.

**8. The honest summary — are we closer to a curious humanoid, or just to a
longer list of green ticks?**

Today, genuinely closer, by one rung and one that matters: `T2.19` shows the
flow head steering around a bimodal obstacle that the parameter-identical
regression head splits down the middle. That is GOAL.md's *"components must EARN
their parameters"* discharged against a null that was well-trained and lost
anyway — the regression arm had the **best content error of all four legs**. It
is a real finding about Jack's body, not about his scaffolding.

Over the week, no. The ladder moved **83 -> 84** in seven days. The builder was
switched off for 95 of the last 96 hours before noon today by an organ acting on
a meter that measures other people's usage. Three weeks of free GPU quota —
about 60 hours — expired against an empty shelf. And the commitments that are
the actual goal (curiosity, unison, needs, living-and-dying) are each at zero or
one PASS while Tier 0 is 13/13.

The machine is honest. It found two of its own false greens today and published
the arithmetic that made it look bad. What it is not, this week, is *fed* — and
the honest reading is that the binding constraint stopped being rigour some time
ago and is now simply **hours of builder wake-time pointed at unimplemented
specs.**

---

## FOR THE BUILDER

**B1 — TONIGHT, and it expires in ~5 hours.** Both `T2.09` pilot artifacts have
landed (`/data/t2_09_pilot_seed7.json` 18:30, `seed90.json` 18:26) and seed 7 is
the one that fired the trap (`icm dwell 0.8337`, `ratio 2.279`). Freeze the seven
bars from **seed 7** — seed 90's `dwell 0.0000` can only freeze the liveness and
exposure bars — set `_CLAIM_ARM` from the pilot rather than by argmax, set
`est_hours` from measured wall time, flip `_GATES_FROZEN`, and dispatch into
W34 before **2026-08-30 00:00 UTC**. This is the only path by which any of
W34's 29.3 free hours gets spent. **If it cannot be done honestly by 23:00 UTC,
say so in the journal and let the hours expire on the record.** Do not lower a
bar, shorten a seed list, or baseline a cost class to manufacture a dispatch —
a fourth week of expiry is cheaper than one bought green.

**B2 — make `decisions.py` enforce the clause it advertises.** `audit()`
inspects `default` only as a non-empty string; SYSTEM.md:126-133 says it
enforces the already-permitted invariant. Minimum executable version, in
priority order:
  1. **A default whose firing moves any `coverage.report()` row from a live
     claim to CLAIM-DEAD is a violation.** Computable today against the real
     rows; `D8` (parks `BA.02`, the only live claim behind `balance`,
     GOAL.md:41) is the known positive and must be flagged.
  2. Refuse a `decide_by` that falls after a dated expiry named in the same
     entry (filed as 39th-audit B3, still open — and `D13`/`D14` are the live
     case: they fire 29 h after the resource they protect died).
  3. Print the default in full. `main()` truncates at `[:110]`; the eleven live
     defaults are 369–1041 characters, so 70–89% of every constitutional clause
     has never appeared in any report.

**B3 — `decisions.py` has no arena check; `champions.py` does.** Add an
`arena:` field to the `DECIDE:` block and a `NAMED-ARENA-MISSING` violation
resolving it against `BY_ID`, in the same idiom as `champions.py`. Known
positive today: `D13` names `SY.01`, which appears exactly once in the
repository — inside `D13` — and is not in the registry. Registering `SY.01`
(the three-arm pace-gate bakeoff, arm C = pace-gate the auditors) is the repair
rule 3 actually asks for; deleting the reference would make the seat safe, which
is the opposite.

**B4 — state a policy for development FAILs, because the ledger cannot infer
one.** `T0.17`'s `2026-08-29T13:14:23` FAIL at `d84101e+dirty` was a new
sub-property failing before `protocol.py` was repaired — not a refutation — and
nothing in the record distinguishes it from one. 66 dirty-stamped runs exist;
`audit_supersedes_fail` reads 24 unauditable pairs against 5 checked. Pick one
and make it executable:
  (a) a run from a dirty tree records with an explicit development status that
      never enters `supersedes_fail`/`supersedes_void`, or
  (b) `run_spec` gains a scratch lane for pre-commit iterations.
Either way the 24 historical unauditable pairs are not back-fillable — say so
**in the record**, with the count frozen and dated, so the number stops reading
as drift. Do not close `T0.27` by narrowing what it audits.

**B5 — register `W.1`–`W.7`.** Fourth consecutive audit. The World seat is held
**BY VERDICT** with a rematch trigger aimed at seven specs that do not exist,
which makes the strongest marking in `CHAMPIONS.md` also the least contestable.
`PL.00`/`PL.02` (plastic-only) and `LT.03`/`LT.04` (curiosity signal) are the
next two families by consequence.

**B6 — housekeeping.** `CHECKLIST.md` is uncommitted in the working tree
(`83 -> 84`, `T2.19` ticked); `bf44b32` committed the ledger and the budget but
not the checklist, so the file in git is one row stale. And `T0.01` / `T0.10`
are the only two PASS specs with no declared control — if that is deliberate
(both are harness liveness checks), say so in the spec text rather than leaving
it to look like an omission.

---

## FOR THE OWNER

**O1 — eleven pre-registered defaults fire in two days, on 2026-08-31.** `D1`
costs **38 specs**, `D10` 8, `D4` 8. Nothing has been extended: I checked every
version of `DECISIONS_NEEDED.md` in git history — 211 `decide_by` values, all
`2026-08-31` or `2026-09-05`, **no deadline has ever moved.** The clock is
honest. If you want any of these ruled rather than defaulted, this weekend is
when.

**O2 — one of those defaults is unsafe as written, and it is `D8`.** Firing it
parks `BA.02`, which is the **only live falsifiable claim behind `balance`** — a
sense you listed as constitutional at GOAL.md:41. The ladder goes from
`0 CLAIM-DEAD` to `1`. SYSTEM.md promises a default *"may only pick among
already-permitted actions … never widening what is allowed"*; removing the last
claim behind a named sense narrows what the ladder can ever demonstrate, which
is the same harm from the other side. Either rule `D8` explicitly before 08-31,
or amend its default to park `BA.02` **only once a replacement balance claim is
registered**. (`D9` blocks the same spec and has the same shape.)

**O3 — W34's ~29.3 free GPU-hours expire tonight at 00:00 UTC; that is roughly
60.6 hours forfeited across three weeks.** The proximate cause is not waste —
every hour actually charged has a ledger row. It is that the builder was
switched off for 95 consecutive hours by the pace gate, which acts on
`week:all models`, a meter five independent measurements now show is **not
driven by this box** (a full Opus iteration moves it zero points; `week:Fable`
rose 66% -> 100% across 72 hours with no Fable request made here). The builder
woke this afternoon because that meter drifted below its line — by calendar, not
by decision. `D13` and `D14` are the two entries that would settle it and they
fire on 08-31, 29 hours after the resource they protect has died. **The ask is a
date, not a decision:** rule them, or say on the record that the hours may go.
A third possibility, and the one SYSTEM.md rule 3 actually points at, is that
neither is your call at all — both arms are implemented in
`scripts/ladder_loop.sh`, which makes this a bakeoff nobody has written
(`SY.01`, see B3).

**O4 — for information, no action.** `T0.27` stays **RED** and is reported here
as `D16`'s armed default requires. The red is correct: a PASS superseded a FAIL
today at a commit that does not exist. The cause was a half-written test, not a
moved threshold, and the builder disclosed it in the commit message the same
minute. The repair is B4.
