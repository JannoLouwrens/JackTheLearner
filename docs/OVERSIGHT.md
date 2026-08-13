# OVERSIGHT — 12th audit, 2026-08-13 00:45 UTC

## VERDICT: DRIFTING

Every number in the ledger is honest. I re-checked all 75 PASS entries against
their implementations, their commits and their controls and found **nothing
wrong** (§1). I diffed every numeric constant and every deleted assertion in
`registry.py`, `registry_expansion.py` and `experiments/tests/` over seven days
— 105 commits — and found **no loosening in either direction** (§2). GPU
accounting **reconciles to the cent** (§5). The loop ran **19 iterations in 24
hours, all `rc=0`, +8 PASS** (§4). Decisions were made cleanly (§7).

The drift is not in the ledger. It is in the **map of what the ledger has
demonstrated**, and therefore in where the builder is being sent.

On 2026-08-12 the builder built the exact instrument that prevents apparatus
from being counted as capability — `COVERS:` kinds, `claim | fixture | rule |
sensor`, with only `claim` buying `n_pass`. It applied that instrument to **2
of 78 declarations** and stopped. The other 76 inherited the implicit default,
`claim`. At least ten of them are, by the specs' own titles and hypotheses,
apparatus or sensor-legibility. `coverage.py` therefore reports **9 zero-pass
commitments when the honest figure is 15**, and the *standing zero-pass rule* —
the builder's primary work-selection rule — has been steering off that flattered
list for two days.

`docs/LESSONS.md:3025` predicted this failure in writing, on the day the
mechanism shipped:

> *"Expect this to recur one level up: the next version is a `(claim)` spec
> whose test in fact measures its own apparatus, and only reading the test
> catches that."*

It recurred immediately, in the same file, in 76 places.

---

## 0. Is the ladder the RIGHT ladder?

`python -m experiments.coverage` → **exit 0. Zero commitments with no declared
spec.** The 2026-08-10 miss has not recurred; the organ is doing its first job.

Headline: **166 specs · 75 PASS · 1 FAIL · 1 VOID · 62 of 166 unreachable.**
By tier: T0 27/27 · T1 13/13 · **T2 33/59 · T3 0/14 · T4 1/23 · T5 0/25** · T6 1/5.

Tier 3 ("earn your parameters") and Tier 5 ("the claims — the thesis itself")
are at **zero**, six days into a project that is 45% green.

**Reported zero-pass commitments: 9 of 23** — touch/contact, damage/nociception,
shelter/building, tool use, proprioception, sleep, social/other agents,
plasticity, curiosity. **The real figure is 15.** See RANK 1.

---

## RANK 1 — The kind mechanism was built and then not deployed: 76 of 78 `COVERS:` declarations still default to `claim`, and six commitments read "started" on the strength of their own fixtures

**Status: no PASS is corrupted and no gate is wrong. The specs are honest; what
they are being *counted as* is not. Damage is to work selection, and it is
active right now.**

### The mechanism, and how far it was deployed

`coverage.py:120-140`, shipped 2026-08-12 in `60686ac`:

```
# A declaration carries a KIND: `COVERS: curiosity (fixture)`. Absent = claim.
#   claim   — a capability test that could have failed; the ONLY kind n_pass counts
#   fixture — apparatus a claim will need (a trap, a world property)
#   rule    — a gate/admission criterion enforced on candidates
#   sensor  — an instrument measures/emits a channel; nothing acts on it yet
```

Measured across `registry.py` + `registry_expansion.py`:

```
COVERS: declarations total          78
carrying an explicit kind            2   (PG.4 fixture, LC.01 rule)
defaulting to `claim`               76
using `(sensor)`                     0   — the kind was defined and never used
```

The commit that introduced the kinds re-typed the two specs the *previous*
audit had named and no others. Nothing swept the rest.

### The ten declarations that are wrong, in the specs' own words

| commitment | credited passing spec | the spec's own title / hypothesis | honest kind |
|---|---|---|---|
| taste | TA.01 | **"The poison FIXTURE**: sub-lethal first dose, visually identical twin" | fixture |
| smell | SM.01 | "The odour **field obeys its own pre-registered rules**" — field model matches its equations to 1% | fixture |
| sight | PG.6 | "The playground **has eyes**" — a linear probe recovers radius/bearing from rendered frames | sensor |
| sight | PG.9 | "The **eye's view** is not mostly obstacle" — <5% near-geometry, ≥35% floor | fixture |
| hearing | PG.5 | "**Procedural contact audio** with localization labels" — a synthesiser | fixture |
| hearing | PG.7 | "The heard-not-seen **fixture** leaks nothing but the intended bit" | fixture |
| balance | BA.01 | "…from which **a linear probe recovers tilt**" | sensor |
| hunger/thirst | PS.01 | "The **drive layer is a real control problem**, and a statue loses" | fixture |
| language (parent) | T1.13 | "The **grounding pairs are real**" — a data-integrity check on the corpus | fixture |
| voice | VO.01 | "…is rendered by ContactAudio…, **is recoverable by a probe** on a LISTENER's audio input" | sensor |

Two more are arguable and I flag them without asserting them: **PS.02**
(thermal — half world-model fidelity, half sensor legibility) and **DP.00**
(fast/slow — its hypothesis is *"there exist states where an agent with a
PERFECT model and unlimited rollouts beats the best reactive policy"*, which is
a property of the world, not evidence Jack has two modes).

None of these specs lied. Every one measures exactly what it says. The defect
is one word in a `notes` field, ten times.

### What it costs, quantitatively

Conservative re-typing — only the ten above, only where the spec's own title or
hypothesis says fixture/sensor/world-property:

```
zero-pass commitments   reported  9 of 23   ->  honest  15 of 23
commitments losing their ONLY credited pass:
    balance · smell · taste · sight · hunger/thirst · language (parent)
commitments retaining a genuine passing capability claim:  4
    hearing            UB.9   (heard-not-seen fusion)
    one brain / unison UB.9
    memory across lives ME.10 (the double dissociation)
    death & retry      XL.00  (the diary crosses)
```

Including the two arguable ones the honest figure is 17 of 23 zero-pass, and
the surviving list does not change.

A fifth, `generality`, is credited to **T1.02** ("on held-out states, a
structured task generalises and a shuffled one does not"). T1.02 is a genuine
claim, but it is a Tier-1 synthetic-task primitive, and GOAL.md's `generality`
is *"a Jack who masters jungle AND desert has abstracted shelter from lean-to"*
(GEN.06). That is a **scope** mismatch, not a kind mismatch, and I raise it
separately rather than folding it into the count.

### Why this is the RANK 1 finding and not a bookkeeping note

The `n_pass` count is not decorative — it *routes work*. The builder's own
words, this iteration (`1b82da6`, 00:20 UTC):

> *"The standing zero-pass rule outranks fan-out… Across the zero-pass
> commitments, PS.03 is the cheapest runnable declared spec."*

The last three units of work — UB.9, BA.01, PS.03 — were all chosen off this
list. A flattered list does two things at once: it hides six real holes, and it
sends the builder to the cheapest survivor of the ones it can still see. The
selection pressure the 11th audit praised as "visibly operating" is operating on
bad input.

`docs/LESSONS.md:3003-3029` already states the governing rule — *"when a single
count aggregates heterogeneous contributors and something routes or ranks on it,
the count silently means whatever its weakest contributor means"* — and notes it
took three consecutive audits to get two declarations fixed. This is the fourth
audit on the same count.

### The generalisable defect

A fix that adds a field to a declaration format is not shipped when the parser
lands. It is shipped when the **existing declarations are migrated**, because
until then the format's default *is* the answer for everything already written —
silently, and with no malformed-declaration report to betray it. `(sensor)` was
defined, documented, and used zero times: the surest sign the sweep never
happened. Appended to LESSONS.md.

---

## RANK 2 — Nine consecutive hand-offs deferred a 20-minute GPU job while 11.8 Kaggle-hours expire on Sunday, and the only runnable curiosity spec has never been implemented

**Status: pure allocation. Nothing is broken; the wrong thing is being built,
every hour, for a defensible-sounding reason.**

### The expiring resource

```
Kaggle 2026-W32   18.1994 h used of 30.0   ->  11.80 h remain
resets Sunday 2026-08-16                    ->  ~3.5 days to spend it
overruns: []      colab_failed: 0.9914 h (the known lost run, now gated by T0.24)
```

### What it could buy, today, with no decision from anyone

`run next` reports 29 runnable. Nine of them are GPU specs with all dependencies
satisfied and no implementation:

```
T2.03 gpu<20min   Pretrained vision features beat random features   frees 2, co-blocks 9 (UB.1-8, T4.01)
T2.04 gpu<20min   Behaviour cloning on scripted trajectories
T2.06 gpu<20min   Language-action alignment beats chance            frees 3
T3.07 gpu<20min   Ablate mood conditioning                          the FIRST Tier-3 spec
T4.02 gpu<20min   No modality collapse                              frees 1
T2.05 gpu<2h      World model beats constant prediction             frees 1
T2.08 gpu<2h      CURIOSITY DRIVES COVERAGE                         frees 3
T2.11 gpu<2h      Skills are distinguishable
T2.14 gpu<2h      Imitation from real motion capture
                                             declared total  ~9.7 h  of 11.80 h available
```

The whole list fits inside the expiring budget with 2 h spare.

### What actually happened

`T2.03` — the 20-minute one — has been named in the "next iteration" hand-off
in **nine consecutive iteration summaries**, 13:23 on 08-12 through 00:20 on
08-13 (`/data/jack-logs/ladder.log` lines 1399, 1410, 1445, 1460, 1471, 1499,
1511, 1528, 1546). It has never been submitted. `LC.03` (frees 7, no GPU) has
the same nine-deferral record.

Each individual deferral was reasoned and each reason was true. The rule that
produced them is the standing zero-pass rule, which outranks fan-out — and
which, per RANK 1, is reading a flattered list. The two findings are one
mechanism: **the flattered count keeps winning the priority contest against the
work that actually unblocks the ladder.**

### The curiosity number

```
specs declaring COVERS: curiosity        12
ever run                                  1   (PG.4 — a fixture)
runnable today                            1   (T2.08, gpu<2h, never implemented)
blocked behind T2.01's FAIL               7   (CU.1-CU.5, CU.7, T5.08)
```

GOAL.md's north star — *"He explores because he wants to… If there is a ladder
with an apple on top, he must try to climb the ladder, fall, and learn from
falling, purely out of curiosity"* — has **one passing spec, and it is the trap
that catches naive curiosity, not curiosity.** T2.08 costs 2 GPU-hours out of
11.8 that vanish on Sunday.

---

## RANK 3 — D1's cost-of-delay line is wrong, and has been since the day it was written

**Status: an owner-facing decision document is understating its own urgency.
Escalated to DECISIONS_NEEDED.md in this commit.**

`docs/DECISIONS_NEEDED.md:87-89`, written 2026-08-09 in `7addc20`:

> *"COST OF DELAY: T2.01/T2.02 and everything downstream of locomotion stay
> blocked. The memory, playground and **curiosity branches are unaffected**."*

`python -m experiments.run blocked`, today:

```
T2.01 = FAIL  frees 26  (blocks 36)  — Locomotion beats a random policy
   frees: CU.1, CU.2, CU.3, CU.4, CU.5, CU.6, CU.7, ME.7, T2.16, T2.17, T2.18,
          T3.02, T3.04, T3.05, T4.04, T4.05, T5.01-T5.05, T5.07,
          T6.01, T6.02, T6.04, T6.05
```

Every curiosity spec sits behind T2.01 via `CU.1 → T2.16 → T2.01`. The CU family
was registered 2026-08-06 (`c02e590`) — **three days before** the line claiming
it was unaffected was written. This was never true.

Line 366 of the *same file*, an overseer entry from 08-10, states the opposite
correctly (*"CU.1–CU.7 (**every curiosity spec**)"*). The file contradicts
itself, and the wrong version is the one at the top, in the block the owner is
asked to decide.

D1 also still lists **"A. Freeze the trunk … RECOMMENDED"** — unconstitutional
under the PLASTIC-ONLY decree of 2026-08-09. The 4th overseer audit raised that
on 2026-08-10 (`DECISIONS_NEEDED.md:599`). **Three days, no answer, and the
recommendation is still at the top of the file.**

D1 is nine days open and gates 26 specs. The two live options after striking A
are B (split trunks) and D (delete the transformer from the control path), C
being unsupported by the plateau data. **That is a two-arm bakeoff, and SYSTEM.md
law 3 says the system runs it rather than arguing about it** — the blocker is
the owner's one-line reconciliation of the menu, not the experiment.

---

## 1. Integrity of the ledger — CLEAN

Checked all 75 PASS entries mechanically:

```
PASS entries                                             75
with no implementation file in experiments/tests/         0
whose recorded `commit` no longer resolves in git         0
declaring a control but recording no control_metrics      0
declaring no control at all                               2  (T0.01, T0.10)
```

T0.01 ("repo imports clean") and T0.10 ("Kaggle job round-trip") are pure
plumbing smoke tests where the failure mode *is* the assertion; a declared
control would be ceremony. Accepted, not a finding.

`run stale` → **zero stale claims.** 34 entries predate `impl_sha` and cannot be
checked at all — down from 47 at the 8th audit, and shrinking by re-run rather
than by amendment. T0.18 ("every PASS is re-derivable and every control is
read") is PASS and re-derived from a clean tree.

## 2. Thresholds and controls over time — CLEAN, no loosening

105 commits touched `registry.py`, `registry_expansion.py` or
`experiments/tests/` in seven days. I diffed every added and removed
module-level numeric constant, every `seeds=`, `control=`, `gate_mode=` and
`falsified_by=` change, and every removed line containing an assertion or a
conjunct. Direction of every change that moved a bar:

```
TF_SPREAD_MIN 2.5  ->  TF_ABS_SPREAD_MIN 2.5 + TF_FALL_SPREAD_MIN 2.5   TIGHTER (one gate became two)
N_CALIB       60   ->  400   (VO.01 SIR calibration)                     TIGHTER
N_OCC         160  ->  2*N_TRAIN = 600 (VO.01 occlusion pairs)           TIGHTER
T0.21 N_PROPERTIES 7 -> 8 ; T0.22 12 -> 13 -> 14 -> 15 ; SM.01 6 -> 7    TIGHTER
T1.07/T1.08 Budget.CPU -> Budget.GPU                                     CORRECTION, justified in 71f7f03
```

The only deleted `_check` conjuncts in the window are BA.01's, removed in
`ed49b73` when the conjunction was **extracted verbatim** into `rig_health()` so
T0.26 could drive the shipped statistic path. I diffed the extraction line by
line: byte-identical predicate, all three gates intact.

Special mention, in the good direction: BA.01 v3 **took back its own PASS**
(74 → 73) when the restored gate caught seed 2's degenerate world. That is Law 4
working under pressure, and it cost the builder two iterations to honour.

I also ran the scan `LESSONS.md:2866` promised — every implemented spec's
declared budget against whether its module actually submits remote work — and
found **zero live mismatches**. The guard that lesson specced ("a test module
that calls `gpu.submit` must declare a `gpu` budget") was never registered as a
spec, but nothing is currently broken by its absence. Noted, not ranked.

## 3. Drift from the goal

**What the builder did in the last 24 h,** and the GOAL.md sentence each serves:

| work | serves |
|---|---|
| PS.02 (thermal field + probe) | *"too cold kills him"* — yes |
| VO.01 (voice rendered and heard) | *"and VOICE — he must be able to make sound"* — yes |
| UB.9 (heard-not-seen fusion) | *"one brain, all senses in unison"* — yes, and it is the strongest pass on the board |
| BA.01 v3/v4 (vestibular probe) | *"proprioception & balance"* — yes |
| PS.03 (damage design, pre-registered) | *"pain"* — yes |
| T0.12/20/21/22/25/26, T0.08 property 6 | *"protects the honesty of watching what happens"* — yes |

**Nothing is drift in the sense of serving no sentence.** Every unit traces.

The honest concern is the **shape**, not the direction. Of 14 ledger runs in the
last 24 h, **10 were harness or learning-primitive specs** (T0.08/12/16/20/21/
22/25/26, T1.08, T2.00) and **4 were Jack-facing** (PS.02, VO.01, UB.9, BA.01).
I must implicate myself here: **5 of those 10 machine runs were my own
FOR THE BUILDER items** (T0.12 ← item 1, T0.21 ← item 2, T0.22 ← B4, T0.26 ← B2,
T0.08 P6 ← B3). The overseer is the largest single source of machine work in
this window. SYSTEM.md's own guard applies to me as much as to the builder:
*"when the machine is sufficient, PROVE it by throughput."* My builder list this
audit is therefore deliberately short, and points at Jack.

**The converse, which is the harder question.** Which parts of GOAL.md have no
passing spec at all? After RANK 1's re-typing, **19 of 23 constitutional
commitments have zero passing capability claims.** The four that survive are
hearing, one-brain/unison (both UB.9), memory-across-lives (ME.10) and
death-&-retry (XL.00).

And a sharper version of the same question, which no organ in this system
currently asks: **is there a single passing spec in which Jack's behaviour gets
better because he lived through something?** I read all 35 non-harness PASS
entries. The answer is essentially **one** — T2.20, "episodic memory helps the
next episode" — plus ME.10's distillation half. Everything else is a world
property, a sensor-legibility probe, a memory-store engineering claim, or
apparatus. The one spec that asserts embodied improvement from experience in so
many words, **T2.01 "locomotion beats a random policy", is the ladder's only
FAIL.**

That is not an accusation of dishonesty. It is the accurate statement of where
the project stands: the senses are being wired and certified beautifully, and
almost nothing yet learns by living.

Also flagged, without a claim attached: **`death & retry` is credited to XL.00**,
which proves the body dies at the predicted rate, respawns at an unchosen pose,
and the diary survives. GOAL.md's actual commitment is *"Life N+1 must be
measurably better than life N **because of** what life N recorded."* No spec
tests that sentence. XL.00 is the substrate for it, not the claim.

## 4. Is the builder alive and productive? — YES, and this is the healthiest section

```
iterations in the last 24 h        19   (hourly, 06:47 -> 00:07, no gaps)
ended rc=0                         19
PASS delta                         67 -> 75   (+8)
specs registered                   +3   (T0.25, T0.26, +1)
tree state                         clean, HEAD == origin/main, nothing unpushed
dead hours                         0    (the 9.7 h owner pause ended 08-12 06:47)
```

No repeated identical failures, no paused loop, no credit exhaustion (the
`OUT OF CREDITS on fable -> falling back to opus` fallback fired 07:07-11:07 on
08-12 and the loop kept working through it — the failover is real). No
iterations aborting on load; every start logged load ≤ 0.20.

This is the most productive 24 h window the project has had. It is also why
RANK 2 matters: the throughput is real and it is being spent on the wrong queue.

## 5. Compute honesty — CLEAN, reconciles exactly

```
2026-W32 kaggle          18.1994 h of 30.0
  tracked in charged_jobs 11.8145 h  (5.5786 + 0.6561 + 5.5798)
  pre-dating job_id       6.3849 h   (spent 08-09, before charged_jobs existed)
  sum                    18.1994 h   EXACT
2026-W32 colab            0.5513 h work + 0.9914 h colab_failed
overruns                  []         (W31 closed at 37.4554 of 30 — never repeated)
remaining                11.80 h, expires Sunday 2026-08-16
```

What the hours bought: **11.16 h on T2.01 v4+v5, both FAIL** (a FAIL is a
result, not waste), **0.66 h on T1.02's recovered PASS**, **0.55 h on T2.00's
post-GAE-fix PASS**. The 0.99 h `colab_failed` is the delivery loss the 9th
audit found; it is now gated by T0.24 and has not recurred.

I specifically re-checked the reattach over-bill the 10th audit found: the
submissions log shows a 35,330 s charge presented against an already-charged
`job_id`. **It never reached the week total** — the idempotency check rejected
it, and `97f9419` fixed the meter so the presented number is right too. Verified
by replaying `weeks["2026-W32"]["kaggle"]` across every commit that touched the
budget file. No unexplained hour exists.

**Waste is zero. Idleness is 11.80 h and counting** — see RANK 2.

## 6. Stuck decisions

| decision | age | state |
|---|---|---|
| **D1** — does the 57M trunk stay in the control path? | 9 days | OPEN. Cost-of-delay line is **wrong** (RANK 3); RECOMMENDED option A is unconstitutional and that was raised 3 days ago unanswered. Gates 26 specs. |
| **D2** — does a VOID dependency block? | closed | **RESOLVED 2026-08-13 by the builder, by ledger replay** — correctly, with a loser, a re-open trigger and an executable invariant (T0.08 P6). Off the desk. |
| **D4** — LC bakeoff labelled `cpu<2h`, research costs it ~20 core-hours | 3 days | OPEN, and it is the thing standing behind LC.03 (frees 7). |
| Was physics-first retired by argument instead of by bakeoff? | 2 days | OPEN. This one the system could answer itself. |
| Owner's hands / LC verdict survives scale | 3-4 days | OPEN design forks, correctly parked. |
| D3, D5, Kaggle grant, /data 95% | closed | Correctly closed or annotated stale. `/data` is now 21% used, 80 G free — that entry is dead and marked so. |

**Anything decided quietly without being recorded?** No. I checked
`DECISIONS_RESOLVED.md` against the last 7 days of commits: D2 is the only new
resolution and it is fully recorded. No owner decision was acted on unrecorded.

**Anything blocked on the owner the system could have settled itself?** Yes, one:
after striking option A, D1 reduces to B-vs-D, which is a two-arm bakeoff with a
learning gate — exactly SYSTEM.md law 3. What genuinely needs the owner is the
one-line menu reconciliation, not the verdict.

## 7. Bakeoff hygiene — CLEAN

`DECISIONS_RESOLVED.md` holds three entries.

- **PS.01/J → VOID.** Three arms below the 3.0σ learning gate; recorded as VOID
  and explicitly *not* as a verdict. Textbook.
- **PS.01/J2 → WINNER `impact_speed`.** Learning gate cleared, margin recorded.
- **D2 → WINNER BLOCK.** Not a bakeoff — a replay of the ledger's own status
  history. Legitimate: it is a measurement over recorded data (retraction
  exposure 9 vs 0; benefit 3 unimplemented specs), and it ships with a loser and
  a re-open trigger. Correctly labelled as decided by replay, not by argument.

No winner chosen inside a noise margin. No VOID treated as a verdict. Nothing
to report.

---

## 8. The honest summary — are we closer to a curious humanoid, or to a longer list of green ticks?

**Both, and the ratio moved the wrong way this week.**

Closer to a creature, genuinely: **UB.9** is the real thing. A task that is
impossible without fusion, 0.993 fused against unimodal and ensemble nulls at
chance, three seeds — that is the project's namesake commitment earning its
first honest pass after 21 specs. **BA.01** survived taking back its own PASS
and came back with every gate byte-identical. **T0.26** turned a twice-burned
scar into an executable guard. The machine caught itself twice in 24 hours and
told the truth both times. That is not nothing; it is the thing SYSTEM.md says
matters most.

Closer to a longer list of ticks: of 75 PASS, **40 are harness and learning
primitives**, 21 are world-and-apparatus fixtures, 12 are memory-store
engineering, and — after honest re-typing — **four are capability claims about
Jack.** Tier 3 is 0/14. Tier 5, "the claims — the thesis itself", is 0/25. The
one spec that asserts Jack gets better at something by doing it is the ladder's
only FAIL, and it has been FAIL or VOID for six days while 11.8 GPU-hours drain
toward Sunday unspent.

The uncomfortable version, stated plainly: **we have built an extremely honest
instrument and pointed it mostly at the instrument.** Every sense now has a
certified channel. Nothing yet lives in the world those channels open onto. The
ladder-and-apple standard — *climbing on attempt 40 after falling on 1-39,
without anyone telling him to* — has not been attempted once, and the twelve
specs that would attempt it have one run between them, on a trap.

The fix is not more rigour. Rigour is not the constraint here; §1, §2, §5 and §7
are all clean, and that took real work. The fix is to **stop letting a flattered
count choose the work**, and to spend the expiring GPU budget on the specs that
open the blocked half of the ladder. Both are one iteration each.

---

## FOR THE BUILDER

Ranked. **B1 before anything else** — it changes what B2 and every later
iteration will choose.

**B1 — Re-type all 78 `COVERS:` declarations, in one commit, and make the
default explicit.** *(RANK 1)*

1. Add the honest kind to the ten declarations in the RANK 1 table:
   `TA.01 (fixture)`, `SM.01 (fixture)`, `PG.6 (sensor)`, `PG.9 (fixture)`,
   `PG.5 (fixture)`, `PG.7 (fixture)`, `BA.01 (sensor)`, `PS.01 (fixture)`,
   `T1.13 (fixture)`, `VO.01 (sensor)`.
2. Add an explicit kind to **every** remaining declaration — including the ones
   that really are `(claim)`. An implicit default on a field that routes work is
   the defect; writing `(claim)` out loud is the fix.
3. Make your own call, in the commit message with reasons, on the two I flagged
   but did not assert: **PS.02** and **DP.00**.
4. **Change no spec's content, hypothesis, gate or threshold.** This is a
   labelling commit. `run stale` must read zero afterwards and no ledger row may
   move.
5. Then extend `coverage.py` so an **undeclared kind is REPORTED like a
   malformed declaration** rather than silently defaulting — the default is what
   let 76 declarations through — and gate that with a T0.21 property, both
   directions (a kindless declaration is reported; an explicit `(claim)` is
   credited), with the defaulting rule kept executable as the control that fails
   it.
6. Expect the zero-pass count to go **9 → 15 (or 17)**. That is the instrument
   getting its eyes back, not a regression. Re-run coverage and let the standing
   rule re-point before you choose your next unit.

**B2 — Spend the expiring Kaggle hours. 11.80 h, gone Sunday.** *(RANK 2)*

Order, and the reason for it:

1. **T2.08 — "Curiosity drives coverage", gpu<2h.** The only runnable spec for
   GOAL.md's north star. Twelve curiosity specs, one run between them, on a
   fixture. Needs implementing. Do this one first even though it is not the
   cheapest: every other curiosity spec is behind T2.01, so T2.08 is the only
   shot at this commitment that exists without an owner decision.
2. **T2.03 — gpu<20min.** Nine consecutive hand-offs, never submitted. Frees 2
   directly and is a co-blocker on UB.1-UB.8 and T4.01.
3. **T3.07 or T4.02 — gpu<20min each.** T3.07 would be the **first Tier-3 pass
   in the project's history** (0/14); T4.02 guards modality collapse, which the
   unison commitment needs before UB.10.

If the hours are going to expire anyway, an implemented spec that FAILs is worth
more than an unspent hour — a FAIL is a result and this ladder has proven it
records them honestly.

**B3 — Say plainly, once, what `death & retry` still owes.** XL.00 proves the
substrate. GOAL.md's sentence is *"Life N+1 must be measurably better than life
N because of what life N recorded"*, and no spec tests it. Either register that
spec or re-type XL.00's declaration so the commitment reads honestly. One line
either way; do not build it this iteration.

**Not asked for:** any new T0 spec. The harness is at 27/27 and five of the last
ten machine runs came from my desk. If you find yourself building the machine
again this iteration, check that it is not me who sent you there.

---

## FOR THE OWNER

**1. D1 is nine days old, gates 26 specs, and what it tells you about the cost
of waiting is wrong.** It says *"the curiosity branch is unaffected."* Every one
of the seven CU specs is blocked behind T2.01, and has been since three days
before that sentence was written. The full correction, with the dependency
trace, is appended to `DECISIONS_NEEDED.md` in this commit.

**2. D1 still recommends an option your own decree forbids.** Option A —
"freeze the trunk, small head does control" — is marked `RECOMMENDED`, and the
PLASTIC-ONLY decree of 2026-08-09 rules out frozen components inside Jack. This
was raised three days ago (`DECISIONS_NEEDED.md:599`) and is unanswered. **The
ask is one line, not a decision:** either strike A, or narrow the decree to say
a frozen *control* trunk is a different question from a frozen *sensory* tower.
Until then the two documents instruct different work, and agents are designing
toward a recommendation the constitution forbids.

If A is struck, what remains is B-vs-D, which the system can settle itself by
bakeoff without further input from you.

**3. The honest coverage number, so you are not reading a flattered one.**
`coverage.py` currently reports 9 of your 23 constitutional commitments with
nothing passing. After correcting ten mislabelled declarations, the figure is
**15 of 23** — and the four commitments with a genuine passing capability claim
are hearing, one-brain/unison, memory-across-lives and death-&-retry.

Balance, smell, taste, sight, hunger/thirst and language each have exactly one
passing spec, and in every case that spec certifies the *apparatus* — the odour
field obeys its equations, the camera resolves what the test needs, the poison
plants are visually identical. **That work is real and it had to happen first.**
It is simply not the same as Jack smelling, seeing or tasting anything. The
builder is fixing the labels this iteration; no number in the ledger changes.

**4. Nothing is broken and nothing is dishonest.** 75 passes, every one checked
against its implementation, its commit and its control. Seven days of threshold
diffs with no loosening in either direction. GPU accounting reconciling to the
cent. A loop that took back its own green tick twice in one day rather than keep
it. The problem this week is not truthfulness — it is that the most rigorous
instrument in the project got pointed at itself, and eleven-point-eight
GPU-hours expire on Sunday.

---

*Audited at `1b82da6`, tree clean, `HEAD == origin/main`. Ledger untouched. No
spec, test or model file modified by this audit.*
