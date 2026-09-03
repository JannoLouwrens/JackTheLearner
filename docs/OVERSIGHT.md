# OVERSIGHT — 66th audit, 2026-09-03 12:40 UTC (HEAD `84f3bcc`, tree clean, 0 unpushed, no runner alive)

## VERDICT: DRIFTING — one paragraph of GOAL.md has declared itself falsifiable since 2026-08-09 and has never had a spec, and `coverage.py` cannot see it because the bucket it falls into is green

The mechanical state of this system is good and I want that on the record before the
finding: **96 of 96 PASS rows have a reachable commit, a declared control, and an
implementation** (§1, no findings); **no threshold moved in the loosening direction
in seven days** and the one downward move I found was legitimate and I say so with
its arithmetic (§2); the builder ran **30 of 30 iterations `rc=0`** in 24 h and moved
demonstrated 94 → 96 (§4). Yesterday's `ON TRACK` was the right call and nothing has
decayed since.

The verdict is `DRIFTING` for a §3 reason, not a §1 one. GOAL.md names **three**
expansions past the jungle. Two have ladders. The third has nothing, and the
instrument built to catch exactly this cannot report it — not because the tool is
broken, but because its commitment names are **topics** and the thing missing is a
**claim inside a topic that already reads 3 pass**.

Ranked by damage to the trustworthiness of the ledger:

| # | finding | damage |
|---|---|---|
| 1 | **THE TOLD WORLD** — GOAL.md:206-212, in GOAL.md since `1859c8f` (2026-08-09, **25 days**), says of itself *"FALSIFIABLE, and it must be tested rather than assumed"*, and **0 of 232 specs make the claim**. Invisible to `coverage` because it lands in `language (parent)`, which reads 3 passing claims | the ladder is missing a rung the constitution names, and no instrument can say so — the 2026-08-10 miss-class verbatim |
| 2 | `T0.32` PASSed today on *"the real-time factor … gates long runs"*. `require_feasible` and `gate_long_run` have **zero callers outside the spec's own test**. The only thing binding a future long run to the gate is a prose sentence in `LF.01`'s notes — and `LF.01` is unimplemented | a green gate that refuses nothing in production; its own `falsified_by` has no watcher |
| 3 | today's registration burst added 7 specs (225 → 232) and **5 of the 7 declare no `COVERS:`**. Two of those five (`SO.01`, `SO.04`) *cannot* declare one: the commitment they serve has no name in `COMMITMENTS` | the ladder grew 3.1% and the commitment table gained one fixture and one claim |
| 4 | W35 Kaggle: **18.93 h of 30 spent, ~11.07 h expire Sunday 2026-09-06**, and every GPU cost class is `NOT FILLABLE`. Fourth consecutive week; **~61 free GPU-hours already lost** across W33–W35 | honest and routed — but the number only goes up, and `D15` fires 09-05 with this as evidence |

---

## THE FOUR MANDATORY INSTRUMENTS (read live at 12:38, post-`84f3bcc`)

| instrument | rc | reading |
|---|---|---|
| `coverage` | **2** | **0 commitments with NO declared spec.** Red is standing and routed: **4 CLAIM-DEAD** (smell, balance, shelter/building, thermal — row DUE 09-11) plus **4 NEW unrunnable GOAL.md citations** (`GEN.02/03/06/09`, all `welded<-LC.07` — row `goal-cites-four-specs-that-resolve-to-corpses` DUE 09-10, routed 1 d ago). `unreachable` **91 of 232, baseline 91** — at floor. The 90→91 growth in `299e735` is **legal**: the tool permits growth with a named justification in the growing commit, the growth log carries it (`SO.04` blocked behind the fresh runnable `SO.01`), and the fraction *fell* 40.0% → 39.2%. |
| `decisions --check` | 0 | **0 MEANS-ESCALATED, 0 UNDECLARED, 0 OVERDUE.** Ratchet 0/10 — the list is fully armed, so **there is nothing for me to arm this audit** and I am not inventing one. Live: `D15`/`D16` due 09-05, `D17` 09-07, `D18` 09-09, `D19` 09-14 (costs 3 specs). |
| `champions --check` | 0 | 27 seats, 11 violations, **every ratchet at baseline, none grown**: `arena_missing` 0/0, unfalsifiable 3/3, uncontestable 3+1/4, unverified verdicts 2/2, trigger debt 3/3. |
| `run review-queue` | 0 | **0 violations.** 28 OPEN / 2 HELD / 2 ACTED of 32; oldest live 10 d; consumer ran today. The 09-06 pile is **7 live rows against a measured capacity of 1/cycle** — unchanged since the 65th audit flagged it, and now printed as `AMBER` by the reader that audit's B6 shipped. Five more dates are amber behind it. |

`scripts/lib_liveness.sh` schedule half: builder hourly at `:07` (30/30 present),
overseer `:37 */6`, Review 06:37 daily (ran 06:45), field watch next 09-07. No
leftover processes: the one `run_spec T0.32` line in `declared_pids` is a stale
stamp from 12:16 whose pid is gone; the live `claude` is this audit's own session.

---

## FINDING 1 (RANK 1) — GOAL.md's third expansion has no ladder, and the tool that exists to catch that reports green

### What GOAL.md says, and how plainly it says it

`GOAL.md:185-212` — *"The jungle is the foundation, not the destination"* — names
**three** expansions past survival, *"different in kind"*:

1. **MORE WORLDS, and transfer between them** → `GEN.06` registered (*"Transfer across worlds: mastery is structure, not fit"*)
2. **OTHER MINDS** → `GEN.02`, `GEN.03`, `GEN.09` registered
3. **THE TOLD WORLD** → **nothing**

And the third is not a passing aspiration. It is the only one of the three that
GOAL.md stops to give a test design, a null, and a consequence:

> **FALSIFIABLE, and it must be tested rather than assumed:** told-knowledge should
> integrate BETTER when it anchors to something he has lived. Tell him a fact
> grounded in his experience ("volcanoes are hot rock that flows") against one that
> is not ("bonds have coupon rates") — if grounding is real, the first is usable and
> the second is parroting, and he should be able to say which is which. **If both
> integrate identically, he is reciting, and the jungle bought nothing.**

That is a hypothesis, an operationalisation, a control arm, and a stated cost of
failure — written by the owner, in the constitution, and it has been there since
commit `1859c8f`, **2026-08-09**. Twenty-five days. Sixty-five audits.

### The search, so this can be checked rather than believed

Against all 232 registered specs, over `title + hypothesis + notes + falsified_by +
null_baseline + control`:

```
told|second-hand|rome|volcano|coupon|hearsay|anchored to lived|grounded fact
   → T0.19 (bakeoff screen), NE.07 (social need), HR.4 (who told him, from the voice)
parrot|recit*|library|culture|book|encyclop|ungrounded
   → ME.6 (skill library), NE.08, W.2, XL.01, GEN.09 (Jack→Jack culture), PS.04
```

Not one of these makes the claim. The nearest neighbours are worth naming because
they are what makes the hole feel filled:

- **`LG.00` (PASS)** tests the *inverse* asymmetry — strip the diary and the learned
  core, his answers about **his own life** must collapse while general knowledge
  survives. That is "he is not a costume". It says nothing about whether **told**
  facts land better when they anchor to lived ones.
- **`GEN.09`** is *"generation 3 knows what generation 1 never knew"* — Jack-to-Jack
  cultural transmission. Other minds, expansion 2. Not the borrowed library.
- **`T2.06` (PASS)**, **`T2.07` (FAIL)**, **`T2.15` (FAIL)** are command→action
  grounding. Words routing to behaviour, not facts integrating into knowledge.

### Why every instrument in this repo reports green on it

`coverage.py:141` — `"language (parent)": (r"\b(language|word|words|grounding|lexic\w*)\b", …)`.
The bucket holds 9 specs and **3 passing claims** (`T2.06`, `LG.00`, `LG.02`), so it
is one of the healthiest lines in the table. A told-world spec would land in that
same bucket. **Its absence is therefore not merely uncounted — it is actively
masked by its neighbours' passes.**

This is the failure mode `coverage.py`'s own comment block at lines 164-170 was
written about, one level up. That block records the 8th–10th audits catching
`PG.4` and `LC.01` — *apparatus* crediting *claims* — and the repair was the `KINDS`
mechanism, which distinguishes claim from fixture. It cannot distinguish **claim A**
from **claim B inside the same topic**. `COMMITMENTS` names topics; the constitution
makes claims; the join is many-to-one and lossy in exactly the direction that hides
work.

So every stated property of the miss-class holds: no id, no `run blocked` row, no
gate, no ratchet, nothing red — and the one tool built for it is green.

### Why this is a `DRIFTING` and not a shrug

The staging argument — *"the told world comes after the jungle, so of course there is
no spec"* — is real and I considered it. It does not survive contact with the file.
`GEN.06` (more worlds) and `GEN.02/03/09` (other minds) are the same distance out and
**are registered**, deliberately, as unreachable-but-named. The project's own practice
is to register the falsifier beside the capability and pay the unreachable ratchet for
it — `299e735` did precisely that today for `SO.04` and raised `UNREACHABLE_BASELINE`
90→91 with the reason written down. Expansion 3 was not deferred by that rule. It was
never entered.

And the stakes are the ones GOAL.md states, not ones I am adding: *"the jungle bought
nothing"* is the failure branch. The entire survival programme — W0, W1, the needs, the
mortality — is justified in GOAL.md by the claim that lived primitives make told
knowledge mean something. **That justification is currently unfalsifiable.** A
premise that expensive should be on the ladder like any other.

---

## FINDING 2 (RANK 2) — the gate that PASSed today gates nothing, and its own falsifier has no watcher

`T0.32` was bought PASS at 12:16 (`5848b5b`, attempt 1, clean tree). The science in it
is real and I checked it rather than assuming: the projection was made **before** the
verification run and compared against the achieved duration (error 0.0008 against a
0.25 bar), the control is genuinely wired (`run_spec(..., control_fn=_control)`) and
genuinely read (`_check` asserts `c["slow_admitted"] is False` **and**
`c["slowdown"] >= 5.0`, so a control that failed to be slow VOIDs the claim rather
than passing it), and the single-source claim is not just an import string —
`experiments/run.py:1797` actually calls `spec_child_timeout_seconds(_spec)`. Good work.

What the certificate does not carry:

```
$ grep -rn 'require_feasible' --include=*.py .
experiments/rtf.py:20      (docstring)
experiments/rtf.py:72      (docstring)
experiments/rtf.py:139     def require_feasible(...)
experiments/tests/t0_32_rtf_gate.py:51   (docstring)
```

**Zero callers.** Same for `gate_long_run` outside `t0_32_rtf_gate.py`. The spec's
`hypothesis` is *"for any declared control path, the harness … REFUSES a run whose
projected duration exceeds the spec's timeout"*, and its `falsified_by` is *"a long
run launching with a projected duration past its own timeout"*. Today, every long run
in this repo launches without consulting the gate, because nothing consults it.

To the builder's credit this is stated, not hidden — `rtf.py:18-21` and the test
docstring at line 50 both say the module cannot prove future callers call it. But
stating a gap is not the same as instrumenting it, and the only thing that closes it
is `LF.01`'s registry notes: *"The real-time factor is therefore a GATE, not a note."*
That sentence is prose. `LF.01` is registered, unimplemented, and there is no edge, no
conjunct and no ratchet anywhere that will notice if it is implemented without the call.

**This is yesterday's rank-1 finding wearing different clothes.** The 65th audit's
lesson — now the last entry in `docs/LESSONS.md` — is *"when a spec's text calls
another spec a prerequisite … that word is a `depends_on` edge and must be registered
as one in the commit that writes the word"*, because **no instrument reads prose**. The
same commit-day, the same repo, a binding written in a `notes` field. The lesson
generalises past `depends_on` and has not yet been applied past it.

The repair is small and it is available now, while there is exactly one long-run spec
to bind: a conjunct in `T0.32` asserting that every implemented spec at
`budget >= cpu<2h` calls `require_feasible`. It is **vacuously true today** (`LF.01` is
unimplemented) and it bites the hour `LF.01` is written — which the builder's own
handoff note says is a credible next unit. Written after `LF.01` exists, it is a
retrofit somebody has to remember; written now, it is free.

---

## FINDING 3 (RANK 3) — the ladder grew 7 specs and the commitment table gained one claim

`299e735` (11:07) registered `LF.01`, `LF.02`, `SO.01`, `SO.02`, `SO.04`, `T0.32`,
`T0.33` from the `DIRECTION_AUDIT.md` queue row. The registration work itself was
careful and I want to say so: the mandatory cross-check **refused two** stubs as
duplicate claims (`LF.03` ← `NE.08`/`XL.01`, `LF.05` ← `T5.08`) and **held eight**
behind Sunday's W1 design rather than pre-empting the Review. Every one of the seven
carries a real, non-decorative control. That is a good hour's work.

But `COVERS:` declarations:

```
LF.01  → death & retry (fixture)
SO.02  → voice (claim)
LF.02  → NO COVERS
SO.01  → NO COVERS
SO.04  → NO COVERS
T0.32  → NO COVERS
T0.33  → NO COVERS
```

`T0.32`/`T0.33` are harness specs and legitimately cover no constitutional commitment
— no complaint. `LF.02` ("a life can be saved and resumed — world, needs, diary,
working memory") plausibly owes one to `memory across lives` or `death & retry` and
should either declare it or say in notes why not.

`SO.01` and `SO.04` are the interesting pair, and they connect to Finding 1.
`SO.01` = *"Jack can be watched: a third-person stream exists"*; `SO.04` = *"Being
watched does not change him"*, whose control (*"deliberately draw one RNG value in the
render path: the detector MUST catch it"*) is a genuinely good observer-effect test.
They serve GOAL.md's *"I want to watch him figure out the world himself"* and
*"allow users to talk to him while he is there doing stuff"*. **`COMMITMENTS` has no
line they can name** — `social/other agents` matches `social|companion|two jacks|second
jack` and does not reach a third-person stream.

The builder saw this and named it on the queue row for the Review (*"spectating has no
line in coverage.py's hand-maintained COMMITMENTS"*), which is the right instinct. I am
recording it here so it has a second owner, and because it is the **second** instance
today of the same shape as Finding 1: a hand-maintained list of topics is drifting
behind a constitution that keeps making claims.

Net: registered 225 → 232 (+3.1%), demonstrated 95 → 96 (+1.1%). Demonstrated share
fell **43.3% (94/217, 24 h ago) → 41.4% (96/232)**. That is not a criticism of the
registrations — naming unbuilt rungs truthfully is the practice — but it is the number
§8 has to answer to.

---

## §1 INTEGRITY OF THE LEDGER — no findings, and the checks that produced that

Every `PASS` row in `experiments/ledger.json`, mechanically:

| check | result |
|---|---|
| PASS rows | **96** |
| `commit` still reachable via `git cat-file -e` | **96 / 96** |
| spec declares a `control` | **96 / 96** |
| declared control, empty `control_metrics` | **2** — `T0.01`, `T0.10`, both stamped *"NONE, BY DECISION (52nd audit B5)"* in the registry, both adjudicated |
| implementation present under `experiments/tests/` | **96 / 96** |
| impl passes a `control_fn` to `run_spec` | **94 / 96**, the two exceptions being the same adjudicated pair |
| `+dirty` stamps on a PASS | **0** |

A PASS whose control was never run is the thing this section exists to find, and there
is not one. I checked the freshest and most load-bearing certificate by hand
(`T0.32`, bought 24 minutes before this audit opened) rather than trusting the
aggregate — see Finding 2 for what that read found and did not find.

## §2 THRESHOLDS AND CONTROLS OVER TIME — no findings

Diffed `experiments/registry.py`, `experiments/registry_expansion.py` and
`experiments/tests/` over 7 days (330 commits), extracting every `NAME = value`
constant that appears on both a `-` and a `+` line with a changed value. **Eleven**
constants moved. Ten moved in the strengthening direction or are neutral:

```
ba_03  N_EVAL          48 → 120     more lives
t3_06  LIVES_PER_ARM   16 → 48      more lives
t3_09  N_LIVES         16 → 32      more lives (extension, not a redraw — prefix-preserving RNG)
t2_19  STEPS          300 → 500     more training
w0diag N_DECISIONS   3200 → 4800    more exposure
vo_02  COORD_MIN     0.55 → 0.70    BAR RAISED
vo_02  COORD_MARGIN  0.20 → 0.35    BAR RAISED
lg_10  TEMP          0.25 → 1.0     more sampler entropy: makes match/unanimity/swap/null all strictly harder
t0_21  N_PROPERTIES    11 → 12      more properties asserted
t0_31  N_PROPERTIES 11→12→13        more properties asserted
```

**One moved downward and I read it in full rather than pattern-matching the direction:**
`t2_09_noisy_tv_control.py:DECAY_MIN` 1.5 → 1.25, commit `44f24c41`, 2026-08-29 19:16.
It matters because `T2.09`'s recorded PASS has `claim_static_decay = 1.424` — **the run
would have FAILED at the old value.** A PASS bought by a bar that moved is the single
worst shape this section looks for, so:

- The move happened in the **gate-freezing** commit. `T2.09` had never run and `run()`
  refused until that commit — this is a `PILOT` placeholder frozen for the first time,
  which SYSTEM.md's protocol permits, not a registered bar weakened.
- Ordering is right: freeze at **19:16**, registered run at **20:11**. 55 minutes, and
  the run drew seeds 0–6 while the pilots were seeds 7 and 90 — the freeze could not
  have seen the numbers it would be judged on.
- The value is **principled, not shaved**: a constant or dead signal has decay
  identically 1.0, so any bar above 1.0 excludes it; 1.25 sits midway between that fixed
  point and the weaker pilot reading (1.472). Shaving to the observed minimum would have
  given ~1.45.
- It was declared in the commit message and the docstring under the heading
  `ONE BAR MOVED, and downward`, with both pilot readings printed — and the same commit
  **strengthened** seeds 3 → 7 and added the registry `control` field that was missing.
- The same commit pre-registered, in the open, that `not_fed` was the gate likely to
  decide the run at 6% headroom, *"so that a FAIL cannot later be narrated as a surprise"*.

That is what an honest bar move looks like. No finding.

Also checked and clean: no `_check` gained an `or`, no seed count was reduced (`T2.09`
3→7 and `T3.09` 16→32 are the only seed/lives changes and both increase), no control
was deleted or weakened. The one control-adjacent change in the window was `T3.09`'s
`19461c4`, which moved the `shuf_gain` control-vacuity lane **above** the claim branch —
a strengthening, and it was the 61st audit's own B1.

## §3 DRIFT FROM THE GOAL

**What the builder did in the last 24 h, and the sentence each serves.** Thirty
iterations, and no drift found — every unit traces:

| work | GOAL.md sentence |
|---|---|
| `HR.5`/`HR.7` registered, run, harvested (`HR.7` PASS) | *"hearing … every sense a human has"* |
| 65th-audit B1–B6 executed (`HR.5→HR.6` edge, ratchet reader, amber-pile, `proc_declare`) | *"protects the honesty of watching what happens when the three meet"* |
| `fba9ecf` — decision `blocks:` becomes an edge `coverage` reads | same |
| `d23f319` — `D1_CONTROL_ARCHITECTURE` queue row processed **by refusal** | same; and it is the best judgement call of the day |
| `299e735` — 7 stubs registered, 2 refused, 8 held | *"he lives, he dies, he remembers"* (`LF.*`), *"I want to watch him figure out the world himself"* (`SO.*`) |
| `T0.32` implemented + PASS | *"the final test in that sequence is the real world"* — it prices the runs that get us there |
| `PS.01/02/03`, `BA.01`, `LG.00`, `LG.02`, `T0.13` clean-tree re-buys | ledger honesty |

**The converse, which is the harder question.** Of the 23 constitutional commitments
`coverage` tracks, **12 have zero passing claims**: smell, balance, shelter/building,
thermal, touch/contact, tool use, proprioception, death & retry, plasticity, sleep,
hunger/thirst, fast/slow. Four of those twelve are **CLAIM-DEAD** — every claim spec
parked or foreclosed — and each park was individually legal and evidence-backed;
`coverage`'s own `PARK-ON-AN-UNREACHABLE-RELEASE` line names three whose stated revival
path cannot be walked today (`BA.02→LT.08`, `SH.01→SH.02`, `SM.02→SM.03`).

The two the brief warns are most likely quietly neglected:

- **curiosity** — 12 specs, **2 passing claims**. This is the north star (*"purely out
  of curiosity … climb the ladder, fall, and learn from falling"*) and it is not
  neglected in attention, but `T3.06` (ablate curiosity) is VOID-FORECLOSED and
  `T2.11` is PARKED after two pilots where the permuted control beat the claim arm.
- **one brain / unison** — **25 specs, 1 passing claim**, and that one is `LC.01`
  credited as a *rule*, not a claim. The largest family on the ladder has demonstrated
  the least. `UB.10` (the fusion bakeoff) is VOID with a routed redesign; `UB.14` is
  VOID-FORECLOSED on the venue.

Neither is drift — both are honest reds with routed repairs. But they are where the
constitution's weight sits and where the ladder is thinnest, and §8 has to say so.

## §4 IS THE BUILDER ALIVE AND PRODUCTIVE — yes, unambiguously

- **30 iteration starts** in the 24 h to 12:07, hourly at `:07`, **zero gaps**.
- **30 of 30 ended `rc=0`.** No repeated identical failure, no abort on load (max load
  seen at start 2.64, during the 09-02 17:07 sweep), no pause, no credit exhaustion —
  `week:all models` read 32/33/34/35% across the last four slots, far under the 90%
  stop, and every one of those slots **printed both meters and named the one it acted
  on**, which is the discipline `84f3bcc` was committed to enforce.
- Demonstrated **94 → 96**. The intermediate dip to 90 on 09-02 20:07 is the accidental
  19:08 VOID sweep the 64th audit found; all four certificates were re-bought from
  clean trees and the recoveries are on the ledger with `supersedes_void`.
- Three iterations in the window ended in a **refusal** rather than a unit
  (`d23f319`'s stale queue row, and two slots with a genuinely empty board). Each wrote
  the refusal and its evidence to the record. That is the correct behaviour and it
  costs the PASS delta — worth naming so the low delta is not misread as idling.

## §5 COMPUTE HONESTY

| week | kaggle hours charged | of 30 | jobs | failed |
|---|---|---|---|---|
| 2026-W32 | 16.61 | 55% | 17 | 4 |
| 2026-W33 | 7.89 | 26% | 22 | 4 |
| 2026-W34 | 1.62 | 5% | 4 | 0 |
| **2026-W35 (current)** | **18.93** | **63%** | 12 | 0 |

**~11.07 h remain and expire Sunday 2026-09-06.** That will be the **fourth
consecutive week** of expiring free quota; W33+W34 alone lost ~50 h and the running
total across W33–W35 will be **~61 h**, which is the figure `coverage.py` already
prints as the cost of leaving a class empty.

Hours-to-ledger: W35's 18.93 h is dominated by **`D1.0`'s 16.17 GPU-h, which bought a
VOID** — found by the 60th audit, now routed with an owner and a clock
(`d10-successor-rerun-under-adopted-gate`, DUE 09-08). That is spend with no PASS to
show, but it is *not* waste in the sense this section hunts: the run completed, the row
is honest, the redesign is dated.

The remaining ~11 h has **no honest buyer** and I verified that rather than accepting
it. `coverage` marks every GPU class `NOT FILLABLE`: `gpu<20min` and `gpu<8h` are
pilot-blocked on measured evidence (`DP.04`, `SM.03`, `LC.07`), `gpu<2h` holds only
`UB.10`, a VOID whose repair is a redesign. Dispatching into that would be spending
quota to manufacture a row. The refusal is correct; the recurring loss is real; `D15`
fires on 09-05 with option (e) explicitly struck *because* this is a measured cost
rather than a hypothetical one.

## §6 STUCK DECISIONS — nothing stuck, and nothing for me to arm

`decisions --check` prints **0 MEANS-ESCALATED, 0 UNDECLARED, 0 OVERDUE**, ratchet
0/10. All five open decisions are armed with a default and a date. I could find no
decision the system could have settled itself with a bakeoff, and no owner decision
quietly acted on without being recorded (I read `DECISIONS_RESOLVED.md`'s `D13`/`D14`
entries against `overseer.sh` and `ladder_loop.sh:271` — both cite the decision by name
in code, and `D13`'s implementation note that it reads `experiments.decisions` rather
than grepping `decide_by:` is the kind of detail that shows the entry was written by
someone who tested it).

**Evidence update ahead of `D16` firing in 2 days** — the 62nd audit recorded the
violation count progressing 1 → 2 → 3 at *"~1 per 1.5 days"*, and `D16`'s default was
armed on a two-violation premise. Live now, from `protocol.audit_supersedes_fail`
against the real ledger at 12:39:

```
3 violations, 9 checked pairs, 23 unauditable
  LG.00  VOID  8faff43+dirty  2026-08-30T18:47:59
  T0.17  FAIL  d84101e+dirty  2026-08-29T13:14:23
  T0.29  FAIL  661a48f+dirty  2026-09-02T09:18:06
```

**The progression has flattened.** No fourth violation in the 27 hours since
`T0.29`'s, across 30 iterations and four certificates bought. That is evidence *for*
option (b) — the warning plus the visible red — and I record it because the 62nd
audit's rate figure was the strongest argument against it and it has not held up.
One number moved the other way and should be watched: **unauditable pairs rose 19 → 23
in twelve hours**, so the guard now examines 9 of 32 pairs (28%). It is not blind, but
the blind region grows faster than the examined one.

## §7 BAKEOFF HYGIENE — no findings

Read `DECISIONS_RESOLVED.md`. No decision resolved without a learning gate; no VOID
treated as a verdict; no winner chosen inside a noise margin. Every recent entry
(`D13`, `D14`, `D17`-class firings) records its **losers by name with the reason**, its
**reversal path**, and its **evidence with file:line** — and `D14`'s entry does the
unusual and honest thing of noting the code was already live before the firing, so the
default *"changes no behaviour"* rather than claiming credit for the artifact.

Adjacent and worth one line because `champions --check` counts it: two seats are held
**BY VERDICT**, the file's strongest marking, on non-verdicts — `Learning core` off
`LC.03=VOID`, and `World` off no named deciding run at all. That is the *inverse* of
this section's question (a VOID treated as a verdict), it is already a counted ratchet
class at baseline 2/2, and its repair is routed. Not a new finding.

---

## §8 THE HONEST SUMMARY — are we closer to a curious humanoid that climbs the ladder?

**Marginally, and today's honest gain is smaller than the day's commit count suggests.**

What genuinely moved him: `HR.7` PASSed — a real A2 auditory stem, 2-channel log-mel
into a conv encoder, worst-seed probe accuracy 0.9453 against a 0.90 gate with the
monotone null at 0.1615. That is a sense he did not have yesterday, and it reopened
downstream work. `LG.02` (yesterday, still fresh) is the strongest recent result in the
project: **he learned which of two advisors lies to him**, from his own attributed
diary, with the swap control migrating. That is a creature doing something.

What moved the *harness*: `T0.32`, the `FILL-HELD` edge, the `HR.5→HR.6` edge, the
amber-pile reader. All good; none of it is Jack. And Finding 2 says `T0.32`'s gate is
not yet wired to anything that runs.

The number that keeps me honest: **12 of 23 constitutional commitments have zero
passing claims**, and the two carrying the most constitutional weight are the
thinnest — `one brain / unison` is **25 specs and 1 passing rule**, curiosity is
12 specs and 2 passing claims. Demonstrated share fell 43.3% → 41.4% in 24 hours.
We are, today, slightly more *registered* than we are *demonstrated*.

And the thing that decides the verdict: for twenty-five days the constitution has
carried a paragraph that states its own falsifier and its own cost of failure — *"if
both integrate identically, he is reciting, and the jungle bought nothing"* — and the
ladder has never had a rung for it. Every organ was working correctly. They were all
reasoning about specs that exist.

So: closer to a longer list of green ticks than to the apple, **but not only that** —
`HR.7` and `LG.02` are real. The correct response is not gloom, it is to put the
missing rung on the ladder this iteration, before anything else, exactly as the
2026-08-10 miss taught.

---

## FOR THE BUILDER

Ranked. **B1 is the "register the uncovered commitment before anything else" rule
firing — take it first, ahead of `T0.33`, `LF.02` and the rest of the fresh runnables.**

**B1 — register a spec for GOAL.md's TOLD WORLD, and give `coverage.py` a name for it.
Both in the same commit** (the standing rule: *"a coverage tool that silently stops
covering something is worse than none"*).

1. Add to `COMMITMENTS` in `experiments/coverage.py:124-148`:
   ```python
   "told world":  (r"\b(told|second.hand|hearsay|anchored|grounded fact|parrot\w*|recit\w*)\b",
                   "owner: the jungle buys MEANING for what he is told"),
   ```
   Choose the pattern so it does **not** collide with `language (parent)` on the
   existing nine — verify by diffing the coverage table before and after; if any of
   `T2.06/T2.07/T2.15/T1.13/LG.00/LG.01/LG.02/LG.10/T3.08` moves bucket, tighten it.
   The `T0.21` re-buy is owed after this edit (`IMPL_DEPS` drift) — pay it clean-tree.
2. Register the spec. GOAL.md:206-212 hands you the whole design and you should take it
   verbatim rather than improving it: two fact sets matched for length and syntactic
   form, one anchored to primitives he has lived (hot, heavy, far, tiring — the
   GOAL.md:187 list), one not; the claim is that **integration differs**, and the
   registry `control` is GOAL.md's own null — *"if both integrate identically, he is
   reciting"*. Suggested id `LG.11` (`LG.10` is the most recent in-family); declare
   `COVERS: told world (claim)`.
3. Expect it to be **unreachable** — it needs `LG.00`'s stripped-core machinery plus
   lived primitives, so it will `depends_on` at minimum `LG.00` and the surviving W1
   line. That is fine and it is the `GEN.02-09` precedent: **register the falsifier
   beside the capability, raise `UNREACHABLE_BASELINE` in the same commit with the
   reason in the growth log.** A truthful red is the deliverable here, not a
   dispatchable spec. Do **not** defer registration to Sunday's W1 design — the
   commitment is 25 days uncovered and the spec's existence is what makes it visible;
   its *world* can be re-parented later.
4. Route a queue row `told-world-has-no-rung` naming (a) whether the fact sets can be
   built before W1 and (b) whether `LG.00`'s frozen-mouth apparatus is reusable.

**B2 — close `T0.32`'s prose binding with an edge, while it is still free.**
Add a conjunct to `experiments/tests/t0_32_rtf_gate.py` asserting that every
**implemented** spec at `budget >= cpu<2h` calls `rtf.require_feasible` (source scan of
its impl file, same idiom as the existing `single_source_ok` check). It is vacuously
true today — `LF.01` is the only such spec and is unimplemented — and it becomes
load-bearing the hour `LF.01` is written, which your own handoff calls a credible next
unit. `T0.32`'s re-buy is owed with it. While there, consider that `single_source_ok`'s
negative half greps for the literal name `_budget_seconds`; a private table under any
other name passes. The positive half (`from .rtf import spec_child_timeout_seconds`,
and I verified `run.py:1797` actually *calls* it) is the load-bearing one, so this is a
tightening, not a defect.

**B3 — declare or disclaim `COVERS:` on today's registrations.**
`LF.02` (save/resume a life) plausibly owes `memory across lives` or `death & retry`;
if it owes neither, say why in `notes`. `SO.01`/`SO.04` cannot declare until B4 exists,
so note that dependency on them rather than leaving them silent.

**B4 — `spectating / being watched` has no line in `COMMITMENTS`.**
You named this yourself on the `DIRECTION_AUDIT` queue row and you were right. GOAL.md
carries *"I want to watch him figure out the world himself"* and *"allow users to talk
to him while he is there doing stuff"*; `SO.01`, `SO.03` and `SO.04` serve it and can
name nothing. Add the line, then declare `COVERS:` on `SO.01`/`SO.04`. Same commit,
same `T0.21` re-buy discipline.

**B5 — the 09-06 pile is still 7 live rows against a measured capacity of 1/cycle.**
Unchanged since the 65th audit flagged it, and the reader now prints it `AMBER` with
five more amber dates behind it. Note that the 65th audit's own B2 added
`hr5-fixture-refuted` to that date. Re-date what can wait, with a reason per row —
`t027-preserved-failimpl` (09-05), `lt01-c2-body-cannot-rise` and the two `d10`
denominator rows are the candidates; `w0-too-shallow` and `lc07-checkpoint-branch` are
the two that genuinely belong to Sunday's design window. Seven promises scheduled to
break together is worse than three re-armed in the open.

---

## FOR THE OWNER

**1. Your own words have been on the ladder's to-do list for 25 days without a rung,
and I want you to see the sentence rather than my summary of it.** GOAL.md:206-212:

> *"FALSIFIABLE, and it must be tested rather than assumed: told-knowledge should
> integrate BETTER when it anchors to something he has lived … If both integrate
> identically, he is reciting, and the jungle bought nothing."*

Zero of 232 specs make that claim. It is not a decision I need from you and I have
routed the repair to the builder as B1 — the constitution already settles that it must
be tested. I raise it because **this claim is the justification for the entire survival
programme.** W0, W1, the needs, the cold that kills, the mortality: GOAL.md justifies
all of it by saying lived primitives are what make told knowledge *mean* something. If
that turns out to be false, the jungle is still a fine world but it is not buying what
the constitution says it buys. It is worth knowing which, and it has been untestable by
construction for 25 days because nobody wrote the rung. No action needed unless you
disagree that it is a commitment.

**2. `D15` and `D16` fire in 2 days (2026-09-05); `D17` on 09-07, `D18` on 09-09.**
All are armed with defaults and reversal paths and none needs you — but two carry
evidence that moved since they were written, and you should have the current numbers if
you intend to rule early rather than let them fire:

- **`D16`** (should `T0.27` stay RED): the "~1 violation per 1.5 days" progression the
  62nd audit recorded has **stopped** — flat at 3 for 27 hours across 30 iterations.
  That strengthens the armed default (keep the red, touch nothing). The number moving
  the other way: unauditable pairs 19 → 23 in twelve hours, so the guard now examines
  28% of the pairs it exists to examine.
- **`D15`** (organ pacing + usage attribution): its option (b) — the one it strikes as
  *"outside the repo and no agent here may take it"* — **remains the largest single
  saving available and is yours to take by hand at any time.**

**3. ~11.07 free GPU-hours expire this Sunday, for the fourth consecutive week.**
Running total lost across W33–W35: **~61 hours.** I checked whether this is avoidable
and it is not, honestly: every GPU cost class is pilot-blocked on *measured* evidence
(`DP.04`'s lifespan has no resolution in W0; `SM.03`'s held-out split is saturated;
`LC.07` projects 14.49 h against an 8.5 h rule; `UB.10` is a VOID awaiting redesign).
The repair is not a dispatch — it is the **W1 world design on Sunday 09-06**, which is
the shared root of `w0-too-shallow`, `sh02-null-saturation`, `dp04-lifespan-has-no-resolution`
and `d10`. Four documents now agree W0 is the bottleneck. That is the meeting that
turns quota back into science.
