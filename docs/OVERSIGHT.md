# OVERSIGHT — 18th audit, 2026-08-19 12:40 UTC

## VERDICT: ON TRACK

The ledger is honest and I checked it the hard way. Nothing in this audit
impeaches a claim. The problems are **throughput and a negative scientific
result**, not integrity — and the negative result is the most valuable thing
the system produced today.

Ranked by damage to the trustworthiness of the ledger:

1. **14 re-run certificates exist only in the working tree.** Verdict-neutral,
   but one `git checkout` from gone. (B1)
2. **T0.27's supersedes-FAIL instrument has audited zero live pairs** since it
   shipped — armed, never fired. (B4)
3. Everything else is clean, and I say so below rather than inventing concerns.

---

## 0. Is the ladder the RIGHT ladder? — `coverage.py` GREEN (rc=0)

**0 commitments with no declared spec.** The 2026-08-10 class of miss — a
constitutional commitment with no falsifiable claim behind it — is closed and
has stayed closed.

The honest number underneath it: **16 of 23 commitments have specs but nothing
passing.** Only 7 commitments have a single passing claim each (damage,
memory-across-lives, generality, language, hearing, curiosity, one-brain).

---

## 1. Integrity of the ledger — NO FINDINGS ON ANY CLAIM

88 rows, **81 PASS**, 3 FAIL (T2.01, T3.07, T4.02), 4 VOID (BA.02, LC.03,
T2.02, T2.05).

| check | result |
|---|---|
| PASS rows whose `commit` is reachable in git | 81 / 81 |
| PASS rows with an implementation file on disk | 81 / 81 |
| PASS rows with a spec-declared `control` | 79 / 81 |
| …whose `_check` **references its control argument** (AST over the 2nd formal) | **79 / 79** |
| PASS rows with declared control but no `control_metrics` recorded | **0** |

The two control-free PASS rows are `T0.01` (repo imports clean) and `T0.10`
(Kaggle job round-trip) — harness fixtures with nothing to control against.
Legitimate.

I checked the direction the existing guard does *not* cover. `run_spec`'s
`UndeclaredControl` catches **control run but not declared**; it cannot catch
**declared but never run**. I checked that second direction independently:
zero offenders.

### RANK 1 — the ledger is uncommitted

`experiments/ledger.json` is dirty: **+1112 / −146 lines**, 14 rows touched
(BA.01, LC.02, PG.2, PG.3, PG.5, PG.6, PG.9, PS.03, SM.01, T0.17, T0.21,
T0.27, T2.20, TA.01).

I diffed it semantically before trusting it. The diff adds **14 `"status":
"PASS"` history rows and nothing else** — zero FAIL→PASS flips, zero status
downgrades, zero metric edits to existing rows. This is the stale-certificate
recovery chain (`/data/stale_rerun_chain.sh`, pid 3979902, still running at
PG.4) doing exactly what commit `f45afa4` said it would. Ledger writes are
`flock`-serialised with re-read-merge and `tmp+fsync+os.replace`, and chain2
blocks on chain1's completion marker, so **there is no write race** between the
two live chains — I checked because two concurrent recorders is exactly the
shape that corrupts a scoreboard.

The risk is not corruption, it is loss: 40+ minutes of re-run evidence lives
nowhere but the working tree.

### The dirty-stamp guard demonstrably works

8 rows carry `+dirty` (PG.2, PG.3, PG.5, SM.01, T0.17, T0.21, T0.27, T2.20) —
they ran 12:12–12:13Z from a tree holding an uncommitted `LEARNING_CORE.md`.
The loop caught this **itself** at `50c46bf` and armed phase 2 to re-run them
clean. The proof the guard is not decorative: **VO.01 was REFUSED on PG.5
DIRTY** rather than recording a convenient PASS off a dirty dependency.

### The one instrument that has never fired

`audit_supersedes_fail` (T0.27's executable form of "a moved threshold after a
FAIL leaves an artifact") reports `violations: 0` — but also **`checked_pairs:
0, unauditable_pairs: 27`**. Every pair predates the instrument. A guard with
zero live coverage is a guard whose first real test will be in production.

---

## 2. Thresholds and controls over time — NO FINDINGS, POSITIVELY VERIFIED

44 commits in 7 days touched the registry or `experiments/tests/`. I isolated
the only ones that **modified an existing gate constant** (removed-and-re-added,
not merely added). There are five, and **not one moved in the loosening
direction**:

| commit | change | verdict |
|---|---|---|
| `1861b18` BA.02 registration | `TOPPLED_FRAC_MIN` 0.60, `RANDOM_UP_FRAC_MAX` 0.80, `IMPROVE_MARGIN_MIN` 0.20, `NOISE_GAIN_FRAC_MAX` 0.50, `VEST_OVER_NOISE_MIN` 0.20 | **comment text only** — `[PILOT-FINAL]` → measured pilot value. All five values byte-identical. The commit message's claim "every PILOT-FINAL constant finalised UNCHANGED" is **true as written**. |
| `c1bd242` BA.01 v3 | `TF_SPREAD_MIN = 2.5` → `TF_ABS_SPREAD_MIN = 2.5` + `TF_FALL_SPREAD_MIN = 2.5` | one name split into two, **both at 2.5**. A gate added, none moved. |
| `0fce271` BA.01 v2 | `HORIZON` stays 80; `OMEGA0_STD` removed | rig redesign (hold-then-release); the kick returns as `KICK_OMEGA_P` in `fe16e06`. Not a weakening. |
| `a703604`, `60686ac` | `N_PROPERTIES` 7 → 8 → 9 | **tightening** — more properties checked. |

- **Seed counts:** every claim spec touched kept `seeds=3`.
- **Budget moves** (T2.08 `GPU`→`CPU`, LC.03 `CPU_LONG`→`CPU_DAYS`) are
  apparatus sizing with measured justification in the commit message; they
  touch no gate.
- **No control deleted, no `_check` gained an `or`, no assertion removed.**

Saying this plainly is the result: **there is no silent loosening in this
repository over the last 7 days.**

---

## 3. Drift from the goal — none in the work; the hole is in the coverage

Everything the builder did since 2026-08-14 traces to a GOAL.md sentence:

| work | GOAL.md sentence it serves |
|---|---|
| **T2.06** grounding (sight+language → anchor, ~0.99 vs 0.1 chance, tf-idf null beaten on all seeds) → **PASS** | *"what he hears can teach what he sees"* — one brain, senses in unison |
| **SH.01** shelter substrate + 3 pilots + oracle probe | *"Cold nights teach shelter-building the way no scripted lesson can"* |
| stale/dirty re-run chains, `+dirty` correction | *"protects the honesty of watching what happens"* |

**No drift.** No work in the last 5 days serves nothing.

The converse question is where the damage is. **16 of 23 commitments have zero
passing specs**, including every one the prompt warns about:

- **curiosity — 12 specs, 1 pass**
- **one brain / unison — 21 specs, 1 pass**
- **fast/slow — 6 specs, 0 pass**
- touch, shelter, tool use, smell, voice, balance, proprioception, thermal,
  hunger/thirst, sleep, death & retry, social, plasticity, taste, sight — **all
  zero.**

Several carry passing *support* rows that coverage correctly refuses to credit
(VO.01 sensor, BA.01 sensor, SM.01/TA.01/PS.01/PS.02 fixtures). That refusal is
right — a fixture is apparatus, not a capability — and it is why the honest
count is 16 and not 7.

---

## 4. Is the builder alive? — alive, but it lost 4 days 18 hours to credits

**The loop was stopped from 2026-08-14T13:23 to 2026-08-19T07:31.** 135 hourly
cron firings logged `STOPPED at 99–100% weekly usage — all agents paused until
the owner resumes`.

This is **correct behaviour, honestly logged** — the owner's 90% rule, working.
It is not a defect and I am not reporting it as one. I am reporting the
magnitude: the binding constraint on this project is neither compute nor ideas.
It is Claude credits, and they cost ~114 working hours in the last 5 days.

Since the reset (last 24 h):

- **6 iteration starts**, 5 `rc=0`, 1 `rc=1` (session limit at 11:07, marked
  lost, recovered and cleared at 12:07 — the lost-iteration machinery worked).
- **PASS delta 80 → 81** (+1, T2.06). One intermediate dip 81→80 at 08:23 was
  an honest VOID recorded off a stale borrow, then repaired.
- No repeated identical failures, no paused-and-forgotten loop, no aborts on
  load (load 0.00–0.71 across the day).

---

## 5. Compute honesty — NO FINDINGS

- **W33 (current): 0.297 h spent of 30 h → ~29.7 h remaining, expires Sunday
  2026-08-23.**
- The single W33 job (`jack-ladder-1787124880`, 1069 s billed, est 0.4 h) →
  **T2.06 PASS**. Sized from a production-config P100 probe first. No overrun.
- Every W32 charged job maps to a named spec or a named probe. The two failed
  attempts (0.1225 h kaggle, 1.053 h colab) are booked as **failed**, not
  hidden.
- The 6.3849 h W32 opening balance stays unattributable and is deliberately
  **not lowered** — over-stating spend is the safe direction, and
  `remaining_range()` reports the honest interval.

**No GPU hours were spent with nothing to show for them.**

---

## 6. Stuck decisions

- **D1 (does the 57M trunk stay in the control path?) — open since 2026-08-04,
  15 days.** It has accumulated six overseer cost updates and is the
  longest-standing block in the project. Evidence is complete; the 15th audit
  correctly refused to let it be answered with "do what the measurements say",
  because the option set was unconstitutional. It still needs the owner.
- **D7 (MovementMoodCoupling failed its ablation)** — fully evidenced with a
  localised cause (training ~20× too weak; span 0.026–0.036 of the designed
  0.6, reference arm reaches 0.52). Decidable now.
- **D8 (BA.02 unmeasurable in the rover body)** — fully evidenced with four
  scratch probes; the claim's measured contrast ceiling (~0.0–0.1 s) sits below
  its own pre-registered floor.

**Nothing is blocked that a bakeoff could have settled.** One note on the
converse: BA.02 has been *parked* per D8 (excluded by name from the re-run
chain). That is a park, not a resolution, and it is recorded in both the chain
script and the commit — acceptable, but a script comment is a weaker record
than `DECISIONS_RESOLVED.md`.

---

## 7. Bakeoff hygiene — NO FINDINGS

Three resolved decisions, all clean:

- **PS.01/J — VOID**, recorded as VOID and explicitly **not** treated as a
  verdict; PS.01/J2 re-ran and produced the winner (`impact_speed`). This is
  the correct handling of a VOID.
- **D2 — VOID BLOCKS its dependents**, resolved by ledger replay with a
  quantified exposure (9 retractions vs 0), a named loser, and a re-open
  trigger. Made executable as T0.08 property 6.

No decision was made without a learning gate. No winner was chosen inside a
noise margin.

---

## 8. The honest summary — are we closer to a curious humanoid?

**Marginally, and honestly. The day's most important result is a negative one.**

SH.01's **oracle reference arm** — the must-succeed arm, handed a unit-vector
direction straight to a working hut — reached **z = 1.028 at N = 12000
decisions against a gate of 3**. The reference cannot learn shelter-seeking at
the full CPU envelope.

That finding is not about shelter. It is about the **learning core**: a
certified core failed to acquire a survival behaviour the body can physically
execute, with the answer handed to it. The loop routed it to LC.04's design
notes and **declined to launch the registered run** — refusing to spend the
ladder's credibility on an experiment it already knew could only record VOID.
That is the single best judgement call in this audit period, and it is the
behaviour the falsification ladder exists to produce.

Set against it:

- **+1 PASS in 5 calendar days** — but T2.06 is a real unison claim (sight and
  language into one anchor, 0.99 vs 0.1 chance), not a green tick.
- **16 of 23 commitments still have nothing passing.** Curiosity — the north
  star, the thing that makes him climb the ladder at all — has 12 specs and one
  passing claim.
- The loop was dead for 4 of the last 5 days.

So: we are closer to a *trustworthy* ladder than we were, and barely closer to
a curious humanoid. The ledger is not the problem — I could not find a single
dishonest claim in it, and I looked in the two places that hide them best
(declared-but-unrun controls, and constants moved under a plausible commit
message). **The rate is the problem, and the rate is bounded by Claude credits
and by a learning core that just failed its easiest survival task.**

---

## FOR THE BUILDER

**B1 (RANK 1) — commit the ledger before anything else.**
`experiments/ledger.json` holds 14 re-run certificates (+1112/−146) that exist
in no commit. Verified verdict-neutral: 14 new PASS history rows, zero
FAIL→PASS flips, zero downgrades. Wait for `=== phase2 done` in
`/data/sh01_stale_chain.log`, then commit. Do not let a third chain start first.

**B2 — verify the dirty stamps actually cleared.** After phase 2, run
`$PY -m experiments.run status` and confirm the `DIRTY STAMPS` section is
**empty**. It currently names 8 rows (PG.2, PG.3, PG.5, SM.01, T0.17, T0.21,
T0.27, T2.20). Report the count, not the intent — `50c46bf`'s own lesson.

**B3 — VO.01 specifically.** Chain 1 refused it on `PG.5 DIRTY`; phase 2 re-runs
it last, after PG.5 is clean. Confirm it **recorded**, and did not get refused a
second time. It is voice's only support row, and voice is a zero-pass
constitutional commitment.

**B4 — give T0.27's supersedes-FAIL audit its first live pair.**
`audit_supersedes_fail` reports `checked_pairs: 0, unauditable_pairs: 27`: the
instrument that makes "a moved threshold after a FAIL leaves an artifact"
executable has never evaluated a single real pair, because all 27 predate it.
Add a synthetic FAIL→PASS fixture pair to T0.27's property set so the checking
path itself is exercised, rather than waiting for a live violation to be its
first test.

**B5 — the standing-rule pick should stay on zero-pass claim specs.** The
journal already names XL.01 / VO.02 / TA.02 / SM.02. All four buy a
constitutional commitment its first passing claim; W33 has ~29.7 Kaggle hours
that die Sunday 2026-08-23.

---

## FOR THE OWNER

**1. D1 is 15 days old and is now the project's longest-running block.**
Open since 2026-08-04, with complete evidence and six separate cost updates
from successive audits. Every audit that touches it has to re-explain why it
still matters. It needs a decision, not another cost update.

**2. Claude credits stopped the loop for 4 days 18 hours — 135 consecutive
hourly firings.** Between 2026-08-14T13:23 and 2026-08-19T07:31 the loop logged
`STOPPED at 99–100% weekly usage` and did no work at all. The 90% rule behaved
exactly as you specified and the logging was honest; this is not a malfunction
report. It is a resource report: **credits, not GPU hours and not ideas, are
what sets this project's rate.** W33 has ~29.7 unused Kaggle hours that expire
Sunday, and the loop may not have the credits to spend them.

**3. Two decisions are fully evidenced and waiting on you: D7 and D8.** Both
have measurements, localised causes, and no remaining experiment that would
change the answer. D7: mood conditioning fails its ablation (0.225/0.275/0.375
vs chance 0.25) but is *not* unlearnable — the shipped training is ~20× too
weak. D8: BA.02's contrast has a measured ceiling of ~0.0–0.1 s against its own
pre-registered floor of 0.20 s — the rover body has no actuator with
directional catch authority.
