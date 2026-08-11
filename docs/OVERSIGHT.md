# OVERSIGHT — 8th audit, 2026-08-11 21:10 UTC

## VERDICT: DRIFTING

**The ledger is honest. The allocation is not.** Sections 1, 2, 5 and 7 are
clean — I looked hard for silent loosening and found the opposite, a week of
strictly-tightening controls. What is wrong is what the machine spent itself on:
in the 12 hours the loop was allowed to run it produced **one** new capability
(SM.01, smell) and thirteen re-runs of its own harness, while the single spec
that blocks 36 others sat untouched and 18.04 Kaggle GPU-hours ticked toward
Sunday's expiry unspent.

And one instrument is reading green over a hole. See RANK 1.

---

## 0. Is the ladder the RIGHT ladder?

`python -m experiments.coverage` → **exit 0. Zero commitments with no declared
spec.** The 2026-08-10 miss has not recurred; every GOAL.md commitment now has
at least one spec that has *declared* `COVERS:` for it.

But the second column is the story: **12 of 23 commitments have specs and
nothing passing**, and two of the eleven that "have something passing" do not
really. That is RANK 1 below.

Headline state: **163 specs · 67 PASS · 2 FAIL · 1 ERROR · 1 VOID · 92 not
implemented. 67 of 163 unreachable. 0/10 of the sensory inventory is
LOAD-BEARING** — the standard GOAL.md actually sets.

---

## RANK 1 — `coverage.py`'s `n_pass` can be discharged by a spec that does not test the commitment

**This is the most damaging finding, because it is the failure mode the tool was
built to prevent, reappearing one level up.**

The two claims closest to the thesis report as covered *with a passing spec*:

| commitment | specs | pass | the passing spec is… |
|---|---|---|---|
| one brain / unison | 21 | 1 | **LC.01** — *"Every candidate core takes every sense into one latent, or it is not a candidate"* |
| curiosity | 12 | 1 | **PG.4** — *"Noisy-TV panel traps naive curiosity"* |

Neither PASS is evidence for the commitment it discharges:

- **LC.01 is an admission RULE**, not a fusion result. It says any *future*
  candidate core must accept every modality. It is a gate on arms that have not
  been run. All 16 UB specs — the ablation matrix, the binding test, the placebo
  modality, the actual "senses fuse" claims — are `NOT_RUN`. The constitutional
  sentence *"a genuinely unified brain where every sense is load-bearing (and we
  PROVE each one is — ablate a sense, something measurable must degrade)"* has
  **zero** passing specs. `run senses` agrees independently: **0/10
  LOAD-BEARING**.
- **PG.4 is a world-fidelity spec.** It proves the playground *contains a
  working noisy-TV trap* — a property of the environment fixture. It is
  evidence that a distractor exists to fool a curious agent, not that Jack is
  curious. All 6 CU specs and T2.08 ("Curiosity drives coverage") are `NOT_RUN`.
  GOAL.md's north-star sentence — *"he explores because he wants to"* — has
  **zero** passing specs.

`coverage.py` is not lying; it counts declared specs that are PASS, and it says
so. The defect is that **a `COVERS:` declaration plus any PASS satisfies the
count, regardless of whether the passing spec bears on the claim.** A trap
fixture and an admission rule are exactly the cheap, adjacent, infrastructure
specs a loop optimising for green ticks will reach for first — and they silence
the one instrument that would have flagged the gap. This is the same species as
the 2026-08-10 miss: *the hole is invisible to the instrument that owns it.*

**Not manufactured:** I checked the converse. The other 9 "has a pass"
commitments (hunger/thirst→PS.01, smell→SM.01, death&retry→XL.00, taste,
memory-across-lives→ME.9/ME.10, language, fast/slow→DP.00, sight, hearing) do
have specs whose PASS bears on the claim. The defect bites on exactly two — but
they are the two GOAL.md calls the thesis.

---

## RANK 2 — T2.01 is the whole project's bottleneck and got zero attention

`run blocked`:

```
T2.01 = FAIL  frees 26  (blocks 36)  — Locomotion beats a random policy
      frees: CU.1-CU.7, ME.7, T2.16-18, T3.02/04/05, T4.04/05,
             T5.01-05, T5.07, T6.01/02/04/05
LC.03 = NOT_RUN  frees 7   ← the next-largest, 3.7x smaller
```

**Every curiosity spec, every Tier-5 claim, and every Tier-6 living-Jack spec is
behind this one FAIL.** RANK 1's curiosity gap is not a separate problem — it is
T2.01's shadow.

The resource and the blocker match exactly and neither moved:

- T2.01 v4 is registered at `gpu<8h`, `est_hours=6.5`, `prefer="kaggle"`.
- **18.04 of 30 Kaggle hours remain in W32 and expire Sunday 2026-08-16.**
- Nothing has been submitted for T2.01 since the v4 re-spec.

In the 12 hours the loop ran (15:57→20:38) it worked on SM.01, VO.01, and 13 T0
harness re-certifications. Each of those traces to a GOAL.md sentence — smell
and voice are named senses, and the harness work is SYSTEM.md's "loop on
itself" — so none of it is drift in the strict sense. But SYSTEM.md's own guard
names this pattern: *"when the machine is sufficient, PROVE it by throughput …
the guard against polishing the machine instead of running it."* Thirteen of
eighteen ledger rows in 24 hours were T0. That is the machine working on itself.

---

## RANK 3 — an orphaned GPU job will write a ledger row into a repo nobody is watching

Live as I write:

```
PID 2034160  PPID 1  (orphaned — its iteration died rc=1 at 20:38)
  python -c "... m = _module_for('T1.02'); m.run(Ledger())"
```

Sequence, from `experiments/gpu_submissions.jsonl`:

| time | event |
|---|---|
| 20:08:13 | T1.02 dispatched to **colab**, `est_hours=0.7` |
| 20:38 | parent iteration hits max turns, `rc=1`; **child survives, reparented to init** |
| 21:03:25 | **owner writes `.loop-paused`** — "does NOT self-expire" |
| 21:07:42 | colab returns `ok=false` after 3569 s; **fails over to kaggle**, still running |

When the Kaggle job lands (~22:07), `m.run(Ledger())` writes T1.02's result
**directly into `experiments/ledger.json`** — correct behaviour, and the row
will be real. But the loop is paused, so **nothing will commit it**. The result
sits as an uncommitted working-tree change indefinitely, and any spec run from
that tree afterwards inherits a `+dirty` provenance stamp.

This is not a integrity failure yet. It is a booby trap for whoever resumes: the
first thing they will see is a dirty tree containing a legitimate, unexplained
ledger row.

---

## RANK 4 — the append-only GPU evidence log contains a line that corresponds to no dispatch

`experiments/gpu_submissions.jsonl` line 2, committed in `b52c0eb`:

```json
{"phase":"selftest"}
```

No `attempt_id`, no `backend`, no `ts`. It was written by the builder while
testing the `+dirty` rule and **committed into the production evidence log.**

`_record_submission`'s own docstring states the scar it exists for: *"An
iteration that never called `submit()` and one whose submission died in flight
left byte-identical evidence… a claim of 'I submitted X' becomes checkable
rather than believable."* An evidence log that test code can write to, and did,
is weaker evidence than one it cannot. `submissions()` parses and returns that
line as a record.

Severity is genuinely low — the line is unmistakably not a dispatch, and nothing
downstream reads the log for decisions. It is reported because the *precedent*
is the problem, and because **no spec gates the log's contents.** The
19:29 iteration fixed the adjacent case (`_probe_submit` writing stubs to the
default journal) one hour before this line landed.

---

## RANK 5 — 40 ledger rows cannot be checked for staleness at all

`run status`: *"40 entries predate `impl_sha` and CANNOT be checked for
staleness."* `run stale` reads **0 DIRTY, 0 CHANGED** — genuinely clean, and the
builder earned that in the last two iterations. But 40 of 67 PASSes are silent
rather than verified: they would not report a changed implementation because
they have no baseline to compare against. The builder measured this correctly on
2026-08-11 (refusing UNVERIFIABLE borrows takes runnable 27→7) and chose the
right trade-off. Noting it as standing exposure, not as a new fault.

---

## RANK 6 — pain and temperature have no sensor at all

`run senses`: **`[ABSENT] pain (nociception)`** and **`[ABSENT] temperature
(thermoception)`** — *"load-bearing: NO SPEC would prove it."* Both are named in
GOAL.md's sensory inventory and both are consequential to the survival world
("too cold kills him"). They have registered specs (PS.02, PS.03, SH.01) but no
implementation, so 8/10 of the inventory is spec'd and 0/10 is load-bearing.

---

## Section-by-section

### 1. Integrity of the ledger — **CLEAN**

Checked all 67 PASS rows programmatically:

| check | result |
|---|---|
| implementation exists in `experiments/tests/` | **67/67** |
| `commit` still resolves in git | **67/67**, 0 missing |
| PASS stamped `+dirty` | **0** |
| spec declares a `control` | 65/67 |
| declares a control **and** ledger has non-empty `control_metrics` | **65/65** |

The two without a declared control are **T0.01** (imports) and **T0.10** (Kaggle
round-trip). T0.10 is a known, self-reported gap: the 19:30 iteration wrote its
control design into the journal and deferred implementing it because doing so
marks T0.10 CHANGED and T2.02 depends on it. That reasoning is correct and the
deferral is recorded. No action beyond keeping it visible.

**No PASS in this ledger is a claim without evidence.**

### 2. Thresholds and controls over time — **CLEAN, and strongly so**

89 commits touched `registry.py`, `registry_expansion.py`, `experiments/tests/`
in 7 days. I diffed the *resolved spec objects* between `5f23e34` (7 days ago)
and HEAD rather than eyeballing 1,902 changed lines, and separately extracted
every number-only edit to a pre-existing line.

- **Controls: 24 changed, 24 in the strengthening direction.** 22 went
  `None → a declared control`. The two rewrites both added a *second* control
  (T1.02: "Two of them…"; T0.12: "Three named broken mechanisms…").
- **Controls deleted or weakened: 0. Specs removed from the registry: 0.**
- **Seeds: 8 specs went 1 → 3** (T1.01, T1.02, T1.03, T1.04, T1.05, T1.12,
  T6.03). One went the other way — **T1.08, `seeds=3 → 1`** — which is the exact
  shape of a silent loosening, so I verified it: T1.08 is the spec *about* seed
  variance, its notes pre-declare the change (*"seeds=1 AT THE SPEC LEVEL,
  deliberately… the GPU job varies seeds [0,1,2] internally in one kernel"*),
  and `t1_08_seed_variance.py` does run three internal arms with a
  `MIN_SPREAD = 1e-6` control that fails if seeding is fake. **Justified.**
- Every other number-only edit is a tightening or a cost/timeout change with a
  measurement in the commit message: `N_PROPERTIES` 6→7, 9→12, 12→13; PG.6
  `RES` 64→96 and `N_TRAIN` 700→1800 (pilot-selected, stated); T0.05's
  `sleep 0.05→0.0002` (a *sharper* kill window — the control now corrupts 12/12).
- Net assertion count in tests: **+3 added, −1 removed.**

**No `_check` gained an `or`. Nothing was loosened. This section is a genuine
pass and I want it recorded as one.**

### 3. Drift from the goal

What the builder did, 15:57 → 20:38, and what each serves:

| work | GOAL.md sentence | verdict |
|---|---|---|
| SM.01 smell — PASS, 3 seeds | *"EVERY SENSE A HUMAN HAS … SMELL"* | serves |
| VO.01 voice — FAIL ×2, recorded | *"and VOICE — he must be able to make sound"* | serves |
| PS.01, PG.5, T0.20, XL.00 re-certification | honesty of measurement | serves |
| T0.22/T0.23, dirty-stamp, dependency freshness | SYSTEM.md "the loop on itself" | serves |

**No item is drift in the strict sense — all four trace to a sentence.** The
problem is the ratio, covered in RANK 2: 13 of 18 ledger rows were T0, and
exactly **one first-ever PASS** (SM.01) in 24 hours. Everything else was either
re-certification of an already-green row (PS.01 attempt 4, PG.5 attempt 4,
XL.00 attempt 5) or harness work.

**The converse — what has no passing spec at all:** unison (RANK 1), curiosity
(RANK 1), generality (4 specs, 0 pass, T1.02 ERROR), plasticity (T5.03/T5.04,
0 pass), thermal, shelter, tool use, damage, social, sleep, proprioception,
balance, touch, voice. **12 of 23 commitments.**

### 4. Is the builder alive and productive?

- **22 h 53 m dead**, 2026-08-10T17:07 → 2026-08-11T15:57: 23 consecutive hourly
  wakes logging `STOPPED at 90-92% weekly usage`. This is the owner's own rule
  working as specified; the resume mechanism it always lacked shipped at 15:56
  (`b1db303`).
- **5 iterations after the resume**, 4 × `rc=0`, 1 × `rc=1` (max turns, 120).
  One `previous iteration still running — skipping` at 16:07.
- **PASS delta 65 → 67** over that window: SM.01 (new capability) and T0.23
  (harness guard).
- **Fable is unusable** — every iteration since 18:07 logs `OUT OF CREDITS on
  fable — falling back to opus`, 3 times. The fallback works; the model
  selection is decorative.
- **The loop is now paused by a second, independent mechanism**: the owner wrote
  `.loop-paused` at 21:03:25 ("does NOT self-expire"), 15 hours *before* the
  usage grant expires. D5's three options are therefore partly moot — the
  binding constraint right now is the manual pause, not the ceiling.

### 5. Compute honesty — **CLEAN, and I watched it work**

- **Kaggle W32: 11.9635 of 30 h used. 18.04 h remain, expiring 2026-08-16.**
- **Colab W32: 0.0015 h succeeded, 0.9914 h failed.**
- The W31 overrun (`kaggle: 37.4554` against a 30.0 ceiling) is a *known,
  already-instrumented* scar — `gpu.py:293-304` cites it by number and the
  overrun detector was added afterward. Not a live finding.

The receipt organ built at 19:29 discharged its job **during this audit**: the
colab attempt at 20:08 failed after 3569 s, and rather than vanishing it was
written to `charged_jobs` as `{"hours": 0.9914, "ok": false}` and to a separate
`colab_failed` bucket. **A GPU hour that bought nothing is now recorded as a
GPU hour that bought nothing.** That is exactly the design intent and it is the
best thing that happened in the last 24 hours.

No GPU hours are unaccounted for.

### 6. Stuck decisions

`DECISIONS_NEEDED.md` (885 lines) has **8 open owner items**. Nothing is blocked
that the system could have resolved itself with a bakeoff, and no owner decision
appears to have been quietly acted on — the D3 push question was properly
recorded as ANSWERED, and previous overseers' STALE/HOUSEKEEPING corrections are
struck in place rather than deleted.

**One item is time-critical: D5, hard deadline 2026-08-12T12:00 UTC (~15 h).**
It now needs restating because the facts moved after it was filed — see FOR THE
OWNER.

### 7. Bakeoff hygiene — **CLEAN**

`DECISIONS_RESOLVED.md` holds 2 decisions, both from PS.01:

- **PS.01/J → VOID**, correctly: three arms below the 3.0σ learning gate, with
  the right reasoning recorded verbatim (*"An arm that has not demonstrably
  learned cannot arbitrate the decision"*). **A VOID was not treated as a
  verdict.**
- **PS.01/J2 → WINNER `impact_speed`**, beating the runner-up by **2.66σ** and
  the null by **10.32σ**, with all 11 gate-eliminated arms named. `MIN_FINISHERS
  = 2` was satisfied. The winner is outside the noise margin, not inside it.

No decision was made without a learning gate.

### 8. The honest summary — are we closer to a curious humanoid?

**Marginally, and less than the ticks suggest.**

Yesterday Jack could not smell. Today he can, with the strongest single number
in the last day attached to it: occluded from the eye, the nose still reads
**477× the noise floor**. That is a real new sense, honestly earned, with its
shortfall (intermittency 0.41/0.55/0.63 against Farrell's 0.852/0.901/0.837)
reported rather than buried. **The builder is not cheating — it is telling the
truth about its own weak results, twice, unprompted.** Sections 1, 2, 5 and 7
say the same thing from four directions.

But the question is whether he is closer to *climbing the ladder out of
curiosity*, and the answer is no. Curiosity has 12 specs and not one of them has
run. The unified brain has 21 specs and not one of them has run. Zero of ten
senses are load-bearing, which is to say: **no sense in this system has yet been
shown to change anything Jack does.** We have built an increasingly trustworthy
instrument and pointed it, for the last day, mostly at itself.

The gap between those two facts is the whole finding. The ledger is worth
believing. What it is being asked to measure is not yet the thesis.

---

## FOR THE BUILDER

Ranked. **1 and 2 first, and 2 is time-boxed by Sunday.**

1. **Make `coverage.py` distinguish a commitment-bearing PASS from an adjacent
   one.** Concretely: a `COVERS:` declaration should carry a kind —
   `COVERS: curiosity (claim)` vs `COVERS: curiosity (fixture)` /
   `(rule)` / `(sensor)` — and the summary's `n_pass` column must count only
   `claim` declarations, reporting the others in a separate column. Then
   re-declare **PG.4 as `(fixture)`** (it certifies the noisy-TV trap, not the
   agent) and **LC.01 as `(rule)`** (it gates future arms, not present fusion).
   Expected effect: commitments-with-nothing-passing goes **12 → 14**, and
   `curiosity` and `one brain / unison` correctly read zero. Add the property to
   T0.21 (`coverage_audit_honest`) in both directions: a ledger where only a
   `(fixture)` spec passes must read 0 pass for that commitment, and the same
   ledger with a `(claim)` spec passing must read 1. **A coverage tool that
   counts the wrong kind of green is the failure it was built to prevent.**

2. **T2.01, this week, on Kaggle.** It frees 26 specs — 3.7× the next blocker —
   and every curiosity and Tier-5 spec is behind it. **18.04 h remain and expire
   2026-08-16;** T2.01 v4 is `est_hours=6.5`, so it fits with room for one
   retry. It has been FAIL since the v4 re-spec with no submission attempted.
   Do not spend the week's remaining GPU on anything cheaper first.

3. **Collect T1.02 before doing anything else.** PID 2034160 is orphaned
   (PPID 1) and running on Kaggle since 21:07:42; it will write T1.02's row
   into `experiments/ledger.json` around 22:07 with the loop paused, leaving an
   uncommitted legitimate result in the tree. On resume: read the row, read
   `reference_gain` *before* verdicting (under 1.5 it is VOID — a statement
   about the task, not the architecture), commit it with its provenance, and do
   **not** re-run it. Note the colab arm already burned 0.9914 h returning
   `ok=false`.

4. **Gate the GPU evidence log's contents.** `experiments/gpu_submissions.jsonl`
   contains `{"phase":"selftest"}` (committed in `b52c0eb`) — a line matching no
   dispatch. Add a property to T0.12's battery: every line in the default
   `SUBMISSION_LOG` must carry `attempt_id`, `backend`, and
   `phase ∈ {attempt, result}`; and make test code structurally unable to write
   to the default path (require `journal` positionally, as was done for
   `_probe_submit`). Check it in both directions — a log containing a stub line
   must FAIL the property, or the property is decorative. Leave the existing
   line in place with a comment; deleting evidence to make a gate pass is the
   wrong lesson.

5. **UB.9 is the cheapest route into the unison hole.** `run blocked` ranks it
   3rd (frees 4, blocks 7) and it is *"Heard, not seen: the task that is
   impossible without fusion"* — the one spec whose PASS would legitimately
   discharge part of the unison commitment. It has been deferred for four
   consecutive iterations and is now the oldest untaken finding.

---

## FOR THE OWNER

**1. D5 needs your answer in ~15 hours, and its facts have changed.** The usage
grant expires **2026-08-12T12:00 UTC**. Since D5 was filed you also wrote
`.loop-paused` (21:03, non-expiring), so the loop is stopped by *two*
independent mechanisms and lifting only one will not restart it. If the pause
was deliberate, D5's three options can wait; if it was meant to be temporary,
`.loop-paused` needs removing **and** the grant extending, or nothing happens at
12:00 either way. Appended to `DECISIONS_NEEDED.md` with this correction.

**2. 18.04 free Kaggle GPU-hours expire Sunday 2026-08-16 and nothing has been
submitted for the project's #1 blocker.** T2.01 (`est_hours=6.5`) frees 26
specs including every curiosity and Tier-5 claim. Those hours only get spent
during hours the loop is allowed to run — which is decision 1. **This is the
concrete cost of the pause: not "a slower week", but the difference between the
curiosity thesis being testable this month or not.**

**3. Nothing is wrong with the ledger.** I went looking for silent loosening
across 89 commits and found 24 controls strengthened, 8 specs moved from 1 to 3
seeds, zero controls deleted, zero specs removed, and zero PASSes without
evidence. The builder reported two of its own FAILs and a shortfall it was not
obliged to disclose. **If you are deciding how much to trust this system's green
ticks: trust them. Decide instead whether it is measuring the right things** —
0 of 10 senses are load-bearing, and curiosity, the north star, has 12 specs of
which none has ever run.
