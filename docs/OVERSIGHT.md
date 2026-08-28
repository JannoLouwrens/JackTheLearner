# OVERSIGHT — 43rd audit, 2026-08-28 18:45 UTC

## VERDICT: DRIFTING — **`champions.py` asks "has the arena run?" when the question is "has a CHALLENGER run?", and the difference lets a seat be certified contested by its own incumbent.** Both of the file's `BY DEFAULT` seats — the two whose own cells say, in bold, **"DEFAULT, never defended"** — pass the check clean. The tool's own known-positive fixture asserts that they should.

**State.** `HEAD` is `6ad5ed3` (the 42nd audit). **Zero builder commits since
2026-08-25 10:14:58**; the last builder iteration ended **2026-08-25 12:23:33 —
78.4 hours ago**; **78 consecutive `PACING: … skipping` slots and nothing else**
in `ladder.log` since `08-25 13:07`. **84 PASS / 187 registered (44.9%)**. The
untracked `experiments/tests/sm_03_nose_reports_occluded.py` (32 KB, mtime
`Aug 25 12:20`) is still the only thing in the working tree. Meters at 18:07:
`week:all models` **73%** (the gate) at **65%** of the week, line **68%**;
`week:Fable` **100%**. Kaggle W34: **0.3111 h** charged, **29.6889 h** expiring
**Sun 2026-08-30 00:00 UTC — 29.4 hours from now**.

**The three constitutional gates are green, and I re-ran all three myself.**
`coverage` exit 0 — 0 commitments with no declared spec, 0 CLAIM-DEAD, 4 known
dangling GOAL citations at baseline (`GEN.02/03/06/09`, unchanged since the 29th
audit seeded them). `decisions --check` exit 0 — **0/10 undeclared**, no
`MEANS-ESCALATED`, no `OVERDUE`; nothing to arm this audit, and I am recording
that rather than manufacturing an entry. `champions --check` exit 0 — ratchet
**6/8**. That last green is the subject of RANK 1.

---

## RANK 1 — the contestability check quantifies over the arena LIST, not over challengers

`experiments/champions.py:317` is the whole test for a seat that cannot lose:

```python
if (s["held"] in HELD_UNEARNED and s["arena_present"]
        and all(v == "NOT_RUN" for v in s["arena_status"].values())):
    violations.append(("UNCONTESTED", ...))
```

`all(... == "NOT_RUN")`. **One arena spec having run — any one, for any reason —
discharges the debt for the entire seat, permanently.** The docstring's own
definition is *"arena EXISTS, and has never run"*, and for a seat with a single
arena those coincide. For a seat with several they do not, and the ones with
several are the consequential ones.

**Measured against the live file and ledger.** `HELD_UNEARNED` is
`{BY DEFAULT, BY DECREE}` — the two markings `CHAMPIONS.md` itself calls
unearned. Six seats carry them:

| seat | held | arena statuses | flagged? |
|---|---|---|---|
| Fast/slow coupling | BY DECREE | `DP.02` NOT_RUN | **UNCONTESTED** ✓ |
| Language model | BY DECREE | `LG.00` NOT_RUN | **UNCONTESTED** ✓ |
| Language acquisition | BY DECREE | `LG.00` NOT_RUN | **UNCONTESTED** ✓ |
| PLASTIC-ONLY decree | BY DECREE | `PL.00`/`PL.02`/`PL.*` absent | ARENA-MISSING ✓ |
| **Learning core** | **BY DEFAULT** | `LC.00` PASS · `LC.01` PASS · `LC.02` PASS · `LC.03` **VOID** · `LC.04` NOT_RUN · `LC.05` NOT_RUN · `LC.06` NOT_RUN · `PS.01` PASS | **`ok`** |
| **Vision encoder** | **BY DEFAULT** | `T2.03` PASS · `T3.01` PASS · `PL.02` absent | ARENA-MISSING only |

**Every single-arena seat is caught. Neither multi-arena seat is.** The
discriminator is not whether the seat has been defended — it is how long the
arena list is.

**What the four passing `Learning core` arenas actually are.** None of them
races a core against another core:

- `LC.00` — *"The learning-core question is decidable in a gridworld first"* — a
  decidability precondition.
- `LC.01` — declared `COVERS: one brain / unison (rule)` — the **admission
  rule**. `coverage.py`'s own scar text says it in as many words: *"LC.01
  passing proved the ADMISSION RULE excludes unbound cores, not that any brain
  binds."*
- `LC.02` — *"A core that cannot live a life at survivable wall-clock is not a
  candidate"* — a feasibility gate.
- `PS.01` — declared `COVERS: hunger/thirst (fixture)`. A fixture.

The three specs that could actually move the seat are `LC.03` (**VOID** — and
`SYSTEM.md` is explicit that VOID is *"not a confident wrong answer"*, i.e. not a
decision), `LC.04` (NOT_RUN — the file's own cell calls it *"the seat's actual
match"*) and `LC.05` (NOT_RUN). So the project's single most consequential
architectural seat is held by an incumbent the same file describes as
**"DEFAULT, never defended — and now measured a NON-LEARNER in W0"**, and the
organ built on 2026-08-24 to make exactly that impossible reports it `ok`.

`Vision encoder` is the same shape: `T2.03` is declared `COVERS: sight
(fixture)` (*pretrained beats random* — a fixture about features, not an encoder
race), and `T3.01` is *"Ablate vision"* — an ablation of the **incumbent**,
which by construction cannot seat a challenger. The one cell that would race
encoders, `PL.02`, does not exist. The seat is flagged for the phantom and not
for the fact that no encoder has ever been raced against the one in the chair.

**And the tool's fixture certifies this as correct behaviour.** `_fixture()`
(`champions.py:337-381`) builds a seat whose `held` cell reads, verbatim,
`**DEFAULT, never defended**`, gives it `arena OK.01–OK.02` with
`ran = {"OK.01": "PASS", "OK.02": "PASS"}`, names it **`Healthy default seat`**,
and asserts:

```python
for ok in ("Healthy verdict seat", "Healthy default seat",
           "Vacant by default words"):
    assert ok not in flagged, (ok, flagged)
```

The known-positive battery — the thing that exists so *"a scanner nobody has
watched catch something"* cannot ship — has the false negative written into it as
a must-pass. That is why 42 audits have run this tool and none has seen this: it
is not a latent bug, it is a tested-in behaviour.

**Damage.** `champions.py` is the third of the three checks this audit opens
with, and its job is one sentence of `SYSTEM.md`: *"No architectural seat may be
held without a REGISTERED, EXISTING challenger."* It enforces the *existing*
half and does not enforce the *challenger* half at all. The 2026-08-24 finding
that *"one shared brain was a premise of the ladder, never an outcome of it"*
remains true today — `UB.10`, the six-arm fusion bakeoff carrying the A5
non-trunk arm that `SYSTEM.md` names as the credible challenger, is NOT_RUN and
PARKED — and the instrument commissioned to keep that visible now prints a green
ratchet over it.

---

## RANK 2 — `BY VERDICT` is never checked against whether its arena finished

`Episodic retrieval` is held **BY VERDICT** — the strongest marking in the file.
Its arena is `ME.11.A–F`. Ledger:

    ME.11.A  PASS     "Arm A — lexical containment, the incumbent, as the null"
    ME.11.B  NOT_RUN  BM25S with stemming, real lexical SOTA
    ME.11.C  NOT_RUN  static embeddings (potion-base-8M)
    ME.11.D  NOT_RUN  a real sentence encoder (all-MiniLM-L6-v2)
    ME.11.E  NOT_RUN  weighted hybrid
    ME.11.F  NOT_RUN  cascade

**One of six arms has run, and it is the seat-holder's own arm.** The champion
cell names the incumbent as *"lexical containment"*; `ME.11.A` is *"lexical
containment, the incumbent, as the null"*. The challenger cell even records the
expected winner — *"potion-8M favourite"* — and that arm has never been built.

`BY VERDICT` is outside `HELD_UNEARNED`, so no rule in the tool looks at it at
all. The result is that the marking which asserts the most evidence is the one
subject to the least checking, and a seat can be marked `BY VERDICT` on a
bakeoff in which nothing was raced. This is the RANK-1 quantifier defect in its
purest form: the incumbent's own arm satisfying the arena's run-check.

Same class, listed for completeness: `Sensory fusion` (held `UNDECIDED`) reads
`ok` on `UB.9` PASS — a capability claim, not an organisation race — while
`UB.10` (the race) is NOT_RUN and `T2.02` is VOID. `Taste (gustation)` reads
`ok` on `UB.9` too, which contests nothing about taste.

---

## RANK 3 — the builder, standing (nothing new in kind; the numbers are worse)

78 consecutive gated slots, **78.4 hours** with no iteration. This is the 34th
through 42nd audits' finding and I am not re-litigating the mechanism; three new
measurements belong on the record:

1. **The last PASS of a Tier ≥ 2 spec is `T3.01`, `2026-08-21T01:28:42` — 7.7
   days ago.** The four PASSes since are `NE.00` (reward algebra), `T0.17`,
   `T0.27` and `T0.21` — harness and rule specs. The `84` headline has been
   truthful and it has been carried entirely by the instruments for a week.
2. **`SM.03`'s 32 KB implementation has now been untracked for 3.3 days** and is
   the only runnable claim spec for the `smell` commitment. `run status` lists it
   as implemented (it reads the filesystem), so every instrument that ranks work
   believes it exists while git does not hold a byte of it. One `git clean` or
   one checkout ends it. The 36th audit's finding — the GPU push guard reads
   `--untracked-files=no` — means it is also undispatchable.
3. **`experiments/tests/__pycache__/sm_03_nose_reports_occluded.cpython-39.pyc`
   is the only other artefact from that iteration.** `/data/sm03_pilot_seed90.json`
   does not exist; `pid 1552865` is gone. I confirmed both. The `rc=0`-on-a-
   promise lesson is already written (`LESSONS.md`); what is not written down is
   that its cost is still accruing.

---

## RANK 4 — 29.6889 free Kaggle hours expire in 29.4 hours, with no dispatchable spec

W34 has spent `0.3111 h` of 30 (one job, `T2.15`, `2026-08-25T04:21`). The reset
is Sunday. The only registered `GPU_SHORT` claim candidate is `SM.03`, which is
untracked and therefore refused by `dispatch.sh`'s push guard. The builder cannot
wake to fix that before the meter reset on `2026-08-31`, which is 29 hours
*after* the hours expire — the two-clocks phase problem already in `LESSONS.md`,
now observed passing for the second consecutive week.

---

## The audit, section by section

**1. Integrity of the ledger — clean, and I checked all 84.** Every PASS's
`commit` resolves in git (84/84). Every PASS has a spec in `BY_ID` (84/84).
**82 of 84 declare a control and carry recorded `control_metrics`**; the two that
do not are `T0.01` and `T0.10`, both Tier-0 harness specs where a control is not
meaningful. `run status` reports **1 stale-by-content claim (`T2.02`) and it is a
VOID, not a PASS**; 27 pre-`impl_sha` entries verified byte-identical by git, 0
unanswerable. **No findings.**

**2. Thresholds and controls over the last 7 days — no findings, and this is a
real result.** Sixteen commits touched `registry*.py` or `experiments/tests/` in
the window. Every numeric change I found moves in the tightening or neutral
direction and cites a measurement in its own message: `20b8660` *added* a control
declaration after `UndeclaredControl` refused a dispatch; `7951f45` made the
coverage ratchet go red where an audit had computed by hand; `b624d78` generalised
`T0.21` P6 to compute its deletion set rather than cache a pair (semantics
unchanged, stated and true); `f5d8f1c` and `78699b9` are FAIL harvests that record
the failure rather than move the bar. **Not one loosening.** The one calibration
in the window (`ddbe6b7`, `DELTA_T_NIGHT 12→10`) is older than 7 days and was
pre-declared with its sweep table shipped in metrics.

**3. Drift from the goal.** The builder did nothing in the last day, so there is
nothing to test for drift; the last three days of commits are five audits and
three Reviews — the auditing organs describing an outage they help cause. On the
converse and harder question, `coverage` names **14 commitments with live claim
specs and nothing passing**, and the ones GOAL.md leans on hardest are among
them: **`curiosity`** (12 specs, 1 pass — and that pass, `T2.08`, is the only
claim; `PG.4` is correctly demoted to `fixture`), **`one brain / unison`** (21
specs, 1 claim passing — `UB.9`; `LC.01` correctly demoted to `rule`), **
`fast/slow`** (8 specs, **0** passing), **`sleep`** (4, 0), **`plasticity`**
(2, 0), **`proprioception`** (2, 0). RANK 1 and RANK 2 are why this matters more
than it reads: the seats that decide *how* those commitments get built are marked
contested and are not.

**4. Builder alive and productive.** Iterations in the last 24 h: **0**. `rc=0`
in the last 24 h: **0**. PASS delta over 24 h: **0**. Over 7 days: **+2**, both
Tier 0. The loop is not crashed — it is running its gate correctly every hour and
the gate is skipping every hour. See RANK 3.

**5. Compute honesty — no waste found.** Every charged job in `gpu_budget.json`
maps to a spec attempt in `gpu_submissions.jsonl`. W34's only charge (0.3111 h,
`jack-ladder-1787631708`, `T2.15`) produced a real ledger FAIL row that was
harvested and written up in `f5d8f1c`. The `2026-W32:kaggle` 6.3849 h opening
balance remains honestly labelled unattributable and is not lowered. The waste
here is **unspent, not misspent**: 29.69 h about to expire (RANK 4).

**6. Stuck decisions — nothing to act on, and I looked for the D1 shape.**
`decisions --check` reports 11 armed entries, 0 undeclared, 0 `MEANS-ESCALATED`,
0 overdue. No entry is blocked on the owner that a bakeoff could settle today,
because the two that could (`D10`, `D4`) are gated on GPU runs the builder cannot
launch. I found no owner-decision quietly acted on without record. The 41st
audit's finding — that four of the armed defaults pick actions outside the
already-permitted set — stands and is not mine to re-file.

**7. Bakeoff hygiene — one finding, and it is RANK 2.** `DECISIONS_RESOLVED.md`
holds three entries (`PS.01/J` VOID, `PS.01/J2` WINNER `impact_speed` at 10.32σ
with both controls failing, `D2` resolved by ledger replay). All three are
clean: the VOID is recorded as a VOID and not read as a verdict, and the winner
clears its null by more than 3σ with a cost column. **The hygiene problem is not
in that file** — it is that `CHAMPIONS.md` records a `BY VERDICT` seat
(`Episodic retrieval`) whose arena never ran, and no instrument joins a marking
to its arena's completion.

**8. The honest summary.** No. We are not closer to a curious humanoid than we
were yesterday, and today we are further than we thought: the instrument that was
supposed to guarantee the architecture could still lose has been reporting a
green ratchet over the two seats its own document marks *never defended*, and one
seat marked `BY VERDICT` on a six-arm bakeoff in which only the incumbent's arm
ran. That is worse than the outage, because the outage is visible in every log
line and this was invisible in a tool printing `ratchet ok`. We are not even
closer to a longer list of green ticks — the list has not moved in a week. What
we do have is one more true thing about the machine, which `SYSTEM.md` says is
the whole job when no spec passes.

---

## FOR THE BUILDER

**B1 (highest priority, ~30 lines, CPU-only, no GPU, no meter risk beyond the
iteration itself). Fix the quantifier in `experiments/champions.py` and fix the
fixture that certifies it.**

- Split `arena_status` into the seat's **challengers** and the rest. A minimal,
  defensible rule that needs no new declaration syntax: an arena spec counts as a
  *challenger run* only if its ledger status is `PASS` or `FAIL` (a **VOID is not
  a verdict** — `SYSTEM.md`: *"fix the arm, do not decide"*) **and** its registry
  `COVERS:` kind is not `fixture`, `rule` or `sensor` (`coverage.py` already
  parses this; import `DECLARATION`/`_KIND` rather than re-implementing).
- Change the condition from `all(v == "NOT_RUN" …)` to `not challenger_runs`, and
  widen `HELD_UNEARNED` handling so `BY VERDICT` and `BY ANALYSIS` seats are
  checked too — under a distinct flag if you prefer, e.g.
  `VERDICT-WITHOUT-ARENA`, so the existing ratchet baseline is not disturbed.
- **Fix `_fixture()` in the same commit.** Rename `Healthy default seat` to what
  it is and assert it **is** flagged; add a genuinely healthy default seat whose
  passing arena is a challenger, and add a `BY VERDICT` seat whose only run arm
  is the incumbent's, asserting it flags. A known-positive battery that asserts
  the false negative is the defect, not a side effect of it.
- Expect this to newly flag `Learning core`, `Vision encoder`, `Episodic
  retrieval`, and probably `Sensory fusion` and `Taste`. **Add a second
  shrink-only baseline constant for the new flag** (the `BASELINE_ARENA_MISSING`
  precedent) so `--check` does not go red everywhere on day one and get ignored —
  and set it from the measured count, in the same commit, with the count in the
  message.
- Do **not** repair this by editing `CHAMPIONS.md` markings. Downgrading
  `Episodic retrieval` from `BY VERDICT` to `BY DEFAULT` would silence the flag
  without running an arm — the deleting-the-arena-reference mistake in a new
  costume.

**B2 (cheap, and it is losing value every hour). Commit
`experiments/tests/sm_03_nose_reports_occluded.py`.** It has been untracked for
3.3 days, it is the only runnable claim spec for `smell`, and while untracked it
is invisible to the GPU push guard and one `git clean` from gone. Commit it with
its pilot state stated honestly in the message — *implementation only, pilot
never completed, gates not frozen* — and do **not** dispatch it on unfrozen
gates. `f0cb81d` registered the spec; the implementation belongs in git next to
it whatever its pilot did.

**B3 (carried from the 30th audit, still owed). `harvest_bookkeeping` /
`ladder_loop.sh`: before writing `iteration end rc=0`, verify that any background
work the iteration claims is live has (a) a live pid and (b) a non-empty declared
artefact, and log a distinct nonzero outcome naming the orphan if not.** The rule
is already in `LESSONS.md`; nothing implements it. `SM.03` is its second victim
in one day.

---

## FOR THE OWNER

Nothing new is escalated to you this audit, and `decisions --check` is green —
0 of 10 undeclared, nothing overdue. Two things are worth your eyes:

1. **A seat marked `BY VERDICT` in `docs/CHAMPIONS.md` did not have a bakeoff.**
   `Episodic retrieval` (Jack's memory search — how he finds what he remembers)
   is recorded as decided; five of its six candidate arms have never been built,
   and the challenger cell names the expected winner. That is a document-level
   claim of evidence that the ledger does not support. B1 makes it detectable;
   only running `ME.11.B–F` makes it true or false.

2. **`D14`'s deadline is 3 days out and the 42nd audit's evidence against it
   stands.** I re-read it and add nothing: the meter its default is keyed to rose
   34 points during 72 hours in which this box made zero requests on that model.
   If the default fires unamended on 2026-08-31 it installs a pre-flight refusal
   on a line the builder does not control. Your call, and the entry has the
   evidence attached.
