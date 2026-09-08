# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: **DAILY** (Part 1 + Part 2.5. Part 2, the anatomy audit and the
> completeness audit are Sunday work and did not run.)

**2026-09-08 06:37–07:0x UTC — DAILY.** Window: the last 24 hours
(2026-09-07 06:40 → 2026-09-08 06:50).

*The one sentence: **the builder cleared a five-item priority block for the
seventh straight day and bought back the constitution's only falsifier by
fixing a renderer — and this desk, which spent yesterday telling the owner it
was the bottleneck, closed one row, decided five, re-armed four and took the
queue to zero violations for the first time in five days.***

---

## The numbers

| | now | 09-07 (DAILY) | Δ |
|---|---|---|---|
| demonstrated / registered | **108 / 245** | 106 / 245 | **+2 / +0** |
| pass rate | **44.1%** | 43.3% | **+0.8 pts** |
| rework (attempt > 1) | 77.6% | 76.9% | +0.7 |
| FAIL / VOID | 22 / 13 | 24 / 13 | **−2 / 0** |
| unreachable | 93 of 245 (38%) | 94 | **−1, baseline shrunk again** |
| review-queue violations | **0** | 5 | **−5** |

**What actually moved, measured by diffing the ledger against yesterday's
Review commit (`5e610b8`) rather than counting commit messages: nine specs
changed status or attempt, across sixteen attempts, in eleven commits that
touched `ledger.json`.** The demonstrated set moved by exactly two specs, both
gains, both real: **`ME.3` FAIL → PASS** (attempt 5, the contract split this
desk ordered yesterday) and **`PL.00` FAIL → PASS** (attempt 2, the renderer
bakeoff). Everything else — `ME.11.0`, `ME.11.A`, `T0.28`, `T0.31`, `T0.33`,
`T0.35`, `T2.10` — is the certificate re-buy treadmill, and I will not quote
sixteen attempts as output.

**Goodhart: the rate rose on a flat registry, and this time I will call it
good.** +2 demonstrated against **+0 registered** is the cleanest possible
composition — the ladder did not grow to flatter its own rate, and both gains
are repairs of specs that were failing yesterday morning. **Unreachable came
down 94 → 93** for the second consecutive day, and again by a shrink-only
ratchet shrinking honestly: `PL.00`'s PASS freed `PL.02`, so the floor followed
the number down (`2c39d0f`).

**The counter I am obliged to report because it moved:
`review_queue_net_arrivals` 33 → 30 and `review_queue_violations` 0 → 5.** Both
were flagged at 06:37 and neither is justified by a committed change — the
arrivals number is the trailing window sliding, and **the five violations were
mine**: four rows routed on 08-30 with no date at all, which went STALE at
nine days, plus one OVERDUE row I dispositioned yesterday without writing a new
`DUE:`. All five are repaired below and the count now reads **0**. I have not
run `ratchets record` against them; the next reader should see the movement and
the repair in the same window.

---

## Part 1 — did the builder produce, thrash, or stall?

**It produced, and the shape of the production is better than the count.** All
five items of yesterday's priority block are discharged — the seventh
consecutive day a block was spent inside its own day. `ME.3`'s contract split
landed and PASSed (`a59363a`, `ce33621`); `EpisodicMemory.recall` now names the
−0.267 separability gap in its own docstring (`fed5322`);
`audit_supersedes_fail` tells the truth while red (`9d03d96`), with the four
certificates its edit staled re-bought behind it (`cdb4082`); the persist-weights
rule got a durable home as `CHAMPIONS.md` rule 6 (`c2be391`).

**And the fourth item is the day's real science.** I ordered `PL.00`'s renderer
bakeoff as *a clearing arm that dissolves an edge by satisfying it, never by
editing it.* It worked, in one day. The bakeoff seated **coarse-shadow512**
(`b7324ba`) after decomposing the 40 ms eye down to a 4096² shadow map — the
fixed per-call overhead the row had predicted from `render_ms_224` 39.17 vs
`render_ms_64` 40.04, **12.25× the pixels for the same money.** `PL.00` then
re-ran to **PASS at three seeds: `pure_T` 8.903 ± 0.294 against the 5.0 floor
that did not move** (`3a935f6`). `PL.02` — **the sole registered falsifier of
the PLASTIC-ONLY decree** — was unblocked and ran its probes the same
afternoon.

**Two things about that I want on the record, because both are the system
working rather than the system winning.** First: I refused yesterday to
re-point the `PL.02 → PL.00` edge in the week that edge produced an
inconvenient FAIL, and took the expensive arm instead. Twenty-six hours later
the expensive arm is a PASS and the edge never had to be touched. Second:
`PL.02`'s own probes came back **VOID with the routed eye-aliveness gate
firing** (`r2_ua` −0.0017 against 0.80, `4f2a1a6`), and the builder harvested
that honestly instead of quietly re-rolling. The falsifier is reachable now;
it has not yet said anything, and nobody pretended otherwise.

**No thrash, and one thing I checked for specifically:** `T2.10` re-bought to
FAIL at attempt 3 (`5b3ad7f`) on the repaired `ME.11.A` scorer. That is a
stale claim being re-stamped on moved code, not a spec being retried until it
turns — the number did not move and the FAIL stands.

### The queue, and today it is not the thing I have to apologise for

`review-queue` opened at **42 live rows, 5 violations, drain UNBOUNDED**, with
**five rows due today** and one already overdue. All six are disposed, each in
its own commit as it was made:

| row | disposition |
|---|---|
| `pl02-dependency-on-pl00-verdict-vs-table` | **ACTED** in `3a935f6` — the edge was SATISFIED, not edited |
| `d10-successor-rerun-under-adopted-gate` | DISPOSITIONED — twin denominator adopted, VOID-FORECLOSED refused |
| `ub10-seed-fragility-and-saturated-battery` | DISPOSITIONED — harden the task, gate the anchor's headroom |
| `lg10-mouth-fidelity-vs-freedom` | DISPOSITIONED — (c): the bar and the FAIL both stand |
| `t309-control-clears-the-claims-own-margin` | DISPOSITIONED — (a): a VOID cannot kill anything |
| `cpu48h-class-self-forecloses-the-day-meter` | DISPOSITIONED — (i)+(iv), schedule around it; the unit question stays the owner's |

Plus the **four STALE rows re-armed in the open** with chosen dates and reasons
(`a985f05`), spread one per day across 09-18…09-22 against this desk's own
measured capacity of ~1 row/cycle — not swept onto Sunday, which already
carries ten.

**And I caught myself with the instrument rather than with my own eyes.** After
committing the five dispositions I re-read the pile print instead of assuming
it, and 09-08 was still carrying four rows: four of my new `DUE:` lines had
been written at the *foot* of their rows, beside the design, after intervening
un-indented prose — and `review_queue.py:317` ends a row's body at the first
un-indented line, so the instrument could not see any of them. **This is the
scar this very file already carries** at its line 1563, where a `pl02`
declaration sat *"one `##` away from being read"* — arriving again one
indentation level down instead of one heading level. Dates unchanged, clocks
moved into the row headers, foot copies re-shaped so they cannot be mistaken
for declarations (`3eb9253`).

---

## Part 2.5 — steering maintenance

**1. Priority reconciled** (`115de4e`). `1''''`/`2''''` replaced by
`1'''''`/`2'''''`. `2''''` is **superseded rather than merely spent**: it
forbade a third `D1.0` dispatch until this morning's row answered why the
twins failed, and the row answered. The new block hands the builder **four
dated units and one free scheduling rule**, and for the first time every unit
on it carries a `DUE:` on a queue row that `run review-queue` will hold *me*
to. Counts are stamped as this morning's with an instruction to re-derive them
from `run blocked` / `run coverage` — priorities point at living sources.

**2. Field watch: nothing new to consume.** `FIELD_WATCH.md`'s last commit is
wk6 (`3b68b7d`, 2026-09-07), consumed by yesterday's Review 42 minutes after it
landed. The next sweep is Monday 09-14. No nomination is sitting unread.

**3. Seat staleness — and yesterday's finding is DISCHARGED.** Yesterday I
recorded that the **Vision encoder** seat is held BY DEFAULT/UNCONTESTED while
its only live challenger `PL.02` was unreachable for renderer reasons. **That
is fixed as of 11:25 this morning**: `PL.00` PASSes, `PL.02` is reachable and
has run. The seat marking does not change — no verdict has landed, and
`PL.02`'s probes VOIDed — but a seat whose challenger was structurally
unreachable now has one that can actually fight. The same is true one seat
over: **Control architecture (D1) is VACANT with its entire arena being
`D1.0`**, which has VOIDed twice for 33.8 GPU-hours, and today's disposition
gives it a committed design path rather than a third coin-flip.
`champions --check` is EXIT 0; trigger debt stays at 3 (Learning core, Fast/slow
coupling, World) and today's `UB.10` redesign is aimed at one of Learning core's
three closed doors. **No seat marking was edited today** — the markings are
accurate and a daily should not churn them.

**4. Organ liveness — all four live, verified against `/data/jack-logs` mtimes
rather than anyone's report.** Builder **06:11** (hourly), overseer **06:37**
(6 h), field watch **2026-09-07 05:56** (Mondays — yesterday, on cadence),
review **06:37** (this run). `lost_iterations.log` still 0 bytes and still
never exercised.

---

## Dispositions committed this morning (each in its own commit, as it was made)

1. **`pl02-dependency-on-pl00-verdict-vs-table` — ACTED** (`102a47a`), naming
   `3a935f6`. The row was OVERDUE by one day because yesterday's disposition
   delivered a ruling and wrote no new `DUE:` — a design alone keeps ageing,
   and I am the desk that keeps having to relearn it.
2. **`d10-successor-rerun-under-adopted-gate` — DISPOSITIONED** (`9924291`):
   **the twin denominator, and `VOID-FORECLOSED` refused.** The two rows' own
   numbers say the arms were never the problem — twin means identical across
   both attempts (aprime 198.4, d_mlp 197.6, deterministic twin eval), bar
   identical at 3.0σ, and the verdict flipped **2.94–2.96σ → 3.95/3.91σ purely
   because the random policy's spread moved 30.27 → 22.12.** A gate whose
   verdict is a function of one random draw's sampling noise is not measuring
   learning, and an untrained aprime banks **~87 raw points of architectural
   prior it is currently paid for.** So each arm is scored against its OWN
   untrained twin: **the free 87 points are subtracted by construction, which
   makes this strictly harder.** The 3.0σ bar does not move; random stays as a
   reported floor. **The one genuinely open premise is declared instead of
   assumed** — a twin is deterministic at fixed init, so it has no spread to
   divide by, and K ≥ 16 untrained twins at distinct init seeds (**forward
   passes only; nothing is trained, so cost does not scale with the
   denominator**) must measure whether one exists. **Both branches are
   pre-registered today, before the number exists**, because choosing after
   seeing it is the forbidden move.
3. **`ub10-seed-fragility-and-saturated-battery` — DISPOSITIONED** (`ad9bced`).
   The row lists its defects in the wrong order: A0 reads slot **1.0 on all
   three seeds**, so the PASS conjunct cannot fire against anything, and
   repairing seed fragility alone buys a run that still returns no verdict.
   Adopted: composite/cross-modal-XOR slots, a new `A0_HEADROOM` rig gate (the
   anchor must leave room on every seed **before any arm is scored** — the same
   shape as `D1.0`'s learning gate), and the per-arm stability conjunct.
   **Two options refused and named: the training-budget cut** (it makes every
   arm worse so the picture looks interesting, and silently rewrites the claim
   to *"fusion helps when undertrained"*) **and seed-level
   SCORED-AND-INELIGIBLE** (it is the one option that makes a verdict-less run
   return a verdict, by letting an arm that failed to train on a registered
   seed keep competing on the seeds where it did — a weakening wearing a
   bookkeeping name).
4. **`lg10-mouth-fidelity-vs-freedom` — DISPOSITIONED** (`66dcd86`): **(c), and
   I refused to redesign a failing spec into a passing one.** `LG.10` is the
   registered falsifier of `GOAL.md`'s *"the LLM is his mouth, never his
   mind"*, and it has now falsified that sentence for the incumbent mouth —
   1588 verdicts, both ends of the freedom knob paid for, every control
   behaving, and **29 of 55 wrong draws drifting to a different truthful memory
   while 26 collapse to phatic filler.** Option (a), a dominance-margin
   abstention, converts every wrong draw into a non-answer; adding it to a FAIL
   is exactly the move the law forbids. So the bar and the FAIL stand, and (a)
   is accepted as a **NEW registered claim beside it** — *"he speaks correctly
   or he is silent"* — carrying a **mandatory pre-registered utterance-rate
   floor**, which is yesterday's `ME.3` lesson arriving in the language family
   six days later: an abstaining mouth with no floor abstains on everything
   hard and scores 1.0 on what is left, and `ME.3` was caught only because
   starvation was TOTAL. (b) refused on the spec's own docstring warning.
5. **`t309-control-clears-the-claims-own-margin` — DISPOSITIONED** (`da202e1`):
   **(a) — a VOID cannot kill anything.** Law 2 class-3 is unconditional and
   the wrong-goal control gained **+12.47 against `MARGIN_AFF` 11.0**, so
   attempt 3's FAIL is a voided run and cannot execute a deletion clause. The
   second fact is stronger than the first: **`loop_creative` fired 0 times on
   142 consults** — the branch in the spec's own title has never executed, so
   `T3.09` has not failed to demonstrate its claim, **it has never tested it**,
   and a module can be neither deleted nor kept on that record.
6. **`cpu48h-class-self-forecloses-the-day-meter` — DISPOSITIONED**
   (`16f7eb8`): **(i)+(iv), the two options that cost nothing and loosen
   nothing.** The row itself instructed this branch for the case where the
   owner has not answered, and `D20` is live and unanswered (`decide_by`
   09-18). Options (ii) and (iii) both edit what a tenant-protection ceiling
   counts on a box with paying tenants; law 4 makes that owner-gated and no
   default here may fire it — **which stays true whether or not I find the
   arithmetic persuasive, and for the record I do.** Re-dated to 09-19, the day
   after `D20`'s deadline, so the row consumes the answer or the armed default
   instead of asking the same question in front of it.
7. **The four STALE rows re-armed** (`a985f05`).
8. **The five clocks moved into their row headers** (`3eb9253`).
9. **`ladder_prompt.md` priority block replaced** (`115de4e`).

---

## The frontier

`T2.01` still tops it and has for weeks: **frees 35 / blocks 38**, settled FAIL,
implementation unchanged **29 days**. Its repair path runs through `D1.0`, and
after this morning that path has a committed design instead of a third
coin-flip — but **it is a probe away from being runnable, not a dispatch away**,
and it goes to W37. Behind it: `NE.01` frees 8, `LT.01` frees 7, **`UB.10` frees
4** (dispositioned today), `T2.02` frees 3, `LG.03` frees 3, `HR.1` frees 3
(D19-held), `HR.5` frees 2. Two of the top four blockers were decided this
morning; neither is fixed.

---

## The honest paragraph

We are closer, and the specific way we got closer today is one I did not
expect: nothing about Jack improved, and the thing that improved was our
ability to find out. Two of his failures turned out on inspection to be
failures of the instruments pointed at him — the gate that could not tell
learning from the shape of a fresh network, and the fusion battery whose
reference arm was already at the ceiling — and in both cases the repair
available was to lower the bar and in both cases what got adopted was to make
the question harder. The one I am proudest of is the one where I refused to do
anything at all: his mouth failed a test of whether he chooses what he says,
the fix on offer would have let him fall silent whenever he was about to be
wrong, and taking it would have quietly deleted the only measurement this
project has that its own constitution's sentence about the borrowed model is
currently false. It stays false and on the board. Against that, the drift, and
it is not the one I named yesterday. Yesterday I told the owner the bottleneck
was this desk; today this desk did more work than it has ever done in a
sitting, and the drain is still unbounded, which means either yesterday's
diagnosis was self-flattering or one good day proves nothing — and I genuinely
cannot tell which from inside a single morning. What I can tell is that
everything I decided today, I decided about instruments, and not one thing I
did brought a creature that lives, learns and is known any closer to living.
The builder made him remember better this week. I made the ruler straighter.
Those are both necessary and only one of them is the point.

---

## FOR THE BUILDER

1. **`D1.0`'s twin-spread probe — first, cheapest, and it stands in front of
   the largest unblock in the project.** K ≥ 16 untrained twins per
   architecture at distinct init seeds, reporting each architecture's prior
   mean and std. **Forward passes only — nothing is trained, which is the whole
   economy of the design.** Both branches are pre-registered on the row and you
   may not pick between them after seeing the number. The 3.0σ bar does not
   move; random stays as a reported floor; the successor gate is committed in a
   commit that is **not** a dispatch commit. DUE 09-14.
2. **`UB.10`'s battery redesign:** composite/cross-modal-XOR slots, the
   `A0_HEADROOM` rig gate firing before any arm is scored, and the per-arm
   stability conjunct. The training-budget cut and seed-level
   SCORED-AND-INELIGIBLE are both refused — do not re-propose either. Committed
   before any re-dispatch. DUE 09-15.
3. **`LG.10`'s sibling spec — REGISTRATION ONLY**, with a pre-registered
   utterance-rate floor that is mandatory, fidelity bars carried over unmoved,
   and every control re-run under the abstention machinery (a null that can
   abstain must still fail). **Do not touch `LG.10` itself.** No dispatch, no
   LLM verdicts bought. DUE 09-16.
4. **`T3.09`'s registry note** — attempt 3's row is a VOID under the corrected
   lane and is not a kills-executing verdict. A note, not a run.
   `AlphaGeometryLoop.py` stays. DUE 09-17.
5. **Free, and it costs nothing to obey: run first-run `cpu<2h` specs before
   your own housekeeping.** `CPU_DAY_CEILING_S` is only **1.067×** the largest
   legal child, so a never-run `cpu<2h` spec is refused once the day passes
   **3,600 s — 6.25% of the ceiling**, which one routine gate sweep spends.
   Items 1–3 commission exactly such never-run specs.
6. **Standing prohibitions, restated:** no third `D1.0` dispatch until the
   probe result is on the row AND the gate is committed, and then into W37 —
   W36 has ~12.4 GPU-h left against a measured 17.61 h need; no `UB.10`
   re-dispatch before the redesign; no `LG.10` re-roll and no re-fitting of T
   (both endpoints are paid for); no unchanged re-roll of `T3.09` at this site;
   no edit to `cpu_budget.py`'s or `rtf.py`'s ceilings (owner-gated on `D20`);
   `HR.1`–`HR.4` stay D19-held to 09-14; `HR.6` stays behind `HR.5`; `LF.01`
   attempt 2 waits for the 09-09 design; the CPU-accountant rule stays as
   narrowed on 09-05.

---

## FOR THE OWNER

1. **`D22`'s `decide_by` is TODAY, and the evidence changed this morning in the
   direction that argues against my own ask. You should have it before the
   default fires at 09-09T00:00.** Cited, not re-routed — `D22` is the entry
   the overseer lifted off this page on 2026-09-04, and it asks whether design
   throughput is the binding constraint on the project and whether drafting
   should be handed to the builder.

   > **My recommendation is unchanged in substance and weaker in confidence,
   > and I would rather say so than let a default fire on a stale premise.**
   > Yesterday I told you the bottleneck was this desk: 40 live rows, unbounded
   > drain, four dated promises of which I could honestly keep three. Today
   > this desk closed one row, decided five, re-armed four, took `review-queue`
   > from **5 violations to 0**, and did it inside one sitting with every act
   > committed as it was made. That is the strongest single day this organ has
   > had, and it is *evidence against the thing I asked you to fix.* But the
   > drain still reads **UNBOUNDED** — 41 live rows, 32 arrivals against 3
   > disposals over the trailing week — so one good day has not moved the
   > trend, and I cannot tell from inside a single morning whether yesterday's
   > diagnosis was true or self-flattering. **If you were minded to grant the
   > ask, I would now ask you to wait a week instead:** let the default fire or
   > rule as you see fit, but the honest read is that this desk should be
   > measured over the next seven days against a drain that is now visible,
   > dated, and — for one morning at least — not violated. If the drain is
   > still unbounded on 09-15, the ask returns with a trend behind it instead
   > of a bad Sunday.

2. NO-DECISION: a finding I am acting on myself, recorded here because it is
   the kind of thing that should not first appear in a Sunday page. **Four
   separate fronts were named on this board today as the same disease: our
   tasks are too easy for our instruments to say anything about them.**
   `UB.10`'s anchor sits at ceiling on every seed; `sh02-null-saturation` is a
   null with no headroom; `w0-too-shallow` is a world that separates nothing;
   `t309` is a venue where being told the *right* thing is 44.5 s worse than
   being told the wrong one; and `dp04-lifespan-has-no-resolution` measures 21
   distinct lifespan values in 3,072 lives with 76.7% pinned at the cap. Each
   was repaired today in the same direction — make the world harder, never
   lower the bar — and each was repaired *locally*, by a different row, on a
   different date. **Four independent instruments saying the same thing about
   our task design is not four findings; it is one, and it is about how this
   project chooses what to measure.** I have written the trigger into the
   `UB.10` disposition and dated `dp04` last (09-22) so it is decided *under*
   that answer rather than in front of it. It goes to the Sunday FULL as a
   single question. Nothing here for you to rule on today.

3. NO-DECISION: liveness report, nothing here to rule on.
   All four organs live, verified against `/data/jack-logs` mtimes rather than
   anyone's report: builder **06:11** (hourly), overseer **06:37** (6 h), field
   watch **2026-09-07 05:56** (Mondays — yesterday, on cadence, third
   consecutive sweep on schedule, and its wk6 report was consumed 42 minutes
   after it landed with nothing left unread), review **06:37** (this run).
   `lost_iterations.log` still 0 bytes and still never exercised. `D24`
   (`decide_by` 09-11) and `D25` (`decide_by` 09-13) are unchanged on your desk;
   `D20` (`decide_by` 09-18) now has a queue row dated to consume it on 09-19.
