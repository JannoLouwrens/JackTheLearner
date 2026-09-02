# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: DAILY (Part 2, the anatomy audit and the completeness audit are Sunday
> work and were deliberately skipped; the last FULL page is 2026-08-31, and its
> findings below are tracked, not repeated).

**2026-09-02 06:3x–07:0x UTC — DAILY.** Window: the last 24 h
(2026-09-01 06:4x → 2026-09-02 06:3x).

*The one sentence: **the largest unblock in the project spent 16.17 GPU-hours
and came back VOID, this desk's "the board is empty" was refuted by the
overseer within eighteen hours, and the builder — acting on that refutation —
bought the owner's liar test, the first genuinely new capability this ladder
has recorded since 2026-08-20.***

---

## The numbers

| | now | 09-01 06:4x (DAILY) | Δ |
|---|---|---|---|
| demonstrated / registered | **94 / 217** | 93 / 213 | **+1 / +4** |
| pass rate | **43.3%** | 43.7% | −0.4 pts |
| FAIL / VOID (live rows) | **20 / 10** | 18 / 7 | +2 / +3 |
| unreachable specs | **85 / 217 (39%)** | 81 / 213 | +4, exactly at the ratcheted baseline |
| rework (ledger rows at attempt > 1) | 82 / 124 = **66.1%** | 64.7% | +1.4 |
| commits, last 24 h | **55** | 62 | −7 |
| builder slots fired | **25 starts, zero `PACING:` skips** | 24/24 | — |
| head-row settlements | **13** (6 PASS / 4 FAIL / 3 VOID) | 14 | — |
| ratchets | coverage **rc=2** (two reds) · review-queue rc=0 (27 rows) · champions rc=0 · decisions rc=0 | — | — |

**One correction to yesterday's table, stated once and not re-argued:** it
printed `coverage rc=0`. Coverage was already rc=2 at that moment — the builder
found it at 13:20 and said so in its own commit. Today it is rc=2 for **two**
reasons; the second is new and is mine (item 4 below).

**The number that matters: of six PASSes, exactly one is a new capability, and
it is the right one.** `LG.02` — *the owner's liar test, queued since
2026-08-09* — passed at attempt 1. Two advisors alternating at 0.9/0.1 claim
accuracy, every claim verified by Jack's own search, joined to that search
**only** through the attributed diary. Worst-seed last-quarter divergence
**0.689 ± 0.103** against a 0.40 gate; stripped-attribution null **0.028** with
the join proven alive; the owner's own swap control migrates 0.711/0.733; prior
measured at exactly 0.5 for both voices; attribution recall 0.95. That is
GOAL.md's *"his diary records whose advice proved true, so trust in a person can
be earned and checked"* — measured, with controls that had to fail and did.

**And it is the first first-ever claim PASS since `T3.01` on 2026-08-20.**
Thirteen days of re-buys, refutations and retirements, then one day where the
creature gained something. The other five PASSes are certificate re-buys
(`T0.12` att 22, `T0.17` att 22, `T0.21` att 22, `T0.31` att 4) or the two
GPU-harness re-aims (`T1.09`, `T1.10`, att 2, onto the P100).

**Goodhart check: the rate fell 0.4 points and the fall is arithmetic, not
decay.** The denominator grew by four (`GEN.02/03/06/09`) on a Review order and
the numerator grew by one. Count rose, rate fell, and the count is the honest
signal this week — but see item 4: those four registrations made an instrument
*worse*, so the +4 denominator did not buy what it was ordered to buy.

**Rework rose 1.4 points to 66.1% and is still not the problem** — same reading
as the last four pages: attempts 2+ are VOID→repair→re-run and certificate
re-buys after deliberate tooling edits. `T3.09` at attempt 3 is the textbook
case: two same-lane VOIDs, one preview-independent repair (`N_LIVES` 16→32,
every gate constant untouched, committed before the run), and then an honest
FAIL.

---

## The frontier — yesterday's good news reversed inside twelve hours

**`D1.0` settled VOID at 18:23:04.** The four-arm control-path bakeoff, three
kernels, **16.17 of this week's 19.2 charged GPU-hours**, was the thing
yesterday's page called *"the largest unblock on the board"* and *"the builder
is working on the largest mass for the first time in eleven days."* Its learning
gate fired on one arm (`c_e2e` at 2.56σ against a 3.0 bar) and one non-learner
voids the arbitration by the `T2.02` precedent — even though `d_mlp`, at 530K
parameters, beat the 57M-parameter arms by 5.99σ. **The arbitration did not
happen. `T2.01` (FAIL, frees 35, blocks 38) is still first on `run blocked` and
the architecture it would re-run under is still undecided.**

The overseer's 60th audit called the consequence correctly and before I did:
*"`D1.0` VOIDed and its 35-spec unblock has no owner."* It now has one row with
a clock (`d10-successor-rerun-under-adopted-gate`, DUE 2026-09-06) and that row
is on **this** desk.

**Transitive-block mass, recomputed live:** 85 of 217 unreachable (39%), exactly
at the ratcheted baseline the builder installed yesterday. `T2.01` frees 35 /
blocks 38. `NE.01` frees 8. `LT.01` frees 7. `UB.10` — un-parked and run
yesterday — is now **VOID**, so the four specs it was to free (`TA.03`, `UB.11`,
`UB.12`, `UB.13`) stayed welded, and with `UB.11` still dead the `T2.12`
fusion-boundary conjunct (Review 08-31 item 4) **still cannot be written.** That
item has now been blocked by its precondition for three consecutive pages.

### The frontier finding is about this desk, not the builder

Yesterday this page said, in bold: *"The rest of the board is empty, and the
emptiness is structural"* and *"the builder is no longer the bottleneck. This
organ is."*

**The first half was false, and the overseer proved it eighteen hours later.**
The 60th audit's finding 3: `coverage.py` named four specs implementable that
day — `LG.10`, `ME.11`, `LG.02`, `T3.09` — all deps PASS, all CPU. Two of them
are GOAL.md commitments in a family with 2 passing claims of 9. I read the same
tool and reported an empty board.

**The builder acted on the audit and built three of the four in six hours.** One
of them is the only capability this project gained this week. The one still
standing — `ME.11` — is the only fresh dispatch left anywhere on the board.

So the correction is not cosmetic. *"Empty board"* is the single most
consequential sentence this organ writes, because the builder is instructed to
believe it and not to go looking for cheaper work. I wrote it wrong, an organ
whose entire job is to assume something is quietly broken caught it, and the
cost of the error was bounded at eighteen hours by the machine working exactly
as designed. **The second half of the sentence stands: 22 of 27 queue rows are
OPEN and eleven are DUE 2026-09-06.** But it stands on its own evidence now, not
on a board reading I did not verify.

---

## Part 2.5 — steering maintenance

**1. `scripts/ladder_prompt.md` PRIORITY block: REPLACED — fourth consecutive
day, and again for the good reason.** Every item of yesterday's block was
executed: `UB.14`'s declaration landed, `UB.10` went park → disposition → grid
pilot → `SELECTED` → registered run → VOID in one day, item 5 put `T1.09` and
`T1.10` on the P100 (both PASS), item 6 registered the four `GEN` ids. The new
block: (i) orders the uncommitted `T3.09` harvest first and tells the builder
what its control result *means*; (ii) installs `ME.11` as the only fresh unit,
**explicitly as an honest RED being bought, with the family disposition reserved
to 09-06 so the builder does not pre-empt it with a new arm**; (iii) names the
`GEN` regression with a standing prohibition on closing it by widening
`GOAL_UNRUNNABLE_BASELINE`; (iv) states the GPU position (19.2 h of 30 charged,
~10.8 h expiring Sunday, nothing dispatchable) with the do-not-manufacture rule;
(v) carries the body paragraph, with its count removed and pointed at the live
row instead. No count or status is cached on the page.

**2. `docs/FIELD_WATCH.md`: unchanged since the last review** (last commit
`469bbaf`, 2026-08-31 05:53, wk5), and wk5 was consumed in full by the 08-31
DAILY. **Nothing to consume.** The field watch fires Mondays (`37 5 * * 1`);
last fire 2026-08-31 05:53, next due 2026-09-07. Alive and on schedule.

**3. Seat staleness — one seat moved, and it moved the wrong way.** The
**Control architecture** seat's arena `D1.0` **ran and VOIDed**, so the seat is
`VACANT` with its ring having opened and closed without a verdict — the second
seat in two days to do that (`Curiosity signal`, whose `LT.01` failed the day
its arena registered). `champions --check` reads **rc=0, 0 phantom arenas, 3/4
unfalsifiable, 3+1/6 uncontestable** — the ratchet is honest; the finding is not
a violation, it is that two of the project's live rings are now closed by
results rather than by neglect. **Vision encoder** and **Emotion** remain BY
DEFAULT and uncontested with real arenas and no progress — carried, not new.

**4. Organ liveness — all four alive, none silent past its cadence.**

| organ | cadence | last fire | verdict |
|---|---|---|---|
| builder | hourly (`7 * * * *`) | 2026-09-02 06:07 | **alive**, 25 starts in 24 h, 0 pace skips |
| overseer | 6-hourly (`37 */6 * * *`) | 2026-09-02 06:37 | **alive** (60th audit 00:37, DRIFTING) |
| field watch | Mondays (`37 5 * * 1`) | 2026-08-31 05:53 | **alive**, next 2026-09-07 |
| review | daily 06:37 + Sunday FULL | 2026-09-02 06:37 | **alive** — this run |

Background work verified rather than asserted: no GPU watcher is running (the
`D1.0`, `UB.10` and `LC.07` watchers all completed and their receipts are
committed), and the only live project process at the start of this run was the
overseer's own session. `experiments/ledger.json` is dirty with `T3.09`'s
attempt-3 row — **the runner's write, correctly not committed by me**; it is the
builder's first unit.

**5. One new review-queue row: `goal-cites-four-specs-that-resolve-to-corpses`,
DUE 2026-09-06** (item 4 below). Ratchet re-verified after the append: **27
routed, 0 violations, rc=0.**

---

## The honest paragraph

Closer — narrowly, and for the first time in a fortnight by the only route that
counts, which is that the creature can do something today it could not do
yesterday. He can tell who lies to him. Not by a rule we wrote him and not by a
number that flattered us: two voices spoke, one of them was usually wrong, he
checked both against what he found himself, and by the end he followed the
honest one and not the other — and when we stripped the names off the diary and
left everything else intact, the effect vanished, which is the only way we ever
believe anything here. That is a small thing next to a jungle and it is
precisely the kind of small thing the whole apparatus exists to make believable.
Set against it: the biggest thing we own spent the largest block of compute we
have on the question of how he is built, and came back saying nothing at all —
not a wrong answer, which we could use, but no answer, because one of four arms
failed to learn and the rules we wrote in advance said that voids the comparison.
We paid full price for silence, and the rules that made us pay are the right
rules. The single most important step toward Jack was `LG.02` passing, and it
matters more for how it arrived than for what it measured: the board's own
reader had been saying for a day that this spec was buildable, this desk said
the board was empty, an auditor called the contradiction, and the builder went
and built it. The machine caught its own chief scientist being wrong and the
creature got something out of it inside a day. The most concerning drift is that
I am now the organ that most needs auditing. Yesterday I misread the board.
Today I found that an order I gave on Sunday — register four ids so a citation
stops dangling — was carried out exactly and made the thing it was meant to fix
worse, because nobody, me least of all, checked whether the ids would land on
anything alive. Eleven rows come due on Sunday and every one of them is a
decision only this desk can take. The builder has been faster than its map for
four days running. The map is the problem, and I draw the map.

---

## REWRITTEN / STRENGTHENED

| spec / file | change | why it is stronger |
|---|---|---|
| `scripts/ladder_prompt.md` | priority head block **replaced**: yesterday's was executed in full within 24 h. New block orders the uncommitted `T3.09` harvest with its control arithmetic quoted, installs `ME.11` as an explicitly-RED buy with the family disposition reserved to 09-06, names the `GEN` regression with a standing prohibition on widening `GOAL_UNRUNNABLE_BASELINE`, states the GPU position with the do-not-manufacture rule, and strips the cached instrument count off the body paragraph | the map was again ordering completed work; and the previous block's "empty board" framing was measurably wrong — the new block names the one real fresh unit instead of describing an emptiness |
| `docs/REVIEW_QUEUE.md` | **new row `goal-cites-four-specs-that-resolve-to-corpses`**, DUE 2026-09-06, downstream of `lc07-checkpoint-branch`, with three options and an explicit ban on closing it by widening the shrink-only baseline | a Review order that made an instrument read worse had no owner and no clock; without a row it would have been discovered again by the next audit rather than repaired by the desk that caused it |
| `docs/PROGRESS_LOG.md` | one row appended | trend line continues |

**No threshold moved. No control softened. No FAILING or VOID spec was
rewritten. No spec file was touched by this run** — Part 2 is Sunday work and
was skipped deliberately.

---

## FOR THE BUILDER — ordered

1. **Harvest `T3.09` first — the row is on disk and uncommitted.** Attempt 3,
   `ran_at` 06:33:18, **FAIL**, `n_affected` 11 so the site-under-exercise VOID
   lane cleared and the spec finally measured what it was built to measure.
   Commit the runner's row as found. **Then route what it says**, because it is
   not a plain red: `creative_contribution` **−9.96** against the 11.0 margin
   (the loop arm is *worse* than the shipped random detour), `loop_creative`
   **0** on every life — the loop never once took its creative branch — and the
   **shuffled control gained +12.47, clearing the margin the claim missed.** A
   control carrying deliberately-wrong information beating the bar is an
   instrument statement: this venue may reward perturbation as such. Quote that
   arithmetic into its own queue row. Do not repair the arm and re-run —
   `AlphaGeometryLoop` earning its parameters is exactly what the spec was built
   to decide, and it decided.

2. **Then `ME.11` — the only fresh dispatch left on the entire board.**
   `cpu<10min`, deps `ME.1` and `ME.11.0` both PASS, no implementation.
   **Buy it knowing it is an honest RED**: `A` measured 0.0000 paraphrase
   recall, `B`/`C`/`D` FAIL, `E`/`F` VOID-FORECLOSED by arithmetic, and the best
   dense ceiling the family ever measured is 0.250 against the registry's 0.80
   bar. Implement the family verdict against the rows already on the ledger; the
   bars are the registry's and **do not move**; do not pre-empt the family
   redesign, which is owed by this desk on 09-06. What it buys is real — a
   commitment moves from *unmeasured* to *measured*, which no registration can
   ever do.

3. **Do not manufacture a GPU dispatch.** `2026-W35` has 19.2 h of 30 charged
   and ~10.8 h expiring Sunday, and every GPU cost class on `coverage` is either
   a VOID arm or pilot-blocked. The quota at an empty class is unspendable
   however awake the loop is; that rule has been on your page since 08-29 and it
   is more binding now, not less, because 16.17 h of this week's spend already
   bought a VOID.

4. **Do NOT close the `GEN` citation regression by widening
   `GOAL_UNRUNNABLE_BASELINE`.** The constant is shrink-only by construction and
   widening it is precisely the move it exists to forbid. Routed as
   `goal-cites-four-specs-that-resolve-to-corpses`, DUE 09-06, and it is the
   Review's to decide, not yours. Named here only so you do not "fix" it.

5. **Declare your detached runs.** The 05:07 slot ended
   `LEFTOVER=1 undeclared process` — pid 363738, 178 s CPU, your own `T3.09`
   run. A legitimately-detached run that nobody declared is indistinguishable
   from an abandoned one to the only instrument that looks for either. One line
   in `declared_pids` is the difference between a receipt and a smell.

6. **`T2.12`'s fusion-boundary conjunct — still blocked, third page running.**
   `UB.10` VOIDed, so `UB.11` is still dead and the conjunct still cannot be
   written. Unchanged in substance when it becomes writable: keep both existing
   controls, add that PAD separability must survive at the fusion boundary in a
   live `UB.11` ablation. Do not attempt it before `UB.10` has a successor.

---

## FOR THE OWNER — two forks

### 1. `D1.0` cost 16.17 GPU-hours to buy silence, and the rule that made it silence is one I recommend keeping

The four-arm control-path bakeoff was to decide the architecture under which
`T2.01` (frees 35) re-runs. Three kernels, 16.17 h, 54% of this week's Kaggle
quota. It returned **VOID** because one of four arms (`c_e2e`) cleared its
learning gate at 2.56σ against a 3.0 bar — and one non-learner voids the
arbitration by the `T2.02` precedent. Meanwhile `d_mlp`, at **530K parameters**,
beat the 57M-parameter arms by **5.99σ**. That is the 57M-vs-54K lesson
appearing for the second time, in a run we are not allowed to conclude from.

**My recommendation is to keep the rule and pay again.** An arbitration that
reports a winner while one arm demonstrably never learned is a confident wrong
answer, which is the exact failure this project was built against. But the
*cost* of the rule is now measured and it is high, so the fork worth your
attention is narrower: **should the learning gate be checked as a cheap
pre-flight rather than as a post-hoc voider?** A short screen that spends 5% of
the envelope confirming all four arms learn, before committing the other 95%,
would have converted 16 hours of silence into either a real arbitration or a
one-hour finding. That is a builder-implementable change and I will specify it
on 09-06 unless you would rather it wait. **No decision needed today** — I am
recording the price because a rule this expensive should be paid knowingly.

### 2. The Review is the organ most in need of a Review, and I have two days of evidence

I am reporting this against myself because nothing else in the system will.

- **Yesterday I wrote "the board is empty" in bold.** The overseer's 60th audit
  read the same tool eighteen hours later and named four specs implementable
  that day. The builder built three of them; one PASSED and is the only
  capability this project gained this week. "Empty board" is the sentence the
  builder is instructed to believe and not to work around, so it is the most
  consequential sentence this desk writes, and I wrote it without verifying it
  against the reader that produces it.
- **On Sunday I ordered four registrations to clear a dangling-citation red.**
  The builder executed the order exactly. Today the same instrument is red for a
  *new* and, by the 59th audit's own words, *worse* reason: four ids that
  resolve to corpses instead of to nothing. Nothing checked whether the ids
  would land on a live root. Nothing in this repo would have.

Both errors were caught — one by the overseer, one by me a day later — and
neither cost more than a day. That is the machine working. But the pattern is
the same in both: **this desk asserts a board state instead of reading one, and
the assertions are load-bearing.** The overseer audits the builder and the
ladder; nobody audits the Review, and I now have two instances in two days.

**My recommendation: add the Review's own last page to the overseer's standing
read.** One line in `scripts/overseer_prompt.md` — *check the previous
PROGRESS.md's board claims against the live tools* — costs the audit a few
minutes and closes the only oversight gap in the system that currently has
measured misses in it. I have not made this change: `overseer_prompt.md` is
another organ's instructions and changing it myself is exactly the
self-marking this finding is about.

### Carried, unchanged, from 2026-09-01 and 2026-08-31 (not re-argued here)

- **The body is parked behind a spec that failed because of the body.** `D9`
  parked the body question until `LT.08`; `D8` re-parented `BA.02` behind
  `LT.08`; `LT.08` sits behind `LT.01`, which failed on a clause about the body
  (`nonladder_rise_max` 0.084 ± 0.067 m against a 0.6 m bar). Recommendation
  unchanged: **register `W0.BAL` as a spec id and seat the body**, so the
  parking is visible as a parked seat rather than as silence. If you would
  rather not, tell me at 09-06 which of the three options on
  `lt01-c2-body-cannot-rise` the design should take.
- **The overseer and the Review both fire at 06:37 every day** (`37 */6` and
  `37 6`). They are running concurrently as I write this. No damage has been
  measured, but cron is outside every organ's mandate, so it can only be fixed
  by you — one minute's edit moves the Review to `07 7` and removes a standing
  race between two agents that both commit to this repo.
- **Re-tier `T6.03` out of Tier 6.** A save/load round-trip bought on day four
  should not be a green tick in the tier GOAL.md calls the finish line.
- **The builder's budget is the untested twin of the Review's.** Not a decision
  request.
- **The next FULL run is 2026-09-06 and eleven rows are due that day.** Week 3's
  rule binds me: a third deferral of the `w0-too-shallow` window would be a lie.
