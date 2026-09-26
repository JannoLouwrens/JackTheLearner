# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Findings are ranked by how much damage they do to the
> trustworthiness of the ledger, not by how interesting they are.

**2026-09-26 12:37–13:0x UTC — the 121st audit.** Six hours after the 120th
(06:37). The window is the builder's seven slots `06:07`–`12:07`, which produced
**fourteen commits and eleven ledger events**. Demonstrated **110/254**, down
one from the 120th's 111 — the `-1` is `T0.28`'s honest FAIL and is the best
thing in this report.

Instrument exit codes, re-derived this sitting and not inherited:
`coverage` **2**, `decisions --check` **1**, `champions --check` **0**,
`run review-queue` **2**, `run status` **0**.

---

## VERDICT: DRIFTING — no integrity risk, seven clean slots, five real units, and not one of the eleven ledger events in this window carries a verdict about Jack; the new finding is that the ledger's memory of WHICH files were uncommitted is erased on the next re-run, so 19 of the 20 dirty rows since the recorder shipped say "unrecorded" about a field they postdate

**Section 1 is clean and I re-derived every cell rather than taking it from any
organ.** All 110 PASS rows resolve their `commit` in git, all 110 declare a
`control` in the registry, all 110 resolve to an implementation module, and
**not one PASS row carries a `+dirty` stamp.** The two rows without
`control_metrics` (`T0.01`, `T0.10`) declare `NONE, BY DECISION` on their face.

**Section 2 found no loosening.** Eight files moved under `registry*.py` /
`experiments/tests/`. `BA.03`'s diff is **comment-only** — I diffed it stripped
of comments and not one numeric line changed, which is what the 120th audit's
option (b) asked for. `T2.11` gained a conjunct joined with `and`, never `or`.
No control was deleted, no seed count reduced, no assertion removed.

What is drifting is unchanged and I will not dress it up: **eleven ledger events
and eleven of eleven are the project buying back its own Tier-0 instruments.**

---

## FINDING 1 — the ledger forgets WHICH files were dirty on the next re-run: 19 of the 20 `+dirty` rows recorded since the recorder shipped carry `dirty_files: None`, the reader tells them they "predate" a field they postdate, and a classifier comment cites a value the record no longer holds (MEDIUM-HIGH — the first new finding in three audits, and it is a record-integrity defect, not a science one)

On **2026-09-13**, commit `8a97fd9` — whose own title is *"The DIRTY-stamp
instrument told a FALSEHOOD about its only live row"* — made `env_stamp` record
the dirty file LIST and not only the bit derived from it. Its stated reason
(`protocol.py:677-683`):

> *the list was computed here and discarded from 2026-08-10 until 2026-09-13,
> which made "was the dirt in the implementation this row names?" unanswerable
> the moment the working tree moved on — and that is the only question a reader
> of a dirty row has.*

**The list is still discarded, one layer down.** `Ledger.record`'s history
projection (`experiments/protocol.py:820-828`) copies a fixed allow-list when a
row is superseded:

```
("status", "ran_at", "commit", "message", "metrics", "control_metrics",
 "impl_sha", "spec_sha", "seeds", "gpu_job_id")   # + amended, supersedes_*
```

`dirty_files` is not in it. So the field survives exactly as long as the row is
the CURRENT one, and every re-buy erases it.

**The measurement, from `experiments/ledger.json` this sitting.** 127 rows have
been run since 2026-09-13. **Twenty of them are stamped `+dirty`. Nineteen carry
`dirty_files: None`:**

```
HR.1 004206c+dirty   LG.02 9b5166b+dirty   ME.1 9075d58+dirty   PL.02 7ffd3c8+dirty
T0.12 81e48d5+dirty  T0.21 8a21968+dirty   T0.21 0b0dcca+dirty  T0.23 b4f123d+dirty
T0.28 2ae81db+dirty  T0.28 853f471+dirty   T0.28 853f471+dirty  T0.28 b72f666+dirty
T0.29 563178e+dirty  T0.31 d4879c0+dirty   T0.31 9d9a99c+dirty  T0.35 d1cf88d+dirty
T0.36 f410abe+dirty  T0.36 1a78862+dirty   T0.36 d996c71+dirty
```

The single survivor is `T6.03` (`03e68bc+dirty`,
`['experiments/tests/t6_03_cross_session.py']`) — and it survives only because
it is **BLOCKED and has not been re-run since**. Across the whole file,
`dirty_files` is populated on **1 of 762 rows**. Note which specs dominate that
list: `T0.21`, `T0.28`, `T0.29`, `T0.31`, `T0.35`, `T0.36`. Those are the
instrument specs — the ones re-bought on every staleness bill, often within the
hour. **The rows most likely to be doc-dirtied are precisely the rows whose
dirty record has the shortest life.**

**Three concrete consequences, each checked rather than inferred:**

1. **`dirty_recoverability` tells those rows a false thing about themselves.**
   `protocol.py:2867-2871` branches on `files is None` and prints *"what else
   ran modified is unrecorded — this row predates `dirty_files`"*. Nineteen
   rows **postdate** it and were stripped. The same function's docstring
   (`:2849-2852`) names this exact shape — *"a red instrument whose sentence is
   false for most rows it prints"* — as the `audit_supersedes_fail` defect it
   was itself repaired against. It has reproduced it.

2. **A classifier comment cites a value the record cannot produce.**
   `protocol.py:80-82`, shipped as the 119th audit's FINDING 4 repair and
   re-read by this morning's fork-(c) work, states:
   *"T0.35 attempt 13 (`d1cf88d+dirty`, dirty_files
   ['docs/DECISIONS_RESOLVED.md']) recorded seven seconds after LG.13's clean
   bakeoff PASS."* The live row (`T0.35`, `2026-09-25T22:29:12`,
   `d1cf88d+dirty`) holds **`dirty_files: None`**. That sentence is the
   load-bearing evidence for moving `docs/DECISIONS_RESOLVED.md` into
   `RUNNER_OUTPUTS`, and **it is no longer re-derivable from the ledger.** I am
   not alleging the move was wrong — the mechanism (`bakeoff.py` appending
   mid-run) is independently verifiable in source. I am saying the *evidence*
   for it has evaporated, and nobody would find out.

3. **A verification performed this morning rests on the empty field.**
   `7055cd9` committed an inherited Review act after checking three facts at
   source, the third being *"0 of 156 ledger rows carry a `docs/.md`
   `dirty_files` entry"*. That is true and it is **vacuous**: the field is
   populated on one row in 762, and never on a doc-dirtied one. A count over a
   field that is null by construction is not a measurement. **This one is not
   the builder's fault** — the conduct (verify at source before committing
   someone else's act) is exactly right, and the instrument it reached for
   silently had no denominator.

**Ranked where it is and no higher, in fairness.** No capability claim rests on
this. The `+dirty` bit itself survives supersession, the `DIRTY` kind survives,
and `Ledger.unsatisfied` / `borrow_metrics` / `gate_precondition` all filter on
the kind — so **every refusal a dirty row earns today, it still earns.** What is
lost is the residual hazard: *which* files were uncommitted, i.e. the one
question `8a97fd9` exists to answer. It is ranked first because it is the only
finding in this window where the record says something false about itself.

---

## FINDING 2 — `run status` prints `!! ABOVE its declared floor` and exits 0; the builder reports "status rc=0" every slot as a clean bill (MEDIUM — the 120th audit's repair landed and is good; this is the half it did not ask for)

The 120th's FINDING 2 asked for two things and **both shipped and I verified
them rather than reading the commit message.** `6a15ad4` joined
`decisions.py`'s five classes into `ratchet_live()`; `ca68c4d` added the durable
guard that asserts the join covers every tool's own declared floored classes,
and `floored_class_gaps` refuses a join naming a counter the live scan does not
compute. `run status` now prints:

```
decisions_default_action_expired = 1  (unchanged since 2026-09-26)
  !! ABOVE its declared floor 0 — growth nobody raised the constant for.
```

**And `run status` exits 0.** `print_ratchet_block` is pure output;
`cmd_ratchets` returns 0 unconditionally (`run.py:2386-2414`) and `cmd_status`
has no ratchet branch at all. No floor state — `ABOVE`, `BELOW` or
`UNVERIFIED` — reaches an exit code in this tool.

**Stated fairly, because the class is not silent.** `decisions --check` exits 1
on it, it is one of the builder's four mandated per-slot checks, and every slot
journal in this window named it correctly (*"decisions rc=1 on D33 (the
Review's)"*). The gap is that the same slots also say *"status rc=0"* in the
same sentence, and a reader takes that as "no ratchet is broken", which today is
false.

**A second, smaller thing in the same block.** The committed reading says
`unchanged since 2026-09-26`. The class has read ≥1 since `D33`'s `decide_by`
passed on **2026-09-23** — the 120th audit measured exactly that. The `at` date
records when the KEY was created, not when the NUMBER broke, so `run status`
alone now dates a three-day-old break to today. Not a laundering; the honest
history is in the 120th's FINDING 2 and in `6a15ad4`'s own message. Worth one
line in the readings file so it does not need an archaeologist next week.

**Also in this block and I am required to say so:** `review_queue_net_arrivals`
= **15, `!! MOVED +1 (clock +0, act +1)`** since 2026-09-26. The act is
arrivals: three rows were routed today (`ba03-registered-run-foreclosed-by-d20-
class-closure`, `t028-p10-reads-an-empty-armed-register-as-a-broken-tool`,
`doc-declarations-restale-three-tier0-certificates-daily`) against this desk's
disposals. Every one of the three is a real defect routed rather than absorbed,
which is the conduct this project wants; the counter moving is the *price* of
that conduct, not a fault, and the drain below is where it lands.

---

## FINDING 3 — `docs/OVERSIGHT.md` is the one organ-written doc this morning's doc-dirt repair left as full code dirt, and by the repair's own measured rule it qualifies for the exempt class (LOW-MEDIUM — cheap, and the mechanism to do it safely shipped in the same commit)

`b4df9bb` is good work and I checked its claims instead of reading them. The
three-class classifier is sound; `is_code_dirt`'s three-valued `declared_docs`
keeps `gate_precondition` and the GPU push guard at the **pre-fix conservative
answer**; `RUNNER_OUTPUTS` beats any declaration; and — the part that matters
most and that I expected to find missing — **`undeclared_doc_readers` gates
`PROSE_DOCS` too**, minus the named `WRITE_ONLY_DOCS` exemption
(`protocol.py`, `undeclared_doc_readers` docstring). That closes the one
direction a declaration cannot repair: a spec that starts reading a prose doc
will fire the detector. The mutation falsifier catching the battery's own
fixture, and the scan correcting the disposition's own PROSE list before it
shipped, are both exactly right.

**The residual.** `docs/OVERSIGHT.md` is in no class, so it is full code dirt
for every spec. The commit's stated reason is *"read by `steering.py` but by NO
spec closure, so it stays plain code dirt: this class is about specs, not
organs."* **I re-ran that measurement independently: 164 specs have a module
path, and zero name `docs/OVERSIGHT.md` anywhere in their closure.** That is the
same evidence — *zero spec closures name it as a live path* — that the same
commit used to widen `docs/LESSONS.md` INTO `PROSE_DOCS`. Applied consistently,
`OVERSIGHT.md` belongs there too.

**Why it is worth the two-line change rather than being left.** The 09-02 scar
this whole fork exists for was *an audit's in-progress doc writes* dirtying a
concurrent runner sweep. After `b4df9bb`, of the three docs this organ writes,
`LESSONS.md` is exempt and `DECISIONS_NEEDED.md` is per-spec — **`OVERSIGHT.md`
is the only one that can still stamp a concurrent registered run `+dirty`.** The
overseer and the Review collide at 06:37 and the builder's slots run to ~:33, so
the window is real, and the report this organ writes is one of the largest doc
writes in the repo. The falsifier already gates the prose class, so the move
cannot rot.

**Honest limit on this finding:** it has never actually bitten. Zero rows in the
whole ledger carry a `docs/*.md` entry in `dirty_files` — though per FINDING 1
that statistic is nearly unfalsifiable, so read it as "no evidence", not as
"no occurrence".

---

## FINDING 4 — the Review has not published a page in two days, today's DAILY died `rc=124` after seven acts, and this organ is required to read a `FOR THE OWNER` section that is describing a different week (HIGH for the owner, MEDIUM for the machine; standing, and the detector is working)

`docs/PROGRESS.md` last moved at `8128a75`, **2026-09-26T01:11** — and that
commit is the machine **stamping the STALE banner**, not a page. The last actual
page is `2026-09-24`. Today's sitting opened at 06:37, committed **seven acts**
(`91392d5`, `3c7f90e`, `c790064`, `1e36765`, `ecd6183`, `8f4529b`, and act 7
inherited by the builder at `7055cd9`), recorded its own death in
`PROGRESS_LOG.md` at 06:57 (`ad102a7`), and **never wrote its report.** That is
the **fifth** incomplete sitting.

**I read the two sections my instructions require, and both are describing a
world that no longer exists.** `FOR THE OWNER` #1 reads *"THE BUILDER IS BACK
AND HAS NOTHING TO DO"* at `week:all models` **19%**; today's slots read
**59–63%** and shipped five units. `FOR THE BUILDER` items 1–4 are all
discharged. `FOR THE OWNER` #4 cites `D32`/`D34` as falling due "TODAY" — both
fired by armed default on 09-25. The banner is doing its job and **nothing is
hidden; nothing is current either**, and the owner-ask reader
(`UNROUTED-OWNER-ASK`, now at 0 against a baseline lowered to 0 this morning) is
reading a two-day-old page for its population.

Tomorrow, **2026-09-27**, is the Sunday FULL that `D36` is about, that carries
**7 dated rows against a measured capacity of 6**, and on which this desk has
publicly pre-committed to DECLINE the W1 authorship if the date breaks a fifth
time.

---

## FINDING 5 — three of the owner's own commitments still have no runnable claim, at 22 days, and the frontier is unchanged (MEDIUM, standing — this is Section 3's hard half)

`coverage` exits 2 with `commitments_uncovered = 0` and **three CLAIM-DEAD**:
**smell** (`SM.02` PARKED, `SM.03` FORECLOSED), **shelter/building** and
**thermal — too cold/too hot KILLS him** (both on `SH.01` PARKED / `SH.02`
FORECLOSED). Every foreclosure carries the same sentence: *the repair is a
redesign, never a dispatch.* `goal_unrunnable` = **7**, unchanged since
2026-09-05 — **21 days**. `NO-LIVE-PATH` reads **6 distinct commitments/seats**,
5 lower bound.

Beyond them: **curiosity** 12 specs, 2 pass, **0 runnable now**; **fast/slow** 8
specs, **0 pass**; **one brain / unison** 28 specs, **1 pass**. The dispatch
queue is **7 deep and 7 of 7 VOID** — *"an arm to repair, not a dispatch"* —
with **three cost classes newly empty and no path in**, including both small
GPU classes on the day 29 free GPU-hours expire.

`coverage` also prints **4 NEW unrunnable citations** above the
`GOAL_UNRUNNABLE_BASELINE` of `{DP.02, DP.03, LC.04}` — `GEN.02`, `GEN.03`,
`GEN.06`, `GEN.09`, all `welded<-LC.07`. **Already routed**, two days ago, as
`gen-four-reparented-to-a-decision-that-had-already-closed` (DUE 10-01), so it
is owned and I am recording it rather than re-raising it.

---

## FINDING 6 — the queue's drain is UNBOUNDED, 14 rows fall due by tomorrow against a capacity of 6, and `DECLINE` is still unused on row 87 of 87 (MEDIUM, standing)

`run review-queue` exits **2** — **39 OPEN, 3 HELD, 20 DISPOSITIONED, 25 ACTED,
0 DECLINED of 87 routed; 62 live rows; oldest live 33 days.** Trailing 7 days:
arrived **25** (3.57/cycle), disposed **10** (1.43/cycle), designed 14 (still
live, still ageing) — **net +15, drain UNBOUNDED.**

**1 VIOLATION, OVERDUE:** `a4-mandatory-collapse-diagnostic-is-declared-and-
computed-nowhere`, promised 2026-09-25, one day old. It is the row `D28`'s
armed default **ranked first** in its own batch on the stated ground that three
open questions converge on the A4 seat and deciding it first stops the project
paying up to three times for one 14.40 core-hour run. Ranking a row first and
then breaking its date is the most expensive shape of lateness available here.

**IMMINENT: 14 live dated rows** fall due on or before the next cycle
(2026-09-27) against a measured capacity of **6**; the tool says **8 of them
cannot be discharged by that cycle**. `2026-09-27` alone carries **7**.

`DECLINE` remains unused across all 87 rows this file has ever carried — the
**80th** consecutive row on which that is true. Three honest repairs exist and
this desk has only ever used two of them.

---

## Section 1 — INTEGRITY OF THE LEDGER: no findings; every cell re-derived from `ledger.json` and `git`

| check | result |
|---|---|
| spec rows in the ledger | 156 (110 PASS, 29 FAIL, 16 VOID, 1 BLOCKED) |
| PASS rows (latest per spec) | **110** |
| `commit` resolves via `git cat-file -e <sha>^{commit}` | **110 / 110** |
| spec declares a `control` in the registry | **110 / 110**, zero exceptions |
| PASS row carries `control_metrics` | 108; the two without are `T0.01` and `T0.10`, both declaring `control="NONE, BY DECISION (52nd audit B5)"` |
| implementation module resolves (`protocol.module_path_for`) | **110 / 110** |
| PASS rows carrying a `+dirty` stamp | **0** |

Standing reporting-only weaknesses, all at floor and none new: **3 UNBACKED
CERTIFICATES** (`LF.02` ← `T6.03`; `T2.03` ← `T1.08` FAIL; `T2.14` ← `T1.08`
FAIL), `pass_on_dead_dependency` = 3 at its declared floor; `T0.27` a
deliberately-red gate. The one PASS-row concern I went looking for and did not
find: no PASS certificate in this window was bought from a dirty tree.

## Section 2 — thresholds and controls over time: no loosening

`git diff e7aaa97..HEAD` over `registry.py`, `registry_expansion.py` and
`experiments/tests/` — 8 files, +451/−7.

- **`cdab11a` — `BA.03`, comment-only.** I stripped comment lines from the diff:
  **not one numeric or executable line changed.** `TILT_GAIN_MIN_FRAC = 0.20`
  and `TILT_VEST_OVER_NOISE_FRAC = 0.05` are byte-identical and the two labels
  that said *"demanding side"* now say **PERMISSIVE** with the measured
  discounts (20% below a derived 0.25; 32% below a derived 0.074). That is
  exactly the 120th's option (b), and refusing option (a) was correct: the
  spec's own freeze binds after the pilot is read, and the pilot was read at
  06:1x the same morning.
- **`a080386` — `T2.11` strengthened.** New claim conjunct joined with `and`:
  `and m["mi_margin"] >= MI_MARGIN_MIN`, `MI_MARGIN_MIN = 0.50` nats. The
  builder measured that the derivation which SET that bar was wrong in the
  **permissive** direction — the zero-arm floor is `−0.2149/−0.2158`, not the
  0.16 the bar was sized as 3.1× of — and **held the bar at 0.50 anyway**, with
  the correction written beside the constant. *"A threshold is not recomputed
  when its justification improves"* is the right rule and this is the first time
  I have seen it applied against the project's own interest. Seven sibling bars
  byte-unchanged; `beats_shuffled` not demoted; `_GATES_FROZEN` still `False`;
  no dispatch, no ledger row.
- **`6a15ad4` — two baselines LOWERED**, `BASELINE_UNDECLARED` 10→0 and
  `BASELINE_UNROUTED_ASKS` 3→0, on the BELOW-floor banner's own order. Lowering
  a shrink-only floor is a tightening. Correct.
- **`b4df9bb` — the one narrowing in the window**, analysed in FINDING 3. It
  narrows what counts as dirt for a DECLARING spec only, keeps the conservative
  answer for every caller without spec context, and ships a mutation falsifier
  that gates both the instrument and the prose class. Justified by measurement,
  and the measurement corrected the ruling that ordered it.

**No control was deleted or weakened. No `_check` gained an `or`. No seed count
was reduced. No assertion was removed.**

## Section 3 — drift from the goal

**Every unit in the window traces to a `GOAL.md` sentence and none of it is
drift.** The five units — BA.03's label repair, the `decisions.py` ratchet join
and its durable guard, the `lib_liveness.sh` guard, the `T2.11` held-out read,
and the doc-dirt fork — all serve `GOAL.md:8-9`, *"or protects the honesty of
watching what happens when the three meet."* The `T2.11` read additionally
serves the skill-discovery arm of one-brain-in-unison.

**The converse is the finding, and it is FINDING 5.** Eleven ledger events in
this window; **zero** carry a verdict about Jack. The last one that did was
`LT.02` FAIL→PASS on 09-25. `BA.03`'s registered run — the only fresh dispatch
this project had — was **refused at the gate by `D20`'s own armed default**
(`4a7a571`), correctly, and the builder routed and escalated it instead of
bypassing it. That refusal is right and it leaves the science board empty.

## Section 4 — is the builder alive and productive? Yes, and the conduct is again the best thing here

Seven iterations in the window (`06:07` … `12:07`), **7 of 7 `rc=0`**, zero dark
slots, `lost_iterations.log` 0 bytes. PASS delta **111 → 110**. The `-1` is
`T0.28` FAIL attempt 22 on `p10_live_document_is_armed_and_readable` — and the
cause is the escalation machinery **succeeding**: `D31`/`D32`/`D34` fired and
were transcribed, the three survivors are all conduct-desk class, so the live
armed register is legitimately empty for the first time in the tool's history.
The builder **proved the pre-edit code returns the identical result on the same
document**, so its own refactor did not cause it, left the FAIL standing
unrelitigated, and routed it. A loop that takes a `-1` it could have argued away
is the loop this project needs.

Four other refusals worth recording, each of which cost it something:

1. `4a7a571` — dispatched the armed `BA.03` run, hit `gate_cpu_child`'s refusal,
   and **did not** bypass it (no direct `run(Ledger())`, no budget relabel, both
   pre-refused in writing). Routed to three surfaces plus a lesson.
2. `a080386` — held `MI_MARGIN_MIN` at 0.50 after falsifying the derivation
   **downward**.
3. `b4df9bb` — the falsifier flagged the battery's own fixture and the builder
   **fixed the fixture rather than declaring a dependency it does not have**.
4. The 11:07 and 12:07 slots refused Fable at `week:Fable` 95% under `D14`'s
   model floor and walked up to Opus, acting on `week:all models` 63% — the
   right line, named, under the 90 stop.

The one soft spot, self-disclosed: the 10:07 slot left a detached background
read that was **dead** (0 bytes, no process) and 34 uncommitted lines; the 11:07
slot found it, re-ran in the foreground with pids declared, and said so.

## Section 5 — compute honesty

**GPU.** `2026-W38`: **0.9176 h drawn of 30**, 2 jobs. **~29.08 free Kaggle
GPU-hours expire tonight at 00:00 — the SECOND consecutive week.** The builder
refused to manufacture a dispatch for the **45th** consecutive slot and I
endorse it for the second audit running: `coverage`'s queue-depth block reads
**every GPU class empty or not fillable**, with `gpu<20min` and `gpu<8h` blocked
on evidence (`DP.04`/`SM.03`, `LC.07`) where *the repair is a redesign*. An hour
spent on a run nothing asked for is worse than an expired one.

`gpu_hours_no_verdict` TOTAL **48.42 h**, of which **`D1.0` is 33.78 h across 2
attempts and 0 verdicts** — unchanged since 09-18 and still the largest single
unredeemed spend on the record. `gpu_unattributed_jobs` = 21, at floor.

**CPU.** Today **8,804.09 s of 57,600 s**, of which **8,390.77 s is the BA.03
tilt pilot** — one declared unit on the sanctioned lane. The remainder is 413 s
across eleven instrument re-buys. `cpu_foreclosed_now` = 37 is the tenant
protection working, not a fault. **No class breach.**

## Section 6 — stuck decisions

**`MEANS-ESCALATED`: 0.** No fork a measurement could settle is on the owner's
desk. That is the `D1` disease and it is absent.

**`UNDECLARED`: 0** against a baseline lowered to 0 this morning — **nothing for
me to arm this audit, and the ratchet cannot shrink further by my hand.**
**`OVERDUE — DEFAULT IS DUE TO FIRE`: 0.** I fired nothing and nothing was due.

**`DEFAULT-ACTION-EXPIRED`: 1, baseline 0 — the project's only broken ratchet**,
`D33`'s clock defect, red since 2026-09-23. It is the desk's to repair (shorten
`decide_by`, or annotate the date `(CLOCK: <whose>)` — the idiom `D36`'s own
default already uses verbatim). The 120th appended the `RATCHET NOTICE` under
`D33`; I am not appending a second. See FINDING 2 for what the repair did and
did not buy.

**Three `CONDUCT-DESK` flags, all the Review's:** `D33` (due 09-23, stale 3 d),
`D35` (due 09-24, stale 2 d — though its live queue row `d35-none-quota` WAS
ruled on limb (ii) at `1e36765` this morning), `D36` (**due today**).

**One `STEERING-DATE-MISMATCH`:** `PROGRESS.md` quotes `D33` as 2026-09-27 where
the register says `decide_by 2026-09-23`. Reporting-only, and it is the stale
page — it will clear when the Review publishes.

**Two owner-asks matched to `D22`/`D31` by citation** rather than routed. Both
are on the 09-24 page; the reader is reading a two-day-old population (FINDING
4).

**Nothing was quietly acted on without being recorded.** I checked the window's
fourteen commits against `DECISIONS_RESOLVED.md` and `DECISIONS_NEEDED.md`: no
owner decision was resolved by anyone in this window.

## Section 7 — bakeoff hygiene: no new findings

`docs/DECISIONS_RESOLVED.md` did not move in this window. The standing item is
unchanged and remains correctly disclosed: `T4.06`'s latent-recovery conjunct
certified at **+0.0187 = 6.9% of the incumbent's own 0.2699 seed spread with 1
of 3 seeds regressing**, marked in the spec as *"not to be quoted as
demonstrated"* and reprinted by `run status`'s ANCHOR-DECIDED CONJUNCTS block
every slot. No decision was made without a learning gate. The project's one
VOID-treated-as-verdict is `CHAMPIONS.md`'s Learning-core seat, which
`champions --check` names as `VERDICT-IS-A-VOID` and which is standing, not new.

`champions --check` exits **0**, ratchet at floor, 10 violations — all standing:
`champions_trigger_debt` **3** (unmoved **23 days**), `champions_unwinnable`
**4** (unmoved **13 days**), the **World** seat still held BY VERDICT with no
deciding row and **no `TRIGGER:` at all**. `ARENA-MISSING` remains **0** and
`UNDECLARED` seats **0 of 0** — both closed by REGISTRATION, the only
non-laundering repair.

## Section 8 — the honest summary

**Are we closer to a curious humanoid that climbs the ladder than we were six
hours ago? No. Closer than yesterday? No.**

Eleven ledger events in this window and **not one of them says anything about
Jack.** The last verdict about the creature was `LT.02` on 09-25. The only fresh
registered dispatch this project owned — `BA.03` — was refused at the gate by a
default this project itself armed, correctly, and there is no second buyer. Three
of the owner's own constitutional commitments have had no runnable falsifiable
claim for twenty-two days. Curiosity: 12 specs, 0 runnable today. Fast/slow: 8
specs, 0 passing. One brain in unison: 28 specs, 1 PASS. Twenty-nine free
GPU-hours expire tonight for the second week running, and the whole dispatch
queue is seven deep and seven-of-seven VOID.

**And the counterweight, which is real and which I will not shrink.** In six
hours this loop: took a `-1` on the board that it had a clean argument against
and did not use; held a threshold at 0.50 after proving its own derivation was
too generous; refused a dispatch it was armed to make rather than relabel a
budget; corrected a label that had called a 20%-permissive discount "demanding";
and shipped a doc-dirt classifier whose mutation falsifier caught **the ruling
that ordered it** and then **caught the fixture that tested it**. That is five
separate occasions in one morning where the machine moved against its own
convenience.

Which is why this is **DRIFTING**, and the shape of the drift is unchanged from
the 120th: *a ladder whose instruments improve faster than its creature does is
being maintained, not climbed.* The bottleneck is not the builder, not the
meter, and no longer the GPU hours. It is that the two 45-spec designs this
project actually needs — `T1.08`'s variance repair (`frees 3 / blocks 45`) and
the W1 world-edit window (three specs **twenty days** unregistered) — have never
once fitted inside a sitting, **five of five of those sittings have now died
incomplete**, and the desk that owes both of them did not publish a page today
either. `D36` is the entry that decides which of the two gets tomorrow, and it
falls due today.

---

## FOR THE BUILDER

1. **Carry `dirty_files` through the history projection (FINDING 1), and it is
   three separate repairs, not one.**
   **(a)** Add `"dirty_files"` to the allow-list at
   `experiments/protocol.py:820-828`. It rides along for exactly the reason
   `impl_sha` and `spec_sha` already do, and the field is `None` on clean rows
   by construction so nothing is invented for the 163 evidence-free entries.
   **(b)** `dirty_recoverability` (`protocol.py:2867-2871`) must stop telling
   rows they *"predate `dirty_files`"* when they postdate it. The honest branch
   is three-valued, not two: recorded / predates the field (`ran_at` <
   2026-09-13) / **recorded and stripped by supersession**. Its own docstring
   at `:2849-2852` names this as the `audit_supersedes_fail` shape — *the
   sentence changes, the alarm does not*.
   **(c)** `protocol.py:80-82` cites `T0.35` a13 as carrying
   `dirty_files ['docs/DECISIONS_RESOLVED.md']`. The row holds `None`. Either
   re-derive it (the `git reflog`/`bakeoff.py` mechanism is independently
   checkable) and say where from, or mark the quoted value as **no longer
   re-derivable from the ledger**. Do not delete the comment — the move it
   justifies is sound; it is the citation that has rotted.
   **The durable half, and it is the item — not (a):** a property that a
   superseded row loses no field the live row carried, over the union of
   `Result`'s dataclass fields minus a named, commented exemption list. `T0.17`
   already owns ledger provenance and `T0.31`'s assert-on-the-TOTAL shape is the
   right idiom. `dirty_files` is the instance; the allow-list that silently
   drops new fields is the defect, and `attempt` and `duration_s` are dropped by
   the same line today.
2. **Stop using "N of M ledger rows carry a `dirty_files` entry" as a
   verification until (1) lands**, and when you do use it, print the
   denominator. `7055cd9`'s third source-check was *"0 of 156 rows"* over a
   field populated on **1 of 762**. The conduct — verifying an inherited act at
   source before committing it — was exactly right; the instrument had no
   denominator and did not say so. This is the same class as the 120th's
   FINDING 2: a number that cannot move is not a check.
3. **Make `run status`'s exit code mean what its slot summaries claim, or make
   the block say it does not (FINDING 2).** Today `print_ratchet_block` prints
   `!! ABOVE its declared floor 0` and `cmd_status` returns 0. Two honest
   repairs, your choice: **(a)** `cmd_status` exits non-zero when any floored
   class reads `ABOVE` — which is `decisions --check`'s own semantics, already
   the project's convention; or **(b)** one printed sentence in the block
   stating that floor state is reporting-only here and names the tool that
   gates it. What is not a repair is a slot summary that pairs *"status rc=0"*
   with a broken ratchet in the same paragraph. Price the staleness bill first
   as usual — `T0.36` is the sole `IMPL_DEPS` on `run.py`.
   Cheap addendum in the same commit: give
   `decisions_default_action_expired`'s readings entry a note that the class
   has read ≥1 since **2026-09-23**, not since the key was created today.
4. **`docs/OVERSIGHT.md` into `PROSE_DOCS` (FINDING 3), with the measurement in
   the commit.** I re-derived it: **164 specs have a module path and zero name
   `docs/OVERSIGHT.md` in any closure** — the same evidence, by the same rule,
   that put `docs/LESSONS.md` in that class in `b4df9bb`. `undeclared_doc_readers`
   already gates the prose class minus `WRITE_ONLY_DOCS`, so the move cannot rot
   silently: if a spec ever starts reading it, the falsifier fires. Leaving it
   as full code dirt keeps the overseer's own report able to `+dirty` a
   concurrent registered run, which is the scar class fork (c) was built for.
   Low priority, two lines, and re-run the scan rather than trusting mine.
5. **Nothing else is owed by you and the empty science board is a true
   reading.** Seven slots, 7/7 `rc=0`, five real units, four refusals that each
   cost you something, and a `-1` you took honestly. The one thing to keep
   doing: you re-derived the board every slot instead of inheriting it, and on
   the 11:07 slot that is what caught the previous slot's dead detached child.
6. **Still do not pre-empt:** `W1.01`/`W1.03`/`W1.04` registration, `D33`/`D35`/
   `D36`, `UB.10`'s successor arm choice, `T1.08`'s pipeline design, the
   `w1-world-edit-window` docket, the `lc03` seat row, the `t306` venue row,
   `A4`'s three-way fork, and the `ba03-registered-run-foreclosed-by-d20`
   options you priced this morning. Tomorrow's 06:37 is the Sunday FULL and
   `D36` decides what it holds — that is the Review's sitting, not yours.

---

## FOR THE OWNER

**1. `D36` falls due TODAY and it is still the one entry that decides whether
anything moves this week. Cited, not re-asked.** Two 45-spec designs want one
Sunday sitting; **five of five** such sittings have now died incomplete (four
Sunday FULLs plus today's DAILY at `rc=124`); `2026-09-27` carries **7 dated
rows against a measured capacity of 6**; and `D33`'s armed default has already
ordered W1 first, which the Review explains — correctly, in my reading — it may
not reverse at its own desk. The realised price is now paid twice: **~29.08 free
Kaggle GPU-hours expire tonight, unbought, for the second consecutive week**,
both weeks behind the same undesigned `T1.08` repair. **The hours are not the
scarce resource. The design sitting is**, and this week it produced seven acts
and no page.

**2. NO-DECISION — your Review did work today and you have not heard from it in
two days.** The 09-26 DAILY disposed or re-dated six rows, delivered the `t211`
METRIC ruling the builder executed four hours later, and died before its report;
the builder found act 7 uncommitted, **verified its three factual claims at
source, and committed it as found**. `docs/PROGRESS.md` carries a
machine-stamped STALE banner over a 09-24 page whose headline (*"the builder is
back and has nothing to do"*, `week:all models` 19%) is false of today, when the
meter read 59–63% and the loop shipped five units. Nothing is hidden and nothing
is current. The desk's own pre-committed stop-rule on `w1-world-edit-window`
fires tomorrow either way.

**3. NO-DECISION, and it is the direct answer to what this project is for.
Three of your own constitutional commitments still have no runnable falsifiable
claim at all:** **SMELL**, **SHELTER**, and **TOO COLD / TOO HOT KILLS HIM.**
Twenty-two days. Every claim spec behind them is PARKED or FORECLOSED and every
foreclosure says *the repair is a redesign, never a dispatch*. No instrument is
broken and no organ is at fault; there is simply nothing on the board to run for
those three, and **nobody has been asked to design the successors** — there is
no queue row that owes one. If you add one thing to the docket beyond `D36`,
this is it. It is the difference between a jungle survival creature and a
locomotion demo.

**4. NO-DECISION, reported because you should know the shape of it: the ledger
forgets what it was told to remember about dirty runs (FINDING 1).** Thirteen
days ago this project noticed that a `+dirty` stamp which does not say WHICH
files were dirty is unauditable, and fixed it. The fix works — and the ledger's
own history projection strips the field on the next re-run, so **19 of the 20
dirty rows since then say "unrecorded"**, and the reader tells them they predate
a field they postdate. No capability claim rests on it and no refusal was
weakened; what is lost is the ability to answer, later, *what else was
uncommitted when that number was bought*. It is routed to the builder as a
three-part repair plus a durable property, and I am telling you only because it
is the second time this exact question has been discarded one layer further
down, and the pattern — a repair that lands correctly at its own layer and is
undone by the layer beneath it — is worth your attention even though this
instance is cheap.
