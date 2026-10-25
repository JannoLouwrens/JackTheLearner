> **INCOMPLETE RUN — THIS IS A DRAFT, NOT A FINDING.**
> The overseer run that wrote this file exited rc=1 and did not
> complete its own checklist (2026-10-25T12:57:16+00:00). Everything below was
> written before the run stopped: any verdict, any section claiming
> "no findings", and any instrument table in it are UNVERIFIED.
> Sealed automatically by scripts/lib_seal.sh; the exit code is in
> the log, and this banner is what joins the two.
> Files this run also left dirty, committed unbannered by the seal: docs/DECISIONS_NEEDED.md, docs/REVIEW_QUEUE.md, scripts/ladder_prompt.md.

# OVERSIGHT.md — the overseer's current-state report

> Current state, not a log. Each audit rewrites this file.

**2026-10-25 12:37–13:0x UTC — 160th audit.** VERDICT: **DRIFTING.**

**Two disclosures first, because both affect how to read every number below.**

1. **The Sunday FULL Review is live in this working tree as I write.** It
   committed acts 1–6 (`963c856` → `f8ba05b`) between 12:28 and 12:40, minted
   two register entries, drove the queue's OVERDUE class **14 → 0**, and still
   holds `docs/CHAMPIONS.md` and `experiments/registry_expansion.py` modified
   and uncommitted. It also **detached HEAD four times** during my sitting
   (`git reflog`: checkouts to `695409f`, `963c856`, `e656a1b`, `cfba2a5` and
   back to `main`) — one of my `git log` reads landed inside that window and
   showed an older history than the branch's. I re-ran every instrument after
   its last commit, and I committed **by filename**, never `-A`.
2. **No audit ran on 2026-10-23 or 2026-10-24.** `git log` of this file ends at
   the 159th. Both days' slots were held by the 90% usage stop, which is itself
   the subject of RANK 1.

True exit codes, taken from the processes and not through a pipe, after my own
edits: `coverage` **2** · `decisions --check` **0** · `champions --check` **0** ·
`run review-queue` **0** (it exited **2** with **14 OVERDUE** at my open; the
Review drove it to 0 during the sitting) · `run status` **2**.

Ratchet line at my open, before I touched anything:

> `8 MOVED (fail_unowned_owned_forms {'disposed': 1, 'queue-row': 27,
> 'repaired_by': 1} -> {'disposed': 2, 'mention-only': 1, 'queue-row': 26,
> 'repaired_by': 1}, gpu_hours_no_verdict [map], pass_on_dead_dependency
> 6 -> 7, projection_vs_estimate [+1], review_queue_net_arrivals -5 -> -4,
> review_queue_piled_on 15 -> 5, review_queue_violation_forms {'OVERDUE': 16}
> -> {'OVERDUE': 13}, review_queue_violations 16 -> 13); no counter refused to
> compute; floors: 2 ABOVE (pass_on_dead_dependency, unreachable), 0 BELOW,
> 0 UNVERIFIED.`

`claim_dead = 3` · `commitments_uncovered = 0` · `fail_unowned = 6` (at floor) ·
`champions_trigger_debt = 3` · `champions_unwinnable = 4` (at floor) ·
`dark_slots = 0` — none of these moved, and the last one is a lie of shape, not
of arithmetic (RANK 4's neighbour, already routed by the 159th).

---

## RANK 1 — a Tier-0 certificate went red under the only organ the usage stop leaves running, and nothing read it for 2.3 days

**This is the most damaging finding available this morning, because it is about
the one direction of failure this project has no reader for: the ledger being
written while every reader of the ledger is switched off.**

`T0.07` (*CPU throughput baseline*) recorded **attempt 7 FAIL** at
`2026-10-23T04:48:54`. Read out of git at both revisions rather than from any
counter:

| revision | status | ran_at |
|---|---|---|
| `a7bd35d` (10-19) | **PASS** attempt 4 | 2026-09-06T17:18:05 |
| `3d3433f` (10-23 02:50) | **PASS** attempt 6 | 2026-10-21T00:48:58 |
| HEAD | **FAIL** attempt 7 | 2026-10-23T04:48:54 |

**The verdict is honest and the number is not a performance regression.** The
binding conjunct is `policy_no_llm_rel_std` **0.272** against
`MAX_POLICY_REL_STD` **0.20** (`experiments/tests/t0_07_throughput.py:244`,
constant at `:51`); every other conjunct is green — `calibration_error` 0.0259
of 0.15, `bare_repeat_spread` 0.0126, `vectorised_rel_std` 0.0020,
`policy_rel_std` 0.0258. The test's own comment says what that conjunct is for:
*"Sound measurement, not a performance bar."* So the honest reading is that the
no-LLM forward pass did not reproduce trial-to-trial at that instant, on a box
this organ has repeatedly recorded as shared with paying tenants.

**Who wrote it, and who could have read it.** Counted by hand off
`/data/jack-logs/ladder.log` and `/data/jack-logs/regate.log`:

```
                     10-22        10-23        10-24        10-25
builder (hourly)     STOPPED 18:07 -> 09:07 on 10-25, ~72 consecutive slots,
                     every line "STOPPED at 90..100% weekly usage"
Review (06:37/12:37) no sitting; docs/PROGRESS.md stamped STALE at 77 h
overseer (37 */6)    no audit on 10-23, none on 10-24
regate re-buy lane   12 sweeps    12 sweeps    12 sweeps    7 so far
```

**The usage stop pauses every READER and leaves the one WRITER running.** The
re-buy lane is not sampled by the gate, so it kept re-deriving certificates
twelve times a day, and the one verdict it changed landed where no organ could
see it. The regate log has been printing the consequence every sweep since —
*"skip PL.00 ... dependency red: T0.07 (FAIL); owned by its
pass_on_dead_dependency row"* — which is the writer correctly reporting its own
damage into a log that no desk file reads.

**What it cost, two readings, both computed:**

- `pass_on_dead_dependency` **6 → 7**, now **4 above its declared floor of 3**.
  The new pair is **`PL.00 ← T0.07 (FAIL)`**: the standing certificate carrying
  the frozen-vs-plastic decree's own arena rests on a dependency the ledger says
  is dead, and `run_spec` would refuse to re-derive it today.
- `fail_unowned_owned_forms` gained its **first-ever `mention-only`** entry, and
  it was `T0.07` (`coverage.fail_unowned()['owned']['T0.07']`). Its entire
  presence in `docs/REVIEW_QUEUE.md` was prose inside other rows' staleness
  bills and a residency note — no `DUE:`, no repair owner. `fail_unowned` stayed
  **6, AT its floor**, so **no exit code anywhere moved**: a Tier-0 red with no
  dated promise behind it reddened nothing.

**ACTED, not just reported.** Routed as
`t007-tier0-certificate-went-red-under-the-only-organ-the-stop-leaves-running`
(`DUE: 2026-10-31`, the tool's own `next_free_due`), which converts the owner
form `mention-only → queue-row` — verified live after the append. The row
**refuses a bar move in advance**: `MAX_POLICY_REL_STD` is a soundness floor,
and widening it to admit 0.272 would convert an honest unreproducible reading
into a certificate. The premise correction is recorded as an **EVIDENCE
ADDENDUM** on the open register entry about the stop's missing record, whose
title says the stop *"pauses every organ"* — measured, it does not.

## RANK 2 — the dirty-tree guard is checked once at sweep entry, and six certificates went dirty mid-sweep seventeen minutes after it was certified live

**This is the only finding on the page that was still happening while I wrote
it.** At 12:29 the Review stamped `regate-bought-nine-certificates-from-its-own-
uncommitted-tree` **ACTED** and verified at the desk that the guard *"is LIVE AT
HEAD."* At 12:43 the re-buy lane opened a sweep over a clean tree, correctly
passed the guard, and then this happened — eight rows of one sweep, in `ran_at`
order, read off `experiments/ledger.json`:

```
12:43:02  sweep start - tree clean, guard correctly passed
12:43:15  T0.21   f8ba05b          clean
12:44:37  T0.28   f8ba05b          clean
          <- experiments/registry_expansion.py edited by the live FULL Review
12:45:36  T0.31   f8ba05b+dirty    ['experiments/registry_expansion.py']
12:45:41  ME.3    f8ba05b+dirty    same
12:45:48  ME.11.A f8ba05b+dirty    same
12:46:16  T0.34   f8ba05b+dirty    same
12:46:19  T1.13   f8ba05b+dirty    same
12:46:22  T0.01   f8ba05b+dirty    same
```

**The ACTED repair is not at fault and this is not a re-route of it** — checked
at source. That row's subject is the tree's state **at entry**, its repair
(`protocol.gate_precondition` on the run path, `scripts/regate.py:273-276`) is
live, and it worked here: the first two certificates are clean *because* the
guard let a clean tree through. The defect is the guard's **temporal scope**.
Three organs write this tree — builder hourly, Review on its sitting, regate
every two hours — a sweep takes minutes, the check takes an instant, and the
predicate is never re-read at the point of spend.

**Two costs, both computed:**

- **`T0.31` was re-bought clean by the Review at 12:35** (`3488778`, 2.14 s,
  *"from the clean tree"*) **and carried a `+dirty` stamp again 61 seconds
  later.** The desk paid for a clean stamp and the lane spent it. `T0.31` is the
  backlog reader's own gate, so the certificate proving this queue cannot be
  quieted by tidying is one of the six that went dirty.
- **`unreachable` moved 96 → 97**, and the mechanism is now visible in
  `run blocked`, which lists two of the six as terminal blockers:
  `T0.01 = PASS but STALE — re-run it  frees 1 (blocks 1)` and
  `ME.3 = PASS but STALE  frees 0 (blocks 3)`. The control reading is the
  Review's own, from its 12:51 anatomy commit: *"unreachable is UNCHANGED by
  this edit: 97 at HEAD, 97 with it"* — so its two new registrations did not
  move the number, and the growth sits inside the window containing this sweep.
  This is the identical knock-on `e735f36` priced on 10-19, when the same class
  took `unreachable` 96 → 101 and made `coverage` **mandate an irreversible
  edit to a shrink-only constant on a two-hour transient**.

Routed as
`regate-dirty-guard-is-checked-once-at-entry-and-the-tree-went-dirty-mid-sweep`
(`DUE: 2026-11-01`), with the repair constrained in advance to re-reading the
same predicate per spec and the three illegitimate repairs — widening what
counts as clean, suppressing the stamp, retroactively re-stamping the six —
refused on the row. The six stand as bought, as the nine before them were left
to stand.

## RANK 3 — I fired an armed default on Jack's character, and the mitigation its own text leans on does not exist

`decisions --check` listed exactly one `OVERDUE — DEFAULT IS DUE TO FIRE` row at
my open (`D44`, `class goal`, `decide_by 2026-10-24`). **The owner did not rule
by 2026-10-24, so the pre-registered default fired** — the no-ruling branch: the
sheet stays exactly as it ships and nothing is documented. Record appended to
`docs/DECISIONS_NEEDED.md`; it re-parses and the register no longer reports an
overdue armed default.

Every number in its measurement block reproduces at source in this venv today —
the five Big Five floats at `Personality.py:135-139`, the import-time singleton
at `:731`, `enable_emotional_state` default True at `UnifiedBrain.py:264`, the
wiring at `:4093`, and **zero callers** of `answer_question` /
`get_system_prompt` / `_generate_with_llm` anywhere in `experiments/`.

**The finding is the price, and it is worse than the entry states.** The
default's text leans on one mitigation: *"The spec registered this morning
proceeds on its own clock and REPORTS whatever it finds either way."* **That
spec does not exist.** Verified three ways:

```
registry             258 specs; the LG family is LG.00..LG.14 and stops there
grep registry*.py    ZERO hits for answer_question / get_system_prompt /
                     'character sheet'; the only 'persona' hits are one
                     unrelated wiped-persona twin control
git log -S'Spec("'   since 2026-10-09 exactly ONE spec was registered:
                     LC.08 (da9ece0, 2026-10-20). No other registration exists.
```

So the contradiction is live in shipped default code, the register entry that
held it is now closed, the queue row that owns the registration has broken a
dated promise on it, and **no instrument can ask for the repair** — an
unregistered spec has no id, which is the one hole `coverage` was built for and
the one it cannot see. This is the third time in eight audits that a
predecessor's declared act turned out not to have landed.

## RANK 4 — this organ's own builder orders parsed to ZERO items, for the third time in nine days

`run status` prints, verbatim:

```
STEERING-PAGE ORDERS — 13 item(s) on 2 of 3 declared page(s)
  docs/OVERSIGHT.md: no `## FOR THE BUILDER` heading
```

The 159th audit titled its section `## 9. FOR THE BUILDER`. The reader matches
the bare heading, so **every order this organ wrote on 10-22 was invisible to
the check that asks whether an order names a spec the runner would refuse** —
and that check did fire on the other page this morning (`docs/PROGRESS.md` item
4, `SUBJECT T2.14 BLOCKED — T1.08 (FAIL)`). Measured across the last twelve
revisions of this file, the exact heading is present on 10-10, 10-11, 10-13,
10-19, 10-20 (×3), 10-21 (×2) and **absent on 10-17, 10-18 and 10-22**. It is
self-inflicted, it recurs, and it costs this organ the only automated check on
its own orders. **This page uses the exact heading.** The durable repair — make
the reader accept a numbered or prefixed heading, so a renumbering cannot blind
it — is builder item 2 below; it is the same class as the 149th audit's finding
that six orders renumbered `B0`–`B5` parsed to zero.

## RANK 5 — the project's live steering page is correctly stamped STALE, and six of its eight builder orders are already dead

`docs/PROGRESS.md` carries three stacked staleness banners, the newest written
by the Review at 12:37 today (*"docs/PROGRESS.md itself last moved 77h ago
against a 25h cadence"*). That stamping is honest. The problem is that the
builder now fires hourly and re-reads the page, so I checked its `FOR THE
BUILDER` block item by item against the live tree:

| item | order | state today |
|---|---|---|
| 0 | act on 28.91 free GPU-hours that *"expire Saturday 2026-10-24"* | **EXPIRED.** `2026-W42` closed at **2.078 h of 30.0**; 27.9 h went unbought |
| 1 | `LC.08` is registered, reachable, unbought (14.40 core-h) | **LIVE** — no ledger row; the only true order on the page |
| 2 | repair the `t306` probe `V2` self-validation | **DONE** `dfb71d1` (10:27 today) |
| 3 | `NE.01` is the right call | **DONE** — attempt 4 FAIL, `2026-10-25T10:44:05` |
| 4 | port `T2.14`'s floor + selftest into `T2.05` | **DONE** `b6069dc` (10-19), run harvested `427ff48` today |
| 5 | choose an option from the regate-dirty-tree row | **DONE** `be3178b` (Review, 12:29 today) |
| 6 | `T0.22`'s *"All nine ran"* guard comment | **DONE** `1001185`; the literal is gone from the tree |
| 7 | `longrun_unbound` is six specs, and `T0.32`'s live row is attempt 9 | **HALF-STALE** — `T0.32`'s live row is **attempt 11**, `2026-10-20T20:51:57`, which postdates the repair the item warns about |

The builder is already paying for this: its 10:07 slot summary records *"PROGRESS
item 4 was a stale carry-forward — I verified before touching anything, and item
5 likewise was done at `e735f36`."* That is the right behaviour and it is not
free; it is a verification tax on every slot, levied by a page that is rewritten
only by a Review sitting. No instrument prices it, and I am not proposing a
gate — a stale carry-forward is legal. It is reported because item 0 is the
dangerous shape: a **live-sounding order to spend a resource that no longer
exists**, at the top of the page, under a heading the builder reads first.

## RANK 6 — the stale CONDUCT-DESK class grew to seven, oldest 32 days

`decisions --check` lists eleven `CONDUCT-DESK` entries — desk-executable, *"execute
it, report it, do not ask"* — of which **seven are past their own dates**: `D33`
(32 d), `D35` (31 d), `D38` (21 d), `D40` (15 d), `D39` (10 d), `D46` (8 d),
`D47` (1 d). The 159th audit reported six; `D47`'s date passed on 10-24 while
every organ was stopped, so the class grew by exactly the mechanism RANK 1
describes. `decisions_default_action_expired` reads **0, at its floor**, with the
tool's own note that the class *"has read >= 1 since 2026-09-23"* — the key dates
the counter, not the breach.

These are not the owner's and they are not mine: a `conduct` entry is the
Review's to execute. I report the count and name the arithmetic rather than
executing them, and note the one thing that has changed since the 159th: the
NEWEST of the seven had its own headline numbers shown by that audit to be a
reading taken four days before the freshness claim they were quoted under, so
these seven now include one whose evidence this organ has formally corrected.
(The literal date is deliberately omitted here: a date written beside a `D` id
on this page makes the steering reader report a deadline mismatch against the
register, which is exactly what the 159th's own table triggered.)

---

## Section-by-section, including the clean results

**1. Integrity of the ledger — CLEAN, and this is the strongest result on the
page.** All **106** PASS rows checked mechanically: every declared
implementation file exists in `experiments/tests/`, **every recorded `commit`
still resolves in git** (0 dangling), and every spec declares a `control`. The
two that read controlless are not: `T0.01` and `T0.10` carry explicit
`NoControlByDecision` objects quoting the 52nd audit's reasoning. The ledger
holds 160 rows against 258 registered specs, so 98 specs have no row at all —
unchanged in shape.

**2. Thresholds and controls over time — NO LOOSENING FOUND.** `git log -p
--since="7 days ago"` over `registry.py`, `registry_expansion.py` and
`experiments/tests/` is 3,015 lines across 16 commits. Every numeric change I
could find moves in the strengthening direction or is a cost re-declaration with
a measurement attached:

- `dfb71d1` (today) replaces the `t306` probe's `abs(diff) > 5e-5` — a tolerance
  exactly equal to 4-dp rounding's maximum legal error, so one ULP decided it —
  with round-to-recorded-precision exact compare plus a 7/7 known-answer selftest
  that must fire on three genuine half-point mismatches. **Strictly stronger**,
  and the row had refused a widened epsilon in advance.
- `b6069dc` adds two VOID gates to `T2.05` (`floor_selftest_ok`,
  `task_not_trivial_all`) and two smoke assertions. No bar moved.
- `1ee9c76` re-declares `W1.02` `cpu<2h → cpu<10min` — a cost class, justified in
  the diff with its own measured pilot (0.030 s/decision × 28,000 ≈ 840 core-s),
  and it **corrects** the ruling's own arithmetic rather than inheriting it.
- `RANDOM_DWELL_MAX` reads **0.0185** live (set 2026-09-22, n-aware order
  statistic); the `0.02` figures in prose are the historical bar the `T3.06` VOID
  was read against. Tightened, not loosened.

One change deserves naming without an accusation. `31a3fd0` removed `LC.08`'s
live ledger read of `LC.03`'s row — correctly, because `T0.22` P9 forbids it —
and replaced the provenance check `seat_row_ran_at != SEAT_ROW_RAN_AT → VOID`
with five pinned aggregates in `SEAT_RECORDED`. The commit argues the pin is
fail-safe because a forbidden re-run would regenerate the artifacts and mismatch
the pins. That argument is sound for a **re-run** and silent about an
**amendment**: an amended `LC.03` row leaves the artifacts untouched and the
pins untouched, so the replay gate can no longer see it. This project amends
rows routinely (`run status`: *"Amendments are not runs and are not counted"*).
`LC.08` holds no certificate, so nothing is invalidated; it is builder item 3.

**3. Drift from the goal — no drift in the day's work; the converse is the
problem.** Three builder slots in the last 24 h, all `rc=0`:

| slot | unit | which `GOAL.md` sentence |
|---|---|---|
| 10:07 | `NE.01` attempt 4, re-run foreground after the predecessor's child was killed at 8 min — FAIL, metrics byte-identical to attempt 3, all 5 lives die of hyperthermia | *"the needs are a real control problem"* — homeostasis, the thing that makes living-in-it costly |
| 11:07 | `t310` ruling executed in full: `null_admissible` admits a target for being **unreadable**, so `shape`'s retention refutes the near-saturated-target reading and relocates the deficit to the 128-d pooled substrate | serves the ladder's trustworthiness, not `GOAL.md` directly |
| 12:07 | `T2.05` attempt 5 harvested — FAIL, and `nonlinear_headroom` **negative on all three seeds** (−1.97/−2.23/−2.19), so the pre-registered second branch fires: **the venue is near-linear and the venue is the defect** | *"learning its world by living in it"* — this is the day's real movement |

Two of four units served `GOAL.md`; two served the instruments. The 12:07 slot is
the first genuine frontier result in a fortnight: a 50-day-old unattributable
`FAIL` is now **attributed**, by measurement, against a byte-unmoved bar. No
PASS was bought (106 → 106), and that is the honest shape of the result.

The converse is where the drift lives, and it has not moved. `coverage`:
**0 commitments with no declared spec**, **3 CLAIM-DEAD** (smell,
shelter/building, thermal — every claim spec parked or foreclosed), and **14
commitments with live claim specs and nothing passing**. The three the prompt
names as most likely to be neglected read: curiosity **2 of 12** passing,
one-brain/unison **1 of 28**, learning-across-lives **2 of 10**. `NO-LIVE-PATH`
stands at 6 distinct holes, every repair a **registration**.

**4. Is the builder alive and productive — YES, as of this morning, after ~72
dark slots.** Iterations in the last 24 h: **3**, all `rc=0`, PASS delta **0**
(106 → 106). Before 10-25T10:07 the last `rc=0` was 10-22T17:07: the log shows
`PACING` skips to 18:07 and then an unbroken wall of *"STOPPED at 90..100%
weekly usage — all agents paused until the owner resumes"* through 10-25T09:07.
The week reset this morning and the meter now reads `week:all models` **15%**.
`dark_slots` reads **0** at its floor, because it is a trailing-streak counter
and the streak ended — the blindness the 159th audit routed as
`dark-slot-ratchet-is-a-streak-so-an-ENDED-blackout-reads-zero` (OPEN,
DUE 2026-11-04). Nothing in `experiments/` holds any trace of the 72 slots.

**5. Compute honesty — the waste is now five consecutive weeks and the cause is
not the loop.**

```
2026-W41   0.000 h of 30.0 drawn      30.0 h expired unbought
2026-W42   2.078 h of 30.0 drawn      27.9 h expired unbought
2026-W43   1.777 h drawn so far       ~28.2 h remain (resets Sunday)
```

The single W43 job is `jack-ladder-1792923207` (T4, 1.78 h, `ok: true`) and it
bought the `T2.05` attempt-5 row above — a GPU hour with a ledger entry and a
pre-registered reading to show for it, which is the standard. Against that,
`gpu_hours_no_verdict` reads **53.24 h total**, of which **`D1.0` is 33.78 h
across 2 attempts and 0 verdicts** — 63% of all unredeemed GPU time sits on one
spec, unchanged. `projection_vs_estimate` gained one row this week (`T2.05`
`@2d6a217`: authorised 1.0 h, declared `est_hours` 1.2 h) — an over-estimate
against its authorisation, declared, not hidden.

**6. Stuck decisions — the `D1` disease is absent; the backlog is procedural.**
`MEANS-ESCALATED`: **zero** — no fork that a measurement could settle is sitting
on the owner's desk. `UNDECLARED`: **zero**, so there was nothing for me to arm
this audit and the ratchet reads `0/0`. Armed and live after my firing: `D51`
(due 11-03, blocks `PS.09`) and `D53` (due 11-08, blocks `SO.07`), both with
monotone defaults that move no threshold. Two entries were minted by the Review
during my sitting (`D53`, `D54`) and both are the same shape — a queue row whose
only remaining repairs are bar moves forbidden to every desk that could make
them. That is now three live instances of the class, which the Review itself
names as a class rather than a coincidence. Nothing I found was acted on without
being recorded. `CONDUCT-MISFILED?` fires on `D44` and is wrong for the reason
the entry pre-empts: reclassing it would hand a desk the authority to decree
Jack's character.

**7. Bakeoff hygiene — two standing violations, both recorded, both at their
ratchet floors, and one has now lost every door it had.** `champions --check`
reports `VERDICT-IS-A-VOID` on the **Learning core** seat: `wm-latent` is held
**BY VERDICT** — the file's strongest marking — off `LC.03`, whose ledger row is
a **VOID**. That is precisely the "VOID treated as a verdict" this section asks
about. It is not hidden: the seating carries its single-arm caveat on its face.
What has decayed is contestability — `TRIGGER-UNREACHABLE`: all three
pre-registered re-open triggers are closed doors (`LC.07` PILOT-BLOCKED, `LC.03`
VOID-FORECLOSED, `UB.10` VOID). The **World** seat is worse in kind:
`VERDICT-UNDECLARED` + `TRIGGER-UNDECLARED` — held BY VERDICT and naming no
deciding run and no re-open trigger at all. `champions_trigger_debt = 3` and
`champions_unwinnable = 4` are unchanged and at floor. No winner was chosen
inside a noise margin this week; the one anchor-decided certificate (`T4.06`)
prints its own margins as fractions of the anchor's seed spread, including the
two arms that regress.

**8. The honest summary — are we closer to a curious humanoid that climbs the
ladder?** **Yes, by one real step, and the step is subtraction rather than
addition.** `T2.05`'s world-model defeat by a single linear map sat
unattributable for 50 days; today a pre-registered reading on a byte-unmoved bar
established that the **venue** is near-linear — the 1-NN non-linear reference is
~3× worse than ridge on identical features — so demanding the world model beat
ridge was demanding it match a near-Bayes-optimal predictor. We now know we were
grading on the wrong world, and we know it by measurement. No tick turned green
for it. That is what progress looks like here and it should be said plainly.

Against that: 106 of 258 specs demonstrated, unchanged across three `rc=0` slots;
three of the owner's own constitutional commitments still CLAIM-DEAD with no
registered successor; the headline unison claim at 1 of 28; the first end-to-end
certificate still unbought over a costume nobody ruled on, now with no spec that
will ever measure it; and seven standing certificates resting on recorded
non-PASS foundations, four more than the floor permits. The machinery remains in
excellent and improving health — the queue's violation class is empty again, the
re-buy lane re-derives twelve times a day, the ledger's 106 certificates all
check out at source. **The thing the machinery exists to measure moved once this
week, and it moved because somebody read a negative number carefully.**

## FOR THE BUILDER

Ordered cheapest and most perishable first. Every number re-derived this audit.

1. **TRANSCRIBE `D44`'s FIRING to `docs/DECISIONS_RESOLVED.md`.** I appended the
   firing record to `docs/DECISIONS_NEEDED.md`; `decisions --check` now reads
   *"34 of 35 firing(s) ... UNTRANSCRIBED: D44 (invisible to the RECORD channel
   and to firing_coverage until transcribed)"*. Same-day, per the `fadf712`
   precedent (`D41`/`D42`, 153rd audit). Bill: 0 certificates.
2. **Make the steering-page order reader accept a numbered or prefixed `FOR THE
   BUILDER` heading.** `run status` prints *"docs/OVERSIGHT.md: no `## FOR THE
   BUILDER` heading"* and counts **13 items on 2 of 3 declared pages**; this
   organ's orders were invisible to the order check on 10-17, 10-18 and 10-22
   (measured over the last twelve revisions of the file). Match on the heading
   TEXT containing `FOR THE BUILDER` rather than on the exact line, so a
   renumbering cannot blind the reader again. Strictly widening the reader's
   domain, so it can only find more orders, never fewer. Bill: price it — the
   reader lives in `experiments/run.py`, which several `IMPL_DEPS` name.
3. **`LC.08`'s replay gate cannot see a ledger AMENDMENT.** `31a3fd0` correctly
   removed the live ledger read (T0.22 P9) but also removed the
   `seat_row_ran_at` provenance check; `SEAT_RECORDED`'s fail-safe argument
   covers a forbidden **re-run** (artifacts regenerate, pins mismatch, VOID) and
   not an **amendment**, which leaves artifacts and pins identical. Either record
   that gap in the constant's comment as a declared limit, or restore the
   provenance assertion in a form P9 permits. `LC.08` holds no certificate, so
   the bill is 0 and the spec_sha is free to move.
4. **`LC.08` is still the cheapest real science on the board** — registered
   `da9ece0`, five dependencies all PASS, **no ledger row**, 14.40 core-h of CPU,
   0-certificate staleness bill. It is the only surviving live order on
   `docs/PROGRESS.md` (item 1) and it answers a live seat question: is 17.3% of
   the seated `A4` arm dead weight? Not a GPU unit, so this week's quota does not
   gate it.
5. **Do NOT act on `docs/PROGRESS.md` item 0.** It orders action on *"28.91 free
   GPU-hours [that] expire Saturday 2026-10-24"*. That window closed:
   `2026-W42` ended at **2.078 h of 30.0**. The live figure is `2026-W43` with
   **1.777 h drawn** (the `T2.05` kernel) and ~28.2 h remaining. Items 2–6 of
   that block are discharged; item 7's own caveat is stale (`T0.32`'s live row is
   attempt **11**, `2026-10-20T20:51:57`, not attempt 9).
6. **Price every new dispatch against the T1.08 wall before quoting GPU
   headroom.** Both specs that could absorb hours (`T2.03`, `T2.14`) read
   `needs T1.08`, which is FAIL at attempt 5 with a sixth attempt prohibited in
   writing, and both are in the `PASS-ON-DEAD-DEPENDENCY` set for exactly that
   reason. Your own 12:07 slot got this right — *"nothing is honestly buyable
   until the T1.08 decision lands"* — and that reasoning should survive into the
   next slot rather than being re-derived.

## FOR THE OWNER

**1. `D44` — I FIRED YOUR ARMED DEFAULT, AND THE REASON TO READ THIS ANYWAY IS
THAT THE DEFAULT'S OWN SAFETY NET IS NOT THERE.** You did not rule by the date
in the register, so the pre-registered default fired on the first legal day: no
ruling, the character sheet stays exactly as it ships, nothing is documented.
Jack still ships five Big Five floats, five `core_values` strings including
`'Kindness -- genuinely cares about others'` and `'Curiosity above all'`, and a
2,585-character persona, all on by default through his shipped answer path,
while `GOAL.md` says in your words *"His kindness is not decreed."* The firing
decrees nothing and one sentence reverses it either way.

What the firing had to add: the default's text promised that *"the spec
registered this morning proceeds on its own clock and REPORTS whatever it finds
either way."* **That spec was never registered** — the registry holds 258 specs,
the language family stops at `LG.14`, and the only registration since 2026-10-09
is `LC.08`. So there is now no instrument that will ever measure whether the
costume matters, and because an unregistered spec has no id, no tool in this
repo can ask for it. The desk's split recommendation stands on the record: rule
the temperament deliberate and document it, rule the decreed values a breach and
pay the 18-certificate bill once on that half alone.

**2. NO-DECISION — the usage stop does not pause what you were told it pauses,
and in the pause just ended it cost a Tier-0 certificate.** Recorded as an
evidence addendum on the open conduct entry about the stop's missing record, so
it reaches the ruling rather than only this page. For ~72 consecutive builder
slots the gate held the builder, both Review sittings and two of this organ's
audits — and the re-buy lane ran twelve sweeps a day straight through. At
`2026-10-23T04:48:54` that lane turned `T0.07` from PASS to FAIL on a
reproducibility conjunct (0.272 against a 0.20 cap), which took
`pass_on_dead_dependency` to **7 against a floor of 3** and left the red owned
by nothing but a prose mention. **Nothing here is on your desk to rule** — the
re-derivation is routed with a date and the premise correction is filed — but
the shape is worth one sentence of your attention: the pause makes the project
look quiet while the ledger is still being written.

**3. NO-DECISION — five consecutive weeks of free compute have now expired
substantially unbought, and this week the cause is not the gate.**
`2026-W41` closed at 0.000 h of 30.0 and `2026-W42` at 2.078 h of 30.0. The
builder is awake, the meter reads 15%, and ~28.2 h remain this week — and the
honest reason nothing is being dispatched is that the two GPU specs that could
absorb hours both sit behind `T1.08`, whose estimator question is on a desk.
This is the same fact your open entries price as a blackout cost; it is now
visible as a **pipeline** cost instead, which is a different repair.

**4. NO-DECISION, reported because the count grew while nobody could sit.**
Seven `CONDUCT-DESK` entries are past their own dates, oldest 32 days. They are
desk-executable by construction — *"execute it, report it, do not ask"* — so
they are not yours and they are not mine. The reason they are on your page at
all is that the instrument that counts their expiry reads **0, at its floor**,
while its own note records that the class has read at least one since
2026-09-23. A backlog that cannot redden anything is the shape this project has
learned to distrust.
