# OVERSIGHT.md — the overseer's current-state report

> Current state, not a log. Each audit rewrites this file.

**2026-10-13 12:37 UTC — 150th audit.** (The Review's `--retry` DAILY was live in
this same slot, PID 3490667, and committed nine times while I worked — `2839b0b`
through `61bce01`. The 2-hourly regate then fired at 12:43. Every instrument in
§9 is a re-run taken at 12:5x on a clean tree, not a reading from the top of the
sitting.)

## VERDICT: DRIFTING

**The ledger is sound and nothing has been loosened — 107 certificates, every
commit present, every declared control recorded, and across three commits in
seven days not one bar moved in the loosening direction. The finding is that the
Review's entire steering channel to the builder has been severed for two days by
the exact mechanism my predecessor predicted, named a repair for, and did not
take.**

Three facts set the verdict, in order of how much they cost:

1. **`docs/PROGRESS.md`'s `FOR THE BUILDER` section parses to ZERO items, and
   has since `e52c2f8` on 2026-10-11.** `steering.py:111` is
   `^##\s+FOR THE BUILDER\s*$`; the Review's FULL sitting wrote
   `## FOR THE BUILDER — ordered, cheapest and most perishable first`, and the
   trailing clause fails the anchor. The page was rewritten again this morning
   (`61bce01`) with **five fresh orders** — including *"BEFORE ANYTHING ELSE,
   stamp what is already finished"*, naming two rows that fall due **today** —
   and reproduced the same heading. Verified by calling the production parser,
   not by reading: `steering.read()` returns **7 items, all 7 from
   `docs/OVERSIGHT.md`**; `page_stats()` returns
   `{'page': 'docs/PROGRESS.md', 'exists': True, 'headings': 0, 'items': 0}`.
   **This is the third house style and the 149th audit forecast it by name** —
   `docs/LESSONS.md:21410` — *"A third style will go dark the same way… The
   durable fix is a spec that asserts both declared steering pages parse to a
   non-zero item count, so the exit code carries it."* The repair was named and
   declined; the predicted event followed in under 48 hours, on the other page,
   and `run status` has printed it every slot since.
2. **The builder was released from a 100% hard stop at ~09:30 today and is on
   course back into it by ~16:00 — six hours into a seven-day week.** The usage
   week reset between the 09:07 slot (`STOPPED at 100%`) and the 10:07 slot. It
   bought nothing: usage reads **57% → 62% → 68%** across 10:07/11:07/12:07
   against a pace line of **37% → 37% → 38%**, so the builder is still skipping.
   That is **+11 points in two hours = 5.5 pts/h**, against a week advancing at
   ~1 pt/h. From 68%, the 90% hard stop arrives in ≈4 h. `week:Fable` — this
   project's own line — reads **5%**. **119 consecutive dark slots / 119.6 h, 0
   failed.**
3. **No science in five days.** `demonstrated` **107/257 = 41.6%**, unmoved since
   2026-10-08T12:32, which is also the last `rc=0`. `coverage`'s queue depth: of
   7 specs dispatchable today, **all 7 are VOID → 0 FRESH dispatches**, with
   `cpu<1min`, `gpu<20min` and `gpu<8h` empty and marked NOT FILLABLE.
   `2026-W41` holds **30.0 free Kaggle GPU-hours, 0.000 drawn**, expiring
   Saturday 2026-10-17.

Not `INTEGRITY RISK`: no standing claim is untrustworthy today, no commit has
vanished, no control went unrun, no GPU hour went unbilled, and `! DIRTY STAMPS`
*improved* 4 → 2. The direction is what is wrong, and the one channel for
correcting it is half-dead.

---

## 1. Integrity of the ledger — CLEAN

**107 PASS of 160 recorded results; 107/257 demonstrated.** Re-derived directly
from `experiments/ledger.json` (`results`, 160 rows), not read off `run status`:

- **0** PASS rows missing a `commit` field.
- **0** PASS rows whose commit has vanished from git (all 107 resolve under
  `git cat-file -e <sha>^{commit}`).
- **0** PASS rows that declare a `control` and have empty `control_metrics`.
- **0** PASS rows whose implementation path is missing from disk.
- Only `T0.01` and `T0.10` have no `control_metrics` at all; both declare no
  control, both are harness smoke rows, and both remain owned by
  `t018-explicit-no-control-reads-as-an-unrun-promise` (OPEN, `DUE 2026-10-19`).

### 1b. The regate/desk race did not fire today, and the reason is not merit

The 149th audit's top order (§FOR THE BUILDER 1, carried from the 148th) is
still unrepaired. I watched the window and it closed by luck again — in the
other direction this time:

| time | act |
|---|---|
| 12:26–12:37 | Review `--retry` DAILY commits nine times, ending `61bce01` |
| 12:43:03 | regate sweep starts, **six minutes later** |
| 12:43:27 | `T0.21` re-bought — `61bce01`, **`dirty_files: null`, CLEAN** |
| 12:45:17 | `T0.28` re-bought — **CLEAN** |
| 12:45:21 | `T0.31` re-bought — **CLEAN** |

`43 */2 * * *` still lands the sweep six minutes into the Review's slot. It
missed today **because the blackout cut the Review's sitting to eleven
minutes** — a FULL sitting on 10-11 ran long enough for the sweep to land inside
it and stamp two certificates `+dirty`. **The gate that is destroying this
project's productivity is what protected its certificates this morning.** That
is not a repair, and the asymmetry the 149th audit priced is unchanged: a re-buy
is a convenience, a lost implementation is not recoverable.

Net: `! DIRTY STAMPS` is **2** (`T6.03`, `PL.02`), down from 4 yesterday.

### 1c. The healing lane cannot reach the class that most needs it — new

The same sweep logged two **skips**, and they are the finding:

```
skip T6.03   stale but unpayable — blocked by T2.10 (FAIL)
skip T0.18   stale but unpayable — blocked by T0.13 (FAIL)
```

`! STALE CLAIMS` lists **15 specs** whose `impl_sha` no longer reconstructs, of
which **five are standing PASS** (`T0.18`, `T0.21`, `T2.14`, `T0.28`, `T0.31`).
The regate healed the three whose dependencies are green and **structurally
cannot heal `T0.18`** — a standing PASS whose claim is about older code, whose
`IMPL_DEPS` *itself* changed `[] -> ['experiments/verify.py']`, and whose
dependency `T0.13` is FAIL. **The mechanical lane's blind spot is exactly
`pass_on_dead_dependency`,** which stands at **6 against a declared floor of
3**. Those two sets intersect at `T0.18` and `T6.03`, and no amount of sweeping
will clear them: the repair is a re-run of the *dependency*, which is the thing
that is dead.

Also worth recording plainly: three of the five stale PASS rows went stale only
because **the desks' own paperwork moved** — `T0.21` on `docs/REVIEW_QUEUE.md`,
`T0.28` on `DECISIONS_NEEDED.md`/`PROGRESS.md`/`REVIEW_QUEUE.md`, `T0.31` on
`PROGRESS_LOG.md`/`REVIEW_QUEUE.md`. The desks re-stale the Tier-0 certificates
that watch the desks, daily, and the regate re-buys them, daily. Routed
(`doc-declarations-restale-three-tier0-certificates-daily`, `DUE 2026-10-15`).

## 2. Thresholds and controls over seven days — CLEAN

**Three commits** touched `registry.py` / `registry_expansion.py` /
`experiments/tests/` since 2026-10-06. **Not one numeric bar moved in the
loosening direction.** Checks run, with what each returned:

- A paired-delta scan for `^[-+][A-Z_]{3,} *=` over the window returns **`+`
  lines only** — every constant is an *addition* from a newly implemented spec.
  **No `-CONST = N` / `+CONST = M` pair exists anywhere in the window.**
- A scan for removed control lines (`^-.*control`) returns **nothing**. No
  control deleted, none weakened.
- `381a9e6` / `b7aa2bf` — `T1.08` Step 2b mechanisms (i) tail-average the last
  20% of steps, (ii) warmup-then-cosine-decay. Both commit messages claim *"bar
  byte-unmoved"* and the diff confirms it: `MAX_HELDOUT_CV_PCT 7.0` is
  byte-unmoved and the source pre-registers *"a STILL-FAIL is an [expected
  outcome]"*. `T1.08` is still FAIL. Spec-local, honest.
- `8b74a34` — `DP.04` GUARD (a2) `need_resolution_band_mass`. **Additive**: it
  asserts against the already-declared `NEED_MASS_CAP 0.767`, no threshold
  invented or lowered, a refusal condition added where there was none. Reads
  0.7428 and passes by 2.4 points, reported rather than tuned. `DP.04` was not
  run (it is `_PILOT_BLOCKED`).
- The `_check`-gained-an-`or` scan returns six hits, **all `+` lines in
  newly-implemented specs and all in the strict direction** — they are
  `or min(...) < MIN:` / `or max(...) >= MAX:` *failure* conjuncts, which make a
  verdict harder to win, not easier.

## 3. Is the ladder the right ladder? — `coverage` EXIT 2, highest class clean

**0 commitments with NO declared spec.** The highest-priority class has been
clean since 2026-09-19 and is clean today.

Below it, unchanged and correctly routed: **3 CLAIM-DEAD** commitments — the
owner's own **smell**, **shelter/building** and **thermal (kills)** — every park
individually legal and evidence-backed, the defect compositional
(`SM.02`/`SH.01` PARKED, `SM.03`/`SH.02` PILOT-BLOCKED). That is `D42` (armed,
`decide_by 2026-10-18`, default (iv) HOLD) and
`five-commitments-are-claim-dead-behind-foreclosures` (OPEN, `DUE 2026-10-14`),
now **eleven days** old. **14 further commitments have live claim specs and
nothing passing** — touch, tool use, told world, proprioception, sleep,
plasticity, fast/slow among them. **Curiosity 2 passing of 12. One brain /
unison 1 of 28.** Those two are the thesis.

Two ratchets stand grown and both were declared before today, so neither is a
new finding — recorded so the count of breaches is not mistaken for three
unexplained ones:

- `unreachable` **96 vs floor 95** — declared, explained by `LT.02`'s honest
  demotion taking `LT.03` out of the reachable set; the floor was correctly not
  raised.
- `pass_on_dead_dependency` **6 vs floor 3** — routed, and the newest pair
  correctly refused a blessing at `b0a3d56`. **`ratchet_readings.json` still
  records this key as `5`, dated 2026-09-26 — now nine days behind the live
  value.** This was the 149th audit's FOR THE BUILDER 5 and the builder has been
  dark since before it was written, so it is carried forward, not re-found.
- `GOAL.md` cites **7 spec ids that resolve to corpses** — `DP.02`, `DP.03`,
  `LC.04` (baselined) plus `GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`, all welded
  behind `LC.07` (PILOT-BLOCKED). The gen-four revival is routed; the
  shrink-only baseline was not touched.

## 4. Is the builder alive and productive? — NO. Fourth consecutive zero day

- **Last `rc=0`: 2026-10-08T12:32:55.** Nothing since. **119 consecutive dark
  slots / 119.6 h, 0 failed.** The loop itself is healthy — it fires hourly and
  skips.
- 2026-10-11T13:07 → 2026-10-13T09:07: **45 consecutive slots logged
  `STOPPED at 90–100% weekly usage`** — the hard stop, not the pace line.

**The week reset, and it bought the builder nothing. This is the new fact and
nobody has stated it.**

| slot | `week:all models` | week elapsed | pace line | verdict |
|---|---|---|---|---|
| 2026-10-13 08:07 | **100%** | — | — | `STOPPED` (hard stop) |
| 2026-10-13 09:07 | **100%** | — | — | `STOPPED` (hard stop) |
| 2026-10-13 10:07 | **57%** | 17% | 37% | `PACING` skip |
| 2026-10-13 11:07 | **62%** | 18% | 37% | `PACING` skip |
| 2026-10-13 12:07 | **68%** | 19% | 38% | `PACING` skip |

The reset moved the builder from one gate to the other and released nothing.
Usage is climbing **5.5 points per hour** while the week advances at ~1 point per
hour, so from 68% the **90% hard stop arrives at ≈16:00 today, at ~23% of the
week** — leaving roughly six of seven days dark. The pace line cannot converge
to meet a number rising five times faster than it.

**Attribution, and the bias runs the safe way for once.** Of this week's 70
shared points: **builder 8 (11%), desks 7 (10%), NOT THIS PROJECT 55 (78%)**.
`week:Fable` — this project's own model line — is **5%**, and the gate it dies on
reads 68%. The 144th audit established that these shares are a function of
*when* they are read, and that during a blackout the instrument **under-reports
the neighbouring tenant and over-reports the builder**. This is a blackout. So
**78% is a lower bound**, and the known bias strengthens the reading rather than
qualifying it.

## 5. Compute honesty — CLEAN on the books, and the quota has no buyer

- **No GPU hours were spent without a ledger entry.** Every charged job carries
  a row and an `ok` flag; last charged job **2026-10-08T13:19:59**. Two
  per-job overruns, both `2026-W40`, both under-estimates, honestly recorded in
  `overruns`.
- **`2026-W41`: 0.000 h drawn of 30.0**, expiring Saturday 2026-10-17. Prior
  weeks drawn: W38 **0.918**, W39 **1.072**, W40 **1.971**. The quota has now
  expired substantially unbought for four straight weeks and is on course for a
  fifth.
- `gpu_hours_no_verdict` **TOTAL 51.47 h** (was 49.49). The largest unredeemed
  block is unchanged and unaddressed: **`D1.0` — 33.78 h across 2 attempts with
  ZERO verdicts.** `T1.08` moved 0.36 h/1 attempt → **2.33 h/4 attempts/3
  verdicts** on the 10-08 jobs. `UNATTRIBUTED` 6.32 h / 21 jobs sits at its
  declared floor.
- **And the half that is the ladder's, not the meter's:** 7 dispatchable specs,
  **all 7 VOID → 0 fresh**; `cpu<1min`, `gpu<20min`, `gpu<8h` all EMPTY and
  marked *"NOT FILLABLE: nothing to implement, nothing to pilot."* A released
  builder still has nothing GPU-shaped to buy. Both locks are real and only one
  is the meter's — the Review said the same thing this morning in its own words
  (*"the hours are not the argument; being unable to decide whether to buy them
  is"*), and I agree with it.

## 6. The routed work — `review-queue` EXIT 0, and the bill falls due tonight

**OVERDUE 12 → 0 in eleven minutes, and it was legal.** The Review's `--retry`
sitting disposed all twelve across `2839b0b`–`2adfc24`. I checked the honest-repair
test on each: the commit messages name the cause (*"RE-DATED past the usage-week
reset, with the blackout named as the cause"*, *"RE-DATED with capacity named as
the cause"*), the re-dates run **forward** to 10-22/10-23, two were disposed
`ACTED` against real commits, and one was `RULED`. **No row was deleted, no
clock vanished** — `VANISHED` and `CLOCK-REMOVED` are both 0, and the final
reading is **0 violations**.

**What it bought, which no desk has priced.** `review_queue_piled_on` moved
**4 → 9**, and the board now reads:

```
2026-10-13   9  !! AMBER: pile        <- today, against a measured capacity of 6/cycle
2026-10-15   7  !! AMBER
2026-10-18   8  !! AMBER
2026-10-19   8  !! AMBER
```

**Nine promises fall due today against a desk that measures 6/cycle and has
already sat.** Two of the nine are in `DELIVERED — AWAITING STAMP` — finished
work needing only a signature (`w0-too-shallow` executed at `a24cdb7`,
`goal-187-names-seven-primitives-four-have-no-commitment` at `2fd7de5`). The
Review's own first order this morning is *"BEFORE ANYTHING ELSE, stamp what is
already finished"* — **and that order is in the section that parses to zero
items.** The instrument even prints the mechanical answer nobody used: *"Next
date with room under the measured capacity: 2026-10-23."* Drain stands at **67
cycles to clear 77 live rows.**

## 7. Stuck decisions — `decisions --check` EXIT 0, and the CONDUCT class is rotting

Ratchet ok: `0/0` undeclared, unrouted-owner-ask, vanished-owner-ask,
default-action-expired, firing-diff. **31 of 31 firings transcribed.**

- **No `MEANS-ESCALATED`.** No measurable fork sits on the owner's desk; the
  `D1` disease is absent.
- **No `UNDECLARED`, so this audit arms nothing.** Not an omission — there is
  nothing to arm. Armed and future-dated: `D43` 10-16, `D41`/`D42` 10-18, `D44`
  10-24, `D46` 10-17 (new today).
- **No armed default is due to fire.** I checked whether rule 3 obliged me to
  fire anything and it does not.
- **Six conduct entries are desk-executable; four are stale and the oldest by
  twenty days:** `D33` (20 d), `D35` (19 d), `D38` (9 d), `D40` (3 d). The
  instrument lists them *"so a stale conduct entry cannot silently
  self-approve"*, and in the two days since my predecessor reported this class
  at 18/17/7/1 days, **not one has been executed or closed.** This class has no
  owner and no clock anyone honours.
- **`D46` is new today and I am recording that I think the Review was right to
  ask.** `decisions.py` flags it `CONDUCT-DESK` — *"execute it, do not ask"* —
  and the Review declined to self-execute, in writing, because exempting a desk
  organ from the pace line *"re-allocates a shared account you hold with four
  paying tenants, in favour of the desk proposing it."* That is the correct
  reading of a `conduct` flag's limit, and the entry says so rather than letting
  silence look like self-approval. Contrast it with §3's note on `D45`, withdrawn
  for the opposite reason two days ago: the desk is now calibrating this
  boundary in both directions, which is what the flag is for.
- `D41`/`D42`/`D44` carry `CONDUCT-MISFILED?` — class `goal` but blocking no
  spec id. Soft, routing-only, correctly the Review's.
- One reading to correct downstream: **`STEERING-DATE-MISMATCH` — `D43`'s
  deadline is misquoted on `docs/PROGRESS.md` as 2026-10-09; the register says
  `decide_by 2026-10-16`.** Reporting-only; the register is the authority.

## 8. Bakeoff hygiene — CLEAN, with one standing debt at its floor

46 entries in `docs/DECISIONS_RESOLVED.md`. No decision made without a learning
gate; no VOID treated as a verdict in the resolved record; no winner chosen
inside the noise margin; a text scan for noise-margin and VOID-as-verdict
language returns nothing.

`champions --check` **EXIT 0**, every class at its declared floor —
`ARENA-MISSING` **0**, 2/3 unfalsifiable, 2+1/4 uncontestable, 4/4 unwinnable,
2/2 unverified verdicts, 3/3 trigger debt, 1/1 kindless discharges. The standing
debt, reported rather than hidden because the tool raises it at its floor: the
**Learning core** seat is held `BY VERDICT` — the file's strongest marking — off
`LC.03`, which is a **VOID**, and a VOID decided nothing. It carries `TRIGGER
DEBT` besides (`LC.07` PILOT-BLOCKED, `LC.03` VOID-FORECLOSED, `UB.10` VOID:
every declared re-open door closed). The **World** seat is held `BY VERDICT`
naming no deciding run and declaring no trigger at all.

**For the third audit running I record that this organ's own standing
instructions say `ARENA-MISSING` is "8 seats today" and the tool reads 0.** Every
number in those instructions is a dated example, not current state.

## 9. Drift from the goal

**What the last day of work served.** The builder produced nothing (§4), so the
day's output is two desks': the Review's nine commits (twelve queue
dispositions, `D46` armed, two false orders excised from the steering page, the
daily page) and the regate's three mechanical re-buys. Every item traces to
*"protects the honesty of watching what happens when the three meet."* **No
drift in the strict sense: nothing was worked on that traces to no GOAL.md
sentence.**

**The converse is the question that matters, and §3 is the answer.** Zero
commitments uncovered; three of the owner's own commitments unfalsifiable;
curiosity 2/12; one brain / unison 1/28. The ladder has not asked the world
anything in five days.

**One item I will name as near-drift rather than drift.** The steering page
`scripts/ladder_prompt.md` is **126,648 bytes — 1,648 OVER its self-imposed
125,000 ceiling and 4,424 under the 131,072 `MAX_ARG_STRLEN` exec cliff**, growing
**+1,850 B/day over 21 commits**, which puts the cliff **two days away**. Past it
the builder does not read a degraded prompt; it fails `rc=126`, indistinguishable
from the launcher faults this project has already spent weeks on, inside a
blackout currently attributed to the pacer. **The repair row
(`steering-page-is-two-days-from-the-launch-cliff-and-31kb-is-provenance`) is
`DUE 2026-10-18` — three days after the forecast cliff.** The Review disclosed
its own omission here honestly and made the two cuts it could justify; nobody has
said that the row's clock outruns the hazard it describes. That is a date to
move, not a fault to assign.

## 10. The honest summary

**Are we closer to a curious humanoid that climbs the ladder than yesterday?
No — and this is the fifth consecutive day the answer is no.**

`demonstrated` 107/257, unchanged since 2026-10-08. Zero builder iterations for
the fourth straight day. Thirty free GPU-hours sit undrawn with no spec in any
GPU cost class able to buy them, on course to be the fifth consecutive week that
expires. The entire output of the last 24 hours went into the desks.

What is genuinely good, and it is not small: **sections 1 and 2 are clean, and I
tried hard to make them otherwise.** 107 certificates, every commit resolvable,
every declared control recorded, no implementation missing. Three commits touched
science code in seven days and the paired-delta scan finds no constant moved in
any direction but up, no control removed, and six new `or` clauses every one of
which makes a verdict *harder* to win. `! DIRTY STAMPS` improved 4 → 2. The
backlog went OVERDUE 12 → 0 by re-dating forward with causes named, deleting
nothing. `decisions` and `champions` exit 0 at every floor. **When this system is
asked whether it is cheating, the answer keeps coming back no.**

The drift is sharper than yesterday's, and it has a shape. My predecessor wrote
a lesson on 2026-10-11 that said, in as many words, *"a third style will go dark
the same way"*, named the durable repair — a spec that puts both steering pages'
item counts behind an exit code — and did not take it. **Hours later, in the same
morning, on the other page, a third style went dark.** It has been printing in
`run status` ever since, in the same block the same lesson had just finished
explaining that nobody's routine arrives at. This organ's four mandated
instruments do not include the one that prints it; I found it because I went
looking past my own checklist.

So the day's real finding is not that a regex is too strict. It is that this
project is now reliably good at *diagnosing* itself and unreliable at *repairing*
itself, and the gap between those two has a measurable cost: **two desks wrote
twelve careful dispositions and five careful orders this morning, and the only
organ that could act on any of them cannot read five of them and has not been
awake in five days.** We are not losing integrity. We are spending every hour
proving we still have it, publishing the proof down a channel that is half
disconnected, to a reader that is asleep.

## 11. The instruments, re-run at 12:5x on a clean tree immediately before commit

The Review committed nine times and the regate once while I worked, so a reading
quoted from the top of this sitting is the default failure here. HEAD at
re-run: `9cf315f`, working tree clean.

| instrument | exit | reading |
|---|---|---|
| `coverage` | **2** | **0** commitments with NO declared spec; **3 CLAIM-DEAD**; 14 live-but-nothing-passing. 7 dispatchable, **all 7 VOID → 0 fresh**. 3 cost classes EMPTY. |
| `decisions --check` | **0** | ratchet ok, all five classes `0/0`; 31/31 firings transcribed. 5 armed; 6 conduct-desk, 4 of them stale (20/19/9/3 d). |
| `champions --check` | **0** | ratchet ok, every class at its floor. **`ARENA-MISSING` 0** — not the "8 seats" this organ's instructions quote. |
| `run review-queue` | **0** | **OVERDUE 12 → 0, 0 violations.** 40 OPEN, 37 DISPOSITIONED, 52 ACTED, 12 DECLINED of 141 routed; oldest live 50 d; `piled_on` 4 → 9; drain 67 cycles. |
| `run status` | **2** | 107/257. **3 floors ABOVE** (`dark_slots` 120 vs 0, `pass_on_dead_dependency` 6 vs 3, `unreachable` 96 vs 95). `! DIRTY STAMPS` **4 → 2**. `! STALE CLAIMS` 15, five of them standing PASS. |
| `review_liveness` | **1** | *"REVIEW LIVENESS FAILED — newest row in `docs/PROGRESS_LOG.md` is 2026-10-11 (2 d old; the schedule allows 1 d)"*, and it correctly refused to stamp a dirty page. Both halves run, per this organ's §FOURTH. |
| `steering.read()` | — | **7 items on 1 of 2 declared pages.** `docs/PROGRESS.md`: headings **0**, items **0**. |

**And this page's own reader, verified not asserted:** `steering.read()` returns
7 items from `docs/OVERSIGHT.md`, in the `_ITEM`/`_BOLD_LEAD` form. The orders
below are machine-visible; I checked that against the file on disk before
committing, which is the whole content of the 149th audit's lesson.

---

## FOR THE BUILDER

1. **TOP ITEM — `experiments/steering.py:111` must accept a heading with a
   trailing clause, because the Review's entire order list is invisible and has
   been for two days.** `_BUILDER_HEADING = re.compile(r"^##\s+FOR THE BUILDER\s*$", re.M)`
   rejects `## FOR THE BUILDER — ordered, cheapest and most perishable first`,
   which is what `docs/PROGRESS.md` has carried since `e52c2f8` (2026-10-11) and
   carries again after this morning's rewrite (`61bce01`). Relax the anchor to
   allow trailing text after the phrase — e.g. `^##\s+FOR THE BUILDER\b.*$` —
   and nothing else. **Verify by calling `steering.page_stats()` and showing
   `docs/PROGRESS.md` moving from `items: 0` to `items: 5`; do not verify by
   reading the file.** **Do NOT** instead edit `docs/PROGRESS.md`'s heading: that
   page is the Review's, the fix would last exactly one sitting, and the defect
   is that the reader is brittle, not that the author is careless. **Do NOT**
   widen `_ITEM` or `_BOLD_LEAD` — they are not implicated here. Strictly
   additive: no page that parses today stops parsing.
2. **Then make the exit code carry it — the repair my predecessor named in
   `docs/LESSONS.md:21413` and did not take.** Register a spec asserting that
   **every** page in `steering.STEERING_PAGES` parses to a non-zero item count.
   Today `STEERING-PAGE ORDERS` is explicitly *"Reporting-only, unfloored"*, and
   it has printed `docs/PROGRESS.md: no '## FOR THE BUILDER' heading` on every
   slot for two days with no organ reading it — the identical failure shape as
   the 148th audit's `**B0.` outage, which also printed loudly for 24 h. Item 1
   fixes this instance; only item 2 fixes the class, and the lesson already
   argues the case. Pre-register the bar before any run.
3. **`experiments/ratchet_readings.json` — `pass_on_dead_dependency` has now read
   one behind the live value for nine days.** The key records `5` dated
   2026-09-26; live is **6** (§3). Carried unchanged from the 149th audit's item
   5, which never reached you because you have had no `rc=0` since 2026-10-08.
   **Record the reading; do NOT raise `PASS_ON_DEAD_DEPENDENCY_BASELINE`,** which
   stays 3. A recorded reading that lags the live one makes the next `!! MOVED`
   line misattribute its delta.
4. **`scripts/lib_liveness.sh:215` — finish the 120th audit's repair on the
   branch that matters.** That audit guarded the cross-file call to
   `_seal_file_age_hours` so a standalone caller gets a named cause instead of
   `command not found`. The guard sits **inside the `rc == 0` branch** — the path
   taken when the Review is FINE. The function's other cross-file call,
   `stale_output` (defined in `lib_seal.sh`), sits on the **`rc != 0` path**, and
   a caller who sources only this file gets `stale_output: command not found`
   there — which I hit this morning. Worse, `sayfn` defaults to `:`, so the
   `REVIEW LIVENESS FAILED` line vanishes too and the caller gets a **silent
   exit 1 with no reason at all.** Production is unaffected (`overseer.sh:31-32`
   sources both in order) so this is a diagnostic-path repair, ranked here
   accordingly. Guard it the same way, with the same message shape. **Do NOT**
   source `lib_seal.sh` from `lib_liveness.sh` — the libraries are deliberately
   independent and the 120th audit declined that shortcut.
5. **The regate sweep must still refuse a spec whose `IMPL_DEPS` are dirty.**
   Carried a third time, unchanged, from the 148th audit's B0 and the 149th
   audit's item 1. It did not fire today and §1b shows why: the blackout cut the
   Review's sitting to eleven minutes, so the 12:43 sweep landed *after* the
   12:37 commit and bought all three clean. That is timing, not repair. The move
   is a **refusal, not a retry**: before buying, if any file in the spec's
   `IMPL_DEPS` is dirty, skip it and log the skip — `protocol.py:220` already
   computes that predicate one step later. **Do NOT** add `REVIEW_QUEUE.md` to
   `PROSE_DOCS` or any exclusion list (`T0.31` genuinely reads it; the 10-04
   repair at `b4df9bb` deliberately declined that). **Do NOT** hold a lock across
   a desk sitting.
6. **Do not treat a dark slot as work you owe, and do not go looking for a spec
   to write in an empty cost class.** 120 dark slots with **0 failed** is the
   pace gate working (§4). `coverage` marks `cpu<1min`, `gpu<20min` and `gpu<8h`
   as having **no path in** — nothing runnable to implement and nothing
   gate-provisional to pilot. Do not split a unit to fit a skipped slot and do
   not spend an iteration filling an empty class. The `T1.08` prohibition is
   live: do not dispatch a sixth attempt. If a slot opens and `T1.08` is the only
   buyer you can find, buy nothing and say so in the journal.
7. **Five of the Review's own orders are ones you have not seen — read
   `docs/PROGRESS.md:135-156` directly this slot, before item 1 lands.** Its item
   1 is *"BEFORE ANYTHING ELSE, stamp what is already finished"*, and §6 shows
   why it is perishable: **nine rows fall due today** against a measured capacity
   of 6, and two of them (`w0-too-shallow` at `a24cdb7`,
   `goal-187-names-seven-primitives-four-have-no-commitment` at `2fd7de5`) are
   finished work in `DELIVERED — AWAITING STAMP` needing only a signature. Its
   remaining orders are `HR.1` arm (a) (~16 s of CPU, now on its third date), the
   three re-dated units in `DUE:` order, and the `detached-metering-lane`
   reporting join. **This is a reading, not a licence to edit that page** — it is
   the Review's.

## FOR THE OWNER

**1. NO-DECISION — the builder was released from a hard stop this morning and
will be back inside it by this evening. `D40` and `D46` already hold this and I
am not re-asking.**

The usage week reset between the 09:07 and 10:07 slots, after **45 consecutive
slots logged `STOPPED at 90–100%`**. It bought nothing: usage reads 57% → 62% →
**68%** across the three slots since, against a pace line of 37% → 38%, so the
builder is still skipping. That is **5.5 points per hour against a week advancing
at ~1 point per hour**, which puts the **90% hard stop at ≈16:00 today, at ~23%
of a seven-day week**. A pace line cannot converge to meet a number rising five
times faster than it does.

**Of this week's 70 shared points: builder 8 (11%), desks 7 (10%), NOT THIS
PROJECT 55 (78%).** This project's own model line (`week:Fable`) reads **5%**;
the gate it dies on reads 68%. And the one correction that matters for trusting
that number: the 144th audit showed these shares are a function of when they are
read, and that during a blackout the instrument **under-reports the neighbour and
over-reports the builder** — so **78% is a lower bound**, and the known bias
argues for the finding rather than against it.

`D30`'s required pairing, counted this sitting: dark **120 slots / 119.6 h, 0
failed**; **`2026-W41` holds 30.0 free Kaggle GPU-hours, 0.000 drawn, expiring
Saturday 2026-10-17**, the fifth consecutive week on course to expire
substantially unbought. I will not call those hours the loss, for the Review's
reason and not a new one: `coverage` reports **0 fresh dispatches in any cost
class** and all three GPU classes EMPTY and NOT FILLABLE, so even a released
builder has nothing to spend them on. **Both locks are real and only one is
yours.**

**2. NO-DECISION, and the fifth consecutive morning it has been reported: three
of your own constitutional commitments have no live falsifier, now eleven days
old.** **smell**, **shelter/building** and **thermal (kills)** are CLAIM-DEAD —
`SM.02`/`SH.01` PARKED, `SM.03`/`SH.02` PILOT-BLOCKED, every park individually
legal and evidence-backed, the defect compositional. This is `D42` (armed,
`decide_by 2026-10-18`, default (iv) HOLD) and
`five-commitments-are-claim-dead-behind-foreclosures` (OPEN, `DUE 2026-10-14`,
**tomorrow**). **Nothing is asked that `D42` does not already ask.** It is
repeated because `D42`'s own price paragraph predicted the daily report would
become wallpaper, and it has: *"too cold kills him"* and *"he builds a shelter"*
are your words, and this morning the ladder can falsify neither.

**3. A report on your desks' own wiring, with nothing to rule on.** The channel
your two desks use to steer the builder has been half-severed for two days: the
Review's `FOR THE BUILDER` section — five orders this morning, including two
items falling due today — parses to **zero items** through the reader the builder
actually uses, because the heading carries a descriptive clause the regex
rejects. The damage realised so far is **zero**, and only because the builder has
been dark the entire time; the damage latent is that the Review's orders would
have failed silently the moment a slot opened. I am telling you because of the
pattern rather than the bug: **my predecessor wrote down this exact forecast on
2026-10-11, named the durable repair, and did not take it, and the forecast came
true the same morning on the other page.** Both halves are routed to the builder
as items 1 and 2 and neither is yours. The system's diagnosis is in excellent
health; its repair loop is the thing that is not.

**4. NO-DECISION: four of your conduct entries have gone stale at the desks, the
oldest by twenty days.** `D33` (20 d), `D35` (19 d), `D38` (9 d), `D40` (3 d)
are marked desk-executable — *"execute it, report it, do not ask"* — and the
instrument lists them precisely *"so a stale conduct entry cannot silently
self-approve."* In the two days since this was last reported at 18/17/7/1 days,
none has been executed or closed. One piece of credit against that, because it is
the first movement in this class: **`D46`, opened this morning, is the Review
declining to self-execute a `conduct`-flagged item and saying why in writing** —
exempting a desk organ from the meter *"re-allocates a shared account you hold
with four paying tenants, in favour of the desk proposing it."* That is the right
reading of the flag's limit. The desk is calibrating this boundary in both
directions now (it withdrew `D45` two days ago for the opposite reason), which is
what the flag is for.

**5. A liveness report you are entitled to.** All four organs fired within
cadence, but two did so damaged: **the Review missed 2026-10-12 in full** — all
five scheduled polls refused by the usage gate — and recovered via a `--retry`
sitting at 12:26 today that lasted eleven minutes, during which it disposed
twelve broken promises. `review_liveness` correctly reports it failed its
schedule and correctly refused to stamp a page that was being written. The
builder has been dark **120 slots / 119.6 h**, legally. `demonstrated` **107/257**,
unchanged since 2026-10-08. `decisions --check` and `champions --check` both
**EXIT 0** at every declared floor, and `review-queue` ends this sitting at
**EXIT 0 with 0 violations, OVERDUE 12 → 0** — bought by re-dating forward with
causes named, deleting nothing, which is the honest repair. **Sections 1 and 2
are clean: nothing has been loosened, no PASS is unsupported, no control went
unrun, no commit has vanished, and no GPU hour went unbilled.** The scoreboard is
honest. It is the thing being measured that has not moved in five days, and this
morning the channel for saying so to the one organ that could was reading half a
page.
