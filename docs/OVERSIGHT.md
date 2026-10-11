# OVERSIGHT.md — the overseer's current-state report

> Current state, not a log. Each audit rewrites this file.

**2026-10-11 06:37 UTC — 149th audit.** (Sunday: the Review's weekly FULL run
started in the same 06:37 slot, PID 2819780, and was still live at 06:47.)

## VERDICT: DRIFTING

**The ledger is sound and nothing has been loosened. Yesterday's destroyed
certificate was recovered — mechanically, by the same cron that broke it. But
the mechanism that broke it fired again INSIDE this audit's window and stamped
two more Tier-0 certificates dirty, and the one channel this organ has for
steering the builder has been unreadable by the builder's own parser for
twenty-four hours, because my predecessor renamed its list items.**

Three facts set the verdict, in order of how much they cost:

1. **My own `FOR THE BUILDER` section parsed to ZERO items.** The 148th audit
   wrote six orders as `**B0.`–`**B5.`; `steering._ITEM` matches `^(\d{1,2})\.`
   and `_BOLD_LEAD` matches `^\*\*(\d{1,2})\.`, so a letter-prefixed item
   matches neither. `run status` has printed *"heading PRESENT (1), ZERO items
   parsed — the orders exist and the reader cannot see them"* every slot since,
   and no organ read it. **Realised damage is zero only because the builder was
   dark for all 24 h** — including B0, the repair for the thing that cost
   `T0.31`. This page is now written in the parsing form, and the reader is
   re-run against it below. Fixed, not just reported.
2. **The regate/desk race is unrepaired and I watched it fire.** At 06:43:02 —
   six minutes into the Review's sitting, as `43 */2 * * *` guarantees every day
   — the sweep found 3 stale certificates and re-bought `T0.21` (06:43:14) and
   `T0.28` (06:44:25) against a dirty tree. Today the dice came up recoverable.
   Yesterday they did not.
3. **No science in 72 hours.** `demonstrated` 107/257 = 41.6%, unmoved since
   2026-10-08. Builder dark **66 consecutive slots / 65.6 h**. `2026-W40` closed
   last night having drawn **1.971 of 30** free Kaggle GPU-hours — **28.029
   expired**, the fourth consecutive week, ≈114.66 h in four weeks. `2026-W41`
   opened this morning with 30.0 h and 0.0 drawn.

Not `INTEGRITY RISK`: no standing claim is untrustworthy today, no commit has
vanished, no control went unrun, and the two certificates stamped dirty this
morning are the *recoverable* class. The direction is what is wrong.

---

## 1. Integrity of the ledger — CLEAN

**107 PASS of 160 recorded results; 107/257 demonstrated.** Re-derived from
`experiments/ledger.json` directly, not read off `run status`:

- **0** PASS rows missing a `commit` field.
- **0** PASS rows whose commit has vanished from git (all 107 resolve with
  `git cat-file -e`).
- **0** PASS rows that declare a `control` and have empty `control_metrics`.
  Only `T0.01` and `T0.10` have empty control metrics at all; both declare no
  control, both are harness smoke rows, and both are already owned by
  `t018-explicit-no-control-reads-as-an-unrun-promise` (OPEN, `DUE 2026-10-19`).

### 1b. `T0.31` was recovered, and I am correcting my predecessor's verdict

The 148th audit closed as `INTEGRITY RISK` *"for this reason and this reason
alone"*: `T0.31`'s implementation destroyed at 06:45:06, *"it cannot be
recovered by anyone."* That was true of those bytes. It is no longer the state
of the certificate.

At **2026-10-10T08:45:48** the 2-hourly regate re-bought `T0.31` from a clean
tree at `d7e141a` — `commit` field `2406d42`, **no `+dirty` suffix,
`dirty_files: None`**. `T0.21` and `T0.28` bought clean in the same sweep.
`run status`'s `! DIRTY STAMPS` block, which carried all three yesterday, this
morning carried only `T6.03` and `PL.02`. **The clean-tree re-run the 148th
audit booked as an outstanding debt was discharged 1 h 51 m after the loss, by
the cron that caused it, and no desk recorded that it had been.**

Stated plainly because it cuts both ways: the loss was real, and the repair was
luck of the same dice. Nothing was designed to heal it.

### 1c. The race fired again during this audit — measured, not inferred

`43 */2 * * *` puts the regate at **:43 of every even hour**. The Review starts
`37 6 * * *`. So the sweep lands **six minutes into every Review sitting, every
day, by construction** — this is sharper than the 148th audit's *"at 06:37 they
are guaranteed to overlap"*, and it is why the collision is not occasional.

Today's sequence, from `/data/jack-logs/regate.log` and the live ledger:

| time | act | stamp | dirty at run time |
|---|---|---|---|
| 06:39:37 | Review commits `bd87cae` (touches `REVIEW_QUEUE.md`) | — | — |
| 06:43:02 | sweep start — **3 cheap stale certificate(s)** | — | — |
| 06:43:14 | `T0.21` re-bought | **`bd87cae+dirty`** | `experiments/tests/dp_04_slow_path_verbal.py` |
| 06:44:25 | `T0.28` re-bought | **`bd87cae+dirty`** | same |
| 06:45:15 | `T0.31` re-bought | `8b74a34` **CLEAN** | none |

**Why the outcomes differ, and it is not merit.** The dirty file at 06:43–06:44
was the Review's in-flight edit to a *test* file, which is outside `T0.21`'s and
`T0.28`'s declared `IMPL_DEPS` — so those two are the benign class ("the
implementation it names IS committed; a re-run buys a clean stamp, not a
recovered implementation"). `REVIEW_QUEUE.md`, which those specs genuinely read,
had been committed at 06:39, four minutes before the sweep read it. And `T0.31`
escaped entirely because the Review committed `8b74a34` in the ~50 s between
`T0.28`'s run and `T0.31`'s.

**Two days, the same mechanism, opposite outcomes, decided by the phase between
a cron and a desk's commit rhythm.** That is the 148th audit's B0 thesis, and it
now has a controlled pair of observations behind it instead of one. Net board
effect: `! DIRTY STAMPS` grew **2 → 4**.

## 2. Thresholds and controls over seven days — CLEAN

Eleven commits touched `registry.py` / `registry_expansion.py` /
`experiments/tests/` since 2026-10-04. **Not one numeric bar moved in the
loosening direction.** I ran down every deletion carrying a number or
comparator:

- `- and c["inference_params_trained_frac"] < 0.9` (`T1.11`) — the diff artifact
  of **adding** a third conjunct (`shipped_callers >= SHIPPED_CALLER_MIN`). Both
  original conjuncts byte-unmoved; the change **demoted `T1.11` PASS → FAIL**. A
  strengthening, recorded as one.
- `XL.01` v3 (`e9086ac`) — I checked this one hard, because its own queue row is
  named `xl01-pooled-conjunct-is-additive-or-it-is-a-loosening`. It is
  **additive**: the source reads *"gated BESIDE — never instead of — the
  existing all-3-seeds per-seed `ok_claim`"*, `RATIO_MAX 0.5` is byte-unmoved,
  substituting pooled FOR per-seed is **PRE-REFUSED** in the docstring along
  with any `>= 2 of 3` form, the pooled statistic is recorded whatever the
  verdict, and the expected verdict (**still FAIL**) was pre-registered before
  the run. Correct on every count.
- `est_hours` edits are **upward** (0.3 → 2.5) and are new dispatches' own
  projections, not science bars.
- No control deleted, no `_check` gained an `or`, no seed count reduced. No
  `-CONST = N` / `+CONST = M` pair anywhere in the window.

`W1.01`, `W1.04`, `T4.04`, `T4.05`, `T6.01`, `LG.14` are newly implemented;
`LG.14`'s commit states *"Bars imported, none moved"* and that checks out.

## 3. Three ratchet floors are breached, and one of them is the builder alarm

`run status`: **`FLOOR STATE … 3 ABOVE, 0 BELOW, 0 UNVERIFIED`**.

- **`dark_slots = 66` against a declared floor of `0` — `!! MOVED +66 since
  2026-09-30`.** This is the finding my predecessor and the Review both missed
  while describing the same fact in prose. The 148th audit §4 said the builder's
  silence was *"legal and so"*; the Review's `FOR THE BUILDER` 3 said *"the pace
  gate is right; leave it alone."* Both are correct about legality. **Neither
  said that a floor pinned at 0 on 2026-09-30 (`418f015`, 131st audit FTB 1) is
  now 66 points above itself.** The instrument is screaming and two desks
  described its subject without reading its reading. Note the honest
  counterweight: a floor of 0 on a counter that legitimately moves with another
  tenant's usage may simply be the wrong instrument shape — but that is a case
  to make in a commit, not to pass over in silence.
- **`pass_on_dead_dependency = 6` against floor `3`** — six standing PASS
  certificates resting on a recorded non-PASS dependency: `LF.02←T6.03`,
  `T0.18←T0.13`, `T0.19←T0.13`, `T1.12←T1.11`, `T2.03←T1.08`, `T2.14←T1.08`.
  **Already routed and correctly refused a blessing:** the newest pair
  (`T1.12←T1.11`, created when the 10-04 strengthening demoted `T1.11`) is
  `t112-pass-stands-on-t111s-orthogonal-added-conjunct`, whose disposition
  (`b0a3d56`) states *"pass_on_dead_dependency stays red 6 vs floor 3, not
  blessed."* That is the ratchet working. **But `experiments/ratchet_readings.json`
  still records this key as `5`, dated 2026-09-26** — the recorded reading has
  been one behind the live one for seven days.
- **`unreachable = 96` against floor `95`** — fully declared: the key's own note
  explains it as `LT.02`'s honest demotion taking `LT.03` out of the reachable
  set, and explicitly refuses to raise the floor. No finding; recorded so the
  count of breaches is not mistaken for three unexplained ones.

## 4. Is the builder alive and productive? — NO. Second consecutive zero day

- **Last iteration that ran: 2026-10-08T12:32:55, `rc=0`.** Nothing since.
- Last 24 h: **24 slots — 22 `PACING` skips, 2 `ABORT` on box load (9.07 and
  6.44 against a 6.0 ceiling), 0 `rc=0`, 0 ledger events.**
- At 06:07 today: *"65 consecutive dark slot(s); 0 failed slots (65.6 h since
  the last rc=0)."* Not a crash, not a refusal, not the hard stop.

**The release forecast, and it has moved AWAY from us.** Extracted hourly from
`ladder.log`:

| | usage (`week:all models`) | week elapsed | pace line | gap |
|---|---|---|---|---|
| 2026-10-10 03:00 | 64% | 38% | 50% | 14 pts |
| 2026-10-11 06:00 | **83%** | 54% | 61% | **22 pts** |

Over 27 h usage rose **+19** points while the line rose **+11**. The gap
**widened from 14 to 22**. Usage has been flat at 83% since 00:00 today, and on
that assumption the line (≈0.67 pts per point of week) reaches 83% at ~87% of
the week ≈ **2026-10-13 08:00 UTC**, leaving ~3.5 days of W41.

**The risk nobody has priced: the hard stop may fire first.** Usage is 83% with
the hard stop at **90% — 7 points of headroom.** Between 17:00 and 23:00
yesterday usage moved +9 points in six hours. A repeat of that surge puts the
builder under a hard stop before the line ever rises to meet it, and **73% of
this week's 53 shared points are another tenant's**, so the input is not ours to
moderate. Stated as a risk, not a forecast.

## 5. Compute honesty — the loss is realised, and the quota has no buyer

- **`2026-W40` closed: 1.971 h drawn of 30 → 28.029 h EXPIRED.** Fourth
  consecutive week (W37 1.379, W38 0.918, W39 1.072, W40 1.971) ≈ **114.66 h
  unbought in four weeks**. Recorded as realised; the 147th audit forecast it
  and the 148th booked it, and nothing has changed it.
- **`2026-W41` opened today** (`Budget._week()` = `2026-W41`, `%Y-W%U`,
  Sunday-start to match Kaggle's reset) with **30.0 h and no entry in
  `gpu_budget.json`** — 0.0 drawn. It expires Saturday 2026-10-17.
- **No GPU hours were spent without a ledger entry.** Last charged job
  2026-10-08T13:19:59; every W40 job carries a charged row and an `ok` flag.
  Two per-job overruns, both W40, both under-estimates (+31%, +109%), honestly
  recorded in `overruns`.
- `gpu_hours_no_verdict` TOTAL rose **49.49 → 51.47 h**. The largest unredeemed
  block is unchanged and unaddressed: **`D1.0` — 33.78 h across 2 attempts with
  ZERO verdicts.** `UNATTRIBUTED` 6.32 h / 21 jobs sits at its declared floor.

**And here is the half that is about the ladder rather than the meter, which no
desk has put beside the GPU number.** `coverage`'s queue depth: of 7 specs
dispatchable today, **7 are VOID → 0 FRESH dispatches.** `gpu<20min` EMPTY,
`gpu<8h` EMPTY, `gpu<2h` holds only `UB.10` (VOID, an arm to repair) — and all
three empty GPU classes are marked **NOT FILLABLE: nothing to implement,
nothing to pilot**. **A released builder has nothing GPU-shaped to spend W41's
30 hours on.** Reporting the pace line as the sole cause of four lost weeks
flatters the ladder; both locks are real and only one of them is the meter's.

## 6. Stuck decisions — the register is armed; the CONDUCT class is rotting

`decisions --check` **EXIT 0**, ratchet ok — `0/0` undeclared, unrouted-owner-ask,
vanished-owner-ask, default-action-expired, firing-diff. 31 of 31 firings
transcribed.

- **No `MEANS-ESCALATED`.** No measurable fork sits on the owner's desk; the
  `D1` disease is absent.
- **No `UNDECLARED`, so this audit arms nothing** — not an omission, there is
  nothing to arm. Armed and future-dated: `D43` 10-16, `D41`/`D42` 10-18, `D44`
  10-24.
- **No armed default is due to fire.** `D40`'s `decide_by` of 2026-10-10 has
  passed, and I checked whether rule 3 obliged me to fire it: it does not.
  `decisions.py` classes `D40` as **`CONDUCT-DESK`**, not armed — and its
  default (v) is the status quo, which carries no action. `default_action_expired`
  reads **0**, down from the recorded 1. Nothing fires; the entry is stale.
- **Four conduct entries are desk-executable and stale: `D33` by 18 days, `D35`
  by 17, `D38` by 7, `D40` by 1.** The instrument lists them *"so a stale
  conduct entry cannot silently self-approve"*, and for 18 days nobody has
  executed or closed one. This class has no owner and no clock anyone honours.
- **`D38`'s collision condition is live as I write, and it may already have
  dissolved.** `D38` asks which of two armed defaults owns the FULL Review's
  first act — *"they collide only on Sundays"* — and **today is a Sunday FULL.**
  But its two colliders are `D28` and `D33`, and `D33` is **MOOT**: its object
  went terminal when `w1-world-edit-window` was DECLINED. If one collider cannot
  fire, `D28` wins by default and `D38` is answerable at a desk today rather
  than being a constitution defect. **Nobody has joined those two readings** —
  the 148th audit recorded `D33`'s mootness and never looked at what it does to
  `D38`. Routed below.
- `D41`/`D42`/`D44` carry `CONDUCT-MISFILED?` — class `goal` but blocking no
  spec id. Soft, routing-only, correctly the Review's.

## 7. Bakeoff hygiene — CLEAN

33 resolved entries in `docs/DECISIONS_RESOLVED.md`. No decision made without a
learning gate; no VOID treated as a verdict in the resolved record; no winner
chosen inside the noise margin; a text scan for noise-margin and
VOID-as-verdict language returns nothing.

One standing debt, reported rather than hidden because `champions --check`
raises it at its floor: the **Learning core** seat is held `BY VERDICT` — the
file's strongest marking — off `LC.03`, which is a **VOID**, and a VOID decided
nothing. It carries `TRIGGER DEBT` as well (`LC.07` PILOT-BLOCKED, `LC.03`
VOID-FORECLOSED, `UB.10` VOID: every declared re-open trigger a closed door).
`champions --check` EXIT 0, every class at its declared floor, **`ARENA-MISSING`
is 0** — and for the second audit running I record that this organ's own
standing instructions still say *"8 seats today"*. Every number in those
instructions is a dated example.

## 8. Drift from the goal

**What the last day of work served.** The builder produced nothing (§4), so the
day's output is two desks': the Review's five OVERDUE disposals and the 148th
audit's report. `dark-slot-counter` ACTED, `dp04-lifespan-has-no-resolution`
under disposal, `T0.21`/`T0.28`/`T0.31` mechanically re-bought. Every item
traces to *"protects the honesty of watching what happens when the three
meet."* **No drift in the strict sense: nothing was worked on that traces to no
GOAL.md sentence.**

**The converse, which is the question that matters.** `coverage` EXIT 2, **0
commitments with no declared spec** — the highest-priority class is clean and
has been since 2026-09-19. But three of the owner's own constitutional
commitments have **no live falsifier at all**, now **nine days** old:

- **smell** — `SM.02` PARKED, `SM.03` FORECLOSED. *"the sense that works when
  sight fails."*
- **shelter/building** — `SH.01` PARKED, `SH.02` FORECLOSED. *"the owner's own
  image of success."*
- **thermal (kills)** — *"too cold kills him, too hot kills him."*

All three are CLAIM-DEAD **compositionally** — every park individually legal and
evidence-backed, and no dispatch anywhere revives the commitment behind them.
This is `D42` (armed, `decide_by 2026-10-18`, default (iv) HOLD) and
`five-commitments-are-claim-dead-behind-foreclosures` (OPEN, `DUE 2026-10-14`).
Correctly routed, correctly the owner's.

Fourteen further commitments have live claim specs and nothing passing —
**touch, tool use, told world, proprioception, sleep, plasticity, fast/slow**
among them. **Curiosity 2 passing of 12. One brain / unison 1 of 28.** Those two
are the thesis.

And `coverage` names **7 GOAL.md citations that resolve to corpses** —
`DP.02`, `DP.03`, `LC.04` (baselined) plus `GEN.02`, `GEN.06`, `GEN.03`,
`GEN.09`, all welded behind `LC.07`. The gen-four revival is routed
(`89c4c71`); the baseline is shrink-only and was not touched.

## 9. The honest summary

**Are we closer to a curious humanoid that climbs the ladder than yesterday?
No. Three days running, the answer is no, and today the reason is not even
interesting.**

The ledger did not move (107/257, unchanged since 10-08). The builder ran zero
iterations for the second straight day. 28 GPU-hours expired overnight and a
fresh 30 arrived this morning with **no spec in any GPU cost class to spend them
on**. The entire output of both desks went into the desks.

What is genuinely good, and it is not small: **section 1 and section 2 are
clean.** 107 certificates, every commit present, every declared control
recorded, and across eleven commits in seven days not one bar moved in the
loosening direction — including one change (`XL.01` v3) that was routed
*specifically* on the suspicion that it might be a loosening and turned out to
be scrupulously additive, with its own expected FAIL pre-registered. **The
backlog board went OVERDUE 5 → 0 while I watched, and `review-queue` finished
this sitting at EXIT 0** — the Review disposed all five, on the day its own
`D28` default orders it to. `decisions` and `champions` both exit 0 at every
floor. **When this system is asked whether it is cheating, the answer keeps
coming back no.**

The drift is that the system has become very good at the one question it can
answer about itself and is no longer asking the world anything. Yesterday's
report ended on the sentence that the project *"caught itself destroying the
evidence that it cannot destroy evidence."* Today the same cron re-bought that
certificate clean, stamped two others dirty, and nobody designed any of it. We
are not losing integrity. We are spending every hour defending it and buying no
science with what is left — and this morning the organ that is supposed to tell
the builder what to do about that had been talking into a parser that could not
hear it.

**One thing I will not let pass as absence.** `dark_slots` has been 66 points
above a floor pinned at 0, and the two desks that each wrote a paragraph about
the builder's silence yesterday both failed to read the number the system
computes about it. A floor nobody reads is not a ratchet; it is decoration. That
is `T0.13`'s own claim — *"No gate in the ladder is decorative"* — and `T0.13`
is FAIL.

## 10. The instruments, re-run at 06:53 immediately before this file was committed

The Review's weekly FULL was live in this same 06:37 slot and committed six
times while I worked, so a reading quoted from the top of a sitting is the
default failure here. **Two of these moved during the audit** and the table is
the 06:53 re-run, not the 06:40 one.

| instrument | exit | reading at 06:53 |
|---|---|---|
| `coverage` | **2** | **0** commitments with NO declared spec; **3 CLAIM-DEAD** (smell, shelter/building, thermal); 14 with live claim specs and nothing passing. 7 dispatchable, **all 7 VOID → 0 fresh**. |
| `decisions --check` | **0** | ratchet ok, all five classes `0/0`. 4 armed (`D41`/`D42` 10-18, `D43` 10-16, `D44` 10-24); 8 unarmed, 4 of them stale conduct. |
| `champions --check` | **0** | ratchet ok, every class at its declared floor. **`ARENA-MISSING` is 0**, not the "8 seats" this organ's instructions still quote. |
| `run review-queue` | **0** *(was 2)* | **OVERDUE 5 → 0.** 42 OPEN, 36 DISPOSITIONED, 50 ACTED, 12 DECLINED of 140 routed; oldest live 48 d. |
| `run status` | **2** | 107/257. **3 floors ABOVE** (`dark_slots` 66 vs 0, `pass_on_dead_dependency` 6 vs 3, `unreachable` 96 vs 95). `! DIRTY STAMPS` **2 → 4** (`T6.03`, `PL.02`, + `T0.21`, `T0.28` from the 06:43 sweep). |

**And the one reading that is this report's own repair, verified rather than
asserted:** `run status` now prints *"STEERING-PAGE ORDERS — **10 item(s) on 2
of 2 declared page(s)**; 0 order(s) name a spec the runner would REFUSE today"*.
Before this commit it read *"3 item(s) on 1 of 2"* with
`docs/OVERSIGHT.md: !! heading PRESENT (1), ZERO items parsed`. The seven orders
below are machine-visible, and I checked that with `steering.builder_items()`
against the file on disk rather than by looking at it.

---

## FOR THE BUILDER

**Read this section through `steering.read()` before trusting it.** The 148th
audit's six orders were written as `**B0.`–`**B5.` and parsed to **zero items**;
these are renumbered into the form `_ITEM`/`_BOLD_LEAD` accept, and I verified
the parse before committing. The orders below carry the 148th audit's B0–B5
forward — **they are not new work, they are the same work, now visible.**

1. **THE REGATE SWEEP MUST REFUSE A SPEC WHOSE `IMPL_DEPS` ARE DIRTY — TOP
   ITEM, and I watched it bill two more certificates this morning.** Carried
   unchanged from the 148th audit's B0, which never reached you. `43 */2 * * *`
   lands the sweep **six minutes into every Review sitting**; at 06:43:14 and
   06:44:25 today it stamped `T0.21` and `T0.28` `bd87cae+dirty`, and yesterday
   at 06:45:06 the same sweep made `T0.31`'s implementation unrecoverable (§1c
   has the controlled pair). The repair is a **refusal, not a retry**: before
   buying a spec, if any file in that spec's `IMPL_DEPS` is dirty, skip it and
   log the skip. `protocol.py:220` already computes that predicate for the
   `+dirty` stamp — it needs calling one step earlier. A re-buy is a
   convenience; a lost implementation is not recoverable, so the asymmetry is
   total. **Do NOT** add `REVIEW_QUEUE.md` to `PROSE_DOCS` or any exclusion list
   — `T0.31` genuinely reads it, and the 10-04 repair (`b4df9bb`) deliberately
   declined that shortcut. **Do NOT** serialise by holding a lock across a whole
   desk sitting. Skip-and-log is the monotone move.
2. **`scripts/lib_seal.sh` — the rescue sweep must not be gated on the sealed
   file.** The 148th audit's B1, unreached. `seal_output()` returns at `:317`
   and `:323` before the dirty-set sweep at `:325`, so a dying run whose `$file`
   is clean gets **no sweep at all** — and `scripts/review_prompt.md:154` orders
   `PROGRESS.md` written LAST, so the sweep can only fire for a run that got far
   enough not to need it. Hoist the partition-and-sweep block so it runs on
   **every** `rc != 0` path. **Do NOT** change which files are swept or the
   `mtime >= run_start` predicate — that partition is the 74th audit's B1 and it
   is correct. Only the block's reachability changes. Report before/after on a
   forced `rc=124` with a clean `PROGRESS.md`.
3. **`scripts/lib_seal.sh:227` — `_seal_file_age_hours` must not count the
   seal's own stamps.** The 148th audit's B2, unreached. It reads
   `git log -1 --format=%ct -- "$file"` and `stale_output()` commits that file,
   so every stamp zeroes the clock it reads: on 10-08 it called a page *"only
   24h old … still current"* whose content was **96 h old and already bannered
   STALE**. Walk back to the last commit touching the file that is not a seal
   stamp. This is a reporting repair — it can only make the seal stamp MORE
   often. **Do NOT** raise `max_clean_age` or drop the already-bannered guard.
4. **`experiments/run.py` — a `+dirty` re-buy message must not read CLEAN.** The
   148th audit's B3, unreached, and this morning produced two fresh examples:
   both new `+dirty` rows record `HASH-SALT DIFFERENTIAL CLEAN`. True of the
   salt differential, false of the tree, and the row is what later readers
   believe. Append the dirty marker to the recorded message when the stamp
   carries `+dirty`. Strictly additive: no verdict moves, and `run status`'s
   `! DIRTY STAMPS` reader keeps its text.
5. **`experiments/ratchet_readings.json` — `pass_on_dead_dependency` has read
   one behind the live value for seven days.** The key records `5` dated
   2026-09-26; live is `6` (§3). The sixth pair is `T1.12←T1.11`, created by the
   10-04 strengthening and correctly dispositioned *"not blessed"* at `b0a3d56`.
   **Record the reading; do NOT raise `PASS_ON_DEAD_DEPENDENCY_BASELINE`,** which
   stays 3. A recorded reading that lags the live one makes the next `!! MOVED`
   line misattribute its delta.
6. **Three of your steering orders on `docs/PROGRESS.md` are already discharged
   and that page cannot tell you** — re-verified this audit, carried from the
   148th audit's B5 because it never reached you either. `T1.08` Steps 0+1
   (`t108-pipeline-repair-has-no-design` **ACTED 2026-10-08**, Step 2b executed
   at `cff86c3`); the additive `XL.01` pooled conjunct (**done at `e9086ac`**,
   and §2 confirms it is additive); and `fieldwatch.py:102`'s `\bfindings?\b`
   (**done**, live in source). **Do not re-execute those three.** A reading for
   you, not an order to edit the page — that page is the Review's.
7. **Do not treat a dark slot as work you owe.** 66 dark slots with **0 failed**
   is the pace gate working, not a bug (§4). Do not split a unit to fit a
   skipped slot. But note what §5 adds: there is **no FRESH dispatch in any cost
   class** and all three GPU classes are empty and NOT FILLABLE, so when a slot
   does open, the binding constraint is the ladder, not the meter.

## FOR THE OWNER

**1. The fourth consecutive ~28 GPU-hours expired last night, a fresh 30 arrived
this morning, and this time the ladder has nothing to spend them on either.
NO-DECISION — `D40` already covers the meter half and its date has passed.**

`2026-W40` closed having drawn **1.971 of 30** free Kaggle hours; **28.029
expired**. With W37 (1.379), W38 (0.918) and W39 (1.072) that is ≈**114.66 hours
in four weeks**. `2026-W41` opened this morning at 30.0 h / 0.0 drawn, expiring
Saturday 2026-10-17.

What is new is not the meter — it is that **both locks are now measured, and
only one of them is the pace line.** `coverage` reports 7 dispatchable specs
today, **all 7 VOID → 0 fresh dispatches**, with `gpu<20min`, `gpu<2h` and
`gpu<8h` all empty or holding only an arm to repair, each marked *"NOT FILLABLE:
nothing to implement, nothing to pilot."* Even a released builder has no
GPU-shaped work. Four weeks of reports have named the pace line as the cause; it
is half the cause, and the other half is ours.

The meter half: usage is **83%** against a pace line of **61%** at 54% of the
week, **73% of the pool another tenant's**, and the gap has **widened** from 14
to 22 points in 27 hours (§4). On a flat 83% the builder releases ~**2026-10-13
08:00 UTC**; but usage sits **7 points under the 90% hard stop** and moved +9 in
six hours last night, so a hard stop before release is a live possibility. `D40`
is `CONDUCT-DESK`, its `decide_by` was 2026-10-10, its default (v) is the status
quo and carries no action — so **nothing fired and nothing is asked of you
here.** If (v) stands, this recurs on 2026-10-17 with the same arithmetic.

**2. NO-DECISION, and the fourth consecutive morning it has been reported:
three of your own constitutional commitments have no live falsifier, now nine
days old.** **smell**, **shelter/building** and **thermal (kills)** are
CLAIM-DEAD — `SM.02`/`SH.01` PARKED, `SM.03`/`SH.02` FORECLOSED, every park
individually legal and evidence-backed, the defect compositional. This is `D42`
(armed, `decide_by 2026-10-18`, default (iv) HOLD) and
`five-commitments-are-claim-dead-behind-foreclosures` (OPEN, `DUE 2026-10-14`).
**Nothing is asked that `D42` does not already ask.** It is repeated because
`D42`'s own price paragraph predicted the daily report would become wallpaper,
and it has: *"too cold kills him"* and *"he builds a shelter"* are your words,
and this morning the ladder can falsify neither.

**3. A loss report with nothing to rule on, and a correction to yesterday's
page.** Yesterday this organ told you a Tier-0 certificate's implementation had
been destroyed and *"cannot be recovered by anyone."* **The bytes were lost; the
certificate was not.** At 08:45:48 the same 2-hourly cron re-bought `T0.31` from
a clean tree, and `run status` dropped it from the dirty-stamp list — a fact no
desk recorded until now. **The mechanism, however, is untouched and it fired
again inside this audit**, stamping `T0.21` and `T0.28` dirty at 06:43 and
06:44, and sparing `T0.31` only because the Review happened to commit in a 50
second window (§1c). Two days, the same cron, opposite outcomes, decided by
timing alone. The repair is routed to the builder as item 1 and it is the top
item; nothing here is yours to decide.

**4. NO-DECISION: four of your conduct entries have gone stale at the desks, one
for eighteen days.** `D33` (18 d), `D35` (17 d), `D38` (7 d), `D40` (1 d) are
marked desk-executable — *"execute it, report it, do not ask"* — and the
instrument lists them precisely *"so a stale conduct entry cannot silently
self-approve."* None has been executed or closed. `D33` is **MOOT** (its object
went terminal), which no desk can clear by firing anything. I am flagging one
consequence nobody has drawn: **`D38` asks which armed default owns the FULL
Review's first act, says the two collide only on Sundays, and today is a Sunday
FULL — but one of its two colliders is `D33`.** If `D33` cannot fire, the
collision `D38` describes may no longer exist, which would make it a desk
question today rather than a constitution defect. Routed to the Review, not to
you.

**5. A liveness report you are entitled to.** All four organs fired within
cadence. The Review is mid-sitting as this commits (its weekly FULL, started
06:37) and **disposed all five OVERDUE rows while I audited — the board went
5 → 0 and `review-queue` ended at EXIT 0**, which is the one number that
improved today. The builder has been dark **66 slots / 65.6 h**, legally, plus
2 slots aborted on box load above the 6.0 ceiling. `demonstrated` **107/257**,
unchanged since 2026-10-08. `decisions --check` and `champions --check` both
**EXIT 0** at every declared floor. **Sections 1 and 2 are clean: nothing has
been loosened, no PASS is unsupported, no control went unrun, and no GPU hour
went unbilled.** The scoreboard is honest. It is the thing being measured that
has stopped moving.
