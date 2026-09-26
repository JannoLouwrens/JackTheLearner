# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Findings are ranked by how much damage they do to the
> trustworthiness of the ledger, not by how interesting they are.

**2026-09-26 06:37–07:0x UTC — the 120th audit.** Six hours after the 119th
(00:37). The window is the builder's six slots `01:0x`–`06:1x`, which produced
**eleven commits and seventeen ledger events since 2026-09-25 06:37** — of which
**three carry a verdict about Jack** and twelve are the project buying back its
own instruments. Demonstrated **111/254**, unchanged across the window.

Instrument exit codes, re-derived this sitting and not inherited:
`coverage` **2**, `decisions --check` **1**, `champions --check` **0**,
`run review-queue` **2**, `run status` **0**.

---

## VERDICT: DRIFTING — the ledger is more honest than yesterday and the creature has not moved; three of the owner's own commitments still have no runnable claim at 21 days, and the one genuine threshold event in the window is a claim floor set 20% below its own stated derivation under a label that names the wrong direction

**There is no integrity risk in this window and I want that said first.** I
re-derived Section 1 from scratch rather than taking it from any organ: all 111
PASS rows resolve their `commit` in git, all 111 declare a control, and the two
without `control_metrics` declare `NONE, BY DECISION` on their face. Nothing on
the scoreboard is unearned by anything I can measure.

What is drifting is the distance between the instruments and the animal. Over
seven days `run status` measures it itself: **51 runs = 7 first-ever, 39 re-buy,
5 status change, and 35 of 51 (69%) instrument-coupled** — this project's own
tool edits are what staled the certificate. In the last 24 hours the ladder
claimed nothing new about Jack at all.

---

## FINDING 1 — BA.03's new claim floor is 20% weaker than its own derivation, and both the source comment and the docstring call the discount "the demanding side" (MEDIUM-HIGH — the one threshold event in the window, and it must be fixed before the registered run)

`702aa56` re-based BA.03's claim on integrated absolute tilt and pre-registered
two fractions of the random arm's in-run episode spread. The pre-registration
paragraph states its own arithmetic (`ba_03_braces_against_a_surface.py:330-334`,
and again in the constant block at `:601-610`):

> required claim signal `T_GAIN_MIN * gain_se` = 3 × 2.4674/√120 = 0.676 s
> against a random-arm episode spread of ~2.7 s **-> 0.25**;
> `VEST_OVER_NOISE_MIN` 0.20 s / ~2.7 s **-> 0.074**. Rounded toward the
> **demanding** side for the claim floor and the lenient side for the noise
> floor: `TILT_GAIN_MIN_FRAC = 0.20`, `TILT_VEST_OVER_NOISE_FRAC = 0.05`.

Both constants gate as **floors on the claim arm** (`:1141` and `:1142-1143`):

```
and m["tilt_gain"] >= TILT_GAIN_MIN_FRAC * m["tilt_sd_random"]
and (m["tilt_gain"] - c["tilt_gain_noise"]
     >= TILT_VEST_OVER_NOISE_FRAC * m["tilt_sd_random"])
```

A **lower** floor is **easier** to clear. So `0.20` against a derived `0.25` is
the claim floor **20% weaker** than its own stated arithmetic, and `0.05`
against `0.074` is the noise-contrast floor **32% weaker**. Both roundings went
the same way — the permissive way. Only the second one says so. The first is
labelled "demanding side" in two places, and that word is the entire
justification offered for the discount.

**What this does NOT do today.** It buys nothing. The harvest (`4761d60`)
recorded, before the registered run, that the claim **forecasts FAIL** at seed
90: `tilt_gain` = **−1.0574 rad·s** at `tilt_gain_se` 0.1881, t ≈ **−5.6**, so
`tilt_gain_positive == 1.0` fails first and neither floor binds. No verdict
anywhere rests on the discount.

**Why it still ranks first.** These two constants are declared unmovable in
either direction now that the pilot has been read, and BA.03's registered
3-seed run is the builder's own handed-forward next dispatch. If any future seed
lands `tilt_gain` between 0.20 and 0.25 × `tilt_sd_random`, the claim PASSes on a
bar its own derivation calls too low, and the permanent record will say the bar
was rounded demanding-side. That is the exact shape of silent loosening, and the
only reason it is not one today is that nobody has cashed it yet.

**Credit where it is owed, because the rest of this commit pair is exemplary and
I checked each claim rather than reading the message.** `void_foreclosed` reads
`None` (the supersession is real, not asserted). `_GATES_FROZEN` was `False`
before the pilot and `run()` refused with a message that forbids sizing any
constant from it. `T_GAIN_MIN` 3.0, `HEADROOM_MIN_MULT` 2.0,
`NOISE_GAIN_FRAC_MAX` 0.50 and `BRACE_CONSISTENCY_MIN` are unmoved; the six
green rig conjuncts are byte-unchanged; the FAIL forecast and the ablation
anatomy (pinning any vestibular block costs the policy nothing, pinning plantar
touch costs +7.36 rad·s) were written down **before** the registered run so they
cannot be narrated afterwards. This is one mislabelled rounding inside an
otherwise model pre-registration — which is why the repair is two words, not a
redesign.

---

## FINDING 2 — `decisions.py` floors five ratchet classes and `run status` joins none of them; one is BROKEN right now and nothing on disk records when it broke or why (MEDIUM)

`experiments/decisions.py:1472` ratchets five classes, shrink-only:

```
UNDECLARED: 10 · UNROUTED-OWNER-ASK: 3 · VANISHED-OWNER-ASK: 0
DEFAULT-ACTION-EXPIRED: 0 · FIRING-DIFF: 0
```

`experiments/run.py:ratchet_live()` (`:1443-1671`) joins **17** counters — seven
from `coverage.py`, two from `champions.py`, four from `review_queue.py`, one
from `cpu_budget.py`, two GPU — and **zero from `decisions.py`**.
`experiments/ratchet_readings.json` therefore holds no key for any of the five,
`run status`'s RATCHET COUNTERS block prints none of them, and `run status`
exits 0.

**Today `DEFAULT-ACTION-EXPIRED` = 1 against baseline 0. The ratchet is broken.**
It has read ≥1 since `D33`'s `decide_by` of 2026-09-23 passed — three days — and
there is no `"at"` date and no commit attribution anywhere for it, because the
readings file has no key to hold them. Contrast the ratchet that broke two days
ago: `review_queue_violations` 0→8 carries `"at": "2026-09-26"`, and the builder
traced it to its cause (clock, not act) and recorded that in `21da307`. The same
work is impossible here.

**The precedent is written in `run.py`'s own source**, on the counter
immediately above where these five belong:

> *"91st audit RANK 2: this class grew from 3 to 4 on 09-12 when `LG.03` was
> foreclosed and NOT ONE COUNTER MOVED, in either tool. It now has a baseline in
> `champions.py`; **surfacing it HERE is the other half** — the builder reads
> `run status` every slot and does not run `champions --check` every slot."*

`decisions.py` is at exactly the stage `champions.py` was in before that repair.

**Stated fairly, the mitigation is real and the class is not invisible.**
`decisions --check` is one of the builder's four mandated per-slot checks, it
does exit 1 on growth, and the 01:xx journal named this red correctly. The gap is
the **committed reading and the `!! MOVED` banner**, not total silence. It is
also the reason no organ can say when this one went red.

**The durable half.** No spec asserts that `ratchet_live()` covers the union of
the tools' own floored-class dictionaries. That property — not the five missing
`take()` lines — is what stops the next instrument shipping a floored class
nobody joins.

---

## FINDING 3 — the Review worked on 09-25 and never published; the owner's page is two days old, and the detector built yesterday caught it correctly (HIGH for the owner, and the mechanism is working)

`docs/PROGRESS.md` last moved at `f7abc08`, **2026-09-24T06:45**. The 09-25
sitting is on disk and it was **productive**: five queue commits between 06:38
and 06:52 (`a208c40` pl02 ACTED, `2c1d1c0` the PS-family legibility ruling,
`5f88609` lt02 RULED (a), `ee3826d` t215, `46d4b60` minting
`t108-pipeline-repair-has-no-design`) plus **`D36` itself** in
`DECISIONS_NEEDED.md`. The desk did the work and then died before its report.

**Yesterday's FINDING 3 mechanism is closed and I verified it rather than taking
the commit message.** `ee788f2` added a second, independent assertion to
`review_liveness` — `PROGRESS.md`'s own git age against the 25 h cadence — where
the old check had been reading a `PROGRESS_LOG.md` row that the *dying* Review
writes as its own death disclosure. It fired at 42 h and stamped the banner
(`8128a75`). Run in production sourcing order this morning it returns
`STILL STALE — docs/PROGRESS.md carries its banner`, rc=1. That is the alarm
being honest about its sibling organ for the first time.

**What is not closed.** The page under that banner is 09-24's, and its headline
is now false of today: it reads *"THE BUILDER IS BACK AND HAS NOTHING TO DO"* at
`week:all models` **19%**, where this morning's slots read **54–57%** and shipped
four units. A banner that discloses staleness is the right repair and it is not
the same thing as a current report. Nothing is hidden; nothing is current either.

Tomorrow, 2026-09-27, is the Sunday sitting that `D36` is about, and that this
desk has publicly pre-committed to as the last instalment of the W1 re-date.

**And the live correction this organ owes on its own finding: the Review's 09-26
DAILY is sitting as I write** — three commits landed mid-audit (`91392d5`,
`3c7f90e`, `c790064`, the last a `t211` metric ruling) and the OVERDUE class went
8 → 5 by its hand. So the two-day silence is being ended by the same organ this
finding is about, and if this sitting reaches its page the banner disappears on
its own. What this finding is *about* survives that: a desk that disposes rows and
dies before its report leaves the owner with nothing, and it has now done so on
four of four Sunday FULLs and on the 09-25 DAILY.

---

## FINDING 4 — the two architectural seats that matter most cannot lose, and neither number has moved in 13 and 23 days (MEDIUM, standing)

`champions --check` exits **0** with the ratchet at floor and **10 violations**.
Two of them are the strongest markings in the file resting on nothing:

- **Learning core** — held **BY VERDICT** off `LC.03 = VOID`. A VOID decided
  nothing (`VERDICT-IS-A-VOID`), and **all three** pre-registered re-open
  triggers are closed doors: `LC.07` PILOT-BLOCKED, `LC.03` VOID-FORECLOSED,
  `UB.10` VOID. `TRIGGER-UNREACHABLE`. This seat cannot be contested by any run
  that exists.
- **World** — held **BY VERDICT**, the file's strongest marking, naming **no
  deciding ledger row** (`VERDICT-UNDECLARED`) and declaring **no `TRIGGER:` at
  all** (`TRIGGER-UNDECLARED`). This is the one seat the project's largest
  standing scientific result — the seven-instrument `W0-too-shallow` finding — is
  *about*, and no evidence can reach it.

`champions_trigger_debt` = **3**, unchanged since 2026-09-03 (**23 days**).
`champions_unwinnable` = **4**, unchanged since 2026-09-13 (**13 days**). The
Review has deferred the World seat on the `w1-world-edit-window`/`D33` ground for
three consecutive sittings and recorded each deferral in the open, which is the
honest way to be late.

**And the good news in the same tool, which belongs in the same paragraph:
`ARENA-MISSING` is 0.** The eight phantom arenas this organ's instructions were
written against were closed by **registering the specs**, not by deleting the
arena references. That ratchet was repaired the only way that is not a
laundering. `UNDECLARED` seats: **0 of 0** — every seat now says what would
unseat it.

---

## FINDING 5 — twelve of seventeen ledger events in the window were the project paying its own instruments' staleness bills, and three constitutional commitments still have no runnable claim (MEDIUM — this is Section 3, and the ratio is the finding)

Seventeen ledger events since 2026-09-25 06:37. **Three carry a verdict about
Jack:** `LT.02` FAIL→PASS at 13:25 (the curiosity venue's noise-trap detector —
a genuine status change and the only `+1` on the board in two days), `PS.09` VOID
(its own newly-promoted known-answer conjunct firing, which is a control working),
`LT.03` VOID (the constitutional ladder test, the 119th audit's FINDING 1). The
other **twelve** are `T0.17`/`T0.33`/`T0.35`/`T0.21`/`LG.13` re-buys bought by
three tool edits.

**Every unit in the window traces to a `GOAL.md` sentence and I am calling none
of it drift.** BA.03 serves the sensory inventory at `GOAL.md:41`
("proprioception & balance"). The hash-salt differential, the `review_liveness`
repair, the `DECISIONS_RESOLVED.md` RUNNER_OUTPUTS move and the coverage fixture
inversion all serve `GOAL.md:8-9` — *"or protects the honesty of watching what
happens when the three meet."* That is a real clause and this is real work
against it.

**The converse, which is the harder question and the one that answers Section 8.**
`coverage` exits 2 with `commitments_uncovered = 0` — no commitment lacks a spec
— and **three CLAIM-DEAD**: every claim spec parked or foreclosed for

- **smell** (`SM.02` PARKED, `SM.03` FORECLOSED — PILOT-BLOCKED),
- **shelter/building** (`SH.01` PARKED, `SH.02` FORECLOSED — PILOT-BLOCKED),
- **thermal — too cold/too hot KILLS him** (same two corpses).

Every one of those foreclosures says the same sentence: *the repair is a
redesign, never a dispatch.* `goal_unrunnable` = **7**, unchanged since
2026-09-05 — **21 days**. Beyond the claim-dead three: **curiosity** 12 specs, 2
pass, **0 runnable now**; **fast/slow** 8 specs, 0 pass, 0 runnable;
**one brain / unison** 28 specs, **1 pass**.

**An asterisk this organ owes on a ratchet that improved.** `claim_dead` moved
4→3 in this window, and the cause is `702aa56` clearing BA.03's VOID-FORECLOSED,
which returned **balance** a runnable claim spec. That spec's own harvested pilot
forecasts **FAIL**, with the vestibular arm **worse** than its blind twin. The
counter counts *runnable* claims and is honest by its own definition; "balance is
no longer claim-dead" is not what a reader will take from it. The builder traced
and recorded this move in `21da307` rather than banking it, which is the right
conduct.

---

## FINDING 6 — the queue's drain is UNBOUNDED and 8 promises broke yesterday; the Review is draining it live as I write, and I re-measured rather than publishing the stale number (MEDIUM, standing)

**At 06:4x, when I took the reading:** `run review-queue` exits 2 — **38 OPEN, 3
HELD, 20 DISPOSITIONED, 23 ACTED, 0 DECLINED of 84 routed; 61 live rows; oldest
live 33 days.** Trailing 7 days: arrived 22 (3.14/cycle), disposed 8
(1.14/cycle), designed 14 (still live, still ageing) — **net +14, drain
UNBOUNDED**. **8 OVERDUE**, every one promised 2026-09-25 and one day old.

**At 07:0x, re-run immediately before committing because this organ and the
Review collide at 06:37:** the Review's DAILY is mid-sitting and three of its
commits landed during my audit (`91392d5`, `3c7f90e`, `c790064`). The live
reading is now **37 OPEN, 3 HELD, 19 DISPOSITIONED, 25 ACTED; 59 live rows;
disposed 10 (1.43/cycle); net +12; 5 OVERDUE.** Still EXIT 2, still UNBOUNDED —
but moving in the right direction, by this desk's own hand, while I was writing.
Both readings are published because the first is what the rest of this report was
computed against and the second is what is true now.

`review_queue_violations` 0→8 was **CLOCK, not act** — the tool's own attribution
says no commit is to blame, the builder re-derived that in `21da307`, and I
re-derived it here and it holds. `DECLINE` remains unused across all 84 rows this
file has ever carried, which is the 77th consecutive row on which that is true.

**Two of the eight were the builder's by the rows' own text**, and one of them —
`waits-on-declared-field` — was **ACTED by the Review at `3c7f90e` during this
audit**, with the honest note that *the field is implemented and unused*. The
other, `cross-organ-doc-race-voids-certificates` (*"what this date owes is
EXECUTION by the builder, not a decision by this desk"*), is still live and is a
legal pick today.

---

## Section 1 — INTEGRITY OF THE LEDGER: no findings, and every claim was re-derived

| check | result |
|---|---|
| PASS rows (latest per spec) | **111** |
| `commit` resolves via `git cat-file -e <sha>^{commit}` | **111 / 111** |
| spec declares a `control` | **111 / 111**, zero exceptions |
| PASS row carries `control_metrics` | 109; the two without are `T0.01` and `T0.10`, both declaring `control="NONE, BY DECISION (52nd audit B5)"` |
| implementation exists under `experiments/tests/` | **111 / 111** (the 11 harness-property specs that do not grep as `BY_ID["<id>"]` each have a file — `t0_18_record_reverdict.py`, `t0_21_coverage_audit_honest.py`, `t0_31_review_queue_cannot_go_quiet.py`, `t0_29_champions_tool_is_honest.py`, `t0_30_gate_cannot_demote.py`, `t0_36_frees_is_the_marginal_set.py`, `pl_00_encoder_cost.py`, …) |
| owning test passes `control_fn=` to `run_spec` | all but `T0.10`, which is the declared-NONE case |

Standing reporting-only weaknesses, all at floor and none new: **3 UNBACKED
CERTIFICATES** (`LF.02` ← `T6.03` → root `T2.10` FAIL; `T2.03` ← `T1.08` FAIL;
`T2.14` ← `T1.08` FAIL) with `pass_on_dead_dependency = 3` at its declared floor;
**2 dirty stamps** (`T6.03`, `PL.02`); **19 STALE CLAIMS** plus one pre-`impl_sha`
stale-by-content (`T2.02`), all on non-PASS rows so no capability claim rests on
them. `T0.27` is a deliberately-red gate at `live_violations = 3`.

## Section 2 — thresholds and controls over time

`git log -p --since="7 days ago"` over `registry.py`, `registry_expansion.py`
and `experiments/tests/` — 28 commits. **One finding, FINDING 1.** Everything
else in the window moves in the strengthening direction and says so with a
measurement:

- `e5e627b` — T2.15's tfidf null promoted **REPORTED → GATED**; new NULL-BEAT
  conjunct, strictly harder, explicitly cannot rescue attempt 2 (seed 2 routed
  5/16 vs TF-IDF 11/16). `CLAIM_MIN` 12 / `NB_REF_MIN` 13 untouched.
- `d186c07` — PS.09 gains a known-answer conjunct; `seed_gates_ok` now **also**
  requires `ka_ok`; every bar untouched; cannot rescue attempt 1's FAIL. It then
  VOIDed PS.09 on its own new gate, which is a control earning its keep.
- `a313122` — PS.08's amputation control was measured **invalid as a control**
  (`control_bal_acc` 0.708 **above** the registered probe 0.583) and the repair
  **deletes the leaking rate rather than masking it**; four bars named and
  unmoved; explicitly cannot rescue attempt 1.
- `88762a2` — LT.02 redesign keeps all five passing attempt-1 controls
  (*"a redesign may not drop a passing control"*), adds two, holds C1 at the
  unmoved 3.0 bar, and discloses that σ was fit on seed 90 only.
- `acad758` — T2.06 strengthened: a strict `>` that decided at zero margin got
  an exogenous margin.
- `da07ede` — LT.02 cost class `cpu<2h → cpu<10min` on a measured sizing record.
  This **loosens admission** and the commit says so, while **tightening** the
  child-kill window 54,000 s → 10,800 s; no ledger row existed under the old
  class. Justified.
- `955b9ef` — HR.1's registry amended to the experiment the row actually holds,
  `spec_sha` **moved on purpose** with the point stated (*"it read 'nothing
  changed' across a total venue replacement"*), and the headline
  `min_channel_leak_margin` made to BE the worst-seed margin the gate decides on
  (it had been reporting a seed-mean 12× looser under a `min_` name). No bar
  moved and no re-run was taken.

No control was deleted or weakened, no `_check` gained an `or`, no seed count was
reduced, no assertion was removed.

## Section 4 — is the builder alive and productive?

**Yes, and the conduct is the best thing in this report.** Seven iterations in
the 24 h window (`00:07`, `01:07`, `02:07`, `03:07`, `04:07`, `05:07`, `06:07`),
**7 of 7 `rc=0`**, zero dark slots, zero detached-pid leaks,
`lost_iterations.log` 0 bytes. PASS delta **111 → 111**.

Four real units landed: the hash-salt differential implemented and live-flown on
its own bill (`5ee32ff`, `9fa26c0`), the 119th audit's four FTB items all
discharged and verified against commits (`afe4d4f..8d8fb62`), BA.03's ruled
redesign implemented **a day ahead of its DUE** and its pilot dispatched,
harvested and frozen (`702aa56`, `4761d60`), and the coverage fixture inversion
with its staleness bill re-bought in the same slot (`a3313e8`, `46ff546`).

Three things specifically worth recording because each is a refusal:

1. The 03:xx slot found its own hand-forward **stale** (it pointed at LT.02's
   noise source, already PASSed at `d377874`) and traced every item to its
   discharging commit instead of inheriting.
2. The 06:1x slot found that `coverage`'s live-file fixture hard-coded a
   now-false premise and had been red since 03:xx while **two slots inherited
   "routed reds at floor" without reading the line** — it disclosed that rather
   than quietly fixing it, and repaired by **inversion, not deletion**.
3. It dispatched BA.03's pilot with a **FAIL forecast pre-registered**, and
   refused to manufacture a W38 GPU dispatch for the 39th consecutive slot.

## Section 5 — compute honesty

**GPU.** `2026-W38`: **0.9176 h drawn of 30** (summed per week from
`gpu_budget.json:charged_jobs`; 2 jobs). **~29.08 free Kaggle GPU-hours expire
tonight at 00:00 — the second consecutive week.** `gpu_hours_no_verdict` TOTAL
**48.42 h**, of which **`D1.0` is 33.78 h across 2 attempts and 0 verdicts** —
unchanged since 09-18 and still the largest single unredeemed spend on the
record. `gpu_unattributed_jobs` = 21, at its declared floor. **The builder's
refusal is correct and I endorse it for the 39th time:** every GPU cost class is
empty or not fillable, and an hour spent on a run nothing asked for is worse
than an expired one. The hours are not the scarce resource — the design sitting
is, which is precisely what `D36` asks the owner.

**CPU.** Today **8,427.66 s of 57,600 s**, of which **8,390.77 s** is the
BA.03 tilt pilot — one declared unit on one meter, on the sanctioned detached
lane, with a lane row and an EXITED stamp. The resulting 37 unaffordable
`cpu<2h` specs are the tenant protection working, not a fault. I checked the
handed-forward next dispatch against the ceilings rather than assuming: BA.03's
projected 3-seed run of ~7.0 h (25,200 s) fits both the `cpu<2h` child-kill
window (7,200 × 3 seeds × 2 = 43,200 s) and today's remaining 49,172 s. **No
class breach is forecast.**

## Section 6 — stuck decisions

**`MEANS-ESCALATED`: 0.** No fork that a measurement could settle is sitting on
the owner's desk. That is the `D1` disease and it is absent — worth stating
plainly, because this section exists because of it.

**`UNDECLARED`: 0** against a baseline of 10, so there is nothing for me to arm
this audit and the ratchet cannot shrink further by my hand.
**`OVERDUE — DEFAULT IS DUE TO FIRE`: 0.** `D31` fired by armed default at
00:5x (119th audit) and is transcribed onto `DECISIONS_RESOLVED.md` per `D13`.

Three `CONDUCT-DESK` flags: `D33` (due 09-23, stale 3 d), `D35` (due 09-24, stale
2 d), `D36` (due today). All three are the desk's own, and `D36` addresses its
own flag in-entry — correctly distinguishing the default it *has* executed from
the recommendation it may not take at its own desk, *"because a desk may not
re-order its own queue to move its own hardest unit off the only sitting big
enough to hold it."* That is the right refusal and it is why `D36` is a real
question rather than a deferral.

One `STEERING-DATE-MISMATCH`: `PROGRESS.md` quotes `D33` as 2026-09-27 where the
register says `decide_by 2026-09-23`. Reporting-only, and it is the stale page.

**Acted, not asked:** I appended a `RATCHET NOTICE` to `D33` recording FINDING 2's
live consequence and the one-line repair, with the observation that `D36`'s own
default already uses the exact `(CLOCK: …)` idiom the instrument is asking for on
`D33`. Nothing re-dated, no deadline extended; `decisions --check` output is
byte-identical before and after my append, which I verified.

## Section 7 — bakeoff hygiene

**One item, and it was already caught before I got here — which is the result
worth recording.** `T4.06` (fusion-balancing bakeoff, PASS attempt 1, the `+1`
that took the ladder to 111 on 09-23) certified `loss_reweight` on its
latent-recovery conjunct at margin **+0.0187 = 6.9% of the incumbent's own seed
spread of 0.2699, with 1 of 3 seeds REGRESSING**, and on `eval_loss_mean` at
**+0.0009 = 8.9% of a 0.0101 spread**. A winner inside the noise is exactly what
this section is for.

It was caught and disclosed: `f7900b5`, a doc-only amend under the 118th audit's
FTB 2, wrote the arrival note into the spec and states *"the ratio result is the
demonstrated thing, the latent-recovery conjunct is not to be quoted as
demonstrated"*, and `run status`'s ANCHOR-DECIDED CONJUNCTS block reprints the
margin, the spread fraction and the per-seed direction **every slot**. No gate was
touched and adoption remains the Review's. No decision in
`DECISIONS_RESOLVED.md` was made without a learning gate, and no VOID is treated
as a verdict there — the project's one VOID-as-verdict is `CHAMPIONS.md`'s
Learning-core seat, which is FINDING 4.

## Section 8 — the honest summary

**Are we closer to a curious humanoid that climbs the ladder than we were
yesterday? No.**

In the last 24 hours the ladder claimed nothing new about Jack. Twelve of
seventeen ledger events were the project paying its own instruments' staleness
bills. The constitutional ladder test came back **VOID** two days ago with every
arm — candidates, controls, random walker and null alike — reading `engaged`
0.0, and has no successor run. Three of the owner's own commitments (**smell**,
**shelter**, **too-cold-kills-him**) have no runnable falsifiable claim and have
not for 21 days, each behind a foreclosure whose own text says the repair is a
redesign nobody has been asked to write. Curiosity has 12 specs and 0 runnable
today. One-brain-in-unison has 28 specs and 1 PASS.

**And the counterweight, which is not small: we are measurably better at telling
the difference than we were yesterday.** Yesterday a false PASS on the headline
Tier-5 claim was found by the builder, in its own disfavour, and withdrawn —
112 → 111. This morning the same loop found a coverage fixture that would have
welded a runnable spec shut and repaired it by inversion; harvested a pilot whose
FAIL it had pre-registered and dispatched nothing to soften it; shipped a
liveness detector that immediately fired on its sibling organ's two-day silence;
and refused the expiring GPU hours for the 39th time. **Three organs got more
honest this week and not one got more optimistic.** That is the machine working
exactly as designed.

Which is also why this is DRIFTING and not ON TRACK. **A ladder whose
instruments improve faster than its creature does is being maintained, not
climbed.** The bottleneck is unchanged from the Review's own diagnosis of 09-24
and has now cost a second week of free GPU hours: the two 45-spec designs this
project actually needs — `T1.08`'s variance repair (`frees 3 / blocks 45`, and
the quotability of nine single-seed PASS certificates) and the W1 world-edit
window (three specs **nineteen days** unregistered) — have never once fitted
inside a sitting, and four of four Sunday FULLs have died at max turns. `D36`
puts that in front of the owner today, which is the right place for it.

---

## FOR THE BUILDER

1. **BA.03's two fractions — before the registered 3-seed run, not after
   (FINDING 1).** The docstring at `:330-334` and the constant comment at
   `:601-610` both say the claim floor was "rounded toward the demanding side",
   and the gate at `:1141` is a floor, so `0.20` against the paragraph's own
   derived `0.25` is **20% more permissive**, not less. Two honest repairs, your
   choice, and the strengthening is **free today because the pilot forecasts
   FAIL**: (a) restore `TILT_GAIN_MIN_FRAC = 0.25` and
   `TILT_VEST_OVER_NOISE_FRAC = 0.074`, or (b) keep 0.20/0.05 and correct the
   two labels to say **permissive**, stating in the same sentence that the claim
   floor sits 20% below and the noise floor 32% below their stated arithmetic,
   with the reason. What is not a repair is leaving a discount justified by a word
   that names the opposite direction. **Do not move either number after a
   registered seed has been read** — your own docstring forbids it and it is
   right to.
2. **Join `decisions.py`'s five ratcheted classes into `run.py:ratchet_live()`
   (FINDING 2)** — `UNDECLARED`, `UNROUTED-OWNER-ASK`, `VANISHED-OWNER-ASK`,
   `DEFAULT-ACTION-EXPIRED`, `FIRING-DIFF` — five `take()` calls in the
   `_champions_unwinnable` idiom, whose own comment is the precedent
   (*"surfacing it HERE is the other half"*), then `run ratchets record` so the
   currently-broken `DEFAULT-ACTION-EXPIRED = 1` gets an `"at"` date and any
   future move gets a `!! MOVED` banner and an attributable cause. Price the
   staleness bill first as usual.
3. **Then the durable half of the same finding:** a property asserting that
   `ratchet_live()` names every class in every tool's floored-class dictionary
   (`coverage.py`, `champions.py`, `review_queue.py`, `decisions.py`), so the
   next instrument cannot ship a floored class nobody joins. `T0.31`'s
   assert-on-the-TOTAL shape is the right idiom. This is the finding, not item 2
   — item 2 is the instance.
4. **`scripts/lib_liveness.sh:189` calls `_seal_file_age_hours`, which is defined
   in `lib_seal.sh` and is not sourced by `lib_liveness.sh`.** In production this
   is fine and I verified it rather than reporting a phantom: `overseer.sh`
   sources both in the correct order (`:31-32`) and the assertion fires live. But
   a caller that sources `lib_liveness.sh` alone gets `command not found`, then
   `[: : integer expression expected`, then `return 1`. It **fails closed**, so
   this is not a repeat of yesterday's FINDING 3 — it is one guard line on a file
   whose entire job is not going quiet. Low priority, cheap.
5. **Nothing else is owed by you today and the empty board is a true reading.**
   Four units in six slots, zero manufactured work, a pre-registered FAIL
   forecast dispatched anyway, and a fixture red you found by reading the line
   two slots had inherited. Keep re-deriving rather than inheriting; it caught a
   real defect this morning.
6. **`cross-organ-doc-race-voids-certificates` is yours by its own text** (*"what
   this date owes is EXECUTION by the builder, not a decision by this desk"*) and
   is a legal pick today. Its sibling `waits-on-declared-field` was **ACTED by
   the Review at `3c7f90e` during this audit**, so check the queue live before
   picking — the Review's DAILY was still sitting when I committed and took the
   OVERDUE class from 8 to 5 while I wrote.
7. **Still do not pre-empt:** `W1.01`/`W1.03`/`W1.04` registration, `D33`/`D35`/
   `D36`, `UB.10`'s successor arm choice, `T1.08`'s pipeline design, the
   `w1-world-edit-window` docket, the `lc03` seat row, the `t306` venue row. The
   W1 registrations are exactly the work you are idle for and `D33` is the open
   entry on whether they are yours; taking them early would pre-empt a question
   this project put on the owner's desk.

---

## FOR THE OWNER

**1. `D36` falls due TODAY, and it is the one entry on your desk that decides
whether anything moves this week. Cited, not re-asked.** Two 45-spec designs want
one Sunday sitting; four of four Sunday FULLs have died at max turns; an armed
default already ordered W1 first, and the Review explains — correctly, in my
reading — why it may not reverse that ordering at its own desk even though it
believes the swap is right. The realised cost is now doubled: **~29.08 free
Kaggle GPU-hours expire tonight, unbought, for the second consecutive week**, and
both weeks have the same cause — `T1.08`'s repair has no design and the only
sitting large enough is taken. The hours are not the scarce resource. The design
sitting is.

**2. NO-DECISION — your Review has not published in two days, and I want you to
know it was not idle.** The 09-25 sitting disposed five queue rows and authored
`D36` itself, all committed between 06:38 and 06:52, and then died before writing
its page. `docs/PROGRESS.md` now carries a machine-stamped STALE banner — built by
the builder yesterday, fired at 42 hours, re-verified by me this morning — so the
page can no longer be read as current state. But the page under the banner is
09-24's, and its headline (*"the builder is back and has nothing to do"*, at
`week:all models` 19%) is false of today, when the meter read 54–57% and the
builder shipped four units. **Nothing is hidden and nothing is current**, and the
desk's own stop-rule on `w1-world-edit-window` fires tomorrow either way.

**3. NO-DECISION, and it is the direct answer to what this project is for.
Three of your own constitutional commitments have no runnable falsifiable claim
at all:** **SMELL**, **SHELTER**, and **TOO COLD / TOO HOT KILLS HIM**. Every
claim spec behind them is PARKED or FORECLOSED, and each foreclosure says the
same sentence — *the repair is a redesign, never a dispatch*. `goal_unrunnable`
has read **7** for **21 days**. No instrument is broken and no organ is at fault;
there is simply nothing on the board to run for those three, and nobody has been
asked to design the successors. `SM.03` and `SH.02` are both PILOT-BLOCKED with no
row that owes a design. If you want one thing added to the docket beyond `D36`,
this is it — it is the difference between a jungle survival creature and a
locomotion demo.

**4. `D33` is carrying the project's only broken ratchet, and the repair is one
line. Appended, not re-asked.** `decisions --check` has read
`DEFAULT-ACTION-EXPIRED = 1, baseline 0` for three days because `D33`'s armed
default names an action dated 2026-09-23 that is already in the past on the first
day the default can fire — so **the default that has fired four times cannot be
executed as written**. The fix is the desk's: shorten `decide_by`, or annotate
the date `(CLOCK: <whose>)`. The idiom already exists in this file — `D36`'s own
default uses it verbatim, authored one day after `D33` went red. I have appended a
`RATCHET NOTICE` under `D33` with the evidence. Nothing re-dated.
