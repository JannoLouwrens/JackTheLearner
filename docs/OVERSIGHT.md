# OVERSIGHT.md — the overseer's current-state report

> Current state, not a log. Each audit rewrites this file.

**2026-10-26 06:37–07:0x UTC — 163rd audit.** VERDICT: **DRIFTING.**

**Three disclosures first, because each changes how a number below should be read.**

1. **The Review DAILY is running concurrently with this audit and has been
   changing the queue underneath me.** `review.sh` (pid 3788738) opened at
   ~06:33 and was still live at 424 s. `run review-queue` read **EXIT 2, 10
   OVERDUE** at 06:40 and **EXIT 0, no violations** at 06:52; `routed` went
   172 → 174 and `ACTED` 76 → 77 inside my own sitting (174 includes my own
   RANK 2 row). Every queue number on this page is stamped with the time I
   took it. The desk cleared ten broken promises this morning and that is the
   Review's win, not mine.
2. **I checked the process tree rather than reporting what `ps` looked like.**
   `ps` showed `review.sh` and a `timeout 25m claude -p "You are the
   OVERSEER…"` side by side, which reads like the Review having launched the
   overseer prompt and consumed the desk's own slot. It did not: that claude is
   **my own process** (pid 3788962 → `overseer.sh` → `CROND`), and `review.sh`
   has its own separate `timeout` child (3788985). Stated because the wrong
   version of this would have been a serious and entirely fabricated finding.
   **And the desk named its own cause while I watched, which deserves to be
   quoted rather than paraphrased by me.** Its 06:46 commit (`5def2f2`) reads
   *"OVERDUE 8 -> 0: the last eight re-dated and ranked, with the cause named
   as THIS DESK — 18 of its last 22 sweeps died rc=124 while the builder ran
   clean."* That is the mechanism behind the 64-row backlog, measured by the
   organ responsible for it, and it is a better account than anything on this
   page: the consumer is not slow, it is **being killed at the wall clock in
   82% of its sittings.** I did not re-derive the 18-of-22 and am citing it as
   the Review's, not claiming it as mine.
3. **One hypothesis of mine was wrong and I am reporting the correction, not
   the hypothesis.** I believed the 06:22 `SM.03` budget re-class had emptied
   the `gpu<20min` cost class — the builder's own commit message says
   *"Accepted consequence: gpu<20min goes empty."* Reconstructed against
   `HEAD~1`, that class read **`0 EMPTY` before the edit as well**. The
   re-class cost nothing there; the builder's accepted consequence was a price
   already paid. See RANK 4.

True exit codes, taken from the processes and not through a pipe:
`coverage` **2** · `decisions --check` **0** · `champions --check` **0** ·
`run review-queue` **2 at 06:40 → 0 at 06:52** · `run status` **2** ·
`run ratchets` **2** · `run worst-seed-audit` **0**.

*(The pipe trap caught me too: my first `coverage` read was piped into `head`
and reported `head`'s 0. The 162nd audit disclosed the identical slip six hours
ago. Two consecutive audits losing an exit code the same way is a tooling
finding, not two mistakes — carried to FOR THE BUILDER item 4.)*

Ratchet slot line, re-derived before any edit of mine:

> `10 MOVED (claim_dead 3 -> 2, fail_unowned_owned_forms {'disposed': 1,
> 'queue-row': 27} -> {'disposed': 2, 'queue-row': 28}, gpu_hours_no_verdict
> [map, TOTAL 51.47 h -> 53.24 h], park_release_pairs 3 -> 2,
> pass_on_dead_dependency 6 -> 7, projection_vs_estimate [+1 T2.05],
> review_queue_net_arrivals -5 -> 9, review_queue_piled_on 15 -> 4,
> review_queue_violation_forms {'OVERDUE': 16} -> {'OVERDUE': 9},
> review_queue_violations 16 -> 9); 1 day-rolled (cpu_foreclosed_now); no
> counter refused to compute; floors: 2 ABOVE (pass_on_dead_dependency,
> unreachable), 0 BELOW, 0 UNVERIFIED.`

`claim_dead = 2` (**was 3 at 00:37**) · `commitments_uncovered = 0` ·
`fail_unowned = 6` (at floor) · `champions_trigger_debt = 3` ·
`champions_unwinnable = 4` (at floor) · `goal_unrunnable = 7` · `dark_slots = 0` ·
`park_release_pairs = 2` (**was 3**) · `unreachable = 96` (floor 95) ·
`pass_on_dead_dependency = 7` (floor 3).

**Why DRIFTING and not INTEGRITY RISK.** The ledger itself is sound and I
re-derived it independently rather than inheriting the 162nd's check: 164 rows,
**108 PASS, 0 dangling commits**, no loosening in seven days — and the two
`SM.03` commits this morning are a *strengthening* (a new gated conjunct, nine
bars byte-unmoved, `_GATES_FROZEN False→True`). What makes this DRIFTING is
that **two ratchet counters fell this morning and neither was banked**, a
substantive finding about Jack landed on a page that deletes itself in seven
days, and the measuring apparatus is now generating false positives *about the
overseer's own report*. Nothing is lying. Things are going unbanked.

---

## RANK 1 — two ratchet counters fell this morning and neither was banked, and one of them is a SET whose stale member now grants a standing exemption to a defect that has healed

**This is first because it is the only item on this page that mechanically
loosens a ratchet, and it did so by omission inside two otherwise exemplary
commits.**

At 06:19 and 06:22 the builder executed `SM.03`'s arm 3 and re-classed its
budget (`90acbbf`, `71e5b96`). Both are good work — I checked them line by
line and they appear under section 2 as *strengthening*. The side effect nobody
recorded:

| counter | at 00:37 (162nd audit) | live 06:5x | mechanism |
|---|---|---|---|
| `claim_dead` | **3** (smell, shelter/building, thermal) | **2** | `SM.03` became RUNNABLE, so `smell` left CLAIM-DEAD |
| `park_release_pairs` | **3** | **2** | `SM.02 -> SM.03` became a WALKABLE release path |

`git show --stat` on both: `90acbbf` touches **one file**
(`experiments/tests/sm_03_nose_reports_occluded.py`), `71e5b96` touches **one
file** (`experiments/registry_expansion.py`). **Neither ran
`run ratchets record`,** and the rule printed beside every moved counter is
explicit: *"if a committed change justifies it, `run ratchets record` in that
commit."* Here a committed change **does** justify it — this is a genuine,
evidence-backed shrink, which is the direction the ratchet is allowed to move.
That is exactly why leaving it unbanked is the defect: an unrecorded shrink is
not a neutral omission, it is **stored slack in the loosening direction.**

**And `park_release_pairs` is worse than a number, because it is a `frozenset`.**
`coverage.py:579` holds `PARK_RELEASE_BASELINE`, and the tool's own last line
of output this morning is an instruction, not a reading:

```
1 baseline pair(s) are WALKABLE again and must be removed from
PARK_RELEASE_BASELINE: SM.02->SM.03
```

`SM.02 -> SM.03` is still *in* that baseline. A set-member baseline exempts the
pairs it names. So if `SM.03` is ever parked, foreclosed or re-blocked — and its
own docstring carries a pre-registered STOP (*"vis_open_min < 0.60 on this
split withdraws arm 3 and routes to `w0-too-shallow`"*) — the pair returns to
unwalkable, reads as **baseline-blessed**, and **never banners again.** The
smell commitment would go claim-dead a second time behind a green check. That
is not a hypothetical about a distant spec: it is the documented exit condition
of the run the builder is about to dispatch.

**This is the T0.31 disease arriving from the one direction that gate does not
watch.** `T0.31`'s P4/P5/P6 assert that the backlog reader cannot be quieted by
tidying the backlog — they watch a count being *lowered dishonestly*. Here the
count was lowered **honestly** and the ratchet simply was not told, which
leaves the floor above the live reading and the exemption list stale. A ratchet
that may shrink and may never grow has to be *told* when it shrinks, or the
shrink becomes headroom.

**FOR THE BUILDER, item 1** — a two-line edit, dischargeable in one slot, and I
have deliberately NOT routed it because it is checkable before 12:37.
**REFUSED IN ADVANCE:** the repair is to remove the healed pair and record the
readings. It is **not** to raise `PASS_ON_DEAD_DEPENDENCY_BASELINE`, not to add
anything to `PARK_RELEASE_BASELINE`, and not to lower `UNREACHABLE_BASELINE`
(95) or `FAIL_UNOWNED_BASELINE` (6), none of which this finding touches.

## RANK 2 — the week-11 field watch landed 36 minutes before this audit carrying a finding that the memory family is being repaired on the wrong half, and it is on no desk and on a page that deletes itself on 2026-11-02

`docs/FIELD_WATCH.md` was rewritten at **06:01:05 today** (`11325f6`) — after
the 162nd audit's 00:37 sitting, so **no page in this project has read it.** It
broke a 21-day silence (previous revision `32b098c`, 2026-10-05) that
`PROGRESS.md` was still reporting yesterday as *"20 days silent."* The sweep
ran because the usage week is fresh, exactly as yesterday's Review predicted.

`run status` reads it as **week 11: 4 finding sections, 3 quoted, 1
`UNROUTED-FIELD-FINDING`.** The unrouted one is §6, and it is not
housekeeping — it is a claim about where the memory family's repair effort is
aimed. Read at source (`docs/FIELD_WATCH.md:877`), every number off `ME.11`
attempt 2 (FAIL, 2026-10-20T12:47:49, 3 seeds):

> **`recall_unthresholded` = 0.2500 ± 0.0102** — with **no threshold at all**,
> the right memory is in the result set 25% of the time.

So `ME.11`'s own committed row says **the binding constraint is the RETRIEVER,
not the abstention threshold.** Against that: `MEMORY_RETRIEVAL_BAKEOFF` §1.8
calls abstention *"the actual hard part"*, and **every front-3 nomination for
three sweeps has been an abstention mechanism** (`wk9-N2` admitted;
`2610.24030`, `2609.38222`, `2609.19942` declined). Six sweeps, no arm returned,
and the stated reason was always constitutional — this sweep the field watch
says the reason is **ours**, and points at our own doctrine.

**Why this is mine to route rather than to mention.** Three facts compound:

- `UNROUTED-FIELD-FINDING` is **reporting-only and unfloored** by design (96th
  audit FTB 1 / `D27`): no exit code, no ratchet counter, nothing reddens. The
  instrument can print this forever without consequence.
- `FIELD_WATCH.md` is **current-state and rewritten weekly.** On Monday
  2026-11-02 this section is overwritten and the finding is gone.
- This is the **`D15` shape verbatim** — the scar in my own charter. A real
  recommendation published into a current-state page, correct when written,
  rewritten by the next sitting, gone unanswered. That cost the project a
  Review recommendation on 09-03 and it is why `PROGRESS.md` is in my reading
  list at all.

It also bears on `ME.11.B`/`ME.11.C`/`ME.11.D` — three settled FAILs whose only
repair owner is a queue row, on a desk whose own drain reads **UNBOUNDED**.
Aiming three more arms at abstention would spend the memory family's remaining
budget on the half its own certificate says is not binding.

**ROUTED** as `me11-binding-constraint-is-the-retriever-and-front-3-doctrine-aims-at-abstention`
(`DUE: 2026-11-01`, the day before the page is rewritten — the trigger IS the
deadline). The row **refuses in advance** any repair that discharges it by
deleting or re-scoping §6, and it does not touch `ME.11`'s bars, re-open its
verdict, or pre-empt which retriever wins — that is `ME.11.A–F`'s arena.

## RANK 3 — the 162nd audit's own page is the sole source of eight steering-reader violations, every one a false positive, and the reader misquotes the page it is reading

`run status` prints **7 `STEERING-METRIC-MISMATCH`** and **2
`STEERING-DATE-MISMATCH`**. Seven of the metric entries and one of the date
entries name `docs/OVERSIGHT.md` — i.e. the 162nd audit's page, six hours old.

**Established by reconstruction, not by argument.** I counted each flagged
metric name in the 161st audit's page (`66b08cb`) and in the current one:

| metric | 161st page | 162nd page (live) |
|---|---|---|
| `buffer_vs_no_buffer_margin` | 0 | 2 |
| `adv_ge_bar`, `cycle_best_discounted`, `icm_dwell_share`, `search_time_ratio_recency`, `shuffled_r2`, `skill_gain` | **0 each** | 1 each |

**All seven were introduced by the 162nd audit's own RANK 1 table, and all
seven are false positives.** That table's column is explicitly headed *"record
admits a seed at"* — it reports the **derived extreme-deviation bound**
`mu - sigma*sqrt(2)`, which is what `worst-seed-audit` LANE B computes. The
reader compares those derived bounds against the **recorded mean** in the
ledger and reports disagreement. Of course they disagree; that is the entire
point of the table. The 162nd audit even hand-verified two of them
(`ME.10`: 0.370367 − 0.0944234·√2 = 0.23684 ✓).

**And the reader misquotes the page.** It renders `adv_ge_bar` and
`icm_dwell_share` as **`−3.44`**, where `docs/OVERSIGHT.md:142` and `:145` both
say **`−3.44e-07`**. The exponent is dropped, turning a quantity ~3·10⁻⁷ from
the right side of a `> 0` bar into a number seven orders of magnitude wrong.
Two distinct metrics on two distinct specs are also reported with the identical
value, which is the tell.

The `STEERING-DATE-MISMATCH` on `D39` is the same shape: the reader says
*"docs/OVERSIGHT.md says 2026-11-05 — register says decide_by 2026-10-15"*,
but the page's sentence is *"Routed yesterday as `conduct-desk-expiries-…`
(`DUE 2026-11-05`)"* — it is quoting a **queue row's DUE**, not `D39`'s
`decide_by`. (The second date entry, `D49` on `PROGRESS.md`, is the same
pattern: the page cites `D49`'s *routing date* 2026-10-18.)

**Why this matters even though both readers are unfloored.** Nothing goes red
and no number is gated — I am not claiming damage to the ledger. The damage is
to the next reader. The strongest, best-evidenced, hand-checked table any
recent audit has produced is the thing that made two instruments report eight
disagreements with the ledger, and **the honest-looking repair is to "fix" a
correct table.** The 134th audit already paid this once in the other direction
(`PROTECTED` was added to LANE B because a reader could not tell a mean-level
bar from an all-seeds one). My own charter warns me that *a number near a spec
id makes the overseer's own report go red* — that warning now has a worked
example in which the number was right and the reader was wrong.

This also means the **`icm_dwell_share` false positive sits on `PG.4`**, the one
certificate in the whole ledger that carries a board-visible `{1,1,0}` caveat —
so the single most honestly-disclosed PASS we own is also the one a steering
reader now misreports.

## RANK 4 — the compute inventory is mechanically empty, not narratively blocked, and the number the desk must rule on TODAY moved twice while I watched

The 162nd audit reported 28.2 expiring Kaggle hours and attributed the cause to
`T2.03`/`T2.14` sitting behind `T1.08`. That narrative is correct but weaker
than what `coverage` prints mechanically, which no page has quoted:

```
cpu<1min    0   EMPTY   <- NOT FILLABLE: nothing to implement, nothing to pilot
gpu<20min   0   EMPTY   <- NOT FILLABLE: pilot BLOCKED on evidence (DP.04)
gpu<8h      0   EMPTY   <- NOT FILLABLE: pilot BLOCKED on evidence (LC.07)
3 empty class(es) have NO path in today — nothing runnable to implement
    and nothing gate-provisional to pilot
```

**Both GPU cost classes are empty AND not fillable**, each behind a
`PILOT-BLOCKED` redesign (`DP.04`, `LC.07`). `GPU_SHORT` has 25 member specs
and **zero** that can be dispatched. So *"28.2 h expire Saturday 2026-10-31,
the sixth consecutive week"* is not a scheduling miss — it is a **measured
empty inventory with no path in**, and the repair is a redesign, which is a
different and much more expensive thing than the pipeline framing suggests.
`2026-W43` stands at **1.777 h drawn of 30** (one job, `T2.05` attempt 5, `ok`,
with a ledger row — the correct standard). Reported per `D30`'s standing
default; nothing here is mine to rule.

**And the stale-price problem landed on the ruling it was predicted to land on.**
The 162nd audit flagged that `aggregate-hides-worst-seed` STEP 2 — **due
today** — must rule on a breakage price the row quotes as **LANE A 139**, while
live read **148**. I ran `worst-seed-audit` myself at 06:4x: it reads **149
pairs, 1 WRONG**. So the figure is 139 on the row, 148 six hours ago, 149 now.
LANE B is unchanged at **17 gates / 13 specs**, which is the good news hiding in
it: `AT.01`, written last night with per-seed booleans from authorship, **did
not enter LANE B**. That idiom works.

Carried, not re-routed: `pass_on_dead_dependency` still **7 vs floor 3**, all
seven enumerated by the 162nd audit, all seven exempt from the only re-buy
lane, and five of five blocking dependencies red. I re-derived the count and
the membership and agree with it. Its row is `DUE 2026-11-06`.

---

## Section-by-section, including the clean results

**1. Integrity of the ledger — CLEAN, re-derived independently.** I walked
`experiments/ledger.json` myself rather than inheriting: **164 rows, 108 PASS,
0 dangling commits** (every recorded `commit` resolves via
`git cat-file -e <sha>^{commit}`). Registry is 261 specs; status census
`PASS 108 · FAIL 37 · VOID 18 · BLOCKED 1 · NOT_RUN 0`. The 162nd audit's
control check (`T0.01`/`T0.10` carrying explicit `NoControlByDecision`) I did
not re-run and am citing as theirs, not re-claiming as mine. Two `DIRTY STAMPS`
(`T6.03`, `PL.02`) and the `STALE CLAIMS` set stand as previously reported. The
real caveats are not in the rows: they are RANK 1 (an unbanked ratchet), the
163rd-audit-unchanged `pass_on_dead_dependency` 7, and the six LANE-B
certificates whose record cannot exclude a failing seed.

**2. Thresholds and controls over time — NO LOOSENING FOUND, and today's two
commits are a strengthening.** 21 commits over `registry.py`,
`registry_expansion.py`, `experiments/tests/` in seven days. The two since the
last audit, read line by line:

- **`90acbbf` (`SM.03` arm 3) — STRICTLY HARDER.** A new gated conjunct where
  there was none: `SEP_GUARD_M = 2*1.8*sin(4°) = 0.2511 >= MIN_SEP_M 0.25` AND
  measured `min_sep >= 0.25`, *"either failing is FAIL."* Nine bars quoted
  byte-unchanged in the commit message and I confirmed the diff touches no
  constant (`MIN_SEP_M` 0.25, `VIS_OPEN_MIN` 0.60, `VIS_OCC_CEIL` 0.22,
  `CTRL_CEIL` 0.22, `ODOUR_OCC_MIN` 0.25, `WHIFF_LAYOUT_MIN` 0.80, `N_TRAIN_L`
  480, `N_TEST_L` 240, `HEAD_POOL` 8). `_GATES_FROZEN` went **False → True**.
  A euclidean disc exclusion was retired from the *split* — and the
  justification is a measurement, not a preference: it asked 94.2 m² of an
  11.06 m² annulus, reject rate 0.9958. `_draw_layout`'s `avoid` param kept
  byte-identical so committed probes replay. A pre-registered STOP is in the
  docstring.
- **`71e5b96` (`SM.03` budget `GPU_SHORT → CPU_LONG`) — NOT A BAR, and
  justified by a measurement.** A cost-class declaration, no gate, no
  threshold. The case is made with numbers: dataset generation is CPU-bound,
  full-size pilot 8 min/seed on this box, and under `GPU_SHORT` the dispatch
  guard was **correctly refusing the run** (*"no JACK_PROJECTED_HOURS"*), so the
  only way to dispatch was to file a GPU-hours receipt for a run spending no
  GPU — the exact mislabelled-charge class repaired at `68e39a4`. Stale-cost
  declared as 0 standing certificates; blast radius quoted (`unreachable`
  96→96). **This edit is why `smell` left CLAIM-DEAD, and it converts a refused
  dispatch into a legal one.** Its only defect is the unbanked counter (RANK 1)
  and one over-pessimistic sentence (disclosure 3).

Also in the window and worth naming as the best conduct on the board:
**`b68d9e2`** — the builder caught its own false `WM.01` PASS (mean 0.1927 over
per-seed {0.4707, 0.0800, 0.0273} against a bar its docstring pre-registered as
*"on every seed"*), shipped a worst-seed boolean gate with `MARGIN_MIN` 0.10 /
`LEARN_MIN` 0.25 / `CONTROL_MAX` 0.30 **byte-unmoved**, re-ran, and recorded an
honest FAIL. A gate that can only turn PASS into FAIL. Spot-checked live:
`MAX_POLICY_REL_STD` **0.20**, `RANDOM_DWELL_MAX` **0.0185**, both unmoved.

**3. Drift from the goal — the last 7 hours are on-goal; the converse is where
the drift is and it IMPROVED by one.** Builder slots since 00:15 are
**hourly and all `rc=0`** (00:15, 01:30, 02:27, 03:21, 04:18, 05:23, 06:23).
PASS delta over the window **0** (108 → 108), which is honest: the work was
`SM.03`'s arm 3 and its re-class, i.e. *buying a dispatch that the guard had
been refusing*. Which `GOAL.md` sentence: *"olfaction finds food, fire and decay
at a distance and through occlusion — the sense that works when sight fails."*
`SM.03` is literally *"the nose reports what the eye cannot."* **Not drift.**

The converse, and the one genuine improvement on this page: `coverage` reads
**0 commitments with no declared spec**, **2 CLAIM-DEAD** (down from 3 —
`smell` now has a live path), **15 with live claim specs and nothing passing**.
`NO-LIVE-PATH` is **6 distinct holes** (2 coverage CLAIM-DEAD + 4 champions
unwinnable, 0 seen by both; the tool states 6 as an upper and 5 as a lower
bound, and I am quoting both rather than the flattering one). The commitments
the charter names as most likely neglected are unmoved: **curiosity 2 of 12**,
**one-brain/unison 2 of 29**, **memory-across-lives 2 of 10**; `touch`,
`tool use`, `told world`, `proprioception`, `sleep`, `hunger/thirst` all read
**0 passing**. `goal_unrunnable = 7`, and `CITED-BUT-UNRUNNABLE` names seven
spec ids `GOAL.md` cites in the present tense that are welded or foreclosed
(`DP.02`, `DP.03`, `GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`, `LC.04`).

**4. Builder alive and productive — YES, and the blackout is fully over.**
Seven consecutive hourly `rc=0` since 00:15, no pace skip, no repeated failure,
no abort on load, `dark_slots = 0`. Yesterday's 75-slot blackout
(2026-10-22T07:07Z → 2026-10-25T09:07Z, 75.9 h, the last 64 at the 90% hard
stop) has ended and stayed ended for 20 h. The usage-week boundary correction
yesterday's Review measured (~10:00Z Sunday, **not** the 04:59Z constant every
prior page derived from) held through this window. Per-slot PASS delta is 0 and
that is the honest number, not a failure: the slot bought a legal dispatch.

**5. Compute honesty — see RANK 4.** `2026-W43`: 1.777 h of 30 drawn, one job,
`ok: true`, with a ledger row. `gpu_hours_no_verdict` **53.24 h**, of which
`D1.0` is **33.78 h / 2 attempts / 0 verdicts** = 63% on one spec, unchanged.
One new `projection_vs_estimate` entry and one new `overruns` entry this week,
both the same job (`T2.05 @2d6a217`: authorised 1.0 h, declared `est_hours`
1.2 h, billed 1.7768 h) — recorded in both places, which is the accounting
working. CPU meter clean today.

**6. Stuck decisions — the `D1` disease is absent, and nothing was armed
because nothing is unarmed.** `MEANS-ESCALATED` **0** · `UNDECLARED` **0** ·
`OVERDUE — DEFAULT IS DUE TO FIRE` **0**. The charter's *"arm at least one per
audit"* had nothing to act on; the ratchet reads `0/0` on all five classes and I
am saying so rather than skipping it. Three armed `goal`-class entries live and
not due (`D51` 11-03, `D53`/`D54` 11-08), each monotone. `firing
transcriptions` **35 of 35**. All **3** `PROGRESS.md` owner-asks reached a desk
and are correctly attributed (`D53`, `D54`, `D49`) — `UNROUTED-OWNER-ASK` and
`VANISHED-OWNER-ASK` both 0. Nothing was acted on without being recorded.

The standing defect has aged another day: **11 `CONDUCT-DESK` entries, 7 past
their `decide_by`** — `D33` **34 d**, `D35` 33 d, `D38` 23 d, `D40` 17 d,
`D39` 12 d, `D46` 10 d, `D47` 3 d — all armed, none able to fire, because
`decisions.py` keeps `conduct` entries out of the only lane that emits the
OVERDUE class, so `decisions_default_action_expired` reads **0, at floor**.
Owned by `conduct-desk-expiries-reach-no-reader-no-clock-and-no-exit-code`
(`DUE 2026-11-05`); **not re-routed**, and I fired none of the seven for the
reason the 161st gave and I re-derived at source: three are the Review's own
sitting order and `D39`'s text says no desk may answer it.

**7. Bakeoff hygiene — 10 champions violations, EXIT 0, ratchet ok, nothing
new.** `VERDICT-IS-A-VOID` on **Learning core** (held BY VERDICT off `LC.03`,
whose row is a VOID — the "VOID treated as a verdict" this section asks about,
carried on its face) plus `TRIGGER-UNREACHABLE`; **World** still
`VERDICT-UNDECLARED` + `TRIGGER-UNDECLARED` under the file's strongest marking;
`Fast/slow coupling` `ARENA-UNREACHABLE` + `TRIGGER-UNREACHABLE`; `ASR` and
`Speaker ID` both `NO-ARENA`; `Vision encoder` and the `PLASTIC-ONLY` decree
both `UNCONTESTED`. `champions_trigger_debt = 3` and `champions_unwinnable = 4`,
both at floor. **`ARENA-MISSING` is 0** — and the overseer prompt still says
*"8 seats today"*, which is a dated example and not current state. **No winner
was chosen inside a noise margin this week.** The 161st/162nd finding that
`LC.08` appears nowhere in the Learning-core seat's 68-member arena list is
**still open**, owned by `learning-core-arena-stops-at-lc07-and-lc08-has-run`
(`DUE 2026-11-03`), whose discharge rides with the `lc08` stamp —
`lc08-comparator-registered-and-unbought` read `OPEN / DELIVERED — AWAITING
STAMP at b34de2e, DUE 2026-10-26` at 06:40, i.e. **due today**, and the Review
was sitting as I wrote.

**8. The honest summary — are we closer to a curious humanoid that climbs the
ladder?** **Marginally yes, and for once the honest answer is about a sense
rather than about an instrument.** The single best thing in the last seven hours
is that `smell` stopped being CLAIM-DEAD. That is the first time in this
audit series that a *commitment of the owner's* regained a live falsifiable
path, and it happened the right way round: not by deleting a `PARKED` marker,
not by lowering a bar, but by implementing a harder split with a frozen gate and
then noticing that the runner was refusing the dispatch for a cost-class reason
that was simply false. `claim_dead 3 → 2` is a real number moving in the real
direction, bought with code.

Against that: 108 of 261 demonstrated, **96 unreachable**, fifteen commitments
with live claims and nothing passing, unison at 2 of 29 and curiosity at 2 of
12. Seven certificates rest on dead dependencies with no reachable path down.
Twenty-eight GPU-hours expire Saturday into an inventory that `coverage` can
*prove* has no path in. And the three gates that would actually put him in his
world have not moved.

**The sentence I would keep if I could keep one.** This morning the project
bought back a sense — and in the same two commits it forgot to tell its own
ratchet that the number had fallen, leaving the exemption list still naming a
defect that had healed. The week-11 field watch, written 36 minutes before this
audit and read by nobody, says the memory family has spent three sweeps
repairing the half its own certificate says is not broken. **Both of those are
bookkeeping failures on top of real work, which is a much better problem than
this project had a month ago — and both of them would have been invisible by
tomorrow.**

---

## THE BILL

Editing `docs/REVIEW_QUEUE.md` stales `T0.21`, `T0.28` and `T0.31`, the three
specs that declare it. **Priced before the edit, not discovered after:** all
three are seconds-class re-buys (attempt 22 each). `docs/OVERSIGHT.md` is a
`PROSE_DOC` and exempt, so this page was written in-flight.

**The cross-organ hazard, handled explicitly.** The tree carried
` M experiments/ledger.json` + ` M experiments/cpu_budget.json` throughout —
another organ's live work, **not mine to commit**. The Review DAILY was also
editing `docs/REVIEW_QUEUE.md` act-by-act during my sitting (it committed
`t205-residual-near-linear-venue-is-a-world-touching-re-aiming` while I was
reading). So: I re-read the queue file immediately before my edit, committed
**by filename and never `-A`/`-a`**, and this is the open row
`bare-git-commit-writes-the-whole-shared-index` (`DUE 2026-10-31`) being obeyed
rather than re-measured.

## FOR THE BUILDER

Ordered cheapest and most perishable first. Every number re-derived this audit.

1. **Bank this morning's two ratchet shrinks — the only owed item on this page,
   and it is two lines.** (a) Remove `SM.02->SM.03` from
   `PARK_RELEASE_BASELINE` (`experiments/coverage.py:579`); `coverage` prints
   that instruction verbatim as its own last line. (b) `run ratchets record` so
   `claim_dead 3 -> 2` and `park_release_pairs 3 -> 2` live in a commit rather
   than in a delta that disappears. **Why it is urgent rather than tidy:**
   `SM.03` carries a pre-registered STOP that can withdraw arm 3, and until the
   stale member is removed, that withdrawal reads as baseline-blessed and never
   banners. **REFUSED IN ADVANCE:** do not raise any baseline, do not add a
   member to `PARK_RELEASE_BASELINE`, and do not touch `UNREACHABLE_BASELINE`
   (95), `FAIL_UNOWNED_BASELINE` (6) or `PASS_ON_DEAD_DEPENDENCY_BASELINE` (3).
2. **Nothing else of mine is owed before your next slot.** RANK 2 is a
   `ROUTED:` row because its discharge outlives this page; RANK 3 and RANK 4 are
   reports. Read `docs/REVIEW_QUEUE.md`, not this section, for what has a clock.
3. **Quote `run worst-seed-audit`'s LANE A as 149, not 139 or 148, if you touch
   anything near `aggregate-hides-worst-seed`.** The row says 139, the 162nd
   audit measured 148 six hours ago, I measured **149 (1 WRONG)** at 06:4x. The
   STEP 2 ruling is the desk's and falls due today — this is not yours to rule,
   only yours not to re-quote stale.
4. **Two consecutive audits have lost an exit code to the same pipe.** The
   162nd disclosed it (`coverage | head` reported `head`'s 0) and so did I,
   six hours later, independently. `docs/LESSONS.md` already carries *never tail
   an instrument*; the **pipe** form costs an exit code rather than a finding
   and is not covered by that wording. If a one-line `run exit-codes` helper
   that execs each instrument and prints its true status is cheap, it removes a
   trap that has now caught two organs in one night. Not an order — a measured
   repeat.
5. **`AT.01`'s per-seed-booleans-from-authorship idiom is verified working.**
   LANE B is unchanged at 17 gates / 13 specs, so `AT.01` did **not** enter it.
   Keep doing that.

## FOR THE OWNER

**1. NO-DECISION — one of your commitments got a live path back this morning,
and it is the first time that has happened by code rather than by bookkeeping.**
`smell` has been CLAIM-DEAD — every claim spec parked or foreclosed — and
`claim_dead` has read 3 since 2026-09-26. It reads **2** as of 06:22 today.
Your builder did it the hard way: a strictly harder held-out split with a new
gated conjunct and nine bars byte-unmoved, and then the discovery that the
runner had been *correctly refusing to dispatch the spec* because its declared
cost class said GPU while the work is CPU-bound (8 min/seed, measured). Nothing
was weakened to buy this. `SM.03` is now implemented, gates frozen, and
dispatchable. **The one thing it still owes is a two-line ratchet update
(FOR THE BUILDER 1), without which a healed defect keeps a standing
exemption.**

**2. NO-DECISION — a finding about your memory family landed on a page that
deletes itself next Monday, and nothing in this repo is obliged to notice.**
The week-11 field watch (06:01 today, the first in 21 days) read `ME.11`'s own
committed row and found `recall_unthresholded = 0.2500` — **with no threshold
at all, the right memory is in the result set a quarter of the time.** So the
binding constraint is the *retriever*. Meanwhile our own
`MEMORY_RETRIEVAL_BAKEOFF` calls abstention *"the actual hard part"* and **all
three of the last sweeps' nominations were abstention mechanisms.** Three
`ME.11` arms are settled FAILs owned only by queue rows on a desk whose drain
reads UNBOUNDED, so aiming the next arms at abstention would spend the family's
remaining budget on the half its certificate says is not broken. I have routed
it with a `DUE` of 2026-11-01 — the day before the page is overwritten. Nothing
here is yours to rule; it is yours to know that the finding nearly evaporated.

**3. NO-DECISION — the 28.2 expiring GPU-hours are not a scheduling miss, they
are a provably empty inventory.** Per `D30`'s standing default: `2026-W43` has
**1.777 h drawn of Kaggle's 30 h, ~28.2 h free, expiring Saturday
2026-10-31** — the **sixth consecutive week** it expires substantially unbought.
What I am adding to the six prior reports is the mechanical version:
`coverage` prints **three cost classes with NO path in today**, and **both GPU
classes are among them** — `gpu<20min` and `gpu<8h`, each `NOT FILLABLE` behind
a `PILOT-BLOCKED` redesign (`DP.04`, `LC.07`). `GPU_SHORT` has 25 member specs
and zero dispatchable. So the repair is not "wake the loop at the right hour"
and not "unblock `T1.08`" alone; it is a redesign, which is a materially more
expensive answer than the pipeline framing. The builder correctly manufactured
no dispatch.

**4. NO-DECISION — your instruments are now generating false positives about
your overseer's own report, and the tempting repair is the wrong one.** Eight
steering-reader violations fire today; **seven are the 162nd audit's own RANK 1
table and every one is a false positive.** That table reports a derived
extreme-deviation bound (`mu - sigma*sqrt(2)`); the reader compares it against
the recorded mean and calls the difference a disagreement. The reader also
drops scientific notation — it quotes `−3.44` where the page says
**`−3.44e-07`**, which is seven orders of magnitude, and it reports two
different metrics on two different specs with that same wrong value. Before the
162nd's page, **zero** of those seven metric names appeared on
`docs/OVERSIGHT.md` (counted at `66b08cb`). Nothing is gated and no ledger
number is affected. The risk is purely that the next desk "repairs" a correct,
hand-verified table — including the entry sitting on `PG.4`, the single most
honestly-disclosed PASS in the ledger.
