> **INCOMPLETE RUN — THIS IS A DRAFT, NOT A FINDING.**
> The overseer run that wrote this file exited rc=124 and did not
> complete its own checklist (2026-10-03T07:02:11+00:00). Everything below was
> written before the run stopped: any verdict, any section claiming
> "no findings", and any instrument table in it are UNVERIFIED.
> Sealed automatically by scripts/lib_seal.sh; the exit code is in
> the log, and this banner is what joins the two.
> Files this run also left dirty, committed unbannered by the seal: docs/LESSONS.md.

# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Findings are ranked by how much damage they do to the
> trustworthiness of the ledger, not by how interesting they are.

**2026-10-03 06:38–07:1x UTC — the 137th audit.** Twenty-four hours after the
136th (the 06:37 overseer slots at 12:37, 18:37 and 00:37 did not produce a
report; `usage_gate` applies to `overseer.sh:47` too). My window is the builder's
slots `07:07` on 2026-10-02 through `06:07` today: **twenty-four slots, ZERO
`rc=0`, twenty-four PACE-SKIPs.** Demonstrated moved **107 → 107**.

Instrument exit codes, re-derived this sitting from source, not inherited:
`coverage` **0**, `decisions --check` **1**, `champions --check` **0**,
`run status` **2**, `run review-queue` **2**, `review_liveness` **1 (FAILED —
`docs/PROGRESS.md` itself last moved 72 h ago against a 25 h cadence)**.

> **THE REVIEW IS MID-SITTING WHILE I WRITE, and it is working on the same facts
> this report is about.** `scripts/review.sh` (pid 3600809) started at
> **06:37:04** beside my own `overseer.sh` — the 06:37 collision `LESSONS.md`
> records. `HEAD` moved **`ec041f0 → d028392 → aaa4580 → 8a4b55e → 070df72 →
> 6f8d7f2`** during this audit (acts 3 through 5b, 06:44–06:5x). **I staged
> nothing of theirs and I `git add` by name.** Two of its acts land directly on
> my findings and are credited where they fall: act 4 took `OVERDUE` **8 → 0**
> (RANK 8), and act 5 **minted `D40`** — a properly-shaped, armed owner question
> for `D30`'s option (i), with the same measurements I took independently and the
> same diagnosis of why a recommendation inside a RESOLVED entry can never be
> answered (RANK 3, FOR THE OWNER 1). Every queue number below is stamped with
> the `HEAD` it was read at and was re-read immediately before commit.

Ratchet delta, quoted from `run status`'s own SLOT LINE. **It moved twice during
this sitting because the Review was disposing rows under me, so both readings are
stamped.** At `ec041f0` (06:40): 5 MOVED — `dark_slots` 0 → 52;
`review_queue_net_arrivals` 32 → 26; `review_queue_piled_on` 4 → 9;
`review_queue_violation_forms` {HOLD-ON-A-RESOLVED-BLOCKER 8, OVERDUE 6} →
{HOLD-ON-A-RESOLVED-BLOCKER 7, OVERDUE 8}; `review_queue_violations` 14 → 15. At
`070df72` (07:0x): `review_queue_violations` **14 → 8**,
`review_queue_net_arrivals` 32 → 24, forms → {HOLD-ON-A-RESOLVED-BLOCKER 7,
**MALFORMED 1**}. **No counter refused to compute. Floors unchanged at both
readings: 4 ABOVE (`dark_slots`, `decisions_default_action_expired`,
`pass_on_dead_dependency`, `unreachable`), 0 BELOW, 0 UNVERIFIED.**

> **One transient I observed and am deliberately not calling a finding.** That
> `MALFORMED 1` was present in the ratchet read at `070df72` and **absent** from
> `run review-queue` at `6f8d7f2` six minutes later, which reports `7
> VIOLATION(S) — HOLD-ON-A-RESOLVED-BLOCKER 7` and no `MALFORMED`. `MALFORMED` is
> a real member of `review_queue.VIOLATIONS` (a receipt without a commit, a
> `WAITS-ON:` naming a row that does not exist, an empty `ORDERED:`). **I did not
> isolate its cause and I will not attribute a transient to a desk that was
> mid-edit** — the intervening commit's own subject is a self-caught slip
> (*"D40's own heading made it invisible to the owner's page"*). Recorded so that
> a future reader comparing the two numbers does not think one of them was
> suppressed.

Ledger: **PASS 107 / FAIL 33 / VOID 16 / BLOCKED 1** over 157 rows with a verdict
against **255 specs** → **42.0 %**, unchanged in 48 h. Rework: **127 of 157 rows
are attempt > 1 (80.9 %)**.

**What I did NOT do, named rather than omitted.** I re-ran no spec, dispatched
nothing, and every number here is read from the committed ledger, from git, from
`/data/jack-logs/`, or produced by a read-only instrument. **I armed no decision:**
`decisions_undeclared` reads **0** — I re-derived that from the tool's own
classes rather than copying the 134th–136th audits' identical refusals, and
manufacturing an entry to meet the per-audit quota is the disease the quota
exists to prevent. **I fired no default:** `D37` is the only armed entry and its
`decide_by` is **2026-10-04** — tomorrow. The audit that sits after midnight
tonight is the one that must fire it. I appended one short EVIDENCE ADDENDUM to
the already-open `D30` correcting a falsified forecast, and opened no entry.

---

## VERDICT: DRIFTING — and **the most damaging thing here is not new, which is the finding.** `docs/PROGRESS.md`'s `FOR THE OWNER` section still reads *"Dark slots **0** — fourteen consecutive `rc=0` builder slots, the blackout stayed closed. The builder is not the constraint."* The live reading is **52** and the builder's last `rc=0` was **52.5 hours ago**. **The 131st audit measured this exact false green four days ago** — its own commit subject is *"every current-state page says '0 dark slots'"* — and it is unrepaired because the organ that executes the repair is the organ that is dark. The genuinely new findings this sitting are smaller and both are about instruments mis-reporting their own subject: **RANK 2**, this audit's append to the owner's register was committed by the Review under the Review's message, because `git add` by name protects nothing on a file both organs may write; and **RANK 3**, the EXEC-CLIFF reader guards the one prompt a repair made immune and is blind to the three that are not.

Why not `INTEGRITY RISK`: the ledger's mechanical integrity held on every check I
ran (§1 — 107/107 PASS implementations resolve on disk, 107/107 recorded commits
resolve under `git cat-file -e`, 2 PASS rows with no control and both carrying an
explicit `NoControlByDecision` object), and **no threshold moved in the loosening
direction in seven days** (§2, by my own extraction over 6161 diff lines, not
inherited). Nothing below touches a capability claim.

Why not `ON TRACK`: **twenty-four of twenty-four slots in my window produced
nothing**, the third consecutive week of free GPU-hours dies tonight unbought,
and the pages that exist to notice are reporting the opposite of the live state
— one too green (RANK 1) and one too red (RANK 3).

---

## RANK 1 — The owner-facing current-state page affirmatively reports **zero dark slots and a closed blackout** while the builder has been dead 52 hours. **Third audit to find it; fifth day it has stood.** `D30`'s armed default made that page the sole delivery vehicle for this exact report, and six of the last eight Review sittings died before reaching it.

**Credit before anything else, because novelty is not what makes this rank
first.** The **131st audit** (`5a69201`, 2026-09-30 06:49) measured this and put
it in its own commit subject: *"the builder has been DARK for 23 slots on a meter
74 % of which this project did not spend, and every current-state page says '0
dark slots'"*. It also named the cause correctly — *"the reporting channel that an
armed default installed for exactly this event did not report, because the organ
that carries it died `rc=124` before reaching its own last line."* The 136th
audit carried it again. **I am the third, four days on, and the only thing I can
add is arithmetic: the streak has gone 23 → 28 → 52 and the sentence has not
moved.** It ranks first not because it is new but because it is the one wrong
reading that reaches the owner, and because the reason it survives — the builder
executes the repair, and the builder is dark — is a genuine deadlock rather than
neglect by anyone.

**The two readings, side by side.**

| | `docs/PROGRESS.md` FOR THE OWNER 2 | live, this sitting |
|---|---|---|
| dark slots | **0** | **52** (`run status` ratchet; loop's own 06:07 line: *"52 consecutive dark slot(s)"*) |
| builder state | *"fourteen consecutive `rc=0` … the blackout stayed closed"* | last `rc=0` **2026-10-01T02:17:07**, **52.5 h** ago |
| *"the builder is not the constraint"* | asserted | 24/24 slots in my window PACE-SKIPped |
| W39 Kaggle | *"30.0 free GPU-hours, 0.0 charged, expiring Saturday 2026-10-03"* | **1.07 h of 30 drawn**; the expiry date is **today** |

**The page's provenance, measured not guessed.** `docs/PROGRESS.md` last moved at
commit `3fcad58`, **2026-09-30 06:37:05** — and that commit is the STALE *stamp*,
not a rewrite. Its content is the **2026-09-28** DAILY sitting. `review_liveness`
exits **1** on the age branch: *"docs/PROGRESS.md itself last moved 72h ago
against a 25h cadence."*

**Why `D30` makes this the most damaging thing here and not merely untidy.**
`D30` was RESOLVED BY ARMED DEFAULT on 2026-09-19 with option **(v) REPORT THE
STREAK, GATE NOTHING, RELAX NOTHING** — and the resolution names its one
mechanism in its own words: *"A builder dark streak past 2× cadence is now a
standing FOR THE OWNER finding on the Review's page, printed beside the week's
GPU-expiry forecast."* The project chose exactly one remedy for a builder
blackout: a sentence on `docs/PROGRESS.md`. The streak is **52** against a 2×
cadence of **2**. The remedy is not late; it is being delivered by a page that
says the opposite.

**Why the page never lands, and it is a design choice that cured a different
disease.** `docs/PROGRESS_LOG.md` carries **9** INCOMPLETE rows, **6 of them in
the last 8 days** — 09-25, 09-26, 09-29, 09-30, 10-01, 10-02 — each written by
`review.sh:181`'s 76th-audit-B4 fallback because the DAILY run exited `rc=124` at
its 20-minute `timeout`. Those sittings were **not idle**: 10-01 made 6 acts
(`1bfed81`…`a65fdd7`), 10-02 made 9 (`6efa5f9`…`1710f69`). **The page is written
LAST, on purpose** — the 09-28 page says so in its own header, *"the page, written
last as the receipt for seven commits that already exist… Nothing was held dirty
while the page was drafted — the 74th audit's scar."* That ordering bought
protection against a dirty tree. The bill it pays is that **the organ which runs
out of clock loses the page every single time and never loses the acts.** And the
dirty-tree reason has since expired: `docs/PROGRESS.md` is a `PROSE_DOCS` member
and exempt from the per-spec staleness bill, so writing it mid-run costs nothing
it used to cost.

**The banner is the only thing standing between the owner and a false green, and
it is now wrong in the other direction.** The stamp's own sentence reads *"It
disappears the next time the review completes a run and rewrites this file."* The
Review has completed two sittings and **15 commits** since that stamp.
`/data/jack-logs/review.log` records the death path's choice verbatim:
`2026-10-02T06:57:11 docs/PROGRESS.md already carries a stale banner — leaving
it`. So the banner's timestamp is frozen at **2026-09-30T06:37:05** and a reader
cannot tell 24 hours of staleness from 120. A reader who checks whether the
Review is alive sees 15 commits and concludes the *banner* is stale rather than
the page — and the 09-28 page records that this already happened once: *"The
builder believed the stale banner at 05:07."*

Credit where it is due, because the 09-28 page's own two builder orders **both
shipped**: `DELIVERED — AWAITING STAMP` now prints 10 live rows off declared
`BUILDER-TRACE:` commits and buys no exemption, and
`HOLD-ON-A-RESOLVED-BLOCKER` now reads *"the window was abandoned, not opened"*
for a `DECLINED` blocker. The desk-to-builder channel works. It is the
desk-to-owner channel that is broken.

## RANK 2 — **This audit's own append to the owner's register was committed by the Review, under the Review's commit message, while I was still writing.** `git add` by name is the project's standing protection against cross-organ races and it gives **zero** protection on a file both organs are entitled to write.

**The measurement, and it is about this sitting.** At 07:0x I appended an EVIDENCE
ADDENDUM to `D30` to `docs/DECISIONS_NEEDED.md` and had not yet committed it.
`git log -S` locates that text in **`6f8d7f2`, 2026-10-03 06:53:35, *"Review DAILY
10-03 act 5b/N: D40's own heading made it invisible to the owner's page"*** — a
Review commit whose message says nothing about it. `git status` then showed
`docs/DECISIONS_NEEDED.md` clean, which is how I found it: **the file I was
editing went clean without me committing anything.**

**Why the standing remedy does not reach this.** `LESSONS.md`, the 06:37
collision note, and my own practice all say the same thing — *stage by name,
never `-A`*. That rule works when two organs write *different* files. It is
**structurally powerless here**, because `docs/DECISIONS_NEEDED.md` is a file
**both** organs are authorised to append to: the overseer brief grants *"append
to `docs/DECISIONS_NEEDED.md`"*, and the Review mints entries there (`D39`
yesterday, `D40` today). Whichever organ commits first takes the other's
in-progress append with it, **by name, correctly, and invisibly.**

**Why this is a record defect and not untidiness.** The register is the owner's
page, and provenance on it is load-bearing: which organ opened an entry, when,
and in which commit is what decides who may rule, whether a default may fire,
and what a phrase like *"state-corrected today in `1197928`"* means. Three
entries on that page already cite their own authoring commit. An overseer
addendum that git attributes to a Review act — in a commit about a markdown
heading — makes the page's own provenance unreliable at exactly the point it is
used. **Nothing was lost and no text was altered; what was corrupted is who did
it and why.**

**Scope, measured rather than assumed.** This is distinct from the already-routed
`cross-organ-doc-race-voids-certificates` (ROUTED 2026-09-03, DISPOSITIONED
09-06, DUE today): that row is about uncommitted doc writes making a concurrent
runner stamp `+dirty` and VOID four certificates, and its repair — the
`PROSE_DOCS` exemption — shipped and works. **It does not touch attribution, and
no row or lesson covers this half.** Both halves have the same root (two desks
writing shared files on a 06:37 collision) and different victims: one cost four
PASS certificates, this one costs the owner's register its authorship.

**What I did about it, stated plainly:** nothing, deliberately. Rewriting history
to re-attribute the hunk is not among my permissions and would be a worse act
than the one it corrects. The addendum's own text names itself *"(overseer, 137th
audit)"*, so the page is self-describing even where git is not. Repair routed as
FOR THE BUILDER 5.

## RANK 3 — The EXEC-CLIFF reader watches the **one** prompt that is structurally immune to the cliff and is blind to the **three** that are not; and its own known-answer self-test is what keeps the obsolete mechanism in place.

`run status` printed this half an hour ago:

```
STEERING-PAGE SIZE — scripts/ladder_prompt.md is 117186 bytes; 13886 below the
  131072 EXEC CLIFF (MAX_ARG_STRLEN), 7814 below the 125000 self-imposed ceiling.
  Past the cliff the builder does not read a degraded prompt — it does not start,
  and the slot looks like an ordinary rc=126.
  growth +884 B/day over 28 commit(s); at that rate the cliff is 16 day(s) away,
  the ceiling 9.
```

**That sentence became false on 2026-09-25.** `D34`'s armed default (iii) landed
at `f06afd1`, 2026-09-25 01:10, and its own commit message states the result:
*"prompt now travels on stdin, not argv … so no execve ever sees the steering
page's text and MAX_ARG_STRLEN=131072 can never again kill a slot at launch."*
`scripts/ladder_loop.sh:292` is `printf '%s' "$PROMPT" | …` — `printf` is a bash
builtin (verified: `type printf` → *"printf is a shell builtin"*), and the
`claude -p` on line 297 is passed no prompt argument at all.

**Transport, measured per organ.** This is the finding, and it is an inversion,
not a staleness:

| organ | prompt file | bytes | transport | argv-bounded | has a cliff reader |
|---|---|---|---|---|---|
| builder | `scripts/ladder_prompt.md` | **117186** | **stdin** (`ladder_loop.sh:292`) | **no** | **YES — the only one** |
| overseer | `scripts/overseer_prompt.md` | 12817 | `claude -p "$(cat …)"` (`overseer.sh:171`) | **yes** | no |
| Review | `scripts/review_prompt.md` | 11837 | `claude -p "$(printf…; cat …)"` (`review.sh:115/122/131`) | **yes** | no |
| field watch | `scripts/field_watch_prompt.md` | 3631 | `claude -p "$(cat …)"` (`field_watch.sh:45/51`) | **yes** | no |

`experiments/steering.py:100` pins `LAUNCH_PAGE = "scripts/ladder_prompt.md"`
and nothing else. So the reader is 89 % of the way to a limit that **cannot
apply**, and the three files that **are** passed as a single argv have no reader
of any kind.

**Said plainly so it is not oversold: there is no imminent outage.** The exposed
pages sit at ~10 % of the cliff (`overseer_prompt.md` has not changed a byte
since `dd3d907`; `review_prompt.md` grew 9868 → 11837 B over 4 commits since
09-01 — decades of headroom). **That is precisely why this is cheap to fix now
and expensive to fix under pressure**, which is the condition the 09-20 outage
was found in.

**The measured cost today, in the currency the project says is scarce.**

- A dated false alarm is scheduled: at +884 B/day the reader crosses the 125000
  ceiling in **~9 days** and the "cliff" in **~16**, at which point
  `render_size()` escalates to *"!! OVER THE EXEC CLIFF — the builder cannot
  launch. **This is not a warning, it is the outage.**"* — the strongest sentence
  in the module — while the builder launches normally. The last time that
  sentence was true, the Review spent a sitting trimming the page 140331 → 85548
  bytes. The project's own `PROGRESS.md` names this desk's design capacity, not
  the builder's throughput, as the measured bottleneck.
- A live routed queue row already prices its urgency on the dead constraint:
  `docs/REVIEW_QUEUE.md:1711` reads *"the page is 96212 B against a 131072 B EXEC
  CLIFF that a phantom order spends real headroom on."* That headroom is not a
  constraint on that page.
- `scripts/ladder_prompt.md:535–553` states the dead mechanism in the **present
  tense** — *"`ladder_loop.sh` passes this file's whole text as a single argv to
  `claude -p "$PROMPT"`"* — and derives THE STANDING SIZE RULE (*"must stay under
  125000 bytes"*) from it. That rule binds the Review hardest, by its own text.

**Why it could not self-correct, and this is the generalisable half.**
`steering.py:_check_size()` is a known-answer fixture that **raises unless**
140331 bytes renders as *"it is the outage"*. The fixture exists to stop the
reader going soft — and it is now the thing that makes telling the truth fail the
self-test. `steering.py` was edited twice after the repair (`5938f9c` 09-27,
`45924a4` 09-30), by two audits working on neighbouring readers **inside this same
file**, and the stale mechanism survived both. Lesson appended to
`docs/LESSONS.md`.

**The honest counterweight, so the next reader does not simply delete the
ceiling.** A 117 KB board is read by the builder on every hourly slot, and tokens
against `week:all models` are the exact resource RANK 3 is about. A size ceiling
may well be worth keeping. What is refuted is its **stated cause**, its
**escalation wording**, and its **coverage** — not necessarily the number.

## RANK 4 — The blackout, re-derived from the gate's own arithmetic: **52 dark slots, 52.5 hours, first legal slot ≈ Tuesday 2026-10-06 15:00 UTC** — and the 136th audit's *"the whole project stops in about six hours"* did **not** happen.

**The gate is a pure function of the clock.** `scripts/lib_usage.sh:85`:
`allow = PACE_FLOOR + ((PACE_CAP − PACE_FLOOR) × elapsed + 99)/100` with
`PACE_FLOOR 25`, `PACE_CAP 90` — 0.3869 pts/h, zero variance. Live at 06:07:
`week:all models` **82 %**, week-elapsed **39 %**, line **51 %**. Release needs
`allow ≥ 82`, i.e. elapsed **≥ 87 %** → **2026-10-06 ≈ 14:50 UTC**, about **81
hours** away, for a total blackout of **≈132 h / 5.5 days** — the longest on
record. The 136th audit derived Tuesday ≈07:40 from a meter at 79 %; **the
release slipped ~7 hours in 24 hours** because the meter rose 3 points, and it
will slip further if the meter does.

**Attribution, from the loop's own line, re-read live:** of this week's **81**
shared points — builder **16 (19 %)**, desks **5 (6 %)**, both 0, **NOT THIS
PROJECT 60 (74 %)**. This project is being paced out of its own week by another
tenant's spend. That is `D30` option (i)'s subject and it remains the owner's.

**The correction this organ owes.** The 136th audit told the owner, as its FOR
THE OWNER item 1 headline, *"THE WHOLE PROJECT STOPS IN ABOUT SIX HOURS UNLESS
YOU ACT"*, deriving the 90 % stop at *"~13:00 UTC today"* from **42 points in
23.7 hours = 1.77 pts/h**. **It did not fire.** The meter read 79 % at
2026-10-02T06:07, 82 % at 10-02T15:07, and **82 % at every reading since — flat
for 15 hours**. The 42-point figure was a **burst** (31 % → 60 % between
10-01T07 and 10-01T11, +23 points in 4 hours) extrapolated as a rate; the
realised rate after that forecast was **0.125 pts/h and then 0.0**, overstating
by ~14×. A six-hour alarm to the owner that does not fire is not free: it teaches
the reader to discount the next one, and the next one is the real stop. Recorded
as a short EVIDENCE ADDENDUM on `D30` so the 1.77 pts/h figure is not inherited
as measured. **The underlying ask is unchanged and still correct** — see FOR THE
OWNER 1.

**A disagreement with `D40`, minted at 06:52 while I was deriving this, and it is
one number in an otherwise correct entry.** `D40` states *"The next expiry is
Saturday 2026-10-10, and on the current meter the builder's first legal slot is
**not before it**."* **I get a different answer and I show the work, because this
sentence is what makes the 10-10 GPU-hours look lost too.** `allow` is integer
bash arithmetic: `allow = 25 + (65×elapsed + 99)/100`. At `elapsed 39` that is
`25 + 2634/100 = 51`, which matches the log exactly. Release needs `allow ≥ 82`
→ `65×elapsed ≥ 5601` → **`elapsed ≥ 87`**. The account's week begins Wednesday
≈11:59 UTC (the 136th audit's measurement, which I reused rather than re-derived),
so `elapsed 87 %` is `0.87 × 168 h = 146 h` after 2026-09-30 11:59 →
**2026-10-06 ≈14:10 UTC**. And a hard ceiling independent of the meter: at the
week reset (`elapsed → 0`) `allow` returns to `PACE_FLOOR 25` and the weekly
percentage resets with it, so **the blackout cannot outlast ≈2026-10-07 12:40
UTC under any meter value.** There is no reading of `pace_gate` that puts the
first legal slot on or after 2026-10-10. If the meter reaches 90 % first,
`usage_gate` — not `pace_gate` — is what holds the builder, and that is the owner
item, not a pace question. **`D40`'s substance, its recommendation and its armed
default are unaffected by this**; it is a sentence to correct, not an entry to
reopen, and it is the Review's to correct.

**What I will and will not forecast.** The week ends (elapsed 100 %) at
≈**2026-10-07 12:40 UTC**. For the 90 % stop to arrive before the week resets the
meter needs only **+0.08 pts/h** from here. The measured external rate over the
last five days spans **0.0 to 1.4 pts/h**. The honest answer is the range, not a
time: the stop is *possible at any moment and not predictable from this record*.
`usage_gate` (`lib_usage.sh:121+`) refuses `ladder_loop.sh`, `overseer.sh`,
`review.sh` and `field_watch.sh` — every organ but the regate sweep — with *"all
agents paused until the owner resumes"*, and resuming requires an owner-written
`.usage-resumed`. There is none on disk.

**Compute honesty (§5), and it is the third week in a row.** `2026-W39` has
**1.07 h charged of 30 free Kaggle GPU-hours**; ~**28.9 h expire at tonight's
Sunday reset**. Preceding weeks, summed from `gpu_budget.json`'s own
`charged_jobs`: **W37 5.25 h, W38 0.92 h**. Three weeks, **~83 free GPU-hours
lost.** `gpu_hours_no_verdict` reads **49.49 h TOTAL**, unchanged — `D1.0` alone
is **33.78 h over 2 attempts for 0 verdicts**, and `gpu_unattributed_jobs` is 21,
AT floor. **No dispatch has been manufactured to spend tonight's hours and none
should be:** both live routes run through `T1.08` (FAIL), whose pipeline-repair
design is the Review's own queue row and went **OVERDUE yesterday**.

## RANK 5 — §1 and §2: **no findings, and that is a real result.**

**§1 Integrity of the ledger — re-derived this sitting, not inherited.** Over all
**107 PASS** rows: **107/107** resolve a declared implementation on disk;
**107/107** recorded `commit` values resolve under `git cat-file -e <c>^{commit}`;
**0** PASS rows carry an undeclared control except **two**, `T0.01` and `T0.10`,
and both hold an explicit `NoControlByDecision(…)` object quoting the 52nd
audit's B5 reasoning (*"an import either raises or it does not"* / *"a sabotaged
upload fails on the service's side, which is the falsifier itself"*). Those are
declared refusals, not holes — and the only two `control=` lines removed in seven
days are these two being reformatted into those objects (`eba3e58`). **No spec
lost a control.**

Caveats the ledger raises about itself, carried forward rather than suppressed:
2 DIRTY STAMPS (`T6.03`, `PL.02`), 15 STALE CLAIMS, 1 STALE PRE-`impl_sha` claim
(`T2.02`), **5 UNBACKED CERTIFICATES** (`LF.02`, `T0.18`, `T0.19`, `T2.03`,
`T2.14` — `pass_on_dead_dependency` = 5, ABOVE its floor of 3 with the cause
written: `T0.13`'s honest re-buy to FAIL took `T0.18`/`T0.19` with it, and the
repair is the per-key ruling owed under `t013-latently-red-28-disarmed-keys`,
DUE 2026-10-05), and 6 PASS rows predating `spec_sha`.

**§2 Thresholds and controls over 7 days — zero loosening, verified four ways by
my own extraction over `git log -p --since="7 days ago"` (6161 lines).** Seven
named numeric constants moved:

- `dp_04_slow_path_verbal.py` — `NEED_MIN_GAIN None → 35.0`, `SCRAM_ABS_NEED
  None → 28.0`, `MUTE_FLOOR_MIN_NEED None → 35.0`, `HEADROOM_MIN_NEED None →
  56.0` (`5b0d4c0`). **First registration**, derived from a disclosed precheck
  record (`ceil(34.0206765975521)`) — a bar arriving, not a bar moving.
- `t0_31_review_queue_cannot_go_quiet.py` — `N_PROPERTIES 22 → 24` (`5651fc1`).
  **Strengthening**: two more properties asserted by the spec that gates the
  review queue's own counting.
- `sm03_readout_sweep_probe.py` — `SHIPPED_POOL 1 → 1`. **Unchanged value**, line
  moved only.
- `t2_11_skills_distinguishable.py` — `CLF_EPOCHS 300 → 900` (`5eba43e`). **Not a
  loosening, and I re-derived the mechanism against the ledger rather than
  accepting the 136th audit's call:** the epoch budget is shared identically by
  the real and the shuffled fits, so raising it strengthens the *control*.
  `shuffle_clf_fit` went **0.5859** (attempt 1, VOID) → **0.9219** (attempt 2)
  against an unmoved `SHUFFLE_FIT_FLOOR 0.60`. `T2.11` stands an honest **FAIL**
  at `margin_vs_shuffled −0.086` (claim 0.8672 vs control 0.9766).

And the four checks a constant diff cannot make: **0 seed counts reduced** (four
added `seeds=` lines, all new registrations or fixtures: `LG.14`'s `seeds=3`, an
`ADM-BATTERY` fixture, `seeds=[0,1,2]`, a `seeds=[3,4]` test double);
**0 `assert`/`raise` lines removed** from any test; no `or` added inside a
`_check` body; **no spec lost a control** (above).

## RANK 6 — §3 Drift: there was no work to drift. The converse question is the one with an answer, and it is unchanged and bad.

**What the builder worked on in the last day: nothing.** 52 dark slots. Its last
unit was `87bc128`, 2026-10-01 02:13 — registering `LG.14`, the structured-decode
mouth spec ordered by the `lg12-abstention-knob` ruling, **seven days ahead of
its DUE**. It traces to `GOAL.md:43` (*"and VOICE — he must be able to make
sound, not only receive it"*) and the mouth/parent-LLM paragraphs. No drift. The
builder's own slot summary recorded the honest half: *"Creature gate: NONE —
seventeenth consecutive, recorded as the violation it is."*

**Which parts of `GOAL.md` have no passing spec.** `coverage` exits **0** and
`commitments_uncovered` is **0**, AT floor — every constitutional commitment has
a declared spec. But:

- **3 CLAIM-DEAD** (`claim_dead` = 3, unchanged since 09-26): **smell**,
  **shelter/building**, **thermal (kills)** — every claim spec parked or
  foreclosed. `SM.02` PARKED / `SM.03` FORECLOSED (PILOT-BLOCKED); `SH.01`
  PARKED / `SH.02` FORECLOSED. These are three of the owner's own named
  commitments, including *"too cold kills him"* and his own image of success.
- **14 commitments with live claim specs and nothing passing**: touch/contact,
  tool use, told world, heavy, far, tiring, worth-it, balance, proprioception,
  plasticity, sleep, hunger/thirst, death & retry, fast/slow.
- **one brain / unison: 1 passing of 28 specs.** **curiosity: 2 of 12.**
  These are named in the audit brief as the claims most likely to be quietly
  neglected, and they are.
- `goal_unrunnable` = **7**, unchanged since 2026-09-05.

**And the shape of the activity, from `run status`'s own SETTLE EVENTS:** over 7
days, **171 recorded runs → 4 first-ever verdicts**, 159 re-buys, 8 status
changes; **147 of 171 (86 %) instrument-coupled** — the project's own tool edits
are what staled the certificate being re-bought. Of the 137 PASS events, **2**
were first-ever.

## RANK 7 — §7 Bakeoff hygiene: the project's most load-bearing architectural seat is held **BY VERDICT off a VOID**, and that is this section's disease by name.

`champions --check` exits **0** with **10 violations**, all at or inside declared
floors (`champions_unwinnable` = 4 AT floor; `champions_trigger_debt` = 3). No
seat lost a door in this window. The two that matter:

- **Learning core — `VERDICT-IS-A-VOID` + `TRIGGER-UNREACHABLE`.** Held BY
  VERDICT — the strongest marking in the file — off `LC.03`, which is **VOID**.
  `SYSTEM.md`'s own rule is *"fix the arm, do not decide"*; a VOID decided
  nothing. And every pre-registered re-open trigger is a closed door: `LC.07`
  PILOT-BLOCKED, `LC.03` VOID-FORECLOSED, `UB.10` VOID. **A VOID treated as a
  verdict, holding the seat the whole ladder sits on.** The repair is a completed
  re-run, an honest re-marking, or the routed redesign — never trusting the
  marking, and never deleting a trigger.
- **World — `VERDICT-UNDECLARED` + `TRIGGER-UNDECLARED`.** Held BY VERDICT and
  naming neither the ledger row that bought the marking nor the specs that could
  fire a rematch. Arena `W.1` FAIL, `W.2` FAIL, `W.3`–`W.8` NOT_RUN.

Also standing: 2 `NO-ARENA` seats (ASR, Speaker ID — nothing that could be run
would unseat either), 2 `UNCONTESTED` (Vision encoder, the PLASTIC-ONLY decree
seat, both turning on `PL.02`, which is VOID), and `Fast/slow coupling`
`ARENA-UNREACHABLE` (its only arena spec `DP.02` is welded behind `LC.03`).
`DECISIONS_RESOLVED.md`'s admissibility defect was repaired on 10-01 (`7d79d9d`):
all five `run_bakeoff` callers supplied no predicate, so every `admitted | yes`
on that page was the default, and the column is now three-valued with the
records annotated. No winner chosen inside a noise margin this window.

## RANK 8 — §6 Stuck decisions: nothing is escalated that a measurement could settle, nothing is armable, and the one broken ratchet class cannot clear at this desk.

`decisions --check` exits **1**.

- **`MEANS-ESCALATED`: none.** No fork a measurement could settle is sitting on
  the owner's desk. This is the `D1` disease and it is absent.
- **`UNDECLARED`: none.** `decisions_undeclared` = 0, AT floor. There is nothing
  to arm, so I armed nothing — re-derived from the tool's class list, not copied.
- **`OVERDUE — DEFAULT IS DUE TO FIRE`: none today.** `D37` is the only armed
  entry; `decide_by 2026-10-04`, default (iii) HOLD `D29` AS IT STANDS, monotone
  and explicitly not the recommendation. **Flagged forward: the next sitting is
  the one that must fire it.**
- **`RATCHET BROKEN: 1 DEFAULT-ACTION-EXPIRED`, baseline 0** — `D33`, whose
  default names 2026-09-23 while `decide_by` is 2026-09-23. The 10-02 Review went
  further and ruled the default **MOOT, not merely expired** (`37e8acd`): its
  object went terminal by that desk's own hand when `w1-world-edit-window` was
  stamped `DECLINED`, so the entry **cannot self-resolve at all**. The class
  therefore cannot be cleared by any desk; it needs an owner ruling or a formal
  decline. `D33` is now **10 days stale** and the Review's published stop-rule
  fires **2026-10-09**.
- 4 `CONDUCT-DESK` entries listed so a stale conduct entry cannot self-approve:
  `D33` (stale 10 d), `D35` (stale 9 d), `D38` (due 10-04), `D39` (due 10-15).
- 1 owner-ask reached a desk and is correctly attributed (`PROGRESS #1` →
  `D22`/`D33`). `decisions_unrouted_owner_ask` and
  `decisions_vanished_owner_ask` both **0**, AT floor.

## RANK 9 — §4 The routed work: the desk cleared all 8 OVERDUE **during this audit**, so the 15 violations I opened on are 7 — and every one of the 7 is the same orphaning the project is deliberately refusing to launder. Drain is still UNBOUNDED.

**Two readings, both stamped, because the subject moved while I read it.** At
`ec041f0` (06:40): `run review-queue` exits **2** with **15 VIOLATIONS —
OVERDUE 8, HOLD-ON-A-RESOLVED-BLOCKER 7**; 53 OPEN / 3 HELD / 31 DISPOSITIONED /
33 ACTED / 2 DECLINED of 122 routed. At `6f8d7f2` (07:0x), after the Review's
acts 3–5b: exits **2** with **7 VIOLATIONS — HOLD-ON-A-RESOLVED-BLOCKER 7,
OVERDUE 0**; 52 OPEN / 3 HELD / 29 DISPOSITIONED / **37 ACTED** / 2 DECLINED of
**123 routed**; **84 live rows**; oldest live **40 d**. **`OVERDUE` 8 → 0 in four
acts is the desk doing exactly the right thing and it is credited here, not
buried:** `lt02` ACTED at `88762a2` with the `nan` it fails on split out rather
than buried, `t215` and the blackout measurement ACTED, and `T1.08`'s design
re-dated onto tomorrow's FULL **as that sitting's first act**.

**Throughput over 7 cycles: arrived 38 (5.43/cycle), disposed 12 (1.71/cycle),
designed 17 (2.43/cycle — still live, still ageing). Drain UNBOUNDED**, arrivals
exceeding disposals by 26 over the window. **9 rows share 2026-10-13** against a
measured one-cycle capacity of 6.

Reading the ratchet moves by cause, which the tool does for me: `OVERDUE` 6 → 8
is **CLOCK** — the calendar reached a date and no commit is to blame.
`HOLD-ON-A-RESOLVED-BLOCKER` 8 → 7 is an **ACT** — a violation cleared by a
commit. `review_queue_piled_on` 4 → 9 is the cost of dating 9 live rows onto days
that already carried their measured capacity. The 7 remaining
`HOLD-ON-A-RESOLVED-BLOCKER` rows are all behind `w1-world-edit-window`
(`DECLINED`), and the red is deliberately not cleared — re-pointing them at a
fresh blocker would launder the largest structural fact the project has.

No finding against this desk's disposals: the 10-01 and 10-02 sittings cleared
`OVERDUE` to 0 twice and it returned by the clock both times, which is arithmetic
and not neglect.

---

## §8 — THE HONEST SUMMARY. Are we closer to a curious humanoid that climbs the ladder than we were yesterday?

**No. We are not closer, and today is the first day in a while where that is not
even arguable, because nothing about Jack happened at all.** Demonstrated has read
**107 for 48 hours**. The builder's last action of any kind was 52.5 hours ago and
it was a *registration* — a spec written down, not a spec run. Over the last seven
days the ladder recorded **171 runs and 4 first-ever verdicts**, of which **2**
were PASSes, and **86 % of those runs were the project's own instruments re-buying
certificates its own tool edits had staled**. The organ that would move the
creature forward is being paced out of a quota pool in which **74 % of the spend
is another tenant's**, and ~28.9 free GPU-hours expire tonight for the third week
running.

**And the longer-run answer is worse than the day's, which is the part worth
saying plainly.** Three of the owner's own constitutional commitments are
**CLAIM-DEAD** — smell, shelter-building, and *"too cold kills him"* — every claim
spec behind them parked or foreclosed, with no successor that is not itself
foreclosed. Fourteen more have live claim specs and **nothing passing**: touch,
tool use, the told world, heavy/far/tiring/worth-it, proprioception, plasticity,
sleep, hunger and thirst, death-and-retry. **One brain / unison stands at 1
passing spec out of 28. Curiosity stands at 2 of 12.** Those are not the easy
wins being deferred; they are the thesis. The architectural seat the whole ladder
rests on is held **BY VERDICT off a VOID**, with every pre-registered re-open
trigger a closed door.

**What we ARE closer to is a rig that tells the truth about itself, and that is
not nothing.** No threshold moved in the loosening direction in seven days. No
PASS lost its control. 107 of 107 certificates still resolve their implementation
and their commit. Four Tier-0 PASSes were given up last week to buy honest FAILs.
The desk cleared eight broken promises during the two hours I was writing this,
and routed the pace question to the owner in the one shape that can actually be
answered. **But a measurement rig that is honest about producing nothing is still
producing nothing.** This week the project's three most consequential acts were
all *refusals* — a conjunct not shipped, an authorship declined, a GPU dispatch
not manufactured — and every one of them was right. A ladder made entirely of
correct refusals does not get climbed.

**The single sentence.** Three organs spent this week measuring, correctly and in
detail, a builder that was not running — and the one page that was supposed to
tell the owner so has said *"the blackout stayed closed"* for five days.

---

## FOR THE BUILDER

**0. THE 135th AND 136th AUDITS' ORDERS ARE ALL STILL OPEN AND NONE OF IT IS
YOUR FAULT.** Your last slot ended **52.5 hours** before this line and both
reports were written after it. **Read the 136th's FTB 1–4 as live and
unexecuted** — the usage-distance reading, `ledger_metrics()`'s control-column
fold, `lib_seal.sh`'s self-resetting clock, and the repeating `PACE-SKIP NOTICE`
— and its item 0, which carries the 135th's four forward. I am not re-ranking
them. **The 136th's item 1 is still the right thing to spend your first slot on**,
and I agree with it rather than displacing it: nothing in `experiments/` reads
the usage meter, so the distance to a stop that pauses every organ reaches no
exit code. Items 1–4 below are new and additive, and they are ordered to be
cheap.

**1. POINT THE CLIFF READER AT THE PAGES THAT CAN ACTUALLY DIE (RANK 2).**
`experiments/steering.py:100` pins `LAUNCH_PAGE = "scripts/ladder_prompt.md"` —
the one prompt that travels on **stdin** since `f06afd1` (`ladder_loop.sh:292`,
`printf` builtin, `D34` default (iii)) and therefore cannot hit
`MAX_ARG_STRLEN`. The three that **are** passed as a single argv have no reader:
`scripts/overseer_prompt.md` (12817 B, `overseer.sh:171`),
`scripts/review_prompt.md` (11837 B, `review.sh:115/122/131`),
`scripts/field_watch_prompt.md` (3631 B, `field_watch.sh:45/51`). Make the reader
a **set of pages, each declaring its transport**, and report the cliff only for
argv pages. **Constraints:** reporting-only and unfloored, as it is today — a
gate here could refuse a legitimate steering edit. Keep the size/growth reading
for the stdin page if you think the token cost justifies it, but report it under
its real reason, not under a kernel constant. **There is no imminent outage** —
the argv pages have ~90 % headroom — which is exactly why this is cheap now.

**2. CORRECT THE TWO PROSE STATEMENTS THAT ASSERT THE DEAD MECHANISM IN THE
PRESENT TENSE, AND MARK THEM SUPERSEDED RATHER THAN EDITING THEM SILENTLY.**
`steering.py:87–88` (*"passes this page's ENTIRE TEXT as a single argv"*) and
`scripts/ladder_prompt.md:549–553` (THE STANDING SIZE RULE, *"Past 131072 the
builder does not read a degraded prompt; it does not launch at all"*). **If the
125000-byte ceiling is kept — and there is a good reason to keep it, the
per-slot token cost of a 117 KB board against the very meter that is currently
dark — re-found it on that reason.** A rule whose stated justification has been
refuted gets deleted by the next reader who checks, and that reader would be
right to.

**3. RE-AIM THE OUTAGE FIXTURE; KEEP THE ARITHMETIC ONE.**
`steering.py:_check_size()` raises unless **140331 bytes renders as *"it is the
outage"***, so the reader cannot be told the truth without failing its own
known-answer test. That is why it survived two edits to the same file after the
repair. Keep the fixtures that assert headroom, the day counts, and
unknown-is-not-zero; move the outage fixture onto an **argv** page at a size over
the cliff. Say in the commit that the 2026-09-20 replay is being retired because
its mechanism was repaired, and where the replay now lives.

**4. GIVE THE STALE BANNER A LIVE AGE AND AN HONEST SENTENCE (RANK 1).** This is
the 136th's FTB 3 reinforced with two new halves, not a new order.
`scripts/lib_seal.sh` returns early when a banner already exists —
`/data/jack-logs/review.log` records the consequence verbatim:
`2026-10-02T06:57:11 docs/PROGRESS.md already carries a stale banner — leaving
it`. So the stamp is frozen at `2026-09-30T06:37:05` and a reader cannot tell 24 h
from 120 h. **(a)** Refresh the banner's numbers on every failed liveness check
even when a banner is present — the banner's job is to carry *how* stale, not
*whether*. **(b)** The sentence *"It disappears the next time the review completes
a run and rewrites this file"* implies the Review has not run; it has sat twice
and committed 15 times since the stamp. Derive and print **"the Review has sat N
times since this stamp without reaching its page"** from `PROGRESS_LOG.md`'s
INCOMPLETE rows, which already exist for exactly this purpose. Keep the `:246`
refusal to stamp a dirty file — that is how two organs share this page safely.

**5. THE PAGE SHOULD NOT BE WRITTEN LAST (RANK 1), and this is a
`scripts/review_prompt.md` change, which is yours to make under the 69th audit's
own B2 precedent — not the desks' and not mine (`D13`).** Six of the last eight
DAILY sittings died `rc=124` at the 20-minute `timeout` after 6–9 real acts, and
the page was the casualty every time because it is written last by design. The
reason that ordering was adopted — the 74th audit's scar, not holding work dirty
while drafting — **no longer applies**: `docs/PROGRESS.md` is a `PROSE_DOCS`
member and exempt from the per-spec staleness bill, so it can be written
mid-sitting at zero certificate cost. Propose the smallest version: have the
Review write its **`FOR THE OWNER` section and its trend row first**, from the
instrument readings it already takes at the top of a sitting, and let the acts
and the narrative fill in afterwards. **Do not remove the INCOMPLETE-row
fallback** — it is the only thing that currently makes these deaths visible, and
it is correct.

**6. GIVE THE SHARED-FILE COMMIT RACE A READER (RANK 2), because `git add` by
name cannot see it.** Measured on this sitting: `6f8d7f2` ("Review DAILY 10-03
act 5b/N") contains my uncommitted `D30` addendum to
`docs/DECISIONS_NEEDED.md`, because both organs are authorised to append to that
file and the Review committed first. Staging by name does not help — the name is
the same. **Smallest honest repair, and it is reporting-only:** a check that, for
each file writable by more than one organ (`docs/DECISIONS_NEEDED.md`,
`docs/LESSONS.md`, `docs/REVIEW_QUEUE.md`), compares the hunks a commit contains
against the committing organ's declared file set and prints **`FOREIGN-HUNK —
<file> contains changes this organ did not author`** when a commit carries
appended text whose own byline names another organ. Every entry on that register
already self-identifies (*"(2026-10-03, Review, DAILY)"*, *"(overseer, 137th
audit)"*), so the byline is a parseable declaration in the existing idiom and no
new convention is needed. **Constraints:** reporting-only, unfloored, and it must
not refuse a commit — a desk legitimately editing a shared file is normal and a
gate here would deadlock the 06:37 overlap. **Do not propose rewriting history**
to re-attribute anything; the repair is visibility, and a misattributed hunk that
is *visible* costs the register nothing.

---

## FOR THE OWNER

**1. The pace question reached you properly this morning as `D40` — that is the
Review's act and not mine, and I am not restating it. What is left over is a
DIFFERENT gate, and it is the one with no mechanism behind it at all.** Read
`D40` first: *pace the builder against this project's own attributed spend rather
than `week:all models`*, armed, `decide_by 2026-10-10`, default (v) = the status
quo. I reached its measurements independently this morning and they hold; its
recommendation is sound; its one wrong sentence is corrected in RANK 3 and is the
Review's to fix. **`D40` deliberately keeps the 90 % hard stop exactly where it
is — so the stop is the part nobody has asked you about.** The 90 % stop
(`lib_usage.sh:121`) refuses `ladder_loop.sh`, `overseer.sh`, `review.sh` and
`field_watch.sh` — **every organ except the regate sweep** — with *"all agents
paused until the owner resumes"*, and resuming requires a `.usage-resumed` file
written **by you**. There is none on disk. `week:all models` reads **82 %**, and
**74 % of this week's 81 shared points were not this project**. Nothing in the
repo can fire a default here, correctly: a default may not loosen a gate, which
is why `D30` option (i) did not fire on 2026-09-19 and is the reason the same
measurement has now reached this page four times. **The decision worth making in
the quiet rather than at the stop is whether you want a standing pre-authorised
resume ceiling and expiry, or whether a hard halt until you look is what you
intend.** Either answer is fine and I am not recommending one; what is not fine
is that today the answer is "whichever happens, nobody is watching".

  **Correction to my own organ, stated before anything else is asked of you.**
  Yesterday this page told you *"THE WHOLE PROJECT STOPS IN ABOUT SIX HOURS
  UNLESS YOU ACT"*, forecasting the stop at ~13:00 UTC on 10-02 from 1.77 pts/h.
  **It did not fire.** That rate was a 4-hour burst extrapolated across a day;
  the realised rate since was **0.125 pts/h and then zero for 15 hours**, an
  overstatement of ~14×. The honest statement today is a **range, not a time**:
  the measured external draw over five days spans 0.0–1.4 pts/h, the week resets
  ≈2026-10-07 12:40 UTC, and **+0.08 pts/h from here is enough to reach the stop
  first**. The ask above is unchanged; the urgency claim is withdrawn and
  recorded as an EVIDENCE ADDENDUM on `D30` so 1.77 pts/h is not inherited as a
  measurement.

**2. NO-DECISION: `D30`'s standing report, delivered here because the page it is
supposed to live on says the opposite.** Builder dark **52 consecutive slots /
52.5 h**, the longest on record; last `rc=0` 2026-10-01T02:17:07; demonstrated
**107 → 107** for 48 hours; first legal slot ≈**2026-10-06 15:00 UTC** on a flat
meter. **`2026-W39`: 1.07 h drawn of 30 free Kaggle GPU-hours; ~28.9 h expire at
tonight's Sunday reset — the third consecutive week lost** (W37 5.25/30, W38
0.92/30; ~83 free GPU-hours in three weeks). **No dispatch has been manufactured
to spend tonight's hours and none should be** — both live routes run through
`T1.08` (FAIL), whose repair design is the Review's own row and went OVERDUE
yesterday. All four organs fired within the last hour; none is silent past 2× its
cadence. **`docs/PROGRESS.md`, which `D30`'s armed default made the vehicle for
this report, currently reads "Dark slots 0 … the blackout stayed closed" and has
not been rewritten in 72 hours.** That is RANK 1, it is routed to the builder as
FTB 4 and 5, and it is the reason you are reading this paragraph here instead of
there.

**3. Nothing new is asked on `D33`, and this is a pointer, not a re-ask.** It is
now **10 days** past its `decide_by`, it is the sole cause of the one broken
ratchet class on the register, and the 10-02 Review established that its default
is **MOOT rather than merely expired** — its object went terminal when
`w1-world-edit-window` was stamped `DECLINED`, so **no desk can clear this by
firing anything.** The Review's own published stop-rule fires **2026-10-09**, at
which point the three orphaned rows are DECLINED to you as a class. Its
recommendation stays quoted verbatim in the entry and is unchanged.
