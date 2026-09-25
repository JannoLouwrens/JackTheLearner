# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Findings are ranked by how much damage they do to the
> trustworthiness of the ledger, not by how interesting they are.

**2026-09-25 00:37–00:5x UTC — the 115th audit.** Six hours after the 114th
(18:37). The window is the builder's six live slots 19:0x–00:1x (`a2b83e6`,
`3193c8b`, `4734b63`, `8fa15de`, `05a582d`, `12a8180`) and no Review sitting —
today's DAILY fires at 06:37, six hours from now.

---

## VERDICT: DRIFTING — the ledger verifies clean and the one code change in the window is a genuine strengthening; what is drifting is the project's MEMORY of its own commitments, and I found a live casualty that no instrument in this repo can see even after the repair already routed for it

Two things happened in this sitting that are worth separating.

**The mandated act.** `D32` and `D34` reached their first legal firing moment at
midnight and I fired both, with the required wording, in
`docs/DECISIONS_NEEDED.md`. Neither firing orders as much work as it looks like:
`D32`'s default is **already implemented on disk** and I verified that by file
and line rather than believing the journal, so the firing is a recording act.
`D34`'s is half delivered and half outstanding, and the outstanding half carries
its own verification precondition. Details in FINDING 2.

**The finding.** While covering `docs/PROGRESS.md` by eye — which I must,
because the instrument that covers it is blind — I found that the 2026-09-23
page's `FOR THE OWNER` item 3 ended with *"That is the next instrument I would
build, and unless you object I will build it rather than write this paragraph a
fourth time"*, that the 2026-09-24 page dropped the item entirely, that the
instrument does not exist, and that **no queue row, no decision entry and no
page carries it.** That is this organ's founding scar happening again. The part
that is new — and the reason this is RANK 1 rather than a re-statement of an
already-routed row — is that **the repair already scheduled for the blind reader
would not have caught it either.** FINDING 1.

---

## FINDING 1 — the owner-ask organ has two holes, one is routed and one is not, and the unrouted one is the one that swallowed a real commitment yesterday (HIGH)

### 1a. The known half, re-derived rather than inherited

`decisions.owner_asks(docs/PROGRESS.md)` returns `[]`. I re-derived the
mechanism at this HEAD:

```
decisions._ITEM = ^(\d{1,2})\.\s+(.*)$          # the digit at column 0
docs/PROGRESS.md FOR THE OWNER item 1 = "**1. NO-DECISION: liveness and ..."
                                          ^^ bold marker before the number
```

Every item under `## FOR THE OWNER` fails `_ITEM`, so the section parses as zero
items, so `UNROUTED-OWNER-ASK` and `VANISHED-OWNER-ASK` are **structurally
unfireable** and `BASELINE_UNROUTED_ASKS = 3` / `BASELINE_VANISHED_ASKS = 0` sit
green on an empty population. Sixteen days.

**This is already found, already diagnosed to the exact regex, and already
routed** — `owner-ask-reader-blind-since-0909`, OPEN, `DUE: 2026-10-01`, routed
2026-09-23 by the builder as a by-product of the `fieldwatch` row. I am not
re-reporting it. I am adding two measurements to it that the routing slot could
not make, and one correction.

**THE CORRECTION, and it makes the repair cheaper and safer than the row
believes.** The row says the counter *"may go red > 3 the moment it can see
again, and that red is information, not a bug"*, and dates the repair to
2026-10-01 partly on that caution. I ran it. Patching `_ITEM` to
`^\*{0,2}(\d{1,2})\.\s+(.*)$` — the one line the row itself proposes — parses
**4 items on the live page and 5 on the previous committed one**, the
`NO-DECISION:` exemption test still matches (`_NO_DECISION` carries
`re.MULTILINE`), and:

```
owner_ask_findings(live, prev, NEEDED, RESOLVED)  ->  0 findings
BASELINE_UNROUTED_ASKS = 3     BASELINE_VANISHED_ASKS = 0
```

**The honest first reading is 0 and 0 — GREEN, not red.** The repair cannot
breach either baseline, so nothing about it needs a quiet day. That reading is
what the row's `DUE:` requires be written into the commit, and it is now written
here so the executing slot does not have to re-derive it.

### 1b. The unrouted half — a repaired parser would STILL print zero

`owner_ask_findings` skips exempt items in **both** loops:

```python
for a in owner_asks(prev_progress_text):
    if a["exempt"] or a["shingles"] & here:
        continue                       # answered-shape, or still on the page
```

`exempt` is true when the item's lead carries `NO-DECISION:`. And the vanished
item was `**3. NO-DECISION: a report on this desk's reliability ...**`.

So: `NO-DECISION:` was built to mean **"this needs no ruling from you"**, and it
correctly suppresses `UNROUTED-OWNER-ASK`. It is also being read as **"this
needs no memory"**, which nobody decided and which suppresses
`VANISHED-OWNER-ASK`. A desk can declare `NO-DECISION:`, make a commitment
inside that item, and have the commitment roll off a current-state page with no
organ able to notice — which is exactly the shape my charter was written after.

**THE CASUALTY, traced end to end.** The 2026-09-23 page
(`51efe92:docs/PROGRESS.md`), `FOR THE OWNER` item 3:

> *"the failure now has a shape — this desk reads its own date lines and not its
> own bodies — and the repair is obvious and cheap: the builder shipped
> `STEERING-METRIC-MISMATCH` yesterday for quoted NUMBERS, and the same
> treatment for asserted ABSENCES would have caught all three. **That is the
> next instrument I would build, and unless you object I will build it rather
> than write this paragraph a fourth time.**"*

Checked, four ways, all at this HEAD:

| check | result |
|---|---|
| is it on the 09-24 page? | **no** — item 3 is now "the bottleneck has moved"; `grep -in "absence\|asserted"` on `docs/PROGRESS.md` returns nothing |
| was the instrument built? | **no** — `experiments/steering.py` carries `STEERING-PAGE ORDERS`, `-DATE-MISMATCH`, `-METRIC-MISMATCH`, `PAGE SIZE` and no absence channel; last touched `5b18cd3` 09-23 21:14, which is the metric-mismatch false-positive fix, not this |
| is it routed to the queue? | **no** — no row matches `asserted absence` / `absence-mismatch` / `steering-absence` |
| is it in a decision file? | **no** |
| would a repaired reader see it? | **no** — `exempt=True`, skipped by the VANISHED loop |

**The mitigation, stated because burying it would be the same sin.** The 09-23
page was sealed under the INCOMPLETE-RUN banner (`51efe92`, `rc=124` at
09:42:09) — *"any verdict, any section claiming no findings ... are
UNVERIFIED"*. So the desk's culpability is real but partial: it published a
commitment from a run that did not finish, and the next morning's complete page
replaced it wholesale, which is what a current-state page is supposed to do.
**The instrument's blindness is not mitigated at all.** The whole point of
`VANISHED-OWNER-ASK` is that a current-state page rewriting itself is normal and
must not be allowed to eat things silently.

**Also worth one line, because it is a pattern and not an accident.** This is the
**third** reader in eight days found reporting an empty class as a green one:
`fieldwatch`'s quotation channel (0 for 5, 09-21), the dark-slot counter
(blinded by the loop's own notice lines, 09-22), and now this. Every one was a
*population selector* that quietly stopped matching. The lesson is appended to
`docs/LESSONS.md` under this audit.

---

## FINDING 2 — `D32` and `D34` fired at the first legal moment; `D32` orders no work at all and `D34`'s own cost premise was stale in the builder's favour (HIGH, and it is discharged in this commit)

The 114th audit recorded that the builder's hand-off of these two to the 18:37
sitting was one sitting early, and that the earliest legal firing was this one.
`decisions --check` at 00:4x listed both under **`OVERDUE — DEFAULT IS DUE TO
FIRE`**. Both firings are written into `docs/DECISIONS_NEEDED.md` with the
required sentence, the reversal, and the transcription owed to the builder per
`D13`/`D22`/`D29`. After the append, `decisions --check` lists **`D31` alone**
under armed.

**`D32` — option (ii) SEE IT AND SAY IT. The default was already implemented and
I verified it by file and line, not from the journal:**

```
scripts/launch_detached.sh:52  setsid ... JACK_DETACHED_LANE="launch_detached.sh $LOG" ...
experiments/run.py:4574        DETACHED_LANE_ENV = "JACK_DETACHED_LANE"
experiments/run.py:4590        def _lane_verdict(...)
experiments/run.py:4656        "this lane is D32's question (the owner's); until it rules ..."
```

Landed 2026-09-20 under the 105th audit's item 1 — which `D32`'s own closing
paragraph anticipated in writing. **The firing orders no code.** The refuse/
permit line is untouched in both directions and the scope question stays on the
owner's desk, which is the part that actually mattered and which (ii) explicitly
does not buy.

**`D34` — option (iii) BOTH, IN THAT ORDER. Half delivered, half outstanding,
both verified:**

```
(ii) trim    DONE      scripts/ladder_prompt.md = 90935 B  (cliff 131072)
(i)  stdin   NOT DONE  scripts/ladder_loop.sh:282  timeout 50m claude -p "$PROMPT"
```

The stdin change is ordered onto the builder **carrying the default's own
precondition verbatim**: verify in the same slot that `claude -p` reads stdin on
this harness, and if it does not, do not take (i) and return the entry with the
measurement.

**And the premise, re-checked BEFORE firing rather than after.** `D34` prices its
urgency on *"24 hours, 0 iterations, 0 ledger events"* and *"~3976 bytes/day ...
about ELEVEN DAYS"*. Measured today: **45 consecutive `rc=0` iterations** since
the last non-zero exit at `2026-09-21T06:07` (`rc=126`), 25 of them in the last
24 h, zero dark; the page at **90935 B, +1035 B/day, 39 days to the cliff**. The
growth rate is a quarter of what the entry assumed and the runway is five weeks,
not eleven days. It does not change the firing — a repair with a computable
expiry is still a repair with an expiry — and it is on the record because *"the
premise dies under a deadline"* is now a four-instance structural shape.

---

## FINDING 3 — `review_queue_violations` 0 → 7 at midnight, exactly as forecast, and the one stop-rule among them would fire a DECLINE on work that is finished (MEDIUM-HIGH, and it is the Review's at 06:37)

`run review-queue` EXIT 2, **7 VIOLATION(S) — OVERDUE 7**, forms `{}` →
`{'OVERDUE': 7}`, baseline 0 set 09-22. Cause is **CLOCK, not act**: seven live
rows carried `DUE: 2026-09-24` and the calendar reached it while no organ with
stamping authority sat. The instrument's own cause store says so; no commit is
to blame and none may record it.

Five (`ps05`, `ps06`, `ps08`, `ps09`, `lt02`) are first breaks on the Review's
own 09-19 redesign dispositions — ordinary work arriving. `t215` is a second
break. **`pl02-eye-gate-reads-the-encoder-not-the-eye` is the fourth**, and it
carries a stop-rule binding on the desk in its own words: *"if this date breaks
too, the row is DECLINED and the finding goes to the owner — a promise renewed
four times is not a promise."*

**I re-verified the underlying debt myself rather than repeating the 114th
audit.** The 09-11 ruling ordered the eye-aliveness VOID gate rebound to a
raw-pixel radius ridge with `EYE_RADIUS_R2_MIN` unmoved at 0.80. `a4132c8` (spec
edit) and `c150187` (smoke) both resolve in git. The ledger's `PL.02` row is
attempt 2, VOID, `ran_at 2026-09-13T01:11:50`, with **`r2_raw_pixel =
0.929242`** — a blind eye cannot produce a 0.93 raw-pixel ridge. **The work
ordered by that date is on disk and on the ledger. A `DECLINE` would tell the
owner the desk cannot produce work the builder finished twelve days ago, and it
would be false.** Third consecutive audit to say this; it costs one stamp.

**And the pile behind it is the real number.** `IMMINENT` reads **15 live dated
rows** due on or before 2026-09-25 against a measured capacity of **6/cycle**,
**9 of which cannot be discharged by that cycle**; the 09-25 column carries 8,
`!! AMBER`. Drain is **UNBOUNDED** — 18 arrived and 8 disposed over 7 cycles, 57
live rows, arrivals exceeding disposals by 10. `review_queue_piled_on` 2 → 4,
traced by the builder to two committed routings onto an already-full 09-25 and
said, not recorded, correctly. `review_queue_net_arrivals` 7 → 10 is the
trailing window sliding and no commit can justify recording it.

---

## FINDING 4 — `WAITS-ON:` shipped clean, strengthened its gate, and has no producer; the obligation to use it lives only in the body of a row about to go terminal (MEDIUM)

This is the only code change in the window and I checked it adversarially,
because it is exactly where an optimistic loop would launder something.

**It is good work and nothing is weakened.** The implementation honours every
clause of the 09-19 disposition: declaration-only; `none` permitted; the grouped
count printed only when every live row on a date declares, and a `WITHHELD —
n of m rows undeclared` line rather than silence when it does not; a corpse
reference MALFORMED on live rows with terminal roots legal. `T0.31` went
**20 → 22 properties** and I read both new ones: P21 asserts the arithmetic, the
gate, and declaration-only in four directions (`counts`, `due_pile`, `piled_on`,
`next_free_due` byte-identical against the same document bare, and the OVERDUE
sets identical so the field buys no exemption); P22 draws the corpse boundary
both ways. The ledger row is `attempt 22, PASS, 1.72 s, commit 05a582d,
dirty_files None, properties_checked 22.0, failed_names ''`, and the control
fails **19** including `p21` and `p22`. Section 2 has nothing against it.

**What it cannot do is its job.** Every date reads:

```
2026-09-24  WITHHELD — 7 of 7 rows undeclared
2026-09-25  WITHHELD — 8 of 8 rows undeclared
... nine dates, nine WITHHELDs, zero declarations in 57 live rows
```

That is the designed behaviour on partial adoption and I am not calling it a
bug. The defect is upstream: **nothing instructs any router to declare.** I
grepped. `WAITS-ON:` appears in `scripts/ladder_prompt.md` once, as the name of
a dated unit; in `docs/SYSTEM.md`, `scripts/review.sh` and this charter, never;
in `review_queue.py`'s contract docstring as an *optional* body line; and there
is deliberately no violation class for an undeclared coupling. The one place the
obligation is written is the disposition's own cost paragraph — *"Every future
router now owes a judgment about coupling on every row it writes, including the
judgment 'this one is independent'"* — **in the body of
`waits-on-declared-field`, which is `DUE: 2026-09-25`, executed, and will be
stamped ACTED at 06:37 today.** By this project's own rule a terminal row is
never re-read.

**This is the 114th audit's FINDING 1 in advance instead of in arrears.** That
one was an obligation parked on a door that had already shut, found eight days
late. This is an obligation about to be parked on a door that shuts in six
hours, and it costs one sentence in a live page to prevent. The repair is not
code and not a new violation class — it is putting the sentence where routers
read: `scripts/ladder_prompt.md`'s routing instructions and the Review's own
sitting order. Routed below.

---

## FINDING 5 — the standing reds, each re-derived, none moved by anyone's hand (LOW, and listing them is the point)

1. **`DEFAULT-ACTION-EXPIRED` = 1, baseline 0** (`decisions --check`, EXIT 1) —
   `D33`, whose default names `2026-09-23` while its own `decide_by` is
   `2026-09-23`, so on the first day the default can fire its action is already
   past. Armed on purpose by the 112th audit and certified onto `T0.28`. It is
   `CONDUCT-DESK` and now **`STALE by 2 day(s)`** — one day worse than
   yesterday. The Review sat at 06:37 on 09-24 and cited `D33` in `FOR THE
   OWNER` without discharging the expired action; both repairs the instrument
   offers (shorten `decide_by`, or declare `(CLOCK: <whose>)`) are untaken. Not
   the builder's and not mine. `D35` is also `CONDUCT-DESK`, `STALE by 1 day`.
2. **`coverage` EXIT 2**: `goal_unrunnable = 7` against a baseline of 3, the
   four `GEN` ids being the overflow — the 114th audit's FINDING 1, **routed by
   the builder in the window** as `gen-four-reparented-to-a-decision-that-had-
   already-closed`, OPEN, `DUE: 2026-10-01`. Verified present. `claim_dead = 4`
   (smell, balance, shelter/building, thermal), `park_release_pairs = 3`
   (`BA.02→LT.08`, `SH.01→SH.02`, `SM.02→SM.03`), `NO-LIVE-PATH` 7 distinct
   commitments/seats, lower bound 6.
3. **`pass_on_dead_dependency = 3`** at floor — `LF.02←T6.03`, `T2.03←T1.08`,
   `T2.14←T1.08`. **`unreachable = 96`** at floor. **`champions_trigger_debt =
   3`** unmoved 22 days, **`champions_unwinnable = 4`** unmoved 12 days.
   `champions --check` EXIT 0, every ratchet intact. The **World** seat still
   declares no `TRIGGER:` at all and names no deciding run, and the Review has
   now deferred it on the `w1-world-edit-window` ground for **three consecutive
   sittings** and said so on its own page. `fail_unowned = 0` at floor with 28
   of its 29 owners being queue rows against a desk whose drain reads UNBOUNDED.

---

## The audit, section by section

**1. Integrity of the ledger — CLEAN.** `run verify`: 109 PASS entries
re-judged, 107 controls probed. **0** verdicts that no longer re-derive, **0**
gates that ignore their control, **0** that could not be replayed, **0** that
could not be audited, **0** controls run but not declared. Every PASS commit
still resolves. Two PASSes with no control at all (`T0.01`, `T0.10`) — existence
claims whose gate was never shown capable of reporting the bad case;
long-known, structural, unchanged. One self-exclusion (`T0.18`), correct by
construction. Fourteen `impl_sha`-stale non-PASS rows and one pre-`impl_sha`
content-stale row (`T2.02`) are re-runs owed, not claims made. Two dirty stamps
(`T6.03` BLOCKED, `PL.02` VOID), both flagged by the instrument, both with
re-buys prohibited or deferred for stated reasons.

**2. Thresholds and controls — NOTHING LOOSENED, and the one change moved the
hard way.** The window's only code touching `experiments/tests/` is `05a582d`,
which takes `T0.31` from 20 to 22 properties and adds both to the control's
required failures (control now fails 19). No numeric threshold appears in the
diff in any direction; no control was deleted or weakened; no `_check` gained an
`or`; no seed count moved; no assertion was removed. I also re-opened the
trailing week rather than trusting prior clearances: the only loosening move in
seven days remains `PS.08`'s `GAP_ABS_MIN` 0.015 → 0.008 (`8f7d1dc`, 09-19),
disclosed in its own commit's first line, justified by a measurement, re-frozen
against the final fixture before commit, audited three times at the time — and
`PS.08` FAILed anyway, so the move bought nothing. It stays clean.

**3. Drift from the goal — no drift in the window; the converse question is
still the damning one.** Six commits: four journal lines, one instrument
implementation ordered by a dated disposition, one `T0.31` re-buy. Every one
traces to measurement hygiene rather than to a `GOAL.md` sentence, and I decline
to call that drift for the same reason my predecessor did: the board was
genuinely empty on every re-derivation and manufacturing a dispatch would have
been worse. The converse: **4 of `GOAL.md`'s own constitutional commitments are
`CLAIM-DEAD`** — *smell*, *balance*, *shelter/building*, *thermal (kills)* —
every claim spec parked or foreclosed, with 3 park→release pairs whose revival
path cannot be walked today. Curiosity: 12 specs, **2** pass. One brain /
unison: 28 specs, **1** pass. Fast/slow: 8 specs, **0** pass. Told world: 1
spec, **0** pass. `QUEUE DEPTH` dispatchable today: **4, of which 4 are VOID — 0
fresh dispatches**, and every one of the seven cost classes is marked NOT
FILLABLE with the repair being a redesign. Those are precisely the claims my
charter warns are quietly neglected, and they are.

**4. Builder alive and productive — alive, disciplined, and the discipline is
the output.** **45 consecutive `rc=0` iterations** since 09-21T06:07; 25 in the
last 24 h; **0 dark slots**; PASS delta **0** (110 → 110). Meters named both
ways every slot, `week:all models` 40 → 41% the gate and the line acted on.
`lost_iterations.log` 0 bytes, no leftover pids, tree clean before each commit,
everything pushed. Specific things it did right that I checked rather than
accepted: it **refused to fire `D32`/`D34` early** at 22:0x after re-verifying
they listed as armed-not-OVERDUE, leaving them to this sitting exactly as the
114th audit ordered; it **traced `piled_on` 2→4 to the two committed routings**
before calling it motion; it **verified four `pgrep` detached-lane hits on `ps`**
and found them to be its own prompt text self-matching; it **left the
`waits-on-declared-field` row unstamped on purpose** because builder slots may
not dispose queue rows; and it **pre-verified tomorrow's unit against
`REVIEW_QUEUE.md:7389` the night before** so this morning's slot owed no
archaeology. No manufactured work, no inherited claim, no threshold touched.

**5. Compute honesty — nothing wasted, and the same perishable loss two days
out.** `2026-W38`: **0.9176 h drawn of 30 across 2 jobs; ~29.08 h expire
Saturday 2026-09-26**, and there is **no legal buyer** — `coverage` prints every
GPU cost class either EMPTY or NOT FILLABLE behind a redesign. The builder
refused to manufacture a dispatch in all six slots and named the refusal each
time; **that is the correct call and I am endorsing it.** Standing historical
waste, unchanged and already recorded: `gpu_hours_no_verdict` TOTAL **48.42 h**,
of which `D1.0` **33.78 h / 2 attempts / 0 verdicts** and `UNATTRIBUTED`
**6.32 h / 21 jobs** at its floor of 21.

**6. Stuck decisions — 0 `MEANS-ESCALATED`, 0 `UNDECLARED`, 2 `OVERDUE` FIRED.**
I checked the class census directly. **Nothing is sitting on the owner's desk
that a measurement could settle.** `D32` and `D34` are fired and off the armed
list; `D31` alone remains armed, `due 2026-09-25`, and its earliest legal firing
is 2026-09-26 — the 108th-audit precedent, and I am not repeating the 114th
audit's correctly-declined early firing in the opposite direction. **There is no
`UNDECLARED` entry to arm this audit**; my charter asks for one per sitting and
the honest report, for the third consecutive audit, is that the class is empty.
I would rather say so than manufacture an arming. One owner decision — the
09-23 `NO-DECISION` commitment in FINDING 1 — **was quietly dropped rather than
quietly acted on**, which is the other half of section 6's question and the
first instance of it I have had to report.

**7. Bakeoff hygiene — one disclosed caveat, no new ones.** `T4.06` remains the
only first-ever PASS in the trailing week and its winner is inside the noise
margin: conjunct (2) certified at **+0.0187 = 6.9% of the incumbent's own seed
spread 0.2699, 1 of 3 seeds regressing**, against `grad_norm` −56.7%. The desk
wrote down (`f7900b5`) that the latent-recovery conjunct is **not to be quoted
as demonstrated**, and `run status`'s `ANCHOR-DECIDED CONJUNCTS` block now
prints the margin structure so a reader cannot miss it. That is a winner inside
the noise margin, disclosed by the organ that owned it, before any auditor
asked. No VOID is being treated as a verdict. No decision was made without a
learning gate in the window.

**8. The honest summary — no, and the number that says it is 45.** We are not
closer to a curious humanoid that climbs the ladder than we were six hours ago.
**110 → 110 for 37 hours across 45 consecutive clean slots.** The one thing that
moved is an instrument that counts the backlog slightly better, and it counts
zero declarations on 57 rows. The builder is doing everything right and produced
nothing, legitimately, because the board is empty and the work that would fill
it — `W1.01`/`W1.03`/`W1.04`, eighteen days unregistered; the `T1.08` repair
that blocks 45 specs; five redesigns — sits at a desk whose drain reads
UNBOUNDED and which will face 15 dated promises at 06:37 against a capacity of
6. **What is new and worse today is not any of that. It is that the organ which
is supposed to remember what this project promised itself cannot see its own
`FOR THE OWNER` section, a commitment made two days ago has already vanished
out of it, and the repair already scheduled for the blindness would not have
caught the thing that vanished.** A ladder nobody is climbing is a stall. A
ladder whose record of its own promises leaks is how a stall stops being
visible.

---

## FOR THE BUILDER

1. **`D34`'s default fired and it carries the one code order on this page.** In
   a live slot, **first** launch one throwaway prompt through stdin and confirm
   a non-empty response from `claude -p` on this harness; **only if that
   succeeds**, change `scripts/ladder_loop.sh:282` to feed the prompt on stdin
   instead of argv (`ladder_loop.sh:270` already has it in `$PROMPT`). **If the
   verification fails, do not make the change** — write the measurement down and
   the entry returns to the owner. That conditional is the default's, not mine.
2. **`D32`'s default fired and orders you NO code.** The implementation landed
   on 09-20 and I verified it at `launch_detached.sh:52`, `run.py:4574/4590/
   4656`. What is owed is the `DECISIONS_RESOLVED.md` transcription for **both**
   `D32` and `D34`, per the `D13` rule and the `D22`/`D29` precedent — quote the
   firing blocks I appended to `docs/DECISIONS_NEEDED.md`. Do not touch
   `_lane_verdict`'s refuse/permit line: the scope question is still the
   owner's.
3. **FINDING 4 — put the `WAITS-ON:` obligation somewhere that is not about to
   go terminal, and do it BEFORE 06:37 if you get a slot.** The only written
   instruction to declare coupling is in the body of `waits-on-declared-field`,
   which is `DUE: 2026-09-25` and will be stamped ACTED this morning; a terminal
   row is never re-read. **Minimum repair, no code:** add one line to
   `scripts/ladder_prompt.md`'s routing instructions saying that a new
   `REVIEW_QUEUE.md` row carries `WAITS-ON: <row id> | why` or `WAITS-ON: none |
   why not`, and route a queue row asking the Review to carry the same sentence
   into its own sitting order (suggested id
   `waits-on-has-no-producer-outside-a-closing-row`; `next_free_due` reads
   **2026-10-01**). **Do not** add a violation class for an undeclared coupling
   — the disposition refused that deliberately and it is the Review's grammar,
   not yours.
4. **FINDING 1a — `owner-ask-reader-blind-since-0909` is cheaper and safer than
   its own row says, and its row explicitly licenses an early slot.** It reads
   *"Nothing forbids an earlier slot taking it if the board is empty; the date
   is a ceiling on silence, not a floor on work"*, and you have had 19 empty
   slots. The repair is `_ITEM` → `^\*{0,2}(\d{1,2})\.\s+(.*)$` plus a fixture
   in the `**N. ` shape the live page uses. **The first honest reading is
   already measured and it is `0 UNROUTED / 0 VANISHED`, not `> 3`** — I ran it
   under the patch; put that in the commit as the row requires, and correct the
   row's own prediction while you are there. `_NO_DECISION` already carries
   `re.MULTILINE`, so the exemption test keeps working.
5. **Do NOT try to fix FINDING 1b yourself.** Whether `NO-DECISION:` should
   exempt an item from `VANISHED-OWNER-ASK` as well as from
   `UNROUTED-OWNER-ASK` is a grammar question about the Review's own page, in
   the same class as the `WAITS-ON:` ruling — route it, do not decide it. It is
   on the owner's page below as well, because a desk exempting itself from being
   remembered is not purely a desk matter.
6. **Standing prohibitions, unchanged and still binding:** `T1.08`'s pipeline
   repair, `A4`, `T2.10`, `SO.07`, `SO.10`, `UB.10`'s successor arm choice, the
   world-edit window, the `lc03` seat row, the `t306` venue row, the
   `W1.01`/`W1.03`/`W1.04` registration, and the GEN citations themselves. `D33`
   and its `DEFAULT-ACTION-EXPIRED` red are the Review's — do not fire, extend
   or touch them, and do not move `BASELINE_ACTION_EXPIRED`.
7. **Credit, specifically.** Refusing to fire `D32`/`D34` at 22:0x after
   re-verifying they were armed-not-OVERDUE was the right call and it is why
   this sitting could fire them cleanly. Leaving the `waits-on` row unstamped
   was correct. Tracing `piled_on` to its two committed routings before calling
   it motion was correct. Keep re-deriving rather than inheriting; it keeps
   paying.

---

## FOR THE OWNER

**1. `D32` AND `D34` — THE OWNER DID NOT RULE BY 2026-09-24, SO BOTH
PRE-REGISTERED DEFAULTS FIRED** at ~00:4x today, the first legal moment. Full
records, reversals and evidence are in `docs/DECISIONS_NEEDED.md`. The short
version: **`D32` took option (ii) SEE IT AND SAY IT and orders nobody any work,
because the code already does it** — the lane declares itself and
`_lane_verdict` names the launcher and this entry, landed 09-20. **The scope
question `D32` actually asks — whether `D20`'s closure covers the wrapped
detached lane at every cost class or only at `cpu<48h` — is NOT answered and is
still yours.** `D34` took (iii) BOTH IN THAT ORDER: the page trim is done
(90935 B against a 131072 cliff), the stdin change is ordered onto the builder
behind its own in-slot verification. **Reversal for either is one revert and no
ledger row.** One correction you should have: `D34`'s stated cost — *"24 hours,
0 iterations"* and *"eleven days"* of page headroom — is stale. The builder has
run **45 consecutive `rc=0` iterations** and the real runway is **39 days**.

**2. A DECISION FOR YOU, and it is small, cheap and about your own visibility.**
`docs/PROGRESS.md`'s `FOR THE OWNER` section is audited by
`decisions.owner_ask_findings`, which has two classes: an ask that reached no
decision file (`UNROUTED`) and an ask that rolled off the page unanswered
(`VANISHED`). **Both are skipped when the item is labelled `NO-DECISION:`.**
That is right for `UNROUTED` — a desk saying "this needs no ruling" is exactly
what the label is for. **It is wrong for `VANISHED`**, and yesterday it cost
something real: the 09-23 page's item 3 ended *"unless you object I will build
it"*, the 09-24 page dropped the item, the instrument does not exist, nothing
routed it, and no organ could notice. **The ask: should `NO-DECISION:` exempt an
item from being remembered, or only from being ruled on?** I have not armed this
as a decision entry because it is a one-line grammar change to a desk's own page
format and `D35`'s rule 2 says route rather than build — but it is on your page
because a desk that can declare itself exempt from memory is not purely a desk
matter. If you would rather it be a numbered entry with a default, say so and
the next sitting arms it.

**3. NO-DECISION: liveness, and it is unambiguously good.** **0 dark slots.
45 consecutive `rc=0` iterations** since 2026-09-21T06:07, 25 in the last 24 h.
The blackout that peaked at 26 consecutive dark slots is over. `D30`'s default
requires the perishable price in the same sentence: **`2026-W38` holds 30 free
Kaggle GPU-hours with 0.9176 drawn — ~29.08 hours expire Saturday 2026-09-26,
one day out, and there is no legal buyer for them.** `coverage` prints all seven
cost classes EMPTY or NOT FILLABLE behind a redesign. **No dispatch has been
manufactured and none should be**; a GPU hour spent on a run nothing asked for
is worse than an expired one, and the builder has now refused this in 45
consecutive slots.

**4. NO-DECISION: the bottleneck is where the Review said it was, and today it
has a number.** `review_queue_violations` went **0 → 7** at midnight on the
clock, not on anyone's act. At 06:37 the Review faces **15 dated promises
against a measured capacity of 6**, with **9 that cannot be discharged by that
cycle**, a drain of 18-arrived-against-8-disposed over seven cycles, and 57 live
rows. One of the seven broken rows (`pl02`) carries a self-imposed stop-rule
that would DECLINE it to you this morning, and **I have verified the work that
row is about is finished** — `a4132c8`, `c150187`, and `PL.02` attempt 2 on the
ledger at `r2_raw_pixel 0.929242`. `PL.02` is the sole registered falsifier of
`GOAL.md`'s PLASTIC-ONLY decree. A DECLINE on it would be false and this is the
third consecutive audit to say so.

**5. `D33` — cited, not re-asked, and now two days past `decide_by`.** Its
`DEFAULT-ACTION-EXPIRED` red (1 against a baseline of 0) has stood since the
112th audit armed it deliberately, and the two repairs the instrument offers are
still untaken. The substantive ask is unchanged and asks for less than the
original: rule the narrow thing — that the world EDIT is IMPLEMENTATION and was
never the Review's to hold under `D22` — and hold that desk to registering
`W1.01`/`W1.03`/`W1.04` itself. The cost today: **nineteen days unregistered**,
against a builder that has been idle and correct for 45 slots. `D31` falls due
today and is armed with a written default; its earliest legal firing is
2026-09-26 and it is not mine to answer before then.
