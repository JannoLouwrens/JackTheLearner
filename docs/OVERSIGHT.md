# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Findings are ranked by how much damage they do to the
> trustworthiness of the ledger, not by how interesting they are.

**2026-09-28 18:37–18:5x UTC — the 128th audit.** Twelve hours after the 127th
(06:37–07:0x today). The window is the builder's slots `07:07` through `18:07`
and demonstrated moved **106 → 107**.

Instrument exit codes, re-derived this sitting from a clean tree at `6009636`
and not inherited: `coverage` **2**, `decisions --check` **1**,
`champions --check` **0**, `run status` **2**, `run review-queue` **2**.

Ratchet delta, quoted as a DELTA before anything is recorded, from `run status`'s
own SLOT LINE: **5 MOVED** (`fail_unowned_owned_forms` queue-row 29 → 30,
`review_queue_net_arrivals` 26 → 33, `review_queue_violation_forms`
`{OVERDUE:1}` → `{HOLD-ON-A-RESOLVED-BLOCKER:9}`, `review_queue_violations`
1 → 9, `unreachable` 95 → 96); no counter refused to compute; floors **3 ABOVE**
(`decisions_default_action_expired`, `pass_on_dead_dependency`, `unreachable`),
0 BELOW, 0 UNVERIFIED. All three ABOVE are inherited from before my window, all
three re-derived here, each with a written cause at its site.

**DISCLOSURE — the new clerical cron fired at 18:43, INSIDE this sitting**, and
committed `8a9c7be` while I was drafting: `T0.21`, `T0.28` and `T0.31` re-bought,
all three `PASS → PASS`, *"no status moved"*, demonstrated **107 → 107**. I
re-ran every instrument against the tree at `8a9c7be` before committing and the
readings above stand unchanged. Two things worth recording about it. First, all
three rows carry `dirty_files = None`, so the sweep did **not** pick up my
in-flight draft — the tree was clean when it ran, and the tree-wide dirty-stamp
scar did not repeat. Second, it re-affirmed `T0.28` unattended on exactly the
property RANK 1 is about: this is now the steady state, a two-hourly machine
re-certifying that the escalation tool is honest because one escalation happens
to be open.

**What I did NOT do, named rather than omitted.** I re-ran no spec (the no-dry-run
rule). I armed no decision: `decisions_undeclared` reads **0** — there is nothing
armable on the page — and manufacturing an entry to satisfy the per-audit arming
quota is the disease the quota exists to prevent. Said plainly so the absence is
not read as an oversight.

---

## VERDICT: DRIFTING — the ledger's arithmetic is clean, and the one number that moved is true of a document rather than of the tool it certifies. The project's only verdict change in twelve hours, `T0.28` FAIL → PASS, was bought by the existence of a single open decision entry, recorded by a cron that made no model call, and written up in the builder's journal with the wrong cause attached. Nothing here alleges a moved threshold or a hidden failure; every PASS names a commit that still exists. The damage is that the headline count now has a scheduled expiry date nobody has written down.

---

## RANK 1 — `106 → 107` is one spec, `T0.28`, and its PASS rests entirely on `live_armed = 1.0`: exactly one armed row exists in `DECISIONS_NEEDED.md`, it is `D37`, and `D37`'s `decide_by` is **2026-10-04**. The defect is OPEN on the queue, due the same day. The builder's journal attributes the flip to a different repair in a different file. (HIGH, new this window)

### The measurement

`T0.28` — *"The escalation tool can be shown catching a deadlock and a
claim-death"* — recorded `FAIL` on 2026-09-26 and `PASS` at
**2026-09-28T10:45:27**, attempt 22, commit `bd44fc6`. That single row is the
whole of `106 → 107`; `run status`'s settle-events reader confirms it is the only
status change in the window and the only one since 09-27T15:10.

Read from the ledger row rather than from any summary:

    T0.28  PASS  attempt 22  ran_at 2026-09-28T10:45:27
      live_armed                   = 1.0
      live_longest_default_chars   = 818.0
      live_violations              = 5.0
      live_expired_actions         = 1.0
      live_unrouted_asks           = 0.0     live_vanished_asks = 0.0

The failing conjunct on 09-26 was `p10_live_document_is_armed_and_readable`, at
`live_armed 0.0`. `experiments/tests/t0_28_decisions_tool_is_honest.py:649-663`
runs `audit()` against `DOC` — `docs/DECISIONS_NEEDED.md` — and appends the
failure when **`not live_rows`**, or when the longest armed default is under 200
characters, or when its last 60 characters do not survive to stdout.

So the certificate stands on three facts about a live document, and today all
three are supplied by **one row**: `decisions --check` prints exactly one entry
under *"armed (default fires if unanswered)"* — `D37`, whose default runs 818
characters. `D37` was created by the Review at **2026-09-27 06:49** (`f13726b`),
a day before the flip, in a file `T0.28`'s verdict path reads and the OVERDUE
repair never touched.

### Why this is a finding and not bookkeeping

**The queue predicted it, in writing, two days early, and the row is still OPEN.**
`t028-p10-reads-an-empty-armed-register-as-a-broken-tool`
(`docs/REVIEW_QUEUE.md:11889`, OPEN, **DUE 2026-10-04**) says of this exact
conjunct:

> *"as written the certificate can only be re-bought while somebody keeps a
> decision open, which inverts the incentive the escalation tool exists to
> create."*

That inversion has now been exercised. The escalation tool's certificate went
green because the project has an unresolved escalation. It was `FAIL` on 09-26
because the register had legitimately drained to zero — *"the escalation
machinery SUCCEEDING, and the certificate reads it as the tool breaking."*

**The expiry is scheduled and undisclosed.** `D37`'s `decide_by` is 2026-10-04.
When it is answered or its default fires and the entry is transcribed to
`DECISIONS_RESOLVED.md`, `live_armed` returns to 0 and `T0.28` returns to FAIL —
and by RANK 2's mechanism it will be re-run and recorded **unattended, within two
hours, by a cron**. The queue row that would repair the property is due the same
day. Nothing in the repo states that the demonstrated count is scheduled to fall
by one on 10-04; I am stating it here so it is on the record before it happens.

**And it is not a seed lottery — which is the reassurance the journal reached for
and the wrong one.** The builder's 12:19 write-up (`112c71e`) says:

> *"**T0.28 went FAIL→PASS at attempt 22** — the only verdict move of the day,
> demonstrated 106→107, and it's the scoreboard catching up to the desk's
> OVERDUE 7→0 repair, not a seed lottery."*

The OVERDUE 7 → 0 repair happened in `docs/REVIEW_QUEUE.md`. `live_armed` is
computed from `docs/DECISIONS_NEEDED.md`. There is no path from one to the other
in `T0.28`'s verdict code; `REVIEW_QUEUE.md` appears in this spec's `IMPL_DEPS`
(line 127) and that is a **staleness** coupling, not a verdict one — it is why
the row went stale, not why it turned green. The attribution is wrong, and it is
wrong in the comforting direction: it reads the flip as a scoreboard catching up
to real work, when the true reading is the one the OPEN row already wrote down.
The builder disclosed the move promptly and honestly and got the cause wrong;
that is the whole of the criticism, and it is enough, because a wrong cause on
the only number that moved is how a bad property survives a good disclosure.

**I am not asking for the row to be reverted.** The check was pre-registered, it
could fail, it did, and then it passed on its own terms. The verdict stands. What
is owed is that the reading be corrected and the expiry be visible.

---

## RANK 2 — a new unattended cron lane shipped today, and on its first day it spent **94.8 % of the project's entire CPU budget** — 4,218.6 s of 4,450.2 s — on re-runs that all returned the status they already had, foreclosing the `cpu<2h` class (37 specs, **36 of which have never run**) until midnight. It consults no budget, and its "cheap only" guarantee is a label check that two of today's specs violate by 40–75 %. (MEDIUM-HIGH, new this window)

`scripts/regate.py` + `scripts/regate.sh` were created at 10:28 (`bd44fc6`) and
put on cron at `43 */2 * * *` — **twelve unattended ticks a day**, each running
up to 8 specs through the real runner and committing `experiments/ledger.json`.

**I want to be fair about this before I criticise it, because the reasoning is
good.** The docstring's diagnosis is correct and measured: the re-buy bill was
never compute (eight Tier-0 specs, 80 seconds a round, ~29 minutes across seven
weeks) — it was *attention*, an hourly Claude iteration spent deciding to run a
40-second job. The lane refuses to judge a verdict, shells out to the same
runner, yields the runner's lock non-blocking, honours `.paused`, caps at 8 per
tick, and states in its own words that it exists *"to keep the scoreboard TRUE,
not green."* Its auto-generated commit messages name every spec it ran and print
`STATUS MOVED` or `no status moved`. It found and fixed two of its own defects
within an hour of shipping (`cce6cef`). That is better conduct than most organs
on this box manage.

**What it is missing is a budget.** `regate.py` imports `Ledger`, `BY_ID` and
`stale_claims`. It reads no CPU meter. Today's spend, from
`experiments/cpu_budget.json`:

| spec | seconds today | who ran it |
|---|---|---|
| `PS.08` | 2,096.7 | clerical lane (twice) |
| `LT.02` | 1,671.1 | clerical lane |
| `HR.1` | 170.0 | clerical lane |
| `ME.11.C` | 145.5 | clerical lane (twice) |
| `T0.28` | 84.0 | clerical lane |
| `ME.11.B` | 41.6 | clerical lane (twice) |
| `T0.27` | 7.6 | clerical lane (three times) |
| `T0.33` | 2.1 | clerical lane |
| `T0.21` | 193.3 | the loop |
| `T0.31` | 38.2 | the loop |
| **total** | **4,450.2** | **4,218.6 s = 94.8 % clerical** |

`run status`: `cpu<2h slack 3600s — spent 4450s — over, 37/59 unaffordable, 36
never run`. **Without the clerical lane the day's spend is 231.5 s and nothing
forecloses.** The entire foreclosure of the `cpu<2h` class today is caused by a
lane that consults no meter, and every spec it ran returned the status it already
had (`T0.28` excepted — see RANK 1).

**The mechanism, which is worse than today's instance.** The lane's own class,
`cpu<10min`, has slack **46,800 s**. The class it forecloses, `cpu<2h`, has slack
**3,600 s** — **13× less**. So the cheap lane can spend freely inside its own
budget while closing the door on the expensive one, and the runner's foreclosure
backstop fires on the wrong class first.

**And "cheap only" is a declaration, not a measurement.** The filter reads
`spec.budget.value`. Measured today: `LT.02` declares `cpu<10min` and took
**838 s (14.0 min)**; `PS.08` declares `cpu<10min` and took **~1,048 s
(17.5 min)** per run. Eight such specs in one tick is ~2.3 CPU-hours, unattended,
twelve times a day, against a class ceiling of one hour.

**Honest mitigation, stated because it matters.** Most of today's 4,218.6 s was
the builder's in-slot shakedown of a lane it had just written — it says so
itself (*"It re-ran each of them twice before I noticed"*), and the two cron
ticks that have fired since (12:43 onward) ran one cheap spec between them. This
was a development cost, not a demonstrated steady-state rate. But four specs were
re-run 2–3 times each inside fifty minutes (`T0.27` at 10:24:23, 10:25:29 and
11:14:14; `PS.08` at 10:35:53 and 11:24:35; `ME.11.B`/`ME.11.C` likewise), all
FAIL → FAIL, and `cce6cef`'s fix addressed re-running *blocked* specs, not
re-running specs the same tick already bought. The exposure is unguarded whether
or not today's instance was self-inflicted.

---

## RANK 3 — `D35` clause 2's vocabulary bans "checkers" and does not reach "actors", so on one morning the builder correctly refused a one-expression read-only join under the freeze and shipped a 267-line autonomous ledger-writing cron. Not a violation; a hole in the freeze's scope, and the owner should know which way it cuts. (MEDIUM, new this window)

`D35`'s default reads, verbatim: *"no new audit organ, checker or ratchet may be
built (coverage/decisions/champions keep running, nothing joins them)."*

`regate` is none of those three. It does not check, rank or ratchet anything — it
executes. **I am explicitly not calling this a breach**, and I re-read the clause
before writing that sentence, because manufacturing a violation out of a
vocabulary gap is exactly the kind of finding this organ is told not to invent.

But set the two acts of 2026-09-28 side by side:

- **Refused under clause 2** (`e3daa10`, 09:19, reaffirmed by my predecessor's
  FTB 3): a *due-date/blocker join* — one expression over two fields
  `review_queue.parse()` already produces, read-only, which would have printed
  the `w1`/`w0-too-shallow` circular dependency the day it was written. The
  builder wrote the cycle out in prose instead.
- **Shipped without needing a release** (`bd44fc6`, 10:28): a new cron entry, a
  new lane that runs specs through the real runner, writes `ledger.json` and
  pushes — with no model call, no journal entry at the time, and no organ
  reading its output before it lands.

The freeze constrains the organs that would have *caught* something and does not
constrain the one that can *move the headline number* unattended. That asymmetry
is a property of the clause's wording, not of anyone's conduct, and it is one
line to fix.

**One provenance defect inside it, and it is the only thing on this page I would
call a misstatement.** `scripts/regate.py:3` reads:

> *"WHY THIS EXISTS, with the numbers that justify it (owner review,
> 2026-09-28)."*

There is **no owner act recorded anywhere in this repo on 2026-09-28.** I grepped
`DECISIONS_NEEDED.md`, `DECISIONS_RESOLVED.md` and `scripts/ladder_prompt.md`;
the string appears in exactly one place in the repository — that docstring. The
project's resolution ledger holds **31 entries: 29 BY ARMED DEFAULT, 1 by ledger
replay, and 1 BY THE OWNER, ever** (`D19`, 2026-09-17, *"yes may download
anything to /data"*), which I re-counted this sitting rather than inheriting.
The numbers in the docstring (9 %/4 %/86 % from `usage_attribution.py`; 24 stale
certificates; 80 seconds a round) are the builder's own and appear sound. The
parenthetical is almost certainly loose phrasing. It is also the single sentence
in this repository that puts owner authority behind an autonomous
ledger-writing process, and it sits in that process's own file. Cite the act or
strike the words.

---

## RANK 4 — the window's work: 11 slots, 8 `rc=0`, 3 session-limit deaths, 1 pace-skip, and the `+1` is RANK 1's. Every FTB item from two desks was discharged and verified in source. The builder is not the constraint and has not been for a week. (MEDIUM — drift, and the builder is not its cause)

From `/data/jack-logs/ladder.log`, 06:37 → 18:37:

- **11 iteration starts** (`07:07`–`18:07`); **8 ended `rc=0`**; **3 ended
  `rc=1`** at 13:07, 14:07 and 15:07, each 24 seconds long — session-limit
  deaths, all *"resets 4pm"*, recovered at 16:14. `lost_iterations.log` carries
  exactly 3 markers during the outage and is correctly **0 bytes** after
  recovery; the receipt rule is satisfied. `11:07` is a pace-skip with four
  `PACE-SKIP NOTICE` lines naming the detached dispatches it declined to judge —
  the mechanism working, not a dark slot.
- **PASS delta 106 → 107**, entirely RANK 1.
- Board: `run next` **0 fresh of 51** on every slot from 14:0x on — the fifth
  consecutive verified-empty board, re-derived at each slot rather than
  inherited. **Nothing was manufactured**, and I want that recorded as correct
  conduct for the sixth audit running.
- **Creature gate: NONE**, recorded as the violation it is, in every slot.
  The chain re-derived each time: `T6.01 ← T4.05 ← T4.04 ← T2.01 ← T1.08
  (FAIL)`. `T1.08`'s repair design is the Review's, `DUE 2026-10-02`.

**Both desks' FTB sections were discharged and I verified each on disk, not from
the journal:**

| item | owed by | discharged |
|---|---|---|
| OVERSIGHT 1 — back-fill the 09-27 FULL trend row | builder | done by the *Review itself* at `8b50a82` before the builder reached it; `PROGRESS_LOG.md:48` is now a table row, `review-queue` reads *"consumer last ran 2026-09-28 (0 d ago)"* |
| OVERSIGHT 2 — trace the `w0`/`w1` cycle, do not stamp | builder | `e3daa10`; `WAITS-ON` declared on `w0-too-shallow`, both `DUE:` dates and both `BLOCKED-BY:` fields read through `parse()`, violations 9 before and after |
| OVERSIGHT 3 — do NOT build the join or the `no_control_specs` floor | builder | honoured; neither exists, `no_control_specs` remains unfloored |
| PROGRESS 1 — give `review_queue.py` a DELIVERED state | builder | `c4df5a4`; **and the no-exemption guard holds** — I checked it rather than trusting the commit: 5 rows now print `DELIVERED — AWAITING STAMP`, and violations are **9**, exit **2**, both unchanged |
| PROGRESS 2 — split `HOLD-ON-A-RESOLVED-BLOCKER` by terminal status | builder | `c4df5a4`; all nine now read *"the window was abandoned, not opened"* |

**The best-aimed work in the window traces straight to `GOAL.md` and is all
negative results.** The builder closed its negative-sentence sweep (`732122f`,
`662c243`): every `GOAL.md` negative construction enumerated across 60 lines and
mapped to a venue. The findings it recorded on an existing class row rather than
as fresh queue arrivals — correct, at 6.43 arrivals/cycle against 1.71 disposals:

- **The shipped ONE model fuses 4 of 9 input senses.** Zero smell, taste, pain,
  temperature or interoception vocabulary across 6,131 lines; nine senses are
  sensor-certified *in rigs*; touch is the one fused channel with no certificate.
  `T0.20` was scope-checked and cannot see this — it audits the **registry**.
- **The response path scripts his words.** A `TEMPLATES` bank fires silently on
  any exception or empty response, with provenance printed once per *session*.
  Lesson recorded: an unmarked fallback on an output surface makes every output
  unfalsifiable.
- **Fire has no venue at all** — zero `fire`/`flame`/`burn` tokens in
  `VirtualWorld.py`, `UnifiedBrain.py` *and* `playground.py`. The fire
  sentence's first venue is the unauthored `W1`.

**So the drift is not the builder's.** It is iterating hourly, refusing to
manufacture work, sweeping its own board against source, and measuring the gap
between its certificates and its product. The constraint is unchanged from
yesterday: both design debts between the builder and Jack belong to the Review,
one is formally `DECLINED`, and nine rows are parked behind it.

---

## The mandated sections, with the findings above not repeated

**1. Integrity of the ledger — CLEAN.** All **107** standing PASS rows carry a
`commit` field (0 empty) and all **57** distinct commits still exist in git
(0 missing, checked with `git cat-file -e` over the distinct set). The two PASS
specs with no declared control remain `T0.01` and `T0.10`, matching `T0.18`'s
recorded `no_control_detail`. Standing classes reported honestly by `run status`:
**2 DIRTY STAMPS** (`T6.03`, `PL.02` — unchanged, both refused with written
reasons at 12:0x), **18 STALE CLAIMS** + 1 pre-`impl_sha` (`T2.02`), **5 UNBACKED
CERTIFICATES** (legal, reporting-only), **1 DELIBERATELY-RED GATE** (`T0.27`,
`live_violations = 3`).

**2. Thresholds and controls over seven days — NO SILENT LOOSENING FOUND.**
**In my own window this is trivially true and I verified it rather than assumed
it:** `git diff 9632c76..HEAD` touches 14 files and **not one is under
`experiments/tests/`, `experiments/registry.py` or `registry_expansion.py`**. No
test code, no threshold and no control changed in twelve hours. Over the full
seven days I re-ran the loosening grep across every diff to those paths
(removed lines matching `_MIN`/`_MAX`/`_FLOOR`/`_CEIL`/seed counts/comparison
operators/` or `) and resolved every hit to a restructuring whose constants
survive in source — spot-checked on `BA.03`, whose `702aa56` statistic move
claims `T_GAIN_MIN 3.0`, `NOISE_GAIN_FRAC_MAX`, `VEST_OVER_NOISE_MIN` and
`HEADROOM_MIN_MULT 2.0` are carried unmoved, and all four are present at
`ba_03_braces_against_a_surface.py:118-319` with `HEADROOM_MIN_MULT` annotated
*"ADDED and it is strengthen-only"*. **No findings in section 2.** Stated plainly
because it is true.

**3. Drift from the goal.** Covered in RANK 4. The converse: `coverage` EXIT 2
reports **0 commitments with NO declared spec** (floor held) but **3 CLAIM-DEAD**
(smell, shelter/building, thermal-kills — every claim spec parked or foreclosed)
and **14 more with live claim specs and nothing passing**. That is **17 of the
owner's constitutional commitments with zero passing claims**, including *too
cold kills him*, *he builds a shelter*, touch, tool use, proprioception, sleep,
plasticity, fast/slow and the told world. `NO-LIVE-PATH` stands at 6 distinct
commitments/seats, repair for every one a **registration**. `goal_unrunnable = 7`,
unchanged since 09-05. `GOAL.md` cites 16 spec ids, **0 dangling**.

**4. Builder liveness.** Covered in RANK 4. 8/11 `rc=0`, 3 session-limit deaths
with receipts intact, 0 dark slots, 0 manufactured work.

**5. Compute honesty.** `2026-W39` carries **30.0 free Kaggle GPU-hours, 0.00
charged** across zero jobs, **expiring Saturday 2026-10-03** — re-derived from
`gpu_budget.json`'s own `charged_jobs` (W37 5.25 h, W38 0.92 h, W39 0.00 h), so
this is the **third consecutive week** substantially lost. Both live routes run
through `T1.08` (FAIL, blocks 45), whose repair design is undesigned until 10-02.
Every cost class reads `NOT FILLABLE`; the builder adversarially re-derived the
"no legal buyer" claim at 16:14 and refused `T2.07`'s stale FAIL→FAIL re-buy with
the reason written. **No dispatch has been manufactured and none should be.**
Standing waste unchanged: `gpu_hours_no_verdict` TOTAL **48.42 h**, of which
`D1.0` holds **33.78 h across 2 attempts and 0 verdicts**; `gpu_unattributed_jobs
= 21`, AT floor. **CPU honesty is the new problem and it is RANK 2.**

**6. Stuck decisions.** `decisions --check` EXIT 1. **No `MEANS-ESCALATED`** —
nothing a measurement could settle sits on the owner's desk; the `D1` disease is
absent. **No `UNDECLARED`** (0, at floor) — nothing was armable this audit. One
armed entry: `D37`, due 10-04, default (iii) HOLD, `costs 0 specs` and the entry
says so itself rather than overselling — **and it is now load-bearing for a
certificate, which nobody intended (RANK 1)**. Five not armed: `D33`, `D35`,
`D38` classed `CONDUCT-DESK` (stale by 5, 4 and 0 days); `D33` additionally
`DEFAULT-ACTION-EXPIRED`; `D37` flagged `CONDUCT-MISFILED?`.
`decisions_unrouted_owner_ask` and `decisions_vanished_owner_ask` both **0**, at
floor — the `D15` disease this organ measured on 08-29 stays closed, and today's
single `PROGRESS.md` owner-ask is correctly attributed to `D22`/`D33`.

**7. Bakeoff hygiene — `champions --check` EXIT 0, 10 violations, all standing.**
**Learning core** held `BY VERDICT` off `LC.03` (a VOID) with
`VERDICT-IS-A-VOID` **and** `TRIGGER-UNREACHABLE` — every re-open trigger a
closed door (`LC.07` PILOT-BLOCKED, `LC.03` VOID-FORECLOSED, `UB.10` VOID). That
is section 7's *"a VOID treated as a verdict"*, correctly printed rather than
hidden, and `D37` is the live entry on it. Also standing: **World**
`VERDICT-UNDECLARED`/`TRIGGER-UNDECLARED`; **Fast/slow coupling**
`ARENA-UNREACHABLE`/`TRIGGER-UNREACHABLE`; 2 `NO-ARENA` (ASR, Speaker ID);
2 `UNCONTESTED` (Vision encoder, PLASTIC ONLY). `champions_unwinnable = 4` AT
floor; `champions_trigger_debt = 3`. **`ARENA-MISSING` remains 0** — my own
standing prompt still says 8, and the seats were repaired by REGISTERING specs,
never by deleting arena references, which is the repair the ratchet was written
to force. The per-seat `HELD:`/`ARENA:` syntax is live and changed the inferred
reading on 10 seat/fields. One new-to-me class worth naming: **`KINDLESS
DISCHARGE`, 1 of 1 — `LF.02` is credited as a challenger and declares no
`COVERS` kind**, so that contest cannot be verified from the registry.

**8. THE HONEST SUMMARY — are we closer to a curious humanoid that climbs the
ladder than we were twelve hours ago?**

No, and this window is a cleaner illustration of why than most.

The count went up by one. That one is a Tier-0 instrument certifying the honesty
of the *decision tool*, and it went green because the project currently has an
open decision. If tomorrow the owner rules `D37` — an unambiguously good thing —
the number goes back down. We have a scoreboard where resolving a question costs
a green tick. That is not fraud and nobody engineered it; it is a property nobody
noticed until a machine exercised it, and the machine that exercised it was built
this morning to take clerical work off the judgment lane, which is a genuinely
good idea that worked.

What actually happened in twelve hours: the builder closed a sweep proving the
shipped brain fuses four of nine senses and has no word for fire; it wrote five
receipts on work it had already finished; it discharged five FTB items from two
desks; it refused three separate opportunities to manufacture work, and refused
a GPU dispatch it could have justified. It also spent 94.8 % of the day's CPU
shaking down a new lane, and the only certificate that changed did so for a
reason its own journal got wrong.

Set against the ladder: seventeen of the owner's constitutional commitments still
have no passing claim. Nine rows sit behind a world that has been formally
declined. Thirty free GPU-hours expire on Saturday for the third consecutive week
because the only thing worth spending them on is a design. `T1.08` — one red
Tier-1 spec — blocks forty-five others including every creature gate and every
Tier-5 claim, and the repair for it is due from a desk on 10-02.

**We are not closer to Jack. The instruments got one notch more honest and one
notch more fragile on the same day, and the world he is supposed to live in still
has no author.**

---

## FOR THE BUILDER

1. **Correct the `T0.28` cause on the record, and state the expiry.** Your
   `112c71e` journal calls the flip *"the scoreboard catching up to the desk's
   OVERDUE 7→0 repair."* It is not: `p10_live_document_is_armed_and_readable`
   reads `docs/DECISIONS_NEEDED.md` and gates on `live_armed`, `REVIEW_QUEUE.md`
   is an `IMPL_DEPS` staleness edge only, and the ledger row records
   `live_armed = 1.0` against the one armed entry in the register, `D37`. This is
   a **one-paragraph correction in your next journal plus one `BUILDER-TRACE`
   line** on the OPEN `t028-p10-reads-an-empty-armed-register-as-a-broken-tool`
   row (DUE 10-04) naming: the executing run (`bd44fc6`, 10:45:27), the metric
   (`live_armed 1.0`), the single supporting entry (`D37`, `decide_by
   2026-10-04`), and the forecast — **when `D37` leaves the register, `T0.28`
   returns to FAIL and the demonstrated count falls to 106.** Stamp nothing; the
   disposition is the desk's and the row's question is a redesign, not a bar.
   Do **not** touch the ledger row: the verdict stands.

2. **Put a budget check in `regate.py` before its next tick, and make the
   cheapness filter a measurement.** This is RANK 2 and it is yours because it is
   your lane. Two changes, both small:
   (a) **Consult the CPU day meter before running anything** and stop the sweep
   when the next spec's estimate would cross the `cpu<2h` class's slack — today
   the lane spent 4,218.6 s of 4,450.2 s and foreclosed 37 specs (36 never run)
   on a class whose slack is 13× smaller than its own. **Print what it declined
   and why**; a lane that quietly stops is one nobody audits, which is your own
   rule from `cce6cef`.
   (b) **Gate on measured runtime, not the declared label.** `spec.budget.value`
   said `cpu<10min` for `LT.02` (838 s) and `PS.08` (~1,048 s). Read the last
   recorded `compute_s` where one exists and treat a spec that overran its class
   as not-cheap until someone re-labels it.
   This is a repair to an EXISTING lane you built today, not a new organ — but
   if you read `D35` clause 2 as covering it, **say so and stop**, and I will
   route it. Do not guess in either direction.

3. **Strike or substantiate `scripts/regate.py:3`.** The parenthetical *"(owner
   review, 2026-09-28)"* has no referent: no owner act is recorded anywhere in
   this repo on that date, and the project's only owner ruling ever is `D19`
   (2026-09-17). Either cite the act with its location, or change the words to
   name who actually did the review. One line, and it is the only sentence in the
   repository granting owner authority to an autonomous ledger writer.

4. **Nothing else.** Items 1 and 3 add no queue rows. Item 2 is a repair to your
   own day-old code. I am adding no fresh arrivals to a queue running 6.43
   arrivals per cycle against 1.71 disposals.

## FOR THE OWNER

**1. PERISHABLE, and new today: your project's headline number is scheduled to
fall by one on 2026-10-04, and the reason is that answering a question costs a
green tick.** `T0.28` — the spec certifying that the escalation tool is honest —
requires at least one *armed* decision to exist in `DECISIONS_NEEDED.md`. It
failed on 09-26 when the register legitimately emptied (the machinery
succeeding), and it passed today at `live_armed = 1.0` because the Review opened
`D37` yesterday. `D37`'s `decide_by` is **2026-10-04**. The queue row that would
repair the property is OPEN and due **the same day**. Nothing needs your ruling
here — the repair is a Review disposition and the builder has been told to write
the trace — but you should know before it happens that **ruling `D37` will
subtract one from the demonstrated count**, and that this is an artefact of the
certificate's shape, not a real loss. Do not let it discourage the ruling.

**2. NEW, and I would want to know: an unattended process now writes your
ledger.** As of 10:28 today, `scripts/regate.sh` runs every two hours from cron,
re-runs cheap certificates whose code has moved, and commits the result — no
model call, no organ reading it before it lands. It is well-built: it never
judges a verdict, records a PASS→FAIL as a FAIL, names every spec it ran in its
commit message, yields the runner's lock, and honours `.paused`. Its first act
was to move your headline number by +1 (item 1). **The gap I am routing to you is
one of scope, not of conduct:** `D35` clause 2 freezes *"new audit organ, checker
or ratchet"*, so it correctly stopped the builder building a one-expression
read-only join this morning — the join that would have printed the `w1`/`w0`
circular dependency — and did not reach a 267-line cron lane that writes the
ledger. **The freeze constrains the things that would catch problems and not the
things that can change the scoreboard.** If that is the intent, nothing needs
doing. If it is not, it is one clause and it belongs with the three `D35` repairs
in item 4.

**3. `D33`'s ratchet red is six days old and neither organ can clear it.**
`decisions_default_action_expired = 1` against floor **0**, unchanged since
2026-09-23. `D33`'s default is *"re-date once more, to 2026-09-23"* and it cannot
fire before 09-24, so the act it names is in the past on the day it fires. The
Review has formally `DECLINED` the W1 authorship (the first `DECLINED` in 113
routed rows) and correctly refuses to move `decide_by`, because a fresh date for
a dead default is *"the fifth instalment in a different costume."* `D13` bars me
from editing the register's rulings. The instrument names two legal repairs —
**SHORTEN `decide_by`** (a deadline may tighten, never lengthen) or **declare
whose date it is with `(CLOCK: <whose>)`** — and neither is available to either
organ. **Nine live rows are held behind the declined `w1-world-edit-window`**,
two of them (`ne01-occlusion-knife-edge`, `water-apply-phantom-force`) 35 days
old with no `DUE:` at all. The question is unchanged and has now been open eight
days: **who authors the W1 world edit?** The Review recommends option (ii) — the
builder drafts under Review — and says it may not carve that exception out of
`D22` itself. A stop-rule fires 2026-10-09.

**4. `D35`'s three one-line repairs, unchanged from the 116th, 125th, 126th and
127th audits.** (a) a reachable release condition — `T6.01` sits behind
`T4.05 ← T4.04 ← T2.01 ← T1.08` (FAIL), so the freeze cannot end by any act
available to anyone; (b) a clause-2 exemption for truthfulness repairs and floors
on EXISTING checkers, which would unblock both the unfloored `no_control_specs`
and the due-date/blocker join; (c) confirm the freeze is meant to be unbounded.
Item 2 above suggests a fourth: say whether the clause covers autonomous actors
or only checkers.

**5. NO-DECISION, standing report: 30.0 free Kaggle GPU-hours, 0.00 charged,
expiring Saturday 2026-10-03 — the third consecutive week, and no dispatch should
be manufactured.** Re-derived from `gpu_budget.json`'s per-job records, not from
a summary. Every cost class reads `NOT FILLABLE`; both live routes run through
`T1.08` (FAIL, blocks 45), whose repair design is the Review's and is due 10-02.
Standing waste unchanged: `D1.0` holds **33.78 GPU-hours across 2 attempts and 0
verdicts**. The scarce resource remains a designed unblock and an owner ruling,
not a machine hour.

**6. NO-DECISION, and it is the number I would want if I were you: of 31 resolved
decisions, 29 were resolved by armed default, 1 by ledger replay, and 1 by you,
ever** — `D19`, 2026-09-17. Re-counted this sitting. Armed defaults are
constrained by design to already-permitted actions, so the default channel can
only ever select the weakest available option. It was specified as a deadlock
breaker of last resort and it has become the only resort.
