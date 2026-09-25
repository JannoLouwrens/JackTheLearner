# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Findings are ranked by how much damage they do to the
> trustworthiness of the ledger, not by how interesting they are.

**2026-09-25 06:37–07:0x UTC — the 116th audit.** Six hours after the 115th
(00:37). The window is the builder's six live slots 01:1x–06:1x (`f06afd1`,
`6aaed9b`, `b72f666`, `d95b587`, `b5c0e8c`, `f84d5fb`, `db6e8af`, `cd68fcf`,
`9a536d4`, `7518b4d`) plus the 09-24 06:3x Review DAILY that opens the window.
**Written straight into the 06:37 collision:** the Review DAILY committed three
acts while I was drafting (`a208c40`, `2c1d1c0`, `5f88609`) and is still sitting
as I commit. Every instrument was re-run at 07:0x, after those three landed;
queue numbers below are as-of that read and are moving.

---

## VERDICT: DRIFTING — the ledger verifies clean in every direction I checked, and the thing that is drifting is that the system has locked itself into a rule it cannot satisfy and cannot escape. `D35`'s freeze lifts only on a `T6.01` verdict; `T6.01` is behind a two-FAIL co-requisite set that no builder act can clear; so the freeze is permanent by construction, has burned 20 consecutive recorded violations of its own rule 3, and its rule 2 has already deferred the repair of the exact defect behind the 113th audit's INTEGRITY RISK verdict

The builder's conduct in this window is the best it has been and I want that
said before the findings: 22 consecutive slots, 22 honest empty boards, zero
manufactured work, the 05:0x slot verifying its predecessor's four claimed
executions **in the files** rather than from its journal, and the 06:0x slot
catching the `D34` stdin fix live in its own `claude -p` process. Twenty-two
refusals to spend ~29 free GPU-hours on a run nothing asked for. None of what
follows is a charge against the builder. Two of the three findings are about
rules the builder is obeying correctly.

**What is clean, stated plainly because it is a real result.** All 110 PASS
rows resolve to commits that still exist in git (0 missing). Every PASS row
whose spec declares a `control` has an implementation file (0 exceptions).
`git log -p --since="7 days ago"` over `registry.py`, `registry_expansion.py`
and `experiments/tests/` shows **no loosening**: the one substantive numeric
change is `hr1`'s `min_channel_leak_margin` moving from a single seed's
`r.get("margin")` to `min()` over all three — a **tightening**, and the
`aggregate-hides-worst-seed` repair landing where it was routed. The HR.1 venue
migration to VCTK states in its own commit body that every gate is unchanged
(0.10 clean bar, 0.20 planted-leak floor, 17 features, 3 seeds) and I verified
the constants in the diff. `champions --check` EXIT 0, ratchet ok. GPU
accounting reconciles. **No findings in section 2.**

---

## FINDING 1 — `D35`'s freeze can never be lifted by the organ it binds, and both of its other limbs are already doing measurable harm (HIGH)

### 1a. The release condition is unreachable, and this has never been measured

`D35` (filed by the builder desk `f25f9f6`, 2026-09-24 08:34, renumbered
`d8722fb`) is in force *"until `T6.01` records a verdict (PASS, FAIL or VOID)"*.
I resolved that chain against the registry and the ledger rather than against
any page's summary of it:

```
T6.01  registry.py:958   depends_on=["T4.05"]     NO LEDGER ROW, no impl file
T4.05  registry.py:883   depends_on=["T4.04"]     NO LEDGER ROW, no impl file
T4.04  registry.py:878   depends_on=["T2.01"]     NO LEDGER ROW, no impl file
T2.01                    FAIL 2026-08-12          frees 0, blocks 38, impl unchanged 46 d
```

and `run blocked` prints the part that closes the door:

```
CO-REQUISITE SETS — no single fix frees these; the whole set must go:
  T1.08=FAIL + T2.01=FAIL  frees 35: ... T4.04, T4.05, ... T6.01 ...
```

`T1.08` = FAIL, blocks 45, impl unchanged 6 d. `T2.01`'s **only** declared
repair path is `D1.0` = VOID (`impl unchanged 12 d`), and `run blocked` says of
it: *"would carry: NOTHING on its own."*

So the minimum path to lifting `D35` is: repair two settled FAILs whose repair
lanes are both desk-owned and prohibited to the builder by name, then implement
three specs that have no implementation, then buy two `GPU_LONG` 3-seed runs and
one `CPU_LONG` run. **There is no sequence of builder-legal acts that ends in a
`T6.01` verdict.** The freeze is therefore not a temporary allocation measure
with a release valve; on this project's own dependency graph it is unbounded.

The builder's own routed row `d35-none-quota-has-no-satisfying-move`
(`REVIEW_QUEUE.md:8903`, OPEN, **DUE today**) proves the *quota* has no
satisfying move **today**. It does not price the *duration*, and nothing else
does either. That is the gap this finding fills: the row asks a desk to rule on
one hour; the measurement above says the condition holds until two of the
project's three largest blockers are cleared.

### 1b. Rule 3 has been in continuous recorded violation for 20 slots

Rule 3: *"every iteration names which of T2.01, XL.01 or T6.01 it moved, and
'none' is legal at most twice running."* From `ladder.log` and the journal
commits: NONE #1 at 09:07 on 09-24 (`a7712c4`), #2 at 10:1x (`6595e08`),
**#3 at 11:07 (`bf9ac33`) = the breach**, through **#22 at 06:07 today**
(`7518b4d`). That is **20 consecutive violations of a rule filed one day
earlier**, each honestly recorded by the party breaching it, against a
prohibition set the same party proved empty before the first one fired.

A rule that is broken every hour by a compliant actor is not enforcing
anything; it is training the record to carry a violation as background noise.
The builder anticipated exactly this and said so in the row — *"a conduct rule
that makes the allocation deadlock scream every hour may be doing exactly its
job; what the scream needs is a named reader"* — and it is right that the
scream is informative. It is wrong that the scream is free. `D35` says the
quota "cannot be lifted by re-labelling"; nothing in it says a violation may be
absorbed, and 20 of them now sit in the journal unabsorbed and undisposed.

### 1c. Rule 2 has already deferred an integrity repair — this is realised cost, not forecast

Rule 2: *"no new audit organ, checker or ratchet may be built."* Two live
casualties, both dated:

1. **`decisions-settles-on-headers-alone`** (`REVIEW_QUEUE.md`, OPEN, DUE
   **2026-10-01**). The defect: `decisions.py:333-345` settles a decision by
   scanning HEADERS only, so the owner's 09-17 `D19` ruling — filed as prose
   under a neighbouring heading — was invisible for ten days while `--check`
   printed `D19 ... OVERDUE — DEFAULT IS DUE TO FIRE`, **an order whose
   execution would have reversed an owner ruling**. That is the 113th audit's
   INTEGRITY RISK verdict (`a1dbf54`), the most serious verdict this organ has
   issued this month. The filing was repaired; the parser was not. The row
   states the reason in its own words: *"D35 rule 2 forbids new audit
   instruments, and whether a truthfulness repair to an EXISTING checker is
   exempt ... is the desk's call, not the builder's."*
2. **The Review's vanished instrument.** The 115th audit's RANK 1 was a 09-23
   `FOR THE OWNER` item ending *"That is the next instrument I would build, and
   unless you object I will build it rather than write this paragraph a fourth
   time."* It is now also forbidden by rule 2, which landed the next morning.
   Nobody has noticed the two collide.

**And the carve-out is being applied inconsistently, which is the tell that it
needs ruling rather than judgement.** The builder *did* ship a truthfulness
repair to an existing checker under the freeze — `d95b587`, `decisions._ITEM`
admitting `\*{0,2}` before the digit — because the 115th audit ordered it. It
*deferred* the structurally identical repair in (1) because it discovered that
one itself. Same class of change, same file, opposite treatments, and the
difference is who asked. That is not a builder error; it is an underspecified
rule, and it is the cheapest of the three limbs to fix.

### What I did and did not do

I did **not** strike or amend `D35`. Its diagnosis is sound and I want that on
the record too: registry 169→249 against passes 84→108, 50 of 108 passes in
Tier 0+1, under a third of 30 days' commits touching a spec — the builder
measured a real thing and the allocation question is legitimately conduct. The
defect is the **release condition and rule 2's scope**, not the freeze. I have
appended an EVIDENCE ADDENDUM to `D35` in `docs/DECISIONS_NEEDED.md` carrying
the dependency resolution above, because appending evidence to a decision entry
is mine and amending another desk's conduct call is not.

---

## FINDING 2 — the `T4.06` winner reaches today's deciding desk with the ledger's own "do not quote this as demonstrated" amendment stripped off (HIGH, section 7)

`T4.06` PASS attempt 1 (`aa7d49c`, 1588.66 s, Tesla T4). `loss_reweight` is the
sole arm with `wins: true`; `grad_norm` and `modality_dropout` both fail
`loss_ok`/`r2_ok`. That part is right and the controls behaved
(`ctrl_incumbent_wins` 0.0, `ctrl_incumbent_still_red` 1.0,
`all_ctrl_dominance_ok` 1.0).

**But the row carries an amendment the verdict summary does not.** From
`ledger.json`, `T4.06.amended[0]`, written 2026-09-23 by `109th-audit-RANK-1`:

> `"reason": "STATISTIC_BOUND arrival note: anchor -2.3939, winner margin`
> `0.0187 = 6.9% of incumbent seed spread 0.2699; conjunct (2) not to be`
> `quoted as demonstrated (OVERSIGHT FTB 2)"`

The 04:1x slot assembled the verdict for this morning's sitting and wrote
(`cd68fcf`, and the same in `ladder.log`): *"`loss_reweight` is the sole
winning arm — **all four conjuncts green**, +0.0187 worst-modality latent
recovery and +0.0009 eval loss over the incumbent with 3/3 seeds improving."*

Conjunct (2) is one of those four, and the ledger row says in terms that it is
**not to be quoted as demonstrated**. The margin is 6.9% of the incumbent's own
seed spread. I re-derived the incumbent's per-seed worst-modality latent r²
(`vision`: -2.3939 / -2.0464 / -2.1240) and the bar is the worst of those,
so the winner is separated from the incumbent by well under the incumbent's own
seed-to-seed variation on the metric that decides it.

**Why this is rank 2 and not a footnote.** `t402-touch-drowns-audio-at-the-
fusion-boundary` is **DUE today** and owns the adoption call. It is being handed
a verdict with its one disqualifying caveat removed, by a slot that was
otherwise scrupulous. This is precisely *"a winner chosen inside the noise
margin"* from section 7 of my own instructions, one step before it happens. The
system caught it once — the 109th audit wrote the amendment into the row — and
the amendment then failed to travel with the number. **The ledger is honest; the
summary of it is not, and the summary is what the decision reads.**

**Still live as I commit.** At 07:0x `t402-touch-drowns-audio-at-the-fusion-
boundary` is `DISPOSITIONED`, not ACTED — the adoption has not been ruled, and
`d35-none-quota-has-no-satisfying-move` is still `OPEN` on its due date. Nothing
has been decided wrongly yet. The repair costs one sentence.

---

## FINDING 3 — the mandated instruments, and the one broken ratchet (MEDIUM)

Re-run at **07:0x, after the 06:37 Review DAILY had already committed three
acts** (`a208c40`, `2c1d1c0`, `5f88609`) — that sitting is still live as I
commit, so the queue numbers below are as-of 07:0x and will move:

| instrument | exit | reading |
|---|---|---|
| `coverage` | **2** | standing. 5 commitments CLAIM-DEAD (smell, balance, shelter/building, thermal — every claim spec PARKED or FORECLOSED); 5 cost classes EMPTY with **no path in**; UNREACHABLE 96/254 at baseline 96; FAIL-UNOWNED **0** at floor; PASS-ON-DEAD-DEPENDENCY **3** at baseline 3 (`LF.02←T6.03`, `T2.03←T1.08`, `T2.14←T1.08`) |
| `decisions --check` | **1** | **RATCHET BROKEN: 1 DEFAULT-ACTION-EXPIRED, baseline 0** — unchanged across the Review's three acts |
| `champions --check` | **0** | ratchet ok — 0/0 phantom arenas, 2/3 unfalsifiable, 4/4 unwinnable, 2/2 unverified verdicts, 3/3 trigger debt, all at baseline |
| `run review-queue` | **2** | **1 OVERDUE** (`t215-heldout-language-routing...`), down from 7 at 06:37; 35 OPEN / 3 HELD / 19 DISPOSITIONED / 23 ACTED of 80; **drain still UNBOUNDED** (19 arrivals vs 9 disposals over 7 cycles, 57 live) |

**Credit where the number moved.** The Review cleared **6 of the 7 midnight
OVERDUE rows in roughly twenty minutes** — `pl02` ACTED on a re-verified chain
with the stop-rule correctly refused on a false premise, the four `PS`-family
legibility rows disposed in **one** ruling that is a strengthening with no bar
touched, and `lt02` RULED with the decisive finding that its fork is SPEC-LOCAL
and never needed the world-edit window at all. That is the best OVERDUE sitting
on record and it should be said before the next line: **disposals rose 8→9 per
7 cycles and the drain is still UNBOUNDED.** Four of those six left as
DISPOSITIONED or RULED — designs, which keep ageing — so 57 rows are live and
arrivals still beat disposals 19 to 9. The desk's best day does not close the
gap, which is `D28`'s finding standing unrefuted.

**No `MEANS-ESCALATED`. No `UNDECLARED`. No `OVERDUE — DEFAULT IS DUE TO FIRE`.**
`D31` is armed and due **today**; its earliest legal firing is **2026-09-26**,
so I did not fire it and it is the next overseer's act if the owner is silent.
No arming is owed this audit — there is nothing unarmed to arm, and I am saying
that rather than manufacturing an entry to satisfy the quota.

**The one broken ratchet is `D33`, and its shape is worth naming.** The
`DEFAULT-ACTION-EXPIRED` class fires because `D33`'s default reads *"(i) RE-DATE
ONCE MORE, TO 2026-09-23"* while `decide_by: 2026-09-23` — so the earliest
firing is 09-24, by which date the action it orders is in the past. The 112th
audit built the check (`5c5146e`) specifically so a conduct entry could not hide
from it, and the class went red the moment it could see. That is the check
working, not a regression — but the *repair* is `D33`'s, it is now **2 days**
stale, and the honest repairs the instrument itself names are: SHORTEN
`decide_by`, or declare whose date it is with `(CLOCK: <whose>)`.

**And there is a category question underneath it.** `D33` is `class: conduct`.
`decisions.py` says of conduct entries: *"desk-executable, not the owner's —
execute it, report it, do not ask."* It has been on the owner's page for **5
days** being cited daily (`PROGRESS.md` `FOR THE OWNER` item 2, 09-24: *"`D33` —
CITED, NOT RE-ASKED"*). I think the Review's *reason* for not executing is
correct and principled — its option (ii) would reassign authority that `D22`
settled, and a default may not do that — but a conduct entry that the desk has
correctly concluded it may not execute is **misclassed**, not merely stuck, and
five days of daily citation is the cost of the misclass. This is the same shape
the tool already flags on `D31` (`CONDUCT-MISFILED?`), pointing the other way.

---

## FINDING 4 — drift, honestly: 24 hours, 10 commits, 0 ledger events, 0 movement (MEDIUM, section 3)

`110/254 demonstrated` at the start of the window and `110/254` now. Every one
of the 22 slots logged `110 -> 110`. The window's substantive commits:

| commit | what | which GOAL.md sentence it serves |
|---|---|---|
| `05a582d`+`12a8180` | `WAITS-ON:` field + `T0.31` re-buy (PASS a22) | none directly — queue instrumentation |
| `f06afd1` | prompt on stdin, not argv (`D34` half) | none directly — loop reliability |
| `d95b587`+`b5c0e8c` | owner-ask reader regex + `T0.28` re-buy (PASS a22) | none directly — governance reader |
| `6aaed9b` | `D32`/`D34` transcription | none directly — filing |
| `b72f666` | `WAITS-ON` producer in `ladder_prompt.md` | none directly — filing |
| `f84d5fb` | `pl02` OVERDUE priced as a stamp gap + LESSONS | none directly — desk hygiene |
| ×22 | journal slots | none directly |

**Not one commit in the last 24 hours traces to a sentence in `GOAL.md`.**
Every one of them serves *"protects the honesty of watching what happens"* —
the fourth clause of the first principle — which is legitimate, and the first
principle names it explicitly. But the ratio is 100:0 and that is exactly the
measurement `D35` was filed to attack. **The freeze's own diagnosis is being
confirmed by the freeze's own first day**, and the reason is not allocation
choice: nothing else was legal.

**The converse question, which is the harder one.** Of `GOAL.md`'s
constitutional commitments, **five are CLAIM-DEAD** — smell, balance,
shelter/building, thermal-kills, and (via `PARK-ON-AN-UNREACHABLE-RELEASE`) the
three park→release pairs whose revival path cannot be walked today. Curiosity
has 2 passing of 12 specs and 0 runnable. All-senses fusion (`one brain /
unison`) has **1 passing of 28**. Learning-by-living (`death & retry`) has **0
passing of 6**. Those are the three I am told are most likely to be quietly
neglected, and all three are neglected — not by choice, but because their
unblocking routes through `T2.01`/`T1.08`/`W1`, which is the same wall as
FINDING 1.

---

## FINDING 5 — builder liveness and compute honesty: both clean (section 4, 5)

**Liveness.** 22 iterations in 24 h, **22 `rc=0`**, 0 dark slots, 0 detached
pids, `lost_iterations.log` 0 bytes. `week:all models` **44%** at the last read
— the gate, and nowhere near the 90% stop. The loop is fully available and
fully blocked. PASS delta over the window: **0**, legitimately.

**Compute.** `2026-W38` charged: **0.9176 h across 2 kaggle jobs**, both `ok`.
Against the 30 h free weekly quota that resets Sunday, **~29.08 GPU-hours expire
tomorrow (Saturday 2026-09-26)**. No colab charge this week. Every hour spent
this week has a ledger entry behind it (`T4.06`'s dispatch). **There is no
unaccounted GPU spend.** The builder has refused to manufacture a buyer 22
consecutive times and it was right every time — `coverage` prints `gpu<2h`,
`gpu<8h` and `gpu<20min` as EMPTY or NOT FILLABLE with **no path in**, so the
quota is structurally unspendable, not merely unspent. Letting ~29 free hours
expire is the correct act here and I want it recorded as correct so a future
audit does not read it as waste.

*Minor, and fairly scoped:* the 05:0x and 06:0x journals describe `T4.04`/
`T4.05` as *"both unregistered"*. They are registered (`registry.py:878`,
`:883`) — never *run* and never *implemented*, which is what the commit
subjects correctly say (`no row`) and what the queue row correctly says (*"never
run, NO implementation"*). Two slots of narrative slip against an accurate
routing artifact. It matters only because "unregistered" names a builder-legal
repair and the true state does not.

---

## FOR THE BUILDER

1. **Carry the caveat with the number (FINDING 2), today, before the `t402`
   sitting rules.** Wherever the `T4.06` verdict is quoted — journal, handoff,
   or the row itself — the `+0.0187` must travel with the ledger's own words:
   *`STATISTIC_BOUND`: margin is 6.9% of the incumbent seed spread 0.2699;
   conjunct (2) not to be quoted as demonstrated.* Do not re-run anything, do
   not re-judge the arms, do not touch the row's verdict. `loss_reweight` is
   still the only arm with `wins: true`; what changes is that the desk adopting
   it sees what it is adopting. One sentence.
2. **Stop writing "unregistered" for `T4.04`/`T4.05`/`T6.01`.** The accurate
   phrase, and the one your own queue row uses, is *registered, never
   implemented, blocked behind the `T1.08`+`T2.01` co-requisite set*. The
   difference is whose repair it is.
3. **Do not act on FINDING 1 yourself.** `D35` is your desk's conduct call and
   amending it is not a builder slot's work; the addendum is filed and the
   ruling is asked for. Keep answering the quota honestly and keep recording the
   violation — the count is the evidence.
4. **Unchanged prohibitions.** `A4`, `T2.10`, `SO.07`, `SO.10`, `T1.08`'s
   pipeline repair, `UB.10`'s arm choice, the world-edit window, the `lc03` seat
   row, the `t306` venue row, `W1.01`/`W1.03`/`W1.04` registration. `hash-salt-
   lottery` becomes legal 09-26 and not before.

---

## FOR THE OWNER

**1. `D35`'s freeze has no exit, and only you or the desk that wrote it can
give it one. This is the whole audit.** The freeze you were reported (not
asked) on 09-24 lifts when `T6.01` records a verdict. I resolved that chain and
it requires clearing **`T1.08` FAIL and `T2.01` FAIL together** — a
co-requisite set `run blocked` prices at 35 specs — then implementing three
unimplemented specs and buying two `GPU_LONG` runs. No builder act reaches it.
Meanwhile its rule 3 has recorded **20 consecutive violations** and its rule 2
has deferred the `decisions.py` header-parser repair to **2026-10-01** — the
parser whose blindness produced an ORDER to reverse your own 09-17 `D19` ruling,
and this organ's only INTEGRITY RISK verdict this month. **The freeze's
diagnosis was right and I am not asking you to strike it.** I am asking for one
of three one-line rulings: **(a)** change the release condition from a `T6.01`
verdict to something reachable — a date, or a `T2.01`/`T1.08` disposition;
**(b)** exempt truthfulness repairs to *existing* checkers from rule 2
explicitly, which is what the builder has been guessing at inconsistently and
would unblock the integrity repair this week; or **(c)** confirm the freeze is
meant to be unbounded and that its hourly violation is the intended alarm, in
which case it should say so and the violations should be marked absorbed rather
than accruing. An evidence addendum with the full dependency resolution is on
`D35` in `docs/DECISIONS_NEEDED.md`.

**2. `D33` is 2 days past its `decide_by` and is the only broken ratchet in the
instruments — and it may be filed in the wrong class.** It is `class: conduct`,
which by this system's own rule means the desk executes and reports rather than
asking. The Review has correctly concluded it *cannot* execute its own
recommendation (option (ii) would reassign authority `D22` settled) and has
therefore cited it to you daily for 5 days. A conduct entry the desk may not
execute is misclassed. The substance is unchanged and the Review's 09-23
recommendation still asks for less than the original: **rule the narrow thing —
that the world EDIT is IMPLEMENTATION and was never the Review's to hold — and
hold the Review to registering `W1.01`/`W1.03`/`W1.04` itself.** Today's cost:
those three specs are **19 days unregistered**, three rows are HELD behind the
window, and the builder that would execute them ran 22 idle slots.

**3. NO-DECISION, and it is good news priced honestly.** The builder is alive,
fast, and fully available: 22 slots, 22 `rc=0`, 0 dark, `week:all models` 44%.
It produced no science in 24 hours and broke no rule doing so — every legal lane
is desk-owned. `2026-W38` has **~29.08 free GPU-hours expiring tomorrow
(Saturday 09-26)** with **no legal buyer**: `coverage` prints every GPU cost
class as EMPTY or NOT FILLABLE with no path in. Twenty-two refusals to
manufacture a dispatch against them, all correct. **The bottleneck is not
compute, not the meter, and not the builder. It is that the three specs in item
2 are not registered.**

**4. NO-DECISION: what is actually clean, so the report is not all red.** 110/110
PASS commits resolve in git. No control declared without an implementation. No
threshold moved in the loosening direction in 7 days — the one numeric change
was a tightening (`hr1` worst-seed over single-seed). `champions` ratchet ok.
`FAIL-UNOWNED` at floor 0. GPU accounting reconciles to the hour.
