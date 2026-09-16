# OVERSIGHT.md — the overseer's current-state report

> Rewritten every audit. Current state, not a log. The running history is the
> git log of this file.
>
> **98th audit — 2026-09-16, 06:37–07:2x UTC.** Opened at HEAD `7ec0c7d`; the
> Review's daily sitting ran concurrently in pid 914716 and committed six times
> underneath me. **Two of my readings moved mid-audit and I have corrected them
> rather than publishing the first one I took** — the queue went 4 OVERDUE → 0
> (RANK 6) and the Review's own repair surfaced a hole in `coverage.py` (RANK 5).
> **The builder has now been dark for 42 consecutive slots.**
> Last productive iteration ended `rc=0` at **2026-09-14T11:14:44** — **43 h 35 m
> ago**. `demonstrated` **108 → 108**, flat since 2026-09-14T04:18 (50.5 h).
> **Zero builder commits in the window.** Every commit since the 97th audit is a
> desk commit.

## VERDICT: INTEGRITY RISK

**Scope it precisely, because the phrase is expensive: the capability ledger is
clean and I checked it independently this morning.** 108 PASS rows, all 108
resolve to a live spec in `BY_ID`, all 57 distinct commits resolve in git, one
dirty row already classified by an existing instrument, zero threshold movement
in the loosening direction across seven days. Nothing on the scoreboard is a lie.

The integrity risk is in the **decision register**. `docs/DECISIONS_NEEDED.md`
contains **two different owner decisions both numbered `D30`**, written into the
file in one commit. `experiments/decisions.py` parses declarations into a dict
keyed by id (`decisions.py:381`, `decls[did] = d`) with **no uniqueness check
anywhere**, so the second block overwrites the first. `decisions --check` prints
**one** `D30`, exits **0**, and reports **`ratchet ok`** — over a file holding an
armed decision it never read.

The shadowed entry is not a minor one. It is the Review's blackout escalation,
`decide_by` **2026-09-18**, whose own `blocks:` field reads: *"no spec id
directly. What it blocks is EVERY spec, because it blocks the organ that runs
them."* It will never print `OVERDUE — DEFAULT IS DUE TO FIRE`, because the desk
that fires defaults cannot see it.

That is `D1`'s disease with a new aetiology. `D1` was visible and nobody could
act. This one is invisible and everybody reports clean.

---

## THE FOUR INSTRUMENTS

| instrument | exit | reading |
|---|---|---|
| `coverage` | 2 | **0 commitments with NO declared spec** — the exit is the claim-dead/park ratchets, not an uncovered commitment. **Read that number with RANK 5: it is 0 against a hand-maintained list that does not name four of `GOAL.md`'s own primitives.** `claim_dead = 4`, unchanged **13 days**. |
| `decisions --check` | 0 | `ratchet ok (0/10 undeclared, 0/3 unrouted-owner-ask, 0/0 vanished, 0/0 default-action-expired, 0/0 firing-diff)` — **and it is wrong**, see RANK 1. `D19` prints `OVERDUE — DEFAULT IS DUE TO FIRE` for the second consecutive audit. |
| `champions --check` | 0 | 0 phantom arenas, 2/3 unfalsifiable, 2+1/4 uncontestable, 4/4 unwinnable, 2/2 unverified verdicts, 3/3 trigger debt — every class AT its declared count, none moved. |
| `run review-queue` | **0** | **4 OVERDUE at 06:4x when I read it; 0 by 07:0x.** The Review ACTED on all four during this audit, each with a named executing commit. See RANK 6. |

No `MEANS-ESCALATED`. No `ARENA-MISSING`. No `NO-ARENA` regression. No
`UNROUTED-OWNER-ASK` or `VANISHED-OWNER-ASK` against `PROGRESS.md` — I read its
`FOR THE BUILDER` and `FOR THE OWNER` sections directly as well as through the
tool, and the tool and my eyes agree.

---

## RANK 1 — Two owner decisions share the id `D30`. One of them is invisible to the instrument built so that no decision can go dark.

**The evidence, from the file and from the parser, not from anyone's report.**

```
$ grep -n '^## D30' docs/DECISIONS_NEEDED.md
6803:## D30 — The builder has been dark for 18 consecutive hourly slots on a meter
      three-quarters of which this project did not spend, and 26.51 free GPU-hours
      expire on Saturday with nobody awake to dispatch them. (2026-09-15, Review DAILY)
6937:## D30 — The colab GPU lane has no ceiling, no overrun mark and no refusal:
      `remaining()` returns infinity for it, and two 1-GPU-h retrieval failures in
      one morning could have been ten. (2026-09-15, overseer, 97th audit)

$ grep '^DECIDE: ' docs/DECISIONS_NEEDED.md | sort | uniq -c | sort -rn | head -1
      2 D30                        # 28 DECIDE blocks, 27 distinct ids

$ grep -n decide_by docs/DECISIONS_NEEDED.md | tail -2
6903:  decide_by: 2026-09-18       # the blackout entry
7038:  decide_by: 2026-09-25       # the colab entry

$ $PY -m experiments.decisions --check | grep -E '^\s+D30'
    D30    costs   0 specs   due 2026-09-25
```

**The mechanism is three lines of `decisions.py` and it is not a race.**
`parse()` walks every `DECIDE:` block in document order and assigns
`decls[did] = d` (`decisions.py:360-381`). Last write wins; there is no
`if did in decls` anywhere in the module. The header scan one function below
*does* collect duplicates correctly (`headers.setdefault(key, []).append(title)`,
line 391) — so the file's own parser holds the evidence of the collision in one
dict and discards it in the other.

**Both blocks were added by a single author in a single commit**, `1466035`
(Review DAILY, 2026-09-15). This is not concurrent-write damage that a lock would
have prevented. The Review read the 97th audit's `FOR THE OWNER` item 3 — which
had explicitly declined to append the colab finding itself — routed it correctly,
routed its own blackout finding correctly, and allocated "the next free id"
**twice**. Good behaviour, no uniqueness check, silent deletion.

**What is lost, field by field.** Everything in the blackout entry's `DECIDE`
block: its `class: goal`, its `blocks:` text, its `default: (v) REPORT THE
STREAK, GATE NOTHING, RELAX NOTHING`, and its `decide_by: 2026-09-18`. Which
means:

- It can never go `OVERDUE`. Its default can never fire. The deadlock-breaking
  machinery `SYSTEM.md` spends four paragraphs on does not apply to it.
- Its default is never safety-checked. `safety_hazards`, `SAFETY-CLAIM-DEAD`
  and `firing_diff_hazards` all operate on `decls` — they run on the colab text
  and have never seen the blackout text. `SYSTEM.md` records at length that a
  default enforced by nobody while a governing document says it is enforced is
  *"author self-certification, the precise thing this file's first law exists to
  distrust."* That sentence is true again today, of a different organ.
- **The owner is given two different dates for one id.** `PROGRESS.md` `FOR THE
  OWNER` item 1 tells the owner `D30` is `decide_by 2026-09-18`. A reader who
  runs the tool is told `D30 ... due 2026-09-25`. Neither number is wrong; the
  register is.
- The tool lists **6** open decisions. The file holds **7** distinct open armed
  entries. Nothing prints the discrepancy.

**This is the first collision in the register's history** — D1 through D29 are
each unique, and 27 of 28 blocks are singletons. It shipped at floor, which is
the cheap moment to catch it.

**AND IT HAS ALREADY CORRUPTED THE RECORD, INSIDE ONE DAY.** This is not a
forecast of harm; it is on the page the Review published at 07:2x this morning,
while I was writing:

> 2026-09-15 `PROGRESS.md`: *"`D30` — NEW, routed this morning (`1466035`),
>   **`decide_by` 2026-09-18**, and it is the only item on this page that you
>   alone can settle."*
> 2026-09-16 `PROGRESS.md`: *"`D30` — cited, not re-asked (**`decide_by`
>   2026-09-25**). The builder is now dark 41 consecutive slots..."*

**Same desk, same entry, same subject — and the date has moved a week**, because
the desk that authored `decide_by: 2026-09-18` read it back out of a tool that
can only show it the other `D30`. Nobody extended a deadline; nobody edited a
date. The register did it, silently, and the organ it happened to is the one that
wrote the entry. That is a deadline drifting by seven days through a parsing
artefact, and it is exactly what `SYSTEM.md` means when it says *"a deadline that
moves when it is reached is the deadlock it replaced"* — except this one moved
without anyone reaching it or deciding it.

**The repair, stated so it is not mistaken for a tidy-up.** RENUMBER the
**colab** entry to `D31` and leave the blackout entry as `D30`, because `D30` is
the id the owner was given on `PROGRESS.md` and an id the owner has been told is
not the desk's to reassign. Then add a duplicate-id check to `decisions.py` that
fails **loud** — a register keyed by a hand-typed id must assert its own
uniqueness, and the count it reports must be the count of blocks it read, not
the size of the dict it built. **Deleting either entry is not one of the
repairs**, and neither is quietly letting the colab one keep the id.

I have appended a `DUPLICATE-ID NOTICE` to `docs/DECISIONS_NEEDED.md` recording
the collision with this evidence, deliberately **without** a `DECIDE:` block —
adding one would deepen the hole it documents. Renumbering is an edit to an
existing entry and is not mine to make; it is `FOR THE BUILDER` 1 and, because
the builder is dark and the Review is not, it is also on the Review's desk.

---

## RANK 2 — The builder is 43.5 hours dark, and the honest statement about its return is a BOUND that lands on the day the GPU quota dies

**Measured, not modelled.**

```
$ scripts/claude_usage.py
session          [#                   ]   9%  resets Sep 16, 10:50am (UTC)
week:Fable       [################### ]  95%  resets Sep 21,  4:59am (UTC)
week:all models  [##############      ]  73%  resets Sep 21,  4:59am (UTC)

$ scripts/usage_attribution.py --line
of this week's 73 shared point(s): builder 7 (9%), desks 2 (2%), both 0 (0%),
NOT THIS PROJECT 64 (87%); 42 consecutive dark slot(s)
```

Twenty-four hours ago the same two lines read `37%` and `NOT THIS PROJECT 28 of
37 (75%)`. **The meter rose 36 points; this project's own attributed share rose
by zero**, because the organ that would spend it was switched off the whole time.
The not-this-project share went 75% → **87%**.

**The bound, in the form `docs/LESSONS.md` §"DO NOT MODEL THE METER" requires —
`at meter M, release cannot come before the first hour where allow > M`:**

    allow(e) = 25 + ceil(0.65 * e)          e = integer % of week elapsed
    now:  e = 29,  allow = 44,  usage = 73          gap 29 points
    release needs allow >= 74  ->  e >= 74  ->  2026-09-19 09:50 UTC

**If the shared meter never rises again, the builder's first unpaced slot is
Saturday 2026-09-19 at 10:07 UTC.** That is a floor, not a forecast: every
further point of meter rise pushes it back ~2.6 h, and 87% of the points are
drawn by hands this repository cannot see. Total blackout at that floor:
**~118 hours**.

**And the floor collides with the perishable resource.** Read from the code, not
from a page:

```
$ python -c "from experiments.gpu import Budget; b=Budget(); ..."
week: 2026-W37     used kaggle 1.379     remaining kaggle  28.621
                   used colab  2.1109    remaining colab   inf
                   afford('colab', 999h) -> True
```

`gpu.py` weeks are `%U` (Sunday-start), so **W37 expires at the end of Saturday
2026-09-19**. The builder's earliest possible wake-up is **09:50 that same
morning — roughly 14 hours before 28.62 free GPU-hours expire**, and only on the
assumption that nobody touches the meter for three days. The precedent is in
`lib_usage.sh`'s own header: W32 lost 8.82 h, W33 lost 22.11 h, *"with no agent
awake to dispatch them — on a project whose owner has ruled free compute only."*
**W37 is on course to be the worst of the three by a factor of three.**

*One measurement note, because the whole `D30` case is costed on this number:*
`PROGRESS.md` publishes **26.51** free hours; `Budget.remaining("kaggle")`
returns **28.621**. The difference is the 2.11 colab hours, which the desk
charges against the 30 h ration by hand and which `remaining()` does not
(`gpu.py:427` subtracts `used_hours("kaggle")` only). The desk's figure is the
conservative one. Two numbers for one quantity, and the larger one is the true
loss.

**This is not a gate malfunction and I want to be as precise about that as the
97th audit was.** `allow` is arithmetic with zero variance; `pace_gate` is
checked only after `usage_gate` has said yes; the 90% stop is the owner's own
rule. The gate is doing exactly what it was built to do. The finding is that the
thing it was built to ration — this project's share of a pool — is now 87%
somebody else's, and the rationing organ is the only one in the building wearing
a gate.

---

## RANK 3 — `D19`'s armed default has now been due to fire for 31 hours and has not fired

`decisions --check` prints `D19 costs 3 specs OVERDUE — DEFAULT IS DUE TO FIRE`
for the **second consecutive audit**. `decide_by` was **2026-09-14**; the default
became firable at **2026-09-15 00:00**. It is now 2026-09-16 06:5x. Thirty-one
hourly slots have passed since it came due; every one of them was a `PACING`
skip. It holds `HR.1`, which `run blocked` ranks as `frees 3`.

Everything the 97th audit said about this is still true and is now 31 hours
truer, so I will not re-argue it — I will add the one thing that changed.
Yesterday this was "a deadline with an unadvertised dependency on a spend meter."
Today the meter's own arithmetic says that dependency does not clear before
**Saturday**. So `D19` is not late by hours any more; on the bound above it
is late by **five days**, unless something fires it off the skip path.

`ladder_loop.sh:112` already carries the governing sentence — *"A PACE SKIP
DEFERS CLAUDE SPEND — IT MUST NOT DEFER WORK THAT COSTS NONE"* — and implements
it for exactly one class of zero-cost work (`harvest_bookkeeping`). Firing an
overdue default is one commit and zero meter. **The deadline has not been
extended and must not be**; a deadline that moves when it is reached is the
deadlock it replaced.

---

## RANK 4 — My own organ's release forecast was falsified inside two hours, and `LESSONS.md` had already forbidden making it

The 97th audit published, in `RANK 2` and again in `FOR THE OWNER` item 1:

> *"The builder releases at roughly **13:00–14:00 UTC today** — call it a
> **26-hour blackout**."*

**The arithmetic was right and the assumption was unstated and wrong.** At usage
37 the line does clear at e ≈ 19%, i.e. 12:55 UTC on 09-15. What the sentence
required and never declared was that the meter stay frozen at 37. At 09:07 that
morning it read **60**. The forecast was dead four hours before its own window
opened, and the blackout it called 26 hours is 43.5 and counting.

`docs/LESSONS.md` §*"DO NOT MODEL THE METER — AND DO NOT MODEL THE LINE EITHER:
COMPUTE IT"* (2026-08-31) already says this, in terms:

> *"Eight forecasts of 'when the gate opens' were published by four organs
> between 08-26 and 08-28; the three that came due were all wrong, all
> optimistic, and **not one would have changed a single action.** ... The only
> honest statement joining the two is a BOUND, not a forecast."*

Yesterday's was the ninth, by a fifth organ, and it was optimistic like the
others. It changed no action either — which is the lesson's own point and is why
this is `RANK 4` and not higher. I am recording it because that lesson's
*meta*-clause is precisely about a rule that lives where the organ breaking it
does not read, and the organ that broke it this time is the one whose job is to
catch exactly this. RANK 2 above is written as a bound for that reason.

---

## RANK 5 — `coverage`'s "0 commitments with NO declared spec" is 0 against a hand-maintained list that does not name four of `GOAL.md`'s own primitives — and another organ found it, not this one

**This one is against my own highest-priority instrument and it was surfaced by
the Review's sitting this morning** (`81fdaba`, `told-world-has-no-rung` ANSWER
(a)), not by me. I am ranking it here rather than burying it in a footnote,
because `coverage`'s top line is the first number this organ reports every audit
and the standing instruction above it is unambiguous: *"If GOAL.md gains a
commitment that `coverage.py` cannot name, add it to COMMITMENTS in the same
commit — a coverage tool that silently stops covering something is worse than
none."*

`GOAL.md:186-187`, the owner's own sentence:

> *"Survival earns him the primitives that make anything else mean something —
> **hot, heavy, far, tiring, dangerous, worth-it, that-person-lied**."*

`COMMITMENTS` holds 25 entries. Mapping the seven against them:

| primitive | commitment entry |
|---|---|
| hot | `thermal (kills)` |
| tiring | `hunger/thirst` (interoception; fatigue is named inside it) |
| dangerous | `damage/nociception` |
| that-person-lied | `social/other agents` |
| **heavy** | **none** |
| **far** | **none** |
| **worth-it** | **none** |

Three have no entry outright; the Review counts four, reading `tiring` as
uncovered too, and I am not going to argue the fourth — the finding does not turn
on it. **Weight, distance and value-of-an-outcome are the three the ladder cannot
see**, and they are not decorative: `heavy` and `far` are the primitives a body
learns from acting in a world, and `worth-it` is the one every needs-vs-curiosity
bakeoff in this project silently assumes he can form.

**Why this is the same disease as the 2026-08-10 miss and not a smaller one.**
That miss was four owner commitments with zero falsifiable claims behind them,
and the lesson recorded was that *a missing spec has no id, blocks nothing, fails
no gate, and is invisible to every instrument this system owns.* `coverage.py`
was built to be the one instrument that could see it. **But `coverage.py` sees
exactly the commitments a human typed into `COMMITMENTS`** — so a `GOAL.md`
sentence that was never transcribed is invisible to the tool built to find
invisible things, and the tool reports `0` with full confidence. It is RANK 1's
shape in a different register: an organ that can see everything it parsed, and
cannot see whether it parsed the whole source.

The repair is to add the missing primitives to `COMMITMENTS` — code, and `D13`
records that the overseer may not edit its own script, so it is `FOR THE BUILDER`
2 and the Review has already opened a dated row for it
(`goal-187-names-seven-primitives-four-have-no-commitment`, DUE 2026-09-18).
**Adding a commitment will make `coverage` exit worse, not better, and that is
correct** — this ratchet may grow when the source grows; what it may never do is
shrink by trimming the list.

---

## RANK 6 — The four OVERDUE queue rows were ACTED during this audit, with commits, and the desk had named all four as its own bill before they broke

Recorded as a **result**, not a complaint, and corrected mid-audit: I read
`run review-queue` at 06:4x and got **EXIT 2, 4 OVERDUE** (all promised
2026-09-15). By 07:0x it reads **EXIT 0, 0 violations**. In between, the Review's
concurrent daily sitting disposed of every one of them — and disposed of them the
expensive way:

| row | disposition | executing commit |
|---|---|---|
| `told-world-has-no-rung` | **ACTED**, (a) answered **NO** | `81fdaba` |
| `reparenting-the-welded-fifteen` | **ACTED** — no re-parent is owed | `34116ca` |
| `goal-cites-four-specs-that-resolve-to-corpses` | **ACTED**, bundled as required | `34116ca` |
| `w100-honest-null-does-not-rescue-pile-a` | **ACTED** — W1 ordering now unconditional | `d521384` |

Not re-dated. Not relabelled `HELD`. Not stamped `ACTED` with no commit — and a
follow-up commit (`36d120c`) went back and stamped the executing commits onto the
rows explicitly. `PROGRESS.md` Part 2.5 §5, written *before* midnight, had said:
*"The four rows still standing on today's date are this desk's decision debt and
they stay there. If they break at midnight the break is mine and it will be
reported as mine."* They broke, they were exactly those four, and the desk paid
them off the next morning with answers rather than dates.

`told-world-has-no-rung`'s answer is where RANK 5 came from, which is worth
saying plainly: **the desk that was four promises overdue spent its repair
finding a hole in the overseer's instrument.** Two structural readings still
stand and are `D28`'s (due 09-21, amendment recommended by the Review, agreed):
`DUE-DATE PILE` has 09-17 and 09-20 each carrying 6 against a measured one-cycle
capacity of 6, and the drain still reads **UNBOUNDED** — 23 settled FAILs whose
only repair owner is a row on this desk.

---

## RANK 7 — Carried forward unrepaired, because the organ that owns them has not been awake

Not re-argued; listed so nothing quietly ages out of the report.

- **`/data/t108_backend_probe.json` is still unread — mtime 2026-09-14 11:22,
  now 43 h 30 m old.** Its `failures[1].stdout_head` still reads
  `JACK_OUT /content`, still settles cause-2, still has three of the colab arm's
  five per-seed `heldout` values in the tail. 97th `FOR THE BUILDER` 1.
- **`declared_pids` is still not pruned, and has grown.** `405151` has now been
  declared live for **44 hours** and dead for 43 (`kill -0` → no such process).
  Yesterday's `638639`/`638640` desk slots were never reaped either, so the file
  now declares **three** processes that do not exist. 97th `FOR THE BUILDER` 4.
- **Colab is still an unmetered lane.** `remaining("colab")` returns `inf`,
  `afford("colab", 999)` returns `True`, `"overruns": []`. 97th `FOR THE BUILDER`
  5 and the (shadowed) colab `D30`.
- **The probe's 2.6716 GPU-h against a 1.20 h authorisation is still invisible to
  `gpu_hours_no_verdict`**, which still reads `T1.08: 0.36 h / 1 attempt / 1
  verdict`. 97th `FOR THE BUILDER` 6.

---

## THE SECTIONS WHERE I FOUND NOTHING, AND THAT IS THE RESULT

**1. Integrity of the ledger — CLEAN, verified independently this morning.**
148 entries, **108 PASS / 25 FAIL / 14 VOID / 1 BLOCKED**. All 108 PASS ids
resolve in `BY_ID` (0 orphans). 57 distinct commits, **all 57 resolve in git**
(`git cat-file -e`). One dirty stamp — `T0.23 @ b4f123d+dirty` — and it is the
same one `dirty_recoverability` already classifies COMMITTED at `run.py:1988`,
with the class asserted against constructed probes so the reporting path cannot
silently stop working. Two PASS rows carry no `control_metrics` (`T0.01`,
`T0.10`) and both are the declared by-decision exceptions `coverage` names. No
change since yesterday and no new hole.

**2. Thresholds and controls over time — CLEAN, and I re-derived it rather than
inheriting it.** No commit has touched `registry.py`, `registry_expansion.py` or
`experiments/tests/` since `300480d` (2026-09-14 ~11:0x) — the builder has been
dark, so the seven-day window is the 97th audit's window minus nothing. Scanning
it fresh: every numeric movement is a **tightening** — `MIN_DISTRACTOR_EVAL`
30 → **59** in three ME specs, `N_DISTRACTOR` 60 → **130**, `N_PROPERTIES`
15 → 16 → 17 → 18 → 19. The two deleted `falsified_by=` lines I flagged on the
first pass are **strengthenings**: `T1.07` and `T1.08` each keep the old text and
**add** a conjunct (`"— or the held-out metric's own seed CV exceeds 7%"`), each
declares itself strengthen-only in the diff, and each deliberately stales its own
certificate and demands a re-run rather than grandfathering the old PASS.
`MAX_HELDOUT_CV_PCT` 7.0 is unmoved and is *imported* by the probe rather than
copied. No `_check` gained an `or`. No control was deleted. No seed count was
reduced. **Nothing to report in section 2.**

**7. Bakeoff hygiene — CLEAN, unchanged.** `docs/DECISIONS_RESOLVED.md` has not
been touched since 2026-09-14. The standing exemplar is unchanged and still
worth naming: `SO.10` produced a 5.79σ winner, its *second* pre-registered gate
disqualified that same winner (divergence negative on all three seeds against
`MIN_MIGRATE` 0.40), the spec recorded **FAIL**, and the Person-model seat stayed
**VACANT** rather than being handed to the best eligible arm after the fact. No
VOID is being read as a verdict; no winner sits inside its noise margin.

---

## 3. Drift from the goal

**What the builder worked on in the last day: nothing. There is no drift to
assess because there was no work.** All eight commits since the 97th audit are
desk commits — the overseer's report, the Review's page and log row, one steering
block, two queue re-datings, one ruling addendum, one ratchet recording. Each
traces to *"protects the honesty of watching what happens when the three meet"*,
which is a real GOAL.md clause and is also the only clause the project has served
for two days. **Zero commits in 43 hours served the brain, the body or the
world.**

**The converse question, which is where the damage is compounding.**
`claim_dead = 4, at 2026-09-03` — now **thirteen days**. The four are **smell**,
**balance**, **thermal (kills)** and **shelter/building**, and `coverage`
annotates two of them in the owner's own framing: *"owner named it
constitutional"*, *"owner's own image of success"*. Against GOAL.md verbatim:

- *"too cold kills him, too hot kills him"* → thermal, claim-dead.
- *"Cold nights teach shelter-building the way no scripted lesson can"* →
  shelter, claim-dead.
- *"EVERY SENSE A HUMAN HAS... Smell and taste are not ornaments"* → smell and
  balance, claim-dead.

Every park was legal and evidence-backed and no `PARKED` marker should be
touched. Three of the four successors (`SH.02`, `SM.03`, `BA.03`) are redesigns
owed by the Review desk that cleared four broken promises this morning and
still reads drain UNBOUNDED, and
`PARK-ON-AN-UNREACHABLE-RELEASE` stands at **3** — parks naming a revival path
nobody can walk today.

The two claims GOAL.md leans on hardest remain the thinnest on the board:
**curiosity 12 specs / 2 passing**, **one brain-unison 27 specs / 1 passing**.
`UNREACHABLE` is 97 of 249 (39%), at floor.

---

## 4. Is the builder alive and productive?

**Alive, correctly gated, and stopped for the second consecutive audit.**

| | reading |
|---|---|
| iterations in the last 24 h | **0** |
| `rc=0` in the last 24 h | 0 of 0 |
| PASS delta, 24 h | **0** (108 → 108, flat 50.5 h) |
| consecutive dark slots | **42** (instrument); wall clock 43 h 35 m since `2026-09-14T11:14:44` |
| commits by the builder, 24 h | **0** |
| `ladder.log` last write | 2026-09-16T06:07 — a `PACING` line |

In the 24 h *before* the blackout: 7 iterations, 6 `rc=0`, one `rc=1` (a 529
Overloaded, transient, server-side). Nothing about the loop itself is broken: the
process fires on cadence, reads both meters, prints its own dark-slot streak, and
exits without spending. **This is a gate working correctly on a pool that is 87%
not ours.**

*One forward-looking note, so the next reader is not surprised.* `week:Fable` is
at **95%** and `MODEL_FLOOR` is **95**, so `model_gate` will **refuse Fable** on
the builder's next unpaced slot and walk the chain to Opus — designed behaviour
with 57 precedents in this log, most recently 09-14 02:07–04:07 where it ran on
Opus and ended `rc=0`. It has not been *printed* during the blackout only because
`pace_gate` (line 183) exits before the model chain is built (line 272). Worth
knowing: when the builder does return it will be spending Opus against the same
`all models` pool that is starving it.

---

## 5. Compute honesty

**W37 (`%U`, Sunday-start; expires end of Saturday 2026-09-19):**

| lane | charged | jobs | ceiling | `remaining()` |
|---|---|---|---|---|
| kaggle | 1.379 h | 3 | 30.0 | **28.621 h** |
| colab | 2.1109 h | 2 | **none** | **inf** |

`"overruns": []`. Nothing is red, and the three code reasons are unchanged from
the 97th audit (overrun mark gated on `backend == "kaggle"`; `remaining()`
returns `inf` for every non-kaggle backend; a PROBE buys no ledger row so its
hours can never join `gpu_hours_no_verdict`). `run status` still prints
`T1.08: 0.36 h / 1 attempt(s) / 1 verdict(s)` for a probe that charged 2.6716 h.
`gpu_hours_no_verdict` TOTAL stands at **48.42 h**, with `D1.0` at 33.78 h across
2 attempts for **0 verdicts** and 6.32 h across 21 `UNATTRIBUTED` jobs.

**The honesty finding this section owes is the perishable one and it is in
RANK 2:** 28.62 free hours, an authorised buyer waiting (the T1.08 colab repair,
~1.05 h, authorised at §9d of the 09-15 ruling), and a release bound of Saturday
09:50 against a Saturday-night expiry. No GPU-hours were spent in this window
because nothing was awake to spend them — which is a different failure from
spending them badly, and worse.

---

## 6. Stuck decisions

Seven distinct open armed entries; the desk can see six. `D19` **OVERDUE, default
due to fire, 31 h unfired** (RANK 3). `D30`-blackout **invisible** (RANK 1).
`D20` due 09-18, `D27` 09-20, `D28` 09-21, `D29` 09-22, `D30`-colab 09-25 — all
armed with legal defaults and future dates.

**Nothing is escalated that a measurement could settle**: no `MEANS-ESCALATED`,
and I checked the blackout entry against rule 3 by hand since the tool cannot
see it — its fork turns on how much of the owner's shared meter the owner's own
work will consume, which is not a number any experiment in this repository can
produce. It is correctly `class: goal`. **No owner decision was quietly acted on
without being recorded.**

---

## 8. The honest summary

**No. We are further away than yesterday, and yesterday we were already
stopped.**

Two days ago the builder shipped a four-hundred-character instrument that told
two indistinguishable failure causes apart on its very first opportunity. That
was real work on the real problem. It has been sitting unread on disk for
forty-three hours, and the arithmetic now says the earliest anyone can read it is
Saturday morning — about fourteen hours before the free compute it was bought to
spend expires.

And in the gap, the thing that broke is the one organ nobody thought to watch.
This project has spent three weeks building machinery so that a decision cannot
go dark: defaults, deadlines, a firing-diff audit, a two-channel identification
for firings, a whole page of `SYSTEM.md` explaining that a governing document
naming an enforcement is making a capability claim and is bound by law 1 like any
other. Yesterday morning the Review escalated the most consequential item on the
board, armed it correctly, dated it, published it to the owner — and typed an id
that was already taken. The register swallowed it without a sound and the
instrument said `ratchet ok`.

That is the same shape as every scar in this building. `D1` was visible and
unactionable. The "Working" README was confident and unmeasured. The pace gate's
own header diagnosed shared-pool starvation in August and then fixed it with a
line that makes the starvation smoother rather than shorter. Each time, the
system's answer was to build an organ that could see the thing. **The failure
today is one layer up: an organ that can see everything except whether it read
the whole file.** A dict keyed by a hand-typed id, no uniqueness assertion, and a
count reported from the size of the dict rather than the number of blocks — three
lines, and they are enough to make an armed deadline stop existing while every
page says the desk is clean.

**And it happened twice this morning, in two different organs, which is what
turns an anecdote into a class.** The decision desk could not see whether it had
read the whole file. `coverage.py` cannot see whether its `COMMITMENTS` list
still matches `GOAL.md`. Both report a count computed from what they successfully
parsed, and neither can compare that against what was actually there to parse.
Every instrument in this repo that resolves hand-written names against a dict has
the same blind spot, and the honest thing to say is that we do not yet know how
many of them are answering narrower questions than their headlines.

The ladder is *mostly* the right ladder, and I have to soften that sentence
today rather than repeat it. `coverage` reports zero uncovered commitments and
I have reported that number in good faith for ninety-seven audits — and this
morning the Review, paying off a promise it had broken at midnight, found that
three of the seven primitives the owner names in one sentence of `GOAL.md` were
never typed into the list `coverage` checks. The instrument was not lying. It
was answering a narrower question than its headline implies, and nobody had
checked its source against the source. The ledger is still honest; I went at it
again this morning and it held. But `demonstrated` has not moved in fifty hours,
"too cold kills him" and "he builds a shelter" have had nothing falsifiable
behind them for thirteen days, twenty-eight free GPU-hours are running toward a
Saturday grave, and the one decision that says all of this out loud is the one
the decision desk cannot see.

We are not building a creature this week. We are finding out which of our
instruments were telling us the truth.

---

## FOR THE BUILDER

Ordered by what stops being repairable first. Items 4–9 are the 97th audit's 1–6
carried forward unchanged; they are not re-derived because nothing has been able
to touch them.

1. **Renumber the colab `D30` to `D31`, and make `decisions.py` refuse a
   duplicate id.** Two `DECIDE: D30` blocks exist (`DECISIONS_NEEDED.md:6864`
   and `:6997`); `parse()` keeps the second and discards the first at
   `decisions.py:381`. Keep **`D30` = the blackout entry** (`decide_by`
   2026-09-18), because that is the id `PROGRESS.md` gave the owner and reassigning
   it under them is not the desk's to do; move the colab entry (`decide_by`
   2026-09-25) to `D31`. **Then the guard, which is the part that matters:**
   `parse()` must raise — not warn — when an id repeats, and `--check` must
   report the number of `DECIDE:` blocks it *read* alongside the number of
   decisions it *resolved*, so the two can never differ in silence again. This is
   a guard, not a threshold; it makes a class of bug impossible rather than
   fixing one. See the DUPLICATE-ID NOTICE I appended to `DECISIONS_NEEDED.md`
   for the full evidence. **Do not delete either entry.**

2. **Add `GOAL.md:187`'s missing primitives to `coverage.py`'s `COMMITMENTS`.**
   The owner's sentence names seven — *hot, heavy, far, tiring, dangerous,
   worth-it, that-person-lied* — and **`heavy`, `far` and `worth-it` have no
   entry**, so `coverage` reports `0 commitments with NO declared spec` over a
   list that never contained them. The standing rule above this organ's own
   instrument says a `GOAL.md` commitment `coverage.py` cannot name must be added
   to `COMMITMENTS` in the same commit. `D13` forbids the overseer editing its
   own script, which is why this is yours. **Expect the ratchet to get worse and
   let it** — this counter may grow when the source grows; what it may never do
   is shrink by trimming the list. The Review has the dated row
   (`goal-187-names-seven-primitives-four-have-no-commitment`, DUE 2026-09-18)
   and read `tiring` as uncovered too; take that call with the row, not from me.

3. **Fire `D19`'s NO-FETCH default, and fire it off the pace-skip path.** It has
   been due since 2026-09-15 00:00 — **31 hours and 31 skipped slots**. Journal it
   with the required wording — *"the owner did not rule by 2026-09-14, so the
   pre-registered default fired"* — record that it fired late and why, and say how
   to reverse it. Then do the structural half the 97th audit asked for and
   nothing has been awake to do: add the `decisions --check` OVERDUE read to the
   skip branch beside `harvest_bookkeeping`, under the comment already written at
   `ladder_loop.sh:112`. On RANK 2's bound the next unpaced slot is **Saturday**;
   an armed default must not wait for it. **The deadline does not move.**

4. **Harvest `/data/t108_backend_probe.json` as the first act of your next
   unpaced slot.** 43 h 30 m on disk. `failures[1].stdout_head` = `JACK_OUT
   /content` → cause-2, the read your own 10:07 journal pre-registered. Also read
   `failures[*].stdout_tail` — three of five per-seed `heldout` values survive
   there, byte-identical across both attempts. Take the branch the ruling
   pre-registered; do not re-derive it; **no third colab dispatch under the
   unchanged mechanism** (the 09-15 ruling §9d forbids it), and step (a) spends
   zero GPU.

5. **The pace-skip path must NOTICE a finished detached artifact.**
   `HARVEST_PATHS` is four in-repo files; the probe writes to `/data/` by design
   and buys no ledger row, which is how `008f2eb` committed the probe's bill and
   left its answer on the floor. Do not widen `git add` — the `add -A` ban
   stands. Check `declared_pids` for an exited pid whose declaration names a
   dispatch and `say` it into `ladder.log` loudly. Committing the interpretation
   still belongs to an unpaced iteration; **noticing costs nothing.**

6. **Prune `declared_pids` on exit.** It now declares **three** dead processes:
   `405151` (44 h stale), `638639` and `638640` (yesterday's desk slots). Reap
   them or stamp `EXITED <ts>`. A declaration that outlives its process is a
   claim that outlived its evidence, and the next waking slot reads it as work in
   flight.

7. **Mark per-job GPU overruns on every backend.** `charge()` should mark and
   print whenever billed hours exceed the declared `est_hours` past a stated
   margin, on **every** backend, not just kaggle (`gpu.py:521`). Both colab
   attempts declared `est_hours: 0.7` and billed 1.03 and 1.08; neither left a
   mark. **This is a report, not permission to invent a colab ceiling** — that
   number is the owner's (FOR THE OWNER 3).

8. **Make PROBE hours visible to `gpu_hours_no_verdict`.** It joins charged jobs
   against `ledger.results` (`run.py:1456`) and a probe is *defined* as buying no
   ledger row, so the one counter built to catch GPU-with-nothing-to-show cannot
   see the one class of spend guaranteed to have nothing to show. Extend the join
   to read `gpu_submissions.jsonl`'s `spec_phase` into a named `PROBE` bucket.
   **Reporting-only and unfloored**, per `D27`'s reasoning: probe spend is
   legitimate and gating it would punish the honest thing.

9. **`T1.07`'s re-buy is still owed** (~0.47 GPU-h, the cheapest ladder-moving
   unit on the board) and its certificate is standing deliberately staled. The
   hours funding it expire Saturday.

10. **Nothing in 1–9 is a threshold and nothing in them is science.** If any turns
   out to need a bar, a ceiling or a spec, it is a routing, not a default — say
   so and route it.

---

## FOR THE OWNER

**1. ACTION, NOT A DECISION — one of your decisions has been lost by a typo and
the tool that watches your desk cannot tell you.** `docs/DECISIONS_NEEDED.md`
holds **two** entries numbered `D30`. `decisions.py` keeps the second and
discards the first without a word. The discarded one is the Review's blackout
escalation — the item its own page told you is *"the only item on this page that
you alone can settle"*, `decide_by` **2026-09-18**. It cannot go overdue, its
default cannot fire, and its default has never been safety-checked. The
surviving `D30` is the colab-ceiling question, `decide_by` 2026-09-25, which is
why the tool shows you a date a week later than your page does.

**And it has already moved a real date.** Yesterday's `PROGRESS.md` told you the
blackout decision was due **2026-09-18**. Today's — same desk, same entry — tells
you **2026-09-25**, because the desk read its own entry back out of the tool and
got the other one. Nobody extended anything. The register drifted a deadline by
seven days on its own, on the item its own text calls the thing that *"blocks
EVERY spec, because it blocks the organ that runs them."* **Treat 2026-09-18 as
the live date for the blackout question.**

**Nothing here needs your ruling and nothing may be deleted** — the fix is a
renumber plus a uniqueness check, and it is `FOR THE BUILDER` 1. You are told
because for one day your desk has been quietly one item short, a date on it moved
by a week with no author, and every instrument said it was clean.

**2. The blackout is now 43.5 hours and the honest statement about its end is a
BOUND, not a forecast.** Yesterday this organ told you "13:00–14:00 today". That
was wrong within two hours — the meter jumped 41% → 60% at 09:07 — and I am
correcting it rather than restating it. What I can say without modelling
anything: the pace line is arithmetic, and **at today's meter reading of 73% the
builder cannot return before Saturday 2026-09-19 at 09:50 UTC**; every further
point pushes that back ~2.6 hours. `usage_attribution` now reads **NOT THIS
PROJECT 64 of 73 points — 87%** (it was 75% yesterday), and this project's own
attributed share has not risen at all in two days because nothing of ours has
run. **28.62 free GPU-hours expire at the end of Saturday**, roughly fourteen
hours after that bound, with a legal authorised buyer already waiting for them.
W32 lost 8.82 h this way and W33 lost 22.11 h; W37 is on course to lose more than
both combined. **You do not have to decide anything for the loop to recover** —
but it will not recover this week by waiting, and that is now arithmetic rather
than a projection.

**3. DECISION STILL NEEDED — should colab have a weekly ceiling, and what is
it?** Unchanged from yesterday and now the *surviving* `D30`. Today
`remaining("colab")` returns **infinity** and `afford("colab", 999h)` returns
**True**; the overrun mark is hard-coded to kaggle; `"overruns"` is empty after a
probe spent 223% of its authorisation, 2.11 h of it in the lane that returned
nothing. If colab genuinely is free and unlimited for you, say so and the right
repair is a comment stating it, not a ceiling. If it is not, the builder needs
the number from you — **an organ may not set its own budget.**

**4. `D19` fired late and is getting later, and I am telling you rather than
extending it.** Due 2026-09-14, firable from 2026-09-15 00:00, **31 hours and 31
skipped slots unfired**, because the organ that fires defaults is pace-gated. On
item 2's bound it will be five days late. It costs 3 specs (`HR.1`–`HR.4`, the
hearing programme's speech half). Its options remain (i) capped `/data` cache,
(ii) relocate `HF_HOME`, (iii) decline. **The deadline was not extended and must
not be.**

**5. NO-DECISION — the queue broke four promises at midnight and had paid all
four off with answers before this audit finished.** They were the Review's own
decision debt, left standing on their date deliberately while the builder's rows
were re-dated on measured cause, under a published sentence binding the desk to
own the break. By 07:0x all four read **ACTED** with named executing commits
(`81fdaba`, `34116ca`, `d521384`), not re-dated and not relabelled; `run
review-queue` is back to **EXIT 0**. I read it at 4 and am reporting it at 0,
because this page is current state. `D28` (due 09-21) is where the structural
drain question lives and the Review has recommended amending its own default
before it fires; I agree and am not re-asking it.

**5b. NO-DECISION, but it is the one worth your eye — the coverage instrument has
a hole and the desk in item 5 found it.** `GOAL.md:187` names seven primitives
survival is supposed to buy — *hot, heavy, far, tiring, dangerous, worth-it,
that-person-lied* — and `coverage.py`'s hand-typed `COMMITMENTS` list does not
name **heavy**, **far** or **worth-it** at all. So the "0 commitments with NO
declared spec" this organ reports to you every morning is 0 against a list that
is missing three of your own words. Nothing for you to rule on; the fix is one
list and it is `FOR THE BUILDER` 2. You are told because that top line is the
number this report leans on hardest, and today it turned out to be narrower than
it sounds.

**6. NO-DECISION — thirteen days now.** `claim_dead = 4, unchanged since
2026-09-03`: **smell**, **balance**, **thermal (too cold kills him)**,
**shelter (he builds a shelter)**. Every park legal, every park evidence-backed,
no marker to be touched. Three of the four successors are redesigns owed by the
desk in item 5. It is the single most goal-relevant number on this page, it has
not moved in nearly two weeks, and it gets said out loud rather than left in a
tool's output.

**7. NO-DECISION — the ledger is clean and I went at it hard.** 108 PASS rows,
every one resolving to a live spec and a live commit save one dirty row an
existing instrument already classifies and reports. Zero loosening in seven days;
every numeric movement in the window is a tightening with the arithmetic stated
in its own commit, and the two `falsified_by` lines that *look* deleted in the
diff are strengthenings that keep the old text and add a conjunct. Whatever else
is wrong today, nothing on the scoreboard is a lie.
