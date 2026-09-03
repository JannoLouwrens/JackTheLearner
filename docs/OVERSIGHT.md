# OVERSIGHT — 64th audit, 2026-09-03 00:45 UTC (HEAD `6c57954`, 0 unpushed, `ledger.json` dirty with PS.01's clean-tree re-buy, no runner alive)

## VERDICT: DRIFTING — three PASS certificates dropped off the board six hours ago, a shrink-only ratchet built for exactly this went red, and **five consecutive iterations, four instruments and one Review desk all failed to say so** — because the tool that noticed was *already* red for an unrelated, blessed reason

**The ledger itself is clean and I want that said first and plainly.** All **91**
PASS rows are mechanically sound: every `commit` resolves in git, **zero** `+dirty`
stamps survive, every PASS id resolves in `BY_ID`, and the only two PASSes with no
`control_metrics` (`T0.01`, `T0.10`) declare `control = "NONE, BY DECISION (52nd
audit B5)"` on their face. **Zero loosened thresholds** across 58 commits in 24 h
and 308 in seven days — I chased four candidates and every one moved the *other*
way (§2). The builder is healthy: **25 iterations, all `rc=0`**.

The drift is not in any number. It is in **who can still read the numbers.** A red
that is permanently red is not an alarm; it is a hiding place, and something real
hid in one tonight.

Findings ranked by damage to the trustworthiness of the ledger.

---

## THE FOUR MANDATORY INSTRUMENTS (rc read live, not quoted)

| instrument | rc | reading |
|---|---|---|
| `coverage` | **2** | 0 commitments with NO spec. **`!! unreachable specs GREW: 89 of 217 vs baseline 85`** — NEW tonight, see FINDING 1. 4 CLAIM-DEAD (smell, balance, shelter/building, thermal); 9 with live claims, nothing passing. 3 park→release pairs unwalkable; 4 cost classes with no path in. |
| `decisions --check` | 0 | **0 MEANS-ESCALATED, 0 UNDECLARED, 0 OVERDUE.** Ratchet 0/10. D15+D16 due 09-05, D17 09-07, D18 09-09. Nothing to arm — the list is fully armed. |
| `champions --check` | 0 | 27 seats. `arena_missing` **0**; unfalsifiable 3/3; uncontestable 3+1/4; unverified verdicts 2/2; trigger debt 3/3. Every ratchet at baseline. |
| `run review-queue` | **2** | **1 VIOLATION — STALE**, new since the 63rd audit (was rc=0). 26 OPEN / 2 HELD / 2 ACTED of 30; oldest live 10 d; consumer ran 1 d ago. See FINDING 3. |

---

## FINDING 1 (RANK 1) — the board lost `PS.02`, `PS.03` and `BA.01` at 19:08, four specs went unreachable behind them, the ratchet built on 09-01 to catch exactly this **fired correctly and was heard by no one for 5½ hours**

### What happened, from the rows

Three certificates that had been PASS since 2026-08-19 are VOID:

| spec | status | ran_at | stamp | duration | message |
|---|---|---|---|---|---|
| `PS.02` | **VOID** (was PASS) | 2026-09-02T19:08:08 | `e5dcb17` | 0.14 s | *run did not test the claim; not a refutation* |
| `PS.03` | **VOID** (was PASS) | 2026-09-02T19:08:10 | `e5dcb17` | 0.14 s | same |
| `BA.01` | **VOID** (was PASS) | 2026-09-02T19:08:11 | `e5dcb17` | 0.14 s | same |

No science ran — 0.14 s each. `protocol.py:borrow_metrics` refused, correctly, because
their source row `PS.01` carried `c7325c2+dirty`. `PS.01` was dirty because the **63rd
overseer audit's own five uncommitted docs** were in the tree while the bounded gate
sweep wrote rows. The builder diagnosed this exactly right at 19:1x. The refusal is
the protocol working as designed; **everything after it is the finding.**

### The blast radius, computed — not estimated

Run against the live ledger with the real function, and again with the three borrower
rows restored to PASS **in memory only** (nothing written to disk, verified after):

```
live:                                    unreachable = 89 of 217
counterfactual (3 borrowers = PASS):     unreachable = 85 of 217
delta attributable to the 19:08 cascade:              +4
```

`UNREACHABLE_BASELINE = 85`. **The entire growth of the project's shrink-only
reachability ratchet is this one accident**, and it is fully reversible by three
clean-tree re-buys. The four specs that went unreachable are, from `run blocked`:

- behind `PS.02` — **`SH.01`, `SH.02`, `XL.01`** — shelter, *"too cold kills him"*, death-and-retry
- behind `BA.01` — **`BA.03`** — balance

Those are four of the commitments the 2026-08-10 coverage miss was built to protect.
They did not go dark because anyone decided anything. They went dark because a
documentation commit landed inside a sweep window.

### Why nobody said so — the part that matters

`coverage` **did** fire. It prints, tonight and since 19:08:

> `!! unreachable specs GREW: 89 of 217 vs baseline 85. Growth is permitted only with a named justification in the commit that grows it…`

Nobody read it. Checked, not assumed:

- **`grep "unreachable" docs/LOOP_JOURNAL.md`** — the last recorded reading of this
  number is **2026-09-01 ~19:1x**: *"unreachable 85/217 baseline UNMOVED."* Four
  slots on 09-01 read it; **zero slots on 09-02 read it.**
- `"unreachable specs GREW"` and `"unreachable 89"` appear **0 times** in
  `LOOP_JOURNAL.md`, `OVERSIGHT.md`, `REVIEW_QUEUE.md`, `PROGRESS_LOG.md`.
- Five 09-02 slots instead assert the *conclusion*: *"coverage rc=2 red **BY DESIGN**
  until 09-06"*, *"the pre-existing routed GEN-corpse red — not touched"*, *"do not
  'fix' it."* Each sentence was true about the red they meant. Each one licensed
  skipping the body of a tool that was, at that moment, printing a different red.

**The sharpest fact in this audit:** on **2026-09-02 13:1x** — six hours *before* the
cascade — the loop shipped `DELIBERATE_RED_METRICS` in `run status`, whose entire
purpose is that a moving number inside a deliberately-red gate must print its delta
(`"T0.27 live_violations = 3 (MOVED +1 since 2026-09-02T04:09:57; was 2)"`). Its own
commit message names the class: *"the class of defect that let three reports call a
moving number 'unchanged' now has a reader instead of a memory."*

It was built for **one metric inside one red gate**. Six hours later the identical
class recurred inside a **different** red gate, and the new reader did not cover it.
A repair scoped to an instance leaves the class open — and the standing red is where
the class hides.

### And the loop's own to-do list quietly shrank

- **19:1x**: *"clean-tree re-buy of **all six rows** running in background (LC.00 and
  LC.01 already re-passed; PS.01 ~14.5 min, **then the three borrowers**)."* That
  background job died before finishing.
- **20:1x, 21:1x, 22:1x** handoffs, verbatim and identical: *"PS.01 still owes a
  clean-tree re-buy."* **The three borrowers are named in none of them.**
- `PS.01` re-bought PASS at **00:30:29** (attempt 7, `6c57954`, metrics byte-identical
  to the dirty run — a pure re-stamp). It sits uncommitted in the tree as I write.

So the last owed item on the board is about to be marked closed while three
certificates and four commitments stay dark. **`PS.02`/`PS.03`/`BA.01` declare no
`depends_on` and their borrow source is now PASS — all three are re-buyable in
seconds, today.**

---

## FINDING 2 (RANK 2) — the root-cause row was promised in the open and never written, so the trap that fired tonight is still armed

The 19:1x iteration summary lists, verbatim: *"route the cross-organ race as a queue
row."* The 20:1x entry reports three completed items and does not mention it.

`grep -i "dirty\|overseer\|cross-organ\|race\|DOC_OUTPUTS" docs/REVIEW_QUEUE.md`
returns nothing matching; the newest `ROUTED:` line in the file is `t309-control-clears-the-claims-own-margin` (09-02). **The row does not exist.**

The diagnosis it was supposed to carry is on record and is correct: `protocol.py`'s
`DOC_OUTPUTS` excludes only the two docs the *builder's* loop writes, so the
overseer's five all count as code dirt — and three of them
(`DECISIONS_NEEDED.md`, `REVIEW_QUEUE.md`, `PROGRESS_LOG.md`) are genuinely
machine-read by instruments, which is why it is a design fork for the Review and not
an exclusion-list one-liner.

Until it is routed, **every audit that commits while a sweep runs will VOID
certificates again** — including this one. I am committing immediately, into a
verified-idle tree (`pgrep` shows no runner; next builder slot 01:07), because the
mitigation available to me is timing and nothing else.

---

## FINDING 3 (RANK 3) — `run review-queue` went red today: the row asking whether the language-routing seat should keep its holder has aged out with no clock

`STALE: t215-router-under-lexical-null` — OPEN **9 d**, past the 8-day consumer cycle,
no `DUE:`. rc **0 → 2** since the 63rd audit.

This is normal backlog arrival, not dishonesty — but the content is not minor. The
row asks whether the shipped `UnifiedBrain` semantic-anchor argmax router keeps the
language-routing seat, on evidence that **on seed 2 it routes worse than both
registered bag-of-words nulls** ([8,9,5]/16 vs a 12/16 bar; NB 14/16, TF-IDF 11/16),
paired with `T2.07`'s independent FAIL. Two FAILs localising a defect in a mechanism
that holds a seat is exactly the material `CHAMPIONS.md` exists for. The honest
repairs are ACT, DECLINE, or re-arm with a new `DUE:` and a reason.

---

## §2 — THRESHOLDS AND CONTROLS: **NO FINDINGS**, and that is a real result

Scanned `git log -p --since="7 days ago"` over `registry.py`,
`registry_expansion.py` and `experiments/tests/` (308 commits; 58 in the last 24 h).
Four constant moves, every one in the strengthening direction:

| change | commit | direction | justified? |
|---|---|---|---|
| `T3.09` `N_LIVES` 16 → 32 | `d36f3f9` | **more evidence** — sequential-RNG extension, old 16 spawns an exact prefix; fired VOID lane `n_affected 6 < 8` | yes, with arithmetic; every gate constant untouched |
| `T3.09` `seeds` 1 → 3 | `19461c4` | **stronger** | yes (61st audit B1.3) |
| `LG.10` `TEMP` 0.25 → 1.0 | `f6d1e3a` | **stronger** — more sampler entropy makes match/unanimity/swap/null all harder; v1 VOIDed on its own variety floor | yes, and the v2 run **FAILed** anyway — the move bought nothing |
| `LG.00` `_check`: `m["verdicts_missing"] > 0` → `max(per_seed) > 0` | `a0aa9cd` | **equal or stronger** (mean > 0 ⟹ max > 0) | yes, part of the purity repair |

No control deleted or weakened; no `_check` gained an `or`; no seed count reduced; no
assertion removed. The opposite happened twice: `T3.09`'s control-vacuity lane was
moved **above** the claim branch so it can fire on a FAIL (`19461c4`), and `LG.00` and
`LG.02` were both refactored so their gates replay from the recorded row alone — which
is why `T0.13` went `keyless_gates 2→0, stale_gates 2→0` and re-passed.

---

## §3 — DRIFT: none. Meta-work is heavy but each piece traces to a GOAL.md sentence

Last 24 h, by file: `ledger.json` 21, `LOOP_JOURNAL` 19, `REVIEW_QUEUE` 9, `run.py` 4,
`champions.py` 4, `registry_expansion.py` 4, `t3_09` 3, plus `lg_10`, `lg_02`,
`protocol.py`, `coverage.py`, `lib_procwatch.sh` ×2 each.

Capability work (6 spec files): `LG.02` **PASS** — *"his diary records whose advice
proved true, so trust in a person can be earned and checked"* (GOAL.md:145). `ME.11`
FAIL, `T3.09` FAIL, `LG.10` FAIL — three honest REDs, each bought knowingly against a
pre-registered bar. `LC.07` pilot-blocked; `LG.00` re-bought.

Instrument work (`champions.py` trigger-reachability, `t0_13`, `t0_29`, `t0_31`,
`protocol.py` peak-RSS, procwatch memory guard): serves *"protects the honesty of
watching what happens when the three meet"* (GOAL.md:8). Legitimate — but see §8.

**The converse, which is harder:** `fast/slow` **0 passing of 8** (3 welded behind
`LC.03`, `DP.04` foreclosed, `BO.01` behind `DP.05`); `one brain / unison` **1 of 23**;
`curiosity` **2 of 12**; `plasticity`, `sleep`, `hunger/thirst`, `touch`,
`proprioception`, `tool use`, `damage`, `death & retry` all **0 passing**. Four
commitments CLAIM-DEAD, and tonight's cascade pushed the shelter/thermal/death/balance
line further from any dispatchable path. **Nothing in this audit is a drift finding;
the shape of the ladder is.**

---

## §4 — BUILDER: alive and honest

**25 iterations** 2026-09-02T00:07 → 2026-09-03T00:17, **25 × `rc=0`**, one
`LEFTOVER=1 undeclared process` warning at 05:39. No pause, no credit exhaustion, no
load aborts (load 0.02–2.64). Meter `week:all models` climbed 5% → 23% across the day;
the gate is read and named every slot, per memory.

**PASS delta: 93 → 91 (net −2).** Honest and traceable: 94→93 (bounded gate caught the
real `T0.13` regression at 16:18), 93→90 (the 19:08 cascade), 90→91 (`LG.02`, then
`T0.13` re-pass and `PS.01`). No number was rounded up anywhere.

---

## §5 — COMPUTE: honest accounting, third straight week of expiring free quota

`2026-W35`: **19.20 h charged, 12 jobs, 0 failed** (kaggle 18.93, colab 0.27) against
30 h. **~11 h expire Sunday 2026-09-07.** Every hour has a ledger row or a named
pilot record: `D1.0` 16.17 h → an honest VOID (routed), `UB.10` 0.30 h → VOID
(routed), `UB.10` grid pilot 0.40 h → SELECTED record, `LC.07` 0.44 h → BRANCH B
(routed), `T1.09`/`T1.10` 0.11 h → two PASSes. **No unattributed GPU spend.**

The waste is structural, not sloppy: `coverage` reports `gpu<20min` and three other
cost classes with **no path in** — nothing runnable to implement, nothing
gate-provisional to pilot. The loop declined to manufacture a dispatch every slot and
said so with the price attached. That is the correct call, and `D15` already names
three consecutive weeks of expired quota as a measured cost.

---

## §6 — STUCK DECISIONS: clean

0 MEANS-ESCALATED, 0 UNDECLARED, 0 OVERDUE; nothing to arm. `D16`'s evidence was
correctly refreshed on 09-02 (`ae78af2`) after `T0.27`'s `live_violations` moved 2→3 —
the premise change was carried onto the entry rather than left stale, which is the
right handling and worth naming. No owner decision was acted on without a record: all
ten 09-01 firings appear in `DECISIONS_RESOLVED.md` with losers and reversal paths.

## §7 — BAKEOFF HYGIENE: no new defect; one standing one

The 09-01 default firings each name their losers and a reversal. The standing defect
is unchanged and is the 62nd/63rd audits': **`D10` seated `wm-latent` BY VERDICT off
`LC.03`, which is a VOID**, and `champions.py` now confirms it independently —
`VERDICT-IS-A-VOID` plus `TRIGGER-UNREACHABLE` on all three of that seat's re-open
doors. I am not re-litigating it; it is on the 09-06 docket and `champions --check`
is at baseline with it counted.

---

## §8 — THE HONEST SUMMARY

**Yesterday was a real day for the creature and a bad day for the instrument panel.**

`LG.02` is the kind of thing this project exists for: two advisors, one truthful and
one not, no script, no label — and he learns which voice to follow from his own
attributed diary and his own verification (worst-seed divergence 0.60 against a 0.40
bar, trust in a stranger exactly 0.5 at first encounter, and a swap control that
migrates when the liar starts telling the truth). That is *"his diary records whose
advice proved true"* moving from a sentence in `GOAL.md` to a row in the ledger. And
`ME.11`'s FAIL is worth nearly as much: the memory commitment went from **unmeasured
to measured**, with a distractor control that caught the dense arm confabulating at
**1.8× its correct-recall rate**. Honest reds are progress. Three landed in a day.

But: `fast/slow` 0 of 8, `unison` 1 of 23, `curiosity` 2 of 12, and tonight four more
specs behind shelter, cold, death and balance became unreachable **by accident, not
by evidence**. The ladder-and-apple standard is not closer than it was yesterday.

And the thing that should worry us most is not on the ladder at all. This system now
has enough instruments that its *reds* have become load-bearing state — `coverage` has
been red "by design" since 09-01 and will stay red until 09-06 — and tonight a genuine
new red hid inside a blessed old one for five and a half hours while five iterations,
one Review desk and three ratchets all read `rc=2` and moved on. The loop *built the
fix for this class at 13:1x and it did not generalise*. We are not, right now, losing
the ability to measure Jack. We are losing the ability to notice that we stopped.

---

# FOR THE BUILDER

**B1 (rank 1, cheap, do it first — a few seconds of CPU buys back 3 certificates and 4 specs).**
Commit `PS.01`'s clean-tree re-buy (already in the tree: PASS attempt 7, `6c57954`,
metrics byte-identical to the dirty run), then re-buy **`PS.02`, `PS.03` and `BA.01`**
from a clean tree — the three rows the 19:1x background job was running when it died,
and which the 20:1x/21:1x/22:1x handoffs dropped. None declares `depends_on`; their
borrow source is PASS again; each VOID cost 0.14 s and no science. Afterwards
`coverage` must print **85 of 217** and the `unreachable specs GREW` line must be gone.
**Do NOT raise `UNREACHABLE_BASELINE`** — the growth is an accident being reversed, not
a registration being justified, and raising it would ratchet in an error.

**B2 (rank 2, the durable repair — generalise the 13:1x reader from an instance to a class).**
`DELIBERATE_RED_METRICS` in `run status` was built for one metric in one red gate
(`T0.27 live_violations`) and correctly reports its delta. Six hours later the same
class recurred in `coverage` and nothing caught it. Extend the same idiom to the
**ratchet counters** every standing-red tool carries, printed with their delta since
the previous committed reading and independent of the shared exit code:
`unreachable_ratchet` count-vs-baseline, the CLAIM-DEAD count, the park-release-pair
count, the champions trigger-debt total, and the review-queue violation total. Build it
**RED-first against tonight's 89-vs-85** — it must fire on the live tree before B1 is
taken, and go quiet after. The rule it enforces: *a tool that is already red for a
blessed reason must still be able to report a number that moved.*

**B3 (rank 3 — write the row you promised).**
The 19:1x summary said *"route the cross-organ race as a queue row"* and no such row
exists. Route it with a `DUE:`, carrying the diagnosis already on record:
`protocol.py:DOC_OUTPUTS` excludes only the builder's two docs, so the overseer's five
count as code dirt while three of them (`DECISIONS_NEEDED.md`, `REVIEW_QUEUE.md`,
`PROGRESS_LOG.md`) are machine-read by instruments — a design fork for the Review, not
an exclusion-list one-liner. Until it lands the trap is armed and the next audit that
commits during a sweep will VOID certificates again.

**B4 (rank 4).** `t215-router-under-lexical-null` is **STALE** (9 d OPEN, no clock) and
`review-queue` is rc=2 because of it. ACT, DECLINE, or re-arm with a new `DUE:` and a
reason. It is the row asking whether the anchor-argmax router keeps the
language-routing seat while losing to a bag-of-words null on seed 2 — do not let it
age quietly.

---

# FOR THE OWNER

**Nothing new needs your ruling, and nothing is stuck on your desk that a measurement
could settle.** All four instruments' escalation classes are empty or at baseline.
Three things to be aware of, none requiring an answer:

1. **Four armed defaults fire this week if you stay silent** — `D15` and `D16` on
   **2026-09-05**, `D17` on 09-07, `D18` on 09-09. Each is written to pick only among
   already-permitted actions, and `D16`'s deliberately picks the option that leaves the
   ladder a *visible red* (`T0.27`) rather than the one that makes it green.
2. **~11 free GPU-hours expire Sunday 2026-09-07, for the third consecutive week** —
   not through inattention: every GPU cost class currently has nothing runnable in it.
   `D15` already names this as a measured cost. Only you can take the option outside
   the repo.
3. **Four of your constitutional commitments have no live claim** — smell, balance,
   shelter/building, and *"too cold kills him"* — and tonight's accident pushed
   `SH.01`/`SH.02`/`XL.01`/`BA.03` behind a dead root as well. B1 reverses the accident.
   The underlying claim-death is on the 09-11 Review row
   `five-commitments-are-claim-dead-behind-foreclosures` and is a redesign question,
   not a re-run.
