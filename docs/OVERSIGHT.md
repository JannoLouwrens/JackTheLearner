# OVERSIGHT — 63rd audit, 2026-09-02 18:45 UTC (HEAD `c7325c2`, 0 unpushed, `ledger.json` dirty by the live sweep pid 505077)

## VERDICT: DRIFTING — a non-negotiable CONDUCT-class constraint was exceeded **5×, live, today**, by a run that the ledger then stamped **PASS**; and the project's most consequential architectural seat lost the last of its three escape hatches 22 hours after it was seated

**No integrity breach in the ledger itself, and I want that said first and
plainly.** All **93** PASS rows are mechanically sound — every `commit` resolves
in git, every spec that declares a control has control metrics recorded, and I
found **zero** loosened thresholds across **97** commits in seven days. The two
candidates I chased both moved the *other* way. The builder is healthy: 26
iterations in 24 h, all `rc=0`, and the net `94 → 93` is *honest* — the bounded
gate it shipped this morning caught a real regression in `T0.13`.

The drift is in the two places no instrument looks: a hard constraint that is
cited by its own guard and enforced by nothing, and a seat whose contestability
was promised by construction and has since decayed to zero.

Findings ranked by damage to the trustworthiness of the ledger. **FINDING 2 is
the one with consequences off the ledger** — it is on a box with paying tenants.

---

## THE FOUR MANDATORY INSTRUMENTS

| instrument | rc | reading |
|---|---|---|
| `coverage` | **2** | **0** commitments with NO spec. 4 CLAIM-DEAD (smell, balance, shelter/building, thermal); 8 more with live claims and nothing passing. 4 NEW unrunnable GOAL.md citations — GEN.02/03/06/09, **all welded behind LC.07** (see FINDING 1). Every dispatch class empty: 3 dispatchable, all VOID, **0 fresh**. |
| `decisions --check` | 0 | **0 MEANS-ESCALATED, 0 UNDECLARED, 0 OVERDUE.** D15/D16 due 09-05, D17 due 09-07. Ratchet 0/10. |
| `champions --check` | 0 | 27 seats, 8 declared violations, `arena_missing` **0**, unfalsifiable 3/3, unverified verdicts 2/2. Every ratchet at or below baseline. |
| `run review-queue` | 0 | **0 violations.** 26 OPEN / 2 HELD / 2 ACTED of 30 routed; oldest live 9 d; consumer ran today. |

I re-derived each reading rather than trusting the journal. `coverage`'s rc=2 is
the pre-existing, already-routed GEN-corpse red (`goal-cites-four-specs-that-resolve-to-corpses`, DUE 09-10) — correctly *not* "fixed" by the builder.

---

## FINDING 1 (RANK 1) — `D10` seated a VOID on a promise of contestability; **all three of its pre-registered re-open triggers are now behind closed doors**, and the promise expired 22 hours after it was made

The 62nd audit found the VOID-as-verdict and correctly credited the instrument
repair (`VERDICT-IS-A-VOID` now fires). **This is the next layer, and no
instrument states it, because each one sees a single arm.**

`D10` fired 2026-09-01 and seated `wm-latent` **BY VERDICT** — the strongest
marking in `CHAMPIONS.md` — off `LC.03`, whose ledger status is **VOID**.
`SYSTEM.md` is explicit: *"VOID: an arm failed the learning gate; fix the arm, do
not decide."* `D10` justified overriding that with a promise, quoted verbatim
from `DECISIONS_RESOLVED.md`:

> **`LC.07` registered in the SAME commit** — the ~10× Kaggle scale-transfer
> re-test … so the seat is seated and contestable in the same breath. The
> ARENA-UNREACHABLE finding on this seat (54th audit) is **discharged by
> construction, not by prose.**

**The timeline, from git:**

| when | what |
|---|---|
| 2026-09-01 00:20 | `D10` default fires; seat marked BY VERDICT; promise made |
| 2026-09-01 03:26 | `c934e47` — `LC.07` implemented, gates PROVISIONAL |
| **2026-09-01 22:10** | **`2b832ed` — LC.07 pilot returns BRANCH B.** `_GATES_FROZEN = False`; `run()` refuses; *"Do not re-run the pilot; it is spent evidence."* |

The promise held for **22 hours**. Checking each of `D10`'s own pre-registered
re-open triggers against the live registry:

1. **"`LC.07` FAIL (the seat reverts to contested-VACANT)"** — **unreachable.**
   `LC.07` has **no ledger row at all** and cannot acquire one: its own `run()`
   refuses (`lc_07_scale_transfer.py:227`). The cheapest class projects 14.49 h
   against an 8.5 h kernel ceiling, the arm 40.86 h, the plan ~526 wall-hours
   against 30 h/week. It cannot produce a FAIL, so the trigger cannot fire.
2. **"any repaired screen returning ≥2 learners"** — **unreachable.** `LC.03` is
   `VOID-FORECLOSED`, and `D10` itself pre-registered *"no v3, no envelope
   growth, no re-roll."*
3. **"the unison gates failing under `wm-latent`"** — the only one with a live
   path, and it runs through `UB.10`, which is **VOID** and blocks 5, awaiting an
   arm redesign (`ub10-seed-fragility-and-saturated-battery`, DUE 09-08).

**The cost, now visible in `coverage`:** `LC.07 = PILOT-BLOCKED, blocks 4`, and
**seven GOAL.md-cited specs are welded behind it** — `GEN.02`, `GEN.03`,
`GEN.06`, `GEN.09` (the four NEW corpse citations), plus `LC.04`, `DP.02`,
`DP.03`. `DP.02` is the spec `GOAL.md` names as the *"connectedness claim … that
can quietly fail"* — the adversarial lesion test for one-brain. It is welded
behind a seat that was seated on a run which measured nothing.

**Why no instrument says this.** `champions.py` checks whether an arena
*resolves* and whether a verdict is a verdict — it does not ask whether a
**re-open trigger is reachable**. `coverage.py` gained exactly that check for
parks today (`PARK-ON-AN-UNREACHABLE-RELEASE`, 62nd-audit B3, `10516fe`). The
same defect class, one file over, is unchecked. This is not a new idea to invent;
it is a shipped idea not yet pointed at seats.

---

## FINDING 2 (RANK 2, and the only finding with off-ledger consequences) — the box's memory ceiling is cited by its own guard, enforced by **nothing**, exceeded **5× today**, and the violating run was stamped **PASS**

`SYSTEM.md`, hard constraints, CONDUCT class — *fixed, and explicitly "not up for
measurement"*:

> This box serves paying tenants. … Stay at `nice 19`, **under ~1.5 GB RAM**, and
> leave no process running.

**Measured live, this audit.** At 18:38 UTC I sampled `run_spec T2.00` (pid
531762, a child of the bounded-gate sweep pid 505077):

```
18:36:1x  rss = 7,230,780 kB  (7.23 GB)   free = 808 MB
18:38:15  rss = 7,574,864 kB  (7.57 GB)   avail = 9,231 MB
18:38:2x  process exits; available recovers to ~15 GB
```

**7.57 GB against a ~1.5 GB ceiling — 5.0×.** `nice 19` *was* honoured; the
memory half was not.

**The ledger already holds the evidence, on a green row, re-stamped today.**

```
T0.07   PASS   ran_at 2026-09-02T16:15:12   commit 4abd917
        policy_peak_rss_mb = 6991.0
```

That row was **re-bought by this very sweep two and a half hours ago** and
recorded 6.99 GB — 4.6× the ceiling — as a PASS. The number is not historical
and not disputed. It is on the scoreboard, green.

**Three organs know and none of them is a guard:**

- `scripts/lib_procwatch.sh` — built by the 52nd audit's B2 — opens by quoting
  **both** halves of the rule: *"`SYSTEM.md` says 'leave no process running' and
  'stay under ~1.5 GB RAM'; until this file, that rule was enforced by
  NOTHING."* It then implements the process half only. `grep -rn
  'rss|RSS|smaps|statm|MemAvailable' scripts/*.sh` returns **zero matches** — no
  script in this repo reads the memory of anything.
- `experiments/tests/lg_01_lived_necessary_probes.py:111` and
  `dp_04_slow_path_verbal.py:1029` both cite the 6.9 GB figure **as a design
  constraint when choosing a model**. So the number is trusted enough to steer
  architecture and never trusted enough to raise an alarm.
- `Budget` prices **wall-clock only** (`"cpu<10min": 1800`, `run.py:1532`). The
  bounded gate shipped this morning (`4abd917`) — a genuinely good organ — budgets
  the sweep on time, which is not the binding constraint on a shared box. It
  re-runs 71 PASSes with no memory bound, and `T2.00` and `T0.07` are both inside
  its `cpu<10min` scope.

**What did NOT happen, stated so this is not inflated.** I checked: `dmesg` shows
**no OOM kills**; swap held flat at 1,559 MB across all samples; and WorldTwin's
`mem_watchdog.sh` reads `docker stats … MemPerc` for the aggregator against *its
own 3 GiB container limit*, so host pressure from this repo does not trip it. **No
tenant was harmed and no service restarted.** The finding is that a class-3
constraint the project calls non-negotiable is currently enforced by prose, and
the margin between 7.57 GB and trouble is luck, not design — 22.9 GB total, and
free memory did reach 808 MB.

---

## FINDING 3 (RANK 3, section 3) — the three claims `GOAL.md` calls the thesis hold **3 passes across 43 specs**, while the last five days' wins are the language family

Not drift — every recent unit traces to a `GOAL.md` sentence, and I checked each.
But the composition is the exact pattern the audit brief warns about, and it
deserves a number rather than a worry.

| `GOAL.md` commitment | specs | passing | runnable NOW |
|---|---:|---:|---:|
| **curiosity** (*"he explores because he wants to"*) | 12 | **2** | **0** |
| **one brain / unison** (*"all senses in unison"*) | 23 | **1** | 1 |
| **fast/slow** (*"one substrate, differentiated function"*) | 8 | **0** | 0 |
| — subtotal, the thesis | **43** | **3** | **1** |
| language (parent) | 9 | 3 | 3 |

The eight capability first-passes of the last five days: `T2.09`, `T2.19`
(08-29), `LG.00`, `LG.01`, `T2.14`, `VO.02` (08-30), `W0.DIAG` (08-31), `LG.02`
(09-02). Three of eight are the `LG.*` family.

**This is not a criticism of that work** — `LG.00` is the falsifier for *"strip
the diary and the core and his answers about his own life must COLLAPSE"*, one of
the sharpest specs in the ladder, and it holds a real seat. It is a statement
about where the ladder currently *can* move: `coverage` reports **0 fresh
dispatches at every cost class**, so the loop is not choosing easy wins over hard
ones. It is taking the only units that exist. The hard families are blocked
behind `T2.01` (FAIL, blocks 38, impl unchanged **24 days**), `LC.03`, `NE.01`,
`LT.01` and `UB.10` — which is why FINDING 1 matters more than it looks.

---

## SECTIONS WITH NO FINDINGS — stated plainly, because they are true

**Section 1 — ledger integrity: CLEAN.** 93 PASS of 125 rows (217 specs
registered). Every PASS `commit` resolves via `git cat-file -e` — **0 missing**.
Every PASS spec declares a control. Exactly two record no control metrics —
`T0.01` and `T0.10` — and both declare *"control: NONE, BY DECISION (52nd audit
B5)"* with a stated reason (an import either raises or it does not; a sabotaged
Kaggle upload fails service-side, which is the falsifier itself). Those are
honest declarations, not gaps. **No PASS in this ledger rests on a control that
never ran.**

**Section 2 — threshold and control drift: CLEAN, and I looked hard.** 97 commits
in 7 days across `registry.py`, `registry_expansion.py`, `experiments/tests/`.
Two hits looked like loosening and both were the opposite:

- `t2_09_noisy_tv_control.py` — `-def _control(seed)` appears in the diff. It was
  **refactored, not deleted**: re-added at the same commit as a `_fold_control`
  wrapper and still wired through `control_fn=_control` in `run_spec`.
- `t2_19_flow_multimodal.py` — `UNI_MIN`/`TIE_BAND` moved from `None`
  placeholders to frozen bars, and the new bars are **tighter** than the old
  inline values (`UNI_MIN 0.8 → 0.90`, `TIE_BAND 0.15 → 0.10`), each justified by
  a named pilot measurement. The same commit records that freezing them turned
  the spec's own dry-table row VOID and it kept the VOID.

**No seed reductions.** Every new spec registered this week is `seeds=3`; the one
`seeds=1` is a structural property check that says so in its docstring.

**Section 4 — the builder is alive and honest.** 26 iterations in 24 h, **26 of
26 `rc=0`**. PASS delta `93 → 93` net: `LG.02` gained 04:16, `T0.13` lost 16:18.
That loss is the system working — the bounded gate the builder shipped this
morning is the first organ capable of catching a decayed certificate, it caught
one on its first run, and the builder reported it against itself in the journal
rather than quietly re-buying. `lib_procwatch` also fired once (05:39,
`LEFTOVER=1 undeclared process`), which is the guard doing its job.

**Section 5 — compute honesty.** W35: 19.20 h charged, **0.00 h wasted** (12/12
jobs `ok=true`). **10.8 h of free Kaggle quota expire Sunday — the fourth
consecutive week.** The builder has priced and journaled this correctly rather
than manufacturing a dispatch: every GPU class is VOID-arm or PILOT-BLOCKED, so
the quota is genuinely unspendable, not merely unspent. Prior weeks: W32 16.61 h
(1.18 wasted), W33 7.89 h (0.26), W34 1.62 h (0.00).

**Section 6 — stuck decisions: nothing improperly parked.** `decisions --check`
is clean on all three classes. The eleven defaults armed for 08-31 all fired
09-01 with journal lines and named artifacts. D15/D16 (09-05) and D17 (09-07) are
armed and current.

**Section 7 — bakeoff hygiene: one, and it is FINDING 1.** `D10` is the only
decision made on a VOID. Every other resolved entry carries its learning gate,
its losers and its re-open trigger. No winner was chosen inside a noise margin
that I could find.

---

## SECTION 8 — THE HONEST SUMMARY

**Are we closer to a curious humanoid that climbs the ladder than yesterday, or
only closer to a longer list of green ticks?**

**Neither, today — and that is the accurate answer rather than a hedge.** The
green list did not grow (93 → 93). What grew is the machine's ability to catch
itself: a bounded regression gate that re-runs certificates cheaply enough to
actually run, and which found a real decayed one within an hour; a coverage check
that asks whether a park's *release* is reachable; a `run blocked` that prints
how long a blocker has been stalled. Every one of those is `SYSTEM.md`'s "the
machine is better than I found it" clause honoured properly.

**But the ladder itself did not move, and it cannot.** `coverage` reports **0
fresh dispatches at every one of seven cost classes** — not because nobody looked,
but because every path forward terminates in a redesign: `T2.01` FAIL blocking 38
for 24 days, `LC.03` VOID-FORECLOSED, `LC.07` PILOT-BLOCKED, `UB.10` VOID,
`NE.01`/`LT.01` FAIL. Six independent specs have now recorded, with numbers, that
**the venue is what failed** — consolidated verbatim into `w0-too-shallow`, DUE
2026-09-06. The system has correctly diagnosed that its world is too shallow to
support the claims it wants to make.

So the honest position: **the instruments are the best they have ever been and
the creature has not moved in five days.** That is not yet the failure mode
`SYSTEM.md` warns about — *"polishing the machine instead of running it"* —
because the machine is not being polished *instead of* running; there is nothing
to run until Sunday's docket redesigns W0. But it becomes that failure mode the
moment the 09-06 docket slips. **The single most important event on this
project's calendar is `w0-too-shallow` on 2026-09-06.** Everything else in this
report is smaller than that.

And one thing did get worse while nobody watched: the one-brain connectedness
test (`DP.02`), which `GOAL.md` singles out as *"the claim that can quietly
fail"*, is now welded behind a seat that cannot be contested (FINDING 1).

---

## FOR THE BUILDER

Ranked. B1 and B2 are each a guard that makes a class of failure impossible
rather than fixing an instance — the standard `SYSTEM.md` asks for.

**B1 — `champions.py`: add a `TRIGGER-UNREACHABLE` class. (rank 1)**
Point today's shipped idea at seats. `coverage.py:park_release()` already
resolves a `RELEASE:` id and fires on DANGLING / PARKED / foreclosed / `welded<-`
/ `blocked<-`. Do the same for the re-open triggers of any seat held BY VERDICT
or BY DECREE: parse the trigger's spec ids, resolve them, and fire when **every**
trigger is unreachable. Reuse `foreclosure()` — do not write a second one.
Build it **RED-first**: on the live registry it must fire on **Learning core**
(all three `D10` triggers: `LC.07` PILOT-BLOCKED, `LC.03` VOID-FORECLOSED,
`UB.10` VOID) before any migration. Ratchet it shrink-only in the `T0.31` idiom,
seeded at whatever it fires on. **Do not repair this by deleting a trigger** —
that is the `ARENA-MISSING → NO-ARENA` inversion in a new costume.
*The scar it closes:* a seat may be seated on a promise of contestability, and
that promise can decay silently — here in 22 hours — while `champions --check`
keeps printing `ok` on the arena because the arena still *resolves*.

**B2 — enforce the memory half of the rule `lib_procwatch.sh` already cites. (rank 2)**
Two parts, and the second is the one that ratchets:
  1. `scripts/lib_procwatch.sh` — while walking pids it already resolves, read
     `/proc/PID/status:VmHWM` (peak, not current — a sample can miss the spike, as
     mine nearly did) and print a `MEMORY` line naming any project python over a
     declared ceiling. Follow the file's own discipline: **name, never kill** — a
     declared detached run is legitimate.
  2. `run_spec` — record `peak_rss_mb` from
     `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss` into **every** row, the way
     `T0.07` already does for itself. Once it is a universal field, a spec that
     violates the box constraint is a number instead of an anecdote, and a future
     ratchet can gate on it.
  Evidence to build against, all live: `T2.00` 7.57 GB measured 18:38 today;
  `T0.07` PASS row `policy_peak_rss_mb = 6991.0`, re-stamped 16:15 today.
  **Do not weaken the 1.5 GB figure to match observed behaviour.** If the ceiling
  is genuinely wrong for a torch process, that is an owner question (see FOR THE
  OWNER) — say so in `DECISIONS_NEEDED.md` with the measurements attached. Moving
  a constraint to fit what already happened is the one thing law 4 forbids.

**B3 — `--max-budget` prices time only; say so where it is read. (rank 3)**
Smaller and honest: the bounded gate is a good organ shipped today, and its
`Budget` enum is wall-clock (`run.py:1532`). Until B2 lands, print a one-line
caveat in the `BOUNDED GATE` banner that the bound is time, not memory. A budget
that names one dimension is read as bounding cost; on this box the binding
dimension is RAM.

**B4 — the `w0-too-shallow` docket, 2026-09-06, outranks all of the above if they
collide.** Six specs have recorded the venue as the fault with numbers. Nothing
in this report is worth a slot that the 09-06 redesign needs. Section 8 says why.

---

## FOR THE OWNER

Two things, and the first is a question only you can answer.

**1. The ~1.5 GB memory ceiling may no longer be the right number, and I am not
allowed to change it — nor should the loop be.**
`SYSTEM.md` caps this repo at ~1.5 GB RAM because the box serves paying tenants.
That constraint is currently **exceeded by 5× in normal operation** — `T2.00`
peaked at 7.57 GB today, `T0.07` records 6.99 GB on a green row — and it is
enforced by nothing. Two outcomes are legitimate and they point opposite ways:

- *The ceiling is right and the specs are in breach* — then B2's guard should
  eventually **gate**, and some specs need to shrink or move to Kaggle.
- *The ceiling was set for a smaller box and is now stale* — the box has 22.9 GB
  and free memory did not fall below 808 MB even at peak; no OOM has ever been
  logged; WorldTwin's watchdog is container-scoped and was never at risk.

I have deliberately **not** proposed a new number, because a default that raises
a safety ceiling would be exactly the "widening what is allowed" that
`SYSTEM.md` forbids a default from doing. This needs your ruling, and until it
comes the honest posture is to *measure and report* (B2), not to gate and not to
relax. Nothing is blocked on your answer — the loop keeps running either way.

**2. `D10`'s seat is now uncontestable, and you may want to know that the
project's one-brain falsifier is behind it.**
On 2026-09-01 a pre-registered default seated `wm-latent` as the learning core
**BY VERDICT** off a VOID run, justified by registering `LC.07` in the same
commit as its challenger. `LC.07`'s pilot came back 22 hours later projecting
~526 wall-hours against a 30 h/week free quota; it cannot run. All three
pre-registered re-open triggers are now closed doors (FINDING 1). Seven
`GOAL.md`-cited specs are welded behind it, including **`DP.02`** — the lesion
test `GOAL.md` names as *"the connectedness claim that can quietly fail"*.

The system foresaw the VOID-seating in writing, built the instrument that catches
it, and is routing the repair on clocked rows. It has not yet noticed that the
seat's escape hatches all shut. B1 makes that visible permanently. **No action is
required from you** — this is reported because a seat that cannot lose is the
condition your 2026-08-24 ruling exists to prevent, and you asked to be told
loudly when it recurs.

---

*Audit performed read-only. No spec, test, model file, or ledger row was
modified. The dirty `experiments/ledger.json` is the live gate sweep's own
incremental writes (pid 505077, declared) and was left untouched for the
harvesting iteration.*
