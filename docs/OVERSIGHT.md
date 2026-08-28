# OVERSIGHT — 40th audit, 2026-08-28 01:00 UTC

## VERDICT: DRIFTING — two of the three constitutional gates I am ordered to run first every audit **count only one of the violation classes they print**, and in both the uncounted class is one edit away from the counted one, in the direction that reads as progress. `champions --check` goes from `6/8` to **`ratchet ok (0/8)`, exit 0**, by doing the exact thing its own source comment forbids. Only `coverage.py` — of the three — has a spec on the ladder guarding it.

**State.** `HEAD` is `6ad0abc`, the 39th audit's own commit. **Zero builder
commits since 2026-08-25 10:14:58**; last builder iteration ended **2026-08-25
12:23:33**, now **60.3 hours** ago, and every one of the **60 hourly slots**
since has logged `PACING: … skipping`. **0 iterations in the last 24 h.**
**84 PASS / 187 registered (44.9%)** — unmoved for 9 days. Working tree still
carries the one untracked file, `experiments/tests/sm_03_nose_reports_occluded.py`.
Live meters read by me at 00:45 UTC: `week:all models` **68%** (the gate) at
**55%** of the week, line **61%**; `week:Fable` **100%** (not the gate); both
reset Mon 2026-08-31 05:00 UTC. Kaggle W34: **0.3111 h** charged, **29.6889 h**
expiring **Sun 2026-08-30 00:00 UTC — 47 hours from now**.

**Clean results, each re-derived by me rather than relayed from the 39th.**

- **§1 ledger integrity — clean.** `run verify` re-judged 83 PASS entries and
  probed 81 controls: **0** verdicts that no longer re-derive, **0** gates that
  ignore their control, **0** controls declared but never run, **0** gates
  unreplayable, **0** entries unauditable, **0** controls run but undeclared.
  Independently of that tool I resolved all 84 PASS entries against git, the
  registry and the filesystem: **0 cite a commit that no longer exists**, **0**
  name a spec absent from `BY_ID`, **0** lack an implementation file. Two PASSes
  declare no control (`T0.01`, `T0.10`) — the long-standing §1.2 existence-claim
  note, not new. `T0.18` self-excludes correctly.
- **§2 thresholds and controls — clean.** Seven-day scan over `registry.py`,
  `registry_expansion.py`, `experiments/tests/`. Every change is an addition
  declaring new `control=` / `null_baseline=` / `falsified_by=` fields
  (`f0cb81d` SH.02+SM.03, `ed2d969` the LG family, `f5d8f1c` T2.15's FAIL
  record). No numeric threshold moved in the loosening direction, no control
  deleted or weakened, no `_check` gained an `or`, no seed count reduced, no
  assertion removed. I re-opened the one hunk that *reads* as a deletion —
  `- depends_on=["DP.00", "VO.01"]` in `ed2d969` — and confirmed on the next
  line `+ depends_on=["DP.00", "VO.01", "LG.00"]`: `DP.04` **gaining** a
  dependency. `b624d78` enlarges T0.21 P6's removal set (strictly tighter).
- **§2b, deadlines — clean, and worth stating given the standing warning.**
  `git log -p --since=20d -- docs/DECISIONS_NEEDED.md` shows **11 `+ decide_by:`
  lines and zero `-` lines**. No audit has ever moved a `decide_by`. The rule
  that a deadline must not slide when it is reached has held.
- **§5 compute — internally honest.** `overruns: []`. Every charged job
  reconciles against the `weeks` counter; no GPU-hour is spent without a ledger
  row. W34's only charge is `T2.15`, 0.3111 h, FAIL, harvested at `f5d8f1c`.
  The books are clean; there is almost nothing in them.
- **§7 bakeoff hygiene — no findings.** `docs/DECISIONS_RESOLVED.md` has not
  changed in 10 days. No decision made without a learning gate, no VOID treated
  as a verdict, no winner chosen inside a noise margin — because no bakeoff has
  run.
- **The three gates all exit 0**, all run by me: coverage — 0 commitments with
  no declared spec, 0 CLAIM-DEAD, 4 known-dangling citations on the shrink-only
  baseline; decisions — **0/10 undeclared**, so there is nothing to arm this
  audit and the standing "arm one per audit" instruction is satisfied by an
  empty candidate set, not skipped; champions — 6/8 phantom arenas.
  **RANK 1 is about what those three exit codes do not mean.**

---

## RANK 1 — the ratchets count one class and print three, and the cheap edit migrates a seat from the counted class to an uncounted one

`champions.py:106–109` contains this warning, in its own source, above the
baseline it protects:

> *"It shrinks by REGISTERING the spec — never by deleting the arena reference,
> which would convert an ARENA-MISSING into a NO-ARENA and make the seat
> permanently safe instead of merely uncontested."*

**The check below it cannot detect that move. It rewards it.** `main()` computes
`missing = sum(1 for k,_,_ in violations if k == "ARENA-MISSING")` (line 449)
and compares only that to `BASELINE_ARENA_MISSING`. `NO-ARENA` is printed in the
violation list and counted nowhere, baselined nowhere.

**Measured, on the real parser, against the real `CHAMPIONS.md` and the real
registry** (in memory; nothing on disk was touched):

```
AS SHIPPED      {'UNCONTESTED': 3, 'ARENA-MISSING': 6, 'NO-ARENA': 3}   ratchet prints 6/8
AFTER DELETING THE 13 PHANTOM IDS
                {'UNCONTESTED': 3,                     'NO-ARENA': 8}   ratchet prints 0/8 -> exit 0
```

Deleting `D1.0 T2.21 LT.03 LT.04 PL.02 PL.00 PL.* W.1…W.7` — a single pass over
one markdown file, no code, no registry change — takes the tool to **a perfect
ratchet** at the exact moment five architectural seats become permanently
unfalsifiable:

```
+ Control architecture (D1)
+ Curiosity signal
+ World                                    (held BY VERDICT — the strongest marking in the file)
+ Audio encoder (world-sound -> brain)
+ DECIDED BY DECREE 2026-08-09: PLASTIC ONLY. NO FROZEN COMPONENTS IN JACK.
```

The Vision encoder seat does better still: it also names `T2.03`/`T3.01`, so
deleting `PL.02` moves it out of the violation list **entirely**. A green tool,
a cleaner file, and `SYSTEM.md`'s standing rule — *"no architectural seat may be
held without a registered, EXISTING challenger"* — silently unenforceable for
the plastic-only decree, the World seat and the control architecture.

### The same shape in `decisions.py`, also measured

`main()` (line 284) computes `undeclared = sum(... k == "UNDECLARED")` against
`BASELINE_UNDECLARED`, then blocks on `("MEANS-ESCALATED", "CLASS", "DATE")`.
**`NO-DEFAULT` is in neither set.** Its own message calls it *"a deadlock
generator"* — the D1 disease by name — and it is unratcheted and non-blocking.
I appended a goal-class entry with an empty `DECIDE:` block to the real text and
ran the real `audit()`:

```
kinds: ['NO-DEFAULT']
undeclared = 0/10  -> ratchet ok ;  blocking = []  -> exit 0
```

So the 39th audit's last `UNDECLARED` entry could have been cleared not by
arming it but by giving it a `class:` line and nothing else: ratchet 1/10 → 0/10,
exit 0, and the entry left in exactly the state D1 was in for twenty days.

### Why this is the top finding, and what makes it fair rather than manufactured

I am *instructed* to run these three tools first, above any question of whether
existing specs are good, and to treat their output as the highest-priority
finding. Their exit codes are the load-bearing claim of the whole audit. Two of
them are ratchets over a single class of a multi-class violation list, and in
both the uncounted class is reachable by a one-line edit that looks like
tidying. No agent has made either edit — this is latent, not damage — but the
next builder iteration told "shrink the champions debt" has a cheap wrong way to
do it that every instrument certifies.

**The repair already exists in this repo, which is why the omission is a
finding and not a wish.** `coverage.py` has exactly this exposure — delete a
commitment and its red disappears — and it is closed by a spec on the ladder:
**`T0.21` property P2, *"the commitment list does not shrink to make the ladder
look covered"***, asserting `CONSTITUTIONAL_GAPS_2026_08_10 ⊆ COMMITMENTS` and
`len(COMMITMENTS) >= 20`. Measured across `experiments/tests/`:

| gate | ledger-backed guard | shrink-by-deletion closed |
|---|---|---|
| `coverage.py` | **T0.21, PASS, 11 properties** | **yes (P2)** |
| `champions.py` | none — no test imports it | no |
| `decisions.py` | none — no test imports it | no |

Both carry an internal `_fixture()`, which is good and is not this. A fixture
proves the parser can see a violation; it says nothing about whether the
*counter* can be gamed by moving a violation between classes.

---

## RANK 2 — every armed decision in the project fires on the same day, and the tool exits 0 on the day they do

`decisions --check` prints eleven armed entries. **All eleven carry
`decide_by: 2026-08-31`.** Measured, not eyeballed:

```
distinct decide_by dates across all armed entries: ['2026-08-31']
cost: D1 38 | D4 8 | D10 8 | D3 D7 D8 D9 D11 D12 D13 D14 all 0
```

Three consequences, each checked against the code:

**(a) The deadline carries no information about cost.** `D1` blocks `T2.01`/
`T2.02` and costs **38 specs** — the curiosity family, the unified-brain family,
six of seven Tier 5 claims. `D7` is whether to keep a cosmetic UI coupling and
costs **0**. They are due the same hour. The `DECIDE:` block was invented so
silence would resolve; a single shared month-end date turns eleven independent
clocks into one calendar event, and the `costs N specs` column I am told to rank
by has no counterpart in the schedule.

**(b) Firing them is one organ's work in one slot, and 54 specs ride on it.**
Simulating the real `audit()` forward:

```
2026-08-31 : rows=11  OVERDUE=0
2026-09-01 : rows=11  OVERDUE=11  specs at stake = 54
2026-09-15 : rows=11  OVERDUE=11  specs at stake = 54   <- nothing changes
```

The instruction is *"fire the default, journal it loudly, say how to reverse
it"*. Eleven of those, several of which (`D1`, `D4`, `D10`, `D12`) rewrite the
premises of multi-spec bakeoffs, is not one audit's work — and **nothing records
which fired**, so a run that discharges three leaves eight OVERDUE and the
system in standing violation of its own "do not let a deadline slide" rule until
they are all done.

**(c) When that day comes, no exit code changes.** `overdue` is a *row field*,
never a violation (`decisions.py:209`), and `--check` returns 1 only on a grown
`UNDECLARED` count or `MEANS-ESCALATED`/`CLASS`/`DATE`. Confirmed at each
simulated date above: **`--check` exits 0 with eleven OVERDUE rows.** The entire
firing mechanism is one sentence in the overseer prompt. If an audit misses it —
and the 32nd audit found three entries that had been miscounted for 17 days —
nothing goes red.

**(d) A one-day off-by-one, stated because it will be quoted.** `overdue` is
`(today - due).days` tested `> 0`, so `decide_by: 2026-08-31` means the defaults
become due to fire on **2026-09-01**, not on the 31st. That is one more day
after the harm in RANK 3, not one less.

---

## RANK 3 — the crossing has moved past the expiry, and this audit series is now the binding constraint on it

The 39th gave a range: gate opens Sat 08-29 ~10:00 (optimistic, +4 pts/day) or
not at all. **A day of data has removed the optimistic arm.** Measured from the
log, builder contribution zero throughout:

```
08-27 00:07  62%     08-27 12:07  65%     08-28 00:07  68%     -> +6 pts/day, steady
```

`pace_gate` skips while `pct >= 25 + ceil(0.65·elapsed)`. `allow` gains
**9.29 pts/day**; `pct` gains **6**; the gap is **7** points now.

| scenario | gap closes | vs. GPU expiry Sun 08-30 00:00 |
|---|---|---|
| consumption continues at the observed +6/day | **Sun 2026-08-30 ~04:00** | **4 h too late** |
| consumption stops entirely | **Fri 2026-08-28 ~18:45** | 29 h of week left |

**The whole difference is the Opus organs, and the arithmetic is small enough to
state per-run.** Five Opus runs happened in the +6-point window (four overseer
audits at `37 */6 * * *`, one Review at `37 6 * * *`); the builder ran none.
That is ~1.2 pts each, and since `allow` rises 9.29 pts/day, **one point of
`pct` postpones the builder's wake-up by 2.6 hours — so each Opus audit costs
the builder roughly 3.1 hours of awake time.** Nine more are scheduled before
the quota expires: **≈11 points, ≈28 hours of delay, against 47 hours of
remaining week.**

Two honesty caveats, because this is an attribution and not a measurement:
`week:all models` is a shared pool and the owner's own interactive sessions draw
on it, so ~1.2 pts/run is an upper bound; and the counterfactual is unmeasured,
which is why `SY.01` (ordered by the 37th, extended by the 38th, still unwritten)
is the instrument and rule 4 forbids me acting on the reasoning. **This audit is
one of the nine.** I am reporting the number that makes me part of the problem
because that is what makes the number credible.

---

## RANK 4 — SM.03 is untracked for a third day, and its pilot log is still 0 bytes

Unchanged from the 36th, 38th and 39th, restated because nothing has touched it
and it is the only implemented candidate for the expiring hours:

```
-rw-r--r--  32086  Aug 25 12:20  experiments/tests/sm_03_nose_reports_occluded.py   (untracked)
-rw-r--r--      0  Aug 25 12:21  /data/sm03_pilot_seed90.json.log
pid 1552865 -> gone
```

`smell` is a constitutional sense — GOAL.md: *"olfaction finds food, fire and
decay at a distance and through occlusion — the sense that works when sight
fails"*. `coverage` reports it 2 specs, **0 passing**, SM.02 PARKED, `SM.03
RUNNABLE`. That RUNNABLE is computed from the registry, which is committed; the
32 KB that would make it runnable is not in git and exists in one place on one
disk. It is `GPU_SHORT` — the exact shape of job the 29.69 expiring hours were
for. The iteration that wrote it closed **`rc=0`** on a detached pilot that
produced nothing: third sighting of that shape (30th, 35th, this one).

---

## RANK 5 — §3 and §8: the ladder is honest, static, and bottom-heavy

PASS by the tier of the spec, re-derived today:

| tier | GOAL.md's name for it | PASS | registered | % |
|---|---|---|---|---|
| 0 | harness (**DONE**) | 29 | 29 | 100% |
| 1 | primitives (**DONE**) | 13 | 13 | 100% |
| 2 | capabilities vs null | 38 | 64 | 59.4% |
| **3** | **earn your parameters** | **1** | 15 | 6.7% |
| **4** | **unison** | **1** | 25 | 4.0% |
| **5** | **the claims — the thesis** | **1** | 35 | 2.9% |
| **6** | **a living Jack** | **1** | 6 | 16.7% |

42 of 84 passes sit in the two tiers GOAL.md already marks DONE. The four in
tiers 3–6 are `T3.01`, `UB.9`, `TA.02`, `T6.03`, and that count has not moved
since **2026-08-21** — seven days. `coverage` says it from the other side: **14
commitments have live claim specs and nothing passing** — touch, tool use,
proprioception, plasticity, sleep, fast/slow (8 specs, 0 pass), shelter,
thermal, smell, voice, balance, social, hunger/thirst, and one-brain/unison (21
specs, 1 pass). Curiosity: 12 specs, 1 pass.

**§3, drift: none, because there was no builder work to drift.** Every commit in
the last 60 hours is an audit of the silence.

**§8, answered directly: no.** We are not closer to a curious humanoid than on
2026-08-25, and not closer to a longer list of green ticks either — the list has
not moved in nine days. What has grown is the record *about* not moving. And the
finding this audit adds cuts at the one thing that had been growing: the
machine's self-knowledge is not as well-guarded as its output, because two of
the three instruments that certify it have never been tested for the thing they
themselves warn about.

---

## FOR THE BUILDER

**B1 (top). Ratchet the classes you print, in both tools.** RANK 1.
- `champions.py`: add `BASELINE_NO_ARENA = 3` beside `BASELINE_ARENA_MISSING`,
  count `NO-ARENA` violations the same way, and fail `--check` if **either**
  grows. Better still, ratchet the sum `ARENA-MISSING + NO-ARENA` as
  `BASELINE_UNCONTESTABLE = 9` in addition to the two individual counters, so
  migration between classes is caught even when both individual baselines are
  respected. Both may shrink, neither may grow.
- `decisions.py`: count `NO-DEFAULT` against its own shrink-only baseline
  (currently **0**) and add it to the `blocking` tuple. An entry with a
  `class:` and no `default:`/`decide_by:` must not exit 0.
- Do **not** change either markdown file to make a new baseline fit. The
  baselines are 6/8 and 3 (champions) and 0 (decisions) as measured today.

**B2. Put both tools on the ladder, as `coverage.py` already is.** Register a
sibling of `T0.21` — call it `T0.23 — the ratchets cannot be satisfied by
deleting the evidence` — with, at minimum, these known-answer properties, each
derived from a measurement in RANK 1 rather than invented:
- **P1** the gaming move: strip the 13 phantom ids from an in-memory copy of
  `CHAMPIONS.md`; `--check` must **fail**, not print `0/8`.
- **P2** a goal-class entry with an empty `DECIDE:` block must make
  `decisions --check` **fail**.
- **P3** the counted-class direction still works: registering a phantom arena
  must shrink the debt and pass.
- **P4** the P2-analogue: neither baseline may be raised — assert the literals
  against the values measured today, so a future edit that grows one is a red
  test rather than a commit message.
Keep both `_fixture()`s; they test the parser, this tests the counter.

**B3. Give the eleven deadlines different dates, ordered by `costs N specs`.**
RANK 2. Shortening is a tightening the ratchet always permits; lengthening is
not, and none of these may move later. `D1` (38) and `D4`/`D10` (8 each) should
not share an hour with `D7`/`D8`/`D9` (0). Then make `OVERDUE` a **violation**
in `decisions.py`, not a row field — `--check` must return 1 while any default
is due and unfired — and add a `fired:` marker so a partially-discharged day is
visible. Today `--check` exits 0 on 2026-09-01 with eleven defaults due.

**B4. Commit `experiments/tests/sm_03_nose_reports_occluded.py`** before
anything else touches the tree — named pathspec, per the `add -A` ban. Its pilot
never ran (0-byte log, dead pid), so **re-run the pilot and record real numbers
before freezing gates**; do not stamp a docstring from the numbers in the
08-25 iteration summary, which were never produced.

**B5. `rc=0` must stop meaning "I launched something."** Third sighting. Before
an iteration may exit 0 having handed off to a detached process, assert the
artifact is **non-empty** ~10 s after launch and record the check in the handoff
line.

**B6. `SY.01`, the three-arm pace-gate bakeoff**, still unwritten after three
audits ordered it. Arms: **A** gate as shipped; **B** `JACK_NO_PACE=1`; **C**
`pace_gate` added to `overseer.sh`, `review.sh`, `field_watch.sh` beside the
`usage_gate` line each already has. Metrics already instrumented: builder slots
run, ledger rows written, free GPU-hours consumed before expiry. A two-arm race
cannot return the repair its own subject names.

## FOR THE OWNER

**Nothing new is being asked. One number should change how much attention the
existing ask gets.**

The 39th audit asked you for **a date, not a decision**: rule `D13`/`D14` before
**Sat 2026-08-29 12:00 UTC**, or say the hours may go. That stands, and a day of
data has made it tighter — at the observed consumption rate the builder's gate
now opens **Sun 2026-08-30 ~04:00 UTC, four hours after the free Kaggle hours
expire**, where yesterday there was still an optimistic Saturday-morning case.
The lever is small and quantified: **each Opus audit run postpones the builder's
wake-up by about 3.1 hours**, and nine more are scheduled before the quota dies.
Pausing the audit series alone would open the gate **tonight, 08-28 ~18:45**,
with 29 hours of GPU week left. I cannot make that call — `D13` is exactly that
question, it is armed with a conservative default, and its deadline is after the
harm.

**The trend that should set the priority:** 8.82 → 22.11 → **29.69** free
GPU-hours expired unspent in three consecutive weeks, 60.6 in total, on a
project whose standing rule is free compute only.

**Two things you should know, neither of which needs an answer.**

1. **Eleven of eleven open decisions are due on the same day, 2026-08-31**, and
   the tool that watches them exits 0 when they all come due. `D1` alone costs
   38 specs. If you rule nothing, the system owes you eleven fired defaults in
   one audit slot, with no record of which fired — builder item B3 splits the
   dates and makes an unfired default go red.
2. **The three gates that certify this project's honesty are not equally
   guarded.** `coverage.py` has a spec on the ladder that stops it being
   satisfied by deleting the evidence; `champions.py` and `decisions.py` do not,
   and I measured that `champions --check` prints a *perfect* ratchet after a
   one-pass edit that makes the plastic-only decree, the World seat and the
   control architecture permanently unfalsifiable. Nobody has made that edit.
   The repair is B1/B2 and it is cheap.
