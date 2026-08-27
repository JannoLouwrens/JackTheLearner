# OVERSIGHT — 39th audit, 2026-08-27 19:00 UTC

## VERDICT: DRIFTING — the free GPU-hours this week are not "at risk", they are **arithmetically already lost**, and the two decisions armed against that loss are dated to fire **29 hours after it happens**. W34 will expire with 29.69 of 30 free Kaggle hours unspent — the third consecutive week and the largest (8.82 → 22.11 → 29.69, **60.6 hours in three weeks**), and the first full week under the pace gate built to prevent exactly this.

**State.** `HEAD` is `ca9c7fa` — the 38th audit's own commit. **Zero builder
commits since 2026-08-25 10:14:58**, which is also the last write to
`experiments/ledger.json` (56 h ago). Last builder iteration ended **2026-08-25
12:23:33**; every one of the **55 hourly slots** since logged `PACING: …
skipping`. **84 PASS / 187 registered (44.9%)**. Working tree still carries the
single untracked `experiments/tests/sm_03_nose_reports_occluded.py` (32,086
bytes, mtime 2026-08-25 12:20). Live meters read by me at 18:50 UTC:
`week:all models` **68%** (the gate) at **51%** of the week, line **59%**;
`week:Fable` **100%** (not the gate); both reset Mon 2026-08-31 05:00 UTC.

**Clean results, each re-derived by me rather than relayed from the 38th.**

- **§1 ledger integrity — clean.** `run verify` re-judged 83 PASS entries from
  the record and probed 81 controls: **0** verdicts that no longer re-derive,
  **0** gates that ignore their control, **0** controls declared but never run,
  **0** gates unreplayable, **0** entries unauditable, **0** controls run but
  undeclared. Independently of that tool I resolved every PASS against git and
  the registry: **0 of 84 PASS entries cite a commit that no longer exists**,
  **0 cite a spec absent from `BY_ID`**. Two PASSes carry no control (`T0.01`,
  `T0.10`) — long-declared existence claims, the standing §1.2 note, not new.
  `T0.18` self-excludes correctly.
- **§2 thresholds and controls — clean, and one hit inspected closely.** The
  seven-day scan over `registry.py`, `registry_expansion.py` and
  `experiments/tests/` shows every change to be an addition declaring new
  `control=` / `null_baseline=` / `falsified_by=` fields. One hunk in `ed2d969`
  *reads* as a dependency being deleted —
  `- depends_on=["DP.00", "VO.01"]` — so I opened it: the next line is
  `+ depends_on=["DP.00", "VO.01", "LG.00"]`. It is `DP.04` **gaining** a
  dependency, exactly as its commit message claims. No threshold moved in the
  loosening direction, no control deleted or weakened, no `_check` gained an
  `or`, no seed count reduced, no assertion removed.
- **§5 compute accounting — internally honest.** `overruns: []`. Every charged
  job reconciles against the `weeks` counter; there is no GPU-hour spent without
  a ledger entry to show for it. W34's sole charge is `T2.15`, 0.3111 h, FAIL,
  harvested at `f5d8f1c`. (W31's 37.4554 h against a 30 h ceiling is the
  *documented* pre-mechanism event that caused `overruns` to be added —
  `gpu.py:481` — not a live discrepancy.) The books are clean. There is almost
  nothing in them.
- **§7 bakeoff hygiene — no findings.** No bakeoff has run since the 29th audit.
  No decision made without a learning gate, no VOID treated as a verdict, no
  winner chosen inside the noise margin.
- **The three constitutional gates all exit 0**, all run by me: **coverage** — 0
  commitments with no declared spec, 0 CLAIM-DEAD, 4 known-dangling GOAL.md
  citations on the shrink-only baseline; **decisions** — **0/10 undeclared**
  after this audit's B-work (was 1/10; see RANK 4); **champions** — ratchet ok,
  6/8 seats with a phantom arena.

---

## RANK 1 — the loss is already determined, and the remedy is dated after it

Eight consecutive audits have reported that the builder is dark and that GPU
hours are expiring. **None has put the deadline of the fix next to the expiry
date of the thing it protects.** Those two numbers live in different files, and
neither of them is wrong, which is why the relation between them was invisible.

### The four clocks, all measured today

| clock | resets / expires | how I know |
|---|---|---|
| **Kaggle free 30 h, `2026-W34` — 29.6889 h unspent** | **Sun 2026-08-30 00:00 UTC** | `gpu.py:369` keys weeks `%U` (Sunday-start, chosen deliberately to match Kaggle); W34 = Sun 08-23 → Sat 08-29 |
| `week:all models` — 68%, the pace gate's meter | Mon 2026-08-31 05:00 UTC | live `claude_usage.py` read |
| `week:Fable` — **100%**, the builder's primary model | Mon 2026-08-31 05:00 UTC | live read |
| **D13 + D14 pre-registered defaults fire** | **2026-08-31** | `DECISIONS_NEEDED.md`, `decide_by:` |

**The free GPU-hours die 29 hours before either default fires.** On 08-31 both
meters reset, the pace gate opens by itself, the builder wakes without anyone
deciding anything, and every organ reports green — with W34's 29.69 hours
already gone. A late default does not fail loudly; it succeeds irrelevantly.

### When does the gate actually open? The arithmetic, and its honest range

`pace_gate` skips while `pct >= 25 + ceil(0.65 · elapsed)`. Now: `pct` 68,
`elapsed` 51, `allow` 59 — a **9-point** gap. `allow` gains **9.29 pts/day**
(the week advances 14.29 pts/day × 0.65). `pct` gains **4–6 pts/day** from
consumption the builder does not make, read off the log: 62% at 08-26 18:07 →
66% at 08-27 18:07 → 68% at 18:50.

- at **+4/day**: the gap closes in 1.7 d → **Sat 2026-08-29 ~10:00 UTC**, leaving
  ~14 hourly slots before expiry;
- at **+6/day**: 2.7 d → **after the quota is already gone**.

So the *optimistic* case is a Saturday-morning opening in which a spec must be
implemented, pre-registered, committed, pushed and dispatched inside ~14 hours,
by an organ that has not run in 55. The pessimistic case is that the gate does
not open this week at all. Neither is a plan.

### The series, and why W34 is the one that matters

| week (`%U`) | Kaggle charged | expired unspent |
|---|---|---|
| W32 (08-09 → 08-15) | 21.18 h | **8.82 h** |
| W33 (08-16 → 08-22) | 7.89 h | **22.11 h** |
| **W34 (08-23 → 08-29)** | **0.3111 h** | **29.69 h projected** |

Monotone, worsening, **60.6 free GPU-hours in three weeks** on a project whose
owner ruled free compute only. And W34 is the **first full week under the pace
gate**, which shipped 2026-08-24 for the stated purpose of keeping the loop
awake when the GPU quota expires (`lib_usage.sh:38–46`). I am not claiming the
gate caused the whole delta — the counterfactual is unmeasured and rule 4 binds.
I am reporting that the instrument built to stop this presided over the worst
instance of it, and that this is now three data points, not one.

**Escalated** to `DECISIONS_NEEDED.md` as *D13 / D14 — THE DEADLINE FALLS AFTER
THE HARM IT IS ARMED AGAINST*, asking the owner for **a date, not a decision**:
rule the two questions already on their desk before **Sat 2026-08-29 12:00 UTC**,
or say the hours may go — which makes the loss a choice on the record rather
than an artefact of arithmetic. I did not move `decide_by` myself: shortening is
a permitted tightening, but the clock is the owner's, and D1's whole repair was
that a deadline stops meaning anything once agents may edit it.

---

## RANK 2 — the only runnable claim spec for SMELL is untracked, and its sole evidence of ever having run is a zero-byte file

Carried from the 36th and 38th as "an orphan file". It is worse than an orphan.

```
-rw-r--r--. 1 opc opc 32086 Aug 25 12:20  experiments/tests/sm_03_nose_reports_occluded.py   (untracked)
-rw-r--r--. 1 opc opc     0 Aug 25 12:21  /data/sm03_pilot_seed90.json.log
$ ps -p 1552865   ->  (no such process)
```

The iteration that wrote it closed **`rc=0`** at 12:23:33 with, verbatim: *"The
pilot is a tracked background task — I'll be re-invoked when it completes (no
extra monitor needed; polling would be waste) … pilot running full-size on seed
90 (pid 1552865, ~667 MB, healthy)."* The re-invocation never came, because the
next 55 slots pace-skipped. **The log has been 0 bytes for 55 hours and no
result JSON was ever written** — the pilot produced nothing at all. This is the
`[s]`-tier shape already in `LESSONS.md`: a detached run's health is an artifact
check ~10 s after launch, never a pid.

Why this ranks above the pace-gate finding it sits beneath in age:

1. **SM.03 is the *only* RUNNABLE claim spec for `smell`**, a constitutional
   sense in GOAL.md ("olfaction finds food, fire and decay … the sense that works
   when sight fails"). `coverage` reports smell as 2 specs, **0 passing**,
   SM.02 PARKED. The whole commitment currently rests on 32 KB of work that
   exists in exactly one place and is not in git.
2. **It is `GPU_SHORT`** — precisely the kind of job the 29.69 expiring hours
   were for. The one implemented candidate for the expiring resource is the one
   piece of work not committed.
3. **`rc=0` again certified a dead pilot** (the 30th and 35th audits found the
   same shape on other runs). An exit code that reports "I launched something"
   as "it worked" is the loop's most persistent lie about itself, and it has now
   been observed three times.

---

## RANK 3 — the gate's incidence, confirmed, plus one correction that makes it sharper

The 38th's finding stands and I re-ran it: `grep -rn pace_gate scripts/` still
returns **exactly one call site** — `ladder_loop.sh:179`, the builder, the only
organ that writes to the ledger. `overseer.sh`, `review.sh` and `field_watch.sh`
run on Opus against `week:all models` and call only the 90% stop.

**One correction to the reasoning around it, which I checked rather than
assumed, and which cuts the other way.** It is tempting to read `week:Fable
100%` as "the builder cannot run anyway". It can. `ladder_loop.sh:45` sets
`FALLBACK_MODELS="opus sonnet"` and `:238` walks the chain on a per-model limit.
So the builder's *next* iteration runs on **Opus** — the same model, and the same
`week:all models` pool, as the three ungated auditors.

That makes the asymmetry worse, not better, and it is new information: from now
until the 08-31 reset, **all four Claude organs draw on the one metered pool, and
exactly one of them is subject to the gate rationing it.** The builder is not
merely ungoverned-by-the-right-meter (the 34th's reading, since superseded) — it
is now queued behind three organs for a pool it genuinely consumes, and each
builder iteration it does win will push `pct` back up and re-close the gate
behind it.

I am one of those three organs. This audit is Opus, it is ungated, and the meter
moved from 66% to 68% across the window in which I read it. Reporting that is
not self-flagellation; it is the measurement that makes RANK 1 credible.

**No action taken.** Arm C of the pace-gate bakeoff (`SY.01`, ordered by the
37th, extended by the 38th) is the right instrument and rule 4 forbids acting on
an auditor's reasoning. It is builder item **B1**.

---

## RANK 4 — the undeclared ratchet could never have reached zero, and every audit was told to make it

Every overseer is instructed: *"Arm at least one per audit; the ratchet may
shrink and may never grow."* For the last stretch there was exactly **one**
`UNDECLARED` candidate left, and **it is un-armable by construction.**

`decisions.py:parse()` keys a header with no `D<n>` prefix by
`title.split("(OPEN")[0].strip()[:52]`. For the physics-first entry that yields
the 52-character string `'Was physics-first retired by argument instead of by '`
— spaces included, trailing space included. The declaration grammar six lines
above is `_DECIDE = ^DECIDE:\s*([A-Za-z0-9._-]+)\s*$`, which cannot match a
string containing a space. Measured:

```
>>> bool(_DECIDE.match("DECIDE: " + key + "\n"))
False
```

**There is no text an auditor could have written that `parse()` would bind to
that candidate.** The tool would have printed `ratchet ok (1/10)` forever while
being structurally incapable of reaching 0, and the standing instruction named
an action its own parser forbade. This is the guard-of-the-guard failure mode
`coverage.py` has hit twice before ("nest" inside "ho-nest"; the `STALE` false
exoneration): *the instrument that finds gaps can have a gap, and it will
flatter you.*

**Resolved this audit, lawfully and without deciding anything.** The entry was
already ruled by the owner on 2026-08-09 — *"schedule the run after T2.01"*,
option (a) RUN IT — in bold, in its own body, and recorded in
`DECISIONS_RESOLVED.md:2557`. Only the `## ` header still said `(OPEN, owner)`,
and the scanner reads headers. The 32nd audit flagged this class on 2026-08-26
(*"the three UNDECLARED entries are ALL already answered, and have been
miscounted for 17 days"*) and closed two of three; this was the third, now at 18
days. A `RESOLVED` header recording the owner's existing ruling takes it out of
`candidates` via `_SETTLED`. **Ratchet 1/10 → 0/10.** No question was answered by
an agent, nothing widened, no threshold moved. The durable repair is builder item
**B2**.

---

## RANK 5 — §3 and §8: 84 green ticks, and 4 of them are in the tiers that contain the thesis

The honest summary demands a number, so here is one nobody has computed. PASS
entries by the tier of their spec, against what the registry holds:

| tier | what GOAL.md calls it | PASS | registered | % |
|---|---|---|---|---|
| 0 | harness — *"measurement works"* (**DONE**) | 29 | 29 | 100% |
| 1 | primitives — *"every part can learn"* (**DONE**) | 13 | 13 | 100% |
| 2 | capabilities vs null (in progress) | 38 | 64 | 59% |
| **3** | **earn your parameters** | **1** | 15 | 6.7% |
| **4** | **unison** | **1** | 25 | 4.0% |
| **5** | **the claims — the thesis itself** | **1** | 35 | 2.9% |
| **6** | **a living Jack** | **1** | 6 | 16.7% |

**Half of the scoreboard (42 of 84) sits in two tiers GOAL.md marks DONE.** The
four passes in tiers 3–6 are `T3.01` (ablate vision), `UB.9` (heard-not-seen
fusion), `TA.02` (one-trial taste aversion) and `T6.03` (cross-session
persistence). Their trajectory, reconstructed from the ledger's own git history:

```
Aug 04  0     Aug 12  2     Aug 21  4
Aug 08  1     Aug 19  3     Aug 27  4   <- unmoved for 6 days
```

Four in twenty-one days, ~1 per 5.25 days. **77 remain.** At the observed rate
that is over a year, and the rate is currently zero.

`coverage` says the same thing from the other side: **14 commitments have live
claim specs and nothing passing** — touch, tool use, proprioception, plasticity,
sleep, fast/slow (8 specs, 0 pass), shelter, thermal, smell, voice, balance,
social, hunger/thirst, and one-brain/unison (21 specs, **1** pass). Curiosity:
12 specs, 1 pass. These are exactly the three the audit brief warns are *"most
likely to be quietly neglected in favour of easy wins"*, and the measurement
confirms the warning.

**§3, drift: there is none, because there was no builder work to drift.** Every
commit in the last 55 hours is an audit or a Review of the silence. The system
is not building the wrong thing. It is not building.

**§8, answered directly: no.** We are not closer to a curious humanoid than we
were on 2026-08-25. We are not even closer to a longer list of green ticks —
the list has not moved in six days for the tiers that matter and eight for the
total. What has grown is the record *about* not moving: ten Opus documents in
48 hours, this one included. The machine's self-knowledge is excellent and its
throughput is zero, and `SYSTEM.md`'s own corollary is the right verdict on
that — *"when the machine is sufficient, PROVE it by throughput"*.

---

## FOR THE BUILDER

Ordered by what they cost. The first is the only one that can still save the
week; do it first even though it is the largest.

**B1 (the week). Run the pace-gate bakeoff `SY.01` with three arms, not two.**
Ordered by the 37th, extended by the 38th, still unwritten. Arms:
**A** = gate as shipped; **B** = `JACK_NO_PACE=1`; **C** = `pace_gate say ||
exit 0` added to `overseer.sh:45`, `review.sh:29` and `field_watch.sh:31`,
beside the `usage_gate` line each already has. Pre-registered metrics, all
already instrumented: builder slots run, ledger rows written, free GPU-hours
consumed before the Sunday expiry. **A and B both leave the incidence asymmetry
standing, so a two-arm race cannot return the repair its own subject names.**
The `SM.03` dispatch is the natural first payload if a window opens.

**B2. Make the undeclared ratchet reachable** (RANK 4). In `decisions.py`, one
of: (a) widen `_DECIDE` to accept a quoted title key —
`^DECIDE:\s*(?:"([^"]+)"|([A-Za-z0-9._-]+))\s*$` — and match it against the same
`[:52]` slice `parse()` computes; or (b) the cleaner fix, give title-keyed
headers a stable short alias the way `COVERS:` does, so a 52-character prose
slice never becomes an identifier. Add a known-answer check that arms a
title-keyed entry and asserts the ratchet drops — the bug is that this was never
exercised. Keep `BASELINE_UNDECLARED` at its current value or lower; it may
never grow.

**B3. Teach `decisions.py` that a deadline can be too late** (RANK 1). Add an
optional `harm_by:` field to the `DECIDE:` block and a violation
`LATE-DEFAULT` when `decide_by >= harm_by`. Entries with no dated harm declare
`harm_by: none` — an explicit "nothing expires" is a real answer and is what
makes the check cheap. This has a scar and a price tag: 29.69 GPU-hours, and
D13/D14 are the fixture. Seed the baseline shrink-only, as `coverage.py` does.

**B4. Commit `experiments/tests/sm_03_nose_reports_occluded.py`** (RANK 2)
before anything else touches the tree — 32 KB, untracked for 55 hours, the only
implemented claim for a constitutional sense. Its pilot never ran: the log is
0 bytes and pid 1552865 is gone, so **re-run the pilot and record real numbers
before freezing gates**; do not stamp a docstring from the numbers in the
iteration summary, which were never produced. Named pathspec only, per the
`add -A` ban.

**B5. `rc=0` must stop meaning "I launched something."** Third sighting (30th,
35th, this one). Before an iteration may exit 0 having handed off to a detached
process, it should assert the artifact is **non-empty** ~10 s after launch and
record the check in the handoff line. A 0-byte file 55 hours later is the whole
finding.

## FOR THE OWNER

**One thing needs you, and it needs a date rather than an answer.**

`D13` and `D14` are correctly armed with conservative, reversible defaults — and
`decide_by: 2026-08-31`. The free Kaggle hours they exist to protect expire
**Sun 2026-08-30 00:00 UTC, 29 hours earlier**, and on 08-31 both Claude meters
reset and the problem clears itself. **So both defaults will fire, appear to
work, and change nothing about a loss already taken.**

Ruling either question before **Sat 2026-08-29 12:00 UTC** is the last point at
which a decision can still buy dispatch slots inside this week. The options are
unchanged and the evidence is attached to each entry in `DECISIONS_NEEDED.md`;
nothing new is being asked of you. Saying "let the hours go" is equally fine —
it converts an arithmetic accident into a choice on the record.

**The number that should decide how much attention this gets:** 8.82 → 22.11 →
**29.69** free GPU-hours expired unspent in three consecutive weeks, 60.6 in
total, on a project whose standing rule is free compute only. The trend is
monotone and the mechanism built to reverse it is presiding over its worst week.

**Not asked, but you should know it:** of 84 passing specs, **42 are in the two
tiers GOAL.md already marks DONE**, and **4** are in tiers 3–6 — earn-your-
parameters, unison, the thesis, a living Jack. That count has not moved in six
days, and its long-run rate is roughly one per five days against 77 remaining.
The ladder is honest and it is very bottom-heavy.
