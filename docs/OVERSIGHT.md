# OVERSIGHT — 41st audit, 2026-08-28 06:45 UTC

## VERDICT: DRIFTING — **the eleven armed defaults have never been read by anything.** SYSTEM.md:130 says a default "may only pick among already-permitted actions" and that "`experiments/decisions.py` enforces this". Measured: `audit()` touches `default` exactly once, as `if not d.get("default")` — a non-empty-string test. The report then prints `r['default'][:110]`, so the constitutional content of every default is invisible to the tool *and* truncated out of the only report an overseer reads. I read all eleven in full for the first time today. **Four fail the invariant in ways I can measure**, and **nine of eleven cannot be executed by the organ that is instructed to fire them.** They fire in **4 days**.

**State.** `HEAD` is `3b59e0e`, the 40th audit's own commit. **Zero builder
commits since 2026-08-25 10:14:58**; last builder iteration ended
**2026-08-25 12:23:33 — 66.3 hours ago**; **0 iterations in the last 24 h**, all
31 hourly slots since the 40th logged `PACING: … skipping`. **84 PASS / 187
registered (44.9%)**, unmoved for 9 days. The untracked
`experiments/tests/sm_03_nose_reports_occluded.py` is still the only thing in
the working tree. Meters at 06:07 UTC: `week:all models` **69%** (the gate) at
**58%** of the week, line **63%**; `week:Fable` **100%** (not the gate). Kaggle
W34: **0.3111 h** charged, **29.6889 h** expiring **Sun 2026-08-30 00:00 UTC —
41 hours from now**.

**Clean results, each re-derived by me today rather than relayed from the 40th.**

- **§1 ledger integrity — clean.** `run verify` re-judged **83 PASS** entries and
  probed **81 controls**: 0 verdicts that no longer re-derive, 0 gates that
  ignore their control, 0 controls declared but never run, 0 gates unreplayable,
  0 entries unauditable, 0 controls run but undeclared. Two PASSes declare no
  control (`T0.01`, `T0.10`) — the standing §1.2 existence-claim note. `T0.18`
  self-excludes correctly.
- **§2 thresholds and controls — clean.** Seven-day `git log -p` over
  `registry.py`, `registry_expansion.py`, `experiments/tests/`. Every change is
  an addition carrying new `control=` / `null_baseline=` / `falsified_by=`
  (`f0cb81d` SH.02+SM.03, `ed2d969` the LG family, `f5d8f1c` T2.15's FAIL
  record). No threshold moved loose, no control deleted or weakened, no `_check`
  gained an `or`, no seed count cut, no assertion removed. I re-opened the one
  hunk that *reads* as a deletion and confirmed it myself:
  `- depends_on=["DP.00", "VO.01"]` → `+ depends_on=["DP.00", "VO.01", "LG.00"]`
  — `DP.04` **gaining** a dependency.
- **§2b deadlines — clean.** No `decide_by` has ever been moved; 11 `+ decide_by:`
  lines and zero `-` lines in 20 days.
- **§5 compute — internally honest, and nearly empty.** `overruns: []`. Every
  charged job reconciles against the `weeks` counter. W34's sole charge is
  `T2.15`, 0.3111 h, FAIL, harvested at `f5d8f1c`.
- **§7 bakeoff hygiene — no findings *in the record*.** `DECISIONS_RESOLVED.md`
  has not changed since 2026-08-12. No bakeoff has run. **But see RANK 1(c): an
  armed default is scheduled to write a §7 violation into that file
  automatically.**
- **The three gates all exit 0**, all run by me: coverage — 0 commitments with no
  declared spec, 0 CLAIM-DEAD, 4 known-dangling citations on the shrink-only
  baseline; decisions — 0/10 undeclared (nothing to arm, so the standing
  "arm one per audit" instruction is satisfied by an empty candidate set, not
  skipped); champions — 6/8 phantom arenas, 3 NO-ARENA, 3 UNCONTESTED,
  **byte-identical to the 40th's output**: none of B1–B6 was actioned, because
  the builder never woke.

---

## RANK 1 — the invariant that makes automatic firing safe is enforced by nothing, and four of the eleven defaults break it

This is the 40th audit's lesson one level up. The 40th found a *prose invariant
sitting next to a numeric guard* inside `champions.py`. The same shape governs
the entire escalation system, and this time the prose is in the constitution.

`SYSTEM.md:126–133`:

> *"…carries a `DECIDE:` block with a **default** and a **decide_by**; if the
> date passes unanswered the default fires… A default may only pick among
> **already-permitted** actions — never editing `GOAL.md`, never weakening a
> threshold, never widening what is allowed… `experiments/decisions.py` enforces
> this; the overseer runs it every audit."*

**It does not enforce this.** `decisions.py:194` is the only line in the module
that reads the field:

```python
missing = [k for k in ("default", "decide_by") if not d.get(k)]
```

A non-empty string satisfies it. `class`, `decide_by` and `blocks` are all
parsed and used; `default` is parsed and never inspected again — except at
`decisions.py:277`, which prints **`r['default'][:110]`**. Measured against the
live file, the eleven defaults run **369 to 1,041 characters**. So the report the
overseer is ordered to run every audit shows between **11% and 30%** of each
default, and the constitutional clauses are all past the cut. **No instrument and
no audit in this project's history has ever seen the text it certifies as
"armed".** I read all eleven today.

### (a) D8 — measured: firing it puts a constitutional sense into CLAIM-DEAD and takes `coverage --check` to exit 2

`balance` is GOAL.md:41, in the sense inventory. Coverage today:

```
balance   2 specs  0 pass  1 now   [support passing, not credited: BA.01 (sensor)]
          claims: BA.02 RUNNABLE
```

`BA.02` is the **only un-parked claim-kind spec** the commitment has. D8's
default parks it. Measured on the real `coverage.report()` rows, in memory:

```
BEFORE  _claim_dead(balance) = False
AFTER   _claim_dead(balance) = True      # BA.02 moved kinds -> parked
```

`coverage.check()` returns **2 if any commitment is CLAIM-DEAD**. So a default
that fires without human involvement on 2026-09-01 takes the project's
highest-priority constitutional gate red.

**The default asserts the opposite, in its own text:** *"the commitment `balance`
goes from 'has a runnable claim spec' to 'has none' — the ratchet SHRINKS, which
is why this branch and not option 2 or 3."* The ratchet it names does not shrink;
the CLAIM-DEAD count goes 0 → 1. That sentence is at character ~640 of a 758-char
default — 530 characters past where `--check` stops printing.

**And the default names two incompatible mechanisms for the same action.** Its
headline is *"PARK BA.02"*; its body is *"BA.02 is **re-parented** in the registry
behind the playground-humanoid line"*. These are different operations with
different outcomes: `_claim_dead`'s docstring says *"Blocked claims do NOT make a
commitment claim-dead — blocked is a queue position"*, so **re-parent → exit 0,
park → exit 2**. Whoever fires it chooses. Worse, the re-parent branch is not
executable as written: **"the playground-humanoid line" is not a spec id.** It
appears nowhere in the registry (grep: only in `DECISIONS_NEEDED.md` prose), the
registry has **0 dangling `depends_on`**, and the nearest real candidate — `PG.8`
— is not named. So the only mechanically executable reading of D8 is the one that
makes the gate red.

**This is the 28th audit's own lesson, re-committed by the 28th audit.** D8 was
armed at *2026-08-25 00:45 UTC (28th overseer audit)*. `coverage.py`'s docstring:
*"since the 28th audit also the case where every claim spec is PARKED (28th
audit: `shelter/building` and `thermal (kills)` both went claim-dead in one
commit when SH.01 was parked)"*. The same audit that taught the tool to detect
parking-induced claim-death armed a default that causes it.

### (b) D3 and D4 — "narrowing" is measured against observed behaviour, not against declared permission

Both defaults ratify an action already taken and call the result a narrowing.

- **D3**: *"FENCE THE OBSERVED PRACTICE… The loop may `git push` commits it
  authored to `origin/main`… This is a NARROWING of the current unbounded
  practice (146 logged pushes under no stated limit at all), it widens nothing
  that is permitted."*
- **D4**: *"RATIFY AND CAP. Option 1 … is recorded as TAKEN on 2026-08-13"* —
  ~90 core-hours spent 15 days before the deadline, against an escalation that
  was never answered.

Against the *practice*, both are narrower. Against the *permission* — which is
the word the invariant uses — both convert an unauthorised action into an
authorised one. That is the definition of widening what is allowed, and it is
exactly the move the invariant exists to prevent: **an escalation that is
ignored long enough becomes a default that legalises the thing that was
escalated.** Neither is malicious and both are arguably the right call on the
merits (D3's alternative costs the Kaggle quota, which is the standing §5
finding). The finding is not that they are wrong — it is that the invariant they
are measured against is the one nothing checks, and both of them state their
compliance with it in a sentence that redefines the baseline.

### (c) D10 — an armed default that seats a champion **BY VERDICT** off a **VOID**, and amends a spec's premise so a single arm counts as an arbitration

`LC.03`'s ledger status is **`VOID`** (commit `0d9ad54`). `SYSTEM.md:154`:

> `VOID     : an arm failed the learning gate; fix the arm, do not decide`

D10's default: *"wm-latent takes the learning-core seat as measured
winner-by-default (CHAMPIONS.md idiom, **seat marked BY VERDICT** with the
single-arm caveat on its face)"* — `BY VERDICT` being, per the 40th audit's own
framing, the strongest marking in the file. And: *"LC.04's premise is amended
from 'arbitrate among screened learners' to 'the screen IS the arbitration when
it returns exactly one'."*

Rule 4 is *"never weaken a threshold, loosen a control."* A bakeoff premise
amended so that **one surviving arm is an arbitration** removes the comparison
the gate is made of — `SYSTEM.md:157`: *"two non-learners cannot arbitrate an
architecture."* This is my §7 checklist item — *"a VOID treated as a verdict, or
a winner chosen inside the noise margin"* — pre-registered to happen by itself in
four days, and `champions.py` will print `Learning core BY VERDICT ok` afterwards
because it reads the table and has no notion of whether a verdict was earned.

To be fair to the entry: it does preserve the VOID in the ledger and keeps the
owner's scale-transfer guard as a precondition of *adoption*. The objection is to
**seating**, and to the premise amendment — neither of which any tool would flag.

### (d) Nine of the eleven defaults cannot be fired by the organ instructed to fire them

The firing instruction lives in the overseer prompt. The overseer's MAY-NOT list
forbids implementing or modifying any spec, test or model code, and its MAY list
is `OVERSIGHT.md`, append-`DECISIONS_NEEDED.md`, append-`LESSONS.md`. Against
that, what each default actually requires:

| | what firing requires | overseer may do it? |
|---|---|---|
| D1 | write a 4-arm bakeoff; correct CHAMPIONS.md | **no** |
| D3 | fence push authority (loop script / SYSTEM.md) | **no** |
| D4 | record the ratification in `DECISIONS_RESOLVED.md` | **no** (not on the MAY list) |
| D7 | registry note + CHAMPIONS.md, "in the same commit" | **no** |
| D8 | re-parent/park `BA.02` in the registry | **no** |
| D10 | mark the CHAMPIONS.md seat BY VERDICT; amend LC.04 | **no** |
| D12 | write guards into the `notes` of LC.04 and LC.05 | **no** |
| D13 | implement a change-gated no-op in `scripts/overseer.sh` | **no** |
| D14 | implement a pre-flight check in `scripts/ladder_loop.sh` | **no** |
| D9 | adopt nothing, re-run nothing | yes (nothing to do) |
| D11 | accept as-is | yes (nothing to do) |

**Only the two defaults that require no action are executable by the organ that
watches the clock.** The overseer can journal *"the owner did not rule by
2026-08-31, so the pre-registered default fired"* — and cannot make it true. A
default recorded as FIRED that nobody implemented is strictly worse than an open
decision, because `decisions --check` and every audit downstream will read the
entry as settled while the registry, `CHAMPIONS.md` and the ledger still describe
the old world. That is D1's disease with a green tick on it.

The only organ that can execute nine of eleven is the builder. It has been dark
**66.3 hours** and, per RANK 2, is not scheduled to wake before **2026-08-30
21:00 UTC** — leaving it roughly one working window before the deadline, and
nothing currently tells it any of this.

---

## RANK 2 — a day of data moved the crossing 17 hours later, past the GPU expiry, to within 8 hours of the week reset — and the margin is now smaller than one audit run

The 40th projected the gate opening Sun 08-30 ~04:00 from +6 pts/day. Measured
over the full day since, builder contribution zero throughout:

```
08-27T06  69 - 7 = 62% at 44%, line 54   ->  gap 8
08-28T06            69% at 58%, line 63   ->  gap 6
```

`pct` gains **7.0 pts/day** (not 6). `allow = 25 + ceil(0.65·elapsed)` gains
**9.29 pts/day**. Crossing `t = 6/(9.29 − r)` from 08-28 06:00:

| `pct` rate | gate opens | vs GPU expiry Sun 08-30 00:00 | vs week reset Mon 08-31 05:00 |
|---|---|---|---|
| observed **+7.0/day** | **Sun 2026-08-30 21:00** | **21 h too late** | 8 h of week left |
| the 40th's +6.0/day | Sun 2026-08-30 01:49 | 2 h too late | 27 h left |
| consumption stops | Fri 2026-08-28 21:30 | 26 h of week left | — |

**The new number is the margin, and it is inside the noise of this organ.** The
maximum `pct` rate that still crosses *before the week resets* is **7.26
pts/day**. Observed is **7.0**. Five Opus runs/day produced those 7 points
(≈1.4 pts each), so the headroom between "the builder gets 8 hours" and "the
held budget is destroyed unspent" is **0.26 points — about one-fifth of a single
audit run.** The pace line's own log message, printed 66 times since the builder
stopped, is *"budget held for later in the week"*. At the observed rate there is
no later in the week.

Two honesty caveats, unchanged and load-bearing: `week:all models` is a shared
pool the owner's own sessions draw on, so ≈1.4 pts/run is an **upper bound**; and
the counterfactual is unmeasured, which is why `SY.01` — ordered by the 37th,
extended by the 38th and 40th, still unwritten — is the instrument and rule 4
forbids me acting on the reasoning. **This audit is one of the five.**

**One thing the 40th counted but did not name — and it cost something today.**
`review.sh` is `37 6 * * *` and `overseer.sh` is `37 */6 * * *`. They collide at
**06:37 every single day** — verified in `ps`, both PIDs alive at the same
second, both Opus, both on the gated meter. That is not only a double draw on the
metered pool in one slot; the two organs share a working tree.

**Measured, on this audit's own commit.** I committed with an explicit
three-path `git add` — no `add -A` — and the resulting commit contained **six**
files: my three, plus `docs/PROGRESS.md` (515 lines), `docs/PROGRESS_LOG.md` and
`scripts/ladder_prompt.md` (80 lines), all authored by the Review still running
in PID 2289663 and **already in the index** when I committed. A named pathspec
does not protect you from another organ's staged work; `git commit` writes the
whole index. For ~90 seconds a mid-flight snapshot of the Review's output sat
under the 41st audit's commit message and verdict. I caught it, `reset --soft`'d,
unstaged the three foreign paths and recommitted; the commit now holds only the
three files I wrote, and the Review's changes are back in the working tree
unmodified for it to commit itself. Nothing was pushed at any point
(`origin/main` is still at the 39th).

This is the `add -A` ban's real failure mode and the ban does not cover it. Two
Opus organs, one tree, one index, one minute — the collision is a scheduling
accident that is one race away from an organ publishing another organ's
half-finished work as its own findings.

---

## RANK 3 — SM.03 is untracked for a fourth day, and the expiring hours are the job it wants

Unchanged from the 36th, 38th, 39th and 40th; restated only because the clock
under it is now 41 hours:

```
-rw-r--r--  32086  Aug 25 12:20  experiments/tests/sm_03_nose_reports_occluded.py   (untracked)
-rw-r--r--      0  Aug 25 12:21  /data/sm03_pilot_seed90.json.log
```

`smell` is constitutional — GOAL.md:46, *"olfaction finds food, fire and decay at
a distance and through occlusion — the sense that works when sight fails"*.
Coverage: 2 specs, **0 passing**, SM.02 PARKED, SM.03 RUNNABLE. That RUNNABLE is
computed from the committed registry; the 32 KB that would make it runnable is
in one place on one disk and in no commit. It is `GPU_SHORT` — the exact shape of
job the 29.69 expiring hours exist for. The iteration that wrote it closed
`rc=0` on a detached pilot that produced a 0-byte log: fourth sighting of that
shape (30th, 35th, 36th, this one).

---

## RANK 4 — §3 and §8: the ladder is honest, static, and bottom-heavy

PASS by tier, re-derived today — identical to the 40th, which is the point:

| tier | GOAL.md's name | PASS | registered | % |
|---|---|---|---|---|
| 0 | harness (**DONE**) | 29 | 29 | 100% |
| 1 | primitives (**DONE**) | 13 | 13 | 100% |
| 2 | capabilities vs null | 38 | 64 | 59.4% |
| **3** | **earn your parameters** | **1** | 15 | 6.7% |
| **4** | **unison** | **1** | 25 | 4.0% |
| **5** | **the claims — the thesis** | **1** | 35 | 2.9% |
| **6** | **a living Jack** | **1** | 6 | 16.7% |

42 of 84 passes sit in the two tiers GOAL.md already marks DONE. The four above
tier 2 are `T3.01`, `UB.9`, `TA.02`, `T6.03` and have not moved since
**2026-08-21 — eight days**. Coverage says it from the other side: **14
commitments have live claim specs and nothing passing** — touch, tool use,
proprioception, plasticity, sleep, fast/slow (8 specs, 0 pass), shelter, thermal,
smell, voice, balance, social, hunger/thirst, one-brain/unison (21 specs, 1
pass). Curiosity: 12 specs, 1 pass.

**§3, drift: none, because there was no builder work to drift.** Every commit in
the last 66 hours is an audit of the silence.

**§6, stuck decisions:** 0 MEANS-ESCALATED, 0 UNDECLARED, 0 OVERDUE today. But
the answer to *"was any owner-decision quietly acted on without being recorded?"*
is now **yes, prospectively and by design** — D3 and D4 are defaults that record
actions already taken (146 pushes; ~90 core-hours), and they will do so
automatically. RANK 1(b).

**§8, answered directly: no.** We are not closer to a curious humanoid than on
2026-08-25, and the green-tick list has not moved in nine days either. What the
last two audits have found is a consistent pattern that is worth naming as the
real answer to §8: **this system's failures have migrated from the science to the
instruments.** §1 and §2 are clean and have been clean for weeks — the ledger is
trustworthy. What is not trustworthy is the layer above it: three gates, of which
one ratchets a subset of what it prints (40th), and one certifies a field it
never reads and truncates in its report (today). The machine's self-knowledge is
the least-guarded thing it owns, and it is the thing every other conclusion rests
on.

---

## FOR THE BUILDER

**B0 (before anything else, and it is small). Read the eleven defaults.**
`python -m experiments.decisions` shows you 110 characters of each. Print them in
full — the RANK 1 findings are all past character 110. Four of eleven need
amending before 2026-08-31, and amending a default *toward* the invariant is a
tightening the ratchet always permits.

**B1. Fix D8's default before it fires.** RANK 1(a). It is the only one that is
mechanically load-bearing on a gate. Two acceptable repairs, both narrowings:
- **preferred**: change *"re-parented behind the playground-humanoid line"* to
  name a **real spec id** (`PG.8` is the only registry candidate), and delete the
  word "PARK" from the option headline so the mechanism is unambiguous. Then
  `balance` goes blocked, not claim-dead, and `coverage --check` stays 0.
- **or**: register a successor claim spec for `balance` in the same commit that
  fires it, exactly as `f0cb81d` did for `shelter`/`thermal` with SH.02+SM.03.
- Either way, **delete the sentence "the ratchet SHRINKS"** — measured, it grows
  the CLAIM-DEAD count 0 → 1.

**B2. Make `decisions.py` read the field it certifies.** RANK 1. Full content
checking is not automatable, but three cheap mechanical checks are, and each
catches one of today's findings:
- **`default` must not be truncated in the report.** Print it whole. The 110-char
  slice is why this went four days unexamined.
- **`DEFAULT-UNEXECUTABLE`**: extract spec-id-shaped tokens from `default:` and
  resolve them against `BY_ID`, exactly as `champions.py` does for arenas. D8's
  *"playground-humanoid line"* names no id at all; flag a `default` that
  prescribes a registry change and names no resolvable spec.
- **`DEFAULT-UNOWNED`**: a `default` whose text names a path under
  `experiments/` or `scripts/` is not dischargeable by the overseer. Print the
  owning organ per row. Nine of eleven are `builder`, and nobody knows it.
- Add all three to the `blocking` tuple, and ratchet `NO-DEFAULT` — still open
  from the 40th's B1.

**B3. Carry the 40th's B1/B2 forward unchanged** — `champions.py` must ratchet
`NO-ARENA` (baseline 3) and the sum `ARENA-MISSING + NO-ARENA` (baseline 9);
`decisions.py` must ratchet `NO-DEFAULT` (baseline 0); and both belong on the
ladder as `T0.23`, the sibling of `T0.21`. Today's finding is the same family and
strengthens the case: `T0.23` should also assert **P5 — a `default:` string that
names an unregistered spec makes `--check` fail**, and **P6 — the report emits
the whole `default`, asserted by length against the file.**

**B4. `OVERDUE` must be a violation, not a row field**, and needs a `fired:`
marker — 40th's B3, unchanged, and now sharper: with nine of eleven defaults
un-dischargeable by the overseer, a partially-fired 2026-09-01 is the *likely*
outcome, not the pathological one. Also split the dates by `costs N specs`;
shortening is always permitted, lengthening never.

**B5. Commit `experiments/tests/sm_03_nose_reports_occluded.py`** — named
pathspec, per the `add -A` ban — and **re-run the pilot for real numbers** before
freezing any gate. The 08-25 iteration summary's numbers were never produced;
the log is 0 bytes.

**B6. `rc=0` must stop meaning "I launched something."** Fourth sighting. Assert
the artifact is non-empty ~10 s after a detached launch, and record the check in
the handoff line.

**B7. `SY.01`, the three-arm pace-gate bakeoff**, still unwritten after four
audits ordered it. Arms: **A** gate as shipped; **B** `JACK_NO_PACE=1`;
**C** `pace_gate` added to `overseer.sh`/`review.sh`/`field_watch.sh` beside the
`usage_gate` line each already has.

**B8. Move one of the two `37 6` cron lines, and guard the index.** `review.sh`
and `overseer.sh` collided in `ps` today and the collision put three of the
Review's in-flight files into this audit's commit despite a named-pathspec
`git add` — because `git commit` writes the whole index, not your pathspec
(RANK 2, with the recovery). Two repairs, both small:
- move `review.sh` off `37 6` (a one-character cron edit; not a bakeoff), and
- have every organ that commits assert the staged set **equals** its intended
  paths before committing — `git diff --cached --name-only` compared to the
  literal list — and abort loudly otherwise. The `add -A` ban was written for a
  single-writer tree and does not cover a shared index.

## FOR THE OWNER

**One new thing needs your attention, and it has a date on it.**

**On 2026-09-01, eleven pre-registered defaults fire because nobody answered them
by 2026-08-31.** That mechanism is correct and it is the fix for D1's twenty
dark days. But the safety clause that makes it safe — *"a default may only pick
among already-permitted actions"* — is described in `SYSTEM.md` as enforced by a
tool, and **the tool never reads the default text**; the report even truncates it
at 110 characters, which is why no audit had read them until today. Reading all
eleven:

1. **D8 would take `coverage --check` to exit 2**, by retiring the only
   falsifiable claim behind `balance` — one of your named senses. Measured, not
   argued. It also names two incompatible mechanisms for itself and points at a
   "playground-humanoid line" that does not exist in the registry.
2. **D10 would seat the learning core BY VERDICT off a VOID bakeoff** with one
   arm, and amend `LC.04` so that a single surviving arm counts as an
   arbitration. `SYSTEM.md` says a VOID means *"fix the arm, do not decide."*
3. **D3 and D4 legalise actions already taken** (146 pushes; ~90 core-hours) and
   describe that as a narrowing — true against the practice, not against the
   permission.
4. **Nine of the eleven cannot be executed by the overseer at all** — they need
   registry, `CHAMPIONS.md` or script edits, which only the builder may make. On
   2026-09-01 you would get eleven defaults journalled as FIRED and at most two
   actually true.

**You do not need to rule on the substance.** The builder items B1/B2 fix all
four before the date, and amending a default toward the invariant is a
tightening. What would help is one line if you disagree with any of the above
readings — particularly D10, where seating a core off a VOID may be a call you
want to make rather than have made for you.

**The standing ask is unchanged and the clock on it is now 41 hours.** Rule
`D13`/`D14`, or say the free hours may go. A day of data made the arithmetic
worse, not better: at the observed consumption rate the builder's gate opens
**Sun 2026-08-30 21:00 UTC** — **21 hours after** the free Kaggle hours expire
and 8 hours before the week resets — where yesterday the estimate was 01:49. The
margin is now smaller than one audit run: **7.26 pts/day is the fastest
consumption that still lets the gate open before the reset; we are at 7.0.**

**The trend that should set the priority:** 8.82 → 22.11 → **29.69** free
GPU-hours expired unspent in three consecutive weeks, **60.6 hours in total**, on
a project whose standing rule is free compute only.
