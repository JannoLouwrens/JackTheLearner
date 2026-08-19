# OVERSIGHT — 19th audit, 2026-08-19 18:45 UTC

## VERDICT: ON TRACK

No claim on the ledger is impeached. I checked the hard directions and say so
plainly below rather than inventing concerns. The real findings are a
**remediation reported as complete that was not**, and a **science result that
is variance-dominated** — plus the standing structural fact that the two
commitments GOAL.md calls the north star rest on one certificate each.

Ranked by damage to the trustworthiness of the ledger:

1. **Four PASS certificates are still stale** after the 13:09 commit that
   reported the IMPL_DEPS cascade closed — including the ONLY credited passing
   claim for **curiosity**, and the spec that certifies GPU-hour accounting.
   Two of the four cost under 11 CPU-minutes to refresh. (B1)
2. **XL.01's attempt-2 FAIL exists only in the working tree** — the most
   valuable output of the day, one `git checkout` from gone. Same class as the
   18th audit's RANK 1, seven hours apart. (B2)
3. **XL.01's measurement is variance-dominated, not merely negative.** The v2
   design pilots read ratio **0.084** on worlds 0–2; the recording on worlds
   3–5 with the identical fixture read **1.003**. A 12× effect and no effect,
   same code. Attempt 3 as designed is a coin flip. (B3)
4. A control gate's **aggregation** was loosened between attempt 1 and attempt
   2 — pre-registered, threshold unchanged, retention-checked, and it bought
   nothing. Cleared, but on standing watch. (B4)
5. `overruns: []` in `gpu_budget.json` is not evidence that no quota was
   exceeded. W31 records 37.46 h against a 30.0 h ceiling. (B5)

---

## 0. Is the ladder the RIGHT ladder? — `coverage.py` GREEN (rc=0)

**0 of 23 commitments have no declared spec.** The 2026-08-10 class of miss is
closed and has stayed closed across five audits.

The honest number underneath it is unchanged: **16 of 23 commitments have specs
but nothing passing.** Seven have exactly one passing claim each — damage
(PS.03), memory-across-lives (ME.10), generality (T1.02), language (T2.06),
hearing (UB.9), curiosity (T2.08), one-brain/unison (UB.9 again).

Two of those seven deserve naming, because they are the ones GOAL.md puts at
the centre:

| commitment | specs | passing claims | note |
|---|---|---|---|
| curiosity | 12 | **T2.08 only** | and T2.08 is **stale** (B1) |
| one brain / unison | 21 | **UB.9 only** | UB.9 is *also* hearing's only pass — one certificate, two commitments |
| sight | 5 | **none** (T3.01, OP.01 unrun) | |

"He explores because he wants to… climb the ladder, fall, and learn from
falling" has one certificate behind it and that certificate is currently a
claim about older code. This is the fourth consecutive audit at which curiosity
has exactly one pass and no curiosity spec was worked on.

---

## 1. Integrity of the ledger — CLEAN

Checked mechanically over all **81 PASS** rows (`{'PASS': 81, 'FAIL': 4,
'VOID': 4, 'NOT_RUN': 0, 'BLOCKED': 0, 'ERROR': 0}` of 169 registered specs):

- **81/81 name a commit that still exists in git.** `git cat-file -e` on every
  `commit` field (dirty suffixes stripped): zero missing.
- **81/81 resolve to an implementation file** via `protocol.module_path_for(id,
  strict=True)` — the strict form, so a duplicate implementation would have
  raised rather than silently shadowed.
- **79/81 declare a control, define `_control()` in the module, AND their
  `_check` actually reads its control argument** — AST over the second formal
  parameter of `_check`, which is the direction `UndeclaredControl` cannot
  catch. Zero PASSes have a declared-but-unread control.
- **The 2 exceptions are T0.01 and T0.10**, harness specs that declare no
  control at all. They are not PASSes whose control went missing; there was
  never one to run. Same two as the 18th audit.

**No PASS is a claim without evidence.**

### Staleness — the one real defect (B1)

`run status` reports **9 stale claims + 1 pre-`impl_sha` content-stale**. Four
are PASS:

| spec | budget | why stale | what it holds up |
|---|---|---|---|
| **T2.08** Curiosity drives coverage | `cpu<10min` | `IMPL_DEPS=["playground.py", …]` — flipped by the shelters commit `761121a` (08-19 08:18) | **curiosity's only credited pass** |
| **T0.12** GPU-hour accounting | `cpu<1min` | `IMPL_DEPS=["experiments/gpu.py"]` — flipped by `25ca0aa` (08-19 07:39), the commit that added `compute_s` | the meter that section 5 relies on |
| T2.03 Pretrained vision | `gpu<20min` | `playground.py`, `UnifiedBrain.py` | sight support |
| T2.04 Behaviour cloning | `gpu<20min` | `TrainingPipeline.py`, `UnifiedBrain.py` | |

Plus five non-PASS (T2.05 VOID, LC.03 VOID, BA.02 VOID, T3.07 FAIL, T4.02 FAIL)
and T2.02 content-stale from before `impl_sha`.

None of these files was edited after its run — I checked: `t2_08_*.py` has no
commit touching it since 2026-08-13 02:37, three minutes before its `ran_at`.
They are stale **through declared dependencies**, which is the detector working
exactly as designed.

**The defect is the closure claim, not the staleness.** Commit `d671ee1`
(13:09) reads *"Stale-cert recovery chain landed: 17 specs re-certified fresh,
26 new PASS rows, zero verdict changes"* and the loop log records *"This closed
the overseer's B1"*. The chain's membership was enumerated by hand and did not
cover the cascade; `run status` still lists ten. Nobody re-queried the
instrument after the remediation. **T0.12 in particular is the certificate for
GPU-hour accounting, staled by a change to GPU-hour accounting code, and it
runs in under a minute.**

Nothing here impeaches a verdict — staleness is disclosed loudly by the
instrument, which is the system working. It is a bookkeeping debt reported as
paid.

---

## 2. Thresholds and controls, over time — NO SILENT LOOSENING (positively verified)

`git log -p --since="7 days ago" -- experiments/registry.py
experiments/registry_expansion.py experiments/tests/` covers **42 commits**. I
did not eyeball the diff; I parsed every `+`/`-` constant assignment per commit
and kept only names that were *both* removed and added with a different value,
or removed and not re-added. **Exactly four hits in seven days:**

| commit | change | direction |
|---|---|---|
| `265e683` 08-19 | `xl_01`: `ALIEN_MIN_DIST` 1.5 → **2.0** | **STRENGTHENED** — restores the first draft's value; the commit message and docstring confess that 1.5 was "the accommodation that disarmed the control" |
| `a703604` 08-13 | `t0_21`: `N_PROPERTIES` 8 → **9** | STRENGTHENED |
| `c1bd242` 08-12 | `ba_01`: `TF_SPREAD_MIN = 2.5` deleted | RENAME — same commit adds `TF_FALL_SPREAD_MIN = 2.5` and `TF_ABS_SPREAD_MIN = 2.5`; the value did not move, it split into two gates |
| `fe16e06` 08-12 | `ba_01`: `OMEGA0_LOG10` deleted | REDESIGN — BA.01 v4 pre-registration, journal design implemented verbatim |

Also checked and clean:
- **Seed counts:** `SEEDS = [0, 1, 2]` in every new file; no `seeds=` in either
  registry moved downward. One budget moved `CPU_LONG → CPU_DAYS` — *more*
  compute, not less.
- **`_check` gaining an `or`:** no gate in any `_check` became disjunctive. The
  `or`s in the diff are waypoint bookkeeping and `len(x) < 2 or len(y) < 2`
  guards that fail closed.
- **Assertions removed:** none. Every `assert` in the window is an addition.

**No silent loosening. The one gate-shaped change in seven days made a control
harder, and its author wrote down that the previous value had been a mistake.**

---

## 3. Drift from the goal — NONE, and the converse is the finding

Everything the builder touched in the last 24 h traces to a GOAL.md sentence:

| work | GOAL.md sentence it serves |
|---|---|
| T2.06 PASS (language–action alignment) | *"he learns words the way every child does: by hearing them used while things happen"* |
| SH.01 substrate, then parked on its oracle | *"too cold kills him"* / *"Cold nights teach shelter-building the way no scripted lesson can"* |
| XL.01 attempt 1 FAIL, attempt 2 FAIL | *"Death is not a reset; it is a page turn"* / *"Life N+1 must be measurably better than life N because of what life N recorded"* |
| T0.27 property 10, stale-cert chain, `compute_s` | *"protects the honesty of watching what happens when the three meet"* |

**Zero drift.** No work this cycle serves no sentence.

The converse is section 0's table and it is the harder answer: **curiosity,
all-senses fusion, and learning-by-living are precisely the neglected ones.**
Curiosity 12 specs / 1 stale pass. Unison 21 specs / 1 pass, shared with
hearing. Sight 5 specs / 0 passing claims. The easy wins are not being taken
instead — the builder spent the whole day on two zero-pass constitutional
commitments and got two honest negatives for it — but the neglect is real and
it is now four audits old.

---

## 4. Is the builder alive and productive? — ALIVE, recovered, +1 PASS

Since 2026-08-19T00:00 UTC: **12 iteration starts, 10 ended `rc=0`.** The two
that did not:

- `11:07` **rc=1** — session limit on fable, then opus, then sonnet; marked as
  a lost iteration and correctly inherited by the 12:07 run.
- `14:07` **rc=124** — 50-minute timeout mid-unit. The 15:07 iteration
  inherited the uncommitted XL.01 implementation cleanly and committed it
  before recording, per the dirty-stamp lesson.
- `04:07` **ABORT: usage unreadable — refusing to run.** Correct fail-closed
  behaviour, single occurrence, not repeated.

**PASS delta today: 80 → 81** (T2.06, landed 08:08). The loop resumed at
07:31 after **4d18h / 135 consecutive hourly firings** blocked at the 90% usage
gate (2026-08-14T13:23 → 2026-08-19T07:31). That outage is measured and
escalated in `DECISIONS_NEEDED.md`; the gate behaved exactly as specified and
logged every refusal.

No repeated identical failures. No paused loop nobody resumed. No iteration
aborting on load — box load ran 0.00–0.76 with 13 GB free all day.

---

## 5. Compute honesty — NO WASTE, and 29.7 h expiring in 4 days

`experiments/gpu_budget.json`, per week:

| week | hours | jobs | failed | produced |
|---|---|---|---|---|
| 2026-W31 | 37.46 kaggle + 7.75 colab | — | — | (historical) |
| 2026-W32 | 21.06 kaggle + 0.76 colab | 17 | 4 (0.12 h, **0.7 %**) | |
| **2026-W33** | **0.297 kaggle** | **1** | 0 | **T2.06 PASS** |

**Every GPU hour spent this week produced a ledger entry.** There are no GPU
hours with nothing to show for them. The waste is the opposite kind:

> **~29.70 h of the W33 Kaggle 30 h expires Sunday 2026-08-23 — four days
> away — with four zero-pass constitutional commitments queued (SM.02, TA.02,
> VO.02, and XL.01's successor).**

### B5 — the overrun detector has a blind spot

`gpu.py:448` flags `used > KAGGLE_WEEKLY_HOURS` on charge, and `overruns` is
`[]`. But W31 records **37.4554 h against a 30.0 h ceiling** — a 7.46 h
overrun the file does not flag. Cause: that figure did not accrue
incrementally. It went 0 → 37.46 in a single commit (`96aa771`, 08-08, *"fix
budget week-key collision"*, a %U/ISO week-key migration), so the charge path
that raises the overrun never saw the crossing.

No claim is affected — over-stating spend is the safe direction and the file
already documents its opening-balance convention honestly. But `overruns: []`
currently means "no *incremental* charge crossed the ceiling", not "the ceiling
was never crossed", and the largest crossing on record is invisible to it.

---

## 6. Stuck decisions — two open, one whose premise just flipped

`docs/DECISIONS_NEEDED.md`:

- **The plasticity fork** (strike option A / freeze-the-trunk vs narrow the
  PLASTIC-ONLY decree). Owner-only by construction — it asks what the decree
  means, which no bakeoff can answer. No new evidence today. Correctly still
  blocked.
- **Credits-vs-expiring-GPU policy**, raised 12:40 today by the 18th audit.
  **Its premise inverted six hours later:** the weekly meter reset and the
  13:09 iteration measured `week:all models` at 23 %, Fable 12 %. Credits are
  *not* the binding constraint this week; the expiring GPU quota is. The policy
  question stands for the next exhaustion, but the urgency has moved — this
  week the system can spend the 29.7 h if it chooses to. I have appended a
  dated evidence update under that entry rather than opening a duplicate.

Nothing blocked that a bakeoff could have settled. **Nothing quietly acted on:**
the 90 % gate was never touched — the loop stopped 135 times rather than run,
which is the strongest possible evidence the owner's threshold was respected.

---

## 7. Bakeoff hygiene — CLEAN

`docs/DECISIONS_RESOLVED.md`'s most recent entry (BLOCK vs NO-BLOCK on VOID
dependencies) is a model of the protocol, not a violation of it: the winner was
chosen on a **measured exposure count replayed from the ledger's own history**
(9 specs would have rested on a refuted foundation vs 0), the benefit side of
the loser was measured and read **zero**, the loser is named with what was right
in it, and a **re-open trigger** is attached to the quantity the decision rests
on. Its entire content is that a VOID must *not* be treated as a verdict, and
T0.08 property 6 now asserts that invariant executably.

No decision made without a learning gate. No VOID treated as a verdict. No
winner chosen inside a noise margin.

---

## 8. The honest summary — closer, but on the strength of three negatives, not the counter

**Are we closer to a curious humanoid that climbs the ladder, or only to a
longer list of green ticks?**

The counter moved **80 → 81** in a day. That is not the answer. What actually
improved is the system's ability to **be wrong cheaply**, and it demonstrated
that three separate times today:

1. **SH.01** — the must-succeed oracle reference read `z = 1.028` against a
   pre-registered gate of 3 at N=12000. The loop declined to spend ~80
   CPU-minutes on a registered run it knew could only VOID, and routed the
   finding to LC.04's design notes instead of onto this ledger. A builder that
   wanted a green tick would have run it.
2. **XL.01 attempt 1** — FAILed on its control (alien store recovered the
   speedup, 0.32 vs 0.75 required), and the commit message **confessed** that
   the constant weakened to make the control constructible was what disarmed
   it. That confession became `LESSONS.md` #123.
3. **XL.01 attempt 2** — the fixture was rebuilt so the control is
   constructible *as originally designed*, `ALIEN_MIN_DIST` was restored
   upward, the recording ran on **fresh worlds 3–5** rather than the worlds the
   design was piloted on, and it FAILed again — this time on the claim.

That third result is the most important number on the board:

```
search_time_ratio  1.0034 ± 0.671   (gate: <= 0.5)
carried_ttf2  19.87 s    vs   wiped_ttf2  27.27 s
carried_ltc    2.67      vs   wiped_ltc    2.00     <- the null reached criterion FASTER
ok_claim       0.333     (needs 1.0 — 1 of 3 seeds)
ok_ref 1.0, ref_fed_frac 1.0, c_fixture_ok 1.0, c_alien_ok 1.0   <- rig valid, control clean
```

The rig is sound and the control finally behaves (pooled alien/wiped = 2.24, a
foreign store genuinely hurts). And in a world where food layouts genuinely
differ across lives, **a carried diary bought nothing.** Attempt 1's apparent
5× speedup was shown by its own control to be a region prior, not memory.

So: *"Life N+1 must be measurably better than life N because of what life N
recorded"* has **no passing claim**, and the one run that looked like it did was
correctly refuted by the system's own control. That is the ladder telling the
truth about the thesis, which is what it is for.

**Against that, the honest debit:** we are *not* closer to a curious humanoid.
Curiosity has one certificate, it is stale, and no curiosity spec was touched
today, yesterday, or during the four-day outage. The ladder-and-apple standard
in GOAL.md — climb, fall, learn from falling, unprompted — has no passing spec
behind it. Sight has none. Unison has one, shared. **We are closer to a system
that cannot fool itself, and no closer to Jack.** For a project whose own
`SYSTEM.md` says the system is the deliverable, that is the intended trade — but
it has now held for four audits, and it stops being a trade if it becomes the
permanent shape.

---

## FOR THE BUILDER

**B1 (RANK 1) — finish the cascade, then verify closure by re-querying the
instrument.**
`run status` lists 10 stale entries, 4 of them PASS, after `d671ee1` reported
the cascade closed. Cheapest first:
- `T0.12` — `cpu<1min`. Stale via `experiments/gpu.py` (`25ca0aa`, `compute_s`).
  This is the GPU-accounting certificate; it is stale because GPU accounting
  changed. Re-run it first.
- `T2.08` — `cpu<10min`. Stale via `playground.py` (`761121a`). This is
  **curiosity's only credited passing claim**; while it is stale, `coverage.py`
  credits curiosity a pass that is about older code.
- `T2.03`, `T2.04` — `gpu<20min` each. Two of the 29.7 expiring W33 Kaggle
  hours; fold them into the next dispatch.
- Non-PASS, lower priority: `T2.05`, `T3.07`, `T4.02`, `LC.03`, `BA.02`, and
  `T2.02` (content-stale, pre-`impl_sha`).

The generalisable half: **a remediation is closed when the detector says so,
not when the planned worklist is finished.** `d671ee1`'s chain membership was
hand-enumerated; `stale_claims()` was never re-run after it landed. Re-run the
detector and paste its output into the closing commit message. (Appended as an
extension to `LESSONS.md` §"An additive edit to a declared dependency…".)

**B2 (RANK 2) — commit the ledger.**
`experiments/ledger.json` carries XL.01 attempt 2 (`+180/−45`, FAIL, `ran_at`
2026-08-19T18:35:10, clean stamp `265e683`) and nothing else. The 18:07
iteration ended at 18:24, eleven minutes before the run landed. Attempt 1's FAIL
is preserved in `history[]` per the T1.02 precedent — verified. Commit it before
anything else; a negative result this expensive should not live in a working
tree, and this is the second audit in a row to open with that sentence.

**B3 (RANK 3) — XL.01 is variance-dominated; do not launch attempt 3 at this
power.**
With the *identical* v2 fixture:

| worlds | carried_ttf2 | wiped_ttf2 | ratio |
|---|---|---|---|
| 0–2 (design pilots) | 4.2 s | 50.0 s | **0.084** |
| 3–5 (recording) | 19.9 s | 27.3 s | **1.003** |

A 12× effect and no effect from the world draw alone. The recording's own
`search_time_ratio_std` is **0.671 on a mean of 1.003** — the instrument cannot
resolve a 2× effect, let alone adjudicate one. Running fresh worlds was the
right call and it is exactly what caught this; the problem is that 3 seeds × 8
lives is not enough to gate a heavy-tailed search time either way.

Before attempt 3, price the power: pilot the carried/wiped contrast on **6–8
worlds at one seed each** and compute the between-world std of the ratio, then
choose `N_LIVES` and `spec.seeds` so the pre-registered gate clears that std by
the margin you intend. This is the same family as the aggregation lesson you
just wrote — the pilot must price the gate's *resolving power*, not only its
threshold and aggregation form. If the required N does not fit `CPU_LONG`, say
so and escalate rather than re-running the coin flip.

**B4 (watch, no action) — the alien gate's aggregation.**
Per-seed all-must-pass → pooled mean, between a FAIL and its retry, on the
control side (pooled is strictly weaker there). I examined it and it clears:
pre-registered at `265e683` *before* the recording run; threshold unchanged at
0.75; discriminating power retention-checked in the docstring *and* verified by
me (attempt 1's 0.32-on-every-seed alien pools to 0.32, still a FAIL); the
claim side stayed strict per-seed; `_check` fails closed on NaN and on a missing
`c_alien_ttf2_s`; `ALIEN_MIN_DIST` was strengthened in the same commit; and it
bought nothing — attempt 2 FAILed on the claim regardless. **No action needed.**
Recorded because this is now two consecutive attempts in which the alien control
was the component adjusted, and the third one should be justified by a
pre-registered power calculation (B3) rather than by another look at the data.

**B5 — make `overruns` mean what it says.**
W31 holds 37.4554 kaggle-hours against `KAGGLE_WEEKLY_HOURS = 30.0` with
`overruns: []`, because the figure was backfilled in one commit (`96aa771`)
rather than charged. Either run the ceiling check on reconciliation/migration
writes as well as on `charge()`, or have `Budget` report the ceiling breach at
read time so a week that exceeded quota cannot present as clean. Low severity —
no claim depends on it — but it is a guard with an unnamed condition, which
`LESSONS.md` already has a lesson about.

**B6 — spend the expiring quota.**
29.70 h of W33 Kaggle dies **Sunday 2026-08-23**; credits are healthy this week
(23 % / 12 % at 13:09). The zero-pass GPU-shaped picks are `SM.02`, `TA.02`,
`VO.02`, and the two stale GPU certificates in B1. On present course this is the
second consecutive week to close with the majority of a free GPU quota expiring
unused while sixteen commitments have nothing passing.

---

## FOR THE OWNER

**1. Nothing needs your decision to keep the ledger honest.** Integrity is
clean on every hard check: 81/81 PASS name a live commit and a real
implementation, 79/79 with a declared control have a `_check` that reads it, and
there was no silent loosening in seven days — verified positively by parsing
every constant change across 42 commits, not by reading the diff.

**2. Your 90 % credit gate cost 4d18h and 135 iterations, and it was obeyed
exactly.** That measurement is already in `DECISIONS_NEEDED.md` from this
morning. The update: **the meter reset and credits are no longer binding this
week** (23 % used). The question you were asked — what the loop should do when
credits exhaust while a GPU quota expires — is unchanged and still yours, but it
is not urgent this week. What *is* time-bound: **29.7 free Kaggle hours expire
Sunday 2026-08-23** and the loop is currently spending its hours on CPU specs.

**3. The scientific news is a negative result, and it is about your thesis.**
XL.01 — *"Death does not erase what he learned"* — failed twice today, honestly,
in two different ways. Attempt 1 looked like a 5× speedup from a carried diary;
its own control proved that was a map of where food generally is, not a memory
of where food was. Attempt 2 rebuilt the world so the control was fair, and the
carried diary then bought **nothing measurable** (ratio 1.003 where ≤ 0.5 was
required). I want to be precise about what this does and does not mean: the
measurement is too noisy to settle the question (B3 — the same code read 0.084
on one set of worlds and 1.003 on another), so this is *not yet* evidence that
cross-life memory does not work. It is evidence that **we cannot currently
measure whether it does**, and the next attempt needs more lives before it can
mean anything.

I raise it because it touches a constitutional sentence — *"Life N+1 must be
measurably better than life N because of what life N recorded"* — and because
the system did the honourable thing twice in a row without being asked to. No
decision requested.

**4. The standing debit, fourth audit running.** Curiosity has one passing
certificate out of twelve specs, and it is currently stale. Unison has one out
of twenty-one, and it is the same certificate that covers hearing. Sight has
none. The builder is not avoiding these for easy wins — it spent today on two
zero-pass constitutional commitments and took two honest failures for it — but
the ladder-and-apple standard on the front page of `GOAL.md` still has no spec
that passes. If that ratio has not moved in another week, it is worth asking
whether the curiosity specs are gated behind something the CPU envelope cannot
buy, which is what SH.01 and XL.01 both turned out to be.

*No threshold, gate, control, budget, or ledger entry was touched by this audit.*
