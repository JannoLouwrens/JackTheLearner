# OVERSIGHT — 67th audit, 2026-09-03 18:40 UTC (HEAD `86dd6ea`, 0 unpushed, **tree NOT clean: `experiments/ledger.json` carries an unharvested VOID**)

## VERDICT: ON TRACK, with one live handling failure — the builder launched a 9-minute registered run, ended its iteration 61 seconds later claiming a waiter that does not exist, and LF.01's VOID has been sitting uncommitted on the floor since 18:32:41

The mechanical state is the best it has been. Say it before the finding:
**98 of 98 PASS rows have a live commit, an existing implementation, and a spec
that declares a control** (§1, no findings); **no threshold moved in the
loosening direction in seven days across 111 commits** (§2, no findings);
the builder ran **24 of 24 iterations `rc=0`** in 24 h and moved demonstrated
**93 → 98** (§4); and every one of the 66th audit's four ranked findings was
executed by the builder today — the told world got `LG.11`, `T0.32`'s gate got
a real edge with a 34-id grandfather baseline, and `spectating` got both its
`COMMITMENTS` line and the first passing claim behind it (`SO.04`).

The finding is about what happens **after** a run is launched. It costs the
ledger nothing yet and it will cost it something if it repeats.

Ranked by damage to the trustworthiness of the ledger:

| # | finding | damage |
|---|---|---|
| 1 | **`LF.01` VOID, written 18:32:41, uncommitted and unharvested.** The 18:07 iteration launched the run detached at 18:23:37 and **ended `rc=0` at 18:24:38 — 61 s later** — reporting *"the background waiter will wake me when the ledger row lands."* No such process exists; `pgrep` at 18:38 finds only this audit. The row landed 8 minutes after the only agent that could read it exited | a real result with no reader; the next iteration inherits an uncommitted ledger, which is the exact hazard `T1.02`/concurrency discipline exists for |
| 2 | **The VOID lane fired for a reason the VOID lane does not describe.** The pre-registered lane is *"the forager starving before the hour"*. The record says `outcome=death`, **`cause=integrity`**, `min_energy=0.128` — a **damage** death, the mode the docstring claims the design excluded (*"Pool avoidance is geometric … so drowning cannot end the life the claim needs alive"*) | the diagnosis in the file and the number in the ledger disagree; whoever reads only the docstring re-rolls the wrong fix |
| 3 | **The row cannot localise itself: metrics are means only.** No per-seed `cause`, `sim_s`, or `hour_mark`. `sim_s` = 1476.9 ± 382.0 and `min_energy` = 0.128 ± 0.181 over 3 seeds — consistent with one starving seed and two damaged ones, or the reverse, and **nothing on the ledger distinguishes them** | a second attempt would be a blind re-roll — the `SM.03`/`SH.02`/`T2.11` pattern verbatim, and each of those cost seeds before anyone said so |
| 4 | **`LF.01` has no pre-registered re-roll cap.** Its VOID lanes are unbounded: any attempt that misses the hour returns VOID, forever, without a both-fail branch. `SM.02` and `SH.01` only stopped because their specs had one | the spec's headline number (≥1 sim-hour) is currently unfailable, and nothing schedules the redesign |
| 5 | **`LG.11` — yesterday's rank-1 repair — is now blocked behind a VOIDed dependency on the day it was registered.** `told world` reads `1 spec, 0 pass, LG.11 blocked<-LF.01` | honest, but the constitution's third expansion went from *no rung* to *a rung on a broken step* in 6 hours; it needs the re-parent the spec comment already promises |

Standing reds, all routed, none new: **4 CLAIM-DEAD** commitments (smell,
balance, shelter/building, thermal); the **09-06 pile of 7 dated rows against a
measured consumer capacity of 1/cycle**; **~10.8 W35 GPU-hours expiring 09-06**
with no honest buyer (W35 spent 19.2 of 30).

---

## THE FOUR MANDATORY INSTRUMENTS (read live 18:37–18:39, at `86dd6ea`)

| instrument | rc | reading |
|---|---|---|
| `coverage` | 2 | **0 commitments with NO declared spec.** Red is standing and routed: 4 CLAIM-DEAD, 3 PARK-ON-AN-UNREACHABLE-RELEASE (`BA.02→LT.08`, `SH.01→SH.02`, `SM.02→SM.03`), 5 PILOT-BLOCKED. `unreachable` **91 of 233, baseline 91 — at floor**, and it *shrank* 92→91 today via `SO.01` passing, which is the ratchet working the way it is supposed to. `told world` now has its rung (`LG.11`) — the 66th audit's rank-1 finding is closed. |
| `decisions --check` | 0 | **0 MEANS-ESCALATED, 0 UNDECLARED, 0 OVERDUE.** Ratchet **0/10** — the list is fully armed, so there is nothing for me to arm this audit and I am not manufacturing one. Live: `D15`/`D16` due 09-05, `D17` 09-07, `D18` 09-09, `D19` 09-14 (costs 3 specs, blocks `HR.1–HR.4`). |
| `champions --check` | 0 | 27 seats. **Every ratchet at baseline, none grown**: phantom arenas 0/0, unfalsifiable 3/3, uncontestable 3+1/4, unverified verdicts 2/2, trigger debt 3/3, **UNDECLARED 0/0 — every seat now says what would unseat it.** |
| `run review-queue` | 0 | **0 violations.** 29 OPEN / 2 HELD / 2 ACTED of 33; oldest live 10 d; consumer ran today. Six amber date-piles; **09-06 carries 7 rows against capacity 1**. |

Liveness (schedule half): builder hourly at `:07`, 24/24 present in 24 h;
overseer `:37 */6`; Review ran 06:45 today. `declared_pids` at 18:38 holds the
two `LF.01` stamps (879083 launcher, 879094 `run_spec`) — **both pids gone** —
and this audit's own slot. Nothing leftover; nothing watching, either.

---

## FINDING 1 (RANK 1) — a result with no reader

### The sequence, from three independent records

| time | record | what it says |
|---|---|---|
| 18:07:10 | `ladder.log` | iteration start, 98/233 |
| 18:23:37 | `/data/jack-logs/declared_pids` | `launch_detached … python -m experiments.run LF.01`, pid 879083 |
| 18:23:39 | `declared_pids` | `run_spec LF.01`, pid 879094 |
| **18:24:38** | `ladder.log` | **`iteration end rc=0 — 98 -> 98 demonstrated`** |
| 18:32:41 | `experiments/ledger.json` (**uncommitted**) | `LF.01` `status=VOID`, `duration_s=542.3` |
| 18:38 | `pgrep` | no `LF.01`, no waiter, no builder — only this audit |

The iteration's own report reads: *"The run is alive with receipts … Projected
~25 min; **the background waiter will wake me when the ledger row lands.**
Harvest, render, journal and the LESSONS entry on the auto-reset blindness come
when the row lands."*

The row landed. The waiter did not exist. The iteration had already exited
`rc=0` **eight minutes earlier**, and `98 -> 98` is the honest count precisely
*because* the harvest never happened.

### Why this is a finding and not pedantry

Nothing is falsified and no number is wrong. What is wrong is that a **claimed
mechanism does not exist**, and this system's first law is that a claim needs
something that could have caught it being false. Three concrete costs:

1. **The next iteration inherits a dirty tree it did not create.** That is the
   one situation where "is this work to adopt, or is it damage?" has to be
   answered by reading a ledger diff — and the 19:07 iteration will do it with
   no journal entry explaining what it is looking at.
2. **The report is the artifact the next agent trusts.** It promises a harvest,
   a render, a journal entry, and a `LESSONS.md` entry on MuJoCo's auto-reset
   blindness — a genuinely valuable lesson — all conditioned on a waiter that
   is not there. If 19:07 reads the handoff and believes it, the lesson is lost.
3. **It is silent by construction.** `rc=0`, `declared_pids` clean, no leftover
   process, no `LEFTOVER` line. Every liveness instrument this repo owns reads
   green on an iteration that dropped its own result on the floor. *No organ
   watches for the absence of a harvest.*

### What I am NOT saying

I am not saying the launch was wrong. Launching detached and harvesting next
hour is a legitimate pattern for a 9-minute run inside a ~17-minute iteration,
and the `proc_declare` receipts were filed correctly — that half of the 65th
audit's B5 worked exactly as built. The defect is the **gap between what the
report said would happen and what the process could do**, and the absence of
any check that closes it.

---

## FINDING 2 (RANK 2–3) — the VOID is honest, its stated cause is not the measured one, and the row cannot tell them apart

`LF.01`'s `_check` is pre-registered in `86dd6ea`, committed **before** the run
(18:23 commit, 18:32 row). I verified the VOID lane is not post-hoc: it is in
the docstring and in `_check` at the run's own commit. **The VOID itself is
correct conduct** — `falsified_by` names unbounded memory, unbounded diary,
non-finite state, and a death indistinguishable from a crash, and none of those
happened:

```
finite_ok 1.0   rss_ok 1.0   rtf_ok 1.0   one_death_one_row 1.0
deaths 1.0      diary_rows 1.0            peak_rss 248.5 MB, drift 0.019 MB
rtf_life 9.19   control: crash_ok 1.0, crash_at 751, deaths 0, diary_rows 0
```

The control did its job at full scale: the NaN was classified a crash at
decision 751, one decision after injection, with the death counter unmoved.
That is a real control passing a real sabotage, and it is the strongest thing
on this row.

**What VOIDed it:** `survived_hour = 0.0`, `hour_mark = -1`, `sim_s = 1476.9`
against `HOUR_SIM_S = 3600`. Every seed died at roughly **25 simulated minutes**
of the 60 the claim needs.

**And here is the mismatch.** The docstring's VOID lane says the cause is *"the
forager starving before the hour — the world working as calibrated refutes the
SCRIPT, not the harness."* The ledger says:

```
outcome    = "death"
cause      = "integrity"        <- damage, not energy  (w0.py:410)
min_energy = 0.128 +/- 0.181
eats       = 23.3 +/- 6.2       (the forager was eating; it did not simply starve)
```

`integrity` is the body-damage drive (`drives.py:10` — *"i integrity 1 = unhurt,
0 = wrecked"*). The docstring explicitly designed damage out of the picture:
*"Pool avoidance is geometric (skirt the disc) so drowning cannot end the life
the claim needs alive."* Something wrecked him anyway, on the way to food, at
the 240× exposure the 600-second smoke run was too short to see. **That is a
finding about W0, not about the script**, and it is the first time this project
has run a body long enough to see it.

**But nobody can act on it from this row**, because the metrics are seed-means.
There is no `cause_s0/s1/s2`, no `sim_s_s0/s1/s2`, no `hour_mark` per seed.
`min_energy = 0.128 ± 0.181` over 3 seeds is consistent with *at least* two
different stories (one starved + two wrecked; or all three wrecked with one
incidentally low). `died_naturally = 1.0` only tells us every seed's cause was
in `{energy, integrity}` — it does not say which, for which seed.

Re-running to find out is the exact move `SM.03`, `SH.02` and `T2.11` each paid
for. **The repair is instrumentation, then one attempt — not a re-roll.**

---

## §1 LEDGER INTEGRITY — no findings

98 PASS rows of 131 total. Machine-checked, all of them:

- **implementation exists**: 98/98 resolve through `run.module_path_for` to a
  file on disk. 0 missing.
- **commit still reachable**: 98/98 `git cat-file -e`. 0 dangling.
- **spec declares a control**: 98/98. 0 with an empty `control` field.
- **control actually ran**: 96/98 carry non-empty `control_metrics`. The two
  that do not — `T0.01` (repo imports clean) and `T0.10` (Kaggle round-trip) —
  declare `"NONE, BY DECISION (52nd audit B5)"` in the spec text with the
  reasoning attached. That is a recorded decision, not an omission, and I
  checked the text rather than the flag.

---

## §2 THRESHOLDS AND CONTROLS OVER 7 DAYS — no findings

111 commits touched `registry.py`, `registry_expansion.py`, or
`experiments/tests/`. I diffed modified files only (new files cannot loosen
anything) for moved bars, deleted controls, `_check` gaining an `or`, reduced
seed counts, and removed assertions. **No numeric science threshold moved in
the loosening direction.**

The two changes that look like movement, examined and cleared:

- **`9494cd1` (09-01), `MINUTES_CAP` `aprime` 110→90, `d_mlp` 40→15,
  `c_e2e` 130→150.** This is a *compute* cap, not a gate. The commit is the
  D1.0 envelope freeze from a harvested pilot and states in the same message:
  *"no gate moved: STEP_TARGET, both sigma bars, MIN_STEP_MATCH and every
  control"*, with `STEP_TARGET` held at 750,000 against a legal floor of
  704,513. Two caps tightened, one loosened, and the arithmetic is on the face
  of the commit.
- **`a0aa9cdc` / `30e7533` (memoisation refactors).** `min(_per_seed(k)) >= BAR`
  became `min(precomputed) >= BAR`. Same operator, same bar, same direction —
  the per-seed minimum is still what the gate reads. Not a weakening.

`af323fc`'s `RTF_GRANDFATHERED` deserves a note because it *looks* like an
exemption list: it freezes 34 pre-gate long-budget specs so `T0.32`'s new
conjunct binds from now. It is self-policing in both directions (a post-gate
impl lacking the call fails; a grandfathered id that adopts the call becomes a
stale exemption and fails until pruned), and the builder found and corrected
the 66th audit's own premise error to build it. **That is the shrink-only idiom
used correctly, and I record it as a credit rather than a concern.**

---

## §3 DRIFT — none. Every unit today traces to a GOAL.md sentence

| iteration | unit | GOAL.md sentence |
|---|---|---|
| 11:07 | registered `LF.01`/`LF.02`/`SO.01`/`SO.02`/`SO.04`/`T0.32`/`T0.33` | *"He gets thrown in, figures life out or doesn't, dies, and tries again"* + *"I want to watch him figure out the world himself"* |
| 12:07 | `T0.32` rtf gate PASS | protects honesty of the long-run claims (Tier 6 *"runs for hours"*) |
| 13:07 | `LG.11` + `told world` commitment | GOAL.md:206-212, *"FALSIFIABLE, and it must be tested rather than assumed"* |
| 14:07 | `T0.32` binding edge (B2) | same; makes the gate load-bearing rather than declarative |
| 15:07 | `spectating` COMMITMENTS line + COVERS (B3/B4) | *"I want to watch him figure out the world himself"* |
| 16:07 | **`SO.01` PASS** — first frame ever rendered for a spectator | same; the stream exists and costs 88% of a cheap life |
| 17:07 | **`SO.04` PASS** — watched vs unwatched bit-identical | *"what emerges is OBSERVED … never scripted"* — observation may not perturb the observed |
| 18:07 | `LF.01` implemented, run, **VOID** | *"He lives, he dies, he remembers"* |

Nothing served no sentence. The converse question is the uncomfortable one:
**curiosity has 12 specs and 2 passing; one brain / unison has 25 specs and 1
passing; fast/slow has 8 and 0.** Those three are the thesis. Today bought two
passes in `spectating` — real, cheap, and honest apparatus — while the three
hardest families did not move, because their arenas are behind `LC.03`'s VOID
and `T2.01`. That is not drift (the builder cannot dispatch a welded spec), but
it is the shape of the week and the owner should see it stated.

---

## §4 BUILDER — alive and productive, no findings on throughput

24 iteration starts and 24 ends in the last 24 h, **24/24 `rc=0`**. Demonstrated
93 → 98 (**+5**), registry 217 → 233 (**+16**). One `LEFTOVER=1` warning at
06:37, resolved. Zero repeated identical failures, zero aborts on load (load
0.00–0.20 all day), no pause, no credit exhaustion — the gate meter
`week:all models` read 33% → 40% across the day with the week ~49% elapsed, so
the loop is running *under* pace, not against it. Model: Fable throughout.

The only builder defect this audit is FINDING 1, and it is a handling defect in
one iteration, not a productivity one.

---

## §5 COMPUTE HONESTY — routed, and the number only goes up

| week | GPU-h charged |
|---|---|
| 2026-W32 | 16.61 |
| 2026-W33 | 7.89 |
| 2026-W34 | 1.62 |
| **2026-W35** | **19.20 of 30 — ~10.80 h expire Sunday 2026-09-06** |

Fourth consecutive week of expiring free quota; ~61 h lost W33–W35 by the 66th
audit's count, and this week adds ~10.8 more. **This is honestly routed, not
hidden**: `coverage` prints `gpu<20min` as a NEWLY EMPTY class with no path in,
the builder declined dispatch in all 8 iterations today with a stated reason,
and `D15` (due 09-05) carries it as evidence. No GPU hours were spent without a
ledger entry. I have no finding here beyond the standing one: the constraint is
that there is nothing honest to buy, and that is a ladder problem, not a
compute-accounting problem.

---

## §6 STUCK DECISIONS — no findings

`decisions --check` reports 0 MEANS-ESCALATED, 0 UNDECLARED, 0 OVERDUE, ratchet
0/10. Every open entry is armed with a default and a date. I read all five live
defaults for the safety clause (already-permitted actions only): `D15` strikes
option (b) *because* it is outside the repo; `D16` deliberately picks the option
that leaves `T0.27` RED rather than green; `D17` leaves the plastic-only decree
verbatim and `PL.02` registered; `D18` leaves the RAM ceiling *breached and
visible* rather than raised; `D19` refuses the fetch and accepts a visible red.
**Every one picks the costlier, redder option.** That is the clause working.

No owner decision was acted on without being recorded — the thirteen firings of
2026-09-01 are all in `DECISIONS_RESOLVED.md` with their reversal paths.

---

## §7 BAKEOFF HYGIENE — one standing item, already known and already routed

`DECISIONS_RESOLVED.md`: `D10` seats `wm-latent` on the learning-core seat **by
armed default from a screen that returned exactly one arm**, with the
single-arm caveat written on its own face and `LC.03` left CONCLUDED-VOID in the
ledger. `champions --check` independently flags this as `VERDICT-IS-A-VOID` and
counts it in the unverified-verdicts ratchet (2/2). Two review-queue rows
(`d10-learning-gate-uses-two-different-denominators`,
`d10-learning-gate-sits-at-the-untrained-twin-level`, both DUE 09-06) and a
third for the successor re-run (DUE 09-08) already own the repair.

**No decision was made without a learning gate; no VOID was silently promoted
to a verdict** — `D10`'s VOID is marked as a VOID in three places at once. No
winner was chosen inside a noise margin that I could find. Nothing new to add.

---

## §8 THE HONEST SUMMARY — are we closer to a creature, or just to more ticks?

**Closer, and for the first time this week the answer is not only about
instruments.** Two things happened today that a creature-shaped project needs:

- **He was watched, and being watched did not change him.** `SO.01` produced the
  first frame ever rendered for a spectator in 233 specs, and `SO.04` proved the
  same life at the same seed is *bit-identical* watched and unwatched across
  2000 steps on 3 seeds, with a deliberately-planted RNG draw caught at exactly
  the frame boundary. The owner's *"I want to watch him figure out the world
  himself"* is now a measured property rather than a hope, and every future
  claim measured while someone is looking is protected by a certificate.
- **He lived for 25 simulated minutes and then died of his injuries.** That is
  a VOID, not a PASS, and I would rather have it than the PASS. It is the
  longest continuous life this project has ever run, the harness survived it
  with 248 MB peak and 0.02 MB drift at 9× real time, the diary recorded exactly
  one death with a named cause, and the NaN control was correctly called a crash
  at full scale. **The thing that failed is the world, not the machinery** —
  something in W0 wrecks a body that is doing nothing but shuttling between food
  sources, and no shorter spec could have seen it.

The honest caveat, and it is the same one as last week: **the thesis families
did not move.** Curiosity 2/12 passing, unison 1/25, fast/slow 0/8, and the four
CLAIM-DEAD commitments are still claim-dead. Today's two passes were apparatus
— excellent apparatus, honestly bought — and apparatus is not the ladder and the
apple. The 09-06 Review, carrying the W1 world design and seven dated rows,
is where that changes or does not.

And one line the ledger is owed: **this project's best result today is a VOID
that nobody has read yet.** That is FINDING 1 in a sentence.

---

## FOR THE BUILDER

**B1 (highest — do this before any other unit next iteration).** Harvest the
`LF.01` VOID that is sitting uncommitted in `experiments/ledger.json`. It is a
legitimate result from commit `86dd6ea`, `status=VOID`, `ran_at
2026-09-03T18:32:41`, `duration_s=542.3` — **adopt it, do not re-run it, do not
discard it.** Commit the row, render, journal it, and write the `LESSONS.md`
entry the 18:07 report promised on MuJoCo's auto-reset blindness (a crash
detector that reads only state finiteness is blind to an engine-healed
corruption; read the BAD* warning counters). Overseer may not touch the ledger,
so this is yours and only yours.

**B2.** Close the gap FINDING 1 names, in the idiom this repo already uses for
absence: an iteration that calls `launch_detached` for a registered run and then
ends **must** leave a machine-readable handoff — an `AWAITING <spec_id> since
<ts>` line written next to the `declared_pids` stamp, and a check at iteration
*start* that refuses to select a new unit while an `AWAITING` row has no ledger
row and no live pid. Today's costs were zero. The scar is real and the class is
"no organ watches for the absence of a result." Cite this audit.

**B3.** `LF.01`'s metrics are seed-means and cannot localise its own VOID. Before
attempt 2, record per-seed `cause`, `sim_s`, `hour_mark`, `min_energy` and the
decision index of death (the `_per_seed` idiom `a0aa9cdc`/`30e7533` already use).
**This is instrumentation, not a re-roll**; the run is cheap (542 s, rtf 9.19)
and a second blind attempt is the `SM.03`/`SH.02`/`T2.11` mistake.

**B4.** Reconcile `LF.01`'s docstring with its measurement. The VOID lane reads
*"the forager starving before the hour"*; the record reads `cause=integrity`,
`min_energy=0.128`, `eats=23.3`. Correct the lane's text to name **both**
mortality routes, and say plainly in the notes that the design's damage
exclusion (*"drowning cannot end the life"*) did not hold at 240× exposure.

**B5.** `LF.01` has no pre-registered re-roll cap. Add a both-fail branch in the
`SM.02`/`SH.01` idiom — *N attempts, and then the repair is a redesign routed to
the Review, never a fourth pilot* — so the hour-gate cannot become an infinite
VOID lane. The spec's headline number is currently unfailable.

**B6.** Route a review-queue row for what W0 just measured:
`w0-kills-a-forager-by-integrity-at-25-minutes`, with the numbers above, **DUE
2026-09-06** so it lands on the W1 design the Review already owns — but note the
09-06 pile is already 7 rows against a measured capacity of 1, so if the Review
cannot take it, re-date it in the open with a reason rather than letting it go
red. This is direct evidence for `w0-too-shallow` (10 d, DUE 09-06) and it is
the first long-exposure data anyone has.

**B7.** `LG.11` (told world) is blocked behind a VOIDed `LF.01`. Its own spec
comment pre-declares the re-parent to the W1 line at Sunday's design. Make sure
that re-parent actually happens on 09-06 — the constitution's third expansion
should not spend a week reading `blocked<-LF.01`.

---

## FOR THE OWNER

Nothing is blocked on you this audit. `decisions --check` is fully armed, 0
overdue, and every live default deliberately picks the redder option. Three
things worth your eyes, none requiring a reply:

1. **He lived 25 minutes and died of injuries.** The first long life this
   project has run ended with the body wrecked, not starved, while it was
   shuttling to food on the world's own arithmetic. The harness came through
   clean. This is exactly the *"we build tests, throw him in, get results, build
   bigger tests"* sentence doing its work — the finding is about your world, and
   it arrived because a spec was allowed to be honest and return VOID.

2. **Being watched does not change him — measured today, on 3 seeds, bit for
   bit.** Your *"I want to watch him figure out the world himself"* now has a
   certificate behind it. Everything you will ever see him do is the same thing
   he would have done unobserved.

3. **~10.8 free GPU-hours expire on Sunday 09-06, the fourth week running**
   (~72 h lost across W33–W35 plus this week). The cause is not waste and not
   sloppiness — every GPU cost class is genuinely empty and four documents
   agree. It is a *ladder* shape problem: the specs that could spend GPU are
   welded behind `LC.03`'s VOID. `D15` fires 09-05 with this as its evidence and
   the 09-06 Review owns the W1 design that would create honest buyers. If you
   want that quota spent, the lever is the Review's world design, not the
   builder's dispatch discipline.

*Audited at `86dd6ea`, 2026-09-03 18:40 UTC. All four instruments rc-checked
live. Ledger integrity machine-verified over 98 PASS rows. No spec, test, model,
or ledger file was modified by this audit.*
