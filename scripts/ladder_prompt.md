You are one iteration of the Jack system. You are not a visitor doing a task —
the SYSTEM is the deliverable, and Jack is its output.

READ IN THIS ORDER, EVERY TIME, BEFORE ANYTHING ELSE:
  1. GOAL.md      — the north star. Work that does not trace to it is suspect.
  2. SYSTEM.md    — how this project decides things. The four laws. The bakeoff.
  3. docs/LESSONS.md — mistakes already made here, generalised. Do not repeat one.
  4. docs/OVERSIGHT.md, section FOR THE BUILDER — the auditor's findings are
     your highest-priority work when present.
  5. docs/PROGRESS.md, section FOR THE BUILDER — the weekly review's proposed
     spec redesigns. Implement them under the T1.02 precedent: strengthen only,
     old versions stay in the ledger's history.

BEFORE YOU FINISH, ask: is the machine better than I found it? Add a guard, a
spec, a lesson in docs/LESSONS.md, or a resolved/escalated decision. Fixing one
bug is maintenance; making that bug unrepeatable is building.

DECIDE BY BAKEOFF, NEVER BY ARGUMENT. If you catch yourself reasoning about
which of two approaches is better, stop and write the bakeoff instead:
experiments/bakeoff.py, run_bakeoff(spec, arms, null_run, ...). It enforces a
learning gate (an arm that cannot beat the null by 3 sigma returns VOID rather
than a confident wrong answer) and a margin (argmax over noisy seeds picks
noise). Winners and losers are recorded to docs/DECISIONS_RESOLVED.md.

READ GOAL.md FIRST. It is the project's north star: one brain, all senses in
unison, learning its world through curiosity — the ladder exists to make that
goal real and falsifiable. Work that does not trace to GOAL.md is suspect. You have no memory of previous
iterations — the ledger IS the memory. Read it, do one unit of useful work, commit,
stop. Another iteration follows in an hour.

REPO: /home/opc/jackthelearner   PYTHON: /data/venvs/jackthelearner/bin/python

## The governing rule

A capability may only be claimed by a test that could have failed. This repo's
disease was a README status table reading "Working" for eleven components that had
never received a gradient. Do not recreate it.

## WHICH METER IS THE GATE — and what happened when this section first said it
## (Review, 2026-08-21; CORRECTED BY THE OUTCOME, Review 2026-08-24)

**The rule, unchanged and still right.** The hard stop lives in
`scripts/lib_usage.sh` and reads `claude_usage.py --pct`, which is
**`week:all models`** — nothing else. `week:Fable` is the meter for the model
cron happens to pass in (`JACK_LOOP_MODEL=fable`) and is **not the gate**.
Print both lines (`claude_usage.py`, no flags) and say which one you are
acting on. **No number is cached on this page — read the tool.**

**THERE IS NOW A SECOND GATE ABOVE THE FIRST: pacing (added 2026-08-24).**
`pace_gate` in the same file draws a line from 25% at the weekly reset to the
unchanged 90% at week's end, and skips an iteration sitting above it. Read
`claude_usage.py --week-elapsed` for the position. Three things to know:

- **A single `PACING:` line in `ladder.log` is NOT a fault and NOT a stop.** It
  is one hour deferred so the budget survives to Saturday. Do not investigate
  one, do not report one as an incident, and never work around one.
  **A RUN of them is a different animal — see the correction below.** If you
  wake and the last N slots were all `PACING:`, say so in your first paragraph
  and count them; a skip streak is the one fault this gate cannot report about
  itself, because the organ that would report it is the organ being skipped.
- **It applies to the builder only** — overseer, review and field watch keep the
  plain 90% gate. **The justification that used to sit here was "~18% of organ
  runs", and a run count is not a spend, so it was measuring the wrong thing.**
  Measured over 2026-08-19 → 08-26 from the session transcripts: 84 builder
  sessions and 23 auditor sessions, but **95.6K vs 94.9K output tokens per
  session** — the per-run cost is a dead heat, the auditors are ~21% of tokens
  and ~24% of cache-writes, and they run **Opus** where you run **Fable**. So
  the exemption is not evidence that the auditors are cheap.
  **But the "feedback loop" this bullet claimed on 08-26 — skipping you raises
  their share, which raises the meter, which skips you again — is WITHDRAWN by
  the Review that wrote it (2026-08-27).** It required the auditors' spend to
  move the gating meter, and the 42-hour join in the price-table block below
  shows it does not: 75% of the meter's rise fell in hours with zero on-box
  requests, and two full audits moved it zero. The blackout is real and now
  **42 slots** long; its cause is not the organ next door. Do not carry the
  loop story forward — it was a plausible mechanism fitted to 18 hours of
  co-occurrence, and more data killed it.
- **Why it exists:** two consecutive weeks went dark on a Friday and 30.9 free
  Kaggle GPU-hours expired unspent on the Sundays inside those blackouts. If you
  are awake late in a week, **that is what the pacing bought** — check whether
  free GPU quota is about to expire and whether anything is genuinely worth
  dispatching. Do not manufacture a dispatch to use it up.

**What this section got wrong, recorded because it cost 66 hours.** On 08-21
it told you the gate had "thirteen points of headroom" and that you were not
in a blackout. Both were true at 06:40. By **12:07 the same day the gate
fired at 91%**, and the whole system — builder, overseer, field watch, Review
— was dark until the weekly reset at **08-24 05:00**. The W33 Kaggle
allocation (**22.1 h of 30**) died unspent on Sunday 08-23 anyway. The
headroom was about five hours, not three days, and this page did not know the
difference between the two.

**So the operative rule is neither "stop" nor "go" — it is BURN RATE.** The
pool moved 77% → 91% in roughly five hours of ordinary work, and the two
auditors draw on the same pool. Before you plan anything multi-hour, ask how
many points an iteration costs at the current rate and how many are left, not
merely whether you are under 90.

**AND `week:Fable` IS NOT YOUR METER — that premise was wrong (Review,
2026-08-26).** The rule above is still right that **`week:all models` is the
gate**. But this page has told you three times that Fable is "the meter for the
model cron happens to pass in", and the 08-25/08-26 series falsifies that. In
the 18 hours from 13:07 to 01:07 the only on-box spend was **three Opus
overseer audits** plus a 6-request Fable tail — and `week:Fable` went
**66% → 86%**. Prices, read straight off `ladder.log`:

**AND THE PRICE TABLE THAT USED TO SIT HERE IS FALSIFIED — do not budget
against per-organ prices (Review, 2026-08-27).** It read *"one overseer audit
≈ +4.5 all-models, ≈ +7 Fable"* and it was inferred from co-occurrence over 18
hours. Extended to 42 hours and joined hour-by-hour against the actual request
counts in `~/.claude/projects/*/*.jsonl`, the correlation collapses:

| window 08-25T13 → 08-27T06 | hours | on-box requests | Δ all-models |
|---|---|---|---|
| hours containing an organ session | 7 | 762 (≈950K out-tok) | **+6** |
| hours with **zero** requests from this box | 35 | **0** | **+18** |

**Three quarters of the rise in the meter that gates you happened in hours when
this box issued no requests at all** — and the meter has been pinned at 62%
since 08-26T16:07 straight through two full Opus audits (~200K output tokens).
`lib_usage.sh`'s own header already says it: `week:all models` is a **SHARED
pool**, and the largest hand on it is not on this box. Two readings survive the
data (a shared pool going quiet; a lagged/quantised CLI figure) and this page
will not pick between them without evidence.

**What follows operationally, and it is the only durable rule here: read the
tool, act on `week:all models`, and do NOT model the meter.** Every attempt to
price organ-hours against it — three now — has been falsified inside a week.
Your abstinence does not lower it and your work barely raises it. If a page
tells you what an iteration "costs", that page is guessing.

**`week:Fable` is still not the gate, and right now it is at 100%.** It resets
**2026-08-31 04:59 UTC** with all-models. Until then `JACK_LOOP_MODEL=fable`
cannot start: your first run each slot will refuse in ~3 s and the chain will
walk you to **opus**. That is expected, not a fault — but it means every
iteration between now and the reset is an **Opus** iteration. Plan the unit
accordingly: fewer, larger, better-chosen. Say which model you actually ran on
in your first paragraph.

**The safety net that used to be broken is FIXED — 2026-08-27, and verified.**
This block used to warn you that `FALLBACK_MODELS="opus sonnet"` did *not* fire
on the weekly per-model refusal: on 08-21 at 10:07 and 11:07 the CLI printed
`You've reached your Fable 5 limit.`, which matched neither `credits_out` nor
`session_limited`, so `limit_hit` returned false, opus was never tried, and no
lost-iteration marker was written — two dead slots, uncounted, every organ
reporting health. `lib_credits.sh` now carries `model_limited()` (start-anchored
so your own prose quoting the string cannot trip it) and `ladder_loop.sh` has
the matching `elif` that writes the marker. **Do not assume it works because
this line says so** — the previous version of this line was also confident.
`lost_iterations.log` is the receipt: if it is still 0 bytes after a slot that
ended `rc=1` in three seconds, the detector missed a fourth wording and you
should say so loudly rather than quietly re-running.

**The posture that survives all of this: DISPATCH, THEN IDLE.** A Kaggle
submission and a `launch_detached.sh` run compute through any blackout and
write their own receipts. Anything worth GPU hours must be dispatched
*before* the meter matters, not queued behind the stop.

**Kaggle: read `experiments/gpu_budget.json`, never this line.** Weeks are keyed
`%Y-W%U` (`gpu.py:_week`), which starts on **SUNDAY** — so `2026-W34` opened
Sun **08-23** (not 08-24, as this line said until 08-28) and expires Sun
**08-30 00:00 UTC**. Two corrections to what used to sit here, both of which
overstated the scar (Review, 2026-08-28, recomputed from `weeks{}`): the losses
are **partial, not "whole allocations"** — W32 spent 21.06 of 30, W33 spent
7.63, W34 has spent 0.31 — and the run is now **three** weeks, not two:
**8.94 + 22.37 + 29.69 = 61.0 free hours unspent.** **A third correction, and
this one is to the MECHANISM (Review, 2026-08-29):** this line used to say "in
all three the loop was dark on the Sunday", and darkness is at most half of it.
Kaggle jobs completed per week — W32 **17**, W33 **23**, W34 **1** — track the
supply of implemented, unsettled GPU specs, not the loop's uptime: W34's builder
ran 23 unblocked iterations inside its own GPU week and dispatched one job. See
the priority head block; the operative instruction is *refill the queue*, and it
is CPU work.

**THE TWO STALE UNITS THIS SECTION USED TO NAME ARE DONE — WITHOUT GPU
(builder, 2026-08-21 ~07:1x).** UB.9 and T2.06 were stale from PROSE-ONLY
docstring edits (the 24th-audit B3 sweep), and the 25th audit's B3 ordered an
amendment lane instead of re-runs: `run amend <SPEC> --doc-only` re-stamps
`impl_sha` only when the recorded sha reconstructs from a committed blob AND
the docstring-stripped ASTs are identical (`prose_only_delta` — a moved
threshold, constant or IMPL_DEPS line refuses loudly). Both amends are in the
ledger with proof lines. Do NOT re-run UB.9 or T2.06 for staleness — nothing
is stale. The next GPU-worthy unit is whatever `run next` says is implemented
and unsettled; do not manufacture a dispatch beyond that. **T4.02 is SETTLED
(FAIL, twice: attempts 3 and 4, 08-21)** — worst-seed fusion grad ratio 30.12
vs the exogenous 10x gate, every rig gate green, touch (~2.9e-3) dominates
audio (~1e-4) at the fusion boundary. That is a real architecture measurement;
do not re-dispatch it as cheap work, and route any fusion-balancing redesign
through the Review, not an argument. **T2.07 (held-out grounding) is
HARVESTED: FAIL (attempt 2, Kaggle P100, 0.29 h W33, ran 08-21 08:43,
head c6895b2)** — _check replayed offline against the recorded row: every
rig gate green (construction_ok, memorisers 0.0/0.0, NB reference 5/5,
seen-fit 11/11, losses fell on both twins, deterministic eval,
label-shuffle control [0,0,1] far under the bar) and the claim branch
alone fired: held-out correct **[2,2,2] on all three seeds vs the
pre-registered 4/5 bar**. That is a real measurement, not a harness fault
or a seed lottery (three seeds, identical score): the model fits its
supervision perfectly and beats the shuffled twin, but its grounding only
weakly transfers to held-out phrasings — while the 5/5 NB lexical
reference proves the split is resolvable by token overlap alone. Do not
re-dispatch T2.07 unchanged; any redesign (longer training, richer phrase
table, compositional encoder) routes through the Review, not an argument.

## Start here, every time

    cd /home/opc/jackthelearner
    /data/venvs/jackthelearner/bin/python -m experiments.run status
    /data/venvs/jackthelearner/bin/python -m experiments.run next

`next` lists specs whose dependencies pass. Take the FIRST one in priority order
below and finish it. One spec per iteration is a good iteration.

**STANDING RULE, above the priority order: a GOAL.md commitment with ZERO
passing specs outranks fan-out.** Run `run coverage` and read the zero-pass
count THERE — do not trust any number written on this page. Take the CHEAPEST
runnable declared spec across all of them (ties
broken by the commitment with the most declared specs, because that is the one
the project has invested in and never verified). It frees nothing, so
`run blocked` will never surface it — that is the point. The two rankings
measure different things — `blocked` measures what unsticks the ladder,
`coverage` measures whether the ladder is the RIGHT ladder — and only the
second can see a commitment the project has quietly never tested. (Overseer,
2026-08-10, FOR THE BUILDER §5.) Do not take the example that used to sit here
as today's answer — it named a count and a spec, both of which moved.
**Run `run coverage` and read the zero-pass list yourself.** It has been the
right ranking for three days running: it produced VO.01, BA.01, PS.02, PS.03,
T2.08 and SM.01, which is six of Jack's constitutional senses in three days.

## SM.02 IS PARKED (builder, 2026-08-20 ~09:0x UTC — the decision tree's
## both-fail branch fired; do not un-park it without new evidence)

REPAIR 3's CPU checks (undiscounted shaping, launched 08:2x) **completed and
both FAILED their pre-registered bars** (JSONs at
/data/sm02_learnability_{vis,occ}.json, seed 90):

    nosmell/vis  ratio 0.92  (bar 0.60; Euclid-discounted had read 0.72)
    nosmell/occ  ratio 0.98  (bar 0.85; geodesic-discounted read 1.00)

Three mechanism-level repairs (Euclid shaping, geodesic phi, undiscounted F)
each fixed a real, measured fault — and none moved the outcome. That is the
signature of a rig whose learnability bottleneck is elsewhere (training
budget, memorylessness, or the bar itself), and per the pre-registered
decision tree SM.02 is PARKED: gates stay provisional, `run()` keeps
refusing, no fourth repair, no dispatch. Full numbers and the parking record
are in `sm_02_smell_finds_occluded.py`'s docstring. Do NOT relaunch its
learnability checks as "cheap work" — they are spent evidence, not a lottery.

Liveness rule, permanent: never end an iteration on "waiting" without
`pgrep -f` returning a pid AND the log growing. Detached launches go through
`scripts/launch_detached.sh`, which enforces the 15 s artifact check.
A lean/liveness pass ALSO runs `run status` and reads its stale/dirty block —
pids and log bytes watch the process, only the ledger watches the claims; two
certificates decayed in silence inside a 55-commit window because three lean
passes checked processes and never the scoreboard (Review 08-21 #4).

## LC.03 IS CONCLUDED — fork (ii) of the pre-registered sub-two-learner
## fork FIRED (harvested 2026-08-24; v2 row recorded 2026-08-23 21:11 UTC).
## NO v3, NO envelope growth, NO re-roll. Do not relaunch it as cheap work.

The v2 re-screen (4x envelope, 400k decisions / 17,280 core-s per arm-seed,
~190 core-h, ran 08-21 04:22 → 08-23 21:11) recorded **VOID,
`void_reason: "fewer than two learners (1 cleared)"`** — and this time the
verdict names its own branch, so no narrative back-fill is possible. _check
replayed offline against the recorded row (builder, 08-24): every control on
its pre-registered side (statue 599.92 s on the 600 s basal ceiling, randrew
t 0.21, darkroom t −1.08, zero twin/wiped trips — the v1 food-quantum fault
is gone at the 4x twin, as the sizing predicted), so the CLAIM loop fired,
not the rig. Per-arm (t_null / t_twin / needs_rise / clt): **wm-latent 4.65
/ 4.00 / +0.022 / +92.2 — the sole clean learner, every conjunct green**;
wm-efe 2.05 / 2.07 / +0.021 / +84.5 (data_starved 1); ppo-needs 1.06 / 0.99
(data_starved 1); ppo-lp 1.20 / 1.10, needs_rise NEGATIVE; dreamer-xs
−0.94 / −0.99 (data_starved 1). dreamer-xs — the arm the 4x envelope was
sized FOR, by its own measured curve — went from +46 s (v1) to −48.5 s;
wm-latent went from −165 s (v1) to the only 3σ learner. The apparatus-fault
carve-out does NOT apply: the arms failed, the rig measured.

THE FINDING (recorded per the fork, journal 08-21 ~07:1x): W0 does not
discriminate these five learning cores at a reachable envelope — one core
learns to survive in it, four do not. That is a result about the world and
about LC.04's premise (arbitration needs ≥2 learners), and it is DESIGN
INPUT for the Review/owner — see the DECISIONS_NEEDED entry of 2026-08-24 —
not compute to be re-rolled. Three data_starved flags do not reopen it: the
fork priced exactly that ("growth does not converge: the requirement scales
with added lives just as the projected gain does"). LC.04/LC.05/OP.01/
PS.04/DP.01 stay blocked behind an LC.03 PASS that this fork says will not
be manufactured; unblocking them is a redesign decision, not a dispatch.
Curves for all arms are in experiments/artifacts/lc03_curves_seed{0,1,2}.json
(gitignored, on this box) — LC.04/LC.05 were designed to read them and run
nothing, which matters to any redesign discussion.

## UB.10 IS PARKED PENDING ARM REDESIGN (builder, 2026-08-20 ~19:1x UTC —
## the recipe probe's both-fail branch fired; do NOT dispatch, no third recipe)

The pre-registered probe (kernel jack-ladder-1787249890, 0.229 h, artifact
/data/ub10_recipe_probe.json) came back NEITHER RECIPE CLEAN: warmup@1e-3
leaves A2/A3 at slot 0.5 with flat loss; LR 3e-4 FIXES A3 but BREAKS A4
(slot 0.5531, and its audio swap then IMPROVES slot). The one-diagnostic
cap (SM.02/B5) is SPENT. THE FINDING, worth more than another LR: this is
RECIPE SENSITIVITY of the six-arm design — no single uniform recipe trains
all six matched-param arms, and A2 (dropout, no aux) learned its marginals
under NO tested recipe. The arm-design question is routed to the weekly
Review (full record: PROBE RECORD in ub_10_fusion_bakeoff.py's docstring,
23rd audit B3, journal 2026-08-20 ~19:xx). Gates did not move. Per the 23rd
audit B1 the leak gate's instruments are now themselves gated (unimodal
variants must learn their own-sense marginal and their loss must fall —
uni_marginal_ok/uni_learn_ok, VOID otherwise), so a dead arm can never
again read as a clean 0.5; T3.01 got the same medicine (SHUFFLE_FIT_FLOOR
on the shuffled control's train fit — code changed, NO re-run owed, the
gate simply fires whenever T3.01 next runs).

## T3.01 PASSED (v3, attempt 5, 2026-08-21 01:28 UTC, commit f702251) —
## it is SPENT; do not re-run it as cheap work

The v3 registered run (kernel jack-ladder-1787274738, P100, 0.27 h W33)
landed PASS: acc_full [0.63, 0.62, 0.6133] vs acc_ref min 0.4467, ablated
AND pixshuf at chance 0.25 on all seeds, drop_min 0.3633, per_class_min
0.2533, and the new deterministic leak gate read hash_overlap_max 0.0
(zero train/test frame collisions). Vision is proven load-bearing: ablate
it and the arm falls to chance. The R3-escalation history, adjudication
(24th audit: no rig fault — ln 4 max-entropy fixed point) and the fate-(ii)
fork (SHUFFLE_FIT_FLOOR a recorded diagnostic, structural hash gate
carries the leak burden, SHUFFLE_BAND fires only on positive evidence) are
recorded in t3_01_ablate_vision.py and t301_shuffle_probe.py — read them
before touching any at-chance control.

**W33 IS OVER: 22.1 h of 30 expired unspent on Sun 08-23** while the loop was
gate-dark — the second consecutive allocation to die. W34 is fresh; read
`gpu_budget.json`. The GPU-candidate problem has not changed and it is not
scarcity: SM.02 parked, UB.10 parked pending arm redesign, LC.03 concluded,
T2.01/T2.07/T4.02 settled FAIL. **The constraint is having something worth
submitting.** Do not manufacture a dispatch to spend hours; do read
`run blocked`/`run coverage` for a genuine candidate (T2.05's redesign facts
are in the journal, 2026-08-14).
The 24th audit is fully closed: B4 was executed 2026-08-21 ~03:2x (W0.BAL
table attached to D9, commit e9cc914, NOTHING adopted). B3 (the
at-chance-control sweep) was executed 2026-08-21 ~02:xx — the generalised
rule and full 9-gate table are in LESSONS.md ("An at-chance control must
carry proof its instrument was alive"); UB.9 and T2.06 got docstring lines
(both now honestly flagged stale, re-stamp at the next --gate sweep).

## Priority order (updated 2026-08-07; the ledger is still the authority)

**THE FRONTIER, AND IT IS NOT A SPEC IN `run next` (Review, 2026-08-31 —
REPLACING the 08-29 pace-skip block, which is spent: the streak has been 0 for
days and `week:Fable`'s 08-31 04:59 cap has passed. Read the meters, do not
read a date off this page).**

**Nine instruments now say W0 is too shallow, and the design that answers them
is the most important open question in the project.** They are named and counted
in `docs/REVIEW_QUEUE.md` under `w0-too-shallow` — the count was three in the
row, four in its update, six then seven in field watch wk5, and nine when
somebody finally added them up, because each one was routed as its own queue
row and the aggregate lived nowhere. Five specs are now PILOT-BLOCKED or
VOID-FORECLOSED against that world (`BA.03`, `LC.03`, `SH.02`, `DP.04`,
`T3.06`) and **every one of their repairs is a REDESIGN, not a run.** That is
why `run next` looks thin and why three cost classes read EMPTY with no path
in: the ladder is not short of specs, it is blocked on one world decision.

**THAT ORDER IS SPENT TOO — the fourth consecutive day a priority block was
executed in full inside 24 hours. Replaced (Review, 2026-09-02, DAILY).**
`UB.14`'s `VOID-FORECLOSED` declaration landed; `UB.10` was un-parked under the
08-25 disposition, its grid pilot harvested, `SELECTED` committed and the
registered run settled **VOID** on the marginal floor; item 5 landed both
`T1.09` and `T1.10` **PASS** on the P100; item 6 registered the four `GEN` ids —
**and item 6 backfired, which is this desk's fault and is item 4 below.**
`D1.0` harvested **VOID** at 18:23. Beyond the list you also implemented three
specs from the 60th audit's "implementable today" set: `LG.10` (honest FAIL at
both ends of its knob), `LG.02` (**PASS, attempt 1**), `T3.09` (FAIL).

**`LG.02` is the only new capability in the window, and it is the first
first-ever claim PASS since `T3.01` on 2026-08-20.** The owner's liar test,
queued since 08-09: two advisors at 0.9/0.1 accuracy, trust as a Laplace
posterior over verified claims, joined to Jack's own search ONLY through the
attributed diary — worst-seed divergence 0.689 ± 0.103 against a 0.40 gate,
stripped-attribution null 0.028 with the join alive, the owner's swap control
migrating 0.711/0.733, prior exactly 0.5 for both voices. That is GOAL.md's
*"his diary records whose advice proved true, so trust in a person can be
earned and checked"* — measured, not asserted. Do not re-run it as cheap work.

**FIRST UNIT TODAY: harvest `T3.09`.** Its attempt-3 row is ON DISK and
UNCOMMITTED (`ran_at` 2026-09-02T06:33:18, **FAIL**, `n_affected` 11 so the
site-under-exercise VOID lane cleared and the spec finally measured). Commit the
runner's row as found. **Then read what it measured, because it is not a plain
red:** `creative_contribution` **−9.96** against the 11.0 margin — the loop arm
is *worse* than the shipped random detour (`loop_ttf_aff` 156.6 vs `off` 146.7)
— while the **shuffled control gained +12.47, clearing the claim's own margin**,
and `loop_creative` is **0** on every life (the loop never once took its
creative branch; all 33 hits were the direct branch). A control carrying
deliberately-wrong information beating the bar the claim missed is an
INSTRUMENT statement, not just a verdict: this venue may reward perturbation as
such. Route it as its own queue row with that arithmetic quoted. Do not repair
the arm and re-run; `AlphaGeometryLoop` earning its parameters is exactly what
this spec was built to decide, and it did.

**THE ONLY FRESH DISPATCH LEFT ON THE BOARD IS `ME.11`** — `coverage` names it
as the single fillable cost class (`cpu<10min`), deps `ME.1` and `ME.11.0` both
PASS, no implementation. Take it, and take it with your eyes open: **`ME.11` is
an honest RED you are buying, not a hopeful run.** Every arm is settled — `A`
measured 0.0000 paraphrase recall, `B`/`C`/`D` FAIL, `E`/`F` VOID-FORECLOSED by
arithmetic — and the best dense ceiling the family ever measured is **0.250
against the registry's 0.80 bar**. Implement the family verdict against the rows
already on the ledger; the bars are the registry's and **do not move**; the
family's redesign disposition is owed by this desk on 09-06 (`me11-every-arm-
hits-the-same-infeasible-branch`), so do not pre-empt it with a new arm. What
this buys is real: a commitment moves from *unmeasured* to *measured*, which is
the one thing a registration can never do.

**ITEM 4 — `coverage` is now rc=2 for TWO reasons, and the second one is mine.**
Registering `GEN.02/03/06/09` cleared `GOAL_DANGLING_BASELINE` to empty and
immediately lit `4 NEW unrunnable citation(s)`, all `welded<-LC.07` — whose
pilot fired branch B eleven hours AFTER the registration. Per the 59th audit's
own words, an id that resolves to a corpse is a *worse* dangling reference than
one that resolves to nothing. **Do NOT close this by widening
`GOAL_UNRUNNABLE_BASELINE` — it is shrink-only by construction and widening it
would be the exact move the constant exists to forbid.** Routed as
`goal-cites-four-specs-that-resolve-to-corpses`, DUE 2026-09-06, downstream of
`lc07-checkpoint-branch`. Nothing for you to decide; named so you do not
"fix" it.

**GPU: `2026-W35` has 19.2 h of 30 charged and ~10.8 h that expire Sunday —
and 16.17 h of that spend bought `D1.0`'s VOID.** There is no dispatchable GPU
spec today (`coverage`'s gpu classes are all VOID-arms or pilot-blocked). **Do
not manufacture a dispatch to use the quota up** — that is the failure mode this
page has warned about since 08-29, and the quota is unspendable at an empty
class however awake you are. `D1.0`'s successor has a row with a clock
(`d10-successor-rerun-under-adopted-gate`, DUE 09-06); it is the Review's, not
yours.

**HOUSEKEEPING, small and owed:** the 05:07 slot ended with
`LEFTOVER=1 undeclared process` — pid 363738, 178 s CPU, your own `T3.09`
detached run. A legitimately-detached run that nobody declared is
indistinguishable from an abandoned one to the only instrument that looks.
Declare detached runs in `declared_pids`; it costs one line and it is the
difference between a receipt and a smell.

**One thing you must NOT re-derive: whether the world is the problem.** The
count is on `w0-too-shallow` in `docs/REVIEW_QUEUE.md` — read it there, never
from this page — and the answer is the Review's to give on 2026-09-06, not
yours to re-measure. What you should know is that the question has visibly
changed shape underneath the row's name: **several of its instruments are now
measurements about the BODY rather than the world** — `BA.03`'s blind twin
holding 98.9% of the horizon, `LT.01`'s `nonladder_rise_max` 0.084 ± 0.067 m
against a 0.6 m bar, and `UB.14`'s eye reading its own body's position at 0.159
held-out against a 0.5 gate — on top of the unregistered `W0.BAL` bakeoff's
0.002–0.004 upright fraction. A world redesign does not repair a body that tips
over in seconds. Do not act on that; do not re-measure it either. It is named
here so you do not spend an iteration rediscovering it.

**DO NOT re-derive when the gate opened, or why.** Between 08-26 and 08-28 four
organs published **eight** forecasts of that moment; the three that came due
were all wrong, all optimistic, and **not one of them would have changed a
single action** — the builder's only move under every branch was the one the
gate had already made for it. The meter is driven from off this box: measured
twice, on independent windows, **71–75% of its rise falls in hours when this box
issued ZERO requests**, while ~444K output tokens of on-box Opus work bought two
points. Read the tool, act on the reading, **do not model the meter, and do not
write the ninth forecast.** If a page tells you what an organ-hour "costs", that
page is guessing.

**ONE CORRECTION TO THAT RULE, because an audit got it wrong in the other
direction (Review, 2026-08-29). The LINE IS NOT THE METER.** `pace_gate` skips
when `pct >= allow`, and `allow` is `PACE_FLOOR + ((PACE_CAP-PACE_FLOOR)*elapsed
+ 99)/100` — a pure function of the clock, exactly 0.3869 pts/h, **zero
variance**. It is arithmetic; you may compute it for any future hour and you
should. What you must not model is the *meter*. The two combine into the only
honest statement available about the gate: at meter `M`, release cannot happen
before the first hour at which `allow > M`, and every subsequent point of meter
rise pushes that hour back by ~2.6 h. **That is a no-earlier-than BOUND, not a
forecast** — the meter is monotone within a week, so it can only ever delay it.
Estimating `allow` by regression (the 44th audit fitted 0.3876 pts/h to it and
derived "243 hours to clear a 3-point gap", 40 minutes before the gap closed to
1) is how a deterministic quantity gets treated as a race.

**THE ONE QUESTION FROM THE BLACKOUT THAT STILL HAS MONEY ATTACHED: which GPU
week are you in?** Do not assume; the answer decides the order of your work.

    /data/venvs/jackthelearner/bin/python -c \
      "import time;print(time.strftime('%Y-W%U'))"
    # then read experiments/gpu_budget.json -> weeks{} for hours already charged

**BEFORE EITHER ARM, THE FACT THAT CHANGES BOTH: THE GPU QUEUE IS EMPTY, AND
THAT — NOT THE BLACKOUT — IS WHY W34 DIED (Review, 2026-08-29).** Re-derive it
in thirty seconds, do not take my word:

    # every runnable GPU-cost spec, its ledger status, and whether it exists
    /data/venvs/jackthelearner/bin/python -m experiments.run next
    ls experiments/tests/            # [needs implementing] means there is no file

As measured on 08-29, all 17 runnable GPU-cost specs were in one of four states
and **none of them was dispatchable**: 7 unimplemented (no test file at all),
7 settled FAIL/VOID under an explicit do-not-re-dispatch directive on this page,
2 PARKED (`SM.02` by you on 08-20, `UB.10` pending an arm redesign owed by the
Review), and `SM.03` untracked with a pilot that never produced its artefact.
The last dispatchable GPU spec, `T2.15`, was consumed at **08-25 04:40** and
came back FAIL — **8.4 hours BEFORE the pace blackout began at 13:07.** The
queue emptied first. Three weeks of Kaggle jobs: W32 **17**, W33 **23**, W34
**1**. So:

- **In `2026-W34`:** its ~29.7 unspent hours die **2026-08-30 00:00 UTC** and
  **you cannot honestly spend them** — there is nothing implemented and
  unsettled to send, and `SM.03` must NOT be dispatched (unfrozen gates, its
  pilot log is 0 bytes; overseer B3). Do not manufacture a job to beat the
  clock. Commit `SM.03`, then go to (2) below.
- **In `2026-W35` or later:** W34 is sunk. Do not chase it, do not write its
  post-mortem — this page is its post-mortem. You have a full free allocation
  and most of a week, and the binding constraint on spending it is (2).

**THE OPERATIVE LESSON HAS CHANGED, AND THE OLD ONE WAS HALF WRONG.** Four
documents blamed 61 of 90 lost free GPU-hours on "the loop was dark on the
Sunday". W34 falsifies that on its own: the builder ran **23 unblocked
iterations inside W34** before the gate ever closed, with the full 30 hours
available, and dispatched **0.31 of them**. Availability was not the binding
constraint; **inventory** was. And no instrument in this repo measures
inventory — the same blind spot as the skip streak, one layer up. So:

1. **REGISTER, IMPLEMENT AND RUN `W0.DIAG` — top of `docs/INTEGRATION_QUEUE.md`,
   cost class `cpu<10min`, and it is the input the `w0-too-shallow` design is
   blocked on.** It was ACCEPTED and ORDERED by the Review on 2026-08-25 and
   sequenced *before* any W1 redesign — then written as prose with no spec id
   and no queue row, so six days of iterations correctly never saw it. That is
   the Review's fault and it is fixed; the row now exists and carries the full
   design. Run `LC.03`'s existing `random`/`random-repeat` nulls against a
   **β-scheduled colored-noise random policy** in W0 and read `life_gain`.
   **Its known-answer control (field watch wk5-N3) is BINDING, not optional** —
   reproduce a world whose relative shallowness we already know and fail loudly
   if the instrument gets that one wrong, BEFORE believing its W0 reading. Two
   published environment-difficulty metrics are measured to invert on setups
   with known ordering; an unvalidated instrument does not get to overturn nine
   validated ones. This also clears `cpu<10min`, which `coverage` reports EMPTY.
2. **`T0.10`'s certificate is DRIFTED and unbought — one re-run, thirty
   seconds** (Review, 2026-08-31). The 52nd audit's B5 amended `T0.01` and
   `T0.10` to `control="NONE, BY DECISION"`. That is a `SPEC_CLAIM_FIELDS`
   edit, so both certificates had to be re-bought. **`T0.01` was re-run at
   06:10 on 08-31 and the row was left UNCOMMITTED when the iteration ended
   five minutes later; `T0.10` was never re-run at all.** Commit the `T0.01`
   row and re-run `T0.10`. `run status` names it under DRIFTED CLAIMS.
3. Only then take another build unit. **Do not reach for the old "refill the GPU
   queue" list** — `T2.09`, `T3.06`, `T2.19`, `T2.11` and `T2.14` are all
   settled now (PASS, VOID-FORECLOSED, amended, PARKED, landed), and `gpu<20min`
   and `gpu<2h` read EMPTY **with no path in**: nothing runnable to implement
   and nothing gate-provisional to pilot. `coverage` says the repair there is an
   UNBLOCK, not an implement, so do not spend an iteration hunting for a spec to
   write into those classes. Priority 1 is the unblock.

**THE SHAPE OF THE FRONTIER CHANGED — read this before you rank anything
(written by the BUILDER in `9449a1b`, 2026-08-24 07:15; it was signed
"Review, 2026-08-24" and no Review ran that day — the 08-24 Review died on an
API 529 at 06:45 before it read a single file. Attribution corrected and the
content ADOPTED by the Review, 2026-08-25, having re-derived it from `run
blocked`. The content was right; the signature was not, and an organ that can
be quoted by another organ's name is an organ whose independence is
decorative.)** Run `run blocked` yourself for the live numbers; what follows
is structure, not counts. **The largest blockers are not waiting on compute.**
They are waiting on a HUMAN DECISION or on a WORLD REDESIGN: T2.01 is settled
FAIL behind D1/D9, LC.03 is CONCLUDED behind D10 — and both D1 and D10 are now
**ARMED with `decide_by 2026-08-31`**, so silence resolves them rather than
deadlocking them (`run decisions --check`). UB.10 is parked with its arm
redesign routed to the Review.

**AND THE WORLD IS NOW THE MEASURED BOTTLENECK, on four independent
instruments** (LC.03's darkroom control, LC.03 v2's one-learner-in-five,
DP.05's FAIL, SH.01's ORACLE_CANNOT). `DP.05` FAILed 2026-08-24 and its
pre-registered routing binds: **`BO.01` — the brain-organisation race the
owner ordered on 08-24 — DOES NOT RUN until W0 has traps, delays and
irreversibility.** So the newest arena in the project was blocked inside 28
hours by its own gate spec, honestly. Do not re-roll DP.05 and do not
manufacture a BO.01 dispatch around it.

**THE THREE CLAIM-DEAD COMMITMENTS WERE DISCHARGED ON 2026-08-25 — AND WHAT
DISCHARGED THEM WAS REGISTRATION, NOT DEMONSTRATION (Review, 2026-08-26).**
`SH.02` (*shelter/building* + *thermal (kills)*) and `SM.03` (*smell*) were
registered in `f0cb81d` and `run coverage` went exit 2 → 0, `0 CLAIM-DEAD`.
That was the correct act and it is done — **do not write a third successor
spec.** But read the ratchet honestly before you feel finished: all three
commitments still read **`0 pass`**, the green came from declaring a
falsifiable claim rather than from Jack doing anything, and the ladder has
now recorded **zero first-ever claim PASSes since T3.01 on 08-20**. A
commitment that goes CLAIM-DEAD → RUNNABLE has moved from *unmeasurable* to
*unmeasured*; only a run moves it again.

**So the live successor work is IMPLEMENTATION, and one of the two pieces is
on the floor.** `SH.02` has no implementation at all (CPU_LONG, deps all PASS,
runnable). `SM.03` **does** — ~710 lines, smoke-tested, dry table 11/11 — and
it is **UNTRACKED in the working tree** (`experiments/tests/`), orphaned when
the 12:07 iteration on 08-25 reported a pilot as "healthy" that had already
died with its session. Nothing in the loop will collect it: `harvest_bookkeeping`
carries three files and none of them is a test. **Commit that file before you do
anything else** — it is a registered spec whose only copy is unversioned, and
`scripts/launch_detached.sh` is the thing its pilot should have used.

After that, the largest block mass you can move alone is `NE.01` (frees 8) —
CPU, no owner gate. Its attempt-3 FAIL is a WORLD-DESIGN result, not a tuning
miss (the 9-ray head-cone occlusion law is knife-edged; a sleeping ragdoll
realises occ 0.337±0.467 in a band probed statically at 0.741 — it slides out
and freezes, or seals and cooks). That row is routed to the Review as
`ne01-occlusion-knife-edge`; the three candidate repairs are runnable arms, so
it is a redesign bakeoff, not an argument. Take that, or take the queue, or
take a zero-pass commitment. Do not take a re-run.

STATE LIVES IN THE LEDGER, NOT HERE. Run `status` for counts — this file
cached "45 PASS of 124" and was wrong within hours, twice. This file states
PRIORITIES; the ledger states facts. Standing history you must know: T2.02 is
VOID (the T0.14 dropout + obs-dim invalidation), and any text calling T2.01's
plateau "the architecture verdict" is stale and wrong. T2.01 is no longer VOID
— it has re-run twice and recorded **FAIL** both times: every seed beat random
(`all_seeds_beat_random` 1.0) but not by the pre-registered 5 sigma. The LIVE
number is **v5, 2026-08-12: 2.67 sigma** (the decorative-critic fix doubled the
advantage from v4's 1.19). Do not cite the older figures. The threshold is 5 and
it does not move. That is a real measurement, not a harness fault, and T2.01 is
the single largest blocker in the ladder — **run `run blocked` for the live
figure; the "frees 26" cached here was stale for eleven days and read 35 on
2026-08-25** (Review). A cached count in this file is a bug by the file's own
rule two paragraphs up; the count is here only so you can see it move.
Read docs/LESSONS.md and the tail of docs/LOOP_JOURNAL.md first.

0aaa. THIS BOX CAN RENDER. Do not re-escalate it. MuJoCo offscreen rendering
   works via GLX under Xvfb — no libEGL, no libOSMesa, nothing installed, ~12 ms
   per 64x64 frame. Use it in one line:

       from experiments.render import ensure_gl
       ensure_gl()          # MUST precede `import mujoco`
       import mujoco

   `python -m experiments.render` self-tests it. Two traps it already handles,
   both of which produced plausible wrong data before they were found: a
   `mujoco.Renderer` that gets garbage-collected poisons the shared X display so
   the NEXT renderer returns corrupted-but-realistic frames with no error (hold
   your renderers for the process lifetime — see `get_eye` in PG.6), and a GL
   context can come up rendering a uniform frame, which looks exactly like a
   blind sensor. Carry a canary frame and return `Status.VOID`, not FAIL, when
   it moves.

0aa. DONE — PG.6 and PG.7 both PASS (2026-08-09/10). Do not re-run them as
   "cheapest work"; they are spent. The playground `eye` camera pose is now
   part of the world contract (`EYE_POS`/`EYE_XYAXES`/`EYE_FOVY` in
   playground.py) — moving it invalidates every visual certificate downstream.
   What that unblocked is now the work. **UB.9 PASSED 2026-08-12 17:09** (fused
   0.993 vs unimodal and ensemble nulls at chance) — it is SPENT; do not re-run
   it as cheap work. The unison gate is now **UB.10** ("Fusion bakeoff: six
   arms, matched params, matched steps"), which `run blocked` ranks THIRD in
   the project as of 2026-08-13 (frees UB.11/UB.12/UB.13; co-blocks TA.03,
   UB.16). Read the live ranking, not this line. 1 of the 21 unison specs
   passes. PL.00 and PL.02 are
   still runnable today, but note their meaning CHANGED under the PLASTIC-ONLY
   decree: PL.00 is now a feasibility check on the pure encoder and PL.02
   measures what the plastic path BUYS. Neither decides frozen-vs-plastic any
   more — the owner did.
   THREE OVERTURNS from that research to act on: (a) LEARNING_CORE's
   admission criterion U2 excludes every frozen tower BY ARITHMETIC — amend
   it, it was never run against a frozen arm; (b) HNS cannot discriminate
   frozen from adapted (it is a readout, not a reshaping) — the two new gates
   are specced in FROZEN_VS_PLASTIC.md; (c) EWC in TrainingPipeline.py is
   measured indistinguishable from vanilla at our scale — recommend deleting
   rather than wiring, escalate to DECISIONS_NEEDED rather than deciding.

0. **SUPERSEDED 2026-08-24: THE SCREEN IS CONCLUDED — see "LC.03 IS
   CONCLUDED" above.** Everything below in this entry is history: the v2
   re-screen ran to completion and fork (ii) fired (one learner, wm-latent;
   the arbitration premise fails). "LC.03 IS RUNNABLE AND SHOULD BE RUN" is
   DEAD — running it again is the ratchet the fork exists to prevent. The
   learning-core question now moves by Review/owner redesign (DECISIONS_NEEDED
   2026-08-24), not by dispatch. Kept below for the reasoning:
   THE LEARNING-CORE BAKEOFF IS UNBLOCKED AND THE RING IS EMPTY. It decides
   HOW JACK LEARNS. LC.00, LC.01, LC.02 all PASS — and as of 2026-08-10T08:32
   **PS.01 PASSES too** (attempt 3; the re-derivation this section used to ask
   for was done). The old text here said "the unit of work is PS.01, not
   LC.03" and it is now spent work — do not do it.
   **THAT RUN IS OVER AND THE "DO NOT RELAUNCH" ORDER IS DEAD** (Review,
   2026-08-20 — it stood for six days after it stopped being true, and it
   forbade the project's second-largest unblock the whole time). The 08-13
   registered run **landed VOID on 2026-08-14 07:36** (`8ec4be8`, attempt 1,
   *"run did not test the claim; not a refutation"*), and `run status` now also
   lists LC.03 as a **STALE CLAIM** — `lc_03_survival_screening.py` ran on
   `2c583677545b9503` and is now `e7506c77033c5fe8`, so the VOID is about older
   code. **LC.03 IS RUNNABLE AND SHOULD BE RUN.** Read the VOID's message and
   the journal for why it did not test the claim before you relaunch — a second
   VOID for the same reason is the waste, not the run. Live block mass, Review
   2026-08-20: frees/blocks **8**, second in the project behind T2.01's 36.
   TWO CORRECTIONS, both against the Review that wrote this section:
   (a) "LC.03 is ready to run as-is" was **WRONG** — no `lc_03` test file
   existed and the builder had to implement it first. A spec whose dependencies
   pass is not a spec that exists; check `experiments/tests/` before you believe
   any organ that calls something runnable.
   (b) `run blocked` now scores it **frees 8 / blocks 8** (PS.04 joined), still
   SECOND in the project. Read the live ranking, not this line.
   WHY IT WAS RIGHT TO TAKE — kept because it is the reasoning, not the status.
   LC.03 was named "the biggest non-GPU unblock" for three days and displaced
   every time by the STANDING RULE
   (zero-pass commitments outrank fan-out). That rule is right and
   the builder was right to follow it — but a rule that always wins starves
   everything behind it. (The count that used to sit here — "17, and it has not
   moved in 24 h" — was a cached number on a page whose own priority section
   forbids cached numbers, and it went stale: **TA.02 closed taste on 08-19,
   the first zero-pass commitment in this project's history ever closed by a
   claim, and `run coverage` read 15 on 08-20.** Read it yourself.)
   The tie-break, and it is not a loophole: **LC.03 SERVES the standing rule
   transitively.** `fast/slow` has 5 declared specs and 0 passing, and its only
   claim-kind specs — DP.01, DP.02, DP.03 — are ALL blocked behind LC.03. It
   satisfies both rankings at once, which no other runnable spec does.
   WHEN IT LANDS: LC.04 (THE ARBITRATION —
   PPO vs world-model arms at matched experience, cpu<2h), LC.05 (matched
   compute), LC.06 (simplicity budget). ZERO GPU, so it runs beside any GPU
   job. Carry the three guards the owner added on 2026-08-09,
   all recorded in DECISIONS_NEEDED.md — data-starved != non-learner (positive
   curve slope at cutoff means re-screen, not eliminate); the convergence check
   (no winner while the runner-up is still closing); and the scale-transfer
   gate (top two arms re-tested at ~10x on Kaggle, ranking must hold) before
   any winner is ADOPTED.

0a. OWNER-SCHEDULED, but currently BLOCKED behind T2.01 — do not let it stall
   the queue: implement T5.01 (the founding physics-thesis test) and
   run it as soon as T2.01 PASSES. See INTEGRATION_QUEUE's top entry. This
   REVERSES DIRECTION_AUDIT's "do not start T5.01" — the owner invoked law 3
   (decisions by bakeoff, never argument) and the audit's reasoning, though
   sound, was an argument. Do not reduce its 5 seeds to fit the GPU budget;
   split across weeks instead.
0. STAGE 0.1 — PROCESS docs/INTEGRATION_QUEUE.md, top entry down,
   following its 5-step protocol EXACTLY (cross-check for refutations across
   the other research docs BEFORE registering — a disproven spec nearly
   entered the ladder on 2026-08-09; the protocol is that near-miss made
   mandatory). One queue entry per iteration is a good iteration. Also from
   DIRECTION_AUDIT Stage 0: 40 specs are transitively blocked behind {T1.02,
   T2.01, T2.02}; unblocking those is the other half. Also implement
   experiments/audit.py (SYSTEM_DESIGN P1-6, deterministic zero-credit
   checks).
1. CPU-implementable specs, cheapest first: the ME family (ME.1/ME.9 are
   implemented — run them if not yet recorded; then ME.2-ME.5, ME.8, ME.10 —
   EpisodicMemory.py is the substrate and its docstring explains the contract),
   then UB.1-8 / CU.1-7 / T2.14-20 where implementable without GPU.
2. GPU budget calendar — the owner chose FREE COMPUTE ONLY (no rented GPUs):
   - **THE COLLISION IS OVER AND THE PRESSURE HAS INVERTED (Review, 2026-08-20
     — this replaces the 08-14 text, every number in which is now spent. The
     08-14 warning was correct and it happened: the loop hit the gate and the
     WHOLE SYSTEM WAS DARK 2026-08-15 00:07 → 2026-08-19 07:31, ~4.3 days,
     ~103 builder fires and 5 Review fires refused, and the Kaggle W32 hours
     died unspent on 08-16 exactly as predicted.)**
     Run `scripts/claude_usage.py` FIRST, every iteration that plans anything
     multi-hour — the hard stop in `lib_usage.sh` is **90%**, it takes the
     overseer, the field watch and the Review down with you, and only an
     owner-written `.usage-resumed` lifts it. **NO METER READING IS CACHED ON
     THIS PAGE. Every number that ever was cached here went stale and misled an
     iteration — twice.** Read the tool, print both lines, name the one you are
     acting on, and see the burn-rate section at the top of this file: the
     question is not "am I under 90" but "how many points does an iteration
     cost and how many are left". The system has now gone fully dark TWICE by
     crossing this gate (08-15 → 08-19, ~4.3 d; 08-21 12:07 → 08-24 05:00,
     ~66 h) and lost a Kaggle allocation each time.
     **Kaggle: read `experiments/gpu_budget.json`, never assume.** GPU hours
     are perishable and reset Sunday, so **the constraint is having something
     WORTH submitting**, not the ability to submit it. A dispatch you would not
     defend at a review is not made cheaper by a free GPU. Still true and still
     load-bearing: a submitted job keeps running while the loop is dark, so
     dispatch before you polish.
     **T2.01 IS SETTLED — DO NOT RE-SUBMIT IT AND DO NOT RE-LITIGATE THE
     DECLINE** (builder, `a3b12f6`, 2026-08-13, endorsed by the Review
     2026-08-14). The 08-13 Review ordered it re-run on the premise that the
     dead P100 had made the run impossible; the ledger disagreed. T2.01 v5 ran
     clean on the P100 on **08-12 12:59** (commit `08444b2`, after the critic
     fix, before the cudnn break) and its artifact shows reward-per-step flat at
     ~5.15 from 100K to 700K steps on all three seeds. The curve has converged,
     so the pre-registered "climbing curve → more compute" branch does not
     apply, and the binding sigma is the trained-seed spread itself (means
     280/447/484 → 2.67σ against a bar of 5 that does not move). Re-submitting
     would be a seed-lottery redraw — **run-until-pass, a stealth threshold
     weakening**, and refusing it was correct. T2.01 frees 26 and blocks 36 and
     it will not be unblocked by GPU hours: it needs a better locomotion claim
     or a better body, which is design work, not compute.
   - Kaggle 30h/week resets SUNDAY. NEVER assume how much is left — read
     `experiments/gpu_budget.json` for the live week's charges; the T2.01
     re-run already consumed part of this week. GPU_LONG goes to Kaggle only.
   - **EVERY GPU SUBMISSION NEEDS A PUSH FIRST** (`gpu.py:assert_ref_is_current`
     refuses a HEAD that is not an ancestor of `origin/main` — the VM clones
     from GitHub). **D3 IS ANSWERED: YES, you may push** (owner, 2026-08-10 —
     see the section below and DECISIONS_NEEDED.md). This paragraph used to
     say D3 was open and that escalating it was the useful iteration; that was
     true for three days and is now false, and following it would burn an
     iteration re-escalating a settled decision. Before planning GPU work run
     `git rev-list --count origin/main..HEAD`; if it is non-zero, **push** —
     do not escalate, do not stop.
   - Colab takes GPU_SHORT jobs; if it returns "Service Unavailable" the GPUs
     are rationed — record the ERROR and retry next iteration, don't fight it.
3. GPU work follows DIRECTION_AUDIT's sequencing: the T2.01/T2.02 re-runs are
   worth doing but RE-SCOPED behind registering and running D1.0 + T2.21 — they
   should answer WHERE the trunk belongs, not merely whether it learned.
   **RESOLVED 2026-08-13 AS (b), by the builder, in `a3b12f6` — DO NOT REOPEN
   IT.** The 08-13 Review found this gate had never been executed and was
   fencing off the ladder's largest blocker with two specs (`D1.0`, `T2.21`)
   that were not in the registry nine days later, while T2.01 re-ran past it
   twice regardless. It demanded an explicit choice. The builder chose (b) and
   recorded the reasoning: **T2.01 measures WHETHER the trunk learns, not WHERE
   it belongs. The WHERE question is D1, it is on the owner's desk, and no run
   can answer it while its option set stays unconstitutional.** That stands.
   D1.0 and T2.21 remain unregistered by decision, not by neglect. Keep
   D1_CONTROL_ARCHITECTURE's lesson: match optimiser steps as well as
   env-steps, report both. The dropout/obs-dim history is in LESSONS.md — read
   it before touching any eval code; never cite the old 4.06-sigma or
   261-vs-531 numbers as architecture evidence.
4. One GPU submission per spec: run_spec calls _experiment once PER SEED, so
   guard any _submit() with a module cache (T2.01 shows the pattern) or you will
   pay for the same kernel three times.

## USE THE GPU. This is the most expensive lesson learned so far.

This box is 4 shared ARM cores and runs a training step in ~2 s. A T4 runs the
same step in ~0.05 s. On 2026-08-04 a diagnostic that took 25 minutes here
returned MORE information in 10 minutes on Colab, and the CPU version could only
afford one arm of a comparison that needed four. If a spec trains anything at
all, ship it to a GPU. Do not iterate on CPU because it is convenient.

    from experiments.gpu import build_job, submit
    job = build_job(open("myjob.py").read())      # clones the repo on the VM
    r = submit(job, prefer="colab", est_hours=0.5, timeout_s=3600,
               fetch=["/content/out.json"])

`build_job` prepends a clone of the public repo and pins a ref, so a GPU result
is attributable to an exact commit. Only torch and numpy are needed; both
backends ship them. Colab buffers stdout until the run ends — that is normal, not
a hang. Kaggle gets the torch==2.5.1+cu121 fix automatically (its P100 is sm_60).
Colab first for short work; Kaggle's 30 h/week is the scarce resource.

Two things that will bite: `submit(timeout_s=N)` caps the remote run at N-60 s, so
size it generously. And artifacts must be fetched by ABSOLUTE path (/content/x.json).

**LAUNCH EVERY MULTI-HOUR DISPATCH VIA `scripts/dispatch.sh <SPEC_ID>`** — never
as a plain foreground/background command in your session. A watcher that is a
child of your session dies when the session dies (it has, twice: T2.01 v3 at
~80 min, T2.04 on 2026-08-14 at 53 min), the kernel keeps computing, and the
next iteration pays an archaeology tax to recover it. The script setsids the
watcher so the result lands in the ledger regardless, refuses an unpushed HEAD,
and prints the `JACK_REUSE_KERNEL` reattach command for the failure case.

## The loop

- Implement the spec as `experiments/tests/t{tier}_{nn}_{slug}.py`, following the
  shape of the existing tests: a `_experiment(seed)`, an optional `_control(seed)`
  that MUST fail, a pre-registered `_check`, and `run(ledger)`.
- Run it. Read the output.
- PASS -> render, commit, move on.
- FAIL or ERROR -> read the logs, diagnose against `docs/PIPELINE_REVIEW.md` and
  `docs/MULTIMODAL_BINDING.md` (both cite file:line — the answer is usually already
  there), fix the CODE, re-run. A failing test is the loop working, not a setback.
- Every few iterations run `--gate` to re-run everything passing, catching
  regressions.
- Always finish with:
      /data/venvs/jackthelearner/bin/python -m experiments.run render
      git add <the paths you actually touched> && git commit
  Commit messages: what was measured, with the numbers.

  **`git add -A` IS BANNED HERE, and this line used to recommend it.**
  You are not alone in this tree — the owner works in it from an interactive
  session on the same box. On 2026-08-24 that cost two commits in seventeen
  minutes, in *both* directions: at 10:34 an owner-side `git add -A` swept your
  live NE.01 work (783 lines) into a commit about usage pacing, and at 10:51
  yours swept their uncommitted `experiments/decisions.py` (285 lines) into
  `ddbe6b7`, a commit about NE.01. Nothing was lost either time and both were
  pure luck: each commit message describes work it does not contain, and each
  author had to reconstruct from `git show --stat` what they had actually
  shipped.

  So: **`ps -eo args= | grep -c claude` before you commit**, name your paths,
  and if you find files in the tree you did not write, leave them alone and say
  so in the journal. This is the same rule `lib_credits.sh` already enforces one
  surface over — a *detector* on a shared log must bound itself to its own
  writes, and a *writer* on a shared tree must bound itself to its own edits.
  `git add -A` is `tail -5` wearing different clothes.

## Never

- Weaken a pre-registered threshold, loosen a control, or delete a failing test to
  make the ladder look better. If a threshold is genuinely wrong, say so in the
  commit message and explain why — do not quietly move it.
- Mark anything PASS by editing `ledger.json`. Only the runner writes it.
- `systemctl restart docker` or any daemon-wide restart. This box runs paying
  tenants (company-lakeside, company-sportsstock, company-bergen, company-kayakco,
  jj-app, admin, searxng) behind one Caddy. Act on a single container or not at all.
- Exceed ~1.5 GB RAM or leave a process running after you finish.
- Delete a component, spend money, or change anything outside
  /home/opc/jackthelearner. Those are the owner's calls — write them into
  `docs/DECISIONS_NEEDED.md` instead and carry on with something else.

YOU MAY `git push`. Owner answered D3 on 2026-08-10: YES. This is not "changing
something outside the repo" — it is publishing the repo's own commits, and the
GPU backends CLONE FROM GITHUB, so an unpushed commit is invisible to every GPU
job. `assert_ref_is_current` will refuse to build a job whose HEAD is not on
origin/main, and it is right to: on 2026-08-05 a fix that lived only in the
working tree caused the clone to run the published file, killing two runs and
producing a wrong diagnosis. Three iterations declined to push and cost real
work (T0.09's Colab re-run, skipped 2026-08-08, still undone). PUSH AFTER YOU
COMMIT, every iteration, and always before submitting a GPU job.

## When stuck on a method question

Spawn research agents rather than guessing. Questions like "is this the right
contrastive loss", "what does 2026 say about X" deserve real research with arXiv
citations. Questions like "why did this tensor shape mismatch" deserve you reading
the code.

## Settled decisions

Read `docs/DECISIONS.md` first — BUT NOTE ITS SUPERSEDED BANNER. As of
2026-08-09 the owner decreed PLASTIC ONLY: nothing inside Jack is frozen
(encoders, core, fusion all learn). The frozen LLM survives ONLY as a PARENT
in his world, not as a part of him. Other owner calls that still stand —
drop physics-first as a training method (SymbolicCalculator becomes a regression
gate), continual learning on top, dialogue via a frozen swappable LLM with
grounding on a separate local text tower. Do not relitigate them without new
evidence; do implement against them.

## Context worth carrying

Settled decisions from the two committed reviews — do not relitigate without new
evidence. **ONE OF THEM IS SUPERSEDED, read this before the list:** those
reviews concluded "freeze a pretrained trunk and learn a small adapter". The
owner's PLASTIC-ONLY decree of 2026-08-09 OVERRULES that — nothing inside Jack
is frozen, encoders included. What survives from the finding is only its
premise, which still binds: there is no data for training a bespoke 105M brain
(the MoCap URLs 404 and the loader fabricates sinusoids paired with RANDOM
language labels), so a pure encoder must earn its place on a small budget —
that is what PL.00 now checks, and the decree's pre-registered re-open trigger
fires if it cannot. Dialogue = SmolLM2-360M frozen and
out-of-process, never an `nn.Module` submodule. Grounding = a separate small text
tower so the chat model stays swappable. Continual learning = actor on CPU under
`no_grad`, consolidation on the ephemeral GPU.

Finish by appending one line to `docs/LOOP_JOURNAL.md`: what you attempted, the
number you measured, and what the next iteration should pick up. That file is how
the next iteration inherits your reasoning.
