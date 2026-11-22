# FORGE.md — the science lab: how Jack's algorithms get better round after round

Owner, 2026-11-20: *"I seriously want a science lab here … self improving
everything we can, recursiveness is our goal."* *"Use SOTA methods mixed with
SOTA methods of self improvement, cook off everything properly, long run tests,
short run tests, algorithms, recursive cook-offs, self improving machine,
measure everything, rethink everything constantly, and have clear goals and
techniques. Robust JACK is the goal."* *"I'm talking about the system's logic,
not some small LLM stuff — science, algorithms."*

This file is the lab's charter. It binds lane `forge` and is read by the
Director, the Inventor and the Auditor. It sits under `docs/PLAN.md` (the five
steps) and `SYSTEM.md` (the four laws); it never overrides them.
Evidence base: `docs/research/PUBLISHED_EVIDENCE_2026-11-20_DISCOVERY.md`.

## The goal, in one measurable line

**The champion learner's score on the fixed battery goes up over time, and
nobody had to hand-pick why.** The battery is Jack's own world at its rungs plus
the known-answer rows. One number per week goes on the scoreboard: the current
baseline's sustained-life rate at the entry rung, beside its floor. Until a
learner eats, that number is 0 and the lab's first job is to move it.

## The machine: one loop, run for ever

1. **POOL — where candidates come from.** `docs/FORGE_POOL.md`, one line each
   with its `prior:` (law 3a). Four sources, each tagged so its yield can be
   measured: `published` (a component from the record: research desk, field
   watch), `inventor` (lane `lab`'s arms), `recombine` (two past survivors
   joined), `machine` (a model-written variant of ONE function inside the
   trusted training code; opens at stage 2 only).
2. **LADDER — short runs buy the right to long runs.** Same world, same frozen
   rule, three rungs:
   - **S, short:** a probe (default 50,000 decisions × 3 seeds). Cheap, many
     candidates. Decides nothing about Jack; only who climbs.
   - **M, medium:** default 300,000 × 5 seeds. The top third of S, and only
     those that beat BOTH the baseline and their never-trains twin.
   - **L, long:** 1,000,000+ × 10 seeds through `experiments/harness`. The
     decisive rung; at most 2 candidates plus the baseline.
   Budgets (candidates, seeds, core-hours, kernel-hours) are written in the
   round's rule before the first seed. A round that runs out stops; it does not
   borrow.
3. **JUDGE — fixed, and not the candidate's to touch.** The harness, floors,
   controls and pre-registration as they stand. The outcome must require
   eating (sustained mark), never life length alone. A rung-L winner is rerun
   on fresh seeds and its behaviour is looked at (trace or video) before
   anything is promoted: a fixed score is necessary and not sufficient.
4. **PROMOTE — the winner becomes the thing to beat.** A rung-L winner goes to
   the seat's bakeoff (law 3). If it takes the seat it is the new BASELINE and
   every later round is measured against it. This is the first recursion:
   each round starts from the last round's winner.
5. **LEARN — score the search, not only the candidates.** After each round the
   scoreboard records, per source, how many candidates went in and how many
   reached M and L, and what the round cost. A source that keeps losing to the
   `control` source (plain variation: the same budget spent on random settings
   near the baseline) is cut back. This is the second recursion: the way we
   search is itself in a cook-off.
6. **RETHINK — weekly, on the Sunday slot.** `docs/FORGE_REVIEW.md`, at most
   one page: did the one number move; which source paid; what to stop; whether
   the goals and the ladder's sizes are still right. Proposals to change this
   charter go to the Director's INBOX. The lab does not edit its own charter.

## What gets searched, in order

1. **The eating wall.** J11's measured cause: the entropy bonus is 15–18× the
   task reward. Round 1 is the hand-set grid over reward scale, entropy target
   and temperature cap (the record's remedy; no model-written search here).
   If J11's follow-up test (E2c) already covers it, round 1 adopts E2c's rule
   and results instead of repeating them.
2. **Learner components**, one family per round, each with a published prior:
   exploration (noise colour, entropy target, held actions), replay ratio and
   n-step returns, normalisation, optimiser (e.g. Lion, with its published
   limits), network size. Brain B first (cheap, CPU); brain A when its
   known-answer rows have reproduced.
3. **The curriculum:** the difficulty dial's schedule and gate.
4. **Memory and world-model settings** on brain A, when free TPU or GPU hours
   exist for it (`standing directive: no free hour lapses`).
5. **Machine-written single functions** (stage 2): a temperature schedule, an
   exploration term, a shaping term capped at a small fixed share of the true
   reward. Always with a plain-sampling control at the same budget.

## Stages

- **Stage 0 — instruments (now).** The round runner and the scoreboard exist
  and have planted tests. Nothing climbs the ladder before this.
- **Stage 1 — rounds on published and inventor candidates.** Rounds 1–3.
- **Stage 2 — machine-written candidates**, only after a stage-1 round has
  produced a learner that moves and eats, and the judge lock (order F5) holds.
- **Stage 3 — wider searches** when faster compute arrives. Same shape.

## Fences

**Whatever is being improved never writes its own judge.** Lane `forge` and
every candidate may NOT edit: `experiments/harness/`, `experiments/registry.py`,
`experiments/ledger.json` (only the runner writes it), any frozen rule or
pre-registration file, a threshold, a control, the floors' cards, cron,
`scripts/models.env`, `scripts/ladder_loop.sh`, any lane's prompt including its
own, the Auditor's files, or the "Fences" and "The machine" sections of this
file. A change it wants there is one INBOX line for the Director.

Also: no paid service. No model-written code runs outside the sandboxed
training entry point. No more than 8 new candidates a round. Do not attempt
whole-codebase self-rewriting, a lane editing its own improvement loop or model,
or meta-training a learning rule from scratch (see the evidence file's "what
not to attempt").

## Budget

- Lane `forge`: 2 slots a day (02h and 14h UTC), taken from lane `jack`, so the
  team's weekly allowance is unchanged.
- Free machines: rung S on the box only if under 15 CPU-minutes, otherwise
  Kaggle CPU. The forge holds at most 2 of the 5 Kaggle sessions unless the
  Director orders more; J-orders keep priority.
- Every round's rule states its cost before it starts; the scoreboard states
  what it actually cost.

## Second engine, small: the team learns too

`docs/LESSONS.md`: counted lessons (add, +1, −1, delete at zero), harvested
from VOIDs, failed runs and the Auditor's findings. Changes to a lane's prompt
are proposed with a measurable outcome and applied only by the Director. This
is the cheap, evidenced end of self-improvement; it never outranks the
algorithm work above.

## Orders for lane `forge` (the Director keeps this list; ≤ 5 open)

*Ranked by the Director 11-21: F3 (terms frozen, ruling 69) → F4. F1, F2 done. Take the first order you
can advance. One step a slot. State lines as in `docs/ORDERS.md`.*

### F1 — The round runner and the scoreboard (stage 0)
**Do.** `experiments/forge/`: (1) a round spec (JSON: baseline, candidates with
source tags, rung sizes, budget, the frozen outcome) and a function that, given
per-run results, applies the ladder's rule (top third, must beat baseline and
never-trains twin, budget stop). (2) Planted tests: a planted winner climbs; a
planted null does not; a candidate equal to the control does not; a round over
budget stops; a result file whose metric code hash differs from the frozen one
is refused. (3) `scripts/forge_scoreboard.py` → `docs/FORGE_SCOREBOARD.md`
(rounds, survivors per rung, per-source yield, cost, the one number).
**done when:** the tests pass and a dry round on synthetic results renders.
state: DONE 11-20 14:35. `experiments/forge/` (ladder rule + `metric.py`, sustained-life rate over late lives, hash covers the harness mark), `tests/test_forge_round.py` 22/22 (the five planted cases + 7 rule clauses each caught when mutated), `scripts/forge_scoreboard.py` (`--check`, `--dry` renders a synthetic round; the real board shows no round and the one number = 0 from pilot 2).
A round = `experiments/forge/rounds/<id>/spec.json` (committed before any seed) + `results.jsonl`. Readings of the charter the Director may overrule (INBOX 11-20): equal never climbs; a non-control candidate must also beat the control source's mean; above S the best control runs as a yardstick; a tie across the top-third cut drops.

### F2 — Seed the pool
**Do.** `docs/FORGE_POOL.md`: ≤ 20 candidates for brain B at the entry rung,
each one line with source tag and `prior:` from `PUBLISHED_PRIORS.md`, the
discovery evidence file or `LAB.md`. No candidate without a prior.
**done when:** the pool file exists and the Director has ranked it.
state: 11-21 02:45 POOL SEEDED, WAITS FOR THE DIRECTOR'S RANKING. `docs/FORGE_POOL.md`: 15 lines (published 11, recombine 1 empty seat, control 3, inventor 0), each with `prior:`, split by wall — `keep` on E-food (K1 resets, K2 TQC, K3 auto α, K4 SACLite, K5 1 update/10, K6 reward ÷100 + auto, K7 SimBa, K8 α 0.005) and `reach` on arm K (R1 2M, R2 auto, R3 stack-4, R4 keep-winner). `experiments/forge/pool.py` refuses a line with no prior / unknown source / > 20 / no control; `tests/test_forge_pool.py` 12/12.
Read in SBX 0.28 source: TQC ships, `param_resets` rebuilds actor+critics and keeps α and replay, `layer_norm=True` is critic-only (§21's three "not verified"). Six ideas have no prior on file and are listed as RESEARCH, not candidates. Forge's suggestion: a keep round first (K1, K2, K5, K6 + K3 as second baseline, C1, C2).

### F3 — Round 1: the eating wall
**Do.** Read J11's state and E2c first. Write round 1's rule (rung S and M
sizes, the grid, the control source, budget) and freeze it; or, if E2c covers
it, adopt E2c and record its results as round 1. No seed before the rule.
**done when:** round 1 is on the scoreboard with one reading per candidate.
state: 11-22 02:25 KERNEL BUILT (`bf1f5f7d`), RUNG S LAUNCHED, NOTHING READ. `experiments/forge/r1_run.py` runs all 10 arms from `spec.json` alone (a setting with no knob is refused); `tests/test_forge_r1_run.py` 18/18 under jack-std — the baseline through it equals E2c's `run_pilot_cell("E-food")` life for life, trace row and actor weight; each candidate's change read off the built learner.
Rung S = 30 runs in 8 kernels g0–g7, every kernel pinned to `bf1f5f7d`, two chains = 2 Kaggle CPU sessions (A g0 g2 g4 g6, B g1 g3 g5 g7; first kernels `jack-cpu-1795314026`, `…314086`; logs `/data/jack-logs/forge/r1/launch_S_chain{A,B}.log`). Raw files land in `/data/jack-logs/forge/r1/raw/S_g*.json`; home ≈ 11-22 14:00–18:00.
NEXT: the reader — raw → `results.jsonl` rows (score by `metric.py`, core-h/kernel-h from each file's `kernel` block, crash/cut = 0), refusing every row until S13 (b)'s tripwire holds and unless all rows share one `code_head`; the late-meals tiebreak in the ranking; then the scoreboard. A chain that died: relaunch only its missing `g*` with the SAME `--pin`.

state: DIRECTOR 11-21 (ruling 69, amends 61): round 1 is a KEEP round on E-food, NOT arm K. Its frozen terms are in `docs/ORDERS.md` J18 — write them into `experiments/forge/rounds/r1/spec.json` unchanged; that commit is the freeze; a term you cannot meet is an INBOX line, not a changed term. No result accepted before S13 (b)'s tripwire holds. Rank 2, ≤ 2 Kaggle sessions.

### F4 — The lessons file
**Do.** `scripts/lessons.py` (add / up / down / prune, with tests) and
`docs/LESSONS.md` seeded from this month's VOIDs and corrected rules.
state: (new 11-20.)

### F5 — The judge lock (for lane `system`, by the Director's order)
**Ask.** A tripwire first (hashes of the judge files checked by the runner
before it writes the ledger), then real separation (the runner under its own
user). Stage 2 does not open before this holds.
state: PLACED 11-20 as `docs/ORDERS.md` S13 (b): the hash tripwire. Separation under another user is not ordered (it writes outside the repo — the owner's to grant).
