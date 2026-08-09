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

## Start here, every time

    cd /home/opc/jackthelearner
    /data/venvs/jackthelearner/bin/python -m experiments.run status
    /data/venvs/jackthelearner/bin/python -m experiments.run next

`next` lists specs whose dependencies pass. Take the FIRST one in priority order
below and finish it. One spec per iteration is a good iteration.

## Priority order (updated 2026-08-07; the ledger is still the authority)

STATE LIVES IN THE LEDGER, NOT HERE. Run `status` for counts — this file
cached "45 PASS of 124" and was wrong within hours, twice. This file states
PRIORITIES; the ledger states facts. Standing history you must know: T2.01
and T2.02 are VOID (the T0.14 dropout + obs-dim invalidation), and any text
calling T2.01's plateau "the architecture verdict" is stale and wrong.
Read docs/LESSONS.md and the tail of docs/LOOP_JOURNAL.md first.

0aa. CHEAPEST HIGH-LEVERAGE WORK IN THE PROJECT, DO IT FIRST: run PG.6 and
   PG.7 (~40 min CPU, both registered). FROZEN_VS_PLASTIC.md measured that
   they unblock NINE specs plus the entire unison ladder — 0 of 37 unison
   specs currently pass, and this is the gate. Then PL.00 and PL.02 (also
   runnable today) decide the frozen-vs-plastic constitutional question.
   THREE OVERTURNS from that research to act on: (a) LEARNING_CORE's
   admission criterion U2 excludes every frozen tower BY ARITHMETIC — amend
   it, it was never run against a frozen arm; (b) HNS cannot discriminate
   frozen from adapted (it is a readout, not a reshaping) — the two new gates
   are specced in FROZEN_VS_PLASTIC.md; (c) EWC in TrainingPipeline.py is
   measured indistinguishable from vanilla at our scale — recommend deleting
   rather than wiring, escalate to DECISIONS_NEEDED rather than deciding.

0. FINISH THE LEARNING-CORE BAKEOFF — it decides HOW JACK LEARNS and it
   is the highest-leverage unblocked work in the project. LC.00 and LC.01 PASS
   (the framing survived its cheapest falsifier; the unison admission gate
   exists). Implement and run, in order: LC.02 (throughput floor, cpu<10min),
   LC.03 (screening: which cores learn to survive at all, cpu<2h), LC.04 (THE
   ARBITRATION — PPO vs the world-model arms at matched experience, cpu<2h),
   LC.05 (matched compute), LC.06 (simplicity budget). ZERO GPU, so it runs
   beside any GPU job. Carry the three guards the owner added on 2026-08-09,
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
   - Kaggle 30h/week resets SUNDAY. Before Sunday assume ~0h left. After reset,
     the FIRST Kaggle job is T2.02 (the 140K-MLP-vs-transformer showdown at 2M
     steps — its kill-criterion settles the D1 trunk decision). GPU_LONG goes to
     Kaggle only.
   - Colab takes GPU_SHORT jobs; if it returns "Service Unavailable" the GPUs
     are rationed — record the ERROR and retry next iteration, don't fight it.
3. GPU work follows DIRECTION_AUDIT's sequencing: the T2.01/T2.02 re-runs are
   worth doing but RE-SCOPED behind registering and running D1.0 + T2.21 — they
   should answer WHERE the trunk belongs, not merely whether it learned. Keep
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
      git add -A && git commit
  Commit messages: what was measured, with the numbers.

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

## When stuck on a method question

Spawn research agents rather than guessing. Questions like "is this the right
contrastive loss", "what does 2026 say about X" deserve real research with arXiv
citations. Questions like "why did this tensor shape mismatch" deserve you reading
the code.

## Settled decisions

Read `docs/DECISIONS.md` first. Those are the owner's calls — freeze the trunk,
drop physics-first as a training method (SymbolicCalculator becomes a regression
gate), continual learning on top, dialogue via a frozen swappable LLM with
grounding on a separate local text tower. Do not relitigate them without new
evidence; do implement against them.

## Context worth carrying

Settled decisions from the two committed reviews — do not relitigate without new
evidence: freeze a pretrained trunk and learn a small adapter (there is no data
for training a bespoke 105M brain; the MoCap URLs 404 and the loader fabricates
sinusoids paired with RANDOM language labels). Dialogue = SmolLM2-360M frozen and
out-of-process, never an `nn.Module` submodule. Grounding = a separate small text
tower so the chat model stays swappable. Continual learning = actor on CPU under
`no_grad`, consolidation on the ephemeral GPU.

Finish by appending one line to `docs/LOOP_JOURNAL.md`: what you attempted, the
number you measured, and what the next iteration should pick up. That file is how
the next iteration inherits your reasoning.
