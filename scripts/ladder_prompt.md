You are continuing the Jack validation ladder. You have no memory of previous
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

## Priority order

1. Tier 0 remaining: T0.03, T0.04, T0.05 (checkpoint fidelity, resume without
   discontinuity, preemption survival), then T0.06-T0.08, then T0.09-T0.11 (Colab
   round-trip, Kaggle round-trip, failover), then T0.12.
2. Tier 1 remaining. Note T1.03 currently FAILS at 16.7% orphan fraction because
   `action_expert` (4.6M) — the module whose output reaches `mj_data.ctrl` — gets
   no gradient from `forward()`; it is only reachable via `act_dual_system`. That
   is an architectural defect to fix in the code, not a threshold to relax.
3. Tier 2 onward: real training, needs GPU. Use Colab
   (`/data/venvs/colab/bin/colab --auth adc run --gpu T4 script.py`) or Kaggle
   (`/data/venvs/kaggle/bin/kaggle`, 30 free hrs/week — spend it deliberately).

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
