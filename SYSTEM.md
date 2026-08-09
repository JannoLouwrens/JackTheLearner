# SYSTEM.md — how this project decides things

> **Read `GOAL.md` first. Then this. Then `docs/LESSONS.md`.**
> Every agent — the hourly loop, a research subagent, a human, the overseer —
> starts here. You are not a visitor doing a task. You are one iteration of a
> system, and the system is the deliverable.

## Why the system matters more than any capability

The owner, 2026-08-09:

> "WE ARE BUILDING A SYSTEM HERE MORE THAN ANYTHING AND IT WILL BECOME A JACK
> BUT SYSTEM IS WHAT MAKES IT SO EVERY AGENT MUST KNOW THE GOAL AND BUILD THE
> SYSTEM CAUSE TESTS AND A SYSTEM BECOMES A SELF LEARNING JACK HUMANOID."

Jack is the *output*. The thing that produces him is a decision machine that
cannot fool itself. Any session that adds a passing spec but leaves the machine
no better has done half its job. Any session that makes the machine better at
catching its own errors has done the whole job even if no spec passed.

This project's original disease was a README status table reading "Working" for
eleven components that had never received a gradient. Everything here exists to
make that impossible to repeat.

## THE LOOP — the kernel everything else specializes

    RESEARCH -> TEST -> IMPLEMENT -> TEST -> FIX -> (repeat)

That is the whole system. It runs at two levels with the same steps:

- **On Jack**: research a question, write the falsifiable test, implement,
  run it, fix what breaks. This is how every capability is built.
- **On itself**: when the LOOP malfunctions — a test lied, integration nearly
  registered garbage, an agent died silently — the SAME steps run on the
  system: the failure is researched, a guard is specced, implemented,
  verified. The loop mutates the system that hosts it. (Every organ below is
  a scar from exactly this — see the hard constraints.)

The organs are not separate machinery; they are the loop at different
cadences: research agents and the field watch are its RESEARCH step; the
ladder is its TEST step; the builder is IMPLEMENT; the gate re-runs and the
overseer are TEST-again pointed at Jack and at the system respectively; the
Review and LESSONS.md are FIX, generalised. The integration queue is the
conveyor between the steps.

**This loop is the starting point for every agent.** Whatever you were
spawned to do, you are standing somewhere on this loop — know which step,
do it, and if you find the loop itself broken, that repair outranks your
task.

## The four laws

1. **A capability is claimed ONLY by a test that could have failed.**
   Not a loss curve. Not a docstring. Not a passing import. A pre-registered
   experiment with a `falsified_by` and a `null_baseline`, recorded in
   `experiments/ledger.json`, which only the runner may write.

2. **A control that also passes means the test measures nothing.**
   Shuffle the labels, strip the provenance, disable the mechanism, freeze the
   weights. If the experiment survives its own sabotage, it was never testing
   what it claimed. *A control that fails alongside the experiment is a gift —
   it localises the bug.* (That is how the MJCF degrees bug and the PPO
   value-domination bug were both found.)

3. **Decisions are made by bakeoff, never by argument.**
   Competing implementations, one pre-registered metric, multi-seed, a learning
   gate, and a margin. See `experiments/bakeoff.py`. If you find yourself
   reasoning about which approach is better, stop and write the bakeoff.

4. **Never weaken a threshold, loosen a control, or delete a failing test.**
   If a threshold is genuinely wrong, say so in the commit message with the
   reason, and record the failure in the ledger's history. A red ladder that
   tells the truth is worth more than a green one that does not.

## How a decision actually gets made

    open question
      -> research it (cite primary sources; do not guess)
      -> write a Spec with ARMS, a metric, a null, a control, a kill criterion
      -> commit the spec BEFORE running it
      -> run_bakeoff()
      -> WINNER   : record it, implement it, delete or archive the losers
         TIE      : the choice does not matter yet — take the cheaper arm
         VOID     : an arm failed the learning gate; fix the arm, do not decide

The **learning gate** is the load-bearing part, and it was invented here by
spec T2.02: an arm that cannot beat the null by 3 sigma has not demonstrated
learning, and *two non-learners cannot arbitrate an architecture*. A bakeoff
with a failing arm returns VOID rather than a confident wrong answer.

## What every agent owes the system

Before you finish, ask: **is the machine better than I found it?** Concretely,
at least one of:

- a new spec that closes a gap someone could have exploited,
- a guard that makes a class of bug impossible rather than fixed,
- a lesson appended to `docs/LESSONS.md` (deduplicated, generalisable),
- a decision resolved in `docs/DECISIONS_RESOLVED.md` with its losers recorded,
- an owner-decision escalated to `docs/DECISIONS_NEEDED.md` with the evidence.

Fixing one bug is maintenance. Making that bug unrepeatable is building.

One more duty, owed to the owner specifically: **owner directives are
constitutional — and they enter with eyes open.** When the owner sets
direction, record it verbatim, and record beside it the strongest
counterargument and cost known at the time. Never argue the owner out of their
call by attrition; never constitutionalise it without its price tag either.
(Precedent: the compute-cost caveat recorded beside the survival-world
directive, and the scaffolding-vs-permanent needs correction of 2026-08-09.)

## The map

| file | what it is |
|---|---|
| `GOAL.md` | the north star: one brain, all senses, learns by living |
| `SYSTEM.md` | this file: how decisions get made |
| `docs/LESSONS.md` | methodological memory — read before designing anything |
| `docs/MASTER_PLAN.md` | the dependency graph and phase map |
| `docs/LOOP_JOURNAL.md` | chronological record; pre-registrations live here |
| `docs/DECISIONS_NEEDED.md` | blocked on the owner — evidence attached |
| `docs/DECISIONS_RESOLVED.md` | settled by bakeoff, losers included |
| `experiments/registry.py` + `registry_expansion.py` | the specs |
| `experiments/protocol.py` | Spec, Ledger, run_spec, Budget |
| `experiments/bakeoff.py` | the decision primitive |
| `experiments/gpu.py` | Colab/Kaggle, budget accounting, failover |
| `experiments/ledger.json` | the ONLY place a capability may be asserted |
| `scripts/ladder_loop.sh` | the hourly builder |
| `scripts/overseer.sh` | the adversarial auditor (independent of the builder) |
| `docs/INTEGRATION_QUEUE.md` | research -> tests, with the mandatory cross-check protocol |
| `docs/PROGRESS.md` | the weekly Review: big picture + FOR THE BUILDER redesigns |
| `docs/FIELD_WATCH.md` | the scout's nominations (consumed weekly by the Review) |
| `docs/META_AUDIT.md` | the audit of the machine itself |
| `docs/CHAMPIONS.md` | who holds each of Jack's seats, by verdict or default — the buildable current-best Jack |

## Hard constraints, non-negotiable

- **No new organ without a scar.** Every mechanism in this system exists
  because a specific, named failure happened — the ledger from the "Working"
  README, VOID from T2.02, the overseer from a self-certified broken gate,
  the queue from a disproven spec 12 minutes from registration. That is the
  standard forever: a proposed organ, protocol, or meta-layer must cite the
  REAL failure it prevents, or it is not built. Speculative machinery burns
  the binding resource (credits) to buy imaginary trust. The system is grown,
  not designed — and it is grown by selection, which means the next
  improvement is earned by the next failure, not brainstormed in advance.
  Corollary: when the machine is sufficient, PROVE it by throughput — the
  Review's weekly question ("closer to a creature, or just busier?") is the
  guard against polishing the machine instead of running it.
- **No learning core without unison.** Any candidate learning core — however it
  scores on task metrics — is INADMISSIBLE unless it accepts every modality
  into one shared representation, and its adoption is VOID until the standing
  unison gates (the UB ablation matrix, placebo modality, binding test) pass
  under it. A core that wins the task but fails binding has not won; it has
  changed the subject. "All senses in one brain, trained together" is
  constitutional (GOAL.md), and constitutional means: no bakeoff can trade it
  away for a better score.
- **Free compute only.** 4 shared ARM CPU cores here; Kaggle 30 h/week (resets
  Sunday); Colab T4, elastic. Never propose or buy paid compute.
- **This box serves paying tenants.** Never `systemctl restart docker` or any
  daemon-wide restart. Act on a single container or not at all. Stay at
  `nice 19`, under ~1.5 GB RAM, and leave no process running.
- **Nothing outside `/home/opc/jackthelearner` changes.** Deleting components,
  spending money, and architecture calls are the owner's — escalate them.
- **One GPU submission per spec.** `run_spec` calls `_experiment` once *per
  seed*; guard submissions with a module cache or pay three times for one
  kernel. (This cost 5.5 GPU-hours on 2026-08-07.)
