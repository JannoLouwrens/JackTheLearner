# Decisions the owner must make

The loop writes here instead of acting when a choice is not its to make.


## Kaggle GPU is not being granted — needs your account action

**Blocks:** ladder specs T0.10 (Kaggle round-trip) and T0.11 (failover). Not Colab —
T0.09 passes and training can proceed on Colab T4 meanwhile.

**Measured 2026-08-04.** A kernel pushed with `--accelerator nvidiaTeslaT4` was accepted,
ran to `COMPLETE`, and returned its artifact — with **no GPU attached**:

    nvidia_smi : absent
    cuda       : false
    torch      : 2.10.0+cpu      <- Kaggle installs the CPU build when no accelerator is present

Nothing in the push, the status, or the output signalled a problem. This is the dangerous
shape of failure: a run that reports success and quietly did the wrong thing.

**Most likely cause:** Kaggle requires **phone verification** before granting GPU/TPU
accelerators. Unverified accounts silently receive CPU rather than an error.

**UPDATE 2026-08-04, after phone verification.** Verification worked — a GPU is now
attached. But Kaggle assigns a **Tesla P100 (compute capability 6.0)** regardless of the
accelerator requested; `--accelerator nvidiaTeslaT4` and `gpuT4x2` both returned P100. And
Kaggle's own preinstalled torch 2.10.0+cu128 ships kernels only for sm_70 ... sm_120:

    torch_arch_list: sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120
    device capability: 6.0  (sm_60 — Pascal, dropped)
    -> CUDA error: no kernel image is available for execution on the device

So the GPU is real and unusable by the torch that Kaggle itself installs. This is a Kaggle
environment incompatibility, not something wrong with our code.

**Options, none urgent — Colab works and training is not blocked:**
 1. Install a torch build with sm_60 kernels inside each kernel run (older cu121 wheels
    included Pascal). Costs 2-3 min per run and adds a version to maintain.
 2. Use Kaggle only for CPU-side work where its 30 h/week still helps.
 3. Skip Kaggle. Colab's T4 is sm_75 and works today; its shorter sessions are already
    covered by checkpoint/resume, which T0.04 and T0.05 prove.

**Recommendation: option 3 for now, revisit if a job genuinely needs a 12-hour session.**

**If you would rather not verify:** say so and I will mark T0.10/T0.11 as SKIP with this
reason recorded, and the plan proceeds on Colab alone. The cost is the 30 free hours/week
of longer-session compute — Colab's sessions are shorter, so multi-hour training would need
more checkpoint/resume cycles (which T0.04 and T0.05 already prove work).

## D1 — Does the 57M trunk stay in the control path? (OPEN, evidence complete)

Raised by the 2026-08-04 multi-agent review ("freeze a pretrained trunk, learn
a small adapter; never train a bespoke 105M brain from scratch"). Never
actioned: every run since has trained the whole trunk through PPO. It gates
the locomotion branch (T2.01, T2.02 and dependents) and nothing else, which is
why 42 other specs passed while it sat open.

EVIDENCE, three independent runs at matched env-steps on Humanoid-v5:

  T2.01 v4   57M trunk        261 return   4.06 sigma (bar 5)   curve PLATEAUED
  MLP probe  54,179 params    531 return   ~6.5 sigma           still climbing
  T2.02      124,707 params   530 return   7.11 sigma
             57M trunk        318 return   2.46 sigma  <- below its own 3-sigma
                                                          learning gate

T2.02 declared itself VOID rather than arbitrate, because an arm that has not
demonstrably learned cannot be compared. That is correct protocol. But the
trunk failing a learning gate that a 125K net clears at 7 sigma, in three
separate runs, is itself a consistent finding.

THE OPTIONS

  A. Freeze the trunk; small dedicated policy head does control.  RECOMMENDED.
     The trunk keeps the jobs it is actually good at and that GOAL.md needs
     from it -- perception, language, memory, cross-modal binding -- and stops
     being asked to be a motor controller. Matches the Aug-04 review, keeps
     "one brain, all senses" intact (the head reads trunk features), and is
     the only option that explains the data rather than fighting it.
  B. Split trunks: separate value/policy networks, trunk untouched elsewhere.
     Cheaper to try, but does not address why 57M params underperform 125K.
  C. Keep training end-to-end and buy more compute. The curve plateaued at
     704K steps/seed. Not supported by evidence.
  D. Delete the transformer from the control path entirely (T2.02's literal
     kill-criterion). Too broad: it would also remove the trunk from the
     multimodal work it has not yet been tested on.

COST OF DELAY: T2.01/T2.02 and everything downstream of locomotion stay
blocked. The memory, playground and curiosity branches are unaffected.

Owner: pick A, B, C or D (or say "do what the measurements say" and it will be
read as A, with the change journalled and T2.01 re-run under the new
architecture).

## STALE — the Kaggle block above was resolved by the system, not by you (raised by the overseer, 2026-08-09)

**Ask:** one line from the owner to strike the "Kaggle GPU is not being granted"
block at the top of this file. It is now false in three ways, and it is the first
thing anyone reads here.

**Evidence.**

1. It states *"Blocks: ladder specs T0.10 (Kaggle round-trip) and T0.11
   (failover)."* Both are **PASS** in `experiments/ledger.json` — T0.10 at
   `2026-08-04T15:50:08`, T0.11 at `2026-08-04T15:53:54`, both at commit
   `bb1659d`.

2. It recommends *"option 3 for now"* — skip Kaggle, run on Colab alone. The
   system implemented **option 1** instead (install a torch build carrying sm_60
   kernels inside each kernel run). Commit `114e8f7`, 2026-08-09: *"T2.02
   postmortem: job's own sb3 install clobbered the P100 torch pin; PIP_CONSTRAINT
   now holds torch==2.5.1 for all later installs."*

3. Kaggle is not merely unblocked, it is the project's primary GPU backend: T2.02
   ran a **6.28-hour Tesla P100 kernel to completion today** (`ran_at
   2026-08-09T07:30:25`, `duration_s 22604.42`, `backend: kaggle`), and
   `experiments/gpu_budget.json` shows 6.3849 of the fresh 30 h week spent on it.

**Why this needs you rather than the loop.** Nothing about the engineering is
open — the work is done and measured. But this block was written as an owner
decision with named options, and SYSTEM.md does not let the loop mark an owner
decision resolved. The honest record is that option 1 was taken; it should say
so, over your name, rather than continue to ask a settled question.

**Suggested resolution:** strike the block and record in
`docs/DECISIONS_RESOLVED.md`: *"Kaggle accelerator — resolved by option 1
(in-kernel torch pin with sm_60 kernels, PIP_CONSTRAINT torch==2.5.1). Losers:
option 2 (CPU-only Kaggle), option 3 (skip Kaggle). Evidence: T0.10/T0.11 PASS,
T2.02 6.28 h P100 kernel 2026-08-09."*

---

## D2 — Does a VOID dependency BLOCK its dependents?

**Raised 2026-08-09** by the overseer audit (§1.3 context, builder item 6).
Code and documentation currently contradict each other, and both are shipped.

`Status.VOID`'s docstring (`experiments/protocol.py:58`) says a VOID spec
*"does not BLOCK its dependents on the grounds that the claim was refuted."*
`Ledger.blocked_by` (`protocol.py:242`) returns any dependency whose status
`is not Status.PASS`, so VOID blocks **exactly like FAIL**. The docstring is
the aspiration; the code is the behaviour. Right now T2.13 and T5.09 are
BLOCKED behind T2.02's VOID.

This is not a bug I can settle by bakeoff — there is no metric to measure, only
a choice about what the ladder means. Both readings are defensible:

- **Block (current behaviour).** An undemonstrated foundation is undemonstrated.
  Building on a run that could not test its own claim is how unearned green
  ticks propagate, which is this repo's original disease.
- **Do not block (current docstring).** VOID means "we learned nothing", not
  "the claim is false". Blocking treats a *failure to measure* as a *negative
  result*, and it is the reason 34 specs are parked: T2.02 refusing to
  arbitrate now has the same downstream force as T2.01 losing outright.

**The loop's recommendation: BLOCK, and fix the docstring** — but make the
blocking *message* distinguish the two, so `run status` says "dependency T2.02
is VOID (not demonstrated)" rather than "dependencies not passing". The
asymmetry that matters is `kills`, which VOID already correctly suppresses; the
dependency graph should stay conservative.

**Cost of the status quo:** none beyond the contradiction itself, since the
code already blocks. The risk is that someone reads the docstring, assumes
dependents are runnable, and is confused by a BLOCKED result.

**One line from you settles it.** Saying *"take the recommendation"* will be
read as block-and-fix-the-docstring, implemented and journalled.

## /data is 95% full and Jack is not the cause (OPEN, owner action)

Found 2026-08-09 by the memory-retrieval agent, then confirmed. /data hit
**100% (661 MB free of 100 GB)**. Jack's share was 17 GB, of which 4.9 GB was
redundant source archives — deleted, verified extracted first, recovery
instructions in /data/jack-data/ARCHIVES_REMOVED.md. That bought 5.6 GB.

THE ACTUAL CONSUMER IS NOT THIS PROJECT:

    /data/history/history.sqlite       75.6 GB
    /data/history/history.sqlite-wal    1.7 GB
    /data/jack-data                    17   GB   <- Jack, now 12 GB
    /data/caches                        7.2 GB

That is WorldTwin's aggregator database, and CLAUDE.md records that a runaway
WAL on this exact database filled /data once before. The WAL is 1.7 GB now,
not catastrophic, but the 75.6 GB main file leaves no headroom for anything.

This is OUTSIDE /home/opc/jackthelearner, so it is not mine to act on, and the
box serves paying tenants (company-lakeside, sportsstock, bergen, kayakco,
jj-app, admin, searxng) behind one Caddy. Owner's call. Options, in the order
a WorldTwin session should consider them:
  - VACUUM the database if it has free pages (needs ~equal free space — it does
    not have that right now, so this may require pruning first);
  - prune old history rows to a retention window;
  - grow the block volume.

RISK IF IGNORED: at 0 bytes free, WorldTwin's writes fail, and Jack's ladder
also stops — the loop already refuses to start below 3 GB free, which is the
only reason it has not been silently corrupting runs. Note the loop's guard
checks / (73% used), NOT /data, so it would not have caught this.

### D1 — CORRECTION 2026-08-09, evidence is confounded. DO NOT DECIDE ON IT.

The section above says "three independent runs at matched env-steps" show the
57M trunk plateauing. That claim is not safe, and the reason was found by
measurement, not argument (docs/research/D1_CONTROL_ARCHITECTURE.md, verified
independently before writing this):

1. DROPOUT WAS LIVE THROUGHOUT. TrainingPipeline never calls .eval()/.train().
   36 nn.Dropout modules at p=0.1 stayed active during rollout, during the PPO
   update, and during "deterministic" evaluation. Measured on the real
   pipeline: two forwards of the SAME state differ in the policy mean by 42%
   of the mean's own magnitude (66% for the value). In eval mode the same
   double-forward is bit-identical. The PPO importance ratio at ZERO policy
   change puts ~20% of samples outside clip_range=0.3 — the update was
   clipping against its own noise.
   SB3's MLP has no dropout and disables training mode for rollouts. So the
   two arms of T2.02 were not the same experiment: one was evaluated with 42%
   action noise injected, the other with none. That is not a fair architecture
   comparison, and the trunk's 261 vs the MLP's 531 cannot be attributed to
   architecture until it is re-run.

2. "MATCHED ENV-STEPS" WAS NOT MATCHED OPTIMISATION. 6,240 vs 99,840 optimiser
   steps — 16x fewer, on a model 457x larger. The ppo_minibatch 64->512
   throughput fix preserved sample-passes and silently divided gradient steps
   by eight. My own change, and I did not notice the consequence.

3. THE PLATEAU IS NOT IN THE LEDGER. curve_seed0 stores [:8] — iterations 1-21
   of 172. I described the curve as plateaued; the stored evidence does not
   cover the region where that would be visible.

4. obs projection pads 28 zeros (mujoco_obs_dim=376 is the Humanoid-v4 value;
   v5 emits 348, confirmed), and JointTokenizer slices a dense LayerNormed
   projection into "17 joint tokens" that contain no joints.

Also demoted, having been checked: value/policy gradient interference
(cos(grad_pg, grad_vf) = 0.102, vf/pg = 0.052 on the trunk) and gradient-norm
clipping (binds, but Adam is scale-invariant). The two intuitive culprits are
NOT the cause.

WHAT THIS CHANGES: option C ("keep training end-to-end") was listed as "not
supported by evidence". It is UNTESTED, not refuted. Option A (freeze + small
head) is still the recommendation, but it must be EARNED by the bakeoff
(T2.21, ~6.3 GPU-h for the Week-32 half), not adopted by argument. Nothing
about D1 should be decided until the dropout fix lands and the comparison is
re-run.

---

## D3 — May the loop `git push`? It has blocked GPU work three times now (OPEN, one-line answer)

**The mechanism.** `experiments/gpu.py:assert_ref_is_current` refuses to build any
GPU job whose HEAD is not an ancestor of `origin/main`, and it is right to: the VM
clones from GitHub, so unpushed work is simply not there. On 2026-08-05 that cost
two GPU runs and produced a wrong diagnosis. **So every GPU submission requires a
push first.** There is no way around it that is not worse.

**The block.** The loop prompt says "change anything outside
/home/opc/jackthelearner" is the owner's call, and pushing publishes to a public
GitHub repo. Iterations have read that both ways:

| date | what happened |
|---|---|
| 2026-08-08 | iteration declined to push; **T0.09's Colab re-run did not happen** and has not happened since (`LOOP_JOURNAL.md:785`) |
| 2026-08-09 13:21–14:04 | six commits **were** pushed, up to `76ccc6c` |
| 2026-08-09 15:15 (this iteration) | declined again; `ddde954` and `49529e6` unpushed |

That is not a stable rule, it is a coin flip, and it decides whether the most
expensive resource in the project can be used at all.

**What it costs right now.** Today is Sunday — the Kaggle quota reset this
morning and **~23.6 of 30 h are unspent, expiring 2026-08-16**. The top GPU
priority is the T2.01/T2.02 re-run (~13 h) that D1 is waiting on and that 34
specs sit behind. It is ready: T0.14 fixed the pipeline, T0.16 (this iteration)
fixed the shipped eval path that would have re-contaminated it, and both PASS.
It cannot be launched because the fix is in commit `49529e6`, which is not on
GitHub. Unspent free quota is not saved; it is lost.

**Note what is actually at stake.** The repo is already public and already
contains every file involved. The commits in question are ladder specs and a
`TrainingPipeline` fix — the same category of content as the 76 commits already
published. This is not a question about *what* gets published, only about
whether the loop may perform the routine step its own toolchain requires.

**Options:**
1. **Standing authorisation** — the loop may `git push` its own commits to
   `origin/main` at any time. Simplest; matches what the toolchain assumes and
   what already happened today.
2. **Authorise pushes only when a GPU submission needs one.** Narrower, and
   covers every case that has actually arisen.
3. **Keep it your call** — then please push manually, and expect the loop to
   escalate here each time GPU work is ready. Under this option the ~23.6 h
   expiring on 08-16 will mostly go unused.

**Recommendation: option 1 or 2.** Either unblocks the re-run today; option 3
should be chosen deliberately, not by default, because its cost is the quota.

## Claude credits are the binding resource and are unmetered (OPEN, owner)

Found by the 2026-08-09 meta-audit. GPU hours are metered to the second
(gpu_budget.json, weekly ledger, affordability gate). Claude usage — which
powers the hourly builder, 6-hourly overseer, weekly field watch, weekly
review, and every research agent — has NO meter at all, and the builder ran
dry 4 times today (fable -> opus fallback engaged). The machine now has four
organs spending the same unmetered budget on schedule.

Owner call, options: (a) accept as-is — fallback chains already prevent dead
slots; (b) set a cadence budget (e.g. drop the builder to every 2h overnight);
(c) plan-level decision about credit allowances. The system cannot see its own
credit balance, so any budget must be time/cadence-based, not token-based.
Note: experiments/audit.py (queued for the builder) gives zero-credit
integrity checking either way.

## The owner's hands — how does a human TOUCH Jack's world? (OPEN, design fork)

Found 2026-08-09 by the owner auditing the design against the nurture itch:
Jack knows you (attributed memory), you can watch him (specced), you can talk
to him (specced) — but there are NO CARE VERBS. You cannot leave food, warm
his shelter, or hand him anything. The human has eyes and a voice in his
world, but no hands. Tamagotchi's entire loop is "you feed it"; our design
has him feed himself (correctly — first principle), which serves the WATCH
itch and starves the NURTURE itch.

Proposed reconciliation, consistent with the first principle and the
emergence stone: the human enters as ENVIRONMENT, not puppeteer — parental
provisioning, biology's own pattern. You may place things in his world; he
still must find, learn, and choose. Teaching-by-telling is already designed
(culture transfer). Attribution gives gratitude a substrate: his diary would
record WHO left it — attachment to his people gets somewhere real to grow
from, unscripted.

Needs: owner's call on whether care verbs exist at all and which; then a
small research pass + SO-family specs (interaction channel, anti-puppeteering
limits — care must never become remote control, or the first principle dies).

**DECIDED 2026-08-09, same day: YES.** Owner: "Can you also drop stuff in for
him... Yes." Care verbs approved on the provisioning-as-environment model.
The anti-puppeteering constraint stands: what is left must still be found,
learned, and chosen. Design work unblocked -> INTEGRATION_QUEUE.

---

## D2 — COST CORRECTION 2026-08-09 18:37 (overseer). The status quo is not free.

D2 above states: *"Cost of the status quo: none beyond the contradiction itself,
since the code already blocks."* That is wrong, and the number matters to the
decision. Measured by walking the dependency graph at `db9fd7b`:

**40 of 136 specs have a VOID in their dependency chain and cannot be attempted.**

| terminal blocker | specs blocked | what they are |
|---|---|---|
| `T2.01 = VOID` | **36** | CU.1–CU.7 (**every curiosity spec**), UB.1–UB.8, T3.02/T3.04/T3.05, T4.01/T4.04/T4.05, T5.01–T5.08, T6.01/T6.02/T6.04/T6.05, ME.7, T2.16–T2.18 |
| `T2.02 = VOID` | **4** | UB.15, UB.16, T2.13, T5.09 |

Two things this changes about the choice:

1. **The blocked set is GOAL.md's headline, not a side branch.** Curiosity has
   0/7 passing and all 7 are unreachable. All-senses unison has 0/16 passing and
   15 of the 16 are unreachable (UB.14 is the only one clear). Tiers 3, 4 and 5
   are 0/24 and entirely unreachable. `LESSONS.md` already carries the warning
   this reproduces: *"be suspicious when the project's headline claim is one of
   the unreachable ones."*

2. **Four of the 40 are blocked behind a run that refused to arbitrate.** T2.02
   declared VOID because an arm missed the 3-sigma learning gate — the protocol
   working exactly as designed. Under the current code that correct refusal has
   the same downstream force as an outright FAIL. That asymmetry is the substance
   of D2, and it is now costing four specs.

**This does not argue for either side.** "An undemonstrated foundation is
undemonstrated" remains a good reason to block, and the loop's recommendation
(block, and fix the docstring, and make the BLOCKED message say *"dependency
T2.02 is VOID — not demonstrated"* rather than *"dependencies not passing"*) is
still defensible. Note also that deciding D2 the other way would **not** by
itself unblock 40 specs — it unblocks T3.02, T2.13, T5.09, UB.16 immediately,
and the rest only as those actually run and pass. The real repair for 36 of the
40 is the T2.01 re-run, which is behind **D3**.

The ask is unchanged and still one line. This note only ensures the price tag is
on the table, per SYSTEM.md's rule that an owner decision enters with its cost
recorded beside it.

*Evidence: `experiments/protocol.py:243` (`blocked_by` returns any dependency
`is not Status.PASS`) vs `protocol.py:59-61` (VOID's docstring says it does not
block). Graph walked over all 136 registered specs against
`experiments/ledger.json` at `db9fd7b`. Full working in `docs/OVERSIGHT.md` §1.3.*

## Does the LC bakeoff's verdict survive scale? (OPEN — owner flagged the risk)

Owner, 2026-08-09: "are you sure it isn't holding us back that agents are
making CPU tasks and not GPU?" Audited. Verdict: the CPU scoping is mostly
LEGITIMATE, with one real gap.

Legitimate: LC.00 is a 2-CPU-minute gridworld falsifier whose stated kill is
"the whole LC programme... before any body, any physics, any torch or any GPU
is involved" — cheapest-falsifier-first, exactly right. LC.01/02/06 are
property checks (unison admission, throughput floor, simplicity) that a GPU
cannot make more true. And the survival world measured 2,826 steps/s on CPU,
faster than Craftax — the world genuinely is a CPU workload.

THE GAP: LC.03/04/05 ARBITRATE the learning core at cpu<2h. RL algorithm
RANKINGS ARE KNOWN TO CHANGE WITH SCALE — the field's own literature is full
of small-scale verdicts that inverted (our own record: DreamerV3 beat PPO
until a tuned PPO at 4M params beat DreamerV3 at 201M). A core crowned in 2
CPU-hours may not be the core that wins at the scale Jack actually lives at,
and LC.04's PROVISIONAL clause only re-checks UNISON gates, not scale.

Needed: a scale-transfer check before the LC winner is adopted — re-run the
top two arms at ~10x experience on Kaggle (~6-10 GPU-h of the ~130/month
available) and require the RANKING to hold. If it inverts, the CPU verdict is
void and the GPU run decides. This is cheap, and it is the difference between
"decided" and "decided at a scale that transfers".

### ADDENDUM — the scale bias is DIRECTIONAL, and it points at PPO

Owner, 2026-08-09: "some stuff needs much more GPU hours to prove, like PPO —
will we properly test it?" Correct, and sharper than the general scale worry:

PPO is DATA-HUNGRY; world models are SAMPLE-EFFICIENT. A cpu<2h arbitration
therefore tests in exactly the regime where PPO looks worst and Dreamer looks
best. LC.03's screening gate (beat random by 3 sigma, beat your untrained twin
by 3 sigma) could ELIMINATE PPO before LC.04 runs — not because it is worse,
but because it had not got going yet. LC.03's "fewer than two arms => VOID"
protects the one-survivor case but NOT the case where two world-model variants
clear and PPO alone is dropped.

REQUIRED FIX, using this project's own precedent (T2.01 v3: "curve still
climbing at cutoff" -> more compute, not a verdict): an arm that fails LC.03
while its learning curve still has a POSITIVE SLOPE at cutoff is NOT
eliminated. It is recorded DATA-STARVED and re-screened at ~10x experience on
Kaggle before any elimination stands. Only a FLAT curve at cutoff justifies
"this core cannot learn". Same rule, symmetric, applied to every arm.

Rationale beyond fairness: Jack's operating regime is CHEAP LIVES IN LARGE
NUMBERS (2,826 steps/s measured on CPU) — the regime where sample efficiency
matters LESS and throughput/simplicity matter MORE. Testing only in the
low-data regime measures the wrong end of Jack's actual life.

### ADDENDUM 2 — the truncation trap: no winner while the gap is still closing

Owner, 2026-08-09: "PPO might be best for more deep learning after 20 hours
when we stop at 19?" Exactly right, and NOT covered by Addendum 1. That rule
protects an arm from ELIMINATION while its curve still rises (LC.03). It does
NOT protect the WINNER decision (LC.04/LC.05): both arms can clear the gate,
the leader wins at cutoff, and the trailing arm crosses one hour later. This
is the standard way cheap benchmarks crown the wrong method, and it is
DIRECTIONAL against PPO for the same reason as before.

REQUIRED, before any LC winner is adopted — the CONVERGENCE CHECK:
  Fit the last third of each finalist's learning curve. Declare WINNER only
  if EITHER (a) the runner-up's slope is <= 0 (it has stopped improving), OR
  (b) the projected crossover lies beyond 3x the tested budget. Otherwise the
  verdict is SPLIT-PENDING: extend BOTH finalists to the projected crossover
  (or 3x, whichever is smaller) and re-decide. Cost is affordable and that is
  the point — ~130 GPU-h/month exist and ~20 are used; an extension is hours,
  not money.

The constraint here was never the budget. It was the CUTOFF CHOICE. A cutoff
picked for convenience and then treated as a verdict is a resource limit
masquerading as a result — and it would have been invisible in the ledger,
because every number in it would have been true.

## Was physics-first retired by argument instead of by bakeoff? (OPEN, owner)

Owner, 2026-08-09: "I thought we started by training on physics, so Jack
inherently learns the patterns of the universe — did we throw it away?"

THE RECORD: physics-first was dropped as a TRAINING METHOD in docs/DECISIONS.md
on literature grounds (arXiv:2507.06952, 2111.05458; nothing that walks in 2026
got there via symbolic physics pre-training). SymbolicCalculator survived as a
frozen regression gate. That document ALSO said: "T5.01 still runs, cheaply and
early, so the decision rests on our own numbers." T5.01 — titled "THE thesis
test" — is still NOT_RUN, and DIRECTION_AUDIT later recommended never starting
it ("superseded premise").

THE PROBLEM: SYSTEM.md law 3 is "decisions are made by bakeoff, never by
argument." We enforced it on PPO, the trunk, memory retrieval and the learning
core. We did NOT enforce it on the project's founding idea — that one was
retired by citation. The audit's "superseded" reasoning is sound but it is
still an argument, and the decision doc itself promised our own numbers.

WHAT SURVIVED ANYWAY (worth stating, because the vision is not the method):
"Jack sees the patterns of the universe" is precisely what a WORLD MODEL does —
predicting the next observation IS learning physics, discovered by living
rather than supplied as symbols. That route is alive and on trial in the LC
bakeoff. Only the SUPERVISED-SYMBOLIC route was retired.

OWNER'S CALL: (a) run T5.01 as promised — it is cheap and it makes the founding
decision rest on our evidence rather than someone else's papers; (b) formally
retire it, recording in DECISIONS_RESOLVED that this one decision was made by
argument, so the exception is visible rather than silent. Either is defensible;
leaving it NOT_RUN and unexplained is not.

**DECIDED 2026-08-09: (a) RUN IT.** Owner: "schedule the run after T2.01."
The founding premise gets tested on our own numbers, as DECISIONS.md promised.
Note this REVERSES DIRECTION_AUDIT's "do not start" recommendation — the audit's
reasoning was sound but it was an argument, and law 3 outranks it. Queued; the
correction to the audit is recorded here so the two documents no longer
contradict each other.
