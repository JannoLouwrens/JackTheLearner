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
