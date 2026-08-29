"""The validation ladder.

Ordering principle: CHEAPEST FALSIFICATION FIRST. A hypothesis that can die on
CPU in 60 seconds must never be allowed to consume GPU quota. Tiers 0-1 cost
nothing and kill most bad ideas.

Composition principle: nothing is combined until it passes alone, and nothing
stays combined without its earlier gates re-running. The system only ratchets
forward, and when a metric moves you know which addition moved it.

Deletion principle: Tier 3 is an ablation tier. A component that cannot show a
measurable contribution gets deleted. This is the direct antidote to the repo's
current state — 45.5M parameters that look like capability and do nothing.
"""
from __future__ import annotations

from .protocol import Budget, Spec

LADDER: list[Spec] = [

    # ===================================================================
    # TIER 0 — HARNESS. Can we measure anything at all?
    # Without these every later number is uninterpretable.
    # ===================================================================
    Spec("T0.01", 0, "Repo imports clean",
         hypothesis="Every live module imports without side effects on CPU.",
         falsified_by="Any ImportError, or an import that starts a training run.",
         null_baseline="n/a — structural precondition.",
         metric="modules_imported", budget=Budget.CPU_FAST,
         kills="Nothing runs until fixed."),

    Spec("T0.02", 0, "Deterministic seeding",
         hypothesis="Same seed produces an identical loss trace on CPU.",
         falsified_by="Two runs at seed 0 differ beyond float tolerance.",
         null_baseline="Different seeds should DIFFER — if they match, the seed is ignored.",
         metric="max_abs_trace_delta", budget=Budget.CPU_FAST, depends_on=["T0.01"],
         control="Different seeds must produce different traces.",
         kills="Every A/B comparison in the ladder. Non-negotiable."),

    Spec("T0.03", 0, "Checkpoint round-trip fidelity",
         hypothesis="save -> load reproduces identical forward outputs.",
         falsified_by="Output delta > 1e-6 after a reload.",
         null_baseline="A randomly re-initialised model gives a large delta.",
         metric="output_delta", budget=Budget.CPU_FAST, depends_on=["T0.02"],
         control="A DIFFERENT random init compared against the saved model must NOT match — a round-trip check whose comparison cannot see two different networks is measuring nothing.",
         kills="All GPU training — a checkpoint that does not restore is wasted compute."),

    Spec("T0.04", 0, "Resume continues, does not restart",
         hypothesis="A checkpoint restoring optimiser state tracks an uninterrupted "
                    "run far more closely than restoring weights alone.",
         falsified_by="Restoring optimiser state is no closer to the reference "
                      "trajectory than a weights-only resume (fidelity ratio < 10).",
         null_baseline="Weights-only resume — the naive implementation.",
         metric="resume_fidelity_ratio", budget=Budget.CPU, depends_on=["T0.03"],
         control="Weights-only resume must diverge markedly further.",
         kills="Any multi-session run. Kaggle caps at 12h; jobs must survive it.",
         notes="THRESHOLD REVISED 2026-08-04, with evidence. The original metric was "
               "'loss jumps >20% at the resume boundary'. Measured, that metric reads "
               "1.326% in BOTH arms — identical — so it cannot discriminate a correct "
               "resume from a broken one: one step after resume the loss is dominated "
               "by the weights, and momentum only compounds over later steps. Trace "
               "divergence separates them 33x (0.0105 restored vs 0.349 weights-only). "
               "Changed because the old threshold was unfit, not because the code "
               "failed it. Note also that exact bitwise resume would additionally "
               "require saving RNG state — the residual 0.0105 is stochastic layers "
               "drawing different values in a freshly-constructed model."),

    Spec("T0.05", 0, "Preemption survival",
         hypothesis="SIGKILL at a random step loses at most one checkpoint interval.",
         falsified_by="Corrupt checkpoint, or >1 interval of progress lost.",
         null_baseline="Non-atomic writes corrupt under kill.",
         metric="steps_lost", budget=Budget.CPU, depends_on=["T0.04"],
         control="The pre-fix NON-ATOMIC writer, replayed under the same SIGKILL hammer: it must produce a corrupt or unloadable checkpoint. If nothing corrupts, the hammer is not landing inside the write window and the atomic save was never tested (it passed by luck for four days for exactly that reason — LESSONS.md).",
         kills="Long GPU runs. Ephemeral VMs die without warning."),

    Spec("T0.06", 0, "Env/policy dimension contract",
         hypothesis="MuJoCo model nu equals the policy action dim, asserted at startup.",
         falsified_by="Mismatch that does not raise.",
         null_baseline="Current code silently writes a wrong-width tensor to mj_data.ctrl.",
         metric="nu_vs_action_dim", budget=Budget.CPU_FAST, depends_on=["T0.01"],
         control="Every WRONG width driven through the real write path (VirtualWorld.apply_action) must be REFUSED. Raw NumPy is not a sufficient guard: nu-1 and nu+1 raise, but width 1 is silently broadcast across all actuators, so a control that only tests +-1 would certify a writer that accepts a scalar.",
         kills="Every locomotion result."),

    Spec("T0.07", 0, "CPU throughput baseline",
         hypothesis="Measured env-steps/s on this ARM box, recorded for planning.",
         falsified_by="n/a — measurement, not a claim.",
         null_baseline="n/a", metric="steps_per_s", budget=Budget.CPU,
         depends_on=["T0.06"],
         control="The same timing harness with the BODY REMOVED — an empty loop must be orders of magnitude faster. If it is not, the reported rate is dominated by timing overhead and measures the harness rather than the subject.",
         notes="Measured 2026-08-04 (warmed, 3 trials, all spreads <1.1%): physics alone "
               "1831 steps/s; with the policy 11.8 steps/s — the forward costs 155x the "
               "physics it drives, so 2M steps = 47 CPU-hours. Sync vectorisation over 8 "
               "envs buys 1.03x, i.e. nothing, until the POLICY is batched with it. "
               "llm_enabled=False changes rollout speed by 0.0%: the 1.71B-param SmolLM2 "
               "is 6.9 GB resident and never runs in forward()."),

    Spec("T0.08", 0, "Metrics land in the ledger, and only the recorded spec moves",
         hypothesis="A run writes metrics retrievable by spec id; an untouched "
                    "spec reads NOT_RUN; a failing dependency yields BLOCKED "
                    "rather than a number; and a writer holding an HOURS-OLD "
                    "snapshot changes EXACTLY the entry it records — every "
                    "other entry keeps its newest metrics, its attempt count "
                    "and its amendments.",
         falsified_by="Missing or unparseable ledger entry; an untouched spec "
                      "reading as passing; a stale writer reverting an entry it "
                      "did not record, inflating its attempt count, or dropping "
                      "an amendment written after the snapshot.",
         null_baseline="The pre-2026-08-10 merge: having re-read the fresh file, "
                       "it wrote every entry the instance was holding back over "
                       "it. It reverts 3 of the 4 properties above and passes "
                       "the count-based concurrency check while doing so.",
         metric="roundtrip_ok", budget=Budget.CPU_FAST,
         control="The pre-fix merge replayed verbatim on the same battery "
                 "(`_prefix_merge_record`). It MUST revert at least one of the "
                 "four stale-writer properties while still landing its own "
                 "result — a control that fails for an unrelated reason "
                 "localises nothing.",
         kills="Nothing. It re-arms the ledger's durability claim, which the "
               "v1 spec asserted and the v1 test could not see.",
         notes="STRENGTHENED 2026-08-10 under the T1.02 precedent; the v1 "
               "verdict stays in this entry's history. v1 declared one "
               "property ('a run writes metrics retrievable by spec id'), no "
               "null and no control, while its test quietly checked five — and "
               "its concurrency property asserted `len(results) >= 15`, a "
               "COUNT. The real failure lost nothing by count: a 5.6 h T2.01 "
               "GPU poll recorded at 2026-08-10T01:17 and REVERTED six entries "
               "in place (LC.01, PG.3, PG.8, T0.08, T0.13, T0.15) plus five "
               "amendments, disguised as history because the fresh verdict was "
               "pushed into `history` with `attempt` incremented. v1 passed "
               "throughout. Same shape as T2.00's loss-ratio: the metric moved "
               "for reasons other than the thing claimed, and could not move "
               "for the thing itself."),

    Spec("T0.09", 0, "Colab T4 job round-trip",
         hypothesis="A script submits to Colab, runs on a T4, returns artifacts, VM torn down.",
         falsified_by="No artifact returned, or the VM persists.",
         null_baseline="n/a", metric="artifact_bytes", budget=Budget.GPU_SHORT,
         depends_on=["T0.03"],
         control="A job requesting an IMPOSSIBLE accelerator must report failure, not success. Silence is not success (LESSONS.md): a download that quietly failed once read as a passing round-trip.",
         notes="Verified working 2026-08-04: Tesla T4 15360MiB, torch 2.11.0+cu128."),

    Spec("T0.10", 0, "Kaggle job round-trip",
         hypothesis="Same contract via the Kaggle kernels API.",
         falsified_by="Kernel fails to run headless or artifacts are unreachable.",
         null_baseline="n/a", metric="artifact_bytes", budget=Budget.GPU_SHORT,
         depends_on=["T0.03"]),

    Spec("T0.11", 0, "Backend failover",
         hypothesis="If Colab refuses a GPU, the job runs on Kaggle unmodified.",
         falsified_by="A job that needs editing to switch backend.",
         null_baseline="n/a", metric="failover_ok", budget=Budget.GPU_SHORT,
         depends_on=["T0.09", "T0.10"],
         control="BOTH backends made impossible: submit() must report failure. A failover mechanism that reports success when there is nowhere to fail over to is reporting on itself, not on the backends.",
         notes="One job spec, two executors. The 30 free Kaggle hrs/week are the "
               "scarce resource; Colab absorbs the short jobs. "
               "STRENGTHENED 2026-08-05: the original passed by checking a job RUNS "
               "on either backend, and never checked that results come BACK. Two "
               "bugs hid behind that gap and surfaced only during a real failover "
               "(T1.08) — Colab keyed artifacts by full remote path while Kaggle "
               "keyed by filename, so lookups returned None after failing over; and "
               "jobs wrote to /content, which does not exist on Kaggle, whose "
               "kernels collect only /kaggle/working. Fixed with JACK_OUT plus "
               "basename keys; this spec must now assert an artifact is retrieved "
               "from BOTH backends."),

    Spec("T0.12", 0, "GPU-hour accounting",
         hypothesis="Every GPU run debits a weekly budget file; the ladder refuses "
                    "to launch past quota.",
         falsified_by="A run proceeds with the budget exhausted.",
         null_baseline="n/a", metric="quota_enforced", budget=Budget.CPU_FAST,
         depends_on=["T0.09"],
         control="Five named broken mechanisms: a Budget whose weeks "
                 "deliberately leak must FAIL isolation; the pre-2026-08-09 "
                 "`charge()` plus `submit()` loop must FAIL every billing "
                 "property; the pre-2026-08-11 dispatch loop, run against a "
                 "HEALTHY meter, must FAIL every receipt property — a submission "
                 "it makes must be indistinguishable from one never made; the "
                 "pre-2026-08-12 reattach meter (window opened at the slug epoch, "
                 "closed at the local clock) must FAIL the kernel-window "
                 "property; and the pre-2026-08-12 stale-writer `charge()` must "
                 "FAIL both concurrency properties.",
         notes="EXTENDED 2026-08-11 (7th overseer audit): a dispatch left no "
               "trace of its own except a budget charge, so a submission that "
               "was REPORTED but never made passed every gate the project owns "
               "— unchanged `gpu_budget.json` reads as 'nothing spent', "
               "unchanged ledger reads as 'not run', and commit 6b001e7's "
               "claim of an in-flight T1.02 poll was contradicted only by prose "
               "no gate reads. `submit()` now writes an append-only receipt to "
               "`gpu_submissions.jsonl` BEFORE each remote call and again after "
               "it, so absence of a receipt means not-dispatched. Asserted in "
               "both directions: a skipped backend must leave NO receipt, or "
               "the log would re-create the defect it exists to prevent. "
               "EXTENDED 2026-08-09 (2nd overseer audit): every property this "
               "spec asserted was checked against synthetic charges the test "
               "made itself, so it could not see that `charge()` billed failed "
               "jobs as work, re-billed a reattached kernel, and billed this "
               "box's wall clock rather than the metered window — week 31 "
               "closed at 37.4554 of 30.0 h and denied T1.02 its 0.7 h with "
               "this spec green. Added: failure bucket, per-job idempotency "
               "(across a reload), overrun marker, and submit()-level wiring. "
               "Strengthened only; no prior assertion was removed. Reconciling "
               "the meter against Kaggle's own reported kernel runtime needs "
               "network and a live kernel, and remains OPEN. "
               "REWRITTEN 2026-08-09: `weeks_isolated` was asserted after the "
               "quota was drained to its ceiling, where remaining() is "
               "max(0, 30-30) = 0 under EVERY implementation including total "
               "isolation failure. True by construction, and the week-key "
               "collision it exists to catch happened on 08-08 with this spec "
               "green. Now asserted at 28.0 of 30 h, with a leaky-Budget control."),

    Spec("T0.13", 0, "No gate in the ladder is decorative",
         hypothesis="Every metric a `_check` reads can change that check's verdict "
                    "at the operating point the run actually produced, and no "
                    "`_check` mixes `and` with an unparenthesised `or`.",
         falsified_by="A PASSing spec's gate references a metric that cannot move "
                      "its verdict, or contains an operator-precedence hazard.",
         null_baseline="n/a", metric="inert_gate_keys", budget=Budget.CPU_FAST,
         depends_on=["T0.08"],
         control="The pre-fix T0.09 `_check`, whose precedence bug made `ok`, "
                 "`cuda_available` and `matmul_finite` unreachable, must be "
                 "flagged by BOTH detectors.",
         notes="Written 2026-08-09 in response to the overseer audit, which found "
               "T0.09's gate bypassed by `and` binding tighter than `or` and "
               "observed that 'this pattern is not detectable by any current "
               "gate'. Law 1 says a capability is claimed only by a test that "
               "COULD have failed; a gate whose assertions cannot fire is that "
               "same disease one level up, in the machine rather than the "
               "science. This spec is the meta-gate: it audits the auditors."),

    # ===================================================================
    # TIER 1 — LEARNING PRIMITIVES. Can each trainable piece learn ANYTHING?
    # Applied per module. These are the tests that catch broken plumbing
    # before it is disguised as a research result.
    # ===================================================================
    Spec("T1.01", 1, "Overfit a single batch",
         hypothesis="Each trainable module drives loss below 1e-2 on ONE fixed batch.",
         falsified_by="Loss plateaus above 1e-2 after 500 steps.",
         null_baseline="A frozen module stays flat.",
         metric="final_loss", budget=Budget.CPU, seeds=3, depends_on=["T0.02"],
         control="Same module with requires_grad=False must NOT fit.",
         kills="The module. If it cannot memorise one batch it will never learn a task.",
         notes="The single highest-yield test in ML. Catches wrong loss, detached "
               "graph, shape bugs, dead ReLUs, and bad LR in one shot."),

    Spec("T1.02", 1, "Shuffled-target control (generalisation)",
         hypothesis="On HELD-OUT states, a structured task generalises and a shuffled "
                    "one does not.",
         falsified_by="Held-out error is the same whether or not a state->action "
                      "mapping exists.",
         null_baseline="Predicting the mean action, and the shuffled-task model.",
         metric="heldout_structure_advantage", budget=Budget.GPU_SHORT, seeds=3, depends_on=["T1.01"],
         control="Two of them. The shuffled task must NOT generalise. And a plain-MSE "
                 "reference learner MUST succeed — if the simplest possible model also "
                 "fails, the task is unlearnable and the run is void, not a failure.",
         kills="The premise that this architecture can learn a state->action mapping "
               "at all. If structure gives no held-out advantage, GPU hours cannot help.",
         notes="REDESIGNED TWICE, both times because the EXPERIMENT was wrong, never to "
               "flatter the model. Recorded in full because the failures are the useful "
               "part.\n"
               "v1 measured training FIT on one batch -> 0.999. Structured and shuffled "
               "were indistinguishable because a 58M network memorises 8 arbitrary pairs "
               "either way, so fit measures capacity. The original spec had predicted "
               "exactly this in its own null_baseline and it was built anyway.\n"
               "v2 measured generalisation, correctly, but drew 64 training samples for "
               "an obs_dim=348 input. That system is underdetermined; no architecture "
               "can pass it. The giveaway was beats_mean_baseline=0.415 — the trained "
               "model was WORSE than predicting the mean.\n"
               "v3 makes the task identifiable (2048 samples, a rank-8 true map) and "
               "adds the safeguard that should have been there from the start: a "
               "plain-MSE REFERENCE ARM. When the simplest possible learner also fails, "
               "the task is at fault, not the model, and the correct verdict is 'void' "
               "rather than 'the architecture cannot learn'. Every later spec that "
               "claims a learning result carries a reference arm for this reason."
               "  COVERS: generality (claim)"),
    Spec("T1.03", 1, "Gradient reaches every trainable parameter",
         hypothesis="After one backward, no trainable tensor has grad None or all-zero.",
         falsified_by="Any orphaned parameter.",
         null_baseline="Current model: 45,538,295 params (38.6%) receive no gradient.",
         metric="params_without_grad", budget=Budget.CPU, seeds=3, depends_on=["T0.01"],
         control="TWO PLANTED ORPHANS in the same brain, under the same loss, read by the same scan: a module that is never called (grad None) and a parameter reached by autograd but multiplied by zero (grad present and all-zero). BOTH must be reported. Added 2026-08-10 — without it, \"0 orphans\" was never shown to be a statement this measurement could contradict on this build. Gated on the plants by NAME, not on orphan_fraction: 80 planted params move the fraction by 1.6e-6, so the headline gate cannot tell a caught plant from a missed one.",
         kills="Silent dead weight. This test is the direct fix for the repo's disease."),

    Spec("T1.04", 1, "Weights actually move",
         hypothesis="||theta_after - theta_before|| > 0 for every trainable module.",
         falsified_by="A module whose weights are unchanged after N steps.",
         null_baseline="Frozen modules must show exactly zero.",
         metric="min_weight_delta", budget=Budget.CPU_FAST, seeds=3, depends_on=["T1.03"],
         control="lr=0 — NOTHING may move. Without it, a delta could be numerical noise in a model that is not learning at all.",
         kills="Any module that is wired but inert. A stuck submodule outside the pre-declared list fails this loudly."),

    Spec("T1.05", 1, "Frozen stays frozen",
         hypothesis="The pretrained trunk/LLM does not change during policy training.",
         falsified_by="Any delta in frozen parameters.",
         null_baseline="n/a", metric="frozen_delta", budget=Budget.CPU,
         depends_on=["T1.04"], seeds=3,
         control="AN UNFROZEN SENTINEL MUST MOVE. The identical module, attached OUTSIDE _PRETRAINED_PREFIXES and left trainable, must be re-randomised by construction AND updated by training. Added 2026-08-10 (OVERSIGHT 1.3): without it, two recorded zeros are satisfied by a measurement that reads zero for reasons of its own — an initialiser that never ran, an optimiser with an empty parameter list, a clone compared against itself.",
         notes="UnifiedBrain calls self.apply(_init_weights) — verify it does not "
               "re-randomise a loaded pretrained backbone. seeds 1 -> 3 on "
               "2026-08-10 alongside the control; the sentinel is randomly "
               "initialised, so one seed was one draw. HYPOTHESIS READS "
               "AGAINST THE PLASTIC-ONLY DECREE and is kept as a MECHANISM "
               "test only: nothing inside Jack ships frozen, but "
               "requires_grad_(False) still does not stop in-place init, and "
               "that trap waits for any tensor loaded from disk. No threshold "
               "touched (annotation was in registry_expansion.py where a "
               "reader of the spec could not find it)."),

    Spec("T1.11", 1, "Train/inference path parity",
         hypothesis="Every module on the INFERENCE path that produces joint commands "
                    "receives gradient from the TRAINING loss.",
         falsified_by="A module the runtime uses to drive actuators gets zero gradient "
                      "from the loss the pipeline optimises.",
         null_baseline="Before the fix: forward() gave action_head 271,889 gradient-"
                       "carrying params and ActionExpert exactly 0, while the runtime "
                       "drove the robot entirely from ActionExpert.",
         metric="inference_params_trained_frac", budget=Budget.CPU, depends_on=["T1.03"],
         control="A loss that touches only the unused head must FAIL this.",
         kills="Any training result. If the module producing joint commands never "
               "learns, the loss curve is measuring something the robot does not use.",
         notes="The defect this exists for, found 2026-08-04: the runtime path "
               "(VirtualWorld -> act_dual_system -> generate_actions_flow_matching -> "
               "ActionExpert) is decorated @torch.no_grad(), training went through a "
               "different module (action_head), and train_flow_matching_step — the only "
               "bridge — had zero callers in the entire repo. The system could have "
               "trained to convergence with the actuator module still at its random "
               "initialisation, and every metric would have looked correct."),

    Spec("T1.12", 1, "Flow matching actually denoises",
         hypothesis="After training on a fixed target, the flow-matching sampler "
                    "reconstructs that target far better than the untrained sampler.",
         falsified_by="Post-training reconstruction error is no better than pre-training.",
         null_baseline="The untrained sampler, and a random action of matched scale.",
         metric="reconstruction_improvement", budget=Budget.CPU, seeds=3, depends_on=["T1.11"],
         control="Shuffled targets must NOT be reconstructable — that would mean the "
                 "sampler ignores its conditioning.",
         kills="ActionExpert and the flow-matching path; fall back to a direct head.",
         notes="Gradient reaching a module proves plumbing, not learning. Flow matching "
               "trains a velocity field at random t but SAMPLES by integrating 10 Euler "
               "steps, so a correct loss can still integrate to nothing. This checks the "
               "sampler, which is what the robot actually runs."),

    Spec("T1.06", 1, "Numerical stability",
         hypothesis="No NaN/Inf in loss or grads over 1000 steps.",
         falsified_by="Any non-finite value.",
         null_baseline="n/a", metric="nonfinite_steps", budget=Budget.CPU, seeds=3,
         depends_on=["T1.01"],
         control="An ABSURD learning rate must break it and be REPORTED non-finite. A NaN detector that has never seen a NaN is decorative — this is the positive control T0.13 exists to demand."),

    Spec("T1.07", 1, "Not knife-edge on learning rate",
         hypothesis="Training succeeds across a 10x LR range.",
         falsified_by="Only one LR works.",
         null_baseline="n/a", metric="lrs_that_converged", budget=Budget.GPU,
         depends_on=["T1.01"],
         control="An ABSURD learning rate outside the claimed range must DIVERGE and lose its advantage over the mean-prediction baseline. Otherwise \"every LR worked\" is a statement about a task nothing can fail.",
         notes="A result that survives only at one LR will not survive a new task."),

    Spec("T1.08", 1, "Seed variance measured",
         hypothesis="Across 3 seeds the metric's std is small relative to the effect.",
         falsified_by="std >= the effect size being claimed.",
         null_baseline="n/a", metric="metric_std", budget=Budget.GPU, seeds=1,
         depends_on=["T1.01"],
         control="Seeds must ACTUALLY change the outcome: an arm in which the seed is ignored must show a std of zero. A small measured std is only a noise floor if the seed was plumbed through at all.",
         kills="Any single-seed claim in this repo.",
         notes="seeds=1 AT THE SPEC LEVEL, deliberately, for a spec ABOUT seed "
               "variance: the GPU job varies seeds [0,1,2] internally in one "
               "session, because three seeds in one job cost one clone and one "
               "queue wait, while spec-level seeds=3 launched three IDENTICAL "
               "jobs (_experiment ignores its seed argument) — 3x quota for zero "
               "information. That is also how a completed 37-min Kaggle result "
               "was thrown away on 2026-08-06: run #2 re-ran the staleness guard "
               "against the budget file run #1 had just dirtied. "
               "BUDGET CPU->GPU 2026-08-12, a correction not a re-scope: this "
               "spec has always dispatched a Colab job (est_hours=0.3, "
               "timeout_s=3000) and the declaration said cpu<10min, so "
               "`run._lock_for` routed it to the LOCAL CPU lock, which it then "
               "held at 0.00 cores for the whole remote poll. That is the exact "
               "failure the overflow slot exists to close, and the slot cannot "
               "fire because it requires every holder to be `remote_only` — read "
               "off THIS field. Measured: it blocked the builder's T0.22 run at "
               "08:36 with the box idle. T1.07 carried the same lie."),

    Spec("T1.09", 1, "Fits in T4 memory",
         hypothesis="Peak VRAM < 14 GB at the intended batch size.",
         falsified_by="OOM on a 16 GB T4.",
         null_baseline="n/a", metric="peak_vram_gb", budget=Budget.GPU_SHORT,
         depends_on=["T0.09"],
         control="An ABSURD batch size must either OOM or exceed the ceiling. The two branches are read as a disjunction because a run that OOMs has no peak to report — a necessary dead operand, and T0.13 exempts it explicitly (LESSONS.md, \"structure cannot separate honest redundancy from a disarmed assertion\")."),

    Spec("T1.10", 1, "CPU and GPU agree",
         hypothesis="Same seed, same data: CPU and T4 losses agree within tolerance.",
         falsified_by="Divergence beyond float32 accumulation error.",
         null_baseline="n/a", metric="cpu_gpu_delta", budget=Budget.GPU_SHORT,
         depends_on=["T1.09", "T0.02"],
         control="A DIFFERENT model seed must NOT agree. An agreement test whose comparison cannot separate two different networks would certify any two numbers as equal.",
         notes="Lets cheap CPU debugging predict GPU behaviour."),

    # ===================================================================
    # TIER 2 — COMPONENT COMPETENCE. Does each block beat its NULL?
    # Every test here names the baseline it must beat. "Loss went down"
    # is not a result.
    # ===================================================================
    Spec("T1.13", 1, "The grounding pairs are real",
         hypothesis="Every language-action pair fed to training is a genuine "
                    "observation, and the language is correlated with the motion "
                    "it is attached to.",
         falsified_by="Shuffling the language labels across the dataset leaves the "
                      "training loss statistically unchanged. If the words can be "
                      "permuted for free, they were never carrying signal.",
         null_baseline="A deliberately shuffled copy of the same dataset. Real "
                       "pairs must separate from it by more than seed noise (T1.08).",
         metric="label_permutation_loss_gap", budget=Budget.CPU,
         control="Also assert the data was not synthesised: no sample may be a "
                 "pure sinusoid, and the loader must fail loudly rather than "
                 "fabricate when a source is unreachable.",
         kills="Every downstream grounding claim. T2.06 and T2.07 measure whether "
               "the MODEL learned the mapping; this measures whether a mapping was "
               "present to learn. Passing those on fabricated data would be the "
               "original disease in a new place.",
         notes="Added 2026-08-04 at the owner's prompting - 'don't we need mocap to "
               "connect words to objects and actions'. This spec exists because the "
               "repo already made exactly this mistake: the MoCap URLs 404, the "
               "error is swallowed, and MoCapLoader.__getitem__ fabricates sinusoids "
               "paired with RANDOMLY DRAWN language labels. That is anti-training - "
               "it teaches that words do not predict motion. No spec currently "
               "checks the data, only the model, so this is the cheapest test on "
               "the ladder that could have caught the worst bug in it."
               "  COVERS: language (parent) (fixture)"),
    Spec("T2.01", 2, "Locomotion beats a random policy",
         hypothesis="Trained policy return exceeds random-action return by >5 sigma.",
         falsified_by="Return within seed noise of random.",
         null_baseline="Random policy on Humanoid: ~60-80 return.",
         metric="episode_return", budget=Budget.GPU_LONG, seeds=3,
         depends_on=["T2.00", "T1.08", "T0.09"],
         control="Untrained network with the same architecture.",
         notes="Budget GPU->GPU_LONG after v3: first healthy run (all seeds "
               "beat random, curve climbing at cutoff) failed only effect "
               "size, 2.21 sigma at 192K steps/seed. Threshold unchanged."),

    Spec("T2.02", 2, "Locomotion beats the honest MLP baseline",
         hypothesis="The chosen architecture beats a ~140K-param MLP actor-critic "
                    "at equal environment steps.",
         falsified_by="The MLP matches or wins.",
         null_baseline="RL-Zoo3 MuJoCo MLP: sac 6232+-280, td3 5567, tqc 7239 at 2M steps.",
         metric="return_at_matched_steps", budget=Budget.GPU_LONG, seeds=3,
         depends_on=["T2.00", "T1.08", "T0.10"],
         control="UNTRAINED versions of BOTH arms must miss the 3-sigma learning gate. This is the control that produced VOID rather than a verdict: the untrained MLP cleared random by 2.74 sigma, so a gate against random alone is nearly cleared by a network that has never received a gradient.",
         kills="The transformer policy. If a 140K MLP wins, use the MLP.",
         notes="RL-Zoo3 publishes NO PPO row for Humanoid. Treat PPO-Humanoid as "
               "a weak baseline and prefer SAC/TD3/TQC. Two edits 2026-08-08, "
               "both explained in t2_02's docstring: (1) depends_on was "
               "[T2.01], but T2.01's v4 FAIL (4.06 sigma, plateaued curve) is "
               "the EVIDENCE motivating this arbitration — gating the "
               "arbitration on its subject passing made the planned run "
               "structurally impossible. v4 established the loop trains (all "
               "seeds beat random); the deps now name the actual foundations. "
               "(2) metric was return_at_2M_steps; at the measured 106 "
               "steps/s, 2M x 3 seeds is ~16h for the transformer arm alone — "
               "over the 9h session cap to re-measure a plateau v4 already "
               "established. Comparison is at matched achieved steps "
               "(~640K/seed, inside the plateau); the hypothesis and kill "
               "criterion are unchanged."),

    Spec("T2.03", 2, "Pretrained vision features beat random features",
         hypothesis="A linear probe on frozen DINOv2/SigLIP features beats the same "
                    "probe on the current 0.24M from-scratch encoder.",
         falsified_by="From-scratch matches pretrained.",
         null_baseline="Random-projection features of equal dimension.",
         metric="probe_accuracy", budget=Budget.GPU_SHORT, seeds=3, depends_on=["T1.08"],
         control="Shuffled (frame, label) pairing: the same probe on the BEST "
                 "feature family must collapse to chance (1/4) when labels are "
                 "shuffled. If it does not, the probe is leaking labels and no "
                 "arm comparison means anything.",
         kills="use_pretrained_vision=False. Currently DINOv2/SigLIP are never loaded.",
         notes="COVERS: sight (fixture)\n"
               "RE-KINDED claim -> fixture 2026-08-13 (overseer B6, the "
               "builder's judgement call), AFTER its PASS and in the open: "
               "the spec's own next sentence has always said what it is — a "
               "measurement of the gap, not a demonstration that Jack sees — "
               "and the coverage-typing lesson predicted exactly this "
               "recurrence ('a (claim) spec whose test in fact measures its "
               "own apparatus'). Its PASS rides on the pretrained yardstick, "
               "which PLASTIC-ONLY bars from being inside Jack; crediting "
               "sight with it let the zero-pass instrument look away from a "
               "commitment that has no passing claim. Sight's claim specs "
               "are T3.01 (ablate vision) and OP.01.\n"
               "Control added 2026-08-13, before first run (strengthen-only, "
               "T1.02 precedent): run_spec refuses an undeclared control, and "
               "the spec had none. PLASTIC-ONLY caution (Review 2026-08-11): "
               "this is a MEASUREMENT of the gap between the current encoder "
               "and a pretrained yardstick, not a seating contest — a frozen "
               "winner cannot take the vision seat under the decree."),

    Spec("T2.04", 2, "Behaviour cloning on scripted trajectories",
         hypothesis="The action head reproduces scripted MuJoCo trajectories above "
                    "a nearest-neighbour baseline.",
         falsified_by="Fails to beat nearest-neighbour retrieval.",
         null_baseline="Nearest-neighbour lookup in the demo set.",
         metric="action_mse", budget=Budget.GPU_SHORT, seeds=3, depends_on=["T1.01"],
         control="The same action path trained on a shuffled (obs, action) "
                 "pairing must NOT beat the nearest-neighbour null. If "
                 "information-free supervision beats real retrieval, the "
                 "metric is not measuring imitation.",
         notes="Procedurally generated in-sim. Needs no external dataset — the "
               "CMU MoCap URLs 404 and the loader fabricates sinusoids.\n"
               "Control added 2026-08-14, BEFORE first run (strengthen-only, "
               "T1.02 precedent; run_spec refuses an undeclared control)."),

    Spec("T2.05", 2, "World model beats constant prediction",
         hypothesis="k-step latent prediction error < a persistence baseline.",
         falsified_by="Predicting 'next state = current state' does as well.",
         null_baseline="The strongest uninformed predictor, per seed: "
                       "min-MSE of persistence (copy current state) and "
                       "mean-state. (Redesign 2026-08-20: the 08-14 VOID "
                       "measured persistence UNINFORMATIVE at K=5, so a "
                       "persistence-only bar was clearable by learning "
                       "marginal statistics.)",
         metric="k_step_mse", budget=Budget.GPU, seeds=3, depends_on=["T1.01"],
         control="The same world-model path trained identically on a shuffled "
                 "(window, target) pairing must NOT beat the strongest "
                 "uninformed null (min of persistence/mean). If "
                 "information-free supervision predicts the future better "
                 "than the null that owns marginal statistics, the metric is "
                 "not measuring prediction.",
         notes="Error is measured in z-scored RAW observation space, not "
               "latent space: a latent ruler is owned by the model under test "
               "(a collapsed latent scores zero error on everything — the "
               "LC.03 twin-control scar, one level down). Horizon K=5 is the "
               "shipped imagination_horizon. Control added 2026-08-14, BEFORE "
               "first run (strengthen-only, T1.02/T2.04 precedent). "
               "REDESIGNED 2026-08-20 per the 08-14 VOID's pre-registered "
               "redesign facts (journal): null = min(mean, persistence) per "
               "seed, and the claim is additionally gated against the ridge "
               "reference arm (mse_wm <= mse_ridge every seed) — a world "
               "model that loses K-step prediction to one linear map has "
               "not demonstrated modelling. Strengthen-only; the VOID row "
               "stays in history."
               "  COVERS: fast/slow (fixture)"),

    Spec("T2.06", 2, "Language-action alignment beats chance",
         hypothesis="Contrastive retrieval of the right action anchor from a command "
                    "beats chance and a bag-of-words baseline.",
         falsified_by="At or near chance (1/n_anchors).",
         null_baseline="Chance = 1/len(ACTION_CATEGORIES); plus TF-IDF nearest match.",
         metric="retrieval_acc", budget=Budget.GPU_SHORT, seeds=3, depends_on=["T1.01"],
         control="Shuffled (command, action) pairing must collapse to chance.",
         notes="COVERS: language (parent) (claim)"),

    Spec("T2.07", 2, "Grounding generalises to held-out phrasings",
         hypothesis="Commands never seen in training map to the right anchor.",
         falsified_by="Accuracy collapses on held-out synonyms.",
         null_baseline="Memorising the ACTION_CATEGORIES synonym table.",
         metric="heldout_retrieval_acc", budget=Budget.GPU_SHORT, seeds=3,
         depends_on=["T2.06"],
         control="Label-shuffled twin (anchor-label lookup permuted by a fixed "
                 "derangement at the single supervision site): its held-out "
                 "accuracy vs TRUE categories must stay below the claim bar, "
                 "else the ruler leaks.",
         kills="SemanticActionAnchors as a grounding mechanism.",
         notes="THE test that separates understanding from a lookup table."
               "  COVERS: language (parent) (claim), generality (claim)"),

    Spec("T2.08", 2, "Curiosity drives coverage",
         hypothesis="Intrinsic reward increases state-space coverage over random exploration.",
         falsified_by="Coverage at or below random.",
         null_baseline="Uniform random actions; and epsilon-greedy (zero-reward "
                       "Q with the experiment's own machinery — the two must "
                       "agree, and any margin is then the reward signal's).",
         metric="state_coverage", budget=Budget.CPU, seeds=3, depends_on=["T1.01"],
         control="Time-permuted, magnitude-matched novelty reward (a uniform "
                 "draw from the agent's own past bonuses, fresh draw per step): "
                 "must NOT beat the random-walk null by the experiment's margin. "
                 "If information-free reward magnitude explores, the metric "
                 "credits the machinery, not the signal.",
         notes="Budget re-declared GPU->CPU 2026-08-13: the honest rig is the "
               "PG.4 rover fixture (~70 s/seed, numpy+MuJoCo, no torch), and a "
               "declared budget that machinery routes on must match the "
               "implementation. Mechanism arbitration is LT.03/LT.04, not this."
               "  COVERS: curiosity (claim)"),

    Spec("T2.09", 2, "Noisy-TV control",
         hypothesis="Injecting an unpredictable observation channel does NOT capture "
                    "the intrinsic reward.",
         falsified_by="The agent fixates on the noise channel.",
         null_baseline="A pure ICM agent is known to fixate (Burda et al.).",
         metric="noise_channel_dwell_frac", budget=Budget.GPU, seeds=7, depends_on=["T2.08"],
         kills="ICM alone. Forces RND or a learning-progress signal.",
         control="The null IS the control: naive percept-driven ICM in the same noisy world, same learner, same metrics, scored on the CLAIM gates. PG.4 certified that it fixates.",
         notes="Also covers the degenerate failure the owner should fear: an agent "
               "that 'explores' by twitching in place to maximise proprioceptive novelty."
               "  COVERS: curiosity (claim)"),

    Spec("T2.10", 2, "Memory retrieval beats recency",
         hypothesis="Retrieval scoring beats a pure-recency baseline on recall questions.",
         falsified_by="Recency-only does as well.",
         null_baseline="Return the k most recent events.",
         metric="recall_at_k", budget=Budget.CPU, seeds=3, depends_on=["T0.08"],
         control="TWO scorers that must LOSE, on the same seeded queries: pure recency, and similarity-only. Two because either one alone can be beaten by a scorer that is merely the other one.",
         notes="Generative Agents scoring: recency*0.5 + relevance*3 + importance*2."),

    Spec("T2.11", 2, "Skills are distinguishable",
         hypothesis="A classifier recovers the skill label from trajectories above chance.",
         falsified_by="Skills are indistinguishable — the MI objective collapsed.",
         null_baseline="Chance = 1/n_skills.",
         metric="skill_classification_acc", budget=Budget.GPU, seeds=3, depends_on=["T1.01"],
         control="Train the identical rig with the skill labels PERMUTED inside "
                 "every discriminator batch: same nets, same optimiser, same "
                 "number of gradient steps, same reward magnitudes reaching the "
                 "policy, no mutual information. If its skills are still "
                 "distinguishable, the metric is reading a per-skill lottery "
                 "rather than DIAYN. (Declared 2026-08-29 with the "
                 "implementation; the spec had no control field and run_spec "
                 "would have raised UndeclaredControl.)",
         kills="SkillDiscovery."),

    Spec("T2.13", 2, "Train to convergence, not to a step count",
         hypothesis="Training can be extended in increments until improvement "
                    "falls below seed noise, and the stopping point is decided by "
                    "measurement rather than by whoever got bored.",
         falsified_by="Held-out performance is still climbing when the criterion "
                      "fires, or the criterion never fires because per-increment "
                      "gains never drop below noise.",
         null_baseline="The seed-variance band from T1.08. An increment that "
                       "improves by less than that has not improved.",
         metric="increments_to_convergence", budget=Budget.GPU_LONG,
         depends_on=["T1.08", "T2.02"],
         control="Shuffle the increment order. A real convergence point must not "
                 "depend on which data arrived when.",
         kills="Any claim that a run is 'done'. Without this, 'trained' means "
               "'stopped', and the two are not the same thing.",
         notes="Added 2026-08-04 at the owner's request: 'at end i would want to "
               "run all steps until they stop showing growth'. This is the spec "
               "that makes that a measurement. It resumes from checkpoints rather "
               "than retraining, which is what T0.04 and T0.05 established is safe "
               "(33x optimiser-state fidelity, and a SIGKILL mid-write cannot "
               "corrupt the file). Every earlier training spec therefore has a "
               "budget that can be raised and rerun; this one decides when to stop "
               "raising it. Pair with --gate: extended training must not regress "
               "anything already demonstrated."),
    Spec("T2.12", 2, "Emotion model produces distinguishable states",
         hypothesis="PAD trajectories under different event streams are separable.",
         falsified_by="Indistinguishable from a random walk.",
         null_baseline="Random walk with matched variance.",
         metric="state_separability", budget=Budget.CPU, seeds=3, depends_on=["T0.02"],
         control="TWO. A variance-matched RANDOM WALK must not be separable, and SHUFFLED labels on the real trajectories must collapse to chance. The first rules out separability that any drifting scalar would show; the second rules out a classifier reading trajectory identity rather than emotion.",
         kills="EmotionalState as an input modality."),

    # ===================================================================
    # TIER 3 — ABLATION. Does each component EARN its parameters?
    # Anything that cannot show a contribution is deleted.
    # ===================================================================
    Spec("T3.01", 3, "Ablate vision", hypothesis="Removing vision measurably hurts a vision-dependent task.",
         falsified_by="No measurable drop.", null_baseline="Full system.",
         metric="delta_vs_full", budget=Budget.GPU, seeds=3, depends_on=["T2.03"],
         kills="The vision encoder.",
         control="Shuffled-label training: same architecture, same budget, "
                 "labels permuted. Its TEST accuracy MUST sit at chance "
                 "(|acc - 0.25| <= 0.10, ~4 sd at n=300). If it clears "
                 "chance the rig leaks episode identity into the frames and "
                 "the run is VOID, not evidence — the same direction T2.03's "
                 "registered control failed (0.0633 dev). "
                 "acc_shuffled_train is RECORDED, not gated (v3, 2026-08-21): "
                 "the registered run read 0.25 / 0.3167 / 0.25 against the "
                 "0.35 reference line, so this control is NOT evidence "
                 "against a distributional leak. Identity leakage is carried "
                 "by hash_overlap (0.0) and the pixel-shuffled arm (0.25 "
                 "every seed).",
         notes="COVERS: sight (claim)"),
    Spec("T3.02", 3, "Ablate proprioception", hypothesis="Removing proprioception hurts control.",
         falsified_by="No measurable drop.", null_baseline="Full system.",
         metric="delta_vs_full", budget=Budget.GPU, seeds=3, depends_on=["T2.01"],
         kills="The proprioception encoder.",
         notes="COVERS: proprioception (claim)"),
    Spec("T3.03", 3, "Ablate the world model", hypothesis="Removing it hurts sample efficiency.",
         falsified_by="Same sample efficiency without it.", null_baseline="Model-free control.",
         metric="steps_to_threshold", budget=Budget.GPU_LONG, seeds=3, depends_on=["T2.05"],
         kills="world_model (2.97M params, currently never invoked)."),
    Spec("T3.04", 3, "Ablate the hierarchical planner", hypothesis="Removing it hurts long-horizon tasks.",
         falsified_by="No drop on multi-step tasks.", null_baseline="Flat policy.",
         metric="task_success_rate", budget=Budget.GPU_LONG, seeds=3, depends_on=["T2.01"],
         kills="hierarchical_planner (37.17M params — larger than the backbone, zero call sites)."),
    Spec("T3.05", 3, "Ablate temporal memory", hypothesis="Removing 50-step memory hurts partially-observed tasks.",
         falsified_by="No drop.", null_baseline="Single-frame policy.",
         metric="delta_vs_full", budget=Budget.GPU, seeds=3, depends_on=["T2.01"],
         kills="temporal_memory (12.64M params, never passed memory=)."),
    Spec("T3.06", 3, "Ablate curiosity", hypothesis="Removing intrinsic reward reduces unprompted coverage.",
         falsified_by="Coverage unchanged.", null_baseline="Extrinsic-only.",
         control="Extrinsic reward PLUS a time-permuted, magnitude-matched "
                 "bonus — a uniform draw from the agent's own past novelty "
                 "bonuses, so the reward's distribution is preserved and only "
                 "its information is destroyed. It must NOT recover the "
                 "coverage the ablation cost (delta_shuf < DELTA_MIN). If it "
                 "does, the measured effect is reward magnitude or Q-value "
                 "noise, not curiosity, and the claim is void of content "
                 "whatever the experiment arm did. Declared 2026-08-29 with "
                 "the implementation; adding a control is strengthening.",
         metric="delta_coverage", budget=Budget.GPU, seeds=3, depends_on=["T2.08"],
         kills="IntrinsicCuriosityModule.",
         notes="COVERS: curiosity (claim)"),
    Spec("T3.07", 3, "Ablate mood conditioning", hypothesis="Mood measurably changes behaviour, not just text.",
         falsified_by="Identical action distributions across moods.",
         null_baseline="Mood token zeroed.", metric="action_dist_divergence",
         control="TWO at-chance arms the rig must hold, both classified with "
                 "real regime labels: (a) the ablation itself — the same "
                 "episodes with MovementMoodCoupling bypassed (raw base "
                 "actions), and (b) the registered null — mood token zeroed "
                 "through the live modulation path. Mood cannot reach the "
                 "action features except through the coupling, so both must "
                 "read ~chance (<= 0.40 vs 0.25); either clearing it means "
                 "the separability is not attributable to mood conditioning "
                 "and the run is VOID, not evidence. Plus ONE must-SUCCEED "
                 "probe: a reference arm (same net, same loss, adequate "
                 "budget) must reach speed span >= 0.30, proving the mood->"
                 "speed map is learnable by this rig — else a weak shipped "
                 "arm cannot be attributed to the shipped system and the run "
                 "is VOID. The shipped arm's own training strength is part "
                 "of the system under test: too weak to separate moods is "
                 "the hypothesis FAILING, never a VOID.",
         budget=Budget.GPU_SHORT, seeds=3, depends_on=["T2.12"],
         kills="MovementMoodCoupling as anything but cosmetics."),
    Spec("T3.08", 3, "Ablate the LLM", hypothesis="The frozen LLM improves command following over a bag-of-words encoder.",
         falsified_by="Bag-of-words matches it.", null_baseline="TF-IDF command encoder.",
         metric="command_success_rate", budget=Budget.GPU, seeds=3, depends_on=["T2.07"],
         kills="Carrying a 1.7B model. Decides SmolLM2 vs something larger or nothing.",
         notes="COVERS: language (parent) (claim)"),

    # ===================================================================
    # TIER 4 — COMPOSITION. Does adding B break A?
    # ===================================================================
    Spec("T4.01", 4, "Modality dropout robustness",
         hypothesis="Performance degrades gracefully when a modality is missing at test time.",
         falsified_by="Catastrophic failure when one sense drops.",
         null_baseline="Full-modality performance.",
         metric="degradation_curve", budget=Budget.GPU, seeds=3, depends_on=["T3.01", "T3.02"],
         notes="COVERS: one brain / unison (claim)"),
    Spec("T4.02", 4, "No modality collapse",
         hypothesis="Per-modality gradient norms stay within an order of magnitude.",
         falsified_by="One modality's gradient dominates by >10x — the others are ignored.",
         null_baseline="Balanced contribution (matched-information fixture: every "
                       "sense carries an equal, measured variance share of the target).",
         control="ONE, a plant the detector must catch: vision's fusion token is "
                 "wrapped forward-identity / backward-x100, and the measured vision "
                 "boundary gradient must dominate the other senses' minimum by >10x. "
                 "If the imposed dominance is not seen, the detector is blind and the "
                 "run is VOID, not evidence.",
         metric="max_modality_grad_ratio", budget=Budget.GPU_SHORT, seeds=3,
         depends_on=["T1.03"],
         notes="The documented failure where vision drowns proprioception. Detected "
               "only by instrumenting gradients per encoder."
               "  COVERS: one brain / unison (rule)"),
    Spec("T4.03", 4, "Fusion actually fuses",
         hypothesis="Shuffling ONE modality across the batch degrades performance.",
         falsified_by="No degradation — the modality is being ignored.",
         null_baseline="Unshuffled.", metric="shuffle_sensitivity",
         budget=Budget.GPU_SHORT, seeds=3, depends_on=["T4.02"],
         kills="CrossModalFusion. Distinguishes real integration from concat-and-project.",
         notes="COVERS: one brain / unison (claim)"),
    Spec("T4.04", 4, "Task interference",
         hypothesis="Training task B does not degrade task A beyond a set tolerance.",
         falsified_by="A drops >10% while learning B.",
         null_baseline="A trained alone.", metric="task_a_retention",
         budget=Budget.GPU_LONG, seeds=3, depends_on=["T2.01"]),
    Spec("T4.05", 4, "Full regression gate",
         hypothesis="Every passing Tier 0-3 test still passes after composition.",
         falsified_by="Any regression.", null_baseline="n/a",
         metric="regressions", budget=Budget.GPU_LONG, depends_on=["T4.04"]),

    # ===================================================================
    # TIER 5 — THE CLAIMS. The thesis stands or falls here.
    # ===================================================================
    Spec("T5.01", 5, "Physics pre-training transfers (THE thesis test)",
         hypothesis="Phase-0 SymPy physics pre-training improves downstream control "
                    "sample-efficiency vs identical architecture without it.",
         falsified_by="No difference beyond seed noise.",
         null_baseline="Same net, random init, same budget. Also: pre-trained on "
                       "SHUFFLED physics targets (structure-free control).",
         metric="steps_to_reward_threshold", budget=Budget.GPU_LONG, seeds=5,
         depends_on=["T2.01"],
         control="Shuffled-physics pre-training must NOT help.",
         kills="The project's entire differentiator. Run this EARLY and cheaply.",
         notes="5 seeds because this is the headline claim and the effect may be small."
               "  COVERS: generality (claim)"),

    Spec("T5.02", 5, "Physics violation detection",
         hypothesis="The model flags dynamics perturbed outside training distribution.",
         falsified_by="Cannot distinguish perturbed from nominal.",
         null_baseline="A simple prediction-error threshold detector.",
         metric="detection_auc", budget=Budget.GPU, seeds=3, depends_on=["T5.01"],
         kills="The 'motor must be weaker than expected' adaptation story.",
         notes="Must beat the trivial detector, or the neuro-symbolic framing adds nothing."),

    Spec("T5.03", 5, "Continual learning: forgetting measured",
         hypothesis="Sequential tasks retain prior performance above a replay-free baseline.",
         falsified_by="Catastrophic forgetting equal to naive sequential fine-tuning.",
         null_baseline="Naive sequential fine-tuning (expected: severe forgetting).",
         metric="backward_transfer", budget=Budget.GPU_LONG, seeds=3, depends_on=["T4.04"],
         notes="COVERS: plasticity (claim)"),

    Spec("T5.04", 5, "Plasticity does not die",
         hypothesis="After N consolidation cycles the model still learns a NEW task as "
                    "fast as a fresh model.",
         falsified_by="Steps-to-threshold on a novel task grows with cycle count.",
         null_baseline="Fresh model on the same novel task.",
         metric="plasticity_ratio", budget=Budget.GPU_LONG, seeds=3, depends_on=["T5.03"],
         kills="The word 'indefinitely'.",
         notes="Dohare et al., Nature 632:768-774 (2024). Instrument dormant-unit "
               "fraction, parameter-norm growth and per-layer effective rank every cycle. "
               "Without this you cannot tell 'converged' from 'can no longer learn'."
               "  COVERS: plasticity (claim)"),

    Spec("T5.05", 5, "Sleep consolidation beats online-only",
         hypothesis="The T2 offline GPU pass improves on the T1 online head alone.",
         falsified_by="No improvement from consolidation.",
         null_baseline="Online head with no consolidation.",
         metric="post_sleep_delta", budget=Budget.GPU, seeds=3, depends_on=["T5.03"],
         notes="COVERS: sleep (claim)"),

    Spec("T5.06", 5, "Unprompted exploration is real",
         hypothesis="Left alone, the agent visits more distinct states than a scripted "
                    "idle behaviour, and its choices depend on what it has already seen.",
         falsified_by="Coverage matches a scripted wander.",
         null_baseline="Current code: a hardcoded string on a frame counter "
                       "(VirtualWorld.py:807).",
         metric="unprompted_coverage", budget=Budget.GPU, seeds=3, depends_on=["T3.06"],
         notes="COVERS: curiosity (claim)"),

    Spec("T5.07", 5, "Behaviour visibly changes after training",
         hypothesis="A blind human rater distinguishes trained from untrained episodes.",
         falsified_by="Rater at chance.",
         null_baseline="Untrained checkpoint, same seed and scene.",
         metric="rater_accuracy", budget=Budget.GPU_SHORT, seeds=3, depends_on=["T2.01"],
         notes="The only test whose result the owner can verify with his own eyes. "
               "Keep it in the ladder for exactly that reason."),

    # ===================================================================
    # TIER 6 — INTEGRATION.
    # ===================================================================
    Spec("T6.01", 6, "Full episode completes",
         hypothesis="A full companion session runs N minutes with no crash or NaN.",
         falsified_by="Any crash, hang, or non-finite action.",
         null_baseline="n/a", metric="minutes_survived", budget=Budget.CPU_LONG,
         depends_on=["T4.05"]),
    Spec("T6.02", 6, "Long-run stability",
         hypothesis="Hours of continuous operation without drift into degenerate behaviour.",
         falsified_by="Action saturation, mood lock, or memory unbounded growth.",
         null_baseline="n/a", metric="hours_stable", budget=Budget.CPU_LONG,
         depends_on=["T6.01"]),
    Spec("T6.03", 6, "Cross-session persistence",
         hypothesis="Save, restart, and the companion recalls prior interaction.",
         falsified_by="State lost or corrupted across restart.",
         null_baseline="Fresh instance with no memory.",
         metric="recall_after_restart", budget=Budget.CPU, seeds=3,
         depends_on=["T2.10", "T0.05"],
         control="A corrupted (truncated) save file must be rejected loudly by "
                 "load_all — silent acceptance of corruption is the falsifier."),
    Spec("T6.04", 6, "Everything at once, end to end",
         hypothesis="With every modality live simultaneously - vision, "
                    "proprioception, touch, audio, language - Jack takes a spoken "
                    "instruction, acts on the right object, and keeps learning, in "
                    "one continuous episode.",
         falsified_by="Any capability demonstrated in isolation degrades below its "
                      "own single-modality result once the others are running.",
         null_baseline="Each capability's own Tier 2 score, measured alone. "
                       "Integration must not cost more than seed noise (T1.08).",
         metric="integrated_vs_isolated_ratio", budget=Budget.GPU_LONG,
         depends_on=["T4.05", "T5.07", "T6.01"],
         control="Silence one modality at a time mid-episode. Behaviour must "
                 "degrade gracefully and specifically, not collapse or freeze - a "
                 "system that does not notice a missing input was not using it.",
         kills="The claim that the parts compose. Tiers 2-5 prove each capability "
               "separately, and separate is not together.",
         notes="Added 2026-08-04 at the owner's request for a phase that tests all "
               "aspects together at the end. T6.01-T6.03 only establish that a full "
               "episode completes, stays up, and persists across sessions - "
               "liveness, not capability. This is the one that asks whether the "
               "whole thing actually works as one system, and it is deliberately "
               "the last spec on the ladder."),
]


# The expanded ladder: GOAL.md made falsifiable (playground, unified brain,
# curiosity, memory). Details in docs/MASTER_PLAN.md and docs/research/.
from .registry_expansion import EXPANSION
LADDER.extend(EXPANSION)

BY_ID = {s.id: s for s in LADDER}


def tier(n: int) -> list[Spec]:
    return [s for s in LADDER if s.tier == n]


def ready(ledger) -> list[Spec]:
    """Specs whose dependencies all pass — the legitimate next moves."""
    from .protocol import Status
    return [s for s in LADDER
            if ledger.status(s.id) is not Status.PASS and not ledger.blocked_by(s)]
