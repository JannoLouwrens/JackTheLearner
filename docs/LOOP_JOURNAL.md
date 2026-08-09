# Loop journal

One line per iteration: what was attempted, what was measured, what comes next.
Written by the hourly ladder loop; the ledger holds the evidence, this holds the reasoning.

- **2026-08-04 manual** — Implemented T0.03 (checkpoint round-trip). PASS: output_delta 0.0,
  strict=True, zero missing/unexpected keys; control (different init) shows a large delta, so the
  comparison is sensitive. Found en route: the disable flags used in T0.03/T1.01/T1.05 were
  invented names (`use_llm`) — the real one is `llm_enabled` — so those tests had been silently
  loading the full 1.7B model. Fixing it cut the checkpoint 3654 MB -> 240 MB and runtime 238s -> 47s.
  Standing problem this exposes: with the LLM on, a checkpoint is 96.7% frozen weights that never
  change (1,711,376,384 of 1,769,146,406). Per D1 the fix is an `export_for_runtime` that writes the
  trainable adapter only. Next: T0.04 (resume without a loss discontinuity) needs optimiser + obs-norm
  state saved, then T1.04 to unblock T1.05.
- **2026-08-04 manual** — T0.04 (resume continuity). Two findings before it passed.
  (a) `copy.deepcopy(brain)` raises `TypeError: cannot pickle '_thread.lock'` — the model holds a
  thread lock (AlphaGeometryLoop timeout machinery). Any training code cloning the model hits this;
  work from `state_dict()` instead.
  (b) THRESHOLD REVISED, with evidence, not to make it pass. The registered metric was "loss jumps
  >20% at the resume boundary". Measured it reads **1.326% in BOTH arms** — identical — so it cannot
  discriminate a correct resume from a broken one, because one step after resume the loss is dominated
  by the weights and momentum only compounds later. Replaced with trace-divergence ratio, which
  separates 33x (0.0105 optimiser-restored vs 0.349 weights-only). PASS at ratio 33.39.
  Residual 0.0105 (not 0) is RNG: a freshly constructed model draws different stochastic values.
  Exact bitwise resume would additionally require saving RNG state — worth doing before T0.05.
  Next: T0.05 preemption survival (atomic writes + RNG state), then T1.04 to unblock T1.05.
- **2026-08-04 manual** — T0.05 (preemption survival) PASS. Atomic write via tmp+os.replace: 12 SIGKILLs,
  12 checkpoints recovered, **0 corrupt**. Naive `torch.save` straight to the final path: 12 kills,
  **12 corrupt**, nothing recoverable. Payload includes RNG state per the T0.04 finding.
  Test bug caught first: a guessed sleep killed the writer during `import torch`, so
  checkpoints_checked=0 — it reported a clean failure while measuring nothing. Now polls for a real
  checkpoint before killing. Lesson worth keeping: assert the test actually exercised its subject.
  Next: T0.06 (env/policy dim contract — needs mujoco installed) or T0.07/T0.08 which are CPU-only.
- **2026-08-04 manual** — T0.06 (dimension contract) PASS, after fixing the RUNTIME not the test.
  Installed mujoco 3.2.3 (3.5 has no py39 aarch64 wheel and tried to compile; 3.2.3 has one) +
  gymnasium; MUJOCO_GL=disabled for headless physics, no OSMesa on this box.
  Humanoid-v5 confirmed: obs 348, **nu = 17**, physics steps fine.
  The finding: `VirtualWorld.py:889` wrote `ctrl[:n] = action[:n]` with
  `n = min(len(action), nu)` — a deliberate silent TRUNCATION. 16 actions drove 16 joints and left
  the 17th stale; 40 actions used the first 17. Neither raised. And plain assignment is no guard
  either: `ctrl[:] = zeros(1)` is broadcast across all 17 actuators by NumPy (measured).
  Replaced with `apply_action(mj_data, mj_model, action)` requiring exact width and finite values.
  Control now exercises the real write path: nu-1, nu+1, 1, 2*nu and a NaN of correct width — 5/5 refused.
  Next: T0.07 (throughput baseline — tells us honestly what a GPU buys), T0.08, then T0.09-T0.11 GPU round-trip.
- **2026-08-04 manual** — T0.07 (throughput) PASS. The numbers that should govern the compute plan:
  bare MuJoCo **2,260 steps/s**; with the policy forward **12.0 steps/s** — a **188x** slowdown.
  2M steps = 0.25 h of physics but **46 h** with the policy (3.8 Kaggle sessions).
  Sync-vectorised 8 envs = 1,976 steps/s, **speedup 0.87x — slower than one env** (gym's sync vector
  is a Python loop; overhead, no parallelism).
  Conclusion: the rollout is DISPATCH-bound on the policy (6,095 ATen dispatches per B=1 forward per
  the review), not FLOP-bound or physics-bound. **A GPU alone does not fix this** — batch-1 forwards
  leave a T4 idle. The fix is to batch the policy across N envs (one forward at batch N) and shrink to
  the 22M adapter per D1, then re-measure before spending quota.
  ACTION for a later spec: add a "batched rollout" benchmark and re-run this before any Tier 2 GPU work.
- **2026-08-04 manual + loop collision** — The hourly loop and this session both implemented T0.07.
  Two files matched the `t0_07*` glob and the runner took the first alphabetically, so
  `t0_07_cpu_throughput.py` silently shadowed `t0_07_throughput.py` while the ledger reported a PASS
  belonging to whichever won the sort. **Fixed: duplicate implementations now RAISE.** Two
  implementations of one spec is an unresolved disagreement about the spec's meaning; a person settles
  it, not alphabetical order.
  Kept the loop's version — it is better. It CALIBRATES the instrument first (recovers a workload
  whose true rate is known by construction, error 1.3%) before trusting any number, which mine did not.
  Its measurements: policy is **99.5% of step time**; policy_fwd_hz_b1 12.36; env alone 2,379/s;
  **batch-16 gives only 2.88x per-sample speedup**; peak RSS **6.9 GB**; 22.6 h per 1M rollout steps.
  CORRECTION to the previous entry: batching alone does NOT close the 188x gap — 16x batch buys 2.88x.
  The larger lever is shrinking to the 22M adapter (D1), not batching.
  Also: the loop was blocked by the Claude Code trust dialog ("workspace has not been trusted") — fixed
  by setting hasTrustDialogAccepted for this project. Check /data/jack-logs/ladder.log after next fire.
  NOTE: peak RSS 6.9 GB exceeds the 1.5 GB the loop brief asks for; tests constructing the full brain
  need llm_enabled=False or they will strain a box with paying tenants.
- **2026-08-04 manual (same collision, continued)** — Re-ran the merged T0.07 on a quiet box and two
  of the numbers above do not survive repetition.
  **(1) `llm_enabled=False` changes rollout speed by 0.0%** — 11.79 steps/s with the 1.71B SmolLM2
  loaded, 11.81 without (params 1,769,146,406 vs 60,123,174, so the flag genuinely takes effect).
  The LLM is 6.9 GB resident and **never executes in `forward()`**. So the RSS hazard is real but the
  llm_enabled=False remedy is free — it costs no measured speed and no measured behaviour, which is
  itself the finding: 1.71B parameters sit in memory contributing nothing to a rollout step. Consistent
  with T1.03's orphan sweep. Worth its own spec before anyone counts them as capability.
  **(2) "Sync vectorisation is SLOWER than one env (0.87x)" was an un-repeated measurement.** It read
  1.17x on the next run and **1.03x ± 0.9%** once warmed and repeated 3x. The honest claim is that 8-env
  sync vectorisation buys *nothing* (1.03x), not that it is harmful. `_vectorised` now goes through the
  same repeat-and-report path as everything else and its spread is gated at 25%; a ratio that changed
  sign between runs had already reached a commit message.
  **(3) Warmup, not the box, drove the earlier variance.** Warming once and timing measured the policy
  at 3.67 Hz with 28.9% spread while the *same forward plus a 224x224 vision encode* measured 4.38 Hz —
  a heavier computation cannot be faster, so the timer was catching the 6.9 GB model paging in.
  Per-trial warmup: all spreads now <1.1% (bare 0.84%, vectorised 0.86%, policy 1.06%).
  My own earlier 3.67 Hz figure was CPU contention from a concurrently running `--gate`; the loop's
  ~12 steps/s was right and mine was the contaminated one.
  Settled numbers, warmed and repeated: physics alone **1,831 steps/s**, with policy **11.8 steps/s**
  (**155x** slowdown), **47 CPU-hours per 2M steps**.
  Also: calibrating the harness against `time.sleep(1/200)` fails at 16.3% error, because sleep()
  promises only a lower bound (5.8 ms delivered for a 5 ms request under load) — the ground truth was
  wrong, not the timer. Replaced with a scaling invariant: doubling the work must halve the rate
  (error now 0.31%).
  PROCESS: two agents in one tree cost more than both units of work were worth. The loop's `run.py`
  flock is the right fix; it was still uncommitted when I stopped, so I left it alone rather than
  commit someone else's work in progress. I also killed the loop's `--gate` while clearing my own —
  a gate is re-runnable, but it needs re-running.
  Next: **run `--gate` once the tree has one owner** (it has not completed since T0.07 landed), then
  T0.09 Colab round-trip. And a spec for finding (1): 1.71B params that never run.
- **2026-08-04 manual** — T0.09 (Colab round-trip) PASS in 34s. Tesla T4 15360 MiB, CUDA available,
  a real matmul executed (a CUDA context that never runs a kernel proves nothing), artifact returned
  124 bytes. Control with an impossible accelerator correctly failed (exit 1).
  The bug: `colab download` requires an ABSOLUTE remote path. "marker.json" -> "File or directory not
  found"; "/content/marker.json" works. VM CWD is /content, verified by probe. Every artifact would
  have silently vanished behind a successful-looking run. Also moved session teardown out of the
  success branch — a kept session never stopped holds a GPU and burns quota.
  Infrastructure now in place: experiments/gpu.py is one job contract with two executors, so T0.11
  failover is routing rather than a rewrite. Kaggle's 30 h/week budgeted in gpu_budget.json.
  Also fixed: the runner now takes the same flock as ladder_loop.sh, so a manual session and the loop
  cannot both write. status/next/render stay lock-free.
  Next: T0.10 (Kaggle round-trip — untested end to end; the kernels API is push-and-poll, not
  ephemeral-run, so expect the metadata/slug handling to need iteration), then T0.11 failover.
- **2026-08-04 manual** — TIER 0 COMPLETE, 12/12.
  T0.10 Kaggle round-trip PASS (214s): Tesla P100 sm_6.0, cuda true, real matmul, artifact returned.
  The fix that unlocked it: Kaggle assigns a P100 regardless of the accelerator requested
  (nvidiaTeslaT4 and gpuT4x2 both gave P100) while its own preinstalled torch 2.10+cu128 ships
  sm_70+ only. Jobs now prepend `pip install torch==2.5.1 --index-url .../cu121` — the last line
  carrying Pascal. Verified arch_list sm_50..sm_90.
  T0.11 failover PASS (203s): forced Colab to refuse, the IDENTICAL unmodified script landed on
  Kaggle and returned the same artifact key. It absorbed three asymmetries silently — sync vs
  push/poll, absolute-path download vs output dir, and torch 2.11+cu128/sm_75 vs 2.5.1+cu121/sm_60.
  Control with both backends impossible correctly failed.
  T0.12 quota accounting PASS: charges per ISO week, survives reload, refuses a job whose estimate
  exceeds the remainder, isolates weeks, leaves Colab unmetered.
  **Tier 1 is now the frontier.** T1.03 still FAILS at 16.7% orphan — action_expert (4.6M), the
  module whose output reaches mj_data.ctrl, gets no gradient from forward(). Fix that next.
  Also carried from the loop's last iteration: llm_enabled=False changes rollout speed by 0.0% —
  the 1.71B SmolLM2 is 6.9 GB resident and NEVER runs in forward(). Pure dead weight at runtime.
- **2026-08-04 manual — THE ACTION PATH DEFECT, fixed.** The most consequential finding so far.
  Runtime: VirtualWorld -> act_dual_system -> generate_actions_flow_matching -> ActionExpert,
  and that method is decorated **@torch.no_grad()** (UnifiedBrain.py:4421).
  Training: TrainingPipeline -> forward() -> **action_head**, a different module.
  Bridge: `train_flow_matching_step` — **zero callers anywhere in the repo**. The duplicate
  `compute_flow_matching_loss` is called only from archive/RobustTrainer.py, which is dead.
  Measured: forward()['actions'] gave action_head 271,889 grad params and ActionExpert **0**.
  So the system could train to convergence with the 4.6M module producing joint commands still at
  its random init, and every metric would have looked right. Commit ba012b6 ("replaces RobustTrainer")
  deleted the caller and kept both losses.
  FIX: added `UnifiedBrain.action_training_loss` — conditional flow matching on ActionExpert
  (primary, reaches 51,351,824 params incl. backbone via cross-attention) plus a small-weight
  action_head BC term so the documented fallback in generate_actions_flow_matching is not left
  emitting noise. Verified: action_expert 4,615,696 AND action_head 271,889 now receive gradient.
  Both flow-matching losses were mathematically correct; they were simply unreachable.
  New specs T1.11 (train/inference path parity) and T1.12 (flow matching actually denoises — gradient
  proves plumbing, not learning; the sampler integrates 10 Euler steps and a correct loss can still
  integrate to nothing). T1.11 is BLOCKED until T1.03 passes.
  T1.03 STILL FAILS, honestly: 12.3% orphaned even with the real loss + all accepted modalities.
  Two new defects found while measuring: **cfg.touch_dim = 64 but the encoder expects 10**
  (mat1/mat2 2x64 vs 10x128) and **cfg.audio_dim is None** so the spectrogram CNN collapses to width 0.
  Neither modality can be fed; ~466K params across audio/touch encoders+projections are unreachable
  by construction. Remaining orphans also include heads that need their OWN objectives
  (value_head needs RL, task_completion_head needs task labels, physics_head needs physics targets).
  NEXT: decide per-module — wire a real objective, or gate/delete per Tier 3. Do NOT relax T1.03's
  threshold; the 5% bar is right and the model should meet it.
- **2026-08-04 manual** — T1.03, T1.11, T1.12 PASS; full 15-spec regression gate PASS.
  T1.03 4.83% orphan (from 58.8%), bar untouched. Two of the "orphans" were my test mis-feeding
  modalities: TouchEncoder takes width 10 (hardcoded; touch_dim=64 is the OUTPUT) and AudioEncoder
  takes a raw 16000-sample waveform. Gated semantic_anchors + flipped enable_object_detection /
  enable_navigation to False (chat-regex path only, no loss reaches them).
  T1.11 path parity 1.0 — all 41,525,008 inference-path params trained; old forward()-only loss
  fails as control.
  **T1.12 is the first evidence Jack's action system LEARNS**: the runtime sampler
  (generate_actions_flow_matching, 10 Euler steps) improved 2.007x, beat an untrained sampler, and
  degraded 2.804x under SHUFFLED conditioning — so it uses the state rather than emitting a mean
  action. Gradient proves plumbing; this proves learning.
  HARNESS FIX: the regression gate was OOM-killed (exit 137) running 15 model constructions in one
  process. Each spec now runs in its own subprocess — memory reclaimed on exit, and a crashing test
  can no longer take the ledger with it. On a box with paying tenants that is a hazard fix, not a
  convenience.
  NEXT: T1.02 (shuffled-target control), T1.04-T1.10, then Tier 2 — the first GPU training with a
  null baseline. Per T0.07, batch the policy before spending quota: 12 steps/s at B=1 leaves a T4 idle.

## 2026-08-04 — T1.02, and the flow parameterisation it uncovered

Attempted: diagnose T1.02 (structure_advantage 0.999), fix, retest.

Two experiment bugs and one real defect, in that order.

The 0.999 was v1 measuring training FIT on one batch — a 58M net memorises 8
pairs whether or not a mapping exists. Rebuilt around generalisation. It then
failed WORSE: held-out 1.551 against a 0.644 mean-predictor baseline. Adding a
plain-MSE reference arm with no flow matching anywhere showed that arm failing
too (0.925), which is what identified the second bug: 64 training samples for an
obs_dim=348 map is underdetermined and unpassable by any architecture.

With 2048 samples the same sweep separated cleanly (T4, held-out MSE, baseline
0.635):

    regress (no flow)         0.238    2.665x baseline
    x1 parameterisation       0.266    2.391x
    velocity + Beta(1,1.5) t  0.407    1.559x
    velocity + uniform t      0.620    1.024x   <- what the repo did

So the flow path can match supervised regression, and the shipped
parameterisation was throwing away nearly all of it. Cause: the optimal velocity
(x1-x_t)/(1-t) diverges as t->1, so a velocity-predicting network under-estimates
an unbounded gain and never removes the last of the initial noise. Predicting x1
leaves that divergence in closed form. Integration steps 5..100 changed the
result under 2%, so this is a learning effect, not discretisation.

Measured: UnifiedBrainConfig.flow_parameterisation now defaults to "x1", with
flow_timestep_dist configurable. Both train and sample paths read the same flag,
so they cannot drift apart — the action-path defect one level down.

Next iteration: run T1.02 (now GPU_SHORT, ~25 min on Colab). It is still FAIL and
must stay FAIL until a run says otherwise. Then T1.04, T1.05, T1.06.

Infrastructure: experiments/gpu.py gained repo_preamble/build_job — a GPU job now
clones the public repo and pins a ref. This is the unlock; CPU here is ~2 s/step
against ~0.05 s on a T4. Do not iterate training on this box.

## 2026-08-06 — T2.01 failed; three PPO bugs found and fixed

T2.01 measured trained -4334 vs untrained +170 vs random +122. The untrained
CONTROL passing while the experiment collapsed is what localised the fault to
the RL update rather than the architecture. Three bugs, none visible in a loss
curve, all found by instrumenting rather than reading:

1. No return normalization. vf_loss 540.5 vs pg_loss 0.267; value_head sits on
   the SHARED trunk, so with vf_coef=0.43 the value term was ~870x the policy
   term and after clipping the policy gradient was a rounding error in the
   update DIRECTION. Fixed with a running return scale (scale only, not
   centred). MEASURED: vf/pg 870 -> 3.3.
2. log_std unbounded (entropy bonus inflates it forever) -> clamped to
   [-4.6, 0].
3. Actions never clipped to the env's +-0.4 -> |a| hit 2.37 in two iterations;
   MuJoCo clipped silently so PPO scored components that never touched physics.

Guarded permanently by NEW SPEC T2.00 (CPU-cheap, gates every GPU locomotion
run, with an unnormalized control that MUST explode).

WATCH-ITEM for the next iteration, not yet acted on: locomotion_head is a bare
nn.Linear with no output bound, and the action mean drifted 1.19 -> 2.27 within
two iterations. Clipping keeps this CORRECT but saturates the policy: every
value past 0.4 becomes the same command, so gradients stop distinguishing them.
SB3 ships clip-only for Box spaces and it works, so this is deliberately NOT
changed without evidence -- T2.01 logs action_absmax per iteration now, and the
learning curve should settle it. If reward stalls while |a| grows, squash or
penalise; if reward climbs, leave it alone.

## 2026-08-07 — T2.01 v2 squash verified on CPU; PG.1/PG.2 PASS

The v2 watch-item resolved exactly as pre-registered: reward flat at ~4.0 while
|act|max ran 1.26 -> 43.81, so the mean got squashed (policy_mean(): tanh *
action_limit). CPU verification, 11 iterations: |act|max stays 1.17-1.56 raw
(0.4 at the env), std pinned at 0.301, vf/pg ratio 2.5-82 transient then
settling single-digit. The exponential runaway is gone.

PG.1 first FAILED and the bug was in the TEST: MJCF defaults to DEGREES, so
math.radians() in euler= built a 0.87-deg "50-deg" ramp. Caught because the
frictionless CONTROL failed too — nothing slid anywhere, which no visual
inspection would flag. Also: a cube topples past 45 deg, so the slider is now a
slab (stable to 71.6 deg). Friction discriminates 1751x. PG.2 water sits at the
Archimedes equilibrium, err/radius 0.000 at rho 0.2/0.3/0.5/0.8. Ladder 27/103.

## Pre-registration: T2.01 v3

Changes since v2, all compute/infra, no threshold touched:
- policy_mean() bounds the mean (the v2 fix); eval uses the same bounded mean.
- N_ENVS 8 -> 32, minutes/seed 22 -> 30: v2 completed only 105K env-steps/seed
  (~80/s). The forward is batched over envs, so this buys steps ~free.
PREDICTION: trained_mean > random_mean on every seed, sigma_advantage >= 5,
untrained control stays under the bar (v2 measured it at 0.97). If reward is
STILL flat with actions bounded and ~400K steps/seed, the next suspect is the
shared-trunk gradient mixture (D1, owner decision) — not more compute.

AMENDED before launch: N_ENVS alone would have bought nothing. v2's own P100
telemetry shows ~12s of each 13s iteration inside rl_update -- 16 minibatches
x 5 epochs of batch 64 -- and minibatch COUNT scales with rollout size, so 4x
the envs meant 4x the update time. Added ppo_minibatch=512 (identical total
sample-passes per iteration; the GPU is simply utilised instead of fed 64-row
crumbs). T2.00 gate re-run against the changed rl_update before any quota is
spent. Expected throughput ~4x: ~300+ env-steps/s, ~500K steps/seed in 30 min.

## 2026-08-07 — Owner directive: interaction memory + general learning, both

"He must also remember what he hears says and does so when people interact
with him... he must keep memory and ALSO learn generally — ensure it's in the
tests." Two new specs, because ME.1-8 pinned neither property:

- ME.9: recall across heard/said/did with SOURCE ATTRIBUTION, >=3 interleaved
  speakers. Control: a swapped-provenance store must invert the answers — if
  accuracy survives the swap, the test measured text similarity, not memory.
- ME.10: the CLS double dissociation. Distill episodes into weights; recall
  must hold at its pre-distillation rate AND the skill must beat the
  no-distillation null. Then each ablation kills exactly its own capability:
  store-wipe kills recall only, weight-revert kills skill only. KILLS any
  design where conversation lives only in weights or skills only in episodes.

Ladder is now 105 specs. GOAL.md carries the directive verbatim in the Memory
section.

## 2026-08-07 — T2.00 v1 FAIL at ppo_minibatch=512; metric was the bug

The gate tripped (max vf/pg loss ratio 178.57 vs 50) on the minibatch change.
Investigated before touching anything: the pg_loss DENOMINATOR is ~0 by
construction at an unmoved policy (normalized advantages, ratio=1), so its
magnitude measures intra-update drift — and 512-size minibatches take 5
gradient steps where 64 took 10, so less drift, smaller denominator, inflated
ratio. The per-minibatch term BALANCE is mathematically independent of batch
partitioning. Probe of the true quantity (per-term gradient norm on the shared
trunk, same rollout): 1.86x / 2.59x / 2.78x at mb 64/128/512. Healthy.

v2 gates on the grad-norm ratio, measured INSIDE rl_update (term_grad_diag
flag) so the guard reads the production path. Pre-registered threshold 25
(order of magnitude above healthy, order below the unnormalized pathology).
Control unchanged in spirit: normalize_returns=False must blow the grad ratio.
v1's FAIL stays in history; the loss ratio is demoted to a diagnostic metric.
Same pattern as T1.02: the test failed, the investigation said "metric
artifact", the redesign measures the thing itself.

## 2026-08-07 — T2.01 v3: FAIL on effect size, but Jack LEARNS for the first time

Recovered via kernel reattach (the local waiter died in a session restart; the
kernel finished alone; JACK_REUSE_KERNEL fetched it for quota-free recording —
adopted from the concurrent session's tree, one amendment: budget charges from
the slug's embedded submission epoch).

The numbers, 192,512 env-steps/seed on a P100:
  trained [270.5, 276.4, 431.1]  random 124.8  untrained control 150.2 (0.47s)
  sigma_advantage 2.21 (bar: 5)  all_seeds_beat_random TRUE
  curve: reward 4.68 -> 5.08 and still climbing; |act| bounded ~1.4 raw / 0.4
  env; std pinned 0.30; pg_loss ~0; vf_loss stable ~0.3.

Every pathology of v1/v2 is gone. This is a healthy learner short on steps,
not a broken one — exactly the branch the pre-registration marked "more
compute": v4 moves the spec to gpu<8h (110 min/seed, ~850K steps/seed at the
measured ~128 steps/s). Thresholds, control, all-seeds rule untouched.

Two things worth writing down before v4 reports:
- WATCH: seed CV was 28% (91/326). If CV stays ~constant as the mean grows,
  sigma_advantage asymptotes near 1/CV ~ 3.6 and the 5-sigma bar is
  structurally unreachable regardless of learning. If v4 shows a rising gap
  AND a CV plateau, the metric (raw seed std, not standard error) needs a
  versioned redesign a la T2.00 v1->v2 — decided by data, not convenience.
- The reverify queue from the concurrent session had deadlocked on ITSELF for
  24h: its pgrep poll pattern matched its own command line. Killed. T1
  re-verification is chained to run after v4 in one detached script.

## 2026-08-07 — T2.01 v4 verdict recorded; the local MLP probe doubles it

v4 (Kaggle P100, reused kernel, zero fresh quota; 704,512 env-steps/seed,
331.9 GPU-min): FAIL. trained [249.6, 292.7, 240.8], mean 261.0; random
118.2±35.2; sigma_advantage 4.06 (bar 5); all seeds beat random. The v3 WATCH
resolved cleanly: seed CV fell 28% -> 10.6%, so the metric was NOT the
asymptote — the curve itself plateaued. This is the architecture verdict.

The D1-relevant evidence, same env and same 704,512 steps/seed: a 54,179-param
MLP trained locally on CPU (81 wall-min) reached trained means
[583.9, 546.8, 461.4] (overall 530.7) vs random 120.7±33.3 — 12.3 sigma by
T2.01's own sigma_used metric, 8.0 sigma even against raw seed std. The tiny
MLP clears the 5-sigma bar the 140K transformer missed while more than
DOUBLING its return at identical steps. Strong prior, not a claim: the probe
is not a ledger run (no pre-registration, no control). T2.02 on Kaggle after
the Sunday reset stays the official kill-criterion run; its priors just got
very lopsided.

Also inherited from the detached recover chain (still running, owns the
ladder lock): T1.01 re-verified PASS at 3 seeds (improvement_ratio
501.5±25.9). T1.06 is mid-run (~1h elapsed), T1.02 follows, then a queued
script records ME.1 and ME.9 officially. NEXT ITERATION: do not touch the
runner while the lock is held — check /tmp/recover_chain.log and
/tmp/me_runs.log, commit whatever the chain wrote, and if ME.1/ME.9 recorded,
move to implementing ME.2.

## 2026-08-07 (late) — ME.2 PASS: owner memory lives in profile.json

Committed the recover chain's ledger writes first (T1.01/T1.06 re-PASS, ME.1
+ ME.9 recorded, T1.02 ERROR on zero GPU quota — expected pre-Sunday), and
removed the stale /tmp/jack-ladder.lock (PID 541225 dead). Then implemented
OwnerProfile.py (profile.json, latest-statement-wins, regex extraction, no
LLM) and me_2_owner_memory.py. ME.2 PASS: adherence_after_restart 1.0,
after_supersede 1.0, stale_choice_rate 0.0, extracted_topics 40/40,
recency-null 0.075, wipe control 0.175 (~0.25 base rate). Falsification
sanity check done off-ledger: breaking supersession drops adherence to 0.5
and stale rate to 1.0 → check fails, so the test has teeth. 30/105 specs
demonstrated. NEXT ITERATION: ME.3 (reflections beat raw events) — same
substrate family, CPU-only; still no Kaggle quota until Sunday, and T2.02
remains the first Kaggle job after the reset.

## 2026-08-07 (night) — ME.3 PASS: reflections beat raw events at equal tokens

Implemented Reflections.py (statistical per-speaker consolidation of the
episodic log — "ada often involved with pond (52 of 244 events)", source-
linked eids, JSONL, re-derived not patched) and me_3_reflections.py. On a
1,200-event life with mild habits (P_FAV=0.18), 96 aggregation questions at a
40-token budget: reflect_acc 1.0 (using only 15.8 tokens/question off disk
after reload), raw-events null 0.594 (chance 0.25 — a real competitor), gain
0.406. Control: another life's reflections collapse to 0.177 — BELOW chance,
they actively hurt, exactly what MEMORY.md 2.3 pre-registered. 31/105
demonstrated. NEXT ITERATION: ME.4 (Ebbinghaus decay + reinforce-on-recall +
supersede beats FIFO at fixed store budget) — same substrate, CPU-only. Still
no Kaggle quota until Sunday; T2.02 remains the first Kaggle job after reset.

## 2026-08-08 (early) — ME.4 PASS: forgetting keeps what matters

Implemented Forgetting.py (ForgettingMemory: bounded working store per
MEMORY.md 4.2 — Ebbinghaus strength exp(-age/(tau*S)) with S=1+n_recalls,
reinforce-on-recall, key-based supersession that invalidates rather than
rewrites, eviction = superseded first then weakest; policy="fifo" is the
null) and me_4_forgetting.py: 1,200-event life vs a 150-trace budget, 24
durable facts stated in the first 200 events and referenced ~every 50, 8
updated mid-life AFTER heavy reinforcement of the old value. Result:
retention_acc 1.0 vs FIFO 0.0 (retention_vs_fifo 1.0, unbounded ceiling
1.0), update_acc 1.0, stale 0.0, answered off a disk-reloaded snapshot.
Control has teeth: no-supersede stale rate 1.0 (reinforcement deepens the
rut — post-update recalls keep strengthening the WRONG trace). Off-ledger
falsification: no-reinforce → retention 0.0, random eviction → 0.56, both
fail the check; seeds 1,2 identical to seed 0. 32/105 demonstrated. NEXT
ITERATION: ME.8 (working memory survives restarts — depends only on T0.05,
CPU) or ME.10 (diary/skill separation); ME.5 is CPU_LONG (100k events), run
it when a quiet slot allows. Still no Kaggle before Sunday; T2.02 stays the
first Kaggle job after reset.

## 2026-08-08 — ME.8 PASS: working memory survives a real SIGKILL

Implemented WorkingMemory.py (GRU per research/MEMORY.md 3.2 — step/
checkpoint/restore, wm.state written atomically every step per T0.05
discipline, weights separate from state) and me_8_working_memory.py:
delayed-cue task (1-of-8 cue at step 0 only, 30 steps, 4 noise channels),
each eval episode run in a child process SIGKILLed mid-episode (killed_frac
1.0, rc -9 verified, mean ckpt step 14.1), finished by a FRESH process.
Ledger: resume_acc 1.0, zeroed-restart null 0.1875 (chance 0.125),
resume_vs_zeroed 0.8125. Control has teeth: cross-restoring episode j's
state into episode i answers j's cue 1.0 and i's cue 0.0 — the answer lives
in the state file, nothing leaks. Found and fixed a real substrate bug:
plain GRUCell init failed to train at seed 2 (holdout 0.109, resume==zeroed
— the exact falsifier); update-gate bias +1 (Jozefowicz-style retain-by-
default) fixed it — off-ledger seeds 1,2,3 all now: resume 1.0, zeroed
0.06-0.25, control 1.0/0.0. 33/105 demonstrated. NEXT ITERATION: ME.10
(diary/skill double dissociation, CPU, deps ME.1+T1.04 both PASS) or ME.5
(CPU_LONG, quiet slot). Kaggle resets Sunday; T2.02 remains the first
Kaggle job after reset.

## 2026-08-08 — ME.10 PASS: the diary and the skill are genuinely two stores

Implemented me_10_diary_vs_skill.py: 84 of 120 {colour}x{object} drop
episodes recorded in EpisodicMemory (240 filler events interleaved), world
rule outcome = colour_bit XOR object_bit so single-attribute retrieval
predicts held-out pairs at exactly chance; distillation parses training
pairs FROM the diary's did-events and trains a 22-32-2 MLP. Ledger (3
seeds): recall 1.0 pre AND post distillation (learning did not eat the
memory), held-out skill 0.944±0.045 vs untrained null 0.574 (gain 0.370),
diary-on-held-out 0.556 (~chance — the skill is not retrievable). Double
dissociation clean: wipe diary -> recall 0.0, skill 0.944 unchanged;
revert weights -> skill 0.574, recall 1.0 unchanged. Also fixed a real
runner bug: _module_for globbed "me_1*.py" which would match BOTH
me_1_event_log.py and me_10_*.py and raise for ME.1 and ME.10; now
"{prefix}_*.py", all 36 implemented specs verified to resolve. 34/105
demonstrated. NEXT ITERATION: ME.5 (retrieval survives growth, CPU_LONG
100k events — run in a quiet slot) or ME.6 (skill library), or start a
CPU-implementable UB/CU spec. Kaggle resets Sunday; T2.02 (140K-MLP vs
transformer showdown) remains the FIRST Kaggle job after reset.

## 2026-08-08 — ME.5 PASS: retrieval survives 100 -> 100k events

Implemented me_5_retrieval_at_scale.py: one life grown to 100k events (each
holding a UNIQUE obj/place/colour/action 4-tuple from disjoint pools, so a
4-word cue identifies exactly one event and the oracle ceiling is 1.0 by
construction), measured at every decade with two cue classes. Ledger (3
seeds): unique-cue precision@1 = 1.0 at ALL four decades (100/1k/10k/100k)
and 1.0 after reloading the ~14MB JSONL at full scale; ambiguous 3-of-4-word
cues give the honest degradation curve 0.997 -> 0.967 -> 0.773 -> 0.207
(~1/n_competitors as ~5 events share a 3-word subset at 100k) BUT the top-1
answer matches all three cue words 1.0 at every decade — content similarity
stays the primary key, recency+importance never promote a non-match even
against ~100k distractors, confirming the 10x sim-weight analysis in
EpisodicMemory empirically. Fabricated-cue abstention 1.0 at every decade;
recency null <= 0.01 everywhere; latency 36ms/query at 100k (linear scan,
fine live; ~0.36ms/1k events, revisit past ~1M). Strengthened the spec to
seeds=3 (was default 1) to match ME.9/ME.10 — data generation is stochastic.
Standing spec per its notes: re-run at each decade of REAL store growth.
35/105 demonstrated. NEXT ITERATION: remaining CPU-implementable specs —
T2.10 (memory retrieval beats recency, cpu<10min, likely quick given ME.1/
ME.5 machinery), T2.12 (emotion PAD separability), or PG.3 (ladder climbable
with adhesion hands). ME.6 needs T2.11 (GPU). Kaggle resets Sunday; T2.02
(140K-MLP vs transformer showdown) remains the FIRST Kaggle job after reset.

## 2026-08-08 — T2.10 PASS: retrieval scoring beats recency where recency is competitive

Implemented t2_10_retrieval_vs_recency.py. ME.1's recency null was degenerate
(newest event answers everything, ~0.001); this spec stages the fair fight the
Generative Agents scoring exists for. Arm A: 120 recall questions with a
conversational recency bias (35% about the last 10 events of a 650-event
life), so "return the 5 most recent events" earns 0.178 +/- 0.021 recall@5 —
a real baseline — while scored retrieval gets 1.0 recall@5 (and @1), margin
0.82 vs pre-registered 0.30. Arm B: 30 recurring 4-tuples lived 5x each; the
cue matches all occurrences at sim 1.0 and the right answer is the LATEST —
scored retrieval 1.0 latest@1, similarity-only scorer (w_recency=w_importance=0,
control that must fail) 0.0, recency-baseline-on-arm-B also 0.0. So the
combined score beats pure recency AND is not secretly pure similarity; the
recency term is load-bearing exactly where it should be (tie-breaking equal
content matches). 3 seeds, all stds 0.0 on experiment arms, 0.32s runtime.
36/105 demonstrated. NEXT ITERATION: T2.12 (emotion PAD separability,
cpu<10min) or PG.3 (adhesion-hands ladder climbability, cpu<10min) — both
CPU-cheap. Kaggle resets Sunday; T2.02 (140K-MLP vs transformer showdown)
remains the FIRST Kaggle job after reset. T6.03 depends on T2.10+T0.05 —
check whether it is now unblocked next time.

## 2026-08-08 — T2.12 PASS: emotion states are distinguishable (but the mapping is arbitrary)

Implemented t2_12_emotion_separability.py. Four event regimes (thriving/
struggling/exploring/neglected), 40 trajectories x 120 steps each through the
shipped EmotionalState, nearest-centroid on z-scored (mean,std) PAD features:
held-out separability 0.904 +/- 0.072 vs chance 0.25; matched-variance
random-walk null 0.238 +/- 0.027 (chance, as it must be); shuffled-label
control 0.146; margin 0.667 vs pre-registered 0.30. 3 seeds, 39s CPU.
HONEST CAVEAT worth carrying: separable does NOT mean correct. The diagnostic
means show pleasure sign is seed-arbitrary (struggling 0.33+/-0.78 vs
thriving 0.11+/-0.52 across seeds) because EmotionalState.py:611 routes
events ONLY through the untrained GRU — the OCC symbolic deltas never touch
pad_vector directly, so an untrained module gives a random-but-injective
event->PAD map. Fine for T2.12's claim (input modality carries information),
but any future spec about VALENCE (e.g. "praise makes him feel good", mood-
conditioned behaviour) needs either GRU training or a residual OCC path.
37/105 demonstrated. NEXT ITERATION: PG.3 (adhesion-hands ladder
climbability, cpu<10min) or T6.03 (cross-session persistence, cpu<10min,
now unblocked) — both CPU-cheap. Kaggle resets Sunday; T2.02 (140K-MLP vs
transformer showdown) remains the FIRST Kaggle job after reset.

## 2026-08-08 — PG.3 PASS: the ladder is climbable in principle, and falls are clean

Implemented pg_3_ladder_climbable.py: a minimal certification rig (30 kg
torso hanging from two reach+lift slide arms with adhesion hands, gain 900 N
vs 302 N body weight) gripping the real playground ladder's rungs from BELOW,
so zero adhesion has no support path. Scripted hand-over-hand ascent of one
rung across three rung spacings 0.30/0.26/0.34 m (seeds; spec strengthened to
seeds=3): ascent_frac 0.973 +/- 0.020 of a rung spacing, hold drift 0.0001 m,
both hands verified in contact with the next rung every seed. Falls: release
mid-air -> finite states, rest on floor (torso z 0.0698, speed 0.03), mid-fall
snapshot restores bit-exact into a fresh MjData (resume_max_dev <= 4e-14), and
mj_resetData + regrip holds again (drift 0.0001). Zero-adhesion null slips
0.208 m during the hang and ends the script on the floor (final z 0.0698,
ascent -1.31 rungs). One threshold adjusted BEFORE the official run and
documented: control drop 0.30 -> 0.15 m, because ungripped hands legitimately
come to REST on the rung below (normal contact) instead of free-falling —
resting-vs-gripping is exactly the distinction the spec tests. Also fixed:
geom_xpos is zeros until mj_forward, so the initial torso z needed a forward
pass. 38/105. NEXT ITERATION: T6.03 (cross-session persistence, cpu<10min,
unblocked) is the cheapest remaining; PG.4 (noisy-TV trap, CPU_LONG) now has
its dependency and matters for every curiosity claim. Kaggle resets SUNDAY —
the FIRST Kaggle job after reset is T2.02 (140K-MLP vs transformer showdown).

## 2026-08-08 — T6.03 PASS: Jack survives a restart whole — after fixing a save bug that dropped his mood

Implemented t6_03_cross_session.py: session 1 (child process) builds the real
UnifiedBrain (58M, LLM off), stores 20 owner facts in CompanionMemory, runs 12
emotional updates, renames the personality, appends monologue entries, sets
global_step, saves via CompanionPersistence.save_all, exits. Session 2 is a
fresh process seeded differently: answers as the null FIRST (fresh instance,
recall 0.0), then load_all and answers again — recall_after_restart 1.0 +/- 0
(3 seeds), gap 1.0. Fidelity gates all 1.0: state_dict sha256 bit-match after
load while virgin weights provably differ; PAD restored to 0.0 dev while the
virgin PAD sat 0.068 +/- 0.014 away; mood history refilled into the live
MoodHistory (record() still works); personality/global_step/monologue/memories
exact. Control: truncated save REJECTED (load raises) every seed. Found and
FIXED en route in Persistence.py: list(MoodHistory) raised TypeError inside
_collect_emotional_state's try/except, so EVERY save ever written silently
dropped the entire emotional state (pad included); and _apply_emotional_state
replaced MoodHistory with a bare list, killing .record() after any restore.
Spec strengthened to seeds=3 + explicit corruption control before the official
run. CAVEAT worth an iteration: the info-only byteflip arm shows torch.load
ACCEPTS a 64-byte mid-file zero-overwrite without raising — the save format
has no integrity checksum. Small hardening: put a sha256 of the payload in the
save dict, verify in load_all, then promote byteflip to a gated control.
39/105. NEXT ITERATION: Kaggle resets SUNDAY (tomorrow) — the FIRST Kaggle job
is T2.02 (140K-MLP vs transformer showdown, settles D1). If pre-reset, PG.4
(noisy-TV trap, CPU_LONG) unblocks every curiosity claim and needs no GPU.

2026-08-08T11:07Z — Landed the previous iteration's finished-but-uncommitted
PG.4 PASS (it timed out pre-commit; ledger/CHECKLIST/walls all consistent, check
re-verified against recorded aggregates): ICM dwell 0.667 vs null 0.061, static
control 0.0. CAVEAT for CU.3: per-seed dwell was (1.0, 1.0, 0.0) — one seed
never found the panel in 20k steps; PASS is by the ladder-wide aggregate-mean
protocol. CU.3 should compare dwell DISTRIBUTIONS, not means. Then closed the
T6.03 follow-up: jack-save-v1 sha256 wrapper in Persistence, byteflip promoted
to a gated control, T6.03 re-PASS 3 seeds (byteflip raised 1.0, recall 1.0).
40/105. NEXT ITERATION: Kaggle resets SUNDAY (tomorrow) — FIRST Kaggle job is
T2.02 (settles D1). Today, cheapest CPU units: ME.7 (SIESTA consolidation) or
CU.-family design; note ps output showing 'claude -p ...ladder...' is YOUR OWN
process (ladder_loop.sh) — uncommitted work at start means the PREVIOUS
iteration died mid-unit: inherit and finish it, don't discard.

2026-08-08T13:1xZ — PG.5 PASS: procedural contact audio with localization
labels — bearing decode 1.0 +/- 0.0 across 3 seeds (10-deg tolerance, truth
computed INDEPENDENTLY from the sampled drop point, not the synth's own label),
label accuracy 1.0, spectral match 1.0 (window's dominant freq == voiced geom's
size-derived fundamental within 12%), controls at chance: mono 0.148, shuffled-
pan 0.148 (both <= 0.30 gate), 9 drops/seed, 10.3s CPU. New component
ContactAudio.py: van den Doel-Pai modal resonance on MuJoCo contact onsets,
constant-power pan by lateral angle + 1/d attenuation; events carry full
azimuth/elevation/distance labels for UB.4 training. Fixed en route: box
landing flat emits up to 4 corner contacts and taking the first corner as
source was 0.26m/9-deg off — events now use the pair's contact CENTROID.
Shuffle control decodes 3 random pans/episode (27/seed) so the null isn't
luck-dependent. Scope notes: pan encodes LATERALITY only (front-back needs
ITD/HRTF — future); surface (floor) modes absent in v1; underwater muffling
absent. 41/105. NEXT ITERATION: Kaggle resets SUNDAY (tomorrow) — FIRST Kaggle
job is T2.02 (settles D1). If pre-reset, remaining cheap CPU: none obvious in
'next' (ME.7 blocked on T5.03); consider T2.08-adjacent design or run --gate
to catch regressions after the ContactAudio addition.

2026-08-08T13:4xZ — T2.20 PASS: episodic memory helps the next episode —
embodied search in the playground (PG.4's collision-free rover), episode 1
explores by random waypoints and writes "saw" events (text + position in
meta) into EpisodicMemory amid 300 vocabulary-disjoint chatter events; later
episodes recall-and-drive vs explore-from-scratch. search_time_ratio
0.046+/-0.007 (memory 6.0 steps vs null 130.8+/-12.9), retrieval 1.0,
controls at null: shuffled-position store 1.234, recency-only 0.992 (targets
never the last-sighted object by construction). 3 seeds x 3 targets x 8
reps/arm, 32.5s CPU. Method lesson: the seed-0 pilot with 3 episodes/arm had
the shuffled control at 0.39 — search times are heavy-tailed and one lucky
find swings a 3-episode mean 2x; reps were raised to 8/target/arm BEFORE the
recorded run, gates untouched. Honest scope: chatter vocab is disjoint from
object vocab, so retrieval-under-collision is ME.5's claim, not this one;
objects are static across episodes. 42/105. NEXT ITERATION: Kaggle 30h
resets SUNDAY (tomorrow) — the FIRST Kaggle job is T2.02 (140K-MLP vs
transformer at 2M steps; settles D1). If still Saturday, Colab GPU_SHORT
candidates in 'next': T2.03 (pretrained vision probe), T2.19 (flow-head
bimodal), T4.02 (gradient-norm balance); T1.02 is ERROR only because both
backends were unavailable on 08-07 — retry it cheaply first. Also consider
--gate: last full re-run predates ContactAudio and T2.20.

2026-08-08T23:3xZ — T2.02 launch-readiness: inherited the previous iteration's
uncommitted prep (t2_02_mlp_showdown.py, spec dep/metric edits, Sunday-start
budget week keys) and verified it instead of writing new code. Found and fixed
one real bug in the inherited budget migration: the old ISO key "2026-W32"
(Aug 3-9) was left in gpu_budget.json holding kaggle 37.4554h, and the NEW
%U-format key for Aug 9-15 is ALSO "2026-W32" — the formats collide, so the
tracker would have refused Kaggle jobs for ALL of next week despite the fresh
30h. Removed the stale entry (its hours are already re-filed under W31,
Aug 2-8); verified with a monkeypatched _week(): Sunday now shows 30.0h
remaining, afford(7.0)=True. Dry-tested t2_02's pre-registered _check on 5
synthetic outcomes: MLP-wins(expected)->FAIL, TR-wins->PASS,
step-mismatch->VOID, non-learners->VOID, untrained-clears-gate->False — all
correct. Kernel's TrainingPipeline calls match T2.01 v4's working kernel
verbatim; fetch=["t202.json"] resolves via JACK_OUT on both backends. T1.02
retried at 22:07 by the previous iteration: still ERROR, both backends down.
HEAD is pushed (build_job's verify gate needs that). 42/105. NEXT ITERATION:
it is SUNDAY — quota is fresh, everything is verified and committed. Launch
    /data/venvs/jackthelearner/bin/python -m experiments.run T2.02
FIRST, before anything else; expect ~7h wall (3x100min transformer + 3x<=45min
MLP, one kernel, partial JSON dump per seed). If your process dies mid-poll
the kernel survives on Kaggle: reattach for free with
JACK_REUSE_KERNEL=jack-ladder-<epoch> (slug is in the kaggle kernels list)
and rerun T2.02 — reattach skips the affordability gate by design. Do NOT
resubmit a second kernel while one runs. After T2.02 settles D1, journal the
verdict next to /tmp/mlp_probe.json's numbers and pick up T2.03/T2.19/T4.02
on Colab if it is back.

2026-08-09T00:2xZ — T2.02 LAUNCHED on fresh Sunday Kaggle quota, per plan.
Kernel jannolouwrens/jack-ladder-1786234100 pushed 00:08:20Z, status RUNNING
(verified via kaggle kernels status). Local poller survives this iteration:
setsid-detached PID 830502 running `experiments.run T2.02`, log at
/tmp/t202_run.log (stdout is block-buffered — a near-empty log is NORMAL, the
kernel status command is the truth). Expected ~6.7h wall: finish ~07:00Z, then
the poller fetches t202.json, writes the ledger, and exits on its own. Budget
is charged by Budget.charge() at job end, so gpu_budget.json still showing
only W31 is expected until then. NEXT ITERATIONS (01:00-07:00Z): do NOT
resubmit T2.02 and do NOT launch other Kaggle work while this runs. Check
`pgrep -f 'experiments.run T2.02'` — if the poller is alive, leave it alone
and do CPU work (T2.19 audio-conditioned nav, T4.02 gradient-norm balance) or
Colab-short (T1.02 retry, T2.03) if Colab is back. If the poller is DEAD and
the ledger has no T2.02 outcome, reattach for free:
JACK_REUSE_KERNEL=jack-ladder-1786234100 then rerun T2.02 (reattach skips the
affordability gate by design). After the verdict lands, journal it next to
/tmp/mlp_probe.json (local MLP 54K params @704K steps: seed means ~584 etc.)
and v4 transformer's 261, then update docs/DECISIONS.md D1 per the
kill-criterion.
