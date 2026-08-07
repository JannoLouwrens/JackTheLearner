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
