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
