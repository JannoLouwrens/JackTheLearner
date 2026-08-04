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
