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
