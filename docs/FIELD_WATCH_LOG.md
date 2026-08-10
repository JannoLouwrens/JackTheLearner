# FIELD_WATCH_LOG.md — one line per sweep, so drift is visible

Append-only. The current state lives in `docs/FIELD_WATCH.md`; this file exists
so that a reader can see what the scout believed on each date without diffing
a rewritten file. One line per item, newest sweep at the bottom.

Format: `YYYY-MM-DD · TAG · headline`
Tags: `NOM` nomination · `WATCH` watchlist · `NONE` no-action on a front ·
`DISC` discipline/methodology finding · `META` about the sweep itself.

---

2026-08-10 · META · First sweep. Window 2026-02→08. Fronts 1 and 2 delivered; fronts 3, 4, 5 did not. Smell/taste/voice not searched — a real gap, queued first for next week.
2026-08-10 · NOM · Certificate-gated identifiability protocol (arXiv:2607.27017, 2026-07-30) → UB.11 pre-gate. Stiffness enters the latent only when touch is a prediction TARGET (R² 0.50) not an INPUT (−0.02); certificate 0.87. ~5M params, one RTX 4060 laptop, <1h. Supplies UB.11 the positive control it lacks.
2026-08-10 · NOM · Anti-collapse regularisers as LEARNING_CORE §5.4 arms A4b/A4c: inverse-dynamics (SMWM, arXiv:2606.20104) and SIGReg/LeJEPA (arXiv:2511.08544 + theory arXiv:2605.26379). Both delete A4's EMA target encoder. SMWM 84% vs SIGReg 59% on OGBench-Cube. Risk: both argue from pixels; SMWM has no multimodal input at all.
2026-08-10 · NOM · Interoceptive precision allocation (arXiv:2608.04232, 2026-08-04, code released) → NEEDS_AND_DEATH §2.4b arm. 2.08× survival over the uniform-precision baseline that IS our current design; anti-aligned control correctly fails below null. Cheapest nomination — runs on 4 ARM cores. Also new evidence bearing on A3's low prior.
2026-08-10 · NOM · Entity-collision protocol (arXiv:2605.29630) → MEMORY_RETRIEVAL_BAKEOFF §2 eval-set design. Floors BM25 by construction; opposite discipline to our lexical-disjointness invariant. Would catch a fixture-dependent verdict before adoption. Weakest provenance of the four (abstract only).
2026-08-10 · WATCH · Simulus (arXiv:2502.11537v4) — mixed-modality tokenisation + prioritised world-model replay, SOTA Atari-100K, beats TWM on Craftax-1M. Blocked on a parameter count and a per-step wall-clock; authors concede token-based WMs train slowly, which is exactly what B4's 5.0 sim-s/real-s floor kills.
2026-08-10 · WATCH · Survival RL (arXiv:2605.31273) — DISAMBIGUATION: "survival" = dwell time at goals, not homeostatic needs. 2–8× over CRL on long-horizon locomotion. Recorded so a future sweep does not chase the title.
2026-08-10 · WATCH · Reward-prediction-biased hippocampal replay (Nat. Comms s41467-025-65354-2) — biology-oracle argument for prioritised replay in NE.05/SIESTA, converging with Simulus from the ML side. UNVERIFIED: fetch failed. Do not cite until read.
2026-08-10 · NONE · MEMORY — the 6-month agent-memory literature (mem0/MAGMA/RecMem/A-MEM/LoCoMo) is generative recall and is constitutionally inadmissible as an arm, however good its numbers. No new CPU encoder beat the incumbents. Only a protocol paper survived the filter.
2026-08-10 · NONE · CURIOSITY & OPEN-ENDEDNESS — nothing would add an arm to disagree/lp/metra/vlm-lp. The noisy-TV literature continues to corroborate the position §1.2 already holds; corroboration is not news.
2026-08-10 · NONE · WORLDS & EMBODIMENT — fidelity ladder W0→W3 unchanged. MJX/Newton are GPU-parallel-env plays on the axis SURVIVAL_WORLD §2.2 already ruled out; Newton 1.0 GA (claimed 475× MJX) recorded only as a re-open trigger should the ladder ever become GPU-bound.
2026-08-10 · DISC · AI-agent-authored preprints now rank in search results. A clawrxiv.io "Curiosity Information Gain" paper matched our open questions almost exactly and claimed +34% states over RND — its own table says ~25%. Caught on the fifth check; the cheap one that catches it alone is "read the table and confirm it says what the abstract says". Nominated to the builder as a LESSONS.md candidate, not written.
