# INTEGRATION_QUEUE — research results become tests, traceably

> The builder processes this queue top-down, one entry per iteration alongside
> its other work. This file exists because integration used to depend on the
> orchestrator being awake: on 2026-08-09 a research doc (NEEDS_AND_DEATH)
> DISPROVED a spec from another doc (PS.00c/PS.02) twelve minutes before the
> builder was due to register the disproven version. The cross-check below is
> that near-miss, made mandatory.

## THE PROTOCOL — every entry, no exceptions, in order

1. CROSS-CHECK: grep the spec's subject terms across every OTHER research doc
   in docs/research/ and docs/LESSONS.md. A refutation or conflict found →
   do NOT register; correct per the refuting analysis or escalate to
   DECISIONS_NEEDED.md. (This step is the PS lesson.)
2. VERIFY: AST-parse the Spec(...), check id collisions and prefix-shadowing
   against the LIVE registry, confirm every depends_on resolves.
3. REGISTER exactly as written — no threshold edits during integration.
4. IMPLEMENT + RUN the cheapest registered spec of the entry (CPU first).
5. MARK the entry: status, commit hash, date. Never delete entries — this
   file is the provenance chain from research to ledger.

## Queue (top = next)

| research doc | specs | status |
|---|---|---|
| LEARNING_CORE.md | LC.00–LC.06 (+PS.01 pairing — register together) | PENDING |
| NEEDS_AND_DEATH.md | NE.00–NE.09 | PENDING — note: doc §9 gates NE.01's constants on §1.2 citation verification (Borbély ratio is load-bearing and open); register all, but do not let NE.01 fix constants until a citation pass closes §1.2 |
| PURPOSE_AND_SCAFFOLDING.md | PS.* | BLOCKED-ON-CORRECTION: PS.00(c)+PS.02 disproven by NEEDS_AND_DEATH (drive-farming cannot exist; exact VI + K&G eLife 2014 theorem). Correct, then register. |
| CURIOSITY_BAKEOFF.md | LT.01–LT.09 | PENDING |
| D1_CONTROL_ARCHITECTURE.md | D1.0, T2.21 | PENDING |
| HEARING_BAKEOFF.md | HR.1–HR.8 | PENDING |
| LANGUAGE_GROUNDING.md | LG.* | PENDING — doc was truncated (agent killed); verify completeness before extracting |
| DIRECTION_AUDIT.md | WP.01–04, LF.01–05, SO.01–05, PS.07, T0.17–18 (stubs) | PENDING — stubs need full Spec fields before registration |
| SURVIVAL_WORLD.md | (agent still writing) | AWAITING DOC |
| UNIFIED_BRAIN_BAKEOFF.md | PG.6–7, UB.9–16 | REGISTERED a3129b2 2026-08-09 |
| MEMORY_RETRIEVAL_BAKEOFF.md | ME.11.0, ME.11.A–F | REGISTERED 0c1ff06 (ME.11.0 PASSING) |
