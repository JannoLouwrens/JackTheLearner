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
6. IF the entry completed a BAKEOFF: update docs/CHAMPIONS.md — the seat, the
   new holder, held-by VERDICT, the commit — and re-run the standing
   integration gates under the new champion before calling adoption done.

## WHEN THE QUEUE IS EMPTY — the loop's research step is YOURS

An empty queue is not "done"; it means the frontier has no design yet. The
correct iteration is then: find the next stage on GOAL.md's path (via
DIRECTION_AUDIT's sequencing) whose question has no docs/research/ document,
and RESEARCH it — dispatch research agents or write the survey yourself, with
citations, arms, costs, and Spec(...) drafts in the house format. The output
is a new research doc AND a new entry in this queue. The loop generates its
own work; it never idles because nobody fed it.

## GAP-FILL designed by the owner's question (2026-08-09) — register with the LC family

**LC.07 — the capacity sweep, and it is a STANDING spec.** LC.06 enforces a
ceiling; nothing finds the optimum, and nothing re-opens the question as
experience grows. Draft:

    Spec("LC.07", 2, "Capacity is swept, not assumed — and re-swept as he lives",
         hypothesis="At a FIXED experience budget, life_gain over trainable "
                    "parameters is an INVERTED U: too small underfits, too "
                    "large starves on limited experience. The adopted size is "
                    "the SMALLEST within 1 sigma of the peak.",
         falsified_by="Monotonic in size (bigger always better) — then the "
                      "experience budget, not capacity, is the binding "
                      "constraint and the simplicity budget is arbitrary. Or "
                      "flat — capacity does not matter here and the smallest "
                      "ships by default.",
         null_baseline="The current champion's size.",
         metric="smallest_within_1sigma_of_peak", budget=Budget.CPU_LONG,
         depends_on=["LC.04"], seeds=3,
         control="Shuffled-experience arm at every size: the inverted U must "
                 "FLATTEN — if big still beats small on scrambled data, the "
                 "sweep is measuring capacity to memorise, not to live.",
         kills="Any size claim made without a sweep, including our own "
               "5M ceiling.",
         notes="STANDING: re-run at each decade of accumulated lifetime "
               "experience. The optimum MOVES — scaling laws say capacity "
               "should track data, and Jack's data grows every life. A size "
               "decided at 10 lives is wrong at 1,000. Measured precedents "
               "that motivate it: 54K beat 57M here; a 4M PPO beat a 201M "
               "DreamerV3 on Crafter.")

## GAP-FILL — THE ANTI-PUPPET TEST (owner, 2026-08-09). Register with the LG family.

The owner's question: "will Jack be smarter than the local LLM on him? He MUST
develop knowledge and connect it to words instead of just the LLM communicating
and PRETENDING to be Jack." Nothing in 136 specs tested this. It is the
project's existential claim and it had no falsifier.

    Spec("LG.00", 4, "Jack knows what his LLM cannot — he is not a puppet",
         hypothesis="On questions about HIS world, HIS body and HIS history, "
                    "full Jack (learned core + diary + LLM) beats LLM-ALONE "
                    "given the identical prompt context, by >=3 sigma. The "
                    "knowledge is in the parts that LIVED, not in the frozen "
                    "weights that never did.",
         falsified_by="LLM-alone matches full Jack on world questions. Then "
                      "Jack is a costume on a language model, the learned core "
                      "and diary are decorative, and the project has not built "
                      "a creature.",
         null_baseline="LLM-alone, same prompt, no diary, no learned core.",
         metric="grounded_knowledge_advantage", budget=Budget.CPU,
         depends_on=["ME.9", "LG.01"], seeds=3,
         control="GENERAL-KNOWLEDGE questions (history, arithmetic, "
                 "vocabulary) — here LLM-alone must MATCH OR BEAT full Jack. "
                 "If Jack wins everywhere, the test is measuring scaffolding "
                 "or prompt advantage, not grounding. The two results together "
                 "are the claim: he is smarter INSIDE his life and dumber "
                 "outside it, which is exactly what a creature should be.",
         kills="The frozen-LLM architecture as implemented. If the mouth is "
               "doing the knowing, the mind was never built.",
         notes="Double dissociation, the ME.10 pattern applied to selfhood: "
               "ablate the diary -> his history answers collapse, general "
               "knowledge survives; ablate the LLM -> he still ACTS correctly "
               "in his world while losing only the ability to say so. "
               "Knowledge in the parts that lived; language as the mouth.")

## GAP-FILL — WHO CHOOSES THE WORDS (owner, 2026-08-09). Register with LG.00.

Owner: "but the LLM will always be the one talking — how will Jack choose what
to say?" Speaking is an ACTION: his core selects it the way it selects any
other, and the LLM is the motor system for language, exactly as actuators are
the motor system for movement. Muscles do not choose the destination.

    Spec("LG.10", 4, "Jack chooses what to say; the LLM only chooses how",
         hypothesis="Utterance MEANING tracks Jack's internal state and diary, "
                    "not the language model. Three independent measurements: "
                    "(a) same state, different LLM sampling seeds -> same "
                    "meaning, different wording; (b) different state, same LLM "
                    "-> different meaning; (c) SWAP THE LLM for a different "
                    "frozen model -> meaning preserved, style changes.",
         falsified_by="Meaning varies with the sampler, or survives a state "
                      "change, or changes when the LLM is swapped. Any of the "
                      "three means the language model is choosing the content "
                      "and Jack is being ventriloquised.",
         null_baseline="LLM free-generation from the same prompt with no "
                       "core-selected intent — its meaning must NOT track his "
                       "state.",
         metric="meaning_tracks_state_not_model", budget=Budget.CPU,
         depends_on=["LG.00"], seeds=3,
         control="SILENCE. Drive his core to a state with nothing to report "
                 "and he must say NOTHING. A mouth that always speaks is a "
                 "generator running free; choosing not to speak is the "
                 "cheapest proof that something is choosing at all.",
         kills="Any speech path where the LLM receives free rein over content. "
               "If the model swap changes what he means, the mind was in the "
               "mouth.",
         notes="Practical form: core emits a structured intent (report/ask/"
               "describe + referent + source) OR selects among LLM-proposed "
               "phrasings; a verification gate rejects any utterance asserting "
               "something not present in his state or diary — the extractive "
               "rule extended from memory to speech. The LLM-swap arm doubles "
               "as a live test of the swappable-LLM decree.")

## FIRST: FINISH THE LC BAKEOFF (LC.02-LC.06) — unblocked, zero GPU

LC.00 PASS (framing survived its cheapest falsifier), LC.01 PASS (unison
admission gate). LC.03/04/05 are the actual PPO-vs-world-model arbitration and
are NOT IMPLEMENTED. This is the highest-leverage unblocked work in the
project and needs no GPU, so it proceeds beside T2.01. Everything below waits
on it, INCLUDING the T5.01 entry — which is blocked on T2.01 anyway and must
not stall the queue behind it.

## SCHEDULED BY THE OWNER — T5.01, the founding thesis test (2026-08-09)

"Schedule the run after T2.01." The physics-pretraining premise was retired by
argument; law 3 says bakeoff. It now runs.

STATE: T5.01 is **NOT IMPLEMENTED** — scheduling it is therefore two jobs:
  1. IMPLEMENT experiments/tests/t5_01_*.py (Phase-0 SymPy physics pretraining
     vs identical architecture without it, downstream control sample-efficiency,
     5 seeds as declared — do NOT reduce seeds to fit budget; law 4).
  2. RUN once T2.01 (its dependency) is PASSING.
PRECONDITION RISK, state it in the ledger if it bites: if T2.01 does not PASS,
T5.01 stays BLOCKED and this schedule cannot execute — say so plainly rather
than silently skipping.
BUDGET: ~17 Kaggle hours remain this week after T2.01's ~6.5. T5.01 is 5 seeds
at gpu<8h. If the implementation's honest estimate exceeds the remaining
budget, SPLIT ACROSS WEEKS (Kaggle resets Sunday) rather than shrinking the
experiment.

## THE GENERALITY MAP — docs/GENERALITY.md (owner, 2026-08-09)

Twelve named barriers between Jack and general intelligence, each with a
falsifiable test. NOT SCHEDULED and NOT competing with the frontier — it is
the map, so the ladder stops being silent about the distance. Two entries are
cheap enough to enter the real queue when their prerequisites land:
  - GEN.02 (two Jacks, learning by watching) — costs a second PROCESS, and it
    is the largest known driver of intelligence that we currently have zero
    specs for.
  - GEN.01 (does capability track world richness?) — measurable the moment the
    W-tier fidelity ladder exists; tells us whether the ceiling is the brain
    or the world, which is the most informative single number available.
The Sunday Review's anatomy audit should check GENERALITY.md alongside
CHAMPIONS.md: a barrier with no seat and no spec is fine, but it must stay
NAMED.

## Queue (top = next)

| research doc | specs | status |
|---|---|---|
| LEARNING_CORE.md | LC.00–LC.06 + PS.01 | REGISTERED by the builder autonomously, 2026-08-09 (registry 128→136; cross-check clean — NEEDS_AND_DEATH §0.2 *supports* LC.00's drive-reduction reward; W0 naming reconciled by LEARNING_CORE §5.0's contract). **LC.00 PASS** same day: 3 of 4 tabular cores beat the null ≥3σ (q_drive 9.1σ, model_vi 6.3σ, model_efe 4.1σ; q_lp 1.5σ ran but did not clear), frozen control −6.3±5.9 ≈ 0 — the world does not drift. **LC.01 PASS** 2026-08-09 (5.7 s, 3 seeds): all five §5.4 arms admitted on U1–U4; the three controls failed on their pre-registered side (unbound cross-modal finite difference exactly 0.0 on every seed, leaky private-path grad 271±44, naive-scales needs share 0.113 < 1/6). The arms now live in `experiments/cores.py` — LC.02/LC.03 import them, they are not re-implemented. Next cheapest: LC.02 (throughput floor, CPU) — but it needs the W0 env, so PS.01 (drive layer, CPU, no body) may be cheaper first. |
| NEEDS_AND_DEATH.md | NE.00–NE.09 | PENDING — note: doc §9 gates NE.01's constants on §1.2 citation verification (Borbély ratio is load-bearing and open); register all, but do not let NE.01 fix constants until a citation pass closes §1.2. NE.08 overlaps W.6 — see SURVIVAL_WORLD §5.0 reconciliation before registering |
| PURPOSE_AND_SCAFFOLDING.md | PS.* | BLOCKED-ON-CORRECTION: PS.00(c)+PS.02 disproven by NEEDS_AND_DEATH (drive-farming cannot exist; exact VI + K&G eLife 2014 theorem). Correct, then register. NOTE: **PS.01 is already registered** (2026-08-09, with the LC family — LC.03 depends on it and LEARNING_CORE §5.6 required one commit; PS.01 is calibration, not implicated in the refutation). Do not register it twice. |
| CURIOSITY_BAKEOFF.md | LT.01–LT.09 | PENDING |
| D1_CONTROL_ARCHITECTURE.md | D1.0, T2.21 | PENDING |
| HEARING_BAKEOFF.md | HR.1–HR.8 | PENDING |
| LANGUAGE_GROUNDING.md | LG.* | PENDING — doc was truncated (agent killed); verify completeness before extracting. MUST INCLUDE THE LIAR TEST (owner-designed 2026-08-09): two advisors, one systematically truthful, one systematically false, advice verifiable by his own experience. PASS = advice-following diverges by advisor track record (trusts the truthful, discounts the liar), with attribution intact. CONTROL = swap the advisors' roles mid-life; the trust must MIGRATE, or it was measuring voices, not veracity. NULL = an agent with attribution stripped must treat both advisors identically. This is the emergence stone's first falsifiable claim: trust earned, checked, and unscripted |
| DIRECTION_AUDIT.md | WP.01–04, LF.01–05, SO.01–05, PS.07, T0.17–18 (stubs) | PENDING — stubs need full Spec fields before registration |
| SURVIVAL_WORLD.md | W.1–W.7 | PENDING — CROSS-CHECK REQUIRED: W.6 overlaps NE.08 (reconciliation written in SURVIVAL_WORLD §5.0); register the reconciled pair or the ledger gets two specs testing one claim. Also carries recommendation 6b: contype/conaffinity audit of the playground, worth ~3x throughput, do before any W.* runs |
| OWNER DECISION: the owner's hands (no doc yet — empty-queue rule applies) | SO-family: provisioning channel (drop-in objects), provenance into the diary ("who left this"), anti-puppeteering limits | PENDING — approved by owner 2026-08-09; needs its research pass then specs. OWNER 2026-08-09: visualisation of needs (hunger bars, spectating) is deliberately LATE — it is a pure read layer, zero science cost. But the camera rolls from day one: NE.* implementation must log needs telemetry (a few floats/step) from the FIRST needful life, so any later viewer can replay any life. Recording is nearly free; retrofitting history is impossible |
| UNIFIED_BRAIN_BAKEOFF.md | PG.6–7, UB.9–16 | REGISTERED a3129b2 2026-08-09 |
| MEMORY_RETRIEVAL_BAKEOFF.md | ME.11.0, ME.11.A–F | REGISTERED 0c1ff06 (ME.11.0 PASSING) |
