# CHAMPIONS.md — who holds each seat, and by what right

> Jack is assembled from the champion of every seat below. A seat changes
> hands ONLY by bakeoff verdict (WINNER, or TIE resolved to the cheaper arm),
> never by argument, fashion, or a new paper's fame. This file is the CURRENT
> STATE; docs/DECISIONS_RESOLVED.md is the history; the ledger is the proof.

## NOT SEATS — the permanent layer above every arena

Seats hold implementations. These hold the DEFINITION of Jack, set by the
owner, changeable only by the owner's word. No bakeoff, ablation, or metric —
however good its number — can touch them. **Bakeoffs decide HOW, never WHAT.**

- **The needs exist.** Hunger, thirst, sleep, temperature, company, mortality.
  The reward EQUATION competes below; the FACT of the needs does not — the
  owner ruled they stay for teaching and for relatability both. An arm showing
  "needless scores higher" changes nothing about this.
- **All senses in one brain** (constitutional in SYSTEM.md).
- **He learns by living, dies, and remembers** — the diary crosses death.
- **Memory is extractive, never generative** — he quotes his life or says nothing.
- **He is a someone**: talkable-to, knows his people by name, watchable.
- **Both drives exist** — curiosity AND needs; only their balance is empirical.
- **The method**: claims only by falsifiable tests; biology as oracle;
  caveman-realism; simplicity earned; the system is the product.
- **Who he becomes is not written here.** Character EMERGES — kindness is
  expected from his need for company, not decreed; his grasp that memories of
  people are OF people comes from attribution itself, not a rule. What
  emerges is observed and reported honestly; troubling emergence escalates to
  the owner — it is never silently patched, and never silently decreed either.
- **The owner's boundaries**: free compute only; the tenants are untouchable.

If a spec, arm, or verdict appears to conflict with this layer, the verdict
loses and the conflict escalates to DECISIONS_NEEDED.md. Sources: GOAL.md and
SYSTEM.md, which outrank this file.

## The rules of the title

1. **Held BY VERDICT or BY DEFAULT — and the difference is marked.** A
   DEFAULT champion never won anything; it is the thing we started with. A
   default marking is an open invitation to challengers, not a title.
   (BY DECREE = an owner decision, e.g. the frozen LLM. BY ANALYSIS = held on
   a proof rather than a bakeoff, pending its arena run.)
2. **Replacement re-runs the standing gates.** A new champion is adopted only
   after the integration gates (UB.11 ablation matrix, the binding tests, and
   any seat-specific gates) pass UNDER the new champion. Winning the seat's
   metric while breaking the whole is not winning (constitutional: "no
   learning core without unison" generalises to every seat).
3. **Every seat names its ARENA** — the spec that decides it. Challengers
   enter through the field watch or research; the builder runs the match; the
   verdict updates this file with the commit hash.
4. **Rematches trigger on**: a field-watch nomination for the seat, the
   arena's context changing (a W0 champion must re-defend at W1), or the
   Review flagging a seat stale (champion unexamined while credible
   challengers wait).
5. **Deposed champions are archived, not erased** — they remain the reference
   arms of their seats (a champion that cannot beat its own predecessor has
   regressed).

## The seats (Jack's anatomy)

| seat | champion | held | arena | challenger status |
|---|---|---|---|---|
| Learning core | PPO (tuned per Moon et al. — F9) | **DEFAULT, never defended** | LC.00–LC.06 (registered) | DreamerV3-class + others — match in progress |
| Control architecture (D1) | **VACANT** — prior holder's evidence voided (T0.14) | — | D1.0 + T2.21 (queued) | frozen-trunk+head vs tuned-PPO vs others |
| Needs/reward form | drive-reduction, n>m≥1, γ<1 | **BY ANALYSIS** (suicide-safety proof) | NE.02 bakeoff (queued) | must confirm the analysis in the arena |
| Curiosity signal | learning-progress | **BY ANALYSIS** (favourite on pilot data) | LT.03/LT.04 (queued) | ICM/RND enter as controls-that-must-fail |
| Episodic retrieval | lexical containment | BY VERDICT (ME.1/ME.9) — **known weakness: 0.000 on paraphrase** | ME.11.A–F (registered) | potion-8M favourite, cascade the risk |
| Sensory fusion | undecided — token-trunk favourite | — | UB.10 (registered) | six arms, matched params |
| Vision encoder | from-scratch 0.24M | **DEFAULT, never defended — and the FROZEN alternative is now CONTESTED, see below** | T2.03 (registered) + PL.02 (pending registration) | frozen DINOv2/SigLIP vs adapters vs plastic vs pure |
| ASR (speech→text) | whisper.cpp | BY ANALYSIS (3.8–8.3× measured on this box) | HR bakeoff (queued) | — |
| Speaker ID | CAM++ / TitaNet-small | BY ANALYSIS (research only) | HR speaker spec (queued) | needs ARM benchmark — no published numbers exist |
| Language model | SmolLM2-360M as a TALKATIVE PARENT — in his world, speaking to him; NOT inside him | **BY DECREE** (owner 2026-08-09; supersedes the earlier 'frozen mouth' framing) | LG.00 anti-puppet applies to any role | any better frozen swap-in |
| Consolidation | SIESTA wake/sleep, sleep-gated | BY ANALYSIS | NE.05 (queued) | — |
| World | MuJoCo playground + needs overlays | **BY VERDICT** (measured 4–6× faster than Craftax AND goal-aligned) | W.1–W.7 fidelity gates (queued) | rematch at each fidelity stage |
| Audio encoder (world-sound → brain) | undecided — mel favourite (raw costs 12–25×, measured) | — | HR audio-entry bakeoff (queued); PL.* applies here too | raw vs mel vs tokens vs no-audio null |
| Language grounding (word → lived skill) | undecided — skills-then-language ordering unproven | — | LG bakeoff (queued; doc needs completeness check) | grounding approaches + the ordering experiment |

| Smell (olfaction) | **VACANT — sense not yet built** | — | under research (FROZEN_VS_PLASTIC.md) | finds food/fire/decay through occlusion |
| Taste (gustation) | **VACANT — sense not yet built** | — | under research | one-trial aversion learning; poison |
| Voice (vocalisation) | **VACANT — he cannot make a sound** | — | needs a spec | prerequisite for emergent language + GEN.02 |
| Language acquisition | LLM-as-parent (speaks to him; he learns by hearing) | **BY DECREE** (owner 2026-08-09) | LG family + LG.00 anti-puppet | pure from-scratch; critical-period |

### OPEN AND CONTESTED: are the PERCEPTION encoders frozen? (2026-08-09)

**NOT DECIDED.** docs/research/FROZEN_VS_PLASTIC.md recommends against frozen
and the reasoning is arithmetic rather than preference: the RESHAPING GAIN —
does training with sense B improve sense A's own representation? — is
identically zero for a frozen tower, so a frozen encoder makes one class of
learning permanently impossible, which GOAL.md's capability target marks as
suspect by definition. Supporting measurements: OpenVLA 47.0% frozen vs 69.7%
fine-tuned vs 68.2% LoRA at 1.4% of parameters; Kleinman/Achille/Soatto
(CVPR 2023) — separately pretrained backbones may fail to encode synergistic
information, AND IT FAILS SILENTLY (93–94% either way, −20 points only when
the task needs synergy). Recommended ranking: adapters+plastic fusion >
fully pure > critical period > current fully-frozen.

**But law 3 governs: this is a RECOMMENDATION, not a verdict.** PL.02 decides
it and is runnable today. Until PL.02 records a result, no spec, agent or
document may treat "no frozen encoders" as settled. What IS settled is only
the LANGUAGE half, by owner decree: the LLM is a parent in his world, not a
component inside him.

**Three consequences already found, pending action:** the learning-core
bakeoff's admission criterion U2 excludes every frozen tower BY ARITHMETIC and
was never run against one; the "Heard, Not Seen" binding test cannot
discriminate frozen from adapted (it measures readout, not reshaping); and EWC
in TrainingPipeline.py measures indistinguishable from vanilla at our scale.

**Future seats, named so they are not forgotten:** the body itself (Humanoid-v5
has ball hands; fingers will compete one day) and the cross-life world
curriculum (how worlds mutate between deaths). They get chairs when their
first challenger exists.

*Maintenance:* the builder updates this file as part of queue step 5 whenever
a bakeoff completes; the daily Review checks seat staleness (rule 4); the
field watch targets nominations at seats by name.
