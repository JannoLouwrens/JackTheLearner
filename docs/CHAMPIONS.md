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
| Learning core | PPO (tuned per Moon et al. — F9) | **DEFAULT, never defended** | LC.00–LC.06 (LC.00–LC.02 + PS.01 PASS; LC.03 registered run IN FLIGHT since 2026-08-13 15:23, ~15–20 h, five arms) | DreamerV3-class + others — screening only: LC.03 declares no winner; the seat's actual match is LC.04, still ahead |
| Deliberation (the slow path) | **VACANT — never contested.** A reactive-only Jack is the *incumbent by default*, which is a position nobody argued for | — | DP.00–DP.04 (registered 2026-08-10) | model-based lookahead vs reactive-only vs verbal inner speech (DP.04). DP.00 asks first whether this world rewards lookahead AT ALL — if not, the seat is abolished rather than filled |
| Fast/slow coupling | **DECREE, contestable by evidence.** Owner 2026-08-10: differentiated function, SHARED substrate — "connected but slightly different purposes" | DECREE | DP.02 (registered) | two private towers. Re-open trigger: DP.02 shows a shared-trunk lesion sparing one mode, i.e. the substrate is already separate in fact |
| Control architecture (D1) | **VACANT** — prior holder's evidence voided (T0.14) | — | D1.0 + T2.21 (queued) | frozen-trunk+head (**barred pending the D1 reconciliation, DECISIONS_NEEDED.md:599** — a frozen control trunk conflicts with the PLASTIC-ONLY decree unless the owner narrows its scope) vs tuned-PPO vs others |
| Needs/reward form | drive-reduction, n>m≥1, γ<1 | **BY ANALYSIS** (suicide-safety proof) | NE.02 bakeoff (queued) | must confirm the analysis in the arena |
| Curiosity signal | learning-progress | **BY ANALYSIS** (favourite on pilot data) | LT.03/LT.04 (queued) | ICM/RND enter as controls-that-must-fail |
| Episodic retrieval | lexical containment | BY VERDICT (ME.1/ME.9) — **known weakness: 0.000 on paraphrase** | ME.11.A–F (registered) | potion-8M favourite, cascade the risk |
| Sensory fusion | undecided — token-trunk favourite | — | UB.10 (registered; `UB.9` **PASS 2026-08-12** — fused 0.993 vs unimodal/ensemble nulls at chance — so the matrix has something to eat, and UB.10 is now the project's third-ranked blocker) | six arms, matched params |
| Vision encoder | from-scratch 0.24M | **DEFAULT, never defended — and the FROZEN alternative is now CONTESTED, see below** | T2.03 **PASS 2026-08-13** (pretrained 0.9867/0.9833/0.9533 vs from-scratch 0.4467/0.4667/0.4933 vs random-projection 0.40, seeds 0/1/2 — the gap is measured, ~half the accuracy range; seats nobody frozen per PLASTIC-ONLY) + PL.02 (pending registration) | frozen DINOv2/SigLIP vs adapters vs plastic vs pure; PROGRESS §7 asks the owner to admit a warm-start-plastic arm |
| ASR (speech→text) | whisper.cpp | BY ANALYSIS (3.8–8.3× measured on this box) | HR bakeoff (queued) | — |
| Speaker ID | CAM++ / TitaNet-small | BY ANALYSIS (research only) | HR speaker spec (queued) | needs ARM benchmark — no published numbers exist |
| Language model | SmolLM2-360M as a TALKATIVE PARENT — in his world, speaking to him; NOT inside him | **BY DECREE** (owner 2026-08-09; supersedes the earlier 'frozen mouth' framing) | LG.00 anti-puppet applies to any role | any better frozen swap-in |
| Consolidation | SIESTA wake/sleep, sleep-gated | BY ANALYSIS | NE.05 (queued) | — |
| World | MuJoCo playground + needs overlays | **BY VERDICT** (measured 4–6× faster than Craftax AND goal-aligned) | W.1–W.7 fidelity gates (queued) | rematch at each fidelity stage |
| Audio encoder (world-sound → brain) | undecided — mel favourite (raw costs 12–25×, measured) | — | HR audio-entry bakeoff (queued); PL.* applies here too | raw vs mel vs tokens vs no-audio null |
| Language grounding (word → lived skill) | undecided — skills-then-language ordering unproven | — | LG bakeoff (queued; doc needs completeness check) | grounding approaches + the ordering experiment |

| Smell (olfaction) | **VACANT — sense fixture certified, seat unclaimed**: `SM.01` (field fidelity) **PASS 2026-08-11** | — | `SM.02` (occluded food, the claim) is runnable today | finds food/fire/decay through occlusion |
| Taste (gustation) | **VACANT — poison fixture certified, seat unclaimed**: `TA.01` **PASS** | — | `TA.02` (one-trial aversion, the claim) is runnable today | one-trial aversion learning; poison |
| Voice (vocalisation) | **VACANT — but he CAN make a sound**: `VO.01` **PASS 2026-08-12** after four honest FAILs (the emitter is certified; no arm is seated) | — | `VO.02` (voice as act, the claim) is runnable today | prerequisite for emergent language + GEN.02 |
| Language acquisition | LLM-as-parent (speaks to him; he learns by hearing) | **BY DECREE** (owner 2026-08-09) | LG family + LG.00 anti-puppet | pure from-scratch; critical-period |

### DECIDED BY DECREE 2026-08-09: PLASTIC ONLY. NO FROZEN COMPONENTS IN JACK.

Owner: *"if there's already published I decide on using plastic only and never
frozen in our Jack."* Everything INSIDE Jack learns. Nothing inside him is
welded shut.

SCOPE, precisely: this governs components INSIDE Jack — his encoders, his
core, his fusion. It does NOT touch the parent LLM, which by the earlier
decree is not inside him at all; it lives in his world and speaks to him.
A frozen thing in his environment is not a frozen part of him.

THE EVIDENCE BEHIND IT (not in dispute): reshaping gain — does training with
sense B improve sense A's own representation? — is identically ZERO for a
frozen tower, by arithmetic. GOAL.md's capability target marks any choice that
forecloses a class of learning as suspect by definition. Published support:
OpenVLA 47.0% frozen vs 69.7% fine-tuned; M3L's vision+touch training improved
VISION-ONLY policies at test time; Kleinman/Achille/Soatto (CVPR 2023) —
separately pretrained backbones may fail to encode synergistic information,
and fail SILENTLY.

THE STRONGEST COUNTERARGUMENT, recorded because directives enter with eyes
open (SYSTEM.md):
  1. The research's OWN top pick was frozen-base + adapters + plastic fusion.
     Fully pure ranked SECOND. This decree overrules the #1 recommendation.
  2. Pure forfeits inherited visual knowledge — a real head start. Expect a
     longer, more data-hungry childhood for his eyes.
  3. It eliminates the PL.* bakeoff rather than winning it: of four arms
     (frozen / frozen+adapters / critical-period / pure), three involve
     freezing at some stage. Only pure survives, so there is nothing left to
     arbitrate. Two cheap CPU tests could have decided this in days; the owner
     chose to decide it on published evidence instead. That is his call and it
     is legitimate — but it is a DECREE, not a measurement, and this file says
     so.
  4. RE-OPEN TRIGGER, pre-registered so the decree is falsifiable rather than
     permanent: if a from-scratch encoder cannot hit the PL.00 throughput
     floor on this hardware, or if visual competence has not cleared its null
     after the budget PL.00 declares, the decision returns to the owner with
     that number attached.

WHAT STILL RUNS: PL.00 (throughput — now a feasibility check on the pure
encoder rather than a comparison) and PL.02 (reshaping gain — now measuring
what the plastic path BUYS, not whether to take it). Both cheap, both CPU.

### Superseded context: the contested phase (kept for the record)

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
