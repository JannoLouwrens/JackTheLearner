"""The expanded ladder: GOAL.md made falsifiable.

Encodes docs/MASTER_PLAN.md — playground (PG), unified brain (UB), curiosity
(CU), memory (ME), and the gap specs — as registry entries. Detail, citations
and full test designs live in docs/research/; each spec here carries enough to
be run and to fail. Imported by registry.py and appended to LADDER.

Tier mapping: PG/ME/T2.x -> tier 2 (component vs null), T3.x -> tier 3,
UB -> tier 4 (composition/unison), CU/T5.x -> tier 5 (the claims),
T6.x -> tier 6 (the living Jack).
"""
from __future__ import annotations

from .protocol import Budget, Spec

EXPANSION: list[Spec] = [

    Spec("T2.00", 2, "The RL update is sane",
         hypothesis="Value and policy losses stay within an order of magnitude "
                    "of each other, log_std stays bounded, and actions reaching "
                    "the environment stay inside its range.",
         falsified_by="vf/pg ratio above 50, log_std outside [-4.6, 0], or an "
                      "action exceeding the env limit.",
         null_baseline="Unnormalized returns — the configuration that produced "
                       "vf/pg ~870 and a policy 100x worse than doing nothing.",
         metric="max_vf_pg_ratio", budget=Budget.CPU, depends_on=["T0.06"],
         control="With normalize_returns disabled the ratio MUST explode — a "
                 "guard that passes in both configurations measures nothing.",
         kills="Every GPU locomotion run. This gates T2.01/T2.02 and costs CPU "
               "minutes, so a broken update can never again burn GPU hours.",
         notes="Written after T2.01 measured -4334 trained vs +170 untrained. "
               "Three bugs, none visible in a loss curve: no return "
               "normalization (vf_loss 540.5 vs pg_loss 0.267 on a SHARED "
               "trunk), unbounded log_std, and actions never clipped to the env "
               "range (|a| hit 2.37 vs a +-0.4 limit, so MuJoCo clipped "
               "silently and PPO scored components that never touched physics)."),


    # ── PLAYGROUND (docs/research/CURIOSITY.md §7) ──────────────────────
    Spec("PG.1", 2, "Playground generates and is physically sound",
         hypothesis="A procedural room (ramp, stairs, ladder, objects, seesaw, "
                    "pool, noise panel) builds from a parameter vector and obeys "
                    "physics: boxes slide iff tan(theta) > mu; energy bounded at rest.",
         falsified_by="Objects jitter at rest, energy diverges, or a parameter "
                      "draw produces an invalid MJCF.",
         null_baseline="n/a — physics validation fixture.",
         metric="physics_checks_passed", budget=Budget.CPU,
         kills="Every curiosity claim — a broken world teaches broken lessons."),

    Spec("PG.2", 2, "Water works: buoyancy + drag",
         hypothesis="A passive ragdoll floats at the equilibrium depth its "
                    "density ratio predicts (±10%); submerged motion feels drag.",
         falsified_by="Ragdoll sinks/launches, or equilibrium depth off >10%.",
         null_baseline="Buoyancy callback disabled.",
         metric="equilibrium_depth_error", budget=Budget.CPU, depends_on=["PG.1"],
         control="With buoyancy disabled the ragdoll MUST sink and swim-speed "
                 "must go to ~0 — else the swim metric measures floor contact."),

    Spec("PG.3", 2, "Ladder is climbable in principle (adhesion hands)",
         hypothesis="Adhesion actuators on the hand geoms let a scripted "
                    "kinematic sequence ascend one rung; falling produces clean, "
                    "resumable episodes.",
         falsified_by="Adhesion cannot support body weight at any gain, or "
                      "falls corrupt the episode stream.",
         null_baseline="Zero adhesion — must slip.",
         metric="scripted_rung_ascent", budget=Budget.CPU, depends_on=["PG.1"]),

    Spec("PG.4", 2, "Noisy-TV panel traps naive curiosity",
         hypothesis="The re-randomizing texture panel is a working trap: a "
                    "prediction-error agent fixates on it; dwell-time metric works.",
         falsified_by="The naive-curiosity control arm does NOT fixate — then "
                      "the fixture cannot certify any curiosity claim.",
         null_baseline="Random walk's dwell time near the panel.",
         metric="icm_dwell_share", budget=Budget.CPU_LONG, depends_on=["PG.1"],
         notes="Every later curiosity claim must report dwell share on this fixture."),

    Spec("PG.5", 2, "Procedural contact audio with localization labels",
         hypothesis="Modal-resonator synthesis on MuJoCo contact events yields "
                    "stereo audio whose panning matches source bearing.",
         falsified_by="Bearing decoded from stereo does not match ground truth.",
         null_baseline="Mono/shuffled-pan audio — bearing must be undecodable.",
         metric="bearing_decode_accuracy", budget=Budget.CPU, depends_on=["PG.1"]),

    # ── TIER-2 GAPS (docs/research/CAPABILITIES.md) ─────────────────────
    Spec("T2.14", 2, "Imitation from real motion capture",
         hypothesis="BC on the CMU corpus reaches held-out action error below "
                    "mean-action AND below nearest-neighbour retrieval.",
         falsified_by="A lookup table (NN retrieval) matches the model.",
         null_baseline="Mean-action; nearest-neighbour retrieval.",
         metric="heldout_vs_nn_ratio", budget=Budget.GPU, seeds=3,
         depends_on=["T1.13", "T1.08"]),

    Spec("T2.15", 2, "Free-form language routes to the right task",
         hypothesis="Novel paraphrases of known commands map to the correct "
                    "command cluster above chance (the LLM->task handoff).",
         falsified_by="Held-out phrasings route at chance.",
         null_baseline="Chance routing; bag-of-words retrieval.",
         metric="paraphrase_routing_accuracy", budget=Budget.GPU_SHORT, seeds=3,
         depends_on=["T2.06"],
         notes="The verb x object grid must be designed BEFORE grounding training "
               "(CAPABILITIES.md L2) or the held-out cells cannot exist."),

    Spec("T2.16", 2, "Hindsight goal-reaching (the flow-matching weld)",
         hypothesis="Hindsight-relabeled flow regression reaches commanded "
                    "outcomes above chance with zero RL machinery.",
         falsified_by="Reach-rate <= a policy trained on shuffled goal labels.",
         null_baseline="Shuffled-goal-label training (the critical null).",
         metric="goal_reach_rate", budget=Budget.GPU, seeds=3,
         depends_on=["T2.01"],
         control="Goals outside the achieved-outcome support (fly 2m up) must "
                 "score ~0 — else the success detector is broken, not the policy."),

    Spec("T2.17", 2, "Progress and success estimation",
         hypothesis="Predicted progress correlates with ground-truth stage on "
                    "held-out rollouts including failures.",
         falsified_by="A linear-in-timestep predictor matches it (the null "
                      "everyone skips).",
         null_baseline="Linear-in-timestep regression.",
         metric="progress_spearman", budget=Budget.GPU_SHORT, seeds=3,
         depends_on=["T2.01"],
         control="Reversed-video rollouts must yield reversed progress.",
         kills="Gates LE4/LE5/PL4 — no RL-beyond-demos without a success signal."),

    Spec("T2.18", 2, "Chunking earns its keep under latency",
         hypothesis="Some chunk length k>1 beats k=1 at matched FLOPs, and "
                    "chunk-overlap beats naive swap under 100-300ms latency.",
         falsified_by="k=1 dominates all k, or overlap gives nothing at latency.",
         null_baseline="Per-step prediction; naive chunk swap.",
         metric="chunk_advantage", budget=Budget.GPU, seeds=3,
         depends_on=["T2.01"],
         control="At zero latency, overlap and naive swap must be equivalent."),

    Spec("T2.19", 2, "Flow head handles multimodal actions",
         hypothesis="On a bimodal task (pass obstacle left OR right) the flow "
                    "head succeeds where MSE regression collapses to the mean.",
         falsified_by="L1/MSE regression matches the flow head — OFT found this "
                      "on some benchmarks; genuine falsification risk, and if it "
                      "happens the flow head loses its justification.",
         null_baseline="Deterministic regression head, same params.",
         metric="bimodal_success_ratio", budget=Budget.GPU_SHORT, seeds=3,
         depends_on=["T1.12"],
         control="On a unimodal task the two heads must tie."),

    Spec("T2.20", 2, "Episodic memory helps the next episode",
         hypothesis="With the episodic store, a hidden object is found faster "
                    "in episode N+1 than by a memoryless agent.",
         falsified_by="Search time does not drop across episodes.",
         null_baseline="Memoryless agent; recency-only retrieval.",
         metric="search_time_ratio", budget=Budget.CPU_LONG, seeds=3,
         depends_on=["ME.1"],
         control="Wiping or shuffling the store must restore null search time."),

    # ── MEMORY (docs/research/MEMORY.md) ────────────────────────────────
    Spec("ME.1", 2, "Event log: what happened is retrievable",
         hypothesis="Cued QA over Jack's own event stream answers >=80% at 1k "
                    "events via recency x importance x similarity scoring.",
         falsified_by="Accuracy at 1k events <= recency-only retrieval.",
         null_baseline="Recency-only; no-memory parametric guess.",
         metric="cued_recall_accuracy", budget=Budget.CPU,
         control="A query about a FABRICATED event must abstain — confabulating "
                 "a match means the retrieval threshold is broken."),

    Spec("ME.2", 2, "Owner memory lives on disk",
         hypothesis="A preference stated once is honoured next session; a later "
                    "contradiction supersedes it.",
         falsified_by="Adherence <= a fresh no-memory agent's base rate.",
         null_baseline="No-memory agent; recency window excluding the preference.",
         metric="preference_adherence", budget=Budget.CPU, depends_on=["ME.1"],
         control="WIPE profile.json and restart: adherence must drop to base "
                 "rate — proving memory is in the file, not weights or cache."),

    Spec("ME.3", 2, "Reflections beat raw events",
         hypothesis="Aggregation questions answer better from consolidated "
                    "reflections than from top-k raw events at equal tokens.",
         falsified_by="No gain over raw top-k.",
         null_baseline="Raw-events-only retrieval.",
         metric="aggregation_qa_gain", budget=Budget.CPU, depends_on=["ME.1"],
         control="Reflections generated from ANOTHER agent's log must hurt."),

    Spec("ME.4", 2, "Forgetting keeps what matters",
         hypothesis="Ebbinghaus decay + reinforce-on-recall + supersede beats "
                    "FIFO eviction at a fixed store budget.",
         falsified_by="FIFO matches it on frequently-referenced old facts.",
         null_baseline="FIFO; unbounded store as ceiling.",
         metric="retention_vs_fifo", budget=Budget.CPU, depends_on=["ME.1"],
         control="Knowledge-update questions must FAIL in the no-supersede "
                 "variant (stale answers) — else the questions never conflicted."),

    Spec("ME.5", 2, "Retrieval survives growth",
         hypothesis="Cued-recall precision@1 stays above the recency null as "
                    "the store grows 100 -> 100k events.",
         falsified_by="Precision falls below recency-only at any decade.",
         null_baseline="Recency-only; hand-picked oracle as ceiling (the gap is "
                       "the degradation curve).",
         metric="precision_at_scale", budget=Budget.CPU_LONG, depends_on=["ME.1"],
         notes="Standing spec: re-run at every decade of real store growth."),

    Spec("ME.6", 2, "Skill library accelerates composites",
         hypothesis="A composite task needing two ledger-verified skills is "
                    "reached far faster than learning from scratch.",
         falsified_by="Retrieve-and-compose ~= from-scratch at equal budget.",
         null_baseline="No library; random-skill retrieval.",
         metric="composite_speedup", budget=Budget.GPU, depends_on=["T2.11"],
         control="Corrupting a retrieved skill's body must break the composite — "
                 "proving execution actually uses it."),

    Spec("ME.7", 5, "Sleep consolidation (SIESTA) holds old knowledge",
         hypothesis="After a sleep phase, old-concept accuracy drops <=2 points "
                    "while new concepts are absorbed beyond wake-only prototypes.",
         falsified_by="Catastrophic forgetting after sleep, or sleep never "
                      "beats wake-only.",
         null_baseline="Wake-only forever; naive fine-tune.",
         metric="old_new_retention", budget=Budget.GPU, seeds=3,
         depends_on=["T5.03"],
         control="Sleeping with the rehearsal buffer EMPTIED must forget."),

    Spec("ME.8", 2, "Working memory survives restarts",
         hypothesis="A recurrent state checkpointed to disk resumes mid-episode "
                    "after a kill; zeroing it mid-episode hurts.",
         falsified_by="Post-restart behavior equals a zeroed-state agent.",
         null_baseline="Zeroed hidden state.",
         metric="resume_vs_zeroed", budget=Budget.CPU, depends_on=["T0.05"]),

    # OWNER DIRECTIVE (2026-08-07): "he must also remember what he hears, says
    # and does so when people interact with him... he must keep memory and ALSO
    # learn generally." Two properties ME.1-8 do not pin down: (a) recall that
    # is ATTRIBUTED — heard vs said vs did, and which person — not just cued;
    # (b) the episodic record and the general skill are SEPARATE stores that
    # both survive the other's ablation (complementary learning systems,
    # McClelland et al. 1995; the double dissociation is the test).
    Spec("ME.9", 2, "He remembers what he hears, says, and does — attributed",
         hypothesis="Cued recall works across all three channels (heard "
                    "utterance, own utterance, own action) at >=80% each, AND "
                    "source attribution survives: 'what did I tell you' is "
                    "answered from heard-events, 'what did you say/do' from "
                    "own-events, per speaker across >=3 interleaved speakers.",
         falsified_by="Any channel at chance, or attribution confuses "
                      "who-said-what once conversations interleave.",
         null_baseline="Channel-blind retrieval over the pooled log (same "
                       "events, provenance stripped) — it must fail the "
                       "attribution questions specifically.",
         metric="attributed_recall_accuracy", budget=Budget.CPU,
         depends_on=["ME.1"], seeds=3,
         control="Swapped-provenance store (his lines relabelled as the "
                 "speaker's and vice versa) must invert attribution answers; "
                 "if accuracy survives the swap, the test never used "
                 "provenance and is measuring text similarity."),

    Spec("ME.10", 2, "Keeps the memory AND learns the general skill",
         hypothesis="After episodes are distilled into weights (practice/"
                    "replay), the verbatim episodic record still answers cued "
                    "recall at its pre-distillation rate, AND the distilled "
                    "skill outperforms no-distillation; then the double "
                    "dissociation: wiping the episodic store leaves the skill "
                    "intact, wiping the weight update leaves recall intact.",
         falsified_by="Distillation degrades recall (learning ate the memory) "
                      "or recall requires the store at skill-time (nothing "
                      "was ever in the weights).",
         null_baseline="No-distillation agent: same store, no weight update — "
                       "its skill gap is what distillation must beat.",
         metric="recall_kept_x_skill_gained", budget=Budget.CPU,
         depends_on=["ME.1", "T1.04"], seeds=3,
         control="The two ablations must each destroy exactly their own "
                 "capability: store-wipe kills recall (not skill), "
                 "weight-revert kills the skill gain (not recall). Either "
                 "ablation killing BOTH means one store is masquerading as "
                 "two.",
         kills="Any design where conversation memory lives only in weights "
               "or skills live only in retrieved episodes."),

    # ── TIER-3 GAPS ─────────────────────────────────────────────────────
    Spec("T3.09", 3, "The creative loop earns its existence",
         hypothesis="Wiring AlphaGeometryLoop into a decision path measurably "
                    "improves something against the same path without it.",
         falsified_by="No measurable difference — currently GUARANTEED, since "
                      "the loop has ZERO call sites: it constructs, prints "
                      "'ENABLED', and is never invoked.",
         null_baseline="Identical system, loop disabled.",
         metric="creative_contribution", budget=Budget.CPU_LONG,
         kills="AlphaGeometryLoop.py (559 lines) — wire it or delete it."),

    Spec("T3.10", 3, "Trunk knowledge survives action training",
         hypothesis="Linear probes on frozen-trunk features (object class, "
                    "color, spatial relation) hold constant through action "
                    "training AND semantic-task success tracks probe quality.",
         falsified_by="Probes drift (gradient leak — a bug), or probes hold "
                      "while semantic tasks sit at chance (knowledge not "
                      "reaching the action head — architecture flaw).",
         null_baseline="Probes on a random-weight trunk.",
         metric="probe_drift", budget=Budget.GPU_SHORT, depends_on=["T2.03"],
         control="Deliberately unfreezing the trunk must reproduce the drift.",
         notes="Cheapest direct evidence for/against decision D1 (arXiv:2505.23705)."),

    # ── UNIFIED BRAIN (docs/research/UNIFIED_BRAIN.md; tier 4 = unison) ─
    Spec("UB.1", 4, "No modality collapse (the ablation matrix)",
         hypothesis="With modality dropout, every sense is load-bearing "
                    "somewhere: zero/noise/shuffle/swap each hurt some task — "
                    "no all-zero column in the tasks x senses matrix.",
         falsified_by="Any sense whose entire column is zero — it is decorative.",
         null_baseline="Twin run WITHOUT dropout (may collapse onto proprio).",
         metric="ablation_matrix_min_column", budget=Budget.GPU, seeds=3,
         depends_on=["T4.01"],
         control="With proprio zeroed, the dropout-trained model must still "
                 "briefly stand from vision."),

    Spec("UB.2", 4, "The shared trunk beats late fusion",
         hypothesis="One self-attention trunk over all modality tokens beats "
                    "equal-parameter separate-encoders-then-concat.",
         falsified_by="Late fusion ties everywhere incl. occlusion tasks — then "
                      "'one brain' adds nothing at this scale; report honestly.",
         null_baseline="Per-modality encoders -> concat -> same flow head.",
         metric="fusion_advantage", budget=Budget.GPU, seeds=3,
         depends_on=["UB.1"],
         control="Cross-modal TIME-SHUFFLE at eval must hurt the shared trunk — "
                 "else attention never crossed modalities."),

    Spec("UB.3", 4, "Cross-modal masking helps the policy",
         hypothesis="Co-training with masked cross-modal prediction (touch from "
                    "vision+proprio, audio-event from dynamics) improves task "
                    "success and few-shot adaptation at equal steps.",
         falsified_by="No downstream improvement — drop the objective.",
         null_baseline="BC-only, same architecture and steps.",
         metric="mask_cotrain_gain", budget=Budget.GPU, seeds=3,
         depends_on=["UB.2"],
         control="Touch-from-SHUFFLED-vision must collapse to the unconditional "
                 "mean — else the head ignores vision and the fusion is fake."),

    Spec("UB.4", 4, "Hearing is load-bearing",
         hypothesis="Jack turns toward an out-of-view falling object and times "
                    "occluded contacts using audio.",
         falsified_by="Muting audio at eval leaves audio-task success unchanged.",
         null_baseline="Audio-muted model; model trained without audio.",
         metric="audio_task_delta", budget=Budget.GPU, seeds=3,
         depends_on=["PG.5", "UB.1"],
         control="Left/right channel swap must invert turning; 500ms audio lag "
                 "must break contact timing — else hearing is decorative."),

    Spec("UB.5", 4, "Touch is load-bearing (or honestly redundant)",
         hypothesis="Touch improves blind push-recovery beyond proprioception.",
         falsified_by="Zeroed touch changes nothing — an HONEST possible "
                      "outcome: foot force is partly inferable from torques. "
                      "That is a finding, not a failure of the test.",
         null_baseline="Touch-zeroed eval; touch-ablated training.",
         metric="blind_recovery_delta", budget=Budget.GPU_SHORT, seeds=3,
         depends_on=["UB.1"],
         control="Permuting the 10 touch channels must cause misattributed "
                 "contacts if touch is load-bearing."),

    Spec("UB.6", 4, "Contrastive binding: keep only if it moves action",
         hypothesis="Audio<->vision alignment improves hearing-task success "
                    "beyond the same compute spent on BC.",
         falsified_by="No task-success delta — binding is retrieval-only here.",
         null_baseline="Same model, alignment weight zero.",
         metric="bind_action_gain", budget=Budget.GPU_SHORT,
         depends_on=["UB.4"],
         control="The aligned model must retrieve audio->vision clips well "
                 "above chance — else the loss never worked and the null result "
                 "is uninformative."),

    Spec("UB.7", 4, "UNISON — the headline claim",
         hypothesis="The shared co-trained trunk beats BOTH per-sense "
                    "specialists AND frozen-separate-encoders at matched "
                    "params/steps, on a battery where each sense matters "
                    "somewhere.",
         falsified_by="The bolt-on baseline ties everywhere.",
         null_baseline="(i) specialists; (ii) frozen separate encoders + concat.",
         metric="unison_advantage", budget=Budget.GPU_LONG, seeds=3,
         depends_on=["UB.2", "UB.3", "UB.4"],
         control="Leave-one-task-family-out retraining must SHIFT other tasks — "
                 "zero shift means the trunk partitioned into covert late fusion.",
         notes="Until this passes, the sentence 'the senses work in unison' "
               "stays OUT of every capability list."),

    Spec("UB.8", 4, "Flow-head attention ablation",
         hypothesis="Interleaved cross+self attention beats cross-only and "
                    "self-only at equal params (SmolVLA's ablation, reproduced).",
         falsified_by="No difference — simplify to cross-only, bank the params.",
         null_baseline="The two single-attention variants.",
         metric="attention_ablation", budget=Budget.GPU_SHORT,
         depends_on=["UB.7"]),

    # ── CURIOSITY (docs/research/CURIOSITY.md; tier 5 = the claims) ─────
    Spec("CU.1", 5, "Goal babbling beats action babbling",
         hypothesis="Sampling goals in OUTCOME space covers more distinct "
                    "outcomes than random action sequences at equal budget.",
         falsified_by="Coverage <= the random-action-repeat null (flailing "
                      "covers ground too).",
         null_baseline="Random repeated action sequences.",
         metric="outcome_coverage_ratio", budget=Budget.CPU_LONG, seeds=3,
         depends_on=["PG.1", "T2.16"]),

    Spec("CU.2", 5, "Learning progress produces an emergent curriculum",
         hypothesis="LP-driven goal sampling yields time-ordered mastery "
                    "(stand -> walk -> push -> ramp) with distinct onsets, and "
                    "higher final multi-goal success than uniform sampling.",
         falsified_by="Mastery onsets simultaneous or seed-random.",
         null_baseline="Uniform goal sampling with identical relabeling.",
         metric="curriculum_ordering", budget=Budget.GPU_LONG, seeds=3,
         depends_on=["CU.1"],
         control="Forever-unlearnable goals ('make the noise panel blue') must "
                 "decay to epsilon allocation — else the competence estimator "
                 "is broken.",
         notes="The first falsifiable form of 'Jack teaches himself'."),

    Spec("CU.3", 5, "Curious without being trapped",
         hypothesis="The LP stack explores (coverage grows) with near-zero "
                    "dwell at the noisy-TV panel.",
         falsified_by="Panel dwell share exceeds the random-walk baseline.",
         null_baseline="Random walk; an ICM arm as the trap-victim reference.",
         metric="coverage_vs_dwell", budget=Budget.CPU_LONG, seeds=3,
         depends_on=["PG.4", "CU.2"],
         control="The ICM control arm MUST fixate on the panel — proving the "
                 "trap works and the LP immunity is real."),

    Spec("CU.4", 5, "Unsupervised skills are real and distilled",
         hypothesis="METRA skills on trunk embeddings are decodable from "
                    "trajectories (>90%) and beat the random-repeat null on "
                    "displacement; distillation carries them into the flow head.",
         falsified_by="Skill classifier at chance (collapse), or displacement "
                      "<= flailing.",
         null_baseline="Random repeated actions; DIAYN as the static-pose "
                       "reference.",
         metric="skill_decodability", budget=Budget.GPU_LONG, seeds=3,
         depends_on=["CU.2"],
         control="Ablating METRA's temporal-distance constraint must degrade "
                 "toward static poses (arXiv:2310.08887)."),

    Spec("CU.5", 5, "The VLM proposes, learning progress disposes",
         hypothesis="VLM-proposed + LP-filtered goals engage the ladder and "
                    "pool earlier and rate more interesting (blind A/B) than "
                    "LP-only.",
         falsified_by="No rating difference, or VLM goals flood the buffer "
                      "while their success stays ~0 (hallucinated curriculum).",
         null_baseline="LP-only at matched goal count.",
         metric="proposal_value", budget=Budget.GPU_LONG,
         depends_on=["CU.2", "PG.3"],
         control="A scrambled-caption VLM (fed another scene) must NOT beat "
                 "LP-only — else the benefit was 'more goals', not grounded "
                 "interestingness."),

    Spec("CU.6", 5, "Affordances emerge from interaction",
         hypothesis="The interaction archive predicts pushability/liftability "
                    "of held-out objects above chance.",
         falsified_by="Prediction at chance on novel mass/shape.",
         null_baseline="Predictor trained on shuffled object-outcome pairs.",
         metric="affordance_transfer", budget=Budget.CPU_LONG, seeds=3,
         depends_on=["CU.1"],
         control="A welded immovable object must classify un-pushable — else "
                 "the representation captures action, not interaction."),

    Spec("CU.7", 5, "Lessons from failure improve retries",
         hypothesis="Retrieved one-line lessons written after failures raise "
                    "retry success beyond pure resampling.",
         falsified_by="Retry rate with lessons equals resampling alone (the "
                      "known confound).",
         null_baseline="Retry with no lesson.",
         metric="lesson_gain", budget=Budget.CPU_LONG, seeds=3,
         depends_on=["ME.1", "CU.2"],
         control="Lessons from UNRELATED failures must not help — else the "
                 "effect is generic prompt padding."),

    # ── TIER-5/6 GAPS ───────────────────────────────────────────────────
    Spec("T5.08", 5, "Open-endedness: learning does not saturate",
         hypothesis="With ACCEL-style scene mutation + interestingness filter, "
                    "distinct mastered outcome clusters grow for 8 weeks "
                    "without plateau.",
         falsified_by="Cluster count plateaus while the fixed-scene null keeps "
                      "pace at equal budget.",
         null_baseline="Fixed single playground, same total steps.",
         metric="cluster_growth_curve", budget=Budget.GPU_LONG,
         depends_on=["CU.2", "T5.06"],
         control="Mutation WITHOUT the learnability filter must degenerate "
                 "into unsolvable scenes — else the filter does nothing."),

    Spec("T5.09", 5, "Skills transfer across bodies",
         hypothesis="Pretraining on morphology variants (limb lengths, masses) "
                    "speeds learning on a new body versus random init.",
         falsified_by="Transfer <= random init, or negative transfer.",
         null_baseline="Random init on the target body.",
         metric="transfer_speedup", budget=Budget.GPU_LONG, seeds=3,
         depends_on=["T2.02"],
         control="Pretraining on white-noise trajectories must give no gain — "
                 "it is structure, not warm optimizer state."),

    Spec("T6.05", 6, "Companion battery",
         hypothesis="Responses are contingent on user-avatar events; intent is "
                    "inferred above majority-class; the user zone is violated "
                    "<1/1000 episodes across reward scales; identity is "
                    "distinguishable from a re-seeded twin.",
         falsified_by="Any leg fails: time-shuffled events show identical "
                      "response stats, intent at chance, safety trades off "
                      "against reward, or the twin is indistinguishable.",
         null_baseline="Time-shuffled events; majority-class; task-only policy; "
                       "wiped-persona twin.",
         metric="companion_battery", budget=Budget.GPU_LONG,
         depends_on=["T6.01", "ME.2"],
         control="Ablating the safety channel must bring violations back; "
                 "persona reset must drop identity to chance."),
]
