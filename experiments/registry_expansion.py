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


    Spec("T0.14", 0, "Evaluation is deterministic and the obs contract holds",
         hypothesis="Two forwards of one state in eval mode are BIT-IDENTICAL; "
                    "rollout leaves the model in eval mode and the PPO update "
                    "in train mode; config.mujoco_obs_dim equals what the env "
                    "actually emits.",
         falsified_by="Any drift between two eval forwards, a mode left wrong "
                      "after rollout or update, or an obs-dim mismatch.",
         null_baseline="n/a — an invariant, not an effect.",
         metric="eval_action_drift", budget=Budget.CPU, depends_on=["T0.06"],
         control="Forced into TRAIN mode the determinism check MUST fail. A "
                 "property that cannot be violated is not being tested — which "
                 "is exactly how this went unnoticed for four GPU runs.",
         kills="Every locomotion result computed before it passes. T2.01 and "
               "T2.02 must be re-run once this holds.",
         notes="TrainingPipeline never called .eval()/.train(), so 36 nn.Dropout "
               "modules at p=0.1 were live during rollout, the PPO update, and "
               "'deterministic' evaluation. Measured: 42% policy-mean drift on "
               "the same state, 66% on value, ~20% of samples outside "
               "clip_range at ZERO policy change. Invisible by inspection "
               "because the SB3 baseline disables training mode for you — so "
               "T2.02 compared one arm with 42% injected action noise against "
               "one with none. Also caught: mujoco_obs_dim=376 is the "
               "Humanoid-v4 value; v5 emits 348, so 28 zeros were padded in."),

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
         metric="scripted_rung_ascent", budget=Budget.CPU, depends_on=["PG.1"],
         seeds=3,
         notes="Seeds map to rung spacings 0.30/0.26/0.34 m — 'climbable' must "
               "hold across the middle of the mutation range, not one geometry."),

    Spec("PG.4", 2, "Noisy-TV panel traps naive curiosity",
         hypothesis="The re-randomizing texture panel is a working trap: a "
                    "prediction-error agent fixates on it; dwell-time metric works.",
         falsified_by="The naive-curiosity control arm does NOT fixate — then "
                      "the fixture cannot certify any curiosity claim.",
         null_baseline="Random walk's dwell time near the panel.",
         metric="icm_dwell_share", budget=Budget.CPU_LONG, depends_on=["PG.1"],
         seeds=3,
         notes="Every later curiosity claim must report dwell share on this fixture. "
               "Control: identical ICM agent with a STATIC panel texture must not "
               "fixate — else dwell measures geometry, not noise."),

    Spec("PG.5", 2, "Procedural contact audio with localization labels",
         hypothesis="Modal-resonator synthesis on MuJoCo contact events yields "
                    "stereo audio whose panning matches source bearing.",
         falsified_by="Bearing decoded from stereo does not match ground truth.",
         null_baseline="Mono/shuffled-pan audio — bearing must be undecodable.",
         metric="bearing_decode_accuracy", budget=Budget.CPU, depends_on=["PG.1"],
         seeds=3,
         control="Mono and shuffled-pan renders of the SAME events must decode "
                 "at chance — else the decoder reads something other than pan."),

    Spec("PG.8", 2, "Jack is IN the playground and can act in it",
         hypothesis="make_playground(with_humanoid=True) yields a model that "
                    "contains the Humanoid body with 17 actuators, settles "
                    "finite at rest, emits the 348-dim observation "
                    "TrainingPipeline expects, and spawns within reach of the "
                    "ladder base.",
         falsified_by="No humanoid body, nu != 17, non-finite state after "
                      "settling, an observation dimension that disagrees with "
                      "the pipeline, or a spawn point from which the ladder "
                      "cannot be reached.",
         null_baseline="The playground as it stands today: bodies are "
                       "[world, apple, obj0-4, seesaw] and nu = 0. It must "
                       "fail every check above — there is nobody in it.",
         metric="humanoid_present_and_actuated", budget=Budget.CPU,
         depends_on=["PG.1", "T0.14"], seeds=3,
         control="A humanoid spawned OUTSIDE the arena must fail the "
                 "ladder-reachability check — otherwise 'reachable' is not "
                 "measuring position and the spec would pass anywhere.",
         kills="Every curiosity claim, and the ladder-and-apple standard "
               "itself. CU.*, LT.* and PG.4's dwell metrics are all defined "
               "over an agent acting in this world; none of them can be run "
               "in an empty one.",
         notes="FOUND 2026-08-09 by the hearing research, verified directly: "
               "the playground has NO humanoid and ZERO actuators. "
               "build_mjcf() takes with_humanoid=False and nothing in the "
               "repo ever passes True. PG.1-PG.7 all PASS and all are honest "
               "— they certify the WORLD's physics: friction discriminates "
               "1751x, water floats at the Archimedes depth, contact audio "
               "pans correctly. PG.3 climbs the ladder with what its own "
               "docstring calls 'a certification jig, not a humanoid'. So the "
               "ladder is climbable, the apple is on top, the pool holds "
               "water — and there is nobody there to climb, swim or fall. "
               "This is the gap between a green ladder and GOAL.md."),

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
         seeds=3,
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

    # OWNER PRINCIPLE (2026-08-09): "isn't it better if it isn't an LLM
    # remembering?" Yes, and this spec makes it structural. Memory is
    # EXTRACTIVE, NEVER GENERATIVE: what Jack reports about his past must be a
    # literal stored record or nothing. A language model may INDEX the log
    # (embeddings are a distance function) but must never author the answer,
    # because a generator cannot abstain honestly -- fluency is not evidence.
    # The weakness this fixes is real and measured: lexical containment nails
    # "the ladder" and abstains on "what did ada say was broken about the
    # steps", i.e. every question a person would actually ask.
    Spec("ME.11", 2, "Finds the memory from a paraphrase, still never invents one",
         hypothesis="Cued recall stays >=80% when cues are PARAPHRASES sharing "
                    "no content words with the stored event (synonyms, "
                    "circumlocutions, indirect questions), while fabricated-"
                    "event abstention stays >=95% and every returned answer is "
                    "byte-identical to a stored record.",
         falsified_by="Paraphrase recall at the lexical baseline (i.e. the "
                      "index did not help), OR abstention degrading as recall "
                      "improves (the retriever bought recall with credulity), "
                      "OR any returned string not present verbatim in the log.",
         null_baseline="The current lexical-containment retriever, which "
                       "measured 0/4 on paraphrased cues.",
         metric="paraphrase_recall_at_fixed_abstention", budget=Budget.CPU,
         depends_on=["ME.1", "ME.11.0"], seeds=3,
         control="A DISTRACTOR store where the paraphrase's true target is "
                 "removed but topically-similar events remain: the retriever "
                 "must abstain rather than return the nearest neighbour. "
                 "Semantic matching makes confabulation EASIER, so the "
                 "abstention floor is the thing under test, not the recall.",
         kills="Any retriever that generates its answer instead of quoting "
               "one, however good its numbers."),

    # ── ME.11 BAKEOFF: the arms that make ME.11 decidable ────────────────
    # From docs/research/MEMORY_RETRIEVAL_BAKEOFF.md (agent, 2026-08-09), which
    # measured three things on this box that reframe the problem:
    #  (1) the incumbent retriever scores 0/8 on paraphrase cues -- ME.1's
    #      0.8667 is real but is about cues that are WORD SUBSETS of their
    #      target, exactly the case lexical containment aces;
    #  (2) the 0.34 abstention floor has a ONE-BASIS-POINT margin (worst real
    #      cue 0.000 vs best fabricated 0.333), so the threshold, not the
    #      encoder, is the hard part;
    #  (3) raw top-1 cosine separates real from fabricated better (AUC
    #      0.975-1.000) than every per-query-normalised statistic the
    #      2024-2026 literature recommends (0.54-0.80) -- on a diary corpus
    #      the standard advice is inverted, so each arm MEASURES its
    #      abstention statistic rather than inheriting one.
    # One shared fixture (experiments/fixtures/paraphrase_eval.py) generates,
    # per seed, a 5,000-event life, 240 cues in 4 registers with MECHANICALLY
    # derived gold SETS, and 600 adversarial negatives in 4 families. Its hash
    # goes into every arm's metrics so two arms cannot silently be scored on
    # different data.
    Spec("ME.11.0", 2, "The paraphrase eval set is honest before anyone is scored",
         hypothesis="Every cue shares NO content word with its target beyond an "
                    "explicitly allowed speaker name; the lexical-containment "
                    "null therefore scores <=0.10 on the cue set; gold sets are "
                    "derived from the generator's concept bindings, not hand "
                    "labels; and the ORACLE ceiling (score events by their "
                    "concept-tuple overlap with the cue's concept constraints, "
                    "re-parsed from the STORED TEXT) is >=0.95, proving the "
                    "questions are answerable at all.",
         falsified_by="Any cue-target content-word intersection outside the "
                      "allowed set, OR lexical null >0.10 (the cues leaked "
                      "surface form), OR oracle ceiling <0.95 (the cues are "
                      "not answerable and every arm's score is a floor effect), "
                      "OR the fixture hash differing across two builds at the "
                      "same seed (the eval set is not frozen).",
         null_baseline="Lexical containment on the cue set — must be ~0 BY "
                       "CONSTRUCTION. This spec exists to verify the "
                       "construction, so its null is its own primary assertion.",
         metric="eval_set_validity", budget=Budget.CPU, depends_on=["ME.1"],
         seeds=3,
         control="A DELIBERATELY LEAKY cue set (cues built by deleting words "
                 "from the target rather than by synonym substitution) must "
                 "make the lexical null score >=0.80. If the leak detector "
                 "cannot detect a planted leak it is not a detector.",
         kills="The entire bakeoff. An arm scored against an unvalidated eval "
               "set produces a number nobody may cite.",
         notes="Also asserts >=19 positives per provenance stratum (the "
               "Mondrian conformal minimum at alpha=0.05) and >=300 tune + "
               ">=300 certify negatives, family-balanced (the Clopper-Pearson "
               "minimum to certify abstention >=0.95 at 95% confidence). "
               "Freezes cue set, gold sets and negatives by hash."),

    Spec("ME.11.A", 2, "Arm A — lexical containment, the incumbent, as the null",
         hypothesis="The shipped EpisodicMemory retriever (content-word "
                    "containment x recency x importance, abstain_below=0.34) "
                    "scores <=0.10 paraphrase recall@1 while abstaining >=0.95 "
                    "on adversarial negatives: honest and useless, quantified.",
         falsified_by="Paraphrase recall@1 >0.30 — in which case the premise of "
                      "ME.11 is wrong, lexical matching does generalise, and no "
                      "encoder is needed. This arm is written to be beatable; if "
                      "it is not beaten the bakeoff is cancelled and the compute "
                      "is saved.",
         null_baseline="Recency-only retrieval (ME.1's null), carried forward "
                       "unchanged so all three specs share one floor.",
         metric="paraphrase_recall_at_fixed_abstention", budget=Budget.CPU,
         depends_on=["ME.11.0"], seeds=3,
         control="On the ME.1-style TEMPLATED cue set this same code must still "
                 "score >=0.80. An arm that fails its own home benchmark is "
                 "mis-wired, and its 0.10 on paraphrases would mean nothing.",
         notes="Measured pilot: 0/8 paraphrase cues, and only 1 of 8 cleared "
               "the 0.34 floor. Report N1 (held-out-target) abstention "
               "separately; that is where the floor is expected to fail."),

    Spec("ME.11.B", 2, "Arm B — BM25S with stemming, real lexical SOTA",
         hypothesis="A properly implemented BM25 (bm25s, Snowball stemming, "
                    "stopwords, k1=1.2 b=0.75) beats Arm A on paraphrase "
                    "recall@1 while keeping lexical retrieval's free abstention "
                    "(a query whose terms appear nowhere returns an EMPTY list, "
                    "no threshold needed), at <=2 ms/query at 100k events.",
         falsified_by="No gain over Arm A — i.e. the incumbent's weakness is "
                      "semantic, not an implementation defect, and stemming "
                      "buys nothing. (Pilot says 0.125 vs 0.000: a real but "
                      "tiny gain.)",
         null_baseline="Arm A.",
         metric="paraphrase_recall_at_fixed_abstention", budget=Budget.CPU,
         depends_on=["ME.11.0"], seeds=3,
         control="Shuffle the term-document matrix rows: recall must collapse "
                 "to ~1/N. A BM25 that scores the same on a shuffled index is "
                 "reading document length, not content.",
         notes="Measured: build 100k = 4.24 s, query = 0.876 ms — 40x FASTER "
               "than the incumbent's 35.4 ms linear scan, so whatever wins on "
               "recall, this replaces the scan on efficiency grounds alone. "
               "BM25S: Lu, arXiv:2407.03618."),

    Spec("ME.11.C", 2, "Arm C — static embeddings (potion-base-8M), near-free semantics",
         hypothesis="A distilled STATIC embedding table (model2vec potion-base-8M, "
                    "256d, 7.56M params, 30 MB, no attention) with corpus "
                    "mean-centering and a split-conformal threshold beats Arm B "
                    "on paraphrase recall@1 by >=0.30 absolute while holding "
                    "certified abstention >=0.95, at <=20 ms/query at 100k events.",
         falsified_by="Recall gain over Arm B <0.30, OR certified abstention "
                      "<0.95 at the conformal threshold, OR the coverage and "
                      "false-answer thresholds proving INFEASIBLE (tau_fpr > "
                      "tau_cov) — semantics bought recall with credulity, which "
                      "ME.11 explicitly forbids.",
         null_baseline="Arm B (BM25S). Also reported: potion-base-2M (64d) and "
                       "static-retrieval-mrl-en-v1 truncated to 256d, as "
                       "within-arm variants — the arm is 'static embeddings', "
                       "not one checkpoint.",
         metric="paraphrase_recall_at_fixed_abstention", budget=Budget.CPU,
         depends_on=["ME.11.0"], seeds=3,
         control="RANDOM-PROJECTION control: replace the learned embedding "
                 "table with a random Gaussian matrix of identical shape, "
                 "re-center, re-calibrate. Recall must collapse to ~chance. If "
                 "a random table scores anywhere near the learned one, the arm "
                 "is measuring sentence length or token count, not meaning.",
         notes="Measured on this box: 0.123 ms/query encode, 15,258 docs/s, "
               "100k index built in 6.6 s and held in 102 MB. Pilot p@1 0.625, "
               "recall@10 1.000. Cheapest arm that could plausibly win, and its "
               "6.6 s reindex (vs MiniLM's 18 min) is an operational argument "
               "in its favour on a tenant-serving box. Model2Vec: Zenodo "
               "10.5281/zenodo.17270888."),

    Spec("ME.11.D", 2, "Arm D — a real sentence encoder (all-MiniLM-L6-v2, ONNX)",
         hypothesis="A 6-layer transformer bi-encoder (22.7M params, ONNX "
                    "CPUExecutionProvider, mean pooling, corpus mean-centering, "
                    "split-conformal threshold) beats Arm C on paraphrase "
                    "recall@1, and the recall it buys is worth its ~13 ms query "
                    "encode and 18-minute cold reindex at 100k.",
         falsified_by="Recall within one seed-std of Arm C — in which case the "
                      "static table wins on cost and the transformer is deleted. "
                      "This is the genuine falsification risk of the whole "
                      "bakeoff and the pilot says it is close (0.625 vs 0.625 "
                      "at 2,030 events).",
         null_baseline="Arm C (static embeddings) — the question is not whether "
                       "MiniLM beats lexical, it is whether it beats FREE "
                       "semantics.",
         metric="paraphrase_recall_at_fixed_abstention", budget=Budget.CPU_LONG,
         depends_on=["ME.11.0"], seeds=3,
         control="Same random-projection control as Arm C, plus a "
                 "SHUFFLED-TOKEN control: encode each event with its word order "
                 "randomised. If recall survives shuffling, the encoder is a "
                 "bag of words with extra steps and Arm C dominates it by "
                 "construction.",
         kills="If Arm D ties Arm C, every transformer encoder is removed from "
               "the memory path and the 90 MB of weights, the onnxruntime "
               "dependency and the 18-minute reindex go with it.",
         notes="Measured: 13.4 ms/query (fp32), 93 docs/s, 1073 s to index 100k. "
               "int8-arm64 dynamic quantization made it SLOWER (17.8 ms) — this "
               "Neoverse-N1 has asimddp but NOT i8mm; int8 is a disk win, not a "
               "speed win. Report both. bge-small-en-v1.5 is a within-arm "
               "variant WITH its query prefix, but note its compressed cosine "
               "band (real 0.617 vs fabricated 0.595) makes it the worst arm "
               "for abstention despite the best BEIR score."),

    Spec("ME.11.E", 2, "Arm E — weighted hybrid, calibrated not assumed",
         hypothesis="Fusing Arm B's lexical scores with the best dense arm's, "
                    "using theoretical-min-max normalisation and a convex "
                    "weight w fit on the CALIBRATION split, beats both parents "
                    "on paraphrase recall@1 AND improves certified abstention, "
                    "because lexical overlap is most informative exactly where "
                    "the dense score is least trustworthy.",
         falsified_by="No gain over the better parent, OR — the specific risk — "
                      "fusion DEGRADING recall, which unweighted RRF already "
                      "did in the pilot (0.375 vs 0.625/0.750).",
         null_baseline="Unweighted RRF at k=60, the default everyone ships. It "
                       "is the null precisely because it is the popular choice "
                       "and it LOST here; beating it is the arm's minimum duty.",
         metric="paraphrase_recall_at_fixed_abstention", budget=Budget.CPU,
         depends_on=["ME.11.0"], seeds=3,
         control="Fit w on the calibration split, then evaluate with w=0 and "
                 "w=1 (each parent alone). If the fitted w lands within noise of "
                 "0 or 1, the hybrid is one parent wearing a costume and must be "
                 "reported as such rather than as a third method.",
         notes="Min-max normalisation is FORBIDDEN here: it forces max=1 for "
               "every query, destroying the absolute-similarity magnitude that "
               "is our only working abstention signal. Use TMM (Bruch et al., "
               "arXiv:2210.11934). The abstention decision is taken on the "
               "DENSE score unless the fused score measurably separates better."),

    Spec("ME.11.F", 2, "Arm F — cascade: cheap recall, cross-encoder rerank, cheap abstention",
         hypothesis="Arm C retrieves top-50 (pilot recall@10 was 1.000, so the "
                    "answer is present), a 22.7M cross-encoder (ms-marco-"
                    "MiniLM-L-6-v2, ONNX int8) reranks them, and the ABSTENTION "
                    "decision stays with Arm C's calibrated first-stage score. "
                    "This yields the highest paraphrase recall of any arm at a "
                    "latency the live agent can still pay.",
         falsified_by="Recall gain over Arm C <0.10, OR mean latency at 100k "
                      "events >250 ms, OR the reranker changing the abstention "
                      "decision at all (it must not — see control).",
         null_baseline="Arm C alone (the cascade's own first stage). The "
                       "reranker must earn its 330 ms.",
         metric="paraphrase_recall_at_fixed_abstention", budget=Budget.CPU_LONG,
         depends_on=["ME.11.0"], seeds=3,
         control="ABSTENTION MUST BE UNCHANGED by reranking. Measured pilot: "
                 "the cross-encoder's own scores do NOT separate real from "
                 "fabricated cues (real-min -9.06 BELOW fabricated-max -7.78), "
                 "so any pipeline that lets the reranker decide whether to "
                 "answer is buying recall with confabulation. The test asserts "
                 "the abstention decision is byte-identical to Arm C's on every "
                 "query, and FAILS the arm if it is not.",
         kills="If Arm F wins on recall but breaks the 250 ms budget, it is "
               "recorded as the OFFLINE-only retriever (reflection generation, "
               "ME.3) and Arm C or E ships in the live loop. Two answers is an "
               "acceptable outcome; a slow live loop is not.",
         notes="Measured rerank of 20 candidates: 516 ms fp32, 329 ms int8. At "
               "top-50 expect ~800 ms, so the arm as specified will likely "
               "BREACH its own 250 ms gate and must be run at top-10 (~165 ms) "
               "too. Report the recall/latency curve over k in {10,20,50}, not "
               "one point. Pilot cascade p@1 was 0.875 — the only configuration "
               "that cleared ME.11's 0.80 hypothesis."),

    # ── UNIFIED BRAIN: the binding evidence ladder ──────────────────────
    # From docs/research/UNIFIED_BRAIN_BAKEOFF.md (agent, 2026-08-09). Two
    # findings reframed this family:
    #  (1) UB.1 was parented UB.1 -> T4.01 -> T3.02 -> T2.01(FAIL), so the
    #      project's NAMESAKE claim was unreachable behind a locomotion
    #      failure. Binding is a PERCEPTION claim -- supervised probes, no
    #      policy, no control loop -- so these parent onto PG/T1 instead.
    #  (2) D1's evidence says nothing about binding: flat locomotion is the one
    #      task where proprioception is SUFFICIENT, so a task where fusion
    #      cannot help is not evidence about fusion either way. UB.16 states
    #      the trunk->readout->controller contract so both D1 outcomes work.
    # Three measurement sharpenings worth knowing before reading these: a
    # PLACEBO modality (matched noise) supplies the empirical null for
    # "decorative"; cross-episode SWAP replaces zeroing as the ablation
    # primitive (destroys correspondence, preserves marginals); and the synergy
    # null is the unimodal LATE ENSEMBLE, which cannot synergise by
    # construction -- beating the best single modality is not synergy.

    # ── FIXTURES for the binding test ───────────────────────────────────

    Spec("PG.6", 2, "The playground has eyes, and they resolve what the test needs",
         hypothesis="An egocentric camera in the playground MJCF renders frames "
                    "from which a linear probe recovers object RADIUS (R^2>=0.8) "
                    "and BEARING (median error <=5 deg) for objects in FOV.",
         falsified_by="Radius or bearing unrecoverable at the chosen resolution "
                      "— then vision cannot carry HNS's identity->position "
                      "channel and UB.9 would measure nothing.",
         null_baseline="Probe on a shuffled-frame/label pairing; probe on a "
                       "constant grey frame.",
         metric="radius_r2_x_bearing_error", budget=Budget.CPU_LONG,
         depends_on=["PG.1"], seeds=3,
         control="Objects OUTSIDE the FOV must be unrecoverable — else the probe "
                 "is reading episode identity, not the image.",
         kills="Any visual claim in UB.9/UB.10 at this resolution. Escalate "
               "resolution or move vision to a frozen tower with cached "
               "embeddings before proceeding.",
         notes="playground.py:217-243 emits no <camera>. This spec adds one and "
               "certifies it. Render on CPU via MUJOCO_GL=osmesa; only ~500 "
               "distinct layouts are needed because HNS reuses layouts across "
               "episodes."),

    Spec("PG.7", 2, "The heard-not-seen fixture leaks nothing but the intended bit",
         hypothesis="In the HNS scene the two candidates are acoustically "
                    "indistinguishable except by modal fundamental: identical "
                    "pan (<1e-6), identical listener distance (<1e-3 m), "
                    "matched impact amplitude, and the candidate (not the "
                    "striker or floor) is the voiced geom on 100% of events.",
         falsified_by="Any leak: an audio-only probe over band energies, "
                      "amplitude and pan classifies which object fell above "
                      "chance+3%.",
         null_baseline="Chance (0.5) for the audio-only probe.",
         metric="audio_only_leak_margin", budget=Budget.CPU,
         depends_on=["PG.5"], seeds=3,
         control="A DELIBERATELY UNBALANCED variant (unequal mass, so amplitude "
                 "tracks size) must be classified WELL above chance by the same "
                 "probe — else the leak detector is blind and its null result "
                 "is worthless.",
         kills="UB.9. A binding test built on a leaky fixture measures the leak.",
         notes="Closes, in order, the seven leaks tabulated in "
               "docs/research/UNIFIED_BRAIN_BAKEOFF.md section 3.2. PG.5's "
               "circularity guard is the precedent: ground truth is computed in "
               "this file's own trig, never from the synth's labels."),

    # ── THE BINDING TEST ────────────────────────────────────────────────

    Spec("UB.9", 4, "Heard, not seen: the task that is impossible without fusion",
         hypothesis="On a scene where audio gives object IDENTITY (modal "
                    "fundamental) but not position, and a pre-event frame gives "
                    "position but not which object fell, the fused model "
                    "identifies the fallen object well above chance (>=0.75 "
                    "mean over 3 seeds, lower bootstrap CI > 0.5).",
         falsified_by="Fused accuracy indistinguishable from 0.5, OR "
                      "indistinguishable from the unimodal late ensemble — "
                      "either way nothing was bound.",
         null_baseline="Three nulls, all at chance BY CONSTRUCTION and all "
                       "measured anyway: (i) audio-only (pan is identical for "
                       "mirrored azimuths, ContactAudio.py:26), (ii) "
                       "vision-only (the frame predates the event), (iii) the "
                       "UNIMODAL LATE ENSEMBLE of (i) and (ii) — the arm that "
                       "is structurally incapable of synergy.",
         metric="hns_accuracy_over_ensemble", budget=Budget.CPU_LONG,
         depends_on=["PG.6", "PG.7", "T1.06"], seeds=3,
         control="SWAP-FLIP: re-render the frame with the two candidates' radii "
                 "exchanged between positions, audio untouched. The correct "
                 "answer flips, so the prediction MUST flip on >=80% of "
                 "previously-correct trials. Also: spectrum-flattened audio "
                 "must fall to chance, and PAN-SHUFFLED audio must NOT change "
                 "anything (pan is uninformative here; sensitivity to it means "
                 "a leak).",
         kills="The sentence 'his senses work in unison'. This is the smallest "
               "experiment that could establish it and it costs no GPU; if it "
               "fails, no larger experiment rescues the claim.",
         notes="I(audio;Y)=0, I(vision;Y)=0, I(audio,vision;Y)=1 bit — physical "
               "XOR, one bit of PURE synergy (PID framework, arXiv:2302.12247). "
               "Proprioception, Jack's dominant modality, is uninformative here "
               "by design, which is precisely why collapse cannot hide."),

    Spec("UB.15", 4, "Heard, not seen — embodied",
         hypothesis="Jack turns toward and reaches the object he heard fall but "
                    "did not see, above the 0.5 bearing-sign chance rate.",
         falsified_by="Reach target at chance, or unchanged when audio is muted.",
         null_baseline="Audio-muted policy; vision-frozen policy; the UB.9 "
                       "discriminative ceiling (the gap is the control cost).",
         metric="embodied_hns_success", budget=Budget.GPU, seeds=3,
         depends_on=["UB.9", "T2.02"],
         control="Left/right channel swap must invert the turn direction. A "
                 "500 ms audio lag must degrade timing but not identity — the "
                 "two channels fail differently, which is itself evidence they "
                 "are separately read.",
         notes="Deliberately the ONLY binding spec that depends on locomotion. "
               "Everything else in this block is falsifiable without a "
               "controller, so decision D1 cannot block the unison claim."),

    # ── THE BAKEOFF ─────────────────────────────────────────────────────

    Spec("UB.10", 4, "Fusion bakeoff: six arms, matched params, matched steps",
         hypothesis="At matched trainable parameters (+-5%), matched tokens per "
                    "modality, matched optimisation steps and matched data "
                    "order, at least one shared-computation arm beats the "
                    "late-concat null on the binding battery, and the ranking "
                    "is stable across 3 paired seeds.",
         falsified_by="A0 (late concat) ties the best arm everywhere — then at "
                      "this scale 'one brain' buys nothing over bolt-on "
                      "encoders and GOAL.md's architecture claim must be "
                      "restated. Report it; do not re-run until it looks "
                      "better.",
         null_baseline="A0 = per-modality encoders -> pool to one vector each "
                       "-> concat -> head ('concatenate and pray'). Plus the "
                       "UNIMODAL LATE ENSEMBLE computed for every arm.",
         metric="arm_ranking_x_synergy_gap", budget=Budget.GPU, seeds=3,
         depends_on=["UB.9", "T2.00"],
         control="Every arm must FAIL the cross-episode SWAP ablation on at "
                 "least one sense (i.e. swapping a sense's stream between "
                 "episodes must hurt). An arm that is invariant to swapping "
                 "every sense has learned a marginal, not a correspondence, and "
                 "its score on the battery is uninterpretable.",
         kills="Five of six architectures. The survivor is the trunk Jack "
               "ships; the rest are deleted, not kept 'for later'.",
         notes="ARMS. A0 late-concat null. A1 shared token trunk (multi-token "
               "per modality, modality-ID embeddings, readout tokens; "
               "arXiv:2205.06175, 2405.12213, 2409.20537). A2 = A1 + modality "
               "dropout with learned [MISSING-m] tokens (arXiv:2410.03010, "
               "2201.01763). A3 = A2 + cross-modal masked prediction, "
               "cross-signal not joint (arXiv:2311.00924, 2410.16424, "
               "2607.13522). A4 = A2 + contrastive alignment with "
               "state-proximity positives (arXiv:2510.01711, 2303.15343) - "
               "NOT episode-identity positives, which are false negatives on "
               "synchronous streams. A5 = per-modality experts + learned router "
               "(arXiv:2509.23468), the credible non-trunk alternative; if A5 "
               "wins, 'one brain' is the wrong shape and we say so. "
               "A3 and A4 are parallel, not cumulative, so architecture and "
               "objective are separated. TOKEN BUDGET IS EQUALISED ACROSS ARMS "
               "or this measures token counts (arXiv:2601.16667). "
               "PAIRED bootstrap CIs and IQM per arXiv:2108.13264 - unpaired "
               "3-seed architecture comparisons resolve nothing at this budget."),

    # ── THE STANDING AUDIT ──────────────────────────────────────────────

    Spec("UB.11", 4, "The modality ablation matrix (standing)",
         hypothesis="On the tasks x senses matrix, every sense shows a "
                    "degradation significantly above the PLACEBO column under "
                    "at least the cross-episode SWAP perturbation; no sense has "
                    "an all-null row of cells.",
         falsified_by="Any sense whose four perturbations are all "
                      "indistinguishable from the placebo modality — it is "
                      "decorative and loses its parameters (Tier-3 rule).",
         null_baseline="A PLACEBO MODALITY: pure noise, identical token count, "
                       "encoder capacity and dropout rate, wired in like a real "
                       "sense. Its column IS the empirical null distribution "
                       "for 'decorative', re-estimated every run.",
         metric="min_sense_margin_over_placebo", budget=Budget.GPU, seeds=3,
         depends_on=["UB.10"],
         control="TWO controls in opposite directions. (a) The placebo column "
                 "must be SMALL: a large placebo Delta means the procedure "
                 "measures off-manifold shock, not information, and every other "
                 "column is uninterpretable. (b) With proprioception replaced "
                 "by its [MISSING] token, a dropout-trained model must still "
                 "briefly stand using vision - vestibular substitution.",
         kills="Any encoder whose column is placebo-indistinguishable. Deletion "
               "is the default action, not a discussion.",
         notes="STANDING SPEC - re-runs on every architecture change, forever, "
               "like ME.5 at every decade of store growth. FOUR perturbations "
               "per cell: zero (off-manifold), matched noise (marginals kept), "
               "within-episode time-shuffle (temporal binding destroyed), "
               "CROSS-EPISODE SWAP (correspondence destroyed, everything else "
               "kept). Swap is the primitive: it is the only one that isolates "
               "correspondence, which is what binding means. Ablation uses the "
               "learned [MISSING-m] token, never zeros, or the matrix measures "
               "brittleness (arXiv:2410.03010). Logged alongside: per-layer "
               "cross-modal attention mass (arXiv:2410.16424) and the learned "
               "binary modality mask (arXiv:2209.07682) - both free, both "
               "necessary-not-sufficient, both red flags rather than claims."),

    Spec("UB.12", 4, "Synergy, not redundancy: beating the unimodal ensemble",
         hypothesis="On every task in the battery the fused model beats the "
                    "UNIMODAL LATE ENSEMBLE (independently trained per-sense "
                    "models, predictions averaged), paired across seeds, with "
                    "a bootstrap CI on the paired difference excluding zero.",
         falsified_by="Fusion >= best single modality but <= the ensemble on "
                      "every task: the model is exploiting redundancy and "
                      "uniqueness, and computes nothing jointly. This is the "
                      "most likely honest outcome and it must be reportable.",
         null_baseline="max_m U_m (the trivial bar) AND the ensemble E (the "
                       "real bar). Beating max_m U_m is not evidence of fusion.",
         metric="synergy_gap", budget=Budget.GPU_SHORT, seeds=3,
         depends_on=["UB.10"],
         control="On UB.9 (pure synergy, all unimodal channels at chance) the "
                 "ensemble MUST sit at chance. An ensemble above chance there "
                 "proves the fixture leaks and PG.7 passed wrongly.",
         notes="The operational definition of 'one brain': the late ensemble is "
               "structurally incapable of synergy because no parameter ever "
               "sees two modalities jointly, so F > E is joint computation by "
               "construction. Costs 5 tiny models per task; compute it for "
               "every arm, every task, forever. Frame results as PID "
               "redundancy/uniqueness/synergy (arXiv:2302.12247)."),

    Spec("UB.13", 4, "Cross-modal retrieval: the gate, never the claim",
         hypothesis="Given a contact-audio window, the matching visual clip is "
                    "retrieved above chance (R@1 and R@10 vs a candidate set of "
                    "known size), including against HARD negatives: the same "
                    "episode at +-0.5 s, and a different object at the same "
                    "instant.",
         falsified_by="At-chance retrieval against hard negatives while easy "
                      "retrieval succeeds — then the model matched onset "
                      "synchrony, not content.",
         null_baseline="Chance = 1/N for the actual candidate-set size, stated "
                       "before the run; plus a retriever over event ONSET TIMES "
                       "only, which is the synchrony-shortcut baseline.",
         metric="hard_negative_recall_at_1", budget=Budget.GPU_SHORT, seeds=3,
         depends_on=["UB.10"],
         control="Time-offset negatives must be harder than random negatives. "
                 "If they are equally easy, the candidate set is trivial.",
         kills="Nothing on its own. This spec exists so that a NULL result on "
               "the contrastive arm (A4) is interpretable: without it, 'A4 did "
               "not help control' cannot be distinguished from 'A4's loss never "
               "trained'. Retrieval is necessary, never sufficient "
               "(arXiv:2603.19233: encoded is not used)."),

    Spec("UB.14", 4, "Cross-modal prediction, against the null that usually wins",
         hypothesis="Masked touch is predicted from vision+proprioception "
                    "better than from proprioception ALONE, and better than the "
                    "unconditional mean, at matched capacity.",
         falsified_by="Proprio-only matches vision+proprio: foot contact is "
                      "inferable from joint torques, so vision adds nothing "
                      "here. An HONEST and likely outcome that must be "
                      "reported, not retried.",
         null_baseline="Unconditional mean (the floor) AND a proprio-only "
                       "predictor of equal capacity (the real bar).",
         metric="touch_r2_over_proprio_only", budget=Budget.CPU_LONG, seeds=3,
         depends_on=["PG.1"],
         control="Touch-from-SHUFFLED-vision must collapse to the "
                 "unconditional mean — else the head ignores its vision input "
                 "and the conditioning is decorative.",
         kills="The vision->touch masked objective in arm A3, if vision adds "
               "nothing over proprio. Run this BEFORE the bakeoff: it costs CPU "
               "minutes and can delete an arm's justification.",
         notes="Calibrate expectations from Kepler-Encoder (arXiv:2607.13522): "
               "fused-vs-vision-only force R^2 of 0.049/-0.001/0.187 across "
               "three robots, one of them NEGATIVE, p<=0.012. Real, clean, "
               "small. A bakeoff expecting a large effect has mis-specified "
               "its success criterion."),

    Spec("UB.16", 4, "Sensory information reaches the controller (D1-agnostic)",
         hypothesis="Zeroing the trunk's percept vector z degrades tasks that "
                    "require non-proprioceptive information, and does NOT "
                    "degrade flat-ground locomotion.",
         falsified_by="z-ablation changes nothing anywhere (the trunk is "
                      "decorative in the control path) OR it degrades flat "
                      "walking too (z is smuggling proprioception the "
                      "controller already has, so the comparison in D1 was "
                      "never about perception).",
         null_baseline="Controller on raw proprioception alone; controller with "
                       "z replaced by its batch mean.",
         metric="z_channel_asymmetry", budget=Budget.GPU, seeds=3,
         depends_on=["UB.11", "T2.02"],
         control="A SHUFFLED-z controller (z drawn from another episode) must "
                 "match the zeroed-z controller. If shuffled-z is WORSE than "
                 "zeroed-z, the controller is reading correspondence, which is "
                 "a stronger result than the hypothesis claims.",
         notes="The asymmetry IS the test, and it holds under either D1 "
               "outcome. If D1 removes the trunk from the control path, z is "
               "the entire sensory channel and this spec certifies it. If the "
               "trunk stays end-to-end, z is the readout-token bundle and the "
               "same measurement applies. Locomotion is the task where "
               "proprioception is SUFFICIENT, so it is the wrong task to judge "
               "a binder by - which is why 'no degradation on flat walking' is "
               "a PASS condition here, not a failure."),

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
