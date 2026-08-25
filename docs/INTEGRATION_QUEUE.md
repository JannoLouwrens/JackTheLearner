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

## OWNER DIRECTIVE, 2026-08-10 — FAST AND SLOW: research owed on a family already registered

**The owner's words:** *"but Will he have fast and slow brain like a human?"* →
*"all I know is we must figure that out right and it must still be connected but
slightly different purposes? it must be in the research and tests"*

**What was done immediately, and what is still owed.** DP.00–DP.03 are
REGISTERED (`registry_expansion.py`, ladder 147 → 151) and GOAL.md carries the
directive as a named section. That is the *tests* half. The *research* half is
outstanding: **there is no `docs/research/DUAL_PROCESS.md`**, so these four specs
are a first cut written from the directive plus this repo's existing evidence,
not from a literature sweep. Under this file's own empty-queue rule that makes
the research the next correct iteration, not an optional extra.

**The specs, so the researcher knows what must survive contact:**

    DP.00  tier 2  CPU       This world rewards looking ahead at all
    DP.01  tier 3  CPU_LONG  Practice moves a behaviour off the deliberative path
    DP.02  tier 3  CPU_LONG  Connected, not two brains: the substrate is shared
    DP.03  tier 4  CPU_LONG  Deliberation is spent where it pays

**Questions the research must answer, each of which can invalidate a spec above.
Treat "the spec is wrong" as the expected outcome, not the failure case:**

1. **Is the habitisation measurement right?** DP.01 uses a difference of
   differences (practised minus unpractised, early minus late) and ablates by
   disabling rollouts rather than zeroing weights. Does the model-based RL
   literature have a better-validated operationalisation — and does the
   devaluation paradigm from animal learning (the actual experimental test for
   habit: does the animal still work for a reward it no longer wants?) transfer
   to an agent? Devaluation may be a *stronger* test than ablation and it is not
   currently in the spec.
2. **Can DP.02's lesion instrument actually discriminate?** It requires a
   shared-trunk lesion to degrade both modes together while a
   deliberately-separated two-tower control dissociates. What lesion magnitude
   and what layer? Is there a published measure of representational sharing
   (CKA, linear-probe transfer between heads, gradient-conflict) that is less
   blunt than lesioning and would work as a corroborating second instrument? A
   single blunt instrument on the project's most load-bearing connectedness
   claim is thin.
3. **Does the matched-rate random gate null exist in this literature?** DP.03
   asserts it is routinely omitted. Verify that claim or delete it — this repo
   does not get to make sweeping claims about a literature it has not read.
   Report FLOPs-per-decision conventions used by adaptive-computation work
   (ACT, PonderNet, MoE routing, early-exit) so DP.03's compute accounting is
   comparable rather than invented here.
4. **What does the alternative look like?** If the evidence favours two
   physically separate pathways in real brains (basal ganglia vs prefrontal is
   the obvious challenge to "one substrate"), say so plainly and put the
   counterargument in front of the owner via DECISIONS_NEEDED.md. The directive
   says connected; the ladder's job is to find out whether connected survives,
   and a research doc that only confirms the decree is worthless.

**CROSS-CHECK BEFORE REGISTERING ANY REVISION** — per this file's step 1, the
terms to grep are: `LEARNING_CORE.md` (LC.04 is already arbitrating reactive
against world-model arms and may have settled part of DP.00), `MEMORY.md` and
`ME.7`/`ME.10` (the fast/slow LEARNING axis — do NOT let it merge with fast/slow
ACTING; GOAL.md now names all three axes precisely to prevent that),
`NEEDS_AND_DEATH` (DP.00's claim that a survival world rewards lookahead is
*expected* and therefore exactly the kind of unverified expectation this repo
has been burned by), and `FROZEN_VS_PLASTIC.md` (a separate deliberative tower
is a frozen-tower-shaped proposal and the plastic-only decree constrains it).

**Status:** REGISTERED, RESEARCH OWED. Registered 2026-08-10. Do not treat the
four specs as settled; a revision that strengthens them is the point.

## DONE — PS.01 v2: the probe policy, not the constants (from PS.01 attempt 2, 2026-08-10)

> **STATUS: DONE. Registered, implemented and run 2026-08-10T08:33 — PS.01
> attempt 3 = PASS at 3 seeds, 864.8 s CPU.** Pre-registration committed unrun
> in `ad55a31`; result in the commit that follows it. All four changes shipped
> as written; protocol steps 1-5 complete (step 6 does not apply — no bakeoff).
> LC.03 is now RUNNABLE and LC.04-LC.06 + OP.01 sit behind it alone.
>
>     drive_dynamic_range   0.777   (gate 0.30)   spread_e 0.778 / spread_i 0.790
>     n_damaging            32.7    (gate >= 5)   n_rest_decisions 2349 (gate >= 100)
>     fall_cost_med         0.161   held out, [0.10, 0.20], 5 fresh runs
>     statue_death_s        600.2 s (gate < 720)  = 1/b exactly, now OBSERVED
>     forager e_min         0.841   at duty 0.216 and 28 floor items eaten
>     P_bar(1)              1407.9 W (unit (a) measured 1434.8 +- 22.2 on
>                                    HELD-OUT seeds 3-5; this is seeds 0-2)
>
> `spread_i` moved **2.96e-5 -> 0.790** on the same integrator and the same
> constants: the variable was never inert, it was unmeasured. The forager
> fixture's drain (2.263e-3 /s) lands under the floor supply (2.392e-3 /s), so
> unit (a)'s C2 is now verified on the shipped path rather than in arithmetic.


**Why this is top.** PS.01 is the only thing between the ladder and LC.03–LC.06,
i.e. between it and the arbitration that decides HOW JACK LEARNS. Unit (a) — the
energy re-derivation — is DONE and shipped (`drives.py`, `PURPOSE_AND_
SCAFFOLDING.md` §2.2–2.3, `experiments/calibrations/ps01_energy.py`, criterion
committed unrun in `92aae6f`). Attempt 2 moved `spread_e` **0.145 → 0.746**
against a 0.30 gate. What remains cannot be reached by any constant, and the
journal (2026-08-10 05:30) directed it here rather than into an in-place
registry edit. **This is unit (b).**

**The three surviving failures are one defect: the probe cannot produce the
events the gates are about.**

| clause | measured, attempt 2 | why no constant fixes it |
|---|---|---|
| `spread_i ≥ 0.30` | **2.96e-5** | A random policy never climbs, so it never falls from height; it never holds still (`rest_frac` 5.3e-5), so `ρ` never heals. 856 contact onsets, **1.7** above `J₀`. The *same* integrator scored 0.161 on a held-out platform fall. The channel is live; the probe cannot get to it. |
| `ok_random_survives` (`e_min > 0`) | **0.0** | A random policy is not a forager — it ate 1.0 items in 600 s. It acts at duty 1.0 and cannot navigate, so it starves under any supply the world can carry. Demanding that flailing beat resting is demanding `κ = 0`. |
| `ok_statue_starves` (`e_min ≤ 0`) | **4.35e-14** | The statue dies at `t = 1/b` = **600 s** and the observation window is **exactly 600 s**. The control's pre-registered failure is scheduled at the last sample and misses by float. |

**PROTOCOL STEP 1 — CROSS-CHECK, run 2026-08-10** over `docs/research/*.md` +
`LESSONS.md` for `statue|dark room|do-nothing|dynamic range|probe`:
`NEEDS_AND_DEATH.md:1196` and PS §5 **G-B** ("the dark room") are the governing
prior art and both **REINFORCE**: G-B provision 1 is *"basal drain exceeds
nothing a motionless agent can earn, so the statue starves to the weakness
floor — verified as a spec, not asserted as a design intention"*, which is
exactly the clause below, and G-B provision 2 makes `C-STATUE` mandatory. No
refutation anywhere. `LESSONS.md` supplies the two rules this redesign is built
from ("a probe policy that cannot produce the event cannot measure the
variable"; "an assertion made against a saturated quantity cannot fail").

**THE REDESIGN — strengthen only; attempt 1 and 2 stay in the ledger's history
(T1.02 precedent).** Three changes, each naming the event it requires next to
the threshold, per the LESSONS rule that motivated it:

1. **The integrity range is measured over a MIXED probe, not a random one.** A
   life of random-policy decisions with scripted drop-spawn segments (the fall
   regime `ps_01_drive_calibration._params(fall=True)` already implements) and
   scripted rest segments. This is a FIXTURE probe of the integrator, not a
   claim about a policy, and the spec must say so — it certifies that `i` has
   usable range over behaviours the world ADMITS, which is what "the drive is a
   control problem" means. Gate the required events, not just the range:
   `n_damaging ≥ 5` and `n_rest_decisions ≥ 100` become PASS conditions, so a
   probe that failed to exercise the variable is a red entry rather than a
   confident 2.96e-5.
2. **The domination clause compares the statue against a FORAGER FIXTURE, not
   against a random policy.** A scripted eater — placed at a food geom,
   consuming on respawn, acting at the derived duty cycle `D* = 0.217` — run
   through the real `DriveLayer`. It must not starve; the statue must. This
   verifies through the shipped path what unit (a) established in rates (C1–C3),
   and it is the honest form of G-B's question: *is the dark room beaten by some
   behaviour this world admits*, not *is it beaten by flailing*. Note the
   fixture needs no locomotion controller, which is why it is affordable today.
3. **The observation window must strictly contain the control's death.**
   `N_DECISIONS` 3,000 → **4,500** (900 s = 1.5 × `1/b`), and add
   `statue_death_s < 0.8 × horizon` as a gate. A control designed to fail at the
   boundary of the window cannot be observed failing.

4. **Report every drain rate NEXT TO the regime it was measured in.** Attempt 1
   recorded `mean_power_w = 293` and `frac_e_zero = 0.848` in the same entry and
   nobody joined them: the power was a *starving* body's, because `gear_scale =
   0.4 + 0.6·min(e, i)` sat at 0.4 for 85% of that run. §2.3 then exonerated `κ`
   on that number. v2 must record `mean_power_w_full_strength` — the same
   measurement with `e = i` pinned at 1 — beside it, so the confound is a field
   in the record rather than a thing a reader has to notice. One extra rollout.

Registering iteration: follow the protocol from step 2. `metric` stays
`drive_dynamic_range`; note that it is `min(spread_e, spread_i)` and is
therefore entirely gated on clause 1 today.

**Cost:** CPU. Attempt 2 was 428 s at 3 seeds; the longer horizon and the extra
fixture put v2 near ~15 min. No GPU, so it runs beside anything.

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

> **STATUS: REGISTERED 2026-08-25 (`ed2d969`), verbatim** — only a `COVERS:`
> marker appended (the declaration idiom postdates this draft). The dangling
> `LG.01` dependency now resolves: it is the probe-set certification fixture,
> registered in the same commit. See the LANGUAGE_GROUNDING.md row above.

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

> **STATUS: REGISTERED 2026-08-25 (`ed2d969`), verbatim** — only a `COVERS:`
> marker appended. See the LANGUAGE_GROUNDING.md row above.

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

## W0.BAL: the rover topples, and LC.03 cannot mean anything until it is decided by bakeoff

> **STATUS: PROCESSED 2026-08-21 (builder) — ESCALATED to DECISIONS_NEEDED.md
> D9, not registered, per protocol step 1.** The cross-check found the entry
> superseded in part and owner-gated in whole, all post-dating it: LC.03 was
> redesigned 08-13/08-20 with rig gates that carry its meaning on the as-built
> body (the title's premise is dead — its registered run is in flight, not
> waiting); D8 (08-14) measured this body's actuation authority in detail and
> established that body changes are world-contract changes on the owner's
> desk; and every spec a body fix would serve (LT.* unregistered, T5.01)
> is blocked behind T2.01/D1, so no bakeoff outcome is adoptable by this desk
> today. The pre-registered arms/metric/null/kill survive VERBATIM in D9 and
> the bakeoff runs unchanged the day the owner orders it (D9 option b) or a
> ladder-branch spec becomes unblocked. Nothing here was decided by argument:
> no arm was picked; the pick was routed to the authority that can adopt it.

**Raised by a MEASUREMENT, 2026-08-09, not by an argument.** LC.02 built the
climber-rover exactly as `CURIOSITY_BAKEOFF.md` §2.3 specifies and ran it. A
30 kg capsule torso standing on a 0.09 m spherical foot is an inverted pendulum
with no balance mechanism, so under random action it **topples within ~20
decisions and then slides on its side**: `upright_cos` goes to −0.041 and stays
there, on all three seeds. LC.02's own claim is unaffected (it is a wall-clock
floor and a body on its side steps at the same speed), and the number is
recorded in its ledger entry rather than hidden. But the arms' `lift` slides
travel along the BODY's z axis, so a toppled rover cannot raise a hand, and
`CURIOSITY_BAKEOFF.md`'s pilot table — zero engaged ladder attempts in 9,000
random decisions — is consistent with a rover that was on its side for most of
them. **LC.03 asks whether an arm learns to survive; on a body that spends its
life prone, a null result would measure the rig.** This is PG.8's finding one
level down: the room now has somebody in it, and he is lying on the floor.

DO NOT pick a fix by argument — that is law 3, and there are at least three
plausible answers, each with a real cost:

    A  accept it        the rover is a slider; ladder specs move to a body that
                        can stand. Cheapest; concedes the ladder test.
    B  gated righting   a bounded restoring torque on the torso, gated on floor
                        contact EXACTLY as the drive is, so it contributes
                        nothing once the feet leave the ground and every metre
                        of ladder height is still earned by the arms. Mirrors
                        the cheat that is already declared; adds a mechanism.
    C  wide base        replace the spherical foot with a plinth and lower the
                        COM until the rig is statically stable under the 600 N
                        drive. No new mechanism; changes PG.3's inherited
                        geometry, so the inheritance-by-construction claim needs
                        re-checking.

METRIC, pre-registerable today and readable without any learning:
`upright_frac` (fraction of decisions with `upright_cos >= 0.7`) and
`hand_reach_z_max` (highest world z any hand geom attains) under an identical
uniform-random policy, 3 seeds x 500 decisions, all three arms in the same
mutated worlds. NULL: the rover as built (arm A) — measured today at
`upright_cos` −0.041. KILL: if no arm reaches a hand above the first rung,
none of them fixes the thing the fix is for, and the ladder branch moves to a
different body rather than to a better rig.

Cross-check status (protocol step 1): grepped `CURIOSITY_BAKEOFF.md`,
`LEARNING_CORE.md`, `PURPOSE_AND_SCAFFOLDING.md` and `NEEDS_AND_DEATH.md` —
none of them states a balance requirement or a stability measurement for this
body, so nothing here is refuted and nothing is duplicated. That silence is
itself the gap.

## FIRST: FINISH THE LC BAKEOFF (LC.02-LC.06) — unblocked, zero GPU

**LC.02 PASS 2026-08-09** — all five admissible arms clear the 5.0 sim-s/real-s
floor, at train_ratio **0.25** (0.125 for `wm-latent`). Those ratios are now
committed and LC.03 must use them. Note the size of the correction they carry:
`LEARNING_CORE.md` §5.1 derived "admits train_ratio up to ~4" from physics and
core costs measured SEPARATELY, and the composed world runs 16x lower. §5.1's
derivation should be corrected in place rather than left standing.

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

## GEN.00 — THE FINAL EXAM (owner, 2026-08-09): buildable EARLY, not far-future

Owner asked whether "we build a learner capable of learning himself" can be
tested concretely. It can, and nothing tested it: all 136 specs ask "did he
learn X" for an X we designed. GEN.00 (docs/GENERALITY.md) is the direct test
— a SEALED challenge, hash-committed before training, designed by someone
blind to the curriculum, consequential to his needs and novel in mechanism.
Null: learning frozen must NOT improve. Control: a sham-novelty challenge must
NOT count. Unlike the rest of GENERALITY.md this is NOT far-future — it runs
in the playground as soon as needs + a learning core exist, and it should be
re-run forever with a fresh sealed challenge each time.
PREREQUISITE, and it is a process one rather than a code one: the sealing.
Whoever designs a challenge must not have seen the training specs. The owner
is the natural sealer; a blind agent is the fallback.

## Queue (top = next)

| research doc | specs | status |
|---|---|---|
| LEARNING_CORE.md | LC.00–LC.06 + PS.01 | REGISTERED by the builder autonomously, 2026-08-09 (registry 128→136; cross-check clean — NEEDS_AND_DEATH §0.2 *supports* LC.00's drive-reduction reward; W0 naming reconciled by LEARNING_CORE §5.0's contract). **LC.00 PASS** same day: 3 of 4 tabular cores beat the null ≥3σ (q_drive 9.1σ, model_vi 6.3σ, model_efe 4.1σ; q_lp 1.5σ ran but did not clear), frozen control −6.3±5.9 ≈ 0 — the world does not drift. **LC.01 PASS** 2026-08-09 (5.7 s, 3 seeds): all five §5.4 arms admitted on U1–U4; the three controls failed on their pre-registered side (unbound cross-modal finite difference exactly 0.0 on every seed, leaky private-path grad 271±44, naive-scales needs share 0.113 < 1/6). The arms now live in `experiments/cores.py` — LC.02/LC.03 import them, they are not re-implemented. Next cheapest: LC.02 (throughput floor, CPU) — but it needs the W0 env, so PS.01 (drive layer, CPU, no body) may be cheaper first. |
| NEEDS_AND_DEATH.md | NE.00–NE.09 | **REGISTERED 2026-08-24** (`20e7b29`, registry 169→179), verbatim, notes-only additions (COVERS annotations + two carried caveats). CROSS-CHECK (step 1): W.6 withdrawn by SURVIVAL_WORLD §5.0 in favour of NE.08 — reconciled, W.6 stays unregistered; **XL.01 (registered 08-19, after this doc) overlaps NE.08 and ran FAIL + power-blocked** — its B3 verdict (cannot resolve 2× at 3 seeds × 8 lives) is carried in NE.08's notes as a BINDING pre-run power calculation; §9's citation gate (Borbély ratio, 28/40 °C bounds unverified) carried in NE.01's notes — **NE.01 must not fix those constants until a citation pass closes §1.2**; drives.py is the 3-need PS suite, the 7-need integrator of §2.3 is TO BUILD. No refutation found. VERIFY (step 2): 10 new ids, no collision, all depends_on resolve. Step 4: **NE.00 PASS same day** (3 seeds, 4.7 s): all five algebra predictions confirmed — DR/CC greedy sets identical at γ∈{0.9,0.95,0.99}, telescope 4.6e-16, best discounted cycle −0.015 < 0, suicide col 11/11 / cc 1/11 / dr 0/11 (direction gated; the lost pilot's 8/11 was its parameterisation — declared in the docstring), clip cycle +0.09 vs exact 0.0, event control differs in 43–45 % of states at every γ. Next cheapest: NE.01 (CPU) — build the seven-need integrator against it, after the §1.2 citation pass. |
| PURPOSE_AND_SCAFFOLDING.md | PS.* | BLOCKED-ON-CORRECTION: PS.00(c)+PS.02 disproven by NEEDS_AND_DEATH (drive-farming cannot exist; exact VI + K&G eLife 2014 theorem). Correct, then register. NOTE: **PS.01 is already registered** (2026-08-09, with the LC family — LC.03 depends on it and LEARNING_CORE §5.6 required one commit; PS.01 is calibration, not implicated in the refutation). Do not register it twice. |
| CURIOSITY_BAKEOFF.md | LT.01–LT.09 | PENDING |
| D1_CONTROL_ARCHITECTURE.md | D1.0, T2.21 | PENDING |
| HEARING_BAKEOFF.md | HR.1–HR.8 | PENDING |
| LANGUAGE_GROUNDING.md | LG.* | **PARTIALLY REGISTERED 2026-08-25** (`ed2d969`, registry 183→187, OVERSIGHT B1(a)): the completeness check CONFIRMED the truncation — §2.2–§11 are headers with no body, §7 ("registry entries") is empty — so what was registered is exactly the owner-designed material that EXISTS: **LG.00** (anti-puppet, the GOAL.md-cited asymmetry, verbatim from this file's GAP-FILL), **LG.10** (who chooses the words, verbatim), **LG.02** (THE LIAR TEST as specified on this row: track-record divergence, mid-life swap control, attribution-stripped null), and **LG.01** (the certification fixture LG.00's depends_on named: every probe question certified lived-necessary per-question against the LLM-alone leg — Finding 1's §1.1 lesson, the only part of the truncated doc that binds a design). CROSS-CHECK (step 1): no refutation; FROZEN_VS_PLASTIC §10.8 strengthens LG.00 (RT-2 knowledge-loss backs the control clause); one prose id collision found and recorded in the registry comment — DIRECTION_AUDIT.md's "LG.00 = certification" numbering loses to GOAL.md's constitutional citation. DP.04.depends_on += LG.00 same commit, per its own notes. Step 4 (implement + run cheapest) NOT done this iteration: LG.01 is the next unit — CPU, ME.9 PASSES, SmolLM2 is cached on this box. **STILL OWED: the doc's §2.2–§11** (understanding test LG.05, grounding bakeoff arms, ordering experiment) — a research pass with citations, per the empty-queue rule; registering unwritten designs would be the disease coverage.py exists to catch. |
| DIRECTION_AUDIT.md | WP.01–04, LF.01–05, SO.01–05, PS.07, T0.17–18 (stubs) | PENDING — stubs need full Spec fields before registration |
| SURVIVAL_WORLD.md | W.1–W.7 | PENDING — CROSS-CHECK REQUIRED: W.6 overlaps NE.08 (reconciliation written in SURVIVAL_WORLD §5.0); register the reconciled pair or the ledger gets two specs testing one claim. Also carries recommendation 6b: contype/conaffinity audit of the playground, worth ~3x throughput, do before any W.* runs |
| OWNER DECISION: the owner's hands (no doc yet — empty-queue rule applies) | SO-family: provisioning channel (drop-in objects), provenance into the diary ("who left this"), anti-puppeteering limits | PENDING — approved by owner 2026-08-09; needs its research pass then specs. OWNER 2026-08-09: visualisation of needs (hunger bars, spectating) is deliberately LATE — it is a pure read layer, zero science cost. But the camera rolls from day one: NE.* implementation must log needs telemetry (a few floats/step) from the FIRST needful life, so any later viewer can replay any life. Recording is nearly free; retrofitting history is impossible |
| FROZEN_VS_PLASTIC.md §8.6 (the missing senses) | SM.01–02, TA.01–03, VO.01–02 | **REGISTERED 2026-08-10** (registry 139→146), verbatim, no threshold edited. Triggered by OVERSIGHT.md §3.2 / FOR THE BUILDER item 7, not by the queue's own order — smell, taste and voice had ZERO specs among 137 and were therefore invisible to `run next`, `run blocked`, `run status` and the Review alike. CROSS-CHECK (step 1) over docs/research/*.md + LESSONS.md for `smell|olfact|taste|gustat|voice|vocal`: no refutation. NEEDS_AND_DEATH designs the DRIVES, not these exteroceptive channels, and *supplies* the delayed illness TA.01 needs; SURVIVAL_WORLD supplies the world content; FROZEN_VS_PLASTIC §P2 (a channel absent during the early transient may never integrate) *reinforces* wiring them at W0 with content at W1. VERIFY (step 2): all 7 ids new, no prefix shadowing, every depends_on resolves (PG.1, PG.5, PG.6, UB.11, and within-family). **PAIN and TEMPERATURE deliberately NOT registered** — their designs are not free-standing (temperature is SURVIVAL_WORLD W.1/W.3, i.e. a whole survival world; pain is an open ARM inside NEEDS_AND_DEATH §2.9, explicitly *"a live question, not a settled design"*), so registering either as written would prejudge an open bakeoff. They stay ABSENT and `run senses` (T0.20) reports them so every time it is run. Step 4 (implement + run the cheapest) NOT done: **VO.01 is the next unit** — CPU, depends only on PG.5 (PASS), and the cheapest constitutional gap in the project. |
| UNIFIED_BRAIN_BAKEOFF.md | PG.6–7, UB.9–16 | REGISTERED a3129b2 2026-08-09 |
| MEMORY_RETRIEVAL_BAKEOFF.md | ME.11.0, ME.11.A–F | REGISTERED 0c1ff06 (ME.11.0 PASSING) |

## FROM A FAILED RUN, not from research — PS.01's integrity clause needs its own probe (2026-08-10)

**Scar:** PS.01 ran and FAILED (`experiments/ledger.json`, 2026-08-10). Two of
its four clauses failed for reasons that are about the PROBE, not the mechanism.
`spread_i` measured **2.4e-5** against a 0.30 gate — while the *same* integrator,
on the *same* seeds, scored a platform fall at **0.162** integrity, held out and
inside its pre-registered [0.10, 0.20] band. A random policy never climbs (so it
never falls from height: 203 contact onsets, 1.3 above `J₀`) and never holds
still (rest fraction 3.6e-5, so the healing term never fires). Both terms of the
integrity equation are unreachable from the probe the spec gates on.

This is the T1.02 situation exactly: a spec whose threshold is right and whose
*experiment* cannot test it. T1.02's precedent governs — **strengthen only, the
old version stays in the ledger's history**, and the redesign gets registered
through this queue rather than edited into the registry in place.

**Draft, to be cross-checked (step 1) against `NEEDS_AND_DEATH`,
`CURIOSITY_BAKEOFF`, `LEARNING_CORE` and `LESSONS.md` before registering:**

    Spec("PS.02", 2, "Integrity has a usable range under a probe that can reach it",
         hypothesis="Over a MIXED probe — random-policy lives interleaved with "
                    "drop-spawn lives at the platform height PS.01 already "
                    "implements — integrity's p90-p10 spread is >= 0.30, and "
                    "the spread is carried by falls and by rest, not by one "
                    "outlier: >= 5 damaging onsets and rest_frac >= 0.05 per "
                    "life.",
         falsified_by="Range still < 0.30 with the events present. Then the "
                      "healing/damage constants are wrong, not the probe, and "
                      "rho and alpha go back to calibration.",
         null_baseline="The drive integrator disabled on the same rollout — "
                       "PS.01's null, which already reads 0.0.",
         metric="integrity_dynamic_range", budget=Budget.CPU, seeds=3,
         depends_on=["PS.01"],
         control="The random-only probe PS.01 used, which MUST still measure "
                 "~0. If the mixed probe and the random probe agree, the "
                 "redesign changed nothing and PS.01's reading stands.",
         kills="The claim that integrity is a live drive at this "
               "parameterisation. It cannot kill the channel — PS.01/J2 "
               "measured that at 0.973 AUC.")

**And the required event counts are the point.** PS.01's record already carries
`n_onsets`, `n_damaging` and `rest_frac` for exactly this reason (LESSONS.md,
"a probe policy that cannot produce the event cannot measure the variable"). Any
successor spec that gates a variable's RANGE must state, next to the threshold,
which events move it and how many the run must contain.

**Do NOT register this until PS.01's energy clause is re-derived (a) —** the
two share a rollout, and re-running the probe twice is the avoidable cost.

---

## FIELD WATCH wk1 CONSUMED by the Review, 2026-08-10 (DAILY)

Source: `docs/FIELD_WATCH.md`, sweep 2026-08-10, four nominations. The scout
nominates; it never adopts. These are the dispositions. Full reasoning in
`docs/PROGRESS.md § FOR THE BUILDER`.

**N1 — ACCEPTED, and it is the one to do first.** Design the certificate
pre-gate for `UB.11` ([arXiv:2607.27017]). `UB.11` is the standing modality
ablation matrix whose verdict is *"deletion is the default action, not a
discussion"* — it has a placebo column (negative control) and **no positive
control**, so it cannot distinguish "this sense is decorative" from "this
fixture gave this sense nothing to say", and it deletes the encoder either way.
The certificate probes the raw observations first, proving a sense was
recoverable before the model is blamed for ignoring it. This is a
STRENGTHENING of an existing spec gating a DESTRUCTIVE action; register it as a
mandatory pre-gate, not an optional arm. ~5M params, <1 h on our substrate.
Take it before UB.9 results begin feeding the matrix.

**N2 — ACCEPTED.** Design bakeoff arms `A4b` (SMWM, [arXiv:2606.20104]) and
`A4c` (SIGReg/LeJEPA, [arXiv:2511.08544]) for `LEARNING_CORE` §5.4. Both would
delete A4's EMA target encoder. They enter as ARMS and are decided in the
arena — law 3, never by argument.

**N4 — ACCEPTED with a caveat that must travel with it.** The entity-collision
eval protocol ([arXiv:2605.29630]) enters `MEMORY_RETRIEVAL_BAKEOFF` §2 as an
eval-set DESIGN question. Caveat: it floors BM25 by construction, which is the
opposite discipline to our lexical-disjointness invariant. Register the design
question; do not adopt its numbers as a result.

**N3 — REJECTED.** Interoceptive precision allocation ([arXiv:2608.04232]).
Reason, one line: SYSTEM.md's *no new organ without a scar* — no failure in
this repo has been traced to uniform interoceptive precision, so there is
nothing yet for it to fix. (The scout argued against its own nomination on this
ground and was right to.) **RE-OPEN TRIGGER:** `NE.03` runs and the uniform
nine-float design underperforms its arms — then this returns with a measured
scar attached. A rejection without a re-open condition is just forgetting.

## FIELD WATCH wk2 + wk3 CONSUMED by the Review, 2026-08-13 (DAILY)

**Why two sweeps at once.** Week 2 (`2026-08-11`) landed *after* the 08-11
Review had already read the file, and the 08-12 Review died on an API 529
before it reached Part 2.5 — so wk2's nominations were never dispositioned, and
wk3 (`2026-08-12`) rewrote `FIELD_WATCH.md` over them. Both sweeps are
reconstructed here from `docs/FIELD_WATCH_LOG.md`, which is append-only and is
the reason nothing was lost. **That is the log earning its keep**: a
state-file-only scout would have had one week of work silently overwritten.
Four nominations, all four dispositioned below. The scout nominates; it never
adopts.

**wk2-N1 — ACCEPTED, and it is the one to do first of these four.** The whiff
clock → `SM.02`. Three independent 2026 groups (arXiv:2605.15938 full-text,
2605.21329, 2605.18881) converge on ONE state variable for intermittent plumes:
**time since last detection**. Our `OdourSensor` carries 12 floats — bilateral
concentration plus a one-step derivative — and **no blank-duration state at
all**, in a field the same literature measures 40–55% blank. Design (a) an
observation arm adding the blank clock, and (b) a tabular-Q CPU-minutes
reference arm, as a **pre-gate on `SM.02`**. The reason it ranks first is not
its novelty, it is the destructiveness of what it gates: `SM.02`'s kills clause
deletes a constitutional sense's wiring on a negative result, and right now a
negative result cannot distinguish "olfaction does not help" from "we withheld
the one state variable the field says the task needs". Same shape as wk1-N1
(the `UB.11` certificate pre-gate) and accepted for the same reason — a
positive control in front of a deletion. Their code is announced, NOT released;
budget for reimplementation.

**wk2-N2 — ACCEPTED as an arm.** RPE-prioritised replay → `NEEDS_AND_DEATH`
§3.4 S1, whose sampling is currently uniform+reservoir. Roscow/Howe/Lepora/
Jones, Nat. Comms s41467-025-65354-2. **Take the control, not the mechanism.**
The scout is right that the mechanism is arguably just PER (Schaul 2015) and
recorded that objection itself; what is genuinely ours to reuse is the
published **double dissociation** — reward-prediction ERROR biases replay
(p<0.05) while reward MAGNITUDE does not (p>0.05), with a shuffled control that
kills the effect. A must-fail control that a wet-lab already ran is worth more
to this ladder than another sampler. Register the magnitude arm as the control
that must fail; if it wins, our RPE signal is measuring salience, not surprise.

**wk3-N1 — ACCEPTED as an arm, and it is the cheapest nomination in three
weeks.** CIG, Conditional Information Gain (arXiv:2605.20878, cs.LG) →
`LEARNING_CORE` §5.4 arm `A3`'s epistemic term, and the same estimator inside
`CURIOSITY_BAKEOFF`'s `disagree`. The finding that earns it entry is a
correctness observation about machinery we had already committed to on two
independent paths: both compute Plan2Explore's per-step ensemble disagreement,
and that estimator conditions on the replay buffer alone — it is blind to what
the *current rollout* has already probed, so a rollout that re-enters its own
novel region collects the bonus repeatedly. Same M=5 ensemble, zero new
hyperparameters, a few dozen lines, no compute budget. **Two conditions travel
with it.** (1) The scout states plainly that it does NOT fit `LT.04` (model-free,
no imagined rollout; the O(T³) term is negligible at imagination horizon T≈15
and is not negligible at T=2000) — do not let it leak into the model-free arm.
(2) **It must not delay `LC.03`.** It enters a bakeoff (`LC.04`+) that cannot
start until the screening round has run, and `LC.03` is already three days
overdue; this is queue material for after it, not a reason to reorder.

**wk3-N2 — ACCEPTED INTO THE ARENA, WITH ITS WIN CONDITION AMENDED. Read this
disposition, it is not a plain accept.** Optimistic World Models
(arXiv:2602.10044, cs.LG) → an arm on `A2`. The numbers are real and now
verified (Atari100K mean HNS 152.68% vs DreamerV3 97.45% at 10 seeds; DMC
proprio-sparse — the only regime resembling `W0` — Acrobot Swingup Sparse
8.4 → 34.6; zero new parameters, +20% train wall-clock).

The scout's lead objection is **constitutional, not empirical**, and it is
correct: the optimistic term deliberately biases the dynamics model toward
high-reward futures, and *in this project the world model IS the unified brain*
(`LEARNING_CORE` §5.4 is `UB`'s binding objective). An arm that improves return
by corrupting the representation would win its own bakeoff and quietly damage
every `UB` gate measured downstream — and in a survival world, an optimistic
model imagines food that is not there.

Rejecting it on that reasoning would be deciding by argument, which SYSTEM.md
law 3 forbids. So the disposition converts the argument into a test instead:
**this arm may not be adopted on task return alone. Its win condition must
additionally include the `UB` representation gates re-run under it** — if it
buys return by degrading what the fusion gates measure, that is a loss, not a
trade. Register the arm and the amended win condition in the same entry. Also
carry: α=1e-4 with "drastic degradation" at 0.1 is two new hyperparameters
against `B2`, and the headline mean HNS is driven by Private Eye and Up N Down
with no median extracted — ask for the median before believing the headline.

**Three DISC (methodology) items, dispositioned separately from the arms:**

- wk1's *"confirm the results table says what the abstract says"* — **CLOSED,
  already written**: `docs/LESSONS.md:1725`. The last two Reviews carried it as
  outstanding; it is not. Stop re-nominating it.
- wk2's *record the arXiv PRIMARY CATEGORY on every watchlist entry* —
  **CLOSED BY ADOPTION.** The scout adopted it in its own file the same week
  (wk3's watchlist carries categories throughout), and two off-target entries
  were deleted by it on first use — one of them econ.GN. It is a scout-local
  convention that is already running; it does not need a LESSONS entry.
- wk3's *"this week the unverified claim was mine"* — **ACCEPTED as a
  `docs/LESSONS.md` entry for the builder to write** (the Review does not
  commit LESSONS.md). It generalises past the scout and is the sharpest thing
  in three sweeps: verify-before-nominating protects a reader of *papers*,
  because there is an abstract to doubt. It protects nobody reading **our own
  ledger**, where the diagnosis is a story the reader wrote itself and there is
  no external claim to check it against. The scar is concrete: a fully
  literature-shaped, correctly-cited diagnosis of `VO.01`'s FAIL (range-blind
  amp recovery), which the scout then falsified with its own arithmetic —
  R² ceiling 0.816 against a 0.50 gate, so the named mechanism could not be the
  cause, and measured amp was 0.432 with brightness unmoved. **Rule to write: a
  diagnosis of one of our own failures must carry the arithmetic that survives,
  not the literature that motivated it.**

**One VERIF worth not losing** (no action, recorded so no future spec reaches
for it): the "walls pass bass" physics behind `VO.01` is confirmed, but its
famous biological corollary — the Acoustic Adaptation Hypothesis, dense habitat
→ lower-frequency calls — is **REFUTED** (Freitas et al., Biol. Rev. 2025,
10.1111/brv.13163; Mikula et al., Ecol. Lett. 2021). No jungle spec may argue
"jungle animals evolved low calls, so Jack's emitter should."

**Scout cadence, noted and endorsed:** three sweeps in three days was justified
each time by a queue item, and the scout says its own mandate is now spent —
fronts 1–3 should not be re-swept before ~2026-08-19. Agreed. Its cron is
Mondays; the extra sweeps were resume-triggered catch-ups, not a runaway.

## FIELD WATCH wk4 CONSUMED by the Review, 2026-08-25 (DAILY)

Source: `docs/FIELD_WATCH.md`, sweep **2026-08-24**, three nominations. Twelve
days since wk3 — the first sweep on the intended weekly-or-slower cadence.
Consumed one day late: the 08-24 Review died on an API 529 before Part 2.5 (the
same failure mode that orphaned wk2; `review.sh` now retries once, `7f3a907`).
The scout nominates; it never adopts. Dispositions below.

**A jurisdictional fact that governs all three, and the scout stated it itself:
`LC.04`/`LC.05` are BLOCKED behind D10, so none of these can enter an
arbitration this week.** D10 is now ARMED with `decide_by 2026-08-31` and a
default of "accept the screen's answer", so the block has an end date for the
first time. All three are therefore ACCEPTED AS DESIGN WORK — written up now,
runnable the week D10 resolves — and none is a dispatch today.

**wk4-N1 — ACCEPTED as an `A4` variant, NARROWED exactly as nominated.** The
spectral parameterisation of the deterministic latent transition, extracted
from Koopman Dreamer ([arXiv:2607.19719], cs.LG) and **nominated ALONE**. The
scout's own framing is the reason this clears a bar four world-model
nominations have died on: all nine of the paper's DMC tasks are
**proprioceptive/state-vector**, which is W0's regime, and the constraint was
confirmed on re-fetch to be decoder-independent — so it ports to `A4` (which is
`A2` minus the decoder) rather than to the arm that measured −0.94. Two
hyperparameters (ρ_min, ρ_max), no new network, no new loss term, and `A4`'s
≈1.37 M parameters stay put or fall. **The condition that travels with it, and
it is the scout's own strongest objection, not mine:** the headline 8/9 win rate
has **no stated seed count and no CIs**, assembled from "available seeds" under
a public protocol — precisely the statistic `UNIFIED_BRAIN_BAKEOFF.md` §1.8
adopted Agarwal et al. to distrust. So it enters as an ARM measured here at
≥3 seeds against plain `A4`; the paper's numbers are the motivation and are not
admissible as evidence about Jack. Register the arm design, do not dispatch.

**wk4-N2 — ACCEPTED, and it is the one to design FIRST of these three, because
the nomination arrives already split into its own control.** PSG-JEPA
([arXiv:2608.06799], cs.RO) → `LEARNING_CORE` §5.4, as **two arms, not one**:
`A4`+`ℒ_dynamic` and `A4`+`ℒ_static`. The scout's asymmetry is correct and is
the whole value of the entry: in W0 the observation dict **already contains**
proprioception, so `ℒ_static` is a decoder on a slice of the input — the exact
thing `A4` deleted and `A2` kept while measuring −0.94 — whereas `ℒ_dynamic`
(multi-horizon joint-angle CHANGE across latent PAIRS) is a temporal quantity
that appears nowhere in any single-step observation. **Run both and they are
each other's control**, distinguishing "grounding helps" from "any auxiliary
reconstruction helps", which one combined arm would leave permanently
ambiguous. **Pre-register the prediction the scout already made: `ℒ_static`
alone should not help and may hurt.** A nomination that ships a falsifiable
prediction about itself is worth more than one that ships a number. Cheapest of
the three: one MLP head per loss, discarded after training, zero inference
parameters, one shared hyperparameter. **Carry objection 4 into the entry**: its
target is joint-angle kinematics and D9's body fork is open — if the owner
adopts a different body the loss's target set changes underneath it, so this
arm is sequenced AFTER D9, not before.

**wk4-N3 — ACCEPTED, but ONLY in its W0-diagnostic form, and that form is
promoted above the other two.** Infant motor noise ([arXiv:2606.16590], cs.LG;
code released, 10 seeds — the best seed discipline in the sweep). **REJECTED as
an exploration arm on `A0`/`A1`**, for the reason the scout raised against its
own nomination: those are PPO, whose exploration *is* the policy distribution,
and injecting autocorrelated noise into an on-policy actor breaks the
likelihood ratio the update is built on. That is a design decision with its own
cost and no paper behind it, and this desk will not make it by argument.
**What IS accepted is the diagnostic**: run the existing `random` and
`random-repeat` nulls that `LC.03` already defines against a β-scheduled random
policy in W0, and read whether temporally-structured *random* action changes
`life_gain` at all. **This is now the cheapest instrument on the most important
open question in the project.** Four independent measurements (LC.03's
darkroom, LC.03 v2, DP.05's FAIL, SH.01's ORACLE_CANNOT) say W0 is too shallow,
and every one of them is expensive. This one costs CPU-minutes and asks a
sharply different question: **is part of what we measured as "the cores cannot
learn" actually "the exploration process never reaches the food"?** If a
β-scheduled null beats plain `random` on `life_gain`, that is evidence about
the world on the cheapest possible substrate, and it lands directly in D10's
fork (b) and in the `w0-too-shallow` Review row. If it does not, the shallowness
finding survives an attack that costs almost nothing — which is what a good
control is for. **Sequence it BEFORE any W1 world redesign**, not after: a
redesign informed by four expensive instruments and one cheap contradicting one
is better than a redesign informed by four.

**DISC items, dispositioned separately from the arms:**

- wk4's **`[s]` marker for search-level claims** (§6) — **ACCEPTED as a
  `docs/LESSONS.md` entry for the builder to write** (the Review does not commit
  LESSONS.md). The scar is real, self-reported, and sitting in an append-only
  file: `FIELD_WATCH_LOG.md`'s 08-12 entries attribute to arXiv:2607.25337 a
  "48× faster" claim and a "single GPU in hours" claim that **two full-text
  fetches confirm appear nowhere in the paper**, plus an OGB-Cube figure the
  table contradicts. The generalisation past the scout is the point: `[c]` means
  "the authors claim this and I have not checked", `[s]` means "a third party
  asserts the authors claim this" — and only the second can be a number **nobody
  ever wrote**. One character, and it separates unchecked from possibly-invented.
  **Retraction accepted and recorded here so it is not re-quoted.**
- wk4's **structural finding — three of four nomination-grade papers report no
  hardware and no wall-clock, two report no parameter count** — **NOTED, no
  action, and it is not a complaint.** `B4` is a throughput gate; the literature
  is increasingly un-pre-priceable against it. The correct response is the one
  the scout already took: every nomination now carries "must be re-measured here
  before it is believed" as a structural property rather than a caveat. Nothing
  to register.
- **Front 4 (curiosity) got two searches and no fetch, and the scout says so
  in its own §1.** Endorsed as an honest gap, and its own queue item 4 ("next
  sweep it goes first, not last") is the right fix. No action here.
- **Conference-proceedings enumeration: the DROP is ENDORSED.** Week 3 set the
  rule that a third "still pending" would be a lie by deferral; the scout dropped
  it with cause rather than renaming existing work. That is the rule working.

**Cadence:** cron is Mondays, wk4 ran Monday 08-24, embargo to ~2026-08-31.
Correct and on schedule for the first time in three weeks.
