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


# ── PLASTIC-ONLY DECREE, 2026-08-09 — what it does to existing specs ─────
# Owner decree: nothing INSIDE Jack is frozen (encoders, core, fusion all
# learn). The parent LLM is unaffected — it is not inside him.
#
# NO THRESHOLD IS TOUCHED (law 4). What changes is what a result MEANS:
#   T2.03  "pretrained vision beats random" — was a decision, is now
#          INFORMATION ONLY. Even if pretrained wins, we do not adopt a frozen
#          encoder; a win would instead be evidence for INITIALISING a plastic
#          encoder from pretrained weights, which the decree permits (it
#          forbids freezing, not inheriting).
#   T3.08  "ablate the LLM" — still valid and now MORE important: it tests
#          whether the parent is load-bearing from outside.
#   T1.05  "frozen stays frozen" — still valid as a MECHANISM test (if we
#          freeze anything, e.g. during a warmup, it must actually stay
#          frozen). It no longer implies we ship frozen parts.
#   T3.10  "trunk knowledge survives action training" — reframed: under
#          plasticity this becomes a catastrophic-forgetting measurement, the
#          risk the decree accepts.
#   PL.*   the frozen-vs-plastic bakeoff collapses: three of four arms freeze
#          at some stage, so only the pure arm survives. PL.00 (throughput)
#          and PL.02 (reshaping gain) still run — as feasibility checks on the
#          plastic path, and as the decree's own RE-OPEN TRIGGER.
#   UB.7 / LC.00 / LC.06 / PG.6 / ME.11.0 — mention frozen only incidentally;
#          unaffected.

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

    Spec("T0.15", 0, "The recorder cannot disarm a threshold",
         hypothesis="For every magnitude a spec might measure, the value "
                    "run_spec hands to check() distinguishes a real nonzero "
                    "from zero, so a pre-registered bound below the recorder's "
                    "resolution still fires.",
         falsified_by="Any nonzero seed metric aggregating to exactly 0.0, or "
                      "a 3e-7 drift satisfying a `<= 0.0` gate.",
         null_baseline="n/a — an invariant of the machine, not an effect.",
         metric="min_resolvable_magnitude", budget=Budget.CPU, seeds=1,
         depends_on=["T0.08"],
         control="The PRE-FIX aggregator (round(x, 6)) run through the same "
                 "checks MUST fail them. Without it this spec would pass on "
                 "any implementation, including the broken one.",
         kills="Nothing directly — it re-arms gates that were silently dead.",
         notes="FOUND 2026-08-09 from PG.8's own ledger entry: two 1e-9 "
               "deviation gates recorded 0.0 because _aggregate did "
               "round(mean, 6) and run_spec calls check() on the AGGREGATE. "
               "Every threshold below ~5e-7 was therefore unenforceable — "
               "T0.14's bit-identity gate (MAX_EVAL_DRIFT = 0.0), T0.02, "
               "T1.10, T1.11, T0.03, T0.04. No PASS was falsely green: "
               "_aggregate short-circuits at one run and all six are seeds=1. "
               "The exposure was latent and pointed exactly where the project "
               "keeps going — re-verify at 3 seeds and the tightest check in "
               "the repo goes quietly dead. Invisible to T0.13, which perturbs "
               "the RECORDED value and finds the gate live either way: the "
               "saturation is manufactured downstream of every test."),

    Spec("T0.16", 0, "The evaluation a spec SHIPS is deterministic, not the one the pipeline owns",
         hypothesis="Replaying the exact call order a locomotion spec's kernel "
                    "performs — untrained eval, rollout, PPO update, trained "
                    "eval — the action-producing path is in eval mode and "
                    "returns bit-identical actions for one identical state at "
                    "BOTH evaluation points.",
         falsified_by="Either evaluation point runs with dropout live, or two "
                      "calls of the shipped eval path on one state differ.",
         null_baseline="n/a — an invariant of the composition, not an effect.",
         metric="max_shipped_eval_drift", budget=Budget.CPU, seeds=1,
         depends_on=["T0.14"],
         control="The PRE-FIX evaluation body — tp.model(tp.project_obs(...)) "
                 "with no mode call, copied verbatim from T2.01 v4 — MUST show "
                 "nonzero drift at both points. Without it this spec passes on "
                 "the broken code it was written to catch.",
         kills="Any locomotion number produced by an eval path that bypasses "
               "act_deterministic.",
         notes="FOUND 2026-08-09 while preparing the T2.01/T2.02 re-runs that "
               "T0.14 made necessary. T0.14 fixed train/eval discipline INSIDE "
               "TrainingPipeline (collect_rollout_vec .eval(), rl_update "
               ".train()) and PASSes. Both locomotion kernels carry their own "
               "eval_policy() that forwards through tp.model directly and never "
               "sets the mode: the untrained control because a fresh nn.Module "
               "defaults to training=True, the trained arm because rl_update "
               "leaves train mode on. Measured at that call site on the real "
               "57M net: 103.6% relative drift between two forwards of ONE "
               "identical state. So the ~13 GPU-hours of re-runs whose entire "
               "purpose was to remove the dropout confound would have "
               "reintroduced it. This is LESSONS' 'a guard built by fixing one "
               "file leaves the file that motivated it unfixed' at composition "
               "scale: T0.14 could not see across the process boundary into a "
               "kernel string."),

    Spec("T0.17", 0, "A verdict that did not come from a run cannot look like one",
         hypothesis="Every change to a ledger entry that was not produced by "
                    "run_spec is attributable from the entry itself — author, "
                    "reason, prior value, commit and time — and no such change "
                    "can set a status that asserts a capability.",
         falsified_by="An amendment landing without author or reason; an "
                      "amendment reaching PASS or FAIL; a run_spec result "
                      "carrying an `amended` note it did not earn; an "
                      "unreconstructible attempt count re-asserting an integer "
                      "after a later run; or an amended verdict re-recorded "
                      "into history with its amendment stripped.",
         null_baseline="The PRE-FIX ledger: a Result has no field that can "
                       "represent 'this did not come from a run', so a "
                       "hand-set status is indistinguishable from a measured "
                       "one — the null detector answers 'run' to everything.",
         metric="amend_provenance_ok", budget=Budget.CPU_FAST, seeds=1,
         depends_on=["T0.08"],
         control="Direct mutation of the row (the actual 9b92d14 hand-edit, "
                 "replayed on a temp ledger) MUST be indistinguishable from a "
                 "recorded verdict under the same audit. Without it this spec "
                 "passes on a ledger where nothing was ever checkable.",
         kills="Nothing. It re-arms the ledger header's own claim that a "
               "capability here came from a test that could have failed.",
         notes="FOUND by the overseer 2026-08-09 (RANK 1): "
               "`experiments/ledger.json` says 'Do not hand-edit' and had been "
               "hand-edited at least twice — T2.01's status FAIL->VOID in "
               "9b92d14 with a prose message written by an agent, and T2.02 "
               "restated when Status.VOID was introduced. Both edits were "
               "substantively RIGHT, which is what makes this a record defect "
               "rather than a science one: the file asserted a distinction "
               "(runner-written vs hand-written) that it had no field to "
               "carry. Same shape as `attempt: 1, history: []` on five "
               "entries and as the Arm.cost lesson — a field that cannot "
               "represent 'unknown' will silently claim a value."),

    Spec("T0.18", 0, "Every PASS is re-derivable from the record, and every control is read",
         hypothesis="Feeding each PASSing entry's RECORDED metrics back through "
                    "its COMMITTED `_check` re-derives PASS for all of them; and "
                    "deleting the control metrics from that same call changes "
                    "the verdict for every spec that ran a control, so no gate "
                    "certifies a capability while ignoring the condition that "
                    "was supposed to fail.",
         falsified_by="Any PASS whose committed gate no longer accepts its own "
                      "recorded numbers; any gate that still returns PASS with "
                      "`control_metrics = {}`; any spec declaring a control it "
                      "never ran; any entry the scan could not judge; or the "
                      "undeclared-control debt growing past its ratchet.",
         null_baseline="The pre-2026-08-10 machine: nothing re-derived a stored "
                       "verdict at all, and 'the control ran' was the strongest "
                       "statement available — a gate that never reads `c` is "
                       "indistinguishable from one that does, under every "
                       "structural check the repo had (grep, non-empty "
                       "control_metrics, and T0.13, which only perturbs keys a "
                       "gate REFERENCES).",
         metric="control_blind_specs", budget=Budget.CPU_FAST, seeds=1,
         depends_on=["T0.08", "T0.13"],
         control="A planted five-entry record scanned by the SAME function: one "
                 "healthy gate that must NOT be flagged, one whose recorded "
                 "metrics no longer clear it, one that ignores its control "
                 "entirely, one declaring a control it never ran, and one "
                 "recording a control its spec does not declare. The scan must "
                 "flag exactly the four and spare the first. Without it a clean "
                 "scan and a scan that never ran are the same output — the "
                 "failure T0.13 shipped on its own first attempt.",
         kills="Nothing. It re-arms law 2 ('a control that also passes means "
               "the test measures nothing'), which was unenforceable for any "
               "gate that never read its control.",
         notes="FROM the overseer 2026-08-10, FOR THE BUILDER item 1, and its "
               "§1.2/§1.3 findings carried across three audits. Scope is "
               "PASS entries only — a claim is what needs re-judging. Two "
               "corrections to the ask, both recorded rather than quietly "
               "applied: (a) probe A does NOT catch a loosened check, it "
               "catches the opposite drift; `impl_sha` catches loosening. (b) "
               "the undeclared-control count is 19 among PASSes, not 20 — the "
               "20th (T2.02) is VOID. That count is gated as a RATCHET, not at "
               "zero: it is a real debt, it went 19->20->19 across audits with "
               "nothing to stop it growing, and a threshold nobody can meet is "
               "a threshold nobody watches. Also exposed as `run verify`, "
               "which runs the fixture first and refuses to report a clean "
               "scan it may not have performed."),

    Spec("T0.20", 0, "The sensory inventory is audited against biology, not against our own map",
         hypothesis="`experiments/senses.py` reports, for every entry of the "
                    "HUMAN sensory inventory, whether the live registry claims "
                    "it — and it can see the bad case: a sense whose specs were "
                    "never written reads ABSENT, a declared spec id that no "
                    "longer resolves LOSES its coverage rather than keeping it, "
                    "a spec that merely CONTAINS a sense's word buys no "
                    "coverage at all, and a passing SENSOR certificate does not "
                    "buy the LOAD-BEARING tier that GOAL.md's 'ablate a sense, "
                    "something measurable must degrade' actually asks for.",
         falsified_by="Any of the seven properties failing. Above all P4: if a "
                      "decoy spec that merely mentions a sense reads as "
                      "coverage, this audit has reproduced the exact artifact "
                      "that hid the hole — the overseer's grep matched 'voiced' "
                      "in PG.5 and voice did not exist.",
         null_baseline="An EMPTY registry: every entry of the inventory must "
                       "read ABSENT. If any sense reads covered against no "
                       "specs at all, the audit is reading its own declarations "
                       "instead of the ladder, and it would report coverage for "
                       "a repository containing nothing.",
         metric="properties_failed", budget=Budget.CPU_FAST, seeds=1,
         depends_on=["T0.01"],
         control="The organ that FAILED, kept as executable code: coverage by "
                 "keyword scan over spec text, which is what the overseer ran "
                 "by hand. Against a registry with the SM/TA/VO families "
                 "removed and PG.5's 'voiced' decoy left in, it must report "
                 "smell/taste/voice as COVERED — breaking P3 and P4 — while the "
                 "declaration-based audit reports them ABSENT. A control that "
                 "also reports ABSENT is not a decoy and this test measures "
                 "nothing (T0.08 property 5, T0.16, T0.19).",
         kills="`run senses` as a trustworthy report. If the battery cannot be "
               "made to pass, the inventory audit is deleted rather than kept "
               "as a green light nobody may rely on.",
         notes="SCAR: OVERSIGHT.md 3.2, 2026-08-10 — smell, taste, voice, pain "
               "and temperature had ZERO specs among 137, and no organ could "
               "say so, because every organ measures this project against what "
               "it wrote down. LESSONS.md:783 named that blindness 30 hours "
               "earlier and prescribed 'at least one recurring audit measured "
               "against a reference from OUTSIDE the project's own documents'; "
               "a lesson prescribing a guard is not a guard. `INVENTORY` is "
               "that outside reference and is deliberately NOT derived from "
               "LADDER or from GOAL.md's prose: adding specs cannot shrink it. "
               "It reports; it never gates a build, because a red exit would "
               "tempt someone to shrink the inventory to clear it."),

    Spec("T0.21", 0, "The GOAL.md coverage audit cannot be flattered by a word",
         hypothesis="`experiments/coverage.py` credits a commitment ONLY to a "
                    "spec that declared `COVERS:`, and it can see both bad "
                    "cases: a spec whose title merely contains a commitment's "
                    "word buys NO coverage, a spec that declares one with an "
                    "unrelated title DOES, a declaration naming no known "
                    "commitment is reported as malformed rather than silently "
                    "dropped, and deleting a declaring spec loses the coverage "
                    "instead of leaving stale credit.",
         falsified_by="Any of the seven properties failing. Above all P3: if "
                      "the decoy spec titled 'The honest baseline' reads as "
                      "coverage for shelter/building, the audit has reproduced "
                      "the exact artifact it was rewritten to end — the owner's "
                      "own image of success credited to the letters in "
                      "'ho-nest'.",
         null_baseline="An EMPTY registry: every commitment must read zero "
                       "specs and zero passes. An audit that reported coverage "
                       "for a repository containing nothing would be reading "
                       "its own commitment list rather than the ladder.",
         metric="properties_failed", budget=Budget.CPU_FAST, seeds=1,
         depends_on=["T0.01"],
         control="The organ that FAILED, kept as executable code: the "
                 "pre-2026-08-10 patterns granting coverage by regex over "
                 "titles, verbatim and without word boundaries. It must break "
                 "P3 and P4 — crediting 'The honest baseline' to "
                 "shelter/building and refusing the declared spec — while the "
                 "declaration rule gets both right. A control that also gets "
                 "them right is not a decoy and this test measures nothing "
                 "(T0.08 property 5, T0.16, T0.19, T0.20).",
         kills="`python -m experiments.coverage` as a report anyone may cite. "
               "If the battery cannot be made to pass, the audit is deleted "
               "rather than kept as a coverage number nobody may rely on.",
         notes="SCAR, and it is the SECOND from this file. The first was a "
               "false NEGATIVE (BA.01 written to close the `balance` hole and "
               "not matched), found within a day and fixed by adding the "
               "`COVERS` marker as an OR alongside the regex. The false "
               "POSITIVE half survived two more days: measured 2026-08-10, "
               "shelter/building reported 4 specs / 1 PASS where the truth was "
               "1 spec / 0 PASS, and the passing spec was ME.11.0, 'The "
               "paraphrase eval set is HONEST before anyone is scored'. "
               "Proprioception's PASS was PG.3, 'Ladder is c-LIMB-able'; `dies` "
               "matched inside `bo-dies`. Word boundaries alone do not fix it "
               "(PG.1's 'physically sound' still matches `hearing`), which is "
               "why the repair is structural: a regex hit is now a NOMINATION "
               "and only a declaration is coverage. P3 and P4 are the "
               "known-answer test LESSONS.md:1673 prescribes — feed it the case "
               "you already know is broken. P5 caught its own author on the "
               "first run: this spec's notes originally wrote the marker "
               "literally in prose and P7 read it as a malformed declaration. "
               "That is the design, not a bug — a marker that buys nothing must "
               "be loud, and the cost is that a spec discussing the mechanism "
               "may not spell it."),

    Spec("T0.22", 0, "A number borrowed from another spec's entry cannot be stale",
         hypothesis="`protocol.borrow_metrics` hands over another spec's "
                    "measured constants ONLY when that entry still describes "
                    "the code that exists now — refusing a source that is not "
                    "PASS, that ran from a modified tree, that predates "
                    "`impl_sha`, or whose implementation hash has moved — while "
                    "still handing over the honest case; it names the source's "
                    "`impl_sha` in the borrower's own record whether it hands "
                    "over or refuses; and no test in the ladder reads another "
                    "spec's metrics around it.",
         falsified_by="Any of the nine properties failing. Above all P3: if a "
                      "PASS entry whose implementation hash has moved still "
                      "yields its numbers, the guard is the rule it replaced.",
         null_baseline="An EMPTY ledger: every borrow must be refused. A "
                       "borrower that produces constants for a spec that never "
                       "ran is reading its own defaults, which is the failure "
                       "this whole mechanism exists to prevent.",
         metric="properties_failed", budget=Budget.CPU_FAST, seeds=1,
         depends_on=["T0.08"],
         control="THE RULE THAT FAILED, kept executable: `status == PASS` and "
                 "nothing else, as `xl_00_death_and_respawn._calibration` "
                 "carried it until 2026-08-10. It must hand over the numbers "
                 "for all three stale fixtures (CHANGED, DIRTY, UNVERIFIABLE) "
                 "while the guard refuses them. A control that also refuses "
                 "them is not a decoy and this spec measures nothing. "
                 "WIDENED 2026-08-11 (P13): the control also carries the "
                 "pre-2026-08-11 `+dirty` predicate (`ledger.json` alone), "
                 "which must classify an append the runner just made to its "
                 "own evidence log as uncommitted CODE. The borrow rule's "
                 "inputs are that predicate's outputs, so a spec that tests "
                 "what a DIRTY row MEANS and never tests which files earn the "
                 "stamp is resting on an unmeasured foundation.",
         kills="`borrow_metrics` as a guard anyone may rely on, and with it the "
               "claim that XL.00 and the LC family compute in the world PS.01 "
               "measured rather than in some earlier one.",
         notes="SCAR, found by the overseer 2026-08-10 (RANK 2), and it is "
               "T0.14's lesson arriving from the other side. T0.14 stopped a "
               "measured constant being COPIED into a second file, where it "
               "drifts from what produced it; XL.00 obeyed that by reading "
               "PS.01's `j0`/`alpha` live from the ledger at run time — and "
               "gated on `status == PASS` alone. Live is not current. PS.01's "
               "numbers are properties of `playground.py`, `w0.py` and "
               "`drives.py`; change any of them and its entry measures a world "
               "that no longer exists while every arm scored in that world "
               "keeps computing. The instance found was benign (PS.01's flag "
               "was the `IMPL_DEPS` widening and the world had not moved) — "
               "the GUARD was absent, and XL.00's own `kills` states the "
               "stakes: 'a wrong answer here is not a wrong answer about the "
               "world; it is a wrong answer about every arm scored in it.' "
               "LC.03/LC.04 score `life_gain` in that same world, which is why "
               "P9 checks the CLASS — no test may read another spec's metrics "
               "off the ledger directly — rather than only the instance."),

    Spec("T0.19", 0, "The bakeoff's `screen` gate eliminates arms without lowering the bar",
         hypothesis="Under `Spec.gate_mode='screen'` an arm below the 3-sigma "
                    "learning gate is ELIMINATED rather than VOIDing the run, "
                    "and every guard that makes that safe holds: two survivors "
                    "are still required, the winner still cleared 3 sigma, the "
                    "eliminated arms are still recorded, an escaped control "
                    "still inverts the verdict to VOID, `validity` behaves "
                    "exactly as before, and the mode is refused without a "
                    "written rationale on the committed Spec.",
         falsified_by="Any of the seven properties failing — above all P2: if "
                      "`screen` changes the verdict of PS.01/J round 1, the "
                      "mode was reverse-engineered to rescue the run that "
                      "motivated it and must be reverted.",
         null_baseline="The pre-2026-08-10 module: one reading of the gate, so "
                       "any 'which observable carries the bit' bakeoff was VOID "
                       "by construction and no such decision could ever be "
                       "made. Round 1 of PS.01/J is that null, and its per-seed "
                       "AUCs are the fixture.",
         metric="properties_failed", budget=Budget.CPU_FAST, seeds=1,
         depends_on=["T0.13"],
         control="The pre-guard machinery kept as executable code: "
                 "`MIN_FINISHERS = 1`, i.e. crown the best survivor however few "
                 "survived — the version a hurried author writes, and the one "
                 "that WOULD have rescued round 1. The same battery must break "
                 "on exactly P1 and P2 under it. A tidied restatement would "
                 "pass while the shipped path stayed broken (T0.08 property 5, "
                 "T0.16).",
         kills="`Spec.gate_mode='screen'` itself. If the battery cannot be made "
               "to pass, the mode is deleted and detector bakeoffs go back to "
               "escalating to the owner instead.",
         notes="SCAR: `experiments/bakeoffs/ps01_impulse.py` round 1, "
               "2026-08-10 — the first real bakeoff this project ran, VOID "
               "because three of four impact CHANNELS could not separate a "
               "fall, which is the finding it existed to produce. The T2.02 "
               "gate assumes arms are LEARNERS, where a missed gate cannot be "
               "told from a broken run; an observable has no run to break. "
               "Same shape as `controls=`, which the curiosity bakeoff forced: "
               "when a framework's validity check and a design's intent point "
               "opposite ways, the framework is missing a category. This spec "
               "is the price of adding one."),

    Spec("T0.23", 0, "A mistyped command cannot spend the GPU budget",
         hypothesis="An argv containing any token the runner does not "
                    "recognise is REFUSED whole — non-zero exit, and no spec "
                    "dispatched, not even the ones it did recognise — while "
                    "every well-formed argv (a read-only command, a bare spec "
                    "id) behaves exactly as before.",
         falsified_by="The runner reaching `cmd_run` for a spec named beside "
                      "an unrecognised token, or the guard refusing an argv "
                      "that was always legal.",
         null_baseline="The dispatch as it stood before 2026-08-11: unknown "
                       "tokens printed `unknown spec <x>` and the recognised "
                       "specs RAN. Replayed verbatim as the control, it must "
                       "reach the spec on the same argv this gate refuses.",
         metric="properties_failed", budget=Budget.CPU_FAST, seeds=1,
         depends_on=["T0.01"],
         control="THE PRE-GUARD DISPATCH, kept executable: `cmd_run(ledger, "
                 "argv)` on the same malformed argv — which is literally the "
                 "line `main()` used to end on. It MUST reach the spec. A "
                 "control that also refuses would mean the fixture argv is "
                 "harmless and this spec measures nothing.",
         kills="The claim that reading `experiments.run`'s output tells you "
               "what it did. If a token can be ignored, the command you typed "
               "and the command that ran are different commands.",
         notes="SCAR, 2026-08-11 20:08 UTC, made by the builder in this "
               "iteration and caught 3 minutes later by an orphaned PID: "
               "`python -m experiments.run show T1.02` — a subcommand that "
               "does not exist — printed `unknown spec show` and then "
               "SUBMITTED T1.02 to Colab, spending free-tier GPU quota that "
               "no one asked for. The typo is not the interesting part; the "
               "shape is. Between `cmd_run` and `gpu.submit()` there is no "
               "further confirmation, so the runner's argv parser is the last "
               "gate standing in front of the scarcest resource this project "
               "has, and it was built to be forgiving. Forgiving is the wrong "
               "setting for a spend. The fixture deliberately uses an "
               "UNIMPLEMENTED cpu spec so the property under test — did "
               "dispatch reach `cmd_run` — is observable without running or "
               "charging anything; P0 fails loudly if that spec ever gains an "
               "implementation, because a fixture that quietly starts doing "
               "work is the same class of bug one level up."),

    Spec("T0.24", 0, "A finished GPU run cannot be lost on the way home",
         hypothesis="Once a remote kernel has COMPUTED the answer, no step "
                    "between the provider and the ledger may discard it: "
                    "Kaggle's console log is parsed into `stdout` so the "
                    "printed RESULT line is reachable, the log is never "
                    "offered as an artifact, `result_json` takes the named "
                    "artifact or the RESULT line and NEVER guesses at some "
                    "other file, and a reattach never routes to Colab.",
         falsified_by="A Kaggle JobResult with an empty stdout when a log was "
                      "downloaded; the log appearing in `artifacts`; "
                      "`result_json` returning a file it was not asked for; or "
                      "`submit` calling Colab while JACK_REUSE_KERNEL is set.",
         null_baseline="The delivery path as it stood on 2026-08-11, replayed "
                       "verbatim as the control: log-in-artifacts plus "
                       "`next(iter(artifacts.values()))`. On the SAME fixture "
                       "it must still raise the original ValueError.",
         metric="properties_failed", budget=Budget.CPU_FAST, seeds=1,
         depends_on=["T0.12"],
         control="THE PRE-FIX DELIVERY, kept executable: collect every "
                 "downloaded file as an artifact, then take an arbitrary one "
                 "as the result. Against the real 2026-08-11 log fixture it "
                 "MUST fail with `dictionary update sequence element #0 has "
                 "length 3; 2 is required`. A control that now succeeds would "
                 "mean the fixture no longer reproduces the bug and this spec "
                 "guards nothing.",
         kills="The assumption that a paid run's cost is bounded by whether it "
               "ran. It is not: the money is spent when the kernel completes, "
               "and every line after that is an uninsured chance to throw the "
               "answer away.",
         notes="SCAR, 2026-08-11 21:47 UTC. Kaggle kernel "
               "`jannolouwrens/jack-ladder-1786482462` ran T1.02 to completion, "
               "charged 0.6561 h, and printed all three seeds' numbers. The "
               "harness then recorded ERROR: `ValueError: dictionary update "
               "sequence element #0 has length 3; 2 is required`. Three "
               "independent defects lined up. (1) `run_on_kaggle` never "
               "populated `stdout` — Kaggle has no stdout pipe, the console "
               "arrives afterwards as a JSON record array — so EVERY spec's "
               "'fall back to the printed RESULT line' branch was dead code on "
               "the one backend that runs the long jobs. (2) That log was "
               "handed back as an artifact. (3) T1.02 looked up "
               "`artifacts['/content/out.json']` — a remote path, while both "
               "backends key by basename, so the lookup could never hit — and "
               "fell through to `next(iter(artifacts.values()))`, which took "
               "the log. Each defect alone is survivable; together they turn a "
               "correct, paid-for measurement into a crash. The answer was "
               "recovered from the log by hand and the run was NOT repeated. "
               "The fourth property is a near-miss found while fixing this: "
               "`submit` walked its normal `prefer` order during a reattach, "
               "so recovering a finished free kernel would have paid for a "
               "fresh Colab job first.\n"
               "Deliberately declares NO `COVERS:` commitment. It guards the "
               "harness, not a capability, and counting a delivery gate toward "
               "`honesty` is the adjacent-PASS inflation the 8th audit asked "
               "the builder to stop doing."),

    Spec("T0.25", 0, "The critic is a baseline, or it is decoration",
         hypothesis="Subtracting a PERFECT value function from the return "
                    "leaves nothing behind. Feed `compute_gae` the analytic "
                    "value function of its own reward sequence and every "
                    "advantage must be zero — at any state of the return "
                    "normaliser, not only at the fresh scale=1 where the two "
                    "unit systems happen to agree.",
         falsified_by="Advantages that survive a perfect critic. The residual "
                      "ratio std(adv | perfect V) / std(adv | V=0) is 0 for "
                      "any correct advantage estimator; anything above 0.02 "
                      "means the value head is not being subtracted in the "
                      "units the rewards are measured in, and PPO is running "
                      "as REINFORCE with a batch-mean baseline.",
         null_baseline="V = 0 (no critic at all). Its advantages are the raw "
                       "lambda-discounted reward sums, and the residual ratio "
                       "is 1.0 by construction — that is the denominator, and "
                       "it is what a critic has to improve on.",
         metric="max_residual_ratio", budget=Budget.CPU_FAST, seeds=1,
         depends_on=["T0.14"],
         control="THE PRE-FIX RECURSION, kept executable: GAE with the critic "
                 "output used verbatim while the rewards stay raw. On the SAME "
                 "fixture, at a warmed normaliser, it MUST leave a residual "
                 "ratio above the threshold. A control that now cancels would "
                 "mean the fixture no longer reproduces the defect and this "
                 "spec guards nothing.",
         kills="The assumption that an actor-critic is doing credit assignment "
               "because it has a value head and its vf_loss is small. A critic "
               "can fit its targets perfectly and still contribute nothing.",
         notes="SCAR, 2026-08-12. T2.01 v4 (Kaggle P100, 692K env-steps/seed, "
               "6.5 GPU-h) recorded FAIL at 1.19 sigma against a 5-sigma bar, "
               "with trained means [231.9, 384.5, 155.3] — and seed 2's "
               "TRAINED policy (155.3) scored below its own UNTRAINED control "
               "(186.0). The rollout curve plateaued by ~300K steps at "
               "mean_reward ~5.1, which is Humanoid-v5's healthy_reward of 5.0 "
               "plus almost nothing: the policy learned to stand a little "
               "longer and never learned to move. The mechanism is a unit "
               "mismatch, not a compute shortage. `vf_loss` fits the critic to "
               "returns AFTER they are divided by the running return-std, so "
               "the critic emits V/scale; GAE's delta then adds RAW rewards to "
               "those normalised values. Measured on the ledger's own numbers: "
               "value_mean ~3.5 while mean_reward ~5.0/step and the true "
               "discounted return at gamma=0.95 is ~100 — the baseline was "
               "~28x too small, so delta reduced to r_t and the advantage "
               "became a discounted reward sum with a constant offset. With a "
               "PERFECTLY trained critic, 79% of the advantage variance "
               "survives (probe, 2026-08-12). Two organs were recommending a "
               "seventh GPU-hour re-run of the same configuration; T2.01's own "
               "pre-registration says a plateaued curve is an architecture "
               "verdict, and this is what the architecture was actually doing.\n"
               "Deliberately declares NO `COVERS:` commitment. It guards the "
               "learning machinery, not a capability."),

    Spec("T0.26", 0, "A rig-health gate refuses a broken world and admits an honest one",
         hypothesis="BA.01's per-seed rig-health gate is live in BOTH "
                    "directions, measured through the spec's own episode and "
                    "statistic path (`rollout_rig` + `rig_health`, never a "
                    "restatement): a world exhibiting its named failure mode "
                    "— every fall on one schedule — scores tf_fall_spread "
                    "BELOW TF_FALL_SPREAD_MIN and is refused (`rig_ok` 0), "
                    "while the honest rig's bulk on the SAME world scores "
                    "ABOVE it and is admitted. Inert and unreachable are the "
                    "two ways a carried constant dies when the rig moves "
                    "underneath it; this asserts both, executably.",
         falsified_by="The declared degenerate rig clearing "
                      "TF_FALL_SPREAD_MIN (the gate is inert — BA.01 v2's "
                      "defect), or the honest rig's bulk falling under it "
                      "(the gate is a tail lottery — BA.01 v3's defect), or "
                      "`rig_ok` disagreeing with its own statistics, or the "
                      "degenerate fixture failing every OTHER rig-health "
                      "gate too (a world broken in all dimensions cannot "
                      "show that THIS gate is the one doing the refusing).",
         null_baseline="THE PRE-FIX (v2) GATE, kept executable as the "
                       "control: toppled_frac + tf_abs_spread only, no "
                       "fall-spread term. On the SAME degenerate episodes it "
                       "MUST certify the broken world healthy — the hold's "
                       "own uniform t_r puts its abs-spread (pilot 11.13) "
                       "4.5x over the 2.5 gate while fall variance is "
                       "exactly zero. A control that refuses the degenerate "
                       "world means the fixture no longer reproduces the v2 "
                       "disease and this spec guards nothing.",
         metric="properties_failed", budget=Budget.CPU, seeds=1,
         depends_on=[],
         control="See null_baseline: the v2 conjunction replayed verbatim "
                 "against the fixture that fooled it.",
         kills="The assumption that a pre-registered threshold survives a "
               "rig change because its number did. A gate is a claim that "
               "the statistic's attainable range under THIS rig straddles "
               "it — and that claim needs re-measuring every time the rig "
               "moves (law 4 protects the number; this protects the "
               "measurement).",
         notes="SCAR, twice in one day (2026-08-12, 11th audit RANK 1 + the "
               "v3 VOID). BA.01 v2 redefined tf_spread from FALL times to "
               "ABSOLUTE topple times, so the rig's own uniform hold "
               "(std 11.85) cleared the unmoved 2.5 gate 4.7x and a "
               "zero-fall-variance world would have read healthy: the gate "
               "kept its number and lost its meaning. v3 restored the gate "
               "to the right statistic and promptly VOIDed on seed 2 — the "
               "2.5 was UNREACHABLE on open ground (contact-solver floor "
               "caps bulk fall std at ~2.2), so v3's passing seeds had been "
               "clearing it on 1-2 structure-outlier falls: a tail lottery, "
               "not a measurement. Overseer B2 asked for the executable "
               "form of the guard. The degenerate rig is DECLARED IN BA.01 "
               "(one fixed 6.3-deg tilt, zero kick, zero arm noise, every "
               "spawn at the model-derived most-open cell) per the LC.01 "
               "lesson — the artifact names the object under audit. Pilot "
               "(world seed 90, 60 episodes/rig): degenerate toppled 1.0, "
               "tf_abs 11.13, tf_fall 0.0, rig_ok 0.0; honest toppled "
               "0.983, tf_abs 16.01, tf_fall 9.38, rig_ok 1.0. A first "
               "fixture that kept arm noise and uniform spawns measured "
               "tf_fall 3.51 — over the gate — because uniform legal spawns "
               "land beside structure often enough to buy outlier falls; "
               "the fixture pins both, and that near-miss is recorded in "
               "its docstring. EXTENSIBILITY: the next spec that carries a "
               "rig-health gate (PS.02 is the standing candidate) should "
               "declare its own degenerate rig and join this battery.\n"
               "Deliberately declares NO `COVERS:` commitment. It guards "
               "the measurement machinery, not a capability."),

    Spec("T0.27", 0, "A threshold moved after a FAIL leaves an artifact, "
                     "not a paragraph",
         hypothesis="Amend-after-FAIL is auditable by someone who is not its "
                    "author, mechanically: (1) a verdict that supersedes a "
                    "FAIL carries the failing evidence IN the record "
                    "(`supersedes_fail`: failing commit, dirty flag, "
                    "impl_sha, measurement, plus machine-readable "
                    "`impl_changed`), and the pairing survives into history "
                    "when superseded again; (2) `audit_supersedes_fail` "
                    "flags, in any PASS record, a FAIL whose implementation "
                    "differs from the run that amended it unless that FAIL "
                    "is stamped at a clean commit that exists in this repo "
                    "and carries its metrics; (3) the LIVE ledger has zero "
                    "such violations; (4) pre-impl_sha pairs read "
                    "unauditable, never violated — absence is a historical "
                    "gap, not evidence.",
         falsified_by="The recorder writing no artifact on a FAIL->PASS "
                      "supersede; the artifact vanishing from history on the "
                      "next supersede; the auditor certifying the T2.08 "
                      "shape (FAIL stamped +dirty, threshold moved, PASS on "
                      "top); the auditor inventing violations for pre-B1 "
                      "verdict-only history; or a live-ledger violation — "
                      "someone amended a FAIL without committing the failing "
                      "implementation, and the gate re-run catches it.",
         null_baseline="The pre-B2 ledger path: T2.08's exact shape (FAIL at "
                       "a +dirty commit with its measurement, impl changed, "
                       "PASS recorded on top) replayed on a fixture. The "
                       "auditor MUST flag it; an auditor that certifies the "
                       "very case that motivated it measures nothing.",
         metric="properties_failed", budget=Budget.CPU_FAST, seeds=1,
         depends_on=[],
         control="See null_baseline: the control fixture is the disease "
                 "this spec exists to make impossible, and it must be "
                 "caught, not certified.",
         kills="The practice of amending a FAIL from an uncommitted tree. "
               "After this spec, a moved threshold whose FAIL is "
               "unrecoverable fails the gate re-run — law 4's escape clause "
               "leaves `git diff <fail> <pass>` as its artifact or it is "
               "not taken.",
         notes="SCAR (overseer 13th audit RANK 1, 2026-08-13): T2.08's "
               "auxiliary floor moved 0.70 -> 0.50 after the run it failed. "
               "The move was honest, loudly disclosed, and structurally "
               "uncheckable: the FAIL was stamped 75a1938+dirty (v1's code "
               "exists in no commit) and the failing 0.6975 survived only "
               "in prose written by the party that moved the floor. B1 gave "
               "history the evidence slots; this is B2 — the recorder now "
               "pairs every amending PASS with the FAIL it amends, and the "
               "auditor makes 'commit the failing implementation before "
               "re-running' executable. impl_sha cannot distinguish a "
               "threshold move from a code fix, so the rule binds the "
               "conservative superset: ANY FAIL amended by different code "
               "must be recoverable. Pre-B1 history (163 verdict-only "
               "entries) is exempt by absence, per B1's no-back-fill rule. "
               "Property (3) reads the LIVE ledger (B3's lesson: a guard "
               "that only ever sees fixtures guards nothing), so this spec "
               "FAILS at gate re-run the next time anyone repeats T2.08's "
               "shape — that is the point, not a flake.\n"
               "Deliberately declares NO `COVERS:` commitment. It guards "
               "the measurement machinery, not a capability."),

    # ── PLAYGROUND (docs/research/CURIOSITY.md §7) ──────────────────────
    Spec("PG.1", 2, "Playground generates and is physically sound",
         hypothesis="A procedural room (ramp, stairs, ladder, objects, seesaw, "
                    "pool, noise panel) builds from a parameter vector and obeys "
                    "physics: boxes slide iff tan(theta) > mu; energy bounded at rest.",
         falsified_by="Objects jitter at rest, energy diverges, or a parameter "
                      "draw produces an invalid MJCF.",
         null_baseline="n/a — physics validation fixture.",
         metric="physics_checks_passed", budget=Budget.CPU,
         control="FRICTIONLESS: the shallow ramp that HELD must now slide. Without it, \"the box held\" could mean the box was wedged on geometry rather than obeying friction — and when both this and the experiment failed together, that is what localised the MJCF radians-vs-degrees bug in one step (LESSONS.md).",
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
         control="ZERO ADHESION, identical script: the hang must slip and nothing may ascend. Otherwise the ascent could be the scripted kinematics dragging the body up geometry it is resting on.",
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
         control="The IDENTICAL ICM agent with a STATIC panel texture must NOT fixate — else dwell measures the geometry of that corner of the room rather than its unpredictability.",
         notes="Every later curiosity claim must report dwell share on this "
               "fixture. The control above lived in this notes field until "
               "2026-08-10: it ran on every seed and was invisible to a grep of "
               "Spec.control, which is the field an auditor reads (OVERSIGHT 1.4)."
               "  COVERS: curiosity (fixture)"),

    Spec("PG.5", 2, "Procedural contact audio with localization labels",
         hypothesis="Modal-resonator synthesis on MuJoCo contact events yields "
                    "stereo audio whose panning matches source bearing.",
         falsified_by="Bearing decoded from stereo does not match ground truth.",
         null_baseline="Mono/shuffled-pan audio — bearing must be undecodable.",
         metric="bearing_decode_accuracy", budget=Budget.CPU, depends_on=["PG.1"],
         seeds=3,
         control="Mono and shuffled-pan renders of the SAME events must decode "
                 "at chance — else the decoder reads something other than pan.",
         notes="COVERS: hearing (fixture)"),

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
               "(CAPABILITIES.md L2) or the held-out cells cannot exist."
               "  COVERS: language (parent) (claim)"),

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
         control="RECENCY-ONLY on the IDENTICAL seeded query sample — the newest event answers every cue, so its precision is ~1/N by construction and must sit below the experiment at EVERY decade. Same questions, or the two curves are not comparable.",
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
         control="Sleeping with the rehearsal buffer EMPTIED must forget.",
         notes="COVERS: sleep (claim)"),

    Spec("ME.8", 2, "Working memory survives restarts",
         hypothesis="A recurrent state checkpointed to disk resumes mid-episode "
                    "after a kill; zeroing it mid-episode hurts.",
         falsified_by="Post-restart behavior equals a zeroed-state agent.",
         null_baseline="Zeroed hidden state.",
         metric="resume_vs_zeroed", budget=Budget.CPU, depends_on=["T0.05"],
         seeds=3,
         control="CROSS-RESTORE: finish episode i from episode j's checkpoint. "
                 "The answer must follow the FILE (j's cue, match_restored >= "
                 "0.80) and accuracy on i's own cue must collapse (<= 0.30). If "
                 "it can still name i's cue, the second half of the episode "
                 "leaks the answer and nothing here measured memory.",
         notes="seeds=3 since 2026-08-10 (OVERSIGHT 1.5, fourth audit). It "
               "recorded PASS at seeds=[0] while its own commit message "
               "(663270b) reads 'GRU retain-bias init fixes seed-2 training "
               "collapse' — the fix was never verified at the seed that "
               "motivated it, and GOAL.md's standard is >=3 seeds where the "
               "claim rests on something trained. No threshold moved."),

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
               "or skills live only in retrieved episodes.",
         notes="COVERS: memory across lives (claim)"),

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
               "episodes."
               "  COVERS: sight (sensor)"),

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
               "this file's own trig, never from the synth's labels."
               "  COVERS: hearing (fixture)"),

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
               "by design, which is precisely why collapse cannot hide."
               "  COVERS: hearing (claim), one brain / unison (claim)"),

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
               "controller, so decision D1 cannot block the unison claim."
               "  COVERS: hearing (claim), one brain / unison (claim)"),

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
               "3-seed architecture comparisons resolve nothing at this budget."
               "  COVERS: one brain / unison (claim)"),

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
               "necessary-not-sufficient, both red flags rather than claims."
               "  COVERS: one brain / unison (claim)"),

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
               "redundancy/uniqueness/synergy (arXiv:2302.12247)."
               "  COVERS: one brain / unison (claim)"),

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
               "(arXiv:2603.19233: encoded is not used).",
         notes="COVERS: one brain / unison (rule)"),

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
               "its success criterion."
               "  COVERS: one brain / unison (claim)"),

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
               "a PASS condition here, not a failure."
               "  COVERS: proprioception (claim), one brain / unison (claim)"),

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
                 "briefly stand from vision.",
         notes="COVERS: one brain / unison (claim)"),

    Spec("UB.2", 4, "The shared trunk beats late fusion",
         hypothesis="One self-attention trunk over all modality tokens beats "
                    "equal-parameter separate-encoders-then-concat.",
         falsified_by="Late fusion ties everywhere incl. occlusion tasks — then "
                      "'one brain' adds nothing at this scale; report honestly.",
         null_baseline="Per-modality encoders -> concat -> same flow head.",
         metric="fusion_advantage", budget=Budget.GPU, seeds=3,
         depends_on=["UB.1"],
         control="Cross-modal TIME-SHUFFLE at eval must hurt the shared trunk — "
                 "else attention never crossed modalities.",
         notes="COVERS: one brain / unison (claim)"),

    Spec("UB.3", 4, "Cross-modal masking helps the policy",
         hypothesis="Co-training with masked cross-modal prediction (touch from "
                    "vision+proprio, audio-event from dynamics) improves task "
                    "success and few-shot adaptation at equal steps.",
         falsified_by="No downstream improvement — drop the objective.",
         null_baseline="BC-only, same architecture and steps.",
         metric="mask_cotrain_gain", budget=Budget.GPU, seeds=3,
         depends_on=["UB.2"],
         control="Touch-from-SHUFFLED-vision must collapse to the unconditional "
                 "mean — else the head ignores vision and the fusion is fake.",
         notes="COVERS: one brain / unison (claim)"),

    Spec("UB.4", 4, "Hearing is load-bearing",
         hypothesis="Jack turns toward an out-of-view falling object and times "
                    "occluded contacts using audio.",
         falsified_by="Muting audio at eval leaves audio-task success unchanged.",
         null_baseline="Audio-muted model; model trained without audio.",
         metric="audio_task_delta", budget=Budget.GPU, seeds=3,
         depends_on=["PG.5", "UB.1"],
         control="Left/right channel swap must invert turning; 500ms audio lag "
                 "must break contact timing — else hearing is decorative.",
         notes="COVERS: hearing (claim), one brain / unison (claim)"),

    Spec("UB.5", 4, "Touch is load-bearing (or honestly redundant)",
         hypothesis="Touch improves blind push-recovery beyond proprioception.",
         falsified_by="Zeroed touch changes nothing — an HONEST possible "
                      "outcome: foot force is partly inferable from torques. "
                      "That is a finding, not a failure of the test.",
         null_baseline="Touch-zeroed eval; touch-ablated training.",
         metric="blind_recovery_delta", budget=Budget.GPU_SHORT, seeds=3,
         depends_on=["UB.1"],
         control="Permuting the 10 touch channels must cause misattributed "
                 "contacts if touch is load-bearing.",
         notes="COVERS: touch/contact (claim), one brain / unison (claim)"),

    Spec("UB.6", 4, "Contrastive binding: keep only if it moves action",
         hypothesis="Audio<->vision alignment improves hearing-task success "
                    "beyond the same compute spent on BC.",
         falsified_by="No task-success delta — binding is retrieval-only here.",
         null_baseline="Same model, alignment weight zero.",
         metric="bind_action_gain", budget=Budget.GPU_SHORT,
         depends_on=["UB.4"],
         control="The aligned model must retrieve audio->vision clips well "
                 "above chance — else the loss never worked and the null result "
                 "is uninformative.",
         notes="COVERS: one brain / unison (claim)"),

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
               "stays OUT of every capability list."
               "  COVERS: one brain / unison (claim)"),

    Spec("UB.8", 4, "Flow-head attention ablation",
         hypothesis="Interleaved cross+self attention beats cross-only and "
                    "self-only at equal params (SmolVLA's ablation, reproduced).",
         falsified_by="No difference — simplify to cross-only, bank the params.",
         null_baseline="The two single-attention variants.",
         metric="attention_ablation", budget=Budget.GPU_SHORT,
         depends_on=["UB.7"],
         notes="COVERS: one brain / unison (claim)"),

    # ── CURIOSITY (docs/research/CURIOSITY.md; tier 5 = the claims) ─────
    Spec("CU.1", 5, "Goal babbling beats action babbling",
         hypothesis="Sampling goals in OUTCOME space covers more distinct "
                    "outcomes than random action sequences at equal budget.",
         falsified_by="Coverage <= the random-action-repeat null (flailing "
                      "covers ground too).",
         null_baseline="Random repeated action sequences.",
         metric="outcome_coverage_ratio", budget=Budget.CPU_LONG, seeds=3,
         # PG.8 is a dependency, not a courtesy: CU.1 is the ROOT of the
         # curiosity tree (CU.2-CU.7 and T5.08 all descend from it), and until
         # PG.8 passes the playground is an empty room with nu=0. Every one of
         # these specs is defined over an agent ACTING in this world, so
         # without it the runner would happily attempt "goal babbling beats
         # action babbling" in a world where no action exists. PG.8's `kills`
         # field said so in prose; this line is what makes it enforced.
         depends_on=["PG.1", "T2.16", "PG.8"],
         notes="COVERS: curiosity (claim)"),

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
         notes="The first falsifiable form of 'Jack teaches himself'."
               "  COVERS: curiosity (claim)"),

    Spec("CU.3", 5, "Curious without being trapped",
         hypothesis="The LP stack explores (coverage grows) with near-zero "
                    "dwell at the noisy-TV panel.",
         falsified_by="Panel dwell share exceeds the random-walk baseline.",
         null_baseline="Random walk; an ICM arm as the trap-victim reference.",
         metric="coverage_vs_dwell", budget=Budget.CPU_LONG, seeds=3,
         depends_on=["PG.4", "CU.2"],
         control="The ICM control arm MUST fixate on the panel — proving the "
                 "trap works and the LP immunity is real.",
         notes="COVERS: curiosity (claim)"),

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
                 "toward static poses (arXiv:2310.08887).",
         notes="COVERS: curiosity (claim)"),

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
                 "interestingness.",
         notes="COVERS: curiosity (claim)"),

    Spec("CU.6", 5, "Affordances emerge from interaction",
         hypothesis="The interaction archive predicts pushability/liftability "
                    "of held-out objects above chance.",
         falsified_by="Prediction at chance on novel mass/shape.",
         null_baseline="Predictor trained on shuffled object-outcome pairs.",
         metric="affordance_transfer", budget=Budget.CPU_LONG, seeds=3,
         depends_on=["CU.1"],
         control="A welded immovable object must classify un-pushable — else "
                 "the representation captures action, not interaction.",
         notes="COVERS: tool use (claim)"),

    Spec("CU.7", 5, "Lessons from failure improve retries",
         hypothesis="Retrieved one-line lessons written after failures raise "
                    "retry success beyond pure resampling.",
         falsified_by="Retry rate with lessons equals resampling alone (the "
                      "known confound).",
         null_baseline="Retry with no lesson.",
         metric="lesson_gain", budget=Budget.CPU_LONG, seeds=3,
         depends_on=["ME.1", "CU.2"],
         control="Lessons from UNRELATED failures must not help — else the "
                 "effect is generic prompt padding.",
         notes="COVERS: curiosity (claim)"),

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
                 "into unsolvable scenes — else the filter does nothing.",
         notes="COVERS: curiosity (claim)"),

    Spec("T5.09", 5, "Skills transfer across bodies",
         hypothesis="Pretraining on morphology variants (limb lengths, masses) "
                    "speeds learning on a new body versus random init.",
         falsified_by="Transfer <= random init, or negative transfer.",
         null_baseline="Random init on the target body.",
         metric="transfer_speedup", budget=Budget.GPU_LONG, seeds=3,
         depends_on=["T2.02"],
         control="Pretraining on white-noise trajectories must give no gain — "
                 "it is structure, not warm optimizer state.",
         notes="COVERS: generality (claim)"),

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
                 "persona reset must drop identity to chance.",
         notes="COVERS: social/other agents (claim)"),

    # ── THE LEARNING CORE (docs/research/LEARNING_CORE.md) ──────────────
    # Two-digit ids from the start. run.py::_module_for globs lc_00_*.py etc;
    # verified by fnmatch on 2026-08-09 that no LC id shadows another. Do NOT
    # add an LC.0 or an LC.1 — see LESSONS.md, "A spec id that is a prefix of
    # another spec id disables one of them".

    Spec("LC.00", 0, "The learning-core question is decidable in a gridworld first",
         hypothesis="In a 12x12 survival gridworld with two depleting needs, "
                    "death on depletion, random respawn and a persistent "
                    "cross-life visit table, all four learning cores "
                    "(tabular Q on drive reduction; the same plus absolute "
                    "learning progress; a tabular latent-transition model with "
                    "value iteration in the model; and the same model scored by "
                    "expected free energy) run to completion, and at least two "
                    "produce a life_gain that beats the random null by 3 sigma "
                    "over 3 seeds.",
         falsified_by="Fewer than two cores clear the null. Then the METRIC is "
                      "wrong or the world is unlearnable, and no amount of "
                      "MuJoCo will repair either — LC.03 onward must not run.",
         null_baseline="Uniform random action on the same gridworld, same "
                       "seeds: life_gain by construction ~0 (lives do not "
                       "lengthen without learning).",
         metric="life_gain_cores_clearing_null", budget=Budget.CPU_FAST,
         depends_on=[], seeds=3,
         control="A FROZEN core — the same tabular agent with learning "
                 "disabled — must record life_gain within noise of zero. If a "
                 "frozen agent's lives get longer, the world drifts and "
                 "life_gain measures the world, not the learner. This control "
                 "is the reason the spec exists; it is cheaper to discover "
                 "here than after 25 CPU-hours.",
         kills="The whole LC programme, for two CPU-minutes. It is the "
               "cheapest thing that can falsify the metric, the world contract "
               "and the four-core framing before any body, any physics, any "
               "torch or any GPU is involved. Modelled on PS.00.",
         notes="No MuJoCo, no torch. Tabular over (x, y, need0_bucket, "
               "need1_bucket). Also emits the pre-registered numeric value of "
               "the FROZEN control's life_gain, which LC.03/LC.04 reuse as "
               "their own control threshold rather than inventing a new one."),

    Spec("LC.01", 2, "Every candidate core takes every sense into one latent, or it is not a candidate",
         hypothesis="For each admissible arm: (U1) every modality key reaches "
                    "the shared state tensor and no modality has a private path "
                    "to the action; (U2) perturbing modality A's input produces "
                    "a NONZERO finite-difference gradient at modality B's "
                    "encoder through the arm's declared binding loss; (U3) each "
                    "modality can be dropped without a shape error and the "
                    "core's internal uncertainty CHANGES when it is; (U4) the "
                    "need-state modality holds at least 1/|M| of the total "
                    "prediction loss at init.",
         falsified_by="Any arm failing any of U1-U4. That arm is EXCLUDED from "
                      "LC.03/LC.04 — not scored and beaten, excluded — per "
                      "SYSTEM.md's constitutional constraint. An arm cannot buy "
                      "admission with a task score.",
         null_baseline="A deliberately unbound core: per-modality encoders "
                       "feeding a concatenation with NO cross-modal loss term. "
                       "U2's finite-difference gradient must read exactly 0.0 "
                       "for it. That number is what U2 is measured against.",
         metric="unison_admission_conjunction", budget=Budget.CPU, seeds=3,
         depends_on=["PG.8"],
         control="TWO. (a) The unbound core above must FAIL U2 — if a core "
                 "with no binding term shows a cross-modal gradient, the probe "
                 "is reading autograd plumbing rather than the objective. (b) A "
                 "PLACEBO modality of matched dimension and matched statistics "
                 "carrying no information must NOT acquire a loss share above "
                 "1/|M| — if noise binds as well as a sense, U4 measures "
                 "capacity, not binding.",
         kills="Bare PPO as a candidate learning core. Per docs/research/"
               "LEARNING_CORE.md 3.7, PPO's senses meet only through a scalar "
               "reward, so an admissible PPO arm must carry "
               "L_masked_cross_modal. Also kills TD-MPC2 outright "
               "(arXiv:2310.16828 is state-based proprioception only, no "
               "vision, by construction).",
         notes="Runs BEFORE any learning. The finite-difference probe is the "
               "load-bearing part: MULTIMODAL_BINDING.md records pi-0.5 "
               "encoding its language prompt at 99.3% linear-probe accuracy "
               "while behaving invariantly to it, so 'the trunk sees it' is not "
               "evidence that the trunk USES it. U4 exists because DreamerV3's "
               "shipped loss_scales.rec is shared across keys: a 64x64x3 image "
               "contributes 12,288 reconstruction terms and a 10-dim needs "
               "vector contributes 10."
               "  COVERS: one brain / unison (rule)"),

    Spec("LC.02", 2, "A core that cannot live a life at survivable wall-clock is not a core",
         hypothesis="Every admissible arm sustains at least 5.0 simulated "
                    "seconds of Jack's life per real second on 3 ARM cores at "
                    "nice 19 with the learner in the loop, at the train_ratio "
                    "this spec selects for it; and the selected train_ratio is "
                    "the largest power-of-two value that clears that floor.",
         falsified_by="An arm below 5.0 sim-s/real-s at every train_ratio down "
                      "to its minimum. That arm is EXCLUDED: GOAL.md requires "
                      "lives, death and cross-life learning, and a core that "
                      "cannot produce a second life inside a builder iteration "
                      "cannot deliver them at any sample efficiency.",
         null_baseline="Physics alone, zero-action, same body and world: the "
                       "throughput ceiling no learner can exceed. Measured for "
                       "the humanoid at 31.6 sim-s/real-s (DIRECTION_AUDIT.md "
                       "4.1); measured here for the climber-rover.",
         metric="sim_seconds_per_real_second", budget=Budget.CPU, seeds=3,
         depends_on=["PG.8", "LC.01"],
         control="The 57M UnifiedBrain trunk in the control path MUST FAIL this "
                 "floor. DIRECTION_AUDIT.md 4.1 measured it at 0.17 sim-s/real-"
                 "s against a 160K MLP's 22.97 — 133x. If the trunk PASSES a "
                 "5.0 floor, the instrument is wrong, not the trunk.",
         kills="Any arm's train_ratio above the largest affordable value, and "
               "any arm that cannot reach the floor at all. NOTE THE "
               "ANTI-GAMING RULE: this spec's _check MAY NOT READ life_gain. "
               "Selecting a hyperparameter by its score is tuning on the "
               "metric; selection here is on wall-clock fit only, and the "
               "chosen value is committed to the ledger before LC.03 runs.",
         notes="train_ratio and model size are the two things DreamerV3 does "
               "NOT hold fixed across its 150+ tasks (arXiv:2301.04104 Table "
               "A.1), and they are exactly the two that decide affordability. "
               "Director (arXiv:2206.04114) ran at one gradient step per "
               "sixteen policy steps — train_ratio ~0.06 — under 24h on one "
               "V100, so a low ratio is not obviously crippling. Measured on "
               "this box 2026-08-09: PPO 13.1 and a 1.9M RSSM at train_ratio 1 "
               "19.6 CPU-core-seconds per 1,000 decisions, physics included."),

    Spec("LC.03", 5, "Screening: which learning cores learn to survive at all",
         hypothesis="At the LC.02-fixed train_ratio, run to the LC.04 envelope, "
                    "each admissible arm's life_gain beats the random null by "
                    ">=3 sigma AND beats its own untrained twin by >=3 sigma, "
                    "over 3 seeds, with n_lives >= 12 per seed.",
         falsified_by="Fewer than two arms clear both gates. Recorded VOID "
                      "'fewer than two learners' — which blocks the decision "
                      "instead of manufacturing one — and LC.04 does not run.",
         null_baseline="Uniform random and random-repeat action, same world "
                       "seeds, same evaluation lives. PLUS, per arm, that arm's "
                       "own UNTRAINED twin: T2.02's untrained MLP already "
                       "cleared random by 2.74 sigma against a 3.00 gate, so a "
                       "gate against random alone is nearly cleared by a "
                       "network that has never received a gradient.",
         metric="life_gain", budget=Budget.CPU_DAYS, seeds=3,
         # Budget AMENDED CPU_LONG -> CPU_DAYS 2026-08-13: the §5.7 envelope
         # (N_STEPS=100k x 5 arms + wiped twins + twins + nulls + controls,
         # per seed) re-costed at LC.02's MEASURED throughput is ~90 core-h,
         # and run.py kills a child at the declared budget's timeout. The
         # declaration must match behaviour (T2.08); the envelope does not
         # shrink to fit a label.
         # XL.00 added 2026-08-10: LC.03 scores `life_gain` over `n_lives >= 12`
         # and `cross_life_transfer`, and until that commit NOTHING IN W0 COULD
         # END A LIFE (`w0.py`'s own header said "W0-2 death — NOT YET"). The
         # dependency was always real; it was simply not written down, so `run
         # blocked` ranked LC.03 as runnable-today work three iterations running
         # and an iteration reached for it before reading the world's header.
         # A dependency a human has to remember is not a dependency.
         depends_on=["LC.00", "LC.01", "LC.02", "PS.01", "XL.00"],
         control="FIVE, each on its pre-registered side. (a) statue (do "
                 "nothing) must die soonest. (b) randrew (fixed random "
                 "stationary reward projection) must miss the gate — it "
                 "controls for 'any optimisation pressure looks like "
                 "learning'. (c) FROZEN: the best arm with the optimiser never "
                 "stepped must record life_gain within noise of zero; if lives "
                 "lengthen without learning, the metric measures the world and "
                 "everything here is void. (d) wiped-store [AMENDED 2026-08-13 "
                 "from 'shuffled-diary permuted before retrieval': no admitted "
                 "core retrieves the diary — its rows cross death unread, "
                 "XL.00's certificate — so permuting them cannot change "
                 "behaviour and that control could never fail for the right "
                 "reason; T0.13, a detector that cannot see its own positive "
                 "control has measured nothing]: each arm's twin with weights, "
                 "optimiser and replay reinitialised from the init seed at "
                 "every death must record life_gain within noise of zero — "
                 "cross_life_transfer IS the paired difference against this "
                 "twin, so the control demonstrates the collapse the shuffle "
                 "was meant to buy, on the store the arms actually use. "
                 "(e) darkroom (rewarded for minimising "
                 "predicted observation entropy) must record strongly NEGATIVE "
                 "life_gain — it is the positive control for the dark-room "
                 "detector, and a detector that never sees its own positive "
                 "control has measured nothing (T0.13).",
         kills="Any arm that cannot survive better than a network which has "
               "never received a gradient. Screening declares NO winner — that "
               "is LC.04's job, and separating them is why LT.03/LT.04 are "
               "separate.",
         notes="Headline life_gain = mean survival time over the final third of "
               "lives minus the mean over the first third, per seed. Reported "
               "alongside and gated as a conjunction: n_lives>=12; "
               "needs_satisfied_rate rising; cross_life_transfer > 0; "
               "panel_dwell <= 0.15 per seed (else DISQUALIFIED, PG.4's own "
               "threshold); chaos_occupancy>=3.0 AND chaos_reward_ratio>=2.0 => "
               "VOID for that arm (CURIOSITY_BAKEOFF.md 2.10). Arm wm-efe "
               "additionally VOIDs if its final-third action_entropy falls "
               "below 10% of dreamer-xs's — the epistemic-term collapse "
               "measured in arXiv:2303.01618, where the intrinsic reward stayed "
               "nonzero while coverage collapsed to one action."),

    Spec("LC.04", 5, "The learning core, arbitrated at matched EXPERIENCE",
         hypothesis="Among the arms that cleared LC.03, one core's life_gain at "
                    "exactly N_STEPS decisions of lived experience beats the "
                    "runner-up by >=1.5 sigma of the pooled seed spread.",
         falsified_by="No arm leads by 1.5 sigma => TIE, resolved to the "
                      "cheapest by trainable parameters. That is a real result: "
                      "the choice of learning core does not matter yet and the "
                      "simplest one ships.",
         null_baseline="The shared random null of LC.03, same seeds, same "
                       "evaluation lives, paired.",
         metric="life_gain_at_matched_experience", budget=Budget.CPU_LONG,
         seeds=3, depends_on=["LC.03"],
         control="Inherits LC.03's five controls, passed to run_bakeoff as "
                 "controls= rather than arms= — a designed-to-fail control "
                 "entered as an Arm would VOID this bakeoff permanently by "
                 "construction (LESSONS.md). A control that CLEARS the learning "
                 "gate inverts the verdict to VOID.",
         kills="Three of four learning cores, and the answer to the owner's "
               "question 'THIS is how it learns'. The winner is PROVISIONAL: "
               "adoption is VOID until UB.9 and UB.11 pass under it "
               "(SYSTEM.md's constitutional unison constraint), and the losers "
               "are NOT deleted until then.",
         notes="ARMS, cost declared in TRAINABLE PARAMETERS of the learning "
               "core, asserted to +-5% against the measured value with VOID on "
               "mismatch: ppo-needs 120841 (measured shape, tuned per "
               "arXiv:2307.03486 — LayerNorm before every dense layer, "
               "normalised value targets — plus L_masked_cross_modal for "
               "admission); ppo-lp ~211000 (+ absolute learning progress, two "
               "value heads); dreamer-xs 1896047 (measured: RSSM 1432160 + "
               "actor/critic 463887, GRU deter 256, 32x8 categoricals, symlog/"
               "twohot/free-bits/unimix/percentile-return-norm); wm-efe "
               "~1900000 + 4 ensemble dynamics heads (dreamer-xs's world model "
               "BYTE-IDENTICAL, only the actor objective differs: expected free "
               "energy with ln C = -d(h) and ensemble information gain). "
               "wm-latent ~1370000 is CONDITIONAL, promoted only if dreamer-xs "
               "clears LC.03. REFERENCE ARM sb3-ppo (~121000) is scored but "
               "INELIGIBLE FOR ADOPTION: if it fails to clear the null the "
               "whole bakeoff is VOID because W0 is not a learnable survival "
               "problem. Cost is parameters and NOT core-seconds on purpose: "
               "compute is already LC.05's axis, and counting it twice would "
               "let the tie-break re-decide the thing LC.05 decides."),

    Spec("LC.05", 5, "The same arms, arbitrated at matched COMPUTE",
         hypothesis="Scored off the SAME stored curves at exactly W_CLOCK "
                    "core-seconds instead of N_STEPS decisions, the LC.04 "
                    "winner still wins by >=1.5 sigma.",
         falsified_by="A different arm wins => SPLIT. Recorded as VOID for the "
                      "core decision and PASS for the finding: sample "
                      "efficiency and compute efficiency point different ways "
                      "at 30 GPU-h/week. Nothing ships; LC.05 re-runs at the "
                      "10x deployment budget to break it.",
         null_baseline="The same random null, scored at the same W_CLOCK.",
         metric="life_gain_at_matched_compute", budget=Budget.CPU_LONG,
         seeds=3, depends_on=["LC.04"],
         control="The two scorings must come from ONE set of runs — each "
                 "arm-seed runs until it has consumed BOTH N_STEPS decisions "
                 "AND W_CLOCK core-seconds, whichever comes later, and both "
                 "axes are recorded. A re-run for the second scoring is an "
                 "ERROR, not a convenience: it would let the arms differ in "
                 "anything other than the ruler.",
         kills="The pretence that there is a neutral single budget. T2.02 "
               "matched env-steps and hid a 16x optimiser-step gap "
               "(LESSONS.md, \"'Matched steps' has more than one meaning\"). "
               "Matching env-steps pre-decides for the world model; matching "
               "wall-clock pre-decides for PPO; so both are pre-registered and "
               "their disagreement is a reportable outcome rather than a "
               "choice made after the numbers exist.",
         notes="Every run records all four budgets — decisions, optimiser "
               "steps, core-seconds (MuJoCo share reported separately) and a "
               "gradient-FLOP estimate — plus a decimated curve of <=200 points "
               "spanning all lives. T2.01 stored curve_seed0[:8], iterations "
               "1-21 of 172, which is why its 'the curve PLATEAUED' claim was "
               "not in the ledger."),

    Spec("LC.06", 3, "The simplicity budget is enforced, not promised",
         hypothesis="The adopted learning core satisfies all four "
                    "pre-registered ceilings: B1 trainable parameters <= "
                    "5,000,000; B2 free hyperparameters <= 25, of which ZERO "
                    "are undocumented in the spec that used them; B3 <= 1,500 "
                    "raw lines in the learning rule and learned model; B4 >= "
                    "5.0 simulated seconds per real second on 3 ARM cores.",
         falsified_by="Any ceiling exceeded. The core is not adopted at that "
                      "size; it is reduced, or the ceiling is raised by the "
                      "procedure in LEARNING_CORE.md 6.4 — a bakeoff in which "
                      "the larger core beats the smaller by >=1.5 sigma at "
                      "matched env-steps AND matched wall-clock — never by "
                      "argument.",
         null_baseline="The shipped codebase as of 2026-08-09, which is what "
                       "the ceilings were written against.",
         metric="simplicity_budget_conjunction", budget=Budget.CPU, seeds=1,
         depends_on=["LC.04"],
         control="THE SHIPPED CODEBASE MUST BREACH ALL FOUR. Measured "
                 "2026-08-09: B1 41,525,008 > 5,000,000 (T1.11); B2 92 "
                 "UnifiedBrainConfig fields + 20 PipelineConfig training knobs "
                 "= 112 > 25; B3 6,114 + 1,220 = 7,334 lines > 1,500; B4 0.17 "
                 "< 5.0 (DIRECTION_AUDIT.md 4.1). A budget checker that cannot "
                 "flag the codebase it was written about is measuring nothing "
                 "(T0.13: a detector that cannot see its own positive control "
                 "has measured nothing). B4 needs this most — nothing in "
                 "experiments/ measures sim-seconds per real second today, so "
                 "the 133x gap went unmeasured until an audit looked.",
         kills="Complexity that has not earned itself. The owner, 2026-08-09: "
               "'it won't be the most complex model that Jack is. It will be "
               "just a system that can learn and get input from every single "
               "sense.' This spec is that sentence with numbers on it, and it "
               "is the guard that makes the 57M-vs-124K lesson unrepeatable "
               "rather than merely remembered.",
         notes="Counting rules, fixed here so they cannot be argued later. A "
               "hyperparameter fixed by a paper STILL COUNTS — DreamerV3's "
               "'one configuration for 150+ tasks' is a claim about tuning "
               "effort, not about count, and count is what determines how many "
               "things can be silently wrong (its configs.yaml is 220 lines "
               "and well over 100 knobs). A default counts twice: the audit "
               "reports both the number of knobs and the number whose value is "
               "never written down in the spec that used them, and the second "
               "number must be ZERO. Frozen perception is excluded from B1 — "
               "it is an input, not a learned parameter — which is what makes "
               "the frozen-swappable-tower principle affordable."),

    # -- PS.01, registered AHEAD of the rest of the PS family on purpose --
    # LC.03 declares depends_on PS.01, and LEARNING_CORE.md 5.6 requires the
    # two to land in one commit or LC.03 is permanently BLOCKED (the UB.1
    # lesson). The PS family stays queue-BLOCKED-ON-CORRECTION for PS.00(c)/
    # PS.02 (NEEDS_AND_DEATH.md 0.2 disproved the drive-cycling exploit);
    # PS.01 is calibration/dynamic-range and is not implicated - cross-checked
    # 2026-08-09 per INTEGRATION_QUEUE.md protocol step 1.
    # Verbatim from PURPOSE_AND_SCAFFOLDING.md 4.4.
    # v2, 2026-08-10 — the PROBE was redesigned, not the world. Attempts 1 and 2
    # stay in the ledger's history (T1.02 precedent); the amendment and the one
    # clause that got EASIER are both spelled out in `notes` below, and the
    # design is INTEGRATION_QUEUE.md's TOP entry, cross-checked 2026-08-10.
    Spec("PS.01", 2, "The drive layer is a real control problem, and a statue loses",
         hypothesis="With PG.8's humanoid, energy and integrity both traverse a "
                    "usable range (10th-90th percentile spread >= 0.3, neither "
                    "pinned at 0 nor at 1) over a 4,500-decision (900 s) MIXED "
                    "FIXTURE probe — random action, scripted platform drops, "
                    "scripted rest — which is itself gated to have exercised "
                    "them (>= 5 damaging impacts and >= 100 resting decisions, "
                    "or the range is unmeasured rather than small); a fall from "
                    "the ladder platform costs 0.10-0.20 integrity on held-out "
                    "runs; floor food alone subsists a body acting at the "
                    "derived duty cycle and does NOT fund constant activity, "
                    "priced against the FULL-STRENGTH drain; and the DO-NOTHING "
                    "policy is strictly dominated: its energy reaches the "
                    "weakness floor strictly inside the observation window "
                    "(< 0.8 x horizon) while a scripted FORAGER fixture's, run "
                    "through the same shipped DriveLayer, never does.",
         falsified_by="A probe that exercised both channels never depletes (the "
                      "drive is inert and cannot pressure anything), or "
                      "flatlines at zero within a minute (no policy can learn "
                      "under it), or the statue is NOT dominated — either "
                      "because no behaviour this world admits stays fed (the "
                      "world is a countdown, not a control problem) or because "
                      "the statue itself survives the window (the dark room is "
                      "a stable optimum and homeostasis will produce a corpse).",
         null_baseline="The playground with the drive integrator disabled: every "
                       "internal variable is constant, so every spread is 0.",
         metric="drive_dynamic_range", budget=Budget.CPU,
         depends_on=["PG.8"], seeds=3,
         control="The do-nothing policy IS the control and it must fail: best "
                 "integrity, worst energy, unable to reach any food, and its "
                 "death must be OBSERVED inside the window rather than "
                 "scheduled at its edge. If doing nothing is survivable "
                 "indefinitely, the calibration is wrong and no homeostatic arm "
                 "can be interpreted.",
         kills="The specific numbers in PURPOSE_AND_SCAFFOLDING.md 2.2-2.3. It "
               "cannot kill the idea, only the parameterisation — which is why "
               "it runs before anything trains and after PS.00.",
         notes="Also measures J_0 (the 95th percentile of the impact channel "
               "under ordinary ground contact) which alpha is calibrated "
               "against, and fixes n and m in the drive function. Every number "
               "in 2.2 is a PROPOSAL until this spec replaces it with a "
               "measurement. "
               "AMENDMENT v2 (2026-08-10, after attempt 2 = FAIL): all three "
               "surviving failures were ONE defect — the probe could not "
               "produce the events the gates were about. A random policy never "
               "climbs, so `i` never moved (spread_i 2.96e-5 while the same "
               "integrator scored 0.161 on a held-out platform fall); a random "
               "policy is not a forager (1.0 items eaten in 600 s), so "
               "`ok_random_survives` demanded that flailing beat resting, i.e. "
               "kappa = 0; and the statue's death at t = 1/b = 600 s was "
               "scheduled at the last sample of a 600 s window and missed by "
               "4.35e-14. FOUR changes, three of them strictly harder: the "
               "range is now gated on the probe having produced the events "
               "(>= 5 damaging impacts, >= 100 resting decisions — a probe "
               "that failed to exercise the variable is now a red entry rather "
               "than a confident 2.96e-5); the horizon is 3,000 -> 4,500 "
               "decisions AND the statue's death must land before 0.8 x it; "
               "and subsistence is priced against the full-strength drain "
               "(mean_power_w_full_strength, e = i pinned at 1) instead of "
               "against the power a body already starved by the shortfall "
               "happens to produce, which is the confound that exonerated "
               "kappa in 2.3. ONE clause got EASIER and it is named here "
               "rather than buried: `ok_random_survives` (a RANDOM policy "
               "outlives the statue) is RETIRED and replaced by "
               "`ok_forager_survives` (a scripted forager fixture does). Its "
               "attempt-2 measurement (0.0) stays in the ledger's history and "
               "is not re-measured. The reason is that PS.01 runs "
               "BEFORE anything trains, so demanding that an untrained flailing "
               "body forage is demanding locomotion the ladder has not built; "
               "5 G-B's actual question is whether the dark room is beaten by "
               "SOME behaviour this world admits. The fixture abstracts "
               "locomotion (food is placed on him when it respawns) and nothing "
               "else — it pays the real drain at the derived duty cycle D* = "
               "0.217 through the shipped DriveLayer, so it verifies unit (a)'s "
               "C2 on the shipped path instead of in arithmetic."
               "  COVERS: hunger/thirst (fixture)"),

    # ══ THE MISSING SENSES — smell, taste, voice ════════════════════════
    #
    # Registered 2026-08-10 in response to OVERSIGHT.md §3.2 (RANK 1 for
    # drift) and FOR THE BUILDER item 7: five senses GOAL.md calls
    # constitutional had ZERO specs among 137, which made them invisible to
    # `run next`, `run blocked`, `run status` and the Review — a capability
    # that was never registered reads as completeness in every organ we have.
    #
    # VERBATIM from docs/research/FROZEN_VS_PLASTIC.md §8.6. No threshold was
    # edited during integration (INTEGRATION_QUEUE protocol step 3).
    #
    # CROSS-CHECK (protocol step 1), run 2026-08-10 over docs/research/*.md and
    # docs/LESSONS.md for `smell|olfact|taste|gustat|voice|vocal`:
    #   - NEEDS_AND_DEATH.md — no conflict; it designs the DRIVES, not the
    #     exteroceptive channels. Its only overlap is the poison/illness
    #     insult TA.01 needs, which it supplies rather than contradicts.
    #   - SURVIVAL_WORLD.md — supplies the world content (§8.7's real cost);
    #     no refutation.
    #   - UNIFIED_BRAIN.md / FROZEN_VS_PLASTIC.md §P2 — REINFORCES: a channel
    #     absent during the early transient may never integrate, so these are
    #     wired at W0 with content arriving at W1. No conflict.
    #   - LESSONS.md — the placebo-channel and blind-probe lessons are already
    #     carried inside these specs' own controls (SM.02 (b), TA.01's
    #     colour-coded variant, VO.01's muted emitter).
    # NOT registered here, and why: PAIN and TEMPERATURE are also 0-of-137 but
    # their designs are NOT free-standing — temperature is SURVIVAL_WORLD.md
    # W.1/W.3 (a whole survival world) and pain is an ARM inside
    # NEEDS_AND_DEATH.md §2.9, explicitly "a live question, not a settled
    # design". Registering either as written would prejudge an open bakeoff.
    # They stay ABSENT and are reported so by `run senses` (T0.20), which is
    # the guard that keeps this hole visible instead of invisible.

    # ── SM: smell ───────────────────────────────────────────────────────

    Spec("SM.01", 2, "The odour field obeys its own pre-registered rules",
         hypothesis="An Odour overlay in the Water pattern produces "
                    "concentrations that match the declared field model to "
                    "within 1%: inverse-exponential falloff with distance for "
                    "O1, downwind displacement of the peak proportional to wind "
                    "speed for O2, and non-zero concentration at a receiver "
                    "with NO line of sight to the source (odour passes "
                    "occlusion; light does not).",
         falsified_by="Concentration at an occluded receiver is zero, or the "
                       "wind term does not move the peak - then the field is a "
                       "distance sensor wearing the word 'smell' and no value "
                       "test built on it means anything.",
         null_baseline="A receiver at the same distance with the source "
                       "DISABLED must read the noise floor.",
         metric="field_rule_max_deviation", budget=Budget.CPU, seeds=3,
         depends_on=["PG.1"],
         control="A DELIBERATELY BROKEN variant (wind term dropped) must be "
                 "CAUGHT by the same assertions - else the fixture checker is "
                 "blind and its pass means nothing (the PG.5 precedent).",
         kills="SM.02 and SM.03. A value test on a leaky or trivial field "
               "measures the field.",
         notes="ARMS for the field model, decided by cost since all three can "
               "satisfy the rules above: O1 static exponential (free, O(sources) "
               "per sample); O2 + analytic drifting plume + one mj_ray per "
               "source per sample for occlusion; O3 Farrell-style filaments for "
               "TURBULENT INTERMITTENCY. O1 is the control that must be beaten "
               "in SM.02: if O1 is as good as O2/O3, smell is a distance sensor "
               "and the intermittency literature does not apply to us. Sampled "
               "at 5 Hz (inside the 4-12 Hz mammalian sniff band). C=4 channels "
               "- food, decay, smoke, water - tagged per source, never "
               "chemistry (the caveman standard). Two receiver sites, left and "
               "right of the head, so bilateral comparison is available. "
               "MEASURE the O3 cost before adopting it; O1/O2 are expected to "
               "sit near the fire CA's measured 0.06% of one core."
               "  COVERS: smell (fixture)"),

    Spec("SM.02", 4, "Smell finds what vision cannot see",
         hypothesis="A Jack with the odour modality reaches OCCLUDED food in "
                    "fewer simulated seconds than an identical no-smell twin, "
                    "AND shows no significant advantage when the same food is "
                    "in plain sight.",
         falsified_by="No advantage when occluded (smell is decorative), OR an "
                      "EQUAL advantage when visible (the channel is helping for "
                      "some reason other than occlusion - extra capacity, a "
                      "distance cue, or a leak).",
         null_baseline="The no-smell twin, identical in every other respect "
                       "including token count; PLUS a PLACEBO odour channel of "
                       "matched dimension carrying noise.",
         metric="occluded_minus_visible_advantage", budget=Budget.GPU, seeds=3,
         depends_on=["SM.01", "PG.6"],
         control="TWO that must fail. (a) SHUFFLED FIELD: odour concentrations "
                 "drawn from a different episode's source layout must give NO "
                 "advantage. (b) The PLACEBO channel must give no advantage. "
                 "And ONE that must pass: with the occluder removed the smell "
                 "and no-smell twins must be statistically indistinguishable.",
         kills="The odour modality. A sense whose ablation column is "
               "placebo-indistinguishable loses its parameters (Tier-3), and "
               "this document carves no exception for a constitutional sense - "
               "constitutional means it EXISTS, not that it is exempt from "
               "earning its wiring.",
         notes="The conditional IS the claim, and it mirrors the measured shape "
               "of the audio result (ManiWAV, Audio-VLA: audio pays when vision "
               "is occluded or ambiguous and approximately nothing otherwise). "
               "A test that only measures the occluded condition cannot "
               "distinguish 'smell works' from 'an extra channel helped'."
               "  COVERS: smell (claim)"),

    # ── TA: taste ───────────────────────────────────────────────────────

    Spec("TA.01", 2, "The poison fixture: sub-lethal first dose, visually identical twin",
         hypothesis="Two plant types are IDENTICAL to a visual probe (a "
                    "classifier on rendered frames is at chance) and DISTINCT "
                    "to the taste vector; the toxic one produces a delayed, "
                    "SURVIVABLE interoceptive insult on a first small dose, "
                    "following a declared dose-response curve.",
         falsified_by="A visual probe distinguishes them above chance (then "
                      "TA.02 is a colour-discrimination task), or the first "
                      "dose is lethal (then there is nothing to learn from - "
                      "one-trial learning requires surviving trial one).",
         null_baseline="Chance for the visual probe over the declared "
                       "candidate set.",
         metric="visual_probe_accuracy_x_first_dose_survival",
         budget=Budget.CPU, seeds=3, depends_on=["PG.6"],
         control="A DELIBERATELY COLOUR-CODED variant must be classified WELL "
                 "above chance by the same probe - else the probe is blind and "
                 "its null result is worthless (PG.7's precedent exactly).",
         kills="TA.02.",
         notes="Neophobia rides here: the world must make sampling cheaper than "
               "consuming, and Jack carries a small innate prior toward small "
               "first bites - one of GOAL.md's 'innate reflex priors', finally "
               "used. The delay D between ingestion and illness is declared in "
               "this spec and is the quantity TA.02's difficulty scales with."
               "  COVERS: taste (fixture)"),

    Spec("TA.02", 5, "Conditioned taste aversion: learning from ONE exposure",
         hypothesis="After exactly ONE ingestion of the toxic plant followed by "
                    "delayed illness, Jack avoids that plant on the next "
                    "encounter above a pre-registered rate, and the aversion "
                    "PERSISTS ACROSS A DEATH via the diary.",
         falsified_by="Avoidance at the base rate after one exposure, or "
                       "aversion that does not survive the life boundary. "
                       "Either way the fastest learning in biology has no "
                       "analogue in this system.",
         null_baseline="Base encounter/consumption rate for the SAFE twin; and "
                       "a standard-RL agent with the same reward and no taste "
                       "trace, which is expected to require many exposures - "
                       "the whole point is that a discounted return cannot "
                       "bridge the delay D.",
         metric="one_trial_avoidance_rate", budget=Budget.GPU, seeds=3,
         depends_on=["TA.01"],
         control="FOUR. Three MUST FAIL: (a) CUE-CONSEQUENCE SWAP - pairing the "
                 "illness with an AUDIOVISUAL cue instead of a taste must "
                 "produce no aversion, or far weaker aversion (Garcia & "
                 "Koelling 1966); (b) SHUFFLED TASTE - illness paired with a "
                 "random taste vector must not produce avoidance of the actual "
                 "poison; (c) the PLACEBO taste channel must not support "
                 "aversion. One MUST PASS: pairing an audiovisual cue with a "
                 "FAST external insult (shock-analogue) MUST produce avoidance "
                 "- if nothing one-shot works, (a) failing proves nothing.",
         kills="The taste fast path. If aversion forms equally to any cue, the "
               "mechanism is a generic one-shot memoriser and the "
               "cue-consequence prior - the thing that makes it BIOLOGICAL "
               "rather than a hack - is not there.",
         notes="Control (a) is the single most beautiful control available to "
               "this project: Garcia & Koelling's 1966 design is ALREADY a "
               "control-that-must-fail, published sixty years before this "
               "ladder existed. Standard RL cannot do this task - with gamma<1 "
               "and D of thousands of steps the credit does not arrive - so a "
               "dedicated fast path is REQUIRED, and it is one of the two such "
               "paths this project budgets for (FROZEN_VS_PLASTIC.md 9.4). "
               "VERIFY Garcia & Koelling 1966 and the CTA delay tolerance "
               "against the primary sources before running; both are currently "
               "carried as [k]."
               "  COVERS: taste (claim)"),

    Spec("TA.03", 3, "Taste earns its parameters",
         hypothesis="Ablating the taste modality degrades survival in a world "
                    "containing the visually-identical toxic twin, "
                    "significantly above the PLACEBO column of UB.11.",
         falsified_by="No degradation - taste is decorative and loses its "
                      "wiring (not its constitutional existence: the owner "
                      "ruled the sense EXISTS; this spec decides whether the "
                      "current implementation of it is load-bearing).",
         null_baseline="UB.11's placebo modality column, re-estimated under "
                       "this architecture.",
         metric="taste_ablation_margin_over_placebo", budget=Budget.GPU,
         seeds=3, depends_on=["TA.02", "UB.11"],
         kills="The current WIRING of taste - its tokens, its stem, its fast "
               "path - if the column is placebo-indistinguishable. Not the "
               "sense itself: the owner ruled that it exists.",
         control="In a world with NO toxic plants, the taste ablation must "
                 "produce NO degradation. If removing taste hurts in a world "
                 "where taste is uninformative, the matrix is measuring "
                 "capacity rather than information.",
         notes="Registered so that a constitutional sense still has to earn its "
               "IMPLEMENTATION. GOAL.md's Tier-3 rule and the owner's decree do "
               "not conflict: the decree says he HAS taste; this says our "
               "wiring of it must do something measurable or be rebuilt."
               "  COVERS: taste (claim)"),

    # ── VO: voice ───────────────────────────────────────────────────────

    Spec("VO.01", 2, "He can make a sound, and it is heard as a sound in the world",
         hypothesis="A policy-driven emission (f0, brightness, amplitude, "
                    "duration) is rendered by ContactAudio into the shared "
                    "stereo stream, is recoverable by a probe on a LISTENER's "
                    "audio input, and attenuates with distance and occlusion by "
                    "the amounts the fixture declares.",
         falsified_by="The emission is not recoverable at the listener, or does "
                      "not attenuate - then it is a wire between two brains "
                      "wearing the word 'voice'.",
         null_baseline="A MUTED emitter: the listener's probe must be at "
                       "chance.",
         metric="listener_recovery_x_attenuation_error", budget=Budget.CPU,
         seeds=3, depends_on=["PG.5"],
         control="A listener BEHIND A WALL must hear it attenuated by the "
                 "declared amount, and a listener with the emitter muted must "
                 "hear nothing above the noise floor.",
         kills="Any emergent-signalling claim, and the two-way half of the "
               "talkative-parent design (FROZEN_VS_PLASTIC.md 10.5, 10.7).",
         notes="Cheapest constitutional gap in the audit: ContactAudio "
               "synthesises in microseconds per event and the path already "
               "exists. The action space grows by 4 dimensions. Deliberately "
               "NOT a symbolic channel - an emergent protocol must survive "
               "distance, occlusion and the listener's own encoder, and its "
               "information content must be measurable AT THE EAR."
               "  COVERS: hearing (sensor), voice (sensor)"),

    Spec("VO.02", 5, "Do two Jacks invent a signal? (gated on a second Jack)",
         hypothesis="With two Jacks in one world and a coordination problem "
                    "that pays only if they act differently, the mutual "
                    "information between an emitter's acoustic output and the "
                    "referent, ESTIMATED AT THE LISTENER'S EAR, rises above the "
                    "shuffled-channel floor, and coordination success rises "
                    "with it.",
         falsified_by="Coordination rises while I(signal;referent) at the ear "
                      "stays at the floor - the pair coordinated through "
                      "something other than the signal (position, timing, turn "
                      "count), which is this field's most common false "
                      "positive.",
         null_baseline="THREE, all cheap and all mandatory (arXiv:1903.05168): "
                       "(i) SCRAMBLED MESSAGES - permute the emission before "
                       "delivery; (ii) UNTRAINED COMMUNICATION PARAMETERS - "
                       "never train the emission head; (iii) a MUTED pair. "
                       "Lowe et al. measured speaker consistency essentially "
                       "UNCHANGED under (i) and (ii): 0.202 default vs 0.198 "
                       "scrambled vs 0.171 untrained on the 2x2 game. Any "
                       "metric that cannot separate those three is measuring "
                       "the shared trunk, not communication.",
         metric="ear_mutual_information_over_scrambled", budget=Budget.GPU,
         seeds=3, depends_on=["VO.01"],
         control="POSITIVE LISTENING, not merely positive signalling: the "
                 "causal influence of communication must exceed its floor. In "
                 "Lowe et al., 89.3% (2x2), 97.9% (4x4) and 99.9% (8x8) of "
                 "games sat within 1.02x of the CIC minimum while LOOKING like "
                 "they communicated. Report the floor and the measured value, "
                 "never the value alone. Diagnostic: with SEPARATE emission and "
                 "action networks their speaker consistency collapses from "
                 "0.510 to 0.124 (4x4), which localises the artifact.",
         kills="Every claim that Jack invented a language.",
         notes="BLOCKED ON GEN.02 (a second Jack), and that is the point: a "
               "lone agent has no reason to signal. STAGE IT CHEAPLY - the "
               "floor of this literature is TABULAR: 2 agents, ZERO "
               "parameters, 2 states/2 signals/2 acts, four Polya urns, "
               "Roth-Erev reinforcement, convergence to a signalling system "
               "with probability 1 (Argiento, Pemantle, Skyrms & Volkov 2009), "
               "measured at ~0.2 s of one CPU core for 10^5 plays. Run that "
               "as the harness check first. The 3x3 game converges only ~90.4% "
               "of the time under basic reinforcement (Barrett 2009); "
               "Roth-Erev WITH FORGETTING fixes it to 100% up to 32 symbols at "
               "no extra cost. EXPECT A HOLISTIC PROTOCOL: compositionality "
               "requires a re-learning bottleneck plus an expressivity "
               "constraint (FROZEN_VS_PLASTIC.md 10.6b), not a bigger "
               "vocabulary."
               "  COVERS: voice (claim), social/other agents (claim)"),

    # ── DP: fast and slow, in ONE brain ──────────────────────────────────
    # Owner decree, 2026-08-10: "we must figure that out... and it must still
    # be connected but slightly different purposes... it must be in the
    # research and tests."
    #
    # DIFFERENTIATED FUNCTION, SHARED SUBSTRATE. Two towers with private
    # representations would satisfy "fast and slow" and violate the whole
    # project — GOAL.md's one interconnected brain, and the plastic-only
    # decree with it. So "connected" is not a design preference here; it is a
    # claim that has to be able to fail, which is what DP.02 exists for.
    #
    # Human biology bundles three unrelated fast/slow axes and the ladder must
    # not: fast/slow ACTING (habit vs deliberation, this family), fast/slow
    # LEARNING (hippocampus vs neocortex - ME.7, ME.10, T5.05), and fast
    # SPECIALISED learners (one-shot taste aversion - TA.01, TA.02). Conflating
    # them is how a system ends up claiming a dual process it never tested.

    Spec("DP.00", 2, "This world rewards looking ahead at all",
         hypothesis="There exist states in Jack's world where an agent with a "
                    "PERFECT model and unlimited rollouts beats the best "
                    "reactive policy by a real margin. Deliberation buys "
                    "something here.",
         falsified_by="With a perfect model and unlimited lookahead, planning "
                      "gains nothing over the reactive policy. Then this world "
                      "has no slow system to find, DP.01-DP.03 are "
                      "unregistrable as written, and the finding is about the "
                      "WORLD - it needs traps, delays or irreversibility "
                      "before any dual-process claim can be made in it.",
         null_baseline="The best reactive policy at matched experience.",
         metric="return_gap_oracle_plan_vs_reactive",
         budget=Budget.CPU, seeds=3, depends_on=["LC.02"],
         control="A world variant that is provably reactive-solvable - dense "
                 "immediate reward, no traps, no irreversible states. Planning "
                 "must NOT gain there. If it does, the measured gain is an "
                 "implementation artifact (more compute, more samples, a "
                 "better optimiser) rather than lookahead, and every later DP "
                 "number inherits it.",
         kills="The entire DP family, and the 'spend compute when it matters' "
               "story with it.",
         notes="CHEAPEST FALSIFIER FIRST, per LC.00's precedent. This costs an "
               "oracle rollout, not a training run: give the planner the "
               "simulator itself as its model, so the question is purely "
               "'does lookahead pay in this world' with learning removed as a "
               "confound. Run it BEFORE building any dual-process machinery - "
               "a survival world with hunger, thirst and death is EXPECTED to "
               "reward lookahead (a trap you can see is a trap you can avoid), "
               "but expectation is not evidence and the jungle is not built "
               "yet."
               "  COVERS: fast/slow (fixture)"),

    Spec("DP.01", 3, "Practice moves a behaviour off the deliberative path",
         hypothesis="For a task practised to criterion, the performance cost of "
                    "ablating the deliberative path FALLS with practice - large "
                    "early, small late - while the same ablation on a "
                    "freshly-introduced task stays large. Habit is the learned "
                    "compression of deliberation into reflex.",
         falsified_by="Ablation cost does not fall with practice (nothing "
                      "habituates), OR it falls equally on the never-practised "
                      "task, which means the planner stopped contributing to "
                      "anything and no behaviour migrated.",
         null_baseline="Ablation cost at initialisation, and on a task the "
                       "agent never practised.",
         metric="planner_ablation_drop_early_vs_late",
         budget=Budget.CPU_LONG, seeds=3, depends_on=["DP.00", "LC.04"],
         control="A task whose optimal response CANNOT be cached: the goal is "
                 "re-randomised every episode, so no fixed reaction exists. Its "
                 "ablation cost must NOT fall. Without this control, a planner "
                 "that simply decays into uselessness looks exactly like a "
                 "brain forming habits - the two are indistinguishable from the "
                 "practised task alone.",
         kills="Any claim that Jack forms habits, and the claim that fast and "
               "slow are one system operating at two depths rather than two "
               "systems.",
         notes="THE MEASUREMENT IS A DIFFERENCE OF DIFFERENCES, not a curve. "
               "Report (early - late) ablation cost on the practised task MINUS "
               "the same quantity on the unpractised task; a single falling "
               "curve is consistent with at least three uninteresting stories "
               "(planner decay, entropy collapse, the task getting easier). "
               "Ablate by DISABLING ROLLOUTS, not by zeroing weights: zeroing "
               "shared weights damages the fast path too and would confound "
               "this with DP.02."
               "  COVERS: fast/slow (claim)"),

    Spec("DP.02", 3, "Connected, not two brains: the substrate is shared",
         hypothesis="Fast and slow read the SAME representation. A lesion to the "
                    "shared trunk degrades BOTH modes together; a lesion to the "
                    "deliberative head degrades ONLY the slow mode.",
         falsified_by="A trunk lesion that damages one mode while sparing the "
                      "other. That is the signature of two systems with private "
                      "representations - two brains wearing one wrapper - and it "
                      "refutes the owner's 'must still be connected' directly.",
         null_baseline="A random lesion of equal magnitude at a matched layer.",
         metric="lesion_dissociation_index",
         budget=Budget.CPU_LONG, seeds=3, depends_on=["DP.01"],
         control="A DELIBERATELY SEPARATED architecture - two towers, no shared "
                 "parameters - must show the dissociation this one must not. A "
                 "connectedness test that cannot detect a genuinely "
                 "disconnected system is measuring nothing, and this is the "
                 "only arm that proves it can.",
         kills="GOAL.md's one-interconnected-brain claim as it applies to "
               "action selection. If refuted, either the architecture changes "
               "or the goal statement does - not silently, and not both ways.",
         notes="THE DIRECTION OF THIS TEST IS UNUSUAL AND IT IS THE POINT. "
               "ME.10 is a double dissociation used to prove two capacities are "
               "SEPARABLE; this one is used to prove two modes are NOT. Same "
               "instrument, opposite verdict, so the control matters more than "
               "usual: without the separated-tower arm, 'both degraded "
               "together' is equally consistent with a lesion that was simply "
               "too big to be selective. Report the magnitude at which the "
               "separated control DOES dissociate, and use a lesion no larger."
               "  COVERS: one brain / unison (claim), fast/slow (claim)"),

    Spec("DP.03", 4, "Deliberation is spent where it pays",
         hypothesis="The slow path is engaged more in novel, ambiguous or "
                    "high-stakes states and less in familiar safe ones, and "
                    "that gating loses less return per unit of compute than any "
                    "fixed policy of when to think.",
         falsified_by="Engagement uncorrelated with novelty or stakes, or a "
                      "RANDOM gate at the same average rate matching it. "
                      "Thinking sometimes is not the claim; thinking at the "
                      "right times is.",
         null_baseline="Always deliberate; never deliberate; and - the null "
                       "that actually bites - deliberate at random with the "
                       "SAME average rate and the same compute.",
         metric="return_per_flop_vs_matched_random_gate",
         budget=Budget.CPU_LONG, seeds=3, depends_on=["DP.01"],
         control="A world stretch where nothing is novel and nothing is "
                 "dangerous. Engagement must NOT rise there. If it does, the "
                 "gate is reading elapsed time, episode index or its own "
                 "uncertainty drift rather than the world.",
         kills="The 'one brain that spends more compute when the situation "
               "warrants it' design. If refuted, a fixed deliberation budget is "
               "the honest default and the adaptive gate should be deleted "
               "rather than kept as decoration.",
         notes="MATCHED-RATE RANDOM IS THE ONLY NULL THAT MATTERS and it is "
               "routinely omitted in this literature: any gate that fires often "
               "enough will beat never-deliberating, which proves deliberation "
               "helps and says nothing about the gate. Report compute in FLOPs "
               "or wall-clock, never in 'number of deliberation events' - a gate "
               "that thinks rarely but deeply is not cheaper. Interaction with "
               "DP.01 is expected and must be reported, not controlled away: as "
               "habits form, a working gate should fire LESS on practised "
               "tasks, which is the same phenomenon seen from the other side."
               "  COVERS: fast/slow (claim)"),

    # ── OP: a thing that goes behind something still exists ──────────────
    # Found by LOOKING through the eye rather than by reading its numbers
    # (2026-08-10): the eye sits 0.8 m behind the ladder and 25.6% of Jack's
    # visual field is rungs. PG.6 handles that correctly for a SENSOR test - it
    # rejects occluded samples geometrically and reports occluded_frac 0.32 -
    # but note what that means: a third of this world is currently treated as
    # measurement noise to be discarded. For a creature it is not noise. It is
    # the single most common perceptual problem in any cluttered world, and
    # every organism that survives one solves it.
    #
    # SEPARATE THE SENSOR CERTIFICATE FROM THE PERCEPTUAL CHALLENGE. PG.6 must
    # keep measuring acuity without confound. This family deliberately keeps
    # the occluder.

    Spec("OP.01", 2, "A thing behind the rail still exists",
         hypothesis="After a moving object passes behind an occluder, a linear "
                    "probe on Jack's internal state recovers its CURRENT "
                    "position better than a snapshot of where it was last seen. "
                    "He carries an object forward, not a photograph of its "
                    "disappearance.",
         falsified_by="The probe does no better than the last-seen-position "
                      "predictor. Then what persists is a memory of a vanishing "
                      "event, not a belief about a thing, and every later claim "
                      "about objects - affordances, tool use, the survival "
                      "world's hidden food - is about visible objects only.",
         null_baseline="The snapshot predictor: the object frozen at its "
                       "last-seen position. Plus a shuffled state/label pairing.",
         metric="occluded_position_error_vs_snapshot",
         budget=Budget.CPU, seeds=3, depends_on=["PG.6", "LC.03"],
         control="TWO, and the first is the one that catches a fake. (a) A "
                 "STATIC occluded object, where the snapshot IS optimal - the "
                 "probe must NOT beat it, or the 'extrapolation' is a bias in "
                 "the probe rather than a belief in Jack. (b) No object at all: "
                 "the probe must not report a phantom.",
         kills="Object permanence, and with it any survival-world claim that "
               "depends on remembering where food, water or a predator went.",
         notes="THE STATIC CONTROL IS THE WHOLE SPEC. A probe with any inertial "
               "prior will beat a snapshot on a moving object for reasons that "
               "have nothing to do with Jack, and that is precisely how a "
               "permanence claim gets made falsely. Do not sample occluders "
               "uniformly - PG.6's lesson - and do not reuse its rejection "
               "filter here: this spec wants the samples that one throws away. "
               "The violation-of-expectation design (remove the object while it "
               "is hidden, measure surprise when the occluder lifts) is the "
               "stronger infant-research instrument and needs a predictive core; "
               "register it as OP.02 if LC.04 adopts a world model."
               "  COVERS: sight (claim)"),

    Spec("DP.04", 5, "The slow path may be verbal, and that is a claim, not a design",
         hypothesis="Given a channel to emit and re-hear his own utterances, "
                    "Jack's performance on lookahead-demanding tasks improves "
                    "beyond a MATCHED-COMPUTE control, and the improvement "
                    "scales with each task's deliberation demand as measured "
                    "independently by DP.00's oracle-planning gap.",
         falsified_by="No gain over matched-compute filler, or equal gain on "
                      "tasks with zero planning demand. Either way the words "
                      "are decoration on extra computation and he is not "
                      "thinking in language, whatever the transcript looks like.",
         null_baseline="The SAME extra internal steps carrying content-free "
                       "tokens at matched FLOPs - the filler-token null. This "
                       "is the null the claim lives or dies on.",
         metric="lookahead_gain_over_matched_compute_filler",
         budget=Budget.GPU_SHORT, seeds=3, depends_on=["DP.00", "VO.01"],
         control="A SCRAMBLED VOCABULARY arm: his own tokens permuted by a "
                 "fixed random map, so the channel carries identical statistics, "
                 "identical bandwidth and identical compute but no learned "
                 "meaning. It must NOT help. And the mute arm must still "
                 "deliberate: if removing the verbal channel destroys lookahead "
                 "entirely, language became load-bearing for thought, which "
                 "contradicts one brain with all senses and a Jack who could "
                 "think before he could speak.",
         kills="Any claim that Jack reasons IN language rather than merely "
               "producing it. Also kills the reading of DP.03 in which the slow "
               "path is assumed verbal.",
         notes="THIS IS CHAIN-OF-THOUGHT, ASKED HONESTLY. In LLMs the gain from "
               "a reasoning trace is known to be partly the extra computation "
               "rather than the content - which is why the filler-token null is "
               "mandatory here and why a transcript that 'looks like reasoning' "
               "is not evidence. The Vygotskian reading is what makes this "
               "Jack-shaped rather than borrowed: a caregiver's external speech "
               "is internalised and becomes the medium of deliberation, and this "
               "project ALREADY decided the LLM is his talkative parent living "
               "in his world (GOAL.md, owner 2026-08-09). So the prediction is "
               "specific and falsifiable: inner speech should appear in the "
               "order the parent's speech did, and should carry HIS meanings "
               "attached to his own life, not the parent's. Do not build a "
               "prompt-engineering scratchpad; that would be borrowing the "
               "mechanism instead of testing whether he grows one. "
               "DEPENDENCY NOT YET EXPRESSIBLE: this also requires LG.00 "
               "(language grounding), which is written in the research docs but "
               "NOT registered as of 2026-08-10 - the registry's depends_on "
               "check refused the reference rather than let it dangle. Add "
               "LG.00 to depends_on the moment it is registered; until then "
               "this spec is blocked in fact even though the ladder shows only "
               "DP.00 and VO.01."
               "  COVERS: fast/slow (claim)"),

    Spec("PG.9", 2, "The eye's view is not mostly obstacle",
         hypothesis="Any camera the ladder certifies has less than 5% of its "
                    "frame occupied by geometry nearer than 1 m, and shows at "
                    "least 35% workspace (floor). A certified eye looks AT the "
                    "world, not INTO a nearby object.",
         falsified_by="Near-field occlusion at or above 5%, or workspace below "
                      "35%. Then every visual certificate taken through that "
                      "camera is measuring what fits between obstructions.",
         null_baseline="None meaningful — this is a property of a fixed camera "
                       "pose, measured directly rather than learned. The "
                       "CONTROL carries the falsifiability here.",
         metric="near_field_occlusion_frac",
         budget=Budget.CPU_FAST, seeds=3, depends_on=["PG.6"],
         control="THE HISTORICAL BAD POSE MUST FAIL. Eye at (0,-3.4) looking "
                 "north, which is where it sat on 2026-08-09 - 0.8 m behind the "
                 "ladder, measured 22.2% of frame inside 1 m. Re-rendered and "
                 "asserted to FAIL this spec's own threshold. A view-quality "
                 "test that cannot flag the view that motivated it is "
                 "decoration.",
         kills="Nothing directly. It is a GUARD on PG.6, UB.9-UB.13 and every "
               "later visual spec: those measure acuity and binding through a "
               "camera whose framing they all assume and none of them check.",
         notes="WRITTEN BECAUSE THE LADDER HAS NO EYES. PG.6 passed FIVE times "
               "- R^2 0.97, bearing 1.27 deg, every null and control behaving - "
               "while the camera stared into a ladder 0.8 m away and a quarter "
               "of Jack's visual field was rungs. A human rendering one frame "
               "saw it instantly; 153 specs never could, because exactly one of "
               "them renders an image at all. This spec converts that blind "
               "spot into a number.\n"
               "PER-GEOM COVERAGE WAS TRIED FIRST AND FAILS: in the bad view no "
               "single geom exceeded 15% of frame (the ladder reached 25.9% only "
               "when rails and rungs were summed), so any per-geom threshold "
               "passes it. Near-field depth needs no names and generalises to "
               "worlds nobody has designed yet - which matters, because the "
               "jungle is coming and it is by definition cluttered.\n"
               "THE 5% AND 35% ARE PRE-REGISTERED FROM A MEASURED SEPARATION, "
               "not tuned: bad pose 22.2% / 51.2%, current pose 0.0% / 61.8%. "
               "The gap is wide enough that the threshold sits in empty space "
               "rather than beside either measurement. Do NOT relax it to admit "
               "a future camera; move the camera."
               "  COVERS: sight (fixture)"),

    # ── THE SURVIVAL WORLD'S MISSING PILLARS ─────────────────────────────
    # Coverage audit 2026-08-10. The owner's directive was explicit: permanent
    # human needs, too cold or too hot KILLS, a jungle, he builds a shelter, he
    # dies and retries and REMEMBERS ACROSS LIVES. The ladder had 154 specs and
    # ZERO about thermal death, zero about damage, zero about shelter, zero
    # about anything surviving a death. The whole survival world rested on
    # PS.01 alone, which is FAIL.
    #
    # Everything here is BLOCKED IN FACT until the survival world exists in
    # code. Registered anyway and deliberately: an unregistered intention is
    # invisible to `run blocked`, to the overseer, and to every audit — which
    # is exactly how four constitutional commitments went a week without a
    # single falsifiable claim behind them.

    Spec("PS.02", 2, "The world can freeze him, and the cold is FELT before it kills",
         hypothesis="The world carries a temperature field with pre-registered "
                    "dynamics - body temperature falls at a measured rate in "
                    "cold, rises near heat, death below a threshold within a "
                    "bounded time - AND the approach of that death is legible "
                    "from Jack's senses beforehand: a probe on his sensory "
                    "vector predicts time-to-freezing well above chance while "
                    "he is still alive.",
         falsified_by="Time-to-death unpredictable from the senses. Then cold "
                      "is an unlearnable instakill, not a need: no agent and no "
                      "architecture could ever adapt to it, and every shelter "
                      "result built on top would be measuring luck.",
         null_baseline="A thermally inert variant where temperature never "
                       "moves: nothing may die. And a shuffled probe pairing.",
         metric="time_to_freeze_probe_r2",
         budget=Budget.CPU, seeds=3, depends_on=[],
         control="SILENT LETHALITY: temperature drops exactly as before, but "
                 "the thermal channel is REMOVED from the sensory vector. The "
                 "probe must fail there. Without this the probe could be "
                 "reading the episode clock - every episode gets colder with "
                 "time - and would report a sense that does not exist.",
         kills="Every survival claim involving cold, and the jungle's entire "
               "motive for shelter.",
         notes="A LETHAL NEED YOU CANNOT PERCEIVE IS NOT A CURRICULUM, IT IS "
               "NOISE. This is the half of 'too cold kills him' that the "
               "directive does not say out loud and that decides whether the "
               "world is teachable. Caveman realism (owner, 2026-08-09): he "
               "does not need thermodynamics, he needs cold hurts / fire helps "
               "/ shelter holds heat, consistent and discoverable. Do not model "
               "chemistry. DO pre-register the rate constants and the lethal "
               "threshold before implementing, so the world cannot be quietly "
               "tuned until the agent survives."
               "  COVERS: thermal (kills) (fixture)"),

    Spec("PS.03", 2, "Damage is a signal, not just an ending",
         hypothesis="Harm produces a GRADED, sensed damage signal that precedes "
                    "death, and a single exposure is enough to shift behaviour "
                    "away from its cause.",
         falsified_by="Damage is binary and instant, or avoidance needs many "
                      "exposures. Either way the only way to learn about a "
                      "danger is to die of it, repeatedly, which no animal "
                      "does and no agent in a survival world can afford.",
         null_baseline="A harmless world variant: no avoidance should form.",
         metric="one_exposure_avoidance_delta",
         budget=Budget.CPU, seeds=3, depends_on=[],
         control="A HARMLESS TWIN - an event visually and acoustically "
                 "identical to the damaging one but with no damage. Avoidance "
                 "must NOT transfer to it, or he learned to avoid novelty and "
                 "surprise rather than injury. (TA.01's identical-twin design, "
                 "reused because it is the same failure mode.)",
         kills="Any claim that Jack learns danger. Also weakens TA.02: taste "
               "aversion would be the ONLY one-shot learner in the system, "
               "which would make it a special case rather than a principle.",
         notes="Nociception is not pain-as-suffering; it is the graded signal "
               "that makes danger learnable before it is fatal. Register the "
               "gradation explicitly - a scalar with a range, not a flag - "
               "because a binary damage bit is exactly the unlearnable case "
               "this spec exists to rule out."
               "  COVERS: damage/nociception (claim)"),

    Spec("PS.04", 5, "He eats because he is hungry - feeding is need-contingent",
         hypothesis="A learner in W0 acquires food CONTINGENT on its sensed "
                    "energy state: feeding events concentrate at low sensed "
                    "energy (the need->behaviour coupling beats a need-blind "
                    "account of the same behaviour by >= 3 sigma across seeds), "
                    "and a SATIETY-CLAMPED twin - hunger channel pinned to "
                    "'full' while the body drains identically - measurably "
                    "reduces its food acquisition. The sensed need, not the "
                    "food's mere presence, is what moves him.",
         falsified_by="Feeding is uncorrelated with sensed need, or the "
                      "satiety-clamped twin forages just as much. Then food "
                      "acquisition is a fixed policy the drive layer merely "
                      "narrates, 'hunger' is bookkeeping rather than a drive, "
                      "and the needs-are-the-curriculum premise (GOAL.md) has "
                      "no mechanism in him.",
         null_baseline="A random policy's need-conditioned feeding rate: with "
                       "no policy coupling, feeding is independent of the "
                       "sensed state and the contingency statistic reads its "
                       "chance level.",
         metric="need_contingent_feeding_margin",
         budget=Budget.CPU_DAYS, seeds=3, depends_on=["PS.01", "LC.03"],
         control="THE SATIETY CLAMP, and its cleanliness is a GATED quantity "
                 "(PS.03's lesson): the clamped twin's sensed energy must read "
                 "full at every decision, its true drain must match the "
                 "experiment arm's within tolerance, and no other channel may "
                 "differ - measured, not assumed. The clamp must reduce "
                 "food-seeking; if it does not, the channel is decorative.",
         kills="The claim that W0's needs teach anything. If feeding is not "
               "need-contingent, cold and hunger are punishments, not "
               "curriculum, and the survival-world directive is running on an "
               "agent that cannot be pressured by it.",
         notes="COVERS: hunger/thirst (claim)\n"
               "Registered under overseer B5 (2026-08-13): hunger/thirst had "
               "no claim-kind spec, so its n_pass could not move no matter "
               "what ran. SCOPE, stated honestly: W0 exposes energy (food) "
               "and integrity; there is no water channel yet, so this spec "
               "demonstrates the HUNGER half. When W0 gains thirst, register "
               "a sibling rather than averaging the two (the ME.11 "
               "per-partition lesson). CAUTION carried from LC.03/T2.08: in "
               "W0 as measured, passivity may maximise life LENGTH - this "
               "claim is about CONTINGENCY (need drives behaviour), not "
               "survival optimality, and a need-blind statue outliving a "
               "forager rescues nothing. Depends on LC.03 because the claim "
               "needs a screened learning core that demonstrably learns in "
               "W0; the implementer should reuse LC.03's harness "
               "(experiments/survival.py) rather than a second loop - the "
               "two-kernels lesson."),

    Spec("SH.01", 5, "Under cold, he shelters - and prefers the shelter that works",
         hypothesis="With a thermal drive active, time spent sheltered rises "
                    "far above an otherwise identical agent whose thermal drive "
                    "is disabled, sheltering BEGINS before the lethal threshold "
                    "rather than after it, and when offered two shelters he "
                    "prefers the one that actually retains heat.",
         falsified_by="No difference from the drive-disabled agent; or "
                      "sheltering only starts after the threshold (a reflex to "
                      "dying, not anticipation); or he is indifferent between "
                      "a working shelter and a cosmetic one.",
         null_baseline="The thermal-drive-disabled agent, and a random-walk "
                       "policy with matched time in the arena.",
         metric="sheltered_fraction_vs_drive_disabled",
         budget=Budget.CPU_LONG, seeds=3, depends_on=["PS.02"],
         control="THE COSMETIC SHELTER. Two shelters, visually identical, one "
                 "with the thermal benefit removed. Preference for the working "
                 "one is the whole claim - without it, 'sheltering' is "
                 "indistinguishable from a preference for enclosed spaces, "
                 "which many agents develop for reasons having nothing to do "
                 "with warmth.",
         kills="The owner's own image of success ('throw him in a jungle and "
               "see how he builds a shelter'). If refuted, the honest report is "
               "that we have an agent that survives cold by some other means, "
               "and the shelter story is ours, not his.",
         notes="OCCUPYING BEFORE BUILDING, DELIBERATELY. Construction is a much "
               "harder claim and a much later spec; this one asks whether the "
               "MOTIVE is real and directed, which is the precondition for "
               "building to mean anything. A Jack who builds a shelter he does "
               "not need has learned a trick. Report the anticipation lead time "
               "(seconds between first sheltering and the lethal threshold) as "
               "a first-class metric - it is the difference between foresight "
               "and reflex, and it connects directly to DP.00's question of "
               "whether this world rewards looking ahead."
               "  COVERS: thermal (kills) (claim), shelter/building (claim)"),

    Spec("XL.00", 2, "He dies, he reappears somewhere he did not choose, and the diary crosses",
         hypothesis="With `lethal=True`, W0 ends a life when energy or "
                    "integrity reaches zero at the rate the drive arithmetic "
                    "predicts (a resting body's implied 1/b is within 2% of "
                    "600 s at two independent starting charges); the body "
                    "reappears at a pose drawn UNIFORMLY from the legal spawn "
                    "set (chi-square z <= 4 over 20,000 draws), always legal, "
                    "and statistically INDEPENDENT of where it died "
                    "(two-sided permutation p >= 0.01 on paired-vs-shuffled "
                    "death->spawn distance); the diary survives every death "
                    "with a life index covering every life; and a NON-LEARNER's "
                    "lives do not lengthen across >= 12 lives (two-sided "
                    "permutation p >= 0.01 on the life-length slope).",
         falsified_by="Death never fires; or the implied drain disagrees with "
                      "the arithmetic; or a spawn lands inside geometry; or "
                      "the spawn distribution is non-uniform or correlated "
                      "with the death site - which is `LT` 2.1's objection "
                      "arriving through the respawn, an experimenter-supplied "
                      "curriculum; or the diary does not survive; or the "
                      "non-learner's lives lengthen anyway, in which case "
                      "LC.03's `life_gain` measures the WORLD and every "
                      "learning-core verdict built on it is void.",
         null_baseline="For the trend: the non-learner (uniform random action) "
                       "itself - a random policy's lives may not lengthen. For "
                       "uniformity and independence: the shuffled pairing and "
                       "the flat multinomial, both computed from the run's own "
                       "draws rather than assumed.",
         metric="death_respawn_diary_conjunction",
         budget=Budget.CPU, seeds=3, depends_on=["LC.02", "PS.01"],
         control="FIVE, each on its pre-registered side, and three of them are "
                 "POSITIVE controls for detectors that would otherwise be "
                 "unfalsifiable. (a) IMMORTAL (`lethal=False`, same decision "
                 "budget, same starting charge): deaths must be 0 - a death "
                 "detector that fires in a world without death is reading "
                 "something else. (b) SPAWN-AT-DEATH (`spawn_sampler` returns "
                 "the death site): the independence p MUST fall below 0.001, "
                 "or the statistic cannot see the very leak it exists to "
                 "exclude. (c) BIASED SAMPLER (draws only from the half of the "
                 "legal set nearest the origin): the uniformity z MUST exceed "
                 "4. (d) WIPED DIARY (the store cleared at every death): "
                 "life-0 rows must NOT survive. (e) DRIFTING WORLD (each new "
                 "body starts with more charge than the last): the trend p "
                 "MUST fall below 0.001 - T0.13's rule, a detector that has "
                 "never seen its own positive control has measured nothing.",
         kills="W0-2 and W0-3 as implemented, and with them LC.03/LC.04 - the "
               "learning-core bakeoff scores `life_gain` and "
               "`cross_life_transfer`, neither of which exists if death, the "
               "respawn or the diary is broken. A wrong answer here is not a "
               "wrong answer about the world; it is a wrong answer about every "
               "arm scored in it.",
         notes="THE SHORT-LIFE FIXTURE IS DECLARED, NOT HIDDEN. Every claim "
               "except the drain arithmetic is invariant to the starting "
               "charge, so lives after the first are started at e=0.10 to buy "
               "16 deaths per seed in ~50 s instead of ~16 min. The drain "
               "arithmetic is certified separately IN THIS SPEC at two full "
               "charges, which is a stronger test of it than one death at "
               "e=1.0 (it checks the RATE, not one endpoint). j0 and alpha are "
               "READ FROM PS.01's LEDGER ENTRY, never copied: a calibration "
               "pasted into a second file is a constant that drifts from its "
               "measurement (T0.14). W0-2's random respawn is the answer to "
               "LEARNING_CORE.md 5.0's own objection that an episode boundary "
               "is a free teleport to a good state, so the independence test "
               "is the load-bearing half of this spec, not a formality. "
               "REVISED after the FAIL of 2026-08-10T10:40, which stays in the "
               "ledger's history: the two permutation gates were z-scores at "
               "3.0, and a permutation z for a linear statistic is bounded by "
               "sqrt(n-1), so at the drift control's n=9 lives the ceiling is "
               "2.83 and the gate was UNREACHABLE - it measured the sample "
               "size, not the trend. Rank p-values have no such ceiling and "
               "are STRICTER here (|z|<=3 admits out to p~0.003; the gate "
               "rejects at 0.01). Same run: the occupied-pose known-answer "
               "probed the point BETWEEN the ladder rails, whose penetration "
               "depends on per-seed mutated geometry, and 1 of 3 seeds "
               "disagreed; it now reads `ladder_railL`'s own position off the "
               "live model. Both are T1.02 repairs - the experiment was "
               "wrong, and neither touched W0-2 or W0-3."
               "  COVERS: death & retry (fixture), memory across lives (fixture)"),

    Spec("XL.01", 5, "Death does not erase what he learned",
         hypothesis="A life that follows earlier lives reaches a survival "
                    "criterion faster than the first life did, and faster than "
                    "a life whose carried memory was wiped at death.",
         falsified_by="No speedup across lives, or the memory-wiped control "
                      "speeds up just as much - in which case the improvement "
                      "lives in the world or the curriculum, not in him, and "
                      "'he remembers across lives' is a description of our "
                      "bookkeeping rather than of Jack.",
         null_baseline="First-life learning curve; and the memory-wiped arm.",
         metric="lives_to_criterion_vs_wiped",
         budget=Budget.CPU_LONG, seeds=3, depends_on=["PS.02", "XL.00"],
         control="ANOTHER JACK'S MEMORIES. Carry a different agent's store into "
                 "the new life: it must NOT help, and should hurt. ME.3's "
                 "precedent - reflections generated from another agent's log "
                 "must hurt - and the same reason: a memory that helps "
                 "regardless of whose life it came from is not memory, it is a "
                 "prior.",
         kills="The owner's survival-world directive at its core. Without this, "
               "death is merely punishment and retry is merely a reset - the "
               "loop would be running an agent that suffers consequences it "
               "cannot accumulate.",
         notes="REPORT WHAT SURVIVED, SEPARATELY. Weights and the episodic "
               "store are different claims and the aggregate hides which one "
               "carried: a system where only weights survive is 'trained by "
               "many deaths', which is ordinary RL; the owner asked for "
               "something that REMEMBERS. Run the two ablations (weights "
               "carried / store wiped, and the reverse) and report both, "
               "because the interesting answer is almost certainly that they "
               "carry different things - the complementary-learning-systems "
               "split (ME.10) predicts exactly that, and this is the first "
               "spec that could show it in a lifetime rather than a session."
               "  COVERS: death & retry (claim), memory across lives (claim)"),

    Spec("BA.01", 2, "He feels himself falling before he falls",
         hypothesis="Jack carries a sensed orientation signal - gravity's "
                    "direction in his own body frame - from which a linear "
                    "probe recovers tilt, and from which time-to-topple is "
                    "predictable while he is still upright.",
         falsified_by="Tilt unrecoverable, or a topple unpredictable until it "
                       "has happened. Then balance is not a sense he has, it is "
                       "an outcome he suffers, and no amount of training "
                       "produces a creature that catches itself.",
         null_baseline="Chance for tilt; and a predictor that only knows "
                       "elapsed time in the episode.",
         metric="time_to_topple_probe_auc",
         budget=Budget.CPU, seeds=3, depends_on=[],
         control="Remove the orientation channel from the sensory vector and "
                 "leave the physics identical. The probe must fail. Without it "
                 "the probe may be reading the episode clock - falls cluster "
                 "late - and would report a sense that is not there. (Same "
                 "design as PS.02's silent-lethality control, and for the same "
                 "reason.)",
         kills="Every locomotion and climbing claim that assumes he can tell "
               "up from down. W0.BAL - 'the rover topples' - has been an open "
               "queue entry rather than a spec; this is the falsifiable form of "
               "it.",
         notes="THE LAST UNCOVERED SENSE. Found by experiments/coverage.py on "
               "2026-08-10: balance was the one commitment in GOAL.md with zero "
               "specs, which is why it went unnoticed while a related problem "
               "sat in the integration queue as prose. A sense with no spec is "
               "invisible to every instrument the system owns. COVERS: balance (sensor)\n"
               "Vestibular in animals is not one signal but two - linear "
               "acceleration (otoliths) and angular velocity (canals). "
               "Register BOTH channels and report them separately; a system "
               "given only gravity's direction cannot distinguish falling from "
               "being carried, and that distinction is exactly what a creature "
               "in a jungle needs."),

    Spec("BA.02", 5, "He catches himself - the felt fall changes what he does",
         hypothesis="A learner given BA.01's vestibular channel ACTS on it: "
                    "trained in a topple-costly regime, it stays upright "
                    "measurably longer than an identical learner trained with "
                    "the channel deleted (>= 3 sigma across seeds), and the "
                    "gain vanishes when the channel is replaced by "
                    "matched-statistics noise.",
         falsified_by="No upright-time gain from having the channel. Then "
                      "balance is decoded but not used - a sense he has and "
                      "ignores - and BA.01's probe measured a spectator, not "
                      "a participant in control.",
         null_baseline="The channel-deprived twin's upright time; and a "
                       "random policy in the same rig.",
         metric="upright_gain_vs_deprived",
         budget=Budget.CPU_LONG, seeds=3, depends_on=["BA.01"],
         control="MATCHED-NOISE CHANNEL: replace the vestibular input with "
                 "amplitude-matched noise (or a shuffled replay of another "
                 "episode's channel). The gain must vanish - this separates "
                 "information content from input-width or regularisation "
                 "effects, which a deprived-vs-present comparison alone "
                 "cannot. BA.01's own control (channel removed, physics "
                 "identical) is inherited by the deprived arm.",
         kills="The assumption, load-bearing in every locomotion and "
               "climbing claim, that he can use up-from-down. If refuted, "
               "balance stays a dashboard light he never reads, and the "
               "honest status of W0.BAL is 'sensed, unused'.",
         notes="COVERS: balance (claim)\n"
               "Registered under overseer B5 (2026-08-13): balance had no "
               "claim-kind spec, so its n_pass could not move no matter what "
               "ran. BA.01's two-channel note applies here as ablation "
               "structure: report the linear-acceleration and "
               "angular-velocity channels' contributions SEPARATELY (ME.11's "
               "per-partition lesson) - a gain carried wholly by one channel "
               "is a finding, not a rounding detail. BUDGET is declared "
               "CPU_LONG on the expectation that BA.01's rover rig trains "
               "far under LC.03's envelope; per the LC.03 budget scar, the "
               "implementer must re-cost the envelope at measured throughput "
               "in the pilot and amend the TIER (never the thresholds) "
               "BEFORE the registered run if it does not fit."),
]
