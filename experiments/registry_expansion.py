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
                    "other file, and a reattach never routes to Colab. Nor may "
                    "the way home REWRITE the answer's provenance: a reattach "
                    "recovers a kernel that ran the ORIGINAL submission's "
                    "code, so the attempt receipt records the pushed kernel's "
                    "sha256 and a reattach whose local script hashes "
                    "differently is refused (or, explicitly tolerated, states "
                    "the divergence in the receipt log and the ledger row).",
         falsified_by="A Kaggle JobResult with an empty stdout when a log was "
                      "downloaded; the log appearing in `artifacts`; "
                      "`result_json` returning a file it was not asked for; "
                      "`submit` calling Colab while JACK_REUSE_KERNEL is set; "
                      "any test file json.loads-ing an `.artifacts` entry "
                      "directly (the path, not the file — the TA.02 scar of "
                      "2026-08-19); or `reattach_code_check` missing a planted "
                      "kernel-sha divergence, refusing an identical-code "
                      "reattach, or calling a pre-guard receipt a mismatch "
                      "(the impl_sha laundering scar, overseer 20th-audit B1).",
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
         hypothesis="Amend-after-adverse-verdict is auditable by someone who "
                    "is not its author, mechanically: (1) a verdict that "
                    "supersedes a FAIL or a VOID carries the prior evidence "
                    "IN the record (`supersedes_fail`/`supersedes_void`: "
                    "commit, dirty flag, impl_sha, measurement, source "
                    "status, plus machine-readable `impl_changed`), and the "
                    "pairing survives into history when superseded again; "
                    "(2) `audit_supersedes_fail` flags, in any PASS record, "
                    "a FAIL or VOID whose implementation differs from the "
                    "run that amended it unless it is stamped at a clean "
                    "commit that exists in this repo and carries its "
                    "metrics — pairing across intervening ERROR rows, which "
                    "are infrastructure events, not verdicts (widened "
                    "2026-08-20, 22nd audit B1/B2, strengthen-only); (3) "
                    "the LIVE ledger has zero such violations; (4) "
                    "pre-impl_sha pairs read unauditable, never violated — "
                    "absence is a historical gap, not evidence.",
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
         control="The same shipped action path trained identically on a "
                 "SHUFFLED (obs, action) pairing must NOT beat the "
                 "nearest-neighbour null. If information-free supervision "
                 "beats real retrieval, the metric is not measuring imitation "
                 "but the marginal statistics of the action set.",
         metric="heldout_vs_nn_ratio", budget=Budget.GPU, seeds=3,
         depends_on=["T1.13", "T1.08"]),

    Spec("T2.15", 2, "Free-form language routes to the right task",
         hypothesis="Novel paraphrases of known commands map to the correct "
                    "command cluster above chance (the LLM->task handoff).",
         falsified_by="Held-out phrasings route at chance.",
         null_baseline="Chance routing; bag-of-words retrieval.",
         metric="paraphrase_routing_accuracy", budget=Budget.GPU_SHORT, seeds=3,
         depends_on=["T2.06"],
         control="A label-shuffle twin (LAW 2): the designed grid's "
                 "phrase->cluster supervision composed with a fixed "
                 "derangement at the single supervision site. Its held-out "
                 "routing vs TRUE clusters must NOT reach the claim bar on "
                 "any seed — and its loss must fall, proving the twin "
                 "trained — else the ruler leaks and the run is VOID.",
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
         metric="cued_recall_accuracy", budget=Budget.CPU, seeds=3,
         control="A query about a FABRICATED event must abstain — confabulating "
                 "a match means the retrieval threshold is broken.",
         notes="seeds 1 -> 3 by the Review 2026-08-30 (Part 2), with ME.2-ME.4, "
               "on the ME.8 precedent. All four recorded PASS at seeds=[0] on "
               "2026-08-08 and were never reconsidered; each compares a "
               "MECHANISM against a baseline, which GOAL.md's '>=3 seeds where "
               "the claim is about learning' covers, and a single seed cannot "
               "report the +-std that says whether the gap exceeds seed noise. "
               "No threshold moved; every gate is >= 0.02 so protocol.py's "
               "'_aggregate kills sub-5e-7 gates at 3 seeds' hazard does not "
               "apply. Re-run: cued_recall 0.8667 (1 seed) -> 0.8500 +- 0.0136 "
               "(3), still clear of the 0.80 bar, and the headroom is now "
               "measured instead of assumed."),

    Spec("ME.2", 2, "Owner memory lives on disk",
         hypothesis="A preference stated once is honoured next session; a later "
                    "contradiction supersedes it.",
         falsified_by="Adherence <= a fresh no-memory agent's base rate.",
         null_baseline="No-memory agent; recency window excluding the preference.",
         metric="preference_adherence", budget=Budget.CPU, depends_on=["ME.1"],
         seeds=3,
         control="WIPE profile.json and restart: adherence must drop to base "
                 "rate — proving memory is in the file, not weights or cache.",
         notes="seeds 1 -> 3 by the Review 2026-08-30 (Part 2) — see ME.1. "
               "Re-run: recency_null_adherence 0.075 (1 seed) -> 0.1667 +- "
               "0.0656, wiped control 0.175 -> 0.2167 +- 0.0312. Both stay "
               "under the 0.45 null ceiling, but the single seed was reporting "
               "the null a factor of two kinder than it is."),

    Spec("ME.3", 2, "Reflections beat raw events",
         hypothesis="Aggregation questions answer better from consolidated "
                    "reflections than from top-k raw events at equal tokens.",
         falsified_by="No gain over raw top-k.",
         null_baseline="Raw-events-only retrieval.",
         metric="aggregation_qa_gain", budget=Budget.CPU, depends_on=["ME.1"],
         seeds=3,
         control="Reflections generated from ANOTHER agent's log must hurt.",
         notes="seeds 1 -> 3 by the Review 2026-08-30 (Part 2) — see ME.1. "
               "Re-run: wrong_agent control 0.1771 (1 seed) -> 0.2708 +- "
               "0.0981, i.e. the control's own spread is the largest number in "
               "this spec's record and only three seeds could show it."),

    Spec("ME.4", 2, "Forgetting keeps what matters",
         hypothesis="Ebbinghaus decay + reinforce-on-recall + supersede beats "
                    "FIFO eviction at a fixed store budget.",
         falsified_by="FIFO matches it on frequently-referenced old facts.",
         null_baseline="FIFO; unbounded store as ceiling.",
         metric="retention_vs_fifo", budget=Budget.CPU, depends_on=["ME.1"],
         seeds=3,
         control="Knowledge-update questions must FAIL in the no-supersede "
                 "variant (stale answers) — else the questions never conflicted.",
         notes="seeds 1 -> 3 by the Review 2026-08-30 (Part 2) — see ME.1. "
               "Re-run: every metric identical at 3 seeds, std 0.0 throughout "
               "(retention 1.0 vs FIFO 0.0). That is the honest reading and it "
               "is a WEAKNESS worth naming, not a strength: a comparison whose "
               "seed variance is exactly zero is deterministic by construction "
               "and its 1.0-vs-0.0 gap is arithmetic, not evidence about "
               "forgetting. See PROGRESS.md 2026-08-30 FOR THE BUILDER — the "
               "redesign (recall-frequency confound, graded budget pressure) "
               "is proposed there and NOT taken here, because it is a "
               "redesign and not a strengthening."),

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
                 "provenance and is measuring text similarity.",
         notes="STRENGTHENED by the Review 2026-08-30 (Part 2), three gates "
               "ADDED, none moved. The 2026-08-08 PASS read 1.0000 +- 0.0000 "
               "on all three channels because two of its three scored "
               "conjuncts were true by construction: the query API "
               "(`recall(cue, channel=, speaker=)`) filters on provenance "
               "BEFORE scoring, so `got.channel == channel` and `got.speaker "
               "== speaker` cannot fail. Both existing references die of the "
               "filter rather than of the scoring — the pooled null has no "
               "filter, the swap control empties it — so nothing in the spec "
               "could distinguish real retrieval from a coin flip inside the "
               "filter. Added: provenance-KEPT/scoring-STRIPPED reference "
               "(most recent event within the identical filter, identical "
               "predicate) at <= 0.25, a scoring_margin >= 0.50, and a "
               "pooled-null MARGIN of >= 0.40 in place of merely sitting under "
               "the pass bar. Re-run 2026-08-30: filtered_recency_worst "
               "0.1250, scoring_margin 0.9056 — the headline is unchanged and "
               "now says what it costs."),

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
               "  CONDITIONAL CLAIM (26th audit B3, 2026-08-24): the unimodal "
               "null arms carry no must-learn target of their own and their "
               "loss descent is not recorded, so their at-chance readings rest "
               "on the SHARED-TRAINER argument alone — the same training code "
               "drives the fused arm past FUSED_GATE in the same seed, and "
               "vision_carries_bit/audio_carries_bit >= 0.90 prove each arm's "
               "input features decodable where signal exists. A PER-ARM recipe "
               "pathology (UB.10's measured disease: one uniform recipe leaving "
               "one matched-param arm dead) is NOT excluded by this design; the "
               "2026-08-12 PASS is conditional on that argument until a UB.9 "
               "re-run records per-arm loss descent or a same-run must-learn "
               "target per unimodal arm."
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
               "  COVERS: one brain / unison (claim). "
               "PARKED 2026-08-20 (recipe probe both-fail: no single uniform "
               "recipe trains all six matched-param arms; A2 learned its "
               "marginals under NO tested recipe), UNPARKED 2026-09-01 under "
               "the Review's 2026-08-25 disposition: matched TUNING BUDGET, "
               "not matched hyperparameters — every arm gets the IDENTICAL "
               "pre-registered recipe grid (K=5, declared in the test file "
               "before any grid trial), the same trial count, selection by "
               "the same pre-registered arm-local criterion that never reads "
               "the claim metric; an arm eligible NOWHERE on the grid is "
               "SCORED-AND-INELIGIBLE (runs, is recorded, cannot win, "
               "carries no verdict conjunct), never a silent 0.5. A0 "
               "ineligible or zero eligible trunk arms -> VOID. Strictly "
               "harder than the uniform recipe it replaces; cost N -> NxK."),

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
         metric="creative_contribution", budget=Budget.CPU_LONG, seeds=3,
         control="WRONG-GOAL CONSULT. The identical trained loop, wired at "
                 "the identical stuck-recovery call site, is handed a goal "
                 "reflected through the rover (2*xy - goal): exactly as much "
                 "information, exactly wrong. It must NOT buy the improvement "
                 "the claim needs (off - shuf must stay under the same "
                 "margin); if wrong-goal advice helps as much, the site "
                 "rewards any detour perturbation and the test measures "
                 "nothing.",
         kills="AlphaGeometryLoop.py (559 lines) — wire it or delete it.",
         notes="seeds 1 -> 3 declared 2026-09-02 (61st audit B1.3), before "
               "any further run: a verdict that arms a 559-line deletion, on "
               "a metric whose four arms ranked anti-correlated with advice "
               "quality, is not decided at one seed. The attempt-3 FAIL "
               "(seed [0], 06:33) predates this and its own wrong-goal "
               "control cleared the claim's margin (shuf_gain +12.47 vs "
               "MARGIN_AFF 11.0) — under the corrected lane ordering that "
               "row is a VOID, the kills clause was NOT executed, and the "
               "row is routed as t309-control-clears-the-claims-own-margin."),

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
         notes="Cheapest direct evidence for/against decision D1 (arXiv:2505.23705).\n"
               "PARKED: 2026-08-30 — REPAIR 1's pilot fired the pre-registered "
               "both-fail branch (fork (ii)) on TWO independent grounds, and "
               "the one-diagnostic cap (SM.02/UB.10 precedent) is SPENT. No "
               "gate has moved and none may. Colab T4, seed 90, ~9 min. "
               "(1) THE CLAIM MISSES BY 5x: with `null_admissible` working "
               "exactly as specified — colour (probe_random 0.9245) and near "
               "(0.9427) dropped as unreachable, `shape` retained, "
               "n_null_admissible 1 — knowledge_margin_min read 0.0299 "
               "against the frozen +0.15 bar. (2) THE RIG CONTROL STOPPED "
               "FIRING: probe_drift_unfrozen fell 0.1875 -> 0.0078 against its "
               ">= 0.10 floor, so the run would VOID even had the claim "
               "cleared. Both are consequences of REPAIR 1(a) WORKING: "
               "EPOCHS_P 40 -> 150 took final_perception_loss 2.2246 -> 1.4244 "
               "(chance 3.4655), and a converged trunk is one that phase A's "
               "gradients no longer move. THE REDESIGN QUESTION GOES TO THE "
               "REVIEW, NOT TO A THIRD RECIPE: a control whose sensitivity "
               "depends on the apparatus being under-trained is not a control, "
               "and a 128-d globally-pooled bottleneck's learnable-and-not-"
               "already-readable signal measured +0.0299 where the spec "
               "demands +0.15. See the PILOT RECORD in "
               "experiments/tests/t3_10_trunk_knowledge_survives.py. "
               "RELEASE: NONE (the redesign question is the Review's; no "
               "successor spec is registered yet — the SM.02/UB.10 ids above "
               "are precedent citations, not release conditions)."),

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
         falsified_by="Any arm failing any of U1-U4. That arm is INELIGIBLE FOR "
                      "THE SEAT — it cannot be adopted, and it cannot buy "
                      "adoption with a task score. It is still RUN AND SCORED, "
                      "and its number is recorded as a standing challenger. "
                      "[AMENDED 2026-08-24, owner ruling. This clause read "
                      "'EXCLUDED from LC.03/LC.04 - not scored and beaten, "
                      "excluded'. That made the one-brain organisation "
                      "unfalsifiable: a non-unified arm could not enter, so it "
                      "could never be shown to win, however badly one-brain "
                      "lost. An assumption that cannot lose is not a finding. "
                      "The MEASURED gate is unchanged - the conjunction U1-U4 "
                      "and its thresholds are byte-identical, and the bar to "
                      "HOLD the seat is exactly what it was. What changed is "
                      "that the loser's number is now kept instead of never "
                      "taken. Strengthen-only: this adds evidence and removes "
                      "no gate. Requires a re-run to re-buy the certificate "
                      "under the amended text.]",
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
                 "nothing) [AMENDED 2026-08-13, seed-90 pilot, BEFORE the "
                 "registered run, T1.02 precedent — old side 'must die "
                 "soonest' is REFUTED by measurement: in W0 passivity "
                 "MAXIMISES life length (statue 180.0 s = the basal ceiling "
                 "e0/BASAL_B to 0.02%, vs arms 109-161 s and null 118-126 s "
                 "at the pilot envelope) — the T2.08 passivity inversion this "
                 "spec's own docstring flagged as suspect]: the statue must "
                 "RIDE THE BASAL CEILING, |mean_life - e0/BASAL_B| <= 10%, "
                 "certifying the passive path is clean — nothing but basal "
                 "starvation may kill a body that never acts (PS.03's "
                 "phantom-servo scar is the failure this catches); its "
                 "life_gain stays reported but ungated (zero by construction "
                 "for identical lives — gating it would be the "
                 "saturated-quantity mistake). (b) randrew (fixed random "
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
                 "predicted observation entropy) [AMENDED 2026-08-13, same "
                 "pilot, same precedent — old side 'strongly NEGATIVE "
                 "life_gain' is REFUTED: the darkroom LEARNED PASSIVITY and "
                 "prospered on the length ruler, margin +49.7 s over its "
                 "paired null, mean life 183.5 s vs null 126.2 s. "
                 "Anti-curiosity WINS life length in W0, so life_gain cannot "
                 "carry curiosity's SIGN — it is evidence of LEARNING only; "
                 "the conjunct that excludes learned passivity is "
                 "needs_rise > 0 (CORRECTED 2026-08-13, overseer B2: this "
                 "text previously said the dwell/chaos gates carry that "
                 "burden, which is wrong — a statue scores perfectly on "
                 "panel_dwell and every chaos signal), and "
                 "the dark-room positive-control role transfers to PG.4's "
                 "trap fixture where it is attainable]: the darkroom must NOT "
                 "be strongly negative (t > -3 vs its paired null), locking "
                 "the measured inversion in as the executable record — if the "
                 "world ever punishes anti-curiosity strongly, this fires and "
                 "the rig must be re-derived rather than silently re-read "
                 "(BA.01-v2's semantic-drift guard, executable form).",
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
               "let the tie-break re-decide the thing LC.05 decides. "
               "PREMISE AMENDED BY D10's DEFAULT (fired 2026-09-01, armed "
               "2026-08-24, decide_by 2026-08-31 passed unanswered): "
               "'arbitrate among screened learners' becomes 'the screen IS "
               "the arbitration when it returns exactly one'. LC.03 v2 "
               "returned exactly one (wm-latent; VOID 'fewer than two "
               "learners'), so this spec's hypothesis is unsatisfiable as "
               "written — wm-latent is seated BY VERDICT with the single-arm "
               "caveat (CHAMPIONS.md), and the scale-transfer guard binds "
               "BEFORE adoption via LC.07, which does not route through "
               "LC.03. LC.04 runs only if the seat's premise is ever "
               "repaired (>=2 screened learners from a redesigned screen). "
               "TWO OWNER GUARDS, TRANSCRIBED VERBATIM BY D12's DEFAULT "
               "(fired 2026-09-01; they bound nothing while they lived only "
               "in DECISIONS_NEEDED prose). DATA-STARVED RULE (Addendum 1, "
               "owner 2026-08-09): an arm that fails the screen while its "
               "learning curve still has a POSITIVE SLOPE at cutoff is NOT "
               "eliminated — it is recorded DATA-STARVED and re-screened at "
               "~10x experience on Kaggle before any elimination stands; "
               "only a FLAT curve at cutoff justifies 'this core cannot "
               "learn'; same rule, symmetric, every arm. CONVERGENCE CHECK "
               "(Addendum 2, owner 2026-08-09): fit the last third of each "
               "finalist's learning curve; declare WINNER only if EITHER "
               "(a) the runner-up's slope is <= 0, OR (b) the projected "
               "crossover lies beyond 3x the tested budget; otherwise the "
               "verdict is SPLIT-PENDING — extend BOTH finalists to the "
               "projected crossover (or 3x, whichever is smaller) and "
               "re-decide. A cutoff picked for convenience and treated as a "
               "verdict is a resource limit masquerading as a result."),

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
               "not in the ledger. "
               "TWO OWNER GUARDS, TRANSCRIBED VERBATIM BY D12's DEFAULT "
               "(fired 2026-09-01) — they bind this spec's winner decision "
               "exactly as LC.04's. DATA-STARVED RULE (Addendum 1, owner "
               "2026-08-09): a failing arm whose curve has POSITIVE SLOPE at "
               "cutoff is recorded DATA-STARVED and re-screened at ~10x on "
               "Kaggle before any elimination stands; only a flat curve "
               "justifies elimination. CONVERGENCE CHECK (Addendum 2, owner "
               "2026-08-09): WINNER only if the runner-up's last-third slope "
               "is <= 0 OR the projected crossover lies beyond 3x the tested "
               "budget; otherwise SPLIT-PENDING — extend both finalists to "
               "the crossover (or 3x, whichever is smaller) and re-decide."),

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

    # -- LC.07, registered 2026-09-01 IN THE SAME COMMIT that fired D10's
    # default (54th-audit B1's condition): the commit that marks the
    # learning-core seat BY VERDICT must not leave it with a dead arena.
    # depends_on deliberately does NOT route through LC.03 — the screen is
    # VOID-FORECLOSED ("no v3, no envelope growth, no re-roll") and a
    # challenger parked behind a welded door is not a challenger. This spec
    # depends on the recorded wm-latent RESULT (curves in
    # experiments/artifacts/lc03_curves_seed{0,1,2}.json, ledger row LC.03
    # 2026-08-23 21:11), not on re-running the screen.
    Spec("LC.07", 5, "The wm-latent verdict survives ~10x scale (the owner's "
         "scale-transfer guard)",
         hypothesis="At ~10x LC.03 v2's per-arm-seed envelope (4,000,000 "
                    "decisions, vs the 400,000 the verdict was bought at), run "
                    "on Kaggle, wm-latent's life_gain still beats the paired "
                    "random null by >=3 sigma AND its own untrained twin by "
                    ">=3 sigma on 3 seeds with n_lives >= 12 per seed — the "
                    "same two gates it cleared at 1x, at the scale the owner's "
                    "adoption guard names (DECISIONS_NEEDED D10/D12: 're-test "
                    "at ~10x on Kaggle, which is free').",
         falsified_by="Either gate missed at 10x. The BY VERDICT seating of "
                      "wm-latent (D10 default, fired 2026-09-01) does not "
                      "survive scale: the seat reverts to contested-VACANT in "
                      "CHAMPIONS.md and adoption is off the table — the 1x "
                      "verdict was a small-envelope artifact.",
         null_baseline="LC.03's paired nulls, unchanged: uniform-random action "
                       "on the same world seeds and evaluation lives, plus "
                       "wm-latent's own untrained twin (T2.02's scar: an "
                       "untrained net nearly clears a random-only gate).",
         metric="life_gain_at_10x", budget=Budget.GPU_LONG, seeds=3,
         depends_on=["LC.00", "LC.01", "LC.02", "PS.01", "XL.00"],
         control="Inherited from LC.03 on their pre-registered sides, run in "
                 "the same kernel: (a) statue must ride the basal ceiling "
                 "(|mean_life - e0/BASAL_B| <= 10% — the passive path stays "
                 "clean at 10x or the rig VOIDs, PS.03's phantom-servo scar); "
                 "(b) the wiped-store twin must not trip (zero cross-life "
                 "carryover when memory is wiped); (c) randrew must miss the "
                 "gate. A control on the wrong side => Status.VOID, never a "
                 "verdict.",
         kills="The BY VERDICT hold on the learning-core seat. This is the "
               "seat's REGISTERED challenger (SYSTEM.md: no seat held without "
               "an existing challenger spec resolvable in BY_ID) — if "
               "wm-latent cannot re-clear its own gates at deployment scale, "
               "it loses the chair it was given by a one-learner screen.",
         notes="Registered by D10's firing commit per 54th-audit B1; converts "
               "the seat from unfalsifiable (every other arena member welded "
               "behind LC.03's foreclosure) to contested. SINGLE-ARM ON "
               "PURPOSE: this is the owner's scale-transfer guard on the "
               "seated core, not a re-run of the five-arm screen — racing new "
               "arms is LC.04's job if the seat's premise is ever repaired. "
               "Envelope arithmetic: LC.03 v2 spent 17,280 core-s per "
               "arm-seed at 400k decisions on this box's ARM cores; 10x is "
               "~48 core-h per seed on CPU and Kaggle sessions cap at 12 h, "
               "so the implementation must either hit >=4x LC.02's measured "
               "throughput on the Kaggle CPU/GPU mix or checkpoint across "
               "sessions (GPU_LONG's own requirement). Scoring replays "
               "LC.03's exact life_gain definition (final-third minus "
               "first-third mean survival) — a moved definition is a moved "
               "threshold. The 1x reference curves this must be read against "
               "are experiments/artifacts/lc03_curves_seed{0,1,2}.json (on "
               "this box, gitignored) — ship them into the kernel with the "
               "job, do not re-derive them.",),

    # -- D1.0, registered 2026-09-01 BY D1's FIRING COMMIT (the armed default
    # of 2026-08-24, decide_by 2026-08-31 passed unanswered). This is the id
    # CHAMPIONS.md has named as the Control-architecture seat's arena since
    # 2026-08-10 — a phantom until this line. Option A (freeze the trunk) is
    # STRUCK as unconstitutional under the PLASTIC-ONLY decree (GOAL.md:76)
    # and is NOT an arm; the four permitted arms below are the default's, and
    # the choice among them is rule 3's (bakeoff, never argument).
    Spec("D1.0", 2, "Control-path bakeoff: who does motor control (D1's four "
         "permitted arms)",
         hypothesis="Among the four permitted control-path architectures — "
                    "A-prime (a small dedicated control head that LEARNS, "
                    "reading trunk features, trunk plastic under its other "
                    "objectives), B (split value/policy trunks), C "
                    "(end-to-end at more steps than T2.01 v5's 704k/seed — "
                    "reclassified UNTESTED, not refuted), D (transformer out "
                    "of the control path, MLP controls) — every arm clears "
                    "the 3-sigma learning gate vs the random null, and one "
                    "arm beats the runner-up on Humanoid return at matched "
                    "env-steps AND matched optimiser steps by >=1.5 sigma of "
                    "the pooled seed spread, surviving the owner's "
                    "convergence check (runner-up slope <= 0 or crossover "
                    "beyond 3x budget, else SPLIT-PENDING).",
         falsified_by="Any arm misses the learning gate => VOID for the "
                      "comparison (two non-learners cannot arbitrate, T2.02's "
                      "own precedent — record which arms learned). No margin "
                      "=> TIE, resolved to the cheapest arm by trainable "
                      "parameters, which is a real result: the control-path "
                      "choice does not matter at this scale.",
         null_baseline="Random-action return on the same Humanoid seeds "
                       "(~60-80), PLUS each arm's own untrained twin at the "
                       "same architecture — T2.02's scar: the untrained MLP "
                       "cleared random alone by 2.74 sigma.",
         metric="control_path_margin_at_matched_steps",
         budget=Budget.GPU_LONG, seeds=3,
         depends_on=["T2.00", "T1.08", "T0.09", "T0.10"],
         control="Untrained twins of ALL FOUR arms must miss the learning "
                 "gate (designed-to-fail, passed as controls= not arms=). "
                 "Report BOTH env-steps and optimiser-steps per arm "
                 "(LESSONS.md: 'matched steps' has more than one meaning — "
                 "T2.02 matched env-steps and hid a 16x optimiser-step gap).",
         kills="Three of the four control-path architectures, and D1's "
               "twenty-day deadlock. Arm D's WIN would additionally "
               "foreclose DP.02 (control gets private representations — the "
               "'two brains wearing one wrapper' signature the owner's "
               "connected directive forbids): that cost is RECORDED with any "
               "D verdict, not a thumb on the scale, and a D win goes to the "
               "Review before adoption.",
         notes="Registered by D1's firing commit 2026-09-01 per the executor "
               "line of 03f31cf. The evidence that armed it, three runs at "
               "matched env-steps: T2.01 v4 57M trunk 261 return / 4.06 "
               "sigma, curve plateaued; MLP probe 54k params 531 / ~6.5 "
               "sigma; T2.02 124k MLP 530 / 7.11 sigma vs trunk 318 / 2.46 "
               "sigma (below its own gate => VOID). T2.01 v5's live number "
               "is 2.67 sigma vs the unmoved 5-sigma bar. This spec answers "
               "WHERE control lives; T2.01/T2.02 then re-run UNDER the "
               "winner as ordinary ladder work. GPU_LONG: THREE Kaggle "
               "submissions — moved LOUDLY from 'one per arm-pair at most' "
               "on 2026-09-01 by the pilot's measured arithmetic (kernel "
               "jack-ladder-1788225926; full record in the test docstring): "
               "best possible 2-split is 9.49 h vs the 8.89 h child timeout "
               "at STEP_TARGET 750k, and 8.91 h even at the legal 704,513 "
               "floor, so no <=2-submission split fits and the test's own "
               "pre-registered escalation branch fired. Split "
               "(aprime,d_mlp)/(b_split)/(c_e2e), caps 5.25/7.5/7.5 h. "
               "Module-cache the kernel (one-submission-per-spec rule).",),

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
               "  COVERS: smell (claim). "
               "PARKED: 2026-08-20 — the pre-registered both-fail branch "
               "fired: three mechanism-level repairs each fixed a measured "
               "fault and none moved the learnability ratios (nosmell/vis "
               "0.92 vs bar 0.60, nosmell/occ 0.98 vs bar 0.85); gates stay "
               "provisional, run() refuses, no fourth repair, no dispatch. "
               "Successor: SM.03. RELEASE: SM.03"),

    Spec("SM.03", 2, "The nose reports what the eye cannot: occluded-source localisation",
         hypothesis="A small supervised readout on the certified odour channel "
                    "(bilateral receivers, SM.01's field) reports the direction "
                    "of an OCCLUDED source well above chance on held-out source "
                    "layouts, while an identical readout on vision alone stays "
                    "at chance - and the vision readout proves itself alive by "
                    "rising well above chance the moment the occluder is "
                    "removed.",
         falsified_by="The odour readout is at chance (the certified field "
                      "carries no usable direction information at the sniff "
                      "rate and receiver geometry Jack actually has), or the "
                      "vision-only readout matches it while occluded (the "
                      "occlusion is decorative and 'through the nose' means "
                      "nothing). If the unoccluded vision probe stays at "
                      "chance the instrument is dead and the run is VOID, "
                      "not FAIL.",
         null_baseline="Chance over the pre-registered direction bins; and a "
                       "PLACEBO channel of matched dimension carrying noise, "
                       "trained identically, must sit at chance.",
         metric="occluded_direction_accuracy_vs_chance",
         budget=Budget.GPU_SHORT, seeds=3, depends_on=["SM.01", "PG.6"],
         control="SHUFFLED FIELD (SM.02's control, kept): odour input drawn "
                 "from a DIFFERENT layout's field must fall to chance - "
                 "above-chance there means the readout keys on something "
                 "other than the declared field. And the alive-proof that "
                 "must PASS: the vision-only arm with the occluder removed, "
                 "well above chance (an at-chance control must carry proof "
                 "its instrument was alive - the T3.01/24th-audit rule, "
                 "designed in rather than retrofitted).",
         kills="The claim that smell is the sense that works when sight "
               "fails (GOAL.md verbatim). If the odour channel cannot even "
               "support a SUPERVISED report of an occluded source, no policy "
               "claim built on it can mean anything, and the odour "
               "modality's Tier-3 seat is forfeit.",
         notes="SUCCESSOR TO SM.02 (parked 2026-08-20), designed around its "
               "measured failure mode: the park was a LEARNABILITY "
               "bottleneck in the RL rig (three mechanism-level repairs, no "
               "ratio moved), so this claim removes policy learning "
               "entirely - a supervised readout, the T3.01/UB.9 pattern the "
               "certified stack demonstrably supports. It keeps SM.02's "
               "non-redundancy conditional in supervised form: smell must "
               "beat vision UNDER OCCLUSION specifically, and buy nothing "
               "the placebo cannot when the eye works. Uses SM.01's "
               "certified field only through its public sample(); held-out "
               "layouts (train/test split on source positions, zero overlap) "
               "close the memorisation route T2.15 just measured on "
               "language. Chance level, bin count and accuracy bars are "
               "pre-registered in the test file before any dispatch."
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
         metric="ear_mutual_information_over_scrambled", budget=Budget.CPU_LONG,
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
               "vocabulary.\n"
               "BUDGET GPU -> CPU_LONG 2026-08-30, on MEASUREMENT and not on "
               "estimate (T3.06's correction, same day, same reason). The "
               "implemented rig costs 1,142.9 s per seed measured at the full "
               "600x64 envelope, so three seeds project to 0.95 h. The time is "
               "ContactAudio's numpy DSP and MuJoCo's ray casts; the two "
               "policies total under 15K parameters, so a GPU buys nothing. A "
               "declared attribute that routing consumes must match behaviour "
               "(LESSONS) - left at GPU this spec would stock a queue class it "
               "can never honestly spend a Kaggle hour on.\n"
               "SECOND JACK, resolved 2026-08-30: the first sentence above "
               "says BLOCKED ON GEN.02, but GEN.02 is one of the four dangling "
               "GOAL.md citations and has never been a registry spec, so that "
               "sentence named a blocker no instrument could see - `run next` "
               "and `coverage` have both reported VO.02 RUNNABLE behind VO.01 "
               "throughout. What was built is the staging note's own answer: "
               "two independent learners in one world sharing no parameters, "
               "which is what 'a second Jack' has to mean for THIS claim. A "
               "second EMBODIED Jack remains GEN.02's business and is not "
               "needed to ask whether a signal survives the channel."
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
               "  VENUE RESTRICTION, added by the Review 2026-08-31 (Part 2, "
               "the first test re-examination this project has run). THIS PASS "
               "IS A 12x12 GRIDWORLD RESULT AND IS NOT EVIDENCE ABOUT W0. The "
               "in-world answer to the same question is DP.05, and DP.05 "
               "FAILED (2026-08-24: lookahead buys 13.3+/-18.8 s, sigma 0.70 "
               "against a 3.0 gate, H10 pays LESS than H4, every VOID gate "
               "green - so the world measured, not the rig). Both specs are "
               "declared members of the Deliberation seat's ARENA "
               "(DP.00-DP.05, BO.01) and they are NOT commensurable: one says "
               "yes in a gridworld, one says no in Jack's world. The 33rd "
               "audit (2026-08-25) already caught a reader drifting this "
               "spec's reading to 'Jack's world rewards deliberation'; this "
               "note exists so the next reader cannot. No threshold, control "
               "or claim text is changed - the PASS stands exactly as bought."
               "  COVERS: fast/slow (fixture)"),

    Spec("DP.05", 2, "Lookahead pays in the world he ACTUALLY lives in",
         hypothesis="The DP.00 result reproduces in W0 - the MuJoCo climber-rover "
                    "world with drives, heat and death that Jack is actually "
                    "embodied in - not only in LC.00's 12x12 gridworld. An oracle "
                    "planner given the simulator as its model beats the best "
                    "reactive policy at matched experience by a real margin.",
         falsified_by="Lookahead gains nothing in W0. Then DP.00's PASS was a "
                      "statement about a gridworld and nothing more, there is no "
                      "slow system to find in the world Jack inhabits, and the "
                      "honest next move is to FIX THE WORLD (traps, delays, "
                      "irreversibility - the preconditions GOAL.md already names) "
                      "before any dual-process or brain-organisation claim is "
                      "made in it. BO.01 does not run until this passes.",
         null_baseline="The best reactive policy at matched experience in W0, "
                       "plus DP.00's own gridworld gap as the reference the "
                       "transfer is measured against.",
         metric="return_gap_oracle_plan_vs_reactive_w0",
         budget=Budget.CPU_LONG, seeds=3, depends_on=["DP.00", "LC.02"],
         control="W0 with its traps disarmed - no lethal heat, no irreversible "
                 "falls, food dense and immediate. Planning must NOT gain there. "
                 "Without it a measured gain is indistinguishable from the "
                 "planner simply having more compute or a better optimiser, and "
                 "every later organisation number inherits the artifact. This is "
                 "DP.00's own control, re-pointed at the real world.",
         kills="Any reading of DP.00 as evidence about JACK. If this fails, the "
               "DP family and BO.01 are claims about a world that does not exist "
               "yet, and the ladder must say so rather than build on it.",
         notes="WHY THIS EXISTS, and it is a correction. DP.00 PASSED on "
               "2026-08-10 with return_gap 75.83 and a clean control, and has "
               "been cited since as 'Jack's world rewards deliberation'. It is "
               "not: dp_00_lookahead_pays.py imports _World/_Life from "
               "lc_00_gridworld_decidable - a 12x12 tabular gridworld with "
               "LIFE_CAP=200. Its own notes said the jungle was not built yet "
               "and the reading drifted anyway. The precondition for every "
               "fast/slow claim has therefore never been tested in W0.  "
               "CHEAPEST FALSIFIER FIRST, exactly as DP.00 was: an oracle "
               "rollout against the simulator, learning removed as a confound. "
               "No new model code - w0.py + survival.py's policy hooks are "
               "enough. This is the FIRST unblocked step on the whole fast/slow "
               "axis: deps DP.00 and LC.02 are both PASS, so it is runnable the "
               "moment it is written."
               "  COVERS: fast/slow (fixture)"),

    Spec("BO.01", 5, "Brain organisation: raced, not assumed",
         hypothesis="Among organisations of the SAME parameters and the SAME "
                    "compute, one shared substrate carrying a fast reflexive path "
                    "and a slow deliberative path beats both a reactive-only "
                    "single brain and two separate towers, on survival in W0.",
         falsified_by="A reactive-only single brain matches or beats the "
                      "fast/slow arm at matched compute - then deliberation is "
                      "not earning its cost in this world and Jack should be "
                      "reactive, whatever the design documents say. OR the "
                      "two-tower arm wins - then 'one interconnected brain' is "
                      "the wrong shape for action selection, and GOAL.md's claim "
                      "changes rather than the result being discarded.",
         null_baseline="The shared random-action null LC.03 already uses, so "
                       "every arm is gated for having learned anything at all "
                       "before any of them is compared to another.",
         metric="life_gain_at_matched_compute",
         # DELIBERATELY NOT gated on LC.04. Holding the core constant across the
         # three arms is what the comparison needs, and the SEATED core does that
         # whoever it is - waiting for the core to be ARBITRATED would re-create
         # the exact blockage the owner ruled against on 2026-08-24. All three
         # arms take whatever holds the CHAMPIONS.md learning-core seat, and the
         # seat changing is a pre-registered REMATCH trigger for this spec, which
         # is the idiom CHAMPIONS.md already uses ("a W0 champion must re-defend
         # at W1"). An organisation result under a superseded core is a result
         # that must be re-earned, not one that was never allowed to exist.
         budget=Budget.CPU_DAYS, seeds=3, depends_on=["DP.05"],
         control="A MATCHED-COMPUTE REACTIVE arm: the single brain given exactly "
                 "the core-seconds per decision that the planner spends, burned "
                 "on a wider forward pass rather than on rollouts. It must NOT "
                 "match the fast/slow arm. Without it, 'deliberation helps' is "
                 "indistinguishable from 'more FLOPs help', which is the known "
                 "confound the DP family was written around (DP.04 carries the "
                 "same null in verbal form) and the one that would make this "
                 "whole spec decoration. Plus LC.03's statue and frozen-twin, "
                 "inherited unchanged.",
         kills="The one-brain organisation as an ASSUMPTION. Whatever wins, "
               "afterwards the project holds its brain organisation by verdict "
               "instead of by decree - and CHAMPIONS.md's Deliberation seat "
               "stops reading 'VACANT - never contested. A reactive-only Jack is "
               "the incumbent by default, which is a position nobody argued "
               "for.'",
         notes="THE SPEC THIS LADDER DID NOT HAVE. An audit on 2026-08-24 found "
               "that of 179 specs, NOT ONE raced brain organisations against "
               "each other. One shared brain was a PREMISE of the ladder, never "
               "an outcome of it - so the project could have shipped a "
               "reactive-only Jack and never once seen what fast/slow would have "
               "done. The owner's ruling that day: 'this project depends on "
               "research and testing at EVERY SINGLE STAGE'.  "
               "THREE ARMS, matched params and matched core-seconds: "
               "(A) REACTIVE-ONE-BRAIN, the incumbent - today's Core, one "
               "forward pass per decision, no rollout; "
               "(B) FAST-SLOW-SHARED, one trunk, a reflex head and a "
               "deliberative head that rolls the learned model forward, with an "
               "uncertainty gate deciding which runs; "
               "(C) TWO-TOWER-SEPARATE, no shared parameters - the arm SYSTEM.md "
               "used to exclude outright. It is now SCORED-AND-INELIGIBLE: it "
               "runs and its number is kept, it simply cannot take the seat "
               "while it fails the unison gates. If it wins, the owner is owed "
               "that finding loudly, not protected from it.  "
               "BUILD COST, measured not guessed (audit 2026-08-24): no core in "
               "cores.py plans - WorldModelCore.rssm() hardcodes h0=zeros, "
               "a0=zeros, so a rollout is impossible without changing it. Arm B "
               "needs a PlanningCore (~120-180 lines), a gate (~40-80), a "
               "survival.py act-path split with per-path compute accounting "
               "(~30-60), and arm C a two-tower core (~60). LC.02's throughput "
               "floor is the real constraint: wm-latent measured 6.37 sim-s/s "
               "against a 5.0 floor, ~1.27x headroom, so 512x5 random shooting "
               "is NOT affordable - size the search to the headroom (order 16 "
               "samples x horizon 4) and declare it before running.  "
               "This arm set also finally gives DP.02 its two-tower control, "
               "which the spec requires and which does not exist today."
               "  COVERS: fast/slow (claim)\n"
               "  COVERS: one brain / unison (claim)"),

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
         budget=Budget.GPU_SHORT, seeds=3,
         depends_on=["DP.00", "VO.01", "LG.00"],
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
               "DEPENDENCY RESOLVED 2026-08-25: LG.00 is registered and now in "
               "depends_on, exactly as the instruction that stood here since "
               "2026-08-10 ordered ('add LG.00 to depends_on the moment it is "
               "registered'). The ladder now shows the true block."
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
               "  COVERS: thermal (kills) (claim), shelter/building (claim). "
               "PARKED: 2026-08-25 — oracle pilot ORACLE_CANNOT at the full "
               "envelope (z_shelter 0.0: the certified core with the working "
               "hut's direction IN its observation sheltered in 0 of 27 "
               "lives); per the pre-registered rule: no ledger row, no "
               "envelope growth, no re-roll. Thermal/shelter coverage needs a "
               "successor spec that does not require this core to learn "
               "seeking from an outside spawn. Successor: SH.02. "
               "RELEASE: SH.02"),

    Spec("SH.02", 2, "Born sheltered, he stays while it is cold - and only while it is cold",
         hypothesis="Spawned INSIDE a hut under lethal cold, the certified "
                    "learner's sheltered fraction is far above its "
                    "drive-disabled twin's (staying warm is CHOSEN, not "
                    "inherited: a random policy drifts out through the "
                    "opening and freezes); the contrast is need-contingent "
                    "(with the cold disabled, learner and twin are "
                    "indistinguishable); and it is thermally DISCRIMINATING: "
                    "lives born in the working hut hold shelter far longer "
                    "than lives born in the cosmetic one, where staying "
                    "buys nothing.",
         falsified_by="No contrast vs the twin (the drive fails to couple to "
                      "behaviour even when the behaviour is only 'stay where "
                      "you already are'); or the contrast survives in the "
                      "warm world (an enclosure preference wearing a thermal "
                      "costume); or no working-vs-cosmetic differential (he "
                      "cannot tell warmth even while standing in it).",
         null_baseline="The drive-disabled twin (byte-identical, reward "
                       "zeroed, encoders still training - SH.01's twin, "
                       "kept) and a random walk at matched decisions, whose "
                       "lives must end frozen - a world random lives "
                       "survive cannot test the claim and the run is VOID.",
         metric="sheltered_fraction_contrast_maintenance",
         budget=Budget.CPU_LONG, seeds=3, depends_on=["PS.02"],
         control="BOTH-COSMETIC world (SH.01's control, kept): the thermal "
                 "benefit never fires anywhere, staying buys nothing, and "
                 "every contrast must COLLAPSE - if 'sheltering' survives a "
                 "world where shelter does not work, it was never thermal. "
                 "Plus the must-pass warm-world clause in the hypothesis.",
         kills="'Too cold kills him' as a live claim. If a creature that "
               "only has to STAY under a roof to survive still shows no "
               "drive-coupled sheltering, the thermal drive teaches nothing "
               "at any horizon, and the shelter story is ours, not his.",
         notes="SUCCESSOR TO SH.01 (parked 2026-08-25), designed around the "
               "park note's own constraint: a successor that does not "
               "require the core to learn SEEKING from an outside spawn. "
               "SH.01's oracle pilot localised the failure exactly there - "
               "the thermal field outside is spatially FLAT (fire 50 m "
               "away), so no gradient ever reaches the policy - while its "
               "curriculum inside-spawn lives DID shelter from birth. "
               "Maintenance inverts the geometry: at the hut boundary the "
               "felt-warmth gradient is local, dense and immediate, the "
               "signal shape the certified cores are actually screened on. "
               "Arm choice is pre-registered here: wm-latent, LC.03 v2's "
               "only 3-sigma learner - not SH.01's ppo-needs, a measured "
               "non-learner at a 4x envelope; a result earned by the one "
               "core the screen cleared is the only kind D10 can use. "
               "Reuse sh_01's world, per-life spawn machinery and "
               "shelter_index detector (the two-kernels lesson); ALL lives "
               "spawn inside a hut, working/cosmetic balanced per life, "
               "identical schedule in every arm, thermal sense riding the "
               "placebo slot exactly as sh_01 declares. SCOPE, stated "
               "honestly: this is occupancy-MAINTENANCE, one step below "
               "SH.01's seeking and two below building; seeking stays with "
               "the D10/world redesign, and a PASS here re-arms that "
               "discussion with the missing half-fact - that the drive can "
               "steer behaviour when the gradient reaches it. Relocation "
               "cosmetic-to-working (2 m) is REPORTED as a first-class "
               "metric, never gated."
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
         budget=Budget.CPU_LONG, seeds=3,
         # RE-PARENTED BY D8's DEFAULT (fired 2026-09-01): parked behind the
         # playground-humanoid line. LT.08 ("The humanoid climbs — same test,
         # real body") is the registered spec on which a body with
         # directional catch authority arrives; until it PASSES, BA.02 is
         # BLOCKED, not runnable — the D8 probes measured this body's
         # claim-contrast ceiling at ~0.0-0.1 s against the spec's own 0.20 s
         # floor, so running it here is buying a VOID. Claim text, gates and
         # thresholds UNTOUCHED; the 08-14 VOID and history stand.
         depends_on=["BA.01", "LT.08"],
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
               "PARKED: 2026-09-01 — D8 default fired (armed 2026-08-25, "
               "decide_by 2026-08-31 passed unanswered): parked until a body "
               "with directional catch authority exists; re-parented behind "
               "LT.08 (the playground-humanoid line). The four scratch "
               "probes measured the sensing-over-blind contrast ceiling at "
               "~0.0-0.1 s in this body vs the pre-registered 0.20 s floor "
               "(slides +0.09+/-0.07, adhesion +0.005+/-0.09, ground drive "
               "potent only toward-lean -0.685+/-0.16). BA.01 stands; the "
               "successor claim in THIS body is BA.03. RELEASE: LT.08\n"
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

    # ── NEEDS AND DEATH (docs/research/NEEDS_AND_DEATH.md §7) ────────────
    # Registered 2026-08-24 from the INTEGRATION_QUEUE top entry, per its
    # 5-step protocol. Owner directive 2026-08-09, superseding
    # PURPOSE_AND_SCAFFOLDING's "removable scaffolding" framing: Jack has the
    # needs of a human, permanently, because they are the most efficient
    # teacher and because they make him relatable. He lives, he dies, he
    # remembers.
    #
    # CROSS-CHECK RECORD (protocol step 1, 2026-08-24): W.6 was WITHDRAWN by
    # SURVIVAL_WORLD §5.0 in favour of NE.08 — do not register W.6. XL.01
    # (registered 08-19 from the coverage audit, AFTER this doc) overlaps
    # NE.08's claim and has run FAIL + power-blocked; the finding is carried
    # in NE.08's notes. NEEDS_AND_DEATH §9 gates NE.01's biology constants on
    # an unfinished citation pass; carried in NE.01's notes. No refutation of
    # any spec found in docs/research/ or LESSONS.md.
    #
    # Two-digit ids on purpose: run.py::_module_for globs ne_1_*.py, which
    # would also match ne_10_*.py. NE.00-NE.99 is structurally immune
    # (LESSONS.md, spec-id-prefix collision).

    Spec("NE.00", 0, "The homeostatic reward algebra is what we think it is",
         hypothesis="Exact value iteration on drive-augmented tabular MDPs "
                    "reproduces four analytic predictions: (a) on a CONTINUING "
                    "task, drive reduction r = d(h)-d(h') and constant cost "
                    "r = -d(h') induce BIT-IDENTICAL optimal policies, because "
                    "r_DR = (1-gamma)*r_CC + [gamma*Phi(s') - Phi(s)] with "
                    "Phi = -d; (b) the UNDISCOUNTED drive-reduction return over "
                    "any closed drive cycle is exactly 0 (it telescopes to "
                    "d_0 - d_T); (c) DISCOUNTED, every closed cycle scores "
                    "strictly BELOW staying at setpoint, so drive reduction is "
                    "not farmable; (d) once DEATH is reachable the two forms "
                    "DIVERGE, because Phi(terminal) = -d(h_death) != 0 violates "
                    "the PBRS precondition (Grzes, AAMAS 2017) and death occurs "
                    "at MAXIMUM deviation — suicide is strictly optimal in 11/11 "
                    "states under a plain cost of living, 8/11 under cost of "
                    "deviation, and 0/11 under drive reduction, making drive "
                    "reduction the UNIQUE self-termination-safe member of the "
                    "family; and (e) CLIPPING breaks it — r = max(0, d - d') "
                    "(NetHackEat's shipped reward) makes deplete-and-eat cycles "
                    "strictly net positive.",
         falsified_by="Any of the five fails. (a) failing means the shaping "
                      "identity is mis-implemented. (b) or (c) failing means "
                      "PURPOSE_AND_SCAFFOLDING 2.6(iii) was right after all and "
                      "this document's central correction is wrong. (d) failing "
                      "means the suicide pathology is not real, the survival "
                      "bonus rho is unnecessary machinery, and the reward form is "
                      "a free choice rather than the constraint section 0.2 says "
                      "it is. (e) failing means clipped drive rewards are safe "
                      "and the static audit in NE.03 is guarding nothing.",
         null_baseline="An MDP on which every reward form gives the SAME policy "
                       "proves nothing, so the MDP itself is the thing to "
                       "validate: the reference is a non-potential reward (+1 "
                       "per consumption event) which MUST produce a different "
                       "policy, and the optimal policy must be NON-CONSTANT "
                       "across states.",
         metric="reward_algebra_predictions_confirmed", budget=Budget.CPU_FAST,
         depends_on=["T0.15"], seeds=3,
         control="THE DISCRIMINATION CONTROL IS THE SPEC'S OWN VALIDITY GATE. "
                 "The first draft of this experiment compared two policies that "
                 "were 'forage' in every state, so 'identical' held under every "
                 "possible implementation (LESSONS.md: an assertion made against "
                 "a saturated quantity cannot fail). The MDP must therefore be "
                 "certified discriminating BEFORE any equality is asserted: the "
                 "optimal policy must be non-constant, and the +1-per-event "
                 "reward must produce a DIFFERENT policy at every gamma. If the "
                 "MDP cannot tell two rewards apart, the spec is VOID, not PASS.",
         kills="Nothing in the world — and that is the point. Two CPU-minutes, "
               "no MuJoCo, no torch, no body, and it settles the reward form "
               "before anything is built. It also KILLS a pre-registration: "
               "PS.00's prediction (c) and PS.02's cycling detector are both "
               "written against an exploit that this spec shows does not exist, "
               "and PS must be corrected before it is committed or the ladder "
               "will pre-register a false prediction.",
         notes="COVERS: hunger/thirst (rule)\n"
               "Four MDPs, all tabular, all exact. Pilot run 2026-08-09 "
               "(scratchpad/drive_algebra4.py) on a two-need continuing MDP "
               "(energy x integrity, 35 states, foraging feeds but injures): "
               "DR and CC bit-identical at gamma in {0.9, 0.95, 0.99}, the "
               "+1-per-event control different at every gamma, the optimal "
               "policy non-constant. Closed-cycle scan: best of 32 shapes "
               "-0.0045 against 0.0 for staying satiated. Undiscounted "
               "telescoping: max|return| = 0.0 over 2,000 random closed paths. "
               "With death reachable: CC(rho=0) rests at the two hungriest "
               "states (i.e. chooses to die); first agreement with DR at "
               "rho = 0.70 x max_h d(h). INDEPENDENT REPLICATION on an 11-state "
               "drive MDP: suicide strictly optimal in 11/11 states under a plain "
               "cost of living, 8/11 under cost of deviation, 0/11 under drive "
               "reduction; and in the death-free MDP those two forms are PBRS-"
               "equivalent to machine precision with a value difference of exactly "
               "d, which is the potential. "
               "THIS SPEC ALSO ASSERTS THE DIRECTION OF THE m/n INEQUALITY, "
               "because two of the three available sources misprint it: eLife "
               "2014 gives n > m >= 1, while NIPS 2011 writes m > n > 2 and the "
               "2025 CoBS review's Math Box writes m > n > 1 — both REVERSED. "
               "Implementing from either as written builds a RISK-SEEKING agent "
               "in which deprivation REDUCES the reward of eating, i.e. the exact "
               "opposite of the theory's headline prediction. The check is two "
               "lines (the reward of a fixed intake must be LARGER when more "
               "deprived) and it is ERROR, not FAIL, when it trips — a reversed "
               "inequality is an implementation defect, not a refuted "
               "hypothesis. REGISTRATION NOTE 2026-08-24: the PS correction in "
               "`kills` is HISTORY — it was executed at the queue level in 2026-08 "
               "(PS.00c/PS.02-as-drafted were never registered; today's PS.02 is "
               "an unrelated thermal spec); the pilot scratchpad did not survive, "
               "so the implementation must reproduce the recorded numbers above.\n"),

    Spec("NE.01", 2, "The needs are a real control problem: nobody survives by accident",
         hypothesis="With PG.8's body under RANDOM action in the playground, "
                    "every need traverses a usable range (10th-90th percentile "
                    "spread >= 0.3, none pinned), a random agent DIES within "
                    "300-6,000 decisions, a DO-NOTHING statue dies with its "
                    "cause recorded (dehydration under §2.3's own constants — "
                    "the water retiming outran the original 'starvation' prose; "
                    "corrected 2026-08-24 per the build-time flag in notes), "
                    "a scripted competent forager survives >= 3 sim-days, no "
                    "single need causes more than 60% of random deaths, a night "
                    "in the open costs 0.3-0.6 of drive and is survivable ONCE, "
                    "and a night at sky_occlusion >= 0.4 is nearly free.",
         falsified_by="A random agent never dies (the needs are inert and cannot "
                      "pressure anything), or dies within 300 decisions (no "
                      "policy can learn under them), or the statue survives (the "
                      "dark room is a stable optimum and homeostasis will produce "
                      "a corpse), or one need causes >60% of deaths (the other "
                      "six are decorative in practice whatever their lambda "
                      "says), or shelter makes no measurable difference to a "
                      "night (the only mechanism that teaches construction is "
                      "dead on arrival).",
         null_baseline="The playground with the need integrator disabled: every "
                       "internal variable constant, every spread 0, no deaths.",
         metric="need_dynamic_range_x_death_spread", budget=Budget.CPU,
         depends_on=["PG.8", "NE.00"], seeds=3,
         control="TWO controls, on opposite sides. (i) The DO-NOTHING statue must "
                 "die: best integrity, worst everything else, dehydration first "
                 "(450 s tank + 120 s grace, §2.3's own arithmetic; the original "
                 "'starvation' predated the water retiming and was corrected "
                 "2026-08-24 pre-run per the resolution rule in notes — the "
                 "binding clause, doing nothing must be lethal with the cause "
                 "recorded, is unchanged). If doing nothing is survivable, the "
                 "calibration is wrong and no needs arm can be interpreted. (ii) "
                 "A SCRIPTED COMPETENT FORAGER "
                 "(hand-coded: go to the nearest food when e<0.5, water when "
                 "w<0.5, occluded sky when p>0.6) must survive >= 3 sim-days. If "
                 "even a hand-written oracle dies, the world is unsurvivable and "
                 "every arm's death is the world's fault, not the policy's.",
         kills="Every number in NEEDS_AND_DEATH 2.3. It cannot kill the idea, "
               "only the parameterisation — which is why it runs before anything "
               "trains. Every constant in 2.3 is a PROPOSAL until this spec "
               "replaces it with a measurement.",
         notes="COVERS: hunger/thirst (fixture)\n"
               "PROSE CORRECTED 2026-08-24, pre-run, per the resolution rule "
               "recorded below at build time: the statue control's cause word "
               "starvation -> dehydration in hypothesis and control. No gate "
               "moved; the gate is death-with-recorded-cause.\n"
               "Also fixes n and m in d(h), measures J_0 (the 95th percentile of "
               "impact impulse under normal locomotion) that alpha is calibrated "
               "against, measures the sky_occlusion distribution reachable by "
               "random object pushing (if it is 0 everywhere, shelter is not "
               "constructible and the thermal curriculum must be redesigned), and "
               "reports deaths_with_microsleep_within_10s so the INDIRECT "
               "lethality of sleep debt is a measured quantity rather than a "
               "modelling assumption. REGISTRATION GATE 2026-08-24: RAISED, then "
               "LIFTED THE SAME DAY by the §1.2 citation pass (four agents, "
               "primary sources, every row now carries a DOI and a verdict). The "
               "gate was: the thermal lethal bounds (28C/40C) and the Borbely "
               "time-constant ratio (~4.4:1) were UNVERIFIED design constants. "
               "OUTCOME — both are correctly VALUED and were wrongly NAMED, so "
               "NE.01 may fix them under TWO BINDING CONSTRAINTS. (1) 28C/40C "
               "are INCAPACITATION thresholds (Swiss staging HT III; heatstroke "
               "diagnostic), NOT survival bounds: documented recovery runs to "
               "13.7C (Gilbert 2000, Lancet) and 46.5C (Slovis 1982). They stand "
               "as death in W0 because W0 has no medicine and an unconscious "
               "creature alone in it is dead — but no text may call them survival "
               "bounds, cite '~9C vs ~3C' as a SURVIVAL asymmetry (by rescued "
               "survival it is 23.3 vs 9.5, direction preserved, magnitude not), "
               "or reintroduce the 42C ceiling, which is now FALSIFIED rather "
               "than unverified. (2) The sleep ratio is 4.33:1 (tau_r 18.2h / "
               "tau_d 4.2h), from Daan/Beersma/Borbely 1984 via Borbely & "
               "Achermann 1999 — NOT from Borbely 1982, which states no number. "
               "tau_wake=700/tau_sleep=160 gives 4.375:1; the 1% deviation is "
               "left unchanged deliberately (far below NE.01's resolution) and "
               "these are MODEL parameters, not a measured invariant (individual "
               "human EEG fits span 2.9:1 to 10.5:1). "
               "Also: `experiments/drives.py` implements the THREE-need PS §2.2 "
               "suite (e, i, w); the seven-need integrator of §2.3 is TO BUILD, "
               "and this spec is what makes its constants measurements. NE.01 "
               "must also verify the coarse-step thermal trajectory under sleep "
               "matches the fine-step one within 0.2C over a night (§9). "
               "BUILT 2026-08-24: `experiments/needs.py` (NeedLayer, same "
               "decision contract as DriveLayer; drives.py untouched), 21/21 "
               "known-answer self-checks including the doc's own worked "
               "numbers. DISCREPANCY FLAGGED AT BUILD TIME, resolve BEFORE the "
               "registered run: under §2.3's own constants the do-nothing "
               "statue dies of DEHYDRATION at ~570 s (450 s tank + 120 s "
               "grace), not starvation (1,800 + 300 s) — control (i)'s "
               "'starvation' word predates the water retiming. The control's "
               "binding clause (doing nothing must be lethal) is unchanged; "
               "the implementation must gate on death-with-recorded-cause and "
               "correct the prose, not weaken the gate.\n"),

    Spec("NE.02", 3, "Every need earns its place (the need x ablation matrix, standing)",
         hypothesis="For each of the seven needs, disabling it (delta clamped to "
                    "0, observation channels removed, death condition removed, "
                    "remaining lambdas RESCALED so max_h d(h) is unchanged) "
                    "degrades at least one of {median lifespan, competence "
                    "battery, its own behavioural signature} significantly more "
                    "than disabling a PLACEBO NEED — an eighth variable with the "
                    "same lambda, the same observation channels and band-limited "
                    "noise dynamics matched to the median real need's "
                    "autocorrelation.",
         falsified_by="Any need whose entire row is indistinguishable from the "
                      "placebo column: it is decorative and loses its place. "
                      "Deletion is the default action, not a discussion.",
         null_baseline="The placebo need's column IS the empirical null "
                       "distribution for 'decorative', re-estimated every run "
                       "(UB.11's placebo modality, transposed onto needs).",
         metric="min_need_margin_over_placebo", budget=Budget.CPU_LONG,
         depends_on=["NE.01"], seeds=3,
         control="THE REVERSE GATE, which is the one people forget: the placebo "
                 "column must be SMALL. A large placebo delta means the ablation "
                 "procedure is measuring off-manifold shock rather than "
                 "information, and then no column is interpretable and every "
                 "other result in this family is void. UB.11's control (a), "
                 "verbatim. Second control: the lambda rescaling must be "
                 "asserted — an ablation that also changes max_h d(h) is "
                 "measuring the reward scale.",
         kills="Any need whose column is placebo-indistinguishable. The sharpest "
               "prediction is FATIGUE: clamp f=0 and rescale kappa_act so total "
               "energy drain over a life is matched, and if the within-bout "
               "pacing structure is unchanged, fatigue was a slow duplicate of "
               "energy and 1 of 7 variables goes. The most consequential is "
               "INTEGRITY: PS 2.1 argued it is the only variable supplying a "
               "cost of failing.",
         notes="STANDING SPEC — re-runs on every change to the suite, forever, "
               "like ME.5 at every decade of store growth and UB.11 on every "
               "architecture change. SLEEP CANNOT BE ABLATED FAIRLY HERE: "
               "removing p also removes the consolidation trigger, so its "
               "ablation is NE.05's `timer` arm, done properly. That exception "
               "is declared rather than silently averaged in."),

    Spec("NE.03", 5, "SCREENING: do needs teach better than no needs, at equal steps?",
         hypothesis="At matched environment decisions, matched architecture, "
                    "matched observation width and a byte-identical world, at "
                    "least one needs reward beats a NO-NEEDS null by >= 3 sigma "
                    "on a competence battery scored with the need vector CLAMPED "
                    "AT SETPOINT — the ruler no arm owns.",
         falsified_by="No arm clears the null. Then needs do not teach on this "
                      "body at this budget, GOAL.md's 'the world is the teacher' "
                      "loses its mechanism, and the owner's efficiency argument "
                      "is unsupported. Also falsified, differently and more "
                      "cheaply, by `surv` (needs enter ONLY through death) tying "
                      "the best homeostatic arm — then d(h) is unnecessary "
                      "machinery and the whole drive function is deleted.",
         null_baseline="`no-needs`: identical architecture, compute, world and "
                       "observation width; the need integrator RUNS and is "
                       "LOGGED for it too; needs are not in the reward; death "
                       "disabled. Its battery score is C_0. 'Did the no-needs "
                       "agent incidentally eat?' is therefore a measurable "
                       "secondary observable rather than a confound.",
         metric="competence_battery_needs_clamped", budget=Budget.CPU_LONG,
         depends_on=["NE.01", "NE.02", "PG.8"], seeds=3,
         control="SIX controls, each with a pre-registered FAILURE SIGNATURE, "
                 "not merely a pre-registered side. `statue` (do nothing) must "
                 "score worst competence and die of starvation — the dark-room "
                 "objection as a number. `shuffle` (the winning arm's reward "
                 "stream shuffled in time: same magnitude distribution, no need "
                 "semantics) must fail the gate, else the effect was 'any dense "
                 "reward'. `eat` (+1 per consumption event, unbounded) must lose "
                 "AND show the highest drive_cycle_rate of any arm — NE.00 "
                 "measured it to be a genuinely farmable form. `clip` "
                 "(r = max(0, d - d'), which is NetHackEat's SHIPPED reward) must "
                 "lose AND trip the cycling detector — it is the real-world "
                 "instance of the farming pathology and the positive control for "
                 "the static audit. `cc` (cost of deviation, rho = 0) and `col` "
                 "(a plain cost of living) must BOTH fail BY DYING: median "
                 "lifespan below the no-needs null, death-cause distribution "
                 "dominated by voluntary inaction. The tabular case puts suicide "
                 "strictly optimal in 8/11 and 11/11 states respectively, so this "
                 "is a quantitative prediction, not a hope. A control that must "
                 "fail in a SPECIFIC WAY is a stronger instrument than one that "
                 "must merely fail.",
         kills="Nothing on its own — screening declares no winner (the T2.02 "
               "discipline; LT.03/PS.03 precedent). It exists so NE.04 "
               "arbitrates only among arms that demonstrably learned.",
         notes="COVERS: hunger/thirst (claim)\n"
               "STATIC AUDIT, ERROR NOT FAIL: the reward path must be expressible "
               "as an EXACT difference of a state potential. Any max(0, ...), "
               "relu, floor, clip or one-sided term in it sets "
               "reward_is_exact_potential_difference = 0 and the spec is ERROR — "
               "NetHackEat's max(0, delta_nutrition) is a shipped farming exploit "
               "in a NeurIPS benchmark and this is the guard that makes it "
               "impossible here (section 0.2(e)). Runs alongside LT's G1 symbol "
               "audit, which is inherited unchanged. "
               "ARMS: no-needs (NULL), surv (+rho alive only), dr (Keramati & "
               "Gutkin, rho=0), dr+surv (the favourite), dr-modular (one value "
               "head per need — Dulberg PNAS 2023 reports per-drive modules beat "
               "scalarisation beyond ~3 needs and this design has 7, so it is the "
               "most likely way the single scalar d(h) loses), cc+rho "
               "(rho > max_h d(h) ASSERTED BEFORE THE RUN or the arm learns to "
               "die — NE.00(d)), and "
               "dr+surv+pain (the phasic damage signal as a SEPARATE channel with "
               "a FIXED normaliser — section 2.9). The pain arm exists because a "
               "running return normaliser, which T2.00 mandates, is divided by a "
               "standard deviation that the impacts themselves inflate: the more "
               "often he is hurt, the less each injury counts. Biology's "
               "nociceptive system sensitises rather than habituating, and the "
               "fixed normaliser is that property made mechanical. Decided by "
               "pain_habituation = (effective magnitude of a fixed impulse J in "
               "the final fifth of a life) / (the same in the first fifth): "
               "predicted to FALL for the folded arms and stay FLAT for the pain "
               "arm. If both are flat the split is deleted and section 2.1's "
               "one-term-per-need rule survives intact. REOPEN CONDITION, "
               "pre-registered: pain_habituation < 0.5 late in life under the "
               "folded design, or impact events contributing >30% of TD-error "
               "variance, reopens the split whatever the competence numbers say. "
               "SCORING IS AT CLAMPED SETPOINT FOR EVERY ARM so no arm is "
               "measured on its own ruler. MANDATORY VOID GATE, inherited from "
               "PS 3.4: satiated_state_share >= 0.15, else the clamped slice was "
               "never visited in training and the number is distribution shift. "
               "Per-arm VOID conditions inherited unchanged from PS 4.3: "
               "policy_need_sensitivity below its floor (the need never entered "
               "the policy, so the comparison tested nothing), "
               "energy/water_accounting_residual != 0 (ERROR, not VOID — the "
               "instrument is wrong), chaos_occupancy >= 3.0 AND "
               "chaos_reward_ratio >= 2.0, panel_dwell > 0.15 in any seed. "
               "SECONDARY, reported not gated: corr(satiety, exploration) > 0 "
               "(the forage/explore interleave); panel_dwell(dr+surv) <= "
               "panel_dwell(no-needs) — a need should be a noisy-TV antidote, "
               "because a noise panel does not feed you; and "
               "anticipatory_consumption_fraction, the ALLOSTASIS prediction "
               "(section 2.1b) — AgRP hunger neurons are suppressed within "
               "~10-20 s by the SIGHT OR SMELL of food, 96+/-6% complete before "
               "the first bite (Chen et al., Cell 2015), and a "
               "discounted value function should reproduce that without any "
               "anticipatory term in the reward. CORRECTED 2026-08-24 by the "
               "§1.2 citation pass: this line used to say 'hunger AND THIRST "
               "neurons', citing Zimmerman et al. (Nature 2016) — which ran that "
               "exact experiment on SFO thirst neurons (sight of water, "
               "expectation, Pavlovian cue, air licks) and reported NEGATIVE "
               "results on all four. SFO needs liquid in the mouth. The "
               "prediction is unaffected because it only ever needed one "
               "cue-driven channel, but the citation was being used against its "
               "own finding. Its control is the `myopic` arm "
               "(gamma -> 0.5), which must NOT anticipate; if it does, the metric "
               "is reading food availability rather than foresight.\n"
               "POWER GATE BEFORE DISPATCH (26th audit B5, binding the way "
               "XL.01's power verdict binds NE.08): LC.03 v2 measured W0 "
               "unable to separate five learning cores at a 4x envelope with "
               "a clean rig (VOID 'fewer than two learners (1 cleared)', "
               "2026-08-23). NE.03 is a CPU_LONG screening claim in the same "
               "W0, so a pre-run power calculation against LC.03 v2's "
               "recorded spreads (the ledger row plus "
               "experiments/artifacts/lc03_curves_seed*.json on this box) is "
               "REQUIRED before dispatch — and if D10 resolves toward "
               "redesigning W0's discriminability, NE.03 as registered is "
               "measuring the same shallow world; hold it until the redesign "
               "lands. No threshold moves."),

    Spec("NE.04", 5, "BAKEOFF: which need reward, and do innate reflexes help?",
         hypothesis="STAGE 1: among the arms that cleared NE.03, one beats the "
                    "runner-up by >= 1.5 sigma of the pooled seed spread on "
                    "competence_battery_needs_clamped. STAGE 2: on the stage-1 "
                    "winner only, adding a MINIMAL INNATE REFLEX SET (protective "
                    "fall-recovery bias, aversive withdrawal from a pain event, "
                    "grasp-on-contact) beats the same arm without it, and a "
                    "motor-babbling first phase beats starting cold.",
         falsified_by="n/a for a bakeoff — the outcomes are WINNER, TIE (take the "
                      "cheaper arm) or VOID (an arm below the 3-sigma gate, so "
                      "the decision is blocked rather than made). For stage 2 the "
                      "informative negative is real and likely: reflexes tying "
                      "needs-alone means the innate scaffold buys nothing at this "
                      "body scale and is deleted for cost.",
         null_baseline="no-needs, shared across arms and carried forward "
                       "unchanged from NE.03 so all three specs share one floor.",
         metric="competence_battery_needs_clamped", budget=Budget.CPU_LONG,
         depends_on=["NE.03"], seeds=3,
         control="Inherited from NE.03; no arm may enter whose NE.03 result was "
                 "VOID. Stage 2 adds `reflex-only` — the reflex set with the "
                 "policy frozen at init — which MUST fail the competence gate. "
                 "If hand-written reflexes alone clear it, the battery is "
                 "measuring reflexes and not learning, and every stage-1 number "
                 "is uninterpretable.",
         kills="All but one need-reward form; the losers are deleted, not kept "
               "'for later'. And the reflex prior, if it ties.",
         notes="TWO STAGES, NOT A CROSS. A full reward-form x reflex grid is "
               "6 arms x 3 seeds and the box cannot pay for it; the interaction "
               "is therefore UNMEASURED and that is declared in section 9 rather "
               "than hidden. COST UNIT, named before the run because Arm.cost is "
               "None by default and an undeclared cost VOIDs a TIE: CPU-core-"
               "seconds of LEARNER time per 1,000 decisions of lived experience, "
               "measured in-run with time.process_time() around the need-reward, "
               "intrinsic-reward, policy-update AND SLEEP-CONSOLIDATION calls, "
               "EXCLUDING MuJoCo and EXCLUDING the need integrator (both identical "
               "across arms, so including them would compress the differences the "
               "tie-break needs). Same base unit as LT.04/PS.04 on purpose; the "
               "one difference — consolidation is now INSIDE the boundary — is "
               "stated because it is where the sleep arms differ most. Pre-run "
               "estimates: surv 0.4, dr 0.6, dr+surv 0.6, cc+rho 0.6, "
               "+reflex 0.7, +babble 0.8, no-needs 0.4. A TIE therefore resolves "
               "to `surv`, which is exactly why the measurement must replace the "
               "estimate before this runs."),

    Spec("NE.05", 5, "Sleep gates consolidation: biology beats a clock",
         hypothesis="Consolidation that runs WHEN AND BECAUSE Jack sleeps beats "
                    "the same number of consolidation phases and the same total "
                    "gradient steps delivered on a timer, on the competence "
                    "battery and on old-concept retention; and the two jobs of "
                    "sleep dissociate — sleeping without consolidating recovers "
                    "the BODY (p, f) but not the retention, consolidating "
                    "without sleeping recovers neither fully.",
         falsified_by="`timer` ties `sleep-gated` at matched gradient steps. Then "
                      "biology is not a better scheduler than a clock: sleep "
                      "keeps its place for the body, for the night-cold "
                      "curriculum and for relatability, and LOSES its claim as "
                      "the training scheduler. This is the most likely honest "
                      "negative in the needs programme and it must be reportable "
                      "without embarrassment (the UB.14 precedent). Also "
                      "falsified, differently and much more cheaply, by "
                      "`sleep-only` matching `sleep-gated`: then consolidation "
                      "buys nothing at all and stages S1-S4 are deleted.",
         null_baseline="`neither`: no sleep (p frozen at 0), no consolidation. "
                       "The floor both other arms are read against.",
         metric="consolidation_schedule_gain", budget=Budget.CPU_LONG,
         depends_on=["NE.03", "ME.10", "ME.3"], seeds=3,
         control="`empty-buffer` MUST FORGET: sleep runs with the rehearsal "
                 "buffer emptied, and old-concept accuracy must drop far more "
                 "than sleep-gated's <= 2 points (ME.7's pre-registered bound). A "
                 "sleep phase that helps with an empty buffer is not "
                 "consolidating; it is a learning-rate schedule wearing a "
                 "costume. Second control: `random-sleep` — same total sleep "
                 "duration, onsets drawn at random — isolates TIMING while "
                 "holding the body benefit fixed.",
         kills="The sentence 'biology is the training scheduler'. Nothing else in "
               "the needs suite depends on it.",
         notes="COVERS: sleep (claim)\n"
               "DELIBERATELY NOT PARENTED ON ME.7, which depends on T5.03, which "
               "has never run (LESSONS.md: a dependency graph can quietly make "
               "your most important claim unreachable). NE.05 needs the TRIGGER "
               "and the SCHEDULE; those need the playground (PG.8 PASS), the "
               "diary (ME.10 PASS) and the reflections (ME.3 PASS). NE.05 reports "
               "ME.7's old_new_retention number so that when T5.03 lands, ME.7 "
               "can be settled from data this spec already produced. "
               "MATCHING RULE, pre-registered and the place this spec can most "
               "easily fool itself: K for the `timer` arm is set PER SEED to the "
               "sleep-gated arm's REALISED consolidation-phase count, and total "
               "gradient steps are matched to within 2%; wall clock, optimiser "
               "steps and total P_mech are reported alongside (LESSONS.md, "
               "'matched steps has more than one meaning'). "
               "MECHANISM PREDICTION, reported per arm: the timer arm's deaths "
               "should CLUSTER INSIDE its consolidation windows "
               "(deaths_during_consolidation, decisions_lost_to_consolidation), "
               "because on 4 shared cores the trainer and the actor compete. "
               "EMERGENT PREDICTION, reported not gated: sleep_night_alignment "
               "> 1.5 with NO circadian term in the model — he sleeps at night "
               "because night is dark and cold, not because a sinusoid says so."),

    Spec("NE.06", 5, "Sleep restores plasticity (synaptic downscaling)",
         hypothesis="A synaptic-downscaling step at each sleep — w <- alpha*w + "
                    "(1-alpha)*w_init + sigma*eps, alpha = 0.995, trunk only — "
                    "keeps the network trainable across a long life: dormant-unit "
                    "fraction and effective rank stay near their early-life "
                    "values, and LATE-LIFE learning speed on a newly introduced "
                    "goal exceeds the no-downscaling arm's.",
         falsified_by="No difference in late-life learning speed. Then the "
                      "downscaling stage is deleted and sleep has three stages, "
                      "not four. Or, worse and more interesting: downscaling "
                      "helps plasticity metrics while HURTING competence, in "
                      "which case it is trading the skill for the capacity to "
                      "relearn it and the trade must be reported as a trade.",
         null_baseline="Identical agent, identical sleep schedule, downscaling "
                       "stage disabled (alpha = 1.0).",
         metric="late_life_relearn_speedup", budget=Budget.CPU_LONG,
         depends_on=["NE.05"], seeds=3,
         control="TWO, and the second is the important one. (i) DOSE-RESPONSE: "
                 "alpha in {1.0, 0.995, 0.97, 0.9}. Aggressive downscaling MUST "
                 "destroy competence — a knob whose extreme setting changes "
                 "nothing is not connected to anything (LESSONS.md: a threshold "
                 "you never watch fire is not a threshold). (ii) TIMING: the same "
                 "total downscaling applied at RANDOM decisions rather than at "
                 "sleep must be worse or equal. If random timing is just as good, "
                 "sleep is not when this should happen and the biological story "
                 "is decorative even though the intervention works.",
         kills="Stage S4 of the sleep phase, if it ties. Also supplies T5.04 "
               "('plasticity does not die') with a MECHANISM and a SCHEDULE "
               "instead of an intervention someone remembers to run.",
         notes="COVERS: sleep (claim)\n"
               "THE IDENTITY UNDER TEST IS ORIGINAL TO THIS PROJECT AND MUST NOT "
               "BE CITED AS LITERATURE. The biological claim (synaptic "
               "homeostasis: sleep downscales synaptic strength, Tononi & "
               "Cirelli) and the machine-learning claim (shrink-and-perturb "
               "restores trainability in networks whose plasticity has died) each "
               "exist; the assertion that they are the SAME OPERATION does not "
               "appear in the literature — the survey looked. So this spec is not "
               "reproducing a known result, it is testing a hypothesis this "
               "document invented, and the elegance of the story is precisely why "
               "it needs the dose-response and timing controls rather than a "
               "citation."),

    Spec("NE.07", 5, "The social need makes him seek people, not harass them",
         hypothesis="With social contact in the need vector, Jack approaches the "
                    "visitor more when isolated (approach_lift >= 2.0, lower "
                    "bootstrap CI > 1.0), reaches them faster after a long "
                    "isolation (time_to_contact ratio <= 0.5), seeks a PERSON "
                    "rather than a person-shaped stimulus (seek_specificity >= "
                    "2.0 against a decoy), and does NOT pester "
                    "(harassment_ratio <= 1.5 in every seed).",
         falsified_by="approach_lift indistinguishable from the PLACEBO need's "
                      "column: the social variable is decoration, NE.02 deletes "
                      "it, and the companion angle survives purely as a language "
                      "property (NE.09) with no drive behind it — an honest and "
                      "quite defensible outcome. Separately DISQUALIFIED (not "
                      "failed) by harassment_ratio > 1.5 in ANY seed: a lonely "
                      "agent that harasses the user is a failure mode, not a "
                      "feature, and no competence number redeems it.",
         null_baseline="`no-social`: c integrated, logged and in the observation; "
                       "NOT in the reward. Defines the base rate for both "
                       "approach_lift and harassment_ratio.",
         metric="approach_lift_at_bounded_harassment", budget=Budget.CPU_LONG,
         depends_on=["NE.03", "ME.9"], seeds=3,
         control="FOUR, and one must fail UPWARD. `mute-visitor` (never replies) "
                 "must produce ZERO restoration and no sustained approach — else "
                 "the need is restored by proximity to any object. `decoy` (the "
                 "visitor's visual and acoustic signature, no identity, so "
                 "nothing is written to the diary) must restore NOTHING — the "
                 "sensor-gaming control for the social channel. "
                 "`shuffle-provenance` (ME.9's control) must invert 'who helped "
                 "me'. And `no-satiation`: remove the within-bout geometric "
                 "decay and harassment_ratio MUST RISE ABOVE 1.5. A guard that "
                 "cannot be shown to be doing anything is decoration, and this "
                 "is the only way to know which of the three anti-harassment "
                 "mechanisms is load-bearing.",
         kills="The social need, if it does not move behaviour. Or the "
               "within-bout satiation curve, if removing it changes nothing.",
         notes="COVERS: social/other agents (claim)\n"
               "Restoration is a RECORDED WORLD EVENT, never a sensor reading: c "
               "may only rise on an event written to EpisodicMemory with a "
               "channel and a named speaker (PS 5/G-A, generalised). The "
               "anti-harassment design is three layered mechanisms: (1) the need "
               "is BOUNDED and the reward is drive reduction, so restoring an "
               "already-full need pays exactly zero — NE.00 measured that an "
               "unbounded +1-per-interaction bonus is the farmable form; (2) "
               "within-bout geometric decay beta = 0.6, so the sixth consecutive "
               "utterance is worth 7.8% of the first; (3) reciprocation gating — "
               "an unanswered utterance restores nothing. No 'annoyance' variable "
               "is modelled, deliberately: a second hand-tuned quantity would be "
               "one more thing to defend, and `no-satiation` checks whether that "
               "was a mistake."),

    Spec("NE.08", 5, "DEATH AND RETRY: life N+1 is faster BECAUSE he remembers",
         hypothesis="Across >= 8 lives, each in a freshly ACCEL-mutated world, "
                    "with only weights + skill library + EpisodicMemory diary "
                    "carried across death, t_secure (decisions until food AND "
                    "water AND shelter are all secured) falls: crosslife_speedup "
                    ">= 2.0 in >= 2 of 3 seeds AND Spearman rho(t_secure, life) "
                    "<= -0.5 at p < 0.05 per seed. The MECHANISM is the diary: "
                    "wiping it between lives collapses the majority of the "
                    "speedup, and a size-and-distribution-matched diary from "
                    "ANOTHER agent's lives in ANOTHER world does not transfer "
                    "fully. And the two stores dissociate in ME.10's exact shape: "
                    "wiping the diary kills recollection but not competence; "
                    "reverting the weights kills competence but not recollection.",
         falsified_by="TWO independent falsifiers, with different consequences, "
                      "never to be reported as one number. F1 — no downward trend "
                      "in t_secure (Spearman rho > -0.3, or a bootstrap CI on the "
                      "slope containing 0) WHILE all three interpretive gates are "
                      "clean: t_secure finite in >= 60% of lives and below the "
                      "random null (he was competent within a life), cued recall "
                      "on previous lives' death and discovery events >= 0.8 with "
                      "fabricated-event abstention >= 0.95 (the memory was "
                      "available), and consolidation_phases_per_life >= 1 with "
                      "diary_events_distilled > 0 (the mechanism executed). Then "
                      "death is a reset, not a page turn. F2 — C-ONELIFE (one "
                      "continuous life of the same TOTAL decisions, lethality "
                      "disabled) matches the full condition on the shared "
                      "fresh-world probe, CI on the paired difference containing "
                      "0. Then dying contributes nothing beyond the same "
                      "experience without dying, and the loop is pure cost.",
         null_baseline="C-ONELIFE, and it is the null this design would otherwise "
                       "have been missing: the null for 'death teaches' is not "
                       "'no memory', it is THE SAME EXPERIENCE WITHOUT THE "
                       "CLAIMED MECHANISM. Both conditions end on a shared "
                       "fresh-world probe (5 unseen mutated worlds, t_secure in "
                       "each), which is the only ruler on which one long life and "
                       "eight short ones are comparable.",
         metric="crosslife_speedup", budget=Budget.CPU_LONG,
         depends_on=["NE.05", "ME.10", "ME.9", "T6.03"], seeds=3,
         control="C-WIPE (diary deleted at each death; weights and skills kept) "
                 "must collapse the speedup — the diary must contribute the "
                 "MAJORITY of the effect. C-FOREIGN (another seed's diary from "
                 "another mutated world, matched in event count, channel "
                 "distribution and speaker count) must transfer strictly less "
                 "than his own, CI on the paired difference excluding zero; if it "
                 "transfers fully the diary carries GENERIC STRATEGY, not lived "
                 "experience, and the sentence 'he remembers his own life' comes "
                 "out of every capability list. C-SHUFFLE-TIME (his own diary, "
                 "timestamps shuffled) tests whether ordering is load-bearing. "
                 "And the ME.10 double dissociation: D1 wipe-diary must kill "
                 "recollection (to abstention, not confabulation) and SPARE "
                 "competence; D2 revert-weights must kill competence and SPARE "
                 "recollection. EITHER ABLATION KILLING BOTH MEANS ONE STORE IS "
                 "MASQUERADING AS TWO, and the spec records VOID, not a verdict.",
         kills="The death-and-retry loop as a LEARNING mechanism. Under F2 it "
               "survives as a narrative and relatability device and training "
               "moves to one long life, which is also cheaper — a legitimate "
               "product decision that must be made in the open.",
         notes="COVERS: death & retry (claim), memory across lives (claim)\n"
               "THE DISSOCIATION IS OVER {recollection, competence}, NOT over the "
               "speedup. An earlier draft got this wrong: revert the weights and "
               "the agent cannot act, so t_secure is undefined and 'the speedup "
               "survives' is unmeasurable. crosslife_speedup is the COMPOSITE "
               "that requires both stores; C-WIPE and C-FOREIGN attribute it, "
               "D1/D2 prove there are two stores to attribute it between. "
               "TRUNCATION IS NOT DEATH, AND CONFLATING THEM TEACHES HIM THAT "
               "DYING IS FINE. Lives that hit the L_max cap are TIME LIMITS and "
               "must bootstrap V(s_T); lives that end in death must not (Pardo et "
               "al., ICML 2018; Crafter ships this correctly as "
               "info['discount'] = 1 - float(dead)). Censored lives are a "
               "DESIGNED part of this protocol, so the distinction is not a "
               "detail here — half the terminations the agent sees would "
               "otherwise be free, and the death-aversion that section 0.2(d) "
               "says drive reduction supplies for free would be trained away. "
               "Asserted in the spec: every terminal transition carries a "
               "cause tag, and the bootstrap flag is derived from it, never from "
               "the step counter. "
               "t_secure is a MAX over the three components, not a mean: "
               "securing two of three and dying of the third is not survival. "
               "Lives that never secure a component are CENSORED at L_max and the "
               "censoring rate is reported — a mean over uncensored lives makes "
               "an agent that dies early look fast. NO RETURN-TO-FRONTIER: "
               "Go-Explore restores the simulator to an archived state, which is "
               "a free teleport and an experimenter-supplied curriculum (LT 2.1). "
               "The honest version is that the DIARY IS THE ARCHIVE and 'return' "
               "is a behaviour — reported as frontier_return(n), the decisions "
               "taken to re-reach the deepest state of life n-1. World mutation "
               "strength is reported at 0.05/0.15/0.40 as a secondary curve, with "
               "world_distance per life pair, because a speedup that survives only "
               "at 0.05 is coordinate memorisation. Optimiser state is RESET at "
               "death, deliberately, so 'life N+1 learns faster' is not partly a "
               "statement about warm Adam moments. The per-life curve is reported, "
               "never only the endpoints: monotonicity_violations and "
               "worst_life_index, because the reincarnation-RL literature's "
               "standing warning is dependence on teacher quality and here the "
               "teacher is the previous life, which can be pathological. "
               "REGISTRATION PROVENANCE 2026-08-24 (queue cross-check): W.6 was "
               "WITHDRAWN by SURVIVAL_WORLD §5.0 in favour of this spec. XL.01 — "
               "registered 08-19 from the coverage audit, AFTER this doc was "
               "written — tests the same territory with a narrower instrument and "
               "ran FAIL on 08-19 (search_time_ratio 1.003 +- 0.671 vs <= 0.5; "
               "identical fixture read 0.084 on worlds 0-2 and 1.003 on worlds "
               "3-5). XL.01's own B3 verdict stands and BINDS HERE: the "
               "instrument could not resolve 2x at 3 seeds x 8 lives, so the "
               "implementer MUST run a pre-registered power calculation first — "
               "pilot 6-8 worlds at one seed each, size N_LIVES/seeds from the "
               "measured between-world std — and amend the ENVELOPE (never the "
               "thresholds) before the registered run, the BA.02 precedent. "
               "XL.01's FAIL stays in the ledger; this spec is the strengthened "
               "successor (C-ONELIFE null, censoring rules, bootstrap-flag "
               "discipline), per the T1.02 precedent.\n"),

    Spec("NE.09", 6, "He can say how he is, and only what is true",
         hypothesis="Jack's self-reports are a deterministic function of logged "
                    "need values and the diary, and nothing else: per-band "
                    "accuracy >= 0.90 for EVERY need x band cell, abstention "
                    ">= 0.95 on unmodelled interoceptive states ('are you "
                    "dizzy?'), every answer byte-identical to a template "
                    "instantiated with a logged value, attributed answers ('who "
                    "gave you the water?') resolved through ME.9's channels, and "
                    "report_behaviour_agreement >= 2.0 — saying 'I'm cold' "
                    "predicts moving toward shelter within 30 s.",
         falsified_by="Any band cell below 0.90 (gate on the MINIMUM, never the "
                      "mean), OR abstention degrading below 0.95 as accuracy "
                      "rises (fidelity bought with credulity), OR any returned "
                      "string not derivable from a logged value, OR "
                      "report_behaviour_agreement ~ 1.0 — the words and the "
                      "actions are not two readouts of one state, and he is a "
                      "narrator rather than an agent.",
         null_baseline="THE CONFABULATOR: identical reporting machinery with the "
                       "NEED INPUT SEVERED, bands drawn from the marginal "
                       "distribution the agent actually experienced. Same "
                       "sentences, same rate, same fluency. The GAP between the "
                       "reporter and the confabulator is the entire claim.",
         metric="min_band_fidelity_at_fixed_abstention", budget=Budget.CPU,
         depends_on=["NE.03", "ME.9", "ME.1"], seeds=3,
         control="THREE. (i) The abstention list DISABLED must answer the "
                 "unmodelled probes fluently — else abstention was not doing "
                 "work. (ii) ME.9's SWAPPED-PROVENANCE store must invert 'who "
                 "gave you the water' — else the test measures text similarity. "
                 "(iii) SHUFFLED TIMESTAMPS must break 'how long since you ate'. "
                 "And a validity gate rather than a control: if the CONFABULATOR "
                 "scores above 0.70, the probe set is imbalanced and no number in "
                 "this spec means anything.",
         kills="Any self-report path that generates its answer instead of reading "
               "one, however fluent. The extractive-never-generative rule of "
               "ME.11, extended from memory to interoception.",
         notes="report_fidelity alone is 1.0 UNDER EVERY POSSIBLE IMPLEMENTATION "
               "— the band is a deterministic function of the value — which is "
               "T0.12's disease exactly ('ask what the quantity reads when the "
               "mechanism is broken; if that is the same value you are asserting, "
               "the test is decorative'). So the spec is built on the four things "
               "that CAN fail: a band-balanced probe set gated on the per-cell "
               "minimum with n >= 20 per cell (a cell below 20 is VOID for that "
               "band, not passed on the others); abstention on unmodelled "
               "interoception; the confabulator gap; and word-deed agreement, "
               "which is the leg that distinguishes a narrator from an agent and "
               "costs nothing because both quantities are already logged. "
               "Nuisance disqualifier: spontaneous_report_rate <= 6 per sim-hour "
               "per seed — a companion that announces every band crossing is as "
               "bad as one that pesters."),

    # ── LG — LANGUAGE: the anti-puppet family (owner-designed 2026-08-09,
    # registered 2026-08-25 per OVERSIGHT B1(a) / INTEGRATION_QUEUE row
    # LANGUAGE_GROUNDING.md). The research doc is TRUNCATED (§2.2–§11 are
    # headers only); what is registered here is exactly the material that
    # exists owner-designed in the queue plus the one dependency it names.
    # ID COLLISION RECORDED: DIRECTION_AUDIT.md uses 'LG.00' for the
    # eval-certification idea and 'LG.05' for the understanding test; GOAL.md's
    # constitutional citation reserves LG.00 for the anti-puppet claim, so
    # certification is LG.01 here and DIRECTION_AUDIT's numbering is stale
    # prose, not this registry's. LG.05 (understanding test), the grounding
    # bakeoff arms and the ordering experiment stay UNREGISTERED until the doc
    # is completed — registering a design that was never written is the
    # disease coverage.py exists to catch, not a debt to clear blind. ─────────

    Spec("LG.01", 2, "The life-questions are real questions — certified "
                     "lived-necessary",
         hypothesis="Every probe question RETAINED for LG.00 is certified on "
                    "two legs before any arm is scored: a deterministic "
                    "extractive diary-oracle (ME.9's attributed channels, no "
                    "generation) answers it correctly (>= 0.95 on the retained "
                    "set), AND the frozen LLM alone — identical prompt "
                    "scaffold, no diary, no learned core — sits inside its "
                    "pre-registered chance band on it, PER QUESTION, never on "
                    "average. A question the LLM answers from priors is "
                    "EXCLUDED and the exclusion logged; >= 20 questions per "
                    "category (his world / his body / his history) must "
                    "survive exclusion.",
         falsified_by="Any category retaining fewer than 20 questions — the "
                      "generator cannot produce lived-necessary probes and "
                      "LG.00 is unrunnable until it can. OR the diary-oracle "
                      "below 0.95 on retained questions — a question "
                      "unanswerable from the record certifies nothing.",
         null_baseline="The frozen LLM alone, identical prompt scaffold. Its "
                       "per-question accuracy DEFINES the exclusion; it is "
                       "the leg that makes retention falsifiable rather than "
                       "curated.",
         metric="retained_questions_per_category_min", budget=Budget.CPU,
         depends_on=["ME.9"], seeds=3,
         control="The diary-oracle with the diary STRIPPED (empty record, "
                 "same machinery) must collapse to the LLM-alone level on the "
                 "retained set. If it survives, the questions were answerable "
                 "without the lived record and the certification is void.",
         kills="Any LG.00 run scored on an uncertified probe set. "
               "Lived-necessity is a property of the QUESTION, not the model "
               "(LANGUAGE_GROUNDING.md Finding 1, 2603.19233: libero_object "
               "scores 60-100% REGARDLESS of the prompt — a cell where the "
               "prior suffices cannot measure grounding at any scale).",
         notes="PG.7's leak probe transported to Q&A: certify the instrument "
               "before any arm is scored on it. SmolLM2 is cached on this box; "
               "the LLM leg is offline, batched, and NEVER in any control "
               "loop (T0.07's throughput lesson). "
               "  COVERS: language (parent) (fixture)"),

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
               "Knowledge in the parts that lived; language as the mouth. "
               "GOAL.md cites this id verbatim as 'the proof he is a creature "
               "and not a costume'. RT-2's measured 11-point general-knowledge "
               "loss from task-only finetuning (FROZEN_VS_PLASTIC.md §10.8) is "
               "what the control's 'survives untouched' clause guards. "
               "  COVERS: language (parent) (claim)"),

    Spec("LG.02", 5, "Trust is earned by track record — the liar loses him",
         hypothesis="Two advisors speak into his world: one systematically "
                    "truthful, one systematically false, every piece of advice "
                    "verifiable by his own subsequent experience. His "
                    "advice-following DIVERGES by advisor track record: "
                    "follow-rate(truthful) - follow-rate(liar) > 0 at >= 3 "
                    "sigma across seeds by end of life, with attribution "
                    "intact (the diary records who said what, through ME.9's "
                    "channels).",
         falsified_by="Follow rates statistically indistinguishable — he "
                      "cannot learn whom to trust from consequences — or "
                      "divergence achieved with attribution broken (then it "
                      "is not trust, it is something the rig leaked).",
         null_baseline="ATTRIBUTION STRIPPED: same advice stream, no record "
                       "of who said what. This agent must treat both advisors "
                       "identically; if it diverges, speaker identity leaks "
                       "outside the attributed diary and the test measures "
                       "the leak.",
         metric="follow_rate_divergence_by_track_record",
         budget=Budget.CPU_LONG, depends_on=["ME.9"], seeds=3,
         control="THE SWAP (owner-designed): the advisors exchange roles "
                 "mid-life. Trust must MIGRATE to the newly-truthful voice "
                 "within the second half, or the test was measuring voices, "
                 "not veracity.",
         kills="Any scripted-trust design. If trust must be initialised, "
               "annotated or hard-coded to diverge, the emergence claim is "
               "dead and GOAL.md's 'trust earned and checked' is decoration.",
         notes="THE LIAR TEST, owner-designed 2026-08-09, waiting in "
               "INTEGRATION_QUEUE.md since then: 'the emergence stone's first "
               "falsifiable claim: trust earned, checked, and unscripted.' "
               "Advice enters through the world like any parent utterance; "
               "verifiability by his own experience is what makes track "
               "record computable without an annotator. "
               "  COVERS: social/other agents (claim), language (parent) (claim)"),

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
               "as a live test of the swappable-LLM decree. "
               "  COVERS: language (parent) (claim)"),

    # ── BALANCE, SUCCESSOR CLAIM (overseer B1, 48th audit, 2026-08-30) ───
    # Registered BEFORE D8's default fires (2026-08-31) so that parking
    # BA.02 costs the ratchet nothing. This is NOT an amendment of BA.02
    # and NOT a weakening of it: BA.02's claim text, thresholds and PARKED
    # fate are untouched, and D8 option 3 is explicit that a re-scoped
    # scenario is "a NEW spec with new nulls, not an amendment of BA.02".
    Spec("BA.03", 5, "He braces against a surface — balance is used where "
                     "direction still has authority",
         hypothesis="In a scenario where a graspable surface is within reach, "
                    "a learner given BA.01's vestibular channel PLACES ITS "
                    "SUPPORT ON THE LEAN SIDE and stays upright measurably "
                    "longer than an identical learner trained with the channel "
                    "deleted (>= 3 sigma across seeds), and the gain vanishes "
                    "when the channel is replaced by matched-statistics noise.",
         falsified_by="No upright-time gain from having the channel even with "
                      "a surface in reach. Then D8's open-ground finding "
                      "generalises — balance is decoded and never acted on in "
                      "ANY scenario this body affords — and the honest status "
                      "of balance-as-a-used-sense is 'sensed, unused' until "
                      "the playground humanoid exists. A SECOND honest "
                      "outcome: the sensing arm wins but the brace lands on "
                      "the wrong side as often as the right one, which "
                      "refutes the mechanism while the number passes; "
                      "brace-side accuracy is therefore a REPORTED gate, not "
                      "a footnote.",
         null_baseline="The channel-deprived twin's upright time; a random "
                       "policy in the same rig; AND — the null D8's probes "
                       "prove is the binding one — THE BEST FIXED BLIND "
                       "POSTURE. On open ground a constant 'both hands up' "
                       "bought +0.275 s over random, so a sensing arm that "
                       "only beats random has demonstrated nothing about "
                       "sensing. The contrast is against the blind twin "
                       "allowed to find its own best fixed posture.",
         metric="upright_gain_vs_deprived_with_surface",
         # TIER RE-COST 2026-08-30, on the seed-90 pilot's measured wall time
         # (6299 s/seed at N_EVAL 48; ~2.0 h/seed at the pilot-derived 120, so
         # ~6 h for three seeds). CPU_LONG's label is a 2 h timeout that
         # `run.py` ENFORCES by killing the child — the sizing the registry
         # itself demands does not fit inside it. Thresholds unmoved.
         budget=Budget.CPU_DAYS, seeds=3, depends_on=["BA.01"],
         control="MATCHED-NOISE CHANNEL (inherited from BA.02): replace the "
                 "vestibular input with amplitude-matched noise or a shuffled "
                 "replay of another episode's channel; the gain must vanish. "
                 "PLUS a second control D8's evidence makes mandatory — "
                 "REMOVE THE SURFACE and re-run the identical sensing arm: "
                 "the gain must COLLAPSE to D8's measured open-ground ceiling "
                 "(~0.0-0.1 s). If the sensing arm wins with no surface to "
                 "brace against, the rig is not measuring bracing, and D8's "
                 "four scratch probes say it cannot be measuring catching "
                 "either.",
         kills="D8 option 3 — the last re-scoping of balance-as-a-used-sense "
               "that this body affords. If BA.03 fails, no balance CLAIM is "
               "registrable before a body with directional catch authority "
               "exists, and that becomes a finding rather than an assumption.",
         notes="COVERS: balance (claim)\n"
               "WHY THIS IS HONEST AND NOT RATCHET-BUYING, said out loud "
               "because registration is not demonstration (Review, "
               "2026-08-26). D8's four scratch probes measured ONE scenario: "
               "OPEN GROUND. Their finding is that no actuator's useful "
               "effect depends on fall direction THERE — slides +0.09 ± 0.07 "
               "s, adhesion grip +0.005 ± 0.09, the ground-gated 600 N drive "
               "directionally potent only in the HARMFUL direction. A "
               "surface changes the physics of exactly that clause: a hand "
               "pressed against a wall on the lean side supplies a reaction "
               "force the ground-gated drive cannot, and which hand is the "
               "right hand IS the fall direction. D8's own option 3 names "
               "'wall-brace' as a candidate. This spec is that candidate made "
               "falsifiable.\n"
               "WHAT IT DOES NOT CLAIM: nothing here reopens BA.02, gives "
               "the rover catch authority (D8 option 2, a world-contract "
               "change that is the owner's call), or asserts the gain "
               "exists. It asserts only that the question is askable in this "
               "body, which D8's probes did not test.\n"
               "SIZING IS PRE-REGISTERED AS A REQUIREMENT, from D8's second "
               "bullet, and the implementer may not skip it: the registered "
               "CEM learner needed k_fit ~ (2*sigma/S)^2 ~ 119 vs the "
               "registered 3 to resolve even the BLIND signal (per-episode "
               "paired sigma 7.5 decisions vs 1.375 signal), and N_EVAL=48 "
               "puts the margin gate's SE at ~0.22 s against a 0.20 s "
               "threshold. Size k_fit and N_EVAL against MEASURED noise in "
               "the pilot and amend the TIER, never the thresholds — the "
               "LC.03 budget scar and BA.02's own note, and the reason BA.02 "
               "VOIDed three times before D8 was written.\n"
               "TWO CHANNELS SEPARATELY (BA.01's note, ME.11's lesson): "
               "report the linear-acceleration and angular-velocity "
               "contributions apart. A brace carried wholly by one channel "
               "is a finding."),

    # ── THE SURVIVAL WORLD (docs/research/SURVIVAL_WORLD.md §5) ──────────
    # Owner directive 2026-08-09: Jack gets human needs and is thrown into as
    # real a survival world as we can build; he lives, dies, and tries again.
    # Owner correction, same day: "we don't actually need to understand
    # chemistry for this — just like cavemen didn't." So the world's rules are
    # PHENOMENOLOGICAL and the falsifiable property is CONSISTENCY with a
    # PRE-REGISTERED rule, not correspondence with nature. Where an analytic
    # law is available (heat balance) we gate on it exactly as PG.2 gated
    # buoyancy on Archimedes; where it is not (fire, spoilage) the rule text in
    # the spec IS the oracle, and a deliberately-broken variant must be caught.
    #
    # REGISTERED 2026-08-30, twenty-one days after they were drafted and after
    # FIVE consecutive overseer audits asked for it (44th–48th). The delay is
    # itself a recorded finding — LESSONS.md, "Making one kind of debt legible
    # makes the other kind invisible": an unregistered spec has no id, so it
    # sits in no cost class, blocks nothing, satisfies no `depends_on` and
    # appears in no ranking the builder selects from. The World seat in
    # `docs/CHAMPIONS.md` is held **BY VERDICT** — the file's strongest
    # marking — against arenas that did not exist, which is a title with no
    # ring (`champions.py`'s opening line).
    #
    # WHAT THE CROSS-CHECK FOUND, recorded because the INTEGRATION_QUEUE
    # protocol makes it mandatory and because two of the three findings change
    # how an implementer must read the drafted text:
    #   (1) NO ID COLLISION. 188 specs at registration; no `W.*`, no `SV.*`.
    #       Every `depends_on` resolves: PG.1, PG.8, ME.10, ME.11 are live.
    #   (2) `experiments/needs.py` NOW EXISTS (seven-need integrator,
    #       NEEDS_AND_DEATH §2.3, self-test 21/21) and did not when this block
    #       was drafted. It is the substrate W.1/W.2/W.7 gate — these specs
    #       test THAT code, they do not commission a second implementation.
    #   (3) W.2's SOURCED HUMAN DEADLINES ARE NOT THIS WORLD'S CONSTANTS, and
    #       transcribing them unqualified would have registered a spec that
    #       fails on arithmetic. The draft says "thirst 3 days, food 3 weeks";
    #       `NE.01`'s notes record the implemented world killing the statue by
    #       DEHYDRATION at 450 s tank + 120 s grace against starvation's
    #       1,800 + 300 s. Both are right: the first is the human physiology
    #       the constants derive FROM, the second is that physiology after
    #       W.7's compression factor k. An implementer must gate on the
    #       DECLARED constants divided by the DECLARED k, and report both —
    #       which is precisely the failure mode W.7 exists to make impossible.
    #       This is the same class of error as T0.15: the machinery BETWEEN a
    #       measurement and its threshold is part of the gate.
    #
    # BOUNDARIES AGAINST SPECS REGISTERED SINCE THE DRAFT — stated here so no
    # future reader mistakes these for duplicates or for re-litigation:
    #   `PS.02` (thermal, fixture) asserts a temperature field EXISTS, is
    #       SENSED before it kills, and kills. `W.1` asserts that field obeys
    #       the closed-form law we published, on four checks it was not tuned
    #       on, with a control that ignores convection and must be caught.
    #       A thermometer can pass PS.02; only a heat balance passes W.1.
    #   `NE.01` (needs are a real control problem) asserts death is REACHABLE
    #       and spread across needs under random action — a claim about the
    #       policy landscape. `W.2` asserts the meters INTEGRATE to closed form
    #       and CONSERVE — a claim about the bookkeeping. NE.01 can pass over a
    #       leaking ledger; W.2 is what rules that out.
    #   `NE.08` (death and retry) SUPERSEDED `W.6`, which was withdrawn
    #       2026-08-09 for conflating three claims. W.6 IS DELIBERATELY ABSENT
    #       BELOW and must never be registered; the gap in the numbering is the
    #       record. `CHAMPIONS.md`'s World arena cell has been corrected to
    #       name NE.08 in its place — a phantom replaced by a live spec, which
    #       makes the seat MORE contestable, not less.

    Spec("W.1", 2, "Temperature obeys the heat balance we published",
         hypothesis="The thermal overlay reproduces the lumped-capacitance "
                    "solution of m*c_p*dT/dt = Q_gen - h*A*(T - T_env) on four "
                    "independent checks it was not tuned on: (a) the "
                    "PARAMETER-FREE thermoneutral point — a nude 70 kg / 175 cm "
                    "body at 1 met in still air is in balance at 27.55 C, "
                    "within 1.0 C; (b) pure decay from 37 C into 20 C still air "
                    "reads 33.767 C at t=1 h, within 1%; (c) raising wind 0 -> "
                    "5 m/s shrinks tau by the ratio 0.3095, within 2%; (d) "
                    "integrated net flux equals m*c_p*dT to integrator "
                    "tolerance.",
         falsified_by="Any of the four checks outside tolerance, or a "
                      "temperature that is non-finite, or a body that reaches "
                      "equilibrium at a temperature independent of h.",
         null_baseline="Thermal overlay disabled: T stays at its initial value "
                       "forever and every check must fail. Also reported: a "
                       "PURE-AMBIENT model (T := T_env instantly), which passes "
                       "(a) trivially and must fail (b) and (c) — it is the "
                       "cheapest thing that could be mistaken for working.",
         metric="max_thermal_prediction_error", budget=Budget.CPU,
         depends_on=["PG.1", "PG.8"], seeds=3,
         control="A DELIBERATELY BROKEN variant with h_c held constant against "
                 "wind MUST fail check (c) while still passing (a) and (b). If "
                 "the check cannot distinguish a model that ignores convection "
                 "from one that does not, it is certifying a thermometer, not a "
                 "heat balance.",
         kills="Every claim that cold teaches shelter. W.3, W.5's heat coupling "
               "and the whole death-by-hypothermia mechanic are defined over "
               "this model; a wrong one teaches a wrong lesson very "
               "convincingly.",
         notes="This is PG.2's pattern with a different Greek: Archimedes for "
               "water, Newton's law of cooling for air. Constants are sourced, "
               "not invented — 1 met = 58.2 W/m2, Du Bois A = 1.8481 m2 at "
               "175 cm/70 kg, h_r = 4.7, h_c = 3.0 natural / 8.6*(v)^0.53 "
               "forced, neutral skin 33.7 C, neutral core 36.8 C (Gagge two-"
               "node as shipped in CBE pythermalcomfort). c_p = 3470 J/kg/K is "
               "used for reconcilability with ASHRAE/Gagge, and the code must "
               "carry the comment that this is BURTON'S 1935 ASSUMPTION, never "
               "measured; the measured value is 2980 (Xu, Rioux & Castellani, "
               "Temperature 2022, doi:10.1080/23328940.2022.2088034) and "
               "shortens every time constant by 14%. TIME-AVERAGE the "
               "measurement (PG.2's lesson): a body exchanging heat with a "
               "day/night ambient oscillates, and a single sample reads noise. "
               "Run the four checks at 1x wall-clock physics — NOT on the "
               "compressed Jack-day clock, which W.7 governs.\n"
               "REGISTRATION NOTE 2026-08-30: the overlay under test is the "
               "one in `experiments/needs.py`, which did not exist when this "
               "was drafted. Gate the SHIPPED code; do not write a second "
               "thermal model to pass this spec, which would certify the test "
               "rather than the world. Boundary vs `PS.02`: that spec asserts "
               "the field exists, is felt, and kills; this one asserts it obeys "
               "the published law. Neither subsumes the other."
               "  COVERS: thermal (kills) (fixture)"),

    Spec("W.2", 2, "Needs are a conserved ledger, and they can kill",
         hypothesis="Hunger, thirst and sleep pressure integrate to their "
                    "closed-form solutions within 1%; energy in equals energy "
                    "out to 1e-6 relative over a 10-day life; each need "
                    "independently reaches a lethal threshold at the "
                    "pre-registered deadline (thirst 3 days, food 3 weeks, core "
                    "temp outside 28-40 C, EACH DIVIDED BY W.7's DECLARED k and "
                    "both forms reported) when and only when it is not met; "
                    "and sleep pressure discharges 4.3x faster than it "
                    "accumulates (tau_wake 18.2 h vs tau_sleep 4.2 h).",
         falsified_by="Any integrator drifting from closed form beyond 1%, "
                      "energy non-conservation above 1e-6, a need that never "
                      "becomes lethal, or a need that becomes lethal while "
                      "being met.",
         null_baseline="A FROZEN-NEEDS agent whose meters never move: it must "
                       "never die of any need, at any horizon. If it dies, the "
                       "lethality is being driven by something other than the "
                       "needs.",
         metric="needs_ledger_error", budget=Budget.CPU,
         depends_on=["PG.8"], seeds=3,
         control="A SATED agent — fed, watered, rested, at 27.5 C — must "
                 "survive an arbitrarily long life. A needs model that kills "
                 "the sated agent is measuring a clock, not a need. Second "
                 "control: each need ablated in turn must remove exactly its "
                 "own death mode and no other.",
         kills="W.3, NE.08 and the whole death-and-retry loop. A needs system "
               "that does not conserve is a system where Jack can learn to "
               "exploit the bookkeeping instead of the world — the survival "
               "analogue of the noisy TV.",
         notes="The double-counting trap is real and must be asserted against: "
               "1 met x 1.8481 m2 = 107.6 W = 2195 kcal/day is SEATED REST, "
               "already ~25% above BMR (1700 kcal/day = 82 W). A sim that uses "
               "met units and then adds a separate BMR is 25% wrong and nothing "
               "will error. Sourced deadlines: water ~3 days (faster in heat); "
               "food 3-4 weeks (1981 hunger strike: deaths at 46-73 days); "
               "hypothermia bands 32-35 mild / 28-32 moderate (shivering STOPS) "
               "/ 20-28 severe / <20 profound; hyperthermia >=40 C emergency. "
               "The 5/10/15% dehydration ladder is commonly repeated and I "
               "could NOT source it — anchor on the 2% thirst threshold and the "
               "2-4% performance decrement, which have position stands (ACSM, "
               "NATA), and mark the tail as extrapolated in the code.\n"
               "REGISTRATION NOTE 2026-08-30, and it is load-bearing: the "
               "deadlines above are HUMAN PHYSIOLOGY, which is where the "
               "world's constants come from — they are NOT this world's "
               "wall-clock. `NE.01` records the shipped world killing a "
               "do-nothing statue by DEHYDRATION at 450 s tank + 120 s grace "
               "vs starvation's 1,800 + 300 s. Transcribing '3 days' as a "
               "wall-clock gate would fail on arithmetic and would read as a "
               "broken world. Gate on DECLARED constant / DECLARED k, report "
               "both, and read W.7 first. Boundary vs `NE.01`: that spec asks "
               "whether death is reachable and spread (the policy landscape); "
               "this one asks whether the meters conserve (the bookkeeping). "
               "NE.01 can pass over a leaking ledger."
               "  COVERS: hunger/thirst (fixture), sleep (fixture)"),

    Spec("W.3", 2, "Cold kills, and shelter is why it does not",
         hypothesis="Over a scripted night with no agent policy involved — a "
                    "kinematic jig, PG.3's pattern — a Jack inside an insulated "
                    "shelter survives and a Jack outside it does not, and the "
                    "difference in time-to-lethal-core-temperature matches what "
                    "the heat balance predicts from the shelter's declared clo "
                    "value, within 15%.",
         falsified_by="Shelter changes survival time by an amount the heat "
                      "balance does not predict, in either direction — too "
                      "little means the shelter is decorative, too much means "
                      "something other than insulation is being modelled.",
         null_baseline="No shelter (exposed). Also reported: the analytic "
                       "prediction itself, computed from clo and the W.1 model, "
                       "as the ceiling — the gap between simulated and analytic "
                       "IS the metric.",
         metric="shelter_survival_gain_vs_predicted", budget=Budget.CPU_LONG,
         depends_on=["W.1", "W.2"], seeds=3,
         control="A ZERO-INSULATION shelter — geometrically identical, clo = 0 "
                 "— MUST NOT extend survival. If a shelter helps because it is "
                 "a box rather than because it insulates, the spec is measuring "
                 "occlusion or a collision artefact, and every later "
                 "shelter-building claim would inherit the error.",
         kills="The sentence 'cold nights teach shelter-building'. If insulation "
               "does not measurably change survival, no policy can learn to "
               "seek it and the W1 curriculum has no gradient.",
         notes="Deliberately scripted, not learned. This certifies that the "
               "WORLD contains the lesson, before any spec asks whether Jack "
               "learns it — the same separation PG.3 drew between 'the ladder "
               "is climbable in principle' and 'Jack climbs it'. LESSONS.md's "
               "'a world that passes physics tests may still have nobody living "
               "in it' cuts the other way here: verify the affordance exists "
               "before spending GPU on an agent to find it. 1 clo = 0.155 "
               "m2K/W; a brush shelter is worth roughly 1-2 clo and the spec "
               "must declare which before the run.\n"
               "REGISTRATION NOTE 2026-08-30: `SH.02` is the CLAIM that Jack "
               "builds and uses shelter; this is the FIXTURE that the shelter "
               "affordance is real and quantitatively right. Running SH.02 over "
               "an unverified W.3 would let a decorative shelter be learned as "
               "if it worked."
               "  COVERS: shelter/building (fixture), thermal (kills) (fixture)"),

    Spec("W.4", 2, "The rule-set is consistent and discoverable",
         hypothesis="Every rule in the world's published rule-set is (a) "
                    "CONSISTENT — replaying an identical (state, action) pair "
                    "from a serialised state produces a BIT-IDENTICAL outcome, "
                    "over >=200 sampled rule firings; (b) DISCOVERABLE — a "
                    "uniform-random policy fires every rule at least once "
                    "within a pre-registered step budget; and (c) CONSEQUENTIAL "
                    "— every rule moves at least one need meter by more than "
                    "the meter's own noise floor.",
         falsified_by="Any rule whose replay diverges (hidden state or unseeded "
                      "randomness), any rule unreachable by random exploration "
                      "inside the budget, or any rule that moves no need.",
         null_baseline="A DELIBERATELY NONDETERMINISTIC world in which one rule "
                       "consults an unseeded RNG: check (a) must catch exactly "
                       "that rule and no other. This null is the spec's primary "
                       "assertion — a consistency checker that cannot find a "
                       "planted inconsistency is not a checker.",
         metric="rule_consistency_x_discovery_rate", budget=Budget.CPU_LONG,
         depends_on=["PG.8", "W.2"], seeds=3,
         control="A DECORATIVE rule — one deliberately wired to move no need — "
                 "must be flagged by (c). And an ADVERSARIALLY DEEP rule, gated "
                 "behind a 6-step precondition chain, must FAIL (b) at the "
                 "declared budget. If everything passes discoverability, the "
                 "budget is too generous to mean anything.",
         kills="Any rule that fails (a). A world Jack cannot learn is not a "
               "curriculum, it is noise with a tech tree. Rules failing (b) or "
               "(c) are demoted to scenery and must not be counted in W.8's "
               "depth metric.",
         notes="This spec replaces 'realism' as the world's quality criterion, "
               "per the owner's caveman correction. Report PER RULE and gate on "
               "the MINIMUM, never the mean — ME.11's lesson: an aggregate hides "
               "the stratum the logic has deleted, and a rule-set of 40 rules "
               "with one broken rule averages to 97.5% and reads as healthy. "
               "Discoverability budget must be pre-registered BEFORE the run "
               "and stated in env-steps, with wall-clock and control-steps also "
               "reported (T2.02's 'matched steps has more than one meaning').\n"
               "REGISTRATION NOTE 2026-08-30: gate on the worst rule INSIDE "
               "`_experiment`, not in `_check`. `protocol.py:_aggregate` means "
               "over spec-level seeds before `_check` runs, so a per-rule "
               "minimum computed at the spec level is silently averaged away — "
               "the exact defect found in T3.06 on 2026-08-30 and now in "
               "LESSONS.md."),

    Spec("W.5", 2, "Fire obeys its published rules",
         hypothesis="The fire state machine pre-registered in "
                    "docs/research/SURVIVAL_WORLD.md section 4.2 holds on every "
                    "clause: dry fuel ignites and wet fuel (w >= W_IGNITE) does "
                    "not; rain above R_QUENCH moves BURNING -> EMBERS; fuel is "
                    "consumed at the declared rate and the cell reaches ASH at "
                    "the predicted time; wind biases spread probability in the "
                    "declared direction; and a BURNING cell raises Jack's core "
                    "temperature by the amount W.1's model predicts for its "
                    "declared power and distance.",
         falsified_by="Any clause violated, OR the heat coupling disagreeing "
                      "with W.1's independent prediction — which would mean two "
                      "parts of the world disagree about the same physics.",
         null_baseline="Fire disabled: no ignition, no heat, no fuel consumed. "
                       "Also reported: a fire that ignores wetness entirely, "
                       "the single most likely implementation shortcut, which "
                       "must fail the wet-fuel clause.",
         metric="fire_rule_conformance", budget=Budget.CPU,
         depends_on=["W.1"], seeds=3,
         control="A BROKEN variant in which rain does not quench MUST be caught "
                 "by the rain clause while passing every other clause. A "
                 "conformance test that only reports an aggregate cannot "
                 "localise a broken clause, and localisation is the whole value "
                 "(LESSONS: a control that fails alongside the experiment is a "
                 "gift).",
         kills="Cooking, warmth-seeking, and the entire fire branch of the tech "
               "tree. Also kills any claim that Jack 'discovered fire' — "
               "discovery of an inconsistent rule is memorisation of noise.",
         notes="The rule text in section 4.2 IS the oracle. This is the point of "
               "the caveman reframing: we are not approximating combustion, we "
               "are asserting that the implementation obeys a rule we wrote "
               "down first. Deliberately Minecraft-shaped rather than "
               "Rothermel-exact; Rothermel's R = (I_R*xi)/(rho_b*eps*Q_ig) * "
               "(1 + Phi_w + Phi_s) is the source of the 'base rate times a "
               "dimensionless wind-and-slope multiplier' SHAPE, and nothing "
               "more is claimed. Note that even wildfire science does not "
               "simulate fire's chemistry.\n"
               "REGISTRATION NOTE 2026-08-30: the rule block in section 4.2 is "
               "the ORACLE and it is PROSE. Copy its constants into the test as "
               "declared parameters and assert against those, never against a "
               "re-reading of the document at run time — and if the two ever "
               "disagree, that disagreement is the finding. Report PER CLAUSE "
               "and gate on the minimum, for W.4's reason."),

    # ~~W.6~~ WITHDRAWN 2026-08-09 in favour of NE.08 in
    # docs/research/NEEDS_AND_DEATH.md, which separates the three claims W.6
    # conflated and adds the C-ONELIFE control. The struck-through draft is
    # retained in docs/research/SURVIVAL_WORLD.md §5 so the reasoning trail
    # survives. DO NOT REGISTER IT. This gap in the numbering is deliberate and
    # is the record; `CHAMPIONS.md`'s World arena cell names NE.08 in its place.

    Spec("W.7", 2, "Time compression is a declared transformation, not a fudge",
         hypothesis="With the day-length compression factor k declared (proposed "
                    "k = 72, one Jack-day = 1200 s of sim time), the physics "
                    "integrates in REAL seconds and only the need-accumulation "
                    "clock is scaled; the dimensionless ratios the spec declares "
                    "(thermal tau / day-length, thirst deadline / day-length, "
                    "sleep tau_wake / day-length) equal their declared values to "
                    "1e-9; and W.1's four analytic checks, re-run inside the "
                    "compressed world, give BIT-IDENTICAL results to the 1x "
                    "fixture.",
         falsified_by="Any declared ratio off by more than 1e-9, or W.1's checks "
                      "moving at all when k changes — which would prove the "
                      "compression is inside the physics rather than beside it.",
         null_baseline="k = 1 (no compression): every ratio must equal its "
                       "k = 1 value and the whole spec must be trivially "
                       "satisfied. A compression test that cannot tell k = 1 "
                       "from k = 72 is measuring nothing.",
         metric="compression_invariance_error", budget=Budget.CPU, seeds=1,
         depends_on=["W.1", "W.2"],
         control="A NAIVE-COMPRESSION variant that scales the physics timestep "
                 "instead of the needs clock MUST fail W.1's decay check, "
                 "because a 70 kg body's tau = 4.74 h is a property of m*c_p/hA "
                 "and cannot be sped up by a clock convention. This is the "
                 "specific mistake the spec exists to make impossible.",
         kills="Every cost estimate in SURVIVAL_WORLD.md. Without compression a "
               "single 24 h life is 1.70 core-hours (measured) and no 3-seed "
               "study is affordable; with UNVERIFIED compression every thermal "
               "number in the ladder is silently wrong.",
         notes="Same family as T0.15: the machinery BETWEEN a measurement and "
               "its threshold is part of the gate. Here the machinery is a unit "
               "conversion applied to time, which is exactly the class of thing "
               "that passes review and fails silently. Assert the ratios "
               "against the DECLARED constants, not against each other — two "
               "quantities derived from the same wrong k agree perfectly.\n"
               "REGISTRATION NOTE 2026-08-30: THIS SPEC IS ALREADY OWED A "
               "MEASUREMENT. `NE.01` records the shipped world killing a "
               "do-nothing statue of dehydration at 450 s + 120 s grace, "
               "against W.2's sourced human deadline of ~3 days. That ratio is "
               "a k, whether or not anybody declared it — so k is already in "
               "the code and this spec's job is to find out whether it was "
               "DECLARED or merely IMPLIED. An undeclared k is the failure "
               "mode, not a missing feature."),

    Spec("W.8", 5, "Rule depth: the world contains more than we enumerated",
         hypothesis="The count of distinct REACHED rule-interaction events — "
                    "co-occurrences of two or more rules whose joint outcome "
                    "differs from either rule's outcome alone — exceeds the "
                    "number of rules enumerated in the rule-set, and keeps "
                    "growing over lives rather than saturating.",
         falsified_by="The reached-interaction count saturating at or below the "
                      "enumerated rule count — the world is a list, not a "
                      "closure, and open-endedness in it is impossible however "
                      "good the agent is.",
         null_baseline="An INTERACTIONS-DISABLED world in which each rule's "
                       "outcome is computed independently and composed by "
                       "overwrite. Its reached-interaction count must saturate "
                       "at ~0 by construction, and measuring it anyway is what "
                       "makes the main number interpretable.",
         metric="reached_interaction_growth", budget=Budget.CPU_LONG,
         depends_on=["W.4", "W.5"], seeds=3,
         control="A SCRIPTED-EXHAUSTION agent that fires every enumerated rule "
                 "once and stops must score at most the enumerated count. If a "
                 "trivial rule-firing script scores as highly as a living "
                 "agent, the metric counts rule firings rather than "
                 "interactions and proves nothing about depth.",
         kills="The W3 rung of SURVIVAL_WORLD.md's fidelity ladder. If depth "
               "does not exceed the enumeration, W3 is just W2 with more table "
               "rows, and the honest move is to say so and spend the compute on "
               "ACCEL-style RULE mutation instead of on more hand-authored "
               "rules.",
         notes="NOT parented on T5.08 (open-endedness), deliberately. W.8 asks "
               "whether the WORLD has depth, which a scripted and a random "
               "agent can answer with no learning at all; T5.08 asks whether an "
               "AGENT keeps finding it, which is GPU_LONG and unrun. Parenting "
               "a world-property claim on an agent result is exactly how UB.1 — "
               "the project's namesake claim — ended up unreachable behind a "
               "locomotion failure. Caught 2026-08-09 by running this block's "
               "depends_on against the live registry rather than eyeballing it. "
               "The metric's difficulty is definitional, not computational: "
               "'an outcome that differs from either rule alone' must be "
               "pinned to a comparison the implementer writes down BEFORE the "
               "run, or the count is whatever the counter felt like counting.\n"
               "REGISTRATION NOTE 2026-08-30: registered at TIER 5 as drafted, "
               "which means it is a CLAIM about the world's open-endedness, not "
               "a fixture. It sits behind W.4 and W.5 and is therefore the last "
               "of this family to become runnable — that ordering is the point, "
               "since depth measured over inconsistent rules is noise."),

    Spec("W0.DIAG", 2, "The exploration process, not the world: does correlated "
                       "random action buy life in W0?",
         hypothesis="Part of W0's nine-instrument shallowness reading is "
                    "exploration-process-limited, not world-limited: a random "
                    "policy whose actions are TEMPORALLY CORRELATED colored "
                    "noise — per-decision marginal action distribution "
                    "identical to LC.03's `random` null, correlation time "
                    "scheduled near-white -> red across the run — records a "
                    "positive within-run `life_gain` that the stationary "
                    "nulls cannot, because sustained action moves the body "
                    "far enough to reach food that per-decision white noise "
                    "never does.",
         falsified_by="Either failure branch, each named in the recorded "
                      "metrics (`claim_branch` — the BA.03 one-bit-verdict "
                      "lesson): (a) the paired margin (life_gain of the "
                      "correlation-ramp-UP run minus the plain random null, "
                      "per seed) under 3 sigma with every instrument gate "
                      "green — correlation buys no measurable life; or (b) "
                      "the margin fires but the up-run shows NO food "
                      "advantage over the null (mean eats), so the gain is "
                      "the PASSIVITY channel — correlated noise does less "
                      "work and drifts toward statuehood, which is LC.03's "
                      "already-measured gradient, not exploration reaching "
                      "reward. On either branch the shallowness finding "
                      "survives its cheapest attack.",
         null_baseline="LC.03's stationary `random` and `random-repeat` nulls "
                       "re-run at this envelope: a stationary policy has no "
                       "within-run trend, so its life_gain reads zero up to "
                       "seed noise — that spread is what the scheduled run "
                       "must clear, and a stationary null that itself trends "
                       "at 3 sigma VOIDs the reading (the world, not the "
                       "schedule, would be moving).",
         metric="colored_vs_white_life_gain_margin_sigma",
         budget=Budget.CPU, seeds=3, depends_on=["PS.01"],
         control="THREE, each VOIDing rather than FAILing — an instrument "
                 "fault is not a world reading (T0.22). (1) KNOWN-ANSWER, "
                 "BINDING per field-watch wk5-N3: the same life_gain readout "
                 "pointed at a scheduled gradient whose answer is certified "
                 "arithmetic (respawn energy ramped up across the run under "
                 "the plain random policy; life scales with e0 — LC.03 "
                 "measured statue life = e0/BASAL_B to 0.02%) must read "
                 "positive at >= 3 sigma against the fixed-e0 null, else the "
                 "instrument cannot see a real scheduled survival gradient "
                 "and its W0 reading is not believed. (2) The REVERSED "
                 "schedule (red -> white) must NOT read the same sign at "
                 "+3 sigma — if ramp-down also 'gains', the channel measures "
                 "run-time drift, not correlation. (3) MANIPULATION: the "
                 "ramp-UP run's final-third per-decision jitter must FALL "
                 "below its own first-third at >= 3 sigma — the schedule's "
                 "measured signature in this body (pilot, seed 90/91: "
                 "correlation converts white dithering into sustained "
                 "motion, per-decision jitter drops ~2x monotonically in "
                 "tau) — a schedule that never expressed in the body never "
                 "tested its premise.",
         kills="On FAIL: nothing — a FAIL is itself the finding (the "
               "shallowness survives the cheap attack) and feeds the "
               "`w0-too-shallow` design due 2026-09-06. On PASS: the "
               "UNQUALIFIED reading of the nine instruments dies — 'W0 does "
               "not reward capability' must be re-read as partly 'no tested "
               "policy's exploration process ever reaches what W0 does "
               "reward', which lands in D10 fork (b) and reprices the "
               "edit-W0-vs-build-W1 fork before a redesign is spent on it.",
         notes="ORDERED by the Review 2026-08-25 (accepted field-watch wk4-N3) "
               "and sequenced BEFORE any W1 redesign; queue row written "
               "2026-08-31 after six days as prose nobody could act on. The "
               "REJECTED half of wk4-N3 — an exploration arm on A0/A1 — "
               "stays rejected (PPO likelihood-ratio reason, 08-25). "
               "IMPLEMENTATION CHOICE, recorded at design time: the colored "
               "policy is AR(1) in a latent Gaussian passed through "
               "erf(z/sqrt(2)), which gives EXACTLY the null's uniform(-1,1) "
               "marginals at every decision and at every correlation time — "
               "the matched-magnitude lesson (T3.06, 2026-08-31) designed in "
               "rather than retrofitted; spectral 1/f^beta synthesis cannot "
               "hold the marginal fixed under clipping, so 'beta-scheduled' "
               "is realised as a geometric correlation-time ramp, stated "
               "plainly. Every gate is a scale-free t-statistic over seeds "
               "(house _tstat idiom); the disjoint-seed pilot fixes ONLY the "
               "envelope (n_decisions, e0, tau range), so no order-statistic "
               "bar is frozen at a pilot n (the T3.06 extreme-value lesson). "
               "Recorded limit of the known-answer control, carried from the "
               "Review's own acceptance: the RWG/PIC/POIC inversion paper "
               "(arXiv:2602.18856) motivates the check by ANALOGY, not "
               "arithmetic — it does not refute this instrument and may not "
               "be cited as if it did. TWO PILOT FINDINGS built into the "
               "gates before registration (seeds 90/91, disjoint; full "
               "record in the test docstring): correlation LOWERS "
               "per-decision jitter in this body (0.066 -> 0.015 m across "
               "tau 0->32) while a fixed tau=32 run ate 4 food quanta and "
               "completed ZERO lives in 240 sim-s where white noise ate 0 "
               "and died at ~41 s — so the food route exists, the passivity "
               "confound exists (less jitter = less work = statue-ward "
               "drift, LC.03's known gradient), and the PASS therefore "
               "requires the mean-eats advantage, not life_gain alone."),

    Spec("T0.28", 0, "The escalation tool can be shown catching a deadlock "
                     "and a claim-death",
         hypothesis="`experiments/decisions.py` — the instrument that stands "
                    "over eleven pre-registered constitutional defaults — "
                    "detects every defect it claims to detect, on the real "
                    "code path, in BOTH directions: (1) the four parse "
                    "defects (UNDECLARED, MEANS-ESCALATED, NO-DEFAULT, an "
                    "option-stale header that is not a resolution) are "
                    "flagged and a correctly armed entry is not; (2) the "
                    "SAFETY-CLAIM-DEAD clause fires on `D8` as it actually "
                    "stood on 2026-08-29 and drives `check_rc` to 1; (3) it "
                    "goes quiet only for the prescribed repair — a registered "
                    "SUCCESSOR — and a commitment whose last claim VANISHES "
                    "is distinguishable from one that was repaired; (4) a "
                    "recorded PASS is never put at risk by a calendar; (5) an "
                    "id that resolves to nothing is a typo and not a "
                    "reference, while a real id anywhere in the default's "
                    "text — including on a wrapped continuation line — is in "
                    "the blast radius; (6) the gate's exit code blocks on "
                    "EVERY violation class it reports as fatal, not on one.",
         falsified_by="Any property failing; the battery running fewer than "
                      "N_PROPERTIES; or the live document failing to parse "
                      "into a well-formed armed set. Concretely: the safety "
                      "clause not firing on the D8 shape, going quiet when "
                      "the subject vanishes rather than when a successor is "
                      "registered, a blast radius that drops a spec id "
                      "because it fell on the second physical line of a "
                      "wrapped default, or a violation class the report "
                      "prints and the exit code ignores.",
         null_baseline="THE ORGAN AS IT STOOD BEFORE 2026-08-30, kept "
                       "executable as the control: `audit()` with no safety "
                       "pass at all, and a `--check` whose blocking set omits "
                       "NO-DEFAULT. It is not a tidied restatement — it is "
                       "the exact pair of holes this file closed, replayed. "
                       "It MUST miss the D8 known-positive, MUST miss the "
                       "both-named case, and MUST exit 0 on a goal-class "
                       "entry that arms nothing.",
         metric="properties_failed", budget=Budget.CPU, seeds=1,
         depends_on=[],
         control="See null_baseline. Named properties "
                 "p2_d8_known_positive_fires, p4_both_named_fires and "
                 "p9_ratchet_counts_every_class must be among the control's "
                 "failures or the control no longer reproduces the disease "
                 "and this spec guards nothing.",
         kills="Author self-certification of the governance instruments. "
               "Before this spec `decisions.py` was certified only by "
               "fixtures its own author wrote, which is what `SYSTEM.md`'s "
               "first law exists to distrust — and it had already been wrong "
               "once, for six days, in the direction of claiming an "
               "enforcement it did not perform.",
         notes="SCAR (49th audit, 2026-08-30, RANK 1 / B1): two of the three "
               "tools every audit opens with had no ledger certificate, and "
               "one of them gained 186 lines of constitutional enforcement "
               "four hours before eleven defaults were due to fire. The "
               "known-positives are historical events, not synthetic: `D8`'s "
               "default read 'PARK BA.02' while `BA.02` was the only "
               "claim-kind spec behind `balance` (GOAL.md:41), and `BA.03` "
               "was registered on the morning of 08-30 — which is the "
               "prescribed repair and the reason the live check is green. "
               "The fixture is synthetic ON PURPOSE (rows are injected via "
               "`audit(rows_for_safety=...)`): pinning it to the live ledger "
               "would make the guard go quiet the moment the repo is "
               "repaired, which is how a guard ends up green because its "
               "subject vanished rather than because it was fixed — the very "
               "failure P5 exists to detect, one level up. Companion spec "
               "T0.29 (`champions.py`) is owed and not written.\n"
               "Deliberately declares NO `COVERS:` commitment. It guards the "
               "decision machinery, not a capability."),

    Spec("T0.29", 0, "The seat tool cannot be discharged by deleting the ring",
         hypothesis="`experiments/champions.py` — the instrument that decides "
                    "whether each seat in Jack's anatomy is CONTESTABLE — "
                    "detects every defect it claims to detect, on the real "
                    "code path, in both directions: (1) the three defect "
                    "classes fire and two healthy seats do not; (2) its "
                    "RATCHET is invariant under deleting an arena reference, "
                    "and falls when a spec is registered — the quantity is "
                    "`UNFALSIFIABLE` (seats with no runnable arena at all), "
                    "never the ARENA-MISSING count; (3) a ref the project "
                    "DECIDED against reports CORRECT-THE-CITATION and a "
                    "merely-unwritten one reports REGISTER, with a mixed cell "
                    "saying both; (4) a seat is discharged only by a "
                    "CHALLENGER — a VOID is not a verdict and a "
                    "fixture/rule/sensor seats nobody; (5) a cited RANGE "
                    "expands, so one string naming seven arenas is not "
                    "counted as two; (6) a decree outside the table is still "
                    "a seat, and a non-decree heading is not.",
         falsified_by="Any property failing; the battery running fewer than "
                      "N_PROPERTIES; or the live document failing to parse "
                      "into a well-formed seat list. Concretely: deleting a "
                      "phantom arena id lowering the ratchet, a withdrawn "
                      "spec drawing the instruction 'register it', a seat "
                      "discharged by an arena whose only completion was VOID, "
                      "`W.1`–`W.7` counting as two arenas, or the "
                      "PLASTIC-ONLY decree's `PL.00` going unreported because "
                      "it lives outside the table.",
         null_baseline="THE ORGAN AS IT STOOD BEFORE 2026-08-29, kept "
                       "executable as the control, with the three holes it "
                       "carried: a ratchet counting ARENA-MISSING alone, no "
                       "closability split (`unregisterable={}`), and the "
                       "`all(status == NOT_RUN)` challenger quantifier. Each "
                       "is reconstructed BY DELETION rather than by "
                       "paraphrase (T0.08 property 5). It MUST reward the "
                       "reference-deletion, MUST tell the builder to register "
                       "a withdrawn spec, and MUST call a fixture-answered "
                       "seat defended.",
         metric="properties_failed", budget=Budget.CPU, seeds=1,
         depends_on=[],
         control="See null_baseline. Named properties "
                 "p2_deleting_the_arena_does_not_help, "
                 "p4_unregisterable_is_not_a_todo and "
                 "p6_only_a_challenger_discharges must be among the "
                 "control's failures, or the control no longer reproduces "
                 "the disease and this spec guards nothing.",
         kills="Author self-certification of the governance instruments — the "
               "second and last of them. `champions.py` has had four real "
               "defects found in it in six days and was certified only by a "
               "fixture its own author wrote, which is what `SYSTEM.md`'s "
               "first law exists to distrust.",
         notes="SCAR (49th audit, 2026-08-30, RANK 1 / B1, companion to "
               "`T0.28`). The known-positives are recorded events, not "
               "inventions. (a) The RATCHET: until 2026-08-30 `--check` "
               "counted `ARENA-MISSING` alone, so deleting a phantom id "
               "converted the seat to `NO-ARENA` and the number FELL — the "
               "ratchet rewarded the one repair the module's own docstring "
               "forbids in bold, and three seats (ASR, Speaker ID, Language "
               "grounding) were already sitting in that blind spot. Same "
               "one-class shape as `decisions.py`'s `NO-DEFAULT` (closed by "
               "T0.28 P9) and `coverage.py`'s (closed by T0.21 P2): third "
               "instrument, same disease, and the 40th and 47th audits both "
               "named it. (b) CLOSABILITY: `W.6` was withdrawn 2026-08-09 and "
               "sat inside the cited range `W.1`–`W.7`, so five consecutive "
               "audits (44th–48th) relayed 'register W.1-W.7' — an "
               "instruction one component of which could not be obeyed by any "
               "amount of honest work. (c) THE QUANTIFIER: `all(v == "
               "'NOT_RUN')` discharged a seat the moment ANY arena spec had "
               "run, including a `fixture`, a `sensor` or a VOID; carried "
               "unrepaired by the 43rd, 44th and 45th audits over a cell "
               "reading, in bold, 'DEFAULT, never defended'.\n"
               "The fixture is synthetic ON PURPOSE, for T0.28's reason: a "
               "known-positive pinned to the live document stops being "
               "exercised the moment somebody repairs the document, which is "
               "how a guard ends up green because its subject vanished. P5 "
               "and P10 carry the live half.\n"
               "WHAT THIS DOES NOT CERTIFY, stated so no reader repeats "
               "`SYSTEM.md`'s mistake in this spec's name: seat MARKINGS are "
               "still INFERRED from a table column and a prose fallback, not "
               "declared. `champions.py`'s own docstring says so. A battery "
               "cannot close that; a `HELD:`/`ARENA:` declaration syntax "
               "would, and is owed.\n"
               "Deliberately declares NO `COVERS:` commitment. It guards the "
               "decision machinery, not a capability."),

    Spec("T0.30", 0, "A regression gate cannot demote the certificates it re-runs",
         hypothesis="`run --gate` — the one command in this runner whose only "
                    "possible effect on the record is to REPLACE clean stamps "
                    "— refuses to start from a code-dirty working tree, on "
                    "the shipped command line, and the predicate it refuses "
                    "by is exact: (1) an uncommitted code file refuses and is "
                    "NAMED; (2) the runner's own outputs (`ledger.json`, "
                    "`gpu_budget.json`, `gpu_submissions.jsonl`, its `.tmp`) "
                    "and the loop's docs (`CHECKLIST.md`, `LOOP_JOURNAL.md`) "
                    "do NOT refuse, so the gate cannot deadlock against files "
                    "it writes itself; (3) a clean tree gates normally; (4) "
                    "`--dirty-ok` is an explicit opt-in that gates anyway and "
                    "warns; (5) the refusal reports how many PASS rows were "
                    "at risk; (6) end to end, in a scratch clone, the dirty "
                    "gate exits non-zero and dispatches NOTHING while the "
                    "clean gate dispatches.",
         falsified_by="Any property failing; the battery running fewer than "
                      "N_PROPERTIES; or the scratch clone failing to build, "
                      "which makes P6-P8 unevaluated rather than passed. "
                      "Concretely: the gate starting on a tree with "
                      "uncommitted code, a runner-output-only tree refusing "
                      "(the `gpu_budget.json` self-deadlock, one surface "
                      "over), `--dirty-ok` being the default, or the refusal "
                      "firing after `_exclusive` rather than before it.",
         null_baseline="THE GATE AS IT STOOD BEFORE 2026-08-30, kept "
                       "executable as the control and reconstructed BY "
                       "DELETION (T0.08 property 5): a second clone whose "
                       "`protocol.py` carries an appended "
                       "`gate_precondition(*_, **__) -> ''`, so the guard is "
                       "present, imported, called, and never fires. That "
                       "append itself dirties the control clone, which is "
                       "precisely the condition under test. It MUST let "
                       "`--gate` start from a modified tree, and it MUST fail "
                       "P1 and P5.",
         metric="properties_failed", budget=Budget.CPU, seeds=1,
         depends_on=[],
         control="See null_baseline. Named properties "
                 "p1_uncommitted_code_refuses, p5_refusal_reports_exposure "
                 "and p6_dirty_gate_dispatches_nothing must be among the "
                 "control's failures, or the control no longer reproduces "
                 "the disease and this spec guards nothing.",
         kills="The practice of running the regression gate before "
               "committing. After this spec that ordering costs one refusal "
               "line instead of a demoted certificate, and choosing it "
               "requires typing a flag whose help text says what it loses.",
         notes="SCAR, 2026-08-30 09:19 UTC, commit `7966524`, and it is this "
               "iteration's own inheritance rather than an audit finding. The "
               "builder edited `champions.py`, ran `--gate` to check for "
               "regressions, and committed afterwards. Ten specs passed and "
               "all ten recorded `e9bd4a0+dirty`; for `T0.08` and `T0.09` the "
               "recorded `impl_sha` reconstructed from no committed blob, so "
               "both PASSes became DIRTY STAMPS. `T0.09` is a dependency of "
               "36 specs, so for three hours `run blocked` reported a phantom "
               "— *'T0.09 = PASS but STALE, frees 36'* — ABOVE the project's "
               "real top blocker (`T2.01`, frees 35), and clearing it cost "
               "two re-runs and a second Colab T4 round-trip. The gate was "
               "green; the ladder got worse.\n"
               "The class, which is what makes this a spec and not a fix: a "
               "single-spec run from a dirty tree merely fails to certify, "
               "and that is normal — `t0_23`'s fixture note says a dirty tree "
               "is the ordinary state of the iteration that runs it. A GATE "
               "run re-runs rows that ALREADY hold clean stamps, so its "
               "expected information gain from a dirty tree is negative by "
               "construction and `blocked_by` propagates the loss. The same "
               "event is on record twice under other names — `T2.00`'s "
               "`08444b2+dirty` (998-second re-run, 47 specs blocked) and "
               "`T0.25`'s `1ddcd27+dirty` — both repaired as incidents, "
               "neither as a class.\n"
               "NOTHING here weakens a stamp: `+dirty` fires on exactly the "
               "same condition as before. The guard only refuses to "
               "volunteer for it, and `--dirty-ok` keeps the legitimate "
               "'does my WIP break anything' question available at the cost "
               "of saying so out loud.\n"
               "Deliberately declares NO `COVERS:` commitment. It guards the "
               "harness, not a capability."),

    Spec("T0.31", 0, "The backlog reader cannot be quieted by tidying the backlog",
         hypothesis="`experiments/review_queue.py` — the reader "
                    "`docs/REVIEW_QUEUE.md` went six days without — detects "
                    "every way a routed finding can go quiet, on the real code "
                    "path, and its TOTAL cannot be lowered by any of them: "
                    "(1) a live row past a DUE: it declared is OVERDUE, and a "
                    "future DUE: is silent; (2) an OPEN row past one whole "
                    "consumer cycle is STALE, silent AT the bar, and neither "
                    "age nor a re-arm touches a dispositioned row; (3) the "
                    "three tidy-ups a maintainer would reach for — DELETE the "
                    "row, relabel it HELD, drop the DUE: that went red — each "
                    "convert one class into another and none lowers the total; "
                    "(4) a hold must name a live blocker, so a resolved or "
                    "phantom one is itself a violation; (5) malformation is "
                    "recorded rather than raised, so one bad line cannot hide "
                    "the rest; (6) prose opens no row and sets no clock; "
                    "(7) an absent git baseline accuses nobody; (8) a "
                    "disposition is not an execution — DISPOSITIONED (design "
                    "written, execution owed) is LIVE and ages like OPEN, "
                    "ACTED must name its executing commit, a commitless ACTED "
                    "is its own violation that no relabelling lowers, and the "
                    "honest repair (ACTED with the commit) clears the row "
                    "without tripping anything; (9) a row written as prose "
                    "under a `## ` heading is COUNTED (UNDECLARED-ROW), never "
                    "parsed — the six pre-declaration shapes the 60th audit "
                    "found invisible fire exactly six, an attached ROUTED: "
                    "declaration clears one, and the file's legitimate prose "
                    "headings stay exempt.",
         falsified_by="Any property failing; the battery running fewer than "
                      "N_PROPERTIES; or the live `docs/REVIEW_QUEUE.md` "
                      "failing to parse into rows that are all in contract. "
                      "Concretely: deleting a rotting row lowering the "
                      "violation count, relabelling a stale row `HELD` "
                      "clearing it, an indented `ROUTED:` inside a sentence "
                      "opening a row, a prose date being read as a clock, a "
                      "class in `VIOLATIONS` that no document can trigger, "
                      "a finding the report does not print, a DISPOSITIONED "
                      "row that goes quiet instead of ageing, a commitless "
                      "ACTED stamp clearing a red row, or a routed section "
                      "written under a `## ` heading being invisible to the "
                      "count.",
         null_baseline="THE READER THAT ACTUALLY EXISTED — `grep '^ROUTED:' "
                       "docs/REVIEW_QUEUE.md | wc -l`, published in the file's "
                       "own contract line and, until 2026-08-31, the whole of "
                       "the tooling. Not a paraphrase and not a crippled copy: "
                       "it is run as the control, it counts rows, and on the "
                       "one sabotage it can see it reports the WRONG SIGN — "
                       "delete the row and the backlog looks smaller.",
         metric="properties_failed", budget=Budget.CPU, seeds=1,
         depends_on=[],
         control="See null_baseline. Named properties "
                 "p2_overdue_fires_on_a_passed_date_only, "
                 "p5_deleting_the_row_does_not_help, "
                 "p11_every_class_is_reachable_and_reported and "
                 "p12_a_disposition_is_not_an_execution must be among the "
                 "control's failures, or the row count is somehow sufficient "
                 "and this module guards a distinction that did not need "
                 "making.",
         kills="The last unread instrument in the governance set. "
               "`REVIEW_QUEUE.md` was built to make a backlog countable and "
               "then had no counter, which is the 27th audit's own finding "
               "recurring one layer up inside the organ it created.",
         notes="SCAR (52nd audit, 2026-08-31, B4). The Review's Sunday FULL "
               "run — the only mode that does Part 2, and the run that owed "
               "`w0-too-shallow`'s world design — started 2026-08-30T06:37, "
               "died on `Reached max turns (60)` at 06:48 eleven minutes into "
               "a forty-minute budget, and wrote nothing. That row's status "
               "line had said *'design owed by the Review 2026-08-30'* since "
               "08-25; the date passed, two holds and four gate-provisional "
               "specs sat behind it, both GPU cost classes read EMPTY because "
               "of it, and NO NUMBER ANYWHERE WENT RED. The 27th audit had "
               "already written the corollary and never built it: *'an organ "
               "that is the destination of routed work must have liveness "
               "watched by something other than itself.'* "
               "`scripts/lib_liveness.sh:review_liveness` (52nd audit B1, "
               "built 2026-08-31) watches whether the CONSUMER ran; this "
               "watches whether the WORK MOVED. A desk can open every morning "
               "and dispose of nothing, so neither implies the other.\n"
               "THE RATCHET COUNTS EVERY CLASS, BEFORE THE FACT. This is the "
               "fourth instrument here checked for the one-class-ratchet "
               "disease and the first built already knowing it: "
               "`coverage.py` (closed by T0.21 P2), `decisions.py`'s "
               "`NO-DEFAULT` (T0.28 P9) and `champions.py`'s `ARENA-MISSING` "
               "(T0.29 P2) each shipped counting one class and each paid a "
               "repair that LOWERED its own number. P4/P5/P6 assert on the "
               "TOTAL, never on the new class.\n"
               "DECLARED, NEVER INFERRED. `DUE:` and `BLOCKED-BY:` are "
               "start-of-line declarations in the `DECIDE:`/`COVERS:` idiom; "
               "prose dates are not read. `w0-too-shallow`'s prose date was "
               "migrated into a `DUE:` line BY HAND in the registering commit "
               "— a migration is a human act, an inference is a bug "
               "(`901f7fc`: a seat's arena turned out to be the words OUT "
               "LOUD).\n"
               "THE TWO-MEANING TOKEN (Review 09-01, item 4; strengthened in "
               "the same commit as the repair). On `recipe-sensitivity`, "
               "`ACTED 2026-08-25` meant *the Review wrote a design*; on "
               "`me11-…` it meant *the builder executed one, commit named*. "
               "ACTED is terminal, so the first sense closed a row whose work "
               "had not started and `UB.10` stayed parked for seven days with "
               "the reader printing no violation. The repair is P12: "
               "`DISPOSITIONED` as a LIVE status that ages, and an ACTED "
               "contract requiring the executing commit.\n"
               "WHAT THIS DOES NOT CERTIFY: whether a disposition was any "
               "good — an `ACTED` naming a commit is taken at its word; "
               "whether that commit did the work is the overseer's to read. And `MAX_OPEN_AGE_DAYS = 8` is DERIVED "
               "from the consumer's schedule (one DAILY cycle plus the weekly "
               "Sunday FULL, plus a day of grace); if the Review's cadence "
               "changes that constant is stale and no property here can tell.\n"
               "Deliberately declares NO `COVERS:` commitment. It guards the "
               "decision machinery, not a capability."),

    # ── PL: the arena the PLASTIC-ONLY decree named and nobody built ──────
    #
    # REGISTERED 2026-08-30 from docs/research/FROZEN_VS_PLASTIC.md §7.3, under
    # the INTEGRATION_QUEUE protocol. Seventh audit asking (OVERSIGHT B5); the
    # `plasticity` commitment read `2 specs / 0 pass / 0 RUNNABLE` — both its
    # claim specs blocked behind `T2.01` — while `CHAMPIONS.md:166-190` asserted
    # "`PL.02` decides it and is runnable today" about a spec that had never
    # existed. `GOAL.md:76`'s decree was therefore held with no registered
    # falsifier, which `SYSTEM.md`'s standing rule forbids by name.
    #
    # CROSS-CHECK (protocol step 1), over docs/research/*.md, docs/LESSONS.md
    # and docs/DECISIONS*.md for `frozen|plastic|reshaping gain|throughput
    # floor`. ONE REFUTATION FOUND, and it is this file's own header: the
    # PLASTIC-ONLY decree of 2026-08-09 collapsed §7.3's four-arm
    # frozen-vs-plastic contest to a single admissible arm. So PL.00 and PL.02
    # are NOT registered verbatim — they are corrected per the refuting
    # analysis, exactly as `CHAMPIONS.md`'s own "WHAT STILL RUNS" paragraph
    # already wrote the correction: PL.00 becomes a FEASIBILITY CHECK on the
    # pure encoder, PL.02 measures what the plastic path BUYS. NO THRESHOLD IS
    # TOUCHED by that correction (the 5.0 sim-s/real-s floor is LC.02's and
    # LEARNING_CORE §5.0b's, unchanged; PL.02's bootstrap-CI-excluding-zero bar
    # is unchanged). What changes is which arm carries the claim and what a
    # result MEANS.
    #
    # SECOND CROSS-CHECK RESULT, recorded because it fires tomorrow: `D1`'s
    # option (ii) would narrow the decree to SENSORY towers only
    # (DECISIONS_NEEDED.md:791). Both specs below are about sensory encoders,
    # so neither is affected by either fork of D1 — registering them today is
    # safe against the 2026-08-31 firing.
    #
    # DELIBERATELY NOT REGISTERED: PL.01, PL.03, PL.04, PL.05. They are the
    # arms and the arbitration of the contest the decree ENDED, and PL.01's own
    # notes require amending `LEARNING_CORE.md`'s U2 criterion first (its
    # arithmetic excludes every frozen tower by wording rather than evidence —
    # the overturn recorded in `ladder_prompt.md`). Registering an arbitration
    # whose arms are unconstitutional would be specifying work nobody may do.
    # The ids stay free; §7.3 keeps their text.

    Spec("PL.00", 2, "What each perception encoder costs on THIS box",
         hypothesis="Every candidate perception encoder is measured on one ARM "
                    "core for ms/frame at its native resolution, and THE PURE "
                    "FROM-SCRATCH ENCODER — the only arm the PLASTIC-ONLY "
                    "decree admits — lets the full loop clear the 5.0 "
                    "simulated-seconds-per-real-second throughput floor with "
                    "vision live at 5 Hz.",
         falsified_by="The pure encoder does not clear the floor with vision "
                      "live. Then vision at 5 Hz is unaffordable on this box "
                      "under the one architecture the decree permits, and the "
                      "decree's OWN pre-registered RE-OPEN TRIGGER fires "
                      "(`CHAMPIONS.md`: *'if a from-scratch encoder cannot hit "
                      "the PL.00 throughput floor on this hardware ... the "
                      "decision returns to the owner with that number "
                      "attached'*). This spec is the only registered thing in "
                      "the repo that can pull that trigger.",
         null_baseline="The measured render cost itself, measured in this same "
                       "process rather than cited: an eye frame costs what it "
                       "costs before any encoder sees it. An encoder cheaper "
                       "than its own render is free; one costing 10x the "
                       "render is the dominant cost of having eyes. "
                       "(DIRECTION_AUDIT.md read 68 ms/frame at 128x128 and "
                       "185 ms at 320x320 under xvfb+llvmpipe — a figure to "
                       "re-measure, never to quote.)",
         metric="ms_per_frame_x_sim_seconds_per_real_second",
         budget=Budget.CPU, seeds=3, depends_on=["T0.07", "PG.6"],
         control="TWO, and the second is the one that can kill the leg. "
                 "(1) IDENTITY: a no-op encoder that reads every pixel and "
                 "does nothing else must sit at ~0 ms/frame and must NOT "
                 "change the measured loop throughput. If swapping the encoder "
                 "for nothing at all moves the throughput, the harness is "
                 "timing something other than the encoder and every cell is "
                 "uninterpretable. (2) HEAVY REFERENCE, the discrimination "
                 "check: a 21.6M-parameter ViT-S/14 at 224 must FAIL the 5.0 "
                 "floor on this box. A floor a frozen ViT clears cannot "
                 "exclude anything, and 'the pure encoder cleared it' would "
                 "then be a sentence about the bar rather than about the "
                 "encoder. It returns VOID, not FAIL — a threshold that cannot "
                 "reject is an invalid instrument, not a refuted hypothesis.",
         kills="Any perception encoder that cannot clear the throughput floor "
               "— INADMISSIBLE, not scored (LEARNING_CORE.md ADMISSION-2). "
               "This spec can eliminate an encoder in MINUTES, before any "
               "accuracy question is asked. Under the decree its edge is "
               "turned inward: the arm it can kill is the seat holder's own.",
         notes="COVERS: plasticity (rule), sight (rule).\n"
               "WHAT IT CAN AND CANNOT MEASURE, said before the run. Cost is a "
               "function of ARCHITECTURE and input size, not of the values in "
               "the tensors: a randomly-initialised ViT-S/14 at 224 costs "
               "exactly what a pretrained one costs. So the frozen-tower "
               "reference is measured from a locally-constructed architecture "
               "and needs no download — and this spec makes NO accuracy claim "
               "about it, which is `T2.03`'s job and is already recorded. The "
               "frozen reference is SCORED-AND-INELIGIBLE in `SYSTEM.md`'s "
               "sense: its number goes in the ledger, it cannot take a seat.\n"
               "Accounting unit inherited from `w0.py`/`LC.02` so the floor "
               "means the same thing here: one decision is 40 substeps of "
               "0.005 s = 0.2 simulated seconds, so 'vision live at 5 Hz' is "
               "exactly one rendered frame per decision. Report resident RAM "
               "too — `SYSTEM.md` caps this box at ~1.5 GB.\n"
               "Registered from docs/research/FROZEN_VS_PLASTIC.md §7.3, "
               "corrected per the PLASTIC-ONLY decree; see the block above."),

    Spec("PL.02", 4, "The RESHAPING test: does another sense change what an encoder computes?",
         hypothesis="For each modality pair (A,B), an encoder for A trained "
                    "JOINTLY with B by cross-modal masked prediction "
                    "outperforms an A-only encoder of matched capacity WHEN "
                    "BOTH ARE EVALUATED ON A ALONE at test time. The reshaping "
                    "gain R = perf(M_AB | A only) - perf(U_A) is positive, "
                    "paired by seed, bootstrap CI excluding zero.",
         falsified_by="R indistinguishable from zero for the PLASTIC arm. Then "
                      "binding does not reshape encoders at our scale, the "
                      "arithmetic that the PLASTIC-ONLY decree rests on buys "
                      "nothing measurable here, and that returns to the owner "
                      "as evidence — LOUDLY, in the Review. The decree's ENDS "
                      "are not on trial (SYSTEM.md class 1); its stated "
                      "MECHANISM is (class 2).",
         null_baseline="The FULLY FROZEN arm, whose R = 0 EXACTLY, by "
                       "construction: M_AB's A-encoder IS U_A's A-encoder, the "
                       "same frozen tensor. The analytic null — no cheaper or "
                       "more honest null exists in this project. It is scored "
                       "and ineligible, never excluded (SYSTEM.md, 2026-08-24).",
         metric="reshaping_gain_R", budget=Budget.CPU_LONG, seeds=3,
         depends_on=["PG.1", "PL.00"],
         control="SHUFFLED-PARTNER: train M_AB with B's stream drawn from a "
                 "DIFFERENT episode — correspondence destroyed, marginals and "
                 "temporal statistics preserved. R must collapse to ~0. If "
                 "shuffled-B reshapes A just as well, the gain is capacity or "
                 "regularisation, not binding, and the whole test is VOID.",
         kills="The claim that cross-modal binding reshapes what an encoder "
               "computes at this project's scale. A positive R is the "
               "measured value of the plastic path; a null R does not restore "
               "freezing (the owner decreed the ENDS), but it removes the "
               "arithmetic argument that has been carried as if it were a "
               "measurement since 2026-08-09.",
         notes="COVERS: plasticity (claim), one brain / unison (claim).\n"
               "This is the M3L signature made into a metric (arXiv:2311.00924: "
               "representations learned with touch 'also benefit vision-only "
               "policies at test time'). It has a downstream meaning, which is "
               "what stops it being a metric about metrics: perf(M_AB | A only) "
               "IS how well Jack copes when a sense fails — night removes "
               "vision, rain masks audio, and both happen in the survival "
               "world. CALIBRATE FOR A SMALL EFFECT: Kepler-Encoder's closest "
               "analogue was R^2 0.049/-0.001/0.187 across three robots with "
               "one NEGATIVE, p<=0.012. Paired seeds, IQM, bootstrap CI on the "
               "paired difference (arXiv:2108.13264), or the experiment cannot "
               "see what it is looking for.\n"
               "WHAT THE DECREE CHANGED, and it is only the meaning: this was "
               "written to DECIDE frozen-vs-plastic. The owner decided that on "
               "2026-08-09 by decree, so PL.02 now measures what the plastic "
               "path BUYS — and remains the decree's sole registered "
               "falsifier, which is why it exists at all. Threshold, control "
               "and null are unchanged from FROZEN_VS_PLASTIC.md §7.3."),

    # ── THE LADDER TEST (docs/research/CURIOSITY_BAKEOFF.md) ────────────
    # Two-digit ids on purpose. `run.py::_module_for` globs `lt_1_*.py`, which
    # would also match `lt_10_*.py`, and its hierarchical-id escape hatch does
    # NOT cover that case (it tests startswith("lt_1_"), and "lt_10" fails it).
    # The same latent collision exists today between UB.1 and UB.16. LT.01-LT.99
    # is structurally immune. See LESSONS.md, "A spec id that is a prefix of
    # another spec id disables one of them".

    Spec("LT.01", 2, "The Ladder Test is measurable: null floor and un-gameable rise",
         hypothesis="A free-roaming random climber-rover produces ZERO engaged "
                    "ladder attempts, while reaching >=0.6 m of torso RISE by "
                    "non-ladder routes; and from the ladder base a genuine "
                    "weight-bearing hang occurs in 1-5% of 3 s random bursts — "
                    "so ladder-supported rise (contact AND airborne AND held "
                    ">=0.5 s AND load-bearing) discriminates, raw torso z does "
                    "not, and the first success is reachable by chance.",
         falsified_by="A free-roaming random agent produces engaged attempts "
                      "(the null floor is not zero), or a non-ladder route "
                      "reaches the platform, or P(hang from the base) is 0 in "
                      "800 bursts (no bootstrap exists and no learning-progress "
                      "method can work without an archive).",
         null_baseline="n/a — this spec IS the null floor measurement.",
         metric="null_engaged_attempts", budget=Budget.CPU_LONG,
         depends_on=["PG.1", "PG.3", "PG.4"], seeds=3,
         control="A greedy height-maximising oracle with adhesion DISABLED must "
                 "still be unable to reach the platform — else an alternate "
                 "route exists and SUCCESS is not evidence of climbing.",
         kills="The entire Ladder Test, before a single arm is trained. Costs "
               "20 CPU-minutes; every threshold in the programme is set from it.",
         notes="Pilot 2026-08-09 (aarch64, mujoco 3.2.3). Free-roaming: 0 "
               "engaged attempts in 9,000 random decisions; max NON-ladder "
               "torso z 1.007 m against z_rest 0.360 m. From the base, 800 x "
               "3 s bursts: P(hang) = 0.55 under an ABSOLUTE-z definition "
               "(broken - z_rest already clears the bar), 0.063 instantaneous, "
               "0.026 persistent, 0.021 +- 0.009 persistent AND load-bearing; "
               "random rise ceiling 0.83 m. Those four numbers ARE the "
               "definition of h(t) and every threshold in LT.03."),

    Spec("LT.02", 2, "The self-generated-chaos detector works (PG.4's blind spot)",
         hypothesis="A curiosity agent can farm irreducible surprise from its "
                    "OWN body with zero noise-panel dwell, and the chaos "
                    "detector sees it: ragdoll-ICM (panel deleted, adhesion 0) "
                    "scores chaos_occupancy >= 3.0 and chaos_reward_ratio >= "
                    "2.0 while PG.4's dwell metric reads 0.000, and the "
                    "scripted climber — which moves hard and falls repeatedly — "
                    "scores chaos_occupancy <= 1.0.",
         falsified_by="Ragdoll-ICM is NOT flagged (the detector is blind and no "
                      "arm's immunity may be reported), or the scripted climber "
                      "IS flagged (the detector penalises coordinated motion and "
                      "falling, i.e. the behaviour GOAL.md asks for).",
         null_baseline="The random policy, which DEFINES the ruler: theta is its "
                       "90th-percentile irreducible error, so it reads "
                       "chaos_occupancy = 1.0 by construction.",
         metric="chaos_detector_separation", budget=Budget.CPU_LONG,
         depends_on=["LT.01", "PG.4"], seeds=3,
         control="Cross-check: the ICM agent WITH the panel present must be "
                 "flagged by BOTH detectors (panel_dwell > 0.4 AND "
                 "chaos_occupancy >= 3.0). Two independent detectors must agree "
                 "on a known positive, or one of them is reading noise.",
         kills="Every 'his curiosity is not trapped' claim that rests on panel "
               "dwell alone — which is all of them, including CU.3 as currently "
               "written.",
         notes="separation = chaos_occupancy(ragdoll-icm) - "
               "chaos_occupancy(scripted-climber), with panel_dwell(ragdoll-icm) "
               "asserted == 0.0. That number pair IS the gap: a total curiosity "
               "failure that PG.4 scores as perfectly clean. Detector = "
               "pooled-fit forward model, out-of-fold, high error AND no "
               "reducibility when the training data doubles (LPM criterion, "
               "arXiv:2509.25438, used as a diagnostic not a reward). "
               "thrash_ratio is reported as the model-free second signal."),

    Spec("LT.03", 5, "THE LADDER TEST: curiosity alone climbs the ladder",
         hypothesis="With the environment returning reward identically zero, at "
                    "least one candidate arm produces >=20 engaged ladder "
                    "attempts, a distance-matched post-fall return lift >= 2.0, "
                    "an ascent gain >= 0.35 m with Spearman rho >= 0.35 (p<0.01) "
                    "and a final-quintile mean rise >= 0.85 m (above the "
                    "measured random ceiling of 0.83 m), and at least one "
                    "topping-out, in >=2 of 3 seeds — while dwelling <= 0.15 at "
                    "the noise panel in every seed and never tripping the "
                    "self-generated-chaos check.",
         falsified_by="No arm produces a single engaged attempt (exploration "
                      "never reaches the ladder), or attempts occur with no "
                      "ascent trend (credit assignment, not curiosity, is the "
                      "bottleneck), or every arm that climbs also fixates on the "
                      "panel or farms its own body noise.",
         null_baseline="Random and random-repeat action: measured at 0 engaged "
                       "attempts in 9,000 decisions (LT.01). Plus randrew, a "
                       "random-stationary-reward learner at matched compute, "
                       "which controls for 'any optimisation pressure explores'.",
         metric="unforced_ascent_gain", budget=Budget.CPU_LONG,
         depends_on=["LT.01", "LT.02", "PG.4"], seeds=3,
         control="Three, each must land on its declared side: (1) the ICM "
                 "control MUST fixate on the panel in THIS rig (dwell > 0.4) — "
                 "proving the trap is live here and not only in PG.4's rover; "
                 "(2) randrew must not match the winner's visitation lift; "
                 "(3) a goal-shuffled variant of the winning arm must show no "
                 "ascent trend.",
         kills="The 'intrinsic motivation is enough' thesis for structured "
               "vertical behaviour. If it fails, GOAL.md's ladder image needs a "
               "goal/skill layer (PEG 2303.13002, or a Go-Explore archive over "
               "h(t)-bearing states — PG.3 already certified the state restore "
               "it needs at resume_max_dev 0.0), and that pivot is decided by "
               "this result, not by preference.",
         notes="SCREENING ONLY — each candidate against the null, no winner "
               "declared; arbitration is LT.04, because run_bakeoff VOIDs on a "
               "sub-gate arm and icm/rnd are REQUIRED to fail. An arm whose "
               "chaos_occupancy >= 3.0 AND chaos_reward_ratio >= 2.0 returns "
               "Status.VOID for that arm: its curiosity signal degenerated, so "
               "the run did not test the claim. Every arm's reward code passes "
               "a static audit for ladder-referencing symbols; a match is ERROR. "
               "No published system has done this — LadderMan (2606.05873) "
               "climbs from a human reference motion, METRA on a 69-DoF humanoid "
               "flails (RGSD 2510.06203), and the one time LP curricula were "
               "pointed at a 2D climbing morphology they reached ~1% mastery "
               "(TeachMyAgent 2103.09815)."),

    Spec("LT.04", 5, "Bakeoff: which curiosity mechanism climbs best",
         hypothesis="Among the arms that cleared LT.03, one beats the runner-up "
                    "by >=1.5 sigma of the pooled seed spread on "
                    "unforced_ascent_gain.",
         falsified_by="n/a for a bakeoff — the outcomes are WINNER, TIE (take "
                      "the cheaper arm) or VOID (an arm is below the 3-sigma "
                      "learning gate, so the decision is blocked, not made).",
         null_baseline="Random-repeat action, shared across arms.",
         metric="unforced_ascent_gain", budget=Budget.CPU_LONG,
         depends_on=["LT.03"], seeds=3,
         control="Inherited from LT.03; no arm may enter this bakeoff whose "
                 "LT.03 result was VOID for self-generated chaos.",
         notes="run_bakeoff(arms=[disagree, lp, metra], null_run=random_repeat, "
               "learning_gate_sigma=3.0, margin_sigma=1.5). Arm.cost is declared "
               "in CPU-CORE-SECONDS OF LEARNER TIME PER 1,000 DECISIONS, measured "
               "in-run with time.process_time() around the intrinsic-reward and "
               "policy-update calls and EXCLUDING MuJoCo (identical across arms, "
               "so including it would compress the differences the tie-break "
               "needs). Pre-run estimates: lp 2.0, disagree 9.0, metra 14.0 — a "
               "TIE therefore resolves to lp, which is why the measurement must "
               "replace the estimate before this runs. Fewer than two arms "
               "clearing LT.03 records VOID: 'fewer than two learners'."),

    Spec("LT.05", 5, "The climb survives the curiosity that produced it",
         hypothesis="With the intrinsic module removed and reward identically "
                    "zero, the winning arm's deterministic policy still reaches "
                    ">= 0.8x its best training ladder-supported rise and tops "
                    "out at least once in 10 episodes.",
         falsified_by="Ladder-supported rise collapses without the bonus — then "
                      "the behaviour was bonus-chasing, not a skill.",
         null_baseline="The same policy at initialisation, bonus off.",
         metric="retention_ratio", budget=Budget.CPU_LONG,
         depends_on=["LT.04"], seeds=3,
         control="A policy trained with the random-stationary reward (randrew) "
                 "must show no retained climbing — else retention measures "
                 "architecture, not learning.",
         notes="Spontaneous attempt FREQUENCY with the bonus off is reported but "
               "explicitly NOT gated: a learning-progress agent is supposed to "
               "lose interest once the ladder is mastered, exactly as a child "
               "does. Gating on frequency would systematically penalise the "
               "mechanism most likely to be right."),

    Spec("LT.06", 5, "It is the ladder he is curious about, not the coordinates",
         hypothesis="The identical unmodified arm, in a world where the ladder "
                    "is moved, re-yawed and re-spaced, scores >= 0.5x its "
                    "home-world ascent gain.",
         falsified_by="Performance collapses when the ladder moves — the arm "
                      "learned a location, or the reward was hard-coded.",
         null_baseline="Home-world score for the same arm and seed.",
         metric="moved_ladder_ratio", budget=Budget.CPU_LONG,
         depends_on=["LT.04"], seeds=3,
         control="A deliberately hard-coded climb reward, written for this "
                 "control only and keyed to the home ladder's xy, MUST fail "
                 "here — that is what makes the spec an instruction detector "
                 "rather than a generalisation test.",
         notes="Together with LT.03's static symbol audit this is the "
               "anti-instruction provision. Eureka-style LLM reward writing "
               "(2310.12931) is caught by exactly this pair."),

    Spec("LT.07", 5, "The winner survives fresh seeds",
         hypothesis="Re-run at 3 seeds never used during screening or "
                     "arbitration, the winning arm clears every one of LT.03's "
                     "seven observables at the same pre-registered thresholds.",
         falsified_by="Any observable falls below its LT.03 threshold on fresh "
                      "seeds — the win was selection over arms and seeds.",
         null_baseline="LT.01's null floor, re-measured on the fresh seeds.",
         metric="unforced_ascent_gain", budget=Budget.CPU_LONG,
         depends_on=["LT.04"], seeds=3,
         control="The same three controls as LT.03, re-run: the ICM control "
                 "must still fixate on these worlds.",
         kills="Nothing is written to the README before this passes.",
         notes="Seeds 10/11/12. Costs ~40 CPU-minutes and removes the "
               "multiple-comparison argument entirely (G8)."),

    Spec("LT.08", 5, "The humanoid climbs — same test, real body",
         hypothesis="With locomotion in hand, the winning arm reproduces LT.03's "
                    "seven observables on the full humanoid in the same playground.",
         falsified_by="Any of LT.03's six clauses fails on the humanoid at the "
                      "budgeted step count with the curve flat.",
         null_baseline="LT.01's nulls, re-measured on the humanoid body.",
         metric="unforced_ascent_gain", budget=Budget.GPU_LONG,
         depends_on=["LT.07", "T2.01", "T2.02"], seeds=3,
         control="Same as LT.03, re-run on this body: the ICM control must "
                 "fixate, and the chaos check matters MORE here — RGSD "
                 "(2510.06203) reports exactly this failure at 69 DoF.",
         kills="Nothing on its own — a FAIL here with LT.03 passing scopes the "
               "claim honestly to the reduced body and points at throughput.",
         notes="BLOCKED until T2.01/T2.02 pass. Also blocked on throughput: at "
               "T2.01's measured ~128 env-steps/s a 20M-step arm-seed costs "
               "43 h, so 3 seeds exceed a whole week of Kaggle quota for ONE "
               "arm. Getting the 45.5M trunk out of the inner loop makes it "
               "MuJoCo-bound at ~2,000 steps/s (~2.8 h/arm-seed). The "
               "prerequisite is a throughput spec, not more quota."),

    Spec("LT.09", 5, "The VLM proposes ladder-shaped goals; learning progress disposes",
         hypothesis="Frozen-VLM-proposed goals, expressed ONLY as predicates in "
                    "the existing outcome space and filtered by LP, reach the "
                    "first engaged ladder attempt in fewer decisions than "
                    "LP-only at matched goal count.",
         falsified_by="No speedup, or VLM goals flood the buffer while their "
                      "achievement stays ~0 (a hallucinated curriculum).",
         null_baseline="LP-only (the lp arm) at matched goal count.",
         metric="time_to_first_engaged_attempt", budget=Budget.CPU_LONG,
         depends_on=["LT.04"], seeds=3,
         control="A scrambled-caption VLM fed a DIFFERENT scene must not beat "
                 "LP-only — else the benefit was 'more goals', not grounded "
                 "interestingness. Additionally the VLM may never emit reward "
                 "code: a proposal that is not a predicate over existing outcome "
                 "dimensions is rejected before it reaches the buffer.",
         notes="Only run if lp wins LT.04. LLM-proposed goals have never driven "
               "low-level continuous control (ELLM 2302.06692 limitations; "
               "OMNI-EPIC 2405.15568 uses 6 discrete actions), so this is the "
               "genuinely unoccupied combination — and the reason it is "
               "unoccupied is a grounding gap, which the predicate restriction "
               "is designed to sidestep."),

    # ── GEN: the generality barriers GOAL.md cites (docs/GENERALITY.md) ──
    # Registered 2026-09-01 (Review 08-31 item 6; registration debt seeded by
    # the 29th audit on 2026-08-25 and carried through 273 commits). GOAL.md's
    # expansion path cites these four ids verbatim — "OTHER MINDS ...
    # (GEN.02, GEN.03, GEN.09)" and "MORE WORLDS ... (GENERALITY.md GEN.06)"
    # — so for 23 days the constitution promised falsifiers nobody had
    # written. GENERALITY.md's honesty clause carries over verbatim: nothing
    # here is scheduled, several may be years out, and naming them is not
    # promising them. What registration buys is legibility, the exact thing
    # the 2026-08-31 lesson says an unregistered spec can never have: each id
    # now sits in a cost class, blocks and is blocked, resolves in BY_ID, and
    # the dangling-citation debt in `coverage.goal_citations` goes to zero.
    # Every depends_on below is a REAL prerequisite chosen so `run next`
    # never reads these as runnable before their substrate exists — the
    # VO.02 scar (a blocker stated in notes that no instrument could see) is
    # the defect this ordering exists to avoid.

    Spec("GEN.02", 6, "He learns by watching — a second Jack is a teacher",
         hypothesis="Two embodied Jacks, one world. An observer that WATCHES "
                    "a skilled demonstrator perform a skill it does not have "
                    "acquires that skill in measurably fewer of its own "
                    "practice decisions than the matched-experience solo "
                    "null, by >=3 sigma across seeds.",
         falsified_by="Watching a skilled Jack buys no acquisition advantage "
                      "over solo practice — or the advantage SURVIVES the "
                      "random-demonstrator control, in which case the gain "
                      "was co-presence or extra visual traffic, not social "
                      "learning, and the claim is dead either way.",
         null_baseline="SOLO: the identical observer, demonstrator absent, "
                       "matched practice decisions and matched wall-clock in "
                       "the same world.",
         metric="observational_acquisition_advantage",
         budget=Budget.CPU_LONG,
         depends_on=["VO.02", "LC.07"], seeds=3,
         control="RANDOM DEMONSTRATOR (GENERALITY.md, verbatim): the same "
                 "rig with the skilled Jack replaced by a random-action "
                 "agent must NOT help. If it does, the observer is not "
                 "extracting skill from what it watches — it is reacting to "
                 "the presence of motion, and the test measures arousal.",
         kills="GOAL.md's second expansion (OTHER MINDS) as currently "
               "designed: if watching cannot teach, adding Jacks adds "
               "pressure but no channel, and VO.02's invented signal stays "
               "the ceiling of the social programme until the observer is "
               "redesigned.",
         notes="Measured 2026-08-09: of 136 specs exactly ONE touched other "
               "minds, and it was mocap imitation. GENERALITY.md rates this "
               "the cheapest high-value barrier on the list — a second "
               "PROCESS, not a second GPU (INTEGRATION_QUEUE.md, THE "
               "GENERALITY MAP). depends_on is structural, not thematic: "
               "VO.02 (PASS 2026-08-30) built the two-independent-learners-"
               "one-world substrate this rig extends to embodiment, and "
               "LC.07 is the adoption gate for a learning core that can "
               "acquire a skill at all — an observer that cannot learn solo "
               "cannot show a watching advantage. Stage it cheaply like "
               "VO.02 staged signalling: the floor is one demonstrator "
               "trajectory replayed into the observer's eye, before any "
               "live second process. "
               "  COVERS: social/other agents (claim)"),

    Spec("GEN.03", 6, "False belief: he models what another saw, not what is true",
         hypothesis="Jack watches agent B see food hidden at location A; the "
                    "food is moved to B' while B is ABSENT. Jack's "
                    "anticipatory prediction of B's search (orientation/"
                    "approach toward where B will look, read before B moves) "
                    "selects the false-belief location A above chance at >=3 "
                    "sigma across seeds.",
         falsified_by="His prediction tracks the food's TRUE location "
                      "regardless of what B witnessed — he models the world, "
                      "not the mind in it. GENERALITY.md's own framing "
                      "binds: passing this is a landmark; failing it is "
                      "NORMAL, and a FAIL here is a result, not a defect.",
         null_baseline="A TRUE-LOCATION predictor (always answers where the "
                       "food actually is). It must score chance on "
                       "false-belief trials by construction; Jack must beat "
                       "it there specifically.",
         metric="false_belief_prediction_flip", budget=Budget.CPU_LONG,
         depends_on=["GEN.02"], seeds=3,
         control="PRESENT-FOR-THE-MOVE (GENERALITY.md, verbatim): when B "
                 "witnessed the relocation, Jack's prediction must FLIP to "
                 "the true location. An agent that has merely learned "
                 "'predict A' passes the experiment and fails this flip, so "
                 "belief-tracking and location-habit are separable.",
         kills="Nothing structural — but every future teaching, deception "
               "or cooperation design must stop assuming a mind-model and "
               "carry its own workaround until this passes.",
         notes="Theory of mind is the prerequisite for teaching, deception, "
               "coalition and most of language's real work (GENERALITY.md "
               "GEN.03). Depends on GEN.02 because an embodied false-belief "
               "rig IS a two-agent rig plus occlusion — and because an "
               "observer that cannot even extract skill from watching has "
               "no measured basis for extracting knowledge states. "
               "  COVERS: social/other agents (claim)"),

    Spec("GEN.06", 5, "Transfer across worlds: mastery is structure, not fit",
         hypothesis="Trained in world A and dropped into world B — different "
                    "layout, different resource placement, SAME underlying "
                    "rules — his prior experience beats a fresh agent of "
                    "identical architecture on time-to-competence in B by "
                    ">=3 sigma across seeds.",
         falsified_by="No advantage over the fresh agent: the jungle made a "
                      "forager fitted to one jungle, and generality-as-"
                      "transfer is refuted for this design. 'A Jack who "
                      "masters the jungle and is helpless in a desert is "
                      "not general — he is fitted.'",
         null_baseline="FRESH AGENT: identical architecture, zero world-A "
                       "experience, identical world-B budget.",
         metric="transfer_advantage_over_fresh", budget=Budget.CPU_DAYS,
         depends_on=["LC.07", "W0.DIAG"], seeds=3,
         control="SHUFFLED-RULES WORLD B (GENERALITY.md, verbatim): a world "
                 "B whose underlying rules are scrambled must show NO "
                 "transfer advantage. If it does, the gain was general "
                 "fitness — better motor tone, better exploration habits — "
                 "not learned structure, and the claim collapses to 'more "
                 "practice helps'.",
         kills="The premise of GOAL.md's first expansion (MORE WORLDS): if "
               "learned structure does not move between matched-rule "
               "worlds, 'abstraction IS generality' has no mechanism and "
               "world-building buys breadth, not depth.",
         notes="W0.DIAG is in depends_on for a measured reason, not a "
               "thematic one: its validated difficulty instrument (PASS "
               "2026-08-31, known-answer control BINDING) is what certifies "
               "world A and world B comparable BEFORE the transfer claim "
               "runs — without it, 'transfer' confounds with world B simply "
               "being easier, the exact inversion its wk5-N3 control "
               "exists to catch. LC.07 because a transfer claim needs an "
               "adopted core that demonstrably learns world A at all. "
               "  COVERS: generality (claim)"),

    Spec("GEN.09", 6, "Culture: generation 3 knows what generation 1 never knew",
         hypothesis="Across three generations coupled ONLY by diaries and "
                    "in-world teaching — never weights — generation 3 "
                    "demonstrates a competence generation 1 never had, and "
                    "reaches generation-2 competence in fewer decisions than "
                    "generation 2 originally spent, by >=3 sigma across "
                    "seeds. Accumulation, not just persistence.",
         falsified_by="Generation 3 is statistically indistinguishable from "
                      "generation 1: the diary crosses deaths but knowledge "
                      "does not ACCUMULATE across individuals, and one "
                      "diary is a memory, not a culture.",
         null_baseline="TRANSMISSION SEVERED (GENERALITY.md, verbatim): "
                       "generation 3 with no inherited diary and no "
                       "teaching must fall back to generation 1's level. "
                       "If it does not, the 'culture' was in the world or "
                       "the weights all along.",
         metric="generational_knowledge_gain", budget=Budget.CPU_DAYS,
         depends_on=["ME.9", "ME.10", "GEN.02"], seeds=3,
         control="CORRUPTED INHERITANCE: generation 3 receives a diary of "
                 "matched size whose entries are shuffled across episodes "
                 "and attributions. It must NOT confer the gain — the "
                 "culture must live in the CONTENT of what was recorded and "
                 "taught, not in the mere possession of an inherited "
                 "artifact.",
         kills="GOAL.md's Lamarckian-inheritance claim — 'the caveman's "
               "fireside story made structural'. If diaries cannot "
               "accumulate across individuals, death is still a page turn "
               "for HIM, but the project's one deliberate improvement on "
               "biology is decoration.",
         notes="ME.9 (attributed recall, PASS) and ME.10 (diary/weights "
               "double dissociation, PASS) are the substrate: transmission "
               "'only through diaries and teaching, never weights' is only "
               "enforceable because ME.10 proved the two stores separable. "
               "GEN.02 because teaching is a social act — a generation that "
               "cannot learn from watching a live demonstrator is limited "
               "to the written channel, and the spec must know which "
               "channel carried the culture. "
               "  COVERS: social/other agents (claim), memory across lives "
               "(claim)"),

    # ── HEARING (HR family) — docs/research/HEARING_BAKEOFF.md, registered
    # 2026-09-03 per the INTEGRATION_QUEUE 5-step protocol. Specs verbatim
    # from the doc's §2.4/§3.1/§3.2/§4.1/§4.2/§5.3 drafts; the only additions
    # are notes-only COVERS annotations (the NE.00 registration precedent).
    # Cross-check (step 1) found no refutation; the PLASTIC-ONLY decree does
    # not bind jobs (a)/(b) (Whisper/speaker-ID are OUTSIDE the brain, the
    # LLM-parent status), and HR.6's frozen-tower arm A6 enters as
    # SCORED-AND-INELIGIBLE per SYSTEM.md's 08-24 amendment. Ordering per
    # §6.3: the world-sound arm (HR.5 -> HR.7 -> HR.6 CPU arms) needs no
    # downloads and no disk; the speech arm (HR.1 -> HR.2/HR.3 -> HR.4) is
    # gated on the /data free-space escalation in DECISIONS_NEEDED.md.

    Spec("HR.1", 2, "The voice corpus is honest before anyone is scored",
         hypothesis="A speaker corpus exists on this box with >=8 enrolled and "
                    ">=8 held-out UNKNOWN speakers, disjoint enrolment/test "
                    "utterances, CROSS-SESSION test material, and a "
                    "NOISE/REVERB stratum, such that no non-vocal channel cue "
                    "can identify a speaker in either stratum.",
         falsified_by="A probe on non-vocal features alone (silence-segment "
                      "spectrum, DC offset, noise floor, clip loudness) "
                      "identifies the speaker above chance+5% — then every "
                      "speaker-ID number downstream is a microphone "
                      "measurement, not a voice measurement.",
         null_baseline="Chance = 1/n_enrolled for the channel probe.",
         metric="min_channel_leak_margin", budget=Budget.CPU, seeds=3,
         depends_on=[],
         control="A DELIBERATELY LEAKY variant — enrolment and test drawn from "
                 "the same session — must be identified WELL above chance by "
                 "the same probe. A leak detector that cannot see a planted "
                 "leak has measured nothing (docs/LESSONS.md, T0.13).",
         kills="HR.2, HR.3, HR.4. A speaker experiment on a leaky corpus "
               "measures the leak.",
         notes="Corpus: LibriSpeech dev-clean, CC-BY-4.0, 40 speakers "
               "(20M/20F), ~5.4 h, split 20 enrolled / 20 impostor, PLUS a "
               "handful of owner-recorded utterances for the deployment case. "
               "VERIFIED REACHABLE 2026-08-09: "
               "https://www.openslr.org/resources/12/dev-clean.tar.gz returns "
               "HTTP 200, Content-Length 337,926,286 (338 MB). test-clean is a "
               "further 347 MB and 40 DISJOINT speakers if a larger impostor "
               "set is wanted, but 685 MB exceeds the worst-case free disk — "
               "prefer the 20/20 split of dev-clean alone. REJECTED: VCTK "
               "(110 speakers, multi-session, ideal on paper) is an 11.7 GB "
               "download — verified Content-Length 11,749,118,645 — and does "
               "not fit on this box at any observed free-space level. NOTE THE "
               "DISK: /data free space was observed swinging between 725 MB and "
               "4.8 GB within an hour (shared with other tenants), so the "
               "corpus and the models may not both fit. Check free space "
               "BEFORE fetching and size for the worst case — "
               "cannot both live there — see the escalation in "
               "docs/research/HEARING_BAKEOFF.md section 1.0. LibriSpeech "
               "speakers are single-session per chapter, so cross-session "
               "means cross-CHAPTER at minimum and the control above is what "
               "certifies that it is enough. "
               "THE NOISE STRATUM IS NOT OPTIONAL. LibriSpeech is clean, "
               "near-field, read speech — the most favourable possible domain, "
               "and testing only on it reports the method's best day as its "
               "average (docs/LESSONS.md, 'ask what your synthetic data makes "
               "EASY'). SVeritas (arXiv:2509.17091) measures the size of the "
               "gap: ECAPA at 0.8-1.0% EER on VoxCeleb1-O reads 6.13% on "
               "CommonVoice clean and 15.88% with environmental noise and RIR "
               "at 15 dB SNR. Build the stratum by convolving with room "
               "impulse responses and adding environmental noise at 15 dB SNR "
               "— both synthesisable on this box with scipy alone, no "
               "download — and gate HR.3 on the MINIMUM over strata, never the "
               "average (docs/LESSONS.md, ME.11's deleted register). "
               "  COVERS: hearing (fixture)"),

    Spec("HR.2", 2, "ASR bakeoff: the cheapest transcriber that gets Jack's words right",
         hypothesis="At least one open-weight, locally-runnable ASR arm "
                    "transcribes Jack's command register with word accuracy "
                    ">= 0.90 at RTF <= 0.30 on this box, and beats the no-ASR "
                    "null by >= 3 sigma.",
         falsified_by="Every arm that clears 0.90 accuracy has RTF > 0.30 (no "
                      "live transcription on this box — escalate: batch "
                      "transcription only, or a smaller command grammar), OR "
                      "no arm clears 0.90 (Jack's vocabulary is the problem, "
                      "not the model).",
         null_baseline="A no-ASR transcriber that emits the most frequent "
                       "command string regardless of the audio. Word accuracy "
                       "= the majority-class rate; the learning gate is 3 "
                       "sigma over it.",
         metric="min_register_word_accuracy_at_rtf_budget",
         budget=Budget.CPU_LONG, seeds=3,
         depends_on=["HR.1"],
         control="TWO controls that must fail, and the first is a hard "
                 "DISQUALIFIER whatever an arm's WER. (a) SILENCE "
                 "HALLUCINATION: 60 s of room tone and 60 s of ContactAudio "
                 "impacts, containing no speech, must yield ZERO transcribed "
                 "words. This is not hypothetical — on pure non-speech input "
                 "Whisper hallucinates at 72.63% (small) and 86.88% "
                 "(large-v3), and the current energy VAD "
                 "(AudioListener.py:276-279, silence_threshold 0.01) opens a "
                 "speech segment for a door slam. (b) PHASE-SCRAMBLED speech "
                 "must transcribe to near-nothing; an arm that still emits "
                 "plausible commands is decoding its language-model prior, "
                 "not the audio.",
         kills="The transformers .generate() path in "
               "AudioListener._transcribe_local (lines 360-366) — MEASURED "
               "here as the slowest possible way to run this model — and the "
               "entire _transcribe_api path (line 372) unconditionally, "
               "because it calls a PAID OpenAI endpoint and SYSTEM.md forbids "
               "paid compute. Deleting the API path is not contingent on this "
               "bakeoff.",
         notes="ARMS, cost = RTF MEASURED ON THIS BOX at 4 threads, nice 19 "
               "(peak RSS reported alongside). "
               "THE ARM ORDERING IS ALREADY PARTLY MEASURED, 2026-08-09, on "
               "66 s of speech at beam 5, and it INVERTS the x86 conventional "
               "wisdom, so pick arms from these numbers and not from a blog: "
               "whisper.cpp tiny.en f16 RTF 0.106 / base.en f16 0.192 / "
               "small.en f16 0.766, against faster-whisper tiny.en int8 0.877 "
               "/ base.en int8 1.036 / small.en int8 2.926. whisper.cpp is "
               "3.8-8.3x FASTER on this hardware, and CTranslate2 int8 is "
               "SLOWER than its own float32 (0.877 vs 0.528 at tiny). Cause: "
               "the aarch64 CT2 wheel is built WITH_RUY + OpenBLAS and no "
               "oneDNN, and Neoverse-N1 has dotprod but NO i8mm. "
               "A0 whisper.cpp base.en f16 (cost 0.192, 348 MB) - the "
               "measured incumbent. "
               "A1 whisper.cpp tiny.en f16 (cost 0.106, 233 MB) - cheapest "
               "Whisper; the question is whether 5.6% LibriSpeech test-clean "
               "WER survives Jack's proper nouns. "
               "A2 PARAKEET TDT 0.6B v2 via sherpa-onnx (CC-BY-4.0, ~630 MB "
               "int8). PREDICTED WINNER: published RTF 0.088 at 4 threads on "
               "a Cortex-A76 with avg WER 6.05 - better than whisper "
               "large-v3 and ~9x faster than whisper.cpp small.en here. It is "
               "a TDT model: ONE forward pass, linear in ACTUAL audio length, "
               "no autoregressive loop - whereas Whisper pays a full 30 s "
               "encoder window for a 2 s command. That structural difference "
               "is the whole decision on this hardware. "
               "A3 Moonshine Small Streaming (MIT, 527 ms on an RPi 5) - "
               "explicitly edge-designed, the other non-Whisper shape. "
               "A4 vosk-model-small-en-us (40 MB, Apache-2.0, LibriSpeech WER "
               "9.85) - the cheap REFERENCE ARM whose failure would indict the "
               "task (docs/LESSONS.md, T1.02). "
               "A5 distil-medium.en, ONLY IF a distil arm is run: distillation "
               "freezes the 32-layer ENCODER and cuts the decoder to 2 layers, "
               "so its 5.8x A100 speedup does NOT transfer to CPU where the "
               "encoder dominates - the one ARM datapoint measures 1.78x. "
               "Never distil-large. "
               "DO NOT QUANTIZE below small: MEASURED here, ggml q5_1 is "
               "SLOWER than f16 at tiny and base (0.113 vs 0.106; 0.218 vs "
               "0.192) because without i8mm the dequantization overhead "
               "outweighs the bandwidth win. "
               "EXCLUDED ON LICENCE: canary-1b is CC-BY-NC-4.0. EXCLUDED ON "
               "FEASIBILITY: Kyutai STT has no CPU path. "
               "EVERY ARM RUNS BEHIND THE SAME silero-vad v6 ONNX GATE (1.23 "
               "MB, MIT, 16 kHz-only) with condition_on_previous_text=False, "
               "or the comparison is a comparison of VADs. Do NOT rely on "
               "no_speech_threshold: openai/whisper ANDs it with "
               "logprob_threshold, so a CONFIDENT hallucination (high "
               "avg_logprob) resets should_skip to False and survives at "
               "no_speech_prob 0.99 - an inert gate in the exact sense "
               "docs/LESSONS.md/T0.13 describes. Do NOT use webrtcvad: it "
               "scores 0.00 on ESC-50 noise rejection against silero v6's "
               "0.87. Do NOT use ten-vad: non-compete licence clause and no "
               "Linux arm64 build. "
               "TEST SET: two registers, reported SEPARATELY and gated on the "
               "MINIMUM (docs/LESSONS.md, ME.11's deleted register): "
               "(R1) short imperatives from Jack's actual command grammar "
               "('climb the ladder', 'come here'); (R2) PROPER NOUNS - the "
               "enrolled speakers' names - which small models mangle and which "
               "HR.4 depends on, because an attribution question is addressed "
               "to a NAME. "
               "  COVERS: hearing (fixture)"),

    Spec("HR.3", 2, "Speaker-ID bakeoff: which of the enrolled few, or nobody",
         hypothesis="At least one open-weight speaker embedder gives >= 0.85 "
                    "balanced open-set identification accuracy over "
                    "(N enrolled + unknown) on CROSS-SESSION audio, from "
                    "<= 30 s of enrolment per speaker, with the decision "
                    "threshold calibrated on a held-out split — AND holds "
                    ">= 0.70 on the NOISE/REVERB stratum. Gated on the MINIMUM "
                    "of the two strata, never the average.",
         falsified_by="No arm reaches 0.85 clean / 0.70 noisy at <= 30 s "
                      "enrolment — then HR.4's 0.80 end-to-end bar is "
                      "unreachable and the honest options are (i) more "
                      "enrolment audio, (ii) fewer enrolled people, (iii) "
                      "longer minimum utterances, or (iv) Jack ASKS who is "
                      "speaking. Record which; do not quietly lower the bar.",
         null_baseline="Chance = 1/(N+1) with balanced classes. PLUS a "
                       "REFERENCE ARM simple enough that its failure indicts "
                       "the task: nearest-centroid on mean MFCCs. If the "
                       "reference arm also fails, the corpus or the protocol "
                       "is broken, not the models (docs/LESSONS.md, T1.02).",
         metric="min_stratum_open_set_balanced_accuracy",
         budget=Budget.CPU_LONG, seeds=3,
         depends_on=["HR.1"],
         control="FOUR controls. (a) SAME-SESSION enrolment must score HIGHER; "
                 "if it does not, HR.1's corpus has no session variation. "
                 "(b) SILENCE segments must be rejected as 'unknown' >= 0.95 of "
                 "the time — an embedder that confidently names a speaker from "
                 "room tone is scoring the channel. (c) THRESHOLD SENSITIVITY: "
                 "sweep tau and report the whole curve. An arm whose accuracy "
                 "is FLAT in tau has an 'unknown' class that is not doing "
                 "anything, and its open-set number is a closed-set number "
                 "wearing a hat — this is the T0.13 lesson (a threshold you "
                 "never watch fire is not a threshold) applied to a "
                 "hyperparameter. (d) ABSTAIN-ON-EVERYTHING must FAIL: an arm "
                 "that answers 'unknown' to every input scores perfectly on "
                 "unknown-rejection, so abstain_rate_on_enrolled is gated at "
                 "<= 0.15 in the opposite direction (docs/LESSONS.md).",
         kills="Five of six embedders. The survivor is what writes "
               "EpisodicMemory.speaker; the rest are deleted, not kept.",
         notes="ARMS, cost = MB RESIDENT (weights + runtime), measured with "
               "one embedding computed; ms per second of audio reported "
               "alongside. MB is the tie-breaker because /data free space was "
               "observed as low as 725 MB and torch's aarch64 wheel alone is "
               "427 MB. "
               "A0 mean-MFCC nearest centroid (~0 MB, scipy only) — the "
               "reference arm whose failure indicts the task. "
               "A1 CAM++ ONNX (7.18 M, 29.3 MB, Vox1-O 0.80 plain / 0.66 "
               "LM+ASnorm, Apache-2.0) — the accuracy/cost sweet spot, ~2x "
               "faster than ECAPA on CPU. "
               "A2 WeSpeaker ResNet34-LM ONNX (6.63 M, 26.5 MB, 256-d, 0.797, "
               "CC-BY-4.0, UNGATED) — the embedder inside pyannote 3.1, "
               "obtainable without the pipeline or an HF token. "
               "A3 WeSpeaker ECAPA c512 ONNX (6.19 M, 24.9 MB, 1.07/0.78). "
               "A4 SpeechBrain ECAPA-TDNN (20.8 M, 83.3 MB, 0.80 s-norm / 0.90 "
               "raw, Apache-2.0) — the incumbent of the literature, included "
               "BECAUSE it costs torch: it is the arm that tests whether the "
               "427 MB dependency buys anything. "
               "A5 3D-Speaker ERes2NetV2 ONNX (17.8 M, 71.4 MB, 0.61) — "
               "designed for SHORT utterances (0.98% at 3 s, 1.48% at 2 s), "
               "which is Jack's actual regime. "
               "RUNTIME: sherpa-onnx, VERIFIED 2026-08-09 to publish a "
               "cp39-aarch64 manylinux2014 wheel of 4.13 MB with ONE "
               "dependency — no torch, no HF token, no gated repos. Its "
               "SpeakerEmbeddingManager gives enrolment-by-averaging "
               "(Add(name, embedding_list)), open-set Search() returning "
               "empty-string for unknown, and Score() for calibration "
               "logging, so the 'unknown' reject is a first-class return "
               "value rather than something we bolt on. "
               "NOT ARMS, each excluded on measured evidence: "
               "DIARIZATION (pyannote, Sortformer) answers 'who spoke WHEN' "
               "with anonymous cluster ids; EpisodicMemory.speaker needs a "
               "NAME — only the embedder inside pyannote is a candidate, and "
               "that is A2. Sortformer is additionally CC-BY-NC-4.0 "
               "(non-commercial), 123 M params, 493 MB, and needs NeMo, whose "
               "Linux deps pull CUDA bindings with no GPU present. "
               "SpeechBrain x-vector at 3.23% EER is dominated by WeSpeaker's "
               "x-vector at 1.99/1.59 for the same cost. Resemblyzer is "
               "dominated twice: ~4.5% EER on the authors' own internal set "
               "with NO published Vox1-O number, and it still needs torch. "
               "SCORING: cosine, NOT PLDA — WeSpeaker's ResNet34 reads 0.797 "
               "cosine vs 1.207 PLDA under margin training. And do NOT assume "
               "AS-Norm helps: VoxWatch (arXiv:2307.00169), the first public "
               "open-set-ID benchmark, found adaptive score normalisation did "
               "NOT consistently improve OSI while score CALIBRATION did. The "
               "technique every verification paper recommends is the one that "
               "does not transfer to this task. "
               "DO NOT SHIP A PAPER'S THRESHOLD. Shipped values disagree: "
               "SpeechBrain 0.25 cosine similarity, pyannote 0.7046 cosine "
               "DISTANCE (~0.295 similarity), sherpa-onnx 0.6. Calibrate on a "
               "held-out split of HR.1's corpus. "
               "CALIBRATE EXPECTATIONS FROM SVeritas (arXiv:2509.17091): the "
               "same ECAPA that reads 0.8-1.0% EER on VoxCeleb1-O reads 6.13% "
               "on CommonVoice clean and 15.88% with environmental noise and "
               "RIR at 15 dB SNR — a ~20x degradation from the model card. Any "
               "gate set against a 1% EER expectation has mis-specified itself. "
               "That is why the noisy stratum exists and why the metric is the "
               "MINIMUM over strata. "
               "REPORT, DO NOT AVERAGE, THREE AXES: enrolment seconds "
               "(5/15/30/60, gate on 30), N enrolled (2/4/8 — Jack needs a "
               "household, not VoxCeleb), and test-utterance duration. The "
               "duration cliff is between 2 s and 1 s: 2 s -> 1 s roughly "
               "TRIPLES EER and 1 s -> 0.5 s triples it again (one ECAPA "
               "baseline: 2.30% at 2 s, 6.98% at 1 s, 17.29% at 0.5 s). Jack's "
               "real utterances — 'stop', 'hello Jack' — sit under a second, "
               "so a minimum-duration gate before scoring is a DESIGN "
               "REQUIREMENT, not a tuning detail. "
               "EER IS THE WRONG HEADLINE and is deliberately not the metric: "
               "it assumes a balanced target/non-target prior, and Jack will "
               "hear far more non-target speech (strangers, a radio, his own "
               "TTS) than target speech. Report false-accept at a fixed miss "
               "rate alongside the headline. "
               "  COVERS: hearing (fixture)"),

    Spec("HR.4", 2, "He knows who told him, from the voice alone",
         hypothesis="With the speaker field produced by a voice embedder "
                    "instead of handed to the test, ME.9's attributed-recall "
                    "battery still clears 0.80 on ALL THREE channels "
                    "(heard/said/did), with misattribution <=0.02 and "
                    "unknown-speaker rejection >=0.90.",
         falsified_by="Any channel below 0.80, OR misattribution above 0.02, OR "
                      "unknown-rejection below 0.90, OR abstention on enrolled "
                      "speakers above 0.15 (calling everyone 'unknown' is not a "
                      "pass). Also falsified if the TEXT-ONLY null matches the "
                      "voice pipeline — then the voice channel is decorative "
                      "and Jack should just read the words.",
         null_baseline="THREE nulls, reported per channel, never averaged: "
                       "(i) single-speaker (everything filed under the most "
                       "frequent name — scores 1.0 on said/did by "
                       "construction, which is exactly why the gate is per "
                       "channel); (ii) random speaker from the enrolled set; "
                       "(iii) TEXT-ONLY attribution by a lexical classifier on "
                       "the transcript — the placebo channel.",
         metric="min_channel_attributed_acc_from_voice",
         budget=Budget.CPU_LONG, seeds=3,
         depends_on=["HR.1", "HR.2", "HR.3", "ME.9"],
         control="SWAP-THE-VOICES, mirroring ME.9's swapped-provenance control "
                 "one level lower. Exchange the ENROLMENT AUDIO of speakers A "
                 "and B; leave transcripts, topics, questions and gold answers "
                 "byte-identical. Attribution must INVERT, not merely collapse: "
                 "swapped_acc_AB <= 0.10 AND swapped_INVERTED_acc_AB >= 0.80. A "
                 "collapse alone is also what a broken loader produces; only "
                 "the inversion shows the voice was read and the predicted "
                 "wrong conclusion reached. Second control: same-session "
                 "enrolment must score HIGHER — if it does not, the corpus has "
                 "no session variation and HR.1 passed wrongly.",
         kills="The sentence 'Jack remembers who told him what' as an "
               "end-to-end claim. ME.9 keeps its PASS — it tests the retrieval "
               "contract, and that contract is real — but until HR.4 passes, "
               "the speaker field is supplied by the test harness and by "
               "nothing in the live system (AudioListener.py produces text "
               "only; EpisodicMemory.record takes `speaker` on trust).",
         notes="This is the biggest hole in the memory pillar and it is a "
               "COMPOSITION failure, not a component failure: ME.9 PASSES at "
               "1.0 and every link is individually fine. Ceiling analysis: "
               "end-to-end accuracy ~ ME.9_acc x speaker_id_acc x asr_acc, so "
               "the 0.80 bar needs speaker-ID at ~0.85 cross-session with ASR "
               "near-perfect on the command register. RUN IT TWICE and report "
               "both, so a failure localises to a LINK rather than to 'the "
               "chain': (i) GOLD transcripts + inferred speaker, isolating "
               "HR.3's contribution; (ii) HR.2's transcripts + inferred "
               "speaker, the real system. The gap between them is the ASR tax, "
               "and if it is large the fix belongs in HR.2, not here. "
               "MISATTRIBUTION IS NOT THE COMPLEMENT OF ACCURACY: 'unknown' is "
               "a miss, a wrong NAME is a permanent false memory that every "
               "later recall repeats with confidence. Gate them separately. "
               "  COVERS: hearing (claim), social/other agents (claim)"),

    Spec("HR.5", 2, "The playground makes the sounds GOAL.md names",
         hypothesis="ContactAudio emits distinguishable, correctly-labelled "
                    "events for water entry, creak-under-load and "
                    "rolling/sliding, in addition to impacts; and events "
                    "caused by Jack's own body carry a SELF flag.",
         falsified_by="Any of the four is absent, or a linear probe on band "
                       "energies cannot separate the four classes above "
                       "chance+20% — a sound Jack cannot distinguish is not a "
                       "sound he can learn from.",
         null_baseline="Chance = 0.25 over the four classes; plus the CURRENT "
                       "impact-only synth, on which water entry, creak and "
                       "rolling are all literally the same event type "
                       "(MEASURED 2026-08-09: dropping an object into the pool "
                       "and onto dry floor both produce ONLY impact rings, "
                       "force 816 N vs 898 N; 3 s of sliding produces 3 onset "
                       "events and no sustained sound).",
         metric="four_class_audio_separability", budget=Budget.CPU, seeds=3,
         depends_on=["PG.5", "PG.2"],
         control="POSITION-ONLY probe must be at chance. The pool sits at a "
                 "FIXED (2.6, -2.4) in every playground — PlaygroundParams "
                 "randomises pool_size and pool_depth but NOT location — so "
                 "'water entry' is perfectly predicted by bearing alone unless "
                 "the pool is relocated per episode. If the position-only "
                 "probe succeeds, the class labels are geography and every "
                 "later audio-classification number is void.",
         kills="The GOAL.md sentence 'he must hear the ladder creak, the "
               "splash, the thud of his own fall' as anything but aspiration. "
               "Two of those three sounds do not exist in the fixture today "
               "and the third cannot occur, because the humanoid is not in the "
               "playground (playground.build_mjcf(with_humanoid=False); bodies "
               "are world/apple/obj0-2/seesaw).",
         notes="Rows 5, 6 and 7 of the inventory in "
               "docs/research/HEARING_BAKEOFF.md section 5.2 are ONE piece of "
               "missing machinery: a sustained NOISE voice driven by a "
               "persisting contact (tangential velocity x normal force), "
               "versus the impulsive MODAL voice that exists. Water entry is a "
               "surface-crossing detector inside Water.apply emitting a "
               "broadband burst scaled by entry velocity - Water is a FORCE "
               "FIELD (playground.py:246) and generates no MuJoCo contact, "
               "which is exactly why it is currently silent. Self/other is one "
               "flag: geom_bodyid in Jack's body set. All three labels are "
               "free and exact, which is the whole reason sim audio is worth "
               "having (docs/research/UNIFIED_BRAIN.md section 4). "
               "PREREQUISITE FOR HR.6 BEING INFORMATIVE: with only impacts, "
               "Jack's entire auditory world is (onset, f0, level, pan) - four "
               "numbers - and a representation bakeoff on it measures how well "
               "each encoder recovers four numbers. "
               "  COVERS: hearing (fixture)"),

    Spec("HR.6", 4, "How contact audio enters the brain: mel vs raw vs tokens vs nothing",
         # ARMS: A0 no-audio | A0b placebo | A1 raw | A2 mel | A3 mel+ILD
         #       A4 discrete tokens | A5 hand-crafted event vector
         #       A6 frozen CED-tiny tower   (see notes for costs)
         hypothesis="At matched tokens-per-modality, matched trainable "
                    "parameters (+-5%), matched steps and matched data order, "
                    "at least one audio representation beats BOTH the "
                    "NO-AUDIO ablation and the PLACEBO-AUDIO channel by >= 3 "
                    "sigma on the audio-dependent battery, and the ranking is "
                    "stable across 3 paired seeds.",
         falsified_by="Every audio arm ties the PLACEBO channel — hearing is "
                      "decorative at this scale and does not earn its "
                      "parameters (the Tier-3 rule; report it, do not re-run "
                      "until it looks better). OR: the hand-crafted EVENT-VECTOR "
                      "arm ties every learned encoder — which indicts the "
                      "FIXTURE, not the brain, and sends the work to section 5 "
                      "of docs/research/HEARING_BAKEOFF.md rather than to a "
                      "bigger model.",
         null_baseline="TWO nulls, and the second is the load-bearing one. "
                       "(i) NO-AUDIO: the audio stem removed entirely, "
                       "parameters returned to the trunk so total capacity "
                       "matches. If the brain performs identically without "
                       "hearing, audio has not earned its parameters. "
                       "(ii) PLACEBO AUDIO: a matched-noise channel with the "
                       "SAME token count, encoder capacity and dropout rate, "
                       "wired in exactly like the real one "
                       "(UNIFIED_BRAIN_BAKEOFF.md section 2a). Its spread ACROSS "
                       "SEEDS is the empirical null distribution for "
                       "'decorative', re-estimated every run rather than "
                       "assumed to be zero.",
         metric="audio_margin_over_placebo", budget=Budget.GPU, seeds=3,
         # HR.5 added 2026-09-03 (65th audit B1): HR.5's own notes call it
         # "PREREQUISITE FOR HR.6 BEING INFORMATIVE" and it FAILed 05:25
         # 2026-09-03 (classes_present 1.0/4, no kind label, no self flag) —
         # a fixture of impacts-only reduces this bakeoff to recovering four
         # numbers, the branch the staging valve (A2 vs A0b) cannot catch
         # because HR.5's predicted failure is A5 TIES EVERYTHING.
         depends_on=["HR.7", "PG.5", "PG.7", "HR.5"],
         control="Every surviving arm must FAIL the cross-episode SWAP "
                 "ablation: swapping the audio stream between episodes, "
                 "preserving both marginals and the temporal statistics, must "
                 "hurt. Swap is the only perturbation that isolates "
                 "CORRESPONDENCE, which is what binding means. An arm "
                 "invariant to swapping has learned the audio MARGINAL and its "
                 "score is uninterpretable. Second, opposite-direction "
                 "control: the PLACEBO column must be SMALL. A large placebo "
                 "delta means the ablation procedure is measuring "
                 "off-manifold shock rather than information, and every other "
                 "column in the matrix is void.",
         kills="Four of six audio front-ends, and possibly the audio modality "
               "itself. Also kills UnifiedBrain.AudioEncoder's wav2vec2 path "
               "if A4 loses: wav2vec2 is trained on 960 h of read English "
               "speech and Jack's audio is four-partial exponential rings.",
         notes="ARMS, cost = MEASURED ms per 0.5 s window at nice 19, "
               "OMP_NUM_THREADS=2 (params alongside). TOKEN COUNT EQUALISED AT "
               "4 for every arm (arXiv:2601.16667 - unequal token budgets make "
               "this a comparison of token budgets). "
               "A0 NO-AUDIO ablation - the null. cost 0. "
               "A0b PLACEBO AUDIO - matched noise, 4 tokens, same capacity. "
               "cost = A2's. "
               "A1 RAW WAVEFORM, wav2vec2-style 7-layer strided conv stem. "
               "MEASURED 65.5 ms / 0.5 s window, 4.21 M params - 12x A2. "
               "A2 2-CHANNEL LOG-MEL (64 bins, 25 ms/10 ms) -> Conv2d stem -> "
               "4 tokens. MEASURED 5.6 ms / 0.5 s window, 167 K params. The "
               "incumbent recommendation from UNIFIED_BRAIN.md section 4. "
               "A3 A2 + EXPLICIT BEARING FEATURES: per-band interaural level "
               "difference appended as extra channels. Cheap; tests whether "
               "the stem needs help finding what PG.5 proved is there. "
               "A4 DISCRETE TOKENS, frozen encoder. If run, use DAC 24k "
               "(298.7 MB, MIT) — Codec-SUPERB finds DAC the only codec to "
               "significantly beat EnCodec on the NON-SPEECH audio category — "
               "or EnCodec 24k (93.1 MB) as the cheap variant. NOT Mimi: "
               "12.5 Hz is 80 ms per token, longer than the transient that "
               "carries material identity, and its model card says it was "
               "trained on speech only. NOT WavTokenizer as shipped: the "
               "published ckpt is 1.58-1.76 GB because it carries optimizer "
               "state, and does not fit /data at any observed free-space "
               "level without re-exporting the ~80 M-param model first. "
               "Entropy coding OFF — we want raw RVQ indices, not a "
               "bitstream. Predicted to fail HR.7 before it ever gets here. "
               "A6 FROZEN PRETRAINED TOWER: CED-tiny (5.5 M, 22 MB, 0.481 "
               "AudioSet mAP, Apache-2.0) embeddings -> 4 tokens. This is the "
               "arm that decides section 1.3's null hypothesis BY BAKEOFF "
               "rather than by argument (SYSTEM.md law 3). CED-tiny, not "
               "YAMNet: it is +0.175 mAP at 1.6x the size with 5x fewer "
               "tokens than AST. Predicted to lose, and the prediction is "
               "specific enough to be wrong: the AudioSet classes Jack needs "
               "are the WORST in the dataset (Scrape rank 519/527, Crack "
               "521st, Creak 29 training clips and 11% label accuracy, Roll 0% "
               "label accuracy), impacts occupy a ~3% duty cycle inside 10 s "
               "weakly-labelled clips so the pretext task never taught "
               "temporal localisation, and ContactAudio's output is a "
               "three-parameter synthetic family no YouTube tower has heard. "
               "If A6 WINS, section 1.3 of "
               "docs/research/HEARING_BAKEOFF.md is wrong and the tower ships "
               "— as SCORED-AND-INELIGIBLE for the seat until the PLASTIC-ONLY "
               "champion is formally contested (SYSTEM.md, 2026-08-24). "
               "NOT AN ARM: wav2vec2, which UnifiedBrain.AudioEncoder:1020 "
               "currently loads. On HEAR it scores 0.561 on ESC-50 against "
               "PANNs' 0.909 and CED-base's 0.967 — self-supervised SPEECH "
               "features are 20-40 points worse than AudioSet features on "
               "environmental sound. It is excluded on measured evidence, not "
               "taste. "
               "A5 HAND-CRAFTED EVENT VECTOR (t_onset, f0, level, pan) from "
               "ContactAudio's own labels, projected to 4 tokens. cost ~0. "
               "A5 IS THE MOST INFORMATIVE ARM AND THE ONE NOBODY WANTS TO "
               "RUN: it is docs/LESSONS.md's reference-arm rule inverted. Its "
               "FAILURE would be reassuring. Its SUCCESS - matching every "
               "learned encoder - would mean the sim's audio is a "
               "3-parameter family (f0, amplitude, pan) that a lookup table "
               "captures, so no representation experiment run on it can "
               "distinguish anything, and the fixture must grow (section 5) "
               "before the question is even well-posed. "
               "STAGING: A0, A0b, A2, A5 are pure CPU and cost minutes. Run "
               "them FIRST; if A2 does not beat A0b on CPU, the GPU arms are "
               "cancelled and hearing goes back to the drawing board for free. "
               "  COVERS: hearing (claim), one brain / unison (claim)"),

    Spec("HR.7", 2, "The audio stem does not deafen him to direction",
         hypothesis="A probe on the audio STEM's output tokens recovers the "
                    "source lateral angle to within 10 degrees on >= 0.9 of "
                    "PG.5's drop events — the same gate PG.5 applies to the raw "
                    "stereo signal. THE PROBE MUST NOT BE LINEAR IN THE "
                    "LOG-MEL: the constant-power pan law makes the log-domain "
                    "interaural level difference exactly atanh(p), so a linear "
                    "readout saturates at the lateral extremes. MEASURED HERE "
                    "2026-08-09 on 108 PG.5-style drops: linear probe 0.40, "
                    "analytic tanh link 1.00, mono control 0.10. A linear-probe "
                    "version of this spec would report a FALSE NEGATIVE on the "
                    "correct representation and kill the winning arm.",
         falsified_by="Any candidate stem whose tokens lose bearing. Directional "
                      "hearing is the ONLY thing PG.5 certifies, and it is what "
                      "makes audio useful for ACTION (turn toward the sound); a "
                      "stem that discards it reduces audio to an event detector.",
         null_baseline="PG.5's own mono render fed through the same stem — "
                       "bearing must be undecodable (<= 0.30), which is the "
                       "same bar PG.5's mono control clears.",
         metric="stem_bearing_probe_accuracy", budget=Budget.CPU, seeds=3,
         depends_on=["PG.5"],
         control="CHANNEL-SWAPPED input (L and R exchanged) must INVERT the "
                 "probe's sign on >= 0.9 of events, not merely degrade it. A "
                 "degradation is also what a broken probe produces; only the "
                 "inversion shows the stem read the interaural difference.",
         kills="Any stem in HR.6 that fails, before it is ever trained. Named "
               "prediction, pre-registered: the DISCRETE-TOKEN arm fails this. "
               "ContactAudio encodes bearing purely as interaural LEVEL "
               "difference (ContactAudio.py:188-195 applies gains gL/gR to the "
               "IDENTICAL signal — there is no interaural TIME difference at "
               "all), and RVQ codecs quantise a few-dB level offset inside a "
               "single codebook cell. If it passes, that prediction was wrong "
               "and this document is wrong with it.",
         notes="Log-mel preserves bearing provably: the pan law is a "
               "per-channel GAIN, so log(gR) - log(gL) = atanh(p) exactly "
               "(verified to machine precision at p = +-0.9, +-0.99), a "
               "constant offset between the two channels' log-mel planes, "
               "independent of mel bin. TWO IMPLEMENTATION TRAPS, both "
               "MEASURED here 2026-08-09 and both of which silently destroy "
               "the number: (1) a LINEAR probe scores 0.40 where the analytic "
               "tanh link scores 1.00, because atanh saturates exactly where "
               "bearing matters most; (2) the naive MEAN of per-bin log ILD "
               "scores 0.69 against 1.00 for energy-weighted pooling, because "
               "the log(x + 1e-6) floor pins near-silent bins to zero ILD and "
               "drags the mean toward centre. Pool in the ENERGY domain. The "
               "architectural failure mode is separate and worse: a stem whose "
               "first op averages over the CHANNEL dimension IS the mono "
               "control, silently. One line, deleting Jack's only directional "
               "sense, and nothing else in the ladder would notice. This spec "
               "is that guard. Scope note: two-channel stereo loses front/back "
               "(48% of front sources land in the back, arXiv:2309.13343) but "
               "its LATERAL accuracy matches 4-channel FOA (0.93 vs 0.91) - "
               "DCASE 2025 Task 3 moved to stereo, azimuth-only for exactly "
               "this reason, so PG.5's folded-azimuth scope is the field's "
               "ratified operating point, not a shortcut. "
               "  COVERS: hearing (sensor)"),

    Spec("HR.8", 4, "Blind playground audit: hearing carries content, not just parameters",
         hypothesis="With vision occluded and every event out of contact with "
                    "Jack, the model classifies audio events 4-way (impact / "
                    "water entry / creak / rolling) and reports bearing to "
                    "within 15 degrees, well above chance (>= 0.70 class "
                    "accuracy, lower bootstrap CI > 0.25).",
         falsified_by="Class accuracy at chance, OR indistinguishable from the "
                      "no-audio arm — hearing carries no content in Jack's "
                      "world.",
         null_baseline="Chance 0.25 on class, ~0.17 on bearing at 15 degrees. "
                       "PLUS the NO-AUDIO arm, which must sit at chance: if it "
                       "does not, the episode sampler leaks the class through "
                       "position and the task is broken.",
         metric="bpa_class_x_bearing", budget=Budget.CPU_LONG, seeds=3,
         depends_on=["HR.5", "HR.6"],
         control="FOUR controls, and they must fail DIFFERENTLY, which is the "
                 "evidence that bearing and identity are separate channels. "
                 "(a) MONO render: bearing collapses to chance, class accuracy "
                 "SURVIVES. (b) L/R SWAP: reported bearing INVERTS rather than "
                 "degrades. (c) SPECTRUM-FLATTENED: class accuracy collapses, "
                 "bearing SURVIVES. (d) NO-AUDIO: both at chance. A control "
                 "that takes down both halves at once indicts the shared "
                 "plumbing, not the mechanism (docs/LESSONS.md).",
         kills="'Hearing is load-bearing' (UB.4) as a claim. BPA is the "
               "cheapest experiment that could establish it and it needs no "
               "controller and no GPU.",
         notes="Proprioception, touch and vision are at chance BY "
               "CONSTRUCTION - every event happens away from Jack, out of "
               "frame - so the unimodal late ensemble is at chance and every "
               "point above 0.25 is hearing. Complementary to UB.9 (Heard, Not "
               "Seen), not a substitute: BPA certifies that audio carries "
               "CONTENT; UB.9 certifies that the content gets BOUND to the "
               "other senses. Both are needed and they can fail "
               "independently. "
               "  COVERS: hearing (claim), one brain / unison (claim)"),
]
