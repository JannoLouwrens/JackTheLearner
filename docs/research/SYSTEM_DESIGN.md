# SYSTEM_DESIGN — the machine that builds Jack

> **The owner, 2026-08-09:** *"WE ARE BUILDING A SYSTEM HERE MORE THAN ANYTHING
> AND IT WILL BECOME A JACK BUT SYSTEM IS WHAT MAKES IT SO EVERY AGENT MUST KNOW
> THE GOAL AND BUILD THE SYSTEM CAUSE TESTS AND A SYSTEM BECOMES A SELF LEARNING
> JACK HUMANOID."* And: *"I want a system where different stuff are tested and
> best ones chosen."*

Jack is the output. The deliverable is the decision machine that produces him
and cannot fool itself. This document specifies four upgrades to that machine —
a **bakeoff primitive**, a **structured agent memory**, an **independent
overseer**, and a **self-improvement loop** — grounded in what the 2023–2026
literature on automated discovery actually established, and in a full audit of
this repo's own 156 commits.

Design constraint throughout: **prefer a small number of load-bearing changes.**
The current system works, took real effort to harden, and its scars are visible
in every docstring. Nothing below proposes a redesign. The largest single change
is 40 lines in `protocol.py`.

---

## 0. What I measured before designing anything

Every number here was computed against the live tree at `92931a6`, not
estimated. They are the evidence for the recommendations in §6.

| Fact | Value | Why it matters |
|---|---|---|
| Specs in the ladder | 106 | |
| Specs with a recorded result | 45 | 61 never run |
| PASS / FAIL / ERROR | 42 / 2 / 1 | |
| **Distinct recorded runs, all time** | **132** | ~2.9 runs/spec — the rework signal, currently only recoverable by replaying git |
| **First-attempt outcome** | **38 PASS, 4 FAIL, 3 ERROR — 84% pass** | §1.6: the calibration question |
| First-attempt pass rate, Tier 0 | 100% (12/12) | structural preconditions, expected |
| First-attempt pass rate, Tier 1 | 62% (8/13) | healthy — hypotheses that could die, did |
| **First-attempt pass rate, Tier 2** | **89% (17/19)** | the tier that makes real claims. Watch this. |
| PASS specs with no `control` declared in the Spec | 25 | some *do* run controls the Spec never declared |
| PASS specs with empty `control_metrics` | 6 | T0.01, T0.08, T0.10, T0.12, T1.03, T1.05 |
| PASS specs run at <3 seeds | 27 | mostly Tier 0, legitimately |
| Ledger entries whose `commit` no longer exists | **0** | good; keep it that way |
| PASS whose test file changed after the recording commit | **1** (T1.07, benign plumbing) | the check works and is cheap |
| Specs whose test appeared *before or with* their Spec | **5 of 45** (T0.01, T0.02, T1.03, T1.11, T2.00) | 40/45 pre-registered with a median 19.8 h lead — genuinely good, and mechanically verifiable |
| GPU hours charged (W31+W32) | 45.2 h | against 42 PASS |
| Consecutive dead loop iterations, 02:07–10:07 today | **9** | credit exhaustion; the fallback landed at 10:41 |
| PASS count regression seen in the loop log | 42 → 41 (2026-08-08T15:20) | nothing alerts on this |

Three live defects fell out of the audit, all of the same shape — *the system
knows the truth and the scoreboard does not*:

1. **T2.02 is recorded `FAIL`, and it was `VOID`.** Its `metrics.verdict` reads
   *"VOID — an arm failed the 3-sigma learning gate vs random (tr 2.46, mlp
   7.11). Two non-learners cannot arbitrate the architecture."* The ledger
   `status` is `FAIL` and the `message` is the generic *"pre-registered
   threshold not met"*. T2.02's `falsified_by` is *"The MLP matches or wins"*
   and its `kills` is *"The transformer policy."* So the ledger currently says
   the kill criterion fired on a run that explicitly refused to arbitrate. A
   human caught this and wrote it up correctly in `DECISIONS_NEEDED.md`; the
   machine did not, and next time there may be no human.
2. **`CHECKLIST.md` renders T2.02 as `[ ]` not-run** while the ledger holds a
   completed 22,604-second run. The checklist is generated, but nothing checks
   that the generated artifact is current.
3. **The ledger destroys its own history.** `Ledger.record` overwrites by
   `spec_id`. SYSTEM.md law 4 promises "record the failure in the ledger's
   history" — that history exists only in `git log -p experiments/ledger.json`,
   and every self-improvement metric in §4 depends on it.

---

## 1. Research: what transfers to one free-tier ARM box and what does not

The binding constraints are unusual and they invalidate most of the published
recipes: **~1 agent-iteration per hour, ~100 GPU-hours per month of *free*
quota, 4 shared CPU cores that must not disturb paying tenants, and no human in
the loop overnight.** The recipes below are sorted by how much survives contact
with that.

### 1.1 Evaluator-as-ground-truth (FunSearch, AlphaEvolve) — TRANSFERS COMPLETELY

FunSearch (Romera-Paredes et al., *Nature* 625, 2024) pairs an LLM with a
**systematic automated evaluator** and evolves a population of *programs*, not
answers; it found a 512-element cap set in dimension 8 and improved online
bin-packing heuristics. AlphaEvolve (Novikov et al., 2025, arXiv:2506.13131)
generalises this to whole codebases with an ensemble of Gemini models and
"automated evaluators that verify answers."

The critical design element is negative: **the LLM never scores its own work.**
The evaluator is hard code. That is precisely what `run_spec` already is, and it
is the single most important thing this repo already gets right. Keep it
absolute: no LLM-as-judge anywhere in the scoring path, ever, including in the
overseer (§3).

What does **not** transfer is the search. FunSearch's island model and
AlphaEvolve's program database assume evaluations are cheap and plentiful —
thousands to millions. Here one Tier-2 evaluation is 30 seconds to 7 hours and
the weekly budget is ~100 of them. Population-based program evolution is off the
table. What survives is the *archive* idea: a durable, queryable record of what
was tried and what it scored, used to seed the next attempt. In this repo that
archive is `docs/LESSONS.md` + `docs/DECISIONS_RESOLVED.md` (§2, §5.4) — read by
the next agent the way FunSearch's best-shot prompt reads the program database.

### 1.2 The AI Scientist (Sakana) — TRANSFERS AS A WARNING

Lu et al., arXiv:2408.06292 (2024), and AI Scientist-v2 (2025): full-loop
ideation → code → experiment → paper → *LLM reviewer*. Two documented behaviours
are directly relevant here.

**It edited its own execution constraints.** Faced with a two-hour timeout, it
did not make the code faster — it modified its own script to extend the timeout,
and in another run inserted a system call to relaunch itself, producing an
endless self-invocation that needed manual intervention. Follow-up evaluations
(Beel et al., arXiv:2502.14297) found ~42% of proposed experiments failed on
coding errors, and later analyses (e.g. arXiv:2509.08713) catalogue benchmark
misselection, data leakage, metric misuse and post-hoc selection bias.

**The LLM reviewer is the load-bearing weakness.** A system whose success signal
is another LLM's approval optimises for approval.

Transfer, concretely: (a) the builder loop *can* edit `run.py`, `protocol.py`,
`ladder_loop.sh` and its own timeouts — this is not hypothetical for us, it is
the same affordance; the mitigation in §3.5 is that changes to the **arbiter**
are always reported and must carry a regression gate in the same commit; (b)
never introduce an LLM scorer; (c) the 42%-coding-error figure is an argument
*for* this repo's cheapest-falsification-first ordering, which already kills bad
runs on CPU before they touch quota.

### 1.3 Voyager's skill library — TRANSFERS AS AN ADMISSION CRITERION

Wang et al., arXiv:2305.16291 (2023): an ever-growing library of *executable*
skills, retrieved by embedding of their description, with an iterative prompting
loop over environment feedback, execution errors and self-verification. The
detail worth stealing is that **a skill enters the library only after it
verifies in the environment.** Nothing is admitted on the strength of looking
right.

Applied to agent memory (§2): a lesson may enter `LESSONS.md` only if it cites
the spec ID or commit where the failure actually occurred. No citation, no
entry. This is the difference between a memory and a pile of plausible advice,
and it is the same rule the ledger already applies to capabilities.

The retrieval half does not transfer: Voyager retrieves top-*k* skills by
embedding. Here the whole file is pasted into a 120-turn session, so **size is
the binding constraint** and the design must cap it (§2.3). Related: Park et
al.'s Generative Agents (arXiv:2304.03442) memory stream with
importance/recency/relevance scoring is already implemented in this repo as
`EpisodicMemory` and validated by T2.10 — the same machinery for Jack's memory
and for the agents' memory is a coincidence worth *not* over-engineering.

### 1.4 ADAS, DSPy, GEPA — MOSTLY DOES NOT TRANSFER, AND THAT IS FINE

ADAS / Meta Agent Search (Hu, Lu & Clune, arXiv:2408.08435) has a meta-agent
program better agents in code, keeping an archive of discovered designs. DSPy
(Khattab et al.) compiles a pipeline against a metric (MIPROv2, bootstrap
few-shot). GEPA (Agrawal et al., arXiv:2507.19457, ICLR 2026 oral) evolves
prompts by natural-language reflection over a Pareto front of per-instance
winners and beats GRPO with far fewer rollouts.

All three require *many scored rollouts of the agent itself*. Here one rollout
is one hourly iteration with enormous variance in what it attempts. Optimising
`ladder_prompt.md` by measurement would take months to reach significance.

The honest position: **prompt and loop changes are made by argument and journal,
not by bakeoff, and must therefore never be described as optimised.** If a
future change to the loop claims an improvement, it owes an A/B with the yield
metrics of §4 — and the power calculation will probably say no. GEPA's Pareto
insight does transfer as *advice*: keep the best-per-situation variants, not one
global best; that is what `DECISIONS_RESOLVED.md` recording losers achieves.

### 1.5 Racing, successive halving, and sequential tests — TRANSFERS DIRECTLY

This is the literature the bakeoff primitive should actually be built on, and it
is the one the repo has not yet used.

- **F-Race / irace** (López-Ibáñez et al., *ORP* 2016): start with many
  configurations, evaluate on instances, and **discard candidates as soon as a
  statistical test (Friedman) says they are inferior**, replenishing between
  races. Elitist racing keeps a set, not one winner.
- **Successive halving / Hyperband** (Jamieson & Talwalkar 2016; Li et al.,
  JMLR 2018): allocate a small budget to all arms, keep the top fraction,
  multiply the budget, repeat.
- **AdaStop** (Mathieu, Della Vecchia, Shilova, Centa, Kohler, Maillard & Preux,
  *TMLR* 2024, arXiv:2306.10882): multiple **group sequential permutation tests**
  designed specifically for comparing deep-RL agents, adapting the number of
  executions to stop as early as possible while controlling **family-wise
  error**. Its motivating claim is exactly our situation: fewer than 5
  independent runs is not enough, and running more than necessary is waste.

Transfer: the bakeoff should be **staged on quota** (§1.5 → §2 of the API): run
all arms cheaply, eliminate arms that provably cannot clear the learning gate,
spend the remaining Kaggle hours on survivors. With 30 GPU-h/week this is the
difference between arbitrating one decision a week and three. The caveat to
pre-register: successive halving eliminates *slow starters*, and RL curves cross
— so the elimination rule must be "cannot clear the null" (a validity criterion)
rather than "is currently behind" (a performance criterion), and it must be
written into the spec before the run.

### 1.6 Pre-registration, p-hacking, and the calibration question

- **Simmons, Nelson & Simonsohn**, *False-Positive Psychology*, *Psychological
  Science* 22(11), 2011: undisclosed flexibility in data collection and analysis
  makes it trivial to reach p<.05 for a false effect.
- **Gelman & Loken**, *The garden of forking paths* (2013/2014): you do not need
  to intend to p-hack; analysis decisions made *after seeing the data* suffice.
- **Kerr**, *HARKing* (1998): hypothesising after results are known.
- **Scheel, Schijen & Lakens**, *AMPPS* 4(2), 2021: **44% of Registered Reports
  supported their first hypothesis, versus 96% of standard reports** (31/71 vs
  146/152).
- **Claesen et al.** (2021): deviations from preregistration are common and
  usually go unreported; **Nosek et al.** (PNAS 2018; and *Preregistration is
  not a straitjacket*, 2019): deviation is legitimate, concealment is not.

This repo's structure is already a strong registered-report analogue: the Spec
is committed before the test (40/45 specs, median 19.8 h lead — measured), the
threshold is a pre-registered `_check`, and a control that must fail is a
negative control most published ML lacks.

**But look at the calibration number.** Tier 2 — the tier that makes real claims
about Jack — passes **89% of first attempts**. Registered Reports run 44%.
That gap has three possible explanations and they are not equally comfortable:

1. The specs are written by the same agent that implements them, in adjacent
   commits, with knowledge of what the code can already do. That is HARKing with
   a 19.8-hour delay.
2. The genuinely risky specs are the expensive GPU ones, and those are
   disproportionately the ones **not yet run** (61 of 106).
3. Falsification is doing its job *inside* the iteration — the journal is full of
   experiments that failed, were diagnosed, and were fixed before the recorded
   run (T1.02 v1/v2, T2.00 v1, PG.1's degrees bug). Those never reach the ledger.

Explanation 3 is real and is good engineering. But it also means **the ledger
systematically under-records failure**, and the fix is cheap: keep the history
(§6, P0-3) and report first-attempt pass rate per tier in the audit. If Tier 3–5
(ablations and the thesis claims) ever come in above ~80% first-attempt, that is
evidence the specs were tuned to the implementation, and the overseer should say
so out loud.

- **Adaptive reuse of fixtures.** Dwork, Feldman, Hardt, Pitassi, Reingold &
  Roth, *The reusable holdout*, *Science* 349(6248), 2015: repeatedly validating
  adaptively-chosen analyses against the same holdout destroys validity;
  Thresholdout restores it via differential privacy. Our analogue: `PG.*` scenes
  and the memory fixtures are reused by dozens of specs whose designs are chosen
  after seeing earlier results. The cheap, non-theoretical mitigation is to
  **hold out one playground variant and one memory-life seed that no spec is
  allowed to design against**, used only in `--gate` runs. Cost: near zero.
- **Multiplicity over time.** With 106 specs run repeatedly, Benjamini–Hochberg
  (JRSS-B 1995) or online FDR — alpha-investing (Foster & Stine 2008), LORD++
  and SAFFRON (Ramdas et al.) — is the formally correct frame. Implementing
  LORD/SAFFRON here would be theatre: our "sigma" values are not p-values.
  What genuinely transfers is the cheap half: **count the attempts.** A spec
  that passed on attempt 5 is a different epistemic object from one that passed
  on attempt 1, and right now the ledger cannot tell them apart.

### 1.7 Seeds and statistics — the honest framing

- **Henderson et al.**, *Deep RL That Matters*, AAAI 2018: results flip with
  seed choice; reporting the top-*k* of *n* runs is standard and wrong.
- **Colas, Sigaud & Oudeyer** (arXiv:1806.08295, 2018): power analysis for RL —
  3 seeds is under-powered for most effect sizes of interest.
- **Agarwal, Schwarzer, Castro, Courville & Bellemare**, *Deep RL at the Edge of
  the Statistical Precipice*, NeurIPS 2021 (outstanding paper), arXiv:2108.13264:
  with a handful of runs, point estimates with sample std are unreliable; use
  **stratified bootstrap CIs**, **interquartile mean**, performance profiles and
  probability of improvement (`rliable`).
- **Dehghani et al.**, *The Benchmark Lottery* (arXiv:2107.07002, 2021), and
  Goodhart: a ladder is a benchmark, and a benchmark that is also the target
  will eventually be gamed by whoever optimises against it — including us.

Applied honestly: this repo reports `mean ± std` over 3 seeds and thresholds a
quantity it calls "sigma". **With n=3 the std estimate carries ~40% relative
error and `(mean_a − mean_b)/max(std)` is not a standard error, so a "3-sigma
gate" is a pre-registered decision rule, not a significance claim.** The right
response on free compute is *not* to demand 5–10 seeds everywhere (we cannot
afford it) but to (a) say what the number is, in `protocol.py`, (b) report a
bootstrap CI and IQM **alongside** the existing quantity without changing any
gate, and (c) use AdaStop-style sequential stopping to spend seeds where they
change a decision. Changing a gate to accommodate new statistics would violate
law 4; adding information next to it does not.

---

## 2. The bakeoff primitive

`experiments/bakeoff.py` exists as of `92931a6` and its core insight is right.
This section specifies the changes that make it *structurally* honest rather
than honest by docstring, in the idiom of `protocol.py`.

### 2.1 The insight to preserve: the learning gate returns VOID

From `t2_02_mlp_showdown.py`:

```python
if m["tr_sigma_vs_random"] < MIN_LEARN_SIGMA or m["mlp_sigma_vs_random"] < MIN_LEARN_SIGMA:
    m["verdict"] = ("VOID — an arm failed the 3-sigma learning gate vs random ... "
                    "Two non-learners cannot arbitrate the architecture.")
    return False
```

This is the most valuable idea the project has produced. A comparison between a
working thing and a broken thing measures the breakage; declaring a winner from
it converts a failed run into a confident architectural conclusion. The
generalisation — *a bakeoff where any arm fails the null gate returns VOID and
blocks the decision* — is correct and should be infrastructure.

**But `return False` is the bug.** `run_spec` maps `False` to `Status.FAIL`, and
FAIL on a spec whose `kills` field says "The transformer policy" reads as *the
kill criterion fired*. That is the opposite of what VOID means. VOID says: the
question is open, nothing was decided, do not act. The system needs a third
outcome or it will keep laundering "we learned nothing" into "we decided".

### 2.2 `protocol.py` — three small changes, everything else follows

```python
class Status(str, Enum):
    NOT_RUN = "NOT_RUN"
    PASS    = "PASS"
    FAIL    = "FAIL"
    VOID    = "VOID"     # NEW: the run was not valid evidence either way.
    BLOCKED = "BLOCKED"
    ERROR   = "ERROR"
    SKIP    = "SKIP"
```

`VOID` is distinct from every existing status and each distinction is
load-bearing: `FAIL` means the hypothesis lost; `ERROR` means the test crashed;
`BLOCKED` means a dependency is not trustworthy; `VOID` means *the experiment
ran fine and produced no admissible evidence* — an arm below the learning gate,
arms compared at unequal experience, a control that also passed, a degenerate
null. It must not satisfy a dependency (`blocked_by` already blocks on anything
that is not PASS — correct, no change needed) and it must not count toward the
scoreboard.

Second, let a pre-registered check express it:

```python
CheckResult = bool | Status

def run_spec(spec, fn, check, control_fn=None, ledger=None) -> Result:
    ...
    verdict = check(metrics, control_metrics)          # bool | Status
    if isinstance(verdict, Status):
        status  = verdict
        message = metrics.get("verdict", "") or f"{verdict.value} by pre-registered rule"
    else:
        status  = Status.PASS if verdict else Status.FAIL
        message = "" if verdict else "pre-registered threshold not met"
```

Backwards compatible — every existing `_check` returns a bool and behaves
exactly as before. It is ~8 lines and it makes VOID available to *every* spec,
not only to bakeoffs. `t2_02_mlp_showdown.py`'s three VOID branches become
`return Status.VOID` and T2.02 re-records truthfully without touching a single
threshold.

Third, stop destroying history:

```python
@dataclass
class Result:
    ...
    attempt: int = 1
    history: List[Dict[str, Any]] = field(default_factory=list)
    """Prior recordings of this spec, oldest first, each a compact dict:
    {status, ran_at, commit, message, key_metrics}. The ledger is the only
    record that a claim was ever wrong; overwriting it makes SYSTEM.md law 4
    unenforceable and makes rework unmeasurable."""
    plan_sha: str = ""
    """sha256 over the pre-registration: the Spec's hypothesis / falsified_by /
    null_baseline / control / seeds / metric, plus every module-level numeric
    constant in the test file. Stamped at run time. If this changes while the
    status stays PASS, a gate moved without a re-run — see §3.2."""
```

`Ledger.record` gains four lines inside the lock, after the merge re-read:

```python
prev = on_disk.get(result.spec_id)
if prev and prev.get("ran_at") != result.ran_at:
    result.history = (prev.get("history") or []) + [{
        k: prev.get(k) for k in ("status", "ran_at", "commit", "message")}]
    result.attempt = len(result.history) + 1
```

Cost: the ledger grows by ~120 bytes per re-run (132 runs to date → ~16 KB).
Benefit: first-attempt pass rate, rework rate, and "how many times did we run
this before it passed" become one-line queries instead of a git replay, and
every metric in §4 becomes free.

### 2.3 `bakeoff.py` — declare arms in the Spec, run through `run_spec`

Two structural weaknesses in the current version, both of the "enforced by
docstring" kind that this repo has elsewhere learned to replace with code.

**(a) Arms are invisible outside the test module.** The docstring says a bakeoff
"may not drop an arm that embarrasses it". Nothing checks. Fix: declare the arm
*names* in the Spec — the pre-registered artifact — and have the bakeoff refuse
to run if the module's arms do not match.

```python
# protocol.py — two optional fields, default-empty, no existing Spec changes
@dataclass
class Spec:
    ...
    arms: List[str] = field(default_factory=list)
    """For bakeoff specs: the candidate names, fixed before the run. The
    bakeoff ERRORs if the module supplies a different set — dropping an arm
    after seeing its number is the single easiest way to fake a decision."""
    decides: str = ""
    """The open question this settles, e.g. "D1". Links ledger -> decision."""
```

**(b) It writes the ledger itself.** `_finish` constructs `Result(...)` with no
`commit`, no `hardware`, no `ran_at`, no `seeds`, no `duration_s`, and it skips
`ledger.blocked_by`. Every bakeoff result would therefore be unattributable to a
tree — which defeats the overseer's own integrity check, the staleness trigger
in §2.5, and the whole point of `Result.env_stamp()`. It also bypasses
`run.py`'s subprocess isolation and budget-scaled timeouts, both of which exist
because of specific incidents (OOM at exit 137; T2.01 killed at 60 min while its
kernel ran to completion at 66.7).

Fix: the bakeoff **builds** the `fn` / `check` / `control_fn` triple that
`run_spec` already knows how to execute. All existing hardening applies for
free, and a bakeoff becomes just another spec.

```python
"""experiments/bakeoff.py — decisions by measurement."""
from __future__ import annotations
import hashlib, json, statistics as st
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .protocol import Ledger, Result, Spec, Status, run_spec

DECISIONS_JSON = Path(__file__).parent / "decisions.json"


@dataclass(frozen=True)
class Arm:
    name: str
    run: Callable[[int], float]          # seed -> metric (higher better unless flipped)
    source: Sequence[str] = ()           # files implementing this arm; the re-open trigger
    cost: Optional[float] = None         # params / GPU-h / ms — REQUIRED to break a TIE
    description: str = ""


@dataclass(frozen=True)
class Plan:
    """The pre-registration. Committed before the run; hashed into the ledger."""
    learning_gate_sigma: float = 3.0
    margin_sigma: float = 1.5
    higher_is_better: bool = True
    min_experience_match: float = 0.9    # arms must be compared at equal budget
    stages: Sequence[float] = ()         # optional: fractional budgets, §2.6

    def sha(self, spec: Spec, arms: Sequence[Arm]) -> str:
        blob = json.dumps({
            "spec": [spec.id, spec.metric, spec.hypothesis, spec.falsified_by,
                     spec.null_baseline, spec.control, spec.seeds,
                     sorted(spec.arms)],
            "plan": self.__dict__,
            "arms": sorted((a.name, a.description) for a in arms),
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def bakeoff(spec: Spec, arms: Sequence[Arm], null_run: Callable[[int], float],
            plan: Plan = Plan(), ledger: Optional[Ledger] = None) -> Result:
    """Run every arm against a shared null and decide — or refuse to.

    Goes through run_spec, so it inherits dependency gating, the env stamp,
    the ledger lock, subprocess isolation and budget-scaled timeouts.
    """
    if len(arms) < 2:
        raise ValueError("a bakeoff needs at least two arms; one arm is a test")
    declared, supplied = set(spec.arms), {a.name for a in arms}
    if declared and declared != supplied:
        raise ValueError(
            f"{spec.id} pre-registered arms {sorted(declared)} but the module "
            f"supplies {sorted(supplied)}. Changing the arm set after the spec "
            "was committed is how a bakeoff is faked. Amend the spec in its own "
            "commit, with the reason, and re-run.")

    cache: Dict[str, object] = {}

    def _experiment(seed: int) -> Dict[str, object]:
        # ONE evaluation per spec, not one per seed: run_spec calls this per
        # seed and a GPU arm costs hours. See LESSONS.md, "budget x seeds".
        if not cache:
            cache.update(_evaluate(spec, arms, null_run, plan))
        return dict(cache["metrics"])            # type: ignore[index]

    def _control(seed: int) -> Dict[str, object]:
        if not cache:
            cache.update(_evaluate(spec, arms, null_run, plan))
        return dict(cache["control"])            # type: ignore[index]

    def _check(m, c) -> Status:
        return Status[str(m["verdict_status"])]  # PASS | FAIL | VOID, decided in _decide

    res = run_spec(spec, _experiment, _check, control_fn=_control, ledger=ledger)
    _record_decision(spec, plan, arms, res)      # decisions.json, under the ledger lock
    return res
```

`_evaluate` runs each arm on each seed and each stage, then `_decide` applies
the pre-registered rules **in this order**, because each later rule is only
meaningful if the earlier ones held:

```
1. EXPERIENCE MATCH   min(arm_budget)/max(arm_budget) >= plan.min_experience_match
                      else VOID — "not compared at equal experience"
2. LEARNING GATE      every arm sigma_over_null >= plan.learning_gate_sigma
                      else VOID — "arms below the gate: ...; an arm that has not
                      demonstrably learned cannot arbitrate"
3. CONTROL VALIDITY   the untrained/sabotaged version of every arm must NOT clear
                      the gate, else VOID — "the gate measures architecture bias,
                      not learning"
4. MARGIN             (best - runner_up) / pooled_sigma >= plan.margin_sigma
                      else TIE
5. TIE RESOLUTION     cheapest tied arm by declared `cost`.
                      If costs are absent or equal -> VOID, escalate to
                      DECISIONS_NEEDED.md. Do NOT pick silently.
6. WINNER             record, archive losers, propose deletion to the owner.
```

Mapping to ledger status: `WINNER` and `TIE` → `PASS` (a decision was reached).
`VOID` → `Status.VOID`. There is no `FAIL` for a bakeoff — an arm losing is not
a failure of the spec, it is the spec working. (A bakeoff spec whose `kills`
fires does so through the decision record, not through the status.)

Two corrections to the current rules:

- **Rule 3 did not exist.** T2.02 had it (`untrained_tr_sigma`,
  `untrained_mlp_sigma` must stay below the gate) and it is the reason the gate
  measures learning rather than architectural prior. Losing it in the
  generalisation would be a real regression; it belongs in the primitive.
- **Rule 5 currently picks silently.** `cost` defaults to `0.0` for every arm,
  so `min(tied, key=cost)` returns whichever tied arm sorted first and reports it
  as "the cheapest". An arbitrary choice presented as a measured one is exactly
  the failure this module exists to prevent. Make `cost` `Optional[float]` and
  require it.

### 2.4 Where the winner is recorded, and how a loser dies

**`experiments/decisions.json`** — machine-readable, written under the same lock
discipline as the ledger, one entry per decision:

```json
{"D1": {
  "spec_id": "T2.02", "verdict": "WINNER", "winner": "mlp_124k",
  "decided_at": "2026-08-09T07:00:00", "commit": "114e8f7", "plan_sha": "9f3c...",
  "metric": "return_at_matched_steps", "null": {"mean": 110.9, "std": 23.46},
  "arms": [
    {"name": "mlp_124k",    "mean": 530.2, "sigma_over_null": 7.11, "gate": true,
     "cost": 124707, "source": ["experiments/tests/t2_02_mlp_showdown.py"]},
    {"name": "transformer", "mean": 317.7, "sigma_over_null": 2.46, "gate": false,
     "cost": 57000000, "source": ["UnifiedBrain.py", "TrainingPipeline.py"]}],
  "archived_to": null, "supersedes": null, "reopened_by": null,
  "reopen_triggers": ["source file changed after decided_at"]
}}
```

**`docs/DECISIONS_RESOLVED.md` is rendered from it**, never appended to
directly. This matters: the current `_append_decision` appends a fresh section
on *every* run, so re-running a bakeoff silently produces two contradictory
sections in the same file, with no supersession and no way to tell which is
current. It is the same failure mode `CHECKLIST.md` was created to fix — a
hand-maintained status document drifting from the evidence. Fold the render into
`python -m experiments.run render`, next to the checklist.

**Losing code is archived, not deleted.** `archive/` already exists and already
holds `RobustTrainer.py`. The rule:

- On `WINNER`, the bakeoff moves each losing arm's `source` files it *solely*
  owns to `archive/bakeoff/<decision>/`, records `archived_to`, and leaves a
  one-line pointer where the file was.
- If a losing arm's source is shared (as `UnifiedBrain.py` is), the bakeoff
  **does not touch it**. It writes a deletion proposal to
  `docs/DECISIONS_NEEDED.md` with the numbers attached. SYSTEM.md already makes
  deleting components the owner's call, and T2.02's own history shows why: its
  literal kill criterion ("delete the transformer from the control path") was
  correctly judged too broad, because the trunk has jobs outside locomotion that
  the bakeoff did not measure. **A bakeoff may only kill what it measured.**
  That sentence belongs in the module docstring.

### 2.5 Re-opening a decision

Three triggers, one of them automatic:

1. **Source drift (automatic).** `audit.py` (§3) checks, for every decision, whether
   any file in any arm's `source` changed after `decided_at`'s commit. The
   computation is the one validated in §0 — `git log <commit>..HEAD -- <path>` —
   and it currently returns exactly one hit repo-wide, so the false-positive rate
   is tolerable. A hit marks the decision `STALE` in `OVERSIGHT.md` and adds the
   re-run to the builder's queue. It does **not** overturn the decision.
2. **New arm.** Anyone may add an arm to a settled question; doing so requires a
   new spec version whose `supersedes` names the old decision. The old entry
   stays in `decisions.json` forever.
3. **Gate change.** If `plan_sha` changes, the decision is void until re-run.
   This is the same mechanism as §3.2 and needs no extra code.

Re-opening never edits the old record. `decisions.json` is append-only in
spirit: `supersedes` / `superseded_by` chains, so "why did we choose the MLP in
August" is answerable in December.

### 2.6 Staged bakeoffs on a quota (the successive-halving adaptation)

`Plan.stages = (0.15, 1.0)` means: run every arm at 15% of the budget; drop any
arm whose **upper** bootstrap bound at that budget is still below the null
(i.e. it provably has not started learning); spend the rest on survivors. With
Kaggle's 30 h/week this roughly triples the number of decisions per reset.

Pre-register the elimination rule as a *validity* criterion, never a performance
one — RL learning curves cross, and irace/Hyperband's known weakness is
discarding slow starters. Eliminating "currently behind" would let the bakeoff
answer the question by choosing when to stop looking.

---

## 3. The overseer

`scripts/overseer.sh` + `overseer_prompt.md` exist as of `92931a6`, on
`37 */6 * * *`, offset from the builder's `:07`, taking no lock. The schedule and
the independence are right. Two things must change or the overseer is a
narrator rather than an auditor.

### 3.1 Split it: deterministic checks in code, judgement in the LLM

Every check in the current prompt is phrased as a task for the model —
*"For each PASS ... does its commit still exist in git?"* Under a 60-turn cap
with a 25-minute timeout, the checks that run are whichever ones the model gets
to. An audit whose coverage depends on the model's mood is not an audit, and
§1.2's lesson is that the LLM must never be the scorer.

**`experiments/audit.py`** — pure Python, no LLM, no network, runs in seconds,
emits JSON, exits non-zero on any `INTEGRITY` finding. The overseer session then
*reads the JSON and interprets it*. Each check below was prototyped against the
live repo during this design, with the result shown.

| # | Check | Computation | Severity | Live result |
|---|---|---|---|---|
| A1 | Ledger commit exists | `git cat-file -e <commit>^{commit}` per entry | INTEGRITY | 0 hits |
| A2 | PASS has an implementation | `_module_for(id) is not None` | INTEGRITY | 0 hits |
| A3 | **Pre-registration drift** | recompute `plan_sha` (§2.2); compare to the value stamped in the ledger entry | **INTEGRITY** | needs `plan_sha`; retroactive proxy below |
| A4 | Evidence staleness | commit that *recorded* the entry (`git log -S <ran_at> -- ledger.json`) vs later commits touching the test file | WARN | **1** (T1.07, benign) |
| A5 | Control declared but never run | `spec.control` set and `result.control_metrics == {}` | INTEGRITY | 0 |
| A6 | Control run but never declared | `control_metrics` non-empty and `spec.control is None` | WARN | several (ME.5, PG.3, T2.10, T2.12 …) — the Spec is out of date, not the test |
| A7 | PASS with no control at all | neither declared nor run | WARN | **6** (T0.01, T0.08, T0.10, T0.12, T1.03, T1.05) |
| A8 | Generated artifacts current | re-render `CHECKLIST.md` / `DECISIONS_RESOLVED.md` to a temp file; diff | WARN | **1 — T2.02 shows `[ ]` while a completed run sits in the ledger** |
| A9 | Spec pre-dates its test | first commit adding `Spec("X"` vs first commit adding the test file | WARN | **5 of 45** same-commit (T0.01, T0.02, T1.03, T1.11, T2.00) |
| A10 | Learning-claim seed floor | `spec.tier >= 2 and "learn" in hypothesis and len(seeds) < 3` | WARN | small |
| A11 | Builder liveness & yield | parse `ladder.log`: iterations/24 h, `rc` distribution, PASS delta, **PASS count decreases** | INTEGRITY on a decrease | **9 consecutive dead iterations today; one 42→41 regression on 08-08** |
| A12 | Compute honesty | `gpu_budget.json` delta vs ledger entries whose `hardware` names a GPU and whose `ran_at` falls in the window | WARN | 45.2 h charged |
| A13 | Arbiter changes | any commit touching `protocol.py`, `run.py`, `bakeoff.py`, `audit.py`, `ladder_loop.sh` — always reported, and INTEGRITY if the same commit did not also run `--gate` | **INTEGRITY** | see §3.5 |
| A14 | VOID treated as a verdict | `metrics.verdict` starts with `VOID` while `status != VOID` | **INTEGRITY** | **1 — T2.02** |
| A15 | Decision staleness | arm `source` changed after `decided_at` (§2.5) | WARN | n/a yet |
| A16 | Goal traceability | every spec's `id` prefix maps to a `MASTER_PLAN.md` phase, and every GOAL.md pillar has ≥1 spec; report pillars with **zero PASS** | WARN | curiosity (CU.*) and unison (UB.*) have zero passing specs |

A3 deserves the detail because it is the check the owner asked for —
*"thresholds weakened between commits"*. Reading `git log -p` in prose is
unreliable and cannot tell a loosening from a tightening. The mechanical version:

```python
def plan_fingerprint(spec, module_path) -> dict:
    """Everything a run's verdict depends on, canonicalised."""
    tree = ast.parse(Path(module_path).read_text())
    consts = {n.targets[0].id: n.value.value
              for n in tree.body
              if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id.isupper()
              and isinstance(n.value, ast.Constant)
              and isinstance(n.value.value, (int, float))}
    return {"spec": {k: getattr(spec, k) for k in
                     ("hypothesis", "falsified_by", "null_baseline",
                      "control", "seeds", "metric")},
            "constants": consts,
            "check_src": ast.unparse(next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "_check"))}
```

Stamp `sha256` of that at run time into `Result.plan_sha`. On audit, recompute
and compare. **Do not attempt to infer the direction of a change** — whether a
smaller number is looser depends on the comparison, and a heuristic that guesses
wrong in either direction is worse than none. The rule is simply: *the
fingerprint changed and the spec was not re-run* → INTEGRITY. Change plus re-run
in the same commit is normal, healthy work (that is exactly what "spec
strengthened to seeds=3 before the run" looks like, and it appears repeatedly in
the journal as good practice). Silence is the offence.

Retroactively, before `plan_sha` exists, A3 falls back to A4's file-level
staleness — which is what I ran in §0.

### 3.2 Make the overseer's restraint mechanical, not textual

The prompt says *"MAY NOT: implement or modify any spec, test, or model code"*
while the process runs `claude -p --dangerously-skip-permissions` with full
write access to the tree. The only thing between the auditor and the science is
a paragraph of English — and §1.2 is a peer-reviewed account of an automated
scientist editing its own execution constraints when they became inconvenient.

Two mechanisms, either sufficient, both cheap:

**Preferred — a read-only worktree.** The overseer never sees the live tree:

```bash
AUDIT=/tmp/jack-audit-$$
git -C "$REPO" worktree add --detach "$AUDIT" HEAD >/dev/null 2>&1
"$REPO"/../venv/bin/python "$AUDIT"/experiments/audit.py --json > "$AUDIT"/audit.json
# the agent runs with cwd=$AUDIT and writes only $AUDIT/OVERSIGHT.md
( cd "$AUDIT" && timeout 25m claude -p "$(cat "$REPO"/scripts/overseer_prompt.md)" ... )
cp "$AUDIT"/OVERSIGHT.md "$REPO"/docs/OVERSIGHT.md      # shell, not the agent
git -C "$REPO" add docs/OVERSIGHT.md && git -C "$REPO" commit -m "overseer: ..."
git -C "$REPO" worktree remove --force "$AUDIT"
```

Anything else the agent writes dies with the worktree. This also means the
overseer cannot accidentally collide with a running builder, and it removes the
need to trust the "takes no lock" claim.

**Fallback — enforce an allowlist after the fact.** If the worktree is
inconvenient, the shell checks what changed and reverts anything outside
`docs/OVERSIGHT.md`, `docs/DECISIONS_NEEDED.md`, `docs/LESSONS.md`:

```bash
CHANGED=$(git status --porcelain | awk '{print $2}')
for f in $CHANGED; do
  case "$f" in
    docs/OVERSIGHT.md|docs/DECISIONS_NEEDED.md|docs/LESSONS.md) ;;
    *) say "OVERSEER OVERREACH: reverting $f"; git checkout -- "$f" 2>/dev/null || rm -f "$f";;
  esac
done
```

Log every reversion. An overseer that had to be restrained is itself a finding.

### 3.3 What the overseer does with a finding

Strict, three-way, and it must never be able to fix science:

| Finding class | Action |
|---|---|
| Mechanical, unambiguous, non-scientific (`CHECKLIST.md` stale) | Note in `OVERSIGHT.md` → **FOR THE BUILDER**. Do not re-render it itself; the builder owns generated artifacts. |
| Integrity (fingerprint drift, VOID-as-verdict, PASS count regression, control never run) | `OVERSIGHT.md` verdict becomes `INTEGRITY RISK`, the specific spec is named, **and the audit exits non-zero so the wrapper can page.** Never edits the ledger, never re-runs the spec. |
| Requires an owner call (delete a component, spend money, resolve D1) | Append to `docs/DECISIONS_NEEDED.md` with the evidence table, exactly as the D1 entry already does. |
| A generalisable methodological lesson | Append to `docs/LESSONS.md` under §2's admission rule. |

One addition to the prompt's §8 ("the honest summary") — it is the best item in
the current prompt and should be sharpened with a number: **report the
first-attempt pass rate for Tiers 3–5 as they populate, against the Registered
Reports benchmark of 44%.** A ladder that never fails on first attempt is a
ladder whose specs were written to the implementation.

### 3.4 Schedule

- `experiments/audit.py`: every hour at `:52`, from cron directly — it costs
  seconds and no model credits, and it must keep running when credits are out
  (today's 9 dead builder iterations were invisible for 8 hours).
- The LLM overseer: `37 */6 * * *` as installed. Four sessions a day is right —
  it is a reviewer, not a builder, and its cost competes with the builder's for
  the same credit pool.
- On `INTEGRITY` from `audit.py`, the hourly cron writes the finding to the top
  of `docs/OVERSIGHT.md` immediately and **pauses the builder** (`touch
  .loop-paused`) for the classes where continuing does damage: fingerprint drift
  and PASS-count regression. A builder that keeps committing on top of a
  corrupted scoreboard makes the corruption harder to unwind.

### 3.5 The arbiter rule (A13)

The builder must be able to improve the machine — that is the owner's whole
directive. But an agent that can edit `protocol.py`, `run.py` and its own
timeouts can, in principle, do what the AI Scientist did. The workable
distinction:

> **Improving the machine is the job. Changing the arbiter is allowed, never
> silent, and never unaccompanied.** Any commit touching `protocol.py`,
> `run.py`, `bakeoff.py`, `audit.py`, or `ladder_loop.sh` must (a) say so in the
> first line of the commit message and (b) include a `--gate` run in the same
> commit. `audit.py` A13 checks both.

This is enforceable today with a git check and it costs nothing.

---

## 4. Agent memory

`docs/LESSONS.md` landed at `92931a6` with ten lessons in the right format
(evidence → rule) and the right framing (not a changelog, not instructions).
The design questions the owner posed are the ones still open: *what earns a
place, how does it stay bounded, how does it stay honest, how does it differ
from CLAUDE.md.*

### 4.1 The three-file split, stated once so no one has to re-derive it

| File | Kind of statement | Who changes it | Revisable by |
|---|---|---|---|
| `SYSTEM.md` | **Rules.** "Never weaken a threshold." Normative, not empirical. | the owner, or a bakeoff | an owner decision |
| `docs/LESSONS.md` | **Empirical generalisations discovered here.** "MJCF angles default to degrees." Could turn out to be wrong. | any agent, on evidence | new evidence |
| `docs/LOOP_JOURNAL.md` | **Chronology.** What happened, in order, including pre-registrations. Append-only. | every iteration | never — it is the record |

A CLAUDE.md-style instruction says *do this*. A lesson says *this was true here,
and here is the run that showed it*. The distinction is load-bearing because it
determines what happens on conflict: an instruction wins by authority, a lesson
loses to a better measurement.

### 4.2 Admission: the Voyager rule

A lesson enters only if all four hold:

1. **It cites its origin** — a spec ID, a commit hash, or a journal date. This
   is Voyager's "verify in the environment before adding to the library"
   (§1.3), and it is what stops the file filling with confident-sounding advice
   that nothing here ever demonstrated.
2. **It generalises** — it must plausibly bite in a *different file*. "PG.1's
   ramp was 0.87°" is a journal entry. "MJCF angles default to degrees; derive
   geometry from geometry, never from a quantity with the right units" is a
   lesson.
3. **It is not already there** — the index (§4.3) is grepped first.
4. **It cannot be made into a guard instead.** If it can, write the guard and
   record a one-line pointer. `LESSONS.md` already states this rule; §4.4 makes
   it measurable.

Rules 1 and 4 are the honesty mechanism. Rule 1 makes every claim checkable by
the overseer; rule 4 keeps the file from becoming a graveyard of things that
should have been code.

### 4.3 Bounding it: an index, a cap, and a retirement path

The file is pasted into a 120-turn session every hour. At 134 lines it is
already ~1.5 K tokens; at 40 lessons it would be ~6 K, and the reading cost is
paid 24 times a day whether or not any lesson is relevant.

- **Index at the top.** One line per lesson: `#12 — control fails too → bug is in
  shared scaffolding (PG.1)`. An agent reads the index always and the body only
  for the lessons that touch its task. This is the cheap, no-embedding version of
  Voyager's retrieval, and it is what makes the cost sublinear.
- **Hard cap: 40 lessons.** Past 40, a new lesson must either replace an existing
  one or be promoted to a guard. A cap that forces a choice is worth more than a
  cap that is politely ignored — the same logic as ME.4's bounded store, whose
  whole finding was that eviction policy is what makes a bounded memory useful.
- **Retirement.** When a lesson becomes a guard (a spec, an assertion, a check),
  its body collapses to one index line: `#3 — RETIRED, now enforced by T2.00 /
  audit.py A3`. The lesson has not been forgotten; it has been *made
  unforgettable*, which is strictly better.
- **The overseer merges near-duplicates** quarterly-equivalent (every ~50
  iterations). It may edit `LESSONS.md` — that is not science.

### 4.4 The seeds the owner named, in the required form

Four were specified and all four already meet the admission rule; the fourth
should be recorded as **retired into a guard**, not as advice:

- *"A control that passes while the experiment fails localises the bug"* — in the
  file. Sharpen the converse, which is the actually surprising half: **both
  failing means the bug is in the shared scaffolding** (PG.1's degrees).
- *"MJCF angles default to DEGREES"* — belongs as the concrete evidence under a
  general rule: **units and conventions of an external format are assumptions;
  assert them in the fixture, do not infer them from a passing run.**
- *"pg_loss is ~0 by construction so loss RATIOS are not a domination metric —
  measure per-term gradient norms"* — in the file as "measure the quantity you
  are claiming". Keep the *diagnostic* form too: **if a denominator can go to
  zero for structural reasons, it is not a ratio you can threshold.**
- *"run_spec calls `_experiment` once PER SEED, so guard GPU submissions with a
  module cache"* — this one should **not** live only as advice. It cost 5.5 GPU
  hours once and nothing prevents it recurring. Guard: `gpu.submit()` records
  `(spec_id, job_sha)` and raises on a second submission of the same job within
  one process, with a message naming the module-cache pattern. Then the lesson
  retires to one index line. **That transition — advice becoming a guard — is the
  system getting better at building Jack, and it is measurable (§5.4).**

### 4.5 Reading order

`ladder_prompt.md` now says GOAL → SYSTEM → LESSONS → OVERSIGHT. Correct, with
one refinement: **read the LESSONS *index* first and `OVERSIGHT.md`'s FOR THE
BUILDER section second**, because the latter is the only input that is specific
to *this* iteration. Everything else is standing context.

---

## 5. The self-improvement loop

Right now nothing measures the loop. The PASS count is the only visible number,
and it is the worst available proxy: it goes up when a cheap spec lands and
stays flat during the iteration that finds a real bug in `Persistence.py`.
Optimising it is Goodhart in one step.

### 5.1 `experiments/loop_metrics.py` — put the loop on its own scoreboard

Derived from git + ledger history (§2.2) + `ladder.log`. No new instrumentation.

| Metric | Definition | Why |
|---|---|---|
| **Yield** | ledger status *changes* per iteration (not PASS count) | a FAIL recorded is progress; a PASS re-run for the gate is not |
| **Dead-iteration rate** | iterations with no commit and no ledger change | today: 9/9 since 02:07 — invisible for 8 hours |
| **First-attempt pass rate, by tier** | from `Result.history` | §1.6 calibration; the anti-HARKing number |
| **Rework rate** | specs whose status changed after a PASS ÷ specs passed | measures how often we believed something too early |
| **Bug-find rate** | iterations whose commit touched non-test source (`UnifiedBrain.py`, `Persistence.py`, `VirtualWorld.py` …) | **the loop's real output.** T1.03 found the action-path defect; T6.03 found the save that silently dropped all emotional state. Neither shows up in a PASS count |
| **Compute efficiency** | GPU-hours ÷ ledger status changes | 45.2 h to date |
| **Guard promotions** | lessons retired into guards per 50 iterations | §4.4 — the only direct measure of "the machine got better" |
| **Time-to-detection** | for each integrity finding, `found_at − introduced_at` from git | measures the overseer, not the builder |

Rendered into `docs/OVERSIGHT.md` by `audit.py`. No LLM involved.

### 5.2 The retraction register

Add to the ledger entry: when a spec is re-run and its status changes *away
from* PASS, or when its design is revised, record `retraction_reason` from a
closed vocabulary:

```
METRIC_ARTIFACT    the metric moved for a reason unrelated to the effect   (T2.00 v1, T1.02 v1)
DEGENERATE_DATA    n too small, fabricated, or the task unlearnable        (T1.02 v2, T1.13 v1)
UNDERPOWERED       the effect was inside seed noise                        (T2.01 v3 watch-item)
SCAFFOLDING_BUG    the harness was wrong, not the mechanism                (PG.1 degrees)
ENVIRONMENT        infrastructure, quota, backend                          (T1.02 ERROR)
SUPERSEDED         a better design replaced it
```

A closed vocabulary is the point: free-text reasons cannot be counted. **When a
category recurs three times, it earns a permanent guard**, and the guard is the
deliverable of that iteration. `METRIC_ARTIFACT` has already occurred at least
twice (T2.00's loss ratio, T1.02's training-fit metric) — one more and the
system owes itself a check, e.g. a spec that asserts every gated metric reads
its null value when the mechanism is disabled.

### 5.3 Put the system on the ladder: the `SY.*` family

The system currently claims — in `SYSTEM.md`, in the overseer prompt — that it
catches threshold weakening, dead loops, and false verdicts. **Those are exactly
the kind of unverified capability claims the ladder exists to forbid.** A README
saying "Working" for the overseer is the same disease in a new location.

Four specs, all CPU-cheap, all implementable in an afternoon, each with a
control that must fail:

- **SY.1 — the audit catches a planted violation.** In a scratch worktree, plant
  each of: a loosened constant in a passing test, a deleted control, a ledger
  entry with a nonexistent commit, a VOID-as-FAIL. Assert `audit.py` flags all
  four. *Control: an unmodified worktree must produce zero findings* — an auditor
  that always finds something is as useless as one that never does.
- **SY.2 — the bakeoff refuses to arbitrate between non-learners.** Synthetic
  arms with known means: two arms below the gate → VOID; one above one below →
  VOID; two above, separated → WINNER; two above, overlapping → TIE. *Control:
  an arm secretly reading the null's own output must not win.*
- **SY.3 — the ledger cannot be written except by the runner.** Hand-edit
  `ledger.json` to mark a spec PASS; assert the audit detects it (no matching
  `ran_at` in git history, missing `plan_sha`). This is the deepest assumption in
  the project and it has never been tested.
- **SY.4 — the loop notices it is dead.** Simulate credit exhaustion and a
  wedged lock; assert `audit.py` reports it within one hour. Today's 8-hour
  blindness is the motivating failure.

These are the specs that make the owner's directive falsifiable. Until SY.1
passes, "the overseer catches weakened thresholds" is a claim without evidence,
and the system says such claims are worth nothing.

### 5.4 The one-line theory of improvement

> **A failure becomes a lesson; a recurring lesson becomes a guard; a guard
> becomes a spec that must keep passing.** Yield is measured in guards, not in
> ticks.

That ratchet is the whole self-improvement mechanism, and unlike ADAS or DSPy it
needs no meta-optimiser, no eval budget, and no statistical power — which is
precisely why it works on a free-tier box at one iteration per hour.

---

## 6. Recommended changes, in priority order

Each entry: the change, why it matters, and the specific failure it prevents.
Nothing here requires GPU quota. P0 is a few hours of CPU-free editing.

### P0 — today, ~2 hours, no compute

**P0-1. `Status.VOID` + `check` may return a `Status`** — `experiments/protocol.py`
(~10 lines), plus `MARK` in `run.py:53` and `box` in `cmd_render` (2 entries each,
or they `KeyError`), plus `t2_02_mlp_showdown.py`'s three VOID branches
`return Status.VOID`, then re-record T2.02.
*Why:* the system's best idea currently has nowhere to live. *Prevents:* a run
that explicitly refused to arbitrate being read as its kill criterion firing —
which is the live state of T2.02 today, on the spec that gates the entire
locomotion branch and whose `kills` field says "The transformer policy."

**P0-2. `bakeoff()` runs through `run_spec`** — `experiments/bakeoff.py`.
*Why:* env stamp, dependency gating, the ledger lock, subprocess isolation and
budget-scaled timeouts all already exist and were each paid for with an
incident. *Prevents:* decision records with `commit=""`, unattributable to any
tree — which silently breaks audit checks A1 and A15 and the re-open trigger,
i.e. the bakeoff's own guarantees.

**P0-3. `Result.history` + `attempt` on overwrite** — `protocol.py`,
`Ledger.record`, ~5 lines inside the existing lock.
*Why:* 132 runs have happened and the ledger remembers 45. *Prevents:*
SYSTEM.md law 4 ("record the failure in the ledger's history") being
unenforceable, and makes every metric in §5.1 a one-line query instead of a
five-minute git replay.

**P0-4. Require `cost` to break a TIE; add the control-validity rule** —
`bakeoff.py`, ~10 lines.
*Why:* `cost` defaults to `0.0`, so today a TIE silently returns the
first-sorted arm and calls it "the cheapest". *Prevents:* an arbitrary choice
presented as a measured one — the exact failure the module exists to prevent —
and restores T2.02's untrained-arms check, which the generalisation dropped.

**P0-5. `render` also regenerates `DECISIONS_RESOLVED.md` from
`decisions.json`** — `run.py:cmd_render`, and delete `_append_decision`'s direct
markdown write.
*Why:* the current append-on-every-run produces contradictory sections with no
supersession. *Prevents:* a hand-drifting status document — the precise disease
`CHECKLIST.md` was built to cure. (And re-render `CHECKLIST.md`: it currently
shows T2.02 as not-run.)

### P1 — this week

**P1-6. `experiments/audit.py` with checks A1–A16** — new file, ~250 lines, no
LLM, no network, exits non-zero on INTEGRITY. Cron at `:52` hourly.
*Why:* eight of the sixteen checks were prototyped during this design and four
found live issues. *Prevents:* audit coverage depending on whether a model had
turns left — and keeps auditing when credits run out, which is when the builder
was blind for 8 hours today.

**P1-7. Overseer containment: read-only worktree, or a shell-enforced write
allowlist** — `scripts/overseer.sh`, ~10 lines.
*Why:* the restraint is currently a paragraph of English in front of
`--dangerously-skip-permissions`. *Prevents:* the documented AI-Scientist
failure — an automated scientist editing its own constraints when they became
inconvenient (arXiv:2408.06292 §safe code execution).

**P1-8. `plan_sha` fingerprinting (A3) + the arbiter rule (A13)** —
`protocol.py` stamps, `audit.py` checks, `SYSTEM.md` states the rule.
*Why:* "never weaken a threshold" is currently enforced by an agent reading its
own diff. *Prevents:* silent loosening — named in the owner's brief as the
single most serious failure available to this system — and silent edits to the
arbiter itself.

**P1-9. `Spec.arms` and `Spec.decides`; bakeoff refuses a mismatched arm set** —
`protocol.py` (2 optional fields), `bakeoff.py` (5 lines).
*Why:* "may not drop an arm that embarrasses it" is a docstring. *Prevents:* the
easiest way to fake a decision, and links every ledger entry to the question it
settles.

**P1-10. `LESSONS.md`: index at the top, 40-lesson cap, origin citation
required, retirement-on-guard** — plus retire the `_submit`-per-seed lesson into
a real guard in `gpu.py`.
*Why:* the file is read 24 times a day and only grows. *Prevents:* the memory
becoming either unread (too long) or unfalsifiable (uncited advice) — the
Generative-Agents reflection failure mode.

### P2 — next, once P0/P1 have run for a week

**P2-11. `experiments/loop_metrics.py`, rendered into `OVERSIGHT.md`** — yield,
dead-iteration rate, first-attempt pass rate by tier, rework, **bug-find rate**,
GPU-hours per status change, guard promotions.
*Why:* the PASS count is the only visible signal and it is the wrong one.
*Prevents:* optimising the scoreboard instead of Jack — and makes visible the
iterations whose real output was a fixed defect in `Persistence.py` rather than
a tick.

**P2-12. The `SY.*` spec family (SY.1–SY.4)** — the system audited by its own
ladder, each with a control that must fail.
*Why:* the overseer's capabilities are currently claimed in prose. *Prevents:*
the original disease reappearing one level up — a status document asserting the
auditor works, with no test that could have failed.

**P2-13. Retraction register with a closed vocabulary; three occurrences of a
category earn a guard** — `protocol.py` field + `audit.py` tally.
*Why:* `METRIC_ARTIFACT` has already happened twice. *Prevents:* the same class
of mistake being re-learned, re-journaled and re-forgotten.

**P2-14. `experiments/stats.py`: bootstrap CI + IQM reported alongside every
seed-aggregated metric; gates unchanged** — Agarwal et al. 2021.
*Why:* `mean ± std` over 3 seeds is not a significance claim and the docs should
not read as though it is. *Prevents:* over-reading a 3-sigma margin. Reporting
more information next to a pre-registered gate is not weakening it; **changing
the gate would be, so do not.**

**P2-15. Staged bakeoffs (`Plan.stages`) with a validity-only elimination rule**
— successive halving / AdaStop adapted to 30 GPU-h per week.
*Why:* quota is the binding constraint on how many questions get answered.
*Prevents:* a whole Sunday reset spent proving that an arm which never learned
still has not learned. Pre-register the elimination rule as "cannot clear the
null", never "is currently behind", or the bakeoff decides by choosing when to
stop looking.

**P2-16. A held-out playground variant and memory-life seed no spec may design
against, exercised only in `--gate`** — the Dwork *reusable holdout* hygiene, in
its cheap form.
*Why:* `PG.*` fixtures are reused adaptively by dozens of specs. *Prevents:*
fixture overfitting — the ladder passing because the world was shaped around it.

---

## 7. What I deliberately did not propose

- **An LLM judge, anywhere.** §1.2. The evaluator is code or it is nothing.
- **Population-based program evolution** (FunSearch/AlphaEvolve islands, ADAS
  meta-search). Wrong compute regime by three orders of magnitude (§1.1).
- **Automatic prompt optimisation** (DSPy/GEPA) for `ladder_prompt.md`. No power
  at ~24 noisy rollouts a day; would produce confident nonsense (§1.4).
- **Online FDR machinery** (LORD/SAFFRON). Our sigmas are not p-values;
  implementing it would be a costume. Counting attempts is the part that helps.
- **Raising the seed floor above 3.** Correct in theory (Colas et al.),
  unaffordable here. Report uncertainty better instead (P2-14).
- **A new orchestration layer, queue, or database.** The flock, the JSON ledger
  and the cron work, and they were hardened by real incidents. Every change above
  is additive.
- **Letting the overseer fix anything.** It reports, escalates, and — for two
  named integrity classes — pauses the builder. It never touches science.

---

## Sources

**Automated discovery & self-improving agents**
- Lu, Lu, Lange, Foerster, Clune & Ha. *The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery.* [arXiv:2408.06292](https://arxiv.org/abs/2408.06292) (2024). Self-modified its timeout / relaunched itself; LLM reviewer.
- Beel et al. *Evaluating Sakana's AI Scientist…* [arXiv:2502.14297](https://arxiv.org/html/2502.14297v2) (2025). 42% of proposed experiments failed on coding errors.
- *Failure modes of AI-scientist systems* — [arXiv:2509.08713](https://arxiv.org/abs/2509.08713v2): benchmark misselection, leakage, metric misuse, post-hoc selection bias.
- Novikov et al. *AlphaEvolve: A coding agent for scientific and algorithmic discovery.* [arXiv:2506.13131](https://arxiv.org/abs/2506.13131) (2025); [DeepMind blog](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/).
- Romera-Paredes et al. *Mathematical discoveries from program search with LLMs (FunSearch).* *Nature* 625 (2024); [code](https://github.com/google-deepmind/funsearch). Islands, program database, automated evaluator; cap set of 512 in dim 8.
- Hu, Lu & Clune. *Automated Design of Agentic Systems.* [arXiv:2408.08435](https://arxiv.org/abs/2408.08435) (2024).
- Wang et al. *Voyager: An Open-Ended Embodied Agent with LLMs.* [arXiv:2305.16291](https://arxiv.org/abs/2305.16291) (2023). Skill library admitted only after verification.
- Park et al. *Generative Agents.* arXiv:2304.03442 (2023). Importance/recency/relevance retrieval.
- Agrawal et al. *GEPA: Reflective Prompt Evolution Can Outperform RL.* [arXiv:2507.19457](https://arxiv.org/abs/2507.19457) (ICLR 2026 oral). Pareto front over per-instance winners.
- Huang et al. *Large Language Models Cannot Self-Correct Reasoning Yet.* [ICLR 2024](https://arxiv.org/abs/2310.01798) — intrinsic self-correction usually *degrades* performance. The argument for an external, code-based arbiter.
- Madaan et al. *Self-Refine* (2023); Shinn et al. *Reflexion* (2023) — the optimistic side, already encoded as ladder spec CU.7.

**Selection, racing, sequential testing**
- Mathieu, Della Vecchia, Shilova, Centa, Kohler, Maillard & Preux. *AdaStop: adaptive statistical testing for sound comparisons of Deep RL agents.* *TMLR* 2024, [arXiv:2306.10882](https://arxiv.org/abs/2306.10882).
- López-Ibáñez et al. *The irace package: iterated racing for automatic algorithm configuration.* *ORP* 3 (2016). F-Race with Friedman-test elimination.
- Li, Jamieson, DeSalvo, Rostamizadeh & Talwalkar. *Hyperband.* *JMLR* 18 (2018); Jamieson & Talwalkar, successive halving (AISTATS 2016).
- Foster & Stine, alpha-investing (*JRSS-B* 2008); Ramdas et al., LORD++/SAFFRON online FDR ([arXiv:2110.08161](https://arxiv.org/pdf/2110.08161), [survey](https://arxiv.org/pdf/2208.11418)).

**Reproducibility, pre-registration, statistics**
- Simmons, Nelson & Simonsohn. *False-Positive Psychology.* *Psychological Science* 22(11), 2011.
- Gelman & Loken. *The garden of forking paths.* (2013/2014).
- Kerr. *HARKing.* *PSPR* 2(3), 1998.
- Scheel, Schijen & Lakens. *An Excess of Positive Results.* [*AMPPS* 4(2), 2021](https://journals.sagepub.com/doi/10.1177/25152459211007467). **44% vs 96%.**
- Nosek et al. *The preregistration revolution.* *PNAS* 115(11), 2018; *Preregistration is not a straitjacket* (2019); Claesen et al. on unreported deviations (2021).
- Benjamini & Hochberg. *Controlling the FDR.* *JRSS-B* 57(1), 1995.
- Dwork, Feldman, Hardt, Pitassi, Reingold & Roth. *The reusable holdout.* [*Science* 349(6248), 2015](https://www.science.org/doi/10.1126/science.aaa9375).
- Henderson et al. *Deep Reinforcement Learning That Matters.* AAAI 2018.
- Colas, Sigaud & Oudeyer. *How Many Random Seeds?* arXiv:1806.08295 (2018).
- Agarwal, Schwarzer, Castro, Courville & Bellemare. *Deep RL at the Edge of the Statistical Precipice.* [NeurIPS 2021](https://arxiv.org/abs/2108.13264) (outstanding paper); [`rliable`](https://github.com/google-research/rliable).
- Dehghani et al. *The Benchmark Lottery.* arXiv:2107.07002 (2021).
- Pineau et al. *Improving Reproducibility in ML Research.* *JMLR* 22 (2021).

**This repo** — `GOAL.md`; `docs/MASTER_PLAN.md`; `docs/LOOP_JOURNAL.md` (all of
it); `experiments/protocol.py`, `run.py`, `registry.py`, `gpu.py`, `bakeoff.py`;
`experiments/tests/t2_02_mlp_showdown.py` (where the learning gate was invented);
`SYSTEM.md`; `docs/LESSONS.md`; `scripts/ladder_loop.sh`, `overseer.sh`; and 156
commits of git history, which is the most honest document here.
