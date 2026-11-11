# Bottleneck audit, 2026-11-11 — what three months actually did

Read-only audit of the whole project while every organ was paused (2026-10-26 →
11-11). Every number below was computed from `experiments/ledger.json`, `git
log`, `/data/jack-logs/*.log`, `/data/jack-logs/usage_ledger.jsonl` and
`experiments/gpu_budget.json` on 2026-11-11. It is the evidence behind the
reorganisation of that day (managers on Fable, workers on Opus, lanes, bounded
orders) and behind rules 19–24 of `docs/LESSONS_CORE.md`.

## 1. The trajectory

| | |
|---|---|
| registry | 261 specs — 108 PASS, 37 FAIL, 19 VOID, 1 BLOCKED, **96 never run** |
| first-ever PASSes by week | W32 **50**, W33 **26**, W34 3, W35 9, W36 14, W37 2, W38 1, W39 6, W40 0, W41 1, W42 0, W43 6 |
| | **76 of 118 first passes (64%) landed in the first two weeks.** ~2 a week since early September |
| passed once, not passing now | 10 — incl. `T1.08` (strengthened, now blocks 45), `T2.10`, `T0.13`, `T0.07` |
| iterations | 760 completed, median 11 min, 169 h of model time |
| iterations that raised the pass count | Aug **22%** (56/260) → Sep **5%** (21/421) → Oct **4%** (3/79) |
| rework | 82% of specs with a ledger row have `attempt > 1`; 782 attempts over 156 specs |
| runnable fresh work | **zero** — `run next`: 57 runnable, 0 fresh, 42 already carry a verdict, 15 held |

## 2. Where the hours went

- **Hourly slots, Aug 4 → Oct 26:** 769 ran; **1,113 were lost to the usage
  budget** (509 at the 90% stop, 604 paced). 90 more hit model-specific limits.
- **The weekly pool (Sep 6 → Oct 26):** builder 253 points, desks 97, everything
  else on the account ~510. The builder got roughly **30%** of what was spent.
  It ran on `fable`, whose own meter hit 100% while the shared one had room.
- **Commits by kind:**

  | | a run landed | spec written | guards/governance | audit/review/steering | journal |
  |---|---|---|---|---|---|
  | Aug | 23% | 10% | 9% | 26% | 8% |
  | Sep | 16% | 4% | 9% | 44% | 16% |
  | Oct | **5%** | 4% | 22% | **49%** | 7% |

- **Code:** governance tooling ≈ 20,000 lines (`run.py` 6,239, `coverage.py`
  4,733, `protocol.py` 4,488, `decisions.py` 2,457, `champions.py` 2,052). The
  creature on the live path ≈ 1,600 lines (`cores.py` 570, `survival.py` 458,
  `w0.py` 598).
- **Reading load:** `ladder_prompt.md` 16 KB → 116 KB; `LESSONS.md` 1.4 MB;
  `LOOP_JOURNAL.md` 2.4 MB; `DECISIONS_NEEDED.md` 750 KB.
- **Desks:** 613 commits mention an audit. The Review died on its clock in 40 of
  81 sittings. Its queue took in 4.14 rows a cycle and disposed 2.86.

## 3. Compute — the mismatch

- **Free GPU, by ISO week (Kaggle h of 30):** W32 21.1, W33 7.6, W34 1.6, W35
  18.9, W36 17.7, then W37 1.4, W38 0.9, W39 1.1, W40 2.0, W41 0, W42 1.1, W43
  1.8. **~9 of 210 free hours used in the last seven weeks.**
- **The science that works is CPU-bound.** Survival runs are MuJoCo-in-the-loop
  on 4 shared ARM cores: `LC.03` cost ~190 core-hours; `LC.07` (the 10× scale
  test of the one learner) is priced at ~526 wall-hours and marked
  venue-unaffordable. `LF.02` (save and resume a life) passes, which is the
  piece multi-session runs need.

## 4. Why the ladder stopped — five structural causes

1. **Failures are recorded, not repaired.** Every terminal blocker's
   implementation was untouched for 22–89 days (`T4.02` 89, `T2.07` 82, `W.1`
   and `W.2` 73, `HR.5` 69, `UB.10` 59, `LT.02` 45, `T1.08` and `NE.01` 22).
   The owner asked on day two, *"but why so little changes to jack??"*
2. **Strengthening turns certificates into roadblocks.** `T1.08` passed twice in
   August, gained a 7% held-out-CV clause, and has blocked 45 specs since
   Sep 13 — including every legacy creature gate. Its CV went 40% → 17% with
   one LR-schedule fix; nobody has tried the cheap next step (more seeds on an
   idle GPU). A 3-seed CV cannot certify a 7% bar in any case.
3. **Rigor eats answers.** `D1.0` ran 17.6 GPU-hours and measured the control
   question: small plastic head **506** (paired t = 20.4 vs its own untrained
   twin), MLP 415, the 57M-trunk arms **344–350**, SB3 reference 518. Recorded
   VOID because untrained twins beat the *random* null at 3.9σ — the null was
   wrong, not the data. `LC.03` found the one learner (`wm-latent`, 4.65σ,
   +92 s across lives) and was VOID for having only one.
4. **Specs are long conjunctions.** `NE.01` fails on 2 of ~12 clauses while its
   central claim holds (the forager survives 18,000 decisions, random dies).
   `XL.01` fails per-seed (1 of 3) while the pooled effect is real (food in
   19.9 s with memory vs 27.3 s wiped). `T0.13` meanwhile counts 13 disarmed
   and 72 inert gate keys in passing specs.
5. **Two Jacks were never reconciled.** Stack B — `UnifiedBrain.py`, the 57M
   transformer, 51 test files — is what the T-family and the Tier-6 gates
   (`T6.01` "companion session") are written against. Stack A — `cores.py` /
   `survival.py` / `w0.py`, 29 test files — is where learning was actually
   demonstrated. Every control measurement since August says the big trunk is
   the worse controller. The creature gates sit on the stack that is losing.

## 5. What is real

- `wm-latent` learns to survive in W0 and carries the gain across deaths.
- One-trial taste aversion (`TA.02`); trust by track record (`LG.02`); he knows
  what his LLM cannot (`LG.00`).
- Sight + hearing fused where neither alone works (`UB.9`, 0.993 vs 0.5).
- Memory that makes the next episode 22× faster (`T2.20`); a life that can be
  saved and resumed (`LF.02`); felt cold, damage, falling.
- A small plastic control head learns Humanoid locomotion to the SB3 reference.
- **W0 is too shallow** — nine instruments agree; lookahead does not pay in it
  (`DP.05`) and a ridge regression beats the world model (`T2.05`). The next
  world (W1) is half built: `W1.01`, `W1.02` PASS; `W1.00` FAIL; `W1.04` VOID.
