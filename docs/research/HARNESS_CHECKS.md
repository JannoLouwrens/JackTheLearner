# HARNESS_CHECKS — the comparison harness checks itself (order J4)

Source: `docs/research/PUBLISHED_PRIORS.md` §8 "Reproduce first"; outcome rule = ruling 19.
Code: `experiments/harness/__init__.py`. Tests: `python -m tests.test_harness` (no pytest in the venv).

| # | check | state | evidence |
|---|---|---|---|
| 1 | same-against-same ≥ 1,000 reps, false alarms ≤ stated rate | RUNNING 11-16 | rule committed first (`80f63594`, `experiments/harness/same_vs_same.py` docstring): `compare()` = two looks 10/20 at 0.025 + futility, n_perm 2000; S1 mixed (34% censored), S2 ties (78% censored), 1,000 reps each; PASS iff ≤ 0.050 each; control 0.05/look must exceed 0.050 else VOID. Logs `/data/jack-logs/harness/svs_S{1,2}.jsonl`. |
| 2 | handicapped copy (lr 0) detected at planned power | OWED (needs B, J3) | — |
| 3 | floor and ceiling calibration arms | OWED (J4) | — |
| 4 | same seed twice → identical logs | OWED (J3) | — |
| 5 | run k sees same world in every arm | OWED (J2/J3 wiring) | — |
| 6 | resume across kernel boundary = uninterrupted | OWED (J3/J5) | — |
| 7 | every decision counted incl. warm-up/prefill | OWED (J3/J5) | — |
| 8 | reward-forever 1/(1−γ); death no bootstrap | OWED (J3) | — |
| 9 | probes on/off leave logs identical | OWED (J3) | — |
| 10 | outcome code right on synthetic logs, edge cases | PASS 11-16 | 10 tests: first good window, exact θ, just-below, window mean, < W lives, crossing at T vs past T, crash/NaN censored at T, no reach-back across a crash, W=1, bad args. Mutation (`>=`→`>`, `cum>T`→`>=`) turns 4 red. |
| 11 | analysis script frozen and hashed | OWED (J4) — `analysis_hash()` exists, not yet frozen in a pre-registration | — |
| 12 | 4/15 vs 0/15 → Fisher 0.100, Boschloo 0.043 | PASS 11-16 | SciPy 1.13.1: Fisher 0.0996, Boschloo 0.0433; arm order symmetric |
| 13 | runs analysed = runs launched | PASS 11-16 | `check_runs_accounted(launched_manifest, analysed)`: balanced books pass; dropped run, never-launched run, duplicate seed, missing arm each raise. Mutation (check made a no-op) turns the test red. |

Reading of ruling 19 that the record does not spell out: a crashed or NaN run never crossed and is
censored at T (stays in the risk set to the end — counts against its arm).
