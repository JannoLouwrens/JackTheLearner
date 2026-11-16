# Brain A — a Dreamer-style world model: source pin (J5 step 1)

PLAN part "His brain", stage 1, step 2 (combine). Prior: `docs/research/PUBLISHED_PRIORS.md` §2, §9 row 7.
**Code only, never weights.** Nothing here has been run yet; no speed is known (§2: "measured nowhere").
Brain A is a candidate for one of his parts, so every line of it learns from zero.

## The pins (checked against GitHub and PyPI on 2026-11-16)

| role | what | pin | licence |
|---|---|---|---|
| reference (the definition of the agent) | `danijar/dreamerv3` | commit `e01491fad6434b2245a3b8ca201dd7faedcc458c` (`e01491f`, 2026-10-04) — its `configs.yaml`, `agent.py`, `rssm.py`, `scores/*.json.gz` | MIT |
| runner (what we execute) | `sheeprl` (Eclectic-Sheep) | PyPI `sheeprl==0.8.5` (uploaded 2026-10-27, Python >=3.10,<3.14); git tag `v0.8.5` = `22b4f164ed2a9b093100d95fc2929e21458e949b` | Apache-2.0 |
| runner algorithm | `algo=dreamer_v3_5` with `algo=dreamer_v3_5_1M` (`size1m`) or `dreamer_v3_5_12M` | files `sheeprl/configs/algo/dreamer_v3_5{,_1M,_12M}.yaml` at `v0.8.5` | — |

`dreamer_v3_5_1M.yaml` at `v0.8.5` reads, in full beyond its parent: `dense_units: 64`,
`cnn_channels_multiplier: 4`, `world_model.discrete_size: 4`,
`recurrent_model.recurrent_state_size: 512`, `recurrent_model.hidden_size: 64` — its header
says "the 1M model of the official implementation (`size1m`)". Not yet diffed line by line
against `configs.yaml` `size1m` at `e01491f`: that diff is owed before any known-answer run.

## Why these and not the others (§2)

- **Not sheeprl's older `dreamer_v3`, and not `dreamerv3-torch`:** both use the v1 actor for
  continuous actions (backprop through the dynamics). It cannot learn from termination —
  InvertedPendulum stayed at return 4 — and Jack's lives end in death. Only the v2 REINFORCE
  actor (`dreamer_v3_5`) learns from it.
- `dreamer_v3_5` is the only maintained port of the v2 agent, runs on a CPU by default, and
  resumes with its replay (`buffer.checkpoint=True`, `checkpoint.resume_from`) — needed for
  12-hour kernel sessions. It is weeks old and has **no published scores**: it earns trust
  only through §2's known-answer rows and the step-0 check, never by its README.
- Ports have had silent bugs (terminated/truncated confused, 1.6% of replayed actions
  corrupted, resume regressions, a README timing from another agent). Each is a check owed
  before A is believed.

## Ours beside theirs (step 4 of J5)

None changed yet. Settings that WILL differ from a published recipe, listed now so that
nobody discovers them later:

| setting | published (paper, DMC proprio, v1 size S) | sheeprl `v0.8.5` example | ours (planned) | why |
|---|---|---|---|---|
| model size | S (v1, 2023 runs) | 12M (`dmc_walker_walk`) | `size1m` first | free T4 budget; **no published score exists for `size1m`** |
| train ratio | 512 | 512 (`replay_ratio: 0.5` = 512/(16×64)); the file notes the official config has 256 | as high as the T4 sustains, logged | §2: more is better up to 1024 |
| action repeat | 2 | 2 | 1 in his world (DMC rows: 2) | a decision is already 0.2 s |
| environments | 4 | 16 | 1 in his world (DMC rows: as published) | — |
| agent version | v1 (2023 score files) | v2 | v2 | REINFORCE actor needed for death; whether the v2 tables are the 2023 runs is unresolved (§2 "Not verified") |

## What comes next in J5

2. Speed, unmodified runner, smallest preset, DMC cartpole-swingup: decisions/s, achieved
   train ratio, wall-clock per 100k decisions — on a T4 (Colab now; Kaggle GPU opens 11-21)
   and on one Kaggle CPU kernel. Arithmetic on record (±2×): T4 `size1m` 20–50 gradient
   steps/s; 4-core CPU 0.5–2.
3. T4-hours for §2's first three known-answer rows (estimate 25–60) and for one 1M-decision life.
4. Fill the table above with what actually changed. Do not wire A to his world in J5.
