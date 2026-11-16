# Brain A — a Dreamer-style world model: source pin (J5 step 1)

PLAN part "His brain", stage 1, step 2 (combine). Prior: `docs/research/PUBLISHED_PRIORS.md` §2, §9 row 7.
**Code only, never weights.** Run so far: a speed pilot only (below); no score has been read from it.
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
says "the 1M model of the official implementation (`size1m`)".

## Line diff: sheeprl `dreamer_v3_5_1M` vs official `size1m` (done 2026-11-16)

Left: `dreamerv3/configs.yaml` at `e01491f` (`defaults` + `size1m` + `dmc_proprio`), fetched from
GitHub. Right: the configuration sheeprl 0.8.5 actually resolved in our box smoke
(`<log_dir>/config.yaml`), not the yaml read by eye.

**Identical (every agent value):** size — deter 512, hidden 64, classes 4, units 64, depth 4;
stoch 32; 8 GRU blocks; dyn/img/obs layers 1/2/1; latent unimix 0.01; free nats 1.0; encoder and
decoder 3 layers, kernel 5, mults [2,3,4,4], symlog, bspace 8; reward and continue heads 1 layer;
255 bins; actor and critic 3 layers, min/max std 0.1/1.0, actor outscale 0.01, critic outscale 0;
loss scales rec 1, rew 1, con 1, dyn 1, rep 0.1, policy 1, value 1, repval 0.3; LaProp lr 4e-5,
betas 0.9/0.999, eps 1e-20, AGC 0.3, warm-up 1000, no weight decay; imagination 15; horizon 333
(gamma 0.996997); lambda 0.95; entropy 3e-4; slow critic rate 0.02 every 1, slowreg 1; return
normalisation percentiles 5/95, rate 0.01, limit 1; `contdisc`, `reward_grad`, `repval_loss`,
`repval_grad` on; batch 16 x 64; replay context 1; online replay queue on.

**Different — these are what a known-answer run must set or disclose:**

| setting | official `dmc_proprio` at `e01491f` | sheeprl 0.8.5 (shipped DMC example + `_1M`) | note |
|---|---|---|---|
| train ratio | **1024** | 512 (`replay_ratio: 0.5`) | the example file's comment says "official has 256" — that is `dmc_vision`; proprio is 1024 |
| action repeat | **1** (`env.dmc.repeat: 1`) | 2 | the 2023 paper's score files used 2 |
| environments | 16 (`run.envs`) | 16 in the example (we use 4, the 2023 protocol) | — |
| compute precision | `bfloat16` | `32-true` | sheeprl offers `bf16-mixed`; not the default |
| replay capacity | 5e6 | 500,000 (the whole example run) | no difference below 500k decisions |
| prefill | none named | `learning_starts: 1024` | — |
| compiled | `jax.jit: True` | `algo.compile.enabled: False` | measured as the second speed arm |
| actor unimix (discrete actions only) | key present, 0.01 | 0.01; the file says official does not apply it | irrelevant to continuous actions; not checked in `agent.py` |

So the size preset is exact; the *run recipe* differs in ratio and action repeat, and the
published 2023 scores (ratio 512, repeat 2, 4 envs, v1 agent, size S) match neither file. A
known-answer row states which of the three recipes it runs.

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

## Speed, measured (J5 step 2) — `experiments/brain_a/speed.py`

A speed pilot: one seed, no score read from it. Unmodified runner, `dreamer_v3_5_1M`, DMC
cartpole-swingup from vectors, 4 envs, action repeat 2, train ratio 512, `32-true`. A decision =
one policy step of one environment. Raw result: `/data/jack-logs/brain_a/speed_colab.json`.

| machine | arm | gradient steps/s | decisions/s | wall per 100k decisions | note |
|---|---|---|---|---|---|
| Colab T4 (2 vCPU, torch 2.14.1+cu130, py 3.13) | default (not compiled) | **2.64** (runner's own timer 2.67) | **5.28** | **5.26 h** | GPU 27% busy, 475 MiB: the Python loop on 2 vCPUs is the limit, not the T4. One 95 s interval (steps 2000→2500) — thin |
| same | `algo.compile.enabled=True` | not measured | — | — | no training step logged in 520 s: compilation ate the whole budget |
| this box (ARM, nice 19, 1 thread) | default | 0.64 | 1.28 | 21.6 h | 109 s smoke |
| Kaggle CPU kernel (4 vCPU, torch 2.11.0+cpu, py 3.13) | default threads | **0.853** (runner's own timer 0.858) | **1.71** | **16.3 h** | `jack-cpu-1794813567`, 2,227 s steady (steps 1200→5000) |
| same | `num_threads=4` | 0.831 | 1.66 | 16.7 h | same kernel, 2,227 s steady; extra threads buy nothing |

**A free CPU kernel trains A at 0.85 gradient steps/s — a T4 as shipped is only 3.1× faster.** 100k decisions
on one Kaggle CPU kernel ≈ 16.5 h, past the 9-hour kernel limit; §2's three rows × 3 seeds ≈ 150
CPU-kernel-hours. "A CPU cannot" now has its number: it can, at a third of the T4's rate.

**The record's arithmetic was 20–50 gradient steps/s on a T4 for `size1m`; the unmodified port
does 2.6 — 8 to 19 times slower.** The achieved ratio is the configured one (the runner enforces
it synchronously); environment stepping alone runs at ~1,150 decisions/s, so learning is >99% of
the wall clock.

Install note: on Python 3.13 (Colab and Kaggle) `pip install dm_control` fails building
`labmaze`; the probe installs dm_control without it (the control suite does not import it).

## What that costs (J5 step 3) — at the measured 2.64 steps/s, uncompiled

- §2's first three known-answer rows (cartpole_balance, cartpole_swingup, ball_in_cup_catch to
  200K env steps = 100k decisions each, 3 seeds, ratio 512): 9 runs × 5.26 h = **≈ 47 T4-hours**
  (the estimate on record was 25–60) — more than one week of Kaggle's 30 h. Five seeds: ≈ 79.
- One 1M-decision life in his world (action repeat 1, ratio 512 → 500k gradient steps):
  **≈ 53 T4-hours**; at the official proprio ratio 1024, ≈ 105. Per life, per seed.
- So on free T4s as shipped, A is not affordable for a 20-seed test. What must be measured
  before any known-answer run: the compiled arm with a budget that outlasts compilation
  (≥ 30 min), `bf16-mixed`, and several runs sharing one T4 (the GPU sat 73% idle).

## What comes next in J5

4. Kaggle CPU row harvested (11-16 13:26). LAUNCHED `speed colab2`: the compiled / `bf16-mixed` / 3-runs-per-T4 arms on Colab,
   25 min of training each, and re-do the two cost lines. Do not wire A to his world in J5.
