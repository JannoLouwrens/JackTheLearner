# Brain A — a Dreamer-style world model: source pin (J5 step 1)

PLAN part "His brain", stage 1, step 2 (combine). Prior: `docs/research/PUBLISHED_PRIORS.md` §2, §9 row 7.
**Code only, never weights.** Runs so far: speed pilots only (below); no score has been read from them.
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
one policy step of one environment. Raw results: `/data/jack-logs/brain_a/speed_colab.json` (run 1), `speed_colab2.json` (compiled / bf16 / ×3).

| machine | arm | gradient steps/s | decisions/s | wall per 100k decisions | note |
|---|---|---|---|---|---|
| Colab T4 (2 vCPU, torch 2.14.1+cu130, py 3.13) | default (not compiled) | **2.64** (runner's own timer 2.67) | **5.28** | **5.26 h** | GPU 27% busy, 475 MiB: the Python loop on 2 vCPUs is the limit, not the T4. One 95 s interval (steps 2000→2500) — thin |
| same | `algo.compile.enabled=True`, 520 s budget (run 1) | not measured | — | — | no training step logged in 520 s: compilation ate the whole budget |
| Colab T4 (`speed colab2`, 11-16, same stack) | **`algo.compile.enabled=True`**, 2,100 s budget | **25.9** (runner's train timer 28.1) | **51.8** | **0.54 h** | 1,274 s steady (steps 2000→68000, 132 log intervals); ratio held (cumulative 0.49 of 0.5); GPU 72%; ~840 s before step 2000 = prefill + compilation, paid once per run |
| same | `fabric.precision=bf16-mixed` | 2.09 | 4.17 | 6.65 h | slower than `32-true`: a T4 has no bf16 tensor cores; GPU 38% |
| same | default ×3 runs sharing one T4 (seeds 42–44) | 0.98 each, **2.93 summed** | 1.96 each, **5.86 summed** | 14.2 h each = **4.74 T4-h per 100k** | GPU 91%, 1,419 MiB; 1.11× one run — the 2 vCPUs, not the T4, are the limit; 764 s steady (thin) |
| this box (ARM, nice 19, 1 thread) | default | 0.64 | 1.28 | 21.6 h | 109 s smoke |
| Kaggle CPU kernel (4 vCPU, torch 2.11.0+cpu, py 3.13) | default threads | **0.853** (runner's own timer 0.858) | **1.71** | **16.3 h** | `jack-cpu-1794813567`, 2,227 s steady (steps 1200→5000) |
| same | `num_threads=4` | 0.831 | 1.66 | 16.7 h | same kernel, 2,227 s steady; extra threads buy nothing |

**A free CPU kernel trains A at 0.85 gradient steps/s — a T4 as shipped is only 3.1× faster.** 100k decisions
on one Kaggle CPU kernel ≈ 16.5 h, past the 9-hour kernel limit; §2's three rows × 3 seeds ≈ 150
CPU-kernel-hours. "A CPU cannot" now has its number: it can, at a third of the T4's rate.

**The record's arithmetic was 20–50 gradient steps/s on a T4 for `size1m`; the unmodified port
does 2.6 — 8 to 19 times slower — and compiled it does 25.9, inside the record's band.** One switch
(`algo.compile.enabled=True`, the official code's `jax.jit` analogue) buys 9.8×; bf16 and sharing
the T4 buy nothing. Uncompiled, the achieved ratio is the configured one (the runner enforces
it synchronously); environment stepping alone runs at ~1,150 decisions/s, so learning is >99% of
the wall clock.

Install note: on Python 3.13 (Colab and Kaggle) `pip install dm_control` fails building
`labmaze`; the probe installs dm_control without it (the control suite does not import it).

## What that costs (J5 step 3) — at the compiled 25.9 steps/s (fastest arm; uncompiled figures kept below)

- §2's first three known-answer rows (cartpole_balance, cartpole_swingup, ball_in_cup_catch to
  200K env steps = 100k decisions each, 3 seeds, ratio 512): 9 runs × (0.54 h + ~0.23 h prefill and
  compilation) = **≈ 7 T4-hours** (estimate on record 25–60; uncompiled 47). Five seeds: ≈ 12.
  One Kaggle GPU week (30 h) holds the three rows at 5 seeds twice over.
- One 1M-decision life in his world (action repeat 1, ratio 512 → 500k gradient steps):
  learning alone 500k / 25.9 = **≈ 5.4 T4-hours** (uncompiled 53); ratio 1024 ≈ 10.7.
  **But in his world the learner is no longer the limit:** his simulator costs 28 ms per decision
  on one core (J6 profile), so run serially one life ≈ 1M × (19 ms + 28 ms) ≈ **13 T4-hours**,
  and a 20-seed test ≈ 260 — nine Kaggle weeks. J6's simulator speed-up, or overlapping
  simulation with learning, now sets A's price; the learner does not.
- Caveats: one seed, one Colab session, cartpole from vectors (A's networks on his senses will be
  larger); compilation cost will recur per run and per shape change. No score is read from these
  runs (the compiled run's episode reward at step 68k was 702 — logged, not claimed).

### Uncompiled, as first measured (2.64 steps/s)

- §2's first three known-answer rows (cartpole_balance, cartpole_swingup, ball_in_cup_catch to
  200K env steps = 100k decisions each, 3 seeds, ratio 512): 9 runs × 5.26 h = **≈ 47 T4-hours**
  (the estimate on record was 25–60) — more than one week of Kaggle's 30 h. Five seeds: ≈ 79.
- One 1M-decision life in his world (action repeat 1, ratio 512 → 500k gradient steps):
  **≈ 53 T4-hours**; at the official proprio ratio 1024, ≈ 105. Per life, per seed.
- So on free T4s as shipped, A is not affordable for a 20-seed test. What must be measured
  before any known-answer run: the compiled arm with a budget that outlasts compilation
  (≥ 30 min), `bf16-mixed`, and several runs sharing one T4 (the GPU sat 73% idle).

## What comes next in J5

J5 is DONE (pin, two speed figures, two cost lines). Next for A, not in J5: the §2 known-answer
rows with `algo.compile.enabled=True` (≈ 7 T4-h at 3 seeds) when Kaggle GPU opens 11-21.
Do not wire A to his world before those rows reproduce.

## MLX port, stage 0 (ORDERS "Next up" 1; PORT report step table)

**0.1 DONE 11-16 22:13** — `mlx_smoke.py`, env `/data/venvs/jack-mlx` (conda-forge `mlx 0.32.4`,
Python 3.12.16, NumPy 2.5.5), run on 1 core (`taskset -c 0`): 9/9 PASS, every control fails.
Differs from the report's recipe (it said Debian 12 container + PyPI wheel):
- the PyPI 0.32.4 aarch64 wheel needs `hypotf@GLIBC_2.35` and `GLIBCXX_3.4.30`; the box has 2.34 /
  3.4.29, and Docker is refused to the worker lane → conda-forge build of the same tag (a different
  binary: every parity rung run here must also pass on the PyPI wheel on Kaggle);
- `mx.compile` on CPU JIT-builds C++ with `g++` from PATH; box g++ 11 and gcc-toolset-14 fail
  (`operator""bf16`), **gcc-toolset-15 works** — the script prepends it. Cold compile 3.7 s (the g++
  build), warm 0.035 s; JIT cache under `$TMPDIR/mlx` (= `/data/tmp/mlx`).
Two traps measured (each re-checked every run, so a fixed MLX flips them):
- `mx.fast.layer_norm`/`rms_norm` on float64 return float64 computed at **float32 precision**
  (4e-7); the gradient itself matches the primitive norm (f32 3.5e-8; f64 primitive vs central
  differences 2e-9);
- float64 **`exp`, `sigmoid`, `softmax`, `logsumexp` are float32-accurate** (rel 3e-7–2e-6);
  matmul/add/mul/log/log1p/sqrt exact, tanh 4e-16, sum 3e-15.
  → "64-bit passes" holds for exp-free ops only; rung 1's 64-bit references (symlog/symexp, two-hot,
  distributions) must be NumPy, never MLX float64.

**Next: 0.2** (one Kaggle CPU session): PyPI `mlx[cpu]==0.32.4` + official `dreamerv3@e01491f` (JAX
CPU), `debug` preset 200 steps on a dummy 74-in/8-out world, losses + versions logged; run
`mlx_smoke.py` there too (the PyPI binary's traps may differ).

**0.3 server half MEASURED 11-17 01:26** — `mlx_spike.py --size 1m` (shape-faithful dummy of the
official train step, random weights: 64-step observe pass + 15-step imagination from 1,024 starts,
both unrolled in Python, full gradients + clipped LaProp-like update inside the step; settings that
differ from the official step listed in its docstring). **647,705 trainable parameters = the
contract's `size1m` count exactly** (shapes right). One core (`taskset -c 0`), nice 19, conda-forge
mlx 0.32.4:

| mode | first step | steady step (median) | peak MLX mem | max RSS |
|---|---|---|---|---|
| eager | 2.33 s | 2.26 s | 408 MB | 560 MB |
| `mx.compile`, cold JIT cache | **37.1 s** | 2.34 s | 341 MB | 537 MB |
| `mx.compile`, warm cache | 2.85 s | 2.33 s | 403 MB | 544 MB |

- Compile of the FULL unrolled step is 37 s cold, of which ~0.5 s is tracing and the rest MLX's
  one-time g++ build of fused kernels (cached under `$TMPDIR/mlx`). G0's size1m half: 37 s ≪ 10 min,
  0.34 GB ≪ 8 GB — no per-step fallback needed at this size.
- Compiled loss equals eager to 7 significant digits (−24786.33 vs −24786.32 at step 0).
- On CPU `mx.compile` buys **no speed** at size1m (2.34 vs 2.26 s): the step is not op-overhead bound.
- Not measured here (the report gives it to Kaggle, and it would need ~16× the activations of a box
  that has ~1.5 GB to spare): **`size12m`, which is what G0 actually decides on.** G0 stays OPEN.

**Next: 0.2 + 0.3 Kaggle half** in one Kaggle CPU session: PyPI `mlx[cpu]==0.32.4`, `mlx_smoke.py`,
`mlx_spike.py --size 1m` and `--size 12m` (compiled and eager), plus the official `debug` preset 200
steps → G0 decided.

**0.2a + 0.3 Kaggle half: KERNEL READY 11-17 01:55** — `python -m experiments.brain_a.g0_kernel kernel`
(PyPI `mlx[cpu]==0.32.4`; smoke, then spike 1m/12m × compiled/eager, 6 steps each, per-run timeouts).
G0 rule frozen in its docstring as the report states it: size12m compiled cold first step < 600 s AND
peak < 8192 MB, finite. 0.2b (official `debug` preset on a dummy world) is NOT in it — needs a driver
wrapper, next. Launch at the first free Kaggle CPU session after J8/J3/J6/J10's queued kernels:
`scripts/launch_detached.sh /data/jack-logs/mlx/launch_g0.log $PY -m experiments.brain_a.g0_kernel kernel`.

**0.2b MEASURED 11-17 02:27 — on the box, no Kaggle session needed** (JAX 0.4.33 has an aarch64 CPU
wheel; scratch venv `/data/venvs/jack-dv3`, Python 3.12, the pin's `requirements.txt` with
`jax[cuda12]` → `jax==0.4.33` CPU; unpinned deps as resolved today: elements 3.22.4, ninjax 3.6.3,
portal 3.8.3, granular 0.25.0, optax 0.2.5, chex 0.1.90, numpy 1.26.4). Checkout
`/data/src/dreamerv3_e01491f` at `e01491f`. `experiments/brain_a/official_debug.py`: the official
`main()` with `--configs defaults debug`, a 74-float-in / 8-continuous-out dummy world registered by
wrapping `make_env` (nothing inside the agent touched), 200 env steps, float32 compute; settings that
differ are in its docstring. Logs `/data/jack-logs/mlx/official_debug_r{1,2}{,.log}`.
- Runs: 17.2 s wall, 16.9 CPU-s, peak RSS 448 MB; optimizer 2,666 params (debug widths); training
  starts at env step 132 (replay prefill), 7 train logs to step 204; all 8 loss series finite:
  con 0.864→0.783, dyn 1→1, rep 1→1 (free nats), policy −1.6e-4, repval 3.2189 and rew 1.6094
  (= ln 25, ln 5: uniform over the debug preset's 5 bins), value 0.166→0.182, vector 38.3→39.9.
  Flat losses are expected: the optimiser warms up over 1,000 updates and this run makes ~20.
- **Same seed, two runs: NOT identical.** Steps line up, dyn/rep identical, but vector loss differs by
  up to 1.37, con 0.032, value 0.0079, rew/repval 2.4e-7. The official TRAINING LOOP is not
  run-to-run reproducible on CPU (env threads + async replay sampling decide which batch the learner
  sees). Consequence for G1 (step 0.4): the golden-file generator must call the agent's train/policy
  functions directly on a fixed batch with file noise, never go through `embodied.run.train` —
  the report's design already says this; this is the measurement that makes it binding.
**Next: G0** (`g0_kernel.py`) at the first free Kaggle CPU session; then 0.4, the golden-file
generator (≤ 60-line patch: file noise for the two samplers, direct `agent.train` on a stored batch,
compiler off), run twice → G1.

**0.4a MEASURED 11-17 02:58 — G1 pre-probe, on the box** (`experiments/brain_a/g1_probe.py`, logs
`/data/jack-logs/mlx/g1_probe_r{1,2}.{json,log}`): the official agent built by `make_agent`
(`defaults debug`, float32, seed 0, compiler AS-IS), a fixed numpy batch (B 8 × T 11), seeds via
`agent._seeds(i)` exactly as `stream()`, 5 direct `agent.train()` calls, two separate processes:
**all 183 params bit-identical (sha256), all 4×30 returned metrics identical.** So 0.2b's
non-repeat is the loop, not the agent: G1 is reachable by calling train directly. NOT yet shown: a
control that must differ (different per-step seed → different params), file noise for the two
samplers, compiler off, intermediates. Next (0.4b): add the seed-changed control, then the
file-noise patch + intermediate dumps on `debug`, twice → G1.

**0.4b MEASURED 11-17 03:35 — the controls that must differ, on the box** (`g1_probe.py --seed`,
`--step-seed-offset`, `--compare`; logs `/data/jack-logs/mlx/g1_probe_{r3,c_seed1,c_step1000}.*`,
55 CPU-s for three runs). Against r1 (seed 0): r2 and r3 (r3 = the edited script, seed 0)
**IDENTICAL 183/183 params, 120/120 metrics**; agent seed 1 **DIFFERS** (14/183 params, 55/120
metrics equal); same init with only the per-step train seeds shifted by 1000 **DIFFERS** (21/183,
65/120) — so `train()` consumes its seed and the identity is not a seed-blind step. The params
equal under both controls are counters, optimiser slots, return-norm bounds, norm scales and the
zero-initialised policy-mean head (`pol/head/action/mean/*` unchanged after 5 steps, as is
`slowval`), i.e. the ones 5 debug steps leave at their deterministic init — a golden file built
on 5 steps does NOT exercise the actor's output layer; G1's file should run long enough (or dump
gradients) that `pol/head/action/mean` moves. Next (0.4c): file noise for the two samplers,
compiler off, intermediate dumps, twice → G1.

**0.4c(i) MEASURED 11-17 03:40 — how long G1 must run, and compiler off, on the box** (`g1_probe.py
--steps 60` with a per-step trace, `--jit False`; logs `/data/jack-logs/mlx/g1_probe_{s60,nojit1}.*`).
- 60 direct train calls, seed 0, compiled: 18 CPU-s, 37 s wall. First step after which each traced
  param differs from its init: **`pol/head/action/mean/{kernel,bias}` and their two Adam slots: step 6**;
  `slowval` kernels step 1, head bias 2, linear0 bias 3, norm0 scale 10. 0.4b's "slowval unmoved"
  was the wrong reading: those params are EQUAL ACROSS THE CONTROLS at step 5, not unmoved (an EMA of
  the value net that both controls track the same way). The real gap holds for the actor head: a
  ≤ 5-step golden file never exercises it (compiled; eager see below).
- **Compiler off, as shipped, cannot run.** Official `jax.jit: False` sets `jax_disable_jit` globally
  (`embodied/jax/internal.py:36`) but `agent.py:457/467` always AOT-lower train/report: with
  ninjax 3.6.3 (the newest at the pin's date, 2026-10-04) it dies in `ninjax.unflatten`'s key assert
  inside `opt.update`; with 3.6.2 it dies on JAX's "Disable jit is not supported in the AOT path";
  `jax.disable_jit()` around train() alone hits the same AOT refusal.
- **0.4c(ii) the fix, 11-17 03:48:** `g1_probe --jit False` skips `Agent._compile_train/_compile_report`
  (2 methods, a listed DIFFERENCE) AND needs ninjax 3.6.2 (`PYTHONPATH=/data/tmp/ninjax362`, a
  second listed difference; 3.6.3 still asserts). Then, 12 train calls, seed 0:
  eager r1 vs r2 **IDENTICAL 183/183 params, 330/330 metrics** (145 CPU-s each, 4 min wall, 8× compiled);
  compiled ninjax 3.6.3 vs 3.6.2 **IDENTICAL** (the ninjax swap changes nothing compiled);
  eager vs compiled **DIFFER** (36/183, 195/330) — XLA fusion is not bit-equal to op-by-op, so the
  golden file must say which mode made it. Eager, the actor head first moves at **step 10** (compiled 6).
  **Rule for G1's file: ≥ 20 train calls in the mode the file is made in** (2× the later first move).
  Eager controls (03:52) DIFFER as they must: agent seed 1 9/183 params, 139/330 metrics equal;
  per-step seed +1000 9/183, 159/330. Eager identity is not seed-blind.
  Next (0.4c iii): file noise for the two samplers + intermediates.

**0.4c(iii) MEASURED 11-17 04:29 — file noise for the two samplers; G1 holds eager, on the box**
(`experiments/brain_a/file_noise.py`, `g1_probe --jit False --noise-file`; ninjax 3.6.2; logs and
noise files `/data/jack-logs/mlx/g1_noise_{r1,r2,c1}.*`, `noise_s{0,1}.npz` = 4M float32 uniforms +
4M normals each, numpy `default_rng(seed)`). `outs.Normal.sample` → `normal[cursor]*std + mean`;
`outs.Categorical.sample` → argmax(logits − log(−log u)), u = `uniform[cursor]` (the Gumbel-max that
`jax.random.categorical` itself is); differences listed in the module docstring.
- 20 eager train calls, seed 0, file s0, twice: **IDENTICAL 183/183 params, 570/570 metrics → G1
  PASSES for the eager parameter/metric file.** Control = the same run with ONLY the noise file
  changed (s1): **DIFFERS** (10/183 params, 209/570 metrics equal) — the samplers really are fed by
  the file, and nothing else random in the step leaks past it (init weights are unpatched threefry,
  identical in all three runs).
- Noise schedule, identical in all runs: `make_agent` consumes 68 normal / 108 categorical calls
  (5,440 / 5,472 values) BEFORE the first train call (its init pass); every train call then consumes
  exactly **34 normal + 54 categorical calls = 21,760 + 21,888 values**. The MLX side must replay
  the init consumption (or start its cursor at 5,440 / 5,472). Binary/dropout never called.
- Actor head `pol/head/action/mean` first moves at step 8 (s0) / 9 (s1) → a 20-call file exercises it.
- Cost: 3 runs in parallel, 3 min 34 s wall at nice 19.
Next (0.4d): dump one step's intermediates (RSSM posterior/prior logits, imagined features, returns,
the 8 losses, grads by name) as .npz from the same eager file-noise run, twice → identical; then G0
(`g0_kernel.py`) at the first free Kaggle CPU session closes stage 0.

**0.4d MEASURED 11-17 04:43 — one train call's intermediates, twice, on the box** (`g1_probe
--dump-step 20 --dump X.npz`, `--compare-npz`; eager, ninjax 3.6.2, file noise; logs
`/data/jack-logs/mlx/g1_dump_{r1,r2,c1,smoke}.*`). Capture = a proxy for `ninjax` as seen by
`embodied/jax/opt.py` (module attribute swapped; no official file edited; listed difference) that, on
the armed call, records every gradient by name (57), the scaled total loss, and the loss's aux
`outs`: encoder tokens, posterior `repfeat` deter/stoch/logit, the 8 per-element losses (con dyn
policy rep repval rew value vector) — 70 arrays, 5,859 values.
- r1 vs r2 (noise s0): **IDENTICAL 70/70 arrays** and 183/183 params, 570/570 metrics. The capture
  perturbs nothing: r1's params/metrics equal 0.4c iii's g1_noise_r1 183/183, 570/570.
- Control c1 (noise s1 only): **DIFFERS 10/70** equal (and 10/183 params, 209/570 metrics).
- **The 10 that stay equal are a hole, not luck:** at call 20 `loss/dyn` and `loss/rep` sit at the
  free-nats floor (exactly 1.0 in all 80 cells) so **every `dyn/prior*` gradient is exactly 0** (8
  arrays). Over 20 calls the dyn loss leaves the floor only at calls 2 (1.0064) and 9 (1.0005); call 1
  (smoke) has prior grads up to 0.0151. Cause: the fixed batch is uniform noise, so the posterior has
  nothing to say that the prior cannot. A golden file dumped at call 20 would pass an MLX port whose
  prior backward is wrong.
- Not reached without editing the agent: imagined features and λ-returns (local to `Agent.loss`).
Next (0.4d ii): dump calls 1, 2 and 20 (prior active + actor head moved) — or a batch with
structure so KL stays above 1 — and assert in the probe that no captured gradient is all-zero
except ones named in a list; then G0 at the first free Kaggle CPU session.

**0.4d ii MEASURED 11-17 05:40 — all 20 calls dumped, twice, plus a zero-gradient guard** (`g1_probe
--dump-step 1,...,20 --allow-zero ...`; keys `c<N>/...`; eager, ninjax 3.6.2, file noise; logs
`/data/jack-logs/mlx/g1_dump20_{r1,r2,c1}.*`). New guard: the probe exits 3 (`ZERO_GRAD_CHECK FAIL`)
when a gradient is exactly zero in EVERY dumped call and not named in `--allow-zero`.
- Known-positives: the guard FAILS 0.4d's call-20 dump (10 always-zero) and the call-1 smoke (8).
- r1 vs r2: **IDENTICAL 1,400/1,400 arrays**, 183/183 params, 570/570 metrics; r1 equals 0.4c iii's
  g1_noise_r1 183/183, 570/570 (capturing every call perturbs nothing). Control c1 (noise s1):
  DIFFERS 209/1,400 arrays, 10/183 params.
- Zero-gradient map over the 20 calls: the 8 `dyn/prior*` grads are non-zero ONLY at calls 1 and 8;
  `pol/head/action/mean/{bias,kernel}` are zero at calls 1–7 and in 4 of the last 6 (bias at 15–17, 20;
  both at 1, 2); `rew`/`val` first layers zero at calls 1–2. **Call 8 is the only call with all 57
  grads non-zero.**
- **0.4d's plan (calls 1, 2, 20) FAILS the guard**: the actor head's mean gradient is zero in all
  three, so that golden file could not catch a wrong actor-mean backward. {1, 8, 20} PASSES (0
  always-zero); call 8 alone passes. Cost: 245 s and 2.4 GB peak per 20-call eager run (three run in
  parallel this once — above the box's 1.5 GB guideline; run them one at a time).
- Why the actor mean's gradient is so often exactly zero is not yet known (not investigated; a
  saturated/clipped mean on noise inputs is a guess, not a finding).
Next: G1 golden file = calls {1, 8, 20} with the guard on, no `--allow-zero`; then G0 (`g0_kernel.py`)
at the first free Kaggle CPU session.

**0.4d iii MEASURED 11-17 05:44 — why the actor mean's gradient is (nearly) zero: the official
zero-init heads under warmup, on a noise batch** (from 0.4d ii's `g1_dump20_r1.npz`, no new run for
the diagnosis; `/data/tmp/j_mlx/scale.py`).
- Not saturation. The actor loss (agent.py:411–414) is `-(logpi * sg(adv_normed) + actent * ent)`;
  the entropy term does not depend on the mean, so the mean's gradient is the advantage term alone.
- `adv = (ret - tarval)/rscale` (rscale ≥ 1) and both sides start at EXACTLY 0: the official
  `rewhead` and `value` have `outscale: 0.0` (zero output layer → uniform logits over the `debug`
  preset's 5 symmetric bins → symexp-twohot mean 0). The reward loss reads ln 5 = 1.609438 at call 1
  and 1.609389 at call 20: the heads barely move, because `opt.warmup: 1000` at lr 4e-5 gives an
  effective lr of 8e-7 at call 20. So adv rounds to exactly 0 in float32 at 11 of 20 calls (1–7,
  15–17, 20) and is ~1e-8 elsewhere.
- **Consequence: {1, 8, 20} passes the zero guard but is still a hole for the port.** At call 8 the
  mean-kernel grad is 3.5e-11 against a largest grad of 6.6e+1 (ratio 5e-13, below float32 relative
  precision); stddev-kernel 4.3e-7. A bit-exact G1 sees it; a tolerance-based JAX-vs-MLX comparison
  (the port's later gate) would pass a wrong logp-by-mean backward. A golden file must make the
  advantage non-trivial.
- **Test of the cause (one run, 245 s, eager, file noise s0, `--set agent.opt.warmup=0` = a LISTED
  difference; `/data/jack-logs/mlx/g1_dump20_w0.*`):** mean-kernel grad zero only at calls 1–2
  (was 11/20); 4.4e-7 at call 8 (was 3.5e-11, ×12,000), 4.3e-6 at call 20 (was 0); reward loss
  1.609438 → 1.602759 (was → 1.609389). Zero guard PASS (prior grads still 0 at call 20). Call 1 is
  unchanged bit for bit in the printed scales (warmup does not touch the first gradient). So
  warmup-throttled zero-init heads are the cause; the advantage grows as they learn.
- Not yet decided (next): what G1's golden file uses. Options, raceable on the box at ~4 min each:
  (a) official config, more calls (≥ ~1,000 to leave warmup — 3.4 h eager, too slow); (b)
  `agent.opt.warmup=0`, calls {1, 8, 20} or later (one listed difference, cheap); (c) compare every
  tensor against its OWN scale in the port gate (per-tensor relative tolerance), which a 5e-13
  global ratio needs anyway. Recommendation: (b) + (c). Then G0 at the first free Kaggle CPU session.

**0.4e MEASURED 11-17 06:20 — G1 golden file chosen (warmup 0), and rung 10's gate made per-tensor**
(`experiments/brain_a/port_gate.py`; logs `/data/jack-logs/mlx/{g1_dump20_w0r2.*,g1_w0_vs_w0r2.log,
port_gate_selftest_{w0,r1}.log}`). Took 0.4d iii's recommendation (b) + (c).
- (b) **G1 for the warmup-0 file: the 20-call eager file-noise run with `agent.opt.warmup=0` (the
  one listed difference), repeated in a fresh process: IDENTICAL 1,400/1,400 arrays, 183/183 params,
  570/570 metrics; zero guard PASS with nothing allowed (call 8: 0/57 zero; calls 1 and 20 carry 8
  each, different ones).** Frozen as the golden file: `/data/jack-logs/mlx/golden_g1_debug_w0_noise_s0.npz`
  (sha256 583e3c30…, + `.json` 0f0e8478…, noise `noise_s0.npz` 88e6654f…). Control: the same run with
  the official warmup (`g1_dump20_r1`) FAILS the gate on 931/1,140 tensors — the flag is visible.
- (c) **Rung 10 says "relative 1e-4 (1e-3 recurrent); cosine ≥ 0.9999" and not relative to what.
  Measured against the whole gradient it is blind:** on the official-warmup file a SIGN-FLIPPED
  actor-mean backward moves the global relative error to 2.8e-9, a 1% scale error to 1.4e-11 — a
  global 1e-4 gate PASSES both. On the warmup-0 file the sign flip reaches 1.7e-4 (caught) but the 1%
  error only 8.4e-7 (missed). `port_gate.py` compares each tensor against its own norm (`dyn/` at 1e-3,
  the rest 1e-4, cosine ≥ 0.9999; a golden-zero tensor must stay exactly zero and is counted
  unchecked). `--selftest` = 8 planted cases, each with the verdict it must give: identical PASS,
  float32 round-off (×(1+2⁻²³·N)) PASS, actor-mean sign flip FAIL, actor-mean ×1.01 FAIL, dyngru ×1.005
  FAIL, dyngru ×1.0002 PASS (the recurrent band is real), a missing key FAIL, the other-noise control
  dump FAIL — **8/8 as required on both the warmup-0 and the official-warmup file.**
- Still open in rung 10 as written: gradients PER LOSS TERM (the dump holds the total loss's gradient;
  splitting needs `Agent.loss` opened — the port's job). Per-tensor tolerances on tensors of size ~1e-8
  (`pol/mlp/*` at call 8) are uncalibrated: an honest MLX build may miss 1e-4 there through rounding
  alone — the first real MLX comparison calibrates, and any loosening is re-registered, not edited.
Next: G0 (`g0_kernel.py`) at the first free Kaggle CPU session closes stage 0.

## G0 — PASS (11-19, Kaggle CPU, `jack-cpu-1795051615`, 48 min, result `/data/jack-logs/mlx/g0.json`)

Rule unchanged from `g0_kernel.py`'s docstring: size12m, `mx.compile`, cold cache, first step < 600 s
AND peak < 8192 MB, loss finite. mlx 0.32.4, Python 3.13.18, x86_64, 4 cores, fresh process per run.

| run | params | first step | steady median | peak (mlx) | max RSS |
|---|---|---|---|---|---|
| 1m compiled | 647,705 | 74.7 s | 17.42 s | 341 MB | 529 MB |
| 1m eager | 647,705 | 17.8 s | 17.68 s | 409 MB | 559 MB |
| **12m compiled (G0)** | 9,765,209 | **216.3 s** | 212.2 s | **1,811 MB** | 2,152 MB |
| 12m eager | 9,765,209 | 212.6 s | 212.4 s | 1,427 MB | 1,701 MB |

- **G0 PASS: 216 s < 600 s, 1.8 GB < 8 GB, finite.** Compile is not a risk: at 12m the compiled
  first step costs ~4 s over a steady step. Losses are bit-identical compiled vs eager at every
  printed step (−54438.61, −5235.11, −57498.49): the compiled graph computes the same thing.
- **But the speed it shows is not a speed to train on.** Kaggle x86 runs the 1m step at 17.4 s;
  this box's single ARM core ran the same step at 2.34 s (table above) — **7.4× slower on 4 x86
  cores than on 1 ARM core.** MLX's x86 CPU path is evidently unoptimised (it is built for Apple
  silicon; Linux x86 is a portability target). At 212 s per 12m step, x86 CPU MLX is ~460× below
  official JAX on this box (0.925 grad steps/s, `9ba4e682`) and ~200× below sheeprl (0.458).
- What G0 therefore decides, and what it does not: the full unrolled step compiles and fits, so
  the port proceeds to rung 1 (the report's rule). It says **nothing** about the Mac's speed —
  the Mac runs Metal, not this CPU path. The port's speed case still rests entirely on the Mac's
  first-day rows (`mac_first_day.py` rows 11–12). Lesson for later rungs: never time MLX on
  Kaggle x86 and quote it; golden-file comparisons there are fine (correctness), speeds are not.
Next: rung 1 (the report's step list), correctness-only on box/Kaggle; speed only on the Mac.

## J16 (a) — row 0.4 / gate G1: already met (0.4e, 11-17)

Row 0.4's done-test ("parameters by name plus one step's intermediates saved; two runs give
identical files") is 0.4e above: the 20-call eager file-noise `debug` run (warmup 0, the one listed
difference), two fresh processes, **IDENTICAL 1,400/1,400 arrays, 183/183 params, 570/570 metrics**;
golden file `/data/jack-logs/mlx/golden_g1_debug_w0_noise_s0.npz` sha256 583e3c30… (re-checked
11-19, unchanged). Not re-run: G1 holds.

## J16 (b) — row 1.2: rung 1's tests, committed while they FAIL (11-19)

`experiments/brain_a/rung1.py` (tests + NumPy float64 references) against `mlx_maths.py` (a stub
that raises until row 1.3). Nine tests, each run eager AND `mx.compile`d: squashing, bins, two-hot
(weights + loss), decode (value + uniform and mirrored exactly 0), 1% unimix, KL with free bits
(values, both stop-gradients exactly 0, both live gradients, clamp → zero gradient), straight-through
(forward exactly one-hot, gradient), bounded normal (mean, std, logp, entropy, both raw gradients),
coin-flip logp (|logit| to 100). Logs `/data/jack-logs/mlx/rung1_{vs_official,stub,planted}.log`.

- **The references are the official code, not our reading:** `--vs-official` (JAX 0.4.33 CPU,
  float32, e01491f's own `outs.py`/`nets.py`/heads formula) **25/25 agree**.
- **Stub: 0/9 pass (FAIL, as row 1.2 requires).** Known-positives: nine planted wrong MLX
  implementations (log(1+x) for log1p, even-branch bins, two-hot in symlog space, naive decode sum,
  mix on logits, KL without stop-gradients, `(onehot+p)-p`, std without +2, log(sigmoid))
  **9/9 CAUGHT**.
- What "1e-6" means is now written down (the report does not say relative to what): per element,
  |port − ref| ≤ 1e-6 · (|ref| + s), s per test; for the decode and the two actor gradients the scale
  is the sum's condition (summed magnitudes), because measured:
  - **the official float32 bins are not the float64 bins**: built with `linspace(dtype=f32)` they sit
    up to 1.24e-6 off (232/255 differ from the float64 value cast). Rung 1's bins reference is the
    official array, pinned bit for bit in `rung1_bins_e01491f.json`;
  - **the official actor-mean gradient misses an absolute 1e-6**: at raw mean 8 XLA's float32 tanh
    returns 1.0 (the nearest float32 is 0.99999976), so the gradient is 0 where float64 says 2.2e-6
    (at 6: off by 1.1e-6) — 1 − y² in float32 carries ~1 ulp of 1 times the upstream gradient. Judged
    against that condition the official code is within 2.6e-7.
- **Exact zero under compilation fails in BOTH libraries for the official recipe:** the official
  symmetric sum gives 0 eager but **−0.163 / −0.172 under `jax.jit`** (uniform; it varies run to run)
  and **+0.164 under `mx.compile`** (mirrored: up to 0.054). Issue #245 reproduced on the box and on
  MLX. So row 1.3 must cancel mirrored probabilities before multiplying (r2dreamer `7a93691`); the
  port then deviates from the official code here on purpose — one row in `mlx_maths.py`'s table.
Next (row 1.3): fill `mlx_maths.py` until `rung1.py` exits 0 (9/9 eager and compiled), each
deviation in its "ours beside theirs" table; then the same on the PyPI wheel on Kaggle.

## J16 (c) — row 1.3: the maths module; rung 1 PASS on the box (11-19)

`experiments/brain_a/mlx_maths.py` filled line for line from the official lines it cites.
**`rung1.py` exits 0: 9/9 tests pass eager AND `mx.compile`d** (mlx 0.32.4, box ARM CPU; log
`/data/jack-logs/mlx/rung1_pass.log`): `squash`, `bins`, `twohot`, `decode`, `unimix`, `kl`,
`straight_through`, `normal`, `binary`. Worst errors, as multiples of rung 1's per-element scale:
bins 1.6e-7, decode 5.7e-7 (cond), KL values 1.9e-7, every gradient ≤ 1.6e-7; uniform and mirrored
decode exactly 0 eager and compiled; both stop-gradients exactly 0; free bits → zero gradient.
`--planted` re-run with the filled module: still 9/9 CAUGHT (`rung1_planted_filled.log`).

Two deviations from the official code, both in the module's "ours beside theirs" table:
- **decode**: `(p3 − p1[::-1]) · b3` — mirrored probabilities cancelled before the multiply. Same sum
  in exact arithmetic (bins mirrored bit for bit); the official form gives +0.164 compiled (probe
  re-printed in the planted log), this one exactly 0.
- **the bins grid — measured, not in the report**: the first fill used `mx.linspace` and FAILED
  `bins` at 1.92e-6 (8/9). The official grid is not the jaxpr's arithmetic: XLA:CPU rewrites
  `iota/127` to `iota · f32(1/127)` (optimized HLO) and LLVM contracts `1 − iota·r` into one fma, so
  each point is `−20 · fma(−i, r, 1)`. Measured on the 128 points: `mx.linspace` misses JAX's grid
  at 41, the jaxpr computed op by op at 41, the fma form at **0**. The port computes that fma as one
  float64 rounding on the CPU stream (an f32·f32 product is exact in float64; bins are built once).
  Remaining bin differences are expm1's last ulp (12/255, ≤ 1.6e-7). Lesson for later rungs: parity
  with the official code means parity with what its COMPILER emits, read from the optimized HLO.

Next: the same `rung1.py` on the PyPI MLX wheel (report's row after 1.3), then rung 2.

## J16 (e) — rung 2: tests written failing, then filled → PASS (11-20, step 1.4)

`experiments/brain_a/rung2.py`: λ-returns (agent.py L482–490, disc 1−1/333 and 1, λ 0.95), percentile
(linear, = `jnp.percentile`), the `perc` return normaliser (utils.py L16–91, retnorm config, debias
False and True), slow-critic average (utils.py L113–119, rate 0.02, every 1 and 3). Forward only (all
under stop-gradient in the official losses; the port's stats must carry zero gradient — tested).
- `--vs-official` (jack-dv3, jax 0.4.33): **20/20 references hold** against the official code.
  Finding: EAGER `jnp.percentile(x, 100)` lands one float32 ulp below the top rank (399.99448 for a
  max of 400.0); under `jax.jit` — how the training step runs — it is exact, so the reference is the
  compiled value (rung 1's lesson). `SlowModel.update` is restated, not run (needs a tracked module).
- port (jack-mlx): **0/4 — stubs in `mlx_maths.py` raise** (step 1.4's "tests exist and fail").
- `--planted`: **4/4 CAUGHT** (TD(λ) without the boot term, nearest-rank percentile, scale without
  limit/debias, slow mix ignoring `every`). Rung 1 re-run: still 9/9.
- (d), the PyPI wheel: cannot load on this box (GLIBC 2.34 < 2.35, see stage 0.1) — needs a Kaggle CPU session.
**FILLED, same slot: `rung2.py` → RUNG 2: 4/4 PASS eager AND `mx.compile`d** (mlx 0.32.4, log
`/data/jack-logs/mlx/rung2_pass.log`): returns 1.4e-7, normaliser 1.4e-7, slow 7.6e-8 (× (|ref|+1)),
stats gradient exactly 0, slow count 10/10; rung 1 still 9/9. One deviation, in the module's table:
percentile = flat sort + linear interpolation, rank/weight in Python float64 (= jit'd `jnp.percentile`).
Next (PORT report): the rest of row 1.4 by the report's step table, or (d) on Kaggle.

## J16 (d) — the PyPI wheel line (11-20): **WHEEL PASS**
Kaggle CPU kernel `jack-cpu-1795207086` (`experiments/brain_a/wheel_kernel.py`, verdict fixed before
the run, `c6611404`), PyPI `mlx[cpu]==0.32.4`, x86_64 / glibc 2.39 / Python 3.13.18; result
`/data/jack-logs/mlx/wheel.json`. All four runs rc 0: `rung1.py` **RUNG 1: 9/9** eager AND compiled
(mlx 0.32.4) -> PASS (34 s); `rung1.py --planted` **9/9 caught**; `rung2.py` **RUNG 2: 4/4** eager AND
compiled -> PASS (15 s); `rung2.py --planted` **4/4 caught**. Tests unchanged from the box's conda-forge run.
Limit, stated: this is PyPI's Linux x86_64 wheel of the same version, not the Mac's macOS arm64 binary —
the source and version the Mac will install pass, the Mac's own binary is first read on the Mac.
Next (PORT report): the rest of row 1.4 by the report's step table.

## J16 row 1.5 — rung 3: the optimiser → PASS (11-21, step 1.5)

`experiments/brain_a/rung3.py` (tests `opt_cfg`, `opt_vis`; configs.yaml exactly + a visible-step
config, 5 steps, agc bind / pmin / free branches). References vs the official `_make_opt` under
`jax.jit`: **14/14** (`--vs-official`). The jitted warm-up ramp is named: XLA:CPU runs
`lr_t = fma(-lr, fma(-count, f32(1/warmup), 1), lr)` (as rung 1's linspace), bit-exact at every
count of four configs. Golden = the jitted official chain, `rung3_opt_golden_e01491f.json`
(`--vs-official --dump-golden`); the port is read against reference AND golden at 1e-6.
Stubs committed failing `dfde3cf9` (0/2); filled → **RUNG 3: 2/2 PASS eager AND `mx.compile`d**
(mlx 0.32.4). `--planted` 5/5 caught + the bug-free chain as a control PASSES; a mutant of the real
port with an unfused outer ramp FAILS (6.5e-6). Rungs 1 (9/9) and 2 (4/4) re-read PASS. Ours beside
theirs: in `mlx_maths.py` above `opt_update`. Wheel re-check not yet run.

## J16 row 2.1 — golden generator v1 at `size1m`: first measurement (11-21 10:31)

`g1_probe.py --preset size1m` (new; `debug` path unchanged): `defaults` + `size1m` + the golden-file
rule's `--jax.platform cpu --jax.prealloc False` (listed differences), float32, eager, file noise
(`--noise-n 12000000`, `/data/jack-logs/mlx/g1v1/noise1m_s0.npz`), warmup 0. Logs `/data/jack-logs/mlx/g1v1/`.
- **647,705 params** — the report's count, exactly. 261 param arrays, 81 gradients dumped.
- Batch 16×65 (published): died at init on ninjax 3.6.3's assert — `/data/tmp/ninjax362` had been
  pruned; reinstalled ninjax 3.6.2 to `/data/tools/ninjax362` (path updated here and in the probe).
  **Re-check: the frozen debug G1 golden re-made with it is IDENTICAL 1,400/1,400** (sha 583e3c30 unchanged).
- Batch 4×65 (`--set batch_size=4`, a difference NOT yet adopted): one eager train call **3 m 12 s wall,
  peak RSS 3.4 GB** (the debug 20-call G1 run peaks ~2.4 GB too) — over the box's ~1.5 GB rule, and
  shrinking the batch did not bring it under. Call 1: **14/81 grads exactly zero** (rew/val MLPs behind
  zero-init heads, `pol/head/action/mean`) → the golden must span ≥ 6 calls (debug: actor mean moves at 6).
- Consequence: the v1 generator (≥ 6–20 calls, run twice + a seed control) belongs on a Kaggle CPU
  kernel (30 GB), not this box. Not yet captured: per-step recurrent and imagination intermediates
  (rungs 6, 7, 9 — `install_capture` reaches grads, loss and `outs` only).

**Row 2.1 golden v1 — GOLDEN PASS (11-21, Kaggle CPU `jack-cpu-1795271133`, 1,468 s, install 20 s).**
Kernel `golden_kernel.py` (`c4020afc`, py3.12 uv venv fix `cf507ea9`; kernel image py3.13, x86_64, 4 CPUs);
result `/data/jack-logs/mlx/g1v1/golden.json`, golden arrays `golden_size1m_A.npz` (251 MB, /data only).
Judged by the GOLDEN line fixed in the docstring before the run:
- runs A, B, control: rc 0, 0, 0 (no call with every dumped grad exactly zero).
- **A vs B: 2,468 / 2,468 dumped items equal** (every array and param hash) → repeatable.
- **A vs control (seed 1): 533 / 2,468 equal** → the seed is read; the control fails as it must.
- in_reset: 1 on even rows, 0 on odd, at observe step t032 (EP_COL 33 − 1) in every dumped call
  (1, 2, 6, 8); 0 at every other of the 256 keys.
- noise file sha256 `91e557dd…cf7a`. This is the size1m reference the Mac port is checked against;
  rows 2.2 (one recurrent step → rung 6) and 2.3 (64-step pass → rung 7) still need per-step RSSM
  intermediates the capture does not reach yet.

## J16 row 2.2 — one recurrent step (rung 6): references first (11-21 16:45)

`rung6.py --vs-official` (jack-dv3 venv, `PYTHONPATH=/data/tools/ninjax362`): NumPy float64 `REF_step`
(reset masking, DictConcat, `_core` with per-block GRU gates, obs layer → posterior logit, `_prior`)
against the official `RSSM._observe` + `_prior` at e01491f, float32 jitted (COMPUTE_DTYPE f32 as A1–A3;
published default bf16). Toy size (deter 64, hidden 32, stoch 4×8, blocks 8), batch 4, rows 0/2 reset.
- **12/12 REFERENCES HOLD**: masked deter + action embedding exact; deter 5.6e-8, post 2.2e-7, prior
  3.3e-7 ×(|ref|+1); reset rows equal the from-zero-carry step; control: kept rows differ (0.75).
- Planted wrong references each CAUGHT: erf gelu (official is the tanh form), GRU gates split flat
  instead of per block, no `max(1,|a|)` action squash. (The first squash mutant was itself a no-op and
  MISSED — fixed.) Posterior sample values NOT compared (rung 7's sampler); only one-hot shape.
- Golden `rung6_step_golden_e01491f.npz` (sha256 b3eddc2e…): params, inputs, official outputs.
- Next: port stub in `mlx_maths.py` + rung 6 tests committed FAILING, then the port, eager and compiled.

## J16 row 2.2 — rung 6: one recurrent step → PASS (11-21, step 2.2)

`experiments/brain_a/rung6.py` (tests `step_ref`, `step_golden`), port `mlx_maths.rssm_step`
(+ `_core`, `_layer`, `_blocklin`, `_rms`, `_gelu`). References `--vs-official` 12/12 (`5109a823`);
tests committed failing `f5fa4cad`; port **2/2 eager AND compiled** on mlx 0.32.4 (box CPU):
masked deter, action embedding, deter', posterior logit, prior logit within 1e-6·(|ref|+1) of the
NumPy-64 reference and of the official f32 jit golden `rung6_step_golden_e01491f.npz`; reset rows
equal the zero-carry step, kept rows differ (control). `--planted` 3/3 caught (erf gelu, rms eps
1e-6, update gate without −1). Toy size (deter 64, blocks 8). The posterior SAMPLE is rung 7's.
No speed read. Wheel re-check not run.

## J16 row 2.3 — the 64-step memory pass (rung 7): references first (11-21 17:10)

`rung7.py --vs-official` (jack-dv3, `PYTHONPATH=/data/tools/ninjax362`): NumPy float64 `REF_observe`
(rung 6's `REF_step` + the OneHot sampler: unimix 1% → log → argmax(+Gumbel of file uniforms)) for 64
steps from the zero carry, against the official `RSSM.observe` scan path at e01491f, f32, run under
`jax.disable_jit()` with `file_noise.FileNoise` so the cursor advances per step. Toy size as rung 6,
batch 4; resets: all rows t=0, row 1 at t=20, row 2 at t=40, row 3 never again.
- **15/15 REFERENCES HOLD**: deter 8.5e-8, post 3.7e-7, prior 6.9e-7 ×(|ref|+1) over all 64 steps
  (no drift growth: deter |d| 0 / 9.9e-8 / 6.8e-8 / 5.5e-8 at t 0/15/31/63); **0 sample flips of 1,024
  latents, 0 near-ties fed** (tie rule margin < 1e-4); row 1's t=20 reset equals the zero-carry step
  (deter exactly 0 at init — weak alone — and post 2.4e-7, |logit| 1.69); row 3 control differs 0.27.
- Controls/planted, each FAILS as it must: constant noise flips 760/1,024; no unimix 9 flips; noise
  one call late 852; posterior sampled from the prior logit 360; resets after t=0 ignored 7.
- **Bookkeeping the port must copy**: creation pass = 65 sampler calls; the scoring pass makes 1
  PRERUN call (ninjax `_prerun` of step 0, 128 uniforms) BEFORE the 64 real ones — a file-noise port
  must skip that block or read the size1m golden one call off.
- Golden `rung7_scan_golden_e01491f.npz` (302 KB, sha256 dd3af760…): params, inputs, the real calls'
  uniforms [T,B,4,8], official deter/post/prior/stoch. Next: port stub + failing tests, then the port.

## J16 row 2.3 — rung 7: the 64-step memory pass → PASS (11-21, step 2.3)

`experiments/brain_a/rung7.py` (tests `scan_ref`, `scan_golden`; committed FAILING `eb4ce9e2`, stub
0/2), port `mlx_maths.rssm_observe` (64 × `rssm_step` + `sample_onehot`). `taskset -c 0 nice -n 19
/data/venvs/jack-mlx/bin/python -I experiments/brain_a/rung7.py` → **RUNG 7: 2/2 eager AND compiled**
(mlx 0.32.4, box, 4 s). Tolerance = ladder row 7's 1e-4 ×(|ref|+1) and 0 flips; read worst: vs
NumPy-64 REF deter 7.9e-8, post 3.0e-7, prior 8.2e-7; vs official f32 golden deter 8.3e-8, post
3.5e-7, prior 6.9e-7; **0 flips of 1,024 latents** against each. Inside each test: samples one-hot;
row 1's t=20 reset == the port's own zero-carry step (post, exact); row 3 control differs 0.27;
constant-noise port flips 760/1,024. `--planted` 4/4 CAUGHT (no unimix, noise one call late, resets
after t=0 ignored, carry fed probs not the sample); unpatched port passes.
- Ours beside theirs: Gumbel from the golden's file uniforms (official: jax.random), clip
  [1e-7, 1-1e-7] as `file_noise.py`; no straight-through gradient (no training at this rung); f32
  (published bf16); the prerun's 128 uniforms are skipped in the golden, so the port reads call 1 on.
- Toy size only. No speed read. Wheel re-check not run. **Rungs 3, 6, 7 all PASS → J16's done-when met.**

## J16 tail — the PyPI wheel line for rungs 3 / 6 / 7 (11-21): **WHEEL PASS**
Kaggle CPU kernel `jack-cpu-1795286363` (`wheel_kernel kernel 367`, verdict fixed before the run,
`fb0cf19d`), PyPI `mlx[cpu]==0.32.4`, x86_64 / glibc 2.39 / Python 3.13.18; result
`/data/jack-logs/mlx/wheel_367.json`, re-read by `wheel_kernel verdict FILE 367`. All six runs rc 0:
`rung3.py` **2/2** eager AND compiled (mlx 0.32.4), `--planted` **5/5 caught**; `rung6.py` **2/2**,
`--planted` **3/3 caught**; `rung7.py` **2/2** (deter 8.7e-8, post 3.0e-7, prior 7.9e-7 vs the NumPy-64
reference, 0 of 1,024 samples flipped, constant-noise control flips 760), `--planted` **4/4 caught**.
Tests unchanged from the box's conda-forge run. Same limit as the rung 1/2 line: Linux x86_64 wheel,
not the Mac's macOS arm64 binary; rungs 6/7 at toy size; no speed read.

## J21 row 2.3 at `size1m` — rung 7 references HOLD; the port FAILS on the activation (11-24 20:35)

`rung7.py --preset size1m` (new; toy path unchanged, re-run 15/15): the official `size1m` rssm block
(`configs.yaml` L121 over the defaults' L91): deter 512, hidden 64, stoch 32, classes 4, blocks 8,
**act silu** (the toy passed gelu; size1m keeps the defaults' silu). Inputs stay the toy's (tokens 12,
toy action space, batch 4) — a listed difference. The NumPy-64 reference now reads `act` (`rung6.ACT`).
- `--vs-official`: **15/15 REFERENCES HOLD** — deter 1.2e-7, post 5.8e-7, prior 1.2e-6 ×(|ref|+1) over
  64 steps, **0 flips of 8,192 latents**, 0 near-ties; prerun 512 uniforms; constant-noise control flips
  4,870; planted 4/4 caught. Golden `rung7_scan_golden_size1m_e01491f.npz` (2.5 MB, sha256 5f34de14…).
- Known-positive: a gelu reference at size1m FAILS (deter 0.112, 406 flips).
- **MLX port at size1m: 0/2, FAIL** (406 flips vs golden) — the same 406 as the gelu mutant: `mlx_maths`
  hardcodes `_gelu` (L377, L390). Tests now fail for the port; NEXT: the port reads `cfg["act"]`.
- Not yet: the g1v1 golden has no per-step RSSM intermediates, so this golden is the official scan at
  size1m widths on toy inputs, not the trained agent's own pass.
- **11-24 20:50 MLX port at size1m: 2/2 PASS eager AND compiled** — `mlx_maths` now reads `cfg["act"]`
  (`_ACT = {gelu, silu}`, `_layer(..., act)` with no default, so a missing act raises instead of
  silently running gelu). Same commit, all green: `rung7.py --preset size1m` 2/2 (constant-noise
  control 4,870/8,192 flips), `rung7.py` toy 2/2, `--planted` 4/4 at both sizes, `rung6.py` 2/2.
  Rung 7 at size1m is DONE; next rows are the world-model loss rungs.

## J21 — rung 8a: dynamics + representation losses with free bits, size1m (11-25): **PASS**
`experiments/brain_a/rung8a.py`, official `rssm.py` `RSSM.loss` L120-133 + `_dist` L173-176 at e01491f,
run end to end (its own observe scan) on rung 7's size1m params, inputs and file noise.
- `--vs-official`: **16/16 REFERENCES HOLD** (`a3fd3c60`) — official loss pass post/prior/stoch equal the
  rung 7 size1m golden bit for bit; dyn and rep (free 1) mean 1.1e-8, element 1.9e-7 ×(|ref|+1).
- **The published floor never binds here:** at size1m init the raw KL is 2.45–11.63 nats (median 7.62),
  so `max(kl, 1)` is the identity on every element and free 1 alone cannot test the floor. A second
  official pass at `free_nats=7.5` (same params, noise cursor rewound) splits 55.5% of elements and
  matches (4.7e-9 mean). Planted references 5/5 caught. Golden `rung8a_dynrep_golden_size1m_e01491f.npz`.
- Port tests `dynrep_ref`, `dynrep_golden` committed FAILING (`b8aafa5f`, stub); then
  `mlx_maths.rssm_loss` (observe → `kl_dyn_rep`) → **RUNG 8a: 2/2 PASS eager AND compiled** (worst
  6.4e-7 ×(|ref|+1) vs golden, both floors; free-0 control differs by 5.05). `--planted` 4/4 caught
  (no unimix, free bits ignored, KL swapped, free bits on the T-mean); rung 7 size1m still 2/2.
  Logs `/data/jack-logs/mlx/rung8a_{pass,planted}.log`.
- Ours beside theirs: `dyn_ent`/`rep_ent` metrics not ported (metrics, not losses). Inputs the toy's
  (rung 7's listed difference). Wheel re-check: rung 7 size1m + 8a since the 11-21 wheel line — the
  next rung is the third → wheel kernel after it.

## J21 — rung 8b: reward (TwoHot) + continue (Binary) head losses, size1m (11-25): **PASS**
`experiments/brain_a/rung8b.py`, official `agent.py` L51-58/L172-177 + `MLPHead` at e01491f, over rung 7's
size1m deter/stoch (feat 640), reward/term drawn from a fixed seed.
- `--vs-official`: **15/15 REFERENCES HOLD** (`ac9a3074`). Published rewhead outscale 0.0 makes the reward
  logits all zero at init (loss log 255 whatever the MLP), so a second official pass runs every head
  parameter replaced by seeded N(0, 0.3) (`noised_params`, regenerated from shapes). Planted refs 6/6.
- Port tests `rewcon_ref`, `rewcon_golden` committed FAILING (`9db7bb98`, stub); then
  `mlx_maths.rewcon_loss` (`feat2tensor` → `_mlp_head` ×2 → `twohot_loss` / `binary_logp`, `con_target`)
  → **RUNG 8b: 2/2 PASS eager AND compiled** (worst 2.35e-6 ×(|ref|+1) vs golden, noised rew; published
  init zero logits / log 255 reproduced; con-discount control differs by 0.011). `--planted` 6/6 caught
  (no rms, relu for silu, eps 1.0, con without horizon discount, linear bins, stoch before deter);
  rungs 1 (9/9) and 8a (2/2) still pass. Logs `/data/jack-logs/mlx/rung8b_{pass,planted}.log`.
- Ours beside theirs: f32 compute (published bf16); heads of exactly one MLP layer (the published
  rew/con heads); outscale is init-only, the port reads the kernel it is given.
  Wheel re-check: rung 7 size1m, 8a, 8b = the third rung since 11-21 → wheel kernel owed next.

## J21 — rung 8c: decoder VECTOR reconstruction losses, size1m (11-25): **PASS**
`experiments/brain_a/rung8c.py`, official `rssm.Decoder` (L253-306, vector branch) + `heads.symlog_mse` +
`outs.Agg`/`MSE` at e01491f, over rung 7's size1m deter/stoch; obs = walker proprio as dm_control
publishes it: orientations (14,), height () (a scalar key, no Agg), velocity (9,); targets from a fixed
seed spanning |x| 0.01–300 (both symlog regions).
- `--vs-official`: **24/24 REFERENCES HOLD** (`971816b9`; worst 2.68e-6 ×(|ref|+1)), published init and
  seeded N(0, 0.3) params; planted refs 8/8 (incl. layers 2 for 3, symexp of pred vs raw target).
  **Scar:** the decoder concatenates STOCH first, then deter — the reverse of `feat2tensor`. Reusing the
  heads' input function would be a silent bug; planted and caught at both stages.
- Port tests `dec_ref`, `dec_golden` committed FAILING (`963d32f1`, stub); then `mlx_maths.dec_vec_loss`
  (`dec_inp` → 3 × linear/rms/silu → per-key Linear → square(pred − symlog(target)) → `_vec_agg` sum)
  → **RUNG 8c: 2/2 PASS eager AND compiled** (worst 5.03e-6 ×(|ref|+1), noised velocity pred);
  sign-flipped-target control differs by 184. `--planted` 6/6 caught; rungs 1 (9/9), 8a, 8b (2/2) still
  pass. Logs `/data/jack-logs/mlx/rung8c_{vs_official,stub,pass,planted}.log`.
- Ours beside theirs: f32 compute (published bf16); continuous keys only — the `categorical` branch
  (discrete vector keys; none in DMC proprio) and the image branch (BlockLinear + conv) are not ported.
  Wheel re-check now covers rungs 7 size1m, 8a, 8b, 8c (owed since 8b; Kaggle was 5/5 J10 at 01:22).

## J21 — rung 9: the FULL world-model loss on one golden batch, size1m (11-25): **PASS**
`experiments/brain_a/rung9.py`, official `agent.py` L80-83 (rec is per obs key) + L240
`sum([v.mean() * scales[k] ...])` + `configs.yaml` loss_scales at e01491f, over 8a/8b/8c's [4,64] batch.
- `--vs-official`: **29/29 REFERENCES HOLD** (`eb440755`; scales read from the pinned configs.yaml, rep 0.1;
  total 179.2374 init / 231.3479 noised, rel 7.5e-9). Limit: the total is L240's arithmetic over the
  three official goldens, not a fresh official `Agent.loss` call (encoder + imagination unported).
- Port tests `wm_total` (L240 on the official f32 terms vs the golden total, both param sets, con-dropped
  control) and `wm_grad` (**gradient routing** through `kl_dyn_rep` on a [1,2] slice of rung 7's size1m
  post/prior: d/d prior == 1.0 × FD, d/d post == 0.1 × FD, float64 central differences from
  `rung1.REF_kl_grads`; swapped-routing control) committed FAILING (`66cfcc16`, stub); then
  `mlx_maths.wm_loss` → **RUNG 9: 2/2 PASS eager AND compiled**. `--planted` 5/5 caught: sum not mean,
  mean-over-T then sum-over-B, **dyn↔rep swapped (the forward-blind bug — caught only by wm_grad)**, no
  stop-gradients, rep's 0.1 ignored. Rungs 1 (9/9), 7 size1m, 8a, 8b, 8c (2/2) still pass.
  Logs `/data/jack-logs/mlx/rung9_{pass,planted}.log`.
- Ours beside theirs: f32 (published bf16); `wm_loss` iterates the SCALES dict (a scaled key with no loss
  raises), official iterates the losses dict — equal key sets for the world-model part. policy / value /
  repval are not in this rung. Port envs read the scales from the golden (no yaml there).
  Wheel re-check: kernel `wheel_kernel kernel s1m` (rungs 7 size1m, 8a, 8b, 8c) launched 11-25 03:27
  (`jack-cpu-1795577221`); rung 9 rides the next wheel set.
  Wheel READ 11-25 03:40: **WHEEL PASS** on PyPI mlx 0.32.4 (Kaggle CPU, x86_64, py 3.13.18, glibc 2.39):
  rung 7 size1m / 8a / 8b / 8c each 2/2 eager AND compiled; planted 4/4, 4/4, 6/6, 6/6 caught, unpatched
  port passes. Verdict file `/data/jack-logs/mlx/wheel_s1m.json`. Rung 9 not yet on the wheel.
  Rung 9 wheel READ 11-25 05:09 (set `9`, `5e98aa3b`, `jack-cpu-1795583306`): **WHEEL PASS** on PyPI mlx
  0.32.4 (Kaggle CPU, x86_64, py 3.13.18, glibc 2.39): RUNG 9 2/2 eager AND compiled (wm_grad d/d prior
  1.0 × FD el 2.95e-8, d/d post 0.1 × FD el 4.68e-9, swapped-routing control differs 0.235); planted 5/5
  caught, unpatched port passes. Verdict file `/data/jack-logs/mlx/wheel_9.json`. Every rung through the
  full world-model loss now passes on the wheel the Mac will run.

## J25 — rung 10: imagination rollout + lambda-returns, size1m (11-25): **PASS**
`experiments/brain_a/rung10.py`, official `rssm.py` `RSSM.imagine` L94-118 + `agent.py` L190-198 (the
[N, H+1] assembly), `imag_loss` L382-405 / `repl_loss` L448-480 / `lambda_return` L482-490 at e01491f.
- `--vs-official`: **36/36 REFERENCES HOLD** (`c0dd858f`). Imagination on rung 7's size1m params from
  N = 32 starts (its golden's last K = 8 steps), H = 15, file noise, BOTH imagine paths (action array;
  callable policy on the carry, the last action appended): deter 1.0e-7, logit 1.1e-6 ×(|ref|+1) over 15
  steps, 0 flips of 15,360 latents (1 near-tie fed golden); squash binds on 26% of policy actions.
  Returns through the official `imag_loss`/`repl_loss` run for real, with the value heads, policy and
  normalisers stubbed (rungs 11-12): ret 1.4e-5 (contdisc and 1-1/333), repl 2.3e-7; λ 1 == Monte
  Carlo + bootstrap and λ 0 == one-step TD (closed forms). Planted references 13/13 caught.
  Golden `rung10_imag_golden_size1m_e01491f.npz` (2.9 MB, sha256 86c42040…).
- Port tests `imag_ref`, `imag_golden`, `ret_ref`, `ret_golden` committed FAILING (`bfefdd8b`, stub); then
  `mlx_maths.imag_rollout` / `imag_return` / `repl_return` → **RUNG 10: 4/4 PASS eager AND compiled**
  (imagination worst 2.3e-6 ×(|ref|+1) vs golden, 0 flips; returns and weights EQUAL to the official f32
  dump, element 0). `--planted` 6/6 caught (no unimix, actions halved, λ on the bootstrap, disc under
  contdisc, core skipped, policy one step late); rungs 1, 6, 7 size1m, 8a, 8b, 8c, 9 still pass.
  Logs `/data/jack-logs/mlx/rung10_{vs_official,pass,planted}.log`.
- No gradient check at this rung, by reading: every rung 10 output is consumed under a stop-gradient in
  the agent (L195 `sg(imgfeat)` with `ac_grads False`; `sg(weight)`, `sg(adv)`, `sg(target)`), so no
  forward-blind routing lives here; the routing claim is rung 13's whole step.
- Ours beside theirs: f32 (published bf16); N 32 starts (published imag_last 0 → B·T); the toy action
  space; the callable path's actor is a fixed function of the carry (the actor is rung 12); the start's
  logit slot is 0 (official: the replay posterior's logit, which the heads never read — `feat2tensor`
  uses deter + stoch only); normalisers / value denormalisation not ported (rungs 11-12). No wheel kernel
  owed until rung 12 (ORDERS J25).

## J25 — rung 11: the critic loss (value + repval, slow target, slowreg, valnorm), size1m (11-25): **PASS**
`experiments/brain_a/rung11.py`, official `agent.py` L65-68 (val / slowval MLPHeads, SlowModel rate 0.02
every 1, valnorm impl 'none'), `imag_loss` value part L396-422, `repl_loss` L460-474, loss scales
value 1.0 / repval 0.3, `utils.py` SlowModel.update, at e01491f.
- `--vs-official`: **25/25 REFERENCES HOLD** (`22067e03`). The official imag_loss / repl_loss run for real
  with REAL value and slowvalue heads (3 layers, units 64, symexp two-hot 255) and the real Normalize;
  only the policy and the ret/adv normalisers are stubbed (rung 12). Published init: logits 0, pred 0.
  Noised heads (N(0, 0.3), output kernel 0.03 — at 0.3 values reached std 6.7e6): val 1.5e-5, ret 2.9e-5,
  value 1.8e-6, repval 2.8e-7 ×(|ref|+1). **Gradient** of the critic total on 33 val elements vs f64
  central differences WITH THE TARGETS HELD (sg(tar_padded), sg(slowvalue.pred())): worst 2.4e-5; the
  same reference without the stop-gradient is 2.6× off and is caught (the first draft had that bug).
  slowval grads exactly 0. SlowModel.update to 5.3e-8. Planted references 6/6 caught.
  Golden `rung11_critic_golden_size1m_e01491f.npz` (1.4 MB, holds the official grads).
- Port tests `critic_ref`, `critic_golden`, `critic_grad`, `slow_update` committed (`4b78ebcd`) at 1/4 —
  `slow_update` passed already (it tests rung 3's `slow_update`, not new code); then
  `mlx_maths.critic_loss` → **RUNG 11: 4/4 PASS eager AND compiled**: total 1.2e-7 vs golden, grads
  3.3e-6 × max|g| vs the official jax.grad on all 11 val params, slowval grads 0. `--planted` 6/6
  caught (no stop-gradient → caught only by the grad test, 1.93×; λ on the bootstrap; 2 layers;
  slowreg dropped; repval scale 1.0; relu for silu); rungs 1, 8b, 9, 10 still pass.
  Logs `/data/jack-logs/mlx/rung11_{vs_official,pass,planted}.log`.
- Ours beside theirs: f32 (published bf16); valnorm 'none' only and slowtar False only (the published
  config; 'perc'/'meanstd' valnorm not ported); contdisc True only; the slow update is rung 3's function
  applied per param (the agent calls it once after each optimiser step — its place in the step is
  rung 13's); boot for repval is a fixed array here (in the agent, the imagined return at each start).

### Rung 12a — return normaliser (perc) + advantage: PASS (11-25)
`experiments/brain_a/rung12a.py`, official `embodied/jax/utils.py` Normalize (perc, rate 0.01, limit 1,
5/95, debias False) + `agent.py imag_loss` L407-411. References vs official 9/9 (`ba33d5e5`): 6 carried
updates, spreads 1..1000..5, limit branch both ways, worst 2.6e-7; planted refs 6/6 caught.
Tests committed failing `d6e7aeae`; port `mlx_maths.retnorm_perc` / `advantage`: tests
`retnorm_ref`, `retnorm_golden` eager+compiled (worst 1.95e-7 / 1.35e-7); `--planted` 5/5.
No gradient test: every output is under stop-gradient (the officials' sg). Rung 12b (actor heads'
logp + entropy, the policy loss) still owed.

### Rung 12b — policy heads (bounded_normal + categorical) logp/entropy + policy loss: PASS (11-25)
`experiments/brain_a/rung12b.py`, official `heads.py` Head.bounded_normal / categorical (Linear layers
tapped), `outs.py` Normal / Categorical, Agg, `agent.py imag_loss` L412-416. References vs official
12/12 (`826b1f74`): forward 3.41e-7, grads jax.grad vs f64 diffs 4.69e-6, planted refs 8/8. Finding:
official `Head.categorical` ignores unimix (only `onehot` uses it) — the published 0.01 never touches a
discrete policy. Tests committed failing `2a749221`; port `mlx_maths.policy_loss`: tests `policy_ref`,
`policy_golden`, `policy_grad` eager+compiled (fwd 4.53e-7; grads 4.68e-6 vs official, 3.8e-7 vs f64);
`--planted` 5/5 incl. logpi under sg (forward-identical, caught only by the grad test). Rung 12 whole;
wheel kernel then rung 13 owed.

## J25 — the PyPI wheel line for rungs 10 / 11 / 12a / 12b (11-25): **WHEEL PASS**
Kaggle CPU kernel `jack-cpu-1795623840` (`wheel_kernel kernel 10_12`, `cdd0db67`, verdict fixed before the
run, planted 3/3 refused on box outputs). PyPI mlx 0.32.4, x86_64, py 3.13.18, glibc 2.39. All eight runs
rc 0: RUNG 10 4/4, RUNG 11 4/4, RUNG 12a 2/2, RUNG 12b 3/3, each eager AND compiled; planted 6/6, 6/6, 5/5,
5/5 caught, unpatched port passes. Verdict file `/data/jack-logs/mlx/wheel_10_12.json`. Same limit as
before: Linux x86_64 wheel of the same version, not the Mac's macOS arm64 binary. Next: rung 13.

### Rung 13a — the whole optimiser + slow-critic step over the official size1m tree: REFERENCES HOLD (11-25)
`experiments/brain_a/rung13.py` (line fixed `6ad44c87` before the run). Golden: one Kaggle CPU kernel
`jack-cpu-1795626717` (418 s), row 2.1's run A exactly + `g1_probe --dump-params` (every agent variable
before/after calls 1, 2, 6, 8), `/data/jack-logs/mlx/rung13/R13.npz` (75 MB, /data only). Read on the box
(NumPy only), log `/data/jack-logs/mlx/rung13_vs_official.log`:
- (R) all 81 gradients at all 4 calls BIT-EQUAL to the frozen row-2.1 golden — the same step, re-made.
- (S) all 647,705 weights (81 tensors, 7 modules, ONE optimiser) at every call; counts +1; ≥ 86% moved.
- (W) slowval (11 tensors) = 0.02 × the POST-step critic + 0.98 × slowval: the slow update follows the
  optimiser step (agent.py L139-142). (U) nothing else moves except opt/step, opt counts, retnorm.
- (P) planted 7/7 caught: global-norm clip, slow from the pre-step critic, no bias correction,
  momentum before rms, state not carried (calls 2+), slow mix swapped, lr × 10.
- Three things named on the way (the reference fixed, never the weight line): (1) the golden is EAGER
  (jit off, to capture grads), so every op rounds in f32 — momentum cancels from call 2 on and a
  float64 reference misses; `REF_step32` rounds as theirs. (2) **Eager JAX computes `0.999 ** int32(t)`
  by square-and-multiply in f32 (78/78), not NumPy's correctly-rounded power; jitted optax equals
  NumPy.** The bias correction moves up to 2e-5 at t = 3, 6, 7, 8 (β₁: t = 7). The port trains jitted:
  it must follow the JITTED value, and a port test against THIS golden must use the eager one.
  (3) moment tolerance re-registered to the operand scale (a cancelled f32 sum carries 1–2 ulp of its
  terms; 12 / 197 / 122 elements otherwise unreachable by any f32 implementation).
- Ours beside theirs: f32 (published bf16); warmup 0 (row 2.1's listed difference; the ramp is rung 3's);
  retnorm's update not checked here (rung 12a's; its inputs are not dumped). Next: the port's test —
  `mlx_maths.opt_update` + `slow_update` over this tree from `pre` + grads → `post`, eager and compiled;
  then rung 13b (the port's own whole-loss gradients vs these 81).

### Rung 13a — the PORT's whole step over the same tree: PASS (11-25, `709fa34e`)
`rung13.py --port` (jack-mlx, box, seconds): `mlx_maths.opt_update` + `slow_update` from each call's `pre`
+ official grads → official `post`, by THE LINE's (S)/(W) clauses. **Calls 1/2/6/8: 0 miss on 81 grads +
11 slow tensors, eager AND compiled (mlx 0.32.4); worst weight 0.997 of the line.** First read FAILED —
the AGC norms were f32 sums (~1e-5 off on size1m tensors; nu ∝ clip²): now summed in f64 on the CPU
stream, then cast; `(1-b2)*(u*u)` as opt.py. The test injects the golden's EAGER bias correction
(`bias=`, test-only); the port's own `mx.power` equals the correctly rounded (jitted) f32 power for
t 1..39 and, read against this eager golden, misses only at calls 6/8 (39/33 tensors) — as predicted.
Rung 3 still 2/2 + planted 5/5; rung 11 4/4. Next: rung 13b, then the wheel kernel.
