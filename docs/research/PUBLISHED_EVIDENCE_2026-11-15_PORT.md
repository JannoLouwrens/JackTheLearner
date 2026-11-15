# Build the MLX port, rent the GPU

Porting the official Dreamer agent to Apple's MLX is a medium-sized, well-defined job, and almost all of it can be written and proven before the Mac arrives. For Jack's case (74 numbers in, 8 continuous actions out, the two smallest size presets, no images) the port means translating **about 1,170 lines of dense JAX code**, writing **about 430 lines of fresh plumbing** and **about 700 lines of tests**. That is roughly **2,300 lines**. A fluent engineer would need an estimated **17–24 working days**, and the project's agents an estimated **60–110 fifty-minute slots**. MLX runs on Linux, so the port can be built on the server (inside a container), checked against "golden files" produced once from the official code on a free Kaggle CPU session, and taken up **13 of a 14-rung proof ladder before any Apple hardware exists**. Only its Mac speed, its agreement on Apple's graphics chip and the last rung (matching learning curves) wait for the Mac or for GPU time. One unknown could sink it: **nobody has measured how long MLX takes to compile Dreamer's unrolled training step, or how much memory that step needs**. So a cheap first spike of 6–8 slots measures exactly that before any port code is written. The bigger finding is about money. **An NVIDIA card rents for $0.16–0.89 an hour in November 2026, and the official code runs on it unchanged.** The large jobs price out at roughly **$5–18** to reproduce published results, **$4–34** for the decisive test with the small brain, **$72–409** for it with the 9.7-million-parameter brain, and **$115–511 a month** for a continuous lifetime. So no paid run needs the port, and the port should never sit on the critical path. It is worth building for two things. First, it makes the Mac a fast, always-on Dreamer lane: a projected **6–25 training steps a second** for the small brain, against 3–12 for the best published Mac route, for long lifetimes that would cost hundreds of dollars a month to rent. Second, people who receive a trained creature could run it on their own Apple machines, and that needs only the acting part, **about a quarter of the port**. Build it as a side lane, behind the spike, with a stop rule.

## How to read this report

Every number is labelled by kind. **Read in code** means the researchers read it in the official repository at the pinned commit. **Checked here** means it was tried on the project's server on 15 November 2026. **Market** means live rental listings seen that day. **(inference)** means arithmetic or judgement from those facts that nobody has measured. Every estimate is a range, and the text says what it rests on.

| Term | Plain meaning |
|---|---|
| Dreamer (DreamerV3) | The world-model brain ("brain A"). It learns to predict its own next senses and reward, then practises inside that imagined world |
| JAX | Google's maths library. The official Dreamer is written in it. On a Mac it runs only on the main processor |
| MLX | Apple's own maths library for its chips. It also runs on Linux, on the processor or on an NVIDIA card |
| Port | A rewrite of the same program for a different library, meant to compute exactly the same thing |
| Parity | Proof that the port computes the same numbers as the original, to a stated tolerance |
| Golden file | A file of exact inputs and outputs recorded once from the original program, which the port's tests compare against |
| Rung | One test on the proof ladder. Each rung checks one piece, and is passed before the next piece is built |
| Preset | A named size of the brain. `size1m` has about 0.65 million adjustable numbers (parameters), `size12m` about 9.8 million |
| Latent | The brain's internal summary of the moment: 32 small multiple-choice variables |
| Gradient step | One adjustment of the brain's weights from one batch of replayed experience. Training speed is counted in these |
| Compile | MLX turning a whole calculation into one fast program before running it. Dreamer's loops get "unrolled" into one long chain first |
| T4 | The free NVIDIA graphics card on Kaggle (about 30 hours a week, used up until 21 November) |
| CUDA, Metal | The software for NVIDIA cards (CUDA) and for Apple's graphics chip (Metal) |
| Container | A sealed mini-system on the server. MLX needs one there because the server's system library (glibc 2.34) is older than MLX needs (2.35) |
| Slot | One 50-minute work period of an hourly agent, with no memory of earlier slots |

## About 2,300 lines to write, and three corrections to what the team was told

**The official agent is small but dense.** The reference is `danijar/dreamerv3` at commit `e01491f` (4 October 2026). Its agent files have not changed since December 2024, so it is a stable target ([repository at e01491f](https://github.com/danijar/dreamerv3/tree/e01491fad6434b2245a3b8ca201dd7faedcc458c)). The repository holds 10,360 lines of Python. The algorithm itself sits in about 1,130 of them, across eight files: the agent's losses and training step, the recurrent world model (`rssm.py`), and five small library files for layers, output heads, probability distributions, the optimiser and the return normaliser (read in code). The other roughly 9,000 lines are plumbing: multi-machine run loops, a disk-backed replay store, logging, image handling and environment suites. For vector senses and continuous actions, a faithful port translates **about 1,170 lines** and replaces the plumbing with **about 430 simpler lines**: an in-memory replay, a single-world driver, a training loop and a checkpoint. A golden-file generator and the parity tests add **about 600–900 lines**. Faithful PyTorch ports of the same agent landed at 2,300–2,800 lines plus tests, which supports the total (read in their repositories, e.g. [sheeprl PR #377](https://github.com/Eclectic-Sheep/sheeprl/pull/377)).

Line counts understate the care needed. One line of the official code often holds a "stop-gradient" (a marker that blocks learning from flowing backwards through a value), a slice and a mask. About 330 lines (the losses, imagination, the return calculation, the replay start-up and the recurrent core) are where every known class of porting bug lives (inference from the bug record below). Five libraries the official code leans on also hide logic that must be ported. They cover parameter naming and random-number order, the optimiser's bias correction and warm-up schedule, a truncated-normal initialiser and a sampling trick, the percentile function, and the training-ratio counter (read in code).

**The brain sizes for Jack, counted from the code:** `size1m` has **647,705** trainable parameters and `size12m` **9,765,209**. Each also keeps a slowly-updated copy of its critic (66,111 and 853,503). The counting rule reproduces an independently reported count of the official 12M world model exactly (5,979,648 for an image input; [r2dreamer issue #10](https://github.com/NM512/r2dreamer/issues/10)). It also reproduces TorchRL's stated 640,867 for its Walker version of `size1m` (inference). The first official print-out the port must reproduce is this per-module table.

Three earlier statements need correcting. **First, the latent is not "32 × 32" at Jack's sizes.** It is always 32 variables, but each has **4 choices in `size1m` and 16 in `size12m`**. Only the 200M default has 64 ([configs.yaml L120–128](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/configs.yaml#L120-L128)). A port built from the paper's "32 × 32" would not match. **Second, the 12M recurrent width is 2,048, not the paper's 1,024** ([issue #131](https://github.com/danijar/dreamerv3/issues/131)). **Third, the earlier figures of about 0.64M and 9.7M parameters assumed a 30-number input.** At 74 numbers they barely move (647,705 and 9,765,209). The earlier "about 850 lines of agent code" counted only the agent files, not the library code that must also be translated.

## MLX has the tools but none of Dreamer's special parts, and fifty-odd details must match exactly

**MLX in November 2026.** The current release is **0.32.4** (14 October 2026). MLX has had about 20 releases in 15 months, is still before version 1.0, and changes behaviour in minor releases ([PyPI](https://pypi.org/project/mlx/0.32.4/); [release notes](https://github.com/ml-explore/mlx/releases)). It has everything a training step needs at the level of basic tools: automatic gradients, a compiler, stop-gradient, JAX-style random keys, sorting, scatter-and-gather and save formats (checked here, in a container). It lacks several things the port must hand-write. There is **no loop primitive (`scan`)**, so the 64-step memory pass and the 15-step imagination become Python loops unrolled into one long compiled chain. There is **no percentile function**: write it as sort plus linear interpolation, tested against NumPy. There is **no LaProp optimiser**, the one Dreamer uses. Its built-in recurrent cell is the PyTorch kind, which is a different cell. Its normalisation layers default to a small constant of 1e-5 where Dreamer uses 1e-4 (checked here; [recurrent.py](https://raw.githubusercontent.com/ml-explore/mlx/main/python/mlx/nn/layers/recurrent.py)). Two recent fixes matter to a parity port. 0.32.0 corrected a wrong gradient in layer normalisation and in sorting. 0.32.1 fixed random state captured inside compiled code ([release notes](https://github.com/ml-explore/mlx/releases)). Pin 0.32.4, and re-run every gradient test on any upgrade.

MLX's compiler also has a trap that fits Dreamer badly: **anything a compiled function uses without receiving it as an argument is frozen as a constant**. A return normaliser or slow critic held as hidden state would silently stop updating. Random sampling inside a compiled step works only if the random state is passed in explicitly ([MLX compile guide](https://ml-explore.github.io/mlx/build/html/usage/compile.html)).

**The numerical contract.** The researchers read the official code line by line and listed **62 details** that decide whether a port computes the same function. The eight most likely to go wrong all differ from the paper or from the code's own defaults.

| Detail | What the code does | Where |
|---|---|---|
| Two-hot (how reward and value are predicted, as weights on two neighbouring "bins") | Done in raw space over 255 exponentially spaced bins. The paper describes squashed values over evenly spaced bins. The two agree at the start and on targets exactly on a bin, then drift apart | [outs.py L311–330](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/embodied/jax/outs.py#L311-L330) |
| Defaults | The configuration file overrides the code's built-in defaults (activation, layer counts, slow target, debiasing, optimiser constant). Port the configuration, never the defaults | [configs.yaml L85–118](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/configs.yaml#L85-L118) |
| Recurrent cell | 8 blocks split their gates inside each block (reset, candidate, update); update gate shifted by −1; weights scaled by the full input width, not the block's | [rssm.py L141–158](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/rssm.py#L141-L158); [nets.py L187–197](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/embodied/jax/nets.py#L187-L197) |
| Where learning flows | Reward and replayed-value losses train the world model too; imagined futures do not | [agent.py L196](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/agent.py#L196) |
| Replayed value target | Built on imagined returns; the return function ignores one of its own arguments | [agent.py L482–490](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/agent.py#L482-L490) |
| Optimiser | First update has learning rate exactly 0, then a 1,000-update warm-up the paper omits; constant 0.999 where the paper says 0.99 | [agent.py L342–379](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/agent.py#L342-L379); [opt.py L126–143](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/embodied/jax/opt.py#L126-L143) |
| Return scale | Starts from zero, not debiased; a floor of 1 holds for the first few hundred updates | [utils.py L39–91](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/embodied/jax/utils.py#L39-L91) |
| Previous action | Shifted one step against the stored actions; the very bug the author fixed in May 2024 | [commit 29eb964](https://github.com/danijar/dreamerv3/commit/29eb964e2918a3f4db04086f7f51b60388e97f3d) |

Two smaller points matter for continuous actions. The actor's samples are **not squashed or clipped by the agent**: only the environment wrapper clips, and the replay keeps the unclipped value. The MLX research note's suggested "tanh/clip bound" would therefore be a mismatch.

One detail is fragile under any compiler. The predicted value is a sum over 255 bins reaching about ±485 million, and it should be exactly zero at the start. **Under compilation the official code itself misses zero**, because a fused multiply-add changes the rounding ([issue #245](https://github.com/danijar/dreamerv3/issues/245)). TorchRL's reward head predicted about 0.15 at the start for the same reason ([pytorch/rl #4290](https://github.com/pytorch/rl/pull/4290)). The port should cancel mirrored probabilities before multiplying, as r2dreamer now does ([commit 7a93691](https://github.com/NM512/r2dreamer/commit/7a93691)), and test for an exact zero under `mx.compile`. A consequence for the golden files (inference): generate them with the official compiler turned off. Otherwise this rounding alone could exceed the tolerances below.

**Known answers at the start**, worked out from the formulas (inference): predicted reward and value exactly 0; reward and value losses exactly ln 255 = 5.541263; actor spread about 0.8927 per action dimension; return scale 1; and no weight changes on the first update.

## Earlier ports broke on bookkeeping more often than on maths

Every serious port of this agent needed months of fixes after release. The researchers sorted the documented bugs of five ports and the official code into two classes. There were **about 18 maths mistranslations and about 20 infrastructure bugs**, and the infrastructure bugs lived longest. sheeprl stored "episode over" for both death and time-outs for about eight months ([sheeprl #229](https://github.com/Eclectic-Sheep/sheeprl/issues/229)). r2dreamer's in-place replay write corrupted about 1.6% of stored actions for about eight months ([r2dreamer #9](https://github.com/NM512/r2dreamer/issues/9)). Its latent write-back silently did nothing ([commit e29b7e5](https://github.com/NM512/r2dreamer/commit/e29b7e5)). dreamerv3-torch's reset flag did nothing because the reset state went to a local variable ([PR #14](https://github.com/NM512/dreamerv3-torch/pull/14)). Resuming from checkpoints broke repeatedly: in sheeprl the restored training counter forced a burst of catch-up updates ("We never found a beautiful solution for this"), and the optimiser's memory came back at lower precision ([#273](https://github.com/Eclectic-Sheep/sheeprl/issues/273); [PR #392](https://github.com/Eclectic-Sheep/sheeprl/pull/392)).

The maths bugs cluster in six places. **Two-hot and its decoding** comes first: NM512 called it one of "the 2 factors [with] strongest effect against performance" ([dreamerv3-torch #2](https://github.com/NM512/dreamerv3-torch/issues/2); [TorchRL #4065](https://github.com/pytorch/rl/pull/4065)). Then **the 1% mixing and the KL term** computed on the wrong logits ([r2dreamer 7a93691](https://github.com/NM512/r2dreamer/commit/7a93691)). Then **initialisation**, including TorchRL scaling the block weights by the per-block width ([pytorch/rl #4128](https://github.com/pytorch/rl/pull/4128)). Then **return indexing and weights** ([dreamerv3-torch #16](https://github.com/NM512/dreamerv3-torch/issues/16)). Then **the return-scale floor**, which RLlib still gets wrong ([ray #61600](https://github.com/ray-project/ray/pull/61600)). Last, **actor numerics** that gave NaN log-probabilities ([sheeprl 191c2e2a](https://github.com/Eclectic-Sheep/sheeprl/commit/191c2e2a)). Every one of these maps onto a rung of the ladder below.

The two most faithful recent ports reached fidelity only after adopting one method: load the same weights and the same batch into both programs and compare losses and gradients. sheeprl's October 2026 rewrite says its "losses, gradients and latent states match the official implementation's on the same weights and batch", and that "every change has a test that fails without it" ([PR #377](https://github.com/Eclectic-Sheep/sheeprl/pull/377); [PR #372](https://github.com/Eclectic-Sheep/sheeprl/pull/372)).

Timelines give a calibration. dreamerv3-torch took about five weeks from first commit to its "modified based on author's implementation" correction. sheeprl and TorchRL needed two to three months of fix bursts by experienced maintainers who had no golden harness at the start (read in their histories). No port states person-hours.

There is **no prior MLX port to borrow from**. The only MLX Dreamer found, `mathDR/dreamer-mlx`, is one week old, 1,774 lines, "with tweaks", and has no verification. No reinforcement-learning agent of any kind was found in MLX. The lesson for Jack's port is that a numerical test of the loss on a fixed batch cannot see bookkeeping bugs. Bookkeeping needs its own rung, written from the code and tested with planted data.

## A fourteen-rung ladder proves the port before it trains

Nobody has published a parity harness for any Dreamer port. The ladder below is assembled from model-conversion practice and reinforcement-learning practice (inference). Hugging Face's conversion guide asks for layer-by-layer comparison of intermediate outputs on copied weights, to 1e-3, and its small-model cross-framework tests use 1e-5 ([HF guide](https://huggingface.co/docs/transformers/add_new_model); [test_modeling_flax_common.py](https://github.com/huggingface/transformers/blob/v4.40.0/tests/test_modeling_flax_common.py)). On the reinforcement-learning side, "The 37 Implementation Details of PPO" pinned its reference at one commit and compared curves on three tasks × three seeds ([ICLR blog 2022](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)). Stable-Baselines3's port check found one algorithm off because of where the optimiser's tiny constant sat ([SB3 #48](https://github.com/DLR-RM/stable-baselines3/issues/48)).

**How the golden files are made.** They are produced once, by a lightly patched official agent. Keep the patch under about 60 lines and store it with the files. Run it on a CPU, in full 32-bit precision, with the compiler off. On a GPU, JAX quietly uses lower-precision matrix maths ([JAX docs](https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html)). The patch replaces the two random draws, latent sampling and action sampling, with noise read from a file, so that both programs see identical randomness. Random numbers can never be matched across libraries: MLX "follows JAX's PRNG design" but promises no equal bits ([MLX random](https://ml-explore.github.io/mlx/build/html/python/random.html)). For the same reason, starting weights cannot be reproduced by seeding, so the port imports the official weights by name. MLX stores linear weights transposed relative to JAX. The decoder takes its inputs in the order [latent, memory] while the heads use [memory, latent].

A full set of golden files for `size1m` is about 50–100 MB (inference). The tests then run offline, with no JAX installed.

| # | Rung | Pass condition (32-bit) | Golden file? |
|---|---|---|---|
| 0 | Names, shapes, per-module counts | exact; 647,705 / 9,765,209 | yes |
| 1 | Pure maths: squashing, bins, two-hot, 1% mix, KL with free bits, sampling gradient, normal and coin-flip losses | 1e-6; uniform and mirrored decode exactly 0, also compiled | no (NumPy reference) |
| 2 | Returns; percentile normaliser over 20 steps, no debias | relative 1e-6 | no |
| 3 | Optimiser: clipping, scaling, momentum over 5 steps; warm-up values at 0, 1, 999, 1000, 1001 | relative 1e-6; schedule exact | yes, once |
| 4 | Initialiser statistics; zero heads exactly zero | within 2% | no |
| 5 | Layers with imported weights | 1e-5 | yes |
| 6 | One recurrent step | 1e-5 | yes |
| 7 | 64-step memory pass with file noise | 1e-4; no sample flips (on a near-tie, feed the golden sample) | yes |
| 8 | World-model losses | relative 1e-5 on means | yes |
| 9 | Imagination, returns, scale, actor and critic losses | relative 1e-4 | yes |
| 10 | Gradients per loss term | relative 1e-4 (1e-3 for recurrent parts); cosine ≥ 0.9999 | yes |
| 11 | Five optimiser steps; step 1 changes nothing | 1e-3 on updates, compared only where the gradient is not near zero; 1e-6 on weights | yes |
| 12 | Bookkeeping: action alignment, reset flags, start latent, replay queue order, latent write-back, death vs time-out, training-ratio count, resume bit-identical | exact | no (written from the code) |
| 13 | Known answers at the start (ln 255, 0.8927, scale 1) | exact or 1e-4 | no |
| 14 | Learning curves: official against port, same preset, ≥ 5 seeds × 2–3 tasks | no significant difference (confidence intervals overlap) | needs official training runs |

The tolerances rest on MLX's own float32 tests, which use 1e-5 to 1e-6 per operation ([mlx tests](https://github.com/ml-explore/mlx/blob/main/python/tests/test_blas.py)), loosened for depth. Two of them are estimates to be calibrated: the 1e-4 for the 64-step pass, and the masked comparison on rung 11. That rung needs a mask because early optimiser updates are nearly "learning rate × sign of gradient", so near-zero gradients flip sign harmlessly (inference).

Run the ladder on every backend: Linux CPU, then Apple's graphics chip, then CUDA if used. Different chips are expected to agree to about 1e-5–1e-6 per operation, not bit for bit. That has not been measured for MLX anywhere. The port matches the 32-bit golden files first. Lower precision is tried only after rung 14 passes in 32-bit, followed by a repeat of rung 14.

## Linux can write and prove the port; only speed and the last rung wait for the Mac

MLX publishes Linux packages for the processor (x86 and ARM) and for NVIDIA cards, and all of them need glibc 2.35 or newer ([MLX install](https://ml-explore.github.io/mlx/build/html/install.html)). On the project's server (glibc 2.34) the package installs but fails to load. **Inside a Debian 12 container on the same server it works** (checked here). In that container the checks covered a matrix product, a correct gradient, compilation, 16-bit and 64-bit numbers, keyed random numbers, and a 64-step unrolled recurrence compiled with gradients in about 0.01 s on one capped core. That toy is far smaller than Dreamer, so it says nothing about the real compile time.

| Machine | What it does for the port | Limits |
|---|---|---|
| ARM server, in a container capped at 1 core and 2 GB | Writing code, rungs 0–6 and 12–13 on `size1m` and the tiny `debug` preset | Shared box: light runs only, never the 12M size |
| Kaggle CPU (4 x86 cores, 12 h, up to 5 at once) | Golden-file generation with official JAX; rungs 7–11; the compile spike at both sizes; smoke training | Kaggle's system library and installs not yet checked for MLX; far too slow for full-ratio training |
| Free T4 (from 21 November) | Optional one-hour check that `mlx[cuda12]` loads and runs the step | Quota is wanted for brain A's reproduction; the T4 is the minimum card MLX's CUDA backend supports ([install page](https://ml-explore.github.io/mlx/build/html/install.html)) and its driver was not checked |
| Mac mini M6 (about late December) | Re-run rungs 0–13 on Apple's graphics chip; real speed, compile time and memory; the MLX side of rung 14 | — |

So: **written and proven equal before the Mac — yes, through rung 13. Trained to published-curve parity before the Mac — mostly no.** The official JAX code does about 0.5–2 `size1m` steps a second on a 4-core CPU (inference). A 250,000-step training run is therefore 35–140 hours, beyond a 12-hour session. Short smoke runs on Kaggle (the `debug` preset, a few thousand steps) show whether losses fall, and that is all. MLX on the T4 could in principle train it. Its speed there has never been published, its CUDA backend only passed all its own tests in May 2026, and missing operations crash rather than fall back ([MLX discussion #2422](https://github.com/ml-explore/mlx/discussions/2422)). Treat the T4 route as a bonus, not the plan.

## A one-day spike decides whether to continue, then each piece waits for the test below it

The cheapest stage that shows whether the port can work at all is **stage 0: four measurements, about 6–8 slots, before any port code is written.** First, does MLX install and run in the server container and on Kaggle? Second, does a shape-faithful dummy training step compile, and how long does that take? The dummy uses random weights but has the real `size1m` and `size12m` shapes: the 8-block recurrent cell, 64-step memory pass, 15-step imagination from 1,024 starts, and full gradients. Third, how long does a steady step take, and what is the peak memory? Fourth, does the official agent run on Kaggle CPU in 32-bit with file noise, and does it write the same golden file twice?

The compile time and memory of the full unrolled step are the one risk no source answers. The step has roughly 10,000–100,000 operations once unrolled (inference). If compile time or memory is too large, the fallback is to compile one time step and loop it in Python, at some speed cost. **Gate G0** (the thresholds are this report's choice, not published): if `size12m` compiles in under 10 minutes and peaks under 8 GB on CPU, proceed. If not, measure the per-step fallback. If that also fails, stop and report. **Gate G1**: the golden file must be identical on two runs. Without that, nothing downstream can be trusted.

**The build plan for the agents.** Each step fits one or two 50-minute slots. Each ends with a test that says it is done. No step starts before the test of the step it depends on passes. Every step's order should name the commit (`e01491f`), the MLX version (0.32.4), the golden-file set and the rung. The order should also point at the section of the numerical contract to copy. The contract and its line links are in the research notes at `/data/jack-research/research_notes/Porting Dreamer to Apple MLX/dreamer_port_spec.md`, section 2. Agents have no memory between slots, so each slot also ends by writing its result and the next step into the work log.

| Step | Slots | Build | Done when |
|---|---|---|---|
| **Stage 0: can it work?** | | | |
| 0.1 | 1 | Container recipe (Debian 12, Python 3.12, `mlx[cpu]==0.32.4`, NumPy), capped at 1 core and 2 GB; smoke script | Smoke script prints versions; gradient, compile, 64-bit all pass |
| 0.2 | 1–2 | Kaggle CPU notebook: install MLX and the official code at `e01491f` (JAX on CPU); run the official `debug` preset for 200 steps on a dummy 74-in, 8-out world | Both libraries import; official losses logged; library versions recorded |
| 0.3 | 2 | Throwaway compile spike at both sizes (server: `size1m` only; Kaggle: both) | Compile time, steady step time and peak memory logged; gate G0 decided |
| 0.4 | 2–3 | Golden generator v0: under-60-line patch (CPU, 32-bit, compiler off, noise from file, dump hooks) on `debug` | Parameters by name plus one step's intermediates saved; two runs give identical files (gate G1) |
| **Stage 1: the leaves (no JAX needed)** | | | |
| 1.1 | 1 | Frozen configuration (merged defaults plus `size1m`/`size12m`) and its test against `configs.yaml` | Every value matches, including the class-default traps |
| 1.2 | 1–2 | Rung 1 tests, with NumPy 64-bit references | Tests exist and fail |
| 1.3 | 1–2 | Maths module: squashing, bins, two-hot with mirrored-difference decode, distributions | Rung 1 passes, eager and compiled |
| 1.4 | 1–2 | Rung 2 tests, then returns, percentile normaliser, slow-critic average | Rung 2 passes |
| 1.5 | 2 | Optimiser golden (warm-up values, 5 steps) from the generator; then clipping, scaling, momentum, warm-up | Rung 3 passes; first update has learning rate 0 |
| 1.6 | 2 | Rung 4 test, then initialiser and layers; rung 5 with imported weights | Rungs 4 and 5 pass |
| 1.7 | 1 | Name map and loader for official weights (transposes, decoder input order) | Rung 0 passes at both sizes |
| **Stage 2: the model** | | | |
| 2.1 | 2–3 | Golden generator v1 at `size1m`: everything for rungs 6–11 | Full set saved; repeat run identical |
| 2.2 | 2 | One recurrent step: core, posterior, prior, reset masking | Rung 6 passes |
| 2.3 | 2 | 64-step memory pass with a sampler that reads file noise or forces golden samples | Rung 7 passes |
| 2.4 | 2 | Encoder, decoder, heads; world-model losses | Rung 8 passes |
| 2.5 | 2–3 | Imagination, returns, scale, actor, critic and replayed-value losses with exact stop-gradients | Rung 9 passes |
| 2.6 | 2 | Gradients per loss term over the seven trained modules | Rung 10 passes |
| 2.7 | 2 | Training step: replay start-up, optimiser, slow critic, write-back output | Rungs 11 and 13 pass |
| 2.8 | 1 | Acting step (reset, sample latent, sample action) | One-step golden matches |
| 2.9 | 1–2 | Compile the real step, state passed explicitly; re-run rungs 7–11 compiled | Rungs pass compiled; speed logged |
| **Stage 3: the plumbing (tests first)** | | | |
| 3.1 | 2–3 | Replay tests with planted data (action_t = t), then the replay: 65-step samples, stored latents, online queue, write-back, first-in-first-out eviction | Rung 12 replay items pass |
| 3.2 | 1–2 | Driver and wrapper tests (death vs time-out, zeroed action at episode end, unclipped storage), then the driver | Rung 12 driver items pass |
| 3.3 | 1 | Training-ratio loop test, then the loop | Updates = floor(steps × ratio ÷ 1024) ± 1; start at 1,024 items |
| 3.4 | 2 | Resume test, then the checkpoint: weights, optimiser memory and counters, slow critic, normaliser, replay with latents, ratio counter, random state | Save, resume, one update: bit-identical to an uninterrupted run |
| 3.5 | 1–2 | Smoke training on Kaggle: `debug`, then `size1m` on a small control task | Losses fall; known answers at step 0; no NaN |

The plan's counts add up to **about 40–55 slots of build**. Add a debugging reserve of 50–100% for the steps where real float drift and real bugs look alike: rungs 7–11, and the first compiled run. That gives **60–110 slots** (inference).

Two side branches are worth naming. **A policy-only port for owners' devices** needs steps 1.1–1.3 (part), 1.6–1.7, 2.2, the encoder and actor parts of 2.4, and 2.8. That is roughly 15–25 slots, and it can be the first useful thing the port delivers (inference). **The MLX-on-T4 check** is one slot, run only if the brain A schedule leaves a spare T4 hour.

**What to leave out at first:** the image encoder and decoder, discrete actions (whose official 1% mixing has an open bug, [issue #219](https://github.com/danijar/dreamerv3/issues/219)), multi-device code, the logging viewer, prioritised replay and 16-bit loss scaling. **What to keep from the first day**, though it looks like plumbing: stored latents with write-back, the online queue, the 65-step sample layout and the action alignment. They change what the loss sees.

## How long: two to five weeks of agent work, then one to two weeks of training

| Piece | Range | What it rests on |
|---|---|---|
| Build through rung 13 | 60–110 slots | step count above; the notes' 17–24 engineer-days (MLX note: 15–25); debugging at rungs 7–11 is the largest uncertainty |
| Calendar for that | about 1–4 weeks at 4–10 slots a day for this lane | the Director's allocation, which this report cannot know |
| Mac day: rungs 0–13 on Apple's graphics chip, plus speed | 1–2 days | the suite runs offline in minutes; speed takes about an hour per size |
| Rung 14, port side, on the Mac | 2–7 days of Mac time | 15 runs × 250K steps at the projected 6–25 steps/s (2.8–11.6 h per run) |
| Rung 14, official side | 30–60 T4-hours (1–2 weeks of quota), or about $2–10 rented | 15 runs × 250K steps; rented on a packed RTX 4090 at an assumed 150–500 combined steps/s (inference) |

The six weeks before the Mac arrives (about 27 December) are enough to finish stages 0–3 with margin, even at the slow end. The official-side curves for rung 14 are worth planning together with brain A's planned test of whether `size1m` reproduces the published curves. That test runs the same official preset on the same control tasks, so one set of official runs can serve both (inference from the earlier review's "last 10%" item 2).

## Rented GPUs make the large runs cheap; the port is for the Mac and for owners' devices

**What a proper GPU costs this week.** Marketplaces bill by the second. Their prepaid balances stop an instance when they run dry, which works as a hard spending cap. Storage keeps billing until an instance is deleted ([Vast docs](https://docs.vast.ai/documentation/instances/rental-types); [RunPod pricing](https://www.runpod.io/pricing)).

| Card | RunPod secure / community, $/h | Vast median (range), $/h | Note |
|---|---|---|---|
| RTX 3090, 24 GB | 0.50 / 0.22 | 0.26 (0.16–0.37) | Cheapest sensible card |
| RTX A5000, 24 GB | 0.27 / 0.16 | 0.23 (0.19–0.23) | Few cores per offer |
| RTX 4090, 24 GB | 0.89 / 0.34 | 0.54 (0.36–0.71) | Vast median offer has 24 cores and 63 GB; halves wall-clock |
| RTX 5090, 32 GB | 1.39 / 0.69 | 0.87 (0.50–1.02) | Worse value at these model sizes |
| A100 80 GB | 2.19 / 1.39 | 0.80 (0.43–1.15) | Little gain for a 1–10M model |

Sources: [RunPod pricing, 12 Nov 2026](https://www.runpod.io/pricing); Vast public offer listings, 15 Nov 2026 ([Vast pricing](https://vast.ai/pricing)). Interruptible bids run 30–60% lower, but suit only short, restartable runs.

**What the three big jobs cost.** No Dreamer speed has been published for any of these cards. The only anchor is a figure of about 10 steps a second for the official 12M image model on an RTX 3080 Ti, quoted in the earlier review, scaled by spec-sheet ratios. So these are wide ranges with 30% added for compiling, evaluation and one failed restart (inference). Reproduction, 9 runs × 100,000 steps of the 9.7M brain, costs **$5–18**. The decisive test with the small brain, 20 runs × 1M steps packed 4–8 to a card, costs **$4–34**, against 4–9 weeks of the free T4. The same test with the 9.7M brain costs **$72–409**. One month of continuous lifetime costs **$115–266** on a 3090 or **$338–511** on a 4090 on demand. Reserved monthly terms are up to 50% off ([Vast pricing](https://vast.ai/pricing)). A used RTX 3090 costs about $1,250–1,460, the price of 4,900–5,700 rented hours, so buying makes sense only if month-long lifetimes become a standing activity ([gpupoet, Sep 2026](https://gpupoet.com/gpu/learn/price/september-2026/nvidia-geforce-rtx-3090)).

**The staging rule.** Free machines and the Mac prove everything. Paid time is spent only on runs that are written down beforehand with their configurations, seeds, expected hours and an abort ceiling, using code that has already passed its gates.

| Stage | Spend | Enters only when |
|---|---|---|
| Free proving (now, then on the Mac) | $0 | — ; it delivers one pinned container of the official code tested on Kaggle and the T4, a clean checkpoint-kill-resume, a proven stall watchdog and self-destruct script, measured T4 speed at both sizes, and a written run sheet |
| First paid test: 1–2 hours on one RTX 4090 with ≥ 24 cores | about $1–2 | free proving done; it measures speed alone and packed 2, 4 and 8 to a card, boot-to-first-step time, a forced kill and resume, and the cost of bringing results home |
| Job 1, reproduction | about $5–20 | the paid test passed; checks the rented setup against published curves |
| Job 2, decisive test (small brain first) | $4–34, then $72–409 | Job 1 reproduced; measured speed fits the budget; resume passed on a rented host; the watchdog has fired once; all 20 configurations frozen |
| Job 3, lifetimes | $115–511 a month | Job 2 showed something worth extending |

Guards against waste: the instance destroys itself on exit or after 20–30 minutes without progress; small prepaid balances with automatic top-up off; delete, never just stop; only scoped, revocable keys on a stranger's machine. None of this spending is authorised yet. This is the plan the owner would approve step by step.

**Where the port fits, and whether it is worth building.** The plain answer: **no paid run needs it.** The official JAX code is the verified thing, it runs on rented NVIDIA cards unchanged, and the port only becomes verified at rung 14. Running the port on rented cards through MLX's CUDA backend is unmeasured and should not be paid for.

The Mac also does not need the port to "prove the system". The world, the harness, brain B, checkpoints and the official Dreamer at small scale on the processor all run there without it. The port adds one more link that must itself be proven.

What the port buys is real, but narrower. The first gain is **speed on the Mac**. The companion review projected MLX at **6–25 steps a second for `size1m`**, against 2–6 for official JAX on the Mac's processor and 3–12 for the PyTorch port on its graphics chip. That PyTorch route also carries a fresh report of silently wrong recurrent gradients ([PyTorch issue 204359](https://github.com/pytorch/pytorch/issues/204359)). If the projection holds, a small Dreamer creature on the Mac lives at roughly 1.2–5 times real time (Jack decides 5 times a second), for about $7 a month in electricity, where renting the same lifetime costs $115–266 a month (inference). The second gain is **owners' Apple devices**. People who receive a trained creature could run it there, and MLX is Apple's native toolkit on the Mac. That use needs only the acting path (encoder, one recurrent step, actor), which makes it the cheapest and clearest-value slice of the port.

Against that, the 9.7M brain on the Mac stays slow even with MLX (1.5–5 steps a second against 15–25 on a rented 4090), and every Mac speed above is a projection from older Apple chips.

So the port is worth building **as a side lane, not a dependency**: stage 0 first; a stop rule if gate G0 fails or if rungs 7–11 stall for more than about 20 slots beyond plan; the policy-only slice early; and the full training port finished in time for the Mac's first week. If the Mac's measured MLX speed comes in at the low end, below what the PyTorch route achieves, keep the port for owners' devices and lifetimes on rented cards.

## Conclusion

The question the owner asked, "how easy is the port?", turns out to have a cheaper answer than the question underneath it, "where should Dreamer's big runs happen?". The port is a few weeks of disciplined, test-first work with one genuine unknown. A rented card at a quarter to half a dollar an hour runs the official code's reproduction and the small-brain decisive test for tens of dollars each. That reverses the earlier plan's assumption that the T4 and the Mac had to carry the decisive test. The Mac's role shifts from "the second Dreamer lane we cannot do without" to "the cheap always-on home for long lives and the proof that the creature runs on Apple hardware".

The port also gives the project something nobody has published: the first parity-tested Dreamer for Apple silicon, together with the first measurements of MLX compile cost, MLX-on-CUDA speed and cross-chip agreement for a recurrent world model. Those measurements, and the golden-file harness that produces them, are worth more than the port's speed. The same harness can certify any later Dreamer change the team makes, on any machine, before a single paid hour is spent.
