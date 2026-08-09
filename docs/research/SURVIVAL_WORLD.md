# SURVIVAL_WORLD.md — what world does Jack live in, and what does each stage cost?

> Research agent, 2026-08-09. Serves the owner directive recorded in `GOAL.md`
> §"The world is the teacher": Jack gets human needs (eat, drink, sleep,
> temperature, company), is thrown into as realistic a survival world as we can
> build, lives, dies, and tries again.
>
> **Mid-research course correction from the owner, incorporated throughout:**
> *"We don't actually need to understand chemistry for this — just like cavemen
> didn't."* Section 3 is written to that brief, and the fidelity ladder in
> section 4 is re-costed under it. Rule DEPTH, not molecular accuracy, is the
> axis.

Every cost figure below that is marked **[measured]** was measured on this box
on 2026-08-09; the raw runs are in section 7. Everything else is cited or
flagged unverified. Constraint that governs all of it: **free compute only** —
4 shared ARM Neoverse-N1 cores here (also serving paying tenants, `nice 19`,
≤1.5 GB), Kaggle 30 h/week P100, Colab T4 elastic (`SYSTEM.md`, "Hard
constraints").

---

## 0. The answer in one page

**Jack should not move to a new simulator. He should stay in MuJoCo and the
world should be grown around him, the way `Water` was.** That conclusion is
forced by two measurements and one requirement, not by preference:

1. **The requirement.** `GOAL.md`'s ladder-and-apple standard needs a body with
   proprioception that can *fall*. Every cheap survival world — Crafter,
   NetHack, Neural MMO, Craftium, Minecraft — gives an abstract avatar whose
   "climb" is a state transition. A gridworld agent cannot fall off a ladder,
   so it cannot produce the datum GOAL.md is about. Avalon and Habitat have 3-D
   bodies but Habitat's humanoid is *kinematically animated, not physically
   simulated* (§2.4) and Avalon needs x86 + NVIDIA and is GPLv3 (§1.6).
2. **The hardware.** Every platform with native survival semantics rich enough
   to matter — OmniGibson/BEHAVIOR-1K, Isaac Lab, InfiniteWorld — requires an
   RTX-class GPU with ray-tracing cores. Kaggle's P100 has none. They are not
   "expensive for us"; they are unreachable at any amount of our free quota.
3. **The measurement.** The needs overlay is nearly free and the *policy* is the
   budget. Adding a thermal + metabolism + sleep-pressure overlay to the live
   playground cost **6.1%** of throughput (3,009 → 2,826 steps/s) **[measured]**.
   A 128×128 fire/heat cellular automaton at 5 Hz costs **0.06% of one core**
   **[measured]**. Meanwhile a 18.9M-parameter transformer policy costs **49.3
   ms per forward** on this box **[measured]** — at 40 Hz control that is 28×
   the entire cost of the physics. **The world is cheap. Jack is expensive.**
   This is the 57M-vs-54K lesson (`docs/LESSONS.md`) arriving from a second,
   independent direction.
4. **The surprise, and it inverts the premise of the whole survey.** Benchmarked
   on these same four cores, a 3-D torque-controlled humanoid with 348-dim
   proprioception (`Humanoid-v5`) runs at **2,222 env-steps/s**, while the
   richest *abstract* needs-world in the literature — full Craftax — runs at
   **365–557 sps** **[measured]**. **Embodiment is 4–6× CHEAPER per decision
   than a rich gridworld needs simulation.** Craftax's famous 250× speedup is
   entirely GPU-conditional: on ARM it lands at roughly the speed of the Python
   original on a 24-core i9. So the choice is not "pay for a body or get needs
   cheaply". Needs are decay scalars appended to an observation; a body cannot
   be bolted onto a gridworld at any price. We are on the cheap side and the
   expressive side at once.

### The ladder, costed

| | world | new capability it can teach | cost on our compute | fidelity gate |
|---|---|---|---|---|
| **W0** | current playground + needs overlays: scalar temperature field, hunger/thirst/sleep meters, edible apple, day/night light cycle | *that states of his own body exist and are controllable* — the first thing an agent can learn that is not about the room | **6.1% throughput** on top of PG.8's world. One Jack-day (20 min sim-time) = **85 s wall on one core** **[measured]** | W.1: heat balance vs the closed-form lumped-capacitance prediction, incl. the parameter-free thermoneutral point at 27.6 °C |
| **W1** | survival playground: weather (wind, rain), fire, cookable food, buildable shelter parts, food scarcity, **death** | *that an action taken now pays off later* — shelter before nightfall. Deferred consequence is the thing W0 cannot teach | ~2× W0 per step; a 10-day life ≈ **14 min wall**, a 3-seed × 10-life study ≈ **7 core-hours** | W.3: an insulated shelter must change survival time by the amount the heat balance predicts, and a zero-insulation shelter must not help |
| **W2** | Crafter/Avalon-class rule depth **on the MuJoCo body**: a recipe/transformation graph with condition gates, tools, spoilage, and rules that interact | *technology* — a chain of actions no single reward describes. Crafter's 22 achievements are the calibrated reference for "is the tree deep enough" | policy-bound, not world-bound. Needs vision (PG.6) which this box **cannot render natively** (§7.4) — Xvfb + llvmpipe, or GPU | W.4 (consistency + discoverability) and W.5 (fire obeys its published rules) |
| **W3** | maximum **rule depth**: rules whose *interactions* generate situations nobody enumerated (fire + rain + wind + wet fuel + wind-driven spread), plus ACCEL-style rule mutation across lives | *open-endedness* — the discovery tree stops being a list we wrote | the honest ceiling is GPU-bound and requires porting every overlay to JAX/MJX (§2.1); costed at 0 today because that port does not exist | W.8: reachable distinct rule-interaction events must exceed the enumerated rule count, and must saturate when interactions are disabled |

### The one thing that must be decided before W0 is built

**Time compression.** Physics for a humanoid needs dt = 0.005 s. A real
24-hour day is 17.28 M physics steps = **1.70 hours of wall clock on one core**
**[measured]**. That is affordable for one life and ruinous for a 3-seed study.
Every survival game solves this the same way — a Minecraft day is 20 minutes —
and so should we: **declare a compression factor `k` (proposed k = 72, one
Jack-day = 20 min sim-time), scale every need rate by it, and pre-register it.**

The hazard is specific and it is the reason this gets its own spec (W.7): a
compressed clock silently invalidates every analytic physics gate, because
Newton's law of cooling has a time constant in *seconds*. The resolution is the
split that PG.2 already used without naming it: **verify the physics in a
dedicated 1× fixture against the analytic prediction; live the world on the
compressed clock; assert the dimensionless ratios are invariant.**

---

## 1. Survival-mechanic worlds

### 1.0 The question that decides this section

For each candidate: (a) which needs are native, (b) rendering cost and
CPU-vs-GPU, (c) observation/action space, (d) licence, and **(e) can a
physically simulated humanoid body with proprioception exist in it?**

(e) is not one criterion among five. It is the gate. `GOAL.md`: *"If there is a
ladder with an apple on top, he must try to climb the ladder, fall, and learn
from falling."* A fall is a physical event with a torque history, a contact
sequence and a proprioceptive signature. In every world in this section,
"climb" is a discrete transition and "fall" is a hitpoint subtraction. Such a
world can teach *planning over a rule graph*; it cannot teach *a body*.

**This does not make them useless to us. It makes them the wrong substrate and
the right specification.** Crafter's achievement tree is the best published
calibration in existence for "is this rule-set deep enough to be interesting,
and can an agent actually climb it" — with baselines including a human score.
We port its *rule depth* onto our body; we do not port our body into it.

### 1.1 Crafter — the reference rule-set, and the one to copy

arXiv:2109.06780, Hafner, ICLR 2022. **MIT licence.** *"An open world survival
game with visual inputs that evaluates a wide range of general abilities within
a single environment"*; agents are scored by success rate over **22
achievements** and their **geometric mean**, which is the right shape for a
coverage metric because it punishes a spiky profile.

**Observation** 64×64×3 RGB (a rendered view of a local grid plus an inventory
HUD). **Action space: 17 discrete** — move ×4, `do` (context-sensitive collect/
attack), `sleep`, place ×4 (stone, table, furnace, plant), make ×6 (wood/stone/
iron pickaxe and sword), noop.

**Native needs — read from the source, not the README** (`crafter/objects.py`,
`crafter/data.yaml`):

```python
# all of health, food, drink, energy: max 9, initial 9
_hunger  += 0.5 if sleeping else 1;  if _hunger  > 25: _hunger  = 0; food   -= 1
_thirst  += 0.5 if sleeping else 1;  if _thirst  > 20: _thirst  = 0; drink  -= 1
_fatigue += -1  if sleeping else 1;  if _fatigue > 30: _fatigue = 0; energy -= 1
                                     if _fatigue < -10: _fatigue = 0; energy += 1
```

So the entire survival clock is: **food runs out in 9 × 25 = 225 steps, water in
9 × 20 = 180 steps**, and sleep halves the drain while recharging energy. There
is a day/night cycle (night spawns zombies). **There is no temperature and no
social need.** That is worth stating plainly: the best-known survival RL
benchmark models *two* of the owner's five needs, and neither of the two the
owner named first as lethal ("too cold kills, too hot kills").

**Rule set:** collect (tree→wood free; stone/coal need a wood pickaxe;
iron needs stone; diamond needs iron; water→drink free; grass→sapling at 10%),
place (stone 1, table 2 wood, furnace 4 stone, plant 1 sapling), make (all
require a nearby table; iron tools also a furnace). **This is a condition-gated
transformation graph and it is exactly the W2 target.**

**Baselines at 1 M env steps:** human **50.5%**, best reported baseline (Curious
Replay) **19.4 ± 1.6**. The gap is the point of the benchmark.

**Cost [measured on this box]: ~100–124 steps/s on one ARM core.** Python-native,
CPU-only, no GPU needed. The paper's own comparison table puts Crafter at 1,580
sps on a 24-core i9, so our per-core figure is consistent. At ~110 sps, 1 M env
steps — the benchmark's own budget — is **~2.5 hours**.

**(e) Embodiment: NO.** A 2-D grid avatar with 4 discrete moves. There is no
body, no joint, no contact, no fall.

### 1.2 Craftax — the richest needs model surveyed, and its speedup is GPU-only

arXiv:2402.16801, Matthews, Beukman, Ellis, Samvelyan, Jackson, Coward,
Foerster; Feb 2024, ICML 2024. **MIT.** A JAX rewrite: Craftax-Classic is *"up
to 250× faster than the Python-native original"* and a PPO run of 1 B
interactions *"finishes in under an hour using only a single GPU"*. Full Craftax
extends the tree with NetHack-inspired elements — **5 intrinsic needs plus RPG
attributes**, the best needs model in the survey — and the shipped v1.6.1 has
**67 achievements** where the paper says 65. Observations: Classic symbolic
`(1345,)`, full symbolic `(8268,)`, Classic pixels 63×63×3, full pixels
`(130,110,3)`.

**The speedup does not survive contact with our hardware [measured on this
box]:**

| configuration | envs | compile | steady steps/s on 4 shared ARM cores |
|---|---|---|---|
| Craftax-Classic-Symbolic | 64–1024 | 16–42 s | **1,500–1,740** |
| **Craftax-Symbolic (full)** | 64–256 | 43–54 s | **365–557** |
| Craftax-Pixels (full) | 32 | 34 s | 514 |

Three consequences, and the first is the one that matters:

1. **Throughput is flat in `num_envs` on CPU** — `vmap` buys nothing once four
   cores saturate. The 250×/169× figures are **entirely GPU-conditional**.
   Craftax-Classic on this ARM box (~1,700 sps) is roughly *equal to original
   Python Crafter on a 24-core i9* (1,580 sps, the paper's own table). For
   reference, the published absolute numbers on an RTX 4090 + i9-13900K are
   405,618 sps (Classic) and 266,961 (full).
2. **Budget arithmetic: Craftax-1M ≈ 30–45 min here (feasible); Craftax-1B ≈
   21–32 days (not feasible).** Note that 1M and 1B are *budget settings on the
   same environment*, not two environments.
3. **Pixel rendering is nearly free** (514 vs 557 sps) because it is a JAX
   tile-blit, fully headless — a notable contrast with our own MuJoCo rendering
   problem (§7.4).

Calibration finding worth carrying into W2/W3: *"existing methods including
global and episodic exploration, as well as unsupervised environment design fail
to make material progress on the benchmark."* The curiosity family CU.1–CU.7 is
about does **not** solve a deep tech tree. A W2 that expects curiosity alone to
climb a 20-rule graph has mis-specified its success criterion.

**(e) Embodiment: NO.** Same gridworld, on a GPU.

### 1.3 MineRL / MineDojo / Minecraft — the largest rule graph, the worst substrate

Minecraft has **~1,000 recipes** (~700 crafting + ~250 smelting/cooking + ~50
smithing, plus brewing), stored as JSON data files, and the best-studied fire
model in games (§3.3e). Natively it has **hunger** (food/saturation), **no
thirst**, and **no temperature** — the temperature mechanics people cite are
mods, not vanilla. MineDojo (arXiv:2206.08853), VPT (arXiv:2206.11795) and
Voyager (arXiv:2305.16291) are the research stack.

Against us: a **Java** process per environment instance at hundreds of MB of RAM
and human-scale frame rates — the opposite of what four shared ARM cores want —
and the Minecraft EULA makes the *content*, as opposed to the research wrappers,
legally awkward to build a product on. Craftium (§2.6) exists precisely to dodge
that, using Luanti/Minetest instead.

**(e) Embodiment: NO.** A voxel character controller: capsule collision, no
joints, no torques, no proprioception.

### 1.4 NetHack Learning Environment — the cheapest death-and-retry loop anywhere

arXiv:2006.13760, Küttler, Nardelli, Miller, Raileanu, Selvatici, Grefenstette,
Rocktäschel; NeurIPS 2020. Built on NetHack 3.6.7 with a Gymnasium API; the
repo is active (~9.8 K commits, papers listed through 2025). MiniHack
(arXiv:2109.13202) is the sandbox variant for designing small tasks.

Its pitch is exactly the tradeoff we care about — *"procedurally generated,
stochastic, rich, and challenging"* while *"requiring fewer computational
resources to gather experience"*. NetHack natively models **hunger** (and
nutrition per food item, which is unusually detailed), **no thirst**, and **no
temperature** in the physiological sense. Death is **permanent and terminal**,
which is why it is the canonical place to study the death-and-retry loop:
the roguelike *is* the loop the owner described.

**Cost [measured on this box]: 18,830 steps/s on one ARM core** for
`NetHackChallenge`, and **3,401 sps** for MiniHack `Room-5x5`. So the
tens-of-thousands folklore is real, it is the fastest *needs-bearing* world in
the survey by two orders of magnitude, and **it does run on aarch64** — which
resolves the ARM question I had flagged unverified. Caveats that remain: NLE
wants **python ≥ 3.10 and cmake ≥ 3.28** while the project venv is **Python
3.9.25**, so it needs its own environment, and it is a C++ build rather than a
wheel drop-in.

**Licence is the real constraint, and it is not the permissive one I assumed:**
NLE ships under the **NetHack General Public License**, not MIT/Apache. MiniHack
is Apache-2.0. Anything NLE-derived inherits a copyleft obligation.

**(e) Embodiment: NO.** An ASCII grid. This is the *furthest* possible world
from a body, and simultaneously the best available laboratory for the *loop*.
That is a genuinely useful split: **if we ever want to study death-and-retry
mechanics in isolation, cheaply, NLE is the place — and nothing learned there
transfers to a body.**

### 1.5 Neural MMO 2.0 — multi-agent survival, and the social need

arXiv:2311.03736, Suárez, Isola, Choe, Bloomin *et al.*, Nov 2023. *"A massively
multi-agent environment"* with **128 agents** in the standard configuration, a
flexible task/reward system, a complete rewrite with **"three-fold improved
performance"**, CleanRL-compatible, free and open source (specific licence
**unverified**). Neural MMO's classic survival mechanics are food and water
foraging with health regeneration; the 2.0 abstract does not enumerate them and
I could not verify the current need set from primary sources (**unverified**).

Its relevance to us is narrow but real: it is the only surveyed world where the
owner's fifth need — **company** — is native, because there are other agents.
Our answer to that need (§4.3) is conversational rather than a second physics
body, for cost reasons; Neural MMO is the reference for what we are choosing not
to do.

**(e) Embodiment: NO.** Tile-grid avatars.

### 1.6 Avalon — the only survey entry with a 3-D body, and three blockers

arXiv:2210.13417, Albrecht *et al.*, NeurIPS 2022 Datasets & Benchmarks. Godot-
based, **20 tasks** from *eat* and *throw* to *hunt* and *navigate*, where agents
*"survive by navigating terrain, hunting or gathering food, and avoiding
hazards"* — and, importantly for benchmark hygiene, tasks differ only by
environment, not by reward function. That is the closest thing in the literature
to what the owner described.

Three blockers, in order of severity:

1. **Licence: GPLv3** for code and resources (CC-BY-SA 4.0 for the human
   rollout dataset, MIT for their Godot engine modifications). That is a real
   constraint on anything we might release.
2. **Hardware: Linux x86_64 with an NVIDIA GPU is mandatory for headless
   rendering.** ARM is not mentioned. It does not run on this box.
3. **Maintenance looks thin** — ~80 commits, v1.0.0, no recent activity.

**(e) Embodiment: PARTIAL, and unverified.** It has a 3-D physically-collided
character, but whether that is an articulated body with joint-torque control or
a capsule controller with scripted climbing **could not be verified**. Given
Godot's standard `CharacterBody3D` idiom, a capsule is the likely answer, in
which case Avalon shares the gridworlds' disqualifier in a prettier costume.
**No temperature mechanic found.**

### 1.7 The others, briefly

**Craftium** (arXiv:2407.03969, ICML 2025, MIT + CC-BY-SA content) — Luanti/
Minetest with a Gymnasium API at **>2,000 steps/s**, mod-able needs. The most
attractive *rule engine* in the survey. No aarch64 wheels; voxel controller, not
a body. Covered in §2.6.

**XLand-MiniGrid** (arXiv:2312.12044) is the speed outlier and it is
extraordinary: **1,728,100 steps/s at 4,096 envs on these four shared ARM
cores**, with a 1.5 s compile — 1 B steps in ~10 minutes, *beating the paper's
own 2×T4 training throughput*. Obs `(5,5,2)` int (tile ID, colour ID —
explicitly not images), 6 actions, 44 registered envs. **Zero needs, zero body:
pure task-inference.** Worth remembering as the place to debug a *learning
algorithm* at negligible cost, and useless as a world for Jack.

**Kinetix** (arXiv:2410.23208, MIT) is the near-miss and its failure mode is
instructive. Size `s` symbolic/continuous runs at **6,056 sps** here; **size `l`
collapses to 59 sps** (161 s compile), so the tier with interesting jointed
bodies is unusable on this box — 1 M steps ≈ 4.7 h before a single policy pass.
And there is a hard design ceiling independent of speed: **only 4 independent
motor bindings and 2 thruster bindings** (`num_motor_bindings=4`), so size `l`'s
6 joints *cannot be addressed independently*. Bodies are low-DoF by
construction. Actions `Box(6,)` / 11 discrete / 16 n-hot; pixels `(125,125,3)`
through the `jaxgl` software rasteriser, headless, no GL.

**JaxMARL**, **Melting Pot**, **DMLab2D** — multi-agent, no needs suite, and no
published absolute sps (**unverified**).

#### Video world models (Genie-class) — not simulators, and mostly not available

Worth stating explicitly, because "as real as possible" invites them and none of
them is a world Jack can live in:

| | availability | speed | verdict |
|---|---|---|---|
| **Genie 1** (arXiv:2402.15391) | **weights explicitly refused in the paper** | 160×90 @10 FPS train, **~1 FPS inference** | 8 *learned latent* actions with no semantics — you cannot command "eat" |
| **Genie 2 / 3** | blog-only; Genie 3 widened ~Jan 2026 as consumer "Project Genie" for AI Ultra subscribers | 720p/24fps/minutes | **no programmatic or agent API confirmed** |
| **MineWorld** | MIT, 300M/700M/1.2B checkpoints | **3.0–5.9 FPS on A100/H100** | no gym `step()` API; the paper concedes it models no *"hunger, health, or inventory"* |
| **Oasis-500M** | MIT; full model closed | H100-class | — |
| **Lucid v1** | weights released, **no licence anywhere — treat as all-rights-reserved** | 25 FPS on RTX 4090, 2 s context | — |
| **DIAMOND** | the only one that is a real RL environment (separate reward + termination models; agents train fully in imagination) | ~12 GB VRAM, **2.9 GPU-days per Atari game on a 4090** | genuinely interesting, and ~30× our entire weekly quota per game |

**None has a needs system, a body, or a state you can query.** A video world
model predicts pixels; Jack needs a core temperature. They are out on capability
before they are out on cost.

### 1.8 Verdict table

**Every throughput figure below was measured on this box's four shared ARM
cores**, so this table compares like with like on the hardware we actually have.

| | steps/s **[measured here]** | native needs | licence | **jointed body + proprioception** |
|---|---|---|---|---|
| XLand-MiniGrid (4,096 envs) | **1,728,100** | none | Apache-2.0 | ❌ |
| NLE (NetHackChallenge, 1 core) | **18,830** | hunger | **NetHack GPL** | ❌ ASCII grid |
| Kinetix size `s` | 6,056 | none | MIT | ⚠️ 2-D, ≤4 motors |
| MiniHack Room-5x5 (1 core) | 3,401 | hunger (inherited) | Apache-2.0 | ❌ |
| **Humanoid-v5 (bare)** | **2,222 env-steps/s** | none | MIT/Apache-2.0 | ✅ **3-D, 17 act / 348 obs** |
| dm_control humanoid/stand | 2,150 | none | Apache-2.0 | ✅ 3-D, 21 act |
| Craftax-Classic | 1,711 | 4 (food/drink/energy/health) | MIT | ❌ |
| **Craftax full** | **365–557** | **5 + RPG attributes** (richest surveyed) | MIT | ❌ |
| Crafter (1 core) | ~100–124 | 4 | MIT | ❌ |
| Kinetix size `l` | 59 | none | MIT | ⚠️ 6 joints, 4 channels |
| dm_control humanoid_CMU (56 DoF) | 521 | none | Apache-2.0 | ✅ |
| **ours (playground + Jack + water + needs)** | **2,826 mj-steps/s** | **all five, because we build them** | Apache 2.0 | ✅ |
| Minecraft family | not run (Java, heavy) | hunger only | EULA-encumbered | ❌ voxel controller |
| Avalon | not run (**x86 + NVIDIA required**) | food, energy | **GPLv3** | ⚠️ articulation unverified |
| Craftium | not run (no aarch64 wheels) | mod-defined | MIT | ❌ voxel controller |

**Two conclusions, and the second is the surprise.**

**(1) No surveyed world both runs on our hardware and has a body.** The only row
satisfying every column is the one we already have.

**(2) The body is CHEAPER than the needs-world.** A 3-D torque-controlled
humanoid with full 348-dim proprioception runs at **2,222 env-steps/s** here,
while the richest gridworld needs-model — full Craftax — runs at **365–557
sps**. **Embodiment costs 4–6× LESS per decision than a rich abstract needs
simulation on identical hardware.** Physics is not the bottleneck; JAX-on-CPU
overhead over a large symbolic state is.

This inverts the intuition the whole survey was built to test, and it settles the
architecture question empirically rather than by preference: **needs bolt onto a
body trivially — they are decay scalars appended to an observation, measured in
§7.1 at 6.1% — whereas a body cannot be bolted onto a gridworld at any price.**
We are on the cheap side of the trade and the expressive side simultaneously.

> **A lever this comparison exposed.** Bare Humanoid-v5 does **11,110 physics
> steps/s** (frame_skip 5 × 2,222) against our playground's **3,009** — a 3.7×
> gap. Both use the same integrator (RK4) **[verified]**; the difference is
> scene complexity, **52 geoms / 21 bodies vs 18 / 14**. Perimeter walls, four
> pool walls, the pool floor, stairs and ladder rungs are contact-checked every
> step. If W1/W2 ever become throughput-bound, culling or simplifying decorative
> collision geometry is worth up to ~3× before any algorithmic change — and it
> is free. Worth a `contype`/`conaffinity` audit rather than an assumption.

What the survey buys us is therefore not a substrate. It is **Crafter's rule
graph and geometric-mean scoring, Craftax's 5-need model, NetHack's death loop,
and Minecraft's fire model** — designs rather than dependencies, all of which
are ours to implement.

---

## 2. Embodied-sim platforms

Surveyed 2026-08-09 with primary sources. Headline: **nothing in the MuJoCo
stack simulates heat** — verified across the MuJoCo overview, the full
`mjModel`/`mjData`/`mjOption` API type reference, the fluid-force computation
page, and the first-party plugin list: zero mentions of temperature, thermal,
or heat. Thermoregulation is ours to build, exactly as buoyancy was.

### 2.1 MuJoCo 3.x — what is native, what we must build

| | status |
|---|---|
| Thermal / temperature | **absent.** Not in `mjOption`, not in any first-party plugin. The extension docs mention a thermodynamically-coupled actuator only as a *hypothetical* example of a stateful plugin. |
| Fluids | Two **stateless force fields**, not a fluid: the inertia-box model (global `density`/`viscosity`) and the ellipsoid model (`<geom fluidshape="ellipsoid"/>`, added in MuJoCo 2.2.1) with blunt/slender/angular drag, Kutta lift, Magnus lift, added mass. **No free surface, no water body, no fluid state.** This is exactly why `playground.Water` exists as a per-geom passive-force callback. |
| Deformables (flex) | **Real, added in MuJoCo 3.0.** 1-D/2-D/3-D flex → capsule/triangle/tetrahedron elements; cloth, rope, soft volumes, with collision and passive+constraint forces via the `elasticity` plugins (1-D cable large-deformation, 2-D shell bending). **This is our most plausible native route to shelter material and cordage.** |
| Metabolism, needs, day/night | absent; all overlay work. |

**MJX (MuJoCo XLA) — the GPU path, and its trap.** Two backends with a feature
gap that lands directly on us:

| | MJX-JAX | MJX-Warp |
|---|---|---|
| Joint types | FREE/BALL/SLIDE/HINGE only | all |
| Flex / deformables | **not supported** | supported |
| Fluid model | **inertia only** (no ellipsoid) | all |
| Autodiff | yes | no |

So the JAX path costs us *both* flex (shelter) and the ellipsoid fluid model.
Throughput from the MuJoCo docs: pure Warp humanoid **3.35 M steps/s**, JAX-FFI-
Warp 2.96 M. Single-scene MJX-JAX is **~10× slower than CPU MuJoCo** — batching
is the entire point.

**Does MJX run on Kaggle's P100?** Yes, with a pin. Per JAX installation docs,
CUDA-12 builds support SM 5.2+; **CUDA-13 builds support SM 7.5+ and therefore
drop Pascal**. P100 is SM 6.0. So: MJX on P100 requires staying on CUDA-12
jaxlib wheels, and a silent upgrade to CUDA 13 will break Kaggle while leaving
Colab's T4 (SM 7.5) working. That is a trap worth a guard.

**MuJoCo Warp (MJWarp)** is beta, "mostly feature complete", audience stated as
"physics engine enthusiasts and learning framework integrators". Performance
degrades sharply above 60 DoF; our humanoid is ~23 DoF, inside the envelope.
Minimum compute capability **unverified** — treat P100 as untested.

> ⚠️ **The architectural fork nobody has costed yet.** `playground.Water.apply`
> is a Python `for` loop over `model.nbody` mutating `data.xfrc_applied`. It
> cannot run under MJX. **Every overlay we add in this style is CPU-only until
> rewritten as JAX.** That is not an argument against building them — the CPU
> box is where W0 and W1 live — but W3's cost estimate is dominated by that port,
> and pretending otherwise would be the kind of number this project does not
> publish. Design overlays as pure functions of `(state, params) → forces/state`
> from day one so the port is mechanical.

**Name collision, worth fixing now:** *MuJoCo Playground* (arXiv:2502.08844,
Zakka et al., DeepMind/Berkeley/Toronto, Feb 2025, Apache 2.0,
playground.mujoco.org) is a well-known MJX-based environment suite. Our
`playground.py` is unrelated. Renaming ours costs nothing today and avoids
confusion in exactly the contexts where we would cite either.

### 2.2 Isaac Lab / Isaac Sim / Isaac Gym / Newton — ruled out, definitively

Isaac Sim 5.1 requirements are unambiguous: **minimum GPU GeForce RTX 4080,
16 GB VRAM**, and — decisively — *"GPUs without RT Cores (A100, H100) are not
supported."* Kaggle's **P100 is Pascal and has no RT cores: definitively out.**
Colab's T4 has RT cores but sits far below the minimum and appears on no
supported list; NVIDIA staff describe the minimum as "the lowest spec GPU tested
for the given release", i.e. below it expect crashes and no support. An
unofficial `isaac-sim-colab` repo exists and self-describes as "demo purposes
only, using various hacks… serious development is not recommended."

Isaac Gym is **deprecated** (Preview 4 final). **Newton** (Linux Foundation,
Sept 2025, NVIDIA + Google DeepMind + Disney, Apache 2.0, Warp + OpenUSD, ships
MuJoCo-Warp and a Vertex Block Descent solver for cloth/cables/deformables)
requires a CUDA GPU with ≥8 GB VRAM but **not** RT cores, so T4/P100 are
*plausible* — compute-capability floor unverified. **No thermal model.**

**Verdict: the entire NVIDIA embodied stack is out of reach on free compute,
and this is a hardware fact, not a budget preference.**

### 2.3 Genesis — the one worth an afternoon of measurement

`Genesis-Embodied-AI/Genesis`, **Apache 2.0**, v1.0 May 2026. Two properties
make it the only non-MuJoCo candidate worth our time:

**(a) It has a real heat-equation solver.** The `TemperatureGrid` sensor
voxelizes a link's AABB and returns a temperature field in °C, advanced each
step by **diffusion + internal source, contact conduction, then radiation and
convection**, using per-link **conductivity, density, specific heat, emissivity,
base temperature**. The solver is semi-implicit spectral: mirror-padded grids and
a real-input FFT to impose zero-flux boundaries, each Fourier mode updated
implicitly. That is a genuine PDE, not a heuristic.

**(b) It has ARM64 CPU wheels.** The compiled dependency `gstaichi` publishes
`manylinux…aarch64` wheels (cp310–cp313); `genesis-world` itself is
`py3-none-any`; the installation matrix marks CPU simulation and headless
rendering as supported on Linux. A CPU-only install on this Neoverse-N1 is
genuinely plausible — but **unverified end-to-end, and this box runs Python
3.9**, which is below `gstaichi`'s floor. That is a real blocker, not a detail.

**The honest caveat, and it decides the matter:** `TemperatureGrid` is presented
as a **sensor**. Whether temperature feeds *back* into physics/material
behaviour, or is read-only instrumentation, **could not be verified** (the doc
page 404'd on direct fetch). For a survival scenario, read-only is still enough
— Jack senses cold, and thermoregulation is our homeostat — but "ice melts,
water boils" would not follow.

**And the performance claims need discounting.** The advertised 43 M FPS means:
a Franka arm plus a ground plane, 430,000× real time. Stone Tao (ManiSkill
author) reproduced it and found the figure drops **43 M → 0.29 M FPS (150×)**
once realism is restored, identifying four inflation mechanisms: 1 physics
substep instead of 2–4; the robot acts once then idles >90% of the benchmark;
**self-collisions disabled by default**; active-object hibernation. He measured
**SAPIEN/ManiSkill 3–10× faster than Genesis** on manipulation.

**Recommendation: Genesis is a scheduled experiment, not a plan.** Before
anything depends on it, answer three questions in one sitting — does it install
and step on aarch64 given the Python-3.9 problem; is `TemperatureGrid`
read-only; what is its actual CPU steps/s for a 23-DoF humanoid versus our
measured 2,826. Until those are numbers in the ledger, W0–W2 are MuJoCo.

### 2.4 Habitat 3.0 — disqualified on embodiment, and unmaintained

Two independent disqualifiers. **The humanoids are not physically simulated.**
Quoting the paper (arXiv:2310.13724): *"The skeleton is used to represent poses
and check for collisions with the environment, whereas the skinned mesh provides
visual fidelity without affecting physics."* Locomotion is a single AMASS walk
cycle looped toward waypoints; reaching is precomputed VPoser interpolation;
grasping is kinematic attach/detach. **No joint-torque control exists.** For
GOAL.md's standard this is fatal — there is no proprioception to bind and
nothing to fall.

**And it is over:** the `habitat-sim` README states *"Beyond v0.3.4 this project
is no longer receiving official active development or maintenance by Meta
internal teams."* Also **no `linux-aarch64` conda build** (issue #2358), so it
does not run on this box regardless. Throughput for reference: 245±19 FPS
single-agent robot, 188±2 humanoid, 1191±3 at 16 parallel envs; trained on 4×
A100. Licence MIT. **No survival physics of any kind.**

### 2.5 AI2-THOR / ProcTHOR / Holodeck and ThreeDWorld — abstracted, not physical

**AI2-THOR** has temperature, but as *a three-valued enum*: Hot / Cold /
RoomTemp. Heat sources (lit StoveBurner, Microwave, Toaster) set contained
objects Hot; Fridge sets Cold; removal decays back over time. Related metadata:
`cookable`, `isCooked`, `canFillWithLiquid`, `isFilledWithLiquid`. The docs are
explicit that this is *"temperature as an abstraction rather than precise
numerical values… rather than realistic thermal simulation."* The agent is not
an articulated body.

**ThreeDWorld** (arXiv:2007.04954, BSD-2) is in **long-term support** — "minor
updates and bug fixes" only. Its Flex particle backend, the part that had fluids
and cloth, is described by TDW's own docs as *"no longer supported by
NVIDIA… a discontinued product with many known bugs"*, and **Flex fluids do not
work on Linux at all**. The replacement, Obi, is CPU-only and much slower. **No
fire, no heat, no thermal simulation.** TDW's real distinction is physics-driven
impact *audio*, which is a thing we already built ourselves (PG.5, PASS).

### 2.6 Unity ML-Agents, Godot RL, Craftium

**Unity ML-Agents** — Release 23 (28 Aug 2025), Apache 2.0 for the package,
actively developed. ARM64 Linux headless player support **unverified**; Unity's
Linux desktop standalone player is generally x86_64. The Unity Engine runtime
licence is separate from ML-Agents' Apache 2.0.

**Godot RL Agents** (arXiv:2112.03636, MIT, active) drives an *exported binary*
over IPC. ARM64 export templates, headless export and per-step IPC cost are all
**unverified** and the README publishes no FPS numbers.

**Craftium** (arXiv:2407.03969, ICML 2025, MIT + CC-BY-SA content) is a
Luanti/Minetest fork with a Gymnasium API at **>2,000 steps/s** — genuinely
attractive as a *rule engine*. But wheels are `manylinux_2_28_x86_64` only (no
aarch64), and the body is a **voxel character controller with no joint
dynamics**. It fails the embodiment gate by construction.

### 2.7 BEHAVIOR-1K / OmniGibson — the right model on the wrong hardware

This is the single most relevant prior art for "a world with human-scale
survival semantics", and it is unreachable for us. Read it for the equations,
not the software.

**The temperature model, precisely** (from `omnigibson/object_states/`):

```python
# temperature.py
DEFAULT_TEMPERATURE = 23.0        # °C
TEMPERATURE_DECAY_SPEED = 0.02    # 1/s   -> tau_ambient = 50 s
values += (DEFAULT_TEMPERATURE - values) * TEMPERATURE_DECAY_SPEED * dt
VALUES[idxs] += (temperature - VALUES[idxs]) * rate * dt   # per heat source

# heat_source_or_sink.py
DEFAULT_TEMPERATURE = 200.0       # °C
DEFAULT_HEATING_RATE = 0.04       # 1/s   -> tau_heating = 25 s
DEFAULT_DISTANCE_THRESHOLD = 0.2  # m
```

Two coupled exponential relaxations per object. Source→object coupling is an
overlap-sphere (or AABB overlap box, when `requires_inside`) query, gated by
`requires_toggled_on` / `requires_closed` / `requires_inside`, with self-heating
excluded. **`OnFire` subclasses `HeatSourceOrSink`**: past `ignition_temperature`
the state latches, temperature pins to `fire_temperature`, and *the burning
object becomes a heat source for its neighbours* — a genuine ignition-and-spread
loop in a few dozen lines. Other states: soaked, dirty, cooked, burnt, frozen,
sliced, diced, covered, filled, toggled; discrete transformations run through a
modular Transition Machine (dough + oven + threshold → pie).

The paper is candid that these are heuristic: *"OmniGibson also simulates
additional, non-kinematic extended object states (e.g., temperature, soaked
level) based on heuristics."* **That candour is the finding.** The best-funded
embodied-AI survival semantics in existence is a per-object scalar with two
exponential relaxations and a sphere query. We can implement it this week.

**Hardware: out.** Ubuntu 22.04+, 32 GB+ RAM, **NVIDIA RTX 2070+, 8 GB+ VRAM**,
inheriting Isaac Sim's constraints; headless requires a ray-tracing-capable GPU
(issue #1865 shows an RTX A6000 failing with *"No device could be created"*).
Docker install currently unavailable. Throughput ~**60 FPS** for a ~60-object
house scene; the FAQ concedes "OmniGibson may not currently be the fastest
simulation platform available."

**Actionable takeaway: port the model, not the software.** MuJoCo already gives
us the body, the contacts and the geom-proximity queries. The equations are
above and they are cheap.

### 2.8 Platform verdict table

| | ARM 4-core headless | Kaggle P100 | Colab T4 | licence | native thermal | physical humanoid body |
|---|---|---|---|---|---|---|
| **MuJoCo 3.2.3 CPU (current)** | ✅ **2,826 steps/s [measured]** | — | — | Apache 2.0 | ❌ (build it) | ✅ |
| MJX | ❌ | ✅ CUDA-12 pin | ✅ | Apache 2.0 | ❌ | ✅ (no flex, no ellipsoid fluid) |
| MJWarp | ❌ | ⚠️ unverified | likely ✅ | Apache 2.0 | ❌ | ✅ |
| Genesis | ⚠️ plausible; py3.9 blocks it today | ⚠️ unverified | ⚠️ unverified | Apache 2.0 | ✅ **heat PDE** (feedback unverified) | ✅ |
| Isaac Sim / Lab | ❌ | ❌ no RT cores | ❌ practically | NVIDIA EULA | ❌ | ✅ |
| Newton | ❌ | ⚠️ unverified | likely ✅ | Apache 2.0 | ❌ | ✅ |
| OmniGibson / BEHAVIOR-1K | ❌ | ❌ | ❌ | MIT + Omniverse EULA | ✅ heuristic | ✅ |
| Habitat 3.0 | ❌ no aarch64 | ✅ | ✅ | MIT | ❌ | ❌ **kinematic only** |
| AI2-THOR | ⚠️ unverified | ✅ | ✅ | ⚠️ unverified | ❌ enum only | ❌ |
| ThreeDWorld | ⚠️ unverified | ✅ | ✅ | BSD-2 | ❌ | ❌ |
| Craftium | ❌ no aarch64 wheels | ✅ | ✅ | MIT | ❌ | ❌ voxel controller |

---

## 3. The chemistry question, answered honestly — and then reframed

### 3.1 The original question, and the number that settles it

The owner asked how close to a realistic world we can get, *"where you can have
chemistry and science"*. The honest answer, with numbers rather than vibes:

| method | system | measured throughput | hardware | **sim-s per wall-s** |
|---|---|---|---|---|
| DFT single-point | 32 / 108 / 256-atom Pt | 34 s / 811 s / 8,280 s per SCF, O(N^2.64) | 1× Xeon Gold 6254 | n/a (one energy) |
| AIMD | 100–300 atoms | >1,000 steps/day ≈ 0.5–1 ps/day | GPU | **~1e-17** |
| MLIP (MACE-MP-0a) | 1,000-atom diamond | 2.2 Msteps/day ≈ 2.2 ns/day | 1× H100 | **2.5e-14** |
| MLIP (Orb-v3, fastest 2025 universal model) | 1,000 atoms | 217 fwd/s, >1e6 MD steps/hour | 1 GPU | **~2e-13** |
| Classical MD (OpenMM) | DHFR, 23,558 atoms | 1,031 ns/day | AMD V620 | **1.2e-11** |
| **ReaxFF** (the only one that actually breaks bonds) | — | 10–50× slower than classical MD, 0.25 fs timestep | — | **~1e-13 – 1e-12** |

**Real chemistry runs at 1e-17 to 1e-10 × realtime.** Three consequences stated
plainly, because the owner deserves the actual scale:

- Simulating **one second** of a 23,558-atom protein at 1,031 ns/day takes
  **2,740 years** of continuous GPU time.
- Simulating **one sim-day of one agent's life** at 1,000-atom MLIP fidelity
  costs **~1.1e11 years** on an H100 — about **7.8× the age of the universe**.
- 1,000 atoms is 1.7e-21 mol. One **gram of water is 3.3e22 molecules**, 1e19×
  more particles than the largest MLIP run that fits on a single GPU.

And this box has no GPU at all. **Chemistry at agent speed is not expensive.
It is impossible, by nineteen orders of magnitude, on any hardware that exists.**

*(The paragraph above is retained deliberately. It was the answer to the
question as originally posed, the owner asked for numbers rather than vibes, and
§3.2 supersedes the framing without contradicting a single figure in it.)*

### 3.2 SUPERSEDING FRAME (owner, mid-research): the caveman standard

> *"We don't actually need to understand chemistry for this — just like cavemen
> didn't."*

This is the correct reframing and it dissolves the problem rather than
compromising on it. Cavemen mastered fire, cooking, shelter, cordage and food
preservation with **zero** mechanistic knowledge. What they exploited were
regularities at the scale of their own senses and actions: dry wood burns, wet
wood doesn't, rain kills fire, cooked meat keeps you alive, night is cold, wind
makes fire spread. Not one of those requires a molecule.

So the requirement on Jack's world is not chemical realism at any fidelity. It
is three properties, and each is testable:

1. **CONSISTENT** — same action in same conditions produces the same outcome, so
   a rule is learnable at all. This is the falsifiable property that *replaces*
   realism. It is stronger than it sounds: it forbids hidden state, unseeded
   randomness, and order-dependence.
2. **DISCOVERABLE** — rules surface through ordinary interaction, within a
   number of steps an exploring agent will actually spend. A rule that requires
   a 12-step precondition chain no curiosity signal can reach is, for learning
   purposes, not in the world.
3. **CONSEQUENTIAL** — outcomes couple to the needs system. A rule that does not
   move warmth, food, water, rest or safety is scenery. Crafter's design is
   exactly this discipline: every achievement is on a path to not dying.

Under this frame, crafting-graph systems stop being "approximations of chemistry
that fall short" and become **the existing art for the thing we actually want:
rule engines**. Evaluate them as such.

### 3.3 What the art actually does — five patterns, and which to copy

**(a) Discrete recipe graphs — O(1) table lookup.** Minecraft: ~700 crafting +
~250 smelting/cooking + ~50 smithing + brewing ≈ **~1,000 recipes**, stored as
JSON data files. Crafter: **22 achievements** *are* the entire tech tree, and
Craftax reimplements it in JAX at ~250× the speed (≈44,630 steps/s on one V100
with 1,024 parallel envs; 1e9 env steps in under an hour on one GPU).

**(b) Material tables with real thermal tokens — Dwarf Fortress.** Materials
carry `[MELTING_POINT]`, `[BOILING_POINT]`, `[IGNITE_POINT]`, `[HEATDAM_POINT]`,
`[COLDDAM_POINT]`, `[SPEC_HEAT]`. Temperature is a 16-bit unsigned integer in
degrees Urist (°U = °F + 9968). The entire update rule: *"every tick, an item
exposed to a hotter/colder environment will adjust its temperature by the
difference divided by the item's specific heat."* That is first-order lumped
capacitance with the timestep folded into the constant. Notably the wiki flags
temperature as *"a known cause of lag"* and the subsystem is **user-disableable**
— even this is too expensive at DF's object count. A warning worth heeding.

**(c) Oxygen Not Included — the deepest shipped chemistry-lite system, and the
one to copy.** Heat unit DTU = 1055.06 J (a renamed BTU); specific heat capacity
in (DTU/g)/°C, where **water is 4.179, matching real water's 4.179 J/g/K
exactly**; thermal conductivity in (DTU/(m·s))/°C with transfer rate-limited by
the *lower* conductivity of the pair; grid elements undergo temperature-driven
phase transitions over 0–10,000 K. **The sim runs at 5 Hz — one tick = 0.2 s.**
And the wiki is explicit: *"ONI does not fully ensure the conservation of energy
or mass and runs quite limited simulations with a lot of hard-coded
transitions."*

That last sentence is the whole lesson of this section. **ONI feels like
chemistry because it uses real units, real specific heats and real phase
transitions on a coarse grid at 5 Hz — while openly abandoning conservation and
hand-authoring the transition table.** It is a thermodynamics game, not a
chemistry game, and it is *more* than deep enough to teach a caveman.

**(d) Object state machines — BEHAVIOR-1K.** Covered in §2.7; the equations are
there and they are ours to take.

**(e) Fire — cellular automata everywhere, including in science.** Minecraft:
scheduled tick every 30–40 game ticks (1.5–2 s), spread over 3×3 horizontally
and up to 4 blocks above / 1 below, ignition degree `= i + 7d + 40a + 30` with
thresholds 100 horizontal and 200/300/400 at +2/+3/+4, halved in humid biomes,
blocked by rain on the block or its four horizontal neighbours. Wildfire
science: **Rothermel (1972)**, `R = (I_R·ξ)/(ρ_b·ε·Q_ig) · (1 + Φ_w + Φ_s)` — a
no-wind base rate times a dimensionless wind-and-slope multiplier, still
underpinning BEHAVE/FARSITE/FlamMap, propagated by cellular automata on a raster
or Huygens-principle polygon growth. Wikipedia's wildfire-modeling page is blunt
that full 3-D combustion DNS *"does not exist, is beyond current
supercomputers."* **Even wildfire science refuses to simulate the chemistry of
fire.** We are in good company.

**(f) The RL-chemistry option, for completeness.** ChemGymRL (arXiv:2305.14177;
RSC Digital Discovery 2024, 10.1039/D3DD00183K) is a Gymnasium chemistry lab
with reaction/extraction/distillation/characterization benches. Its reaction
bench is literal rate-law ODEs — `∂[X]/∂t = −k[X][Y]` with Arrhenius
`k = A·exp(−Ea/RT)`, the Wurtz reaction as 6 coupled reactions — costing
microseconds per step. **Licence GPL-3.0**, which makes linking it a copyleft
event; take the *model* (rate laws + Arrhenius + a vessel state object), not the
code. Note also that ChemCrow (arXiv:2304.05376) and Coscientist (Nature, Dec
2023) do not simulate chemistry either: they *delegate* it to databases and real
robots. That is the honest architecture, and it is the opposite of a
self-contained world.

### 3.4 What is reachable, stated as a budget

Jack's world gets **1–10 ms per step, total, on cores shared with paying
tenants.** Within that:

**Reachable, and defensible as "real":**
- A discrete transformation graph with **element-count stoichiometry that
  actually balances** — O(1) lookup plus an integer conservation check.
- A **per-object lumped thermal ODE** (one multiply-add per object per tick):
  the Dwarf Fortress rule with honest SI units. **[measured: free]**
- **Tile-grid heat diffusion with latent-heat phase change**, decoupled from the
  agent tick and run at ONI's 5 Hz. **[measured: 0.12 ms per 128×128 tick =
  0.06% of one core at 5 Hz]**
- **Cellular-automaton fire** with a Rothermel-shaped `base × (1 + wind + slope)`
  multiplier.
- **Arrhenius kinetics on a hand-authored network of ≤20 species**, if we ever
  want cooking chemistry to have a temperature-dependent *rate* rather than a
  threshold. Dozens of ODE integrations per step is trivially affordable.

**Fantasy, and named as such so nobody re-litigates it:**
- Any atomistic simulation of anything. Any DFT, any MD, classical or reactive.
- Predicting the products of a reaction **not in the authored table**. Emergent
  novel chemistry. Discovering that mixing two things makes a third thing we
  did not write down.
- Molar quantities. Combustion chemistry. Metabolism from first principles.

### 3.5 The verification pattern, restated — because this is the load-bearing part

PG.2 checked buoyancy against Archimedes. It is tempting to read that as "we
verify against reality". **That is not what happened, and the distinction now
matters.** Archimedes' principle was *the rule we chose to implement*. The spec
pre-registered it, the implementation was gated on reproducing it to
err/radius 0.000, and the buoyancy-disabled control had to sink. What was
verified was that the implementation obeys its own published rule.

Fire, cooking and spoilage are the same, minus the luxury of a rule with a
Greek name. So:

> **The world-fidelity gate for a phenomenological rule is: the rule is
> PRE-REGISTERED in the spec, in prose and in closed form; the implementation
> reproduces it to a stated tolerance under conditions it was not tuned on; and
> a deliberately-broken variant must be caught by the same check.**
> **CONSISTENCY is the falsifiable property, not realism.**

This is strictly *more* testable than realism, not less, because it forbids the
escape hatch of "well, real fire is complicated". A pre-registered rule has no
complications it did not declare. And it inherits every existing lesson:
time-average what oscillates (PG.2), report per-partition and gate on the
minimum (ME.11), assert where the state can still tell outcomes apart (T0.12).

Where an analytic rule *is* available — the thermal model, because heat balance
is physics and we chose to implement the physics — we use it, and W.1 does.
Where it is not — fire, spoilage, tool-making — the rule *is* the spec text.

### 3.6 The honest caveat: a fixed rule-set has a floor

A hand-authored rule-set is a finite discovery tree. When Jack has climbed it,
open-ended learning stalls, and no amount of curiosity machinery will produce
novelty that the world cannot express. Crafter's 22 achievements are a *ceiling*
as well as a calibration.

The named lever is **ACCEL-style mutation of the world AND the rules across
lives** (arXiv:2203.01302), which `PlaygroundParams.mutate` already anticipates
for geometry and T5.08 already specs for scenes. Extending mutation from
geometry to *rules* (fuel ignition thresholds, spoilage rates, which materials
combine) is the W3 lever. **Do not design it now.** It is recorded here so that
when learning plateaus, the diagnosis is "the world ran out of tree", not "the
agent stopped learning" — those look identical in a coverage curve and this
paragraph is the difference.

---

## 4. THE FIDELITY LADDER — W0 → W3

Design principles, applied at every rung:

- **Overlays, not forks.** Each rung extends `playground.py` in the `Water`
  pattern: a class with pre-computed per-body constants, an `apply`/`step`
  method called around `mj_step`, an `enabled` flag so its own control can
  disable it, and a spec that gates it against a pre-registered rule.
- **Pure functions where possible**, so the eventual JAX port (§2.1) is
  mechanical rather than a rewrite.
- **Every rung has a control that must fail.** `SYSTEM.md` law 2.
- **Nothing new is claimed until PG.8's world is the one being extended.** PG.8
  PASSes, so Jack is in the room; the needs go on the Jack who is actually there.

### 4.0 The clock, decided first

| clock | 1 sim-day | physics steps @ dt=0.005 | wall on one core @ 2,826 steps/s |
|---|---|---|---|
| real time (k=1) | 86,400 s | 17.28 M | **1.70 h** |
| **Jack-day (k=72)** | **1,200 s (20 min)** | **240,000** | **85 s** |

k = 72 is proposed by analogy to Minecraft's 20-minute day, and it turns a
3-seed × 10-life × 10-day study from 510 core-hours into **7.1 core-hours**.
Need rates scale by k: a hunger deadline of 3 days becomes 60 min of sim-time;
thirst at 3 days likewise; sleep pressure's τ_wake of 18.2 h becomes 15.2 min.

**The hazard, and why W.7 exists:** the thermal model's time constants are in
*seconds* and do not scale with a wall-clock convention. If we naively speed the
clock, a 70 kg body's τ = 4.74 h in still air becomes 4 min of Jack-time, which
is a different physics, not a faster one. The rule is:

> **Physics runs at 1×, always. The CLOCK that drives need accumulation is
> scaled. The thermal ODE is integrated in real seconds, and the coupling to
> hunger/thirst/sleep uses the scaled clock. The two rates must be declared,
> and W.7 asserts that the dimensionless ratios (τ_thermal / day-length,
> hunger-deadline / day-length) are what the spec says they are.**

That still means Jack cools 72× "slower" relative to a Jack-day than a human
does relative to a human day — a deliberate, declared distortion, and the
alternative (scaling `h` or `c_p` to compensate) would break W.1's analytic gate.
Say it out loud in the spec rather than tuning it away.

### 4.1 W0 — the playground plus needs

**What is added.** Four overlays on the PG.8 world, in the `Water` pattern:

1. **`Thermal`** — a scalar temperature field over the arena (ambient as a
   function of position, height and time-of-day) plus a two-node body model on
   Jack (core + skin), driven by the standard heat balance. Heat sources are
   OmniGibson-style: radius-gated relaxation toward a source temperature.
2. **`Metabolism`** — hunger, thirst and sleep pressure as integrators with
   published rates; the apple becomes edible (it already exists, carries no
   reward, and PG.1 certifies its physics).
3. **`DayNight`** — a light cycle driving both the rendered scene and the
   ambient temperature term. Night is *colder*: that single coupling is what
   makes the day/night cycle a curriculum rather than a lighting effect.
4. **`Vitals`** — the death predicate. Not death mechanics yet (that is W1),
   just the thresholds and a terminal flag, so W0's gates can fire.

**What it teaches that the current playground cannot.** That Jack has *internal
state that his own actions change*. Everything in PG.1–PG.8 is about the room;
this is the first thing that is about him. It is also the minimum world in which
"eat the apple" is a discovery rather than a decoration — today the apple on the
platform is, by the playground's own docstring, an object like any other.

**Cost. [measured]** Thermal + metabolism + sleep pressure + 8 heat sources,
stepped at 50 Hz alongside 200 Hz physics: **3,009 → 2,826 steps/s, −6.1%**. The
fire/heat CA that W1 needs is already measured at **0.06% of one core**. Memory
increase is a few hundred bytes. **W0 is, to within measurement noise, free.**

**Fidelity gate: W.1**, and it is the PG.2 pattern exactly. The heat balance has
a closed-form solution and a parameter-free prediction:

```
m·c_p·dT/dt = Q_gen − h·A·(T − T_env)
τ = m·c_p/(h·A)          T_eq = T_env + Q_gen/(h·A)
```

With Du Bois `A = 0.007184·H^0.725·W^0.425` = **1.8481 m²** for 175 cm / 70 kg
**[computed]**, 1 met = 58.2 W/m² → **Q_gen = 107.56 W [computed]**, still-air
`h = h_c + h_r = 3.0 + 4.7 = 7.7 W/m²K` → `hA = 14.23 W/K`, and ~20 W of
respiratory/insensible evaporation:

- **Thermoneutral point: T_air = 33.7 − (107.56−20)/14.23 = 27.55 °C
  [computed].** The measured nude thermoneutral zone is 27–31 °C. **This is the
  Archimedes of the temperature model: a closed-form number with no fitted
  parameters that the implementation either hits or does not.**
- **Pure-decay gate:** Q_gen = 0, T₀ = 37 °C, T_env = 20 °C, still air, τ =
  4.7413 h → **T(1 h) = 33.7674 °C [computed]**. Assert to 1%.
- **Wind gate:** h_c(5 m/s) = 8.6·5^0.53 = 20.18 → h = 24.88, and τ must shrink
  by **exactly 0.3095** **[computed]**. A model whose τ does not move with wind
  is not modelling convection.
- **Conservation gate:** ∫Q dt = m·c_p·ΔT to integrator tolerance.

Constants, sourced rather than invented: 1 met = 58.2 W/m²; 1 clo = 0.155
m²K/W; neutral skin 33.7 °C, neutral core 36.8 °C (Gagge two-node, as shipped in
CBE's `pythermalcomfort`); h_r = 4.7 W/m²K; h_c natural = 3.0·p^0.53, forced =
8.6·(v·p)^0.53; sweating `m_rsw = 170·warm_b·exp(warm_sk/10.7)` capped at 500
g/h/m² (= 0.92 L/h ≈ 617 W of evaporative cooling at 1.85 m²).

> **A citation trap worth recording.** The familiar `c_p = 3,470 J/kg/K` is
> **Burton (1935), who assumed the body is 50% fat / 50% blood — it was never
> measured.** Xu, Rioux & Castellani (*Temperature* 2022;10(2):235–239,
> doi:10.1080/23328940.2022.2088034) recompute it from tissue data as **2,980
> J/kg/K (range 2,440–3,390)**. Use 3,470 so our numbers reconcile with the
> ASHRAE/Gagge reference implementations any reviewer will check against, and
> record in the code comment that the physiologically correct value is ~2,980
> and shortens every time constant by 14%. This is the "assert contracts against
> the source of truth" lesson applied to a physical constant.

Metabolic constants: BMR ≈ 1,700 kcal/day = **82 W**; active 2,500 kcal/day =
**121 W**. Note the double-counting trap: 1 met × 1.8481 m² = 107.6 W = 2,195
kcal/day is *higher* than BMR because **1 met is seated rest, not basal**. If
the sim uses met units it must not also apply a separate BMR.

Sleep, two-process model (Borbély 1982; Daan, Beersma & Borbély 1984, *Am J
Physiol* 246:R161–R178): `dS/dt = (UA−S)/τ_w` with **τ_w = 18.2 h** awake,
`dS/dt = −(S−LA)/τ_s` with **τ_s = 4.2 h** asleep. Sleep pressure discharges
~4.3× faster than it accumulates, which is why 8 h clears 16 h.

### 4.2 W1 — the survival playground

**What is added.** Weather (wind speed driving `h_c`, rain), **fire** as a
pre-registered state machine, cookable food, spoilage timers, shelter-buildable
materials (MuJoCo 3.0 flex for cloth/cordage; welded/jointed geoms for
structure), food scarcity, and **death with respawn** (§6).

**What it teaches that W0 cannot: deferred consequence.** W0 teaches "I am cold,
move toward warm". W1 teaches "the sun is going down and I must build *now* for a
cost I will not pay for twenty minutes". No W0 gradient contains that. It is
also the first world where the diary can pay: the location of last night's
shelter is worth remembering, which is what makes W.6 (life N+1 beats life N)
measurable at all.

**The fire rule, pre-registered here so W.5 can gate it.** Deliberately
Minecraft-shaped rather than Rothermel-exact, because the caveman standard is
consistency, not combustion science:

```
state(fuel_cell) ∈ {UNLIT, BURNING, EMBERS, ASH}
wetness w ∈ [0,1]           # raised by rain and immersion, decays with exp(-t/tau_dry)
ignite:   UNLIT -> BURNING   iff  (adjacent BURNING or ignition source)
                             and  w < W_IGNITE            # dry wood burns, wet does not
burn:     BURNING consumes fuel at rate r*(1 + a*wind);   emits P_fire watts
spread:   P(ignite neighbour) = p0 * (1 + k_wind * wind·direction) * (1 - w_nb)
rain:     BURNING -> EMBERS  when rain_rate > R_QUENCH
exhaust:  BURNING -> EMBERS -> ASH   as fuel -> 0
heat:     any BURNING cell is a heat source for W.1's thermal model
```

Every constant in that block is a declared parameter. **W.5 asserts the
implementation obeys it, not that it resembles fire.** The interactions —
rain wets fuel which then will not ignite; wind both accelerates consumption and
biases spread — are where W3's depth will come from, and they are already
present in embryo.

**Cost.** The fire CA is measured free. Weather is a scalar. Shelter geometry
adds bodies: our measurement shows 8 bodies → 21 bodies (Jack) costs 6,632 →
3,404 steps/s **[measured]**, so shelter parts are the one W1 addition with a
real price — budget a further 20–40% for a dozen structural geoms and *measure
it* rather than assuming. A 10-Jack-day life at ~2,000 steps/s ≈ **20 min wall**;
3 seeds × 10 lives ≈ **10 core-hours**, i.e. one overnight run on two cores.

**Fidelity gates: W.3** (an insulated shelter changes survival time by the
amount the heat balance predicts; a zero-insulation shelter must not help) and
**W.5** (fire obeys the block above; a broken variant must be caught).

### 4.3 W2 — Crafter-class rule depth, on the MuJoCo body

**This rung is where the embodiment question is resolved, and the resolution is:
we do not adopt a gridworld, we adopt its rule graph.**

**What is added.** A transformation graph with condition gates and tools: raw →
cooked (heat + time), fresh → spoiled (timer, slowed by cold or smoke), wood +
effort → shelter parts, stone + wood → tool, tool → faster effort. Plus the
social need, which has been deferred through W0 and W1 and needs stating: the
owner's "he needs company" is met by a *conversational* channel (the ME family
already proves the substrate) and an avatar presence, not by a second physics
body — a second humanoid roughly doubles physics cost and buys nothing the
first one has not yet earned.

**What it teaches that W1 cannot: technology.** A chain of actions that no single
reward describes, discovered because each link is individually consequential.
Crafter is the calibrated reference for whether such a tree is climbable —
22 achievements with published baselines for DreamerV2/V3, PPO, Rainbow and
human play. We should use its *depth* as the target and its *score definition*
as the shape of our coverage metric, while the body stays ours.

**Cost, and the honest blocker.** W2 needs vision — a tool is recognised, not
proprioceived — and **this box cannot render.** `MUJOCO_GL=osmesa` fails here
with `'NoneType' object has no attribute 'glGetError'` **[measured]**, consistent
with the known box limitation (Xvfb + llvmpipe is the workaround already
established for WorldTwin screenshots). **PG.6, the spec that would certify the
playground's camera, is NOT_RUN.** So W2's real cost is not the rule graph
(which is table lookups) but the perception stack, and it should be sequenced
behind PG.6. Rendering at 64×64 through llvmpipe will be the throughput
bottleneck, not physics; measure it before planning around it.

**Fidelity gates: W.4** (consistency + discoverability of the whole rule-set)
and **W.5** extended to the transformation graph.

### 4.4 W3 — maximum rule DEPTH (re-costed under the caveman frame)

**Under the original framing this rung was "maximum physical realism" and its
true cost was Genesis or OmniGibson — i.e. unreachable. Under the owner's
correction it becomes something we can actually build, and it gets dramatically
cheaper.**

**What is added.** Not better physics. **Rules that interact**, such that
situations arise that we never enumerated: wind-driven fire spread through fuel
whose wetness depends on rain that depends on the weather model that depends on
the day cycle; smoke that both preserves food and drives Jack out of a shelter;
a shelter that traps heat and therefore also traps smoke. Plus, later, ACCEL-
style **mutation of the rule parameters across lives** so the tree itself moves.

**What it teaches that W2 cannot: open-endedness.** In W2 the discovery tree is
a list we wrote and Jack can finish it. In W3 the reachable state set is the
*closure* of the rules under composition, which is not a list anyone wrote.

**Cost, honestly.** Rule interactions are nearly free per step — they are the
same table lookups with more edges. The cost of W3 is in two places, and neither
is physics:

1. **Episode volume.** Open-endedness is measured over many lives, and the
   T5.08 spec already asks for eight weeks of non-plateauing growth. At 20 min
   wall per 10-day life, thousands of lives is thousands of core-hours: this is
   where Kaggle's 30 h/week matters, and it is gated on the JAX port (§2.1),
   because our overlays are Python callbacks that MJX cannot run. **Cost the
   port before costing the science.** Vendor MJX throughput figures (3.35 M
   steps/s) are on GPUs far newer than a P100; assume 5–20× worse and measure.
2. **The measurement itself.** Counting "situations nobody enumerated" requires
   a definition that cannot be gamed, which is W.8's whole difficulty.

**Fidelity gate: W.8.** Reachable distinct rule-interaction events must exceed
the enumerated rule count, and — the control — must saturate at the enumerated
count when interactions are disabled.

### 4.5 What each rung teaches, in one line each

| rung | the sentence that becomes true |
|---|---|
| W0 | "I have a body whose state I can change." |
| W1 | "What I do now determines whether I am alive later." |
| W2 | "Things combine into other things, and tools make it faster." |
| W3 | "The world has more in it than anyone told me." |

---

## 5. SPECS

Eight entries were designed; **seven are live** after the reconciliation in §5.0
withdrew W.6 in favour of `NE.08`. `W.` prefix. Checked against the live
registry twice during this session — **128 ids at the start, 136 at the end**
(a concurrent agent registered `LC.00`–`LC.06` and `PS.01` while this was being
written) — and on both checks, none
beginning `W.` or `SV.`, no collision. **Note the prefix hazard from
`docs/LESSONS.md`** — `_module_for` globs `w_1_*.py`, which would match
`w_1_0_*.py`; if any of these ever gains arms, resolve parent and child before
writing a test.

Root dependency for all of them is **PG.8** (Jack is in the world and can act),
which PASSes. That is deliberate and follows the CU.1 precedent: a needs spec
run in an empty room measures nothing.

```python
    # ── THE SURVIVAL WORLD (docs/research/SURVIVAL_WORLD.md) ────────────
    # Owner directive 2026-08-09: Jack gets human needs and is thrown into as
    # real a survival world as we can build; he lives, dies, and tries again.
    # Owner correction, same day: "we don't actually need to understand
    # chemistry for this — just like cavemen didn't." So the world's rules are
    # PHENOMENOLOGICAL and the falsifiable property is CONSISTENCY with a
    # PRE-REGISTERED rule, not correspondence with nature. Where an analytic
    # law is available (heat balance) we gate on it exactly as PG.2 gated
    # buoyancy on Archimedes; where it is not (fire, spoilage) the rule text in
    # the spec IS the oracle, and a deliberately-broken variant must be caught.

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
               "compressed Jack-day clock, which W.7 governs."),

    Spec("W.2", 2, "Needs are a conserved ledger, and they can kill",
         hypothesis="Hunger, thirst and sleep pressure integrate to their "
                    "closed-form solutions within 1%; energy in equals energy "
                    "out to 1e-6 relative over a 10-day life; each need "
                    "independently reaches a lethal threshold at the "
                    "pre-registered deadline (thirst 3 days, food 3 weeks, core "
                    "temp outside 28-40 C) when and only when it is not met; "
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
         kills="W.3, W.6 and the whole death-and-retry loop. A needs system "
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
               "NATA), and mark the tail as extrapolated in the code."),

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
               "must declare which before the run."),

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
               "reported (T2.02's 'matched steps has more than one meaning')."),

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
               "simulate fire's chemistry."),

    # ~~W.6~~ WITHDRAWN 2026-08-09 in favour of NE.08 in
    # docs/research/NEEDS_AND_DEATH.md, which separates the three claims this
    # version conflated and adds the C-ONELIFE control. Retained below, struck
    # through, so the reasoning trail survives. DO NOT REGISTER. See §5.0.
    Spec("W.6", 5, "[WITHDRAWN — superseded by NE.08] Life N+1 is better than life N, because of what life N wrote down",
         hypothesis="Across >=10 consecutive lives in the same mutated world "
                    "family, time-to-death increases and the number of distinct "
                    "needs successfully met per life increases, and the gain is "
                    "attributable to the persistent diary: an agent whose "
                    "episodic store carries across death beats an identical "
                    "agent whose store is wiped at every death, at matched "
                    "total environment steps.",
         falsified_by="No improvement across lives, OR improvement that "
                      "survives wiping the diary — in which case the gain is in "
                      "the weights or in the optimiser state and the claim "
                      "'death is a page turn, not a reset' is unsupported.",
         null_baseline="WIPED-DIARY agent (same weights, same steps, store "
                       "cleared on death) and a SHUFFLED-DIARY agent (store "
                       "carried but its entries permuted across lives, "
                       "destroying the correspondence between a memory and the "
                       "world it came from) — the second is the real bar, "
                       "because carrying ANY text across death changes "
                       "retrieval statistics.",
         metric="cross_life_survival_gain", budget=Budget.CPU_LONG,
         depends_on=["W.2", "ME.10", "ME.11"], seeds=3,
         control="The shuffled-diary agent MUST NOT improve. If permuting the "
                 "diary leaves the gain intact, the agent is being helped by "
                 "the presence of memories rather than by their content, which "
                 "is ME.9's provenance-swap control transplanted to the death "
                 "loop.",
         kills="GOAL.md's sentence 'Death is not a reset; it is a page turn.' "
               "This is the smallest experiment that could establish it, and if "
               "it fails, no larger one rescues the claim.",
         notes="Report per life, not averaged over lives (PG.4's per-seed "
               "lesson): a monotone improvement and a single lucky life have "
               "the same mean. The episode boundary is a DEATH, not a step "
               "count, so lives have unequal length by construction and every "
               "comparison must be at matched TOTAL ENV STEPS with the other "
               "three budgets reported. ME.10's double dissociation is the "
               "substrate this stands on: it already proves the diary and the "
               "skill are separable stores."),

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
         kills="Every cost estimate in this document. Without compression a "
               "single 24 h life is 1.70 core-hours (measured) and no 3-seed "
               "study is affordable; with UNVERIFIED compression every thermal "
               "number in the ladder is silently wrong.",
         notes="Same family as T0.15: the machinery BETWEEN a measurement and "
               "its threshold is part of the gate. Here the machinery is a unit "
               "conversion applied to time, which is exactly the class of thing "
               "that passes review and fails silently. Assert the ratios "
               "against the DECLARED constants, not against each other — two "
               "quantities derived from the same wrong k agree perfectly."),

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
         # NOT parented on T5.08 (open-endedness), deliberately. W.8 asks
         # whether the WORLD has depth, which a scripted and a random agent can
         # answer with no learning at all; T5.08 asks whether an AGENT keeps
         # finding it, which is GPU_LONG and unrun. Parenting a world-property
         # claim on an agent result is exactly how UB.1 -- the project's
         # namesake claim -- ended up unreachable behind a locomotion failure.
         # Caught 2026-08-09 by running this block's depends_on against the
         # live registry rather than eyeballing it.
         kills="The W3 rung. If depth does not exceed the enumeration, W3 is "
               "just W2 with more table rows, and the honest move is to say so "
               "and spend the compute on ACCEL-style RULE mutation instead of "
               "on more hand-authored rules.",
         notes="The metric's difficulty is definitional, not computational: "
               "'an outcome that differs from either rule alone' must be "
               "computed by actually running the counterfactual single-rule "
               "outcomes, which doubles the world's cost during measurement "
               "and is why this is CPU_LONG. Report the growth CURVE, never a "
               "final count (ME.5's standing-spec pattern). Honest caveat "
               "recorded in section 3.6: a fixed rule-set has a floor, and when "
               "this saturates the diagnosis is 'the world ran out of tree', "
               "not 'the agent stopped learning' — the two look identical in a "
               "coverage curve."),
```

### 5.0 RECONCILIATION with `docs/research/NEEDS_AND_DEATH.md`

A companion document was written in parallel on 2026-08-09 and registers
`NE.00`–`NE.09`. **Two of my eight specs collide with it, and leaving two specs
claiming one capability is exactly the corruption the ledger exists to prevent.**
Resolved here, in the open:

| my spec | theirs | resolution |
|---|---|---|
| **W.2** (needs are a conserved ledger) | **NE.01** (the needs are a real control problem: nobody survives by accident) | **Both stand — they are different levels.** W.2 is the PG.1-style *fixture* gate: the integrators match closed form, energy conserves, a sated agent never dies, each need's ablation removes exactly its own death mode. NE.01 is the *agent-level* claim built on top. Register W.2 as a dependency of NE.01, the way PG.1 gates every curiosity spec. Without a fixture gate underneath it, a failing NE.01 cannot distinguish "the agent can't survive" from "the bookkeeping is broken" — which is `docs/LESSONS.md`'s "when the simplest possible learner also fails, the TASK is broken". |
| **W.6** (life N+1 beats life N because of the diary) | **NE.08** (DEATH AND RETRY: life N+1 is faster BECAUSE he remembers) | **W.6 is WITHDRAWN in favour of NE.08.** NE.08 is strictly stronger: it separates three claims my version conflated (A: cross-life improvement exists; B: the diary is the mechanism; C: the death loop beats the same experience without death), and carries a control I did not have — `C-ONELIFE`, which tests whether *dying* is doing any work at all rather than merely costing time. My W.6 would have measured A and B and silently assumed C. It stays in this document, struck through, because the reasoning trail matters and because the withdrawal is itself the finding. |

**The thing worth carrying forward from W.6 into NE.08** is one measurement
hygiene point that its §5 does not appear to state: because the episode boundary
is a *death*, lives have unequal length by construction, so every cross-life
comparison must be at matched **total env steps** with the other three budgets
reported (T2.02's "matched steps has more than one meaning"). If NE.08 already
does this, ignore; if not, it is a one-line addition that prevents a confound
where the better agent simply lived longer and therefore trained more.

**Live spec set after reconciliation: W.1, W.2, W.3, W.4, W.5, W.7, W.8 (seven).**

**Status note, verified at the end of this session:** neither `NE.*` nor `W.*`
is in the live registry yet — both documents are designs awaiting registration.
So this reconciliation is cheap to act on *now* and expensive later: once both
families are registered, two specs claiming one capability is a ledger defect
rather than a document disagreement. Whoever registers first should carry §5.0
with them.

### 5.1 Notes on the spec set

- **W.1 and W.5 are the two world-fidelity gates in the PG.1/PG.2 tradition.**
  W.1 has an analytic oracle (heat balance) and is the strict analogue of
  Archimedes. W.5 has a *published-rule* oracle, which is what §3.5 argues is
  the correct generalisation when nature offers no closed form.
- **W.4 is the spec that makes the owner's correction enforceable.** Without it,
  "consistent, discoverable, consequential" is a design intention. With it,
  a hidden-state bug in a rule is a red ledger entry.
- **W.7 exists because of a near-miss found while costing this document**: the
  compression factor that makes the whole ladder affordable also silently
  invalidates every thermal constant, and nothing else in the ladder would have
  caught it.
- **W.6 was the payoff spec and is withdrawn** (§5.0). Its one durable
  contribution — parenting the cross-life claim on ME.10/ME.11 rather than on
  any locomotion result, so a D1 outcome cannot block it (the UB.1 dependency
  lesson) — should be checked against NE.08's own `depends_on`.
- **Nothing here depends on a GPU.** Six of the seven are `Budget.CPU` or
  `CPU_LONG` by design: the world is cheap (§7.1) and a hypothesis about the
  world should die on CPU minutes before it reaches Kaggle's quota. That is the
  ladder's own front-loading rule, and this family happens to satisfy it
  completely.
- Suggested tiers: W.1–W.5 and W.7 at tier 2 (a component vs a null), W.8 at
  tier 5 (the claims).

---

## 6. DEATH mechanics

### 6.1 What happens physically at death, per candidate world

| world | physical death | respawn semantics |
|---|---|---|
| **Ours (W0–W3, MuJoCo)** | Actuation is zeroed and the body becomes a passive ragdoll under gravity and contact. This is not a cosmetic choice: it means death has a *proprioceptive signature* — the last seconds of a life contain the collapse. Nothing is teleported and no state is discarded at the instant of death. | The episode terminates on a Vitals predicate; the world is rebuilt from `PlaygroundParams` (optionally mutated), Jack is re-spawned at `params.spawn()`, needs reset to their initial values, **the diary and the weights persist**. |
| Crafter | health reaches 0; episode ends | full env reset, no persistence |
| NetHack / NLE | death is terminal and permanent per episode; the roguelike loop *is* death-and-retry, which is why it is the cheapest place to study the pattern | new dungeon seed |
| Minecraft family | health 0; items drop; respawn at spawn point | world persists, agent does not |
| Neural MMO | agent removed from a persistent multi-agent world | others continue |
| Avalon | death on hazard/starvation | task reset |

The row that matters is the first, and its distinguishing property is that
**death is continuous with life** — a ragdoll collapse is a physical event in the
same state space as everything before it. In every other world death is a
discrete flag. If we want Jack to *learn from dying* rather than merely to be
reset by it, the last seconds have to be data.

### 6.2 The episode boundary when a life is days of sim time

This is the genuinely hard design question in the brief, and it has three
candidate answers. They are not equivalent and the choice must be
pre-registered, because every RL quantity — returns, advantages, bootstrapping,
"matched steps" — is defined relative to it.

**(a) The life is the episode.** Clean semantics, and the only choice that makes
"time-to-death" the return. Fatal problem at our scale: a life of 10 Jack-days
is 2.4 M physics steps and 480 K control steps, so a PPO batch would contain a
fraction of one episode and credit assignment over a 480 K-step horizon is
beyond anything we can train on free compute.

**(b) The day is the episode; death interrupts it.** Bootstraps at the day
boundary (not a true terminal), terminates genuinely at death. Horizon is 48 K
control steps — still long, but tractable with value bootstrapping. This is
Crafter's shape (fixed-length episodes with early termination on death).

**(c) A fixed step budget is the episode; both death and survival are outcomes.**
Shortest horizon, most stable learning, and it decouples the *learning* episode
from the *narrative* life entirely.

**Recommendation: (b) for training, (a) for reporting.** The RL episode is a
Jack-day with bootstrapping; the *life* is the unit W.6 measures and the unit
the owner watches. They are different objects and conflating them is how
"matched steps has more than one meaning" (LESSONS.md) would bite here. **State
both, always.**

### 6.3 What survives death — the design that makes the loop worth running

Three tiers, and the boundaries between them are what W.6 tests:

1. **Persists always: the diary.** The episodic store (`EpisodicMemory`,
   `Persistence`) carries across death untouched. This is the owner's "what
   survives death is the point".
2. **Persists always: the weights.** Skills distilled from experience.
   ME.10 already proves diary and skill are separable stores with a double
   dissociation, which is precisely the substrate W.6 needs.
3. **Never persists: body state and needs.** Position, velocity, hunger, core
   temperature. A new life starts cold and hungry, or death costs nothing.

**The open question:** should the *world* persist? If Jack's shelter is still
standing in life N+1, improvement across lives is confounded — he inherits his
own construction rather than his own knowledge, and a diary-wipe control cannot
separate them, because wiping the diary would not remove the shelter.
**Recommendation: the world resets and only the diary and weights carry, so the
cross-life spec measures learning rather than inheritance.** A persistent world
is a legitimate later variant and should be its own spec, not a default.
`docs/research/NEEDS_AND_DEATH.md` §5.3 ("The world regenerates — and how far")
addresses this from the needs side; **the two documents must agree before either
spec runs**, and if they do not, it belongs in `docs/DECISIONS_NEEDED.md`.

### 6.4 Respawn hygiene

- **Re-spawn must not be a state leak.** `PlaygroundParams.humanoid_spawn` exists
  and PG.8's control already uses it to place Jack outside the arena. A new life
  must rebuild the model, not reset `qpos` in place, or accumulated solver state
  and `xfrc_applied` residue carry across the boundary.
- **The death predicate must be evaluated on the same clock as the needs.** A
  predicate polled at physics rate against needs updated at 50 Hz will report
  death up to 20 ms early or late — irrelevant physically, but it makes death
  time non-reproducible, which W.4's consistency check would then flag as a
  world bug. Fix the ordering once, in the overlay.
- **Log the cause of death, always.** Not for the ledger: for the diary. "I died
  cold at the pool" is the memory that can make life N+1 different, and W.6 is
  unmeasurable if the cause is not recorded.

---

## 7. Measurements taken on this box, 2026-08-09

All at `nice 19`, single process, MuJoCo 3.2.3 / gymnasium 1.1.1 / torch 2.8.0+cpu
/ numpy 2.0.2 / Python 3.9.25 on aarch64, 4 cores, 22 GB RAM.

### 7.1 Playground physics throughput

Timestep 0.005 s, RK4, after a 200-step settle, over 20,000 steps:

| configuration | nbody | nu | ngeom | steps/s | × realtime |
|---|---|---|---|---|---|
| world only, no water callback | 8 | 0 | 35 | **6,632** | 33.2 |
| world only, water callback | 8 | 0 | 35 | **6,117** | 30.6 |
| **Jack in world, no water callback** | 21 | 17 | 52 | **3,404** | 17.0 |
| **Jack in world, water callback** | 21 | 17 | 52 | **3,009** | 15.0 |
| **Jack + water + needs overlay @ 50 Hz** | 21 | 17 | 52 | **2,826** | 14.1 |

**The needs overlay costs 6.1%.** Putting Jack in the room costs 44%. The `Water`
callback costs 8–12%. Every one of these is affordable; only the policy is not.

### 7.2 Fire / heat cellular automaton

128×128 grid, numpy roll-based 4-neighbour update with fuel consumption:
**0.1202 ms per tick**. At ONI's 5 Hz tick rate that is **0.06% of one core**. A
tile-grid heat/fire layer is, for our purposes, free.

### 7.3 Policy forward cost — the actual budget

Single-threaded (`OMP_NUM_THREADS=1`), `eval()` mode, batch 1:

| policy | params | ms/forward | forwards/s | cost per Jack-day @ 40 Hz control (48 K steps) |
|---|---|---|---|---|
| MLP 348→256→256→17 | 0.16 M | **0.098** | 10,234 | **4.7 s** (+5.5% over physics) |
| MLP 348→1024×7→17 | 6.67 M | **2.359** | 424 | **113 s** (+133%) |
| Transformer 6L d512, 32 tokens | 18.9 M | **49.334** | 20 | **2,368 s = 39 min** (28× physics) |

**This is the finding that should govern the whole programme.** The world can be
made ten times richer for a few percent. Making the *brain* three times bigger
costs more than the entire world. The 57M-vs-54K lesson, arriving from the
compute side.

### 7.4 Rendering: unavailable natively

`MUJOCO_GL=osmesa` fails on this box: `AttributeError: 'NoneType' object has no
attribute 'glGetError'`. Consistent with the established box limitation
(WorldTwin's screenshot pipeline uses **Xvfb + llvmpipe**). Consequences: **PG.6
(the playground's camera) is NOT_RUN and is on W2's critical path**, and any
vision-dependent survival claim must budget software rasterisation before it
budgets learning.

### 7.5 Derived costs

| unit | physics steps | wall @ 2,826 steps/s, one core |
|---|---|---|
| 1 sim-hour (k=1) | 720,000 | 4.25 min |
| **1 real day (k=1)** | 17.28 M | **1.70 h** |
| **1 Jack-day (k=72, 20 min sim)** | 240,000 | **85 s** |
| 10-Jack-day life | 2.4 M | **14.2 min** |
| 3 seeds × 10 lives × 10 Jack-days | 72 M | **7.1 core-hours** |

With two cores left free for tenants and two used at `nice 19`, a 3-seed W1
study is a single overnight run. **The clock decision in §4.0 is what makes that
true, and W.7 is what makes it honest.**

### 7.6 Thermoregulation constants, computed

| quantity | value |
|---|---|
| Du Bois BSA, 175 cm / 70 kg | **1.8481 m²** |
| Q_gen at 1 met | **107.56 W** |
| hA, still air (h = 7.7) | 14.23 W/K |
| hA, 5 m/s wind (h = 24.88) | 45.98 W/K |
| **Thermoneutral T_air, still air** | **27.55 °C** (measured nude TNZ 27–31 °C) |
| Thermoneutral T_air, 5 m/s | 31.80 °C |
| τ, still air, c_p = 3470 | 17,069 s = **4.74 h** |
| τ, 5 m/s wind | 5,282 s = 1.47 h |
| τ, water immersion (h ≈ 200) | 657 s = **11 min** |
| τ, still air, c_p = 2980 (measured c_p) | 14,658 s = 4.07 h |
| **Decay gate: T(1 h), 37 °C → 20 °C still air** | **33.7674 °C** |
| **Wind gate: τ ratio still→5 m/s** | **0.30947** |

These are W.1's oracle. They were computed here, from the sourced constants, not
copied from a secondary summary.

### 7.7 Comparative environment throughput, same box, same cores

Measured on these four shared ARM cores; full table with needs/licence/body
columns in §1.8. Reproduced here because the *ordering* is the finding:

```
XLand-MiniGrid (4096 envs) 1,728,100 sps    no needs, no body
NLE NetHackChallenge (1c)     18,830 sps    hunger,  no body
Kinetix size s                 6,056 sps    no needs, 2-D low-DoF body
MiniHack Room-5x5 (1c)         3,401 sps    hunger,  no body
--- our playground + Jack + water + needs   2,826 sps  ALL NEEDS + FULL BODY ---
Humanoid-v5 (bare)             2,222 sps    no needs, full 3-D body
dm_control humanoid/stand      2,150 sps    no needs, full 3-D body
Craftax-Classic                1,711 sps    4 needs, no body
Craftax full               365 -   557 sps  5 needs + RPG attrs, no body
Crafter (1 core)           ~100 -   124 sps 4 needs, no body
Kinetix size l                    59 sps    no needs, 6-joint body (unusable)
```

**Read the middle of that list.** Our own world — every need plus a full
proprioceptive body — sits *above* bare `Humanoid-v5` on env-steps and **5–8×
above full Craftax**, which has fewer needs and no body at all. The
configuration this document recommends is not a compromise made affordable by
scoping; it is, on measured throughput, one of the cheaper options available.

Two reconciliations, stated so the numbers can be trusted:
- Our 2,826 is **`mj_step`s**; Humanoid-v5's 2,222 is **env-steps at
  frame_skip=5**, i.e. 11,110 `mj_step`s. Bare Humanoid-v5 is genuinely 3.7×
  faster *per physics step* — see the geometry lever in §1.8.
- Craftax's compile time (16–54 s) is excluded from its steady-state figure and
  is a real cost for short runs.

---

## 8. Recommendation

1. **Build W0 now.** It is measured at 6.1% overhead on a world that already
   PASSes PG.8, it needs no new dependency, and W.1 gives it an Archimedes-grade
   gate on day one.
2. **Decide the clock before writing a line of overlay** (§4.0) and land W.7
   alongside W.1, because the compression factor that makes the ladder
   affordable is also the thing most likely to silently corrupt it.
3. **Port OmniGibson's thermal/fire equations, not its software** (§2.7). They
   are a per-object scalar, two exponential relaxations, a sphere query and a
   latching ignition threshold, and they come with provenance we can cite.
4. **Treat Genesis as a scheduled measurement, not a plan** (§2.3): three
   questions, one sitting, into the ledger. Python 3.9 on this box is a real
   blocker.
5. **Sequence W2 behind PG.6**, because this box cannot render and vision is
   W2's actual cost.
6. **Do not adopt a gridworld** — and note this is now an argument from cost as
   well as from capability. Adopt Crafter's rule depth, Craftax's 5-need model
   and Crafter's geometric-mean score definition; keep the body. A gridworld
   agent cannot fall off a ladder, falling off the ladder is the project, and
   on measured throughput the body is the *cheaper* half (§1.8, §7.7).
6b. **Audit the playground's collision geometry** before treating any W1/W2 cost
   as fixed. 52 geoms against bare Humanoid-v5's 18 accounts for a 3.7× gap in
   physics throughput **[measured]**; perimeter walls, pool walls and the pool
   floor are contact-checked every step. Up to ~3× is available for free, which
   is larger than any algorithmic saving on the table.
7. **Rename `playground.py`** to avoid the MuJoCo Playground collision
   (arXiv:2502.08844) — free now, confusing later.

---

## Appendix A — sources

**Simulators.** MuJoCo docs (overview, API types, fluid computation, extensions,
MJX, MJWarp) · MuJoCo Playground arXiv:2502.08844 · JAX installation docs (CUDA
12 = SM 5.2+, CUDA 13 = SM 7.5+) · Isaac Sim 5.1 requirements · Isaac Lab
arXiv:2511.04831 · Newton (Linux Foundation, Sept 2025) ·
Genesis-Embodied-AI/Genesis + Genesis sensor docs + Stone Tao, "How fast is the
new hyped Genesis simulator?" · Habitat 3.0 arXiv:2310.13724 + habitat-sim README
+ issue #2358 · AI2-THOR object-types docs · ProcTHOR arXiv:2206.06994 ·
Holodeck arXiv:2312.09067 · ThreeDWorld arXiv:2007.04954 · Unity ML-Agents
Release 23 · Godot RL Agents arXiv:2112.03636 · Avalon arXiv:2210.13417 ·
Craftium arXiv:2407.03969 · BEHAVIOR-1K arXiv:2403.09227 + `omnigibson/
object_states/{temperature,heat_source_or_sink}.py` + issue #1865 · ManiSkill3
arXiv:2410.00425 · RoboCasa arXiv:2406.02523 · GRUtopia arXiv:2407.10943 ·
InfiniteWorld arXiv:2412.05789.

**Rule engines / crafting.** Crafter arXiv:2109.06780 · Craftax arXiv:2402.16801
· Minecraft wiki (Fire, recipes) · Dwarf Fortress wiki (Temperature) · Oxygen
Not Included wiki (Units, Thermal Conductivity) · Rothermel 1972, USFS GTR-371 ·
Wikipedia, Wildfire modeling · ChemGymRL arXiv:2305.14177 + RSC Digital Discovery
10.1039/D3DD00183K · ChemCrow arXiv:2304.05376 · ACCEL arXiv:2203.01302.

**Chemistry cost.** MACE-MP-0 arXiv:2401.00096 · MACE fine-tuning arXiv:2506.21935
· Orb-v3 arXiv:2504.06231 · MatterSim arXiv:2405.04967 · UMA arXiv:2506.23971 ·
OMat24 arXiv:2410.12771 · OpenMM 8 arXiv:2310.03121 · ReaxFF, npj Comput. Mater.
2016 · GNoME (Nature/DeepMind, 2023).

**Physiology.** ASHRAE 55 / ISO 7730 / Fanger · Gagge two-node as implemented in
CBE `pythermalcomfort` · Du Bois BSA (StatPearls) · Xu, Rioux & Castellani,
*Temperature* 2022;10(2):235–239, doi:10.1080/23328940.2022.2088034 (c_p is
2,980, not Burton's 1935 assumption of 3,470) · de Dear et al. 1997 (segment
convective/radiative coefficients) · Borbély 1982; Daan, Beersma & Borbély 1984,
*Am J Physiol* 246:R161–R178 (τ_w 18.2 h, τ_s 4.2 h) · Van Dongen et al., *Sleep*
2003;26(2):117–126 · Rechtschaffen et al., *Sleep* 1989 · Mifflin–St Jeor · ACSM
and NATA fluid-replacement position stands · Minnesota Starvation Experiment
(Keys, 1944–45).

**Resolved by measurement on this box** (previously flagged unverified): Crafter,
Craftax-Classic, Craftax-full, NLE, MiniHack, Kinetix, XLand-MiniGrid,
Humanoid-v5 and dm_control throughput on aarch64; NLE's ARM buildability; NLE's
licence (**NetHack GPL**, not permissive); Craftax's speedup as GPU-conditional;
Craftax's shipped achievement count (67, vs 65 in the paper).

**Still explicitly unverified**, and flagged wherever used: MJWarp's minimum
compute capability; Genesis's GPU compute-capability floor and whether
`TemperatureGrid` couples back into physics; Unity Linux ARM64 headless player;
Godot ARM64 / headless export and IPC cost; **Avalon's body representation**
(the one open question that could change §1.6's verdict) and its FPS; Craftium
aarch64 source build; AI2-THOR licence and ARM support; ThreeDWorld ARM support;
Kinetix `SYMBOLIC_ENTITY`/`PIXELS` at size `l` (compile exceeded 10 min);
Genie 3 programmatic API existence; Lucid v1's licence; HumanoidBench exact
obs/act dims; MJX-on-ARM throughput; JaxMARL and Melting Pot/DMLab2D absolute
sps (no published figure exists); Neural MMO 2.0's current need set; the
5/10/15% dehydration ladder; a citable lethal core-temperature band above 40 °C.
