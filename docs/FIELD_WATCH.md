# FIELD_WATCH.md — the scout's current-state report

> Rewritten each sweep (Mon, Thu). A state, not a log — history is one line per
> item in `docs/FIELD_WATCH_LOG.md`. This page NOMINATES arms; it adopts
> nothing and changes no spec, threshold or decision.

**Sweep:** 2026-11-16 (Mon). First sweep under `docs/PLAN.md` (11-15, "research
first"), so it was aimed at what `PUBLISHED_PRIORS.md` marks as open or
unverified, plus new papers 11-09 → 11-13 (nothing newer was listed).

**Evidence marks.** **[V]** read in the page text by a fetch tool (quotes capped
at 125 chars) · **[V†]** only a summariser's extraction of that page · **[c]**
claim, source not opened · **[R]** read in this repository today.

## Coverage

| Front | Enumerated (arXiv API, newest first) | Opened |
|---|---|---|
| 1 Cores | world model + RL; Dreamer / TD-MPC2 (titles back to July); model-based RL; active inference; SAC / replay ratio | 2611.11868, 2611.10316, 2607.19749, 2611.14955 |
| 2 Fusion | multimodal + RL; modality imbalance; tactile + RL | none cleared |
| 3 Memory | episodic memory; plasticity / continual RL; replay / consolidation | 2607.19749 |
| 4 Curriculum, curiosity | curriculum + RL; UED / open-ended; intrinsic motivation; sparse-reward exploration | 2611.11868 |
| 5 Worlds | Craftax / survival; MuJoCo; MuJoCo changelog; PyPI `mujoco`, `mujoco-core`, `dm-control` | changelog [V] |
| 6 Adults | keyword spotting; small ASR; TTS (search page) | abstracts only |
| Senses / biology | olfaction, vocal learning (search pages) | 2611.05738 |
| The record's own sources | `danijar/dreamerv3`, sheeprl, r2dreamer, MLX, SBX, `dreamer-mlx` | repos / PyPI [V] |

**Gaps — not covered, not "nothing found".** The arXiv API refused (HTTP 429)
after ~20 queries, and four fallback search pages returned zero results because
of my query syntax. So this sweep did NOT read: LLM/VLM as teacher or judge,
homeostatic / needs RL, q-bio.NC, from-scratch small ears, word learning in RL
agents. No conference enumeration. No PDF read. The six-month window was
re-read for Dreamer only, and there by title.

---

## HAZARD (front 5) — MuJoCo 4.x is, by its own changelog, a different world for him

Carried from 11-12 and now sharper. `requirements.txt:23` still reads
`mujoco>=3.1.0` with no upper bound [R].

- **The integrator he lives under changed.** 4.0.0: RK4 is now fourth-order for
  free and ball joint orientations — "This changes the simulation of models
  with ball or free joints that use `RK4`" [V]. `playground.py:527` sets
  `integrator="RK4"`, his root joint is `type="free"` (`:280`) and loose objects
  are free joints (`:417`, `:447`) [R]. A PASS carries the world it was earned
  in (LESSONS 4): every World certificate was earned on 3.x.
- **Other result-changing items in 4.0.0 [V]:** saturated `actuatorfrcrange`
  actuators no longer contribute implicit derivatives (changes `implicit` /
  `implicitfast`, the integrator PRIORS §10 prices as the speed route; no
  `actuatorfrcrange` in `playground.py` or `w0.py` [R]); contact pairs with
  undefined `solref` are mixed differently; convex-collision fixes "can
  slightly change contacts". 4.0.1 fixes a wrong Newton Hessian factor that
  4.0.0 introduced.
- **Packaging [V]:** `mujoco` 4.x is a pure-Python shell; the engine moved to
  `mujoco-core` (wheels for Linux aarch64 and macOS arm64, Python 3.11–3.15;
  3.10 dropped). `dm-control` 1.0.52 (11-11) requires `mujoco>=4.0.1`, so an
  unpinned `dm-control` install pulls 4.x. `mujoco` 3.17.0 remains on PyPI with
  aarch64 and macOS-arm64 wheels for Python 3.10–3.15 — PRIORS §5's pin
  (`3.17.0` + `dm-control==1.0.50`) installs on this box and on the Mac.

**For the Director.** The Mac (late December) is a fresh environment and PLAN's
ready-list already says "re-certified, running on macOS". Either pin it to the
version the certificates were earned on, or make 4.x a deliberate
re-certification — never a silent resolve. Not an arm. I cannot see the
installed version (no shell).

---

## NOMINATIONS

### N1 — Coloured action noise, applied only when choosing the action (brain B's exploration test)

**Source.** *Fade to… Red? Scheduling Coloured Noise Exploration…* —
arXiv:2611.11868v2, cs.RO, 11-09, López & Cruz, accepted ACRA 2026 [V]. No code
link found.

**Verified.** SAC + HER in Stable-Baselines3 on four sparse-reward tasks
(Fetch Reach / Push / PickAndPlace, ShadowHand rotate), 50 noise conditions,
"trained with 30 random seeds" each [V]. Learning efficiency as z vs white
noise, 95% paired-bootstrap CIs: **gSDE −0.75 [−0.88, −0.63]**, pink +0.35
[+0.18, +0.52], red +0.26 [+0.11, +0.42]; end performance: gSDE −0.52, pink
0.00, red −0.03 [V]. Movement jerk vs white: pink 0.53, red 0.32 [V]. The
colour touches action selection only — "gradient updates use the standard
i.i.d. Gaussian reparameterisation" — and the noise's lowest frequency is set
by episode length ("f_min = 1/T_ep") [V].

**Seat / arena.** Learning core (arena LC.*), as brain B's exploration setting:
PRIORS §5 lists "which exploration finds food first" in the last 10% and offers
the Zoo gSDE recipe or pink noise. This is 30-seed evidence on that fork: real
SB3 gSDE trailed plain white noise on all four sparse tasks pooled, and it
shows the clean form of pink noise that avoids the trap §5 names (the
pink-noise package also recolours the noise inside the update). It is also the
new evidence LAB 11-12 "Exploration in bouts" lacked for its unrun stage B
(coloured noise inside the learner).

**Cost.** Zero parameters, CPU. One or two extra arms (pink, red; sequence as
long as a life) in the exploration test B already owes: 5 seeds each at B's
pilot budget.

**Might win:** in his world, pink or red selection noise reaches the first meal
in fewer decisions than B's current held-noise default (SBX `use_sde`), with
red adding smoother movement for the human-like body later. **Might lose:**
(i) white never fails outright here and end performance is equal — the gain is
speed only; (ii) HER relabelling does much of the sparse-reward work and Jack
has no goal to relabel; (iii) on PickAndPlace 15 of 48 coloured schedules
learned slower than white [V]; (iv) Fetch-Reach drives most of the pooled gain
[V†]; (v) our own noise-only probe found pink roams 1.7× wider and feeds no
more often — direction, not range.

### N2 — Smell has a published record now: a prior for the Smell seat, and two world-side arms

**Source.** *OdorSpaces: Visual-Olfactory Embodied Navigation in 3D
Environments* — arXiv:2611.05738, cs.RO, 11-03, Lee et al. (Maryland),
"Accepted at … CoRL 2026" [V]. Project page listed; no code link found.

**Verified.** Gas fields from a fire-dynamics simulator, queried inside
Habitat scenes; the agent must find an invisible propane leak or an early
fire. The nose samples gas density at the body centre "plus four directional
offsets" at several heights [V]. Policy: PPO, separate encoders for vision,
smell and pose history into "a recurrent memory backbone" [V]. Start distance
is a curriculum, "from 1.5 m to 9.0 m over 30M training steps" [V]. Propane,
SPL % (two scene sets): temporal-gradient search 4.6 / 1.8, greedy chemotaxis
17.3 / 27.1, moth-like casting 36.1 / 33.6, learned 65.3 / 63.8 [V]. On fire
plumes every hand-written searcher scores 0.00 and the learned policy 24.7
[V]. One sensing height instead of several: fire SPL 0.00 [V†]. Recurrence
helps propane (54.0 → 65.3) and hurts fire (28.7 → 24.7) [V†].

**Seat / arena.** Smell — VACANT, arena `SM.02`. PLAN has "Other senses" at
step 0, and LAB killed smell three times ("a sense that works goes unused by
the reactive learner") at 13k–52k decisions. The one published from-scratch
smell learner used a start-distance curriculum, a nose with spatial spread
and memory, and **30M steps** — about 600× our largest smell run. Under
LESSONS 17 those kills are pilots below the published budget, not verdicts on
the sense. Nomination: (a) enter this as the Smell row of `PUBLISHED_PRIORS.md`;
(b) two arms when smell is next tested — a multi-point nose (centre + offsets)
against today's single reading, and start distance as a dial (it is already
J2's dial for food).

**Cost.** None to read. Their simulator is not ours (offline fluid dynamics,
days per scene on 8–16 cores [V†]); our smell is an analytic field and stays
CPU-cheap. A multi-point nose is four more field reads per decision.

**Might win:** smell that carries its own spatial gradient lets a memoryless
learner climb, and our `steer` result (14/40 fed vs babble 3/40 with the want
given) says a measured rise is enough. **Might lose:** (i) one training seed,
three test seeds [V]; (ii) I found no no-smell control, so "the policy uses
smell" is not shown in what I read; (iii) their target is invisible, ours is
food his rays may also see; (iv) 30M PPO steps is far outside our budget and
nothing shows the result holds at 1M; (v) idealised sensor, no wind, no noise
— the authors' own limits.

---

## CHECKS ON THE RECORD (`PUBLISHED_PRIORS.md`) — nothing overturned

- `danijar/dreamerv3` head is still `e01491f` (Oct 4, 2026) [V].
- sheeprl: stable 0.8.5 (10-27), `1.0.0rc7` the same day; the README carries
  no DreamerV3.5 score — only a single-run training-time table "on 4 CPUs",
  task unnamed [V]. §2's "no published scores" stands.
- MLX: 0.32.4 is still the latest [V]. **One line to check before the port
  spike:** the 0.32.3 notes say "Support shapeless compilation of scan
  operations" [V], while §11 lists scan as missing. I could not tell whether
  that is a loop primitive or cumulative-sum-style scans.
- `mathDR/dreamer-mlx`: 6 commits, one-line README, no result [V]. SBX: PyPI
  0.28.0 [V].
- **A first, partial T4 number for a Dreamer-shaped model** (§2 says speed is
  measured nowhere): arXiv:2611.10316 profiles an RSSM (512-unit GRU, 32×32
  latents, PyTorch) on a T4 — 0.387 J per batch-32, 20-step imagined rollout at
  68.19 W, "about 176 calls per second" [V]. Derived from power, inference
  only, no gradients, no encoder: it bounds nothing about training. E3 remains
  ours to measure.

---

## WATCHLIST

| Item | What it is | Promote on |
|---|---|---|
| **Dream rehearsal** 2607.19749, code | Dreamer-style agent on MiniGrid task chains: the world model keeps old tasks (reward retention 0.99–1.06) while the actor loses them; cloning the best *imagined* trajectories passes 3/3 four-task chains vs plain replay 0/3 [V]. Bears on Consolidation (`NE.05`) and "still learning without losing what he knew". | Brain A living in his world; more than n = 3, a single author and a 17M model. |
| **Posterior-prior gap** 2611.14955 | A probe for "obstacle ahead" reads AUC 0.85 on filtered states and 0.71 after 15 imagined steps; ensemble disagreement does not flag the loss (0.52) [V]. A cheap diagnostic for A: does "food ahead" survive 15 steps of imagination? | More than one seed; a probe retrained on imagined states (the authors' own confound). |
| **Antalia-2 Mini** 2611.13811 | 7.6M-parameter TTS, weights Apache-2.0, "41 times real time on an Apple M5 laptop CPU (2 threads)" [V]. Turkish only. An ADULT voice, never his. | An English model of this class, timed on our hardware. |
| **Marimba** 2611.18330, code + weights | Yes/no questions about speech (term present, turn ended, speaker changed) read from a frozen 0.6B ASR; Speech Commands 4-way 0.93 [V]. A listener-judge candidate. | A licence and a CPU or T4 timing. |
| Carried, not re-checked | The six 11-12 watchlist items (Phonon TTS, hawkmoth, leverage-score replay, failure-driven restarts, child-scale LMs, world2vec). | As 11-12. |

**Carried live, unrun:** 11-12's N1 (matched-counterfactual gate for any adult
that scores him, 2611.13596) and N2 (permanent/transient value split,
2611.07327); 11-11's N1–N3. Neither 11-12 id appears in `PUBLISHED_PRIORS.md` [R].

---

## NO-ACTION

- **1 Cores.** Nothing at the small end. In window: a cyber-defence Dreamer
  (3 days on one GPU), a lander, quantised TD-MPC2 at 317M, Koopman MBRL. No
  small-Dreamer score, no Apple-silicon Dreamer, no new port.
- **2 Fusion.** Nothing new; IPR (2611.13782) was dropped 11-12. No tactile +
  RL paper in the window.
- **3 Memory.** Retrieval: LLM-agent stores only. Plasticity / continual RL:
  no paper in the window.
- **4 Curriculum, curiosity.** No UED or open-ended paper in the window; the
  curriculum hits were network planning and proof repair.
- **5 Worlds.** No new survival or needs world.
- **6 Adults.** Thin and partly unread (see Gaps).
- **Voice.** No vocal-learning paper since 11-01.

**Philosophy served:** 3 (a dead sense and an assumed exploration recipe each
meet a published record), 4 (the engine version is part of every World
certificate), 5 (research before the next race; the record's own sources
re-checked), 2 (adult voices and listeners kept outside him).
