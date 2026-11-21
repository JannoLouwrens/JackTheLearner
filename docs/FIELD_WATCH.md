# FIELD_WATCH.md — the scout's current-state report

> Rewritten each sweep (Mon, Thu), ≤ 12 KB; history in `FIELD_WATCH_LOG.md`.
> Nominates arms; adopts nothing, changes no spec, threshold or decision.

**Sweep:** 2026-11-19 (Thu): fronts 11-16 left unread (homeostatic RL, adults,
small ears, word learning), then papers 11-10 → 11-18, then the record's sources.

**Marks.** [V] verbatim · [V†] summariser's extraction (HTML past ~40k chars) ·
[c] source not opened · [R] read in this repo today.

## Coverage

| Front | Enumerated | Opened |
|---|---|---|
| 1 Cores | DreamerV3, TD-MPC2, model-based RL, active inference, replay ratio, world model + RL | 2611.24115, 2611.21465, 2611.01666, 2609.39151 |
| 2 Fusion | multimodal world model (newest Aug); tactile + RL (newest 11-05) | none cleared |
| 3 Memory | episodic + RL, plasticity loss (both newest 10-30); continual RL; replay + sleep | 2609.38598, 2611.21598, 2611.21879, 2611.22470 |
| 4 Curriculum | UED; intrinsic motivation; open-ended; novelty (newest 11-09); Craftax to Sept | 2610.04604, 2610.15328 |
| 5 Worlds | MuJoCo changelog; playground, Craftax, TD-MPC2 repos | changelog [V] |
| 6 Adults | VLM reward; LLM teacher; keyword spotting; release cards | 2611.18597, 2610.13447, 2611.19389, 2611.19844, ZONOS3-Mini, ARK-ASR cards |
| Senses, biology | homeostatic RL (18, all read); vocal learning; child-directed speech | 2608.27843, 2611.22499, 2611.21969 |

**Gaps (not "nothing found").** arXiv API: 2 of 12 queries, then 429; search
pages 33/33, ~15 rows each verbatim. Zero rows by my wording: MuJoCo humanoid
from scratch, small CPU ASR, small-data speech. **Unread** (`OR` floods):
autotelic, olfaction, vocal learning, child-directed speech. q-bio.NC not
enumerated. No PDF read; long papers' limitations mostly [V†].

---

## NOMINATIONS

### N1 — A non-commutative rotating register, against the seated ring (Working memory)

**Source.** *RL with Complex (valued) Memories*, arXiv:2609.38598 (09-29;
Bhethanabhotla, Gavves, Biedenkapp; EWRL 2026); code `github.com/Sathya98/qurl`
(JAX, POBAX) [V].

**Verified.** Three unitary cells replace a GRU in recurrent PPO, same
hidden size, 5 seeds [V]: diagonal phases × Householder × Fourier × permutation,
or layered 2×2 rotations (L = 2 only) — non-commuting; no write gate [V].
RockSample (11,11): "roughly 2×" the GRU and PPO-LD plateaus; Craftax
(no-inventory, 100M steps): two of the three "far exceed the two GRU
baselines" [V]. **T-Maze with a 75-step corridor — the one pure memory probe —
all cells "converge to similar returns"** as the GRU [V]; Battleship no gain
[V†]; the input-conditioned cell fails on Craftax, put down to "insufficient
tuning" [V]. No parameter counts; the abstract's "2–3×" matches no number in the text.

**Seat / arena.** Working memory, BY VERDICT `WM.02` (ring 0.902 vs GRU
0.579); LAB 11-11's open "non-commutative register", published with code. Arm:
the rotation cell at the ring's size in `WM.02`'s task, controls unchanged.

**Cost.** CPU minutes a seed (ring: 1,031 weights). Their GPU JAX runs
(5M–100M steps) and Craftax scores do not transfer; only the cell does.

**Might win:** the ring's rotations commute, so it can hold *how much* but not
*in what order*; a non-commuting product can, and order is what "heard, said,
did" needs. **Might lose:** (i) on the paper's own memory probe it ties the
GRU — the gains are on tasks where memory is one of several things; (ii) it is
ungated and LAB's no-write-gate ring died at 0.119; (iii) sizes are not
parameter-matched in the paper; (iv) workshop paper, untuned by the authors'
own word.

### N2 — If the forager's clone misses, the published cure is 2,304 weights of leaky linear memory (J15, row E2b)

**Source.** *Linear Recurrent Memory Suffices to Distil a World-Model Policy
for Robot Air Hockey*, arXiv:2609.39151v2 (11-15; Fan & Obst, UNSW); code
`github.com/unswei/airhockey-distillation` [V].

**Verified.** A DreamerV3 teacher (1M steps) cloned by plain regression into
students of 5,602–28,898 weights, 5 matched seeds, main result on a test split
"opened only after all models and analyses were frozen" [V]. With the puck
hidden for 400 ms: memoryless student **44.4% [33.5, 56.4]** saves; a ten-step
input stack 86.0% [79.0, 91.8]; a 64-channel memory whose only recurrence is
`z ← diag(tanh α)·z + B·x` (12,002 weights, 2,304 in the memory, timescales
initialised 40 ms–2 s) **98.3% [96.5, 99.6]**; GRU-64 (28,898) no better
(paired difference 0.5, CI [−1.2, 2.5] [V†]) [V]. "Nonlinear recurrence adds
no clear gain" [V]. One round of the teacher labelling the student's own
states lifted a pilot from 67.1% to 98.7% [V]. 23 µs a step on one Intel thread — not our hardware.

**Seat / arena.** J15 row E2b (no seat moves; `WM.02` for the cell). J15
discloses [R]: the teacher holds a remembered target on 0.551 of decisions, the
frozen student is memoryless (256-256-256-8), "a memoryless clone may miss for
the forager's memory, not the eye" — this paper's situation. **Declared
follow-up arm if E2b misses**: same data, bar, permuted twin, plus a diagonal
leaky memory, so a miss reads as memory or eye. Frozen E2b rule untouched.

**Cost.** CPU; memoryless clone 4.4 CPU-s per 1e5 samples (J15); recurrent unmeasured.

**Might win:** half the teacher's decisions depend on something the eye does
not show this instant; a 2 s leaky trace is the smallest thing that can carry
it. **Might lose:** (i) the forager's other 45% is a coin held 25 decisions
that no memory of the *input* recovers (PRIORS §17); (ii) their blackout is
0.4 s, his lives are 9,000 steps; (iii) one simulated set-piece, one teacher;
(iv) the student is cloned, never learns on — this says nothing about a
learner that must find the behaviour himself.

### N3 — The teacher-proposed curriculum arm, with its must-fail control (Curriculum, `CR.01`)

**Source.** *Anticipating the Consequences of Curriculum Decisions with LLMs*,
arXiv:2610.04604 (10-03; Pappalardo, Herr, Rocktäschel, UCL); code
`github.com/Octavio-Pappalardo/curriculum-foresight` [V].

**Verified.** The learner is from scratch (PPO, ~6.2M weights [V†]) on 256
textual goals in Craftax [V]. The LLM never picks a task during training: once,
offline, it writes a graph of "training on s helps t" and which goals are
feasible from nothing; task choice is then learning progress × (weight +
downstream potential from that graph) [V†]. Abstract: "the strongest gains
when optimizing for individual target tasks" [V]. Single target after ~1.5B
steps, 5 seeds: Furnace 79% vs 0% for learning progress alone, Dungeon 44% vs
2% [V†]. With uniform weights over all goals the two "end similar" [V†]. A
**shuffled-graph control (PERM) "still retains some benefit"** [V†].

**Seat / arena.** Curriculum, VACANT, `CR.01` (GOAL: "what the adults around
him say and propose"; 11-11 gave easy-only and reversed controls). This is the
unpuppeted adult arm — the adult writes the world's order once, his learning
progress still chooses — with its control: the permuted graph must lose.

**Cost.** A few hundred calls to a free open LLM (theirs: closed Gemini).
Their 1.5B–3B steps are ~10,000× a life; only the design transfers; we have a
handful of goals, not 256.

**Might win:** when one hard thing matters (fire, shelter), learning progress
alone never samples its prerequisites; a told dependency order does. **Might
lose:** (i) the permuted graph kept part of the gain — some of the effect is
reweighting, not knowledge; (ii) no gain when all goals count equally; (iii)
the effect numbers are [V†], one sigma shading, no intervals; (iv) with five
goals a graph may be nothing a person could not write on a card — in which
case the card is the arm.

---

## CHECKS ON THE RECORD — nothing overturned

- **MuJoCo:** 4.0.1 (11-11) is still the newest release [V]. `requirements.txt:23`
  is still `mujoco>=3.1.0`, unbounded [R]; `dm-control` 1.0.52 requires
  `mujoco>=4.0.1` [V]. The 11-16 hazard stands unchanged (RK4 on free joints
  changes results in 4.x; pin the Mac or re-certify deliberately). Also 4.0.0:
  tendon `sidesite` "can change the behavior of existing models"; flex elasticity fixed [V].
- `danijar/dreamerv3` head still `e01491f` (10-04) [V]. MLX still 0.32.4 [V];
  the "scan" line in 0.32.3 is unresolved a second sweep. SBX 0.28.0 [V].
- `mujoco_playground` v1.0.1 (11-09), notes not read; Craftax v1.7.0
  (10-27); TD-MPC2: no successor [V].
- **11-16's N2 (smell) is in the record** as PRIORS §14 [R]. 11-12's N1 (judge
  gate) reads "yes, queued" in `DECISIONS_RESOLVED.md` [R]. Still live and
  unentered: 11-16 N1 (pink / red selection noise, 2611.11868 — §5 names pink
  noise, the paper is not cited), 11-12 N2 (permanent/transient split,
  2611.07327), 11-11 N1–N3.

---

## WATCHLIST

| Item | What it is | Promote on |
|---|---|---|
| **Pixel-KWS** 2611.19389 | Open-vocabulary ear from scratch: 4-layer Conformer + text encoder, **0.85–0.86M weights**, LibriSpeech phrases only; hard / easy error 12.4 / 0.63% at that size, 9.6 / 0.4% at 11.4M [V]. Next rung after J14's 12 words: "is this the word I was told?" for any word. 4 GPUs; **no code**. | J14's ear row in band; code or a T4 timing. |
| **Oído** 2610.13447, code | 13M streaming English ASR on a $5 microcontroller, one core, real-time factor 0.54–0.72, 7.1 / 16.4% word error [V]. Built from NVIDIA's released weights [V] — an ADULT listener, never his ear. | A licence read at the source ([V†] GPLv3) and a timing on this box. |
| **ZONOS3-Mini** (Zyphra, 11-09) | TTS, Apache-2.0, 800M weights, 100M active per token, 43 languages, CPU runtime exists [V]. No CPU speed given. Adult voice. | A CPU real-time factor on ARM. |
| **ARK-ASR-0.6B** (Edge0) | ASR, Apache-2.0, 19 languages [V]; speeds on H200 only. | A CPU timing. |
| **Evolutionary Neuroplastic Agent** 2611.21598, code | Gradient-free archive of evolved 4+2-unit specialists; after a physics change retains 115.6% vs PPO 39.2% (20 runs) [V†]. Members are FROZEN (ineligible); evidence "does not isolate" the cause [V]. | A continual-learning order opening; a non-frozen variant. |
| **Episodic repulsion in planning** 2611.22470 | States visited this episode repel the planner; success 31.8 → 46.7% on one world model, 3 seeds [V]. Random cost of equal scale ties it on one task; settings chosen on the test tasks [V†]. | Brain A planning in his world. |
| **Does the model hear its own action?** 2611.01666 | A randomisation test for whether a world model's prediction moves with the action; the usual zeroed-input check gave false effects [V†]. Low power: TD-MPC2 1 of 6 tasks [V]. | A second reading. |
| **Skills return cheaply** 2611.24844, code | From-scratch language models: a withheld skill returns at 1.1–22% of the data [V†]. Bears on "a sense added later". | An RL replication. |
| Carried, not re-checked | The four 11-16 items; six 11-12 items. | As before. |

**Biology, one line.** 94 mother–child pairs at age 2: adult–child–adult
exchanges separate the low and high language groups (η² = 0.22), cognitive
scores do not [V]. Measured at one age — a correlation, not a cause. For the
talkative parent: answering him may matter more than talking near him.

---

## NO-ACTION

- **Cores:** nothing at our size — 2611.24115 is 4M atop 200M DreamerV3, A100, object detector; Crafter 15.8±2.0 vs 14.5±1.6 [V†]; 2611.21465 no closed-loop numbers.
- **Fusion:** no paper; ninth sweep without one.
- **Memory:** nothing since 10-30; 2611.21879 tabular theory; "Replay, Don't Resample" is LLM distillation.
- **Curiosity/UED:** nothing since 11-09; 2610.15328 +4.8 Craftax at 2B, both parts removed 45.1±2.8 vs 46.7 [V†], value "open".
- **Worlds:** no new survival or needs world.
- **Needs:** whole listing read; one new, 2608.27843, hand-built Bayesian controller — not a learner.
- **Voice/adults:** 2611.22499 hand-built, no sound, nothing learned; 2611.18597 frozen CLIP on H200, no adult-removed arm.

**Philosophy served:** 3 (challengers for a seat and a vacancy; J15's miss read
in advance), 4 (each arm with its must-fail control), 2 (ears sorted his/adults'), 5.
