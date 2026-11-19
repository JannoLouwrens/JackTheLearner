# FIELD_WATCH.md — the scout's current-state report

> Rewritten each sweep (Mon, Thu). A state, not a log — history is one line per
> item in `docs/FIELD_WATCH_LOG.md`. This page NOMINATES arms; it adopts
> nothing and changes no spec, threshold or decision.

**Sweep:** 2026-11-19 (Thu). Aimed first at the fronts 11-16 left unread
(homeostatic RL, adults as teachers and judges, small ears, word learning),
then new papers 11-10 → 11-18, then the record's own sources.

**Evidence marks.** **[V]** page text read verbatim by a fetch tool · **[V†]**
only a summariser's extraction of that page (long HTML is cut at ~40k
characters; the rest arrives summarised) · **[c]** claim, source not opened ·
**[R]** read in this repository today.

## Coverage

| Front | Enumerated | Opened |
|---|---|---|
| 1 Cores | search pages: DreamerV3, TD-MPC2, model-based RL, active inference, replay ratio; API: world model + RL | 2611.24115, 2611.21465, 2611.01666, 2609.39151 |
| 2 Fusion | multimodal world model (3 results, newest August); tactile + RL (newest 11-05) | none cleared |
| 3 Memory | episodic memory + RL (newest 10-30); loss of plasticity (newest 10-30); continual RL; replay + sleep (2 results) | 2609.38598, 2611.21598, 2611.21879, 2611.22470 |
| 4 Curriculum, curiosity | UED; intrinsic motivation; curriculum + open-ended; curiosity + novelty (newest 11-09); Craftax back to September | 2610.04604, 2610.15328 |
| 5 Worlds | Craftax; MuJoCo changelog; mujoco_playground, Craftax, TD-MPC2 repos | changelog [V] |
| 6 Adults | VLM reward + RL; LLM teacher + RL; keyword spotting; two release cards | 2611.18597, 2610.13447, 2611.19389, 2611.19844, ZONOS3-Mini, ARK-ASR cards |
| Senses, biology | homeostatic RL (18 results, read whole); vocal learning; child-directed speech | 2608.27843, 2611.22499, 2611.21969 |

**Gaps — not covered, not "nothing found".** The arXiv API answered 2 of 12
queries, then HTTP 429; the search pages answered 33 of 33 but only the first
~15 rows of each page are verbatim. Three queries returned zero rows through
my own wording (MuJoCo humanoid from scratch; small CPU ASR; small-data speech
representation). Four `OR` queries (autotelic, olfaction, vocal learning,
child-directed speech) returned unrelated floods — those four topics are
**unread**, not empty. q-bio.NC was not enumerated at all. No PDF was read;
limitations sections of long papers are mostly [V†].

---

## NOMINATIONS

### N1 — A non-commutative rotating register, against the seated ring (Working memory)

**Source.** *Reinforcement Learning with Complex (valued) Memories* —
arXiv:2609.38598, cs.LG, 09-29; Bhethanabhotla, Gavves, Biedenkapp; "Accepted
at the 19th European Workshop on Reinforcement Learning 2026"; code
`github.com/Sathya98/qurl` (JAX, on the POBAX benchmark) [V].

**Verified.** Three unitary recurrent cells dropped into recurrent PPO in
place of a GRU, same hidden size, 5 seeds [V]. The plain cell's transition is
a product of diagonal phases, Householder reflections, a Fourier transform and
a fixed permutation; the tunable one is layered 2×2 rotations (only L = 2 run)
[V] — products that do not commute. No write gate in the equations [V].
RockSample (11,11): "roughly 2×" the GRU and PPO-LD plateaus; Craftax
(no-inventory, 100M steps): two of the three "far exceed the two GRU
baselines" [V]. **T-Maze with a 75-step corridor — the one pure memory probe —
all cells "converge to similar returns"** as the GRU [V]; Battleship no gain
[V†]; the input-conditioned cell fails on Craftax, put down to "insufficient
tuning" [V]. Parameter counts are not given; the abstract's "2–3×" has no
matching number in the text I read.

**Seat / arena.** Working memory — BY VERDICT `WM.02` (ring 0.902 vs GRU
0.579). LAB 11-11 lists as open "a non-commutative register". This is that
register, published, with code. Arm: the layered-rotation cell at the ring's
size inside `WM.02`'s own task, shuffled and no-rotation controls unchanged.

**Cost.** CPU, minutes per seed at `WM.02`'s size (the ring is 1,031
parameters). The paper's runs are GPU-batched JAX at 5M–100M steps — its
speeds and its Craftax scores do not transfer; only the cell does.

**Might win:** the ring's rotations commute, so it can hold *how much* but not
*in what order*; a non-commuting product can, and order is what "heard, said,
did" needs. **Might lose:** (i) on the paper's own memory probe it ties the
GRU — the gains are on tasks where memory is one of several things; (ii) it is
ungated and LAB's no-write-gate ring died at 0.119; (iii) sizes are not
parameter-matched in the paper; (iv) workshop paper, untuned by the authors'
own word.

### N2 — If the forager's clone misses, the published cure is 2,304 weights of leaky linear memory (J15, row E2b)

**Source.** *Linear Recurrent Memory Suffices to Distil a World-Model Policy
for Robot Air Hockey* — arXiv:2609.39151v2, cs.RO, 09-30 / 11-15; Fan & Obst
(UNSW); code `github.com/unswei/airhockey-distillation` [V].

**Verified.** A DreamerV3 teacher (1M steps) cloned by plain regression into
students of 5,602–28,898 weights, 5 matched seeds, main result on a test split
"opened only after all models and analyses were frozen" [V]. With the puck
hidden for 400 ms: memoryless student **44.4% [33.5, 56.4]** saves; a ten-step
input stack 86.0% [79.0, 91.8]; a 64-channel memory whose only recurrence is
`z ← diag(tanh α)·z + B·x` (12,002 weights, 2,304 in the memory, timescales
initialised 40 ms–2 s) **98.3% [96.5, 99.6]**; GRU-64 (28,898) no better
(paired difference 0.5, CI [−1.2, 2.5] [V†]) [V]. "Nonlinear recurrence adds
no clear gain" [V]. One round of the teacher labelling the student's own
states lifted a pilot from 67.1% to 98.7% [V]. Timed on one desktop Intel
thread (23 µs a step) — not our hardware, and not the claim nominated.

**Seat / arena.** Row E2b of order J15 (no seat moves; Working memory's
`WM.02` for the cell itself). J15's own disclosure [R]: the teacher holds a
remembered target on 0.551 of decisions, the frozen student is memoryless
(256-256-256-8), and "a memoryless clone may miss for the forager's memory,
not the eye". This paper is that exact situation with numbers. Nominated as
the **declared follow-up arm if E2b misses** — same data, same bar, same
permuted twin, plus a diagonal leaky memory — so a miss is read as memory or
eye, not left ambiguous. The frozen E2b rule is untouched.

**Cost.** CPU; J15 measured 4.4 CPU-s per 1e5 samples for the memoryless
clone — a recurrent student costs more per sample (unmeasured here).

**Might win:** half the teacher's decisions depend on something the eye does
not show this instant; a 2 s leaky trace is the smallest thing that can carry
it. **Might lose:** (i) the forager's other 45% is a coin held 25 decisions
that no memory of the *input* recovers (PRIORS §17); (ii) their blackout is
0.4 s, his lives are 9,000 steps; (iii) one simulated set-piece, one teacher;
(iv) the student is cloned, never learns on — this says nothing about a
learner that must find the behaviour himself.

### N3 — The teacher-proposed curriculum arm, with its must-fail control (Curriculum, `CR.01`)

**Source.** *Anticipating the Consequences of Curriculum Decisions with Large
Language Models* — arXiv:2610.04604, cs.LG, 10-03; Pappalardo, Herr,
Rocktäschel (UCL); code `github.com/Octavio-Pappalardo/curriculum-foresight`
[V].

**Verified.** The learner is from scratch (PPO, ~6.2M weights [V†]) on 256
textual goals in Craftax [V]. The LLM never picks a task during training: once,
offline, it writes a graph of "training on s helps t" and which goals are
feasible from nothing; task choice is then learning progress × (weight +
downstream potential from that graph) [V†]. Abstract: "the strongest gains
when optimizing for individual target tasks" [V]. Single target after ~1.5B
steps, 5 seeds: Furnace 79% vs 0% for learning progress alone, Dungeon 44% vs
2% [V†]. With uniform weights over all goals the two "end similar" [V†]. A
**shuffled-graph control (PERM) "still retains some benefit"** [V†].

**Seat / arena.** Curriculum — VACANT, `CR.01`. GOAL names "what the adults
around him say and propose" as part of what is raced; 11-11 nominated the
easy-only and reversed-order controls. This adds the form of the adult arm
that keeps him unpuppeted — the adult writes an order of the world once, his
own learning progress still chooses — and its control: the same graph with
labels permuted must lose.

**Cost.** The graph is a few hundred calls to a free open LLM (theirs was a
closed Gemini — not free, not ours). Their 1.5B–3B steps are ~10,000× a
300k-decision life; only the design transfers. Our goal set is a handful of
needs and rules, not 256.

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
  changes results in 4.x; pin the Mac or re-certify deliberately). Two more
  4.0.0 items read this time: tendon `sidesite` semantics "can change the
  behavior of existing models"; flex elasticity constants corrected [V].
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
| **Pixel-KWS** 2611.19389 | An open-vocabulary ear from scratch: 4-layer Conformer + small text encoder, **0.85–0.86M weights**, trained on LibriSpeech-derived phrases only; hard / easy error 12.4 / 0.63% at that size, 9.6 / 0.4% at 11.4M [V]. The next rung after J14's 12-word row: "is this the word I was told?" for any word. 4 GPUs, model unnamed; **no code**. | J14's ear row in band; code or a T4 timing. |
| **Oído** 2610.13447, code | 13M streaming English ASR on a $5 microcontroller, one core, real-time factor 0.54–0.72, 7.1 / 16.4% word error [V]. Built from NVIDIA's released weights [V] — an ADULT listener, never his ear. | A licence read at the source ([V†] GPLv3) and a timing on this box. |
| **ZONOS3-Mini** (Zyphra, 11-09) | TTS, Apache-2.0, 800M weights, 100M active per token, 43 languages, CPU runtime exists [V]. Only H100 and phone first-audio figures — no CPU speed. Adult voice. | A CPU real-time factor on ARM. |
| **ARK-ASR-0.6B** (Edge0) | ASR, Apache-2.0, 19 languages [V]; speeds on H200 only. | A CPU timing. |
| **Evolutionary Neuroplastic Agent** 2611.21598, code | Gradient-free; evolved nets of 4+2 hidden units kept as an archive of specialists; after a physics change retains 115.6% vs PPO 39.2% (20 runs) [V†]. But the archive is FROZEN members (scored-and-ineligible), and the authors say the evidence "does not isolate" the cause [V]. | A continual-learning order opening; a non-frozen variant. |
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

- **1 Cores.** Nothing at our size. Relational world model (2611.24115): 4M
  on top of a 200M DreamerV3, A100, needs an object detector; Crafter 15.8 ±
  2.0 vs 14.5 ± 1.6 — overlapping [V†]. Jacobian warm start (2611.21465): no
  closed-loop numbers. No new small-Dreamer score, no Apple-silicon Dreamer.
- **2 Fusion.** No paper in the window; ninth sweep without one.
- **3 Memory.** No episodic-retrieval or plasticity paper since 10-30.
  2611.21879 is tabular theory. "Replay, Don't Resample" is LLM distillation.
- **4 Curiosity / UED.** Nothing since 11-09. Anchor-and-Harvest (2610.15328):
  +4.8 on Craftax at 2B steps, but removing both of its named parts leaves
  45.1 ± 2.8 vs 46.7 [V†] and the authors call its value over simpler
  curricula "open".
- **5 Worlds.** No new survival or needs world.
- **Needs.** The whole homeostatic-RL listing read: one new item since
  August, 2608.27843 — a hand-built Bayesian controller with a frozen text
  encoder in a scripted exchange. Not a learner.
- **Voice.** 2611.22499 is hand-built spiking articulation, no sound, nothing
  learned. Adults as reward: 2611.18597 uses a frozen CLIP on H200 nodes and
  no arm with the adult removed.

**Philosophy served:** 3 (a seated memory and a vacant curriculum each get a
published challenger; a miss in J15 gets its reading before it happens), 4
(every arm above arrives with the control that must fail), 2 (ears and voices
sorted into his and the adults'), 5 (research before the race).
