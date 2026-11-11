# FIELD_WATCH.md — the scout's current-state report

> Rewritten each sweep (Mon, Thu). A state, not a log — history is one line per
> item in `docs/FIELD_WATCH_LOG.md`. This page NOMINATES arms; it adopts
> nothing and changes no spec, threshold or decision.

**Sweep:** 2026-11-11 · **new-paper window:** 2026-10-24 → 2026-11-10 (the
six months to 10-26 were covered by the previous report). The log holds no
line between 2026-10-26 and today: 16 days unreported, cause not knowable
without a shell.

**Evidence marks.** **[V]** short passages quoted verbatim from the abstract or
HTML full text, fetched this sweep · **[V†]** fetched, but that part reached me
as a summariser's extraction · **[c]** authors' claim, not checked · **[R]**
read in this repository today. **Lower evidence class than 10-26:** the arXiv
API was readable only through a fetch tool that caps verbatim quotes, so no
abstract was read whole. Nothing below rests on a paper I did not open by id.

## Coverage

| Front | Searched (arXiv, newest first) | Opened by id |
|---|---|---|
| 1 Cores | world model × model-based/latent/JEPA; sample-efficiency / active inference × RL | 2610.40637 |
| 2 Fusion | multimodal × imbalance/collapse/dominance/sensor fusion | 2611.02651 |
| 3 Memory | episodic memory / consolidation / replay × agent/RL | none |
| 4 Curriculum, curiosity | UED / learning progress / autotelic / open-ended; intrinsic motivation × RL | 2610.40364, 2611.08787, 2611.01183, 2611.13113 |
| 5 Worlds | homeostatic/foraging/survival/MuJoCo/Craftax; web: MuJoCo changelog | none |
| 6 Adults | LLM/VLM × teacher/reward/judge × RL; small speech models; web: new small ASR | 2611.14421, 2611.03622 |
| Biology | `q-bio.NC` × RL/infant/development/replay | 2611.04565 |

**Gaps.** No PDF read; no conference enumeration; fronts 3 and 5 enumerated by
title only; TTS, VLM-as-eyes, smell/taste/voice not searched. The 10-26 queued
question on the predictor-width nomination (does width scale with what the
predictor reads, or what it predicts?) is **still open** — the paper's HTML
is 404 and the abstract does not say [V].

---

## NOMINATIONS

### N1 — Two controls for the curriculum cook-off (order J5): an EASY-ONLY arm and a REVERSED-ORDER arm

**Sources.** (a) *Is It the Curriculum or Just the Easy Start? Benchmarking UED
on Quadrupeds* — arXiv:2610.40364, cs.LG, 10-29, Gilewicz, Galus, Wojdas; code
MIT `Solvro/ml-wojtek-ued`. (b) *Curriculum unlearning in humans and neural
networks* — arXiv:2611.04565, cs.LG, 11-03, Rubruck, Zhang, Saxe, Summerfield.

**Verified.** (a) 14 curricula, one protocol, PPO, five MuJoCo-MJX quadrupeds,
"five seeds per method and robot", 350 runs [V]. `DR-easy` (every level shrunk
to difficulty 0.1) is a named control to "isolate 'train on easy levels' from
curation" [V]. "On three robots the leader on in-distribution random levels is
not the leader on held-out terrain" [V]. Held-out success, Go1: DR 30.2,
DR-easy 38.6, Terrain-CL 45.8; Spot: DR 9.9, DR-easy 8.0, PAIRED+BiBC 23.2
[V; significance marks lost in extraction]. (b) Humans (n = 207) and a
32-hidden-unit network: simple-to-complex beats a matched-exposure interleaved
control (humans log-odds 0.78 [0.20, 1.36]; networks 0.024 [0.007, 0.040]);
complex-to-simple is worse on the hard items [V].

**Seat / arena.** Curriculum — no seat yet; J5 creates it VACANT and registers
one cook-off (none / hand-staged / learning-progress / teacher-proposed,
control = shuffled stages). `CU.1`–`CU.7` have never run [R]. Proposed
additions to that one spec:
- **easy-only arm** — the easiest stage for the whole budget. If hand-staged
  cannot beat it, staging bought only a gentle start.
- **reversed-order arm** — hardest first; a must-fail with a measured
  direction, stronger than a shuffle.
- **score on a held-out world** as well as the training worlds.

**Cost.** Two more arms at whatever J5's pilot measures per arm; a stage
schedule, no new model code. CPU.

**Might win:** 14 published curricula were raced and the cheapest control led
often enough to become the title; without it our cook-off cannot lose that way.
**Might lose:** (i) most real curricula DID beat DR-easy on held-out terrain,
so the control may just fail as designed; (ii) 300M–1B steps on an H100 per
run — hardware unlike ours, only the design transfers; (iii) Jack's world has
no single difficulty scalar, so "easiest stage" is the builder's definition
and could be a convenient one; (iv) (b) is an 8-stimulus task with a
~161-parameter network — an analogy to a life, not a measurement of one.

### N2 — A must-fail control for the needs reward: the cheap relief action

**Source.** *Satisfied, Not Solving: Wireheading of Bounded Homeostatic Rewards
in RL* — arXiv:2611.01183, cs.LG, 10-31, Chase J. Metoyer (single author);
code + data `chasemetoyer/satisfied-not-solving`.

**Verified.** Reward is drive reduction, `r = D(h_t) − D(h_{t+1})`,
`D(h) = Σ h_k²`, total return capped at 0.75 [V]. Given "a cheap action that
resets its needs", 2 of 80 PPO seeds still solve DoorKey-8x8 against 71 of 80
without it; KeyCorridor 0 of 20 against 20 of 20; "a per-press cost of 0.1
restored the task in every seed" [V]. 23 hypotheses pre-registered, "9 … were
not supported" [V]. One laptop CPU, no GPU, ~16 h summed [V†].

**Seat / arena.** Needs/reward form — HELD BY ANALYSIS, arena `NE.02`.
`NEEDS_AND_DEATH.md` §0.2 is headed "drive-reduction reward is NOT farmable"
[R]. The paper does not break that theorem (the reward stays bounded); it
measures what the theorem leaves out — a bounded reward is *satisfied* by the
cheapest relief on offer and nothing else gets learned. The arm: one world
variant in `NE.02` with a costless relief action. The reward form holds its
seat only if the learner there is measurably worse at the real skill; the
world passes only if no such action exists in it. `W.2` (needs are a conserved
ledger) has been FAIL 73 days [R] — a ledger leak is a cheap relief action.

**Cost.** CPU; one extra world variant × seeds in an existing rig.

**Might win:** it turns an analysis-held seat into a run whose control must
fail, and aims at a world defect already red.
**Might lose:** (i) their needs sit beside an external task; Jack has none —
eating *is* cheap relief and is correct, so the control must separate relief
by consequence from relief by leak, a design unit; (ii) single author,
MiniGrid, PPO; (iii) their fix is a price on the action, which our world
already charges in time and energy; (iv) nine of their own predictions failed.

### N3 — Probe-guided gradient balancing as a balancing arm (`T4.06` successor / `UB.10`) — the weakest of the three

**Source.** arXiv:2611.02651, cs.CV, 11-02, Vo, Fayek, Nguyen; code
`VuVoCodes/PGGB`. **Verified.** A linear probe on each modality's
stop-gradient features sets a gradient scale `min(1 + α·w_m, s_max)` for the
weakest modality; loss and architecture unchanged [V]. CREMA-D 71.45 ± 1.71,
+2.31 pp over OGM-GE [V]; eight benchmarks, five seeds, ~3K params per probe,
+1.6% wall-clock [V†].
**Seat / arena.** Sensory fusion (UNDECIDED, `UB.10`). `T4.06` certified
`loss_reweight` on thin margins and refuted `grad_norm` [R]; this is a third
family — scale by measured readability. **Cost.** Dozens of lines, CPU-minutes
on the `T4.06` fixture.
**Might win:** `T4.06` read three senses as unreadable from the fused latent
in all 12 runs, and this is the first method steered by exactly that reading.
**Might lose:** the authors report gains "absorbed under Adam" and that it
"would amplify a noise-only modality" [V†]; no touch, proprioception or action
in any benchmark; and with three senses at the floor it pushes all three to
`s_max` — a fixed reweighting, roughly the arm that already won.

---

## WATCHLIST

| Item | What it is | Promote on |
|---|---|---|
| **world2vec** 2610.40637, cs.LG | Reward-free, decoder-free latent dynamics, "263k parameters", minutes on one A100 [V] — under a third of our seated core. | Code (project page only) and a non-pixel result: 64×64 pixels, three easy DMC tasks, worked "in 2 of 3", seed count unstated [V]. |
| **Training Stability under Adaptive Curricula** 2611.08787, cs.LG | Under UED the *student* loses trainability; critic weight clipping + categorical value loss restores it; five seeds, Kinetix [V]. Bears on "never stop learning". | Effect sizes in text, code, a run below 2 billion steps. Meanwhile: a reason for J5 to log a plasticity readout per arm. |
| **Building World Models from Intrinsic Motivation** 2611.13113, cs.AI | 24 intrinsic objectives in one DreamerV3 frame, 25 proprioceptive DMC tasks. Plain disagreement leads (+19.8%); CIG — this desk's 08-12 nomination — trails (+10.7%) [V†]. | Seeds, hardware, code (none stated). Today: support for the incumbent `disagree`, evidence against paying for CIG. 12M–200M models. |
| **Earlybird** 2611.14421, cs.SD | 29M English streaming ASR, MIT, distilled from Moonshine Tiny; WER 11.4 vs Whisper tiny.en 12.0; 39 ms per 80 ms step on one Xeon thread [V]. An ADULT listener only — never his ear. | A timing on this box's ARM cores (x86 speed figures have failed to transfer three times). `HEARING_BAKEOFF` holds Moonshine Small at 7.84 [R]. |

**Carried live, unrun as of 10-26, not re-checked today:** predictor width
(2610.32562), probe diagnostics (2610.22655), two routes to need prioritisation
(2610.18617), per-sense curiosity (2609.40134), matched continued-training
control (2610.19375), ALARM pre-gate (2610.02159), known-generator calibration
(2609.36097), critical-`d` probe (2508.21038), interoceptive precision
(2608.04232), entity collision (2605.29630), `UB.11` certificate (2607.27017).

---

## NO-ACTION

- **1 Cores.** Nothing. Koopman variants, DreamerV3 add-ons, VLA-scale work;
  world2vec is the only small-end entry and it is pixels-only.
- **3 Memory.** Nothing. New agent-memory papers have an LLM rewrite the store
  (inadmissible). Not opened: HSGNav (2611.10547, invalidates stored records
  the world contradicts — the staleness gap raised 10-26) and leverage-score
  replay (2611.05929). Open next sweep or drop.
- **5 Worlds.** Nothing moves the fidelity ladder. No MuJoCo release found
  past 3.9.0 [web search only]; new work is GPU-parallel, the axis already
  ruled out for one serial life on shared cores.
- **6 Adults.** No nomination. The strongest VLM-as-judge result (D&A,
  2611.03622) runs on paid closed APIs at $0.015–0.035 per trajectory and
  steers pretrained policies [V†] — not free. No from-scratch small ear
  appeared; the "learned ear" order has no new challenger.
- **Biology.** One item, used in N1. Not opened: hawkmoth multimodal active
  sensing (2611.11011).

**Philosophy served:** 3 (a seat held by analysis and a seat not yet created
get contests that can be lost), 4 (controls that must fail), 5 (the curriculum
ranks with the creature).
