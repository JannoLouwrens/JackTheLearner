# FIELD_WATCH.md — the scout's current-state report

> Rewritten each sweep (Mon, Thu), ≤ 12 KB; history in `FIELD_WATCH_LOG.md`.
> Nominates arms; adopts nothing, changes no spec, threshold or decision.

**Sweep:** 2026-11-26 (Thu). New papers 11-17 → 11-25 on all six fronts, memory
from 10-30, q-bio.NC for the week (first time enumerated), software record.
The log has no line for Mon 11-23; cause not knowable from this desk.

**Marks.** [V] short quote or table cell relayed from page text · [V†] a
summariser's extraction (HTML past ~40k chars) · [c] source not opened ·
[R] read in this repo today. Every paper reached me through a fetch agent that
caps quotes at 125 characters: no abstract was read whole.

## Coverage

| Front | Enumerated (rows in window) | Opened |
|---|---|---|
| 1 Cores | DreamerV3 (4), TD-MPC2 (1), model-based RL (7), active inference (2), world model (3), plasticity (1) | 7 papers |
| 2 Fusion, senses | multimodal world model (2), tactile RL (3), interoception (1), body schema (1); modality imbalance, olfaction, odor, homeostatic RL (0 each) | 3 |
| 3 Memory | episodic + RL (3), continual RL (7), loss of plasticity (2), replay + sleep (1); complementary learning systems, working memory (0) | 5 |
| 4 Curriculum | API only: UED (1), intrinsic motivation (2), Craftax (3); autotelic, open-ended, learning progress (0) | none cleared |
| 5 Worlds | release pages; API: survival + hunger (0) | 4 pages |
| 6 Adults, ears | API: keyword spotting (2), VLM reward (16+), child-directed speech (1); vocal learning, language advice, grounded words (0) | 3 |
| Biology | q-bio.NC past week, 42 entries, 4 on topic | 3 |

**Gaps (not "nothing found").** `arxiv.org/search` answered ~20 pages, then
HTTP 429 on 23: fronts 4, 5, 6 went through the API host at 15–25 rows a
query, a thinner net. Front 5: one API query, one web search, both empty —
weak evidence of absence. No ASR / TTS release-card sweep ("TTS Lite", 112M,
CPU, is [c]). SBX and dm-control on PyPI unread (bot challenge). 2611.29888
(replay in noisy RNNs) HTML 404. Humanoid and procedural-world queries never ran.

---

## NOMINATIONS

### N1 — Measure the moment after a change at 15-step resolution (hand-over gate `LF.06`; "learn then lose")

**Source.** *The Stability Gap in Continual Reinforcement Learning*,
arXiv:2611.26252 (11-19); code `github.com/sebapusch/stability-gap-drl` [V].
Whole HTML read by the fetch agent, no truncation.

**Verified.** Tasks keep dynamics and reward and differ only by an orthogonal
transform of the observation [V]. DQN and SAC-D on CartPole, DDPG and SAC on
InvertedPendulum; two hidden layers of 256; settings tuned on 10 seeds, results
on **50 new seeds**, 95% bootstrap intervals [V]. Old-task performance is
evaluated **every 15 steps for the first 30,000 steps** of each new task [V].
"All four algorithms exhibit a sharp, substantial drop in previous-task
performance after at least one task transition" [V] — and it persists when the
optimiser state and the replay buffer are kept, when exploration is reset, and
under IQM [V]. Weights interpolated between checkpoints show large dips for DQN
and DDPG, not for SAC-D or SAC [V]. Numbers are in figures only; total steps
per task and hardware not found.

**Seat / arena.** Death & persistence (VACANT; `LF.02`, `XL.*`) and END-FIT row 2,
whose hand-over gate `LF.06` is built and parked [R]. Not a method: a
**measurement arm and a known-answer**. (a) The CartPole / SAC-D row is our
size — a published result brain B's family can reproduce (LESSONS 17). (b) Any
gate reading "still learning without losing what he knew" samples the old skill
densely straight after the change and prints the minimum, not the end value.
`MEMORY.md:39` cites only the supervised original (2306.01904) [R].

**Cost.** CPU minutes a seed (2×256 on CartPole); 50 seeds is their number, not a need.

**Might win:** a transient collapse is invisible to an end-of-task score, and
in a world that kills a transient is a death; CHAMPIONS records "loses it in
10 of 10" for brain B [R]. **Might lose:** (i) their change rotates the
observation — a hand-over changes nothing he senses, so the gap may not appear;
(ii) four classic-control tasks, no effect sizes in text; (iii) a symptom, no
cure; (iv) the forge's loss happens with NO task change, which this paper does
not study.

### N2 — Resets WITHOUT layer norm, as one more forge candidate (Learning core, brain B)

**Source.** *Towards Understanding the Impact of Plasticity Loss on RL in
Stochastic Environments*, arXiv:2610.42100 (10-30; EWRL 2026); code
`github.com/PhilippBordne/noisy-plasticity` [V].

**Verified.** SAC on three DeepMind Control tasks with controlled observation
and action noise, "one million environment steps with default hyperparameters",
replay ratios 1 / 4 / 8, "mean and std over 5 seeds per method and its best
replay ratio" [V]. Four arms: plain, layer norm, resets, resets + layer norm
[V]. **"Resetting alone outperforms resets combined with LN at all three
observation noise levels"** [V]; resets help on the two harder tasks and more
as observation noise grows [V†]; under action noise no consistent ranking [V].
No scores in text — figures only; the authors do not test the mechanism [V†].

**Seat / arena.** Learning core, forge pool. `FORGE_POOL.md` has LayerNorm on
the critic ON in the baseline ("lost 10/10") and K1 = resets on top of it [R].
The paper's one clean sentence is about exactly that pairing. Arm for a LATER
round (round 1's terms are frozen, untouched): K1 with the critic's layer norm
off, beside K1 as written. Control: the baseline with layer norm off and no resets.

**Cost.** One candidate at the forge's own run size, on Kaggle CPU kernels.

**Might win:** his eye is the noisy channel (J17: the critic runs away on eye
content [R]) — the regime where the paper's ordering is clearest. **Might
lose:** (i) PRIORS §21's recipe (BRO, SimBa) puts layer norm IN, on far more
tasks than three; (ii) their noise is added Gaussian, his is content;
(iii) "best replay ratio per method" is a per-arm selection; (iv) effect size
unknown — no number reached me.

---

## EVIDENCE ON LIVE NOMINATIONS

- **11-19 N3 (teacher-proposed curriculum) — a point against the cheap form.**
  arXiv:2611.25742 (11-19): a VLM names goals for a PPO learner in
  Craftax-Classic at every step. The open model as shipped (Qwen3-VL-8B)
  scores **26.7 ± 3.3 vs PPO alone 24.8 ± 5.7** (5 seeds, 5M steps, ± is std)
  [V]; fine-tuned on 2M samples built with closed models, on four H100s, it
  reaches 34.9 ± 0.7 [V]. No shuffled-goal control, no code. An untuned free
  adult consulted every step is inside the noise of no adult — which favours
  N3's write-the-order-once form.
- **11-12 N1 (judge gate; "yes, queued" [R]).** arXiv:2611.22854, code: a
  pairwise judge seeing only rollout frames picked the better candidate in
  169 / 180 comparisons, confidence poorly calibrated (ECE 0.157) [V†]. The
  judges are closed GPT-5.x, $227 in calls [V†] — not free, no open judge tried.
- **11-16 N1 (pink noise) was run:** `PILOT_B.md:163` — "not seen at this rung" [R]. Closed as tested.
- Still unentered, no repo hit [R]: 11-19 N1 (2609.38598), N2 (2609.39151),
  N3 (2610.04604); 11-12 N2 (2611.07327); 11-11 N1–N3.

## CHECKS ON THE RECORD

- **MLX 0.32.5 (11-25)** [V]: "Fix strided_scan reading/writing out-of-bounds".
  So a scan primitive exists (0.32.3: "shapeless compilation of scan
  operations" [V†]) and had a memory bug through 0.32.4. PRIORS §11 lists scan
  as missing — likely stale; any Mac-port rung (J21) that touches scan should
  record its MLX version.
- **MuJoCo:** 4.0.1 (11-11) still newest [V]; unreleased notes already remove
  `mjContact.H` [V]. `requirements.txt:23` still `mujoco>=3.1.0`, unbounded [R]
  — the 11-16 hazard stands.
- **mujoco_playground v1.1.0 (11-23)** [V]; notes point to an off-site changelog, unread.
- `danijar/dreamerv3` head still `e01491f` (10-04) [V]; Craftax 1.7.0 [V];
  TD-MPC2 no successor announced [V]. SBX, dm-control: unread.

---

## WATCHLIST

| Item | What it is | Promote on |
|---|---|---|
| **When a world model needs context** 2610.06651v2 | DreamerV3 (9.7M), 3 seeds, 1M steps, **one 24 GB L4** [V†]. Where a per-episode variable is decodable from current state and action (R² ≥ 0.93) a context encoder adds little; context-blind Dreamer leads at slow drift, conditioning wins as drift grows, several gaps inside two standard errors [V†]. A pre-test for any "world changes between lives" arm: probe decodability first. Anonymous code. | PRIORS' food-re-drawn fork ordered; a named repo. |
| **VIBES** 2611.34641 | Dreamer, pixel reconstruction replaced by an adversarial sufficiency term, **3.26M**; under distractors 701 ± 29 vs DreamerV3-12M 277 ± 10 (5 seeds), term removed 606 ± 27; a tie without distractors [V†]. No code. | Code; brain A failing on eye content as B does. |
| **Can a world model doubt?** 2611.34830 | Uncertainty read from a DreamerV3 prior is attenuated by the recurrent path; the posterior recovers it but is unusable in imagination [V†]. One model per family, H200. Bears on disagreement curiosity on brain A. | A second instance; code. |
| **Spectral successor features** 2611.26504 | Eigenvectors of the transition graph as a fixed basis; goal moves cost "only brief drops" [V]. Online variant sees its cell index; walls never changed; no numbers, no code. Kin to the carried place key (ruling 63). | Layout change; a from-senses basis. |
| **Load-balancing neurons** 2611.32691 | Firing-rate thresholds, no trained weights; ~+1 point over resets on continual CIFAR / ImageNet [V†]. A100, 11M nets. | An RL result at our size; code. |
| **HaM-World** 2605.05951v3, code | TD-MPC2 + memory; abstract (91.9) and extracted table (117.9 vs 107.7) disagree. | Numbers reconciled. |
| Carried | Evolutionary Neuroplastic Agent (re-read: 0.0% at threshold on HalfCheetah vs SAC 47.1% [V†]); the seven other 11-19 items. | As 11-19. |

**Biology, one line.** A ring model moving from narrow to wide excitation
(2611.28601, rate units, no learner): wide bumps hold **7–10 items against ≥ 40**
for narrow, relax 35–60× slower, and need a 35–230× stronger cue to add an item
[V†] — capacity is bought with fragility; a dial the seated ring (`WM.02`) has
not turned.

---

## NO-ACTION

- **Cores:** nothing at our size in a needs world. 2611.34656: a coding agent reads the simulator's source and writes the policy — nobody's brain. CommDreamer 2611.24244: no seeds.
- **Fusion:** tenth sweep without a paper. Tactile work rides π0.5 or human demonstrations; in 2611.32101 touch is a training reward only, absent at deployment [V].
- **Body schema:** 2611.22713 imitates on a frozen DINOv2; schema removed 24 / 90 vs 75 / 90, one seed [V†].
- **Memory / retrieval:** no episodic-store paper for a from-scratch agent since 10-30. 2611.07327 (11-12 N2) unchanged.
- **Curiosity / UED / open-ended:** nothing new past 2611.13113 (11-11).
- **Worlds:** no new survival, needs or crafting world found (thin search).
- **Smell, taste, needs, interoception:** zero rows; 2611.27795 is a wellbeing framework.
- **Ears / voice:** no from-scratch small ear in window. 2606.16807 finds words in speech from image captions using frozen HuBERT and three captioners (spotting 85.3 vs 53.8) [V] — an adult's method, never his.

**Philosophy served:** 4 (a gate that samples the dip, not the end), 3 (an arm
the forge's own baseline makes askable; a live nomination weakened by new
evidence), 7 (hand-over row), 2 (adults sorted from parts), 5.
