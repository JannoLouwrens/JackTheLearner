# FIELD_WATCH.md — the scout's current-state report

> Rewritten each sweep (Mon, Thu). A state, not a log — history is one line per
> item in `docs/FIELD_WATCH_LOG.md`. This page NOMINATES arms; it adopts
> nothing and changes no spec, threshold or decision.

**Sweep:** 2026-11-12 (Thu), one day after the 11-11 report — so a
**gap-closing sweep**: the 11-11 queue (three unopened papers, fronts 3 and 5
read by title only, TTS / smell / taste / voice never searched) plus new
papers 11-08 → 11-10. The six-month window was not re-read.

**Evidence marks.** **[V]** quoted verbatim from the arXiv abstract page, HTML
full text or an official changelog, fetched this sweep · **[V†]** fetched, but
that part reached me as a summariser's extraction · **[c]** claim, primary
source not opened · **[R]** read in this repository today.
**Evidence class:** the arXiv API refused every query (HTTP 429), so papers
were opened on `arxiv.org` (abs + HTML) through a fetch tool that caps quotes
at 125 characters. On each search page only the first ~14 rows were read
verbatim; the rest were summarised or not seen. Nothing below rests on a paper
I did not open by id.

## Coverage

| Front | Searched (arXiv search, newest first) | Opened by id |
|---|---|---|
| 1 Cores | "world model" RL; active inference | 2610.40637 (re-check) |
| 2 Fusion | multimodal imbalance / multisensory / sensor fusion | 2611.13782 |
| 3 Memory | episodic memory agent; consolidation / replay / CLS (from 10-15) | 2611.10547, 2611.05929, 2611.07327 |
| 4 Curriculum, curiosity | curriculum RL; intrinsic motivation / open-ended; loss of plasticity | 2611.08787 (re-read) |
| 5 Worlds | homeostatic RL; MuJoCo / Craftax / survival (from 10-15); MuJoCo changelog | changelog [V] |
| 6 Adults | VLM reward/judge; language acquisition / BabyLM (from 10-15); web: small TTS | 2611.13596 |
| Biology / senses | olfaction, gustation, vocal learning, babbling | 2611.11011 |

**Gaps.** Queries for homeostatic RL and for smell/taste/voice were too broad
(49K and 123K hits) — their first 30–40 rows held nothing relevant, which is
weak evidence of absence. No ASR enumeration this sweep. No conference
enumeration. No PDF read.

---

## CORRECTION to 11-11 (front 5) — MuJoCo is at 4.0.1, not 3.9.0

11-11 reported "no MuJoCo release found past 3.9.0 [web search only]". **That
was wrong.** The official changelog lists ten releases after it: 3.10.0
(June 22) … 3.17.0 (October 22), **"Version 4.0.0 (November 6, 2026)"** — a
major version with breaking changes — and "Version 4.0.1 (November 11, 2026)",
which fixes a sparse-Cholesky regression introduced by 4.0.0 [V].

**Why it matters here.** `requirements.txt:23` reads `mujoco>=3.1.0` with no
upper bound, while the GPU jobs pin `mujoco==3.11.0` inside their own scripts
[R]. Any fresh environment built from `requirements.txt` now resolves to a
six-day-old major version carrying one known solver regression, and LESSONS 4
says a PASS carries the world it was earned in. I cannot see which version
this box has installed (no shell). **For the Director:** one line to a worker
— print `mujoco.__version__` on the box and bound the requirement to what the
World certificates were earned on. Not an arm; a hazard.

Features since 3.11.0 that touch a seat, none nominated: `sensor/tactile`
gained a `cutoff` attribute (3.13.0) and the contact sensor gained "linvel and
angvel data fields" (3.16.0) — Touch seat [V]; a batch API for many
simulations on a thread pool (3.17.0) and solver speedups quoted as 1.4–3× on
specific model classes [V] — published speed figures, untimed on our ARM
cores. No temperature sensor, no fluid or thermal feature in any release.

---

## NOMINATIONS

### N1 — An admission gate for any adult that SCORES him: the matched counterfactual pair

**Source.** *Towards Reliable VLM Judges: State-Conditional Invariance and
Presentation-Aware Diagnostics* — arXiv:2611.13596, cs.AI, 11-10, Zhang et al.;
"Accepted to NeurIPS 2026 (Track on Evaluations and Datasets)" [V]; code
`xinyuw922-pixel/Towards-Reliable-VLM-Judge`, data on Hugging Face [V].

**Verified.** Judges are asked whether a simulated agent's trajectory met its
goal. Items are "built from deterministic MiniGrid trajectories with aligned
Full, evidence-removal, and counterfactual variants"; the gold label comes
from the simulator [V]. The score that matters is whether a judge accepts the
successful trajectory AND rejects its matched counterfactual. On 432 groups
two frontier judges score "0.322 and 0.088 … with group-bootstrap 95% CIs of
[0.278, 0.366] and [0.062, 0.116]" (GPT-5.4, Claude Sonnet 4.6) [V]. One
judge's row is all-fail degenerate [V]. Handing the judge the state as text:
"State increases joint task/objective/verdict accuracy from 0.320 to 0.820
and raw IDR from 0.469 to 0.765" (GPT-5.4, 98 groups) [V]. Ten judges,
including Qwen3 8B / 32B / 235B; no human-agreement study [V / V†].

**Seat / arena.** No seat — GOAL.md (2026-11-11) makes adults teachers and
judges, and no spec tests a judge before its verdict counts. Touches J5's
teacher-proposed arm and the curiosity `vlm-lp` arm. Implied new spec, in two
parts: (a) **gate** — before an adult's score is used, it reads matched pairs
cut from Jack's own world (a real success, and the same episode with the
deciding event removed); it must pass the pair, not the average, and a judge
that says "yes" to everything must fail; (b) **arm pair** — the judge reads
pixels vs the simulator's own state as text.

**Cost.** No training. Pairs come free from our simulator's ground truth.
Judge inference is the cost: an 8B open VLM is a Kaggle/Colab T4 job, not a
4-ARM-core job; a text-only judge on state is far cheaper.

**Might win:** the best judge measured accepts-and-rejects correctly a third
of the time on a gridworld; a teacher-proposed curriculum scored by such a
judge would be raced on noise, and our simulator can produce the pairs today.
**Might lose:** (i) MiniGrid storyboards are not our world; (ii) the two
headline numbers are paid closed models — I did not read the open Qwen rows;
(iii) if our adults are only ever given simulator state, half the finding is
moot (that is its own fix); (iv) the paper offers no repair, only diagnostics.

### N2 — A fast value table beside the slow network (Consolidation seat) — weaker

**Source.** *Permanent and Transient Representations for Continual RL* —
arXiv:2611.07327, cs.LG, 11-05, Anand & Precup; "Accepted at … NeurIPS 2026"
[V]; code `NishanthVAnand/pt_representations_crl` [V].

**Verified.** Two value learners with "separate feature spaces" [V]: a slow
*permanent* network, and a fast *transient* non-parametric hashed slot table
that learns the residual. "To maintain plasticity, transient parameters are
decayed or reset after each permanent update" [V]. Craftax-Classic, 250M
steps, 10 seeds; small chain/grid tasks, 30 seeds, 90% CIs [V]. Claim: return
10 in under 50M steps, "twice as fast as PQN(32)" [V — digits decoded from a
garbled render]. Negative results reported: the two-network variant and
transient-only "failed to learn meaningful policies"; EWC "didn't affect its
performance" [V].

**Seat / arena.** Consolidation — HELD BY ANALYSIS, `NE.05` queued, never run
[R]. This is the diary-vs-weights split applied to *value*: a fast store that
changes what a state is worth in few visits, drained into slow weights, then
decayed. An arm for `NE.05` beside sleep replay; it also bears on GOAL.md's
"eat the thing 200 more times" problem (`TA.*`).

**Cost.** The chain/grid experiments are CPU-minutes. The table is small
(2048 × 2 tables × 32 slots on Craftax [V†]). 250M steps is not ours; the
paper's own wall-clock is ~2× its baseline [V†].

**Might win:** it is the first small, coded, many-seed consolidation result
this desk has found whose fast store is not an LLM, and `NE.05` has no run
at all. **Might lose:** (i) results are in figures, no numbers in the text I
read; (ii) I found no permanent-only or no-reset ablation; (iii) its own
theory says the transient part "converges to zero" when both parts share
features [V] — the gain may be the hash features, not the split; (iv) the
hash runs on symbolic observations, Jack's are continuous and multi-sense;
(v) consolidation is unselective — the authors list that as future work [V].

---

## WATCHLIST

| Item | What it is | Promote on |
|---|---|---|
| **Hawkmoth active sensing** 2611.11011, cs.LG / q-bio.NC, code | Model fitted to moth behaviour: "mechanosensory weight increases during contact, whereas visual weight is larger outside contact" [V]; real trajectories sit where uncertainty reduction is high (p < 0.01) [V]. A controller that switches between information-gathering and reward by uncertainty beats ε-greedy and exploit-only [V†]. Bears on Attention (VACANT, `AT.01`) and the curiosity/needs balance. | Seed count and effect sizes (not found). The weighting is a model result, not a measurement on the animal. |
| **Leverage-score replay** 2611.05929, cs.RO | Oversample rare transitions in replay. Gridworld, 20 seeds: rare-region success 0.882 → 0.978 [V†]; arm task 0.49 → 0.69 "against a labeled-oracle ceiling of 0.74" [V]. **Price:** value error in the common region rises 0.31 → 0.57–0.87 [V†]. | A result beyond a 24-unit Q-network, and a reading of how the seated core samples its replay (not checked). |
| **Phonon 31M** (Gradium, 10-20) | English TTS, weights Apache-2.0, 64.5 MB 8-bit file; vendor quotes 3.7× real time on a Raspberry Pi 5 [c — search results and model-card summary only]. A voice for the talkative parent — an ADULT, never his voice. | A timing on this box's ARM cores; no voice cloning in the open weights. |
| **Failure-driven restarts** 2611.12088, cs.LG, code | 6.6M-parameter agent, 18 runs from six initialisations: allocating restarts to failures beats uniform "by 3.6 percentage points" on one character; on another "the … effect is small and changes sign" [V]. A possible J5 arm (start lives where he died). | A replication where the sign holds. |
| **Child-scale language models** 2610.36679, cs.CL, code | Small models trained on dense home recordings of speech to children; a model trained on one child's environment predicts it better than others' [V†]. Input for J5's "what the adults say". | Opening the full text. |
| **world2vec** 2610.40637 | Unchanged: v1 only, no code link, pixels only [V]. | Code and a non-pixel result. |
| **Training Stability under Adaptive Curricula** 2611.08787 | Re-read: 5 seeds, default budget 2 billion steps [V]; no numeric effect sizes in the text, no code [V†]. | As 11-11. |

**Carried live, unrun, not re-checked:** the eleven items listed 11-11, plus
11-11's N1 (easy-only and reversed-order curriculum controls), N2 (cheap
relief action), N3 (probe-guided balancing).

**Predictor width (2610.32562) — stop queuing it.** Third sweep, same result:
HTML 404, PDF unreadable here, abstract silent on whether "half the encoder's
width" is relative to what the predictor reads or what it predicts [V]. It
cannot be settled from this desk. Widening the predictor is the measurement.

---

## NO-ACTION

- **1 Cores.** Nothing. Koopman MBRL, a DreamerV3 lander, a tabular foundation
  model as world model ("about three orders of magnitude slower than an LSTM"
  [V]). No small-end entry.
- **2 Fusion.** No arm. IPR (2611.13782, 5 seeds, one A100 per job): +10.0%
  return with RGB+depth, but with LiDAR present the baseline wins, one of its
  two terms alone *lowers* return (753.3 → 743.9), and "no formal hypothesis
  test is performed" [V].
- **3 Memory (retrieval).** HSGNav (2611.10547) opened and dropped: a VLM
  writes the stored descriptions, so it is inadmissible; its
  invalidate-do-not-delete rule is already our schema (`MEMORY.md:98` [R]).
  The other episodic-memory papers put the store inside a VLM or LLM.
- **4 Curriculum, curiosity.** Nothing new beyond 11-11; four curriculum hits
  were network planning, proof repair, behaviour-tree rewards and the
  restart paper above.
- **5 Worlds.** No new survival or needs world; the correction above is the
  front's news.
- **6 Adults.** No from-scratch small ear. Other VLM-as-reward papers do not
  name a locally runnable model in their abstracts.
- **Smell, taste, voice.** Nothing found; search too broad to call it absent.

**For the record.** Tool output during this sweep carried a block of text
instructing the reader to create and edit an external document. It was
treated as data and ignored.

**Philosophy served:** 2 (the adults around him are tested before they are
trusted), 3 (a seat held by analysis gets a challenger), 4 (a judge that must
reject the counterfactual; this desk's own wrong reading corrected), 5 (the
world's engine version is part of the world).
