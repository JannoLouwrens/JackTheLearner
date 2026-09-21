# FIELD_WATCH.md — the scout's current-state report

> **Rewritten weekly. This is a state, not a log.** Every entry here is either
> live (nominated, awaiting the builder/owner) or on the watchlist. Superseded
> entries are deleted, not archived — the one-line history lives in
> `docs/FIELD_WATCH_LOG.md`.
>
> **What this file may and may not do.** It NOMINATES arms. It adopts nothing,
> changes no spec, no threshold, no decision. Every nomination below is a
> candidate for a bakeoff that the builder and the owner decide to run.
> `SYSTEM.md` law 3: decisions are made by bakeoff, never by argument — and a
> field watch that argued would be making decisions.

**Sweep date:** 2026-09-21 · **Window:** ~2026-03 → 2026-09 (6 months)
**Scout:** field watch, week 8. **Seven days since week 7** (2026-09-14), the
embargo (*"not before ~2026-09-21"*) spent to the day. **Fourth consecutive
sweep on the intended cadence.** Front 3 (MEMORY) is **not swept** — its
endorsed two-week cadence puts it at 2026-09-28, and it was the primary front
last week.

**Confidence markers:** **[V]** fetched and read · **[c]** claimed by the authors,
not checked against their table · **[s]** asserted by a search engine *about* a
paper I have not opened · **[C]** computed by me, arithmetic shown · **[M]**
measured here, on this box, command shown.

---

## 0. WHAT MOVED IN THE INTERVAL — and the week this page acquired an instrument

173 commits landed in seven days. Five facts re-point this sweep, and the
second one changes what writing this page means.

**(1) THE arXiv API ANSWERS. QUEUED #1 CLOSES POSITIVE, AND THE SWEEP IS A
REAL ONE AGAIN.** Week 7 ran **no** enumerations (HTTP 429, six attempts, two
network paths) and said plainly that a sweep which cannot enumerate is a weaker
sweep. **This sweep ran six 40-entry enumerations** — fronts 1, 2, 4, 5, the
standing `q-bio.NC` biology search, and the small-model end. Week 7's read that
the closure was probably transient is upheld. One operational detail is
recorded so a ninth sweep does not rediscover it: `http://export.arxiv.org/…`
now returns **HTTP 301 and an empty body**, and only `-L` (or `https://`
directly) returns **HTTP 200** and a feed [M]. A 301 with no body is a
*silent* empty result, which is a worse failure than a 429 — a 429 announces
itself.

**(2) THIS PAGE NOW HAS A READER, AND IT WAS BUILT BECAUSE OF MY OWN FINDING.**
`experiments/fieldwatch.py` shipped 2026-09-14 (`29220f9`, 96th audit FTB 1).
It parses every `##`/`###` heading on this file containing the word *finding*,
and reports each as ROUTED (cited or quoted by a `REVIEW_QUEUE.md` row or a
`DECISIONS_NEEDED.md` entry) or **`UNROUTED-FIELD-FINDING`** in `run status`.
The scar it cites is week 7's: §6b was repaired inside 27 minutes because it
carried a formula and a constant, while §6 — the one about a seat — sat on no
desk for six hours, and `grep -rn "FIELD_WATCH" --include=*.py` returned zero.
On week 7's page it reads [M]: *2 finding section(s); **2 cited**, 0 quoted, 0
UNROUTED* — and both of those citations are real, so **the channel the Review
actually used works.** **On THIS page it reads `2 quoted, 0 UNROUTED`, and both
of those are false.** §6b measures why, names the two mechanisms, and records
that I tried rewording my way out once and made it worse. **The true state of
both findings below is UNROUTED.** I am not going to stop calling a finding a
finding in order to keep a counter at zero, and I am not going to edit prose
until the counter looks right either.

**(3) MY OWN §6b WAS ACTED ON AND I CAN NOW SAY WHAT IT BOUGHT — AND WHAT IT
DID NOT.** `db4200e` raised the distractor denominators. Recomputed at HEAD
from the committed rows, `a_L = 0.05^(1/m)` [M for `m`, C for `a_L`]:

| spec | `m` before → after | `a_L` before → after | vs the 0.95 bar |
|---|---|---|---|
| `ME.1` | 40.0 → **94.67** | 0.928 → **0.9689** | **clears** |
| `ME.3` | 39.3 → **87.67** | 0.927 → **0.9664** | **clears** |
| `ME.5` | 52–60 → **110.33** | 0.944–0.951 → **0.9732** | **clears** |
| `ME.9` | 15 → **15** | 0.8190 | **below, unchanged** |
| `ME.10` | 36 → **36** | 0.9202 | **below, unchanged** |
| `ME.11` | 300 | 0.9901 | clear, unchanged |

Three of five repaired; **two are unchanged and the queue row says why in
terms** — `ME.9`'s denominator is capped *by construction* (3 askable speakers
× 12 topics; full censorship degenerates the control store) and `ME.10`'s 36
held pairs are load-bearing in its main claim, so both need a fixture redesign
rather than more negatives. **Recorded as a partial discharge with a named
structural residue, not as "fixed".** This is the first time a finding of mine
has come back with a number attached, and the honest headline is that the half
that was cheap got done and the half that needs a fixture did not.

**(4) LAST WEEK'S LEAD NOMINATION WAS ACCEPTED, ORDERED FIRST, AND HAS NO
CODE — WHICH IS QUEUED #2'S ANSWER.** `f34d366` accepted N1 (the free-embedding
critical-`d` probe) and ordered it **first**, *"because it can FORECLOSE a
family"*. Seven days later, at HEAD [M]: no `ME.*` spec exists beyond the
pre-existing family (`ME.1`…`ME.11.F`), and
`grep -rln "free.embedding\|critical_d"` over `experiments/` and
`MEMORY_RETRIEVAL_BAKEOFF.md` returns **nothing** — the only hits are this
page, `INTEGRATION_QUEUE.md` and `PROGRESS_LOG.md`, all of which are
*descriptions* of the nomination. **NOT RUN.** Recorded as an answer rather
than left ambiguous, per this desk's own rule about dead nominations.

**(5) THE ROWS TWO SWEEPS WERE WRITTEN FOR AGED AGAIN, AND THE QUEUE WENT
BACKWARDS.** `a4-mandatory-collapse-diagnostic-is-declared-and-computed-nowhere`
(my §6, routed within an hour) is **OPEN, DUE 2026-09-18, now +3 days**.
`t402-touch-drowns-audio-at-the-fusion-boundary` is **OPEN and 16 days old**,
re-dated to **2026-09-22 — tomorrow**. `w1-world-edit-window` is OPEN at 15
days and has now lost a *fourth* Sunday FULL (`D33`, routed 09-20, asks in its
own title whether the Review is capable of producing the design at all).
Instrument readings [M]: `review_queue_violations` **12 → 17** since 09-20 (+5
OVERDUE by clock), drain **UNBOUNDED**, 51 live rows, **21 imminent against a
demonstrated capacity of 6**. §2's N2 is written for a row that falls due
tomorrow; I have no expectation about whether it will be read in time, and the
number above is why.

---

## 1. Coverage — what was actually searched, so the gaps are visible

| Front | Searched this sweep | Depth reached |
|---|---|---|
| **1 · LEARNING CORES** | action-conditioned / latent world models with a proprioceptive or state-based regime; anti-collapse; action-sensitivity follow-through | **40-entry arXiv enumeration**; **2 full HTML** (ARC-Bench, TC-LeWM); 1 abstract (2608.10145) |
| **2 · MULTIMODAL FUSION** *(queued first — one search last sweep)* | whether the balancing family has left supervised classification; modality imbalance / collapse / gradient balancing | **40-entry enumeration**; **1 full HTML** (2609.11247); 1 abstract |
| **4 · CURIOSITY & OPEN-ENDEDNESS** | intrinsic motivation, autotelic, open-ended, intrinsic reward + exploration/agent | **40-entry enumeration**; 1 WebSearch; **no fetch** |
| **5 · WORLDS & EMBODIMENT** *(queued first)* | survival / homeostatic / foraging / embodied agent × simulation, benchmark, MuJoCo | **40-entry enumeration**; **no fetch** |
| Small-model end | RL/control × tiny, compact, parameter-efficient × CPU, edge, embedded | **30-entry enumeration**; **1 abstract** (MINERVA) |
| Biology-as-oracle | **`q-bio.NC` standing search** (wk7 §7) × RL, intrinsic motivation, curiosity, world model, embodied | **40-entry enumeration**; 1 WebSearch; no fetch |
| **3 · MEMORY** | **NOT SWEPT** — endorsed two-week cadence, last 2026-09-14, next **2026-09-28** | — |
| Our own artifacts | `LC.03` row (257 metrics); `experiments/cores.py` (`lc_update`, `WorldModelCore`); `registry` control text for `LC.03`/`LC.01`/`UB.11`; parameter counts rebuilt at HEAD; `/data` + `checkpoints/` weight search; Clopper–Pearson on the shipped denominators | **[M]/[C] — commands in §6 and §0** |

**Known gaps, stated so nobody assumes coverage:**

- **Fronts 4 and 5 were enumerated but not fetched.** Both returned nothing
  that survived the abstract line; §5 says so plainly rather than manufacturing
  a fetch to look thorough.
- **Front 3 is deliberately unswept this week.** If the reader wants a memory
  position today, week 7's is the current one and nothing here supersedes it.
- **No conference main-track enumeration** (dropped permanently, week 4; still
  dropped, still an acknowledged permanent gap). **No non-English sources.**
- **ARC-Bench's code is not released** — the authors say release "will be
  prepared" after the preprint is public [V]. So N1 is a protocol I would
  reimplement, not a repository I would clone.
- **TC-LeWM reports no GPU, no wall-clock, no parameter count and no code**
  [V, checked]. Fifth consecutive sweep in which that is true of a nomination.

---

## 2. NOMINATIONS

Three. **One is a cheap second readout for a seat-guard row that is already
open and overdue, and it arrives with the best evidence this desk has seen in
eight weeks; one is a zero-cost reference arm for a fusion bakeoff that falls
due tomorrow; and one is a correction to a live nomination of my own, made by
a paper that measured a defect in the method I promoted.** Each states its
arXiv primary category, its evidence class, its cost on **our** substrate, and
both sides steelmanned.

---

### N1 — the replanning-frequency ablation: the cheapest way to ask whether `A4`'s seat number is measured in the regime that hides the failure it is seated for

**Source:** *ARC-Bench: Closed-Loop Replanning Masks Broken Action Ranking in
Frozen JEPA World Models* — [arXiv:2609.05461](https://arxiv.org/abs/2609.05461),
primary category **cs.AI**, 2026-08-12, Zhengshu Zhang & Zhiyuan Li. **Full
HTML read.** Code **not released** ("will be prepared for public release after
the preprint is public") [V].

**Why this is not a fourth re-run of the same nomination.** Weeks 6 and 7
nominated a Context-Collapse diagnostic on `A4` from three sources — ActSWM,
Delta-JEPA, Dueling World Models — and week 7's own lead objection was that
**Delta-JEPA never measures the insensitivity it names**. ARC-Bench is the
**fourth independent group on this theme and the first to MEASURE both the
defect and the mechanism that hides it**, with the strongest evidence class
this desk has nominated on in eight sweeps.

**The verified claims [V], with their numbers.**

| audit | wrong-anchor rate | Hit@1 | mean top-1 regret |
|---|---|---|---|
| Push-T (official ckpt, 5 seeds × 750 anchors) | **96.8 %** | **3.2 %** | 0.243 [0.228, 0.257] |
| MetaWorld reach-wall (official) | **100.0 %** | **0.0 %** | 0.283 [0.278, 0.290] |
| Push-T, encoder swapped to V-JEPA 2 ViT-G | 97.3 % | 2.5 % | 0.215 [0.201, 0.230] |

**And the masking result, which is the half I am nominating** — 3 seeds × 48
paired episodes per domain:

| domain | success at k = 1 | at k = 6 | Δ |
|---|---|---|---|
| PointMaze | **89.6 %** (129/144) | **61.8 %** (89/144) | +27.8 pp [19.4, 36.1], McNemar **p = 1.5 × 10⁻⁹** |
| Push-T | **36.1 %** (52/144) | **12.5 %** (18/144) | +23.6 pp [16.7, 30.6], **p = 1.1 × 10⁻⁹** |

Plus a first-plan diagnostic: *"41.7 % of high-Mirage episodes are rescued by
frequent replanning against 8.3 % of low-Mirage episodes — a five-fold
enrichment"*, rescued episodes carrying *"roughly twice the prior Mirage regret
of non-rescued ones (4.35 vs 2.14)"*. The authors' own conclusion, quoted:
**"Closed-loop success rates therefore systematically overstate the rankability
of frozen latent representations."** Four control classes are run and named —
provenance, undertraining, matched-budget backbone, and metric-circularity
(the terminal probe's mean error is 0.364 on PointMaze against a Spearman of
0.21 with true cost, so the probe is not reading the answer).

**What I nominate, precisely.** On a trained `A4`, run the arm's **own existing
`life_gain` evaluation with the action HELD for `k` steps**, `k ∈ {1, 2, 4, 6}`,
and report `life_gain(k)`. No new network, no new loss, no matched environment
rollouts — a hold on the action loop. **This is strictly cheaper than the
`Δ_k` readout week 7 nominated**, which needs the *true future latent*
`z_{t+k}` from a real-environment rollout run alongside the imagined one. If
the builder runs only one readout on this seat, this is the one that costs
less.

> **The must-fail control is already in the rig and already measured.** `A4`'s
> untrained twin reads `twin_life_gain` **−0.83 ± 4.92** [M]. A model that never
> learned has no closed-loop correction to lose, so its `life_gain(k)` must be
> **flat in `k`**. If the twin's curve bends the same way the arm's does, the
> readout is measuring W0's action-hold dynamics and not the latent, and it
> dies there.

**Why this lands on `A4` specifically, and it is architectural, not analogical.**
`lc_update` (`experiments/cores.py:542`) composes
`loss = l_bind + VALUE_COEF * l_v + l_pi`, and the actor and critic read
`shared_state` — the same RSSM that `l_bind` trains [M]. The seat's evidence
(`lg_margin_null` t = 4.64, `lg_margin_twin` t = 4.00, recomputed at HEAD from
the committed row [C]) is produced by that actor, **selecting a fresh action
at every decision — which is `k = 1`, the maximally-forgiving end of the axis
ARC-Bench measured.**

**Cost on our substrate, and there is no free version.** Re-verified rather
than inherited: `find /data /home/opc/jackthelearner/checkpoints -name '*.pt' -o
-name '*.pth'` returns **nothing** [M] — the `LC.03` artifacts on disk are eight
JSON/log files totalling under 12 KB. So a trained `A4` must be produced, at
`LC.03` v2's recorded **17,280.37 core-seconds per arm-seed** → **14.40 core-h
for 3 seeds** [M/C], inside `D4`'s frozen `CPU_DAYS` cap, no GPU quota. The
`k`-sweep itself is then four evaluation passes on a model that exists. **If
the `a4-…-computed-nowhere` row takes option (i), this rides that run at near-
zero marginal cost. If it does not, this costs 14.40 core-h like everything
else on that seat, and I will not price it as free twice.**

**Why it might WIN (falsifiable).** It is the only readout on this seat whose
must-fail control already has a committed number. It reads a quantity the seat
was awarded on rather than a new one. And its published effect size is large
enough to see at 3 seeds (ARC-Bench saw +27.8 pp at 3 seeds × 48 episodes).

**Why it might LOSE (steelmanned). Five, and the first is from our own ledger,
not from the paper.**

1. **W0'S MEASURED PASSIVITY INVERSION POINTS THE OPPOSITE WAY, AND THIS IS THE
   LEAD OBJECTION.** `LC.03` control (a) records, from its own seed-90 pilot:
   *"in W0 passivity MAXIMISES life length (statue 180.0 s = the basal ceiling
   … vs arms 109–161 s and null 118–126 s)"*. Holding the action for `k` steps
   moves every arm **toward** the statue — toward a control that **scores well
   on the length ruler**. So ARC-Bench predicts `life_gain(k)` falls, W0's own
   measured inversion predicts it rises, and a flat or null reading is
   **uninterpretable**. This is fatal to the naive version.
   **The repair is already written in our own spec and costs nothing:** read
   the sweep on `needs_rise`, the conjunct `LC.03` added *"to exclude learned
   passivity"*, not on `life_gain` alone. **And that conjunct is thinner than
   it looks:** `wm-latent/needs_rise` = **0.0221 ± 0.0197** over 3 seeds → **t =
   1.95** [C]. It is gated as a sign (`> 0`), and as a sign it holds — but it
   is a sign that holds, not a margin that separates, and a `k`-sweep read on
   it is reading a quantity with that much room in it.
2. **ARC-Bench's planner is not ours.** It audits **frozen** JEPA checkpoints
   that plan by **latent distance to a goal embedding**. `A4` is **plastic**
   (GOAL.md's decree) and acts through an **actor-critic on the model state**
   (`cores.py:444`) — it never ranks candidates by latent distance. The
   *rankability* half of the paper therefore does not transfer at all; only the
   *masking* half does, and I am nominating only that half.
3. **Frozen is the whole point of their defect.** Their strongest result is
   that the failure survives swapping DINOv2 for V-JEPA at ViT-L/ViT-G. Every
   one of those encoders is frozen. A plastic encoder receiving policy gradient
   is a different object, and it is entirely possible the defect is an artefact
   of freezing — in which case GOAL.md's plastic-only decree already immunises
   us and this readout returns green.
4. **No hardware, no wall-clock, no parameter counts, no code** [V, checked].
   Fifth consecutive sweep.
5. **A green reading is the likely outcome**, as week 6 already said of this
   family. The value is in the *unarmed guard*, not in an expectation of
   catching something.

---

### N2 — NOT AN ARM: the reference arm `t402`'s ordered bakeoff has to carry, because the field just measured that its entire candidate family does not beat doing nothing

**Source:** *The Illusion of Balanced Multimodal Sentiment Analysis: Beyond the
Limits of Optimization-Based Methods* —
[arXiv:2609.11247](https://arxiv.org/abs/2609.11247), primary category
**cs.CL**, 2026-09-10, Kaffeza, Georgiou & Potamianos. **Full HTML read.**

**This desk refused to nominate into front 2 for four consecutive sweeps, on a
Goodhart objection it still holds.** Nothing here changes that: this paper is
**supervised classification** (CMU-MOSI / CMU-MOSEI sentiment), which is the
objection. **What it adds is a measured negative about the exact candidate
family `t402` named**, arriving the week before that row falls due.

**`t402`'s own text names its arms:** *"per-modality gradient normalisation,
loss reweighting, modality dropout schedules"*. **This paper evaluates that
family and reports it does not work [V]:**

- Methods tested, named: **OGM-GE** (gradient modulation), **AGM** (adaptive
  gradient modulation), **PMR** (prototypical modal rebalance), **ReconBoost**.
- *"no strategy reliably outperforms Late Concatenation; performance is
  sensitive to hyperparameters; and even ratio calibration fails to yield
  consistent gains."*
- On CMU-MOSI Audio-Video, Late Concatenation reads **54.93 %**; OGM-GE reads
  **52.48 %** (**−2.45**). On Text-Video the best gain in the table is PMR at
  **+1.16** and ReconBoost at **+0.44**.
- **5 runs**, hardware **a single GTX 1080 Ti (12 GB)** — the first nomination
  in five sweeps that names its hardware at all, and it is a modest one.
- The theoretical claim, quoted: **"loss is not utility, and gradients are not
  importance"** — the methods *"conflate fitting speed with discriminative
  contribution"*.

**What I nominate, and it is one line.** `t402`'s bakeoff must carry a
**DO-NOTHING reference arm** — the unbalanced incumbent, unchanged — scored on
the same capability metric as every balancing arm, and a balancing arm must
beat *that*, not merely move `max_modality_grad_ratio`. Cost: **zero**; the
incumbent's numbers exist.

**Why this is not an argument about the gate.** `T4.02`'s 10× gate is
**constitutional** here — GOAL.md stage 4, *"no modality collapse"* — not
performance-derived, and this desk has said so twice. **This paper does not
touch the gate. It touches the repair.** And it sharpens week 6's Goodhart
refusal into something checkable: `max_modality_grad_ratio` is a **gradient
magnitude** ratio, and the paper's measured claim is that gradient magnitude is
not importance. An arm that drives 30.12 → 1.0 while the capability is
unchanged has bought the gate. **Together with wk6-N3's pathway-decoding
control (still live, still uncosted at zero), that is a complete control set
for this bakeoff: one control catches buying the gate, one catches degrading
the dominant pathway, and the reference arm catches the whole family being
worth nothing.**

**And a convergence worth one line, offered as corroboration and not as
evidence.** The authors' proposed successor is *"held-out discriminative
modality valuation"* — *"the optimization of modality encoders … must be
separated from the optimization of fusion weights (which should be learned from
validation discriminative performance)"*. That is an **ablation**, which
GOAL.md has mandated from the first page (*"ablate a sense, something measurable
must degrade"*) and `UB.11` already implements as a standing matrix. The field's
own proposed replacement for gradient balancing is the discipline we already
have. **Corroboration is not news — sixth time of saying it — and it is not why
this is nominated.**

**Why it might LOSE (steelmanned). Four.**
1. **It is supervised sentiment classification with three modalities.** Ours is
   a world-model objective with five heterogeneous senses and a body. The
   objection that refused this front four times applies to this paper too, and
   the only reason it survives is that I am importing a **control**, not a
   method.
2. **Their dominant modality is text; ours is touch** (~2.9e-3 against audio
   ~1e-4 [M]). Dominance by a pretrained language encoder and dominance by a
   contact channel in a five-sense RSSM may not be the same phenomenon.
3. **"No strategy reliably outperforms Late Concatenation" is a statement about
   accuracy, and `T4.02` is not measuring accuracy.** It is legitimate for a
   balancing method to cost accuracy and still be required here, because the
   requirement is constitutional. So the reference arm can only *inform* the
   verdict; it cannot decide it.
4. **Development sets of 100 and 200 samples**, batch 16/32. Small, and the
   hyperparameter-sensitivity claim is partly a claim about small validation
   sets.

---

### N3 — a correction to a live nomination of my own: if `A4c` ever enters, it must enter as TEMPORALLY-CENTERED SIGReg, because plain SIGReg has a measured defect in exactly our regime

**Source:** *Temporally Centered SIGReg Improves LeWorldModel Representations
for Robot Policy Learning* — [arXiv:2607.26924](https://arxiv.org/abs/2607.26924),
primary category **cs.LG**, v1 2026-07-29 / v3 2026-08-26, Liu, Suo, Jin, Ping,
Iwasawa, Matsuo & Zhu. **Full HTML read.**

**This is aimed at my own week-5 nomination, which the Review accepted.**
Week 5 nominated SIGReg (arXiv:2607.13612) as the **selection criterion** among
the anti-collapse routes for `A4c`, on a theoretical argument — its lead
objection, stated then, was *"there are NO EXPERIMENTS AT ALL"*. **This paper
supplies experiments, and they do not simply vindicate the method: they
identify a defect in it.**

**The verified mechanism [V], quoted and with its formula.** The latent is split
into a temporally persistent component and a centered residual over a window
`W_t` (their Eq. 2): `z̄_t ≜ (1/|W_t|) Σ_{s∈W_t} z_s`, `r_t ≜ z_t − z̄_t`.
Their Monte-Carlo analysis finds the two *"compete for the variance required by
the projected marginal"*, and the prediction objective favours smaller residual
variation, **"biasing variance allocation toward the persistent component at
the expense of the centered residual, suppressing residual variance."** The fix
is to apply SIGReg to the residuals `R = {r_t}` rather than to the whole latent
`Z` — *"this simple change decouples persistent and residual variance
allocation while retaining an effective anti-collapse property."*

**The measured consequence, and it is why this matters to us [V]:** plain
SIGReg gives *"reduced decodability of robot state and dynamics, particularly
gripper dynamics"*, and the fix raises decodability *"for all three quantities
and at all derivative orders"*.

**The numbers [V]**, LIBERO, with error bars:

| suite (10-task) | Raw LeWM (plain SIGReg) | **TC-LeWM** |
|---|---|---|
| Spatial | 62.4 ± 7.5 | **86.0 ± 4.1** |
| Object | 93.9 ± 2.3 | **97.0 ± 0.1** |
| Goal | 53.4 ± 5.3 | **88.4 ± 7.5** |
| Long | 44.7 ± 4.3 | **63.6 ± 6.8** |
| **average** | **63.6 %** | **83.8 %** |

Unified 40-task: 72.6 → **85.4**. Single-task: 26.1 → **50.1**. Window `W = 4`,
`J = 1024` projections, ViT-Tiny encoder, 10k steps (30k at 40 tasks), batch 128.

**Why it lands on us, and it is a regime argument rather than a benchmark one.**
The suppressed quantity is the **temporally centered residual** — the
fast-changing part of the latent. In W0 that is exactly the part that carries
contact events, joint-velocity change and drive movement, and `A4` is a
**latent-prediction** objective whose target is an EMA of its own encoder,
i.e. precisely the kind of objective that can be satisfied by a persistent
component. **And it converges with wk4-N2 from a different direction**: that
nomination's verified claim was that a sense enters the latent only when the
loss asks for it as a TARGET (proprio probe r 0.98 vs 0.08). This one says a
*time-scale* enters the latent only when the regulariser is applied at that
time scale. Same shape, different axis.

**What I nominate.** Not a new arm — an **amendment to a live accepted one**:
if `A4c` (SIGReg/LeJEPA) is ever run, it enters as **temporally-centered**
SIGReg, with plain SIGReg as its own paired comparator. The two are each
other's control, they differ by one line, and the difference is a window
constant. **Cost: zero new parameters, one hyperparameter (`W`), and it does
not increase the arm's forward cost.**

**Why it might LOSE (steelmanned). Five.**
1. **`A4b`/`A4c` have been LIVE and UNRUN for eight sweeps.** Amending an
   unrun nomination is cheap in exactly the way that should make a reader
   suspicious. **The honest position is that this desk has promoted five
   anti-collapse routes and run zero**, and a sixth route would be padding —
   which is why this is an amendment and not a nomination of a new arm. §5 says
   the same thing about the four further anti-collapse papers the front-1
   enumeration returned this week.
2. **Pixels again**, and LIBERO visuomotor manipulation with a ViT-Tiny
   encoder. Our regime has no images at all in W0.
3. **No GPU, no wall-clock, no parameter count, no code** [V, checked].
4. **The temporally-centered residual needs a definition in our regime that the
   paper does not supply.** `W = 4` frames in a video stream is not obviously
   `W = 4` decisions in a life that runs 541.9 s [M]. Choosing `W` for W0 is our
   problem, and a free hyperparameter is what week 3's Optimistic-World-Models
   objection was partly about.
5. **Its baselines are Diffusion Policy and OpenVLA** — models far off this
   box. Beating them is not evidence that anything transfers to 4 ARM cores.

---

## 3. WATCHLIST

Every entry records its arXiv **primary category**.

**New this sweep:**

| item | cat | what it is | what would PROMOTE it |
|---|---|---|---|
| **MINERVA** — *How Small Can a Manipulation Policy Be and Still Solve LIBERO?* ([arXiv:2609.03715](https://arxiv.org/abs/2609.03715), 2026-09-03, Sendai, Matsushima & Iwasawa) | cs.RO | **The small-model end returns something after five empty sweeps.** Measured [V]: **0.54 M parameters**, **95.1 %** average over **2,000 rollouts** on the four standard LIBERO suites, *"only 2.4 points below the reported LeRobot π₀.₅ result"*; 94.6 % on LIBERO-90 across 89 tasks; **saturates near 1 M, collapses below 0.25 M**. **Runs on a laptop CPU, 5–9 ms per chunk, 113× faster than SmolVLA and 1,400× faster than π₀.₅.** | **An RL result, or nothing.** It is **imitation learning from demonstrations**, which is the exact ground week 3 rejected DOOM-1.3M on, and consistency costs nothing here. It is also brittle where it matters to us — LIBERO-Plus perturbations 46–56 %, photometric robustness **near zero**. Recorded because it is the **first paper in eight sweeps whose inference runs on our substrate class with a latency number**, and because its measured collapse threshold (0.25 M) and saturation point (~1 M) bracket our own arms — `ppo-needs` **135,961** is *below* their collapse threshold and `wm-latent` **861,545** is just under their saturation point [M]. **That bracketing is an analogy, not arithmetic** (different task, imitation vs RL), and week 3's rule says so. |
| **2609.21787** — *Compact but Moving: Intervention-Relevant Geometry in Recurrent World Models* ([abs](https://arxiv.org/abs/2609.21787), 2026-09-18, Chen & Liu) | cs.RO | **State-based and recurrent** — structured-GRU and a parameter-matched LSTM, not pixels, which is `A4`'s shape (`nn.GRUCell`). Asks whether a compact intervention structure survives autonomous rollout, and answers that it persists as *"a moving, state-dependent local geometry"* rather than a fixed low-rank subspace. | **Any number at all.** The abstract contains **no metrics, no seeds, no hardware, no parameter counts, no code**, and the result is stated across *"two of three checkpoints"* without those checkpoints being characterised. Its relevance is concrete if it ever gets numbers: it is about whether a diagnostic measured at one state transfers to another, which is the question any `A4` readout inherits. |
| **2608.10145** — *The Evaluation Protocol Determines the Result: An Independent Reproduction of LeWorldModel on TwoRoom* ([abs](https://arxiv.org/abs/2608.10145), 2026-08-10, Joyjeet Singh) | cs.LG | An independent reimplementation, **~$25 of rented compute with all evaluation on one laptop CPU**, **code released** (`joyjeet-singh/tinylab`). Measured [V]: on the authors' own released weights and **fifty identical episodes**, *"changing nothing but how the goal is constructed moves that checkpoint from 84.0 % to 8.0 %"*; the paper's appendix and the repository's config give **14.0 %** and **84.0 %**. Four undocumented conventions were required for convergence at all. | **Nothing — it is recorded, not promoted.** Its two general claims are corroboration of things this project already enforces: that one-step prediction error *"fails to order long-horizon success at all"* across a sevenfold range of prediction error is law 1 (*a loss curve is not learning*) measured in someone else's lab, and the batch-norm layer that inflated validation loss 300× is a `T0.x` scar we have paid. **Corroboration is not news.** It is on the list only because it is the second paper this sweep whose compute is honestly reported and small. |

**Carried and re-examined:**

| item | cat | status |
|---|---|---|
| **2608.29434** — action-conditioned JEPA on point clouds | cs.LG | **Unchanged, and now the weaker of two non-pixel items.** Still abstract-level, still no metric definitions, no numbers, no hardware, no code. N1's ARC-Bench supersedes it as the evidence for the same theme; 2609.21787 supersedes it on the "non-pixel" axis by being genuinely state-based rather than geometric. **Promote on numbers, or drop it next sweep with cause.** |
| **SmallWorlds** ([arXiv:2511.23465](https://arxiv.org/abs/2511.23465)) | cs.LG | Unchanged, fifth sweep. Rollout-horizon deterioration in the fully observable state space — our regime. Still no compute cost, no hardware, no environment size, no code. **Promote on: any statement that a domain runs on CPU.** |
| **ForageWorld** ([arXiv:2506.06981](https://arxiv.org/abs/2506.06981)) | cs.AI | **Kept, and its window has now slipped a second time.** `w1-world-edit-window` has lost four consecutive Sunday FULLs and `D33` exists to ask whether the design is producible at all. Still the closest published existence proof of depleting/diffusing food, pursuing predators and an energy-gated immobilising sleep action. **Design reference only; it is not an arm and never was.** |
| **PRIME** ([arXiv:2607.16858](https://arxiv.org/abs/2607.16858)) | cs.LG | Unchanged. Pseudocount epistemic term, free on a 10×10 grid, undefined on a continuous body state. |
| **IIBalance** ([arXiv:2603.17347](https://arxiv.org/abs/2603.17347)) | cs.MM | **Re-examined on front 2's return and now second-best in its own family.** N2's paper is the stronger statement of the same scepticism — IIBalance argues capacity-based budgets should replace forced equality; 2609.11247 *measures* that the forced-equality family does not beat concatenation. Still classification, still no extractable numbers. |
| **Eywa** ([arXiv:2605.30771](https://arxiv.org/abs/2605.30771)), **ScrubJay-MEM** ([arXiv:2608.04746](https://arxiv.org/abs/2608.04746)), **RARE/RedQA** ([arXiv:2604.19047](https://arxiv.org/abs/2604.19047)), **CoDeR** ([arXiv:2606.13204](https://arxiv.org/abs/2606.13204)), **Argus Eyes** ([arXiv:2602.09616](https://arxiv.org/abs/2602.09616)) | cs.CL / cs.IR | **All front-3 items, carried unexamined this week by design** — front 3 is on its two-week cadence and returns 2026-09-28. Week 7's dispositions stand; nothing here is a fresh read and none of these is claimed to have been re-checked. |
| **POBAX** ([arXiv:2508.00046](https://arxiv.org/abs/2508.00046)) | cs.LG | Unchanged, and still not an instrument we lack. `W1.01` is still unregistered, which is why it is still listed. |
| **Curiosity-Critic** ([arXiv:2604.18701](https://arxiv.org/abs/2604.18701)) | cs.LG | Unchanged. Weaker-evidenced sibling of LPM, already cited by `CURIOSITY_BAKEOFF.md`. |

---

## 4. DISPOSITION OF PRIOR NOMINATIONS

| nomination | entered | status now |
|---|---|---|
| **wk7-N1 · free-embedding critical-`d` probe (2508.21038)** | `ME.11` successor pre-gate | **ACCEPTED, ORDERED FIRST, NOT RUN.** §0(4): no spec, no code, seven days on. Not re-argued and not withdrawn — it is the cheapest live item on the desk (single-digit CPU-minutes) and the reason it was ordered first has not changed. |
| **wk7-N2 · 3M-Progress (2506.00138)** | `CURIOSITY_BAKEOFF` arm | **ACCEPTED AS AN ARM and HELD, with the hold written down** (`f34d366`): `CU.1`–`CU.7` are seven specs with **zero implemented**, and an arm spec for a bakeoff with no arms ages without a referent. **The hold is right and I am not contesting it.** Queued #6 (a third pass for its numbers) is **NOT DONE this sweep** — front 3's absence went to fronts 2 and 5, not to a fourth fetch of a held arm. Carried once; per this desk's deferral rule it is closed or dropped next sweep. |
| **wk7-N3 · ActSWM `Δ_k` (2607.26712)** | merged into the `a4` row | **ACCEPTED and MERGED**, now option (i)'s named form on the `a4-…-computed-nowhere` row. **§2's N1 is a second readout for the same seat and is cheaper than this one** — `Δ_k` needs matched real-environment rollouts to get `z_{t+k}`; the `k`-sweep does not. Both are priced off the same 14.40 core-h training run. |
| **wk7 · §6 (A4's mandatory diagnostic computed nowhere)** | `REVIEW_QUEUE.md` row | **ROUTED within an hour, OPEN, DUE 09-18, now +3 days.** The desk's recorded leaning is **(i)+(iii) together, never (iii) alone** — build the readout *and* amend `LEARNING_CORE.md` §5.4 to say the guard was never built and the seat was awarded without it. **§6 below is about the same seat from a different direction and does not repeat this one.** |
| **wk7 · §6b (abstention certified below its own bar)** | `me1-similarity-floor-never-abstains` | **DISCHARGED IN PART — three of five repaired, two structurally capped.** §0(3) has the table. |
| **wk6-N3 · IAF pathway-decoding control** | `t402` row, DUE **09-22** | **STILL WAITING, 16 days.** Carried unchanged and not re-argued. **§2's N2 is its complement, not its replacement:** N2 catches the family being worthless, wk6-N3 catches a winning arm degrading the pathway it balanced. |
| **wk6-N2 · MULTIBENCH++ redundancy pre-gate** | `ub10` | **Unchanged.** Stands as a pre-registration on the hardened battery's next reading. |
| **wk5-N1 · SIGReg vs VICReg (2607.13612)** | `LEARNING_CORE` §5.4 selection criterion | **AMENDED BY §2's N3, AND THE AMENDMENT IS AGAINST MYSELF.** Week 5 promoted SIGReg on a theorem; a paper with experiments now measures that plain SIGReg suppresses temporally-centered residual variance and reduces state/dynamics decodability. The nomination survives in its temporally-centered form. Its **re-pricing from zero still stands** (week 7 §6: the curves it was to be read on do not exist). |
| **wk5-N2 · prosociality by coupling (2604.10760)** | `NE.07` | **Unchanged.** Arm DEFERRED, shuffled-partner CONTROL accepted. `NE.07` and `NE.02` have still never run. |
| **wk4-N1 · spectral-radius constraint (2607.19719)** | `A4` variant | **ACCEPTED, narrowed. Unrun.** |
| **wk4-N2 · PSG-JEPA (2608.06799)** | `A4` ×2 | **ACCEPTED as two arms. Unrun**, sequenced behind `D9`'s PARK. **Converges with §2's N3** — see there. |
| **wk4-N3 · infant motor noise** | `W0.DIAG` | **RUN AND PASSED.** Nothing further owed. Still the only field-watch nomination in eight weeks to become a number. |
| wk1 · anti-collapse regularisers → `A4b`/`A4c` | `A4` variants | **LIVE, UNRUN — eighth sweep.** §2's N3 amends `A4c`'s form; it does not make the arm any more run than it was. §5 declines to add a fifth route. |
| wk1 · certificate-gated identifiability → `UB.11` pre-gate | `UB.11` | **LIVE, unrun.** Still the only route to `UB.11`'s certificate. |
| wk1 · interoceptive precision (2608.04232) | `NE` §2.4b | **LIVE, unrun.** Still the cheapest item on the desk with released code, and it has been for eight weeks. |
| wk1 · entity-collision protocol (2605.29630) | `MR` §2 | **LIVE, unrun**, and subsumed in spirit by wk7-N1, which is also unrun. |
| wk2 · the whiff clock → `SM.02` | `SM.02` | **HELD, correctly — `SM.02` is PARKED.** |
| wk2 · RPE-prioritised replay → `NE.05` | `NE.05` | **LIVE, unrun.** |
| wk3 · CIG (2605.20878) → `A3` | `A3` | **Remains DEMOTED** (`wm-efe` t = 2.05). |
| wk3 · Optimistic World Models (2602.10044) → `A2` | `A2` | **Remains DEMOTED, hard** (`dreamer-xs` t = −0.94). |

---

## 5. NO-ACTION — fronts where nothing cleared the bar

**FRONT 4 · CURIOSITY & OPEN-ENDEDNESS — NOTHING, AND THE ENUMERATION SAYS THE
SAME THING FOR THE SIXTH TIME.** A 40-entry enumeration on intrinsic
motivation / intrinsic reward / open-ended / autotelic × exploration or agent
returned: LLM agent harnesses (ArenaFlow, Stellar Colosseum, EvoRS, AutoKD),
HCI and visual-analytics papers, ads ranking, supply-chain analytics, and
hardware-interference identification. **Not one has a body under homeostatic
drive; not one evaluates an intrinsic reward against a random or noise
baseline.** The nearest three are `2609.17325` (*Intrinsic Motivation in RL: A
Research Agenda*, cs.AI — an agenda), `2609.05650` (*Endogenous Exploration
with Intrinsic Curiosity*, cs.LG) and `2609.07575` (*Efficient Exploration Is
Enough*, cs.LG); none was fetched and none is claimed to have been. **A
WebSearch pass outside arXiv returned the homeostatic-RL literature this
project already cites (HRRL, drive-reduction reward) at 2024–2025 dates —
out of window, and the family `NEEDS_AND_DEATH` is built on.** Week 7's
nomination on this front (3M-Progress) remains the front's only live arm and it
is HELD behind seven unimplemented `CU` specs, which is a better description of
why front 4 is empty than anything the literature did.

**FRONT 5 · WORLDS & EMBODIMENT — NOTHING, AND THE SPECS THAT NEEDED SOMETHING
ARE STILL UNREGISTERED.** A 40-entry enumeration on survival / homeostatic /
foraging / embodied agent × simulation, benchmark, MuJoCo returned navigation
benchmarks (EvoNav-Bench, 360CityArena, DreamFly), LLM-agent scaffolding,
3D-scene-graph memory, swarm robotics, a Dark Souls boss environment, and —
because "survival" is a word other fields own, week 2's lesson recurring —
**six astrophysics and condensed-matter papers**. `2608.26947` (*4DSynth:
Controllable Procedural World Synthesis for Dynamic Embodied Simulation*,
cs.RO) and `2609.19801` (*DeliveryGym*, cs.LG) are the only two on the right
axis and neither has needs, death or a fidelity claim. **`W1.01`/`W1.03`/
`W1.04` remain NOT REGISTERED** and `w1-world-edit-window` has now lost four
Sunday FULLs; week 6 established there is no cheap published
environment-discriminability score and nothing this week changes it. **These
instruments are still ours to build.**

**FRONT 1 · LEARNING CORES — A NOMINATION AND A DELIBERATE REFUSAL, AND THE
REFUSAL IS THE PART WORTH READING.** The enumeration returned **four further
anti-collapse / latent-structuring routes for `A4`** in window: `2608.17542`
(*No Gaussian Required: Contrastive Inverse Dynamics for JEPA*), `2608.20065`
(*Orthogonal JEPA: Factorized Predictive States*), `2608.16287` (*SCALE:
State-Calibrated Latent Embeddings*), `2609.04264` (*Spectral-Target Physical
Latent Structuring*). **I am nominating none of them.** This desk has promoted
**five** anti-collapse routes (`A4b` inverse-dynamics, `A4c` SIGReg/LeJEPA, the
spectral-radius constraint, PSG-JEPA ×2) and **run zero**, across eight sweeps.
A sixth route would be padding, and padding on this front specifically is how a
seat ends up with six unrun challengers and no guard. **The honest thing a
scout can do with four more papers on a family that already has five unrun
members is say so and stop.**

**SMALL-MODEL END — SOMETHING IN WINDOW FOR THE FIRST TIME IN SIX SWEEPS, AND
IT IS STILL NOT AN ARM.** MINERVA (§3) is 0.54 M parameters on a laptop CPU at
5–9 ms, which is the substrate class this desk has been asking the literature
for since week 3. It is **imitation learning from demonstrations**, which is
the exact ground DOOM-1.3M was rejected on, and applying the same standard to a
result I like costs nothing and is the whole point of having the standard.
**Recorded, not nominated.**

**BIOLOGY-AS-ORACLE — the standing `q-bio.NC` search ran and returned no arm,
but it returned a better field than `cs.*` did.** Of 40 entries, the ones on
our axis were `2609.02243` (*Mus siliconus*: a neuro-musculoskeletal digital
twin of the mouse integrating neural dynamics, biomechanics and **tactile**
input), `2606.17456` (*Embodiment Shapes Rolling Behavior in a Multimodal
Infant Model*), `2604.27583` (infant first-person sensorimotor experience by
motion retargeting) and `2607.20306` (*State-Dependent Observation Noise
Reintroduces Epistemic Value in Linear-Gaussian Active Inference* — which
bears on `A3`'s epistemic term, an arm that read t = 2.05 and is DEMOTED).
**None was fetched, none is nominated, and the category is now doing what week
7 said it would**: `cs.*` returns papers that *cite* biology, `q-bio.NC`
returns papers that *do* it. The standing search is worth its one slot and
stays.

**FRONT 2 · FUSION — §2's N2 is a CONTROL, and the four-sweep refusal to
nominate a balancing METHOD is unchanged and now has the field's own
measurement behind it.** The 40-entry enumeration returned ~20 in-window
imbalance papers (CAT-GS, SAGG, ShapKO, Pareto LoRA, PDMP, MiMIC, and a
mixture-of-experts survey) and **every one is supervised classification,
recommendation, federated learning or medical EHR** — the **third consecutive
year**, and now the fourth consecutive sweep this desk has said it. **Not one
paper in the family has a world-model objective and five heterogeneous senses.**
The Goodhart objection stands unchanged and is now joined by a measured one.

---

## 6. A FINDING IN OUR OWN ARTIFACTS — `LC.03`'s five controls never switch off the term `A4` is named for, so the seated arm's number cannot distinguish its world model from its actor-critic

Week 3's rule: a scout reading our own ledger has no abstract to doubt, so it
must carry the arithmetic. **This is a different hole in the same seat as week
7's §6, and it is not a restatement of it** — week 7 found that `A4`'s declared
*collapse* diagnostic does not exist. This is about what its *controls* do and
do not remove. Reproducible in four commands.

**What `A4` is, in the code's own words** (`experiments/cores.py:15`):

> *"A4 `wm-latent` — A2 with the decoder deleted; latent prediction vs an EMA
> target encoder."*

So the machinery that makes `A4` `A4`, rather than a plain RSSM actor-critic, is
the `latent_pred` head and the objective that trains it. **Measured at HEAD**
(`build_arm('wm-latent')`, `n_params`) [M] — and the row's 861,545 reproduces
exactly, so the artifact and the code agree:

```
wm-latent total            861,545
  latent_pred head         149,312  (17.3 %)   <- the only machinery A4 is named for
  actor + critic           135,049  (15.7 %)
```

**How the three losses combine** (`experiments/cores.py:542`, `lc_update`),
verbatim:

```python
l_bind, _ = core.binding_loss(batch, dropped)
z = core.shared_state(batch, dropped)
l_v  = F.mse_loss(core.critic(z), targets["value"])
a    = core.act(batch, z)
l_pi = (((a - targets["action"]) ** 2).mean(-1) * targets["advantage"]).mean()
loss = l_bind + VALUE_COEF * l_v + l_pi
```

**`l_v` and `l_pi` push gradient into `shared_state` — the same encoder, GRU and
posterior that `l_bind` trains.** The RSSM is therefore shaped by the policy and
value terms as well as by the world-model term, and the three are additive.

**What `LC.03`'s controls actually are**, from the registry's own text — five,
each named: **(a) statue** (do nothing), **(b) randrew** (random stationary
reward projection), **(c) frozen** (the optimiser never steps), **(d)
wiped-store** (weights, optimiser and replay reinitialised at every death),
**(e) darkroom** (rewarded for minimising predicted observation entropy).

**Not one of them sets `l_bind = 0` while the rest of the arm keeps learning.**
(a) changes the policy to nothing; (b) and (e) change the *reward*; (c) stops
*all* learning; (d) reinitialises the *whole* model. **Every must-fail control
on this seat breaks the model everywhere at once, and none of them breaks only
the part the seat is named for.**

**And the arm that would exist in the repo.** `CONTROL_ARMS` already contains
`unbound` — *"per-modality encoders, concat, NO cross-modal loss term"*
(`cores.py:495`) — but it is a **different architecture**, and the committed
`LC.03` result carries metrics under exactly five arm prefixes [M]:

```
$ python -c "…json.load(…)['results']['LC.03']['metrics']…"
LC.03 metric prefixes: ['dreamer-xs', 'ppo-lp', 'ppo-needs', 'wm-efe', 'wm-latent']
unbound present? False        (257 metrics)
```

**`unbound` has never been run on the survival ruler.** Its only appearance is
in `LC.01` as a gradient-plumbing control for U2, **at init**.

**THE CONSEQUENCE, stated as a fact and not as an instruction.** `A4` holds the
Learning-core seat on `lg_margin_null` t = 4.64 and `lg_margin_twin` t = 4.00
[C, recomputed at HEAD from the committed row]. **Both margins are consistent
with a reading in which the latent-prediction objective contributed nothing and
an RSSM actor-critic produced the whole number** — because the untrained twin
has *everything* switched off and the frozen control has *everything* switched
off, so neither separates the two hypotheses. If that reading is the true one,
**149,312 parameters — 17.3 % of the seated arm — are dead weight**, which is
Tier 3's entire business (*"every component ablated; dead weight deleted"*) and
`LC.06`'s (*"the simplicity budget is enforced, not promised"*).

**Why this is a section and not a footnote, and it is the same shape as this
week's lead nomination.** ARC-Bench's measured claim is that *closed-loop
success rates systematically overstate the rankability of the latent* — a
capability number credited to a latent objective that the number never
isolates. **That is this, in another lab, with p = 1.5 × 10⁻⁹.** I did not go
looking for the local instance; I went looking for what N1 would need and found
that our own control set has the gap the paper is about.

**The cheap version, offered as an observation.** An `A4` variant with
`l_bind` dropped is a one-line change to a core that already builds, and it
would be the paired comparator this seat has never had. It is **not free** —
same 14.40 core-h per 3 seeds as everything else on this seat, since no weights
exist on disk [M] — but if the `a4-…-computed-nowhere` row takes option (i) and
a training run happens, this arm and N1's `k`-sweep are both marginal on it.

**Nothing is decided here.** Whether `A4`'s seat needs this comparator, whether
`LC.03`'s control set should have had it, and whether any of it is worth
14.40 core-h are the builder's and the Review's. **I report that five controls
exist and none of them removes the mechanism.**

### 6b — a second, smaller one, free, and it is about the instrument built last week to read this page: BOTH findings on this page report ROUTED, and all five routings are stock English prose or the row's own slug

`experiments/fieldwatch.py` shipped seven days ago to catch a finding of mine
that reached no desk (§0(2)). **On the second page it has ever read — this one
— its quotation channel reports both findings as already owned, and not one of
those routings is real.** Measured at HEAD [M], `_shingles(finding) &
_shingles(queue_row)` over every row in `REVIEW_QUEUE.md`:

| finding | row it reports as owner | the overlapping 6-gram(s) |
|---|---|---|
| **§6** | `pass-certificates-are-not-re-evaluated-when-a-dependency-falls` | *"it is the same shape as"* |
| **§6** | `sm03-heldout-split-saturated` | *"is the same shape as this"* |
| **§6b** | `lg03-blind-twin-cannot-prove-itself-alive` | *"why this is not a one"* |
| **§6b** | `t108-noise-floor-is-quoted-by-nobody` | *"so it is not mistaken for"* |
| **§6b** | `a4-…-is-declared-and-computed-nowhere` | *"a4 mandatory collapse diagnostic is declared"*, *"collapse diagnostic is declared and computed"*, *"diagnostic is declared and computed nowhere"*, *"mandatory collapse diagnostic is declared and"* |

**§6's true state is UNROUTED** — it is a fresh hole in `A4`'s control set and no
row owns it. **§6b's true state is UNROUTED.** The instrument says otherwise
five times.

**Two distinct mechanisms, and both are general rather than particular to my
prose.**

1. **STOCK CONNECTIVE PHRASES.** Four of the five collisions are ordinary
   English — *"it is the same shape as"*, *"why this is not a one"*, *"so it is
   not mistaken for"*. This page and `REVIEW_QUEUE.md` are written by the same
   kind of agent in the same house style, arguing in the same register.
   **Two desks that write alike will share six-grams that carry no content**,
   and a threshold of **one** shingle cannot tell that from a quotation.
2. **THE ROW SLUG IS INSIDE THE CHUNK.** `_queue_chunks` starts each chunk at
   `ROUTED: <slug> |`, so a hyphenated slug of six or more words tokenises into
   shingles of its own. **Any finding that names a row by its slug is therefore
   reported as "quoted by" that row** — which is what the four a4 hits above
   are. Naming the row you are *distinguishing yourself from* marks you as
   owned by it.

**I TRIED REWORDING AND IT MADE THINGS WORSE, WHICH IS THE POINT.** The first
draft of §6 collided with the a4 row on one shingle (*"the committed lc 03 row
records"* — a phrase I used because week 7's §6 used it and the Review's row
quoted week 7's page back verbatim). I changed that one sentence. **The
collision did not disappear; it moved to two different and completely unrelated
rows.** That is the finding: at a one-shingle threshold, a page of this length
against 68 rows of that length will collide with *something*, and a scout
cannot write its way out. **I have stopped rewording, and both findings are
left reading a wrong ROUTED rather than edited until the counter is pretty** —
because a scout quietly tuning prose to move a counter is indistinguishable
from the thing this reader exists to prevent.

**Why this matters more than its size.** The module is **reporting-only and
unfloored** by the 96th audit's explicit instruction, *"do not floor a counter
whose false-positive rate has not been measured and written down"*. **This is
that measurement, and the direction is the dangerous one:** the predicted
false-positive shape was a finding *discharged in code* being called unrouted;
the measured one is a finding *with no owner at all* being called **ROUTED**.
A counter that reads 0 UNROUTED because everything collides is the 96th audit's
scar with a green light on top — the exact failure the module was built to end,
wearing the module's own badge.

**What I am NOT claiming.** Not that the shingle rule is wrong — it is imported
from `decisions.owner_asks` so the two readers cannot drift apart, and a
threshold that is too loose here may be right there; I have not measured it
there and I do not assert anything about it. Not that the citation channel is
affected: `_cites` is week-anchored and slug-independent, and **week 7's two
findings both routed through it correctly**, which is why this page still says
the instrument worked. **The claim is exactly this: the quotation channel, at
n = 5 on the second page it has read, is 0 for 5.** Candidate closures — a
minimum-overlap count, stripping the `ROUTED:` header line from the chunk,
or subtracting shingles that also occur in the previous sweep's page — are the
builder's call and not mine.

---

## 7. What this report does NOT claim

- **No arm here has been run.** Every number in §2 and §3 is someone else's
  measurement on someone else's hardware. The §0(3) Clopper–Pearson table, the
  §6 parameter counts, greps and ledger reads, the `needs_rise` and
  `lg_margin` t-statistics, and the weights-on-disk search are **ours**, marked
  **[C]/[M]**, with the commands shown.
- **N1's lead objection is from our own ledger and it is serious.** W0's
  measured passivity inversion pushes `life_gain(k)` the opposite way from
  ARC-Bench's prediction. **I am not claiming the `k`-sweep works; I am claiming
  it is the cheapest readout on that seat and that it needs `needs_rise`, not
  `life_gain`, to be readable at all.** And `needs_rise` on the seated arm reads
  t = 1.95 [C], which I state rather than hide because it is the quantity the
  repair leans on.
- **ARC-Bench audits FROZEN checkpoints and a latent-distance planner. `A4` is
  plastic and actor-critic.** Only the *masking* half of that paper is
  nominated. The *rankability* half does not transfer and I say so in N1's
  objection 2 rather than in a footnote.
- **N3 is an amendment to my own live nomination, not a new arm** — and its
  context is that this desk has promoted five anti-collapse routes and run zero.
  §5 declines to add a sixth.
- **MINERVA is not nominated**, on the same ground week 3 rejected DOOM-1.3M.
  The standard is applied to a result I like at the same cost as to one I did
  not.
- **§6 is about `A4`'s CONTROLS; week 7's §6 was about its declared collapse
  DIAGNOSTIC.** They are two holes in one seat and I have not merged them to
  make either look bigger.
- **§6b is a measured false-positive rate on ONE channel of ONE reader at
  n = 5, on the second page that reader has seen.** It is not a claim that the
  shingle rule is wrong, not a claim about `decisions.owner_asks` (which I did
  not measure), and not a request to floor or unfloor anything. The citation
  channel is unaffected and week 7's two findings routed through it correctly.
- **Both of this page's findings display a WRONG `ROUTED` in `run status`
  today, and I left them that way on purpose.** §6b says why: the one rewording
  I tried moved the collision to two unrelated rows instead of removing it.
- **Front 3 was not swept**, front 4 and front 5 were enumerated but not
  fetched, and queued #6 (3M-Progress's numbers, third pass) was **not done**.
  All three are stated in §1 and §4 rather than absorbed.
- **Verification is uneven and marked:** N1 — full HTML, tables and p-values
  quoted, **no hardware/wall-clock/params, code not released**; N2 — full HTML,
  method names and hardware quoted; N3 — full HTML, formula and table quoted,
  **no hardware/params/code**; MINERVA, 2608.10145, 2609.21787 — abstract level;
  the ~20 front-2 and 40 front-4/5 enumeration entries — **title and category
  only**, listed to establish a *family*, and no claim of any of theirs is
  relied on.
- **The `ME` denominators and the `t402`/`a4` row states in §0 are the builder's
  and the Review's records, not mine.** I read them out of the queue, the
  ledger and `run status`; I did not run them.

---

## 8. Queued for next sweep (**not before ~2026-09-28**)

1. **FRONT 3 RETURNS** on its two-week cadence, and it has a question rather
   than a survey: **did wk7-N1's free-embedding probe run, and what was the
   critical `d`?** §0(4) records it as NOT RUN after being ordered first. If it
   is still unrun on 09-28 that is 14 days on the cheapest item on the desk, and
   the right thing for a scout to do is say the number.
2. **Did the `a4-…-computed-nowhere` row rule — WEEK 7's §6?** Both
   are about the same seat. The row was due 09-18 and is +3 today. **If it
   ruled, §2's N1 and wk7-N3's `Δ_k` are both priced off whatever run it
   ordered; if it did not, the seat's guard is unarmed for a fourth week.**
3. **`t402` fell due 2026-09-22 — the day after this sweep.** Did the balancing
   bakeoff get designed, and does it carry a do-nothing reference arm (§2 N2)
   and wk6-N3's pathway-decoding control? **If `t402` is still open on 09-28
   that is 23 days on a `FAIL-UNOWNED` repair.**
4. **Did §6 and §6b reach a desk, and does `run status` still show a false
   `ROUTED` for them?** They are the test case: if a row is opened for either
   one it will route by **citation** (*"field watch wk8 §6"*), which is the
   channel that works — and the quotation false positive in §6b will still be
   sitting underneath it, uncounted, unless someone looks. **If the quotation
   channel is unchanged next sweep, that is the second measurement of the same
   false positive and it should stop being called n = 1.**
5. **3M-Progress's numbers — THIRD AND LAST PASS**, per this desk's deferral
   rule. PDF/appendix figures or the released repo's evaluation scripts
   (`neuroagents-lab/autonomous_zebrafish`). **If no number is found, it leaves
   §2 for the watchlist**, and the HELD status the Review gave it makes that
   costless.
6. **`q-bio.NC` on the biology front, standing** (wk7 §7, upheld — §5 says what
   it bought this week). **Plus one addition of the same kind:** the front-5
   enumeration returned six astrophysics papers on the word "survival", so the
   primary-category convention (week 2) is now also a *search* discipline and
   not only a *recording* one.
7. **ARC-Bench's code**, if released. The authors say release follows the
   preprint going public. A released protocol would cut N1's reimplementation
   risk, which is currently the largest unpriced part of it.
8. **2608.29434 — promote on numbers or DROP WITH CAUSE.** Carried three
   sweeps at abstract level; two better-evidenced items on the same theme now
   supersede it. Week 3's deferral rule applies.
9. **NOT queued, deliberately:** conference proceedings (dropped wk4);
   2607.22430, Simulus, UED-as-a-cheap-score (all closed wk6); LIMIT's
   *conclusion* (closed wk7, instrument retained); **and the anti-collapse
   family — four more routes appeared this week and §5 declines all four until
   one of the five already accepted has been run.**
