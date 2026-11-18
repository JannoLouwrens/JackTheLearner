# The prediction for stage 2, written down from the record

PLAN stage 1's last line: "The prediction for stage 2, written down from the
record." Written 2026-11-18 (JACK, ORDERS "Next up" item 1). **Nothing new was
read or run for this page.** Every number is copied from
`docs/research/PUBLISHED_PRIORS.md` (§n = its section) or from a measured row in
this repo, and says which. `(inf)` = the priors page marked it an inference, not
a published result. This page is what stage 2's rule is checked against; it is
not that rule.

## 1. What stage 2 is (PLAN)

A (Dreamer-style world model) against B (replay actor-critic, SAC family) in his
own world, 1,000,000 decisions each, run count set by a 20-run pilot; plus the
battery (memory, exploration, delayed reward, an added sense).

## 2. Decisions to first reliable eating

| brain | first meals | reliable eating | source | strength |
|---|---|---|---|---|
| B | improvement by 10k–50k | 100k–200k | §1 candidate B, from Banana DQN ~138k (community), lidar Goal SAC/TD3 ~100k | (inf) — no published world with our movement drain |
| A | 10^4–10^5 | 10^5–10^6 | §1 candidate A; DayDreamer ~57k–72k on robots | (inf) — and our net is 10–100× below any published Dreamer score (§1, §2: no `size1m` score anywhere) |
| PPO (reference) | — | 0.8–2M | §1 "Already known" | same ray/lidar tasks, no movement drain |

**The caveat that may dominate both rows (§13):** every published eater lived
where moving costs nothing extra. Ours costs 2–3× basal; "no from-scratch
learner is on record eating reliably under a 2–3× movement drain." In energy,
random foraging does not pay at d 0.75 (needs 1.3× random skill; 5× at d 1), so
stillness is the body's correct answer until he forages better than random.
Prediction from §13: under today's world and reward, **both A and B slide to
stillness** unless one of the record's cures is in (cost ramp tied to d; pay on
intake only; longer horizon). J11's pilot is the measurement of which cure; its
reading, not this page, sets stage 2's world. If J11 finds no setup where B eats,
the 100k–200k row above is void for his world, and stage 2 is premature.

**Budget fit:** 1M decisions is ≥ 5× B's predicted reliable point and at the top
of A's band. The record therefore predicts B reaches reliable eating inside the
budget and A *may* (inf). A "tie at 1M" is the expected outcome on the simple
body (PLAN: "A is close behind on today's simple body").

**Seeds:** a one-SD difference takes ~20 seeds; sparse tasks have seed SDs of
300–390 on 0–1,000 (§1, §9). Fewer than 5 seeds per arm cannot tell correct from
broken.

## 3. What each battery task should show (§9 table; A / B references)

These are the published references at the published budgets. **A's numbers come
from 10M–200M models at GPU budgets (§9)**: our small A is predicted at or below
them; read it relative to B, not to the table.

| faculty | task (§9 row) | A | B | predicted ordering |
|---|---|---|---|---|
| basics (harness) | bsuite catch / cartpole (1) | 0.96 / 0.93 | DQN 0.92 / 0.85 | tie; random 0.00 / 0.04 must fail |
| exploration | deep_sea (3) | 0.00 | 0.00 | **both fail** (§2: Dreamer deep_sea 0.00 even at 200M) — no separation expected |
| exploration | mountain_car (3) | 0.92 | 0.93 | tie; PPO 0.10, random 0.10 |
| delayed reward | umbrella_length / discounting_chain (4) | 0.78 / 0.40 | 0.39 / 0.25 | **A ahead** |
| memory | memory_len / memory_size (5) | 0.65 / 0.59 | 0.04 / 0.00 | **A ahead, the largest gap on record**; memoryless B fails above length 1 (signature, §9) |
| distraction | umbrella_distract (6) | 0.74 | 0.30 | A ahead |
| continuous control | DMC cartpole swingup at 100k (7) | 478 ± 20 (official file; §9 flags 528 ± 36 — recompute) | SAC 835 ± 8 | **B ahead at 100k**; A catches up by 200k–500k (778 ± 19 / 853 ± 13, §2) |
| sparse continuous | swingup_sparse / cup catch at 500k (8) | 504 ± 393 / 966 ± 9 | SAC 531 ± 376 / 978 ± 0 | tie inside seed noise |
| an added sense | none (§9 "last 10%", §12) | no number | no number | **(inf) A catches up (it reconstructs the sense); SAC under-uses it, more so when added late** (§12) |

**Where the record says A should pull ahead:** memory, delayed reward,
distraction, and (inference only) the added sense. **Where B should lead:**
sample efficiency on dense continuous control at ≤ 100k, and cost (A ~2,100 min
vs 15 for tuned PPO at 1M on Craftax-classic, §1). **Where neither separates:**
deep_sea, mountain_car, sparse continuous at 500k.

## 4. The decision the record predicts

PLAN's rule shape: a brain must pass its known answers, eat within budget, and
show its predicted advantage. Predicted outcome from the record: B eats within
1M (conditional on J11's cure); A eats within 1M with lower confidence; the
eating race ties or favours B on sample count; the battery separates them on
memory (0.65 vs 0.04) and delayed reward (0.78 vs 0.39) in A's favour; the added
sense is the open question the record cannot answer (E8, §1). **If the record is
right, stage 2 seats A on the battery and the added sense, not on eating** —
which is why the battery and the sense must be in the rule before the runs.

## 5. Ways this prediction can be wrong (what would surprise the record)

1. Neither eats in 1M under the movement drain (§13's open risk) — the most
   likely surprise; J11 tests it first.
2. Small A (`size1m`-class) fails its own known answers (§2: no score below
   8M/12M; "naively bigger" is not the cure, shrinking is the risk: 12M needed
   38M Crafter steps for what 400M reached at 7M).
3. A cannot learn from death with the v1 continuous actor (§2: InvertedPendulum
   stayed at return 4) — the REINFORCE actor is the published fix.
4. Our home-made AWR arms are not B: none reproduced a row (§15; J3), so they
   are not the stage-2 control.

**Not verified:** every row inherits its source section's "Not verified" list;
in particular the B eating band rests on a community Banana report and an
inferred unit conversion (§1), and the A battery numbers mix model sizes (§9).
