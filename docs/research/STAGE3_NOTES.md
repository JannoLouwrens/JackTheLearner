# Stage 3 notes — memory key, smell, diary retrieval (notes only, no runs)

Written 2026-11-19 (ORDERS "Next up" item 7). **Nothing here is a rule, a
seat or a result.** Each note says what the record and our own lab already
hold, what stays open, and the smallest pre-registerable test that a stage-3
worker would freeze *before* running. Sources: PUBLISHED_PRIORS §6, §12, §14;
LAB `experiments/find_himself.py` (rule `fe459803`, follow-on `2a6ffa69`);
journal 2026-11-17 23:07. No new reading was done.

## 1. A turn-free memory key over the eye's row

**What we hold (lab, 11-17, not the record).** Finding himself after
rebirth by nearest remembered look: body frame, 128 rays, *turning the look
through all 128 azimuths* → place HIT .933 (16 rays .795; chance .021;
scrambled .06–.12). The follow-on asked for a key that needs no turning
(one comparison per memory, which is what a learned store does): the
amplitude spectrum |rfft| of the row over azimuth (65 × 2 numbers, invariant
to any head turn). It hit .845 vs .932 with turning — **but VOID**: the
scrambled-eye control read .157 > its .121 line. Post-hoc cause: the look's
mean alone recalls .31; with the mean (DC bin) dropped, control .03 and key
.795. That drop was decided after the data, so .795 is a pilot reading, not
a result.

**Record.** Rotation-invariant ring descriptors re-localise lidar robots
(Scan Context — cited from memory in the lab docstring, source not opened).
Nothing for a from-scratch agent. §6: retrieval that worked was always by
*learned* similarity (MERLIN, HCAM, NEC) — a hand-made spectrum key is a
scaffold for measurement, not a part of him (philosophy 2).

**Open, and the test a stage-3 worker would freeze.**
- Re-register the DC-dropped spectrum key on fresh worlds (not 700–724),
  controls unchanged (tag ≤ chance+.05, scrambled ≤ chance+.10), bar
  unchanged from `2a6ffa69` (HIT ≥ .50, HIT − chance ≥ .25 on ≥ 4/5,
  within .10 of turning). Box minutes.
- The real question for him: does a *learned* key (the brain's own latent,
  §6 "keys = the world model's latent state") reach the turning ceiling?
  Arms: latent key, spectrum key (reference, scored), turning search
  (ceiling, scored-and-ineligible — it is not something a store does).
  Only after stage 2 picks the brain, since the latent is that brain's.

## 2. Smell, as §14 says

**Our three smell failures used the one setup the record says cannot
learn** (memoryless, raw 12 numbers, eat-only, 13k–52k decisions; ruling
36). Every published odour follower had memory, an encoding of the cue,
shaping or a start-distance ramp, and ≥ 500k decisions.

**Order a stage-3 worker would take (each a §14 line, nothing new):**
1. Cheap known-answer, box minutes: the scripted nose-follower on the new
   encoding (log per nostril, (L−R)/(L+R) × gain, d(log mean)/dt) must
   still reach the perfect-map ceiling (.375 today). If it does not, the
   encoding lost the cue — stop there.
2. Encoding B vs raw, same 0.16 m nose, on the chosen brain *with memory*,
   start-distance dial, ≥ 500k decisions, 5 seeds, **shuffled-nose control
   that must fail**. Predicted: B out-eats shuffled, raw does not.
3. Smell added beside the `row128` eye: predicted no gain in the open
   (eye HIT .909 at d0), gain for hidden food — `SM.02`'s claim. Needs a
   hidden-food world variant; none exists yet (not built in this note).
4. Nose span 0.16 vs 0.5 m only if (2) passes; O2 puffs only with a wind
   sense (no precedent without one). END-FIT: a 1 m nose (OdorSpaces) is not
   a human head — scored-and-ineligible if raced.

**Cost:** ~1 Kaggle CPU kernel per arm at 500k (§14). A miss at 500k is
weak (the record's first sign is ≥ 500k); say so in any rule.

## 3. Diary retrieval keyed on learned state (§6; ruling 20)

**Why it is stage 3.** ME.11/.B/.C/.D and T2.10 were FAIL-DISPOSED to
ruling 20: our diary matches words, and scores 0 on reworded questions —
the record predicts exactly that (§6: "retrieval is always by learned
similarity, never by word matching").

**The design to take as-is (§6):** keys = the world model's latent state,
values = what happened (return, need change); k nearest neighbours,
inverse-distance kernel (NEC: ≤ 5×10^5 rows, 50 neighbours); kept at death;
re-encode keys during sleep (inf: keys drift as the model learns).

**Expectation from the record:** NEC-style gain is early and fades (16.7%
vs −0.7% at 1M frames, overtaken at 40M); event recall worked only with
prediction-trained writes at ~10^9 steps. So at our budgets the claim to
test is the early gain, not recall.

**Test a stage-3 worker would freeze (§6 "last 10%", the third bullet):**
food re-drawn every K lives; diary kept / wiped / **swapped with another
seed's diary (the control that must fail)**; probe at the start of each
life. ~240 kernel-hours as priced in §6 — too big for one slot; the first
pilot is one K, 3 seeds, said to be a pilot. Predicted: no effect if food
never moves; kept diary helps earliest if it moves.

**Shared dependency of all three:** the stage-2 brain. None of these runs
before stage 2 names it (PLAN: stage 3 grows him on *that* brain).
