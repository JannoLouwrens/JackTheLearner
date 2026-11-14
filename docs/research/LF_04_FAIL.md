# LF.04 FAIL (att 2, 2026-11-14, `bc631a8`): why, and what must change

**Verdict.** Kaggle CPU, 6 kernels, head `f90881f`. Learner's late third vs its frozen
twin: +4 / −127 / −230 s on seeds 100/101/102, against MARGIN_S 272.8. Learner vs its own
early third: +49 / +29 / −176. Both controls held: twinB 0/3, shuffled 0/3. The statue
(zero action) lives 599.9 s.

| seed | early L | late L | early twin | late twin | late meals L / twin |
|---|---|---|---|---|---|
| 100 | 517 | 566 | 397 | 562 | 0 / 0 |
| 101 | 409 | 438 | 408 | 565 | 3 / 0 |
| 102 | 514 | 339 | 386 | 569 | 0 / 0 |

**1. The venue measures how little energy he burns unless he eats.** Every one of the 120
claim lives died of `energy`. Late meals were 0, 3 and 0 across five lives. With no food
a life is at most the statue's 600 s (BASAL_B = 1/600 s⁻¹). The frozen twin's late lives
are already 562–569 s, because its exploration std decays toward stillness (the twin
exists to carry exactly this). So the most any non-eating arm can beat the twin by is
**31–38 s, which is 0.11–0.14 of the margin.** Without foraging the gate cannot fire.

**2. What a pass needs** (from `drives.py` constants; one floor food is ν = 0.08, which is
48 s of basal life, and the apple is 0.5, which is 300 s). The late mean must be at least
late twin + 272.8, about **838 s**. At drain f × basal, a life lasts
600·(1 + 0.08k)/f, so k ≥ (1.40·f − 1)/0.08 floor meals per life:
5.0 at f = 1 (still between meals), 13.8 at f = 1.5, 22.5 at f = 2. Alternatively,
**one apple per late life** at f ≤ 1.07. The world can supply this: S_max ≥ drain(1)/0.8
(C1, PS.01). The venue is passable; he does not forage.

**3. The learner also wrecks itself (seed 102: 598 → 158 s by life 9).** This is the
11-12 diagnosis: `cores.py:542` `l_pi = ||mu − a||²·adv` is unbounded below for
negative advantages, mu is unsquashed, and actions saturate. The 11-12 race (s90, 12 lives
per arm): base 263 s, tanh 359 s, **AWR 536 s at 0.00 saturated**, tanh+AWR 503 s. AWR
landed as a challenger in `experiments/policy_loss.py` and was never seated. This claim
ran on the collapsing loss.

**What must change (in order; each is a measurement, not an argument):**
- (a) Seat a bounded policy loss in the creature. AWR won the 11-12 race on the collapse
  question. Seating it re-buys LC.02, LC.08 and LF.03 (priced 0.78 CPU-h on 11-12). AWR
  alone gives stillness (~536 s) and no meals, so (a) is necessary but cannot pass LF.04 by
  itself.
- (b) Foraging must be learnable within 15 lives (~40k decisions). Two measured leads:
  J3's finding that the seated core (wm-latent) trains no weight that reads an action
  (wm-dyn is the action-conditioned challenger in LC.09); and the curriculum (CR.01; the
  weaning pilot was dead at its dose because meals came from contact at birth, not from
  approach).
- Re-run LF.04 only when a pilot learner reaches at least 5 floor meals (or 1 apple) per
  life within 15 lives on s90. Until then a re-run can only FAIL again. Keep N, margin,
  seeds and controls unchanged: a learner that forages clears 272.8 s.
