# J1 step 1 — the policy-loss race: verdict

Rule declared and committed before any cell: `experiments/policy_loss_race.py`
docstring. Arms base / tanh / awr / tanh-awr, fresh seeds 150–152, 12 lives each,
Kaggle CPU. All 12 cells current against the code stamp at `1c06417`
(`policy_loss_race status`: 12 current, 0 stale, 0 missing).

**Tally (`python -m experiments.policy_loss_race tally`, 2026-11-15 10:10 UTC): WINNER tanh-awr.**

| arm | early→late s150 | s151 | s152 | late mean | late sat | q1 sat≤0.05 | q2 no drop | qualifies |
|---|---|---|---|---|---|---|---|---|
| base | 443→533 | 433→488 | 362→294 | 438.2 | 0.649 | no | yes | no |
| tanh | 440→591 | 445→567 | 440→580 | 579.3 | 0.987 | no | yes | no |
| awr | 400→560 | 399→556 | 409→574 | 563.1 | 0.0001 | yes | yes | yes |
| tanh-awr | 387→524 | 401→557 | 399→732 | **604.5** | 0.0 | yes | yes | **yes** |

Readings, not part of the rule:
- tanh-awr's lead over awr (604.5 vs 563.1) comes from one seed: s152 late 732. On s150
  and s151 awr is ahead or level (560/556 vs 524/557). The rule picks the highest pooled
  late mean, so the seat goes to tanh-awr. That margin is thin, and the seat holds only
  until something beats it.
- Only the AWR arms keep the actor off the rails. tanh lives longest of the
  non-qualifiers while pinned at saturation 0.99, which is exactly the case q1 exists to refuse.
- Every arm's late third outlives its early third except base s152 (362→294, which
  stays above the 0.5 drop bar). No arm forages yet: these are life spans of a
  creature that still dies of energy (LF_04_FAIL.md). This race seats the LOSS only.

What it buys (ORDERS J1): tanh-awr is seated in `cores.py` (`Core.act` squashes, and
`lc_update`'s policy term is AWR). Next: re-buy LC.02, LC.08 and LF.03 (LF.03 with its ruling-9
statue twin first), then the forager race.
