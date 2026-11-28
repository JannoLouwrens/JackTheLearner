# A2-off — brain A at the authors' own preset: DRAFT, BLOCKED (no band at an affordable frame count)

STATUS: DRAFT — NOT FROZEN. No seed may run on this file. Order J33 (1); ruling 139.
Written 2026-11-28 12:30 UTC (JACK). Nothing below was measured by us unless it says "ours".

## What the order asked
cartpole_swingup, the official preset unchanged; band = MR.Q's interval at the
frames it was read at, taken from the source; seeds 30–32, twin (lr 0) on 30;
if one seed to the band's frame count costs > 3 T4-h, read at the EARLIEST
frame count the source tabulates.

## What the source says (checked 12:25, primary files)
1. **MR.Q's DreamerV3 number is NOT the current preset.** MR.Q (arXiv
   2501.16142, App. B.4, `scratch_sec2r_B/dl_mrq/mrq.txt` l.1459) re-ran
   danijar/dreamerv3 at commit **`251910d` (2024-07-29)**, "author-suggested
   hyperparameters from the DMC benchmark", "modified slightly to match our
   evaluation protocol", action repeat 2. At `251910d`, `dmc_proprio` =
   **`size12m`, train_ratio 512** (configs.yaml l.232–237, local clone
   `scratch_sec2r_A/dl/repo`). The current preset (since `f8817c4`,
   2024-12-07; head `e01491f`) is **`size1m`, train_ratio 1024, repeat 1** —
   for which no score is published at any frame count (§2).
2. **Only one frame count is tabulated:** Table 5, FINAL at 500K agent steps
   (= 1M frames): cartpole_swingup **736 [591, 838]**, a 95 % bootstrap CI of
   the 10-seed MEAN — not a per-seed band.
3. **No per-step DreamerV3 curve is released.** `facebookresearch/MRQ`
   `results/dmc_results.txt` (fetched 12:25, 606,073 B, sha256 `dee06808…`,
   copy `/data/jack-research/mrq_results/`) holds MR.Q's own per-seed curves
   only (every 5k steps, 10 seeds); DreamerV3's runs are figures only.

## Cost (inferred from our T4 timings, §2 "ours")
- `251910d` preset to 500K agent steps: 500K × 512 / 1024 = 250K gradient
  steps at size12m; ours: 1m 25.9 steps/s, S 2.76 → 12m ≈ 3–6 → **≈ 12–23 T4-h a seed**.
- current preset to 1M frames: 1M × 1024 / 1024 = 1M steps at 1m, 25.9/s →
  **≈ 10.7 T4-h a seed**, and no band to read it against.
Both exceed the order's 3 T4-h, and the earliest tabulated count IS the final one.

## So
The order's branch fires: "the source tabulates no interval at an affordable
frame count → do not invent a band: RESEARCH line, order waits." No band is
chosen here; seeds, twin and reader are not written. Two facts for whoever
re-sizes the row: (a) a band from MR.Q belongs to `251910d`/12m/512, not to
the current preset; (b) a 3-seed mean set against a 10-seed-mean CI is a
different test from A1–A3's per-seed bands and must say so.
