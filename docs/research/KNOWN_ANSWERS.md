# KNOWN_ANSWERS — our builds against published numbers (ORDERS J3, PUBLISHED_PRIORS §5)

A build is trusted only after it lands a published number on a small standard
task. Environment: `/data/venvs/jack-std` (§5 pins). Code: `experiments/brain_b.py`.
Rules are in that file's docstring, committed before each run.

| # | task | run | steps | published | ours | seeds | verdict |
|---|---|---|---|---|---|---|---|
| 1 | Pendulum-v1 | SB3 SAC, Zoo entry unmodified (`lib`) | 20,000 | −298.4 ± 10.3; accept −314.7…−282.1 | running (kernel jack-cpu-1794791696) | 0–4 | — |
| 1 | Pendulum-v1 | Brain B: SBX SAC + LayerNorm, Zoo entry (`ours`) | 20,000 | same band | running | 0–4 | — |
| 1 | Pendulum-v1 | control: never trains (`ctrl`), must fail | 20,000 | outside band | box run 11-16: −1227.6 ± 35.5 | 0–4 | holds |
| 2 | Pendulum-v1 | PPO, Zoo entry | 102,400 | −187.6 ± 17.0; accept −213.2…−162.0 | — | — | — |
| 3 | MountainCarContinuous-v0 | SB3 SAC gSDE Zoo entry; then any SBX/pink variant | 50,000 | > 90 on ≥ 3 of 5 | — | — | — |
| 4 | MountainCarContinuous-v0 | PPO, Zoo entry | 20,000 | 87.5 ± 4.2 | — | — | — |
