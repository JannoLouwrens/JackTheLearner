# LAB — the inventor's notebook (≤ 10 KB)

One entry per idea: date · seat · the bet · the number · ALIVE / DEAD / ENTERED.
A dead idea with a number is a result. Do not retry one without a new reason.

## Already known from the ladder (so nobody re-invents them)

- 2026-08-23 · learning core · five cores screened in W0 · only `wm-latent`
  learned to survive (4.65σ vs null, +92 s carried across deaths); `wm-efe` 2.05σ,
  `ppo-lp` 1.20σ, `ppo-needs` 1.06σ, `dreamer-xs` −0.94σ · seat PENDING.
- 2026-09-07 · control path · small plastic head 506, MLP 415, 57M-trunk arms
  344–350, SB3 reference 518 · VOID on the wrong null; being re-run as `D1.1`.
- 2026-08-24 · deliberation · lookahead gains 13 s ± 19 in W0 — the world is too
  shallow for thinking ahead to pay · needs the deeper world first.
- 2026-08-13 · mood → movement · does not reach behaviour (0.23–0.38 vs chance
  0.25) · DEAD as built.

## Ideas

- 2026-11-11 · **Working memory** (VACANT; incumbent GRU buffer FAILs `WM.01`
  on 2 of 3 seeds) · **SEATED — `WM.02` PASS 2026-11-12** (`2103829`) · `RingMemory.py`.
  **Claim run** (seeds 100–104): ring 0.902 ± 0.196, GRU 0.579, null 0.347; mean lead
  +0.323; shuffled 0.127, no-rotation margin −0.047 (both controls failed as required).
  **Bet:** (1) a store whose events ROTATE the held state — K planes,
  input-dependent angles, a gated absolute write, everything learned — the
  head-direction circuit's move. (2) It might beat the GRU because composing
  events is composing rotations and a rotation cannot forget or blow up, where
  a gated additive cell must discover modular update the hard way. (3) Killed
  by: any seed under +0.10 over the no-buffer null, losing to the GRU, or the
  no-rotation twin doing as well.
  **Pilot** (seeds 0–4, `WM.01`'s venue/recipe/bars, 1,031 params vs the GRU's
  4,872): ring **1.000 / 1.000 / 1.000 / 0.871 / 1.000**; GRU 0.546 ± 0.203
  (ledger); null 0.354; shuffled-state control 0.11–0.14; rotation disabled
  0.318 / 0.283. Not a one-trick: on the absolute last-write venue ring 0.998 /
  1.000 / 1.000 (GRU 1.000). Dead variants: angle init ×4 → 0.106 (chance);
  rotation with no write gate → 0.119.
  **Registered** with gates declared first, claim seeds 100–104 (never
  piloted), ~25 CPU-min; run handed to the workers (ORDERS INBOX).
  **Open after a PASS:** it is a 40-step fixture on a cyclic-group task — next
  ring should be a non-commutative register, then ME.8 / LF.02 with the ring
  in place of the GRU. Serves philosophy 3 (a seat raced, biology as oracle)
  and 4 (two controls that must fail).
