# W1.04 conjunct (c) cannot pass in any world — measured 2026-11-14 02:40 UTC (JACK, J7)

**Finding.** W1.04 as registered can never PASS: every outcome is VOID or FAIL(c),
whatever W1's energy clock is set to. J7's planned next step ("race W1's energy
clock, re-run W1.04") therefore cannot turn W1.04 green, and should not be run as
the repair until the spec is re-registered.

**Why.** Each arm is ONE rollout of N_DEC = 28000 decisions (5600 s). Lives are
sequential inside it (death -> reset E0 -> next life). So:
- V4 VOIDs unless every lethal arm has >= MIN_LIVES = 20 **completed** lives;
- 20+ completed lives share one 5600 s window, so their spans sum to < 5600 s;
- conjunct (c) needs the 5th percentile of all spans >= 5600 s, i.e. ~95% of spans
  at the horizon — but only one span (the censored fragment) can reach it.

Through the spec's own `_spans_and_causes`/`_p5`, the best possible partition
(equal lives, k = 20..219) gives **max p5 = 266.6 s** (vs bar 5600 s). Zero completed
lives -> `_check` returns VOID "under minimum lives for a percentile". The FAIL on
the ledger (640e329: p5 54.3/52.0 s, ~272 energy deaths, 77–102 lives per arm) is
therefore a reading of the apparatus, not only of the world. The control (death off,
one life censored at 5600 s) passes exactly because it has zero completed lives — the
only configuration (c) can pass, which V4 forbids for lethal arms.

**Second defect (semantic).** (c) gates W1.01 at its worse arm = the PASSIVE arm,
whose claim is "passivity dies" and whose (a) needs its deaths at <= H/3. Requiring
the passive arm to survive the horizon contradicts the claim it is supposed to
make real.

**Re-registration this points to (not done here; strengthen, never weaken):**
measure (c) on INDEPENDENT lives — n >= 20 fresh lives per arm-seed, each run until
death or censored at HORIZON_S — so p5 >= HORIZON_S is attainable by a survivable
world and unattainable by today's (lives ~54 s). Gate (c) on the arms whose claims
need long lives (W1.01 oracle, W1.02 repeat), not on the arm that must die. Keep
the bar 5600 s and the control (death off must reach the horizon). Only then race
the energy clock against it.
