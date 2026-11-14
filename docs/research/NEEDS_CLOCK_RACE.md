# J6(b) needs-clock race — result 2026-11-13 (harness 386bfba, cells /data/jack-logs/needs_race)

Arms: shipped (B_E 1/1800, B_W 1/450, TAU 700/160) vs k72 (1/25200, 1/3600, 910/210). Seeds 0-4, 20 cells, nice 19, 2 workers.
Rule declared in experiments/needs_clock_race.py BEFORE the run. **Verdict: k72 does NOT ship** (`k72_ships: false`).

| clause (seed mean)      | shipped | k72 |
|-------------------------|---------|-----|
| W.2 (c) passes all seeds| no      | **yes** |
| ok_forager              | 0.8     | 0.2 |
| ok_forager_exercised    | 1.0     | 0.0 |
| ok_spread               | 0.4     | 0.0 |
| other ok_* (6)          | 1.0     | 1.0 |
| max cause share (gate 5)| 1.0 hyperthermia | 1.0 hyperthermia |
| random-life decisions to death | 460-574 | 445-549 |

Why k72 lost — every loss is HEAT, in both arms, not hunger or thirst:
* All 50 random lives die of hyperthermia at ~450-575 decisions under BOTH clocks; the needs clock never acts on them.
  k72's energy spread inside that heat-bounded life is 0.04 (shipped 0.33) -> ok_spread 0. The spread clause measures
  need motion inside a life that heat ends; a slower need clock cannot move in it.
* The scripted forager dies of hyperthermia in k72 4/5 (1180-3573 s) vs 1/5 shipped. Mechanism read from the cells:
  k72 forager sleeps with occ 0.998 (shipped 0.946), meals 0-3, relocations 0 — with no thirst to wake him at night
  (shipped wakes at w<0.12 and walks to water) he stays under the roof the docstring says "cooks him in ~3 min" at occ 1.0.
  **Shipped's forager survives partly BECAUSE thirst interrupts an overheating sleep** — its thermoregulation is
  coupled to the very need rates under test.
Finding: NE.01's red gate 5 (share 1.0 hyperthermia) is the binding fault of W0; the needs clock is downstream of it.
Next (not done, needs declaration first): race the heat channel / shelter geometry (J6(c)), and a forager whose
heat rule does not depend on thirst, before any needs-clock re-race. No world file was edited.

## J6(c) — the heat race (sweat), harvested 2026-11-14 01:49 UTC

`experiments/heat_race.py`, rule declared 11-13 ~16:30 before any cell; seeds 0-4, 45 cells, log
`/data/tmp/heat_race.log`, cells `/data/jack-logs/heat_race/`. Apparatus known-positive: shipped replays the J6(b)
cells 10/10 exactly. **Verdict as declared: NONE SHIPS.** No world file edited.

| NE.01 (seed-mean)          | shipped            | sweat               | sweat_paid          |
|----------------------------|--------------------|---------------------|---------------------|
| ok_random_deaths           | 1.0                | 0.2                 | 0.6                 |
| ok_spread                  | 0.4                | 0.0                 | 0.0                 |
| ok_forager                 | 0.8                | 1.0                 | 1.0                 |
| other ok_* (7)             | 1.0                | 1.0                 | 1.0                 |
| random deaths, cause       | 25/25 hyperthermia | 19/25 dehydration   | 23/25 dehydration   |
| max cause share            | 1.0                | 1.0                 | 1.0                 |
| random-life decisions      | 460-574            | 2364-5835           | 1433-5489           |
| forager survives 3 days    | 4/5 (1 heat death) | 5/5                 | 5/5                 |
| spread_T (every seed)      | 0.72-0.81          | 0.283               | 0.283               |

W.1 and W.2: identical flags in all three arms (W.2(c) still red, needs-clock rate; W.1 PASS 5/5).
Lost on: both arms — `ok_random_deaths`, `ok_spread`, clause (2) `max_cause_share not lower`.

What it shows:
* Sweat does what biology says: it moves W0's binding fault off heat. Random flailers no longer cook in ~100 s; they
  die of THIRST, in tight clock-set clusters (sweat ~2365-2440 dec, paid ~1435-1510 dec in 8 of 10 seeds) — the water
  clock, not behaviour, decides the moment. The fault is now the one J6(b) named: thirst runs ~6.3x past k = 72.
* `ok_random_deaths` falls because 2-6 of 25 random lives per arm outlive the 6000-decision day (deaths are not all
  in-window); `ok_spread` falls because a sweating body regulates core T (spread 0.283 on all 10 seeds — a
  thermoregulated homeotherm, which NE.01's ">= 0.30 on every need" gate reads as a fault) and min spread 0.14-0.25.
* Gate 5 (one cause < 60%) cannot pass in a world whose random deaths are all set by one clock, whichever clock it is.
Apparatus note: the tally's `*_pass` suffix also catches `W.2:c_widest_tol_that_would_pass` (a tolerance, lower is
better); tied at 14.82 in all arms, so this verdict is unaffected — fix before the next race re-uses `tally()`.
Next arm (declare first): sweat_paid + the water clock moved toward k = 72 together, on the same probes; and ask
whether NE.01's T-spread clause is right for a body that thermoregulates (re-register with a reason if not).

## J6(c) next arm — the water clock on the sweating body (launched 2026-11-14 03:24 UTC, `3d7c0a0`)

`experiments/water_clock_race.py`: swp_w72 (sweat_paid + B_W 1/3600) and swp_k72 (sweat_paid + full k72 clock) vs
shipped, the heat race's rule verbatim, declared in the module before any cell; cells `/data/jack-logs/water_race/`,
log `/data/tmp/water_race.log` (NE.01 cells ~10-17 min each, harvest with `python -m experiments.water_clock_race tally`).
Apparatus: sweat_paid W.2 s0 replays the heat race's cell exactly; W.1 `_check` passes 10/10.

**Finding read off the first cells (s0 W.2; heat race cells re-read):** sweat makes heat death UNREACHABLE in W0.
* W.2(c) `c_hot_reachable` = 1 for shipped on all 5 seeds (roof route peaks 40.33 C), **0 for sweat and sweat_paid on
  all 5 seeds** (roof route peaks 37.34 C). The heat race's "W.1 and W.2: identical flags in all three arms" was true
  of the `*_pass` aggregates only — (c) was already red on the clocks, so this sub-clause's flip was masked.
* Hyperthermia deaths across both sweat arms: 0 of 50 random lives, 0 of 10 three-day forager runs.
* swp_k72 puts BOTH clocks inside W.2(c)'s 2x band (thirst ratio 0.97, hunger 1.30 — as k72 did in J6(b)) and still
  fails (c) on `c_hot_reachable` alone. swp_w72 fails (c) on hunger (widest tol 14.79), as arithmetic said.
* So no sweating arm can pass W.2(c) as registered, whatever its clock. This is biology talking, not a harness bug:
  at T_DAY 30 C with an evaporative channel a resting human does not cook, under a roof or not. W0's only heat death
  was the missing sweat. A world that kills him "the way a human body dies" needs a REAL heat hazard to keep (c)'s
  hot route honest — exertion in hotter sun, a fire, a sealed hot space whose air temperature actually exceeds skin,
  or humidity capping evaporation — raced, not assumed. Next arm to declare: sweat_paid + k72 clock + one such
  hazard, with W.2's roof route pointed at it (re-registered with this reason; bar 40 C unmoved).
