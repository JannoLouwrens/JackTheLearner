# NE.01 gate 1 (`ok_spread`): the binding need is integrity, and its spread is set by the probe's schedule

Measured 2026-11-14 ~06:25 UTC, JACK lane, J6(c). Not a ledger run. NE.01's own `_drop_probe`
was called through its module at HEAD 68e1207 (scratch script `/data/tmp/ne01_drop/probe.py`), seeds 0-4, ~15 s each.

## What the record says
NE.01 att 5 (56c6519, FAIL): `ok_spread` 1/3 seeds, `min_need_spread` 0.248 +/- 0.042, and the minimum is
`spread_i` (integrity). The other six clear: f 0.380 +/- 0.007, T 0.760, w 0.711, c 0.796, p 0.546, e 0.333.
**So J6(c)'s "band geometry for spread_f/spread_T" is aimed at the wrong channel.** f and T clear on every seed.
`ok_night_shelter` is 1.0 at att 5, so the band-geometry arm has no red gate left to serve.

## Per-seed drop probe (delta_i: 0 = unhurt, 1 = wrecked)
| seed | spread_i (p90-p10) | p10 | p90 | max | damaging hits |
|---|---|---|---|---|---|
| 0 | 0.304 | 0.077 | 0.380 | 0.381 | 6 |
| 1 | 0.201 | 0.191 | 0.392 | 0.394 | 6 |
| 2 | 0.239 | 0.110 | 0.349 | 0.350 | 7 |
| 3 | 0.338 | 0.126 | 0.463 | 0.464 | 9 |
| 4 | 0.198 | 0.095 | 0.293 | 0.294 | 6 |

s0-s2 reproduce the ledger (0.248 mean, 1/3 pass). 2/5 seeds clear 0.30.

## Why
The probe runs (drop, 74 random, 200 rest) x 3. One fall from the ladder top costs **0.10-0.14** of delta_i
(s1 jumps: 0.112, 0.127, 0.135; s4: 0.105, 0.116, 0.097, plus small follow-on knocks).
A 200-decision rest leg is 40 s, and `RHO_HEAL` = 1/900 s^-1 heals **0.044** in that time. So the trace is a
**ratchet**: three stair-steps that barely heal. p90-p10 measures how far the first step sits below the third.
That depends on the schedule (how many falls, how much rest between them), not on any property of the world.

Consequences, measured rather than argued:
* A probe that rested to full heal between falls (900 s = 4,500 decisions) would read spread_i at about ONE
  fall's damage, 0.10-0.22. It would FAIL worse. The world's single-fall integrity cost is below the 0.30 bar.
* A probe with more falls and no rest would ratchet him toward injury death and read near 1.0. It would pass,
  but only because of the schedule.
* Either edit picks the verdict through the apparatus. **Neither is honest without a declared reason first.**

## What has to be decided before NE.01 is re-run (no threshold moves)
Gate 1 reads "range over admitted behaviours". For integrity, the probe has to name which behaviour that means,
in world terms:
(A) **The worst injury the world admits short of death**: repeated falls until delta_i first crosses a stated
    level, or until injury death. That is the integrity analogue of the statue dying of thirst.
(B) **A typical bad day**: K falls inside one heal constant (900 s), with K taken from a measurement of how often
    random/learning bodies fall. Nothing measures that yet.
(C) **Keep the probe and attack the world**: alpha or ladder height. PS.01 fixed alpha, so this is a world edit.
    It would have to be raced (ruling 1's race rule), and PS.01 would go stale.
The recommendation is (A). It mirrors how every other LETHAL need is exercised (to the edge), and it is
schedule-free once the stop rule is declared. It needs the spec's notes amended before the run, with this file cited.

## (A) implemented and probed — 2026-11-14 ~06:55 UTC (JACK, J6(c)); no ledger run
`_drop_probe` now repeats (drop, 74 random, 200 rest) until death or `DROP_MAX_CYCLES` = 40. A **planted control**
was added beside it: the same schedule placed at ground height, so nothing falls. NE.01 now also requires
`ok_drop_control`: the control's spread_i must stay below SPREAD_MIN (0.30, unmoved). That is a strengthening, because
gate 1 can no longer be passed by the random flailing alone. Probe (scratch `/data/tmp/ne01_dropA/probe.py`):
| seed | ladder spread_i | ends | decisions | control spread_i (control ends) |
|---|---|---|---|---|
| 0 | 0.835 | injury (i=1.0) | 2204 | 0.032 (dehydration) |
| 1 | 0.569 | dehydration (i max .92) | 1953 | 0.106 (hyperthermia) |
| 2 | 0.785 | dehydration (i max .94) | 1909 | — |
Under the fast clock thirst kills him before the edge on 2/3 seeds, so the stop rule is "first death". That is recorded
per seed as `drop_cause`, not hidden. Seeds 3-4 were not probed. NE.01 is not re-run until the gate-2 ruling (INBOX 05:25).
