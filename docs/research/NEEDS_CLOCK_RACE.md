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
