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

## J6(c) pre-declaration: which heat hazard CAN kill a sweating Jack (2026-11-14 03:55 UTC, no ledger)

`python -m experiments.heat_hazard_table` (pure arithmetic on needs.thermal_step under the sweat_paid patch; output
`/data/tmp/heat_hazard_table.txt`). Humidity modelled, declared in the module, as a vapour-gradient factor
f = (p_sat(35) - RH*p_sat(T_env))/p_sat(35) on the sweat clamp (580 W).
* **Exertion cannot do it.** Full power adds only 200 W (kappa_act = 2*M_BASAL/p_max): M_max = 300 W. Dry air needs
  T_env >= **59.6 C** at full power in the open to reach 40 C for 20 s. No dry-heat hazard at a physical temperature exists.
* **Humidity can, at biological thresholds.** Rest survives 35 C/80%, 40 C/20%, 45 C/20%; dies at 35 C/100% (240 s),
  40 C/80% (106 s), 45 C/50% (115 s). Full power dies already at 35 C/80% (94 s), 40 C/50% (96 s), and at shipped
  T_DAY 30 C only near saturation (30/80% roof 258 s, 30/100% open 322 s). Anchor: Vecellio 2022's uncompensable line
  (~31 C wet-bulb, light activity: ~36/80%, ~40/50%) sits between this table's rest and full-power rows, as it should.
* The roof is NOT protective in humid heat: 30 C/80% full power dies under the roof (258 s) and survives in the open
  (k_eff 4.29 vs 14.29) — sealed air is the hazard, which is the biology the W.2 roof route meant to probe.
=> Next arm to declare (on swp_k72's clocks once the water race tallies): humidity as a W0 world field (f on the sweat
clamp; humid hot spell or sealed humid shelter), W.2(c)'s hot route re-aimed to "sealed humid heat" with this table's
arithmetic as its pre-registration; bar 40 C unmoved. Unmeasured: what humidity does to NE.01's random-death causes.

## J6(c) humidity arm — route table on the swp_k72 body (2026-11-14 04:25 UTC, no ledger)

`python -m experiments.humid_race check` (output `/data/tmp/humid_route_table.txt`). Each candidate world is
swp_k72 + humidity patched into `needs.thermal_step` itself (same signature; f scales BOTH the sweat clamp and the
wet-skin channel; RH_local = RH_OPEN + (RH_SEALED - RH_OPEN) * occlusion — sealed air saturates with his own vapour).
Fixed ambient, 3000 s, DEATH_T_S dwell.

| route (start 37 C)   | swp_k72 (no humidity) | hum30: 30 C, RH .5 -> 1.0 sealed | hum34: 34 C day / 20 C night, RH .5 -> 1.0 |
|----------------------|-----------------------|----------------------------------|--------------------------------------------|
| rest, open, day      | lives (37.00)         | lives (37.00)                    | lives (37.48)                              |
| rest, roof, day (W.2 hot route) | lives (37.34) | lives (38.31)                 | **hyperthermia 186 s** (40.25)             |
| rest, soak, day      | lives                 | lives                            | lives (refuge: pool)                       |
| full power, open, day| lives (37.93)         | lives (38.44)                    | lives (39.15)                              |
| full power, roof, day| lives (38.32)         | **hyperthermia 78 s**            | **hyperthermia 59 s**                      |
| rest, soak, night (W.2 cold route) | hypothermia 70 s | hypothermia 71 s            | hypothermia 71 s                           |
| rest, roof / open, night | lives             | lives                            | lives                                      |

* **hum34 restores W.2(c)'s hot route AS REGISTERED** — the resting statue under the roof dies of hyperthermia —
  with the pool as refuge and open air survivable, so no re-registration of the route is needed. Its cost: T_DAY
  30 -> 34 breaks W.2's documented fixed point K_DRY = M_BASAL/(T_SETPOINT - T_DAY) (only a dry NON-sweating statue
  depended on it) and re-opens NE.01's DELTA_T_NIGHT calibration (held at a 20 C night by DELTA 14).
* **hum30 changes no temperature** and kills only EXERTION under a seal (exertional heat stroke), so W.2's hot route
  would have to be re-aimed to "work under a seal" (re-registration, reason this table) to be reachable.
* Both make the roof the learnable trap the W.2 docstring calls "the most GOAL-shaped thing this measurement
  found" — but now for a biological reason (sealed humid air), not a missing sweat gland.
* Note: f < 1 everywhere RH > 0, so the shipped e_evap is implicitly bone-dry air; humid arms shed less heat at
  every ambient. Unmeasured: NE.01 random-death causes under either (flailers peak 39.15 C in hum34's open sun).
=> Race to declare after the water race tallies: hum30, hum34 vs shipped on heat_race's rule, seeds 0-4,
W.1/W.2/NE.01; reported, not gated: `c_hot_reachable`, cause shares, decisions per random life.

## J6(c) humidity race — W.1/W.2 legs, rule declared first (2026-11-14 ~04:40 UTC, no ledger)

Rule declared in `experiments/humid_race.py` before any cell (heat/water race rule verbatim, arms hum30, hum34 vs
shipped, seeds 0-4; apparatus = swp_k72 W.2 s0 through the humid wrapper must equal the water race's cell).
Apparatus **OK** (byte-equal). W.1 + W.2 legs run (20 cells + apparatus); NE.01 NOT run: the verdict is already fixed.

| arm | W.2 c_pass (s0-4) | lost vs shipped (rule 1/3) |
|---|---|---|
| shipped, swp_k72 | 0/5 | — |
| hum30 | 0/5 | W.1 d_pass 0.8 (s2), W.1 _check lost s2 |
| hum34 | **5/5** | W.1 d_pass 0.8 (s2), W.1 _check lost s2, **W.2 a_pass 0/5** |

**=> NONE SHIPS** under the declared rule, whatever NE.01 reads. Why each loss happens:
- **W.2(a), hum34: real, and the declared cost.** (a) checks thirst against `1 - B_W t` on the deprived statue. That form
  assumes 37 C is the fixed point (`K_DRY = M_BASAL/(T_SETPOINT - T_DAY)`, shipped). At T_DAY 34 the open statue settles
  above 37 C and sweats, so water drains faster: a_err_w 0.158 vs INTEGRATOR_TOL 0.01. A hot world needs either K_DRY
  re-derived for the new T_DAY, or (a) re-registered against the sweating closed form. Racing those options is the next step.
- **W.1(d), both arms: the arm's own build is the fault, not the world.** (d) integrates `_ship_flux_w`, which reads `N.e_evap`
  directly; the humid arms scale evaporation by f(RH, T_env) only inside `thermal_step` (e_evap has no ambient
  argument). Where s2's scenario sweats, the stepper and the flux ledger diverge (d_rel_err_fine 0.0091 vs 5e-3,
  convergence ratio 0.95 vs 2.0). Repair: move humidity into a shared needs function (e.g. `evap_factor(T_env, occ)`)
  that both `thermal_step` and W.1's flux read, then re-run the W.1 leg.
hum34 is the first arm since sweat to make W.2(c) pass on every seed: the hot route as registered works again.

## J6(c) water race — harvested 2026-11-14 05:10 UTC (cells `/data/jack-logs/water_race/`, log `/data/tmp/water_race.log`)

All 65 cells done (rc 0). Apparatus **OK** (2/2 byte-equal replays). Verdict under the declared rule: **NONE SHIPS.**

| arm | NE.01 ok_random_deaths | random lives died | ok_spread | ok_forager_exercised | min spread (s0) | lost vs shipped |
|---|---|---|---|---|---|---|
| shipped | 1.0 | 25/25, all hyperthermia (dec 460-574) | 0.4 | 1.0 | — | — |
| sweat_paid (s0 only, apparatus) | — | 5/5, **all dehydration**, last at dec 2479 | — | — | f 0.228 | — |
| swp_w72 | **0.0** | **0/25** | 0.0 | 1.0 | f 0.254 | random deaths, spread, cause share |
| swp_k72 | **0.0** | **0/25** | 0.0 | **0.0** (2-3 meals/3 days vs MIN_MEALS 5) | **e 0.082** | + forager_exercised |

What this settles:
* **Under sweat, thirst was the only thing that killed a random flailer.** sweat_paid s0 had 5/5 dehydration deaths.
  Slow thirst to k = 72 (3600 s; about 1150 s at the flailer's 3.5x drain, against RANDOM_LIFE_CAP = 6000 dec = 1200 s)
  and nobody dies inside NE.01's one-day window. The knife-edge arithmetic in the module was too optimistic: 0 of 25.
* So NE.01 gate 2 (5/5 random lives die within one day) **cannot be met by any k = 72 arm that lacks an acute killer.**
  Shipped holds ok_random_deaths = 1.0, so under rule (1) no such arm can ship, whatever its other clauses read.
  Of the arms built so far, only the humidity arms add an acute killer. Gate 2 is probed alone under hum30/hum34
  (`experiments/random_death_probe.py`, NE.01's own `_random_lives`) before their NE.01 legs (~2 h) are spent.
* **The k = 72 energy clock collapses NE.01 gates 1 and 4.** spread_e falls to 0.082 (bar 0.30) because energy hardly moves
  in a probe of at most 3 sim-days when starvation takes 21. The forager eats 2-3 times in 3 days against MIN_MEALS 5.
  These gates were calibrated on the fast clock. A k = 72 world will need them re-registered against the declared clock
  (e.g. probe length scaled to it), with the reason stated, before any full-k72 arm can pass NE.01. That is a fact
  about the experiment, not the world, and it goes to the Director with the T2.10-style ruling queue.
* ok_spread also fails for shipped (0.4): spread_f 0.228 and spread_T 0.283 sit below 0.30 in every arm. That is
  (c)'s band-geometry item, still open.

## J6(c) gate-2 probe — humidity arms vs NE.01's random flailers (2026-11-14 05:25 UTC, no ledger)

`experiments/random_death_probe.py` runs NE.01's own `_random_lives` under `humid_race._apply`. Log `/data/tmp/random_death_probe.log`.

| arm, seed 0 | died | decisions at death (cap 6000) | causes |
|---|---|---|---|
| swp_k72 (apparatus, known negative) | 0/5 | — | reproduces the water race's cell |
| hum30 | 1/5 | 5996 | dehydration |
| hum34 | 3/5 | 3664, 4143, 4150 | hyperthermia, dehydration x2 |

Gate 2 needs 5/5 deaths inside [300, 6000] on a seed, and shipped's seed-mean is 1.0. Seed 0 alone puts both humidity arms
below that, so **neither humidity arm can ship under the declared rule, whatever NE.01's other clauses or seeds read.**
Their NE.01 legs (~2 h CPU) are not spent. Cells s1/s2 were left to finish as description only.

**The conflict, stated plainly.** Ruling 1 gives W0 one clock, k = 72. At that clock thirst kills in about 3 world-days and
hunger in about 21, which is what a human body does. NE.01 gate 2 demands that EVERY random flailer die within ONE world-day
(RANDOM_LIFE_CAP 6000 dec = 1200 s). That was calibrated when thirst killed in 570 s. Under k = 72 only an acute hazard that
kills nearly every random life within a day can meet it. A world that lethal is not the human one ruling 1 asked for:
hum34, the harshest arm built, kills 3/5. Gates 1 (spread_e) and 4 (MIN_MEALS 5 in 3 days) carry the same fast-clock
calibration (water race above). So NE.01's windows have to be re-registered against the declared clock (probe lengths in
world-days at k = 72), or ruling 1 has to be qualified. Either way, it is a ruling. A worker cannot widen gate 2's window
alone, because that loosens a registered bar. Until then, no k = 72 arm can pass NE.01 and no further J6 needs-clock race
can ship.
Probe finished 05:37 (PROBE EXIT). Remaining cells, description only: hum30 s1 0/5; hum34 s1 3/5 (dehydration x3, dec 4154-4257),
hum34 s2 1/5 (dehydration @4362). Pooled: hum30 1/10, hum34 7/15, every seed below 5/5. hum34's random deaths are 6/7 thirst, so humidity mostly speeds up
thirst through sweat rather than killing directly.

## J6(c) re-read against the re-registered NE.01 (`e054e0a`, windows in T_thirst/T_hunger) — 2026-11-14 13:35 UTC, no ledger

Windows under each arm's own `_apply` (read live from `_windows()`, computed in-process):

| arm | T_thirst s | T_hunger s | gate 2 window (dec) | life cap | forager horizon (dec) | night |
|---|---|---|---|---|---|---|
| shipped | 570 | 2100 | [300, 6000] | 6000 | 18000 | 2000 |
| k72 / swp_k72 / hum30 / hum34 | 3720 | 25500 | [1958, 39158] | 39158 | 218571 | 2000 |

**k72 (needs-clock race, 11-13 cells, re-scored from saved metrics — no run).** Every one of the 25 random lives died of
hyperthermia at decisions 445-549 (per-seed min/max 451/549, 475/546, 483/510, 445/533, 453/528). The re-registered window opens
at 1958 (2/19 T_thirst = 391.6 s). In-window 0/5 on every seed -> `ok_random_deaths` 1.0 -> **0.0** under the registered spec.
The old PASS was the fast-clock window admitting a ~100 s heat death; on the k72 clock that death is far too early to be "a
random life that dies of its needs". Rule (2) needs seed-mean >= shipped's 1.0, so k72 is now refused on gate 2 as well as
on forager/spread (e4c10cd). The pre-sweat body is the cause; swp_k72 (sweat paid) is the arm that replaced it.

**Humidity + swp_k72 gate 2 (probe re-run in flight).** The 05:25 probe ran at cap 6000 (copied to
`/data/jack-logs/random_death_probe_cap6000_1114/`). `random_death_probe` reads `_windows()`, so re-running it IS the re-read:
cells swp_k72 s0, hum30 s0-2, hum34 s0-2 relaunched 13:23 (logs `/data/tmp/rdp_reg_{a,b}.log`, 2 niced workers). Lives now run to
39158 dec, so cells take >10 min each (cap 6000 cells took ~2-12). Next worker: `python -m experiments.random_death_probe tally`.
Either way neither humidity arm ships yet: hum34 fails W.2(a) and both fail W.1(d) (b1272c3); this reading says only whether the
re-registration lets a k72-clock world meet gate 2 at all.

**Price of the long NE.01 legs (J6 c).** The forager horizon at k72 is 218571 dec, 12.1x shipped's 18000. The 11-13 k72 NE.01
cells took ~40 min/seed on the box with the 18000 horizon. A k72-clock NE.01 seed is therefore several hours of CPU, one Kaggle CPU
kernel per seed, never the box.
