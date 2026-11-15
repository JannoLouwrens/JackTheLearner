# ORDERS — what the workers do. Written by the Director.
Director sitting: 2026-11-15 06:40 UTC

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5). Machine problems go under INBOX, one
line each. The Director rewrites this file daily; git keeps the history.
Rulings 1–8 cited here are in `docs/DECISIONS_RESOLVED.md` under 2026-11-15.

**He does not eat.** `LF.04` FAIL and `CR.01` FAIL say one thing: every life
ends of energy, a brain that learned nothing lives as long as one that did,
and no curriculum can teach a learner that cannot forage. Until an arm
forages, every creature-level cook-off only re-measures that. **J1 is
foraging and comes first in every JACK slot.** Parked behind it (ruling 7):
`LF.06`, the `LC.09` claim run, any `CR.01` re-race, an `LF.04` re-run.
**The freeze of 11-14 is lifted** (`LF.04` was dispatched): `cores.py` may be
edited under J1 only. World files (`w0.py`, `needs.py`, `survival.py`) still
change only when a raced arm wins.
**CPU (ruling 5).** Day ceiling 57,600 s; detached jobs ≤ 34,560 s a day and
≤ 2 COMPUTING detached processes at once. A Kaggle poller (billing < 0.10
cores) does not count toward the 2; at most 6 pollers alive. Kaggle runs
≥ 4 CPU kernels beside a GPU kernel (measured 11-15) — use 4.
**Kaggle CPU kernels** (`prefer="kaggle-cpu"`) cost no box CPU and no GPU
hours: every learner-side run goes there, `cwd="/tmp/jack"` for children.
**Reattach.** Never `JACK_REUSE_KERNEL` on a multi-unit spec (S7 closes the
hole). Single-kernel harvests may use it with the pin-swap proof in the commit.
**GPU.** `D1.0` holds Kaggle (watcher 4040958). Kaggle serves T4 now, not
P100. Colab: `colab --auth adc sessions`, one session per account.

---

## JACK — build the creature, his world and his curriculum (max 7 open, ranked)

*Order of work. CPU: **J1** (every slot, steps in order) → **J6** (harvest
what lands, then the fatigue re-registration) → **J7** → J4 → J9. GPU: **J2**
— read the log, act only on an exit. A CPU order and a GPU order may be
advanced in the same slot. Waiting on a kernel is not work: push the next
order's kernel in the same slot.*

### J1 — He must learn to eat: a loss that does not wreck him, then a forager
**Why.** `docs/research/LF_04_FAIL.md`: 120/120 lives died of energy; late
meals 0/3/0; the unbounded policy loss (`cores.py:542`) collapsed s102 from
598 to 158 s. The world can feed him (`NE.01`'s forager: 24–27 meals). Lab
bets on senses, weaning, credit horizon and meal memory all died — the
untested lever is the learner itself: 7 arms are built (`LC.09`).
**Do — declare each rule in the module docstring and commit BEFORE any cell.**
(1) **Policy-loss race**, Kaggle CPU: base / tanh / AWR / tanh+AWR × fresh
seeds 150–152 × 12 lives, as on 11-12 (`experiments/policy_loss.py`). Winner
= highest mean late-third life span among arms with saturated-action
fraction ≤ 0.05 and no seed whose late third is below half its early third.
Seat the winner in `cores.py` BY that tally; re-buy `LC.02`, `LC.08`,
`LF.03`. No arm qualifies → say so and go on with base. **Before that
`LF.03` re-buy (ruling 9):** it gains a statue twin (zero action, training on)
that its gate must tell from the learner, else FAIL; meals per seed go on the
row; and its `TRAIN_RATIO` is read from `LC.02`'s committed ratio for the
seated arm (the row says 0.125; `lf_03:109` hard-codes 0.25).
(2) **Forager race**, Kaggle CPU, 4 kernels at a time: all 7 `LC.09` arms
(reuse its `pilot` path, guard on) + a random-policy floor × fresh seeds
153–154 × 15 lives. Readout per arm-seed: floor meals and apples per life,
late third. Bar (from `LF_04_FAIL.md`, fixed): ≥ 5 floor meals or ≥ 1 apple
per late life on BOTH seeds, and above the random floor.
(3) No arm clears → run the top two at 4× the lives and write the meals-per-
life curve: slow learning and no learning are different findings and lead to
different orders. An arm clears → write it here; the Director orders
`LF.04` re-run and the seat contest the next morning.
serves: M1→M7 · philosophy 1 (learns by living), 3, 4.
**done when:** the loss seat is ruled by tally and step 2's table (and 3's
curve if needed) is in `docs/research/FORAGER_RACE.md`.
state: 11-15 10:30 (1) DONE: tally WINNER tanh-awr (`f72aac0`, docs/research/POLICY_LOSS_RACE.md): late mean 604.5 sat 0 > awr 563.1 sat .0001; base .65/tanh .99 sat fail q1. Lead rests on s152 (732). SEATED `a2bd4df` (Core.act=tanh, lc_update=AWR; policy_loss selftest flipped ok).
Re-buys: LC.02 PASS at a2bd4df (`eaa9302`; ratio stays 0.125; salt replay diverged, INBOX). LF.03 RUNNING detached (pid 88692, /data/tmp/lf03_rebuy.log, ~40 min). LC.08 REFUSED: cpu<48h closed (D20/D39), old PASS stands stale — needs re-registration or an off-box path (Director).
(2) forager race PUSHED 10:20-10:28, all 4 kernels: K0 `jack-cpu-1794738176` (poller pid 88890), K1 `…8207`, K2 `…8357`, K3 `…8507` (in-session, pollers died; detached cap counted K0's poller, INBOX).
Next: harvest LF.03 row (commit ledger, `run render`); harvest each K when COMPLETE: `JACK_REUSE_KERNEL=jannolouwrens/jack-cpu-<id> $PY -m experiments.forager_race kernel K` (guard may refuse in-session pushes: JACK_REATTACH_ACCEPT_MISMATCH=1, then check `forager_race status` = current); then `forager_race tally` -> FORAGER_RACE.md.

### J2 — The control-path contest (`D1.0`) — W46 slice in flight
**Why.** `T2.01` (he moves better than random) frees 25 specs, red since
Aug 12, waiting on which controller holds the seat.
**Do.** Read `/data/tmp/dispatch_d1_0.log` each slot; act only on an exit;
never kill the watcher. The W46 slice ends PARTIAL by design. **Shorten the
three-week plan:** this week also run one `c_e2e` unit on Colab
(`JACK_D10_ONLY`, units cache by impl_sha) — if a Colab session survives a
325-min unit, run the rest there and the verdict lands in W47. Before any
OTHER GPU spec is dispatched, re-derive a P100-frozen `MINUTES_CAP` from a
T4 rate (ruling 8). On a verdict: seat the winner BY VERDICT, write which
END-FIT rows it leaves reachable, re-run `T2.01` under it.
serves: M3, M4 · END-FIT row 1 · philosophy 3, 7 · objective 3 (free GPU).
**done when:** `D1.0` has a verdict.
state: 11-15 05:51 D1.0 att 9 **VOID** (harvested `651c805`): on T4 aprime/b_split/c_e2e hit 60/60/61% of STEP_TARGET 750k under P100-frozen caps 90/150/150 (d_mlp 100%, ~11 min).
Means (not a verdict): aprime 459 > d_mlp 422 > c_e2e 305 > b_split 259; margin .33 sigma. Fix `db262d9`: MINUTES_CAP 195/325/325/15 (T4-measured x1.3; no gate moved) -> all 13 units re-buy,
~40 GPU-h expected / 53 worst > 30/week. `70505b3`: JACK_D10_ONLY buys a declared subset; omitted units -> PARTIAL ERROR (no verdict), bought units cached by impl_sha 873e861e.
06:11 W46 slice dispatched (21.2 h proj vs 22.22 left): ref, aprime 0-2, d_mlp 0-2 (Colab), b_split 0; watcher 4040958. It ENDS in a PARTIAL ERROR by design.
Next: on that exit, W47 (Sun 11-22): JACK_D10_ONLY=b_split:1,b_split:2,c_e2e:0,c_e2e:1 (~25.5 h), W48: c_e2e:2 -> verdict. Never JACK_REUSE_KERNEL on a multi-unit spec.

### J6 — W0's red world rules (`W.2`, `NE.01`)
**Why.** `NE.01` frees 8, `W.2` frees 3. Five races shipped nothing; the last
red both arms share is `NE.01`'s fatigue-spread clause, which reads fatigue's
start-up transient — it measures how short the life was (`64d949a`).
**Do.** (a) Harvest the 10 humid `NE.01` cells as they land (commands in
state), `humid_race tally`, write the verdict. (b) Ruling 3: re-register
`NE.01`'s fatigue clause as an edge probe in the shape of the integrity one
— exertion to exhaustion then rest; fatigue must cross ≥ 0.30 of its range
and recover; planted control with fatigue frozen must FAIL; bar 0.30
unmoved; written and committed before any run; the old transient reading
stays on the row, reporting-only. (c) Re-run `NE.01` on shipped, then
re-tally DRINK (dB) and humid (hum34) against it — on Kaggle CPU. (d) An arm
that wins ships; then `W.2` and `NE.01` are re-run on the shipped world.
serves: M2 · END-FIT row 4 · philosophy 1, 3, 4.
**done when:** `W.2` and `NE.01` each have a new verdict with every
formerly-red clause explained.
state: (a) DONE `e054e0a`. (c) DONE: hum30/hum34/swp_k72 FAIL gate 2. f-spread DIRECTOR CALL open (`64d949a`). W.2(a) re-reg sweating CF `bc7b99f`; W.1 wrapper fix `39d8a66`.
`2131a04` W.2 cells current (hum34 5/5, shipped+hum30 0/5). **DRINK race verdict: none ships** (40 cells current): dB loses only NE.01 ok_spread 0/5 vs 5/5 (the f-spread call). Humid verdict waits on NE.01 (0/5 both arms).
10:25 (a) DONE: h3 harvested (ACCEPT_MISMATCH; in-kernel shas current) -> 10/10 (h1's stamp differs only by f65ad58's tally-only humid_race.py edit). `humid_race tally`: **NONE SHIPS** — hum30/hum34 NE.01 ok_spread 0/5 (f-spread), ok_random_deaths .6/.8; shipped ok_spread 1.0.
(b) APPLIED + COMMITTED before any run `eaa9302`. Found in review: the 720 s probe would die of thirst on shipped (water kills at 570 s), so all non-f needs are held at start (FATIGUE_HELD). Apparatus smoke s0 shipped (no ledger): spread_f .81, rise .87, rest undoes .87, no death; frozen control 0.0.
Next (c): NE.01 on shipped + dB + hum34 under the new clause, Kaggle CPU (kcpu full now: 4 forager + ladder) -> re-tally DRINK and humid against it.

### J7 — The deeper world (W1): `W1.04` ruled, then its energy clock
**Why.** W0 is too shallow for thinking ahead to pay (`DP.05`). In W1's venue
lives end at ~53 s against a 5600 s horizon: the energy clock is the defect.
**Do.** (1) Ruling 2: `W1.04`(a) and (c) contradicted because both read
`W1.02`'s repeat arm. (c) is measured on the COMPETENT arm only — the oracle,
20 fresh lives per seed, p5 ≥ horizon, control unchanged — plus one new
clause: the oracle's p5 must exceed the repeat arm's p95 (survival by
competence, not by a soft world). (a) is unchanged. Re-register, fixtures,
then `run W1.04` once on Kaggle CPU; a FAIL at ~53 s is the honest reading.
(1b) Ruling 9: `W1.02`'s detection clause injects +5.0 into an outcome whose
mean is 5.4 — it cannot miss. Re-register the injection at the smallest gain
a W1 claim must resolve (size declared from `W1.01`/`DP.05` before any run),
bar ≥ 3σ unmoved. A red is W1's resolution finding and joins step 2's judges.
(2) Race W1's energy clock (`W1.00`(b)): ≥ 3 clocks declared first, judged
by `W1.00`–`W1.04` unchanged. (3) Register `DP.05`'s question in W1.
serves: M2 · END-FIT row 4 · philosophy 4, 5.
**done when:** `W1.04` and `W1.00` have new verdicts and `DP.05`-in-W1 is registered.
state: 11-15 06:39 kcpu PATH BUILT `36ce96d`: run() = 1 Kaggle-CPU kernel/seed (5 arms, 4-worker pool), units cached by impl_sha in /data/tmp/w104/units, `… w1_04_horizon_longer_than_consequence kernel <seed>` buys a seed ahead. Fixtures 15/15.
Pushed s0 under the PRE-ruling-2 spec (`jack-cpu-1794724769`, RUNNING; ruling landed mid-slot). That kernel is a PILOT only: re-registration changes impl_sha -> no unit reuse, and multi-unit reattach is forbidden (ruling 6).
s1/s2 pushes REFUSED: "Maximum batch CPU session count of 5 reached" (measured: 5 CPU + 1 GPU concurrent, cap is 5).
(1) DONE `f6c6177`: re-registered per ruling 2 BEFORE any run: (c) = oracle p5 >= 5600 s AND oracle p5 > p95(fresh repeat); repeat p5 reported only; fixtures 16/16; impl_sha 9cbe4f1e.
11-15 10:45 (1b) MEASURED, spec not yet edited: W1.02 row (9f91377) G mean 5.38 s, sd 1.60, t_inj 15.9 at +5.0 (ruling 9 confirmed: +5.0 = 0.93x the mean). 3-sigma floor at n=48/arm = 3*1.60*sqrt(2/48) = 0.98 s of G, so G resolves ~1 s; +5.0 tests 5x coarser than that. Candidate sizes to convert into G units before declaring: DP.05 gap_s 13.3 s span (plan 141 vs react 120), W1.01 oracle-passive gap 10.2 s span (70.2 vs 60.0). Units differ (span s vs satisfied-s), so no number is declared yet.
Next: do (1b) FIRST — `w1_02_outcomes_have_resolution.py` is in W1.04's IMPL_DEPS (and its depends_on), so editing it after the kernels re-keys and stales them. Then, as kcpu slots free (5 max; s0 pilot ends ~4 h), launch 3 detached pollers `launch_detached.sh /data/tmp/w104_kcpu_s<s>.log $PY -m experiments.tests.w1_04_horizon_longer_than_consequence kernel <s>` (s=0,1,2), then `run W1.04` reuses the units.

### J4 — The oldest blockers nobody owns (`PS.02`, `PG.6`, `T2.03`)
**Why.** `PS.02` (cold is felt before it kills) is FAIL with its code
unchanged 94 days and frees `LF.01`, `SH.01`, `SH.02`, `XL.01`; `PG.6` and
`T2.03` are PASS-but-stale (96 and 82 days) and free 3 by a re-run alone.
**Do.** (1) Re-run `PG.6` and `T2.03` (GPU ones wait for `D1.0`'s slice).
(2) `PS.02`: one slot to read its last row and code and write the cause in
state; then repair the world or the body, or — if the test is wrong —
re-register with reasons. Bars unmoved.
serves: M2, M5, M7 · philosophy 4 (a failure is fixed, not filed).
**done when:** `PG.6`, `T2.03` are current and `PS.02` has a new verdict.
state: 11-15 08:42 (2) PS.02 CAUSE PINNED (dump, no ledger write): the thermal law is fine. 15/16 s2 cold lives die within ~2 s of `time_to_lethal_s` at both trees. Exactly ONE differs: s2 life 0 (t_cold -0.37, predicted 48.8 s, the same life v1 FAILed on). Before c510b92 it was censored at 80 s (fire-saved). At HEAD the water repair moved the random walk, so the fire PROLONGS it to a death at 78.2 s: min_dist 0.84 m, cooling 0.126 vs no-fire 0.204 degC/s, `_fire_explains`=True.
Censoring (v2) covers only fire-explained lives that SURVIVE, so this one enters the probe's training set and the death_s_max gate. Dropping that life alone moves R2 from -0.265 to 0.713.
Next (CORRECTED 08:50): do NOT just "censor any fire-explained life". `_fire_explains` is True on ALL 16 s2 lives (cooling/no-fire ratios 0.617-1.000; clause (b) has no margin), so that rule censors everything. v3 needs a rule fixed independently of this outcome. Options to race or rule: (i) a pre-registered fire-received margin (ratio <= 0.75 catches only life 0; lives 3/5/15 sit at 0.78-0.86, so the threshold is a knob and must be argued), or (ii) retarget the probe to the CURRENT law's time-to-lethal from felt state, not the realized death that future random fire visits decide. All bars kept. Dump scripts: /data/tmp/ps02dump/{dump,diag,count}.py.

### J9 — The two ends with no test: fire (`W.5`) and advice
**Why.** `coverage`: GOAL's "tree → fire → shelter world" and "advised, not
commanded" have no live claim spec; END-FIT reads "none registered".
**Do.** Register only (no build): `W.5` fire (wood burns, warms, cooks; a
control that must fail) and the advice gate (true advice makes him try
sooner; a lying adviser loses him; a mute adviser control). Each with
seeds, bar and control written before any code.
serves: M2, M6 · END-FIT · philosophy 4, 7.
**done when:** both are in the registry and `coverage` no longer lists the two ends CLAIM-DEAD.
state: open — ordered 11-15; lowest rank, one slot.

**Parked behind J1 (ruling 7), built and ready:** `LF.06` hand-over (`2f632ba`);
`LC.09` claim run (`8ee2edd`; needs checkpoint chaining past 12 h); `CR.01`
re-race; `LF.04` re-run (unchanged). The diary in his decisions likewise.
**Next up, as slots free:** 1. Paraphrase recall (`T2.10`): a learned scorer
vs the seated one. 2. Inside/around sweep (ruling 11-11): re-credit `LG.00`,
`HR.4`, `T3.08`, `LG.10/12/13/14`, `ME.11.C/D/E` as parent-rig results; delete
`AudioListener._transcribe_api`. 3. Ring memory in the creature (`ME.8`/`LF.02`).
4. Learned ear vs a no-ASR null; `HR.1`, `HR.5` under it. 5. Grip-rule and
size cook-offs. 6. `UB.11`–`UB.13`, `TA.03` (GPU, after `D1.0`). 7. Sense
added later vs from birth; `XL.01` at power.

---

## SYSTEM — work on the machine (max 3 open)

*Order of work: **S3** (minutes) → **S6** (the owner has waited four days) → **S7**.*

### S3 — `run verify` exits 0, and the CPU cap is seen to hold
**Why.** `verify` is red a fifth day. Its last red is `T0.34` behind `T0.33`,
FAIL because `W1.00` is classed `cpu<2h` × 20 seeds = 360,000 s > the day.
**Do.** (1) Ruling 4: `W1.00`'s budget class becomes `kcpu<12h` and it runs
there (a class says where a run really goes). `T0.33`'s clauses untouched.
(2) Re-buy `T0.33` → `T0.34` → `run verify`; `BD.01` gets its first row.
(3) Ruling 5 in `cpu_budget`: a detached process billing < 0.10 cores is a
poller — not counted in the 2-alive cap, capped at 6; planted 7th refused.
(4) Write in state the `detached:*` total for 11-15 at day end.
(5) Ruling 9, one commit: the GPU week key (`gpu.py:495`) follows Kaggle's
reset (Saturday 00:00 UTC, `gmtime`), and `T0.12` gains a fake-clock probe —
Fri 23:59:59 and Sat 00:00:01 land in different weeks; re-buy `T0.12`, `T0.24`.
serves: philosophy 4, 5, 6.
**done when:** `run verify` exits 0 on a clean tree and still exits non-zero
on a planted PASS-on-FAIL-dependency; one day ends with `detached:*` ≤ 34,560 s.
state: 11-15 09:15 (1) DONE `6623eb0`: W1.00 kcpu<12h, all 80 tasks = ONE Kaggle-CPU kernel (~1 h, 0 box CPU; `... w1_00_null_is_strongest_nonlearner kernel` buys it ahead). cpu_foreclosed [W1.00] -> [].
(2) T0.33 PASS -> T0.34 PASS `01dfc2c`. `run verify` still exits 2 on ONE red: LF.03 STATUS:VOID (J1's ruling-9 re-registration `ad6a644`; heals with J1's LF.03 re-buy, not ours).
BD.01 still has no row: budget CPU_DAYS (3 seeds), no kcpu path; needs a re-class like W1.00 before it can run (Director: say if ruling 4 covers it).
Earlier: (3)-pollers built `d95d569`/`d5a5ab6`/`f662f48`; W0.PL PASS a4. Open: (3) idle pollers still count toward DETACHED_MAX_ALIVE (ruling 5 code), (4) day-end detached total 11-15, (5) GPU week key + T0.12 probe.

### S6 — A video the owner can watch (owner request 11-12) — START TODAY
**Why.** Owner: "a video he can watch". Card line 1: he can be known.
**Do.** `scripts/life_video.py` renders OFFLINE from the most recent recorded
life — never inside a scored run — with `experiments/render.py` (Xvfb +
llvmpipe): third-person view, hunger / thirst / temperature overlaid,
phone-playable H.264 mp4 ≤ 20 MB at `/data/jack-video/latest.mp4` (+ a line
naming the life, seed and commit). The most recent life, no picking. Run by
hand at nice 19; no cron, no service. If no life is recorded with poses,
record one statue-free s90 life first (detached, ≤ 600 CPU-s). A first
rough 30-second clip today beats a finished one next week.
serves: M8 (he can be watched) · philosophy 1.
**done when:** the mp4 exists, plays, and is of the latest recorded life.
state: open — not started (fourth day). Next SYSTEM slot after S3 step 1–2.

### S7 — A remote run is one code version, and a reattach proves itself
**Why.** `D1.0` att 8: a hand-set reattach leaked to 7 unit bodies and a seat
verdict was saved only by a crash. The guard it overrides refuses every
reattach after any push, so it is overridden by hand in every harvest.
**Do, in `gpu.py`, smaller if possible, one commit each.** (1) `JACK_REUSE_KERNEL`
applies to the first `submit()` only and is refused when a second body asks
for the same slug; a reuse does not skip pre-flight for later fresh units.
(2) `reattach_code_check` does the pin-swap proof itself (pull, swap pin to
HEAD, compare sha, `git diff pin..HEAD` ∩ IMPL_DEPS empty) and matches the
receipt by slug (± 2 s, nearest); drop `ACCEPT_MISMATCH`. A refusal writes
no ERROR ledger row. (3) A multi-call spec pins calls 2..N to call 1's HEAD
and runs the push guard once; `docs/LAB.md`, `docs/ORDERS.md` are prose.
(4) The watcher timeout starts at RUNNING, not at QUEUED; `gpu.py:186`
reports the device Kaggle actually gave. Each with a planted test.
serves: philosophy 4 (no mislabelled data in a verdict), 5.
**done when:** a planted second-body reuse is refused, a pin-only reattach
passes with no override, and a planted IMPL_DEPS change is refused.
state: open — ordered 11-15 (INBOX lines of 11-14/15 folded in here).

**Next up:** (1) pre-flight prints UNKNOWN for the shared account unless probed. (2) **S4** —
six PASSes gated on a seed AVERAGE that admits a failing seed (`ME.10`
first). (3) `ran_at`-pinned rows read only the live ledger row. (4) `T0.13`'s
13 clauses that can never fire. (5) `protocol.py` in IMPL_DEPS stales GPU
certificates on an enum addition. (6) `coverage` baselines (exit 2,
bookkeeping). (7) D52 scout retry; `run status` STEERING reader parses 0.

---

## INBOX — one line each, workers add, the Director empties
- 11-15 10:30 JACK/J1: LC.02 re-buy PASS but HASH-SALT replay diverged on 3 deciding wall-clock metrics (meter_live, wm-latent/clears@0.125, conjunction 1->0) — a throughput floor is load-dependent, so its salt replay measures box load, not determinism; earlier LC.02 replays were clean.
- 11-15 10:30 JACK/J1: LC.08 cannot be re-bought as ordered: registered cpu<48h, which the runner refuses (D20/D39); its 0.03 s Oct PASS now sits stale on the seat change. Needs re-class/off-box route.
- 11-15 10:30 JACK: launch_detached's cap counts a Kaggle poller as one of the 2 detached jobs (refused K1-K3 with LF.03 + K0's poller alive), contrary to ruling 5 (pollers < 0.10 cores don't count); workers fall back to in-session pushes, which feed the reattach-guard dead end above.
- **OWNER QUESTIONS 11-15 (via session): "only 120 iterations? isn't that too little" and "how do we know winner of this is best brain overall and not tweaks or more training needed".** Session reading, for the Director to rule: (1) the forager race (2 seeds × 15 lives) is a SCREEN, not a crowning — write its result as "eats within 15 lives / does not", never as "best brain"; no seat changes hands on it without the owner's CONVERGENCE CHECK (Addendum 2: runner-up's late slope ≤ 0 or crossover beyond 3× budget, else extend both) and ≥ 3 seeds. (2) Add ONE long-horizon run per top arm — ~1M decisions on one Kaggle CPU kernel each (~10 h, free) — with the meals-per-life curve, so "slow learner" and "cannot learn" are told apart with room to spare; 4× (≈160k decisions) may still be below where this family starts to learn (LC.03 needed 400k for a small gain). (3) Say in the race write-up what each arm was NOT given: tuning effort per arm, and whether any arm's hyperparameters were touched after seeing results. (4) A foraging winner still owes END-FIT (body, hand-over, words, chains) before adoption — unchanged.
- 11-15 08:45 JACK: reattach guard dead end — in-session push of h4 (`…4063`) recorded code sha 4821824c that no committed tree reproduces (45050dd hashes 74e82f24, HEAD d88f2b02), so the refusal's advice "reattach from the submitting commit" cannot work; meanwhile a worktree lacking the receipt log harvested with only "unverifiable — proceeding". The cell's own in-kernel shas (all = HEAD) were the real check. A failed main-tree reattach also appends a receipt that later reattaches compare against.

- 11-15 06:43 JACK/J7: Kaggle's batch CPU cap MEASURED = 5 (6th push refused "Maximum batch CPU session count of 5 reached"; 5 CPU + 1 GPU ran together). Header's "use 4" leaves 1 spare; until ~11:00 one slot holds W1.04 s0 `jack-cpu-1794724769`, a pre-ruling-2 pilot with no unit value (pushed before the 06:40 rewrite reached me).
## DONE (kept one day)

- **`LF.04` FAIL** att 2 (`bc631a8`, Kaggle CPU, controls held): living does
  not yet teach him — 120/120 lives die of energy. Finding → J1.
- **`CR.01` FAIL** att 1 (`f519826`): no curriculum beats none (.163); the
  Curriculum seat stays with null BY VERDICT. Re-race parked behind J1.
- **Meal memory DEAD at its claim gate** (`f0c5e4e`): margin vs yoked .0985
  < .10 as declared; not registered. **DRINK race: none ships** (`2131a04`).
- **`T6.03` re-registered and PASS; `LF.03` PASS** re-bought 11-14 08:52 (M1).
- **`W0.PL` PASS a4** (regate 02:49); S2's cap, poller billing and timeout
  built (`d95d569`, `d5a5ab6`, `f662f48`) — folded into S3.
- **INBOX of 11-14/15 (11 lines)**: rulings 4, 5, 8; S7; SYSTEM Next-up 5, 6.
