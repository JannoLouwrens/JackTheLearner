# ORDERS — what the workers do. Written by the Director.
Director sitting: 2026-11-13 08:45 UTC

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5). Machine problems go under INBOX, one
line each. The Director rewrites this file daily; git keeps the history.
Rulings cited here are in `docs/DECISIONS_RESOLVED.md` under the same date
(1–4 at 06:40, 5–7 at 08:45).

**The owner spoke today (11-13).** The finished Jack — human-like body; trained
here, handed to people, still learning there; advised in words — now judges
every design (END-FIT, top of `docs/CHAMPIONS.md`; card line 7). Every
architecture cook-off says in its registration which END-FIT rows its arms
serve; no winner is adopted that closes a row. `LF.04` still comes first.
**GPU today.** Kaggle's GPU week is spent by the shared account (our budget
file cannot see that — trust Kaggle's refusal, not the file). Colab holds
`D1.0` (ONE session per account — `colab ls` first; probe with
`/data/venvs/colab/bin/colab --auth adc`). "In flight" needs a live pid.
**CPU today.** The 57,600 s day ceiling is shared by everything; 35,852 s were
used by 08:40. Until `LF.04` has a verdict, **no new detached learner-side
pilot** (ruling 3): the CPU goes to registered runs (J1, J7) and J6's world race.

---

## JACK — build the creature, his world and his curriculum (max 7 open, ranked)

*Order of work. CPU: **J1 step 1** every slot the box is quiet (else S1's
first step) → **J7** harvest → **J6** (run `W.1`, build `W.7`, race the needs
clock) → **J8** → J5 → J3. GPU: **J2** — read the log, act only on an exit.
A CPU order and a GPU order may be advanced in the same slot.*

### J1 — Does living teach him? (`LF.04`) — the creature gate, 48 h unrun
**Why.** `LF.03` shows he lives, dies and resumes; nothing shows he gets
BETTER. `LF.04` (learner vs his own frozen twin, two controls) is registered
and has never run, for two reasons, both now ruled.
**Do.** (1) **`LC.02` first.** It went PASS → FAIL on 11-12 10:52 while a
detached race loaded the box; it is a wall-clock claim, `LF.04` depends on it,
and it blocks 19. Re-run it on a quiet box (`uptime` load < 1.5, no detached
job of ours alive; say the load in the commit). PASS → the FAIL was contention:
add a precondition to the spec that returns VOID (not FAIL) above a stated
load, with a planted-load test. FAIL when quiet → the core got slower: bisect
`cores.py`/`survival.py` since `d5d26b0`'s last PASS and write what you find.
(2) **`LF.04` and `BD.01` run on Kaggle CPU** (ruling 2; SYSTEM S1 builds the
route, `LF.04` is its first customer). Do not re-budget them to `cpu<2h`.
If it FAILs, that is the finding — diagnose, do not soften.
serves: M1→M7 · philosophy 1 (learns by living), 4.
**done when:** `LC.02` has a quiet-box verdict and `LF.04` has a verdict.
**Not waiting.** S1 has no code yet and is `LF.04`'s only route: a JACK-lane
worker who finds the box loud does S1's first step instead of idling. Try
`LC.02` FIRST in every slot, before launching anything detached; if no quiet
minute comes by the 11-14 sitting the Director rules on the precondition.
state: J1(1) DONE (4b67b0b, e9e07c2). J1(2) route DONE (1913430, 93b463b). LF.04 <- LF.03 <- LF.02 <- T6.03 <- T2.10 FAIL (ruling pending, INBOX 19:55).
11-14 00:08 LC.02 re-run (3f3e121): VOID, foreign_cores_max 1.92 > 1.0 at midnight (another tenant's stt proof job + root pmdalinux, ~1 core each);
contended reading 4/5 arms clear (ppo-lp 7.89, ppo-needs 7.22, wm-efe 6.19, dreamer-xs 6.01; wm-latent 4.97). XL.00 needs LC.02 -> not run.
Next: LC.02 again when `top` shows no foreign busy process (not just load<1.5) AND heat_race has exited; then XL.00; then per T2.10 ruling.

### J2 — The control-path contest (`D1.0`) — attempt 4 in flight on Colab
**Why.** `T2.01` (he moves better than random) is the biggest blocker in the
ladder: frees 25, red since Aug 12, waiting on which controller holds the
seat. `D1.0` attempt 3 computed 1.49 GPU-h and lost it in the download; S5
(`d9c94ba`) now recovers results from stdout, and Colab is back.
**Do.** When `UB.10`'s session ends (`colab ls` empty): dispatch `D1.0` via
`scripts/dispatch.sh` unchanged (13 units of ≤ 1.8 h, resume between units).
Read its log each slot; act only on an exit. On a verdict: seat the winner BY
VERDICT, then re-run `T2.01` under it.
serves: M3, M4 · END-FIT row 1 (human-like body) · philosophy 3, 7 · objective 3 (free GPU).
On a verdict, write in the seat row which END-FIT rows the winner leaves reachable (fast loop inside one brain; hands).
**done when:** `D1.0` has a verdict, or an ERROR with a named cause that is
not a lost download.
state: 11-14 00:12 Colab T4 assigns with `colab --auth adc new`. 11-14 01:15 T1.14 PASS on Kaggle T4 (9d24f8d; 3673 GPU-s, heldout_cv 17.3%).
11-14 ~01:55 D1.0 DISPATCHED unchanged, `dispatch.sh D1.0 --projected-hours 25.05` (D56 default: smallest-first, quota refusal final, overflow Colab);
pre-flight CLEAR (Kaggle W45 used 1.02h, 28.98h left); watcher pid 3422501, log /data/tmp/dispatch_d1_0.log.
Next: read that log each slot, act only on an exit; never kill the watcher. On a verdict seat the winner BY VERDICT, re-run T2.01, write END-FIT rows.

### J6 — W0's red world rules: one clock (`W.1`, `W.2`), then `NE.01`
**Why.** `W.1`(b) and `W.2`(c) are one fault, ruled today (ruling 1): **W0 has
ONE declared clock, k = 72.** Thermal already obeys it (TAU_T 240 s =
4.74 h / 72). Hunger and thirst do not: they run 14.8× and 6.3× past it, so he
dies of thirst in 570 s — under half a world-day, before a night can teach him
anything. That may be why living does not teach him.
**Do.** (a) Re-register `W.1`(b) and `W.7`'s control against the DECLARED
clock: a channel whose rate differs from k = 72 by more than the existing
tolerance must fail; shipped thermal should pass, shipped needs must fail
`W.7`. Bars byte-unmoved. (b) Race the needs clock, no world file edited until
it wins: playground arm B_E, B_W at k = 72 (starve 25,200 s, thirst 3,600 s)
vs shipped, on `NE.01`'s probes and `W.2`'s checks, seeds 0–4, rule declared
first: the k = 72 arm ships if it passes `W.2`(c) and is no worse than shipped
on every `NE.01` clause. Report decisions-per-life for both (the cost). A
race-won edit may ship (ruling 1 lifts D43's hold for exactly that); re-run
`LF.03` after. (c) Then the band-geometry arm and the drop probe for
`ok_spread`.
serves: M2 · END-FIT row 4 (late payoffs need a world whose deadlines allow them) · philosophy 1, 3, 4.
**done when:** `W.1`, `W.2`, `NE.01` each have a new verdict with every
formerly-red clause explained.
state: (a) DONE (fa0f7fc, 98134d7, 2685dd9). (b) k72 RACE: does NOT ship (e4c10cd). (c) HEAT RACE: NONE SHIPS (af9b980; NEEDS_CLOCK_RACE.md).
11-14 03:24 WATER RACE LAUNCHED (3d7c0a0): swp_w72 (sweat_paid+B_W 1/3600), swp_k72 (+full k72) vs shipped, heat-race rule; tally fix in; apparatus 2/2 exact (W.2+NE.01 s0). Log /data/tmp/water_race.log.
FINDING (40d47e4): sweat makes heat death UNREACHABLE: W.2(c) c_hot_reachable 1->0 on 10/10 sweat seeds (roof 40.33->37.34 C), 0/60 heat deaths.
swp_k72 has both clocks in band (thirst 0.97, hunger 1.30) and still fails (c) on that alone -> no sweating arm can pass W.2(c) as registered.
Next: harvest `python -m experiments.water_clock_race tally` (~80 min); then declare sweat_paid+k72+a REAL heat hazard (hot sun/fire/humidity), roof route re-aimed.

### J7 — The deeper world (W1): run the two repaired specs
**Why.** W0 is too shallow for thinking ahead to pay (`DP.05`); fast/slow
cannot be raced until it does. Both repairs are committed and unrun.
**Do.** `W1.04` (float-clock repair `8495d56`), then `W1.00` attempt 2 (n = 20,
seeds 300–319, ~65 min on 3 workers, `launch_detached.sh`), one at a time.
Then register `DP.05`'s question in W1.
serves: M2 · END-FIT row 4 · philosophy 5.
**done when:** both have new verdicts and `DP.05`-in-W1 is registered or run.
state: 11-13 15:45 W1.00 att 3 FAIL (b) at n=20 (edd3e61). 11-14 02:35 W1.02 att 10 PASS (quantum ratio 0.0083; W1.01/W1.03 may quote 0.0138).
11-14 02:40 W1.04 PROVEN UNPASSABLE (93f4040, docs/research/W1_04_UNPASSABLE.md): lives are sequential in ONE 5600 s rollout, V4 needs >=20
completed lives, so (c)'s p5 is capped at 266.6 s (bar 5600) in ANY world; and (c) gates W1.01's passive arm, which must die.
=> racing the energy clock cannot green W1.04 as registered. Next: re-register (c) on independent fresh lives (>=20/arm-seed, censored
at 5600 s), gated on W1.01-oracle + W1.02-repeat, bar + control unchanged (INBOX: passive-arm ruling); then race W1's energy clock; DP.05-in-W1.

### J8 — The hand-over gate: a Jack trained here still learns on another machine
**Why.** Owner 11-13: the finished Jack is "trained here, handed to people,
still learning on their machines"; END-FIT (top of `CHAMPIONS.md`) lists the
hand-over gate as "none registered". Resume is proven bit-exact on THIS box
only (`LF.02`). No winner may be adopted that closes this row, so the row
needs a test that can fail.
**Do.** Register ONE spec (next free `LF.` id), exempt from repair-before-
register by this order: train the seated core here for N lives, save one
checkpoint (weights + diary + ring memory + optimiser state); load it on a
different machine (a Kaggle CPU kernel through S1 — other OS image, other
CPU). Claim, ≥ 3 seeds: (a) the first frozen exam there matches the last one
here within the seed noise `T1.14` publishes; (b) after M more lives there he
beats his own frozen hand-over twin. Controls that must fail: a checkpoint
with the diary and optimiser stripped must fail (a) or (b); a fresh-weights
twin must fail (a). Size N, M from `LF.03`/`LF.04`'s rows; bars declared
before any run. Then, same commit or next: change `cores.py:75`'s comment —
the `language` slot is "words through his own learned ear; EMPTY today", not
"the frozen text tower" (card line 2; no dimension changes).
serves: M7, M8 · END-FIT row 2 · philosophy 1, 2, 7.
**done when:** the spec is registered with both controls and queued on S1
behind `LF.04`; `cores.py` no longer names a frozen tower inside him.
state: LF.06 REGISTERED 11-13 12:30; experiments/tests/lf_06_handover.py BUILT (e7b45bd; strip/fresh controls, selftest OK). Budget CPU_DAYS.
11-13 14:25 pilot HARVESTED: TOL_S 20.0 (= floor; sd_exam 1.31, n 10). Exams 597.6-601.2 s, all energy deaths, ate<=1; both pilot
salts byte-identical (salt changed nothing; suspect L.k=HORIZON pins std; n is really 5). ARM/x86 gap 18.2 s (LF.04 smoke) = 0.91 of TOL.
LF.06 depends on LF.03 (J1's T2.10 block). 11-13 16:25 MEASURED the hand-over file: 19.6 MB (13.4 MB gz), ~all learner
(861,545 params + Adam + ring); world/needs/diary < 16 KB. gpu.submit has NO input channel (repo clone + fetch-out only) -> INBOX.
11-14 00:40 cores.py:75 comment fixed (389ffca): language = 'words through his own learned ear; EMPTY today'; only LC.02 staled, re-stamped by amend --doc-only (prose_only_delta).
Next: run() with here/there roles once S1 names the channel; OR reverse legs (no input channel needed: 'here' = Kaggle kernel trains + fetch-out, 'there' = this box) - Director's call; LF.03 dep still blocks.

### J5 — The curriculum cook-off (`CR.01`)
**Why.** Owner 11-11: the curriculum is "as important as jack himself". The
research page and the VACANT seat exist; six arms are built and sized at
~39 CPU-h — too big for this box.
**Do.** Register `CR.01` now exactly as sized (null, hand-staged, shuffled
control, learning-progress, frontier, matched-meal control; 3 claim seeds ≠
88/89; margin .20 fed-fraction, N = 47 exam lives per arm-seed), budget class
= the Kaggle-CPU class S1 creates. It runs second on S1, after `LF.04`. The
teacher-proposed arm joins later, behind the judge gate. If J6 ships the
k = 72 needs clock first, re-pilot the null's spread before the claim seeds.
serves: M2 · philosophy 3, 5.
**done when:** `CR.01` is registered with both controls and has a verdict.
state: (1)+(2) DONE 11-13 (CURRICULUM.md). CR.00 pilot s89: exam fed null 5/8, staged 2/8, shuffled 0/9 (anecdote); arms built 05:35.
11-13 17:20 CR.01 REGISTERED (KAGGLE_CPU, dep LC.02, seeds 130-132; CHAMPIONS arena NONE -> CR.01): TRAIN 40k, N_EXAM 47 lives (cap 200k),
X wins iff on every seed fed(X)-fed(null)>=.20 AND -fed(matched)>=.20 (staged also vs shuffled), pooled median span >= null's; VOID if
null 0/47 or 47/47 or < 47 exam lives. If J6 ships a W0 edit first: re-pilot null on s89, re-declare N by the same 2-SE rule, MARGIN fixed.
11-13 17:35 RUNNER BUILT (777b150, tests/cr_01_curriculum_cookoff.py): cr00_pilot exam-by-lives mode; selftest 13/13; smoke ~47 ms/dec
-> ~1.8 h per arm-seed, 2 per Kaggle kernel, 9 kernels. Next: queue CR.01 on S1 behind LF.04 (`run CR.01` dispatches its own kernels).

### J3 — The learning core at 3× scale (owner priority 11-12)
**Why.** Owner: "when will we run it bigger scale against the others". The
Learning-core seat rests on a 1× screen; `LC.07` (~10×) is priced at
200–270 Kaggle-CPU session-hours and stays untouched.
**Do.** Register ONE new spec (next free `LC.` id), its own claim, not a
shrunk `LC.07`: all five core arms plus the ring working memory, ~3× the
screen's experience, ≥ 3 seeds, the collapse diagnostic of
`docs/research/LEARNING_CORE.md` §5.4 implemented, weights persisted, a
control that must fail. Pilot one arm-seed on Kaggle CPU to price it; write
sessions needed in `state:`.
state: LC.09 REGISTERED 17:45 (+re-regs 20:15, 20:40); all 7 arms built (d80d85c); control (a) wipe_at_death built (c014834).
11-13 23:00 PRICING ROUTE BUILT (4969084): LC.02 now times wm-dyn + wm-latent+ring through their own loop (_run_rec: step+efference,
stored state per update); priced, NOT in LC.02's gate; lc_09 _borrow_lc09 adds their ratios. Contended smoke (foreign 1.1-1.7, not a
price): wm-dyn 7.99 @1/8; ring 3.43 @1/8 (may be EXCLUDED by the floor). LC.02 is STALE by this commit (LC.03/LC.09 refuse until re-run).
Next: re-run LC.02 on a quiet box once the day ceiling admits (>00:00 UTC), read the challengers' ratios; then pilot s89 on Kaggle CPU.

**Next up, in this order, as slots free:**
1. **The diary in his decisions** (opens when `LF.04` has a verdict): core
   with retrieved diary rows in its input vs without; wiped-diary twin must lose (M7).
2. **Inside/around sweep** (ruling 11-11): re-credit `LG.00`, `HR.4`, `T3.08`,
   `LG.10/12/13/14`, `ME.11.C/D/E` as parent-rig results; delete the paid
   `AudioListener._transcribe_api`; re-run `LG.00` with the learned core in the loop.
3. **Ring memory in the creature**: `ME.8`/`LF.02` with the seated ring in place of the GRU.
4. **Learned ear** vs a no-ASR null (M5/M6); `HR.1`, `HR.5` repaired under it.
5. **Judge admission gate** (FIELD_WATCH N1) — opens with J5's teacher arm.
6. **END-FIT's other unregistered gates**, one order each as slots free (ROADMAP M2/M4/M6): W2 — the tool chain
   tree→wood→fire→cooked food and a buildable shelter (build W1's fire and parts first, `W.5`); the grip-rule
   cook-off (hands that hold, carry, strike); the advice gate (told a step he does it sooner than an untold twin;
   a lying adviser loses him); the size cook-off with a real-time-on-a-personal-computer cap.
7. **`UB.11`–`UB.13`, `TA.03`** on the declared trunk A0 (ruling 5 today) — GPU, after `D1.0`.
8. **Sense added later vs from birth** (M5); `XL.01` at power; `PS.02` (stale 92 d, frees 3).

---

## SYSTEM — work on the machine (max 3 open)

### S1 — Free CPU off this box: Kaggle CPU kernels, `LF.04` first
**Why.** Measured 11-12: a CPU-only Kaggle kernel (`enable_gpu:false`) is
accepted with the GPU quota spent; 2 learners per session at 34 dec/s each
(this box: 20.5). The creature gate, the curriculum race and the scale race
are all refused on this box (D20/D39) — this is their only route.
**Do.** In `experiments/gpu.py` + the budget table only: a `kaggle-cpu`
budget class that submits the existing Kaggle job with `enable_gpu:false`,
`pip install mujoco`, the https repo URL, ≤ 2 worker processes, 12 h session
cap, charges NO GPU-hours and NO box CPU-seconds, and lands the ledger row
through the same `JACKRESULT`/`JACKFILE` path S5 built. Resume across
sessions only if `LF.04` does not fit one (it should: ~4 h). Move `LF.04` and
`BD.01` to the class (budget only — claim, seeds, gates, controls byte-same)
and dispatch `LF.04`. Extend `T0.10`/`T0.12` with a CPU-kernel round-trip
that must show 0 GPU-h charged. Net growth ≤ ~80 lines.
serves: M1→M7, M2, M3 (J1, J5, J3) · philosophy 5, 6 (free compute only).
**done when:** `LF.04` has a ledger verdict that came from a Kaggle CPU kernel.
state: pilot 11-12 (owner): 1 proc 28 dec/s, 2 procs 68 total, 4 no gain.
11-13 09:10 step 1 DONE (06b87bd): `submit(prefer="kaggle-cpu")` pushes enable_gpu:false, no accelerator, no torch fix, never
fails over, charges 0 GPU-h; T0.12 PASS with property 15 (control = GPU route fails both). Live: jack-cpu-1794560915 accepted
with GPU quota spent, 47 s, artifact back, 0.0 GPU-h. T0.10 left GPU-only (no GPU quota to re-buy it). Next: `kaggle-cpu` Budget
class in protocol.py + LF.04's job body (pip mujoco, ≤2 procs, JACKRESULT) and dispatch LF.04. Unmeasured: parallel CPU sessions.
11-13 15:40 route DONE by JACK (1913430, 93b463b). Dispatch BLOCKED: LF.04 <- LF.03 stale <- LF.02 <- T6.03 <- T2.10 FAIL (J1 INBOX).

### S3 — A verdict the record cannot re-derive; guards red by design
**Why.** `run verify` exits 2: `W0.PL`'s gate judges from process memory
(`_SEEDS`/`_CTRL`), not the row, and the row stores the MEAN of per-seed
minima while the gate is on the MINIMUM. `BD.01` has the same shape and has no
row yet. `coverage` exits 2 on a by-design counter; `T0.28` p10 reads an empty
decision register as a broken one.
**Do.** (1) `W0.PL` and `BD.01`: record the per-seed `_min`/`_max` the gate
uses and make `_check(m, c)` read only its arguments; re-run `W0.PL` (bars
unmoved). Add to `T0.13` (or `verify`) a check that fails any spec whose gate
reads module state. (2) `coverage`: by-design-red counters out of the exit
code, still printed; FAIL-UNOWNED reads this file. (3) `T0.28` p10 tells empty
from unreadable; a planted unreadable page still fails. (4) After J1's
quiet-box `LC.02` verdict, bank the then-live `pass_on_dead_dependency` and
`unreachable` with `ratchets record --only` — not before (ruling 4).
(5) `coverage.py` COMMITMENTS gains one line for each of the four ends in
GOAL.md's closing section (owner 11-13), each naming its END-FIT gate.
Nothing guarding a threshold, control or seed count leaves an exit code.
serves: philosophy 4, 5 (a truthful machine).
**done when:** on a clean tree `run verify` and `coverage` exit 0 and each
still exits non-zero on a planted PASS-on-FAIL-dependency.
state: 11-13 15:40 (2)+(4) DONE (b7a0364): coverage rc 2 -> 0; claim_dead, empty classes, unreachable/park growth, stale_* printed
only; planted 7th PASS-on-FAIL pair -> rc 2. FAIL-UNOWNED reads ORDERS (4 -> 1, T2.07). Banked podd 6, unreachable 115 (ruling 4).
11-14 (3) DONE (see journal): p10 passes an empty register only if DECIDE blocks parse and every open id has one; planted page with
mangled DECIDE lines still fails. Probe: exp 20/20 clean, ctl 13 fail, _check True. T0.28 stale -> regate. Open: (1) needs quiet box; (5).

### S6 — A video the owner can watch (owner request 11-12)
**Why.** Owner: "a video he can watch". Card line 1: he can be known.
**Do.** A script (`scripts/life_video.py`) that renders OFFLINE from the most
recent recorded life — never inside a scored run — with `experiments/render.py`
(Xvfb + llvmpipe): third-person view of the body in his world, hunger / thirst
/ temperature overlaid, phone-playable H.264 mp4 ≤ 20 MB, written to
`/data/jack-video/latest.mp4` (+ a line saying which life, seed and commit).
The most recent life, whatever it shows — no picking. Run by hand at nice 19
after a life; no cron, no service. Name the path in the order's `state:`.
serves: M8 (he can be watched) · philosophy 1.
**done when:** the mp4 exists, plays, and is of the latest recorded life.
state: open — not started.

**Next up:** (1) **S4** — six standing PASSes gated on a seed AVERAGE that
admits a failing seed (`ME.10`, `NE.00`, `LT.01`, `PG.4`, `PS.02`, `T2.20`):
gate per seed, bars unmoved, start with `ME.10`. (2) `T0.13`'s 13 gate clauses
that can never fire. (3) `run status`'s STEERING reader parses 0 items — aim
it at this file or delete it. (4) `T0.33` `slack_predicts_refusal` reads False. (5) the dispatch
BUDGET line prints the shared Kaggle account's quota as 'remaining 30 h' when Kaggle refuses — print UNKNOWN unless probed. (6) D52 fired:
`field_watch.sh` retries within its ISO week on a gate refusal, one completed sweep per week at most.

---

## INBOX — one line each, workers add, the Director empties
- 11-14 02:40 JACK J7: W1.04 (c) can never PASS (max p5 266.6 s vs 5600 s, 93f4040). Re-registration drops W1.01's PASSIVE arm from (c) (its claim is that it dies) — strengthen or loosen? Default: drop it, measure (c) on independent lives, bar unchanged.

- 11-14 02:35 JACK: W1.02's HASH-SALT DIFFERENTIAL (reporting-only) timed out (TimeoutExpired) and took ~12 min of 3 workers for a 0.22 s test; it costs the box and still reports nothing.
- (JACK 11-14 00:30) Supersedes 20:50: Colab needs NO owner re-auth - `colab --auth adc new -s p --gpu T4` assigned a T4 at 00:12 (plain `colab sessions` without `--auth adc` still prints the OAuth URL; gpu.py always passes adc). Kaggle GPU quota also reset by 00:13 (T1.14 running). LC.02 VOID at midnight from foreign load (another tenant's stt proof job + root pmdalinux ~1 core each) - a 'quiet box' may not come on its own; Director may want the 11-14 precondition ruling anyway.
- (LAB 11-13 23:55) Sense lookahead (`experiments/sense_lookahead.py`, Deliberation seat) pilot DEAD by its rule: fed births /40 babble 5, want 4, told 6, shuffled control 7 (LAB.md). Nothing to grind. Route note for S1: a kernel body that spawns `python -m experiments.*` children must pass `cwd="/tmp/jack"` — without it the kernel ran 48 s, 8x ModuleNotFoundError, and `submit` reported ok=True with no rows; 8 arm-seeds of 10k physics decisions took 17 min at 2 procs.
- (SYSTEM 11-13 21:15) Guards: decisions/champions exit 0; coverage exits 2 on PASS-ON-DEAD-DEPENDENCY 8 vs banked 6 (new: T2.03, T2.14 <- T1.14 ERROR) — by design, repair is re-runs, not the baseline; also 3 GOAL_UNRUNNABLE_BASELINE entries (DP.02, DP.03, LC.04) live again, and champions parses the END-FIT table (CHAMPIONS.md:51-57) as 6 MALFORMED-ROW seats. All bookkeeping, untouched. S1 still blocked on the T2.10 ruling.
- (JACK 11-13 20:50) Colab CLI asks for OAuth AGAIN (`colab new -s p --gpu T4` prints an accounts.google.com consent URL, aborted): at 19:55 it did not. J2's GPU route needs the owner to re-auth; nothing dispatched.
- (JACK 11-13 20:15) LC.09 guard re-registered as the 19:45 line proposed (c09df29): action gain >= .03 added, Delta_5 and rank clauses kept unmoved. Consequence to know: the SEATED wm-latent (and dreamer-xs, wm-efe) has no next-latent head, so LC.09 will VOID it on this clause - the Learning-core seat cannot be re-earned at 3x without the dynamics repair. Colab T4 assign 'Service Unavailable' 7th today.
- (JACK 11-13 19:55) For the T2.10 ruling (14:25 line): T6.03's runner (t6_03_cross_session.py:268-269) stores 'the owner keeps the <v> <topic> near the <p>' and queries 'what do you remember about the <topic>' — the cue SHARES the content word, i.e. T2.10's templated venue, where T2.10's latest row reads recall@1 1.0 / recency 0.0 (passes). T6.03 never touches the stem-disjoint paraphrase venue that is T2.10's only red conjunct (0.0). So LF.04's chain waits on a capability no link in it measures. Options for the Director: depend T6.03 on T2.10's templated conjuncts only (a re-registration the runner cannot express today), or ADD a paraphrase query to T6.03 (strengthening, keeps the dep honest but leaves it red). J1 does nothing to the deps without a ruling.
- (JACK 11-13 19:55) J2 probe: `colab --auth adc new -s p --gpu T4` NO LONGER stops at the OAuth prompt (17:45 line superseded) — it reaches assign and gets 'Service Unavailable' (6th today); `colab ls` empty, nothing dispatched.
- (JACK 11-13 19:45) LC.09 guard, measured (06a4502): Delta_5 (`delta5_seq`) rolls with z HELD at z_t and scores cosine on state; for wm-dyn it FALLS with training (.015 -> .007-.0085) while the arm's K=5 own-roll action share RISES from ~0 untrained to +.055..+.135 of z variance (beats persistence 3/3). It cannot tell learned dynamics from random-init sensitivity. Proposed re-registration (strengthening): guard = gap_shuf(K=5, own-prediction roll, held-out) > the arm's untrained-init value + declared margin, >= 3 seeds; supersedes the 18:50 line. J3 will do it next unless ruled otherwise.
- (JACK 11-13 18:50) LC.09 guard correction (supersedes my 18:30 "predictable VOID"): the -0.006..-0.008 untrained Delta_5 came from rolling a ZERO start state. Filtered from 16 decisions of real past, the UNTRAINED seated wm-latent reads +0.0147 (eda11bf probe). A random GRU gives Delta_5 > 0, so the ">0" clause cannot fail for an arm that learned nothing. Strengthening wanted before any claim seed: Delta_5 must beat the arm's own untrained init (or a shuffled-action roll) on the same windows. Re-registration, not a loosening.
- (JACK 11-13 18:30) J3 finding for the Learning-core seat (CHAMPIONS): the seated wm-latent (and dreamer-xs, wm-efe) is a world model in name only. `cores.py` rssm() steps the GRU once from a zero state with a zero action, so no weight that reads an action or a past state ever gets gradient (measured 0.0, ef335cf). LC.03 v2 seated a feed-forward encoder. LC.09's ActSWM guard will VOID it; the repair (a real action-conditioned dynamics arm) touches cores.py + survival.py and stales LC.01/02/03/07/08, LF.03, SH.01/02. Ruling wanted only on ORDER (J3 builds it next vs after LF.04), not on whether to race it.
- (JACK 11-13 17:45) J2 probe: `colab new -s p --gpu T4` now stops at an OAuth prompt ('To authorize colab-cli, visit this URL', 'Enter the authorization code') — the Colab CLI token has expired since 16:40; only the owner's browser can re-authorise. Every Colab dispatch fails until then. Nothing dispatched.
- (JACK 11-13 17:20) J2 probe: Colab T4 assign 'Service Unavailable' 5th time today (`colab sessions` empty); nothing dispatched.
- (JACK 11-13 16:40) Colab CLI 0.6.0: `colab ls` lists FILES in a session, not sessions; the session list is `colab sessions` (empty output = none). ORDERS J2 / the 'GPU today' header say `colab ls` first. Also: CPU-runtime assign works while T4 is refused, so a T4 probe is the only honest pre-flight; dispatch.sh could run it before writing any row.
- (JACK 11-13 16:25) J2/J8, two machine gaps. (1) `gpu.submit`/`build_job` can only clone the repo and fetch outputs: no way to SEND a file to a kernel. LF.06's 'there' leg must carry a 19.6 MB (13.4 MB gz) checkpoint made here; committing it to git x3 seeds is wrong. S1 needs an input channel (private Kaggle dataset attached as dataset_sources?). (2) dispatch pre-flight reads our own gpu_budget.json (W45: 0.00 h used) but the Kaggle ACCOUNT is shared and was at its 30.00 h cap; a dispatch that will be refused still writes an ERROR row (T1.14 PASS-stale -> ERROR today). Pre-flight should probe the real quota; also protocol.py in IMPL_DEPS stales GPU certificates (T1.14 -> D1.0) on enum additions. Colab CLI 0.6.0 (0.8.0 out), T4 assign 'Service Unavailable' 3x today.
- (JACK 11-13 15:45) J1 could not take its next step (XL.00 re-run, ~20 min): the day CPU ceiling was already spent (58472/57600 s at 15:20, needs_clock_race + w1_00_attempt4 meters), so `admit` refuses. J1 still waits on the T2.10 ruling below; slot went to J7.
- (JACK 11-13 14:25) J1(2) is blocked one level deeper than the order says: LF.04 <- LF.03 (stale) <- LF.02 (stale) <- T6.03 BLOCKED <- T2.10 FAIL (paraphrase recall@1 0.0, conjunct added 08-31, ME.11's known weakness). LF.02 cannot re-certify save/resume while a language-memory paraphrase claim is red, so the creature gate (and LF.06) waits on T2.10. Ruling wanted: is T6.03->LF.02 a true prerequisite of 'a life can be saved and resumed', or does J1 now include repairing T2.10? Note CHAMPIONS.md:102: the ME.11 arena SETTLED 08-31 with all five scorers INFEASIBLE on paraphrase (even pretrained MiniLM .0667, ineligible inside him anyway), so T2.10 has no known path to green and LF.04 waits indefinitely without a ruling.
- (JACK 11-13 12:30) J8's cores.py:75 comment fix (`language` = 'his own learned ear; EMPTY today') NOT made: impl_sha hashes cores.py raw bytes for 8 specs (LC.01/02/03/07/08, LF.03, SH.01/02), so a comment edit stales LC.02 (needs a quiet box) and LC.03 (~90 core-h) for no code change. Ruling wanted: batch it with the next real cores.py change, or make IMPL_DEPS hashing comment-blind (SYSTEM).
- (JACK 11-13 11:50) W0.DIAG's hash-salt differential child hangs rather than computes: errored TimeoutExpired after ~25 min in the 10:38 chain (spec itself 0.03 s), and a clean re-run's child sat at 1 s CPU for 10 min. Costs every W0.DIAG re-buy ~25 min of lock; also a dirty doc (docs/LAB.md) stamped W0.DIAG dirty and blocked W1.00 even though prose docs are meant to be exempt.
- (LAB 11-13 11:40) Meal memory (salience-kept replay ring, `experiments/meal_memory.py`) pilot ALIVE, thinly: fed .484 vs fifo .370 vs yoked control .376 (s91–93, 9 runs, ~55 CPU-min). Claim gate is declared in the module docstring; the run to grind, once ruling 3 lifts: seeds 94–98 × 3 arms × 13k (`meal_memory run <arm> <seed> 13000`, ~15 runs × 13 min, 2 at a time), then `tally`. Not yet a registered bake-off — Director to say whether it enters under the Consolidation seat (`NE.05`) or the learning core's.
- (JACK 11-13 10:30) LC.02 contention: J1(1)'s PASS branch shipped (e9e07c2, VOID-on-contended-FAIL); the 11-14 ruling below may be moot. Foreign load on a 'quiet' box read up to .75 core in one timed window vs the 1.0 threshold.
- (JACK 11-13 09:45) LC.02: a W0.PL correctness commit (c510b92) silently cost 11% physics throughput; nothing caught it except LC.02 going red a day later under noise. LC.02's T is timed over 40 decisions (5 updates at tr .125), so one stall moves an arm-seed by 2+ sim-s/real-s (dreamer-xs probe 5.31-7.93). For the 11-14 ruling: VOID-above-load, or a longer timed window re-registered.
- (SYSTEM 11-13 09:12) J2: `D1.0` attempt 4 ERROR 09:01 (ledger, committed by SYSTEM): unit 1 ran 1.71 h on Colab then "fetch failed"; unit 2 hit Colab `Service Unavailable`, Kaggle refused (quota). S5 stdout recovery did not save unit 1 — J2 to diagnose. [JACK 10:20: wrong — unit 1 = the SB3 reference, recovered + cached + reused by attempt 5; the ERROR is unit 2's Colab assign.] `W1.04` FAIL 08:48 also committed unread (J7).
- 11-13 13:10 JACK/J7: ran_at-pinned recorded rows read only the LIVE ledger row, so any re-run of the pinned spec VOIDs the pinner forever (W1.00 sat 52 d on T3.06; W1.02 now on W1.00). W1.00 fixed locally (_pinned, 3a2513b); a shared history-aware helper + a check for pins whose row left live would catch the rest (grep RAN_AT in experiments/tests).

## DONE (kept one day)

- **J4 — `UB.10` FAIL (`de563c4`), the one-brain contest's first verdict in 59
  days.** Medians A0 .8719 A1 .8844 A2 .8875 A3 .8844 A4 .8344 (disqualified)
  A5 .8781; A2 beats A0 on every seed but the top arm changes by seed. Fusion
  seat: UNDECIDED, raced, no arm separates; experts+router did not beat one
  brain. Ruling 5: `UB.11`–`13`, `TA.03` run on trunk A0.
- **`WM.02` PASS** (`2103829`): ring memory 0.902 vs GRU 0.579, both controls
  failed; Working-memory seat held BY VERDICT.
- **S5** (`d9c94ba`): a finished Colab run is recovered from stdout when its
  download dies. `T0.12` PASS.
- **Owner's INBOX lines 11-13 (END-FIT, vision)**: placed — J8, Next-up 6,
  S3(5), ROADMAP M2/M3/M4/M6, rulings 6–7.
