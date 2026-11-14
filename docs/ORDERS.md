# ORDERS — what the workers do. Written by the Director.
Director sitting: 2026-11-14 06:45 UTC

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5). Machine problems go under INBOX, one
line each. The Director rewrites this file daily; git keeps the history.
Rulings 1–8 cited here are in `docs/DECISIONS_RESOLVED.md` under 2026-11-14.

**The creature gate is unblocked today.** `LC.02` passed on a quiet box at
06:10 and ruling 1 cuts the edge that held `LF.04` behind a language-memory
claim no link in its chain measures. `LF.04` is 70 h unrun: it comes first,
in every slot, until it is dispatched.
**Frozen until `LF.04` is dispatched (ruling 2):** `cores.py`, `survival.py`,
`w0.py`, `needs.py`, `lc_02*.py`, `lf_0[2-4]*.py`. No lower order edits them —
not a comment, not a race-won world edit. A stale certificate on J1's path
costs a day.
**CPU (ruling 3).** Day ceiling 57,600 s. Detached jobs (races, pilots) may
use at most **34,560 s a day in total and 2 worker processes at once across
all of them**; the rest is reserved for registered runs and the clerk. Before
launching anything detached: sum today's `detached:*` in
`experiments/cpu_budget.json` and `pgrep -f launch_detached`. Used by 06:40
today: 18,591 s (detached 13,447 s).
**GPU.** `D1.0` holds Kaggle (watcher pid 3422501). Colab: one session per
account — list them with `colab --auth adc sessions` (NOT `colab ls`).
**Kaggle CPU kernels** (`prefer="kaggle-cpu"`) cost no box CPU and no GPU
hours: every learner-side run goes there. A kernel body that spawns
`python -m experiments.*` children must pass `cwd="/tmp/jack"`.

---

## JACK — build the creature, his world and his curriculum (max 7 open, ranked)

*Order of work. CPU: **J1** (steps in order; nothing else until `LF.04` is
dispatched) → **J5** step 1 (a dispatch, minutes) → **J6** → **J7** → J8 → J3.
GPU: **J2** — read the log, act only on an exit. A CPU order and a GPU order
may be advanced in the same slot.*

### J1 — Does living teach him? (`LF.04`) — the creature gate, 70 h unrun
**Why.** `LF.03` shows he lives, dies and resumes; nothing shows he gets
BETTER. `LF.04` (learner vs his own frozen twin, two controls) has never run.
Its chain was `LF.04 ← LF.03 ← LF.02 ← T6.03 ← T2.10 FAIL`. `T2.10`'s only red
conjunct is paraphrase recall; `T6.03` stores and queries on the templated
venue and never touches it (ruling 1).
**Do, in this order, one commit each.**
(1) `T6.03`: add a clause measured in its own run, on its own store and cues —
the scorer's recall@1 must beat recency-only by ≥ 0.10 (`T2.10`'s own margin),
else FAIL. Then change its `depends_on` to `["T0.05", "T0.08"]`. `T2.10` is
untouched and stays FAIL. Check with `_experiment`/`_control` before the run.
(2) Re-run `T6.03` → `LF.02` → `XL.00` → `LF.03`, serially, box quiet
(`uptime` < 1.5), nothing detached alive. A red here is a finding: stop, write
it in `state:`, diagnose.
(3) Dispatch `LF.04` on Kaggle CPU (`run LF.04`; route `1913430`). Write the
kernel slug in `state:`. If it FAILs, that is the finding — do not soften.
serves: M1→M7 · philosophy 1 (learns by living), 4.
**done when:** `LF.04` has a ledger verdict from a Kaggle CPU kernel.
state: 11-14 06:10 `LC.02` PASS quiet (att 22, load 0.18, 5/5 arms + both challengers clear). Route built (`1913430`, `93b463b`).
11-14 06:45 Director: ruling 1 cuts `T6.03 → T2.10`. Next: step (1).

### J2 — The control-path contest (`D1.0`) — attempt 5 in flight
**Why.** `T2.01` (he moves better than random) frees 25 specs, red since
Aug 12, waiting on which controller holds the seat.
**Do.** Read `/data/tmp/dispatch_d1_0.log` each slot; act only on an exit;
never kill the watcher. On a verdict: seat the winner BY VERDICT, write in the
seat row which END-FIT rows it leaves reachable (fast loop inside one brain;
hands), re-run `T2.01` under it. On a Kaggle quota refusal: that is final for
the week (D56) — resume the remaining units on Colab.
serves: M3, M4 · END-FIT row 1 · philosophy 3, 7 · objective 3 (free GPU).
**done when:** `D1.0` has a verdict, or an ERROR with a named cause that is
not a lost download.
state: 11-14 01:15 `T1.14` PASS on Kaggle T4 (`9d24f8d`). 01:55 `D1.0` dispatched unchanged (25.05 h projected, 13 units).
04:25 unit 1 done (recovered from stdout), unit 2 running. 06:25 watcher alive 4 h 34 m, no exit.

### J5 — The Kaggle-CPU queue: curriculum cook-off (`CR.01`), then meal memory
**Why.** Owner 11-11: the curriculum is "as important as jack himself".
`CR.01` is registered, built, depends only on `LC.02` (green since 06:10) and
`run next` lists it. It needs no box CPU. Nothing is waiting but the dispatch.
**Do.** (1) `run CR.01` now (it dispatches its own kernels, 2 arm-seeds each,
9 kernels). Dispatch 2 kernels first and write whether Kaggle accepts parallel
CPU sessions; then the rest as it allows. Harvest as they land. (2) Meal
memory (LAB, ALIVE: fed .484 vs fifo .370 vs yoked .376) enters under the
Consolidation seat as a lab cook-off (ruling 6): run its declared grind —
seeds 94–98 × 3 arms × 13k, `meal_memory run <arm> <seed> 13000`, then
`tally` — in Kaggle CPU kernels behind `CR.01`. Gate as declared in the module
docstring, unchanged. Clears → register it against `NE.05`'s seat; dies → say so.
serves: M2, M7 · philosophy 3, 5.
**done when:** `CR.01` has a verdict and the meal-memory tally is written.
state: `CR.01` REGISTERED 11-13 17:20 (seeds 130–132, N 47 exam lives, margin .20); runner built (`777b150`), ~1.8 h per arm-seed.

### J6 — W0's red world rules (`W.2`, `NE.01`)
**Why.** `W.1` is PASS (05:28). `W.2` and `NE.01` are red. Four races (k = 72,
heat, water, humidity) shipped nothing, and two of the reasons were the
test's, not the world's: `NE.01`'s windows were sized on the fast clock, and
its integrity probe measured its own schedule. Both are ruled (ruling 4).
**Do.** (a) Finish re-registering `NE.01`: every window stated as a multiple
of the world's own lethal times (the multiples it has today on the shipped
clock, computed and written BEFORE any run — the shipped world's numbers must
come out byte-same). The integrity probe is option (A), falls until death,
with its planted control (`33a1acc` — accepted). Bars (5/5 random deaths,
spread 0.30, 5 meals) unmoved. (b) Re-run `NE.01` once on the shipped world
(detached, inside ruling 3's cap, after J1 step 2 has had its quiet box).
(c) Re-read the k = 72 and humidity arms against the re-registered spec;
their long `NE.01` legs run in a Kaggle CPU kernel, not on the box. (d) Then
the shared humidity function with `K_DRY` re-derived, raced by the same rule.
No world file changes until an arm wins AND `LF.04` is dispatched.
serves: M2 · END-FIT row 4 · philosophy 1, 3, 4.
**done when:** `W.2` and `NE.01` each have a new verdict with every
formerly-red clause explained.
state: k72 / heat / water / humidity races: NONE ships (`e4c10cd`, `af9b980`, `21188f3`, `b1272c3`). Sweat made heat death unreachable in W0;
hum34 restores `W.2`(c) 5/5 but breaks `W.2`(a) and `W.1`(d) by how the arm was built.
11-14 06:40 option (A) IMPLEMENTED (`33a1acc`, no ledger): drop cycles until death (cap 40) + planted ground-height control, gate `ok_drop_control`.
Probe: ladder s0/s1/s2 spread_i .835/.569/.785 (s0 injury death; s1/s2 thirst kills first, fast clock); control .032/.106.
11-14 06:45 Director: gate-2 ruling is ruling 4 — windows in multiples of lethal times. Next: (a), then (b).

### J7 — The deeper world (W1): re-register `W1.04`, then its energy clock
**Why.** W0 is too shallow for thinking ahead to pay (`DP.05`). `W1.04`(c) is
proven unpassable in any world (`docs/research/W1_04_UNPASSABLE.md`): lives
run in sequence inside one 5600 s rollout, so its 5th-percentile span is
capped at 266.6 s against a 5600 s bar.
**Do.** (1) Re-register `W1.04`(c) (ruling 5): independent fresh lives, ≥ 20
per arm-seed, censored at 5600 s, bar and control unchanged; the passive arm
leaves (c) — `W1.01` already claims it dies, and that claim stays. (2) Run it
(detached, inside ruling 3's cap). (3) Race W1's energy clock for `W1.00`(b).
(4) Register `DP.05`'s question in W1.
serves: M2 · END-FIT row 4 · philosophy 4, 5.
**done when:** `W1.04` and `W1.00` have new verdicts and `DP.05`-in-W1 is registered.
state: 11-13 15:45 `W1.00` att 3 FAIL (b) at n = 20 (`edd3e61`). 11-14 02:35 `W1.02` att 10 PASS. 02:40 `W1.04` unpassable proven (`93f4040`).

### J8 — The hand-over gate (`LF.06`): trained there, still learning here
**Why.** Owner 11-13: the finished Jack is "trained here, handed to people,
still learning on their machines". `LF.06` is registered and built; it stalled
on having no way to SEND a 19.6 MB checkpoint to a kernel.
**Do.** Reverse the legs (ruling 7): "here" = a Kaggle CPU kernel trains and
the checkpoint comes back through the existing fetch-out; "there" = this box
(other OS image, other CPU) runs the first frozen exam and the M further
lives. Claim, seeds, bars and both controls unchanged. No input channel is
built. Runs after `LF.04` is dispatched and `LF.03` is green.
serves: M7, M8 · END-FIT row 2 · philosophy 1, 2, 7.
**done when:** `LF.06` has a verdict.
state: `LF.06` REGISTERED 11-13, runner built (`e7b45bd`), TOL_S 20.0, ARM/x86 gap 18.2 s (0.91 of TOL). `cores.py:75` comment fixed (`389ffca`).
Hand-over file measured: 19.6 MB (13.4 MB gz), nearly all learner (861,545 params + Adam + ring).

### J3 — The learning core at 3× scale (`LC.09`)
**Why.** Owner 11-12: "when will we run it bigger scale against the others".
All 7 arms are built and priced. Finding: the seated core (wm-latent) trains
no weight that reads an action or a past state — a world model in name only.
**Do.** `LC.09` runs as registered (ruling 8): the guard will VOID the seated
arm and that is the finding, not a reason to repair the seat first — wm-dyn is
the action-conditioned challenger already in the race. (1) Rewrite `run()`'s
refusal to item (2) only. (2) Build the Kaggle-CPU pilot path (one arm-seed,
s89, like `cr_01`'s `_JOB`), price sessions, write them in `state:`. (3)
Dispatch behind `CR.01`. No edit to the frozen files; no local learner pilot.
serves: M3 · END-FIT rows 2, 3 · philosophy 3.
**done when:** `LC.09` has a verdict.
state: `LC.09` REGISTERED 11-13 (+2 strengthenings of its guard); arms built (`d80d85c`); wipe-at-death control built (`c014834`).
11-14 06:10 all 7 arms priced from `LC.02`: ppo-needs .5, ppo-lp / dreamer-xs / wm-efe .25, wm-latent .125, wm-dyn .5, ring .125.

**Next up, in this order, as slots free:**
1. **The diary in his decisions** (opens when `LF.04` has a verdict): core
   with retrieved diary rows in its input vs without; wiped-diary twin must lose (M7).
2. **Paraphrase recall** (`T2.10`'s red conjunct): a learned scorer raced
   against the seated one on the certified fixture — the honest owner of that red.
3. **Inside/around sweep** (ruling 11-11): re-credit `LG.00`, `HR.4`, `T3.08`,
   `LG.10/12/13/14`, `ME.11.C/D/E` as parent-rig results; delete the paid
   `AudioListener._transcribe_api`; re-run `LG.00` with the learned core in the loop.
4. **Ring memory in the creature**: `ME.8`/`LF.02` with the seated ring in place of the GRU.
5. **Learned ear** vs a no-ASR null (M5/M6); `HR.1`, `HR.5` repaired under it.
6. **END-FIT's unregistered gates**, one order each: W2 (tree→wood→fire→cooked
   food, shelter; `W.5` fire first — it is runnable and unbuilt); the grip-rule
   cook-off; the advice gate; the size cook-off.
7. **`UB.11`–`UB.13`, `TA.03`** on trunk A0 — GPU, after `D1.0`.
8. Sense added later vs from birth (M5); `XL.01` at power; `PS.02` (frees 3).

---

## SYSTEM — work on the machine (max 3 open)

*Order of work: **S3 step 1** while the box is quiet → **S2** → S6.*

### S3 — A verdict the record cannot re-derive (`W0.PL`, `BD.01`)
**Why.** `run verify` exits 2 for the third day: `W0.PL`'s gate judges from
process memory (`_SEEDS`/`_CTRL`), and its row stores the MEAN of per-seed
minima while the gate is on the MINIMUM. `BD.01` has the same shape.
**Do.** (1) Both specs: record the per-seed `_min`/`_max` the gate uses; make
`_check(m, c)` read only its arguments; re-run `W0.PL` on a quiet box (bars
unmoved). Add to `T0.13` or `verify` a check that fails any spec whose gate
reads module state. (2) `coverage.py` COMMITMENTS gains one line for each of
the four ends in GOAL.md's closing section, each naming its END-FIT gate.
(3) `champions` reads the END-FIT table (`CHAMPIONS.md:51-57`) as six
malformed seats: skip a table that is not the seat table, and fail on a
planted malformed seat row as before.
serves: philosophy 4, 5 (a truthful machine).
**done when:** on a clean tree `run verify` exits 0 and still exits non-zero
on a planted PASS-on-FAIL-dependency; `champions` prints no malformed row.
state: 11-13 coverage rc 2 → 0 (`b7a0364`). 11-14 `T0.28` p10 tells empty from unreadable (PASS, att 22). Open: (1), (2), (3).

### S2 — The CPU day: a cap the machine enforces, and children that hang
**Why.** Two days running the day ceiling was overrun by detached jobs
(59,599 s on 11-13); registered runs and the clerk were refused for six hours
and stale cheap certificates went 19 → 32. Load hit 6.7 on a 4-core box that
serves paying customers.
**Do.** In the existing `admit` / `launch_detached.sh` path only (ruling 3):
(1) refuse a detached launch when today's `detached:*` total plus its
estimate exceeds 34,560 s, or when it would put more than 2 detached worker
processes alive at once; a planted over-cap launch must be refused in a test.
(2) The hash-salt differential child (reporting-only) hung 12 min × 3 workers
on `W1.02` and ~25 min on `W0.DIAG`: give it a timeout of 60 s or 20× the
spec's own run time, whichever is larger, 1 worker, and print TIMED-OUT.
Net growth ≤ ~40 lines.
serves: philosophy 5, 6 (this box serves paying customers).
**done when:** both tests pass and a day ends with `detached:*` ≤ 34,560 s.
state: open — not started.

### S6 — A video the owner can watch (owner request 11-12)
**Why.** Owner: "a video he can watch". Card line 1: he can be known.
**Do.** `scripts/life_video.py` renders OFFLINE from the most recent recorded
life — never inside a scored run — with `experiments/render.py` (Xvfb +
llvmpipe): third-person view, hunger / thirst / temperature overlaid,
phone-playable H.264 mp4 ≤ 20 MB at `/data/jack-video/latest.mp4` (+ a line
naming the life, seed and commit). The most recent life, no picking. Run by
hand at nice 19; no cron, no service.
serves: M8 (he can be watched) · philosophy 1.
**done when:** the mp4 exists, plays, and is of the latest recorded life.
state: open — not started (third day; it is the next SYSTEM slot after S3 step 1).

**Next up:** (1) the GPU week key (`gpu.py:495`) starts Sunday; Kaggle resets
Saturday 00:00 UTC — key it on Kaggle's reset. (2) Dispatch pre-flight prints
the shared account's quota as "remaining" — print UNKNOWN unless probed, and
write no ERROR row on a refusal. (3) **S4** — six PASSes gated on a seed
AVERAGE that admits a failing seed (`ME.10` first). (4) `ran_at`-pinned rows
read only the live ledger row (a shared history-aware helper). (5) `T0.13`'s
13 clauses that can never fire. (6) `protocol.py` in IMPL_DEPS stales GPU
certificates on an enum addition. (7) D52: `field_watch.sh` retries within its
ISO week. (8) `run status`'s STEERING reader parses 0 items.

---

## INBOX — one line each, workers add, the Director empties

## DONE (kept one day)

- **`LC.02` PASS on a quiet box** (`68e1207`, att 22): the 11-12 FAIL and the
  midnight VOID were contention. Unblocks `CR.01`, `LC.03`, `LC.09` pricing.
- **S1 — Kaggle CPU route built** (`06b87bd`, `1913430`, `93b463b`): CPU
  kernels accepted with the GPU quota spent, 0 GPU-h charged. First customer
  is `LF.04` (J1 step 3).
- **`T1.14` PASS** on Kaggle T4 (`9d24f8d`); **`W.1` PASS**, **`T0.28` PASS**
  (regate 05:28); `W1.02` PASS att 10.
- **INBOX of 11-13/14 (24 lines)**: ruled or placed — rulings 1–8, S2, S3(3),
  SYSTEM Next-up 1, 2, 4, 6. Sense-lookahead pilot: dead by its own rule, closed.
