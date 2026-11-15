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
state: DONE (step 3) `bc631a8`: `LF.04` **FAIL** att 2, Kaggle CPU (6 kernels, f90881f). Late learner vs twin +4/-127/-230 s (margin 272.8), vs self +49/+29/-176;
controls held (twinB 0/3, shuffled 0/3); statue 599.9 s. FINDING (`docs/research/LF_04_FAIL.md`): all 120 lives died of energy, late meals 0/3/0. Twin is already
562-569 s, so a non-eater can gain at most 31-38 s: only FORAGING passes (>= 5 floor meals/life at rest, ~14 at 1.5x drain, or 1 apple). s102 collapses
598->158 s: the unbounded `cores.py:542` loss from 11-12, AWR raced then and never seated. Next (Director): seat a bounded loss (re-buys LC.02/LC.08/LF.03),
then a learner that forages on s90 (wm-dyn in LC.09, CR.01). Re-run LF.04 unchanged only when a s90 pilot reaches >= 5 meals/life.

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
state: 11-14 21:27 D1.0 att 8 ERROR (ZeroDivisionError): the reattach of COMPLETE `…135` (= b_split s2, pin-only proven: receipt 28e7b086 = pulled bytes, pin->fc52aca = 398179cc =
local_sha) harvested b_split s2 OK, but JACK_REUSE_KERNEL+ACCEPT_MISMATCH applied to EVERY uncached unit: c_e2e s0-2 and d_mlp s0-2 each "reattached" …135 and cached b_split s2's runs under
their own keys; their arms then had 0 runs -> _stats([]) /0. No false verdict was written. The 6 poisoned files moved to /data/d10_units_quarantine_20261114 (byte-identical to b_split_seed2.json).
Cache now 7 genuine: ref + aprime 0-2 + b_split 0-2. Fresh dispatch of the 6 remaining units 21:29 (10.4 h proj, W45 left 14.61 h): watcher 3902173, first unit c_e2e (attempt …1758124).
Next: read /data/tmp/dispatch_d1_0.log; on a verdict seat BY VERDICT + re-run T2.01. NEVER reattach a multi-unit spec with JACK_REUSE_KERNEL again (INBOX).

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
state: `CR.01` REGISTERED 11-13 (seeds 130–132, N 47, margin .20); runner `777b150`. Kaggle takes 2 parallel CPU sessions, not shared with the GPU queue. 8/9 units cached (unofficial exam fed):
s130 null .255 staged .213 lp .043 frontier .255 shuffled .128 matched .085; s131 null .128 staged .191 lp .106 frontier .043 shuffled .128 matched .085; s132 null .106 staged .064 lp .043 frontier .191 (all exam deaths energy). No WIN possible.
11-15 00:10 HARVESTED 132:lp:frontier (…2332, 8983 s). Guard misjoined again (cited …6945's 7222443b); …2332's own receipt 25904654 = pulled bytes, pin 01a2954->c0db7e6 swap = local e4285e1b exactly, 01a2954..c0db7e6 touches no IMPL_DEPS -> ACCEPT_MISMATCH honest.
Last unit 132:shuffled:matched `jack-cpu-1794694643` (head 84cbc7d, pushed 22:17) RUNNING at 00:10.
Next: when COMPLETE, pull it, prove pin-only vs HEAD, `JACK_REATTACH_ACCEPT_MISMATCH=1 JACK_REUSE_KERNEL=jack-cpu-1794694643 $PY -m experiments.tests.cr_01_curriculum_cookoff prefetch 132:shuffled:matched`; then `run CR.01` (9/9 reused).
(2) 11-14 20:50 runner + CLAIM gate BUILT (this commit, `experiments/meal_memory_kcpu.py`): `meal_memory.tally` grades by the PILOT rule (2/3), not the declared claim gate (4/5 + born_d>=0.7) -> claim_gate verbatim, fixtures 5/5 (each clause fails alone). Smoke 260 dec fifo+meal s94 ok. 15 cells, 3/kernel, ~20 min each. After CR.01's queue: `$PY -m experiments.meal_memory_kcpu kernel fifo:94,meal:94,yoked:94` (…95…98), then `… gate`.

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
state: (a) DONE `e054e0a`. (c) DONE: hum30/hum34/swp_k72 FAIL gate 2. (b) held by the detached cap. (d) declared `4c316fa`, dB built `c9dd457`, kcpu runner `dfc577c`, tally `184c4c1` (needs 30 current cells).
dA/dC not built (pool literal in playground.py:421/886 + ne_01 + needs.py:820 = world-file refactor, Director call). 11-14 19:50 `647d45c`: pre-dispatch smoke of the kernel path (dB:W.2:0, 40 s) caught dB's swallow
gate leaking into W.2's geometry rig (rig_pool 0.0). Fixed so the gate applies only inside substep: rig 1.0, *_pass unchanged, (ii) re-PASS byte-same. Note: rig_thermoneutral_day = 0 is the swp_k72 base's (shipped 1). No cells exist yet, so none are stale.
Next, when a kcpu slot frees (after CR.01's queue): push in-session `> log`: `$PY -m experiments.drink_race_kcpu kernel dB:NE.01:0,shipped:NE.01:0,dB:W.1:0` (3 cells/kernel),
reattach with JACK_REUSE_KERNEL; `... status` lists missing; then `$PY -m experiments.drink_race_tally`.

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
state: 11-13 15:45 `W1.00` att 3 FAIL (b) at n = 20 (`edd3e61`). 11-14 02:40 `W1.04` unpassable proven (`93f4040`).
11-14 10:55 (1) DONE `eeeb1ff`: (c) = 20 fresh worlds/arm-seed (oracle + repeat), censored 5600 s; passive out; fixtures 15/15; pilot s90 lives ~53 s, ctrl censors.
Priced 0.024 s/dec: control alone ~11 core-h -> budget KAGGLE_CPU. FINDING for the Director: (a)'s W1.02 t2c (>=20 completed SEQUENTIAL repeat lives,
mean <= H/3) and (c)'s W1.02 p5 >= H on the SAME process still contradict in any survivable world — W1.04 can only FAIL until that is ruled.
Next: (2) `run W1.04` on Kaggle CPU once the kcpu lock frees (LF.04 ~17:30); expected FAIL (c) at ~53 s — the honest reading of today's world.

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
state: 11-14 13:30 legs REVERSED + run() BUILT `2f632ba` (re-registered before any run: budget KAGGLE_CPU, TOL_S 20.0 fixed). Kernel per seed = 10 lives + file + exam_here,
fetched + sha-checked; box = exam, learner, frozen twin, strip(+own exam), exam_fresh, <=2 niced children billed `LF.06:there:<role>`. Smoke s110 N=M=K=1: all roles ok,
exam here=there=strip 45.8 s, twin frozen (opt 464), learner trains (464->861). FINDING: fresh-init exam 577.6 s vs s90 pilot's trained exams 597.6-601.2 -> control 2 may not fail (a).
Box price ~3.9k s billed/seed (~11.7k s for 3) -> needs a fresh CPU day (today 42k/57.6k used). Next: `launch_detached.sh /data/tmp/lf06_run.log $PY -m experiments.run LF.06`
early on a quiet day, after LF.04's verdict and behind CR.01's kcpu slots (it takes 1 kcpu session/seed, serial). A control-2 FAIL is the finding — do not soften.

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
11-14 12:40 (1) DONE + (2) BUILT `8ee2edd`: run() now refuses only on the missing kcpu claim path; `… lc_09_learning_core_3x pilot` = Kaggle-CPU kernel,
s89, 6k dec × wm-latent/wm-dyn/ring/ppo-needs + frozen twin, guard on -> /data/tmp/lc09_pilot.json. Box price (LC.02 T×5 dec/s): one 1.2M arm-seed = 9.6–12.3 h
(ppo-needs 12.3 > Kaggle's 12 h) before twin/nulls -> the claim path likely needs per-session checkpoint chaining. NOT dispatched (CPU-session concurrency unknown; LF.04 in flight).
Next: after LF.04's verdict, `launch_detached.sh /data/tmp/lc09_pilot.log $PY -m experiments.tests.lc_09_learning_core_3x pilot`; write kernel h_at_1_2M here.

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
state: 11-14 09:10 (1) code DONE `abb707f`: W0.PL/BD.01 write per-seed min/sums on the row, `_check` reads args only (0/8000 old≠new); verify probe D counts gates reading mutated module state (live hits 0, planted fixture).
(3) DONE `133c2ac`: END-FIT table skipped, 35 seats identical, fixture plants foreign table + narrow row. (2) DONE `8fadd25`: 4 GOAL ends in COMMITMENTS.
Open: W0.PL re-run (90 s; LF.03 landed; 21:07 refused — CPU day at 0, fixed by `d5a5ab6` from 11-15) → verify's last red (old row judged by its issuing gate = VOID) clears. T0.18/T0.29/T0.21 stale by dep → regate.

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
(3) `docs/LAB.md` dirt stamped `W0.DIAG` dirty and blocked `W1.00` (11-13):
add it to the prose-exempt set (`PROSE_DOCS`); a dirty code file must still stamp.
Net growth ≤ ~40 lines.
serves: philosophy 5, 6 (this box serves paying customers).
**done when:** both tests pass and a day ends with `detached:*` ≤ 34,560 s.
state: 11-14 21:40 (1)(2)(3) BUILT `d95d569`; `d5a5ab6`: a wrapped job whose session uses < 0.10 cores over a heartbeat bills its CPU,
not wall (pollers were ~30k of 11-14's 65k s; 2 s sleep now bills 0.08 s, T0.34 clause 9, fails on old code), and all-kcpu specs take
`/tmp/jack-ladder-kcpu.lock` + count remote. Open: a day ending ≤ 34,560 s (first chance 11-15); idle pollers still count toward
DETACHED_MAX_ALIVE; cpu_budget/run/protocol edits stale T0 certs → regate after midnight (day budget 0 now).

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

- 11-14 21:40 JACK: `JACK_REUSE_KERNEL` (+ACCEPT_MISMATCH) leaks to every `submit()` of a multi-unit spec: D1.0 att 8 reattached one kernel (b_split s2) for 7 different unit bodies and cached its runs under c_e2e/d_mlp keys (quarantined to /data/d10_units_quarantine_20261114). Each attempt row recorded the slug with the new body's sha, so the mismatch chain looked tolerated. Scope reuse to the first submit, or refuse a second body on the same slug.
- 11-14 SYSTEM: `coverage` exits 2 again at HEAD (bookkeeping only: GEN.06 new unrunnable citation, SM.02→SM.03 release pair, UNREACHABLE/PASS-ON-DEAD baselines lagging) — not touched. Real finding from S3(2): GOAL ends "advised not commanded" and "tree-fire-shelter world" have NO live claim spec (CLAIM-DEAD) — the advice gate and W2/`W.5` in JACK Next-up 6.

- 11-14 JACK: multi-call Kaggle-CPU specs re-clone the CURRENT HEAD on every call (LF.04 calls ran on 4 different heads) — a code commit mid-run would silently mix versions inside one verdict; pin calls 2..N to call 1's head. Harmless for LF.04 today (code paths unchanged).
- 11-14 JACK: the GPU push guard (`gpu.assert_ref_is_current`) runs on EVERY call of a multi-call remote spec, and `is_code_dirt` counts `docs/LAB.md` and `docs/ORDERS.md` as code — one lane's uncommitted LAB.md at 11:33 killed LF.04 at call 6/6 (2 h of kcpu) AND D1.0 at unit 7/13 (≈10 GPU-h of units survive only via D1.0's own cache). Fix: guard once per spec run (pin calls to call 1's head, cf. the pin-head line) and/or class LAB.md/ORDERS.md as prose. Both re-dispatched 12:08.

- 11-14 15:25 JACK: consequence of the poller-billing line above: `cpu_budget admit` REFUSES every detached launch today (48,545 s > 34,560 s), so J5's next `CR.01` prefetch (pure Kaggle polling, ~0 box CPU) cannot start. Separately, D1.0's GPU kernel `jack-ladder-1794658135` is still QUEUED at 15:23 (since 12:08) with NO kcpu kernel running, so the queue is not a session cap it shares with CPU kernels. Possibly the shared account's GPU quota.
- 11-14 16:10 JACK: `gpu.reattach_code_check` join 2 misjoins concurrent kernels: the slug epoch is int(t0), so it can sit a fraction of a second BEFORE its own receipt ts (`0 <= epoch - ts` fails -> jack-cpu-1794672505 reads "unverifiable"), and the 600 s window accepts an earlier kernel's receipt (jack-cpu-1794672525 joins …505's -> false "mismatch"). Fix: allow ts within ±2 s of epoch and pick the nearest receipt, or record the slug in the attempt receipt.
- 11-14 17:15 JACK: (1) a SYSTEM edit to `experiments/protocol.py` (S2 `d95d569`, a salt-child timeout) staled `T1.14`, a 1 GPU-h certificate bought at 01:15, which BLOCKED D1.0's free reattach — every GPU cert listing protocol.py in IMPL_DEPS is re-bought on each apparatus edit; (2) `gpu.submit`'s watcher timeout (est×1.25+1800 s) counts Kaggle QUEUE time, so a 4 h queue killed a kernel's watcher 17 min after it started RUNNING; (3) a `JACK_REUSE_KERNEL` dispatch skips pre-flight for the WHOLE run, so D1.0's later FRESH units go out with no budget check.

- 11-14 19:15 JACK: every reattach after ANY push is refused as a code mismatch, because the kernel pins the dispatch HEAD sha (by design) and `_kernel_sha256` hashes that pin. Each refusal also writes an ERROR ledger row (T1.14 att 7). The proof is mechanical: pull the kernel, swap the pin to HEAD, compare sha, and check `git diff pin..HEAD` against IMPL_DEPS. `reattach_code_check` could do this itself and drop the hand-set ACCEPT_MISMATCH.

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
