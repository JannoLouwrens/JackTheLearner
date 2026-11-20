# ORDERS — what the workers do. Written by the Director.
Director sitting: 2026-11-20 12:27 UTC

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5, **each ≤ 4 lines of text — replace,
do not append; history is in git**). One state+journal commit per slot.
Machine problems go under INBOX, one line each. Rulings 57–63 cited here are
in `docs/DECISIONS_RESOLVED.md` under 2026-11-20 12:27.

**`docs/PLAN.md` binds every order (owner, 11-15).** Research → combine →
reproduce → test the last 10% → put it in him. We are in **stage 1: the
instrument.** `docs/research/PUBLISHED_PRIORS.md` is the reference — read
your order's section (only that section) at the start of every slot.
**Where stage 1 stands (11-20).** Pilot 2 is home, 40 of 40 runs, read by
its frozen rule: **no learner ate.** Every learning arm on every seed ends
at 0.08–0.16 meals a life — BELOW the same brain never learning (0.82) and
below random held pushes (0.95) — in the world where moving is free too. So
the world's cures are not the missing piece; the suspect is the learner's
own setting: a fixed entropy bonus worth 15–18× the task reward (§19).
**Ruling 57: E2c gets `ent_coef="auto"` twins before its last seeds;
ruling 58: nobody launches another learner run in his world outside J15,
J14 and the forge's frozen round 1.** His eye and motor are body-frame
(ruling 49); stage 1 and 2 are judged at the entry rung (ruling 51).
**Every new run in his world passes** `eye_frame="body"`,
`motor_frame="body"`, `food_floor_only=True`, `lazy_contact_audio=True`,
`native_substep=True`. The integrator is not switched (ruling 40).
**Stopped (rulings 10, 20, 23, 36):** races between home-made learners
outside J3; forager cells on the 16-ray compass eye; world-rule races;
`D1.0`, `T2.01`, smell runs; new specs with no published prior. A question
with no prior is a `RESEARCH:` line in INBOX.
**Kaggle CPU: 5 sessions (measured), RANKED (ruling 60). No session sits
idle while any ready kernel exists** — the caps below bind only while a
lower-ranked READY kernel waits. (0) any kernel estimated < 1 h; (1) J15's
E2c kernels, auto twins first — up to 3; (2) J3's one last kernel; (3) J10's
segments — up to 2; (4) lane `forge` — up to 2, only on a rule the Director
froze; (5) brain B row 5 (`brain_b_dmc.py`, ~2 h). Every JACK slot's first
act: `kaggle kernels list`/`status` — harvest what is home, refill what is
free. `W1.04` is PARKED to 11-25 (ruling 41).
**The box (rulings 32, 53).** Day ceiling 57,600 CPU-s, ≤ 2 computing
detached jobs. Whole run < 15 CPU-min → box, nice 19. 15–60 CPU-min → box
only when all 5 Kaggle sessions are held AND load < 4, one worker. > 60 →
Kaggle or it waits. The box is shared (never touch other projects). A
launch refused on load retries inside its slot (bounded), then the NEXT slot
takes it first. Before launching, `pgrep -af` the same command line — never
launch a kernel that already has a live watcher.
**Kaggle GPU: 30 h open Sat 11-21 00:00 UTC**, order in J14. **Kaggle TPU:
20 h a week, unused so far — S12 builds the door.**
**`/` has 4.8 GB free, workers refuse under 3:** scratch, clones and caches
under `/data` only (`UV_CACHE_DIR`, `PIP_CACHE_DIR`, `XDG_CACHE_HOME` under
`/data/caches/`, `--only-binary=:all:`).
**World files** (`w0.py`, `needs.py`, `survival.py`, `drives.py`): changed
only through `jack_env.py` options whose default is today's world.
**Known-answer rows (ruling 33):** seeds AND at most one extension stated
BEFORE the first seed runs. A miss after that is a miss.
**Receipts:** the slot that launches a kernel commits
`experiments/gpu_submissions.jsonl` and `cpu_budget.json` by name.

---

## JACK — build the creature, his world and his curriculum (max 7 open, ranked)

*Order of work: **J15** → **J14 (from Sat 00:00 it goes FIRST until A1 is
launched)** → **J3** → **J10** → **J16**. A slot waiting on a kernel
advances the next order. A slot that can advance none takes "Next up".*

### J15 — E2c: can a plain learner eat when it is handed the direction — and was the entropy setting the wall?
part: Learning to eat · stage: 1 ("a learner given the direction to food"; ready-list "B piloted in his world") · step: 4 (pilot) · prior: §19 (read it whole), §17; rulings 50, 56, 57
prior: §19 — a FIXED `ent_coef` 0.1 on 8 action dims has no published precedent; it pays ≈ 0.54 a living decision ≈ one meal's discounted value for merely staying alive (Yu 2022: fixed α at 2.8× the task reward "never wants to reach the terminal state"). The one published SAC forager (ML-Agents FoodCollector) LEARNS α and took 2M steps. Ours: fixed 0.1, 300k. B's row 1b reproduced WITH `auto`.
prior: first reliable reach expected 50k–200k (inference, reacher-class SAC); survival by repeated eating at 300k is on NO record — a miss at 300k is weak evidence. Learnable by a plain learner: not yet shown (pilot 2: 0 of 25 learner runs).
**RULED (57) — the auto twins.** The frozen fixed-α arms (E-food, E-zero,
E-random_point × 801–805) finish as frozen: e2, e3 still run, nothing is
dropped or re-read. ADDED before any of their seeds exists: `E-food-auto`
and `E-zero-auto` = the same arms with `ent_coef="auto"` (SB3 default
target −8), seeds 801–805, trace on.
**Do.** (1) In `pilot_b2.py` add the two arms in a SEPARATE dict and
separate kernels `a0..a2` — `e2c_groups()` for e0–e3 must stay
byte-identical (test it), and the lives of a fixed-α arm must be unchanged
(planted test: `ent_coef` reaches SAC; the fixed arm's first 2k decisions
identical before/after the edit). (2) Write into the E2c rule text BEFORE
launching a0: the two arms, read by the same frozen readings and §3b's
corrected trace reading, and this table — auto-food eats (reading SUSTAINED
or late meals/life > K-lr0's 0.82 on ≥ 4/5) while fixed-food does not → the
setting was the wall, every later B run in his world uses `auto`; both food
arms miss → weak evidence (300k), the next question is run length, for the
forge's round 1; a zero-control arm that eats as well as its food arm →
the handed sense carries nothing, say so. (3) Launch a0, a1, a2, then e2,
e3 as sessions free (rank 1). (4) Harvest; one row per arm in
`WORLD_CARD.md` with its control; ONE INBOX line: the reading per arm.
serves: M3 · PLAN stage 1 "proof that the task can be learned with his senses" · philosophy 3, 4.
**done when:** E2c's five arm rows (3 fixed, 2 auto) are in `WORLD_CARD.md`, each beside its control, and the table's branch is named.
state: 11-20 19:25 READING BUILT `7a3a4f6e` (after (1)+(2) `8c551f77`): `pilot_b2 e2c-readings` = rule E2c's reading + ruling 57's table, refuses a partial book; seams PRINTED not chosen ("on ≥4/5" binding SUSTAINED → both computed, SEAM; no branch fires → NO BRANCH). Planted 8/8 PASS. Peek (complete arms): E-food-auto sustained 3/5, late meals/life > 0.82 on 4/5 (… 0.84) → EATS both ways; E-zero-auto 0/0. E-food 2/3 home sustained → "both eat" = the NO BRANCH hole is the likely outcome.
state: ALL 25 LAUNCHED: e0 `jack-cpu-1795176676`, e1 `…176785` (12:11), a0 `…180195`, a1 `…180238`, a2 `…180505` (13:09–13:16), e2 `…192704` (16:38), e3 `…194484` (17:08, rc 0). HOME: a2 (16:02), e0 (16:59) = 6 of 25.
state: 19:30 18/25 home (a0–a2, e0, e1; both auto arms complete), no verdict: 8/8 food runs learn whole lives (first cap 18–47k) then lose them (last cap 86–189k); 10 control runs, 2 capped lives. CRITIC OVERESTIMATES 18/18; done-wiring ruled out (18:50). 19:45: replay eviction ruled out (buffer = max(T,50k) = 300k ≥ T). Q is NOT tied to eating: it climbs smoothly in every run, no break at the last capped life (Q there 236–673); controls that never eat climb as high or higher (300k median 996 vs food 788; at 90k ≈ equal) → the overestimate is generic, not the loss's mechanism.
state: 19:30 ENTROPY RECOVERED from the trace (H = −8 + ent_coef_loss/log α; `e2c-table` third section, planted test PASS): fixed 0.1 does NOT hold the policy wide — H 5.4 → below 0 by 15–35k → −10…−20 at 300k in 8/8 fixed runs, controls too (sharper than auto's −8). Auto α bottoms 0.04–0.06 at 15–25k then CLIMBS to 0.13–0.65. §19 cause 1's signature (H stays ≈ 5) and its "auto α < 0.01 by 50k" are both contradicted. Descriptive; cause of the loss still not shown.
next: e3 HOME 19:52 → 21/25 (missing E-random_point s803, s804 and E-food, E-zero s804); harvest e2 (RUNNING 20:08, watcher alive) → at 25/25 `$PY -m experiments.pilot_b2 e2c-readings` + `e2c-table` → WORLD_CARD rows + the branch it prints (if NO BRANCH: say so, INBOX the Director); the critic and entropy sections go beside them as named findings.

### J14 — The free GPU week (opens Sat 11-21 00:00 UTC): brain A's rows, A in his world, the ear's row
part: His brain (A) + Ear · stage: 1 · step: 3 (reproduce) · prior: §2, §9 row 7, §7, §18 item 4; J5 (`ca4dcf00`: 51.8 dec/s on a T4); rulings 33, 47, 48, 52, 62
prior: §2's three rows at the early points of the published curves, ~2.6 T4-h each. A has reproduced nothing yet. Ear: BC-ResNet-1 (9.2K weights), Speech Commands v2 12-class, published 96.9 ± 0.30 over 5 seeds; ours: model vendored verbatim, 9,232 weights, corpus and official test set reproduce.
**From Sat 11-21 00:00 UTC, pushed HEAD, in this order (rulings 52, 62):**
A1 → A2 → A3 (`scripts/launch_detached.sh
/data/jack-logs/brain_a/rows/launch_A1.log $PY -m experiments.brain_a.rows
kernel A1`) → A in his world (`… -m experiments.brain_a.in_world kernel`)
→ the ear (`… -m experiments.kws_row kernel s0 s1 s2 twin`) → **the
needs-hidden row** (`… -m experiments.needs_grid_row kernel <cells…>`, ~2
T4-h, rule frozen `5464c5b8`) → extension seeds only if a rule calls them.
≈ 17 of 30 h. **Every JACK slot on Saturday checks "is A1 launched?" before
anything else;** a slot refused on load or a failed push hands it to the
next slot. A quota refusal is final for the week → Colab through `gpu.py`.
**The hours left (ruling 62):** only if A1–A3 are in band AND the smoke
passes, DRAFT (text, no seed) the rule for A's pilot on arm K's world —
pilot 2's frozen readings, 300k decisions (≈ 1.6 T4-h a run), seeds and
never-trains twin stated — into `docs/research/prereg/pilot_a.md` and one
INBOX line; the Director freezes it. If A misses a row: fix A, no pilot.
serves: M3, M5 · PLAN ready-list "brain A reproduced … and smoke-tested in his world"; part "Ear" 2 → 3 · objective 3 · philosophy 2, 3.
**done when:** A1–A3, the ear row and the needs-hidden row are in `KNOWN_ANSWERS.md`, in band or a named miss, and A's smoke line.
state: A rows: rule frozen `286d6451` (A1 cartpole_balance, A2 swingup, A3 ball_in_cup; seeds 0–2, ext 3–4; never-trains twin below band else VOID); kernels + tests `31286dac`; control corrected to replay_ratio 0.0. Nothing left for A1–A3 before Saturday.
state: (a) DONE `57574dc7`: rule frozen in `experiments/kws_row.py` (authors' clone sha256-pinned, band 96.3–97.5, seeds 0–2 + ext 3–4, counts 36923/4445/4890 or stop, shuffled twin ≤ 15.0%); box smoke ran the real main.py 200 epochs both arms, guard fired; `test_kws_row` PASS. T4 cost a GUESS (1.5 h/run) — first kernel measures it. Ear on 11-21: `scripts/launch_detached.sh /data/jack-logs/kws_row/launch_k1.log $PY -m experiments.kws_row kernel s0 s1 s2 twin`.
state: (b) DONE `388c2582`: `experiments/brain_a/in_world.py` — rows' BASE unmodified, env swapped via sheeprl `env=gym` + gymnasium `module:id` (no runner edit); pilot world arm K, body eye, body motor (INBOX default; `--motor world` exists). DONE rule in docstring: steps / ≥2 logged loss steps / all finite / dec/s. Box 1,200-decision smoke: rc 0, env ~290 dec/s, losses finite at the one logged step (reward 5.537 vs §2 5.541); the learns clause is left to the kernel.
state: 11-19 14:30 PRE-FLIGHT for Sat (no push, no GPU): `gpu.week_key` turns W46→W47 at 11-21 00:00:01 UTC (W46 33.76 h spent, W47 empty); A1/A2/A3/A1-ext, `in_world` and the ear `s0 s1 s2 twin` kernel bodies + `build_job` scripts all build and parse; in_world's `make` passes all five world flags (body eye, body motor, floor food, lazy audio, native substep). Est. 3×2.6 + 1.0 + 6.3 = 15.1 h of the 30 h.

### J3 — DONE (closed, see state) — The home-made brains' fair chance: ONE last entry, one setting for both rows
part: His brain · stage: 1 · step: 3 (reproduce) · prior: §15, §5; rulings 24, 33, 44, 56, 59
prior: §15 — all seven arms are ONE on-policy AWR learner with different auxiliary losses; five FAIL both rows unrepaired. Measured 11-20: noise held 32 decisions PASSES sparse row 3 (4/5) and alone KILLS dense row 1 (0/5 learn; γ 0.99 alone 5/5 learn). No published value for the hold on this learner — the 8 below is ours, stated before the run, not from the record.
prior: no in-house arm has reproduced BOTH rows; B has (1b, 2b, 3, 3b).
**RULED (59):** a brain that goes in one creature passes both rows with ONE
setting — no per-row hold. **Do.** One kernel, rule written first:
`wm-latent@hold8` (noise held 8 decisions, γ 0.99, everything else as
`@repair`), rows 1 and 3, fresh seeds 10–14, the same verdict clauses,
never-trains control must fail, no extension. PASS both → the other arms
get `@hold8`, one session each, same seeds. FAIL either → J3 CLOSES: every
arm set aside with its numbers on record (PLAN: "may return if it is
repaired"), the hold finding kept in `KNOWN_ANSWERS.md`. No third entry.
serves: M3 · PLAN "a fair chance for our own brains" · philosophy 3, 4.
**done when:** `KNOWN_ANSWERS.md` has `@hold8`'s row pair and J3's closing line or the other arms' rows.
state: 11-19 17:15 (5) `wm-latent@repair` → Row J3-5: **row 3 PASS** 4/5 (first in-house row reproduced), **row 1 FAIL** 0/5, no learning.
state: 11-20 12:20 (5′) DIAG HARVESTED (`b4aa10f3`, KNOWN_ANSWERS J3-5 addendum): **HELD NOISE kills row 1.** `@hold-only` 0/5 learn (−1199…−1265 ≈ control); `@gamma-only` 5/5 (−284…−759); `@plain59` 5/5 (−354…−521, not VOID). γ/value-scale suspect REFUTED.
state: 11-20 14:25 (6) RULE WRITTEN before any cell `5ec9879e`: `wm-latent@hold8` in `inhouse_rows.py` (hold 8, γ 0.99, rows 1+3, seeds 10–14, same verdict, no extension); `test_hold8` PASS (file 6/6). NOT launched: Kaggle 5/5 (J15 e0, e1, a0–a2); J15's e2/e3 take the next free sessions first.
state: 11-20 18:08 `@hold8` LAUNCHED `jack-cpu-1795198133`; home 19:58 (6,594 s).
state: 11-20 20:15 **DONE — J3 CLOSED (ruling 59)**, KNOWN_ANSWERS Row J3-6: `@hold8` row 1 **FAIL** 0/5 (best windows −993…−1098, edge −314.7; control −1171…−1225), row 3 PASS 3/5 (95.6, 94.4, 91.5; 88.6, 82.9). Hold 8 gives back a little of row 1 vs hold 32 (`@repair` −1188…−1290), nowhere near the band. Every in-house arm set aside with its numbers; return only through FORGE_POOL with a repaired exploration scheme.

### J10 — The walking layer: reproduce a published walk (owner 11-15)
part: Body and movement · stage: 1 side job (stage 3's part) · step: 3 (reproduce) · prior: §16, §4; ruling 45
prior: §16 — the cheapest published walk: SAC, Zoo entry, Humanoid-v4 — 4,825 [3,784–5,866] at 1M (10 seeds), 6,232 ± 280 at 2M. A return near 5,000 is STANDING: a walk row must also demand forward metres. The 2018 helper-force paper states no seeds or success rate.
**Do, Kaggle only (rank 3).** The frozen row
(`experiments/humanoid_sac_row.py`): seeds 0–2 in 500k segments —
`kernel <seed> <segment>`; harvest, launch the next segment of the same
seed, to 2M each; then `score`. A miss is a miss (the rule is frozen). (2)
Yu 2018 stays third; no full run before this row is written. (3) Our
humanoid: not before a row reproduces. Inside him the layer KEEPS LEARNING.
serves: M4 · PLAN stage 3 "a separate, fast walking layer is built and checked in parallel from now" · philosophy 1, 2.
**done when:** the row is in `KNOWN_ANSWERS.md`, pass or fail.
state: (2′) RULE FROZEN `experiments/humanoid_sac_row.py`: Zoo entry, seeds 0–2, PASS R ≥ 5,950 AND ≥ 12 m/1,000 steps on ≥ 2/3, 1M median in [3,784, 5,866], late-start control < 500 else VOID; resume = 500k segments, model + replay (float64, ~6.1 GB) carried; box resume_check OK (`23e375be`).
state: 11-20 seg 1 HOME both (the segment mount works): **1M check** s0 R 4,065 M 3.47 (3/10 eps fell), s1 R 5,294 M 5.08 (0/10 fell) — both inside [3,784, 5,866]; median waits for s2. M ≪ 12 at 1M: the 2M PASS needs +7 m/1k steps.
state: 11-20 09:40 LAUNCHED `kernel 2 0` `jack-cpu-1795167564` (`launch_s2_seg0.log`) and `kernel 0 2` `…167600` (`launch_s0_seg2.log`), push rc 0. `kernel 1 2` refused twice by the load guard, then LAUNCHED 09:43 `…167808` (`launch_s1_seg2.log`, rc 0) at load 5.97.
state: 11-20 18:09 s2 seg0 (500k), s0 seg2, s1 seg2 (1.5M) all HOME (hit_target). s2 seg1 LAUNCHED `jack-cpu-1795198143` (`launch_s2_seg1.log`); 19:08 s0 seg3 (→ 2M) LAUNCHED `jack-cpu-1795201683` (`launch_s0_seg3.log`, rc 0).
state: 11-20 20:08 s1 seg3 (1.5M → 2M) LAUNCHED `jack-cpu-1795205296` (`launch_s1_seg3.log`, push rc 0).
next: harvest s2 seg1 (→1M check, gives the median), s0 seg3, s1 seg3; then s2 seg2–3; 2M eval decides.

### J16 — Brain A on the Mac: the wheel check, then rung 2
part: His brain (A on the Mac) · stage: 1 (ready-list "first-day speed test for the Mac") · step: 2 (combine) · prior: `docs/research/PUBLISHED_EVIDENCE_2026-11-15_PORT.md` (its step table, the rows after 1.3); `experiments/brain_a/README.md`
prior: nobody has published Dreamer on Apple hardware. G0, G1 and rung 1 PASS here (9/9 eager and compiled; parity is with the compiler's output). No speed is read anywhere but the Mac.
**Do, box only (< 15 CPU-min a run, nice 19), the report's rows unchanged
and in its order.** (d) `rung1.py` again on the PyPI MLX wheel (the build
the Mac will use) in a venv under `/data/venvs/` — same 9/9, planted 9/9,
or the differing test named. (e) Rung 2, the report's next rows: NumPy
64-bit references from the official code first, tests committed while they
FAIL, then the module until they pass eager and compiled; "ours beside
theirs" table in the docstring. No training, no speed claims.
serves: M3 · PLAN "the decisive test … on the Mac mini" — every Mac hour an hour of the real test · philosophy 5, 6, 7.
**done when:** the wheel line and rung 2's pass are in `experiments/brain_a/README.md` with the test names.
state: (a)–(c) DONE `07a65c3d`: G1 IDENTICAL 1,400/1,400; rung 1 PASS 9/9 eager and compiled (mlx 0.32.4); official grid = XLA:CPU's fma form.
state: 11-20 (e) RUNG 2 PASS `df2b7bc2`: `rung2.py` committed FAILING 0/4 vs stubs `d5d840d1` (refs 20/20 vs official, planted 4/4), then filled → 4/4 eager and compiled (returns, percentile, perc normaliser, slow critic; ≤1.4e-7×(|ref|+1)); rung 1 still 9/9. Found: eager `jnp.percentile` q=100 is 1 ulp off, jit exact.
state: (d) the PyPI wheel cannot load here (GLIBC 2.34 < 2.35) → needs a Kaggle CPU session. 11-20 20:24 KERNEL READY `c6611404` (not launched — order says box only): `experiments/brain_a/wheel_kernel.py` runs rung1/rung2 + both planted suites on PyPI `mlx[cpu]==0.32.4`; WHEEL verdict fixed (all 4 rc 0 + PASS/OK lines + mlx 0.32.4 printed). Test 3/3 incl. PASS on the box's real output and 6 planted non-PASSes.
state: 11-20 20:39 (d) **WHEEL PASS** `jack-cpu-1795207086` (ruling 60 rank 0, 0.3 h): PyPI `mlx[cpu]==0.32.4` on x86_64 Linux — rung 1 9/9, planted 9/9, rung 2 4/4, planted 4/4, eager AND compiled; README line written. Limit: Linux x86 wheel, not the Mac's arm64 binary.
**DONE** — (d) and (e) both in `experiments/brain_a/README.md` with test names (commit below). Next for the Mac: the rest of row 1.4 by the PORT step table, when the Director orders it.

**Lane `forge` (the science lab, owner 11-20; ruling 61)** reads its orders
in `docs/FORGE.md`: F1 → F2 → F4; F3 (round 1, the eating wall) after F1's
planted tests pass AND J15's reading is home — it does NOT adopt E2c; its
rule is frozen by the Director before any seed. If F1 has not passed by the
11-22 sitting, round 1's rule is drafted in this lane instead. Lane `lab`:
a candidate learner or setting for brain B enters through
`docs/FORGE_POOL.md` with its prior; probes of other seats stay in `LAB.md`.
**Parked or closed:** `D1.0`, `T2.01` (→ J10), `LF.06`, `LC.09`, `CR.01`,
`LF.04` — "to earn again" on the chosen brain (ruling 20). `NE.01` → stage 3
(ruling 35). `PS.02`/`.05`/`.06`/`.08` → stage 3 (ruling 42). `W1.04` →
re-read 11-25 (ruling 41). Brain B row 7 → dropped (ruling 52). The lab's
place key (11-20) → stage 3 "memory and sleep", no order now (ruling 63).
**Next up, in order (idle slots; box < 15 CPU-min unless said):**
1. Brain A fed one key per sense (§18 item 2): the `keys=per_sense` path in
`brain_a/in_world.py`'s `make` (Dict space, one decoder head per key) +
smoke against that file's DONE (1)–(4). The frozen A1–A3 rows untouched.
   state: 11-20 12:40 BUILT `3a35ec8d`: `make(keys="per_sense")` (Dict of 5 Boxes: vision 32/audio 16/touch 8/proprio 12/needs 6, runner unedited) + `overrides(keys=)` + `smoke N --per-sense`; tests 7/7 (bit-exact round trip over 40 steps, a one-off split caught). A1–A3 untouched.
   state: box smoke 1,200 dec: rc 0, sheeprl Encoder/Decoder MLP keys = 5 senses, no first-key warning; (1) STEPS + (3) NO NaN yes (8 Loss/* finite, reward_loss 5.537 = flat smoke); (2) LEARNS undecidable at 1 logged step — same as the flat smoke; (4) 2.2 dec/s loaded box, no claim.
   state: 11-20 17:40 `kernel --per-sense` BUILT `3d3d2962`: flat GPU job byte-identical (sha pinned), per-sense adds only `keys=` and fetches `inworld_s0_body_per_sense.json`; tests 8/8.
   next: (2) when a session is free and the Director ranks it: `$PY -m experiments.brain_a.in_world kernel --per-sense` → `/data/jack-logs/brain_a/in_world/inworld_per_sense.json`, judge (2) LEARNS.
2. `world_card`: a per-food eat count per life (today `BOUT_GAP` = 5 is
shorter than regrowth, so `floor_bouts` = `floor_eats_all` and "meals from
one food" cannot be read) — with a planted test; no old row re-scored.
   state: 11-20 14:55 DONE `7f20a8c3`: `FoodTally` → `floor_eats_by_food` + `max_eats_one_food` per life; `test_world_card_by_food` 4/4 (planted; control: one food ×3 vs three ×1 both (3,3) on old fields, max 3 vs 1; real d8_d00 s153 1,500 dec old keys identical). Seen: at d=0, 254 of 262 eats from ONE food (body parked on it). Reported only.
3. `row128` vs 16 rays under B (body frame): rule text only; it runs after
J15 names the setting.
   state: 11-20 15:35 DRAFT `docs/research/prereg/row128_vs_rays16.md` (text, no code, no seed): arms K-rays16 / K-row128 / K-row128-blind (must fail: width, no content) / K-lr0-row128, PILOT_RULE world, setting S from J15's table, seeds 921–925, 300k, pilot_b2 readings W=2 + MWU U≥21/25 on late meals/life and ≥4/5 pairs. Gate: does NOT run if both E2c food arms miss. Director freezes.
4. Stage-2 rule, first draft from `STAGE2_PREDICTION.md` + pilot 2's
reading → `docs/research/prereg/stage2_a_vs_b.md` (text; the Director
freezes it).
   state: 11-20 15:55 DRAFT `docs/research/prereg/stage2_a_vs_b.md` (text, no seed): gate = B eats somewhere (E2c; both food arms miss → stage 2 waits on F3) + A1–A3 in band + 13 harness checks + the 20-run pilot; arms A, B, B@A-ratio, 4 floors + B-shuffled must fail, f4 ceiling must pass; §8 log-rank 2 looks @0.025, censored at 1M; battery with predicted ordering; PLAN's decision table. OPEN for the Director: θ=5 vs SUSTAINED W, battery seeds, added-sense task, ceiling on the 16-ray eye, which shortening.

---

## SYSTEM — work on the machine (max 3 open)

*Order of work: **S12 (a)** (free hours expire weekly) → **S9** → **S12
(b)–(c)** → **S13**. A fix a RUNNING Jack kernel needs goes first (ruling 54).*

### S12 — The free TPU gets a door; `gpu.py` tells a refusal from a result
part: (machine — free compute, PLAN "honest limits") · stage: — · step: —
prior: Kaggle's own meter (`/data/venvs/kaggle/bin/kaggle quota`): TPU 20 h a week, 0.00 used, lapses Sat 00:00 UTC. Nobody here has run Dreamer on a TPU; the T4 number to beat is 25.9 grad steps/s (compiled, Colab).
**Do.** (a) `gpu.py`: an `accelerator` argument (default today's T4,
byte-identical job for every existing caller — test it) that can push
`--accelerator Tpu1VmV38`; TPU hours in their own weekly line of
`gpu_budget.json`. Then the owner's gate (11-20), one kernel: the official
`danijar/dreamerv3` @ `e01491f` debug preset on the TPU — runs clean or
names the unsupported op; peak memory; grad steps/s beside 25.9. Verdict
line "TPU viable / not" + the table → `PUBLISHED_EVIDENCE` and INBOX. No
TPU plan before the verdict. (b) A push refused by Kaggle ("Maximum batch
CPU session count", quota) is reported as a REFUSAL by name, never `rc=0`
+ "no artifact". (c) `dispatch.sh` pre-flight prints `kaggle quota` beside
our own meter (the account is shared; ours counts only our runs).
serves: objective 3 (spend the free compute) · philosophy 5, 6.
**done when:** the TPU verdict line exists with its table, and a planted refused push is named as a refusal.
state: (a) door DONE `7689b52b`; gate kernel `experiments/brain_a/tpu_probe.py`.
state: 11-20 NO TPU ATTACHED twice: the 16:04 push (`Tpu1VmV38`) and the 21:09 push (`TpuV6E8`, `4937120d`) were both accepted, metadata says TPU, but the box was CPU (`jax.devices()=[CpuDevice]`, no libtpu) and `kaggle quota` billed 0.00 TPU h. 16:06's "TPU NOT viable" was FALSE — `verdict()` now refuses to judge without a TPU device (real output → NO TPU ATTACHED; planted TpuDevice still judged). This week's 20 TPU h lapse unspent.
state: blocked on the owner (INBOX 21:15): the API batch push does not get a TPU on this account — maybe a TPU needs the Kaggle website (phone-verified account, interactive session) or a different id. No more sessions spent guessing ids. (b): refusal reads `push failed: …session count…`, ok=False, 0 h; a silent CPU fallback is now named by the probe, not by `gpu.py`.

### S9 — Every settled FAIL gets a named fate; the two red ratchets get an owner
part: (machine — nothing faked) · stage: — · step: —
**Do, ≤ 8 ids per slot; never a baseline raise, never a re-run for a better
number.** Fates: (i) `repaired_by` naming the order or milestone; (ii)
`FAIL-DISPOSED: <authority> <date> — reason` (marker BEFORE any `COVERS:`
text); (iii) an INBOX ask. Left: `T0.17` → (i): demote `T2.14`'s row by the
registry's own drift mechanism (its PASS was bought under claim words since
amended; not re-run on the old core), then T0.17 re-earns; `T0.23` re-earns
once `run ratchets` is green — **`live_unauditable_pairs` 22 → 16 is
lowered in `protocol.py` in the same commit that next re-runs `T1.14` (J14's
GPU week), not before** (it would stale a GPU certificate blocking 46);
then DP.05 HR.5 LT.02 T0.35 T3.09 T4.02 W.2 W1.00.
serves: philosophy 4 · the auditor's exit codes.
**done when:** `coverage` and `run ratchets` exit 0 with this id list empty.
state: slots 1–3: 18 fated by (ii); floors UNREACHABLE 115→107, PASS-ON-DEAD-DEPENDENCY 6→4.
state: slot 4 (`47f20b1c`): PS.05/06/08 → ruling-42; T0.32 → ruling-43 (repaired_by only takes registry ids, so written as (ii)) + LESSONS_CORE rule 7; WM.01 → WM.02 (losing challenger); XL.01 repaired_by PS.02. Floors FAIL_UNOWNED 1→0, UNREACHABLE 107→104. T0.23: 6 W46 Colab jobs = J5 brain-A probe, receipts written, unattributed 27→21 (at floor).
state: `run ratchets` exits 1, only `live_unauditable_pairs` 16 < 22: the floor is in `protocol.py`, which T1.14's IMPL_DEPS hashes — lowering it stales a GPU cert (blocks 46). Lower it in the commit that next re-runs T1.14 (J14's week), or Director rules.
state: Left: T0.17 (demote T2.14 by drift), then T0.23 re-earns; DP.05 HR.5 LT.02 T0.35 T3.09 T4.02 W.2 W1.00.

### S13 — The judge lock's tripwire (forge F5) and the lane-guard test that went red
part: (machine — nothing faked; FORGE.md "Fences") · stage: — · step: —
**Do.** (a) Reproduce `scripts/test_lane_guard.sh`'s 5–6 FAILs of 11-20
11:35 ("wrapped lane is VISIBLE", seat cap rc) on a clean tree: fix by
cause or name why the test is wrong; the lane guard is what keeps two
workers off one seat. (b) The tripwire: hashes of the judge files
(`experiments/harness/`, `registry.py`, the frozen rule files a forge round
names) checked by the forge round runner before it accepts a result; a
planted edited judge is refused. Real separation (own user) is NOT ordered
— it writes outside the repo; it goes to the owner if stage 2 needs it.
serves: philosophy 4 · FORGE stage 2 does not open before (b) holds.
**done when:** `test_lane_guard.sh` exits 0 or its wrong clauses are named, and the planted judge edit is refused.
state: (new 11-20.)

**Next up:** (0) **S11** as written 11-19 (`git show 11d1aeba:docs/ORDERS.md`):
FIELD_WATCH over its 12 KB cap, `research_prompt.md` line 50, the
root-disk prune of old CLI copies with a planted test. (1) **S6** step 3 —
the video of the latest recorded life. (2) `launch_detached.sh` refuses a
second live watcher with the same command line. (3) `prereg.judge()` door
for unequal arms — before the decisive test's rule. (4) A detached job
`queued` on Kaggle > 2 h gives up its box seat + one INBOX line. (5) `run
status`'s STEERING-PAGE ORDERS parser reads this page's headings. (6)
**S8** — a seat held BY RECORD in `champions --check`. (7) `XL.00` re-buy
with a planted positive (stale 74 d, in front of 19 specs). (8) **S4** —
six PASSes gated on a seed average that admits a failing seed. (9)
`T0.13`'s dead clauses; `verify` VENUE 266 vs 256.

---

## INBOX — one line each, workers add, the Director empties
(emptied 11-20 12:27 — every line of 11-19/20 is ruled in 57–63 or became J15, J3, J14, S12, S13, Next-up.)
- **FORGE 14:35 (F1 DONE, F3 can be frozen):** round runner + scoreboard pass 22/22 planted tests. Round 1's rule is a `spec.json` under `experiments/forge/rounds/r1/` (baseline, never-trains twin, ≤ 8 candidates each with source + prior, ≥ 1 `control` candidate, S/M/L seeds never reused, budget, `metric_sha256`); a planted example prints with `python -m scripts.forge_scoreboard --dry`. Four readings of FORGE.md are mine, overrule if wrong: equal never climbs; a non-control candidate must beat the control source's MEAN; above S the best control runs as a yardstick that cannot climb; a tie across the top-third cut drops. S/M score = share of late-half lives that are sustained (ruling 22's mark) — at S (50k) pilot 2 says every arm reads 0, so round 1's rule needs either a longer S or a second eating-based tiebreak; yours to freeze.

- **JACK 14:30 (J16 (d) cannot run "box only"):** the PyPI MLX wheel needs GLIBC 2.35 / GLIBCXX 3.4.30 (box 2.34/3.4.29, measured 11-16) — rung 1+2 on the wheel needs one Kaggle CPU session (both files run standalone, ~1 min); rank it or rule conda-forge parity sufficient until the Mac. Rung 2 PASSED meanwhile (`df2b7bc2`).
- **JACK 15:35 (session race):** S12's `tpu_wait.sh` fires on the FIRST free Kaggle session (deadline 21:30), and J15's e2/e3 (JACK rank 1) wait for the same session — whoever polls first wins; no rule orders them. Rule which goes first (TPU quota lapses Sat 00:00; e2/e3 hold J3/J15 behind them).
- **RESEARCH 16:30 (§20 filed — M8, "a living Jack"; rule 3, no order's prior was missing):** the record has NO small agent still improving at the end of an hours-long no-reset life (Jelly Bean World: learns ~1M steps, then collects nothing; PPO on a stationary Ant collapses 3–20M), and with several needs the moving food is the one dropped. To change: (1) brain A's in-world checkpoint carries no buffer (`in_world.py:30`; sheeprl default saves it by path, same machine) — the Mac hand-over needs `buffer.checkpoint=True` + `checkpoint.memmap_buffer=copy`, or A resumes empty; (2) M8's spec, when registered: geometric mean over needs, a stop-resume twin, a mid-life push, late ≥ early; (3) a box-minutes resume twin on row 1b is the one cheap open test. `PUBLISHED_PRIORS.md` is 118 KB: sections kept ≤ 6 KB (ruling 55), the 70 KB file cap not enforced by me (S11 b).
- **JACK 16:45 (J15, learned then lost):** E-food-auto s805 learned to eat and survive (4 capped lives, 13k–89k) then LOST it (≈0 meals/life after 105k). The frozen E2c readings look at the run's END — a learned-then-lost skill reads as "never ate". Not a change to the frozen rule; a question for F3/the forge (stability, not only run length).
- **JACK 17:15 (J15, E2c's table has a hole, rule it before 25/25):** fixed E-food crosses W=2 on 2/2 seeds home (s801, s802) and is on track to read SUSTAINED; ruling 57's table has branches only for "auto eats, fixed does not", "both miss", "zero eats like food" — none for "fixed eats" (with or without auto). Correction to 16:45: the W-mark reads crossings ANYWHERE in the run, so learned-then-lost IS seen by SUSTAINED; only "late meals/life" misses it (all 3 food runs lost it by ~120k). `pilot_b2 e2c-table` prints the capped marks.

- **JACK 18:35 (J15, the critic overestimates — a lead for the forge/§19, no rule touched):** in 14/14 E2c runs home, SAC's own value estimate (−actor_loss, min of twin critics) climbs all run, never-eating controls included, to 1.2–8.3× the ceiling the run's eating supports (fixed α: 1.7–8.3×; E-random_point s801 Q 2,174 vs 263). Record to read: van Hasselt 2018 'deadly triad' (arXiv 1812.02648), reward size 100/meal with γ 0.995. Whether this causes the learned-then-lost skill is NOT shown — a twin with critic-side fixes would be a forge candidate. Also (machine): `pilot_b.trace` stores SB3's `train/std` under the trace's own `std` key, so every window after the first loses the per-dim action std (SBX `get_std()` is a stub = 0.0).

- **JACK 19:45 (J15, narrows my 18:35 critic lead — read it as weaker):** on 18/18 traces Q (−actor_loss) climbs smoothly with no break at the last capped life, and the 10 control runs that never eat reach as high or higher (300k median 996 vs food 788; ≈ equal at 90k) — the overestimate is generic to this SAC in this world, not the loss's mechanism. Replay eviction also ruled out (buffer 300k ≥ T). The loss's cause is still unshown; forge/§19 should not build on the critic lead alone.
- **JACK 20:24 (J16 (d)):** the wheel check is now ONE command, ~0.3 h of a Kaggle CPU session, no GPU-hours (`wheel_kernel.py`, `c6611404`); 1 of 5 sessions free tonight (J15 e2 + J10 ×3 RUNNING). Rank it and the next idle slot launches it.

- **SYSTEM 21:15 (S12 (a), the TPU never came):** two pushes asking Kaggle for a TPU (old id `Tpu1VmV38` at 16:04, current `TpuV6E8` at 21:09) were accepted and ran on a plain CPU box — 0.00 TPU h billed, no libtpu. The 16:06 "TPU NOT viable" in the log was false and is now refused by the probe itself. The batch API does not seem to hand this account a TPU: owner, can a TPU session be started from the Kaggle website (is the account phone-verified for TPU)? If not, rule S12 (a) closed as "TPU unreachable from the box" and drop the TPU line.

## DONE (kept one day)

- **J11 — pilot 2** (`b4aa10f3`, `PILOT_B2.md` §4): 40/40 home, rule `e2c3c378`, W = 2. B0 NEAR RANDOM, K/P/G/KP SLIDE; 0 sustained lives in any learner; meals/life late 0.08–0.16 vs never-learning K-lr0 0.82, held8 0.95. §13's "K eats inside 300k" failed. No seat moves.
- **J16 (a)–(c)** (`07a65c3d`): G1 identical files; rung 1 PASS 9/9 eager and compiled.
- **J3 (4), (5), (5′):** `wm-latent+ring` FAIL both; `@repair` row 3 PASS 4/5, row 1 FAIL 0/5; diag: held noise is the cause, γ is not.
- **J15 (1)–(3):** f5 gate MISSES 5/20 (shuffled 0/20) — THIS memoryless hand rule does not meet the bar; E2b closed for this teacher.
- **Next-up 11-19:** dial race parts (`95e2e4cb`), needs-grid rule + kernel (`5464c5b8`), per-sense design note.
- **J15 13:11 (machine):** `e2c-kernel` pushes without checking Kaggle's 5-session cap — a2 was refused twice ("Maximum batch CPU session count of 5 reached"), each writing an attempt+ERROR receipt pair; a pre-flight count of RUNNING kernels would refuse it locally. J15 E2c: a0–a2 (auto) in flight 13:16.
- **J15 19:30 (E2c 18/25, the trace's lost std recovered as entropy):** fixed `ent_coef` 0.1 never held the policy wide — policy entropy falls from 5.4 to below 0 by 15–35k and to −10…−20 nats by 300k in 8/8 fixed runs (zero and random-point controls included), i.e. SHARPER than the auto arms (held at −8 by an α that bottoms at 0.04–0.06 and then climbs to 0.13–0.65). §19 candidate 1 (entropy bonus dominates, std stays 0.4–1) and its prediction "auto drops α below 0.01 inside 50k" are contradicted in 8/8 and 10/10; with the 18/18 critic overestimate this points at the critic/reward size (+100 a meal, unnormalised), not the temperature. PILOT_B2 §3b's frozen band reads pooled action std, which the trace lost — rule whether H may stand in for it at 25/25, and whether F3's round 1 gets a reward-scale / critic candidate. `pilot_b2 e2c-table`, third section.
