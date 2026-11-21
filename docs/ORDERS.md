# ORDERS — what the workers do. Written by the Director.
Director sitting: 2026-11-21 06:41 UTC

Workers: take the FIRST open order in your lane that you can advance. Update
only your order's `state:` lines (≤ 5, **each ≤ 4 lines of text — replace,
do not append; history is in git**). One state+journal commit per slot.
Machine problems go under INBOX, one line each. Rulings 64–75 cited here are
in `docs/DECISIONS_RESOLVED.md` under 2026-11-21.

**`docs/PLAN.md` binds every order (owner, 11-15).** Research → combine →
reproduce → test the last 10% → put it in him. We are in **stage 1: the
instrument.** `docs/research/PUBLISHED_PRIORS.md` is the reference — read
your order's section (only that section) at the start of every slot.
**Where stage 1 stands (11-21).** E2c is read, 25 of 25: **handed the
direction to food, plain SAC (brain B) eats whole lives (4/5 fixed α, 3/5
learned α; a sense that carries nothing: 0/5) — and in 10 of 10 food runs it
has stopped eating by the end.** With pilot 2 (16 rays, no learner ate) that
names two walls: **the EYE** (reaching food without being handed the
direction → J17) and **KEEPING** (→ forge round 1). The same win-then-lose
shape cost the published walk a seed (J10). Brain A: rows A1, A3 in band, A2
a named miss at 100K (→ J14's fix). **Ruling 64: every new B run in his
world uses `ent_coef="auto"` (the published setting). Ruling 65: "eats"
means the LATE half of the run from now on, in every new rule. Ruling 58
stands, amended: learner runs in his world only through J17, J14 and the
forge's frozen round 1.**
**Every new run in his world passes** `eye_frame="body"`,
`motor_frame="body"`, `food_floor_only=True`, `lazy_contact_audio=True`,
`native_substep=True`. The integrator is not switched (ruling 40).
**Stopped (rulings 10, 20, 23, 36, 59):** home-made learners (J3 closed;
return only through `FORGE_POOL.md`); forager cells on the 16-ray compass
eye; world-rule races; `D1.0`, `T2.01`, smell runs; new specs with no
published prior. A question with no prior is a `RESEARCH:` line in INBOX.
**Kaggle CPU: 5 sessions, RANKED (ruling 60, re-ranked 11-21). No session
sits idle while any ready kernel exists** — caps bind only while a
lower-ranked READY kernel waits. (0) any kernel estimated < 1 h; (1) J17's
kernels — up to 3; (2) forge round 1 — up to 2; (3) J10-2's segments — up
to 2, only after J10 (a) passes; (4) brain B row 6. Every JACK slot's first
act: `kaggle kernels list`/`status` — harvest what is home, refill what is
free.
**The box (rulings 32, 53).** Day ceiling 57,600 CPU-s, ≤ 2 computing
detached jobs. Whole run < 15 CPU-min → box, nice 19. 15–60 CPU-min → box
only when all 5 Kaggle sessions are held AND load < 4, one worker. > 60 →
Kaggle or it waits. The box is shared (never touch other projects). Before
launching, `pgrep -af` the same command line — never launch a kernel that
already has a live watcher.
**Kaggle GPU: 30 h this week (W47, ends Sat 11-28 00:00 UTC); 3.8 used at
06:40.** Order in J14. At most 2 GPU sessions at once (Kaggle's cap); a
third job goes to Colab through `gpu.py`. **TPU: closed (ruling 71) — no
push asks for one.**
**`/` has 4.3 GB free, workers refuse under 3:** scratch, clones and caches
under `/data` only (`UV_CACHE_DIR`, `PIP_CACHE_DIR`, `XDG_CACHE_HOME` under
`/data/caches/`, `--only-binary=:all:`).
**World files** (`w0.py`, `needs.py`, `survival.py`, `drives.py`): changed
only through `jack_env.py` options whose default is today's world.
**Known-answer rows (ruling 33):** seeds AND at most one extension stated
BEFORE the first seed runs. A miss after that is a miss. **A row that failed
on OUR apparatus (not the learner) may be re-registered ONCE, new id, fresh
seeds, bands unchanged, the fault named (ruling 70).**
**Receipts:** the slot that launches a kernel commits
`experiments/gpu_submissions.jsonl` and `cpu_budget.json` by name.

---

## JACK — build the creature, his world and his curriculum (max 7 open, ranked)

*Order of work: **J14** (harvest what is home, keep 2 GPU sessions busy) →
**J17** → **J18** → **J10** → **J16**. A slot waiting on a kernel advances
the next order. A slot that can advance none takes "Next up".*

### J17 — The eye is the wall: 128-ray row against 16 rays under brain B
part: Learning to eat (+ Eye) · stage: 1 ("proof the task can be learned with his senses") · step: 4 (pilot) · prior: §12, §19; `docs/research/prereg/row128_vs_rays16.md`; rulings 37, 51, 64, 65, 66
prior: published — nothing says 16 undifferentiated rays can tell food from a wall (§12); nearest: "What if Eye" 2025 needed ~10 eyes × 4 receptors to navigate. Ours: hand forager on row128, body frame, 20/20 worlds, 7.5 meals a life (shuffled 2.17); floor food in sight .905 (row128) vs .287 (16 rays).
prior: learnable by a plain learner — shown ONLY with the direction handed (E2c 4/5); on 16 rays 0 of 25 (pilot 2). B has reproduced rows 1b, 2b, 3, 3b. Steps: first reach expected 50k–200k (§19, inference); 300k is OUR budget.
**FROZEN by ruling 66 = the draft as written, with these pins:** setting S =
`ent_coef="auto"`; arms K-rays16 / K-row128 / K-row128-blind (must fail) /
K-lr0-row128; PILOT_RULE world, seeds 921–925, 300k decisions; primary =
late meals/life, MWU U ≥ 21/25 and ≥ 4/5 pairs, row128 over rays16;
reported beside it, not read: first capped life, last capped life (reached
then lost). No extension.
**Do.** (1) Copy the pins into the draft, change its status line to FROZEN
with this ruling, commit — that commit is the freeze. (2) Build the arms in
`pilot_b2.py` in a SEPARATE dict (pilot 2's and E2c's groups byte-identical
— test it; planted test: the blind arm's eye carries width and no content).
(3) Launch (rank 1, ≤ 3 sessions); harvest; one row per arm in
`WORLD_CARD.md` beside its control; ONE INBOX line with the reading.
serves: M3 · PLAN "Learning to eat" last 10% ("can 16 rays tell food from a wall?") · philosophy 1, 3, 4.
**done when:** the four arm rows are in `WORLD_CARD.md` and the rule's verdict is named — row128 eats / neither eats / both eat.
state: 11-21 07:15 (1) FROZEN `98e22036` (pins copied; "late" = ruling 65's late half = lives STARTING at decision ≥ 150k; the draft's last-quarter number printed beside, not read — INBOX). 07:32 (2) BUILT `59bbdf90`: `J17_ARMS` separate, r0..r4 = one seed's four arms each; old groups byte-identical (5 sha pins); blind arm 298 wide, 256 vision numbers 0, rest identical to K-row128 over 2,000 actions.
state: FOUND + FIXED before launch: row128's `mj_multiRay` call died under the kernel pin mujoco 3.17.0 (3 of 4 arms would have crashed = VOID); `fast_sim._multi_ray` takes both signatures, first look bit-identical on 3.2.3 and 3.17.0. Learner smoke, NON-rule seed 931, 1,500 decisions, all 4 arms: auto α .89 → .66, lr0 arm flat 1.0.
state: 07:35–07:37 (3) LAUNCHED r0 `jack-cpu-1795246505`, r1 `…246570`, r2 `…246635` (Kaggle CPU RUNNING, est 7.5 h each; logs `/data/jack-logs/pilot_b2/launch_j17_r{0,1,2}.log` → `pilot_b2/j17_r*.json`). **r3, r4 NOT launched** — the box's 6-poller cap was full; launch them the moment a watcher exits (`pilot_b2 j17-kernel --name r3|r4`).
state: NEXT, before any kernel is home (~15:00): build the reading — `j17-table` (late-half meals/life per run, MWU U ≥ 21/25 + ≥ 4/5 pairs, row128 vs rays16 and vs blind; first/last capped life beside) with planted tests; then harvest → four rows in `WORLD_CARD.md`.

### J14 — The free GPU week (W47): A's fix, the needs-hidden row again, the ear
part: His brain (A) + Ear + Other senses · stage: 1 · step: 3 (reproduce) · prior: §2, §7, §18 item 4; `docs/research/prereg/a2_size_s.md`; rulings 33, 62, 67, 68, 70
prior: A1 PASS (782 @100K, 984 @200K), A3 PASS (956 @200K), A2 named MISS @100K (393.3 vs 438–518), in band @200K. Our rows ran sheeprl's 1m size at 32-true; the published band was earned at size S, float16, prefill 1,024. Ear: BC-ResNet-1, published 96.9 ± 0.30.
**Do, in this order; keep 2 Kaggle GPU sessions busy, never 0:**
(a) Harvest the ear (`jack-ladder-1795231396`, check `EAR_TORCH_REPIN 0`) →
`KNOWN_ANSWERS.md` row, in band or a named miss.
(b) **A2-v1S, FROZEN by ruling 67:** arm (a) of `a2_size_s.md` only —
sheeprl `dreamer_v3_S`, cartpole_swingup, seeds 0–2, same bands, same
control, one extension (seeds 3–4); pins: `32-true` (as A1–A3, so ONLY size
moves; float16 goes in "ours beside theirs"), `learning_starts=1024`. Timing
cell first; one seed per kernel. Mark the draft FROZEN, then launch. Reading
(write it in before seed 0): in band @100K → the build is right and the 1m
miss is SIZE, a known limit — then draft `prereg/pilot_a.md` (ruling 62's
terms, text only); miss → a build fault, diff ours against official
`ec526b9` setting by setting. The 12m arm is struck for this week.
(c) **Needs-hidden, re-registered ONCE (ruling 70): NH-2 =
`docs/research/prereg/needs_hidden_nh2.md` as drafted `5e043f6c` BEFORE the
pilot read — FROZEN.** One change (μ on the walkable corner, as the
published map), every NH-1 band and clause unchanged, ONE amendment — seeds 10–19 (never 2,
5 or NH-1's 0–9: the pilot peeked at two of them), no extension, own output tag. Its gate stands as written:
launch only if BOTH box pilot seeds read Δ < 5.0; else no NH-2 — the next
pilot is the stat bounds and ONE INBOX line. Mark the draft FROZEN, build
`kernel --mu-inside`, launch. NH-1 stays a FAIL on the record either way.
(d) A's in-world checkpoint carries no replay (`in_world.py:30`): add
`buffer.checkpoint=True` + `checkpoint.memmap_buffer=copy` behind an option,
with a box test that a resumed run's buffer length equals the saved one.
Before any A run longer than the smoke.
serves: M3, M5 · PLAN ready-list "brain A reproduced … and smoke-tested in his world"; "Ear" 2 → 3 · objective 3 · philosophy 2, 3, 4.
**done when:** the ear row, A2-v1S's row and NH-2 (or its named no) are in `KNOWN_ANSWERS.md`, and (d)'s test passes.
state: 11-21 A1–A3 scored (`420217d2`): A1 PASS, A3 PASS, A2 named MISS @100K. (a) EAR `jack-ladder-1795231396` STILL Kaggle RUNNING at 07:38 (4 h 15 m; watcher `kws_row/launch_k2.log`) — harvest when home, check `EAR_TORCH_REPIN 0`.
state: (b) A2-v1S FROZEN `8f13ab28` (reading written before seed 0), arm built `bdeda9b6` (`rows.py` BASE_V1S; hydra: dense 512, deter 512, discrete 32, 32-true, prefill 1024; frozen A1–A3 lines tested byte-identical). TIMING CELL RUNNING 07:34 `jack-ladder-1795246441` (Kaggle GPU; log `brain_a/rows/launch_v1s_timing.log` → `rows/v1s_timing.json`, key `projection.hours`; no score read).
state: (b) NEXT: read the projection → `rows v1s-kernel 0 --control --est=<h>` then seeds 1, 2, one per kernel (`v1s-score` judges by row A2's rule). If projection is None the runner's `Time/*` scalars did not log — read `time_scalars` in the json, do not guess.
state: (c) pilot HOME 07:30: μ inside reads Δ **1.22 (s2), 1.87 (s5)** vs frozen-μ 17.2 / 25.0 → gate (< 5.0 both) OPEN. NH-2 FROZEN + built `37dcc858` (gate in code, seeds 10–19, tag `nh2_grid_*`). **NH-2 RUNNING 07:34 on Colab T4** (Kaggle refused: 2 GPU sessions held; session `ladder-1795246472`, watcher pid 2440962, log `needs_grid_row/launch_nh2.log`, ~2.4 h). Harvest: `needs_grid_row score-nh2` → `KNOWN_ANSWERS.md` Row NH-2.
state: (d) NOT started (replay in A's in-world checkpoint + resume-length test) — before any A run longer than the smoke.

### J18 — Forge round 1 is a KEEP round (the rule's terms; lane `forge` runs it)
part: Learning to eat + The science lab · stage: 1 · step: 4 · prior: §19, §21, §22; `docs/FORGE_POOL.md`; rulings 61 (amended), 65, 69
prior: E2c — on E-food the baseline reaches (4/5) and loses (10/10, last capped life 81–189k). Record: resets + LN critic are the best-supported keep recipe (Nauman 2024) but ours already has LN and lost; TQC (Kuznetsov 2020); the one published SAC forager updates 1 per 10 decisions and pays ±1 a food; `n_steps=3` helps sparse reward (Rainbow, D4PG). Nobody has published a small agent still improving at the end of a long life (§20).
**FROZEN TERMS (ruling 69) — lane `forge` writes them into
`experiments/forge/rounds/r1/spec.json`; the commit is the freeze; any term
it cannot meet is an INBOX line, not a changed term.** World: E2c's E-food
arm exactly. Baseline: fixed α 0.1 as E2c ran it; never-trains twin.
Candidates (8): K1 resets (interval set so ≥ 1 reset falls inside rung S),
K2 TQC, K3 auto α, K5 1-per-10 updates, K6 reward ÷100 + auto, `n_steps=3`
(§22), and 2 plain-variation controls. Rungs: **S = 150k**, M = 300k. Score
= share of LATE-half lives that are sustained (ruling 22's mark); tiebreak
late meals/life. Seeds fresh, never 801–805, never reused across rungs; 3 a
candidate at S. The forge's four readings (INBOX 14:35) stand. ≤ 2 Kaggle
sessions (rank 2). No result is accepted before S13 (b)'s tripwire holds.
**Lane `jack` does nothing here** unless the forge has not committed
`spec.json` by 11-22 06:00 — then a JACK slot writes it to these terms.
serves: M3, M7 ("he keeps learning") · philosophy 1, 3, 4.
**done when:** round 1's rung S is on the scoreboard, one reading per candidate beside the baseline, twin and controls.
state: (new 11-21.)

### J10 — The walking layer: is the resume clean, then the row again with enough seeds
part: Body and movement · stage: 1 side job (stage 3's part) · step: 3 (reproduce) · prior: §16, §21; rulings 45, 72
prior: §16 — SAC, Zoo entry, Humanoid-v4: 4,825 [3,784–5,866] at 1M (10 seeds), 6,232 ± 280 at 2M, each ONE unbroken run. Ours: Row J10-1 FAIL (1 of 3: s2 6,490 / 20.1 m; s0 5,834 / 11.91, inside the eval's 13% cross-CPU noise; s1 collapsed 5,621 → 581 in its last segment; two re-seeded forks of s1 both KEPT). §21: no published per-seed record of a late Zoo-SAC collapse.
**Row J10-1 stays FAIL. Nobody cites it as "the published walk does not
reproduce" or as evidence about SAC until (a) is read (auditor, ruling 72).**
(a) Box-minutes, no new seed: a resumed twin against an unbroken one — the
same short run (≤ 20k steps) once straight through and once saved + resumed
at the midpoint through the row's own resume path; compare α, optimiser
state, normaliser, replay order and the first 1,000 updates after the
resume. Identical or the differing item NAMED → `KNOWN_ANSWERS.md` J10-1
addendum. A fault found is fixed before (b).
(b) Rule text first, `humanoid_sac_row.py` as J10-2: the frozen recipe (no
cures — a reproduction runs the published recipe), fresh seeds 3–7, PASS R ≥
5,950 AND ≥ 12 m/1,000 steps on ≥ 4/5, eval = 50 episodes on ONE stated
machine (Kaggle CPU), 1M median band and control as before, no extension.
Then segments at rank 3, ≤ 2 sessions.
serves: M4 · PLAN stage 3 "a separate, fast walking layer is built and checked in parallel from now" · philosophy 1, 4.
**done when:** (a)'s addendum is written, and J10-2 is in `KNOWN_ANSWERS.md`, pass or fail.
state: J10-1 (rule `humanoid_sac_row.py`, seeds 0–2, 500k segments): 1M median 4,065 IN BAND, control R 68 valid; 2M 1 of 3 → FAIL (00:12).
state: 01:15 why (`1fdd3d58`): all 11 `model.zip` re-evaluated on the box; load is exact (12/17 early falls on the identical step); s1 collapsed in the LAST segment, α .009→.024, Q rising while return fell.
state: 05:15 forks `30447a6f`: f1 R 5,863 M 12.19, f2 R 5,303 M 14.07 → {KEPT, KEPT} → CHAOS; 2 forks say nothing about a rate.

### J16 — Brain A on the Mac: the next rows of the port table
part: His brain (A on the Mac) · stage: 1 (ready-list "first-day speed test for the Mac") · step: 2 (combine) · prior: `docs/research/PUBLISHED_EVIDENCE_2026-11-15_PORT.md` step table rows 1.5 → 2.3; `experiments/brain_a/README.md`
prior: nobody has published Dreamer on Apple hardware. Rungs 1 and 2 PASS here, eager and compiled, and on the PyPI wheel (Linux x86). No speed is read anywhere but the Mac.
**Do, box only (< 15 CPU-min a run, nice 19), the table's rows in its
order, one row a slot:** 1.5 (optimiser golden → rung 3; first update has
learning rate 0), then 2.1 (golden generator at `size1m`, repeat run
identical), 2.2 (one recurrent step → rung 6), 2.3 (64-step memory pass →
rung 7). Each row: NumPy 64-bit references from the official code first,
tests committed while they FAIL, then the module until they pass eager and
compiled; "ours beside theirs" in the docstring. No training, no speed
claims. A wheel re-check is one < 1 h Kaggle CPU kernel (rank 0) per rung.
serves: M3 · PLAN "the decisive test … on the Mac mini" — every Mac hour an hour of the real test · philosophy 5, 6, 7.
**done when:** rungs 3, 6 and 7 pass, with test names in `experiments/brain_a/README.md`.
state: 11-20 rung 1 9/9, rung 2 4/4 (`df2b7bc2`), eager and compiled; WHEEL PASS `jack-cpu-1795207086` (PyPI `mlx[cpu]==0.32.4`).

**Lane `forge`** reads `docs/FORGE.md`; its F3 is J18 above (terms frozen);
F4 after. Lane `lab`: a candidate learner or setting for brain B enters
through `docs/FORGE_POOL.md` with its prior; probes of other seats stay in
`LAB.md`.
**Parked or closed:** `D1.0`, `T2.01` (→ J10), `LF.06`, `LC.09`, `CR.01`,
`LF.04` — "to earn again" on the chosen brain (ruling 20). `NE.01` → stage 3
(ruling 35). `PS.02`/`.05`/`.06`/`.08` → stage 3 (ruling 42). `W1.04` →
re-read 11-25 (ruling 41). The lab's place key → stage 3 (ruling 63).
**Next up, in order (idle slots; box < 15 CPU-min unless said):**
1. Stage-2 rule draft (`docs/research/prereg/stage2_a_vs_b.md`), text only:
write ruling 65 in — the mark and the gate read the LATE half of the run (a
brain that eats at 50k and starves at 300k does not win); B's gate lists
row 5 as a named miss (ruling 68). Still a draft; the Director freezes it.
2. Brain B row 6 by its frozen rule (Kaggle rank 4); harvest into
`KNOWN_ANSWERS.md`.
3. `pilot_b.trace`: SB3's `train/std` overwrites the trace's own `std` key
after the first window — store both, planted test; no old run re-read.
4. §20's cheap test: a stop-resume twin on B's row 1b (box-minutes) — the
resumed run's next 1,000 updates match the unbroken one, or the item named.

---

## SYSTEM — work on the machine (max 3 open)

*Order of work: **S11 (c) FIRST** (the disk is ~2 days from stopping every
lane) → **S14** → **S13** → S11 (a)–(b). A fix a RUNNING Jack kernel needs
goes first (ruling 54).*

### S11 — The root disk leaks at the source; two steering files over their caps
part: (machine — a dead machine outranks everything) · stage: — · step: —
**Do.** (c) FIRST: `/` is at 4.3 GB, falling 0.6 a day, workers refuse at 3.
Prune old CLI copies in `~/.local/share/claude/versions/`, keeping the
launcher's target and anything a running process has open — with a planted
test; run by hand, no new cron; then `~/.cache` entries that belong to THIS
project's tools only (pip/uv caches now live under `/data/caches/`).
Never touch another project's files, `/var/log`, or docker. State line:
free GB before/after. (a) `docs/FIELD_WATCH.md` back inside 12 KB + the one
prompt line ("rewritten inside 12 KB each sweep; NO-ACTION is one line per
seat"). (b) `scripts/research_prompt.md`: the 70 KB file cap is replaced by
"≤ 6 KB a section; workers read one section" (rulings 55, 74) — words
removed, none added.
serves: philosophy 5, 6 · standing objective 4.
**done when:** `/` has ≥ 6 GB free with the before/after in the state line; FIELD_WATCH ≤ 12 KB; the two prompt lines changed.
state: (ordered 11-19, never started — it sat behind the TPU door.)

### S14 — A usage refusal carries its evidence (46 slots were refused on a number that later fell)
part: (machine — truthful gates) · stage: — · step: —
**Do. The stop's number, the pace switch and `.paused` are NOT touched —
they are the owner's.** Reporting only: every STOPPED/DEFER line that
`ladder`, `overseer`, `review`, `regate` and `field_watch` write gains the
pct read, the meter line it came from, and the meter's printed reset time;
`scripts/usage_attribution.py` already detects a fall inside one reset
window (`meter_non_monotone`) — make it also print how many slots were
refused between the two contradicting readings. Planted test: a log with
96% then 16% in one window reports the refused count.
serves: philosophy 5 (the loop cheaper and truer) · auditor finding 3.
**done when:** the planted log reports its refused count, and a real STOPPED line (or a planted one through the real code path) shows pct + reset.
state: (new 11-21.)

### S13 — The judge lock's tripwire (forge F5) and the lane-guard test that went red
part: (machine — nothing faked; FORGE.md "Fences") · stage: — · step: —
**Do.** (b) FIRST now (J18 accepts no result without it): hashes of the
judge files (`experiments/harness/`, `registry.py`, the frozen rule files a
forge round names, the round's `spec.json`) checked by the forge round
runner before it accepts a result; a planted edited judge is refused. (a)
Reproduce `scripts/test_lane_guard.sh`'s 5–6 FAILs of 11-20 11:35 ("wrapped
lane is VISIBLE", seat cap rc) on a clean tree: fix by cause or name why the
test is wrong. Real separation (own user) is NOT ordered.
serves: philosophy 4.
**done when:** the planted judge edit is refused, and `test_lane_guard.sh` exits 0 or its wrong clauses are named.
state: (new 11-20, not started.)

**Next up:** (0) **S9** as written 11-20 (`git show 8fde84f5:docs/ORDERS.md`):
left `T0.17` (demote `T2.14` by drift), `T0.23`, then DP.05 HR.5 LT.02
T0.35 T3.09 T4.02 W.2 W1.00; `live_unauditable_pairs` 22 → 16 only in the
commit that next re-runs `T1.14`. (1) `gpu.py`: the Kaggle preamble's torch
pin moves to 2.6.0+cu126 once (A2b proved it) so each GPU job stops
re-pinning; `tests.test_gpu_accelerator` `tpu_own_weekly_line` FAIL fixed
or the TPU line removed (ruling 71); `dispatch.sh` pre-flight prints
`kaggle quota` beside our meter; `e2c-kernel`-style pushes count RUNNING
kernels before pushing. (2) **S6** step 3 — the video of the latest
recorded life. (3) `launch_detached.sh` refuses a second live watcher with
the same command line. (4) `prereg.judge()` door for unequal arms. (5) `run
status`'s STEERING parser reads this page's headings. (6) **S8** — a seat
held BY RECORD in `champions --check`. (7) `XL.00` re-buy with a planted
positive (stale 75 d, in front of 19 specs). (8) **S4**. (9) `T0.13`'s dead
clauses; `verify` VENUE 266 vs 256.

---

## INBOX — one line each, workers add, the Director empties
(emptied 11-21 06:45 — every line of 11-20/21 is ruled in 64–75 or became J17, J18, J14 (b)–(d), J10, S11, S14, Next-up.)
- 11-21 07:15 JACK · J17 freeze: the draft says primary "late" = last QUARTER of lives, ruling 65 (which names J17) says late HALF of the run. Frozen `98e22036` as ruling 65's late half (lives starting ≥ 150k), the last-quarter number printed beside and not read. No kernel is home before ~15:00 — say so before then if the other window was meant.
- 11-21 07:32 JACK · MACHINE/WORLD: the box venv runs mujoco 3.2.3, every Kaggle kernel pins 3.17.0, and the worlds are NOT the same world: same seed, same 2,000 random actions on K-row128 → 0 meals (3.2.3) vs 2 meals (3.17.0); first look identical. Any box row (hand forager 20/20, floors) set beside a Kaggle learner row crosses versions. RESEARCH/ruling: one version for both, or every box row re-read on 3.17.0 (`/data/venvs/jack-std` has it).
- 11-21 07:32 JACK · MACHINE: `fast_sim` row/multiray eyes needed each caller to remember `sim_speed_kcpu._compat_multiray()`; `pilot_b2` did not (fixed in the eye itself, `59bbdf90`). `eye_streak.py:173` and `find_himself.py:132` still call the 3.2.3 signature directly.
- 11-21 07:25 JACK · MACHINE: no venv on the box has `pytest`, and `tests/test_needs_grid_row.py` imports it (ran through a stand-in); `test_pilot_b2_kernels` / `test_e2c_kernel` print FAIL under the project venv (no `sbx`) and PASS under `/data/venvs/jack-std` — the header says so, the `run` tooling does not.
- 11-21 07:10 JACK · MACHINE: `launch_detached.sh`'s cap counts a GPU/CPU-kernel WATCHER as a busy job at launch, so with 2 box jobs computing no kernel can be pushed (the A2-v1S timing cell waited 25 min behind the two needs pilots while a Kaggle GPU seat sat empty); and the 6-poller cap left J17's r3, r4 unlaunched with 2 Kaggle CPU sessions idle (ruling 60: "no session sits idle").

## DONE (kept one day)

- **J15 — E2c** (`814aac92`, `WORLD_CARD.md` "E2c, the full book"): 25/25. Rule's clause PASS: E-food sustained 4/5 vs E-zero 0/5 → learnable by B when the direction is given; J11's miss is the eye. Auto α also eats (3/5; late > 0.82 on 4/5). Ruling 57's table: NO BRANCH → ruled (64). 10/10 food runs learn then lose it. Fixed α does NOT hold the policy wide (H → −10…−18); the critic overestimate is generic (controls too).
- **J3 — CLOSED** (ruling 59, Row J3-6): `@hold8` row 1 FAIL 0/5, row 3 PASS 3/5. Every in-house arm set aside with its numbers.
- **J16 (d), (e)** (`df2b7bc2`, `jack-cpu-1795207086`): rung 2 4/4; PyPI wheel PASS.
- **S12 — CLOSED** (ruling 71): door built `7689b52b`; two pushes got a CPU box, 0.00 TPU-h; refusals are named. D58 holds the owner's question.
- **Brain B row 5** (rule `0bf84989`): lib MISSES at k = 15 (833.8 vs 841.7–864.3; one seed 643.6), ours 862.6 → a named miss (ruling 68).
- **Next-up 11-20:** per-sense keys for A built + smoke; `world_card` per-food eat count (`7f20a8c3`); row128 and stage-2 drafts.
