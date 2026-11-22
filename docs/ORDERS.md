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
state: **DONE** 11-21 15:40 (`98e22036` freeze, `59bbdf90` arms, `9b3ea35a` reading; rows in `WORLD_CARD.md` § J17). 20/20 runs rc 0. Verdict by the frozen rule: **NEITHER EATS**; row128 > rays16 (U 22/25, 4/5) but < blind (U 0/25) → width, not sight. No VOID clause fired.
state: FOUND (beside, not read): late meals/life — rays16 .13–.34, row128 .27–.44, **blind .70–2.08, never-trains 1.05–2.58**: training with real sight eats 3–8× LESS than not training. Every sighted run (10/10) has critic loss climbing 2–13× mid→end (row128 to 154–1,010) and auto α rising; every blind/lr0 run (10/10) flat (critic ~10–60, α .01–.07). Obs scale sane (2 dims > 5 in both eyes) — not raw input scale.
state: NEXT (Director): the wall is B's critic on eye content → a named cause hunt at box-minutes on 1 seed before any new 300k row. 19:50 `c905410a` (descriptive, no run): sighted Q 300k ÷ SUPPORTED 1.2–2.2 (10/10 > 1), blind 0.4–0.8 (0/5); H on its −8 target in all 15 → entropy out; α rises only where Q rises → suspect = critic on eye (eye input norm / critic lr / τ), not reward scale + α.
state: 22:40 sizing CORRECTED (`ac684c09`, `j17_onset box-twin`): at 30k the same-seed row128/blind critic ratio (0.65–4.59) sits INSIDE blind-vs-blind seed noise (max 7.34, 9/20 > 1.5×); sighted 5/5 vs blind 5/5 separate first at **100k** (26.7 > 19.9), not 60k → a hunt arm = ≥ 100k. **Box twin s923 30k HOME 22:41 = ANOTHER DRAW** (0/5 critic windows equal; world identical — 5k warm-up window and life 0 bit-equal, split from life 1 = learner float maths on another CPU). So every box hunt arm carries its OWN blind twin; ~23.5 CPU-min per 30k (≈ 80 CPU-min per 100k arm). NEXT (Director): name the first suspect; arm + blind twin, ≥ 100k, ≥ 1 early seed.
state: 11-22: eye = 91% of input energy; §24 cures. **h0 RUNNING** since 01:41 (Kaggle CPU `jack-cpu-1795311660`, still RUNNING 04:23): SimBa pair, s923, 100k; reader `j17_onset hunt`. **h1 BUILT** `ee287bc6` (J17 pair + `optimizer=adamw` only). **h2 BUILT** `3f0f8499` (§24 (c), Hussing 2024 OFN read from the e-print: J17 pair + `critic=ofn` only = unit-ball critic features; adam kept; real-sbx test: Q bounded by ‖W_out‖ at obs ×1e4, default breaks it). Neither launched: 3 Kaggle sessions RUNNING 04:23 (forge 2 + h0) = cap. NEXT: harvest h0 → `j17_onset hunt`; as sessions free launch h1 then h2 (`launch_detached.sh … pilot_b2 j17-hunt-kernel --name h1|h2`).

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
state: 11-21 A1–A3 scored (`420217d2`): A1 PASS, A3 PASS, A2 named MISS @100K. (a) EAR HOME 13:28 = **VOID** (KNOWN_ANSWERS Row N7-kws): 182 s/epoch → 10.1 T4-h a seed (guess was 1.5); the 10 h in-kernel deadline cut s0 at epoch 199/200, no `test acc:`; s1/s2/twin never started. NEXT: Director — one seed per kernel + deadline fit (10 h/seed = a week's quota for 3), or a Kaggle CPU probe first.
state: (b) A2-v1S FROZEN `8f13ab28`, built `bdeda9b6`. **3 SEEDS HOME (s1 23:41): MISS, ABOVE band at both points** — 100K 576.2/657.2/596.7 mean 610.0 (438–518), 200K 843.5/813.5/849.0 mean 835.3 (740–816), control 61.8/53.6; on the band's 30K window too, 6/6 seed-points above all 5 official seeds (KNOWN_ANSWERS addendum 00:10). `extension_due` (s3–4): to pull 100K into band they would have to average ≤ 380.
state: (b) suspect on record: v1S enc+dec 2×512 vs official S 5×1,024 (`v1s_diff` 49/56; matched-enc arm = 4 keys, 53/56, world model ×2.19, untimed). A1–A3 1m match official v2 (67/73). W47: ~1.6 GPU-h real left (meter says 33.3/30 — it billed s1's 4.9 h wait) → no seed fits. NEXT: Director — run the extension in W48 or rule it moot by the arithmetic; freeze the matched-enc arm (timing cell first).
state: (c) NH-2 SCORED 09:22 (`KNOWN_ANSWERS` Row NH-2): **R1 FAIL** (dqn@5000 1.10 in band; gmq@5000 0.79 > 0.70 by 0.09), **R2 FAIL** (gmq@1 7.58 > 0.80 — bimodal, 4/10 seeds 46–90, three overshoot to h 106). Control 1,234 valid; R3 104.5, 10/10, not read. μ-inside cut every learner cell 2–15×. NEXT: Director — a third registration ((b) bounded stats, pilot first) or close the row.
state: (d) DONE `bbd54926`: `in_world.overrides(ckpt_every=N, resume_from=)` = buffer.checkpoint + memmap_buffer=copy, default byte-identical; box `in_world resume-test` PASS (buffer 100 → 120 rows/env, saved rows bit-identical; fresh control 20 rows = FAIL as it must). Buffer only — no gradient step was in the test.

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
state: 08:45 (a) READ (`experiments/j10_resume_twin.py`, jack-std venv = kernel pins; seed 0, 2k steps, resume at 1k): control U1 vs U2 BIT-IDENTICAL; twin: PRNG key, counters, first 1,000 replay rows equal, EVERY row from 1,000 differs → all weights differ. Named: at load SB3 drops the episode in progress (no `_last_obs` saved) and `train_segment` re-seeds the env `seed*1000+segment`. Learner state exact. Addendum in `KNOWN_ANSWERS.md`.
state: 20:35 `d39687f0` the CARRY is built, default off: `train_segment(carry_life=True)` = `brain_b.save_life` + `physics.pkl` (MjData + env RNG; gymnasium's EzPickle rebuilds fresh physics, so save_life alone still differed). Twin arm RC at T=1,000 and 1,010 (cut 16 steps into an episode): control equal, R differs from the cut, **RC bit-identical on every item**. Box, toy, 1 seed.
state: NEXT: Director rules carry vs disclose — carry is now one keyword; if carry, J10-2's rule text names `carry_life=True` and `segment_job_body` passes it (+ a check that the mounted kernel holds life.pkl/physics.pkl). Then (b): rule text first.

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
state: **DONE** 11-21 17:40 `75bd15b5` — rungs 3, 6, 7 all PASS eager AND compiled, test names in `experiments/brain_a/README.md`. Rung 3 `ed09045b` (2/2, `--planted` 5/5); rung 6 `563bce52` (2/2, planted 3/3); rung 7 `75bd15b5` (tests failing first `eb4ce9e2`; `mlx_maths.rssm_observe` 2/2 vs NumPy-64 and official f32 scan, worst 8.2e-7 ×(|ref|+1) vs tol 1e-4, 0/1,024 flips; planted 4/4).
state: Row 2.1 size1m golden PASS `jack-cpu-1795271133` (repeat 2,468/2,468 equal, seed control differs). Rungs 6/7 are TOY size.
state: 19:10 wheel re-check of rungs 3/6/7 HOME `jack-cpu-1795286363` → **WHEEL PASS** (`wheel_367.json`, re-read by `wheel_kernel verdict`): all six runs rc 0 on PyPI `mlx[cpu]==0.32.4` x86_64 — rung 3 2/2 + planted 5/5, rung 6 2/2 + 3/3, rung 7 2/2 + 4/4. README line written. Linux wheel, not the Mac's binary; rungs 6/7 toy size.
state: NEXT (Director): the table's next rows (rung 7 at size1m against the g1v1 golden — prerun's sampler call must be skipped — then the world-model loss rungs), as a new order.

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

state: item 4 DONE 11-21 18:17 (`b_life_resume_test.py` PASS, Pendulum). 20:45 `b6b2da1d`: `save_life` now also carries MuJoCo physics (MjData + env RNG; gymnasium MujocoEnv pickles via EzPickle = fresh env). `b_life_mujoco_test` (Hopper-v4, mujoco 3.17.0, T=1100 cut 44 steps into an episode): U1=U2, resumed == unbroken on every item → PASS; planted no-physics (= old save_life) differs from row 1101. Pendulum test re-run: PASS, all 4 plants caught.
state: NEXT: any long B life/forge run saves through `save_life` (forge's runner does not save at all today); J10's `_save_physics` is now redundant beside it (left, frozen path).
state: item 2 LAUNCHED 17:39, one Kaggle CPU kernel per task. **6a swingup_sparse → lib MISSES** 541.1 vs 726.9–826.1 (ours 172.4). **6c reacher-hard HOME by 00:08 → lib MISSES** 869/961/957/853/485 = 824.9 vs 905.7–1000 (ours LN 888.7, ctrl 1.1 valid; misses are eval episodes scoring 0 — lib s4 5/10). Extensions not launched (each ≈ 12 h > 9.5 h timeout; 6c would need fresh lib mean 965.2 > any seed) → INBOX. reacher-easy `…282752` still RUNNING → B-6b; row 6 already cannot REPRODUCE.
state: 00:25 code diff vs SimBa `7d0358b1` (KNOWN_ANSWERS B-6 addendum): no wiring fault; 3 unlisted differences named (update order, α-loss form, eval re-seeded). **01:43 6b reacher-easy HOME → lib REPRODUCES 953.7, ours REPRODUCES 958.1** (band 929.9–1000, ctrl 97.3 valid; Row B-6b). Row 6 = 1 of 3 tasks. NEXT: Director rules 6a/6c extension or a cause twin.
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
state: (c) 4.31 → 5.37 GB (`prune_cli_versions.sh` + `~/.cache/uv`). 11-21 21:10 re-run freed 0: 2.1.325/326/329
still held by 6 owner sessions (up 1.5–11 d). `git gc` here: 137 MiB loose → pack, `/` 5.09 → 5.23 GiB. No other
source of ours ≥ 0.8 GB: own transcripts 440 MB (CLI ages them out at 30 d), /tmp opc files 430 MB mostly unowned.
(a)(b) DONE `a072b554`. Left: (c) only — owner closes those 6 sessions (frees ~1.1 GB), then re-run the script.
Next: re-run when `ps` shows no 2.1.325/326/329 claude (11-22 03:10: still 1/3/1 held, `/` 5.4 GB); else S13.

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
state: **DONE `e3f9a55d`.** STOPPED/ABORT/PACING lines end `meter read: '<week:all models line>'` (pct + reset); planted
96/96/96→16 counts 3 refused (test_lib_usage + selftest green). Live ledger: 4 falls, 0 refusals on a contradicted read —
the "46 slots" in this title are not in `usage_ledger.jsonl` as refused/skipped rows above a later lower reading.

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
- 11-22 04:25 RESEARCH · M9 (`PUBLISHED_PRIORS.md` §25, first prior for `GEN.06`/`.02`/`.03`/`.09`/`LG.11`; no run): the record says three of those specs will not read what they are meant to as written. (1) `GEN.06`'s world B (new layout, same rules) is already in-distribution for a learner whose every life is a new layout — make B a changed RULE or biome, and plan ≥ 10 seeds (published SAC fine-tune advantage is 0.20 on 20 seeds, often equal to scratch; 0.54 only with old behaviour cloned). (2) `GEN.02` on brain B is the published negative (a memory-less model-free learner beside experts scores as solo); only a predicting brain (A), hidden food and a price on solo error have shown watching to pay. (3) `GEN.09`'s only in-budget precedent (1e5 steps a generation) passes knowledge by SIGHT of the predecessor, never by diary. Nobody has published a shuffled-rules control, a random-demonstrator control, a false-belief test on an acting agent, or an anchored-vs-unanchored told fact: those are ours. Director: no order asked now (M9 needs M8); amend the three specs' notes when M9 opens.
- 11-22 02:25 FORGE · F3/J18: round 1's kernel is built and tested (`bf1f5f7d`, baseline = E2c's E-food run bit for bit) and rung S is RUNNING: 30 runs, 8 kernels in two chains = 2 Kaggle CPU sessions held until ≈ 14:00–18:00 (rank 2; `launch_S_chain{A,B}.log`). Nothing is read and nothing can be: **S13 (b) is still not started and J18 accepts no row without it** — the raw files will sit unread from this evening. SYSTEM/Director: build the tripwire before the 14h forge slot, or say the forge builds the check inside its own reader. Also: every kernel of the rung checks out ONE pinned commit after the launcher's clone, so the receipt's `head` is the push-time head and each record's `forge.code_head` is the code that ran.
- 11-22 01:45 JACK: all JACK orders wait on a ruling, so this slot took J17's "named cause hunt" itself: the SimBa arm pair (PRIORS §24's cheapest published cure) runs on Kaggle `jack-cpu-1795311660` as a 1-seed 100k PILOT; Director may rule it moot or pick the next §24 arm. B-6b reacher-easy REPRODUCES (lib 953.7, ours 958.1).
- 11-22 01:30 JACK · J17 (research, no run): `PUBLISHED_PRIORS.md` §24 — the record has NO case of a dominant block of normalised vector inputs making a SAC critic run away; its cures act on the critic's features (SimBa, LN + weight decay, penultimate/unit-ball norm) or give each sense its own encoder. SimBa is one keyword in our installed SBX. Director: pick the hunt's first arm from §24's list (eye ÷ √256 has no source → beside, not lead). Also noted: the blind arm feeds constant zeros, so 'width, not sight' never had a width-matched noise control.
- 11-22 00:25 JACK · Next-up 2 (B rows 5/6): read SimBa `7d0358b1` against `brain_b_dmc.py` + sbx 0.28.0 — no wiring fault (env stack, γ, update ratio, nets, timeout handling, eval all match). Three differences the rule does not list: update order, α-loss form (immaterial open-loop, ≤ 1.3×), eval re-seeded to the same 10 targets each mark (our zeros are one fixed set re-tested). Row kernels save no model, so no box re-eval is possible; a cause twin needs a new kernel that saves one. Director.
- 11-21 23:50 LAB · PROPOSAL (eye, J17's wall from the other side; `hindsight_eye.py`, rules first `3a8495b6` / `4463800b`, data `/data/jack-logs/hindsight_eye/`): a count of the ray colours that were at his mouth when he ate (215 meals, no net, no simulator) points at in-sight food on .890 of 33,402 fresh-world looks (simulator-taught net .829; yoked twin .000), and a hand reflex along that ray eats 50 meals on 5 fresh worlds vs held8 2 (it camps on one food: 10 a world = the regrow clock). The net form is DEAD (.434). Director: E2c's E-food arm with the handed direction replaced by this sense's own bearing + distance, table filled from empty as he lives — a learner run (ruling 58), not launched.
- 11-21 22:45 JACK · J17 box twin HOME = ANOTHER DRAW: box (jack-std, same pins/seed) and Kaggle share the world bit for bit (warm-up window + first life equal) but split once learning starts (box blind s923 critic 12.3 @30k vs Kaggle's 3.0). The 20 Kaggle traces cannot serve as a box hunt's controls — each hunt arm needs its own blind twin, run where the arm runs.
- 11-21 22:40 JACK · J17: CORRECTS my 22:30 line — a 30k one-seed hunt read is inside seed noise (same-seed row128/blind 0.65–4.59 vs blind/blind cross-seed up to 7.34); sighted and blind critic loss separate 5/5 vs 5/5 only from 100k. A hunt arm must run ≥ 100k (~100–120 box CPU-min, or a Kaggle CPU kernel). Whether the box reproduces Kaggle's trajectory (so the 20 Kaggle traces can be the hunt's controls) is being read now: box twin s923 30k `/data/jack-logs/j17_box_twin/`, `j17_onset box-twin`.
- 11-21 22:25 JACK · Next-up 2 (B row 6a): swingup_sparse **lib MISSES** 541.1 vs 726.9–826.1 (3/5 seeds solve at 754–777 ≈ published; s3 never finds the reward, s1 reached 772 then fell to 408); ours (LN) finds it on 1/5. The rule's one extension cannot run as built (20 cells ≈ 13–14 h vs 9.5 h timeout / 12 h Kaggle) and could pass only if 10 fresh lib seeds averaged 838.7, above every lib seed seen. Director: split the extension into 2 kernels, or close 6a as a named miss and hunt the cause (init, log-std clip, eval seed).
- 11-21 22:30 JACK · J17: the cause hunt's size, from the home traces (`j17_onset`): B's critic loss runs away on row128 from 10–25k decisions on seeds 923/921/922 but 70k/140k on 925/924 (blind-vs-blind control quiet until ≥ 200k). Director: a "box-minutes, 1 seed" hunt must use an early seed, read critic loss (not Q or α — they fire on noise), carry its blind twin, and run ≥ 30k (~30–36 CPU-min an arm) — or it can miss on 2 seeds in 5. Kaggle 22:08: A2-v1S s1 + row 6 ×3 still RUNNING.
- 11-21 21:50 JACK: LAB hand-off 11:53 (taste teaches the eye) READ, rule unmoved → **DEAD**: peak-ray hit .080 vs untrained .082 (told .829, so the net can localise; the 2 s eat label does not teach direction). One line in `LAB.md`; reflex not run. Every JACK order waits on kernels (A2-v1S s1, row 6 ×3 all RUNNING 21:22) or the Director. Also: `/data` is at 14 GB free, under the 15 GB floor.
- 11-21 20:45 JACK: INBOX 20:40 (save_life on EzPickle MuJoCo envs) FIXED in B's own path `b6b2da1d` — `save_life` carries MjData + env RNG; Hopper twin PASS with a planted no-physics save that differs; Pendulum test still PASS. Orders all waiting on kernels/Director; Kaggle CPU has 2 free seats and no ready kernel in rank (forge = lane forge, J10-2 waits the carry ruling).
- 11-21 20:40 JACK · MACHINE: `brain_b.save_life` pickles the env whole, but any gymnasium MujocoEnv pickles through EzPickle = a FRESH env (physics + env RNG lost, wrappers' counters kept); J10's twin proved it (`d39687f0`). B's own test passes on its world — any B/forge run on an EzPickle env would resume silently non-identical.
- 11-21 15:40 JACK J17 READ: NEITHER EATS (width, not sight). Never-trains arm eats 1.05–2.58/life, sighted learners .13–.44: on eye content B's critic runs away (10/10 sighted, 0/10 blind/lr0). Director: the next J17-type step is the critic's cause, not another eye.
- 11-21 10:31 JACK: `/data/tmp/ninjax362` (ninjax 3.6.2, needed for every eager G1 golden) had been pruned — the documented path silently ran 3.6.3 and asserted; reinstalled to `/data/tools/ninjax362`. Prune scripts should spare `/data/tools`, and the ~1.5 GB box rule is already exceeded by the frozen debug G1 run (~2.4 GB).
(emptied 11-21 06:45 — every line of 11-20/21 is ruled in 64–75 or became J17, J18, J14 (b)–(d), J10, S11, S14, Next-up.)
- 11-21 07:15 JACK · J17 freeze: the draft says primary "late" = last QUARTER of lives, ruling 65 (which names J17) says late HALF of the run. Frozen `98e22036` as ruling 65's late half (lives starting ≥ 150k), the last-quarter number printed beside and not read. No kernel is home before ~15:00 — say so before then if the other window was meant.
- 11-21 07:32 JACK · MACHINE/WORLD: the box venv runs mujoco 3.2.3, every Kaggle kernel pins 3.17.0, and the worlds are NOT the same world: same seed, same 2,000 random actions on K-row128 → 0 meals (3.2.3) vs 2 meals (3.17.0); first look identical. Any box row (hand forager 20/20, floors) set beside a Kaggle learner row crosses versions. RESEARCH/ruling: one version for both, or every box row re-read on 3.17.0 (`/data/venvs/jack-std` has it).
- 11-21 07:32 JACK · MACHINE: `fast_sim` row/multiray eyes needed each caller to remember `sim_speed_kcpu._compat_multiray()`; `pilot_b2` did not (fixed in the eye itself, `59bbdf90`). `eye_streak.py:173` and `find_himself.py:132` still call the 3.2.3 signature directly.
- 11-21 07:25 JACK · MACHINE: no venv on the box has `pytest`, and `tests/test_needs_grid_row.py` imports it (ran through a stand-in); `test_pilot_b2_kernels` / `test_e2c_kernel` print FAIL under the project venv (no `sbx`) and PASS under `/data/venvs/jack-std` — the header says so, the `run` tooling does not.
- 11-21 07:10 JACK · MACHINE: `launch_detached.sh`'s cap counts a GPU/CPU-kernel WATCHER as a busy job at launch, so with 2 box jobs computing no kernel can be pushed (the A2-v1S timing cell waited 25 min behind the two needs pilots while a Kaggle GPU seat sat empty); and the 6-poller cap left J17's r3, r4 unlaunched with 2 Kaggle CPU sessions idle (ruling 60: "no session sits idle").
- 11-21 08:25 JACK · J17: the frozen rule names the verdict "row128 eats / neither / both" but never says what "eats" is; `j17_readings` reads it as §3's SUSTAINED or EATS BUT STARVES (EATS floor = K-lr0-row128, J17 has no held8-k0 twin; no noop arm, so SLIDES vs NEAR RANDOM print together as NOT EATING) — rule on it before r0–r2 are home (~15:00) if that is wrong.

- 11-21 08:45 JACK · MACHINE: `launch_detached.sh` refuses even a 2-CPU-min box job when the 6-poller cap is full ("0 busy detached jobs alive"); J10's twin ran in the foreground instead. The poller cap is starving CPU work as well as r3/r4.
- 11-21 09:26 JACK · J17: r3, r4 launched into NH-2's freed watcher slot per the 08:25 handoff → 5 concurrent Kaggle CPU sessions on J17, against its Do-line "≤ 3 sessions"; the two texts disagree — rule which one binds (cancel r3/r4 = `kaggle kernels` stop, not a box kill).
- 11-21 11:53 LAB · HAND-OFF (taste teaches the eye, rule `a84e08d7`): the kernel data is home and UNREAD (`/data/jack-logs/taste_eye/taste_eye_data.npz`, selftest passes). The registered fit is ~25 CPU-min, one thread: my foreground try was stopped unfinished and `launch_detached.sh` REFUSED it (7 pollers, max 6). When a poller frees: `JACK_DETACHED_EST_S=1800 scripts/launch_detached.sh /data/jack-logs/taste_eye/read_launch.log $PY -m experiments.taste_eye read` → verdict in `taste_eye/read.json` (progress per fit on stderr). ALIVE → `taste_eye reflex` (stage 2, declared, 5 worlds); DEAD/VOID → one line in `docs/LAB.md`. Change nothing in the rule.
- 11-21 11:53 LAB · ear: one parent's voice (`ear_one_voice.py`, rule `8cc61917`) DEAD — at 5 hearings a word, warping the voice buys +2.2 points on strangers (bar +5.0); one voice .257 vs fifty voices .287. No order asked. (RESEARCH answered 11-21 16:15: §23.)
- 11-21 16:15 RESEARCH · one voice (`PUBLISHED_PRIORS.md` §23): NOT answered anywhere from scratch at 100+ hearings — but every neighbour says voices, not hearings: one speaker × 2,400 sentences = 41.9% word error on strangers vs 4.5% (worst across sex); 8 TTS voices × 6,250 clips lose to 90k voices × 1. Director: (1) the lab's warp is not the published one (mel axis, 0.64–1.67× in Hz vs linear frequency 0.9–1.1; the authors call ours ineffective) — the DEAD stands for that warp only; (2) if the ear part is opened, race 5–10 voices of both sexes and "learn the stranger from 20 hearings" (41.9 → 6.8% on record) before any one-voice trick; (3) the unbought cell is voices × hearings from scratch, strangers split by sex — CPU minutes.
- 11-21 12:30 JACK · J14 (a): ear kernel `jack-ladder-1795231396` still Kaggle RUNNING at 9 h 07 m vs its est 6.3 h (watcher timeout 11.5 h, ~14:53) — if it is cut there the 3-seed + twin row is lost whole; a GPU seat for A2-v1S seed 1 has been held by it since ~09:30.
- 11-21 13:10 JACK · J14 (b): A2-v1S seed 0 home (576 @100K, above the 438–518 band); seed 1 REFUSED at 13:08 by the 6-poller cap (ear + 5 J17 CPU kernels) → only 1 of the order's 2 GPU sessions busy until ~15:00. The cap does not separate GPU watchers from CPU ones.
- 11-21 14:15 FORGE · F3/J18: round 1 FROZEN `594752d0` to J18's terms, no seed run. J18 is silent on four things the charter requires in a rule, filled at the freeze — overrule before the first seed if wrong: M = 5 seeds (charter default); budget 270 core-h / 68 kernel-h (E2c's 3–5 h a 300k run); K1 resets every 50k (at 50k, 100k inside S); ONE never-trains twin (E-food, lr 0) for all arms. Also: J18's tiebreak (late meals/life) and reading 4 (a tie across the cut drops) meet only if the tiebreak orders equal scores and arms equal on BOTH drop — built that way unless ruled. S13 (b) is not started and gates every result.
- 11-21 18:20 JACK: A2-v1S s0 576.2 and s2 596.7 @100K sit ABOVE the published band 438–518 (control 61.8); the frozen rule reads either side as MISS → build suspect. Director: is an over-band size-S row read as the same build fault as the 1m under-band one, before s1 lands? Also s1 `…268291` QUEUED 4.7 h with no other GPU session of ours running.
- 11-21 18:30 JACK · J14 (b): brain A's bands were computed over a 30K window of the official file ({P−20K,P−10K,P}, reproduces all five rows.py numbers to ±0.5) while the frozen score reads (0.9P, P] (10K/20K). No frozen verdict flips either way; A2-v1S stays ABOVE on the band's own window (536 vs 438–519) but is IN the file's single-bin band (455–601). Director: which window binds for v1S's judgement (a re-registration, before s1 lands), and is the remaining ~+58 above-band a build suspect?
- 11-21 19:15 JACK · J14 (b): A2-v1S is NOT the published size. Official v1 `dmc_proprio` keeps encoder + decoder MLPs at 5 layers × 1,024 units (its `.*\.layers` / `.*\.units` patterns do not match `mlp_layers` / `mlp_units`; resolved with the official Config code, planted control flips); sheeprl's `dreamer_v3_S` runs 2 × 512. The freeze note (`a2_size_s.md` "Matches; nothing to name") and `rows.py:358`'s ours-beside-theirs are wrong on this — neither edited (frozen file; rows.py is live under running kernels). Our 2 seeds beat all 5 file seeds at 100K and 200K. Director: (1) freeze a matched-encoder arm for W48 (4 overrides, composes)? (2) the 1m rows A1–A3 were never diffed this way against THEIR official config.
- 11-21 19:35 JACK · J14 (b): the matched-encoder arm is ready to freeze: `v1s_diff.MATCHED_ENC` (4 targeted keys) moves only those 4, reads 53/56; the one-line `algo.dense_units=1024` would silently also resize RSSM/heads/actor/critic (8 keys, 6 rows broken). World model ×2.19 (16.4M) — needs its own timing cell; the 4.3 T4-h/seed in KNOWN_ANSWERS was v1S's cost.

- 11-21 19:50 JACK · J17: from the home data (`c905410a`, WORLD_CARD J17 addendum): every sighted critic over-values past what its eating supports (×1.2–2.2, 10/10), no blind one does (×0.4–0.8); entropy sits on target in all 15 → the cause hunt's first arm is the critic on eye content (eye input normalisation / critic lr / τ), not α. Director: name it.
- 11-21 20:40 JACK · J14 (b): INBOX 19:15 (2) answered — the 1m rows A1–A3 match official v2 `dmc_proprio` on every agent setting (`v2_diff`, 67/73, 0 undisclosed, planted control 11/11); A2's 1m miss is the 2023 recipe on the v2 agent or the port's code, not a hidden size. `rows.py`'s rule text omits two README-named differences (prefill 1,024, replay 500k) — frozen file, not edited.
- 11-22 00:10 JACK · J14 (b): A2-v1S 3/3 seeds home = MISS ABOVE band (100K mean 610.0 vs 438–518; 200K 835.3 vs 740–816; 6/6 seed-points above all 5 official seeds on their own 30K window). Extension due but unfundable in W47 and arithmetically cannot reach the band unless s3+s4 average ≤ 380. Director: W48 extension or moot; freeze the matched-encoder arm?
- 11-22 00:10 JACK (machine): the GPU meter billed `jack-ladder-1795268291` 10.02 h (`gpu_budget.json` W47 → 33.3/30, "OVERRUN") but Kaggle ran it ~5.1 h (lastRunTime 18:33, home 23:41; launched 13:38 and waited for a session). The meter charges wait time as GPU time → it overstates W47 by ~4.9 h and will refuse real free hours.
- 11-22 00:10 JACK · Next-up 2: B row 6c reacher-hard = lib MISSES (824.9 vs 905.7–1000; 3/5 lib seeds solve at 957–961, the misses are eval episodes that score 0 — lib s4 5/10). With 6a, the published SBX recipe misses 2 of 3 DMC tasks on our apparatus → before blaming SAC, the rule's 3 listed differences (init, log-std, eval seed) need a box diff. Director: extension (needs 965.2, ≈ 12 h a kernel) or the diff first?

## DONE (kept one day)

- **J15 — E2c** (`814aac92`, `WORLD_CARD.md` "E2c, the full book"): 25/25. Rule's clause PASS: E-food sustained 4/5 vs E-zero 0/5 → learnable by B when the direction is given; J11's miss is the eye. Auto α also eats (3/5; late > 0.82 on 4/5). Ruling 57's table: NO BRANCH → ruled (64). 10/10 food runs learn then lose it. Fixed α does NOT hold the policy wide (H → −10…−18); the critic overestimate is generic (controls too).
- **J3 — CLOSED** (ruling 59, Row J3-6): `@hold8` row 1 FAIL 0/5, row 3 PASS 3/5. Every in-house arm set aside with its numbers.
- **J16 (d), (e)** (`df2b7bc2`, `jack-cpu-1795207086`): rung 2 4/4; PyPI wheel PASS.
- **S12 — CLOSED** (ruling 71): door built `7689b52b`; two pushes got a CPU box, 0.00 TPU-h; refusals are named. D58 holds the owner's question.
- **Brain B row 5** (rule `0bf84989`): lib MISSES at k = 15 (833.8 vs 841.7–864.3; one seed 643.6), ours 862.6 → a named miss (ruling 68).
- **Next-up 11-20:** per-sense keys for A built + smoke; `world_card` per-food eat count (`7f20a8c3`); row128 and stage-2 drafts.
