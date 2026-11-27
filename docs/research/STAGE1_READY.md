# STAGE 1 READY-LIST — pointers only (ORDERS Next-up 2)

Written 2026-11-27 by JACK from `docs/PLAN.md` l. 59–66 ("Ready-list"). **No new claims:** each line
points at the commit, row or file that proves it, or says NOT YET and what is missing. A pointer here
is only as good as the row it names — read the row, not this page. Update a line when its row moves.

| # | PLAN's ready-list line | Status | Proof / what is missing |
|---|---|---|---|
| 1a | His world: the first rung liveable, the test card passed | **YES (rows)** | `docs/research/WORLD_CARD.md` § Rows (registered Kaggle CPU rows a–g, d8, d32, b2/f2/g2, harvested 11-15/16, sha256 per file) + § Answers (11-16); entry rung `d0` § "J2 item 5" (rule `8af5f2e1`) |
| 1b | Food that looks like food (ruling 111) | **YES on the kind paint, 500k** | KNOWN_ANSWERS row **J30** (`e406b65f`, 11-27): B+K1 on `bins16` kind eye EATS, late meals/life 27.41 vs blind 3.34 / shuffled 2.36, 3/3 seeds; WORLD_CARD § J30. Debts in the row: 500k only; per-geom paint untested (`prereg/eye_geom_500k.md` DRAFT) |
| 2 | The interface correct at death (the dying body's senses, not the reborn one's) | **NOT YET** | No planted box test found (grep `dying/reborn/final_observation` over `tests/` and `experiments/tests/`, 11-27). Owed: ORDERS Next-up 3 — one test on the wrapper J30 and the forge use |
| 3a | Brain B reproduced | **PARTLY** | KNOWN_ANSWERS B table: in band 1b (Pendulum), 3/3b (MountainCar), B-6b (reacher-easy), B-6c-2 (reacher-hard, re-registered); **named misses** B-5 (cartpole-swingup, lib), B-6a (swingup_sparse). Memory arm: row **B-7** FROZEN `c848d888`, twin FAILS as required (−1490.2), B7mem seeds RUNNING on Kaggle (J29) |
| 3b | Brain B piloted in his world on free CPUs ("does anything learn to eat") | **YES (one setting)** | Row **J30** above (B + K1 resets, Kaggle CPU). Plain SAC on the same eye: J24 NOT SHOWN AT 500k (collapse). Round 1-L (who still learns at 1M) RUNNING — J18, `docs/FORGE.md` F3 |
| 4a | Brain A reproduced at the early points of its published curves on the free GPU | **PARTLY** | KNOWN_ANSWERS A table: **A1** in band (782.3 / 984.1), **A3** in band (956.1); **A2** (1m) named miss at 100K (393.3 < 438), in band at 200K; A2-v1S ABOVE both bands (size S, not published). A2-v1S-enc seeds 10–12 armed for Sat 11-28 (J14 (a)) |
| 4b | Brain A smoke-tested in his world | **YES (smoke, not a row)** | KNOWN_ANSWERS l. 71 "A · smoke · his world, 20k, 1 seed · (1)–(4) hold — not a record row" |
| 5a | The harness's thirteen checks passed | **NOT SHOWN HERE** | Checks listed in `experiments/harness/__init__.py` l. 28 ("The thirteen self-checks … who pays each"); this page found no single row/commit certifying all thirteen PASS. Owed: a pointer to that certificate, or the check that it does not exist |
| 5b | The rule of the race written down, sized for the cloud | **DRAFT** | `docs/research/prereg/stage2_a_vs_b.md` (draft; PLAN l. 59: re-sized from the record before freeze — J27; memory control B-mem per ruling 136) |
| 6 | The eye researched and designed, so "add a sense" can be part of the decisive test | **RESEARCHED; design in race** | PUBLISHED_PRIORS §12 (his eye, a sense added later), §24, §26 (hindsight eye), §27, §28; kind eye `bins16` + K1 EATS (J30); hindsight eye ENTERED, raced next (ruling 134) |

**Side job, not on PLAN's list but its body line:** W-1 (body known-answer row) DRAFT
`prereg/w1_walkerwalk.md` (ruling 133), freezes after J28 cell 1.

Lines NOT YET / NOT SHOWN: **2** (death interface test), **5a** (thirteen-checks certificate).
