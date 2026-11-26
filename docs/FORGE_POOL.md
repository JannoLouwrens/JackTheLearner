# FORGE POOL — candidates for brain B at the entry rung (order F2)

Seeded 2026-11-21 02:40 UTC by lane `forge`. **Not ranked yet: the Director ranks it** and
freezes round 1 from it (at most 8 non-control candidates a round). Checked by
`python -m experiments.forge.pool` (format, a `prior:` on every line, ≤ 20, a control).

**The baseline every line changes ONE thing in** (`pilot_b.make_b`): SBX SAC 0.28, 2×256,
LayerNorm on the critic only, lr 3e-4, batch 256, γ 0.995, 1 update a decision (32 every 32),
5,000 warm-up decisions, FIXED entropy coefficient 0.1, reward `dr100-eat` (+100 a meal),
observations normalised, replay = the whole run.

**Two walls, two bases** (both at the entry rung, body eye, body motor):
- `reach` — base **arm K** (his own eye). Pilot 2: 0 sustained lives in 25 learner runs at
  300k; learners end at 0.08–0.16 meals a life, below the never-trains twin (0.82).
- `keep` — base **E-food** (arm K + the direction to food handed in). E2c: sustained on 4/5
  seeds, then lost in 10/10 food runs (last capped life 81–189k of 300k; meals a life,
  first 20 lives → last 20: 9.9–26.6 → 0.4–3.0). The never-trains twin and E-zero do not eat.
  A keep candidate is judged on LATE lives; one that only reaches is not a winner.

Cost yardstick (E2c receipts): one 300k run ≈ 3–5 h on one Kaggle CPU core, 4 runs a
kernel. Rung S (50k × 3) ≈ 2 kernel-h a candidate; on `keep` rung S ends BEFORE the loss
begins (81k+), so it can only show that a candidate still reaches — the Director may want
rung S at 150k for a keep round.

## Candidates

- **K1** · source: published · wall: keep · change: full network resets every 200,000 decisions, replay kept (`param_resets=[200000, …]`; SBX rebuilds actor and critics, keeps the temperature and the buffer — read in `off_policy_algorithm.py`) · prior: §21 recipe (b), §19 — Nikishin 2022 (2205.07802) DMC IQM 501 → 656 with resets every 2e5; Nauman 2024 "the most robust intervention" · ready: knob · note: the last reset must leave ≥ 100k before the reading (first capped life came at 18–47k); at 300k that is one reset at 200k
- **K2** · source: published · wall: keep · change: TQC in place of SAC (`sbx.TQC`, 2 top quantiles dropped per net, everything else the baseline) · prior: §21 recipe (c) — Kuznetsov 2020 (2005.04269) truncation "seems to reduce the seed variance on Humanoid"; §16 Zoo TQC Humanoid 7,239 vs SAC 6,232 · ready: knob · note: SBX 0.28 ships TQC (was "not verified" in §21)
- **K3** · source: published · wall: keep · change: learned temperature, target −8 (`ent_coef="auto"`) · prior: §19 — SAC v2 (1812.05905), every Zoo entry but two, ML-Agents FoodCollector · ready: knob · note: ALREADY RUN as E-food-auto: sustained 3/5 (fixed 4/5), late meals > 0.82 on 4/5 (fixed 2/5), lost in 5/5 — no difference shown at 5 seeds; in the pool so a round can carry it as the second baseline, not as a new bet
- **K4** · source: published · wall: keep · change: entropy taken OUT of the critic's target, kept in the actor's loss (SACLite) · prior: §19 — Yu 2022 (2201.12434): the entropy term in the target is a pay-for-being-alive; SACLite cures it where a smaller α "will not eradicate" it · ready: code · note: needs a subclass of SBX's update and §19's known-answer check first (Yu's 5-node chain: fixed α 0.2 fails, SACLite reaches the end; seconds on the box)
- **K5** · source: published · wall: keep · change: 1 update per 10 decisions instead of 1 per decision (`gradient_steps` 3 every 32) · prior: §19 table — the one published SAC forager (ML-Agents FoodCollector) updates 0.1 a step, ours 1: "10× the fitting on less data"; §21 Nauman: the damage grows with the replay ratio · ready: knob · note: also ~10× cheaper per run, so it can run on `reach` at the same price
- **K6** · source: published · wall: keep · change: reward as the learner sees it ÷ 100 (a meal = +1) WITH the learned temperature of K3 · prior: §19 table (published foragers pay ±1 a food; ours +100) and SAC v1 Fig 3b (reward size is SAC's decisive setting); discovery evidence §1 (Auto MC-Reward locks ±1 sparse, ±0.1 dense) · ready: code · note: a learner-side wrapper, the world's reward option is not touched; with a FIXED 0.1 this would only make the entropy term 100× larger, hence the pairing
- **K7** · source: published · wall: keep · change: SimBa network (`SimbaSACPolicy`: residual blocks + LayerNorm in actor and critic) · prior: §21 recipe (d) — SimBa (2410.09754) into SAC matches the best on DMC, MyoSuite, HumanoidBench · ready: knob · note: §5 says big-critic methods need a GPU — time one update on a Kaggle CPU before it enters a round
- **K8** · source: published · wall: keep · change: fixed entropy coefficient 0.005 instead of 0.1 · prior: §19 cure (c) — the only other fixed entry in the Zoo (BipedalWalkerHardcore 0.005); charter's round-1 grid (temperature) · ready: knob · note: WEAK — E2c contradicted §19's cause 1 (fixed 0.1 did not hold the policy wide: H fell to −10…−18 in 8/8); kept as the grid's temperature cell only
- **K9** · source: published · wall: keep · change: K1's full resets every 200,000 decisions with the critic's LayerNorm OFF (`policy_kwargs=dict(layer_norm=False)`), run beside K1 as written · prior: scout N2 11-26, ruling 129 — arXiv 2610.42100 (EWRL 2026; read by the scout, not by the lab): SAC, 3 DMC tasks, 1M steps, 5 seeds, replay ratios 1/4/8: "Resetting alone outperforms resets combined with LN at all three observation noise levels"; AGAINST it §21 recipe (BRO, SimBa keep LN in, on far more tasks) · ready: knob · note: for ROUND 2 only (round 1-L's terms are frozen); figures only, no score reached us, "best replay ratio per method" is a per-arm pick; their noise is added Gaussian, his is eye content; entered by the lab 11-26
- **C4** · source: control · wall: keep · change: K9's own control — the baseline with the critic's LayerNorm off and NO resets (`layer_norm=False`), so K9 cannot win on the missing layer norm alone · prior: ruling 129 (the control the scout names); 2610.42100's plain-SAC arm · ready: knob · note: must not hold what K9 holds; if it does, the finding is about layer norm, not resets
- **R1** · source: published · wall: reach · change: run length 2,000,000 decisions instead of 300,000 · prior: §19 cause 4 — the one SAC forager took 2M steps with an easier body; flat learners ≈ 0 at 25M on an 8-dim body (Duan 2016) · ready: code · note: ≈ 20–33 h a run, over one Kaggle session — needs J10's checkpoint-and-resume segments; and E2c says a learner that reaches at 300k has lost it by 189k, so longer alone is expected to read the same unless a keep candidate wins first
- **R2** · source: published · wall: reach · change: learned temperature (`ent_coef="auto"`) on arm K · prior: §19 "what the record does not answer" item 2 — fixed 0.1 vs auto on arm K, 5 seeds, 300k (E2c ran it on E-food only) · ready: knob · note: §19 predicted "meals ≥ twin NOT predicted"
- **R3** · source: published · wall: reach · change: the last 4 observations stacked · prior: §5 — stacking 4 observations equals a GRU on position-only tasks (Ni 2022); "whether the rover needs memory" is §5's listed last-10% race · ready: code · note: an observation wrapper, world untouched
- **R4** · source: recombine · wall: reach · change: the winner of a keep round, run on arm K · prior: FORGE.md "The machine" 4 (each round starts from the last round's winner); E2c's reading "J11's miss is the eye" · ready: knob · note: EMPTY until a keep round has a survivor — it is here so the seat is not forgotten
- **R5** · source: published · wall: reach · change: pink (1/f, temporally correlated) action noise, drawn as one sequence per episode, in place of SAC's white Gaussian sample (fixed α 0.1 kept) · prior: §27 (3) C — Eberhard 2023 (T.H.1), SB3 SAC, 20 seeds, 1e6 steps, cartpole-swingup_sparse: white **377**, OU 608, **pink 730**; MountainCarContinuous 0 / 90 / 94 — B-6a's own task, aimed at Fault 6a-s3 (frozen explorer, below) · ready: code · note: needs an SBX port (the package targets SB3) and §27's known-answer check first (SB3 SAC + pink, MountainCarContinuous 0 → 94, box-minutes); §27's prediction on B-6a 5 × 150k: zero seeds 1 → 0 of 5, silent on win-then-lose; differs from the LAB 11-12 DEAD pink/OU arm (a home-made run on his world, 4/3/6 of 40), which priced no published recipe — entered at the next round, added 11-25 by ORDERS Next-up 2
- **C1** · source: control · wall: keep · change: plain variation, draw 1 — lr log-uniform in [1.5e-4, 6e-4], batch in {128, 256, 512}, tau log-uniform in [0.0025, 0.01], warm-up in {2,500, 5,000, 10,000}; drawn by `random.Random("forge-control-1")`, the drawn values written in the round's spec before its first seed · prior: FORGE.md "The machine" 5; discovery evidence (Gideoni 2026: plain sampling at the same budget matched or beat the search on 4 of 9 and 6 of 9 problems) · ready: knob
- **C2** · source: control · wall: keep · change: plain variation, draw 2 — the same box, `random.Random("forge-control-2")` · prior: as C1 · ready: knob
- **C3** · source: control · wall: reach · change: plain variation, draw 3 — the same box on arm K, `random.Random("forge-control-3")` · prior: as C1 · ready: knob

## Sources, counted

published 13 · recombine 1 (empty until a survivor exists) · inventor 0 · machine 0 (stage 2) ·
control 4. `inventor` is 0 on purpose: `LAB.md`'s one ALIVE arm (run-and-tumble steer, 14/40
with the want given) is a hand controller, not a setting of brain B.

## Not in the pool, and why (so nobody re-invents them)

- **Already lost here:** pink / OU exploration in his world (LAB 11-12 DEAD: 4/3/6 of 40; pilot item 6 — the published recipe on B-6a's sparse task is R5);
  γ 0.99 (LAB DEAD, 1 of 3) and γ 0.999 (pilot 2 arm G: SLIDE); never-negative alive reward
  (arms P, KP: SLIDE); a hand-forager teacher in replay (E2b: the teacher missed its own gate
  5/20); LayerNorm on the critic (already ON in the baseline and lost 10/10).
- **Below a pool member on the record:** ReDo / dormant-neuron recycling (Nauman 2024: below
  full resets); a kept-best checkpoint (§21: not a learning fix, breaks "never stops learning").
- **No prior in the files yet — a RESEARCH question, not a candidate (law 3a):** 3-step
  returns, a temperature cap or a different entropy target (the charter names the knobs, the
  record read so far gives no value), weight decay / spectral norm on the critic, DroQ / CrossQ
  on a CPU, a narrow 2×64 net at a high update ratio.

## Faults brain B shows on a published task (J23 / ruling 94; from Row B-6a, `KNOWN_ANSWERS.md` J19 addenda)

Not candidates (no change named yet); each is a wall a candidate must be scored against. Lib arm, DMC
cartpole-swingup_sparse, 150k, 5 seeds → 541.1 vs band 726.9–826.1; both faults are needed to explain the miss.
- **Fault 6a-s3, frozen explorer (the `reach` wall's shape on a sparse task)** · prior: §21 (signature 3: it found no
  published source tying α's movement to a failure — here α FALLS, the opposite sign), §19 (SAC v2 1812.05905 auto-α) · seen: no
  reward ever found; α 5.0e-2 @10k → 3e-5 @40k → 0 from 60k, critic loss 0, Q ≈ 0, 100/100 fixed episodes 0.0 at
  every mark — the α loss drives α to zero before any reward, and the actor's gradient vanishes with it. Not a runaway critic (Q climb 0.01 < 1.32).
- **Fault 6a-s1, win-then-lose (the `keep` wall's shape on a published task)** · prior: §21 (Nikishin 2205.07802
  resets; Nauman 2403.00514; Kuznetsov 2005.04269 TQC — the K1/K2 bets) · seen: fixed-100 772.2 @125k (in band) →
  407.7 @150k while Q still climbed 80 → 105 and critic loss sat at its run high (.48) — the J10 1M-s1 and E2c shape,
  so a keep candidate can be checked on a 150k DMC row before a 300k run in his world.

## What the forge would run first (a suggestion; the Director ranks)

A **keep** round on E-food, because that is where the baseline already eats and the measured
failure is: K1 resets, K2 TQC, K5 fewer updates, K6 reward size + learned temperature, with
K3 as second baseline and C1, C2 as the control source. K4 and K7 need a known-answer check
and a timing first. The `reach` wall (R1–R3) waits for a keep survivor (R4), since a learner
that cannot keep eating will read 0 at any length.
