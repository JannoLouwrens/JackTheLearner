# Pre-registration DRAFT — brain A's pilot on his world's 128-ray eye (ORDERS J14 (d))

**Status: DRAFT, text only, binds nothing.** Written 2026-11-24 19:10 UTC by a JACK slot (every
Kaggle session, the Colab cell and the box were busy). No seed of any arm below exists; no code was
written or run for it. Ruling 62 sets the terms: *the rule is drafted after A1–A3 are in band and the
smoke passes, and frozen at a sitting before any seed.* Every `OPEN:` line is a choice left to that
sitting, with this draft's default beside it.
Sources: PUBLISHED_PRIORS §2 (building brain A), §12 (his eye), §24 (one big block of inputs);
`WORLD_CARD.md` § J17 + hunt; KNOWN_ANSWERS "A's smoke in his world"; `prereg/a2_size_s_enc.md`;
rulings 62, 65, 67, 86, 92, 96, 99. Nothing new was read from the field for this draft.

## Gate — nothing below runs until ALL hold

1. **A is reproduced at the published size.** Row A2-v1S-enc (`a2_size_s_enc.md`, frozen 11-24) reads
   **in band**. Today: A1 PASS, A3 PASS, A2 a named MISS at 100K with the 1m network, A2-v1S a named
   MISS ABOVE; A2-v1S-enc has no seed yet (its timing cell is on Colab now). A miss there is a build
   fault to find first — "no A pilot claims the build before it is found" (that file's READING).
2. **The smoke passed** — it did: `in_world.py` DONE (1)–(4), 20k decisions, Kaggle T4, 11-21
   (18.7 decisions/s, world-model loss 26.19 → 5.27). On the **16-ray** eye; see build debt 1.
3. **A timing cell for THIS shape on the venue it is sent to** (rulings 86, 99): the 298-number
   observation is a learner shape nobody has timed. No score read.
4. **The Director freezes this file** before seed 923 runs in any arm.

## The question

In his world, on the body-frame 128-ray eye, does brain A (a Dreamer-style world model) **use sight
to eat** — sighted late meals/life above BOTH a blind twin and a shuffled-ray twin — where brain B
(SAC) did not?

## What the record and our own rows already say (so this asks only the rest)

- **Published.** Nobody has published Dreamer (or anything) on a 1-D ray row. Dreamer's eye is 64×64
  RGB at the 200M-class size; **no `size1m` result from any image-like sense exists** (§12). From
  pixels Dreamer needs about twice the steps it needs from state, and small targets lose most
  (reacher_hard 339 vs 949 at 500k). A needs-based precedent (EVAAA, §2): Dreamer survived longest;
  some agents learned to end the episode when food could not be found. Dreamer fails at directed
  exploration (§2). Entropy-only Dreamer finding food 3.8 m away in a 3,000-decision life was
  PREDICTED at "about half the seeds" (§2 last-10% item 4) — a prediction, not a result.
- **Ours, the world is learnable by sight.** Hand forager on this eye, body frame, mujoco 3.17.0:
  19/20 at d 0.75 (7.86 meals/life), shuffled-ray control 0/20 (3.06), random 0.42 (J20).
- **Ours, brain B does not learn it.** J17: the flat 256 ray numbers → NEITHER EATS, the critic runs
  away. Four published critic cures, one seed each: none holds. J22 (conv encoder + random shift,
  s923, 100k): late meals/life sighted **0.56**, blind **0.31**, shuffled **0.87** → NO CURE; B is
  closed on the wide eye for stage 1 (ruling 92) and "the eye waits for brain A's pilot".
- **Ours, A in his world.** Only the smoke: it runs, it takes gradient steps, no NaN. Nothing about
  meals was read (the rule forbade it at 20k).

So the record is silent and B's answer is no. This is PLAN's last 10%, and at this size it is a
**pilot**: one seed, three cells, no capability claimed whatever it reads.

## Arms — J22's three cells, with A as the learner

| cell | what the learner is fed | role |
|---|---|---|
| `A-row128` | the world's observation, `eye="row128"` (298 numbers: vision 256 = 128 rays × [distance, material]; audio 16, touch 8, proprio 12, needs 6 — measured 11-24, jack-std) | candidate |
| `A-row128-blind` | the same, every vision number zero (`blind_eye.BlindEye`) | twin — must not beat sighted |
| `A-row128-shuffled` | the same, whole rays permuted afresh each step (`blind_eye.ShuffledEye`, own RNG) | twin — must not beat sighted |

- Both twins are the wrappers J17 / J22 used, unchanged, applied env-side. **Consequence for a world
  model, said now:** the decoder's reconstruction TARGET is the blinded / shuffled observation too
  (the env feeds both). That is the twin's meaning here — the same learner in a world whose eye
  carries no arrangement — not "A with its encoder masked".
- Disclosed, as J22 did: the shuffled twin keeps "food is visible somewhere, this near". In J22 it ate
  MOST. A sighted cell that does not beat it has not shown that A reads where things are.
- **Seed 923, 100,000 decisions a cell** (J22's size, so the three A cells sit beside the three B
  cells). `OPEN:` 100k vs 300k. Default 100k: it is what J22 read, and it is what one free GPU
  session can hold (below). Cost of the default, said now: §12 says an image-like sense needs ~2×
  the steps; a NO at 100k is weaker evidence than a YES (see READING).
- No never-trains twin in the pilot (J22 had none); it belongs to the frozen row that follows a YES.

## World and learner — every setting, ours beside theirs

World: `pilot_b2.PILOT_RULE` exactly as J17/J22 (mujoco 3.17.0, ruling 76; `eye_frame="body"`,
`motor_frame="body"`, `food_floor_only=True`, `lazy_contact_audio=True`, `native_substep=True`;
arm K's reward `dr100-eat`, random needs, d 0.75, life cap as `pilot_b`), with `eye="row128"`.

| setting | published (official DreamerV3 / §2 "for him") | this pilot | why it differs |
|---|---|---|---|
| implementation | official JAX | sheeprl 0.8.5 `dreamer_v3_5`, runner unedited, torch pinned 2.6.0 | the reproduced build (A1, A3) |
| agent | v2: REINFORCE actor (learns from death) | same (`dreamer_v3_5`) | — |
| size | S and up for published scores; 200M-class for images | `OPEN:` `dreamer_v3_5_1M` (the smoke's; A1/A3 passed with it) vs size S with the 5 × 1,024 encoder (the size A2-v1S-enc tests). Default: **whichever size is in band on A2 when this is frozen** — a size that missed its known-answer row does not carry the pilot | his size is small (ruling 67: 12m is not his size) |
| precision | float16 | 32-true (ruling 67) | as every A row here |
| environments | §2 for him: **one** | `OPEN:` 1 vs 4. Default **1** (§2; and B's cells ran one). The smoke and the rows ran 4 — the timing cell must time the form that is frozen | — |
| action repeat | 2 on DMC; 1 for him | 1 (a decision is already 40 physics sub-steps) | §2 |
| train ratio | 512 (sheeprl `replay_ratio` 0.5), "as high as the hardware sustains" | 0.5, pinned | the rows' value; raising it is a raced arm, not a default |
| prefill | §2 for him: 5,000–10,000 decisions | `OPEN:` 1,024 (the rows' and the smoke's) vs 5,000. Default 5,000 (§2) | — |
| eye | 64×64 image through a CNN | 256 ray numbers in the vector, through the MLP encoder | the record has no 1-D-row recipe; a Conv1d encoder for A is a changed runner = a later raced arm |
| senses as keys | one decoder head per input | `OPEN:` flat (one key, the rows' form) vs `keys="per_sense"` (five heads, runner unedited, smoke (1)–(3) hold). Default **flat** | either way vision is 256/298 = **86 %** of the reconstruction loss and needs are 2 % (§18, §24: the loss is summed by width). Printed beside every reading; width-normalising is a changed runner |
| episode end | `terminated` only on true death | death = terminated; the life cap = truncated | §2's silent-bug list — checked by a planted test (debt 3) |
| replay | 5e6 | the run's length (100k) — cannot bind | — |

## Venue and cost — arithmetic, to be replaced by the timing cell

Smoke: 18.7 decisions/s on a Kaggle T4 (4 envs, 74-number observation, 1m, compile included).
At that rate 100k decisions ≈ **1.5 T4-h a cell, ≈ 4.5 T4-h for the three**. The 298-number
observation, one env, a larger size and a 5,000 prefill all move it; none is measured.
- Timing cell: 3,000 decisions past the prefill on the frozen form, decisions/s and gradient
  steps/s printed, no score read. Colab T4 (ruling 96) or Kaggle from Sat 11-28.
- A cell ≤ 3 h → one cell per Colab session, same GPU type for all three (said in the row).
  Longer → the cell saves and resumes (below). > 9 h a cell anywhere → one INBOX line, no seed.
- **Resume path: `bbd54926`** — `in_world.overrides(ckpt_every=N)` (sheeprl's own
  `buffer.checkpoint=True`, `checkpoint.memmap_buffer=copy`) and `resume_from=`. Its box test proved
  the REPLAY travels bit-identical (control failed as it must). **It took no gradient step** (both
  runs sat under `learning_starts`) — see build debt 4 before a scored cell leans on it.

## Reading — written before any seed

Primary, ruling 65: **late meals/life = meals per finished life over lives STARTING at decision
≥ 50,000** (`pilot_b2.j17_late_meals(rec, 50_000)`, J22's own call), per cell. With more than one
env the clock is the run's summed decisions, as sheeprl counts them.

| what the three cells read | reading | branch |
|---|---|---|
| sighted ≥ BOTH twins, and sighted > 0.87 (the best any B cell read on this seed and size) | **A MAY SEE** — a pilot, one seed | CONFIRM on seeds 921, 922 (three cells each). Holds on ≥ 2 of 3 → a frozen row is written at a sitting: J17-2's terms (`git show b4b709ce:docs/ORDERS.md`) with A as the learner, a never-trains twin and the hand forager as ceiling. No stage-2 claim from the pilot. |
| sighted ≥ both twins but ≤ 0.87 | ORDERED, NOT EATING — the eye's arrangement helps A a little and he still eats less than B's shuffled cell | no confirm at 100k. `OPEN:` one 300k trio (the record's "2× the steps" is the prior) or stop. Default: one 300k trio, then stop. |
| sighted < either twin | **NO at 100k** — as B | one row, one INBOX line. Not "A cannot": §12 says this size may be too short. `OPEN:` as above; default the same one 300k trio, and a NO there closes A on the flat wide eye for stage 1 as ruling 92 closed B. |
| a cell cut, errored, non-finite loss, or fewer than 5 finished late lives | VOID — no reading, no branch | re-run from the pinned commit |

Printed beside, NOT read: first and last capped life ("reached then lost"); median life length;
each cell's world-model loss by term at 25k / 50k / 100k (does reconstruction fall on the sighted
cell and not on the shuffled one — a world model that cannot predict a shuffled eye is expected, and
is not evidence of eating); episode-ending behaviour (EVAAA's "learned to end the episode": share of
late lives shorter than the do-nothing floor's median).

What this pilot cannot say, whatever it reads: that A eats (one seed; no never-trains twin); that A
beats B (different learners at different update ratios, no matched-update control — stage 2's
question, `stage2_a_vs_b.md`); anything about A at a size other than the one frozen.

## Build debt — code this draft needs, none written yet (tests first, each with its control)

1. `in_world.make` takes `eye=` and `twin=` (none / blind / shuffled); default = today's 16-ray
   world, byte-identical overrides (as `ckpt_every` was checked against HEAD). Test: the row128
   observation is 298 wide and its first 256 numbers are the eye; blind zeroes exactly those;
   shuffled keeps each step's multiset of rays; the default arm carries none of it.
2. Life rows A's runner does not write: wrap the env so each life lands in `pilot_b`'s life-row
   form with a partial file (ruling 99: a cut run keeps its trace), readable by `j17_late_meals`
   unchanged. Test: a planted 3-life record reads the same through B's reader; a `late_from` beyond
   the run raises (ORDERS Next-up 2).
3. Planted test that the life cap reaches sheeprl as `truncated` and death as `terminated`
   (§2: ports' silent bug).
4. A resume test that CROSSES `learning_starts`: resumed-at-N against unbroken, world-model loss
   and replay compared; if sheeprl's resume is not exact after gradient steps, say by how much and
   run each cell in one session instead.
5. The timing cell on the frozen form (gate 3).

## Debt carried

- One seed. B's own rows on this eye disagree across machines; all three cells go on one GPU type.
- The 0.87 bar is B's best cell on s923 at 100k — a number from one run, used only to keep
  "beats its twins at the floor" from reading as eating.
- A flat MLP reading 256 ray numbers is the form that failed for B. For A it is the unedited
  reproduced runner; the record's image recipe for Dreamer (a CNN encoder + image decoder) has no
  1-D form in sheeprl, so it would be a changed runner and enters only as a raced arm after this.
