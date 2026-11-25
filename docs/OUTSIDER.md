# OUTSIDER — read cold, 2026-11-25

## 1. The one thing most likely wrong

**"Nothing learns to eat from his own eye" has been read as a fact about
brains. It is a fact about how the world paints food.**

`w0.py:225`: `_tex = ((arange(ngeom) * 0.37) % 0.8) + 0.1`. What a ray reports
about a thing is a hash of its position in the model file. 0.37 × 13 = 4.81 =
6 × 0.8 + 0.01, so any two geoms 13 apart differ by 0.01. Measured today in
world 941 (`j24_class_probe.world_table`):

| food | code | nearest other code |
|---|---|---|
| obj0 | 0.49 | poolwall0, 0.01 away |
| obj1 | 0.86 | poolwall1, 0.01 away |
| obj2 | 0.43 | poolwall2, 0.01 away |

So in his world three foods look like three unrelated things, each of them
looks almost exactly like a pool wall, and in the next world the codes move
(the probe's own result: 23 of 23 worlds). After `VecNormalize`
(`pilot_b.py:471`) 0.01 is about 1/25 of a standard deviation of that input.
Brain B's flat MLP has to carve three slots that narrow, separately for each
of 128 rays (no shared weights), from a reward that arrives once per ~1,200
random decisions.

Three things the team's pages say that this undoes:

- `FORAGER_RACE.md`: "the sight probe rules out blindness: random rays hit
  floor food on 19–24% of decisions". A hit is not a recognition. He was
  blind to *food* in LF.04, in the seven-brain race, in J17 and in J22.
- PLAN, "the last 10%": "Can 16 distance rays tell food from a wall?" No
  number of rays can, with this code. Never an open question.
- The learnability proof. The hand forager (`world_card.py:64–81`) does not
  see food either: it touches every small blob, keeps a table of which code
  paid, and bans the rest. That is a memory plus a shape rule, not an eye. It
  proves a table can eat, not that a memoryless network can.

Today's order stopped on "a ruling that gives the eye a class". That reads
it as handing him knowledge. It is not. In every world on record, and in
nature, things of one kind look alike: that is a property of the WORLD
(GOAL: "consistent, discoverable"), not a gift to the creature. He must still
learn that this look feeds him. And the finished Jack needs it: a brain
trained here and handed over cannot carry "food" to a machine where food
wears a different hash.

## 2. Is the first job the right job?

Eating from his own eye is the right first job. The way it is being waited on
is not. A strong group would, this week: (a) make appearance a property of
the kind of thing (one look for floor food, one for apple, one for wall,
water, ramp), coded as a few channels per ray, never one scalar; (b) run the
free probe below before any learner; (c) stop spending sessions on the ear's
timing, the walking row and the Mac port while nothing eats.

They would also not plan Jan–Feb around "A against B at 1M decisions on the
rover". The record PLAN quotes predicts both eat there, so the race is built
to tie. The result that separates brains for the owner's ends is already in
hand: brain B loses the skill in 10 of 10 runs. A brain that is handed to
people "still learning" is chosen on keeping, not on first meal.

One more number: the world alone runs 60.7 decisions per CPU-second
(`pilot_b2.py` docstring); under brain B it runs 11–18. The learner, not the
simulator, now eats about three quarters of every free hour.

## 3. Three experiments

**E1. The probe, no learner.** Log 200k decisions of the babble
policy (~2 h on one Kaggle CPU). For each, the
simulator, as judge, labels the bearing of the nearest food in range. Train
brain B's own 256-256 MLP by plain regression on four inputs: today's row;
the row with one code for all floor food; a class one-hot per ray; 16 bins
per class. Test on held-out worlds. Prediction: today's row explains under
10% of the bearing in new worlds and learns slowly even in its own; the
other three pass 80%. If today's row passes, I am wrong and the wall is
credit or exploration; if it fails, every eye result since 11-14 is void as
a verdict on a brain.

**E2. J24 on a world where food looks like food.** Three seeds, 500k, blind
and shuffled twins: 9 runs, 4.5M decisions, ~45 CPU-hours, 9 kernel slots.
Prediction: sighted pulls clear of both twins by 300k and eats at least
twice babble by 500k. If not, the eye is cleared and the learner is the
suspect — go to E3.

**E3. An adult shows him (Jump-Start RL).** The hand forager is an adult of
his world, not a part of him. It drives the first H decisions of each life,
brain B the rest; H shrinks to 0 by 150k. Three seeds and three unguided
twins at 300k: 1.8M decisions, ~18 CPU-hours. Prediction: he eats above
babble inside 100k, then loses it after the guide leaves. If he keeps it,
exploration was the wall and a teacher is the curriculum. If he loses it,
first meals were never the problem and the keep round is the main line.

## 4. What the field knows that this code ignores

- **A category is never fed as one number.** Entity embeddings / one-hot
  (Guo & Berkhahn 2016). Networks fit smooth functions first (Rahaman 2019,
  spectral bias); value networks suffer it badly (Yang 2022, "Overcoming the
  Spectral Bias of Neural Value Approximation").
- **Even the published lidar agents are handed more than a class.** Safety
  Gym's default is one lidar per kind; its own docs say the natural lidar
  "will be significantly more difficult". The published budgets (0.5–3M) are
  for the easy form.
- **Things of a kind look alike** in every world that taught an agent to
  forage from sight: Crafter, Animal-AI, ViZDoom health gathering,
  FoodCollector's tags.
- **A weak guide removes the exploration wall:** Jump-Start RL (Uchendu
  2023): the guide need only beat random; the horizon cost falls from
  exponential to polynomial.

## 5. What they are doing right

Blind and shuffled twins on every run, rules frozen before the run, a
one-run verdict withdrawn. `j24_class_probe.py` asked "is the
signal there?" before spending 4.5M decisions — ask that first, every time.
