"""J24 (2), box proof 2: the hand forager on the FOOD BINS of the kind paint.

PLAN stage 1, parts "Eye" + "Learning to eat" (the learnability proof), step
2 -> 4. Prior: PUBLISHED_PRIORS §28 (every published forager's ray returns a
class; rllab Gather / Safety Gym pool it into closeness bins), §27; our J20
row: the hand forager f4 on the per-geom row128 eye, 19/20 at d0.75, its
shuffled twin g4 0/20, white random b4 mean 0.42 meals a life.

NOT A LEARNER, NOT A LEDGER SPEC. A fixed hand-written ruler, an adult of
his world, scored-and-ineligible, never a part of him. A pass says: floor
food is findable from `eye="bins16"` on `paint="kind"` by a plain rule that
reads NOTHING but those 80 numbers. It does not say a brain learns it.

=============================================================================
THE PROOF — frozen in this commit, before any of its 40 cells runs (11-26).
=============================================================================
Ruler   `BinsForager` below. Reads ONLY the 16 food bins (vision[0:16]; the
        other 64 vision numbers and every other sense are never read):
          * some food bin > 0: push straight at the centre of the bin with
            the highest closeness (ties: the lowest bin), azimuth 2*pi*j/16
            in his own body frame;
          * no food bin lit: wander — a push along a random one of the 16
            bin azimuths, held WANDER_HOLD = 25 decisions (f4's hold);
          * give up: when the highest food closeness has not risen by more
            than STALL_EPS = 0.005 (4 cm) for STALL_N = 50 decisions (10 s:
            stuck behind something, or standing on a food that has been
            eaten and is regrowing — it keeps its look), wander for one
            hold with the food bins ignored, then look again.
        Arms and lift stay at world_card.NEUTRAL, as f4/f5.
        What it is NOT told: which channel is food is fixed by hand (it is
        an adult); no texture, no geom id, no position, no yaw, no needs.
World   gate 2's (eye_forager.make_env row f4: difficulty 0.75, body eye,
        body motor, lazy audio, native substep, shipped calibration, default
        layout) with ONLY the eye changed: eye="bins16", paint="kind". The
        world itself is byte-identical under the paint
        (tests/test_jack_env_paint.py (b): every other sense and every ray
        distance equal on every look).
Seeds   261..280, 10 lives a seed, cap 9,000 decisions — gate 2's worlds and
        `eye_forager.run_seed`'s life loop, unchanged. Not fresh worlds (f4
        ran in them) and chosen for that: the same 20 worlds f4 was read in,
        and the same white-random floor. The ruler above was written without
        running it in any of them; the only smoke is one life on seed 153.
Rows    f6  the ruler
        g6  CONTROL that must fail: f6 with its 16 food bins permuted by a
            fresh permutation every decision (it still knows food is in
            sight and how near; only the bearing is destroyed)
        b4  the floor: white random in gate 2's world — NOT re-run. Its 200
            lives are J20's (`/data/jack-logs/j20/b4_d075_s*.jsonl`, box,
            mujoco 3.17.0): white random reads no observation and the paint
            changes no world state, so those lives are this world's lives.
Bar     UNMOVED (world_card.score): on a seed, mean floor meals a life over
        its 10 lives >= 5.0 AND > b4's mean on the same seed; f6 PASSES iff
        that holds on >= 18 of 20 seeds. g6 is scored by the same function
        and must hold on < 18; if g6 passes, f6's pass is VOID.
        (`floor_meals` counts obj0 + obj1 as run_seed always has; the dial's
        third food obj2 carries the food look and is eaten but not counted.)
Extension  NONE. A miss is a miss: one INBOX line, no learner seed (ORDERS
        J24 (2)).
Where   the box, detached, 2 workers, nice 19, jack-std (mujoco 3.17.0);
        resumable per cell as j20_reread. Size from J20: f4 1.54M + g4 1.07M
        decisions at ~12 ms = ~8.7 CPU-h; today's ledger had 54,800 s free.
Out     /data/jack-logs/j24/bins_forager/<row>_d075_s<seed>.jsonl

    python -m experiments.j24_bins_forager smoke          # seed 153, 1 life each
    python -m experiments.j24_bins_forager run            # box, detached
    python -m experiments.j24_bins_forager score
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

from experiments import eye_forager as EF
from experiments import world_card as WC

OUT = Path("/data/jack-logs/j24/bins_forager")
B4_DIR = Path("/data/jack-logs/j20")
SEEDS = EF.GATE2_SEEDS            # 261..280
D = 0.75
ROWS = ("f6", "g6")
WORKERS = 2
N_BINS = 16
WANDER_HOLD = 25
STALL_N = 50
STALL_EPS = 0.005
SMOKE_SEED = 153


class BinsForager:
    def __init__(self, seed, shuffle_bins: bool = False):
        self.rng = np.random.RandomState(seed * 6553 + 29)
        self.shuffle = (np.random.RandomState(seed * 6553 + 31)
                        if shuffle_bins else None)
        self.new_life()

    def new_life(self):
        self.wander_left = 0
        self.wander_ang = 0.0
        self.ignore_food = False
        self.best = 0.0
        self.stall = 0

    def _wander(self, a):
        if self.wander_left <= 0:
            self.wander_ang = 2 * math.pi * int(self.rng.randint(N_BINS)) / N_BINS
            self.wander_left = WANDER_HOLD
        self.wander_left -= 1
        return EF.a_set(a, self.wander_ang)

    def act(self, o):
        food = np.asarray(o, dtype=float)[:N_BINS]
        if self.shuffle is not None:
            food = food[self.shuffle.permutation(N_BINS)]
        a = WC.NEUTRAL.copy()
        if self.ignore_food:                       # giving up, one hold
            if self.wander_left > 0:
                return self._wander(a)
            self.ignore_food = False
            self.best, self.stall = 0.0, 0
        top = float(food.max())
        if top <= 0.0:
            self.best, self.stall = 0.0, 0
            return self._wander(a)
        if top > self.best + STALL_EPS:
            self.best, self.stall = top, 0
        else:
            self.stall += 1
            if self.stall >= STALL_N:
                self.ignore_food = True
                self.wander_left = 0
                return self._wander(a)
        self.wander_left = 0                       # a fresh draw next time
        return EF.a_set(a, 2 * math.pi * int(np.argmax(food)) / N_BINS)


def make_env(seed: int, d: float = D, eye: str = "bins16", row: str = "f6"):
    from experiments.jack_env import JackEnv
    return JackEnv(seed=seed, difficulty=d, eye="bins16", paint="kind",
                   eye_frame="body", motor_frame="body",
                   lazy_contact_audio=True, native_substep=True)


def run_cell(row: str, seed: int, lives: int = WC.LIVES) -> list:
    """eye_forager.run_seed's life loop, unchanged, in this file's world."""
    pol = BinsForager(seed, shuffle_bins=(row == "g6"))
    keep = EF.make_env
    EF.make_env = make_env
    try:
        return EF.run_seed(row, seed, D, lives=lives, policy=pol)
    finally:
        EF.make_env = keep


def _cell_file(row: str, seed: int, out: Path = OUT) -> Path:
    return out / f"{row}_d075_s{seed}.jsonl"


def _lives(f: Path) -> list:
    if not f.exists():
        return []
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()]


def todo() -> list:
    out = []
    for row in ROWS:
        for s in SEEDS:
            f = _cell_file(row, s)
            if len(_lives(f)) >= WC.LIVES:
                continue
            if f.exists():
                f.unlink()                 # partial: re-run from the seed
            out.append((row, s))
    return out


def run(workers: int = WORKERS) -> int:
    import subprocess
    import time
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    cells = todo()
    print(f"mujoco {mujoco.__version__}; {len(cells)} cells to run", flush=True)
    live, bad = [], 0
    while cells or live:
        while cells and len(live) < workers:
            row, s = cells.pop(0)
            f = _cell_file(row, s)
            err = open(str(f) + ".err", "w")
            live.append((row, s, f, err, subprocess.Popen(
                [sys.executable, "-m", "experiments.j24_bins_forager", "cell",
                 row, str(s), str(f)], cwd=str(EF.REPO),
                stdout=subprocess.DEVNULL, stderr=err)))
        time.sleep(2)
        for item in list(live):
            row, s, f, err, p = item
            if p.poll() is None:
                continue
            live.remove(item)
            err.close()
            bad += p.returncode != 0
            L = _lives(f)
            print(f"{row} s{s}: rc {p.returncode}, {len(L)} lives, mean floor "
                  f"meals {np.mean([x['floor_meals'] for x in L]) if L else float('nan'):.2f}",
                  flush=True)
    print(f"DONE bad={bad}", flush=True)
    return bad


def score() -> dict:
    rows = {r: [x for s in SEEDS for x in _lives(_cell_file(r, s))] for r in ROWS}
    b = [x for s in SEEDS for x in _lives(_cell_file("b4", s, B4_DIR))]
    full = len(SEEDS) * WC.LIVES
    out = {"lives": {**{r: len(v) for r, v in rows.items()}, "b4": len(b)}}
    for r in ROWS:
        if len(rows[r]) == full and len(b) == full:
            out[r] = WC.score(rows[r], b, SEEDS)
            out[r]["seed_means"] = [round(WC.seed_mean(rows[r], s), 2) for s in SEEDS]
            out[r]["mean"] = round(float(np.mean([x["floor_meals"] for x in rows[r]])), 3)
    out["b4_mean"] = round(float(np.mean([x["floor_meals"] for x in b])), 3) if b else None
    if "f6" in out and "g6" in out:
        out["verdict"] = ("PASS" if out["f6"]["passes"] and not out["g6"]["passes"]
                          else "VOID (control passes)" if out["g6"]["passes"]
                          else "MISS")
    else:
        out["verdict"] = "INCOMPLETE"
    return out


def main(argv):
    if argv[:1] == ["cell"]:
        row, seed, f = argv[1], int(argv[2]), Path(argv[3])
        with open(f, "w") as fh:
            for rec in run_cell(row, seed):
                fh.write(json.dumps(rec) + "\n")
        return 0
    if argv[:1] == ["smoke"]:
        import time
        for row in ROWS:
            t = time.process_time()
            (rec,) = run_cell(row, SMOKE_SEED, lives=1)
            print(row, {k: rec[k] for k in ("decisions", "floor_meals", "cause",
                                             "floor_eats_all", "lifetime_s")},
                  f"{time.process_time() - t:.0f} CPU-s", flush=True)
        return 0
    if argv[:1] == ["run"]:
        return run()
    if argv[:1] == ["score"]:
        print(json.dumps(score(), indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
