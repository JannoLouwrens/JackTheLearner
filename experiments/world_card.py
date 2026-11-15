"""world_card.py — the world test card: can anything live in today's world? (ORDERS J1)

PLAN stage 1, part "His world, needs and lessons", step 2 (combine: measure the
outlier before redesigning it). Prior: PUBLISHED_PRIORS.md §3 (our floor meal
is 8% of the bar, the smallest of any world reviewed; "the entry settings at
which a ray-sensing rover's random policy eats must be measured here") and §8
"Gates" (floor arms <= 1 success in total; a scripted forager must pass; else
the venue is void for judging learners).

NOT A LEARNER, NOT A LEDGER SPEC. Every row is a fixed ruler held against the
world. The foragers are rulers, never parts of him. Results go to
`docs/research/WORLD_CARD.md`.

=============================================================================
THE RULE — committed before any registered run (11-15, this commit).
=============================================================================
World      `jack_env.JackEnv`, default options (= today's world, proven by
           `tests/test_jack_env.py`), default calibration, e0 = 1.0.
Seeds      REGISTERED: 201..220 (20 worlds; each seed mutates its world as W0
           does). 153/154 are SMOKE seeds only — the hand forager was written
           and tuned on 153/154 and never looked at 201..220 before the run.
Lives      10 consecutive lives per seed, in one persisting world (W0's lives
           share a world: regrowth timers carry across a death). A life still
           going at LIFE_CAP_DECISIONS (9,000 = 30 sim-minutes, 3x the statue's
           600 s) is ended by the runner and recorded `censored`; the SAME
           world continues and the runner respawns him via `W0.respawn()`.
Per life   lifetime (sim s), floor meals (obj0+obj1 eat events), apples,
           drinks (today's W0 has no drink event: reported "n/a"), cause of
           death (energy / integrity / censored), summed env reward.
Rows       a  do-nothing: the zero action (arms mid, adhesion 0.5, no drive).
           b  random, white noise per decision: uniform(-1,1)^8, exactly
              survival.run_survival's "random" (rng seed*6553+11).
           c  pink noise per life: each of the 8 dims an independent 1/f
              (beta=1) Gaussian sequence over the life (FFT method, as the
              `colorednoise` package used by Eberhard et al. 2023), scaled to
              uniform's std (1/sqrt 3), clipped to [-1,1].
           d8, d32  uniform(-1,1)^8 held 8 / 32 decisions.
           e  the privileged NE.01 forager (next slot; hook only here).
           f  the HAND-WRITTEN forager below, fed ONLY the observation vector
              (74 floats). No positions, no geom ids, no world state.
           g  CONTROL that must fail: (f) with its 16 rays' (distance, texture)
              pairs permuted by a fresh random permutation every decision.
Bar (f)    unmoved from LF_04_FAIL.md: on a seed, the mean floor meals per
           life over its 10 lives >= 5.0 AND > row b's mean on the same seed.
           (f) PASSES iff that holds on >= 18 of the 20 seeds.
Control    (g) is scored by the same test and must hold on < 18 seeds. If (g)
           passes, (f)'s pass is VOID (its food came from something other than
           sight) and is reported as such.
Gate §8    floor arms (a, b) must total <= 1 seed meeting (f)'s bar; else the
           bar is too easy for this world and the card says so.
Answers    (1) intake vs drain from drives.py constants beside row e's lifetime;
           (2) summed reward per life against lifetime (does dying early pay?);
           (3) from f/g: can 16 rays tell food from a wall, and if not, which
           sense is missing. Written into WORLD_CARD.md, numbers from the rows.
Where      a full row is up to ~200 lives x <= 9,000 decisions: Kaggle CPU
           kernels. Only a 2-seed smoke runs on the box (`smoke`).

THE HAND FORAGER (row f), and what it is allowed to know. It reads the 74-wide
observation and nothing else. It knows the observation's LAYOUT (which index is
ray k's distance, the drive's frame) and two DESIGN CONSTANTS of the world, both
disclosed: rays are fixed in the world frame like the drive (so pushing along
ray k moves toward what ray k sees), and a floor food regrows in
RESPAWN_FLOORFOOD_S = 66.9 s. It does NOT know which texture is food: it finds
out by touching things and watching its own energy (needs[0]) jump. Rules:
  see      a "blob" = a run of <= 2 adjacent rays sharing one texture that both
           neighbours do not share (a small object). Walls span many rays.
  choose   the best blob by priority / distance: a texture that has paid and is
           off cooldown = 3, an untried texture = 1; banned or cooling = never.
  track    the target is held as a world-frame offset and dead-reckoned by the
           observed root velocity (proprio vx, vy) between sightings, so it
           survives flickering out of the ray fan.
  go       full drive along the offset; arms neutral; adhesion off.
  learn    energy up > 0.03 in one decision = a meal: if the target is within
           0.6 m, its texture is marked FOOD and cools for 66.9 s (a meal
           with no near target credits nothing). A FOOD texture missed 3
           times in a row is demoted and banned. A contact (offset < 0.15 m) with
           no meal within 10 decisions, or a pursuit longer than 150 decisions,
           is a MISS: an untried texture cools 66.9 s and is banned at its 2nd
           miss (it may have been unready food the first time); a FOOD
           texture cools 30 s (20 s on a timed-out pursuit). Knowledge persists across lives (same world).
  wander   with no target: head along the most open of 16 ray directions
           (random among the >= 70th-percentile distances), hold it <= 25
           decisions or until that ray reads < 0.8 m.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

REGISTERED_SEEDS = tuple(range(201, 221))
SMOKE_SEEDS = (153, 154)
LIVES = 10
LIFE_CAP_DECISIONS = 9000
BAR_MEALS = 5.0
BAR_SEEDS = 18
ROWS = ("a", "b", "c", "d8", "d32", "f", "g")      # e: next slot

# observation layout (jack_env SENSES order; widths from cores.MODALITIES)
N_RAYS = 16
RAY_RANGE = 8.0
I_VX, I_VY = 66, 67            # proprio root vx, vy (world frame)
I_E = 68                       # needs[0]
DT = 0.2
RESPAWN_FLOORFOOD_S = 66.9     # design constant, disclosed in the docstring
RAY_ANG = np.array([2 * math.pi * k / N_RAYS for k in range(N_RAYS)])
NEUTRAL = np.array([0, 0, 0, 0, -1, -1, 0, 0], dtype=float)


# ── rows a-d: the floor arms ────────────────────────────────────────────────
class DoNothing:
    def __init__(self, seed): pass
    def new_life(self): pass
    def act(self, o): return np.zeros(8)


class WhiteRandom:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed * 6553 + 11)
    def new_life(self): pass
    def act(self, o): return self.rng.uniform(-1.0, 1.0, 8)


class Held:
    def __init__(self, seed, k):
        self.rng = np.random.RandomState(seed * 6553 + 11 + k)
        self.k, self.left, self.a = k, 0, np.zeros(8)
    def new_life(self): self.left = 0
    def act(self, o):
        if self.left <= 0:
            self.a, self.left = self.rng.uniform(-1.0, 1.0, 8), self.k
        self.left -= 1
        return self.a


def pink(n: int, dims: int, rng: np.random.RandomState) -> np.ndarray:
    """1/f Gaussian noise, (n, dims), unit std per dim (colorednoise's method)."""
    f = np.fft.rfftfreq(n)
    s = np.zeros_like(f)
    s[1:] = f[1:] ** -0.5                     # amplitude ~ f^(-beta/2), beta=1
    s[0] = s[1]
    out = np.empty((n, dims))
    for d in range(dims):
        spec = s * (rng.randn(len(f)) + 1j * rng.randn(len(f)))
        x = np.fft.irfft(spec, n)
        out[:, d] = (x - x.mean()) / (x.std() + 1e-12)
    return out


class Pink:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed * 6553 + 17)
    def new_life(self):
        self.buf = np.clip(pink(LIFE_CAP_DECISIONS, 8, self.rng) / math.sqrt(3), -1, 1)
        self.t = 0
    def act(self, o):
        a = self.buf[min(self.t, len(self.buf) - 1)]
        self.t += 1
        return a


# ── row f: the hand-written forager, observation only ───────────────────────
def blobs(vis: np.ndarray):
    """[(ray_index_float, distance_m, texture)] for small objects in view."""
    d, t = vis[0::2], vis[1::2]
    hit = d < 1.0
    same = lambda a, b: abs(a - b) < 1e-4
    out, k = [], 0
    seen = np.zeros(N_RAYS, bool)
    for k in range(N_RAYS):
        if not hit[k] or seen[k]:
            continue
        run = [k]
        j = (k + 1) % N_RAYS
        while hit[j] and same(t[j], t[k]) and j != k and len(run) < 4:
            run.append(j); j = (j + 1) % N_RAYS
        i = (k - 1) % N_RAYS
        while hit[i] and same(t[i], t[k]) and i not in run and len(run) < 4:
            run.insert(0, i); i = (i - 1) % N_RAYS
        for r in run:
            seen[r] = True
        if len(run) > 2:
            continue
        left, right = (run[0] - 1) % N_RAYS, (run[-1] + 1) % N_RAYS
        if (hit[left] and same(t[left], t[k])) or (hit[right] and same(t[right], t[k])):
            continue
        ang = math.atan2(np.mean([math.sin(RAY_ANG[r]) for r in run]),
                         np.mean([math.cos(RAY_ANG[r]) for r in run]))
        out.append((ang, float(min(d[r] for r in run)) * RAY_RANGE, float(t[k])))
    return out


class HandForager:
    def __init__(self, seed, shuffle_rays: bool = False):
        self.rng = np.random.RandomState(seed * 6553 + 23)
        self.shuffle = shuffle_rays
        self.food = set()          # textures that paid
        self.miss = {}             # texture -> contacts without a meal
        self.banned = set()
        self.cool = {}             # texture -> time it may be tried again
        self.food_miss = {}        # FOOD texture -> consecutive misses
        self.clock = 0.0
        self.new_life()

    def new_life(self):
        self.target = None         # dict(tex, rel=np.array([dx,dy]), age)
        self.contact_wait = 0
        self.heading, self.hold = None, 0
        self.prev_e = None

    def _food_missed(self, k):
        self.food_miss[k] = self.food_miss.get(k, 0) + 1
        if self.food_miss[k] >= 3:          # it was never food: demote
            self.food.discard(k)
            self.banned.add(k)

    @staticmethod
    def _key(t):
        return round(float(t), 4)

    def _prio(self, tex):
        k = self._key(tex)
        if k in self.banned or self.cool.get(k, -1) > self.clock:
            return 0.0
        return 3.0 if k in self.food else 1.0

    def act(self, o):
        o = np.asarray(o, dtype=float)
        vis = o[:2 * N_RAYS].copy()
        if self.shuffle:
            p = self.rng.permutation(N_RAYS)
            vis = vis.reshape(N_RAYS, 2)[p].reshape(-1)
        v = o[[I_VX, I_VY]]
        e = o[I_E]
        self.clock += DT
        ate = self.prev_e is not None and e - self.prev_e > 0.03
        self.prev_e = e

        # learn from the last decision
        if self.target is not None:
            k = self._key(self.target["tex"])
            if ate and np.linalg.norm(self.target["rel"]) < 0.6:
                self.food.add(k)
                self.food_miss[k] = 0
                self.cool[k] = self.clock + RESPAWN_FLOORFOOD_S
                self.target = None
            elif ate:
                pass                # bumped food en route: credit nothing
            else:
                self.target["rel"] = self.target["rel"] - v * DT
                self.target["age"] += 1
                if np.linalg.norm(self.target["rel"]) < 0.15:
                    self.contact_wait += 1
                if self.contact_wait > 10:
                    if k in self.food:
                        self._food_missed(k)
                        self.cool[k] = self.clock + 30.0
                    else:
                        self.miss[k] = self.miss.get(k, 0) + 1
                        if self.miss[k] >= 2:
                            self.banned.add(k)
                        else:
                            self.cool[k] = self.clock + RESPAWN_FLOORFOOD_S
                    self.target = None
                elif self.target["age"] > 150:
                    if k in self.food:
                        self._food_missed(k)
                        self.cool[k] = self.clock + 20.0
                    else:           # never reached: counts as a miss
                        self.miss[k] = self.miss.get(k, 0) + 1
                        if self.miss[k] >= 2:
                            self.banned.add(k)
                        else:
                            self.cool[k] = self.clock + RESPAWN_FLOORFOOD_S
                    self.target = None
        elif ate:
            pass                    # a meal with no target: nothing to credit

        # see and choose
        seen = blobs(vis)
        if self.target is not None:
            # refresh the target from a sighting of its texture near where expected
            best = None
            for ang, dist, tex in seen:
                if self._key(tex) != self._key(self.target["tex"]):
                    continue
                rel = np.array([math.cos(ang), math.sin(ang)]) * dist
                err = np.linalg.norm(rel - self.target["rel"])
                if best is None or err < best[0]:
                    best = (err, rel)
            if best is not None and best[0] < 1.5:
                self.target["rel"] = best[1]
        if self.target is None:
            cand = [(self._prio(tex) / max(dist, 0.2), ang, dist, tex)
                    for ang, dist, tex in seen if self._prio(tex) > 0]
            if cand:
                _, ang, dist, tex = max(cand)
                self.target = {"tex": tex, "age": 0,
                               "rel": np.array([math.cos(ang), math.sin(ang)]) * dist}
                self.contact_wait = 0

        a = NEUTRAL.copy()
        if self.target is not None:
            r = self.target["rel"]
            n = np.linalg.norm(r)
            a[6:8] = r / n if n > 1e-6 else 0.0
            self.heading, self.hold = None, 0
            return a
        # wander
        d = vis[0::2] * RAY_RANGE
        if self.heading is None or self.hold <= 0 or d[self.heading] < 0.8:
            open_ = np.flatnonzero(d >= np.percentile(d, 70))
            self.heading = int(self.rng.choice(open_))
            self.hold = 25
        self.hold -= 1
        a[6:8] = [math.cos(RAY_ANG[self.heading]), math.sin(RAY_ANG[self.heading])]
        return a


def make_policy(row: str, seed: int):
    if row == "a":
        return DoNothing(seed)
    if row == "b":
        return WhiteRandom(seed)
    if row == "c":
        return Pink(seed)
    if row in ("d8", "d32"):
        return Held(seed, int(row[1:]))
    if row == "f":
        return HandForager(seed)
    if row == "g":
        return HandForager(seed, shuffle_rays=True)
    raise ValueError(f"row {row!r} not built here (row e is the next slot)")


# ── the runner ───────────────────────────────────────────────────────────────
def run_seed(row: str, seed: int, lives: int = LIVES,
             cap: int = LIFE_CAP_DECISIONS) -> list:
    from experiments.jack_env import JackEnv
    env = JackEnv(seed=seed)
    o, info = env.reset(seed=seed)
    pol = make_policy(row, seed)
    out = []
    for life in range(lives):
        pol.new_life()
        ate0 = dict(info["ate_total"])
        rew, n, t0 = 0.0, 0, env.w.sim_seconds
        cause = "censored"
        while n < cap:
            o, r, term, trunc, info = env.step(pol.act(o))
            rew += r
            n += 1
            if term:
                cause = info["death_cause"]
                break
        ate = {k: info["ate_total"][k] - ate0.get(k, 0) for k in info["ate_total"]}
        out.append({"row": row, "seed": seed, "life": life,
                    "lifetime_s": round(env.w.sim_seconds - t0, 1),
                    "floor_meals": int(ate.get("obj0", 0) + ate.get("obj1", 0)),
                    "apples": int(ate.get("apple", 0)), "drinks": "n/a",
                    "cause": cause, "reward": round(rew, 5), "decisions": n})
        if term:
            o, info = env.reset()
        else:
            env.w.respawn()                 # censored: same world, new body
            env._pending_obs = None
            o, info = env.reset()
    return out


def seed_mean(rows: list, seed: int) -> float:
    m = [r["floor_meals"] for r in rows if r["seed"] == seed]
    return float(np.mean(m)) if m else float("nan")


def score(f_rows: list, b_rows: list, seeds=REGISTERED_SEEDS) -> dict:
    """(f)'s bar, as registered. Same function scores the control (g)."""
    ok = [s for s in seeds
          if seed_mean(f_rows, s) >= BAR_MEALS and seed_mean(f_rows, s) > seed_mean(b_rows, s)]
    return {"seeds_meeting": len(ok), "of": len(seeds), "passes": len(ok) >= BAR_SEEDS}


# ── the off-box runner: Kaggle CPU, the way forager_race3 launched ──────────
# One kernel per row: 4 child processes, each one cell = this row x 5 seeds,
# run sequentially, written as JSONL. Worst case per cell 5 x 10 x 9000 =
# 450k decisions (~3.3 h at the box's 38.8/s); the children time out at 11 h.
OUT = Path("/data/jack-logs/world_card")
CELLS_PER_KERNEL = 4
CHILD_TIMEOUT_S = int(11.0 * 3600)
REMOTE_TIMEOUT_S = int(12.2 * 3600)


def kernel_cells(row: str, seeds=REGISTERED_SEEDS) -> list:
    if row not in ROWS:
        raise ValueError(f"kernel: row {row!r} not in {ROWS}")
    seeds = list(seeds)
    k = -(-len(seeds) // CELLS_PER_KERNEL)
    return [(row, seeds[i:i + k]) for i in range(0, len(seeds), k)]


def run_parallel(cells: list, outdir) -> dict:
    import subprocess
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    procs = []
    for row, seeds in cells:
        name = f"{row}_s{seeds[0]}-{seeds[-1]}"
        f = outdir / f"{name}.jsonl"
        procs.append((name, f, subprocess.Popen(
            [sys.executable, "-m", "experiments.world_card", "cell", row,
             ",".join(map(str, seeds)), str(f)], cwd=REPO,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)))
    res = {}
    for name, f, p in procs:
        try:
            _, err = p.communicate(timeout=CHILD_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            p.kill()
            _, err = p.communicate()
        lives = ([json.loads(x) for x in f.read_text().splitlines() if x.strip()]
                 if f.exists() else [])
        res[name] = {"lives": lives, "rc": p.returncode}
        if p.returncode != 0:
            res[name]["error"] = f"rc={p.returncode}: {err.decode()[-800:]}"
    return res


_JOB = r'''
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q", "mujoco", "gymnasium"], check=True)
import json, os
os.chdir("/tmp/jack")
from experiments.world_card import run_parallel
r = run_parallel(__CELLS__, os.environ["JACK_OUT"])
json.dump(r, open(os.path.join(os.environ["JACK_OUT"], "world_card_cells.json"), "w"))
print("DONE", flush=True)
'''


def kernel(row: str) -> dict:
    """Submit one row (20 registered seeds) as one Kaggle CPU kernel; write
    every returned life to OUT/<row>.jsonl (partial cells are kept: a cell
    that dies mid-way still returns the lives it finished)."""
    from experiments.gpu import build_job, result_json, submit
    cells = kernel_cells(row)
    body = _JOB.replace("__CELLS__", repr(cells))
    sub = submit(build_job(body), prefer="kaggle-cpu", est_hours=4.0,
                 timeout_s=REMOTE_TIMEOUT_S, fetch=["world_card_cells.json"])
    if not sub.ok:
        raise RuntimeError(f"world_card kcpu row {row}: {sub.message}")
    r = result_json(sub, "world_card_cells.json")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{row}.jsonl", "w") as fh:
        for name, rec in r.items():
            for life in rec["lives"]:
                fh.write(json.dumps(life) + "\n")
            print(f"{name}: {len(rec['lives'])} lives"
                  + (f" ERROR {rec['error']}" if "error" in rec else ""), flush=True)
    return r


def main(argv):
    if len(argv) >= 5 and argv[1] == "cell":      # cell <row> <s1,s2,..> <out.jsonl>
        for s in argv[3].split(","):
            res = run_seed(argv[2], int(s))
            with open(argv[4], "a") as fh:
                fh.write("\n".join(json.dumps(r) for r in res) + "\n")
        return 0
    if len(argv) >= 3 and argv[1] == "kernel":    # kernel <row>
        r = kernel(argv[2])
        print(json.dumps({k: ("error" in v) for k, v in r.items()}))
        return 0
    if len(argv) >= 2 and argv[1] == "smoke":
        lives = int(argv[2]) if len(argv) > 2 else 2
        rows = argv[3].split(",") if len(argv) > 3 else list(ROWS)
        for row in rows:
            for s in SMOKE_SEEDS:
                t = time.time()
                res = run_seed(row, s, lives=lives)
                print(json.dumps({"row": row, "seed": s, "wall_s": round(time.time() - t, 1),
                                  "lives": [(r["lifetime_s"], r["floor_meals"], r["apples"],
                                             r["cause"], r["reward"]) for r in res]}), flush=True)
        return 0
    if len(argv) >= 4 and argv[1] == "row":       # row <row> <seed> [out.jsonl]
        res = run_seed(argv[2], int(argv[3]))
        line = "\n".join(json.dumps(r) for r in res)
        if len(argv) > 4:
            with open(argv[4], "a") as fh:
                fh.write(line + "\n")
        print(line)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
